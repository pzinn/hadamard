if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# based on makemore.py
import os
import sys
import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from params import config, work_dir, learning_rate, training_batch_size, weight_decay, device, writer

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

    



@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):
    def __init__(self, words, block_size):
        self.words = words
        self.block_size = block_size
    def __len__(self):
        return len(self.words)
    def contains(self, word):
        return word in self.words
    def __getitem__(self, idx):
        ix = torch.tensor(self.words[idx], dtype=torch.long)
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.zeros(self.block_size, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)
    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------

model = Transformer(config)
model.to(device)
model.need_reload = True

out_path = os.path.join(work_dir, "model.pt")

def load_model():
    if model.need_reload:
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(out_path, weights_only=True))
        model.need_reload=False

def save_model():
    print(f"saving model to workdir")
    torch.save(model.state_dict(), out_path)
    model.need_reload=True

def get_loss(dataset,step,name):
    loss = evaluate(model, dataset, batch_size=100, max_batches=10)
    writer.add_scalar("Loss/"+name, loss, step)
    writer.flush()
    if name == 'test':
        print(f"{step=} {name} {loss=:.6f}",end='\t'); sys.stdout.flush()
    return loss

def train(train_data,test_data,**kwargs):
    resume = kwargs.get("resume",False)
    num_workers = kwargs.get("num_workers",4)
    max_steps = kwargs.get("max_steps",-1)
    seed = kwargs.get("seed",3407)
    # optimization -> slowly being moved to params.py
    #batch_size = kwargs.get("batch_size",32)
    #learning_rate = kwargs.get("learning_rate",5e-4)
    #weight_decay = kwargs.get("weight_decay",0.01)
    batch_size = training_batch_size
    eval_freq = kwargs.get("eval_freq",500)

    # system inits
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #os.makedirs(work_dir, exist_ok=True)

    block_size = config.block_size
    vocab_size = config.vocab_size # should one check that this is correct?

    #print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if resume:
        try:
            load_model()
        except FileNotFoundError:
            pass
    print(f"number of examples in the dataset: {len(train_data)+len(test_data)}")
    print(f"max word length+1: {block_size}")
    print(f"number of unique characters in the vocabulary: {vocab_size}")
        
    # wrap in dataset objects
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    # training loop
    step = 0
    get_loss(train_dataset,step,"train")
    best_loss=get_loss(test_dataset,step,"test")
    while True:
        #t0 = time.time()
        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        X, Y = batch
        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        #t1 = time.time()

        # evaluate the model
        step += 1
        if step % eval_freq == 0 or step == max_steps:
            get_loss(train_dataset,step,"train")
            test_loss=get_loss(test_dataset,step,"test")
            # save the model to disk if it has improved
            if test_loss < best_loss:
                save_model()
                best_loss = test_loss
                if step == max_steps:
                    max_steps += eval_freq # don't quit on a winning streak
            elif test_loss > best_loss+.2 or step == max_steps: # termination conditions: done, or we've probably massively overfitted
                break


def crop(row):
    return tuple(row[:next((i for i, x in enumerate(row) if x == 0), len(row))])
        
def sample(**kwargs):
    num_samples = kwargs.get("num_samples",1000)
    seed = kwargs.get("seed",3407)
    top_k = kwargs.get("top_k",-1) # -1 means no top-k

    # system inits
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #os.makedirs(work_dir, exist_ok=True)

    block_size = config.block_size

    load_model()

    X_init = torch.zeros(num_samples, 1, dtype=torch.long).to(device)
    top_k = top_k if top_k != -1 else None
    X_samp = generate(model, X_init, block_size-1, top_k=top_k, do_sample=True).to('cpu')
    samples = [ crop(row[1:].tolist()) for row in X_samp ]
    return samples

