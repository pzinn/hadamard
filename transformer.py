if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# based on makemore.py
import os
import sys
import math
import random

import torch
import torch.nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from params import config, n, nn, device, weight_decay, training_batch_size, string_length, training_size, writer, work_dir, stacking

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(torch.nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(torch.nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)
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

class Block(torch.nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = torch.nn.ModuleDict(dict(
            c_fc    = torch.nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = torch.nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(torch.nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe = torch.nn.Embedding(config.block_size, config.n_embd),
            h = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
    model.train()  # reset model back to training mode
    return mean_loss


"""
# cyclically rotate a tuple
def rot(i,a):
    #    return tuple(map(int,np.roll(np.array(a).reshape(4,nn),i,axis=1).ravel()))
    return a[i:nn]+a[:i]+a[nn+i:2*nn]+a[nn:nn+i]+a[2*nn+i:3*nn]+a[2*nn:2*nn+i]+a[3*nn+i:]+a[3*nn:3*nn+i]
"""


# conversion string <-> matrix
def char_to_sign(c, i):
    return int(2 * ((c >> i) & 1) - 1)
# effectively do nn % stacking == 0 first because simpler
nice = nn % stacking == 0
quarter_string_length = string_length//4
my_range = range(n) if nice else list(i for j in range(4) for i in range(j * quarter_string_length*stacking, j * quarter_string_length*stacking + nn))  # list for reusability
def string_to_array(s): # really, tensor to tuple by now!
    return tuple(
        char_to_sign(s[i // stacking] - 1, i % stacking)
        for i in my_range
    )
# Prepare powers-of-two weights [1, 2, 4, 8, ...] efficiently
powers_of_two = 2 ** torch.arange(stacking, dtype=torch.long)
def array_to_string(tensor): # tensor to tensor
    # Convert input tuple (+1/-1) directly to tensor on GPU or CPU, -1 → 0, +1 → 1
    tensor = (1+tensor >> 1).reshape(4, nn)
    # added: random rotation
    tensor = torch.roll(tensor, shifts=random.randrange(nn), dims=1)  # TODO use pytorch random instead
    # added: second rotation
    tensor[2] = torch.roll(tensor[2], shifts=random.randrange(nn), dims=0)
    if not nice:
        tensor = F.pad(tensor, (0, quarter_string_length*stacking-nn), mode='constant', value=0)  # pad
    # Compute integer encoding using vectorized matrix multiplication
    return 1 + tensor.reshape(string_length, stacking).matmul(powers_of_two)

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
        ix = array_to_string(self.words[idx])
        # ix = array_to_string(rot(random.randrange(nn),self.words[idx]))
        # ix = array_to_string(rot(idx%nn,self.words[idx]))  # cheap trick FAIL, figure out why?
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.zeros(self.block_size, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1  # index -1 will mask the loss at the inactive locations
        return x, y

# -----------------------------------------------------------------------------


model = Transformer(config)
model.to(device)
model.need_reload = True

out_path = os.path.join(work_dir, "model.pt")


def load_model():
    if model.need_reload:
        model.load_state_dict(torch.load(out_path, weights_only=True))
        print('resuming from existing model in the workdir')


def save_model():
    print('saving model to workdir')
    torch.save(model.state_dict(), out_path)


def get_loss(dataset, step, name):
    loss = evaluate(model, dataset, batch_size=100, max_batches=10)
    writer.add_scalar("Loss/"+name, loss, step)
    writer.flush()
    if name == 'test':
        print(f"{step=} {name} {loss=:.6f}", end='\t'); sys.stdout.flush()
    return loss


test_set_size = 1000
if training_size <= test_set_size:
    raise SystemExit("{training_size=} must be greater than {test_set_size=}")


def train(data, **kwargs):
    resume = kwargs.get("resume", False)
    num_workers = kwargs.get("num_workers", 3)  # should be parameterisable TODO
    max_steps = kwargs.get("max_steps", -1)
    seed = kwargs.get("seed", 3407)
    # optimization -> slowly being moved to params.py
    # batch_size = kwargs.get("batch_size", 32)
    # weight_decay = kwargs.get("weight_decay", 0.01)
    learning_rate = kwargs.get("learning_rate", 5e-4)
    batch_size = training_batch_size
    eval_freq = kwargs.get("eval_freq", 500)

    # system inits
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.makedirs(work_dir, exist_ok=True)

    block_size = config.block_size
    vocab_size = config.vocab_size  # should one check that this is correct?

    # print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if resume:
        try:
            load_model()
        except FileNotFoundError:
            pass
    model.need_reload = True  # we will change the model no matter what
    data_len = len(data)
    print(f"number of examples in the dataset: {data_len}")
    print(f"max word length+1: {block_size}")
    print(f"number of unique characters in the vocabulary: {vocab_size}")

    # convert to torch tensors
    data = torch.tensor(list(data), dtype=torch.long).share_memory_()
    for i in range(test_set_size):
        j = random.randrange(i+1, data_len)
        data[i], data[j] = data[j], data[i]
    test_data = data[:test_set_size]
    train_data = data[test_set_size:]
    print(f"split up the dataset into {len(train_data)} training examples and {len(test_data)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init sampler, dataloader
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10))
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    batch_iter = iter(train_loader)  # wrap loader in an iterator explicitly

    # training loop
    step = 0
    save_step = None
    get_loss(train_dataset, step, "train")
    best_loss = get_loss(test_dataset, step, "test")
    gpu_batch = (t.to(device, non_blocking=True) for t in next(batch_iter))  # note that batch_loader produces tuples of length 2
    while True:
        # get the next batch, ship to device, and unpack it to input and target
        try:
            next_batch = next(batch_iter)
        except StopIteration:
            # Restart iterator if at end of epoch
            batch_iter = iter(train_loader)
            next_batch = next(batch_iter)
        gpu_next_batch = (t.to(device, non_blocking=True) for t in next_batch)
        # gpu_batch = (t.to(device, non_blocking=True) for t in next_batch)

        # Train on the current batch
        # feed into the model
        logits, loss = model(*gpu_batch)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # evaluate the model
        step += 1
        if step % eval_freq == 0 or step == max_steps:
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            get_loss(train_dataset, step, "train")
            test_loss = get_loss(test_dataset, step, "test")
            # save the model to disk if it has improved
            if test_loss < best_loss:
                save_model()
                best_loss = test_loss
                save_step = step
                if step == max_steps:
                    max_steps += eval_freq  # don't quit on a winning streak
            elif test_loss > best_loss+.2 or step == max_steps:  # termination conditions: done, or we've probably massively overfitted
                break

        gpu_batch = gpu_next_batch

    print("")
    return save_step


# def crop(row):
#    return tuple(row[:next((i for i, x in enumerate(row) if x == 0), len(row))])


def sample(**kwargs):
    num_samples = kwargs.get("num_samples", 1000)
    seed = kwargs.get("seed", 3407)
    top_k = kwargs.get("top_k", -1)  # -1 means no top-k

    # system inits
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.makedirs(work_dir, exist_ok=True)

    block_size = config.block_size

    load_model()
    model.need_reload = False

    X_init = torch.zeros(num_samples, 1, dtype=torch.long).to(device)
    top_k = top_k if top_k != -1 else None
    X_samp = generate(model, X_init, block_size-1, top_k=top_k, do_sample=True).cpu()
    # samples = [ crop(row[1:].tolist()) for row in X_samp ]
    # here we assume that the length is entirely fixed -> no need for crop. revert if encoding has variable length
    samples = [string_to_array(row[1:string_length+1].tolist()) for row in X_samp if len(row) >= string_length+1]
    return samples
