if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# based on makemore.py
import os
import sys
import math

import torch
import torch.nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import params  # for work_dir
from params import na, nn, device, config, resume_training, rotate
import logger

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Transformer Language Model

class MyGELU(torch.nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return x / (1.0 + torch.exp(-1.6*x))
        # return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

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
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

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
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

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
            c_fc    = torch.nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = torch.nn.Linear(config.n_embd2, config.n_embd),
            act     = MyGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

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
        print("number of transformer parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
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


def init_model():
    global model  # to simplify, model, etc, are global
    global model_path
    global powers_of_two
    global my_range
    global string_length
    #global segment_string_length
    global nice
    model = Transformer(config)
    model.to(device)
    model.need_reload = True
    model = torch.compile(model)
    model_path = os.path.join(params.work_dir, "model.pt")
    # stuff for coding/decoding arrays
    powers_of_two = 2 ** torch.arange(config.stacking, dtype=torch.int8)  # Prepare powers-of-two weights [1, 2, 4, 8, ...] efficiently
    string_length = config.block_size - 1
    # segment_string_length = string_length//nm
    # nice = nn % config.stacking == 0  # effectively do nn % stacking == 0 first because simpler
    nice = na % config.stacking == 0
    my_range = range(na) # if nice else list(i for j in range(nm) for i in range(j * segment_string_length*config.stacking, j * segment_string_length*config.stacking + nn))  # list for reusability


def load_model():
    if model.need_reload:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print('resuming from existing model in the workdir')


def save_model():
    print('saving model to workdir')
    torch.save(model.state_dict(), model_path)


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model


@torch.no_grad()
def generate(idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for i in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx[:, :i+1] if i < block_size else idx[:, i+1-block_size:i+1]
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
            idx[:, i+1] = torch.multinomial(probs, num_samples=1).view(-1)
        else:
            _, idx[:, i+1] = torch.topk(probs, k=1, dim=-1).view(-1)


@torch.inference_mode()
def evaluate(sample):
    model.eval()
    batch = [t.to(device) for t in sample]
    logits, loss = model(*batch)
    mean_loss = loss.mean().item()
    model.train()  # reset model back to training mode
    return mean_loss


bit_positions = torch.arange(config.stacking, device=device)
@torch.no_grad()
def string_to_array(X):  # really, int tensor to float tensor
    X = X[:,1:] - 1  # remove initial zero
    bits = (X.unsqueeze(-1) >> bit_positions) & 1  # shape (B, n, stacking)
    signs = (bits * 2 - 1).to(torch.int8) # now in {-1, +1}
    result = signs.reshape(config.sample_batch_size, string_length*config.stacking)
    return result[:,:na]

def array_to_string(array0):  # (dtype=long) tensor to tensor
    # code updated to make it clearer that we don't want to change the original array!
    array = array0.clone()
    rotate(array)
    if score:  # for testing purposes: does the randomisation respect score?
        if not torch.all(array0.abs()==1) or not torch.all(array.abs()==1):
            raise RuntimeError("array not +-1",array)
        scores = score(torch.stack((array0,array.view(na))))
        if torch.abs(scores[0]-scores[1]) > 1e-5:
            raise RuntimeError("score not preserved by randomisation", scores, torch.abs(scores[0]-scores[1]).item())
    # Convert -1 → 0, +1 → 1
    #array1 = (1+array>>1).view(nm,nn)
    array1 = 1+array>>1
    # pad if necessary
    if not nice:
        array1 = F.pad(array1, (0,string_length*config.stacking-na), mode='constant', value=0)
    #if not nice:
    #    array1 = F.pad(array1, (0, segment_string_length*config.stacking-nn), mode='constant', value=0)
    # Compute integer encoding using vectorized matrix multiplication
    return 1 + array1.view(string_length, config.stacking).matmul(powers_of_two)


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
        x = torch.cat([torch.tensor([0], dtype=torch.long), ix])
        y = torch.cat([ix, torch.tensor([-1], dtype=torch.long)])  # index -1 will mask the loss at the inactive locations
        return x, y


def train(data, **kwargs):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('high')  # dangerous, can cause NaN
    test_set_size = config.test_set_size
    block_size = config.block_size
    vocab_size = config.vocab_size  # should one check that this is correct?
    training_batch_size = config.training_batch_size
    data_len = len(data)
    if test_set_size >= data_len:
        raise SystemExit("training_size must be greater than test_set_size")
    print(f"number of examples in the dataset: {data_len}")
    print(f"max word length+1: {block_size}")
    print(f"number of unique characters in the vocabulary: {vocab_size}")

    # these parameters are adjusted dynamically during the run
    max_steps = kwargs.get("max_steps", -1)
    eval_freq = kwargs.get("eval_freq", 500)

    # learning rate is now a function of steps
    lr_sched = kwargs.get("lr_sched", lambda step: 5e-4)

    # for testing purposes only: scoring function
    global score
    score = kwargs.get("score", None)

    if resume_training:
        try:
            load_model()
        except FileNotFoundError:
            pass
    model.need_reload = True  # we will change the model no matter what

    data.share_memory_()
    for i in range(test_set_size):
        j = torch.randint(i, data_len, ()).item()
        data[[i, j]] = data[[j, i]]
    test_data = data[:test_set_size]
    train_data = data[test_set_size:]
    print(f"split up the dataset into {len(train_data)} training examples and {len(test_data)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)

    # init optimiser
    if device.startswith('cuda'):
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr_sched(0), weight_decay=config.weight_decay, betas=(0.9, 0.99), fused=True)
    else:
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr_sched(0), weight_decay=config.weight_decay, betas=(0.9, 0.99))

    # init sampler, dataloader
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10))
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=training_batch_size, pin_memory=True, num_workers=config.num_workers)
    batch_iter = iter(train_loader)  # wrap loader in an iterator explicitly
    # test_loader = DataLoader(test_dataset, shuffle=True, batch_size=100, num_workers=0)  # default sampler with shuffle = True is RandomSampler(replacement=False)
    test_sample = [torch.stack(ts, dim=0) for ts in zip(*test_dataset)]  # just get it all

    # training loop
    step = 0
    save_step = 0
    best_loss = evaluate(test_sample)
    logger.record_loss(best_loss, step, "test")
    while True:
        # get the next batch, ship to device, and unpack it to input and target
        try:
            batch = next(batch_iter)  # note that batch_loader produces tuples of length 2
        except StopIteration:
            # Restart iterator if at end of epoch
            batch_iter = iter(train_loader)
            batch = next(batch_iter)
        # Train on the current batch
        # feed into the model
        logits, loss = model(*(t.to(device, non_blocking=True) for t in batch))
        if not torch.isfinite(loss):
            raise RuntimeError("loss is NaN")
        
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr_sched(step)


        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        # periodically test/save the model
        step += 1
        if step % eval_freq == 0 or step == max_steps:
#            print(f"{step=}, {lr_sched(step)=} ", end='\t')
            print(f"{step=} ", end='\t')
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            logger.record_loss(loss, step, "train")
            # evaluate the model
            test_loss = evaluate(test_sample)
            logger.record_loss(test_loss, step, "test")
            # save the model to disk if it has improved
            if test_loss < best_loss:
                save_model()
                sys.stdout.flush()
                best_loss = test_loss
                save_step = step
            else:
                print('') # to have nicely aligned test / train stats :)
                sys.stdout.flush()
                if test_loss - best_loss + (step-save_step)/max_steps > .3:  # termination condition 1: we've probably massively overfitted
                    break
            if step == max_steps:  # termination condition 2: hard cutoff
                break
    print('')
    with open(logger.stats_file, 'a') as file:
        file.write(f'training: {best_loss=} at {save_step=}\n')


# def crop(row):
#    return tuple(row[:next((i for i, x in enumerate(row) if x == 0), len(row))])

if True: # not device.startswith('cuda'):
    # unoptimised version of sample if cuda not installed
    @torch.no_grad()
    def sample():
        load_model()
        model.need_reload = False
        torch.set_float32_matmul_precision('high')
        X = torch.zeros(config.sample_batch_size, config.block_size, dtype=torch.long, device=device)
        arrays_cpu = torch.empty((config.sample_size,na), dtype=torch.int8)
        for i in range(0, config.sample_size, config.sample_batch_size):
            j = i + config.sample_batch_size
            print('*', end=''); sys.stdout.flush()
            X.zero_()
            generate(X, config.block_size-1, do_sample=True)
            arrays_cpu[i:j] = string_to_array(X)
        return arrays_cpu
else:
    # sample with CPU double buffering TODO reinstate at some point
    stream = torch.cuda.Stream()
    @torch.no_grad()
    def sample():
        load_model()
        model.need_reload = False
        if device.startswith('cuda'):
            torch.cuda.empty_cache()  # Free memory
        torch.set_float32_matmul_precision('high')
        num_batches = config.sample_size // config.sample_batch_size
        arrays_cpu_full = torch.empty((0,na), dtype=torch.float32)  # TODO fix type?
        if num_batches == 0:
            return arrays_cpu
        X = torch.zeros(config.sample_batch_size, config.block_size, dtype=torch.long, device=device)
        arrays_cpu = [torch.empty((config.sample_batch_size,na), dtype=torch.float32, device='cpu', pin_memory=True) for _ in range(2)]
        event = [torch.cuda.Event() for _ in range(2)]
        idx = 0
        for i in range(num_batches+1):
            if i > 0:
                event[idx].synchronize()
            if i < num_batches:
                print('*', end=''); sys.stdout.flush()
                with torch.cuda.stream(stream):
                    X.zero_()
                    generate(X, config.block_size-1, do_sample=True)
                    arrays_cpu[1-idx].copy_(string_to_array(X), non_blocking=True)
                    stream.record_event(event[1-idx])
            if i > 0:
                arrays_cpu_full = torch.cat((arrays_cpu, arrays_cpu[idx]), dim=0)  # TODO clearly not optimized!
            idx = 1 - idx
        return new_arrays_set
