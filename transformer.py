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
from params import na, nn, nn2, nm, device, config, resume_training, rotate
import logger

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Transformer Language Model

class myActiv(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.6*x)


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
        self.causal_mask = torch.triu(torch.ones(config.block_size, config.block_size,
                                               device=device, dtype=torch.bool),diagonal=1).view(1, 1, config.block_size, config.block_size)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
        nh = self.n_head
        hs = C // nh
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
            act     = myActiv(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(torch.nn.Module):
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

    def forward(self, idx0, compute_loss=False):
        b = idx0.shape[0]
        idx = idx0[:, :self.block_size-1]  # in training, remove last token since don't need to predict next one
        t = idx.shape[1] + 1
        # forward the transformer itself
        pos_emb = self.transformer.wpe.weight[:t]  # position embeddings of shape (1, t, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t-1, n_embd)
        x = pos_emb.repeat(b, 1, 1)  # (b, t, n_embd)
        x[:, 1:, :] += tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # if requested, also calculate the loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx0.view(-1)) if compute_loss else None
        return logits, loss

# -----------------------------------------------------------------------------


def init_model():
    global model  # to simplify, model, etc, are global
    global model_path
    global bit_positions
    global bit_positions_cpu  # eww
    global nn_pad
    model = Transformer(config)
    model.to(device)
    model.need_reload = True
    model = torch.compile(model)
    model_path = os.path.join(params.work_dir, "model.pt")
    # stuff for coding/decoding arrays
    bit_positions = torch.arange(config.stacking, device=device, dtype=torch.int)
    bit_positions_cpu = torch.arange(config.stacking, dtype=torch.int)
    segment_string_length = config.block_size // nm
    nn_pad = segment_string_length * config.stacking

def load_model():
    if model.need_reload:
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        print('resuming from existing model in the workdir')


def save_model():
    print('saving model to workdir')
    torch.save(model.state_dict(), model_path)


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model


@torch.inference_mode()
def generate(idx, max_new_tokens, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for i in range(max_new_tokens):
        idx_cond = idx[:, :i]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / config.temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx[:, i] = torch.multinomial(probs, num_samples=1).view(-1)
        else:
            _, idx[:, i] = torch.topk(probs, k=1, dim=-1).view(-1)


@torch.inference_mode()
def evaluate(sample):
    model.eval()
    batch = sample.to(device)
    logits, loss = model(batch, compute_loss=True)
    mean_loss = loss.mean().item()
    model.train()  # reset model back to training mode
    return mean_loss

# conversion string of tokens <-> array of signs
@torch.no_grad()
def string_to_array(X):  # really, int tensor to int8 tensor
    B = X.shape[0]
    signs = ((((X.unsqueeze(-1) >> bit_positions) & 1) << 1) - 1).view(B, nm, nn_pad)
    return signs[:, :, :nn].to(dtype=torch.int8).view(B, na)

@torch.no_grad()
def string_to_array_cpu(X):  # really, int tensor to int8 tensor
    B = X.shape[0]
    signs = ((((X.unsqueeze(-1) >> bit_positions_cpu) & 1) << 1) - 1).view(B, nm, nn_pad)
    return signs[:, :, :nn].to(dtype=torch.int8).view(B, na)

@torch.no_grad()
def array_to_string(signs):  # int8 tensor to int tensor
    B = signs.shape[0]
    signs1 = torch.zeros((B, nm, nn_pad), device=device, dtype=torch.int)
    signs1[:, :, :nn] = signs.view(B, nm, nn)
    # Convert -1 → 0, +1 → 1
    signs1 += 1
    signs1 >>= 1
    return (signs1.view(B, config.block_size, config.stacking) << bit_positions).sum(dim=2)


def train(data, **kwargs):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('high')  # dangerous, can cause NaN
    data_len = len(data)
    test_len = min(config.test_set_size, data_len//2)
    vocab_size = config.vocab_size  # should one check that this is correct?
    string_length = config.block_size
    print(f"number of examples in the dataset: {data_len}")
    print(f"max word length: {string_length}")
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

    # below is no no: data is read-only. otoh the data is supposed to be already randomised
    """
    for i in range(test_len):
        j = torch.randint(i, data_len, ()).item()
        data[[i, j]] = data[[j, i]]
    """
    training_batch_size = config.training_batch_size
    train_len = (data_len - test_len) // training_batch_size * training_batch_size
    test_len = data_len - train_len
    test_data = data[:test_len]
    train_data = data[test_len:]
    test_len = len(test_data)
    train_len = len(train_data)
    print(f"split up the dataset into {train_len} training examples and {test_len} test examples")

    # init optimiser
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr_sched(0), weight_decay=config.weight_decay, betas=(0.9, 0.99), fused=True)

    test_sample = array_to_string(rotate(test_data)).cpu()  # just get it all, and encode it too

    # training loop
    step = 0
    save_step = 0
    best_loss = evaluate(test_sample)
    logger.record_loss(best_loss, step, "test")
    idx = train_len
    while True:
        if idx == train_len:
            idx = 0
            perm = torch.randperm(train_len)
        # feed into the model
        batch = array_to_string(rotate(train_data[perm[idx:idx+training_batch_size]].to(device, non_blocking=True)))
        idx += training_batch_size
        logits, loss = model(batch, compute_loss=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{step=}: loss is NaN")
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr_sched(step)
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        # periodically test/save the model
        step += 1
        if step % eval_freq == 0 or step == max_steps:
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
                print('')  # to have nicely aligned test / train stats :)
                sys.stdout.flush()
                if test_loss - loss + (step-save_step)/max_steps > .3:  # termination condition 1: we've probably massively overfitted
                    break
            if step == max_steps:  # termination condition 2: hard cutoff
                break
    print('')
    with open(logger.stats_file, 'a') as file:
        file.write(f'training: {best_loss=} at {save_step=}\n')


if True:  #not device.startswith('cuda'):
    # unoptimised version of sample if cuda not installed
    @torch.no_grad()
    def sample():
        load_model()
        model.need_reload = False
        if device.startswith('cuda'):
            torch.cuda.empty_cache()  # Free memory
        torch.set_float32_matmul_precision('high')
        X = torch.empty(config.sample_batch_size, config.block_size, dtype=torch.int, device=device)
        arrays_cpu = torch.empty((config.sample_size, na), dtype=torch.int8, pin_memory=True)
        for i in range(0, config.sample_size, config.sample_batch_size):
            j = i + config.sample_batch_size
            print('*', end=''); sys.stdout.flush()
            generate(X, config.block_size, do_sample=True)
            arrays_cpu[i:j] = string_to_array(X)
        print('')
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
        arrays_cpu = torch.empty((config.sample_size, na), dtype=torch.int8)
        if num_batches == 0:
            return arrays_cpu
        X = torch.empty(config.sample_batch_size, config.block_size, dtype=torch.int, device=device)
        X_cpu = [torch.zeros_like(X, device='cpu', pin_memory=True) for _ in range(2)]
        event = [torch.cuda.Event() for _ in range(2)]
        idx = 0
        for i in range(num_batches+1):
            if i > 0:
                event[idx].synchronize()
            if i < num_batches:
                print('*', end=''); sys.stdout.flush()
                with torch.cuda.stream(stream):
                    generate(X, config.block_size, do_sample=True)
                    X_cpu[1-idx].copy_(X, non_blocking=True)
                    stream.record_event(event[1-idx])
            if i > 0:
                arrays_cpu[(i-1)*config.sample_batch_size:i*config.sample_batch_size] = string_to_array_cpu(X_cpu[idx])
            idx = 1 - idx
        print('')
        return arrays_cpu
