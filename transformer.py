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
from params import n, na, nn, nn2, nm, device, config, resume_training, rotate, fft, cst
import logger

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Transformer Language Model

class myActiv(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.6*x)


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        C, nh = config.n_embd, config.n_head
        self.c_attn = torch.nn.Linear(C, 3 * C)
        self.c_proj = torch.nn.Linear(C, C)
        self.n_head, self.n_embd = nh, C

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
        nh, hs = self.n_head, C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)

class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = torch.nn.ModuleDict(dict(
            c_fc    = torch.nn.Linear(config.n_embd, config.n_embd2),
            c_score = torch.nn.Linear(nn2+1, config.n_embd2),
            c_proj  = torch.nn.Linear(config.n_embd2, config.n_embd),
            act     = myActiv(),
        ))
        m = self.mlp
        self.mlpf = lambda x, s: m.c_proj(m.act(m.c_fc(x)+m.c_score(s).unsqueeze(1)))  # MLP forward

    def forward(self, x, s):  # x: (B, T, n_embd)  s: (B, nn2+1)
        x = x + self.attn(self.ln_1(x))
        x = self.ln_2(x)
        x = x + self.mlpf(x, s)
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

    def forward(self, batch0, score_batch, offset=0, compute_loss=False):  # self.training => offset=0
        b = batch0.shape[0]  # batch0 (b, m=nm|1, t-1<=segment_string_length), score_batch (b, m, nn2+1)
        m = batch0.shape[1]  # if not self.training, m should be 1, otherwise nm
        # in training, remove last token since don't need to predict next one:
        batch = batch0[:, :,  :segment_string_length-1] if self.training else batch0
        t = batch.shape[2] + 1  # add one since we're going to predict the next token (as well as all previous ones)
        # forward the transformer itself
        pos_emb = self.transformer.wpe.weight[offset:offset+t*m].view(m, t, config.n_embd)  # position embeddings of shape (m, t, n_embd)
        tok_emb = self.transformer.wte(batch)  # token embeddings of shape (b, m, t-1, n_embd)
        x = pos_emb.repeat(b, 1, 1, 1)  # (b, m, t, n_embd)
        x[:, :, 1:, :] += tok_emb  # careful, shift by 1 !!
        xx = x.view(b*m, t, config.n_embd)
        s = score_batch.view(b*m, nn2+1)
        for block in self.transformer.h:
            xx = block(xx, s)
        xx = self.transformer.ln_f(xx)
        logits = self.lm_head(xx)
        # if requested, also calculate the loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch0.view(-1)) if compute_loss else None
        return logits, loss

# -----------------------------------------------------------------------------


def init_model():
    global model  # to simplify, model, etc, are global
    global model_path
    global bit_positions
    global bit_positions_cpu  # eww
    global nn_pad
    global segment_string_length
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
        model.need_reload = False


def save_model():
    print('saving model to workdir')
    torch.save(model.state_dict(), model_path)


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model


@torch.inference_mode()
def generate(batch):
    #block_size = model.get_block_size()
    B = batch.shape[0]
    ff = torch.ones(B, 1, nn2+1, device=device, dtype=torch.float32)
    for j in range(nm):
        offset = j * segment_string_length
        for i in range(segment_string_length):
            batch_cond = batch[:, offset:offset+i].view(B, 1, i)
            # forward the model to get the logits for the index in the sequence
            logits, _ = model(batch_cond, ff, offset)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / config.temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            batch[:, offset+i] = torch.multinomial(probs, num_samples=1).view(-1)
        # that part sucks: copied from string_to_array and fft -- TODO absorb string_to_array here
        signs = ((((batch[:, offset:offset+segment_string_length].unsqueeze(-1) >> bit_positions) & 1) << 1) - 1).view(B,nn_pad)[:, :nn]
        f = cst * torch.fft.rfft(signs, dim=1)  # B, nn2+1
        #ff -= torch.view_as_real(f).square().sum(dim=-1).unsqueeze(1)  # TODO should we relu? cause negative doesn't make much sense
        ff = torch.clamp(ff - torch.view_as_real(f).square().sum(dim=-1).unsqueeze(1), min=0)


"""
@torch.inference_mode()
def evaluate(sample):
    model.eval()
    batch = sample.to(device)
    logits, loss = model(batch, compute_loss=True)
    mean_loss = loss.mean().item()
    model.train()  # reset model back to training mode
    return mean_loss
"""

# conversion string of tokens <-> array of signs
@torch.no_grad()
def string_to_array(X):  # really, int tensor to int8 tensor
    B = X.shape[0]
    signs = ((((X.unsqueeze(-1) >> bit_positions) & 1) << 1) - 1).view(B, nm, nn_pad)
    return signs[:, :, :nn].to(dtype=torch.int8).view(B, na)

"""
@torch.no_grad()
def string_to_array_cpu(X):  # really, int tensor to int8 tensor
    B = X.shape[0]
    signs = ((((X.unsqueeze(-1) >> bit_positions_cpu) & 1) << 1) - 1).view(B, nm, nn_pad)
    return signs[:, :, :nn].to(dtype=torch.int8).view(B, na)
"""

@torch.no_grad()
def array_to_string(signs):  # int8 tensor to int tensor
    B = signs.shape[0]
    signs1 = torch.zeros((B, nm, nn_pad), device=device, dtype=torch.int)
    signs1[:, :, :nn] = signs.view(B, nm, nn)
    # Convert -1 → 0, +1 → 1
    signs1 += 1
    signs1 >>= 1
    return (signs1.view(B, nm, segment_string_length, config.stacking) << bit_positions).sum(dim=3)


def train(data, **kwargs):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('high')  # dangerous, can cause NaN
    data_len = len(data)
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

    model.train()

    batch_size = config.training_batch_size

    # init optimiser
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr_sched(0), weight_decay=config.weight_decay, betas=(0.9, 0.99), fused=True)

    # training loop
    step = 0
    total_loss = 0
    while True:
        # feed into the model
        batch = rotate(data[torch.randint(data_len,(batch_size,))].to(device, non_blocking=True))
        string_batch = array_to_string(batch)
        f = fft(batch)
        #score_batch = torch.log(torch.real(f*f.conj()).sum(dim=1))
        ff = torch.view_as_real(f).square().sum(dim=-1)  # (batch_size, nm, nn2+1)
        for i in range(1,nm):
            ff[:, :i, :] += ff[:, i, :].unsqueeze(1)
        _, loss = model(string_batch, ff, compute_loss=True)
        total_loss += loss
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
        if step % eval_freq == 0:
            print(f"{step=} ({step*batch_size/data_len:.1f} epochs)", end='\t')
            logger.record_loss(total_loss/eval_freq, step, "train")
            total_loss = 0
            save_model()
        if step == max_steps:
            save_model()
            break
        #
    print('')

# if not device.startswith('cuda'):
# unoptimised version of sample if cuda not installed
@torch.no_grad()
def sample():
    load_model()
    model.eval()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('high')
    X = torch.empty(config.sample_batch_size, config.block_size, dtype=torch.int, device=device)
    arrays_cpu = torch.empty((config.sample_size, na), dtype=torch.int8, pin_memory=True)
    for i in range(0, config.sample_size, config.sample_batch_size):
        j = i + config.sample_batch_size
        print('*', end=''); sys.stdout.flush()
        generate(X)
        arrays_cpu[i:j] = string_to_array(X)
    print('')
    return arrays_cpu
"""
else:
    # sample with CPU double buffering TODO reinstate at some point
    stream = torch.cuda.Stream()
    @torch.no_grad()
    def sample():
        load_model()
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
"""
