if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# Transformer model and training utilities.
import os
import sys

import torch
import torch.nn
from torch.nn import functional as F
import params  # for work_dir
from params import na, nn, nm, device, config, resume_training
import logger
from symmetry import randomise_symmetry
from timestamped_print import print

# Transformer language model.

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
        B, T, C = x.shape  # batch, sequence, embedding (n_embd)
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
            c_proj  = torch.nn.Linear(config.n_embd2, config.n_embd),
            act     = myActiv(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

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
        # GFlowNet partition-function estimate (learned scalar; unused when params.gflow is False).
        self.log_Z = torch.nn.Parameter(torch.zeros(()))
        # Report total parameter count.
        n_params = sum(p.numel() for p in self.parameters())
        print("number of transformer parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, batch0, compute_loss=False, compute_logpf=False):
        b = batch0.shape[0]
        batch = batch0[:, :-1] if self.training else batch0
        t = batch.shape[1] + 1
        pos_emb = self.transformer.wpe.weight[:t].view(1, t, config.n_embd)
        x = pos_emb.repeat(b, 1, 1)
        if batch.shape[1] > 0:
            x[:, 1:, :] += self.transformer.wte(batch)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch0.view(-1)) if compute_loss else None
        log_pf = None
        if compute_logpf:
            # logits[:, t, :] predicts batch0[:, t]; sum log-prob of the actual tokens.
            log_probs = F.log_softmax(logits, dim=-1)
            log_pf = log_probs.gather(-1, batch0.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        return logits, loss, log_pf

def init_model():
    global model
    global model_path
    global bit_positions
    global nn_pad
    global segment_string_length
    model = Transformer(config).to(device)
    model.need_reload = True
    if device.startswith('cuda'):
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
    model_path = os.path.join(params.work_dir, "model.pt")
    # Bit-packing helpers for array<->token conversion.
    bit_positions = torch.arange(config.stacking, device=device, dtype=torch.int)
    segment_string_length = config.block_size // nm
    nn_pad = segment_string_length * config.stacking

def load_model():
    if model.need_reload:
        load_result = model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        print('resuming from existing model in the workdir')
        if load_result.missing_keys or load_result.unexpected_keys:
            print('warning: partial model load')
            if load_result.missing_keys:
                print(f'missing keys: {load_result.missing_keys}')
            if load_result.unexpected_keys:
                print(f'unexpected keys: {load_result.unexpected_keys}')
        model.need_reload = False


def save_model():
    print('saving model to workdir')
    torch.save(model.state_dict(), model_path)


# Sampling/evaluation helpers.
@torch.inference_mode()
def decode_segment_tokens(X, dtype=torch.int8):
    B = X.shape[0]
    signs = ((((X.unsqueeze(-1) >> bit_positions) & 1) << 1) - 1).view(B, nn_pad)
    return signs[:, :nn].to(dtype=dtype)


@torch.inference_mode()
def generate(batch, arrays):
    """
    Fill token positions autoregressively in-place for a batch of sequences.
    """
    temperature = config.temperature + params.gen * config.temperature_delta
    block_size = model.get_block_size()
    for i in range(block_size):
        batch_cond = batch[:, :i]
        logits, _, _ = model(batch_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        batch[:, i] = torch.multinomial(probs, num_samples=1).view(-1)
    arrays.copy_(string_to_array(batch))

# Conversion between token strings and +/-1 arrays.
@torch.no_grad()
def string_to_array(X):
    B = X.shape[0]
    signs = decode_segment_tokens(X.view(B * nm, segment_string_length), dtype=torch.int8)
    return signs.view(B, na)

@torch.no_grad()
def array_to_string(signs):
    B = signs.shape[0]
    signs1 = torch.zeros((B, nm, nn_pad), device=device, dtype=torch.int)
    signs1[:, :, :nn] = signs.view(B, nm, nn)
    # Map -1 -> 0 and +1 -> 1.
    signs1 += 1
    signs1 >>= 1
    return (signs1.view(B, config.block_size, config.stacking) << bit_positions).sum(dim=2)

def log_reward(arrays, tau):
    """Compute log R(x) = -score(x)/tau for a batch of ±1 arrays, clamped from below.

    Runs inside torch.no_grad() so the returned tensor is a plain tensor with no
    grad requirement — safe to use as a target in the trajectory-balance loss.
    """
    with torch.no_grad():
        m = arrays.view(-1, nm, nn).to(dtype=params.real_dtype)
        f = params.cst * torch.fft.rfft(m, dim=2)
        ff = torch.view_as_real(f).square().sum(dim=(1, 3))
        s = 2 * (ff - 1 - torch.log(ff))
        scores = 2 * s.sum(dim=1) - s[:, 0]
        return (-scores / tau).clamp(min=config.gflow_clip_logR)


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

    # Runtime-adjusted training parameters.
    max_steps = kwargs.get("max_steps", -1)
    eval_freq = kwargs.get("eval_freq", 500)

    # Learning rate schedule.
    lr_sched = kwargs.get("lr_sched", lambda step: 5e-4)

    if resume_training:
        try:
            load_model()
        except FileNotFoundError:
            pass
    model.train()

    batch_size = config.training_batch_size

    # Initialize optimizer.  In GFlowNet mode log_Z gets its own faster LR.
    if params.gflow:
        main_params = [p for n, p in model.named_parameters() if not n.endswith("log_Z")]
        logZ_params = [p for n, p in model.named_parameters() if n.endswith("log_Z")]
        optim_groups = [
            {"params": main_params, "lr": lr_sched(0), "weight_decay": config.weight_decay},
            {"params": logZ_params, "lr": config.gflow_logZ_lr, "weight_decay": 0.0},
        ]
    else:
        optim_groups = [{"params": model.parameters(), "lr": lr_sched(0), "weight_decay": config.weight_decay}]
    optimiser_kwargs = dict(betas=(0.9, 0.99))
    if device.startswith('cuda'):
        optimiser_kwargs["fused"] = True
    try:
        optimiser = torch.optim.AdamW(optim_groups, **optimiser_kwargs)
    except (TypeError, RuntimeError):
        optimiser_kwargs.pop("fused", None)
        optimiser = torch.optim.AdamW(optim_groups, **optimiser_kwargs)

    if params.gflow:
        tau = max(config.gflow_tau * (config.gflow_tau_delta ** params.gen), config.gflow_tau_min)
        print(f"GFlowNet training: tau={tau:.4f}, initial log_Z={model.log_Z.item():.3f}")

    # Training loop.
    step = 0
    total_loss = 0
    while True:
        # Sample a batch, apply random symmetry, and train.
        batch = torch.randint(data_len, (batch_size,))
        arrays = randomise_symmetry(data[batch].to(device, non_blocking=True), params.symmetry_ctx)
        string_batch = array_to_string(arrays)
        if params.gflow:
            _, _, log_pf = model(string_batch, compute_logpf=True)
            log_R = log_reward(arrays, tau)
            loss = ((model.log_Z + log_pf - log_R) ** 2).mean()
        else:
            _, loss, _ = model(string_batch, compute_loss=True)
        total_loss += loss.item()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{step=}: loss is NaN")
        # Update main-param LR (log_Z keeps its own LR).
        optimiser.param_groups[0]['lr'] = lr_sched(step)
        # Backpropagation step.
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        # Periodic logging/checkpointing.
        step += 1
        if step % eval_freq == 0:
            print(f"{step=}", end='\t')
            logger.record_loss(total_loss/eval_freq, step, "train")
            if params.gflow:
                logger.record_loss(model.log_Z.item(), step, "log_Z")
            total_loss = 0
            save_model()
        if step == max_steps:
            save_model()
            break
        #
    print('')

# Sampling
@torch.no_grad()
def sample():
    load_model()
    model.eval()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('high')
    X = torch.empty(config.sample_batch_size, config.block_size, dtype=torch.int, device=device)
    arrays_cpu = torch.empty((config.sample_size, na), dtype=torch.int8, pin_memory=True)
    arrays = torch.empty((config.sample_batch_size, na), dtype=torch.int8, device=device)
    for i in range(0, config.sample_size, config.sample_batch_size):
        j = min(i + config.sample_batch_size, config.sample_size)
        cur_batch = X[:j-i]
        cur_arrays = arrays[:j-i]
        print('*', end=''); sys.stdout.flush()
        generate(cur_batch, cur_arrays)
        arrays_cpu[i:j] = cur_arrays
    print('')
    return arrays_cpu
