if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# Transformer model and training utilities.
import os
import sys

import torch
import torch.nn
from torch.nn import functional as F
import params  # for work_dir
from params import na, nn, nn2, nm, device, config, resume_training, fft, cst
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
        modules = dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe = torch.nn.Embedding(config.block_size, config.n_embd),
            h = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embd),
        )
        self.uses_score = config.transformer_uses_score
        if self.uses_score:
            modules["wse"] = torch.nn.Embedding(nn2 + 1, config.n_embd)
        self.transformer = torch.nn.ModuleDict(modules)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Report total parameter count.
        n_params = sum(p.numel() for p in self.parameters())
        print("number of transformer parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, batch0, score_batch=None, offset=0, compute_loss=False):
        b = batch0.shape[0]
        if self.uses_score:
            if score_batch is None:
                raise RuntimeError("score_batch is required when transformer_uses_score=True")
            score_batch = score_batch.to(dtype=self.transformer.wse.weight.dtype)
            m = batch0.shape[1]
            batch = batch0[:, :, :-1] if self.training else batch0
            t = batch.shape[2] + 1
            pos_emb = self.transformer.wpe.weight[offset:offset + t * m].view(1, m, t, config.n_embd)
            x = pos_emb.repeat(b, 1, 1, 1)
            x[:, :, 0, :] += score_batch @ self.transformer.wse.weight
            x = x.view(b * m, t, config.n_embd)
            batch = batch.view(b * m, batch.shape[2])
        else:
            batch = batch0[:, :-1] if self.training else batch0
            t = batch.shape[1] + 1
            pos_emb = self.transformer.wpe.weight[:t].view(1, t, config.n_embd)
            x = pos_emb.repeat(b, 1, 1)
        if batch.numel() > 0:
            x[:, 1:, :] += self.transformer.wte(batch)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch0.view(-1)) if compute_loss else None
        return logits, loss

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
    if model.uses_score:
        B = batch.shape[0]
        ff = torch.ones(B, 1, nn2+1, device=device, dtype=model.transformer.wse.weight.dtype)
        for j in range(nm):
            offset = j * segment_string_length
            for i in range(segment_string_length):
                batch_cond = batch[:, offset:offset+i].view(B, 1, i)
                logits, _ = model(batch_cond, score_batch=ff, offset=offset)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                batch[:, offset+i] = torch.multinomial(probs, num_samples=1).view(-1)
            signs = decode_segment_tokens(batch[:, offset:offset+segment_string_length], dtype=model.transformer.wse.weight.dtype)
            arrays[:, j*nn:(j+1)*nn] = signs.to(dtype=torch.int8)
            f = cst * torch.fft.rfft(signs, dim=1)
            ff = torch.clamp(ff - torch.view_as_real(f).square().sum(dim=-1).unsqueeze(1), min=0)
        return
    block_size = model.get_block_size()
    for i in range(block_size):
        batch_cond = batch[:, :i]
        logits, _ = model(batch_cond)
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
    if config.transformer_uses_score:
        return (signs1.view(B, nm, segment_string_length, config.stacking) << bit_positions).sum(dim=3)
    return (signs1.view(B, config.block_size, config.stacking) << bit_positions).sum(dim=2)

@torch.no_grad()
def prepare_training_inputs(batch):
    string_batch = array_to_string(batch)
    if model.uses_score:
        ff = torch.view_as_real(fft(batch)).square().sum(dim=-1).to(dtype=model.transformer.wse.weight.dtype)
        ff = torch.flip(torch.cumsum(torch.flip(ff, dims=(1,)), dim=1), dims=(1,))
        return string_batch, {"score_batch": ff, "compute_loss": True}
    return string_batch, {"compute_loss": True}

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

    # Initialize optimizer.
    optimiser_kwargs = dict(lr=lr_sched(0), weight_decay=config.weight_decay, betas=(0.9, 0.99))
    if device.startswith('cuda'):
        optimiser_kwargs["fused"] = True
    try:
        optimiser = torch.optim.AdamW(model.parameters(), **optimiser_kwargs)
    except (TypeError, RuntimeError):
        optimiser_kwargs.pop("fused", None)
        optimiser = torch.optim.AdamW(model.parameters(), **optimiser_kwargs)

    # Training loop.
    step = 0
    total_loss = 0
    while True:
        # Sample a batch, apply random symmetry, and train.
        batch = randomise_symmetry(data[torch.randint(data_len, (batch_size,))].to(device, non_blocking=True), params.symmetry_ctx)
        model_input, model_kwargs = prepare_training_inputs(batch)
        logits, loss = model(model_input, **model_kwargs)
        total_loss += loss.item()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{step=}: loss is NaN")
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr_sched(step)
        # Backpropagation step.
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        # Periodic logging/checkpointing.
        step += 1
        if step % eval_freq == 0:
            print(f"{step=}", end='\t')
            last_loss = total_loss/eval_freq
            logger.record_loss(last_loss, step, "train")
            total_loss = 0
            save_model()
        if step == max_steps:
            save_model()
            with open(logger.stats_file, 'a') as file:
                file.write(f'training: final loss={last_loss}\n')
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
