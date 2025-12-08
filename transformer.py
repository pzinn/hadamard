if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# based on makemore.py
import os
import sys
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import params  # for work_dir
from params import na, device, config, resume_training, rotate
import logger

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Autoencoder

class myActiv(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.6*x)


# -------------------------
# Pre-activation MLP Residual Block
# -------------------------
class PreActResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.ln1(x)
        out = self.act(out)
        out = self.fc1(out)

        out = self.ln2(out)
        out = self.act(out)
        out = self.fc2(out)

        return x + out

class DeepAE(nn.Module):
    def __init__(self, config, latent_dim=64, width=512, depth=4):
        super().__init__()

        self.register_buffer("latent_mu", None)
        self.register_buffer("latent_cov", None)

        d=na #TEMP
        # Encoder input projection
        self.enc_in = nn.Linear(d, width)

        # Encoder residual tower
        self.encoder_blocks = nn.Sequential(
            *[PreActResBlock(width) for _ in range(depth)]
        )

        # Projection to latent space
        self.to_latent = nn.Linear(width, latent_dim)

        # Decoder projection from latent
        self.dec_in = nn.Linear(latent_dim, width)

        # Decoder residual tower (mirrors encoder)
        self.decoder_blocks = nn.Sequential(
            *[PreActResBlock(width) for _ in range(depth)]
        )

        # Final projection back to input dim
        self.to_output = nn.Linear(width, d)  # logits; binarization applied outside

    def decode_from_latent(self, z):
        h = self.dec_in(z)
        h = self.decoder_blocks(h)
        logits = self.to_output(h)
        return torch.sign(logits)

    def estimate_latent_statistics(self, batch):
        """
        Computes latent mean and covariance from the encoder outputs.
        Returns:
            mu:       (latent_dim,) mean vector
            cov:      (latent_dim, latent_dim) covariance matrix
        """
        _, Z = model(batch)

        self.latent_mu = Z.mean(dim=0)         # (latent_dim,)

        # print(self.latent_mu)

        # Compute covariance:
        # cov = E[zz^T] - mu mu^T
        Z_centered = Z - self.latent_mu
        self.latent_cov = (Z_centered.t() @ Z_centered) / (Z_centered.size(0) - 1)

    def latent_gaussian_sampler(self, batch_size):
        """
        Samples z ~ N(mu, cov) using Cholesky factorization.
        """
        latent_dim = self.to_latent.out_features
        L = torch.linalg.cholesky(self.latent_cov + 1e-5 * torch.eye(latent_dim, device=device))  # TODO precompute?
        eps = torch.randn(batch_size, latent_dim, device=device)
        return self.latent_mu + eps @ L.t()

    def forward(self, x):
        h = self.enc_in(x)
        h = self.encoder_blocks(h)
        z = self.to_latent(h)

        h = self.dec_in(z)
        h = self.decoder_blocks(h)
        logits = self.to_output(h)

        return logits, z


# -----------------------------------------------------------------------------


def init_model():
    global model  # to simplify, model, etc, are global
    global model_path
    model = DeepAE(config)
    model.to(device)
    model.need_reload = True
    model = torch.compile(model)
    model_path = os.path.join(params.work_dir, "model.pt")

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
def generate(batch_size):
    z = model.latent_gaussian_sampler(batch_size)
    x_hat = model.decode_from_latent(z)
    return x_hat


def train(data, **kwargs):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('high')  # dangerous, can cause NaN
    data_len = len(data)
    print(f"number of examples in the dataset: {data_len}")

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

    batch_size = config.training_batch_size

    # init optimiser
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr_sched(0), weight_decay=config.weight_decay, betas=(0.9, 0.99), fused=True)

    # training loop
    step = 0
    while True:
        # feed into the model
        batch = rotate(data[torch.randint(data_len,(batch_size,))].to(device, non_blocking=True)).view(batch_size,na).to(dtype=torch.float32)
        logits, _ = model(batch)
        loss = ((logits - batch)**2).mean()  # or sum()?
        if not torch.isfinite(loss):
            raise RuntimeError(f"{step=}: loss is NaN")
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr_sched(step)
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        # periodically test/save the model
        if step % eval_freq == 0 or step == max_steps:
            loss2 = ((logits.sign() - batch)**2).mean()
            print(f"{step=}", end='\t')
            logger.record_loss(loss, step, "train")
            logger.record_loss(loss2, step, "train2")
            save_model()
            if step == max_steps:
                model.estimate_latent_statistics(batch)
                break
        step += 1
        #
    print('')

@torch.no_grad()
def sample():
    load_model()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('high')
    X = torch.empty(config.sample_batch_size, config.block_size, dtype=torch.int, device=device)
    arrays_cpu = torch.empty((config.sample_size, na), dtype=torch.int8, pin_memory=True)
    for i in range(0, config.sample_size, config.sample_batch_size):
        j = i + config.sample_batch_size
        print('*', end=''); sys.stdout.flush()
        arrays_cpu[i:j] = generate(config.sample_batch_size)
    print('')
    return arrays_cpu
