import torch
import math

if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# hadamard matrix parameters
n = 148  # size of matrix
# segment_sums = (1, 3, 3, 13)  # sum of squares must be n. must be a tuple (not a list!)

# the parameters below are sweepable: use values, or lists for a sweep

# training parameters
sample_size = 1_000_000
training_size = sample_size//20  # must be > test_set_size
learning_rate = 1e-3
training_batch_size = 1024  # for training. much smaller, obviously
weight_decay = 0.01
max_iterations = 30
training_steps = 150_000  # will be adjusted dynamically (to be less than that)
num_improve = 0  # number of times data get improved per generation. only used by improve2

# transformer parameters
n_layer = 4
n_embd = 64
# n_embd2 = 4*n_embd  # default choice
n_head = 4
stacking = 7  # [5,6,7,8,9,10]  # preferably a divisor of nn
temperature = 1.  # [.5, .75, 1, 1.25, 1.5, 1.75, 2]

# less important parameters
gen_decay = 0.0
sample_batch_size = 100_000  # for sampling. must be a divisor of sample_size, and < 65536
score_batch_size = None  # for scoring/improving. None means no batching
test_set_size = None  # None | < training_size, no more than 10% ideally
num_workers = None  # for cpu parallelisation

resume = False  # whether to resume a previous run
# resume = True
# if True, obviously, Hadamard parameters must be the same
# as well as transformer parameters (including stacking) unless resume_training = False
# training parameters can be different though
# also, for now resume is not compatible with sweep
if resume:
    pass
    # provide work_dir manually, default is latest
    # provide gen, default is latest

skip_first_training = False  # only meaningful if resume: start by sampling from existing model rather than training. leave False if unsure
skip_first_improve = resume  # leave as is unless you know what you're doing
resume_training = True  # whether to use previous model (not just previous data). True is a lot faster, False might be more accurate (?) leave True if unsure

test_score = False  # for debugging purposes, test whether randomisation of arrays (rotation) and other transformations preserves score


import time
random_seed = 1666 # int(time.time())  # 1746533706

device = 'cuda'  # device to use for compute, examples: cpu|cuda|cuda:2|mps

logging = 'wandb'  # '' | 'tensorboard' | 'wandb'
logging_mode = 'online'  # 'online' | 'offline' -- for wandb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug logging")

import subprocess
try:
    git_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stderr=subprocess.DEVNULL
    ).strip().decode()
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL
    ).strip().decode()
    version = git_branch + " " + git_commit
except subprocess.CalledProcessError:
    version = "git not available"
except FileNotFoundError:
    version = "git not available"

if n % 4 != 0:
    raise SystemExit("good luck!")
if n % 8 != 4:
    raise SystemExit("not implemented")

print(f'{n=}')

# array encoding -- do not change
nn = n // 4
nm = 4  # number of blocks
na = nm * nn  # length of array
nn2 = (nn-1) // 2

if 'segment_sums' not in globals():
    segment_sums = None

fixed_sums = segment_sums is not None
if fixed_sums:
    assert sum(i*i for i in segment_sums) == n
    print(f"{segment_sums=}")
    num_ones = torch.tensor([(segment_sums[j]+nn)//2 for j in range(nm)], dtype=torch.int8, device=device)
else:
    num_ones = None

hparams_list = ['n', 'segment_sums', 'n_layer', 'n_embd', 'n_head', 'stacking', 'sample_size', 'training_size', 'learning_rate', 'max_iterations', 'training_steps', 'training_batch_size', 'num_improve', 'weight_decay', 'version', 'random_seed', 'sample_batch_size', 'score_batch_size', 'test_set_size', 'gen_decay', 'temperature']

import ast
# hparams can be updated in command line
for param in hparams_list:
    parser.add_argument(f"--{param}")
args = parser.parse_args()
debugging = args.debug
for param in hparams_list:
    val = getattr(args, param)
    if val is not None:
        globals()[param] = ast.literal_eval(val)

# special cases: coupled default values
if getattr(args, "sample_size") and not getattr(args, "training_size"):
    training_size = sample_size//20
#if getattr(args, "n_embd") and not getattr(args, "n_embd2"):  # done in logger.py now to avoid sweep issue
#    n_embd2 = 4*n_embd


hparams = {name: globals().get(name) for name in hparams_list}

is_sweep = any(isinstance(v, list) for v in hparams.values())

if is_sweep:
    if resume:
        raise SystemExit("resume not supported with sweeps")
    sweep_config = {
        "method": "grid",
        "parameters": {
            k: { ("values" if isinstance(v, list) else "value"): v }
            for k, v in hparams.items()
            }
        }


class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Automatically computed values
        if isinstance(self.stacking, int):
            self.block_size = nm * ((nn-1)//self.stacking+1)  # n//stacking  only works if stacking | n
            self.vocab_size = 1 << self.stacking  # vocab_size is all the possible characters
        if isinstance(self.n_embd, int) and not hasattr(self,'n_embd2'):
            self.n_embd2 = 4 * self.n_embd
    def update(self):
        if is_sweep:
            import wandb
            self.__init__(**wandb.config)
            wandb.config.block_size = self.block_size
            wandb.config.vocab_size = self.vocab_size
            wandb.config.n_embd2 = self.n_embd2

config = ModelConfig(**hparams)


# symmetries
from itertools import permutations

# Prepare automorphisms
aut = torch.tensor([i for i in range(1, nn) if math.gcd(i, nn) == 1], device=device)

# Prepare permutations
perms = torch.tensor(list(p for p in permutations(range(nm)) if not fixed_sums or tuple(segment_sums[i] for i in p) == segment_sums), dtype=torch.long, device=device)
#print("order of symmetry: ", rndmod.prod().item())

def rotate(array0):
    B = array0.shape[0]
    array0 = array0.to(device=device).view(B, nm, nn)
    # --- random parameters per batch ---
    perm_idx = torch.randint(len(perms), (B,), device=device)
    a_idx    = torch.randint(len(aut), (B,), device=device)      # automorphism
    flips    = torch.randint(2, (B, nm), device=device) * 2 - 1  # ±1
    shifts   = torch.randint(nn, (B, nm), device=device)         # translation
    # --- combined affine action on Z/nnZ ---
    base = torch.arange(nn, device=device)                       # 0..nn-1
    a = aut[a_idx].unsqueeze(1)                                  # (1,1)
    coeff = a * flips                                            # ±a  (B,nm)
    shift_idx = (coeff.unsqueeze(-1) * base + shifts.unsqueeze(-1)) % nn  # (B,nm,nn)
    # Apply combined index transformation
    array = torch.gather(array0, 2, shift_idx)
    # --- block permutation per batch ---
    perm = perms[perm_idx]
    # array = array[torch.arange(B)[:, None], perm]
    perm_expanded = perm.unsqueeze(-1).expand(B, nm, nn)
    array = torch.gather(array, 1, perm_expanded)
    if not fixed_sums:
        signs = torch.randint(2, (B, nm), device=device, dtype=torch.int8) * 2 - 1  # ±1
        # --- independent overall signs ---
        array *= signs.unsqueeze(-1)
    return array

# obsolete: only one scoring function implemented
# score_function = 'fft log determinant'

real_dtype = torch.float32
complex_dtype = torch.complex64

cst = 1 / math.sqrt(n)
def fft(m):
    return cst * torch.fft.rfft(m.view(-1, nm, nn), dim=2)  # cst there for accuracy
@torch.inference_mode()
def score_fft_int(ff):
    s = -2*torch.log(ff)
    return s[:,0]+2*s[:,1:].sum(dim=1)
def score_fft(f):  # score in terms of precomputed fft f (b, nm, nn2+1)
    return score_fft_int(torch.view_as_real(f).square().sum(dim=(1,3)))  # sum over nm copies, over real/imag
def score(m):
    return score_fft(fft(m))
def randomise_score_weights():
    global score_weights
    with torch.random.fork_rng():
        score_weights = torch.rand(nn2+1, dtype=real_dtype, device=device)
def random_score_fft_int(ff):
    s = -torch.log(ff) + ff - 1
    return (s*score_weights).sum(dim=1)
def random_score_fft(f):
    return random_score_fft_int(torch.view_as_real(f).square().sum(dim=(1,3)))  # sum over nm copies, over real/imag
def random_score(m):
    return random_score_fft(fft(m))

eps = 2e-5  # scores are heavily discretised so can be made large
