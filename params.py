import torch
import math

if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# hadamard matrix parameters
n = 172  # size of matrix
# segment_sums = (1, 3, 3, 13)  # sum of squares must be n. must be a tuple (not a list!)

# the parameters below are sweepable: use values, or lists for a sweep

# training parameters
sample_size = 1_000_000
training_size = sample_size//20
learning_rate = 1e-3
training_batch_size = 1024  # for training. much smaller, obviously
weight_decay = 0.01
max_iterations = 30
training_steps = 150_000  # for gen 0. automatically decreases with gen
num_improve = 1  # number of times data get improved per generation beyond a quick local search pass

# transformer parameters
n_layer = 4
n_embd = 128
# n_embd2 = 4*n_embd  # default choice; only include if *not* default choice (can't be in hparams_list because of potential sweep issue)
n_head = 4
stacking = 7  # [5,6,7,8,9,10]  # preferably a divisor of nn
left_pad_stacks = False
transformer_uses_score = False
temperature = .6  # [.5, .75, 1, 1.25, 1.5, 1.75, 2]
temperature_delta = .02


# less important parameters
gen_decay = 0.0
sample_batch_size = 100_000  # for sampling; on some older GPUs this may need to stay below 65536
score_batch_size = None  # for scoring/improving. None means no batching

resume = False  # True | False, whether to resume a previous run
# if True, obviously, Hadamard parameters must be the same
# as well as transformer parameters (including stacking) unless resume_training = False
# training parameters can be different though
# also, for now resume is not compatible with sweep
if resume:
    # provide work_dir manually, default is latest
    # provide gen, default is latest
    pass

skip_first_training = False  # only meaningful if resume: start by sampling from existing model rather than training. leave False if unsure
skip_first_improve = resume  # leave as is unless you know what you're doing
resume_training = True  # whether to use previous model (not just previous data). True is a lot faster, False might be more accurate (?) leave True if unsure

test_score = False  # for debugging purposes, test whether randomisation of arrays (rotation) and other transformations preserves score


import time
random_seed = int(time.time())

logging = 'wandb'  # '' | 'tensorboard' | 'wandb'
logging_mode = 'online'  # 'online' | 'offline' -- for wandb

eps = 2e-5  # score accuracy. scores are heavily discretised so can be made fairly large

device = 'cuda'  # device to use for compute, examples: cpu|cuda|cuda:2|mps
# anything below this line shouldn't be changed
if device.startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit(f"{device=} but CUDA is not available")
if device == 'mps' and not torch.backends.mps.is_available():
    raise SystemExit(f"{device=} but MPS is not available")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
verbose = False

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

if 'segment_sums' not in globals():
    segment_sums = None

hparams_list = ['n', 'segment_sums', 'n_layer', 'n_embd', 'n_head', 'stacking', 'left_pad_stacks', 'transformer_uses_score', 'sample_size', 'training_size', 'learning_rate', 'max_iterations', 'training_steps', 'training_batch_size', 'num_improve', 'weight_decay', 'version', 'random_seed', 'sample_batch_size', 'score_batch_size', 'gen_decay', 'temperature', 'temperature_delta']

import ast
# hparams can be updated in command line
for param in hparams_list:
    parser.add_argument(f"--{param}")
sweep_config = None


class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Automatically computed values
        if isinstance(self.stacking, int):
            self.block_size = nm * ((nn-1)//self.stacking+1)  # n//stacking  only works if stacking | n
            self.vocab_size = 1 << self.stacking  # vocab_size is all the possible characters
        if isinstance(self.n_embd, int) and not hasattr(self, 'n_embd2'):
            self.n_embd2 = 4 * self.n_embd
    def update(self):
        if is_sweep:
            import wandb
            self.__init__(**wandb.config)
            wandb.config.block_size = self.block_size
            wandb.config.vocab_size = self.vocab_size
            wandb.config.n_embd2 = self.n_embd2

def compute_derived():
    global nn, nm, na, nn2
    global fixed_sums, num_ones
    global hparams, is_sweep, sweep_config, config
    global aut, perms, cst
    if not isinstance(n, int) or isinstance(n, bool):
        raise SystemExit("n must be an integer; sweeps over n are not supported")
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
    fixed_sums = segment_sums is not None
    if fixed_sums:
        assert len(segment_sums) == nm
        assert sum(i*i for i in segment_sums) == n
        print(f"{segment_sums=}")
        num_ones = torch.tensor([(segment_sums[j]+nn)//2 for j in range(nm)], dtype=torch.int8, device=device)
    else:
        num_ones = None
    hparams = {name: globals().get(name) for name in hparams_list}
    is_sweep = any(isinstance(v, list) for v in hparams.values())
    if is_sweep:
        if resume:
            raise SystemExit("resume not supported with sweeps")
        sweep_config = {
            "method": "grid",
            "parameters": {
                k: {("values" if isinstance(v, list) else "value"): v}
                for k, v in hparams.items()
                }
            }
    else:
        sweep_config = None
    config = ModelConfig(**hparams)
    # Prepare automorphisms / permutations
    aut = torch.tensor([i for i in range(1, nn) if math.gcd(i, nn) == 1], device=device)
    perms = torch.tensor(
        list(p for p in permutations(range(nm)) if not fixed_sums or tuple(segment_sums[i] for i in p) == segment_sums),
        dtype=torch.long,
        device=device,
    )
    # scoring normalisation constant
    cst = 1 / math.sqrt(n)

def init_from_argv(argv=None):
    global verbose, training_size, symmetry_ctx
    args = parser.parse_args(argv)
    verbose = args.verbose
    for param in hparams_list:
        val = getattr(args, param)
        if val is not None:
            globals()[param] = ast.literal_eval(val)
    # special cases: coupled default values
    if getattr(args, "sample_size") and not getattr(args, "training_size"):
        training_size = sample_size//20
    compute_derived()
    from symmetry import build_context
    symmetry_ctx = build_context()

# symmetries
from itertools import permutations

#print("order of symmetry: ", rndmod.prod().item())

real_dtype = torch.float32
complex_dtype = torch.complex64

def fft(m):
    m = m.view(-1, nm, nn).to(dtype=real_dtype)
    return cst * torch.fft.rfft(m, dim=2)  # cst there for accuracy
@torch.inference_mode()
def score_fft_int(ff):
    s = 2*(ff-1-torch.log(ff))
    return 2*s.sum(dim=1)-s[:,0]
def score_fft(f):  # score in terms of precomputed fft f (b, nm, nn2+1)
    return score_fft_int(torch.view_as_real(f).square().sum(dim=(1, 3)))  # sum over nm copies, over real/imag
def score(m):
    return score_fft(fft(m))
