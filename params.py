if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# hadamard matrix parameters
n = 140  # size of matrix

# the parameters below are sweepable: use values, or lists for a sweep

# scoring
score_function = 'fft log determinant'
# score_function = 'quartic'
# score_function = 'one'

# training parameters
sample_size = 400_000
training_size = sample_size//10  # must be > test_set_size
learning_rate = 2e-3
training_batch_size = 1024  # for training. much smaller, obviously
weight_decay = 0.01
max_iterations = 30
training_steps = 100_000  # will be adjusted dynamically (to be less than that)
num_improve = 1  # number of times data get improved per generation. only used by improve2

# transformer parameters
n_layer = 4
n_embd = 64
n_embd2 = 4*n_embd  # default choice
n_head = 4
stacking = 7  # [5,6,7,8,9,10]  # preferably a divisor of nn
gen_decay = .01 # [0., .025, .05, .075, .1, .15, .2]
temperature = 1. # [.5, .75, 1, 1.25, 1.5, 1.75, 2]

# less important parameters
sample_batch_size = sample_size//10  # for sampling. must be a divisor of sample_size
score_batch_size = None  # for scoring/improving. None means no batching
test_set_size = 1024  # must be less than training_size, no more than 10% ideally
num_workers = 6  # for cpu parallelisation

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
random_seed = int(time.time())  # 1746533706

device = 'cuda'  # device to use for compute, examples: cpu|cuda|cuda:2|mps

logging = 'wandb'  # '' | 'tensorboard' | 'wandb'
logging_mode = 'online'  # 'online' | 'offline' -- for wandb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--bignum", action="store_true", help="Enable debug logging")  # PZJ: what is this for?

import subprocess
try:
    version = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stderr=subprocess.DEVNULL
    ).strip().decode()
except subprocess.CalledProcessError:
    version = "git not available"
except FileNotFoundError:
    version = "git not available"

import ast

hparams_list = ['n', 'n_layer', 'n_embd', 'n_embd2', 'n_head', 'stacking', 'sample_size', 'training_size', 'learning_rate', 'max_iterations', 'training_steps', 'training_batch_size', 'score_function', 'num_improve', 'weight_decay', 'version', 'random_seed', 'sample_batch_size', 'score_batch_size', 'test_set_size', 'num_workers', 'gen_decay', 'temperature']

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
if getattr(args,"sample_size") and not getattr(args,"training_size"):
    training_size = sample_size//20
if getattr(args,"n_embd") and not getattr(args,"n_embd2"):
    n_embd2 = 4*n_embd
if getattr(args,"sample_size") and not getattr(args,"sample_batch_size"):
    sample_batch_size = sample_size//10


hparams = {name: globals().get(name) for name in hparams_list}

is_sweep = any(isinstance(v, (list, tuple)) for v in hparams.values())

if is_sweep:
    if resume:
        raise SystemExit("resume not supported with sweeps")
    sweep_config = {
        "method": "grid",
        "parameters": {
            k: {"values": list(v)} if isinstance(v, (list, tuple)) else {"value": v}
            for k, v in hparams.items()
            }
        }

if n % 4 != 0:
    raise SystemExit("good luck!")

print(f'{n=}')

# array encoding -- do not change
nn = n // 4
nm = 4  # number of blocks
na = nm * nn  # length of array
nn2 = (nn-1) // 2

class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Automatically computed values
        if isinstance(self.stacking, int):
            self.block_size = nm * ((nn-1)//self.stacking+1)  # n//stacking  only works if stacking | n
            self.vocab_size = 1 << self.stacking  # vocab_size is all the possible characters
    def update(self):
        if is_sweep:
            import wandb
            self.__init__(**wandb.config)
            wandb.config.block_size = self.block_size
            wandb.config.vocab_size = self.vocab_size


config = ModelConfig(**hparams)

# symmetries
from itertools import permutations
import torch
import math

# Prepare automorphisms
aut = [ i for i in range(1,nn) if math.gcd(i,nn) == 1 ]
aut_inds = torch.tensor([[(i*j)%nn for j in range(nn)] for i in aut])

# Prepare permutations -- note that these tensor are on cpu, if rotate used on gpu this needs to be changed
perms = torch.tensor(list(p for p in permutations(range(nm))), dtype=torch.long)
rndmod = torch.tensor([len(perms), len(aut), 2*nn, 2*nn, 2*nn, 2*nn, 2, 2, 2, 2], dtype=torch.int64)
nrnd = rndmod.shape
print("order of symmetry: ", rndmod.prod().item())

def rotate(array0):
    arrayx = torch.empty_like(array0)
    array0 = array0.view(-1,nm,nn)
    array = arrayx.view(-1,nm,nn)  # can do batches too (not currently used)
    rnd = torch.remainder(torch.empty(nrnd, dtype=torch.int64).random_(), rndmod)
    # automorphisms
    array.copy_(array0[:,:,aut_inds[rnd[1].item()]])
    # symmetry: random permute
    array.copy_(array[:,perms[rnd[0]]])
    # symmetry: random rotation/flip
    for j in range(nm):
        array[:,j] = torch.roll(array[:,j] if rnd[j+2] < nn else torch.flip(array[:,j], (1,)), shifts=rnd[j+2].item(), dims=1)
    # symmetry: random signs
    array.mul_((rnd[6:10]*2-1).unsqueeze(1))
    #
    return arrayx

