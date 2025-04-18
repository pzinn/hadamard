if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

import os
# import torch
import time

# hadamard matrix parameters
nn = 20  # size of basic block
n = 4 * nn  # size of matrix
print(f'{n=}')

# string encoding
stacking = 10  # preferably a divisor of nn

# scoring
score_function = 'fft log determinant'
# score_function = 'quartic'
# score_function = 'one'

# training parameters
sample_size = 100000
training_size = sample_size//10  # must be > test_set_size
learning_rate = 2e-3
sample_batch_size = sample_size//4  # for sampling. preferably a divisor of sample_size
score_batch_size = 2*sample_size  # for scoring/improving. one should have sample_batch_size < score_batch_size
training_batch_size = 256  # for training. much smaller, obviously
weight_decay = 0.01
max_iterations = 20
training_steps = 200000  # will be adjusted dynamically (to be less than that)
num_improve = 5  # number of times data get improved per generation
test_set_size = 1024  # must be less than training_size, no more than 10% ideally
num_workers = 3  # for cpu parallelisation


# transformer parameters
n_layer = 4
n_embd = 64
n_head = 4
stacking = 10

# array encoding -- do not change
nm = 4  # number of blocks
na = nm * nn  # length of array


class ModelConfig:
    def __init__(self, n_layer=4, n_embd=64, n_head=4, stacking=10):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.stacking = stacking
        # Automatically computed values
        # string_length = n//stacking # only works if stacking | n
        # string_length = 1 + (n-1) // stacking # +1, -1 in case stacking doesn't divide n
        string_length = (1+(nn-1)//stacking)*nm  # more padding
        self.block_size = string_length + 1  # block_size : <START> token followed by string
        nchars = 1 << stacking
        self.vocab_size = nchars + 1  # vocab_size is all the possible characters and special 0 token


config = ModelConfig(n_layer, n_embd, n_head, stacking)

resume = False  # whether to resume a previous run
# if True, obviously, Hadamard parameters must be the same
# as well as transformer parameters (including stacking) unless resume_training = False
# training parameters can be different though
# also, for now resume is not compatible with sweep
# resume = True
if resume:
    pass
    # provide work_dir manually, default is latest
    # provide gen, default is latest

skip_first_training = False  # only meaningful if resume: start by sampling from existing model rather than training. leave False if unsure

resume_training = True  # whether to use previous model (not just previous data). True is a lot faster, False might be more accurate (?) leave True if unsure

random_seed = int(time.time())

device = 'cuda'  # device to use for compute, examples: cpu|cuda|cuda:2|mps

logging = 'wandb'
# logging = 'tensorboard'


import argparse
parser = argparse.ArgumentParser(description="Script with logging levels")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()
debugging = args.debug
