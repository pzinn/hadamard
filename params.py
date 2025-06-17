if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# hadamard matrix parameters
nn = 35  # size of basic block
n = 4 * nn  # size of matrix
print(f'{n=}')

# the parameters below are sweepable: use values, or lists for a sweep

# scoring
score_function = 'fft log determinant'
# score_function = 'quartic'
# score_function = 'one'

# training parameters
sample_size = 1_000_000
training_size = sample_size//20  # must be > test_set_size
learning_rate = 2e-3
training_batch_size = 1024  # for training. much smaller, obviously
weight_decay = 0.01
max_iterations = 30
training_steps = 200000  # will be adjusted dynamically (to be less than that)
num_improve = 5  # number of times data get improved per generation

# transformer parameters
n_layer = 6
n_embd = 128
n_embd2 = 4*n_embd  # default choice
n_head = 4
stacking = 7  # [5,6,7,8,9,10]  # preferably a divisor of nn

# less important parameters
sample_batch_size = sample_size//10  # for sampling. must be a divisor of sample_size
score_batch_size = None  # 2*sample_size  # for scoring/improving. None means no batching
test_set_size = 1024  # must be less than training_size, no more than 10% ideally
num_workers = 3  # for cpu parallelisation

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

# array encoding -- do not change
nm = 4  # number of blocks
na = nm * nn  # length of array


import time
random_seed = int(time.time())  # 1746533706

device = 'cuda'  # device to use for compute, examples: cpu|cuda|cuda:2|mps

logging = 'wandb'
# logging = 'tensorboard'
# logging = ''
logging_mode = 'offline' # 'online' | 'offline'

# GW 10/6: don't like argparse in called file, commenting out for now.
#import argparse
#parser = argparse.ArgumentParser(description="Script with logging levels")

#args = parser.parse_args()
#debugging = args.debug


# version = subprocess.check_output(["git", "show", "-s", "--pretty='%D %h'"]).strip().decode()
import subprocess

try:
    version = subprocess.check_output(
        ["git", "show", "-s", "--pretty=%D %h"], stderr=subprocess.DEVNULL
    ).strip().decode()
except subprocess.CalledProcessError:
    version = "git not available"
except FileNotFoundError:
    version = "git not available"



