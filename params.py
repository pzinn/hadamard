if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

# hadamard matrix parameters
nn = 20  # size of basic block
n = 4 * nn  # size of matrix
print(f'{n=}')

# the parameters below are sweepable: use values, or lists for a sweep

# scoring
score_function = 'fft log determinant'
# score_function = 'quartic'
# score_function = 'one'

# training parameters
sample_size = 100000
training_size = 10000  # sample_size//10  # must be > test_set_size
learning_rate = 2e-3
training_batch_size = 256  # for training. much smaller, obviously
weight_decay = 0.01
max_iterations = 20
training_steps = 200000  # will be adjusted dynamically (to be less than that)
num_improve = 5  # number of times data get improved per generation

# transformer parameters
n_layer = 4
n_embd = 64
n_head = 4
stacking = 5  # [5,6,7,8,9,10]  # preferably a divisor of nn

# the parameters below are not sweepable
sample_batch_size = sample_size//4  # for sampling. preferably a divisor of sample_size
score_batch_size = 2*sample_size  # for scoring/improving. one should have sample_batch_size < score_batch_size
test_set_size = 1024  # must be less than training_size, no more than 10% ideally
num_workers = 3  # for cpu parallelisation

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

# array encoding -- do not change
nm = 4  # number of blocks
na = nm * nn  # length of array


import time
random_seed = int(time.time())

device = 'cuda'  # device to use for compute, examples: cpu|cuda|cuda:2|mps

logging = 'wandb'
# logging = 'tensorboard'


import argparse
parser = argparse.ArgumentParser(description="Script with logging levels")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()
debugging = args.debug


import subprocess
version = subprocess.check_output(["git", "show", "-s", "--pretty='%D %h'"]).strip().decode()


hparams_list = ['n', 'n_layer', 'n_embd', 'n_head', 'stacking', 'sample_size', 'training_size', 'learning_rate', 'max_iterations', 'training_steps', 'training_batch_size', 'score_function', 'num_improve', 'weight_decay', 'version', 'random_seed']

hparams = {name: globals().get(name) for name in hparams_list}

is_sweep = any(isinstance(v, (list, tuple)) for v in hparams.values())

if is_sweep:
    sweep_config = {
        "method": "grid",
        "parameters": {
            k: {"values": list(v)} if isinstance(v, (list, tuple)) else {"value": v}
            for k, v in hparams.items()
            }
        }


class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Automatically computed values
        if isinstance(self.stacking, int):
            # string_length = n//stacking  # only works if stacking | n
            string_length = (1+(nn-1)//self.stacking)*nm  # including padding if stacking doesn't divide nn
            self.block_size = string_length + 1  # block_size : <START> token followed by string
            nchars = 1 << self.stacking
            self.vocab_size = nchars + 1  # vocab_size is all the possible characters and special 0 token
    def update(self):
        if is_sweep:
            import wandb
            self.__init__(**wandb.config)


config = ModelConfig(**hparams)
