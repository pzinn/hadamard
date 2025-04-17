if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

from dataclasses import dataclass  # , asdict
import datetime
import os
# import torch
import glob
import re
import time
import math

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
max_iterations = 100
training_steps = 200000  # will be adjusted dynamically (to be less than that)
num_improve = 5  # number of times data get improved per generation
test_set_size = 1024  # must be less than training_size, no more than 10% ideally
num_workers = 3  # for cpu parallelisation


# transformer parameters
@dataclass
class ModelConfig:
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4
    # these are computed automatically
    block_size: int = None  # length of the input sequences of integers
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    stacking: int = None  # copied for convenience


# array encoding -- do not change
nm = 4  # number of blocks
na = nm * nn  # length of array

nchars = 1 << stacking
# string_length = n//stacking # only works if stacking | n
# string_length = 1 + (n-1) // stacking # +1, -1 in case stacking doesn't divide n
string_length = (1+(nn-1)//stacking)*nm  # more padding

config = ModelConfig(vocab_size=nchars+1, block_size=string_length+1, stacking=stacking)
# vocab_size is all the possible characters and special 0 token
# block_size : <START> token followed by string

logging = 'wandb'

# helper function
def find_latest_gen():
    # Get all filenames matching the pattern
    files = glob.glob(work_dir+"GEN-*.txt")
    # Extract the numerical part using regex
    indices = [int(re.search(r"GEN-(\d{2})\.txt", f).group(1)) for f in files if re.search(r"GEN-(\d{2})\.txt", f)]
    return max(indices) if indices else 0  # Return max index, or 0 if no files found


resume = False  # whether to resume a previous run
# resume = True
if resume:
    # provide work_dir manually
    work_dir = os.readlink("latest")
    gen = find_latest_gen()  # generation to pick up from. leave as is for latest, otherwise specify explicitly
    # obviously, Hadamard parameters must be the same
    # as well as transformer parameters (including stacking) unless resume_training = False
    # training parameters can be different though
    skip_first_training = False  # start by sampling from existing model rather than training. leave False if unsure
else:
    # make directory
    date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    work_dir = f'training/{n}/{stacking}/{date}_{sample_size}_{training_size}/'
    os.makedirs(work_dir, exist_ok=True)
    gen = 0
    skip_first_training = False
if not work_dir.endswith('/'):
    work_dir += '/'  # add trailing /
if not work_dir.endswith('latest/'):
    try:
        os.unlink("latest")
    except FileNotFoundError:
        pass
    os.symlink(work_dir, "latest")

resume_training = True  # whether to use previous model (not just previous data). True is a lot faster, False might be more accurate (?) leave True if unsure

random_seed = int(time.time())

stats_file = work_dir + 'stats.txt'  # where to save logs
hada_file = work_dir + 'hada.txt'  # where to save Hadamard matrices

device = 'cuda'  # device to use for compute, examples: cpu|cuda|cuda:2|mps

# logging
# header of stats file + hparams
norm = 1/(math.log(2)*stacking)  # renormalise loss so it starts at 1
import subprocess
version = subprocess.check_output(["git", "show", "-s", "--pretty='%D %h'"]).strip().decode()
hparam_list = ['n', 'sample_size', 'training_size', 'learning_rate', 'config', 'max_iterations', 'stacking', 'training_steps', 'training_batch_size', 'score_function', 'num_improve', 'version', 'random_seed']
with open(stats_file, 'a') as file:
    file.writelines(f"{name}={globals().get(name)!r}\n" for name in hparam_list)
# logging = 'tensorboard'
if logging == 'tensorboard':
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=work_dir)
    layout = {"combined": {"loss": ["Multiline", ["Loss/train", "Loss/test"]],
                           "score": ["Multline", ["Score/sample", "Score/improved", "Score/selected"]],
                           "zero_score": ["Multline", ["Zero_score/sample", "Zero_score/improved", "Zero_score/selected"]],
                           }
              }
    writer.add_custom_scalars(layout)
    def record_loss(loss, step, name):
        writer.add_scalar("Loss/"+name, norm*loss, step)
        writer.flush()
        print(f"{name} {loss=:.6f}", end='\t')
    def record_score(prefix, mean_score, nh, gen):
        writer.add_scalar("Score/"+prefix, mean_score, gen)
        writer.add_scalar("Zero_score/"+prefix, nh, gen)
        writer.flush()
    # unfortunately below is buggy
    """
    hparam_list.remove('config') # need to treat separately <sigh>
    hparam_dict = {name: globals().get(name) for name in hparam_list}
    hparam_dict.update(asdict(config))
    #print(hparam_dict)
    writer.add_hparams(hparam_dict,{},run_name='./')
    """
elif logging == 'wandb':
    import wandb
    wandb.init(entity='pzinn-the-university-of-melbourne', project='sekrit', name=work_dir, config=config, resume=resume, id=re.sub('/', '', work_dir))
    def record_loss(loss, step, name):
        wandb.log({"step": step, "loss/"+name+"/"+str(gen): norm*loss}, commit=name == 'test')  # hacky
        print(f"{name} {loss=:.6f}", end='\t')
    def record_score(prefix, mean_score, nh):
        wandb.log({"score/"+prefix: mean_score, "zero score/"+prefix: nh, "gen": gen})

import argparse
parser = argparse.ArgumentParser(description="Script with logging levels")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()
debugging = args.debug
