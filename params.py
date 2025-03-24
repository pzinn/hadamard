if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

from dataclasses import dataclass, asdict
import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import subprocess
version=subprocess.check_output(["git", "describe", "--always"]).strip().decode()
import glob
import re

###########INITIAL-PARAMETERS###########

# hadamard matrix parameters
nn = 20 # size of matrix = 4*nn
n = 4 * nn

# encoding
stacking = 5 # preferably a divisor of nn

# scoring
score_function = 'log determinant'
#score_function = 'quartic'

# training parameters
sample_size = 100000
training_size = sample_size//2 # must be > test_set_size
learning_rate = 5e-4
sample_batch_size=sample_size//4 # for sampling. preferably a divisor of sample_size
score_batch_size=sample_size # for scoring/improving. one should have sample_batch_size < score_batch_size
training_batch_size=128 # for training. much smaller, obviously
weight_decay=0.01
max_iterations = 100
training_steps = 300000 # will be adjusted dynamically (to be less than that)
#training_steps = (2*nn*training_size)//training_batch_size # 2 epochs??

# transformer parameters
@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4

nchars = 1<<stacking
#string_length = n//stacking # only works if stacking | n
#string_length = 1 + (n-1) // stacking # +1, -1 in case stacking doesn't divide n
string_length = (1+(nn-1)//stacking)*4 # more padding

config = ModelConfig(vocab_size=nchars+1, block_size=string_length+1)
# vocab_size is all the possible characters and special 0 token
# block_size : <START> token followed by string

#helper function
def find_latest_gen():
    # Get all filenames matching the pattern
    files = glob.glob(work_dir+"GEN-*.txt")
    # Extract the numerical part using regex
    indices = [int(re.search(r"GEN-(\d{2})\.txt", f).group(1)) for f in files if re.search(r"GEN-(\d{2})\.txt", f)]
    return max(indices) if indices else 0  # Return max index, or 0 if no files found

resume=False # whether to resume a previous run
#resume=True
if resume:
    # provide work_dir manually
    work_dir = "./training/80/5/testing/" # don't forget the trailing /
    gen = find_latest_gen() # generation to pick up from. leave as is for latest, otherwise specify explicitly
    # obviously, transformer parameters must be the same (and Hadamard parameters including stacking)
    # training parameters can be different though
    skip_first_training=False # start by sampling from existing model rather than training. leave False if unsure
else:
    # make directory
    date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    work_dir = f'./training/{n}/{stacking}/{date}_{sample_size}_{config.n_embd}/'
    os.makedirs(work_dir, exist_ok=True)
    gen = 0
    skip_first_training=False
try:
    os.unlink("latest")
except FileNotFoundError:
    pass
os.symlink(work_dir,"latest")

resume_training = True # whether to use previous model (not just previous data). True is a lot faster, False might be more accurate (?) leave True if unsure

stats_file = work_dir + 'stats.txt' # where to save logs
hada_file = work_dir + 'hada.txt' # where to save Hadamard matrices

device='cuda' #device to use for compute, examples: cpu|cuda|cuda:2|mps

writer = SummaryWriter(log_dir=work_dir)
layout = { "combined" : { "loss" : [ "Multiline", ["Loss/train","Loss/test"]],
                          "score": [ "Multline", ["Score/sample","Score/improved","Score/selected"]],
                          "zero_score":[ "Multline", ["Zero_score/sample","Zero_score/improved","Zero_score/selected"]],
                         }
          }
writer.add_custom_scalars(layout)

# header of stats file + hparams
hparam_list = ['n','sample_size','learning_rate','config','max_iterations','stacking','training_steps','training_batch_size','score_function','version']
with open(stats_file, 'a') as file:
    file.writelines(f"{name}={globals().get(name)!r}\n" for name in hparam_list)
"""
hparam_list.remove('config') # need to treat separately <sigh>
hparam_dict = {name: globals().get(name) for name in hparam_list}
hparam_dict.update(asdict(config))
#print(hparam_dict)
writer.add_hparams(hparam_dict,{},run_name='./')
"""
