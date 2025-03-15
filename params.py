from dataclasses import dataclass
import datetime
import os

###########INITIAL-PARAMETERS###########

# hadamard matrix parameters
nn = 12 # size of matrix = 4*nn
n = 4 * nn
# encoding
stacking = 4 # preferably a divisor of nn

# training parameters
sample_size = 50000
training_size = sample_size//2
max_steps = 20000 # first few steps have more
learning_rate = 1e-3
sample_batch_size=sample_size//5 # for sampling. preferably a divisor of sample_size
score_batch_size=sample_size//4 # for scoring/improving. one should have sample_batch_size < score_batch_size
training_batch_size=32 # for training. much smaller, obviously
weight_decay=0.01
max_iterations = 100

# transformer parameters
@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 32
    n_head: int = 4

nchars = 1<<stacking
#string_length = n//stacking # only works if stacking | n
#string_length = 1 + (n-1) // stacking # +1, -1 in case stacking doesn't divide n
string_length = (1+(nn-1)//stacking)*4 # more padding

config = ModelConfig(vocab_size=nchars+1, block_size=string_length+1)
# vocab_size is all the possible characters and special 0 token
# block_size : <START> token followed by string

resume=False # whether to resume a previous run
#resume=True
if resume:
    # provide work_dir manually
    work_dir = "./training/140/5/2025-03-15-08-58-09_150000_256/"
    k = 0
    # obviously, transformer parameters must be the same (and Hadamard parameters including stacking)
    # training parameters can be different though
else:
    # make directory
    date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    work_dir = f'./training/{n}/{stacking}/{date}_{sample_size}_{config.n_embd}/'
    os.makedirs(work_dir, exist_ok=True)
    k = 0
try:
    os.unlink("latest")
except FileNotFoundError:
    pass
os.symlink(work_dir,"latest")

# header of stats file
stats_file = work_dir + 'stats.txt'
with open(stats_file, 'a') as file:
  file.write(f'n={n}\n{sample_size=}\n{training_size=}\n{learning_rate=}\n{config=}\n{max_iterations=}\n{stacking=}\n{max_steps=}\n{training_batch_size=}\n')

device='cuda' #device to use for compute, examples: cpu|cuda|cuda:2|mps

