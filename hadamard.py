#!/usr/bin/env python
# coding: utf-8

import random
import math
import numpy as np
import torch
import heapq
from itertools import islice
# debugging stuff
import sys
from collections import Counter
from timeit import default_timer as timer # to measure exec time
import logging
import argparse
parser = argparse.ArgumentParser(description="Script with logging levels")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(levelname)s: %(message)s"
)

#
import transformer
from params import *

eps = 1e-6 # is that too big? need to think

def generate_random_array():
    return tuple(random.choices([-1,1],k=n))

# conversion string <-> matrix
def char_to_sign(c, i):
    return 2 * ((c >> i) & 1) - 1

def string_to_array(s): # really, tuple to tuple by now!
    return tuple(char_to_sign(s[i // stacking] - 1, i % stacking) for i in range(n))

# really, tuple to tuple by now!
quarter_string_length = string_length//4
array_to_string = (lambda M: tuple(
    1 + sum((1 if M[i * stacking + bit] == 1 else 0) << bit for bit in range(stacking)) # encoding 1
    #1 + sum((1 if M[i + bit * string_length] == 1 else 0) << bit for bit in range(stacking)) # encoding 2
            for i in range(string_length)
)) if nn % stacking == 0 else (lambda M: tuple( # or do I want n % stacking == 0 ? would be kind of OK
    #1 + sum((1 if M[i * stacking + bit] == 1 else 0) << bit for bit in range(stacking if i<string_length-1 else n%stacking)) # encoding 1
    1 + sum((1 if M[j * nn + i * stacking + bit] == 1 else 0) << bit
            for bit in range(stacking if i<quarter_string_length-1 else nn%stacking)) # encoding 1, more padding
    for j in range(4)
    for i in range(quarter_string_length)
))

# cyclically rotate a tuple
def rot(i,a):
    #    return tuple(map(int,np.roll(np.array(a).reshape(4,nn),i,axis=1).ravel()))
    return a[i:nn]+a[:i]+a[nn+i:2*nn]+a[nn:nn+i]+a[2*nn+i:3*nn]+a[2*nn:2*nn+i]+a[3*nn+i:]+a[3*nn:3*nn+i]

########### MAIN-DEFINITIONS ###########

def best_from(arrays_dict):
    #heapq is slightly more efficient but requires no nan
    # preserves ordering
    items = arrays_dict.items()
    smallest_keys = {k for k, _ in heapq.nsmallest(training_size, items , key=lambda item: item[1])}
    return {k: v for k, v in items if k in smallest_keys}
    #doesn't
    #return dict(heapq.nsmallest(training_size,arrays_dict.items(),key=lambda item: item[1][0]))
    """
    arrays_items = list(arrays_dict.items())
    arrays_items.sort(key=lambda item: item[1][0])
    return dict(arrays_items[:training_size])
    """

def write_arrays(file_path,arrays):
    with open(file_path, 'w') as file:
        for s in arrays:
            file.write("".join(map(lambda x:"+" if x==1 else "-", s))+"\n")

# for keeping track of stats
def record_stats(arrays_dict,prefix=""):
    # compute autocorrelation by MC
    arrays = list(arrays_dict.keys())
    mc_size = 1000
    s = 0
    for _ in range(mc_size):
      a1 = random.choice(arrays)
      a2 = random.choice(arrays)
      s += sum(x*y for x,y in zip(a1,a2))
    s/=(mc_size*n)
    print(f"Correlation: {s}")
    
    # now scores
    vals = arrays_dict.values()
    scores = np.array([val[0] for val in vals], dtype=float)
    scores = scores[scores < np.inf] # get rid of infinities

    min_score = np.min(scores)
    print(f"Min score: {min_score}")

    mean_score = np.mean(scores)
    print(f"Mean score: {mean_score}")

    max_score = np.max(scores)
    print(f"Max score: {max_score}")

    #tally=Counter([val[1] for val in vals])
    tally=dict(Counter([val[1] for val in vals]).most_common())
    print(f"Gen tally: {tally}")

    hada_dict = { k: v[1] for k, v in arrays_dict.items() if v[0] < eps }
    nh = len(hada_dict) / len(arrays_dict)
    print(f"Hadamard ratio: {nh}")
    #hada_tally = Counter(hada_dict.values())
    hada_tally = dict(Counter(hada_dict.values()).most_common())
    print(f"Hadamard gen tally: {hada_tally}")
    
    with open(stats_file, 'a') as file:
        if not hasattr(record_stats, "has_run"):
            record_stats.has_run = True
            file.write(f"{'gen':<3} {'':<10}: {'min score':>10} {'mean score':>10} {'max score':>10} {'autocorrel':>10} {'H-ratio':>10} tally / H-tally\n")
        file.write(f"{k:<3} {prefix:<10}: {min_score:10.6f} {mean_score:10.6f} {max_score:10.6f} {s:10.6f} {nh:10.6f} {tally} {hada_tally}\n")

    write_arrays(hada_file, hada_dict.keys())

    if prefix:
        writer.add_scalar("Score/"+prefix, mean_score, k)
        writer.add_scalar("Ratio/"+prefix, nh, k)

# torch functions
torch.cuda.set_device(0)  # Use GPU 0
torch.cuda.empty_cache()  # Free memory before large computation

# Generate row indices for circulant
indices = torch.arange(nn).repeat(nn, 1)  # Shape: (nn, nn)
shifts = torch.arange(nn).unsqueeze(1)  # Shape: (nn, 1)
rolled_indices = (indices - shifts) % nn  # Shape: (nn, nn)
def circulant_torch(m):
    return m[..., rolled_indices]

def upblock_torch(x):
    batch_size = x.shape[0]
    _aa, _bb, _cc, _dd = x.reshape(batch_size,4,nn).unbind(dim=1)
    
    A = circulant_torch(_aa)
    B = circulant_torch(_bb)
    C = torch.flip(circulant_torch(_cc), dims=[2])
    D = circulant_torch(_dd)
    
    return torch.cat([
        torch.cat((A, B, C, D), dim=2),
        torch.cat((-B, A, -D, C), dim=2),
        torch.cat((-C, D, A, -B), dim=2),
        torch.cat((-D, -C, B, A), dim=2)
    ], dim=1)

cst = n/2 * math.log(n)
def score_torch(m):
    """Compute -log determinant of circulant matrix using PyTorch."""
    C = upblock_torch(m)  # Generate circulant matrix on GPU
    _, logdet = torch.linalg.slogdet(C)  # Compute sign and log determinant
    return cst-logdet  # Negate to match original function

def batch_score(k,arrays):
    torch.cuda.empty_cache()  # Free memory
    arrays_tensor = torch.tensor(arrays, dtype=torch.float32, device=device)  # Convert to tensor
    scores = score_torch(arrays_tensor)  # Compute scores in parallel
    return {x: (s.item(),k) for x, s in zip(arrays, scores)}  # Convert back to dict

def subbatch_score(k,arrays): # same but in batches of score_batch_size
    total_size = len(arrays)
    updated_dict = {}
    for start in range(0, total_size, score_batch_size):
        end = min(start + score_batch_size, total_size)
        updated_dict.update(batch_score(k,arrays[start:end]))
    return updated_dict

# parameters of step 2 (can be adjusted)
max_k = int(math.sqrt(n))
n_attempts = n
def batch_improve(arrays_items):
    arrays, values = zip(*arrays_items)  # Keys are tuples, values are scores
    scores, gens = zip(*values)
    torch.cuda.empty_cache()  # Free memory
    arrays_tensor = torch.tensor(arrays, dtype=torch.float32, device=device)  # Convert to tensor
    #scores = score_torch(arrays_tensor)  # Compute scores in parallel
    scores = torch.tensor(scores, dtype=torch.float32, device=device)  # Convert to tensor
    # step 1: this is the analogue of my old "simple_search2"
    for i in range(n):
        print(f"1-{i} ",end=''); sys.stdout.flush()
        arrays_tensor[:, i] *= -1  # Flip only the i-th bit
        # Compute new scores for all batch elements in parallel
        new_scores = score_torch(arrays_tensor)
        # Identify which flips improved the score
        mask = new_scores < scores  # True where improvement happens
        # Apply successful bit flips
        arrays_tensor[~mask, i] *= -1  # Only revert for elements where no improvement
        scores[mask] = new_scores[mask]  # Update scores accordingly
    # step 2: this is the analogue of my old "simple_search3" except it doesn't stop at first success
    for i in range(n_attempts):
        print(f"2-{i} ",end=''); sys.stdout.flush()
        # Choose k unique bits to flip, same for entire batch
        k = random.randint(2,max_k)
        flip_indices = torch.randperm(n, device=device)[:k]
        # Flip selected bits for all arrays in batch
        arrays_tensor[:, flip_indices] *= -1
        # Compute new scores after flipping k bits
        new_scores = score_torch(arrays_tensor)
        # Identify improvements
        mask = new_scores < scores
        # Revert changes for arrays where score did not improve
        # arrays_tensor[(~mask).nonzero(), flip_indices] *= -1 # ugly... especially cause most of the mask will be False
        arrays_tensor[:, flip_indices] *= -1
        arrays_tensor[mask.nonzero(), flip_indices] *= -1 # this might be faster?
        #arrays_tensor[~mask, flip_indices] *= -1 # that doesn't work, left to remember: can't mix masks and tensors of indices
        # Update scores where improvements occurred
        scores[mask] = new_scores[mask]
    # Convert back to dict
    return {tuple(map(int,x.cpu().numpy())): (s.item(),g) for x, s, g in zip(arrays_tensor, scores, gens) if torch.isfinite(s)}

def subbatch_improve(arrays_items):
    total_size = len(arrays_items)
    updated_dict = {}
    it = iter(arrays_items)  # Convert dictionary to iterator
    while True:
        batch = list(islice(it, score_batch_size))  # Take next batch_size items
        if not batch:
            break
        updated_dict.update(batch_improve(batch))
    return updated_dict

# #### STEP 0

# initial info
stats_file = work_dir + 'stats.txt'
hada_file = work_dir + 'hada.txt'

if resume:
    # use existing sample
    init_sample = work_dir + f'/GEN-{k:02d}.txt'
    print(f'***Loading initial sample from {init_sample}***')
    arrays = list(map(lambda x: x.strip(),open(init_sample, 'r').read().splitlines()))
    arrays = list(map(lambda x: tuple(1 if c=="+" else -1 for c in x),arrays))
else:
    # generate initial sample
    print(f'***Generating initial sample***')
    arrays = list(generate_random_array() for _ in range(sample_size))

arrays_dict = subbatch_score(k,arrays)
record_stats(arrays_dict,prefix="sample")

test_set_size = 1+1000//nn

########### MAIN-LOOP ###########

while k<max_iterations:
    if resume:
        resume=False
    else:
        # compute GEN-(k)
        start=timer()
        print(f"\n***Improving***")
        arrays_dict = subbatch_improve(arrays_dict.items())
        logging.debug(f"improving: {timer() - start}")
        record_stats(arrays_dict,"improved")
        print(f"\n***Selecting***")
        arrays_dict = best_from(arrays_dict)
        record_stats(arrays_dict,"selected")
        arrays=arrays_dict.keys()
        write_arrays(work_dir + f'/GEN-{k:02d}.txt',arrays)
    if skip_first_training:
        skip_first_training=False
    else:
        # train on GEN-k
        print(f"\n***Training***\nTraining makemore on GEN-{k:02d}...")
        start=timer()
        arrays=list(arrays)
        random.shuffle(arrays)
        test_arrays=arrays[:test_set_size]
        train_arrays=arrays[test_set_size:]
        train_words = [array_to_string(rot(i,a)) for i in range(nn) for a in train_arrays] # added: rotation to increase training size
        test_words = [array_to_string(rot(i,a)) for i in range(nn) for a in test_arrays] # added: rotation to increase training size
        print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")
        logging.debug(f"splitting: {timer() - start}")
        max_steps = training_steps if k==0 or not resume_training else training_steps//5
        start=timer()
        transformer.train(train_words,test_words,resume=resume_training and (k>0 or resume),max_steps=max_steps,eval_freq=500)
        logging.debug(f"training: {timer() - start}")
    # sample from model to get GEN-(k+1)-a
    print(f"\n***Sampling from transformer trained on GEN-{k:02d}.txt")
    k+=1
    #to avoid oom we do it in batches of sample_batch_size -- is it clear that samples are independent?    
    start=timer()
    new_arrays_dict={}
    for start in range(0, sample_size, sample_batch_size):
        b = min(sample_batch_size, sample_size-start)
        new_strings = transformer.sample(num_samples=b,seed=start*11407)
        new_arrays = [x for x in ( string_to_array(str) for str in new_strings if len(str) == string_length ) if x not in arrays_dict and x not in new_arrays_dict] #throw out strings of incorrect length
        new_arrays_dict.update(batch_score(k,new_arrays))
    record_stats(new_arrays_dict,prefix="sample") # do we produce similar scores as training data?
    new_arrays_dict.update(arrays_dict) # old ones last to avoid overwriting old gen during improving
    arrays_dict=new_arrays_dict
    logging.debug(f"sampling: {timer() - start}")
    writer.flush()





