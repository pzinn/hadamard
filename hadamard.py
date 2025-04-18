#!/usr/bin/env python
# coding: utf-8

import random
import math
import numpy as np
import torch
import heapq
from itertools import islice
import params  # for gen, work_dir
from params import n, na, nn, device, training_size, score_function, score_batch_size, sample_size, sample_batch_size, resume_training, training_steps, learning_rate, max_iterations, num_improve, random_seed, is_sweep, debugging, training_batch_size, config
import transformer
# logging/debugging
import logger
import sys
from collections import Counter
from timeit import default_timer as timer  # to measure exec time
import glob
import re
import datetime
import os

eps = 1e-5  # scores are heavily discretised so can be made large

# torch functions
torch.cuda.set_device(0)  # Use GPU 0
torch.cuda.empty_cache()  # Free memory before large computation
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


def generate_random_array():
    return tuple((2 * torch.randint(2, (na,)) - 1).tolist())

# MAIN-DEFINITIONS #


def best_from(arrays_dict):
    # preserves ordering
    items = arrays_dict.items()
    smallest_keys = {k for k, _ in heapq.nsmallest(training_size, items, key=lambda item: item[1][0])}  # heapq requires no nan
    return {k: v for k, v in items if k in smallest_keys or v[0] < score_threshold + eps}  # always keep H-matrices
    # doesn't
    # return dict(heapq.nsmallest(training_size,arrays_dict.items(),key=lambda item: item[1]))
    # some other discarded alternatives
    """
    arrays_items = list(arrays_dict.items())
    arrays_items.sort(key=lambda item: item[1])
    return dict(arrays_items[:training_size])
    """
    """
    scores, _ = zip(*arrays_dict.values())
    threshold = np.partition(np.array(scores), training_size)[training_size]
    return {k: v for k, v in arrays_dict.items() if v[0] <= threshold}
    """


def write_arrays(file_path, arrays):
    with open(file_path, 'w') as file:
        for s in arrays:
            file.write("".join(map(lambda x: "+" if x == 1 else "-", s)) + "\n")


# for keeping track of stats
def record_stats(arrays_dict, prefix=""):
    if len(arrays_dict) == 0:
        return
    arrays_items = arrays_dict.items()
    arrays, values = zip(*arrays_items)
    scores, gens = zip(*values)

    # compute autocorrelation by MC
    mc_size = 1000
    s = 0
    for _ in range(mc_size):
        a1 = random.choice(arrays)
        a2 = random.choice(arrays)
        s += sum(x*y for x, y in zip(a1, a2))
    s /= (mc_size * n)
    print(f"Correlation: {s}")

    # now scores
    scores = normalise(np.array(scores, dtype=float))
    # if debugging:
    #     print(f'Score tally: {dict(zip(*np.unique(np.round(scores, decimals=5), return_counts=True)))}')

    min_score = np.min(scores)
    print(f"Min score: {min_score}")

    mean_score = np.mean(scores)
    print(f"Mean score: {mean_score}")

    max_score = np.max(scores)
    print(f"Max score: {max_score}")

    # tally=Counter([val[1] for val in vals])
    tally = dict(Counter(gens).most_common())
    print(f"Gen tally: {tally}")

    hada_dict = {arrays[i]: gens[i] for i in range(len(arrays_dict)) if scores[i] < eps}
    nh = len(hada_dict) / len(arrays_dict)
    print(f"Hadamard ratio: {nh}")
    # hada_tally = Counter(hada_dict.values())
    hada_tally = dict(Counter(hada_dict.values()).most_common())
    print(f"Hadamard gen tally: {hada_tally}")

    if hasattr(record_stats, "total_hada_dict"):
        hada_dict.update(record_stats.total_hada_dict)
    record_stats.total_hada_dict = hada_dict
    print(f"Total number of Hadamard: {len(record_stats.total_hada_dict)}")

    with open(stats_file, 'a') as file:
        if not record_stats.has_run:
            record_stats.has_run = True
            file.write(f"{'gen':>3} {'':<10}: {'min score':>10} {'mean score':>10} {'max score':>10} {'autocorrel':>10} {'H-ratio':>10} {'H-number':>10} tally / H-tally\n")
        file.write(f"{params.gen:>3} {prefix:<10}: {min_score:10.6f} {mean_score:10.6f} {max_score:10.6f} {s:10.6f} {nh:10.6f} {len(record_stats.total_hada_dict):>10} {tally} {hada_tally}\n")

    write_arrays(hada_file, record_stats.total_hada_dict.keys())

    if prefix:
        logger.record_score(prefix, mean_score, nh)
        # if params.gen == max_iterations:  # only do histograms for last gen
        logger.record_histogram(prefix, scores, gens)


if score_function != 'fft log determinant':
    # Generate row indices for circulant
    indices = torch.arange(nn, device=device).repeat(nn, 1)  # Shape: (nn, nn)
    shifts = torch.arange(nn, device=device).unsqueeze(1)  # Shape: (nn, 1)
    rolled_indices = (indices - shifts) % nn  # Shape: (nn, nn)
    V = torch.arange(2*n, device=device).reshape(8, nn)  # Shape: (8,nn) -- the original array and its negation, for convenience
    X = V[:, rolled_indices]
    X[3] = torch.flip(X[3], dims=[1])
    X[7] = torch.flip(X[7], dims=[1])
    full_indices = torch.cat([
        torch.cat((X[0], X[1], X[2], X[3]), dim=1),
        torch.cat((X[5], X[0], X[7], X[2]), dim=1),
        torch.cat((X[6], X[3], X[0], X[5]), dim=1),
        torch.cat((X[7], X[6], X[1], X[0]), dim=1)
    ], dim=0)
    # create the n x n block circulant matrix out of the n bits
    def block_circulant(x):
        return torch.cat((x, -x), dim=1)[..., full_indices]

# for device=='cuda', float is pretty much compulsory
# float16 may be faster but may lead to accuracy issues
if score_function == 'log determinant':
    score_type = torch.float32  # slogdet needs float32
    score_threshold = - n/2 * math.log(n)
    score_normalisation = 1
    def score(m):
        return -torch.linalg.slogdet(block_circulant(m))[1]
elif score_function == 'fft log determinant':
    score_type = torch.float32
    # score_threshold = - n/4 * math.log(n)
    score_threshold = 0  # see renormalisation of m below
    score_normalisation = .5
    cst = 1 / math.sqrt(n)
    def score(m):
        f = cst * torch.fft.rfft(m.view(-1, 4, nn), dim=2)  # cst improves accuracy
        # we do separately real pieces for accuracy reasons
        s = - torch.log(torch.real(f[:, :, 0].pow(2).sum(dim=1)))
        if nn % 2 == 0:
            s -= torch.log(torch.real(f[:, :, nn//2].pow(2).sum(dim=1)))
            f = f[:, :, 1:-1]
        else:
            f = f[:, :, 1:]
        ff = f[:, :3, :].pow(2).sum(dim=1)
        f = f * f.conj()
        ff = ff * ff.conj()
        s -= torch.log(torch.real(ff+f[:, 3]*(2*f.sum(dim=1)-f[:, 3]))).sum(dim=1)
        return s
elif score_function == 'quartic':
    score_type = torch.float16
    score_threshold = n**1.5
    score_normalisation = 2*math.sqrt(n)
    def score(m):
        C = block_circulant(m)
        return torch.linalg.matrix_norm(torch.matmul(C, torch.transpose(C, 1, 2)))
elif score_function == 'one':
    score_type = torch.float16
    score_threshold = 0
    score_normalisation = n
    Idn = n * torch.eye(n, device=device, dtype=score_type)
    def score(m):
        C = block_circulant(m)
        return torch.linalg.matrix_norm(torch.matmul(C, torch.transpose(C, 1, 2))-Idn, ord=1)
else:
    raise Exception('unknown score_function')


def normalise(sc):
    return (sc-score_threshold)/score_normalisation


# scoring. technically we don't need this since the scores could be computed when improving;
# but useful for logging/stats
def batch_score(arrays):
    torch.cuda.empty_cache()  # Free memory
    torch.set_float32_matmul_precision('highest')
    arrays_tensor = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor
    scores = score(arrays_tensor)  # Compute scores in parallel
    return {x: (s, params.gen) for x, s in zip(arrays, scores.tolist()) if math.isfinite(s)}  # Convert back to dict


def subbatch_score(arrays):  # same but in batches of score_batch_size
    total_size = len(arrays)
    updated_dict = {}
    for start in range(0, total_size, score_batch_size):
        end = min(start + score_batch_size, total_size)
        updated_dict.update(batch_score(arrays[start:end]))
    return updated_dict


# parameters of step 2 (can be adjusted)
# max_k = int(math.sqrt(n))
n_attempts = na * num_improve
p = .3/math.sqrt(na)


def batch_improve(arrays_items):
    arrays, values = zip(*arrays_items)
    scores, gens = zip(*values)
    torch.cuda.empty_cache()  # Free memory
    arrays_tensor = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor and float
    # scores = score(arrays_tensor)  # Recompute scores in parallel
    scores = torch.tensor(scores, dtype=score_type, device=device)  # Convert to tensor and float
    # step 1: this is the analogue of my old "simple_search2"
    for j in range(num_improve):
        if debugging:
            cnt = torch.tensor(0, device=device, dtype=torch.int64)
        print(f"1({j})", end=''); sys.stdout.flush()
        for i in range(na):
            arrays_tensor[:, i] *= -1  # Flip only the i-th bit
            # Compute new scores for all batch elements in parallel
            new_scores = score(arrays_tensor)
            # Identify which flips improved the score
            mask = new_scores < scores  # True where improvement happens
            if debugging:
                cnt += torch.sum(mask)
            # Apply successful bit flips
            arrays_tensor[~mask, i] *= -1  # Only revert for elements where no improvement
            scores[mask] = new_scores[mask]  # Update scores accordingly
        if debugging:
            print(f' improve success rate: {cnt/len(arrays_items)}')
    # step 2
    if debugging:
        cnt.zero_()
    print('2', end=''); sys.stdout.flush()
    for i in range(n_attempts):
        a = torch.randint(na, ()).item()
        b = torch.randint(na, ()).item()
        if a > b:
            a, b = b, a
        # Flip selected bits for all arrays in batch
        arrays_tensor[:, a:b+1] *= -1
        # Compute new scores after flipping
        new_scores = score(arrays_tensor)
        # Identify improvements
        mask = new_scores < scores
        if debugging:
            cnt += torch.sum(mask)
        # Revert changes for arrays where score did not improve
        arrays_tensor[~mask, a:b+1] *= -1
        # Update scores where improvements occurred
        scores[mask] = new_scores[mask]
        """
        # step 3
        batch_size = arrays_tensor.shape[0]
        arrays_view = arrays_tensor.view(batch_size,nm,nn)  # reshape, forcing view
        base = torch.arange(nn, device=device).repeat(batch_size, 1)
        for j in range(2):
            shifts = torch.zeros(batch_size, dtype=torch.int64, device=device)
            for i in range(1,nn):
                arrays_view[:,j]=torch.roll(arrays_view[:,j], shifts=1, dims=1)
                new_scores = score(arrays_tensor)
                mask = new_scores < scores  # True where improvement happens
                if debugging:
                    cnt3 += torch.sum(mask)
                shifts[mask] = i
                scores[mask] = new_scores[mask]  # Update scores accordingly
            # slightly annoying: re-roll according to shifts
            rolled_indices = (base - 1 - shifts.unsqueeze(1)) % nn
            arrays_view[:, j] = arrays_view[: ,j].gather(1, rolled_indices)
        """
    if debugging:
        print(f' improve success rate: {cnt/len(arrays_items)}')
    # step 3: this is the analogue of my old "simple_search3" except it doesn't stop at first success
    if debugging:
        cnt.zero_()
    print('3', end=''); sys.stdout.flush()
    for i in range(n_attempts):
        # Choose k unique bits to flip, same for entire batch
        # flip_indices = torch.randperm(n, device=device)[:random.randint(2,max_k)]
        # variation
        flip_indices = torch.rand(na, device=device) < p
        flip_indices[i % na] = True  # just because
        # Flip selected bits for all arrays in batch
        arrays_tensor[:, flip_indices] *= -1
        # Compute new scores after flipping k bits
        new_scores = score(arrays_tensor)
        # Identify improvements
        mask = new_scores < scores
        if debugging:
            cnt += torch.sum(mask)
            # Revert changes for arrays where score did not improve
            # arrays_tensor[(~mask).nonzero(), flip_indices] *= -1  # ugly... especially cause most of the mask will be False
        arrays_tensor[:, flip_indices] *= -1
        arrays_tensor[mask.nonzero(), flip_indices] *= -1  # this might be faster?
        # arrays_tensor[~mask, flip_indices] *= -1  # that doesn't work, left to remember: can't mix masks and tensors of indices
        # Update scores where improvements occurred
        scores[mask] = new_scores[mask]
    if debugging:
        print(f' improve success rate: {cnt/len(arrays_items)}')
    else:
        print('')
    # Convert back to dict
    # return {tuple(map(int,x.cpu().numpy())): (s.item(),g) for x, s, g in zip(arrays_tensor, scores, gens) if torch.isfinite(s)}
    # return {tuple(1 if b>0 else -1 for b in x): (s.item(),g) for x, s, g in zip(arrays_tensor.cpu(), scores.cpu(), gens) if torch.isfinite(s)}
    # return {tuple(x): (s,g) for x, s, g in zip(arrays_tensor.int().tolist(), scores.tolist(), gens) if math.isfinite(s)}
    return {tuple(x): (s, g) for x, s, g in zip(torch.where(arrays_tensor > 0, 1, -1).tolist(), scores.tolist(), gens) if math.isfinite(s)}


def subbatch_improve(arrays_items):
    updated_dict = {}
    it = iter(arrays_items)  # Convert dictionary to iterator
    while True:
        batch = list(islice(it, score_batch_size))  # Take next batch_size items
        if not batch:
            break
        updated_dict.update(batch_improve(batch))
    return updated_dict


# helper function
def find_latest_gen():
    # Get all filenames matching the pattern
    files = glob.glob(params.work_dir+"GEN-*.txt")
    # Extract the numerical part using regex
    indices = [int(re.search(r"GEN-(\d{2})\.txt", f).group(1)) for f in files if re.search(r"GEN-(\d{2})\.txt", f)]
    return max(indices) if indices else 0  # Return max index, or 0 if no files found


def main():
    global stats_file, hada_file, version
    # directory, gen
    if params.resume:
        # existing directory, default is latest
        if not hasattr(params, "work_dir"):
            params.work_dir = os.readlink("latest")
        # initialise gen if necessary
        if not hasattr(params, "gen"):
            params.gen = find_latest_gen()
    else:
        # make directory
        date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        params.work_dir = f'training/{n}/{config.stacking}/{date}_{sample_size}_{training_size}/'
        os.makedirs(params.work_dir, exist_ok=True)
        #
        params.gen = 0
    if not params.work_dir.endswith('/'):
        params.work_dir += '/'  # add trailing /
    try:
        os.unlink("latest")
    except FileNotFoundError:
        pass
    os.symlink(params.work_dir, "latest")
    stats_file = params.work_dir + 'stats.txt'  # where to save logs
    hada_file = params.work_dir + 'hada.txt'  # where to save Hadamard matrices

    # logging: text stats file + fancy (tensorboard or wandb)
    # header of stats file
    import subprocess
    version = subprocess.check_output(["git", "show", "-s", "--pretty='%D %h'"]).strip().decode()
    hparam_list = ['n', 'sample_size', 'training_size', 'learning_rate', 'config', 'max_iterations', 'training_steps', 'training_batch_size', 'score_function', 'num_improve', 'version', 'random_seed']  # TODO failing now REDO
    with open(stats_file, 'a') as file:
        file.writelines(f"{name}={globals().get(name)!r}\n" for name in hparam_list)
    record_stats.has_run = False  # we could leave it undefined, but not in case of sweep
    # start fancy logging
    logger.init_logging()

    # initialise transformer
    if is_sweep:
        config.__init__(wandb.config.n_layer, wandb.config.n_embd, wandb.config.n_head, wandb.config.stacking)  # hack TODO clean up
    transformer.init_model()

    # STEP 0

    # initial info
    if params.resume:
        # use existing sample
        init_sample = params.work_dir + f'GEN-{params.gen:02d}.txt'
        try:
            with open(init_sample, 'r') as f:
                arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for line in f]
            print(f'***Loading initial sample from {init_sample}***')
        except FileNotFoundError:
            arrays = []
    else:
        # generate initial sample
        print('***Generating initial sample***')
        arrays = list(generate_random_array() for _ in range(sample_size))

    arrays_dict = subbatch_score(arrays)
    record_stats(arrays_dict, prefix="sample" if not params.resume else "")  # who knows where the data come from if resuming

    # MAIN-LOOP #

    while True:
        if params.resume:
            params.resume = False
        else:
            # improve existing data, write to GEN-(gen)
            start_timer = timer()
            print('\n***Improving***')
            arrays_dict = subbatch_improve(arrays_dict.items())
            if debugging:
                print(f"improving: {timer() - start_timer}")
            record_stats(arrays_dict, "improved")
            print('\n***Selecting***')
            arrays_dict = best_from(arrays_dict)
            record_stats(arrays_dict, "selected")
            arrays = arrays_dict.keys()
            write_arrays(params.work_dir + f'GEN-{params.gen:02d}.txt', arrays)
        if params.gen == max_iterations:
            break
        if params.skip_first_training:
            params.skip_first_training = False
        else:
            # train on GEN-gen
            print(f"\n***Training on GEN-{params.gen:02d}***")
            coeff = 1 if params.gen == 0 or not resume_training else .01+.99*math.sqrt(sum(1 for v in arrays_dict.values() if v[1] == params.gen)/len(arrays_dict))  # decrease training steps depending on how much new stuff added
            if debugging:
                print(f"{coeff=}")
            max_steps = int(training_steps*coeff)
            eval_freq = int(500*coeff)
            start_timer = timer()
            save_step = transformer.train(arrays, resume=resume_training, max_steps=max_steps, eval_freq=eval_freq, learning_rate=learning_rate*coeff)
            if debugging:
                print(f"training: {timer() - start_timer}")
            with open(stats_file, 'a') as file:
                file.write(f'training {save_step=}\n')
        # sample from model to get new data
        print(f"\n***Sampling from transformer trained on GEN-{params.gen:02d}***")
        params.gen += 1
        # to avoid oom we do it in batches of sample_batch_size -- is it clear that samples are independent?
        start_timer = timer()
        new_arrays_dict = {}
        for start in range(0, sample_size, sample_batch_size):
            print('*', end=''); sys.stdout.flush()
            b = min(sample_batch_size, sample_size-start)
            new_arrays = transformer.sample(num_samples=b)
            new_arrays = [x for x in new_arrays if x not in arrays_dict and x not in new_arrays_dict]  # remove duplicates
            new_arrays_dict.update(batch_score(new_arrays))
        record_stats(new_arrays_dict, prefix="sample")  # do we produce similar scores as training data?
        new_arrays_dict.update(arrays_dict)  # old ones last to avoid overwriting old gen during improving
        arrays_dict = new_arrays_dict
        if debugging:
            print(f"sampling: {timer() - start_timer}")


if is_sweep:
    import wandb
    wandb.agent(logger.sweep_id, function=main)
else:
    main()
