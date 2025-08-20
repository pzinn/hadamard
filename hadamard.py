#!/usr/bin/env python
# coding: utf-8

import random
import math
import numpy as np
import torch
import heapq
from itertools import islice
import params
from params import n, na, nm, nn, device, resume, resume_training, random_seed, is_sweep, debugging, config
import logger
import transformer
# logging/debugging
import sys
from collections import Counter
from timeit import default_timer as timer  # to measure exec time

eps = 1e-5  # scores are heavily discretised so can be made large


def generate_random_arrays(batch_size):
    return 2 * torch.randint(2, (batch_size, na), device=device, dtype=score_type) - 1

# MAIN-DEFINITIONS #


def best_from(arrays_dict):
    # preserves ordering
    items = arrays_dict.items()
    smallest_keys = {k for k, _ in heapq.nsmallest(config.training_size, items, key=lambda item: item[1][0])}  # heapq requires no nan
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


def fmt_array(s):
    return "".join("+" if x > 0 else "-" for x in s)

def write_arrays(file_path, arrays):
    with open(file_path, 'w') as file:
        for s in arrays:
            file.write(fmt_array(s) + "\n")


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
        a1 = np.array(random.choice(arrays)).reshape(nm,nn)
        a2 = np.array(random.choice(arrays)).reshape(nm,nn)
        fft_a1 = np.fft.fft(a1, axis=1)
        fft_a2 = np.fft.fft(a2, axis=1)
        corr = np.fft.ifft(fft_a1 * np.conj(fft_a2), axis=1).real
        s += np.max(np.sum(np.abs(corr), axis=0))
        # s += sum(x*y for x, y in zip(a1, a2))
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
    total_nh = len(record_stats.total_hada_dict)
    print(f"Total number of Hadamard: {total_nh}")
    if total_nh>0:
        print(f"Here's one: {fmt_array(next(iter(record_stats.total_hada_dict)))}")

    with open(logger.stats_file, 'a') as file:
        if not record_stats.has_run:
            record_stats.has_run = True
            file.write(f"{'gen':>3} {'':<10}: {'min score':>10} {'mean score':>10} {'max score':>10} {'autocorrel':>10} {'H-ratio':>10} {'H-number':>10} tally / H-tally\n")
        file.write(f"{params.gen:>3} {prefix:<10}: {min_score:10.6f} {mean_score:10.6f} {max_score:10.6f} {s:10.6f} {nh:10.6f} {len(record_stats.total_hada_dict):>10} {tally} {hada_tally}\n")

    write_arrays(logger.hada_file, record_stats.total_hada_dict.keys())

    if prefix:
        logger.record_scores(prefix, scores, gens, mean_score, nh)


def init_score_function():
    global score, normalise, score_type, score_threshold
    if config.score_function != 'fft log determinant':
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
    if config.score_function == 'log determinant':
        score_type = torch.float32  # slogdet needs float32
        score_threshold = - n/2 * math.log(n)
        score_normalisation = 1
        def score(m):
            return -torch.linalg.slogdet(block_circulant(m))[1]
    elif config.score_function == 'fft log determinant':
        score_type = torch.float32
        # score_threshold = - n/4 * math.log(n)
        score_threshold = 0  # see renormalisation of m below
        score_normalisation = .5
        cst = 1 / math.sqrt(n)
        def score(m):
            f = cst * torch.fft.rfft(m.view(-1, nm, nn), dim=2)  # cst there for accuracy
            # we do separately real pieces for accuracy reasons
            s = - torch.log(torch.real(f[:, :, 0].pow(2).sum(dim=1)))
            if nn % 2 == 0:
                s -= torch.log(torch.real(f[:, :, nn//2].pow(2).sum(dim=1)))
                f = f[:, :, 1:-1]
            else:
                f = f[:, :, 1:]
            ff = f[:, :3, :].pow(2).sum(dim=1)
            f.mul_(f.conj())  # f = f * f.conj()
            ff.mul_(ff.conj())  # ff = ff * ff.conj()
            s -= torch.log(torch.real(ff+f[:, 3]*(2*f.sum(dim=1)-f[:, 3]))).sum(dim=1)
            return s
    elif config.score_function == 'quartic':
        score_type = torch.float16
        score_threshold = n**1.5
        score_normalisation = 2*math.sqrt(n)
        def score(m):
            C = block_circulant(m)
            return torch.linalg.matrix_norm(torch.matmul(C, torch.transpose(C, 1, 2)))
    elif config.score_function == 'one':
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
# but useful for logging/stats. also generates random data at gen 0
def parallel_score(arrays):
    if isinstance(arrays,torch.Tensor):
        arrays_tensor = arrays
        arrays = [tuple(row) for row in torch.where(arrays_tensor > 0, 1, -1).tolist()]
    else:
        arrays_tensor = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor
    scores = score(arrays_tensor)  # Compute scores in parallel
    return {x: (s, params.gen) for x, s in zip(arrays, scores.tolist()) if math.isfinite(s)}  # Convert back to dict

def batch_generator(arrays):
    it = iter(arrays)
    while True:
        batch = list(islice(it, config.score_batch_size))
        if not batch:
            break
        yield batch

def random_batch_generator():
    n_full_batches = config.sample_size // config.score_batch_size
    remainder = config.sample_size % config.score_batch_size
    for _ in range(n_full_batches):
        yield generate_random_arrays(config.score_batch_size)
    if remainder:
        yield generate_random_arrays(remainder)

def batch_score(arrays):  # same as parallel_score but in batches of score_batch_size
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        return parallel_score(list(arrays) if arrays is not None else generate_random_arrays(config.sample_size))
    updated_dict = {}
    if arrays is None:
        batches = random_batch_generator()
    else:
        batches = batch_generator(arrays)
    for batch in batches:
        updated_dict.update(parallel_score(batch))
    return updated_dict


def parallel_improve(arrays_items,new_arrays_dict):
    p = 1/math.sqrt(na)  # what's the optimum value?
    arrays, values = zip(*arrays_items)
    scores, gens = zip(*values)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    arrays_tensor = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor and float
    # scores = score(arrays_tensor)  # Recompute scores in parallel
    scores = torch.tensor(scores, dtype=score_type, device=device)  # Convert to tensor and float
    for k in range(config.num_improve):
        # step 1: this is the analogue of my old "simple_search2"
        for j in range(config.num_improve):
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
        for i in range(na):  # used to be na * num_improve
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
        if debugging:
            print(f' improve success rate: {cnt/len(arrays_items)}')
        # update
        # new_arrays_dict.update({tuple(x): (s, g) for x, s, g in zip(torch.where(arrays_tensor > 0, 1, -1).tolist(), scores.tolist(), gens) if math.isfinite(s)})
        temp_arrays={tuple(x): (s, g) for x, s, g in zip(torch.where(arrays_tensor > 0, 1, -1).tolist(), scores.tolist(), gens) if math.isfinite(s)}
        if debugging:
            record_stats(temp_arrays, "improved")
        new_arrays_dict.update(temp_arrays)
        # select
        new_arrays_dict=best_from(new_arrays_dict)
        if k<config.num_improve-1:
            # step 3: this is the analogue of my old "simple_search3" except it doesn't stop at first success
            # Choose k unique bits to flip, same for entire batch
            # variation
            flip_indices = torch.rand(na, device=device) < p
            # Flip selected bits for all arrays in batch
            arrays_tensor[:, flip_indices] *= -1
            # Compute new scores after flipping k bits
            scores = score(arrays_tensor)
        if not debugging:
            print('')
    return new_arrays_dict

def batch_improve(arrays_dict,new_arrays_dict):
    if config.score_batch_size is None:
        return parallel_improve(arrays_dict.items(),new_arrays_dict)
    it = iter(arrays_dict.items())  # Convert dictionary to iterator
    while True:
        batch = list(islice(it, config.score_batch_size))  # Take next batch_size items
        if not batch:
            break
        new_arrays_dict = parallel_improve(batch,new_arrays_dict)
    return new_arrays_dict

def main():
    # logging: text stats file + fancy (tensorboard or wandb)
    logger.init_logging()
    record_stats.has_run = False  # we could leave it undefined, but not in case of sweep

    # scoring
    init_score_function()

    # initialise transformer
    transformer.init_model()

    # torch functions
    torch.manual_seed(random_seed)
    if device.startswith('cuda'):
        torch.cuda.set_device(0)  # Use GPU 0
        torch.cuda.empty_cache()  # Free memory before large computation
        torch.cuda.manual_seed_all(random_seed)

    # STEP 0

    # initial info
    if resume:
        # use existing sample
        init_sample = params.work_dir + f'GEN-{params.gen:02d}.txt'
        try:
            with open(init_sample, 'r') as f:
                arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for line in f]
            print(f'***Loading initial sample from {init_sample}***')
        except FileNotFoundError:
            arrays = []
    else:
        arrays = None

    arrays_dict = batch_score(arrays)
    record_stats(arrays_dict, prefix="sample" if not resume else "")  # who knows where the data come from if resuming

    # MAIN-LOOP #

    while True:
        if params.skip_first_improve:
            params.skip_first_improve = False
        else:
            # improve existing data, write to GEN-(gen)
            start_timer = timer()
            print('\n***Improving/selecting***')
            arrays_dict = batch_improve(arrays_dict,{})
            if debugging:
                print(f"improving: {timer() - start_timer}")
            record_stats(arrays_dict, "selected")
            arrays = arrays_dict.keys()
            write_arrays(params.work_dir + f'GEN-{params.gen:02d}.txt', arrays)
        if params.gen == config.max_iterations:
            break
        if params.skip_first_training:
            params.skip_first_training = False
        else:
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            # train on GEN-gen
            print(f"\n***Training on GEN-{params.gen:02d}***")
            coeff = 1 if params.gen == 0 or not resume_training else .01+.99*math.sqrt(sum(1 for v in arrays_dict.values() if v[1] == params.gen)/len(arrays_dict))  # decrease training steps depending on how much new stuff added
            # linear warmup with fixed base learning rate afterwards:
            def get_lr(step, warmup_steps=10000):
                return config.learning_rate * coeff * (.01+.99*step / warmup_steps if step < warmup_steps else 1)
            if debugging:
                print(f"{coeff=}")
            max_steps = int(config.training_steps*coeff)
            eval_freq = int(500*coeff)
            start_timer = timer()
            transformer.train(arrays, score=score if params.test_randomisation else None, max_steps=max_steps, eval_freq=eval_freq, lr_sched=get_lr)
            if debugging:
                print(f"training: {timer() - start_timer}")
        # sample from model to get new data
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        print(f"\n***Sampling from transformer trained on GEN-{params.gen:02d}***")
        params.gen += 1
        start_timer = timer()
        new_arrays = transformer.sample()
        new_arrays_dict = batch_score(new_arrays)
        record_stats(new_arrays_dict, prefix="sample")  # do we produce similar scores as training data?
        new_arrays_dict.update(arrays_dict)  # old ones last to avoid overwriting old gen (including during improving)
        arrays_dict = new_arrays_dict
        if debugging:
            print(f"sampling: {timer() - start_timer}")


if is_sweep:
    import wandb
    wandb.agent(logger.sweep_id, function=main)
else:
    main()
