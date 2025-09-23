#!/usr/bin/env python
# coding: utf-8

import random
import math
import numpy as np
import torch
import heapq
from itertools import islice
import params
from params import n, na, nn, nn2, device, resume, resume_training, random_seed, is_sweep, debugging, config
import logger
import transformer
# logging/debugging
import sys
from collections import Counter
from timeit import default_timer as timer  # to measure exec time

eps = 1e-5  # scores are heavily discretised so can be made large


import torch

"""
@torch.no_grad()
def rs_like_batch(n: int, B: int, dtype=torch.float32):
    m=1
    while (1<<m)<n:
        m+=1
    L = 1
    # Work in int32 to avoid overflow of intermediate sums
    P = torch.ones(B, 1, dtype=torch.int32, device=device)
    Q = torch.ones(B, 1, dtype=torch.int32, device=device)
    # Draw all σ_j ∈ {±1} for the whole batch in one go
    sigmas = (torch.randint(0, 2, (B, m), device=device,dtype=torch.int32) * 2 - 1)
    for j in range(m):
        s = sigmas[:, j].view(B, 1)  # shape [B,1]
        L2 = L << 1
        # Allocate next level
        Pn = torch.empty(B, L2, dtype=torch.int32, device=device)
        Qn = torch.empty_like(Pn)
        # Coeff recurrences (no padding/cat; just place blocks)
        Pn[:, :L]  = P
        Pn[:, L:]  = s * Q
        Qn[:, :L]  = P
        Qn[:, L:]  = -s * Q
        P, Q = Pn, Qn
        L = L2
    # Map coefficients to {±1}; treat zeros (very rare) as +1
    A = torch.sign(P).to(dtype)
    A[A == 0] = 1
    # crop
    starts = torch.randint(0, (1<<m) - n + 1, (B,1), device=device)
    rots   = torch.randint(0, n, (B,1), device=device)
    # Indices for cropped-then-rotated view:
    # (i,k) -> A[i, starts[i] + ((k - rots[i]) mod n)]
    base = torch.arange(n, device=device).view(1,-1)
    idx2d = starts + ((base - rots) % n)
    out = A.gather(dim=1, index=idx2d)     # [B, n], still ±1
    return out
"""

@torch.inference_mode()
def generate_random_arrays(batch_size):
    return 2 * torch.randint(2, (batch_size, na), device=device, dtype=score_type) - 1
    """
    # for now rs_like not diverse enough (check autocorrel)
    a=rs_like_batch(nn2,batch_size)
    b=rs_like_batch(nn2,batch_size)
    c=rs_like_batch(nn2,batch_size)
    d=rs_like_batch(nn,batch_size)
    return torch.cat((a,b,c,d),dim=1)
    """

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
    perms = params.perms.tolist()
    s = 0
    for _ in range(mc_size):
        a1 = np.array(random.choice(arrays))
        a2 = np.array(random.choice(arrays))
        a13 = a1[:3*nn2].reshape(3,nn2)
        a23 = a1[:3*nn2].reshape(3,nn2)
        a11 = a1[3*nn2:]
        a21 = a2[3*nn2:]
        fft_a1 = np.fft.fft(a11, axis=0)
        fft_a2 = np.fft.fft(a21, axis=0)
        corr1 = np.fft.ifft(fft_a1 * np.conj(fft_a2), axis=0).real
        corr2 = np.fft.ifft(fft_a1 * fft_a2, axis=0).real
        corr = np.stack([corr1,corr2],axis=0)
        corr = np.abs(corr)
        s += np.max(corr)
        ss = 0
        for p in perms:
            sss = np.sum(a13[p]*a23,axis=(0,1))
            if sss>ss:
                ss=sss
        s += ss
    s /= (mc_size * na)
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

cst = 1 / math.sqrt(n)

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
        def score(m0):
            # TEMP reduce to non sym case
            ones=torch.ones((m0.size(0),1),device=m0.device,dtype=score_type)
            m=torch.cat((m0[:,:nn2],ones,torch.flip(m0[:,:nn2],(1,)),
                         m0[:,nn2:2*nn2],ones,torch.flip(m0[:,nn2:2*nn2],(1,)),
                         m0[:,2*nn2:3*nn2],ones,torch.flip(m0[:,2*nn2:3*nn2],(1,)),
                         m0[:,3*nn2:]),dim=1)
            return -torch.linalg.slogdet(block_circulant(m))[1]
    elif config.score_function == 'fft log determinant':
        score_type = torch.float32
        # score_threshold = - n/4 * math.log(n)
        score_threshold = 0  # see renormalisation of m below
        score_normalisation = 1
        def score(m0):
            # reduce to non sym case but simplify due to phase alignment of first 3 fft
            nm=4
            ones=torch.ones((m0.size(0),1),device=m0.device,dtype=score_type)  # take out
            m=torch.cat((ones,m0[:,:nn2],torch.flip(m0[:,:nn2],(1,)),
                         ones,m0[:,nn2:2*nn2],torch.flip(m0[:,nn2:2*nn2],(1,)),
                         ones,m0[:,2*nn2:3*nn2],torch.flip(m0[:,2*nn2:3*nn2],(1,)),
                         m0[:,3*nn2:]),dim=1)
            f = cst * torch.fft.rfft(m.view(-1, nm, nn), dim=2)  # cst there for accuracy
            ff = torch.real(f*f.conj())
            s = torch.log(ff.sum(dim=1))
            return -2*s[:,0]-4*s[:,1:].sum(dim=1)
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

@torch.inference_mode()
def improve1(arrays_tensor,scores):  # used by parallel_improve: flip a single bit
    if debugging:
        cnt = torch.tensor(0, device=device, dtype=torch.int64)
    print(f"1", end=''); sys.stdout.flush()
    p = torch.randperm(na)  # should it be on device?
    for i in range(na):
        arrays_tensor[:, p[i]] *= -1  # Flip only the i-th bit
        # Compute new scores for all batch elements in parallel
        new_scores = score(arrays_tensor)
        # Identify which flips improved the score
        mask = new_scores < scores  # True where improvement happens
        if debugging:
            cnt += torch.sum(mask)
        # Apply successful bit flips
        arrays_tensor[~mask, p[i]] *= -1  # Only revert for elements where no improvement
        scores[mask] = new_scores[mask]  # Update scores accordingly
    if debugging:
        print(f' improve success rate: {cnt/arrays_tensor.shape[0]}')

"""
@torch.inference_mode()
def improve2(arrays_tensor,scores):  # used by parallel_improve: flip contiguous sequences of bits
    if debugging:
        cnt = torch.tensor(0, device=device, dtype=torch.int64)
    print('2', end=''); sys.stdout.flush()
    for i in range(na*config.num_improve//2):
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
        print(f' improve success rate: {cnt/arrays_tensor.shape[0]}')
"""

@torch.inference_mode()
def improve2(m0,scores):  # try to infer one of the 3 nn2 arrays from the rest
    # as in score
    nm=4
    ones=torch.ones((m0.size(0),1),device=m0.device,dtype=score_type)  # take out
    m=torch.cat((ones,m0[:,:nn2],torch.flip(m0[:,:nn2],(1,)),
                 ones,m0[:,nn2:2*nn2],torch.flip(m0[:,nn2:2*nn2],(1,)),
                 ones,m0[:,2*nn2:3*nn2],torch.flip(m0[:,2*nn2:3*nn2],(1,)),
                 m0[:,3*nn2:]),dim=1).view(-1,nm,nn)  # lazy
    f = cst * torch.fft.rfft(m, dim=2)  # cst there for accuracy
    ff = torch.real(f*f.conj())
    ff1 = ff[:,1:].sum(dim=1)  # for now do only one
    mask = torch.all(ff1<1,dim=1)
    print("possible",mask.sum())
    if mask.any():
        g = torch.sqrt(1-ff1[mask])
        h = torch.sign(torch.fft.irfft(g,n=nn))
        h = h * h[:,0:1]  # make first a plus
        #print(h)
        hf = cst * torch.fft.rfft(h,dim=1)
        hff = torch.real(hf*hf.conj())
        s = torch.log(ff1[mask]+hff)
        new_scores = -2*s[:,0]-4*s[:,1:].sum(dim=1)
        improved = new_scores < scores[mask]
        print("improved",improved.sum())
        scores[mask][improved]=new_scores[improved]
        m[mask][improved,0]=h[improved]
        m0[mask][improved,:nn2]=h[improved,1:nn2+1]

def mod_score(m):
    #return score(torch.tanh(m))
    return score(m) + torch.sum(m**2,dim=1)

# optimisation of improve3a
def improve3(x,steps=10000,lr=.01,mixed_precision=True):
    x.requires_grad_(True)
    scaler = torch.amp.GradScaler(device,enabled=mixed_precision)

    # opt = torch.optim.SGD([x], lr=lr)
    opt = torch.optim.AdamW([x], lr=lr)

    not_improved = None
    active_mask = torch.ones(x.shape[0], device=device, dtype=torch.bool)

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device,enabled=mixed_precision):
            scores = mod_score(x[active_mask])
            loss = scores.sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            x.clamp_(min=-1, max=1)  # projection
            if t==0:
                prev_scores = scores
            else:
                new_active_mask = active_mask.clone()
                improve = (scores - prev_scores[active_mask]) < -eps
                if not improve.any():
                    print(f"All rows converged. Stopping at step {t}.")
                    break
                new_active_mask[active_mask] = improve
                prev_scores[active_mask] = scores
                active_mask=new_active_mask

    return torch.where(x > 0, 1., -1.).detach()

"""
def improve3a(x,steps=1000,lr=.01,mixed_precision=True):
    x.requires_grad_(True)
    scaler = torch.amp.GradScaler(device,enabled=mixed_precision)

    # opt = torch.optim.SGD([x], lr=lr)
    opt = torch.optim.AdamW([x], lr=lr)

    prev_scores = None
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device,enabled=mixed_precision):
            scores = mod_score(x)
            loss = scores.sum()
            if prev_scores is not None:
                not_improved = (scores - prev_scores) > -eps
                if not_improved.all():
                    print(f"stop at {t}")
                    break
            prev_scores = scores.detach()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    return torch.where(x > 0, 1., -1.).detach()
"""

vec = torch.rand((nn,),device=device,dtype=torch.float32)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
def mysort(arrays_tensor):
    if params.test_randomisation:
        s1 = score(arrays_tensor)
    # 1st phase: permute the 3xnn2 parts
    B = arrays_tensor.shape[0]
    m=3
    a=arrays_tensor[:,:m*nn2].view(B,m,nn2)
    # start with identity permutation for each batch
    perm = torch.arange(m, device=device).expand(B, m).clone()
    # stable sort by last key first, then previous..., up to first column
    for k in range(nn2):
        key = a[:, :, k]                 # (B, m)
        key_in_curr_order = key.gather(1, perm)
        ordk = torch.argsort(key_in_curr_order, dim=1, stable=True)
        perm = perm.gather(1, ordk)
    # apply permutation to rows
    sorted_a = a.gather(1, perm.unsqueeze(-1).expand(-1, -1, nn2))
    a.copy_(sorted_a)
    # 2nd phase: cyclically permute/reflect/negate the remaining length nn part
    a=arrays_tensor[:,m*nn2:]
    fft_a = torch.fft.rfft(a, dim=1)  # use fft to quickly compute scalar product with some random vector for ordering
    sp_rot = torch.fft.irfft(fft_conj_vec[None, :] * fft_a, n=nn, dim=1)  # (B, nn)
    sp_rev = torch.fft.irfft(fft_vec[None, :] * fft_a, n=nn, dim=1)  # (B, nn)
    sps = torch.cat([sp_rot, sp_rev], dim=1)   # (B, 2 * nn)
    flat_idx = sps.abs().argmax(dim=1)         # (B,) over 2*nn options
    # Gather the chosen transform from the original 'a'
    signed_base = torch.where(flat_idx >= nn,-1,1).unsqueeze(1) * base.unsqueeze(0)
    idx = ( signed_base + flat_idx.unsqueeze(1)) % nn
    transformed = a.gather(1, idx)  # (B, nn)
    # negate if the chosen scalar product is > 0
    chosen_sps = sps.gather(1, flat_idx.unsqueeze(1)).squeeze(1)  # (B,)
    a.copy_(torch.where(chosen_sps > 0, -1, 1).unsqueeze(1) * transformed)
    if params.test_randomisation:
        s2 = score(arrays_tensor)
        if (s1-s2).abs().max() > 1e-5:
            raise RuntimeError("score not preserved by sort", s1, s2, (s1-s2).abs().max().item())

def parallel_improve(arrays_items,new_arrays_dict):
    arrays, values = zip(*arrays_items)
    scores, gens = zip(*values)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    arrays_tensor = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor and float
    # scores = score(arrays_tensor)  # Recompute scores in parallel
    scores = torch.tensor(scores, dtype=score_type, device=device)  # Convert to tensor and float
    improve2(arrays_tensor, scores)  # TEST
    # step A: demultiply data
    #arrays_tensor1=(0.2+0.4*torch.rand(arrays_tensor.shape,device=device))*arrays_tensor
    arrays_tensor1=(0.45 + 0.3*torch.rand((),device=device) + 0.1*torch.rand((1,na),device=device))*arrays_tensor
    arrays_tensor1 = improve3(arrays_tensor1)
    arrays_tensor2 = torch.stack((arrays_tensor,arrays_tensor1),dim=0)
    scores2 = torch.stack((scores,score(arrays_tensor1)),dim=0)
    arrays_tensor = arrays_tensor2.view(-1,na)
    scores = scores2.view(-1)
    if debugging:
        # analyse two batches separately
        for i in range(2):
            temp_arrays={tuple(x): (s, g) for x, s, g in zip(torch.where(arrays_tensor2[i] > 0, 1, -1).tolist(), scores2[i].tolist(), gens) if math.isfinite(s)}
            print(f"pre -improve batch {i}:")
            record_stats(temp_arrays)
    # step B
    improve2(arrays_tensor, scores)
    for _ in range(config.num_improve):
        improve1(arrays_tensor, scores)
    improve2(arrays_tensor, scores)  # TEST
    # step C: do some sorting
    mysort(arrays_tensor)
    # update
    for i in range(2):
        temp_arrays={tuple(x): (s, g) for x, s, g in zip(torch.where(arrays_tensor2[i] > 0, 1, -1).tolist(), scores2[i].tolist(), gens) if math.isfinite(s)}
        new_arrays_dict.update(temp_arrays)
        if debugging:
            print(f"post-improve batch {i}:")
            record_stats(temp_arrays)
    # select
    new_arrays_dict=best_from(new_arrays_dict)  # how often should I do this?
    return new_arrays_dict  # needed because of best_from

def batch_improve(arrays_dict,new_arrays_dict):
    if config.score_batch_size is None:
        return parallel_improve(arrays_dict.items(),new_arrays_dict)
    it = iter(arrays_dict.items())  # Convert dictionary to iterator
    while True:
        batch = list(islice(it, config.score_batch_size))  # Take next batch_size items
        if not batch:
            break
        new_arrays_dict = parallel_improve(batch,new_arrays_dict)
    return new_arrays_dict  # needed because of best_from

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
            print('\n***Improving***')
            arrays_dict = batch_improve(arrays_dict,{})
            if debugging:
                print(f"improving time: {timer() - start_timer}")
            print('\n***Selecting***')  # technically already done, but left for clarity of output
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
            coeff = 1 if params.gen == 0 or not resume_training else .1
            # linear warmup with fixed base learning rate afterwards:
            def get_lr(step, warmup_steps=10000):
                return config.learning_rate * (.01+.99*step / warmup_steps if step < warmup_steps else 1)
            max_steps = int(config.training_steps*coeff)
            eval_freq = int(500*coeff)
            start_timer = timer()
            transformer.train(arrays, score=score if params.test_randomisation else None, max_steps=max_steps, eval_freq=eval_freq, lr_sched=get_lr)
            if debugging:
                print(f"training time: {timer() - start_timer}")
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
            print(f"sampling time: {timer() - start_timer}")


if is_sweep:
    import wandb
    wandb.agent(logger.sweep_id, function=main)
else:
    main()
