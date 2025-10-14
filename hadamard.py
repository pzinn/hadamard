#!/usr/bin/env python
# coding: utf-8

import math
import torch
import params
from params import n, na, nn, nn2, device, resume, resume_training, random_seed, is_sweep, debugging, config
import logger
import transformer
# logging/debugging
import sys
from timeit import default_timer as timer  # to measure exec time

eps = 2e-5  # scores are heavily discretised so can be made large

real_dtype = torch.float32
complex_dtype = torch.complex64

import torch
"""
if debugging:
    # Start recording memory snapshot history
    torch.cuda.memory._record_memory_history(max_entries=100000)
"""

@torch.inference_mode()
def generate_random_arrays(batch_size, device):  # used to be pure gpu, maybe reinstate at some point?
    return 2 * torch.randint(2, (batch_size, na), device=device, dtype=real_dtype) - 1

# MAIN-DEFINITIONS #
"""
def best_from(arrays_dict):
    # preserves ordering
    items = arrays_dict.items()
    smallest_keys = {k for k, _ in heapq.nsmallest(config.training_size, items, key=lambda item: item[1][0])}  # heapq requires no nan
    return {k: v for k, v in items if k in smallest_keys or v[0] < eps}  # always keep H-matrices
"""

def write_arrays_buffer(buffer, a):
    plus, minus = ord('+'), ord('-')
    rows = torch.where(a > 0, plus, minus).to(torch.uint8)
    for r in rows:
        buffer.write(bytes(r.tolist()))
        buffer.write(b"\n")

def write_arrays(file_path, a):
    with open(file_path, 'wb') as file:
        write_arrays_buffer(file, a)

def print_arrays(a):
    write_arrays_buffer(sys.stdout.buffer, a)
    sys.stdout.flush()

# for keeping track of stats
def record_stats(arrays, scores, gens, prefix=""):
    B = len(arrays)
    if B == 0:
        return
    # compute autocorrelation by MC
    mc_size = 1000
    perms = params.perms.tolist()
    s = torch.tensor(0., device=arrays.device)
    for _ in range(mc_size):
        a1 = arrays[torch.randint(B,())]
        a2 = arrays[torch.randint(B,())]
        a13 = a1[:3*nn].reshape(3,nn)
        a23 = a1[:3*nn].reshape(3,nn)
        a11 = a1[3*nn:]
        a21 = a2[3*nn:]
        fft_a1 = torch.fft.fft(a11, dim=0)
        fft_a2 = torch.fft.fft(a21, dim=0)
        corr1 = torch.fft.ifft(fft_a1 * torch.conj(fft_a2), dim=0).real
        corr2 = torch.fft.ifft(fft_a1 * fft_a2, dim=0).real
        corr = torch.stack([corr1,corr2],dim=0)
        corr = torch.abs(corr)
        s += torch.max(corr)
        ss = 0
        for p in perms:
            sss = torch.sum(a13[p]*a23,dim=(0,1))
            if sss>ss:
                ss=sss
        s += ss
    s /= (mc_size * na)
    s=s.item()
    print(f"Correlation: {s}")

    # now scores
    # if debugging:
    #     print(f'Score tally: {dict(zip(*np.unique(np.round(scores, decimals=5), return_counts=True)))}')

    min_score = torch.min(scores)
    print(f"Min score: {min_score}")

    mean_score = torch.mean(scores)
    print(f"Mean score: {mean_score}")

    max_score = torch.max(scores)
    print(f"Max score: {max_score}")

    # tally=Counter([val[1] for val in vals])
    gens_count = torch.bincount(gens)
    gens_count, gens_order = gens_count.sort(descending=True)
    gens_tally = { g : c for g, c in zip(gens_order.tolist(), gens_count.tolist()) if c > 0 }
    print(f"Gen tally: {gens_tally}")

    hada_inds = torch.nonzero(scores < eps, as_tuple=True)[0]
    nh = len(hada_inds) / len(arrays)
    print(f"Hadamard ratio: {nh}")

    hada_count = torch.bincount(gens[hada_inds])
    hada_count, hada_order = hada_count.sort(descending=True)
    hada_tally = { g : c for g, c in zip(hada_order.tolist(), hada_count.tolist()) if c > 0 }
    print(f"Hadamard gen tally: {hada_tally}")

    if prefix == "selected":  # don't spam with H-matrices...
        record_stats.hada_tensor = torch.unique(torch.cat((record_stats.hada_tensor,arrays[hada_inds]), dim=0), dim=0)
        total_nh = len(record_stats.hada_tensor)
        print(f"Total number of Hadamard: {total_nh}")
        if total_nh>0:
            print_arrays(record_stats.hada_tensor[0:1])  # doesn't appear on log??
            write_arrays(logger.hada_file, record_stats.hada_tensor)

    with open(logger.stats_file, 'a') as file:
        if not record_stats.has_run:
            record_stats.has_run = True
            file.write(f"{'gen':>3} {'':<10}: {'min score':>10} {'mean score':>10} {'max score':>10} {'autocorrel':>10} {'H-ratio':>10} {'H-number':>10} tally / H-tally\n")
        file.write(f"{params.gen:>3} {prefix:<10}: {min_score:10.6f} {mean_score:10.6f} {max_score:10.6f} {s:10.6f} {nh:10.6f} {len(hada_inds):>10} {gens_tally} {hada_tally}\n")

    if prefix:
        logger.record_scores(prefix, scores, gens, mean_score, nh)

cst = 1 / math.sqrt(n)

def init_score_function():
    global score, score_cpu, score_fft, fft
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
        def score(m):
            return n/2 * math.log(n) - torch.linalg.slogdet(block_circulant(m))[1]
    elif config.score_function == 'fft log determinant':
        nm=4
        @torch.inference_mode()
        def fft(m):
            return cst * torch.fft.rfft(m.view(-1, nm, nn), dim=2)  # cst there for accuracy
        @torch.inference_mode()
        def score_fft(f):  # score in terms of precomputed fft
            # we do separately real pieces for accuracy reasons
            s = - torch.log(torch.real(f[:, :, 0].pow(2).sum(dim=1)))
            if nn % 2 == 0:
                s -= torch.log(torch.real(f[:, :, nn//2].pow(2).sum(dim=1)))
                f = f[:, :, 1:-1]
            else:
                f = f[:, :, 1:]
            ff = f[:, :3, :].pow(2).sum(dim=1)
            f2 = f * f.conj()  # f.mul_(f.conj())  # can't do that since f is kept
            ff.mul_(ff.conj())
            s -= torch.log(torch.real(ff+f2[:, 3]*(2*f2.sum(dim=1)-f2[:, 3]))).sum(dim=1)
            return 2*s
        @torch.inference_mode()
        def score(m):
            return score_fft(fft(m))
        def score_cpu(m):
            return score_fft(fft(m))
    elif config.score_function == 'quartic':
        score_threshold = n**1.5
        score_normalisation = 2*math.sqrt(n)
        def score(m):
            C = block_circulant(m)
            nrm = torch.linalg.matrix_norm(torch.matmul(C, torch.transpose(C, 1, 2)))
            return (nrm-score_threshold)/score_normalisation
    elif config.score_function == 'one':
        score_threshold = 0
        score_normalisation = n
        Idn = n * torch.eye(n, device=device, dtype=real_dtype)
        def score(m):
            C = block_circulant(m)
            nrm = torch.linalg.matrix_norm(torch.matmul(C, torch.transpose(C, 1, 2))-Idn, ord=1)
            return (nrm-score_threshold)/score_normalisation
    else:
        raise Exception('unknown score_function')

# scoring. technically we don't need this since the scores could be computed when improving;
# but useful for logging/stats. also generates random data at gen 0
def parallel_score(arrays):
    scores = score(arrays)  # Compute scores in parallel
    return scores.cpu()  # move back to cpu

def batch_generator(arrays):
    B = arrays.shape[0]
    for i in range(0, B, config.score_batch_size):
        j = i + config.score_batch_size
        yield arrays[i:j].to(device=device, dtype=real_dtype)

"""
def random_batch_generator():
    n_full_batches = config.sample_size // config.score_batch_size
    remainder = config.sample_size % config.score_batch_size
    for _ in range(n_full_batches):
        yield generate_random_arrays(config.score_batch_size)
    if remainder:
        yield generate_random_arrays(remainder)
"""

def batch_score(arrays):  # same as parallel_score but in batches of score_batch_size
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        if arrays is None:
            arrays_gpu = generate_random_arrays(config.sample_size, device)
            arrays = arrays_gpu.to(device='cpu', dtype=torch.int8)
        else:
            arrays_gpu = arrays.to(device=device, dtype=real_dtype)
        return arrays, parallel_score(arrays_gpu)
    scores = torch.empty((0,), dtype=real_dtype)
    if arrays is None:
        arrays = generate_random_arrays(config.sample_size, 'cpu')  # lame? reinstate old way?
    batches = batch_generator(arrays)
    for batch in batches:
        new_scores = parallel_score(batch)
        scores = torch.cat((scores, new_scores), dim=0)
    return arrays, scores

# precompute roots of unity for fft delta
w = torch.exp(2j * torch.tensor(torch.pi, device=device, dtype=real_dtype) / nn)
rng0 = torch.arange(nn, device=device, dtype=real_dtype)
rng = torch.arange(nn2+1, device=device, dtype=real_dtype)
wrng = 2 * cst * w ** torch.outer(rng0,rng)
wrng1 = -torch.conj(wrng)
wrng_all = torch.zeros((na,4*(nn2+1)), device=device, dtype=complex_dtype)
for i in range(4):
    wrng_all[i*nn:(i+1)*nn,i*(nn2+1):(i+1)*(nn2+1)] = wrng1

k = 9
@torch.inference_mode()
def improve1p(arrays, scores):  # combined optimised 1-bit flip / opportunistic k-bit flip
    start_timer = timer()
    print(f"improve1p ", end=''); sys.stdout.flush()
    B=arrays.shape[0]
    active_rows = torch.nonzero(scores>=eps, as_tuple=True)[0]  # don't bother with H-matrices
    scores1 = torch.empty((B,na), device=device, dtype=real_dtype)
    while True:
        M = active_rows.numel()
        print(f'{M/B}')
        cur_rows = active_rows
        while True:
            f = fft(arrays[cur_rows])  # better than flip updating for accuracy
            fl = f.view(-1,4*(nn2+1))
            fmod = torch.empty_like(f)
            flmod = fmod.view(-1, 4*(nn2+1))
            for j in range(na):
                torch.mul(arrays[cur_rows, j].to(complex_dtype).unsqueeze(1), wrng_all[j], out=flmod)
                flmod.add_(fl)
                scores1[cur_rows, j] = score_fft(fmod)
            mask = (scores1[cur_rows] < scores[cur_rows].unsqueeze(1)).any(dim=1)
            if not mask.any():
                break
            # easy ones: 1-bit flip.
            cur_rows = cur_rows[mask]
            min_scores, inds = scores1[cur_rows].min(dim=1)
            scores[cur_rows] = min_scores
            arrays[cur_rows,inds] *= -1
        # hard ones: brute force k best candidates
        _, indsk = torch.topk(scores1[active_rows], k, dim=1, sorted=False, largest=False)
        cur=torch.gather(arrays[active_rows], 1, indsk)
        f = fft(arrays[active_rows])
        fl = f.view(M, 4*(nn2+1))
        mask = torch.zeros((M,), device=device, dtype=torch.bool)
        for i in range(1,1<<k):
            j = (i & -i).bit_length() - 1  # index of bit to flip
            inds = indsk[:,j]  # actual index for each sample
            fl += cur[:,j].unsqueeze(1) * wrng_all[inds]
            cur[:,j] *= -1  # need to keep track of these two
            new_scores = score_fft(f)
            improved = new_scores < scores[active_rows]
            mask[improved] = True  # these will get saved for next round
            improved_rows = active_rows[improved]
            scores[improved_rows] = new_scores[improved]
            arrays[improved_rows.unsqueeze(1).expand(-1,k),indsk[improved]] = cur[improved]  # ugly and slow
        if not mask.any():
            print(f"improve1p time: {timer() - start_timer}")
            break
        active_rows=active_rows[mask]  # eliminate those that haven't been improved at all

"""
@torch.inference_mode()
def improve1(arrays, scores):
    print(f"improve1", end=''); sys.stdout.flush()
    # first let's do it the stupidest way (will recode later)
    B = arrays.shape[0]
    active_mask = torch.ones(B, device=device, dtype=torch.bool)
    active_rows = torch.arange(B, device=device)
    while True:
        # rows we’re actively trying to improve this round
        M = active_rows.numel()
        # indices of best bit (−1 means no improvement found)
        inds = torch.full((M,), -1, dtype=torch.int64, device=device)
        # Try flipping each bit, keep any improvement
        for i in range(na):
            arrays[active_rows, i] *= -1          # flip bit i
            new_scores = score(arrays[active_rows])
            improved = new_scores < scores[active_rows]                  # where this flip helps
            if improved.any():
                scores[active_rows[improved]] = new_scores[improved]            # write into base `scores`
                inds[improved] = i
            arrays[active_rows, i] *= -1          # flip back
        # rows that actually improved this round
        improved_any = inds >= 0
        if not improved_any.any():
            break
        # Apply the winning flips once
        active_rows = active_rows[improved_any]
        active_cols = inds[improved_any]
        arrays[active_rows, active_cols] *= -1
        # Next round: only keep rows that improved (they might improve again)
        active_mask.zero_()
        active_mask[active_rows] = True
        if debugging:
            #print(f' {torch.bincount(active_cols, minlength=na)} improve success rate: {active_mask.sum()/B}')
            print(f' improve success rate: {active_mask.sum()/B}')

@torch.inference_mode()
def improve1a(arrays,scores):  # the old improve1: flip a single bit greedily
    if debugging:
        cnt = torch.tensor(0, device=device, dtype=torch.int64)
    print(f"1", end=''); sys.stdout.flush()
    p = torch.randperm(na)  # should it be on device?
    for i in range(na):
        arrays[:, p[i]] *= -1  # Flip only the i-th bit
        # Compute new scores for all batch elements in parallel
        new_scores = score(arrays)
        # Identify which flips improved the score
        mask = new_scores < scores  # True where improvement happens
        if debugging:
            cnt += torch.sum(mask)
        # Apply successful bit flips
        arrays[~mask, p[i]] *= -1  # Only revert for elements where no improvement
        scores[mask] = new_scores[mask]  # Update scores accordingly
    if debugging:
        print(f' improve success rate: {cnt/arrays.shape[0]}')

# greedy 2-bit flip -- probably too slow for large n
@torch.inference_mode()
def improve2(x,scores):
    print("improve2 ", end=''); sys.stdout.flush()
    if debugging:
        cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for i in range(1,na):
        x[:, i] *= -1
        for j in range(i):
            x[:, j] *= -1
            new_scores = score(x)
            mask = new_scores < scores
            if debugging:
                cnt += torch.sum(mask)
            x[~mask, j] *= -1  # Only revert for elements where no improvement
            x[mask, i] *= -1  # !!!
            scores[mask] = new_scores[mask]  # Update scores accordingly
        x[:, i] *= -1
    if debugging:
        print(f' success rate: {cnt/x.shape[0]}')
    else:
        print('')
"""

# greedy random k-bit flip
@torch.inference_mode()
def improve2(x,scores):
    B=x.shape[0]
    # precompute fft
    f = fft(x)
    fl = f.view(B,4*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B,4*(nn2+1))
    if debugging:
        cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for k in range(2,10):
        if debugging:
            cnt.zero_()
        for _ in range(na):
            inds = torch.multinomial(torch.ones(na, device=device), num_samples=k, replacement=False)
            torch.matmul(x[:, inds].to(complex_dtype), wrng_all[inds], out=flmod)
            flmod.add_(fl)
            new_scores = score_fft(fmod)
            improved = new_scores < scores
            improved_inds = torch.nonzero(improved, as_tuple=True)[0]
            fl[improved_inds] = flmod[improved_inds]
            x[improved_inds.unsqueeze(1),inds] *= -1
            if debugging:
                cnt += improved.sum()
            scores[improved_inds] = new_scores[improved_inds]
        if debugging:
            print(f'{k=} {cnt/B}')
        # recompute scores for accuracy?
        # f = fft(x)
        # scores = score(x)

"""
# greedy random k-bit flip -- slower version
@torch.inference_mode()
def improve2w(x,scores):
    B=x.shape[0]
    if debugging:
        cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for k in range(1,10):
        if debugging:
            cnt.zero_()
        for _ in range(na):
            inds = torch.multinomial(torch.ones(na, device=device), num_samples=k, replacement=False)
            x[:, inds] *= -1
            new_scores = score(x)
            x[:, inds] *= -1
            improved = new_scores < scores
            improved_inds = torch.nonzero(improved, as_tuple=True)[0]
            x[improved_inds.unsqueeze(1),inds] *= -1
            if debugging:
                cnt += improved.sum()
            #print(inds,cnt)
            scores[improved_inds] = new_scores[improved_inds]
        if debugging:
            print(f'{k=} {cnt/B}')
"""

@torch.inference_mode()
def improve3(m):
    print(f"improve3 ", end=''); sys.stdout.flush()
    B=m.shape[0]
    f = fft(m)
    ff = f*f.conj()
    ffs = ff.sum(dim=1)
    lst = []
    for j in range(4):
        ffs1 = ffs - ff[:,j]
        # the ambitious version
        #threshold = 1 - .2 * torch.rand((), device=device) # randomise a bit
        mask = (ffs1.real <= 1).all(dim=1)
        M = mask.sum()
        print(f'success rate ({j}) {M/B}')
        if M > 0:
            x = m[mask].clone()
            h = torch.sqrt(1-ffs1[mask])
            h[:,1:] *= torch.exp(1j * 2 * torch.pi * torch.rand((M,nn2), device=device))
            hh = torch.fft.irfft(h,n=nn,dim=1)  # should be a 1/cst but doesn't matter since we're gonna sign it
            x[:, j*nn:(j+1)*nn] = torch.where(hh > 0, 1., -1.)
            lst.append(x)
    return torch.cat(lst, dim=0) if len(lst) > 0 else None
    """
    # the less ambitious version -- but maybe should be kept? though all it does is increase # samples?
    z, _ = ffs.real.max(dim=1)
    if debugging:
        print(f'success {(z<=1).sum()/B}')
    z=torch.max(z,torch.tensor(1))
    ffs /= z.unsqueeze(1)
    h = torch.sqrt(1-ffs)
    h[:,1:] *= torch.exp(1j * 2 * torch.pi * torch.rand((B,nn2),device=device))
    x2=torch.fft.irfft(h,n=nn,dim=1)
    return torch.cat((x1,torch.where(x2 > 0, 1., -1.)),dim=1)
    """

"""
# failed gradient descent
def unfold2(m1,m2):
    return torch.stack((mir(m1[:,:nn2]),
                        mir(m1[:,nn2:2*nn2]),
                        mir(m1[:,2*nn2:]),
                        m2),dim=1)
def alt_score0(m):
    f = alt_cst * torch.fft.rfft(m, dim=2)  # cst there for accuracy
    ff = torch.real(f*f.conj())
    ffs = ff.sum(dim=1)
    s = (ffs-1)**2
    return s[:,0]+2*s[:,1:].sum(dim=1)
def score2(m1,m2):
    return alt_score0(unfold2(m1,m2))

def improve3(arrays,max_iterations=5,inner_steps=100,lr=.05,mixed_precision=True):
    x=arrays.clone()
    print(f"3", end=''); sys.stdout.flush()
    B = x.shape[0]
    scaler = torch.amp.GradScaler(device,enabled=mixed_precision)

    x1 = torch.nn.Parameter(x[:, :3*nn2].detach())
    x2 = torch.nn.Parameter(x[:, 3*nn2:].detach())
    opt1 = torch.optim.AdamW([x1], lr=lr)
    opt2 = torch.optim.AdamW([x2], lr=lr)
    #opt1 = torch.optim.SGD([x1], lr=lr)
    #opt2 = torch.optim.SGD([x2], lr=lr)

    global alt_cst

    for t in range(max_iterations):
        #first set
        alt_cst = math.sqrt(3) / math.sqrt(3*nn2)  # controls average size of abc -- (cst0/cst)^2 where cst0 = 1/sqrt(3nn2)
        start_timer=timer()
        x2d=torch.zeros_like(x2)
        #x2.detach()  # could be moved out of the loop
        #x2d=torch.where(x2>0,1.,-1.).detach()
        for _ in range(inner_steps):
            opt1.zero_grad(set_to_none=True)
            with torch.amp.autocast(device,enabled=mixed_precision):
                fake_scores = score2(x1,x2d)
                loss = fake_scores.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt1)
            scaler.step(opt1)
            scaler.update()
            with torch.no_grad():
                x1.clamp_(min=-1, max=1)  # projection
        print(f'1a {t=} : {torch.min(fake_scores)} {torch.mean(fake_scores)} {torch.max(fake_scores)} time={timer()-start_timer}')
        print(x1.abs().mean(),x2.abs().mean())
        #alt_cst = 1 / math.sqrt(n)
        if debugging:
            real_x = torch.where(x > 0, 1., -1.)  # not needed, for testing only
            real_scores = score(real_x)
            print(f'1b {t=} : {torch.min(real_scores)} {torch.mean(real_scores)} {torch.max(real_scores)} time={timer()-start_timer}')
        #second set
        alt_cst = 2 / math.sqrt(n)  # controls average size of d --- how exactly?
        start_timer=timer()
        x1d=x1.detach()  # could be moved out of the loop
        for _ in range(inner_steps):
            opt2.zero_grad(set_to_none=True)
            with torch.amp.autocast(device,enabled=mixed_precision):
                fake_scores = score2(x1d,x2)
                loss = fake_scores.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            scaler.step(opt2)
            scaler.update()
            with torch.no_grad():
                x2.clamp_(min=-1, max=1)  # projection
        if debugging:
            print(f'2a {t=} : {torch.min(fake_scores)} {torch.mean(fake_scores)} {torch.max(fake_scores)} time={timer()-start_timer}')
            print(x1.abs().mean(),x2.abs().mean())
        #next
        #alt_cst = 1 / math.sqrt(n)
        real_x = torch.where(x > 0, 1., -1.)
        real_scores = score(real_x)
        if debugging:
            print(f'2b {t=} : {torch.min(real_scores)} {torch.mean(real_scores)} {torch.max(real_scores)} time={timer()-start_timer}')
        if t==0:
            arrays.copy_(real_x)
            scores.copy_(real_scores)
        else:
            improve = real_scores <= scores
            arrays[improve]=real_x[improve]
            scores[improve]=real_scores[improve]

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

vec = torch.rand((nn,),device=device,dtype=real_dtype)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
def mysort(arrays, scores):
    if params.test_score:
        scores1 = score(arrays)
        if (scores-scores1).abs().max() > eps:
            raise RuntimeError("score incorrect", scores, scores1, (scores-scores1).abs().max().item(),(scores-scores1).abs().mean().item())
    # 0th phase: cyclically permute/reflect/negate the 3*nn
    B = arrays.shape[0]
    m=3
    a=arrays[:,:m*nn].view(B,m,nn)
    fft_a = torch.fft.rfft(a, dim=2)  # use fft to quickly compute scalar product with some random vector for ordering
    sp_rot = torch.fft.irfft(fft_conj_vec[None, :] * fft_a, n=nn, dim=2)  # (B, m, nn)
    sp_rev = torch.fft.irfft(fft_vec[None, :] * fft_a, n=nn, dim=2)  # (B, m, nn)
    sps = torch.cat([sp_rot, sp_rev], dim=2)   # (B, m, 2 * nn)
    flat_idx = sps.abs().sum(dim=1).argmax(dim=1)         # (B,) over 2*nn options
    # Gather the chosen transform from the original 'a'
    signed_base = torch.where(flat_idx >= nn,-1,1).unsqueeze(1) * base.unsqueeze(0)
    idx = ( signed_base + flat_idx.unsqueeze(1)) % nn
    transformed = a.gather(2, idx.unsqueeze(1).expand(B,m,nn))  # (B, m, nn)
    # negate if the chosen scalar product is > 0
    chosen_sps = sps.gather(2, flat_idx.unsqueeze(1).expand(B,m).unsqueeze(2)).squeeze(2)  # (B,m)
    a.copy_(torch.where(chosen_sps > 0, -1, 1).unsqueeze(2) * transformed)
    # 1st phase: permute the 3xnn parts
    # start with identity permutation for each batch
    perm = torch.arange(m, device=device).expand(B, m).clone()
    # stable sort by last key first, then previous..., up to first column
    for k in range(nn):
        key = a[:, :, k]                 # (B, m)
        key_in_curr_order = key.gather(1, perm)
        ordk = torch.argsort(key_in_curr_order, dim=1, stable=True)
        perm = perm.gather(1, ordk)
    # apply permutation to rows
    sorted_a = a.gather(1, perm.unsqueeze(-1).expand(-1, -1, nn))
    a.copy_(sorted_a)
    # 2nd phase: cyclically permute/reflect/negate the remaining length nn part
    a=arrays[:,m*nn:]
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
    if params.test_score:
        scores2 = score(arrays)
        if (scores1-scores2).abs().max() > eps:
            raise RuntimeError("score not preserved by sort", scores1, scores2, (scores1-scores2).abs().max().item(),(scores1-scores2).abs().mean().item())

def parallel_improve(arrays, scores, gens):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    #scores = score(arrays)  # Recompute scores in parallel -- for accuracy reasons, safer
    # step A: demultiply data
    start_timer = timer()
    arrays1=improve3(arrays)
    if debugging:
        print(f"improve3 time: {timer() - start_timer}")
    if arrays1 is not None:
        scores1 = score(arrays1)
        gens1 = torch.full((arrays1.shape[0],),params.gen,device=device,dtype=torch.uint8)
        if debugging:
            # analyse two batches separately
            print(f"pre -improve batch 0:")
            record_stats(arrays, scores, gens)
            print(f"pre -improve batch 1:")
            record_stats(arrays1, scores1, gens1)
        arrays = torch.cat((arrays,arrays1),dim=0)
        scores = torch.cat((scores,scores1),dim=0)
        gens = torch.cat((gens,gens1),dim=0)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    # step B: main improvement
    start_timer = timer()
    improve1p(arrays, scores)
    if debugging:
        print(f"improve1p time: {timer() - start_timer}")
    scores = score(arrays)  # don't trust improve1p
    #
    start_timer = timer()
    for _ in range(config.num_improve):
        improve2(arrays, scores)
        scores = score(arrays)  # don't trust improve2
    if debugging:
        print(f"improve2 time: {timer() - start_timer}")
    # step C: rotate the arrays to a standard form
    start_timer = timer()
    mysort(arrays, scores)
    if debugging:
        print(f"mysort time: {timer() - start_timer}")
    return (arrays, scores, gens)

def best_from(arrays, scores, gens):
    # deduplicate
    arrays, inv = torch.unique(arrays, dim=0, return_inverse=True, sorted=False)
    B = arrays.shape[0]
    min_gens = torch.empty(B, device=device, dtype=torch.uint8)
    min_gens.scatter_reduce_(0, inv, gens, reduce='amin', include_self=False)
    # normally scores should be equal but who knows
    min_scores = torch.empty(B, device=device, dtype=real_dtype)
    min_scores.scatter_reduce_(0, inv, scores, reduce='amin', include_self=False)
    # select
    if B <= config.training_size:
        return arrays, min_scores, min_gens
    scores, idx = torch.topk(min_scores, k=config.training_size, largest=False, sorted=False)
    arrays = arrays[idx]
    gens = min_gens[idx]
    return arrays, scores, gens

def batch_improve(arrays0, scores0, gens0):
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        arrays, scores, gens = parallel_improve(arrays0.to(device=device, dtype=real_dtype), scores0.to(device=device), gens0.to(device=device))
        # select
        arrays, scores, gens = best_from(arrays, scores, gens)
    else:
        B = arrays0.shape[0]
        arrays = torch.empty((0,na), dtype=real_dtype, device=device)
        scores = torch.empty((0,), dtype=real_dtype, device=device)
        gens = torch.empty((0,), dtype=torch.uint8, device=device)
        for i in range(0, B, config.score_batch_size):
            j = i + config.score_batch_size
            new_arrays, new_scores, new_gens = parallel_improve(arrays0[i:j].to(device=device, dtype=real_dtype), scores0[i:j].to(device=device), gens0[i:j].to(device=device))
            arrays = torch.cat((arrays, new_arrays), dim=0)
            scores = torch.cat((scores, new_scores), dim=0)
            gens = torch.cat((gens, new_gens), dim=0)
            arrays, scores, gens = best_from(arrays, scores, gens)
    return arrays.to(device='cpu', dtype=torch.int8), scores.cpu(), gens.cpu()

def main():
    # logging: text stats file + fancy (tensorboard or wandb)
    logger.init_logging()
    record_stats.has_run = False  # we could leave it undefined, but not in case of sweep
    record_stats.hada_tensor = torch.empty((0,na), dtype=torch.int8)  # empty the hadamard list

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
                arrays = torch.tensor([tuple(1 if c == "+" else -1 for c in line.strip()) for line in f], dtype=torch.int8)
            print(f'***Loading initial sample from {init_sample}***')
        except FileNotFoundError:
            arrays = []
    else:
        arrays = None

    arrays, scores = batch_score(arrays)
    gens = torch.full(scores.shape, params.gen, dtype=torch.uint8)
    record_stats(arrays, scores, gens, prefix="sample" if not resume else "")  # who knows where the data come from if resuming

    # MAIN-LOOP #

    while True:
        if params.skip_first_improve:
            params.skip_first_improve = False
        else:
            # improve existing data, write to GEN-(gen)
            print('\n***Improving***')
            start_timer = timer()
            arrays, scores, gens = batch_improve(arrays, scores, gens)
            if debugging:
                print(f"improving time: {timer() - start_timer}")
            print('\n***Selecting***')  # technically already done, but left for clarity of output
            record_stats(arrays, scores, gens, "selected")
            write_arrays(params.work_dir + f'GEN-{params.gen:02d}.txt', arrays)
        if params.gen == config.max_iterations:
            """
            if debugging:
                torch.cuda.memory._dump_snapshot("profile.pkl")
                torch.cuda.memory._record_memory_history(enabled=None)
            """
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
            transformer.train(arrays, score=score_cpu if params.test_score else None, max_steps=max_steps, eval_freq=eval_freq, lr_sched=get_lr)
            if debugging:
                print(f"training time: {timer() - start_timer}")
        # sample from model to get new data
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        print(f"\n***Sampling from transformer trained on GEN-{params.gen:02d}***")
        params.gen += 1
        start_timer = timer()
        new_arrays = transformer.sample()
        new_arrays, new_scores = batch_score(new_arrays)
        new_gens = torch.full(new_scores.shape, params.gen, dtype=torch.uint8)
        record_stats(new_arrays, new_scores, new_gens, prefix="sample")  # do we produce similar scores as training data?
        arrays = torch.cat((arrays, new_arrays), dim=0)
        scores = torch.cat((scores, new_scores), dim=0)
        gens = torch.cat((gens, new_gens), dim=0)
        if debugging:
            print(f"sampling time: {timer() - start_timer}")


if is_sweep:
    import wandb
    wandb.agent(logger.sweep_id, function=main)
else:
    main()
