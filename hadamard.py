#!/usr/bin/env python
# coding: utf-8

import math
import torch
import params
from params import n, na, nm, nn, nn2, device, resume, resume_training, random_seed, is_sweep, debugging, config, aut_inds, score, score_fft, fft, fixed_sums, num_ones
from pt import parallel_tempering, nT
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

if fixed_sums:
    def generate_random_blocks(B, n, k, device):  # Generate a (B, n) tensor of ±1 with exactly k entries of +1 per row.
        a = -torch.ones((B, n), dtype=torch.int8, device=device)
        rand = torch.rand((B, n), device=device)
        topk = rand.argsort(dim=1)[:, :k]        # (B, k) random unique positions per row
        rows = torch.arange(B, device=device).unsqueeze(1)
        a[rows, topk] = 1
        return a
    @torch.inference_mode()
    def generate_random_arrays(batch_size, device):  # used to be pure gpu, maybe reinstate at some point?
        return torch.cat([generate_random_blocks(batch_size, nn, num_ones[j], device) for j in range(4)], dim=1)
else:
    @torch.inference_mode()
    def generate_random_arrays(batch_size, device):  # used to be pure gpu, maybe reinstate at some point?
        return 2 * torch.randint(2, (batch_size, na), device=device, dtype=torch.int8) - 1

# MAIN-DEFINITIONS #

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
    print(f"{prefix} stats:")
    B = len(arrays)
    if B == 0:
        return
    # compute autocorrelation by MC
    mc_size = 1000
    s = (arrays[torch.randint(B,(mc_size,))] * arrays[torch.randint(B,(mc_size,))]).sum(dim=1).abs().sum()/(mc_size*na)
    """
    # proper/slow way: REDO ONE DAY
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
    s /= mc_size * na
    """
    s=s.item()
    print(f"Correlation: {s}")

    if fixed_sums:
        # check #1's
        a = arrays.view(B, 4, nn)
        k = (a.sum(dim=2).to(device=device)+nn)//2
        kcheck = (k == num_ones).all(dim=1)
        print(f"Correct # ones: {kcheck.sum()/B}")

    # now scores
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

    record_stats.hada_tensor = torch.unique(torch.cat((record_stats.hada_tensor,arrays[hada_inds].cpu()), dim=0), dim=0)
    if prefix == "selected":  # don't spam with H-matrices...
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

# scoring. technically we don't need this since the scores could be computed when improving;
# but useful for logging/stats. also generates random data at gen 0
def parallel_score(arrays):
    scores = score(arrays)  # Compute scores in parallel
    return scores.cpu()  # move back to cpu

def batch_generator(arrays):
    B = arrays.shape[0]
    for i in range(0, B, config.score_batch_size):
        j = i + config.score_batch_size
        yield arrays[i:j].to(device=device)

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
            arrays_gpu = arrays.to(device=device)
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
cst = 1 / math.sqrt(n)
w = torch.exp(2j * torch.tensor(torch.pi, device=device, dtype=real_dtype) / nn)
rng0 = torch.arange(nn, device=device, dtype=real_dtype)
rng = torch.arange(nn2+1, device=device, dtype=real_dtype)
wrng = 2 * cst * w ** torch.outer(rng0,rng)
wrng1 = -torch.conj(wrng)
wrng_all = torch.zeros((na,nm*(nn2+1)), device=device, dtype=complex_dtype)
for i in range(nm):
    wrng_all[i*nn:(i+1)*nn,i*(nn2+1):(i+1)*(nn2+1)] = wrng1

k = 9
gray_code = [ (i & -i).bit_length() - 1 for i in range(1,1<<k) ]
@torch.inference_mode()
def improve1p(arrays, scores):  # combined optimised 1-bit flip / opportunistic k-bit flip
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
            fl = f.view(-1, nm*(nn2+1))
            fmod = torch.empty_like(f)
            flmod = fmod.view(-1, nm*(nn2+1))
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
        fl = f.view(M, nm*(nn2+1))
        mask = torch.zeros((M,), device=device, dtype=torch.bool)
        for j in gray_code:
            inds = indsk[:,j]  # actual index for each sample
            fl += cur[:,j].unsqueeze(1) * wrng_all[inds]
            cur[:,j] *= -1  # need to keep track of these two
            new_scores = score_fft(f)
            improved = new_scores < scores[active_rows]
            if improved.any():
                mask[improved] = True  # these will get saved for next round
                improved_rows = active_rows[improved]
                scores[improved_rows] = new_scores[improved]
                #arrays[improved_rows.unsqueeze(1).expand(-1,k),indsk[improved]] = cur[improved]  # ugly and slow
                arrays.index_put_((improved_rows.unsqueeze(1).expand(-1,k), indsk[improved]), cur[improved])
        if not mask.any():
            break
        active_rows=active_rows[mask]  # eliminate those that haven't been improved at all

# greedy random k-bit flip
@torch.inference_mode()
def improve_greedy(x,scores):
    print(f"improve_greedy ", end=''); sys.stdout.flush()
    B=x.shape[0]
    # precompute fft
    f = fft(x)
    fl = f.view(B, nm*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B, nm*(nn2+1))
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for k in range(2,10):
        cnt.zero_()
        for _ in range(na):
            inds = torch.multinomial(torch.ones(na, device=device), num_samples=k, replacement=False)
            torch.matmul(x[:, inds].to(complex_dtype), wrng_all[inds], out=flmod)
            flmod.add_(fl)
            new_scores = score_fft(fmod)
            improved_inds = torch.nonzero(new_scores < scores, as_tuple=True)[0]
            fl[improved_inds] = flmod[improved_inds]
            x[improved_inds.unsqueeze(1),inds] *= -1
            scores[improved_inds] = new_scores[improved_inds]
            cnt += improved_inds.shape[0]
        print(f'{k=} {cnt/B}')
        # recompute scores for accuracy?
        # f = fft(x)
        # scores = score(x)

@torch.inference_mode()
def improve_joker(arrays, scores):
    print(f"improve_joker ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    B = arrays.shape[0]
    f = fft(arrays)
    a = arrays.view(B, nm, nn)
    for j in range(nm):
        cnt.zero_()
        ff = f*f.conj()
        ffs1 = ff.sum(dim=1) - ff[:,j]
        inds = torch.nonzero((ffs1.real <= 1).all(dim=1), as_tuple=True)[0]
        M = inds.shape[0]
        if M == 0:
            continue
        h = f[inds,j] * torch.sqrt((1-ffs1[inds])/ff[inds,j])
        fmod = torch.empty((M,nn2+1), device=device, dtype=complex_dtype)
        x = torch.empty((M,nn), device=device, dtype=torch.int8)
        x2 = torch.empty((M,nn), device=device, dtype=real_dtype)
        for t in range(100*n):  # ?
            torch.fft.irfft(h, n=nn, dim=1, out=x2)  # should be a 1/cst but doesn't matter
            x.fill_(-1)
            if fixed_sums:
                x.scatter_(1, torch.topk(x2, num_ones[j], dim=1).indices, 1)
            else:
                x.masked_fill_(x2 > 0, 1)
            torch.fft.rfft(x, dim=1, out=fmod)
            fmod *= cst
            s = -2*torch.log(torch.real(ffs1[inds] + fmod*fmod.conj()))
            new_scores = s[:,0]+2*s[:,1:].sum(dim=1)
            improved = new_scores < scores[inds]
            improved_inds = inds[improved]
            a[improved_inds,j] = x[improved]
            scores[improved_inds] = new_scores[improved]
            f[improved_inds,j] = fmod[improved]
            cnt += improved_inds.shape[0]
            h[:,1:] *= torch.exp(1j * 2 * torch.pi * torch.rand((M,nn2), device=device))
        print(f'({j}) {M} ({M/B}) {cnt} ({cnt/B})')

# greedy random k-bit rotate
@torch.inference_mode()
def improve_greedy_fixed(x,scores):
    print("improve_greedy_fixed ", end=''); sys.stdout.flush()
    B=x.shape[0]
    # precompute fft
    f = fft(x)
    fl = f.view(B, nm*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B, nm*(nn2+1))
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    k = 3  # 3,5,..,11
    ns = 5 * n  # dunno
    while ns > 0 and k <= nn2:
        cnt.zero_()
        # create all at once a bunch of subsets to sample
        lst=[]
        for j in range(4):
            r1=j*nn
            r=nn
            r2=r+r1
            lst.append(r1+torch.topk(torch.rand(ns, r, device=device), k).indices.sort(dim=1).values)
        all_inds = torch.unique(torch.cat(lst,dim=0),dim=0)
        n_inds = all_inds.shape[0]
        #print("temp",k,n_inds)
        perm = torch.randperm(n_inds)
        for i in range(n_inds):
            inds = all_inds[perm[i]]
            xx = torch.roll(x[:, inds], shifts=1, dims=1)
            torch.matmul((x[:, inds]-xx).to(complex_dtype), .5*wrng_all[inds], out=flmod)
            flmod.add_(fl)
            new_scores = score_fft(fmod)
            improved_inds = torch.nonzero(new_scores < scores, as_tuple=True)[0]  # better than mask when few True expected
            fl[improved_inds] = flmod[improved_inds]
            x[improved_inds.unsqueeze(1),inds] = xx[improved_inds]
            scores[improved_inds] = new_scores[improved_inds]
            cnt += improved_inds.shape[0]
        print(f'{k=} {cnt} ({cnt/B})')
        ns >>= 1
        k += 2

sw0 = torch.tensor([[-1, -1, 1, 1], [-1, 1, -1, 1], [-1, 1, 1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1]], device=device, dtype=torch.int8)
psw, ksw = sw0.shape  # psw = ksw choose ksw/2
sw_grids = torch.meshgrid(*[torch.arange(psw, device=device) for _ in range(4)], indexing='ij')
sw_idx = torch.stack(sw_grids, dim=-1).reshape(-1, 4)    # (p^4, 4)
sw = sw0[sw_idx].reshape(-1, 4 * ksw)

@torch.inference_mode()
def improve4x4_fixed(x,scores):  # optimal 4x4 bit switch
    print(f"improve4x4_fixed ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    B=x.shape[0]
    f = fft(x)
    fl = f.view(B,4*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B,4*(nn2+1))
    # first find 4x(2+2) locations for optimal flips
    best_scores = torch.full((B,4,2,2), float('inf'), dtype=real_dtype, device=device)
    inds = torch.empty((B,4,2,2), dtype=torch.long, device=device)
    for j in range(4):
        r1=j*nn
        r=nn
        r2=r+r1
        for i in range(r1,r2):
            torch.mul(x[:, i].to(complex_dtype).unsqueeze(1), wrng_all[i], out=flmod)
            flmod.add_(fl)
            scores1 = score_fft(fmod)
            mask = x[: ,i] > 0
            minuses = torch.nonzero(~mask, as_tuple=True)[0]
            pluses = torch.nonzero(mask, as_tuple=True)[0]
            # update minuses
            mask = scores[minuses].unsqueeze(1) < best_scores[minuses,j,0]  # (B,2)
            # mask[:,0] means highest score
            minuses1=minuses[mask[:,0]]
            best_scores[minuses1,j,0,1]=best_scores[minuses1,j,0,0]
            best_scores[minuses1,j,0,0]=scores1[minuses1]
            inds[minuses1,j,0,1]=inds[minuses1,j,0,0]
            inds[minuses1,j,0,0]=i
            # mask[:,1] means next to highest score
            minuses1=minuses[~mask[:,0] & mask[:,1]]
            best_scores[minuses1,j,0,1]=scores1[minuses1]
            inds[minuses1,j,0,1]=i
            # update pluses
            mask = scores[pluses].unsqueeze(1) < best_scores[pluses,j,1]  # (B,2)
            # mask[:,0] means highest score
            pluses1=pluses[mask[:,0]]
            best_scores[pluses1,j,1,1]=best_scores[pluses1,j,1,0]
            best_scores[pluses1,j,1,0]=scores1[pluses1]
            inds[pluses1,j,1,1]=inds[pluses1,j,1,0]
            inds[pluses1,j,1,0]=i
            # mask[:,1] means next to highest score
            pluses1=pluses[~mask[:,0] & mask[:,1]]
            best_scores[pluses1,j,1,1]=scores1[pluses1]
            inds[pluses1,j,1,1]=i
    # now try every combo
    inds = inds.view(B,16)
    base = torch.arange(B, device=device)
    #print(inds,x,scores)
    cur=torch.gather(x, 1, inds)
    for i in range(sw.shape[0]):
        x[base.unsqueeze(1), inds] = sw[i]
        new_scores = score(x)  # TODO use fft
        #print(x,new_scores)
        improved = new_scores < scores
        scores[improved] = new_scores[improved]
        cur[improved] = sw[i]
        cnt += torch.sum(improved)
    x.scatter_(1, inds, cur)
    print(f'{cnt/B}')

def fix_num_ones(arrays):  # fix # 1s. shouldn't happen too often
    a = arrays.view(-1, 4, nn)
    for j in range(4):
        while True:
            k = (a[:, j]==1).sum(dim=1)
            mask1 = k < num_ones[j]
            mask2 = k > num_ones[j]
            if not mask1.any() and not mask2.any():
                break
            a[mask1, j, torch.randint(nn,())] = 1  # lazy
            a[mask2, j, torch.randint(nn,())] = -1

vec = torch.rand((nn,),device=device,dtype=real_dtype)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
@torch.inference_mode()
def derotate(arrays, scores):
    if params.test_score:
        scores1 = score(arrays)
        if (scores-scores1).abs().max() > eps:
            raise RuntimeError("score incorrect", scores, scores1, (scores-scores1).abs().max().item(),(scores-scores1).abs().mean().item())
    # 1st phase: cyclically permute/reflect/negate the nm*nn
    B = arrays.shape[0]
    a=arrays.view(B, nm, nn)
    fft_a = torch.fft.rfft(a, dim=2)  # use fft to quickly compute scalar product with some random vector for ordering
    sp_rot = torch.fft.irfft(fft_conj_vec[None, None, :] * fft_a, n=nn, dim=2)  # (B, m, nn)
    sp_rev = torch.fft.irfft(fft_vec[None, None, :] * fft_a, n=nn, dim=2)  # (B, m, nn)
    sps = torch.cat([sp_rot, sp_rev], dim=2)   # (B, m, 2 * nn)
    if fixed_sums:
        flat_idx = sps.argmax(dim=2)         # (B, m) over 2*nn options
    else:
        flat_idx = sps.abs().argmax(dim=2)         # (B, m) over 2*nn options
    # Gather the chosen transform from the original 'a'
    signed_base = torch.where(flat_idx >= nn,-1,1).unsqueeze(-1) * base
    idx = ( signed_base + flat_idx.unsqueeze(-1)) % nn
    transformed = a.gather(2, idx)  # (B, m, nn)
    if fixed_sums:
        a.copy_(transformed)
    else:
        # negate if the chosen scalar product is > 0
        chosen_sps = sps.gather(2, flat_idx.unsqueeze(-1)).squeeze(-1)  # (B,m)
        a.copy_(torch.where(chosen_sps > 0, -1, 1).unsqueeze(-1) * transformed)
    if not fixed_sums:  # TODO even w/ fixed sums, there can be a partial permutation symmetry
        # 2nd phase: permute the nmxnn parts
        # start with identity permutation for each batch
        perm = torch.arange(nm, device=device).expand(B, nm).clone()
        # stable sort by last key first, then previous..., up to first column
        for k in range(nn):
            key = a[:, :, k]                 # (B, m)
            key_in_curr_order = key.gather(1, perm)
            ordk = torch.argsort(key_in_curr_order, dim=1, stable=True)
            perm = perm.gather(1, ordk)
        # apply permutation to rows
        sorted_a = a.gather(1, perm.unsqueeze(-1).expand(-1, -1, nn))
        a.copy_(sorted_a)
    if params.test_score:
        scores2 = score(arrays)
        if (scores1-scores2).abs().max() > eps:
            raise RuntimeError("score not preserved by sort", scores1, scores2, (scores1-scores2).abs().max().item(),(scores1-scores2).abs().mean().item())

aut1 = torch.tensor([ i for i in range(1,nn2+1) if math.gcd(i,nn) == 1 ], device=device)  # variant of aut that stops at nn2
aut_inds_gpu = aut_inds.to(device=device)

@torch.inference_mode()
def apply_aut(idx,arrays0):
    B = arrays0.shape[0]
    arrays04 = arrays0.view(B, nm, nn)
    arrays = torch.empty_like(arrays0)
    arrays4 = arrays.view(B, nm, nn)
    # automorphism
    inds = aut_inds_gpu[idx]
    inds_expanded = inds.unsqueeze(1).expand(B, nm, nn)
    arrays4.scatter_(2, inds_expanded, arrays04)
    return arrays

@torch.inference_mode()
def find_aut(arrays):
    f = fft(arrays)
    f = f.abs().sum(dim=1)  # (B,nn2+1)
    idx = f[:,aut1].argmax(dim=1)   # (B,) over nn2+1 options
    # now apply aut
    arrays1 = apply_aut(idx, arrays)
    return arrays1

def parallel_improve(arrays, scores, gens):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    # step Z: fix segment sums
    if fixed_sums:
        fix_num_ones(arrays)
    # step A: parallel tempering
    start_timer = timer()
    scores, inds = torch.sort(scores)
    arrays = arrays[inds]
    gens = gens[inds]
    B0 = torch.searchsorted(scores, eps)  # don't touch H-matrices
    B1 = arrays.shape[0]//4  # roughly at most 3/4 used
    B = (max(B0,B1)//nT)*nT
    parallel_tempering(arrays[B:], scores[B:], gens[B:])
    if debugging:
        print(f"pt time: {timer() - start_timer}")
        record_stats(arrays, scores, gens)
    # step B: improvement
    for _ in range(config.num_improve):
        start_timer = timer()
        improve_joker(arrays, scores)
        scores = score(arrays)  # don't trust improve_joker
        if debugging:
            print(f"improve_joker time: {timer() - start_timer}")
            record_stats(arrays, scores, gens)
        #
        start_timer = timer()
        if fixed_sums:
            improve_greedy_fixed(arrays,scores)
        else:
            improve1p(arrays, scores)
        scores = score(arrays)  # don't trust improve1p
        if debugging:
            print(f"improve1 time: {timer() - start_timer}")
            record_stats(arrays, scores, gens)
        #
        start_timer = timer()
        if fixed_sums:
            improve4x4_fixed(arrays, scores)
        else:
            improve_greedy(arrays, scores)
        scores = score(arrays)  # don't trust improve2
        if debugging:
            print(f"improve2 time: {timer() - start_timer}")
            record_stats(arrays, scores, gens)
    start_timer = timer()
    improve_joker(arrays, scores)
    scores = score(arrays)  # don't trust improve_joker
    if debugging:
        print(f"improve_joker time: {timer() - start_timer}")
        record_stats(arrays, scores, gens)
    # step C: rotate the arrays to a standard form
    start_timer = timer()
    arrays = find_aut(arrays)
    derotate(arrays, scores)
    if debugging:
        print(f"derotate time: {timer() - start_timer}")
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
    _, idx = torch.topk(min_scores * (1 + config.gen_decay * (params.gen - min_gens)), k=config.training_size, largest=False, sorted=False)
    arrays = arrays[idx]
    scores = min_scores[idx]
    gens = min_gens[idx]
    return arrays, scores, gens

def batch_improve(arrays0, scores0, gens0):
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        arrays, scores, gens = parallel_improve(arrays0.to(device=device), scores0.to(device=device), gens0.to(device=device))
        # select
        arrays, scores, gens = best_from(arrays, scores, gens)
    else:
        B = arrays0.shape[0]
        arrays = torch.empty((0,na), dtype=torch.int8, device=device)
        scores = torch.empty((0,), dtype=real_dtype, device=device)
        gens = torch.empty((0,), dtype=torch.uint8, device=device)
        for i in range(0, B, config.score_batch_size):
            j = i + config.score_batch_size
            new_arrays, new_scores, new_gens = parallel_improve(arrays0[i:j].to(device=device), scores0[i:j].to(device=device), gens0[i:j].to(device=device))
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
            arrays = torch.empty((0,na), dtype=torch.int8)
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
            transformer.train(arrays, score=score if params.test_score else None, max_steps=max_steps, eval_freq=eval_freq, lr_sched=get_lr)
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
        # combine, but mix
        A = arrays.shape[0]  # should be training_size
        B = new_arrays.shape[0]  # should be sample_size
        perm = torch.randperm(A+B)  # overkill but whatever
        combined_arrays = torch.empty((A+B, na), dtype=torch.int8)
        combined_scores = torch.empty(A+B, dtype=real_dtype)
        combined_gens = torch.empty(A+B, dtype=torch.uint8)
        combined_arrays[perm[:A]] = arrays
        combined_arrays[perm[A:]] = new_arrays
        combined_scores[perm[:A]] = scores
        combined_scores[perm[A:]] = new_scores
        combined_gens[perm[:A]] = gens
        combined_gens[perm[A:]] = new_gens
        arrays, scores, gens = combined_arrays, combined_scores, combined_gens
        if debugging:
            print(f"sampling time: {timer() - start_timer}")


if is_sweep:
    import wandb
    wandb.agent(logger.sweep_id, function=main)
else:
    main()
