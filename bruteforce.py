#!/usr/bin/env python
# coding: utf-8

# script to brute force

import sys
import torch
import torch.nn.functional as F
import math
from collections import Counter
from timeit import default_timer as timer  # to measure exec time
import os

torch.set_printoptions(threshold=sys.maxsize,sci_mode=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
score_type = torch.float32
n = 108   # must be a multiple of 4
assert(n%4==0)
nn = n//4 # must be odd
assert(nn%2==1)
nn2 = (nn-1)//2
na1 = 3*nn2
na2 = nn
na = na1 + na2  # length of array
debugging = False
num_samples = 50_000
eps=1e-5
max_iterations = 100
stop_at_first = False  # stop at first Hadamard

def fmt_array(s):
    return "".join("+" if x > 0 else "-" for x in s)


def generate_random_arrays(batch_size):
    return 2 * torch.rand((batch_size, na), device=device, dtype=score_type) - 1



cst = 1 / math.sqrt(n)
one = torch.tensor([[1]],device=device,dtype=torch.float32)
def mir(a):
    return torch.cat((one.expand(a.shape[0],1),a,torch.flip(a,(1,))),dim=1)
def unfold(m0):
    return torch.stack((mir(m0[:,:nn2]),
                        mir(m0[:,nn2:2*nn2]),
                        mir(m0[:,2*nn2:3*nn2]),
                        m0[:,3*nn2:]),dim=1)
def unfold2(m1):
    B=m1.shape[0]
    m1=m1.view(B,3,nn2)
    return torch.cat((one.expand(B,3,1),m1,torch.flip(m1,(2,))),dim=2)

def fft(m):
    return cst * torch.fft.rfft(m, dim=2)  # cst there for accuracy

def score_fft(f):  # score in terms of precomputed fft
    s = -2*torch.log(torch.real(f*f.conj()).sum(dim=1))
    return s[:,0]+2*s[:,1:].sum(dim=1)

def score(m0):
    return score_fft(fft(unfold(m0)))

"""
def alt_score0(m):
    f = cst * torch.fft.rfft(m, dim=2)  # cst there for accuracy
    ff = torch.real(f*f.conj())
    ffs = ff.sum(dim=1)
    s = (ffs-1)**2
    return s[:,0]+2*s[:,1:].sum(dim=1)  # what happens if we vary relative coefficient? nothing good apparently
def alt_score(m0):
    return alt_score0(unfold(m0))  # makes things worse
"""

w = torch.exp(2j * torch.tensor(torch.pi, device=device, dtype=torch.float32) / nn)
rng0 = torch.arange(nn, device=device, dtype=torch.float32)
rng = torch.arange(nn2+1, device=device, dtype=torch.float32)
wrng = 2 * cst * w ** torch.outer(rng0,rng)
wrng012 = (wrng + torch.conj(wrng))[1:nn2+1]
wrng3 = torch.conj(wrng)
wrng_all = torch.cat((wrng012,wrng012,wrng012,wrng3),dim=0)

def flip_fft(j, a, f):  # j = which bit to flip, a = which way, f = fft
    l = j // nn2 if j < 3*nn2 else 3
    f[:,l] -= a.unsqueeze(1) * wrng_all[j]


k=10
@torch.inference_mode()
def improve1p(arrays_tensor, scores):  # combined optimised 1-bit flip / opportunistic k-bit flip
    start_timer = timer()
    print(f"improve1p ", end=''); sys.stdout.flush()
    B=arrays_tensor.shape[0]
    active_rows = torch.arange(B, device=device)
    while True:
        M = active_rows.numel()
        if M==0:
            print(f"improve1p time: {timer() - start_timer}")
            break
        scores1 = torch.zeros((M,na),device=device,dtype=torch.float32)
        # precompute fft
        f = fft(unfold(arrays_tensor[active_rows]))
        # Try flipping each bit, keep any improvement
        for i in range(na):
            """
            arrays_tensor[:, i] *= -1          # flip bit i
            scores1[:, i] = score(arrays_tensor)
            arrays_tensor[:, i] *= -1          # flip back
            """
            # v2
            flip_fft(i, arrays_tensor[active_rows, i], f)
            scores1[:, i] = score_fft(f)
            flip_fft(i, -arrays_tensor[active_rows, i], f)
        mask = (scores1 < scores[active_rows].unsqueeze(1)).any(dim=1)
        # easy ones: 1-bit flip.
        easy_rows = active_rows[mask]
        min_scores, inds = scores1[mask].min(dim=1)
        scores[easy_rows] = min_scores
        arrays_tensor[easy_rows,inds] *= -1
        # hard ones: brute force k best candidates
        hard_rows = active_rows[~mask]
        M2 = hard_rows.numel()
        print(f'ratios: {M/B} {M2/B}')
        #print(f'{hard_rows=}')
        if M2 > 0:
            hard_inds = torch.nonzero(~mask, as_tuple=True)[0]  # <sigh>
            _, inds = torch.topk(scores1[~mask], k, dim=1, sorted=False, largest=False)
            cur=arrays_tensor[hard_rows].clone()
            f = fft(unfold(cur))
            cur_rows = torch.arange(M2, device=device)
            for i in range(1,1<<k):
                j = (i & -i).bit_length() - 1  # index of bit to flip
                cols = inds[:,j]  # actual index for each sample
                l = torch.where(cols < 3*nn2, cols//nn2, 3)  # imitate fft_flip except now indices vary per sample
                f[cur_rows,l] -= cur[cur_rows,cols].unsqueeze(1) * wrng_all[cols]
                cur[cur_rows,cols] *= -1  # need to keep track of these two
                new_scores = score_fft(f)
                improved = new_scores < scores[hard_rows]
                mask[hard_inds[improved]] = True  # these will get saved for next round
                scores[hard_rows[improved]] = new_scores[improved]
                arrays_tensor[hard_rows[improved]] = cur[improved]
            active_rows=active_rows[mask]  # eliminate those that haven't been improved at all


@torch.inference_mode()
def improve1(arrays_tensor, scores):
    start_timer = timer()
    print(f"improve1 ", end=''); sys.stdout.flush()
    B = arrays_tensor.shape[0]
    active_mask = torch.ones(B, device=device, dtype=torch.bool)
    active_rows = torch.arange(B, device=device)
    flip_counts = torch.zeros(na, dtype=torch.int64, device=device) if debugging else None    
    while True:
        # rows we’re actively trying to improve this round
        M = active_rows.numel()
        print(f'{M/B} ', end='')
        # indices of best bit (−1 means no improvement found)
        inds = torch.full((M,), -1, dtype=torch.int64, device=device)
        # precompute fft
        f = fft(unfold(arrays_tensor[active_rows]))
        # Try flipping each bit, keep any improvement
        for i in range(na):
            """
            arrays_tensor[active_rows, i] *= -1          # flip bit i
            new_scores = score(arrays_tensor[active_rows])
            arrays_tensor[active_rows, i] *= -1          # flip back
            """
            # v2
            flip_fft(i, arrays_tensor[active_rows, i], f)
            new_scores = score_fft(f)
            flip_fft(i, -arrays_tensor[active_rows, i], f)
            #
            improved = new_scores < scores[active_rows]                  # where this flip helps
            if improved.any():
                scores[active_rows[improved]] = new_scores[improved]            # write into base `scores`
                inds[improved] = i
        # rows that actually improved this round
        improved_any = inds >= 0
        if not improved_any.any():
            print('')
            if debugging:
                print(flip_counts)
            print(f"improve1 time: {timer() - start_timer}")
            break
        # Apply the winning flips once
        active_rows = active_rows[improved_any]
        active_cols = inds[improved_any]
        arrays_tensor[active_rows, active_cols] *= -1
        if debugging:
            flip_counts += torch.bincount(active_cols, minlength=na)

def flip_fft_mask(j, a, f, mask):  # j = which bit to flip, a = which way, f = fft TODO merge with non mask
    l = j // nn2 if j < 3*nn2 else 3
    f[mask,l] -= a[mask].unsqueeze(1) * wrng_all[j]

# greedy 2-bit flip
@torch.inference_mode()
def improve2(x,scores):
    start_timer = timer()
    print("improve2 ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    # precompute fft
    f = fft(unfold(x))
    for i in range(1,na):
        #x[:, i] *= -1
        flip_fft(i,x[:,i],f)
        for j in range(i):
            #x[:, j] *= -1
            flip_fft(j,x[:,j],f)
            #new_scores = score(arrays_tensor)
            new_scores = score_fft(f)
            mask = new_scores < scores
            cnt += torch.sum(mask)
            #x[~mask, j] *= -1  # Only revert for elements where no improvement
            #x[mask, i] *= -1  # !!!
            flip_fft_mask(i,-x[:,i],f,mask)
            flip_fft_mask(j,-x[:,j],f,~mask)
            x[mask, i] *= -1
            x[mask, j] *= -1
            scores[mask] = new_scores[mask]  # Update scores accordingly
        flip_fft(i,-x[:,i],f)
        #x[:, i] *= -1
    print(f'{cnt/x.shape[0]}')
    print(f"improve2 time: {timer() - start_timer}")

# greedy 2-bit flip, restricted to intrablocks
@torch.inference_mode()
def improve2a(arrays_tensor,scores):
    print("improve2 ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for k in range(4):
        if k<3:
            x=arrays_tensor[:,k*nn2:(k+1)*nn2]
        else:
            x=arrays_tensor[:,3*nn2:]
        for i in range(1,x.shape[1]):
            x[:, i] *= -1
            for j in range(i):
                x[:, j] *= -1
                new_scores = score(arrays_tensor)
                mask = new_scores < scores
                cnt += torch.sum(mask)
                x[~mask, j] *= -1  # Only revert for elements where no improvement
                x[mask, i] *= -1  # !!!
                scores[mask] = new_scores[mask]  # Update scores accordingly
            x[:, i] *= -1
    print(f'{cnt/x.shape[0]}')

if len(sys.argv) < 2:
    arrays_tensor = generate_random_arrays(num_samples)
    scores=score(arrays_tensor)
    mask=scores < float("inf")
    arrays_tensor = arrays_tensor[mask]
    scores = scores[mask]
    print(f'{mask.sum()} {torch.min(scores)} {torch.mean(scores)} {torch.max(scores)}')
else:
    filename = sys.argv[1]
    try:
        with open(filename, 'r') as f:
            arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for _, line in zip(range(num_samples), f)]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    arrays_tensor = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor
    # complete with garbage
    arrays_tensor = torch.cat((arrays_tensor,generate_random_arrays(num_samples-arrays_tensor.shape[0])),dim=0)

vec = torch.rand((nn,),device=device,dtype=torch.float32)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
def mysort(arrays_tensor):
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


file_path = "test.txt"  # TEMP
try:
    os.remove(file_path)
except FileNotFoundError:
    pass

@torch.inference_mode()
def batch_gradient_descent(
        x,                           # (B, n) float tensor on CUDA
        lr=5e-2,
        mixed_precision=True,
        tolerance=.5                     # proportion of steps of no improve before giving up
):
    B = x.shape[0]
    #scaler = torch.amp.GradScaler(device,enabled=mixed_precision)

    x1 = x[:, :3*nn2]

    #counter = torch.zeros(x.shape[0], device=device, dtype=torch.int64)

    #prev_scores = None
    for t in range(max_iterations):
        if device.startswith('cuda'):
            torch.cuda.empty_cache()  # Free memory
        #first set
        start_timer=timer()
        m = unfold2(x1)
        f = cst * torch.fft.rfft(m, dim=2)
        #ff = f*f.conj()  # in fact, just f**2
        ff = f**2
        ffs = ff.sum(dim=1)
        z = torch.max(ffs.real.max(dim=1)[0],torch.tensor(1))
        ffs /= z.unsqueeze(1)
        h = torch.sqrt(1-ffs)
        h[:,1:] *= torch.exp(1j * 2 * torch.pi * torch.rand((B,nn2),device=device))
        x[:,3*nn2:]=1/cst*torch.fft.irfft(h,n=nn,dim=1)  # actually 1/cst doesn't matter since we're gonna sign it
        if t==max_iterations-1+1:  # save for testing purposes; remove +1 to enable
            print(f'mod_score: {torch.min(scores)} {torch.mean(scores)} {torch.max(scores)} time={timer()-start_timer}')
            scores=score(x)
            print(f'    score: {torch.min(scores)} {torch.mean(scores)} {torch.max(scores)} time={timer()-start_timer}')
            with open('dump.txt', 'w') as file:
                for s in x.tolist():
                    for r in s:
                        file.write(str(r)+", ");
                    file.write("\n")
        #next
        x.copy_(torch.where(x > 0, 1., -1.))
        scores = score(x)
        print(f'pre improve {t=} : {torch.min(scores)} {torch.mean(scores)} {torch.max(scores)}')
        #improve1(x, scores)
        #improve2(x,scores)
        improve1p(x,scores)
        print(f'postimprove {t=} : {torch.min(scores)} {torch.mean(scores)} {torch.max(scores)}')
        success = scores<eps
        if success.any():
            l = x[success]
            mysort(l)
            with open(file_path, 'a') as file:
                for s in l.tolist():
                    file.write(fmt_array(s) + "\n")
            if stop_at_first:
                return
        x[success] = generate_random_arrays(success.sum())

batch_gradient_descent(arrays_tensor)
