#!/usr/bin/env python
# coding: utf-8

# beam search

import sys
import torch
import torch.nn.functional as F
import math
from collections import Counter
from timeit import default_timer as timer  # to measure exec time
import os

torch.set_printoptions(threshold=sys.maxsize,sci_mode=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
random_seed = 1
torch.manual_seed(random_seed)
if device.startswith('cuda'):
    torch.cuda.manual_seed_all(random_seed)

real_dtype = torch.float32
n = 236   # must be a multiple of 4
assert(n%4==0)
nm = 4
nn = n//4
nn2 = (nn-1)//2
na = n  # length of array
num_samples = 50_000
eps=2e-5
max_iterations = 1000
stop_at_first = True  # stop at first Hadamard

def fmt_array(s):
    return "".join("+" if x > 0 else "-" for x in s)


def generate_random_arrays(batch_size):
#    return 2 * torch.rand((batch_size, na), device=device, dtype=real_dtype) - 1
    return 2 * torch.randint(2, (batch_size, na), device=device, dtype=torch.int8) - 1

@torch.no_grad()
def legendre_pm1(p: int, *, dtype=torch.float32):
    """
    Return length-p tensor in {+1,-1}: a_j = (j|p) with a_0 := +1,
    randomized by a cyclic shift and a global sign flip.
    Requires odd prime p.
    """
    if p < 3 or p % 2 == 0:
        raise ValueError("p must be an odd prime")
    # Mark quadratic residues mod p
    # Using all k=1..p-1 (or k=1..(p-1)//2 also works) and setting mask[(k*k)%p] = True
    k = torch.arange(1, p, device=device, dtype=torch.int64)
    residues = (k * k) % p                           # [p-1]
    mask = torch.zeros(p, dtype=torch.bool, device=device)
    mask[residues] = True  # mark all quadratic residues; 0 stays False for now
    # Build a in {+1,-1}; set a_0 = +1 explicitly (0 is a residue too)
    a = torch.full((p,), -1, dtype=dtype, device=device)
    a[mask] = 1
    a[0] = 1
    return a

leg = legendre_pm1(nn)

cst = 1 / math.sqrt(n)
def fft(m):
    return cst * torch.fft.rfft(m.view(-1,nm,nn), dim=2)  # cst there for accuracy

def score_fft(f):  # score in terms of precomputed fft
    ff = torch.real(f*f.conj())
    s = -2*torch.log(ff.sum(dim=1)) # + 2 * F.relu(ff-1).sum(dim=1) + F.relu(ff[:,:3].sum(dim=1)-1)
    return s[:,0]+2*s[:,1:].sum(dim=1)

def score(m0):
    return score_fft(fft(m0))


if len(sys.argv) < 2:
    x = generate_random_arrays(num_samples)
    scores=score(x)
    mask=scores < float("inf")
    x = x[mask]
    scores = scores[mask]
    print(f'{mask.sum()} {torch.min(scores)} {torch.mean(scores)} {torch.max(scores)}')
else:
    filename = sys.argv[1]
    try:
        with open(filename, 'r') as f:
            arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for line in f]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    x = torch.tensor(arrays, dtype=torch.int8, device=device)  # Convert to tensor
    x = torch.cat((leg.unsqueeze(0).expand(x.shape[0],leg.shape[0]), x), dim=1)
    scores = score(x)
    if x.shape[0] > num_samples:
        scores, idx = torch.topk(scores, k=num_samples, largest=False, sorted=False)
        x = x[idx]

vec = torch.rand((nn,),device=device,dtype=real_dtype)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
@torch.inference_mode()
def derotate(arrays):
    # 1st phase: cyclically permute/reflect/negate the nm*nn
    B = arrays.shape[0]
    a=arrays.view(B, nm, nn)
    fft_a = torch.fft.rfft(a, dim=2)  # use fft to quickly compute scalar product with some random vector for ordering
    sp_rot = torch.fft.irfft(fft_conj_vec[None, None, :] * fft_a, n=nn, dim=2)  # (B, m, nn)
    sp_rev = torch.fft.irfft(fft_vec[None, None, :] * fft_a, n=nn, dim=2)  # (B, m, nn)
    sps = torch.cat([sp_rot, sp_rev], dim=2)   # (B, m, 2 * nn)
    flat_idx = sps.abs().argmax(dim=2)         # (B, m) over 2*nn options
    # Gather the chosen transform from the original 'a'
    signed_base = torch.where(flat_idx >= nn,-1,1).unsqueeze(-1) * base
    idx = ( signed_base + flat_idx.unsqueeze(-1)) % nn
    transformed = a.gather(2, idx)  # (B, m, nn)
    # negate if the chosen scalar product is > 0
    chosen_sps = sps.gather(2, flat_idx.unsqueeze(-1)).squeeze(-1)  # (B,m)
    a.copy_(torch.where(chosen_sps > 0, -1, 1).unsqueeze(-1) * transformed)
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

aut1 = torch.tensor([ i for i in range(1,nn2+1) if math.gcd(i,nn) == 1 ], device=device)  # variant of aut that stops at nn2
aut = [ i for i in range(1,nn) if math.gcd(i,nn) == 1 ]
aut_inds = torch.tensor([[(i*j)%nn for j in range(nn)] for i in aut], device=device)

@torch.inference_mode()
def apply_aut(idx,arrays0):
    B = arrays0.shape[0]
    arrays04 = arrays0.view(B, nm, nn)
    arrays = torch.empty_like(arrays0)
    arrays4 = arrays.view(B, nm, nn)
    # automorphism
    inds = aut_inds[idx]
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



file_path = "test.txt"  # TEMP
try:
    os.remove(file_path)
except FileNotFoundError:
    pass

def record_stats(msg):
    print(f'{msg} {t=} {x.shape[0]} : {torch.min(scores):10.6f} {torch.mean(scores):10.6f} {torch.max(scores):10.6f}')
    success = scores<eps
    if success.any():
        l = x[success]
        derotate(l)
        with open(file_path, 'a') as file:
            for s in l.tolist():
                file.write(fmt_array(s) + "\n")
        if stop_at_first:
            sys.exit("success")


hash_vec = torch.rand((na,),device=device,dtype=torch.float32)
hash_code = 1_000_000_007
def hash(x):
    return (x*hash_vec).sum(dim=1).view(torch.int32) % hash_code
#hash_mask = torch.ones((hash_code,), device=device, dtype=torch.bool)  # can be moved to cpu if necessary
hash_mask = torch.ones((hash_code,), dtype=torch.bool)

# build moves
moves = torch.ones((na,na), device=device, dtype=real_dtype)
b=torch.arange(na, device=device)
moves[b,b]=-1
n_moves = moves.shape[0]
batch_size = int(math.sqrt(n_moves))  # reasonable

for t in range(max_iterations):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    record_stats('')
    lst=[]
    for i in range(0, n_moves, batch_size):
        x1 = x[None,:,:] * moves[i:i+batch_size,None,:]
        x1 = torch.unique(x1.view(-1,na), dim=0)
        scores1 = score(x1)
        _, idx = torch.topk(scores1, k=num_samples, largest=False, sorted=False)
        lst.append(x1[idx])
    x = torch.cat(lst, dim=0)
    x = find_aut(x)
    derotate(x)
    x = torch.unique(x, dim=0)
    h = hash(x)
    mask = hash_mask[h.cpu()].to(device=device)
    x = x[mask]
    h = h[mask]
    scores = score(x)  # lazy
    if x.shape[0] < num_samples:
        print('hash code too low?')
        hash_mask[h.cpu()]=False
    else:
        scores, idx = torch.topk(scores, k=num_samples, largest=False, sorted=False)
        x = x[idx]
        hash_mask[h[idx].cpu()]=False
