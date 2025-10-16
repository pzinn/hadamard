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
n = 140   # must be a multiple of 4
assert(n%4==0)
nn = n//4 # must be odd
assert(nn%2==1)
nn2 = (nn-1)//2
na1 = 3*nn2
na2 = nn
na = na1 + na2  # length of array
debugging = True
num_samples = 150_000
eps=1e-5
max_iterations = 1000
stop_at_first = True  # stop at first Hadamard

k0 = [1, 3, 7, 9]  # sum of squares must be n, for the first 3, k0=2-nn=nn [4] which fixes their sign, last one is just odd
k0 = [i if (nn-i)%4 == 0 else -i for i in k0]  # fix signs if necessary
k = [(k0[j]+nn-2)//4 if j<3 else (k0[j]+nn)//2 for j in range(4)]


def fmt_array(s):
    return "".join("+" if x > 0 else "-" for x in s)


def generate_random_blocks(B, n, k, device):
    """
    Generate a (B, n) tensor of ±1 with exactly k entries of +1 per row.
    """
    # Start with all -1s
    a = -torch.ones((B, n), dtype=torch.int8, device=device)
    # For each row, randomly choose k positions to set to +1
    # We'll use torch.rand and argsort to get k random indices per row
    rand = torch.rand((B, n), device=device)
    topk = rand.argsort(dim=1)[:, :k]        # (B, k) random unique positions per row
    # Use advanced indexing to assign +1
    rows = torch.arange(B, device=device).unsqueeze(1)
    a[rows, topk] = 1
    return a


@torch.inference_mode()
def generate_random_arrays(batch_size):
    return torch.cat([generate_random_blocks(batch_size, nn2 if j<3 else nn, k[j], device) for j in range(4)], dim=1)




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
    ff = torch.real(f*f.conj())
    s = -2*torch.log(ff.sum(dim=1)) # + 2 * F.relu(ff-1).sum(dim=1) + F.relu(ff[:,:3].sum(dim=1)-1)
    return s[:,0]+2*s[:,1:].sum(dim=1)

def score(m0):
    return score_fft(fft(unfold(m0)))

"""
def flip_fft(j, a, f):  # j = which bit to flip, a = which way, f = fft
    fl = f.view(-1,4*(nn2+1))
    fl += a.unsqueeze(1) * wrng_all[j]


w = torch.exp(2j * torch.tensor(torch.pi, device=device, dtype=torch.float32) / nn)
rng0 = torch.arange(nn, device=device, dtype=torch.float32)
rng = torch.arange(nn2+1, device=device, dtype=torch.float32)
wrng = 2 * cst * w ** torch.outer(rng0,rng)
wrng012 = -(wrng + torch.conj(wrng))[1:nn2+1]
wrng3 = -torch.conj(wrng)
wrng_all = torch.zeros((na,4*(nn2+1)), device=device, dtype=torch.complex64)
for i in range(3):
    wrng_all[i*nn2:(i+1)*nn2,i*(nn2+1):(i+1)*(nn2+1)] = wrng012
wrng_all[3*nn2:,3*(nn2+1):] = wrng3
"""


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
    x = torch.tensor(arrays, dtype=real_dtype, device=device)  # Convert to tensor
    scores = score(x)
    if x.shape[0] > num_samples:
        scores, idx = torch.topk(scores, k=num_samples, largest=False, sorted=False)
        x = x[idx]

vec = torch.rand((nn,),device=device,dtype=torch.float32)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
def mysort(x):
    # 1st phase: permute the 3xnn2 parts
    B = x.shape[0]
    m=3
    a=x[:,:m*nn2].view(B,m,nn2)
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
    a=x[:,m*nn2:]
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

aut1 = torch.tensor([ i for i in range(1,nn2+1) if math.gcd(i,nn) == 1 ], device=device)  # variant of aut that stops at nn2
aut = [ i for i in range(1,nn) if math.gcd(i,nn) == 1 ]
aut_inds1 = torch.tensor([[(i*j)%nn for j in range(nn)] for i in aut], device=device)
aut_inds3 = torch.tensor([[min((i*j)%nn,nn-(i*j)%nn)-1 for j in range(1,nn2+1)]  for i in aut], device=device)

def apply_aut(idx,arrays0):
    B = arrays0.shape[0]
    arrays03=arrays0[:,:3*nn2].view(-1,3,nn2)
    arrays01=arrays0[:,3*nn2:]
    inds1 = aut_inds1[idx]
    inds3 = aut_inds3[idx]
    arrays = torch.empty_like(arrays0)
    arrays3=arrays[:,:3*nn2].view(-1,3,nn2)
    arrays1=arrays[:,3*nn2:]
    # automorphism
    base = torch.arange(B, device=device)
    arrays1[base[:,None], inds1] = arrays01
    #arrays3[base[:,None,None], torch.arange(3, device=device)[None,:,None],inds3[:,None,:]] = arrays03
    inds3_expanded = inds3.unsqueeze(1).expand(-1, 3, -1)
    arrays3.scatter_(2, inds3_expanded, arrays03)
    return arrays

def find_aut(arrays):
    f = fft(unfold(arrays))
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
        mysort(l)
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
n_moves = 1000  # dunno
moves = torch.ones((n_moves,na), device=device, dtype=torch.int8)
for j in range(4):
    r1=j*nn2
    r=nn2 if j<3 else nn
    r2=r+r1
    a = torch.arange(j*n_moves//4,(j+1)*n_moves//4, device=device)
    moves[a.unsqueeze(1),r1+torch.multinomial(torch.ones(r, device=device), num_samples=2, replacement=False)] = -1

batch_size = int(math.sqrt(n_moves))  # reasonable

for t in range(max_iterations):        
    record_stats('')
    lst=[]
    for i in range(0, n_moves, batch_size):
        x1 = x[None,:,:] * moves[i:i+batch_size,None,:]
        x1 = torch.unique(x1.view(-1,na), dim=0)
        scores1 = score(x1)
        _, idx = torch.topk(scores1, k=num_samples, largest=False, sorted=False)
        lst.append(x1[idx])
    x = torch.cat(lst, dim=0)
    #x = find_aut(x)
    mysort(x)
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
        

