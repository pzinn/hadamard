#!/usr/bin/env python
# coding: utf-8

# a script to count matrices up to all symmetries. takes input from stdin

import torch
import math

score_type = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

import sys

nn = 35  # size of basic block
nm = 4
n = nm * nn  # size of matrix
na = nm * nn  # length of array
nn2 = (nn-1)//2


vec = torch.rand((nn,),device=device,dtype=torch.float32)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
def derotate(arrays):
    # 1st phase: cyclically permute/reflect/negate the 4*nn
    B = arrays.shape[0]
    m=4
    a=arrays.view(B,m,nn)
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

# Prepare automorphisms
aut = [ i for i in range(2,nn) if math.gcd(i,nn) == 1 ]
aut_inds = torch.tensor([[(i*j)%nn for j in range(nn)] for i in aut], device=device)
aut1 = torch.tensor([ i for i in range(1,nn2+1) if math.gcd(i,nn) == 1 ], device=device)

def fft(m):
    return torch.fft.rfft(m.view(-1, nm, nn), dim=2)

def apply_aut(idx,arrays0):
    B = arrays0.shape[0]
    arrays04 = arrays0.view(B,4,nn)
    arrays = torch.empty_like(arrays0)
    arrays4 = arrays.view(B,4,nn)
    # automorphism
    inds = aut_inds[idx]
    inds_expanded = inds.unsqueeze(1).expand(-1, 4, -1)
    arrays4.scatter_(2, inds_expanded, arrays04)
    return arrays

def find_aut(arrays):
    f = fft(arrays)
    f = f.abs().sum(dim=1)  # (B,nn2+1)
    idx = f[:,aut1].argmax(dim=1)   # (B,) over nn2+1 options
    # now apply aut
    arrays1 = apply_aut(idx, arrays)
    return arrays1


filename = sys.argv[1]
try:
    with open(filename, 'r') as f:
        arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for line in f]
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
arrays = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor
len0 = arrays.shape[0]
arrays1 = find_aut(arrays)
derotate(arrays1)

h = torch.unique(arrays1.to(torch.int8), dim=0, sorted=False)
#h = set(tuple(a) for a in arrays.tolist())
len1 = len(h)
print(f'{len1}/{len0}')
