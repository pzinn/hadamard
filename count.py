#!/usr/bin/env python
# coding: utf-8

# a script to count matrices up to all symmetries. takes input from stdin
# "sym" version

import torch
import math

score_type = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

import sys

nn = 35  # size of basic block. must be odd for this version!
n = 4 * nn  # size of matrix
nn2 = (nn-1)//2
na = 3*nn2 + nn  # length of array


vec = torch.rand((nn,),device=device,dtype=torch.float32)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
def mysort(arrays):
    # 1st phase: permute the 3xnn2 parts
    B = arrays.shape[0]
    m=3
    a=arrays[:,:m*nn2].view(B,m,nn2)
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
    a=arrays[:,m*nn2:]
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

# Prepare automorphisms
aut = [ i for i in range(2,nn) if math.gcd(i,nn) == 1 ]
aut_inds1 = [ torch.tensor([(i*j)%nn for j in range(nn)], device=device) for i in aut]
aut_inds3 = [ torch.tensor([min((i*j)%nn,nn-(i*j)%nn)-1 for j in range(1,nn2+1)], device=device) for i in aut]

def rotate(i,arrays0,arrays):
    arrays03=arrays0[:,:3*nn2].view(-1,3,nn2)
    arrays01=arrays0[:,3*nn2:]
    arrays3=arrays[:,:3*nn2].view(-1,3,nn2)
    arrays1=arrays[:,3*nn2:]
    # automorphism
    arrays1.copy_(arrays01[:,aut_inds1[i]])
    arrays3.copy_(arrays03[:,:,aut_inds3[i]])

vec2 = torch.rand((na,),device=device,dtype=torch.float32)  # doesn't really matter, used for ordering

filename = sys.argv[1]
try:
    with open(filename, 'r') as f:
        arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for line in f]
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
arrays = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor
len0 = arrays.shape[0]
mysort(arrays)

arrays0 = arrays.clone()
s = (arrays*vec2).sum(dim=1)
arrays1 = torch.empty_like(arrays)
for i in range(len(aut)):
    rotate(i,arrays0,arrays1)
    mysort(arrays1)
    s1 = (arrays1*vec2).sum(dim=1)
    mask = s1 < s
    arrays[mask]=arrays1[mask]
    s[mask]=s1[mask]

h = torch.unique(arrays.to(torch.int8), dim=0, sorted=False)
#h = set(tuple(a) for a in arrays.tolist())
len1 = len(h)
print(f'{len1}/{len0}')
