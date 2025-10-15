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
aut = [ i for i in range(1,nn) if math.gcd(i,nn) == 1 ]
aut_inds1 = torch.tensor([[(i*j)%nn for j in range(nn)] for i in aut], device=device)
aut_inds3 = torch.tensor([[min((i*j)%nn,nn-(i*j)%nn)-1 for j in range(1,nn2+1)]  for i in aut], device=device)
aut1 = torch.tensor([ i for i in range(1,nn2+1) if math.gcd(i,nn) == 1 ], device=device)

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
    #arrays1.copy_(torch.gather(arrays01, 1, inds1))  # WRONG WAY!
    #arrays3.copy_(torch.gather(arrays03, 2, inds3.unsqueeze(1).expand(-1,3,-1)))
    base = torch.arange(B, device=device)
    arrays1[base.unsqueeze(1), inds1] = arrays01
    arrays3[base.unsqueeze(1).unsqueeze(2), torch.arange(3, device=device)[None,:,None],inds3[:,None,:]] = arrays03
    return arrays

# need a score that's invariant under other symmetries: permute, dihedral symmetry of last one (but not full permutation!)
# idea: take say sum of abs of fft, these will be just permuted by automorphism -> find smallest entry whose index is prime with nn?

one = torch.tensor([[1]],device=device,dtype=torch.float32)
def mir(a):
    return torch.cat((one.expand(a.shape[0],1),a,torch.flip(a,(1,))),dim=1)
def unfold(m0):
    return torch.stack((mir(m0[:,:nn2]),
                        mir(m0[:,nn2:2*nn2]),
                        mir(m0[:,2*nn2:3*nn2]),
                        m0[:,3*nn2:]),dim=1)
def fft(m):
    return torch.fft.rfft(m, dim=2)

def find_aut(arrays):
    f = fft(unfold(arrays))
    f = f.abs().sum(dim=1)  # (B,nn2+1)
    idx = f[:,aut1].argmax(dim=1)   # (B,) over nn2+1 options
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
mysort(arrays1)
h = torch.unique(arrays1.to(torch.int8), dim=0, sorted=False)
#h = set(tuple(a) for a in arrays.tolist())
len1 = len(h)
print(f'{len1}/{len0}')
