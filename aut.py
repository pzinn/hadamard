#!/usr/bin/env python
# coding: utf-8

# a script to find symmetries of matrices (excluding (Z/nnZ)^* automorphisms)

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

from itertools import permutations
# Prepare permutations -- note that these tensor are on cpu, if rotate used on gpu this needs to be changed
perms = torch.tensor(list(p for p in permutations(range(nm))), dtype=torch.long)
rndmod = torch.tensor([len(perms), 2*nn, 2*nn, 2*nn, 2*nn, 2, 2, 2, 2], dtype=torch.int64)
nrnd = rndmod.shape
print("order of symmetry: ", rndmod.prod().item())

def rotate(array0,arrayx,rnd):  # for reference only
    array0 = array0.view(-1,nm,nn)
    array = arrayx.view(-1,nm,nn)
    # automorphisms
    #array.copy_(array0[:,:,aut_inds[rnd[1].item()]])
    array.copy_(array0)
    # symmetry: random permute
    array.copy_(array[:,perms[rnd[0]]])
    # symmetry: random rotation/flip
    for j in range(nm):
        array[:,j] = torch.roll(array[:,j] if rnd[j+1] < nn else torch.flip(array[:,j], (1,)), shifts=rnd[j+1].item(), dims=1)
    # symmetry: random signs
    array.mul_((rnd[5:9]*2-1).unsqueeze(1))


filename = sys.argv[1]
try:
    with open(filename, 'r') as f:
        arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for line in f]
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
arrays = torch.tensor(arrays, dtype=torch.int8, device=device)  # Convert to tensor
B = arrays.shape[0]
print(f"number of data: {B}")
arrays = arrays.view(B,nm,nn)
for k in range(2*nn):
    arrays1 = torch.roll(arrays if k < nn else torch.flip(arrays, (2,)), shifts=k, dims=2)
    flag = False
    for j in range(3):
        for jj in range(j if k>0 else j+1,4):
            mask1 = (arrays1[:,j] == arrays[:,jj]).all(dim=1)
            mask2 = (-arrays1[:,j] == arrays[:,jj]).all(dim=1)
            if mask1.any() or mask2.any():
                flag=True
                print(f"{j} {jj} {k} : {mask1.sum().item()} {mask2.sum().item()}", end='\t')
    if flag:
        print("")
