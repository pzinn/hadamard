#!/usr/bin/env python
# coding: utf-8

# script to brute force improve using gradient descent

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
na = 3*nn2 + nn  # length of array
debugging = False
max_samples = 50_000
#num_improve = 10
eps=1e-5
eval_freq=100
max_iterations = 25_000 # special value -1 means stop at first Hadamard

"""
# range of #s of random flips. optimum value?
r1=int(.5*math.sqrt(na))
r2=int(2*math.sqrt(na))
"""

def fmt_array(s):
    return "".join("+" if x > 0 else "-" for x in s)


def generate_random_arrays(batch_size):
    return 2 * torch.randint(2, (batch_size, na), device=device, dtype=score_type) - 1


if len(sys.argv) < 2:
    arrays_tensor = generate_random_arrays(max_samples)
else:
    filename = sys.argv[1]
    try:
        with open(filename, 'r') as f:
            arrays = [tuple(1 if c == "+" else -1 for c in line.strip()) for _, line in zip(range(max_samples), f)]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    arrays_tensor = torch.tensor(arrays, dtype=score_type, device=device)  # Convert to tensor
    # complete with garbage
    arrays_tensor = torch.cat((arrays_tensor,generate_random_arrays(max_samples-arrays_tensor.shape[0])),dim=0)


cst = 1 / math.sqrt(n)
def score(m0):
    #print(m0.shape)
    # reduce to non sym case but simplify due to phase alignment of first 3 fft
    nm=4
    ones=torch.ones((m0.size(0),1),device=m0.device,dtype=score_type)  # take out
    m=torch.cat((m0[:,:nn2],ones,torch.flip(m0[:,:nn2],(1,)),
                 m0[:,nn2:2*nn2],ones,torch.flip(m0[:,nn2:2*nn2],(1,)),
                 m0[:,2*nn2:3*nn2],ones,torch.flip(m0[:,2*nn2:3*nn2],(1,)),
                 m0[:,3*nn2:]),dim=1)
    f = cst * torch.fft.rfft(m.view(-1, nm, nn), dim=2)  # cst there for accuracy
    ff = torch.real(f*f.conj())
    s = torch.log(ff.sum(dim=1))
    return -2*s[:,0]-4*s[:,1:].sum(dim=1)

@torch.inference_mode()
def improve1(x,scores):  # TODO update in same way as hadamard.py
    active_mask = torch.ones(x.shape[0],device=device,dtype=torch.bool)
    new_scores = scores.clone()
    t=0
    while active_mask.any():
        t += 1
        new_active_mask = torch.zeros(x.shape[0],device=device,dtype=torch.bool)
        p = torch.randperm(na)
        for i in range(na):
            x[:, p[i]] *= -1  # Flip only the i-th bit
            # Compute new scores for all batch elements in parallel
            new_scores[active_mask] = score(x[active_mask])
            # Identify which flips improved the score
            mask = new_scores < scores  # True where improvement happens
            new_active_mask[mask] = 1
            x[~mask, p[i]] *= -1  # Only revert for elements where no improvement
            scores[mask] = new_scores[mask]  # Update scores accordingly
        active_mask=new_active_mask
    print(f"improv stopped at {t=}")

cf=1
def mod_score(m):
    #return score(torch.tanh(m))
    return score(m)+cf*torch.sum(m**2,dim=1)


file_path = "test.txt"  # TEMP
try:
    os.remove(file_path)
except FileNotFoundError:
    pass

def batch_gradient_descent(
        x,                           # (B, n) float tensor on CUDA
        steps=-1,                       # -1 = infinite
        lr=1e-2,
        mixed_precision=True,
        tolerance=.5                     # proportion of steps of no improve before giving up
):
    """
    Returns: (opt_data, history) where opt_data is the final tensor (B, n),
             history is a dict with last scores and optional traces.
    """
    x.requires_grad_(True)
    scaler = torch.amp.GradScaler('cuda',enabled=mixed_precision)

    # opt = torch.optim.SGD([x], lr=lr)
    opt = torch.optim.AdamW([x], lr=lr)

    counter = torch.zeros(x.shape[0], device=device, dtype=torch.int64)

    prev_scores = None
    t=0
    start_timer=timer()
    while t != steps:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device,enabled=mixed_precision):
            scores = mod_score(x)
            loss = scores.sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            x.clamp_(min=-1, max=1)  # projection
            if t>0:
                improv = (scores - prev_scores) < -eps
                counter[~improv] += 1
                counter[improv] = 0
            prev_scores = scores
            t += 1
            if t % eval_freq == 0:
                real_x = torch.where(x > 0, 1., -1.)
                real_scores = score(real_x)
                print(f'{t=} : {torch.min(real_scores)} {torch.mean(real_scores)} {torch.max(real_scores)} time={timer()-start_timer}')
                # traditional improve
                start_timer=timer()
                improve1(real_x,real_scores)
                print(f'{t=} : {torch.min(real_scores)} {torch.mean(real_scores)} {torch.max(real_scores)} time={timer()-start_timer}')
                if torch.min(real_scores) < eps:
                    l = real_x[real_scores<eps].tolist()
                    with open(file_path, 'a') as file:
                        for s in l:
                            file.write(fmt_array(s) + "\n")
                    if max_iterations == -1:
                        return
                #
                kaput = counter >= tolerance * eval_freq
                #x[kaput] *= 0.01 + 0.5*torch.rand((1,na),device=device)
                x[kaput] = (0.45 + 0.3*torch.rand((),device=device) + 0.1*torch.rand((1,na),device=device))*real_x[kaput]
                #x[kaput] = torch.sqrt(torch.rand((1,na),device=device))*real_x[kaput]
                print(f'{kaput.sum()} {counter.sum()}')
                counter.zero_()  # reset counter
                # added: try to keep min -- silly
                #i = torch.argmin(real_scores)
                #x[i]=real_x[i]
                #
                start_timer=timer()
            if t == max_iterations:
                return


batch_gradient_descent(arrays_tensor)
