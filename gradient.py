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
eps=1e-5
max_iterations = 100
inner_steps = 0

def fmt_array(s):
    return "".join("+" if x > 0 else "-" for x in s)


def generate_random_arrays(batch_size):
    return 2 * torch.randint(2, (batch_size, na), device=device, dtype=score_type) - 1
    #return 2*(torch.rand((batch_size, na), device=device, dtype=score_type) - .5)



cst = 1 / math.sqrt(n)
one = torch.tensor([[1]],device=device,dtype=torch.float32)
def mir(a):
    return torch.cat((one.expand(a.shape[0],1),a,torch.flip(a,(1,))),dim=1)
def unfold(m0):
    return torch.stack((mir(m0[:,:nn2]),
                        mir(m0[:,nn2:2*nn2]),
                        mir(m0[:,2*nn2:3*nn2]),
                        m0[:,3*nn2:]),dim=1)
nm=4
def score0(m):
    f = cst * torch.fft.rfft(m, dim=2)  # cst there for accuracy
    ff = torch.real(f*f.conj())
    ffs = ff.sum(dim=1)
    s = torch.log(ffs)
    return -2*s[:,0]-4*s[:,1:].sum(dim=1)
def score(m0):
    return score0(unfold(m0))

@torch.inference_mode()
def improve1(arrays_tensor, scores):
    print(f"1 ", end=''); sys.stdout.flush()
    # first let's do it the stupidest way (will recode later)
    B = arrays_tensor.shape[0]
    active_mask = torch.ones(B, device=device, dtype=torch.bool)
    active_rows = torch.arange(B, device=device)
    while True:
        # rows we’re actively trying to improve this round
        M = active_rows.numel()
        # indices of best bit (−1 means no improvement found)
        inds = torch.full((M,), -1, dtype=torch.int64, device=device)
        # Try flipping each bit, keep any improvement
        for i in range(na):
            arrays_tensor[active_rows, i] *= -1          # flip bit i
            new_scores = score(arrays_tensor[active_rows])
            improved = new_scores < scores[active_rows]                  # where this flip helps
            if improved.any():
                scores[active_rows[improved]] = new_scores[improved]            # write into base `scores`
                inds[improved] = i
            arrays_tensor[active_rows, i] *= -1          # flip back
        # rows that actually improved this round
        improved_any = inds >= 0
        if not improved_any.any():
            print('')
            break
        # Apply the winning flips once
        active_rows = active_rows[improved_any]
        active_cols = inds[improved_any]
        arrays_tensor[active_rows, active_cols] *= -1
        # Next round: only keep rows that improved (they might improve again)
        active_mask.zero_()
        active_mask[active_rows] = True
        print(f'{active_mask.sum()/B} ',end='')

cf=1.25
def mod_score(m):
    #return score(torch.tanh(m))
    return score(m)+cf*torch.sum(m**2,dim=1)


if len(sys.argv) < 2:
    arrays_tensor = generate_random_arrays(max_samples)
    scores=score(arrays_tensor)
    mask=torch.isfinite(scores)
    arrays_tensor = arrays_tensor[mask]
    scores = scores[mask]
    print(f'{mask.sum()} {torch.min(scores)} {torch.mean(scores)} {torch.max(scores)}')
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
    B=x.shape[0]
    x.requires_grad_(True)
    scaler = torch.amp.GradScaler('cuda',enabled=mixed_precision)

    # opt = torch.optim.SGD([x], lr=lr)
    opt = torch.optim.AdamW([x], lr=lr)

    counter = torch.zeros(B, device=device, dtype=torch.int64)

    prev_scores = None
    for t in range(max_iterations):
        start_timer=timer()
        for _ in range(inner_steps):
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
        #real_x = torch.where((x+.1*(torch.rand(x.shape,device=device)-.05)) > 0, 1., -1.)
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
        kaput = counter >= tolerance * inner_steps
        tst = x[kaput].abs().min(dim=1)[0].mean()
        print(f'{tst=}')
        # x[kaput] = 2 * torch.rand((kaput.sum(), na), device=device, dtype=score_type) - 1  #baseline
        #x[kaput] *= 0.01 + 0.5*torch.rand((1,na),device=device)
        with torch.no_grad():
            if inner_steps==0:
                x.copy_(generate_random_arrays(B))  # for testing purposes
            else:
                x[kaput] = (0.4 + 0.1*torch.rand((),device=device) + 0.1*torch.rand((1,na),device=device))*real_x[kaput]
                #x[kaput] = 0.5*real_x[kaput]
        print(f'{kaput.sum()} {counter.sum()}')
        counter.zero_()  # reset counter
        # added: try to keep min -- silly
        #i = torch.argmin(real_scores)
        #x[i]=real_x[i]
        #


batch_gradient_descent(arrays_tensor)
