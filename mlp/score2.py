#!/usr/bin/env python
# coding: utf-8

# a naive attempt to get a vanilla neural network (MLP) to learn the score function (log det)

import torch
import torch.nn
import torch.nn.functional as F
import math
import os
from itertools import permutations
import sys

nm=4
nn=10
n=nm*nn
sample_size = 500000
# training params
max_steps = 10000
eval_freq = 500
batch_size = 1024
test_set_size = 1024
# model params
n_embd = 256
n_layer = 4

resume=False

debugging = False
#torch.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps=1e-5

# the final score
cst = 1 / math.sqrt(n)
@torch.no_grad()
def score(arrays):
    f = cst * torch.fft.rfft(arrays.view(-1, nm, nn), dim=2)  # cst there for accuracy
    # we do separately real pieces for accuracy reasons
    s = - torch.log(torch.real(f[:, :, 0].pow(2).sum(dim=1)))
    if nn % 2 == 0:
        s -= torch.log(torch.real(f[:, :, nn//2].pow(2).sum(dim=1)))
        f = f[:, :, 1:-1]
    else:
        f = f[:, :, 1:]
    ff = torch.imag(f[:, 1, :] * torch.conj(f[:, 0, :]) + f[:, 2, :] * torch.conj(f[:, 3, :]));  # ! note symmetries
    f.mul_(f.conj())
    s -= torch.log(torch.real(f).sum(dim=1).pow(2)-4*ff.pow(2)).sum(dim=1)
    s[s>n]=n
    #return 2*s  # usual normalisation
    return s


# the neural network
class MyGELU(torch.nn.Module):
    def forward(self, x):
        return x / (1.0 + torch.exp(-1.6*x))

"""
class makepos(torch.nn.Module):
    def forward(self, x):
        return x*x/(1+x*x/n)
"""

class MyEncoder(torch.nn.Module):
    #def __init__(self, W: torch.Tensor, b: torch.Tensor):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.cat((x,-x),dim=1))  # apparently this is called CReLU


class VNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        # added first convert a bit
        # W = torch.tensor([[(j&1)*2-1 if i==j>>1 else 0 for i in range(n)] for j in range(2*n)],dtype=torch.float32,device=device)
        # this is silly: W is mostly zeroes
        #layers.append(FixedLinear(W))
        layers.append(MyEncoder())
        #
        layers.append(torch.nn.Linear(2*n, n_embd))
        for _ in range(n_layer):
            #layers.append(torch.nn.ReLU(inplace=True))
            layers.append(MyGELU())
            layers.append(torch.nn.Linear(n_embd, n_embd))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(n_embd, 1))
        #layers.append(makepos())
        #layers.append(torch.nn.ReLU(inplace=True))
        layers.append(MyGELU())
        self.net = torch.nn.Sequential(*layers)
        n_params = sum(p.numel() for p in self.net.parameters())
        print("number of transformer parameters: %.2fM" % (n_params/1e6,))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)

model = VNet().to(device)
model = torch.compile(model)

work_dir = "./"  # TEMP
model_path = os.path.join(work_dir, "model.pt")


def save_model():
    print('saving model to workdir')
    torch.save(model.state_dict(), model_path)

def load_model():
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print('resuming from existing model in the workdir')

if resume:
    load_model()

perms = torch.tensor(list(p for p in permutations(range(nm)) if p[3] == 3), dtype=torch.long)
rndmod = torch.tensor([len(perms), 2*nn, 2*nn, 2, 2, 2, 2], dtype=torch.int64, device=device)
def rotate(array):
    array=array.view(-1,nm,nn)
    rnd = torch.remainder(torch.empty(rndmod.shape, dtype=torch.int64, device=device).random_(), rndmod)
    # symmetry: random permute
    array.copy_(array[:,perms[rnd[0]]])
    # symmetry: random rotation/flip
    array.copy_(torch.roll(array if rnd[1] < nn else torch.flip(array, (2,)), shifts=rnd[1].item(), dims=2))
    # symmetry: second rotation/flip
    array[:,3] = torch.roll(array[:,3] if rnd[2] < nn else torch.flip(array[:,3], (1,)), shifts=rnd[2].item(), dims=1)
    # symmetry: random signs
    array.mul_((rnd[3:7]*2-1).unsqueeze(1))


# training function
def train(data,scores,lr=1e-3):
    torch.set_float32_matmul_precision('high')  # dangerous, can cause NaN
    data_len=arrays.shape[0]
    # split into test and training data
    for i in range(test_set_size):
        j = torch.randint(i, data_len, ()).item()
        data[[i, j]] = data[[j, i]]
        scores[[i, j]] = scores[[j, i]]
    test_data = data[:test_set_size]
    train_data = data[test_set_size:]
    test_scores = scores[:test_set_size]
    train_scores = scores[test_set_size:]
    train_len = data_len - test_set_size
    print(f"split up the dataset into {train_len} training examples and {test_set_size} test examples")
    def get_lr(step, warmup_steps=5000):
        return lr * (.01+.99*step / warmup_steps if step < warmup_steps else 1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=get_lr(0), weight_decay=0.02, betas=(0.9, 0.99))
    step = 0
    #best_loss = F.smooth_l1_loss(model(test_data), test_scores)
    best_loss = F.l1_loss(model(test_data), test_scores)
    print(f"{step=} \ttest loss={best_loss.item():.4f}")
    while True:
        batch = torch.randint(train_len, (batch_size,))
        batch_arrays = train_data[batch]
        rotate(batch_arrays)
        batch_scores = train_scores[batch]  # TODO have an option to recompute on the fly
        v_pred = model(batch_arrays)
        #loss = F.smooth_l1_loss(v_pred, batch_scores)
        loss = F.l1_loss(v_pred, batch_scores)
        if not math.isfinite(loss):
            #raise RuntimeError(f"{step=} loss is NaN")
            print(f"{step=} loss is NaN")
            load_model()  # like that's going to help
            return
        optimiser.zero_grad(); loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # TODO rethink
        optimiser.step()
        step += 1
        for param_group in optimiser.param_groups:
            param_group['lr'] = get_lr(step)
        if step % eval_freq == 0 or step == max_steps:
            #test_loss = F.smooth_l1_loss(model(test_data), test_scores)
            test_loss = F.l1_loss(model(test_data), test_scores)
            print(f"{step=} \ttrain loss={loss.item():.4f} \ttest loss={test_loss.item():.4f}",end="\t")
            if test_loss < best_loss:
                save_model()
                best_loss = test_loss
            else:
                print("")
            if step == max_steps:
                break
    # restore best model
    load_model()
    return

# random sampling
def random_sample(size):
    arrays = 2 * torch.randint(2, (size, n), device=device, dtype=torch.float32) - 1
    scores = score(arrays)
    # added: randomly set to zero
    #p = 0.01  # probability of zeroing
    #mask = torch.rand_like(arrays) > p
    #arrays *= mask
    if debugging:
        print(arrays,scores)
    return arrays, scores

# greedy play: why is it so bad?
@torch.no_grad()
def greedy_play():
    a = torch.zeros(n,dtype=torch.float32,device=device)
    for k in range(n):
        with torch.no_grad():
            # compute scores of one-step completions (bellman) -- 2n possibilities, some excluded
            sc = torch.stack([ mod_score(loc,val,a.view(1,n),k==n-1).view(1) for loc in range(n) for val in [-1,1] ],dim=0)
            sc1, pick = torch.min(sc,dim=0)
            pick = pick.item()
            val=2*(pick&1)-1
            loc=pick>>1
            print(sc1.item(),loc,val)
            if debugging:
                print(a,sc.view(2*n),pick)
            a[loc]=val
    return a, sc1.item()

@torch.no_grad()
def mod_score(loc, val, arrays, flag):  # where and how to update, arrays, whether to use real score or model score
    scores = torch.full((arrays.shape[0],),torch.inf,dtype=torch.float32,device=device)
    mask = arrays[:,loc] == 0  # these are the arrays for which we can add something
    if torch.any(mask):
        arrays[mask,loc] = val
        masked_arrays = arrays[mask]
        masked_scores = score(masked_arrays) if flag else model(masked_arrays)
        scores[mask] = masked_scores
        arrays[mask,loc] = 0
    return scores

#improve
num_improve=5
na=n
def parallel_improve(arrays_tensor,scores):
    #if device.startswith('cuda'):
    #    torch.cuda.empty_cache()  # Free memory
    # step 1: this is the analogue of my old "simple_search2"
    for j in range(num_improve):
        if debugging:
            cnt = torch.tensor(0, device=device, dtype=torch.int64)
        print(f"1({j})", end=''); sys.stdout.flush()
        for i in range(na):
            arrays_tensor[:, i] *= -1  # Flip only the i-th bit
            # Compute new scores for all batch elements in parallel
            new_scores = score(arrays_tensor)
            # Identify which flips improved the score
            mask = new_scores < scores  # True where improvement happens
            if debugging:
                cnt += torch.sum(mask)
            # Apply successful bit flips
            arrays_tensor[~mask, i] *= -1  # Only revert for elements where no improvement
            scores[mask] = new_scores[mask]  # Update scores accordingly
        if debugging:
            print(f' improve success rate: {cnt/len(arrays_items)}')
    # step 2
    if debugging:
        cnt.zero_()
    print('2', end=''); sys.stdout.flush()
    for i in range(na):  # used to be na * num_improve
        a = torch.randint(na, ()).item()
        b = torch.randint(na, ()).item()
        if a > b:
            a, b = b, a
        # Flip selected bits for all arrays in batch
        arrays_tensor[:, a:b+1] *= -1
        # Compute new scores after flipping
        new_scores = score(arrays_tensor)
        # Identify improvements
        mask = new_scores < scores
        if debugging:
            cnt += torch.sum(mask)
        # Revert changes for arrays where score did not improve
        arrays_tensor[~mask, a:b+1] *= -1
        # Update scores where improvements occurred
        scores[mask] = new_scores[mask]
    if debugging:
        print(f' improve success rate: {cnt/len(arrays_items)}')


# main loop
step=0
beta=.1  # starting inverse temperature
while True:
    torch.set_float32_matmul_precision('highest')
    print(f"training {step}")
    if step<n:
        arrays, scores = random_sample(sample_size*(n-step)//n) # produces a pair of torch tensors, sequences and scores
    else:
        arrays = torch.zeros((0,n),dtype=torch.float32,device=device)
        scores = torch.zeros((0,),dtype=torch.float32,device=device)
        beta+=.1  # what's the right growth of beta?
    if step>0:
        arrays1 = torch.zeros((sample_size//n,n),dtype=torch.float32,device=device)
        for k in range(n):
            with torch.no_grad():
                # compute scores of one-step completions (bellman) -- 2n possibilities, some excluded
                scores1 = torch.stack([ mod_score(loc,val,arrays1,k==n-1) for loc in range(n) for val in [-1,1] ],dim=1)
                # mod_score = check if l/2 is available, if so score it using model if k<n-1, real score if k==n-1
                # we do 2 things with these scores: (1) we sample exp(-beta score) (2) we pick the min which is our new score
                probs=F.softmax(-beta*scores1,dim=1)
                scores1, inds = torch.min(scores1,dim=1)
                #
                if k>=n-step:
                    arrays=torch.cat((arrays,arrays1))  # TODO better?
                    scores=torch.cat((scores,scores1))
                #
                picks=torch.multinomial(probs,num_samples=1).view(-1) if k<n-1 else inds  # just pick optimal last step, no reason not to
                vals=(2*(picks&1)-1).to(dtype=torch.float32)
                rows=torch.arange(arrays1.shape[0])
                arrays1[rows,picks>>1]=vals
                #print(k,scores1[0].item(),inds[0].item(),picks[0].item(),vals[0].item(),arrays1[0].to(dtype=torch.int64).tolist())
                if debugging:
                    print(k,scores1,inds,probs,picks,vals,arrays1)
        # stats on final scores
        min_score=torch.min(scores1).item()
        mean_score=torch.mean(scores1).item()
        max_score=torch.max(scores1).item()
        print(f"{min_score=} {mean_score=} {max_score=}")
        if step>n:
            print("improving")
            parallel_improve(arrays1,scores1)
            min_score=torch.min(scores1).item()
            mean_score=torch.mean(scores1).item()
            max_score=torch.max(scores1).item()
            print(f"{min_score=} {mean_score=} {max_score=}")
            if min_score<eps:
                H_mask = scores1 < eps
                print(torch.where(arrays1[H_mask]>0,1,-1).tolist())
    # train on it
    train(arrays,scores)
    # for fun
    a, sc = greedy_play()
    print(f"greedy play={torch.where(a>0,1,-1).tolist()} {sc:.4f}")
    step += 1
