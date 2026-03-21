import torch
from params import na, nm, nn, score, device, fixed_sums, config, eps

# parallel tempering
nT = 16  # number of temperatures. between say 10 and 20
r = .25  # log10 of ratio of successive temperatures. empirical formula at n=188
# T = torch.logspace(0, -r * (nT-1), nT, device=device, dtype=torch.float32)
logT = r * torch.arange(nT, device=device, dtype=torch.float32)
T = .1 ** logT

@torch.inference_mode()
def improve_T(x, scores, k):  # random k-bit flip at finite T.
    BperT = x.shape[1]
    flip_inds = torch.rand((nT, BperT, na), device=device).topk(k, dim=2).indices
    # propose flips
    flip_vals = x.gather(2, flip_inds)
    x.scatter_(2, flip_inds, -flip_vals)
    scores_prop = score(x).view(nT, BperT)
    # Metropolis acceptance
    dE = scores_prop - scores
    #accept = (dE < 0) | (torch.rand_like(dE) < torch.exp(-dE * invT[:, None]))
    #accept = (scores > eps) & ((dE < 0) | (dE < -torch.log(torch.rand_like(dE)) * T[:,None]))  # remove first test?
    accept = (scores > eps) & (dE < -torch.log(torch.rand_like(dE)) * T[:,None])  # slower but simpler
    # broadcast the mask to k bits and gather flips to revert
    reject_mask_exp = (~accept).unsqueeze(2).expand(-1, -1, k)           # (nT,B,k)
    # reflip only rejected ones
    x.scatter_(2, flip_inds, torch.where(reject_mask_exp, flip_vals, -flip_vals))
    scores[accept] = scores_prop[accept]

@torch.inference_mode()
def improve_T_fixed_sums(x, scores, k):  # random k-bit rotate at finite T.
    BperT = x.shape[1]
    j = torch.randint(nm, (), device=device)
    flip_inds = j*nn + torch.rand((nT, BperT, nn), device=device).topk(k, dim=2).indices
    # propose flips
    flip_vals = x.gather(2, flip_inds)
    flip_vals_rot = torch.roll(flip_vals, shifts=1, dims=2)
    x.scatter_(2, flip_inds, flip_vals_rot)
    scores_prop = score(x).view(nT, BperT)
    # Metropolis acceptance
    dE = scores_prop - scores
    # accept = (dE < 0) | (torch.rand_like(dE) < torch.exp(-dE * invT[:, None]))
    accept = (dE < 0) | (dE < -torch.log(torch.rand_like(dE)) * T[:,None])  # remove first test?
    # broadcast the mask to k bits and gather flips to revert
    reject_mask_exp = (~accept).unsqueeze(2).expand(-1, -1, k)           # (nT,B,k)
    # reflip only rejected ones
    x.scatter_(2, flip_inds, torch.where(reject_mask_exp, flip_vals, flip_vals_rot))
    scores[accept] = scores_prop[accept]

@torch.no_grad()
def attempt_swaps(x, scores, gens):
    #  x: (nT, BperT, n)
    #  scores, gens: (nT, BperT)
    accepted = torch.zeros((nT-1,), device=device, dtype=torch.long)
    for i in range(nT-1):
        T1, T2 = T[i], T[i+1]
        E1, E2 = scores[i], scores[i+1]              # (BperT,)
        accept = (E1 - E2) * (T1 - T2) < -torch.log(torch.rand_like(E1)) * T1 * T2
        accepted[i] += accept.sum(dim=0)
        xc = x[i, accept].clone()
        x[i, accept] = x[i+1, accept]
        x[i+1, accept] = xc
        scoresc = scores[i, accept].clone()
        scores[i, accept] = scores[i+1, accept]
        scores[i+1, accept] = scoresc
        gensc = gens[i, accept].clone()
        gens[i, accept] = gens[i+1, accept]
        gens[i+1, accept] = gensc
    return accepted

swap_interval = 50
iterations = na * swap_interval * config.num_improve
p = .25
invlogp = 1 / torch.log(torch.tensor(p)).item()
def parallel_tempering(x, scores, gens):
    global logT, T
    B = x.shape[0]
    BperT = B // nT  # should divide please
    if BperT == 0:
        return
    x = x.view(nT, BperT, na)
    scores = scores.view(nT, BperT)
    gens = gens.view(nT, BperT)
    for t in range(iterations):
        # --- local search per temperature ---
        if fixed_sums:
            k = 3 + 2 * torch.floor(torch.log(torch.rand(()))*invlogp)  # average k is (3-p)/(1-p)
            k = int(k.clamp(3, nn//2))
            improve_T_fixed_sums(x, scores, k=k)
        else:
            k = 1 + torch.floor(torch.log(torch.rand(()))*invlogp)  # average k is 1/(1-p)
            k = int(k.clamp(1, na//2))
            improve_T(x, scores, k=k)
        # --- replica swaps ---
        if t % swap_interval == 0:
            acc = attempt_swaps(x, scores, gens) / BperT
            if t > swap_interval * 10:
                # autotune Ts
                for i in range(nT-1):
                    if acc[i] < .2:
                        logT[i+1:] -= .05 * (logT[i+1] - logT[i])
                    elif acc[i] > .3:
                        logT[i+1:] += .05 * (logT[i+1] - logT[i])
                T = .1 ** logT
            print(f"{t:5d}:  mean score = {scores.mean():6.3f}  swap acc = {acc.mean():6.3f}  T={T[0]:6.3f} : {scores[0].mean():6.3f}  T={T[nT-1]:6.3e} : {scores[nT-1].mean():6.3f}")
