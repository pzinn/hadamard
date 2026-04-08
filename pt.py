import torch
from params import na, nm, nn, score, device, fixed_sums, config, eps, real_dtype, verbose
from timestamped_print import print

# parallel tempering
nT = 16  # number of temperatures. between say 10 and 20
r = .25  # log10 of ratio of successive temperatures. empirical formula at n=188
# T = torch.logspace(0, -r * (nT-1), nT, device=device, dtype=torch.float32)
logT = r * torch.arange(nT, device=device, dtype=real_dtype)
T = .1 ** logT

@torch.inference_mode()
def improve_T(x, scores, k):  # random k-bit flip at finite T.
    BperT = x.shape[1]
    flip_inds = torch.randint(na, (nT, BperT, k), device=device)
    flip_mask = torch.zeros((nT, BperT, na), device=device, dtype=torch.bool)
    flip_mask.scatter_(2, flip_inds, True)
    x[flip_mask] *= -1
    scores_prop = score(x).view(nT, BperT)
    dE = scores_prop - scores
    accept = (scores > eps) & (dE < -torch.log(torch.rand_like(dE)) * T[:, None])
    reject_mask = flip_mask & (~accept).unsqueeze(2)
    x[reject_mask] *= -1
    scores[accept] = scores_prop[accept]

@torch.inference_mode()
def improve_T_fixed_sums(x, scores, k):  # random k-bit rotate at finite T.
    BperT = x.shape[1]
    j = torch.randint(nm, (), device=device)
    base_inds = torch.rand(nn, device=device, dtype=real_dtype).topk(k).indices.view(1, 1, k)
    shifts = torch.randint(nn, (nT, BperT, 1), device=device)
    flip_inds = j * nn + (base_inds + shifts) % nn
    flip_vals = x.gather(2, flip_inds)
    flip_vals_rot = torch.roll(flip_vals, shifts=1, dims=2)
    x.scatter_(2, flip_inds, flip_vals_rot)
    scores_prop = score(x).view(nT, BperT)
    dE = scores_prop - scores
    accept = (dE < 0) | (dE < -torch.log(torch.rand_like(dE)) * T[:, None])
    reject_mask_exp = (~accept).unsqueeze(2).expand(-1, -1, k)
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

@torch.no_grad()
def attempt_swaps_vectorised(x, scores, gens):  # not used. faster but performs slightly worse
    #  x: (nT, BperT, n)
    #  scores, gens: (nT, BperT)
    accepted = torch.zeros((nT-1,), device=device, dtype=torch.long)
    for parity in range(2):  # even-odd decomposition: independent pairs can swap in parallel
        idx = torch.arange(parity, nT-1, 2, device=device)
        if idx.numel() == 0:
            continue
        T1 = T[idx]                                     # (P,)
        T2 = T[idx + 1]                                 # (P,)
        E1 = scores[idx]                                # (P, BperT)
        E2 = scores[idx + 1]                            # (P, BperT)
        accept = (E1 - E2) * (T1 - T2).unsqueeze(1) < -torch.log(torch.rand_like(E1)) * (T1 * T2).unsqueeze(1)
        accepted[idx] = accept.sum(dim=1)
        accept_3d = accept.unsqueeze(2)                 # (P, BperT, 1) for broadcasting over na
        x_lo, x_hi = x[idx].clone(), x[idx + 1].clone()
        x[idx]     = torch.where(accept_3d, x_hi, x_lo)
        x[idx + 1] = torch.where(accept_3d, x_lo, x_hi)
        s_lo, s_hi = scores[idx].clone(), scores[idx + 1].clone()
        scores[idx]     = torch.where(accept, s_hi, s_lo)
        scores[idx + 1] = torch.where(accept, s_lo, s_hi)
        g_lo, g_hi = gens[idx].clone(), gens[idx + 1].clone()
        gens[idx]     = torch.where(accept, g_hi, g_lo)
        gens[idx + 1] = torch.where(accept, g_lo, g_hi)
    return accepted

p = .5
invlogp = 1 / torch.log(torch.tensor(p, device=device, dtype=real_dtype)).item()
def parallel_tempering(x, scores, gens):
    print("parallel_tempering", flush=True)
    global logT, T
    iterations = na * 50 * config.num_improve
    avg_k = (3-p)/(1-p) if fixed_sums else 1/(1-p)
    swap_interval = max(na//avg_k, 10)
    warmup_swaps = 5 * swap_interval
    B = x.shape[0]
    assert B % nT == 0, f"parallel_tempering requires batch size divisible by {nT}, got {B}"
    BperT = B // nT
    if BperT == 0:
        return
    x = x.view(nT, BperT, na)
    scores = scores.view(nT, BperT)
    gens = gens.view(nT, BperT)
    for t in range(iterations):
        # --- local search per temperature ---
        if fixed_sums:
            k = 3 + 2 * torch.floor(torch.log(torch.rand((), device=device, dtype=real_dtype))*invlogp)
            k = int(k.clamp(3, nn//2))
            improve_T_fixed_sums(x, scores, k=k)
        else:
            k = 1 + torch.floor(torch.log(torch.rand((), device=device, dtype=real_dtype))*invlogp)
            k = int(k.clamp(1, na//2))
            improve_T(x, scores, k=k)
        # --- replica swaps ---
        if t % swap_interval == 0:
            acc = attempt_swaps(x, scores, gens) / BperT
            if t > warmup_swaps:
                # autotune Ts
                for i in range(nT-1):
                    if acc[i] < .2:
                        logT[i+1:] -= .05 * (logT[i+1] - logT[i])
                    elif acc[i] > .3:
                        logT[i+1:] += .05 * (logT[i+1] - logT[i])
                T = .1 ** logT
            if verbose:
                print(f"{t:5d}:  mean score = {scores.mean():6.3f}  swap acc = {acc.mean():6.3f}  T={T[0]:6.3f} : {scores[0].mean():6.3f}  T={T[nT-1]:6.3e} : {scores[nT-1].mean():6.3f}")
