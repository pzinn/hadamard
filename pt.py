import torch
from params import na, score, device, config

# parallel tempering
nT = 16  # number of temperatures. between say 10 and 20
r = .25 * nT  # log10 of ratio hiT/loT=1/loT. empirical formula

#torch.set_printoptions(threshold=nT,sci_mode=False)

@torch.inference_mode()
def improve_T(x, scores, k):  # random k-bit flip at finite T.
    #nT, BperT, na = x.shape
    BperT=x.shape[1]
    #flip_inds = torch.randint(na, (nT, BperT, k), device=device)
    flip_inds = torch.rand((nT, BperT, na), device=device).topk(k, dim=2).indices
    # propose flips
    flip_vals = x.gather(2, flip_inds)
    x.scatter_(2, flip_inds, -flip_vals)
    scores_prop = score(x).view(nT, BperT)
    # Metropolis acceptance
    dE = scores_prop - scores
    accept = (dE < 0) | (torch.rand_like(dE) < torch.exp(-dE * invT[:, None]))
    # broadcast the mask to k bits and gather flips to revert
    reject_mask_exp = (~accept).unsqueeze(2).expand(-1, -1, k)           # (nT,B,k)
    # reflip only rejected ones
    x.scatter_(2, flip_inds, torch.where(reject_mask_exp, flip_vals, -flip_vals))
    scores[accept] = scores_prop[accept]

@torch.no_grad()
def attempt_swaps(x, scores, gens):
    #  x: (nT, BperT, n)
    #  scores, gens: (nT, BperT)
    accepted = 0
    total = 0
    for i in range(nT - 1, 0, -1):
        invT1, invT2 = invT[i], invT[i-1]
        E1, E2 = scores[i], scores[i-1]              # (BperT,)
        dE = (E2 - E1) * (invT1 - invT2)
        prob = torch.exp(-dE)
        accept = (torch.rand_like(prob) < prob)
        accepted += accept.sum()
        total += accept.numel()  # lazy
        # swap where accepted
        swap_mask = accept[:, None]            # (BperT,1)
        xc = x[i,accept].clone()
        x[i,accept] = x[i-1,accept]
        x[i-1,accept] = xc
        scoresc = scores[i,accept].clone()
        scores[i,accept] = scores[i-1,accept]
        scores[i-1,accept] = scoresc
        gensc = gens[i,accept].clone()
        gens[i,accept] = gens[i-1,accept]
        gens[i-1,accept] = gensc
    return accepted / total

swap_interval = 50
iterations = na * swap_interval * config.num_improve
p = .25
invlogp = 1 / torch.log(torch.tensor(p))
def parallel_tempering(x, scores, gens):
    global r, invT
    invT = torch.logspace(r, 0, nT, device=device)
    B = x.shape[0]
    BperT = B // nT  # should divide please
    x = x.view(nT, BperT, na)
    scores = scores.view(nT, BperT)
    gens = gens.view(nT, BperT)
    acc_mavg = 0.0
    for t in range(iterations):
        # --- local search per temperature ---
        k = 1 + torch.floor(torch.log(torch.rand(()))*invlogp)  # average k is 1/(1-p)
        k = int(k.clamp(1,na//2))
        improve_T(x, scores, k=k)
        # --- replica swaps ---
        if t % swap_interval == 0:
            acc = attempt_swaps(x, scores, gens)
            #print(f"{t:5d}: swap_acc={acc:.3f} flip_acc={(cnt/(swap_interval*BperT)).tolist()} mean score={scores.mean()} {scores.mean(1).tolist()}")
            #cnt.zero_()
            print(f"{t:5d}: swap_acc={acc:.3f} mean score={scores.mean():.3f} | "+' '.join(f'{v:.3f}' for v in scores.mean(1)))
            acc_avg = acc if t==0 else 0.9 * acc_avg + 0.1 * acc
    # autotune Ts
    if acc_avg < 0.2:
        r -= .2
    elif acc_avg > 0.3:
        r += .2
