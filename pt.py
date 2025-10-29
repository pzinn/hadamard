import torch
from params import na, nm, nn, score, device, fixed_sums, config

# parallel tempering
nT = 16  # number of temperatures. between say 10 and 20
r = .25 * nT  # log10 of ratio hiT/loT=1/loT. empirical formula

#torch.set_printoptions(threshold=nT,sci_mode=False)

#cnt = torch.zeros((nT,), dtype=torch.long, device=device)

@torch.inference_mode()
def improve_T(x, scores, k):  # random k-bit flip at finite T.
    #global cnt
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
    #cnt += accept.sum(dim=1)
    # broadcast the mask to k bits and gather flips to revert
    reject_mask_exp = (~accept).unsqueeze(2).expand(-1, -1, k)           # (nT,B,k)
    # reflip only rejected ones
    x.scatter_(2, flip_inds, torch.where(reject_mask_exp, flip_vals, -flip_vals))
    scores[accept] = scores_prop[accept]

@torch.inference_mode()
def improve_T_fixed_sums(x, scores, k):  # random k-bit rotate at finite T.
    #global cnt
    #nT, BperT, na = x.shape
    BperT=x.shape[1]
    j=torch.randint(nm, (), device=device)
    flip_inds = j*nn + torch.rand((nT, BperT, nn), device=device).topk(k, dim=2).indices
    # propose flips
    flip_vals = x.gather(2, flip_inds)
    flip_vals_rot = torch.roll(flip_vals,shifts=1,dims=2)
    x.scatter_(2, flip_inds, flip_vals_rot)
    scores_prop = score(x).view(nT, BperT)
    # Metropolis acceptance
    dE = scores_prop - scores
    accept = (dE < 0) | (torch.rand_like(dE) < torch.exp(-dE * invT[:, None]))
    #cnt += accept.sum(dim=1)
    # broadcast the mask to k bits and gather flips to revert
    reject_mask_exp = (~accept).unsqueeze(2).expand(-1, -1, k)           # (nT,B,k)
    # reflip only rejected ones
    x.scatter_(2, flip_inds, torch.where(reject_mask_exp, flip_vals, flip_vals_rot))
    scores[accept] = scores_prop[accept]

@torch.no_grad()
def attempt_swaps_vectorised(x, scores, gens):  # not used
    accepted = 0.
    total = 0
    for offset in (1,0):
        # 1. Prepare E and invT for adjacent pairs
        # E1, E2 are the scores for T_i and T_{i+1} across all (nT-1) pairs
        x1 = x[offset:-1:2]
        x2 = x[offset+1::2]
        s1 = scores[offset:-1:2]
        s2 = scores[offset+1::2]
        g1 = gens[offset:-1:2]
        g2 = gens[offset+1::2]
        # invT1, invT2 are the inverse temperatures for the same pairs
        invT1 = invT[offset:-1:2].unsqueeze(1)  # (nT-1, 1)
        invT2 = invT[offset+1::2].unsqueeze(1)   # (nT-1, 1)
        # 2. Calculate Acceptance Probability (Metropolis Criterion)
        # The term is: dE = (E2 - E1) * (invT1 - invT2)
        # invT are column vectors, E are matrices, ensuring correct broadcasting
        prob = torch.exp(-(s2 - s1) * (invT1 - invT2))  # (nT-1, BperT)
        # 3. Acceptance Mask
        swap_mask = (torch.rand_like(prob) < prob)  # (nT-1, BperT)
        # 4. Calculate Acceptance Rate
        accepted += swap_mask.sum().float()
        total += swap_mask.numel()
        # 5. Perform Swaps (In-place on x, scores, gens)
        # Expand mask for the swap operation across n and BperT
        swap_mask_x = swap_mask.unsqueeze(-1)  # (nT-1, BperT, 1)
        # Get the segments to be updated: x[i] and x[i+1] (the first and second halves)
        # Swap x
        x1_swapped = torch.where(swap_mask_x, x2, x1) # New states for T_i
        x2_swapped = torch.where(swap_mask_x, x1, x2) # New states for T_{i+1}
        x1.copy_(x1_swapped)
        x2.copy_(x2_swapped)
        # Swap scores
        s1_swapped = torch.where(swap_mask, s2, s1)
        s2_swapped = torch.where(swap_mask, s1, s2)
        s1.copy_(s1_swapped)
        s2.copy_(s2_swapped)
        # Swap gens
        g1_swapped = torch.where(swap_mask, g2, g1)
        g2_swapped = torch.where(swap_mask, g1, g2)
        g1.copy_(g1_swapped)
        g2.copy_(g2_swapped)
    return accepted / total

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
        """
        x1, x2 = x[i].clone(), x[i-1].clone()
        s1, s2 = E1.clone(), E2.clone()
        g1, g2 = gens[i].clone(), gens[i-1].clone()
        x[i] = torch.where(swap_mask, x2, x1)
        x[i-1] = torch.where(swap_mask, x1, x2)
        scores[i] = torch.where(accept, s2, s1)
        scores[i-1] = torch.where(accept, s1, s2)
        gens[i] = torch.where(accept, g2, g1)
        gens[i-1] = torch.where(accept, g1, g2)
        """
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
        if fixed_sums:
            k = 3 + 2 * torch.floor(torch.log(torch.rand(()))*invlogp)  # average k is (3-p)/(1-p)
            k = int(k.clamp(3,nn//2))
            improve_T_fixed_sums(x, scores, k=k)
        else:
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
