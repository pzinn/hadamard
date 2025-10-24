import torch
from params import na, score, device

# parallel tempering
invT = torch.logspace(2.5, 0, 10, device=device)  # inverse temperatures for tempering TODO choose wisely/autotune
nT = len(invT)

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
def attempt_swaps(x, scores):
    """
    x: (nT, BperT, n)
    scores: (nT, BperT)
    """
    accepted = 0
    total = 0
    for i in range(nT - 1):
        invT1, invT2 = invT[i], invT[i+1]
        E1, E2 = scores[i], scores[i+1]              # (BperT,)
        dE = (E2 - E1) * (invT1 - invT2)
        prob = torch.exp(-dE)
        accept = (torch.rand_like(prob) < prob)
        accepted += accept.sum()
        total += accept.numel()  # lazy
        # swap where accepted
        if accept.any():
            swap_mask = accept[:, None]            # (BperT,1)
            x1, x2 = x[i].clone(), x[i+1].clone()
            s1, s2 = E1.clone(), E2.clone()
            x[i] = torch.where(swap_mask, x2, x1)
            x[i+1] = torch.where(swap_mask, x1, x2)
            scores[i] = torch.where(accept, s2, s1)
            scores[i+1] = torch.where(accept, s1, s2)
    acc_rate = accepted.float() / total
    return acc_rate

def optimise_parallel_tempering(x, scores, iterations=5000, swap_interval=50):
    B = x.shape[0]
    BperT = B // nT  # should divide please
    scores = scores.view(nT, BperT)
    x = x.view(nT, BperT, na)
    acc_mavg = 0.0
    for t in range(iterations):
        k = 1 + t % 3  # TODO fix
        # --- local search per temperature ---
        improve_T(x, scores, k=k)
        #print((scores-score(x).view(nT,BperT)).abs().max())  # TESTING
        # --- replica swaps ---
        if t % swap_interval == 0:
            acc = attempt_swaps(x, scores)
            print(f"{t:5d}: swap_acc≈{acc:.3f} mean score={scores.mean()} {scores.mean(1).tolist()}")

