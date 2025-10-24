import torch
from params import na, score, device

# parallel tempering
T_vals = torch.logspace(0, -2.5, 10, device=device)  # temperatures for tempering TODO choose wisely/autotune
nT = len(T_vals)

#torch.set_printoptions(threshold=nT,sci_mode=False)

@torch.inference_mode()
def improve_T(x, scores, k, T):  # random k-bit flip at finite T
    B=x.shape[0]
    flip_inds = torch.randint(0, na, (B, k), device=device)
    # propose flips
    x_prop = x.clone()
    x_prop.scatter_(1, flip_inds, -x.gather(1, flip_inds))
    scores_prop = score(x_prop)
    # Metropolis acceptance
    dE = scores_prop - scores
    accept = (dE < 0) | (torch.rand_like(dE) < torch.exp(-dE / T))
    x[accept] = x_prop[accept]
    scores[accept] = scores_prop[accept]

@torch.no_grad()
def attempt_swaps(x, scores):
    """
    x: (nT, B_per_T, n)
    scores: (nT, B_per_T)
    """
    accepted = 0
    total = 0
    for i in range(nT - 1):
        T1, T2 = T_vals[i], T_vals[i+1]
        E1, E2 = scores[i], scores[i+1]              # (B_per_T,)
        dE = (E2 - E1) * (1/T1 - 1/T2)
        prob = torch.exp(-dE)
        accept = (torch.rand_like(prob) < prob)
        accepted += accept.sum()
        total += accept.numel()  # lazy
        # swap where accepted
        if accept.any():
            swap_mask = accept[:, None]            # (B_per_T,1)
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
    B_per_T = B // nT  # should divide please
    scores = scores.view(nT, B_per_T)
    x = x.view(nT, B_per_T, na)
    acc_mavg = 0.0
    for t in range(iterations):
        k = 1 + (t // 100) % 3  # TODO fix
        # --- local search per temperature ---
        for i in range(nT):
            improve_T(x[i], scores[i], k=k, T=T_vals[i])
        #print((scores-score(x).view(nT,B_per_T)).abs().max())  # TESTING
        # --- replica swaps ---
        if t % swap_interval == 0:
            acc = attempt_swaps(x, scores)
            acc_mavg = 0.9 * acc_mavg + 0.1 * acc.item()  # exponential moving avg
            print(f"{t:5d}: swap_acc≈{acc_mavg:.3f} mean score={scores.mean()} {scores.mean(1)}")

