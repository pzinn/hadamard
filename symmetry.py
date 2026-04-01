import math
from itertools import permutations
from types import SimpleNamespace

import torch

import params


def build_context(n=None, segment_sums=None, device=None, real_dtype=None):
    if n is None:
        n = params.n
        nn = params.nn
        na = params.na
        nn2 = params.nn2
        if segment_sums is None and getattr(params, "fixed_sums", False):
            segment_sums = params.segment_sums
    else:
        nn = n // 4
        na = 4 * nn
        nn2 = (nn - 1) // 2
    if device is None:
        device = params.device
    if real_dtype is None:
        real_dtype = params.real_dtype
    nm = 4
    fixed_sums = segment_sums is not None
    aut = torch.tensor([i for i in range(1, nn) if math.gcd(i, nn) == 1], device=device)
    perms = torch.tensor(
        list(
            p for p in permutations(range(nm))
            if not fixed_sums or tuple(segment_sums[i] for i in p) == segment_sums
        ),
        dtype=torch.long,
        device=device,
    )
    aut1 = aut[aut <= nn2]
    vec = torch.frac(torch.exp(0.1 * torch.arange(1, nn + 1, device=device, dtype=real_dtype)))
    fft_vec = torch.fft.rfft(vec)
    return SimpleNamespace(
        n=n,
        nm=nm,
        nn=nn,
        na=na,
        nn2=nn2,
        fixed_sums=fixed_sums,
        aut=aut,
        aut1=aut1,
        perms=perms,
        base=torch.arange(nn, device=device),
        fft_vec=fft_vec,
        fft_conj_vec=torch.conj(fft_vec),
        device=device,
    )


@torch.inference_mode()
def derotate(arrays, ctx, scores=None, score_fn=None, eps=None):
    if score_fn is not None:
        scores1 = score_fn(arrays)
        if scores is not None and (scores - scores1).abs().max() > eps:
            raise RuntimeError(
                "score incorrect",
                scores,
                scores1,
                (scores - scores1).abs().max().item(),
                (scores - scores1).abs().mean().item(),
            )
    B = arrays.shape[0]
    a = arrays.view(B, ctx.nm, ctx.nn)
    fft_a = torch.fft.rfft(a, dim=2)
    sp_rot = torch.fft.irfft(ctx.fft_conj_vec[None, None, :] * fft_a, n=ctx.nn, dim=2)
    sp_rev = torch.fft.irfft(ctx.fft_vec[None, None, :] * fft_a, n=ctx.nn, dim=2)
    sps = torch.cat([sp_rot, sp_rev], dim=2)
    if ctx.fixed_sums:
        flat_idx = sps.argmax(dim=2)
    else:
        flat_idx = sps.abs().argmax(dim=2)
    signed_base = torch.where(flat_idx >= ctx.nn, -1, 1).unsqueeze(-1) * ctx.base
    idx = (signed_base + flat_idx.unsqueeze(-1)) % ctx.nn
    transformed = a.gather(2, idx)
    if ctx.fixed_sums:
        a.copy_(transformed)
    else:
        chosen_sps = sps.gather(2, flat_idx.unsqueeze(-1)).squeeze(-1)
        a.copy_(torch.where(chosen_sps > 0, -1, 1).unsqueeze(-1) * transformed)
    if ctx.fixed_sums:
        candidates = a[:, ctx.perms, :].reshape(B, len(ctx.perms), ctx.na)
        perm_order = torch.arange(len(ctx.perms), device=ctx.device).expand(B, len(ctx.perms)).clone()
        for k in range(ctx.na - 1, -1, -1):
            key_in_order = candidates[:, :, k].gather(1, perm_order)
            ordk = torch.argsort(key_in_order, dim=1, stable=True)
            perm_order = perm_order.gather(1, ordk)
        best = candidates[torch.arange(B, device=ctx.device), perm_order[:, 0]].view(B, ctx.nm, ctx.nn)
        a.copy_(best)
    else:
        perm = torch.arange(ctx.nm, device=ctx.device).expand(B, ctx.nm).clone()
        for k in range(ctx.nn):
            key = a[:, :, k]
            key_in_curr_order = key.gather(1, perm)
            ordk = torch.argsort(key_in_curr_order, dim=1, stable=True)
            perm = perm.gather(1, ordk)
        sorted_a = a.gather(1, perm.unsqueeze(-1).expand(-1, -1, ctx.nn))
        a.copy_(sorted_a)
    if score_fn is not None:
        scores2 = score_fn(arrays)
        if (scores1 - scores2).abs().max() > eps:
            raise RuntimeError(
                "score not preserved by sort",
                scores1,
                scores2,
                (scores1 - scores2).abs().max().item(),
                (scores1 - scores2).abs().mean().item(),
            )


@torch.inference_mode()
def apply_aut(idx, arrays0, ctx):
    B = arrays0.shape[0]
    arrays04 = arrays0.view(B, ctx.nm, ctx.nn)
    arrays = torch.empty_like(arrays0)
    arrays4 = arrays.view(B, ctx.nm, ctx.nn)
    a = ctx.aut[idx]
    inds = torch.outer(a, torch.arange(ctx.nn, device=ctx.device)) % ctx.nn
    arrays4.scatter_(2, inds.unsqueeze(1).expand(B, ctx.nm, ctx.nn), arrays04)
    return arrays


@torch.inference_mode()
def find_aut_heuristic(arrays, ctx, fft_fn):
    f = fft_fn(arrays)
    f = f.abs().sum(dim=1)
    idx = f[:, ctx.aut1].argmax(dim=1)
    return apply_aut(idx, arrays, ctx)


@torch.inference_mode()
def find_aut_exact(arrays, ctx):
    B = arrays.shape[0]
    best = None
    for i in range(len(ctx.aut1)):
        idx = torch.full((B,), i, device=ctx.device, dtype=torch.long)
        candidate = apply_aut(idx, arrays, ctx)
        derotate(candidate, ctx)
        if best is None:
            best = candidate
            continue
        undecided = torch.ones(B, device=ctx.device, dtype=torch.bool)
        better = torch.zeros(B, device=ctx.device, dtype=torch.bool)
        for k in range(ctx.na):
            rows = torch.nonzero(undecided, as_tuple=True)[0]
            if rows.numel() == 0:
                break
            cand_k = candidate[rows, k]
            best_k = best[rows, k]
            gt = cand_k > best_k
            lt = cand_k < best_k
            better[rows[gt]] = True
            undecided[rows[gt | lt]] = False
        best[better] = candidate[better]
    return best
