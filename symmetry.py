import math
from itertools import permutations
from types import SimpleNamespace

import torch

import params


def build_context(nn=None, nm=None, segment_sums=None, device=None, real_dtype=None):
    if nn is None:
        nn = params.nn
        na = params.na
        nn2 = params.nn2
        nm = params.nm
        if segment_sums is None and getattr(params, "fixed_sums", False):
            segment_sums = params.segment_sums
    else:
        if nm is None:
            raise ValueError("build_context requires nm when nn is provided")
        na = nm * nn
        nn2 = (nn - 1) // 2
    if device is None:
        device = params.device
    if real_dtype is None:
        real_dtype = params.real_dtype
    fixed_sums = segment_sums is not None
    aut_values = [1] if nn == 1 else [i for i in range(1, nn) if math.gcd(i, nn) == 1]
    aut = torch.tensor(aut_values, device=device)
    perms = torch.tensor(
        list(
            p for p in permutations(range(nm))
            if not fixed_sums or tuple(segment_sums[i] for i in p) == segment_sums
        ),
        dtype=torch.long,
        device=device,
    )
    aut1 = aut[aut <= nn2]
    if aut1.numel() == 0:
        aut1 = aut[:1]
    vec = torch.frac(torch.exp(0.1 * torch.arange(1, nn + 1, device=device, dtype=real_dtype)))
    fft_vec = torch.fft.rfft(vec)
    return SimpleNamespace(
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
        real_dtype=real_dtype,
    )


@torch.inference_mode()
def canonicalise_local_symmetry(arrays, ctx, scores=None, score_fn=None, eps=None):
    # Canonicalise the non-automorphism symmetries: blockwise dihedral actions,
    # optional sign changes, and allowed block permutations.
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
def _apply_aut_heuristic(arrays, ctx, fft_fn):
    f = fft_fn(arrays)
    f = f.abs().sum(dim=1)
    idx = f[:, ctx.aut1].argmax(dim=1)
    return apply_aut(idx, arrays, ctx)




@torch.inference_mode()
def count_local_symmetry_matches(target_arrays, source_arrays, ctx):
    B = target_arrays.shape[0]
    target = target_arrays.view(B, ctx.nm, ctx.nn).to(dtype=ctx.real_dtype)
    source = source_arrays.view(B, ctx.nm, ctx.nn).to(dtype=ctx.real_dtype)
    target_fft = torch.fft.rfft(target, dim=2)
    source_fft = torch.fft.rfft(source, dim=2)
    source_rev_fft = torch.fft.rfft(source.flip(-1), dim=2)
    corr_rot = torch.fft.irfft(target_fft[:, :, None, :].conj() * source_fft[:, None, :, :], n=ctx.nn, dim=3).round().to(torch.int16)
    corr_rev = torch.fft.irfft(target_fft[:, :, None, :].conj() * source_rev_fft[:, None, :, :], n=ctx.nn, dim=3).round().to(torch.int16)
    if ctx.fixed_sums:
        pair_counts = (corr_rot == ctx.nn).sum(dim=3) + (corr_rev == ctx.nn).sum(dim=3)
    else:
        pair_counts = (corr_rot.abs() == ctx.nn).sum(dim=3) + (corr_rev.abs() == ctx.nn).sum(dim=3)
    per_perm = pair_counts[:, None, torch.arange(ctx.nm, device=ctx.device), ctx.perms]
    return per_perm.prod(dim=3).sum(dim=2).squeeze(1).to(torch.long)


@torch.inference_mode()
def stabiliser_orders(arrays, ctx):
    arrays = arrays.view(-1, ctx.na)
    B = arrays.shape[0]
    orders = torch.zeros(B, dtype=torch.long, device=ctx.device)
    arrays4 = arrays.view(B, ctx.nm, ctx.nn)
    for a in ctx.aut1.tolist():
        transformed = torch.empty_like(arrays)
        transformed4 = transformed.view(B, ctx.nm, ctx.nn)
        inds = (a * torch.arange(ctx.nn, device=ctx.device)) % ctx.nn
        transformed4.scatter_(2, inds.view(1, 1, ctx.nn).expand(B, ctx.nm, ctx.nn), arrays4)
        orders += count_local_symmetry_matches(arrays, transformed, ctx)
    return orders

@torch.inference_mode()
def canonicalise_heuristic(arrays, ctx, fft_fn, scores=None, score_fn=None, eps=None):
    arrays = _apply_aut_heuristic(arrays, ctx, fft_fn)
    canonicalise_local_symmetry(arrays, ctx, scores, score_fn, eps)
    return arrays


@torch.inference_mode()
def canonicalise_automorphism_exact(arrays, ctx):
    B = arrays.shape[0]
    best = None
    for i in range(len(ctx.aut1)):
        idx = torch.full((B,), i, device=ctx.device, dtype=torch.long)
        candidate = apply_aut(idx, arrays, ctx)
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


@torch.inference_mode()
def canonicalise_exact(arrays, ctx):
    B = arrays.shape[0]
    best = None
    for i in range(len(ctx.aut1)):
        idx = torch.full((B,), i, device=ctx.device, dtype=torch.long)
        candidate = apply_aut(idx, arrays, ctx)
        canonicalise_local_symmetry(candidate, ctx)
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
