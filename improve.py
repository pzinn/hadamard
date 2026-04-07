import torch
import params
from params import n, na, nm, nn, nn2, device, score, score_fft, score_fft_int, fft, fixed_sums, num_ones, real_dtype, complex_dtype, eps, gen_decay, cst, segment_sums
from timestamped_print import print
import sys

# precompute roots of unity for fft delta
w = torch.exp(2j * torch.tensor(torch.pi, device=device, dtype=real_dtype) / nn)
rng0 = torch.arange(nn, device=device, dtype=real_dtype)
rng = torch.arange(nn2+1, device=device, dtype=real_dtype)
wrng = 2 * cst * w ** torch.outer(rng0, rng)
wrng1 = -torch.conj(wrng)
wrng_all = torch.zeros((na, nm*(nn2+1)), device=device, dtype=complex_dtype)
for i in range(nm):
    wrng_all[i*nn:(i+1)*nn, i*(nn2+1):(i+1)*(nn2+1)] = wrng1

k = min(11, na)
gray_code = [(i & -i).bit_length() - 1 for i in range(1, 1 << k)]
@torch.inference_mode()
def improve1p(arrays, scores):  # optimised k-bit flip
    print(f"improve1p ", end=''); sys.stdout.flush()
    B = arrays.shape[0]
    active_rows = torch.nonzero(scores >= eps, as_tuple=True)[0]  # don't bother with H-matrices
    while True:
        M = active_rows.numel()
        print(f'{M/B}')
        scores1 = torch.empty((M, na), device=device, dtype=real_dtype)
        f = fft(arrays[active_rows])  # better than flip updating for accuracy
        fl = f.view(M, nm*(nn2+1))
        fmod = torch.empty_like(f)
        flmod = fmod.view(-1, nm*(nn2+1))
        for j in range(na):
            torch.mul(arrays[active_rows, j].to(complex_dtype).unsqueeze(1), wrng_all[j], out=flmod)
            flmod.add_(fl)
            scores1[:, j] = score_fft(fmod)
        # k best flip candidates
        _, indsk = torch.topk(scores1, k, dim=1, sorted=False, largest=False)
        cur = torch.gather(arrays[active_rows], 1, indsk)
        mask = torch.zeros((M,), device=device, dtype=torch.bool)
        for j in gray_code:
            inds = indsk[:, j]  # actual index for each sample
            fl += cur[:, j].unsqueeze(1) * wrng_all[inds]
            cur[:, j] *= -1  # need to keep track of these two
            new_scores = score_fft(f)
            improved = new_scores < scores[active_rows]
            if improved.any():
                mask[improved] = True  # these will get saved for next round
                improved_rows = active_rows[improved]
                scores[improved_rows] = new_scores[improved]
                # arrays[improved_rows.unsqueeze(1).expand(-1,k),indsk[improved]] = cur[improved]  # ugly and slow
                arrays.index_put_((improved_rows.unsqueeze(1).expand(-1, k), indsk[improved]), cur[improved])
        if not mask.any():
            break
        active_rows = active_rows[mask]  # eliminate those that haven't been improved at all

if segment_sums is not None:
    ss = torch.tensor([cst*segment_sums[j] for j in range(nm)], dtype=real_dtype, device=device)
def penalty(f):  # penalty to stray from correct segment sums
    return torch.abs(torch.real(f[:,:,0])-ss[None,:]).sum(dim=1)
def mod_score_fft(f, z):
    return score_fft(f) + z * penalty(f)
zmul = 1.5  # adjustable parameter
@torch.inference_mode()
def improve1p_fixed(arrays, scores):  # optimised k-bit flip -- progressively enforcing segment_sums
    print(f"improve1p_fixed ", end=''); sys.stdout.flush()
    z = cst  # is that the correct scaling with n?
    oldz = 0
    B = arrays.shape[0]
    active_rows = torch.nonzero(scores >= eps, as_tuple=True)[0]  # don't bother with H-matrices
    while True:
        M = active_rows.numel()
        scores1 = torch.empty((M, na), device=device, dtype=real_dtype)
        f = fft(arrays[active_rows])  # better than flip updating for accuracy
        pen = penalty(f)
        mask = pen > eps  # always continue with ones violating segment_sums
        print(f"{M/B} {mask.sum()/B}")
        scores[active_rows] += (z-oldz) * pen  # adjust scores to new value of z
        fl = f.view(M, nm*(nn2+1))
        fmod = torch.empty_like(f)
        flmod = fmod.view(-1, nm*(nn2+1))
        for j in range(na):
            torch.mul(arrays[active_rows, j].to(complex_dtype).unsqueeze(1), wrng_all[j], out=flmod)
            flmod.add_(fl)
            scores1[:, j] = mod_score_fft(fmod, z)
        # k best flip candidates
        _, indsk = torch.topk(scores1, k, dim=1, sorted=False, largest=False)
        cur = torch.gather(arrays[active_rows], 1, indsk)
        for j in gray_code:
            inds = indsk[:, j]  # actual index for each sample
            fl += cur[:, j].unsqueeze(1) * wrng_all[inds]
            cur[:, j] *= -1  # need to keep track of these two
            new_scores = mod_score_fft(f, z)
            improved = new_scores < scores[active_rows]
            if improved.any():
                mask[improved] = True  # these will get saved for next round
                improved_rows = active_rows[improved]
                scores[improved_rows] = new_scores[improved]
                # arrays[improved_rows.unsqueeze(1).expand(-1,k),indsk[improved]] = cur[improved]  # ugly and slow
                arrays.index_put_((improved_rows.unsqueeze(1).expand(-1, k), indsk[improved]), cur[improved])
        if not mask.any():
            break
        active_rows = active_rows[mask]  # eliminate those that haven't been improved at all
        oldz = z
        z *= zmul

# greedy random k-bit flip
p = .5
invlogp = 1 / torch.log(torch.tensor(p, device=device, dtype=real_dtype)).item()
@torch.inference_mode()
def improve_greedy(x, scores):
    print("improve_greedy ", end=''); sys.stdout.flush()
    B = x.shape[0]
    # precompute fft
    f = fft(x)
    fl = f.view(B, nm*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B, nm*(nn2+1))
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for _ in range(50):  #?
        k = 2 + torch.log(torch.rand((), device=device, dtype=real_dtype))*invlogp
        k = int(k.clamp(2, na//2))
        n_inds = na // k
        # create all at once a bunch of subsets to sample
        all_inds = torch.randperm(na, device=device)[:n_inds*k].view(n_inds,k)
        for i in range(n_inds):
            inds = all_inds[i]
            torch.matmul(x[:, inds].to(complex_dtype), wrng_all[inds], out=flmod)
            flmod.add_(fl)
            new_scores = score_fft(fmod)
            improved_inds = torch.nonzero(new_scores < scores, as_tuple=True)[0]  # better than mask when few True expected
            fl[improved_inds] = flmod[improved_inds]
            x[improved_inds.unsqueeze(1), inds] *= -1
            scores[improved_inds] = new_scores[improved_inds]
            cnt += improved_inds.shape[0]
    print(f'{cnt} ({cnt/B})')

@torch.inference_mode()
def improve_greedy_fixed(x, scores):
    print("improve_greedy_fixed ", end=''); sys.stdout.flush()
    B = x.shape[0]
    # precompute fft
    f = fft(x)
    fl = f.view(B, nm*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B, nm*(nn2+1))
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for _ in range(300):  #?
        j = torch.randint(nm, (1,1), device=device)
        k = 3 + 2 * torch.floor(torch.log(torch.rand((), device=device, dtype=real_dtype))*.5*invlogp)
        k = int(k.clamp(3, nn//2))
        n_inds = nn // k
        # create all at once a bunch of subsets to sample
        all_inds = j*nn + torch.randperm(nn, device=device)[:n_inds*k].view(n_inds,k)
        for i in range(n_inds):
            inds = all_inds[i]
            xx = .5*x[:, inds].to(complex_dtype)
            w = wrng_all[inds]
            xx2 = xx.clone()
            changed = torch.zeros(B, device=device, dtype=torch.bool)
            for _ in range(k-1):
                xx2 = torch.roll(xx2, shifts=1, dims=1)
                torch.matmul(xx-xx2, w, out=flmod)
                flmod.add_(fl)
                new_scores = score_fft(fmod)
                improved_inds = torch.nonzero(new_scores < scores, as_tuple=True)[0]  # better than mask when few True expected
                fl[improved_inds] = flmod[improved_inds]
                xx[improved_inds] = xx2[improved_inds]
                scores[improved_inds] = new_scores[improved_inds]
                changed[improved_inds] = True
                cnt += improved_inds.shape[0]
            changed_inds = torch.nonzero(changed, as_tuple=True)[0]
            x[changed_inds.unsqueeze(1), inds] = (2*xx[changed_inds]).real.to(torch.int8)
    print(f'{cnt} ({cnt/B})')

sw0 = torch.tensor([[-1, -1, 1, 1], [-1, 1, -1, 1], [-1, 1, 1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1]], device=device, dtype=torch.int8)
psw, ksw = sw0.shape  # psw = ksw choose ksw/2
sw_grids = torch.meshgrid(*[torch.arange(psw, device=device) for _ in range(nm)], indexing='ij')
sw_idx = torch.stack(sw_grids, dim=-1).reshape(-1, nm)    # (p^nm, k)
sw = sw0[sw_idx].reshape(-1, nm * ksw)

@torch.inference_mode()
def improve_phases(arrays, scores):
    print(f"improve_phases ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    B = arrays.shape[0]
    f = fft(arrays)
    a = arrays.view(B, nm, nn)
    for j in range(nm):
        cnt.zero_()
        ff = torch.view_as_real(f).square().sum(dim=-1)
        ffs1 = ff.sum(dim=1) - ff[:, j]
        inds = torch.nonzero(((ffs1 <= 1) & (ff[:, j] > 0)).all(dim=1), as_tuple=True)[0]
        M = inds.shape[0]
        if M == 0:
            continue
        h = f[inds, j] * torch.sqrt((1-ffs1[inds])/ff[inds, j])
        fmod = torch.empty((M, nn2+1), device=device, dtype=complex_dtype)
        x = torch.empty((M, nn), device=device, dtype=torch.int8)
        x2 = torch.empty((M, nn), device=device, dtype=real_dtype)
        for t in range(100*na):  #?
            torch.fft.irfft(h, n=nn, dim=1, out=x2)  # should be a 1/cst but doesn't matter
            x.fill_(-1)
            if fixed_sums:
                x.scatter_(1, torch.topk(x2, num_ones[j], dim=1).indices, 1)
            else:
                x.masked_fill_(x2 > 0, 1)
            torch.fft.rfft(x, dim=1, out=fmod)
            fmod *= cst
            new_scores = score_fft_int(ffs1[inds] + torch.view_as_real(fmod).square().sum(dim=-1))
            improved = new_scores < scores[inds]
            improved_inds = inds[improved]
            a[improved_inds, j] = x[improved]
            scores[improved_inds] = new_scores[improved]
            f[improved_inds, j] = fmod[improved]
            cnt += improved_inds.shape[0]
            h[:, 1:] *= torch.exp(1j * (torch.rand((M, nn2), device=device)-.5))
        print(f'({j}) {M} ({M/B}) {cnt} ({cnt/B})')

@torch.inference_mode()
def improve4x4_fixed(x, scores):  # optimal 4x4 bit switch
    print(f"improve4x4_fixed ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    B = x.shape[0]
    f = fft(x)
    fl = f.view(B, nm*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B, nm*(nn2+1))
    # first find nmx(2+2) locations for optimal flips
    best_scores = torch.full((B, nm, 2, 2), float('inf'), dtype=real_dtype, device=device)
    inds = torch.full((B, nm, 2, 2), -1, dtype=torch.long, device=device)
    for j in range(nm):
        r1 = j*nn
        r = nn
        r2 = r+r1
        for i in range(r1, r2):
            torch.mul(x[:, i].to(complex_dtype).unsqueeze(1), wrng_all[i], out=flmod)
            flmod.add_(fl)
            scores1 = score_fft(fmod)
            mask = x[:, i] > 0
            minuses = torch.nonzero(~mask, as_tuple=True)[0]
            pluses = torch.nonzero(mask, as_tuple=True)[0]
            # update minuses
            cand = scores1[minuses]
            mask = cand.unsqueeze(1) < best_scores[minuses, j, 0]
            minuses1 = minuses[mask[:, 0]]
            best_scores[minuses1, j, 0, 1] = best_scores[minuses1, j, 0, 0]
            best_scores[minuses1, j, 0, 0] = cand[mask[:, 0]]
            inds[minuses1, j, 0, 1] = inds[minuses1, j, 0, 0]
            inds[minuses1, j, 0, 0] = i
            minuses1 = minuses[~mask[:, 0] & mask[:, 1]]
            best_scores[minuses1, j, 0, 1] = cand[~mask[:, 0] & mask[:, 1]]
            inds[minuses1, j, 0, 1] = i
            # update pluses
            cand = scores1[pluses]
            mask = cand.unsqueeze(1) < best_scores[pluses, j, 1]
            pluses1 = pluses[mask[:, 0]]
            best_scores[pluses1, j, 1, 1] = best_scores[pluses1, j, 1, 0]
            best_scores[pluses1, j, 1, 0] = cand[mask[:, 0]]
            inds[pluses1, j, 1, 1] = inds[pluses1, j, 1, 0]
            inds[pluses1, j, 1, 0] = i
            pluses1 = pluses[~mask[:, 0] & mask[:, 1]]
            best_scores[pluses1, j, 1, 1] = cand[~mask[:, 0] & mask[:, 1]]
            inds[pluses1, j, 1, 1] = i
    # now try every combo
    inds = inds.view(B, nm*ksw)
    base = torch.arange(B, device=device)
    cur = torch.gather(x, 1, inds)
    for i in range(sw.shape[0]):
        x[base.unsqueeze(1), inds] = sw[i]
        new_scores = score(x)
        improved = new_scores < scores
        scores[improved] = new_scores[improved]
        cur[improved] = sw[i]
        cnt += torch.sum(improved)
    x.scatter_(1, inds, cur)
    print(f'{cnt/B}')
