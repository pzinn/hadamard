import math
import torch
from params import n, na, nm, nn, nn2, device, score, score_fft, fft, real_dtype, complex_dtype, eps
import sys

# precompute roots of unity for fft delta
cst = 1 / math.sqrt(n)
w = torch.exp(2j * torch.tensor(torch.pi, device=device, dtype=real_dtype) / nn)
rng0 = torch.arange(nn, device=device, dtype=real_dtype)
rng = torch.arange(nn2+1, device=device, dtype=real_dtype)
wrng = 2 * cst * w ** torch.outer(rng0, rng)
wrng1 = -torch.conj(wrng)
wrng_all = torch.zeros((na, nm*(nn2+1)), device=device, dtype=complex_dtype)
for i in range(nm):
    wrng_all[i*nn:(i+1)*nn, i*(nn2+1):(i+1)*(nn2+1)] = wrng1

k = 11
gray_code = [(i & -i).bit_length() - 1 for i in range(1, 1 << k)]
@torch.inference_mode()
def improve1p(arrays, scores):  # combined optimised 1-bit flip / opportunistic k-bit flip
    print(f"improve1p ", end=''); sys.stdout.flush()
    B = arrays.shape[0]
    active_rows = torch.nonzero(scores >= eps, as_tuple=True)[0]  # don't bother with H-matrices
    scores1 = torch.empty((B, na), device=device, dtype=real_dtype)
    while True:
        M = active_rows.numel()
        print(f'{M/B}')
        cur_rows = active_rows
        while True:
            f = fft(arrays[cur_rows])  # better than flip updating for accuracy
            fl = f.view(-1, nm*(nn2+1))
            fmod = torch.empty_like(f)
            flmod = fmod.view(-1, nm*(nn2+1))
            for j in range(na):
                torch.mul(arrays[cur_rows, j].to(complex_dtype).unsqueeze(1), wrng_all[j], out=flmod)
                flmod.add_(fl)
                scores1[cur_rows, j] = score_fft(fmod)
            mask = (scores1[cur_rows] < scores[cur_rows].unsqueeze(1)).any(dim=1)
            if not mask.any():
                break
            # easy ones: 1-bit flip.
            cur_rows = cur_rows[mask]
            min_scores, inds = scores1[cur_rows].min(dim=1)
            scores[cur_rows] = min_scores
            arrays[cur_rows, inds] *= -1
        # hard ones: brute force k best candidates
        _, indsk = torch.topk(scores1[active_rows], k, dim=1, sorted=False, largest=False)
        cur = torch.gather(arrays[active_rows], 1, indsk)
        f = fft(arrays[active_rows])
        fl = f.view(M, nm*(nn2+1))
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
                #arrays[improved_rows.unsqueeze(1).expand(-1,k),indsk[improved]] = cur[improved]  # ugly and slow
                arrays.index_put_((improved_rows.unsqueeze(1).expand(-1, k), indsk[improved]), cur[improved])
        if not mask.any():
            break
        active_rows = active_rows[mask]  # eliminate those that haven't been improved at all


# greedy random k-bit flip
p = .5
invlogp = 1 / math.log(p)
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
        k = 2 + torch.log(torch.rand(()))*invlogp
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
def improve_phases(arrays, scores):
    print(f"improve_phases ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    B = arrays.shape[0]
    f = fft(arrays)
    a = arrays.view(B, nm, nn)
    for j in range(nm):
        cnt.zero_()
        ff = f*f.conj()
        ffs1 = ff.sum(dim=1) - ff[:, j]
        inds = torch.nonzero((ffs1.real <= 1).all(dim=1), as_tuple=True)[0]
        M = inds.shape[0]
        if M == 0:
            continue
        h = f[inds, j] * torch.sqrt((1-ffs1[inds])/ff[inds, j])
        fmod = torch.empty((M, nn2+1), device=device, dtype=complex_dtype)
        x = torch.empty((M, nn), device=device, dtype=torch.int8)
        x2 = torch.empty((M, nn), device=device, dtype=real_dtype)
        for t in range(100*n):  # ?
            torch.fft.irfft(h, n=nn, dim=1, out=x2)  # should be a 1/cst but doesn't matter
            x.fill_(-1)
            x.masked_fill_(x2 > 0, 1)
            torch.fft.rfft(x, dim=1, out=fmod)
            fmod *= cst
            s = -2*torch.log(torch.real(ffs1[inds] + fmod*fmod.conj()))
            new_scores = s[:, 0]+2*s[:, 1:].sum(dim=1)
            improved = new_scores < scores[inds]
            improved_inds = inds[improved]
            a[improved_inds, j] = x[improved]
            scores[improved_inds] = new_scores[improved]
            f[improved_inds, j] = fmod[improved]
            cnt += improved_inds.shape[0]
            h[:, 1:] *= torch.exp(1j * 2 * torch.pi * torch.rand((M, nn2), device=device))
        print(f'({j}) {M} ({M/B}) {cnt} ({cnt/B})')

