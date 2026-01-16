import math
import torch
import params
from params import n, na, nm, nn, nn2, device, score, score_fft, score_fft_int, fft, fixed_sums, num_ones, real_dtype, complex_dtype, eps, gen_decay
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

@torch.inference_mode()
def improve_tabu(arrays, scores, gens):
    print(f"improve_tabu ", end=''); sys.stdout.flush()
    B = arrays.shape[0]
    active_rows = torch.nonzero((scores >= eps) & (gens < params.gen) & (torch.rand((B,), device=device) < gen_decay), as_tuple=True)[0]  # don't bother with H-matrices, recent ones
    M = active_rows.numel()
    if M == 0:
        return
    x = arrays[active_rows]
    s = torch.empty((M,), dtype=real_dtype, device=device)
    s1 = torch.empty((M,), dtype=real_dtype, device=device)
    tabu = torch.ones((M, na), dtype=torch.bool, device=device)
    free = torch.ones((M,), dtype=torch.bool, device=device)
    inds = torch.empty((M,), dtype=torch.long, device=device)
    a = torch.arange(M, dtype=torch.long, device=device)
    prob = 1 - 2/na  # so na/2 average time
    for _ in range(na):  # rethink: might want early stop
        s[free] = float('inf')  # ignore current scores of free ones
        s1.copy_(s)
        f = fft(x)
        fl = f.view(-1, nm*(nn2+1))
        inds.fill_(-1)
        for j in range(na):
            tabu_mask = tabu[:, j]
            flmod = fl[tabu_mask] + torch.mul(x[tabu_mask, j].to(complex_dtype).unsqueeze(1), wrng_all[j])
            fmod = flmod.view(-1, nm, nn2+1)
            s1[tabu_mask] = score_fft(fmod)
            mask = s1 < s
            s[mask] = s1[mask]
            inds[mask] = j
        a = torch.nonzero(inds >= 0, as_tuple=True)[0]
        # print(t,a.numel(),free.sum().item())
        inds1 = inds[a]
        x[a, inds1] *= -1
        tabu[a, inds1] = False
        free &= torch.rand((M,), device=device) < prob  # freeze some
    arrays[active_rows] = x
    scores[active_rows] = s
    print('')

"""
# greedy random k-bit flip
@torch.inference_mode()
def improve_greedy_alt(x, scores):
    print("improve_greedy ", end=''); sys.stdout.flush()
    B = x.shape[0]
    # precompute fft
    f = fft(x)
    fl = f.view(B, nm*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B, nm*(nn2+1))
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    k = 2
    ns = 5 * n  # dunno
    while ns > 0 and k <= nn2:
        cnt.zero_()
        # create all at once a bunch of subsets to sample
        all_inds = torch.unique(torch.topk(torch.rand(ns, na, device=device), k).indices.sort(dim=1).values, dim=0)
        n_inds = all_inds.shape[0]
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
        print(f'{k=} {cnt} ({cnt/B})')
        ns >>= 1
        k += 1

# greedy random k-bit rotate
@torch.inference_mode()
def improve_greedy_fixed_alt(x, scores):
    print("improve_greedy_fixed ", end=''); sys.stdout.flush()
    B = x.shape[0]
    # precompute fft
    f = fft(x)
    fl = f.view(B, nm*(nn2+1))
    fmod = torch.empty_like(f)
    flmod = fmod.view(B, nm*(nn2+1))
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    k = 3  # 3,5,..,11
    ns = 5 * n  # dunno
    while ns > 0 and k <= nn2:
        cnt.zero_()
        # create all at once a bunch of subsets to sample
        lst = []
        for j in range(nm):
            lst.append(j*nn+torch.topk(torch.rand(ns, nn, device=device), k).indices.sort(dim=1).values)
        all_inds = torch.unique(torch.cat(lst, dim=0), dim=0)
        n_inds = all_inds.shape[0]
        perm = torch.randperm(n_inds)
        for i in range(n_inds):
            inds = all_inds[perm[i]]
            xx = torch.roll(x[:, inds], shifts=1, dims=1)
            torch.matmul(((x[:, inds]-xx) >> 1).to(complex_dtype), wrng_all[inds], out=flmod)
            flmod.add_(fl)
            new_scores = score_fft(fmod)
            improved_inds = torch.nonzero(new_scores < scores, as_tuple=True)[0]  # better than mask when few True expected
            fl[improved_inds] = flmod[improved_inds]
            x[improved_inds.unsqueeze(1), inds] = xx[improved_inds]
            scores[improved_inds] = new_scores[improved_inds]
            cnt += improved_inds.shape[0]
        print(f'{k=} {cnt} ({cnt/B})')
        ns >>= 1
        k += 2
"""

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
        k = 3 + 2 * torch.floor(torch.log(torch.rand(()))*.5*invlogp)
        k = int(k.clamp(3, nn//2))
        n_inds = nn // k
        # create all at once a bunch of subsets to sample
        all_inds = j*nn + torch.randperm(nn, device=device)[:n_inds*k].view(n_inds,k)
        for i in range(n_inds):
            inds = all_inds[i]
            xx = .5*x[:, inds].to(complex_dtype)
            w = wrng_all[inds]
            xx2 = xx.clone()
            for _ in range(k-1):
                xx2 = torch.roll(xx2, shifts=1, dims=1)
                torch.matmul(xx-xx2, w, out=flmod)
                flmod.add_(fl)
                new_scores = score_fft(fmod)
                improved_inds = torch.nonzero(new_scores < scores, as_tuple=True)[0]  # better than mask when few True expected
                fl[improved_inds] = flmod[improved_inds]
                xx[improved_inds] = xx2[improved_inds]
                scores[improved_inds] = new_scores[improved_inds]
                cnt += improved_inds.shape[0]
            x[:, inds] = (2*xx).real.to(torch.int8)
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
        inds = torch.nonzero((ffs1 <= 1).all(dim=1), as_tuple=True)[0]
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
    inds = torch.empty((B, nm, 2, 2), dtype=torch.long, device=device)
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
            mask = scores[minuses].unsqueeze(1) < best_scores[minuses, j, 0]  # (B,2)
            # mask[:,0] means highest score
            minuses1 = minuses[mask[:, 0]]
            best_scores[minuses1, j, 0, 1] = best_scores[minuses1, j, 0, 0]
            best_scores[minuses1, j, 0, 0] = scores1[minuses1]
            inds[minuses1, j, 0, 1] = inds[minuses1, j, 0, 0]
            inds[minuses1, j, 0, 0] = i
            # mask[:,1] means next to highest score
            minuses1 = minuses[~mask[:, 0] & mask[:, 1]]
            best_scores[minuses1, j, 0, 1] = scores1[minuses1]
            inds[minuses1, j, 0, 1] = i
            # update pluses
            mask = scores[pluses].unsqueeze(1) < best_scores[pluses, j, 1]  # (B,2)
            # mask[:,0] means highest score
            pluses1 = pluses[mask[:, 0]]
            best_scores[pluses1, j, 1, 1] = best_scores[pluses1, j, 1, 0]
            best_scores[pluses1, j, 1, 0] = scores1[pluses1]
            inds[pluses1, j, 1, 1] = inds[pluses1, j, 1, 0]
            inds[pluses1, j, 1, 0] = i
            # mask[:,1] means next to highest score
            pluses1 = pluses[~mask[:, 0] & mask[:, 1]]
            best_scores[pluses1, j, 1, 1] = scores1[pluses1]
            inds[pluses1, j, 1, 1] = i
    # now try every combo
    inds = inds.view(B, nm*ksw)
    base = torch.arange(B, device=device)
    #print(inds,x,scores)
    cur = torch.gather(x, 1, inds)
    for i in range(sw.shape[0]):
        x[base.unsqueeze(1), inds] = sw[i]
        new_scores = score(x)  # TODO use fft
        #print(x,new_scores)
        improved = new_scores < scores
        scores[improved] = new_scores[improved]
        cur[improved] = sw[i]
        cnt += torch.sum(improved)
    x.scatter_(1, inds, cur)
    print(f'{cnt/B}')
