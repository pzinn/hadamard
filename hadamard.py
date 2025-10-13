#!/usr/bin/env python
# coding: utf-8

import math
import torch
import params
from params import n, na, nn, nn2, k, device, resume, resume_training, random_seed, is_sweep, debugging, config
import logger
import transformer
# logging/debugging
import sys
from timeit import default_timer as timer  # to measure exec time

eps = 2e-5  # scores are heavily discretised so can be made large

real_dtype = torch.float32
complex_dtype = torch.complex64

import torch
"""
if debugging:
    # Start recording memory snapshot history
    torch.cuda.memory._record_memory_history(max_entries=100000)
"""

def generate_random_blocks(B, n, k, device):
    """
    Generate a (B, n) tensor of ±1 with exactly k entries of +1 per row.
    """
    # Start with all -1s
    a = -torch.ones((B, n), dtype=torch.int8, device=device)
    # For each row, randomly choose k positions to set to +1
    # We'll use torch.rand and argsort to get k random indices per row
    rand = torch.rand((B, n), device=device)
    topk = rand.argsort(dim=1)[:, :k]        # (B, k) random unique positions per row
    # Use advanced indexing to assign +1
    rows = torch.arange(B, device=device).unsqueeze(1)
    a[rows, topk] = 1
    return a


@torch.inference_mode()
def generate_random_arrays(batch_size, device):  # used to be pure gpu, maybe reinstate at some point?
    return torch.cat([generate_random_blocks(batch_size, nn2 if j<3 else nn, k[j], device) for j in range(4)], dim=1)

# MAIN-DEFINITIONS #
"""
def best_from(arrays_dict):
    # preserves ordering
    items = arrays_dict.items()
    smallest_keys = {k for k, _ in heapq.nsmallest(config.training_size, items, key=lambda item: item[1][0])}  # heapq requires no nan
    return {k: v for k, v in items if k in smallest_keys or v[0] < eps}  # always keep H-matrices
"""

def write_arrays_buffer(buffer, a):
    plus, minus = ord('+'), ord('-')
    rows = torch.where(a > 0, plus, minus).to(torch.uint8)
    for r in rows:
        buffer.write(bytes(r.tolist()))
        buffer.write(b"\n")

def write_arrays(file_path, a):
    with open(file_path, 'wb') as file:
        write_arrays_buffer(file, a)

def print_arrays(a):
    write_arrays_buffer(sys.stdout.buffer, a)
    sys.stdout.flush()

# for keeping track of stats
def record_stats(arrays, scores, gens, prefix=""):
    B = len(arrays)
    if B == 0:
        return
    # compute autocorrelation by MC
    mc_size = 1000
    #perms = params.perms.tolist()
    s = torch.tensor(0., device=arrays.device)
    for _ in range(mc_size):
        a1 = arrays[torch.randint(B,())]
        a2 = arrays[torch.randint(B,())]
        a13 = a1[:3*nn2].reshape(3,nn2)
        a23 = a1[:3*nn2].reshape(3,nn2)
        a11 = a1[3*nn2:]
        a21 = a2[3*nn2:]
        fft_a1 = torch.fft.fft(a11, dim=0)
        fft_a2 = torch.fft.fft(a21, dim=0)
        corr1 = torch.fft.ifft(fft_a1 * torch.conj(fft_a2), dim=0).real
        corr2 = torch.fft.ifft(fft_a1 * fft_a2, dim=0).real
        corr = torch.stack([corr1,corr2],dim=0)
        #corr = torch.abs(corr)
        s += torch.max(corr)
        """
        ss = 0
        for p in perms:
            sss = torch.sum(a13[p]*a23,dim=(0,1))
            if sss>ss:
                ss=sss
        s += ss
        """
        s += torch.sum(a13*a23,dim=(0,1))
    s /= (mc_size * na)
    s=s.item()
    print(f"Correlation: {s}")

    # now scores
    # if debugging:
    #     print(f'Score tally: {dict(zip(*np.unique(np.round(scores, decimals=5), return_counts=True)))}')

    # check k's
    kk = torch.empty(B, 4, dtype=torch.int8, device=device)
    for j in range(4):
        kk[:, j] = (arrays[:, j*nn2:((j+1)*nn2 if j<3 else na)]==1).sum(dim=1)
    kcheck = (kk == k).all(dim=1)
    print(f"Correct k: {kcheck.sum()/B}")

    min_score = torch.min(scores)
    print(f"Min score: {min_score}")

    mean_score = torch.mean(scores)
    print(f"Mean score: {mean_score}")

    max_score = torch.max(scores)
    print(f"Max score: {max_score}")

    # tally=Counter([val[1] for val in vals])
    gens_count = torch.bincount(gens)
    gens_count, gens_order = gens_count.sort(descending=True)
    gens_tally = { g : c for g, c in zip(gens_order.tolist(), gens_count.tolist()) if c > 0 }
    print(f"Gen tally: {gens_tally}")

    hada_inds = torch.nonzero(scores < eps, as_tuple=True)[0]
    nh = len(hada_inds) / len(arrays)
    print(f"Hadamard ratio: {nh}")

    hada_count = torch.bincount(gens[hada_inds])
    hada_count, hada_order = hada_count.sort(descending=True)
    hada_tally = { g : c for g, c in zip(hada_order.tolist(), hada_count.tolist()) if c > 0 }
    print(f"Hadamard gen tally: {hada_tally}")

    if prefix == "selected":  # don't spam with H-matrices...
        if not hasattr(record_stats, "hada_tensor"):
            record_stats.hada_tensor = torch.empty((0,na), dtype=torch.int8)
        record_stats.hada_tensor = torch.unique(torch.cat((record_stats.hada_tensor,arrays[hada_inds]), dim=0), dim=0)
        total_nh = len(record_stats.hada_tensor)
        print(f"Total number of Hadamard: {total_nh}")
        if total_nh>0:
            print_arrays(record_stats.hada_tensor[0:1])  # doesn't appear on log??
            write_arrays(logger.hada_file, record_stats.hada_tensor)

    with open(logger.stats_file, 'a') as file:
        if not record_stats.has_run:
            record_stats.has_run = True
            file.write(f"{'gen':>3} {'':<10}: {'min score':>10} {'mean score':>10} {'max score':>10} {'autocorrel':>10} {'H-ratio':>10} {'H-number':>10} tally / H-tally\n")
        file.write(f"{params.gen:>3} {prefix:<10}: {min_score:10.6f} {mean_score:10.6f} {max_score:10.6f} {s:10.6f} {nh:10.6f} {len(hada_inds):>10} {gens_tally} {hada_tally}\n")

    if prefix:
        logger.record_scores(prefix, scores, gens, mean_score, nh)

cst = 1 / math.sqrt(n)
one = torch.tensor([[1]],device=device,dtype=real_dtype)
def mir(a):
    return torch.cat((one.expand(a.shape[0],1),a,torch.flip(a,(1,))),dim=1)
def unfold(m0):
    return torch.stack((mir(m0[:,:nn2]),
                        mir(m0[:,nn2:2*nn2]),
                        mir(m0[:,2*nn2:3*nn2]),
                        m0[:,3*nn2:]),dim=1)
one_cpu = torch.tensor([[1]],device='cpu',dtype=real_dtype)
def mir_cpu(a):
    return torch.cat((one_cpu.expand(a.shape[0],1),a,torch.flip(a,(1,))),dim=1)
def unfold_cpu(m0):
    return torch.stack((mir_cpu(m0[:,:nn2]),
                        mir_cpu(m0[:,nn2:2*nn2]),
                        mir_cpu(m0[:,2*nn2:3*nn2]),
                        m0[:,3*nn2:]),dim=1)


def init_score_function():
    global score, score_cpu, score_fft, fft, unfold
    if config.score_function != 'fft log determinant':
        # Generate row indices for circulant
        indices = torch.arange(nn, device=device).repeat(nn, 1)  # Shape: (nn, nn)
        shifts = torch.arange(nn, device=device).unsqueeze(1)  # Shape: (nn, 1)
        rolled_indices = (indices - shifts) % nn  # Shape: (nn, nn)
        V = torch.arange(2*n, device=device).reshape(8, nn)  # Shape: (8,nn) -- the original array and its negation, for convenience
        X = V[:, rolled_indices]
        X[3] = torch.flip(X[3], dims=[1])
        X[7] = torch.flip(X[7], dims=[1])
        full_indices = torch.cat([
            torch.cat((X[0], X[1], X[2], X[3]), dim=1),
            torch.cat((X[5], X[0], X[7], X[2]), dim=1),
            torch.cat((X[6], X[3], X[0], X[5]), dim=1),
            torch.cat((X[7], X[6], X[1], X[0]), dim=1)
        ], dim=0)
        # create the n x n block circulant matrix out of the n bits
        def block_circulant(x):
            return torch.cat((x, -x), dim=1)[..., full_indices]
        # for device=='cuda', float is pretty much compulsory
    # float16 may be faster but may lead to accuracy issues
    if config.score_function == 'log determinant':
        def score(m0):
            m=unfold(m0)
            return n/2 * math.log(n) - torch.linalg.slogdet(block_circulant(m))[1]
    elif config.score_function == 'fft log determinant':
        nm=4
        @torch.inference_mode()
        def fft(m):
            return cst * torch.fft.rfft(m, dim=2)  # cst there for accuracy
        @torch.inference_mode()
        def score_fft(f):  # score in terms of precomputed fft
            s = -2*torch.log(torch.real(f*f.conj()).sum(dim=1))
            return s[:,0]+2*s[:,1:].sum(dim=1)
        @torch.inference_mode()
        def score(m0):
            return score_fft(fft(unfold(m0)))
        def score_cpu(m0):
            return score_fft(fft(unfold_cpu(m0)))
    elif config.score_function == 'quartic':
        score_threshold = n**1.5
        score_normalisation = 2*math.sqrt(n)
        def score(m):
            C = block_circulant(m)
            nrm = torch.linalg.matrix_norm(torch.matmul(C, torch.transpose(C, 1, 2)))
            return (nrm-score_threshold)/score_normalisation
    elif config.score_function == 'one':
        score_threshold = 0
        score_normalisation = n
        Idn = n * torch.eye(n, device=device, dtype=real_dtype)
        def score(m):
            C = block_circulant(m)
            nrm = torch.linalg.matrix_norm(torch.matmul(C, torch.transpose(C, 1, 2))-Idn, ord=1)
            return (nrm-score_threshold)/score_normalisation
    else:
        raise Exception('unknown score_function')

# scoring. technically we don't need this since the scores could be computed when improving;
# but useful for logging/stats. also generates random data at gen 0
def parallel_score(arrays):
    scores = score(arrays)  # Compute scores in parallel
    return scores.cpu()  # move back to cpu

def batch_generator(arrays):
    B = arrays.shape[0]
    for i in range(0, B, config.score_batch_size):
        j = i + config.score_batch_size
        yield arrays[i:j].to(device=device, dtype=real_dtype)

"""
def random_batch_generator():
    n_full_batches = config.sample_size // config.score_batch_size
    remainder = config.sample_size % config.score_batch_size
    for _ in range(n_full_batches):
        yield generate_random_arrays(config.score_batch_size)
    if remainder:
        yield generate_random_arrays(remainder)
"""

def batch_score(arrays):  # same as parallel_score but in batches of score_batch_size
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        if arrays is None:
            arrays_gpu = generate_random_arrays(config.sample_size, device)
            arrays = arrays_gpu.to(device='cpu', dtype=torch.int8)
        else:
            arrays_gpu = arrays.to(device=device, dtype=real_dtype)
        return arrays, parallel_score(arrays_gpu)
    scores = torch.empty((0,), dtype=real_dtype)
    if arrays is None:
        arrays = generate_random_arrays(config.sample_size, 'cpu')  # lame? reinstate old way?
    batches = batch_generator(arrays)
    for batch in batches:
        new_scores = parallel_score(batch)
        scores = torch.cat((scores, new_scores), dim=0)
    return arrays, scores

# precompute roots of unity for fft delta
w = torch.exp(2j * torch.tensor(torch.pi, device=device, dtype=real_dtype) / nn)
rng0 = torch.arange(nn, device=device, dtype=real_dtype)
rng = torch.arange(nn2+1, device=device, dtype=real_dtype)
wrng = 2 * cst * w ** torch.outer(rng0,rng)
wrng012 = -(wrng + torch.conj(wrng))[1:nn2+1]
wrng3 = -torch.conj(wrng)
wrng_all = torch.zeros((na,4*(nn2+1)), device=device, dtype=complex_dtype)
for i in range(3):
    wrng_all[i*nn2:(i+1)*nn2,i*(nn2+1):(i+1)*(nn2+1)] = wrng012
wrng_all[3*nn2:,3*(nn2+1):] = wrng3

# improve: greedy 2-bit switch per block
# TODO: rewrite using wrng_all
@torch.inference_mode()
def improve2(arrays_tensor,scores):
    print("improve2 ", end=''); sys.stdout.flush()
    cnt = torch.tensor(0, device=device, dtype=torch.int64)
    for k in range(4):
        if k<3:
            x=arrays_tensor[:,k*nn2:(k+1)*nn2]
        else:
            x=arrays_tensor[:,3*nn2:]
        for i in range(1,x.shape[1]):
            xi = x[:,i].clone()
            for j in range(i):
                x[:, i] = x[:, j]
                x[:, j] = xi
                new_scores = score(arrays_tensor)
                mask = new_scores < scores
                cnt += torch.sum(mask)
                xi[mask] = x[mask, i]  # make these changes permanent
                x[~mask, j] = x[~mask, i]  # Only revert for elements where no improvement
                x[~mask, i] = xi[~mask]
                scores[mask] = new_scores[mask]  # Update scores accordingly
    print(f'{cnt/x.shape[0]}')

def fixk(arrays):  # fix k's. shouldn't happen too often
    for j in range(4):
        r1=j*nn2
        r2=(j+1)*nn2 if j<3 else na
        while True:
            kk = (arrays[:, r1:r2]==1).sum(dim=1)
            mask1 = kk < k[j]
            mask2 = kk > k[j]
            if not mask1.any() and not mask2.any():
                break
            arrays[mask1, torch.randint(r1,r2,())] = 1  # lazy
            arrays[mask2, torch.randint(r1,r2,())] = -1

vec = torch.rand((nn,),device=device,dtype=real_dtype)  # doesn't really matter, used for ordering
fft_vec = torch.fft.rfft(vec)
fft_conj_vec = torch.conj(fft_vec)
base = torch.arange(nn, device=device)
def mysort(arrays, scores):
    if params.test_score:
        scores1 = score(arrays)
        if (scores-scores1).abs().max() > eps:
            raise RuntimeError("score incorrect", scores, scores1, (scores-scores1).abs().max().item(),(scores-scores1).abs().mean().item())
    """
    # 1st phase: permute the 3xnn2 parts
    B = arrays.shape[0]
    m=3
    a=arrays[:,:m*nn2].view(B,m,nn2)
    # start with identity permutation for each batch
    perm = torch.arange(m, device=device).expand(B, m).clone()
    # stable sort by last key first, then previous..., up to first column
    for k in range(nn2):
        key = a[:, :, k]                 # (B, m)
        key_in_curr_order = key.gather(1, perm)
        ordk = torch.argsort(key_in_curr_order, dim=1, stable=True)
        perm = perm.gather(1, ordk)
    # apply permutation to rows
    sorted_a = a.gather(1, perm.unsqueeze(-1).expand(-1, -1, nn2))
    a.copy_(sorted_a)
    """
    # 2nd phase: cyclically permute/reflect the remaining length nn part
    a=arrays[:,3*nn2:]
    fft_a = torch.fft.rfft(a, dim=1)  # use fft to quickly compute scalar product with some random vector for ordering
    sp_rot = torch.fft.irfft(fft_conj_vec[None, :] * fft_a, n=nn, dim=1)  # (B, nn)
    sp_rev = torch.fft.irfft(fft_vec[None, :] * fft_a, n=nn, dim=1)  # (B, nn)
    sps = torch.cat([sp_rot, sp_rev], dim=1)   # (B, 2 * nn)
    flat_idx = sps.argmax(dim=1)         # (B,) over 2*nn options
    # Gather the chosen transform from the original 'a'
    signed_base = torch.where(flat_idx >= nn,-1,1).unsqueeze(1) * base.unsqueeze(0)
    idx = ( signed_base + flat_idx.unsqueeze(1)) % nn
    transformed = a.gather(1, idx)  # (B, nn)
    # negate if the chosen scalar product is > 0
    a.copy_(transformed)
    if params.test_score:
        scores2 = score(arrays)
        if (scores1-scores2).abs().max() > eps:
            raise RuntimeError("score not preserved by sort", scores1, scores2, (scores1-scores2).abs().max().item(),(scores1-scores2).abs().mean().item())

def parallel_improve(arrays, scores, gens):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    # step A: fix k
    fixk(arrays)
    scores = score(arrays)  # Recompute scores
    # step B: main improvement
    start_timer = timer()
    for _ in range(config.num_improve):
        improve2(arrays, scores)
    #scores = score(arrays)  # don't trust improve2
    if debugging:
        print(f"improve2 time: {timer() - start_timer}")
    # step C: rotate the arrays to a standard form
    start_timer = timer()
    mysort(arrays, scores)
    if debugging:
        print(f"mysort time: {timer() - start_timer}")
    return (arrays, scores, gens)

def best_from(arrays, scores, gens):
    # deduplicate
    arrays, inv = torch.unique(arrays, dim=0, return_inverse=True, sorted=False)
    B = arrays.shape[0]
    min_gens = torch.empty(B, device=device, dtype=torch.uint8)
    min_gens.scatter_reduce_(0, inv, gens, reduce='amin', include_self=False)
    # normally scores should be equal but who knows
    min_scores = torch.empty(B, device=device, dtype=real_dtype)
    min_scores.scatter_reduce_(0, inv, scores, reduce='amin', include_self=False)
    # select
    if B <= config.training_size:
        return arrays, min_scores, min_gens
    scores, idx = torch.topk(min_scores, k=config.training_size, largest=False, sorted=False)
    arrays = arrays[idx]
    gens = min_gens[idx]
    return arrays, scores, gens

def batch_improve(arrays0, scores0, gens0):
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        arrays, scores, gens = parallel_improve(arrays0.to(device=device, dtype=real_dtype), scores0.to(device=device), gens0.to(device=device))
        # select
        arrays, scores, gens = best_from(arrays, scores, gens)
    else:
        B = arrays0.shape[0]
        arrays = torch.empty((0,na), dtype=real_dtype, device=device)
        scores = torch.empty((0,), dtype=real_dtype, device=device)
        gens = torch.empty((0,), dtype=torch.uint8, device=device)
        for i in range(0, B, config.score_batch_size):
            j = i + config.score_batch_size
            new_arrays, new_scores, new_gens = parallel_improve(arrays0[i:j].to(device=device, dtype=real_dtype), scores0[i:j].to(device=device), gens0[i:j].to(device=device))
            arrays = torch.cat((arrays, new_arrays), dim=0)
            scores = torch.cat((scores, new_scores), dim=0)
            gens = torch.cat((gens, new_gens), dim=0)
            arrays, scores, gens = best_from(arrays, scores, gens)
    return arrays.to(device='cpu', dtype=torch.int8), scores.cpu(), gens.cpu()

def main():
    # logging: text stats file + fancy (tensorboard or wandb)
    logger.init_logging()
    record_stats.has_run = False  # we could leave it undefined, but not in case of sweep

    # scoring
    init_score_function()

    # initialise transformer
    transformer.init_model()

    # torch functions
    torch.manual_seed(random_seed)
    if device.startswith('cuda'):
        torch.cuda.set_device(0)  # Use GPU 0
        torch.cuda.empty_cache()  # Free memory before large computation
        torch.cuda.manual_seed_all(random_seed)

    # STEP 0

    # initial info
    if resume:
        # use existing sample
        init_sample = params.work_dir + f'GEN-{params.gen:02d}.txt'
        try:
            with open(init_sample, 'r') as f:
                arrays = torch.tensor([tuple(1 if c == "+" else -1 for c in line.strip()) for line in f], dtype=torch.int8)
            print(f'***Loading initial sample from {init_sample}***')
        except FileNotFoundError:
            arrays = []
    else:
        arrays = None

    arrays, scores = batch_score(arrays)
    gens = torch.full(scores.shape, params.gen, dtype=torch.uint8)
    record_stats(arrays, scores, gens, prefix="sample" if not resume else "")  # who knows where the data come from if resuming

    # MAIN-LOOP #

    while True:
        if params.skip_first_improve:
            params.skip_first_improve = False
        else:
            # improve existing data, write to GEN-(gen)
            print('\n***Improving***')
            start_timer = timer()
            arrays, scores, gens = batch_improve(arrays, scores, gens)
            if debugging:
                print(f"improving time: {timer() - start_timer}")
            print('\n***Selecting***')  # technically already done, but left for clarity of output
            record_stats(arrays, scores, gens, "selected")
            write_arrays(params.work_dir + f'GEN-{params.gen:02d}.txt', arrays)
        if params.gen == config.max_iterations:
            """
            if debugging:
                torch.cuda.memory._dump_snapshot("profile.pkl")
                torch.cuda.memory._record_memory_history(enabled=None)
            """
            break
        if params.skip_first_training:
            params.skip_first_training = False
        else:
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            # train on GEN-gen
            print(f"\n***Training on GEN-{params.gen:02d}***")
            coeff = 1 if params.gen == 0 or not resume_training else .1
            # linear warmup with fixed base learning rate afterwards:
            def get_lr(step, warmup_steps=10000):
                return config.learning_rate * (.01+.99*step / warmup_steps if step < warmup_steps else 1)
            max_steps = int(config.training_steps*coeff)
            eval_freq = int(500*coeff)
            start_timer = timer()
            transformer.train(arrays, score=score_cpu if params.test_score else None, max_steps=max_steps, eval_freq=eval_freq, lr_sched=get_lr)
            if debugging:
                print(f"training time: {timer() - start_timer}")
        # sample from model to get new data
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        print(f"\n***Sampling from transformer trained on GEN-{params.gen:02d}***")
        params.gen += 1
        start_timer = timer()
        new_arrays = transformer.sample()
        new_arrays, new_scores = batch_score(new_arrays)
        new_gens = torch.full(new_scores.shape, params.gen, dtype=torch.uint8)
        record_stats(new_arrays, new_scores, new_gens, prefix="sample")  # do we produce similar scores as training data?
        arrays = torch.cat((arrays, new_arrays), dim=0)
        scores = torch.cat((scores, new_scores), dim=0)
        gens = torch.cat((gens, new_gens), dim=0)
        if debugging:
            print(f"sampling time: {timer() - start_timer}")


if is_sweep:
    import wandb
    wandb.agent(logger.sweep_id, function=main)
else:
    main()
