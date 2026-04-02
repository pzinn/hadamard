#!/usr/bin/env python
# coding: utf-8

import torch
import params
params.init_from_argv()
from params import na, nm, nn, nn2, device, resume, resume_training, is_sweep, debugging, config, score, fft, fixed_sums, num_ones, aut, perms, real_dtype, eps
from improve import improve1p, improve_greedy, improve_phases, improve_greedy_fixed, improve4x4_fixed, improve1p_fixed
from pt import parallel_tempering, nT
import logger
import transformer
from symmetry import build_context, canonicalise_exact, canonicalise_heuristic
# logging/debugging
import sys
from timeit import default_timer as timer  # to measure exec time

if fixed_sums:
    @torch.inference_mode()
    def generate_random_blocks(B, n, k):  # Generate a (B, n) tensor of ±1 with exactly k entries of +1 per row.
        a = -torch.ones((B, n), dtype=torch.int8)
        rand = torch.rand((B, n))
        topk = rand.argsort(dim=1)[:, :k]        # (B, k) random unique positions per row
        rows = torch.arange(B).unsqueeze(1)
        a[rows, topk] = 1
        return a
    @torch.inference_mode()
    def generate_random_arrays(batch_size):  # used to be done on gpu during first scoring, maybe reinstate at some point?
        return torch.cat([generate_random_blocks(batch_size, nn, num_ones[j]) for j in range(nm)], dim=1)
else:
    @torch.inference_mode()
    def generate_random_arrays(batch_size):  # used to be done on gpu during first scoring, maybe reinstate at some point?
        return 2 * torch.randint(2, (batch_size, na), dtype=torch.int8) - 1

# MAIN-DEFINITIONS #

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
    print(f"{prefix} stats:")
    B = len(arrays)
    if B == 0:
        return
    # compute autocorrelation by MC
    mc_size = 1000
    with torch.random.fork_rng():
        s = (arrays[torch.randint(B, (mc_size,), device=arrays.device)] * arrays[torch.randint(B, (mc_size,), device=arrays.device)]).sum(dim=1).abs().sum()/(mc_size*na)
    s = s.item()
    print(f"Correlation: {s}")

    # now scores
    min_score = torch.min(scores)
    print(f"Min score: {min_score}")

    med_score = torch.median(scores)
    print(f"Med score: {med_score}")

    avg_score = torch.mean(scores)
    print(f"Avg score: {avg_score}")

    max_score = torch.max(scores)
    print(f"Max score: {max_score}")

    def tally_str(data):
        fmt = (lambda x: x.item()) if data.ndim == 1 else (lambda x: tuple(x.tolist()))
        unique_data, counts = torch.unique(data, dim=0, return_counts=True)
        idx = torch.argsort(counts, descending=True)
        return "{" + ", ".join(f"{fmt(unique_data[i])}: {counts[i].item():_}" for i in idx) + "}"

    gens_tally = tally_str(gens)
    print(f"Gen tally: {gens_tally}")

    hada_inds = torch.nonzero(scores < eps, as_tuple=True)[0]
    nh = len(hada_inds) / len(arrays)
    print(f"Hadamard ratio: {nh}")

    segment_sums = arrays.view(B, nm, nn).sum(dim=2)
    segment_sums = torch.sort(segment_sums.abs(), dim=1).values
    ss_tally = tally_str(segment_sums)
    print(f"Segment sums tally: {ss_tally}")

    hada_gens_tally = tally_str(gens[hada_inds])
    hada_ss_tally = tally_str(segment_sums[hada_inds])

    if len(hada_inds) > 0:
        print(f"Hadamard gen tally: {hada_gens_tally}")
        print(f"Hadamard segment sums tally: {hada_ss_tally}")
        new_hada_tensor = canonicalise_exact(arrays[hada_inds].to(device=device), symmetry_ctx)
        record_stats.hada_tensor = torch.unique(torch.cat((record_stats.hada_tensor, new_hada_tensor.cpu()), dim=0), dim=0)
        total_nh = len(record_stats.hada_tensor)
        print(f"Total number of Hadamard: {total_nh}")
        print_arrays(record_stats.hada_tensor[0:1])  # doesn't appear on log??
        write_arrays(logger.hada_file, record_stats.hada_tensor)

    with open(logger.stats_file, 'a') as file:
        if not record_stats.has_run:
            record_stats.has_run = True
            file.write(f"{'gen':>3} {'':<10}: {'min score':>10} {'med score':>10} {'avg score':>10} {'max score':>10} {'autocorrel':>10} {'H-ratio':>10} {'H-number':>10} gens, H-gens, H-segment sums tallies\n")
        file.write(f"{params.gen:>3} {prefix:<10}: {min_score:10.6f} {med_score:10.6f} {avg_score:10.6f} {max_score:10.6f} {s:10.6f} {nh:10.6f} {len(hada_inds):>10} {gens_tally} {hada_gens_tally} {hada_ss_tally}\n")

    if prefix and not prefix.startswith("debug"):
        logger.record_scores(prefix, scores, avg_score, nh)


# scoring. technically we don't need this since the scores could be computed when improving;
# but useful for logging/stats.
def batch_score(arrays):  # score but in batches of score_batch_size, and move back and forth to cpu
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        return score(arrays.to(device=device)).cpu()
    B = arrays.shape[0]
    scores = torch.empty(B, dtype=real_dtype)
    for i in range(0, B, config.score_batch_size):
        j = i + config.score_batch_size
        scores[i:j] = score(arrays[i:j].to(device=device))
    return scores

def fix_num_ones(arrays):  # fix # 1s. shouldn't happen too often
    a = arrays.view(-1, nm, nn)
    for j in range(nm):
        while True:
            k = (a[:, j] == 1).sum(dim=1)
            mask1 = k < num_ones[j]
            mask2 = k > num_ones[j]
            if not mask1.any() and not mask2.any():
                break
            a[mask1, j, torch.randint(nn, (), device=device)] = 1  # lazy
            a[mask2, j, torch.randint(nn, (), device=device)] = -1

symmetry_ctx = build_context()

@torch.inference_mode()
def parallel_improve(arrays, scores, gens):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    # step A: fix segment sums if fixed sums
    if fixed_sums:
        fix_num_ones(arrays)
    # step B: first pass of local search
    start_timer = timer()
    improve_phases(arrays, scores)
    scores = score(arrays)  # don't trust improve
    if debugging:
        print(f"improve B1 time: {timer() - start_timer}")
        record_stats(arrays, scores, gens, prefix="debug B1")
    #
    start_timer = timer()
    if fixed_sums:
        #improve4x4_fixed(arrays, scores)
        improve1p_fixed(arrays, scores)
    else:
        improve1p(arrays, scores)
    scores = score(arrays)  # don't trust improve
    if debugging:
        print(f"improve B2 time: {timer() - start_timer}")
        record_stats(arrays, scores, gens, prefix="debug B2")
    # step C: parallel tempering (if num_improve>0)
    start_timer = timer()
    scores, inds = torch.sort(scores, descending=True)
    arrays = arrays[inds]
    gens = gens[inds]
    if arrays.shape[0] == 0:
        return arrays, scores, gens
    B = arrays.shape[0]
    print(f"identical ratio = {(arrays[1:] == arrays[:-1]).all(dim=1).sum()/B}")
    B1 = (B // nT) * nT  # round down to a multiple of nT
    if B1 > 0 and scores[B1-1] < eps:  # don't touch H-matrices
        B1 = int(torch.nonzero(scores < eps, as_tuple=True)[0][0])
        B1 = (B1 // nT) * nT
    parallel_tempering(arrays[:B1], scores[:B1], gens[:B1])
    if debugging:
        print(f"pt time: {timer() - start_timer}")
        record_stats(arrays, scores, gens, prefix="debug pt")
    # step D: second pass of local search (if num_improve>0)
    for _ in range(config.num_improve):
        start_timer = timer()
        if fixed_sums:
            #improve4x4_fixed(arrays, scores)
            improve1p_fixed(arrays, scores)
        else:
            improve1p(arrays, scores)
        scores = score(arrays)  # don't trust improve
        if debugging:
            print(f"improve D1 time: {timer() - start_timer}")
            record_stats(arrays, scores, gens, prefix="debug D1")
        #
        start_timer = timer()
        if fixed_sums:
            improve_greedy_fixed(arrays, scores)
        else:
            improve_greedy(arrays, scores)
        scores = score(arrays)  # don't trust improve
        if debugging:
            print(f"improve D2 time: {timer() - start_timer}")
            record_stats(arrays, scores, gens, prefix="debug D2")
        #
        start_timer = timer()
        improve_phases(arrays, scores)
        scores = score(arrays)  # don't trust improve
        if debugging:
            print(f"improve D3 time: {timer() - start_timer}")
            record_stats(arrays, scores, gens, prefix="debug D3")
        #
    # step E: rotate the arrays to a standard form
    start_timer = timer()
    arrays = canonicalise_heuristic(arrays, symmetry_ctx, fft, scores, score if params.test_score else None, eps)
    if debugging:
        print(f"derotate time: {timer() - start_timer}")
    return (arrays, scores, gens)

@torch.inference_mode()
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
    # _, idx = torch.topk(min_scores, k=config.training_size, largest=False, sorted=False)
    _, idx = torch.topk(min_scores * (1 + config.gen_decay * (params.gen - min_gens)), k=config.training_size, largest=False, sorted=False)
    arrays = arrays[idx]
    scores = min_scores[idx]
    gens = min_gens[idx]
    return arrays, scores, gens

@torch.inference_mode()
def batch_improve(arrays0, scores0, gens0):
    torch.set_float32_matmul_precision('highest')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()  # Free memory
    if config.score_batch_size is None:
        arrays, scores, gens = parallel_improve(arrays0.to(device=device), scores0.to(device=device), gens0.to(device=device))
        # select
        arrays, scores, gens = best_from(arrays, scores, gens)
    else:
        B = arrays0.shape[0]
        for i in range(0, B, config.score_batch_size):
            j = i + config.score_batch_size
            new_arrays, new_scores, new_gens = parallel_improve(arrays0[i:j].to(device=device), scores0[i:j].to(device=device), gens0[i:j].to(device=device))
            if i == 0:
                arrays, scores, gens = new_arrays, new_scores, new_gens
            else:
                arrays = torch.cat((arrays, new_arrays), dim=0)
                scores = torch.cat((scores, new_scores), dim=0)
                gens = torch.cat((gens, new_gens), dim=0)
            arrays, scores, gens = best_from(arrays, scores, gens)
    return arrays.cpu(), scores.cpu(), gens.cpu()

def main():
    # logging: text stats file + fancy (tensorboard or wandb)
    logger.init_logging()
    record_stats.has_run = False  # we could leave it undefined, but not in case of sweep
    record_stats.hada_tensor = torch.empty((0, na), dtype=torch.int8)  # empty the hadamard list

    # initialise transformer
    transformer.init_model()

    # torch functions
    seed = int(config.random_seed)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.set_device(0)  # Use GPU 0
        torch.cuda.empty_cache()  # Free memory before large computation
        torch.cuda.manual_seed_all(seed)

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
            arrays = torch.empty((0, na), dtype=torch.int8)
    else:
        arrays = generate_random_arrays(config.sample_size)

    scores = batch_score(arrays)
    gens = torch.full(scores.shape, params.gen, dtype=torch.uint8)
    record_stats(arrays, scores, gens, prefix="sample" if not resume else "")  # who knows where the data come from if resuming

    # MAIN-LOOP

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
            break
        if params.skip_first_training:
            params.skip_first_training = False
        else:
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            # train on GEN-gen
            print(f"\n***Training on GEN-{params.gen:02d}***")
            coeff = 1 if params.gen == 0 or not resume_training else (1+params.gen)**-1.5
            # linear warmup with fixed base learning rate afterwards:
            def get_lr(step, warmup_steps=10000):
                return coeff * config.learning_rate * (.01+.99*step / warmup_steps if step < warmup_steps else 1)
            max_steps = config.training_steps if coeff == 1 else config.training_steps//10
            eval_freq = 1000
            start_timer = timer()
            transformer.train(arrays, score=score if params.test_score else None, max_steps=max_steps, eval_freq=eval_freq, lr_sched=get_lr)
            if debugging:
                print(f"training time: {timer() - start_timer}")
        # sample from model to get new data
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        print(f"\n***Sampling from transformer trained on GEN-{params.gen:02d}***")
        params.gen += 1
        start_timer = timer()
        new_arrays = transformer.sample()
        new_scores = batch_score(new_arrays)
        new_gens = torch.full(new_scores.shape, params.gen, dtype=torch.uint8)
        record_stats(new_arrays, new_scores, new_gens, prefix="sample")  # do we produce similar scores as training data?
        # combine, but mix
        A = arrays.shape[0]  # should be training_size
        B = new_arrays.shape[0]  # should be sample_size
        perm = torch.randperm(A+B)  # overkill but whatever
        combined_arrays = torch.empty((A+B, na), dtype=torch.int8)
        combined_scores = torch.empty(A+B, dtype=real_dtype)
        combined_gens = torch.empty(A+B, dtype=torch.uint8)
        combined_arrays[perm[:A]] = arrays
        combined_arrays[perm[A:]] = new_arrays
        combined_scores[perm[:A]] = scores
        combined_scores[perm[A:]] = new_scores
        combined_gens[perm[:A]] = gens
        combined_gens[perm[A:]] = new_gens
        arrays, scores, gens = combined_arrays, combined_scores, combined_gens
        if debugging:
            print(f"sampling time: {timer() - start_timer}")


if is_sweep:
    import wandb
    wandb.agent(logger.sweep_id, function=main)
else:
    main()
