#!/usr/bin/env python
# coding: utf-8

# script to compute exactly # GS H-matrices

import sys
import torch

from symmetry import build_context, canonicalise_automorphism_exact

device = "cuda" if torch.cuda.is_available() else "cpu"
if len(sys.argv) < 2:
    n = 60   # must be a multiple of 4
else:
    n = int(sys.argv[1])
if len(sys.argv) < 3:
    segment_sums = None
else:
    segment_sums = tuple(int(x) for x in sys.argv[2].split(","))
assert(n%4==0)
nn = n//4
autocorrel_dtype = torch.int8 if nn < 127 else torch.int16
nm = 4
print(f"{n=}")
if segment_sums is not None:
    assert len(segment_sums) == nm
    assert sum(s * s for s in segment_sums) == n
    print(f"{segment_sums=}")
eps = 2e-5
symmetry_ctx = build_context(nn=nn, nm=1, device=device, real_dtype=torch.float32)
sequence_batch_size = 1 << 16


def pm1_sequence_batches(n: int, batch_size: int):
    rows = 1 << n
    shifts = torch.arange(n - 1, -1, -1, device=device)
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        numbers = torch.arange(start, stop, dtype=torch.int64, device=device).unsqueeze(1)
        bits = (numbers >> shifts) & 1
        yield (bits * 2 - 1).to(dtype=torch.int8)


def necklace_batches_with_sum(length: int, total_sum: int, batch_size: int):
    """Generate ±1 necklace representatives with fixed sum, in batches.

    Yields (sequences, orbit_sizes) tensors.  Uses Sawada's fixed-content
    necklace algorithm so only one representative per cyclic orbit is produced;
    orbit_sizes carries the weight (= minimal period) for each representative.
    """
    if (length + total_sum) % 2 != 0:
        return
    num_ones = (length + total_sum) // 2
    if num_ones < 0 or num_ones > length:
        return
    if length == 0:
        yield (torch.ones((1, 0), dtype=torch.int8, device=device),
               torch.ones((1,), dtype=torch.long, device=device))
        return
    divs = [d for d in range(1, length) if length % d == 0]

    a = [0] * (length + 1)
    remaining = [length - num_ones, num_ones]
    buf_seq = []
    buf_per = []

    def min_period():
        for d in divs:
            if all(a[i + 1] == a[(i % d) + 1] for i in range(d, length)):
                return d
        return length

    def gen(t, p):
        if t > length:
            if length % p == 0:
                buf_seq.append(list(a[1:length + 1]))
                buf_per.append(min_period())
            return
        for j in range(a[t - p], 2):
            if remaining[j] > 0:
                a[t] = j
                remaining[j] -= 1
                gen(t + 1, p if j == a[t - p] else t)
                remaining[j] += 1

    gen(1, 1)
    for i in range(0, len(buf_seq), batch_size):
        buf_slice_seq = buf_seq[i:i + batch_size]
        buf_slice_per = buf_per[i:i + batch_size]
        seqs = torch.tensor(buf_slice_seq, dtype=torch.int8, device=device) * 2 - 1
        periods = torch.tensor(buf_slice_per, dtype=torch.long, device=device)
        yield seqs, periods

def unique_rows_with_counts(rows, counts=None):
    if counts is None:
        return torch.unique(rows, dim=0, return_counts=True)
    unique_rows, inverse = torch.unique(rows, dim=0, return_inverse=True)
    num_unique = unique_rows.shape[0]
    unique_counts = torch.empty((num_unique,), device=device, dtype=counts.dtype)
    unique_counts.scatter_reduce_(0, inverse, counts, reduce='sum', include_self=False)
    return unique_rows, unique_counts


def print_table_stats(label, rows, counts):
    print(
        f"{label}: classes={rows.shape[0]} total={counts.sum().item()} "
        f"min={counts.min().item()} mean={counts.to(dtype=torch.float32).mean().item():.3f} max={counts.max().item()}"
    )


def merge_tables(rows1, counts1, rows2, counts2):
    if rows1 is None:
        return rows2, counts2
    merged_rows = torch.cat((rows1, rows2), dim=0)
    merged_counts = torch.cat((counts1, counts2), dim=0)
    return unique_rows_with_counts(merged_rows, merged_counts)


def build_block_table(required_sum=None):
    autocorrelation_classes = None
    class_multiplicities = None
    if required_sum is None:
        iterator = ((batch, None) for batch in pm1_sequence_batches(nn, sequence_batch_size))
    else:
        iterator = necklace_batches_with_sum(nn, required_sum, sequence_batch_size)
    for sequences, weights in iterator:
        spectra = torch.fft.rfft(sequences, dim=1)
        power_spectra = torch.view_as_real(spectra).square().sum(dim=-1)
        mask = (power_spectra <= n + eps).all(dim=-1)
        power_spectra = power_spectra[mask]
        if power_spectra.numel() == 0:
            continue
        chunk_classes = torch.fft.irfft(power_spectra, n=nn).round().to(dtype=autocorrel_dtype)
        chunk_weights = weights[mask] if weights is not None else None
        chunk_classes, chunk_counts = unique_rows_with_counts(chunk_classes, chunk_weights)
        autocorrelation_classes, class_multiplicities = merge_tables(
            autocorrelation_classes, class_multiplicities, chunk_classes, chunk_counts
        )
    if autocorrelation_classes is None:
        autocorrelation_classes = torch.empty((0, nn), dtype=autocorrel_dtype, device=device)
        class_multiplicities = torch.empty((0,), dtype=torch.long, device=device)
    return autocorrelation_classes, class_multiplicities


tables_by_sum = {}
if segment_sums is None:
    base_table = build_block_table()
    print(f"initial autocorrelation classes: {base_table[0].shape[0]} ({base_table[0].shape[0] / 2**nn:.6f} of all sequences)")
    print_table_stats("autocorrelation multiplicities", base_table[0], base_table[1])
    block_tables = [base_table] * nm
else:
    for s in sorted(set(segment_sums)):
        table = build_block_table(required_sum=s)
        tables_by_sum[s] = table
        print_table_stats(f"sum {s} block table", table[0], table[1])
    block_tables = [tables_by_sum[s] for s in segment_sums]

partial_sums, partial_counts = block_tables[0]
partial_sums = canonicalise_automorphism_exact(partial_sums, symmetry_ctx)
partial_sums, partial_counts = unique_rows_with_counts(partial_sums, partial_counts)
print_table_stats("after automorphism canonicalisation", partial_sums, partial_counts)

for i, (block_classes, block_multiplicities) in enumerate(block_tables[1:], start=1):
    if partial_sums.shape[0] < block_classes.shape[0]:
        large_table = block_classes
        small_table = partial_sums
        large_counts = block_multiplicities
        small_counts = partial_counts
    else:
        small_table = block_classes
        large_table = partial_sums
        small_counts = block_multiplicities
        large_counts = partial_counts
    new_partial_sums = None
    new_partial_counts = None
    print(f"combination step {i}")
    for j in range(small_table.shape[0]):
        summed_autocorrelations = large_table + small_table[j].unsqueeze(0)
        spectra = torch.fft.rfft(summed_autocorrelations, dim=1).real
        mask = (spectra <= n+eps).all(dim=-1)
        if mask.any():
            canonical_sums = canonicalise_automorphism_exact(summed_autocorrelations[mask], symmetry_ctx)
            new_partial_sums, new_partial_counts = merge_tables(
                new_partial_sums,
                new_partial_counts,
                canonical_sums,
                small_counts[j] * large_counts[mask],
            )
    if new_partial_sums is None:
        partial_sums = torch.empty((0, large_table.shape[1]), device=device, dtype=partial_sums.dtype)
        partial_counts = torch.empty((0,), device=device, dtype=torch.long)
    else:
        partial_sums = new_partial_sums
        partial_counts = new_partial_counts
    partial_sums = canonicalise_automorphism_exact(partial_sums, symmetry_ctx)
    partial_sums, partial_counts = unique_rows_with_counts(partial_sums, partial_counts)
    print_table_stats(f"after step {i}", partial_sums, partial_counts)

print(f"final exact GS count: {partial_counts.sum().item()}")
