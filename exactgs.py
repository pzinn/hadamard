#!/usr/bin/env python
# coding: utf-8

# script to compute exactly # GS H-matrices

import sys
import torch

from symmetry import build_context, canonicalise_automorphism_exact

device = "cuda" if torch.cuda.is_available() else "cpu"
score_type = torch.float32
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
nn2 = nn//2
nm = 4
na = n
print(f"{n=}")
if segment_sums is not None:
    assert len(segment_sums) == nm
    assert sum(s * s for s in segment_sums) == n
    print(f"{segment_sums=}")
eps = 2e-5
symmetry_ctx = build_context(nn=nn, nm=1, device=device, real_dtype=score_type)

def all_pm1_sequences(n: int) -> torch.Tensor:
    # number of rows = 2^n
    rows = 1 << n
    # make integers [0,...,2^n-1] directly on device
    numbers = torch.arange(rows, dtype=torch.int64, device=device).unsqueeze(1)
    # extract bits by shifting
    shifts = torch.arange(n - 1, -1, -1, device=device)
    bits = (numbers >> shifts) & 1
    # map {0,1} -> {-1,+1}
    return (bits * 2 - 1).to(dtype=torch.int8)


def all_pm1_sequences_with_sum(n: int, total_sum: int) -> torch.Tensor:
    if (n + total_sum) % 2 != 0:
        return torch.empty((0, n), dtype=torch.int8, device=device)
    num_ones = (n + total_sum) // 2
    if num_ones < 0 or num_ones > n:
        return torch.empty((0, n), dtype=torch.int8, device=device)
    if num_ones == 0:
        return -torch.ones((1, n), dtype=torch.int8, device=device)
    if num_ones == n:
        return torch.ones((1, n), dtype=torch.int8, device=device)
    positions = torch.combinations(torch.arange(n, device=device), r=num_ones)
    sequences = -torch.ones((positions.shape[0], n), dtype=torch.int8, device=device)
    rows = torch.arange(positions.shape[0], device=device).unsqueeze(1)
    sequences[rows, positions] = 1
    return sequences

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

def build_block_table(required_sum=None):
    if required_sum is None:
        sequences = all_pm1_sequences(nn)
    else:
        sequences = all_pm1_sequences_with_sum(nn, required_sum)
    spectra = torch.fft.rfft(sequences, dim=1)
    power_spectra = torch.view_as_real(spectra).square().sum(dim=-1)
    mask = (power_spectra <= n).all(dim=-1)
    power_spectra = power_spectra[mask]
    autocorrelation_classes = torch.fft.irfft(power_spectra, n=nn).round().to(dtype=torch.int16)
    autocorrelation_classes, class_multiplicities = unique_rows_with_counts(autocorrelation_classes)
    del sequences, spectra, power_spectra, mask
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
partial_sums = canonicalise_automorphism_exact(partial_sums.clone(), symmetry_ctx)
partial_sums, partial_counts = unique_rows_with_counts(partial_sums, partial_counts.clone())
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
    new_partial_sums = torch.empty((0, large_table.shape[1]), device=device, dtype=partial_sums.dtype)
    new_partial_counts = torch.empty((0,), device=device, dtype=torch.long)
    print(f"combination step {i}")
    for j in range(small_table.shape[0]):
        summed_autocorrelations = large_table + small_table[j].unsqueeze(0)
        spectra = torch.fft.rfft(summed_autocorrelations, dim=1).real
        mask = (spectra <= n+eps).all(dim=-1)
        if mask.any():
            canonical_sums = canonicalise_automorphism_exact(summed_autocorrelations[mask], symmetry_ctx)
            new_partial_sums = torch.cat((new_partial_sums, canonical_sums), dim=0)
            new_partial_counts = torch.cat((new_partial_counts, small_counts[j] * large_counts[mask]), dim=0)
            new_partial_sums, new_partial_counts = unique_rows_with_counts(new_partial_sums, new_partial_counts)
    partial_sums = new_partial_sums
    partial_counts = new_partial_counts
    partial_sums = canonicalise_automorphism_exact(partial_sums, symmetry_ctx)
    partial_sums, partial_counts = unique_rows_with_counts(partial_sums, partial_counts)
    print_table_stats(f"after step {i}", partial_sums, partial_counts)

print(f"final exact GS count: {partial_counts.sum().item()}")
