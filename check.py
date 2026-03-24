#!/usr/bin/env python
# coding: utf-8

import argparse
import base64
import hashlib
import sys
import torch
from PIL import Image

device = "cuda"
if device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit(f"{device=} but CUDA is not available")
if device == "mps" and not torch.backends.mps.is_available():
    raise SystemExit(f"{device=} but MPS is not available")


def parse_args():
    parser = argparse.ArgumentParser(description="Check Hadamard candidates from files or stdin.")
    parser.add_argument("files", nargs="*", help="Input files. Reads stdin if omitted.")
    parser.add_argument("--num-pictures", type=int, default=1, help="Number of Hadamard hits to export as images.")
    parser.add_argument("--group-segment-sums", action=argparse.BooleanOptionalAction, default=True,
                        help="Group segment sums by sorting their absolute values before summarising.")
    return parser.parse_args()


def mir(a):
    return torch.cat((torch.ones_like(a[..., :1]), a, torch.flip(a, (-1,))), dim=-1)
def unfold(m):
    na = m.shape[-1]
    nn = (2*na+3)//5
    nn2 = (nn-1) // 2
    return torch.cat((mir(m[...,:nn2]),
                      mir(m[...,nn2:2*nn2]),
                      mir(m[...,2*nn2:3*nn2]),
                      m[...,3*nn2:]),dim=-1)

def circulant(rows):
    n = rows.shape[1]
    idx = (torch.arange(n).unsqueeze(1) - torch.arange(n).unsqueeze(0)) % n
    return rows[:, idx]


def upblock(batch):
    b, n = batch.shape
    nn = n // 4
    aa, bb, cc, dd = batch.view(b, 4, nn).unbind(dim=1)
    a = circulant(aa)
    b = circulant(bb)
    c = circulant(cc)
    d = circulant(dd)
    df = d.flip(-1)
    # fd = d.flip(-2)
    row0 = torch.cat((a, b, c, df), dim=2)
    row1 = torch.cat((-b, a, -df, c), dim=2)
    row2 = torch.cat((-c, df, a, -b), dim=2)
    row3 = torch.cat((-df, -c, b, a), dim=2)
    return torch.cat((row0, row1, row2, row3), dim=1)

def upblock_mod(vector):
    n = vector.shape[0]
    nn = n // 4
    aa, bb, cc, dd = vector.view(4, nn)
    a = circulant(aa.unsqueeze(0))[0]
    b = 2 * circulant(bb.unsqueeze(0))[0]
    c = 3 * circulant(cc.unsqueeze(0))[0]
    d = 4 * circulant(dd.unsqueeze(0))[0]
    df = d.flip(-1)
    row0 = torch.cat((a, b, c, df), dim=1)
    row1 = torch.cat((-b, a, -df, c), dim=1)
    row2 = torch.cat((-c, df, a, -b), dim=1)
    row3 = torch.cat((-df, -c, b, a), dim=1)
    return torch.cat((row0, row1, row2, row3), dim=0)


def score_matrices(matrices):
    n = matrices.shape[1]
    matrices = matrices.to(torch.float64)
    eye = torch.eye(n, dtype=torch.float64, device=matrices.device).unsqueeze(0)
    diff = matrices @ matrices.transpose(1, 2) - n * eye
    return diff.square().sum(dim=(1, 2)).to(torch.int64)


def convert_lines(lines):
    values = [[1 if c == "+" else -1 for c in line] for line in lines]
    return unfold(torch.tensor(values, dtype=torch.int64, device=device))


def tensor_to_bytes(tensor):
    return bytes(tensor.to("cpu").contiguous().view(-1).tolist())


def save_image(tensor, path, mode):
    h, w = tensor.shape[:2]
    Image.frombytes(mode, (w, h), tensor_to_bytes(tensor)).save(path)


def short_hash(tensor):
    digest = hashlib.blake2b(tensor.to(torch.int8).to(torch.uint8).to("cpu").contiguous().view(-1).numpy().tobytes(), digest_size=4).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")[:5]


def save_pictures(vector, matrix):
    device = matrix.device
    n = vector.shape[0]
    stem = f"hadamard-{n}-{short_hash(vector)}"
    gray = ((matrix + 1) // 2 * 255).to(torch.uint8)
    save_image(gray, f"{stem}.png", "L")
    Image.frombytes("L", (n, n), tensor_to_bytes(gray)).resize((2 * n, 2 * n), Image.NEAREST).save(f"{stem}-large.png")
    mod_matrix = upblock_mod(vector.to(device))
    color = torch.zeros((n, n, 3), dtype=torch.uint8, device=device)
    color[mod_matrix == 1] = torch.tensor((255, 0, 0), dtype=torch.uint8, device=device)
    color[mod_matrix == 2] = torch.tensor((0, 255, 0), dtype=torch.uint8, device=device)
    color[mod_matrix == 3] = torch.tensor((0, 0, 255), dtype=torch.uint8, device=device)
    color[mod_matrix == 4] = torch.tensor((255, 255, 0), dtype=torch.uint8, device=device)
    save_image(color, f"{stem}-colour.png", "RGB")
    Image.frombytes("RGB", (n, n), tensor_to_bytes(color)).resize((2 * n, 2 * n), Image.NEAREST).save(f"{stem}-colour-large.png")
    perm_rows = torch.randperm(n, device=device)
    perm_cols = torch.randperm(n, device=device)
    signs_x = 2 * torch.randint(0, 2, (n,), dtype=torch.int64, device=device) - 1
    signs_y = 2 * torch.randint(0, 2, (n,), dtype=torch.int64, device=device) - 1
    randomised = signs_x[:, None] * signs_y[None, :] * matrix[perm_rows][:, perm_cols]
    gray_randomised = ((randomised + 1) // 2 * 255).to(torch.uint8)
    save_image(gray_randomised, f"{stem}-randomised.png", "L")


def process_lines(lines, source_name, pictures_remaining, group_segment_sums):
    stripped = [line.strip() for line in lines if line.strip()]
    summary = []
    groups = {}
    for line in stripped:
        if any(c not in "+-" for c in line):
            print(f"invalid input: only '+' and '-' are allowed: {line!r}", file=sys.stderr)
            continue
        """
        if len(line) % 4 != 0:
            print(f"invalid input length (must be divisible by 4): {len(line)}", file=sys.stderr)
            continue
        """
        groups.setdefault(len(line), []).append(line)
    for na, strings in groups.items():
        batch = convert_lines(strings)
        n = batch.shape[1]
        segment_sums = batch.view(batch.shape[0], 4, -1).sum(dim=2)
        if group_segment_sums:
            segment_sums = torch.sort(segment_sums.abs(), dim=1).values
        matrices = upblock(batch)
        scores = score_matrices(matrices)
        pairs = torch.cat((segment_sums.to(torch.int64), scores.unsqueeze(1)), dim=1)
        unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)
        for pair, count in zip(unique_pairs.tolist(), counts.tolist()):
            summary.append((f"n={n} segment_sums={tuple(pair[:4])} score={pair[4]}", count))
        if pictures_remaining > 0:
            hadamard_idx = torch.nonzero(scores == 0, as_tuple=True)[0].tolist()
            for j in hadamard_idx[:pictures_remaining]:
                save_pictures(batch[j], matrices[j])
            pictures_remaining -= min(pictures_remaining, len(hadamard_idx))
    print(f"==> {source_name} <==")
    if summary:
        for result, count in sorted(summary):
            print(f"{count:7d} {result}")
    else:
        print("(no valid input lines)")
    return pictures_remaining


def process_file(path, pictures_remaining, group_segment_sums):
    with open(path) as handle:
        return process_lines(handle, path, pictures_remaining, group_segment_sums)


def main():
    args = parse_args()
    pictures_remaining = max(args.num_pictures, 0)
    if args.files:
        for i, path in enumerate(args.files):
            if i > 0:
                print()
            pictures_remaining = process_file(path, pictures_remaining, args.group_segment_sums)
        return
    process_lines(sys.stdin, "stdin", pictures_remaining, args.group_segment_sums)


if __name__ == "__main__":
    main()
