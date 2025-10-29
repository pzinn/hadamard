#!/usr/bin/env python
# coding: utf-8

# a script to check that matrices are Hadamard. takes input from stdin
# also export H-matrix as picture

import sys
import numpy as np
import scipy.linalg as sl
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)


def upblock(x):
    na = x.shape[0]
    nn = na // 3
    _aa, _cc, _dd = x.reshape(3, nn)
    _bb = _aa
    A = sl.circulant(_aa)
    B = sl.circulant(_bb)
    C = sl.circulant(_cc)
    D = sl.circulant(_dd)
    F = np.fliplr(np.identity(nn))
    return np.block([[A, B@F, C@F, D@F],
                     [-B@F, A, -F@D, F@C],
                     [-C@F, F@D, A, -F@B],
                     [-D@F, -F@C, F@B, A]])

def upblock_mod(x):
    na = x.shape[0]
    nn = na // 3
    _aa, _cc, _dd = x.reshape(3, nn)
    _bb = _aa
    A = sl.circulant(_aa)
    B = 2*sl.circulant(_bb)
    C = 3*sl.circulant(_cc)
    D = 4*sl.circulant(_dd)
    F = np.fliplr(np.identity(nn))
    return np.block([[A, B@F, C@F, D@F],
                     [-B@F, A, -F@D, F@C],
                     [-C@F, F@D, A, -F@B],
                     [-D@F, -F@C, F@B, A]])


def score(a):
    na = len(a)
    n = na // 3 * 4
    m = upblock(a)
    neye = n * np.eye(n, dtype=np.int64)
    mm = m.dot(m.T)-neye
    return np.sum(mm**2)
    # cst = n/2 * math.log(n)
    # return cst-nl.slogdet(m)[1]


def convert(s):
    return np.array([1 if c == "+" else -1 for c in s], dtype=np.int64)

img_saved = False

def treat(s):
    global img_saved
    a = convert(s)
    n = len(a)//3*4
    # print(m)
    s = score(a)
    print(n, s)
    if s==0 and not img_saved:
        m = upblock(a)
        # Convert ±1 to 0/255
        img_array = ((m + 1) // 2 * 255).astype(np.uint8)
        # Create image
        img = Image.fromarray(img_array, mode='L')  # 'L' = grayscale
        # Save as PNG
        img.save(f"hadamard-{n}.png")
        # Optional: enlarge for visibility (nearest-neighbor)
        img.resize((2*n, 2*n), Image.NEAREST).save(f"hadamard-{n}-large.png")
        # now colour version
        m = upblock_mod(a)
        color_map = {
            -1: (0, 0, 0),        # black
            -2: (0, 0, 0),
            -3: (0, 0, 0),
            -4: (0, 0, 0),
             1: (255, 0, 0),      # red
             2: (0, 255, 0),      # green
             3: (0, 0, 255),      # blue
             4: (255, 255, 0),    # yellow
        }
        img_array = np.zeros((n, n, 3), dtype=np.uint8)
        for val, color in color_map.items():
            img_array[m == val] = color
        # Create and save image
        img = Image.fromarray(img_array, mode="RGB")
        img.save(f"hadamard-{n}-colour.png")
        # Optional enlargement (for display)
        img.resize((2*n, 2*n), Image.NEAREST).save(f"hadamard-{n}-colour-large.png")
        #
        img_saved = True  # only do it once

def main():
    for line in sys.stdin:  # Reads each line from standard input
        line = line.strip()  # Remove trailing newline and spaces
        if line:  # Ignore empty lines
            treat(line)  # Apply function and print result


if __name__ == "__main__":
    main()
