#!/usr/bin/env python
# coding: utf-8

# a script to count matrices up to all symmetries. takes input from stdin

import sys
from itertools import permutations

def rot(i,a):
    return a[i:nn]+a[:i]+a[nn+i:2*nn]+a[nn:nn+i]+a[2*nn+i:3*nn]+a[2*nn:2*nn+i]+a[3*nn+i:]+a[3*nn:3*nn+i]

def rot2(i,a):
    return a[:2*nn]+a[2*nn+i:3*nn]+a[2*nn:2*nn+i]+a[3*nn:]

def minrot(a):
    aa=a;
    for _ in range(nn):
        a=rot(1,a)
        if a<aa:
            aa=a
    a=aa
    for _ in range(nn):
        a=rot2(1,a)
        if a<aa:
            aa=a
    a=aa[::-1]
    for _ in range(nn):
        a=rot(1,a)
        if a<aa:
            aa=a
    a=aa
    for _ in range(nn):
        a=rot2(1,a)
        if a<aa:
            aa=a
    return aa

perms = list(p for p in permutations(range(4)) if p[2]==2)

def flip(bool,s):
    return (''.join('+' if x=='-' else '-' for x in s) if bool else s)

def minsym(a):
    global nn
    n = len(a)
    nn = n//4
    # include signs and permutations
    aa=a
    al=[a[:nn],a[nn:2*nn],a[2*nn:3*nn],a[3*nn:]]
    for j in range(16):
        for p in perms:
            b = ''.join([flip((j>>i)&1,al[p[i]]) for i in range(4)])
            bb = minrot(b)
            if bb<aa:
                aa=bb
    return aa

strings = {minsym(line.strip()) for line in sys.stdin}

print(len(strings))
