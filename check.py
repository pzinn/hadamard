import sys
import numpy as np, numpy.random as nr, numpy.linalg as nl
import scipy.optimize as so, scipy.linalg as sl, scipy.sparse as sp
import math
import random

np.set_printoptions(threshold=sys.maxsize)

def upblock(x):
    _aa, _bb, _cc, _dd = x.reshape(4,-1)
    A = sl.circulant(_aa)
    B = sl.circulant(_bb)
    C = np.fliplr(sl.circulant(_cc))
    D = sl.circulant(_dd)
    return np.block([[A, B, C, D], [-B, A, -D, C], [-C, D, A, -B], [-D, -C, B, A]])

def score(a):
    n = len(a)
    m = upblock(a)
    neye = n * np.eye(n,dtype=np.int64)
    mm=m.dot(m.T)-neye
    return np.sum(mm**2)
    #cst = n/2 * math.log(n)
    #return cst-nl.slogdet(m)[1]

def random_rotate(x):
    arr = x.reshape(4,-1)
    n = arr.shape[1]
    k = random.randrange(n)
    rotated = np.roll(arr, k, axis=1)
    return rotated.ravel()  # Flatten back into 1D array

def random_rotate2(x): #NO
    arr = x.reshape(4,-1)
    n = arr.shape[1]
    k = random.randrange(n)
    rotated = np.array([np.roll(arr[i], random.randrange(n)) for i in range(4)])  # Rotate each subarray
    return rotated.ravel()  # Flatten back into 1D array

def random_rotate3(x):
    arr = x.reshape(4,-1)
    n = arr.shape[1]
    k = random.randrange(n)
    kk = random.randrange(n)
    rotated = np.array([np.roll(arr[i], k if i!=2 else kk) for i in range(4)])  # Rotate each subarray
    return rotated.ravel()  # Flatten back into 1D array

def convert(s):
    return np.array([1 if c=="+" else -1 for c in s],dtype=np.int64)

def treat(s):
    a = convert(s)
    n = len(a)
    #print(m)
    print(n,score(a))

def main():
    for line in sys.stdin:  # Reads each line from standard input
        line = line.strip()  # Remove trailing newline and spaces
        if line:  # Ignore empty lines
            treat(line)  # Apply function and print result

if __name__ == "__main__":
    main()
