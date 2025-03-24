import sys
import numpy as np, numpy.linalg as nl
import scipy.linalg as sl
import math

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
