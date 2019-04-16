import numpy as np
from scipy.linalg import block_diag
import time
import pandas as pd

def generate_mat(params):
    n = params[0]
    mat = []
    for i in range(n):
        tmpv = np.zeros([n])
        tmpv[i] = 1
        colv, _ = hss(params, tmpv)
        mat.append(colv)
    return np.column_stack(mat)    


def hss(params, vec):
    n, k, p, u, v, x, y, s, diags = params
    res = np.zeros([n])
    tmpX = np.zeros([2**(p+1)-2, k])
    tmpY = np.zeros([2**(p+1)-2, k])
    card = int(n / (2**p))
    
    # setup lookup table
    get = {}
    for L in range(1, p+1):
        for i in range(2**L):
            get[(L, i)] = (2**L - 1) + i - 1
    for L in range(1, p):
        offset = int((4**L - 4)/3)
        for i in range(2**(L-1)):
            r = 2*i
            c = 2*i+1
            get[(L, r, c)] = offset + r * (2**L) + c
            get[(L, c, r)] = offset + c * (2**L) + r    
    
    start = time.time()
    
    # step 1
    L = p
    si = 0
    for i in range(2**L):
        ei = si + card
        tmpX[get[(L, i)]] = v[i].T.dot(vec[si:ei])
        si = ei

    # step 2
    for L in range(p-1, 0, -1):
        for i in range(2**L):
            tmpX[get[(L,i)]] = y[get[(L,i)]].T.dot(
                np.hstack([tmpX[get[(L+1, 2*i)]], tmpX[get[(L+1, 2*i+1)]]]))

    # step 3
    for L in range(1, p):
        for i in range(2**(L-1)):
            tmpY[get[(L,2*i)]] = s[get[(L,2*i,2*i+1)]].dot(tmpX[get[(L,2*i+1)]])
            tmpY[get[(L,2*i+1)]] = s[get[(L,2*i+1,2*i)]].dot(tmpX[get[(L,2*i)]])

    # step 4
    for L in range(1, p):
        for i in range(2**L):
            tmp = x[get[(L, i)]].dot(tmpY[get[(L,i)]])
            tmpY[get[(L+1,2*i)]] += tmp[:k]
            tmpY[get[(L+1,2*i+1)]] += tmp[k:]

    # step 5
    L = p
    si = 0
    for i in range(2**L):
        ei = si + card
        res[si:ei] = u[i].dot(tmpY[get[(L,i)]]) + diags[i].dot(vec[si:ei])
        si = ei
    
    end = time.time()
    return res, (end - start)


def setup(p, n0, k):
    n = (2**p) * n0
    
    # setup vectors
    u = np.random.random([2**p, n0, k])  # O(nk)
    v = np.random.random([2**p, n0, k])  # O(nk)
    x = np.random.random([2**p-2, 2*k, k])  # O(2nk^2/n0)
    y = np.random.random([2**p-2, 2*k, k])  # O(2nk^2/n0)
    s = np.random.random([int((4**p-4)/3), k, k])  # O(nk^2/n0)

    # generate diagonal blocks
    diags = np.random.random([2**p, n0, n0])
    
    size = np.product(u.shape) + np.product(v.shape) + \
        np.product(x.shape) + np.product(y.shape) + np.product(s.shape)
    ratio = n * n / size

    pa = (n, k, p, u, v, x, y, s, diags)
    mat = generate_mat(pa)
    return pa, mat, ratio


def exp(p, n0, k):
    pa, mat, ratio = setup(p, n0, k)
    n = pa[0]
    
    # run experiments
    vec = np.random.random([n])
    repeat = 20
    
    # run numpy dot
    start = time.time()
    for _ in range(repeat):
        ref = mat.dot(vec)
    end = time.time()
    time_dot = (end - start) * 1000 / repeat
    
    # run naive for loop implementation
    ref2 = np.zeros([n])
    start = time.time()
    for _ in range(repeat):
        for i in range(n):
            ref2[i] = 0  # in case accumulation
            for j in range(n):
                ref2[i] += mat[i,j]*vec[j]
    end = time.time()
    time_naive = (end - start) * 1000 / repeat
    
    # run our hss
    t = 0
    for _ in range(repeat):
        ours, tmpt = hss(pa, vec)
        t += tmpt
    time_hss = t * 1000 / repeat
    
    # check for correctness
    assert np.allclose(ref, ref2) and np.allclose(ours, ref2)
    print("p={}, n0={}, n={}, k={}, ratio={:0.2f}, dot={:0.2f}, naive={:0.2f}, hss={:0.2f}".format(
        p, n0, n, k, ratio, time_dot, time_naive, time_hss))


def profilehss(p, n0, k):
    pa, mat, ratio = setup(p, n0, k)
    n = pa[0]
    
    # run experiments
    vec = np.random.random([n])
    repeat = 20
    
    # run our hss
    t = 0
    for _ in range(repeat):
        ours, tmpt = hss(pa, vec)
        t += tmpt
    time_hss = t * 1000 / repeat
    
    # show results
    print("p={}, n0={}, n={}, k={}, ratio={:0.2f}, hss={:0.2f}".format(
        p, n0, n, k, ratio, time_hss))
    res = {'p': p, 'n0': n0, 'n': n, 'k': k, 'ratio':ratio, 'hss':time_hss}
    return res

    
if __name__ == "__main__":
    exp(p=8, n0=4, k=1)
    
