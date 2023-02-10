import numpy as np
from scipy.linalg.interpolative import interp_decomp, id_to_svd
from sklearn.datasets import make_low_rank_matrix
from stage_a import *
    
# 5.1
def direct_svd(A, Q):
    
    B = Q.T @ A
    U_tilde, S, V = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    
    return U, S, V



# 5.2
def row_extract_svd(A, Q, k, p):
    """
    Iff || A - (Q.T @ Q @ A) || <= eps
    """
    l = k + p
    idx, proj = interp_decomp(Q.T, eps_or_k=l)
    X = np.zeros(Q.T.shape)
    X[:, idx[:l]] = np.eye(l)
    X[:, idx[l:]] = proj
    
    Aj = A[idx[:l], :]
    U_tilde, S, V = np.linalg.svd(Aj, full_matrices=False)
    U = X.T @ U_tilde
    
    return U, S, V

def get_omega(A, k, p=0):
    """
    k: rank
    p: oversampling
    """
    return np.random.normal(0, 1, (A.shape[1], k+p))

def combo1(A, k, p, q):
    
    Omega = get_omega(A, k, p)
    Q = power_iters(A, Omega, q)
    
    return *direct_svd(A, Q), Q

def combo2(A, k, p, q):
    Omega = get_omega(A, k, p)
    Q = power_iters(A, Omega, q)

    return *row_extract_svd(A, Q, k, p), Q

from time import time

if __name__ == "__main__":

    st = time()

    combo1(make_low_rank_matrix(1000, 1000, effective_rank=50), 100, 5, 2)

    print(time() - st)

    st = time()

    combo2(make_low_rank_matrix(1000, 1000, effective_rank=50), 100, 5, 2)

    print(time() - st)