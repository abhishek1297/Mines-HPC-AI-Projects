import numpy as np

# 4.3
def power_iters(A, Omega, power_iter=2):
    
    Y = A @ Omega
    for q in range(power_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    
    return Q