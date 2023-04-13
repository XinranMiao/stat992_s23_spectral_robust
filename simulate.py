import numpy as np
from scipy.stats import bernoulli
def find_j_eigenvec(values, vectors, j = 1):
    """
    Find (eigenvalue, eigenvector) for the j-th largest eigenvalue (in absolute value)
    """
    n = len(values)
    sort_index = np.argsort(abs(values), axis = 0)
    val = values[sort_index[n - j]]
    vec = vectors[:, sort_index[n - j]]
    
    return val, vec

def conn2adj(B):
    """
    Generate adjacency matrix from connectivity matrix
    """
    n = B.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            A[i, j] = bernoulli.rvs(size = 1, p = B[i, j])
    return A + A.transpose()

def cal_mis_rate(y, yhat):
    rate = np.mean(abs(y - yhat))
    return rate