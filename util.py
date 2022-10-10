import numpy as np
import numpy.random as npr

def random_rotation(n, theta=None): # n: data dimension, theta: angle of rotation
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()
    if n == 1:
        return np.random.rand() * np.eye(1)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)

def sum_tuples(a, b):
    assert a or b
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(ai + bi for ai, bi in zip(a, b))

def kron_A_N(A, N):  # Simulates np.kron(A, np.eye(N))
    m,n = A.shape
    out = np.zeros((m,N,n,N),dtype=A.dtype)
    r = np.arange(N)
    out[:,r,:,r] = A
    out.shape = (m*N,n*N)
    return out