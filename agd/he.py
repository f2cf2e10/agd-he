import numpy as np
from agd.seal3_7_2.seal import Evaluator, Ciphertext, CKKSEncoder, \
    GaloisKeys, RelinKeys

def acg_qp(Q: np.ndarray, p: np.ndarray, beta: float, alpha: float, n: int, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def f(x): return x.T.dot(Q).dot(x) + p.dot(x)
    def df(x): return Q.dot(x) + p
    x = x0
    y = x0
    kappa = beta/alpha
    gamma = (kappa**0.5-1)/(kappa**0.5+1)
    for t in range(n):
        y_ = y
        y = x - 1/beta * df(x)
        x_ = x
        x = (1 + gamma) * y - gamma * y_
    tol = np.abs(f(x) - f(x_))
    print("The error after {} steps is: {} ".format(n, tol))
    return x, tol


def he_acg_qp(Q: Ciphertext, p: Ciphertext, beta: float, kappa: float, n: int, c0: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder, gal_keys: GaloisKeys, relin_keys: RelinKeys) -> Tuple[Ciphertext, Ciphertext]:
    c = c0
    d = c0
    gamma = (kappa**0.5-1)/(kappa**0.5+1)
    for t in range(n):
        d_ = d
        # d = evaluator.add_inplace(c, - 1/beta * df(c)
        c_ = c
        c = (1 + gamma) * d - gamma * d_
    tol = np.abs(f(c) - f(c_))
    return c, tol