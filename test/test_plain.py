from agd.matrix.utils import lin_trans, squarify
from agd.matrix.ours import generate_vk_matrix,  generate_wk_matrix
import numpy as np
from agd.matrix.jlks import phi_permutation_k, psi_permutation_k, sigma_permutation, \
tau_permutation


def test_lin_trans():
    d = 20
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    b = np.array([np.random.randn(d)])
    assert np.max(np.abs(lin_trans(A, b[0], d) - A.dot(b.T).T[0])) < 1E-10

def test_matrix_multiplication_jkls():
    d = 2
    N = d*d
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    B = np.array([np.random.randn(d*d)]).reshape((d, d))
    a = A.flatten()
    b = B.flatten()
    usigma = sigma_permutation(d, N)
    utau = tau_permutation(d, N)
    a0 = lin_trans(usigma, a, d*d)
    b0 = lin_trans(utau, b, d*d)
    ak = []
    bk = []
    for k in range(1, d):
        vk = phi_permutation_k(k, d, N)
        wk = psi_permutation_k(k, d, N)
        ak = ak + [lin_trans(vk, a0, d*d)]
        bk = bk + [lin_trans(wk, b0, d*d)]
    acc = a0 * b0
    for k in range(d-1):
        acc = acc + ak[k] * bk[k]
    assert np.max(np.abs(A.dot(B) - acc.reshape(d, d))) < 1E-10


def test_matrix_multiplication_ours():
    d = 20
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    B = np.array([np.random.randn(d*d)]).reshape((d, d))
    a = A.flatten()
    b = B.flatten()
    acc = 0 * a
    for l in range(d):
        al = lin_trans(generate_vk_matrix(d, l), a, d*d)
        bl = lin_trans(generate_wk_matrix(d, l), b, d*d)
        acc = acc +  al * bl
    assert np.max(np.abs(A.dot(B) - acc.reshape(d, d))) < 1E-10


def test_matrix_multiplication_ours_withzeros():
    d = 20
    N = 2**10
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    B = np.array([np.random.randn(d*d)]).reshape((d, d))
    a = np.concatenate([A.flatten(), np.zeros(N - d*d)])
    b = np.concatenate([B.flatten(), np.zeros(N - d*d)])
    acc = 0 * a
    for l in range(d):
        Vk = squarify(generate_vk_matrix(d, l), 0, N)
        Wk = squarify(generate_wk_matrix(d, l), 0, N)
        acc = acc + lin_trans(Vk, a, d*d) * lin_trans(Wk, b, d*d)
    assert np.max(np.abs(A.dot(B) - acc[0:d*d].reshape(d, d))) < 1E-10

