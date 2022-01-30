import numpy as np
import itertools
from agd.seal.seal import Evaluator, Ciphertext, CKKSEncoder, \
    CiphertextVector, GaloisKeys, RelinKeys
from agd.matrix.utils import lin_trans_enc, ca_x_cb


def sigma_permutation(d, nmax, val=1.0):
    n = d*d
    u_sigma_matrix = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        u_sigma_matrix[d*i+j, d*i+((i+j) % d)] = val
    output = np.kron(np.eye(2), u_sigma_matrix)
    return output[0:nmax, 0:nmax]


def tau_permutation(d, nmax, val=1.0):
    n = d*d
    u_tau_matrix = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        u_tau_matrix[d*i+j, d*((i+j) % d)+j] = val
    output = np.kron(np.eye(2), u_tau_matrix)
    return output[0:nmax, 0:nmax]


def phi_permutation_k(k, d, nmax, val=1.0):
    n = d*d
    vk_matrix = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        vk_matrix[d*i+j, d*i+((j+k) % d)] = val
    return vk_matrix


def psi_permutation_k(k, d, nmax, val=1.0):
    n = d*d
    wk_matrix = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        wk_matrix[d*i+j, d*((i+k) % d)+j] = val
    return wk_matrix


def matrix_multiplication(ct_a: Ciphertext, ct_b: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
          gal_keys: GaloisKeys, relin_keys: RelinKeys, d: int, scale: float = 1.0) -> Ciphertext:
    """
    Matrix product of CiphertextA and CiphertextB
    Inspired by "Secure Outsourced Matrix Computationand Application to Neural Networks?"
    Link: https://eprint.iacr.org/2018/1041.pdf
    """
    nmax = encoder.slot_count()
    if d*d > nmax/2:
        raise Exception("Matrix dimenson is higher than the one suported by the encoder")
    u_sigma_matrix = sigma_permutation(d, nmax, np.sign(scale) * np.abs(scale)**0.5)
    u_tau_matrix = tau_permutation(d, nmax, np.abs(scale)**0.5)
    ct_a0 = lin_trans_enc(u_sigma_matrix, ct_a, evaluator,
                          encoder, gal_keys, relin_keys)
    ct_b0 = lin_trans_enc(u_tau_matrix, ct_b, evaluator, encoder, gal_keys, relin_keys)
    ct_ak = CiphertextVector()
    ct_bk = CiphertextVector()
    for k in range(1, d):
        vk_matrix = phi_permutation_k(k, d, nmax, 1.0)
        wk_matrix = psi_permutation_k(k, d, nmax, 1.0)
        if vk_matrix.sum() == 0 or wk_matrix.sum() == 0:
            continue

        ct_ak.append(lin_trans_enc(vk_matrix, ct_a0, evaluator,
                     encoder, gal_keys, relin_keys))
        ct_bk.append(lin_trans_enc(wk_matrix, ct_b0, evaluator,
                     encoder, gal_keys, relin_keys))

    ct_ab = ca_x_cb(ct_a0, ct_b0, evaluator, relin_keys)
    for k in range(len(ct_ak)):
        ct_abk = ca_x_cb(ct_ak[k], ct_bk[k], evaluator, relin_keys)
        parms_id = ct_abk.parms_id()
        evaluator.mod_switch_to_inplace(ct_ab, parms_id)
        ct_abk.set_scale(ct_ab.scale())
        evaluator.add_inplace(ct_ab, ct_abk)
    return ct_ab



