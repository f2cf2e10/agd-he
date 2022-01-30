import numpy as np
import itertools
from agd.seal.seal import Evaluator, Ciphertext, CKKSEncoder, \
    CiphertextVector, GaloisKeys, RelinKeys
from agd.matrix.utils import lin_trans_enc, ca_x_cb


def generate_vk_matrix(d: int, k: int, val: float = 1.0) -> np.ndarray:
    n = d*d
    uk_matrix = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        uk_matrix[d*i + j, d*i + ((i + j + k) % d)] = val
    return uk_matrix


def generate_wk_matrix(d: int, k: int, val: float = 1.0) -> np.ndarray:
    n = d*d
    wk_matrix = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        wk_matrix[d*i+j, d*((i+j+k) % d)+j] = val
    return wk_matrix


def matrix_multiplication(ct_a: Ciphertext, ct_b: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
           gal_keys: GaloisKeys, relin_keys: RelinKeys, d: int, scale: float = 1.0) -> Ciphertext:
    """
    Proposed method for Matrix product of CiphertextA and CiphertextB 
    """
    nmax = encoder.slot_count()
    if d*d > nmax/2:
        raise Exception("Matrix dimenson is higher than the one suported by the encoder")
    ct_ak = CiphertextVector()
    ct_bk = CiphertextVector()
    for k in range(0, d):
        vk_matrix = generate_vk_matrix(d, k, scale)
        wk_matrix = generate_wk_matrix(d, k, 1.0)
        if vk_matrix.sum() == 0 or wk_matrix.sum() == 0:
            continue

        ct_ak.append(lin_trans_enc(vk_matrix, ct_a, evaluator,
                     encoder, gal_keys, relin_keys))
        ct_bk.append(lin_trans_enc(wk_matrix, ct_b, evaluator,
                     encoder, gal_keys, relin_keys))

    ct_ab = Ciphertext()
    for k in range(len(ct_ak)):
        ct_abk = ca_x_cb(ct_ak[k], ct_bk[k], evaluator, relin_keys)
        if (k == 0):
            ct_ab = ct_abk
        else:
            evaluator.add_inplace(ct_ab, ct_abk)
    return ct_ab
