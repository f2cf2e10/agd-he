import numpy as np
from agd.seal.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, Encryptor, CoeffModulus, \
    Evaluator, Decryptor, CKKSEncoder, IntVector, Plaintext, Ciphertext, \
    GaloisKeys, RelinKeys, PublicKey, sec_level_type, DoubleVector
from typing import Tuple

from agd.matrix.ours import matrix_multiplication as ours
from agd.matrix.utils import rescale_and_mod_switch_y_and_add_x, \
    rescale_and_mod_switch_y_and_multiply_x

def gd_qp(Q: np.ndarray, p: np.ndarray, n: int, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def f(x): return x.T.dot(Q).dot(x) + p.dot(x)
    def df(x): return Q.dot(x) + p
    # Pre-calculations
    eigenvals = np.linalg.eigvals(Q)
    beta = np.max(eigenvals)
    alpha = np.min(eigenvals)
    x_ = x = x0
    eta = 2/(alpha+beta)
    for t in range(n):
        x_ = x
        x = x - eta * df(x)
    tol = np.abs(f(x) - f(x_))
    print("The error after {} steps is: {} ".format(n, tol))
    return x, tol


def he_gd_qp(Q: Ciphertext, p: Ciphertext, d:int, alpha: float, beta: float, T: int,
              x0: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
              gal_keys: GaloisKeys, relin_keys: RelinKeys) -> Ciphertext:
    x_ = x0
    eta = 2/(alpha+beta)

    eta_ = Plaintext()
    encoder.encode(-eta, p.scale(), eta_)
    p_eta= Ciphertext()
    evaluator.multiply_plain(p, eta_, p_eta)
    evaluator.relinearize_inplace(p_eta, relin_keys)
    evaluator.rescale_to_next_inplace(p_eta)

    for t in range(T):
        MMultQ_enc_x_enc_ = ours(Q, x_, evaluator, encoder, gal_keys, relin_keys, d, -eta)
        p_eta.set_scale(MMultQ_enc_x_enc_.scale())

        evaluator.mod_switch_to_inplace(
            p_eta, MMultQ_enc_x_enc_.parms_id())
        evaluator.add_inplace(MMultQ_enc_x_enc_, p_eta)
        x = rescale_and_mod_switch_y_and_add_x(
            MMultQ_enc_x_enc_, x_, evaluator)
        x_ = x

    return x