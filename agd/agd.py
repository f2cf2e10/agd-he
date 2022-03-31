import numpy as np
from agd.seal.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, Encryptor, CoeffModulus, \
    Evaluator, Decryptor, CKKSEncoder, IntVector, Plaintext, Ciphertext, \
    GaloisKeys, RelinKeys, PublicKey, sec_level_type, DoubleVector
from agd.matrix.ours import matrix_multiplication as ours
from agd.matrix.utils import rescale_and_mod_switch_y_and_add_x, \
    rescale_and_mod_switch_y_and_multiply_x
from typing import Tuple

def agd_qp(Q: np.ndarray, p: np.ndarray, n: int, x0: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    def f(x): return x.T.dot(Q).dot(x) + p.dot(x)
    def df(x): return Q.dot(x) + p
    eigenvals = np.linalg.eigvals(Q)
    beta = np.max(eigenvals)
    alpha = np.min(eigenvals)
    kappa = beta/alpha
    gamma = (kappa**0.5-1)/(kappa**0.5+1)
    print("kappa: {}".format(kappa))
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


def he_agd_qp(Q: Ciphertext, p: Ciphertext, d:int, beta: float, kappa: float, T: int,
              x0: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
              gal_keys: GaloisKeys, relin_keys: RelinKeys) -> Ciphertext:
    y_ = x0
    x_ = x0
    gamma = (kappa**0.5-1)/(kappa**0.5+1)

    beta_ = Plaintext()
    encoder.encode(-1./beta, p.scale(), beta_)
    p_beta = Ciphertext()
    evaluator.multiply_plain(p, beta_, p_beta)
    evaluator.relinearize_inplace(p_beta, relin_keys)
    evaluator.rescale_to_next_inplace(p_beta)

    for t in range(T):
        MMultQ_enc_x_enc_ = ours(Q, x_, evaluator, encoder, gal_keys, relin_keys, d, -1./beta)
        p_beta.set_scale(MMultQ_enc_x_enc_.scale())

        evaluator.mod_switch_to_inplace(
            p_beta, MMultQ_enc_x_enc_.parms_id())
        evaluator.add_inplace(MMultQ_enc_x_enc_, p_beta)
        y = rescale_and_mod_switch_y_and_add_x(
            MMultQ_enc_x_enc_, x_, evaluator)

        gamma_1_y_enc = rescale_and_mod_switch_y_and_multiply_x(
            y, 1+gamma, evaluator, encoder, relin_keys)
        gamma_y_enc_ = rescale_and_mod_switch_y_and_multiply_x(
            y_, -gamma, evaluator, encoder, relin_keys)

        x = rescale_and_mod_switch_y_and_add_x(
            gamma_1_y_enc, gamma_y_enc_, evaluator)
        y_ = y
        x_ = x

    return x