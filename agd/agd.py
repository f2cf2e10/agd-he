import numpy as np
from agd.seal.seal import Encryptor, Evaluator, CKKSEncoder, \
    Ciphertext, GaloisKeys, RelinKeys, Plaintext
from agd.matrix.ours import matrix_multiplication as ours
from agd.matrix.utils import encrypt_array, rescale_and_mod_switch_y_and_add_x, \
    rescale_and_mod_switch_y_and_multiply_x, squarify, encode


def agd_qp(Q: np.ndarray, p: np.ndarray, n: int, x0: np.ndarray) \
        -> np.ndarray:
    def f(x): return x.T.dot(Q).dot(x) + p.dot(x)

    def df(x): return Q.dot(x) + p

    eigenvals = np.linalg.eigvals(Q)
    beta = np.max(eigenvals)
    alpha = np.min(eigenvals)
    kappa = beta / alpha
    gamma = (kappa ** 0.5 - 1) / (kappa ** 0.5 + 1)
    x = x0
    y = x0
    steps = [None] * (n + 1)
    steps[0] = x
    for t in range(1, n + 1):
        y_ = y
        y = x - 1 / beta * df(x)
        x = (1 + gamma) * y - gamma * y_
        steps[t] = x
    return steps


def he_agd_qp_ckks(Q_enc: Ciphertext, p_enc: Ciphertext, d: int, alpha: float, beta: float, T: int,
                   x0_enc: Ciphertext, y0_enc: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
                   gal_keys: GaloisKeys, relin_keys: RelinKeys, encryptor: Encryptor,
                   scale: float = 2 ** 40) -> Ciphertext:
    y_enc_ = y0_enc
    x_enc_ = x0_enc
    kappa = beta / alpha
    gamma = (kappa ** 0.5 - 1) / (kappa ** 0.5 + 1)

    beta_ = Plaintext()
    encoder.encode(-1./beta, p_enc.scale(), beta_)
    p_enc_beta = Ciphertext()
    evaluator.multiply_plain(p_enc, beta_, p_enc_beta)
    evaluator.relinearize_inplace(p_enc_beta, relin_keys)
    evaluator.rescale_to_next_inplace(p_enc_beta)

    steps = [None] * (T + 1)
    steps[0] = x_enc_
    for t in range(1, T + 1):
        Q_dot_x_enc = ours(Q_enc, x_enc_, evaluator,
                           encoder, gal_keys, relin_keys,
                           d, -1. / beta, scale)

        p_enc_beta.set_scale(Q_dot_x_enc.scale())
        evaluator.mod_switch_to_inplace(
            p_enc_beta, Q_dot_x_enc.parms_id())
        evaluator.add_inplace(Q_dot_x_enc, p_enc_beta)
        y_enc = rescale_and_mod_switch_y_and_add_x(
            Q_dot_x_enc, x_enc_, evaluator)

        # print("Modulus chain index for y_enc: {}".format(
        #    context.get_context_data(y_enc.parms_id()).chain_index()))

        gamma_1_y_enc = rescale_and_mod_switch_y_and_multiply_x(
            y_enc, 1 + gamma, evaluator, encoder, relin_keys)
        gamma_y_enc_ = rescale_and_mod_switch_y_and_multiply_x(
            y_enc_, -gamma, evaluator, encoder, relin_keys)

        # print("Modulus chain index for y_enc: {}".format(
        #    context.get_context_data(y_enc.parms_id()).chain_index()))
        x_enc = rescale_and_mod_switch_y_and_add_x(
            gamma_1_y_enc, gamma_y_enc_, evaluator)
        y_enc_ = y_enc
        x_enc_ = x_enc

        steps[t] = x_enc

    return steps
