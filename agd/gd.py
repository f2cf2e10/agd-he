import numpy as np
from agd.seal.seal import Encryptor, Evaluator, CKKSEncoder, \
    Ciphertext, GaloisKeys, RelinKeys, Plaintext
from agd.matrix.ours import matrix_multiplication as ours
from agd.matrix.utils import encrypt_array, squarify, rescale_and_mod_switch_y_and_add_x

def gd_qp(Q: np.ndarray, p: np.ndarray, T: int, x0: np.ndarray) -> np.ndarray:
    def f(x): return x.T.dot(Q).dot(x) + p.dot(x)
    def df(x): return Q.dot(x) + p
    # Pre-calculations
    eigenvals = np.linalg.eigvals(Q)
    beta = np.max(eigenvals)
    alpha = np.min(eigenvals)
    steps = [None] * (T+1)
    x = x0
    steps[0] = x
    eta = 2/(alpha+beta)
    for t in range(1,T+1):
        x = x - eta * df(x)
        steps[t] = x
    return steps 


def he_gd_qp(Q_enc: Ciphertext, p_enc: Ciphertext, d:int, alpha: float, beta: float, T: int,
              x0_enc: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
              gal_keys: GaloisKeys, relin_keys: RelinKeys, encryptor:Encryptor,
              scale: float=2.0**40) -> Ciphertext:
    steps = [None] * (T+1)
    x_enc = x0_enc
    steps[0] = x_enc
    eta = 2/(alpha+beta)

    eta_ = Plaintext()
    encoder.encode(-eta, p_enc.scale(), eta_)
    p_enc_eta = Ciphertext()
    evaluator.multiply_plain(p_enc, eta_, p_enc_eta)
    evaluator.relinearize_inplace(p_enc_eta, relin_keys)
    evaluator.rescale_to_next_inplace(p_enc_eta)

    for t in range(1, T+1):
        Q_dot_x_enc = ours(Q_enc, x_enc, evaluator,
                                 encoder, gal_keys, relin_keys, 
                                 d, -eta, scale)

        p_enc_eta.set_scale(Q_dot_x_enc.scale())
        evaluator.mod_switch_to_inplace(
            p_enc_eta, Q_dot_x_enc.parms_id())
        evaluator.add_inplace(Q_dot_x_enc, p_enc_eta)
        x_enc = rescale_and_mod_switch_y_and_add_x(
            Q_dot_x_enc, x_enc, evaluator)

        steps[t] = x_enc
        
        #Q_dot_x_enc = ours(Q_enc, x, evaluator, encoder, gal_keys, relin_keys, d, -eta, scale)
        #p_enc = ours(I1, p_enc, evaluator, encoder, gal_keys, relin_keys, d, -eta if t==1 else 1., scale)
        #x = ours(I, x, evaluator, encoder, gal_keys, relin_keys, d, 1., scale)
        #nabla = Ciphertext()
        #evaluator.add(Q_dot_x_enc, p_enc, nabla)
        #evaluator.add_inplace(x, nabla)
        #steps[t] = x

    return steps

def he_random_gd_qp(Q_enc: Ciphertext, p_enc: Ciphertext, d:int, alpha: float, beta: float, T: int,
              x0_enc: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
              gal_keys: GaloisKeys, relin_keys: RelinKeys, encryptor:Encryptor,
              scale: float=2.0**40) -> Ciphertext:
    steps = [None] * (T+1)
    nablas = [None] * (T+1)
    x = x0_enc
    steps[0] = x
    eta = 2/(alpha+beta)
    I = encrypt_array(squarify(np.eye(d), 0.0, d), encryptor, encoder, scale)

    for t in range(1, T+1):
        x = ours(I, x, evaluator, encoder, gal_keys, relin_keys, d, 1., scale)
        rand = 1E-7*np.random.randn(2).reshape(2,1)
        nabla = encrypt_array(squarify(rand, 0.0, d), encryptor, encoder, scale)
        nabla.set_scale(x.scale())
        nablas[t] = nabla
        evaluator.mod_switch_to_inplace(nabla, x.parms_id())
        evaluator.add_inplace(x, nabla)
        steps[t] = x

    return steps

