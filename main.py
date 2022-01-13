# Inspired by "Secure Outsourced Matrix Computationand Application to Neural Networks?"
# Link: https://eprint.iacr.org/2018/1041.pdf
from agd.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, \
    Encryptor, CoeffModulus, Evaluator, Decryptor, \
    CKKSEncoder, IntVector, Plaintext, Ciphertext, GaloisKeys, \
    RelinKeys
import numpy as np
from agd.matrix.jlks import matrix_multiplication 
from agd.matrix.utils import encrypt_array, squarify, decrypt_array, \
    rescale_and_mod_switch, rescale_and_mod_switch_y_and_add_x, \
    rescale_and_mod_switch_y_and_multiply_x


# Problem Inputs plain AGD
Q = np.array([[2, .5], [.5, 1]])
p = np.array([[1.0], [1.0]])
# Matrix dimension
d, _ = Q.shape

x0 = np.array([np.random.randn(d)]).T
# Steps
T = 14

# Pre-calculations
eigenvals = np.linalg.eigvals(Q)
beta = np.max(eigenvals)
alpha = np.min(eigenvals)
kappa = beta/alpha
gamma = (kappa**0.5-1)/(kappa**0.5+1)

# HE parameters
scale = 2.0**40

parms = EncryptionParameters(scheme_type.CKKS)

poly_modulus_degree = 2**(T+1)
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(
    poly_modulus_degree, IntVector([60] + [40]*T + [60])))

context = SEALContext.Create(parms)
print_parameters(context)

keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()
relin_keys = keygen.relin_keys()
gal_keys = keygen.galois_keys()
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)

ckks_encoder = CKKSEncoder(context)

# Problem Inputs HE AGD
Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
p_enc = encrypt_array(p, encryptor, ckks_encoder, scale)
x0_enc = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
beta_ = Plaintext()
ckks_encoder.encode(-1./beta, p_enc.scale(), beta_)
p_enc_beta = Ciphertext()
evaluator.multiply_plain(p_enc, beta_, p_enc_beta)
evaluator.relinearize_inplace(p_enc_beta, relin_keys)
evaluator.rescale_to_next_inplace(p_enc_beta)


def f(x): return x.T.dot(Q).dot(x)


def df(x): return Q.dot(x) + p


def df_enc(c: Ciphertext, q_matrix: Ciphertext, p_vector: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder, gal_keys: GaloisKeys, relin_keys: RelinKeys, n: int):
    q_matrix_dot_c = matrix_multiplication(q_matrix, c, evaluator, encoder, gal_keys, relin_keys, n)
    return rescale_and_mod_switch_y_and_add_x(q_matrix_dot_c, p_vector, evaluator)


# Initialization AGD
y_ = x0
x_ = x0

# Initialization HE
y_enc_ = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
x_enc_ = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)

for t in range(T):
    # Plain
    y = x_ - 1/beta * df(x_)
    x = (1 + gamma) * y - gamma * y_
    x_ = x
    y_ = y

    # Enc
    print("Modulus chain index for y_enc: {}".format(
        context.get_context_data(y_enc_.parms_id()).chain_index()))

    MMultQ_enc_x_enc_ = matrix_multiplication(Q_enc, x_enc_, evaluator,
                              ckks_encoder, gal_keys, relin_keys, d, -1./beta)

    p_enc_beta.set_scale(MMultQ_enc_x_enc_.scale())
    evaluator.mod_switch_to_inplace(p_enc_beta, MMultQ_enc_x_enc_.parms_id())
    evaluator.add_inplace(MMultQ_enc_x_enc_, p_enc_beta)
    y_enc = rescale_and_mod_switch_y_and_add_x(
        MMultQ_enc_x_enc_, x_enc_, evaluator)

    print("Modulus chain index for y_enc: {}".format(
        context.get_context_data(y_enc.parms_id()).chain_index()))

    gamma_1_y_enc = rescale_and_mod_switch_y_and_multiply_x(
        y_enc, 1+gamma, evaluator, ckks_encoder, relin_keys)
    gamma_y_enc_ = rescale_and_mod_switch_y_and_multiply_x(
        y_enc_, -gamma, evaluator, ckks_encoder, relin_keys)

    print("Modulus chain index for y_enc: {}".format(
        context.get_context_data(y_enc.parms_id()).chain_index()))
    x_enc = rescale_and_mod_switch_y_and_add_x(
        gamma_1_y_enc, gamma_y_enc_, evaluator)
    y_enc_ = y_enc
    x_enc_ = x_enc

    # Comparison
    y_dec = decrypt_array(y_enc, decryptor, ckks_encoder, d, d)
    print(
        "Abs difference in x encrypted/decrypted and plain: {}".format(np.abs(y[0][0] - y_dec[0][0])))

    print("==================================================================================")
