# Inspired by "Secure Outsourced Matrix Computationand Application to Neural Networks?"
# Link: https://eprint.iacr.org/2018/1041.pdf
from seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, \
    Encryptor, CoeffModulus, Evaluator, Decryptor, \
    CKKSEncoder, IntVector
import numpy as np
from agd_he import *

# Problem Inputs

Q = np.array([[2, .5], [.5, 1]])
p = np.array([[1.0], [1.0]])
x = np.array([np.random.randn(2)]).T
y = x
T = 14

# Problem calculations
eigenvals = np.linalg.eigvals(Q)
beta = np.max(eigenvals)
alpha = np.min(eigenvals)
kappa = beta/alpha
gamma = (kappa**0.5-1)/(kappa**0.5+1)

# HE inputs
n = 2
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

Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
p_enc = encrypt_array(p, encryptor, ckks_encoder, scale)
x_enc = encrypt_array(squarify(x, 0.0, n), encryptor, ckks_encoder, scale)
y_enc = encrypt_array(squarify(x, 0.0, n), encryptor, ckks_encoder, scale)
beta_ = Plaintext()
ckks_encoder.encode(-1./beta, p_enc.scale(), beta_)
p_enc_beta = Ciphertext()
evaluator.multiply_plain(p_enc, beta_, p_enc_beta)
evaluator.relinearize_inplace(p_enc_beta, relin_keys)
evaluator.rescale_to_next_inplace(p_enc_beta)

def f(x): return x.T.dot(Q).dot(x)

def df(x): return Q.dot(x) + p

def df_enc(c: Ciphertext, Q: Ciphertext, p: Ciphertext, evaluator: Evaluator, encoder:CKKSEncoder, gal_keys:GaloisKeys, relin_keys:RelinKeys, n:int):
    Qdotc = cA_dot_cB(Q, c, evaluator, encoder, gal_keys, relin_keys, n)
    return rescale_and_mod_switch_y_and_add_x(Qdotc, p, evaluator)

def df_enc_scaled(c: Ciphertext, Q: Ciphertext, p_enc_beta: Ciphertext, evaluator: Evaluator, encoder:CKKSEncoder, gal_keys:GaloisKeys, relin_keys:RelinKeys, n:int, scale:float):
    Qdotc = cA_dot_cB(Q, c, evaluator, encoder, gal_keys, relin_keys, n, scale)
    p_enc_beta.set_scale(Qdotc.scale())
    evaluator.mod_switch_to_inplace(p_enc_beta, Qdotc.parms_id())
    evaluator.add_inplace(Qdotc, p_enc_beta )
    return Qdotc

for t in range(T):
    #Plain
    y_ = y
    y = x - 1/beta * df(x)
    x_ = x
    x = (1 + gamma) * y - gamma * y_
    
    #Enc
    print("Modulus chain index for x_enc: {}".format(context.get_context_data(x_enc.parms_id()).chain_index()))
    print("Modulus chain index for y_enc: {}".format(context.get_context_data(y_enc.parms_id()).chain_index()))
    y_enc_ = y_enc 
    ############
    #df_x_enc = df_enc(x_enc, Q_enc, p_enc, evaluator, ckks_encoder, gal_keys, relin_keys, n)
    #df_x_enc_scaled_by_beta = rescale_and_mod_switch_y_and_multiply_x(df_x_enc, -1./beta, evaluator, ckks_encoder, relin_keys)
    #df_x_enc_scaled_by_beta_directly = df_enc_scaled(x_enc, Q_enc, p_enc, evaluator, ckks_encoder, gal_keys, relin_keys, n, -1./beta)
    #a = decrypt_array(df_x_enc_scaled_by_beta, decryptor, ckks_encoder, n, n)
    #b = decrypt_array(df_x_enc_scaled_by_beta_directly, decryptor, ckks_encoder, n, n)
    ############
    df_x_enc_scaled_by_beta = df_enc_scaled(x_enc, Q_enc, p_enc_beta, evaluator, ckks_encoder, gal_keys, relin_keys, n, -1./beta)
    ############
    print("Modulus chain index for x_enc: {}".format(context.get_context_data(x_enc.parms_id()).chain_index()))
    print("Modulus chain index for y_enc: {}".format(context.get_context_data(y_enc.parms_id()).chain_index()))
    
    print("Modulus chain index for x_enc: {}".format(context.get_context_data(x_enc.parms_id()).chain_index()))
    print("Modulus chain index for y_enc: {}".format(context.get_context_data(y_enc.parms_id()).chain_index()))
    
    print("Modulus chain index for df_x_enc_scaled_by_beta: {}".format(context.get_context_data(df_x_enc_scaled_by_beta.parms_id()).chain_index()))
    y_enc = rescale_and_mod_switch_y_and_add_x(df_x_enc_scaled_by_beta, x_enc, evaluator)
    x_enc_ = x_enc

    print("Modulus chain index for x_enc: {}".format(context.get_context_data(x_enc.parms_id()).chain_index()))
    print("Modulus chain index for y_enc: {}".format(context.get_context_data(y_enc.parms_id()).chain_index()))
    gamma_1 = Plaintext()
    gamma_ = Plaintext()

    gamma_1_y_enc  = rescale_and_mod_switch_y_and_multiply_x(y_enc, 1+gamma, evaluator, ckks_encoder, relin_keys)
    gamma_y_enc_ =  rescale_and_mod_switch_y_and_multiply_x(y_enc_, -gamma, evaluator, ckks_encoder, relin_keys)

    print("Modulus chain index for x_enc: {}".format(context.get_context_data(x_enc.parms_id()).chain_index()))
    print("Modulus chain index for y_enc: {}".format(context.get_context_data(y_enc.parms_id()).chain_index()))
    x_enc = rescale_and_mod_switch_y_and_add_x(gamma_1_y_enc, gamma_y_enc_, evaluator)

    #rescale and modswitch
    auto_rescale_and_mod_switch([y_enc, y_enc_, x_enc, x_enc_], evaluator)
    #scales = [y_enc.scale(), y_enc_.scale(), x_enc.scale(), x_enc_.scale()]
    #parms = [y_enc.parms_id(), y_enc_.parms_id(), x_enc.parms_id(), x_enc_.parms_id()]
    #new_scale = np.max(scales)
    #i_scale = np.argmax(scales)
    #new_parms = parms[i_scale]
    #rescale_and_mod_switch(y_enc, new_scale, new_parms, evaluator) 
    #rescale_and_mod_switch(y_enc_, new_scale, new_parms, evaluator) 
    #rescale_and_mod_switch(x_enc, new_scale, new_parms, evaluator) 
    #rescale_and_mod_switch(x_enc_, new_scale, new_parms, evaluator) 

    #Comparison
    x_dec = decrypt_array(x_enc, decryptor, ckks_encoder, n, n)
    print("Norm2 of difference in x encrypted/decrypted and plain: {}".format(sum((x - x_dec[:, [0]])**2)[0]**0.5))

    print("Modulus chain index for x_enc: {}".format(context.get_context_data(x_enc.parms_id()).chain_index()))
    print("Modulus chain index for y_enc: {}".format(context.get_context_data(y_enc.parms_id()).chain_index()))
    print("Modulus chain index for x_enc_: {}".format(context.get_context_data(x_enc_.parms_id()).chain_index()))
    print("Modulus chain index for y_enc_: {}".format(context.get_context_data(y_enc_.parms_id()).chain_index()))
    print("==================================================================================")


tmp_enc = df_enc(x_enc, Q_enc, p_enc, evaluator, ckks_encoder, gal_keys, relin_keys, n)

tmp = decrypt_array(tmp_enc, decryptor, ckks_encoder, n, n)
val = df(x)
print()