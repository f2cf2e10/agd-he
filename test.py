from agd.seal.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, Encryptor, CoeffModulus, \
    Evaluator, Decryptor, CKKSEncoder, IntVector, Plaintext, Ciphertext, \
    GaloisKeys, RelinKeys, PublicKey, sec_level_type, DoubleVector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agd.matrix.ours import matrix_multiplication as ours
from agd.matrix.utils import encrypt_array, squarify, decrypt_array, \
    rescale_and_mod_switch, rescale_and_mod_switch_y_and_add_x, \
    rescale_and_mod_switch_y_and_multiply_x

T = 5
R = 3

d = 2
Q = np.random.randn(d*d).reshape(d,d)
p = np.random.rand(d).reshape(d,1)
x0 = np.ones(d).reshape(d,1)

scale = 2.0**40

parms = EncryptionParameters(scheme_type.ckks)

poly_modulus_degree = 32768
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, IntVector([60] + [40]*15 + [60])))

context = SEALContext(parms)
print_parameters(context)

keygen = KeyGenerator(context)
secret_key = keygen.secret_key()
    
public_key = PublicKey()
keygen.create_public_key(public_key)
   
relin_keys = RelinKeys()
keygen.create_relin_keys(relin_keys)

gal_keys = GaloisKeys() 
keygen.create_galois_keys(gal_keys)

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)
ckks_encoder = CKKSEncoder(context)

Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
p_enc = encrypt_array(squarify(p, 0.0, d), encryptor, ckks_encoder, scale)
x_enc0 = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
I_enc = encrypt_array(squarify(np.eye(d), 0.0, d), 
        encryptor, ckks_encoder, scale)

Q_dec = decrypt_array(Q_enc, decryptor, ckks_encoder, d, d)
p_dec = decrypt_array(p_enc, decryptor, ckks_encoder, d, d)
x0_dec = decrypt_array(x_enc0, decryptor, ckks_encoder, d, d)

def f(x):
    return Q.dot(x) + p

def f_enc(x_enc: Ciphertext) -> Ciphertext: 
    Qx_enc = ours(Q_enc, x_enc, evaluator, ckks_encoder, gal_keys, None, d, 1)
    Qx_enc1 = ours(Q_enc, x_enc, evaluator, ckks_encoder, gal_keys, relin_keys, d, 1)
    test = ours(I_enc, p_enc, evaluator, ckks_encoder, gal_keys, None, d, 1)
    test1 = ours(I_enc, p_enc, evaluator, ckks_encoder, gal_keys, relin_keys, d, 1)
    Qx_enc.scale()
    Qx_enc.parms_id()
    for i in range(3):
        evaluator.relinearize_inplace(Qx_enc, relin_keys)
        evaluator.rescale_to_next_inplace(Qx_enc)
        Qx_dec += [decrypt_array(Qx_enc, decryptor, ckks_encoder, d, d)]
        Qx_enc.scale()
        Qx_enc.parms_id()

    Qx_enc.set_scale(p_enc.scale())
    Qx_dec += [decrypt_array(Qx_enc, decryptor, ckks_encoder, d, d)]
    evaluator.mod_switch_to_inplace(p_enc, Qx_enc.parms_id())
    Qx_dec += [decrypt_array(Qx_enc, decryptor, ckks_encoder, d, d)]
    z_enc = Ciphertext()
    evaluator.add(p_enc, Qx_enc, z_enc)
    z_dec = decrypt_array(z_enc, decryptor, ckks_encoder, d, d)
    return z_enc

def f_enc(x_enc: Ciphertext) -> Ciphertext: 
    Qx_enc = ours(Q_enc, x_enc, evaluator, ckks_encoder, gal_keys, relin_keys, d, 1)
    Qx_enc.set_scale(p_enc.scale())
    evaluator.mod_switch_to_inplace(p_enc, Qx_enc.parms_id())
    z_enc = Ciphertext()
    evaluator.add(p_enc, Qx_enc, z_enc)
    return z_enc

def g_enc(x_enc: Ciphertext) -> Ciphertext: 
    Qx_enc = ours(Q_enc, x_enc, evaluator, ckks_encoder, gal_keys, relin_keys, d, 1)
    p_enc_1 = ours(I_enc, p_enc, evaluator, ckks_encoder, gal_keys, relin_keys, d, 1)
    z_enc = Ciphertext()
    evaluator.add(p_enc_1, Qx_enc, z_enc)
    return z_enc, p_enc_1

x_enc1 = f_enc(x_enc0)
x_enc1_, p_enc1 = g_enc(x_enc0)
x_dec1 = decrypt_array(x_enc1, decryptor, ckks_encoder, d, d)
x_dec1_ = decrypt_array(x_enc1, decryptor, ckks_encoder, d, d)
x1 = f(x0)

x_enc2 = f_enc(x_enc1)
x_dec2 = decrypt_array(x_enc2, decryptor, ckks_encoder, d, d)
x2 = f(x1)

x_enc3 = f_enc(x_enc2)
x_dec3 = decrypt_array(x_enc3, decryptor, ckks_encoder, d, d)
x3 = f(x2)

x_enc4 = f_enc(x_enc3)
x_dec4 = decrypt_array(x_enc4, decryptor, ckks_encoder, d, d)
x4 = f(x3)
