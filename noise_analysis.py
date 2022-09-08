from agd.seal.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, \
    Encryptor, CoeffModulus, Evaluator, Decryptor, \
    CKKSEncoder, IntVector, Plaintext, Ciphertext, GaloisKeys, \
    RelinKeys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agd.matrix.jlks import matrix_multiplication as jlks
from agd.matrix.ours import matrix_multiplication as ours
from agd.matrix.utils import encrypt_array, lin_trans, lin_trans_enc, squarify, decrypt_array, \
    rescale_and_mod_switch, rescale_and_mod_switch_y_and_add_x, \
    rescale_and_mod_switch_y_and_multiply_x

scale = 2.0**40

parms = EncryptionParameters(scheme_type.ckks)
N = 16
poly_modulus_degree = 32768
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(
    poly_modulus_degree, IntVector([60] + [40]*N + [60])))

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

x0 = np.array([np.ones([10])])
x = encrypt_array(squarify(x0, 0.0, 10), encryptor, ckks_encoder, scale)
y = encrypt_array(squarify(x0.T, 0.0, 10), encryptor, ckks_encoder, scale)

for i in range(int(N/2)):
    x = lin_trans_enc(np.eye(10), x, evaluator, ckks_encoder, gal_keys, relin_keys) 
    y = lin_trans_enc(np.eye(10), y, evaluator, ckks_encoder, gal_keys, relin_keys) 
    z = jlks(x, y, evaluator, ckks_encoder, gal_keys, relin_keys, 10, 1.)
    z_dec = decrypt_array(z, decryptor, ckks_encoder, 10, 10) 
    print(np.abs(x0.dot(x0.T)[0][0] - z_dec[0, 0]))

