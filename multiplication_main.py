import numpy as np
from agd.matrix.utils import encrypt_array, decrypt_array
from agd.seal.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, Encryptor, CoeffModulus, \
    Evaluator, Decryptor, CKKSEncoder, IntVector, Plaintext, Ciphertext, \
    GaloisKeys, RelinKeys, PublicKey, sec_level_type, DoubleVector

# Steps
bits = [27, 54, 109, 218, 438, 881]
poly_degree = [2**i for i in range(10, 16)]
T = 5
R = 3
scale = 2.0**40

parms = EncryptionParameters(scheme_type.ckks)

poly_modulus_degree = poly_degree[bits.index(
    next((x for x in bits if x > (120+T*R*40)), None))]
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(
poly_modulus_degree, IntVector([60] + [40]*T*R + [60])))

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

ones = np.ones(4) 
x_enc = encrypt_array(ones, encryptor, ckks_encoder, scale)
y_enc = encrypt_array(ones, encryptor, ckks_encoder, scale)

xs = [None] * (T*R+1)
xs[0] = x_enc
for t in range(1, T*R+1):
    x = Ciphertext()
    evaluator.multiply(xs[t-1], y_enc, x)
    evaluator.relinearize_inplace(x, relin_keys)
    evaluator.rescale_to_next_inplace(x)
    y_enc.set_scale(x.scale())
    #y_enc.set_scale(2**40)
    evaluator.mod_switch_to_inplace(y_enc, x.parms_id())
    xs[t] = x 

xs_dec = [decrypt_array(x, decryptor, ckks_encoder, 1, 4) for x in xs]
plot([np.sum((ones-x)**2)**.5 for x in xs_dec])