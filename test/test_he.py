from agd.matrix.utils import lin_trans, encrypt_array, decrypt_array, \
    lin_trans_enc
from agd.matrix.ours import matrix_multiplication as ours, generate_vk_matrix, \
    generate_wk_matrix
from agd.seal3_7_2.seal import EncryptionParameters, scheme_type, SEALContext, \
    print_parameters, KeyGenerator, Encryptor, CoeffModulus, Evaluator, Decryptor, \
    CKKSEncoder, IntVector, PublicKey, RelinKeys, GaloisKeys
import numpy as np
from agd.matrix.jlks import matrix_multiplication as jkls, phi_permutation_k, \
    psi_permutation_k, sigma_permutation, tau_permutation


def test_lin_trans():
    d = 20
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    b = np.array([np.random.randn(d)])
    assert np.max(np.abs(lin_trans(A, b[0]) - A.dot(b.T).T[0])) < 1E-10


def test_lin_trans_enc():
    d = 20
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    b = np.array([np.random.randn(d)])

    scale = 2.0**40

    parms = EncryptionParameters(scheme_type.ckks)

    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree,
                                                IntVector([60, 40, 40, 60])))

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

    # Problem Inputs HE AGD
    b_enc = encrypt_array(b, encryptor, ckks_encoder, scale)
    ab_enc = lin_trans_enc(A, b_enc, evaluator,
                           ckks_encoder, gal_keys, relin_keys)
    ab = decrypt_array(ab_enc, decryptor, ckks_encoder, d, 1)
    assert np.max(np.abs(ab - A.dot(b.T))) < 1E-4


def test_matrix_multiplication_jkls():
    d = 20
    N = d*d
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    B = np.array([np.random.randn(d*d)]).reshape((d, d))
    a = A.flatten()
    b = B.flatten()
    usigma = sigma_permutation(d, N)
    utau = tau_permutation(d, N)
    a0 = lin_trans(usigma, a)
    b0 = lin_trans(utau, b)
    ak = []
    bk = []
    for k in range(1, d):
        vk = phi_permutation_k(k, d, N)
        wk = psi_permutation_k(k, d, N)
        ak = ak + [lin_trans(vk, a0)]
        bk = bk + [lin_trans(wk, b0)]
    acc = a0 * b0
    for k in range(d-1):
        acc = acc + ak[k] * bk[k]
    assert np.max(np.abs(A.dot(B) - acc.reshape(d, d))) < 1E-10


def test_matrix_multiplication_ours():
    d = 20
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    B = np.array([np.random.randn(d*d)]).reshape((d, d))
    a = A.flatten()
    b = B.flatten()
    acc = 0 * a
    for l in range(d):
        acc = acc + lin_trans(generate_vk_matrix(d, l), a) * \
            lin_trans(generate_wk_matrix(d, l), b)
    assert np.max(np.abs(A.dot(B) - acc.reshape(d, d))) < 1E-10


def test_matrix_multiplication_jkls_enc():
    d = 5
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    B = np.array([np.random.randn(d*d)]).reshape((d, d))

    scale = 2.0**40

    parms = EncryptionParameters(scheme_type.ckks)

    T = 14
    poly_modulus_degree = 2**(T+1)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40]*T + [60])))

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

    # Problem Inputs HE AGD
    a_enc = encrypt_array(A, encryptor, ckks_encoder, scale)
    b_enc = encrypt_array(B, encryptor, ckks_encoder, scale)
    ab_enc = jkls(a_enc, b_enc, evaluator, ckks_encoder,
                  gal_keys, relin_keys, d, 0.5)
    AB = decrypt_array(ab_enc, decryptor, ckks_encoder, d, d)
    assert np.max(np.abs(AB - 0.5*A.dot(B))) < 1E-4


def test_matrix_multiplication_ours_enc():
    d = 5
    A = np.array([np.random.randn(d*d)]).reshape((d, d))
    B = np.array([np.random.randn(d*d)]).reshape((d, d))

    scale = 2.0**40

    parms = EncryptionParameters(scheme_type.ckks)

    T = 14
    poly_modulus_degree = 2**(T+1)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40]*T + [60])))

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

    # Problem Inputs HE AGD
    ct_a = encrypt_array(A, encryptor, ckks_encoder, scale)
    ct_b = encrypt_array(B, encryptor, ckks_encoder, scale)
    ab_enc = ours(ct_a, ct_b, evaluator, ckks_encoder,
                  gal_keys, relin_keys, d, 0.5)
    AB = decrypt_array(ab_enc, decryptor, ckks_encoder, d, d)
    assert np.max(np.abs(AB - 0.5*A.dot(B))) < 1E-4
