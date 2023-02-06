import numpy as np
from agd.seal.seal import EncryptionParameters, scheme_type, SEALContext, \
    print_parameters, KeyGenerator, Encryptor, CoeffModulus, Evaluator, Decryptor, \
    CKKSEncoder, BatchEncoder, IntVector, PublicKey, RelinKeys, GaloisKeys, PlainModulus
from agd.matrix.utils import encrypt_array, decrypt_array, lin_trans_enc, squarify
from agd.matrix.jlks import matrix_multiplication as jkls
from agd.matrix.ours import matrix_multiplication as ours


def test_lin_trans_enc_ckks():
    d = 20
    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    b = np.array([np.random.randn(d)])

    scale = 2.0 ** 40

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


def test_lin_trans_enc_bfv():
    d = 2

    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    b = np.array([np.random.randn(d)])
    scale = 1E-8

    parms = EncryptionParameters(scheme_type.bfv)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    plain_modulus = PlainModulus.Batching(poly_modulus_degree, 60)
    parms.set_plain_modulus(plain_modulus)

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

    batch_encoder = BatchEncoder(context)

    # Problem Inputs HE AGD
    b_enc = encrypt_array(b, encryptor, batch_encoder, scale)
    ab_enc = lin_trans_enc(A, b_enc, evaluator, batch_encoder, gal_keys, relin_keys, scale)
    ab = decrypt_array(ab_enc, decryptor, batch_encoder, d, 1, scale)
    assert np.max(np.abs(ab - A.dot(b.T))) < 1E-4


def test_lin_trans_enc_bgv():
    d = 2

    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    b = np.array([np.random.randn(d)])
    scale = 1E-8

    parms = EncryptionParameters(scheme_type.bgv)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    plain_modulus = PlainModulus.Batching(poly_modulus_degree, 60)
    parms.set_plain_modulus(plain_modulus)

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

    batch_encoder = BatchEncoder(context)

    # Problem Inputs HE AGD
    b_enc = encrypt_array(b, encryptor, batch_encoder, scale)
    ab_enc = lin_trans_enc(A, b_enc, evaluator, batch_encoder, gal_keys, relin_keys, scale)
    ab = decrypt_array(ab_enc, decryptor, batch_encoder, d, 1, scale)
    assert np.max(np.abs(ab - A.dot(b.T))) < 1E-4


def test_matrix_multiplication_jkls_enc_ckks():
    d = 5
    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    B = np.array([np.random.randn(d * d)]).reshape((d, d))

    scale = 2.0 ** 40

    parms = EncryptionParameters(scheme_type.ckks)

    T = 14
    poly_modulus_degree = 2 ** (T + 1)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40] * T + [60])))

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
    assert np.max(np.abs(AB - 0.5 * A.dot(B))) < 1E-4


def test_matrix_multiplication_ours_enc_ckks():
    d = 5
    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    B = np.array([np.random.randn(d * d)]).reshape((d, d))

    scale = 2.0 ** 40

    parms = EncryptionParameters(scheme_type.ckks)

    T = 14
    poly_modulus_degree = 2 ** (T + 1)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40] * T + [60])))

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
                  gal_keys, relin_keys, d, 0.5, scale)
    AB = decrypt_array(ab_enc, decryptor, ckks_encoder, d, d)
    assert np.max(np.abs(AB - 0.5 * A.dot(B))) < 1E-4


def test_matrix_multiplication_ours_enc_bfv():
    d = 5
    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    B = np.array([np.random.randn(d * d)]).reshape((d, d))

    scale = 1E-8

    parms = EncryptionParameters(scheme_type.bfv)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    plain_modulus = PlainModulus.Batching(poly_modulus_degree, 60)
    parms.set_plain_modulus(plain_modulus)

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

    batch_encoder = BatchEncoder(context)

    # Problem Inputs HE AGD
    a_enc = encrypt_array(A, encryptor, batch_encoder, scale)
    b_enc = encrypt_array(B, encryptor, batch_encoder, scale)
    ab_enc = ours(a_enc, b_enc, evaluator, batch_encoder,
                  gal_keys, relin_keys, d, 0.5)
    AB = decrypt_array(ab_enc, decryptor, batch_encoder, d, d)
    assert np.max(np.abs(AB - 0.5 * A.dot(B))) < 1E-4


def test_matrix_vector_multiplication_ours_enc_ckks():
    d = 5
    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    b = squarify(np.array([np.random.randn(d)]).reshape((d, 1)), 0)

    scale = 2.0 ** 40

    parms = EncryptionParameters(scheme_type.ckks)

    T = 14
    poly_modulus_degree = 2 ** (T + 1)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40] * T + [60])))

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
    ct_b = encrypt_array(squarify(b, 0), encryptor, ckks_encoder, scale)
    ab_enc = ours(ct_a, ct_b, evaluator, ckks_encoder,
                  gal_keys, relin_keys, d, 0.5)
    AB = decrypt_array(ab_enc, decryptor, ckks_encoder, d, d)
    assert np.max(np.abs(AB - 0.5 * A.dot(b))) < 1E-4


def test_matrix_multiplication_ours_enc_twice_ckks():
    d = 5
    A = np.array([np.random.randn(d * d)]).reshape((d, d))
    B = np.array([np.random.randn(d * d)]).reshape((d, d))

    scale = 2.0 ** 40

    parms = EncryptionParameters(scheme_type.ckks)

    T = 14
    poly_modulus_degree = 2 ** (T + 1)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40] * T + [60])))

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
                  gal_keys, relin_keys, d, 0.5, scale)
    abb_enc = ours(ab_enc, ct_b, evaluator, ckks_encoder,
                   gal_keys, relin_keys, d, 1., scale)

    ABB = decrypt_array(abb_enc, decryptor, ckks_encoder, d, d)
    assert np.max(np.abs(ABB - 0.5 * A.dot(B).dot(B))) < 1E-4
