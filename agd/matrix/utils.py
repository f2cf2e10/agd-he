import numpy as np
from agd.seal.seal import Encryptor, Evaluator, Decryptor, \
    CKKSEncoder, Evaluator, Ciphertext, CKKSEncoder, \
    CiphertextVector, GaloisKeys, RelinKeys, DoubleVector, \
    Plaintext, BatchEncoder, Int64Vector
from typing import List, Union


def squarify(matrix: np.ndarray, val: float, n: int = None):
    a, b = matrix.shape
    if n is None:
        if a > b:
            padding = ((0, 0), (0, a - b))
        else:
            padding = ((0, b - a), (0, 0))
    else:
        padding = ((0, max(n - a, 0)), (0, max(n - b, 0)))
    return np.pad(matrix, padding, mode='constant', constant_values=val)


def diag_repr(u_matrix: np.ndarray, i: int, pad_zero: bool = True) -> np.ndarray:
    N, M = u_matrix.shape
    if (N != M):
        raise Exception("diag_repr only works for square matrices")
    if pad_zero:
        diag = np.diagonal(u_matrix, i)
        if i > 0:
            return np.concatenate([diag, np.zeros(N - len(diag))])
        else:
            return np.concatenate([np.zeros(N - len(diag)), diag])
    else:
        result = np.concatenate(
            [np.diag(u_matrix, i), np.diag(u_matrix, -N + i)])
        if len(result) != N:
            raise Exception("diag_repr index is out of bounds")
        return result


def lin_trans(u_matrix: np.ndarray, c: np.ndarray, d: int) -> np.ndarray:
    N, M = u_matrix.shape
    if (N != M):
        raise Exception("lin_trans only works for square matrices")
    if (N > 2 * d):
        acc = diag_repr(u_matrix, 0) * c
        for l in range(1, d):
            ul_vec = diag_repr(u_matrix, l)
            c_l = np.roll(c, -l)
            ul_vec_ = diag_repr(u_matrix, -l)
            c_l_ = np.roll(c, l)
            acc = acc + ul_vec * c_l + ul_vec_ * c_l_
        return acc
    elif (d == N):
        acc = c * 0
        for l in range(d):
            ul_vec = diag_repr(u_matrix, l, False)
            c_l = np.roll(c, -l)
            acc = acc + ul_vec * c_l
        return acc
    else:
        raise Exception(
            "lin_trans only works with matrices with dimension N>2d or N=d")


def lin_trans_enc(u_matrix: np.ndarray, ct: Ciphertext, evaluator: Evaluator, encoder: Union[BatchEncoder, CKKSEncoder],
                  gal_keys: GaloisKeys, relin_keys: RelinKeys = None, scale: float = None) -> Ciphertext:
    """
    From page 5 at https://eprint.iacr.org/2018/1041.pdf

    NOTE: We denote homomorphic multiplication and constant multiplication by Mult and CMult

    In general, an arbitrary linear transformation L: Rn -> Rn over plaintext vectors can be
    represented as L:m -> U·m for some matrix U \in Rn×n. We can express the matrix-vector
    multiplication by combining rotation and constant multiplication operations. Specifically,
    for 0<=l< n, we define the l-th diagonal vector of U by

    ul= (U_{0,l} , U_{1,l+1}, ..., U_{n−l−1,n−1} ,U_{n−l,0}, ..., U_{n−1,l−1} ) \in Rn.
evaluator.multiply_plain(ct
    ul = (U__{k, (l+k) mod n}), k=0,...,n-1

    Then we have

    U \dot m= \sum_{0<=l<n} (u_l . ρ(m;l))

    where . denotes the component-wise multiplication between vectors.
    Given a matrix U \in Rn×n and an encryptionct of the vector m, the following algorithm
    describes how to compute a ciphertext of the desired vector U \dot m.

    procedure LinTrans(ct;U)
    1: ct_ <- CMult(ct;u0)
    2: for l = 1 to n−1 do
    3:   ct_ <- Add(ct_,CMult(Rot(ct;l); ul))
    4: end for
    5: return ct_
    """
    M, N = u_matrix.shape
    if (N != M):
        raise Exception("lin_trans only works for square matrices")
    nmax = encoder.slot_count()
    if N * M > nmax / 2:
        raise Exception(
            "Matrix dimenson is higher than the one suported by the encoder")
    if scale is None:
        scale = ct.scale()
    parms_id = ct.parms_id()
    acc = CiphertextVector()

    def get_diag_rotate_vec_and_multiply(matrix: np.ndarray, diag: int, array: Ciphertext, rotate: int):
        matrix_diag = diag_repr(matrix, diag)
        if sum(matrix_diag) == 0:
            return None
        vec_diag_enc = Plaintext()
        encode(matrix_diag, encoder, vec_diag_enc, scale)
        vec_rot = Ciphertext()
        if isinstance(encoder, CKKSEncoder):
            evaluator.mod_switch_to_inplace(vec_diag_enc, parms_id)
            evaluator.rotate_vector(array, rotate, gal_keys, vec_rot)
            evaluator.multiply_plain_inplace(vec_rot, vec_diag_enc)
        if isinstance(encoder, BatchEncoder):
            evaluator.rotate_rows(array, rotate, gal_keys, vec_rot)
            evaluator.multiply_plain_inplace(vec_rot, vec_diag_enc)
        return vec_rot

    val = get_diag_rotate_vec_and_multiply(u_matrix, 0, ct, 0)
    if val is not None:
        acc.append(val)
    for l in range(1, N):
        val = get_diag_rotate_vec_and_multiply(u_matrix, l, ct, l)
        if val is not None:
            acc.append(val)
        val = get_diag_rotate_vec_and_multiply(u_matrix, -l, ct, -l)
        if val is not None:
            acc.append(val)

    out = Ciphertext()
    evaluator.add_many(acc, out)
    if relin_keys is not None:
        evaluator.relinearize_inplace(out, relin_keys)
        if isinstance(encoder, CKKSEncoder):
            evaluator.rescale_to_next_inplace(out)
    #if isinstance(encoder, BatchEncoder):
    #    evaluator.mod_switch_to_next_inplace(out)

    return out


def ca_x_cb(ct_a: Ciphertext, ct_b: Ciphertext, evaluator: Evaluator, relin_keys: RelinKeys = None,
            scale: float = None):
    """
    Element-wise product of cipherthextA and ciphertextB
    """
    auto_rescale_and_mod_switch([ct_a, ct_b], evaluator, scale)
    ct_c = Ciphertext()
    evaluator.multiply(ct_a, ct_b, ct_c)
    if relin_keys is not None:
        evaluator.relinearize_inplace(ct_c, relin_keys)
        evaluator.rescale_to_next_inplace(ct_c)
    # if scale is not None:
    #    ct_c.set_scale(scale)
    return ct_c


def encode(matrix: np.ndarray, encoder: Union[BatchEncoder, CKKSEncoder], plain: Plaintext, scale: float):
    if isinstance(encoder, CKKSEncoder):
        encoder.encode(DoubleVector(matrix.flatten().tolist()), scale, plain)
    if isinstance(encoder, BatchEncoder):
        encoder.encode(Int64Vector((matrix / scale).astype(np.int64).flatten().tolist()), plain)


def decode(encoder: Union[BatchEncoder, CKKSEncoder], plain: Plaintext, scale: float = None) -> np.ndarray:
    if isinstance(encoder, CKKSEncoder):
        vec = DoubleVector()
        encoder.decode(plain, vec)
        return np.array(vec)
    if isinstance(encoder, BatchEncoder):
        vec = Int64Vector()
        encoder.decode(plain, vec)
        return np.array(vec) * scale


def encrypt_array(matrix: np.ndarray, encryptor: Encryptor, encoder: Union[BatchEncoder, CKKSEncoder],
                  scale: float) -> Ciphertext:
    plain = Plaintext()
    cmatrix = Ciphertext()
    encode(matrix, encoder, plain, scale)
    encryptor.encrypt(plain, cmatrix)
    return cmatrix


def decrypt_array(cipher: Ciphertext, decryptor: Decryptor, encoder: Union[BatchEncoder, CKKSEncoder], m: int, n: int,
                  scale: float = None) -> np.ndarray:
    plain = Plaintext()
    decryptor.decrypt(cipher, plain)
    vec = decode(encoder, plain, scale)
    matrix = np.array(vec[0:(m * n)]).reshape(m, n)
    return matrix


def rescale_and_mod_switch(x: Ciphertext, new_scale: float, new_parms: List[int], evaluator: Evaluator):
    x.set_scale(new_scale)
    evaluator.mod_switch_to_inplace(x, new_parms)


def auto_rescale_and_mod_switch(x: List[Ciphertext], evaluator: Evaluator, scale: float = None):
    scales = [i.scale() for i in x]
    parms = [i.parms_id() for i in x]
    new_scale = np.max(scales) if scale is None else scale
    i_scale = np.argmax(scales)
    new_parms = parms[i_scale]
    for xi in x:
        rescale_and_mod_switch(xi, new_scale, new_parms, evaluator)


def rescale_and_mod_switch_y_and_add_x(x: Ciphertext, y: Ciphertext, evaluator: Evaluator) -> Ciphertext:
    # Although the scales of all three terms are approximately 2^40, their exact v
    # alues are different, hence they cannot be added together.
    # There are many ways to fix this problem. Since the prime numbers are really close
    # we can simply "lie" to Microsoft SEAL and set the scales to be the
    # same.
    # print("The exact scales of all three terms are different:")
    # print("    + Exact scale in x: {0:0.10f}".format(x.scale()))
    # print("    + Exact scale in y: {0:0.10f}".format(y.scale()))
    # print("    + scale ratio: {0:0.10f}".format(1-x.scale()/y.scale()))
    y.set_scale(x.scale())
    # We still have a problem with mismatching encryption parameters. This is easy
    # to fix by using traditional modulus switching (no rescaling). CKKS supports
    # modulus switching , allowing us to switch away parts of the coefficient modulus
    # when it is simply not needed.
    z = Ciphertext()
    evaluator.mod_switch_to_inplace(y, x.parms_id())
    evaluator.add(x, y, z)
    return z


def rescale_and_mod_switch_y_and_multiply_x(x: Ciphertext, y: float, evaluator: Evaluator, encoder: CKKSEncoder,
                                            relin_keys: RelinKeys, scale: float = None):
    y_ = Plaintext()
    _scale = scale if scale is not None else x.scale()
    encode(y, encoder, y_, _scale)
    evaluator.mod_switch_to_inplace(y_, x.parms_id())
    z = Ciphertext()
    evaluator.multiply_plain(x, y_, z)
    evaluator.relinearize_inplace(z, relin_keys)
    evaluator.rescale_to_next_inplace(z)
    return z
