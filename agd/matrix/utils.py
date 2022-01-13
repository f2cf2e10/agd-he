import numpy as np
from agd.seal import Encryptor, Evaluator, Decryptor, \
    CKKSEncoder, Evaluator, Ciphertext, CKKSEncoder, \
    CiphertextVector, GaloisKeys, RelinKeys, DoubleVector, \
    Plaintext
from typing import List


def squarify(matrix: np.ndarray, val: float, n: int = None):
    a, b = matrix.shape
    if n is None:
        if a > b:
            padding = ((0, 0), (0, a-b))
        else:
            padding = ((0, b-a), (0, 0))
    else:
        padding = ((0, max(n-a, 0)), (0, max(n-b, 0)))
    return np.pad(matrix, padding, mode='constant', constant_values=val)


def diag_repr(u_matrix: np.ndarray, i: int) -> np.ndarray:
    N, M = u_matrix.shape
    if (N != M):
        raise Exception("diag_repr only works for square matrices")
    else:
        result = np.concatenate([np.diag(u_matrix, i), np.diag(u_matrix, -N+i)])
        if len(result) != N:
            raise Exception("diag_repr index is out of bounds")
        return result


def lin_trans(u_matrix: np.ndarray, c: np.ndarray) -> np.ndarray:
    N = len(c)
    acc = c * 0
    for l in range(N):
        ul_vec = diag_repr(u_matrix, l)
        c_l = np.roll(c, -l)
        acc = acc + ul_vec * c_l
    return acc


def lin_trans_enc(u_matrix: np.ndarray, ct: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
                  gal_keys: GaloisKeys, relin_keys: RelinKeys = None) -> Ciphertext:
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

    ############# Algorithm #############
    procedure LinTrans(ct;U)
    1: ct_ <- CMult(ct;u0)
    2: for l = 1 to n−1 do
    3:   ct_ <- Add(ct_,CMult(Rot(ct;l); ul))
    4: end for
    5: return ct_
    #####################################
    """
    M, N = u_matrix.shape
    if (N != M):
        raise Exception("lin_trans only works for square matrices")
    scale = ct.scale()
    parms_id = ct.parms_id()
    acc = CiphertextVector()
    for l in range(N):
        ul_vec_np = diag_repr(u_matrix, l)
        if np.abs(ul_vec_np).sum() == 0:
            continue
        ul_vec = DoubleVector(ul_vec_np.tolist())
        ul = Plaintext()
        encoder.encode(ul_vec, scale, ul)
        # Encrypted addition and subtraction require that the scales of the inputs are
        # the same, and also that the encryption parameters (parms_id) match. If there
        # is a mismatch, Evaluator will throw an exception.
        # Here we make sure to encode ul with appropriate encryption parameters (parms_id).
        evaluator.mod_switch_to_inplace(ul, parms_id)
        ct_l = Ciphertext()
        evaluator.rotate_vector(ct, l, gal_keys, ct_l)
        evaluator.multiply_plain_inplace(ct_l, ul)
        acc.append(ct_l)
    out = Ciphertext()
    evaluator.add_many(acc, out)
    if relin_keys is not None:
        evaluator.relinearize_inplace(out, relin_keys)
        evaluator.rescale_to_next_inplace(out)
    return out


def ca_x_cb(ct_a: Ciphertext, ct_b: Ciphertext, evaluator: Evaluator, relin_keys: RelinKeys = None):
    """
    Element-wise product of cipherthextA and ciphertextB
    """
    auto_rescale_and_mod_switch([ct_a, ct_b], evaluator)
    ct_c = Ciphertext()
    evaluator.multiply(ct_a, ct_b, ct_c)
    if relin_keys is not None:
        evaluator.relinearize_inplace(ct_c, relin_keys)
        evaluator.rescale_to_next_inplace(ct_c)
    return ct_c 


def encrypt_array(matrix: np.ndarray, encryptor: Encryptor, encoder: CKKSEncoder, scale: float) -> Ciphertext:
    M, N = matrix.shape
    max_n = encoder.slot_count()
    plain = Plaintext()
    cmatrix = Ciphertext()
    encoder.encode(DoubleVector((matrix.flatten().tolist())
                   * int(max_n/N/M)), scale, plain)
    encryptor.encrypt(plain, cmatrix)
    return cmatrix


def decrypt_array(cipher: Ciphertext, decryptor: Decryptor, encoder: CKKSEncoder, m: int, n: int) -> np.ndarray:
    plain = Plaintext()
    decryptor.decrypt(cipher, plain)
    vec = DoubleVector()
    encoder.decode(plain, vec)
    matrix = np.array(vec[0:(m*n)]).reshape(m, n)
    return matrix


def rescale_and_mod_switch(x: Ciphertext, new_scale: float, new_parms: List[int], evaluator: Evaluator):
    x.set_scale(new_scale)
    evaluator.mod_switch_to_inplace(x, new_parms)


def auto_rescale_and_mod_switch(x: List[Ciphertext], evaluator: Evaluator):
    scales = [i.scale() for i in x]
    parms = [i.parms_id() for i in x]
    new_scale = np.max(scales)
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
    #print("The exact scales of all three terms are different:")
    #print("    + Exact scale in x: {0:0.10f}".format(x.scale()))
    #print("    + Exact scale in y: {0:0.10f}".format(y.scale()))
    #print("    + scale ratio: {0:0.10f}".format(1-x.scale()/y.scale()))
    y.set_scale(x.scale())
    # We still have a problem with mismatching encryption parameters. This is easy
    # to fix by using traditional modulus switching (no rescaling). CKKS supports
    # modulus switching , allowing us to switch away parts of the coefficient modulus
    # when it is simply not needed.
    z = Ciphertext()
    evaluator.mod_switch_to_inplace(y, x.parms_id())
    evaluator.add(x, y, z)
    return z


def rescale_and_mod_switch_y_and_multiply_x(x: Ciphertext, y: float, evaluator: Evaluator, encoder: CKKSEncoder, relin_keys: RelinKeys):
    y_ = Plaintext()
    encoder.encode(y, x.scale(), y_)
    evaluator.mod_switch_to_inplace(y_, x.parms_id())
    z = Ciphertext()
    evaluator.multiply_plain(x, y_, z)
    evaluator.relinearize_inplace(z, relin_keys)
    evaluator.rescale_to_next_inplace(z)
    return z
