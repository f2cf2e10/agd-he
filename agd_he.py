from typing import Tuple, List
from numpy.core import numeric
from seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, \
    Encryptor, CoeffModulus, Evaluator, Decryptor, \
    Plaintext, Ciphertext, IntegerEncoder, PlainModulus, \
    BatchEncoder, CKKSEncoder, Int64Vector, UInt64Vector, \
    IntVector, DoubleVector, CiphertextVector, GaloisKeys, \
    RelinKeys
import numpy as np
import itertools


def squarify(M: np.ndarray, val: float, N: int = None):
    a, b = M.shape
    if N is None:
        if a > b:
            padding = ((0, 0), (0, a-b))
        else:
            padding = ((0, b-a), (0, 0))
    else:
        padding = ((0, max(N-a, 0)), (0, max(N-b, 0)))
    return np.pad(M, padding, mode='constant', constant_values=val)


def diag_repr(U: np.ndarray, i: int) -> np.ndarray:
    N, M = U.shape
    if (N != M):
        raise Exception("diag_repr only works for square matrices")
    else:
        result = np.concatenate([np.diag(U, i), np.diag(U, -N+i)])
        if len(result) != N:
            raise Exception("diag_repr index is out of bounds")
        return result


def lin_trans(U: np.ndarray, ct: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
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
    M, N = U.shape
    if (N != M):
        raise Exception("lin_trans only works for square matrices")
    scale = ct.scale()
    parms_id = ct.parms_id()
    acc = CiphertextVector()
    for l in range(N):
        ul_vec_np = diag_repr(U, l)
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
    if not (relin_keys is None):
        evaluator.relinearize_inplace(out, relin_keys)
        evaluator.rescale_to_next_inplace(out)
    return out


def sigma_permutation(d, nmax, val=1.0):
    n = d*d
    Usigma = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        Usigma[d*i+j, d*i+((i+j) % d)] = val 
    output = np.kron(np.eye(2), Usigma)
    return output[0:nmax, 0:nmax]


def tau_permutation(d, nmax, val=1.0):
    n = d*d
    Utau = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        Utau[d*i+j, d*((i+j) % d)+j] = val
    output = np.kron(np.eye(2), Utau)
    return output[0:nmax, 0:nmax]


def phi_permutation_k(k, d, nmax, val=1.0):
    n = d*d
    Vk = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        Vk[d*i+j, d*i+((j+k) % d)] = val 
    return Vk


def psi_permutation_k(k, d, nmax, val=1.0):
    n = d*d
    Wk = np.zeros(n*n).reshape(n, n)
    for i, j in itertools.product(range(d), range(d)):
        Wk[d*i+j, d*((i+k) % d)+j] = val 
    return Wk


def cA_x_cB(A: Ciphertext, B: Ciphertext, evaluator: Evaluator, relin_keys: RelinKeys):
    """
    Element-wise product of cipherthextA and ciphertextB
    """
    auto_rescale_and_mod_switch([A,B], evaluator)
    C = Ciphertext()
    evaluator.multiply(A, B, C)
    evaluator.relinearize_inplace(C, relin_keys)
    evaluator.rescale_to_next_inplace(C)
    return C


def cA_dot_cB(ct_A: Ciphertext, ct_B: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder,
                gal_keys: GaloisKeys, relin_keys: RelinKeys, d: int, scale:float=1.0) -> Ciphertext:
    """
    Matrix product of CiphertextA and CiphertextB
    Inspired by "Secure Outsourced Matrix Computationand Application to Neural Networks?"
    Link: https://eprint.iacr.org/2018/1041.pdf
    """
    nmax = encoder.slot_count()
    n = d*d
    if n > nmax/2:
       raise("Matrix dimenson is higher than the one suported by the encoder")
    Usigma = sigma_permutation(d, nmax, np.sign(scale) * np.abs(scale)**0.5)
    Utau = tau_permutation(d, nmax, np.abs(scale)**0.5)
    ct_A0 = lin_trans(Usigma, ct_A, evaluator, encoder, gal_keys, relin_keys)
    ct_B0 = lin_trans(Utau, ct_B, evaluator, encoder, gal_keys, relin_keys)
    ct_Ak = CiphertextVector()
    ct_Bk = CiphertextVector()
    for k in range(1, d):
        Vk = phi_permutation_k(k, d, nmax, 1.0)
        Wk = psi_permutation_k(k, d, nmax, 1.0)
        if Vk.sum() == 0 or Wk.sum() == 0:
            continue

        ct_Ak.append(lin_trans(Vk, ct_A0, evaluator,
                     encoder, gal_keys, relin_keys))
        ct_Bk.append(lin_trans(Wk, ct_B0, evaluator,
                     encoder, gal_keys, relin_keys))

    ct_AB = cA_x_cB(ct_A0, ct_B0, evaluator, relin_keys)
    for k in range(len(ct_Ak)):
        ct_ABk = cA_x_cB(ct_Ak[k], ct_Bk[k], evaluator, relin_keys)
        parms_id = ct_ABk.parms_id()
        evaluator.mod_switch_to_inplace(ct_AB, parms_id)
        ct_ABk.set_scale(ct_AB.scale())
        evaluator.add_inplace(ct_AB, ct_ABk)
    return ct_AB


def encrypt_array(matrix: np.ndarray, encryptor: Encryptor, encoder: CKKSEncoder, scale: float) -> Ciphertext:
    M, N = matrix.shape
    Nmax = encoder.slot_count()
    plain = Plaintext()
    cmatrix = Ciphertext()
    encoder.encode(DoubleVector((matrix.flatten().tolist())
                   * int(Nmax/N/M)), scale, plain)
    encryptor.encrypt(plain, cmatrix)
    return cmatrix


def decrypt_array(cipher: Ciphertext, decryptor: Decryptor, encoder: CKKSEncoder, M: int, N: int) -> np.ndarray:
    plain = Plaintext()
    decryptor.decrypt(cipher, plain)
    vec = DoubleVector()
    encoder.decode(plain, vec)
    matrix = np.array(vec[0:(M*N)]).reshape(M, N)
    return matrix

def rescale_and_mod_switch(x:Ciphertext, new_scale:float, new_parms:List[int], evaluator:Evaluator):
    x.set_scale(new_scale)
    evaluator.mod_switch_to_inplace(x, new_parms)

def auto_rescale_and_mod_switch(x: List[Ciphertext], evaluator:Evaluator):
    scales = [i.scale() for i in x]
    parms = [i.parms_id() for i in x]
    new_scale = np.max(scales)
    i_scale = np.argmax(scales)
    new_parms = parms[i_scale]
    for xi in x:
        rescale_and_mod_switch(xi, new_scale, new_parms, evaluator) 

def rescale_and_mod_switch_y_and_add_x(x: Ciphertext, y:Ciphertext, evaluator:Evaluator) -> Ciphertext:
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

def rescale_and_mod_switch_y_and_multiply_x(x: Ciphertext, y:float, evaluator:Evaluator, encoder:CKKSEncoder, relin_keys:RelinKeys):
    y_ = Plaintext()
    encoder.encode(y, x.scale(), y_)
    evaluator.mod_switch_to_inplace(y_, x.parms_id())
    z = Ciphertext()
    evaluator.multiply_plain(x, y_, z)
    evaluator.relinearize_inplace(z, relin_keys)
    evaluator.rescale_to_next_inplace(z)
    return z

def acg_qp(Q: np.ndarray, p: np.ndarray, beta: float, alpha: float, n: int, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = lambda x: x.T.dot(Q).dot(x) + p.dot(x)
    df = lambda x: Q.dot(x) + p
    x = x0
    y = x0
    kappa = beta/alpha
    gamma = (kappa**0.5-1)/(kappa**0.5+1)
    for t in range(n):
        y_ = y
        y = x - 1/beta * df(x)
        x_ = x
        x = (1 + gamma) * y - gamma * y_
    tol = np.abs(f(x) - f(x_))
    print("The error after {} steps is: {} ".format(n, tol))
    return x, tol


def he_acg_qp(Q: Ciphertext, p: Ciphertext, beta: float, kappa: float, n: int, c0: Ciphertext, evaluator: Evaluator, encoder: CKKSEncoder, gal_keys: GaloisKeys, relin_keys: RelinKeys) -> Tuple[Ciphertext, Ciphertext]:
    c = c0
    d = c0
    gamma = (kappa**0.5-1)/(kappa**0.5+1)
    for t in range(n):
        d_ = d
        #d = evaluator.add_inplace(c, - 1/beta * df(c)
        c_=c
        c=(1 + gamma) * d - gamma * d_
    tol=np.abs(f(c) - f(c_))
    return c, tol
