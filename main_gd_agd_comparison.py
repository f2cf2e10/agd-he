from agd.agd import agd_qp, he_agd_qp_ckks
from agd.gd import gd_qp, he_gd_qp_ckks
from agd.seal.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, Encryptor, CoeffModulus, \
    Evaluator, Decryptor, CKKSEncoder, IntVector, Plaintext, Ciphertext, \
    GaloisKeys, RelinKeys, PublicKey
import numpy as np
from scipy.stats import ortho_group
import pandas as pd
import matplotlib.pyplot as plt
from agd.matrix.utils import decrypt_array, encrypt_array, squarify
import time

# Problem Inputs plain GD
# fixing seed
np.random.seed(1717171)


def get_random_Q_p(k, size, integer):
    S = ortho_group.rvs(size)
    R = np.random.randint(1, 100, size) if integer else np.random.rand(size)
    Rmin = np.min(R)
    Rmax = np.max(R)
    L = ((R - Rmin) / (Rmax - Rmin) * (k - 1) + 1) * Rmin
    Q = S.dot(np.diag(L)).dot(S.T)
    p = np.random.rand(size, 1)
    return Q, p


def get_random_Q_p_test(k, size):
    A = np.random.randn(size, size)
    B = A.T.dot(A)
    n_zeros = np.random.randint(0, size * (size - 1) / 2 + 1)
    for _ in range(n_zeros):
        i = np.random.randint(1, size)
        j = np.random.randint(0, size - 1)
        B[i, j] = 0.0
    R, S = np.linalg.eig(B)
    Rmin = np.min(R)
    Rmax = np.max(R)
    L = ((R - Rmin) / (Rmax - Rmin) * (k - 1) + 1) * Rmin
    Q = S.dot(np.diag(L)).dot(S.T)
    p = np.random.rand(size, 1)
    return Q, p


def f(x):
    return 0.5 * x.T.dot(Q).dot(x) + p.T.dot(x)


def df(x):
    return Q.dot(x) + p


def norm2(x):
    return np.sum(x ** 2) ** 0.5


# Steps
T_gd = 9
T_agd = 6

# Matrix params
ds = [2, 4, 8]
ks = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
N = 1
r = 0.5

gd_last_step = []
agd_last_step = []
opt = []

for sample in range(N):
    t0 = time.time()
    gd_last_step_i = np.zeros((len(ks), len(ds)))
    agd_last_step_i = np.zeros((len(ks), len(ds)))
    opt_i = np.zeros((len(ks), len(ds)))
    for j, d in enumerate(ds):
        # Variables to store data for analysis later on
        for i, k in enumerate(ks):
            Q, p = get_random_Q_p(k, d, False)
            x_opt = -np.linalg.inv(Q).dot(p)
            r_i = np.random.randn(d, 1)
            x0 = x_opt + r_i * r / norm2(r_i)
            f_opt = f(x_opt)[0][0]
            opt_i[i][j] = f_opt

            eigenvals = np.linalg.eigvals(Q)
            beta = np.max(eigenvals)
            alpha = np.min(eigenvals)
            kappa = beta / alpha
            gamma = (kappa ** 0.5 - 1) / (kappa ** 0.5 + 1)
            print("kappa: {}".format(kappa))

            # solution
            print("optimal argument: {}".format(x_opt))
            print("optimal value: {}".format(f(x_opt)))

            # HE parameters
            scale = 2.0 ** 40

            parms = EncryptionParameters(scheme_type.ckks)

            poly_modulus_degree = 32768
            parms.set_poly_modulus_degree(poly_modulus_degree)
            parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, IntVector([60] + [40] * 18 + [60])))

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

            # Problem Inputs HE GD
            Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
            p_enc = encrypt_array(squarify(p, 0.0, d), encryptor, ckks_encoder, scale)
            x0_enc = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
            x0_dec = decrypt_array(x0_enc, decryptor, ckks_encoder, d, d)
            xs_gd = gd_qp(Q, p, T_gd, x0)
            xs_gd_enc = he_gd_qp_ckks(Q_enc, p_enc, d, alpha, beta, T_gd, x0_enc, evaluator,
                                      ckks_encoder, gal_keys, relin_keys, encryptor, scale)
            xs_gd_dec = [decrypt_array(val, decryptor, ckks_encoder, d, d)[:, 0].reshape(d, 1) for val in xs_gd_enc]
            x_gd_dec_last = xs_gd_dec[len(xs_gd_dec) - 1]
            gd_last_step_i[i][j] = f(x_gd_dec_last)[0][0]

            # Problem Inputs HE GD
            Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
            p_enc = encrypt_array(squarify(p, 0.0, d), encryptor, ckks_encoder, scale)
            x0_enc = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
            y0_enc = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
            x0_dec = decrypt_array(x0_enc, decryptor, ckks_encoder, d, d)
            xs_agd = agd_qp(Q, p, T_agd, x0)
            xs_agd_enc = he_agd_qp_ckks(Q_enc, p_enc, d, alpha, beta, T_agd, x0_enc, y0_enc, evaluator,
                                        ckks_encoder, gal_keys, relin_keys, encryptor, scale)
            xs_agd_dec = [decrypt_array(val, decryptor, ckks_encoder, d, d)[:, 0].reshape(d, 1) for val in xs_agd_enc]
            x_agd_dec_last = xs_agd_dec[len(xs_agd_dec) - 1]
            agd_last_step_i[i][j] = f(x_agd_dec_last)[0][0]
    df_gd_last_step_i = pd.DataFrame(gd_last_step_i, columns=ds, index=ks)
    df_gd_last_step_i.to_csv("./data/comparison/gd" + str(sample) + ".csv")
    gd_last_step += [df_gd_last_step_i]
    df_agd_last_step_i = pd.DataFrame(agd_last_step_i, columns=ds, index=ks)
    df_agd_last_step_i.to_csv("./data/comparison/agd" + str(sample) + ".csv")
    agd_last_step += [df_agd_last_step_i]
    df_opt_i = pd.DataFrame(opt_i, columns=ds, index=ks)
    df_opt_i.to_csv("./data/comparison/opt" + str(sample) + ".csv")
    opt += [df_opt_i]
    print("=======================================")
    print(time.time() - t0)
    print("=======================================")

import seaborn as sns
from functools import reduce

gd_last_step = []
agd_last_step = []
opt = []
for i in range(100):
    gd_last_step += [pd.read_csv("./data/comparison/gd" + str(i) + ".csv", index_col=0)]
    agd_last_step += [pd.read_csv("./data/comparison/agd" + str(i) + ".csv", index_col=0)]
    opt += [pd.read_csv("./data/comparison/opt" + str(i) + ".csv", index_col=0)]
N = len(opt)
agd_err = [agd_last_step[i] - opt[i] for i in range(N)]
gd_err = [gd_last_step[i] - opt[i] for i in range(N)]
mean_agd_err = reduce(lambda x, y: x + y, agd_err) / N
mean_gd_err = reduce(lambda x, y: x + y, gd_err) / N
median_agd_err = pd.DataFrame(np.zeros(mean_agd_err.shape), index=mean_agd_err.index, columns=mean_agd_err.columns)
median_gd_err = pd.DataFrame(np.zeros(mean_gd_err.shape), index=mean_gd_err.index, columns=mean_gd_err.columns)
for i in range(len(median_gd_err.index)):
    for j in range(len(median_gd_err.columns)):
        median_agd_err.iloc[i][j] = np.median(np.array([agd_err[k].iloc[i][j] for k in range(N)]))
        median_gd_err.iloc[i][j] = np.median(np.array([gd_err[k].iloc[i][j] for k in range(N)]))

median_gd_err.T.plot.bar(logy=True, ylabel='f(x) -f(x*)', xlabel='matrix dimension')
plt.legend(loc='upper center', mode='expand', ncol=len(median_gd_err.index))

median_agd_err.T.plot.bar(logy=True, ylabel='f(x) -f(x*)', xlabel='matrix dimension')
plt.legend(loc='upper center', mode='expand', ncol=len(median_agd_err.index))

ds = agd_err[0].columns
ks = agd_err[0].index
N = 100
df_gd = pd.DataFrame(np.zeros((N * len(ks) * len(ds), 3)), columns=['k', 'd', 'v'])
df_agd = pd.DataFrame(np.zeros((N * len(ks) * len(ds), 3)), columns=['k', 'd', 'v'])
i = 0
for j in range(N):
    for k in ks:
        for d in ds:
            df_gd.iloc[i]['v'] = gd_err[j].loc[k][d]
            df_agd.iloc[i]['v'] = agd_err[j].loc[k][d]
            df_gd.iloc[i]['k'] = df_agd.iloc[i]['k'] = k
            df_gd.iloc[i]['d'] = df_agd.iloc[i]['d'] = int(d)
            i += 1

matplotlib.rc('axes', titlesize=22)
matplotlib.rc('axes', labelsize=22)
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)

figure(figsize=(12,9))
ax = sns.boxplot(x="d", y="v", hue="k", data=df_gd)
ax.set_yscale('log')
ax.legend(loc='upper center', mode='expand', ncol=len(median_agd_err.index))
ax.set_ylim(1E-11, 10)
ax.set(xlabel='matrix dimension')
ax.set(ylabel='f(x) - f(x*)')

figure(figsize=(12,9))
ax = sns.boxplot(x="d", y="v", hue="k", data=df_agd)
ax.set_yscale('log')
ax.legend(loc='upper center', mode='expand', ncol=len(median_agd_err.index))
ax.set_ylim(1E-11, 10)
ax.set(xlabel='matrix dimension')
ax.set(ylabel='f(x) - f(x*)')
