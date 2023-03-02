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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Problem Inputs plain GD
# fixing seed
np.random.seed(1717171)


def get_random_Q_p(k, size, integer):
    S = ortho_group.rvs(size)
    R = np.random.randint(1, 100, d) if integer else np.random.rand(d)
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


# Steps
T_gd = 9
T_agd = 6

# Matrix params
ds = [2, 4, 8]
ks = np.arange(1.5, 10.5, 0.5)

gd_err = pd.DataFrame(np.zeros((len(ks), len(ds))), columns=ds, index=ks)
agd_err = pd.DataFrame(np.zeros((len(ks), len(ds))), columns=ds, index=ks)

for d in ds:
    # Variables to store data for analysis later on
    x_cols = ["x[" + str(i) + "]" for i in range(d)]
    step_gd = dict(
        zip(range(T_gd + 1), [pd.DataFrame(np.zeros((len(ks), d)), columns=x_cols) for i in range(T_gd + 1)]))
    step_he_gd = dict(
        zip(range(T_gd + 1), [pd.DataFrame(np.zeros((len(ks), d)), columns=x_cols) for i in range(T_gd + 1)]))
    f_step_gd = pd.DataFrame(np.zeros((len(ks), T_gd + 1)))
    f_step_he_gd = pd.DataFrame(np.zeros((len(ks), T_gd + 1)))
    step_agd = dict(
        zip(range(T_agd + 1), [pd.DataFrame(np.zeros((len(ks), d)), columns=x_cols) for i in range(T_agd + 1)]))
    step_he_agd = dict(
        zip(range(T_agd + 1), [pd.DataFrame(np.zeros((len(ks), d)), columns=x_cols) for i in range(T_agd + 1)]))
    f_step_agd = pd.DataFrame(np.zeros((len(ks), T_agd + 1)))
    f_step_he_agd = pd.DataFrame(np.zeros((len(ks), T_agd + 1)))
    x_opt = pd.DataFrame(np.zeros((len(ks), d)))
    f_opt = pd.DataFrame(np.zeros(len(ks)))

    for i, k in enumerate(ks):
        Q, p = get_random_Q_p(k, d, False)
        x_opt_i = -np.linalg.inv(Q).dot(p)
        x0 = x_opt_i + 0.5
        x_opt.iloc[i] = x_opt_i.T[0]
        f_opt.iloc[i] = f(x_opt_i)[0][0]

        eigenvals = np.linalg.eigvals(Q)
        beta = np.max(eigenvals)
        alpha = np.min(eigenvals)
        kappa = beta / alpha
        gamma = (kappa ** 0.5 - 1) / (kappa ** 0.5 + 1)
        print("kappa: {}".format(kappa))

        # solution
        print("optimal argument: {}".format(x_opt_i))
        print("optimal value: {}".format(f(x_opt_i)))

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
        step_gd[0].iloc[[i]] = x0[:, 0]
        step_he_gd[0].iloc[[i]] = x0_dec[:, 0]
        f_step_he_gd[0][i] = f(x0_dec[:, 0])
        f_step_gd[0][i] = f(x0[:, 0])
        xs_gd = gd_qp(Q, p, T_gd, x0)
        xs_gd_enc = he_gd_qp_ckks(Q_enc, p_enc, d, alpha, beta, T_gd, x0_enc, evaluator,
                                  ckks_encoder, gal_keys, relin_keys, encryptor, scale)
        xs_gd_dec = [decrypt_array(val, decryptor, ckks_encoder, d, d) for val in xs_gd_enc]
        for t in range(len(xs_gd_dec)):
            x_gd = xs_gd[t]
            x_gd_dec = xs_gd_dec[t]
            step_gd[t].iloc[[i]] = x_gd[:, 0]
            step_he_gd[t].iloc[[i]] = x_gd_dec[:, 0]
            f_step_he_gd[t][i] = f(x_gd_dec[:, 0])
            f_step_gd[t][i] = f(x_gd[:, 0])

        # Problem Inputs HE GD
        Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
        p_enc = encrypt_array(squarify(p, 0.0, d), encryptor, ckks_encoder, scale)
        x0_enc = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
        y0_enc = encrypt_array(squarify(x0, 0.0, d), encryptor, ckks_encoder, scale)
        x0_dec = decrypt_array(x0_enc, decryptor, ckks_encoder, d, d)
        step_agd[0].iloc[[i]] = x0[:, 0]
        step_he_agd[0].iloc[[i]] = x0_dec[:, 0]
        f_step_he_agd[0][i] = f(x0_dec[:, 0])
        f_step_agd[0][i] = f(x0[:, 0])
        xs_agd = agd_qp(Q, p, T_agd, x0)
        xs_agd_enc = he_agd_qp_ckks(Q_enc, p_enc, d, alpha, beta, T_agd, x0_enc, y0_enc, evaluator,
                                    ckks_encoder, gal_keys, relin_keys, encryptor, scale)
        xs_agd_dec = [decrypt_array(val, decryptor, ckks_encoder, d, d) for val in xs_agd_enc]
        for t in range(len(xs_agd_dec)):
            x_agd = xs_agd[t]
            x_agd_dec = xs_agd_dec[t]
            step_agd[t].iloc[[i]] = x_agd[:, 0]
            step_he_agd[t].iloc[[i]] = x_agd_dec[:, 0]
            f_step_he_agd[t][i] = f(x_agd_dec[:, 0])
            f_step_agd[t][i] = f(x_agd[:, 0])

    gd_err[d] = (f_step_gd[[T_gd]] - f_opt.values).values
    agd_err[d] = (f_step_agd[[T_agd]] - f_opt.values).values

bar(range(6), [(step_gd[i] - step_he_gd[i]).mean()[0] for i in range(6)])

ax = (f_step_he_gd - f(x_opt_i)[0][0]).boxplot()
ax.set_yscale('log')
plt.show()

ax = (step_he_gd[0]).plot(kind='scatter', x=0, y=1, c='b', marker='.')
(step_he_gd[1]).plot(kind='scatter', x=0, y=1, ax=ax, c='r', marker='.')
(step_he_gd[2]).plot(kind='scatter', x=0, y=1, ax=ax, c='y', marker='.')
(step_he_gd[3]).plot(kind='scatter', x=0, y=1, ax=ax, c='g', marker='.')
(step_he_gd[4]).plot(kind='scatter', x=0, y=1, ax=ax, c='c', marker='.')
(step_he_gd[5]).plot(kind='scatter', x=0, y=1, ax=ax, c='r', marker='.')
(step_he_gd[6]).plot(kind='scatter', x=0, y=1, ax=ax, c='y', marker='.')
(step_he_gd[7]).plot(kind='scatter', x=0, y=1, ax=ax, c='g', marker='.')
(step_he_gd[8]).plot(kind='scatter', x=0, y=1, ax=ax, c='c', marker='.')
avg = np.array([(x.mean().values) for x in step_he_gd.values()])
plot(avg[:, 0], avg[:, 1], ':')

axins1 = zoomed_inset_axes(ax, zoom=20000000, loc='right')
(step_he_gd[0]).plot(kind='scatter', x=0, y=1, c='b', ax=axins1, marker='.')
# fix the number of ticks on the inset axes
axins1.set_xlim(step_he_gd[0]['x[0]'].min(), step_he_gd[0]['x[0]'].max())
axins1.set_ylim(step_he_gd[0]['x[1]'].min(), step_he_gd[0]['x[1]'].max())
axins1.yaxis.get_major_locator().set_params(nbins=1)
axins1.xaxis.get_major_locator().set_params(nbins=1)
axins1.tick_params(labelleft=False, labelbottom=False)
mark_inset(ax, axins1, loc1=1, loc2=2, fc="none", ec="0.5")

axins8 = zoomed_inset_axes(ax, zoom=20000, loc='lower right')
(step_he_gd[8]).plot(kind='scatter', x=0, y=1, c='m', ax=axins8, marker='.')
# fix the number of ticks on the inset axes
axins8.set_xlim(step_he_gd[8]['x[0]'].min(), step_he_gd[8]['x[0]'].max())
axins8.set_ylim(step_he_gd[8]['x[1]'].min(), step_he_gd[8]['x[1]'].max())
axins8.yaxis.get_major_locator().set_params(nbins=1)
axins8.xaxis.get_major_locator().set_params(nbins=1)
axins8.tick_params(labelleft=False, labelbottom=False)
mark_inset(ax, axins8, loc1=2, loc2=3, fc="none", ec="0.5")
