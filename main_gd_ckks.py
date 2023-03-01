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
# Matrix params
d = 2
ks = np.arange(1.5, 10.5, 0.5)

def get_random_Q_p(k, size, integer):
    S = ortho_group.rvs(size)
    R = np.random.randint(1, 100, d) if integer else np.random.rand(d)
    Rmin = np.min(R)
    Rmax = np.max(R)
    L = ((R - Rmin) / (Rmax - Rmin) * (k - 1) + 1) * Rmin
    Q = S.dot(np.diag(L)).dot(S.T)
    p = np.random.rand(size, 1)
    return Q, p


# Steps
bits = [27, 54, 109, 218, 438, 881]
poly_degree = [2 ** i for i in range(10, 16)]
T = 9
R = 2


# Variables to store data for analysis later on
step = dict(zip(range(T + 1), [pd.DataFrame(np.zeros((len(ks), 2)), columns=["x[0]", "x[1]"]) for i in range(T + 1)]))
step_he = dict(zip(range(T + 1), [pd.DataFrame(np.zeros((len(ks), 2)), columns=["x[0]", "x[1]"]) for i in range(T + 1)]))
norm2_noise = pd.DataFrame(np.zeros((len(ks), T + 1)))
f_step = pd.DataFrame(np.zeros((len(ks), T + 1)))
f_step_he = pd.DataFrame(np.zeros((len(ks), T + 1)))
np.random.seed(1717171)


def f(x):
    return 0.5 * x.T.dot(Q).dot(x) + p.T.dot(x)

def df(x):
    return Q.dot(x) + p


for i, k in enumerate(ks):
    Q, p = get_random_Q_p(k, d, False)
    # x_opt = 1 * np.ones(d).reshape(d, -1)
    x_opt = -np.linalg.inv(Q).dot(p)

    eigenvals = np.linalg.eigvals(Q)
    beta = np.max(eigenvals)
    alpha = np.min(eigenvals)
    kappa = beta / alpha
    gamma = (kappa ** 0.5 - 1) / (kappa ** 0.5 + 1)
    print("kappa: {}".format(kappa))

    # solution
    print("optimal argument: {}".format(x_opt))
    print("optimal value: {}".format(f(x_opt)))

    # x0 = np.array([np.random.randn(d)]).T
    x0 = x_opt + 0.5
    # HE parameters
    scale = 2.0 ** 40

    parms = EncryptionParameters(scheme_type.ckks)

    poly_modulus_degree = poly_degree[bits.index(
        next((x for x in bits if x > (120 + T * R * 40)), None))]
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40] * T * R + [60])))

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
    Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
    p_enc = encrypt_array(squarify(p, 0.0, d), encryptor, ckks_encoder, scale)
    x0_enc = encrypt_array(squarify(x0, 0.0, d),
                           encryptor, ckks_encoder, scale)

    x0_dec = decrypt_array(x0_enc, decryptor, ckks_encoder, d, d)
    step[0].iloc[[i]] = x0[:, 0]
    step_he[0].iloc[[i]] = x0_dec[:, 0]
    f_step_he[0][i] = f(x0_dec[:, 0])
    f_step[0][i] = f(x0[:, 0])
    norm2_noise[0][i] = sum((x0_dec[:, 0] - x0[:, 0]) ** 2) ** 0.5

    xs = gd_qp(Q, p, T, x0)
    xs_enc = he_gd_qp_ckks(Q_enc, p_enc, d, alpha, beta, T, x0_enc, evaluator,
                           ckks_encoder, gal_keys, relin_keys, encryptor, scale)
    xs_dec = [decrypt_array(val, decryptor, ckks_encoder, 2, 2) for val in xs_enc]
    for t in range(len(xs_dec)):
        x = xs[t]
        x_dec = xs_dec[t]
        step[t].iloc[[i]] = x[:, 0]
        step_he[t].iloc[[i]] = x_dec[:, 0]
        f_step_he[t][i] = f(x_dec[:, 0])
        f_step[t][i] = f(x[:, 0])

norm2_noise.plot(kind='box', logy=True, legend=False)
bar(range(6), [(step[i] - step_he[i]).mean()[0] for i in range(6)])

ax = (f_step_he - f(x_opt)[0][0]).boxplot()
ax.set_yscale('log')
plt.show()

ax = (step_he[0]).plot(kind='scatter', x=0, y=1, c='b', marker='.')
(step_he[1]).plot(kind='scatter', x=0, y=1, ax=ax, c='r', marker='.')
(step_he[2]).plot(kind='scatter', x=0, y=1, ax=ax, c='y', marker='.')
(step_he[3]).plot(kind='scatter', x=0, y=1, ax=ax, c='g', marker='.')
(step_he[4]).plot(kind='scatter', x=0, y=1, ax=ax, c='c', marker='.')
(step_he[5]).plot(kind='scatter', x=0, y=1, ax=ax, c='r', marker='.')
(step_he[6]).plot(kind='scatter', x=0, y=1, ax=ax, c='y', marker='.')
(step_he[7]).plot(kind='scatter', x=0, y=1, ax=ax, c='g', marker='.')
(step_he[8]).plot(kind='scatter', x=0, y=1, ax=ax, c='c', marker='.')
avg = np.array([(x.mean().values) for x in step_he.values()])
plot(avg[:, 0], avg[:, 1], ':')

axins1 = zoomed_inset_axes(ax, zoom=20000000, loc='right')
(step_he[0]).plot(kind='scatter', x=0, y=1, c='b', ax=axins1, marker='.')
# fix the number of ticks on the inset axes
axins1.set_xlim(step_he[0]['x[0]'].min(), step_he[0]['x[0]'].max())
axins1.set_ylim(step_he[0]['x[1]'].min(), step_he[0]['x[1]'].max())
axins1.yaxis.get_major_locator().set_params(nbins=1)
axins1.xaxis.get_major_locator().set_params(nbins=1)
axins1.tick_params(labelleft=False, labelbottom=False)
mark_inset(ax, axins1, loc1=1, loc2=2, fc="none", ec="0.5")

axins8 = zoomed_inset_axes(ax, zoom=20000, loc='lower right')
(step_he[8]).plot(kind='scatter', x=0, y=1, c='m', ax=axins8, marker='.')
# fix the number of ticks on the inset axes
axins8.set_xlim(step_he[8]['x[0]'].min(), step_he[8]['x[0]'].max())
axins8.set_ylim(step_he[8]['x[1]'].min(), step_he[8]['x[1]'].max())
axins8.yaxis.get_major_locator().set_params(nbins=1)
axins8.xaxis.get_major_locator().set_params(nbins=1)
axins8.tick_params(labelleft=False, labelbottom=False)
mark_inset(ax, axins8, loc1=2, loc2=3, fc="none", ec="0.5")
