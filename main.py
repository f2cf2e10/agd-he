from agd.seal.seal import EncryptionParameters, scheme_type, \
    SEALContext, print_parameters, KeyGenerator, Encryptor, CoeffModulus, \
    Evaluator, Decryptor, CKKSEncoder, IntVector, Plaintext, Ciphertext, \
    GaloisKeys, RelinKeys, PublicKey, sec_level_type, DoubleVector
import numpy as np
from scipy.stats import ortho_group    
import operator as op
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from agd.matrix.jlks import matrix_multiplication as jlks
from agd.matrix.ours import matrix_multiplication as ours
from agd.matrix.utils import encrypt_array, squarify, decrypt_array, \
    rescale_and_mod_switch, rescale_and_mod_switch_y_and_add_x, \
    rescale_and_mod_switch_y_and_multiply_x

# Problem Inputs plain AGD
d = 2
x_opt = -1*np.ones(d).reshape(d, -1)

def get_random_Q_p(k, size=d):
    S = ortho_group.rvs(d)
    L = np.random.rand(d)
    idxmax = np.argmax(L)
    idxmin = np.argmin(L)
    L[idxmax] = k * L[idxmin]
    Q = S.dot(np.diag(L)).dot(S.T)
    p = np.random.rand(size, 1)
    return Q, p

# Steps
bits = [27, 54, 109, 218, 438, 881]
poly_degree = [2**i for i in range(10, 16)]
T = 5
R = 3

repeat_each = 10
step = 10
kappas = np.array([i for i in range(2, 103, step) for _ in range(repeat_each)])
repeat = len(kappas) 

step = pd.DataFrame(np.zeros((repeat, T)))
step_he = pd.DataFrame(np.zeros((repeat, T)))
norm2_noise = pd.DataFrame(np.zeros((repeat, T)))

np.random.seed(1717171)

for i in range(len(kappas)):
    #k = np.random.randint(2, 1000)
    k = kappas[i]
    Q, _ = get_random_Q_p(k)
    #setting p to fix x_opt
    p = -Q.dot(x_opt)  

    # Pre-calculations
    eigenvals = np.linalg.eigvals(Q)
    beta = np.max(eigenvals)
    alpha = np.min(eigenvals)
    kappa = beta/alpha
    gamma = (kappa**0.5-1)/(kappa**0.5+1)
    print("kappa: {}".format(kappa))

    # solution
    def f(x): return 0.5*x.T.dot(Q).dot(x) + p.T.dot(x)
    def df(x): return Q.dot(x) + p
    print("optimal argument: {}".format(x_opt))
    print("optimal value: {}".format(f(x_opt)))
 
    #x0 = np.array([np.random.randn(d)]).T
    x0 = x_opt
    # HE parameters
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

    # Problem Inputs HE AGD
    Q_enc = encrypt_array(Q, encryptor, ckks_encoder, scale)
    p_enc = encrypt_array(squarify(p, 0.0, d), encryptor, ckks_encoder, scale)
    x0_enc = encrypt_array(squarify(x0, 0.0, d),
                           encryptor, ckks_encoder, scale)
    beta_ = Plaintext()
    ckks_encoder.encode(-1./beta, p_enc.scale(), beta_)
    p_enc_beta = Ciphertext()
    evaluator.multiply_plain(p_enc, beta_, p_enc_beta)
    evaluator.relinearize_inplace(p_enc_beta, relin_keys)
    evaluator.rescale_to_next_inplace(p_enc_beta)

    # Initialization AGD
    y_ = x0
    x_ = x0

    # Initialization HE
    y_enc_ = encrypt_array(squarify(x0, 0.0, d),
                           encryptor, ckks_encoder, scale)
    x_enc_ = encrypt_array(squarify(x0, 0.0, d),
                           encryptor, ckks_encoder, scale)

    for t in range(T):
        # Plain
        y = x_ - 1/beta * df(x_)
        x = (1 + gamma) * y - gamma * y_
        x_ = x
        y_ = y

        # Enc
        print("Modulus chain index for y_enc: {}".format(
            context.get_context_data(y_enc_.parms_id()).chain_index()))

        MMultQ_enc_x_enc_ = ours(Q_enc, x_enc_, evaluator,
                                 ckks_encoder, gal_keys, relin_keys, d, -1./beta)

        p_enc_beta.set_scale(MMultQ_enc_x_enc_.scale())
        evaluator.mod_switch_to_inplace(
            p_enc_beta, MMultQ_enc_x_enc_.parms_id())
        evaluator.add_inplace(MMultQ_enc_x_enc_, p_enc_beta)
        y_enc = rescale_and_mod_switch_y_and_add_x(
            MMultQ_enc_x_enc_, x_enc_, evaluator)

        print("Modulus chain index for y_enc: {}".format(
            context.get_context_data(y_enc.parms_id()).chain_index()))

        gamma_1_y_enc = rescale_and_mod_switch_y_and_multiply_x(
            y_enc, 1+gamma, evaluator, ckks_encoder, relin_keys)
        gamma_y_enc_ = rescale_and_mod_switch_y_and_multiply_x(
            y_enc_, -gamma, evaluator, ckks_encoder, relin_keys)

        print("Modulus chain index for y_enc: {}".format(
            context.get_context_data(y_enc.parms_id()).chain_index()))
        x_enc = rescale_and_mod_switch_y_and_add_x(
            gamma_1_y_enc, gamma_y_enc_, evaluator)
        y_enc_ = y_enc
        x_enc_ = x_enc

        # Comparison
        x_dec = decrypt_array(x_enc, decryptor, ckks_encoder, d, d)

        step_he[t][i] = f(x_dec[:, 0])
        step[t][i] = f(x[:, 0])
        norm2_noise[t][i] = sum((x_dec[:, 0]-x[:,0])**2)**0.5
        print("==================================================================================")

lm=lambda y: LinearRegression(fit_intercept = True).fit(np.arange(T).reshape(-1, 1), y)
figure()
plot(kappas, (step_he-step).apply(lambda x: lm(np.log(x)).coef_[0], axis=1).values, '.')
figure()
pd.DataFrame((step_he-step).apply(lambda x: lm(np.log(x)).coef_[0], axis=1).values.reshape(len(kappas)//repeat_each, repeat_each).T,
columns=kappas.reshape(len(kappas)//repeat_each, repeat_each).T[0]).boxplot()

figure()
plot(kappas, (norm2_noise).apply(lambda x: lm(np.log(x)).coef_[0], axis=1).values, '.')
figure()
pd.DataFrame((norm2_noise).apply(lambda x: lm(np.log(x)).coef_[0], axis=1).values.reshape(len(kappas)//repeat_each, repeat_each).T,
columns=kappas.reshape(len(kappas)//repeat_each, repeat_each).T[0]).boxplot()
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(6,6))

# plot first pandas frame in subplot style
step.boxplot(ax = axs[0,0]) 
axs[0,0].set_title('Plain AGD')

step_he.boxplot(ax = axs[0,1]) 
axs[0,1].set_title('HE AGD')

(step_he-step).boxplot(ax = axs[1,0]) 
axs[1,0].set_title('Difference')

(100*(step/step_he-1)).boxplot(ax = axs[1,1]) 
axs[1,1].set_title('% ratio')

axs[0,0].set(xlabel='step', ylabel='f(x)')
axs[0,1].set(xlabel='step', ylabel='f(x)')
axs[1,0].set(xlabel='step', ylabel='f(x_he) - f(x)')
axs[1,1].set(xlabel='step', ylabel='100*(f(x)/f(x_he)-1)')

plt.show()

step_he.to_csv("./data/step_he_large_kappa.csv")
step.to_csv("./data/step_large_kappa.csv")
norm2_noise.to_csv("./data/noise_large_kappa.csv")