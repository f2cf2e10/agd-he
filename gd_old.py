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
#kappas = np.array([i for i in range(2, 103, step) for _ in range(repeat_each)])
kappas = np.array([100.]*repeat_each)
repeat = len(kappas) 

f_step = pd.DataFrame(np.zeros((repeat, T+1)))
f_step_he = pd.DataFrame(np.zeros((repeat, T+1)))

step = dict(zip(range(T+1), [pd.DataFrame(np.zeros((repeat, 2))) for i in range(T+1)] ))
step_he = dict(zip(range(T+1), [pd.DataFrame(np.zeros((repeat, 2))) for i in range(T+1)] ))
norm2_noise = pd.DataFrame(np.zeros((repeat, T+1)))

np.random.seed(1717171)

k = kappas[0]
Q, _ = get_random_Q_p(k)
p = -Q.dot(x_opt)  

for i in range(len(kappas)):
    #k = np.random.randint(2, 1000)
    k = kappas[i]
    #Q, _ = get_random_Q_p(k)
    #setting p to fix x_opt
    #p = -Q.dot(x_opt)  

    # Pre-calculations
    eigenvals = np.linalg.eigvals(Q)
    beta = np.max(eigenvals)
    alpha = np.min(eigenvals)
    kappa = beta/alpha
    eta = 2/(alpha + beta)
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
    ###
    #poly_modulus_degree = 65536
    ###
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, IntVector([60] + [40]*T*R + [60])))

    context = SEALContext(parms)
    ###
    #NOTE: the none means to ommit securiy checks, we are doing it here to push as much as 
    #possible the poly_modulud_degree
    #context = SEALContext(parms, True, sec_level_type.none)
    ###
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
    eta_ = Plaintext()
    ckks_encoder.encode(-eta, p_enc.scale(), eta_)
    p_enc_eta = Ciphertext()
    evaluator.multiply_plain(p_enc, eta_, p_enc_eta)
    evaluator.relinearize_inplace(p_enc_eta, relin_keys)
    evaluator.rescale_to_next_inplace(p_enc_eta)

    # Initialization AGD
    x = x0

    # Initialization HE
    x_enc = x0_enc
    x0_dec = decrypt_array(x0_enc, decryptor, ckks_encoder, d, d)
    step[0].iloc[[i]] = x0[:, 0]
    step_he[0].iloc[[i]] = x0_dec[:, 0]
    f_step_he[0][i] = f(x0_dec[:, 0])
    f_step[0][i] = f(x0[:, 0])
    norm2_noise[0][i] = sum((x0_dec[:, 0]-x0[:,0])**2)**0.5
    for t in range(1,T+1):
        print("({}, {})".format(i, t))
        # Plain
        x = x - eta * df(x)

        # Enc
        #print("Modulus chain index for y_enc: {}".format(
        #    context.get_context_data(y_enc_.parms_id()).chain_index()))

        MMultQ_enc_x_enc_ = ours(Q_enc, x_enc, evaluator,
                                 ckks_encoder, gal_keys, relin_keys, 
                                 d, -eta, scale)

        p_enc_eta.set_scale(MMultQ_enc_x_enc_.scale())
        evaluator.mod_switch_to_inplace(
            p_enc_eta, MMultQ_enc_x_enc_.parms_id())
        evaluator.add_inplace(MMultQ_enc_x_enc_, p_enc_eta)
        x_enc = rescale_and_mod_switch_y_and_add_x(
            MMultQ_enc_x_enc_, x_enc, evaluator)
        #print("Modulus chain index for y_enc: {}".format(
        #    context.get_context_data(y_enc.parms_id()).chain_index()))

        # Comparison
        x_dec = decrypt_array(x_enc, decryptor, ckks_encoder, d, d)
        print(x_dec)

        step[t].iloc[[i]] = x[:, 0]
        step_he[t].iloc[[i]] = x_dec[:, 0]
        f_step_he[t][i] = f(x_dec[:, 0])
        f_step[t][i] = f(x[:, 0])
        norm2_noise[t][i] = sum((x_dec[:, 0]-x[:,0])**2)**0.5
        #print("==================================================================================")

lm=lambda y: LinearRegression(fit_intercept = True).fit(np.arange(T).reshape(-1, 1), y)
plt.plot(kappas, (step_he-step).apply(lambda x: lm(np.log(x)).coef_[0], axis=1).values, '.')
pd.DataFrame((step_he-step).apply(lambda x: lm(np.log(x)).coef_[0], axis=1).values.reshape(len(kappas)//repeat_each, repeat_each).T,
             columns=kappas.reshape(len(kappas)//repeat_each, repeat_each).T[0]).boxplot()

plt.plot(kappas, (norm2_noise).apply(lambda x: lm(np.log(x)).coef_[0], axis=1).values, '.')
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