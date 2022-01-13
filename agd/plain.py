import numpy as np
from typing import Callable

Q = np.array([[2, .5], [.5, 1]])
p = np.array([[1.0], [1.0]])
f = lambda x: x.T.dot(Q).dot(x) + p.dot(x)
df = lambda x: Q.dot(x) + p
eigenvals = np.linalg.eigvals(Q)
beta = np.max(eigenvals)
alpha = np.min(eigenvals)

def alpha_strong_convex_beta_smooth(f: Callable, df:Callable, beta:float, alpha:float, epsilon:float=1e-5) -> np.ndarray:
    t = 0
    v0 = 1E100*np.array( [np.random.randn(2)]).T
    x = v0
    y = v0
    tol = 2*epsilon + 1
    k = beta/alpha
    gamma = (k**0.5-1)/(k**0.5+1)
    while (tol > epsilon):
        y_ = y
        y = x - 1/beta * df(x)
        x_ = x
        x = (1 + gamma) * y - gamma * y_
        t = t+1
        tol = np.abs(f(x) - f(x_))
    R2 = (x-v0).T.dot(x-v0)
    T = -k**0.5 * np.log(2*epsilon/(alpha + beta)/R2) + 1
    print("Conveged in {} steps and it should have taken less than {} steps".format(t, T[0][0]))
    return x

def beta_smooth(f:Callable, df:Callable, beta:float, epsilon:float=1E-5) -> np.ndarray:
    t = 0
    v0 = 1E100*np.array( [np.random.randn(2)]).T
    x = v0
    y = v0
    lamb_tm1 = 0
    tol = 2*epsilon + 1
    while (tol > epsilon):
        lamb_t = (1 + (1 + 4*lamb_tm1**2)**0.5)/2
        lamb_tp1 = (1 + (1 + 4*lamb_t**2)**0.5)/2
        gamma = (1 - lamb_t)/lamb_tp1
        y_ = y
        y = x - 1/beta * df(x)
        x_ = x
        x = (1 - gamma) * y + gamma * y_
        t = t+1
        tol = np.abs(f(x) - f(x_))
    R = ((x-v0).T.dot(x-v0))**0.5
    T = R * (beta/epsilon)**0.5
    print("Conveged in {} steps and it should have taken less than {} steps".format(t, T[0][0]))
    return x


np.linalg.inv(Q).dot(-p)
alpha_strong_convex_beta_smooth(f, df, beta, alpha)
beta_smooth(f, df, beta)
