# preliminaries
import numpy as np
import pandas as pd
import time
from datetime import date
from fredapi import Fred
import matplotlib.pyplot as plt
from objects import *
from functions import *
from scipy.stats import wishart

# nowcast parameters
date_start = pd.to_datetime('1980-01-01').date()
date_end = pd.to_datetime('2019-12-31').date()
#date_end = date.today()
series = ['GDPC1', 'GDI', 'PCEC', 'GPDIC1']

# pull series for nowcast
fred = Fred(api_key='7973903b6bd0035600fc889f0f1246c7')
data_raw = fred.get_series(series[0]).to_frame(name=series[0])
for i in range(1, len(series)):
    series_x = fred.get_series(series[i]).to_frame(name=series[i])
    data_raw = data_raw.join(series_x, how='outer')

# clean series
data_clean = 400*(np.log(data_raw) - np.log(data_raw.shift(1)))
data_clean.index = data_clean.index.date
data_clean = data_clean.loc[date_start:date_end]

# estimate bayesian nowcast
n_save = 2000
n_burnin = int(0.2*n_save)
n_draws = n_save + n_burnin

# state-space model parameters
m = 1
n = len(series)
tt = data_clean.shape[0]
Z = 0.1*np.ones(shape=(n, m))
T = np.identity(m)
R = np.identity(m)
H = np.identity(n)
Q = np.identity(m)
mu0 = np.zeros(shape=(m, 1))
sigma0 = np.identity(m)
y = data_clean.to_numpy().T

# parameter priors
Z_prior_mean = 0*Z
Z_prior_sigma = np.zeros(shape=(m, m))
T_prior_mean = 0*T
T_prior_sigma = np.zeros(shape=(m, m))
eps_prior_nu = 0
eps_prior_s2 = np.identity(n)
ups_prior_nu = 0
ups_prior_s2 = np.identity(m)

# result storage matrices
x_draws = np.zeros(shape=(m, tt, n_save))
Z_draws = np.zeros(shape=(n, m, n_save))
T_draws = np.zeros(shape=(m, m, n_save))
GDP_draws = np.zeros(shape=(1, tt, n_save))

# gibbs sampler
print_progress_bar(0, n_draws, prefix='Progress:', suffix='Complete', length=50)
idraw_save = 0
for idraw in range(0, n_draws):
    # create model object
    model = SSM(Z, T, R, H, Q, mu0, sigma0, y)

    # draw x ~ p(x|Z, T, H, Q)
    x = model.simulation_smoother()

    # draw H ~ p(H |x, Z, T, Q)
    eps_post_nu = eps_prior_nu + tt
    Z_post_sigma = Z_prior_sigma + x.dot(x.T)
    Z_post_mean = np.linalg.inv(Z_post_sigma)*(Z_prior_sigma*Z_prior_mean + x.dot(y.T).reshape(-1, 1))
    temp = (eps_prior_s2 + Z_prior_mean.dot(Z_prior_sigma).dot(Z_prior_mean.T) + y.dot(y.T))
    eps_post_s2 = temp - Z_post_mean.dot(Z_post_sigma).dot(Z_post_mean.T)
    H = abs(np.linalg.inv(wishart.rvs(eps_post_nu, np.linalg.inv(abs(eps_post_s2)))))

    # draw Q ~ p(Q |x, Z, T, H)
    k = x.shape[1]
    x_t = x[:, 1:k]
    x_t1 = x[:, 0:k - 1]
    ups_post_nu = ups_prior_nu + tt
    T_post_sigma = T_prior_sigma + x_t1.dot(x_t1.T)
    T_post_mean = np.linalg.inv(T_post_sigma) * (T_prior_sigma * T_prior_mean + x_t1.dot(x_t.T).reshape(-1, 1))
    temp = (ups_prior_s2 + T_prior_mean.dot(T_prior_sigma).dot(T_prior_mean.T) + x_t.dot(x_t.T))
    ups_post_s2 = temp - T_post_mean.dot(T_post_sigma).dot(T_post_mean.T)
    Q = np.linalg.inv(wishart.rvs(ups_post_nu, np.linalg.inv(ups_post_s2)).reshape(-1, 1))

    # draw Z ~ p(Z |x, T, H, Q)
    mean = Z_post_mean.flatten()
    cov = np.kron(np.linalg.inv(Z_post_sigma), H)
    Z = np.random.multivariate_normal(mean, cov).reshape(-1, 1)

    # draw T ~ p(T |x, Z, H, Q)
    mean = T_post_mean.flatten()
    cov = np.kron(np.linalg.inv(T_post_sigma), Q)
    T = np.random.multivariate_normal(mean, cov).reshape(-1, 1)

    # print progress
    print_progress_bar(idraw + 1, n_draws, prefix='Progress:', suffix='Complete', length=50)

    # save results
    if idraw >= n_burnin:
        x_draws[:, :, idraw_save] = x
        Z_draws[:, :, idraw_save] = Z
        T_draws[:, :, idraw_save] = T
        GDP_draws[:, :, idraw_save] = x.T.dot(Z[0, 0]).T
        idraw_save = idraw_save + 1

# compute xtiles of draws
x_draw_mean = np.mean(x_draws, 2)
Z_draw_mean = np.mean(Z_draws, 2)
T_draw_mean = np.mean(T_draws, 2)
GDP_draw_mean = np.mean(GDP_draws, 2)

# plot GDP vs nowcast
plt.figure(0)
plt.plot(y[0, :], label='GDP')
plt.plot(GDP_draw_mean.T, label='GDP Nowcast')
plt.title('GDP vs Nowcast')
plt.legend()
plt.show()

# T parameter draws
plt.figure(1)
plt.plot(T_draws[0, 0, :])
plt.title('T Parameter Draws')
plt.show()

# Z parameter draws
plt.figure(2)
plt.plot(Z_draws[0, 0, :])
plt.title('Z Parameter Draws')
plt.show()