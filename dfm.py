
# FILE: dfm.py
# AUTHOR: Christopher Simard
# DESCRIPTION: estimates DFM from with data
# y and number of state variables k using
# notation from Durbin & Koopman (2011)

# NOTATION:
# y_t = Za_t + e_t,      e_t ~ N(0, H)
# a_{t+1} = Ta_t + Rn_t, n_t ~ N(0, Q)
#                        a_1 ~ N(a_1, P_1)

# ARGUMENTS:
# y: data (Txp)
# m: number of state variables
# a: state means (Txm)
# P: state variance (mxmxT)
# T: transition matrix (mxm)
# Z: estimation matrix (pxm)
# R: state error loadings (mxm)
# H: estimation error variance (pxp)
# Q: transition error variance (mxm)

import numpy as np
from scipy.optimize import minimize

class DFM(object):
    def __init__(self, y, m, a=None, P=None, T=None, Z=None, R=None, H=None, Q=None):
        self.y = np.array(y)
        self.m = m
        tt = self.y.shape[0]
        p = self.y.shape[1]

        # default arguments
        if a is None:
            self.a = np.zeros([tt, m])
        else:
            self.a = a

        if P is None:
            i = np.identity(m)
            self.P = np.dstack([i] * tt)
        else:
            self.P = P

        if T is None:
            self.T = np.identity(m)
        else:
            self.T = T

        if Z is None:
            self.Z = np.eye(p, m)
        else:
            self.Z = Z

        if R is None:
            self.R = np.identity(m)
        else:
            self.R = R

        if H is None:
            self.H = np.identity(p)
        else:
            self.H = H

        if Q is None:
            self.Q = np.identity(m)
        else:
            self.Q = Q

    def kfilt(self):
        # matrix parameters
        tt = self.y.shape[0]
        p = self.y.shape[1]
        m = self.a.shape[1]

        # storage matrices
        temp = np.zeros((m, p))
        self.K = np.dstack([temp] * tt)
        temp = np.zeros((p, p))
        self.F = np.dstack([temp] * tt)
        self.v = np.zeros((tt, p))

        # aux variables
        y = self.y
        a = self.a
        P = self.P
        T = self.T
        Z = self.Z
        R = self.R
        H = self.H
        Q = self.Q
        K = self.K
        F = self.F
        v = self.v

        # run kalman filter
        for t in range(0, tt):
            yt = y[t]
            at = a[t]
            Pt = P[:, :, t]
            vt = yt - Z.dot(at)
            Ft = Z.dot(Pt).dot(np.transpose(Z)) + H
            Kt = T.dot(Pt).dot(np.transpose(Z)).dot(np.linalg.inv(Ft))

            # updating equations
            at_t = at + Pt.dot(np.transpose(Z)).dot(np.linalg.inv(Ft)).dot(vt)
            Pt_t = Pt - Pt.dot(np.transpose(Z)).dot(np.linalg.inv(Ft)).dot(Z).dot(Pt)

            # predicting equations
            at_1 = T.dot(at) + Kt.dot(vt)
            Pt_1 = T.dot(Pt).dot(np.transpose(T - Kt.dot(Z))) + R.dot(Q).dot(R)

            # store results
            a[t] = at_t
            P[:, :, t] = Pt_t
            K[:, :, t] = Kt
            F[:, :, t] = Ft
            v[t] = vt
            if t < tt - 1:
                a[t + 1] = at_1
                P[:, :, t + 1] = Pt_1

        # return results
        self.a = a
        self.P = P
        self.K = K
        self.F = F
        self.v = v

    # compute likelihood
    def klike(self):
        v = self.v
        F = self.F
        tt = v.shape[0]
        negloglik = 0
        for t in range(0, tt):
            temp = (-1 / 2) * (v[t].dot(F[:, :, t]).dot(np.transpose(v[t])) + sum(np.log(abs(F[:, :, t]))))
            negloglik = temp + negloglik

        self.loglik = negloglik

    # estimate DFM
    def estimate(self, n=100, progress=True):
        for i in range(1, n):
            self.kfilt()
            #self.ksmooth()
            self.klike()
            if i % 10 == 0 and progress is True:
                print('Iteration {}:'.format(i) + ' ll = {}'.format(self.loglik))
