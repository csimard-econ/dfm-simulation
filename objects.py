# state-space model class
#   y_t = Z_t*y_t + eps_t
#   x_t = T_t*x_{t-1} + R_t*u_t

# preliminaries
import numpy as np
import pandas as pd


# define state-space model class
class SSM:
    def __init__(self, Z, T, R, H, Q, mu0, sigma0, y):
        # parameters
        self.Z = np.array(Z)
        self.T = np.array(T)
        self.R = np.array(R)
        self.H = np.array(H)
        self.Q = np.array(Q)
        self.mu0 = np.array(mu0)
        self.sigma0 = np.array(sigma0)
        self.y = np.array(y)

        # matrix dimensions
        self.m = self.mu0.shape[0]
        self.n = self.y.shape[0]
        self.tt = self.y.shape[1]

    # draws (x, y) ~ p(x, y)
    def simulate_SSM(self):
        # matrix size parameters
        m = self.m
        n = self.n
        tt = self.tt

        # noise vectors
        u = np.random.normal(0, 1, size=(m, tt))
        eps = np.random.normal(0, 1, size=(n, tt))

        # simulate initial draw from state
        x_t = self.mu0 + np.dot(np.sqrt(self.sigma0), np.random.normal(0, 1, size=(m, 1)))

        # simulate SSM
        y_plus = np.empty(shape=(n, tt))
        x_plus = np.empty(shape=(m, tt))
        for t in range(0, tt):
            # check for time-varying parameters
            Z_t = self.Z[:, :, t] if self.Z.ndim > 2 else self.Z
            T_t = self.T[:, :, t] if self.T.ndim > 2 else self.T
            R_t = self.R[:, :, t] if self.R.ndim > 2 else self.R
            H_t = self.H[:, :, t] if self.H.ndim > 2 else self.H
            Q_t = self.Q[:, :, t] if self.Q.ndim > 2 else self.Q
            u_t = np.sqrt(Q_t)*u[:, t]
            eps_t = np.sqrt(H_t).dot(eps[:, t])

            # iterate one step
            y_t = Z_t.dot(x_t) + eps_t.reshape((-1, 1))
            x_t = T_t.dot(x_t) + R_t.dot(u_t)

            # save results
            y_plus[:, t] = y_t.T
            x_plus[:, t] = x_t.T

        # return draw
        return x_plus, y_plus

    # draws x ~ p(x|y)
    def simulation_smoother(self):
        # draw from state-space model
        x_plus, y_plus = self.simulate_SSM()

        # create artificial observations
        y_star = self.y - y_plus

        # matrix size parameters
        m = self.m
        n = self.n
        tt = self.tt

        # initialize state variables
        x_t = self.mu0
        P_t = self.sigma0

        # initial forward recursion
        x_hat = np.zeros(shape=(m, tt))
        v = np.zeros(shape=(n, tt))
        F = np.zeros(shape=(n, n, tt))
        L = np.zeros(shape=(m, m, tt))
        P = np.zeros(shape=(m, m, tt))
        for t in range(0, tt):
            # check for time-varying parameters
            Z_t = self.Z[:, :, t] if self.Z.ndim > 2 else self.Z
            T_t = self.T[:, :, t] if self.T.ndim > 2 else self.T
            R_t = self.R[:, :, t] if self.R.ndim > 2 else self.R
            H_t = self.H[:, :, t] if self.H.ndim > 2 else self.H
            Q_t = self.Q[:, :, t] if self.Q.ndim > 2 else self.Q
            y_star_t = y_star[:, t]

            # filtering equations
            v_t = y_star_t.reshape((-1, 1)) - Z_t.dot(x_t)
            F_t = Z_t.dot(P_t).dot(Z_t.T) + H_t
            K_t = T_t.dot(P_t).dot(Z_t.T).dot(np.linalg.inv(F_t))
            L_t = T_t.reshape((-1, 1)) - K_t.dot(Z_t)
            x_t = T_t.dot(x_t) + K_t.dot(v_t)
            P_t = T_t.dot(P_t).dot(L_t.T) + R_t.dot(Q_t).dot(R_t.T)

            # save outputs
            x_hat[:, t] = x_t.T
            v[:, t] = v_t.T
            F[:, :, t] = F_t.T
            L[:, :, t] = L_t.T
            P[:, :, t] = P_t.T

        # backward recursion
        r = np.zeros(shape=(m, tt))
        r_t = np.zeros(shape=(m, 1))
        for t in range(tt - 1, 0, -1):
            # check for time-varying parameters
            Z_t = self.Z[:, :, t] if self.Z.ndim > 2 else self.Z
            F_t = F[:, :, t] if F.ndim > 2 else F
            L_t = L[:, :, t] if L.ndim > 2 else L
            v_t = v[:, t]

            # iterate backward
            r[:, t] = r_t.T
            r_t = Z_t.transpose().dot(np.linalg.inv(F_t)).dot(v_t) + L_t.transpose().dot(r_t)

        # final forward recursion
        x_hat_star = np.zeros(shape=(m, tt))
        x_hat_star_t = self.mu0.reshape((-1, 1)) + self.sigma0.dot(r[:, 0])
        x_hat_star[:, 0] = x_hat_star_t
        for t in range(1, tt):
            # check for time-varying parameters
            T_t = self.T[:, :, t] if self.T.ndim > 2 else self.T
            R_t = self.R[:, :, t] if self.R.ndim > 2 else self.R
            Q_t = self.Q[:, :, t] if self.Q.ndim > 2 else self.Q
            r_t = r[:, t]

            # iterate forward
            x_hat_star_t = T_t.dot(x_hat_star_t) + R_t.dot(Q_t).dot(R_t.transpose()).dot(r_t)
            x_hat_star[:, t] = x_hat_star_t.T

        # return draw from state
        x_tilde = x_hat_star + x_plus
        return x_tilde