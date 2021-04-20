
# FILE: main.py
# AUTHOR: Christopher Simard
# DESCRIPTION: Sample file to estimate DFM
# and plot results

# import
import numpy as np
from dfm import *

# simulate data
tt = 200
p = 1
A = np.array([1])
m = A.shape[0]
var = 10
y = np.zeros([tt, p])
for t in range(m, tt):
    temp = np.take(y, range(t-m, t))
    np.flip(A)
    y[t] = temp.dot(A) + np.random.normal(0, var, size=(1, p))

# initialize DFM object
model = DFM(y, m)

# estimate DFM with MLE
model.estimate()