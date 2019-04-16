"""
=====================================
MTW Handwritten Digits Classification
=====================================

This example performs classification of Handwritten digits using MTW.
Each digit recognition is learned as a sparse regression task. This example
can be used to reproduce the results of (Janati et al., Aistats'19).
"""

import numpy as np
import os
from download import download

from mtw import MTW, utils


print(__doc__)

seed = 42
rnd = np.random.RandomState(seed)

# set n_samples
n_samples = 5
n_features = 240

# take only 3 tasks to run example fast
tasks = [0, 1, 2, 4, 5, 6]
tasks = [0, 6]
n_tasks = len(tasks)
mtgl_only = False
positive = False

"""Download data. The images 'X' are grouped and sorted. Generate true
labels 'Y' accordingly."""
if not os.path.exists('./data'):
    os.mkdir('./data')
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-pix"
path = download(url, ".data/digits.txt", replace=True)
Xraw = np.loadtxt(".data/digits.txt")
Xraw = Xraw.reshape(10, 200, 240)
yraw = np.zeros((10, 2000))
for k in range(10):
    yraw[k, 200 * k: 200 * (k + 1)] = 1.
yraw = yraw.reshape(10, 10, 200)

"""Each digit corresponds to a task. Reshape data to fit a multi-task
learner and split it into a cv and validation set.
Here the design matrix X is the same for all tasks."""

samples = np.arange(200)
samples = rnd.permutation(samples)[:n_samples]
mask_valid = np.ones(200).astype(bool)
mask_valid[samples] = False
ycv = yraw[tasks][:, tasks][:, :, samples].reshape(n_tasks, -1)
yvalid = yraw[tasks][:, tasks][:, :, mask_valid].reshape(n_tasks, -1)
yvalid = np.argmax(yvalid, axis=0)
Xvalid = Xraw[tasks][:, mask_valid].reshape(-1, n_features)
X = Xraw[tasks][:, samples]
X = X.reshape(n_tasks * n_samples, n_features)
scaling = X.std(axis=0)
scaling[scaling == 0] = 1
X = X / scaling
Xcv = np.array(n_tasks * [X])


"""Compute a Euclidean Ground metric M on a 2D grid."""
x = np.arange(16).reshape(-1, 1).astype(float)
y = np.arange(15).reshape(-1, 1).astype(float)

xx, yy = np.meshgrid(x, y)
M1 = abs(xx - yy) ** 2
M = M1[:, np.newaxis, :, np.newaxis] + M1[np.newaxis, :, np.newaxis, :]
M = M.reshape(n_features, n_features) ** 0.5
M_ = M ** 2
M_ /= np.median(M_)

"""Create an MTW instance and fit"""
epsilon = 10. / n_features
betamax = np.array([x.T.dot(y) for x, y in zip(Xcv, ycv)]).max() / n_samples
alpha = 10. / n_samples
beta = 0.35 * betamax
gamma = utils.compute_gamma(0.8, M)
mtw = MTW(M=M_, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma,
          positive=positive, stable=False, tol_ot=1e-5, maxiter_ot=15,
          tol=1e-4, tol_cd=1e-4, maxiter=1000)

mtw.fit(Xcv, ycv)
coefs_ = mtw.coefs_
ypred = np.argmax(Xvalid.dot(coefs_), axis=1)
errors = (ypred != yvalid).reshape(n_tasks, -1).mean(axis=1)
