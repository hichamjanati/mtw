from smtr import utils
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from smtr import groundmetric
from smtr.estimators import MTW


def test_mtw_warmstart():

    # Estimator params
    alpha = 10.
    beta_fr = 0.3

    seed = 653
    width, n_tasks = 12, 2
    nnz = 1
    overlap = 0.
    denoising = False
    binary = False
    corr = 0.9

    # Gaussian Noise
    snr = 4
    # Deduce supplementary params
    n_features = width ** 2
    n_samples = n_features // 2

    # ot params
    epsilon = 1. / n_features
    stable = False
    gamma = 10.
    Mbig = utils.groundmetric2d(n_features, p=2, normed=False)
    m = np.median(Mbig)
    M = groundmetric(width, p=2, normed=False)
    M /= m
    # M = Mbig / m

    # Generate Coefs
    coefs = utils.generate_dirac_images(width, n_tasks, nnz=nnz,
                                        seed=seed, overlap=overlap,
                                        binary=binary)
    coefs_flat = coefs.reshape(-1, n_tasks)
    # # Generate X, Y data
    std = 1 / snr
    X, Y = utils.gaussian_design(n_samples, coefs_flat,
                                 corr=corr,
                                 sigma=std,
                                 denoising=denoising,
                                 scaled=True,
                                 seed=seed)

    betamax = np.array([x.T.dot(y) for x, y in zip(X, Y)]).max()
    beta = beta_fr * betamax

    mtw_model = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma,
                    stable=stable, tol_ot=1e-6, tol=1e-6,
                    maxiter_ot=50, maxiter=5000)
    # first fit
    mtw_model.fit(X, Y)
    assert mtw_model.log_['dloss'][-1] < 1e-5

    # small change of hyperparamters
    # mtw_model.beta += 0.01
    mtw_model.alpha += 1

    mtw_model.warmstart = True
    mtw_model.fit(X, Y)
    assert mtw_model.log_['dloss'][-1] < 1e-5
    print(np.min(mtw_model.log_['loss']))
    coefs_warmstart = mtw_model.coefs_

    mtw_model.warmstart = False
    mtw_model.fit(X, Y)
    assert mtw_model.log_['dloss'][-1] < 1e-5
    print(np.min(mtw_model.log_['loss']))
    coefs_no_warmstart = mtw_model.coefs_

    assert_array_almost_equal(coefs_warmstart, coefs_no_warmstart, decimal=3)
