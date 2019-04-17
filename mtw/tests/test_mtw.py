from mtw.examples_utils import generate_dirac_images, gaussian_design
import numpy as np
from numpy.testing import assert_allclose
from mtw import MTW, utils
import pytest
from itertools import product


@pytest.mark.parametrize("positive, alpha, epsilon",
                         product([False, True], [10., 0.], [5., 0.5]))
def test_mtw_convolution(positive, alpha, epsilon):

    # Estimator params
    seed = 42
    width, n_tasks = 12, 2
    nnz = 2
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
    epsilon = epsilon / n_features
    stable = False
    Mbig = utils.groundmetric2d(n_features, p=2, normed=False)
    m = np.median(Mbig)
    M = utils.groundmetric(width, p=2, normed=True)
    M = utils.groundmetric(width, p=2, normed=False)
    M /= m
    # M = Mbig / m

    # Generate Coefs
    coefs = generate_dirac_images(width, n_tasks, nnz=nnz,
                                  seed=seed, overlap=overlap,
                                  binary=binary,
                                  positive=positive)
    coefs_flat = coefs.reshape(-1, n_tasks)
    # # Generate X, Y data
    std = 1 / snr
    X, Y = gaussian_design(n_samples, coefs_flat,
                           corr=corr, sigma=std,
                           denoising=denoising,
                           scaled=True, seed=seed)

    betamax = np.array([abs(x.T.dot(y)) for x, y in zip(X, Y)]).max()
    beta_fr = 0.5
    beta = beta_fr * betamax / n_samples
    alpha = alpha / n_samples

    callback_options = {'callback': True,
                        'x_real': coefs_flat.reshape(- 1, n_tasks),
                        'verbose': True, 'rate': 1, 'prc_only': False}
    gamma = utils.compute_gamma(0.8, M)

    """Fit mtw_model using convolutions to compute OT barycenters."""
    mtw_model = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon,
                    gamma=gamma, stable=stable, tol_ot=1e-8, tol=1e-5,
                    maxiter_ot=30, maxiter=200, **callback_options,
                    positive=positive)
    # first fit
    mtw_model.fit(X, Y)
    assert mtw_model.log_['dloss'][-1] < 1e-4

    M = Mbig / m
    # mtw_model using standard sinkhorn
    mtw_model2 = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon,
                     gamma=gamma, stable=stable, tol_ot=1e-8, tol=1e-5,
                     maxiter_ot=30, maxiter=200, **callback_options,
                     positive=positive)
    mtw_model2.fit(X, Y)
    assert mtw_model2.log_['dloss'][-1] < 1e-4
    assert_allclose(mtw_model.coefs_, mtw_model2.coefs_, atol=1e-5, rtol=1e-5)

    # get positive / negative parts
    coefs1, coefs2 = utils.get_unsigned(mtw_model.coefs_)
