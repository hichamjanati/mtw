"""Solvers for multitask group-stl with a simplex constraint."""
import numpy as np
import warnings
from joblib import Parallel, delayed
from time import time

from . import utils
from .solver_cd import cython_wrapper
from .otfunctions import (barycenterkl, barycenterkl_img, barycenterkl_log,
                          barycenterkl_img_log)
try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


def solver(X, Y, M, alpha=1., beta=0., epsilon=0.01, gamma=1., sigma0=0.,
           stable=False, maxiter=2000, callback=None, tol=1e-5,
           maxiter_ot=10000, tol_ot=1e-4, positive=False, n_jobs=1,
           tol_cd=1e-4, gpu=True, verbose=True):
    """Perform Alternating Optimization of concomittant MTW."""

    log = {'loss': [], 'dloss': [], 'log_sinkhorn1': [], 'log_sinkhorn2': [],
           'stable': stable, 't_cd': 0., 't_ot': 0., "times": [], "objcd": [],
           "fot1": [], "fot2": []}

    n_tasks, n_samples, n_features = X.shape
    coefs01 = np.ones((n_features, n_tasks))
    coefs02 = np.ones((n_features, n_tasks))
    marginals1 = np.ones((n_tasks, n_features)) / n_features
    marginals2 = np.ones((n_tasks, n_features)) / n_features

    Xf = [np.asfortranarray(X[k]) for k in range(n_tasks)]
    mXf = [- np.asfortranarray(X[k]) for k in range(n_tasks)]

    theta1 = coefs01.copy()
    theta2 = coefs02.copy()
    theta = theta1 - theta2

    thetaold = theta.copy()
    Ls = (X ** 2).sum(axis=1)
    Ls[Ls == 0.] = Ls[Ls != 0].min()

    ot_img = True
    if len(M) == n_features:
        ot_img = False
    # if ot_img, fast convolutions will be used in Sinkhorn

    update_ot_1 = set_ot_func(stable, ot_img)
    update_ot_2 = set_ot_func(stable, ot_img)

    t_cd = []
    t_ot = 0.
    xp = utils.set_module(gpu)
    M = xp.asarray(- M / epsilon)

    thetabar1 = np.ones_like(coefs01).mean(axis=-1)
    thetabar2 = np.ones_like(coefs02).mean(axis=-1)

    if positive:
        theta2 *= 0.
        thetabar2 *= 0.
        theta = theta1.copy()
    n_tasks = len(X)
    a = n_samples * alpha * gamma
    b = n_samples * beta
    if alpha == 0.:
        theta1 *= 0.
        theta2 *= 0.
    if sigma0:
        sigma0 *= np.linalg.norm(Y, axis=1).min() / (n_samples ** 0.5)
    sigmas = np.ones(n_tasks)
    R, b1, b2 = None, None, None
    with Parallel(n_jobs=n_jobs, backend="threading") as pll:
        if alpha == 0.:
            t = time()
            theta1 = np.asfortranarray(theta1)
            theta, R, sigmas = update_coefs(pll, Xf, Y, Ls, marginals1,
                                            sigmas,
                                            a, b,
                                            sigma0,
                                            coefs0=theta1,
                                            R=R,
                                            tol=tol_cd, maxiter=20000,
                                            positive=positive)
            obj = 0.5 * (R ** 2).sum(axis=1).dot(1 / sigmas) / n_samples
            obj += beta * abs(theta).sum() + 0.5 * sigmas.sum()
            theta1, theta2 = utils.get_unsigned(theta)
            t_cd.append(time() - t)
            log["t_cd"] = sum(t_cd)
            log['loss'].append(obj)
            log['dloss'].append(0)

            if callback:
                callback(theta, v=obj)

            return theta1, theta2, thetabar1, thetabar2, log, sigmas
        for i in range(maxiter):
            t0 = time()
            if not positive:
                Y1 = utils.residual(X, - theta2, Y)
            else:
                Y1 = Y
            theta1, R, sigmas = update_coefs(pll, X, Y1, Ls, marginals1,
                                             sigmas,
                                             a, b,
                                             sigma0,
                                             coefs0=theta1,
                                             R=R,
                                             tol=tol_cd, maxiter=10000)
            if not positive:
                Y2 = utils.residual(X, theta1, Y)
                theta2, R, sigmas, = update_coefs(pll, mXf, Y2, Ls,
                                                  marginals2,
                                                  sigmas,
                                                  a, b,
                                                  sigma0,
                                                  coefs0=theta2,
                                                  R=R,
                                                  tol=tol_cd,
                                                  maxiter=10000)
                theta = theta1 - theta2
            else:
                theta = theta1.copy()

            obj = 0.5 * (R ** 2).sum(axis=1).dot(1 / sigmas) / n_samples
            obj += beta * (theta1 + theta2).sum() + 0.5 * sigmas.sum()
            log["objcd"].append(obj)
            tc = time() - t0
            t_cd.append(tc)
            dx = abs(theta - thetaold).max() / max(1, thetaold.max(),
                                                   theta.max())

            thetaold = theta.copy()

            t1 = time()
            if alpha:
                if (theta1 > 1e-10).any(0).all():
                    fot1, log_ot1, marginals1, b1, q1 = \
                        update_ot_1(theta1, M, epsilon, gamma,
                                    b=b1, tol=tol_ot, maxiter=maxiter_ot)
                    if fot1 is None or not theta1.max(0).all():
                        warnings.warn("""Nan found when computing barycenter,
                                         re-fit in log-domain.""")
                        b1 = xp.log(b1 + 1e-100, out=b1)
                        stable = True
                        update_ot_1 = set_ot_func(True, ot_img)
                        fot1, log_ot1, marginals1, b1, q1 = \
                            update_ot_1(theta1, M, epsilon, gamma, b=b1,
                                        tol=tol_ot, maxiter=maxiter_ot)
                        utils.free_gpu_memory(xp)

                    log["log_sinkhorn1"].append(log_ot1)
                    thetabar1 = q1
                    log["fot1"].append(fot1)
                    obj += alpha * fot1
                if not positive:
                    if (theta2 > 1e-10).any(0).all():
                        fot2, log_ot2, marginals2, b2, q2 = \
                            update_ot_2(theta2, M, epsilon, gamma,
                                        b=b2, tol=tol_ot, maxiter=maxiter_ot)
                        utils.free_gpu_memory(xp)

                        if fot2 is None or not theta2.max(0).all():
                            warnings.warn("""Nan found in negative, re-fit in
                                             log-domain.""")
                            b2 = xp.log(b2 + 1e-100, out=b2)
                            stable = True
                            update_ot_2 = set_ot_func(True, ot_img)
                            fot2, log_ot2, marginals2, b2, q2 = \
                                update_ot_2(theta2, M, epsilon, gamma,
                                            b=b2, tol=tol_ot,
                                            maxiter=maxiter_ot)
                            utils.free_gpu_memory(xp)

                        log["log_sinkhorn2"].append(log_ot2)
                        thetabar2 = q2
                        log["fot2"].append(fot2)
                        obj += alpha * fot2

            t_ot += time() - t1
            if callback:
                callback(theta, v=obj)

            log['loss'].append(obj)
            log['dloss'].append(dx)
            log['times'].append(time() - t0)

            if dx < tol:
                break
    if i == maxiter - 1:
        print("\n"
              "******** WARNING: Stopped early in main loop. *****\n"
              "\n"
              "You may want to increase mtw.maxiter.")

    if verbose:
        print("Time ot %.1f | Time cd %.1f" % (t_ot, sum(t_cd)))

    log['stable'] = stable
    log['t_cd'] = sum(t_cd)
    log['t_ot'] = t_ot

    if positive:
        theta2 *= 0.
        thetabar2 = xp.zeros_like(thetabar1)
        try:
            thetabar2 = thetabar2.get()
        except AttributeError:
            pass
    return theta1, theta2, thetabar1, thetabar2, log, sigmas


def set_ot_func(stable, ot_img):
    """Set barycenter function."""
    if stable:
        update_ot = barycenterkl_log
    else:
        update_ot = barycenterkl

    if ot_img:
        if stable:
            update_ot = barycenterkl_img_log
        else:
            update_ot = barycenterkl_img
    else:
        if stable:
            update_ot = barycenterkl_log
        else:
            update_ot = barycenterkl
    return update_ot


def update_coefs(pll, X, y, Ls, marginals, sigmas, a, b, sigma0,
                 coefs0=None, R=None, maxiter=20000, tol=1e-4,
                 positive=False):
    """BCD in numba."""
    n_tasks, n_samples = y.shape
    n_features = Ls.shape[1]

    if coefs0 is None:
        R = y.copy()
        coefs0 = np.zeros((n_features, n_tasks))
    elif R is None:
        R = utils.residual(X, coefs0, y)

    dell = delayed(cython_wrapper)
    it = (dell(n_samples, n_features, np.asfortranarray(X[k].copy()), y[k],
               Ls[k], marginals[k], coefs0[:, k], R[k].copy(),
               sigmas[k:k + 1], a, b, sigma0, tol, maxiter, positive)
          for k in range(n_tasks))
    output = pll(it)

    thetas, R, sigmas = list(zip(*output))
    thetas = np.stack(thetas, axis=1)
    R = np.stack(R, axis=0)

    sigmas = np.r_[sigmas]
    return thetas, R, sigmas
