"""Unbalanced Optimal transport functions KL div + convolutions.
"""
import numpy as np
import warnings
from . import utils

try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


def barycenterkl_log(P, M, epsilon, gamma, b=None, tol=1e-4,
                     maxiter=1000):
    """KL OT Barycenters for 2D images."""
    xp = get_module(M)
    frac = gamma / (gamma + epsilon)
    n_tasks = P.shape[-1]
    n_features = M.shape[-1]
    psum = P.sum()
    support = (P > 1e-8).any(axis=1)
    logps = np.log(P[support] + 1e-100)
    logps = xp.asarray(logps)
    M = M[xp.asarray(support)]
    utils.free_gpu_memory(xp)

    if b is None:
        Kb = utils.logsumexp(M, axis=1)
        Kb = xp.tile(Kb, (n_tasks, 1)).T

    else:
        Kb = xp.zeros((len(M), n_tasks))
        for k in range(n_tasks):
            Kb[:, k] = utils.logsumexp(b[:, k][None, :] + M, axis=1)
            utils.free_gpu_memory(xp)
    log = {'cstr': [], 'flag': 0, 'obj': []}
    weights = xp.ones(n_tasks) / n_tasks
    logweights = xp.log(weights)[None, :]
    qold = xp.ones(n_features)[:, None]
    Ka = xp.zeros((n_features, n_tasks))
    for i in range(maxiter):
        a = frac * (logps - Kb)  # it's actually a
        for k in range(n_tasks):
            Ka[:, k] = utils.logsumexp(a[:, k][:, None] + M, axis=0)
            utils.free_gpu_memory(xp)
        logq = logweights + Ka * (1 - frac)  # it's weighted ka
        logq = utils.logsumexp(logq, axis=1)  # this is q in log
        logq = (1 / (1 - frac)) * logq
        b = frac * (logq[:, None] - Ka)
        for k in range(n_tasks):
            Kb[:, k] = utils.logsumexp(b[:, k][None, :] + M, axis=1)
            utils.free_gpu_memory(xp)
        q = xp.exp(logq)
        if i % 5 == 0:
            cstr = float(abs(q - qold).max())
            cstr /= float(max(q.max(), qold.max(), 1.))
        qold = q.copy()
        log["cstr"].append(cstr)
        # uncomment to compute exact loss
        # marginals = np.exp(a + Kb).T
        # f = - (2 * gamma + epsilon) * marginals.sum()
        # f += gamma * (psum + q.sum() * n_tasks)
        # f += (((gamma + epsilon) * a + gamma * (Kb - logps)) *
        # marginals.T).sum()
        # log["obj"].append(f)
        if cstr < tol and i > 3:
            break
        # f = utils.wklobjective_log(a, Kb, P, q, K0, epsilon, gamma)
        # log["obj"].append(f)

    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = 3

    try:
        a = a.get()
        Kb = Kb.get()
        q = q.get()
        logps = logps.get()
        utils.free_gpu_memory(xp)

    except AttributeError:
        pass
    marginals = np.exp(a + Kb).T
    marginals[~np.isfinite(marginals)] = 1

    m = np.zeros((n_tasks, n_features))
    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    # uncomment to compute exact loss
    # f = - (2 * gamma + epsilon) * marginals.sum()
    # f += gamma * (psum + q.sum() * n_tasks)
    # f += (((gamma + epsilon) * a + gamma * (Kb - logps)) * marginals.T).sum()
    # log["obj"].append(f)
    m[:, support] = marginals
    marginals = m
    b[~np.isfinite(b)] = 0.

    return f, log, marginals, b, q


def barycenterkl(P, M, epsilon, gamma, b=None, tol=1e-4,
                 maxiter=1000):
    """Compute Unblanced Wasserstein barycenter.
    """
    xp = get_module(M)
    frac = gamma / (gamma + epsilon)
    psum = P.sum()
    n_features, n_tasks = P.shape
    frac = gamma / (gamma + epsilon)
    support = (P > 1e-8).any(axis=1)
    if len(support) == 0:
        support = P.any(axis=1)
    P = P[support]
    P = xp.asarray(P)
    M = M[xp.asarray(support)]
    M = xp.exp(M)
    if b is None:
        b = xp.ones((n_features, n_tasks))
    Kb = M.dot(b)

    log = {'cstr': [], 'flag': 0, 'obj': []}
    weights = xp.ones(n_tasks) / n_tasks
    q = xp.ones(n_features)
    qold = q.copy()
    return_nan = False
    utils.free_gpu_memory(xp)

    for i in range(maxiter):
        a = (P / Kb) ** frac
        utils.free_gpu_memory(xp)

        Ka = M.T.dot(a)
        q = ((Ka ** (1 - frac)).dot(weights))
        q = q ** (1 / (1 - frac))
        Q = q[:, None]
        utils.free_gpu_memory(xp)
        if i % 5 == 0:
            cstr = float(abs(q - qold).max() / max(q.max(), qold.max(), 1.))
        qold = q.copy()
        b_old = b.copy()
        b = (Q / Ka) ** frac
        utils.free_gpu_memory(xp)

        if not xp.isfinite(b).all():
            return_nan = True
            break
        Kb = M.dot(b)
        log["cstr"].append(cstr)
        if abs(cstr) < tol and i > 3:
            break
        # marginals = (a * Kb).T
        utils.free_gpu_memory(xp)
        # uncomment to compute exact loss
        # f = - (2 * gamma + epsilon) * marginals.sum()
        # f += gamma * (psum + q.sum() * n_tasks)
        # f += (((gamma + epsilon) * np.log(a + 1e-100) + gamma *
        #      (np.log(Kb + 1e-100) - np.log(P + 1e-100))) * marginals.T).sum()
        # log["obj"].append(f)
    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = - 1
    marginals = (a * Kb).T

    try:
        marginals = marginals.get()
        q = q.get()
        utils.free_gpu_memory(xp)
        P = P.get()
        Kb = Kb.get()
        a = a.get()

    except AttributeError:
        pass

    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    m = np.zeros((n_tasks, n_features))
    marginals[~np.isfinite(marginals)] = 1
    m[:, support] = marginals
    marginals = m
    if return_nan or xp.isnan(f):
        f = None
        b = b_old
    # else:
    #     uncomment to compute exact loss
    #     f = - (2 * gamma + epsilon) * marginals.sum()
    #     f += gamma * (psum + q.sum() * n_tasks)
    #     xx = ((Kb / P) ** gamma) * (a ** (epsilon + gamma))
    #     xx[xx == 0.] = 0.
    #     f += (np.log(xx) * marginals.T).sum()
    #     log["obj"].append(f)
    return f, log, marginals, b, q


def barycenterkl_img_log(P, M, epsilon, gamma, b=None, tol=1e-4,
                         maxiter=1000):
    """KL OT Barycenters for 2D images."""
    xp = get_module(M)
    psum = P.sum()
    P = P.reshape(xp.r_[M.shape, -1])
    n_tasks = P.shape[-1]
    n_features = P.size // n_tasks
    frac = gamma / (gamma + epsilon)
    if b is None:
        b = xp.zeros_like(P)
    Kb = utils.kls(b, M)
    log = {'cstr': [], 'flag': 0, 'obj': []}
    weights = xp.ones(n_tasks) / n_tasks
    logweights = xp.log(weights)[None, None, :]
    logp = xp.log(P + 1e-10)
    qold = P.mean(axis=-1) + 1e-10
    for i in range(maxiter):
        a = frac * (logp - Kb)
        Ka = utils.kls(a, M.T)
        kaw = logweights + (Ka) * (1 - frac)
        logq = utils.logsumexp(kaw, axis=-1) - xp.log(weights.sum())
        logq = (1 / (1 - frac)) * logq
        logQ = logq[:, :, xp.newaxis]
        b = frac * (logQ - Ka)
        Kb = utils.kls(b, M)
        q = xp.exp(logq)

        if i % 10 == 0:
            cstr = float((abs(q - qold)).max())
            cstr /= float(max(q.max(), qold.max(), 1.))
        qold = q.copy()

        log["cstr"].append(cstr)
        if cstr < tol and i > 5:
            break

        # f = utils.wklobjective_log(a, Kb, P, q, 0, epsilon, gamma)
        # log["obj"].append(f)
        utils.free_gpu_memory(xp)

    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = 3

    marginals = xp.exp(a + Kb).reshape(n_features, n_tasks).T
    try:
        marginals = marginals.get()
        q = q.get()
    except AttributeError:
        pass

    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    # f = utils.wklobjective_log(a, Kb, P, q, 0, epsilon, gamma)

    return f, log, marginals, b, q


def barycenterkl_img(P, M, epsilon, gamma, b=None, tol=1e-4,
                     maxiter=1000):
    """KL OT Barycenters for 2D images."""
    xp = get_module(M)
    psum = P.sum()
    P = P.reshape(xp.r_[M.shape, -1])
    frac = gamma / (gamma + epsilon)
    n_tasks = P.shape[-1]
    if b is None:
        b = xp.ones_like(P)
    M = xp.exp(M)
    Kb = utils.klconv1d_list(b, M)

    log = {'cstr': [], 'flag': 0, 'obj': []}
    weights = xp.ones(n_tasks) / n_tasks
    # qold = P.mean(axis=-1)
    return_nan = False
    margs_old = P.copy()

    for i in range(maxiter):
        a = (P / Kb) ** frac
        Ka = utils.klconv1d_list(a, M.T)
        q = ((Ka) ** (1 - frac)).dot(weights)
        q = (q / (weights.sum())) ** (1 / (1 - frac))
        Q = q[:, :, xp.newaxis]
        b_old = b.copy()
        b = (Q / Ka) ** frac

        Kb = utils.klconv1d_list(b, M)
        margs = a * Kb
        if not xp.isfinite(margs).all():
            return_nan = True
            break

        if i % 10 == 0:
            cstr = float((abs(margs - margs_old)).max())
            cstr /= float(max(margs.max(), margs_old.max(), 1.))
        margs_old = margs.copy()

        log["cstr"].append(cstr)
        if cstr < tol:
            break
        utils.free_gpu_memory(xp)
    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = 3

    marginals = margs.reshape(- 1, n_tasks).T
    try:
        marginals = marginals.get()
        q = q.get()
    except AttributeError:
        pass
    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    # f = utils.wklobjective(a, Kb, P, q, 0, epsilon, gamma)

    if return_nan:
        f = None
        b = b_old
    return f, log, marginals, b, q
