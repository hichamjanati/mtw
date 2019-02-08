"""Unbalanced Optimal transport functions KL div + convolutions.
"""
import numpy as np
import warnings
from . import utils
from ot import emd2


try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


def ot1dkl_(p, q, M, epsilon=0.01, gamma=1., maxiter=20000, tol=1e-2):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_features, n_hists)
        Must be non-negative.
    q: numpy array (n_features, )
        Must be non-negative.
    M: numpy array (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    gamma: float > 0.
        Kullback-Leibler marginal constraint weight w.r.t q.
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    xp = get_module(M)
    n_features = M.shape[0]
    K = xp.exp(- M / epsilon)
    Kb = K.mean(axis=1)

    if p.ndim > 1:
        p = p.reshape(len(p), -1).copy()
        q = q.reshape(len(q), -1).copy()
        Kb = Kb.reshape(-1, 1)
    frac = gamma / (gamma + epsilon)

    a, b = xp.ones((2, n_features))
    log = {'cstr': [], 'obj': [], 'flag': 0, 'objexact': [], 'b': [], 'a': []}
    f0 = K.sum()
    f = f0
    cstr = 10
    for i in range(maxiter):
        a = (p / Kb) ** frac
        Ka = K.T.dot(a)
        b = (q / Ka) ** frac
        oldf = f
        Kb = K.dot(b)
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma)
        cstr = abs(f - oldf) / max(abs(f), abs(oldf), 1)
        log["cstr"].append(cstr)
        log["obj"].append(f)
        if cstr < tol:
            break

    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    if not log['obj']:
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma)
    out = f
    return out, log


def ot1d_(p, q, M, epsilon=0.01, maxiter=20000, tol=1e-2):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_features, n_hists)
        Must be non-negative.
    q: numpy array (n_features, )
        Must be non-negative.
    M: numpy array (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    xp = get_module(M)
    n_features = M.shape[0]
    K = xp.exp(- M / epsilon)
    Kb = K.mean(axis=1)

    if p.ndim > 1:
        p = p.reshape(len(p), -1).copy()
        q = q.reshape(len(q), -1).copy()
        Kb = Kb.reshape(-1, 1)

    a, b = xp.ones((2, n_features))
    log = {'cstr': [], 'obj': [], 'flag': 0, 'objexact': [], 'b': [], 'a': []}
    cstr = 10
    for i in range(maxiter):
        a = p / Kb
        Ka = K.T.dot(a)
        b = q / Ka
        Kb = K.dot(b)
        if i % 10 == 0:
            cstr = abs(a * Kb - p).max()
            log["cstr"].append(cstr)
            if cstr < tol:
                break
    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    f = ((np.log(a + 1e-100) - 1) * p + np.log(b + 1e-100) * q).sum()
    f *= epsilon

    return f, log


def ot1d_log(p, q, M, epsilon=0.01, maxiter=20000, tol=1e-2):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_features, n_hists)
        Must be non-negative.
    q: numpy array (n_features, )
        Must be non-negative.
    M: numpy array (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    xp = get_module(M)
    K = xp.exp(- M / epsilon)
    Kb = np.log(K.mean(axis=1))

    log = {'cstr': [], 'obj': [], 'flag': 0, 'objexact': [], 'b': [], 'a': []}
    cstr = 10
    logp = np.log(p + 1e-100)
    logq = np.log(q + 1e-100)
    for i in range(maxiter):
        a = logp - Kb
        Ka = utils.logsumexp(a[None, :] - M.T / epsilon, axis=1)
        b = logq - Ka
        Kb = utils.logsumexp(b[None, :] - M / epsilon, axis=1)

        if i % 10 == 0:
            cstr = abs(np.exp(a + Kb) - p).max()
            log["cstr"].append(cstr)
            if cstr < tol:
                break
    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    f = ((a - 1) * p + b * q).sum()
    f *= epsilon

    return f, log


def ot1dkl_log(p, q, M, epsilon=0.01, gamma=1., maxiter=20000,
               tol=1e-2):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_features, n_hists)
        Must be non-negative.
    q: numpy array (n_features, )
        Must be non-negative.
    M: numpy array (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    gamma: float > 0.
        Kullback-Leibler marginal constraint weight w.r.t q.
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    xp = get_module(M)
    n_features = M.shape[0]
    p, q, M = p, q, M

    K = xp.exp(- M / epsilon)
    Ks = K.copy()
    Kb = K.sum(axis=1)

    if p.ndim > 1:
        p = p.reshape(len(p), -1).copy()
        q = q.reshape(len(q), -1).copy()
        Kb = Kb.reshape(-1, 1)

    frac = gamma / (gamma + epsilon)

    a, b = xp.ones((2, n_features))
    u, v = xp.zeros((2, n_features))
    log = {'cstr': [], 'obj': [], 'flag': 0, 'objexact': []}
    f0 = K.sum()
    f = f0
    cstr = 10
    for i in range(maxiter):
        a = (p / (Kb + 1e-16)) ** frac * xp.exp(- u / (epsilon + gamma))
        Ka = Ks.T.dot(a)
        b = (q / (Ka + 1e-16)) ** frac * xp.exp(- v / (epsilon + gamma))

        if (a > 1e5).any() or (b > 1e5).any():
            u += epsilon * xp.log(a + 1e-16)
            v += epsilon * xp.log(b + 1e-16)
            Ks = xp.exp((u.reshape(-1, 1) + v.reshape(1, -1) - M) / epsilon)
            b = xp.ones(n_features)
        Kb = Ks.dot(b)

        oldf = f
        f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma, u=u)

        cstr = abs(f - oldf) / max(abs(f), abs(oldf), 1)
        log["cstr"].append(cstr)
        log["obj"].append(f)

        if cstr < tol:
            break
    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        print("Last error: ", cstr)

        log['flag'] = 3

    f = utils.wklobjective(a, Kb, p, q, f0, epsilon, gamma, u=u)

    return f, log


def ot2dkl_log(x, y, M, epsilon=1, gamma=1.,
               maxiter=20000, tol=1e-2, warmstart=None):
    """OT distance for 2D images - log domain."""
    xp = get_module(x)
    width1, width2 = x.shape
    frac = gamma / (gamma + epsilon)
    Kb = warmstart
    x_ = x[:, :, None]
    b = np.zeros_like(x_)

    if warmstart is None:
        Kb = utils.kls(b, - M / epsilon)
    f0 = np.exp(- M / epsilon).sum() ** 2
    log = {'cstr': [], 'flag': 0, 'obj': []}
    logx = xp.log(x + 1e-10)[:, :, None]
    logy = xp.log(y + 1e-10)[:, :, None]

    for i in range(maxiter):
        a = frac * (logx - Kb)
        Ka = utils.kls(a, - M.T / epsilon)
        b = frac * (logy - Ka)
        Kb = utils.kls(b, - M / epsilon)
        if i % 10 == 0:
            cstr = abs(x - np.exp(Kb + a)).max()
            log["cstr"].append(cstr)
            if cstr < tol:
                break

    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        print("Last error: ", cstr)
        log['flag'] = 3
    marginals = xp.exp(a + Kb)
    f = utils.wklobjective_converged(x.flatten(), y.flatten(), f0,
                                     marginals.sum(),
                                     epsilon, gamma)

    # return f, a.flatten(), b.flatten(), log
    return f, log


def ot2d_log(x, y, M, epsilon, maxiter=20000,
             tol=1e-2, warmstart=None):
    """OT distance for 2D images - log domain."""
    xp = get_module(x)
    width1, width2 = x.shape
    Kb = warmstart
    b = np.zeros_like(x)

    if warmstart is None:
        Kb = utils.kls1d(b, - M / epsilon)
    log = {'cstr': [], 'flag': 0, 'obj': []}
    logx = xp.log(x + 1e-10)
    logy = xp.log(y + 1e-10)

    for i in range(maxiter):
        a = logx - Kb
        Ka = utils.kls1d(a, - M.T / epsilon)
        b = logy - Ka
        Kb = utils.kls1d(b, - M / epsilon)
        if i % 10 == 0:
            cstr = abs(x - np.exp(Kb + a)).max()
            log["cstr"].append(cstr)
            if cstr < tol:
                break
    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        print("Last error: ", cstr)

        log['flag'] = 3
    f = ((a - 1) * x + b * y).sum()
    f *= epsilon

    # return f, a.flatten(), b.flatten(), log
    return f, log


def ot2dkl_(x, y, M, epsilon=1, gamma=1.,
            maxiter=20000, tol=1e-2, warmstart=None):
    """OT distance for 2D images."""
    width1, width2 = x.shape
    frac = gamma / (gamma + epsilon)
    Kb = warmstart
    b = np.ones_like(x)
    K = np.exp(- M / epsilon)
    if warmstart is None:
        Kb = utils.klconv1d(b, K)
    f0 = K.sum() ** 2
    log = {'cstr': [], 'flag': 0, 'obj': []}

    for i in range(maxiter):
        a = (x / Kb) ** frac
        Ka = utils.klconv1d(a, K.T)
        b = (y / Ka) ** frac
        Kb = utils.klconv1d(b, K)
        if i % 10 == 0:
            cstr = abs(a ** (1 / frac) * Kb - x).max()
            log["cstr"].append(cstr)
            if cstr < tol:
                break

    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        print("Last error: ", cstr)

        log['flag'] = 3
    marginals = a * Kb
    f = utils.wklobjective_converged(x.flatten(), y.flatten(), f0,
                                     marginals.sum(),
                                     epsilon, gamma)

    # return f, a.flatten(), b.flatten(), log
    return f, log


def ot2d_(x, y, M, epsilon, maxiter=20000,
          tol=1e-2, warmstart=None):
    """OT distance for 2D images."""
    width1, width2 = x.shape
    Kb = warmstart
    b = np.ones_like(x)
    K = np.exp(- M / epsilon)
    if warmstart is None:
        Kb = utils.klconv1d(b, K)
    log = {'cstr': [], 'flag': 0, 'obj': []}

    for i in range(maxiter):
        a = x / Kb
        Ka = utils.klconv1d(a, K.T)
        b = y / Ka
        Kb = utils.klconv1d(b, K)
        if i % 10 == 0:
            cstr = abs(a * Kb - x).max()
            log["cstr"].append(cstr)
            if cstr < tol:
                break

    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        print("Last error: ", cstr)

        log['flag'] = 3

    f = ((np.log(a + 1e-100) - 1) * x + np.log(b + 1e-100) * y).sum()
    f *= epsilon

    # return f, a.flatten(), b.flatten(), log
    return f, log


def ot2d(x, y, M, epsilon, log=True, **kwargs):
    """General OT 2d function."""
    if log:
        ot = ot2d_log
    else:
        ot = ot2d_
    output = ot(x, y, M, epsilon, **kwargs)

    return output


def ot1d(x, y, M, epsilon, log=True, **kwargs):
    """General OT 2d function."""
    if log:
        ot = ot1d_log
    else:
        ot = ot1d_
    output = ot(x, y, M, epsilon, **kwargs)

    return output


def ot1dkl(x, y, M, epsilon, log=True, **kwargs):
    """General OT 2d function."""
    if log:
        ot = ot1dkl_log
    else:
        ot = ot1dkl_
    output = ot(x, y, M, epsilon, **kwargs)

    return output


def ot2dkl(x, y, M, epsilon, log=True, **kwargs):
    """General OT 2d function."""
    if log:
        ot = ot2dkl_log
    else:
        ot = ot2dkl_
    output = ot(x, y, M, epsilon, **kwargs)

    return output


def ot_amari(x, y, M, epsilon, gamma=0., wyy0=None, normed=True, **kwargs):
    """OT metric for 2D images."""
    assert x.shape == y.shape
    if x.ndim == 3:
        axis = (0, 1)
    else:
        axis = 0
    if wyy0 is None:
        wyy = 0.
    else:
        wyy = wyy0
    if gamma == 0.:
        x = x / x.sum(axis=axis)
        y = y / y.sum(axis=axis)

        if x.ndim == 3:
            ot_dist = ot2d
        else:
            ot_dist = ot1d
    else:
        if x.ndim == 3:
            ot_dist = ot2dkl
        else:
            ot_dist = ot1dkl

    wxy, wxx = 0., 0.
    for a, b in zip(x.T + 1e-8, y.T + 1e-8):
        wxy += ot_dist(a.T, b.T, M, epsilon, **kwargs)[0]
        if np.isnan(wxy):
            break
        if normed:
            wxx += ot_dist(a.T, a.T, M, epsilon, **kwargs)[0]
            if wyy0 is None:
                wyy += ot_dist(b.T, b.T, M, epsilon, **kwargs)[0]

    f = wxy - (wxx + wyy) / 2

    return f


def emd(x, y, M):
    n_tasks = x.shape[-1]
    assert x.shape == y.shape
    assert len(x) == len(M)
    if not y.max(0).all():
        return 1e100
    x = x / x.sum(axis=0)
    y = y / y.sum(axis=0)
    f = 0.
    for a, b in zip(x.T, y.T):
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        f += emd2(a, b, M)
    return f / n_tasks
