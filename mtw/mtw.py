import numpy as np
from .solver import solver
from .examples_utils import inspector_mtw
try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


class NotFittedError(AttributeError):
    """Raised if an estimator is used before fitting."""


class MTW:
    """A class for MultiTask Regression with Wasserstein penalization.

    Attributes
    ----------


    """
    def __init__(self, M, alpha=1., beta=0., epsilon=0.1, gamma=1.,
                 sigma0=0., stable=True, callback=False, maxiter_ot=1000,
                 maxiter=4000, tol_ot=1e-5, tol_cd=1e-4, tol=1e-5,
                 positive=False, n_jobs=1, gpu=False,
                 **kwargs):
        """Constructs instance of MTW.

        Parameters
        ----------
        M: array, shape (n_features, n_features)
            Ground metric matrix defining the Wasserstein distance.
        alpha: float >= 0.
            hyperparameter of the Wasserstein penalty.
        beta : float >= 0.
            hyperparameter of the l1 penalty.
        epsilon: float > 0.
            OT parameter. Weight of the entropy regularization.
        gamma: float > 0.
            OT parameter. Weight of the Kullback-Leibler marginal relaxation.
        sigma0: float >=0.
            Lower bound of the noise standard deviation. If positive, the l1
            penalty is adaptively scaled to the noise std estimation
            (corresponds to concomitant MTW, or MWE algorithm). If 0, noise std
            are not inferred (classic MTW).
        stable: boolean. optional (default False)
            if True, use log-domain Sinhorn stabilization from the first iter.
            if False, the solver will automatically switch to log-domain if
            numerical errors are encountered.
        callback: boolean. optional.
            if True, set a printing callback function to the solver.
        maxiter_ot: int > 0
            maximum Sinkhorn iterations
        maxiter_cd: int > 0
            maximum coordinate descent iterations
        maxiter: int > 0
            maximum outer loop iterations
        tol_ot: float >=0.
            relative maximum change of the Wasserstein barycenter.
        tol_cd: float >=0.
            relative maximum change of the coefficients in coordinate descent.
        tol: float >=0.
            relative maximum change of the coeffcients in the outer loop.
        positive: boolean.
            if True, coefficients must be positive.
        n_jobs: int > 1.
            number of threads used in coordinate descents
        gpu: boolean.
            if True, Sinkhorn iterations are performed on gpus using cupy.
        """
        self.callback = callback
        self.callback_kwargs = kwargs
        self.n_jobs = n_jobs
        self.wyy0 = 0.
        self.M = M
        self.xp = get_module(M)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.stable = stable
        self.maxiter_ot = maxiter_ot
        self.tol_ot = tol_ot
        self.tol = tol
        self.maxiter = maxiter
        self.positive = positive
        self.n_jobs = n_jobs
        self.tol_cd = tol_cd
        self.sigma0 = sigma0
        self.gpu = gpu
        self._set_callback()

    def _set_callback(self):
        """Set callback if `callback` is True."""
        self.callback_f = None
        if self.callback:
            self.callback_f = inspector_mtw(**self.callback_kwargs)

    def fit(self, X, Y, verbose=True):
        """Launch MTW solver.

        Parameters
        ----------
        X: numpy array (n_tasks, n_samples, n_features).
            Regression data.
        Y: numpy arrays (n_tasks, n_samples,).
            Target data.

        Returns
        -------
        instance of self.
        """
        self.t_ot = 0.
        self.t_cd = 0.

        coefs1, coefs2, bar1, bar2, log, sigmas = \
            solver(X, Y, M=self.M, alpha=self.alpha, beta=self.beta,
                   epsilon=self.epsilon, gamma=self.gamma, sigma0=self.sigma0,
                   stable=self.stable, tol=self.tol, callback=self.callback_f,
                   maxiter=self.maxiter, tol_ot=self.tol_ot,
                   maxiter_ot=self.maxiter_ot, positive=self.positive,
                   n_jobs=self.n_jobs, tol_cd=self.tol_cd, gpu=self.gpu,
                   verbose=verbose)
        self.coefs1_ = coefs1.copy()
        self.coefs2_ = coefs2.copy()
        self.coefs_ = coefs1 - coefs2
        self.sigmas_ = sigmas
        self.barycenter1_ = bar1
        self.barycenter2_ = bar2
        self.barycenter_ = bar1 - bar2

        self.log_ = log
        self.t_ot += log["t_ot"]
        self.t_cd += log["t_cd"]

        return self
