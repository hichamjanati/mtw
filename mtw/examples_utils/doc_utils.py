import numpy as np
import warnings

from sklearn.metrics import average_precision_score

try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


def set_module(gpu):
    try:
        import cupy as cp
        if gpu:
            return cp
        return np
    except ImportError:
        return np


def generate_dirac_images(width, n_tasks, nnz, seed=None,
                          overlap=0.5, binary=False, positive=False):
    """Generate dirac images."""

    rnd = np.random.RandomState(seed)
    n_features = width ** 2
    nnz = min(n_features, max(nnz, 1))
    coef_overlap = np.zeros((width, width))
    coefs_no_overlap = np.zeros((width, width, n_tasks))
    no = max(int(nnz * overlap), 1)
    p = round(nnz ** 0.5)
    q = nnz // p
    dist_min = max((width - 4) // (min(p, q) + 1), 6)
    i = 0
    while(True):
        i += 1
        indo = rnd.randint(2, width - 2, size=(2, nnz))
        indx_o, indy_o = indo
        distx = (indx_o[:, None] - indx_o[None, :]) ** 2
        disty = (indy_o[:, None] - indy_o[None, :]) ** 2
        dists = distx + disty + n_features * np.eye(nnz)
        if dists.min() ** 0.5 > dist_min:
            break
        if i > 5000:
            raise ValueError("Generating images taked eternity !")
    if overlap:
        nn = nnz - (overlap > 0) * no
        vals = (- 1) ** rnd.randint(2, size=no)
        coef_overlap[indx_o[nn:], indy_o[nn:]] = vals

    if 1 - overlap:
        nn = nnz - (overlap > 0) * no
        nn_per_region = round(nn * n_tasks / nnz)
        radius = max(round((nn_per_region * 0.5) ** 0.5) // 2, 1)
        radius = 1
        for q in range(nn):
            j = q % nn
            posx, posy = indo[:, j]
            maxx = min(posx + radius, width)
            minx = max(posx - radius, 1)
            maxy = min(posy + radius, width)
            miny = max(posy - radius, 1)
            i = 0
            while(True):
                i += 1
                indx_no = rnd.randint(minx, maxx, size=n_tasks)
                indy_no = rnd.randint(miny, maxy, size=n_tasks)
                indno = np.vstack((indx_no, indy_no))
                if np.unique(indno, axis=1).shape[1] == n_tasks:
                    if not coefs_no_overlap[indx_no, indy_no].any():
                        break
                if i > 10000:
                    raise ValueError("Generating images taked eternity !")
            val = (- 1) ** rnd.randint(2)
            coefs_no_overlap[indx_no, indy_no, np.arange(n_tasks)] = val

    if binary is False:
        coefs_no_overlap *= 20. + 5 * rnd.rand(width, width, n_tasks)
        coef_overlap *= 20. + 5 * rnd.rand(width, width)

    coef_overlap = coef_overlap[:, :, None]
    coefs = coefs_no_overlap + (overlap > 0) * coef_overlap

    if positive:
        coefs = abs(coefs)
    return coefs


def toeplitz_2d(width, corr=0.9, p=2):
    """Generate 2d toeplitz covariance matrix."""
    n_features = width ** 2
    M = groundmetric2d(n_features, p=p, normed=False)
    cov = corr ** M
    return cov


def gaussian_design(n_samples, theta, corr=0.1, seed=None,
                    denoising=False, scaled=True, sigma=1.):
    """Generate multi-task regression data.

    Generates multi-task regression data: Triplet (X, Y, Theta)
    according to the equation Yk = Xk.Thetak + eps for each task k
    where eps is a random gaussian rv.

    Xk are random gaussians matrices with a covariance matrices given by
    toeplitz matrices. The correlation between features is controled by
    the corr argument.

    Parameters
    ----------
    n_samples: int
        number of samples per task > 0.
    theta: array, shape (n_features, n_tasks).
        Each column is the regression coef of a task.
    snr: float. (optional, default 0.)
        Signal-noise-ratio in decibels.
    corr: float (optional, default 0.5)
        Feature correlation parameter in [0, 1]
    denoising: bool (optional, False)
        if True, denoising problem. All X == Identity.
    scaled: bool (optional, False)
        if True, features are scaled to unit variance.

    Returns
    -------
    X: list of arrays
        list of training arrays Xk of shape (n_samples, n_features)
    Y: list of arrays
        list of target arrays Yk of shape (n_samples,)

    """
    rnd = np.random.RandomState(seed)
    n_tasks_all = 20
    noise = rnd.randn(n_tasks_all, n_samples)
    n_features, n_tasks = theta.shape
    width = int(n_features ** 0.5)
    # Create lists of training and target data Xk, Yk:
    X, Y = [], []

    for t in range(n_tasks):
        cov = toeplitz_2d(width, corr=corr)
        if denoising:
            x = np.eye(n_samples)
        else:
            x = rnd.multivariate_normal(np.zeros(n_features),
                                        cov=cov, size=n_samples)
            if scaled:
                x /= x.std(axis=0)
        y = x.dot(theta[:, t])
        X.append(x)
        Y.append(y)

    Y = np.array(Y)
    Y += sigma * noise[:n_tasks] * Y.std(axis=1).max()

    return np.array(X), Y


def groundmetric(n_features, p=1, normed=False):
    """Compute ground metric matrix on the 1D grid 0:`n_features`.

    Parameters
    ----------
    n_features: int > 0.
    p: int > 0.
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: 2D array (n_features, n_features).

    """
    x = np.arange(0, n_features).reshape(-1, 1).astype(float)
    xx, yy = np.meshgrid(x, x)
    M = abs(xx - yy) ** p
    if normed:
        M /= np.median(M)
    return M


def groundmetric2d(n_features, p=1, normed=False):
    """Compute ground metric matrix on the 2D grid of width `n_features ** 0.5`.

    Parameters
    ----------
    n_features: int > 0.
    p: int > 0.
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: 2D array (n_features, n_features).

    """
    d = int(n_features ** 0.5)
    n_features = d ** 2
    M = groundmetric(d, p=2, normed=False)
    M = M[:, np.newaxis, :, np.newaxis] + M[np.newaxis, :, np.newaxis, :]
    M = M.reshape(n_features, n_features) ** (p / 2)

    if normed:
        M /= np.median(M)
    return M


def auc_prc(coefs_true, coefs_pred, precision=0, mean=True):
    """Compute Area under the Precision-Recall curve.

    Compute Area under the Precision-Recall curve for binary support
    prediction.

    Parameters
    ----------
    coefs_true : array of shape (n_features, n_tasks)
        True coefs.
    coefs_pred: array of shape (n_features, n_tasks)
        Estimated MT coefs.
    precision: threshold to be considered 0.
    mean: compute mean of aucs across tasks.

    Returns
    -------
    float.
        AUC PRC.

    """
    if not coefs_true.any():
        warnings.warn("TRUE COEFS ARE ALL ZERO !")
        return 0.
    auc = 0.
    y_true = (abs(coefs_true) > precision).astype(int)
    if mean:
        for y_t, y_p in zip(y_true.T, abs(coefs_pred.T)):
            auc += average_precision_score(y_t, y_p)
        auc /= coefs_true.shape[1]
    else:
        y_true = y_true.flatten()
        y_pred = coefs_pred.flatten()
        auc = average_precision_score(y_true, y_pred)
    return auc


def inspector_mtw(x_real, rescaling=1., verbose=False, rate=50,
                  precision=0, prc_only=False):
    """Constructs callaback closure for MTW estimator.

    Constructs callaback closure called to update metrics after each
    iteration.

    Parameters
    ----------
    x_real : array, shape (n_features, n_tasks).
        optimal coefs.
    x_bar : array, shape (n_features).
        optimal barycenter.
    rate : int.
        printing rate.
    precision : float.
        between (0 - 1). sparsity threshold.

    """
    objectives = []
    errors = []
    aucs = []
    cstrs = []
    it = [0]

    def inspector_cl(xk, v=None):
        """Nested function."""
        if not prc_only:
            obj = v
            err = ((xk / rescaling - x_real) ** 2).mean()
            err /= max((xk / rescaling).max(), x_real.max(), 1)
            objectives.append(obj)
            errors.append(err)
        prc = auc_prc(x_real, xk / rescaling, precision=precision)
        aucs.append(prc)
        titles = ["it", "f(t)", "\t  RMSE(t - t^*)",
                  "\t AUC - Precision-Recall"]
        if verbose:
            if it[0] == 0:
                print("----------------")
                strings = [name.center(8) for name in titles]
                print(' | '.join(strings))
            if it[0] % rate == 0:
                strings = [("%d" % it[0]).rjust(8),
                           ("%.2e" % obj).rjust(8),
                           ("%.2e" % err).rjust(18),
                           ("%.4f" % prc).rjust(18)]
                print(' | '.join(strings))
            it[0] += 1
    inspector_cl.obj = objectives
    inspector_cl.rmse = errors
    inspector_cl.prc = aucs
    inspector_cl.cstr = cstrs

    return inspector_cl


def set_grid(ax):
    # Major ticks every 20, minor ticks every 5
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    major_ticks = np.arange(xmin, xmax, 5)
    minor_ticks = np.arange(ymin, ymax, 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


def contour_coefs(coef, ax, cmaps, title=''):
    w, w, n = coef.shape
    for i, cmap in enumerate(cmaps[:n]):
        m2 = coef[:, :, i].max()

        if m2:
            levels = np.logspace(-4, 0., 4) * m2
            ax.contourf(coef[:, :, i].T, cmap=cmap,
                        antialiased=False, alpha=0.6,
                        levels=levels, origin='lower')
        else:
            ax.contour(coef[:, :, i].T, cmap=cmap,
                       antialiased=True, alpha=0.6,
                       levels=[0], vmax=1., origin='lower')
        ax.set_ylim([0, w])
        ax.set_xlim([0, w])
    ax.set_title(title)
    set_grid(ax)
