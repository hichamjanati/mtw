
|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.com/hichamjanati/mtw.svg?branch=master
.. _Travis: https://travis-ci.com/hichamjanati/mtw

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/l7g6vywwwuyha49l?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/hichamjanati/mtw

.. |Codecov| image:: https://codecov.io/gh/hichamjanati/mtw/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hichamjanati/mtw

.. |CircleCI| image:: https://circleci.com/gh/hichamjanati/mtw.svg?style=svg
.. _CircleCI: https://circleci.com/gh/hichamjanati/mtw/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mtw/badge/?version=latest
.. _ReadTheDocs: https://mtw.readthedocs.io/en/latest/


Documentation
=============
Detailed examples can be found `here <https://mtw.readthedocs.io>`_.

Multi-task Wasserstein (mtw)
============================

Wasserestein regularization for sparse Multi-task regression.

Given high dimensional regression datasets :math:`(X^t, y^t) t = 1..T` , MTW solves
the optimization problem:

|eq1|

where:

|eq2|

with

|eq3|

 where W is the Unbalanced KL Wasserstein distance.


Install the development version
===============================

From a console or terminal clone the repository and install MTW:

::

    git clone https://github.com/hichamjanati/mtw.git
    cd mtw/
    conda env create --file environment.yml
    source activate mtw-env
    pip install --no-deps -e .

Demos & Examples
================

Given a ground metric `M` and the entropy parameter that define the Wasserstein
metric, an MTW object can be created and fitted on multi-task regression data
`(X, y)`. Where the shapes of `X` and `Y` are (n_tasks, n_samples, n_features)
and (n_tasks, n_samples)

.. code:: python

   >>> from mtw import MTW
   >>> n_tasks, n_samples, n_features = 2, 10, 50
   >>> # Compute M as Euclidean distances matrix if not given
   >>> grid = np.arange(n_features)
   >>> M = (grid[:, None] - grid[None, :]) ** 2
   >>> # Some data X and y
   >>> X = np.random.randn(n_tasks, n_samples, n_features)
   >>> y = np.random.randn(n_tasks, n_samples)
   >>> epsilon = 1. / n_features
   >>> alpha = 0.1
   >>> beta = 0.1
   >>> mtw = MTW(alpha=alpha, beta=beta, M=M, epsilon=epsilon)
   >>> mtw = mtw.fit(X, y)
   >>> coefs = mtw.coefs_


A concomittant version where the standard deviation of each task is inferred.
The lower bound on sigma can be set via the `sigma0` parameter of MTW. The
following example sets this lower bound to 1% of the initial std estimation
`np.std(Y)`.

.. code:: python

    >>> from mtw import MTW
    >>> n_tasks, n_samples, n_features = 2, 10, 50
    >>> grid = np.arange(n_features)
    >>> M = (grid[:, None] - grid[None, :]) ** 2
    >>> # Some data X and y
    >>> X = np.random.randn(n_tasks, n_samples, n_features)
    >>> y = np.random.randn(n_tasks, n_samples)
    >>> epsilon = 1. / n_features
    >>> alpha = 0.1
    >>> beta = 0.1
    >>> sigma0 = 0.01
    >>> mtw = MTW(alpha=alpha, beta=beta, M=M, epsilon=epsilon, sigma0=sigma0)


See `the Documentation <https://mtw.readthedocs.io>`_ for details.


Dependencies
============

All dependencies are in ``./environment.yml``

Cite
====

If you use this code, please cite:

::

    @InProceedings{janati19a,
    author={Hicham Janati and Marco Cuturi and Alexandre Gramfort},
    title={Wasserstein regularization for sparse multi-task regression},
    booktitle = {Proceedings of the Twenty-second International Conference on Artificial Intelligence and Statistics},
    year = 	 {2019},
    volume = 	 {89},
    series = 	 {Proceedings of Machine Learning Research},
    month = 	 {16--19 Apr},
    publisher = 	 {PMLR},
    }

ArXiv link: https://arxiv.org/abs/1805.07833

If you use the concomittant version of MTW, please cite:
::
    @InProceedings{janati19b,
    author={Hicham Janati and Thomas Bazeille and Bertrand Thirion and Marco Cuturi and Alexandre Gramfort},
    title={Group level M-EEG source imaging via Optimal transport: Minimum Wasserstein Estimates},
    booktitle = {Proceedings of the Fifty-th Conference on Information Processing and Medical Imaging},
    year = 	 {2019},
    month = 	 {02--07 June},
    publisher = 	 {Springer},
    }

.. |eq1| image:: https://latex.codecogs.com/gif.latex?\min_{\substack{\theta^1,&space;\dots,&space;\theta^T&space;\\&space;\bar{\theta}&space;\in&space;\mathbb{R}^p}&space;}&space;\frac{1}{2n}&space;\sum_{t=1}^T{\|&space;X^t&space;\theta^t&space;-&space;Y^t&space;\|^2}&space;&plus;&space;H(\theta^1,&space;\dots,&space;\theta^T;&space;\bar{\theta})
.. |eq2| image:: https://latex.codecogs.com/gif.latex?H(\theta^1,&space;\dots,&space;\theta^T;&space;\bar{\theta})&space;=&space;\frac{\mu}{T}&space;\overbrace{&space;\sum_{t=1}^{T}&space;\widetilde{W}(\theta^t,&space;\bar{\theta})}^{&space;\text{supports&space;proximity}}&space;&plus;&space;\frac{\lambda}{T}&space;\overbrace{&space;\sum_{t=1}^T&space;\|\theta^t\|_1}^{\text{sparsity}},
.. |eq3| image:: https://latex.codecogs.com/gif.latex?\widetilde{W}(\theta^t,&space;\bar{\theta})&space;=&space;W(\theta_&plus;^t,&space;\bar{\theta}_&plus;)&space;&plus;&space;W(\theta_-^t,&space;\bar{\theta}_-)
