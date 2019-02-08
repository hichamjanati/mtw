Multi-task Wasserstein (mtw)
============================

Wasserestein regularized Multi-task regression.

Given high dimensional regression datasets :math:`(X^t, y^t) t = 1..T`, MTW solves
the optimization problem:

.. math::

     \min_{\substack{\theta^1, \dots, \theta^T \\ \bar{theta} \in \bbR^p} } \frac{1}{2n} \sum_{t=1}^T{\| X^t \theta^t - Y^t \|^2}  +  H(\theta^1, \dots,  \theta^T; \bar{theta})


where:

.. math::

    H(\theta^1, \dots,  \theta^T; \bar{theta})  = \frac{\mu}{T} \overbrace{ \sum_{t=1}^{T} \Delta(\theta^t, \bar{theta})}^{ \text{supports proximity}}  +  \frac{\lambda}{T} \overbrace{ \sum_{t=1}^T \|\theta^t\|_1}^{\text{sparsity}}
    \Delta(\theta^t, \bar{theta}) = W(\theta_+^t, \bar{theta}_+)  + W(\theta_-^t, \bar{theta}_-)


and W the unbalanced Wasserstein distance.

Install the development version
===============================

From a console or terminal clone the repository and install CELER:


    git clone https://github.com/hichamjanati/mtw.git
    cd mtw/
    conda env create --file environment.yml
    source activate mtw-env
    pip install --no-deps -e .

Demos & Examples
================



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
    year = 	 {2019},
    volume = 	 {89},
    series = 	 {Proceedings of Machine Learning Research},
    month = 	 {16--19 Apr},
    publisher = 	 {PMLR},
    }

ArXiv link: https://arxiv.org/abs/1805.07833
