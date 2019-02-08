Multi-task Wasserstein (mtw)
============================

Wasserestein regularized Multi-task regression.

Given high dimensional regression datasets :math:`(X^t, y^t) t = 1..T` , MTW solves
the optimization problem:

|eq1|

where:

|eq2| with |eq3|

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

.. |eq1| image:: https://latex.codecogs.com/gif.latex?\min_{\substack{\theta^1,&space;\dots,&space;\theta^T&space;\\&space;\bar{\theta}&space;\in&space;\mathbb{R}^p}&space;}&space;\frac{1}{2n}&space;\sum_{t=1}^T{\|&space;X^t&space;\theta^t&space;-&space;Y^t&space;\|^2}&space;&plus;&space;H(\theta^1,&space;\dots,&space;\theta^T;&space;\bar{\theta})
.. |eq2| image:: https://latex.codecogs.com/gif.latex?H(\theta^1,&space;\dots,&space;\theta^T;&space;\bar{\theta})&space;=&space;\frac{\mu}{T}&space;\overbrace{&space;\sum_{t=1}^{T}&space;\widetilde{W}(\theta^t,&space;\bar{\theta})}^{&space;\text{supports&space;proximity}}&space;&plus;&space;\frac{\lambda}{T}&space;\overbrace{&space;\sum_{t=1}^T&space;\|\theta^t\|_1}^{\text{sparsity}},
.. |eq3| image:: https://latex.codecogs.com/gif.latex?\widetilde{W}(\theta^t,&space;\bar{\theta})&space;=&space;W(\theta_&plus;^t,&space;\bar{\theta}_&plus;)&space;&plus;&space;W(\theta_-^t,&space;\bar{\theta}_-)
