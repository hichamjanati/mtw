########################
Quick Start with the mtw
########################

This package serves as a skeleton package aiding at MNE
compatible packages.

Whatever


2. Edit the documentation
-------------------------

.. _Sphinx: http://www.sphinx-doc.org/en/stable/

The documentation is created using Sphinx_. In addition, the examples are
created using ``sphinx-gallery``. Therefore, to generate locally the
documentation, you are required to install the following packages::

    $ pip install sphinx sphinx-gallery sphinx_rtd_theme matplotlib numpydoc pillow

The documentation is made of:

* a home page, ``doc/index.rst``;
* an API documentation, ``doc/api.rst`` in which you should add all public
  objects for which the docstring should be exposed publicly.
* a User Guide documentation, ``doc/user_guide.rst``, containing the narrative
  documentation of your package, to give as much intuition as possible to your
  users.
* examples which are created in the `examples/` folder. Each example
  illustrates some usage of the package. the example file name should start by
  `plot_*.py`.

The documentation is built with the following commands::

    $ cd doc
    $ make html

3. Setup the continuous integration
-----------------------------------

The project template already contains configuration files of the continuous
integration system. Basically, the following systems are set:

* Travis_ CI is used to test the package in Linux. We provide you with an
  initial ``.travis.yml`` configuration file. So you only need to create
  a Travis account, activate own repository and trigger a build.

* AppVeyor is used to test the package in Windows. You need to activate
  AppVeyor for your own repository. Refer to the AppVeyor documentation.

* Circle CI is used to check if the documentation is generated properly. You
  need to activate Circle CI for your own repository. Refer to the Circle CI
  documentation.

* ReadTheDocs is used to build and host the documentation. You need to activate
  ReadTheDocs for your own repository. Refer to the ReadTheDocs documentation.

* CodeCov for tracking the code coverage of the package. You need to activate
  CodeCov for you own repository.

* PEP8Speaks for automatically checking the PEP8 compliance of your project for
  each Pull Request.

.. _Travis: https://travis-ci.com/getting_started

Publish your package
====================

.. _PyPi: https://packaging.python.org/tutorials/packaging-projects/
.. _conda-foge: https://conda-forge.org/

You can make your package available through PyPi_ and conda-forge_. Refer to
the associated documentation to be able to upload your packages such that
it will be installable with ``pip`` and ``conda``. Once published, it will
be possible to install your package with the following commands::

    $ pip install mne-foo
    $ conda install -c conda-forge mne-foo
