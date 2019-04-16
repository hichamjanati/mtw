from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


def readme():
    with open('README.md') as f:
        return f.read()


extensions = [
    Extension(
        "mtw.solver_cd",
        ['mtw/solver_cd.pyx'],
    ),
]

INSTALL_REQUIRES = ['numpy', 'scipy', 'cython', 'joblib', 'numba',
                    'scikit-learn']

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
        'download'

    ]
}

if __name__ == "__main__":
    setup(name="mtw",
          packages=find_packages(),
          ext_modules=cythonize(extensions),
          include_dirs=[numpy.get_include()],
          install_requires=INSTALL_REQUIRES,
          extras_require=EXTRAS_REQUIRE,
          )
