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
        ['mtw.solver_cd.pyx'],
    ),
]

if __name__ == "__main__":
    setup(name="mtw",
          packages=find_packages(),
          ext_modules=cythonize(extensions),
          include_dirs=[numpy.get_include()]
          )
