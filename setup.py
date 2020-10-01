#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy
# from Cython.Build import cythonize


setup(
    name='episimlab',
    version="0.1.0",
    author='Ethan Ho',
    author_email='eho@tacc.utexas.edu',
    keywords='seir model covid-19',
    package_dir={'':'src'},
    packages=find_packages("src", exclude=[]),
    # package_data={'episimlab': list()},
    # include_package_data=False,
    python_requires='>=3.6, <4',
    install_requires=[
        'xarray-simlab',
        'dask',
        'dask[distributed]'
    ],
    extras_require={
        'dev': ['pytest','tox'],
        'test': ['pytest','tox']
    },
    ext_modules=list(),
    include_dirs=[numpy.get_include()],
)

