#!/usr/bin/env python

from distutils.core import setup
import roundtable

with open('README.rst') as f:
    long_description = f.read()

setup(name='roundtable',
      version=roundtable.__version__,
      description="King Arthur's finest collection of table-like containers",
      author='Jim Kitchen',
      author_email='jim22k@gmail.com',
      url='https://github.com/jim22k/roundtable',
      packages=['roundtable'],
      long_description=long_description,
     )