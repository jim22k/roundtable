#!/usr/bin/env python

from distutils.core import setup
import roundtable

with open('README.rst') as f:
    # Strip off first 5 lines
    for i in range(5):
        f.readline()
    long_description = f.read()

setup(name='roundtable',
      version=roundtable.__version__,
      description="King Arthur's finest collection of table-like containers",
      license='BSD',
      keywords='table',
      author='Jim Kitchen',
      author_email='jim22k@gmail.com',
      url='https://github.com/jim22k/roundtable',
      packages=['roundtable'],
      long_description=long_description,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries ',
      ],
     )