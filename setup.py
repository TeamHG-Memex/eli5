#!/usr/bin/env python
from setuptools import setup, find_packages
import re
import os


def get_version():
    fn = os.path.join(os.path.dirname(__file__), "eli5", "__init__.py")
    with open(fn) as f:
        return re.findall("__version__ = '([\d\.\w]+)'", f.read())[0]


setup(
    name='eli5',
    version=get_version(),
    author='Mikhail Korobov',
    author_email='kmike84@gmail.com',
    license='MIT license',
    long_description=open('README.rst').read() + "\n\n" + open('CHANGES.rst').read(),
    description="Debug machine learning classifiers and explain their predictions",
    url='https://github.com/TeamHG-Memex/eli5',
    zip_safe=False,
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy >= 1.9.0',
        'scipy',
        'singledispatch >= 3.4.0.3',
        'six',
        'typing',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
