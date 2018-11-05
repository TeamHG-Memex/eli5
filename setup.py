#!/usr/bin/env python
from setuptools import setup, find_packages
import re
import os


def get_version():
    fn = os.path.join(os.path.dirname(__file__), "eli5", "__init__.py")
    with open(fn) as f:
        return re.findall("__version__ = '([\d.\w]+)'", f.read())[0]


def get_long_description():
    readme = open('README.rst').read()
    changelog = open('CHANGES.rst').read()
    return "\n\n".join([
        readme,
        changelog.replace(':func:', '').replace(':ref:', '')
    ])

setup(
    name='eli5',
    version=get_version(),
    author='Mikhail Korobov, Konstantin Lopuhin',
    author_email='kmike84@gmail.com, kostia.lopuhin@gmail.com',
    license='MIT license',
    long_description=get_long_description(),
    description="Debug machine learning classifiers and explain their predictions",
    url='https://github.com/TeamHG-Memex/eli5',
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'attrs > 16.0.0',
        'jinja2',
        'numpy >= 1.9.0',
        'scipy',
        'six',
        'scikit-learn >= 0.18',
        'typing',
        'graphviz',
        'tabulate>=0.7.7',
    ],
    extras_require={
        ":python_version<'3.5.6'": [
            'singledispatch >= 3.4.0.3',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
