Contributing
============

ELI5 uses MIT license; contributions are welcome!

* Source code: https://github.com/TeamHG-Memex/eli5
* Issue tracker: https://github.com/TeamHG-Memex/eli5/issues

ELI5 supports Python 2.7 and Python 3.4+
To run tests make sure tox_ Python package is installed, then run

::

    tox

from source checkout.

We like high test coverage and mypy_ type annotations.

Making releases
---------------

Note: releases are made from master by eli5 maintainers.
When contributing a pull request, please do not update release notes
or package version.

To make a new release:

* Write a summary of changes to CHANGES.rst
* Bump version in ``eli5/__init__.py``
* Make a release on PyPI using twine_
* Tag a commit in git and push it

.. _tox: https://tox.readthedocs.io/en/latest/
.. _mypy: https://github.com/python/mypy
.. _twine: https://pypi.org/project/twine/
