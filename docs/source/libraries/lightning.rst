.. _library-lightning:

lightning
=========

eli5 supports lightning_ library, which contains linear classifiers
with API largely compatible with scikit-learn_.

.. _lightning: https://github.com/scikit-learn-contrib/lightning
.. _scikit-learn: https://github.com/scikit-learn/scikit-learn

Using eli5 with estimators from lightning is exactly the same as
using it for scikit-learn built-in linear estimators - see
:ref:`sklearn-additional-kwargs` and :ref:`sklearn-linear-estimators`.

Supported lightning estimators:

* AdaGradClassifier_
* AdaGradRegressor_
* CDClassifier_
* CDRegressor_
* FistaClassifier_
* FistaRegressor_
* LinearSVC_
* LinearSVR_
* SAGAClassifier_
* SAGARegressor_
* SAGClassifier_
* SAGRegressor_
* SDCAClassifier_
* SDCARegressor_
* SGDClassifier_
* SGDRegressor_

.. _AdaGradClassifier: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.AdaGradClassifier.html#lightning.classification.AdaGradClassifier
.. _AdaGradRegressor: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.AdaGradRegressor.html#lightning.regression.AdaGradRegressor
.. _CDClassifier: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.CDClassifier.html#lightning.classification.CDClassifier
.. _CDRegressor: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.CDRegressor.html#lightning.regression.CDRegressor
.. _FistaClassifier: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.FistaClassifier.html#lightning.classification.FistaClassifier
.. _FistaRegressor: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.FistaRegressor.html#lightning.regression.FistaRegressor
.. _LinearSVC: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.LinearSVC.html#lightning.classification.LinearSVC
.. _LinearSVR: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.LinearSVR.html#lightning.regression.LinearSVR
.. _SAGAClassifier: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.SDCAClassifier.html#lightning.classification.SDCAClassifier
.. _SAGARegressor: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.SDCARegressor.html#lightning.regression.SDCARegressor
.. _SAGClassifier: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.SAGClassifier.html
.. _SAGRegressor: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.SAGRegressor.html#lightning.regression.SAGRegressor
.. _SDCAClassifier: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.SDCAClassifier.html#lightning.classification.SDCAClassifier
.. _SDCARegressor: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.SDCARegressor.html#lightning.regression.SDCARegressor
.. _SGDClassifier: http://contrib.scikit-learn.org/lightning/generated/lightning.classification.SGDClassifier.html#lightning.classification.SGDClassifier
.. _SGDRegressor: http://contrib.scikit-learn.org/lightning/generated/lightning.regression.SGDRegressor.html#lightning.regression.SGDRegressor
