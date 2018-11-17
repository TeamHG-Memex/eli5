# -*- coding: utf-8 -*-
"""
An impementation of LIME (http://arxiv.org/abs/1602.04938), an algorithm to
explain predictions of black-box models.
"""
from __future__ import absolute_import
from typing import Any, Callable, Dict, Optional

import numpy as np  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.linear_model import SGDClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.base import clone, BaseEstimator  # type: ignore

import eli5
from eli5.sklearn.utils import sklearn_version
from eli5.lime.samplers import BaseSampler
from eli5.lime.textutils import DEFAULT_TOKEN_PATTERN, CHAR_TOKEN_PATTERN
from eli5.lime.samplers import MaskingTextSamplers
from eli5.lime.utils import (
    fit_proba,
    score_with_sample_weight,
    mean_kl_divergence,
    fix_multiclass_predict_proba,
    rbf
)
from eli5.lime._vectorizer import SingleDocumentVectorizer


class TextExplainer(BaseEstimator):
    """
    TextExplainer allows to explain predictions of black-box text classifiers
    using LIME algorithm.

    Parameters
    ----------
    n_samples : int
        A number of samples to generate and train on. Default is 5000.

        With larger n_samples it takes more CPU time and RAM to explain
        a prediction, but it could give better results. Larger n_samples
        could be also required to get good results if you don't want to
        make strong assumptions about the black-box classifier
        (e.g. char_based=True and position_dependent=True).
    char_based : bool
        True if explanation should be char-based, False if it should be
        token-based. Default is False.
    clf : object, optional
        White-box probabilistic classifier. It should be supported by eli5,
        follow scikit-learn interface and provide predict_proba method.
        When not set, a default classifier is used (logistic regression with
        elasticnet regularization trained with SGD).
    vec : object, optional
        Vectorizer which converts generated texts to feature vectors
        for the white-box classifier. When not set, a default vectorizer is
        used; which one depends on ``char_based`` and ``position_dependent``
        arguments.
    sampler : MaskingTextSampler or MaskingTextSamplers, optional
        Sampler used to generate modified versions of the text.
    position_dependent : bool
        When True, a special vectorizer is used which takes
        each token or character (depending on ``char_based`` value)
        in account separately. When False (default) a vectorized passed in
        ``vec`` or a default vectorizer is used.

        Default vectorizer converts text to vector using bag-of-ngrams
        or bag-of-char-ngrams approach (depending on ``char_based`` argument).
        It means that it may be not powerful enough to approximate a black-box
        classifier which e.g. takes in account word FOO in the beginning of
        the document, but not in the end.

        When ``position_dependent`` is True the model becomes powerful enough
        to account for that, but it can become more noisy and require
        larger ``n_samples`` to get an OK explanation.

        When ``char_based=False`` the default vectorizer uses word bigrams
        in addition to unigrams; this is less powerful than
        ``position_dependent=True``, but can give similar results in practice.
    rbf_sigma : float, optional
        Sigma parameter of RBF kernel used to post-process cosine similarity
        values. Default is None, meaning no post-processing
        (cosine simiilarity is used as sample weight as-is).
        Small ``rbf_sigma`` values (e.g. 0.1) tell the classifier to pay
        more attention to generated texts which are close to the original text.
        Large ``rbf_sigma`` values (e.g. 1.0) make distance between text
        irrelevant.

        Note that if you're using large ``rbf_sigma`` it could be more
        efficient to use custom ``samplers`` instead, in order to generate
        text samples which are closer to the original text in the first place.
        Use e.g. ``max_replace`` parameter of :class:`~.MaskingTextSampler`.
    random_state : integer or numpy.random.RandomState, optional
        random state
    expand_factor : int or None
        To approximate output of the probabilistic classifier generated
        dataset is expanded by ``expand_factor`` (10 by default)
        according to the predicted label probabilities. This is a workaround
        for scikit-learn limitation (no cross-entropy loss for non 1/0 labels).
        With larger values training takes longer, but probability output
        can be approximated better.

        expand_factor=None turns this feature off; pass None when you know
        that black-box classifier returns only 1.0 or 0.0 probabilities.
    token_pattern : str, optional
        Regex which matches a token. Use it to customize tokenization.
        Default value depends on ``char_based`` parameter.

    Attributes
    ----------
    rng_ : numpy.random.RandomState
        random state

    samples_ : list[str]
        A list of samples the local model is trained on.
        Only available after :func:`fit`.

    X_ : ndarray or scipy.sparse matrix
        A matrix with vectorized ``samples_``.
        Only available after :func:`fit`.

    similarity_ : ndarray
        Similarity vector. Only available after :func:`fit`.

    y_proba_ : ndarray
        probabilities predicted by black-box classifier
        (``predict_proba(self.samples_)`` result).
        Only available after :func:`fit`.

    clf_ : object
        Trained white-box classifier. Only available after :func:`fit`.

    vec_ : object
        Fit white-box vectorizer. Only available after :func:`fit`.

    metrics_ : dict
        A dictionary with metrics of how well the local
        classification pipeline approximates the black-box pipeline.
        Only available after :func:`fit`.
    """
    def __init__(self,
                 n_samples=5000,  # type: int
                 char_based=None,  # type: bool
                 clf=None,
                 vec=None,
                 sampler=None,  # type: BaseSampler
                 position_dependent=False,  # type: bool
                 rbf_sigma=None,  # type: float
                 random_state=None,
                 expand_factor=10,  # type: Optional[int]
                 token_pattern=None,  # type: str
                 ):
        # type: (...) -> None
        self.n_samples = n_samples
        self.random_state = random_state
        self.expand_factor = expand_factor
        self.rng_ = check_random_state(random_state)
        if clf is None:
            clf = self._default_clf()
        self.clf = clf

        if char_based is None:
            if token_pattern is None:
                self.char_based = False  # type: Optional[bool]
                self.token_pattern = DEFAULT_TOKEN_PATTERN
            else:
                self.char_based = None
                self.token_pattern = token_pattern
        else:
            if token_pattern is None:
                token_pattern = (CHAR_TOKEN_PATTERN if char_based
                                 else DEFAULT_TOKEN_PATTERN)
            self.char_based = char_based
            self.token_pattern = token_pattern

        if sampler is None:
            sampler = MaskingTextSamplers(
                sampler_params=[{'bow': False}, {'bow': True}],
                weights=[0.7, 0.3],
                token_pattern=self.token_pattern,
                random_state=self.rng_,
            )
        self.sampler = sampler
        self.rbf_sigma = rbf_sigma
        self.position_dependent = position_dependent
        if position_dependent:
            if vec is not None:
                raise ValueError("Custom vectorizers are only supported with "
                                 "position_dependent=False (because "
                                 "position_dependent=True uses its own "
                                 "vectorizer)")
        else:
            if vec is None:
                if self.char_based:
                    vec = CountVectorizer(
                        analyzer='char',
                        ngram_range=(2, 5),
                    )
                else:
                    vec = CountVectorizer(
                        token_pattern=self.token_pattern,
                        ngram_range=(1, 2),
                    )
            self.vec = vec

    def fit(self,
            doc,             # type: str
            predict_proba,   # type: Callable[[Any], Any]
            ):
        # type: (...) -> TextExplainer
        """
        Explain ``predict_proba`` probabilistic classification function
        for the ``doc`` example. This method fits a local classification
        pipeline following LIME approach.

        To get the explanation use :meth:`show_prediction`,
        :meth:`show_weights`, :meth:`explain_prediction` or
        :meth:`explain_weights`.

        Parameters
        ----------
        doc : str
            Text to explain
        predict_proba : callable
            Black-box classification pipeline. ``predict_proba``
            should be a function which takes a list of strings (documents)
            and return a matrix of shape ``(n_samples, n_classes)`` with
            probability values - a row per document and a column per output
            label.
        """
        self.doc_ = doc

        if self.position_dependent:
            samples, sims, mask, text = self.sampler.sample_near_with_mask(
                doc=doc,
                n_samples=self.n_samples
            )
            self.vec_ = SingleDocumentVectorizer(
                token_pattern=self.token_pattern
            ).fit([doc])
            X = ~mask
        else:
            self.vec_ = clone(self.vec).fit([doc])
            samples, sims = self.sampler.sample_near(
                doc=doc,
                n_samples=self.n_samples
            )
            X = self.vec_.transform(samples)

        if self.rbf_sigma is not None:
            sims = rbf(1-sims, sigma=self.rbf_sigma)

        self.samples_ = samples
        self.similarity_ = sims
        self.X_ = X
        self.y_proba_ = predict_proba(samples)
        self.clf_ = clone(self.clf)

        self.metrics_ = _train_local_classifier(
            estimator=self.clf_,
            samples=X,
            similarity=sims,
            y_proba=self.y_proba_,
            expand_factor=self.expand_factor,
            random_state=self.rng_
        )
        return self

    def show_prediction(self, **kwargs):
        """
        Call :func:`eli5.show_prediction` for the locally-fit
        classification pipeline. Keyword arguments are passed
        to :func:`eli5.show_prediction`.

        :func:`fit` must be called before using this method.
        """
        self._fix_target_names(kwargs)
        return eli5.show_prediction(self.clf_, self.doc_, vec=self.vec_,
                                    **kwargs)

    def explain_prediction(self, **kwargs):
        """
        Call :func:`eli5.explain_prediction` for the locally-fit
        classification pipeline. Keyword arguments are passed
        to :func:`eli5.explain_prediction`.

        :func:`fit` must be called before using this method.
        """
        self._fix_target_names(kwargs)
        return eli5.explain_prediction(self.clf_, self.doc_, vec=self.vec_,
                                       **kwargs)

    def show_weights(self, **kwargs):
        """
        Call :func:`eli5.show_weights` for the locally-fit
        classification pipeline. Keyword arguments are passed
        to :func:`eli5.show_weights`.

        :func:`fit` must be called before using this method.
        """
        self._fix_target_names(kwargs)
        return eli5.show_weights(self.clf_, vec=self.vec_, **kwargs)

    def explain_weights(self, **kwargs):
        """
        Call :func:`eli5.show_weights` for the locally-fit
        classification pipeline. Keyword arguments are passed
        to :func:`eli5.show_weights`.

        :func:`fit` must be called before using this method.
        """
        self._fix_target_names(kwargs)
        return eli5.explain_weights(self.clf_, vec=self.vec_, **kwargs)

    def _fix_target_names(self, kwargs):
        target_names = kwargs.get('target_names', None)
        if not target_names:
            return
        kwargs['target_names'] = np.array(target_names)[self.clf_.classes_]

    def _default_clf(self):
        kwargs = dict(
            loss='log',
            penalty='elasticnet',
            alpha=1e-3,
            random_state=self.rng_
        )
        if sklearn_version() >= '0.19':
            kwargs['tol'] = 1e-3
        return SGDClassifier(**kwargs)



def _train_local_classifier(estimator,
                            samples,
                            similarity,        # type: np.ndarray
                            y_proba,           # type: np.ndarray
                            expand_factor=10,  # type: Optional[int]
                            test_size=0.3,     # type: float
                            random_state=None,
                            ):
    # type: (...) -> Dict[str, float]
    rng = check_random_state(random_state)

    (X_train, X_test,
     similarity_train, similarity_test,
     y_proba_train, y_proba_test) = train_test_split(samples,
                                                     similarity,
                                                     y_proba,
                                                     test_size=test_size,
                                                     random_state=rng)

    # XXX: in the original lime code instead of a probabilitsic classifier
    # they build several regression models which try to output probabilities.
    #
    # XXX: Probability information is helpful because it could be hard
    # to get enough examples of all classes automatically, so we're fitting
    # classifier to produce the same probabilities, not only the same
    # best answer.

    # TODO: feature selection
    # Ideally, it should be supported as a Pipeline (i.e. user should
    # be able to configure it).
    fit_proba(estimator, X_train, y_proba_train,
              expand_factor=expand_factor,
              sample_weight=similarity_train,
              random_state=rng)

    y_proba_test_pred = estimator.predict_proba(X_test)
    if y_proba_test_pred.shape != y_proba_test.shape:
        # Sometimes generated training labels may contain only a subset of
        # target classes; it means it could happen that dimensions
        # of predicted probability matrices don't match.
        #
        # XXX: the fix is not complete; to explain predictions
        # of the fitted estimator one still have to take care of target_names.
        if not hasattr(estimator, 'classes_'):
            raise ValueError("Result dimensions don't match and estimator"
                             "doesn't provide 'classes_' attribute; can't"
                             "figure out how are columns related.")
        seen_classes = estimator.classes_
        complete_classes = np.arange(y_proba.shape[1])
        y_proba_test_pred = fix_multiclass_predict_proba(
            y_proba=y_proba_test_pred,
            seen_classes=seen_classes,
            complete_classes=complete_classes
        )

    return {
        'mean_KL_divergence': mean_kl_divergence(
            y_proba_test_pred,
            y_proba_test,
            sample_weight=similarity_test
        ),
        'score': score_with_sample_weight(estimator,
                                          X_test,
                                          y_proba_test.argmax(axis=1),
                                          sample_weight=similarity_test)
    }
