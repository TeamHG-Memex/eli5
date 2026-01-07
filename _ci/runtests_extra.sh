#!/usr/bin/env bash
py.test --doctest-modules \
        --ignore tests/test_lime.py \
        --ignore tests/test_formatters.py \
        --ignore tests/test_samplers.py \
        --ignore tests/test_sklearn_explain_prediction.py \
        --ignore tests/test_sklearn_explain_weights.py \
        --ignore tests/test_sklearn_vectorizers.py \
        --ignore tests/test_utils.py \
        --ignore eli5/lightning.py \
        --cov=eli5 --cov-report=html --cov-report=term "$@"
