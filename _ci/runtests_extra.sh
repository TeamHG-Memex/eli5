#!/usr/bin/env bash
EXTRA_IGNORE=""
if [ -z "${LIGHTGBM_CHECKOUT}" ]; then
    EXTRA_IGNORE="--ignore eli5/lightgbm.py"
fi

py.test --doctest-modules \
        --ignore tests/test_lime.py \
        --ignore tests/test_formatters.py \
        --ignore tests/test_samplers.py \
        --ignore tests/test_sklearn_explain_prediction.py \
        --ignore tests/test_sklearn_explain_weights.py \
        --ignore tests/test_sklearn_vectorizers.py \
        --ignore tests/test_utils.py \
        --ignore eli5/lightning.py ${EXTRA_IGNORE} \
        --cov=eli5 --cov-report=html --cov-report=term "$@"
