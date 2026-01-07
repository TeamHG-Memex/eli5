#!/usr/bin/env bash

py.test --doctest-modules \
        --ignore eli5/xgboost.py \
        --ignore eli5/lightgbm.py \
        --ignore eli5/catboost.py \
        --ignore eli5/keras \
        --ignore eli5/formatters/image.py \
        --ignore tests/utils_image.py \
        --cov=eli5 --cov-report=html --cov-report=term  "$@"
