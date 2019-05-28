#!/usr/bin/env bash

py.test --doctest-modules \
        --ignore eli5/xgboost.py \
        --ignore eli5/lightgbm.py \
        --ignore eli5/catboost.py \
        --cov=eli5 --cov-report=html --cov-report=term  "$@"
