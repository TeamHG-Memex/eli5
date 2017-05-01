#!/usr/bin/env bash

py.test --doctest-modules \
        --ignore eli5/xgboost.py \
        --ignore eli5/lightgbm.py \
        --cov=eli5 --cov-report=html --cov-report=term  "$@"
