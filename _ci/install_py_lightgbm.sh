#!/usr/bin/env bash
if [ -z "${LIGHTGBM_CHECKOUT}" ]; then
    echo "LIGHTGBM_CHECKOUT is not set; LightGBM is not installed"
    exit 0
fi

pushd "${LIGHTGBM_CHECKOUT}/python-package/"
/usr/bin/env python setup.py install
popd
