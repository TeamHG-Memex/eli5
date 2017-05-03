#!/usr/bin/env bash
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ;
pushd build
cmake ..
make -j 1
popd