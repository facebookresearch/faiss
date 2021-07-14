#!/bin/sh#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

wget -O - https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2-linux-aarch64.tar.gz | tar xzf -
cp -R cmake-3.20.2-linux-aarch64/* $PREFIX
