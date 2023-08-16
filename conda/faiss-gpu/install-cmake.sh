#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

FAISS_DISABLE_CPU_FEATURES=AVX2 LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss.so
LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss_avx2.so
