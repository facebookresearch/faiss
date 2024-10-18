#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

FAISS_OPT_LEVEL= LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss.so
FAISS_OPT_LEVEL=AVX2 LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss_avx2.so
FAISS_OPT_LEVEL=AVX512 LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss_avx512.so
