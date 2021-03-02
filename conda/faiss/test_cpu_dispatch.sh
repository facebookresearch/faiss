#!/bin/sh

set -e

FAISS_DISABLE_CPU_FEATURES=AVX2 LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss.so
LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss_avx2.so
