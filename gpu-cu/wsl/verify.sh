#!/bin/bash
# Verify the installed faiss-gpu-${FAISS_CUDA_TAG} wheel works (CPU + GPU).
# Usage (from PowerShell):
#   wsl -e bash gpu-cu/wsl/verify.sh
# To install first, pass --install:
#   wsl -e bash gpu-cu/wsl/verify.sh --install

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

VARIANT="${FAISS_VARIANT:-gpu-${FAISS_CUDA_TAG}$(faiss_sm_suffix)}"
# pip normalises hyphens to underscores in wheel filenames
WHEEL_PREFIX="faiss_${VARIANT//-/_}"
WHEEL=$(ls "$FAISS_ROOT"/build_output/${WHEEL_PREFIX}-*.whl 2>/dev/null | head -1)

# Fallback: plain "faiss" wheel (no variant)
if [[ -z "$WHEEL" ]]; then
    WHEEL=$(ls "$FAISS_ROOT"/build_output/faiss-*.whl 2>/dev/null | head -1)
fi

if [[ "${1:-}" == "--install" ]]; then
    if [[ -z "$WHEEL" ]]; then
        echo "ERROR: No wheel found in $FAISS_ROOT/build_output/"
        exit 1
    fi
    echo "Installing $WHEEL ..."
    pip3 install "$WHEEL" --break-system-packages --force-reinstall
fi

echo ""
echo "========================================="
echo " FAISS verify"
echo "========================================="
python3 - <<'EOF'
import faiss

print(f"  faiss version : {faiss.__version__}")
print(f"  GPU count     : {faiss.get_num_gpus()}")

# CPU sanity check
import numpy as np
d = 64
nb = 1000
xb = np.random.rand(nb, d).astype("float32")
index = faiss.IndexFlatL2(d)
index.add(xb)
D, I = index.search(xb[:5], 4)
assert I[0][0] == 0, "Self-search failed"
print(f"  CPU search    : OK ({nb} vectors, top-4)")

# GPU sanity check (if available)
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    D2, I2 = gpu_index.search(xb[:5], 4)
    assert I2[0][0] == 0, "GPU self-search failed"
    print(f"  GPU search    : OK (GPU 0)")

print("=========================================")
print("✓ All checks passed")
EOF
