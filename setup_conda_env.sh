#!/bin/bash
set -e

ENV_NAME="${1:-faiss-cuvs}"

echo "Creating conda environment: $ENV_NAME"

conda create -y -n "$ENV_NAME" python=3.12
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

conda config --set solver libmamba

# Base packages
conda install -y -q cmake=3.30.4 make=4.2 swig=4.0 "numpy>=2.0,<3.0" scipy=1.16 pytest=7.4 gflags=2.2 setuptools -c conda-forge

# X86_64 compiler and sysroot
conda install -y -q -c conda-forge gxx_linux-64=14.2 sysroot_linux-64=2.17

# MKL
conda install -y -q mkl=2024.2.2 mkl-devel=2024.2.2 -c conda-forge

# CUDA + cuVS
conda install -y -q libcuvs=26.06 'cuda-version=12.9' cuda-toolkit=12.9 -c rapidsai -c rapidsai-nightly -c conda-forge

# PyTorch (GPU)
conda install -y -q "pytorch>=2.7" "pytorch-gpu>=2.7" -c pytorch -c "nvidia/label/12.9"

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
