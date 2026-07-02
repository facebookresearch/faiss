#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Single source of truth for the CUDA version targeted by this toolkit.
# All build scripts source this file.
#
# Specify the version at build time with EITHER variable (the other is derived):
#
#   FAISS_CUDA_VER=13.3  make build            # human-readable version
#   FAISS_CUDA_TAG=cu133 make build            # short wheel/library tag
#
# Defaults to CUDA 13.2 (cu132) when neither is set.
#
# Derived everywhere as:
#   wheel/package name : faiss-gpu-${FAISS_CUDA_TAG}
#   C++ library names  : libfaiss-{arch}-${FAISS_CUDA_TAG}.so
#
# On a host with multiple toolkits installed (e.g. /usr/local/cuda-13.2 and
# /usr/local/cuda-13.3), CUDA_HOME is auto-resolved to the directory matching
# the requested version, so the correct nvcc is used. An explicit CUDA_HOME
# always wins.

# --- Resolve FAISS_CUDA_VER / FAISS_CUDA_TAG (specify either, derive the other) ---
if [ -n "${FAISS_CUDA_VER:-}" ]; then
    : "${FAISS_CUDA_TAG:=cu${FAISS_CUDA_VER//./}}"          # 13.3 -> cu133
elif [ -n "${FAISS_CUDA_TAG:-}" ]; then
    _faiss_cuda_digits="${FAISS_CUDA_TAG#cu}"               # cu133 -> 133
    : "${FAISS_CUDA_VER:=${_faiss_cuda_digits%?}.${_faiss_cuda_digits: -1}}"  # 133 -> 13.3
    unset _faiss_cuda_digits
fi
export FAISS_CUDA_VER="${FAISS_CUDA_VER:-13.2}"
export FAISS_CUDA_TAG="${FAISS_CUDA_TAG:-cu${FAISS_CUDA_VER//./}}"

# --- Resolve CUDA_HOME to the matching toolkit when not explicitly set ---
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda-${FAISS_CUDA_VER}" ]; then
        export CUDA_HOME="/usr/local/cuda-${FAISS_CUDA_VER}"
    else
        export CUDA_HOME="/usr/local/cuda"
    fi
fi
export PATH="$CUDA_HOME/bin:$PATH"

# --- Sanity: warn if the resolved nvcc does not match the requested version ---
if command -v nvcc >/dev/null 2>&1; then
    _faiss_nvcc_ver="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
    if [ -n "$_faiss_nvcc_ver" ] && [ "$_faiss_nvcc_ver" != "$FAISS_CUDA_VER" ]; then
        echo "[cuda_env] WARNING: nvcc reports CUDA $_faiss_nvcc_ver but FAISS_CUDA_VER=$FAISS_CUDA_VER" >&2
        echo "[cuda_env]          (CUDA_HOME=$CUDA_HOME). Set CUDA_HOME or FAISS_CUDA_VER to match." >&2
    fi
    unset _faiss_nvcc_ver
fi

# Echo "-sm<arch>" when the build targets exactly one GPU arch, else nothing.
# Used to tag single-arch wheels/libraries (e.g. faiss-gpu-cu133-sm121) so a
# wheel that only runs on one GPU generation is identifiable by name. Reads
# CUDA_ARCHS (arg overrides), tolerant of ";"/","/"\;" separators and
# -real/-virtual suffixes.
faiss_sm_suffix() {
    local archs="${1:-${CUDA_ARCHS:-}}"
    local uniq
    uniq=$(printf '%s' "$archs" \
        | sed -E 's/[\\,;]+/\n/g; s/-(real|virtual)//g; s/[[:blank:]]//g' \
        | sed '/^$/d' | sort -u)
    if [ -n "$uniq" ] && [ "$(printf '%s\n' "$uniq" | wc -l | tr -d ' ')" -eq 1 ]; then
        printf -- '-sm%s' "$uniq"
    fi
}
