.PHONY: help check build build-lib build-pkg wheel wheel-only clean env-create env-info test build-aarch64 check-aarch64

# Recipes rely on bash (cuda_env.sh uses bash parameter expansion).
SHELL := /bin/bash

# FAISS GPU Wheel Build Makefile
#
# CUDA version is a single knob: bump FAISS_CUDA_TAG / FAISS_CUDA_VER (or edit
# gpu-cu/scripts/cuda_env.sh) to retarget cu132 → cu133. Nothing else hard-codes it.
#
# The default targets (build, build-lib, build-pkg, wheel) build the x86_64
# variant (Intel MKL, AVX2/AVX512). Use the *-aarch64 targets for the
# ARM / DGX Spark variant (OpenBLAS + cuVS). Both publish the SAME package
# name faiss-gpu-$(FAISS_CUDA_TAG); the wheel's manylinux platform tag
# (x86_64 vs aarch64) selects the right one at install time.

CUDA_ARCHS ?= 75\;80\;86\;89\;90\;120
# CUDA_HOME: leave empty so cuda_env.sh resolves the toolkit matching
# FAISS_CUDA_VER (e.g. /usr/local/cuda-13.3 on a multi-toolkit host).
# Set explicitly to override.
CUDA_HOME ?=
FAISS_BUILD_JOBS ?= $(shell nproc)
PYTHON ?= python3
FAISS_ENABLE_CUVS ?= ON
# CUDA version is the single build input. Specify FAISS_CUDA_VER (e.g. 13.3);
# the tag is derived (13.3 -> cu133). Mirrors gpu-cu/scripts/cuda_env.sh.
#   make build FAISS_CUDA_VER=13.3
FAISS_CUDA_VER ?= 13.2
FAISS_CUDA_TAG := cu$(subst .,,$(FAISS_CUDA_VER))
GPU_CU_ROOT ?= gpu-cu
GPU_CU_SCRIPTS ?= $(GPU_CU_ROOT)/scripts
# Wheel package name = faiss-$(FAISS_VARIANT). Leave empty so the packaging
# script derives it: gpu-$(FAISS_CUDA_TAG), plus -sm<arch> when CUDA_ARCHS
# targets a single GPU arch. Set explicitly to force a custom name.
FAISS_VARIANT ?=

# Export the version knobs to every recipe so the build scripts (which source
# cuda_env.sh) see the requested version and resolve CUDA_HOME accordingly.
export FAISS_CUDA_VER
export FAISS_CUDA_TAG

help:
	@echo "FAISS GPU Wheel Build (CUDA $(FAISS_CUDA_VER))"
	@echo ""
	@echo "Targets (x86_64 — Intel MKL, AVX2/AVX512):"
	@echo "  make build             - Full build (lib + pkg + wheel)"
	@echo "  make build-lib         - Build C++ library only"
	@echo "  make build-pkg         - Build library and Python package"
	@echo "  make wheel             - Package wheel (full build chain)"
	@echo "  make wheel-only        - Repackage wheel from existing .so artifacts"
	@echo "  make check             - Check prerequisites"
	@echo ""
	@echo "Targets (aarch64 / DGX Spark — OpenBLAS + cuVS):"
	@echo "  make build-aarch64     - Full ARM build (lib + pkg + wheel)"
	@echo "  make check-aarch64     - Check ARM build prerequisites"
	@echo ""
	@echo "Common:"
	@echo "  make clean             - Remove build artifacts"
	@echo "  make env-create        - Create conda environment"
	@echo "  make env-info          - Show build configuration"
	@echo "  make test              - Run tests"
	@echo ""
	@echo "Environment Variables:"
	@echo "  FAISS_CUDA_VER         - CUDA version to build (default: 13.2). e.g. make build FAISS_CUDA_VER=13.3"
	@echo "  CUDA_ARCHS             - GPU architectures (default: 75;80;86;89;90;120)"
	@echo "  CUDA_HOME              - CUDA toolkit path (auto: /usr/local/cuda-<ver> if present)"
	@echo "  FAISS_BUILD_JOBS       - Parallel build jobs"
	@echo "  PYTHON                 - Python executable"
	@echo "  FAISS_ENABLE_CUVS      - Enable cuVS (ON/OFF)"
	@echo "  FAISS_VARIANT          - Optional wheel suffix (example: gpu-cu132)"
	@echo "  GPU_CU_ROOT            - Toolkit root (default: gpu-cu)"

build: build-lib build-pkg wheel
	@echo "✓ Full build complete"
	@ls -lh build_output/*.whl 2>/dev/null || echo "No wheel found"

build-lib:
	@export CUDA_ARCHS="$(CUDA_ARCHS)"; \
	export CUDA_HOME=$(CUDA_HOME); \
	export FAISS_BUILD_JOBS=$(FAISS_BUILD_JOBS); \
	export FAISS_ENABLE_CUVS=$(FAISS_ENABLE_CUVS); \
	export PYTHON=$(PYTHON); \
	bash $(GPU_CU_SCRIPTS)/build_lib_x86_64.sh

build-pkg: build-lib
	@export CUDA_ARCHS="$(CUDA_ARCHS)"; \
	export CUDA_HOME=$(CUDA_HOME); \
	export FAISS_BUILD_JOBS=$(FAISS_BUILD_JOBS); \
	export FAISS_ENABLE_CUVS=$(FAISS_ENABLE_CUVS); \
	export PYTHON=$(PYTHON); \
	bash $(GPU_CU_SCRIPTS)/build_pkg_x86_64.sh

wheel: build-pkg
	@export PYTHON=$(PYTHON); \
	export CUDA_ARCHS="$(CUDA_ARCHS)"; \
	export FAISS_VARIANT=$(FAISS_VARIANT); \
	bash $(GPU_CU_SCRIPTS)/package_wheel_x86_64.sh

wheel-only:
	@export PYTHON=$(PYTHON); \
	export CUDA_ARCHS="$(CUDA_ARCHS)"; \
	export FAISS_VARIANT=$(FAISS_VARIANT); \
	bash $(GPU_CU_SCRIPTS)/package_wheel_x86_64.sh

check:
	@export CUDA_ARCHS=$(CUDA_ARCHS); \
	export CUDA_HOME=$(CUDA_HOME); \
	export PYTHON=$(PYTHON); \
	bash $(GPU_CU_SCRIPTS)/build_wheel_x86_64.sh check

# aarch64 / DGX Spark targets (OpenBLAS + cuVS). The aarch64 orchestrator
# manages its own variant, build dirs, and prerequisite checks.
build-aarch64:
	@export FAISS_BUILD_JOBS=$(FAISS_BUILD_JOBS); \
	export FAISS_ENABLE_CUVS=$(FAISS_ENABLE_CUVS); \
	export PYTHON=$(PYTHON); \
	bash $(GPU_CU_SCRIPTS)/build_wheel_aarch64.sh all

check-aarch64:
	@export PYTHON=$(PYTHON); \
	bash $(GPU_CU_SCRIPTS)/build_wheel_aarch64.sh check

clean:
	@bash $(GPU_CU_SCRIPTS)/clean_build.sh

env-create:
	@echo "Creating conda environment: faiss-gpu-cu132-py314"
	conda env create -f $(GPU_CU_ROOT)/environment.yml
	@echo ""
	@echo "To activate, run:"
	@echo "  conda activate faiss-gpu-cu132-py314"

env-info:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Build Configuration"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@. $(GPU_CU_SCRIPTS)/cuda_env.sh; \
	 echo "  CUDA version:      $$FAISS_CUDA_VER (tag $$FAISS_CUDA_TAG, wheel faiss-gpu-$$FAISS_CUDA_TAG)"; \
	 echo "  CUDA_HOME:         $$CUDA_HOME  (resolved)"
	@echo "  CUDA_ARCHS:        $(CUDA_ARCHS)"
	@echo "  FAISS_BUILD_JOBS:  $(FAISS_BUILD_JOBS)"
	@echo "  PYTHON:            $(PYTHON)"
	@echo "  Python version:    $$($(PYTHON) --version 2>&1)"
	@echo ""
	@echo "  GPU architectures:"
	@echo "    75 = RTX 2080, RTX 2060 (Turing)"
	@echo "    80 = A100, RTX 3090 (Ampere)"
	@echo "    86 = RTX 3080 Ti, RTX 3070 (Ampere)"
	@echo "    89 = RTX 4090, RTX 4080 (Ada)"
	@echo "    90 = H100 (Hopper)"
	@echo "   120 = GB200, B200, RTX 5090 (Blackwell)"
	@echo "   121 = DGX Spark GB10 Grace Blackwell (aarch64 — use build-aarch64)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test:
	@export PYTHONPATH=$$(find . -path ./build_output -prune -o -name "build/lib*" -type d -print); \
	echo "Running FAISS tests..."; \
	$(PYTHON) -m pytest tests/ -v 2>/dev/null || echo "pytest not found. Install with: pip install pytest"

install-wheel:
	@wheel=$$(ls -t build_output/faiss_gpu*.whl 2>/dev/null | head -1); \
	if [ -z "$$wheel" ]; then \
		echo "ERROR: No wheel found in build_output/"; \
		exit 1; \
	fi; \
	echo "Installing $$wheel"; \
	$(PYTHON) -m pip install "$$wheel"
	@echo "Verifying installation..."; \
	$(PYTHON) -c "import faiss; print('✓ FAISS installed'); print('  Version:', faiss.__version__); print('  GPU devices:', faiss.gpuGetNumDevices())"

# Convenient shorthand targets
.PHONY: lib pkg all
lib: build-lib
pkg: build-pkg
all: build

# Phony targets
.PHONY: clean help build build-lib build-pkg wheel check env-create env-info test
