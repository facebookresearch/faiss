# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# C++ benchmark suite (Google Benchmark). The Python suite in python/ is
# driven by pytest and needs no CMake integration.
add_subdirectory(cpp)

# GPU benchmark suite (Google Benchmark). Added after cpp/ so it can reuse the
# FetchContent'd googlebenchmark target and the `benchmarks` meta-target; the
# subdirectory no-ops when FAISS_ENABLE_GPU=OFF.
add_subdirectory(gpu)
