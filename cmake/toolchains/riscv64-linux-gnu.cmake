# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Cross-compilation toolchain for RISC-V 64-bit (lp64d ABI) on Ubuntu/Debian.
#
# Requires these packages installed on the build host:
#   gcc-riscv64-linux-gnu  g++-riscv64-linux-gnu
#
# Target libraries (e.g. libopenblas-dev:riscv64) are installed via apt
# multiarch to /usr/lib/riscv64-linux-gnu/.  CMake's ONLY find-root mode
# searches ${CMAKE_FIND_ROOT_PATH}/usr/lib/riscv64-linux-gnu/ (via the
# compiler's multiarch tuple), so the CI script creates a symlink:
#   /usr/riscv64-linux-gnu/usr/lib/riscv64-linux-gnu
#     -> /usr/lib/riscv64-linux-gnu
# before invoking cmake.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# GCC 14+: the RVV SQ kernels rely on tuple types (vuint8m1x3_t) and
# zvfhmin f16 intrinsics that GCC 13's <riscv_vector.h> does not export.
set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc-14)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++-14)

# Cross-compiler sysroot provided by gcc-riscv64-linux-gnu.
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)

# Never look for host-side tools (cmake, python, …) inside the sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Look for target libraries/headers/packages only inside the sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
