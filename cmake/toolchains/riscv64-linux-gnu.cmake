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

set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# Cross-compiler sysroot provided by gcc-riscv64-linux-gnu.
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)

# Never look for host-side tools (cmake, python, …) inside the sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Look for target libraries/headers/packages only inside the sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
