# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Cross-compilation toolchain for RISC-V 64-bit (lp64d ABI) on Ubuntu 24.04.
#
# Requires:
#   gcc-14-riscv64-linux-gnu
#   g++-14-riscv64-linux-gnu
#
# Target libraries:
#   libopenblas-dev:riscv64
#
# Ubuntu multiarch installs target libraries into:
#   /usr/lib/riscv64-linux-gnu
#
# The CI creates:
#   /usr/riscv64-linux-gnu/usr/lib/riscv64-linux-gnu
#       -> /usr/lib/riscv64-linux-gnu
#
# so CMake can locate target libraries while using ONLY root mode.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# GCC 14 cross compiler
set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc-14)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++-14)

# Do NOT set CMAKE_SYSROOT here. Ubuntu's cross toolchain resolves target
# libraries through its built-in paths (/usr/riscv64-linux-gnu/lib). Passing
# --sysroot=/usr/riscv64-linux-gnu makes ld prepend the sysroot to the
# absolute paths inside libc6-dev-riscv64-cross's libc.so linker script,
# producing doubled paths such as
#   /usr/riscv64-linux-gnu/usr/riscv64-linux-gnu/lib/libc.so.6
# and failing the CMake compiler check at link time.

set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)

# Never search host binaries inside sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Target libraries/headers/packages only.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
