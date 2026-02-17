# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
CMake Reader: Tests deserialization of conda-written index files using
cmake-built Faiss.
Validates that all index types written by conda can be loaded and searched
without crashing.
"""

import sys
from common_io import read_test_all_files


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cmake_reader.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    exit_code = read_test_all_files(
        reader="cmake", writer="conda", input_dir=input_dir
    )
    sys.exit(exit_code)
