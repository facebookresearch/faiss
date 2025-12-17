# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Conda Reader: Tests deserialization of cmake-written index files using
conda-built Faiss.
Validates that all index types written by cmake can be loaded and searched
without crashing.
"""

import sys
from common_io import read_test_all_files


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python conda_reader.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    exit_code = read_test_all_files(
        reader="conda", writer="cmake", input_dir=input_dir
    )
    sys.exit(exit_code)
