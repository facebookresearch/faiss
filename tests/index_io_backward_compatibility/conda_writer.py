# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Conda Writer: Tests serialization of all Faiss index types using conda-built
Faiss.
Creates index files that will be read by conda-built Faiss to test index
serialization backward compatibility.
"""

import sys
from common_io import write_test_all_files


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python conda_writer.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]
    exit_code = write_test_all_files(
        writer="conda", output_dir=output_dir, seed=5678
    )
    sys.exit(exit_code)
