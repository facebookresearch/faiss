# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import numpy as np

from dataset import create_dataset_from_oivf_config


class TestDatasetFormats(unittest.TestCase):
    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as directory:
            np.zeros((2, 4), dtype=np.float32).tofile(
                os.path.join(directory, "vectors.bin")
            )
            config = {
                "d": 4,
                "datasets": {
                    "database": {
                        "root": directory,
                        "size": 2,
                        "files": [
                            {
                                "name": "vectors.bin",
                                "format": "unsupported",
                                "dtype": "float32",
                                "size": 2,
                            }
                        ],
                    }
                },
            }
            with self.assertRaisesRegex(ValueError, "current format"):
                create_dataset_from_oivf_config(config, "database")
