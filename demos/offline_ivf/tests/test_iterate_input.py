# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
from typing import List
from utils import load_config
from tests.testing_utils import TestDataCreator
import tempfile
from dataset import create_dataset_from_oivf_config

DIMENSION: int = 768
SMALL_FILE_SIZES: List[int] = [100, 210, 450]
TEST_BATCH_SIZE: int = 500
SMALL_SAMPLE_SIZE: int = 1000
NUM_FILES: int = 3


class TestUtilsMethods(unittest.TestCase):

    def test_iterate_input_file_smaller_than_batch(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=DIMENSION,
                data_type=np.float16,
                file_sizes=SMALL_FILE_SIZES,
            )
            data_creator.create_test_data()
            args = data_creator.setup_cli()
            cfg = load_config(args.config)
            db_iterator = create_dataset_from_oivf_config(cfg, args.xb).iterate(
                0, TEST_BATCH_SIZE, np.float32
            )

            for i in range(len(SMALL_FILE_SIZES) - 1):
                vecs = next(db_iterator)
                if i != 1:
                    self.assertEqual(vecs.shape[0], TEST_BATCH_SIZE)
                else:
                    self.assertEqual(
                        vecs.shape[0], sum(SMALL_FILE_SIZES) - TEST_BATCH_SIZE
                    )

    def test_get_vs_iterate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=DIMENSION,
                data_type=np.float16,
                file_size=SMALL_SAMPLE_SIZE,
                num_files=NUM_FILES,
                normalise=True,
            )
            data_creator.create_test_data()
            args = data_creator.setup_cli()
            cfg = load_config(args.config)
            ds = create_dataset_from_oivf_config(cfg, args.xb)
            vecs_by_iterator = np.vstack(list(ds.iterate(0, 317, np.float32)))
            self.assertEqual(
                vecs_by_iterator.shape[0], SMALL_SAMPLE_SIZE * NUM_FILES
            )
            vecs_by_get = ds.get(list(range(vecs_by_iterator.shape[0])))
            self.assertTrue(np.all(vecs_by_iterator == vecs_by_get))
            vecs_chunk = np.vstack(
                [
                    next(ds.iterate(i, 543, np.float32))
                    for i in range(0, SMALL_SAMPLE_SIZE * NUM_FILES, 543)
                ]
            )
            self.assertTrue(np.all(vecs_by_iterator == vecs_chunk))
