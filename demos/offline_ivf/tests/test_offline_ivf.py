# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
from utils import load_config
import pathlib as pl
import tempfile
from typing import List
from tests.testing_utils import TestDataCreator
from run import process_options_and_run_jobs

KNN_RESULTS_FILE: str = (
    "/my_test_data_in_my_test_data/knn/I0000000000_IVF256_PQ4_np2.npy"
)

A_INDEX_FILES: List[str] = [
    "I_a_gt.npy",
    "D_a_gt.npy",
    "vecs_a.npy",
    "D_a_ann_IVF256_PQ4_np2.npy",
    "I_a_ann_IVF256_PQ4_np2.npy",
    "D_a_ann_refined_IVF256_PQ4_np2.npy",
]

A_INDEX_OPQ_FILES: List[str] = [
    "I_a_gt.npy",
    "D_a_gt.npy",
    "vecs_a.npy",
    "D_a_ann_OPQ4_IVF256_PQ4_np200.npy",
    "I_a_ann_OPQ4_IVF256_PQ4_np200.npy",
    "D_a_ann_refined_OPQ4_IVF256_PQ4_np200.npy",
]


class TestOIVF(unittest.TestCase):

    def assert_file_exists(self, filepath: str) -> None:
        path = pl.Path(filepath)
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_consistency_check(self) -> None:
        """
        Test the OIVF consistency check step, that it throws if no other steps
        have been ran.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float16,
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("consistency_check")
            self.assertRaises(
                AssertionError, process_options_and_run_jobs, test_args
            )

    def test_index_shard_unequal_file_sizes(self) -> None:
        """
        Test the case where the shard size is not a divisor of the database
        size and is greater than the first file size.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_sizes = [20000, 15001, 13990]
            index_shard_size = 30000
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float16,
                index_factory=["IVF256,PQ4"],
                file_sizes=file_sizes,
                nprobe=2,
                k=2,
                index_shard_size=index_shard_size,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("index_shard")
            cfg = load_config(test_args.config)
            process_options_and_run_jobs(test_args)
            num_shards = 2
            for i in range(num_shards):
                index_shard_file = (
                    cfg["output"]
                    + "/my_test_data/"
                    + cfg["index"]["prod"][-1].replace(",", "_")
                    + f".shard_{i}"
                )
                self.assert_file_exists(index_shard_file)

    def test_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            query_batch_size = 10000
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
                index_shard_size=10000,
                query_batch_size=query_batch_size,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("index_shard")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("search")
            cfg = load_config(test_args.config)
            process_options_and_run_jobs(test_args)
            knn_file = cfg["output"] + KNN_RESULTS_FILE
            self.assert_file_exists(knn_file)

    def test_evaluate_without_margin(self) -> None:
        """
        Test evaluate step using test data objects, no margin evaluation,
        single index.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
                index_shard_size=10000,
                query_batch_size=10000,
                with_queries_ds=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("index_shard")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("merge_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("evaluate")
            process_options_and_run_jobs(test_args)
            common_path = tmpdirname + "/my_queries_data_in_my_test_data/eval/"
            for filename in A_INDEX_FILES:
                file_to_check = common_path + "/" + filename
                self.assert_file_exists(file_to_check)

    def test_evaluate_without_margin_OPQ(self) -> None:
        """Same as test_evaluate_without_margin, with an OPQ pretransform."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["OPQ4,IVF256,PQ4"],
                num_files=3,
                file_size=10000,
                nprobe=200,
                k=2,
                index_shard_size=10000,
                query_batch_size=10000,
                with_queries_ds=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("index_shard")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("merge_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("evaluate")
            process_options_and_run_jobs(test_args)
            common_path = tmpdirname + "/my_queries_data_in_my_test_data/eval/"
            for filename in A_INDEX_OPQ_FILES:
                file_to_check = common_path + filename
                self.assert_file_exists(file_to_check)
