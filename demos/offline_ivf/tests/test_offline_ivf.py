# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
from utils import load_config
from offline_ivf import OfflineIVF
import pathlib as pl
import tempfile
import shutil
import os
from typing import List
from tests.testing_utils import TestDataCreator
from run import process_options_and_run_jobs

INDEX_TEMPLATE_FILE: str = "/tests/test_data/IVF256_PQ4.empty.faissindex"
OPQ_INDEX_TEMPLATE_FILE: str = (
    "/tests/test_data/OPQ4_IVF256_PQ4.empty.faissindex"
)
KNN_RESULTS_FILE: str = (
    "/my_test_data_in_my_test_data/knn/I0000000000_IVF256_PQ4_np2.npy"
)
TEST_INDEX_A: str = "/tests/test_data/goku_lang/IVF256_PQ4.faissindex"
TEST_INDEX_DATA_A: str = (
    "/tests/test_data/goku_lang/IVF256_PQ4.faissindex.ivfdata"
)
TEST_INDEX_B: str = "/tests/test_data/coco_lang/IVF256_PQ4.faissindex"
TEST_INDEX_DATA_B: str = (
    "/tests/test_data/coco_lang/IVF256_PQ4.faissindex.ivfdata"
)
TEST_INDEX_OPQ: str = "/tests/test_data/goku_lang/OPQ4_IVF256_PQ4.faissindex"
TEST_INDEX_DATA_OPQ: str = (
    "/tests/test_data/goku_lang/OPQ4_IVF256_PQ4.faissindex.ivfdata"
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

B_INDEX_FILES: List[str] = [
    "I_b_gt.npy",
    "vecs_b_gt.npy",
    "vecs_b_ann_IVF256_PQ4_np2.npy",
    "D_b_ann_gt_IVF256_PQ4_np2.npy",
    "I_b_ann_gt_IVF256_PQ4_np2.npy",
    "margin_refined_IVF256_PQ4_np2.npy",
    "idx_b_ann_IVF256_PQ4_np2.npy",
]


class TestOIVF(unittest.TestCase):
    """
    Unit tests for OIVF. Some of these unit tests first copy the required test data objects and puts them in the tempdir created by the context manager.
    """

    def assert_file_exists(self, filepath: str) -> None:
        path = pl.Path(filepath)
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_consistency_check(self) -> None:
        """
        Test the OIVF consistency check step, that it throws if no other steps have been ran.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float16,
                index_factory=["OPQ4,IVF256,PQ4"],
                training_sample=9984,
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("consistency_check")
            self.assertRaises(
                AssertionError, process_options_and_run_jobs, test_args
            )

    def test_train_index(self) -> None:
        """
        Test the OIVF train index step, that it correctly produces the empty.faissindex template file.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float16,
                index_factory=["OPQ4,IVF256,PQ4"],
                training_sample=9984,
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            cfg = load_config(test_args.config)
            process_options_and_run_jobs(test_args)
            empty_index = (
                cfg["output"]
                + "/my_test_data/"
                + cfg["index"]["prod"][-1].replace(",", "_")
                + ".empty.faissindex"
            )
            self.assert_file_exists(empty_index)

    def test_index_shard_equal_file_sizes(self) -> None:
        """
        Test the case where the shard size is a divisor of the database size and it is equal to the first file size.
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            test_index_path = os.getcwd() + INDEX_TEMPLATE_FILE
            new_path = f"{tmpdirname}/my_test_data/"
            os.makedirs(new_path, exist_ok=True)
            shutil.copy(test_index_path, new_path)
            index_shard_size = 10000
            num_files = 3
            file_size = 10000
            xb_ds_size = num_files * file_size
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float16,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                num_files=num_files,
                file_size=file_size,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
                index_shard_size=index_shard_size,
                query_batch_size=1000,
                evaluation_sample=100,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("index_shard")
            cfg = load_config(test_args.config)
            process_options_and_run_jobs(test_args)
            num_shards = xb_ds_size // index_shard_size
            if xb_ds_size % index_shard_size != 0:
                num_shards += 1
            print(f"number of shards:{num_shards}")
            for i in range(num_shards):
                index_shard_file = (
                    cfg["output"]
                    + "/my_test_data/"
                    + cfg["index"]["prod"][-1].replace(",", "_")
                    + f".shard_{i}"
                )
                self.assert_file_exists(index_shard_file)

    def test_index_shard_unequal_file_sizes(self) -> None:
        """
        Test the case where the shard size is not a divisor of the database size and is greater than the first file size.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_index_path = os.getcwd() + INDEX_TEMPLATE_FILE
            new_path = f"{tmpdirname}/my_test_data/"
            os.makedirs(new_path, exist_ok=True)
            shutil.copy(test_index_path, new_path)
            file_sizes = [20000, 15001, 13990]
            xb_ds_size = sum(file_sizes)
            index_shard_size = 30000
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float16,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                file_sizes=file_sizes,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
                index_shard_size=index_shard_size,
                evaluation_sample=100,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("index_shard")
            cfg = load_config(test_args.config)
            process_options_and_run_jobs(test_args)
            num_shards = xb_ds_size // index_shard_size
            if xb_ds_size % index_shard_size != 0:
                num_shards += 1
            print(f"number of shards:{num_shards}")
            for i in range(num_shards):
                index_shard_file = (
                    cfg["output"]
                    + "/my_test_data/"
                    + cfg["index"]["prod"][-1].replace(",", "_")
                    + f".shard_{i}"
                )
                self.assert_file_exists(index_shard_file)

    def test_search(self) -> None:
        """
        Test search step using test data objects to bypass dependencies on previous steps.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_index_path = os.getcwd() + INDEX_TEMPLATE_FILE
            new_path = f"{tmpdirname}/my_test_data/"
            os.makedirs(new_path, exist_ok=True)
            shutil.copy(test_index_path, new_path)
            os.makedirs(tmpdirname + "/my_test_data/", exist_ok=True)
            num_files = 3
            for i in range(num_files):
                test_index_shard = (
                    os.getcwd()
                    + f"/tests/test_data/goku_lang/IVF256_PQ4.shard_{i}"
                )
                shutil.copy(test_index_shard, tmpdirname + "/my_test_data/")
            file_size = 10000
            query_batch_size = 10000
            total_batches = num_files * file_size // query_batch_size
            if num_files * file_size % query_batch_size != 0:
                total_batches += 1
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
                index_shard_size=10000,
                query_batch_size=query_batch_size,
                evaluation_sample=100,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("search")
            cfg = load_config(test_args.config)
            process_options_and_run_jobs(test_args)
            # TODO: add check that there are number of batches total of files
            knn_file = cfg["output"] + KNN_RESULTS_FILE
            self.assert_file_exists(knn_file)

    def test_evaluate_without_margin(self) -> None:
        """
        Test evaluate step using test data objects, no margin evaluation, single index.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_index_path = os.getcwd() + INDEX_TEMPLATE_FILE
            new_path = f"{tmpdirname}/my_test_data/"
            os.makedirs(new_path, exist_ok=True)
            shutil.copy(test_index_path, new_path)
            goku_index_file = os.getcwd() + TEST_INDEX_A
            goku_index_data = os.getcwd() + TEST_INDEX_DATA_A
            shutil.copy(goku_index_file, new_path)
            shutil.copy(goku_index_data, new_path)
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
                index_shard_size=10000,
                query_batch_size=10000,
                evaluation_sample=100,
                with_queries_ds=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("evaluate")
            process_options_and_run_jobs(test_args)
            common_path = tmpdirname + "/my_queries_data_in_my_test_data/eval/"
            for filename in A_INDEX_FILES:
                file_to_check = common_path + filename
                self.assert_file_exists(file_to_check)

    def test_evaluate_without_margin_OPQ(self) -> None:
        """
        Test evaluate step using test data objects, no margin evaluation, single index.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_index_path = os.getcwd() + OPQ_INDEX_TEMPLATE_FILE
            new_path = f"{tmpdirname}/my_test_data/"
            os.makedirs(new_path, exist_ok=True)
            shutil.copy(test_index_path, new_path)
            goku_index_file = os.getcwd() + TEST_INDEX_OPQ
            goku_index_data = os.getcwd() + TEST_INDEX_DATA_OPQ
            shutil.copy(goku_index_file, new_path)
            shutil.copy(goku_index_data, new_path)

            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["OPQ4,IVF256,PQ4"],
                training_sample=9984,
                num_files=3,
                file_size=10000,
                nprobe=200,
                k=2,
                metric="METRIC_L2",
                index_shard_size=10000,
                query_batch_size=10000,
                evaluation_sample=100,
                with_queries_ds=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("evaluate")
            process_options_and_run_jobs(test_args)
            common_path = tmpdirname + "/my_queries_data_in_my_test_data/eval/"
            for filename in A_INDEX_OPQ_FILES:
                file_to_check = common_path + filename
                self.assert_file_exists(file_to_check)

    def test_evaluate_with_margin(self) -> None:
        """
        Test evaluate step using test data objects with margin evaluation, pair of indexes A and B case.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_index_path = os.getcwd() + INDEX_TEMPLATE_FILE
            new_path = f"{tmpdirname}/my_test_data/"
            os.makedirs(new_path, exist_ok=True)
            shutil.copy(test_index_path, new_path)
            goku_index_file = os.getcwd() + TEST_INDEX_A
            goku_index_data = os.getcwd() + TEST_INDEX_DATA_A
            shutil.copy(goku_index_file, new_path)
            shutil.copy(goku_index_data, new_path)
            coco_index_file = os.getcwd() + TEST_INDEX_B
            coco_index_data = os.getcwd() + TEST_INDEX_DATA_B
            queries_path = f"{tmpdirname}/my_queries_data/"
            os.makedirs(queries_path, exist_ok=True)
            shutil.copy(coco_index_file, queries_path)
            shutil.copy(coco_index_data, queries_path)

            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                num_files=3,
                file_size=10000,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
                index_shard_size=10000,
                query_batch_size=10000,
                evaluation_sample=100,
                with_queries_ds=True,
                evaluate_by_margin=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("evaluate")
            process_options_and_run_jobs(test_args)
            common_path = tmpdirname + "/my_queries_data_in_my_test_data/eval/"

            for filename in A_INDEX_FILES + B_INDEX_FILES:
                file_to_check = common_path + filename
                self.assert_file_exists(file_to_check)

    def test_split_batch_size_bigger_than_file_sizes(self) -> None:
        """
        Test split_files step, batch size bigger than file sizes.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_file_sizes = [19999, 20001, 30000, 10000]
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                file_sizes=test_file_sizes,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
                index_shard_size=10000,
                query_batch_size=40000,
                evaluation_sample=100,
                with_queries_ds=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("index_shard")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("search")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("split_files")
            process_options_and_run_jobs(test_args)

            common_path = tmpdirname + "/my_queries_data_in_my_test_data/knn/"
            I_groundtruth_files = [
                "I0000000000_IVF256_PQ4_np2.npy",
                "I0000040000_IVF256_PQ4_np2.npy",
            ]
            first_file_gt = I_groundtruth_files.pop(0)
            I_groundtruth = np.load(common_path + first_file_gt)
            for batched_file in I_groundtruth_files:
                I_groundtruth = np.vstack(
                    [I_groundtruth, np.load(common_path + batched_file)]
                )
            split_files = sorted(
                [
                    "mm5_p5.x2y.002.idx.npy",
                    "mm5_p5.x2y.003.idx.npy",
                    "mm5_p5.x2y.000.idx.npy",
                    "mm5_p5.x2y.001.idx.npy",
                ]
            )
            first_file = split_files.pop(0)
            output_path = (
                common_path
                + "dists5_p5.my_test_data-my_queries_data.IVF256_PQ4.k2.np2.fp32-shard/"
            )

            I_all_splits = np.load(output_path + first_file)
            for filename in split_files:
                self.assert_file_exists(output_path + filename)
                I = np.load(output_path + filename)
                I_all_splits = np.vstack([I_all_splits, I])

            self.assertTrue((I_all_splits == I_groundtruth).all())

    def test_split_batch_size_smaller_than_file_sizes(self) -> None:
        """
        Test split_files step, the batch size less than file sizes
        """
        test_file_sizes = [14995, 5005]
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                file_sizes=test_file_sizes,
                nprobe=2,
                k=2,
                metric="METRIC_L2",
                index_shard_size=10000,
                query_batch_size=5000,
                evaluation_sample=100,
                with_queries_ds=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("index_shard")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("search")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("split_files")
            process_options_and_run_jobs(test_args)

            common_path = tmpdirname + "/my_queries_data_in_my_test_data/knn/"
            I_groundtruth_files = [
                "D_approx0000000000_IVF256_PQ4_np2.npy",
                "D_approx0000005000_IVF256_PQ4_np2.npy",
                "D_approx0000010000_IVF256_PQ4_np2.npy",
                "D_approx0000015000_IVF256_PQ4_np2.npy",
            ]

            first_file_gt = I_groundtruth_files.pop(0)
            I_groundtruth = np.load(common_path + first_file_gt)
            for batched_file in I_groundtruth_files:
                I_groundtruth = np.vstack(
                    [I_groundtruth, np.load(common_path + batched_file)]
                )
            split_files = sorted(
                ["mm5_p5.x2y.000.dist.npy", "mm5_p5.x2y.001.dist.npy"]
            )
            output_path = (
                common_path
                + "dists5_p5.my_test_data-my_queries_data.IVF256_PQ4.k2.np2.fp32-shard/"
            )
            first_file = split_files.pop(0)
            I_all_splits = np.load(output_path + first_file)
            for filename in split_files:
                self.assert_file_exists(output_path + filename)
                I = np.load(output_path + filename)
                I_all_splits = np.vstack([I_all_splits, I])

            self.assertTrue((I_all_splits == I_groundtruth).all())

    def test_split_files_with_corrupted_input_file(self) -> None:
        """
        Test split_files step, the batch size less than file sizes
        """
        test_file_sizes = [14995, 5005]
        with tempfile.TemporaryDirectory() as tmpdirname:
            k = 2
            data_creator = TestDataCreator(
                tempdir=tmpdirname,
                dimension=8,
                data_type=np.float32,
                index_factory=["IVF256,PQ4"],
                training_sample=9984,
                file_sizes=test_file_sizes,
                nprobe=2,
                k=k,
                metric="METRIC_L2",
                index_shard_size=10000,
                query_batch_size=5000,
                evaluation_sample=100,
                with_queries_ds=True,
            )
            data_creator.create_test_data()
            test_args = data_creator.setup_cli("train_index")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("index_shard")
            process_options_and_run_jobs(test_args)
            test_args = data_creator.setup_cli("search")
            process_options_and_run_jobs(test_args)
            # Corrupts the last file
            common_path = tmpdirname + "/my_queries_data_in_my_test_data/knn/"
            D_corrupt_file = np.empty((0, k), dtype=np.float32)
            np.save(
                common_path + "D_approx0000015000_IVF256_PQ4_np2.npy",
                D_corrupt_file,
            )
            test_args = data_creator.setup_cli("split_files")

            self.assertRaises(
                AssertionError, process_options_and_run_jobs, test_args
            )
