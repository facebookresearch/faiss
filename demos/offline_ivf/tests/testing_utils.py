# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import yaml
import numpy as np
from typing import Dict, List, Optional

OIVF_TEST_ARGS: List[str] = [
    "--config",
    "--xb",
    "--xq",
    "--command",
    "--cluster_run",
    "--no_residuals",
]


def get_test_parser(args) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument(arg)
    return parser


class TestDataCreator:
    def __init__(
        self,
        tempdir: str,
        dimension: int,
        data_type: np.dtype,
        index_factory: Optional[List] = ["OPQ4,IVF256,PQ4"],
        training_sample: Optional[int] = 9984,
        index_shard_size: Optional[int] = 1000,
        query_batch_size: Optional[int] = 1000,
        evaluation_sample: Optional[int] = 100,
        num_files: Optional[int] = None,
        file_size: Optional[int] = None,
        file_sizes: Optional[List] = None,
        nprobe: Optional[int] = 64,
        k: Optional[int] = 10,
        metric: Optional[str] = "METRIC_L2",
        normalise: Optional[bool] = False,
        with_queries_ds: Optional[bool] = False,
        evaluate_by_margin: Optional[bool] = False,
    ) -> None:
        self.tempdir = tempdir
        self.dimension = dimension
        self.data_type = np.dtype(data_type).name
        self.index_factory = {"prod": index_factory}
        if file_size and num_files:
            self.file_sizes = [file_size for _ in range(num_files)]
        elif file_sizes:
            self.file_sizes = file_sizes
        else:
            raise ValueError("no file sizes provided")
        self.num_files = len(self.file_sizes)
        self.training_sample = training_sample
        self.index_shard_size = index_shard_size
        self.query_batch_size = query_batch_size
        self.evaluation_sample = evaluation_sample
        self.nprobe = {"prod": [nprobe]}
        self.k = k
        self.metric = metric
        self.normalise = normalise
        self.config_file = self.tempdir + "/config_test.yaml"
        self.ds_name = "my_test_data"
        self.qs_name = "my_queries_data"
        self.evaluate_by_margin = evaluate_by_margin
        self.with_queries_ds = with_queries_ds

    def create_test_data(self) -> None:
        datafiles = self._create_data_files()
        files_info = []

        for i, file in enumerate(datafiles):
            files_info.append(
                {
                    "dtype": self.data_type,
                    "format": "npy",
                    "name": file,
                    "size": self.file_sizes[i],
                }
            )

        config_for_yaml = {
            "d": self.dimension,
            "output": self.tempdir,
            "index": self.index_factory,
            "nprobe": self.nprobe,
            "k": self.k,
            "normalise": self.normalise,
            "metric": self.metric,
            "training_sample": self.training_sample,
            "evaluation_sample": self.evaluation_sample,
            "index_shard_size": self.index_shard_size,
            "query_batch_size": self.query_batch_size,
            "datasets": {
                self.ds_name: {
                    "root": self.tempdir,
                    "size": sum(self.file_sizes),
                    "files": files_info,
                }
            },
        }
        if self.evaluate_by_margin:
            config_for_yaml["evaluate_by_margin"] = self.evaluate_by_margin
        q_datafiles = self._create_data_files("my_q_data")
        q_files_info = []

        for i, file in enumerate(q_datafiles):
            q_files_info.append(
                {
                    "dtype": self.data_type,
                    "format": "npy",
                    "name": file,
                    "size": self.file_sizes[i],
                }
            )
        if self.with_queries_ds:
            config_for_yaml["datasets"][self.qs_name] = {
                "root": self.tempdir,
                "size": sum(self.file_sizes),
                "files": q_files_info,
            }

        self._create_config_yaml(config_for_yaml)

    def setup_cli(self, command="consistency_check") -> argparse.Namespace:
        parser = get_test_parser(OIVF_TEST_ARGS)

        if self.with_queries_ds:
            return parser.parse_args(
                [
                    "--xb",
                    self.ds_name,
                    "--config",
                    self.config_file,
                    "--command",
                    command,
                    "--xq",
                    self.qs_name,
                ]
            )
        return parser.parse_args(
            [
                "--xb",
                self.ds_name,
                "--config",
                self.config_file,
                "--command",
                command,
            ]
        )

    def _create_data_files(self, name_of_file="my_data") -> List[str]:
        """
        Creates a dataset "my_test_data" with number of files (num_files), using padding in the files
        name. If self.with_queries is True, it adds an extra dataset "my_queries_data" with the same number of files
        as the "my_test_data". The default name for embeddings files is "my_data" + <padding>.npy.
        """
        filenames = []
        for i, file_size in enumerate(self.file_sizes):
            # np.random.seed(i)
            db_vectors = np.random.random((file_size, self.dimension)).astype(
                self.data_type
            )
            filename = name_of_file + f"{i:02}" + ".npy"
            filenames.append(filename)
            np.save(self.tempdir + "/" + filename, db_vectors)
        return filenames

    def _create_config_yaml(self, dict_file: Dict[str, str]) -> None:
        """
        Creates a yaml file in dir (can be a temporary dir for tests).
        """
        filename = self.tempdir + "/config_test.yaml"
        with open(filename, "w") as file:
            yaml.dump(dict_file, file, default_flow_style=False)
