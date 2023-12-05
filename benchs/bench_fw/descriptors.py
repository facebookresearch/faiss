# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IndexDescriptor:
    bucket: Optional[str] = None
    # either path or factory should be set,
    # but not both at the same time.
    path: Optional[str] = None
    factory: Optional[str] = None
    construction_params: Optional[List[Dict[str, int]]] = None
    search_params: Optional[Dict[str, int]] = None
    # range metric definitions
    # key: name
    # value: one of the following:
    #
    # radius
    #    [0..radius) -> 1
    #    [radius..inf) -> 0
    #
    # [[radius1, score1], ...]
    #    [0..radius1) -> score1
    #    [radius1..radius2) -> score2
    #
    # [[radius1_from, radius1_to, score1], ...]
    #    [radius1_from, radius1_to) -> score1,
    #    [radius2_from, radius2_to) -> score2
    range_metrics: Optional[Dict[str, Any]] = None
    radius: Optional[float] = None


@dataclass
class DatasetDescriptor:
    # namespace possible values:
    # 1. a hive namespace
    # 2. 'std_t', 'std_d', 'std_q' for the standard datasets
    #    via faiss.contrib.datasets.dataset_from_name()
    #    t - training, d - database, q - queries
    #    eg. "std_t"
    # 3. 'syn' for synthetic data
    # 4. None for local files
    namespace: Optional[str] = None

    # tablename possible values, corresponding to the
    # namespace value above:
    # 1. a hive table name
    # 2. name of the standard dataset as recognized
    #    by faiss.contrib.datasets.dataset_from_name()
    #    eg. "bigann1M"
    # 3. d_seed, eg. 128_1234 for 128 dimensional vectors
    #    with seed 1234
    # 4. a local file name (relative to benchmark_io.path)
    tablename: Optional[str] = None

    # partition names and values for hive
    # eg. ["ds=2021-09-01"]
    partitions: Optional[List[str]] = None

    # number of vectors to load from the dataset
    num_vectors: Optional[int] = None

    def __hash__(self):
        return hash(self.get_filename())

    def get_filename(
        self,
        prefix: str = None,
    ) -> str:
        filename = ""
        if prefix is not None:
            filename += prefix + "_"
        if self.namespace is not None:
            filename += self.namespace + "_"
        assert self.tablename is not None
        filename += self.tablename
        if self.partitions is not None:
            filename += "_" + "_".join(self.partitions).replace("=", "_")
        if self.num_vectors is not None:
            filename += f"_{self.num_vectors}"
        filename += "."
        return filename
