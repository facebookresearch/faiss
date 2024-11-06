# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss  # @manual=//faiss/python:pyfaiss

from .benchmark_io import BenchmarkIO
from .utils import timer

logger = logging.getLogger(__name__)


# Important: filenames end with . without extension (npy, codec, index),
# when writing files, you are required to filename + "npy" etc.

@dataclass
class IndexDescriptorClassic:
    bucket: Optional[str] = None
    # either path or factory should be set,
    # but not both at the same time.
    path: Optional[str] = None
    factory: Optional[str] = None
    codec_alias: Optional[str] = None
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
    training_size: Optional[int] = None

    def __hash__(self):
        return hash(str(self))

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

    embedding_column: Optional[str] = None

    embedding_id_column: Optional[str] = None

    sampling_rate: Optional[float] = None

    # sampling column for xdb
    sampling_column: Optional[str] = None

    # blob store
    bucket: Optional[str] = None
    path: Optional[str] = None

    # desc_name
    desc_name: Optional[str] = None

    def __hash__(self):
        return hash(self.get_filename())

    def get_filename(
        self,
        prefix: Optional[str] = None,
    ) -> str:
        if self.desc_name is not None:
            return self.desc_name

        filename = ""
        if prefix is not None:
            filename += prefix + "_"
        if self.namespace is not None:
            filename += self.namespace + "_"
        assert self.tablename is not None
        filename += self.tablename
        if self.partitions is not None:
            filename += "_" + "_".join(
                self.partitions
            ).replace("=", "_").replace("/", "_")
        if self.num_vectors is not None:
            filename += f"_{self.num_vectors}"
        filename += "."

        self.desc_name = filename
        return self.desc_name

    def get_kmeans_filename(self, k):
        return f"{self.get_filename()}kmeans_{k}."

    def k_means(self, io, k, dry_run):
        logger.info(f"k_means {k} {self}")
        kmeans_vectors = DatasetDescriptor(
            tablename=f"{self.get_filename()}kmeans_{k}"
        )
        kmeans_filename = kmeans_vectors.get_filename() + "npy"
        meta_filename = kmeans_vectors.get_filename() + "json"
        if not io.file_exist(kmeans_filename) or not io.file_exist(
            meta_filename
        ):
            if dry_run:
                return None, None, kmeans_filename
            x = io.get_dataset(self)
            kmeans = faiss.Kmeans(d=x.shape[1], k=k, gpu=True)
            _, t, _ = timer("k_means", lambda: kmeans.train(x))
            io.write_nparray(kmeans.centroids, kmeans_filename)
            io.write_json({"k_means_time": t}, meta_filename)
        else:
            t = io.read_json(meta_filename)["k_means_time"]
        return kmeans_vectors, t, None

@dataclass
class IndexBaseDescriptor:
    d: int
    metric: str
    desc_name: Optional[str] = None
    flat_desc_name: Optional[str] = None
    bucket: Optional[str] = None
    path: Optional[str] = None
    num_threads: int = 1

    def get_name(self) -> str:
        raise NotImplementedError()

    def get_path(self, benchmark_io: BenchmarkIO) -> Optional[str]:
        if self.path is not None:
            return self.path
        self.path = benchmark_io.get_remote_filepath(self.desc_name)
        return self.path

    @staticmethod
    def param_dict_list_to_name(param_dict_list):
        if not param_dict_list:
            return ""
        l = 0
        n = ""
        for param_dict in param_dict_list:
            n += IndexBaseDescriptor.param_dict_to_name(param_dict, f"cp{l}")
            l += 1
        return n

    @staticmethod
    def param_dict_to_name(param_dict, prefix="sp"):
        if not param_dict:
            return ""
        n = prefix
        for name, val in param_dict.items():
            if name == "snap":
                continue
            if name == "lsq_gpu" and val == 0:
                continue
            if name == "use_beam_LUT" and val == 0:
                continue
            n += f"_{name}_{val}"
        if n == prefix:
            return ""
        n += "."
        return n


@dataclass
class CodecDescriptor(IndexBaseDescriptor):
    # either path or factory should be set,
    # but not both at the same time.
    factory: Optional[str] = None
    construction_params: Optional[List[Dict[str, int]]] = None
    training_vectors: Optional[DatasetDescriptor] = None

    def __post_init__(self):
        self.get_name()

    def is_trained(self):
        return self.factory is None and self.path is not None

    def is_valid(self):
        return self.factory is not None or self.path is not None

    def get_name(self) -> str:
        if self.desc_name is not None:
            return self.desc_name
        if self.factory is not None:
            self.desc_name = self.name_from_factory()
            return self.desc_name
        if self.path is not None:
            self.desc_name = self.name_from_path()
            return self.desc_name
        raise ValueError("name, factory or path must be set")

    def flat_name(self) -> str:
        if self.flat_desc_name is not None:
            return self.flat_desc_name
        self.flat_desc_name = f"Flat.d_{self.d}.{self.metric.upper()}."
        return self.flat_desc_name

    def path(self, benchmark_io) -> str:
        if self.path is not None:
            return self.path
        return benchmark_io.get_remote_filepath(self.get_name())

    def name_from_factory(self) -> str:
        assert self.factory is not None
        name = f"{self.factory.replace(',', '_')}."
        assert self.d is not None
        assert self.metric is not None
        name += f"d_{self.d}.{self.metric.upper()}."
        if self.factory != "Flat":
            assert self.training_vectors is not None
            name += self.training_vectors.get_filename("xt")
        name += IndexBaseDescriptor.param_dict_list_to_name(self.construction_params)
        return name

    def name_from_path(self):
        assert self.path is not None
        filename = os.path.basename(self.path)
        ext = filename.split(".")[-1]
        if filename.endswith(ext):
            name = filename[:-len(ext)]
        else: # should never hit this rather raise value error
            name = filename
        return name

    def alias(self, benchmark_io: BenchmarkIO):
        if hasattr(benchmark_io, "bucket"):
            return CodecDescriptor(desc_name=self.get_name(), bucket=benchmark_io.bucket, path=self.get_path(benchmark_io), d=self.d, metric=self.metric)
        return CodecDescriptor(desc_name=self.get_name(), d=self.d, metric=self.metric)


@dataclass
class IndexDescriptor(IndexBaseDescriptor):
    codec_desc: Optional[CodecDescriptor] = None
    database_desc: Optional[DatasetDescriptor] = None

    def __hash__(self):
        return hash(str(self))

    def __post_init__(self):
        self.get_name()

    def is_built(self):
        return self.codec_desc is None and self.database_desc is None

    def get_name(self) -> str:
        if self.desc_name is None:
            self.desc_name = self.codec_desc.get_name() + self.database_desc.get_filename(prefix="xb")

        return self.desc_name

    def flat_name(self):
        if self.flat_desc_name is not None:
            return self.flat_desc_name
        self.flat_desc_name = self.codec_desc.flat_name() + self.database_desc.get_filename(prefix="xb")
        return self.flat_desc_name

    # alias is used to refer when index is uploaded to blobstore and refered again
    def alias(self, benchmark_io: BenchmarkIO):
        if hasattr(benchmark_io, "bucket"):
            return IndexDescriptor(desc_name=self.get_name(), bucket=benchmark_io.bucket, path=self.get_path(benchmark_io), d=self.d, metric=self.metric)
        return IndexDescriptor(desc_name=self.get_name(), d=self.d, metric=self.metric)

@dataclass
class KnnDescriptor(IndexBaseDescriptor):
    index_desc: Optional[IndexDescriptor] = None
    gt_index_desc: Optional[IndexDescriptor] = None
    query_dataset: Optional[DatasetDescriptor] = None
    search_params: Optional[Dict[str, int]] = None
    reconstruct: bool = False
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
    k: int = 1

    range_ref_index_desc: Optional[str] = None

    def __hash__(self):
        return hash(str(self))

    def get_name(self):
        name = self.index_desc.get_name()
        name += IndexBaseDescriptor.param_dict_to_name(self.search_params)
        name += self.query_dataset.get_filename("q")
        name += f"k_{self.k}."
        name += f"t_{self.num_threads}."
        if self.reconstruct:
            name += "rec."
        else:
            name += "knn."
        return name

    def flat_name(self):
        if self.flat_desc_name is not None:
            return self.flat_desc_name
        name = self.index_desc.flat_name()
        name += self.query_dataset.get_filename("q")
        name += f"k_{self.k}."
        name += f"t_{self.num_threads}."
        if self.reconstruct:
            name += "rec."
        else:
            name += "knn."
        self.flat_desc_name = name
        return name
