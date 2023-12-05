# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from time import perf_counter
from typing import ClassVar, Dict, List, Optional

import faiss  # @manual=//faiss/python:pyfaiss_gpu

import numpy as np
from faiss.contrib.evaluation import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    OperatingPointsWithRanges,
)

from faiss.contrib.factory_tools import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    reverse_index_factory,
)
from faiss.contrib.ivf_tools import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    add_preassigned,
    replace_ivf_quantizer,
)

from .descriptors import DatasetDescriptor

logger = logging.getLogger(__name__)


def timer(name, func, once=False) -> float:
    logger.info(f"Measuring {name}")
    t1 = perf_counter()
    res = func()
    t2 = perf_counter()
    t = t2 - t1
    repeat = 1
    if not once and t < 1.0:
        repeat = int(2.0 // t)
        logger.info(
            f"Time for {name}: {t:.3f} seconds, repeating {repeat} times"
        )
        t1 = perf_counter()
        for _ in range(repeat):
            res = func()
        t2 = perf_counter()
        t = (t2 - t1) / repeat
    logger.info(f"Time for {name}: {t:.3f} seconds")
    return res, t, repeat


def refine_distances_knn(
    D: np.ndarray, I: np.ndarray, xq: np.ndarray, xb: np.ndarray, metric
):
    return np.where(
        I >= 0,
        np.square(np.linalg.norm(xq[:, None] - xb[I], axis=2))
        if metric == faiss.METRIC_L2
        else np.einsum("qd,qkd->qk", xq, xb[I]),
        D,
    )


def refine_distances_range(
    lims: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    xq: np.ndarray,
    xb: np.ndarray,
    metric,
):
    with ThreadPool(32) as pool:
        R = pool.map(
            lambda i: (
                np.sum(np.square(xq[i] - xb[I[lims[i]:lims[i + 1]]]), axis=1)
                if metric == faiss.METRIC_L2
                else np.tensordot(
                    xq[i], xb[I[lims[i]:lims[i + 1]]], axes=(0, 1)
                )
            )
            if lims[i + 1] > lims[i]
            else [],
            range(len(lims) - 1),
        )
    return np.hstack(R)


# The classes below are wrappers around Faiss indices, with different
# implementations for the case when we start with an already trained
# index (IndexFromCodec) vs factory strings (IndexFromFactory).
# In both cases the classes have operations for adding to an index
# and searching it, and outputs are cached on disk.
# IndexFromFactory also decomposes the index (pretransform and quantizer)
# and trains sub-indices independently.
class IndexBase:
    def set_io(self, benchmark_io):
        self.io = benchmark_io

    @staticmethod
    def param_dict_list_to_name(param_dict_list):
        if not param_dict_list:
            return ""
        l = 0
        n = ""
        for param_dict in param_dict_list:
            n += IndexBase.param_dict_to_name(param_dict, f"cp{l}")
        return n

    @staticmethod
    def param_dict_to_name(param_dict, prefix="sp"):
        if not param_dict:
            return ""
        n = prefix
        for name, val in param_dict.items():
            if name != "noop":
                n += f"_{name}_{val}"
        if n == prefix:
            return ""
        n += "."
        return n

    @staticmethod
    def set_index_param_dict_list(index, param_dict_list):
        if not param_dict_list:
            return
        index = faiss.downcast_index(index)
        for param_dict in param_dict_list:
            assert index is not None
            IndexBase.set_index_param_dict(index, param_dict)
            index = faiss.try_extract_index_ivf(index)

    @staticmethod
    def set_index_param_dict(index, param_dict):
        if not param_dict:
            return
        for name, val in param_dict.items():
            IndexBase.set_index_param(index, name, val)

    @staticmethod
    def set_index_param(index, name, val):
        index = faiss.downcast_index(index)

        if isinstance(index, faiss.IndexPreTransform):
            Index.set_index_param(index.index, name, val)
        elif name == "efSearch":
            index.hnsw.efSearch
            index.hnsw.efSearch = int(val)
        elif name == "efConstruction":
            index.hnsw.efConstruction
            index.hnsw.efConstruction = int(val)
        elif name == "nprobe":
            index_ivf = faiss.extract_index_ivf(index)
            index_ivf.nprobe
            index_ivf.nprobe = int(val)
        elif name == "k_factor":
            index.k_factor
            index.k_factor = int(val)
        elif name == "parallel_mode":
            index_ivf = faiss.extract_index_ivf(index)
            index_ivf.parallel_mode
            index_ivf.parallel_mode = int(val)
        elif name == "noop":
            pass
        else:
            raise RuntimeError(f"could not set param {name} on {index}")

    def is_flat(self):
        codec = faiss.downcast_index(self.get_model())
        return isinstance(codec, faiss.IndexFlat)

    def is_ivf(self):
        codec = self.get_model()
        return faiss.try_extract_index_ivf(codec) is not None

    def is_pretransform(self):
        codec = self.get_model()
        if isinstance(codec, faiss.IndexRefine):
            codec = faiss.downcast_index(codec.base_index)
        return isinstance(codec, faiss.IndexPreTransform)

    # index is a codec + database vectors
    # in other words: a trained Faiss index
    # that contains database vectors
    def get_index_name(self):
        raise NotImplementedError

    def get_index(self):
        raise NotImplementedError

    # codec is a trained model
    # in other words: a trained Faiss index
    # without any database vectors
    def get_codec_name(self):
        raise NotImplementedError

    def get_codec(self):
        raise NotImplementedError

    # model is an untrained Faiss index
    # it can be used for training (see codec)
    # or to inspect its structure
    def get_model_name(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def transform(self, vectors):
        transformed_vectors = DatasetDescriptor(
            tablename=f"{vectors.get_filename()}{self.get_codec_name()}transform.npy"
        )
        if not self.io.file_exist(transformed_vectors.tablename):
            codec = self.fetch_codec()
            assert isinstance(codec, faiss.IndexPreTransform)
            transform = faiss.downcast_VectorTransform(codec.chain.at(0))
            x = self.io.get_dataset(vectors)
            xt = transform.apply(x)
            self.io.write_nparray(xt, transformed_vectors.tablename)
        return transformed_vectors

    def knn_search_quantizer(self, index, query_vectors, k):
        if self.is_pretransform():
            pretransform = self.get_pretransform()
            quantizer_query_vectors = pretransform.transform(query_vectors)
        else:
            pretransform = None
            quantizer_query_vectors = query_vectors

        QD, QI, _, QP = self.get_quantizer(pretransform).knn_search(
            search_parameters=None,
            query_vectors=quantizer_query_vectors,
            k=k,
        )
        xqt = self.io.get_dataset(quantizer_query_vectors)
        return xqt, QD, QI, QP

    def get_knn_search_name(
        self,
        search_parameters: Optional[Dict[str, int]],
        query_vectors: DatasetDescriptor,
        k: int,
    ):
        name = self.get_index_name()
        name += Index.param_dict_to_name(search_parameters)
        name += query_vectors.get_filename("q")
        name += f"k_{k}."
        return name

    def knn_search(
        self,
        search_parameters: Optional[Dict[str, int]],
        query_vectors: DatasetDescriptor,
        k: int,
    ):
        logger.info("knn_seach: begin")
        filename = (
            self.get_knn_search_name(search_parameters, query_vectors, k)
            + "zip"
        )
        if self.io.file_exist(filename):
            logger.info(f"Using cached results for {filename}")
            D, I, R, P = self.io.read_file(filename, ["D", "I", "R", "P"])
        else:
            xq = self.io.get_dataset(query_vectors)
            index = self.get_index()
            Index.set_index_param_dict(index, search_parameters)

            if self.is_ivf():
                xqt, QD, QI, QP = self.knn_search_quantizer(
                    index, query_vectors, search_parameters["nprobe"]
                )
                index_ivf = faiss.extract_index_ivf(index)
                if index_ivf.parallel_mode != 2:
                    logger.info("Setting IVF parallel mode")
                    index_ivf.parallel_mode = 2

                (D, I), t, repeat = timer(
                    "knn_search_preassigned",
                    lambda: index_ivf.search_preassigned(xqt, k, QI, QD),
                )
            else:
                (D, I), t, _ = timer("knn_search", lambda: index.search(xq, k))
            if self.is_flat() or not hasattr(self, "database_vectors"):  # TODO
                R = D
            else:
                xb = self.io.get_dataset(self.database_vectors)
                R = refine_distances_knn(D, I, xq, xb, self.metric_type)
            P = {
                "time": t,
                "index": self.get_index_name(),
                "codec": self.get_codec_name(),
                "factory": self.factory if hasattr(self, "factory") else "",
                "search_params": search_parameters,
                "k": k,
            }
            if self.is_ivf():
                stats = faiss.cvar.indexIVF_stats
                P |= {
                    "quantizer": QP,
                    "nq": int(stats.nq // repeat),
                    "nlist": int(stats.nlist // repeat),
                    "ndis": int(stats.ndis // repeat),
                    "nheap_updates": int(stats.nheap_updates // repeat),
                    "quantization_time": int(
                        stats.quantization_time // repeat
                    ),
                    "search_time": int(stats.search_time // repeat),
                }
            self.io.write_file(filename, ["D", "I", "R", "P"], [D, I, R, P])
        logger.info("knn_seach: end")
        return D, I, R, P

    def range_search(
        self,
        search_parameters: Optional[Dict[str, int]],
        query_vectors: DatasetDescriptor,
        radius: Optional[float] = None,
    ):
        logger.info("range_search: begin")
        filename = (
            self.get_range_search_name(
                search_parameters, query_vectors, radius
            )
            + "zip"
        )
        if self.io.file_exist(filename):
            logger.info(f"Using cached results for {filename}")
            lims, D, I, R, P = self.io.read_file(
                filename, ["lims", "D", "I", "R", "P"]
            )
        else:
            xq = self.io.get_dataset(query_vectors)
            index = self.get_index()
            Index.set_index_param_dict(index, search_parameters)

            if self.is_ivf():
                xqt, QD, QI, QP = self.knn_search_quantizer(
                    index, query_vectors, search_parameters["nprobe"]
                )
                index_ivf = faiss.extract_index_ivf(index)
                if index_ivf.parallel_mode != 2:
                    logger.info("Setting IVF parallel mode")
                    index_ivf.parallel_mode = 2

                (lims, D, I), t, repeat = timer(
                    "range_search_preassigned",
                    lambda: index_ivf.range_search_preassigned(
                        xqt, radius, QI, QD
                    ),
                )
            else:
                (lims, D, I), t, _ = timer(
                    "range_search", lambda: index.range_search(xq, radius)
                )
            if self.is_flat():
                R = D
            else:
                xb = self.io.get_dataset(self.database_vectors)
                R = refine_distances_range(
                    lims, D, I, xq, xb, self.metric_type
                )
            P = {
                "time": t,
                "index": self.get_codec_name(),
                "codec": self.get_codec_name(),
                "search_params": search_parameters,
                "radius": radius,
                "count": len(I),
            }
            if self.is_ivf():
                stats = faiss.cvar.indexIVF_stats
                P |= {
                    "quantizer": QP,
                    "nq": int(stats.nq // repeat),
                    "nlist": int(stats.nlist // repeat),
                    "ndis": int(stats.ndis // repeat),
                    "nheap_updates": int(stats.nheap_updates // repeat),
                    "quantization_time": int(
                        stats.quantization_time // repeat
                    ),
                    "search_time": int(stats.search_time // repeat),
                }
            self.io.write_file(
                filename, ["lims", "D", "I", "R", "P"], [lims, D, I, R, P]
            )
        logger.info("range_seach: end")
        return lims, D, I, R, P


# Common base for IndexFromCodec and IndexFromFactory,
# but not for the sub-indices of codec-based indices
# IndexFromQuantizer and IndexFromPreTransform, because
# they share the configuration of their parent IndexFromCodec
@dataclass
class Index(IndexBase):
    d: int
    metric: str
    database_vectors: DatasetDescriptor
    construction_params: List[Dict[str, int]]
    search_params: Dict[str, int]

    cached_codec_name: ClassVar[str] = None
    cached_codec: ClassVar[faiss.Index] = None
    cached_index_name: ClassVar[str] = None
    cached_index: ClassVar[faiss.Index] = None

    def __post_init__(self):
        if isinstance(self.metric, str):
            if self.metric == "IP":
                self.metric_type = faiss.METRIC_INNER_PRODUCT
            elif self.metric == "L2":
                self.metric_type = faiss.METRIC_L2
            else:
                raise ValueError
        elif isinstance(self.metric, int):
            self.metric_type = self.metric
            if self.metric_type == faiss.METRIC_INNER_PRODUCT:
                self.metric = "IP"
            elif self.metric_type == faiss.METRIC_L2:
                self.metric = "L2"
            else:
                raise ValueError
        else:
            raise ValueError

    def supports_range_search(self):
        codec = self.get_codec()
        return not type(codec) in [
            faiss.IndexHNSWFlat,
            faiss.IndexIVFFastScan,
            faiss.IndexRefine,
            faiss.IndexPQ,
        ]

    def fetch_codec(self):
        raise NotImplementedError

    def train(self):
        # get triggers a train, if necessary
        self.get_codec()

    def get_codec(self):
        codec_name = self.get_codec_name()
        if Index.cached_codec_name != codec_name:
            Index.cached_codec = self.fetch_codec()
            Index.cached_codec_name = codec_name
        return Index.cached_codec

    def get_index_name(self):
        name = self.get_codec_name()
        assert self.database_vectors is not None
        name += self.database_vectors.get_filename("xb")
        return name

    def fetch_index(self):
        index = faiss.clone_index(self.get_codec())
        assert index.ntotal == 0
        logger.info("Adding vectors to index")
        xb = self.io.get_dataset(self.database_vectors)

        if self.is_ivf():
            xbt, QD, QI, QP = self.knn_search_quantizer(
                index, self.database_vectors, 1
            )
            index_ivf = faiss.extract_index_ivf(index)
            if index_ivf.parallel_mode != 2:
                logger.info("Setting IVF parallel mode")
                index_ivf.parallel_mode = 2

            _, t, _ = timer(
                "add_preassigned",
                lambda: add_preassigned(index_ivf, xbt, QI.ravel()),
                once=True,
            )
        else:
            _, t, _ = timer(
                "add",
                lambda: index.add(xb),
                once=True,
            )
        assert index.ntotal == xb.shape[0] or index_ivf.ntotal == xb.shape[0]
        logger.info("Added vectors to index")
        return index

    def get_index(self):
        index_name = self.get_index_name()
        if Index.cached_index_name != index_name:
            Index.cached_index = self.fetch_index()
            Index.cached_index_name = index_name
        return Index.cached_index

    def get_code_size(self):
        def get_index_code_size(index):
            index = faiss.downcast_index(index)
            if isinstance(index, faiss.IndexPreTransform):
                return get_index_code_size(index.index)
            elif isinstance(index, faiss.IndexHNSWFlat):
                return index.d * 4  # TODO
            elif type(index) in [faiss.IndexRefine, faiss.IndexRefineFlat]:
                return get_index_code_size(
                    index.base_index
                ) + get_index_code_size(index.refine_index)
            else:
                return index.code_size

        codec = self.get_codec()
        return get_index_code_size(codec)

    def get_operating_points(self):
        op = OperatingPointsWithRanges()

        def add_range_or_val(name, range):
            op.add_range(
                name,
                [self.search_params[name]]
                if self.search_params and name in self.search_params
                else range,
            )

        op.add_range("noop", [0])
        codec = faiss.downcast_index(self.get_codec())
        codec_ivf = faiss.try_extract_index_ivf(codec)
        if codec_ivf is not None:
            add_range_or_val(
                "nprobe",
                [
                    2**i
                    for i in range(12)
                    if 2**i <= codec_ivf.nlist * 0.25
                ],
            )
        if isinstance(codec, faiss.IndexRefine):
            add_range_or_val(
                "k_factor",
                [2**i for i in range(11)],
            )
        if isinstance(codec, faiss.IndexHNSWFlat):
            add_range_or_val(
                "efSearch",
                [2**i for i in range(3, 11)],
            )
        return op

    def get_range_search_name(
        self,
        search_parameters: Optional[Dict[str, int]],
        query_vectors: DatasetDescriptor,
        radius: Optional[float] = None,
    ):
        name = self.get_index_name()
        name += Index.param_dict_to_name(search_parameters)
        name += query_vectors.get_filename("q")
        if radius is not None:
            name += f"r_{int(radius * 1000)}."
        else:
            name += "r_auto."
        return name


# IndexFromCodec, IndexFromQuantizer and IndexFromPreTransform
# are used to wrap pre-trained Faiss indices (codecs)
@dataclass
class IndexFromCodec(Index):
    path: str
    bucket: Optional[str] = None

    def get_quantizer(self):
        if not self.is_ivf():
            raise ValueError("Not an IVF index")
        quantizer = IndexFromQuantizer(self)
        quantizer.set_io(self.io)
        return quantizer

    def get_pretransform(self):
        if not self.is_ivf():
            raise ValueError("Not an IVF index")
        quantizer = IndexFromPreTransform(self)
        quantizer.set_io(self.io)
        return quantizer

    def get_codec_name(self):
        assert self.path is not None
        name = os.path.basename(self.path)
        name += Index.param_dict_list_to_name(self.construction_params)
        return name

    def fetch_codec(self):
        codec = self.io.read_index(
            os.path.basename(self.path),
            self.bucket,
            os.path.dirname(self.path),
        )
        assert self.d == codec.d
        assert self.metric_type == codec.metric_type
        Index.set_index_param_dict_list(codec, self.construction_params)
        return codec

    def get_model(self):
        return self.get_codec()


class IndexFromQuantizer(IndexBase):
    ivf_index: Index

    def __init__(self, ivf_index: Index):
        self.ivf_index = ivf_index
        super().__init__()

    def get_codec_name(self):
        return self.get_index_name()

    def get_codec(self):
        return self.get_index()

    def get_index_name(self):
        ivf_codec_name = self.ivf_index.get_codec_name()
        return f"{ivf_codec_name}quantizer."

    def get_index(self):
        ivf_codec = faiss.extract_index_ivf(self.ivf_index.get_codec())
        return ivf_codec.quantizer


class IndexFromPreTransform(IndexBase):
    pre_transform_index: Index

    def __init__(self, pre_transform_index: Index):
        self.pre_transform_index = pre_transform_index
        super().__init__()

    def get_codec_name(self):
        pre_transform_codec_name = self.pre_transform_index.get_codec_name()
        return f"{pre_transform_codec_name}pretransform."

    def get_codec(self):
        return self.get_codec()


# IndexFromFactory is for creating and training indices from scratch
@dataclass
class IndexFromFactory(Index):
    factory: str
    training_vectors: DatasetDescriptor

    def get_codec_name(self):
        assert self.factory is not None
        name = f"{self.factory.replace(',', '_')}."
        assert self.d is not None
        assert self.metric is not None
        name += f"d_{self.d}.{self.metric.upper()}."
        if self.factory != "Flat":
            assert self.training_vectors is not None
            name += self.training_vectors.get_filename("xt")
        name += Index.param_dict_list_to_name(self.construction_params)
        return name

    def fetch_codec(self):
        codec_filename = self.get_codec_name() + "codec"
        if self.io.file_exist(codec_filename):
            codec = self.io.read_index(codec_filename)
            assert self.d == codec.d
            assert self.metric_type == codec.metric_type
        else:
            codec = self.assemble()
            if self.factory != "Flat":
                self.io.write_index(codec, codec_filename)
        return codec

    def get_model(self):
        model = faiss.index_factory(self.d, self.factory, self.metric_type)
        Index.set_index_param_dict_list(model, self.construction_params)
        return model

    def get_pretransform(self):
        model = faiss.index_factory(self.d, self.factory, self.metric_type)
        assert isinstance(model, faiss.IndexPreTransform)
        sub_index = faiss.downcast_index(model.index)
        if isinstance(sub_index, faiss.IndexFlat):
            return self
        # replace the sub-index with Flat
        codec = faiss.clone_index(model)
        codec.index = faiss.IndexFlat(codec.index.d, codec.index.metric_type)
        pretransform = IndexFromFactory(
            d=codec.d,
            metric=codec.metric_type,
            database_vectors=self.database_vectors,
            construction_params=self.construction_params,
            search_params=self.search_params,
            factory=reverse_index_factory(codec),
            training_vectors=self.training_vectors,
        )
        pretransform.set_io(self.io)
        return pretransform

    def get_quantizer(self, pretransform=None):
        model = self.get_model()
        model_ivf = faiss.extract_index_ivf(model)
        assert isinstance(model_ivf, faiss.IndexIVF)
        assert ord(model_ivf.quantizer_trains_alone) in [0, 2]
        if pretransform is None:
            training_vectors = self.training_vectors
        else:
            training_vectors = pretransform.transform(self.training_vectors)
        centroids = self.k_means(training_vectors, model_ivf.nlist)
        quantizer = IndexFromFactory(
            d=model_ivf.quantizer.d,
            metric=model_ivf.quantizer.metric_type,
            database_vectors=centroids,
            construction_params=None,  # self.construction_params[1:],
            search_params=None,  # self.construction_params[0],  # TODO: verify
            factory=reverse_index_factory(model_ivf.quantizer),
            training_vectors=centroids,
        )
        quantizer.set_io(self.io)
        return quantizer

    def k_means(self, vectors, k):
        kmeans_vectors = DatasetDescriptor(
            tablename=f"{vectors.get_filename()}kmeans_{k}.npy"
        )
        if not self.io.file_exist(kmeans_vectors.tablename):
            x = self.io.get_dataset(vectors)
            kmeans = faiss.Kmeans(d=x.shape[1], k=k, gpu=True)
            kmeans.train(x)
            self.io.write_nparray(kmeans.centroids, kmeans_vectors.tablename)
        return kmeans_vectors

    def assemble(self):
        model = self.get_model()
        codec = faiss.clone_index(model)
        if isinstance(model, faiss.IndexPreTransform):
            sub_index = faiss.downcast_index(model.index)
            if not isinstance(sub_index, faiss.IndexFlat):
                # replace the sub-index with Flat and fetch pre-trained
                pretransform = self.get_pretransform()
                codec = pretransform.fetch_codec()
                assert codec.is_trained
                transformed_training_vectors = pretransform.transform(
                    self.training_vectors
                )
                transformed_database_vectors = pretransform.transform(
                    self.database_vectors
                )
                # replace the Flat index with the required sub-index
                wrapper = IndexFromFactory(
                    d=sub_index.d,
                    metric=sub_index.metric_type,
                    database_vectors=transformed_database_vectors,
                    construction_params=self.construction_params,
                    search_params=self.search_params,
                    factory=reverse_index_factory(sub_index),
                    training_vectors=transformed_training_vectors,
                )
                wrapper.set_io(self.io)
                codec.index = wrapper.fetch_codec()
                assert codec.index.is_trained
        elif isinstance(model, faiss.IndexIVF):
            # replace the quantizer
            quantizer = self.get_quantizer()
            replace_ivf_quantizer(codec, quantizer.fetch_index())
            assert codec.quantizer.is_trained
            assert codec.nlist == codec.quantizer.ntotal
        elif isinstance(model, faiss.IndexRefine) or isinstance(
            model, faiss.IndexRefineFlat
        ):
            # replace base_index
            wrapper = IndexFromFactory(
                d=model.base_index.d,
                metric=model.base_index.metric_type,
                database_vectors=self.database_vectors,
                construction_params=self.construction_params,
                search_params=self.search_params,
                factory=reverse_index_factory(model.base_index),
                training_vectors=self.training_vectors,
            )
            wrapper.set_io(self.io)
            codec.base_index = wrapper.fetch_codec()
            assert codec.base_index.is_trained

        xt = self.io.get_dataset(self.training_vectors)
        codec.train(xt)
        return codec
