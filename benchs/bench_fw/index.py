# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from copy import copy
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

import faiss  # @manual=//faiss/python:pyfaiss_gpu

import numpy as np
from faiss.contrib.evaluation import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    OperatingPointsWithRanges,
    knn_intersection_measure,
)
from faiss.contrib.factory_tools import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    reverse_index_factory,
)
from faiss.contrib.ivf_tools import (  # @manual=//faiss/contrib:faiss_contrib_gpu
    add_preassigned,
    replace_ivf_quantizer,
)

from .descriptors import DatasetDescriptor
from .utils import distance_ratio_measure, get_cpu_info, timer, refine_distances_knn, refine_distances_range

logger = logging.getLogger(__name__)


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

    @staticmethod
    def set_index_param_dict_list(index, param_dict_list, assert_same=False):
        if not param_dict_list:
            return
        index = faiss.downcast_index(index)
        for param_dict in param_dict_list:
            assert index is not None
            IndexBase.set_index_param_dict(index, param_dict, assert_same)
            index = faiss.try_extract_index_ivf(index)
            if index is not None:
                index = index.quantizer

    @staticmethod
    def set_index_param_dict(index, param_dict, assert_same=False):
        if not param_dict:
            return
        for name, val in param_dict.items():
            IndexBase.set_index_param(index, name, val, assert_same)

    @staticmethod
    def set_index_param(index, name, val, assert_same=False):
        index = faiss.downcast_index(index)
        val = int(val)
        if isinstance(index, faiss.IndexPreTransform):
            Index.set_index_param(index.index, name, val)
            return
        elif name == "snap":
            return
        elif name == "lsq_gpu":
            if val == 1:
                ngpus = faiss.get_num_gpus()
                icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
                if isinstance(index, faiss.IndexProductLocalSearchQuantizer):
                    for i in range(index.plsq.nsplits):
                        lsq = faiss.downcast_Quantizer(index.plsq.subquantizer(i))
                        if lsq.icm_encoder_factory is None:
                            lsq.icm_encoder_factory = icm_encoder_factory
                else:
                    if index.lsq.icm_encoder_factory is None:
                        index.lsq.icm_encoder_factory = icm_encoder_factory
            return
        elif name in ["efSearch", "efConstruction"]:
            obj = index.hnsw
        elif name in ["nprobe", "parallel_mode"]:
            obj = faiss.extract_index_ivf(index)
        elif name in ["use_beam_LUT", "max_beam_size"]:
            if isinstance(index, faiss.IndexProductResidualQuantizer):
                obj = [faiss.downcast_Quantizer(index.prq.subquantizer(i)) for i in range(index.prq.nsplits)]
            else:
                obj = index.rq
        elif name == "encode_ils_iters":
            if isinstance(index, faiss.IndexProductLocalSearchQuantizer):
                obj = [faiss.downcast_Quantizer(index.plsq.subquantizer(i)) for i in range(index.plsq.nsplits)]
            else:
                obj = index.lsq
        else:
            obj = index
        
        if not isinstance(obj, list):
            obj = [obj]
        for o in obj:
            test = getattr(o, name)
            if assert_same and not name == 'use_beam_LUT':
                assert test == val
            else:
                setattr(o, name, val)

    @staticmethod
    def filter_index_param_dict_list(param_dict_list):
        if param_dict_list is not None and param_dict_list[0] is not None and "k_factor" in param_dict_list[0]:
            filtered = copy(param_dict_list)
            del filtered[0]["k_factor"]
            return filtered
        else:
            return param_dict_list

    def is_flat(self):
        model = faiss.downcast_index(self.get_model())
        return isinstance(model, faiss.IndexFlat)

    def is_ivf(self):
        model = self.get_model()
        return faiss.try_extract_index_ivf(model) is not None

    def is_2layer(self):
        def is_2layer_(index):
            index = faiss.downcast_index(index)
            if isinstance(index, faiss.IndexPreTransform):
                return is_2layer_(index.index)
            return isinstance(index, faiss.Index2Layer)

        model = self.get_model()
        return is_2layer_(model)

    def is_decode_supported(self):
        model = self.get_model()
        if isinstance(model, faiss.IndexPreTransform):
            for i in range(model.chain.size()):
                vt = faiss.downcast_VectorTransform(model.chain.at(i))
                if isinstance(vt, faiss.ITQTransform):
                    return False
        return True

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

    def get_construction_params(self):
        raise NotImplementedError

    def transform(self, vectors):
        transformed_vectors = DatasetDescriptor(
            tablename=f"{vectors.get_filename()}{self.get_codec_name()}transform.npy"
        )
        if not self.io.file_exist(transformed_vectors.tablename):
            codec = self.get_codec()
            assert isinstance(codec, faiss.IndexPreTransform)
            transform = faiss.downcast_VectorTransform(codec.chain.at(0))
            x = self.io.get_dataset(vectors)
            xt = transform.apply(x)
            self.io.write_nparray(xt, transformed_vectors.tablename)
        return transformed_vectors

    def snap(self, vectors):
        transformed_vectors = DatasetDescriptor(
            tablename=f"{vectors.get_filename()}{self.get_codec_name()}snap.npy"
        )
        if not self.io.file_exist(transformed_vectors.tablename):
            codec = self.get_codec()
            x = self.io.get_dataset(vectors)
            xt = codec.sa_decode(codec.sa_encode(x))
            self.io.write_nparray(xt, transformed_vectors.tablename)
        return transformed_vectors

    def knn_search_quantizer(self, query_vectors, k):
        if self.is_pretransform():
            pretransform = self.get_pretransform()
            quantizer_query_vectors = pretransform.transform(query_vectors)
        else:
            pretransform = None
            quantizer_query_vectors = query_vectors

        quantizer, _, _ = self.get_quantizer(dry_run=False, pretransform=pretransform)
        QD, QI, _, QP, _ = quantizer.knn_search(
            dry_run=False,
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
        reconstruct: bool = False,
    ):
        name = self.get_index_name()
        name += Index.param_dict_to_name(search_parameters)
        name += query_vectors.get_filename("q")
        name += f"k_{k}."
        name += f"t_{self.num_threads}."
        if reconstruct:
            name += "rec."
        else:
            name += "knn."
        return name

    def knn_search(
        self,
        dry_run,
        search_parameters: Optional[Dict[str, int]],
        query_vectors: DatasetDescriptor,
        k: int,
        I_gt=None,
        D_gt=None,
    ):
        logger.info("knn_search: begin")
        if search_parameters is not None and search_parameters["snap"] == 1:
            query_vectors = self.snap(query_vectors)
        filename = (
            self.get_knn_search_name(search_parameters, query_vectors, k)
            + "zip"
        )
        if self.io.file_exist(filename):
            logger.info(f"Using cached results for {filename}")
            D, I, R, P = self.io.read_file(filename, ["D", "I", "R", "P"])
        else:
            if dry_run:
                return None, None, None, None, filename
            index = self.get_index()
            Index.set_index_param_dict(index, search_parameters)

            if self.is_2layer():
                # Index2Layer doesn't support search
                xq = self.io.get_dataset(query_vectors)
                xb = index.reconstruct_n(0, index.ntotal)
                (D, I), t, _ = timer("knn_search 2layer", lambda: faiss.knn(xq, xb, k))
            elif self.is_ivf() and not isinstance(index, faiss.IndexRefine):
                index_ivf = faiss.extract_index_ivf(index)
                nprobe = (
                    search_parameters["nprobe"]
                    if search_parameters is not None
                    and "nprobe" in search_parameters
                    else index_ivf.nprobe
                )
                xqt, QD, QI, QP = self.knn_search_quantizer(
                    query_vectors=query_vectors, 
                    k=nprobe,
                )
                if index_ivf.parallel_mode != 2:
                    logger.info("Setting IVF parallel mode")
                    index_ivf.parallel_mode = 2

                (D, I), t, repeat = timer(
                    "knn_search_preassigned",
                    lambda: index_ivf.search_preassigned(xqt, k, QI, QD),
                )
                # Dref, Iref = index.search(xq, k)
                # np.testing.assert_array_equal(I, Iref)
                # np.testing.assert_allclose(D, Dref)
            else:
                xq = self.io.get_dataset(query_vectors)
                (D, I), t, _ = timer("knn_search", lambda: index.search(xq, k))
            if self.is_flat() or not hasattr(self, "database_vectors"):  # TODO
                R = D
            else:
                xq = self.io.get_dataset(query_vectors)
                xb = self.io.get_dataset(self.database_vectors)
                R = refine_distances_knn(xq, xb, I, self.metric_type)
            P = {
                "time": t,
                "k": k,
            }
            if self.is_ivf() and not isinstance(index, faiss.IndexRefine):
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
        P |= {
            "index": self.get_index_name(),
            "codec": self.get_codec_name(),
            "factory": self.get_model_name(),
            "construction_params": self.get_construction_params(),
            "search_params": search_parameters,
            "knn_intersection": knn_intersection_measure(
                I, I_gt,
            ) if I_gt is not None else None,
            "distance_ratio": distance_ratio_measure(
                I, R, D_gt, self.metric_type,
            ) if D_gt is not None else None,
        }
        logger.info("knn_search: end")
        return D, I, R, P, None

    def reconstruct(
        self,
        dry_run,
        parameters: Optional[Dict[str, int]],
        query_vectors: DatasetDescriptor,
        k: int,
        I_gt,
    ):
        logger.info("reconstruct: begin")
        filename = (
            self.get_knn_search_name(parameters, query_vectors, k, reconstruct=True)
            + "zip"
        )
        if self.io.file_exist(filename):
            logger.info(f"Using cached results for {filename}")
            P, = self.io.read_file(filename, ["P"])
            P["index"] = self.get_index_name()
            P["codec"] = self.get_codec_name()
            P["factory"] = self.get_model_name()
            P["reconstruct_params"] = parameters
            P["construction_params"] = self.get_construction_params()
        else:
            if dry_run:
                return None, filename
            codec = self.get_codec()
            codec_meta = self.fetch_meta()
            Index.set_index_param_dict(codec, parameters)
            xb = self.io.get_dataset(self.database_vectors)
            xb_encoded, encode_t, _ = timer("sa_encode", lambda: codec.sa_encode(xb))
            xq = self.io.get_dataset(query_vectors)
            if self.is_decode_supported():
                xb_decoded, decode_t, _ = timer("sa_decode", lambda: codec.sa_decode(xb_encoded))
                mse = np.square(xb_decoded - xb).sum(axis=1).mean().item()
                _, I = faiss.knn(xq, xb_decoded, k, metric=self.metric_type)
                asym_recall = knn_intersection_measure(I, I_gt)
                xq_decoded = codec.sa_decode(codec.sa_encode(xq))
                _, I = faiss.knn(xq_decoded, xb_decoded, k, metric=self.metric_type)
            else:
                mse = None
                asym_recall = None
                decode_t = None
                # assume hamming for sym
                xq_encoded = codec.sa_encode(xq)
                bin = faiss.IndexBinaryFlat(xq_encoded.shape[1] * 8)
                bin.add(xb_encoded)
                _, I = bin.search(xq_encoded, k)
            sym_recall = knn_intersection_measure(I, I_gt)
            P = {
                "encode_time": encode_t,
                "decode_time": decode_t,
                "mse": mse,
                "sym_recall": sym_recall,
                "asym_recall": asym_recall,
                "cpu": get_cpu_info(),
                "num_threads": self.num_threads,
                "index": self.get_index_name(),
                "codec": self.get_codec_name(),
                "factory": self.get_model_name(),
                "reconstruct_params": parameters,
                "construction_params": self.get_construction_params(),
                "codec_meta": codec_meta,
            }
            self.io.write_file(filename, ["P"], [P])
        logger.info("reconstruct: end")
        return P, None

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

    def range_search(
        self,
        dry_run,
        search_parameters: Optional[Dict[str, int]],
        query_vectors: DatasetDescriptor,
        radius: Optional[float] = None,
    ):
        logger.info("range_search: begin")
        if search_parameters is not None and search_parameters["snap"] == 1:
            query_vectors = self.snap(query_vectors)
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
            if dry_run:
                return None, None, None, None, None, filename
            xq = self.io.get_dataset(query_vectors)
            index = self.get_index()
            Index.set_index_param_dict(index, search_parameters)

            if self.is_ivf():
                xqt, QD, QI, QP = self.knn_search_quantizer(
                    query_vectors, search_parameters["nprobe"]
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
        P |= {
            "index": self.get_index_name(),
            "codec": self.get_codec_name(),
            "factory": self.get_model_name(),
            "construction_params": self.get_construction_params(),
            "search_params": search_parameters,
        }
        logger.info("range_seach: end")
        return lims, D, I, R, P, None


# Common base for IndexFromCodec and IndexFromFactory,
# but not for the sub-indices of codec-based indices
# IndexFromQuantizer and IndexFromPreTransform, because
# they share the configuration of their parent IndexFromCodec
@dataclass
class Index(IndexBase):
    num_threads: int
    d: int
    metric: str
    database_vectors: DatasetDescriptor
    construction_params: List[Dict[str, int]]
    search_params: Dict[str, int]

    cached_codec: ClassVar[OrderedDict[str, faiss.Index]] = OrderedDict()
    cached_index: ClassVar[OrderedDict[str, faiss.Index]] = OrderedDict()

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

    def get_codec(self):
        codec_name = self.get_codec_name()
        if codec_name not in Index.cached_codec:
            Index.cached_codec[codec_name], _, _ = self.fetch_codec()
            if len(Index.cached_codec) > 1:
                Index.cached_codec.popitem(last=False)
        return Index.cached_codec[codec_name]

    def get_index_name(self):
        name = self.get_codec_name()
        assert self.database_vectors is not None
        name += self.database_vectors.get_filename("xb")
        return name

    def fetch_index(self):
        index = self.get_codec()
        index.reset()
        assert index.ntotal == 0
        logger.info("Adding vectors to index")
        xb = self.io.get_dataset(self.database_vectors)

        if self.is_ivf() and not isinstance(index, faiss.IndexRefine):
            xbt, QD, QI, QP = self.knn_search_quantizer(
                query_vectors=self.database_vectors, 
                k=1,
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
        return index, t

    def get_index(self):
        index_name = self.get_index_name()
        if index_name not in Index.cached_index:
            Index.cached_index[index_name], _ = self.fetch_index()
            if len(Index.cached_index) > 3:
                Index.cached_index.popitem(last=False)
        return Index.cached_index[index_name]

    def get_construction_params(self):
        return self.construction_params

    # def get_code_size(self):
    #     def get_index_code_size(index):
    #         index = faiss.downcast_index(index)
    #         if isinstance(index, faiss.IndexPreTransform):
    #             return get_index_code_size(index.index)
    #         elif isinstance(index, faiss.IndexHNSWFlat):
    #             return index.d * 4  # TODO
    #         elif type(index) in [faiss.IndexRefine, faiss.IndexRefineFlat]:
    #             return get_index_code_size(
    #                 index.base_index
    #             ) + get_index_code_size(index.refine_index)
    #         else:
    #             return index.code_size

    #     codec = self.get_codec()
    #     return get_index_code_size(codec)

    def get_sa_code_size(self, codec=None):
        if codec is None:
            codec = self.get_codec()
        try:
            return codec.sa_code_size()
        except:
            return None

    def get_operating_points(self):
        op = OperatingPointsWithRanges()

        def add_range_or_val(name, range):
            op.add_range(
                name,
                [self.search_params[name]]
                if self.search_params and name in self.search_params
                else range,
            )

        add_range_or_val("snap", [0])
        model = self.get_model()
        model_ivf = faiss.try_extract_index_ivf(model)
        if model_ivf is not None:
            add_range_or_val(
                "nprobe",
                # [
                #     2**i
                #     for i in range(12)
                #     if 2**i <= model_ivf.nlist * 0.5
                # ],
                [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28] + [
                    i
                    for i in range(32, 64, 8)
                    if i <= model_ivf.nlist * 0.1
                ] + [
                    i
                    for i in range(64, 128, 16)
                    if i <= model_ivf.nlist * 0.1
                ] + [
                    i
                    for i in range(128, 256, 32)
                    if i <= model_ivf.nlist * 0.1
                ] + [
                    i
                    for i in range(256, 512, 64)
                    if i <= model_ivf.nlist * 0.1
                ] + [
                    2**i
                    for i in range(9, 12)
                    if 2**i <= model_ivf.nlist * 0.1
                ],
            )
        model = faiss.downcast_index(model)
        if isinstance(model, faiss.IndexRefine):
            add_range_or_val(
                "k_factor",
                [2**i for i in range(13)],
            )
        elif isinstance(model, faiss.IndexHNSWFlat):
            add_range_or_val(
                "efSearch",
                [2**i for i in range(3, 11)],
            )
        elif isinstance(model, faiss.IndexResidualQuantizer) or isinstance(model, faiss.IndexProductResidualQuantizer):
            add_range_or_val(
                "max_beam_size",
                [1, 2, 4, 8, 16, 32],
            )
            add_range_or_val(
                "use_beam_LUT",
                [1],
            )
        elif isinstance(model, faiss.IndexLocalSearchQuantizer) or isinstance(model, faiss.IndexProductLocalSearchQuantizer):
            add_range_or_val(
                "encode_ils_iters",
                [2, 4, 8, 16],
            )
            add_range_or_val(
                "lsq_gpu",
                [1],
            )
        return op


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

    def get_model_name(self):
        return os.path.basename(self.path)

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
        return codec, None, None

    def get_model(self):
        return self.get_codec()


class IndexFromQuantizer(IndexBase):
    ivf_index: Index

    def __init__(self, ivf_index: Index):
        self.ivf_index = ivf_index
        super().__init__()

    def get_model_name(self):
        return self.get_index_name()

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

    def fetch_meta(self, dry_run=False):
        meta_filename = self.get_codec_name() + "json"
        if self.io.file_exist(meta_filename):
            meta = self.io.read_json(meta_filename)
            report = None
        else:
            _, meta, report = self.fetch_codec(dry_run=dry_run)
        return meta, report

    def fetch_codec(self, dry_run=False):
        codec_filename = self.get_codec_name() + "codec"
        meta_filename = self.get_codec_name() + "json"
        if self.io.file_exist(codec_filename) and self.io.file_exist(meta_filename):
            codec = self.io.read_index(codec_filename)
            assert self.d == codec.d
            assert self.metric_type == codec.metric_type
            meta = self.io.read_json(meta_filename)
        else:
            codec, training_time, requires = self.assemble(dry_run=dry_run)
            if requires is not None:
                assert dry_run
                if requires == "":
                    return None, None, codec_filename
                else:
                    return None, None, requires
            codec_size = self.io.write_index(codec, codec_filename)
            assert codec_size is not None
            meta = {
                "training_time": training_time,
                "training_size": self.training_vectors.num_vectors,
                "codec_size": codec_size,
                "sa_code_size": self.get_sa_code_size(codec),
                "cpu": get_cpu_info(),
            }
            self.io.write_json(meta, meta_filename, overwrite=True)

        Index.set_index_param_dict_list(
            codec, self.construction_params, assert_same=True
        )
        return codec, meta, None

    def get_model_name(self):
        return self.factory

    def get_model(self):
        model = faiss.index_factory(self.d, self.factory, self.metric_type)
        Index.set_index_param_dict_list(model, self.construction_params)
        return model

    def get_pretransform(self):
        model = self.get_model()
        assert isinstance(model, faiss.IndexPreTransform)
        sub_index = faiss.downcast_index(model.index)
        if isinstance(sub_index, faiss.IndexFlat):
            return self
        # replace the sub-index with Flat
        model.index = faiss.IndexFlat(model.index.d, model.index.metric_type)
        pretransform = IndexFromFactory(
            num_threads=self.num_threads,
            d=model.d,
            metric=model.metric_type,
            database_vectors=self.database_vectors,
            construction_params=self.construction_params,
            search_params=None,
            factory=reverse_index_factory(model),
            training_vectors=self.training_vectors,
        )
        pretransform.set_io(self.io)
        return pretransform

    def get_quantizer(self, dry_run, pretransform=None):
        model = self.get_model()
        model_ivf = faiss.extract_index_ivf(model)
        assert isinstance(model_ivf, faiss.IndexIVF)
        assert ord(model_ivf.quantizer_trains_alone) in [0, 2]
        if pretransform is None:
            training_vectors = self.training_vectors
        else:
            training_vectors = pretransform.transform(self.training_vectors)
        centroids, t, requires = training_vectors.k_means(self.io, model_ivf.nlist, dry_run)
        if requires is not None:
            return None, None, requires
        quantizer = IndexFromFactory(
            num_threads=self.num_threads,
            d=model_ivf.quantizer.d,
            metric=model_ivf.quantizer.metric_type,
            database_vectors=centroids,
            construction_params=self.construction_params[1:]
            if self.construction_params is not None
            else None,
            search_params=None,
            factory=reverse_index_factory(model_ivf.quantizer),
            training_vectors=centroids,
        )
        quantizer.set_io(self.io)
        return quantizer, t, None

    def assemble(self, dry_run):
        logger.info(f"assemble {self.factory}")
        model = self.get_model()
        opaque = True
        t_aggregate = 0
        try:
            reverse_index_factory(model)
            opaque = False
        except NotImplementedError:
            opaque = True
        if opaque:
            codec = model
        else:
            if isinstance(model, faiss.IndexPreTransform):
                logger.info(f"assemble: pretransform {self.factory}")
                sub_index = faiss.downcast_index(model.index)
                if not isinstance(sub_index, faiss.IndexFlat):
                    # replace the sub-index with Flat and fetch pre-trained
                    pretransform = self.get_pretransform()
                    codec, meta, report = pretransform.fetch_codec(dry_run=dry_run)
                    if report is not None:
                        return None, None, report
                    t_aggregate += meta["training_time"]
                    assert codec.is_trained
                    transformed_training_vectors = pretransform.transform(
                        self.training_vectors
                    )
                    # replace the Flat index with the required sub-index
                    wrapper = IndexFromFactory(
                        num_threads=self.num_threads,
                        d=sub_index.d,
                        metric=sub_index.metric_type,
                        database_vectors=None,
                        construction_params=self.construction_params,
                        search_params=None,
                        factory=reverse_index_factory(sub_index),
                        training_vectors=transformed_training_vectors,
                    )
                    wrapper.set_io(self.io)
                    codec.index, meta, report = wrapper.fetch_codec(dry_run=dry_run)
                    if report is not None:
                        return None, None, report
                    t_aggregate += meta["training_time"]
                    assert codec.index.is_trained
                else:
                    codec = model
            elif isinstance(model, faiss.IndexIVF):
                logger.info(f"assemble: ivf {self.factory}")
                # replace the quantizer
                quantizer, t, requires = self.get_quantizer(dry_run=dry_run)
                if requires is not None:
                    return None, None, requires
                t_aggregate += t
                codec = faiss.clone_index(model)
                quantizer_index, t = quantizer.fetch_index()
                t_aggregate += t
                replace_ivf_quantizer(codec, quantizer_index)
                assert codec.quantizer.is_trained
                assert codec.nlist == codec.quantizer.ntotal
            elif isinstance(model, faiss.IndexRefine) or isinstance(
                model, faiss.IndexRefineFlat
            ):
                logger.info(f"assemble: refine {self.factory}")
                # replace base_index
                wrapper = IndexFromFactory(
                    num_threads=self.num_threads,
                    d=model.base_index.d,
                    metric=model.base_index.metric_type,
                    database_vectors=self.database_vectors,
                    construction_params=IndexBase.filter_index_param_dict_list(self.construction_params),
                    search_params=None,
                    factory=reverse_index_factory(model.base_index),
                    training_vectors=self.training_vectors,
                )
                wrapper.set_io(self.io)
                codec = faiss.clone_index(model)
                codec.base_index, meta, requires = wrapper.fetch_codec(dry_run=dry_run)
                if requires is not None:
                    return None, None, requires
                t_aggregate += meta["training_time"]
                assert codec.base_index.is_trained
            else:
                codec = model

        if self.factory != "Flat":
            if dry_run:
                return None, None, ""
            logger.info(f"assemble, train {self.factory}")
            xt = self.io.get_dataset(self.training_vectors)
            _, t, _ = timer("train", lambda: codec.train(xt), once=True)
            t_aggregate += t

        return codec, t_aggregate, None
