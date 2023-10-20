import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional
from zipfile import ZipFile

import faiss  # @manual=//faiss/python:pyfaiss_gpu

import numpy as np

from .descriptors import DatasetDescriptor, IndexDescriptor

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkIO:
    path: str

    def __post_init__(self):
        self.cached_ds = {}
        self.cached_codec_key = None

    def get_filename_search(
        self,
        factory: str,
        parameters: Optional[dict[str, int]],
        level: int,
        db_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        k: Optional[int] = None,
        r: Optional[float] = None,
        evaluation_name: Optional[str] = None,
    ):
        assert factory is not None
        assert level is not None
        assert self.distance_metric is not None
        assert query_vectors is not None
        assert self.distance_metric is not None
        filename = f"{factory.lower().replace(',', '_')}."
        if level > 0:
            filename += f"l_{level}."
        if db_vectors is not None:
            filename += db_vectors.get_filename("d")
        filename += query_vectors.get_filename("q")
        filename += self.distance_metric.upper() + "."
        if k is not None:
            filename += f"k_{k}."
        if r is not None:
            filename += f"r_{int(r * 1000)}."
        if parameters is not None:
            for name, val in parameters.items():
                if name != "noop":
                    filename += f"{name}_{val}."
        if evaluation_name is None:
            filename += "zip"
        else:
            filename += evaluation_name
        return filename

    def get_filename_knn_search(
        self,
        factory: str,
        parameters: Optional[dict[str, int]],
        level: int,
        db_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        k: int,
    ):
        assert k is not None
        return self.get_filename_search(
            factory=factory,
            parameters=parameters,
            level=level,
            db_vectors=db_vectors,
            query_vectors=query_vectors,
            k=k,
        )

    def get_filename_range_search(
        self,
        factory: str,
        parameters: Optional[dict[str, int]],
        level: int,
        db_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        r: float,
    ):
        assert r is not None
        return self.get_filename_search(
            factory=factory,
            parameters=parameters,
            level=level,
            db_vectors=db_vectors,
            query_vectors=query_vectors,
            r=r,
        )

    def get_filename_evaluation_name(
        self,
        factory: str,
        parameters: Optional[dict[str, int]],
        level: int,
        db_vectors: DatasetDescriptor,
        query_vectors: DatasetDescriptor,
        evaluation_name: str,
    ):
        assert evaluation_name is not None
        return self.get_filename_search(
            factory=factory,
            parameters=parameters,
            level=level,
            db_vectors=db_vectors,
            query_vectors=query_vectors,
            evaluation_name=evaluation_name,
        )

    def get_local_filename(self, filename):
        return os.path.join(self.path, filename)

    def download_file_from_blobstore(
        self,
        filename: str,
        bucket: Optional[str] = None,
        path: Optional[str] = None,
    ):
        return self.get_local_filename(filename)

    def upload_file_to_blobstore(
        self,
        filename: str,
        bucket: Optional[str] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
    ):
        pass

    def file_exist(self, filename: str):
        fn = self.get_local_filename(filename)
        exists = os.path.exists(fn)
        logger.info(f"{filename} {exists=}")
        return exists

    def get_codec(self, index_desc: IndexDescriptor, d: int):
        if index_desc.factory == "Flat":
            return faiss.IndexFlat(d, self.distance_metric_type)
        else:
            if self.cached_codec_key != index_desc.factory:
                codec = faiss.read_index(
                    self.get_local_filename(index_desc.path)
                )
                assert (
                    codec.metric_type == self.distance_metric_type
                ), f"{codec.metric_type=} != {self.distance_metric_type=}"
                logger.info(f"Loaded codec from {index_desc.path}")
                self.cached_codec_key = index_desc.factory
                self.cached_codec = codec
            return self.cached_codec

    def read_file(self, filename: str, keys: List[str]):
        fn = self.download_file_from_blobstore(filename)
        logger.info(f"Loading file {fn}")
        results = []
        with ZipFile(fn, "r") as zip_file:
            for key in keys:
                with zip_file.open(key, "r") as f:
                    if key in ["D", "I", "R", "lims"]:
                        results.append(np.load(f))
                    elif key in ["P"]:
                        t = io.TextIOWrapper(f)
                        results.append(json.load(t))
                    else:
                        raise AssertionError()
        return results

    def write_file(
        self,
        filename: str,
        keys: List[str],
        values: List[Any],
        overwrite: bool = False,
    ):
        fn = self.get_local_filename(filename)
        with ZipFile(fn, "w") as zip_file:
            for key, value in zip(keys, values, strict=True):
                with zip_file.open(key, "w") as f:
                    if key in ["D", "I", "R", "lims"]:
                        np.save(f, value)
                    elif key in ["P"]:
                        t = io.TextIOWrapper(f, write_through=True)
                        json.dump(value, t)
                    else:
                        raise AssertionError()
        self.upload_file_to_blobstore(filename, overwrite=overwrite)

    def get_dataset(self, dataset):
        if dataset not in self.cached_ds:
            self.cached_ds[dataset] = self.read_nparray(
                os.path.join(self.path, dataset.name)
            )
        return self.cached_ds[dataset]

    def read_nparray(
        self,
        filename: str,
    ):
        fn = self.download_file_from_blobstore(filename)
        logger.info(f"Loading nparray from {fn}\n")
        nparray = np.load(fn)
        logger.info(f"Loaded nparray {nparray.shape} from {fn}\n")
        return nparray

    def write_nparray(
        self,
        nparray: np.ndarray,
        filename: str,
    ):
        fn = self.get_local_filename(filename)
        logger.info(f"Saving nparray {nparray.shape} to {fn}\n")
        np.save(fn, nparray)
        self.upload_file_to_blobstore(filename)

    def read_json(
        self,
        filename: str,
    ):
        fn = self.download_file_from_blobstore(filename)
        logger.info(f"Loading json {fn}\n")
        with open(fn, "r") as fp:
            json_dict = json.load(fp)
        logger.info(f"Loaded json {json_dict} from {fn}\n")
        return json_dict

    def write_json(
        self,
        json_dict: dict[str, Any],
        filename: str,
        overwrite: bool = False,
    ):
        fn = self.get_local_filename(filename)
        logger.info(f"Saving json {json_dict} to {fn}\n")
        with open(fn, "w") as fp:
            json.dump(json_dict, fp)
        self.upload_file_to_blobstore(filename, overwrite=overwrite)
