# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import io
import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from zipfile import ZipFile

import faiss  # @manual=//faiss/python:pyfaiss

import numpy as np
import submitit  # @manual=fbsource//third-party/submitit:submitit
from faiss.contrib.datasets import (  # @manual=//faiss/contrib:faiss_contrib
    dataset_from_name,
)

logger = logging.getLogger(__name__)


# merge RCQ coarse quantizer and ITQ encoder to one Faiss index
def merge_rcq_itq(
    # pyre-ignore[11]: `faiss.ResidualCoarseQuantizer` is not defined as a type
    rcq_coarse_quantizer: faiss.ResidualCoarseQuantizer,
    itq_encoder: faiss.IndexPreTransform,
    # pyre-ignore[11]: `faiss.IndexIVFSpectralHash` is not defined as a type.
) -> faiss.IndexIVFSpectralHash:
    # pyre-ignore[16]: `faiss` has no attribute `IndexIVFSpectralHash`.
    index = faiss.IndexIVFSpectralHash(
        rcq_coarse_quantizer,
        rcq_coarse_quantizer.d,
        rcq_coarse_quantizer.ntotal,
        itq_encoder.sa_code_size() * 8,
        1000000,  # larger than the magnitude of the vectors
    )
    index.replace_vt(itq_encoder)
    return index


@dataclass
class BenchmarkIO:
    path: str  # local path

    def __init__(self, path: str):
        self.path = path
        self.cached_ds: Dict[Any, Any] = {}

    def clone(self):
        return BenchmarkIO(path=self.path)

    def get_local_filepath(self, filename):
        if len(filename) > 184:
            fn, ext = os.path.splitext(filename)
            filename = (
                fn[:184] + hashlib.sha256(filename.encode()).hexdigest() + ext
            )
        return os.path.join(self.path, filename)

    def get_remote_filepath(self, filename) -> Optional[str]:
        return None

    def download_file_from_blobstore(
        self,
        filename: str,
        bucket: Optional[str] = None,
        path: Optional[str] = None,
    ):
        return self.get_local_filepath(filename)

    def upload_file_to_blobstore(
        self,
        filename: str,
        bucket: Optional[str] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
    ):
        pass

    def file_exist(self, filename: str):
        fn = self.get_local_filepath(filename)
        exists = os.path.exists(fn)
        logger.info(f"{filename} {exists=}")
        return exists

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
        fn = self.get_local_filepath(filename)
        with ZipFile(fn, "w") as zip_file:
            for key, value in zip(keys, values, strict=True):
                with zip_file.open(key, "w", force_zip64=True) as f:
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
            if (
                dataset.namespace is not None
                and dataset.namespace[:4] == "std_"
            ):
                if dataset.tablename not in self.cached_ds:
                    self.cached_ds[dataset.tablename] = dataset_from_name(
                        dataset.tablename,
                    )
                p = dataset.namespace[4]
                if p == "t":
                    self.cached_ds[dataset] = self.cached_ds[
                        dataset.tablename
                    ].get_train(dataset.num_vectors)
                elif p == "d":
                    self.cached_ds[dataset] = self.cached_ds[
                        dataset.tablename
                    ].get_database()
                elif p == "q":
                    self.cached_ds[dataset] = self.cached_ds[
                        dataset.tablename
                    ].get_queries()
                else:
                    raise ValueError
            elif dataset.namespace == "syn":
                d, seed = dataset.tablename.split("_")
                d = int(d)
                seed = int(seed)
                n = dataset.num_vectors
                # based on faiss.contrib.datasets.SyntheticDataset
                d1 = 10
                rs = np.random.RandomState(seed)
                x = rs.normal(size=(n, d1))
                x = np.dot(x, rs.rand(d1, d))
                x = x * (rs.rand(d) * 4 + 0.1)
                x = np.sin(x)
                x = x.astype(np.float32)
                self.cached_ds[dataset] = x
            else:
                self.cached_ds[dataset] = self.read_nparray(
                    os.path.join(self.path, dataset.tablename),
                    mmap_mode="r",
                )[: dataset.num_vectors].copy()
        return self.cached_ds[dataset]

    def read_nparray(
        self,
        filename: str,
        mmap_mode: Optional[str] = None,
    ):
        fn = self.download_file_from_blobstore(filename)
        logger.info(f"Loading nparray from {fn}")
        nparray = np.load(fn, mmap_mode=mmap_mode)
        logger.info(f"Loaded nparray {nparray.shape} from {fn}")
        return nparray

    def write_nparray(
        self,
        nparray: np.ndarray,
        filename: str,
    ):
        fn = self.get_local_filepath(filename)
        logger.info(f"Saving nparray {nparray.shape} to {fn}")
        np.save(fn, nparray)
        self.upload_file_to_blobstore(filename)

    def read_json(
        self,
        filename: str,
    ):
        fn = self.download_file_from_blobstore(filename)
        logger.info(f"Loading json {fn}")
        with open(fn, "r") as fp:
            json_dict = json.load(fp)
        logger.info(f"Loaded json {json_dict} from {fn}")
        return json_dict

    def write_json(
        self,
        json_dict: dict[str, Any],
        filename: str,
        overwrite: bool = False,
    ):
        fn = self.get_local_filepath(filename)
        logger.info(f"Saving json {json_dict} to {fn}")
        with open(fn, "w") as fp:
            json.dump(json_dict, fp)
        self.upload_file_to_blobstore(filename, overwrite=overwrite)

    def read_index(
        self,
        filename: str,
        bucket: Optional[str] = None,
        path: Optional[str] = None,
    ):
        fn = self.download_file_from_blobstore(filename, bucket, path)
        logger.info(f"Loading index {fn}")
        ext = os.path.splitext(fn)[1]
        if ext in [".faiss", ".codec", ".index"]:
            index = faiss.read_index(fn)
        elif ext == ".pkl":
            with open(fn, "rb") as model_file:
                model = pickle.load(model_file)
                rcq_coarse_quantizer, itq_encoder = model["model"]
                index = merge_rcq_itq(rcq_coarse_quantizer, itq_encoder)
        logger.info(f"Loaded index from {fn}")
        return index

    def write_index(
        self,
        index: faiss.Index,
        filename: str,
    ):
        fn = self.get_local_filepath(filename)
        logger.info(f"Saving index to {fn}")
        faiss.write_index(index, fn)
        self.upload_file_to_blobstore(filename)
        assert os.path.exists(fn)
        return os.path.getsize(fn)

    def launch_jobs(self, func, params, local=True):
        if local:
            results = [func(p) for p in params]
            return results
        logger.info(f"launching {len(params)} jobs")
        executor = submitit.AutoExecutor(folder="/checkpoint/gsz/jobs")
        executor.update_parameters(
            nodes=1,
            gpus_per_node=8,
            cpus_per_task=80,
            # mem_gb=640,
            tasks_per_node=1,
            name="faiss_benchmark",
            slurm_array_parallelism=512,
            slurm_partition="scavenge",
            slurm_time=4 * 60,
            slurm_constraint="bldg1",
        )
        jobs = executor.map_array(func, params)
        logger.info(f"launched {len(jobs)} jobs")
        for job, param in zip(jobs, params):
            logger.info(f"{job.job_id=} {param[0]=}")
        results = [job.result() for job in jobs]
        print(f"received {len(results)} results")
        return results
