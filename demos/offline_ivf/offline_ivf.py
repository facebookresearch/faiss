# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import numpy as np
import os
from tqdm import tqdm, trange
import sys
import logging
from faiss.contrib.ondisk import merge_ondisk
from faiss.contrib.big_batch_search import big_batch_search
from faiss.contrib.exhaustive_search import knn_ground_truth
from faiss.contrib.evaluation import knn_intersection_measure
from utils import (
    get_intersection_cardinality_frequencies,
    margin,
    is_pretransform_index,
)
from dataset import create_dataset_from_oivf_config

logging.basicConfig(
    format=(
        "%(asctime)s.%(msecs)03d %(levelname)-8s %(threadName)-12s %(message)s"
    ),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

EMBEDDINGS_BATCH_SIZE: int = 100_000
NUM_SUBSAMPLES: int = 100
SMALL_DATA_SAMPLE: int = 10000


class OfflineIVF:
    def __init__(self, cfg, args, nprobe, index_factory_str):
        self.input_d = cfg["d"]
        self.dt = cfg["datasets"][args.xb]["files"][0]["dtype"]
        assert self.input_d > 0
        output_dir = cfg["output"]
        assert os.path.exists(output_dir)
        self.index_factory = index_factory_str
        assert self.index_factory is not None
        self.index_factory_fn = self.index_factory.replace(",", "_")
        self.index_template_file = (
            f"{output_dir}/{args.xb}/{self.index_factory_fn}.empty.faissindex"
        )
        logging.info(f"index template: {self.index_template_file}")

        if not args.xq:
            args.xq = args.xb

        self.by_residual = True
        if args.no_residuals:
            self.by_residual = False

        xb_output_dir = f"{output_dir}/{args.xb}"
        if not os.path.exists(xb_output_dir):
            os.makedirs(xb_output_dir)
        xq_output_dir = f"{output_dir}/{args.xq}"
        if not os.path.exists(xq_output_dir):
            os.makedirs(xq_output_dir)
        search_output_dir = f"{output_dir}/{args.xq}_in_{args.xb}"
        if not os.path.exists(search_output_dir):
            os.makedirs(search_output_dir)
        self.knn_dir = f"{search_output_dir}/knn"
        if not os.path.exists(self.knn_dir):
            os.makedirs(self.knn_dir)
        self.eval_dir = f"{search_output_dir}/eval"
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        self.index = {}  # to keep a reference to opened indices,
        self.ivls = {}  # hstack inverted lists,
        self.index_shards = {}  # and index shards
        self.index_shard_prefix = (
            f"{xb_output_dir}/{self.index_factory_fn}.shard_"
        )
        self.xq_index_shard_prefix = (
            f"{xq_output_dir}/{self.index_factory_fn}.shard_"
        )
        self.index_file = (  # TODO: added back temporarily for evaluate, handle name of non-sharded index file and remove.
            f"{xb_output_dir}/{self.index_factory_fn}.faissindex"
        )
        self.xq_index_file = (
            f"{xq_output_dir}/{self.index_factory_fn}.faissindex"
        )
        self.training_sample = cfg["training_sample"]
        self.evaluation_sample = cfg["evaluation_sample"]
        self.xq_ds = create_dataset_from_oivf_config(cfg, args.xq)
        self.xb_ds = create_dataset_from_oivf_config(cfg, args.xb)
        file_descriptors = self.xq_ds.file_descriptors
        self.file_sizes = [fd.size for fd in file_descriptors]
        self.shard_size = cfg["index_shard_size"]  # ~100GB
        self.nshards = self.xb_ds.size // self.shard_size
        if self.xb_ds.size % self.shard_size != 0:
            self.nshards += 1
        self.xq_nshards = self.xq_ds.size // self.shard_size
        if self.xq_ds.size % self.shard_size != 0:
            self.xq_nshards += 1
        self.nprobe = nprobe
        assert self.nprobe > 0, "Invalid nprobe parameter."
        if "deduper" in cfg:
            self.deduper = cfg["deduper"]
            self.deduper_codec_fn = [
                f"{xb_output_dir}/deduper_codec_{codec.replace(',', '_')}"
                for codec in self.deduper
            ]
            self.deduper_idx_fn = [
                f"{xb_output_dir}/deduper_idx_{codec.replace(',', '_')}"
                for codec in self.deduper
            ]
        else:
            self.deduper = None
        self.k = cfg["k"]
        assert self.k > 0, "Invalid number of neighbours parameter."
        self.knn_output_file_suffix = (
            f"{self.index_factory_fn}_np{self.nprobe}.npy"
        )

        fp = 32
        if self.dt == "float16":
            fp = 16

        self.xq_bs = cfg["query_batch_size"]
        if "metric" in cfg:
            self.metric = eval(f'faiss.{cfg["metric"]}')
        else:
            self.metric = faiss.METRIC_L2

        if "evaluate_by_margin" in cfg:
            self.evaluate_by_margin = cfg["evaluate_by_margin"]
        else:
            self.evaluate_by_margin = False

        os.system("grep -m1 'model name' < /proc/cpuinfo")
        os.system("grep -E 'MemTotal|MemFree' /proc/meminfo")
        os.system("nvidia-smi")
        os.system("nvcc --version")

        self.knn_queries_memory_limit = 4 * 1024 * 1024 * 1024  # 4 GB
        self.knn_vectors_memory_limit = 8 * 1024 * 1024 * 1024  # 8 GB

    def input_stats(self):
        """
        Trains the index using a subsample of the first chunk of data in the database and saves it in the template file (with no vectors added).
        """
        xb_sample = self.xb_ds.get_first_n(self.training_sample, np.float32)
        logging.info(f"input shape: {xb_sample.shape}")
        logging.info("running MatrixStats on training sample...")
        logging.info(faiss.MatrixStats(xb_sample).comments)
        logging.info("done")

    def dedupe(self):
        logging.info(self.deduper)
        if self.deduper is None:
            logging.info("No deduper configured")
            return
        codecs = []
        codesets = []
        idxs = []
        for factory, filename in zip(self.deduper, self.deduper_codec_fn):
            if os.path.exists(filename):
                logging.info(f"loading trained dedupe codec: {filename}")
                codec = faiss.read_index(filename)
            else:
                logging.info(f"training dedupe codec: {factory}")
                codec = faiss.index_factory(self.input_d, factory)
                xb_sample = np.unique(
                    self.xb_ds.get_first_n(100_000, np.float32), axis=0
                )
                faiss.ParameterSpace().set_index_parameter(codec, "verbose", 1)
                codec.train(xb_sample)
                logging.info(f"writing trained dedupe codec: {filename}")
                faiss.write_index(codec, filename)
            codecs.append(codec)
            codesets.append(faiss.CodeSet(codec.sa_code_size()))
            idxs.append(np.empty((0,), dtype=np.uint32))
        bs = 1_000_000
        i = 0
        for buffer in tqdm(self._iterate_transformed(self.xb_ds, 0, bs, np.float32)):
            for j in range(len(codecs)):
                codec, codeset, idx = codecs[j], codesets[j], idxs[j]
                uniq = codeset.insert(codec.sa_encode(buffer))
                idxs[j] = np.append(
                    idx,
                    np.arange(i, i + buffer.shape[0], dtype=np.uint32)[uniq],
                )
            i += buffer.shape[0]
        for idx, filename in zip(idxs, self.deduper_idx_fn):
            logging.info(f"writing {filename}, shape: {idx.shape}")
            np.save(filename, idx)
        logging.info("done")

    def train_index(self):
        """
        Trains the index using a subsample of the first chunk of data in the database and saves it in the template file (with no vectors added).
        """
        assert not os.path.exists(self.index_template_file), (
            "The train command has been ran, the index template file already"
            " exists."
        )
        xb_sample = np.unique(
            self.xb_ds.get_first_n(self.training_sample, np.float32), axis=0
        )
        logging.info(f"input shape: {xb_sample.shape}")
        index = faiss.index_factory(
            self.input_d, self.index_factory, self.metric
        )
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        index_ivf.by_residual = True
        faiss.ParameterSpace().set_index_parameter(index, "verbose", 1)
        logging.info("running training...")
        index.train(xb_sample)
        logging.info(f"writing trained index {self.index_template_file}...")
        faiss.write_index(index, self.index_template_file)
        logging.info("done")

    def _iterate_transformed(self, ds, start, batch_size, dt):
        assert os.path.exists(self.index_template_file)
        index = faiss.read_index(self.index_template_file)
        if is_pretransform_index(index):
            vt = index.chain.at(0)  # fetch pretransform
            for buffer in ds.iterate(start, batch_size, dt):
                yield vt.apply(buffer)
        else:
            for buffer in ds.iterate(start, batch_size, dt):
                yield buffer

    def index_shard(self):
        assert os.path.exists(self.index_template_file)
        index = faiss.read_index(self.index_template_file)
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        assert self.nprobe <= index_ivf.quantizer.ntotal, (
            f"the number of vectors {index_ivf.quantizer.ntotal} is not enough"
            f" to retrieve {self.nprobe} neighbours, check."
        )
        cpu_quantizer = index_ivf.quantizer
        gpu_quantizer = faiss.index_cpu_to_all_gpus(cpu_quantizer)

        for i in range(0, self.nshards):
            sfn = f"{self.index_shard_prefix}{i}"
            try:
                index.reset()
                index_ivf.quantizer = gpu_quantizer
                with open(sfn, "xb"):
                    start = i * self.shard_size
                    jj = 0
                    embeddings_batch_size = min(
                        EMBEDDINGS_BATCH_SIZE, self.shard_size
                    )
                    assert (
                        self.shard_size % embeddings_batch_size == 0
                        or EMBEDDINGS_BATCH_SIZE % embeddings_batch_size == 0
                    ), (
                        f"the shard size {self.shard_size} and embeddings"
                        f" shard size  {EMBEDDINGS_BATCH_SIZE} are not"
                        " divisible"
                    )

                    for xb_j in tqdm(
                        self._iterate_transformed(
                            self.xb_ds,
                            start,
                            embeddings_batch_size,
                            np.float32,
                        ),
                        file=sys.stdout,
                    ):
                        if is_pretransform_index(index):
                            assert xb_j.shape[1] == index.chain.at(0).d_out
                            index_ivf.add_with_ids(
                                xb_j,
                                np.arange(start + jj, start + jj + xb_j.shape[0]),
                            )
                        else:
                            assert xb_j.shape[1] == index.d
                            index.add_with_ids(
                                xb_j,
                                np.arange(start + jj, start + jj + xb_j.shape[0]),
                            )
                        jj += xb_j.shape[0]
                        logging.info(jj)
                        assert (
                            jj <= self.shard_size
                        ), f"jj {jj} and shard_zide {self.shard_size}"
                        if jj == self.shard_size:
                            break
                logging.info(f"writing {sfn}...")
                index_ivf.quantizer = cpu_quantizer
                faiss.write_index(index, sfn)
            except FileExistsError:
                logging.info(f"skipping shard: {i}")
                continue
        logging.info("done")

    def merge_index(self):
        ivf_file = f"{self.index_file}.ivfdata"

        assert os.path.exists(self.index_template_file)
        assert not os.path.exists(
            ivf_file
        ), f"file with embeddings data {ivf_file} not found, check."
        assert not os.path.exists(self.index_file)
        index = faiss.read_index(self.index_template_file)
        block_fnames = [
            f"{self.index_shard_prefix}{i}" for i in range(self.nshards)
        ]
        for fn in block_fnames:
            assert os.path.exists(fn)
        logging.info(block_fnames)
        logging.info("merging...")
        merge_ondisk(index, block_fnames, ivf_file)
        logging.info("writing index...")
        faiss.write_index(index, self.index_file)
        logging.info("done")

    def _cached_search(
        self,
        sample,
        xq_ds,
        xb_ds,
        idx_file,
        vecs_file,
        I_file,
        D_file,
        index_file=None,
        nprobe=None,
    ):
        if not os.path.exists(I_file):
            assert not os.path.exists(I_file), f"file {I_file} does not exist "
            assert not os.path.exists(D_file), f"file {D_file} does not exist "
            xq = xq_ds.sample(sample, idx_file, vecs_file)

            if index_file:
                D, I = self._index_nonsharded_search(index_file, xq, nprobe)
            else:
                logging.info("ground truth computations")
                db_iterator = xb_ds.iterate(0, 100_000, np.float32)
                D, I = knn_ground_truth(
                    xq, db_iterator, self.k, metric_type=self.metric
                )
                assert np.amin(I) >= 0

            np.save(I_file, I)
            np.save(D_file, D)
        else:
            assert os.path.exists(idx_file), f"file {idx_file} does not exist "
            assert os.path.exists(
                vecs_file
            ), f"file {vecs_file} does not exist "
            assert os.path.exists(I_file), f"file {I_file} does not exist "
            assert os.path.exists(D_file), f"file {D_file} does not exist "
            I = np.load(I_file)
            D = np.load(D_file)
        assert I.shape == (sample, self.k), f"{I_file} shape mismatch"
        assert D.shape == (sample, self.k), f"{D_file} shape mismatch"
        return (D, I)

    def _index_search(self, index_shard_prefix, xq, nprobe):
        assert nprobe is not None
        logging.info(
            f"open sharded index: {index_shard_prefix}, {self.nshards}"
        )
        index = self._open_sharded_index(index_shard_prefix)
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        logging.info(f"setting nprobe to {nprobe}")
        index_ivf.nprobe = nprobe
        return index.search(xq, self.k)

    def _index_nonsharded_search(self, index_file, xq, nprobe):
        assert nprobe is not None
        logging.info(f"index {index_file}")
        assert os.path.exists(index_file), f"file {index_file} does not exist "
        index = faiss.read_index(index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logging.info(f"index size {index.ntotal} ")
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        logging.info(f"setting nprobe to {nprobe}")
        index_ivf.nprobe = nprobe
        return index.search(xq, self.k)

    def _refine_distances(self, xq_ds, idx, xb_ds, I):
        xq = xq_ds.get(idx).repeat(self.k, axis=0)
        xb = xb_ds.get(I.reshape(-1))
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            return (xq * xb).sum(axis=1).reshape(I.shape)
        elif self.metric == faiss.METRIC_L2:
            return ((xq - xb) ** 2).sum(axis=1).reshape(I.shape)
        else:
            raise ValueError(f"metric not supported {self.metric}")

    def evaluate(self):
        self._evaluate(
            self.index_factory_fn,
            self.index_file,
            self.xq_index_file,
            self.nprobe,
        )

    def _evaluate(self, index_factory_fn, index_file, xq_index_file, nprobe):
        idx_a_file = f"{self.eval_dir}/idx_a.npy"
        idx_b_gt_file = f"{self.eval_dir}/idx_b_gt.npy"
        idx_b_ann_file = (
            f"{self.eval_dir}/idx_b_ann_{index_factory_fn}_np{nprobe}.npy"
        )
        vecs_a_file = f"{self.eval_dir}/vecs_a.npy"
        vecs_b_gt_file = f"{self.eval_dir}/vecs_b_gt.npy"
        vecs_b_ann_file = (
            f"{self.eval_dir}/vecs_b_ann_{index_factory_fn}_np{nprobe}.npy"
        )
        D_a_gt_file = f"{self.eval_dir}/D_a_gt.npy"
        D_a_ann_file = (
            f"{self.eval_dir}/D_a_ann_{index_factory_fn}_np{nprobe}.npy"
        )
        D_a_ann_refined_file = f"{self.eval_dir}/D_a_ann_refined_{index_factory_fn}_np{nprobe}.npy"
        D_b_gt_file = f"{self.eval_dir}/D_b_gt.npy"
        D_b_ann_file = (
            f"{self.eval_dir}/D_b_ann_{index_factory_fn}_np{nprobe}.npy"
        )
        D_b_ann_gt_file = (
            f"{self.eval_dir}/D_b_ann_gt_{index_factory_fn}_np{nprobe}.npy"
        )
        I_a_gt_file = f"{self.eval_dir}/I_a_gt.npy"
        I_a_ann_file = (
            f"{self.eval_dir}/I_a_ann_{index_factory_fn}_np{nprobe}.npy"
        )
        I_b_gt_file = f"{self.eval_dir}/I_b_gt.npy"
        I_b_ann_file = (
            f"{self.eval_dir}/I_b_ann_{index_factory_fn}_np{nprobe}.npy"
        )
        I_b_ann_gt_file = (
            f"{self.eval_dir}/I_b_ann_gt_{index_factory_fn}_np{nprobe}.npy"
        )
        margin_gt_file = f"{self.eval_dir}/margin_gt.npy"
        margin_refined_file = (
            f"{self.eval_dir}/margin_refined_{index_factory_fn}_np{nprobe}.npy"
        )
        margin_ann_file = (
            f"{self.eval_dir}/margin_ann_{index_factory_fn}_np{nprobe}.npy"
        )

        logging.info("exact search forward")
        # xq -> xb AKA a -> b
        D_a_gt, I_a_gt = self._cached_search(
            self.evaluation_sample,
            self.xq_ds,
            self.xb_ds,
            idx_a_file,
            vecs_a_file,
            I_a_gt_file,
            D_a_gt_file,
        )
        idx_a = np.load(idx_a_file)

        logging.info("approximate search forward")
        D_a_ann, I_a_ann = self._cached_search(
            self.evaluation_sample,
            self.xq_ds,
            self.xb_ds,
            idx_a_file,
            vecs_a_file,
            I_a_ann_file,
            D_a_ann_file,
            index_file,
            nprobe,
        )

        logging.info(
            "calculate refined distances on approximate search forward"
        )
        if os.path.exists(D_a_ann_refined_file):
            D_a_ann_refined = np.load(D_a_ann_refined_file)
            assert D_a_ann.shape == D_a_ann_refined.shape
        else:
            D_a_ann_refined = self._refine_distances(
                self.xq_ds, idx_a, self.xb_ds, I_a_ann
            )
            np.save(D_a_ann_refined_file, D_a_ann_refined)

        if self.evaluate_by_margin:
            k_extract = self.k
            margin_threshold = 1.05
            logging.info(
                "exact search backward from the k_extract NN results of"
                " forward search"
            )
            # xb -> xq AKA b -> a
            D_a_b_gt = D_a_gt[:, :k_extract].ravel()
            idx_b_gt = I_a_gt[:, :k_extract].ravel()
            assert len(idx_b_gt) == self.evaluation_sample * k_extract
            np.save(idx_b_gt_file, idx_b_gt)
            # exact search
            D_b_gt, _ = self._cached_search(
                len(idx_b_gt),
                self.xb_ds,
                self.xq_ds,
                idx_b_gt_file,
                vecs_b_gt_file,
                I_b_gt_file,
                D_b_gt_file,
            )  # xb and xq ^^^ are inverted

            logging.info("margin on exact search")
            margin_gt = margin(
                self.evaluation_sample,
                idx_a,
                idx_b_gt,
                D_a_b_gt,
                D_a_gt,
                D_b_gt,
                self.k,
                k_extract,
                margin_threshold,
            )
            np.save(margin_gt_file, margin_gt)

            logging.info(
                "exact search backward from the k_extract NN results of"
                " approximate forward search"
            )
            D_a_b_refined = D_a_ann_refined[:, :k_extract].ravel()
            idx_b_ann = I_a_ann[:, :k_extract].ravel()
            assert len(idx_b_ann) == self.evaluation_sample * k_extract
            np.save(idx_b_ann_file, idx_b_ann)
            # exact search
            D_b_ann_gt, _ = self._cached_search(
                len(idx_b_ann),
                self.xb_ds,
                self.xq_ds,
                idx_b_ann_file,
                vecs_b_ann_file,
                I_b_ann_gt_file,
                D_b_ann_gt_file,
            )  # xb and xq ^^^ are inverted

            logging.info("refined margin on approximate search")
            margin_refined = margin(
                self.evaluation_sample,
                idx_a,
                idx_b_ann,
                D_a_b_refined,
                D_a_gt,  # not D_a_ann_refined(!)
                D_b_ann_gt,
                self.k,
                k_extract,
                margin_threshold,
            )
            np.save(margin_refined_file, margin_refined)

            D_b_ann, I_b_ann = self._cached_search(
                len(idx_b_ann),
                self.xb_ds,
                self.xq_ds,
                idx_b_ann_file,
                vecs_b_ann_file,
                I_b_ann_file,
                D_b_ann_file,
                xq_index_file,
                nprobe,
            )

            D_a_b_ann = D_a_ann[:, :k_extract].ravel()

            logging.info("approximate search margin")

            margin_ann = margin(
                self.evaluation_sample,
                idx_a,
                idx_b_ann,
                D_a_b_ann,
                D_a_ann,
                D_b_ann,
                self.k,
                k_extract,
                margin_threshold,
            )
            np.save(margin_ann_file, margin_ann)

        logging.info("intersection")
        logging.info(I_a_gt)
        logging.info(I_a_ann)

        for i in range(1, self.k + 1):
            logging.info(
                f"{i}: {knn_intersection_measure(I_a_gt[:,:i], I_a_ann[:,:i])}"
            )

        logging.info(f"mean of gt distances: {D_a_gt.mean()}")
        logging.info(f"mean of approx distances: {D_a_ann.mean()}")
        logging.info(f"mean of refined distances: {D_a_ann_refined.mean()}")

        logging.info("intersection cardinality frequencies")
        logging.info(get_intersection_cardinality_frequencies(I_a_ann, I_a_gt))

        logging.info("done")
        pass

    def _knn_function(self, xq, xb, k, metric, thread_id=None):
        try:
            return faiss.knn_gpu(
                self.all_gpu_resources[thread_id],
                xq,
                xb,
                k,
                metric=metric,
                device=thread_id,
                vectorsMemoryLimit=self.knn_vectors_memory_limit,
                queriesMemoryLimit=self.knn_queries_memory_limit,
            )
        except Exception:
            logging.info(f"knn_function failed: {xq.shape}, {xb.shape}")
            raise

    def _coarse_quantize(self, index_ivf, xq, nprobe):
        assert nprobe <= index_ivf.quantizer.ntotal
        quantizer = faiss.index_cpu_to_all_gpus(index_ivf.quantizer)
        bs = 100_000
        nq = len(xq)
        q_assign = np.empty((nq, nprobe), dtype="int32")
        for i0 in trange(0, nq, bs):
            i1 = min(nq, i0 + bs)
            _, q_assign_i = quantizer.search(xq[i0:i1], nprobe)
            q_assign[i0:i1] = q_assign_i
        return q_assign

    def search(self):
        logging.info(f"search: {self.knn_dir}")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")

        ngpu = faiss.get_num_gpus()
        logging.info(f"number of gpus: {ngpu}")
        self.all_gpu_resources = [
            faiss.StandardGpuResources() for _ in range(ngpu)
        ]
        self._knn_function(
            np.zeros((10, 10), dtype=np.float16),
            np.zeros((10, 10), dtype=np.float16),
            self.k,
            metric=self.metric,
            thread_id=0,
        )

        index = self._open_sharded_index()
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        logging.info(f"setting nprobe to {self.nprobe}")
        index_ivf.nprobe = self.nprobe
        # quantizer = faiss.index_cpu_to_all_gpus(index_ivf.quantizer)
        for i in range(0, self.xq_ds.size, self.xq_bs):
            Ifn = f"{self.knn_dir}/I{(i):010}_{self.knn_output_file_suffix}"
            Dfn = f"{self.knn_dir}/D_approx{(i):010}_{self.knn_output_file_suffix}"
            CPfn = f"{self.knn_dir}/CP{(i):010}_{self.knn_output_file_suffix}"

            if slurm_job_id:
                worker_record = (
                    self.knn_dir
                    + f"/record_{(i):010}_{self.knn_output_file_suffix}.txt"
                )
                if not os.path.exists(worker_record):
                    logging.info(
                        f"creating record file {worker_record} and saving job"
                        f" id: {slurm_job_id}"
                    )
                    with open(worker_record, "w") as h:
                        h.write(slurm_job_id)
                else:
                    old_slurm_id = open(worker_record, "r").read()
                    logging.info(
                        f"old job slurm id {old_slurm_id} and current job id:"
                        f" {slurm_job_id}"
                    )
                    if old_slurm_id == slurm_job_id:
                        if os.path.getsize(Ifn) == 0:
                            logging.info(
                                f"cleaning up zero length files {Ifn} and"
                                f" {Dfn}"
                            )
                            os.remove(Ifn)
                            os.remove(Dfn)

            try:
                if is_pretransform_index(index):
                    d = index.chain.at(0).d_out
                else:
                    d = self.input_d
                with open(Ifn, "xb") as f, open(Dfn, "xb") as g:
                    xq_i = np.empty(
                        shape=(self.xq_bs, d), dtype=np.float16
                    )
                    q_assign = np.empty(
                        (self.xq_bs, self.nprobe), dtype=np.int32
                    )
                    j = 0
                    quantizer = faiss.index_cpu_to_all_gpus(
                        index_ivf.quantizer
                    )
                    for xq_i_j in tqdm(
                        self._iterate_transformed(
                            self.xq_ds, i, min(100_000, self.xq_bs), np.float16
                        ),
                        file=sys.stdout,
                    ):
                        xq_i[j:j + xq_i_j.shape[0]] = xq_i_j
                        (
                            _,
                            q_assign[j:j + xq_i_j.shape[0]],
                        ) = quantizer.search(xq_i_j, self.nprobe)
                        j += xq_i_j.shape[0]
                        assert j <= xq_i.shape[0]
                        if j == xq_i.shape[0]:
                            break
                    xq_i = xq_i[:j]
                    q_assign = q_assign[:j]

                    assert q_assign.shape == (xq_i.shape[0], index_ivf.nprobe)
                    del quantizer
                    logging.info(f"computing: {Ifn}")
                    logging.info(f"computing: {Dfn}")
                    prefetch_threads = faiss.get_num_gpus()
                    D_ann, I = big_batch_search(
                        index_ivf,
                        xq_i,
                        self.k,
                        verbose=10,
                        method="knn_function",
                        knn=self._knn_function,
                        threaded=faiss.get_num_gpus() * 8,
                        use_float16=True,
                        prefetch_threads=prefetch_threads,
                        computation_threads=faiss.get_num_gpus(),
                        q_assign=q_assign,
                        checkpoint=CPfn,
                        checkpoint_freq=7200,  # in seconds
                    )
                    assert (
                        np.amin(I) >= 0
                    ), f"{I}, there exists negative indices, check"
                    logging.info(f"saving: {Ifn}")
                    np.save(f, I)
                    logging.info(f"saving: {Dfn}")
                    np.save(g, D_ann)

                    if os.path.exists(CPfn):
                        logging.info(f"removing: {CPfn}")
                        os.remove(CPfn)

            except FileExistsError:
                logging.info(f"skipping {Ifn}, already exists")
                logging.info(f"skipping {Dfn}, already exists")
                continue

    def _open_index_shard(self, fn):
        if fn in self.index_shards:
            index_shard = self.index_shards[fn]
        else:
            logging.info(f"open index shard: {fn}")
            index_shard = faiss.read_index(
                fn, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
            self.index_shards[fn] = index_shard
        return index_shard

    def _open_sharded_index(self, index_shard_prefix=None):
        if index_shard_prefix is None:
            index_shard_prefix = self.index_shard_prefix
        if index_shard_prefix in self.index:
            return self.index[index_shard_prefix]
        assert os.path.exists(
            self.index_template_file
        ), f"file {self.index_template_file} does not exist "
        logging.info(f"open index template: {self.index_template_file}")
        index = faiss.read_index(self.index_template_file)
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        ilv = faiss.InvertedListsPtrVector()
        for i in range(self.nshards):
            fn = f"{index_shard_prefix}{i}"
            assert os.path.exists(fn), f"file {fn} does not exist "
            logging.info(fn)
            index_shard = self._open_index_shard(fn)
            il = faiss.downcast_index(
                faiss.extract_index_ivf(index_shard)
            ).invlists
            ilv.push_back(il)
        hsil = faiss.HStackInvertedLists(ilv.size(), ilv.data())
        index_ivf.replace_invlists(hsil, False)
        self.ivls[index_shard_prefix] = hsil
        self.index[index_shard_prefix] = index
        return index

    def index_shard_stats(self):
        for i in range(self.nshards):
            fn = f"{self.index_shard_prefix}{i}"
            assert os.path.exists(fn)
            index = faiss.read_index(
                fn, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
            index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
            il = index_ivf.invlists
            il.print_stats()

    def index_stats(self):
        index = self._open_sharded_index()
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        il = index_ivf.invlists
        list_sizes = [il.list_size(i) for i in range(il.nlist)]
        logging.info(np.max(list_sizes))
        logging.info(np.mean(list_sizes))
        logging.info(np.argmax(list_sizes))
        logging.info("index_stats:")
        il.print_stats()

    def consistency_check(self):
        logging.info("consistency-check")

        logging.info("index template...")

        assert os.path.exists(self.index_template_file)
        index = faiss.read_index(self.index_template_file)

        offset = 0  # 2**24
        assert self.shard_size > offset + SMALL_DATA_SAMPLE

        logging.info("index shards...")
        for i in range(self.nshards):
            r = i * self.shard_size + offset
            xb = next(self.xb_ds.iterate(r, SMALL_DATA_SAMPLE, np.float32))
            fn = f"{self.index_shard_prefix}{i}"
            assert os.path.exists(fn), f"There is no index shard file {fn}"
            index = self._open_index_shard(fn)
            index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
            index_ivf.nprobe = 1
            _, I = index.search(xb, 100)
            for j in range(SMALL_DATA_SAMPLE):
                assert np.where(I[j] == j + r)[0].size > 0, (
                    f"I[j]: {I[j]}, j: {j}, i: {i}, shard_size:"
                    f" {self.shard_size}"
                )

        logging.info("merged index...")
        index = self._open_sharded_index()
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        index_ivf.nprobe = 1
        for i in range(self.nshards):
            r = i * self.shard_size + offset
            xb = next(self.xb_ds.iterate(r, SMALL_DATA_SAMPLE, np.float32))
            _, I = index.search(xb, 100)
            for j in range(SMALL_DATA_SAMPLE):
                assert np.where(I[j] == j + r)[0].size > 0, (
                    f"I[j]: {I[j]}, j: {j}, i: {i}, shard_size:"
                    f" {self.shard_size}")

        logging.info("search results...")
        index_ivf.nprobe = self.nprobe
        for i in range(0, self.xq_ds.size, self.xq_bs):
            Ifn = f"{self.knn_dir}/I{i:010}_{self.index_factory_fn}_np{self.nprobe}.npy"
            assert os.path.exists(Ifn)
            assert os.path.getsize(Ifn) > 0, f"The file {Ifn} is empty."
            logging.info(Ifn)
            I = np.load(Ifn, mmap_mode="r")

            assert I.shape[1] == self.k
            assert I.shape[0] == min(self.xq_bs, self.xq_ds.size - i)
            assert np.all(I[:, 1] >= 0)

            Dfn = f"{self.knn_dir}/D_approx{i:010}_{self.index_factory_fn}_np{self.nprobe}.npy"
            assert os.path.exists(Dfn)
            assert os.path.getsize(Dfn) > 0, f"The file {Dfn} is empty."
            logging.info(Dfn)
            D = np.load(Dfn, mmap_mode="r")
            assert D.shape == I.shape

            xq = next(self.xq_ds.iterate(i, SMALL_DATA_SAMPLE, np.float32))
            D_online, I_online = index.search(xq, self.k)
            assert (
                np.where(I[:SMALL_DATA_SAMPLE] == I_online)[0].size
                / (self.k * SMALL_DATA_SAMPLE)
                > 0.95
            ), (
                "the ratio is"
                f" {np.where(I[:SMALL_DATA_SAMPLE] == I_online)[0].size / (self.k * SMALL_DATA_SAMPLE)}"
            )
            assert np.allclose(
                D[:SMALL_DATA_SAMPLE].sum(axis=1),
                D_online.sum(axis=1),
                rtol=0.01,
            ), (
                "the difference is"
                f" {D[:SMALL_DATA_SAMPLE].sum(axis=1), D_online.sum(axis=1)}"
            )

        logging.info("done")
