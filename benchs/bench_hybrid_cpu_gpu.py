# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import time
from multiprocessing.pool import ThreadPool

import faiss
import numpy as np

try:
    from faiss.contrib.datasets_fb import dataset_from_name
except ImportError:
    from faiss.contrib.datasets import dataset_from_name

from faiss.contrib.evaluation import OperatingPointsWithRanges
from faiss.contrib.ivf_tools import replace_ivf_quantizer

#################################################################
# Preassigned search functions
#################################################################


def search_preassigned(xq, k, index, quantizer, batch_size=0):
    """
    Explicitly call the coarse quantizer and the search_preassigned
    on the index.
    """
    n, d = xq.shape
    nprobe = index.nprobe
    if batch_size == 0:
        batch_size = n + 1
    D = np.empty((n, k), dtype='float32')
    I = np.empty((n, k), dtype='int64')
    for i0 in range(0, n, batch_size):
        Dq, Iq = quantizer.search(xq[i0:i0 + batch_size], nprobe)
        D[i0:i0 + batch_size], I[i0:i0 + batch_size] = \
            index.search_preassigned(xq[i0:i0 + batch_size], k, Iq, Dq)
    return D, I


def tiled_search_preassigned(xq, k, index, quantizer, batch_size=32768):
    """
    Explicitly call the coarse quantizer and the search_preassigned
    on the index. Allow overlapping between coarse quantization and
    scanning the inverted lists.
    """
    n, d = xq.shape

    # prepare a thread that will run the quantizer
    qq_pool = ThreadPool(1)
    nprobe = index.nprobe

    def coarse_quant(i0):
        if i0 >= n:
            return None
        i1 = min(i0 + batch_size, n)
        return quantizer.search(xq[i0:i1], nprobe)

    D = np.empty((n, k), dtype='float32')
    I = np.empty((n, k), dtype='int64')
    qq = coarse_quant(0)

    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        qq_next = qq_pool.apply_async(coarse_quant, (i0 + batch_size, ))
        Dq, Iq = qq
        index.search_preassigned(
            xq[i0:i1], k, Iq=Iq, Dq=Dq, I=I[i0:i1], D=D[i0:i1])
        qq = qq_next.get()

    qq_pool.close()
    return D, I


#################################################################
# IVF index objects with a separate coarse quantizer
#################################################################

class SeparateCoarseQuantizationIndex:
    """
    Separately manage the coarse quantizer and the IVF index.
    """

    def __init__(self, quantizer, index, bs=-1, seq_tiling=False):
        self.index = index
        self.index_ivf = extract_index_ivf(index)
        if isinstance(self.index_ivf, faiss.IndexIVF):
            self.index_ivf.parallel_mode
            self.index_ivf.parallel_mode = 3

        self.quantizer = quantizer
        assert self.quantizer.d == self.index_ivf.d
        # populate quantizer if it was not done before
        if quantizer.ntotal > 0:
            assert quantizer.ntotal == self.index_ivf.nlist
        else:
            centroids = self.index_ivf.quantizer.reconstruct_n()
            print(f"adding centroids size {centroids.shape} to quantizer")
            quantizer.train(centroids)
            quantizer.add(centroids)
        self.bs = bs
        self.seq_tiling = seq_tiling

    def search(self, xq, k):
        # perform coarse quantization
        if isinstance(self.index, faiss.IndexPreTransform):
            # print("applying pre-transform")
            assert self.index.chain.size() == 1
            xq = self.index.chain.at(0).apply(xq)
        if self.bs <= 0:
            # non batched
            nprobe = self.index_ivf.nprobe
            Dq, Iq = self.quantizer.search(xq, nprobe)

            return self.index_ivf.search_preassigned(xq, k, Iq, Dq)
        if self.seq_tiling:
            return search_preassigned(
                xq, k, self.index_ivf, self.quantizer, self.bs)
        else:
            return tiled_search_preassigned(
                xq, k, self.index_ivf, self.quantizer, self.bs)


class ShardedGPUIndex:
    """
    Multiple GPU indexes, each on its GPU, with a common coarse quantizer.
    The Python version of IndexShardsIVF
    """
    def __init__(self, quantizer, index, bs=-1, seq_tiling=False):
        self.quantizer = quantizer
        self.cpu_index = index
        if isinstance(index, faiss.IndexPreTransform):
            index = faiss.downcast_index(index.index)
        ngpu = index.count()
        self.pool = ThreadPool(ngpu)
        self.bs = bs
        if bs > 0:
            self.q_pool = ThreadPool(1)

    def __del__(self):
        self.pool.close()
        if self.bs > 0:
            self.q_pool.close()

    def search(self, xq, k):
        nq = len(xq)
        # perform coarse quantization
        index = self.cpu_index
        if isinstance(self.cpu_index, faiss.IndexPreTransform):
            assert index.chain.size() == 1
            xq = self.cpu_index.chain.at(0).apply(xq)
            index = faiss.downcast_index(index.index)
        ngpu = index.count()
        sub_index_0 = faiss.downcast_index(index.at(0))
        nprobe = sub_index_0.nprobe

        Dall = np.empty((ngpu, nq, k), dtype='float32')
        Iall = np.empty((ngpu, nq, k), dtype='int64')
        bs = self.bs
        if bs <= 0:

            Dq, Iq = self.quantizer.search(xq, nprobe)

            def do_search(rank):
                gpu_index = faiss.downcast_index(index.at(rank))
                Dall[rank], Iall[rank] = gpu_index.search_preassigned(
                    xq, k, Iq, Dq)
            list(self.pool.map(do_search, range(ngpu)))
        else:
            qq_pool = self.q_pool
            bs = self.bs

            def coarse_quant(i0):
                if i0 >= nq:
                    return None
                return self.quantizer.search(xq[i0:i0 + bs], nprobe)

            def do_search(rank, i0, qq):
                gpu_index = faiss.downcast_index(index.at(rank))
                Dq, Iq = qq
                Dall[rank, i0:i0 + bs], Iall[rank, i0:i0 + bs] = \
                    gpu_index.search_preassigned(xq[i0:i0 + bs], k, Iq, Dq)

            qq = coarse_quant(0)

            for i0 in range(0, nq, bs):
                qq_next = qq_pool.apply_async(coarse_quant, (i0 + bs, ))
                list(self.pool.map(
                    lambda rank: do_search(rank, i0, qq),
                    range(ngpu)
                ))
                qq = qq_next.get()

        return faiss.merge_knn_results(Dall, Iall)


def extract_index_ivf(index):
    """ extract the IVF sub-index from the index, supporting GpuIndexes
    as well """
    try:
        return faiss.extract_index_ivf(index)
    except RuntimeError:
        if index.__class__ == faiss.IndexPreTransform:
            index = faiss.downcast_index(index.index)
        if isinstance(index, faiss.GpuIndexIVF):
            return index
        raise RuntimeError(f"could not extract IVF index from {index}")


def set_index_parameter(index, name, val):
    """
    Index parameter setting that works on the index lookalikes defined above
    """
    if index.__class__ == SeparateCoarseQuantizationIndex:
        if name == "nprobe":
            set_index_parameter(index.index_ivf, name, val)
        elif name.startswith("quantizer_"):
            set_index_parameter(
                index.quantizer, name[name.find("_") + 1:], val)
        else:
            raise RuntimeError()
        return

    if index.__class__ == ShardedGPUIndex:
        if name == "nprobe":
            set_index_parameter(index.cpu_index, name, val)
        elif name.startswith("quantizer_"):
            set_index_parameter(
                index.quantizer, name[name.find("_") + 1:], val)
        else:
            raise RuntimeError()
        return

    # then it's a Faiss index
    index = faiss.downcast_index(index)

    if isinstance(index, faiss.IndexPreTransform):
        set_index_parameter(index.index, name, val)
    elif isinstance(index, faiss.IndexShardsIVF):
        if name != "nprobe" and name.startswith("quantizer_"):
            set_index_parameter(
                index.quantizer, name[name.find("_") + 1:], val)
        else:
            for i in range(index.count()):
                sub_index = index.at(i)
                set_index_parameter(sub_index, name, val)
    elif (isinstance(index, faiss.IndexShards) or
          isinstance(index, faiss.IndexReplicas)):
        for i in range(index.count()):
            sub_index = index.at(i)
            set_index_parameter(sub_index, name, val)
    elif name.startswith("quantizer_"):
        index_ivf = extract_index_ivf(index)
        set_index_parameter(
            index_ivf.quantizer, name[name.find("_") + 1:], val)
    elif name == "efSearch":
        index.hnsw.efSearch
        index.hnsw.efSearch = int(val)
    elif name == "nprobe":
        index_ivf = extract_index_ivf(index)
        index_ivf.nprobe
        index_ivf.nprobe = int(val)
    else:
        raise RuntimeError(f"could not set param {name} on {index}")


#####################################################################
# Driver routine
#####################################################################


def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')
    aa('--nq', type=int, default=int(10e5),
       help="nb queries (queries will be duplicated if below that number")
    aa('--db', default='bigann10M', help='dataset')

    group = parser.add_argument_group('index options')
    aa('--indexname', default="", help="override index name")
    aa('--mmap', default=False, action='store_true', help='mmap index')
    aa('--shard_type', default=1, type=int, help="set type of sharding")
    aa('--useFloat16', default=False, action='store_true',
       help='GPU cloner options')
    aa('--useFloat16CoarseQuantizer', default=False, action='store_true',
       help='GPU cloner options')
    aa('--usePrecomputed', default=False, action='store_true',
       help='GPU cloner options')
    group = parser.add_argument_group('search options')
    aa('--k', type=int, default=100)
    aa('--search_type', default="cpu",
        choices=[
            "cpu", "gpu", "gpu_flat_quantizer",
            "cpu_flat_gpu_quantizer", "gpu_tiled", "gpu_ivf_quantizer",
            "multi_gpu", "multi_gpu_flat_quantizer",
            "multi_gpu_sharded", "multi_gpu_flat_quantizer_sharded",
            "multi_gpu_sharded1", "multi_gpu_sharded1_flat",
            "multi_gpu_sharded1_ivf",
            "multi_gpu_Csharded1", "multi_gpu_Csharded1_flat",
            "multi_gpu_Csharded1_ivf",
        ],
        help="how to search"
    )
    aa('--ivf_quant_nlist', type=int, default=1024,
       help="nb of invlists for IVF quantizer")
    aa('--batch_size', type=int, default=-1,
       help="batch size for tiled CPU / GPU computation (-1= no tiling)")
    aa('--n_autotune', type=int, default=300,
        help="max nb of auto-tuning steps")
    aa('--nt', type=int, default=-1, help="force number of CPU threads to this")

    group = parser.add_argument_group('output options')
    aa('--quiet', default=False, action="store_true")
    aa('--stats', default="", help="pickle to store output stats")

    args = parser.parse_args()
    print("args:", args)

    if not args.quiet:
        # log some stats about the machine
        os.system("grep -m1 'model name' < /proc/cpuinfo")
        os.system("grep -E 'MemTotal|MemFree' /proc/meminfo")
        os.system("nvidia-smi")

    print("prepare dataset", args.db)
    ds = dataset_from_name(args.db)
    print(ds)

    print("Faiss nb GPUs:", faiss.get_num_gpus())

    xq = ds.get_queries()
    if args.nq > len(xq):
        xqx = []
        n = 0
        while n < args.nq:
            xqx.append(xq[:args.nq - n])
            n += len(xqx[-1])
        print(f"increased nb queries from {len(xq)} to {n}")
        xq = np.vstack(xqx)

    if args.nt != -1:
        print("setting nb openmp threads to", args.nt)
        faiss.omp_set_num_threads(args.nt)

    print("loading index")

    if args.mmap:
        io_flag = faiss.IO_FLAG_READ_ONLY | faiss.IO_FLAG_MMAP
    else:
        io_flag = 0

    print(f"load index {args.indexname} {io_flag=:x}")
    index = faiss.read_index(args.indexname, io_flag)
    index_ivf = faiss.extract_index_ivf(index)

    print("prepare index")
    op = OperatingPointsWithRanges()
    op.add_range(
        "nprobe", [
            2 ** i for i in range(20)
            if 2 ** i < index_ivf.nlist * 0.1 and 2 ** i <= 4096
        ]
    )

    # prepare options for GPU clone

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = args.useFloat16
    co.useFloat16CoarseQuantizer = args.useFloat16CoarseQuantizer
    co.usePrecomputed = args.usePrecomputed
    co.shard_type = args.shard_type

    if args.search_type == "cpu":
        op.add_range(
            "quantizer_efSearch",
            [2 ** i for i in range(10)]
        )
    elif args.search_type == "gpu":
        print("move index to 1 GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        op.add_range(
            "quantizer_efSearch",
            [2 ** i for i in range(10)]
        )
        op.restrict_range("nprobe", 2049)
    elif args.search_type == "gpu_tiled":
        print("move index to 1 GPU")
        new_quantizer = faiss.IndexFlatL2(index_ivf.d)
        quantizer_hnsw = replace_ivf_quantizer(index_ivf, new_quantizer)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        op.add_range(
            "quantizer_efSearch",
            [2 ** i for i in range(10)]
        )
        op.restrict_range("nprobe", 2049)
        index = SeparateCoarseQuantizationIndex(
            quantizer_hnsw, index, bs=args.batch_size)
    elif args.search_type == "gpu_ivf_quantizer":
        index_ivf = faiss.extract_index_ivf(index)
        centroids = index_ivf.quantizer.reconstruct_n()
        replace_ivf_quantizer(index_ivf, faiss.IndexFlatL2(index_ivf.d))
        res = faiss.StandardGpuResources()
        new_quantizer = faiss.index_factory(
            index_ivf.d, f"IVF{args.ivf_quant_nlist},Flat")
        new_quantizer.train(centroids)
        new_quantizer.add(centroids)
        index = SeparateCoarseQuantizationIndex(
            faiss.index_cpu_to_gpu(res, 0, new_quantizer, co),
            faiss.index_cpu_to_gpu(res, 0, index, co),
            bs=args.batch_size, seq_tiling=True
        )
        op.add_range(
            "quantizer_nprobe",
            [2 ** i for i in range(9)]
        )
        op.restrict_range("nprobe", 1025)
    elif args.search_type == "gpu_flat_quantizer":
        index_ivf = faiss.extract_index_ivf(index)
        new_quantizer = faiss.IndexFlatL2(index_ivf.d)
        replace_ivf_quantizer(index_ivf, new_quantizer)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        op.restrict_range("nprobe", 2049)
    elif args.search_type == "cpu_flat_gpu_quantizer":
        index_ivf = faiss.extract_index_ivf(index)
        quantizer = faiss.IndexFlatL2(index_ivf.d)
        res = faiss.StandardGpuResources()
        quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer, co)
        index = SeparateCoarseQuantizationIndex(
            quantizer, index, bs=args.batch_size)
        op.restrict_range("nprobe", 2049)
    elif args.search_type in ("multi_gpu", "multi_gpu_sharded"):
        print(f"move index to {faiss.get_num_gpus()} GPU")
        co.shard = "sharded" in args.search_type
        index = faiss.index_cpu_to_all_gpus(index, co=co)
        op.add_range(
            "quantizer_efSearch",
            [2 ** i for i in range(10)]
        )
        op.restrict_range("nprobe", 2049)
    elif args.search_type in (
            "multi_gpu_flat_quantizer", "multi_gpu_flat_quantizer_sharded"):
        index_ivf = faiss.extract_index_ivf(index)
        new_quantizer = faiss.IndexFlatL2(ds.d)
        replace_ivf_quantizer(index_ivf, new_quantizer)
        index = faiss.index_cpu_to_all_gpus(index, co=co)
        op.restrict_range("nprobe", 2049)
    elif args.search_type in (
            "multi_gpu_sharded1", "multi_gpu_sharded1_flat",
            "multi_gpu_sharded1_ivf"):
        print(f"move index to {faiss.get_num_gpus()} GPU")
        new_quantizer = faiss.IndexFlatL2(index_ivf.d)
        hnsw_quantizer = replace_ivf_quantizer(index_ivf, new_quantizer)
        co.shard
        co.shard = True
        gpus = list(range(faiss.get_num_gpus()))
        res = [faiss.StandardGpuResources() for _ in gpus]
        index = faiss.index_cpu_to_gpu_multiple_py(res, index, co, gpus)
        op.restrict_range("nprobe", 2049)
        if args.search_type == "multi_gpu_sharded1":
            op.add_range(
                "quantizer_efSearch",
                [2 ** i for i in range(10)]
            )
            index = ShardedGPUIndex(hnsw_quantizer, index, bs=args.batch_size)
        elif args.search_type == "multi_gpu_sharded1_ivf":
            centroids = hnsw_quantizer.storage.reconstruct_n()
            quantizer = faiss.index_factory(
                centroids.shape[1], f"IVF{args.ivf_quant_nlist},Flat")
            quantizer.train(centroids)
            quantizer.add(centroids)
            co.shard = False
            quantizer = faiss.index_cpu_to_gpu_multiple_py(
                res, quantizer, co, gpus)
            index = ShardedGPUIndex(quantizer, index, bs=args.batch_size)

            op.add_range(
                "quantizer_nprobe",
                [2 ** i for i in range(9)]
            )
            op.restrict_range("nprobe", 1025)
        elif args.search_type == "multi_gpu_sharded1_flat":
            quantizer = hnsw_quantizer.storage
            quantizer = faiss.index_cpu_to_gpu_multiple_py(
                res, quantizer, co, gpus)
            index = ShardedGPUIndex(quantizer, index, bs=args.batch_size)
        else:
            raise RuntimeError()
    elif args.search_type in (
            "multi_gpu_Csharded1", "multi_gpu_Csharded1_flat",
            "multi_gpu_Csharded1_ivf"):
        print(f"move index to {faiss.get_num_gpus()} GPU")
        co.shard = True
        co.common_ivf_quantizer
        co.common_ivf_quantizer = True
        op.restrict_range("nprobe", 2049)
        if args.search_type == "multi_gpu_Csharded1":
            op.add_range(
                "quantizer_efSearch",
                [2 ** i for i in range(10)]
            )
            index = faiss.index_cpu_to_all_gpus(index, co)
        elif args.search_type == "multi_gpu_Csharded1_flat":
            new_quantizer = faiss.IndexFlatL2(index_ivf.d)
            quantizer_hnsw = replace_ivf_quantizer(index_ivf, new_quantizer)
            index = faiss.index_cpu_to_all_gpus(index, co)
        elif args.search_type == "multi_gpu_Csharded1_ivf":
            quantizer = faiss.index_factory(
                index_ivf.d, f"IVF{args.ivf_quant_nlist},Flat")
            quantizer_hnsw = replace_ivf_quantizer(index_ivf, quantizer)
            op.add_range(
                "quantizer_nprobe",
                [2 ** i for i in range(9)]
            )
            index = faiss.index_cpu_to_all_gpus(index, co)
        else:
            raise RuntimeError()
    else:
        raise RuntimeError()

    totex = op.num_experiments()
    experiments = op.sample_experiments()
    print(f"total nb experiments {totex}, running {len(experiments)}")

    print("perform search")
    gt = ds.get_groundtruth(100)

    # piggyback on operating points so that this gets stored in the stats file
    op.all_experiments = []
    op.platform = {
        "loadavg": open("/proc/loadavg", "r").readlines(),
        "procesor": [l for l in open("/proc/cpuinfo") if "model name" in l][0],
        "GPU": list(os.popen("nvidia-smi", "r")),
        "mem": open("/proc/meminfo", "r").readlines(),
        "pid": os.getpid()
    }
    op.args = args
    if args.stats:
        print(f"storing stats in {args.stats} after each experiment")

    for cno in experiments:
        key = op.cno_to_key(cno)
        parameters = op.get_parameters(key)
        print(f"{cno=:4d} {str(parameters):50}", end=": ", flush=True)

        (max_perf, min_time) = op.predict_bounds(key)
        if not op.is_pareto_optimal(max_perf, min_time):
            print(f"SKIP, {max_perf=:.3f} {min_time=:.3f}", )
            continue

        for name, val in parameters.items():
            set_index_parameter(index, name, val)

        if cno == 0:
            # warmup
            for _ in range(5):
                D, I = index.search(xq, 100)

        t0 = time.time()
        try:
            D, I = index.search(xq, 100)
        except RuntimeError as e:
            print(f"ERROR {e}")
            continue
        t1 = time.time()

        recalls = {}
        for rank in 1, 10, 100:
            recall = (gt[:, :1] == I[:ds.nq, :rank]).sum() / ds.nq
            recalls[rank] = recall

        print(f"time={t1 - t0:.3f} s recalls={recalls}")
        perf = recalls[1]
        op.add_operating_point(key, perf, t1 - t0)
        op.all_experiments.append({
            "cno": cno,
            "key": key,
            "parameters": parameters,
            "time": t1 - t0,
            "recalls": recalls
        })

        if args.stats:
            pickle.dump(op, open(args.stats, "wb"))


if __name__ == "__main__":
    main()
