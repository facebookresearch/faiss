# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import faiss
import time
import argparse
import rmm
import ctypes

try:
    from faiss.contrib.datasets_fb import \
        DatasetSIFT1M, DatasetDeep1B, DatasetBigANN
except ImportError:
    from faiss.contrib.datasets import \
        DatasetSIFT1M, DatasetDeep1B, DatasetBigANN


# ds = DatasetDeep1B(10**6)
# ds = DatasetBigANN(nb_M=1)
ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()

xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

M = d / 2

######################################################
# Command-line parsing
######################################################

parser = argparse.ArgumentParser()


def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)


group = parser.add_argument_group('benchmarking options')

aa('--bm_train', default=True,
   help='whether to benchmark train operation on GPU index')
aa('--bm_add', default=True,
   help='whether to benchmark add operation on GPU index')
aa('--bm_search', default=True,
   help='whether to benchmark search operation on GPU index')


group = parser.add_argument_group('IVF options')
aa('--nlist', default=1024, type=np.int64,
    help="number of IVF centroids")
aa('--bits_per_code', default=8, type=np.int64, help='bits per code. Note that < 8 is only supported when cuVS is enabled')


group = parser.add_argument_group('searching')

aa('--k', default=10, type=int, help='nb of nearest neighbors')
aa('--nprobe', default=10, help='nb of IVF lists to probe')

args = parser.parse_args()

print("args:", args)

gt = gt[:, :args.k]
nlist = args.nlist
bits_per_code = args.bits_per_code

rs = np.random.RandomState(123)

res = faiss.StandardGpuResources()

# Use an RMM pool memory resource for device allocations
mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
rmm.mr.set_current_device_resource(mr)


def eval_recall(neighbors, t):
    speed = t * 1000 / nq
    qps = 1000 / speed

    corrects = (gt == neighbors).sum()
    recall = corrects / (nq * args.k)

    return recall, qps


def bench_train_milliseconds(trainVecs, use_cuvs):
    config = faiss.GpuIndexIVFPQConfig()
    config.use_cuvs = use_cuvs
    index = faiss.GpuIndexIVFPQ(res, d, 1024, 32, 8, faiss.METRIC_L2, config)
    t0 = time.time()
    index.train(trainVecs)
    return 1000*(time.time() - t0)


#warmup
xw = rs.rand(nt, d)
bench_train_milliseconds(xw, True)


if args.bm_train:
    print("=" * 40)
    print("GPU Train Benchmarks")
    print("=" * 40)

    cuvs_gpu_train_time = bench_train_milliseconds(xt, True)
    classical_gpu_train_time = bench_train_milliseconds(xt, False)
    print("TRAIN, dim: %d, nlist %d, numTrain: %d, classical GPU train time: %.3f milliseconds, cuVS enabled GPU train time: %.3f milliseconds" % (
        d, nlist, nt, classical_gpu_train_time, cuvs_gpu_train_time))


def bench_add_milliseconds(addVecs, index_cpu, use_cuvs):
    # construct a GPU index using the same trained coarse quantizer
    config = faiss.GpuClonerOptions()
    config.use_cuvs = use_cuvs
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, config)
    assert(index_gpu.is_trained)
    t0 = time.time()
    index_gpu.add(addVecs)
    return 1000*(time.time() - t0)


if args.bm_add:
    print("=" * 40)
    print("GPU Add Benchmarks")
    print("=" * 40)
    quantizer = faiss.IndexFlatL2(d)
    index_cpu = faiss.IndexIVFPQ(quantizer, d, 1024, 32, 8, faiss.METRIC_L2)
    index_cpu.train(xt)
    cuvs_gpu_add_time = bench_add_milliseconds(xb, index_cpu, True)
    classical_gpu_add_time = bench_add_milliseconds(xb, index_cpu, False)
    print("ADD, dim: %d, nlist %d, numAdd: %d, classical GPU add time: %.3f milliseconds, cuVS enabled GPU add time: %.3f milliseconds" % (
        d, nlist, nb, classical_gpu_add_time, cuvs_gpu_add_time))


def bench_search_milliseconds(index, queryVecs, nprobe, k, use_cuvs):
    co = faiss.GpuClonerOptions()
    co.use_cuvs = use_cuvs
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
    index_gpu.nprobe = nprobe
    t0 = time.time()
    _, I = index_gpu.search(queryVecs, k)
    return I, 1000*(time.time() - t0)


# Search benchmarks: both indexes have identical IVF centroids and lists. 
if args.bm_search:
    print("=" * 40)
    print("GPU Search Benchmarks")
    print("=" * 40)
    index_cpu = faiss.IndexIVFPQ(quantizer, d, 1024, 32, 8, faiss.METRIC_L2)
    index_cpu.train(xt)
    index_cpu.add(xb)

    cuvs_indices, cuvs_gpu_search_time = bench_search_milliseconds(
        index_cpu, xq, args.nprobe, args.k, True)
    classical_gpu_indices, classical_gpu_search_time = bench_search_milliseconds(
        index_cpu, xq, args.nprobe, args.k, False)
    cuvs_recall, cuvs_qps = eval_recall(cuvs_indices, cuvs_gpu_search_time)
    classical_recall, classical_qps = eval_recall(classical_gpu_indices, classical_gpu_search_time)
    print("SEARCH, dim: %d, nlist: %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, classical GPU qps: %.3f, cuVS enabled GPU qps: %.3f"  % (
        d, nlist, nb, nq, args.nprobe, args.k, classical_qps, cuvs_qps))
