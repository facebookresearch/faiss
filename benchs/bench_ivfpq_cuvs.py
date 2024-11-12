# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

######################################################
# Command-line parsing
######################################################

parser = argparse.ArgumentParser()

from datasets import load_sift1M, evaluate


print("load data")
xb, xq, xt, gt = load_sift1M()

def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)


group = parser.add_argument_group('benchmarking options')
aa('--cuvs_only', default=False, action='store_true',
   help='whether to only produce cuVS enabled benchmarks')

group = parser.add_argument_group('IVF options')
aa('--bits_per_code', default=8, type=int, help='bits per code. Note that < 8 is only supported when cuVS is enabled')
aa('--pq_len', default=2, type=int, help='number of vector elements represented by one PQ code')
aa('--use_precomputed', default=True, type=bool, help='use precomputed codes (not with cuVS enabled)')

group = parser.add_argument_group('searching')
aa('--k', default=10, type=int, help='nb of nearest neighbors')
aa('--nprobe', default=50, type=int, help='nb of IVF lists to probe')

args = parser.parse_args()

print("args:", args)

rs = np.random.RandomState(123)

res = faiss.StandardGpuResources()

# Use an RMM pool memory resource for device allocations
mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
rmm.mr.set_current_device_resource(mr)

# A heuristic to select a suitable number of lists
def compute_nlist(numVecs):
    nlist = np.sqrt(numVecs)
    if (numVecs / nlist < 1000):
        nlist = numVecs / 1000
    return int(nlist)


def bench_train_milliseconds(index, trainVecs, use_cuvs):
    co = faiss.GpuMultipleClonerOptions()
    # use float 16 lookup tables to save space
    co.useFloat16LookupTables = True
    co.use_cuvs = use_cuvs
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
    t0 = time.time()
    index_gpu.train(trainVecs)
    return 1000*(time.time() - t0)

n_rows, n_cols = xb.shape
n_train, _ = xt.shape
M = n_cols // args.pq_len
nlist = compute_nlist(n_rows)
index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(nlist, M, args.bits_per_code))

print("=" * 40)
print("GPU Train Benchmarks")
print("=" * 40)
cuvs_gpu_train_time = bench_train_milliseconds(index, xt, True)
if args.cuvs_only:
    print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, cuVS enabled GPU train time: %.3f milliseconds" % (
        n_cols, nlist, M, args.bits_per_code, n_train, cuvs_gpu_train_time))
else:
    classical_gpu_train_time = bench_train_milliseconds(
        index, xt, False)
    print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, classical GPU train time: %.3f milliseconds, cuVS enabled GPU train time: %.3f milliseconds" % (
        n_cols, nlist, M, args.bits_per_code, n_train, classical_gpu_train_time, cuvs_gpu_train_time))


def bench_add_milliseconds(index, addVecs, use_cuvs):
    co = faiss.GpuMultipleClonerOptions()
    # use float 16 lookup tables to save space
    co.useFloat16LookupTables = True
    co.use_cuvs = use_cuvs
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
    index_gpu.copyFrom(index)
    t0 = time.time()
    index_gpu.add(addVecs)
    return 1000*(time.time() - t0)

print("=" * 40)
print("GPU Add Benchmarks")
print("=" * 40)
index.train(xt)
cuvs_gpu_add_time = bench_add_milliseconds(index, xb, True)
if args.cuvs_only:
    print("Method: IVFPQ, Operation: ADD, dim: %d, n_centroids %d numSubQuantizers %d, bitsPerCode %d, numAdd %d, cuVS enabled GPU add time: %.3f milliseconds" % (
        n_cols, nlist, M, args.bits_per_code, n_rows, cuvs_gpu_add_time))
else:
    classical_gpu_add_time = bench_add_milliseconds(
        index, xb, False)
    print("Method: IVFFPQ, Operation: ADD, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numAdd %d, classical GPU add time: %.3f milliseconds, cuVS enabled GPU add time: %.3f milliseconds" % (
        n_cols, nlist, M, args.bits_per_code, n_rows, classical_gpu_add_time, cuvs_gpu_add_time))


def bench_search_milliseconds(index, addVecs, queryVecs, nprobe, k, use_cuvs):
    co = faiss.GpuMultipleClonerOptions()
    co.use_cuvs = use_cuvs
    co.useFloat16LookupTables = True
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
    index_gpu.copyFrom(index)
    index_gpu.add(addVecs)
    index_gpu.nprobe = nprobe
    t0 = time.time()
    index_gpu.search(queryVecs, k)
    return 1000*(time.time() - t0)


if args.bm_search:
    print("=" * 40)
    print("GPU Search Benchmarks")
    print("=" * 40)
    queryset_sizes = [1, 10, 100, 1000, 10000]
    n_train, n_cols = xt.shape
    n_add, _ = xb.shape
    print(xq.shape)
    M = n_cols // args.pq_len
    nlist = compute_nlist(n_add)
    index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(nlist, M, args.bits_per_code))
    index.train(xt)
    for n_rows in queryset_sizes:
        queryVecs = xq[np.random.choice(xq.shape[0], n_rows, replace=False)]
        cuvs_gpu_search_time = bench_search_milliseconds(
            index, xb, queryVecs, args.nprobe, args.k, True)
        if args.cuvs_only:
            print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, cuVS enabled GPU search time: %.3f milliseconds" % (
                n_cols, nlist, M, args.bits_per_code, n_add, n_rows, args.nprobe, args.k, cuvs_gpu_search_time))
        else:
            classical_gpu_search_time = bench_search_milliseconds(
                index, xb, queryVecs, args.nprobe, args.k, False)
            print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, classical GPU search time: %.3f milliseconds, cuVS enabled GPU search time: %.3f milliseconds" % (
                n_cols, nlist, M, args.bits_per_code, n_add, n_rows, args.nprobe, args.k, classical_gpu_search_time, cuvs_gpu_search_time))