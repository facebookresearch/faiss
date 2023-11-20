# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     h ttp://www.apache.org/licenses/LICENSE-2.0
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


def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)


group = parser.add_argument_group('benchmarking options')

aa('--bm_train', default=False, action='store_true',
   help='whether to benchmark train operation on GPU index')
aa('--bm_add', default=False, action='store_true',
   help='whether to benchmark add operation on GPU index')
aa('--bm_search', default=True,
   help='whether to benchmark search operation on GPU index')
aa('--raft_only', default=False, action='store_true',
   help='whether to only produce RAFT enabled benchmarks')


group = parser.add_argument_group('IVF options')
aa('--n_centroids', default=256, type=int,
    help='number of IVF centroids')
aa('--bits_per_code', default=8, type=int, help='bits per code. Note that < 8 is not supported for classical GPU')
aa('--pq_len', default=16, type=int, help='number of vector elements represented by one PQ code')
aa('--use_precomputed', default=True, type=bool, help='use precomputed codes')

group = parser.add_argument_group('searching')

aa('--k', default=50, type=int, help='nb of nearest neighbors')
aa('--nprobe', default=50, help='nb of IVF lists to probe')

args = parser.parse_args()

print("args:", args)

rs = np.random.RandomState(123)

res = faiss.StandardGpuResources()

# set the RMM pool memory resource for faster allocations
mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=2**30, maximum_pool_size=2**31)
rmm.mr.set_current_device_resource(mr)

def bench_train_milliseconds(index, trainVecs, use_raft):
    co = faiss.GpuMultipleClonerOptions()
    # use float 16 lookup tables to save space
    co.useFloat16LookupTables = True
    co.use_raft = use_raft
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
    t0 = time.time()
    index_gpu.train(trainVecs)
    return 1000*(time.time() - t0)


if args.bm_train:
    print("=" * 40)
    print("GPU Train Benchmarks")
    print("=" * 40)
    trainset_sizes = [10000, 100000, 1000000, 5000000]
    dataset_dims = [128, 256, 1024]
    for n_rows in trainset_sizes:
        for n_cols in dataset_dims:
            M = n_cols // args.pq_len
            index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(args.n_centroids, M, args.bits_per_code))
            trainVecs = rs.rand(n_rows, n_cols).astype('float32')
            raft_gpu_train_time = bench_train_milliseconds(
                index, trainVecs, True)
            if args.raft_only:
                print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, RAFT enabled GPU train time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, M, args.bits_per_code, n_rows, raft_gpu_train_time))
            else:
                classical_gpu_train_time = bench_train_milliseconds(
                    index, trainVecs, False)
                print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, classical GPU train time: %.3f milliseconds, RAFT enabled GPU train time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, M, args.bits_per_code, n_rows, classical_gpu_train_time, raft_gpu_train_time))


def bench_add_milliseconds(index, addVecs, use_raft):
    co = faiss.GpuMultipleClonerOptions()
    # use float 16 lookup tables to save space
    co.useFloat16LookupTables = True
    co.use_raft = use_raft
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
    index_gpu.copyFrom(index)
    t0 = time.time()
    index_gpu.add(addVecs)
    return 1000*(time.time() - t0)


if args.bm_add:
    print("=" * 40)
    print("GPU Add Benchmarks")
    print("=" * 40)
    addset_sizes = [5000, 10000, 100000, 1000000]
    dataset_dims = [128, 256, 1024]
    n_train = 10000
    for n_cols in dataset_dims:
        M = n_cols // args.pq_len
        trainVecs = rs.rand(n_train, n_cols).astype('float32')
        index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(args.n_centroids, M, args.bits_per_code))
        index.train(trainVecs)
        for n_rows in addset_sizes:
            addVecs = rs.rand(n_rows, n_cols).astype('float32')
            raft_gpu_add_time = bench_add_milliseconds(index, addVecs, True)
            if args.raft_only:
                print("Method: IVFPQ, Operation: ADD, dim: %d, n_centroids %d numSubQuantizers %d, bitsPerCode %d, numAdd %d, RAFT enabled GPU add time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, M, args.bits_per_code, n_rows, raft_gpu_add_time))
            else:
                classical_gpu_add_time = bench_add_milliseconds(
                    index, addVecs, False)
                print("Method: IVFFPQ, Operation: ADD, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numAdd %d, classical GPU add time: %.3f milliseconds, RAFT enabled GPU add time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, M, args.bits_per_code, n_rows, classical_gpu_add_time, raft_gpu_add_time))


def bench_search_milliseconds(index, addVecs, queryVecs, nprobe, k, use_raft):
    co = faiss.GpuMultipleClonerOptions()
    co.use_raft = use_raft
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
    queryset_sizes = [5000, 10000, 100000]
    n_train = 10000
    n_add = 100000
    search_bm_dims = [128, 256, 512]
    for n_cols in search_bm_dims:
        M = n_cols // args.pq_len
        trainVecs = rs.rand(n_train, n_cols).astype('float32')
        index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(args.n_centroids, M, args.bits_per_code))
        index.train(trainVecs)
        addVecs = rs.rand(n_add, n_cols).astype('float32')
        for n_rows in queryset_sizes:
            queryVecs = rs.rand(n_rows, n_cols).astype('float32')
            raft_gpu_search_time = bench_search_milliseconds(
                index, addVecs, queryVecs, args.nprobe, args.k, True)
            if args.raft_only:
                print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, RAFT enabled GPU search time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, M, args.bits_per_code, n_add, n_rows, args.nprobe, args.k, raft_gpu_search_time))
            else:
                classical_gpu_search_time = bench_search_milliseconds(
                    index, addVecs, queryVecs, args.nprobe, args.k, False)
                print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, classical GPU search time: %.3f milliseconds, RAFT enabled GPU search time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, M, args.bits_per_code, n_add, n_rows, args.nprobe, args.k, classical_gpu_search_time, raft_gpu_search_time))

    print("=" * 40)
    print("Large RAFT Enabled Benchmarks")
    print("=" * 40)
    # More RAFT enabled benchmarks for
    # 1. larger number of queries
    # 2. More number of subquantizers
    # 3. Large k
    # NB: classical FAISS GPU runs out of GPU memory quickly and does not scale for these cases
    queryset_sizes = [500000, 1000000, 10000000]
    n_train = 10000
    n_add = 100000
    large_search_bm_dims = [128, 256, 1024]
    pq_len = 8
    k = 100
    for n_cols in large_search_bm_dims:
        M = n_cols // pq_len
        trainVecs = rs.rand(n_train, n_cols).astype('float32')
        index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(args.n_centroids, M, args.bits_per_code))
        index.train(trainVecs)
        addVecs = rs.rand(n_add, n_cols).astype('float32')
        for n_rows in queryset_sizes:
            queryVecs = rs.rand(n_rows, n_cols).astype('float32')
            raft_gpu_search_time = bench_search_milliseconds(
                index, addVecs, queryVecs, args.nprobe, k, True)
            print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, RAFT enabled GPU search time: %.3f milliseconds" % (
                n_cols, args.n_centroids, M, args.bits_per_code, n_add, n_rows, args.nprobe, k, raft_gpu_search_time))
