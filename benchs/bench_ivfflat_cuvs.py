# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and its affiliates.
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


def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)


group = parser.add_argument_group('benchmarking options')

aa('--bm_train', default=False, action='store_true',
   help='whether to benchmark train operation on GPU index')
aa('--bm_add', default=False, action='store_true',
   help='whether to benchmark add operation on GPU index')
aa('--bm_search', default=True,
   help='whether to benchmark search operation on GPU index')
aa('--cuvs_only', default=False, action='store_true',
   help='whether to only produce cuVS enabled benchmarks')


group = parser.add_argument_group('IVF options')
aa('--n_centroids', default=256, type=int,
    help="number of IVF centroids")


group = parser.add_argument_group('searching')

aa('--k', default=100, type=int, help='nb of nearest neighbors')
aa('--nprobe', default=50, help='nb of IVF lists to probe')

args = parser.parse_args()

print("args:", args)

rs = np.random.RandomState(123)

res = faiss.StandardGpuResources()

# Use an RMM pool memory resource for device allocations
mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
rmm.mr.set_current_device_resource(mr)

def bench_train_milliseconds(index, trainVecs, use_cuvs):
    co = faiss.GpuMultipleClonerOptions()
    co.use_cuvs = use_cuvs
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
    t0 = time.time()
    index_gpu.train(trainVecs)
    return 1000*(time.time() - t0)


if args.bm_train:
    print("=" * 40)
    print("GPU Train Benchmarks")
    print("=" * 40)
    trainset_sizes = [5000, 10000, 100000, 1000000, 5000000]
    dataset_dims = [128, 256, 1024]
    for n_rows in trainset_sizes:
        for n_cols in dataset_dims:
            index = faiss.index_factory(n_cols, "IVF{},Flat".format(args.n_centroids))
            trainVecs = rs.rand(n_rows, n_cols).astype('float32')
            cuvs_gpu_train_time = bench_train_milliseconds(
                index, trainVecs, True)
            if args.cuvs_only:
                print("Method: IVFFlat, Operation: TRAIN, dim: %d, n_centroids %d, numTrain: %d, cuVS enabled GPU train time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, n_rows, cuvs_gpu_train_time))
            else:
                classical_gpu_train_time = bench_train_milliseconds(
                    index, trainVecs, False)
                print("Method: IVFFlat, Operation: TRAIN, dim: %d, n_centroids %d, numTrain: %d, classical GPU train time: %.3f milliseconds, cuVS enabled GPU train time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, n_rows, classical_gpu_train_time, cuvs_gpu_train_time))


def bench_add_milliseconds(index, addVecs, use_cuvs):
    co = faiss.GpuMultipleClonerOptions()
    co.use_cuvs = use_cuvs
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
    trainVecs = rs.rand(n_train, n_cols).astype('float32')
    index = faiss.index_factory(
        n_cols, "IVF" + str(args.n_centroids) + ",Flat")
    index.train(trainVecs)
    for n_rows in addset_sizes:
        for n_cols in dataset_dims:
            addVecs = rs.rand(n_rows, n_cols).astype('float32')
            cuvs_gpu_add_time = bench_add_milliseconds(index, addVecs, True)
            if args.cuvs_only:
                print("Method: IVFFlat, Operation: ADD, dim: %d, n_centroids %d, numAdd: %d, cuVS enabled GPU add time: %.3f milliseconds" % (
                    n_train, n_rows, n_cols, args.n_centroids, cuvs_gpu_add_time))
            else:
                classical_gpu_add_time = bench_add_milliseconds(
                    index, addVecs, False)
                print("Method: IVFFlat, Operation: ADD, dim: %d, n_centroids %d, numAdd: %d, classical GPU add time: %.3f milliseconds, cuVS enabled GPU add time: %.3f milliseconds" % (
                    n_train, n_rows, n_cols, args.n_centroids, classical_gpu_add_time, cuvs_gpu_add_time))


def bench_search_milliseconds(index, addVecs, queryVecs, nprobe, k, use_cuvs):
    co = faiss.GpuMultipleClonerOptions()
    co.use_cuvs = use_cuvs
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
    queryset_sizes = [5000, 10000, 100000, 500000]
    n_train = 10000
    n_add = 100000
    search_bm_dims = [8, 16, 32]
    for n_cols in search_bm_dims:
        index = faiss.index_factory(n_cols, "IVF{},Flat".format(args.n_centroids))
        trainVecs = rs.rand(n_train, n_cols).astype('float32')
        index.train(trainVecs)
        addVecs = rs.rand(n_add, n_cols).astype('float32')
        for n_rows in queryset_sizes:
            queryVecs = rs.rand(n_rows, n_cols).astype('float32')
            cuvs_gpu_search_time = bench_search_milliseconds(
                index, addVecs, queryVecs, args.nprobe, args.k, True)
            if args.cuvs_only:
                print("Method: IVFFlat, Operation: SEARCH, dim: %d, n_centroids: %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, cuVS enabled GPU search time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, n_add, n_rows, args.nprobe, args.k, cuvs_gpu_search_time))
            else:
                classical_gpu_search_time = bench_search_milliseconds(
                    index, addVecs, queryVecs, args.nprobe, args.k, False)
                print("Method: IVFFlat, Operation: SEARCH, dim: %d, n_centroids: %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, classical GPU search time: %.3f milliseconds, cuVS enabled GPU search time: %.3f milliseconds" % (
                    n_cols, args.n_centroids, n_add, n_rows, args.nprobe, args.k, classical_gpu_search_time, cuvs_gpu_search_time))

    print("=" * 40)
    print("Large cuVS Enabled Benchmarks")
    print("=" * 40)
    # Avoid classical GPU Benchmarks for large datasets because of OOM for more than 500000 queries and/or large dims as well as for large k
    queryset_sizes = [100000, 500000, 1000000]
    large_search_bm_dims = [128, 256, 1024]
    for n_cols in large_search_bm_dims:
        trainVecs = rs.rand(n_train, n_cols).astype('float32')
        index = faiss.index_factory(
            n_cols, "IVF" + str(args.n_centroids) + ",Flat")
        index.train(trainVecs)
        addVecs = rs.rand(n_add, n_cols).astype('float32')
        for n_rows in queryset_sizes:
            queryVecs = rs.rand(n_rows, n_cols).astype('float32')
            cuvs_gpu_search_time = bench_search_milliseconds(
                index, addVecs, queryVecs, args.nprobe, args.k, True)
            print("Method: IVFFlat, Operation: SEARCH, numTrain: %d, dim: %d, n_centroids: %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, cuVS enabled GPU search time: %.3f milliseconds" % (
                n_cols, args.n_centroids, n_add, n_rows, args.nprobe, args.k, cuvs_gpu_search_time))
