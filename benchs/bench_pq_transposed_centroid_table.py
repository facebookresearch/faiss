#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import time
import random

import faiss.contrib.datasets


# copied from benchs/bench_all_ivf/bench_all_ivf.py
def unwind_index_ivf(index):
    if isinstance(index, faiss.IndexPreTransform):
        assert index.chain.size() == 1
        vt = index.chain.at(0)
        index_ivf, vt2 = unwind_index_ivf(faiss.downcast_index(index.index))
        assert vt2 is None
        return index_ivf, vt
    if hasattr(faiss, "IndexRefine") and isinstance(index, faiss.IndexRefine):
        return unwind_index_ivf(faiss.downcast_index(index.base_index))
    if isinstance(index, faiss.IndexIVF):
        return index, None
    else:
        return None, None


def test_bigann10m(index_file, index_parameters):
    ds = faiss.contrib.datasets.DatasetBigANN(nb_M=10)

    xq = ds.get_queries()
    xb = ds.get_database()
    gt = ds.get_groundtruth()

    nb, d = xb.shape
    nq, d = xq.shape

    print("Reading index {}".format(index_file))
    index = faiss.read_index(index_file)

    ps = faiss.ParameterSpace()
    ps.initialize(index)

    index_ivf, vec_transform = unwind_index_ivf(index)

    print('params                                                                      regular    transp_centroids   regular   R@1    R@10   R@100')
    for index_parameter in index_parameters:
        ps.set_index_parameters(index, index_parameter)

        print(index_parameter.ljust(70), end=' ')

        k = 100

        # warmup
        D, I = index.search(xq, k)

        # warmup
        D, I = index.search(xq, k)

        # eval
        t2_0 = time.time()
        D, I = index.search(xq, k)
        t2_1 = time.time()

        # eval
        index_ivf.pq.sync_transposed_centroids()
        t3_0 = time.time()
        D, I = index.search(xq, k)
        t3_1 = time.time()

        # eval
        index_ivf.pq.clear_transposed_centroids()
        t4_0 = time.time()
        D, I = index.search(xq, k)
        t4_1 = time.time()

        print("   %9.5f  " % (t2_1 - t2_0), end=' ')
        print("   %9.5f  " % (t3_1 - t3_0), end=' ')
        print("   %9.5f  " % (t4_1 - t4_0), end=' ')

        for rank in 1, 10, 100:
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            print("%.4f" % (n_ok / float(nq)), end=' ')
        print()


if __name__ == "__main__":
    faiss.contrib.datasets.dataset_basedir = '/home/aguzhva/ANN_SIFT1B/'

    # represents OPQ32_128,IVF65536_HNSW32,PQ32 index
    index_file_1 = "/home/aguzhva/ANN_SIFT1B/run_tests/bench_ivf/indexes/hnsw32/.faissindex"

    nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    quantizer_efsearch_values = [4, 8, 16, 32, 64, 128, 256, 512]
    ht_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 256]

    # represents OPQ32_128,IVF65536(IVF256,PQHDx4fs,RFlat),PQ32 index
    index_file_2 = "/home/aguzhva/ANN_SIFT1B/run_tests/bench_ivf/indexes/pq4/.faissindex"

    quantizer_k_factor_rf_values = [1, 2, 4, 8, 16, 32, 64]
    quantizer_nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128]

    # test the first index
    index_parameters_1 = []
    for _ in range(0, 20):
        nprobe = random.choice(nprobe_values)
        quantizer_efsearch = random.choice(quantizer_efsearch_values)
        ht = random.choice(ht_values)
        index_parameters_1.append(
            "nprobe={},quantizer_efSearch={},ht={}".format(
                nprobe,
                quantizer_efsearch,
                ht)
        )

    test_bigann10m(index_file_1, index_parameters_1)

    # test the second index
    index_parameters_2 = []
    for _ in range(0, 20):
        nprobe = random.choice(nprobe_values)
        quantizer_k_factor_rf = random.choice(quantizer_k_factor_rf_values)
        quantizer_nprobe = random.choice(quantizer_nprobe_values)
        ht = random.choice(ht_values)
        index_parameters_2.append(
            "nprobe={},quantizer_k_factor_rf={},quantizer_nprobe={},ht={}".format(
                nprobe,
                quantizer_k_factor_rf,
                quantizer_nprobe,
                ht)
        )

    test_bigann10m(index_file_2, index_parameters_2)
