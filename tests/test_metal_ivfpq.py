# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the Metal IVF-PQ search backend.

The per-(query, probe) lookup-table path must add the coarse-quantizer score
for inner product: its tables hold <q_m, pq_m> only, while the CPU reference
adds dis0 = <q, centroid> (see QueryTables::precompute_list_tables_IP).
These tests force the lookup-table path with configurations the
precomputed-table path declines, and compare GPU results against the CPU
index.
"""

import functools
import unittest

import numpy as np

import faiss


_HAS_METAL = "MAC_METAL" in faiss.get_compile_options()
needs_metal_build = unittest.skipUnless(
    _HAS_METAL, "needs a faiss Metal build"
)


def needs_metal_device(test_func):
    """Skip the test when no Metal device is available."""

    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        if faiss.get_num_gpus() == 0:
            raise unittest.SkipTest("no Metal device available")
        return test_func(*args, **kwargs)

    return wrapper


@needs_metal_build
class TestMetalIVFPQ(unittest.TestCase):

    def check_matches_cpu(
        self, metric, d, M, nlist, nb, nq, nprobe, k
    ):
        rs = np.random.RandomState(1234)
        xb = rs.rand(nb, d).astype(np.float32)
        xq = rs.rand(nq, d).astype(np.float32)

        quantizer = (
            faiss.IndexFlatIP(d)
            if metric == faiss.METRIC_INNER_PRODUCT
            else faiss.IndexFlatL2(d)
        )
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8, metric)
        cpu_index.nprobe = nprobe
        cpu_index.train(xb)
        cpu_index.add(xb)
        ref_D, ref_I = cpu_index.search(xq, k)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        got_D, got_I = gpu_index.search(xq, k)

        for q in range(nq):
            valid = ref_I[q] != -1
            self.assertTrue(np.any(valid))
            self.assertEqual(set(got_I[q][valid]), set(ref_I[q][valid]))
            np.testing.assert_allclose(
                got_D[q][valid], ref_D[q][valid], atol=1e-2
            )

    @needs_metal_device
    def test_lookup_table_path_ip_large_k(self):
        # k > 512 declines the precomputed-table path. With nprobe == 1 the
        # missing coarse term keeps the ranking but shifts every score.
        self.check_matches_cpu(
            faiss.METRIC_INNER_PRODUCT,
            d=64,
            M=8,
            nlist=8,
            nb=500,
            nq=5,
            nprobe=1,
            k=600,
        )

    @needs_metal_device
    def test_lookup_table_path_l2_large_k(self):
        # L2 tables hold the complete residual distance, so the coarse term
        # must stay off: guard the flag plumbing in both directions.
        self.check_matches_cpu(
            faiss.METRIC_L2,
            d=64,
            M=8,
            nlist=8,
            nb=500,
            nq=5,
            nprobe=1,
            k=600,
        )
