# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np

from faiss.contrib import datasets


def _recall(Iref, Inew, k):
    return np.mean([
        len(set(Iref[i]) & set(Inew[i])) / k for i in range(Iref.shape[0])
    ])


@unittest.skipIf(faiss.get_num_gpus() == 0, "no GPU available")
class TestGpuIndexHNSW(unittest.TestCase):
    """Exercise the SWIG-exposed standard search() path of GpuIndexHNSW.

    Builds a CPU faiss::IndexHNSW via index_factory, moves it to the GPU with
    index_cpu_to_gpu (the GpuCloner routes IndexHNSWFlat/IndexHNSWSQ to
    GpuIndexHNSW), and compares GPU search() against the same CPU HNSW graph.
    Both traverse the identical graph, so GPU<->CPU recall must be high.

    This covers: the SWIG bindings, SearchParametersGpuHNSW, index_factory
    string coverage for the GPU path, and the on-device label-conversion
    kernel in searchImpl_ (uint64 neighbor ids -> idx_t, UINT64_MAX -> -1).
    """

    def do_parity(self, factory_str, metric, dim=64, ef=128, k=10):
        ds = datasets.SyntheticDataset(dim, 0, 4000, 200)
        xb = ds.get_database()
        xq = ds.get_queries()
        if metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(xb)
            faiss.normalize_L2(xq)

        cpu_index = faiss.index_factory(dim, factory_str, metric)
        cpu_index.add(xb)
        cpu_index.hnsw.efSearch = ef
        Dref, Iref = cpu_index.search(xq, k)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.assertIsInstance(gpu_index, faiss.GpuIndexHNSW)

        params = faiss.SearchParametersGpuHNSW()
        params.ef = ef
        Dnew, Inew = gpu_index.search(xq, k, params=params)

        # Shape / sentinel sanity: the label kernel must fill every slot and
        # only emit -1 (empty) or valid in-range ids.
        self.assertEqual(Inew.shape, (xq.shape[0], k))
        valid = Inew[Inew != -1]
        self.assertTrue(np.all(valid >= 0))
        self.assertTrue(np.all(valid < ds.nb))

        recall = _recall(Iref, Inew, k)
        self.assertGreater(
            recall, 0.90,
            f"GPU<->CPU HNSW recall@{k} too low for {factory_str} "
            f"metric={metric}: {recall:.4f}")

    # Flat storage (fp32)
    def test_flat_l2(self):
        self.do_parity("HNSW32,Flat", faiss.METRIC_L2)

    def test_flat_ip(self):
        self.do_parity("HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)

    # Scalar-quantizer storage
    def test_sq8_l2(self):
        self.do_parity("HNSW32,SQ8", faiss.METRIC_L2)

    def test_sqfp16_l2(self):
        self.do_parity("HNSW32,SQfp16", faiss.METRIC_L2)

    def test_sqfp16_ip(self):
        self.do_parity("HNSW32,SQfp16", faiss.METRIC_INNER_PRODUCT)

    def test_search_params_binding(self):
        """SearchParametersGpuHNSW fields are settable from Python."""
        params = faiss.SearchParametersGpuHNSW()
        self.assertEqual(params.ef, 200)  # documented default
        params.ef = 321
        params.search_width = 8
        self.assertEqual(params.ef, 321)
        self.assertEqual(params.search_width, 8)


if __name__ == "__main__":
    unittest.main()
