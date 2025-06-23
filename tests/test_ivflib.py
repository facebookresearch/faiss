# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import unittest
import faiss
import numpy as np
import os
import random


class TestIVFlib(unittest.TestCase):

    def test_methods_exported(self):
        methods = ['check_compatible_for_merge', 'extract_index_ivf',
                   'merge_into', 'search_centroid',
                   'search_and_return_centroids', 'get_invlist_range',
                   'set_invlist_range', 'search_with_parameters']

        for method in methods:
            assert callable(getattr(faiss, method, None))


def search_single_scan(index, xq, k, bs=128):
    """performs a search so that the inverted lists are accessed
    sequentially by blocks of size bs"""

    # handle pretransform
    if isinstance(index, faiss.IndexPreTransform):
        xq = index.apply_py(xq)
        index = faiss.downcast_index(index.index)

    # coarse assignment
    nprobe = min(index.nprobe, index.nlist)
    coarse_dis, assign = index.quantizer.search(xq, nprobe)
    nlist = index.nlist
    assign_buckets = assign // bs
    nq = len(xq)

    rh = faiss.ResultHeap(nq, k)
    index.parallel_mode |= index.PARALLEL_MODE_NO_HEAP_INIT

    for l0 in range(0, nlist, bs):
        bucket_no = l0 // bs
        skip_rows, skip_cols = np.where(assign_buckets != bucket_no)
        sub_assign = assign.copy()
        sub_assign[skip_rows, skip_cols] = -1

        index.search_preassigned(
            xq, k, sub_assign, coarse_dis,
            D=rh.D, I=rh.I
        )

    rh.finalize()

    return rh.D, rh.I


class TestSequentialScan(unittest.TestCase):

    def test_sequential_scan(self):
        d = 20
        index = faiss.index_factory(d, 'IVF100,SQ8')

        rs = np.random.RandomState(123)
        xt = rs.rand(5000, d).astype('float32')
        xb = rs.rand(10000, d).astype('float32')
        index.train(xt)
        index.add(xb)
        k = 15
        xq = rs.rand(200, d).astype('float32')

        ref_D, ref_I = index.search(xq, k)
        D, I = search_single_scan(index, xq, k, bs=10)

        assert np.all(D == ref_D)
        assert np.all(I == ref_I)


class TestSearchWithParameters(unittest.TestCase):

    def test_search_with_parameters(self):
        d = 20
        index = faiss.index_factory(d, 'IVF100,SQ8')

        rs = np.random.RandomState(123)
        xt = rs.rand(5000, d).astype('float32')
        xb = rs.rand(10000, d).astype('float32')
        index.train(xt)
        index.nprobe = 3
        index.add(xb)
        k = 15
        xq = rs.rand(200, d).astype('float32')

        stats = faiss.cvar.indexIVF_stats
        stats.reset()
        Dref, Iref = index.search(xq, k)
        ref_ndis = stats.ndis

        # make sure the nprobe used is the one from params not the one
        # set in the index
        index.nprobe = 1
        params = faiss.IVFSearchParameters()
        params.nprobe = 3

        Dnew, Inew, stats2 = faiss.search_with_parameters(
            index, xq, k, params, output_stats=True)

        np.testing.assert_array_equal(Inew, Iref)
        np.testing.assert_array_equal(Dnew, Dref)

        self.assertEqual(stats2["ndis"], ref_ndis)

    def test_range_search_with_parameters(self):
        d = 20
        index = faiss.index_factory(d, 'IVF100,SQ8')

        rs = np.random.RandomState(123)
        xt = rs.rand(5000, d).astype('float32')
        xb = rs.rand(10000, d).astype('float32')
        index.train(xt)
        index.nprobe = 3
        index.add(xb)
        xq = rs.rand(200, d).astype('float32')

        Dpre, _ = index.search(xq, 15)
        radius = float(np.median(Dpre[:, -1]))
        stats = faiss.cvar.indexIVF_stats
        stats.reset()
        Lref, Dref, Iref = index.range_search(xq, radius)
        ref_ndis = stats.ndis

        # make sure the nprobe used is the one from params not the one
        # set in the index
        index.nprobe = 1
        params = faiss.IVFSearchParameters()
        params.nprobe = 3

        Lnew, Dnew, Inew, stats2 = faiss.range_search_with_parameters(
            index, xq, radius, params, output_stats=True)

        np.testing.assert_array_equal(Lnew, Lref)
        np.testing.assert_array_equal(Inew, Iref)
        np.testing.assert_array_equal(Dnew, Dref)

        self.assertEqual(stats2["ndis"], ref_ndis)


class TestSmallData(unittest.TestCase):
    """Test in case of nprobe > nlist."""

    def test_small_data(self):
        d = 20
        # nlist = (2^4)^2 = 256
        index = faiss.index_factory(d, 'IMI2x4,Flat')

        # When nprobe >= nlist, it is equivalent to an IndexFlat.
        rs = np.random.RandomState(123)
        xt = rs.rand(100, d).astype('float32')
        xb = rs.rand(1000, d).astype('float32')

        index.train(xt)
        index.add(xb)
        index.nprobe = 2048
        k = 5
        xq = rs.rand(10, d).astype('float32')

        # test kNN search
        D, I = index.search(xq, k)
        ref_D, ref_I = faiss.knn(xq, xb, k)
        assert np.all(D == ref_D)
        assert np.all(I == ref_I)

        # test range search
        thresh = 0.1   # *squared* distance
        lims, D, I = index.range_search(xq, thresh)
        ref_index = faiss.IndexFlat(d)
        ref_index.add(xb)
        ref_lims, ref_D, ref_I = ref_index.range_search(xq, thresh)
        assert np.all(lims == ref_lims)
        assert np.all(D == ref_D)
        assert np.all(I == ref_I)


class TestIvfSharding(unittest.TestCase):
    d = 32
    nlist = 100
    nb = 1000

    def custom_sharding_function(self, i, _):
        return 1 if i % 2 == 0 else 7

    # Mimics the default in DefaultShardingFunction.
    # This impl is just used for verification.
    def default_sharding_function(self, i, shard_count):
        return i % shard_count

    def verify_sharded_ivf_indexes(
            self, template, xb, shard_count, sharding_function, generate_ids=True):
        sharded_indexes_counters = [0] * shard_count
        sharded_indexes = []
        for i in range(shard_count):
            if xb[0].dtype.name == 'uint8':
                index = faiss.read_index_binary(template % i)
            else:
                index = faiss.read_index(template % i)
            sharded_indexes.append(index)

        # Reconstruct and verify each centroid
        if generate_ids:
            for i in range(len(xb)):
                shard_id = sharding_function(i, shard_count)
                reconstructed = sharded_indexes[shard_id].quantizer.reconstruct(i)
                np.testing.assert_array_equal(reconstructed, xb[i])
        else:
            for i in range(len(xb)):
                shard_id = sharding_function(i, shard_count)
                reconstructed = sharded_indexes[shard_id].quantizer.reconstruct(
                    sharded_indexes_counters[shard_id])
                sharded_indexes_counters[shard_id] += 1
                np.testing.assert_array_equal(reconstructed, xb[i])

        # Clean up
        for i in range(shard_count):
            os.remove(template % i)

    def test_save_index_shards_by_centroids_no_op(self):
        quantizer = faiss.IndexFlatL2(self.d)
        index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        with self.assertRaises(RuntimeError):
            faiss.shard_ivf_index_centroids(
                index,
                10,
                "shard.%d.index",
                None
            )

    def test_save_index_shards_by_centroids_flat_quantizer_default_sharding(
            self):
        xb = np.random.rand(self.nb, self.d).astype('float32')
        quantizer = faiss.IndexFlatL2(self.d)
        index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        shard_count = 3

        index.quantizer.add(xb)

        template = str(random.randint(0, 100000)) + "shard.%d.index"
        faiss.shard_ivf_index_centroids(
            index,
            shard_count,
            template,
            None,
            True
        )
        self.verify_sharded_ivf_indexes(
            template, xb, shard_count, self.default_sharding_function)

    def test_save_index_shards_by_centroids_flat_quantizer_custom_sharding(
            self):
        xb = np.random.rand(self.nb, self.d).astype('float32')
        quantizer = faiss.IndexFlatL2(self.d)
        index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        shard_count = 20

        index.quantizer.add(xb)

        template = str(random.randint(0, 100000)) + "shard.%d.index"
        faiss.shard_ivf_index_centroids(
            index,
            shard_count,
            template,
            self.custom_sharding_function,
            True
        )
        self.verify_sharded_ivf_indexes(
            template, xb, shard_count, self.custom_sharding_function)

    def test_save_index_shards_by_centroids_hnsw_quantizer(self):
        xb = np.random.rand(self.nb, self.d).astype('float32')
        quantizer = faiss.IndexHNSWFlat(self.d, 32)
        index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        shard_count = 17

        index.quantizer.add(xb)

        template = str(random.randint(0, 100000)) + "shard.%d.index"
        faiss.shard_ivf_index_centroids(
            index,
            shard_count,
            template,
            None,
            True
        )
        self.verify_sharded_ivf_indexes(
            template, xb, shard_count, self.default_sharding_function)

    def test_save_index_shards_by_centroids_binary_flat_quantizer(self):
        xb = np.random.randint(256, size=(self.nb, int(self.d / 8))).astype('uint8')
        quantizer = faiss.IndexBinaryFlat(self.d)
        index = faiss.IndexBinaryIVF(quantizer, self.d, self.nlist)
        shard_count = 11

        index.quantizer.add(xb)

        template = str(random.randint(0, 100000)) + "shard.%d.index"
        faiss.shard_binary_ivf_index_centroids(
            index,
            shard_count,
            template,
            None,
            True
        )
        self.verify_sharded_ivf_indexes(
            template, xb, shard_count, self.default_sharding_function)

    def test_save_index_shards_by_centroids_binary_hnsw_quantizer(self):
        xb = np.random.randint(256, size=(self.nb, int(self.d / 8))).astype('uint8')
        quantizer = faiss.IndexBinaryHNSW(self.d, 32)
        index = faiss.IndexBinaryIVF(quantizer, self.d, self.nlist)
        shard_count = 13

        index.quantizer.add(xb)

        template = str(random.randint(0, 100000)) + "shard.%d.index"
        faiss.shard_binary_ivf_index_centroids(
            index,
            shard_count,
            template,
            None,
            True
        )
        self.verify_sharded_ivf_indexes(
            template, xb, shard_count, self.default_sharding_function)

    def test_save_index_shards_without_id_generation(self):
        xb = np.random.randint(256, size=(self.nb, int(self.d / 8))).astype('uint8')
        quantizer = faiss.IndexBinaryHNSW(self.d, 32)
        index = faiss.IndexBinaryIVF(quantizer, self.d, self.nlist)
        shard_count = 5

        index.quantizer.add(xb)

        template = str(random.randint(0, 100000)) + "shard.%d.index"
        faiss.shard_binary_ivf_index_centroids(
            index,
            shard_count,
            template,
            None,
            False
        )
        self.verify_sharded_ivf_indexes(
            template, xb, shard_count, self.default_sharding_function, False)

        xb = np.random.rand(self.nb, self.d).astype('float32')
        quantizer = faiss.IndexHNSWFlat(self.d, 32)
        index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        shard_count = 23

        index.quantizer.add(xb)

        template = str(random.randint(0, 100000)) + "shard.%d.index"
        faiss.shard_ivf_index_centroids(
            index,
            shard_count,
            template,
            None,
            False
        )
        self.verify_sharded_ivf_indexes(
            template, xb, shard_count, self.default_sharding_function, False)
