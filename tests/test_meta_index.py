# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import faiss
import unittest

from common_faiss_tests import Randu10k

from faiss.contrib.datasets import SyntheticDataset

ru = Randu10k()

xb = ru.xb
xt = ru.xt
xq = ru.xq
nb, d = xb.shape
nq, d = xq.shape


class IDRemap(unittest.TestCase):

    def test_id_remap_idmap(self):
        # reference: index without remapping

        index = faiss.IndexPQ(d, 8, 8)
        k = 10
        index.train(xt)
        index.add(xb)
        _Dref, Iref = index.search(xq, k)

        # try a remapping
        ids = np.arange(nb)[::-1].copy().astype("int64")

        sub_index = faiss.IndexPQ(d, 8, 8)
        index2 = faiss.IndexIDMap(sub_index)

        index2.train(xt)
        index2.add_with_ids(xb, ids)

        _D, I = index2.search(xq, k)

        assert np.all(I == nb - 1 - Iref)

    def test_id_remap_ivf(self):
        # coarse quantizer in common
        coarse_quantizer = faiss.IndexFlatIP(d)
        ncentroids = 25

        # reference: index without remapping

        index = faiss.IndexIVFPQ(coarse_quantizer, d, ncentroids, 8, 8)
        index.nprobe = 5
        k = 10
        index.train(xt)
        index.add(xb)
        _Dref, Iref = index.search(xq, k)

        # try a remapping
        ids = np.arange(nb)[::-1].copy().astype("int64")

        index2 = faiss.IndexIVFPQ(coarse_quantizer, d, ncentroids, 8, 8)
        index2.nprobe = 5

        index2.train(xt)
        index2.add_with_ids(xb, ids)

        _D, I = index2.search(xq, k)
        assert np.all(I == nb - 1 - Iref)


class Shards(unittest.TestCase):

    def add_flat_shards(
        self,
        shard_index,
        xbase,
        metric=faiss.METRIC_L2,
        metric_arg=0,
        ni=3,
        wrap=None,
    ):
        """Split xbase over ni IndexFlat shards of shard_index. xbase may be
        None to leave the shards empty, e.g. to fill them via shard_index.add()
        """
        for i in range(ni):
            n = 0 if xbase is None else len(xbase)
            shard = faiss.IndexFlat(shard_index.d, metric)
            shard.metric_arg = metric_arg
            if xbase is not None:
                shard.add(xbase[i * n // ni : (i + 1) * n // ni])
            shard_index.add_shard(shard if wrap is None else wrap(shard))

    @unittest.skipIf(
        os.name == "posix" and os.uname().sysname == "Darwin",
        "There is a bug in the OpenMP implementation on OSX.",
    )
    def test_shards(self):
        k = 32
        ref_index = faiss.IndexFlatL2(d)

        ref_index.add(xb)
        _Dref, Iref = ref_index.search(xq, k)

        # Create both threaded and non-threaded shard indexes
        shard_index_nonthreaded = faiss.IndexShards(
            d, False
        )  # explicitly non-threaded
        shard_index_threaded = faiss.IndexShards(d, True)  # explicitly threaded
        shard_index_2 = faiss.IndexShards(d, True, False)

        self.add_flat_shards(shard_index_nonthreaded, xb)
        self.add_flat_shards(shard_index_threaded, xb)
        # populated below by the parallel add rather than shard by shard
        self.add_flat_shards(shard_index_2, None, wrap=faiss.IndexIDMap)

        # test parallel add
        shard_index_2.verbose = True
        shard_index_2.add(xb)

        for test_no in range(3):
            with_threads = test_no == 1

            if with_threads:
                remember_nt = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(1)
                # Use the threaded index
                test_index = shard_index_threaded
            else:
                # Use the non-threaded index
                test_index = shard_index_nonthreaded

            if test_no != 2:
                _D, I = test_index.search(xq, k)
            else:
                _D, I = shard_index_2.search(xq, k)

            if with_threads:
                faiss.omp_set_num_threads(remember_nt)

            ndiff = (I != Iref).sum()
            # IndexShards merges per-shard top-k by distance; float32 ULP ties
            # at the k=32 boundary reorder neighbors vs. the unsharded
            # IndexFlatL2 reference (amplified by the threaded shard merge).
            # Allow ~1% mismatches; a real merge regression collapses
            # thousands of the nq*k cells, far above this floor.
            assert ndiff < nq * k / 100.0, f"too many mismatches: {ndiff}"

    def test_shards_metrics(self):
        # IndexShards merges the per-shard results itself, so it has to know
        # whether the metric ranks by similarity (largest first) or by distance
        # (smallest first). Only METRIC_L2 used to be treated as a distance, so
        # every other dis-similarity metric returned the FARTHEST vectors once
        # results crossed a shard boundary.
        rs = np.random.RandomState(123)
        dim, n, nquery, k = 16, 600, 50, 10
        # Components in [0, 1) keep every metric well-defined: Jaccard needs
        # positive components and GOWER numeric dimensions in [0, 1].
        base = rs.rand(n, dim).astype("float32")
        queries = rs.rand(nquery, dim).astype("float32")
        p = 1.5  # metric_arg for METRIC_Lp

        for metric in (
            faiss.METRIC_INNER_PRODUCT,
            faiss.METRIC_L2,
            faiss.METRIC_L1,
            faiss.METRIC_Linf,
            faiss.METRIC_Lp,
            faiss.METRIC_Canberra,
            faiss.METRIC_BrayCurtis,
            faiss.METRIC_JensenShannon,
            faiss.METRIC_Jaccard,
            faiss.METRIC_NaNEuclidean,
            faiss.METRIC_GOWER,
        ):
            with self.subTest(metric=metric):
                ref_index = faiss.IndexFlat(dim, metric)
                ref_index.metric_arg = p
                ref_index.add(base)
                Dref, _Iref = ref_index.search(queries, k)

                shard_index = faiss.IndexShards(dim, False, True)
                self.add_flat_shards(shard_index, base, metric, metric_arg=p)
                D, _I = shard_index.search(queries, k)

                if faiss.is_similarity_metric(metric):
                    self.assertTrue(np.all(D[:, :-1] >= D[:, 1:]))
                else:
                    self.assertTrue(np.all(D[:, :-1] <= D[:, 1:]))
                # Same neighbors as the unsharded index. Distances are compared
                # rather than labels so equidistant neighbors may come in
                # either order.
                np.testing.assert_array_almost_equal(D, Dref, decimal=5)

    def test_shards_ivf(self):
        ds = SyntheticDataset(32, 1000, 100, 20)
        ref_index = faiss.index_factory(ds.d, "IVF32,SQ8")
        ref_index.train(ds.get_train())
        xb = ds.get_database()
        ref_index.add(ds.get_database())

        Dref, Iref = ref_index.search(ds.get_database(), 10)
        ref_index.reset()

        sharded_index = faiss.IndexShardsIVF(
            ref_index.quantizer, ref_index.nlist, False, True
        )
        for shard in range(3):
            index_i = faiss.clone_index(ref_index)
            index_i.add(xb[shard * nb // 3 : (shard + 1) * nb // 3])
            sharded_index.add_shard(index_i)

        Dnew, Inew = sharded_index.search(ds.get_database(), 10)

        np.testing.assert_equal(Inew, Iref)
        np.testing.assert_allclose(Dnew, Dref)

    def test_shards_ivf_train_add(self):
        ds = SyntheticDataset(32, 1000, 600, 20)
        quantizer = faiss.IndexFlatL2(ds.d)
        sharded_index = faiss.IndexShardsIVF(quantizer, 40, False, False)

        for _ in range(3):
            sharded_index.add_shard(faiss.index_factory(ds.d, "IVF40,Flat"))

        sharded_index.train(ds.get_train())
        sharded_index.add(ds.get_database())
        Dnew, Inew = sharded_index.search(ds.get_queries(), 10)

        index_ref = faiss.IndexIVFFlat(quantizer, ds.d, sharded_index.nlist)
        index_ref.train(ds.get_train())
        index_ref.add(ds.get_database())
        Dref, Iref = index_ref.search(ds.get_queries(), 10)
        np.testing.assert_equal(Inew, Iref)
        np.testing.assert_allclose(Dnew, Dref)

        # mess around with the quantizer's centroids
        centroids = quantizer.reconstruct_n()
        centroids = centroids[::-1].copy()
        quantizer.reset()
        quantizer.add(centroids)

        D2, I2 = sharded_index.search(ds.get_queries(), 10)
        self.assertFalse(np.all(I2 == Inew))
