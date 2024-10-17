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
        ids = np.arange(nb)[::-1].copy().astype('int64')

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

        index = faiss.IndexIVFPQ(coarse_quantizer, d,
                                        ncentroids, 8, 8)
        index.nprobe = 5
        k = 10
        index.train(xt)
        index.add(xb)
        _Dref, Iref = index.search(xq, k)

        # try a remapping
        ids = np.arange(nb)[::-1].copy().astype('int64')

        index2 = faiss.IndexIVFPQ(coarse_quantizer, d,
                                        ncentroids, 8, 8)
        index2.nprobe = 5

        index2.train(xt)
        index2.add_with_ids(xb, ids)

        _D, I = index2.search(xq, k)
        assert np.all(I == nb - 1 - Iref)


class Shards(unittest.TestCase):

    @unittest.skipIf(os.name == "posix" and os.uname().sysname == "Darwin",
                     "There is a bug in the OpenMP implementation on OSX.")
    def test_shards(self):
        k = 32
        ref_index = faiss.IndexFlatL2(d)

        ref_index.add(xb)
        _Dref, Iref = ref_index.search(xq, k)

        shard_index = faiss.IndexShards(d)
        shard_index_2 = faiss.IndexShards(d, True, False)

        ni = 3
        for i in range(ni):
            i0 = int(i * nb / ni)
            i1 = int((i + 1) * nb / ni)
            index = faiss.IndexFlatL2(d)
            index.add(xb[i0:i1])
            shard_index.add_shard(index)

            index_2 = faiss.IndexFlatL2(d)
            irm = faiss.IndexIDMap(index_2)
            shard_index_2.add_shard(irm)

        # test parallel add
        shard_index_2.verbose = True
        shard_index_2.add(xb)

        for test_no in range(3):
            with_threads = test_no == 1

            if with_threads:
                remember_nt = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(1)
                shard_index.threaded = True
            else:
                shard_index.threaded = False

            if test_no != 2:
                _D, I = shard_index.search(xq, k)
            else:
                _D, I = shard_index_2.search(xq, k)

            if with_threads:
                faiss.omp_set_num_threads(remember_nt)

            ndiff = (I != Iref).sum()
            assert (ndiff < nq * k / 1000.)

    def test_shards_ivf(self):
        ds = SyntheticDataset(32, 1000, 100, 20)
        ref_index = faiss.index_factory(ds.d, "IVF32,SQ8")
        ref_index.train(ds.get_train())
        xb = ds.get_database()
        ref_index.add(ds.get_database())

        Dref, Iref = ref_index.search(ds.get_database(), 10)
        ref_index.reset()

        sharded_index = faiss.IndexShardsIVF(
            ref_index.quantizer, ref_index.nlist, False, True)
        for shard in range(3):
            index_i = faiss.clone_index(ref_index)
            index_i.add(xb[shard * nb // 3: (shard + 1)* nb // 3])
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
