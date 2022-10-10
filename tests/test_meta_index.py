# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import faiss
import unittest

from common_faiss_tests import Randu10k

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

        print('ref search')
        ref_index.add(xb)
        _Dref, Iref = ref_index.search(xq, k)
        print(Iref[:5, :6])

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

            print('shard search test_no = %d' % test_no)
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

            print(I[:5, :6])

            if with_threads:
                faiss.omp_set_num_threads(remember_nt)

            ndiff = (I != Iref).sum()

            print('%d / %d differences' % (ndiff, nq * k))
            assert (ndiff < nq * k / 1000.)


if __name__ == '__main__':
    unittest.main()
