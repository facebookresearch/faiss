# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

""" more elaborate that test_index.py """

import numpy as np
import unittest
import faiss


class TestRemove(unittest.TestCase):

    def test_remove(self):
        # only tests the python interface

        index = faiss.IndexFlat(5)
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10) + 1000
        index.add(xb)
        index.remove_ids(np.arange(5) * 2)
        xb2 = faiss.vector_float_to_array(index.xb).reshape(5, 5)
        assert np.all(xb2[:, 0] == xb[np.arange(5) * 2 + 1, 0])

    def test_remove_id_map(self):
        sub_index = faiss.IndexFlat(5)
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10) + 1000
        index = faiss.IndexIDMap2(sub_index)
        index.add_with_ids(xb, np.arange(10) + 100)
        assert index.reconstruct(104)[0] == 1004
        index.remove_ids(np.array([103]))
        assert index.reconstruct(104)[0] == 1004
        try:
            index.reconstruct(103)
        except:
            pass
        else:
            assert False, 'should have raised an exception'

    def test_remove_id_map_2(self):
        # from https://github.com/facebookresearch/faiss/issues/255
        rs = np.random.RandomState(1234)
        X = rs.randn(10, 10).astype(np.float32)
        idx = np.array([0, 10, 20, 30, 40, 5, 15, 25, 35, 45], np.int64)
        remove_set = np.array([10, 30], dtype=np.int64)
        index = faiss.index_factory(10, 'IDMap,Flat')
        index.add_with_ids(X[:5, :], idx[:5])
        index.remove_ids(remove_set)
        index.add_with_ids(X[5:, :], idx[5:])

        print (index.search(X, 1))

        for i in range(10):
            _, searchres = index.search(X[i:i + 1, :], 1)
            if idx[i] in remove_set:
                assert searchres[0] != idx[i]
            else:
                assert searchres[0] == idx[i]



class TestRangeSearch(unittest.TestCase):

    def test_range_search_id_map(self):
        sub_index = faiss.IndexFlat(5, 1)  # L2 search instead of inner product
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10) + 1000
        index = faiss.IndexIDMap2(sub_index)
        index.add_with_ids(xb, np.arange(10) + 100)
        dist = float(np.linalg.norm(xb[3] - xb[0])) * 0.99
        res_subindex = sub_index.range_search(xb[[0], :], dist)
        res_index = index.range_search(xb[[0], :], dist)
        assert len(res_subindex[2]) == 2
        np.testing.assert_array_equal(res_subindex[2] + 100, res_index[2])


class TestUpdate(unittest.TestCase):

    def test_update(self):
        d = 64
        nb = 1000
        nt = 1500
        nq = 100
        np.random.seed(123)
        xb = np.random.random(size=(nb, d)).astype('float32')
        xt = np.random.random(size=(nt, d)).astype('float32')
        xq = np.random.random(size=(nq, d)).astype('float32')

        index = faiss.index_factory(d, "IVF64,Flat")
        index.train(xt)
        index.add(xb)
        index.nprobe = 32
        D, I = index.search(xq, 5)

        index.make_direct_map()
        recons_before = np.vstack([index.reconstruct(i) for i in range(nb)])

        # revert order of the 200 first vectors
        nu = 200
        index.update_vectors(np.arange(nu), xb[nu - 1::-1].copy())

        recons_after = np.vstack([index.reconstruct(i) for i in range(nb)])

        # make sure reconstructions remain the same
        diff_recons = recons_before[:nu] - recons_after[nu - 1::-1]
        assert np.abs(diff_recons).max() == 0

        D2, I2 = index.search(xq, 5)

        assert np.all(D == D2)

        gt_map = np.arange(nb)
        gt_map[:nu] = np.arange(nu, 0, -1) - 1
        eqs = I.ravel() == gt_map[I2.ravel()]

        assert np.all(eqs)


class TestPCAWhite(unittest.TestCase):

    def test_white(self):

        # generate data
        d = 4
        nt = 1000
        nb = 200
        nq = 200

        # normal distribition
        x = faiss.randn((nt + nb + nq) * d, 1234).reshape(nt + nb + nq, d)

        index = faiss.index_factory(d, 'Flat')

        xt = x[:nt]
        xb = x[nt:-nq]
        xq = x[-nq:]

        # NN search on normal distribution
        index.add(xb)
        Do, Io = index.search(xq, 5)

        # make distribution very skewed
        x *= [10, 4, 1, 0.5]
        rr, _ = np.linalg.qr(faiss.randn(d * d).reshape(d, d))
        x = np.dot(x, rr).astype('float32')

        xt = x[:nt]
        xb = x[nt:-nq]
        xq = x[-nq:]

        # L2 search on skewed distribution
        index = faiss.index_factory(d, 'Flat')

        index.add(xb)
        Dl2, Il2 = index.search(xq, 5)

        # whiten + L2 search on L2 distribution
        index = faiss.index_factory(d, 'PCAW%d,Flat' % d)

        index.train(xt)
        index.add(xb)
        Dw, Iw = index.search(xq, 5)

        # make sure correlation of whitened results with original
        # results is much better than simple L2 distances
        # should be 961 vs. 264
        assert (faiss.eval_intersection(Io, Iw) >
                2 * faiss.eval_intersection(Io, Il2))


class TestTransformChain(unittest.TestCase):

    def test_chain(self):

        # generate data
        d = 4
        nt = 1000
        nb = 200
        nq = 200

        # normal distribition
        x = faiss.randn((nt + nb + nq) * d, 1234).reshape(nt + nb + nq, d)

        # make distribution very skewed
        x *= [10, 4, 1, 0.5]
        rr, _ = np.linalg.qr(faiss.randn(d * d).reshape(d, d))
        x = np.dot(x, rr).astype('float32')

        xt = x[:nt]
        xb = x[nt:-nq]
        xq = x[-nq:]

        index = faiss.index_factory(d, "L2norm,PCA2,L2norm,Flat")

        assert index.chain.size() == 3
        l2_1 = faiss.downcast_VectorTransform(index.chain.at(0))
        assert l2_1.norm == 2
        pca = faiss.downcast_VectorTransform(index.chain.at(1))
        assert not pca.is_trained
        index.train(xt)
        assert pca.is_trained

        index.add(xb)
        D, I = index.search(xq, 5)

        # do the computation manually and check if we get the same result
        def manual_trans(x):
            x = x.copy()
            faiss.normalize_L2(x)
            x = pca.apply_py(x)
            faiss.normalize_L2(x)
            return x

        index2 = faiss.IndexFlatL2(2)
        index2.add(manual_trans(xb))
        D2, I2 = index2.search(manual_trans(xq), 5)

        assert np.all(I == I2)


if __name__ == '__main__':
    unittest.main()
