# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" more elaborate that test_index.py """
from __future__ import absolute_import, division, print_function

import numpy as np
import unittest
import faiss
import os
import shutil
import tempfile
import platform

from common_faiss_tests import get_dataset_2, get_dataset
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.inspect_tools import make_LinearTransform_matrix
from faiss.contrib.evaluation import check_ref_knn_with_draws

class TestRemoveFastScan(unittest.TestCase):
    def do_test(self, ntotal, removed):
        d = 20
        xt, xb, _ = get_dataset_2(d, ntotal, ntotal, 0)
        index = faiss.index_factory(20, 'IDMap2,PQ5x4fs')
        index.train(xt)
        index.add_with_ids(xb, np.arange(ntotal).astype("int64"))
        before = index.reconstruct_n(0, ntotal)
        index.remove_ids(np.array(removed))
        for i in range(ntotal):
            if i in removed:
                # should throw RuntimeError as this vector should be removed
                try:
                    after = index.reconstruct(i)
                    assert False
                except RuntimeError:
                    pass
            else:
                after = index.reconstruct(i)
                np.testing.assert_array_equal(before[i], after)
        assert index.ntotal == ntotal - len(removed)

    def test_remove_last_vector(self):
        self.do_test(993, [992])

    # test remove element from every address 0 -> 31
    # [0, 32 + 1, 2 * 32 + 2, ....]
    # [0,   33  ,     66    , 99, 132, .....]
    def test_remove_every_address(self):
        removed = (33 * np.arange(32)).tolist()
        self.do_test(1100, removed)

    # test remove range of vectors and leave ntotal divisible by 32
    def test_leave_complete_block(self):
        self.do_test(1000, np.arange(8).tolist())


class TestRemove(unittest.TestCase):

    def do_merge_then_remove(self, ondisk):
        d = 10
        nb = 1000
        nq = 200
        nt = 200

        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        quantizer = faiss.IndexFlatL2(d)

        index1 = faiss.IndexIVFFlat(quantizer, d, 20)
        index1.train(xt)

        filename = None
        if ondisk:
            filename = tempfile.mkstemp()[1]
            invlists = faiss.OnDiskInvertedLists(
                index1.nlist, index1.code_size,
                filename)
            index1.replace_invlists(invlists)

        index1.add(xb[:int(nb / 2)])

        index2 = faiss.IndexIVFFlat(quantizer, d, 20)
        assert index2.is_trained
        index2.add(xb[int(nb / 2):])

        Dref, Iref = index1.search(xq, 10)
        index1.merge_from(index2, int(nb / 2))

        assert index1.ntotal == nb

        index1.remove_ids(faiss.IDSelectorRange(int(nb / 2), nb))

        assert index1.ntotal == int(nb / 2)
        Dnew, Inew = index1.search(xq, 10)

        assert np.all(Dnew == Dref)
        assert np.all(Inew == Iref)

        if filename is not None:
            os.unlink(filename)

    def test_remove_regular(self):
        self.do_merge_then_remove(False)

    @unittest.skipIf(platform.system() == 'Windows',
                     'OnDiskInvertedLists is unsupported on Windows.')
    def test_remove_ondisk(self):
        self.do_merge_then_remove(True)

    def test_remove(self):
        # only tests the python interface

        index = faiss.IndexFlat(5)
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10, dtype='int64') + 1000
        index.add(xb)
        index.remove_ids(np.arange(5, dtype='int64') * 2)
        xb2 = faiss.vector_float_to_array(index.codes)
        xb2 = xb2.view("float32").reshape(5, 5)
        assert np.all(xb2[:, 0] == xb[np.arange(5) * 2 + 1, 0])

    def test_remove_id_map(self):
        sub_index = faiss.IndexFlat(5)
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10) + 1000
        index = faiss.IndexIDMap2(sub_index)
        index.add_with_ids(xb, np.arange(10, dtype='int64') + 100)
        assert index.reconstruct(104)[0] == 1004
        index.remove_ids(np.array([103], dtype='int64'))
        assert index.reconstruct(104)[0] == 1004
        try:
            index.reconstruct(103)
        except RuntimeError:
            pass
        else:
            assert False, 'should have raised an exception'

    def test_factory_idmap2_suffix(self):
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10) + 1000
        index = faiss.index_factory(5, "Flat,IDMap2")
        ids = np.arange(10, dtype='int64') + 100
        index.add_with_ids(xb, ids)
        assert index.reconstruct(104)[0] == 1004
        index.remove_ids(np.array([103], dtype='int64'))
        assert index.reconstruct(104)[0] == 1004

    def test_factory_idmap2_prefix(self):
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10) + 1000
        index = faiss.index_factory(5, "IDMap2,Flat")
        ids = np.arange(10, dtype='int64') + 100
        index.add_with_ids(xb, ids)
        assert index.reconstruct(109)[0] == 1009
        index.remove_ids(np.array([100], dtype='int64'))
        assert index.reconstruct(109)[0] == 1009

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

        for i in range(10):
            _, searchres = index.search(X[i:i + 1, :], 1)
            if idx[i] in remove_set:
                assert searchres[0] != idx[i]
            else:
                assert searchres[0] == idx[i]

    def test_remove_id_map_binary(self):
        sub_index = faiss.IndexBinaryFlat(40)
        xb = np.zeros((10, 5), dtype='uint8')
        xb[:, 0] = np.arange(10) + 100
        index = faiss.IndexBinaryIDMap2(sub_index)
        index.add_with_ids(xb, np.arange(10, dtype='int64') + 1000)
        assert index.reconstruct(1004)[0] == 104
        index.remove_ids(np.array([1003], dtype='int64'))
        assert index.reconstruct(1004)[0] == 104
        try:
            index.reconstruct(1003)
        except RuntimeError:
            pass
        else:
            assert False, 'should have raised an exception'

        # while we are there, let's test I/O as well...
        fd, tmpnam = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index_binary(index, tmpnam)
            index = faiss.read_index_binary(tmpnam)
        finally:
            os.remove(tmpnam)

        assert index.reconstruct(1004)[0] == 104
        try:
            index.reconstruct(1003)
        except RuntimeError:
            pass
        else:
            assert False, 'should have raised an exception'


class TestRangeSearch(unittest.TestCase):

    def test_range_search_id_map(self):
        sub_index = faiss.IndexFlat(5, 1)  # L2 search instead of inner product
        xb = np.zeros((10, 5), dtype='float32')
        xb[:, 0] = np.arange(10) + 1000
        index = faiss.IndexIDMap2(sub_index)
        index.add_with_ids(xb, np.arange(10, dtype=np.int64) + 100)
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
        index.update_vectors(np.arange(nu).astype('int64'),
                             xb[nu - 1::-1].copy())

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


@unittest.skipIf(platform.system() == 'Windows', \
                 'Mmap not supported on Windows.')
class TestRareIO(unittest.TestCase):

    def compare_results(self, index1, index2, xq):

        Dref, Iref = index1.search(xq, 5)
        Dnew, Inew = index2.search(xq, 5)

        assert np.all(Dref == Dnew)
        assert np.all(Iref == Inew)

    def do_mmappedIO(self, sparse, in_pretransform=False):
        d = 10
        nb = 1000
        nq = 200
        nt = 200
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        quantizer = faiss.IndexFlatL2(d)
        index1 = faiss.IndexIVFFlat(quantizer, d, 20)
        if sparse:
            # makes the inverted lists sparse because all elements get
            # assigned to the same invlist
            xt += (np.ones(10) * 1000).astype('float32')

        if in_pretransform:
            # make sure it still works when wrapped in an IndexPreTransform
            index1 = faiss.IndexPreTransform(index1)

        index1.train(xt)
        index1.add(xb)

        _, fname = tempfile.mkstemp()
        try:

            faiss.write_index(index1, fname)

            index2 = faiss.read_index(fname)
            self.compare_results(index1, index2, xq)

            index3 = faiss.read_index(fname, faiss.IO_FLAG_MMAP)
            self.compare_results(index1, index3, xq)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_mmappedIO_sparse(self):
        self.do_mmappedIO(True)

    def test_mmappedIO_full(self):
        self.do_mmappedIO(False)

    def test_mmappedIO_pretrans(self):
        self.do_mmappedIO(False, True)


class TestIVFFlatDedup(unittest.TestCase):

    def test_dedup(self):
        d = 10
        nb = 1000
        nq = 200
        nt = 500
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        # introduce duplicates
        xb[500:900:2] = xb[501:901:2]
        xb[901::4] = xb[900::4]
        xb[902::4] = xb[900::4]
        xb[903::4] = xb[900::4]

        # also in the train set
        xt[201::2] = xt[200::2]

        quantizer = faiss.IndexFlatL2(d)
        index_new = faiss.IndexIVFFlatDedup(quantizer, d, 20)

        index_new.verbose = True
        # should display
        # IndexIVFFlatDedup::train: train on 350 points after dedup (was 500 points)
        index_new.train(xt)

        index_ref = faiss.IndexIVFFlat(quantizer, d, 20)
        assert index_ref.is_trained

        index_ref.nprobe = 5
        index_ref.add(xb)
        index_new.nprobe = 5
        index_new.add(xb)

        Dref, Iref = index_ref.search(xq, 20)
        Dnew, Inew = index_new.search(xq, 20)

        check_ref_knn_with_draws(Dref, Iref, Dnew, Inew)

        # test I/O
        fd, tmpfile = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index_new, tmpfile)
            index_st = faiss.read_index(tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)
        Dst, Ist = index_st.search(xq, 20)

        check_ref_knn_with_draws(Dnew, Inew, Dst, Ist)

        # test remove
        toremove = np.hstack((np.arange(3, 1000, 5), np.arange(850, 950)))
        toremove = toremove.astype(np.int64)
        index_ref.remove_ids(toremove)
        index_new.remove_ids(toremove)

        Dref, Iref = index_ref.search(xq, 20)
        Dnew, Inew = index_new.search(xq, 20)

        check_ref_knn_with_draws(Dref, Iref, Dnew, Inew)


class TestSerialize(unittest.TestCase):

    def test_serialize_to_vector(self):
        d = 10
        nb = 1000
        nq = 200
        nt = 500
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        index = faiss.IndexFlatL2(d)
        index.add(xb)

        Dref, Iref = index.search(xq, 5)

        writer = faiss.VectorIOWriter()
        faiss.write_index(index, writer)

        ar_data = faiss.vector_to_array(writer.data)

        # direct transfer of vector
        reader = faiss.VectorIOReader()
        reader.data.swap(writer.data)

        index2 = faiss.read_index(reader)

        Dnew, Inew = index2.search(xq, 5)
        assert np.all(Dnew == Dref) and np.all(Inew == Iref)

        # from intermediate numpy array
        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(ar_data, reader.data)

        index3 = faiss.read_index(reader)

        Dnew, Inew = index3.search(xq, 5)
        assert np.all(Dnew == Dref) and np.all(Inew == Iref)


@unittest.skipIf(platform.system() == 'Windows',
                 'OnDiskInvertedLists is unsupported on Windows.')
class TestRenameOndisk(unittest.TestCase):

    def test_rename(self):
        d = 10
        nb = 500
        nq = 100
        nt = 100

        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        quantizer = faiss.IndexFlatL2(d)

        index1 = faiss.IndexIVFFlat(quantizer, d, 20)
        index1.train(xt)

        dirname = tempfile.mkdtemp()

        try:

            # make an index with ondisk invlists
            invlists = faiss.OnDiskInvertedLists(
                index1.nlist, index1.code_size,
                dirname + '/aa.ondisk')
            index1.replace_invlists(invlists)
            index1.add(xb)
            D1, I1 = index1.search(xq, 10)
            faiss.write_index(index1, dirname + '/aa.ivf')

            # move the index elsewhere
            os.mkdir(dirname + '/1')
            for fname in 'aa.ondisk', 'aa.ivf':
                os.rename(dirname + '/' + fname,
                          dirname + '/1/' + fname)

            # try to read it: fails!
            try:
                index2 = faiss.read_index(dirname + '/1/aa.ivf')
            except RuntimeError:
                pass   # normal
            else:
                assert False

            # read it with magic flag
            index2 = faiss.read_index(dirname + '/1/aa.ivf',
                                      faiss.IO_FLAG_ONDISK_SAME_DIR)
            D2, I2 = index2.search(xq, 10)
            assert np.all(I1 == I2)

        finally:
            shutil.rmtree(dirname)


class TestInvlistMeta(unittest.TestCase):

    def test_slice_vstack(self):
        d = 10
        nb = 1000
        nq = 100
        nt = 200

        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 30)

        index.train(xt)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        # faiss.wait()

        il0 = index.invlists
        ils = []
        ilv = faiss.InvertedListsPtrVector()
        for sl in 0, 1, 2:
            il = faiss.SliceInvertedLists(il0, sl * 10, sl * 10 + 10)
            ils.append(il)
            ilv.push_back(il)

        il2 = faiss.VStackInvertedLists(ilv.size(), ilv.data())

        index2 = faiss.IndexIVFFlat(quantizer, d, 30)
        index2.replace_invlists(il2)
        index2.ntotal = index.ntotal

        D, I = index2.search(xq, 10)
        assert np.all(D == Dref)
        assert np.all(I == Iref)

    def test_stop_words(self):
        d = 10
        nb = 1000
        nq = 1
        nt = 200

        xt, xb, xq = get_dataset_2(d, nt, nb, nq)

        index = faiss.index_factory(d, "IVF32,Flat")
        index.nprobe = 4
        index.train(xt)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        il = index.invlists
        maxsz = max(il.list_size(i) for i in range(il.nlist))

        il2 = faiss.StopWordsInvertedLists(il, maxsz + 1)
        index.own_invlists
        index.own_invlists = False

        index.replace_invlists(il2, False)
        D1, I1 = index.search(xq, 10)
        np.testing.assert_array_equal(Dref, D1)
        np.testing.assert_array_equal(Iref, I1)

        # cleanup to avoid segfault on exit
        index.replace_invlists(il, False)

        # voluntarily unbalance one invlist
        i = int(I1[0, 0])
        index.add(np.vstack([xb[i]] * (maxsz + 10)))

        # introduce stopwords again
        index.replace_invlists(il2, False)

        D2, I2 = index.search(xq, 10)
        self.assertFalse(i in list(I2.ravel()))

        # avoid mem leak
        index.replace_invlists(il, True)


class TestSplitMerge(unittest.TestCase):

    def do_test(self, index_key, subset_type):
        xt, xb, xq = get_dataset_2(32, 1000, 100, 10)
        index = faiss.index_factory(32, index_key)
        index.train(xt)
        nsplit = 3
        sub_indexes = [faiss.clone_index(index) for i in range(nsplit)]
        index.add(xb)
        Dref, Iref = index.search(xq, 10)
        nlist = index.nlist
        for i in range(nsplit):
            if subset_type in (1, 3):
                index.copy_subset_to(sub_indexes[i], subset_type, nsplit, i)
            elif subset_type in (0, 2):
                j0 = index.ntotal * i // nsplit
                j1 = index.ntotal * (i + 1) // nsplit
                index.copy_subset_to(sub_indexes[i], subset_type, j0, j1)
            elif subset_type == 4:
                index.copy_subset_to(
                    sub_indexes[i], subset_type,
                    i * nlist // nsplit, (i + 1) * nlist // nsplit)

        index_shards = faiss.IndexShards(False, False)
        for i in range(nsplit):
            index_shards.add_shard(sub_indexes[i])
        Dnew, Inew = index_shards.search(xq, 10)
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

    def test_Flat_subset_type_0(self):
        self.do_test("IVF30,Flat", subset_type=0)

    def test_Flat_subset_type_1(self):
        self.do_test("IVF30,Flat", subset_type=1)

    def test_Flat_subset_type_2(self):
        self.do_test("IVF30,PQ4np", subset_type=2)

    def test_Flat_subset_type_3(self):
        self.do_test("IVF30,Flat", subset_type=3)

    def test_Flat_subset_type_4(self):
        self.do_test("IVF30,Flat", subset_type=4)


class TestIndependentQuantizer(unittest.TestCase):

    def test_sidebyside(self):
        """ provide double-sized vectors to the index, where each vector
        is the concatenation of twice the same vector """
        ds = SyntheticDataset(32, 1000, 500, 50)

        index = faiss.index_factory(ds.d, "IVF32,SQ8")
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4
        Dref, Iref = index.search(ds.get_queries(), 10)

        select32first = make_LinearTransform_matrix(
            np.eye(64, dtype='float32')[:32])

        select32last = make_LinearTransform_matrix(
            np.eye(64, dtype='float32')[32:])

        quantizer = faiss.IndexPreTransform(
            select32first,
            index.quantizer
        )

        index2 = faiss.IndexIVFIndependentQuantizer(
            quantizer,
            index, select32last
        )

        xq2 = np.hstack([ds.get_queries()] * 2)
        quantizer.search(xq2, 30)
        Dnew, Inew = index2.search(xq2, 10)

        np.testing.assert_array_equal(Dref, Dnew)
        np.testing.assert_array_equal(Iref, Inew)

        # test add
        index2.reset()
        xb2 = np.hstack([ds.get_database()] * 2)
        index2.add(xb2)
        Dnew, Inew = index2.search(xq2, 10)

        np.testing.assert_array_equal(Dref, Dnew)
        np.testing.assert_array_equal(Iref, Inew)

    def test_half_store(self):
        """ the index stores only the first half of each vector
        but the coarse quantizer sees them entirely """
        ds = SyntheticDataset(32, 1000, 500, 50)
        gt = ds.get_groundtruth(10)

        select32first = make_LinearTransform_matrix(
            np.eye(32, dtype='float32')[:16])

        index_ivf = faiss.index_factory(ds.d // 2, "IVF32,Flat")
        index_ivf.nprobe = 4
        index = faiss.IndexPreTransform(select32first, index_ivf)
        index.train(ds.get_train())
        index.add(ds.get_database())

        Dref, Iref = index.search(ds.get_queries(), 10)
        perf_ref = faiss.eval_intersection(Iref, gt)

        index_ivf = faiss.index_factory(ds.d // 2, "IVF32,Flat")
        index_ivf.nprobe = 4
        index = faiss.IndexIVFIndependentQuantizer(
            faiss.IndexFlatL2(ds.d),
            index_ivf, select32first
        )
        index.train(ds.get_train())
        index.add(ds.get_database())

        Dnew, Inew = index.search(ds.get_queries(), 10)
        perf_new = faiss.eval_intersection(Inew, gt)

        self.assertLess(perf_ref, perf_new)

    def test_precomputed_tables(self):
        """ see how precomputed tables behave with centroid distance estimates from a mismatching
        coarse quantizer """
        ds = SyntheticDataset(48, 2000, 500, 250)
        gt = ds.get_groundtruth(10)

        index = faiss.IndexIVFIndependentQuantizer(
            faiss.IndexFlatL2(48),
            faiss.index_factory(16, "IVF64,PQ4np"),
            faiss.PCAMatrix(48, 16)
        )
        index.train(ds.get_train())
        index.add(ds.get_database())

        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        index_ivf.nprobe = 4

        Dref, Iref = index.search(ds.get_queries(), 10)
        perf_ref = faiss.eval_intersection(Iref, gt)

        index_ivf.use_precomputed_table = 1
        index_ivf.precompute_table()

        Dnew, Inew = index.search(ds.get_queries(), 10)
        perf_new = faiss.eval_intersection(Inew, gt)

        # to be honest, it is not clear which one is better...
        self.assertNotEqual(perf_ref, perf_new)

        # check IO while we are at it
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        D2, I2 = index2.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(Dnew, D2)
        np.testing.assert_array_equal(Inew, I2)



class TestSearchAndReconstruct(unittest.TestCase):

    def run_search_and_reconstruct(self, index, xb, xq, k=10, eps=None):
        n, d = xb.shape
        assert xq.shape[1] == d
        assert index.d == d

        D_ref, I_ref = index.search(xq, k)
        R_ref = index.reconstruct_n(0, n)
        D, I, R = index.search_and_reconstruct(xq, k)

        np.testing.assert_almost_equal(D, D_ref, decimal=5)
        self.assertTrue((I == I_ref).all())
        self.assertEqual(R.shape[:2], I.shape)
        self.assertEqual(R.shape[2], d)

        # (n, k, ..) -> (n * k, ..)
        I_flat = I.reshape(-1)
        R_flat = R.reshape(-1, d)
        # Filter out -1s when not enough results
        R_flat = R_flat[I_flat >= 0]
        I_flat = I_flat[I_flat >= 0]

        recons_ref_err = np.mean(np.linalg.norm(R_flat - R_ref[I_flat]))
        self.assertLessEqual(recons_ref_err, 1e-6)

        def norm1(x):
            return np.sqrt((x ** 2).sum(axis=1))

        recons_err = np.mean(norm1(R_flat - xb[I_flat]))

        print('Reconstruction error = %.3f' % recons_err)
        if eps is not None:
            self.assertLessEqual(recons_err, eps)

        return D, I, R

    def test_IndexFlat(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.IndexFlatL2(d)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=0.0)

    def test_IndexIVFFlat(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 32, faiss.METRIC_L2)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.nprobe = 4
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=0.0)

    def test_IndexIVFPQ(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, 32, 8, 8)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.nprobe = 4
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=1.0)

    def test_MultiIndex(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.index_factory(d, "IMI2x5,PQ8np")
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", 4)
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=1.0)

    def test_IndexTransform(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.index_factory(d, "L2norm,PCA8,IVF32,PQ8np")
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", 4)
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq)


class TestSearchAndGetCodes(unittest.TestCase):

    def do_test(self, factory_string):
        ds = SyntheticDataset(32, 1000, 100, 10)

        index = faiss.index_factory(ds.d, factory_string)

        index.train(ds.get_train())
        index.add(ds.get_database())

        index.nprobe
        index.nprobe = 10
        Dref, Iref = index.search(ds.get_queries(), 10)

        D, I, codes = index.search_and_return_codes(
            ds.get_queries(), 10, include_listnos=True)

        np.testing.assert_array_equal(I, Iref)
        np.testing.assert_array_equal(D, Dref)

        # verify that we get the same distances when decompressing from
        # returned codes (the codes are compatible with sa_decode)
        for qi in range(ds.nq):
            q = ds.get_queries()[qi]
            xbi = index.sa_decode(codes[qi])
            D2 = ((q - xbi) ** 2).sum(1)
            np.testing.assert_allclose(D2, D[qi], rtol=1e-5)

    def test_ivfpq(self):
        self.do_test("IVF20,PQ4x4np")

    def test_ivfsq(self):
        self.do_test("IVF20,SQ8")

    def test_ivfrq(self):
        self.do_test("IVF20,RQ3x4")
