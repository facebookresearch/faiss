# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import numpy as np
import faiss
from faiss.contrib import datasets

from common_faiss_tests import for_all_simd_levels, NoneSIMDLevel


def make_index(d, nlist, mem_dim, nb_bits, filename,
               metric=faiss.METRIC_L2):
    """Rotation to a multiple of 64 dimensions, then IndexIVFRaBitQSSD."""
    rot = faiss.HadamardRotation(d, 12345, False)
    if metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(rot.d_out)
    else:
        quantizer = faiss.IndexFlatIP(rot.d_out)
    index_ivf = faiss.IndexIVFRaBitQSSD(
        quantizer, rot.d_out, nlist, mem_dim, nb_bits, filename, metric)
    return faiss.IndexPreTransform(rot, index_ivf)


def recall_at_1(I, gt):
    return (I[:, 0] == gt[:, 0]).mean()


@for_all_simd_levels
@unittest.skipUnless(
    hasattr(faiss, "IndexIVFRaBitQSSD"), "IndexIVFRaBitQSSD not built")
class TestIVFRaBitQSSD(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()
        super().tearDown()

    def path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def build(self, ds, nlist, mem_dim, nb_bits, metric=faiss.METRIC_L2):
        index = make_index(ds.d, nlist, mem_dim, nb_bits,
                           self.path("index.ssd"), metric)
        index.train(ds.get_train())
        index.add(ds.get_database())
        return index

    def test_recall(self):
        # same accuracy as IndexIVFRaBitQ with unquantized queries
        ds = datasets.SyntheticDataset(100, 2000, 5000, 50)
        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), 10)
        for nb_bits in 2, 4, 9:
            with self.subTest(nb_bits=nb_bits):
                index = self.build(ds, 32, 64, nb_bits)
                index_ivf = faiss.downcast_index(
                    faiss.extract_index_ivf(index))
                self.assertIsInstance(index_ivf, faiss.IndexIVFRaBitQSSD)
                index_ivf.nprobe = 16
                D, I = index.search(ds.get_queries(), 10)

                rot = faiss.HadamardRotation(ds.d, 12345, False)
                index_rq = faiss.IndexIVFRaBitQ(
                    faiss.IndexFlatL2(rot.d_out), rot.d_out, 32,
                    faiss.METRIC_L2, True, nb_bits)
                index_rq.qb = 0
                index_rq.nprobe = 16
                index_ref = faiss.IndexPreTransform(rot, index_rq)
                index_ref.train(ds.get_train())
                index_ref.add(ds.get_database())
                _, I_rq = index_ref.search(ds.get_queries(), 10)

                self.assertGreaterEqual(
                    recall_at_1(I, Iref), recall_at_1(I_rq, Iref) - 0.02)
                if nb_bits == 9:
                    self.assertGreaterEqual(recall_at_1(I, Iref), 0.97)
                    np.testing.assert_allclose(D[:, 0], Dref[:, 0], rtol=0.01)

    def test_inner_product(self):
        ds = datasets.SyntheticDataset(64, 2000, 5000, 50, metric="IP")
        Dref, Iref = faiss.knn(
            ds.get_queries(), ds.get_database(), 10,
            faiss.METRIC_INNER_PRODUCT)
        index = self.build(ds, 32, 64, 9, faiss.METRIC_INNER_PRODUCT)
        faiss.extract_index_ivf(index).nprobe = 32
        D, I = index.search(ds.get_queries(), 10)
        self.assertGreaterEqual(recall_at_1(I, Iref), 0.95)
        # similarities, in decreasing order
        self.assertTrue(np.all(D[:, :-1] >= D[:, 1:]))
        np.testing.assert_allclose(D[:, 0], Dref[:, 0], rtol=0.01)

    def test_same_results_as_none_level(self):
        ds = datasets.SyntheticDataset(100, 2000, 3000, 30)
        index = self.build(ds, 16, 64, 5)
        faiss.extract_index_ivf(index).nprobe = 8
        # with k = nb nothing is pruned, so the order in which the reads
        # complete does not matter: compare the distances of each vector
        k = ds.nb
        D, I = index.search(ds.get_queries(), k)
        with NoneSIMDLevel():
            D2, I2 = index.search(ds.get_queries(), k)
        for q in range(ds.nq):
            n = (I[q] >= 0).sum()
            self.assertEqual(n, (I2[q] >= 0).sum())
            dis = dict(zip(I[q, :n], D[q, :n]))
            dis2 = dict(zip(I2[q, :n], D2[q, :n]))
            self.assertEqual(dis.keys(), dis2.keys())
            ids = list(dis)
            np.testing.assert_allclose(
                [dis[i] for i in ids], [dis2[i] for i in ids],
                rtol=1e-5, atol=1e-4)

    def test_io(self):
        ds = datasets.SyntheticDataset(100, 2000, 3000, 30)
        index = self.build(ds, 16, 64, 5)
        faiss.extract_index_ivf(index).nprobe = 8
        # the reads complete in an order that varies between runs, which
        # can change what is pruned: with k = nb nothing is pruned
        k = ds.nb
        D, I = index.search(ds.get_queries(), k)

        fname = self.path("index.faiss")
        faiss.write_index(index, fname)
        for io_flags in 0, faiss.IO_FLAG_READ_ONLY:
            index2 = faiss.read_index(fname, io_flags)
            index_ivf2 = faiss.downcast_index(faiss.extract_index_ivf(index2))
            self.assertIsInstance(index_ivf2, faiss.IndexIVFRaBitQSSD)
            self.assertIsInstance(
                faiss.downcast_InvertedLists(index_ivf2.invlists),
                faiss.RaBitQSSDInvertedLists)
            self.assertEqual(index_ivf2.nprobe, 8)
            D2, I2 = index2.search(ds.get_queries(), k)
            np.testing.assert_array_equal(D, D2)
            # equal distances can come out in either order
            for q in range(ds.nq):
                self.assertEqual(
                    dict(zip(I[q], D[q])), dict(zip(I2[q], D2[q])))

    def test_search_parameters(self):
        ds = datasets.SyntheticDataset(100, 2000, 3000, 30)
        index = self.build(ds, 16, 64, 5)
        index_ivf = faiss.extract_index_ivf(index)
        xq = ds.get_queries()

        # a selector, applied after reading the ids from disk
        sel = faiss.IDSelectorRange(1000, 2000)
        params = faiss.SearchParametersPreTransform(
            index_params=faiss.IVFRaBitQSSDSearchParameters(
                nprobe=16, sel=sel))
        D, I = index.search(xq, 10, params=params)
        self.assertTrue(np.all((I >= 1000) & (I < 2000)))

        # the 8-bit look-up tables give nearly the same results
        params16 = faiss.IVFRaBitQSSDSearchParameters(
            nprobe=16, use_hacc=True)
        params8 = faiss.IVFRaBitQSSDSearchParameters(
            nprobe=16, use_hacc=False)
        rot = faiss.downcast_VectorTransform(index.chain.at(0))
        xr = rot.apply(xq)
        D16, I16 = index_ivf.search(xr, 10, params=params16)
        D8, I8 = index_ivf.search(xr, 10, params=params8)
        overlap = np.mean([
            len(set(I8[q]) & set(I16[q])) / 10 for q in range(len(xq))])
        self.assertGreaterEqual(overlap, 0.9)

        # a smaller drain_alpha does not read more pages
        stats = faiss.cvar.indexIVFRaBitQSSD_stats
        pages = []
        for drain_alpha in 1.0, 0.3:
            params = faiss.IVFRaBitQSSDSearchParameters(
                nprobe=16, drain_alpha=drain_alpha, qd_per_thread=1)
            stats.reset()
            index_ivf.search(xr, 10, params=params)
            pages.append(stats.n_pages_read)
        self.assertGreater(pages[0], 0)
        self.assertLessEqual(pages[1], pages[0])

        # invalid parameters
        for kwargs in dict(qd_per_thread=0), dict(drain_alpha=1.5):
            params = faiss.IVFRaBitQSSDSearchParameters(nprobe=16, **kwargs)
            self.assertRaises(
                RuntimeError, index_ivf.search, xr, 10, params=params)

    def test_index_factory(self):
        ds = datasets.SyntheticDataset(64, 2000, 2000, 10)
        index = faiss.index_factory(ds.d, "IVF16,RaBitQSSD5_64")
        self.assertIsInstance(index, faiss.IndexIVFRaBitQSSD)
        # created without inverted lists: attach them before adding
        self.assertIsNone(index.invlists)
        index.train(ds.get_train())
        self.assertRaises(RuntimeError, index.add, ds.get_database())
        invlists = faiss.RaBitQSSDInvertedLists(
            16, ds.d, 64, 4, self.path("factory.ssd"))
        index.replace_invlists(invlists, True)
        invlists.this.disown()
        index.add(ds.get_database())

        ref = faiss.IndexIVFRaBitQSSD(
            faiss.IndexFlatL2(ds.d), ds.d, 16, 64, 5, self.path("ref.ssd"))
        ref.train(ds.get_train())
        ref.add(ds.get_database())
        index.nprobe = ref.nprobe = 16
        # with k = nb nothing is pruned
        k = ds.nb
        D, I = index.search(ds.get_queries(), k)
        Dref, Iref = ref.search(ds.get_queries(), k)
        for q in range(ds.nq):
            self.assertEqual(
                dict(zip(I[q], D[q])), dict(zip(Iref[q], Dref[q])))

        # with the rotation: d = 150 is padded to 192, not 256
        index = faiss.index_factory(150, "HRk,IVF16,RaBitQSSD5_128")
        self.assertEqual(index.chain.at(0).d_out, 192)
        self.assertEqual(faiss.extract_index_ivf(index).d, 192)

    def test_unsupported(self):
        ds = datasets.SyntheticDataset(64, 1000, 1000, 10)
        index = self.build(ds, 8, 64, 2)
        index_ivf = faiss.extract_index_ivf(index)
        self.assertRaises(
            RuntimeError, index_ivf.remove_ids,
            faiss.IDSelectorRange(0, 10))
        self.assertRaisesRegex(
            RuntimeError, "reconstruct not supported", index_ivf.reconstruct, 0)
        self.assertRaises(
            RuntimeError, index.range_search, ds.get_queries(), 1.0)
        # nb_bits = 1 is rejected
        self.assertRaises(
            RuntimeError, faiss.IndexIVFRaBitQSSD, faiss.IndexFlatL2(64),
            64, 8, 64, 1, self.path("other.ssd"))
