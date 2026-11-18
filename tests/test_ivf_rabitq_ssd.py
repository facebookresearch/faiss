# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import shutil
import tempfile
import threading
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
        quantizer, rot.d_out, nlist, mem_dim, nb_bits, metric)
    index_ivf.attach_file(filename)
    return faiss.IndexPreTransform(rot, index_ivf)


def recall_at_1(I, gt):
    return (I[:, 0] == gt[:, 0]).mean()


def make_data(n, d, ncenters, seed):
    """n vectors around ncenters Gaussian centers"""
    rs = np.random.RandomState(seed)
    centers = 2 * rs.randn(ncenters, d)
    return (centers[np.arange(n) % ncenters] + rs.randn(n, d)).astype(
        "float32")


def make_queries(xb, nq, seed):
    """queries near some of the vectors of xb"""
    rs = np.random.RandomState(seed)
    xq = xb[np.arange(nq) * 37 % len(xb)]
    return (xq + 0.3 * rs.randn(*xq.shape)).astype("float32")


def same_results(D1, I1, D2, I2, rtol=0):
    """The second search finds the ids of the first, with the same distances
    up to rtol. The order of equal distances can differ between searches."""
    for q in range(len(I1)):
        dis = dict(zip(I1[q], D1[q]))
        for i, v in zip(I2[q], D2[q]):
            if i not in dis or abs(dis[i] - v) > rtol * max(1, abs(dis[i])):
                return False
    return True


def list_ids(il, list_no):
    """the ids of a list, read from the file"""
    n = il.list_size(list_no)
    if n == 0:
        return np.zeros(0, dtype="int64")
    ptr = il.get_ids(list_no)
    ids = faiss.rev_swig_ptr(ptr, n).copy()
    il.release_ids(list_no, ptr)
    return ids


def count_aio_contexts():
    """libaio contexts of the process: each maps its ring as "[aio]" """
    with open("/proc/self/maps") as f:
        return sum("[aio]" in line for line in f)


def lut_step(q, mem_dim, use_hacc):
    """Step of the look-up table of the in-memory scan: all the entries, the
    sums of q over subsets of 4 consecutive dimensions, are quantized with one
    offset and one step."""
    subsets = (np.arange(16)[:, None] >> np.arange(4)) & 1
    v = q[:mem_dim].reshape(-1, 4) @ subsets.T
    return (v.max() - v.min()) / (65535 if use_hacc else 255)


def reference_estimates(index, xq, xb, assign, use_hacc):
    """Reference estimates of the distances (similarities for IP) between the
    queries and the database vectors, computed in double from the
    RaBitQuantizer codes:
        <q - c, r> ~ |r|^2 <xb, q - c> / <xb, r>
    Also returns bounds on their difference with the index's float
    computation, which quantizes <x_u, q> over [0, mem_dim) with the look-up
    table. Both have shape (nq, nb)."""
    d, mem_dim = index.d, index.mem_dim
    nb_bits = index.rabitq.nb_bits
    ex_bits = nb_bits - 1
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    scales = faiss.vector_to_array(index.centroid_scales)
    if scales.size > 0:
        centroids *= scales[:, None]
    rq = faiss.RaBitQuantizer(d, index.metric_type, nb_bits)
    # the sign bits, SignBitFactorsWithError, then the extra bits
    ex_offset = d // 8 + 12
    assert rq.code_size == ex_offset + d * ex_bits // 8 + 8
    codes = np.empty((len(xb), rq.code_size), dtype="uint8")
    for l in np.unique(assign):
        rows = np.where(assign == l)[0]
        x = np.ascontiguousarray(xb[rows])
        out = np.empty((len(rows), rq.code_size), dtype="uint8")
        rq.compute_codes_core(
            faiss.swig_ptr(x), faiss.swig_ptr(out), len(rows),
            faiss.swig_ptr(centroids[l]))
        codes[rows] = out
    signs = np.unpackbits(codes[:, :d // 8], axis=1, bitorder="little")
    ex = np.unpackbits(
        codes[:, ex_offset:ex_offset + d * ex_bits // 8], axis=1,
        bitorder="little")
    ex = ex.reshape(len(xb), d, ex_bits) @ (1 << np.arange(ex_bits))
    xbq = signs * float(1 << ex_bits) + ex - ((1 << ex_bits) - 0.5)
    c = centroids[assign].astype(float)
    r = xb.astype(float) - c
    ip_xb_r = (xbq * r).sum(1)
    r2 = (r * r).sum(1)
    ip_r_c = (r * c).sum(1)
    xb_abs_max = np.abs(xbq).max(1)
    nonzero = ip_xb_r != 0
    ip_xb_r_safe = np.where(nonzero, ip_xb_r, 1)
    scale = np.where(nonzero, r2 / np.abs(ip_xb_r_safe), 0)
    is_l2 = index.metric_type == faiss.METRIC_L2
    k = 2 if is_l2 else 1
    est = np.empty((len(xq), len(xb)))
    tol = np.empty((len(xq), len(xb)))
    for i, q in enumerate(xq.astype(float)):
        qc = q - c
        ip_xb_qc = (xbq * qc).sum(1)
        ip_q_c = c @ q
        qc2 = (qc * qc).sum(1)
        ip_qc_r = np.where(nonzero, r2 * ip_xb_qc / ip_xb_r_safe, 0)
        # each look-up table entry is rounded to the nearest step
        lut_error = mem_dim // 4 * lut_step(q, mem_dim, use_hacc) / 2
        # float rounding: relative to the magnitude of the terms
        magnitude = (
            qc2 + r2 + np.abs(ip_qc_r) + np.abs(ip_q_c) + np.abs(ip_r_c)
            + scale * xb_abs_max * np.sqrt(d)
            * np.sqrt(qc2 + ip_q_c ** 2 / (qc2 + 1e-30) + 1))
        tol[i] = k * scale * (1 << ex_bits) * lut_error + 2e-4 * magnitude
        est[i] = qc2 + r2 - 2 * ip_qc_r if is_l2 else ip_q_c + ip_r_c + ip_qc_r
    return est, tol


@unittest.skipUnless(
    hasattr(faiss, "IndexIVFRaBitQSSD"), "IndexIVFRaBitQSSD not built")
class SSDTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.TemporaryDirectory()
        try:
            fd = os.open(self.path("probe"),
                         os.O_CREAT | os.O_RDWR | os.O_DIRECT)
        except OSError:
            self.tmpdir.cleanup()
            self.skipTest("O_DIRECT not supported in " + self.tmpdir.name)
        os.close(fd)

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

    def build_flat(self, d=128, mem_dim=64, nb_bits=5, metric=faiss.METRIC_L2,
                   quantizer_metric=faiss.METRIC_L2, use_hacc=True, nlist=8,
                   nb=1000, seed=123, name="flat.ssd"):
        """An index without rotation on clustered data, and its vectors."""
        xb = make_data(nb, d, 3 * nlist, seed)
        index = faiss.IndexIVFRaBitQSSD(
            faiss.IndexFlat(d, quantizer_metric), d, nlist, mem_dim, nb_bits,
            metric)
        index.attach_file(self.path(name))
        index.use_hacc = use_hacc
        index.train(xb)
        index.add(xb)
        return index, xb


@for_all_simd_levels
class TestIVFRaBitQSSDKernels(SSDTestCase):
    """The tests whose results depend on the SIMD kernels."""

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

    def check_estimates(self, use_hacc=True, **kwargs):
        """Search with k = nb: no vector is pruned, all are re-ranked, and
        their distances must match the reference estimates."""
        index, xb = self.build_flat(use_hacc=use_hacc, **kwargs)
        nb, nlist = index.ntotal, index.nlist
        is_ip = index.metric_type == faiss.METRIC_INNER_PRODUCT
        # only inner-product indexes have scales
        self.assertEqual(index.centroid_scales.size(), nlist if is_ip else 0)
        assign = index.quantizer.assign(xb, 1).ravel()
        xq = make_data(4, index.d, 5, 456)
        params = faiss.IVFRaBitQSSDSearchParameters(
            nprobe=nlist, use_hacc=use_hacc)
        D, I = index.search(xq, nb, params=params)
        est, tol = reference_estimates(index, xq, xb, assign, use_hacc)
        for q in range(len(xq)):
            self.assertEqual(sorted(I[q]), list(range(nb)))
            order = D[q, :-1] >= D[q, 1:] if is_ip else D[q, :-1] <= D[q, 1:]
            self.assertTrue(np.all(order))
            bad = np.abs(D[q] - est[q, I[q]]) > tol[q, I[q]]
            if bad.any():
                j = np.argmax(bad)
                self.fail(
                    f"{bad.sum()} bad estimates, query {q} id {I[q, j]}: "
                    f"{D[q, j]} != {est[q, I[q, j]]} "
                    f"(tolerance {tol[q, I[q, j]]})")

    def test_estimates(self):
        L2, IP = faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT
        configs = [{"nb_bits": nb_bits, "mem_dim": mem_dim}
                   for nb_bits in (2, 5, 9) for mem_dim in (64, 128)]
        configs += [{"metric": IP, "quantizer_metric": IP, "nb_bits": nb_bits,
                     "mem_dim": mem_dim}
                    for nb_bits in (2, 4, 9) for mem_dim in (64, 128)]
        configs += [
            # the coarse distances come from a quantizer with the other
            # metric
            {"quantizer_metric": IP},
            {"metric": IP, "quantizer_metric": L2},
            # 8-bit look-up tables
            {"use_hacc": False},
            {"metric": IP, "quantizer_metric": IP, "use_hacc": False},
        ]
        # several 8-bit scan segments, records larger than a page
        shapes = {"nb": 300, "nlist": 4}
        configs += [
            {"d": 1536, "mem_dim": 1280, "nb_bits": 3, **shapes},
            {"d": 1536, "mem_dim": 1280, "nb_bits": 3, "use_hacc": False,
             **shapes},
            # 512 sub-quantizers: their 8-bit sums exceed 2^16 unless split
            {"d": 2048, "mem_dim": 2048, "nb_bits": 2, "use_hacc": False,
             **shapes},
            {"d": 4096, "mem_dim": 256, "nb_bits": 9, **shapes},
            {"d": 192, "mem_dim": 192, "nb_bits": 2, **shapes},
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                self.check_estimates(**cfg)

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


class TestIVFRaBitQSSD(SSDTestCase):

    def test_layout(self):
        il = faiss.RaBitQSSDInvertedLists(4, 192, 64, 3, self.path("a.ssd"))
        self.assertEqual(il.batch_bytes, 32 * 64 // 8 + 32 * 3 * 4)
        # 16 sign bytes, 72 bytes of extra bits, 2 factors, id
        self.assertEqual(il.record_size, 16 + 72 + 8 + 8)
        self.assertEqual(il.page_size, 4096)
        self.assertEqual(il.records_per_page, 4096 // 104)

        # a record larger than a page
        il = faiss.RaBitQSSDInvertedLists(4, 4096, 2048, 8, self.path("b.ssd"))
        self.assertEqual(il.record_size, 256 + 4096 + 8 + 8)
        self.assertEqual(il.page_size, 8192)
        self.assertEqual(il.records_per_page, 1)

        # d must be a multiple of 64, mem_dim a multiple of 64 in [64, d],
        # and ex_bits in [1, 8]
        for d, mem_dim, ex_bits in [
                (100, 64, 1), (120, 64, 1), (128, 0, 1), (128, 32, 1),
                (128, 96, 1), (128, 192, 1), (128, 64, 0), (128, 64, 9)]:
            self.assertRaises(
                RuntimeError, faiss.RaBitQSSDInvertedLists, 4, d, mem_dim,
                ex_bits, self.path("c.ssd"))

    def test_bound_decides_reads(self):
        # The in-memory lower bound decides whether a page is read. A list
        # holds one full page, then its farthest vector from the centroid
        # alone on a second page. With one read in flight and k = 1, the
        # second page is considered after the first is re-ranked, and is
        # read iff the lower bound of that vector does not exceed the best
        # distance on the first page. With inner product, the center of the
        # list is 7 times the centroid of the quantizer.
        d, mem_dim, nb_bits = 128, 64, 5
        stats = faiss.cvar.indexIVFRaBitQSSD_stats
        for metric in faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT:
            is_l2 = metric == faiss.METRIC_L2
            scale = np.float32(1 if is_l2 else 7)
            centroid = np.full(d, np.float32(0.5) / scale, dtype="float32")
            quantizer = faiss.IndexFlat(d, metric)
            quantizer.add(centroid[None])
            index = faiss.IndexIVFRaBitQSSD(
                quantizer, d, 1, mem_dim, nb_bits, metric)
            index.attach_file(self.path(f"bound{metric}.ssd"))
            if not is_l2:
                faiss.copy_array_to_vector(
                    np.array([scale], dtype="float32"), index.centroid_scales)
                index.is_trained = True
            center = centroid * scale
            il = index.get_ssd_invlists()
            nb = il.records_per_page + 1
            xb = make_data(nb, d, 1, 321)
            index.add(xb)
            self.assertEqual(il.list_pages(0), 2)

            # the vector alone on page 1 is the farthest from the centroid
            dis = ((xb - center) ** 2).sum(1) if is_l2 else -(xb @ center)
            last = int(np.argmax(dis))
            self.assertEqual(il.get_single_id(0, nb - 1), last)
            others = np.delete(np.arange(nb), last)

            # the in-memory lower bound of the last vector uses its sign bits
            rq = faiss.RaBitQuantizer(d, metric, nb_bits)
            code = np.empty(rq.code_size, dtype="uint8")
            o = xb[last].copy()
            rq.compute_codes_core(
                faiss.swig_ptr(o), faiss.swig_ptr(code), 1,
                faiss.swig_ptr(center))
            x1 = np.unpackbits(code[:d // 8], bitorder="little") - 0.5
            c = center.astype(float)
            r = o.astype(float) - c
            k = 2 if is_l2 else 1
            # the search minimizes -<q, o> for IP
            sign = 1 if is_l2 else -1

            rs = np.random.RandomState(5)
            params = faiss.IVFRaBitQSSDSearchParameters(
                nprobe=1, qd_per_thread=1)
            n_read = n_pruned = 0
            for _ in range(200):
                q = (0.5 + 0.8 * rs.randn(1, d)).astype("float32")
                # best distance on page 0
                est, tol = reference_estimates(
                    index, q, xb, np.zeros(nb, dtype="int64"), True)
                j = others[np.argmin(sign * est[0, others])]
                distk = sign * est[0, j]
                qd = q[0].astype(float)
                qc = qd - c
                ip_x1_qc = x1[:mem_dim] @ qc[:mem_dim]
                ip_qc_r = (r @ r) * (d / mem_dim) * ip_x1_qc / (x1 @ r)
                if is_l2:
                    est_last = qc @ qc + r @ r - k * ip_qc_r
                else:
                    est_last = -(qd @ c + r @ c + ip_qc_r)
                low = est_last - k * 1.9 * np.sqrt(
                    (r @ r) * (qc @ qc) / mem_dim)
                # margin for the look-up table and float rounding
                if abs(low - distk) < 2 * tol[0, j] + 1e-3 * abs(distk):
                    continue
                expect_read = low <= distk
                stats.reset()
                index.search(q, 1, params=params)
                self.assertEqual(
                    stats.n_pages_read, 2 if expect_read else 1,
                    f"metric {metric} low {low} distk {distk}")
                if expect_read:
                    n_read += 1
                else:
                    n_pruned += 1
            # both cases occur
            self.assertGreater(n_read, 5)
            self.assertGreater(n_pruned, 5)

    def test_pruning(self):
        # pruning keeps the results of an exhaustive re-ranking and reads
        # few pages
        stats = faiss.cvar.indexIVFRaBitQSSD_stats
        nq, k = 50, 10
        for metric in faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT:
            for use_hacc in False, True:
                with self.subTest(metric=metric, use_hacc=use_hacc):
                    index, xb = self.build_flat(
                        nb_bits=9, metric=metric, quantizer_metric=metric,
                        nb=5000, nlist=16, use_hacc=use_hacc)
                    xq = make_queries(xb, nq, 789)
                    params = faiss.IVFRaBitQSSDSearchParameters(
                        nprobe=16, use_hacc=use_hacc)
                    # exhaustive: with k = nb nothing is pruned
                    _, I_all = index.search(xq, index.ntotal, params=params)
                    il = index.get_ssd_invlists()
                    total_pages = sum(il.list_pages(l) for l in range(16))
                    stats.reset()
                    _, I = index.search(xq, k, params=params)
                    # the bound holds with high probability: the same
                    # results, from a fraction of the pages
                    recall = np.mean([
                        len(set(I[q]) & set(I_all[q, :k])) / k
                        for q in range(nq)])
                    self.assertGreaterEqual(recall, 0.99)
                    self.assertLess(stats.n_pages_read, nq * total_pages / 4)
                    self.assertGreater(stats.n_pages_read, 0)

    def test_retrain_centroid_norms(self):
        index, _ = self.build_flat(
            quantizer_metric=faiss.METRIC_INNER_PRODUCT, name="a.ssd")
        fresh, xb = self.build_flat(
            quantizer_metric=faiss.METRIC_INNER_PRODUCT, seed=456,
            name="b.ssd")
        index.reset()
        self.assertEqual(index.centroid_norms.size(), 0)
        index.quantizer.reset()
        index.train(xb)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.centroid_norms),
            faiss.vector_to_array(fresh.centroid_norms))
        index.add(xb)
        xq = make_queries(xb, 3, 789)
        index.nprobe = fresh.nprobe = 8
        D, I = index.search(xq, len(xb))
        Dref, Iref = fresh.search(xq, len(xb))
        np.testing.assert_array_equal(D, Dref)
        self.assertTrue(same_results(Dref, Iref, D, I))

    def test_ip_list_scales(self):
        # train() of an inner-product index: needed even with a filled
        # quantizer, sets centroid_scales and keeps the centroids
        d, nlist, nb = 128, 4, 2000
        xb = make_data(nb, d, 3, 77)
        # centroids of norm 2: a vector of each of the 3 clusters, and the
        # opposite of the first one, which no vector is assigned to
        centroids = xb[:nlist].copy()
        centroids[3] = -centroids[0]
        faiss.normalize_L2(centroids)
        centroids *= 2
        quantizer = faiss.IndexFlatIP(d)
        quantizer.add(centroids)
        index = faiss.IndexIVFRaBitQSSD(
            quantizer, d, nlist, 64, 5, faiss.METRIC_INNER_PRODUCT)
        index.attach_file(self.path("index.ssd"))
        self.assertFalse(index.is_trained)
        self.assertRaises(RuntimeError, index.add, xb)

        assign = quantizer.assign(xb, 1).ravel()
        index.train(xb)
        self.assertTrue(index.is_trained)
        np.testing.assert_array_equal(
            quantizer.reconstruct_n(0, nlist), centroids)
        # scale = <mean of the list, centroid> / |centroid|^2
        counts = np.bincount(assign, minlength=nlist)
        self.assertTrue(np.all(counts[:3] > 0))
        self.assertEqual(counts[3], 0)
        ref = np.array([
            (xb[assign == l].astype(float) @ centroids[l]).mean() / 4
            for l in range(3)])
        scales = faiss.vector_to_array(index.centroid_scales)
        np.testing.assert_allclose(scales[:3], ref, rtol=1e-4)
        # a list without training vectors gets the mean scale of the others
        np.testing.assert_allclose(scales[3], ref.mean(), rtol=1e-4)

        # the scales are saved with the index, and kept by train() on a
        # non-empty index and by reset()
        index.add(xb)
        fname = self.path("index.faiss")
        faiss.write_index(index, fname)
        index2 = faiss.read_index(fname)
        self.assertTrue(index2.is_trained)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index2.centroid_scales), scales)
        index.train(xb[:nb // 2])
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.centroid_scales), scales)
        index.reset()
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.centroid_scales), scales)

    def test_inner_product_unnormalized(self):
        # Vectors of norm ~500. The centroids of spherical k-means have
        # norm 1: with the scales of train(), the codes are relative to the
        # centroids scaled to the means of the lists.
        d, nlist, nb, nq = 256, 4, 12000, 20
        rs = np.random.RandomState(123)
        center = 32 * rs.randn(d).astype("float32")
        xb = center + rs.randn(nb, d).astype("float32")
        xq = xb[np.arange(nq) * 37 % nb]
        xq = xq + 0.3 * rs.randn(nq, d).astype("float32")
        _, Iref = faiss.knn(xq, xb, 10, faiss.METRIC_INNER_PRODUCT)

        index = make_index(d, nlist, d, 9, self.path("index.ssd"),
                           faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        index_ivf.nprobe = nlist
        scales = faiss.vector_to_array(index_ivf.centroid_scales)
        self.assertEqual(scales.size, nlist)
        stats = faiss.cvar.indexIVFRaBitQSSD_stats
        stats.reset()
        _, I = index.search(xq, 10)
        self.assertGreaterEqual(recall_at_1(I, Iref), 0.95)
        pages = stats.n_pages_read

        # the same index centered on the norm-1 centroids reads more pages
        unit = make_index(d, nlist, d, 9, self.path("unit.ssd"),
                          faiss.METRIC_INNER_PRODUCT)
        unit.train(xb)
        unit_ivf = faiss.downcast_index(faiss.extract_index_ivf(unit))
        faiss.copy_array_to_vector(
            np.ones(nlist, dtype="float32"), unit_ivf.centroid_scales)
        unit.add(xb)
        unit_ivf.nprobe = nlist
        stats.reset()
        unit.search(xq, 10)
        self.assertLess(pages * 3, stats.n_pages_read)

        # A quantizer filled by the caller: the index is not trained before
        # train(), which keeps the centroids and sets the same scales.
        rot = faiss.downcast_VectorTransform(index.chain.at(0))
        xr = rot.apply(xb)
        centroids = index_ivf.quantizer.reconstruct_n(0, nlist)
        quantizer = faiss.IndexFlatIP(d)
        quantizer.add(centroids)
        index2 = faiss.IndexIVFRaBitQSSD(
            quantizer, d, nlist, d, 9, faiss.METRIC_INNER_PRODUCT)
        index2.attach_file(self.path("index2.ssd"))
        self.assertFalse(index2.is_trained)
        self.assertRaises(RuntimeError, index2.add, xr[:10])
        index2.train(xr)
        self.assertTrue(index2.is_trained)
        np.testing.assert_array_equal(
            quantizer.reconstruct_n(0, nlist), centroids)
        np.testing.assert_allclose(
            faiss.vector_to_array(index2.centroid_scales), scales, rtol=1e-5)

        # With an L2 quantizer on the same centroids, <q, c> is not derived
        # from the coarse distance: the same distances. With k = nb nothing
        # is pruned.
        quantizer_l2 = faiss.IndexFlatL2(d)
        quantizer_l2.add(centroids)
        index_l2 = faiss.IndexIVFRaBitQSSD(
            quantizer_l2, d, nlist, d, 9, faiss.METRIC_INNER_PRODUCT)
        index_l2.attach_file(self.path("index_l2.ssd"))
        faiss.copy_array_to_vector(scales, index_l2.centroid_scales)
        index_l2.is_trained = True
        assign = index_ivf.quantizer.assign(xr, 1).ravel()
        index_l2.add_core(
            nb, faiss.swig_ptr(xr), None, faiss.swig_ptr(assign))
        index_l2.nprobe = nlist
        xqr = rot.apply(xq[:3])
        D1, I1 = index_ivf.search(xqr, nb)
        D2, I2 = index_l2.search(xqr, nb)
        self.assertTrue(same_results(D1, I1, D2, I2, 1e-5))

    def test_recall_large_d(self):
        # d = 768 with the sign bits of 256 dimensions in memory
        rs = np.random.RandomState(7)
        d, nb, nq = 768, 4000, 20
        xb = rs.randn(nb, d).astype("float32") + 2 * rs.randn(d).astype(
            "float32")
        xq = xb[:nq] + 0.1 * rs.randn(nq, d).astype("float32")
        for metric in faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT:
            with self.subTest(metric=metric):
                _, Iref = faiss.knn(xq, xb, 1, metric)
                index = make_index(d, 8, 256, 9,
                                   self.path("large_d.ssd"), metric)
                index.train(xb)
                index.add(xb)
                faiss.extract_index_ivf(index).nprobe = 8
                _, I = index.search(xq, 10)
                self.assertGreaterEqual(recall_at_1(I, Iref), 0.95)

    def test_hnsw_quantizer(self):
        xb = make_data(3000, 128, 60, 123)
        index = faiss.IndexIVFRaBitQSSD(
            faiss.IndexHNSWFlat(128, 16), 128, 32, 64, 5)
        index.attach_file(self.path("index.ssd"))
        index.train(xb)
        index.add(xb)
        index.nprobe = 8
        _, I = index.search(xb[:20], 1)
        self.assertGreaterEqual((I[:, 0] == np.arange(20)).sum(), 18)

    def test_fewer_results_than_k(self):
        for metric in faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT:
            with self.subTest(metric=metric):
                index, _ = self.build_flat(
                    metric=metric, quantizer_metric=metric, nb=50, nlist=4)
                index.nprobe = 4
                D, I = index.search(make_data(1, 128, 1, 456), 80)
                self.assertTrue(np.all(I[0, :50] >= 0))
                np.testing.assert_array_equal(I[0, 50:], -1)
                finfo = np.finfo("float32")
                empty = finfo.max if metric == faiss.METRIC_L2 else finfo.min
                np.testing.assert_array_equal(D[0, 50:], empty)

    def test_incremental_add(self):
        # adding in several batches gives the same distances as a single add
        index, xb = self.build_flat(name="a.ssd")
        index2 = faiss.IndexIVFRaBitQSSD(faiss.IndexFlatL2(128), 128, 8, 64, 5)
        index2.attach_file(self.path("b.ssd"))
        index2.train(xb)
        for i0 in range(0, len(xb), 70):
            index2.add(xb[i0:i0 + 70])
        self.assertEqual(index2.ntotal, len(xb))
        # several extents: consecutive pages of a list not adjacent in the
        # file
        il = index2.get_ssd_invlists()
        self.assertTrue(any(
            il.page_offset(l, p + 1) != il.page_offset(l, p) + il.page_size
            for l in range(il.nlist) for p in range(il.list_pages(l) - 1)))
        xq = make_data(1, 128, 1, 456)
        index.nprobe = index2.nprobe = 8
        D1, I1 = index.search(xq, len(xb))
        D2, I2 = index2.search(xq, len(xb))
        self.assertTrue(same_results(D1, I1, D2, I2))

        index.reset()
        self.assertEqual(index.ntotal, 0)
        self.assertEqual(index.get_ssd_invlists().total_pages, 0)
        index.add_with_ids(xb, np.arange(1000, 1000 + len(xb)))
        _, I = index.search(xb[7:8], 1)
        self.assertEqual(I[0, 0], 1007)

    def test_search_matches_preassigned(self):
        # search() runs the coarse quantizer in the search threads: the same
        # results as the coarse quantizer on all the queries, then
        # search_preassigned
        for metric in faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT:
            with self.subTest(metric=metric):
                index, xb = self.build_flat(
                    metric=metric, quantizer_metric=metric)
                # the coarse distances of a batch of queries can come from
                # BLAS, those of one query do not: they can differ in the
                # last bits
                nq = 30
                xq = make_queries(xb, nq, 789)
                index.nprobe = 8
                ivf_stats = faiss.cvar.indexIVF_stats
                ivf_stats.reset()
                # with k = nb nothing is pruned
                D, I = index.search(xq, len(xb))
                self.assertEqual(ivf_stats.nq, nq)
                self.assertEqual(ivf_stats.nlist, nq * 8)
                Dq, Iq = index.quantizer.search(xq, 8)
                D2, I2 = index.search_preassigned(xq, len(xb), Iq, Dq)
                self.assertTrue(same_results(D, I, D2, I2, 1e-5))

    def test_io(self):
        ds = datasets.SyntheticDataset(100, 2000, 3000, 30)
        index = self.build(ds, 16, 64, 5)
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        index_ivf.nprobe = 8
        index_ivf.drain_alpha = 0.8
        index_ivf.qd_per_thread = 8
        # the reads complete in an order that varies between runs, which
        # can change what is pruned: with k = nb nothing is pruned
        k = ds.nb
        D, I = index.search(ds.get_queries(), k)

        fname = self.path("index.faiss")
        faiss.write_index(index, fname)
        for io_flags in faiss.IO_FLAG_SKIP_IVF_DATA, faiss.IO_FLAG_MMAP:
            self.assertRaises(RuntimeError, faiss.read_index, fname, io_flags)
        for io_flags in 0, faiss.IO_FLAG_READ_ONLY:
            index2 = faiss.read_index(fname, io_flags)
            index_ivf2 = faiss.downcast_index(faiss.extract_index_ivf(index2))
            self.assertIsInstance(index_ivf2, faiss.IndexIVFRaBitQSSD)
            self.assertIsInstance(
                faiss.downcast_InvertedLists(index_ivf2.invlists),
                faiss.RaBitQSSDInvertedLists)
            self.assertEqual(index_ivf2.nprobe, 8)
            self.assertAlmostEqual(index_ivf2.drain_alpha, 0.8, places=6)
            self.assertEqual(index_ivf2.qd_per_thread, 8)
            D2, I2 = index2.search(ds.get_queries(), k)
            np.testing.assert_array_equal(D, D2)
            # equal distances can come out in either order
            self.assertTrue(same_results(D, I, D2, I2))
            if io_flags == faiss.IO_FLAG_READ_ONLY:
                # rejected before changing the lists, not by the descriptor
                for method, args in [(index2.add, (ds.get_database()[:10],)),
                                     (index2.reset, ())]:
                    self.assertRaisesRegex(
                        RuntimeError, "opened read-only", method, *args)
                self.assertEqual(index2.ntotal, ds.nb)
                D2, _ = index2.search(ds.get_queries(), k)
                np.testing.assert_array_equal(D, D2)
            del index2, index_ivf2

        # IO_FLAG_ONDISK_SAME_DIR: the data file is the one next to the
        # index file. It is moved, so that no file is left at the stored
        # path; the open descriptor of index follows it.
        copy_dir = self.path("copy")
        os.mkdir(copy_dir)
        shutil.copyfile(self.path("index.faiss"),
                        os.path.join(copy_dir, "index.faiss"))
        os.rename(self.path("index.ssd"), os.path.join(copy_dir, "index.ssd"))
        index3 = faiss.read_index(
            os.path.join(copy_dir, "index.faiss"),
            faiss.IO_FLAG_ONDISK_SAME_DIR)
        il3 = faiss.downcast_index(faiss.extract_index_ivf(index3)).invlists
        self.assertEqual(
            faiss.downcast_InvertedLists(il3).filename,
            os.path.join(copy_dir, "index.ssd"))
        D3, _ = index3.search(ds.get_queries(), k)
        np.testing.assert_array_equal(D, D3)
        del index3, il3
        os.rename(os.path.join(copy_dir, "index.ssd"), self.path("index.ssd"))

        # the ids are read back from the file
        il = index_ivf.get_ssd_invlists()
        n = 0
        for l in range(il.nlist):
            ids = list_ids(il, l)
            for j, i in enumerate(ids):
                self.assertEqual(il.get_single_id(l, j), i)
            n += len(ids)
        self.assertEqual(n, ds.nb)

        # resetting and refilling the file invalidates the saved index
        index.reset()
        index.add(ds.get_database())
        self.assertRaises(RuntimeError, faiss.read_index, fname)
        faiss.write_index(index, fname)
        # a new index on the same file invalidates the saved index
        other = faiss.IndexIVFRaBitQSSD(
            faiss.IndexFlatL2(index_ivf.d), index_ivf.d, 16, 64, 5)
        other.attach_file(self.path("index.ssd"))
        self.assertRaises(RuntimeError, faiss.read_index, fname)

    def test_search_parameters(self):
        ds = datasets.SyntheticDataset(100, 2000, 3000, 30)
        index = self.build(ds, 16, 64, 5)
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
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

        # with one read in flight, pages are still waiting after the last
        # list, and a smaller drain_alpha reads fewer of them
        stats = faiss.cvar.indexIVFRaBitQSSD_stats
        pages = []
        for drain_alpha in 1.0, 0.3:
            params = faiss.IVFRaBitQSSDSearchParameters(
                nprobe=16, drain_alpha=drain_alpha, qd_per_thread=1)
            stats.reset()
            index_ivf.search(xr, 10, params=params)
            pages.append(stats.n_pages_read)
        self.assertGreater(pages[0], 0)
        self.assertLess(pages[1], pages[0])

        # store_pairs: the labels are (list, offset) pairs. With k = nb
        # nothing is pruned.
        k = ds.nb
        Dq, Iq = index_ivf.quantizer.search(xr, 16)

        def search_preassigned(store_pairs, params):
            D = np.empty((len(xr), k), dtype="float32")
            I = np.empty((len(xr), k), dtype="int64")
            index_ivf.search_preassigned_c(
                len(xr), faiss.swig_ptr(xr), k, faiss.swig_ptr(Iq),
                faiss.swig_ptr(Dq), faiss.swig_ptr(D), faiss.swig_ptr(I),
                store_pairs, params)
            return D, I

        params = faiss.IVFRaBitQSSDSearchParameters(nprobe=16)
        Dref, Iref = search_preassigned(False, params)
        D2, I2 = search_preassigned(True, params)
        np.testing.assert_array_equal(D2, Dref)
        il = index_ivf.get_ssd_invlists()
        ids2 = np.full(I2.shape, -1)
        for l in range(il.nlist):
            mask = (I2 >= 0) & (I2 >> 32 == l)
            ids2[mask] = list_ids(il, l)[I2[mask] & 0xffffffff]
        self.assertTrue(same_results(Dref, Iref, D2, ids2))
        # a selector does not apply to (list, offset) pairs
        params = faiss.IVFRaBitQSSDSearchParameters(nprobe=16, sel=sel)
        self.assertRaisesRegex(
            RuntimeError, "cannot be combined", search_preassigned, True,
            params)

        # invalid parameters
        for kwargs in ({"qd_per_thread": 0}, {"drain_alpha": 1.5},
                       {"max_codes": 100}, {"max_lists_num": 2},
                       {"ensure_topk_full": True},
                       {"max_empty_result_buckets": 1}):
            params = faiss.IVFRaBitQSSDSearchParameters(nprobe=16, **kwargs)
            self.assertRaises(
                RuntimeError, index_ivf.search, xr, 10, params=params)
        index_ivf.parallel_mode = index_ivf.PARALLEL_MODE_NO_HEAP_INIT
        self.assertRaisesRegex(
            RuntimeError, "NO_HEAP_INIT", index_ivf.search, xr, 10)
        index_ivf.parallel_mode = 0

    def test_io_contexts_kept(self):
        # the libaio contexts of the searches are kept between searches and
        # destroyed with the index
        gc.collect()
        n0 = count_aio_contexts()
        index, xb = self.build_flat()
        self.assertEqual(count_aio_contexts(), n0)
        xq = make_queries(xb, 50, 789)
        index.search(xq, 10)
        n1 = count_aio_contexts()
        self.assertGreater(n1, n0)
        self.assertLessEqual(n1, n0 + faiss.omp_get_max_threads())
        for _ in range(3):
            index.search(xq, 10)
            self.assertEqual(count_aio_contexts(), n1)
        del index
        self.assertEqual(count_aio_contexts(), n0)

    def test_concurrent_searches(self):
        # searches from several threads at once, on two indexes, give the
        # same results as one search at a time
        indexes = [self.build_flat(seed=123, name="a.ssd")[0],
                   self.build_flat(seed=456, name="b.ssd")[0]]
        xq = make_data(10, 128, 5, 789)
        refs = []
        for index in indexes:
            index.nprobe = 4
            # with k = nb nothing is pruned
            refs.append(index.search(xq, index.ntotal))
        errors = []
        compared = []

        def run(t):
            faiss.omp_set_num_threads(2)
            for rep in range(4):
                j = (t + rep) % 2
                try:
                    D, I = indexes[j].search(xq, indexes[j].ntotal)
                except Exception as e:
                    errors.append((t, rep, repr(e)))
                    continue
                if not same_results(*refs[j], D, I):
                    errors.append((t, rep, "different results"))
                compared.append((t, rep))

        threads = [threading.Thread(target=run, args=(t,)) for t in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(compared), 16)

    def test_exception_during_search(self):
        # a search that fails with reads in flight leaves its libaio
        # contexts ready for the next searches
        index, xb = self.build_flat(nb=3000)
        xq = make_data(20, 128, 5, 789)
        index.nprobe = 8
        # with k = nb nothing is pruned
        Dref, Iref = index.search(xq, len(xb))
        n_calls = [0]

        def is_member(i):
            n_calls[0] += 1
            if n_calls[0] > 50:
                raise ValueError("too many calls")
            return True

        # the selector is called when re-ranking, while other reads are in
        # flight
        sel = faiss.PyCallbackIDSelector(is_member)
        params = faiss.IVFRaBitQSSDSearchParameters(
            nprobe=8, sel=sel, qd_per_thread=64)
        for _ in range(3):
            n_calls[0] = 0
            # ValueError if the calling thread runs a failing callback,
            # RuntimeError if only OpenMP worker threads do
            self.assertRaises(
                (ValueError, RuntimeError), index.search, xq, len(xb),
                params=params)
            # it failed in the callback
            self.assertGreater(n_calls[0], 50)
            D, I = index.search(xq, len(xb))
            self.assertTrue(same_results(Dref, Iref, D, I))

    def test_index_factory(self):
        ds = datasets.SyntheticDataset(64, 2000, 2000, 10)
        index = faiss.index_factory(ds.d, "IVF16,RaBitQSSD5_64")
        self.assertIsInstance(index, faiss.IndexIVFRaBitQSSD)
        # created without inverted lists: attach a file before adding
        self.assertIsNone(index.invlists)
        index.reset()
        index.train(ds.get_train())
        self.assertRaises(RuntimeError, index.add, ds.get_database())
        # it can be written and read back without inverted lists
        fname = self.path("factory.faiss")
        faiss.write_index(index, fname)
        index = faiss.read_index(fname)
        self.assertTrue(index.is_trained)
        self.assertIsNone(index.invlists)
        index.attach_file(self.path("factory.ssd"))
        index.add(ds.get_database())
        # a new file would drop the vectors
        self.assertRaisesRegex(
            RuntimeError, "needs an empty index", index.attach_file,
            self.path("other.ssd"))

        ref = faiss.IndexIVFRaBitQSSD(faiss.IndexFlatL2(ds.d), ds.d, 16, 64, 5)
        ref.attach_file(self.path("ref.ssd"))
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

        # nb_bits = 1, and mem_dim not a multiple of 64
        for bad in "IVF8,RaBitQSSD1_64", "IVF8,RaBitQSSD5_100":
            self.assertRaises(RuntimeError, faiss.index_factory, 128, bad)

    def test_unsupported(self):
        ds = datasets.SyntheticDataset(64, 1000, 1000, 10)
        index = self.build(ds, 8, 64, 2)
        index_ivf = faiss.extract_index_ivf(index)
        self.assertRaises(
            RuntimeError, index_ivf.remove_ids,
            faiss.IDSelectorRange(0, 10))
        self.assertRaisesRegex(
            RuntimeError, "reconstruct not supported", index_ivf.reconstruct, 0)
        xr = faiss.downcast_VectorTransform(index.chain.at(0)).apply(
            ds.get_queries())
        stats = faiss.cvar.indexIVFRaBitQSSD_stats
        stats.reset()
        for method, args in [
                (index.range_search, (ds.get_queries(), 1.0)),
                (index_ivf.reconstruct_n, (0, 2)),
                (index_ivf.search_and_reconstruct, (xr, 10)),
                (index_ivf.search_and_return_codes, (xr, 10)),
                (index_ivf.sa_encode, (xr,)),
                (index_ivf.update_vectors, (np.arange(1), xr[:1]))]:
            self.assertRaises(RuntimeError, method, *args)
        # rejected before reading anything
        self.assertEqual(stats.n_pages_read, 0)
        self.assertEqual(stats.n_reranked, 0)
        self.assertRaises(RuntimeError, index_ivf.merge_from, index_ivf, 0)
        # moving codes between inverted lists
        other = faiss.IndexIVFFlat(faiss.IndexFlatL2(64), 64, 8)
        self.assertRaisesRegex(
            RuntimeError, "copy_subset_to not supported",
            index_ivf.copy_subset_to, other, 0, 0, 10)
        # nb_bits = 1 is rejected
        self.assertRaises(
            RuntimeError, faiss.IndexIVFRaBitQSSD, faiss.IndexFlatL2(64),
            64, 8, 64, 1)


@unittest.skipUnless(
    hasattr(faiss, "IndexIVFRaBitQSSD"), "IndexIVFRaBitQSSD not built")
class TestIVFRaBitQSSDCompileOptions(unittest.TestCase):

    def test_compile_options(self):
        self.assertIn("DISK_IVF", faiss.get_compile_options())
