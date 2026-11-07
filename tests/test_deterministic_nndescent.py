# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DeterministicNNDescent, a KNN graph build that does not depend
on the thread count."""

import unittest

import faiss
import numpy as np

from common_faiss_tests import for_all_simd_levels, get_dataset_2

DIM = 32
NB = 5000
K = 10


def graph_recall(dnd, xb, metric, k=K):
    """recall of the first k neighbors against the exact KNN graph"""
    _, gt = faiss.knn(xb, xb, k + 1, metric)
    I = faiss.vector_to_array(dnd.ids).reshape(len(xb), -1)[:, :k]
    hits = 0
    for i in range(len(xb)):
        hits += len(set(I[i]) & (set(gt[i]) - {i}))
    return hits / (len(xb) * k)


def build(xb, metric=faiss.METRIC_L2, nthreads=None, **params):
    storage = faiss.IndexFlat(xb.shape[1], metric)
    storage.add(xb)
    dnd = faiss.DeterministicNNDescent()
    dnd.K = 2 * K
    for name, value in params.items():
        setattr(dnd, name, value)
    if nthreads is not None:
        faiss.omp_set_num_threads(nthreads)
    dnd.build(storage)
    return dnd, storage


def split_by_counts(arrays, counts):
    """the items of each list for each destination process"""
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [
        [a[bounds[q] : bounds[q + 1]] for a in arrays]
        for q in range(len(counts))
    ]


def expected_remote_ids(rows, prev, lo, hi, received, R, begin, end):
    """the remote ids of the home rows [begin, end) of the process that owns
    [lo, hi), computed from the rows: an entry is new if it is not in the
    row of the previous round (sample = 0), and a home row keeps its R
    nearest reverse edges by (distance, source)"""
    t, src, ed, en = received
    home = rows[lo:hi]
    is_new = home >= 0
    if prev is not None:
        is_new &= ~(home[:, :, None] == prev[lo:hi, None, :]).any(axis=2)
    o = np.lexsort((src, ed, t))
    t, src, en = t[o] - lo, src[o], en[o]
    first = np.searchsorted(t, t)
    keep = np.arange(len(t)) - first < R
    t, src, en = t[keep], src[keep], en[keep]
    has_new = is_new.any(axis=1)
    has_new[t[en != 0]] = True
    sel = np.zeros(hi - lo, dtype=bool)
    sel[begin:end] = True
    sel &= has_new
    ids = np.concatenate([home[sel].ravel(), src[sel[t]]])
    return np.unique(ids[(ids >= 0) & ((ids < lo) | (ids >= hi))])


def sharded_rounds(xb, I, D, metric, K, R, nproc, sample, nrounds,
                   remote_vectors="float32", chunks=1,
                   check_remote_ids=False):
    """nrounds NNDescentShard rounds over nproc simulated processes; returns
    the rows, then the changed rows of each round.
    remote_vectors: "float32", "rounded" (float32 vectors rounded to
    float16), "float16" (join_fp16()), "sq8_decoded" (float32 vectors
    through 8-bit codes) or "sq8" (join_sq8()). chunks: join() calls per
    round, each on a range of the home rows (select_rows()).
    check_remote_ids: compare remote_ids with expected_remote_ids()."""
    n, d = xb.shape
    bounds = np.linspace(0, n, nproc + 1).astype("int64")
    shards = []
    for p in range(nproc):
        lo, hi = bounds[p], bounds[p + 1]
        s = faiss.NNDescentShard()
        s.R = R
        s.sample = sample
        s.set_rows(
            K, metric, d, p, nproc, faiss.swig_ptr(bounds),
            faiss.swig_ptr(xb[lo:hi]),
            faiss.swig_ptr(np.ascontiguousarray(I[lo:hi])),
            faiss.swig_ptr(np.ascontiguousarray(D[lo:hi])),
        )
        shards.append(s)

    def route(lists):
        # lists[i] = (counts, arrays) from a process; returns the arrays that
        # each process receives, in the order of lists
        parts = [split_by_counts(a, c) for c, a in lists]
        return [
            [np.concatenate([parts[i][q][j] for i in range(len(lists))])
             for j in range(len(lists[0][1]))]
            for q in range(nproc)
        ]

    vec = faiss.vector_to_array
    changed = []
    vmin = xb.min(axis=0)
    vdiff = np.maximum(xb.max(axis=0) - vmin, 1e-12).astype("float32")

    def sq8_encode(v):
        return np.clip(
            np.floor((v - vmin) / vdiff * 255), 0, 255
        ).astype("uint8")

    def sq8_decode(c):
        return (vmin + (c.astype("float32") + 0.5) / 255 * vdiff).astype(
            "float32"
        )

    prev = None
    for _ in range(nrounds):
        sent = []
        for s in shards:
            c, t, src, ed, en = (faiss.Int64Vector(), faiss.Int64Vector(),
                                 faiss.Int64Vector(), faiss.Float32Vector(),
                                 faiss.UInt8Vector())
            s.reverse_edges(c, t, src, ed, en)
            sent.append((vec(c), [vec(t), vec(src), vec(ed), vec(en)]))
        received = route(sent)
        for s, (t, src, ed, en) in zip(shards, received):
            s.set_reverse_edges(len(t), faiss.swig_ptr(t),
                                faiss.swig_ptr(src), faiss.swig_ptr(ed),
                                faiss.swig_ptr(en))
        rows = np.concatenate([vec(s.ids).reshape(-1, K) for s in shards])
        rdis = np.concatenate([vec(s.dis).reshape(-1, K) for s in shards])
        kth = np.where(rows[:, -1] >= 0, rdis[:, -1], np.inf).astype("float32")
        sent = []
        keep = []
        for p, s in enumerate(shards):
            for ch in range(chunks):
                nh = s.nhome()
                begin, end = nh * ch // chunks, nh * (ch + 1) // chunks
                if chunks > 1:
                    s.select_rows(begin, end)
                h = vec(s.remote_ids)
                if check_remote_ids:
                    np.testing.assert_array_equal(h, expected_remote_ids(
                        rows, prev, bounds[p], bounds[p + 1], received[p], R,
                        begin, end))
                hx = np.ascontiguousarray(xb[h])
                hk = np.ascontiguousarray(kth[h])
                if remote_vectors == "rounded":
                    hx = hx.astype("float16").astype("float32")
                if remote_vectors == "sq8_decoded":
                    hx = np.ascontiguousarray(sq8_decode(sq8_encode(hx)))
                keep.append((hx, hk))
                c, t, cand, od = (faiss.Int64Vector(), faiss.Int64Vector(),
                                  faiss.Int64Vector(), faiss.Float32Vector())
                if remote_vectors == "float16":
                    hx16 = hx.astype("float16").view("uint16")
                    keep.append(hx16)
                    s.join_fp16(faiss.swig_ptr(hx16), faiss.swig_ptr(hk),
                                c, t, cand, od)
                elif remote_vectors == "sq8":
                    codes = np.ascontiguousarray(sq8_encode(hx))
                    keep.append(codes)
                    s.join_sq8(faiss.swig_ptr(codes), faiss.swig_ptr(vmin),
                               faiss.swig_ptr(vdiff), faiss.swig_ptr(hk),
                               c, t, cand, od)
                else:
                    s.join(faiss.swig_ptr(hx), faiss.swig_ptr(hk),
                           c, t, cand, od)
                sent.append((vec(c), [vec(t), vec(cand), vec(od)]))
        changed.append(0)
        for s, (t, cand, od) in zip(shards, route(sent)):
            s.add_offers(len(t), faiss.swig_ptr(t), faiss.swig_ptr(cand),
                         faiss.swig_ptr(od))
            changed[-1] += s.end_round()
        prev = rows
    I2 = np.concatenate([vec(s.ids).reshape(-1, K) for s in shards])
    D2 = np.concatenate([vec(s.dis).reshape(-1, K) for s in shards])
    return I2, D2, changed


def init_rows(xb, metric=faiss.METRIC_L2, **params):
    """the storage, the DeterministicNNDescent and the rows after a random
    init()"""
    storage = faiss.IndexFlat(xb.shape[1], metric)
    storage.add(xb)
    dnd = faiss.DeterministicNNDescent()
    dnd.K = 2 * K
    dnd.R = 15
    dnd.ntrees = 0
    for name, value in params.items():
        setattr(dnd, name, value)
    dnd.init(storage)
    I = faiss.vector_to_array(dnd.ids).reshape(-1, dnd.K).astype("int64")
    D = faiss.vector_to_array(dnd.dis).reshape(-1, dnd.K)
    return storage, dnd, I, D


@for_all_simd_levels
class TestDeterministicNNDescent(unittest.TestCase):

    def test_recall(self):
        _, xb, _ = get_dataset_2(DIM, 0, NB, 0)
        for metric in faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT:
            for ntrees in 0, 2:
                with self.subTest(metric=metric, ntrees=ntrees):
                    dnd, _ = build(xb, metric, ntrees=ntrees)
                    self.assertGreater(graph_recall(dnd, xb, metric), 0.97)

    def test_deterministic(self):
        # with BLAS, some pairs get two distances (direct in init(), from the
        # norms in the joins) at d = 32, and none at d = 8
        self.addCleanup(faiss.omp_set_num_threads, faiss.omp_get_max_threads())
        for d in DIM, 8:
            with self.subTest(d=d):
                _, xb, _ = get_dataset_2(d, 0, NB, 0)
                a, _ = build(xb, nthreads=1, sample=8)
                b, _ = build(xb, nthreads=8, sample=8)
                for field in "ids", "dis":
                    np.testing.assert_array_equal(
                        faiss.vector_to_array(getattr(a, field)),
                        faiss.vector_to_array(getattr(b, field)),
                    )

    def test_rows_sorted_and_distinct(self):
        # 601 copies of one vector, more than a leaf: no pivot separates
        # them, so each tree splits their segment by position
        _, xb, _ = get_dataset_2(DIM, 0, 2000, 0)
        xb[1000:1600] = xb[0]
        dnd, _ = build(xb, max_rounds=3)
        I = faiss.vector_to_array(dnd.ids).reshape(len(xb), -1)
        D = faiss.vector_to_array(dnd.dis).reshape(len(xb), -1)
        self.assertTrue((I >= 0).all())
        self.assertTrue((I != np.arange(len(xb))[:, None]).all())
        self.assertTrue((np.diff(D, axis=1) >= 0).all())
        for row in I:
            self.assertEqual(len(set(row)), len(row))

    def test_distances(self):
        # the joins take the distances from the Gram kernel of the SIMD
        # level; d = 33 leaves a tail for each kernel
        _, xb, _ = get_dataset_2(33, 0, 2000, 0)
        x = xb.astype("float64")
        nrm = (x**2).sum(1)
        for metric in faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT:
            with self.subTest(metric=metric):
                dnd, _ = build(xb, metric, ntrees=0)
                I = faiss.vector_to_array(dnd.ids).reshape(len(xb), -1)
                D = faiss.vector_to_array(dnd.dis).reshape(len(xb), -1)
                ip = np.einsum("id,ijd->ij", x, x[I])
                if metric == faiss.METRIC_L2:
                    ref = nrm[:, None] + nrm[I] - 2 * ip
                else:
                    ref = -ip
                np.testing.assert_allclose(D, ref, atol=1e-5 * nrm.max())

    def test_small_n(self):
        # K = n - 1 and R > K, as for a small partition of a distributed build
        _, xb, _ = get_dataset_2(8, 0, 2, 0)
        dnd, _ = build(xb, K=1, R=50, max_rounds=3)
        np.testing.assert_array_equal(faiss.vector_to_array(dnd.ids), [1, 0])

    def test_short_rows_stop(self):
        # the padding of rows shorter than K is not new
        _, xb, _ = get_dataset_2(DIM, 0, 10, 0)
        storage, dnd, _, _ = init_rows(xb, K=64, sample=2, ntrees=8)
        changed = [dnd.round(storage) for _ in range(6)]
        # a round without changes leaves no new entries, so all later rounds
        # also change nothing
        self.assertEqual(changed[5], 0)

    def test_ndis(self):
        # ndis counts the distances that init() computes, before the cut
        _, xb, _ = get_dataset_2(DIM, 0, 1000, 0)
        _, dnd, _, _ = init_rows(xb, K=10)
        self.assertGreater(dnd.ndis, 1000 * 10)

    def test_inner_product_overflow(self):
        # the inner product of these finite vectors overflows to NaN
        xb = np.empty((300, 4), dtype="float32")
        xb[::2] = (-1e20, 1e20, 1e20, 1e20)
        xb[1::2] = (1e20, -1e20, 1e20, 1e20)
        dnd, _ = build(xb, faiss.METRIC_INNER_PRODUCT, ntrees=0, max_rounds=3)
        self.assertFalse(np.isnan(faiss.vector_to_array(dnd.dis)).any())

    def test_invalid_storage(self):
        dnd = faiss.DeterministicNNDescent()
        with self.assertRaisesRegex(RuntimeError, "d must be > 0"):
            dnd.init(faiss.IndexFlatL2(0))
        # a quantizer without a decoder
        storage = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_0bit)
        _, xb, _ = get_dataset_2(DIM, 0, 100, 0)
        storage.train(xb)
        storage.add(xb)
        with self.assertRaises(RuntimeError):
            dnd.init(storage)

    def test_sharded_rounds_equal_rounds(self):
        # random data has no equal distances, so the rows are identical. With
        # chunks = 3, join() calls on 3 row ranges give the rows of one call.
        _, xb, _ = get_dataset_2(DIM, 0, 3000, 0)
        nrounds = 4
        for metric, nproc, sample, chunks in (
            (faiss.METRIC_L2, 3, 0, 1),
            (faiss.METRIC_INNER_PRODUCT, 4, 6, 3),
        ):
            with self.subTest(metric=metric, sample=sample, chunks=chunks):
                storage, dnd, I, D = init_rows(xb, metric, sample=sample)
                I2, D2, changed = sharded_rounds(
                    xb, I, D, metric, 2 * K, 15, nproc, sample, nrounds,
                    chunks=chunks, check_remote_ids=sample == 0,
                )
                self.assertEqual(
                    changed, [dnd.round(storage) for _ in range(nrounds)]
                )
                np.testing.assert_array_equal(
                    I2, faiss.vector_to_array(dnd.ids).reshape(-1, 2 * K)
                )
                np.testing.assert_array_equal(
                    D2, faiss.vector_to_array(dnd.dis).reshape(-1, 2 * K)
                )

    def test_sharded_rounds_with_coded_remote_vectors(self):
        # join_fp16() and join_sq8() give the rows of join() with the remote
        # vectors rounded to float16 or decoded from the 8-bit codes
        _, xb, _ = get_dataset_2(DIM, 0, 3000, 0)
        _, _, I, D = init_rows(xb)
        for coded, decoded in ("float16", "rounded"), ("sq8", "sq8_decoded"):
            with self.subTest(remote_vectors=coded):
                results = [
                    sharded_rounds(xb, I, D, faiss.METRIC_L2, 2 * K, 15, 3, 0,
                                   3, remote_vectors=coding)
                    for coding in (decoded, coded)
                ]
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_array_equal(results[0][1], results[1][1])
                self.assertEqual(results[0][2], results[1][2])

    def test_shard_errors_are_catchable(self):
        # process 0 of 2 owns the ids 10 and 11; the remote id checks run in
        # OpenMP loops and must throw after the loop
        d = 4
        bounds = np.array([10, 12, 14], dtype="int64")
        x = np.random.RandomState(1).rand(2, d).astype("float32")
        D = np.ones((2, 1), dtype="float32")
        s = faiss.NNDescentShard()

        def set_rows(I):
            I = np.array(I, dtype="int64")
            s.set_rows(1, faiss.METRIC_L2, d, 0, 2, faiss.swig_ptr(bounds),
                       faiss.swig_ptr(x), faiss.swig_ptr(I),
                       faiss.swig_ptr(D))

        def set_reverse_edges(target, source):
            t = np.array(target, dtype="int64")
            src = np.array(source, dtype="int64")
            ed = np.ones(len(t), dtype="float32")
            en = np.ones(len(t), dtype="uint8")
            s.set_reverse_edges(len(t), faiss.swig_ptr(t),
                                faiss.swig_ptr(src), faiss.swig_ptr(ed),
                                faiss.swig_ptr(en))

        for bad in 9, 14:
            with self.assertRaisesRegex(RuntimeError, "row id out of range"):
                set_rows([[bad], [10]])
        set_rows([[12], [10]])
        ids = np.arange(14, dtype="int64")
        out = np.empty((2, 1), dtype="int64")
        with self.assertRaisesRegex(RuntimeError, "not in remote_ids"):
            s.original_ids(faiss.swig_ptr(ids), faiss.swig_ptr(ids),
                           faiss.swig_ptr(out))
        with self.assertRaisesRegex(RuntimeError, "source out of range"):
            set_reverse_edges([11], [14])
        set_reverse_edges([11], [13])
        np.testing.assert_array_equal(
            faiss.vector_to_array(s.remote_ids), [12, 13])
        # the remote ids of the rows only: 13 of the reverse edge is missing
        s.set_remote_ids_of_rows()
        c, t, cand, od = (faiss.Int64Vector(), faiss.Int64Vector(),
                          faiss.Int64Vector(), faiss.Float32Vector())
        with self.assertRaisesRegex(RuntimeError, "not in remote_ids"):
            s.join(None, None, c, t, cand, od)
        s.select_rows(0, 2)
        remote_x = np.random.RandomState(2).rand(2, d).astype("float32")
        kth = np.full(2, np.inf, dtype="float32")
        s.join(faiss.swig_ptr(remote_x), faiss.swig_ptr(kth), c, t, cand, od)
        t = np.array([10], dtype="int64")
        cand = np.array([14], dtype="int64")
        with self.assertRaisesRegex(RuntimeError, "candidate out of range"):
            s.add_offers(1, faiss.swig_ptr(t), faiss.swig_ptr(cand),
                         faiss.swig_ptr(D))

    def test_storage_types(self):
        _, xb, _ = get_dataset_2(DIM, 1000, NB, 0)
        # a subclass of IndexFlat with another code layout
        storage = faiss.IndexFlatL2Panorama(DIM, 4)
        storage.add(xb)
        dnd = faiss.DeterministicNNDescent()
        dnd.K = 2 * K
        dnd.build(storage)
        self.assertGreater(graph_recall(dnd, xb, faiss.METRIC_L2), 0.97)
        # a storage that cannot reconstruct raises in the calling thread
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(DIM), DIM, 16)
        ivf.train(xb)
        ivf.add(xb)
        with self.assertRaises(RuntimeError):
            dnd.init(ivf)
        # round() checks its storage too
        flat = faiss.IndexFlatL2(DIM)
        flat.add(xb)
        dnd.init(flat)
        with self.assertRaises(RuntimeError):
            dnd.round(ivf)

    def test_sq8_storage(self):
        _, xb, _ = get_dataset_2(DIM, 1000, NB, 0)
        storage = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit)
        storage.train(xb)
        storage.add(xb)
        dnd = faiss.DeterministicNNDescent()
        dnd.K = 2 * K
        dnd.build(storage)
        self.assertGreater(graph_recall(dnd, xb, faiss.METRIC_L2), 0.9)

    def test_index_nndescent(self):
        # the index forwards its settings to DeterministicNNDescent
        _, xb, _ = get_dataset_2(DIM, 0, NB, 0)
        index = faiss.IndexNNDescentFlat(DIM, 2 * K)
        index.nndescent.deterministic = True
        index.nndescent.iter = 5
        index.add(xb)
        storage = faiss.IndexFlatL2(DIM)
        storage.add(xb)
        dnd = faiss.DeterministicNNDescent()
        dnd.K = 2 * K
        dnd.max_rounds = index.nndescent.iter
        dnd.seed = index.nndescent.random_seed
        dnd.build(storage)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.nndescent.final_graph),
            faiss.vector_to_array(dnd.ids),
        )
