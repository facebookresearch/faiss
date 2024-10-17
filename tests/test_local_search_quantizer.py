# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the implementation of Local Search Quantizer
"""

import numpy as np

import faiss
import unittest

from faiss.contrib import datasets

faiss.omp_set_num_threads(4)

sp = faiss.swig_ptr


def construct_sparse_matrix(codes, K):
    n, M = codes.shape
    B = np.zeros((n, M * K), dtype=np.float32)
    for i in range(n):
        for j in range(M):
            code = codes[i, j]
            B[i, j * K + code] = 1
    return B


def update_codebooks_ref(x, codes, K, lambd):
    n, d = x.shape
    M = codes.shape[1]

    B = construct_sparse_matrix(codes, K)
    reg = np.identity(M * K) * float(lambd)
    reg = reg.astype(np.float32)

    # C = (B'B + lambd * I)^(-1)B'X
    bb = np.linalg.inv(B.T @ B + reg)
    C = bb @ B.T @ x
    C = C.reshape(M, K, d)

    return C


def compute_binary_terms_ref(codebooks):
    M, K, d = codebooks.shape

    codebooks_t = np.swapaxes(codebooks, 1, 2)  # [M, d, K]
    binaries = 2 * codebooks.dot(codebooks_t)   # [M, K, M, K]
    binaries = np.swapaxes(binaries, 1, 2)      # [M, M, K, K]

    return binaries


def compute_unary_terms_ref(codebooks, x):
    codebooks_t = np.swapaxes(codebooks, 1, 2)  # [M, d, K]
    unaries = -2 * x.dot(codebooks_t)  # [n, M, K]
    code_norms = np.sum(codebooks * codebooks, axis=2)  # [M, K]
    unaries += code_norms
    unaries = np.swapaxes(unaries, 0, 1)  # [M, n, K]

    return unaries


def icm_encode_step_ref(unaries, binaries, codes):
    M, n, K = unaries.shape

    for m in range(M):
        objs = unaries[m].copy()  # [n, K]

        for m2 in range(M):  # pair, m2 != m
            if m2 == m:
                continue

            for i in range(n):
                for code in range(K):
                    code2 = codes[i, m2]
                    objs[i, code] += binaries[m, m2, code, code2]

        codes[:, m] = np.argmin(objs, axis=1)

    return codes


def decode_ref(x, codebooks, codes):
    n, d = x.shape
    _, M = codes.shape
    decoded_x = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        for m in range(M):
            decoded_x[i] += codebooks[m, codes[i, m]]

    return decoded_x


def icm_encode_ref(x, codebooks, codes):
    n, d = x.shape
    M, K, d = codebooks.shape

    codes = codes.copy()
    for m in range(M):
        objs = np.zeros((n, K), dtype=np.float32)  # [n, K]
        for code in range(K):
            new_codes = codes.copy()
            new_codes[:, m] = code

            # decode x
            decoded_x = decode_ref(x, codebooks, new_codes)
            objs[:, code] = np.sum((x - decoded_x) ** 2, axis=1)

        codes[:, m] = np.argmin(objs, axis=1)

    return codes


class TestComponents(unittest.TestCase):

    def test_decode(self):
        """Test LSQ decode"""
        d = 16
        n = 500
        M = 4
        nbits = 6
        K = (1 << nbits)

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype(np.float32)
        codes = rs.randint(0, K, (n, M)).astype(np.int32)
        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.train(x)

        # decode x
        pack_codes = np.zeros((n, lsq.code_size)).astype(np.uint8)
        decoded_x = np.zeros((n, d)).astype(np.float32)
        lsq.pack_codes(n, sp(codes), sp(pack_codes))
        lsq.decode_c(sp(pack_codes), sp(decoded_x), n)

        # decode in Python
        codebooks = faiss.vector_float_to_array(lsq.codebooks)
        codebooks = codebooks.reshape(M, K, d).copy()
        decoded_x_ref = decode_ref(x, codebooks, codes)

        np.testing.assert_allclose(decoded_x, decoded_x_ref, rtol=1e-6)

    def test_update_codebooks(self):
        """Test codebooks updatation."""
        d = 16
        n = 500
        M = 4
        nbits = 6
        K = (1 << nbits)

        # set a larger value to make the updating process more stable
        lambd = 1e-2

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype(np.float32)
        codes = rs.randint(0, K, (n, M)).astype(np.int32)

        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.lambd = lambd
        lsq.train(x)  # just for allocating memory for codebooks

        codebooks = faiss.vector_float_to_array(lsq.codebooks)
        codebooks = codebooks.reshape(M, K, d).copy()

        lsq.update_codebooks(sp(x), sp(codes), n)
        new_codebooks = faiss.vector_float_to_array(lsq.codebooks)
        new_codebooks = new_codebooks.reshape(M, K, d).copy()

        ref_codebooks = update_codebooks_ref(x, codes, K, lambd)

        np.testing.assert_allclose(new_codebooks, ref_codebooks, atol=1e-3)

    def test_update_codebooks_with_double(self):
        """If the data is not zero-centering, it would be more accurate to
        use double-precision floating-point numbers."""
        ds = datasets.SyntheticDataset(16, 1000, 1000, 0)

        xt = ds.get_train() + 1000
        xb = ds.get_database() + 1000

        M = 4
        nbits = 4

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        err_double = eval_codec(lsq, xb)

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.update_codebooks_with_double = False
        lsq.train(xt)
        err_float = eval_codec(lsq, xb)

        # 6533.377 vs 25457.99
        self.assertLess(err_double, err_float)

    def test_compute_binary_terms(self):
        d = 16
        n = 500
        M = 4
        nbits = 6
        K = (1 << nbits)

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype(np.float32)
        binaries = np.zeros((M, M, K, K)).astype(np.float32)

        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.train(x)  # just for allocating memory for codebooks

        lsq.compute_binary_terms(sp(binaries))

        codebooks = faiss.vector_float_to_array(lsq.codebooks)
        codebooks = codebooks.reshape(M, K, d).copy()
        ref_binaries = compute_binary_terms_ref(codebooks)

        np.testing.assert_allclose(binaries, ref_binaries, atol=1e-4)

    def test_compute_unary_terms(self):
        d = 16
        n = 500
        M = 4
        nbits = 6
        K = (1 << nbits)

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype(np.float32)
        unaries = np.zeros((M, n, K)).astype(np.float32)

        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.train(x)  # just for allocating memory for codebooks

        lsq.compute_unary_terms(sp(x), sp(unaries), n)

        codebooks = faiss.vector_float_to_array(lsq.codebooks)
        codebooks = codebooks.reshape(M, K, d).copy()
        ref_unaries = compute_unary_terms_ref(codebooks, x)

        np.testing.assert_allclose(unaries, ref_unaries, atol=1e-4)

    def test_icm_encode_step(self):
        d = 16
        n = 500
        M = 4
        nbits = 6
        K = (1 << nbits)

        rs = np.random.RandomState(123)

        # randomly generate codes and unary terms
        codes = rs.randint(0, K, (n, M)).astype(np.int32)
        new_codes = codes.copy()
        unaries = rs.rand(M, n, K).astype(np.float32)

        # binary terms should be symmetric, because binary terms
        #  represent cached dot products between the code C1 in codebook M1
        #  and the code C2 in codebook M2.
        # so, binaries[M1, M2, C1, C2] == binaries[M2, M1, C2, C1]
        #
        # generate binary terms in a standard way that provides
        #  the needed symmetry
        codebooks = rs.rand(M, K, d).astype(np.float32)
        binaries = compute_binary_terms_ref(codebooks)
        binaries = np.ascontiguousarray(binaries)

        # do icm encoding given binary and unary terms
        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.icm_encode_step(
            sp(new_codes),
            sp(unaries),
            sp(binaries),
            n,
            1)

        # do icm encoding given binary and unary terms in Python
        ref_codes = icm_encode_step_ref(unaries, binaries, codes)
        np.testing.assert_array_equal(new_codes, ref_codes)

    def test_icm_encode(self):
        d = 16
        n = 500
        M = 4
        nbits = 4
        K = (1 << nbits)

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype(np.float32)

        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.train(x)  # just for allocating memory for codebooks

        # compute binary terms
        binaries = np.zeros((M, M, K, K)).astype(np.float32)
        lsq.compute_binary_terms(sp(binaries))

        # compute unary terms
        unaries = np.zeros((M, n, K)).astype(np.float32)
        lsq.compute_unary_terms(sp(x), sp(unaries), n)

        # randomly generate codes
        codes = rs.randint(0, K, (n, M)).astype(np.int32)
        new_codes = codes.copy()

        # do icm encoding given binary and unary terms
        lsq.icm_encode_step(
            sp(new_codes),
            sp(unaries),
            sp(binaries),
            n,
            1)

        # do icm encoding without pre-computed unary and bianry terms in Python
        codebooks = faiss.vector_float_to_array(lsq.codebooks)
        codebooks = codebooks.reshape(M, K, d).copy()
        ref_codes = icm_encode_ref(x, codebooks, codes)

        np.testing.assert_array_equal(new_codes, ref_codes)


def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum()


class TestLocalSearchQuantizer(unittest.TestCase):

    def test_training(self):
        """check that the error is in the same ballpark as PQ."""
        ds = datasets.SyntheticDataset(32, 3000, 3000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        M = 4
        nbits = 4

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        err_lsq = eval_codec(lsq, xb)

        pq = faiss.ProductQuantizer(ds.d, M, nbits)
        pq.train(xt)
        err_pq = eval_codec(pq, xb)

        self.assertLess(err_lsq, err_pq)


class TestIndexLocalSearchQuantizer(unittest.TestCase):

    def test_IndexLocalSearchQuantizer(self):
        ds = datasets.SyntheticDataset(32, 1000, 200, 100)
        gt = ds.get_groundtruth(10)

        ir = faiss.IndexLocalSearchQuantizer(ds.d, 4, 5)
        ir.train(ds.get_train())
        ir.add(ds.get_database())
        Dref, Iref = ir.search(ds.get_queries(), 10)
        inter_ref = faiss.eval_intersection(Iref, gt)

        # 467
        self.assertGreater(inter_ref, 460)

        AQ = faiss.AdditiveQuantizer
        ir2 = faiss.IndexLocalSearchQuantizer(
            ds.d, 4, 5, faiss.METRIC_L2, AQ.ST_norm_float)

        ir2.train(ds.get_train())  # just to set flags properly
        ir2.lsq.codebooks = ir.lsq.codebooks

        ir2.add(ds.get_database())
        D2, I2 = ir2.search(ds.get_queries(), 10)
        np.testing.assert_array_almost_equal(Dref, D2, decimal=5)
        self.assertLess((Iref != I2).sum(), Iref.size * 0.01)

        # test I/O
        ir3 = faiss.deserialize_index(faiss.serialize_index(ir))
        D3, I3 = ir3.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(Iref, I3)
        np.testing.assert_array_equal(Dref, D3)

    def test_coarse_quantizer(self):
        ds = datasets.SyntheticDataset(32, 5000, 1000, 100)
        gt = ds.get_groundtruth(10)

        quantizer = faiss.LocalSearchCoarseQuantizer(ds.d, 2, 4)
        quantizer.lsq.nperts
        quantizer.lsq.nperts = 2

        index = faiss.IndexIVFFlat(quantizer, ds.d, 256)
        index.quantizer_trains_alone = True

        index.train(ds.get_train())

        index.add(ds.get_database())
        index.nprobe = 4

        Dref, Iref = index.search(ds.get_queries(), 10)

        inter_ref = faiss.eval_intersection(Iref, gt)

        # 249
        self.assertGreater(inter_ref, 235)

    def test_factory(self):
        index = faiss.index_factory(20, "LSQ5x6_Nqint8")
        self.assertEqual(index.lsq.M, 5)
        self.assertEqual(index.lsq.K, 1 << 6)
        self.assertEqual(
            index.lsq.search_type,
            faiss.AdditiveQuantizer.ST_norm_qint8
        )

        index = faiss.index_factory(20, "LSQ5x6_Ncqint8")
        self.assertEqual(
            index.lsq.search_type,
            faiss.AdditiveQuantizer.ST_norm_cqint8
        )

        index = faiss.index_factory(20, "LSQ5x6_Ncqint4")
        self.assertEqual(
            index.lsq.search_type,
            faiss.AdditiveQuantizer.ST_norm_cqint4
        )



class TestIndexIVFLocalSearchQuantizer(unittest.TestCase):

    def test_factory(self):
        index = faiss.index_factory(20, "IVF1024,LSQ5x6_Nqint8")
        self.assertEqual(index.nlist, 1024)
        self.assertEqual(index.lsq.M, 5)
        self.assertEqual(index.lsq.K, 1 << 6)
        self.assertEqual(
            index.lsq.search_type,
            faiss.AdditiveQuantizer.ST_norm_qint8
        )

        index = faiss.index_factory(20, "IVF1024,LSQ5x6_Ncqint8")
        self.assertEqual(
            index.lsq.search_type,
            faiss.AdditiveQuantizer.ST_norm_cqint8
        )

    def eval_index_accuracy(self, factory_key):
        # just do a single test, most search functions are already stress
        # tested in test_residual_quantizer.py
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)
        index = faiss.index_factory(ds.d, factory_key)

        index.train(ds.get_train())
        index.add(ds.get_database())

        inters = []
        for nprobe in 1, 2, 5, 10, 20, 50:
            index.nprobe = nprobe
            D, I = index.search(ds.get_queries(), 10)
            inter = faiss.eval_intersection(I, ds.get_groundtruth(10))
            inters.append(inter)

        inters = np.array(inters)
        # in fact the results should be the same for the decoding and the
        # reconstructing versions
        self.assertTrue(np.all(inters[1:] >= inters[:-1]))

        # do a little I/O test
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        D2, I2 = index2.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

    def test_index_accuracy_reconstruct(self):
        self.eval_index_accuracy("IVF100,LSQ4x5")

    def test_index_accuracy_reconstruct_LUT(self):
        self.eval_index_accuracy("IVF100,LSQ4x5_Nfloat")

    def test_index_accuracy_cqint(self):
        self.eval_index_accuracy("IVF100,LSQ4x5_Ncqint8")

    def test_deterministic(self):
        ds = datasets.SyntheticDataset(d=16, nt=1000, nb=10005, nq=1000)
        index = faiss.index_factory(ds.d, "IVF100,LSQ4x4_Nqint8")

        k = 1
        xq = ds.get_queries()
        xb = ds.get_database()
        xt = ds.get_train()

        index.train(xt)
        index.add(xb)
        D, I = index.search(xq, k=k)

        index.reset()
        index.train(xt)
        index.add(xb)
        D2, I2 = index.search(xq, k=k)

        np.testing.assert_array_equal(I, I2)


class TestProductLocalSearchQuantizer(unittest.TestCase):

    def test_codec(self):
        """check that the error is in the same ballpark as PQ."""
        ds = datasets.SyntheticDataset(64, 3000, 3000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        nsplits = 2
        Msub = 2
        nbits = 4

        plsq = faiss.ProductLocalSearchQuantizer(ds.d, nsplits, Msub, nbits)
        plsq.train(xt)
        err_plsq = eval_codec(plsq, xb)

        pq = faiss.ProductQuantizer(ds.d, nsplits * Msub, nbits)
        pq.train(xt)
        err_pq = eval_codec(pq, xb)

        self.assertLess(err_plsq, err_pq)

    def test_with_lsq(self):
        """compare with LSQ when nsplits = 1"""
        ds = datasets.SyntheticDataset(32, 3000, 3000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        M = 4
        nbits = 4

        plsq = faiss.ProductLocalSearchQuantizer(ds.d, 1, M, nbits)
        plsq.train(xt)
        err_plsq = eval_codec(plsq, xb)

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        err_lsq = eval_codec(lsq, xb)

        self.assertEqual(err_plsq, err_lsq)

    def test_lut(self):
        """test compute_LUT function"""
        ds = datasets.SyntheticDataset(16, 1000, 0, 100)

        xt = ds.get_train()
        xq = ds.get_queries()

        nsplits = 2
        Msub = 2
        nbits = 4
        nq, d = xq.shape
        dsub = d // nsplits

        plsq = faiss.ProductLocalSearchQuantizer(ds.d, nsplits, Msub, nbits)
        plsq.train(xt)

        subcodebook_size = Msub * (1 << nbits)
        codebook_size = nsplits * subcodebook_size
        lut = np.zeros((nq, codebook_size), dtype=np.float32)
        plsq.compute_LUT(nq, sp(xq), sp(lut))

        codebooks = faiss.vector_to_array(plsq.codebooks)
        codebooks = codebooks.reshape(nsplits, subcodebook_size, dsub)
        xq = xq.reshape(nq, nsplits, dsub)
        lut_ref = np.zeros((nq, nsplits, subcodebook_size), dtype=np.float32)
        for i in range(nsplits):
            lut_ref[:, i] = xq[:, i] @ codebooks[i].T
        lut_ref = lut_ref.reshape(nq, codebook_size)

        np.testing.assert_allclose(lut, lut_ref, rtol=5e-04)


class TestIndexProductLocalSearchQuantizer(unittest.TestCase):

    def test_accuracy1(self):
        """check that the error is in the same ballpark as LSQ."""
        recall1 = self.eval_index_accuracy("PLSQ4x3x5_Nqint8")
        recall2 = self.eval_index_accuracy("LSQ12x5_Nqint8")
        self.assertGreaterEqual(recall1, recall2)  # 622 vs 551

    def test_accuracy2(self):
        """when nsplits = 1, PLSQ should be almost the same as LSQ"""
        recall1 = self.eval_index_accuracy("PLSQ1x3x5_Nqint8")
        recall2 = self.eval_index_accuracy("LSQ3x5_Nqint8")
        diff = abs(recall1 - recall2)  # 273 vs 275 in OSX
        self.assertGreaterEqual(5, diff)

    def eval_index_accuracy(self, index_key):
        ds = datasets.SyntheticDataset(32, 1000, 1000, 100)
        index = faiss.index_factory(ds.d, index_key)

        index.train(ds.get_train())
        index.add(ds.get_database())
        D, I = index.search(ds.get_queries(), 10)
        inter = faiss.eval_intersection(I, ds.get_groundtruth(10))

        # do a little I/O test
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        D2, I2 = index2.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

        return inter

    def test_factory(self):
        AQ = faiss.AdditiveQuantizer
        ns, Msub, nbits = 2, 4, 8
        index = faiss.index_factory(64, f"PLSQ{ns}x{Msub}x{nbits}_Nqint8")
        assert isinstance(index, faiss.IndexProductLocalSearchQuantizer)
        self.assertEqual(index.plsq.nsplits, ns)
        self.assertEqual(index.plsq.subquantizer(0).M, Msub)
        self.assertEqual(index.plsq.subquantizer(0).nbits.at(0), nbits)
        self.assertEqual(index.plsq.search_type, AQ.ST_norm_qint8)

        code_size = (ns * Msub * nbits + 7) // 8 + 1
        self.assertEqual(index.plsq.code_size, code_size)


class TestIndexIVFProductLocalSearchQuantizer(unittest.TestCase):

    def eval_index_accuracy(self, factory_key):
        ds = datasets.SyntheticDataset(32, 1000, 1000, 100)
        index = faiss.index_factory(ds.d, factory_key)

        index.train(ds.get_train())
        index.add(ds.get_database())

        inters = []
        for nprobe in 1, 2, 4, 8, 16:
            index.nprobe = nprobe
            D, I = index.search(ds.get_queries(), 10)
            inter = faiss.eval_intersection(I, ds.get_groundtruth(10))
            inters.append(inter)

        inters = np.array(inters)
        self.assertTrue(np.all(inters[1:] >= inters[:-1]))

        # do a little I/O test
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        D2, I2 = index2.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

        return inter

    def test_index_accuracy(self):
        self.eval_index_accuracy("IVF32,PLSQ2x2x5_Nqint8")

    def test_index_accuracy2(self):
        """check that the error is in the same ballpark as LSQ."""
        inter1 = self.eval_index_accuracy("IVF32,PLSQ2x2x5_Nqint8")
        inter2 = self.eval_index_accuracy("IVF32,LSQ4x5_Nqint8")
        self.assertGreaterEqual(inter1 * 1.1, inter2)

    def test_factory(self):
        AQ = faiss.AdditiveQuantizer
        ns, Msub, nbits = 2, 4, 8
        index = faiss.index_factory(64, f"IVF32,PLSQ{ns}x{Msub}x{nbits}_Nqint8")
        assert isinstance(index, faiss.IndexIVFProductLocalSearchQuantizer)
        self.assertEqual(index.nlist, 32)
        self.assertEqual(index.plsq.nsplits, ns)
        self.assertEqual(index.plsq.subquantizer(0).M, Msub)
        self.assertEqual(index.plsq.subquantizer(0).nbits.at(0), nbits)
        self.assertEqual(index.plsq.search_type, AQ.ST_norm_qint8)

        code_size = (ns * Msub * nbits + 7) // 8 + 1
        self.assertEqual(index.plsq.code_size, code_size)
