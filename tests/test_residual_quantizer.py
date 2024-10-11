# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest

from faiss.contrib import datasets
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks

###########################################################
# Reference implementation of encoding with beam search
###########################################################

faiss.omp_set_num_threads(4)


def pairwise_distances(a, b):
    anorms = (a ** 2).sum(1)
    bnorms = (b ** 2).sum(1)
    return anorms.reshape(-1, 1) + bnorms - 2 * a @ b.T


def beam_search_encode_step_ref(cent, residuals, codes, L):
    """ Reference beam search implementation
    encodes a residual table.
    """
    K, d = cent.shape
    n, beam_size, d2 = residuals.shape
    assert d == d2
    n2, beam_size_2, m = codes.shape
    assert n2 == n and beam_size_2 == beam_size

    # compute all possible new residuals
    cent_distances = pairwise_distances(residuals.reshape(n * beam_size, d), cent)
    cent_distances = cent_distances.reshape(n, beam_size, K)

    # TODO write in vector form

    if beam_size * K <= L:
        # then keep all the results
        new_beam_size = beam_size * K
        new_codes = np.zeros((n, beam_size, K, m + 1), dtype=int)
        new_residuals = np.zeros((n, beam_size, K, d), dtype='float32')
        for i in range(n):
            new_codes[i, :, :, :-1] = codes[i]
            new_codes[i, :, :, -1] = np.arange(K)
            new_residuals[i] = residuals[i].reshape(1, d) - cent.reshape(K, d)
        new_codes = new_codes.reshape(n, new_beam_size, m + 1)
        new_residuals = new_residuals.reshape(n, new_beam_size, d)
        new_distances = cent_distances.reshape(n, new_beam_size)
    else:
        # keep top-L results
        new_beam_size = L
        new_codes = np.zeros((n, L, m + 1), dtype=int)
        new_residuals = np.zeros((n, L, d), dtype='float32')
        new_distances = np.zeros((n, L), dtype='float32')
        for i in range(n):
            cd = cent_distances[i].ravel()
            jl = np.argsort(cd)[:L]    # TODO argpartition
            js = jl // K     # input beam index
            ls = jl % K      # centroid index
            new_codes[i, :, :-1] = codes[i, js, :]
            new_codes[i, :, -1] = ls
            new_residuals[i, :, :] = residuals[i, js, :] - cent[ls, :]
            new_distances[i, :] = cd[jl]

    return new_codes, new_residuals, new_distances


def beam_search_encode_step(cent, residuals, codes, L, assign_index=None):
    """ Wrapper of the C++ function with the same interface """
    K, d = cent.shape
    n, beam_size, d2 = residuals.shape
    assert d == d2
    n2, beam_size_2, m = codes.shape
    assert n2 == n and beam_size_2 == beam_size

    assert L <= beam_size * K

    new_codes = np.zeros((n, L, m + 1), dtype='int32')
    new_residuals = np.zeros((n, L, d), dtype='float32')
    new_distances = np.zeros((n, L), dtype='float32')

    sp = faiss.swig_ptr
    codes = np.ascontiguousarray(codes, dtype='int32')
    faiss.beam_search_encode_step(
        d, K, sp(cent), n, beam_size, sp(residuals),
        m, sp(codes), L, sp(new_codes), sp(new_residuals), sp(new_distances),
        assign_index
    )

    return new_codes, new_residuals, new_distances


def beam_search_encoding_ref(centroids, x, L):
    """
    Perform encoding of vectors x with a beam search of size L
    """
    n, d = x.shape
    beam_size = 1
    codes = np.zeros((n, beam_size, 0), dtype=int)
    residuals = x.reshape((n, beam_size, d))
    distances = (x ** 2).sum(1).reshape(n, beam_size)

    for cent in centroids:
        codes, residuals, distances = beam_search_encode_step_ref(
            cent, residuals, codes, L)

    return (codes, residuals, distances)


###########################################################
# Unittests for basic routines
###########################################################


class TestBeamSearch(unittest.TestCase):

    def do_test(self, K=70, L=10, use_assign_index=False):
        """ compare C++ beam search with reference python implementation """
        d = 32
        n = 500
        L = 10 # beam size

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype('float32')

        cent = rs.rand(K, d).astype('float32')

        # first quant step --> input beam size is 1
        codes = np.zeros((n, 1, 0), dtype=int)
        residuals = x.reshape(n, 1, d)

        assign_index = faiss.IndexFlatL2(d) if use_assign_index else None

        ref_codes, ref_residuals, ref_distances = beam_search_encode_step_ref(
            cent, residuals, codes, L
        )

        new_codes, new_residuals, new_distances = beam_search_encode_step(
            cent, residuals, codes, L, assign_index
        )

        np.testing.assert_array_equal(new_codes, ref_codes)
        np.testing.assert_array_equal(new_residuals, ref_residuals)
        np.testing.assert_allclose(new_distances, ref_distances, rtol=1e-5)

        # second quant step:
        K = 50
        cent = rs.rand(K, d).astype('float32')

        codes, residuals = ref_codes, ref_residuals

        ref_codes, ref_residuals, ref_distances = beam_search_encode_step_ref(
            cent, residuals, codes, L
        )

        new_codes, new_residuals, new_distances = beam_search_encode_step(
            cent, residuals, codes, L
        )

        np.testing.assert_array_equal(new_codes, ref_codes)
        np.testing.assert_array_equal(new_residuals, ref_residuals)
        np.testing.assert_allclose(new_distances, ref_distances, rtol=1e-5)

    def test_beam_search(self):
        self.do_test()

    def test_beam_search_assign_index(self):
        self.do_test(use_assign_index=True)

    def test_small_beam(self):
        self.do_test(L=1)

    def test_small_beam_2(self):
        self.do_test(L=2)


def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum()


class TestResidualQuantizer(unittest.TestCase):

    def test_training(self):
        """check that the error is in the same ballpark as PQ """
        ds = datasets.SyntheticDataset(32, 3000, 1000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        rq = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq.verbose
        rq.verbose = True
        #
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.cp.verbose
        # rq.cp.verbose = True
        rq.train(xt)
        err_rq = eval_codec(rq, xb)

        pq = faiss.ProductQuantizer(ds.d, 4, 6)
        pq.train(xt)
        err_pq = eval_codec(pq, xb)

        # in practice RQ is often better than PQ but it does not the case here, so just check
        # that we are within some factor.
        self.assertLess(err_rq, err_pq * 1.2)

    def test_beam_size(self):
        """ check that a larger beam gives a lower error """
        ds = datasets.SyntheticDataset(32, 3000, 1000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        rq0 = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq0.train_type = faiss.ResidualQuantizer.Train_default
        rq0.max_beam_size = 2
        rq0.train(xt)
        err_rq0 = eval_codec(rq0, xb)

        rq1 = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq1.train_type = faiss.ResidualQuantizer.Train_default
        rq1.max_beam_size = 10
        rq1.train(xt)
        err_rq1 = eval_codec(rq1, xb)

        self.assertLess(err_rq1, err_rq0)

    def test_training_with_limited_mem(self):
        """ make sure a different batch size gives the same result"""
        ds = datasets.SyntheticDataset(32, 3000, 1000, 0)

        xt = ds.get_train()

        rq0 = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq0.train_type = faiss.ResidualQuantizer.Train_default
        rq0.max_beam_size = 5
        # rq0.verbose = True
        rq0.train(xt)
        cb0 = get_additive_quantizer_codebooks(rq0)

        rq1 = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq1.train_type = faiss.ResidualQuantizer.Train_default
        rq1.max_beam_size = 5
        rq1.max_mem_distances
        rq1.max_mem_distances = 3000 * ds.d * 4 * 3
        # rq1.verbose = True
        rq1.train(xt)
        cb1 = get_additive_quantizer_codebooks(rq1)

        for c0, c1 in zip(cb0, cb1):
            self.assertTrue(np.all(c0 == c1))

    def test_clipping(self):
        """ verify that a clipped residual quantizer gives the same
        code prefix + suffix as the full RQ """
        ds = datasets.SyntheticDataset(32, 1000, 100, 0)

        rq = faiss.ResidualQuantizer(ds.d, 5, 4)
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.max_beam_size = 5
        rq.train(ds.get_train())

        rq.max_beam_size = 1   # is not he same for a large beam size
        codes = rq.compute_codes(ds.get_database())

        rq2 = faiss.ResidualQuantizer(ds.d, 2, 4)
        rq2.initialize_from(rq)
        self.assertEqual(rq2.M, 2)
        # verify that the beginning of the codes are the same
        codes2 = rq2.compute_codes(ds.get_database())

        rq3 = faiss.ResidualQuantizer(ds.d, 3, 4)
        rq3.initialize_from(rq, 2)
        self.assertEqual(rq3.M, 3)
        codes3 = rq3.compute_codes(ds.get_database() - rq2.decode(codes2))

        # verify that prefixes are the same
        for i in range(ds.nb):
            br = faiss.BitstringReader(faiss.swig_ptr(codes[i]), rq.code_size)
            br2 = faiss.BitstringReader(faiss.swig_ptr(codes2[i]), rq2.code_size)
            self.assertEqual(br.read(rq2.tot_bits), br2.read(rq2.tot_bits))
            br3 = faiss.BitstringReader(faiss.swig_ptr(codes3[i]), rq3.code_size)
            self.assertEqual(br.read(rq3.tot_bits), br3.read(rq3.tot_bits))


###########################################################
# Test index, index factory sa_encode / sa_decode
###########################################################

def unpack_codes(rq, packed_codes):
    nbits = faiss.vector_to_array(rq.nbits)
    if np.all(nbits == 8):
        return packed_codes.astype("uint32")
    nbits = [int(x) for x in nbits]
    nb = len(nbits)
    n, code_size = packed_codes.shape
    codes = np.zeros((n, nb), dtype="uint32")
    for i in range(n):
        br = faiss.BitstringReader(faiss.swig_ptr(packed_codes[i]), code_size)
        for j, nbi in enumerate(nbits):
            codes[i, j] = br.read(nbi)
    return codes


def retrain_AQ_codebook(index, xt):
    """ reference implementation of codebook retraining """
    rq = index.rq

    codes_packed = index.sa_encode(xt)
    n, code_size = codes_packed.shape

    x_decoded = index.sa_decode(codes_packed)
    MSE = ((xt - x_decoded) ** 2).sum() / n

    codes = unpack_codes(index.rq, codes_packed)
    codebook_offsets = faiss.vector_to_array(rq.codebook_offsets)

    # build sparse code matrix (represented as a dense matrix)
    C = np.zeros((n, rq.total_codebook_size))

    for i in range(n):
        C[i][codes[i] + codebook_offsets[:-1]] = 1

    # import pdb; pdb.set_trace()
    # import scipy
    # B, residuals, rank, singvals = np.linalg.lstsq(C, xt, rcond=None)
    if True:
        B, residuals, rank, singvals = np.linalg.lstsq(C, xt, rcond=None)
    else:
        import scipy.linalg
        B, residuals, rank, singvals = scipy.linalg.lstsq(C, xt, )

    MSE = ((C @ B - xt) ** 2).sum() / n

    # replace codebook
    # faiss.copy_array_to_vector(B.astype('float32').ravel(), index.rq.codebooks)
    # update codebook tables
    # index.rq.compute_codebook_tables()

    return C, B


class TestIndexResidualQuantizer(unittest.TestCase):

    def test_io(self):
        ds = datasets.SyntheticDataset(32, 1000, 100, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        ir = faiss.IndexResidualQuantizer(ds.d, 3, 4)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir.train(xt)
        ref_codes = ir.sa_encode(xb)

        b = faiss.serialize_index(ir)
        ir2 = faiss.deserialize_index(b)
        codes2 = ir2.sa_encode(xb)

        np.testing.assert_array_equal(ref_codes, codes2)

    def test_equiv_rq(self):
        """
        make sure it is equivalent to search a RQ and to search an IVF
        with RCQ + RQ with the same codebooks.
        """
        ds = datasets.SyntheticDataset(32, 3000, 1000, 50)

        # make a flat RQ
        iflat = faiss.IndexResidualQuantizer(ds.d, 5, 4)
        iflat.rq.train_type = faiss.ResidualQuantizer.Train_default
        iflat.train(ds.get_train())
        iflat.add(ds.get_database())

        # ref search result
        Dref, Iref = iflat.search(ds.get_queries(), 10)

        # get its codebooks + encoded version of the dataset
        codebooks = get_additive_quantizer_codebooks(iflat.rq)
        codes = faiss.vector_to_array(iflat.codes).reshape(-1, iflat.code_size)

        # make an IVF with 2x4 + 3x4 = 5x4 bits
        ivf = faiss.index_factory(ds.d, "IVF256(RCQ2x4),RQ3x4")

        # initialize the codebooks
        rcq = faiss.downcast_index(ivf.quantizer)
        faiss.copy_array_to_vector(
            np.vstack(codebooks[:rcq.rq.M]).ravel(),
            rcq.rq.codebooks
        )
        rcq.rq.is_trained = True
        # translation of AdditiveCoarseQuantizer::train
        rcq.ntotal = 1 << rcq.rq.tot_bits
        rcq.centroid_norms.resize(rcq.ntotal)
        rcq.rq.compute_centroid_norms(rcq.centroid_norms.data())
        rcq.is_trained = True

        faiss.copy_array_to_vector(
            np.vstack(codebooks[rcq.rq.M:]).ravel(),
            ivf.rq.codebooks
        )
        ivf.rq.is_trained = True
        ivf.is_trained = True

        # add the codes (this works because 2x4 is a multiple of 8 bits)
        ivf.add_sa_codes(codes)

        # perform exhaustive search
        ivf.nprobe = ivf.nlist

        Dnew, Inew = ivf.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)

    def test_factory(self):
        index = faiss.index_factory(5, "RQ2x16_3x8_6x4")

        np.testing.assert_array_equal(
            faiss.vector_to_array(index.rq.nbits),
            np.array([16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4])
        )

    def test_factory_norm(self):
        index = faiss.index_factory(5, "RQ8x8_Nqint8")
        self.assertEqual(
            index.rq.search_type,
            faiss.AdditiveQuantizer.ST_norm_qint8)

    def test_search_decompress(self):
        ds = datasets.SyntheticDataset(32, 1000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        ir = faiss.IndexResidualQuantizer(ds.d, 3, 4)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir.train(xt)
        ir.add(xb)

        D, I = ir.search(ds.get_queries(), 10)
        gt = ds.get_groundtruth()

        recalls = {
            rank: (I[:, :rank] == gt[:, :1]).sum() / len(gt)
            for rank in [1, 10, 100]
        }
        # recalls are {1: 0.05, 10: 0.37, 100: 0.37}
        self.assertGreater(recalls[10], 0.35)

    def do_exact_search_equiv(self, norm_type):
        """ searching with this normalization should yield
        exactly the same results as decompression (because the
        norms are exact) """
        ds = datasets.SyntheticDataset(32, 1000, 1000, 100)

        # decompresses by default
        ir = faiss.IndexResidualQuantizer(ds.d, 3, 6)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir.train(ds.get_train())
        ir.add(ds.get_database())
        Dref, Iref = ir.search(ds.get_queries(), 10)

        ir2 = faiss.IndexResidualQuantizer(
            ds.d, 3, 6, faiss.METRIC_L2, norm_type)

        # assumes training is reproducible
        ir2.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir2.train(ds.get_train())
        ir2.add(ds.get_database())
        D, I = ir2.search(ds.get_queries(), 10)

        np.testing.assert_allclose(D, Dref, atol=1e-5)
        np.testing.assert_array_equal(I, Iref)

    def test_exact_equiv_norm_float(self):
        self.do_exact_search_equiv(faiss.AdditiveQuantizer.ST_norm_float)

    def test_exact_equiv_norm_from_LUT(self):
        self.do_exact_search_equiv(faiss.AdditiveQuantizer.ST_norm_from_LUT)

    def test_reestimate_codebook(self):
        ds = datasets.SyntheticDataset(32, 1000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        ir = faiss.IndexResidualQuantizer(ds.d, 3, 4)
        ir.train(xt)

        # ir.rq.verbose = True
        xb_decoded = ir.sa_decode(ir.sa_encode(xb))
        err_before = ((xb - xb_decoded) ** 2).sum()

        # test manual call of retrain_AQ_codebook

        ref_C, ref_codebook = retrain_AQ_codebook(ir, xb)
        ir.rq.retrain_AQ_codebook(len(xb), faiss.swig_ptr(xb))

        xb_decoded = ir.sa_decode(ir.sa_encode(xb))
        err_after = ((xb - xb_decoded) ** 2).sum()

        # ref run: 8347.857 vs. 7710.014
        self.assertGreater(err_before, err_after * 1.05)

    def test_reestimate_codebook_2(self):
        ds = datasets.SyntheticDataset(32, 1000, 0, 0)
        xt = ds.get_train()

        ir = faiss.IndexResidualQuantizer(ds.d, 3, 4)
        ir.rq.train_type = 0
        ir.train(xt)

        xt_decoded = ir.sa_decode(ir.sa_encode(xt))
        err_before = ((xt - xt_decoded) ** 2).sum()

        ir = faiss.IndexResidualQuantizer(ds.d, 3, 4)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_refine_codebook
        ir.train(xt)

        xt_decoded = ir.sa_decode(ir.sa_encode(xt))
        err_after_refined = ((xt - xt_decoded) ** 2).sum()

        # ref run 7474.98 / 7006.1777
        self.assertGreater(err_before, err_after_refined * 1.06)





###########################################################
# As a coarse quantizer
###########################################################

class TestIVFResidualCoarseQuantizer(unittest.TestCase):

    def test_IVF_resiudal(self):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        gt = ds.get_groundtruth(1)

        # RQ 2x6 = 12 bits = 4096 centroids
        quantizer = faiss.ResidualCoarseQuantizer(ds.d, 2, 6)
        rq = quantizer.rq
        rq.train_type = faiss.ResidualQuantizer.Train_default
        index = faiss.IndexIVFFlat(quantizer, ds.d, 1 << rq.tot_bits)
        index.quantizer_trains_alone
        index.quantizer_trains_alone = True

        index.train(xt)
        index.add(xb)

        # make sure that increasing the nprobe increases accuracy

        index.nprobe = 10
        D, I = index.search(ds.get_queries(), 10)
        r10 = (I == gt[None, :]).sum() / ds.nq

        index.nprobe = 40
        D, I = index.search(ds.get_queries(), 10)
        r40 = (I == gt[None, :]).sum() / ds.nq

        self.assertGreater(r40, r10)

        # make sure that decreasing beam factor decreases accuracy
        quantizer.beam_factor
        quantizer.beam_factor = 1.0
        index.nprobe = 10
        D, I = index.search(ds.get_queries(), 10)
        r10_narrow_beam = (I == gt[None, :]).sum() / ds.nq

        self.assertGreater(r10, r10_narrow_beam)

    def test_factory(self):
        ds = datasets.SyntheticDataset(16, 500, 1000, 100)

        index = faiss.index_factory(ds.d, "IVF1024(RCQ2x5),Flat")
        index.train(ds.get_train())
        index.add(ds.get_database())

        Dref, Iref = index.search(ds.get_queries(), 10)

        b = faiss.serialize_index(index)
        index2 = faiss.deserialize_index(b)

        Dnew, Inew = index2.search(ds.get_queries(), 10)

        np.testing.assert_equal(Dref, Dnew)
        np.testing.assert_equal(Iref, Inew)

    def test_ivfsq(self):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        gt = ds.get_groundtruth(1)

        # RQ 2x5 = 10 bits = 1024 centroids
        index = faiss.index_factory(ds.d, "IVF1024(RCQ2x5),SQ8")
        quantizer = faiss.downcast_index(index.quantizer)
        rq = quantizer.rq
        rq.train_type = faiss.ResidualQuantizer.Train_default

        index.train(xt)
        index.add(xb)

        # make sure that increasing the nprobe increases accuracy

        index.nprobe = 10
        D, I = index.search(ds.get_queries(), 10)
        r10 = (I == gt[None, :]).sum() / ds.nq

        index.nprobe = 40
        D, I = index.search(ds.get_queries(), 10)
        r40 = (I == gt[None, :]).sum() / ds.nq

        self.assertGreater(r40, r10)

    def test_rcq_LUT(self):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        # RQ 2x5 = 10 bits = 1024 centroids
        index = faiss.index_factory(ds.d, "IVF1024(RCQ2x5),SQ8")

        quantizer = faiss.downcast_index(index.quantizer)
        rq = quantizer.rq
        rq.train_type = faiss.ResidualQuantizer.Train_default

        index.train(xt)
        index.add(xb)
        index.nprobe = 10

        # set exact centroids as coarse quantizer
        all_centroids = quantizer.reconstruct_n(0, quantizer.ntotal)
        q2 = faiss.IndexFlatL2(32)
        q2.add(all_centroids)
        index.quantizer = q2
        Dref, Iref = index.search(ds.get_queries(), 10)
        index.quantizer = quantizer

        # search with LUT
        quantizer.set_beam_factor(-1)
        Dnew, Inew = index.search(ds.get_queries(), 10)

        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)
        np.testing.assert_array_equal(Iref, Inew)

        # check i/o
        CDref, CIref = quantizer.search(ds.get_queries(), 10)
        quantizer2 = faiss.deserialize_index(faiss.serialize_index(quantizer))
        quantizer2.search(ds.get_queries(), 10)
        CDnew, CInew = quantizer2.search(ds.get_queries(), 10)
        np.testing.assert_array_almost_equal(CDref, CDnew, decimal=5)
        np.testing.assert_array_equal(CIref, CInew)

        # check that you can load the index without computing the tables
        quantizer.set_beam_factor(2.0)
        self.assertNotEqual(quantizer.rq.codebook_cross_products.size(), 0)
        quantizer3 = faiss.deserialize_index(
            faiss.serialize_index(quantizer),
            faiss.IO_FLAG_SKIP_PRECOMPUTE_TABLE
        )
        self.assertEqual(quantizer3.rq.codebook_cross_products.size(), 0)
        CD3, CI3 = quantizer3.search(ds.get_queries(), 10)


###########################################################
# Test search with LUTs
###########################################################


class TestAdditiveQuantizerWithLUT(unittest.TestCase):

    def test_RCQ_knn(self):
        ds = datasets.SyntheticDataset(32, 1000, 0, 123)
        xt = ds.get_train()
        xq = ds.get_queries()

        # RQ 3+4+5 = 12 bits = 4096 centroids
        rcq = faiss.index_factory(ds.d, "RCQ1x3_1x4_1x5")
        rcq.train(xt)

        aq = rcq.rq

        cents = rcq.reconstruct_n(0, rcq.ntotal)

        sp = faiss.swig_ptr

        # test norms computation

        norms_ref = (cents ** 2).sum(1)
        norms = np.zeros(1 << aq.tot_bits, dtype="float32")
        aq.compute_centroid_norms(sp(norms))

        np.testing.assert_array_almost_equal(norms, norms_ref, decimal=5)

        # test IP search

        Dref, Iref = faiss.knn(
            xq, cents, 10,
            metric=faiss.METRIC_INNER_PRODUCT
        )

        Dnew = np.zeros_like(Dref)
        Inew = np.zeros_like(Iref)

        aq.knn_centroids_inner_product(len(xq), sp(xq), 10, sp(Dnew), sp(Inew))

        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)
        np.testing.assert_array_equal(Iref, Inew)

        # test L2 search

        Dref, Iref = faiss.knn(xq, cents, 10, metric=faiss.METRIC_L2)

        Dnew = np.zeros_like(Dref)
        Inew = np.zeros_like(Iref)

        aq.knn_centroids_L2(len(xq), sp(xq), 10, sp(Dnew), sp(Inew), sp(norms))

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)


class TestIndexResidualQuantizerSearch(unittest.TestCase):

    def test_search_IP(self):
        ds = datasets.SyntheticDataset(32, 1000, 200, 100)

        xt = ds.get_train()
        xb = ds.get_database()
        xq = ds.get_queries()

        ir = faiss.IndexResidualQuantizer(
            ds.d, 3, 4, faiss.METRIC_INNER_PRODUCT)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir.train(xt)

        ir.add(xb)

        Dref, Iref = ir.search(xq, 4)

        AQ = faiss.AdditiveQuantizer
        ir2 = faiss.IndexResidualQuantizer(
            ds.d, 3, 4, faiss.METRIC_INNER_PRODUCT, AQ.ST_LUT_nonorm)
        ir2.rq.codebooks = ir.rq.codebooks    # fake training
        ir2.rq.is_trained = True
        ir2.is_trained = True
        ir2.add(xb)

        D2, I2 = ir2.search(xq, 4)

        np.testing.assert_array_equal(Iref, I2)
        np.testing.assert_array_almost_equal(Dref, D2, decimal=5)

    def test_search_L2(self):
        ds = datasets.SyntheticDataset(32, 1000, 200, 100)

        xt = ds.get_train()
        xb = ds.get_database()
        xq = ds.get_queries()
        gt = ds.get_groundtruth(10)

        ir = faiss.IndexResidualQuantizer(ds.d, 3, 4)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir.rq.max_beam_size = 30
        ir.train(xt)

        # reference run w/ decoding
        ir.add(xb)
        Dref, Iref = ir.search(xq, 10)

        # 388
        inter_ref = faiss.eval_intersection(Iref, gt)

        AQ = faiss.AdditiveQuantizer
        for st in AQ.ST_norm_float, AQ.ST_norm_qint8, AQ.ST_norm_qint4, \
                AQ.ST_norm_cqint8, AQ.ST_norm_cqint4:

            ir2 = faiss.IndexResidualQuantizer(ds.d, 3, 4, faiss.METRIC_L2, st)
            ir2.rq.max_beam_size = 30
            ir2.train(xt)   # to get the norm bounds
            ir2.rq.codebooks = ir.rq.codebooks    # fake training
            ir2.add(xb)

            D2, I2 = ir2.search(xq, 10)

            if st == AQ.ST_norm_float:
                np.testing.assert_array_almost_equal(Dref, D2, decimal=5)
                self.assertLess((Iref != I2).sum(), Iref.size * 0.05)
            else:
                inter_2 = faiss.eval_intersection(I2, gt)
                self.assertGreaterEqual(inter_ref, inter_2)


###########################################################
# IVF version
###########################################################


class TestIVFResidualQuantizer(unittest.TestCase):

    def do_test_accuracy(self, by_residual, st):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)

        quantizer = faiss.IndexFlatL2(ds.d)

        index = faiss.IndexIVFResidualQuantizer(
            quantizer, ds.d, 100, 3, 4,
            faiss.METRIC_L2, st
        )
        index.by_residual = by_residual

        index.rq.train_type
        index.rq.train_type = faiss.ResidualQuantizer.Train_default
        index.rq.max_beam_size = 30

        index.train(ds.get_train())
        index.add(ds.get_database())

        inters = []
        for nprobe in 1, 2, 5, 10, 20, 50:
            index.nprobe = nprobe
            D, I = index.search(ds.get_queries(), 10)
            inter = faiss.eval_intersection(I, ds.get_groundtruth(10))
            inters.append(inter)

        # do a little I/O test
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        D2, I2 = index2.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

        inters = np.array(inters)

        if by_residual:
            # check that we have increasing intersection measures with
            # nprobe
            self.assertTrue(np.all(inters[1:] >= inters[:-1]))
        else:
            self.assertTrue(np.all(inters[1:3] >= inters[:2]))
            # check that we have the same result as the flat residual quantizer
            iflat = faiss.IndexResidualQuantizer(
                ds.d, 3, 4, faiss.METRIC_L2, st)
            iflat.rq.train_type
            iflat.rq.train_type = faiss.ResidualQuantizer.Train_default
            iflat.rq.max_beam_size = 30
            iflat.train(ds.get_train())
            iflat.rq.codebooks = index.rq.codebooks

            iflat.add(ds.get_database())
            Dref, Iref = iflat.search(ds.get_queries(), 10)

            index.nprobe = 100
            D2, I2 = index.search(ds.get_queries(), 10)
            np.testing.assert_array_almost_equal(Dref, D2, decimal=5)
            # there are many ties because the codes are so short
            self.assertLess((Iref != I2).sum(), Iref.size * 0.2)

    def test_decompress_no_residual(self):
        self.do_test_accuracy(False, faiss.AdditiveQuantizer.ST_decompress)

    def test_norm_float_no_residual(self):
        self.do_test_accuracy(False, faiss.AdditiveQuantizer.ST_norm_float)

    def test_decompress(self):
        self.do_test_accuracy(True, faiss.AdditiveQuantizer.ST_decompress)

    def test_norm_float(self):
        self.do_test_accuracy(True, faiss.AdditiveQuantizer.ST_norm_float)

    def test_norm_cqint(self):
        self.do_test_accuracy(True, faiss.AdditiveQuantizer.ST_norm_cqint8)
        self.do_test_accuracy(True, faiss.AdditiveQuantizer.ST_norm_cqint4)

    def test_norm_from_LUT(self):
        self.do_test_accuracy(True, faiss.AdditiveQuantizer.ST_norm_from_LUT)

    def test_factory(self):
        index = faiss.index_factory(12, "IVF1024,RQ8x8_Nfloat")
        self.assertEqual(index.nlist, 1024)
        self.assertEqual(
            index.rq.search_type,
            faiss.AdditiveQuantizer.ST_norm_float
        )

        index = faiss.index_factory(12, "IVF1024,RQ8x8_Ncqint8")
        self.assertEqual(
            index.rq.search_type,
            faiss.AdditiveQuantizer.ST_norm_cqint8
        )
        index = faiss.index_factory(12, "IVF1024,RQ8x8_Ncqint4")
        self.assertEqual(
            index.rq.search_type,
            faiss.AdditiveQuantizer.ST_norm_cqint4
        )

    def do_test_accuracy_IP(self, by_residual):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100, "IP")

        quantizer = faiss.IndexFlatIP(ds.d)

        index = faiss.IndexIVFResidualQuantizer(
            quantizer, ds.d, 100, 3, 4,
            faiss.METRIC_INNER_PRODUCT, faiss.AdditiveQuantizer.ST_decompress
        )
        index.cp.spherical = True
        index.by_residual = by_residual

        index.rq.train_type
        index.rq.train_type = faiss.ResidualQuantizer.Train_default
        index.train(ds.get_train())

        index.add(ds.get_database())

        inters = []
        for nprobe in 1, 2, 5, 10, 20, 50:
            index.nprobe = nprobe
            index.rq.search_type = faiss.AdditiveQuantizer.ST_decompress
            D, I = index.search(ds.get_queries(), 10)
            index.rq.search_type = faiss.AdditiveQuantizer.ST_LUT_nonorm
            D2, I2 = index.search(ds.get_queries(), 10)
            np.testing.assert_array_almost_equal(D, D2, decimal=5)
            # there are many ties because the codes are so short
            self.assertLess((I != I2).sum(), I.size * 0.1)

            # D2, I2 = index2.search(ds.get_queries(), 10)

            inter = faiss.eval_intersection(I, ds.get_groundtruth(10))
            inters.append(inter)
        self.assertTrue(np.all(inters[1:4] >= inters[:3]))

    def test_no_residual_IP(self):
        self.do_test_accuracy_IP(False)

    def test_residual_IP(self):
        self.do_test_accuracy_IP(True)


############################################################
# Test functions that use precomputed codebook products
############################################################


def precomp_codebooks(codebooks):
    M = len(codebooks)
    codebook_cross_prods = [
        [codebooks[m1] @ codebooks[m].T for m1 in range(m)] for m in range(M)
    ]
    cent_norms = [
        (c ** 2).sum(1)
        for c in codebooks
    ]
    return codebook_cross_prods, cent_norms


############################################################
# Reference imelementation of table-based beam search (use_beam_LUT=1)
############################################################

def beam_search_encode_step_tab(codes, L, distances, codebook_cross_prods_i,
                                query_cp_i, cent_norms_i):
    """ Reference beam search implementation
    encodes a residual table.
    """
    n, beam_size, m = codes.shape

    n2, beam_size_2 = distances.shape
    assert n2 == n and beam_size_2 == beam_size
    n2, K = query_cp_i.shape
    assert n2 == n
    K2, = cent_norms_i.shape
    assert K == K2
    assert len(codebook_cross_prods_i) == m

    # n, beam_size, K
    new_distances = distances[:, :, None] + cent_norms_i[None, None, :]
    new_distances -= 2 * query_cp_i[:, None, :]

    dotprods = np.zeros((n, beam_size, K))

    for j in range(m):
        cb = codebook_cross_prods_i[j]
        for i in range(n):
            for b in range(beam_size):
                dotprods[i, b, :] += cb[codes[i, b, j]]

    new_distances += 2 * dotprods
    cent_distances = new_distances

    # TODO write in vector form

    if beam_size * K <= L:
        # then keep all the results
        new_beam_size = beam_size * K
        new_codes = np.zeros((n, beam_size, K, m + 1), dtype=int)
        for i in range(n):
            new_codes[i, :, :, :-1] = codes[i]
            new_codes[i, :, :, -1] = np.arange(K)
        new_codes = new_codes.reshape(n, new_beam_size, m + 1)
        new_distances = cent_distances.reshape(n, new_beam_size)
    else:
        # keep top-L results
        new_beam_size = L
        new_codes = np.zeros((n, L, m + 1), dtype=int)
        new_distances = np.zeros((n, L), dtype='float32')
        for i in range(n):
            cd = cent_distances[i].ravel()
            jl = np.argsort(cd)[:L]    # TODO argpartition
            js = jl // K     # input beam index
            ls = jl % K      # centroid index
            new_codes[i, :, :-1] = codes[i, js, :]
            new_codes[i, :, -1] = ls
            new_distances[i, :] = cd[jl]

    return new_codes, new_distances


def beam_search_encoding_tab(codebooks, x, L, precomp, implem="ref"):
    """
    Perform encoding of vectors x with a beam search of size L
    """
    compare_implem = "ref" in implem and "cpp" in implem

    query_cross_prods = [
        x @ c.T for c in codebooks
    ]

    M = len(codebooks)
    codebook_offsets = np.zeros(M + 1, dtype='uint64')
    codebook_offsets[1:] = np.cumsum([len(cb) for cb in codebooks])
    codebook_cross_prods, cent_norms = precomp
    n, d = x.shape
    beam_size = 1
    codes = np.zeros((n, beam_size, 0), dtype='int32')
    distances = (x ** 2).sum(1).reshape(n, beam_size)

    for m, cent in enumerate(codebooks):

        if "ref" in implem:
            new_codes, new_distances = beam_search_encode_step_tab(
                codes, L,
                distances, codebook_cross_prods[m][:m],
                query_cross_prods[m], cent_norms[m]
            )
            new_beam_size = codes.shape[1]

        if compare_implem:
            codes_ref = new_codes
            distances_ref = new_distances

        if "cpp" in implem:
            K = len(cent)
            new_beam_size = min(beam_size * K, L)
            new_codes = np.zeros((n, new_beam_size, m + 1), dtype='int32')
            new_distances = np.zeros((n, new_beam_size), dtype="float32")
            if m > 0:
                cp = np.vstack(codebook_cross_prods[m][:m])
            else:
                cp = np.zeros((0, K), dtype='float32')

            sp = faiss.swig_ptr
            faiss.beam_search_encode_step_tab(
                K, n, beam_size,
                sp(cp), cp.shape[1],
                sp(codebook_offsets),
                sp(query_cross_prods[m]), query_cross_prods[m].shape[1],
                sp(cent_norms[m]),
                m,
                sp(codes), sp(distances),
                new_beam_size,
                sp(new_codes), sp(new_distances)
            )

        if compare_implem:
            np.testing.assert_array_almost_equal(
                new_distances, distances_ref, decimal=5)
            np.testing.assert_array_equal(
                new_codes, codes_ref)

        codes = new_codes
        distances = new_distances
        beam_size = new_beam_size

    return (codes, distances)


class TestCrossCodebookComputations(unittest.TestCase):

    def test_precomp(self):
        ds = datasets.SyntheticDataset(32, 1000, 1000, 0)

        # make sure it work with varying nb of bits
        nbits = faiss.UInt64Vector()
        nbits.push_back(5)
        nbits.push_back(6)
        nbits.push_back(7)

        rq = faiss.ResidualQuantizer(ds.d, nbits)
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.train(ds.get_train())

        codebooks = get_additive_quantizer_codebooks(rq)
        precomp = precomp_codebooks(codebooks)
        codebook_cross_prods_ref, cent_norms_ref = precomp

        # validate that the python tab-based encoding works
        xb = ds.get_database()
        ref_codes, _, _ = beam_search_encoding_ref(codebooks, xb, 7)
        new_codes, _ = beam_search_encoding_tab(codebooks, xb, 7, precomp)
        np.testing.assert_array_equal(ref_codes, new_codes)

        # check C++ precomp tables
        rq.compute_codebook_tables()
        codebook_cross_prods = faiss.vector_to_array(
            rq.codebook_cross_products)
        ofs = 0
        for m in range(1, rq.M):
            py_table = np.vstack(codebook_cross_prods_ref[m])
            kk = rq.codebook_offsets.at(m)
            K = 1 << rq.nbits.at(m)
            cpp_table = codebook_cross_prods[ofs:ofs + K * kk].reshape(kk, K)
            ofs += kk * K
            np.testing.assert_allclose(py_table, cpp_table, atol=1e-5)

        cent_norms = faiss.vector_to_array(rq.centroid_norms)
        np.testing.assert_array_almost_equal(
            np.hstack(cent_norms_ref), cent_norms, decimal=5)

        # validate the C++ beam_search_encode_step_tab function
        beam_search_encoding_tab(codebooks, xb, 7, precomp, implem="ref cpp")

        # check implem w/ residuals
        n = ref_codes.shape[0]
        sp = faiss.swig_ptr
        ref_codes_packed = np.zeros((n, rq.code_size), dtype='uint8')
        ref_codes_int32 = ref_codes.astype('int32')
        rq.pack_codes(
            n, sp(ref_codes_int32),
            sp(ref_codes_packed), rq.M * ref_codes.shape[1]
        )

        rq.max_beam_size = 7
        codes_ref_residuals = rq.compute_codes(xb)
        np.testing.assert_array_equal(ref_codes_packed, codes_ref_residuals)

        rq.use_beam_LUT = 1
        codes_new = rq.compute_codes(xb)
        np.testing.assert_array_equal(codes_ref_residuals, codes_new)


class TestProductResidualQuantizer(unittest.TestCase):

    def test_codec(self):
        """check that the error is in the same ballpark as PQ."""
        ds = datasets.SyntheticDataset(64, 3000, 3000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        nsplits = 2
        Msub = 2
        nbits = 4

        prq = faiss.ProductResidualQuantizer(ds.d, nsplits, Msub, nbits)
        prq.train(xt)
        err_prq = eval_codec(prq, xb)

        pq = faiss.ProductQuantizer(ds.d, nsplits * Msub, nbits)
        pq.train(xt)
        err_pq = eval_codec(pq, xb)

        self.assertLess(err_prq, err_pq)

    def test_with_rq(self):
        """compare with RQ when nsplits = 1"""
        ds = datasets.SyntheticDataset(32, 3000, 3000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        M = 4
        nbits = 4

        prq = faiss.ProductResidualQuantizer(ds.d, 1, M, nbits)
        prq.train(xt)
        err_prq = eval_codec(prq, xb)

        rq = faiss.ResidualQuantizer(ds.d, M, nbits)
        rq.train(xt)
        err_rq = eval_codec(rq, xb)

        self.assertEqual(err_prq, err_rq)


class TestIndexProductResidualQuantizer(unittest.TestCase):

    def test_accuracy1(self):
        """check that the error is in the same ballpark as RQ."""
        recall1 = self.eval_index_accuracy("PRQ4x3x5_Nqint8")
        recall2 = self.eval_index_accuracy("RQ12x5_Nqint8")
        self.assertGreaterEqual(recall1 * 1.1, recall2)  # 657 vs 665

    def test_accuracy2(self):
        """when nsplits = 1, PRQ should be the same as RQ"""
        recall1 = self.eval_index_accuracy("PRQ1x3x5_Nqint8")
        recall2 = self.eval_index_accuracy("RQ3x5_Nqint8")
        self.assertEqual(recall1, recall2)

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
        index = faiss.index_factory(64, f"PRQ{ns}x{Msub}x{nbits}_Nqint8")
        assert isinstance(index, faiss.IndexProductResidualQuantizer)
        self.assertEqual(index.prq.nsplits, ns)
        self.assertEqual(index.prq.subquantizer(0).M, Msub)
        self.assertEqual(index.prq.subquantizer(0).nbits.at(0), nbits)
        self.assertEqual(index.prq.search_type, AQ.ST_norm_qint8)

        code_size = (ns * Msub * nbits + 7) // 8 + 1
        self.assertEqual(index.prq.code_size, code_size)


class TestIndexIVFProductResidualQuantizer(unittest.TestCase):

    def eval_index_accuracy(self, factory_key):
        ds = datasets.SyntheticDataset(32, 1000, 1000, 100)
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
        # 1.05: test relaxed for OSX on ARM
        self.assertTrue(np.all(inters[1:] * 1.05 >= inters[:-1]))

        # do a little I/O test
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        D2, I2 = index2.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

        return inter

    def test_index_accuracy(self):
        self.eval_index_accuracy("IVF100,PRQ2x2x5_Nqint8")

    def test_index_accuracy2(self):
        """check that the error is in the same ballpark as RQ."""
        inter1 = self.eval_index_accuracy("IVF100,PRQ2x2x5_Nqint8")
        inter2 = self.eval_index_accuracy("IVF100,RQ4x5_Nqint8")
        self.assertGreaterEqual(inter1 * 1.1, inter2)

    def test_factory(self):
        AQ = faiss.AdditiveQuantizer
        ns, Msub, nbits = 2, 4, 8
        index = faiss.index_factory(64, f"IVF100,PRQ{ns}x{Msub}x{nbits}_Nqint8")
        assert isinstance(index, faiss.IndexIVFProductResidualQuantizer)
        self.assertEqual(index.nlist, 100)
        self.assertEqual(index.prq.nsplits, ns)
        self.assertEqual(index.prq.subquantizer(0).M, Msub)
        self.assertEqual(index.prq.subquantizer(0).nbits.at(0), nbits)
        self.assertEqual(index.prq.search_type, AQ.ST_norm_qint8)

        code_size = (ns * Msub * nbits + 7) // 8 + 1
        self.assertEqual(index.prq.code_size, code_size)
