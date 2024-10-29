# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" test byte codecs """

from __future__ import print_function
import numpy as np
import unittest
import faiss

from common_faiss_tests import get_dataset_2
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks


class TestEncodeDecode(unittest.TestCase):

    def do_encode_twice(self, factory_key):
        d = 96
        nb = 1000
        nq = 0
        nt = 2000

        xt, x, _ = get_dataset_2(d, nt, nb, nq)

        assert x.size > 0

        codec = faiss.index_factory(d, factory_key)

        codec.train(xt)

        codes = codec.sa_encode(x)
        x2 = codec.sa_decode(codes)

        codes2 = codec.sa_encode(x2)

        if 'IVF' in factory_key or 'RQ' in factory_key:
            # some rows are not reconstructed exactly because they
            # flip into another quantization cell
            nrowdiff = (codes != codes2).any(axis=1).sum()
            self.assertTrue(nrowdiff < 10)
        else:
            self.assertTrue(np.all(codes == codes2))

        x3 = codec.sa_decode(codes2)

        if 'IVF' in factory_key or 'RQ' in factory_key:
            diffs = np.abs(x2 - x3).sum(axis=1)
            avg = np.abs(x2).sum(axis=1).mean()
            diffs.sort()
            assert diffs[-10] < avg * 1e-5
        else:
            self.assertTrue(np.allclose(x2, x3))

    def test_SQ8(self):
        self.do_encode_twice('SQ8')

    def test_IVFSQ8(self):
        self.do_encode_twice('IVF256,SQ8')

    def test_PCAIVFSQ8(self):
        self.do_encode_twice('PCAR32,IVF256,SQ8')

    def test_PQ6x8(self):
        self.do_encode_twice('PQ6np')

    def test_PQ6x6(self):
        self.do_encode_twice('PQ6x6np')

    def test_IVFPQ6x8np(self):
        self.do_encode_twice('IVF512,PQ6np')

    def test_LSH(self):
        self.do_encode_twice('LSHrt')

    def test_RQ6x8(self):
        self.do_encode_twice('RQ6x8')


class TestIndexEquiv(unittest.TestCase):

    def do_test(self, key1, key2):
        d = 96
        nb = 1000
        nq = 0
        nt = 2000

        xt, x, _ = get_dataset_2(d, nt, nb, nq)

        codec_ref = faiss.index_factory(d, key1)
        codec_ref.train(xt)

        code_ref = codec_ref.sa_encode(x)
        x_recons_ref = codec_ref.sa_decode(code_ref)

        codec_new = faiss.index_factory(d, key2)
        codec_new.pq = codec_ref.pq

        # replace quantizer, avoiding mem leak
        oldq = codec_new.q1.quantizer
        oldq.this.own()
        codec_new.q1.own_fields = False
        codec_new.q1.quantizer = codec_ref.quantizer
        codec_new.is_trained = True

        code_new = codec_new.sa_encode(x)
        x_recons_new = codec_new.sa_decode(code_new)

        self.assertTrue(np.all(code_new == code_ref))
        self.assertTrue(np.all(x_recons_new == x_recons_ref))

        codec_new_2 = faiss.deserialize_index(
            faiss.serialize_index(codec_new))

        code_new = codec_new_2.sa_encode(x)
        x_recons_new = codec_new_2.sa_decode(code_new)

        self.assertTrue(np.all(code_new == code_ref))
        self.assertTrue(np.all(x_recons_new == x_recons_ref))

    def test_IVFPQ(self):
        self.do_test("IVF512,PQ6np", "Residual512,PQ6")

    def test_IMI(self):
        self.do_test("IMI2x5,PQ6np", "Residual2x5,PQ6")


class TestAccuracy(unittest.TestCase):
    """ comparative accuracy of a few types of indexes """

    def compare_accuracy(self, lowac, highac, max_errs=(1e10, 1e10)):
        d = 96
        nb = 1000
        nq = 0
        nt = 2000

        xt, x, _ = get_dataset_2(d, nt, nb, nq)

        errs = []

        for factory_string in lowac, highac:

            codec = faiss.index_factory(d, factory_string)
            print('sa codec: code size %d' % codec.sa_code_size())
            codec.train(xt)

            codes = codec.sa_encode(x)
            x2 = codec.sa_decode(codes)

            err = ((x - x2) ** 2).sum()
            errs.append(err)

        self.assertGreater(errs[0], errs[1])

        self.assertGreater(max_errs[0], errs[0])
        self.assertGreater(max_errs[1], errs[1])

        # just a small IndexLattice I/O test
        if 'Lattice' in highac:
            codec2 = faiss.deserialize_index(
                faiss.serialize_index(codec))
            codes = codec2.sa_encode(x)
            x3 = codec2.sa_decode(codes)
            self.assertTrue(np.all(x2 == x3))

    def test_SQ(self):
        self.compare_accuracy('SQ4', 'SQ8')

    def test_SQ2(self):
        self.compare_accuracy('SQ6', 'SQ8')

    def test_SQ3(self):
        self.compare_accuracy('SQ8', 'SQfp16')

    def test_SQ4(self):
        self.compare_accuracy('SQ8', 'SQbf16')

    def test_PQ(self):
        self.compare_accuracy('PQ6x8np', 'PQ8x8np')

    def test_PQ2(self):
        self.compare_accuracy('PQ8x6np', 'PQ8x8np')

    def test_IVFvsPQ(self):
        self.compare_accuracy('PQ8np', 'IVF256,PQ8np')

    def test_Lattice(self):
        # measured low/high: 20946.244, 5277.483
        self.compare_accuracy('ZnLattice3x10_4',
                              'ZnLattice3x20_4',
                              (22000, 5400))

    def test_Lattice2(self):
        # here the difference is actually tiny
        # measured errs: [16403.072, 15967.735]
        self.compare_accuracy('ZnLattice3x12_1',
                              'ZnLattice3x12_7',
                              (18000, 16000))


swig_ptr = faiss.swig_ptr


class LatticeTest(unittest.TestCase):
    """ Low-level lattice tests """

    def test_repeats(self):
        rs = np.random.RandomState(123)
        dim = 32
        for _i in range(1000):
            vec = np.floor((rs.rand(dim) ** 7) * 3).astype('float32')
            vecs = vec.copy()
            vecs.sort()
            repeats = faiss.Repeats(dim, swig_ptr(vecs))
            code = repeats.encode(swig_ptr(vec))
            vec2 = np.zeros(dim, dtype='float32')
            repeats.decode(code, swig_ptr(vec2))
            assert np.all(vec == vec2)

    def test_ZnSphereCodec_encode_centroid(self):
        dim = 8
        r2 = 5
        ref_codec = faiss.ZnSphereCodec(dim, r2)
        codec = faiss.ZnSphereCodecRec(dim, r2)
        assert ref_codec.nv == codec.nv
        s = set()
        for i in range(ref_codec.nv):
            c = np.zeros(dim, dtype='float32')
            ref_codec.decode(i, swig_ptr(c))
            code = codec.encode_centroid(swig_ptr(c))
            assert 0 <= code < codec.nv
            s.add(code)
        assert len(s) == codec.nv

    def test_ZnSphereCodecRec(self):
        dim = 16
        r2 = 6
        codec = faiss.ZnSphereCodecRec(dim, r2)
        for i in range(codec.nv):
            c = np.zeros(dim, dtype='float32')
            codec.decode(i, swig_ptr(c))
            code = codec.encode_centroid(swig_ptr(c))
            assert code == i

    def run_ZnSphereCodecAlt(self, dim, r2):
        # dim = 32
        # r2 = 14
        codec = faiss.ZnSphereCodecAlt(dim, r2)
        rs = np.random.RandomState(123)
        n = 100
        codes = rs.randint(codec.nv, size=n, dtype='uint64')
        x = np.empty((n, dim), dtype='float32')
        codec.decode_multi(n, swig_ptr(codes), swig_ptr(x))
        codes2 = np.empty(n, dtype='uint64')
        codec.encode_multi(n, swig_ptr(x), swig_ptr(codes2))

        assert np.all(codes == codes2)

    def test_ZnSphereCodecAlt32(self):
        self.run_ZnSphereCodecAlt(32, 14)

    def test_ZnSphereCodecAlt24(self):
        self.run_ZnSphereCodecAlt(24, 14)

    def test_lattice_index(self):
        index = faiss.index_factory(96, "ZnLattice3x10_4")
        rs = np.random.RandomState(123)
        xq = rs.randn(10, 96).astype('float32')
        xb = rs.randn(20, 96).astype('float32')
        index.train(xb)
        index.add(xb)
        D, I = index.search(xq, 5)
        for i in range(10):
            recons = index.reconstruct_batch(I[i, :])
            ref_dis = ((recons - xq[i]) ** 2).sum(1)
            np.testing.assert_allclose(D[i, :], ref_dis, atol=1e-4)


class TestBitstring(unittest.TestCase):

    def test_rw(self):
        """ Low-level bit string tests """
        rs = np.random.RandomState(1234)
        nbyte = 1000
        sz = 0

        bs = np.ones(nbyte, dtype='uint8')
        bw = faiss.BitstringWriter(swig_ptr(bs), nbyte)

        if False:
            ctrl = [(7, 0x35), (13, 0x1d74)]
            for nbit, x in ctrl:
                bw.write(x, nbit)
        else:
            ctrl = []
            while True:
                nbit = int(1 + 62 * rs.rand() ** 4)
                if sz + nbit > nbyte * 8:
                    break
                x = int(rs.randint(1 << nbit, dtype='int64'))
                bw.write(x, nbit)
                ctrl.append((nbit, x))
                sz += nbit

        bignum = 0
        sz = 0
        for nbit, x in ctrl:
            bignum |= x << sz
            sz += nbit

        for i in range(nbyte):
            self.assertTrue(((bignum >> (i * 8)) & 255) == bs[i])

        br = faiss.BitstringReader(swig_ptr(bs), nbyte)

        for nbit, xref in ctrl:
            xnew = br.read(nbit)
            self.assertTrue(xnew == xref)

    def test_arrays(self):
        nbit = 5
        M = 10
        n = 20
        rs = np.random.RandomState(123)
        a = rs.randint(1<<nbit, size=(n, M), dtype='int32')
        b = faiss.pack_bitstrings(a, nbit)
        c = faiss.unpack_bitstrings(b, M, nbit)
        np.testing.assert_array_equal(a, c)

    def test_arrays_variable_size(self):
        nbits = [10, 5, 3, 12, 6, 7, 4]
        n = 20
        rs = np.random.RandomState(123)
        a = rs.randint(1<<16, size=(n, len(nbits)), dtype='int32')
        a &= (1 << np.array(nbits)) - 1
        b = faiss.pack_bitstrings(a, nbits)
        c = faiss.unpack_bitstrings(b, nbits)
        np.testing.assert_array_equal(a, c)


class TestIVFTransfer(unittest.TestCase):

    def test_transfer(self):

        ds = SyntheticDataset(32, 2000, 200, 100)
        index = faiss.index_factory(ds.d, "IVF20,SQ8")
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 10)
        index.reset()

        codes = index.sa_encode(ds.get_database())
        index.add_sa_codes(codes)

        Dnew, Inew = index.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)


class TestIDMap(unittest.TestCase):
    def test_idmap(self):
        ds = SyntheticDataset(32, 2000, 200, 100)
        ids = np.random.randint(10000, size=ds.nb, dtype='int64')
        index = faiss.index_factory(ds.d, "IDMap2,PQ8x2")
        index.train(ds.get_train())
        index.add_with_ids(ds.get_database(), ids)
        Dref, Iref = index.search(ds.get_queries(), 10)

        index.reset()

        index.train(ds.get_train())
        codes = index.index.sa_encode(ds.get_database())
        index.add_sa_codes(codes, ids)
        Dnew, Inew = index.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)
        


class TestRefine(unittest.TestCase):

    def test_refine(self):
        """ Make sure that IndexRefine can function as a standalone codec """

        ds = SyntheticDataset(32, 500, 100, 0)
        index = faiss.index_factory(ds.d, "RQ2x5,Refine(ITQ,LSHt)")
        index.train(ds.get_train())

        index1 = index.base_index
        index2 = index.refine_index

        codes12 = index.sa_encode(ds.get_database())
        codes1 = index1.sa_encode(ds.get_database())
        codes2 = index2.sa_encode(ds.get_database())

        np.testing.assert_array_equal(
            codes12,
            np.hstack((codes1, codes2))
        )

    def test_equiv_rcq_rq(self):
        """ make sure that the codes generated by the standalone codec are the same
        between an
           IndexRefine with ResidualQuantizer
        and
           IVF with ResidualCoarseQuantizer
        both are the centroid id concatenated with the code.
        """
        ds = SyntheticDataset(16, 400, 100, 0)
        index1 = faiss.index_factory(ds.d, "RQ2x3,Refine(Flat)")
        index1.train(ds.get_train())
        irq = faiss.downcast_index(index1.base_index)
        # because the default beam factor for RCQ is 4
        irq.rq.max_beam_size = 4

        index2 = faiss.index_factory(ds.d, "IVF64(RCQ2x3),Flat")
        index2.train(ds.get_train())
        quantizer = faiss.downcast_index(index2.quantizer)
        quantizer.rq = irq.rq
        index2.is_trained = True

        codes1 = index1.sa_encode(ds.get_database())
        codes2 = index2.sa_encode(ds.get_database())

        np.testing.assert_array_equal(codes1, codes2)

    def test_equiv_sh(self):
        """ make sure that the IVFSpectralHash sa_encode function gives the same
        result as the concatenated RQ + LSH index sa_encode """
        ds = SyntheticDataset(32, 500, 100, 0)
        index1 = faiss.index_factory(ds.d, "RQ1x4,Refine(ITQ16,LSH)")
        index1.train(ds.get_train())

        # reproduce this in an IndexIVFSpectralHash
        coarse_quantizer = faiss.IndexFlat(ds.d)
        rq = faiss.downcast_index(index1.base_index).rq
        centroids = get_additive_quantizer_codebooks(rq)[0]
        coarse_quantizer.add(centroids)

        encoder = faiss.downcast_index(index1.refine_index)

        # larger than the magnitude of the vectors
        # negative because otherwise the bits are flipped
        period = -100000.0

        index2 = faiss.IndexIVFSpectralHash(
            coarse_quantizer,
            ds.d,
            coarse_quantizer.ntotal,
            encoder.sa_code_size() * 8,
            period
        )

        # replace with the vt of the encoder. Binarization is performed by
        # the IndexIVFSpectralHash itself
        index2.replace_vt(encoder)

        codes1 = index1.sa_encode(ds.get_database())
        codes2 = index2.sa_encode(ds.get_database())

        np.testing.assert_array_equal(codes1, codes2)
