# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

""" test byte codecs """

import numpy as np
import unittest
import faiss
import tempfile
import os

from catalyzer.quantizers import ITQ

from common import get_dataset_2


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

        if 'IVF' not in factory_key:
            self.assertTrue(np.all(codes == codes2))
        else:
            # some rows are not reconstructed exactly because they
            # flip into another quantization cell
            nrowdiff = (codes != codes2).any(axis=1).sum()
            self.assertTrue(nrowdiff < 10)

        x3 = codec.sa_decode(codes2)
        if 'IVF' not in factory_key:
            self.assertTrue(np.allclose(x2, x3))
        else:
            diffs = np.abs(x2 - x3).sum(axis=1)
            avg = np.abs(x2).sum(axis=1).mean()
            diffs.sort()
            assert diffs[-10] < avg * 1e-5

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

        print(errs)
        self.assertGreater(errs[0], errs[1])

        self.assertGreater(max_errs[0], errs[0])
        self.assertGreater(max_errs[1], errs[1])

        # just a small IndexLattice I/O test
        if 'Lattice' in highac:
            codec2 = faiss.deserialize_index(
                faiss.serialize_index(codec))
            codes = codec.sa_encode(x)
            x3 = codec.sa_decode(codes)
            self.assertTrue(np.all(x2 == x3))

    def test_SQ(self):
        self.compare_accuracy('SQ4', 'SQ8')

    def test_SQ2(self):
        self.compare_accuracy('SQ6', 'SQ8')

    def test_SQ3(self):
        self.compare_accuracy('SQ8', 'SQfp16')

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


class TestITQ(unittest.TestCase):
    """tests done in 3 levels from lowest to highest. Trainings are
    repeated 10x because jitter is quite important."""

    def fit_rotation(self, A, B):
        """ solve argmin norm_F (Omega * A - B) for Omega orthogonal
        (Orthogonal Procrustes problem) """
        M = np.dot(B, A.T)
        U, sigma, Vt = np.linalg.svd(M)
        R = np.dot(U, Vt)
        residual = ((np.dot(R, A) - B) ** 2).sum()
        return R, residual

    def test_ITQ1(self):
        """ Just test the ITQ optimization with a given initialization """
        d = 15
        nb = 1000
        nq = 0
        nt = 10000

        xt, xb, _ = get_dataset_2(d, nt, nb, nq)
        for xp in range(10):
            xti = xt[xp * 1000: (xp + 1) * 1000]

            ref_itq = ITQ(d, do_pca=False)
            ref_itq.train(xti)
            x_bin_ref = ref_itq.quantize(xb)
            rot_ref = ref_itq._rotation

            new_itq = faiss.ITQMatrix(d)
            mean = xti.mean(0)
            training_data = xti - mean
            faiss.normalize_L2(training_data)
            new_itq.train(training_data)

            xb_normed = xb - mean
            faiss.normalize_L2(xb_normed)

            x_new = new_itq.apply_py(xb_normed)
            x_bin_new = np.where(x_new > 0, 1, -1)

            # compute objective value using computed rotation matrix
            err_ref = ((x_bin_ref - np.dot(xb_normed, rot_ref))**2).sum()
            rot_new = faiss.vector_to_array(new_itq.A).reshape(d, d)
            err_new = ((x_bin_new - np.dot(xb_normed, rot_new.T))**2).sum()

            self.assertTrue(err_ref * 1.01 > err_new)

    def test_ITQ2(self):
        """ test ITQTransform (that does the normalization) """
        d = 15
        nb = 1000
        nq = 0
        nt = 10000

        xt, xb, _ = get_dataset_2(d, nt, nb, nq)

        for xp in range(10):
            xti = xt[xp * 1000: (xp + 1) * 1000]

            ref_itq = ITQ(d, do_pca=False)
            ref_itq.train(xti)
            x_bin_ref = ref_itq.quantize(xb)

            new_itq = faiss.ITQTransform(d, d, False)
            new_itq.train(xti)

            x_new = new_itq.apply_py(xb)
            x_bin_new = np.where(x_new > 0, 1, -1)

            # compute objective value recomputing optimal rotation
            _, err_ref = self.fit_rotation(xb.T, x_bin_ref.T)
            _, err_new = self.fit_rotation(xb.T, x_bin_new.T)

            # test is more lax here because we are not directly
            # optimizing the same objective
            self.assertTrue(err_ref * 1.05 > err_new)

    def test_ITQ3(self):
        """ Direct test with an index, include PCA """
        d = 40
        nb = 1000
        nq = 0
        nt = 10000
        d_out = 16

        xt, xb, _ = get_dataset_2(d, nt, nb, nq)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        _, gt_I = index.search(xb, 10)

        def to_binary(x):
            n, d = x.shape
            assert d % 8 == 0
            xs = ((x.reshape(-1, 8) > 0) << np.arange(8)).sum(1).astype('uint8')
            return xs.reshape(n, d // 8)

        for xp in range(10):
            xti = xt[xp * 1000: (xp + 1) * 1000]
            ref_itq = ITQ(d_out, do_pca=True)
            ref_itq.train(xti)
            x_bin_ref = to_binary(ref_itq.quantize(xb))

            new_itq = faiss.ITQTransform(d, d_out, True)
            new_itq.train(xti)
            x_new = new_itq.apply_py(xb)
            x_bin_new = to_binary(np.where(x_new > 0, 1, -1))

            ninter_ref = None
            for x_bin in x_bin_ref, x_bin_new:

                index = faiss.IndexBinaryFlat(d_out)
                index.add(x_bin)
                _, I = index.search(x_bin, 10)
                ninter = faiss.eval_intersection(I, gt_I)
                if ninter_ref is None:
                    ninter_ref = ninter
                else:
                    ninter_new = ninter

            self.assertTrue(abs(ninter_ref - ninter_new) < 100)

            print(ninter_ref, ninter_new)

            # test I/O for one experiment
            if xp != 0:
                continue

            try:
                _, tmpfile = tempfile.mkstemp()
                faiss.write_VectorTransform(new_itq, tmpfile)
                new_itq_2 = faiss.read_VectorTransform(tmpfile)
            finally:
                if os.path.exists(tmpfile):
                    os.unlink(tmpfile)

            x_new_2 = new_itq_2.apply_py(xb)
            self.assertTrue(np.all(x_new == x_new_2))


swig_ptr = faiss.swig_ptr


class LatticeTest(unittest.TestCase):
    """ Low-level lattice tests """

    def test_repeats(self):
        rs = np.random.RandomState(123)
        dim = 32
        for i in range(1000):
            vec = np.floor((rs.rand(dim) ** 7) * 3).astype('float32')
            vecs = vec.copy()
            vecs.sort()
            repeats = faiss.Repeats(dim, swig_ptr(vecs))
            rr = [repeats.repeats.at(i) for i in range(repeats.repeats.size())]
            # print([(r.val, r.n) for r in rr])
            code = repeats.encode(swig_ptr(vec))
            #print(vec, code)
            vec2 = np.zeros(dim, dtype='float32')
            repeats.decode(code, swig_ptr(vec2))
            # print(vec2)
            assert np.all(vec == vec2)

    def test_ZnSphereCodec_encode_centroid(self):
        dim = 8
        r2 = 5
        ref_codec = faiss.ZnSphereCodec(dim, r2)
        codec = faiss.ZnSphereCodecRec(dim, r2)
        # print(ref_codec.nv, codec.nv)
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
        # print("nv=", codec.nv)
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
        codes = rs.randint(codec.nv, size=n).astype('uint64')
        x = np.empty((n, dim), dtype='float32')
        codec.decode_multi(n, swig_ptr(codes), swig_ptr(x))
        codes2 = np.empty(n, dtype='uint64')
        codec.encode_multi(n, swig_ptr(x), swig_ptr(codes2))

        assert np.all(codes == codes2)

    def test_ZnSphereCodecAlt32(self):
        self.run_ZnSphereCodecAlt(32, 14)

    def test_ZnSphereCodecAlt24(self):
        self.run_ZnSphereCodecAlt(24, 14)


class TestBitstring(unittest.TestCase):
    """ Low-level bit string tests """

    def test_rw(self):
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
                x = rs.randint(1 << nbit)
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

        for i in range(nbyte):
            print(bin(bs[i] + 256)[3:], end=' ')
        print()

        br = faiss.BitstringReader(swig_ptr(bs), nbyte)

        for nbit, xref in ctrl:
            xnew = br.read(nbit)
            print('nbit %d xref %x xnew %x' % (nbit, xref, xnew))
            self.assertTrue(xnew == xref)
