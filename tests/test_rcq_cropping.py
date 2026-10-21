# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RCQ cropping tests. Ported from test_RCQ_cropping.cpp."""

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


class TestRCQCropping(unittest.TestCase):
    def test_search_params(self):
        nq = 10
        nt = 2000
        nb = 1000
        d = 32

        buf = faiss.rand_smooth_vectors(nq + nb + nt, d)
        xt = buf[:nt]
        xb = buf[nt : nt + nb]
        xq = buf[nt + nb :]

        nbits = faiss.UInt64Vector()
        nbits.push_back(3)
        nbits.push_back(6)
        nbits.push_back(3)
        quantizer = faiss.ResidualCoarseQuantizer(d, nbits)
        ntotal = 1 << quantizer.rq.tot_bits
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, ntotal, faiss.ScalarQuantizer.QT_8bit
        )
        index.quantizer_trains_alone = True

        index.train(xt)
        index.add(xb)

        index.nprobe = 10

        k = 4
        beam_factor_1 = 8.0
        quantizer.set_beam_factor(beam_factor_1)
        D1, I1 = index.search(xq, k)

        # change from 8 to 1
        quantizer.set_beam_factor(1.0)
        D2, I2 = index.search(xq, k)

        # make sure it changes the result
        self.assertFalse(np.array_equal(I1, I2))
        self.assertFalse(np.array_equal(D1, D2))

        # override the class level beam factor
        params1 = faiss.SearchParametersResidualCoarseQuantizer()
        params1.beam_factor = beam_factor_1
        params = faiss.SearchParametersIVF()
        params.nprobe = index.nprobe
        params.quantizer_params = params1

        D3, I3 = index.search(xq, k, params=params)

        # make sure we find back the original results
        np.testing.assert_array_equal(I1, I3)
        np.testing.assert_array_equal(D1, D3)

    def test_cropping(self):
        nq = 10
        nt = 2000
        nb = 1000
        d = 32

        buf = faiss.rand_smooth_vectors(nq + nb + nt, d, 1234)
        xt = buf[:nt]
        xq = buf[nt + nb :]

        nbits = [5, 4, 4]
        nbits_vec = faiss.UInt64Vector()
        for nb_i in nbits:
            nbits_vec.push_back(nb_i)
        rcq = faiss.ResidualCoarseQuantizer(d, nbits_vec)

        rcq.train(xt)

        # the test below works only for beam size == nprobe
        rcq.set_beam_factor(1.0)

        nprobe = 15
        Dref, Iref = rcq.search(xq, nprobe)

        # crop to the first 2 quantization levels
        last_nbits = nbits.pop()
        nbits_vec = faiss.UInt64Vector()
        for nb_i in nbits:
            nbits_vec.push_back(nb_i)
        rcq_cropped = faiss.ResidualCoarseQuantizer(d, nbits_vec)
        rcq_cropped.initialize_from(rcq)

        self.assertEqual(rcq_cropped.ntotal, rcq.ntotal >> last_nbits)

        Dnew, Inew = rcq_cropped.search(xq, nprobe)

        # these bits are in common between the two RCQs
        mask = (1 << rcq_cropped.rq.tot_bits) - 1
        for q in range(nq):
            for i in range(nprobe):
                fine = int(Iref[q, i])
                self.assertGreaterEqual(fine, 0)
                # fine should be generated from a path that passes
                # through coarse
                found = any(
                    (fine & mask) == Inew[q, j] for j in range(nprobe)
                )
                self.assertTrue(found)
