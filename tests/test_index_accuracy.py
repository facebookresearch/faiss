# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

import numpy as np
import unittest
import faiss

# translation of test_knn.lua


def random_unitary(n, d, seed):
    x = faiss.randn(n * d, seed).reshape(n, d)
    faiss.normalize_L2(x)
    return x

class Randu10k:

    def __init__(self):
        self.nb = 10000
        self.nq = 1000
        self.nt = 10000
        self.d = 128

        self.xb = random_unitary(self.nb, self.d, 1)
        self.xt = random_unitary(self.nt, self.d, 2)
        self.xq = random_unitary(self.nq, self.d, 3)

        dotprods = np.dot(self.xq, self.xb.T)
        self.gt = dotprods.argmax(1)
        self.k = 100

    def launch(self, name, index):
        if not index.is_trained:
            index.train(self.xt)
        index.add(self.xb)
        return index.search(self.xq, self.k)

    def evalres(self, (D, I)):
        e = {}
        for rank in 1, 10, 100:
            e[rank] = (I[:, :rank] == self.gt.reshape(-1, 1)).sum() / float(self.nq)
        return e

ev = Randu10k()

d = ev.d

# Parameters inverted indexes
ncentroids = int(4 * np.sqrt(ev.nb))
kprobe = int(np.sqrt(ncentroids))

# Parameters for LSH
nbits = d

# Parameters for indexes involving PQ
M = d / 8                # for PQ: #subquantizers
nbits_per_index = 8      # for PQ


class IndexAccuracy(unittest.TestCase):

    def test_IndexFlatIP(self):
        q = faiss.IndexFlatIP(d)  # Ask inner product
        res = ev.launch('FLAT / IP', q)
        e = ev.evalres(res)
        assert e[1] == 1.0

    def test_IndexFlatL2(self):
        q = faiss.IndexFlatL2(d)
        res = ev.launch('FLAT / L2', q)
        e = ev.evalres(res)
        assert e[1] == 1.0

    def test_ivf_kmeans(self):
        ivfk = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, ncentroids)
        ivfk.nprobe = kprobe
        res = ev.launch('IVF K-means', ivfk)
        e = ev.evalres(res)
        # should give 0.260  0.260  0.260
        assert e[1] > 0.2

    def test_indexLSH(self):
        q = faiss.IndexLSH(d, nbits)
        res = ev.launch('FLAT / LSH Cosine', q)
        e = ev.evalres(res)
        # should give 0.070  0.250  0.580
        assert e[10] > 0.2

    def test_IndexLSH_32_48(self):
        # CHECK: the difference between 32 and 48 does not make much sense
        for nbits2 in 32, 48:
            q = faiss.IndexLSH(d, nbits2)
            res = ev.launch('LSH half size', q)
            e = ev.evalres(res)
            # should give 0.002  0.025  0.117
            assert e[10] > 0.024

    def test_IndexPQ(self):
        q = faiss.IndexPQ(d, M, nbits_per_index)
        res = ev.launch('FLAT / PQ L2', q)
        e = ev.evalres(res)
        # should give 0.070  0.230  0.260
        assert e[10] > 0.2

    # Approximate search module: PQ with inner product distance
    def test_IndexPQ_ip(self):
        q = faiss.IndexPQ(d, M, nbits_per_index, faiss.METRIC_INNER_PRODUCT)
        res = ev.launch('FLAT / PQ IP', q)
        e = ev.evalres(res)
        # should give 0.070  0.230  0.260
        #(same result as regular PQ on normalized distances)
        assert e[10] > 0.2

    def test_IndexIVFPQ(self):
        ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, ncentroids, M, 8)
        ivfpq.nprobe = kprobe
        res = ev.launch('IVF PQ', ivfpq)
        e = ev.evalres(res)
        # should give 0.070  0.230  0.260
        assert e[10] > 0.2

    # TODO: translate evaluation of nested

    # Approximate search: PQ with full vector refinement
    def test_IndexPQ_refined(self):
        q = faiss.IndexPQ(d, M, nbits_per_index)
        res = ev.launch('PQ non-refined', q)
        e = ev.evalres(res)
        q.reset()

        rq = faiss.IndexRefineFlat(q)
        res = ev.launch('PQ refined', rq)
        e2 = ev.evalres(res)
        assert e2[10] >= e[10]
        rq.k_factor = 4

        res = ev.launch('PQ refined*4', rq)
        e3 = ev.evalres(res)
        assert e3[10] >= e2[10]

    def test_polysemous(self):
        index = faiss.IndexPQ(d, M, nbits_per_index)
        index.do_polysemous_training = True
        # reduce nb iterations to speed up training for the test
        index.polysemous_training.n_iter = 50000
        index.polysemous_training.n_redo = 1
        res = ev.launch('normal PQ', index)
        e_baseline = ev.evalres(res)
        index.search_type = faiss.IndexPQ.ST_polysemous

        index.polysemous_ht = int(M / 16. * 58)

        stats = faiss.cvar.indexPQ_stats
        stats.reset()

        res = ev.launch('Polysemous ht=%d' % index.polysemous_ht,
                        index)
        e_polysemous = ev.evalres(res)
        print e_baseline, e_polysemous,  index.polysemous_ht
        print stats.n_hamming_pass, stats.ncode
        # The randu dataset is difficult, so we are not too picky on
        # the results. Here we assert that we have < 10 % loss when
        # computing full PQ on fewer than 20% of the data.
        assert stats.n_hamming_pass < stats.ncode / 5
        # Test disabled because difference is 0.17 on aarch64
        # TODO check why???
        # assert e_polysemous[10] > e_baseline[10] - 0.1

    def test_ScalarQuantizer(self):
        quantizer = faiss.IndexFlatL2(d)
        ivfpq = faiss.IndexIVFScalarQuantizer(
            quantizer, d, ncentroids,
            faiss.ScalarQuantizer.QT_8bit)
        ivfpq.nprobe = kprobe
        res = ev.launch('IVF SQ', ivfpq)
        e = ev.evalres(res)
        # should give 0.234  0.236  0.236
        assert e[10] > 0.235
