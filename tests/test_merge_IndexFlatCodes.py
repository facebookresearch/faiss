# Test merge_from method for all IndexFlatCodes Types
import unittest

import faiss
import numpy as np

D: int = 64
N_TOTAL: int = 200
M: int = 8  # for PQ: #subquantizers
NBITS_PER_INDEX: int = 4  # for PQ
XB: np.array = np.random.rand(N_TOTAL, D).astype('float32')


def do_test(index1: faiss.Index):
        index1.train(XB)
        index1.add(XB)
        _, Iref = index1.search(XB[50:150], 5)
        index1.reset()
        index2 = faiss.clone_index(index1)
        index1.add(XB[:100])
        index2.add(XB[100:])
        index1.merge_from(index2)
        _, Inew = index1.search(XB[50:150], 5)
        assert (Iref == Inew).all()


class IndexFlatCodes_merge(unittest.TestCase):

    def test_merge_IndexFlat(self):
        do_test(faiss.IndexFlat(D))

    def test_merge_IndexPQ(self):
        do_test(faiss.IndexPQ(D, M, NBITS_PER_INDEX))

    def test_merge_IndexLSH(self):
        do_test(faiss.IndexLSH(D, NBITS_PER_INDEX))

    def test_merge_IndexScalarQuantizer(self):
        do_test(faiss.IndexScalarQuantizer(D, NBITS_PER_INDEX))
