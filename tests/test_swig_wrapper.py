# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# a few tests of the swig wrapper 

import unittest
import faiss
import numpy as np


class TestSWIGWrap(unittest.TestCase):
    """ various regressions with the SWIG wrapper """

    def test_size_t_ptr(self):
        # issue 1064
        index = faiss.IndexHNSWFlat(10, 32)

        hnsw = index.hnsw
        index.add(np.random.rand(100, 10).astype('float32'))
        be = np.empty(2, 'uint64')
        hnsw.neighbor_range(23, 0, faiss.swig_ptr(be), faiss.swig_ptr(be[1:]))

    def test_id_map_at(self):
        # issue 1020
        n_features = 100
        feature_dims = 10

        features = np.random.random((n_features, feature_dims)).astype(np.float32)
        idx = np.arange(n_features).astype(np.int64)

        index = faiss.IndexFlatL2(feature_dims)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(features, idx)

        [index.id_map.at(int(i)) for i in range(index.ntotal)]

    def test_downcast_Refine(self):

        index = faiss.IndexRefineFlat(
            faiss.IndexScalarQuantizer(10, faiss.ScalarQuantizer.QT_8bit)
        )

        # serialize and deserialize
        index2 = faiss.deserialize_index(
            faiss.serialize_index(index)
        )

        assert isinstance(index2, faiss.IndexRefineFlat)

    def do_test_array_type(self, dtype):
        """ tests swig_ptr and rev_swig_ptr for this type of array """
        a = np.arange(12).astype(dtype)
        ptr = faiss.swig_ptr(a)
        a2 = faiss.rev_swig_ptr(ptr, 12)
        np.testing.assert_array_equal(a, a2)

    def test_all_array_types(self):
        self.do_test_array_type('float32')
        self.do_test_array_type('float64')
        self.do_test_array_type('int8')
        self.do_test_array_type('uint8')
        self.do_test_array_type('int16')
        self.do_test_array_type('uint16')
        self.do_test_array_type('int32')
        self.do_test_array_type('uint32')
        self.do_test_array_type('int64')
        self.do_test_array_type('uint64')

    def test_int64(self):
        # see https://github.com/facebookresearch/faiss/issues/1529
        v = faiss.Int64Vector()

        for i in range(10):
            v.push_back(i)
        a = faiss.vector_to_array(v)
        assert a.dtype == 'int64'
        np.testing.assert_array_equal(a, np.arange(10, dtype='int64'))

        # check if it works in an IDMap
        idx = faiss.IndexIDMap(faiss.IndexFlatL2(32))
        idx.add_with_ids(
            np.random.rand(10, 32).astype('float32'),
            np.random.randint(1000, size=10, dtype='int64')
        )
        faiss.vector_to_array(idx.id_map)

    def test_asan(self): 
        # this test should fail with ASAN
        index = faiss.IndexFlatL2(32)
        index.this.own(False)   # this is a mem leak, should be catched by ASAN

    def test_SWIG_version(self): 
        self.assertLess(faiss.swig_version(), 0x050000)


class TestRevSwigPtr(unittest.TestCase):

    def test_rev_swig_ptr(self):

        index = faiss.IndexFlatL2(4)
        xb0 = np.vstack([
            i * 10 + np.array([1, 2, 3, 4], dtype='float32')
            for i in range(5)])
        index.add(xb0)
        xb = faiss.rev_swig_ptr(index.get_xb(), 4 * 5).reshape(5, 4)
        self.assertEqual(np.abs(xb0 - xb).sum(), 0)


class TestException(unittest.TestCase):

    def test_exception(self):

        index = faiss.IndexFlatL2(10)

        a = np.zeros((5, 10), dtype='float32')
        b = np.zeros(5, dtype='int64')

        # an unsupported operation for IndexFlat
        self.assertRaises(
            RuntimeError,
            index.add_with_ids, a, b
        )
        # assert 'add_with_ids not implemented' in str(e)

    def test_exception_2(self):
        self.assertRaises(
            RuntimeError,
            faiss.index_factory, 12, 'IVF256,Flat,PQ8'
        )
        #    assert 'could not parse' in str(e)


@unittest.skipIf(faiss.swig_version() < 0x040000, "swig < 4 does not support Doxygen comments")
class TestDoxygen(unittest.TestCase):

    def test_doxygen_comments(self):
        maxheap_array = faiss.float_maxheap_array_t()

        self.assertTrue("a template structure for a set of [min|max]-heaps"
                        in maxheap_array.__doc__)
