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

        # Verify deserialized index is serializable again
        index3 = faiss.deserialize_index(
            faiss.serialize_index(index2)
        )
        assert isinstance(index3, faiss.IndexRefineFlat)

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

    def test_attribute_validation(self):
        """Test that setting invalid attributes raises AttributeError"""

        # Test IndexPreTransform - the main use case from the issue
        index = faiss.index_factory(256, "OPQ64,IVF16384,PQ64")

        # Should raise AttributeError when trying to set nprobe directly on
        # wrapper
        with self.assertRaises(AttributeError) as cm:
            index.nprobe = 16

        # Check that the error message contains basic information
        error_msg = str(cm.exception)
        self.assertIn("IndexPreTransform", error_msg)
        self.assertIn("nprobe", error_msg)

        # Test with other IVF parameters that should be blocked
        with self.assertRaises(AttributeError) as cm:
            index.nlist = 8192

        # Valid attributes should still work
        index.verbose = True  # This should be allowed
        self.assertEqual(index.verbose, True)

    def test_attribute_validation_other_indexes(self):
        """Test that other index types allow normal attribute setting"""

        # Test with a regular IndexFlat - should allow most attributes now
        index = faiss.IndexFlatL2(10)

        # Valid attributes should work
        index.verbose = False
        self.assertEqual(index.verbose, False)

    def test_rabitq(self):
        """Test that other index types allow normal attribute setting"""

        # Test with a regular IndexFlat - should allow most attributes now
        index = faiss.IndexRaBitQ(10)
        # Valid attributes should work
        index.qb = 4
        index.centered = True

        with self.assertRaises(AttributeError):
            index.centered2 = False

    def test_ivfsq_turboq_search_parameters(self):
        params = faiss.IVFSQTurboQSearchParameters()
        self.assertEqual(params.qb, 0)
        self.assertFalse(params.int_qjl)
        params.qb = 4
        params.int_qjl = True
        self.assertEqual(params.qb, 4)
        self.assertTrue(params.int_qjl)


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


class TestAddSACodes(unittest.TestCase):
    """Regression tests for add_sa_codes and sa_encode wrapper correctness."""

    D = 32
    NB = 200

    def setUp(self):
        rng = np.random.default_rng(42)
        self.xb = rng.random((self.NB, self.D), dtype=np.float32)
        self.xtrain = rng.random((500, self.D), dtype=np.float32)
        # IDMap2 wrapping PQ8x2: outer index stores (id → code) pairs;
        # inner PQ8x2 encodes vectors via sa_encode / add_sa_codes.
        self.index = faiss.index_factory(self.D, "IDMap2,PQ8x2")
        self.index.train(self.xtrain)

    def test_add_sa_codes_int32_ids_stored_correctly(self):
        """int32 ids must be coerced to int64 before the C++ boundary.

        Without coercion swig_ptr hands the C++ side a pointer to 4-byte
        int32 data; C++ interprets every 8 bytes as one idx_t, producing
        garbage ids with no exception."""
        codes = self.index.index.sa_encode(self.xb)

        ids_int64 = np.arange(self.NB, dtype="int64") + 1000
        ids_int32 = ids_int64.astype("int32")

        self.index.add_sa_codes(codes, ids_int32)

        stored = faiss.vector_to_array(self.index.id_map)
        np.testing.assert_array_equal(stored, ids_int64)

    def test_add_sa_codes_int64_ids_unchanged(self):
        """int64 ids must pass through as-is (np.ascontiguousarray is a no-op
        when the array is already int64 and contiguous)."""
        codes = self.index.index.sa_encode(self.xb)

        ids = np.arange(self.NB, dtype="int64") + 500
        self.index.add_sa_codes(codes, ids)

        stored = faiss.vector_to_array(self.index.id_map)
        np.testing.assert_array_equal(stored, ids)

    def test_sa_encode_preallocated_matches_auto_allocated(self):
        """sa_encode with a pre-allocated buffer must return the same codes
        as the auto-allocated path, exercising both branches of the hoisted
        sa_code_size() call."""
        inner = self.index.index
        codes_auto = inner.sa_encode(self.xb)

        buf = np.empty_like(codes_auto)
        codes_pre = inner.sa_encode(self.xb, buf)

        np.testing.assert_array_equal(codes_auto, codes_pre)
        self.assertIs(codes_pre, buf)


class TestSearchPreassignedDqNone(unittest.TestCase):
    """Dq=None must behave like Dq=zeros for search_preassigned and
    range_search_preassigned (regression: swig_ptr(None) raised ValueError)."""

    def _check(self, index, xq, Iq, k, thresh, zero_dtype):
        Dq_zero = np.zeros(Iq.shape, dtype=zero_dtype)
        _, I_zero = index.search_preassigned(xq, k, Iq, Dq_zero)
        _, I_none = index.search_preassigned(xq, k, Iq, None)
        np.testing.assert_array_equal(I_zero, I_none)
        lz, _, Iz = index.range_search_preassigned(xq, thresh, Iq, Dq_zero)
        ln, _, In = index.range_search_preassigned(xq, thresh, Iq, None)
        np.testing.assert_array_equal(lz, ln)
        np.testing.assert_array_equal(Iz, In)

    def test_float_ivf(self):
        rng = np.random.default_rng(42)
        xb = rng.random((200, 16), dtype=np.float32)
        xq = rng.random((10, 16), dtype=np.float32)
        index = faiss.index_factory(16, "IVF4,Flat")
        index.train(xb)
        index.add(xb)
        index.nprobe = 2
        _, Iq = index.quantizer.search(xq, 2)
        self._check(index, xq, Iq, k=5, thresh=2.0, zero_dtype=np.float32)

    def test_binary_ivf(self):
        rng = np.random.default_rng(42)
        xb = rng.integers(0, 256, size=(200, 8), dtype=np.uint8)
        xq = rng.integers(0, 256, size=(10, 8), dtype=np.uint8)
        index = faiss.IndexBinaryIVF(faiss.IndexBinaryFlat(64), 64, 4)
        index.train(xb)
        index.add(xb)
        index.nprobe = 2
        _, Iq = index.quantizer.search(xq, 2)
        self._check(index, xq, Iq, k=5, thresh=20, zero_dtype=np.int32)


@unittest.skipIf(faiss.swig_version() < 0x040000, "swig < 4 does not support Doxygen comments")
class TestDoxygen(unittest.TestCase):

    def test_doxygen_comments(self):
        maxheap_array = faiss.float_maxheap_array_t()

        self.assertTrue("a template structure for a set of [min|max]-heaps"
                        in maxheap_array.__doc__)


class TestMapLong2Long(unittest.TestCase):
    def test_add_coerces_keys_and_vals(self):
        for dtype in (np.int32, np.int64):
            with self.subTest(dtype=dtype):
                m = faiss.MapLong2Long()
                keys = np.array([0, 1, 2, 3, 4], dtype=dtype)
                vals = np.array([100, 101, 102, 103, 104], dtype=dtype)
                m.add(keys, vals)
                for k, v in zip(range(5), range(100, 105)):
                    self.assertEqual(m.search(k), v)

    def test_search_multiple_coerces_int32_keys(self):
        m = faiss.MapLong2Long()
        m.add(
            np.arange(5, dtype=np.int64),
            np.arange(5, dtype=np.int64) + 300,
        )
        result = m.search_multiple(np.array([0, 2, 4], dtype=np.int32))
        np.testing.assert_array_equal(result, [300, 302, 304])


class TestInvertedListsDowncast(unittest.TestCase):
    def test_downcast_ArrayInvertedListsPanorama(self):
        # downcast_InvertedLists() triggers the typemap; raw .invlists access
        # does not (typemap applies to function returns, not member getters).
        d, nlist, n_levels = 32, 8, 2
        index = faiss.IndexIVFFlatPanorama(
            faiss.IndexFlatL2(d), d, nlist, n_levels
        )
        il = faiss.downcast_InvertedLists(index.invlists)
        self.assertIsInstance(il, faiss.ArrayInvertedListsPanorama)
        self.assertEqual(il.n_levels, n_levels)

    def test_downcast_SliceInvertedLists(self):
        d, nlist = 4, 4
        # code_size = d * sizeof(float)
        backing = faiss.ArrayInvertedLists(nlist, d * 4)
        sil = faiss.SliceInvertedLists(backing, 0, nlist)
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
        # own=False: index doesn't delete sil; Python locals keep both alive.
        index.replace_invlists(sil, False)
        inner = faiss.downcast_InvertedLists(index.invlists)
        self.assertIsInstance(inner, faiss.SliceInvertedLists)
        self.assertEqual((inner.i0, inner.i1), (0, nlist))
