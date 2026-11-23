# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import faiss
import numpy as np


class TestSWIGWrap(unittest.TestCase):

    def test_id_map_at(self):
        # issue 1020
        n_features = 100
        feature_dims = 10

        features = np.random.random((n_features, feature_dims)).astype(
            np.float32
        )
        idx = np.arange(n_features).astype(np.int64)

        index = faiss.IndexFlatL2(feature_dims)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(features, idx)

        [index.id_map.at(int(i)) for i in range(index.ntotal)]

    def test_downcast_Refine(self):

        index = faiss.IndexRefineFlat(
            faiss.IndexScalarQuantizer(10, faiss.ScalarQuantizer.QT_8bit)
        )

        index2 = faiss.deserialize_index(faiss.serialize_index(index))

        assert isinstance(index2, faiss.IndexRefineFlat)

        index3 = faiss.deserialize_index(faiss.serialize_index(index2))
        assert isinstance(index3, faiss.IndexRefineFlat)

    def do_test_array_type(self, dtype):
        """tests swig_ptr and rev_swig_ptr for this type of array"""
        a = np.arange(12).astype(dtype)
        ptr = faiss.swig_ptr(a)
        a2 = faiss.rev_swig_ptr(ptr, 12)
        np.testing.assert_array_equal(a, a2)

    def test_all_array_types(self):
        self.do_test_array_type("float32")
        self.do_test_array_type("float64")
        self.do_test_array_type("int8")
        self.do_test_array_type("uint8")
        self.do_test_array_type("int16")
        self.do_test_array_type("uint16")
        self.do_test_array_type("int32")
        self.do_test_array_type("uint32")
        self.do_test_array_type("int64")
        self.do_test_array_type("uint64")

    def test_int64(self):
        # see https://github.com/facebookresearch/faiss/issues/1529
        v = faiss.Int64Vector()

        for i in range(10):
            v.push_back(i)
        a = faiss.vector_to_array(v)
        assert a.dtype == "int64"
        np.testing.assert_array_equal(a, np.arange(10, dtype="int64"))

    def test_SWIG_version(self):
        self.assertLess(faiss.swig_version(), 0x050000)

    def test_attribute_validation(self):
        index = faiss.index_factory(256, "OPQ64,IVF16384,PQ64")

        with self.assertRaises(AttributeError) as cm:
            index.nprobe = 16

        error_msg = str(cm.exception)
        self.assertIn("IndexPreTransform", error_msg)
        self.assertIn("nprobe", error_msg)

        index.verbose = True
        self.assertEqual(index.verbose, True)

    def test_ivfsq_turboq_search_parameters(self):
        params = faiss.IVFSQTurboQSearchParameters()
        self.assertEqual(params.qb, 0)
        self.assertFalse(params.int_qjl)

    def test_scalar_quantizer_qtype_values(self):
        # qtype is serialized as an int: these values must not change
        sq = faiss.ScalarQuantizer
        self.assertEqual(sq.QT_0bit, 9)
        self.assertEqual(sq.QT_2bit_tq, 15)
        self.assertEqual(sq.QT_3bit_tq, 16)
        self.assertEqual(sq.QT_4bit_tq, 17)
        self.assertEqual(sq.QT_5bit_tq, 18)


class TestRevSwigPtr(unittest.TestCase):

    def test_rev_swig_ptr(self):

        index = faiss.IndexFlatL2(4)
        xb0 = np.vstack(
            [i * 10 + np.array([1, 2, 3, 4], dtype="float32") for i in range(5)]
        )
        index.add(xb0)
        xb = faiss.rev_swig_ptr(index.get_xb(), 4 * 5).reshape(5, 4)
        self.assertEqual(np.abs(xb0 - xb).sum(), 0)


class TestException(unittest.TestCase):

    def test_exception(self):

        index = faiss.IndexFlatL2(10)

        a = np.zeros((5, 10), dtype="float32")
        b = np.zeros(5, dtype="int64")

        # an unsupported operation for IndexFlat
        self.assertRaises(RuntimeError, index.add_with_ids, a, b)

    def test_exception_2(self):
        self.assertRaises(
            RuntimeError, faiss.index_factory, 12, "IVF256,Flat,PQ8"
        )


class TestAddSACodes(unittest.TestCase):
    """Regression tests for add_sa_codes and sa_encode wrapper correctness."""

    D = 32
    NB = 200

    def setUp(self):
        rng = np.random.default_rng(42)
        self.xb = rng.random((self.NB, self.D), dtype=np.float32)
        self.xtrain = rng.random((500, self.D), dtype=np.float32)
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

    def test_sa_encode_preallocated_matches_auto_allocated(self):
        """sa_encode into a caller buffer returns that buffer with the same
        codes."""
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


@unittest.skipIf(
    faiss.swig_version() < 0x040000,
    "swig < 4 does not support Doxygen comments",
)
class TestDoxygen(unittest.TestCase):

    def test_doxygen_comments(self):
        maxheap_array = faiss.float_maxheap_array_t()

        self.assertTrue(
            "a template structure for a set of [min|max]-heaps"
            in maxheap_array.__doc__
        )


class TestMapLong2Long(unittest.TestCase):
    def test_add_and_search_multiple_coerce_int32(self):
        m = faiss.MapLong2Long()
        keys = np.arange(5, dtype=np.int32)
        vals = np.arange(5, dtype=np.int32) + 300
        m.add(keys, vals)
        result = m.search_multiple(keys)
        np.testing.assert_array_equal(result, vals)


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


class TestNumpyScalarInputs(unittest.TestCase):

    @staticmethod
    def integer_types():
        names = (
            'byte', 'ubyte', 'short', 'ushort', 'intc', 'uintc',
            'int_', 'uint', 'long', 'ulong', 'longlong', 'ulonglong',
            'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
            'int64', 'uint64', 'intp', 'uintp',
        )
        return tuple(dict.fromkeys(
            np.__dict__[name] for name in names
            if name in np.__dict__ and
            issubclass(np.__dict__[name], np.integer)
        ))

    def test_integer_scalar_families(self):
        for scalar in self.integer_types():
            with self.subTest(scalar=scalar):
                for kind in ('Int8', 'Int16', 'Int32', 'Int64',
                             'UInt8', 'UInt16', 'UInt32', 'UInt64'):
                    vector = getattr(faiss, kind + 'Vector')()
                    vector.push_back(scalar(4))
                    self.assertEqual(vector.at(np.uint64(0)), 4)
                self.assertEqual(faiss.IndexFlatL2(scalar(4)).d, 4)
                self.assertEqual(faiss.IndexFlatIP(scalar(4)).d, 4)

    def test_integer_boundaries(self):
        for kind, dtype in (
            ('Int8', np.int8), ('Int16', np.int16),
            ('Int32', np.int32), ('Int64', np.int64),
            ('UInt8', np.uint8), ('UInt16', np.uint16),
            ('UInt32', np.uint32), ('UInt64', np.uint64),
        ):
            limits = np.iinfo(dtype)
            with self.subTest(kind=kind):
                vector = getattr(faiss, kind + 'Vector')()
                for value in (limits.min, limits.max):
                    vector.push_back(dtype(value))
                    self.assertEqual(vector.at(vector.size() - 1), value)
                for value in (limits.min - 1, limits.max + 1):
                    candidates = [value]
                    for scalar in self.integer_types():
                        source_limits = np.iinfo(scalar)
                        if source_limits.min <= value <= source_limits.max:
                            candidates.append(scalar(value))
                    # Overloaded SWIG dispatch may report TypeError for a
                    # range mismatch. Match the stock Python-int outcome.
                    with self.assertRaises(
                            (TypeError, OverflowError)) as caught:
                        vector.push_back(value)
                    for bad in candidates:
                        with self.assertRaises(type(caught.exception)):
                            vector.push_back(bad)
                        self.assertEqual(vector.size(), 2)
                vector.push_back(dtype(1))
                self.assertEqual(vector.at(2), 1)

    def test_float_scalar_families_and_ranges(self):
        for kind in ('Float32', 'Float64'):
            for scalar in (np.float16, np.float32, np.float64):
                with self.subTest(kind=kind, scalar=scalar):
                    vector = getattr(faiss, kind + 'Vector')()
                    for value in (1.25, float('nan'), float('inf'),
                                  -float('inf')):
                        vector.push_back(scalar(value))
                        actual = vector.at(vector.size() - 1)
                        if np.isnan(value):
                            self.assertTrue(np.isnan(actual))
                        else:
                            self.assertEqual(actual, value)
                    for value in (float(np.finfo(np.float32).max) * 2,
                                  -float(np.finfo(np.float32).max) * 2):
                        def outcome(x):
                            v = getattr(faiss, kind + 'Vector')()
                            try:
                                v.push_back(x)
                                return ('value', v.at(0))
                            except (TypeError, OverflowError) as exc:
                                self.assertEqual(v.size(), 0)
                                return ('error', type(exc))
                        self.assertEqual(outcome(np.float64(value)),
                                         outcome(float(value)))

    def test_members_and_search(self):
        params = faiss.ClusteringParameters()
        params.niter = np.int32(3)
        self.assertEqual(params.niter, 3)
        params.early_stop_threshold = np.float32(0.125)
        self.assertEqual(params.early_stop_threshold, 0.125)
        stats = faiss.ClusteringIterationStats()
        stats.obj = np.float32(1.25)
        self.assertEqual(stats.obj, 1.25)
        selector = faiss.IDSelectorRange(1, 4)
        self.assertTrue(selector.is_member(np.int64(2)))
        self.assertFalse(selector.is_member(np.int64(4)))
        index = faiss.IndexFlatL2(2)
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        index.add(x)
        d, ids = index.search(x[:1], np.int64(1))
        np.testing.assert_array_equal(ids, [[0]])
        np.testing.assert_array_equal(d, [[0]])

    def test_scalar_pointer_overload(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        out = np.zeros(3, dtype=np.float32)
        for scalar in (np.float16, np.float32, np.float64):
            faiss.fvec_add(3, faiss.swig_ptr(a), scalar(2.5),
                           faiss.swig_ptr(out))
            np.testing.assert_array_equal(out, [3.5, 4.5, 5.5])
        out[:] = 99
        with self.assertRaises((TypeError, OverflowError)) as caught:
            faiss.fvec_add(3, faiss.swig_ptr(a), 1e300,
                           faiss.swig_ptr(out))
        with self.assertRaises(type(caught.exception)):
            faiss.fvec_add(3, faiss.swig_ptr(a), np.float64(1e300),
                           faiss.swig_ptr(out))
        np.testing.assert_array_equal(out, [99, 99, 99])
        faiss.fvec_add(3, faiss.swig_ptr(a), faiss.swig_ptr(b),
                       faiss.swig_ptr(out))
        np.testing.assert_array_equal(out, [5, 7, 9])

    def test_integer_rejection_controls(self):
        class Indexable:
            def __index__(self):
                return 4

        class RaisingInteger(np.int64):
            def __index__(self):
                raise RuntimeError('not a builtin NumPy scalar')

        # These are current stock-converter boundaries, not added support.
        for value in (np.bool_(True), np.float32(4), np.float64(4),
                      np.array(4), np.array([4]), np.complex64(4),
                      Indexable(), RaisingInteger(4)):
            with self.subTest(value=type(value)):
                vector = faiss.Int32Vector()
                with self.assertRaises(TypeError):
                    vector.push_back(value)
                self.assertEqual(vector.size(), 0)
                vector.push_back(4)
                self.assertEqual(vector.at(0), 4)

    def test_enum_bool_controls(self):
        # Enums and bool deliberately retain the current stock mappings.
        index = faiss.IndexFlat(4, faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)
        for value in (np.int64(faiss.METRIC_L2), np.int32(faiss.METRIC_L2)):
            with self.assertRaises(TypeError):
                faiss.IndexFlat(4, value)
            with self.assertRaises(TypeError):
                index.metric_type = value
            self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)
        selector = faiss.IDSelectorRange(0, 4, True)
        self.assertTrue(selector.assume_sorted)
        with self.assertRaises(TypeError):
            faiss.IDSelectorRange(0, 4, np.bool_(True))

    def test_size_t_and_member_boundaries(self):
        vector = faiss.Int32Vector()
        vector.push_back(7)
        self.assertEqual(vector.at(np.uint64(0)), 7)
        for value in (-1, np.int64(-1), 2 ** 64):
            with self.assertRaises(OverflowError):
                vector.at(value)
        params = faiss.ClusteringParameters()
        for value in (np.int64(3), np.uint64(3)):
            params.decode_block_size = value
            self.assertEqual(params.decode_block_size, 3)
        for value in (-1, np.int64(-1), 2 ** 64):
            with self.assertRaises(OverflowError):
                params.decode_block_size = value
            self.assertEqual(params.decode_block_size, 3)
        for value in (2 ** 40, np.int64(2 ** 40)):
            params.niter = 3
            with self.assertRaises(OverflowError):
                params.niter = value
            self.assertEqual(params.niter, 3)
        params.niter = np.int64(-4)
        self.assertEqual(params.niter, -4)

    def test_float_and_stl_boundaries(self):
        class Floatable:
            def __float__(self):
                return 1.25

        class CompatibleFloat(np.float64):
            def __float__(self):
                raise RuntimeError('stock float subclass needs no protocol')

        class RaisingFloat(np.float32):
            def __float__(self):
                raise RuntimeError('not a builtin NumPy scalar')

        # Stock traits and non-target inputs are deliberately unchanged.
        for kind in ('Float32', 'Float64'):
            vector = getattr(faiss, kind + 'Vector')()
            vector.push_back(4)
            self.assertEqual(vector.at(0), 4.0)
            for value in (np.int32(4), np.int64(4), np.uint64(4),
                          RaisingFloat(1.25), Floatable(), np.bool_(True),
                          np.array(1.25), np.array([1.25]),
                          np.complex64(1.25)):
                with self.subTest(kind=kind, value=type(value)):
                    with self.assertRaises(TypeError):
                        vector.push_back(value)
                    self.assertEqual(vector.size(), 1)
            if np.longdouble not in (np.float16, np.float32, np.float64):
                with self.assertRaises(TypeError):
                    vector.push_back(np.longdouble(1.25))
            vector.push_back(np.float32(1.25))
            self.assertEqual(vector.at(1), 1.25)
            # np.float64 inherits Python float acceptance in stock SWIG.
            vector.push_back(CompatibleFloat(1.25))
            self.assertEqual(vector.at(2), 1.25)
        # These vector wrappers do not accept Python sequence constructors,
        # with ordinary or NumPy elements. Do not broaden STL traits here.
        for values in ([1, 2], [np.int32(1), np.int32(2)]):
            with self.assertRaises(TypeError):
                faiss.Int32Vector(values)
