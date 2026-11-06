# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""make sure that the referenced objects are kept"""

import numpy as np
import unittest
import faiss
import sys
import gc
import weakref

d = 10
xt = np.random.rand(100, d).astype("float32")
xb = np.random.rand(20, d).astype("float32")


class TestReferenced(unittest.TestCase):

    def test_SearchParameters_native_clone_not_exposed(self):
        # Native shallow copies do not carry Python field owners. Keep that
        # operation internal instead of exposing a copy with dangling fields.
        for name, cls in vars(faiss).items():
            if (isinstance(cls, type) and
                    issubclass(cls, faiss.SearchParameters)):
                with self.subTest(parameter_type=name):
                    self.assertFalse(hasattr(cls, "clone"))

    def test_SearchParameters_wrapper_selector_gc(self):
        # Internal per-call copies borrow fields while the original parameter
        # tree keeps the Python owners alive, including after explicit GC.
        for refine in (False, True):
            with self.subTest(refine=refine):
                base = faiss.IndexFlatL2(1)
                wrapper = (faiss.IndexRefineFlat(base) if refine else
                           faiss.IndexPreTransform(base))
                index = faiss.IndexIDMap(wrapper)
                xb = np.arange(4, dtype="float32").reshape(-1, 1)
                ids = np.array([30, 10, 40, 20], dtype="int64")
                index.add_with_ids(xb, ids)
                outer = faiss.IDSelectorRange(10, 21)
                inner = faiss.IDSelectorRange(15, 31)
                leaf = faiss.SearchParameters(sel=inner)
                params = (faiss.IndexRefineSearchParameters(
                    sel=outer, base_index_params=leaf) if refine else
                    faiss.SearchParametersPreTransform(
                        sel=outer, index_params=leaf))
                owners = [weakref.ref(obj) for obj in (outer, inner, leaf)]
                del outer, inner, leaf
                gc.collect()
                self.assertTrue(all(ref() is not None for ref in owners))
                for _ in range(2):
                    D, I = index.search(xb[:1], 2, params=params)
                    np.testing.assert_array_equal(I, [[20, -1]])
                    self.assertEqual(D[0, 0], 9)
                    lims, distances, labels = index.range_search(
                        xb[:1], 10, params=params)
                    np.testing.assert_array_equal(lims, [0, 1])
                    np.testing.assert_array_equal(labels, [20])
                    np.testing.assert_array_equal(distances, [9])
                    gc.collect()
                    self.assertTrue(all(ref() is not None for ref in owners))
                del params
                gc.collect()
                self.assertTrue(all(ref() is None for ref in owners))

    def test_IndexIVF(self):
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 10)
        index.train(xt)
        index.add(xb)
        del quantizer
        gc.collect()
        index.add(xb)

    def test_count_refs(self):
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 10)
        refc1 = sys.getrefcount(quantizer)
        del index
        gc.collect()
        refc2 = sys.getrefcount(quantizer)
        assert refc2 == refc1 - 1

    def test_IndexPreTransform(self):
        ltrans = faiss.NormalizationTransform(d)
        sub_index = faiss.IndexFlatL2(d)
        index = faiss.IndexPreTransform(ltrans, sub_index)
        index.add(xb)
        del ltrans
        gc.collect()
        index.add(xb)
        del sub_index
        gc.collect()
        index.add(xb)

    def test_IndexPreTransform_2(self):
        sub_index = faiss.IndexFlatL2(d)
        index = faiss.IndexPreTransform(sub_index)
        ltrans = faiss.NormalizationTransform(d)
        index.prepend_transform(ltrans)
        index.add(xb)
        del ltrans
        gc.collect()
        index.add(xb)
        del sub_index
        gc.collect()
        index.add(xb)

    def test_IDMap(self):
        sub_index = faiss.IndexFlatL2(d)
        index = faiss.IndexIDMap(sub_index)
        index.add_with_ids(xb, np.arange(len(xb), dtype="int64"))
        del sub_index
        gc.collect()
        index.add_with_ids(xb, np.arange(len(xb), dtype="int64"))

    def test_shards(self):
        index = faiss.IndexShards(d)
        for _i in range(3):
            sub_index = faiss.IndexFlatL2(d)
            sub_index.add(xb)
            index.add_shard(sub_index)
        gc.collect()
        index.search(xb, 10)

    def test_SearchParameters_setattr_no_leak(self):
        # Regression: assigning IDSelectorBatch to params.sel after
        # construction must not leak. Before the fix, the SWIG property
        # setter flipped IDSelectorBatch.thisown to 0 but
        # SearchParameters never freed it -> ~80 MB orphaned per iter.
        n_ids = 5_000_000
        n_iters = 10

        def body():
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(np.arange(n_ids))
            del params

        body()  # warmup: prime allocator
        gc.collect()
        start_kb = faiss.get_mem_usage_kb()
        for _ in range(n_iters):
            body()
            gc.collect()
        growth_mb = (faiss.get_mem_usage_kb() - start_kb) / 1024.0
        # One leaked IDSelectorBatch is ~80 MB; 10 iters would leak
        # ~800 MB. 200 MB cap leaves generous headroom for allocator
        # noise but is well below the leak signal.
        self.assertLess(growth_mb, 200)

    def test_SearchParameters_reassignment_no_accumulation(self):
        # reassigning a field must release the previous selector
        params = faiss.SearchParameters()
        sel1 = faiss.IDSelectorBatch(np.arange(10))
        sel2 = faiss.IDSelectorBatch(np.arange(10))
        before1 = sys.getrefcount(sel1)
        params.sel = sel1
        params.sel = sel2
        self.assertEqual(sys.getrefcount(sel1), before1)

    def test_SearchParameters_setattr_else_branch(self):
        # None drops the ref of that field only
        params = faiss.SearchParametersIVF()
        sel = faiss.IDSelectorBatch(np.arange(10))
        quant = faiss.SearchParameters()
        sel_before = sys.getrefcount(sel)
        quant_before = sys.getrefcount(quant)

        params.sel = sel
        params.quantizer_params = quant
        self.assertEqual(sys.getrefcount(sel) - sel_before, 1)
        self.assertEqual(sys.getrefcount(quant) - quant_before, 1)

        params.sel = None
        self.assertEqual(sys.getrefcount(sel), sel_before)
        self.assertEqual(sys.getrefcount(quant) - quant_before, 1)


dbin = 32
xtbin = np.random.randint(256, size=(100, int(dbin / 8))).astype("uint8")
xbbin = np.random.randint(256, size=(20, int(dbin / 8))).astype("uint8")


class TestReferencedBinary(unittest.TestCase):

    def test_binary_ivf(self):
        index = faiss.IndexBinaryIVF(faiss.IndexBinaryFlat(dbin), dbin, 10)
        gc.collect()
        index.train(xtbin)

    def test_wrap(self):
        index = faiss.IndexBinaryFromFloat(faiss.IndexFlatL2(dbin))
        gc.collect()
        index.add(xbbin)


if __name__ == "__main__":
    unittest.main()
