# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""make sure that the referenced objects are kept"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest
import faiss
import sys
import gc

d = 10
xt = np.random.rand(100, d).astype('float32')
xb = np.random.rand(20, d).astype('float32')


class TestReferenced(unittest.TestCase):

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

    def test_IndexIVF_2(self):
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 10)
        index.train(xt)
        index.add(xb)

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
        index.add_with_ids(xb, np.arange(len(xb), dtype='int64'))
        del sub_index
        gc.collect()
        index.add_with_ids(xb, np.arange(len(xb), dtype='int64'))

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

    def test_SearchParameters_subclass_no_double_wrap(self):
        # Regression: SWIG does not generate per-class __setattr__, so
        # the ownership-protecting override must be installed only on
        # the base SearchParameters; otherwise subclasses inherit and
        # re-wrap, doubling the refcount on the assigned selector.
        params = faiss.SearchParametersIVF()
        sel = faiss.IDSelectorBatch(np.arange(10))
        before = sys.getrefcount(sel)
        params.sel = sel
        delta = sys.getrefcount(sel) - before
        self.assertEqual(delta, 1)

    def test_SearchParameters_reassignment_no_accumulation(self):
        # Reassigning the same field on a long-lived SearchParameters
        # must release the prior ref. Without per-field tracking the
        # protection list grows unbounded - reproducing the original
        # leak shape under a different access pattern.
        params = faiss.SearchParameters()
        sel1 = faiss.IDSelectorBatch(np.arange(10))
        sel2 = faiss.IDSelectorBatch(np.arange(10))
        before1 = sys.getrefcount(sel1)
        params.sel = sel1
        params.sel = sel2
        # sel1 ref dropped when sel2 took its slot
        self.assertEqual(sys.getrefcount(sel1), before1)

    def test_SearchParameters_setattr_else_branch(self):
        # Else branch coverage: setting a field to None drops the prior
        # SWIG ref; assigning two different fields keeps both alive
        # independently so dropping one does not affect the other.
        params = faiss.SearchParametersIVF()
        sel = faiss.IDSelectorBatch(np.arange(10))
        quant = faiss.SearchParameters()
        sel_before = sys.getrefcount(sel)
        quant_before = sys.getrefcount(quant)

        params.sel = sel
        params.quantizer_params = quant
        self.assertEqual(sys.getrefcount(sel) - sel_before, 1)
        self.assertEqual(sys.getrefcount(quant) - quant_before, 1)

        # Drop sel via None; quantizer_params untouched.
        params.sel = None
        self.assertEqual(sys.getrefcount(sel), sel_before)
        self.assertEqual(sys.getrefcount(quant) - quant_before, 1)


dbin = 32
xtbin = np.random.randint(256, size=(100, int(dbin / 8))).astype('uint8')
xbbin = np.random.randint(256, size=(20, int(dbin / 8))).astype('uint8')


class TestReferencedBinary(unittest.TestCase):

    def test_binary_ivf(self):
        index = faiss.IndexBinaryIVF(faiss.IndexBinaryFlat(dbin), dbin, 10)
        gc.collect()
        index.train(xtbin)

    def test_wrap(self):
        index = faiss.IndexBinaryFromFloat(faiss.IndexFlatL2(dbin))
        gc.collect()
        index.add(xbbin)

if __name__ == '__main__':
    unittest.main()
