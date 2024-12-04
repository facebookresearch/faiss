# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import unittest
import numpy as np
import faiss
from enum import Enum
from faiss.contrib.datasets import SyntheticDataset


class DeletionSite(Enum):
    BEFORE_TRAIN = 1
    BEFORE_ADD = 2
    BEFORE_SEARCH = 3


def do_test(idx, index_to_delete, db, deletion_site: DeletionSite):
    if deletion_site == DeletionSite.BEFORE_TRAIN:
        del index_to_delete

    idx.train(db)

    if deletion_site == DeletionSite.BEFORE_ADD:
        del index_to_delete

    idx.add(db)

    if deletion_site == DeletionSite.BEFORE_SEARCH:
        del index_to_delete

    idx.search(db, 1)


def do_multi_test(idx, index_to_delete, db):
    for site in DeletionSite:
        do_test(idx, index_to_delete, db, site)


#
# Test
#


class TestRefs(unittest.TestCase):
    d = 32
    nv = 1000
    nlist = 10
    res = faiss.StandardGpuResources()  # pyre-ignore
    db = np.random.rand(nv, d)

    # These GPU classes reference another index.
    # This tests to make sure the deletion of the other index
    # does not cause a crash.

    def test_GpuIndexIVFFlat(self):
        index_to_delete = faiss.IndexIVFFlat(
            faiss.IndexFlat(self.d), self.d, self.nlist
        )
        idx = faiss.GpuIndexIVFFlat(
            self.res, index_to_delete, faiss.GpuIndexIVFFlatConfig()
        )
        do_multi_test(idx, index_to_delete, self.db)

    def test_GpuIndexBinaryFlat(self):
        ds = SyntheticDataset(64, 1000, 1000, 200)
        index_to_delete = faiss.IndexBinaryFlat(ds.d)
        idx = faiss.GpuIndexBinaryFlat(self.res, index_to_delete)
        tobinary = faiss.index_factory(ds.d, "LSHrt")
        tobinary.train(ds.get_train())
        xb = tobinary.sa_encode(ds.get_database())
        do_multi_test(idx, index_to_delete, xb)

    def test_GpuIndexFlat(self):
        index_to_delete = faiss.IndexFlat(self.d, faiss.METRIC_L2)
        idx = faiss.GpuIndexFlat(self.res, index_to_delete)
        do_multi_test(idx, index_to_delete, self.db)

    def test_GpuIndexIVFPQ(self):
        index_to_delete = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(self.d),
            self.d, self.nlist, 2, 8)
        idx = faiss.GpuIndexIVFPQ(self.res, index_to_delete)
        do_multi_test(idx, index_to_delete, self.db)

    def test_GpuIndexIVFScalarQuantizer(self):
        index_to_delete = faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlat(self.d, faiss.METRIC_L2),
            self.d,
            self.nlist,
            faiss.ScalarQuantizer.QT_8bit_direct,
            faiss.METRIC_L2,
            False
        )
        idx = faiss.GpuIndexIVFScalarQuantizer(self.res, index_to_delete)
        do_multi_test(idx, index_to_delete, self.db)
