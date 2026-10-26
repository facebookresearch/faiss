# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for merging IVF shards. Ported from test_merge.cpp."""

import os
import platform
import tempfile
import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


d = 64
nb = 1000
nq = 100
nindex = 4
k = 10
nlist = 40
shard_size = nb // nindex


class CommonData:
    def __init__(self):
        rng = np.random.RandomState(123)
        self.database = rng.rand(nb, d).astype(np.float32)
        self.queries = rng.rand(nq, d).astype(np.float32)
        self.ids = np.array(
            [123 + 456 * i for i in range(nb)], dtype=np.int64)
        self.quantizer = faiss.IndexFlatL2(d)
        # just to train the quantizer
        iflat = faiss.IndexIVFFlat(self.quantizer, d, nlist)
        iflat.train(self.database)


def compare_merged(index_shards, shards):
    ref_D, ref_I = index_shards.search(cd.queries, k)

    for i in range(1, nindex):
        faiss.merge_into(shards[0], shards[i], True)

    new_D, new_I = shards[0].search(cd.queries, k)
    return np.count_nonzero(ref_I != new_I)


def compare_merged_shards(index_shards, shift_ids, tmpdir,
                          standard_merge=True):
    ref_D, ref_I = index_shards.search(cd.queries, k)

    if standard_merge:
        for i in range(1, nindex):
            faiss.merge_into(
                index_shards.at(0), index_shards.at(i), shift_ids)
        index_shards.syncWithSubIndexes()
    else:
        lists = faiss.InvertedListsPtrVector()
        index0 = None
        ntotal = 0
        for i in range(nindex):
            index_ivf = faiss.downcast_index(index_shards.at(i))
            assert isinstance(index_ivf, faiss.IndexIVF)
            if i == 0:
                index0 = index_ivf
            lists.push_back(index_ivf.invlists)
            ntotal += index_ivf.ntotal

        il = faiss.OnDiskInvertedLists(
            index0.nlist, index0.code_size,
            os.path.join(tmpdir, "merged.ivfdata"))
        il.merge_from_multiple(lists.data(), lists.size(), shift_ids)
        index0.replace_invlists(il, True)
        il.this.disown()
        index0.ntotal = ntotal

    new_D, new_I = index_shards.at(0).search(cd.queries, k)

    adjust_ids = shift_ids and not standard_merge
    if adjust_ids:
        new_I = np.fmod(new_I, shard_size)
    return np.count_nonzero(ref_I != new_I)


cd = CommonData()


class TestMerge(unittest.TestCase):
    def test_merge_flat_no_ids(self):
        shards = [
            faiss.IndexIVFFlat(cd.quantizer, d, nlist)
            for _ in range(nindex)
        ]
        index_shards = faiss.IndexShards(d)
        index_shards.successive_ids = True
        for shard in shards:
            index_shards.add_shard(shard)
        self.assertTrue(index_shards.is_trained)
        index_shards.add(cd.database)
        prev_ntotal = index_shards.ntotal
        ndiff = compare_merged(index_shards, shards)
        self.assertEqual(prev_ntotal, shards[0].ntotal)
        self.assertEqual(0, ndiff)

    def test_merge_flat(self):
        shards = [
            faiss.IndexIVFFlat(cd.quantizer, d, nlist)
            for _ in range(nindex)
        ]
        index_shards = faiss.IndexShards(d)
        index_shards.successive_ids = False
        for shard in shards:
            index_shards.add_shard(shard)
        self.assertTrue(index_shards.is_trained)
        index_shards.add_with_ids(cd.database, cd.ids)
        prev_ntotal = index_shards.ntotal

        ref_D, ref_I = index_shards.search(cd.queries, k)
        for i in range(1, nindex):
            faiss.merge_into(shards[0], shards[i], False)
        _, new_I = shards[0].search(cd.queries, k)
        ndiff = np.count_nonzero(ref_I != new_I)

        self.assertEqual(prev_ntotal, shards[0].ntotal)
        self.assertEqual(0, ndiff)

    def test_merge_flat_vt(self):
        index_shards = faiss.IndexShards(d, False, False)

        rot = faiss.RandomRotationMatrix(d, d)
        rot.init(1234)
        quantizer = faiss.IndexFlatL2(d)

        # just to train the quantizer
        iflat = faiss.IndexIVFFlat(quantizer, d, nlist)
        ipt = faiss.IndexPreTransform(rot, iflat)
        ipt.train(cd.database)

        for _ in range(nindex):
            index_shards.add_shard(faiss.IndexPreTransform(
                faiss.Cloner().clone_VectorTransform(rot),
                faiss.IndexIVFFlat(quantizer, d, nlist)))
        self.assertTrue(index_shards.is_trained)
        index_shards.add_with_ids(cd.database, cd.ids)
        prev_ntotal = index_shards.ntotal
        with tempfile.TemporaryDirectory() as tmpdir:
            ndiff = compare_merged_shards(index_shards, False, tmpdir)
        self.assertEqual(prev_ntotal, index_shards.ntotal)
        self.assertEqual(0, ndiff)

    @unittest.skipIf(
        platform.system() == "Windows",
        "OnDiskInvertedLists is unsupported on Windows.",
    )
    def test_merge_flat_ondisk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_shards = faiss.IndexShards(d, False, False)
            for i in range(nindex):
                ivf = faiss.IndexIVFFlat(cd.quantizer, d, nlist)
                if i == 0:
                    il = faiss.OnDiskInvertedLists(
                        ivf.nlist, ivf.code_size,
                        os.path.join(tmpdir, "shard0.ivfdata"))
                    ivf.replace_invlists(il, True)
                    il.this.disown()
                index_shards.add_shard(ivf)

            self.assertTrue(index_shards.is_trained)
            index_shards.add_with_ids(cd.database, cd.ids)
            ndiff = compare_merged_shards(index_shards, False, tmpdir)
            del index_shards, ivf, il

        self.assertEqual(0, ndiff)

    @unittest.skipIf(
        platform.system() == "Windows",
        "OnDiskInvertedLists is unsupported on Windows.",
    )
    def test_merge_flat_ondisk_2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_shards = faiss.IndexShards(d, False, False)
            for _ in range(nindex):
                index_shards.add_shard(
                    faiss.IndexIVFFlat(cd.quantizer, d, nlist))
            self.assertTrue(index_shards.is_trained)
            index_shards.add_with_ids(cd.database, cd.ids)
            ndiff = compare_merged_shards(
                index_shards, False, tmpdir, standard_merge=False)
            del index_shards

        self.assertEqual(0, ndiff)

    @unittest.skipIf(
        platform.system() == "Windows",
        "OnDiskInvertedLists is unsupported on Windows.",
    )
    def test_merge_flat_ondisk_3(self):
        ids = np.arange(nb, dtype=np.int64) % shard_size
        with tempfile.TemporaryDirectory() as tmpdir:
            index_shards = faiss.IndexShards(d, False, False)
            for _ in range(nindex):
                index_shards.add_shard(
                    faiss.IndexIVFFlat(cd.quantizer, d, nlist))
            self.assertTrue(index_shards.is_trained)
            index_shards.add_with_ids(cd.database, ids)
            ndiff = compare_merged_shards(
                index_shards, True, tmpdir, standard_merge=False)
            del index_shards

        self.assertEqual(0, ndiff)
