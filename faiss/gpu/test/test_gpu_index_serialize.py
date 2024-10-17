# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import faiss

def make_t(num, d):
    rs = np.random.RandomState(123)
    return rs.rand(num, d).astype('float32')

class TestGpuSerialize(unittest.TestCase):
    def test_serialize(self):
        res = faiss.StandardGpuResources()

        d = 32
        k = 10
        train = make_t(10000, d)
        add = make_t(10000, d)
        query = make_t(10, d)

        # Construct various GPU index types
        indexes = []

        # Flat
        indexes.append(faiss.GpuIndexFlatL2(res, d))

        # IVF
        nlist = 5

        # IVFFlat
        indexes.append(faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2))

        # IVFSQ
        config = faiss.GpuIndexIVFScalarQuantizerConfig()
        config.use_raft = False
        indexes.append(faiss.GpuIndexIVFScalarQuantizer(res, d, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2, True, config))

        # IVFPQ
        indexes.append(faiss.GpuIndexIVFPQ(res, d, nlist, 4, 8, faiss.METRIC_L2))

        for index in indexes:
            index.train(train)
            index.add(add)

            orig_d, orig_i = index.search(query, k)

            ser = faiss.serialize_index(faiss.index_gpu_to_cpu(index))
            cpu_index = faiss.deserialize_index(ser)
             
            gpu_cloner_options = faiss.GpuClonerOptions()
            if isinstance(index, faiss.GpuIndexIVFScalarQuantizer):
                gpu_cloner_options.use_raft = False
            gpu_index_restore = faiss.index_cpu_to_gpu(res, 0, cpu_index, gpu_cloner_options)

            restore_d, restore_i = gpu_index_restore.search(query, k)

            self.assertTrue(np.array_equal(orig_d, restore_d))
            self.assertTrue(np.array_equal(orig_i, restore_i))

            # Make sure the index is in a state where we can add to it
            # without error
            gpu_index_restore.add(query)
