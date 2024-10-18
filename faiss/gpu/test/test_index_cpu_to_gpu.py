# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
import faiss


class TestMoveToGpu(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res = faiss.StandardGpuResources()

    def create_index(self, factory_string):
        dimension = 128
        n = 2500
        db_vectors = np.random.random((n, dimension)).astype('float32')
        index = faiss.index_factory(dimension, factory_string)
        index.train(db_vectors)
        if factory_string.startswith("IDMap"):
            index.add_with_ids(db_vectors, np.arange(n))
        else:
            index.add(db_vectors)
        return index

    def create_and_clone(self, factory_string,
                         allowCpuCoarseQuantizer=None,
                         use_cuvs=None):
        idx = self.create_index(factory_string)
        config = faiss.GpuClonerOptions()
        if allowCpuCoarseQuantizer is not None:
            config.allowCpuCoarseQuantizer = allowCpuCoarseQuantizer
        if use_cuvs is not None:
            config.use_cuvs = use_cuvs
        faiss.index_cpu_to_gpu(self.res, 0, idx, config)

    def verify_throws_not_implemented_exception(self, factory_string):
        try:
            self.create_and_clone(factory_string)
        except Exception as e:
            if "not implemented" not in str(e):
                self.fail("Expected an exception but no exception was "
                          "thrown for factory_string: %s." % factory_string)

    def verify_clones_successfully(self, factory_string,
                                   allowCpuCoarseQuantizer=None,
                                   use_cuvs=None):
        try:
            self.create_and_clone(
                factory_string,
                allowCpuCoarseQuantizer=allowCpuCoarseQuantizer,
                use_cuvs=use_cuvs)
        except Exception as e:
            self.fail("Unexpected exception thrown factory_string: "
                      "%s; error message: %s." % (factory_string, str(e)))

    def test_not_implemented_indices(self):
        self.verify_throws_not_implemented_exception("PQ16")
        self.verify_throws_not_implemented_exception("LSHrt")
        self.verify_throws_not_implemented_exception("HNSW")
        self.verify_throws_not_implemented_exception("HNSW,PQ16")
        self.verify_throws_not_implemented_exception("IDMap,PQ16")
        self.verify_throws_not_implemented_exception("IVF256,ITQ64,SH1.2")

    def test_implemented_indices(self):
        self.verify_clones_successfully("Flat")
        self.verify_clones_successfully("IVF1,Flat")
        self.verify_clones_successfully("IVF32,PQ8")
        self.verify_clones_successfully("IDMap,Flat")
        self.verify_clones_successfully("PCA12,IVF32,Flat")
        self.verify_clones_successfully("PCA32,IVF32,PQ8")
        self.verify_clones_successfully("PCA32,IVF32,PQ8np")

        # set use_cuvs to false, these index types are not supported on cuVS
        self.verify_clones_successfully("IVF32,SQ8", use_cuvs=False)
        self.verify_clones_successfully(
            "PCA32,IVF32,SQ8", use_cuvs=False)

    def test_with_flag(self):
        self.verify_clones_successfully("IVF32_HNSW,Flat",
                                        allowCpuCoarseQuantizer=True)
        self.verify_clones_successfully("IVF256(PQ2x4fs),Flat",
                                        allowCpuCoarseQuantizer=True)

    def test_with_flag_set_to_false(self):
        try:
            self.verify_clones_successfully("IVF32_HNSW,Flat",
                                            allowCpuCoarseQuantizer=False)
        except Exception as e:
            if "set the flag to true to allow the CPU fallback" not in str(e):
                self.fail("Unexepected error message thrown: %s." % str(e))
