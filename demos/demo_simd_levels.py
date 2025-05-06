# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import faiss
import numpy as np  
import os
from collections import defaultdict 
from faiss.contrib.datasets import SyntheticDataset 


print("compile options", faiss.get_compile_options())
print("SIMD level: ", faiss.SIMDConfig.get_level_name())


ds = SyntheticDataset(32, 8000, 10000, 8000)


index = faiss.index_factory(ds.d, "PQ16x4fs")
# index = faiss.index_factory(ds.d, "IVF64,PQ16x4fs")
# index = faiss.index_factory(ds.d, "SQ8")

index.train(ds.get_train())
index.add(ds.get_database())


if False: 
    faiss.omp_set_num_threads(1)
    print("PID=", os.getpid())
    input("press enter to continue")
    # for simd_level in faiss.NONE, faiss.AVX2, faiss.AVX512F: 
    for simd_level in faiss.AVX2, faiss.AVX512F: 
      
        faiss.SIMDConfig.set_level(simd_level)
        print("simd_level=", faiss.SIMDConfig.get_level_name())
        for run in range(1000): 
            D, I = index.search(ds.get_queries(), 10)

times = defaultdict(list)

for run in range(10): 
    for simd_level in faiss.SIMDLevel_NONE, faiss.SIMDLevel_AVX2, faiss.SIMDLevel_AVX512F: 
        faiss.SIMDConfig.set_level(simd_level)

        t0 = time.time()    
        D, I = index.search(ds.get_queries(), 10)
        t1 = time.time()

        times[faiss.SIMDConfig.get_level_name()].append(t1 - t0)

for simd_level in times: 
    print(f"simd_level={simd_level} search time: {np.mean(times[simd_level])*1000:.3f} ms")
