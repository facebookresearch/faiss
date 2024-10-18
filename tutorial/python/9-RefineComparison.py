# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss

from faiss.contrib.evaluation import knn_intersection_measure
from faiss.contrib import datasets

# 64-dim vectors, 50000 vectors in the training, 100000 in database,
# 10000 in queries, dtype ('float32')
ds = datasets.SyntheticDataset(64, 50000, 100000, 10000)
d = 64                           # dimension

# Constructing the refine PQ index with SQfp16 with index factory
index_fp16 = faiss.index_factory(d, 'PQ32x4fs,Refine(SQfp16)')
index_fp16.train(ds.get_train())
index_fp16.add(ds.get_database())

# Constructing the refine PQ index with SQ8
index_sq8 = faiss.index_factory(d, 'PQ32x4fs,Refine(SQ8)')
index_sq8.train(ds.get_train())
index_sq8.add(ds.get_database())

# Parameterization on k factor while doing search for index refinement
k_factor = 3.0
params = faiss.IndexRefineSearchParameters(k_factor=k_factor)

# Perform index search using different index refinement
D_fp16, I_fp16 = index_fp16.search(ds.get_queries(), 100, params=params)
D_sq8, I_sq8 = index_sq8.search(ds.get_queries(), 100, params=params)

# Calculating knn intersection measure for different index types on refinement
KIM_fp16 = knn_intersection_measure(I_fp16, ds.get_groundtruth())
KIM_sq8 = knn_intersection_measure(I_sq8, ds.get_groundtruth())

# KNN intersection measure accuracy shows that choosing SQ8 impacts accuracy
assert (KIM_fp16 > KIM_sq8)

print(I_sq8[:5])
print(I_fp16[:5])
