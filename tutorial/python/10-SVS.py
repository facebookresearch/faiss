# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss                        # make faiss available
index = faiss.IndexSVS(d)           # build the index (DynamicVamana, float32)

# index = faiss.IndexSVSFlat(d)     # build the SVSFlat index
# index = faiss.IndexSVSLVQ(d)      # build the SVSLVQ index, quantization parameters
# index = faiss.IndexSVSLeanVec(d)   # build the SVSLeanVec index, quantization parameters

print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors

print(f"{k} nearest neighbors of the first 5 vectors")
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(f"{k} nearest neighbors of the 5 first query vectors")
print(I[:5])                   # neighbors of the 5 first queries
print(f"{k} nearest neighbors of the 5 last query vectors")
print(I[-5:])                  # neighbors of the 5 last queries

faiss.write_index(index, "index.faiss")
reloaded = faiss.read_index("index.faiss")

D, I = reloaded.search(xq, k)  # search with the reloaded
print(f"{k} nearest neighbors of the 5 first query vectors (after reloading)")
print(I[:5])                   # neighbors of the 5 first queries
print(f"{k} nearest neighbors of the 5 last query vectors (after reloading)")
print(I[-5:])                  # neighbors of the 5 last queries

lvq_idx = faiss.IndexSVSLVQ(d, faiss.METRIC_L2, faiss.LVQ_4x8) # example of using SVS LVQ
lvq_idx_fac = faiss.index_factory(d, 'SVS_LVQ_4x8', faiss.METRIC_L2) # example of using factory for SVS LVQ
lvq_idx_fac.add(xb)
lvq_idx_fac.search(xq, k)


leanvec_idx = faiss.IndexSVSLeanVec(d, faiss.METRIC_L2, 0, faiss.LeanVec_4x4) # example of using SVS LeanVec
leanvec_idx_fac = faiss.index_factory(d, 'SVS_LeanVec_4x4,32', faiss.METRIC_L2) # example of using factory for SVS LeanVec
leanvec_idx_fac.train(xb)
leanvec_idx_fac.add(xb)
leanvec_idx_fac.search(xq, k)
