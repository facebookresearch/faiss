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
index.num_threads = 72

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
