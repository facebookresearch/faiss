# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss


d = 64       # dimension
nb = 100000  # database size
nq = 10000   # number of queries

np.random.seed(1234)

xb = np.random.random((nb, d)).astype("float32")
xb[:, 0] += np.arange(nb) / 1000.0

xq = np.random.random((nq, d)).astype("float32")
xq[:, 0] += np.arange(nq) / 1000.0

k = 4

index = faiss.IndexHNSWFlat(d, 32)  # M=32: neighbors per node
index.hnsw.efConstruction = 40      # graph construction quality
index.add(xb)
index.hnsw.efSearch = 64            # higher = better recall, slower

# sanity check
D, I = index.search(xb[:5], k)
print(I)
print(D)

D, I = index.search(xq, k)

print("I=")
print(I[-5:])

print("D=")
print(D[-5:])
