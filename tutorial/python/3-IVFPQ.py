# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Note on recall for IVFPQ
# ------------------------
# IVFPQ is a lossy index: the inverted-file step prunes the search to a few
# Voronoi cells, and the product quantizer replaces each residual with its
# nearest codebook entry. Both steps assume the database has exploitable
# cluster structure, so that (a) most true neighbours of a query fall in
# the few cells visited, and (b) residuals concentrate around a small set
# of prototypes. Real-world embedding datasets (image features such as
# SIFT1M, sentence embeddings, learned representations) carry that
# structure and give the recall numbers reported in the Faiss benchmarks.
#
# Uniform random vectors do not. With no clusters to exploit, nprobe=10
# out of nlist=100 cells visits roughly 10% of the database and per-cell
# quantization noise is large, so top-k recall is typically an order of
# magnitude below what is seen on clustered data. The synthetic data
# below uses a single coordinate ramp (xb[:, 0] += arange / 1000.) to
# inject light structure for a runnable demo; if you swap it for purely
# uniform vectors, expect recall to drop sharply. This is a property of
# the data, not a bug in the index. See Jegou, Douze, Schmid, "Product
# Quantization for Nearest Neighbor Search," IEEE TPAMI 2011.

import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss

nlist = 100
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])
