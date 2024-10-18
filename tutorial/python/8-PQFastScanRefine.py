# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')    # 64-dim *nb queries
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

m = 8  # 8 specifies that the number of sub-vector is 8
k = 4  # number of dimension in etracted vector
n_bit = 4  # 4 specifies that each sub-vector is encoded as 4 bits
bbs = 32  # build block size ( bbs % 32 == 0 ) for PQ

index = faiss.IndexPQFastScan(d, m, n_bit, faiss.METRIC_L2)
index_refine = faiss.IndexRefineFlat(index)
# construct FastScan and run index refinement

assert not index_refine.is_trained
index_refine.train(xb)  # Train vectors data index within mockup database
assert index_refine.is_trained

index_refine.add(xb)
params = faiss.IndexRefineSearchParameters(k_factor=3)
D, I = index_refine.search(xq[:5], 10, params=params)
print(I)
print(D)
index.nprobe = 10  # make comparable with experiment above
D, I = index.search(xq[:5], k)  # search
print(I[-5:])
