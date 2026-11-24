# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GPU-native IVF-RaBitQ with cuVS 26.10 or newer.

Construction accepts float32 vectors and L2 distance. train(xb) and the first
add(xb) both build the complete index; do not call add(xb) after train(xb).
Incremental additions, custom IDs, CPU cloning and Faiss serialization are not
supported. reset() discards the index and allows another complete build.
"""

import faiss
import numpy as np

rng = np.random.RandomState(123)
d, nb, nq, nlist = 64, 4096, 16, 32
xb = rng.random_sample((nb, d)).astype("float32")
xq = rng.random_sample((nq, d)).astype("float32")

resources = faiss.StandardGpuResources()
config = faiss.GpuIndexIVFRaBitQConfig()
config.use_cuvs = True  # Explicitly select the cuVS implementation.
config.bitsPerDim = 3
config.nprobe = 16
index = faiss.GpuIndexIVFRaBitQ(resources, d, nlist, faiss.METRIC_L2, config)
index.add(xb)
assert index.is_trained and index.ntotal == nb

# Per-search parameters can override the defaults without rebuilding.
params = faiss.SearchParametersIVFRaBitQ()
params.nprobe = nlist
params.searchMode = faiss.IVFRaBitQSearchMode_LUT16
distances, labels = index.search(xq, 10, params=params)
print(labels[:5])
print(distances[:5])

index.reset()
assert not index.is_trained and index.ntotal == 0
