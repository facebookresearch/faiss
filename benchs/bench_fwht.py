# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark HadamardRotation (HD3 FWHT) vs RandomRotationMatrix (BLAS sgemm).

Measures:
  1. Speed: wall-clock time for transform.apply(x)
  2. Recall@1: nearest-neighbor recall after pre-transform + Flat index
"""

import faiss
import numpy as np
import time

n = 10000
nq = 200
n_trials = 15

print(f'n={n} vectors, nq={nq} queries, {n_trials} speed trials')

# --- Speed benchmark ---
print('\n=== Speed ===')
print(f'{"dim":>6s}  {"HR (ms)":>10s}  {"RRM (ms)":>10s}  {"speedup":>8s}')
print('-' * 42)

dims = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
for d in dims:
    np.random.seed(42)
    x = np.random.randn(n, d).astype('float32')

    hr = faiss.HadamardRotation(d, 42)
    hr.apply(x[:100])  # warmup
    times_hr = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        hr.apply(x)
        t1 = time.perf_counter()
        times_hr.append((t1 - t0) * 1000)
    hr_ms = sorted(times_hr)[1]

    rr = faiss.RandomRotationMatrix(d, d)
    rr.train(x[:100])
    rr.apply(x[:100])  # warmup
    times_rr = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        rr.apply(x)
        t1 = time.perf_counter()
        times_rr.append((t1 - t0) * 1000)
    rr_ms = sorted(times_rr)[1]

    speedup = rr_ms / hr_ms
    print(f'{d:6d}  {hr_ms:10.2f}  {rr_ms:10.2f}  {speedup:7.1f}x')

# --- Recall benchmark with IVF (where rotation actually matters) ---
print(f'\n=== Recall@1 with IVF (L2, n={n}, nq={nq}, k=1) ===')
print(f'{"dim":>6s}  {"HR R@1":>8s}  {"RRM R@1":>8s}  {"none R@1":>8s}')
print('-' * 40)

for d in [64, 128, 256, 768, 1024, 2048, 4096]:
    np.random.seed(42)
    xb = np.random.randn(n, d).astype('float32')
    xq = np.random.randn(nq, d).astype('float32')

    # Ground truth (brute force, no transform)
    gt_index = faiss.IndexFlatL2(d)
    gt_index.add(xb)
    _, gt_I = gt_index.search(xq, 1)

    nlist = 64
    recalls = {}
    for label, factory_str in [
        ('HR', f'HR,IVF{nlist},Flat'),
        ('RRM', f'RR,IVF{nlist},Flat'),
        ('none', f'IVF{nlist},Flat'),
    ]:
        index = faiss.index_factory(d, factory_str)
        index.train(xb)
        index.add(xb)
        faiss.extract_index_ivf(index).nprobe = 8
        _, I = index.search(xq, 1)
        recall = (I[:, 0] == gt_I[:, 0]).mean()
        recalls[label] = recall

    print(
        f'{d:6d}  {recalls["HR"]:8.4f}'
        f'  {recalls["RRM"]:8.4f}  {recalls["none"]:8.4f}'
    )
