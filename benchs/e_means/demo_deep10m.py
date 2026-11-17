# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run e-means at Table 3's DEEP* 10M budget (2 epochs), with one seed.

    python demo_deep10m.py

Requires the software and curl command in README.md, which downloads
the first 10,010,000 rows to /tmp/base.1B.fbin.part.
Uses the paper's timing and test split, with float32 scoring;
starting rows are sampled without replacement with a seeded NumPy generator.
This runs one budget; it does not search for the first crossing or run Flash.
"""

import gc
import time

import numpy as np
import torch
from faiss.contrib.e_means import QuantizedEmeans
from faiss.contrib.e_means.quantizer import CastQuantizer

seed = 123


# TF32 off, including for the test-error scorer.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


####################### Load data
path = "/tmp/base.1B.fbin.part"
print(
    f"Load data: {path}, 10,010,000 vectors of dimension 96, float32",
    flush=True,
)
load_start = time.perf_counter()
_, d = np.fromfile(path, dtype="<i4", count=2)
if d != 96:
    raise ValueError("DEEP* requires vectors of dimension 96")
# The header describes the full corpus; map only the downloaded prefix.
vectors = np.memmap(
    path, dtype="<f4", mode="r", offset=8, shape=(10_010_000, d)
)

# Split train - test
train = vectors[:10_000_000]
n, d = train.shape
k = 65536
x = torch.tensor(train, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
print(
    f"Loaded {n:,} x {d} training values onto {torch.cuda.get_device_name()}: "
    f"{x.numel() * x.element_size() / 1e9:.2f} GB "
    f"in {time.perf_counter() - load_start:.2f} s",
    flush=True,
)
permutation = np.random.RandomState(1234).permutation(10_000)
test = vectors[10_000_000 + permutation[5_000:]]
print(
    f"Sample test set: {len(test):,} x {d} "
    "from 10,000 evaluation rows, seed 1234",
    flush=True,
)
rows = np.random.default_rng(seed).choice(n, k, replace=False)
init = torch.from_numpy(np.array(train[rows], dtype=np.float32)).cuda()
print(
    f"Sample initialization: {k:,} x {d} from {n:,} training rows, seed {seed}",
    flush=True,
)

# Identical untimed warm-up, then median of three synchronized calls if <2 s.
# Loading, upload, initialization and scoring are outside the training timer.
print(
    f"E-means warm-up: n={n:,}, d={d}, k={k:,}, epochs=2, seed={seed}, "
    "quality/CuTe/bfloat16",
    flush=True,
)
warmup_start = time.perf_counter()
km = QuantizedEmeans(
    d, k, nepochs=2.0, seed=seed, batch_law="quality",
    assignment_backend="cute", quantizer=CastQuantizer(d, torch.bfloat16),
)
km.train(x, init_centroids=init)
torch.cuda.synchronize()
print(
    f"Warm-up complete: {time.perf_counter() - warmup_start:.4f} s, "
    f"batch={km.recipe['batch_size']:,}, steps={km.recipe['total_steps']}, "
    f"alpha0={km.recipe['alpha0']:.6g}, "
    f"last-pass objective={km.iteration_stats[-1]['obj']:.6g}",
    flush=True,
)
del km
calls = []
for i in range(3):
    print(
        f"E-means run {i + 1}: n={n:,}, d={d}, k={k:,}, epochs=2, "
        f"seed={seed}, quality/CuTe/bfloat16",
        flush=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    km = QuantizedEmeans(
        d, k, nepochs=2.0, seed=seed, batch_law="quality",
        assignment_backend="cute", quantizer=CastQuantizer(d, torch.bfloat16),
    )
    km.train(x, init_centroids=init)
    out = km.centroids
    torch.cuda.synchronize()
    calls.append(time.perf_counter() - start)
    print(
        f"Run {i + 1} complete: {calls[-1]:.4f} s, "
        f"batch={km.recipe['batch_size']:,}, steps={km.recipe['total_steps']}, "
        f"alpha0={km.recipe['alpha0']:.6g}, "
        f"last-pass objective={km.iteration_stats[-1]['obj']:.6g}",
        flush=True,
    )
    if i == 0:
        centroids = out.detach().float().cpu().numpy()
    del out, km
    if calls[0] >= 2.0:
        break
seconds = float(np.median(calls))
del x, init
gc.collect()
torch.cuda.empty_cache()

# Rank centroids and compute squared distances in float32.
c = torch.tensor(centroids, device="cuda")
queries = torch.tensor(test, device="cuda")
norms = (c * c).sum(1)
best = torch.full((len(test),), float("inf"), device="cuda")
nearest = torch.zeros(len(test), dtype=torch.int64, device="cuda")
for j in range(0, k, 16384):
    distances = (
        norms[j:j + 16384][None, :] - 2.0 * (queries @ c[j:j + 16384].t())
    )
    values, indices = distances.min(1)
    better = values < best
    best = torch.where(better, values, best)
    nearest = torch.where(better, indices + j, nearest)
diff = queries - c[nearest]
mse = float((diff * diff).sum()) / len(test)
print(f"DEEP* 10M: {n:,} vectors, {k:,} clusters, 2 epochs, seed {seed}")
print(f"GPU: {torch.cuda.get_device_name()}")
print("Input: float32, cast to bfloat16 per batch; CuTe assignment")
print(f"Training time: {seconds:.4f} s")
print("Timed calls: " + ", ".join(f"{s:.4f} s" for s in calls))
print(f"Test MSE ({len(test):,} held-out vectors): {mse:.8f}")
