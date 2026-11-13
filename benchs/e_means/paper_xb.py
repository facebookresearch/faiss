# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The paper's timing protocol (xbench) in one small module, for faiss.contrib.e_means.

Every timed run is its own process (isolated). That process:
  1. loads the dataset (paper_data.py checks the public files against the paper's arrays);
  2. takes the paper's starting rows: the first k of a permutation drawn by a seeded CUDA
     generator, exactly as the paper's runs drew them, so both methods of a (dataset, seed)
     start from the same rows as the paper's own runs (starts);
  3. warms the method up with an untimed call: for e-means the identical call, for Flash-KMeans
     two iterations at the exact shape, which is what compiles and tunes its kernels;
  4. frees GPU memory and times one call between two GPU synchronizations; when that call took
     under 2 s it times two more from the same freed state and keeps the median (timed);
  5. scores the first call's centroids on the test half with one exact scorer: ranking by an
     IEEE float32 product with TF32 off, distances recomputed by direct difference in float64.
paper_bench.py runs both methods of a (dataset, seed) one after the other in the same job, so on
the same GPU.

e-means assigns in bfloat16 with its CuTe kernels. Its input reaches bfloat16 by one of two
routes, which must give the same result (comparing them is a check on the library):
  cast  float32 rows on the GPU, converted batch by batch by CastQuantizer(d, torch.bfloat16)
        (the documented way to train on float32 data; without a quantizer e-means would assign
        in real float32, which is far slower)
  bf16  rows converted to bfloat16 once, before training, and passed with quantizer=None
The default is cast, and bf16 where a float32 copy would not fit one 80 GB GPU (SimSearchNet++
at 100M rows). Every integer corpus here, and every rescaled one (a division by a power of two),
is exact in bfloat16, so there the two routes see identical values.

Flash-KMeans runs the way its own code dispatches data held on the GPU: its standard function
batch_kmeans_Euclid, in float16. Two corpora follow the paper's float32-norm rule: SPACEV uses the
patched copy of that function, and SimSearchNet++, whose squared norms overflow float16 even there,
uses the patched large-N function. paper_flash_matched.py builds the patched copy.

    python paper_xb.py --one '{"method": "emeans", "dataset": "deep1b-1M", "seed": 10,
                               "epochs": 2, "input": "cast"}'
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as data  # noqa: E402

CALLS, CALLS_MAX_S = 3, 2.0            # three timed calls when the first took under 2 s
FLASH_BLOCK = 1 << 20                  # kmeans_largeN's block of rows
MATCHED = ("spacev", "fb-ssnpp")       # Flash with float32 norms (the paper's rule)
LARGE_N = ("fb-ssnpp",)                # ... through its large-N function
FLOAT32_MAX_BYTES = 60 * 2**30         # largest float32 copy the cast route puts on one GPU
FLASH_MATCHED = os.path.join(data.CACHE, "flash_matched")
FLASH_VERSION = "0.3.1"                # the Flash-KMeans release the paper timed


def k_for(n):
    """The paper's cluster count at each scale."""
    return 16384 if n <= 2 * 10**6 else 65536 if n <= 2 * 10**7 else 262144


def batch_law(n):
    """The paper's batch: the larger "throughput" batch at a million rows, the "quality" batch
    (one eighth of it) at ten and a hundred million."""
    return "throughput" if n <= 2 * 10**6 else "quality"


def default_input(n, d):
    return "cast" if 4 * n * d <= FLOAT32_MAX_BYTES else "bf16"


def pin_precision():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync():
    import torch
    torch.cuda.synchronize()


def free():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def upload(train, name, dtype, chunk=1 << 20):
    """The training rows as a GPU tensor of `dtype`, rescaled as the paper rescaled them and
    converted in chunks, so no full float32 copy is ever made on the host."""
    import torch
    x = torch.empty(train.shape, dtype=dtype, device="cuda")
    for s in range(0, train.shape[0], chunk):
        x[s:s + chunk] = torch.from_numpy(data.rows(train, name, s, s + chunk)).cuda().to(dtype)
    return x


def starts(n, k, seed):
    """The paper's starting rows: the first k of a seeded permutation drawn on the GPU (the
    paper's gpu_kmeans_init(..., init="random", seed=seed)). Checked to reproduce the paper's
    starting centroids exactly with PyTorch 2.14."""
    import torch
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    return torch.randperm(n, generator=generator, device="cuda")[:k].cpu().numpy()


def test_error(centroids, test, chunk_rows=32768, chunk_k=16384):
    """Mean squared distance from each test row to its nearest centroid: ranked with an IEEE
    float32 product (TF32 off), then the distance to the chosen centroid recomputed by direct
    difference in float64, so the error has no norm cancellation."""
    import torch
    pin_precision()
    c = torch.as_tensor(np.ascontiguousarray(centroids, dtype=np.float32), device="cuda")
    cn = (c.double() ** 2).sum(1).float()                               # (k,)
    total = 0.0
    for s in range(0, test.shape[0], chunk_rows):
        xb = torch.from_numpy(np.ascontiguousarray(test[s:s + chunk_rows])).cuda()  # (m, d)
        best = torch.full((xb.shape[0],), float("inf"), device="cuda")
        arg = torch.zeros(xb.shape[0], dtype=torch.int64, device="cuda")
        for j in range(0, c.shape[0], chunk_k):
            v, a = (cn[j:j + chunk_k][None, :] - 2.0 * (xb @ c[j:j + chunk_k].t())).min(1)
            better = v < best
            best = torch.where(better, v, best)
            arg = torch.where(better, a + j, arg)
        diff = xb.double() - c[arg].double()                            # (m, d)
        total += float((diff * diff).sum())
    return total / test.shape[0]


def timed(call, warmup):
    """Untimed warmup(), then call() timed between two synchronizations: once, or three times
    when the first took under 2 s, each from freed memory. Returns (the first call's centroids
    as a NumPy array, the median time, every call's time)."""
    warmup()
    sync()
    first, secs = None, []
    for i in range(CALLS):
        free()
        sync()
        start = time.perf_counter()
        out = call()
        sync()
        secs.append(time.perf_counter() - start)
        if i == 0:
            first = out.detach().float().cpu().numpy()
        del out
        if secs[0] >= CALLS_MAX_S:
            break
    return first, statistics.median(secs), secs


def library():
    """Which faiss.contrib.e_means ran: its folder and a short hash of its source files."""
    import faiss.contrib.e_means as em
    folder = os.path.dirname(em.__file__)
    h = hashlib.sha256()
    for f in sorted(os.listdir(folder)):
        if f.endswith(".py"):
            h.update(f.encode() + open(os.path.join(folder, f), "rb").read())
    return dict(emeans_path=folder, emeans_source_sha256=h.hexdigest()[:16])


def run_emeans(name, train, init, seed, epochs, route):
    """faiss.contrib.e_means, bfloat16 assignment with its CuTe kernels, the input reaching
    bfloat16 by `route` (module docstring)."""
    import torch
    from faiss.contrib.e_means import Emeans, QuantizedEmeans
    from faiss.contrib.e_means.base import _resolve_recipe
    from faiss.contrib.e_means.quantizer import CastQuantizer
    n, d = train.shape
    k = k_for(n)
    model = Emeans
    kw = dict(seed=seed, batch_law=batch_law(n), assignment_backend="cute")
    if epochs < 1:                     # below one pass the paper keeps the one-pass batch, step
        kw["batch_size"], kw["alpha0"] = _resolve_recipe(n, k, 1.0, kw["batch_law"], None, None)
    if route == "cast":
        x = upload(train, name, torch.float32)                          # (n, d)
        model = QuantizedEmeans
        kw["quantizer"] = CastQuantizer(d, torch.bfloat16)
    elif route == "bf16":
        x = upload(train, name, torch.bfloat16)                         # (n, d)
    else:
        raise ValueError("input must be cast or bf16, got %r" % route)
    start = torch.from_numpy(init).cuda()                               # (k, d) float32

    def call():
        km = model(d, k, nepochs=epochs, **kw)
        km.train(x, init_centroids=start)
        call.recipe = km.recipe
        return km.centroids

    centroids, seconds, calls = timed(call, warmup=call)
    recipe = call.recipe
    return centroids, seconds, calls, dict(
        input=route, batch=int(recipe["batch_size"]), alpha0=float(recipe["alpha0"]),
        representation=str(recipe.get("representation")),
        assignment_backend=str(recipe.get("assignment_backend")), **library())


def run_flash(name, train, init, iterations):
    """Flash-KMeans, float16 rows on the GPU, every iteration forced (tol=-1). Only the release
    the paper timed is accepted: another one would give numbers that look right and are not the
    paper's baseline."""
    import torch
    version = importlib.metadata.version("flash-kmeans")
    if version != FLASH_VERSION:
        sys.exit("the paper timed flash-kmeans %s, found %s: pip install flash-kmeans==%s"
                 % (FLASH_VERSION, version, FLASH_VERSION))
    if name.startswith(MATCHED):
        marker = os.path.join(FLASH_MATCHED, "VERSION")
        if not os.path.exists(marker) or open(marker).read().strip() != FLASH_VERSION:
            sys.exit("%s needs Flash-KMeans %s with float32 norms: run python "
                     "paper_flash_matched.py" % (name, FLASH_VERSION))
        sys.path.insert(0, FLASH_MATCHED)
    import flash_kmeans
    n, d = train.shape
    k = k_for(n)
    x = upload(train, name, torch.float16)                              # (n, d)
    start = torch.from_numpy(init).cuda()                               # (k, d) float32
    if name.startswith(LARGE_N):
        from flash_kmeans.kmeans_large import kmeans_largeN

        def solve(iters, rows=n):
            out = kmeans_largeN(x[:rows].contiguous(), k, max_iters=iters, tol=-1,
                                BLOCK_N=FLASH_BLOCK, init_centroids=start.clone(),
                                device=x.device, dtype=torch.float16)
            return out[1] if isinstance(out, (tuple, list)) else out

        sizes = [min(n, FLASH_BLOCK)] + ([n % FLASH_BLOCK] if n > FLASH_BLOCK and n % FLASH_BLOCK
                                         else [])
        centroids, seconds, calls = timed(lambda: solve(iterations),
                                          warmup=lambda: [solve(2, rows) for rows in sizes])
        api = "kmeans_largeN"
    else:
        from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid
        start16 = start.half()[None]                                    # (1, k, d)

        def solve(iters):
            out = batch_kmeans_Euclid(x[None], k, max_iters=iters, tol=-1,
                                      init_centroids=start16.clone())
            assert int(out[2]) == iters, "Flash ran %s iterations, %d asked" % (out[2], iters)
            return out[1][0]

        centroids, seconds, calls = timed(lambda: solve(iterations), warmup=lambda: solve(2))
        api = "batch_kmeans_Euclid"
    return centroids, seconds, calls, dict(
        flash_api=api, flash_norms="float32" if name.startswith(MATCHED) else "float16",
        flash_package=os.path.dirname(flash_kmeans.__file__), flash_version=version)


def one(spec):
    """One timed run, in this process. Returns its record."""
    import torch
    import triton
    pin_precision()
    train, test = data.load(spec["dataset"])
    n, d = train.shape
    rows = starts(n, k_for(n), spec["seed"])
    order = np.argsort(rows)
    init = np.empty((len(rows), d), dtype=np.float32)                  # train[rows], (k, d)
    init[order] = data.rows(train, spec["dataset"], index=rows[order])
    if spec["method"] == "emeans":
        spec = dict(spec, input=spec.get("input") or default_input(n, d))
        centroids, seconds, calls, extra = run_emeans(spec["dataset"], train, init, spec["seed"],
                                                      spec["epochs"], spec["input"])
    else:
        centroids, seconds, calls, extra = run_flash(spec["dataset"], train, init,
                                                     spec["iterations"])
    return dict(spec, n=n, k=k_for(n), seconds=seconds, seconds_calls=calls,
                test_mse=test_error(centroids, test), gpu=torch.cuda.get_device_name(),
                device=str(torch.device("cuda", torch.cuda.current_device())),
                visible_gpus=torch.cuda.device_count(),
                torch=torch.__version__, triton=triton.__version__, **extra)


def isolated(spec, python=None):
    """One timed run in a fresh process (python paper_xb.py --one), as the paper's runs were.
    `python` runs it under another interpreter, for example to time Flash-KMeans in another
    PyTorch and Triton environment; Flash-KMeans runs under $XB_FLASH_PYTHON when it is set."""
    if python is None and spec["method"] == "flash":
        python = os.environ.get("XB_FLASH_PYTHON") or None
    out = subprocess.run([python or sys.executable, os.path.join(HERE, "paper_xb.py"), "--one",
                          json.dumps(spec)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("run %s failed:\n%s" % (spec, out.stderr[-3000:]))
    return json.loads(out.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", required=True, help="JSON spec of one timed run")
    print(json.dumps(one(json.loads(ap.parse_args().one))), flush=True)
