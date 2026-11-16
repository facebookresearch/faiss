# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Three-epoch FP32 CPU training: exhaustive assignment versus IVF256/4.

Run with 'exact' or 'ivf' in optikm-cute on separate CPU allocations with
at least 32 available CPUs each.
Training times include coarse training, index rebuilds, revival and closing.
Final training/test MSE uses one exhaustive Faiss FP32 evaluator for every
model, outside training timings. Results are written next to this script.
"""

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import time

import faiss
import faiss.contrib
import numpy as np
import torch

# Load this checkout's Python code while retaining the installed Faiss core.
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
faiss.contrib.__path__.insert(0, os.path.join(repo_root, "contrib"))
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.e_means import EmeansIVF, QuantizedEmeans
from faiss.contrib.e_means.base import _resolve_recipe


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("assignment", choices=("exact", "ivf"))
assignment = parser.parse_args().assignment
threads, seed = 32, 1234
n, k, d, epochs, nq = 2_000_000, 1 << 15, 256, 3, 50_000
nlist, nprobe = 256, 4
affinity = sorted(os.sched_getaffinity(0))
if len(affinity) < threads:
    raise RuntimeError(f"need {threads} available CPUs; affinity has {len(affinity)}")
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)
faiss.omp_set_num_threads(threads)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_ivf_vs_bruteforce_results", assignment)
os.makedirs(output_dir, exist_ok=True)

cpu = "unknown"
with open("/proc/cpuinfo") as file:
    for line in file:
        if line.startswith("model name"):
            cpu = line.partition(":")[2].strip()
            break
physical_cores = set()
for logical in affinity:
    topology = f"/sys/devices/system/cpu/cpu{logical}/topology"
    with open(topology + "/physical_package_id") as file:
        package = int(file.read())
    with open(topology + "/core_id") as file:
        core = int(file.read())
    physical_cores.add((package, core))
batch, alpha0 = _resolve_recipe(n, k, epochs, "throughput", None, None)
metadata = dict(assignment=assignment, host=platform.node(), cpu=cpu, threads=threads, affinity=affinity,
    available_physical_cores=len(physical_cores), seed=seed, n=n, k=k, d=d,
    epochs=epochs, nq=nq, nlist=nlist, nprobe=nprobe, batch_size=batch, alpha0=alpha0,
    faiss_version=faiss.__version__, torch_version=torch.__version__, numpy_version=np.__version__,
    revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
    source_diff=subprocess.check_output(["git", "diff", "--stat"], cwd=repo_root, text=True),
    mse_evaluator="exhaustive FP32 Faiss IndexFlatL2; all training rows and 50K held-out queries")
with open(os.path.join(output_dir, "metadata.json"), "w") as file:
    json.dump(metadata, file, indent=2)
print(f"CPU: {cpu}; host={platform.node()}; threads={threads}; "
      f"affinity={len(affinity)} logical / {len(physical_cores)} physical CPUs", flush=True)
print(f"SyntheticDataset: train={n:,}, test={nq:,}, d={d}, seed={seed}; "
      f"k={k:,}, epochs={epochs}, batch={batch:,}, alpha0={alpha0:.6f}", flush=True)
print(f"IVF{nlist}, nprobe={nprobe}; FP32 data, statistics and search; "
      "cost-based revival enabled; identical initialization and epoch shuffles", flush=True)
start = time.perf_counter()
dataset = SyntheticDataset(d, n, 0, nq, seed=seed)
x = torch.from_numpy(dataset.get_train())
queries = dataset.get_queries()
indices = np.random.RandomState(seed).permutation(n)[:k]
initial = x[torch.from_numpy(indices)].clone()
print(f"Data generation completed in {time.perf_counter() - start:.2f}s", flush=True)


def exhaustive_mse(index, rows, name):
    total, start = 0., time.perf_counter()
    for begin in range(0, len(rows), 8192):
        scores, _ = index.search(rows[begin:begin + 8192], 1)
        total += scores.sum(dtype=np.float64)
    mse = float(total / len(rows))
    print(f"  {name}: exhaustive MSE={mse:.6f}, evaluation={time.perf_counter() - start:.2f}s", flush=True)
    return mse


results = []
methods = ("brute_force",) if assignment == "exact" else ("ivf_torch", "ivf_faiss")
for name in methods:
    print(f"\nTraining {name}", flush=True)
    options = dict(nepochs=epochs, batch_size=batch, alpha0=alpha0, seed=seed,
                   assignment_backend="native", verbose=True)
    if name == "brute_force":
        model = QuantizedEmeans(d, k, quantizer=None, **options)
    else:
        engine = "torch" if name == "ivf_torch" else "faiss"
        model = EmeansIVF(d, k, nlist=nlist, nprobe=nprobe, ivf_engine=engine,
                         ivf_tie_policy="lowest_id" if engine == "torch" else "faiss", **options)
    start = time.perf_counter()
    centroids, labels = model.train(x, initial)
    elapsed = time.perf_counter() - start
    print(f"  Training complete: {elapsed:.2f}s, revivals={model.n_splits:,}", flush=True)
    index = faiss.IndexFlatL2(d)
    index.add(centroids.numpy())
    train_mse = exhaustive_mse(index, dataset.get_train(), "training")
    test_mse = exhaustive_mse(index, queries, "test")
    moving_mse = float(model.obj[-1] / n)
    result = dict(method=name, training_s=elapsed, train_mse=train_mse, test_mse=test_mse,
                  final_moving_pass_mse=moving_mse, revivals=model.n_splits)
    results.append(result)
    np.save(os.path.join(output_dir, name + "_centroids.npy"), centroids.numpy())
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(output_dir, name + "_training.json"), "w") as file:
        json.dump(dict(recipe=model.recipe, iteration_stats=model.iteration_stats), file, indent=2)
    del model, centroids, labels, index
    gc.collect()

print("\nMethod          Training (s)    Train MSE    Test MSE", flush=True)
for result in results:
    print(f"{result['method']:15s} {result['training_s']:11.2f} "
          f"{result['train_mse']:12.6f} {result['test_mse']:11.6f}", flush=True)
print(f"Results saved in {output_dir}", flush=True)
