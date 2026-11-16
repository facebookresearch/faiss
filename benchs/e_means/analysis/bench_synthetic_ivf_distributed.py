# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare one 128-thread CPU worker with four 32-thread CPU/Gloo workers.

Uses the same global data, initialization and schedule as
bench_synthetic_ivf_vs_bruteforce.py. Centroids are replicated; each worker
supplies one quarter of each global batch. Both IVF engines run sequentially.
"""

import csv
import datetime
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
import torch.distributed as dist

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
faiss.contrib.__path__.insert(0, os.path.join(repo_root, "contrib"))
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.e_means import DistributedEmeansIVF, EmeansIVF
from faiss.contrib.e_means.base import _epoch_seed, _resolve_recipe
from faiss.contrib.e_means.e_means_distributed import _fill_initial_axes


class BenchmarkEmeansIVF(EmeansIVF):
    """Use the replicated trainer's initial Oja directions for comparison."""

    def _initial_removal_axes(self):
        axes = torch.empty_like(self._centroids)
        _fill_initial_axes(axes, np.random.Generator(np.random.PCG64(self.seed)))
        return axes

rank = int(os.environ["SLURM_PROCID"])
ranks = int(os.environ["SLURM_NTASKS"])
job = os.environ["SLURM_JOB_ID"]
threads, seed = int(os.environ["SLURM_CPUS_PER_TASK"]), 1234
n, k, d, epochs, nq = 2_000_000, 1 << 15, 256, 3, 50_000
nlist, nprobe = 256, 4
if (ranks, threads) not in ((4, 32), (1, 128)):
    raise ValueError("launch four 32-thread tasks or one 128-thread task")
affinity = sorted(os.sched_getaffinity(0))
if len(affinity) < threads:
    raise RuntimeError(f"rank {rank}: need {threads} available CPUs")
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)
faiss.omp_set_num_threads(threads)
os.environ["GLOO_SOCKET_IFNAME"] = "lo"

cache_root = "/storage/home/matthijs/.cache/faiss-emeans-cpu-ivf"
os.makedirs(cache_root, exist_ok=True)
dist.init_process_group("gloo", rank=rank, world_size=ranks,
    init_method="file://" + os.path.join(cache_root, "synthetic_gloo_" + job),
    timeout=datetime.timedelta(minutes=20))

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "synthetic_ivf_distributed_results" if ranks == 4
                         else "synthetic_ivf_single128_results")
data_dir = os.path.join(cache_root, "synthetic_n2M_k32K_d256_seed1234")
physical = set()
for logical in affinity:
    topology = f"/sys/devices/system/cpu/cpu{logical}/topology"
    with open(topology + "/physical_package_id") as file:
        package = int(file.read())
    with open(topology + "/core_id") as file:
        physical.add((package, int(file.read())))
cpu = "unknown"
with open("/proc/cpuinfo") as file:
    for line in file:
        if line.startswith("model name"):
            cpu = line.partition(":")[2].strip()
            break
local_metadata = dict(rank=rank, host=platform.node(), cpu=cpu,
    threads=threads, affinity=affinity, physical_cores=len(physical))
metadata_parts = [None] * ranks
dist.all_gather_object(metadata_parts, local_metadata)
if len({part["host"] for part in metadata_parts}) != 1:
    raise ValueError("this benchmark simulates four workers on one machine")
if any(set(part["affinity"]) & set(other["affinity"])
       for i, part in enumerate(metadata_parts) for other in metadata_parts[i + 1:]):
    raise ValueError("worker logical CPU affinities must be disjoint")

batch, alpha0 = _resolve_recipe(n, k, epochs, "throughput", None, None)
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    if not all(os.path.exists(os.path.join(data_dir, name + ".npy"))
               for name in ("train", "test", "initial")):
        start = time.perf_counter()
        dataset = SyntheticDataset(d, n, 0, nq, seed=seed)
        initial = dataset.get_train()[np.random.RandomState(seed).permutation(n)[:k]]
        for name, array in (("train", dataset.get_train()), ("test", dataset.get_queries()), ("initial", initial)):
            np.save(os.path.join(data_dir, name + ".npy"), array)
        del dataset, initial
        gc.collect()
        print(f"Generated and cached data in {time.perf_counter() - start:.2f}s", flush=True)
    metadata = dict(job=job, workers=metadata_parts, ranks=ranks,
        threads_per_rank=threads, n=n, k=k, d=d, epochs=epochs, nq=nq,
        seed=seed, nlist=nlist, nprobe=nprobe, global_batch=batch,
        local_batch=batch // ranks, alpha0=alpha0, transport="dense",
        communication="Gloo TCP loopback; replicated centroids" if ranks > 1 else "local EmeansIVF; no training collectives",
        initial_axes="common CPU PCG64 stream for local and replicated training",
        faiss_version=faiss.__version__, torch_version=torch.__version__, numpy_version=np.__version__,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip())
    with open(os.path.join(output_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)
    print(f"Host={platform.node()}, {cpu}, {ranks} ranks × {threads} threads; "
          f"global batch={batch:,}, local batch={batch // ranks:,}", flush=True)
    for part in metadata_parts:
        print(f"  rank {part['rank']}: {len(part['affinity'])} logical CPUs / "
              f"{part['physical_cores']} physical cores", flush=True)
dist.barrier()

# Copy-on-write memory maps share the file-backed pages between the workers.
train = np.load(os.path.join(data_dir, "train.npy"), mmap_mode="c")
queries = np.load(os.path.join(data_dir, "test.npy"), mmap_mode="c")
initial = torch.from_numpy(np.load(os.path.join(data_dir, "initial.npy"), mmap_mode="c"))
x = torch.from_numpy(train)
results = []


def exhaustive_mse(index, rows):
    total = 0.
    for begin in range(0, len(rows), 8192):
        distance, _ = index.search(rows[begin:begin + 8192], 1)
        total += distance.sum(dtype=np.float64)
    return float(total / len(rows))


try:
    for engine in ("torch", "faiss"):
        if rank == 0:
            print(f"\nTraining replicated IVF256/4 ({engine})", flush=True)
        trainer = DistributedEmeansIVF if ranks > 1 else BenchmarkEmeansIVF
        distributed_options = dict(group=dist.group.WORLD, transport="dense") if ranks > 1 else {}
        model = trainer(d, k, **distributed_options,
            nepochs=epochs, batch_size=batch, alpha0=alpha0,
            seed=seed, assignment_backend="native", verbose=True,
            nlist=nlist, nprobe=nprobe, ivf_engine=engine,
            ivf_tie_policy="lowest_id" if engine == "torch" else "faiss")
        dist.barrier()
        start = time.perf_counter()
        model.begin(n, initial)
        for epoch, budget in enumerate(model.recipe["epoch_rows"]):
            order = torch.randperm(n, generator=torch.Generator().manual_seed(_epoch_seed(seed, epoch)))
            for begin in range(0, budget, batch):
                size = min(batch, budget - begin) // ranks
                first = begin + rank * size
                indices = order[first:first + size]
                model.update(model.accumulate(x[indices]))
                if rank == 0 and (model._epoch_steps == 1 or model._epoch_steps % 4 == 0):
                    print(f"  epoch={epoch + 1}/{epochs}, step={model._epoch_steps}/"
                          f"{model.recipe['epoch_steps'][epoch]}, elapsed={time.perf_counter() - start:.1f}s, "
                          f"revivals={model.n_splits:,}", flush=True)
            model.end_epoch()
        centroids = model.finalize()
        dist.barrier()
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

        # Verify identical final replicas outside the training timer.
        reference = centroids.clone()
        dist.broadcast(reference, src=0)
        equal = torch.tensor(int(torch.equal(reference, centroids) and torch.isfinite(centroids).all()), dtype=torch.int64)
        dist.all_reduce(equal, op=dist.ReduceOp.MIN)
        if not equal.item():
            raise AssertionError("final centroid replicas differ or contain non-finite values")
        if rank == 0:
            print(f"  Training complete: {elapsed.item():.2f}s; replicas identical; "
                  f"revivals={model.n_splits:,}", flush=True)
            index = faiss.IndexFlatL2(d)
            index.add(centroids.numpy())
            eval_start = time.perf_counter()
            train_mse = exhaustive_mse(index, train)
            test_mse = exhaustive_mse(index, queries)
            print(f"  Exhaustive MSE: train={train_mse:.6f}, test={test_mse:.6f}; "
                  f"evaluation={time.perf_counter() - eval_start:.2f}s", flush=True)
            result = dict(engine=engine, ranks=ranks, threads_per_rank=threads,
                training_s=elapsed.item(), train_mse=train_mse, test_mse=test_mse,
                final_moving_pass_mse=float(model.obj[-1] / n), revivals=model.n_splits,
                replicas_identical=True)
            results.append(result)
            np.save(os.path.join(output_dir, engine + "_centroids.npy"), centroids.numpy())
            with open(os.path.join(output_dir, "results.csv"), "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(result))
                writer.writeheader()
                writer.writerows(results)
            with open(os.path.join(output_dir, engine + "_training.json"), "w") as file:
                json.dump(dict(recipe=model.recipe, iteration_stats=model.iteration_stats), file, indent=2)
        dist.barrier()
        del model, centroids, reference
        gc.collect()
    if rank == 0:
        print("\nEngine       Training (s)    Train MSE    Test MSE", flush=True)
        for result in results:
            print(f"{result['engine']:10s} {result['training_s']:12.2f} "
                  f"{result['train_mse']:12.6f} {result['test_mse']:11.6f}", flush=True)
        print(f"Results saved in {output_dir}", flush=True)
finally:
    dist.destroy_process_group()
