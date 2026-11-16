# SyntheticDataset CPU benchmarks

Measured on 2026-11-16 using Intel Xeon Platinum 8462Y+ CPUs and the existing
`optikm-cute` environment: Faiss 1.13.2, Torch 2.14.0+cu130, NumPy 2.4.1.
All runs used CPU computation.

## Common configuration and measurements

- `SyntheticDataset(256, 2_000_000, 0, 50_000, seed=1234)`:
  2M training vectors, 256 dimensions, 50K held-out test queries.
- 32,768 centroids, 3 epochs, seed 1234.
- Global batch 131,072, initial EMA step `0.3581079200175551`.
- IVF256, nprobe=4, 10 coarse training iterations.
- FP32 data, search and statistics; cost-based revival and final closing means.

Training time includes coarse training, repeated index builds, assignment,
statistics, revival, communication where applicable, shuffling and closing.
Data generation/loading, final quality evaluation and file output are excluded.
Final MSE uses exhaustive FP32 `faiss.IndexFlatL2` against the final centroids,
over all training vectors and all test queries. Squared distances are FP32;
their aggregate is summed in FP64.

Torch IVF uses lowest-ID ties. Faiss IVF explicitly opts into
`ivf_tie_policy="faiss"`, which follows the native scanner's tie contract.
These are individual measurements, rather than repeated timing trials.

## Exact versus IVF assignment: one worker, 32 threads

Exact assignment and IVF ran in separate Slurm jobs on different hosts with
the same CPU model. Each allocation exposed 64 logical CPUs / 32 physical
cores, with Torch and Faiss configured for 32 threads. The jobs used identical
data, initial centroids, epoch permutations and the local trainer's default
Torch Oja initialization.

| Assignment | Training time | Speedup vs exact | Train MSE | Test MSE |
|---|---:|---:|---:|---:|
| Exhaustive Torch | 245.30 s | 1.00× | 59.195248 | 63.246789 |
| IVF Torch | 112.56 s | 2.18× | 59.674629 | 63.288473 |
| IVF Faiss | 116.28 s | 2.11× | 59.675436 | 63.288083 |

IVF was 2.11–2.18× faster, with about 0.81% higher training MSE and 0.066%
higher test MSE. The small runtime difference between the IVF engines does
not establish which engine is generally faster.

The archived logs are `exact-32-threads.log` (job 936191, `h100-219-011`)
and `ivf-32-threads.log` (job 936205, `h100-215-007`). See the gist below.

## One 128-thread worker versus four 32-thread workers

Both configurations ran sequentially in Slurm allocation **945590**, on
`h100-193-149`. They used exactly the same 128 logical CPUs / 64 physical
cores with SMT. The four workers each had a disjoint affinity of 32 logical
CPUs / 16 physical cores; their union matched the single worker's affinity.

The configurations shared data, initial centroids, CPU PCG64 Oja directions,
global batches and epoch permutations. The local benchmark overrides only
the Oja initializer to match the replicated trainer's stream. This differs
from the default local initialization used in the earlier 32-thread runs.

Distributed training used `DistributedEmeansIVF`, full centroid replicas,
Gloo TCP loopback and dense statistics reductions. Each rank processed
500,000 vectors per epoch, contributing one quarter of each global batch
(local batch 32,768). The single worker used `EmeansIVF` with local batch
131,072 and no training collectives. Torch and Faiss ran sequentially within
each configuration.

| IVF engine | Workers × threads | Training time | Train MSE | Test MSE | Revivals |
|---|---:|---:|---:|---:|---:|
| Torch | 1 × 128 | 109.47 s | 59.680533 | 63.379447 | 10,667 |
| Torch | 4 × 32 | 35.34 s | 59.677414 | 63.356919 | 10,668 |
| Faiss | 1 × 128 | 113.54 s | 59.681574 | 63.328883 | 10,677 |
| Faiss | 4 × 32 | 32.56 s | 59.684299 | 63.323892 | 10,728 |

Four workers were **3.10× faster with Torch IVF** and **3.49× faster with
Faiss IVF**, relative to one worker with the same total thread count.
All four final centroid replicas were finite and bit-for-bit identical for
each engine. Reduction order can change the optimization trajectory, so
local and distributed runs need not produce identical centroids.

The measured speedup includes worker decomposition, thread scaling and NUMA
placement. Loopback communication does not measure network performance
between machines.

The archived logs are `single-128-threads.log` and
`distributed-4x32-threads.log`.

## Archived logs and metadata

GitHub gist **`mdouze/4e9dabe4ff45f65be005f6c62428eaf7`** contains the four
logs named above, configuration metadata including CPU affinities, and full
CSV results for both comparisons. The metadata files use the corresponding
log name with a `-metadata.json` suffix.

## Reproduce

Activate a Python environment with Faiss and Torch. For exact versus IVF,
run these commands in two separate CPU jobs with 32 available physical cores:

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 python benchs/e_means/analysis/bench_synthetic_ivf_vs_bruteforce.py exact
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 python benchs/e_means/analysis/bench_synthetic_ivf_vs_bruteforce.py ivf
```

For the comparison with matched CPU resources, submit from the checkout root
using your Slurm partition and account:

```bash
sbatch --partition=... --qos=... --account=... \
  benchs/e_means/analysis/run_synthetic_cpu_comparison.slurm
```

The launcher runs one 128-thread worker and then four 32-thread workers in
the same allocation. The shared dataset is cached under
`~/.cache/faiss-emeans-cpu-ivf`. Generated CSVs, metadata, logs and centroid
arrays remain local in the ignored `analysis/*_results/` directories.
