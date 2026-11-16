# SyntheticDataset: one 128-thread worker versus four 32-thread workers

Measured on 2026-11-16 in Slurm allocation **945590**, on `h100-193-149`
(Intel Xeon Platinum 8462Y+). Both configurations used the same 128 logical
CPUs / 64 physical cores with SMT. They ran sequentially, without GPUs.
The four workers had disjoint affinities of 32 logical CPUs / 16 physical
cores each; their union exactly matched the single worker's affinity.

## Configuration

- SyntheticDataset: 2,000,000 training vectors, 256 dimensions, 50,000 test queries.
- 32,768 centroids, 3 epochs, seed 1234, FP32 data/search/statistics.
- IVF256, nprobe=4, 10 coarse training iterations; cost-based revival and closing.
- Global batch 131,072; local batch 131,072 for one worker or 32,768 for four.
- Same training data, initial centroids, CPU PCG64 Oja directions, epoch
  permutations and EMA schedule (`alpha0=0.3581079200175551`).
- Local training uses `EmeansIVF` with a benchmark-only initializer override
  matching the replicated trainer's Oja stream. It performs no training collectives.
- Distributed training uses `DistributedEmeansIVF`, replicated centroids and
  Gloo TCP loopback with dense reductions. Each rank processes 500,000 rows
  per epoch, contributes its quarter of each global batch, and updates all
  centroids from the globally reduced statistics and synchronized revival.
- Torch and Faiss engines run sequentially for each configuration.

## Results

| IVF engine | Workers × threads | Training time | Train MSE | Test MSE | Revivals |
|---|---:|---:|---:|---:|---:|
| Torch | 1 × 128 | 109.47 s | 59.680533 | 63.379447 | 10,667 |
| Torch | 4 × 32 | 35.34 s | 59.677414 | 63.356919 | 10,668 |
| Faiss | 1 × 128 | 113.54 s | 59.681574 | 63.328883 | 10,677 |
| Faiss | 4 × 32 | 32.56 s | 59.684299 | 63.323892 | 10,728 |

Four workers were **3.10× for Torch IVF**, **3.49× for Faiss IVF**, relative to one worker with the same total
thread count. All four final centroid replicas were finite and bit-for-bit
identical for each engine.

Training time includes coarse training, index rebuilding, assignment,
statistics, revival, communication, shuffling and closing. Data loading,
final MSE evaluation and file output are excluded. MSE uses all training
rows and all 50,000 held-out queries, with exhaustive FP32 `faiss.IndexFlatL2`
against the final centroids. Squared distances are FP32; their aggregate is
summed in FP64. Distributed reduction changes floating-point summation order,
so the local and distributed optimizations need not end at identical centroids.

These are single measurements, not repeated trials. The speedup includes
the effects of worker decomposition, thread scaling and NUMA placement.
Loopback communication does not measure network performance between hosts.

Full values are in [comparison.csv](comparison.csv), [distributed results](results.csv)
and [single-worker results](../synthetic_ivf_single128_results/results.csv).
Metadata, affinities, per-run statistics and stdout are saved alongside them.
Centroid arrays are written locally and excluded from Git.

## Earlier 32-thread exact versus IVF reference

See [the earlier benchmark](../synthetic_ivf_vs_bruteforce_results/README.md)
for the 245.30 s exhaustive run and 112.56 s / 116.28 s local IVF runs.
Those runs used the local trainer's default Torch Oja initialization and
different hosts. The table above is the comparison with matched resources
and initial state; it replaces the earlier comparison against 32 threads.

## Reproduce

Activate the Python environment with Faiss and Torch, then submit from the
checkout root using your Slurm partition/account:

```bash
sbatch --partition=... --qos=... --account=... \
  benchs/e_means/analysis/run_synthetic_cpu_comparison.slurm
```

The launcher runs both configurations sequentially in one 128-CPU allocation.
The script caches the shared dataset under `~/.cache/faiss-emeans-cpu-ivf`.
Measurements used `optikm-cute`: Faiss 1.13.2, Torch 2.14.0+cu130, NumPy 2.4.1.
