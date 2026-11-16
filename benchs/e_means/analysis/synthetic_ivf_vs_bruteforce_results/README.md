# SyntheticDataset: CPU IVF versus exhaustive assignment

Measured on 2026-11-16 in `optikm-cute`, with 32 threads and FP32 computation.
Two separate Slurm jobs ran on Intel Xeon Platinum 8462Y+ CPUs:

- Exact: job 936191, `h100-219-011`.
- IVF (Torch and Faiss sequentially): job 936205, `h100-215-007`.
- Each allocation exposed 64 logical CPUs / 32 physical cores.

Both jobs generated identical `SyntheticDataset(256, 2_000_000, 0, 50_000, seed=1234)` data.
The runs used 32,768 centroids, three epochs, seed 1234, the same initial centroids and epoch permutations,
batch size 131,072 and initial EMA step 0.3581079200. IVF used 256 lists and four probes.
Cost-based revival and ordinary final closing means were enabled.

| Assignment | Training time | Speedup | Training MSE | Test MSE | Training MSE change | Test MSE change |
|---|---:|---:|---:|---:|---:|---:|
| Exhaustive (Torch) | 245.30 s | 1.00× | 59.195248 | 63.246789 | +0.000% | +0.000% |
| IVF256/4 (Torch) | 112.56 s | 2.18× | 59.674629 | 63.288473 | +0.810% | +0.066% |
| IVF256/4 (Faiss) | 116.28 s | 2.11× | 59.675436 | 63.288083 | +0.811% | +0.065% |

Training time includes coarse training, repeated index builds, assignment, statistics, revival, shuffling and closing.
Data generation, final quality evaluation and file output are excluded.
Training MSE uses every training vector; test MSE uses all 50,000 held-out queries.
All final MSE values use the same exhaustive FP32 `faiss.IndexFlatL2` evaluator against the final centroids.
The moving-pass MSE and revival counts are recorded separately in CSV and per-run training JSON.

This is one run per configuration. The 3.3% runtime difference between the IVF engines needs repeated timing
before drawing a general conclusion about which CPU engine is faster.
Faiss IVF explicitly uses its scanner tie policy (`ivf_tie_policy="faiss"`); Torch uses lowest-ID ties.

## Reproduce

Run the following in two separate CPU jobs with at least 32 available physical cores each:

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 python benchs/e_means/analysis/bench_synthetic_ivf_vs_bruteforce.py exact
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 python benchs/e_means/analysis/bench_synthetic_ivf_vs_bruteforce.py ivf
```

`exact/` and `ivf/` contain metadata, per-model training records and results.
Final centroid arrays are written locally and excluded from Git.
The benchmark uses the Python implementation in this checkout and the installed Faiss compiled core.
Faiss 1.13.2, PyTorch 2.14.0+cu130, NumPy 2.4.1.
