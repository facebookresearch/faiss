# Full 1B / 2Mi clustering with reader-side routing

Completed successfully on h100-217-143, four H100 GPUs, 2026-11-04.
All four ranks reported success; the process exited with status 0.
The current code includes reader-side coarse routing and fused BF16 query
evaluation, on top of commit 947d318b8 / PR #5832.

## Timing comparison

| Phase | Previous Faiss, BF16 evaluation | Current Faiss, reader routing | Reference |
| --- | ---: | ---: | ---: |
| Two training epochs, including loading and preparation | 655.3 s (10m55s) | 451.5 s (7m31s) | 412.9 s (6m53s) |
| Recorded query evaluation | 1.277 s | 1.298 s | 0.4 s |
| Saved-assignment closing over 1B rows | 68.9 s (1m09s) | 71.2 s (1m11s) | 182.6 s (3m03s) |
| Initialization and other overhead | approximately 22.0 s | approximately 15.2 s | approximately 52.3 s |
| **Total** | **747.5 s (12m27s)** | **539.2 s (8m59s)** | **648.3 s (10m48s)** |

Reader routing reduces measured training time by **203.8 s (3m24s), 31.1%**.
The total falls by **208.3 s (3m28s), 27.9%** versus the previous Faiss run.
It is now **16.8% faster overall** than the measured reference run, saving
**109.1 s (1m49s)**. Closing remains approximately 2.57 times faster than
the reference; training is still **9.3% slower**, a gap of **38.5 s**.

The seven-minute training target is missed by about **31.5 s**. Closing
comfortably meets the 3.5-minute target.

## Where the remaining training gap occurs

| Epoch, training including loading and preparation | Faiss | Reference |
| --- | ---: | ---: |
| 1 | 243.7 s | 205.5 s |
| 2 | 207.8 s | 207.4 s |

Almost all of the remaining training gap occurs in epoch 1, which includes
first-time loader/index setup. Epoch 2 is within approximately 0.3 s of the
reference. These totals do not separately attribute setup costs.

Steady Faiss batches take approximately **0.30 s** in epoch 1 and
**0.32–0.33 s** in epoch 2, down from approximately 0.52 and 0.55–0.56 s.
Window boundaries still produce pauses. First-batch readiness took
**59.566 s** and **15.480 s** in the two epochs, excluding permutation
construction of **9.885 s** and **8.896 s**. Query evaluations took
**0.436, 0.409, and 0.453 s**.

Training excludes query evaluation and closing. It is derived from the
maximum final cumulative iteration time minus the preceding evaluation
duration. Epoch 2 is the difference between that training total and the
first epoch. Independent rank maxima and rounded timers make the totals
approximate. Reference timings are from the earlier successful full run,
not a new reference rerun. Precision, initialization, and evaluation schedule
differences remain as documented in the earlier comparisons.

## Command

The existing diagnostic wrapper times the normal demo operations:

```sh
python -u /tmp/faiss_epoch_diag_20261102.py \
  --launch 4 --data /tmp/DINO10B \
  --n 1000_000_000 --k 2097152 --epochs 2 \
  --nlist 4096 --nprobe 4 --seed 1 --final-closing-mean \
  --batch-size 2097152 --ssd-ram-gib 100 --ssd-read-threads 4 \
  --query-precision bf16
```

All arguments match the preceding BF16 Faiss run. Reader routing is selected
by the implementation, recorded as `coarse_routing="reader"` in the recipe.
The wrapper adds phase timing and synchronization without changing clustering.
Streaming uses O_DIRECT; Linux caches were not flushed. Startup confirms
32 MiB of gathered coarse-list IDs and an 8 GiB gathered FP32 matrix per GPU,
100 GiB of pinned SSD windows per rank, and 100 prefetched batches per window.
Each epoch executes all 477 steps, with alpha0=0.040624309347889555.

## Quality and memory

| Metric, original-coordinate squared L2 per vector | Previous Faiss | Current Faiss |
| --- | ---: | ---: |
| Final held-out MSE, all 100K queries | 147480.3540 | 147503.1045 |
| Fixed-code closing MSE | 141548.7515 | 141548.5347 |
| Moving-pass MSE | 142578.5376 | 142579.2023 |

Quality remains similar: final held-out MSE differs by about 0.0154%.
These full runs are not bitwise reproducibility checks; the controlled
frozen-batch routing comparison found identical labels and counts.

Maximum allocated GPU memory across ranks remains **34.46 GiB**. Closing
covers all **1,000,000,000 rows** from saved final-epoch labels without search.

## Artifacts

- `faiss.log`: full stdout and diagnostic phase timings.
- `faiss.json`: successful reports from all four ranks.
- `faiss_phases.json`: extracted diagnostic durations.
- `summary.json`: totals, epoch training times, comparison ratios, and quality.

See also [the prior BF16 run](../full_bf16_20261103/REPORT.md) and
[the frozen-batch routing comparison](../COARSE_ROUTING.md).
