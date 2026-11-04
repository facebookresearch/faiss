# Full 1B / 2Mi clustering with fused BF16 evaluation

Completed successfully on h100-217-143 on four H100 GPUs, 2026-11-03.
All four ranks reported success, and the process exited with status 0.
Code: commit 947d318b8, rebased on PR #5832, plus the current BF16 evaluation changes.

## Timing comparison

| Phase | Previous Faiss, FP32 evaluation | Current Faiss, BF16 evaluation | Reference |
| --- | ---: | ---: | ---: |
| Two training epochs, including loading and preparation | 651.5 s (10m51s) | 655.3 s (10m55s) | 412.9 s (6m53s) |
| Recorded query evaluation | 219.2 s (3m39s) | 1.277 s | 0.4 s |
| Saved-assignment closing over 1B rows | 69.9 s (1m10s) | 68.9 s (1m09s) | 182.6 s (3m03s) |
| Initialization and other overhead | approximately 15.6 s | approximately 22.0 s | approximately 52.3 s |
| **Total** | **956.1 s (15m56s)** | **747.5 s (12m27s)** | **648.3 s (10m48s)** |

The new total saves **208.6 s (3m29s)**, or **21.8%**, versus the previous
Faiss run. It is now **15.3% slower overall** than the reference, a gap of
**99.2 s (1m39s)**. The evaluation time fell by approximately **171.6 times**
in the actual full run. Training time changed by approximately 0.6%.

The seven-minute training target is still unmet. Closing remains below the
3.5-minute target and is approximately 2.65 times faster than the reference.

Query evaluations took **0.453, 0.414, and 0.410 seconds**, including BF16
mirror preparation and query transfers. First-batch readiness took 63.583 s
in epoch 1 and 15.134 s in epoch 2, excluding permutation construction of
10.029 and 9.575 s. Steady training batches remained approximately 0.52 s
in epoch 1 and 0.55–0.56 s in epoch 2.

Training excludes query evaluation and closing. It is derived from the maximum
final cumulative iteration time minus the preceding evaluation duration.
Rank maxima and rounded diagnostic timers make phase totals approximate.
The reference timings are from the earlier successful full run, not a new
reference rerun. Its evaluation precision and schedule differ as described
in [the previous comparison](../full_current_20261103/REPORT.md).

## Command

The existing diagnostic wrapper was used to time the normal demo operations:

```sh
python -u /tmp/faiss_epoch_diag_20261102.py \
  --launch 4 --data /tmp/DINO10B \
  --n 1000_000_000 --k 2097152 --epochs 2 \
  --nlist 4096 --nprobe 4 --seed 1 --final-closing-mean \
  --batch-size 2097152 --ssd-ram-gib 100 --ssd-read-threads 4 \
  --query-precision bf16
```

All other benchmark parameters match the previous Faiss run. The wrapper adds
timing and synchronization at phase boundaries, without changing clustering.
The run used all-gather batches, alpha0=0.040624309347889555, 477 steps per
epoch, and saved-code closing with 65,536-row routing chunks. SSD streaming
used O_DIRECT with 100 prefetched batches per window and 100 GiB of pinned
buffers per rank. Linux caches were not flushed.

## Quality and memory

| Metric, original-coordinate squared L2 per vector | Previous Faiss | Current Faiss |
| --- | ---: | ---: |
| Final held-out MSE, all 100K queries | 147486.4144 | 147480.3540 |
| Fixed-code closing MSE | 141548.7067 | 141548.7515 |
| Moving-pass MSE | 142578.5364 | 142578.5376 |

The held-out metrics use different ranking precision. Floating training
reductions also cause small run-to-run differences, so the held-out change
cannot be attributed entirely to BF16 evaluation. Training and closing quality
are similar to the prior run.

Maximum allocated GPU memory across ranks was **34.46 GiB**. Closing covered
all **1,000,000,000 rows**, using the saved final-epoch labels without search.

## Artifacts

- `faiss.log`: complete stdout and diagnostic phase timings.
- `faiss.json`: successful reports from all four ranks.
- `faiss_phases.json`: extracted phase durations.
- `summary.json`: comparison totals and final metrics.
