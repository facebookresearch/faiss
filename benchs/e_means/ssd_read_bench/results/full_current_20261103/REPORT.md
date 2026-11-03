# Full 1B / 2Mi clustering comparison, 2026-11-03

Both jobs completed successfully on h100-217-143, sequentially on four H100s.
Faiss used the current modified code: all-gather batches, the reference mask/inverse SSD planner, and closing from saved final-pass assignments.

## Timing

| Phase | Faiss | Reference |
| --- | ---: | ---: |
| Two training epochs, including loading and epoch preparation | 651.5 s (10m51s) | 412.9 s (6m53s) |
| Recorded query evaluation | 219.2 s (3m39s) | 0.4 s |
| Final closing mean over 1B saved assignments | 69.9 s (1m10s) | 182.6 s (3m03s) |
| Initialization and remaining overhead | approximately 15.6 s | approximately 52.3 s |
| **Total** | **956.1 s (15m56s)** | **648.3 s (10m48s)** |

Faiss is **47.5% slower overall**, a gap of **5m08s**. Its training epochs are **57.8% slower**, a gap of **3m59s**. Its closing mean is **2.61 times faster**.
The reference meets the seven-minute training target. Faiss does not yet meet that target. Both meet the 3.5-minute closing target.

Training timing includes iterator loading and permutation preparation, and excludes query evaluation and closing. Faiss training is derived from the final cumulative iteration time minus the preceding query evaluation. Phase timing uses rank maxima where available; rounded diagnostic durations and independent rank maxima mean the phase totals are approximate.

## Commands

Faiss arguments, run through a timing wrapper around the normal demo:

```sh
python demo_training_from_ssd.py --launch 4 --data /tmp/DINO10B \
  --n 1000_000_000 --k 2097152 --epochs 2 \
  --nlist 4096 --nprobe 4 --seed 1 --final-closing-mean \
  --batch-size 2097152 --ssd-ram-gib 100 --ssd-read-threads 4
```

Reference arguments, run through a wrapper that saves per-rank timing reports:

```sh
torchrun --nproc_per_node=4 run_anns_optimization.py \
  --dataset dino10b-10B --n-train 1000000000 \
  --n-clusters 2097152 --n-iter 2 --batch-size 2097152 \
  --operating-point quality --seed 0 --center --dynamic-batches \
  --distributed --centroid-shards 4 \
  --index-type internal --index-factory IVF4096,Flat \
  --search-params nprobe=4 --final-closing-mean --no-save-centroids \
  --cache-train-to-scratch /scratch/train-cache/reference-full-current-20261103 \
  --keep-scratch --ssd-prefetch auto --ssd-io \
  --ssd-prefetch-chunk-groups 4 --ssd-prefetch-ram-gb 400 \
  --nccl-timeout-min 600 --log-batch-mse --verbose --verbose-every 10
```

Both read the first 1B vectors of the same DINO corpus. Reference scratch shards were already staged and verified against the source; no new 1TB staging copy is included. The SSD streaming reads use O_DIRECT, as authorized, without clearing Linux caches. Initialization and query reads are not necessarily direct reads.

## Remaining differences

- **Training throughput:** steady Faiss batches take approximately 0.52–0.56 s, versus approximately 0.32 s for the reference. The remaining training gap persists outside startup pauses.
- **Coarse routing and communication:** Faiss computes coarse query routing on every centroid owner. The reference routes on the reading rank and gathers list IDs along with the batch. Faiss still uses separate minimum-distance and minimum-label reductions; the reference uses a packed argmin reduction. These are implementation differences to address next; this full run does not individually measure their contributions.
- **Evaluation:** Faiss runs exhaustive FP32 assignment on all 100k shipped queries three times, with 4,096-centroid chunks; each evaluation takes about 73 s. The reference evaluates a 10k training sample and the two 50k query halves after each epoch, using its sharded fused assignment path and the default BF16 centroid mirror, then computes the chosen distances in FP32. The recorded 0.4 s is for the two epoch evaluations, excluding initial evaluation. The evaluation timings therefore compare different precision and execution paths as well as different evaluation schedules.
- **Host allocation:** Faiss's first batch takes 54.8 s to become ready in epoch 1 and 15.1 s in epoch 2; permutation construction takes 10.0 and 9.4 s separately. Reusing pinned buffers across epochs remains relevant.
- **Memory and reader settings:** Faiss reserves 100 GiB of pinned buffers per rank, 400 GiB across the node, with 100 prefetched batches per window and two windows. Reference uses approximately 399.4 GB decimal across the node, 93 prefetched batches, and one SSD reader per rank. Faiss uses four readers per rank. These are similar but unequal memory budgets.
- **I/O:** Faiss reports approximately 5.15 TB of total SSD reads across four ranks, including training, closing, and initialization. Reference reports approximately 2.304 TB read per training epoch, more than Faiss's approximately 2.056 TB. Thus Faiss's slower training cannot be explained by a larger training read volume. Loading still causes pauses, and scan time overlaps compute; these measurements do not attribute the whole gap to I/O.

Faiss peak GPU allocation was approximately 34.46 GiB per rank. The updated Faiss total is 2.71 times faster than the previous 43m12s full run, while still slower than the reference.

## Quality

| Metric, squared L2 per vector in original coordinates | Faiss | Reference |
| --- | ---: | ---: |
| Final held-out MSE, all 100k shipped queries | 147486.41 | 147419.77 |
| Final fixed-code closing MSE | 141548.71 | 141251.01 |

Reference held-out MSE combines its equally sized validation and test halves and restores scale squared. Reference fixed-code MSE also restores the dimension factor (1024), because that log is per coordinate. Seeds differ (1 versus 0), as do centering samples and assignment precision, so these are quality comparisons rather than an equality check.

## Artifacts

- `faiss.log`: full stdout, including startup information and diagnostic phase timers.
- `faiss.json`: successful reports from all four ranks, including iteration statistics, reader statistics, and GPU allocation.
- `faiss_phases.json`: diagnostic phase durations.
- `reference.log`: full reference stdout.
- `reference.rank0.json` through `reference.rank3.json`: timing, quality, reader, and device reports.

The wrappers add timing and reporting only; they do not alter the clustering algorithms.
