# All-gather training batches

The sharded training demos use all-gather without a transport CLI option.
The library always uses all-gather; the broadcast batch implementation and
transport selector have been removed. The transport is recorded in the recipe.

Every accumulation call gathers rank-local rows into one global matrix in
group-rank order. IVF now routes the local rows first and also gathers compact
coarse-list IDs, which search consumes without rerouting the global matrix.
Local centroid shards assign the gathered rows, then one
distance reduction and one global-ID reduction choose winners. IVF searches
all gathered queries together within its existing workspace budget. Native
assignment remains split into bounded compute chunks. Owned statistics are
accumulated in the existing source/chunk order; captured labels contain only
the caller's supplied rows. Partial accumulation, indexed requests, ragged
final batches, integer tie handling, and saved-code closing are preserved.

## Correctness

- The original two-GPU transport tests compared broadcast and all-gather labels, counts,
  sums, objectives, EMA updates, final means, and retained label buffers.
  They cover IVF, ordinary and spherical assignment, INT8/UINT8, indexed
  partial accumulation, and ragged batches.
- After broadcast removal, tests compare whole with partial indexed
  accumulation, and the existing independent sharded-statistics tests run
  through the all-gather path.
- Collective-count tests verify one batch all-gather and two winner reductions
  for native assignment, plus one list-ID gather and one occupancy reduction
  for IVF, with no batch broadcasts.
- The two-GPU SSD self-test passes with all-gather and saved-code closing.
- A four-GPU, two-epoch real training smoke run (100K DINO vectors, k=4096,
  global batch=65536, IVF64/nprobe4) passes. Compared with the prior broadcast
  run, final normalized centroid differences are at most 9.54e-7; test MSE
  changes by about 2.73e-7 on a value of 237632.66. These are within FP32
  rounding tolerances. Raw comparison: `faiss_all_gather_smoke.json`.

## Target-size frozen-batch benchmark

Host h100-217-143, 2026-11-03. Four H100 GPUs, d=1024, global batch and
centroid count both 2,097,152, IVF4096/nprobe4, assignment chunks=65536.
Actual DINO vectors and fixed centroid shards are used. Both paths are
warmed; three trials alternate transports. Timing includes IVF rebuild,
communication, assignment, and statistics. SSD reads, initialization,
evaluation, and EMA updates are excluded.
These historical comparison results were recorded before removing broadcast.

| Transport | Slowest-rank times (s) | Median (s) | Batch exchanges | Reductions |
| --- | --- | ---: | --- | ---: |
| Broadcast | 0.9170, 0.9085, 0.9177 | 0.9170 | 32 broadcasts | 65 |
| All-gather | 0.4802, 0.4853, 0.4843 | 0.4843 | 1 all-gather | 3 |

All-gather is **1.89 times faster**, or **47.2% less time** in this benchmark.
Labels and counts are identical on every rank in all three comparisons.
The reduction count includes one existing IVF global occupancy reduction:
winner selection accounts for the other two all-gather reductions.

The gathered FP32 matrix adds 8 GiB per rank at this size, plus additional
query and search temporaries. Maximum allocated GPU memory across all ranks
and trials was **29.61 GiB broadcast** and **41.66 GiB all-gather**. Rank 0
alone peaked around 15.5/27.6 GiB; memory differs across centroid owners.
Startup logs report the gathered matrix size, not total peak GPU memory.

This benchmark does not measure a new end-to-end billion-vector training run.
At the time of these measurements, coarse IVF routing was computed on every
centroid owner. Reader routing is now implemented and measured separately in
[COARSE_ROUTING.md](COARSE_ROUTING.md). A packed single winner reduction
remains a possible change.
Raw measurements: `faiss_batch_transport_20261103.json`.
