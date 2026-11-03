# Billion-vector SSD reader benchmark

These are the initial measurements, before the production planner replacement.
The follow-up Faiss run with the reference planner took **92.98 seconds** for
shuffled loading, versus 146.44 seconds here (36.5% less time). The same
benchmark configuration, requests, and physical read bytes were used, and all
320 sampled payload checks passed. Details are in
`ssd_read_faiss_reference_planner_1b.json`. No training or GPU work was timed.

Host: h100-217-143. Date: 2026-11-03. Each run delivered all 1,000,000,000
DINO vectors (1024 uint8 dimensions) using four CPU workers. All timed data
reads used O_DIRECT; Linux caches were not flushed, as authorized by the user.

| Reader | Access | Read time (s) | Setup (s) | Payload (GB/s) | Disk (GB/s) | Disk bytes (TB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reference | Sequential | 70.27 | 17.31 | 14.57 | 14.57 | 1.024000 |
| Faiss | Sequential | 32.17 | 16.65 | 31.83 | 31.95 | 1.028055 |
| Reference | Shuffled | 98.00 | 27.10 | 10.45 | 20.90 | 2.048000 |
| Faiss | Shuffled | 146.44 | 27.10 | 6.99 | 14.04 | 2.056110 |

GB and TB are decimal. Disk bytes come from each worker's `/proc/self/io`
`read_bytes` counter, sampled only around loading calls. Read time sums the
slowest rank's duration for each window; setup is the maximum across ranks.
Detailed window times and per-rank statistics are in the adjacent JSON files.

## Method

- One reader and four extraction threads per worker, 64 MiB read chunks.
- One pageable 50 GiB output window per worker, allocated and first-touched
  before timing. Five windows per worker, including the partial final window.
- Shuffled runs use four chunk groups and seed 1 plus worker rank. Each
  permutation respects its source's native physical chunk boundaries; the
  distribution matches, but the exact row permutations differ.
- Reference calls `online_k-means/ssd_prefetch.py::fill_window` on four staged
  flat uint8 shards. Faiss calls `SSDPrefetch.fill_window` or
  `SSDPrefetch.fill_sequential` on original bvecs, using the current borrowed
  views and reusable read buffers. Its CUDA constructor is bypassed.
- No training, GPU transfer, pinned allocation, or GPU assignment is timed.
  Request planning and extraction are included. Order construction and
  allocation are setup. Validation and rank barriers are outside read time.
- Each run checked 16 dispersed payload rows per window per worker against
  the original bvecs: 320 checks per run. All checks passed, all runs exited.
- Reference staging took 702.6 seconds and is excluded from reader times.
  Its 1.024 TB of new shards remain in
  `/scratch/train-cache/ssd-bench-1b-20261103` for reuse.

## Interpretation

Faiss's sequential reading path delivers the corpus in 32.2 seconds, showing
substantial storage headroom relative to the previous multi-minute closing
pass. Reading alone does not explain that closing time; this benchmark omits
assignment, distributed communication, and GPU transfer.

Faiss's shuffled reader takes 1.49 times the reference's time. Both shuffled
layouts read about twice the source corpus: the final partial window needs
chunks from all four groups. Avoiding rereads and reducing request planning
and extraction work remain useful targets. These measurements do not isolate
the cause of the remaining difference between implementations.

The reference sequential result measures its general prefetch reader, which
still plans requests and scatters rows. It does not measure the specialized
contiguous reader used by the reference closing pass. These are single runs,
not repeated trials or an end-to-end training comparison.
