# Reader-side coarse IVF routing

Host: h100-217-143, four H100 GPUs, 2026-11-04.
Frozen DINO batches: d=1024, global batch=2,097,152, k=2,097,152,
IVF4096, nprobe=4, assignment chunks=65,536.

## Implementation

Each rank routes its own 524,288 rows against the shared coarse quantizer
and global nonempty-list mask before all-gathering the vectors. A second
all-gather carries INT32 coarse-list IDs in the same group-rank/row order.
Each centroid owner passes those IDs to `search_1nn(coarse_idx=...)`, which
skips repeating query routing on the full global batch.

The list-ID matrix adds 32 MiB per GPU at these settings. Larger list counts
that do not fit INT32 use INT64. Assignment preparation still rebuilds the
local centroid index and reduces global occupancy before routing. Statistics,
winner reductions, captured labels, and EMA updates use the existing paths.
Native assignment uses its existing single row gather. Fresh-assignment
closing uses reader routing; saved-assignment closing performs no search.

## Controlled frozen-batch results

Three warmed trials per mode alternate order and use the same frozen
centroid shards and real DINO rows. The old routing is recreated only inside
the benchmark by overriding the assignment gather hook. CUDA events time
query coarse routing separately. Wall times include index rebuild, gathers,
search, winner reductions, and statistics; SSD reads, initialization, and
EMA updates are excluded. Mock call histories are cleared between trials
to avoid retaining gathered matrices in the memory measurements.

| Routing | Routed rows per GPU | Coarse routing median | Whole batch median | Gather calls |
| --- | ---: | ---: | ---: | ---: |
| Previous owner routing | 2,097,152 | 0.301741 s | 0.485069 s | 1 |
| Reader routing | 524,288 | 0.075219 s | 0.260801 s | 2 |

Coarse routing is **4.01 times faster**. Whole-batch processing is
**1.86 times faster**, or **46.2% less time**. Labels and counts are exactly
identical on every rank in every trial. Maximum allocated GPU memory was
21.593 GiB for owner routing and 21.632 GiB for reader routing.

The measured difference of 0.224267 seconds per batch extrapolates to
**214 seconds (3m34s)** over 954 training steps. This is a compute-path
extrapolation, not a measured full-run saving. Evolving centroid/list
distributions, SSD waits, and startup costs can change the realized saving.
The full 1B run subsequently confirmed a **203.8-second training saving**:
training fell from 10m55s to 7m31s and total time from 12m27s to 8m59s.
See [the full-run report](full_reader_20261104/REPORT.md).

## Correctness checks

- Three distributed tests passed: all-gather batches, saved closing means,
  and sharded centroid contracts, including raw INT8/UINT8 Triton/CuTe paths.
- IVF tests compare reader routing with the previous owner routing using
  nprobe=2, indexed partial contributions, and a ragged final batch.
- A dedicated occupancy test has every reader query lists owned entirely
  by the other centroid shard. It verifies the gathered IDs against full
  routing, one local routing call, two gathers, and the expected sums/counts.
- The two-GPU SSD demo self-test passed with BF16/FP32 evaluation and
  saved-assignment closing.

## Reproduction

From `benchs/e_means/ssd_read_bench` on the data host:

```sh
torchrun --standalone --nproc_per_node=4 bench_coarse_routing.py \
  --output results/coarse_routing_20261104.json
```

Raw measurements: `coarse_routing_20261104.json`.
