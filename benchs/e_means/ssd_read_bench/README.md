# SSD reader comparison

Run these scripts on the data host using `torchrun --nproc_per_node=4`.
`stage_reference.py` creates flat uint8 reference shards from local bvecs.
Pass `--output PATH` to all scripts; benchmarks write detailed JSON there.

`bench_reference.py` calls `ssd_prefetch.fill_window` from online_k-means.
`bench_faiss.py` calls the Faiss SSD loader's `fill_window` or `fill_sequential`.
Both load a complete billion-row corpus using 50 GiB windows per worker,
64 MiB reads, one reader and four extractors per worker by default.

Use `--mode sequential` for one sequential corpus scan, or `--mode shuffled`
for the production chunk-grouped window distribution. Shuffled permutations
respect each implementation's physical row/chunk boundaries, so physical
bytes read can differ; both deliver every vector exactly once.

Setup, allocation, and first-touch zeroing are reported separately from read
time. No GPU transfer or training is performed. The window timers include
request planning, reads, and extraction; throughput counts loader scan bytes,
kernel disk-read bytes, and delivered payload bytes separately. Sixteen rows per window per worker
are validated against the original DINO files outside the reading timer.

All timed reads require O_DIRECT. `--flush-cache` also runs `sync` and drops
Linux caches immediately before timing; it requires passwordless sudo and
fails the benchmark if flushing fails. SSD firmware caches are not reset.

Example (from this directory on h100-217-143):

```bash
TORCHRUN=/home/matthijs/.conda/envs/optikm-cute/bin/torchrun
$TORCHRUN --standalone --nproc_per_node=4 stage_reference.py --output /tmp/ssd-stage.json
$TORCHRUN --standalone --nproc_per_node=4 bench_reference.py --mode sequential --output /tmp/reference-sequential.json
$TORCHRUN --standalone --nproc_per_node=4 bench_faiss.py --mode sequential --output /tmp/faiss-sequential.json
$TORCHRUN --standalone --nproc_per_node=4 bench_reference.py --mode shuffled --output /tmp/reference-shuffled.json
$TORCHRUN --standalone --nproc_per_node=4 bench_faiss.py --mode shuffled --output /tmp/faiss-shuffled.json
```

Staging refuses to overwrite existing shards; skip that command when reusing
the prepared data. The reference uses raw 1024-byte vectors while Faiss reads
1028-byte bvecs records. The reference sequential benchmark uses the general
prefetch reader, not the specialized reader in `final_closing_mean.py`.

The completed billion-vector measurements are in [results/REPORT.md](results/REPORT.md).

For controlled planner, extraction, and layout ablations, run `isolate.py`.
It reproduces the shuffled planner/extractor around the actual Faiss
`_scan_chunks` and `read_views` functions, adding phase timers. All variants
use identical requested rows and chunk boundaries. The default tests one
50 GiB window per rank and repeats the baseline. A full-corpus confirmation:

```bash
$TORCHRUN --standalone --nproc_per_node=4 isolate.py --bind-cpus --windows 5 --variants baseline,reference-plan,read-only --output /tmp/isolate-full.json
```

`--bind-cpus` partitions available physical CPU cores between workers,
keeping SMT siblings together and respecting the inherited affinity.
Read-only variants skip extraction and payload validation. Other variants
validate sixteen requested rows per window per worker. Thread timing sums
are concurrent worker durations, not additive wall-clock phases.
See [results/ISOLATION.md](results/ISOLATION.md).

`bench_closing.py` measures the production saved-assignment closing path on
real DINO SSD data with uniform synthetic CPU codes, excluding training and
code generation. It forbids assignment calls and checks observed counts.
Synthetic labels do not represent trained cluster quality or imbalance.

```bash
$TORCHRUN --standalone --nproc_per_node=4 bench_closing.py --output /tmp/closing-1b.json
```

See [results/CLOSING.md](results/CLOSING.md) for results and correctness checks.

`bench_batch_transport.py` times all-gather accumulation using fixed centroid
shards and real DINO vectors at the target batch/centroid dimensions. It warms
the path, then runs three timed trials; SSD reads, initialization, and EMA
updates are excluded. The original broadcast comparison remains in the saved
results; its implementation has been removed.

```bash
$TORCHRUN --standalone --nproc_per_node=4 bench_batch_transport.py --output /tmp/batch-transport.json
```

See [results/BATCH_TRANSPORT.md](results/BATCH_TRANSPORT.md).

`bench_coarse_routing.py` alternates the previous owner routing with reader
routing on identical frozen IVF4096 batches at 2Mi rows and centroids. It
checks identical labels/counts, counts routed rows and gathers, and times
coarse routing separately with CUDA events. The previous routing is recreated
only inside this benchmark, by overriding the assignment gather hook.

```bash
$TORCHRUN --standalone --nproc_per_node=4 bench_coarse_routing.py --output /tmp/coarse-routing.json
```

See [results/COARSE_ROUTING.md](results/COARSE_ROUTING.md).

`bench_query_mse.py` compares exhaustive FP32 Triton query assignment with
fused BF16 assignment at 2Mi centroids and 100K shipped DINO queries. It uses
frozen centroids from source rows, warms assignment compilation, and includes
centroid mirror preparation, query transfers, winner communication, and FP32
distance recomputation. It does not train centroids. Centroid source loading
is excluded; query batches use the demo's memory-mapped query reader.

```bash
$TORCHRUN --standalone --nproc_per_node=4 bench_query_mse.py --output /tmp/query-mse.json
```

See [results/QUERY_MSE.md](results/QUERY_MSE.md).
