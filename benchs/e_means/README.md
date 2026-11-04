# Sharded e-means on DINO

These demos show sharded e-means training on the public DINO10B dataset.

Most of the training code is in the e_means contrib directory. The code here
mainly demonstrates it at different scales, with steps of increasing complexity.
All the code is made to run on a single multi-GPU machine.

Every worker holds one contiguous centroid shard and an equal row contribution
per global batch. `k`, `n`, and the resolved global batch size must divide by
the number of workers. Initialize NCCL before constructing the model; the
communication group belongs to the caller. Results are local FP32 centroid
shards, located by `centroid_begin`.

## Download DINO10B

The public corpus is at
https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/.
All training demos need the first base chunk and the held-out test queries:

- `chunked_base_10B/chunk_0000.bvecs`: 200,000,000 vectors,
  205,600,000,000 bytes (about 191.5 GiB).
- `queries_clean.bvecs`: 100,000 held-out queries,
  102,800,000 bytes (about 98 MiB).

Both files contain 1,024 uint8 coordinates per vector, preceded by a four-byte
little-endian dimension header. Use an SSD-backed location with enough free
space. The commands below use `/tmp/DINO10B`; replace that root throughout
and pass `--data PATH` to the demos if you use another location.

Download the public instructions and both data files:

```bash
mkdir -p /tmp/DINO10B/chunked_base_10B
wget -c -O /tmp/DINO10B/README.md \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/README.md
wget -c -O /tmp/DINO10B/chunked_base_10B/chunk_0000.bvecs.part \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/chunked_base_10B/chunk_0000.bvecs
wget -c -O /tmp/DINO10B/queries_clean.bvecs.part \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/queries_clean.bvecs
```

On Meta devservers, use `curl` with the forward proxy from your terminal:

```bash
mkdir -p /tmp/DINO10B/chunked_base_10B
curl $(fwdproxy-config curl) -fL \
    -o /tmp/DINO10B/README.md \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/README.md
curl $(fwdproxy-config curl) -fL \
    -o /tmp/DINO10B/chunked_base_10B/chunk_0000.bvecs.part \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/chunked_base_10B/chunk_0000.bvecs
curl $(fwdproxy-config curl) -fL \
    -o /tmp/DINO10B/queries_clean.bvecs.part \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/queries_clean.bvecs
```

`wget -c` resumes interrupted downloads. Validate the completed files before
making them visible to the benchmarks:

```bash
python3 - <<'PYTHON'
from pathlib import Path
root = Path("/tmp/DINO10B")
for name, size in (("chunked_base_10B/chunk_0000.bvecs", 205_600_000_000),
                   ("queries_clean.bvecs", 102_800_000)):
    part = root / (name + ".part")
    assert part.stat().st_size == size
    with part.open("rb") as stream:
        assert int.from_bytes(stream.read(4), "little") == 1024
    part.rename(root / name)
PYTHON
```

For the full 10B corpus, follow the public README and download the remaining
base chunks into the same directory; the complete corpus takes about
10.28 TB. These demos do not need `train_queries_99M.bvecs` or nearest-neighbor
ground truth.

### Held-out MSE

Both training demos evaluate every test query after each epoch and again
after finalization, including any final closing-mean update. `test_mse` is
the average squared L2 distance to the nearest centroid across all shards,
in original DINO units: sum over the 1,024 coordinates, then divide by
100,000 queries. Queries use the training mean and scale and never update
centroids or training statistics. Evaluation defaults to exhaustive fused
Triton BF16 assignment, even when training uses IVF. It rounds queries and
centroids to BF16 for ranking, selects the global winner with lowest-ID ties,
then reports its distance using the original FP32 operands and FP64 SSE
accumulation. This can select a different centroid from FP32 assignment.
`--query-precision fp32` retains the previous FP32 assignment path.
`--query-batch-size` (default 1024) bounds query GPU memory; the centroid table
remains sharded. BF16 mirrors are prepared once per evaluation and released
afterward. Exhaustive evaluation adds work proportional to the query count
times the centroid count.

Per-epoch values are stored in `iteration_stats[*].test_mse`; the final
`test_mse`, `test_query_count`, and `query_precision` are saved with the centroid
shards. Training moving-pass MSE and closing assignment MSE remain separate
training-set measurements.

## Resident data and exact assignment

```bash
buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_sharded_centroids -- \
    --launch 2 --n 10000000 --k 10000 --epochs 2
```

From the repository root, use `python benchs/e_means/demo_sharded_centroids.py --launch 2` with Faiss,
Torch and Triton installed. The demo reads `/tmp/DINO10B/chunked_base_10B/*.bvecs`,
keeps the rank's rows in GPU RAM as FP32, subtracts their global mean and trains.
Use `--data` to change the corpus root and `--output PREFIX` to save each rank's
centroids, mean, recipe and iteration statistics. `--assignment-backend native`
selects the existing tiled Torch exact search.

`--self-test` compares with the existing `Emeans` lifecycle through multiple
EMA updates, global revival, a final partial batch and closing means. The
ordinary finalization uses means from the assignments made during the last
moving training pass.

## Resident data with IVF assignment

```bash
buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_sharded_centroids -- \
    --launch 2 --ivf --n 10000000 --k 100000 --nlist 256 --nprobe 4
```

`EmeansShardedCentroidsIVF` uses the Torch `IndexIVFFlat` padded BMM search
ported from the reference `anns_indexing.py`. It trains a shared coarse
quantizer on the GPUs once, rebuilds each owner's candidate lists at every
batch, and routes using global list occupancy. Fine and coarse assignment use
FP16 operands; EMA, sums and reported SSE use FP32 source values. Revival
separates split centroids in the assignment precision. `--self-test --ivf`
compares exhaustive probes with an independent FP16 oracle and tests ties,
empty lists and a hot list under a small workspace budget.

## Streaming from SSD

```bash
buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_training_from_ssd -- \
    --launch 2 --n 200000000 --k 1000000 --epochs 2 \
    --nlist 16384 --nprobe 4 \
    --ssd-ram-gib 48 --ssd-read-threads 4 \
    --final-closing-mean \
    --output /tmp/e_means_ssd_200m_1m
```

From the repository root, run `python benchs/e_means/demo_training_from_ssd.py` with the same arguments.
Use Linux on x86_64 or aarch64 with io_uring and membarrier allowed by the
kernel and seccomp policy.
`ssd_streaming.py` lives alongside the demos. Each rank reads its contiguous
range directly from the downloaded SSD files. Initial centroids use queued
random reads through `ctypes_liburing.py`, with no flat copy on disk. Reads
are aligned to 4 KiB, merge rows sharing a starting block and span additional
blocks when a bvecs payload crosses a boundary. `--ssd-io-depth` defaults to
256 concurrent reads with reusable aligned buffers. `--initialization-only`
verifies a sample against the bvecs reader, reports initialization time and
bytes read, and exits before training. Up to 10,000 sampled rows determine the global mean and scalar standard deviation, as in the reference.

Training uses two pinned host windows per rank. A background scan reads only
requested 64 MiB chunks, in file order, and extraction workers restore the
requested random row order. Distinct-row windows use the reference's linear
mask/inverse planner; duplicate requests fall back to a stable sort. Planning
allocates a one-byte mask and then a four-byte inverse entry per shard row,
plus request-sized arrays (a 1 GB inverse map at 250 million rows per rank).
These planning arrays are additional to the host-window RAM budget.
Aligned read buffers are reused within each scan;
extraction reads payload views directly from them, skipping bvecs headers
without an intermediate full-chunk copy. A buffer is reused only after its
extraction completes. A dedicated CUDA stream copies the next batch
while the current batch trains. Prefetch is always enabled and automatically
chooses the number of batches per window within `--ssd-ram-gib` (default
16 GiB per rank, for both windows). `--ssd-read-threads`,
`--ssd-extract-threads` and `--ssd-chunk-mib` control the scan pipeline.

Both sharded training demos use all-gather for batch communication.
Each accumulation call gathers the full global batch in rank order and
reduces winner arrays once for that batch. IVF routes each rank's local rows
using global list occupancy, gathers their coarse-list IDs in matching order,
and searches the gathered batch without repeating coarse routing on each
centroid owner. IDs add 32 MiB per GPU at a 2Mi batch and nprobe=4.
IVF uses its existing search workspace budget; native assignment still
bounds compute chunks. Owned sums are accumulated in the existing source/chunk
order, and captured labels describe only this rank's supplied rows.
The gathered FP32 buffer uses `global_batch * d * 4` additional bytes per
GPU (8 GiB at 2,097,152 rows and d=1024); startup logs show this allocation.
Saved-assignment closing continues to route rows directly to their owners.

`--batch-size` overrides the global training batch size across all workers.
For example, `--launch 4 --batch-size 262144` uses 65,536 rows per worker
per batch. The value must be positive, no larger than `--n`, and divisible
by the worker count. The fitted EMA step is recomputed for the overridden
batch size. Omitting the flag uses the default quality recipe.

Progress logs show `steps=current/total` within each epoch and `batch_mse`
at `--verbose-every` intervals, including the first and final batches.
Every batch's MSE is also saved in each epoch's `iteration_stats["batch_mse"]`
list in the reports and output files. These are squared L2 distances in the
original data units, summed across centroid owners and divided by the global
batch row count, against the centroids before that batch's update.

`--ssd-prefetch-chunk-groups` defaults to 4: it shuffles scan chunks into
four groups, then
shuffles each group's rows and the full windows across groups. Every row is
used once per epoch. Batches within a window draw from that group's subset of
chunks; this changes batch composition compared with a uniform permutation.
Use `--ssd-prefetch-chunk-groups 1` for the uniform permutation.

The default `--final-closing-mean` saves final-epoch labels as CPU int32 codes
in original source-row order, then makes a sequential no-search pass. Each
row is sent only to its saved centroid owner; closing reuses the final-pass
sum table and checks that counts match the saved training assignments.
Empty clusters retain their final EMA positions. The codes use four bytes
per source row, additional to the prefetch budget (0.931 GiB per rank for
1B rows across four ranks). This matches the reference's fixed-code closing
algorithm and changes results compared with the previous fresh-assignment
scan against frozen final centroids. `--no-final-closing-mean` finalizes the
already-accumulated last-pass sums without another SSD pass.
Reports include moving-pass MSE, fixed-code closing MSE and closing time,
SSD bytes read, consumer wait time and peak GPU allocation.
Saved centroid shards include the centering mean and scale; reconstruct
original-space centroids as `centroids * scale + mean`.

Sequential closing scans use one batch per pinned host window, with two
windows per worker, regardless of the training prefetch RAM budget. This lets
GPU processing start after reading one batch and overlaps the next batch's
read with computation.
Routing exchanges use chunks of at most 65,536 rows per rank to bound
temporary GPU memory independently of the training batch size.

`--self-test` verifies direct reads across file boundaries, duplicate random
requests, permutation coverage, partial batches, asynchronous copies, the
single sequential scan and saved-assignment closing means without search.

For a smaller SSD run with `N=10M` and `k=2**17`:

```bash
buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_training_from_ssd -- \
    --launch 2 --n 10000000 --k 131072 --epochs 2 \
    --nlist 256 --nprobe 4 \
    --ssd-ram-gib 4 --ssd-read-threads 4 \
    --final-closing-mean --output /tmp/e_means_ssd_10m_131072
```
