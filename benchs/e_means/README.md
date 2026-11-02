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
centroids or training statistics. Evaluation uses exhaustive FP32 assignment
even when training uses IVF. `--query-batch-size` (default 1024) bounds query
GPU memory; the centroid table remains sharded. Exhaustive evaluation adds
work proportional to the query count times the centroid count.

Per-epoch values are stored in `iteration_stats[*].test_mse`; the final
`test_mse` and `test_query_count` are printed and saved with the centroid
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
requested random row order. A dedicated CUDA stream copies the next batch
while the current batch trains. Prefetch is always enabled and automatically
chooses the number of batches per window within `--ssd-ram-gib` (default
16 GiB per rank, for both windows). `--ssd-read-threads`,
`--ssd-extract-threads` and `--ssd-chunk-mib` control the scan pipeline.

`--ssd-prefetch-chunk-groups` defaults to 4: it shuffles scan chunks into
four groups, then
shuffles each group's rows and the full windows across groups. Every row is
used once per epoch. Batches within a window draw from that group's subset of
chunks; this changes batch composition compared with a uniform permutation.
Use `--ssd-prefetch-chunk-groups 1` for the uniform permutation.

The default `--final-closing-mean` makes a fresh sequential pass, assigns
against one frozen final centroid snapshot, and replaces occupied centroids
by their assigned means. It scans each source row once without random-window
rescans. `--no-final-closing-mean` uses the existing closing means from the
last moving training pass. Reports include the moving-pass MSE, frozen closing
assignment MSE, SSD bytes read, consumer wait time and peak GPU allocation.
Saved centroid shards include the centering mean and scale; reconstruct
original-space centroids as `centroids * scale + mean`.

`--self-test` verifies direct reads across file boundaries, duplicate random
requests, permutation coverage, partial batches, asynchronous copies, the
single sequential scan and frozen closing means.

For a smaller SSD run with `N=10M` and `k=2**17`:

```bash
buck2 run -c hpc_comms.use_nccl=2.29 \
    fbcode//faiss/benchs/e_means:demo_training_from_ssd -- \
    --launch 2 --n 10000000 --k 131072 --epochs 2 \
    --nlist 256 --nprobe 4 \
    --ssd-ram-gib 4 --ssd-read-threads 4 \
    --final-closing-mean --output /tmp/e_means_ssd_10m_131072
```
