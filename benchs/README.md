# FAISS Benchmarks

CPU kernel and index-level benchmarks:

- `cpp/` — C++ suite using
  [Google Benchmark](https://github.com/google/benchmark)
- `python/` — pytest-benchmark suite exercising the public Python API
  (see [python/README.md](python/README.md))
- `gpu/` — C++ Google Benchmark suite for the GPU / cuVS index API, built only
  when `FAISS_ENABLE_GPU=ON` (see [gpu/README.md](gpu/README.md))
- `python/bench_fw/` — end-to-end benchmarking framework for multi-dataset
  accuracy/speed sweeps and automatic index selection
  (see [python/bench_fw/README.md](python/bench_fw/README.md))

## Prerequisites

- CMake ≥ 3.24
- C++17 compiler with OpenMP support
- [gflags](https://github.com/gflags/gflags)
- BLAS library (MKL or OpenBLAS)
- for the Python suite: an importable faiss build and `pytest-benchmark`

Google Benchmark is fetched automatically via CMake's `FetchContent`.

## Building

Run all commands from the **faiss repository root** (not from `benchs/`).
The suite is wired into the top-level build via `add_subdirectory(benchs)`
guarded by `FAISS_BUILD_BENCHMARKS`; `benchs/CMakeLists.txt` is not
a standalone project, so configuring inside `benchs/` fails.

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_BUILD_BENCHMARKS=ON \
    -DFAISS_OPT_LEVEL=dd \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF

cmake --build build --config Release --target benchmarks -j$(nproc)
```

Always build in `Release` (`-DCMAKE_BUILD_TYPE=Release`) — the default
unoptimized build produces meaningless benchmark numbers. `FAISS_OPT_LEVEL=dd`
selects the dynamic-dispatch build, which compiles all SIMD variants (AVX2,
AVX512, AVX512-SPR) and picks the best at runtime; pin a single level instead
(e.g. `-DFAISS_OPT_LEVEL=avx2` or `avx512`) to benchmark one code path. This
builds all benchmark executables into `build/benchs/cpp/`.

## Running

### Run the full C++ suite

Each executable is standalone; run them all and collect JSON results with a
loop:

```bash
mkdir -p results
for bench in build/benchs/cpp/bench_*; do
    "$bench" --benchmark_format=json \
        --benchmark_out="results/$(basename "$bench").json" \
        --benchmark_out_format=json
done
```

On multi-socket systems, prefix the invocation with
`numactl --cpunodebind=0 --membind=0` for stable numbers.

### Running on a real dataset (SIFT1M and others)

Several index benchmarks (`bench_index_flat`, `bench_index_ivf`,
`bench_index_ivf_fastscan`, `bench_index_flat_quantized`, `bench_index_graph`,
`bench_index_pq_polysemous`) run on synthetic data by default and additionally emit
real-data cases when the dataset files are found. Point them at the dataset
directory with `--data_dir` (default: `sift1M`):

```bash
./build/benchs/cpp/bench_index_ivf --data_dir=/path/to/sift1M
```

By default the directory must contain the standard SIFT1M files
`sift_learn.fvecs`, `sift_base.fvecs`, `sift_query.fvecs`, and
`sift_groundtruth.ivecs`, and the cases are labeled `sift1m/*`. If the files
are absent the real-data cases are silently skipped and only the synthetic
cases run.

Nothing about the filenames is hardcoded: each is overridable, so any dataset
with the same framing (train/base/query vectors as `.fvecs` float32 or
`.bvecs` uint8, ground truth as `.ivecs`) works without renaming files. The
reader is picked per file from the extension. The real-data case labels are
derived from the base filename (e.g. `cohere1M_base.fvecs` → `cohere1M/*`):

```bash
./build/benchs/cpp/bench_index_ivf \
    --data_dir=/path/to/cohere1M \
    --train_file=cohere1M_learn.fvecs \
    --base_file=cohere1M_base.fvecs \
    --query_file=cohere1M_query.fvecs \
    --gt_file=cohere1M_groundtruth.ivecs
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `sift1M` | dataset directory |
| `--train_file` | `sift_learn.fvecs` | train/learn vectors (`.fvecs`/`.bvecs`) |
| `--base_file` | `sift_base.fvecs` | base/database vectors (`.fvecs`/`.bvecs`) |
| `--query_file` | `sift_query.fvecs` | query vectors (`.fvecs`/`.bvecs`) |
| `--gt_file` | `sift_groundtruth.ivecs` | ground-truth neighbors (`.ivecs`) |

Note: these six benches search the loaded data under `METRIC_L2` only, so the
ground truth must be computed under L2. For datasets that are conventionally
searched with inner product / cosine (e.g. Cohere embeddings), use
`bench_index_ivf_factory --metric=IP`, which accepts the same filename flags (or its
convention-based `--db=<name>` shortcut).

### Run a single benchmark

```bash
./build/benchs/cpp/bench_kern_distances --benchmark_format=console
```

### Run a specific test case within a benchmark

Use `--benchmark_filter` with a regex to select specific cases:

```bash
# Run only L2 distance benchmarks
./build/benchs/cpp/bench_kern_distances --benchmark_filter="L2"

# Run a specific dimension
./build/benchs/cpp/bench_kern_distances --benchmark_filter="L2.*d:128"

# Run multiple patterns
./build/benchs/cpp/bench_index_ivf --benchmark_filter="(Search|Add)"
```

### Useful Google Benchmark flags

| Flag | Description |
|------|-------------|
| `--benchmark_filter=<regex>` | Only run benchmarks matching the regex |
| `--benchmark_format=console\|json\|csv` | Output format (default: console) |
| `--benchmark_out=<file>` | Write results to a file |
| `--benchmark_out_format=json` | Format for `--benchmark_out` |
| `--benchmark_repetitions=N` | Repeat each benchmark N times |
| `--benchmark_min_time=Ns` | Minimum time per benchmark (e.g. `2s`) |
| `--benchmark_list_tests=true` | List available test names without running |
| `--benchmark_counters_tabular=true` | Align custom counters in columns |

### Examples

```bash
# List all tests in bench_index_flat without running them
./build/benchs/cpp/bench_index_flat --benchmark_list_tests=true

# Run heap benchmarks with 5 repetitions and report statistics
./build/benchs/cpp/bench_kern_heap --benchmark_repetitions=5

# Export PQ fastscan results as JSON
./build/benchs/cpp/bench_codec_pq_fastscan \
    --benchmark_format=json \
    --benchmark_out=pq_fastscan.json

# Run with a longer minimum time for more stable results
./build/benchs/cpp/bench_kern_distances --benchmark_min_time=3s --benchmark_filter="L2"
```

### Python suite (pytest)

`benchs/python/` holds the pytest-benchmark counterpart of the C++ suite,
exercising faiss through the public Python API. It needs an importable faiss
build and `pip install pytest-benchmark`.

```bash
cd benchs/python
pytest bench_index_ivf.py --nlist=1024 --nprobe=1,8 --benchmark-json=out.json
pytest --help   # see the "benchmarks" parameter options
```

See [python/README.md](python/README.md) for the full C++ ↔ Python mapping.

## Benchmark catalog

### Kernel and codec microbenchmarks

Low-level computational kernels:

| Executable | Covers |
|------------|--------|
| `bench_kern_distances` | L2, inner product distance computations |
| `bench_kern_extra_distances` | Additional distance metrics |
| `bench_kern_hamming` | Hamming distance and hamming knn |
| `bench_kern_heap` | Heap operations |
| `bench_kern_partitioning` | Partitioning algorithms |
| `bench_kern_sorting` | Sorting routines |
| `bench_kern_fp16` | FP16 conversion |
| `bench_kern_fwht` | Fast Walsh-Hadamard transform |
| `bench_kern_rabitq` | RaBitQ quantization |
| `bench_kern_visited_table` | `VisitedTable` set/get/advance (vector vs hashset strategies) |
| `bench_kern_result_handlers` | Top-k result-handler cost in IVF search over a k/nprobe/index-type sweep |
| `bench_codec_pq` | PQ training and encoding |
| `bench_codec_pq_adc` | PQ code distance tables |
| `bench_codec_pq_fastscan` | PQ fast-scan kernels |
| `bench_codec_sq` | SQ encoding/decoding |
| `bench_codec_rq` | Residual/additive quantizer encode/decode (RQ, LSQ, PRQ, PLSQ) |
| `bench_codec_kmeans` | K-means clustering (assignment + SuperKMeans/Clustering training) |
| `bench_codec_sa_decode` | `cppcontrib` SADecodeKernels (PQ/IVFPQ/Residual+PQ) vs `Index::sa_decode` |

### Index-level benchmarks

End-to-end index operations (build, add, search):

| Executable | Covers |
|------------|--------|
| `bench_index_flat` | Flat (brute-force) indexes |
| `bench_index_flat_quantized` | Flat quantized indexes (PQ incl. nbits sweep, PQFastScan, SQ) |
| `bench_index_ivf` | IVF indexes (incl. SQ rangestat sweep) |
| `bench_index_ivf_fastscan` | IVF with fast-scan (PQ and LSQ/RQ additive-quantizer variants) |
| `bench_index_ivf_big_batch` | Big-batch IVF search |
| `bench_index_ivf_parallel_mode` | IVF parallel_mode comparison |
| `bench_index_ivf_factory` | Generic index_factory train/add/search (L2/IP) with recall@1/10/100 + ParameterSpace autotune |
| `bench_index_ivf_selector` | `IDSelector` (IDSelectorAll/null) search overhead vs plain search |
| `bench_index_ivfpq_add` | IVFPQ add throughput |
| `bench_index_rabitq` | RaBitQ indexes (flat and IVF) |
| `bench_index_graph` | Graph-based indexes (HNSW, NSG, NNDescent, HNSWSQ with prune_headroom) |
| `bench_index_binary` | Binary indexes |
| `bench_index_composite` | Composite/compound indexes |
| `bench_index_io` | Index serialization I/O |
| `bench_index_panorama` | Panorama progressive-dimension indexes |
| `bench_index_pq_polysemous` | Polysemous PQ training, ST_PQ baseline and `polysemous_ht` sweep search |
| `bench_index_rcq` | ResidualCoarseQuantizer search |

## Output format

Both suites emit machine-readable JSON for regression tracking:

- C++: `--benchmark_out=<file> --benchmark_out_format=json` produces the
  standard [Google Benchmark JSON](https://github.com/google/benchmark/blob/main/docs/user_guide.md#output-formats)
  (context block plus one entry per benchmark with `real_time`, `cpu_time`,
  `iterations` and the custom counters such as `d`, `nb`, `items_per_second`).
- Python: `pytest --benchmark-json=<file>` produces the
  [pytest-benchmark JSON](https://pytest-benchmark.readthedocs.io/en/latest/usage.html)
  (machine info plus per-benchmark stats: min/max/mean/stddev/rounds).

## Tips for reliable results

- Set the CPU governor to `performance`:
  ```bash
  sudo cpupower frequency-set -g performance
  ```
- Disable turbo boost:
  ```bash
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
  ```
- On multi-socket systems, install `numactl` for automatic NUMA pinning.
- Close other workloads during benchmark runs.
