# faiss GPU benchmark suite (C++ / Google Benchmark)

GPU counterpart of the CPU suite in [`../cpp`](../cpp). Each executable is a
standalone [Google Benchmark](https://github.com/google/benchmark) binary that
exercises the faiss **host-side GPU index API** (`GpuIndex*`, `StandardGpuResources`,
`index_cpu_to_gpu`) — it contains no device kernels of its own, so it builds
as plain C++ against a GPU-enabled faiss.

## Requirements

- a faiss build configured with `-DFAISS_ENABLE_GPU=ON` (CUDA or ROCm)
- for the cuVS comparison cases: `-DFAISS_ENABLE_CUVS=ON` (otherwise those
  cases register but skip at runtime with a clear message)
- CMake ≥ 3.24, a CUDA/HIP toolkit, gflags, OpenMP, BLAS
- Google Benchmark is fetched automatically via the parent `benchs/cpp`
  CMake integration

## Building

From the **faiss repository root**:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_BUILD_BENCHMARKS=ON \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_CUVS=ON \
    -DFAISS_OPT_LEVEL=dd \
    -DFAISS_ENABLE_PYTHON=OFF

cmake --build build --config Release --target benchmarks -j$(nproc)
```

Executables land in `build/benchs/gpu/`. The `benchmarks` meta-target builds
both the CPU (`benchs/cpp`) and GPU (`benchs/gpu`) suites. Always build in
`Release`.

## Running

Same Google Benchmark interface as the CPU suite:

```bash
mkdir -p results
for bench in build/benchs/gpu/bench_*; do
    "$bench" --benchmark_format=json \
        --benchmark_out="results/$(basename "$bench").json" \
        --benchmark_out_format=json
done
```

By default each bench runs on **synthetic** float data (128-dim, 1M base, 10K
queries) with exact ground truth computed on the CPU, so recall is always
reported. Point `--data_dir` at a SIFT1M-style dataset to run on real data
(same `--data_dir` / `--train_file` / `--base_file` / `--query_file` /
`--gt_file` flags as the CPU suite):

```bash
./build/benchs/gpu/bench_gpu_index_flat_ivfpq --data_dir=/path/to/sift1M
```

Every benchmark also takes `--help` for its own flags and example invocations.

## Benchmarks

| Executable | Covers |
|------------|--------|
| `bench_gpu_index_flat_ivfpq` | single-GPU exact `GpuIndexFlatL2` (k sweep) + approximate `IVF4096,PQ64` cloned to GPU with fp16 lookup tables (nprobe sweep), recall@{1,10,100} |
| `bench_gpu_index_ivf_1bn` | large-scale IVF clone-and-add + nprobe search over single / sharded / replicated GPUs; `--float16`, `--noptables`, `--ngpu` |
| `bench_gpu_index_ivf_hybrid` | CPU vs whole-GPU vs hybrid (GPU coarse quantizer + CPU list scan via `search_preassigned`) IVF search, nprobe sweep |
| `bench_gpu_index_ivfflat_cuvs` | `GpuIndexIVFFlat` **train / add / search** — classical GPU vs cuVS, side by side, with recall |
| `bench_gpu_index_ivfpq_cuvs` | `GpuIndexIVFPQ` (nlist=1024, M=32, 8-bit) **train / add / search** — classical GPU vs cuVS, with recall |

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `sift1M` | dataset directory; synthetic data used if absent |
| `--nb` / `--nq` / `--nt` / `--d` | 1M / 10K / 100K / 128 | synthetic working-set shape |
| `--device` | `0` | GPU device ordinal |
| `--nprobe` | per-bench | comma-separated nprobe sweep |
| `--nlist` | `1024` | IVF list count (cuVS benches) |
| `--iterations` | `0` (auto) | fixed Google Benchmark iteration count |

### Notes

- The `*_cuvs` benches run each of train/add/search twice (classical vs cuVS)
  so the pair sits together in the output. Without `FAISS_ENABLE_CUVS`, the
  cuVS cases skip cleanly.
- `bench_gpu_index_ivf_1bn` and `bench_gpu_index_ivf_hybrid` measure
  clone-and-add, sharded/replicated multi-GPU search, and the split CPU/GPU
  coarse-quantize + scan pipeline. Set `--data_dir` at a large dataset to run
  them at scale.
