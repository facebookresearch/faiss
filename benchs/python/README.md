# faiss Python benchmark suite (pytest)

pytest-based counterpart of the C++ Google Benchmark suite in `../cpp`. It
benchmarks faiss through the public Python API using
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/).

The `bench_fw/` subdirectory contains the **bench_fw** end-to-end evaluation
framework for multi-dataset accuracy/speed sweeps — see
[bench_fw/README.md](bench_fw/README.md).

## Requirements

- a faiss build with Python bindings importable (`import faiss`) — build
  with `-DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=dd -DFAISS_ENABLE_PYTHON=ON`
  and put the bindings on `PYTHONPATH` (and `libfaiss.so` on
  `LD_LIBRARY_PATH` for shared-library builds); a non-Release faiss makes the
  numbers meaningless. `FAISS_OPT_LEVEL=dd` is the dynamic-dispatch build
  (all SIMD variants compiled, best chosen at runtime)
- `pip install pytest pytest-benchmark`

## Running

```bash
cd benchs/python

pytest                                  # full default sweep (long!)
pytest bench_index_flat.py              # one file
pytest -k "flat_search" --collect-only  # see what would run
pytest --help                           # see the "benchmarks" option group
```

Every benchmark parameter can be overridden with a comma-separated list,
mirroring the C++ suite's gflags:

```bash
pytest bench_index_flat.py --d=128 --nb=10000 --nq=1,10 --k=10
pytest bench_index_ivf.py --nlist=1024 --nprobe=1,8,32
pytest bench_index_graph.py --M=32 --efSearch=64 --threads=8
```

JSON output for regression tracking (same spirit as the C++ suite runs):

```bash
pytest bench_index_ivf.py --benchmark-json=results.json
```

Real-data benchmarks: every file whose C++ counterpart registers `sift1m/...`
benchmarks has matching `*_sift1m` benchmarks that load a SIFT1M-layout
dataset (`sift_learn.fvecs`, `sift_base.fvecs`, `sift_query.fvecs`,
`sift_groundtruth.ivecs`) from `--data_dir` (default `sift1M`, also spelled
`--data-dir`) and report recall via `extra_info`. They skip when the
directory or its files are missing; the synthetic sweeps are unaffected.

```bash
pytest bench_index_flat.py -k sift1m --data_dir=/path/to/sift1M
```

## Layout and C++ mapping

Benchmarks are named `bench_*` in `bench_*.py` files (see `pytest.ini`);
parameter sweeps are declared with the `@params` decorator in
`bench_utils.py` and every declared parameter is overridable from the CLI.

| Python file              | Covers (cpp/ counterpart)                            |
| ------------------------ | ---------------------------------------------------- |
| bench_kern_distances.py  | bench_kern_distances.cpp, bench_kern_extra_distances.cpp (via faiss.knn / pairwise_distances) |
| bench_kern_fwht.py       | bench_kern_fwht.cpp                                  |
| bench_kern_partitioning.py | bench_kern_partitioning.cpp                        |
| bench_kern_result_handlers.py | bench_kern_result_handlers.cpp                  |
| bench_codec_quantizers.py | bench_codec_pq.cpp, bench_codec_sq.cpp, bench_codec_rq.cpp |
| bench_codec_kmeans.py    | bench_codec_kmeans.cpp                               |
| bench_index_flat.py      | bench_index_flat.cpp                                 |
| bench_index_flat_quantized.py | bench_index_flat_quantized.cpp, bench_index_pq_polysemous.cpp |
| bench_index_ivf.py       | bench_index_ivf.cpp, bench_index_ivf_parallel_mode.cpp, bench_index_ivfpq_add.cpp |
| bench_index_ivf_fastscan.py | bench_index_ivf_fastscan.cpp                      |
| bench_index_ivf_big_batch.py | bench_index_ivf_big_batch.cpp                     |
| bench_index_ivf_factory.py | bench_index_ivf_factory.cpp                        |
| bench_index_graph.py     | bench_index_graph.cpp                                |
| bench_index_binary.py    | bench_index_binary.cpp, bench_kern_hamming.cpp (via binary indexes) |
| bench_index_composite.py | bench_index_composite.cpp, bench_index_rcq.cpp       |
| bench_index_io.py        | bench_index_io.cpp                                   |
| bench_index_rabitq.py    | bench_index_rabitq.cpp                               |
| bench_index_panorama.py  | bench_index_panorama.cpp                             |

### C++-only benchmarks (no public Python entry point)

These live only in the C++ suite (`cpp/`); the kernels have no public Python
API.

| C++ file                     | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| bench_kern_heap.cpp          | Heap replace-top and bulk addn operations            |
| bench_kern_sorting.cpp       | Sorting kernels                                      |
| bench_kern_fp16.cpp          | fp16 conversion kernels                              |
| bench_kern_rabitq.cpp        | Bitwise dot-products, popcount, rearrange_bit_planes, qb sweep |
| bench_kern_visited_table.cpp | Visited-table set/reset/check operations             |
| bench_codec_pq_adc.cpp      | PQ code-distance kernels                             |
| bench_codec_pq_fastscan.cpp  | PQ fast-scan kernels                                 |
| bench_codec_sa_decode.cpp    | SA decode kernels                                    |
| bench_index_ivf_selector.cpp | IVF selector-filtered search                         |

## Conventions

- default OpenMP threads = 1 (override with `--threads=N`), matching the C++
  suite
- search benchmarks share cached trained indexes (`bench_utils.built_index`)
  so sweeps over nq/k/nprobe don't retrain; build/add benchmarks always
  construct fresh indexes via `benchmark.pedantic(..., setup=...)`
- features missing from the installed faiss build are skipped, not failed
  (`bench_utils.require_attr`)
