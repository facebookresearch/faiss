# bench_fw — end-to-end benchmarking framework

This directory contains the **bench_fw** evaluation framework and its driver
scripts. Unlike the per-kernel/per-index micro-benchmarks in the parent
`benchs/python/` directory (and the C++ suite in `benchs/cpp/`), bench_fw
orchestrates full accuracy/speed sweeps across datasets and index
configurations, producing JSON result files.

## Driver scripts

| Script | What it does |
|--------|-------------|
| `bench_fw_codecs.py` | Multi-codec accuracy sweep (SQ, ITQ+LSH, OPQ+PQ, OPQ+RQ, LSQ, PRQ, PLSQ) on SIFT1M / BigANN / Deep / Contriever |
| `bench_fw_ivf.py` | IVF accuracy/speed operating-point tables on SIFT1M and BigANN at 1M–50M scale |
| `bench_fw_optimize.py` | Automatic index selection (Optimizer) to find the best config meeting a minimum accuracy target |
| `bench_fw_range.py` | Range-search accuracy evaluation on SSNPP data with weighted recall |

## Usage

Run the drivers directly from the `benchs/python/` directory:

```bash
cd benchs/python
python -m bench_fw.bench_fw_ivf sift1M /path/to/results
python -m bench_fw.bench_fw_codecs sift1M /path/to/results
python -m bench_fw.bench_fw_optimize bigann /path/to/results
python -m bench_fw.bench_fw_range ssnpp /path/to/results
```

Each driver requires external datasets and writes results to the specified
directory.

## Module layout

| File | Role |
|------|------|
| `benchmark.py` | Core `Benchmark` class — trains, builds, searches, and evaluates indexes |
| `benchmark_io.py` | `BenchmarkIO` — dataset loading, result I/O, job launching |
| `descriptors.py` | `DatasetDescriptor`, `IndexDescriptorClassic` — configuration objects |
| `index.py` | Index wrapper with train/build/search/reconstruct methods |
| `optimize.py` | `Optimizer` — automatic index selection for a target accuracy |
| `utils.py` | Evaluation helpers (recall, range-search metrics) |
