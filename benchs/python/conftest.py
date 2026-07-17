# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""pytest configuration for the faiss Python benchmark suite.

Run `pytest --help` and look for the "benchmarks" option group: every benchmark
parameter (d, nb, nq, k, nlist, nprobe, M, ...) can be overridden with a
comma-separated list, e.g.:

    pytest bench_index_flat.py --d=128 --nq=1,10 --benchmark-sort=name

Without overrides, each benchmark runs its built-in default sweep (declared
per benchmark with the @params decorator, see bench_utils.py).
"""

import os

import pytest

import bench_utils

# bench_fw/ is an end-to-end evaluation framework, not a pytest suite.
collect_ignore_glob = [os.path.join(os.path.dirname(__file__), "bench_fw", "*")]

# Every sweep parameter used across the suite: name -> help text. A benchmark
# only picks up the parameters it declares via @params; the CLI override
# applies globally to all benchmarks declaring that parameter.
PARAM_OPTIONS = {
    "d": "vector dimensions",
    "n": "batch/database sizes (kernel benchmarks)",
    "nb": "database sizes",
    "nq": "query batch sizes",
    "k": "search result counts",
    "nlist": "IVF list counts",
    "nprobe": "IVF probe counts",
    "M": "PQ subquantizer counts / graph degrees",
    "nbits": "bits per quantizer code",
    "efConstruction": "HNSW construction depth",
    "efSearch": "HNSW search depth",
    "pmode": "IVF parallel modes",
    "metric": "metric names (L2, IP)",
    "sq_type": "scalar quantizer type names (QT_8bit, QT_fp16, ...)",
    "R": "NSG graph degrees",
    "search_L": "NSG search depths",
    "code_size": "binary/PQ code sizes in bytes",
    "bounded": "HNSW bounded-queue settings (0/1)",
    "precomp": "IVFPQ precomputed-table settings (0/1)",
    "implem": "FastScan implementation ids",
    "beam": "RCQ/RQ beam sizes",
    "qb": "RaBitQ query quantization bits",
    "k_factor": "refine-stage k multipliers",
    "headroom": "HNSW prune-headroom fractions (floats)",
    "rangestat": "scalar quantizer range statistics (RS_* names)",
    "rangestat_arg": "scalar quantizer rangestat argument values (floats)",
    "rescale": "AQ fastscan rescale_norm settings (0/1)",
    "method": "big-batch search method names",
    "factory": "index_factory keys (e.g. 'IVF256,Flat')",
    "id_type": "partition id array types (int64, int32)",
    "pca": "Panorama PCA pre-transform toggle (0/1)",
    "polysemous_ht": "polysemous Hamming thresholds",
    "dsub": "PQ sub-vector dimension (d = M*dsub) for table benchmarks",
    "ils_iters": "LSQ/PLSQ encode_ils_iters counts",
    "retain_locks": "HNSW retain-locks batched-add settings (0/1)",
    "prefix": "FWHT recall pre-transform prefix names (HR, RR, none)",
}


def pytest_addoption(parser):
    group = parser.getgroup(
            "benchmarks", "faiss benchmark parameters (comma-separated lists)")
    for name, help_text in PARAM_OPTIONS.items():
        group.addoption(
                f"--{name}",
                default=None,
                help=f"override {help_text}, e.g. --{name}=1,10",
        )
    group.addoption(
            "--threads",
            type=int,
            default=1,
            help="OpenMP thread count for faiss (default: 1)",
    )
    group.addoption(
            "--data-dir",
            "--data_dir",
            default="sift1M",
            help="path to a SIFT1M-layout dataset directory (optional); "
                 "dataset benchmarks skip when it is missing",
    )


def _parse_list(name, raw):
    # index_factory keys contain commas ("IVF256,Flat"), so --factory is
    # split on ';' rather than ',' (e.g. --factory='IVF256,Flat;IVF1024,PQ8').
    sep = ";" if name == "factory" else ","
    tokens = [tok for tok in raw.split(sep) if tok]
    if not tokens:
        raise pytest.UsageError(
                f"--{name}={raw!r}: expected a {sep!r}-separated list")
    try:
        return [int(tok) for tok in tokens]
    except ValueError:
        pass
    try:
        return [float(tok) for tok in tokens]  # e.g. --k_factor=1.5
    except ValueError:
        return tokens  # string-valued parameter (e.g. --metric=L2,IP)


def pytest_generate_tests(metafunc):
    sweeps = getattr(metafunc.function, bench_utils.SWEEPS_ATTR, None)
    if not sweeps:
        return
    for name, defaults in sweeps.items():
        if name not in metafunc.fixturenames:
            continue
        override = metafunc.config.getoption(name, default=None)
        values = _parse_list(name, override) if override else defaults
        metafunc.parametrize(
                name, values, ids=[f"{name}:{v}" for v in values])


@pytest.fixture(autouse=True, scope="session")
def _omp_threads(request):
    import faiss

    faiss.omp_set_num_threads(request.config.getoption("threads"))


@pytest.fixture(scope="session")
def data_dir(request):
    return request.config.getoption("data_dir")
