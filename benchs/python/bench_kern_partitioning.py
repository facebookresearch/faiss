# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python benchmark for CMax_uint16_partition_fuzzy on an AlignedTableUint16,
# exact (q_min == q_max == n/2) and fuzzy (q_min = n/2, q_max = n/2 + n/4,
# i.e. the 100..150 / 1000..1500 / 10000..15000 ranges). The partition
# mutates both the value table and the ids array, so both are recopied in an
# untimed setup before every timed call (pedantic with iterations=1).

import functools

import faiss
import numpy as np

from bench_utils import params, require_attr

ROUNDS = 500  # averages over many runs; the op is µs-scale


@functools.lru_cache(maxsize=8)
def partition_data(n, id_type="int64", maxval=65536, seed=123):
    """Values/ids for the benchmark.

    The full size/split sweep runs for both int64 and
    int32 id arrays; id_type selects which.
    """
    rs = np.random.RandomState(seed)
    vals = rs.randint(maxval, size=n).astype("uint16")
    ids = (rs.permutation(n) + 12345).astype(id_type)
    return vals, ids


def run_partition(benchmark, n, q_min, q_max, id_type="int64"):
    require_attr(faiss, "CMax_uint16_partition_fuzzy")
    require_attr(faiss, "AlignedTableUint16")
    require_attr(faiss, "copy_array_to_AlignedTable")
    vals, ids = partition_data(n, id_type)
    sp = faiss.swig_ptr
    tab = faiss.AlignedTableUint16()
    q_out = np.zeros(1, dtype="uint64")

    def setup():
        # partition_fuzzy mutates the table and the ids in place — refill
        # both before every timed call.
        faiss.copy_array_to_AlignedTable(vals, tab)
        return (ids.copy(),), {}

    def run(ids_copy):
        faiss.CMax_uint16_partition_fuzzy(
                tab.get(), sp(ids_copy), n, q_min, q_max,
                None if q_min == q_max else sp(q_out))

    # iterations=1 so setup() runs before every timed call.
    benchmark.pedantic(run, setup=setup, rounds=ROUNDS, iterations=1)


@params(n=[200, 2000, 20000], id_type=["int64", "int32"])
def bench_partition_exact(benchmark, n, id_type):
    run_partition(benchmark, n, n // 2, n // 2, id_type)


@params(n=[200, 2000, 20000], id_type=["int64", "int32"])
def bench_partition_fuzzy(benchmark, n, id_type):
    run_partition(benchmark, n, n // 2, n // 2 + n // 4, id_type)
