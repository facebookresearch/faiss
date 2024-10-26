# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import faiss  # @manual=//faiss/python:pyfaiss
import numpy as np
from faiss.contrib.datasets import (  # @manual=//faiss/contrib:faiss_contrib
    Dataset,
    SyntheticDataset,
)

US_IN_S = 1_000_000


@dataclass
class PerfCounters:
    wall_time_s: float = 0.0
    user_time_s: float = 0.0
    system_time_s: float = 0.0


@contextmanager
def timed_execution() -> Generator[PerfCounters, None, None]:
    pcounters = PerfCounters()
    wall_time_start = time.perf_counter()
    rusage_start = resource.getrusage(resource.RUSAGE_SELF)
    yield pcounters
    wall_time_end = time.perf_counter()
    rusage_end = resource.getrusage(resource.RUSAGE_SELF)
    pcounters.wall_time_s = wall_time_end - wall_time_start
    pcounters.user_time_s = rusage_end.ru_utime - rusage_start.ru_utime
    pcounters.system_time_s = rusage_end.ru_stime - rusage_start.ru_stime


def is_perf_counter(key: str) -> bool:
    return key.endswith("_time_us")


def accumulate_perf_counter(
    phase: str,
    t: PerfCounters,
    counters: Dict[str, int]
):
    counters[f"{phase}_wall_time_us"] = int(t.wall_time_s * US_IN_S)
    counters[f"{phase}_user_time_us"] = int(t.user_time_s * US_IN_S)


def run_on_dataset(
    ds: Dataset,
    M: int,
    num_threads: int,
    num_add_iterations: int,
    num_search_iterations: int,
    efSearch: int = 16,
    efConstruction: int = 40,
    search_bounded_queue: bool = True,
) -> Dict[str, int]:
    xq = ds.get_queries()
    xb = ds.get_database()

    nb, d = xb.shape
    nq, d = xq.shape

    k = 10
    # pyre-ignore[16]: Module `faiss` has no attribute `omp_set_num_threads`.
    faiss.omp_set_num_threads(num_threads)
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction  # default
    with timed_execution() as t:
        for _ in range(num_add_iterations):
            index.add(xb)
    counters = {}
    accumulate_perf_counter("add", t, counters)
    counters["nb"] = nb
    counters["num_add_iterations"] = num_add_iterations

    index.hnsw.efSearch = efSearch
    index.hnsw.search_bounded_queue = search_bounded_queue
    with timed_execution() as t:
        for _ in range(num_search_iterations):
            D, I = index.search(xq, k)
    accumulate_perf_counter("search", t, counters)
    counters["nq"] = nq
    counters["efSearch"] = efSearch
    counters["efConstruction"] = efConstruction
    counters["M"] = M
    counters["d"] = d
    counters["num_search_iterations"] = num_search_iterations

    return counters


def run(
    d: int,
    nb: int,
    nq: int,
    M: int,
    num_threads: int,
    num_add_iterations: int = 1,
    num_search_iterations: int = 1,
    efSearch: int = 16,
    efConstruction: int = 40,
    search_bounded_queue: bool = True,
) -> Dict[str, int]:
    ds = SyntheticDataset(d=d, nb=nb, nt=0, nq=nq, metric="L2", seed=1338)
    return run_on_dataset(
        ds,
        M=M,
        num_add_iterations=num_add_iterations,
        num_search_iterations=num_search_iterations,
        num_threads=num_threads,
        efSearch=efSearch,
        efConstruction=efConstruction,
        search_bounded_queue=search_bounded_queue,
    )


def _accumulate_counters(
    element: Dict[str, int], accu: Optional[Dict[str, List[int]]] = None
) -> Dict[str, List[int]]:
    if accu is None:
        accu = {key: [value] for key, value in element.items()}
        return accu
    else:
        assert accu.keys() <= element.keys(), (
            "Accu keys must be a subset of element keys: "
            f"{accu.keys()} not a subset of {element.keys()}"
        )
        for key in accu.keys():
            accu[key].append(element[key])
        return accu


def main():
    parser = argparse.ArgumentParser(description="Benchmark HNSW")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=5)
    parser.add_argument("--warm-up-iterations", type=int, default=0)
    parser.add_argument("--num-search-iterations", type=int, default=1)
    parser.add_argument("--num-add-iterations", type=int, default=1)
    parser.add_argument("--num-repetitions", type=int, default=1)
    parser.add_argument("--ef-search", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=40)
    parser.add_argument("--search-bounded-queue", action="store_true")

    parser.add_argument("--nb", type=int, default=5000)
    parser.add_argument("--nq", type=int, default=500)
    parser.add_argument("--d", type=int, default=128)
    args = parser.parse_args()

    if args.warm_up_iterations > 0:
        print(f"Warming up for {args.warm_up_iterations} iterations...")
        # warm-up
        run(
            num_search_iterations=args.warm_up_iterations,
            num_add_iterations=args.warm_up_iterations,
            d=args.d,
            nb=args.nb,
            nq=args.nq,
            M=args.M,
            num_threads=args.num_threads,
            efSearch=args.ef_search,
            efConstruction=args.ef_construction,
            search_bounded_queue=args.search_bounded_queue,
        )
    print(
        f"Running benchmark with dataset(nb={args.nb}, nq={args.nq}, "
        f"d={args.d}), M={args.M}, num_threads={args.num_threads}, "
        f"efSearch={args.ef_search}, efConstruction={args.ef_construction}"
    )
    result = None
    for _ in range(args.num_repetitions):
        counters = run(
            num_search_iterations=args.num_search_iterations,
            num_add_iterations=args.num_add_iterations,
            d=args.d,
            nb=args.nb,
            nq=args.nq,
            M=args.M,
            num_threads=args.num_threads,
            efSearch=args.ef_search,
            efConstruction=args.ef_construction,
            search_bounded_queue=args.search_bounded_queue,
        )
        result = _accumulate_counters(counters, result)
    assert result is not None
    for counter, values in result.items():
        if is_perf_counter(counter):
            print(
                "%s t=%.3f us (± %.4f)" % 
                (counter, np.mean(values), np.std(values))
            )


if __name__ == "__main__":
    main()
