# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, List

import faiss  # @manual=//faiss/python:pyfaiss
import numpy as np

try:
    import manifold.clients.python  # noqa: F401 @manual=//manifold/clients/python:manifold_client

    from faiss.contrib.datasets_fb import (  # @manual=//faiss/contrib:faiss_contrib
        dataset_from_name,
    )
except ImportError:
    from faiss.contrib.datasets import (  # @manual=//faiss/contrib:faiss_contrib
        dataset_from_name,
    )

US_IN_S = 1_000_000

SUPPORTED_DATASETS = ["sift1M", "gist1M"]

DEFAULT_FACTORY_STRINGS = [
    "IVF1024,Flat",
    "IVF1024,SQ4",
    "IVF1024,SQ8",
#    "RR,IVF1024,RaBitQ",
#    "RR,IVF1024,RaBitQ2",
#    "RR,IVF1024,RaBitQ4",
#    "RR,IVF1024,RaBitQ8",
#    "RR,IVF1024,RaBitQfs",
#    "RR,IVF1024,RaBitQfs2",
#    "RR,IVF1024,RaBitQfs4",
#    "RR,IVF1024,RaBitQfs8",
]

DEFAULT_QB = 4


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


def evaluate_recall(
    ground_truth: np.ndarray,
    result_ids: np.ndarray,
    k: int,
) -> float:
    assert ground_truth.shape[1] >= k
    assert result_ids.shape[1] >= k
    gt_k = ground_truth[:, :k]
    res_k = result_ids[:, :k]
    nq = gt_k.shape[0]
    correct = sum(
        len(set(gt_k[i]).intersection(set(res_k[i])))
        for i in range(nq)
    )
    return correct / (nq * k)


def get_index_memory_bytes(index: faiss.Index) -> int:
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return writer.data.size()


def format_bytes(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.2f} GB"


@dataclass
class BenchResult:
    factory_str: str
    memory_bytes: int
    qps: float
    recall: float


def run_single(
    factory_str: str,
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    num_threads: int,
    nprobe: int,
    k: int,
    qb: int,
    num_search_iterations: int,
) -> BenchResult:
    nb, d = xb.shape
    nq, _ = xq.shape

    faiss.omp_set_num_threads(num_threads)

    index = faiss.index_factory(d, factory_str)

    index.train(xt)
    index.add(xb)

    memory_bytes = get_index_memory_bytes(index)

    ivf = faiss.extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = nprobe

    if hasattr(index, "qb"):
        index.qb = qb

    with timed_execution() as t:
        for _ in range(num_search_iterations):
            if "RaBitQ" in factory_str:
                params = faiss.IVFRaBitQSearchParameters()
                params.nprobe = nprobe
                params.qb = qb
                _D, I, _ = faiss.search_with_parameters(
                    index, xq, k, params, output_stats=True
                )
            else:
                _D, I = index.search(xq, k)
    total_queries = nq * num_search_iterations
    qps = total_queries / t.wall_time_s

    recall = evaluate_recall(gt, I, k)

    return BenchResult(
        factory_str=factory_str,
        memory_bytes=memory_bytes,
        qps=qps,
        recall=recall,
    )


def print_table(results: List[BenchResult]) -> None:
    hdr_factory = "Index Factory"
    hdr_memory = "Memory"
    hdr_qps = "QPS"
    hdr_recall = "Recall@k"

    col_factory = max(
        len(hdr_factory), *(len(r.factory_str) for r in results)
    )
    col_memory = max(
        len(hdr_memory),
        *(len(format_bytes(r.memory_bytes)) for r in results),
    )
    col_qps = max(len(hdr_qps), 10)
    col_recall = max(len(hdr_recall), 8)

    sep = (
        f"+-{'-' * col_factory}-"
        f"+-{'-' * col_memory}-"
        f"+-{'-' * col_qps}-"
        f"+-{'-' * col_recall}-+"
    )
    header = (
        f"| {hdr_factory:<{col_factory}} "
        f"| {hdr_memory:>{col_memory}} "
        f"| {hdr_qps:>{col_qps}} "
        f"| {hdr_recall:>{col_recall}} |"
    )

    print(sep)
    print(header)
    print(sep)
    for r in results:
        mem_str = format_bytes(r.memory_bytes)
        print(
            f"| {r.factory_str:<{col_factory}} "
            f"| {mem_str:>{col_memory}} "
            f"| {r.qps:>{col_qps}.1f} "
            f"| {r.recall:>{col_recall}.4f} |"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark search on standard datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sift1M",
        choices=SUPPORTED_DATASETS,
        help="Dataset to benchmark (default: sift1M)",
    )
    parser.add_argument(
        "--index-factory",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more faiss index factory strings. "
            "If not specified, a default set is used."
        ),
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--qb", type=int, default=DEFAULT_QB,
        help=f"Query bits for RaBitQ (default: {DEFAULT_QB})",
    )
    parser.add_argument(
        "--num-search-iterations", type=int, default=3
    )
    args = parser.parse_args()

    factory_strings = (
        args.index_factory
        if args.index_factory
        else DEFAULT_FACTORY_STRINGS
    )

    ds = dataset_from_name(args.dataset)
    print(f"Loading {args.dataset} dataset ...")
    xq = ds.get_queries()
    xb = ds.get_database()
    xt = ds.get_train()
    gt = ds.get_groundtruth(k=args.k)
    nb, d = xb.shape
    nq, _ = xq.shape
    print(
        f"  d={d}, nb={nb}, nq={nq}, "
        f"num_threads={args.num_threads}, "
        f"nprobe={args.nprobe}, k={args.k}, "
        f"qb={args.qb}\n"
    )

    results: List[BenchResult] = []
    for factory_str in factory_strings:
        print(f"Benchmarking: {factory_str} ...")
        try:
            result = run_single(
                factory_str=factory_str,
                xb=xb,
                xq=xq,
                xt=xt,
                gt=gt,
                num_threads=args.num_threads,
                nprobe=args.nprobe,
                k=args.k,
                qb=args.qb,
                num_search_iterations=args.num_search_iterations,
            )
            results.append(result)
            print(
                f"  memory={format_bytes(result.memory_bytes)}, "
                f"QPS={result.qps:.1f}, "
                f"recall@{args.k}={result.recall:.4f}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")

    if results:
        print(f"\n=== {args.dataset} Benchmark Results ===\n")
        print_table(results)


if __name__ == "__main__":
    main()
