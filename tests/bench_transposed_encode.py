# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import time
import unittest

import faiss
import numpy as np


def _time_compute_codes(
    pq: faiss.ProductQuantizer, xb: np.ndarray, n_repeat: int
) -> tuple[float, int]:
    # warmup (also primes any lazy state)
    codes = pq.compute_codes(xb)
    checksum = int(codes.astype(np.int64).sum())
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        pq.compute_codes(xb)
    dt = time.perf_counter() - t0
    return dt / n_repeat, checksum


def run_case(
    d: int,
    m: int,
    nbits: int,
    nb: int,
    ntrain: int,
    n_repeat: int,
) -> None:
    dsub = d // m
    rng = np.random.RandomState(1234)
    xt = rng.rand(ntrain, d).astype("float32")
    xb = rng.rand(nb, d).astype("float32")

    pq = faiss.ProductQuantizer(d, m, nbits)
    pq.train(xt)
    # populate transposed_centroids -> compute_1_code uses the
    # fvec_L2sqr_ny_nearest_y_transposed kernel.
    pq.sync_transposed_centroids()

    threads = faiss.omp_get_max_threads()
    print(
        f"\n=== d={d} M={m} dsub={dsub} nbits={nbits} "
        f"nb={nb:,} repeats={n_repeat} (threads={threads}) ==="
    )
    print(f"{'level':<10} {'ms/encode-call':>15} {'vs AVX512':>12}  checksum")

    levels = [
        faiss.SIMDLevel_NONE,
        faiss.SIMDLevel_AVX2,
        faiss.SIMDLevel_AVX512,
    ]
    names = {
        faiss.SIMDLevel_NONE: "NONE",
        faiss.SIMDLevel_AVX2: "AVX2",
        faiss.SIMDLevel_AVX512: "AVX512",
    }

    results: dict[str, float] = {}
    checks: dict[str, int] = {}
    for lvl in levels:
        if not faiss.SIMDConfig.is_simd_level_available(lvl):
            print(f"{names[lvl]:<10} {'(not available)':>15}")
            continue
        faiss.SIMDConfig.set_level(lvl)
        per_call, checksum = _time_compute_codes(pq, xb, n_repeat)
        results[names[lvl]] = per_call
        checks[names[lvl]] = checksum

    avx512 = results.get("AVX512")
    for name in ("NONE", "AVX2", "AVX512"):
        if name not in results:
            continue
        ms = results[name] * 1e3
        ratio = (results[name] / avx512) if avx512 else float("nan")
        print(f"{name:<10} {ms:>15.3f} {ratio:>11.2f}x  {checks[name]}")

    # correctness: all levels must produce identical codes
    distinct = set(checks.values())
    status = "YES" if len(distinct) == 1 else f"NO {checks}"
    print(f"codes identical across levels: {status}")
    if "NONE" in results and avx512:
        speedup = results["NONE"] / avx512
        print(f"==> AVX512 fix speedup vs pre-fix NONE: {speedup:.2f}x")


def main() -> None:
    print(
        "faiss build SIMD level (auto-detected):",
        faiss.SIMDConfig.get_level(),
    )
    print(
        "AVX512 available:",
        faiss.SIMDConfig.is_simd_level_available(faiss.SIMDLevel_AVX512),
    )
    # Single-threaded for a clean per-core kernel signal.
    faiss.omp_set_num_threads(1)

    # dsub in {1,2,4,8} -> the fast _D specialization is taken.
    run_case(d=128, m=32, nbits=8, nb=1_000_000, ntrain=100_000, n_repeat=5)
    run_case(d=64, m=16, nbits=8, nb=1_000_000, ntrain=100_000, n_repeat=5)
    # dsub=16 -> _D fast path NOT taken (control: no AVX512 vs NONE gap).
    run_case(d=128, m=8, nbits=8, nb=1_000_000, ntrain=100_000, n_repeat=5)
    sys.stdout.flush()


class BenchTransposedEncode(unittest.TestCase):
    def test_bench(self) -> None:
        main()


if __name__ == "__main__":
    main()
