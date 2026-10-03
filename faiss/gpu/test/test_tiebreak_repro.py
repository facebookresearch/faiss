#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tie-break and padding tests for the classic (non-cuVS) GPU IVF search.

`TieBreakComparator<float>` orders an equal distance by the value, which pass 1
sets to the user id. Two properties follow, and this file tests both:

- Pass 2 must reject a padded entry by the list length. A pad ties with the
  init key, so it can now win and make pass 2 decode a sentinel index. That
  raised `CUDA error 700 an illegal memory access was encountered`.
- The GPU must break a tie the same way the CPU does.

The shapes below give short lists on purpose, so a probed list holds fewer than
`k` vectors and pass 1 pads the rest.
"""

import unittest

import faiss
import numpy as np

D = 32
N = 20000
NQ = 256
SEED = 1234

# k above the average list length (N / nlist) leaves padded entries in the
# pass-1 output for pass 2 to select over. k stops at 512: at 1024 the scan
# asks for 48 KB of shared memory and fails to launch, a separate limit.
SHAPES = [(10, 1), (100, 1), (100, 8), (100, 64), (512, 8)]
FACTORIES = ["IVF1024,SQ8", "IVF1024,Flat", "IVF1024,PQ8"]

# Copies of each distinct vector, so a query ties against a whole group.
DUPLICATES = 8
REPEATS = 20


def build_cpu_index(
    factory: str, duplicates: int = 1
) -> tuple[faiss.Index, np.ndarray]:
    """Build an index over N vectors.

    `duplicates` > 1 repeats each distinct vector, so a query ties exactly
    against a whole group and the tie straddles the k-th place.
    """
    rs = np.random.RandomState(SEED)
    xb = rs.rand(N // duplicates, D).astype("float32")
    xb = np.ascontiguousarray(np.repeat(xb, duplicates, axis=0))
    index = faiss.index_factory(D, factory, faiss.METRIC_L2)
    index.train(xb)
    index.add(xb)
    return index, xb


class TieBreakSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.res = faiss.StandardGpuResources()
        rs = np.random.RandomState(SEED + 1)
        self.xq: np.ndarray = rs.rand(NQ, D).astype("float32")

    def _to_gpu(self, cpu_index: faiss.Index) -> faiss.Index:
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        return faiss.index_cpu_to_gpu(self.res, 0, cpu_index, co)

    def test_short_lists_do_not_fault(self) -> None:
        for factory in FACTORIES:
            cpu_index, _ = build_cpu_index(factory)
            gpu_index = self._to_gpu(cpu_index)
            for k, nprobe in SHAPES:
                with self.subTest(factory=factory, k=k, nprobe=nprobe):
                    gpu_index.nprobe = nprobe
                    _, ids = gpu_index.search(self.xq, k)
                    out_of_range = int((ids[ids >= 0] >= N).sum())
                    self.assertEqual(
                        out_of_range,
                        0,
                        f"{factory} k={k} nprobe={nprobe} returned "
                        f"{out_of_range} ids outside [0, {N})",
                    )

    def test_empty_lists_do_not_fault(self) -> None:
        """A sharded index probes lists that hold no vectors at all, because
        the shard owns only some of the lists. Keep one list to reproduce it.
        """
        factory = "IVF1024,SQ8"
        cpu_index, xb = build_cpu_index(factory)
        ivf = faiss.extract_index_ivf(cpu_index)
        # Keep list 0 only, so every other probed list is present but empty.
        for list_no in range(1, ivf.nlist):
            ivf.invlists.resize(list_no, 0)
        ivf.ntotal = ivf.invlists.list_size(0)

        gpu_index = self._to_gpu(cpu_index)
        for k, nprobe in SHAPES:
            with self.subTest(k=k, nprobe=nprobe):
                gpu_index.nprobe = nprobe
                _, ids = gpu_index.search(self.xq, k)
                out_of_range = int((ids[ids >= 0] >= len(xb)).sum())
                self.assertEqual(
                    out_of_range,
                    0,
                    f"k={k} nprobe={nprobe} returned {out_of_range} ids "
                    f"outside [0, {len(xb)})",
                )

    def test_tied_distances_select_the_same_neighbours(self) -> None:
        """`PQ8` approximates the distance, so the CPU comparison below cannot
        assert on it. This covers it by repeating the same search instead.
        """
        for factory in FACTORIES:
            cpu_index, _ = build_cpu_index(factory, duplicates=DUPLICATES)
            gpu_index = self._to_gpu(cpu_index)
            for k, nprobe in SHAPES:
                with self.subTest(factory=factory, k=k, nprobe=nprobe):
                    gpu_index.nprobe = nprobe
                    first_d, first_i = gpu_index.search(self.xq, k)
                    ties = self._count_boundary_ties(first_d)
                    for repeat in range(1, REPEATS):
                        d, i = gpu_index.search(self.xq, k)
                        np.testing.assert_array_equal(
                            i,
                            first_i,
                            f"{factory} k={k} nprobe={nprobe} repeat {repeat} "
                            f"returned different neighbours ({ties} queries "
                            "tie at the k-th place)",
                        )
                        np.testing.assert_array_equal(d, first_d)

    def test_gpu_breaks_a_tie_the_way_the_cpu_does(self) -> None:
        """The CPU keeps the smaller id on a tie, in `CMax::cmp2` in
        `utils/ordered_key_value.h`. `SQ8` and `Flat` compute the same
        distances as the CPU, so the GPU must return the same ids in the same
        order. `PQ8` approximates the distance, so it only reports.

        Measured on one A100 at k=100, nprobe=8, SQ8: without the tie-break
        135/256 queries match the CPU as a set and 0/256 match in order. With
        the tie-break, 256/256 on both.
        """
        for factory in FACTORIES:
            cpu_index, _ = build_cpu_index(factory, duplicates=DUPLICATES)
            gpu_index = self._to_gpu(cpu_index)
            for k, nprobe in [(100, 8), (512, 8)]:
                with self.subTest(factory=factory, k=k, nprobe=nprobe):
                    cpu_index.nprobe = nprobe
                    gpu_index.nprobe = nprobe
                    cpu_d, cpu_i = cpu_index.search(self.xq, k)
                    gpu_d, gpu_i = gpu_index.search(self.xq, k)

                    ties = self._count_boundary_ties(gpu_d)
                    same_set = sum(
                        set(cpu_i[q].tolist()) == set(gpu_i[q].tolist())
                        for q in range(NQ)
                    )
                    same_order = int((cpu_i == gpu_i).all(axis=1).sum())
                    print(
                        f"{factory} k={k} nprobe={nprobe}: "
                        f"{ties}/{NQ} queries tie at the k-th place, "
                        f"{same_set}/{NQ} match the CPU as a set, "
                        f"{same_order}/{NQ} match the CPU in order",
                        flush=True,
                    )
                    if "PQ" not in factory:
                        self.assertEqual(same_set, NQ)
                        self.assertEqual(same_order, NQ)

    @staticmethod
    def _count_boundary_ties(distances: np.ndarray) -> int:
        """Queries whose k-th and (k-1)-th neighbour are the same distance.

        A count of 0 means the case proves nothing, so the tests report it.
        """
        if distances.shape[1] < 2:
            return 0
        return int((distances[:, -1] == distances[:, -2]).sum())
