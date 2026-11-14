# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""BF16 removal revival: CPU search/collectives and optional CUDA kernels."""

import datetime
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.ivf_flat import IndexIVFFlat
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.sharded_centroids import (
    EmeansShardedCentroids, _partial_removal_donors, _protect_removal_candidates,
    _removal_candidates, _removal_scalar_stats,
)
from faiss.contrib.e_means.sharded_centroids_ivf import EmeansShardedCentroidsIVF
from faiss.contrib.e_means.torch_backend import TorchBackend


class _CPUAssignment(TorchBackend):
    """Evaluate rounded BF16 operands with CPU FP32 dots."""

    def assign(self, rows, centroids, *args, **kwargs):
        return super().assign(rows.float(), centroids.float(), *args, **kwargs)


def _index(coarse, candidates, nprobe, **kwargs):
    index = IndexIVFFlat(coarse, nprobe=nprobe, **kwargs)
    index.dtype = "bf16"
    index.add(candidates)
    return index


def _ivf_protection_worker(rank, rendezvous, device_type):
    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo",
                            init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120))
    try:
        check = unittest.TestCase()
        initial = torch.tensor([[0., 0.], [.25, 0.], [10., 0.],
                                [10.25, 0.], [30., 0.], [50., 0.]], device=device)
        coarse = initial[[0, 2, 5]]
        shard = slice(rank * 3, (rank + 1) * 3)
        model = EmeansShardedCentroidsIVF(2, 6, group=dist.group.WORLD,
                                        nlist=3, nprobe=1, batch_size=2)
        QuantizedEmeans.begin(model, 20, initial[shard])
        model._ivf = _index(coarse, initial[shard], 1)
        model._prepare_assignment()  # Training prepares global routing occupancy.
        # Self is excluded, including on a different owner. The isolated
        # list returns the sentinel without an exhaustive fallback.
        sentinel = torch.iinfo(torch.int64).max
        expected = [1, 0, 3, 2, 3, sentinel]
        with mock.patch.object(model._ivf, "route", wraps=model._ivf.route) as route:
            np.testing.assert_array_equal(model._nearest_other(np.arange(6)), expected)
            check.assertEqual(route.call_count, 1 if rank == 0 else 0)
        check.assertEqual(len(model._nearest_other(np.empty(0, dtype="int64"))), 0)
        # Revival reuses the training snapshot after update: queries move,
        # but stored candidates and list membership stay frozen.
        if rank == 1:
            model._centroids[1, 0] = 49.
        with mock.patch.object(model._ivf, "add", side_effect=AssertionError("revival rebuilt IVF")):
            np.testing.assert_array_equal(model._nearest_other(np.array([4, 5])), [5, sentinel])
        # The next training batch rebuilds the lists and sees the moved row.
        model._prepare_assignment()
        np.testing.assert_array_equal(model._nearest_other(np.array([4, 5])), [5, 4])
        # Probing every list agrees with a represented exhaustive oracle.
        model._ivf.nprobe = model.nprobe = 3
        all_centroids = initial.clone()
        all_centroids[4, 0] = 49.
        represented = all_centroids.bfloat16().float()
        scores = represented.square().sum(1)[None] - 2 * represented @ represented.T
        scores.fill_diagonal_(torch.inf)
        np.testing.assert_array_equal(model._nearest_other(np.arange(6)),
                                      scores.argmin(1).cpu().numpy())
        # Duplicate centroids can put self first or second in the local pair.
        # Equal scores on different owners must retain the lowest global ID.
        all_centroids[1] = all_centroids[0]
        all_centroids[3] = all_centroids[2]
        model._centroids.copy_(all_centroids[shard])
        model._prepare_assignment()
        np.testing.assert_array_equal(model._nearest_other(np.arange(6)),
                                      [1, 0, 3, 2, 5, 4])
        # Only rank 0 ranks the donors. The two equal-gain winners must be
        # IDs 1 and 3, and both owners receive the same split decisions.
        state = torch.tensor([[1., 0., 0.], [20., 0., 10.], [2., 0., 0.],
                              [20., 0., 10.], [20., 0., 10.], [20., 0., 10.]], device=device)
        model._usage.fill_(1)
        model._removal_stats.copy_(state[shard])
        model._axes[:] = torch.tensor([1., 0.], device=device)
        oracle = QuantizedEmeans(2, 6, quantizer=CastQuantizer(2, torch.bfloat16))
        oracle.begin(20, all_centroids)
        oracle._usage.fill_(1)
        oracle._removal_stats.copy_(state)
        oracle._axes[:] = torch.tensor([1., 0.], device=device)
        oracle._apply_splits(torch.tensor([1, 3], device=device),
                             torch.tensor([0, 2], device=device))
        check.assertEqual(model._revive_unused(torch.Generator()), 2)
        for name in ("_centroids", "_usage", "_removal_stats", "_axes", "_moved"):
            torch.testing.assert_close(getattr(model, name), getattr(oracle, name)[shard], rtol=0, atol=0)
        # A split invalidates cached lengths. A check before another update
        # recomputes them; only rank 0 receives the global FP32 table.
        scalars = model._gather_removal_scalars()
        if rank == 0:
            torch.testing.assert_close(scalars,
                _removal_scalar_stats(oracle._removal_stats, oracle._usage).float())
        else:
            check.assertEqual(scalars.shape, (0, 3))
        model._removal_stats[:, -1].fill_(10.)
        with mock.patch.object(model, "_nearest_other", side_effect=AssertionError("searched without candidates")):
            check.assertEqual(model._revive_unused(torch.Generator()), 0)
        model._removal_stats[:, -1].zero_()
        model._usage.zero_()
        check.assertEqual(model._revive_unused(torch.Generator()), 0)  # Candidates, but no donors.
    finally:
        dist.destroy_process_group()


def _ivf_statistics_worker(rank, rendezvous):
    """Exercise the default IVF backend against its unfused statistics path."""
    from faiss.contrib.e_means import ivf_kernels, triton_assignment

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    try:
        check = unittest.TestCase()
        for d in (17, 1024):
            generator = torch.Generator(device=device).manual_seed(53)
            initial = torch.randn(32, d, device=device, generator=generator)
            rows = torch.randn(64, d, device=device, generator=generator)
            shard = slice(rank * 16, (rank + 1) * 16)
            local = rows[rank * 32:(rank + 1) * 32]
            models = []
            for _ in range(2):
                model = EmeansShardedCentroidsIVF(d, 32, group=dist.group.WORLD,
                    nlist=2, nprobe=2, batch_size=64, nepochs=3, alpha0=.5, seed=7)
                model.begin(256, initial[shard])
                model._ivf = _index(initial[:2], initial[shard], 2)
                # Include an axis whose squared norm overflows FP32.
                model._removal_stats[1, :-1] = 1e20
                models.append(model)
            actual, expected = models
            with mock.patch.object(expected, "_use_fused_owned_statistics", return_value=False), \
                 mock.patch.object(expected, "_update_removal_statistics", return_value=False), \
                 mock.patch.object(ivf_kernels, "accumulate_owned", wraps=ivf_kernels.accumulate_owned) as accumulate, \
                 mock.patch.object(triton_assignment, "removal_update", wraps=triton_assignment.removal_update) as update:
                for _ in range(2):
                    stats = []
                    for model in models:
                        check.assertIsNone(model.accumulate(local[:16])["sums"])
                        stats.append(model.accumulate(local[16:]))
                    check.assertEqual(actual._assignment_backend, "native")
                    for key in ("sums", "counts", "removal", "objective"):
                        torch.testing.assert_close(stats[0][key], stats[1][key], rtol=1e-4, atol=1e-4)
                    for model, batch in zip(models, stats):
                        model.update(batch)
                    for name in ("_centroids", "_usage", "_removal_stats", "_axes", "_batch_removal"):
                        torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                                   rtol=1e-4, atol=1e-4)
                    check.assertFalse(bool(actual._batch_removal.any()))
                    expected_scalars = _removal_scalar_stats(expected._removal_stats, expected._usage).float()
                    actual_norms = torch.linalg.vector_norm(actual._removal_stats[:, :-1],
                                                           dim=1, dtype=torch.float64).float()
                    torch.testing.assert_close(actual._axis_norms, actual_norms, rtol=1e-6, atol=1e-6)
                    full_expected = expected._gather_local(expected_scalars)
                    # No vector-table scan or CPU cost transfer is needed
                    # after a normal update writes the cached axis lengths.
                    with mock.patch.object(torch.linalg, "vector_norm",
                                           side_effect=AssertionError("recomputed cached norms")):
                        scalars = actual._gather_removal_scalars()
                    check.assertEqual(scalars.dtype, torch.float32)
                    if rank == 0:
                        torch.testing.assert_close(scalars, full_expected, rtol=1e-4, atol=1e-4)
                    else:
                        check.assertEqual(scalars.shape, (0, 3))
                check.assertEqual(accumulate.call_count, 4)
                check.assertEqual(update.call_count, 2)
    except BaseException:
        # Report an assertion before NCCL teardown waits for another rank.
        import traceback
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def _collective_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    try:
        check = unittest.TestCase()
        initial = torch.tensor([[0., 0.], [2., 0.], [2., 0.], [10., 0.]])
        rows = torch.tensor([[.25, 0.], [2., .5], [9., .25], [1., 0.]])
        shard = slice(rank * 2, (rank + 1) * 2)
        for ivf in (False, True):
            cls = EmeansShardedCentroidsIVF if ivf else EmeansShardedCentroids
            kwargs = dict(nlist=2, nprobe=2) if ivf else dict(quantizer=CastQuantizer(2, torch.bfloat16))
            model = cls(2, 4, group=dist.group.WORLD, nepochs=2.6,
                        batch_size=4, alpha0=.5, assignment_chunk_size=1, **kwargs)
            # Exercise production owner logic on CPU; public begin requires CUDA.
            QuantizedEmeans.begin(model, 40, initial[shard])
            model._execution_backend = _CPUAssignment(model.centroid_chunk_size)
            if ivf:
                model._ivf = _index(initial[[0, 3]], initial[shard], 2)
            model._axes[:] = torch.tensor([1., 0.])
            local = rows[shard] if ivf else rows[shard].bfloat16()
            partial = model.accumulate(local[:1])
            check.assertIsNone(partial["sums"])
            stats = model.accumulate(local[1:])
            distances = (rows[:, None] - initial[None]).square().sum(-1)
            order = torch.argsort(distances, dim=1, stable=True)
            winner = order[:, 0]
            costs = distances.gather(1, order[:, 1:2]).flatten() - distances.gather(1, order[:, :1]).flatten()
            owned = (winner >= rank * 2) & (winner < (rank + 1) * 2)
            sums, counts = torch.zeros(2, 2), torch.zeros(2)
            removal = torch.zeros(2, 3)
            for i in torch.nonzero(owned).flatten():
                j = winner[i] - rank * 2
                sums[j] += rows[i]
                counts[j] += 1
                y = rows[i] - initial[winner[i]]
                removal[j, :2] += y[0] * y
                removal[j, 2] += costs[i]
            torch.testing.assert_close(stats["sums"], sums)
            torch.testing.assert_close(stats["counts"], counts)
            torch.testing.assert_close(stats["removal"], removal)
            check.assertEqual(stats["n"], 4)
            model.update(stats)
            torch.testing.assert_close(model._removal_stats, removal * .5)
            torch.testing.assert_close(model._batch_removal, torch.zeros_like(removal))

            # Victim 0 protects centroid 1 on rank 0. Victim 2 protects
            # centroid 1 across ranks. Oja gain picks donors 3 then 1.
            model._centroids[:] = initial[shard]
            model._usage[:] = 1
            state = torch.tensor([[1., 0., 0.], [10., 0., 10.],
                                  [2., 0., 0.], [20., 0., 10.]])
            model._removal_stats[:] = state[shard]
            model._axes[:] = torch.tensor([1., 0.])
            nearest = model._nearest_other(np.arange(4, dtype="int64"))
            np.testing.assert_array_equal(nearest, [1, 2, 1, 1])
            oracle = QuantizedEmeans(2, 4, quantizer=CastQuantizer(2, torch.bfloat16))
            oracle.begin(40, initial)
            oracle._execution_backend = _CPUAssignment(oracle.centroid_chunk_size)
            oracle._usage[:] = 1
            oracle._removal_stats[:] = state
            oracle._axes[:] = torch.tensor([1., 0.])
            oracle._apply_splits(torch.tensor([3, 1]), torch.tensor([0, 2]))
            check.assertEqual(model._revive_unused(torch.Generator(), chunk_rows=1), 2)
            for name in ("_centroids", "_usage", "_removal_stats", "_axes", "_moved"):
                torch.testing.assert_close(getattr(model, name), getattr(oracle, name)[shard], rtol=0, atol=0)
            check.assertEqual(model.n_splits, 2)
            # T=2.6 schedules [40, 24, 40]; no moves in the closing pass.
            model._epoch, model._epoch_rows = 2, 0
            before = model._centroids.clone()
            check.assertEqual(model._revive_unused(torch.Generator()), 0)
            torch.testing.assert_close(model._centroids, before)

            # A sub-epoch run can split during its only, closing pass. Those
            # means must exclude stale assignments from before the split.
            short = cls(2, 4, group=dist.group.WORLD, nepochs=.6,
                        batch_size=4, alpha0=.5, assignment_chunk_size=1, **kwargs)
            QuantizedEmeans.begin(short, 40, initial[shard])
            short._execution_backend = _CPUAssignment(short.centroid_chunk_size)
            if ivf:
                short._ivf = _index(initial[[0, 3]], initial[shard], 2)
            short_rows = rows[shard].repeat(6, 1)
            if not ivf:
                short_rows = short_rows.bfloat16()
            for batch in short_rows.split(2):
                short.update(short.accumulate(batch))
            short.end_epoch()
            check.assertGreater(short.n_splits, 0)
            expected = short._centroids.clone()
            means = short._pass_sums / short._pass_counts.clamp_min(1).float()[:, None]
            close = (short._pass_counts > 0) & ~short._moved
            expected[close] = means[close]
            torch.testing.assert_close(short.finalize(), expected)

        # One centroid per owner still has a global second candidate.
        model = EmeansShardedCentroids(1, 2, group=dist.group.WORLD,
                                      quantizer=CastQuantizer(1, torch.bfloat16), batch_size=2)
        QuantizedEmeans.begin(model, 2, torch.tensor([[float(rank)]]))
        check.assertIsNotNone(model._removal_stats)
        first = torch.tensor([0., 1., 0.]) if rank == 0 else torch.tensor([0., 0., 2.])
        labels = torch.zeros(3, dtype=torch.int64)
        winner, best, second = model._reduce_top2(first, labels, torch.full((3,), torch.inf))
        torch.testing.assert_close(winner, torch.tensor([0, 1, 0]))
        torch.testing.assert_close(best, torch.tensor([0., 0., 0.]))
        torch.testing.assert_close(second, torch.tensor([0., 1., 2.]))
    finally:
        dist.destroy_process_group()


class TestBF16Removal(unittest.TestCase):
    def test_candidate_selection_and_protection(self):
        self._check_candidate_selection_and_protection("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_candidate_selection_and_protection(self):
        self._check_candidate_selection_and_protection("cuda")

    def _check_candidate_selection_and_protection(self, device):
        rng = np.random.RandomState(37)
        for n in (0, 1, 2, 17, 128, 1025):
            values = rng.randint(-3, 20, size=n).astype("float32")
            for special in (None, float("nan"), float("inf"), -float("inf")):
                removal = values.copy()
                if n and special is not None:
                    removal[-1] = special
                with np.errstate(invalid="ignore"):
                    median = np.median(removal.astype("float64")) if n else np.nan
                    candidates = np.flatnonzero((removal <= 0) | (removal.astype("float64") < .3 * median))
                candidates = candidates[np.argsort(removal[candidates], kind="stable")]
                actual = _removal_candidates(torch.as_tensor(removal, device=device))
                np.testing.assert_array_equal(actual.cpu().numpy(), candidates)
        # FP32 thresholds must still be compared in the original FP64 units.
        removal = np.array([.3, np.nextafter(np.float32(.3), np.float32(0)), 1., 1., 1.])
        np.testing.assert_array_equal(_removal_candidates(torch.as_tensor(removal, device=device)).cpu(), [1])
        for n in (0, 1, 17, 1025, 9001):
            candidates = rng.permutation(n * 2)[:n].astype("int64")
            for kind in ("random", "chain", "star", "self"):
                if kind == "random":
                    nearest = rng.randint(0, max(1, n * 2), size=n).astype("int64")
                    nearest[::11] = np.iinfo("int64").max
                elif kind == "chain":
                    nearest = np.roll(candidates, -1)
                    if n:
                        nearest[-1] = np.iinfo("int64").max
                elif kind == "star":
                    nearest = np.full(n, candidates[-1] if n else 0, dtype="int64")
                else:
                    nearest = candidates.copy()
                kept, protected = [], set()
                for j, neighbor in zip(candidates, nearest):
                    if j not in protected:
                        kept.append(j)
                        protected.add(neighbor)
                actual = _protect_removal_candidates(torch.as_tensor(candidates, device=device),
                                                    torch.as_tensor(nearest, device=device))
                np.testing.assert_array_equal(actual.cpu().numpy(), np.array(kept, dtype="int64"))

    def test_partial_donor_selection(self):
        self._check_partial_donor_selection("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_partial_donor_selection(self):
        self._check_partial_donor_selection("cuda")

    def _check_partial_donor_selection(self, device):
        rng = np.random.RandomState(83)
        gains = rng.randint(0, 7, size=1003).astype("float64")
        # Stable ties include FP64 gains that collapse to the same FP32 value.
        gains[3:7] = [np.inf, np.nan, 1. + 1e-10, 1.]
        usage = (rng.rand(len(gains)) > .25).astype("float64")
        for count in (0, 1, 5, 111, 750, 1003):
            kept = rng.permutation(len(gains))[:count]
            eligible = usage > 0
            eligible[kept] = False
            donors = np.flatnonzero(eligible)
            expected = donors[np.argsort(-gains[donors].astype("float32"), kind="stable")][:count]
            with mock.patch.object(torch, "argsort", wraps=torch.argsort) as sort:
                actual = _partial_removal_donors(kept,
                    torch.as_tensor(usage, device=device),
                    torch.as_tensor(gains, device=device))
                np.testing.assert_array_equal(actual.cpu().numpy(), expected)
                if len(expected):
                    sort.assert_called_once()
                    self.assertEqual(len(sort.call_args.args[0]), len(expected))
                else:
                    sort.assert_not_called()
        # Every gain is tied: select the lowest eligible IDs, not arbitrary
        # IDs from the kth-value kernel. Cover eligible NaNs as well.
        for gain in (0., float("inf"), float("nan")):
            result = _partial_removal_donors(np.array([1, 4]),
                torch.ones(7, device=device), torch.full((7,), gain, device=device))
            np.testing.assert_array_equal(result.cpu().numpy(), [0, 2])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_ivf_revival_protection(self):
        self._check_ivf_revival_protection("cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
                         "requires two CUDA GPUs and NCCL")
    def test_cuda_ivf_revival_protection(self):
        self._check_ivf_revival_protection("cuda")

    def _check_ivf_revival_protection(self, device_type):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_ivf_protection_worker,
                args=("file://" + directory + "/rendezvous", device_type),
                nprocs=2, start_method="spawn")

    def test_ivf_top2_chunked(self):
        self._check_ivf_top2_chunked("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_ivf_top2_chunked(self):
        self._check_ivf_top2_chunked("cuda")

    def _check_ivf_top2_chunked(self, device):
        torch.set_num_threads(1)
        generator = torch.Generator(device=device).manual_seed(19)
        coarse = torch.tensor([[0., 0.], [10., 0.], [20., 0.]], device=device)
        candidates = torch.randn(31, 2, generator=generator, device=device) * .25
        candidates[10:20, 0] += 10
        candidates[20:, 0] += 20
        candidates[1] = candidates[0]
        queries = torch.cat((candidates[:2], torch.randn(13, 2, generator=generator, device=device) * 10))
        reference = candidates.bfloat16().float()
        represented = queries.bfloat16().float()
        scores = reference.square().sum(1) - 2 * represented @ reference.T
        ids = torch.argsort(scores, dim=1, stable=True)[:, :2]
        for budget in (1, 20, 1 << 26):
            index = _index(coarse, candidates, 3, chunk_budget=budget)
            actual_scores, actual_ids = index.search_2nn(queries)
            torch.testing.assert_close(actual_ids.long(), ids)
            torch.testing.assert_close(actual_scores, scores.gather(1, ids))
            first_scores, first_ids = index.search_1nn(queries)
            torch.testing.assert_close(first_scores, actual_scores[:, 0])
            torch.testing.assert_close(first_ids, actual_ids[:, 0])

    def test_ivf_missing_candidates(self):
        coarse = torch.tensor([[0.], [10.], [20.]])
        for candidates in (torch.empty(0, 1), torch.tensor([[0.]]), torch.tensor([[0.], [0.]])):
            index = _index(coarse, candidates, 3, chunk_budget=1)
            scores, ids = index.search_2nn(torch.zeros(2, 1))
            expected = [0, 1] if len(candidates) == 2 else [0, -1] if len(candidates) == 1 else [-1, -1]
            torch.testing.assert_close(ids, torch.tensor([expected, expected], dtype=torch.int32))
            self.assertTrue(torch.isinf(scores[ids < 0]).all())
            empty_scores, empty_ids = index.search_2nn(torch.empty(0, 1))
            self.assertEqual(empty_scores.shape, (0, 2))
            self.assertEqual(empty_ids.shape, (0, 2))

    def test_bf16_selection_and_lazy_binding(self):
        initial = torch.tensor([[0.], [2.]])
        model = QuantizedEmeans(1, 2, batch_size=2)
        model.begin(4, initial)
        self.assertIsNone(model._removal_stats)
        model._bind_input_dtype(torch.bfloat16, cast=True)
        self.assertIsNone(model._unused_threshold)
        self.assertIsNone(model.recipe["unused_threshold"])
        self.assertEqual(model._removal_stats.shape, (2, 2))
        for dtype in (torch.float16, torch.float32):
            other = QuantizedEmeans(1, 2, quantizer=None)
            other.begin(4, initial)
            other._bind_input_dtype(dtype)
            self.assertEqual(other._removal_stats.shape, (2, 2))
            self.assertIsNone(other._unused_threshold)
            self.assertIsNone(other.recipe["unused_threshold"])
        for dtype in (torch.float8_e4m3fn, torch.int8):
            other = QuantizedEmeans(1, 2, quantizer=None)
            other.begin(4, initial)
            other._bind_input_dtype(dtype)
            self.assertIsNone(other._removal_stats)
            self.assertIsNotNone(other._unused_threshold)
        for threshold in (0, .1):
            other = QuantizedEmeans(1, 2, quantizer=CastQuantizer(1, torch.bfloat16), unused_threshold=threshold)
            other.begin(4, initial)
            self.assertIsNone(other._removal_stats)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_owner_statistics_and_cost_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_collective_worker,
                                                 args=("file://" + directory + "/rendezvous",),
                                                 nprocs=2, start_method="spawn")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_owner_statistics(self):
        from faiss.contrib.e_means.ivf_kernels import accumulate_owned

        generator = torch.Generator(device="cuda").manual_seed(31)
        for d in (17, 128, 1024):
            for dtype in (torch.float32, torch.bfloat16):
                for cost in (False, True):
                    k, n, begin = 32, 513, 32
                    rows = torch.randn(n, d, device="cuda", generator=generator).to(dtype)
                    centroids = torch.randn(k, d, device="cuda", generator=generator)
                    axes = torch.randn_like(centroids)
                    axes /= axes.norm(dim=1, keepdim=True)
                    winners = torch.randint(2 * k, (n,), device="cuda", generator=generator)
                    margins = torch.rand(n, device="cuda", generator=generator) if cost else None
                    if cost:
                        winners[:3] = begin
                        margins[:3] = torch.tensor([float("nan"), float("inf"), float("-inf")], device="cuda")
                    reference = QuantizedEmeans(d, k, quantizer=None)
                    reference._centroids, reference._axes = centroids, axes
                    reference.centroid_begin = begin
                    expected = torch.zeros(k, d + 1, device="cuda")
                    expected_removal = torch.zeros_like(expected)
                    expected_obj = torch.zeros((), device="cuda", dtype=torch.float64)
                    reference._batch_removal = expected_removal
                    EmeansShardedCentroids._add_owned(reference, rows, winners,
                        expected[:, :-1], expected[:, -1], expected_obj, removal=margins)
                    actual, actual_removal = torch.zeros_like(expected), torch.zeros_like(expected)
                    actual_obj = torch.zeros_like(expected_obj)
                    self.assertTrue(accumulate_owned(rows, winners, centroids, axes,
                        actual, actual_removal if cost else None, actual_obj, margins, begin))
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                    if cost:
                        torch.testing.assert_close(actual_removal, expected_removal, rtol=1e-4, atol=1e-4)
                    torch.testing.assert_close(actual_obj, expected_obj, rtol=1e-10, atol=1e-6)

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
                         "requires two CUDA GPUs and NCCL")
    def test_cuda_ivf_fused_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_ivf_statistics_worker,
                args=("file://" + directory + "/rendezvous",), nprocs=2, start_method="spawn")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_centered_bf16_statistics(self):
        import importlib.util

        initial = torch.tensor([[1000., 1000.], [1002., 1000.], [1010., 1000.], [1020., 1000.]], device="cuda")
        rows = initial + torch.tensor([[.25, .5], [.5, -.25], [-.5, .25], [-1., .5]], device="cuda")
        backends = ["native"]
        if importlib.util.find_spec("triton") is not None:
            backends.append("triton")
            if importlib.util.find_spec("cutlass") is not None:
                backends.append("cute")
        for backend in backends:
            with self.subTest(backend=backend):
                quantizer = CastQuantizer(2, torch.bfloat16, center=True)
                quantizer.train(rows)
                model = QuantizedEmeans(2, 4, nepochs=1, batch_size=4, alpha0=.5,
                                        assignment_backend=backend, quantizer=quantizer)
                model.begin(4, initial)
                model._axes[:] = torch.tensor([1., 0.], device="cuda")
                stats = model.accumulate(rows)
                encoded = quantizer.compute_codes(rows).float()
                represented = quantizer.compute_codes(initial).float()
                distances = (encoded[:, None] - represented[None]).square().sum(-1)
                order = distances.argsort(dim=1, stable=True)
                winner = order[:, 0]
                values = quantizer.decode(encoded.bfloat16())
                y = values - initial[winner]
                removal = torch.zeros(4, 3, device="cuda")
                margin = distances.gather(1, order[:, 1:2]) - distances.gather(1, order[:, :1])
                removal.index_add_(0, winner, torch.cat((y[:, :1] * y, margin), dim=1))
                sums = torch.zeros_like(initial).index_add_(0, winner, values)
                torch.testing.assert_close(stats["sums"], sums)
                torch.testing.assert_close(stats["removal"], removal, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_external_nearest_kernel(self):
        from faiss.contrib.e_means.triton_assignment import nearest_other

        for d in (2, 128):
            generator = torch.Generator(device="cuda").manual_seed(9)
            centroids = torch.randn(131, d, device="cuda", generator=generator)
            centroids[3] = centroids[4]
            ids = torch.tensor([0, 3, 4, 130], device="cuda")
            expected = (centroids[ids, None].double() - centroids[None].double()).square().sum(-1)
            expected[torch.arange(len(ids), device="cuda"), ids] = torch.inf
            for begin, end in ((0, 1), (1, 131), (0, 131)):
                scores, labels = nearest_other(centroids[begin:end], ids, torch.tensor(len(ids), device="cuda"),
                                               queries=centroids[ids], centroid_begin=begin)
                distances = expected[:, begin:end]
                torch.testing.assert_close(scores, distances.min(dim=1).values, rtol=1e-14, atol=1e-14)
                valid = torch.isfinite(scores)
                torch.testing.assert_close(labels[valid], distances[valid].argmin(dim=1))


if __name__ == "__main__":
    unittest.main()
