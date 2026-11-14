# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Closing consistency on the base sharded/caller APIs; no cost integration."""
import datetime
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids
from faiss.contrib.e_means.torch_backend import TorchBackend


class _CPUAssignment(TorchBackend):
    """Evaluate rounded BF16 operands with CPU FP32 dots."""

    def assign(self, rows, centroids, *args, **kwargs):
        return super().assign(rows.float(), centroids.float(), *args, **kwargs)



def _completed_closing_model(group, rows, labels, initial, budget, dtype, *, quantizer=None, spherical=False):
    """Small caller-driven completed snapshot; CPU uses the existing begin seam."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    local_k = len(initial) // size
    shard = slice(rank * local_k, (rank + 1) * local_k)
    model = EmeansShardedCentroids(initial.shape[1], len(initial), group=group,
        nepochs=budget, batch_size=len(rows), alpha0=1, unused_threshold=0,
        assignment_chunk_size=1, quantizer=quantizer, spherical=spherical, assignment_backend="native")
    if rows.is_cuda:
        model.begin(len(rows), initial[shard])
    else:
        QuantizedEmeans.begin(model, len(rows), initial[shard])
        model._execution_backend = _CPUAssignment(model.centroid_chunk_size)
    local_n = len(rows) // size
    model._resolve_first_input(rows[rank * local_n:(rank + 1) * local_n])
    represented = rows if quantizer is None else quantizer.decode(quantizer.compute_codes(rows))
    for epoch_rows in model.recipe["epoch_rows"]:
        indices = torch.cat([torch.arange(r * local_n, r * local_n + epoch_rows // size,
                                         device=rows.device) for r in range(size)])
        sums = torch.zeros(local_k, rows.shape[1], dtype=dtype, device=rows.device)
        counts = torch.zeros(local_k, dtype=dtype, device=rows.device)
        for j in range(local_k):
            selected = represented[indices[labels[indices] == j + rank * local_k]].to(dtype)
            sums[j] = selected.sum(0)
            counts[j] = len(selected)
        model.update(dict(sums=sums, counts=counts, objective=torch.zeros((), dtype=torch.float64,
                           device=rows.device), n=epoch_rows))
        model.end_epoch()
    return model, represented



def _closing_consistency_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        check = unittest.TestCase()
        rows = torch.tensor([0., 1., 2., 40., 50., 3., 4., 6., 8., 10.])[:, None]
        labels = torch.tensor([0] * 5 + [1] * 5)
        initial = torch.tensor([[0.], [4.], [20.], [30.]])
        shard = slice(rank * 2, rank * 2 + 2)
        local = rows[rank * 5:(rank + 1) * 5]
        for dtype in (torch.float32, torch.float64, torch.int64):
            for budget in (.6, 1.4, 2.6):
                model, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, budget, dtype)
                model._moved = torch.tensor([budget < 2, rank == 1])
                if rank == 0:
                    model._centroids[0] = 10
                frozen = model._centroids.clone()
                coverage = model.recipe["epoch_rows"][-1] // 2
                codes = torch.full((5,), -1, dtype=torch.int32)
                codes[:coverage] = rank
                previews = []
                where = torch.where
                def recorded_where(*args, **kwargs):
                    result = where(*args, **kwargs)
                    if result.dtype == torch.float32 and result.shape == (2, 1):
                        previews.append(result.clone())
                    return result
                alias = model._pass_sums
                immutable_sums = alias.clone()
                reductions = []
                reduce = dist.all_reduce
                def recorded_reduce(value, *args, **kwargs):
                    reductions.append(tuple(value.shape))
                    return reduce(value, *args, **kwargs)
                with mock.patch.object(torch.cuda, "synchronize"), \
                     mock.patch.object(torch, "where", side_effect=recorded_where), \
                     mock.patch.object(dist, "all_reduce", side_effect=recorded_reduce):
                    result = model.final_closing_mean([local[:2], local[2:]], assignments=codes, chunk_rows=1)
                check.assertEqual(reductions, [(1,), (2,)])
                check.assertGreaterEqual(len(previews), 2)
                check.assertTrue(torch.equal(previews[0].view(torch.int32), result.view(torch.int32)))
                global_result = [torch.empty_like(result) for _ in range(2)]
                dist.all_gather(global_result, result)
                returned = torch.cat(global_result)
                index = torch.cat((torch.arange(coverage), torch.arange(5, 5 + coverage)))
                expected_error = (rows[index].double() - returned[labels[index]].double()).square().sum().item() / len(index)
                check.assertAlmostEqual(model.closing_stats["assignment_mse"], expected_error, places=6)
                check.assertEqual(model.closing_stats["rows"], 2 * coverage)
                if rank == 0:
                    mean = rows[5:5 + coverage].to(dtype).sum(0).to(
                        torch.float64 if dtype != torch.float32 else torch.float32) / coverage
                    torch.testing.assert_close(result[1], mean.float(), rtol=0, atol=0)
                    if budget < 2:
                        torch.testing.assert_close(result[0], frozen[0], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(result, frozen, rtol=0, atol=0)
                torch.testing.assert_close(alias, immutable_sums, rtol=0, atol=0)
                check.assertIs(model.finalize(), result)
                with check.assertRaisesRegex(RuntimeError, "complete epochs"):
                    model.final_closing_mean([local], assignments=codes)

                # Same completed state, now a full fresh scan and its old snapshot error.
                fresh, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, budget, dtype)
                fresh._moved = torch.tensor([True, rank == 1])
                if rank == 0:
                    fresh._centroids[0] = 10
                snapshots = [torch.empty_like(fresh._centroids) for _ in range(2)]
                dist.all_gather(snapshots, fresh._centroids)
                snapshot = torch.cat(snapshots)
                distance = (rows[:, None] - snapshot[None]).square().sum(2)
                winners = distance.argmin(1)
                expected = snapshot.clone()
                for j in range(4):
                    selected = rows[winners == j].to(dtype)
                    if len(selected):
                        expected[j] = (selected.sum(0).to(torch.float64 if dtype != torch.float32 else torch.float32) / len(selected)).float()
                result = fresh.final_closing_mean([local[:2], local[2:]])
                torch.testing.assert_close(result, expected[shard], rtol=0, atol=0)
                check.assertEqual(fresh.closing_stats["assignment_mse"], distance.min(1).values.double().sum().item() / len(rows))
                check.assertIs(fresh.finalize(), result)
                print("closing CPU seam", rank, dtype, budget, "passed", flush=True)
    finally:
        dist.destroy_process_group()



class TestClosingConsistency(unittest.TestCase):
    def test_pure_closing_values_and_strided_finalizer(self):
        from faiss.contrib.e_means.e_means_torch import Emeans, _closing_centroid_values

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64, torch.int64):
                with self.subTest(device=device, dtype=dtype):
                    backing = torch.full((4, 6), -99, dtype=dtype, device=device)
                    sums = backing[:, ::2]
                    sums.copy_(torch.tensor([[6, 12, 18], [13, 7, 4], [0, 0, 0], [9, 12, 15]], device=device))
                    count_backing = torch.full((8,), -99, dtype=torch.float64, device=device)
                    counts = count_backing[::2]
                    counts.copy_(torch.tensor([3, 2**24 + 1, 0, 5], dtype=torch.float64, device=device))
                    current = torch.tensor([[10., 20., 30.], [4., 5., 6.], [-0., 7., 8.], [9., 8., 7.]],
                                           device=device).t().contiguous().t()
                    moved_backing = torch.zeros(8, dtype=torch.bool, device=device)
                    moved = moved_backing[::2]
                    moved[0] = True
                    originals = [v.clone() for v in (backing, count_backing, current, moved_backing)]
                    # Original finalize arithmetic on an owned copy, including count conversion.
                    promote = torch.float64 if dtype != torch.float32 else torch.float32
                    reference = sums.to(promote).clone()
                    divisor = counts.clamp_min(1).to(promote)[:, None]
                    if promote == torch.float64:
                        reference.div_(divisor)
                        reference = reference.float()
                    else:
                        reference = reference / divisor
                    expected = torch.where(((counts > 0) & ~moved)[:, None], reference, current)
                    values = _closing_centroid_values(sums, counts, current, moved)
                    self.assertTrue(torch.equal(values.view(torch.int32), expected.view(torch.int32)))
                    model = Emeans(3, 4, nepochs=1, batch_size=4, unused_threshold=0, assignment_backend="native")
                    model.begin(4, current)
                    model.update(dict(sums=torch.zeros(4, 3, device=device), counts=torch.ones(4, device=device),
                                      objective=torch.zeros((), dtype=torch.float64, device=device), n=4))
                    model.end_epoch()
                    # A value-only completed carrier exercises admitted strided working views.
                    model._centroids, model._pass_sums, model._pass_counts, model._moved = current, sums, counts, moved
                    result = model.finalize()
                    self.assertTrue(result.is_contiguous())
                    self.assertTrue(torch.equal(result.view(torch.int32), expected.contiguous().view(torch.int32)))
                    self.assertIs(model.finalize(), result)
                    for actual, original in zip((backing, count_backing, current, moved_backing), originals):
                        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_cpu_gloo_saved_and_fresh_closing(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_closing_consistency_worker,
                args=("file://" + directory + "/closing",), nprocs=2, start_method="spawn")

    @unittest.skipUnless(torch.cuda.is_available() and dist.is_nccl_available(), "requires CUDA and NCCL")
    def test_cuda_typed_saved_and_fresh_closing(self):
        from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer

        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method="file://" + directory + "/closing",
                                    rank=0, world_size=1, timeout=datetime.timedelta(seconds=90))
            try:
                device = torch.device("cuda")
                # Actual supported single-rank CUDA path; no additional GPU allocation.
                for dtype in (torch.float32, torch.float64, torch.int64):
                    for budget in (.6, 1.4, 2.6):
                        with self.subTest(dtype=dtype, budget=budget):
                            rows = torch.zeros(10, 1, device=device)
                            labels = torch.zeros(10, dtype=torch.int64, device=device)
                            initial = torch.tensor([[10.], [20.]], device=device)
                            model, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, budget, dtype)
                            model._centroids[0] = 10
                            model._moved = torch.tensor([True, False], device=device)
                            coverage = model.recipe["epoch_rows"][-1]
                            codes = torch.full((10,), -1, dtype=torch.int32)
                            codes[:coverage] = 0
                            result = model.final_closing_mean([rows[:3], rows[3:]], assignments=codes, chunk_rows=2)
                            torch.testing.assert_close(result, initial, rtol=0, atol=0)
                            self.assertEqual(model.closing_stats["assignment_mse"], 100.)
                            self.assertEqual(model.closing_stats["rows"], coverage)
                            self.assertIs(model.finalize(), result)
                            with self.assertRaisesRegex(RuntimeError, "complete epochs"):
                                model.final_closing_mean([rows])

                            fresh, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, budget, dtype)
                            fresh._centroids[0] = 10
                            fresh._moved = torch.tensor([True, False], device=device)
                            result = fresh.final_closing_mean([rows[:3], rows[3:]])
                            torch.testing.assert_close(result, torch.tensor([[0.], [20.]], device=device), rtol=0, atol=0)
                            self.assertEqual(fresh.closing_stats["assignment_mse"], 100.)
                            self.assertNotIn("returned_mse", fresh.closing_stats)
                            self.assertIs(fresh.finalize(), result)

                floating = torch.tensor([0., .5, 1., 2., 2.5, 3.], device=device)[:, None]
                codecs = [None, CastQuantizer(1, torch.bfloat16, center=True), Int8Quantizer(1)]
                for q in codecs:
                    if q is not None:
                        q.train(floating)
                    rows = floating
                    labels = torch.zeros(len(rows), dtype=torch.int64, device=device)
                    initial = torch.tensor([[10.], [20.]], device=device)
                    model, represented = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float32, quantizer=q)
                    model._centroids[0] = 10
                    model._moved = torch.tensor([True, False], device=device)
                    codes = torch.zeros(len(rows), dtype=torch.int32)
                    scan = q.compute_codes(rows) if q is not None and q.dtype == torch.int8 else rows
                    result = model.final_closing_mean([scan], assignments=codes, chunk_rows=2)
                    torch.testing.assert_close(result, initial, rtol=0, atol=0)
                    expected = (represented.double() - result[0].double()).square().sum().item() / len(rows)
                    self.assertAlmostEqual(model.closing_stats["assignment_mse"], expected, places=6)

                for dtype in (torch.int8, torch.uint8):
                    rows = torch.tensor([-2, -1, 0, 1, 2, 3] if dtype == torch.int8 else [0, 1, 2, 3, 4, 5],
                                        dtype=dtype, device=device)[:, None]
                    labels = torch.zeros(len(rows), dtype=torch.int64, device=device)
                    initial = torch.tensor([[10.], [20.]], device=device)
                    model, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.int64)
                    model._centroids[0] = 10
                    model._moved = torch.tensor([True, False], device=device)
                    result = model.final_closing_mean([rows], assignments=torch.zeros(len(rows), dtype=torch.int32), chunk_rows=2)
                    torch.testing.assert_close(result, initial, rtol=0, atol=0)
                    self.assertAlmostEqual(model.closing_stats["assignment_mse"],
                        (rows.double() - 10).square().sum().item() / len(rows), places=6)

                rows = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], device=device)
                labels = torch.zeros(4, dtype=torch.int64, device=device)
                initial = torch.tensor([[1., 0.], [-1., 0.]], device=device)
                model, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float32, spherical=True)
                model._centroids.copy_(initial)
                model._moved = torch.tensor([True, False], device=device)
                result = model.final_closing_mean([rows])
                expected = torch.nn.functional.normalize(torch.tensor([[1., 1.], [-1., 0.]], device=device), dim=1)
                torch.testing.assert_close(result, expected)
                self.assertEqual(model.closing_stats["assignment_error"], 1.)
                self.assertNotIn("assignment_mse", model.closing_stats)
                self.assertIs(model.finalize(), result)

                rejected, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float32, spherical=True)
                with self.assertRaisesRegex(ValueError, "spherical=False"):
                    rejected.final_closing_mean([rows], assignments=torch.zeros(4, dtype=torch.int32))

                rows = torch.zeros(4, 1, device=device)
                labels = torch.zeros(4, dtype=torch.int64, device=device)
                initial = torch.tensor([[10.], [20.]], device=device)
                rejected, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float32)
                with self.assertRaisesRegex(ValueError, "counts differ"):
                    rejected.final_closing_mean([rows], assignments=torch.ones(4, dtype=torch.int32))
                self.assertTrue(rejected._failed)
                self.assertIsNone(rejected.centroids)
                rejected, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float32)
                with self.assertRaisesRegex(ValueError, "cover global_n"):
                    rejected.final_closing_mean([rows[:1]])
                self.assertTrue(rejected._failed)

                # Controlled zero-observation state: omitted rows contribute no error.
                empty, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float64)
                empty._pass_counts.zero_()
                empty._centroids.copy_(initial)
                empty._moved = torch.tensor([True, False], device=device)
                result = empty.final_closing_mean([rows], assignments=torch.full((4,), -1, dtype=torch.int32))
                torch.testing.assert_close(result, initial, rtol=0, atol=0)
                self.assertEqual(empty.closing_stats["rows"], 0)
                self.assertEqual(empty.closing_stats["assignment_mse"], 0.)

                # Scoring and finalization share the same two bounded pure blocks.
                width, count = 65536, 16
                rows = torch.ones(count, width, device=device)
                labels = torch.zeros(count, dtype=torch.int64, device=device)
                initial = torch.full((count, width), 7., device=device)
                wide, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float64)
                wide._centroids[0] = 10
                wide._moved = torch.zeros(count, dtype=torch.bool, device=device)
                wide._moved[0] = True
                previews = []
                where = torch.where
                def observed_preview(*args, **kwargs):
                    result = where(*args, **kwargs)
                    if result.dtype == torch.float32 and result.ndim == 2 and result.shape[1] == width:
                        previews.append(result.cpu())
                    return result
                with mock.patch.object(torch, "where", side_effect=observed_preview), \
                     mock.patch.object(wide, "_prepare_assignment", side_effect=AssertionError("saved scan searched")):
                    result = wide.final_closing_mean([rows[:8], rows[8:]],
                        assignments=torch.zeros(count, dtype=torch.int32), chunk_rows=4)
                self.assertEqual([len(block) for block in previews], [10, 6, 16])
                self.assertTrue(torch.equal(torch.cat(previews[:2]).view(torch.int32), result.cpu().view(torch.int32)))
                self.assertTrue(torch.equal(torch.cat(previews[2:]).view(torch.int32), result.cpu().view(torch.int32)))
                self.assertTrue(result.is_contiguous())
                self.assertEqual(wide.closing_stats["assignment_mse"], 81. * width)

                # Failure after division must not publish the owned but unfinished output.
                failed, _ = _completed_closing_model(dist.group.WORLD, rows, labels, initial, 1, torch.float64)
                unchanged = failed._pass_sums.clone()
                with mock.patch.object(torch, "where", side_effect=RuntimeError("injected closing block")), \
                     self.assertRaisesRegex(RuntimeError, "injected closing block"):
                    failed.finalize()
                self.assertTrue(failed._failed)
                self.assertIsNone(failed.centroids)
                torch.testing.assert_close(failed._pass_sums, unchanged, rtol=0, atol=0)
            finally:
                dist.destroy_process_group()



if __name__ == "__main__":
    unittest.main()
