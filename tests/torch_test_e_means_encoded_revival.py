# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Small independent encoded cost/closing and replicated-statistics witnesses."""
import datetime
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from faiss.contrib.e_means import DistributedEmeans
from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer

CUDA = torch.cuda.is_available() and not torch.version.hip
CAP = torch.cuda.get_device_capability() if CUDA else None


def family(name, device):
    """One separated geometry, shared with the sharded transfer witness."""
    initial = torch.tensor([[0., 0.], [4., 0.], [8., 0.], [12., 0.]], device=device)
    rows = torch.tensor([[.25, .5], [-.25, -.5], [8.25, .5], [7.75, -.5]], device=device)
    q = None
    if name in ("int8", "uint8"):
        offset = 128 if name == "uint8" else 0
        initial = initial * 8 + offset
        rows = (rows * 8 + offset).to(torch.uint8 if offset else torch.int8)
    elif name == "fp8":
        rows = rows.to(torch.float8_e4m3fn)
    elif name == "int8_cal":
        q = Int8Quantizer(2)
        q.train(torch.cat((initial, rows)))
        rows = q.compute_codes(rows)
    elif name == "cast":
        q = CastQuantizer(2, torch.bfloat16)
    else:
        q = Fp8Quantizer(2)
        if name == "fp8_center":
            initial, rows = initial + 100, rows + 100
            q.train(torch.cat((initial, rows)))
        else:
            q.train(torch.tensor([[-16., -8.], [0., 0.], [16., 8.]], device=device).repeat(1024, 1))
    return initial, rows, q


def route_available(name, capability):
    if name in ("int8", "uint8", "int8_cal"):
        return capability in ((8, 0), (9, 0), (10, 0))
    if name == "cast":
        return capability >= (8, 0)
    return capability in ((9, 0), (10, 0), (10, 3))


class UnitAxes:
    def _initial_removal_axes(self):
        self.init_calls = getattr(self, "init_calls", 0) + 1
        axes = torch.zeros_like(self._centroids)
        axes[:, 0] = 1
        return axes


class FixedAxes(UnitAxes, QuantizedEmeans):
    pass


class FixedReplica(UnitAxes, DistributedEmeans):
    pass


class ClosingObserved(FixedAxes):
    def _choose_splits(self, rng):
        return torch.tensor([2], device=self._centroids.device), torch.tensor([0], device=self._centroids.device)

    def finalize(self):
        self.before_close = self._centroids.clone()
        self.sums_at_close, self.counts_at_close = self._pass_sums.clone(), self._pass_counts.clone()
        return super().finalize()


def replica_worker(rank, store):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=store,
                            timeout=datetime.timedelta(minutes=5))
    try:
        cap = torch.tensor(torch.cuda.get_device_capability(), device=device)
        peers = [torch.empty_like(cap) for _ in range(2)]
        dist.all_gather(peers, cap)
        assert all(tuple(p.tolist()) in ((8, 0), (9, 0), (10, 0)) for p in peers)
        initial, pattern, q = family("int8_cal", device)
        options = dict(quantizer=q, nepochs=1.5, batch_size=8, alpha0=1, seed=7,
                       assignment_backend="triton", assignment_chunk_size=2)
        replica = FixedReplica(2, 4, group=dist.group.WORLD, **options)
        reference = FixedAxes(2, 4, **options)
        caller = initial + rank * 1000
        saved = caller.clone()
        centers, labels = replica.train(pattern.repeat(6, 1), caller)
        expected, ref_labels = reference.train(pattern.repeat(12, 1), initial)
        assert replica.init_calls == reference.init_calls == 1
        assert replica.n_splits == reference.n_splits and replica.n_splits > 0
        torch.testing.assert_close(centers, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(replica.obj, reference.obj, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(replica._pass_counts, reference._pass_counts, rtol=0, atol=0)
        torch.testing.assert_close(labels, ref_labels[:len(labels)], rtol=0, atol=0)
        peers = [torch.empty_like(centers) for _ in range(2)]
        dist.all_gather(peers, centers)
        torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
        torch.testing.assert_close(caller, saved, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(CUDA and CAP >= (8, 0), "NVIDIA SM80+ required")
class TestEncodedCostIntegration(unittest.TestCase):
    def test_local_cost_moves_and_encoded_handover(self):
        for name in ("int8", "fp8", "int8_cal", "cast"):
            if not route_available(name, CAP):
                continue  # The admission table below checks the unavailable route.
            with self.subTest(name=name):
                initial, pattern, q = family(name, "cuda")
                rows = pattern.repeat(12, 1)
                saved = rows.view(torch.uint8).clone()
                codes = rows if q is None or rows.dtype == q.dtype else q.compute_codes(rows)
                decoded = codes.float() if q is None else q.decode(codes)
                model = ClosingObserved(2, 4, quantizer=q, nepochs=1, batch_size=8,
                                        alpha0=1, seed=7, assignment_backend="triton",
                                        capture_assignments=True, assignment_chunk_size=8)
                centers, labels = model.train(rows, initial)
                self.assertEqual(model.n_splits, 1)
                self.assertGreater(model.closing_handover_stats["searched"], 0)
                sums = torch.zeros_like(initial, dtype=torch.float64).index_add_(0, labels, decoded.double())
                counts = torch.bincount(labels, minlength=4)
                populated = counts > 0
                torch.testing.assert_close(centers[populated],
                                           (sums[populated] / counts[populated, None]).float(),
                                           rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(centers[~populated], model.before_close[~populated], rtol=0, atol=0)
                torch.testing.assert_close(model.sums_at_close.double(), sums, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(model.counts_at_close.long(), counts, rtol=0, atol=0)
                torch.testing.assert_close(rows.view(torch.uint8), saved, rtol=0, atol=0)

    def test_owned_removal_foreign_rejection_and_explicit_mass(self):
        name = "int8_cal" if route_available("int8_cal", CAP) else "fp8_center"
        initial, rows, q = family(name, "cuda")
        options = dict(quantizer=q, nepochs=1, batch_size=4, assignment_backend="triton")
        model, foreign = FixedAxes(2, 4, **options), FixedAxes(2, 4, **options)
        model.begin(4, initial)
        foreign.begin(4, initial)
        stats, foreign_stats = model.accumulate(rows), foreign.accumulate(rows)
        if name == "int8_cal":
            decoded = q.decode(rows)
            distances = (rows.long()[:, None] - q.compute_codes(initial).long()[None]).square().sum(2)
            nearest = distances.argsort(1, stable=True)[:, :2]
            labels = nearest[:, 0]
            residual = decoded - initial[labels]
            margin = (distances.gather(1, nearest)[:, 1] - distances.gather(1, nearest)[:, 0]).float()
            margin = torch.ldexp(margin, q._distance_exponent)
            expected = torch.zeros((4, 3), device="cuda")
            expected.index_add_(0, labels, torch.cat((residual[:, :1] * residual, margin[:, None]), 1))
            torch.testing.assert_close(stats["removal"], expected, rtol=1e-5, atol=1e-5)
            self.assertAlmostEqual(stats["objective"].item(), residual.double().square().sum().item())
        saved = [x.clone() for x in (model._centroids, model._usage, model._batch_removal)]
        with self.assertRaisesRegex(ValueError, "this model's accumulate"):
            model.update(foreign_stats)
        for actual, wanted in zip((model._centroids, model._usage, model._batch_removal), saved):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        copied = dict(stats, sums=stats["sums"].clone(), counts=stats["counts"].clone())
        self.assertIs(copied["removal"], model._batch_removal)
        model.update(copied)
        model.end_epoch()
        self.assertTrue(bool(torch.isfinite(model.finalize()).all()))
        mass = FixedAxes(2, 4, unused_threshold=.1, **options)
        mass.begin(4, initial)
        stats = mass.accumulate(rows)
        self.assertEqual(mass.init_calls, 1)
        self.assertIs(stats["removal"], mass._batch_removal)
        torch.testing.assert_close(stats["removal"][:, -1], torch.zeros(4, device="cuda"), rtol=0, atol=0)

    def test_unavailable_encoded_routes_fail_before_training(self):
        for name, backend, legal, message in (
            ("int8", "triton", route_available("int8", CAP), "SM80, SM90, or SM100"),
            ("fp8", "triton", route_available("fp8", CAP), "SM90, SM100, or SM103"),
            ("int8", "cute", CAP == (9, 0), "exactly SM90"),
        ):
            if legal:
                continue
            initial, rows, q = family(name, "cuda")
            model = QuantizedEmeans(2, 4, quantizer=q, assignment_backend=backend,
                                    nepochs=1, batch_size=4)
            with self.assertRaisesRegex(RuntimeError, message):
                model.begin(4, initial)
                model.accumulate(rows)

    @unittest.skipUnless(dist.is_nccl_available() and torch.cuda.device_count() >= 2,
                         "two GPUs/NCCL required")
    def test_replicated_calibrated_cost_and_fractional_tail(self):
        if any(torch.cuda.get_device_capability(i) not in ((8, 0), (9, 0), (10, 0)) for i in range(2)):
            self.skipTest("replicated INT8 requires two admitted devices")
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(replica_worker, args=("file://" + directory + "/group",),
                                        nprocs=2, join=True)
