# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded genuine-decoder and parked-IVF inherited score-domain controls."""
import datetime
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids
from faiss.contrib.e_means.sharded_centroids_ivf import (
    EmeansShardedCentroidsIVF,
)
from torch_test_e_means_encoded_revival import family


class UnitAxes:
    def _initial_removal_axes(self):
        axes = torch.zeros_like(self._centroids)
        axes[:, 0] = 1
        return axes


class MatrixOwner(UnitAxes, EmeansShardedCentroids):
    pass


class ObservedOwner(UnitAxes, EmeansShardedCentroids):
    def _assign_rows(self, *args, **kwargs):
        self.seen.append((len(args), dict(kwargs), args[5] is None))
        return super()._assign_rows(*args, **kwargs)


class FixedOwner(ObservedOwner):
    def _assign_rows(
        self, rows, centers, norm=None, first=None, labels=None, second=None
    ):
        raise AssertionError("incompatible peer must fail in preflight")


class TransferOwner(UnitAxes, EmeansShardedCentroids):
    def _choose_sharded_removal_splits(self):
        pair = (torch.tensor([2], device=self._centroids.device),
                torch.tensor([0], device=self._centroids.device))
        return pair if self._rank == 0 else (None, None)


def key_case(rank, device, rows, centers, q, backend, winner, margin, decoded):
    model = ObservedOwner(rows.shape[1], 2, group=dist.group.WORLD, quantizer=q,
        assignment_backend=backend, nepochs=1, batch_size=2, alpha0=1, assignment_chunk_size=1)
    model.seen = []
    initial = centers[rank : rank + 1].clone()
    saved_rows, saved_initial = rows.view(torch.uint8).clone(), initial.clone()
    model.begin(2, initial)
    with mock.patch.object(model, '_gather_local', wraps=model._gather_local) as gathers:
        stats = model.accumulate(rows)
        if model._input_dtype in (torch.float32, torch.float16, torch.bfloat16):
            first = gathers.call_count
            assert model._score_hooks_checked
            model._prepare_assignment()
            assert gathers.call_count == first  # No repeated hook collective.
    assert stats["n"] == 2 and stats["labels"].tolist() == [winner]
    integer = rows.dtype in (torch.int8, torch.uint8)
    scored = model._input_dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    )
    assert model.seen == [(6, {"scores": True} if scored else {}, True)] * 2
    storage = torch.int64 if integer and q is None else torch.float32
    assert stats["sums"].dtype == stats["counts"].dtype == storage
    owned = rank == winner
    sums = (
        (2 * decoded).to(storage)[None]
        if owned
        else torch.zeros_like(stats["sums"])
    )
    torch.testing.assert_close(stats["sums"], sums, rtol=0, atol=0)
    assert stats["counts"].item() == (2 if owned else 0)
    residual = decoded.float() - centers[winner]
    expected = model._batch_removal.new_zeros((1, rows.shape[1] + 1))
    if owned:
        expected[0, :-1] = 2 * (residual[0] * residual)
        expected[0, -1] = 2 * margin
    torch.testing.assert_close(model._batch_removal, expected, rtol=1e-05, atol=0)
    objective = (
        (2 * residual.square().sum(dtype=torch.float64)).item()
        if owned
        else 0.0
    )
    assert stats["objective"].item() == objective
    copied = dict(stats, sums=stats['sums'].clone(), counts=stats['counts'].clone())
    saved_sums, saved_counts = copied["sums"].clone(), copied["counts"].clone()
    model.update(copied)
    torch.testing.assert_close(copied["sums"], saved_sums, rtol=0, atol=0)
    torch.testing.assert_close(copied["counts"], saved_counts, rtol=0, atol=0)
    model.end_epoch()
    result = model.finalize()
    mean = decoded[None] if owned else initial
    torch.testing.assert_close(result, mean, rtol=1e-5, atol=0)
    torch.testing.assert_close(rows.view(torch.uint8), saved_rows, rtol=0, atol=0)
    torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)


def key_cases(rank, device, integers, fp8):
    rows = torch.tensor([[8192.0, 0.0]], device=device)
    centers = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device)
    key_case(rank, device, rows, centers, None, "triton", 1, 1.0, rows[0])
    if integers:
        rows = torch.full((1, 512), -128, dtype=torch.int8, device=device)
        rows[0, -1] = 0
        centers = torch.full((2, 512), 127.0, device=device)
        centers[:, -1] = torch.tensor([1.0, 0.0], device=device)
        key_case(rank, device, rows, centers, None, "triton", 1, 1.0, rows[0].float())
    if fp8:
        centers = torch.full((4, 128), 448.0, device=device)
        centers[:, -1] = torch.tensor([0.25, 0.125, 0.5, 0.75], device=device)
        rows = torch.full((2, 128), 448.0, device=device)
        rows[:, -1] = 0
        model = MatrixOwner(128, 4, group=dist.group.WORLD, assignment_backend='triton', nepochs=1,
            batch_size=4, alpha0=1, assignment_chunk_size=2)
        model.begin(4, centers[2 * rank:2 * rank + 2].clone())
        stats = model.accumulate(rows.to(torch.float8_e4m3fn))
        assert stats["labels"].tolist() == [0, 0]
        expected = torch.zeros_like(model._batch_removal)
        if rank == 0:
            expected[0, -1] = 4 * -0.046875
        torch.testing.assert_close(model._batch_removal, expected, rtol=0, atol=0)
        model.update(stats)
        model.end_epoch()
        model.finalize()
    # A harmless observer cannot restore rounded keys; an incapable peer
    # instead fails collectively before any candidate MIN.
    cls = ObservedOwner if rank == 0 else FixedOwner
    model = cls(2, 2, group=dist.group.WORLD, quantizer=None, assignment_backend='triton',
        nepochs=1, batch_size=2)
    model.seen = []
    model.begin(2, torch.tensor([[0.0, 1.0 - rank]], device=device))
    with mock.patch.object(dist, 'all_reduce', side_effect=AssertionError('candidate MIN reached')):
        with unittest.TestCase().assertRaisesRegex(TypeError, 'incompatible.*1'):
            model.accumulate(torch.tensor([[8192.0, 0.0]], device=device))
    assert model._failed and model.seen == []
    with unittest.TestCase().assertRaises(RuntimeError):
        model.update({})


def transfer_case(rank, device):
    centers, pattern, q = family("int8_cal", device)
    rows = pattern.repeat(6, 1)
    saved = rows.view(torch.uint8).clone()
    model = TransferOwner(2, 4, group=dist.group.WORLD, quantizer=q, assignment_backend='triton',
        nepochs=1, batch_size=8, alpha0=1, assignment_chunk_size=4)
    model.begin(48, centers[2 * rank : 2 * rank + 2].clone())
    labels = []
    for start in range(0, 24, 4):
        stats = model.accumulate(rows[start : start + 4])
        labels.append(stats["labels"].cpu().clone())
        model.update(stats)
        if model._step == 5:
            assert model.n_splits == 1
            assert model._pass_counts[0].item() == 0
            state = torch.cat((model._removal_stats[0], model._axes[0]))[None]
            peers = model._gather_local(state)
            torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
            positions = model._gather_local(model._centroids[:1])
            represented = q.compute_codes(positions).float()
            assert not bool((represented[0] == represented[1]).all())
    assert model.n_splits > 0
    model.end_epoch()
    labels = torch.cat(labels)
    decoded = q.decode(rows)
    current = model._centroids.clone()
    all_labels = model._gather_local(labels.to(device))
    all_decoded = model._gather_local(decoded)
    expected = current.clone()
    for j in range(model.k):
        selected = all_decoded[
            all_labels == j + model.centroid_begin
        ]
        if len(selected):
            expected[j] = selected.double().mean(0).float()
    # Explicit saved closing uses every supplied label without
    # search, independently of ordinary post-split finalization.
    model._assign_rows = mock.Mock(side_effect=AssertionError('saved closing searched'))
    result = model.final_closing_mean([rows[:13], rows[13:]], assignments=labels, chunk_rows=4)
    full = model._gather_local(result)
    direct = (
        (decoded.double() - full[labels.to(device)].double()).square().sum()
    )
    dist.all_reduce(direct)
    unittest.TestCase().assertAlmostEqual(model.closing_stats['assignment_mse'],
        direct.item() / 48, delta=1e-05 * max(1.0, direct.item() / 48))
    torch.testing.assert_close(result, expected, rtol=1e-05, atol=1e-05)
    torch.testing.assert_close(rows.view(torch.uint8), saved, rtol=0, atol=0)


class LegacyIVF(UnitAxes, EmeansShardedCentroidsIVF):
    def _assign_rows(
        self, rows, centers, norm=None, first=None, labels=None, second=None
    ):
        raise AssertionError("IVF search must not invoke the new scored hook")

    def _local_assignment_batch(
        self, rows, prepared, routing=None, second_out=None
    ):
        self.second_dtype = None if second_out is None else second_out.dtype
        result = super()._local_assignment_batch(rows, prepared, routing, second_out)
        if second_out is not None:
            self.local_second_absent = bool(torch.isposinf(second_out).all())
        return result


def boundary_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group('nccl', rank=rank, world_size=2, init_method=rendezvous,
        timeout=datetime.timedelta(minutes=5))
    try:
        capability = torch.tensor(torch.cuda.get_device_capability(), device=device)
        peers = [torch.empty_like(capability) for _ in range(2)]
        dist.all_gather(peers, capability)
        capabilities = [tuple(p.tolist()) for p in peers]
        integers = all(p in ((8, 0), (9, 0), (10, 0)) for p in capabilities)
        fp8 = all(p in ((9, 0), (10, 0), (10, 3)) for p in capabilities)
        key_cases(rank, device, integers, fp8)
        if integers:
            transfer_case(rank, device)
        backend = "triton"
        q = CastQuantizer(2, torch.bfloat16)
        original = q._matrix
        seen = []

        def matrix(
            rows, *, fitting=False, _original=original, _q=q, _seen=seen
        ):
            value = _original(rows, fitting=fitting)
            if value.dtype == _q.dtype:
                _seen.append(len(value))
                return 2 * value
            return value

        q._matrix = matrix  # Witness the fused owned-statistics decoder guard.
        model = MatrixOwner(2, 2, group=dist.group.WORLD, quantizer=q, assignment_backend=backend,
            nepochs=1, batch_size=2, alpha0=1, assignment_chunk_size=1)
        model.begin(2, torch.tensor([[10.0 * rank, 0.0]], device=device))
        rows = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16, device=device)
        stats = model.accumulate(rows)
        assert seen == [
            1,
            1,
        ]  # Decode source chunks, including unowned ones.
        owned = rank == 0
        sums = torch.tensor([[4.0, 0.0] if owned else [0.0, 0.0]], device=device)
        counts = torch.tensor([2.0 if owned else 0.0], device=device)
        removal = torch.tensor([[8.0, 0.0, 160.0] if owned else [0.0, 0.0, 0.0]], device=device)
        torch.testing.assert_close(stats["sums"], sums, rtol=0, atol=0)
        torch.testing.assert_close(stats['counts'], counts, rtol=0, atol=0)
        torch.testing.assert_close(model._batch_removal, removal, rtol=0, atol=0)
        assert stats["objective"].item() == (8.0 if owned else 0.0)
        model.update(stats)
        model.end_epoch()
        model.finalize()
        dtype = torch.int8
        model = LegacyIVF(2, 2, group=dist.group.WORLD, quantizer=None, nlist=1, nprobe=1,
            coarse_niter=1, nepochs=1, batch_size=2, alpha0=1, assignment_chunk_size=1)
        model.begin(2, torch.tensor([[0.5 + 0.25 * rank, 0.0]], device=device))
        stats = model.accumulate(torch.zeros((1, 2), dtype=dtype, device=device))
        assert (
            model.second_dtype == torch.float32
            and model.local_second_absent
        )
        assert (
            stats["sums"].dtype == stats["counts"].dtype == torch.int64
        )
        expected = torch.tensor([[0.5, 0.0, 0.625] if rank == 0 else [0.0, 0.0, 0.0]],
            device=device)
        torch.testing.assert_close(model._batch_removal, expected, rtol=0, atol=0)
        assert stats["counts"].item() == (2 if rank == 0 else 0)
        assert stats["objective"].item() == (0.5 if rank == 0 else 0.0)
        model.update(stats)
        model.end_epoch()
        model.finalize()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.is_available()
    and not torch.version.hip
    and torch.cuda.device_count() >= 2
    and dist.is_nccl_available()
    and all(torch.cuda.get_device_capability(i) >= (8, 0) for i in range(2)),
    "two SM80+ CUDA/NCCL devices required",
)
class TestShardedCostBoundaries(unittest.TestCase):
    def test_sharded_keys_codec_transfer_decoder_and_legacy_ivf(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(boundary_worker,
                args=("file://" + directory + "/group",), nprocs=2, join=True)
