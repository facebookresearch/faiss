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
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer
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
        if self.forced:
            pair = (
                torch.tensor([2], device=self._centroids.device),
                torch.tensor([0], device=self._centroids.device),
            )
            return pair if self._rank == 0 else (None, None)
        return super()._choose_sharded_removal_splits()


def key_case(rank, device, rows, centers, q, backend, winner, margin, decoded):
    model = ObservedOwner(
        rows.shape[1],
        2,
        group=dist.group.WORLD,
        quantizer=q,
        assignment_backend=backend,
        nepochs=1,
        batch_size=2,
        alpha0=1,
        assignment_chunk_size=1,
    )
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
    torch.testing.assert_close(
        model._batch_removal, expected, rtol=1e-5, atol=0
    )
    objective = (
        (2 * residual.square().sum(dtype=torch.float64)).item()
        if owned
        else 0.0
    )
    assert stats["objective"].item() == objective
    copied = dict(
        stats, sums=stats["sums"].clone(), counts=stats["counts"].clone()
    )
    saved_sums, saved_counts = copied["sums"].clone(), copied["counts"].clone()
    model.update(copied)
    torch.testing.assert_close(copied["sums"], saved_sums, rtol=0, atol=0)
    torch.testing.assert_close(copied["counts"], saved_counts, rtol=0, atol=0)
    model.end_epoch()
    result = model.finalize()
    mean = decoded[None] if owned else initial
    torch.testing.assert_close(result, mean, rtol=1e-5, atol=0)
    torch.testing.assert_close(
        rows.view(torch.uint8), saved_rows, rtol=0, atol=0
    )
    torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)


def key_cases(rank, device, cute, integers, fp8):
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        arms = ["triton"]
        if dtype == torch.float32 or "out_dtype" in (torch.addmm.__doc__ or ""):
            arms += ["native"]
        if cute and dtype == torch.bfloat16:
            arms += ["cute"]
        for backend in arms:
            rows = torch.tensor([[8192.0, 0.0]], dtype=dtype, device=device)
            centers = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device)
            key_case(
                rank,
                device,
                rows,
                centers,
                None,
                backend,
                1,
                1.0,
                rows[0].float(),
            )
    for dtype in (torch.float16, torch.bfloat16):
        q = CastQuantizer(2, dtype, center=True)
        q.train(torch.tensor([[-1.0, 3.0], [1.0, 5.0]], device=device))
        for backend in ["triton"] + (
            ["cute"] if cute and dtype == torch.bfloat16 else []
        ):
            rows = torch.tensor([[8192.0, 4.0]], device=device)
            centers = torch.tensor([[0.0, 5.0], [0.0, 4.0]], device=device)
            key_case(rank, device, rows, centers, q, backend, 1, 1.0, rows[0])
    for name in (("int8", "uint8", "calibrated") if integers else ()):
        q = None
        codes = torch.full((1, 512), -128, dtype=torch.int8, device=device)
        codes[0, -1] = 0
        centers = torch.full((2, 512), 127.0, device=device)
        centers[:, -1] = torch.tensor([1.0, 0.0], device=device)
        if name == "uint8":
            codes = (codes.to(torch.int16) + 128).to(torch.uint8)
            centers += 128
        if name == "calibrated":
            q = Int8Quantizer(512)
            q.train(
                torch.stack(
                    (
                        -torch.ones(512, device=device),
                        torch.ones(512, device=device),
                    )
                )
            )
            assert q.exponent == 6 and bool((q.shift == 0).all())
            centers /= 64
        for backend in ["triton"] + (["cute"] if cute else []):
            decoded = codes[0].float() / (64 if q is not None else 1)
            key_case(
                rank,
                device,
                codes,
                centers,
                q,
                backend,
                1,
                1.0 / (4096 if q is not None else 1),
                decoded,
            )
    if fp8:
        for backend in ["triton"] + (["cute"] if cute else []):
            centers = torch.full((2, 128), 448.0, device=device)
            centers[:, -1] = torch.tensor([0.5, 0.25], device=device)
            rows = torch.full((1, 128), 448.0, device=device)
            rows[:, -1] = 0
            key_case(
                rank,
                device,
                rows.to(torch.float8_e4m3fn),
                centers,
                None,
                backend,
                1,
                0.1875,
                rows[0],
            )
    # FP8's completed hierarchical rule differs from union/ranking-key choice.
    for tails, winner, margin in (
        (
            ([0.25, 0.125, 0.1875, 0.5], 2, 0.02734375),
            ([0.25, 0.125, 0.5, 0.75], 0, -0.046875),
        )
        if fp8
        else ()
    ):
        for backend in ["triton"] + (["cute"] if cute else []):
            centers = torch.full((4, 128), 448.0, device=device)
            centers[:, -1] = torch.tensor(tails, device=device)
            rows = torch.full((2, 128), 448.0, device=device)
            rows[:, -1] = 0
            model = MatrixOwner(
                128,
                4,
                group=dist.group.WORLD,
                quantizer=None,
                assignment_backend=backend,
                nepochs=1,
                batch_size=4,
                alpha0=1,
                assignment_chunk_size=2,
            )
            model.begin(4, centers[2 * rank : 2 * rank + 2].clone())
            stats = model.accumulate(rows.to(torch.float8_e4m3fn))
            assert stats["labels"].tolist() == [winner, winner]
            expected = torch.zeros_like(model._batch_removal)
            if winner // 2 == rank:
                expected[winner % 2, -1] = 4 * margin
            torch.testing.assert_close(
                model._batch_removal, expected, rtol=0, atol=0
            )
            model.update(stats)
            model.end_epoch()
            model.finalize()
    # A harmless observer cannot restore rounded keys; an incapable peer
    # instead fails collectively before any candidate MIN.
    cls = ObservedOwner if rank == 0 else FixedOwner
    model = cls(
        2,
        2,
        group=dist.group.WORLD,
        quantizer=None,
        assignment_backend="triton",
        nepochs=1,
        batch_size=2,
    )
    model.seen = []
    model.begin(2, torch.tensor([[0.0, 1.0 - rank]], device=device))
    with mock.patch.object(
        dist, "all_reduce", side_effect=AssertionError("candidate MIN reached")
    ):
        with unittest.TestCase().assertRaisesRegex(
            TypeError, "incompatible.*1"
        ):
            model.accumulate(torch.tensor([[8192.0, 0.0]], device=device))
    assert model._failed and model.seen == []
    with unittest.TestCase().assertRaises(RuntimeError):
        model.update({})


def transfer_cases(rank, device, cute, integers, fp8):
    for name in (
        "int8",
        "uint8",
        "fp8",
        "int8_cal",
        "fp8_plain",
        "fp8_center",
        "f32",
        "f16",
        "bf16",
    ):
        if name in ("int8", "uint8", "int8_cal") and not integers:
            continue
        if name in ("fp8", "fp8_plain", "fp8_center") and not fp8:
            continue
        if name in ("f32", "f16", "bf16"):
            centers = torch.tensor(
                [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]], device=device
            )
            pattern = torch.tensor(
                [[0.25, 0.5], [-0.25, -0.5], [8.25, 0.5], [7.75, -0.5]],
                device=device,
            )
            pattern = pattern.to(
                {
                    "f32": torch.float32,
                    "f16": torch.float16,
                    "bf16": torch.bfloat16,
                }[name]
            )
            q = None
        else:
            centers, pattern, q = family(name, device)
        for backend in ["triton"] + (
            ["cute"] if cute and name not in ("f32", "f16") else []
        ):
            for forced in (False, True):
                rows = pattern.repeat(6, 1)
                saved = rows.view(torch.uint8).clone()
                model = TransferOwner(
                    2,
                    4,
                    group=dist.group.WORLD,
                    quantizer=q,
                    assignment_backend=backend,
                    nepochs=1,
                    batch_size=8,
                    alpha0=1,
                    assignment_chunk_size=4,
                )
                model.forced = forced
                model.begin(48, centers[2 * rank : 2 * rank + 2].clone())
                labels = []
                for start in range(0, 24, 4):
                    stats = model.accumulate(rows[start : start + 4])
                    labels.append(stats["labels"].cpu().clone())
                    model.update(stats)
                    if forced and model._step == 5:
                        assert model.n_splits == 1
                        assert model._pass_counts[0].item() == 0
                        state = torch.cat(
                            (model._removal_stats[0], model._axes[0])
                        )[None]
                        peers = model._gather_local(state)
                        torch.testing.assert_close(
                            peers[0], peers[1], rtol=0, atol=0
                        )
                        positions = model._gather_local(model._centroids[:1])
                        if q is not None:
                            represented = q.compute_codes(positions).float()
                        elif rows.dtype in (torch.int8, torch.uint8):
                            lo, hi = (
                                (-128, 127)
                                if rows.dtype == torch.int8
                                else (0, 255)
                            )
                            represented = (
                                positions.round()
                                .clamp(lo, hi)
                                .to(rows.dtype)
                                .float()
                            )
                        else:
                            represented = positions.to(rows.dtype).float()
                        assert not bool(
                            (represented[0] == represented[1]).all()
                        )
                assert model.n_splits > 0
                model.end_epoch()
                labels = torch.cat(labels)
                codes = (
                    rows
                    if q is None or rows.dtype == q.dtype
                    else q.compute_codes(rows)
                )
                decoded = codes.float() if q is None else q.decode(codes)
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
                model._assign_rows = mock.Mock(
                    side_effect=AssertionError("saved closing searched")
                )
                result = model.final_closing_mean(
                    [rows[:13], rows[13:]], assignments=labels, chunk_rows=4
                )
                full = model._gather_local(result)
                direct = (
                    (decoded.double() - full[labels.to(device)].double())
                    .square()
                    .sum()
                )
                dist.all_reduce(direct)
                unittest.TestCase().assertAlmostEqual(
                    model.closing_stats["assignment_mse"],
                    direct.item() / 48,
                    delta=1e-5 * max(1.0, direct.item() / 48),
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    rows.view(torch.uint8), saved, rtol=0, atol=0
                )


class LegacyIVF(UnitAxes, EmeansShardedCentroidsIVF):
    def _assign_rows(
        self, rows, centers, norm=None, first=None, labels=None, second=None
    ):
        raise AssertionError("IVF search must not invoke the new scored hook")

    def _local_assignment_batch(
        self, rows, prepared, routing=None, second_out=None
    ):
        self.second_dtype = None if second_out is None else second_out.dtype
        result = super()._local_assignment_batch(
            rows, prepared, routing, second_out
        )
        if second_out is not None:
            self.local_second_absent = bool(torch.isposinf(second_out).all())
        return result


def boundary_worker(rank, rendezvous, case):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=datetime.timedelta(minutes=5),
    )
    try:
        capability = torch.tensor(
            torch.cuda.get_device_capability(), device=device
        )
        peers = [torch.empty_like(capability) for _ in range(2)]
        dist.all_gather(peers, capability)
        capabilities = [tuple(p.tolist()) for p in peers]
        cute = all(p == (9, 0) for p in capabilities)
        integers = all(p in ((8, 0), (9, 0), (10, 0)) for p in capabilities)
        fp8 = all(p in ((9, 0), (10, 0), (10, 3)) for p in capabilities)
        if case == "matrix":
            backends = ["triton"]
            if cute:
                backends += ["cute"]
            for backend in backends:
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

                q._matrix = matrix  # Actual instance metadata-helper override.
                model = MatrixOwner(
                    2,
                    2,
                    group=dist.group.WORLD,
                    quantizer=q,
                    assignment_backend=backend,
                    nepochs=1,
                    batch_size=2,
                    alpha0=1,
                    assignment_chunk_size=1,
                )
                model.begin(
                    2, torch.tensor([[10.0 * rank, 0.0]], device=device)
                )
                rows = torch.tensor(
                    [[1.0, 0.0]], dtype=torch.bfloat16, device=device
                )
                stats = model.accumulate(rows)
                assert seen == [
                    1,
                    1,
                ]  # Decode source chunks, including unowned ones.
                owned = rank == 0
                sums = torch.tensor(
                    [[4.0, 0.0] if owned else [0.0, 0.0]], device=device
                )
                counts = torch.tensor([2.0 if owned else 0.0], device=device)
                removal = torch.tensor(
                    [[8.0, 0.0, 160.0] if owned else [0.0, 0.0, 0.0]],
                    device=device,
                )
                torch.testing.assert_close(stats["sums"], sums, rtol=0, atol=0)
                torch.testing.assert_close(
                    stats["counts"], counts, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    model._batch_removal, removal, rtol=0, atol=0
                )
                assert stats["objective"].item() == (8.0 if owned else 0.0)
                model.update(stats)
                model.end_epoch()
                model.finalize()

            class InheritedCast(CastQuantizer):
                pass

            from faiss.contrib.e_means import ivf_kernels

            for backend in backends:
                q = InheritedCast(2, torch.bfloat16)
                model = MatrixOwner(
                    2,
                    2,
                    group=dist.group.WORLD,
                    quantizer=q,
                    assignment_backend=backend,
                    nepochs=1,
                    batch_size=2,
                    alpha0=1,
                    assignment_chunk_size=1,
                )
                model.begin(
                    2, torch.tensor([[10.0 * rank, 0.0]], device=device)
                )
                used, original = [], ivf_kernels.accumulate_owned

                def accumulate(*args, _original=original, _used=used, **kwargs):
                    value = _original(*args, **kwargs)
                    _used.append(value)
                    return value

                with mock.patch.object(
                    ivf_kernels, "accumulate_owned", side_effect=accumulate
                ):
                    stats = model.accumulate(
                        torch.tensor(
                            [[1.0, 0.0]], dtype=torch.bfloat16, device=device
                        )
                    )
                assert used == [True]
                expected = torch.tensor(
                    [[2.0, 0.0] if rank == 0 else [0.0, 0.0]], device=device
                )
                torch.testing.assert_close(
                    stats["sums"], expected, rtol=0, atol=0
                )
                model.update(stats)
                model.end_epoch()
                model.finalize()
        elif case == "ivf":
            # Real old IVF BF16 search, native byte rows and local K1: its
            # absent local alternatives must remain floating +inf.
            for dtype in (torch.int8, torch.uint8):
                model = LegacyIVF(
                    2,
                    2,
                    group=dist.group.WORLD,
                    quantizer=None,
                    nlist=1,
                    nprobe=1,
                    coarse_niter=1,
                    nepochs=1,
                    batch_size=2,
                    alpha0=1,
                    assignment_chunk_size=1,
                )
                model.begin(
                    2, torch.tensor([[0.5 + 0.25 * rank, 0.0]], device=device)
                )
                stats = model.accumulate(
                    torch.zeros((1, 2), dtype=dtype, device=device)
                )
                assert (
                    model.second_dtype == torch.float32
                    and model.local_second_absent
                )
                assert (
                    stats["sums"].dtype == stats["counts"].dtype == torch.int64
                )
                expected = torch.tensor(
                    [[0.5, 0.0, 0.625] if rank == 0 else [0.0, 0.0, 0.0]],
                    device=device,
                )
                torch.testing.assert_close(
                    model._batch_removal, expected, rtol=0, atol=0
                )
                assert stats["counts"].item() == (2 if rank == 0 else 0)
                assert stats["objective"].item() == (0.5 if rank == 0 else 0.0)
                model.update(stats)
                model.end_epoch()
                model.finalize()
        elif case == "keys":
            key_cases(rank, device, cute, integers, fp8)
        else:
            transfer_cases(rank, device, cute, integers, fp8)
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
    def test_owner_raw_key_collisions_k1_hierarchy_and_peer_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                boundary_worker,
                args=("file://" + directory + "/group", "keys"),
                nprocs=2,
                join=True,
            )

    def test_owner_cross_shard_codec_transfers_and_actual_saved_closing(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                boundary_worker,
                args=("file://" + directory + "/group", "transfers"),
                nprocs=2,
                join=True,
            )

    def test_genuine_cast_matrix_values_and_source_cardinality(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                boundary_worker,
                args=("file://" + directory + "/group", "matrix"),
                nprocs=2,
                join=True,
            )

    def test_legacy_ivf_native_byte_floating_second_local_k1(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                boundary_worker,
                args=("file://" + directory + "/group", "ivf"),
                nprocs=2,
                join=True,
            )
