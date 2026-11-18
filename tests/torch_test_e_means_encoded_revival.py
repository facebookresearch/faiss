# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.
"""Small independent typed-statistics, encoded closing and replica-init laws."""
import datetime
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from faiss.contrib.e_means import DistributedEmeans
from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer

CUDA = torch.cuda.is_available() and not torch.version.hip and torch.cuda.get_device_capability() == (9, 0)
FAMILIES = ('int8', 'uint8', 'fp8', 'int8_cal', 'fp8_plain', 'fp8_center')


class FixedAxes(QuantizedEmeans):
    def _initial_removal_axes(self):
        self.init_calls = getattr(self, 'init_calls', 0) + 1
        axes = torch.zeros_like(self._centroids)
        axes[:, 0] = 1
        return axes


class FixedReplica(DistributedEmeans):
    def _initial_removal_axes(self):
        self.init_calls = getattr(self, 'init_calls', 0) + 1
        axes = torch.zeros_like(self._centroids)
        axes[:, 0] = 1
        return axes


class ClosingObserved(FixedAxes):
    def _choose_splits(self, rng):
        self.choose_calls = getattr(self, 'choose_calls', 0) + 1
        return torch.tensor([2], device=self._centroids.device), torch.tensor([0], device=self._centroids.device)

    def finalize(self):
        self.before_close = self._centroids.clone()
        self.moved_at_close = self._moved.clone()
        return super().finalize()


def family(name, device):
    initial = torch.tensor([[0., 0.], [4., 0.], [8., 0.], [12., 0.]], device=device)
    rows = torch.tensor([[.25, .5], [-.25, -.5], [8.25, .5], [7.75, -.5]], device=device)
    q = None
    if name in ('int8', 'uint8'):
        offset = 128 if name == 'uint8' else 0
        initial = initial * 8 + offset
        rows = (rows * 8 + offset).to(torch.uint8 if offset else torch.int8)
    elif name == 'fp8':
        rows = rows.to(torch.float8_e4m3fn)
    elif name == 'int8_cal':
        q = Int8Quantizer(2)
        q.train(torch.cat((initial, rows)))
        rows = q.compute_codes(rows)
    else:
        q = Fp8Quantizer(2)
        if name == 'fp8_center':
            initial, rows = initial + 100, rows + 100
            q.train(torch.cat((initial, rows)))
            assert q.shift is not None
        else:
            q.train(torch.tensor([[-16., -8.], [0., 0.], [16., 8.]], device=device).repeat(1024, 1))
            assert q.shift is None
    return initial, rows, q


def represented(rows, initial, q):
    codes = rows if q is None or rows.dtype == q.dtype else q.compute_codes(rows)
    decoded = codes.float() if q is None else q.decode(codes)
    if q is not None:
        centers = q.compute_codes(initial).float()
    elif rows.dtype == torch.int8:
        centers = initial.round().clamp(-128, 127)
    elif rows.dtype == torch.uint8:
        centers = initial.round().clamp(0, 255)
    else:
        centers = initial.to(rows.dtype).float()
    metric = (codes.float().double()[:, None] - centers.double()[None]).square().sum(2)
    order = metric.argsort(dim=1, stable=True)
    labels = order[:, 0]
    gap = (metric.gather(1, order[:, 1:2]) - metric.gather(1, order[:, :1])).flatten().float()
    if q is not None:
        gap = torch.ldexp(gap, q._distance_exponent)
    return decoded, labels, gap


def raw_oracle(rows, current, axes, q, native_integer):
    values, labels, margin = represented(rows, current, q)
    dtype = torch.int64 if native_integer else torch.float32
    sums = torch.zeros_like(current, dtype=dtype).index_add_(0, labels, values.to(dtype))
    counts = torch.bincount(labels, minlength=len(current)).to(dtype)
    residual = values.float() - current[labels]
    projected = (residual * axes[labels]).sum(1, keepdim=True)
    removal = torch.zeros((len(current), current.shape[1] + 1), device=current.device)
    removal.index_add_(0, labels, torch.cat((projected * residual, margin[:, None]), 1))
    return sums, counts, removal


def same_choice(model):
    # Compare actual selectors on the same populated, separated first-batch
    # state; neither selector applies a split. No exact FP8 reranking oracle.
    assert model._device_choice_eligible() and not model._device_check()
    saved = [value.clone() for value in (model._centroids, model._usage, model._removal_stats, model._axes)]
    device = model._choose_splits(model._split_rng)
    with mock.patch.object(model, '_device_choice_eligible', return_value=False):
        host = model._choose_splits(model._split_rng)
    assert device[0] is not None and host[0] is not None
    for actual, expected in zip(device, host):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip((model._centroids, model._usage, model._removal_stats, model._axes), saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def replica_case(name, backend, rank, device, transport='sparse'):
    initial, data, q = family(name, device)
    options = dict(quantizer=q, assignment_backend=backend, nepochs=2.5, batch_size=8,
                   alpha0=1, seed=7, assignment_chunk_size=2)
    model = FixedReplica(2, 4, group=dist.group.WORLD, transport=transport, **options)
    reference = FixedAxes(2, 4, **options)
    caller = initial + rank * 1000
    saved_caller = caller.clone()
    # The fifth check is active; final full/fractional tails remain divisible
    # across two ranks. A nonzero automatic root-applied split is required.
    model.begin(44, caller)
    reference.begin(44, initial.clone())
    forbidden = ('broadcast', 'all_reduce', 'all_gather', 'all_gather_object')
    model.accumulate(data[:0])
    reference.accumulate(data[:0])
    assert model._removal_stats is not None and model.init_calls == reference.init_calls == 1
    for epoch_rows in model.recipe['epoch_rows']:
        done = 0
        while done < epoch_rows:
            logical = min(8, epoch_rows - done)
            local_n = logical // 2
            local = data[(torch.arange(local_n, device=device) + rank * local_n) % len(data)]
            whole = data[torch.arange(logical, device=device) % len(data)]
            before, axes = model.current_centroids().clone(), model._axes.clone()
            # The initial separated dyadic geometry has exact first/second
            # ordering. Later trajectories compare the existing local engine,
            # without imposing exact ranking on approximate FP8 selection.
            expected = raw_oracle(local, before, axes, q, name in ('int8', 'uint8')) if model._step == 0 else None
            saved_local = local.view(torch.uint8).clone()
            with mock.patch.multiple(dist, **{method: mock.Mock(side_effect=AssertionError('partial communication'))
                                             for method in forbidden}):
                partial = model.accumulate(local[:1])
                empty = model.accumulate(local[:0])
                assert partial['sums'] is None and empty['sums'] is None
                assert partial['removal'] is empty['removal'] is model._batch_removal
                stats = model.accumulate(local[1:]) if local_n > 1 else model.accumulate(local[:0])
            assert stats['removal'] is model._batch_removal
            dtype = torch.int64 if name in ('int8', 'uint8') else torch.float32
            assert stats['sums'].dtype == stats['counts'].dtype == dtype
            if expected is not None:
                torch.testing.assert_close(stats['sums'], expected[0], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(stats['counts'], expected[1], rtol=0, atol=0)
                torch.testing.assert_close(model._batch_removal, expected[2], rtol=1e-5, atol=1e-5)
            assert model._batch_removal.dtype == torch.float32
            copied = dict(stats, sums=stats['sums'].clone(), counts=stats['counts'].clone())
            saved_sums, saved_counts = copied['sums'].clone(), copied['counts'].clone()
            ref_stats = reference.accumulate(whole)
            model.update(copied)
            reference.update(ref_stats)
            if model._step == 1:
                same_choice(model)
                same_choice(reference)
            torch.testing.assert_close(copied['sums'], saved_sums, rtol=0, atol=0)
            torch.testing.assert_close(copied['counts'], saved_counts, rtol=0, atol=0)
            torch.testing.assert_close(local.view(torch.uint8), saved_local, rtol=0, atol=0)
            for key in ('_centroids', '_usage', '_removal_stats', '_axes', '_moved', '_pass_counts'):
                torch.testing.assert_close(getattr(model, key), getattr(reference, key), rtol=1e-5, atol=1e-5)
            assert model.n_splits == reference.n_splits
            peers = [torch.empty_like(model._centroids) for _ in range(2)]
            dist.all_gather(peers, model._centroids)
            torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
            done += logical
        model.end_epoch()
        reference.end_epoch()
    assert model.n_splits == reference.n_splits and model.n_splits > 0
    torch.testing.assert_close(model.finalize(), reference.finalize(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model.obj, reference.obj, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(caller, saved_caller, rtol=0, atol=0)


def worker(rank, store):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', rank=rank, world_size=2, init_method=store,
                            timeout=datetime.timedelta(minutes=5))
    try:
        for name in FAMILIES:
            for backend in ('triton', 'cute'):
                replica_case(name, backend, rank, device)
                if name in ('int8', 'fp8'):
                    replica_case(name, backend, rank, device, transport='dense')
        # Actual stock common-axis law for known calibrated dtype before first rows.
        for name in ('int8_cal', 'fp8_plain', 'fp8_center'):
            initial, data, q = family(name, device)
            m = DistributedEmeans(2, 4, group=dist.group.WORLD, quantizer=q, seed=7 + rank,
                                  assignment_backend='triton', batch_size=8, nepochs=1)
            m.begin(24, initial)
            rng = np.random.Generator(np.random.PCG64(7))
            raw = rng.standard_normal((4, 2))
            expected = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype('float32')
            torch.testing.assert_close(m._axes.cpu(), torch.from_numpy(expected), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(CUDA, 'current encoded CuTe/Triton SM90 routes required')
class TestEncodedCostIntegration(unittest.TestCase):
    @unittest.skipUnless(dist.is_nccl_available() and torch.cuda.device_count() >= 2, 'two GPUs/NCCL required')
    def test_replicated_typed_decoded_contributions_seed_and_fractional_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(worker, args=('file://' + directory + '/group',), nprocs=2, join=True)

    def test_local_codec_moves_and_actual_encoded_handover(self):
        device = torch.device('cuda')
        for name in ('int8', 'uint8', 'fp8', 'int8_cal', 'fp8_plain', 'fp8_center'):
            for backend in ('triton', 'cute'):
                with self.subTest(name=name, backend=backend):
                    initial, pattern, q = family(name, device)
                    rows = pattern.repeat(12, 1)
                    codes = rows if q is None or rows.dtype == q.dtype else q.compute_codes(rows)
                    decoded = codes.float() if q is None else q.decode(codes)
                    saved_rows, saved_initial = rows.view(torch.uint8).clone(), initial.clone()
                    model = ClosingObserved(2, 4, quantizer=q, assignment_backend=backend, nepochs=1,
                                            batch_size=8, alpha0=1, seed=7, capture_assignments=True,
                                            assignment_chunk_size=8)
                    centers, labels = model.train(rows, initial)
                    assert model.n_splits == 1 and model.choose_calls == 1
                    assert model.closing_handover_stats['searched'] > 0
                    sums = torch.zeros_like(initial, dtype=torch.float64).index_add_(0, labels, decoded.double())
                    counts = torch.bincount(labels, minlength=4)
                    stable = ~model.moved_at_close & (counts > 0)
                    torch.testing.assert_close(centers[stable], (sums[stable] / counts[stable, None]).float(), rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(centers[model.moved_at_close], model.before_close[model.moved_at_close], rtol=0, atol=0)
                    torch.testing.assert_close(rows.view(torch.uint8), saved_rows, rtol=0, atol=0)
                    torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)

    def test_owned_copies_foreign_guard_and_genuine_assignment_decode_hooks(self):
        for name in FAMILIES:
            for backend in ('triton', 'cute'):
                with self.subTest(name=name, backend=backend):
                    initial, data, q = family(name, torch.device('cuda'))
                    options = dict(quantizer=q, assignment_backend=backend, nepochs=1,
                                   batch_size=4, alpha0=1, assignment_chunk_size=2)
                    model, foreign = FixedAxes(2, 4, **options), FixedAxes(2, 4, **options)
                    model.begin(4, initial)
                    foreign.begin(4, initial)
                    foreign_stats = foreign.accumulate(data)
                    # Observers delegate through the existing public signatures.
                    # A genuine decoder override must be used for both statistics.
                    with mock.patch.object(model, '_assign_rows', wraps=model._assign_rows) as assign:
                        if q is None:
                            stats = model.accumulate(data)
                        else:
                            with mock.patch.object(q, 'decode', wraps=q.decode) as decode:
                                stats = model.accumulate(data)
                                self.assertGreater(decode.call_count, 0)
                    self.assertEqual(assign.call_count, 2)
                    for call in assign.call_args_list:
                        self.assertEqual(len(call.args), 6)
                        self.assertEqual(call.args[5].dtype, torch.int64 if name in ('int8', 'uint8', 'int8_cal') else torch.float32)
                    saved = [value.clone() for value in (model._centroids, model._usage, model._batch_removal)]
                    with self.assertRaisesRegex(ValueError, "this model's accumulate"):
                        model.update(foreign_stats)
                    for actual, expected in zip((model._centroids, model._usage, model._batch_removal), saved):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    copied = dict(stats, sums=stats['sums'].clone(), counts=stats['counts'].clone())
                    self.assertIs(copied['removal'], model._batch_removal)
                    model.update(copied)
                    model.end_epoch()
                    self.assertTrue(bool(torch.isfinite(model.finalize()).all()))

    def test_explicit_mass_keeps_no_cost_axes(self):
        for name in FAMILIES:
            initial, data, q = family(name, torch.device('cuda'))
            m = FixedAxes(2, 4, quantizer=q, unused_threshold=.1, nepochs=1,
                          batch_size=4, assignment_backend='triton')
            m.begin(4, initial)
            stats = m.accumulate(data)
            assert m._removal_stats is None and not hasattr(m, 'init_calls')
            assert 'removal' not in stats
