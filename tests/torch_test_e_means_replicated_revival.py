# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.
"""Small independent laws for caller-driven replicated floating revival."""
import datetime
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from faiss.contrib.e_means import DistributedEmeans, DistributedNumpyEmeans, NumpyEmeans
from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means import e_means_distributed as replica


def _host(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _pairs(model):
    """Enumerate the law from readout/state; never call a production selector."""
    removal = _host(model._removal_stats).astype('float64')
    usage = _host(model._usage)
    gain = (np.linalg.norm(removal[:, :-1], axis=1) * getattr(model, '_axis_scale', 1.)).astype('float32')
    candidates = np.flatnonzero((removal[:, -1] <= 0) | (removal[:, -1] < .3 * np.median(removal[:, -1])))
    kept = candidates[np.argsort(removal[candidates, -1], kind='stable')].tolist()
    donors = [int(i) for i in np.argsort(-gain, kind='stable') if usage[i] > 0 and i not in kept]
    n = min(len(donors), len(kept))
    if not n:
        return None, None
    source, target = np.array(donors[:n], dtype='int64'), np.array(kept[:n], dtype='int64')
    if isinstance(model._centroids, np.ndarray):
        return source, target
    return torch.from_numpy(source).to(model._centroids.device), torch.from_numpy(target).to(model._centroids.device)


def _reference_revival(model, rng):
    source, target = _pairs(model)
    return 0 if source is None else model._apply_splits(source, target)


def _agree(value, group):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    copies = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(copies, tensor, group=group)
    for other in copies:
        torch.testing.assert_close(tensor, other, rtol=0, atol=0)


class _InitWitness(DistributedEmeans):
    def _initialize_removal_statistics(self):
        self.init_calls = getattr(self, 'init_calls', 0) + 1
        super()._initialize_removal_statistics()


class _DecodedCast(CastQuantizer):
    def decode(self, codes):
        self.decode_calls = getattr(self, 'decode_calls', 0) + 1
        return super().decode(codes) + .125


class _CentroidInitWitness(_InitWitness):
    def _initialize_removal_statistics(self):
        super()._initialize_removal_statistics()
        self.init_centroid_sign = float(self._centroids[0, 0])
        self._axes.mul_(1. if self.init_centroid_sign > 0 else -1.)


class _NumpyCentroidInitWitness(DistributedNumpyEmeans):
    def _initial_removal_axes(self):
        self.init_calls = getattr(self, 'init_calls', 0) + 1
        self.init_centroid_sign = float(self._centroids[0, 0])
        return super()._initial_removal_axes() * (1. if self.init_centroid_sign > 0 else -1.)


def _caller_case(group, device, transport, *, numpy_state=False, dtype=torch.float32,
                 centered=False, custom_decode=False, mixed=False, epochs=1):
    rank = dist.get_rank(group)
    initial_cpu = np.array([[100.], [200.], [0.], [300.], [400.], [500.], [600.], [2.]], dtype='float32')
    initial = initial_cpu.copy() if numpy_state else torch.from_numpy(initial_cpu).to(device)
    options = dict(nepochs=epochs, batch_size=4, alpha0=.5, seed=7, assignment_chunk_size=1)
    q = None
    if centered or custom_decode:
        q = (_DecodedCast if custom_decode else CastQuantizer)(1, torch.bfloat16, center=True)
        q.train(torch.full((2, 1), 100., device=device))
    input_dtype = torch.float32 if q is not None else dtype
    if numpy_state:
        model = DistributedNumpyEmeans(1, 8, group=group, transport=transport, **options)
        reference = NumpyEmeans(1, 8, **options)
    else:
        model = _InitWitness(1, 8, group=group, transport=transport, quantizer=q, **options)
        reference = QuantizedEmeans(1, 8, quantizer=q, **options)
    caller_initial = initial + 1000 * rank
    saved_initial = _host(caller_initial).copy()
    population = 28 if epochs == 2.5 else 26  # Includes a real global tail.
    model.begin(population, caller_initial)
    reference.begin(population, initial)
    np.testing.assert_array_equal(_host(caller_initial), saved_initial)
    before_rng = None if numpy_state else model._split_rng.get_state().clone()
    empty = np.empty((0, 1), dtype='float32') if numpy_state else torch.empty((0, 1), dtype=input_dtype, device=device)
    # Initial binding and all subsequent caller partials communicate nothing.
    methods = ('broadcast', 'all_reduce', 'all_gather', 'all_gather_object')
    with mock.patch.multiple(dist, **{name: mock.Mock(side_effect=AssertionError('partial collective')) for name in methods}):
        model.accumulate(empty)
    reference.accumulate(empty)
    if numpy_state:
        reference._axes[:] = model._axes
    else:
        reference._axes.copy_(model._axes)
        torch.testing.assert_close(model._split_rng.get_state(), before_rng, rtol=0, atol=0)
        assert model.init_calls == 1
    _agree(model._axes, group)
    reference._revive_unused = types.MethodType(_reference_revival, reference)
    choice_calls = [0]
    original_choice = model._choose_splits

    def checked_choice(rng):
        saved = [_host(v).copy() for v in (model._centroids, model._usage, model._removal_stats, model._axes)]
        old_count = model.n_splits
        actual = original_choice(rng)
        wanted = _pairs(model)
        for a, b in zip(actual, wanted):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_array_equal(_host(a), _host(b))
        for value, old in zip((model._centroids, model._usage, model._removal_stats, model._axes), saved):
            np.testing.assert_array_equal(_host(value), old)
        assert model.n_splits == old_count
        choice_calls[0] += 1
        return actual

    model._choose_splits = checked_choice
    first_axes = _host(model._axes).copy()
    for epoch, epoch_n in enumerate(model.recipe['epoch_rows']):
        for step, begin in enumerate(range(0, epoch_n, 4)):
            full_cpu = np.array([[0.], [.5], [2.], [2.5]], dtype='float32')
            n = min(4, epoch_n - begin)
            full = full_cpu[:n] if numpy_state else torch.from_numpy(full_cpu[:n]).to(device=device, dtype=input_dtype)
            local = full[rank * (n // 2):(rank + 1) * (n // 2)]
            with mock.patch.multiple(dist, **{name: mock.Mock(side_effect=AssertionError('partial collective')) for name in methods}):
                if rank == 0 and len(local) > 1:
                    partial = model.accumulate(local[:1])
                    assert partial['sums'] is None
                    model.accumulate(empty)
                    stats = model.accumulate(local[1:])
                else:
                    stats = model.accumulate(local)
            reference_stats = reference.accumulate(full)
            # S/N are caller-owned copies. Removal remains the documented
            # borrowed model provenance object, consumed by the shared EMA.
            stats = dict(stats, **{name: value.copy() if isinstance(value, np.ndarray) else value.clone() if isinstance(value, torch.Tensor) else value
                                   for name, value in stats.items() if name in ('sums', 'counts', 'objective')})
            saved_sums, saved_counts = _host(stats['sums']).copy(), _host(stats['counts']).copy()
            if mixed:
                assert not numpy_state
                stats = dict(stats, sums=stats['sums'].double().requires_grad_(),
                             counts=stats['counts'].double().requires_grad_(),
                             objective=stats['objective'].clone().requires_grad_())
                reference_stats = dict(reference_stats, sums=reference_stats['sums'].double(),
                                       counts=reference_stats['counts'].double())
            model.update(stats)
            reference.update(reference_stats)
            np.testing.assert_array_equal(_host(stats['sums']), saved_sums)
            np.testing.assert_array_equal(_host(stats['counts']), saved_counts)
            for name in ('_centroids', '_usage', '_removal_stats', '_axes', '_pass_counts'):
                actual, expected = _host(getattr(model, name)), _host(getattr(reference, name))
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
                _agree(getattr(model, name), group)
            if model._pass_sums is not None:
                np.testing.assert_allclose(_host(model._pass_sums), _host(reference._pass_sums), rtol=1e-5, atol=1e-5)
            if epoch == step == 0:
                wanted_r = np.zeros(8, dtype='float32'); wanted_r[[2, 7]] = [3., 5.]
                offset = .125 if custom_decode else 0.
                expected_gain = .5 * (offset**2 + (.5 + offset)**2)
                wanted_u = np.zeros((8, 1), dtype='float32'); wanted_u[[2, 7]] = expected_gain * first_axes[[2, 7]]
                np.testing.assert_array_equal(_host(model._removal_stats)[:, -1], wanted_r)
                effective = _host(model._removal_stats)[:, :-1] * getattr(model, '_axis_scale', 1.)
                np.testing.assert_array_equal(effective, wanted_u)
                if mixed:
                    assert model._batch_stats.dtype == torch.float64 and model._batch_removal.dtype == torch.float32
            assert '_communication' not in vars(model)
            assert model._epoch_rows == begin + n and model.n_splits == reference.n_splits
        model.end_epoch(); reference.end_epoch()
        np.testing.assert_allclose(_host(model._objective), _host(reference._objective), rtol=1e-5, atol=1e-5)
    assert choice_calls[0] > 0 if rank == 0 else choice_calls[0] == 0
    assert model.n_splits == reference.n_splits and model.n_splits > 0
    if custom_decode:
        assert q.decode_calls > 0
    np.testing.assert_allclose(_host(model.finalize()), _host(reference.finalize()), rtol=1e-5, atol=1e-5)
    assert '_communication' not in vars(model) and model._axis_seed is None


def _typed_transport(group, device, transport):
    rank = dist.get_rank(group)
    sn = torch.zeros((8, 2), dtype=torch.int64, device=device)
    ru = torch.zeros((8, 2), dtype=torch.float32, device=device)
    sn[2] = torch.tensor([2**24 + 1, 1], device=device)
    ru[2, 0] = .125 * (rank + 1)  # Zero R cannot exclude this U record.
    if rank:
        sn[7] = torch.tensor([2**53 + 1, 1], device=device)
        ru[7] = .5
    saved = sn.clone(), ru.clone()
    ids, values = replica._combine_payloads((sn.clone(), ru.clone()), sn[:, -1], group, transport, 4)
    expected_sn, expected_ru = sn.new_zeros((8, 2)), ru.new_zeros((8, 2))
    expected_sn[2] = torch.tensor([2**25 + 2, 2], device=device)
    expected_sn[7] = torch.tensor([2**53 + 1, 1], device=device)
    expected_ru[2, 0], expected_ru[7] = .375, .5
    for actual, expected in zip(values, (expected_sn, expected_ru)):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected if ids is None else expected[ids], rtol=0, atol=0)
    for actual, expected in zip((sn, ru), saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _analytical_batch(group, device, transport, numpy_state=False):
    rank = dist.get_rank(group)
    initial = np.array([[0.], [2.]], dtype='float32')
    options = dict(group=group, transport=transport, seed=7, nepochs=1, batch_size=2, alpha0=.5)
    if numpy_state:
        model = DistributedNumpyEmeans(1, 2, **options)
    else:
        model = DistributedEmeans(1, 2, quantizer=None, **options)
        initial = torch.from_numpy(initial).to(device)
    model.begin(4, initial)
    empty = np.empty((0, 1), dtype='float32') if numpy_state else torch.empty((0, 1), device=device)
    model.accumulate(empty)
    np.testing.assert_array_equal(_host(model._axes)[0], [1.])
    x = np.array([[.25 if rank == 0 else -.25]], dtype='float32')
    stats = model.accumulate(x if numpy_state else torch.from_numpy(x).to(device))
    model.update(stats)
    np.testing.assert_array_equal(_host(model._removal_stats)[:, -1], [4., 0.])
    effective = _host(model._removal_stats)[:, :-1] * getattr(model, '_axis_scale', 1.)
    np.testing.assert_array_equal(effective[:, 0], [.0625, 0.])
    np.testing.assert_array_equal(_host(model._pass_counts), [2., 0.])
    np.testing.assert_array_equal(_host(model._pass_sums), [[0.], [0.]])
    _agree(model._removal_stats, group)
    _agree(model._axes, group)


def _seed_and_legacy_contracts(group, device):
    rank = dist.get_rank(group)
    initial = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3)
    for seed in (-1, 2**63 + 7):
        model = _InitWitness(3, 4, group=group, quantizer=None, seed=seed,
                             batch_size=4, nepochs=1)
        ambient = torch.random.get_rng_state().clone()
        model.begin(4, initial)
        assert model._axes is not None and model._axis_seed == torch.Generator().manual_seed(seed).initial_seed()
        _agree(model._axes, group)
        sums, counts = torch.zeros((4, 3), device=device), torch.zeros(4, device=device)
        with mock.patch.object(dist, 'all_reduce', side_effect=AssertionError('invalid update communicated')):
            with unittest.TestCase().assertRaisesRegex(ValueError, "this model's accumulate"):
                model.update({})  # Provenance validation precedes n access and communication.
            with unittest.TestCase().assertRaisesRegex(ValueError, "this model's accumulate"):
                model.update(dict(sums=sums, counts=counts, objective=torch.zeros((), device=device), n=2))
        model._split_rng.manual_seed(991 + rank)
        saved = model._split_rng.get_state().clone()
        with mock.patch.object(dist, 'broadcast', side_effect=AssertionError('partial broadcast')):
            model.accumulate(torch.empty((0, 3), device=device))
        _agree(model._axes, group)
        torch.testing.assert_close(model._split_rng.get_state(), saved, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), ambient, rtol=0, atol=0)
        with mock.patch.object(dist, 'all_reduce', side_effect=AssertionError('missing R communicated')):
            with unittest.TestCase().assertRaisesRegex(ValueError, "this model's accumulate"):
                model.update(dict(sums=sums, counts=counts, objective=torch.zeros((), device=device), n=2))
        assert model.init_calls == 1 and '_communication' not in vars(model)
    # Explicit mass keeps caller-owned staging/coercion and the model RNG;
    # landed #5915 also retains its donor-axis/removal state.
    for transport in ('dense', 'sparse', 'auto'):
        legacy = DistributedEmeans(3, 4, group=group, transport=transport, quantizer=None,
                                   unused_threshold=.1, batch_size=4, nepochs=1)
        legacy.begin(4, initial)
        legacy._split_rng.manual_seed(991 + rank)
        before_rng = legacy._split_rng.get_state().clone()
        assert legacy._axes is not None and legacy._axis_seed is None
        sums = torch.zeros((4, 3), device=device, dtype=torch.float64, requires_grad=True)
        counts = torch.tensor([2, 0, 0, 0], device=device, dtype=torch.int64)
        before = sums.clone(), counts.clone()
        legacy.update(dict(sums=sums, counts=counts, objective=torch.zeros((), device=device), n=2))
        for value, expected in zip((sums, counts), before):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        torch.testing.assert_close(legacy._split_rng.get_state(), before_rng, rtol=0, atol=0)
        assert '_communication' not in vars(legacy) and legacy._removal_stats is not None
        legacy.end_epoch(); legacy.finalize()


def _canonical_init_hook(group, device, numpy_state=False):
    rank = dist.get_rank(group)
    canonical = np.array([[-1.], [2.]], dtype='float32')
    placeholder = canonical if rank == 0 else -canonical
    if numpy_state:
        caller = placeholder.copy()
        model = _NumpyCentroidInitWitness(1, 2, group=group, batch_size=2, alpha0=.5, nepochs=1, seed=7)
        expected_rng = np.random.RandomState(7)
    else:
        caller = torch.from_numpy(placeholder).to(device)
        q = CastQuantizer(1, torch.bfloat16) if device.type == 'cuda' else None
        model = _CentroidInitWitness(1, 2, group=group, quantizer=q,
                                    batch_size=2, alpha0=.5, nepochs=1, seed=7)
        expected_rng = np.random.Generator(np.random.PCG64(7))
    saved = _host(caller).copy()
    model.begin(4, caller)
    if model._axes is None:
        model.accumulate(torch.empty((0, 1), device=device))
    expected = expected_rng.standard_normal((2, 1))
    expected = -(expected / np.linalg.norm(expected, axis=1, keepdims=True)).astype('float32')
    np.testing.assert_array_equal(_host(model._centroids), canonical)
    np.testing.assert_array_equal(_host(caller), saved)
    np.testing.assert_array_equal(_host(model._axes), expected)
    assert model.init_calls == 1 and model.init_centroid_sign == -1.
    _agree(model._axes, group)
    x = np.array([[-.75 if rank == 0 else -1.25]], dtype='float32')
    stats = model.accumulate(x if numpy_state else torch.from_numpy(x).to(device))
    model.update(stats)
    np.testing.assert_array_equal(_host(model._removal_stats)[:, -1], [9., 0.])
    u = _host(model._removal_stats)[:, :-1] * getattr(model, '_axis_scale', 1.)
    np.testing.assert_array_equal(u[0], expected[0] * .0625)
    _agree(model._removal_stats, group)


def _worker(rank, world, rendezvous, cuda=False, subgroup=False):
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device('cuda', rank) if cuda else torch.device('cpu')
    dist.init_process_group('nccl' if cuda else 'gloo', init_method=rendezvous,
                            rank=rank, world_size=world, timeout=datetime.timedelta(seconds=120))
    group = dist.new_group([1, 2]) if subgroup else dist.group.WORLD
    try:
        if not subgroup or rank in (1, 2):
            _seed_and_legacy_contracts(group, device)
            _canonical_init_hook(group, device)
            if not cuda:
                _canonical_init_hook(group, device, numpy_state=True)
            for transport in ('dense', 'sparse', 'auto'):
                _typed_transport(group, device, transport)
                _analytical_batch(group, device, transport)
                _caller_case(group, device, transport, epochs=2.5 if subgroup else 1)
                if not cuda:
                    _analytical_batch(group, device, transport, numpy_state=True)
                    _caller_case(group, device, transport, numpy_state=True)
                    _caller_case(group, device, transport, mixed=True)
            if cuda:
                for dtype in (torch.float16, torch.bfloat16):
                    _caller_case(group, device, 'sparse', dtype=dtype)
                _caller_case(group, device, 'dense', dtype=torch.bfloat16, centered=True)
                _caller_case(group, device, 'dense', dtype=torch.bfloat16, custom_decode=True)
        dist.barrier()
    finally:
        if subgroup:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


class TestReplicatedFloating(unittest.TestCase):
    def spawn(self, *, cuda=False, subgroup=False):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(_worker, args=(3 if subgroup else 2, 'file://' + directory + '/group', cuda, subgroup),
                                        nprocs=3 if subgroup else 2, join=True)

    def test_cpu_gloo_global_cost_and_typed_transports(self):
        self.spawn()

    def test_cpu_gloo_nonzero_global_root_and_fractional_budget(self):
        self.spawn(subgroup=True)

    @unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2 and torch.cuda.get_device_capability() >= (8, 0),
                         'two NVIDIA SM80+ GPUs/NCCL required for BF16 arms')
    def test_two_gpu_nccl_native_cast_and_centered_cost(self):
        self.spawn(cuda=True)

    def test_initializer_bound_and_generator_position(self):
        for numpy_rng in (False, True):
            factory = (lambda: np.random.RandomState(7)) if numpy_rng else (lambda: np.random.Generator(np.random.PCG64(7)))
            expected_rng = factory()
            expected = expected_rng.standard_normal((3, 65))
            expected = (expected / np.linalg.norm(expected, axis=1, keepdims=True)).astype('float32')
            for budget in (1 << 25, 512):
                rng = factory(); actual = np.empty((3, 65), dtype='float32')
                with mock.patch.object(replica, '_AXIS_HOST_BYTES', budget):
                    replica._fill_initial_axes(actual, rng)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(rng.standard_normal(7), expected_rng.standard_normal(7))
                expected_rng = factory(); expected_rng.standard_normal((3, 65))

    def test_auto_accounts_for_complete_typed_payload(self):
        self.assertEqual(replica._select_transport(4, 1 << 20, 126, 2), 'dense')
        self.assertEqual(replica._select_transport(4, 1 << 20, 126, 2, record_bytes=1024), 'sparse')

    def test_local_initial_axis_draws_and_rng_are_unchanged(self):
        initial = np.arange(12, dtype='float32').reshape(4, 3)
        n = NumpyEmeans(3, 4, seed=7, batch_size=4, nepochs=1)
        n.begin(8, initial)
        rng = np.random.RandomState(7)
        raw = rng.standard_normal((4, 3))
        np.testing.assert_array_equal(n._axes, (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype('float32'))
        np.testing.assert_array_equal(n._split_rng.standard_normal(7), rng.standard_normal(7))
        tensor = QuantizedEmeans(3, 4, seed=7, quantizer=None, batch_size=4, nepochs=1)
        tensor.begin(8, torch.from_numpy(initial))
        tensor.accumulate(torch.empty((0, 3)))
        generator = torch.Generator().manual_seed(7)
        raw = torch.randn(4, 3, generator=generator)
        torch.testing.assert_close(tensor._axes, raw / raw.norm(dim=1, keepdim=True), rtol=0, atol=0)
        torch.testing.assert_close(tensor._split_rng.get_state(), generator.get_state(), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip
                         and torch.cuda.get_device_capability() >= (8, 0), 'Triton NVIDIA SM80+ required')
    def test_custom_apply_keeps_gpu_choice_and_parent_fp32_keys(self):
        class Hooked(QuantizedEmeans):
            def _apply_splits(self, source, target):
                self.apply_calls = getattr(self, 'apply_calls', 0) + 1
                return super()._apply_splits(source, target)

        m = Hooked(2, 3, quantizer=None, assignment_backend='triton', batch_size=2, nepochs=1)
        m.begin(4, torch.tensor([[0., 0.], [2., 0.], [4., 0.]], device='cuda'))
        m.accumulate(torch.empty((0, 2), device='cuda'))
        m._usage.fill_(1)
        m._removal_stats.zero_(); m._removal_stats[:, -1] = torch.tensor([0., 100., 100.], device='cuda')
        m._removal_stats[1, 0] = 1
        m._removal_stats[2, :2] = torch.tensor([1., 2**-13], device='cuda')
        self.assertTrue(m._device_choice_eligible())
        self.assertFalse(m._device_check())
        before = m._centroids.clone(), m._removal_stats.clone(), m._usage.clone()
        source, target = m._choose_splits(m._split_rng)
        self.assertEqual((source.tolist(), target.tolist()), ([1], [0]))
        for actual, expected in zip((source, target), _pairs(m)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Parent host and fused rules both rank FP32 keys. The double norms
        # differ, but their equal FP32 keys retain the lower donor ID.
        host = m._choose_removal_splits(m._removal_stats[:, -1].double().cpu().numpy(),
                                       m._usage.double().cpu().numpy(),
                                       m._removal_stats[:, :-1].double().norm(dim=1).cpu().numpy())
        self.assertEqual((host[0].tolist(), host[1].tolist()), ([1], [0]))
        for actual, expected in zip((m._centroids, m._removal_stats, m._usage), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        counter = torch.zeros((), dtype=torch.int64, device='cuda')
        victims, donors, count = m._removal_pairs(counter)
        self.assertEqual((donors[:int(count)].tolist(), victims[:int(count)].tolist()), ([1], [0]))
        self.assertEqual(int(counter), 1)  # Existing fused FP32 key order retained.
        m._revive_unused(m._split_rng)
        self.assertEqual((m.apply_calls, m.n_splits), (1, 1))
