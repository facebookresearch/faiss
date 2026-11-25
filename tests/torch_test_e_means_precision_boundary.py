# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Real-row precision regression and separately labelled small branch laws."""
import copy
import datetime
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from faiss.contrib.e_means import DistributedEmeans, DistributedNumpyEmeans, NumpyEmeans
from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means import e_means as numpy_engine
from faiss.contrib.e_means import e_means_torch as torch_engine
from faiss.contrib.e_means import e_means_quantized as quantized_engine


def _host(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


@contextmanager
def _small_branch():
    # Branch proof ONLY: actual rows, recipe/share and integer bounds stay real.
    with mock.patch.object(numpy_engine, '_FLOAT32_COUNT_LIMIT', 4), \
            mock.patch.object(torch_engine, '_FLOAT32_COUNT_LIMIT', 4), \
            mock.patch.object(quantized_engine, '_FLOAT32_COUNT_LIMIT', 4):
        yield


@contextmanager
def _no_partial_comms():
    names = ('broadcast', 'all_reduce', 'all_gather', 'all_gather_object')
    with mock.patch.multiple(dist, **{name: mock.Mock(side_effect=AssertionError('partial communication'))
                                      for name in names}):
        yield


class _PrepareWitness(DistributedEmeans):
    def _prepare_statistics(self, stats, expected_n):
        self.calls = getattr(self, 'calls', []) + ['prepare']
        return super()._prepare_statistics(stats, expected_n)

    def _apply_statistics(self, *args, **kwargs):
        self.calls.append('apply')
        return super()._apply_statistics(*args, **kwargs)


class _CostWitness(_PrepareWitness):
    def _initial_removal_axes(self):
        self.init_calls = getattr(self, 'init_calls', 0) + 1
        return torch.ones_like(self._centroids)


class _NumpyCostWitness(DistributedNumpyEmeans):
    def _initial_removal_axes(self):
        self.init_calls = getattr(self, 'init_calls', 0) + 1
        return np.ones_like(self._centroids)


def _models(numpy_state, *, group=None, cost=False, transport='dense', n=10, batch=8):
    initial = np.array([[1e9], [0.], [-1e9], [5e8]], dtype='float32')
    options = dict(nepochs=1, batch_size=batch, alpha0=1, seed=7,
                   unused_threshold=None if cost else 0, assignment_chunk_size=4)
    if numpy_state:
        cls = (_NumpyCostWitness if cost else DistributedNumpyEmeans) if group is not None else NumpyEmeans
        model = cls(1, 4, **options, **({} if group is None else dict(group=group, transport=transport)))
        caller = initial.copy()
    else:
        cls = (_CostWitness if cost else _PrepareWitness) if group is not None else QuantizedEmeans
        model = cls(1, 4, quantizer=None, **options,
                    **({} if group is None else dict(group=group, transport=transport)))
        caller = torch.from_numpy(initial.copy())
    model.begin(n, caller)
    np.testing.assert_array_equal(_host(caller), initial)
    return model, initial


def _branch_case(numpy_state, transport, mode='native', cost=False):
    model, initial = _models(numpy_state, group=dist.group.WORLD, transport=transport, cost=cost)
    rank = dist.get_rank()
    rows = np.array([[2**24], [1.], [-2**24], [1.]], dtype='float32')
    values = rows if numpy_state else torch.from_numpy(rows.copy())
    empty = values[:0]
    with _small_branch(), _no_partial_comms():
        model.accumulate(empty)
        cut = 1 if rank == 0 else 2
        partial = model.accumulate(values[:cut])
        assert partial['sums'] is None and partial['counts'] is None
        model.accumulate(empty)
        stats = model.accumulate(values[cut:])
    assert stats['n'] == 4 and _host(stats['sums']).dtype == np.float64
    if not numpy_state:
        assert stats['stat_schema'] == torch.float32
    np.testing.assert_array_equal(_host(stats['sums'])[:, 0], [0., 2., 0., 0.])
    edit = mode in ('edit', 'replace')
    independent = mode in ('deep', 'independent', 'edit', 'replace', 'fp64', 'unannotated')
    if mode == 'dict':
        stats = dict(stats)
    elif independent:
        removal = stats.get('removal')
        stats = copy.deepcopy(stats) if mode == 'deep' else dict(stats)
        stats['sums'] = stats['sums'].copy() if numpy_state else stats['sums'].clone()
        stats['counts'] = stats['counts'].copy() if numpy_state else stats['counts'].clone()
        if removal is not None:
            stats['removal'] = removal  # Required existing removal provenance.
    if edit:
        stats['sums'][1, 0] += .25
        if mode == 'replace':
            stats['counts'] = stats['counts'].copy() if numpy_state else stats['counts'].clone()
    if not numpy_state and mode == 'fp64':
        stats['stat_schema'] = torch.float64
    if not numpy_state and mode == 'unannotated':
        stats.pop('stat_schema')
    saved = {key: _host(stats[key]).copy() for key in ('sums', 'counts')}
    model.update(stats)
    global_sum = 4.5 if edit else 4.
    expected_dtype = np.float64 if not numpy_state and mode in ('fp64', 'unannotated') else np.float32
    assert _host(model._batch_stats).dtype == expected_dtype
    np.testing.assert_array_equal(_host(model._pass_sums)[:, 0], [0., global_sum, 0., 0.])
    np.testing.assert_array_equal(_host(model._pass_counts), [0., 8., 0., 0.])
    np.testing.assert_array_equal(_host(model.current_centroids())[1], [global_sum / 8])
    if independent:
        for key, value in saved.items():
            np.testing.assert_array_equal(_host(stats[key]), value)
    if not numpy_state:
        assert model.calls == ['prepare', 'apply']
    if cost:
        assert model.init_calls == 1
        assert _host(model._removal_stats).dtype == np.float32
        # Independent D1 oracle: y=x at center0; all axes are +1.
        all_rows = np.concatenate((rows, rows)).astype('float64')[:, 0]
        distances = np.square(all_rows[:, None] - initial[:, 0].astype('float64'))
        nearest_two = np.sort(distances, axis=1)[:, :2]
        expected_u = np.square(all_rows).sum()
        expected_r = np.diff(nearest_two, axis=1).sum()
        np.testing.assert_allclose(_host(model._removal_stats)[1], [expected_u, expected_r], rtol=1e-5, atol=1e-5)
    with _no_partial_comms():
        after = model.accumulate(empty)
        assert _host(after['sums']).dtype == expected_dtype and 'stat_schema' not in after
        # Actual next logical tail is2, below the lowered4 threshold.
        tail = np.array([[2.]], dtype='float32')
        stats = model.accumulate(tail if numpy_state else torch.from_numpy(tail))
        assert _host(stats['sums']).dtype == expected_dtype
    model.update(stats)
    model.end_epoch()
    final = _host(model.finalize())
    np.testing.assert_allclose(final[1], [(global_sum + 4) / 10], rtol=0, atol=np.spacing(np.float32(.85)))
    np.testing.assert_array_equal(final[[0, 2, 3]], initial[[0, 2, 3]])


def _small_declared_raw(transport):
    model, _ = _models(False, group=dist.group.WORLD, transport=transport, n=4, batch=4)
    rank = dist.get_rank()
    stats = dict(n=2, sums=torch.zeros((4, 1), dtype=torch.float64),
                 counts=torch.tensor([0, 2, 0, 0], dtype=torch.int64),
                 objective=torch.tensor(0., dtype=torch.float64), stat_schema=torch.float32)
    stats['sums'][1, 0] = (1 + 2**-24) if rank == 0 else -1
    saved = stats['sums'].clone()
    model.update(stats)
    # Early per-rank FP32 narrowing erases the nonzero exact global sum.
    np.testing.assert_array_equal(_host(model._pass_sums)[1], [2**-24])
    np.testing.assert_array_equal(_host(model.current_centroids())[1], [2**-26])
    torch.testing.assert_close(stats['sums'], saved, rtol=0, atol=0)
    assert model.accumulate(torch.empty((0, 1)))['sums'].dtype == torch.float32


def _real_case(numpy_state):
    size, share = 2**24 + 2, 2**23 + 1
    initial = np.array([[0.], [1.]], dtype='float32')
    options = dict(nepochs=1, batch_size=size, alpha0=1, seed=7,
                   unused_threshold=0, assignment_chunk_size=1 << 18)
    cls = DistributedNumpyEmeans if numpy_state else DistributedEmeans
    model = cls(1, 2, group=dist.group.WORLD, transport='dense', **options,
                **({} if numpy_state else dict(quantizer=None)))
    caller = initial.copy() if numpy_state else torch.from_numpy(initial.copy())
    model.begin(size, caller)
    rows = np.full((share, 1), np.float32(.1), dtype='float32')
    values = rows if numpy_state else torch.from_numpy(rows)
    cut = share // 3 if dist.get_rank() == 0 else share // 2
    with _no_partial_comms():
        model.accumulate(values[:cut])
        model.accumulate(values[:0])
        stats = model.accumulate(values[cut:])
    assert stats['n'] == share and _host(stats['sums']).dtype == np.float64
    if not numpy_state:
        assert stats['stat_schema'] == torch.float32
    model.update(stats)
    exact_sum = size * float(np.float32(.1))
    np.testing.assert_array_equal(_host(stats['sums'])[:, 0], [exact_sum, 0.])
    np.testing.assert_array_equal(_host(stats['counts']), [size, 0.])
    np.testing.assert_array_equal(_host(model._pass_sums)[:, 0], [np.float32(exact_sum), 0.])
    np.testing.assert_array_equal(_host(model._pass_counts), [size, 0.])
    expected = torch.tensor([[np.float32(.1)], [1.]], dtype=torch.float32)
    if dist.get_rank() == 0:
        local_cls = NumpyEmeans if numpy_state else QuantizedEmeans
        reference = local_cls(1, 2, **options, **({} if numpy_state else dict(quantizer=None)))
        reference.begin(size, initial.copy() if numpy_state else torch.from_numpy(initial.copy()))
        whole = np.full((size, 1), np.float32(.1), dtype='float32')
        reference_stats = reference.accumulate(whole if numpy_state else torch.from_numpy(whole))
        assert _host(reference_stats['sums']).dtype == np.float32 and 'stat_schema' not in reference_stats
        reference.update(reference_stats)
        np.testing.assert_array_equal(_host(reference._pass_sums), _host(model._pass_sums))
        expected.copy_(torch.from_numpy(_host(reference.current_centroids()).copy()))
    dist.broadcast(expected, src=0)
    np.testing.assert_allclose(_host(model.current_centroids()), expected.numpy(), rtol=0,
                               atol=np.spacing(np.float32(.1)))
    np.testing.assert_array_equal(_host(caller), initial)
    np.testing.assert_array_equal(rows, np.full_like(rows, np.float32(.1)))
    model.end_epoch()
    assert model.n_splits == 0
    np.testing.assert_allclose(_host(model.finalize()), expected.numpy(), rtol=0,
                               atol=np.spacing(np.float32(.1)))


def _pending_schema_case():
    for tag in (torch.float64, 'invalid', None, torch.int64, 'removed'):
        model, _ = _models(False, group=dist.group.WORLD, n=8)
        values = torch.tensor([[1.], [2.], [3.], [4.]])
        with _small_branch(), _no_partial_comms():
            stats = model.accumulate(values)
            assert stats['sums'].dtype == torch.float64 and stats['stat_schema'] == torch.float32
            if tag == 'removed':
                stats.pop('stat_schema')
            else:
                stats['stat_schema'] = tag  # Edit the actual published pending record.
            after_empty = model.accumulate(values[:0])
        assert 'stat_schema' not in after_empty if tag == 'removed' else after_empty['stat_schema'] == tag
        if tag in (torch.float64, 'removed'):
            model.update(after_empty)
            expected_dtype = torch.float64
        else:
            with unittest.TestCase().assertRaisesRegex(ValueError, 'stat_schema'):
                model.update(after_empty)
            assert not model._failed and model._step == 0
            assert model._batch_stats.dtype == torch.float32
            # Retry using explicitly repaired metadata; original supplied values remain.
            model.update(dict(after_empty, stat_schema=torch.float32))
            expected_dtype = torch.float32
        assert model._batch_stats.dtype == expected_dtype
        np.testing.assert_array_equal(_host(model._pass_sums)[:, 0], [0., 20., 0., 0.])
        np.testing.assert_array_equal(_host(model.current_centroids())[1], [2.5])
        with _no_partial_comms():
            assert model.accumulate(values[:0])['sums'].dtype == expected_dtype
        model.end_epoch()
        np.testing.assert_array_equal(_host(model.finalize())[1], [2.5])


def _worker(rank, init, kind):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method=init,
                            timeout=datetime.timedelta(seconds=240))
    try:
        if kind.startswith('real'):
            _real_case(kind == 'real_numpy')
        elif kind == 'pending_schema':
            _pending_schema_case()
        else:
            for transport in ('dense', 'sparse', 'auto'):
                for numpy_state in (True, False):
                    for mode in ('native', 'dict', 'deep', 'independent', 'edit', 'replace'):
                        _branch_case(numpy_state, transport, mode)
                    _branch_case(numpy_state, transport, cost=True)
                for mode in ('fp64', 'unannotated'):
                    _branch_case(False, transport, mode)
                _small_declared_raw(transport)
    finally:
        dist.destroy_process_group()


class TestPrecisionBoundary(unittest.TestCase):
    def spawn(self, kind):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(_worker, args=('file://' + directory + '/group', kind),
                                        nprocs=2, join=True)

    def test_real_numpy_rows(self):
        self.spawn('real_numpy')

    def test_real_tensor_rows(self):
        self.spawn('real_tensor')

    def test_reduced_threshold_authority_schema_and_transports(self):
        self.spawn('branch')

    def test_pending_schema_survives_empty_accumulation(self):
        self.spawn('pending_schema')

    def test_local_small_and_wide_completion_laws(self):
        for numpy_state in (True, False):
            for lowered in (False, True):
                model, _ = _models(numpy_state, n=8)
                raw = np.array([[2**24], [1.], [-2**24], [1.]] * 2, dtype='float32')
                with _small_branch() if lowered else mock.patch.object(numpy_engine, '_FLOAT32_COUNT_LIMIT', 2**24):
                    stats = model.accumulate(raw if numpy_state else torch.from_numpy(raw))
                assert _host(stats['sums']).dtype == np.float32 and 'stat_schema' not in stats
                # Lowered wide branch is exact; ordinary sequential FP32 law
                # depends on backend addition order, so only its schema is fixed here.
                if lowered:
                    np.testing.assert_array_equal(_host(stats['sums'])[1], [4.])
                model.update(stats)

    def test_invalid_metadata_and_running_priority(self):
        model = QuantizedEmeans(1, 2, quantizer=None, unused_threshold=0, batch_size=4, nepochs=1)
        invalid = dict(n=4, sums=torch.zeros((2, 1)), counts=torch.ones(2),
                       objective=torch.tensor(0., dtype=torch.float64), stat_schema='float32')
        with self.assertRaises(TypeError):
            model.update(invalid)
        model.begin(4, torch.tensor([[0.], [1.]]))
        saved = model.current_centroids().clone()
        for schema, sums, counts in (
                (None, torch.float64, torch.float64), ('float32', torch.float64, torch.float64),
                (True, torch.float64, torch.float64), (torch.int64, torch.float64, torch.float64),
                (torch.complex64, torch.float64, torch.float64), (torch.float16, torch.float64, torch.float64),
                (torch.float32, torch.int64, torch.int64), (torch.float64, torch.int32, torch.int64)):
            stats = dict(invalid, stat_schema=schema, sums=torch.zeros((2, 1), dtype=sums),
                         counts=torch.ones(2, dtype=counts))
            with self.subTest(schema=schema, sums=sums):
                with self.assertRaisesRegex(ValueError, 'stat_schema'):
                    model.update(stats)
                assert not model._failed and model._batch_stats.dtype == torch.float32
                assert model._step == 0
                torch.testing.assert_close(model.current_centroids(), saved, rtol=0, atol=0)
        for key in ('sums', 'counts'):
            with self.assertRaisesRegex(ValueError, 'stat_schema'):
                model.update(dict(invalid, stat_schema=torch.float32, **{key: None}))
            assert not model._failed and model._step == 0
        model._failed = True
        with self.assertRaises(RuntimeError):
            model.update(invalid)
        numpy = NumpyEmeans(1, 2, unused_threshold=0, batch_size=4, nepochs=1)
        numpy.begin(4, np.array([[0.], [1.]], dtype='float32'))
        with self.assertRaisesRegex(ValueError, 'Tensor'):
            numpy.update(dict(n=4, sums=np.zeros((2, 1), dtype='float32'), counts=np.ones(2, dtype='float32'),
                              objective=np.float64(0), stat_schema=torch.float32))
