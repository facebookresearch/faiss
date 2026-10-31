# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import gc
import unittest
import weakref
import subprocess
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from faiss.contrib.e_means import Emeans
from faiss.contrib.e_means.e_means_quantizer import CastQuantizer
from faiss.contrib.e_means.e_means_fp8 import Fp8Quantizer
from faiss.contrib.e_means.e_means_int8 import Int8Quantizer
from faiss.contrib.e_means import e_means
from e_means_test_utils import known_torch_centroids


def _ready():
    if not torch.cuda.is_available() or getattr(torch.version, 'hip', None) is not None:
        return False
    if torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        return (importlib.metadata.version('quack-kernels') == '0.6.5'
                and importlib.metadata.version('nvidia-cutlass-dsl') == '4.7.1'
                and importlib.metadata.version('cuda-python').startswith('13.'))
    except importlib.metadata.PackageNotFoundError:
        return False


class TestEmeansCute(unittest.TestCase):
    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_byte_exact_distance_output(self):
        from faiss.contrib.e_means.e_means_cute import _prepare_centroids, _assign_rows
        for dtype, offset in ((torch.int8, 0), (torch.uint8, 128)):
            for d, k in ((8192, 1), (320, 128)):
                rows = torch.full((1, d), offset, dtype=dtype, device='cuda')
                c = torch.full((k, d), float(offset), device='cuda')
                if d == 8192:
                    c[:, :1024] -= 128
                    c[:, 1024] += 1
                    expected = 2**24 + 1
                else:
                    rows[:, :259] = 127 + offset
                    c[:, :259] -= 128
                    expected = 259 * 255**2
                prepared = _prepare_centroids(c, dtype=dtype)
                for output_dtype in (torch.float32, torch.int64, torch.float32, torch.int64):
                    output = torch.full((3,), -123, dtype=output_dtype, device='cuda')
                    distance, labels = _assign_rows(rows, prepared, output[1:-1])
                    self.assertEqual(distance.data_ptr(), output[1:-1].data_ptr())
                    self.assertEqual(distance.item(), torch.tensor(expected, dtype=output_dtype).item())
                    self.assertEqual((labels.item(), output[[0, -1]].tolist()), (0, [-123, -123]))
                    saved = distance.clone()
                    normal, _ = _assign_rows(rows, prepared)
                    self.assertEqual(normal.dtype, torch.float32)
                    torch.testing.assert_close(distance, saved, rtol=0, atol=0)
        prepared = _prepare_centroids(torch.zeros((1, 2), device='cuda'))
        output = torch.full((1,), -123, dtype=torch.int64, device='cuda')
        with self.assertRaisesRegex(ValueError, 'quantizer=None'):
            _assign_rows(torch.zeros((1, 2), dtype=torch.bfloat16, device='cuda'), prepared, output)
        self.assertEqual(output.item(), -123)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_captured_binding_survives_eviction_and_retires(self):
        rows = (torch.arange(10247, device='cuda') % 2).bfloat16()
        rows = rows[:, None].expand(-1, 16).contiguous()
        initial = torch.tensor([0., 1.], device='cuda')[:, None].expand(-1, 16).contiguous()
        get_chunk = Emeans._get_assignment_chunk_size
        capture_step = Emeans._capture_step
        end_epoch = Emeans.end_epoch

        # Diagnostic geometry creates a third binding size without fixing the
        # production resolver or increasing its cache limit.
        def chunk(model, is_torch, cuda=False):
            return 37 if model.assignment_backend == 'cute' and cuda else get_chunk(model, is_torch, cuda)

        for fail_after_tail in (False, True):
            with self.subTest(exception=fail_after_tail):
                owners = []
                graph_owner = None

                def capture(
                    model,
                    *args,
                    _capture_step=capture_step,
                    _owners=owners,
                    **kwargs,
                ):
                    nonlocal graph_owner
                    captured = _capture_step(model, *args, **kwargs)
                    _owners.extend(
                        weakref.ref(binding)
                        for binding in captured[1].bindings.values()
                    )
                    graph_owner = weakref.ref(captured[0])
                    return captured

                def finish_epoch(
                    model,
                    _end_epoch=end_epoch,
                    _fail_after_tail=fail_after_tail,
                    _owners=owners,
                ):
                    _end_epoch(model)
                    gc.collect()
                    self.assertTrue(_owners)
                    self.assertTrue(all(owner() is not None for owner in _owners))
                    if _fail_after_tail:
                        raise RuntimeError('injected post-tail failure')

                model = Emeans(16, 2, niter=2, seed=10, assignment_backend='cute',
                               capture_assignments=True)
                with mock.patch.object(Emeans, '_get_assignment_chunk_size', chunk), \
                     mock.patch.object(Emeans, '_capture_step', capture), \
                     mock.patch.object(Emeans, 'end_epoch', finish_epoch):
                    gc.collect()
                    if fail_after_tail:
                        with self.assertRaisesRegex(RuntimeError, 'injected post-tail failure'):
                            model.train(rows, initial)
                        self.assertIsNone(model.assignments)
                    else:
                        model.train(rows, initial)
                        torch.testing.assert_close(model.centroids, initial, rtol=0, atol=0)
                        torch.testing.assert_close(model.obj, torch.zeros_like(model.obj), rtol=0, atol=0)
                        torch.testing.assert_close(model.assignments, rows[:, 0].long(), rtol=0, atol=0)
                        distances, labels = model.assign(rows)
                        torch.testing.assert_close(distances, torch.zeros_like(distances), rtol=0, atol=0)
                        torch.testing.assert_close(labels, rows[:, 0].long(), rtol=0, atol=0)
                gc.collect()
                self.assertIsNotNone(graph_owner)
                self.assertIsNone(graph_owner())
                model.reset()
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))

    def test_scope(self):
        model = Emeans(2, 2, assignment_backend='cute')
        for x in (np.zeros((2, 2), dtype='float32'), torch.zeros(2, 2)):
            with self.assertRaisesRegex(ValueError, 'CuTe assignment requires CUDA'):
                model.train(x)
            with self.assertRaisesRegex(ValueError, 'CuTe assignment requires CUDA'):
                model.begin(2, x)
            with self.assertRaisesRegex(RuntimeError, 'train must be called'):
                model.assign(x)
            model.reset()
        with self.assertRaisesRegex(ValueError, 'requires BF16'):
            Emeans(2, 2, assignment_backend='cute', quantizer=CastQuantizer(2, torch.float16))
        with self.assertRaisesRegex(ValueError, 'int32'):
            Emeans(2, 2**31 - 127, assignment_backend='cute')

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_and_jit_training(self):
        x = torch.tensor([[0.], [2.], [10.], [12.]], device='cuda')
        initial = x[[0, 2]]
        native = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, assignment_backend='cute',
                        capture_assignments=True)
        jit = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, assignment_backend='cute',
                     quantizer=CastQuantizer(1, torch.bfloat16), capture_assignments=True)
        native.train(x.bfloat16(), initial)
        jit.train(x, initial)
        for model in (native, jit):
            torch.testing.assert_close(model.centroids, torch.tensor([[1.], [11.]], device='cuda'))
            self.assertEqual(model.obj.item(), 8.)
            torch.testing.assert_close(model.assignments, torch.tensor([0, 0, 1, 1], device='cuda'), rtol=0, atol=0)
        distance, labels = native.assign(x.bfloat16())
        torch.testing.assert_close(distance, torch.ones(4, device='cuda'))
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], device='cuda'))
        with self.assertRaisesRegex(ValueError, 'requires native BF16/E4M3'):
            Emeans(1, 2, assignment_backend='cute').train(x, initial)

    def test_optional_stack_and_device_scope(self):
        # CUDA initialization may probe optional packages: do it before the
        # missing-package blocker. Native use must still avoid CuTe imports.
        code = '''
import importlib.abc, sys, torch, warnings
with warnings.catch_warnings(record=True) as imported:
    warnings.simplefilter('always')
    from faiss.contrib.e_means import Emeans
assert not any('CuTe grouping is using Torch' in str(item.message) for item in imported)
cuda = torch.cuda.is_available() and not torch.version.hip
if cuda: torch.cuda.get_device_capability()
class NoCute(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('quack', 'cutlass'): raise ImportError('blocked optional CuTe')
sys.meta_path.insert(0, NoCute())
Emeans(1, 1, niter=1, batch_size=2, alpha0=.5).train(torch.zeros(2, 1))
assert not any(n.split('.')[0] in ('quack', 'cutlass') for n in sys.modules)
if cuda:
    try: Emeans(1, 1, assignment_backend='cute').train(torch.zeros(2, 1, dtype=torch.bfloat16, device='cuda'))
    except RuntimeError as error: assert 'CuTe assignment requires' in str(error)
    else: raise AssertionError('missing optional stack silently accepted')
'''
        child = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=60)
        self.assertEqual(child.returncode, 0, child.stderr)
        from faiss.contrib.e_means import e_means_cute as backend
        from faiss.contrib.e_means import e_means_grouping as grouping
        with mock.patch.object(backend, '_stack_status', return_value=(True, 'ready')), \
             mock.patch.object(grouping, '_grouping_provider',
                               side_effect=ImportError('broken CCCL')) as provider:
            backend._require_stack()
            backend._require_stack()
        provider.assert_not_called()

        model = Emeans(16, 65536, assignment_backend='auto')
        rows = SimpleNamespace(device=torch.device('cuda', 17), dtype=torch.bfloat16)
        with mock.patch.object(e_means, '_is_torch_tensor', return_value=True), \
             mock.patch.object(backend, '_stack_status', return_value=(True, 'ready')), \
             mock.patch.object(torch.cuda, 'device', return_value=nullcontext()), \
             mock.patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
             mock.patch.object(torch.cuda, 'get_device_properties',
                               return_value=mock.Mock(shared_memory_per_block_optin=232448)), \
             mock.patch.object(e_means, '_get_triton_assign_rows', return_value=object()), \
             mock.patch.object(grouping, '_grouping_provider',
                               side_effect=ImportError('broken CCCL')) as provider:
            model._resolve_assignment_backend(rows)
        self.assertEqual(model._resolved_assignment_backend, 'cute')
        provider.assert_not_called()
        try:
            for major in ('12.9', '14.0'):
                backend._stack_status.cache_clear()
                with mock.patch.object(
                    importlib.metadata,
                    'version',
                    side_effect=lambda name, _major=major: (
                        _major
                        if name == 'cuda-python'
                        else {
                            'quack-kernels': '0.6.5',
                            'nvidia-cutlass-dsl': '4.7.1',
                        }[name]
                    ),
                ):
                    self.assertEqual(backend._stack_status(), (False, 'cuda-python requires major version 13'))
        finally:
            backend._stack_status.cache_clear()
        # Narrow adapter guard diagnostics use an unchecked fictitious device;
        # cache clearing prevents prior successful device checks hiding errors.
        try:
            for capability in ((8, 0), (10, 0), (10, 3)):
                backend._check_cuda_device.cache_clear()
                with mock.patch.object(torch.cuda, 'device', return_value=nullcontext()), \
                     mock.patch.object(backend, '_require_stack'), \
                     mock.patch.object(torch.cuda, 'get_device_capability', return_value=capability), \
                     self.assertRaisesRegex(RuntimeError, 'exactly SM90'):
                    backend._check_device('cuda:17')
            backend._check_cuda_device.cache_clear()
            with mock.patch.object(torch.cuda, 'device', return_value=nullcontext()), \
                 mock.patch.object(backend, '_require_stack'), \
                 mock.patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
                 mock.patch.object(torch.cuda, 'get_device_properties', return_value=mock.Mock(shared_memory_per_block_optin=0)), \
                 self.assertRaisesRegex(RuntimeError, 'opt-in shared memory'):
                backend._check_device('cuda:17')
        finally:
            backend._check_cuda_device.cache_clear()
        with mock.patch.object(torch.version, 'hip', 'rejection-only probe'), \
             self.assertRaisesRegex(RuntimeError, 'NVIDIA CUDA SM90'):
            backend._check_device('cuda:17')

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_non_grouped_cute_ignores_broken_cccl(self):
        from faiss.contrib.e_means import e_means_grouping as grouping
        from faiss.contrib.e_means.e_means_cute import _prepare_centroids, _assign_rows

        rows = torch.zeros((3, 16), dtype=torch.bfloat16, device='cuda')
        centers = torch.zeros((128, 16), device='cuda')
        broken = ImportError('broken installed CCCL')
        with mock.patch.object(grouping, '_grouping_provider', side_effect=broken) as provider:
            prepared = _prepare_centroids(centers, dtype=torch.bfloat16)
            distances, labels = _assign_rows(rows, prepared)
        provider.assert_not_called()
        self.assertNotEqual(prepared.bindings[len(rows)].launcher.__name__, 'launch_grouped')
        torch.testing.assert_close(distances, torch.zeros_like(distances), rtol=0, atol=0)
        torch.testing.assert_close(labels, torch.zeros_like(labels), rtol=0, atol=0)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_bf16_destinations_ragged_ties_and_owned_outputs(self):
        from faiss.contrib.e_means.e_means_cute import _prepare_centroids, _assign_rows
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for n, k, d in ((1, 128, 1), (3, 256, 17), (33, 512, 96), (129, 384, 65), (7, 193, 257)) + (((17, 384, 385), (3, 384, 513)) if dtype is torch.float8_e4m3fn else ()):
                with self.subTest(dtype=dtype, n=n, k=k, d=d):
                    g = torch.Generator(device='cuda').manual_seed(n+k+d)
                    rows = torch.randint(-2, 3, (n, d), generator=g, device='cuda').to(dtype)
                    c = torch.randint(-2, 3, (k, d), generator=g, device='cuda').float()
                    c[1].copy_(c[0])
                    if (k, d) == (384, 65):
                        c[-1].fill_(3)
                        rows[-1].fill_(3)
                    before_rows, before_centroids = rows.clone(), c.clone()
                    prepared = _prepare_centroids(c, dtype=dtype)
                    distances = torch.full((n+2,), -123., device='cuda')
                    labels = torch.full((n+2,), -123, dtype=torch.int64, device='cuda')
                    result = _assign_rows(rows, prepared, distances[1:-1], labels[1:-1])
                    self.assertEqual(tuple(t.data_ptr() for t in result),
                                     (distances[1:-1].data_ptr(), labels[1:-1].data_ptr()))
                    truth = (rows.double()[:, None]-c.to(dtype).double()[None]).square().sum(2)
                    torch.testing.assert_close(distances[1:-1].double(), truth.min(1).values, rtol=0, atol=2e-4 if dtype is torch.bfloat16 else 0)
                    torch.testing.assert_close(labels[1:-1], truth.argmin(1), rtol=0, atol=0)
                    if (k, d) == (384, 65):
                        self.assertEqual(labels[-2].item(), k-1)
                    self.assertEqual((distances[[0, -1]].tolist(), labels[[0, -1]].tolist()), ([-123., -123.], [-123, -123]))
                    caller_saved = distances.clone(), labels.clone()
                    owned = _assign_rows(rows, prepared)
                    saved = tuple(t.clone() for t in owned)
                    _assign_rows((-rows.float()).to(dtype), prepared)
                    _assign_rows(rows[:1], prepared)
                    for actual, expected in zip(owned, saved):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    model = known_torch_centroids(c, assignment_backend='cute', assignment_chunk_size=max(1, n-1))
                    public = model.assign(rows)
                    torch.testing.assert_close(public[0].double(), truth.min(1).values, rtol=0, atol=2e-4 if dtype is torch.bfloat16 else 0)
                    torch.testing.assert_close(public[1], truth.argmin(1), rtol=0, atol=0)
                    for actual, expected in zip((distances, labels), caller_saved):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(rows.view(torch.uint8), before_rows.view(torch.uint8), rtol=0, atol=0)
                    torch.testing.assert_close(c, before_centroids, rtol=0, atol=0)
        c = torch.full((193, 17), 10., device='cuda')
        c[0].zero_()
        c[127].zero_()
        c[5].fill_(3)
        c[192].fill_(-3)
        c[10].zero_()
        c[11].zero_()
        c[10, 0] = .99609375
        c[11, 0] = 1.00390625
        rows = torch.stack((c[5], c[192], torch.nn.functional.pad(torch.ones(1, device='cuda'), (0, 16)), torch.zeros(17, device='cuda')))
        before_rows, before_centroids = rows.clone(), c.clone()
        model = known_torch_centroids(c, assignment_backend='cute', quantizer=CastQuantizer(17, torch.bfloat16))
        distance, labels = model.assign(rows)
        self.assertEqual(labels[:2].tolist(), [5, 192])
        self.assertIn(labels[2].item(), (10, 11))
        self.assertEqual(labels[3].item(), 0)
        truth = (rows.bfloat16().double()[:, None]-c.bfloat16().double()[None]).square().sum(2)
        torch.testing.assert_close(distance.double(), truth.gather(1, labels[:, None]).flatten(), rtol=0, atol=1e-5)
        torch.testing.assert_close(rows, before_rows, rtol=0, atol=0)
        torch.testing.assert_close(c, before_centroids, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_bf16_norm_refresh_and_subnormal_square(self):
        from faiss.contrib.e_means.e_means_cute import _prepare_centroids
        rows = torch.tensor([[-.4678138, .03789843, -.6960376]], device='cuda')
        c = torch.tensor([[-.8843690, 1.0697185, -.7981653], [.4984003, .5734861, -.5365824]], device='cuda')
        truth = (rows.bfloat16().float()[:, None]-c.bfloat16().float()[None]).square().sum(2)
        wrong = c.square().sum(1)-2*(rows.bfloat16().float()[:, None]*c.bfloat16().float()[None]).sum(2)
        self.assertEqual((truth.argmin(1).item(), wrong.argmin(1).item()), (0, 1))
        for q in (None, CastQuantizer(3, torch.bfloat16)):
            model = known_torch_centroids(c, assignment_backend='cute', quantizer=q)
            distance, labels = model.assign(rows if q else rows.bfloat16())
            self.assertEqual(labels.item(), 0)
            torch.testing.assert_close(distance, truth[:, 0], rtol=0, atol=1e-6)
        prepared = None
        for changed in (c, c*.5+.25):
            prepared = _prepare_centroids(changed, prepared)
            torch.testing.assert_close(prepared.norms, changed.bfloat16().float().square().sum(1), rtol=0, atol=0)
        # Both states are exactly halfway between BF16 values; RN-even gives
        # 1 before mutation and 2 after mutation, hence exact norms 1 and 4.
        halfway = torch.tensor([[1.+2.**-8]], device='cuda')
        prepared = _prepare_centroids(halfway)
        torch.testing.assert_close(prepared.norms, torch.tensor([1.], device='cuda'), rtol=0, atol=0)
        halfway.mul_(2)
        prepared = _prepare_centroids(halfway, prepared)
        torch.testing.assert_close(prepared.norms, torch.tensor([4.], device='cuda'), rtol=0, atol=0)
        with torch.inference_mode():
            model = known_torch_centroids(torch.tensor([[0.], [2.]], device='cuda'), assignment_backend='cute')
            distance, labels = model.assign(torch.tensor([[2.]], device='cuda', dtype=torch.bfloat16))
            self.assertEqual((distance.item(), labels.item()), (0., 1))
            model.centroids[0].fill_(2)
            model.centroids[1].zero_()
            distance, labels = model.assign(torch.tensor([[2.]], device='cuda', dtype=torch.bfloat16))
            self.assertEqual((distance.item(), labels.item()), (0., 0))
        tiny = torch.full((1, 1), 2.**-70, device='cuda')
        prepared = _prepare_centroids(tiny)
        torch.testing.assert_close(prepared.norms, torch.tensor([2.**-140], device='cuda'), rtol=0, atol=0)
        distance, _ = known_torch_centroids(tiny, assignment_backend='cute').assign(torch.zeros(1, 1, dtype=torch.bfloat16, device='cuda'))
        torch.testing.assert_close(distance, torch.tensor([2.**-140], device='cuda'), rtol=0, atol=0)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_bf16_high_ids_and_planted_large_batch(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                from faiss.contrib.e_means.e_means_cute import _prepare_centroids, _assign_rows
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                c = torch.full((16384, 96), 4., device='cuda')
                c[0].zero_()
                c[128].zero_()
                c[777].fill_(1)
                c[1024].fill_(1)
                c[-1].fill_(-1)
                rows = torch.zeros(8192, 96, dtype=dtype, device='cuda')
                rows[1].fill_(1)
                rows[2].fill_(-1)
                rows[3].fill_(.25)
                distance, labels = known_torch_centroids(c, assignment_backend='cute', assignment_chunk_size=8192).assign(rows)
                self.assertEqual(labels[:4].tolist(), [0, 777, 16383, 0])
                torch.testing.assert_close(distance[:4], torch.tensor([0., 0., 0., 6.], device='cuda'), rtol=0, atol=1e-5)
                self.assertTrue(bool((labels[4:] == 0).all()))
                torch.testing.assert_close(distance[4:], torch.zeros_like(distance[4:]), rtol=0, atol=1e-5)
                c = torch.zeros(65537, 1, device='cuda')
                c[-1].fill_(2)
                labels = torch.full((4,), -123, dtype=torch.int64, device='cuda')
                distance, _ = _assign_rows(torch.full((2, 1), 2., dtype=dtype, device='cuda'),
                                           _prepare_centroids(c, dtype=dtype), labels_out=labels[1:-1])
                self.assertEqual(labels.tolist(), [-123, 65536, 65536, -123])
                torch.testing.assert_close(distance, torch.zeros_like(distance), rtol=0, atol=0)
                print('CuTe correctness stress cost', dtype, time.monotonic()-start,
                      'seconds, peak', torch.cuda.max_memory_allocated(), 'bytes')

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_bf16_decoded_tail_statistics_closing_and_sync(self):
        x = torch.tensor([[0., 0.], [.25, .5], [2., 2.], [2.25, 2.5], [4., 4.], [4.25, 4.5], [4.5, 4.75]], device='cuda')
        initial = x[[0, 2, 4]].clone()
        for q in (None, CastQuantizer(2, torch.bfloat16)):
            rows = x if q else x.bfloat16()
            before = rows.clone()
            args = dict(niter=2, batch_size=3, alpha0=.5, seed=7, quantizer=q,
                        assignment_backend='cute', assignment_chunk_size=1, capture_assignments=True)
            matrix, caller = Emeans(2, 3, **args), Emeans(2, 3, **args)
            matrix.train(rows, initial)
            caller.begin(len(rows), initial)
            for epoch in range(2):
                order = e_means._randperm(len(rows), e_means._epoch_seed(7, epoch), True, rows.device)
                for begin in range(0, len(rows), 3):
                    batch = rows[order[begin:begin+3]]
                    represented = batch.bfloat16().float()
                    snapshot = caller.current_centroids().clone()
                    operands = snapshot.bfloat16().float()
                    truth = (represented[:, None]-operands[None]).square().sum(2)
                    labels = truth.argmin(1)
                    partial = caller.accumulate(batch[:1])
                    if len(batch) > 1:
                        self.assertIsNone(partial['sums'])
                        self.assertIsNone(partial['counts'])
                        torch.testing.assert_close(caller.current_centroids(), snapshot, rtol=0, atol=0)
                        stats = caller.accumulate(batch[1:])
                    else:
                        stats = partial
                    if epoch == caller.niter - 1:
                        torch.testing.assert_close(stats['labels'], labels[-len(stats['labels']):], rtol=0, atol=0)
                    else:
                        self.assertNotIn('labels', stats)
                    empty = caller.accumulate(batch[:0])
                    torch.testing.assert_close(empty['sums'], stats['sums'], rtol=0, atol=0)
                    torch.testing.assert_close(empty['counts'], stats['counts'], rtol=0, atol=0)
                    sums = torch.zeros_like(stats['sums'])
                    sums.index_add_(0, labels, represented)
                    # At most three small quarter-valued addends: bounded exact
                    # FP32 sums are independent of the atomic addition order.
                    torch.testing.assert_close(stats['sums'], sums, rtol=0, atol=0)
                    torch.testing.assert_close(stats['counts'], torch.bincount(labels, minlength=3).float(), rtol=0, atol=0)
                    torch.testing.assert_close(stats['objective'], truth.min(1).values.double().sum(), rtol=1e-6, atol=3e-5)
                    caller.update(stats)
                caller.end_epoch()
            counts, sums, state = caller._pass_counts, caller._pass_sums, caller._centroids
            closing = torch.where(counts[:, None] > 0, sums/counts.clamp_min(1).float()[:, None], state)
            torch.testing.assert_close(caller.finalize(), closing, rtol=0, atol=0)
            torch.testing.assert_close(caller.centroids, matrix.centroids, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(caller.obj, matrix.obj, rtol=1e-6, atol=3e-5)
            torch.testing.assert_close(rows, before, rtol=0, atol=0)
            saved = caller.centroids.clone()
            output = caller.centroids
            caller.reset()
            torch.testing.assert_close(output, saved, rtol=0, atol=0)
            warm = Emeans(2, 3, niter=1, batch_size=3, alpha0=.5, quantizer=q,
                          assignment_backend='cute', assignment_chunk_size=1)
            warm.begin(3, initial)
            warm.update(warm.accumulate(rows[:3]))
            warm.begin(3, initial)
            torch.cuda.synchronize()
            previous = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode('error')
                warm.update(warm.accumulate(rows[:3]))
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            warm.end_epoch()
            warm.finalize()
        x = torch.zeros(10, 17, device='cuda')
        x[:, 0] = torch.tensor([0., 1., 2., 3., 4., 10., 11., 12., 13., 14.], device='cuda')
        c = torch.zeros(3, 17, device='cuda')
        c[:, 0] = torch.tensor([0., 10., 100.], device='cuda')
        model = Emeans(17, 3, niter=2, batch_size=2, alpha0=.5, quantizer=CastQuantizer(17, torch.bfloat16), assignment_backend='cute')
        finalize = model.finalize
        retained = {}
        def close():
            counts, sums, state = model._pass_counts, model._pass_sums, model._centroids
            retained['closing'] = torch.where(counts[:, None] > 0, sums/counts.clamp_min(1).float()[:, None], state)
            return finalize()
        with mock.patch.object(model, 'finalize', close):
            model.train(x, c)
        self.assertGreater(model.n_splits, 0)
        torch.testing.assert_close(model.centroids, retained['closing'], rtol=0, atol=0)
        self.assertGreater(abs(model.obj[-1].item()-model.assign(x)[0].double().sum().item()), .001)
        model.centroids[0].fill_(100)
        model.centroids[1].zero_()
        self.assertEqual(model.assign(x[:1])[1].item(), 1)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_bf16_ordered_refresh_and_noncurrent_device(self):
        for dtype, offset in ((torch.bfloat16, 0), (torch.float8_e4m3fn, 0),
                              (torch.int8, 0), (torch.uint8, 128)):
            for k in ((2, 128) if dtype in (torch.int8, torch.uint8) else (2,)):
                with self.subTest(dtype=dtype, k=k):
                    from faiss.contrib.e_means.e_means_cute import _prepare_centroids, _assign_rows
                    current = torch.cuda.current_device()
                    device = torch.device('cuda', (current+1) % torch.cuda.device_count())
                    c = torch.full((k, 17), 100.+offset, device=device)
                    c[0].fill_(offset)
                    c[1].fill_(1+offset)
                    rows = torch.full((257, 17), offset, dtype=dtype, device=device)
                    model = known_torch_centroids(c, assignment_backend='cute')
                    first = model.assign(rows)
                    saved = tuple(t.clone() for t in first)
                    torch.cuda.synchronize(device)
                    producer, side = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
                    with torch.cuda.stream(producer):
                        torch.cuda._sleep(50_000_000)
                        rows.fill_(1+offset)
                    side.wait_stream(producer)
                    with torch.cuda.stream(side):
                        one = model.assign(rows)
                        model.centroids[0].fill_(1+offset)
                        model.centroids[1].fill_(offset)
                        changed = model.assign(rows)
                        model.assign(rows[:3])
                        model.assign(rows[:2])
                    torch.cuda.current_stream(device).wait_stream(side)
                    for actual, expected in zip(first, saved):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for pair, label in ((one, 1), (changed, 0)):
                        self.assertEqual((pair[0].device, pair[1].device), (device, device))
                        torch.testing.assert_close(pair[0], torch.zeros_like(pair[0]), rtol=0, atol=0)
                        self.assertTrue(bool((pair[1] == label).all()))
                    self.assertEqual(torch.cuda.current_device(), current)
                    if torch.cuda.device_count() >= 2:
                        with self.assertRaisesRegex(ValueError, 'same device'):
                            model.assign(rows.to(torch.device('cuda', current)))
                    prepared = _prepare_centroids(model.centroids, dtype=dtype)
                    with self.assertRaisesRegex(ValueError, 'contiguous CUDA BF16'):
                        _assign_rows(rows.float(), prepared)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_fp8_recalibration_after_training_matches_fresh_assignment(self):
        x = torch.linspace(-1, 1, 32, device='cuda')[:, None].expand(32, 128).contiguous()
        initial = x[[4, 12, 20, 28]].clone()
        q = Fp8Quantizer(128)
        q.train(x)
        model = Emeans(128, 4, niter=2, quantizer=q, assignment_backend='cute')
        model.train(q.compute_codes(x), initial)
        retained = model.assign(q.compute_codes(x[:9]))
        saved = tuple(value.clone() for value in retained)
        for offset in (0.0625, 0.125, 0.25):
            q.train(x + offset)
            rows = q.compute_codes(x[:9])
            actual = model.assign(rows)
            fresh = known_torch_centroids(model.centroids.clone(), quantizer=q, assignment_backend='cute')
            expected = fresh.assign(rows)
            for observed, reference in zip(actual, expected):
                torch.testing.assert_close(observed, reference, rtol=0, atol=0)
            for observed, reference in zip(retained, saved):
                torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_fp8_native_and_jit_share_decoded_state_and_output_units(self):
        x = torch.tensor([[0.], [2.], [10.], [12.]], device='cuda')
        q = Fp8Quantizer(1)
        q.train(x)
        models = [Emeans(1, 2, niter=1, batch_size=4, alpha0=.5,
                         quantizer=q, assignment_backend='cute') for _ in range(2)]
        for model,rows in zip(models, (q.compute_codes(x),x)):
            model.train(rows, x[[0,2]])
            torch.testing.assert_close(model.centroids,x.new_tensor([[1.],[11.]]))
            self.assertEqual(model.obj.item(),8.)
            distance,labels = model.assign(rows)
            torch.testing.assert_close(distance,torch.ones(4,device=x.device))
            torch.testing.assert_close(labels,torch.tensor([0,0,1,1],device=x.device))
        low, high = torch.full((5, 65), .1, device='cuda'), torch.full((5, 65), 2.3, device='cuda')
        x = torch.cat((low, high))
        q = Fp8Quantizer(65)
        q.train(x)
        self.assertTrue(q.centered)
        represented = q.decode(q.compute_codes(x))
        for rows in (q.compute_codes(x), x):
            model = Emeans(65, 2, niter=1, batch_size=10, alpha0=1., quantizer=q, assignment_backend='cute')
            model.train(rows, torch.stack((low[0], high[0])))
            torch.testing.assert_close(model.centroids, torch.stack((represented[:5].mean(0), represented[5:].mean(0))))
            distance, labels = model.assign(rows)
            self.assertEqual((model.centroids.dtype, distance.dtype, labels.dtype), (torch.float32, torch.float32, torch.int64))
            self.assertEqual(labels.tolist(), [0]*5+[1]*5)
            torch.testing.assert_close(distance, torch.zeros_like(distance))
            query = low[:1].clone()
            query[0, 0] += .125
            distance, labels = model.assign(query)
            self.assertEqual(labels.item(), 0)
            torch.testing.assert_close(distance, torch.tensor([.015625], device='cuda'))

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_fp8_seeded_chunks_selected_distance_and_explicit_validation(self):
        g = torch.Generator(device='cuda').manual_seed(31)
        x = torch.randn(256, 128, generator=g, device='cuda').mul_(.25)
        initial = torch.randn(512, 128, generator=g, device='cuda').mul_(.25)
        q = Fp8Quantizer(128)
        q.train(x)
        codes = q.compute_codes(x)
        before = codes.view(torch.uint8).clone()
        represented, operands = q.decode(codes).double(), q.decode(q.compute_codes(initial)).double()
        for chunk in (len(x), 64):
            model = known_torch_centroids(initial, quantizer=q, assignment_backend='cute',
                                          assignment_chunk_size=chunk)
            for rows in (x, codes):
                with self.subTest(chunk=chunk, native=rows is codes):
                    distance, labels = model.assign(rows)
                    self.assertEqual((distance.shape, labels.shape), ((len(x),), (len(x),)))
                    self.assertTrue(bool(((labels >= 0) & (labels < len(initial))).all()))
                    selected = (represented - operands[labels]).square().sum(1)
                    torch.testing.assert_close(distance.double(), selected, rtol=2e-6, atol=1e-3)
                    saved = tuple(value.clone() for value in (distance, labels))
                    model.assign(rows[:3])
                    for observed, reference in zip((distance, labels), saved):
                        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        torch.testing.assert_close(codes.view(torch.uint8), before, rtol=0, atol=0)
        for position, value in ((0, 1000.), (3, 1000.), (5, 1000.),
                                (2, float('nan')), (4, float('inf')), (1, float('-inf'))):
            query = torch.zeros(6, 128, device='cuda')
            query[position, 0] = value
            with self.subTest(position=position, value=value), self.assertRaisesRegex(ValueError, 'representable range'):
                q.validate(query)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_fp8_decoded_caller_tail_revival_and_closing(self):
        x = torch.tensor([[0., 0.], [.25, .5], [2., 2.], [2.25, 2.5],
                          [4., 4.], [4.25, 4.5], [4.5, 4.75]], device='cuda')
        initial = torch.tensor([[0., 0.], [2., 2.], [100., 100.]], device='cuda')
        q = Fp8Quantizer(2)
        q.train(torch.tensor([[-128., -128.], [128., 128.]], device='cuda').repeat(1024, 1))
        self.assertFalse(q.centered)
        codes = q.compute_codes(x)
        before = codes.view(torch.uint8).clone()
        self.assertTrue(bool((codes.float().abs().sum(0)*512 < 2**24).all()))
        for incoming in (codes, x):
            with self.subTest(native=incoming is codes):
                args = dict(niter=2, batch_size=3, alpha0=.5, seed=7, quantizer=q,
                            assignment_backend='cute', assignment_chunk_size=2, capture_assignments=True)
                matrix, caller = Emeans(2, 3, **args), Emeans(2, 3, **args)
                matrix.train(incoming, initial)
                caller.begin(len(x), initial)
                for epoch in range(2):
                    order = e_means._randperm(len(x), e_means._epoch_seed(7, epoch), True, x.device)
                    for start in range(0, len(x), 3):
                        ids = order[start:start+3]
                        represented = q.decode(codes.float()[ids])
                        snapshot = caller.current_centroids().clone()
                        operands = q.decode(q.compute_codes(snapshot))
                        truth = (represented[:, None]-operands[None]).square().sum(2)
                        labels = truth.argmin(1)
                        partial = caller.accumulate(incoming[ids[:1]])
                        if len(ids) > 1:
                            self.assertIsNone(partial['sums'])
                            self.assertIsNone(partial['counts'])
                            torch.testing.assert_close(caller.current_centroids(), snapshot, rtol=0, atol=0)
                            stats = caller.accumulate(incoming[ids[1:]])
                        else:
                            stats = partial
                        if epoch == caller.niter - 1:
                            torch.testing.assert_close(stats['labels'], labels[-len(stats['labels']):], rtol=0, atol=0)
                        else:
                            self.assertNotIn('labels', stats)
                        empty = caller.accumulate(incoming[ids[:0]])
                        torch.testing.assert_close(empty['sums'], stats['sums'], rtol=0, atol=0)
                        torch.testing.assert_close(empty['counts'], stats['counts'], rtol=0, atol=0)
                        sums = torch.zeros_like(stats['sums'])
                        sums.index_add_(0, labels, represented)
                        # At most three bounded dyadic addends, uncentered
                        # normal decode: exact totals do not depend on order.
                        torch.testing.assert_close(stats['sums'], sums, rtol=0, atol=0)
                        torch.testing.assert_close(stats['counts'], torch.bincount(labels, minlength=3).float(), rtol=0, atol=0)
                        torch.testing.assert_close(stats['objective'], truth.min(1).values.double().sum(), rtol=1e-6, atol=3e-5)
                        caller.update(stats)
                    caller.end_epoch()
                counts, sums, state = caller._pass_counts, caller._pass_sums, caller._centroids
                closing = torch.where(counts[:, None] > 0, sums/counts.clamp_min(1).float()[:, None], state)
                torch.testing.assert_close(caller.finalize(), closing, rtol=0, atol=0)
                torch.testing.assert_close(caller.centroids, matrix.centroids, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(caller.obj, matrix.obj, rtol=1e-6, atol=3e-5)
                self.assertGreater(caller.n_splits, 0)
        torch.testing.assert_close(codes.view(torch.uint8), before, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_fp8_seed_decode_and_independent_closing_distortion(self):
        x = torch.tensor([[-1.625, -1.875], [-.375, .125], [.375, .625], [1.125, 1.375],
                          [1.625, 1.875], [.125, .375], [-.625, -.875], [.75, 1.25], [-1.25, -1.75]], device='cuda')
        for centered, all_rows in ((False, False), (True, False), (True, True)):
            rows = (x + 10 if centered else x)[:2 if all_rows else len(x)]
            q = Fp8Quantizer(2)
            q.train(rows.cpu() if centered else torch.tensor([[-3., -3.], [3., 3.]]).repeat(2048, 1))
            q.to(rows.device)
            self.assertEqual(q.centered, centered)
            self.assertNotEqual(q.scale, 1.)
            codes = q.compute_codes(rows)
            before = codes.view(torch.uint8).clone()
            indices = torch.arange(len(rows), device=rows.device) if all_rows else e_means._randperm(len(rows), 7, True, rows.device)[:2]
            initial = q.decode(codes.float()[indices])
            args = dict(niter=2, batch_size=3, alpha0=.5, seed=7, quantizer=q,
                        assignment_backend='cute', assignment_chunk_size=2)
            native, jit, explicit = (Emeans(2, 2, **args) for _ in range(3))
            for model, incoming in ((native, codes), (jit, rows)):
                with mock.patch.object(model, 'begin', wraps=model.begin) as begin:
                    model.train(incoming)
                torch.testing.assert_close(begin.call_args.args[1], initial, rtol=0, atol=0)
            explicit.train(rows, initial)
            for model in (native, explicit):
                torch.testing.assert_close(model.centroids, jit.centroids, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(model.obj, jit.obj, rtol=1e-6, atol=3e-5)
            torch.testing.assert_close(codes.view(torch.uint8), before, rtol=0, atol=0)
        d, k = 1024, 4
        g = torch.Generator(device='cuda').manual_seed(1821)
        base = (torch.arange(d, device='cuda') % 31 + 128).float()
        initial = torch.stack([base + 32*i for i in range(k)]).to(torch.float8_e4m3fn).float()
        rows = (initial[torch.arange(128, device='cuda') % k] + torch.randint(-8, 9, (128, d), generator=g, device='cuda')).to(torch.float8_e4m3fn)
        scores = []
        for backend in ('triton', 'cute'):
            model = Emeans(d, k, niter=2, seed=1234, assignment_backend=backend)
            model.train(rows, initial)
            self.assertTrue(bool(torch.isfinite(model.centroids).all()))
            # Returned closing centroids stay FP32: this fresh oracle does not
            # call assign or compare the moving-pass training objective.
            direct = (rows.double()[:, None] - model.centroids.double()[None]).square().sum(2)
            scores.append(direct.min(1).values)
        torch.testing.assert_close(scores[1], scores[0], rtol=1e-5, atol=1e-3)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_bytes_integer_oracles_destinations_and_refresh(self):
        from faiss.contrib.e_means.e_means_cute import _prepare_centroids, _assign_rows
        for dtype, offset in ((torch.int8, 0), (torch.uint8, 128)):
            for n, k, d in ((2, 3, 1), (5, 385, 79), (3, 257, 257),
                            (2, 384, 343), (2, 384, 344), (1, 2, 43919)):
                with self.subTest(dtype=dtype, n=n, k=k, d=d):
                    g = torch.Generator(device='cuda').manual_seed(n+k+d)
                    signed = torch.randint(-128, 128, (n, d), generator=g, device='cuda')
                    c = torch.randint(-128, 128, (k, d), generator=g, device='cuda').float()
                    if d == 79:
                        signed[0].zero_()
                        signed[1].fill_(127)
                        c[0].zero_()
                        c[192].zero_()
                        c[-1].fill_(127)
                    elif d in (343, 344):
                        signed.fill_(127)
                        c.zero_()
                        c[0].fill_(-128)
                        c[1].fill_(127)
                    elif d == 43919:
                        signed.fill_(127)
                        c.fill_(-128)
                    rows = (signed+offset).to(dtype)
                    c.add_(offset)
                    before_rows, before_c = rows.clone(), c.clone()
                    model = known_torch_centroids(c, assignment_backend='cute')
                    owned = model.assign(rows)
                    saved = tuple(t.clone() for t in owned)
                    truth = (rows.cpu().long()[:, None]-c.cpu().long()[None]).square().sum(2)
                    expected = (truth.min(1).values.float(), truth.argmin(1))
                    for actual, wanted in zip(owned, expected):
                        torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                    if d == 79:
                        self.assertEqual(owned[1][:2].tolist(), [0, k-1])
                    elif d in (343, 344):
                        self.assertEqual(owned[1].tolist(), [1, 1])
                    elif d == 43919:
                        self.assertEqual((truth[0, 0].item(), owned[1].item()), (2855832975, 0))
                    distance = torch.full((n+2,), -123., device='cuda')
                    labels = torch.full((n+2,), -123, dtype=torch.int64, device='cuda')
                    prepared = _prepare_centroids(model.centroids, dtype=dtype)
                    out = _assign_rows(rows, prepared,
                                       distance_out=distance[1:-1], labels_out=labels[1:-1])
                    for actual, wanted in zip(out, expected):
                        torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                    self.assertEqual((out[0].data_ptr(), out[1].data_ptr()),
                                     (distance[1:-1].data_ptr(), labels[1:-1].data_ptr()))
                    self.assertEqual((distance[[0, -1]].tolist(), labels[[0, -1]].tolist()),
                                     ([-123., -123.], [-123, -123]))
                    caller_before = (distance.clone(), labels.clone())
                    changed_rows = rows.clone()
                    changed_rows[0].fill_(offset+1)
                    changed_truth = (changed_rows.cpu().long()[:, None]-c.cpu().long()[None]).square().sum(2)
                    for actual, wanted in zip(
                        _assign_rows(changed_rows, prepared),
                        (
                            changed_truth.min(1).values.float(),
                            changed_truth.argmin(1),
                        ),
                    ):
                        torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                    for actual, wanted in zip(
                        _assign_rows(rows[:1], prepared),
                        (expected[0][:1], expected[1][:1]),
                    ):
                        torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                    backing = torch.empty(n, d*2, dtype=dtype, device='cuda')
                    backing[:, ::2].copy_(rows)
                    for actual, wanted in zip(model.assign(backing[:, ::2]), expected):
                        torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                    for actual, wanted in zip(owned, saved):
                        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                    torch.testing.assert_close(rows, before_rows, rtol=0, atol=0)
                    torch.testing.assert_close(c, before_c, rtol=0, atol=0)
                    if d == 79:
                        for _ in range(2):
                            model.centroids[0, 0] = float('nan')
                            with self.assertRaisesRegex(ValueError, 'finite'):
                                model.assign(rows)
                        model.centroids.copy_(before_c)
                        for actual, wanted in zip(model.assign(rows), expected):
                            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                    for actual, wanted in zip((distance, labels), caller_before):
                        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            # One-unit score gaps above FP32's exact-integer range, with both
            # zero and high-dot queries. No implementation/dispatch assertion.
            c = torch.full((2, 1537), 127., device='cuda')
            c[:, 0] = torch.tensor([1., 0.], device='cuda')
            model = known_torch_centroids(c+offset, assignment_backend='cute')
            distance, labels = model.assign(torch.full((1, 1537), offset, dtype=dtype, device='cuda'))
            self.assertEqual((distance.item(), labels.item()), (float(1536*127**2), 1))
            model.centroids.fill_(127+offset)
            model.centroids[0, 0] = 126+offset
            distance, labels = model.assign(torch.full((1, 1537), 127+offset, dtype=dtype, device='cuda'))
            self.assertEqual((distance.item(), labels.item()), (0., 1))
            model.centroids[1, 0] = 125+offset
            distance, labels = model.assign(torch.full((1, 1537), 127+offset, dtype=dtype, device='cuda'))
            self.assertEqual((distance.item(), labels.item()), (1., 0))
            # Direct prepared norms are the minimal arithmetic observation;
            # their object/cache identity is not a public contract.
            c = torch.zeros(1, 257, device='cuda')+offset
            c[0, :6] = torch.tensor([-1.5, -.5, .5, 1.5, 200., -200.], device='cuda')+offset
            wanted = torch.zeros(1, 257, dtype=torch.int64, device='cuda')
            wanted[0, :6] = torch.tensor([-2, 0, 0, 2, 127, -128], device='cuda')
            model = known_torch_centroids(c, assignment_backend='cute')
            for changed in (False, True):
                if changed:
                    model.centroids.fill_(offset)
                    wanted.zero_()
                prepared = _prepare_centroids(model.centroids, dtype=dtype)
                torch.testing.assert_close(prepared.centroids[:, :257].long(), wanted, rtol=0, atol=0)
                torch.testing.assert_close(prepared.norms[:1].long(), wanted.square().sum(1), rtol=0, atol=0)
                query = torch.full((1, 257), offset, dtype=dtype, device='cuda')
                distance, labels = model.assign(query)
                self.assertEqual((distance.item(), labels.item()), (float(wanted.square().sum()), 0))

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_bytes_stress_and_high_integer_ids(self):
        from faiss.contrib.e_means.e_means_cute import _prepare_centroids, _assign_rows
        for dtype, offset in ((torch.int8, 0), (torch.uint8, 128)):
            with self.subTest(dtype=dtype):
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                g = torch.Generator(device='cuda').manual_seed(32768)
                c = torch.randint(-128, 128, (16384, 128), generator=g, device='cuda').float()+offset
                c[128].copy_(c[0])
                rows = torch.full((32768, 128), offset, dtype=dtype, device='cuda')
                rows[:4].copy_((torch.randint(-128, 128, (4, 128), generator=g, device='cuda')+offset).to(dtype))
                rows[0].copy_(c[0])
                model = known_torch_centroids(c, assignment_backend='cute', assignment_chunk_size=32768)
                distance, labels = model.assign(rows)
                probes = (rows[:4].cpu().long()[:, None]-c.cpu().long()[None]).square().sum(2)
                torch.testing.assert_close(labels[:4].cpu(), probes.argmin(1), rtol=0, atol=0)
                torch.testing.assert_close(distance[:4].cpu(), probes.min(1).values.float(), rtol=0, atol=0)
                zero = (c.cpu().long()-offset).square().sum(1)
                self.assertTrue(bool((labels[4:] == zero.argmin()).all()))
                torch.testing.assert_close(distance[4:], zero.min().float().to(distance.device).expand_as(distance[4:]), rtol=0, atol=0)
                self.assertEqual(labels[0].item(), 0)
                c = torch.full((65537, 1), offset, dtype=torch.float32, device='cuda')
                c[-1].fill_(offset+2)
                sentinels = torch.full((4,), -123, dtype=torch.int64, device='cuda')
                distance, labels = _assign_rows(
                    torch.full((2, 1), offset+2, dtype=dtype, device='cuda'),
                    _prepare_centroids(c, dtype=dtype),
                    labels_out=sentinels[1:-1],
                )
                self.assertEqual(sentinels.tolist(), [-123, 65536, 65536, -123])
                torch.testing.assert_close(distance, torch.zeros_like(distance), rtol=0, atol=0)
                print('CuTe integer stress cost', dtype, time.monotonic()-start,
                      'seconds, peak', torch.cuda.max_memory_allocated(), 'bytes')

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_bytes_revival_and_raw_code_workflow(self):
        for dtype, value in ((torch.int8, 0), (torch.uint8, 128)):
            rows = torch.full((10, 17), value, dtype=dtype, device='cuda')
            model = Emeans(17, 2, niter=5, batch_size=10, alpha0=.5,
                           assignment_chunk_size=4, assignment_backend='cute')
            centroids, labels = model.train(rows, rows[:2].float())
            self.assertIs(centroids, model.centroids)
            self.assertIs(labels, model.assignments)
            self.assertEqual(model.obj[-1], 0.)
            self.assertEqual(model.n_splits, 1)
            children = model.centroids.round().clamp(value-128, value+127).to(dtype)
            self.assertEqual(model.assign(children)[1].tolist(), [0, 1])
        g = torch.Generator(device='cuda').manual_seed(7117)
        source = torch.randn(32, 80, generator=g, device='cuda')
        q = Int8Quantizer(80)
        q.train(source[:16])
        codes = q.compute_codes(source)
        model = Emeans(80, 4, niter=1, batch_size=8, alpha0=.5, assignment_backend='cute')
        model.train(codes, codes[:4].float())  # q deliberately NOT supplied.
        distance, labels = model.assign(codes)
        represented = model.centroids.round().clamp(-128, 127).cpu().long()
        truth = (codes.cpu().long()[:, None]-represented[None]).square().sum(2)
        torch.testing.assert_close(labels.cpu(), truth.argmin(1), rtol=0, atol=0)
        torch.testing.assert_close(distance.cpu(), truth.min(1).values.float(), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(q.decode(model.centroids)).all()))
        source = torch.tensor([[-.25, .25], [1., 2.], [3., 4.]], device='cuda')
        q = Int8Quantizer(2)
        q.train(source)
        codes = q.compute_codes(source)
        decoded = q.decode(codes)
        for k in (2, 128):
            initial = decoded[0].expand(k, 2).clone()
            initial[1] = decoded[2]
            results = [known_torch_centroids(initial, quantizer=q, assignment_backend=backend).assign(codes)
                       for backend in ('triton', 'cute')]
            for actual, wanted in zip(results[1], results[0]):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_bytes_and_caller_quantizer_use_decoded_units(self):
        x = torch.tensor([[0.],[2.],[10.],[12.]],device='cuda')
        q = Int8Quantizer(1)
        q.train(x)
        for rows,quantizer,offset in (
            (x.to(torch.int8),None,0),
            ((x+128).to(torch.uint8),None,128),
            (q.compute_codes(x),q,0),
        ):
            with self.subTest(dtype=rows.dtype,calibrated=quantizer is not None):
                model = Emeans(1,2,niter=1,batch_size=4,alpha0=.5,
                               quantizer=quantizer,assignment_backend='cute')
                model.train(rows,(x+offset)[[0,2]])
                torch.testing.assert_close(model.centroids,x.new_tensor([[1.],[11.]])+offset)
                self.assertEqual(model.obj.item(),8.)
                distance,labels = model.assign(rows)
                torch.testing.assert_close(distance,torch.ones(4,device=x.device))
                torch.testing.assert_close(labels,torch.tensor([0,0,1,1],device=x.device))
        with self.assertRaisesRegex(TypeError,'no JIT INT8'):
            Emeans(1,2,quantizer=q,assignment_backend='cute').train(x)


if __name__ == "__main__":
    unittest.main()
