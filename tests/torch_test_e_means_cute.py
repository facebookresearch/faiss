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

from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.e_means_torch import Emeans
from faiss.contrib.e_means import torch_backend
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer
from faiss.contrib.e_means import e_means_torch as e_means
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
    def test_quack_helper_accepts_three_positional_operands(self):
        from faiss.contrib.e_means import cute

        def renamed(copy, source, thread):
            raise AssertionError('capability check must not execute the helper')

        def incompatible(copy, source, thread, required):
            raise AssertionError('capability check must not execute the helper')

        versions = {'quack-kernels': '0.6.5', 'nvidia-cutlass-dsl': '4.7.1',
                    'cuda-python': '13.4.1'}
        for helper, expected in ((renamed, True), (incompatible, False)):
            modules = {
                'cuda.bindings.driver': SimpleNamespace(CUstream=object),
                'quack.epilogue.ops': SimpleNamespace(_lane_warp_info_n=helper),
                'quack.gemm_sm90': SimpleNamespace(GemmSm90=object),
                'quack.tile_scheduler': SimpleNamespace(TileScheduler=object),
                'quack.gemm_tvm_ffi_utils': SimpleNamespace(compile_gemm_kernel=object),
            }
            cute._stack_status.cache_clear()
            try:
                with mock.patch.object(cute.importlib.metadata, 'version', side_effect=versions.__getitem__), \
                     mock.patch.object(cute.importlib.util, 'find_spec', return_value=object()), \
                     mock.patch.object(cute.importlib, 'import_module', side_effect=modules.__getitem__):
                    self.assertEqual(cute._stack_status()[0], expected)
            finally:
                cute._stack_status.cache_clear()

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_fractional_training_reuses_dynamic_decoder(self):
        from collections import OrderedDict
        from faiss.contrib.e_means import cute_argmin, cute_kernel

        rows = torch.zeros((31, 256), dtype=torch.bfloat16, device='cuda')
        rows[:, 0] = torch.arange(31, device='cuda').float() / 4
        initial = rows[[0, 10, 20, 30]].float()
        compile_kernel, prepare = cute_argmin.cute.compile, cute_kernel.prepare_argmin_decode
        compiled, sizes, handles, counts, finals = [], [], [], [], []

        def compile(operation, *args, **kwargs):
            result = compile_kernel(operation, *args, **kwargs)
            if isinstance(operation, cute_argmin._DecodeArgminOutputsSm90):
                compiled.append(result)
            return result

        def bind(keys, codes, scores):
            result = prepare(keys, codes, scores)
            sizes.append(keys.numel())
            handles.append(result.compiled)
            return result

        with mock.patch.object(cute_argmin, '_COMPILED', OrderedDict()), \
             mock.patch.object(cute_argmin.cute, 'compile', new=compile), \
             mock.patch.object(cute_kernel, 'prepare_argmin_decode', new=bind):
            for _ in range(2):
                before = len(compiled)
                model = Emeans(256, 4, nepochs=1.25, batch_size=7, alpha0=.2, seed=7,
                               assignment_backend='cute', capture_assignments=True)
                model.train(rows, initial)
                counts.append(len(compiled) - before)
                order = torch.randperm(31, device=rows.device, generator=e_means._torch_generator(
                    rows.device, e_means._epoch_seed(7, 1)))
                labels = model.assignments[order]
                self.assertTrue(bool(((labels >= 0) & (labels < 4)).all()))
                for label in range(4):
                    selected = rows[order][labels == label].float()
                    if len(selected):
                        torch.testing.assert_close(model.centroids[label], selected.double().mean(0).float(),
                                                   atol=1e-6, rtol=1e-6)
                finals.append(model.centroids.clone())
        self.assertEqual(counts, [1, 0])
        self.assertEqual(set(sizes), {1, 3, 7})
        self.assertTrue(all(handle is handles[0] for handle in handles))
        torch.testing.assert_close(finals[0], finals[1], atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_decoder_bits_tail_reset_and_retained_graph(self):
        from collections import OrderedDict
        from faiss.contrib.e_means import cute_argmin as decode

        score_pattern = torch.tensor([-float('inf'), -2.5, 0., 2**-149, .25, float('inf')], device='cuda')
        label_pattern = torch.tensor([0, 17, 2**31 - 1], device='cuda', dtype=torch.int64)
        retained = []
        with mock.patch.object(decode, '_COMPILED', OrderedDict()):
            for dtype in (torch.int32, torch.int64):
                for n in (1, 257):
                    scores = score_pattern[torch.arange(n, device='cuda') % len(score_pattern)]
                    labels = label_pattern[torch.arange(n, device='cuda') % len(label_pattern)]
                    if n > 5:
                        labels[5] = 0  # Existing Inf row is the real neutral key.
                    bits = scores.view(torch.int32).long() & 0xFFFFFFFF
                    ordered = torch.where(bits >> 31 != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
                    payload = ((ordered - 0x80000000) << 32) | labels
                    if n > 5:
                        self.assertEqual(payload[5].item(), decode.INITIAL_KEY)
                    owners = (torch.full((n + 2,), -123, dtype=torch.int64, device='cuda'),
                              torch.full((n + 2,), -123, dtype=dtype, device='cuda'),
                              torch.full((n + 2,), -123., device='cuda'))
                    key, code, score = (owner[1:-1] for owner in owners)
                    key.copy_(payload)
                    bound = decode.prepare_argmin_decode(key, code, score)
                    retained.append((bound, payload, labels, scores))
                    decode.launch_argmin_decode(bound)
                    torch.testing.assert_close(code.long(), labels, rtol=0, atol=0)
                    torch.testing.assert_close(score.view(torch.int32), scores.view(torch.int32), rtol=0, atol=0)
                    self.assertTrue(bool((key == decode.INITIAL_KEY).all()))
                    for owner in owners:
                        self.assertEqual(owner[[0, -1]].tolist(), [-123, -123])
                self.assertIs(retained[-1][0].compiled, retained[-2][0].compiled)
            self.assertIsNot(retained[0][0].compiled, retained[2][0].compiled)
            bound, payload, labels, scores = retained[1]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph, stream=stream):
                        decode.launch_argmin_decode(bound)
                    decode._COMPILED.clear()
                    for shift in (0, 1):
                        bound.keys.copy_(payload.roll(shift))
                        graph.replay()
                        stream.synchronize()
                        torch.testing.assert_close(bound.codes.long(), labels.roll(shift), rtol=0, atol=0)
                        torch.testing.assert_close(bound.scores.view(torch.int32),
                                                   scores.roll(shift).view(torch.int32), rtol=0, atol=0)
                        self.assertTrue(bool((bound.keys == decode.INITIAL_KEY).all()))
            finally:
                stream.synchronize()
                graph.reset()
                torch.cuda.current_stream().wait_stream(stream)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_spherical_natural_grouped_route_and_represented_statistics(self):
        from faiss.contrib.e_means import cute_grouped

        centers = torch.zeros((32768, 17), device='cuda')
        centers[:, 0] = -1
        centers[5, 0] = centers[133, 0] = 1
        centers[-1].zero_()
        centers[-1, 1] = 1
        centers[257].zero_()
        centers[257, :3] = 1 / (3**.5)
        roster = torch.zeros((6, 17), device='cuda', dtype=torch.bfloat16)
        roster[0, 0] = roster[1, 1] = 1
        roster[2, 0] = roster[4, 1] = -1
        roster[5, :3] = 1 / (3**.5)
        membership = torch.arange(65536, device='cuda') % len(roster)
        rows = roster[membership]
        # Rounded unit directions can have IP>1; real model statistics must
        # retain the resulting negative spherical objective, not clamp it.
        ip = torch.tensor([1., 1., 1., 0., 0., 1.002685546875], device='cuda')[membership]
        ids = torch.tensor([5, 32767, 0, 0, 0, 257], device='cuda')[membership]
        model = Emeans(17, 32768, nepochs=1, batch_size=len(rows), alpha0=1,
                       assignment_backend='cute', assignment_chunk_size=len(rows), spherical=True)
        model.begin(len(rows), centers)
        snapshot = model.current_centroids().clone()
        selected = []
        prepare = cute_grouped.prepare_grouped

        def observe(*args, **kwargs):
            selected.append((len(args[0]), args[1].spherical))
            return prepare(*args, **kwargs)

        with mock.patch.object(cute_grouped, 'prepare_grouped', new=observe):
            stats = model.accumulate(rows)
        self.assertEqual(selected, [(len(rows), True)])
        counts = torch.bincount(ids, minlength=model.k).float()
        sums = torch.zeros_like(centers).index_add_(0, ids, rows.float())
        torch.testing.assert_close(stats['labels'], ids, rtol=0, atol=0)
        torch.testing.assert_close(stats['counts'], counts, rtol=0, atol=0)
        torch.testing.assert_close(stats['sums'], sums, rtol=0, atol=0)
        torch.testing.assert_close(stats['objective'], (2 - 2*ip).double().sum(), rtol=0, atol=0)
        model.update(stats)
        model.end_epoch()
        final = model.finalize()
        wanted = torch.where((counts > 0)[:, None], torch.nn.functional.normalize(sums, dim=1), snapshot)
        torch.testing.assert_close(final, wanted, rtol=2e-6, atol=2e-7)
        # Restore the initial directions to check actual public similarity
        # conversion independently of the already-checked closing means.
        model.centroids.copy_(snapshot)
        similarity, labels = model.assign(rows)
        torch.testing.assert_close(similarity, ip, rtol=0, atol=0)
        torch.testing.assert_close(labels, ids, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_spherical_grouped_fp64_oracle_and_captured_refresh(self):
        from faiss.contrib.e_means import cute

        rng = torch.Generator().manual_seed(1417)
        rows = torch.randn(129, 17, generator=rng)
        centers = torch.randn(256, 17, generator=rng)
        rows = torch.nn.functional.normalize(rows, dim=1).to(device='cuda', dtype=torch.bfloat16)
        centers = torch.nn.functional.normalize(centers, dim=1).cuda()
        prepared = cute._prepare_centroids(centers, spherical=True)
        # Explicit legal leaf geometry: this is NOT a small-work auto-route claim.
        bound = cute._new_grouped_binding(len(rows), prepared, torch.float32, plan=(128, 64, 2))

        def launch():
            bound.workspace.copy_(rows)
            bound.launcher(bound.kernel, bound.labels)

        launch()
        truth = rows.double() @ prepared.centroids.double().T
        absolute = rows.double().abs() @ prepared.centroids.double().abs().T
        gamma = 17 * 2**-24 / (1 - 17 * 2**-24)
        error, regret = gamma * absolute.max(1).values, 4 * gamma * absolute.max(1).values
        chosen = truth[torch.arange(len(rows), device='cuda'), bound.labels]
        best, wanted = truth.max(1)
        self.assertTrue(bool(((-.5*bound.scores - chosen).abs() <= error).all()))
        self.assertTrue(bool((best - chosen <= regret).all()))
        margin = truth.topk(2, dim=1).values
        far = margin[:, 0] - margin[:, 1] > regret
        torch.testing.assert_close(bound.labels[far], wanted[far], rtol=0, atol=0)
        rows.zero_()
        rows[:, 0] = 1
        centers.zero_()
        centers[:, 0] = -1
        cute._prepare_centroids(centers, cached=prepared, spherical=True)
        launch()
        torch.testing.assert_close(bound.scores, torch.full_like(bound.scores, 2), rtol=0, atol=0)
        torch.testing.assert_close(bound.labels, torch.zeros_like(bound.labels), rtol=0, atol=0)
        centers[7].zero_()  # A zero direction must beat all negative similarities.
        cute._prepare_centroids(centers, cached=prepared, spherical=True)
        launch()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    prepared.source_version = None
                    cute._prepare_centroids(centers, cached=prepared, spherical=True)
                    launch()
                centers[7, 0] = -1
                centers[9].zero_()
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(bound.scores, torch.zeros_like(bound.scores), rtol=0, atol=0)
                torch.testing.assert_close(bound.labels, torch.full_like(bound.labels, 9), rtol=0, atol=0)
                self.assertEqual(prepared.norms.count_nonzero().item(), 0)
        finally:
            stream.synchronize()
            graph.reset()
            torch.cuda.current_stream().wait_stream(stream)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_native_byte_exact_distance_output(self):
        from faiss.contrib.e_means.cute import _prepare_centroids, _assign_rows

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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_captured_binding_survives_eviction_and_retires(self):
        rows = (torch.arange(10247, device='cuda') % 2).bfloat16()
        rows = rows[:, None].expand(-1, 16).contiguous()
        initial = torch.tensor([0., 1.], device='cuda')[:, None].expand(-1, 16).contiguous()
        capture_step = Emeans._capture_step
        end_epoch = Emeans.end_epoch

        for fail_after_tail in (False, True):
            with self.subTest(exception=fail_after_tail):
                owners = []
                graph_owner = None
                prepared_owner = None

                def capture(
                    model,
                    *args,
                    _capture_step=capture_step,
                    _owners=owners,
                    **kwargs,
                ):
                    nonlocal graph_owner, prepared_owner
                    captured = _capture_step(model, *args, **kwargs)
                    prepared_owner = weakref.ref(captured[1])
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
                    self.assertEqual(model.recipe['epoch_rows'], [len(rows), 17, len(rows)])
                    gc.collect()
                    self.assertTrue(_owners)
                    self.assertTrue(all(owner() is not None for owner in _owners))
                    self.assertIs(model._execution_backend.cache, prepared_owner())  # noqa: B023
                    if model._epoch == 2:
                        # The17-row middle pass evicts fullB, but graph owners
                        # keep that binding alive for the final full pass.
                        self.assertNotIn(model.recipe['batch_size'], prepared_owner().bindings)  # noqa: B023
                    if _fail_after_tail and model._epoch == len(model.recipe['epoch_rows']):
                        raise RuntimeError('injected post-tail failure')

                model = Emeans(
                    16,
                    2,
                    nepochs=2 + 17 / len(rows),
                    seed=10,
                    assignment_backend='cute',
                    capture_assignments=True,
                )
                with mock.patch.object(
                    Emeans, '_capture_step', capture
                ), mock.patch.object(
                    Emeans, 'end_epoch', finish_epoch
                ):
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
                del model
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))
                self.assertIsNone(prepared_owner())

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_native_and_jit_training(self):
        x = torch.tensor([[0.], [2.], [10.], [12.]], device='cuda')
        initial = x[[0, 2]]
        native = Emeans(
            1,
            2,
            nepochs=1,
            batch_size=4,
            alpha0=0.5,
            assignment_backend='cute',
            capture_assignments=True,
        )
        jit = QuantizedEmeans(
            1,
            2,
            nepochs=1,
            batch_size=4,
            alpha0=0.5,
            assignment_backend='cute',
            quantizer=CastQuantizer(1, torch.bfloat16),
            capture_assignments=True,
        )
        native.train(x.bfloat16(), initial)
        jit.train(x, initial)
        for model in (native, jit):
            torch.testing.assert_close(model.centroids, torch.tensor([[1.], [11.]], device='cuda'))
            self.assertEqual(model.obj.item(), 8.)
            torch.testing.assert_close(model.assignments, torch.tensor([0, 0, 1, 1], device='cuda'), rtol=0, atol=0)
        distance, labels = native.assign(x.bfloat16())
        torch.testing.assert_close(distance, torch.ones(4, device='cuda'))
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], device='cuda'))
        for query in (x, x.bfloat16()):
            cast_distance, cast_labels = jit.assign(query)
            self.assertEqual(jit.assignment_recipe['representation'], 'bfloat16')
            torch.testing.assert_close(cast_distance, distance, rtol=0, atol=0)
            torch.testing.assert_close(cast_labels, labels, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            QuantizedEmeans(1, 2, quantizer=None, assignment_backend='cute').train(x, initial)

    def test_optional_stack_and_device_scope(self):
        # CUDA initialization may probe optional packages: do it before the
        # missing-package blocker. Native use must still avoid CuTe imports.
        code = """
import importlib.abc, pathlib, sys, torch
import faiss.contrib
faiss.contrib.__path__.insert(0, str(pathlib.Path(sys.argv[1]).resolve().parents[1]))
from faiss.contrib.e_means.e_means_torch import Emeans
assert pathlib.Path(sys.modules[Emeans.__module__].__file__).resolve() == pathlib.Path(sys.argv[1]).resolve()
cuda = torch.cuda.is_available() and not torch.version.hip
if cuda: torch.cuda.get_device_capability()
class NoCute(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('quack', 'cutlass'): raise ImportError('blocked optional CuTe')
sys.meta_path.insert(0, NoCute())
Emeans(1, 1, nepochs=1, batch_size=2, alpha0=.5).train(torch.zeros(2, 1))
assert not any(n.split('.')[0] in ('quack', 'cutlass') for n in sys.modules)
if cuda:
    try:
        Emeans(1, 1, assignment_backend='cute').train(
            torch.zeros(2, 1, dtype=torch.bfloat16, device='cuda'))
    except RuntimeError as error: assert 'CuTe assignment requires' in str(error)
    else: raise AssertionError('missing optional stack silently accepted')
"""
        child = subprocess.run([sys.executable, '-c', code, str(e_means.__file__)],
                               text=True, capture_output=True, timeout=60)
        self.assertEqual(child.returncode, 0, child.stderr)
        from faiss.contrib.e_means import cute as backend
        from faiss.contrib.e_means import e_means_grouping as grouping
        with mock.patch.object(backend, '_stack_status', return_value=(True, 'ready')), \
             mock.patch.object(grouping, '_grouping_provider',
                               side_effect=ImportError('broken CCCL')) as provider:
            backend._require_stack()
            backend._require_stack()
        provider.assert_not_called()

        model = Emeans(16, 65536, assignment_backend='auto')
        rows = SimpleNamespace(device=torch.device('cuda', 17), dtype=torch.bfloat16)
        with mock.patch.object(torch.version, 'hip', None), \
             mock.patch.object(backend, '_stack_status', return_value=(True, 'ready')), \
             mock.patch.object(torch.cuda, 'device', return_value=nullcontext()), \
             mock.patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
             mock.patch.object(torch.cuda, 'get_device_properties',
                               return_value=mock.Mock(shared_memory_per_block_optin=232448)), \
             mock.patch.object(torch_backend, '_get_triton_assign_rows', return_value=object()), \
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
                     mock.patch.object(torch.version, 'hip', None), \
                     mock.patch.object(backend, '_require_stack'), \
                     mock.patch.object(torch.cuda, 'get_device_capability', return_value=capability), \
                     self.assertRaisesRegex(RuntimeError, 'exactly SM90'):
                    backend._check_device('cuda:17')
            backend._check_cuda_device.cache_clear()
            with mock.patch.object(torch.cuda, 'device', return_value=nullcontext()), \
                 mock.patch.object(torch.version, 'hip', None), \
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_non_grouped_cute_ignores_broken_cccl(self):
        from faiss.contrib.e_means import e_means_grouping as grouping
        from faiss.contrib.e_means.cute import _prepare_centroids, _assign_rows

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
        from faiss.contrib.e_means.cute import _prepare_centroids, _assign_rows

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
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0
                        )
                    model = known_torch_centroids(
                        c,
                        model_class=(
                            Emeans
                            if dtype == torch.bfloat16
                            else QuantizedEmeans
                        ),
                        assignment_backend='cute',
                        assignment_chunk_size=max(1, n - 1),
                    )
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
        model = known_torch_centroids(
            c,
            model_class=QuantizedEmeans,
            assignment_backend='cute',
            quantizer=CastQuantizer(17, torch.bfloat16),
        )
        distance, labels = model.assign(rows)
        self.assertEqual(labels[:2].tolist(), [5, 192])
        self.assertIn(labels[2].item(), (10, 11))
        self.assertEqual(labels[3].item(), 0)
        truth = (rows.bfloat16().double()[:, None]-c.bfloat16().double()[None]).square().sum(2)
        torch.testing.assert_close(distance.double(), truth.gather(1, labels[:, None]).flatten(), rtol=0, atol=1e-5)
        torch.testing.assert_close(rows, before_rows, rtol=0, atol=0)
        torch.testing.assert_close(c, before_centroids, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_bf16_norm_refresh_and_subnormal_square(self):
        from faiss.contrib.e_means.cute import _prepare_centroids

        rows = torch.tensor([[-.4678138, .03789843, -.6960376]], device='cuda')
        c = torch.tensor([[-.8843690, 1.0697185, -.7981653], [.4984003, .5734861, -.5365824]], device='cuda')
        truth = (rows.bfloat16().float()[:, None]-c.bfloat16().float()[None]).square().sum(2)
        wrong = c.square().sum(1)-2*(rows.bfloat16().float()[:, None]*c.bfloat16().float()[None]).sum(2)
        self.assertEqual((truth.argmin(1).item(), wrong.argmin(1).item()), (0, 1))
        for q in (None, CastQuantizer(3, torch.bfloat16)):
            model = known_torch_centroids(
                c,
                model_class=QuantizedEmeans,
                assignment_backend='cute',
                quantizer=q,
            )
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_bf16_high_ids_and_planted_large_batch(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                from faiss.contrib.e_means.cute import (
                    _prepare_centroids,
                    _assign_rows,
                )

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
                rows[3].fill_(0.25)
                distance, labels = known_torch_centroids(
                    c,
                    model_class=(
                        Emeans
                        if dtype == torch.bfloat16
                        else QuantizedEmeans
                    ),
                    assignment_backend='cute',
                    assignment_chunk_size=8192,
                ).assign(rows)
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_bf16_decoded_tail_statistics_closing_and_sync(self):
        x = torch.tensor([[0., 0.], [.25, .5], [2., 2.], [2.25, 2.5], [4., 4.], [4.25, 4.5], [4.5, 4.75]], device='cuda')
        initial = x[[0, 2, 4]].clone()
        for q in (None, CastQuantizer(2, torch.bfloat16)):
            rows = x if q else x.bfloat16()
            before = rows.clone()
            args = dict(nepochs=2, batch_size=3, alpha0=.5, seed=7,
                        assignment_backend='cute', assignment_chunk_size=1, capture_assignments=True)
            model_class = Emeans if q is None else QuantizedEmeans
            if q is not None:
                args['quantizer'] = q
            matrix, caller = model_class(2, 3, **args), model_class(2, 3, **args)
            matrix.train(rows, initial)
            caller.begin(len(rows), initial)
            for epoch in range(2):
                order = torch.randperm(
                    len(rows),
                    generator=e_means._torch_generator(
                        rows.device, e_means._epoch_seed(7, epoch)
                    ),
                    device=rows.device,
                )
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
                    if epoch == caller.nepochs - 1:
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
            del caller
            torch.testing.assert_close(output, saved, rtol=0, atol=0)
            warm = QuantizedEmeans(
                2,
                3,
                nepochs=1,
                batch_size=3,
                alpha0=0.5,
                quantizer=q,
                assignment_backend='cute',
                assignment_chunk_size=1,
            )
            warm.begin(3, initial)
            warm.update(warm.accumulate(rows[:3]))
            warm = QuantizedEmeans(
                2,
                3,
                nepochs=1,
                batch_size=3,
                alpha0=0.5,
                quantizer=q,
                assignment_backend='cute',
                assignment_chunk_size=1,
            )
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
        x[:, 0] = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            device="cuda",
        )
        c = torch.zeros(3, 17, device='cuda')
        c[:, 0] = torch.tensor([0.0, 10.0, 100.0], device="cuda")
        model = QuantizedEmeans(
            17,
            3,
            nepochs=2,
            batch_size=2,
            alpha0=0.5,
            quantizer=CastQuantizer(17, torch.bfloat16),
            assignment_backend='cute',
        )
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_bf16_ordered_refresh_and_noncurrent_device(self):
        for dtype, offset in ((torch.bfloat16, 0), (torch.float8_e4m3fn, 0),
                              (torch.int8, 0), (torch.uint8, 128)):
            for k in ((2, 128) if dtype in (torch.int8, torch.uint8) else (2,)):
                with self.subTest(dtype=dtype, k=k):
                    from faiss.contrib.e_means.cute import (
                        _prepare_centroids,
                        _assign_rows,
                    )

                    current = torch.cuda.current_device()
                    device = torch.device('cuda', (current+1) % torch.cuda.device_count())
                    c = torch.full((k, 17), 100.+offset, device=device)
                    c[0].fill_(offset)
                    c[1].fill_(1+offset)
                    rows = torch.full((257, 17), offset, dtype=dtype, device=device)
                    model_class = (
                        Emeans
                        if dtype == torch.bfloat16
                        else QuantizedEmeans
                    )
                    model = known_torch_centroids(
                        c, model_class=model_class, assignment_backend='cute'
                    )
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_fp8_recalibration_after_training_matches_fresh_assignment(self):
        x = torch.linspace(-1, 1, 32, device='cuda')[:, None].expand(32, 128).contiguous()
        initial = x[[4, 12, 20, 28]].clone()
        q = Fp8Quantizer(128)
        q.train(x)
        model = QuantizedEmeans(
            128, 4, nepochs=2, quantizer=q, assignment_backend='cute'
        )
        model.train(q.compute_codes(x), initial)
        retained = model.assign(q.compute_codes(x[:9]))
        saved = tuple(value.clone() for value in retained)
        for offset in (0.0625, 0.125, 0.25):
            q.train(x + offset)
            rows = q.compute_codes(x[:9])
            actual = model.assign(rows)
            fresh = known_torch_centroids(
                model.centroids.clone(),
                model_class=QuantizedEmeans,
                quantizer=q,
                assignment_backend='cute',
            )
            expected = fresh.assign(rows)
            for observed, reference in zip(actual, expected):
                torch.testing.assert_close(observed, reference, rtol=0, atol=0)
            for observed, reference in zip(retained, saved):
                torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_fp8_native_and_jit_share_decoded_state_and_output_units(self):
        x = torch.tensor([[0.], [2.], [10.], [12.]], device='cuda')
        q = Fp8Quantizer(1)
        q.train(x)
        models = [QuantizedEmeans(1, 2, nepochs=1, batch_size=4, alpha0=.5,
                         quantizer=q, assignment_backend='cute') for _ in range(2)]
        for model,rows in zip(models, (q.compute_codes(x),x)):
            model.train(rows, x[[0,2]])
            torch.testing.assert_close(model.centroids,x.new_tensor([[1.],[11.]]))
            self.assertEqual(model.obj.item(),8.)
            distance,labels = model.assign(rows)
            torch.testing.assert_close(distance,torch.ones(4,device=x.device))
            torch.testing.assert_close(labels,torch.tensor([0,0,1,1],device=x.device))
            original_distance, original_labels = model.assign(x)
            self.assertEqual(model.assignment_recipe['representation'], 'float8_e4m3fn')
            torch.testing.assert_close(original_distance, distance, rtol=0, atol=0)
            torch.testing.assert_close(original_labels, labels, rtol=0, atol=0)
        low, high = torch.full((5, 65), .1, device='cuda'), torch.full((5, 65), 2.3, device='cuda')
        x = torch.cat((low, high))
        q = Fp8Quantizer(65)
        q.train(x)
        self.assertTrue(q.centered)
        represented = q.decode(q.compute_codes(x))
        for rows in (q.compute_codes(x), x):
            model = QuantizedEmeans(
                65,
                2,
                nepochs=1,
                batch_size=10,
                alpha0=1.0,
                quantizer=q,
                assignment_backend='cute',
            )
            model.train(rows, torch.stack((low[0], high[0])))
            torch.testing.assert_close(model.centroids, torch.stack((represented[:5].mean(0), represented[5:].mean(0))))
            distance, labels = model.assign(rows)
            self.assertEqual((model.centroids.dtype, distance.dtype, labels.dtype), (torch.float32, torch.float32, torch.int64))
            self.assertEqual(labels.tolist(), [0]*5+[1]*5)
            torch.testing.assert_close(distance, torch.zeros_like(distance))
            query = low[:1].clone()
            query[0, 0] += .125
            distance, labels = model.assign(query)
            self.assertEqual(model.assignment_recipe['representation'], 'float8_e4m3fn')
            self.assertEqual(labels.item(), 0)
            torch.testing.assert_close(distance, torch.tensor([.015625], device='cuda'))

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_fp8_workspace_bytes_match_public_encoding_before_reuse(self):
        from faiss.contrib.e_means.cute import _prepare_centroids, _assign_rows

        midpoint = torch.tensor([2**-10, 3*2**-10, 15*2**-10, 17*2**-10,
                                 1.0625, 31., 62., 432.], device='cuda')
        positive = torch.cat((midpoint,
                              torch.nextafter(midpoint, torch.zeros_like(midpoint)),
                              torch.nextafter(midpoint, torch.full_like(midpoint, float('inf'))),
                              midpoint.new_tensor([0., 448.])))
        target = torch.cat((positive, -positive))[:, None].expand(-1, 65)
        plain = torch.tensor([-128., 128.], device='cuda')[:, None].expand(-1, 65).repeat(1024, 1)
        centered = torch.tensor([.1, 2.3], device='cuda')[:, None].expand(-1, 65)
        for sample, center in ((plain, False), (centered, True)):
            with self.subTest(centered=center):
                q = Fp8Quantizer(65)
                q.train(sample)
                self.assertEqual(q.centered, center)
                rows = target / q.scale
                if q.shift is not None:
                    rows = rows + q.shift
                rows = rows.contiguous()
                codes = q.compute_codes(rows)
                before = codes.view(torch.uint8).clone()
                prepared = _prepare_centroids(rows[[4, len(positive)+4]].clone(), transform=q, dtype=q.dtype)
                distance, labels = _assign_rows(rows, prepared)
                saved = distance.clone(), labels.clone()
                # Consume these actual codes BEFORE the next equal-size call
                # can overwrite the shared assignment workspace.
                encoded = prepared.bindings[len(rows)].workspace.view(torch.uint8).clone()
                torch.testing.assert_close(encoded, before, rtol=0, atol=0)
                matching = _assign_rows(codes, prepared)
                for actual, expected in zip(matching, saved):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(codes.view(torch.uint8), before, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
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
            model = known_torch_centroids(
                initial,
                model_class=QuantizedEmeans,
                quantizer=q,
                assignment_backend='cute',
                assignment_chunk_size=chunk,
            )
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_fp8_decoded_caller_tail_revival_and_closing(self):
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.25, 0.5],
                [2.0, 2.0],
                [2.25, 2.5],
                [4.0, 4.0],
                [4.25, 4.5],
                [4.5, 4.75],
            ],
            device="cuda",
        )
        initial = torch.tensor([[0., 0.], [2., 2.], [100., 100.]], device='cuda')
        q = Fp8Quantizer(2)
        q.train(
            torch.tensor(
                [[-128.0, -128.0], [128.0, 128.0]], device="cuda"
            ).repeat(1024, 1)
        )
        self.assertFalse(q.centered)
        codes = q.compute_codes(x)
        before = codes.view(torch.uint8).clone()
        self.assertTrue(bool((codes.float().abs().sum(0)*512 < 2**24).all()))
        for incoming in (codes, x):
            with self.subTest(native=incoming is codes):
                args = dict(nepochs=2, batch_size=3, alpha0=.5, seed=7, quantizer=q,
                            assignment_backend='cute', assignment_chunk_size=2, capture_assignments=True)
                matrix, caller = QuantizedEmeans(
                    2, 3, **args
                ), QuantizedEmeans(2, 3, **args)
                matrix.train(incoming, initial)
                caller.begin(len(x), initial)
                for epoch in range(2):
                    order = torch.randperm(
                        len(x),
                        generator=e_means._torch_generator(
                            x.device, e_means._epoch_seed(7, epoch)
                        ),
                        device=x.device,
                    )
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
                        if epoch == caller.nepochs - 1:
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
                counts, sums, state = (
                    caller._pass_counts,
                    caller._pass_sums,
                    caller._centroids,
                )
                closing = torch.where(
                    counts[:, None] > 0,
                    sums / counts.clamp_min(1).float()[:, None],
                    state,
                )
                torch.testing.assert_close(
                    caller.finalize(), closing, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    caller.centroids, matrix.centroids, rtol=1e-6, atol=1e-6
                )
                torch.testing.assert_close(
                    caller.obj, matrix.obj, rtol=1e-6, atol=3e-5
                )
                self.assertGreater(caller.n_splits, 0)
        torch.testing.assert_close(codes.view(torch.uint8), before, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_fp8_seed_decode_and_independent_closing_distortion(self):
        x = torch.tensor([[-1.625, -1.875], [-.375, .125], [.375, .625], [1.125, 1.375],
                          [1.625, 1.875], [.125, .375], [-.625, -.875], [.75, 1.25], [-1.25, -1.75]], device='cuda')
        for centered, all_rows in ((False, False), (True, False), (True, True)):
            rows = (x + 10 if centered else x)[:2 if all_rows else len(x)]
            q = Fp8Quantizer(2)
            q.train(
                rows.cpu()
                if centered
                else torch.tensor([[-3.0, -3.0], [3.0, 3.0]]).repeat(2048, 1)
            )
            q.to(rows.device)
            self.assertEqual(q.centered, centered)
            self.assertNotEqual(q.scale, 1.0)
            codes = q.compute_codes(rows)
            before = codes.view(torch.uint8).clone()
            indices = (
                torch.arange(len(rows), device=rows.device)
                if all_rows
                else torch.randperm(
                    len(rows),
                    generator=e_means._torch_generator(rows.device, 7),
                    device=rows.device,
                )[:2]
            )
            initial = q.decode(codes.float()[indices])
            args = dict(nepochs=2, batch_size=3, alpha0=.5, seed=7, quantizer=q,
                        assignment_backend='cute', assignment_chunk_size=2)
            native, jit, explicit = (
                QuantizedEmeans(2, 2, **args) for _ in range(3)
            )
            for model, incoming in ((native, codes), (jit, rows)):
                with mock.patch.object(model, 'begin', wraps=model.begin) as begin:
                    model.train(incoming)
                torch.testing.assert_close(begin.call_args.args[1], initial, rtol=0, atol=0)
            explicit.train(rows, initial)
            for model in (native, explicit):
                torch.testing.assert_close(model.centroids, jit.centroids, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(model.obj, jit.obj, rtol=1e-6, atol=3e-5)
            torch.testing.assert_close(
                codes.view(torch.uint8), before, rtol=0, atol=0
            )
        d, k = 1024, 4
        g = torch.Generator(device='cuda').manual_seed(1821)
        base = (torch.arange(d, device='cuda') % 31 + 128).float()
        initial = torch.stack([base + 32*i for i in range(k)]).to(torch.float8_e4m3fn).float()
        rows = (initial[torch.arange(128, device='cuda') % k] + torch.randint(-8, 9, (128, d), generator=g, device='cuda')).to(torch.float8_e4m3fn)
        scores = []
        for backend in ('triton', 'cute'):
            model = QuantizedEmeans(
                d, k, nepochs=2, seed=1234, assignment_backend=backend
            )
            model.train(rows, initial)
            self.assertTrue(bool(torch.isfinite(model.centroids).all()))
            # Returned closing centroids stay FP32: this fresh oracle does not
            # call assign or compare the moving-pass training objective.
            direct = (rows.double()[:, None] - model.centroids.double()[None]).square().sum(2)
            scores.append(direct.min(1).values)
        torch.testing.assert_close(scores[1], scores[0], rtol=1e-5, atol=1e-3)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_native_bytes_integer_oracles_destinations_and_refresh(self):
        from faiss.contrib.e_means.cute import _prepare_centroids, _assign_rows

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
                    model = known_torch_centroids(
                        c,
                        model_class=QuantizedEmeans,
                        assignment_backend='cute',
                    )
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
            model = known_torch_centroids(
                c + offset,
                model_class=QuantizedEmeans,
                assignment_backend='cute',
            )
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
            model = known_torch_centroids(
                c, model_class=QuantizedEmeans, assignment_backend='cute'
            )
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_native_bytes_stress_and_high_integer_ids(self):
        from faiss.contrib.e_means.cute import _prepare_centroids, _assign_rows

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
                model = known_torch_centroids(
                    c,
                    model_class=QuantizedEmeans,
                    assignment_backend='cute',
                    assignment_chunk_size=32768,
                )
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

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
    def test_native_bytes_revival_and_raw_code_workflow(self):
        for dtype, value in ((torch.int8, 0), (torch.uint8, 128)):
            rows = torch.full((10, 17), value, dtype=dtype, device='cuda')
            model = QuantizedEmeans(
                17,
                2,
                nepochs=5,
                batch_size=10,
                alpha0=0.5,
                assignment_chunk_size=4,
                assignment_backend='cute',
            )
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
        model = QuantizedEmeans(
            80, 4, nepochs=1, batch_size=8, alpha0=0.5, assignment_backend='cute'
        )
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
            results = [
                known_torch_centroids(
                    initial,
                    model_class=QuantizedEmeans,
                    quantizer=q,
                    assignment_backend=backend,
                ).assign(codes)
                for backend in ('triton', 'cute')
            ]
            for actual, wanted in zip(results[1], results[0]):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    @unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
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
                model = QuantizedEmeans(1,2,nepochs=1,batch_size=4,alpha0=.5,
                               quantizer=quantizer,assignment_backend='cute')
                model.train(rows,(x+offset)[[0,2]])
                torch.testing.assert_close(model.centroids,x.new_tensor([[1.],[11.]])+offset)
                self.assertEqual(model.obj.item(),8.)
                distance,labels = model.assign(rows)
                torch.testing.assert_close(distance,torch.ones(4,device=x.device))
                torch.testing.assert_close(labels,torch.tensor([0,0,1,1],device=x.device))
        with self.assertRaises((TypeError, ValueError)):
            QuantizedEmeans(
                1, 2, quantizer=q, assignment_backend='cute'
            ).train(x)


if __name__ == "__main__":
    unittest.main()
