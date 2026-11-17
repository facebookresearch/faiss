# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.
"""Small score-domain/raw-first interface laws; owner integration is separate."""
from types import SimpleNamespace
import unittest

import torch

from faiss.contrib.e_means.e_means_torch import Emeans
from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer
from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
from faiss.contrib.e_means.torch_backend import TorchBackend, TritonBackend, CuteBackend


class Observed(Emeans):
    def _assign_rows(self, *args, **kwargs):
        self.seen = (len(args), dict(kwargs))
        return super()._assign_rows(*args, **kwargs)


class FixedSix(Emeans):
    def _assign_rows(self, rows, centroids, norm=None, first=None, labels=None, second=None):
        return super()._assign_rows(rows, centroids, norm, first, labels, second)


def geometry(device='cpu', dtype=torch.float32):
    rows = torch.tensor([[8192., 0.], [0., 0.]], device=device, dtype=dtype)
    centers = torch.tensor([[0., 1.], [0., 0.]], device=device)
    return rows, centers


class TestOwnerAssignmentScoreInterface(unittest.TestCase):
    def test_native_keys_preserve_difference_and_spherical_units(self):
        rows, centers = geometry()
        backend = TorchBackend(1)  # The second spans two centroid chunks.
        norms = centers.square().sum(1)
        second = torch.empty(len(rows))
        keys, labels = backend.assign(rows, centers, norms, second_out=second, scores=True)
        torch.testing.assert_close(keys, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.ones(2), rtol=0, atol=0)
        self.assertEqual(labels.tolist(), [1, 1])
        ordinary_second = torch.empty_like(second)
        distance, old_labels = backend.assign(rows, centers, norms, second_out=ordinary_second)
        torch.testing.assert_close(old_labels, labels, rtol=0, atol=0)
        self.assertEqual((ordinary_second - distance)[0].item(), 0)
        self.assertEqual((second - keys)[0].item(), 1)
        unit = torch.eye(2)
        a, ai = backend.assign(unit, unit, torch.zeros(2), True, second_out=second)
        b, bi = backend.assign(unit, unit, torch.zeros(2), True, second_out=ordinary_second, scores=True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(ai, bi, rtol=0, atol=0)
        torch.testing.assert_close(second, ordinary_second, rtol=0, atol=0)

    def test_genuine_observer_and_fixed_six_domains(self):
        rows, centers = geometry()
        norms, second = centers.square().sum(1), torch.empty(2)
        model = Observed(2, 2)
        keys, labels = model._assign_rows(rows, centers, norms, None, None, second, scores=True)
        self.assertEqual(model.seen, (6, {'scores': True}))
        torch.testing.assert_close(keys, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.ones(2), rtol=0, atol=0)
        self.assertEqual(labels.tolist(), [1, 1])
        legacy = FixedSix(2, 2)
        old, _ = legacy._assign_rows(rows, centers, norms, None, None, second)
        self.assertEqual(old[0].item(), 2**26)
        with self.assertRaisesRegex(TypeError, 'scores'):
            legacy._assign_rows(rows, centers, norms, None, None, second, scores=True)

    def test_false_omits_keyword_at_legacy_backend(self):
        class OldBackend(TorchBackend):
            def assign(self, rows, centers, norm=None, spherical=False, first=None, labels=None, second=None):
                return super().assign(rows, centers, norm, spherical, first, labels, second)
        rows, centers = geometry()
        model = Emeans(2, 2)
        model._execution_backend = OldBackend(2)
        first, labels = model._assign_rows(rows, centers, centers.square().sum(1))
        self.assertEqual(first[0].item(), 2**26)
        self.assertEqual(labels.tolist(), [1, 1])
        with self.assertRaisesRegex(TypeError, 'scores'):
            model._assign_rows(rows, centers, centers.square().sum(1), scores=True)

    def test_encoded_true_fails_before_provider(self):
        for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn):
            rows = torch.zeros((2, 2), dtype=dtype)
            centers = rows.clone()
            for backend in (TorchBackend(2), TritonBackend(2), CuteBackend(2)):
                with self.subTest(dtype=dtype, backend=backend.name):
                    with self.assertRaisesRegex(ValueError, 'unencoded'):
                        backend.assign(rows, centers, scores=True)
            # Deferred FP8 can present FP32 rows but still has encoded operands.
            with self.assertRaisesRegex(ValueError, 'unencoded'):
                CuteBackend(2).assign(rows.float(), SimpleNamespace(centroids=centers), scores=True)
        for q in (Int8Quantizer(2), Fp8Quantizer(2)):
            model = QuantizedEmeans(2, 2, quantizer=q)
            with self.assertRaisesRegex(ValueError, 'unencoded'):
                model._assign_rows(torch.zeros((2, 2)), torch.zeros((2, 2)), scores=True)

    @unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip, 'CUDA required')
    def test_gpu_pre_norm_keys_and_local_k1(self):
        capability = torch.cuda.get_device_capability()
        arms = [('native', torch.float32)]
        if capability >= (8, 0):
            arms += [('triton', torch.float32), ('triton', torch.float16), ('triton', torch.bfloat16)]
            if 'out_dtype' in (torch.addmm.__doc__ or ''):
                arms += [('native', torch.float16), ('native', torch.bfloat16)]
        if capability == (9, 0):
            arms += [('cute', torch.bfloat16)]
        for backend, dtype in arms:
            with self.subTest(backend=backend, dtype=dtype):
                rows, centers = geometry('cuda', dtype)
                saved_rows, saved_centers = rows.clone(), centers.clone()
                model = QuantizedEmeans(2, 2, quantizer=None, assignment_backend=backend)
                rows = model._prepare_input(rows)
                prepared = model._prepare_torch_assignment(centers)
                second = torch.empty(2, device='cuda')
                keys, labels = model._assign_rows(rows, *prepared, None, None, second, scores=True)
                torch.testing.assert_close(keys, torch.zeros_like(keys), rtol=0, atol=0)
                torch.testing.assert_close(second, torch.ones_like(second), rtol=0, atol=0)
                self.assertEqual(labels.tolist(), [1, 1])
                old_second = torch.empty_like(second)
                old, old_labels = model._assign_rows(rows, *prepared, None, None, old_second)
                torch.testing.assert_close(labels, old_labels, rtol=0, atol=0)
                self.assertEqual((old_second - old)[0].item(), 0)
                one = model._prepare_torch_assignment(centers[:1])
                first, single = model._assign_rows(rows, *one, None, None, second, scores=True)
                torch.testing.assert_close(first, torch.ones_like(first), rtol=0, atol=0)
                self.assertTrue(bool(torch.isposinf(second).all()))
                self.assertEqual(single.tolist(), [0, 0])
                torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
                torch.testing.assert_close(centers, saved_centers, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip, 'CUDA required')
    def test_calibrated_integer_raw_first_only_and_pair(self):
        if torch.cuda.get_device_capability() not in ((8, 0), (9, 0), (10, 0)):
            self.skipTest('current integer assignment requires SM80/SM90/SM100')
        for backend in ('triton', 'cute'):
            if backend == 'cute' and torch.cuda.get_device_capability() != (9, 0):
                continue
            with self.subTest(backend=backend):
                centers = torch.tensor([[-1., 0.], [1., 0.]], device='cuda')
                q = Int8Quantizer(2)
                q.train(centers)
                rows = q.compute_codes(torch.tensor([[-1., 0.], [.5, 0.], [1., 0.]], device='cuda'))
                saved = rows.clone()
                model = QuantizedEmeans(2, 2, quantizer=q, assignment_backend=backend)
                rows = model._prepare_input(rows)
                prepared = model._prepare_torch_assignment(centers)
                ordinary, labels = model._assign_rows(rows, *prepared)
                ordinary = ordinary.clone()
                first = torch.empty(3, dtype=torch.int64, device='cuda')
                formatted, ids = model._assign_rows(rows, *prepared, first, None, None)
                codes = q.compute_codes(centers).to(torch.int64)
                truth = (rows.to(torch.int64)[:, None] - codes[None]).square().sum(2)
                order = truth.argsort(dim=1, stable=True)
                torch.testing.assert_close(first, truth.gather(1, order[:, :1]).flatten(), rtol=0, atol=0)
                self.assertEqual(formatted.dtype, torch.float32)
                torch.testing.assert_close(formatted, ordinary, rtol=0, atol=0)
                torch.testing.assert_close(ids, labels, rtol=0, atol=0)
                second = torch.empty_like(first)
                paired, pair_ids = model._assign_rows(rows, *prepared, first, None, second)
                torch.testing.assert_close(paired, ordinary, rtol=0, atol=0)
                torch.testing.assert_close(pair_ids, labels, rtol=0, atol=0)
                torch.testing.assert_close(second, truth.gather(1, order[:, 1:2]).flatten(), rtol=0, atol=0)
                # The first-only request must also work at local K1.
                one = model._prepare_torch_assignment(centers[:1])
                single, single_ids = model._assign_rows(rows, *one, first, None, None)
                torch.testing.assert_close(first, truth[:, 0], rtol=0, atol=0)
                torch.testing.assert_close(single, torch.ldexp(first.float(), q._distance_exponent), rtol=0, atol=0)
                self.assertEqual(single_ids.tolist(), [0, 0, 0])
                torch.testing.assert_close(rows, saved, rtol=0, atol=0)
