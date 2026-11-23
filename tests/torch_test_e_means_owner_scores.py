# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Small score-domain/raw-first interface laws; owner integration is separate."""
from types import SimpleNamespace
import unittest

import torch

from faiss.contrib.e_means.e_means_torch import Emeans
from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer
from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
from faiss.contrib.e_means.torch_backend import (
    TorchBackend,
    TritonBackend,
    CuteBackend,
)


class Observed(Emeans):
    def _assign_rows(self, *args, **kwargs):
        self.seen = (len(args), dict(kwargs))
        return super()._assign_rows(*args, **kwargs)


class FixedSix(Emeans):
    def _assign_rows(
        self, rows, centroids, norm=None, first=None, labels=None, second=None
    ):
        return super()._assign_rows(
            rows, centroids, norm, first, labels, second
        )


def geometry(device="cpu", dtype=torch.float32):
    rows = torch.tensor([[8192.0, 0.0], [0.0, 0.0]], device=device, dtype=dtype)
    centers = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device)
    return rows, centers


class TestOwnerAssignmentScoreInterface(unittest.TestCase):
    def test_native_keys_preserve_difference_and_spherical_units(self):
        rows, centers = geometry()
        backend = TorchBackend(1)  # The second spans two centroid chunks.
        norms = centers.square().sum(1)
        second = torch.empty(len(rows))
        keys, labels = backend.assign(
            rows, centers, norms, second_out=second, scores=True
        )
        torch.testing.assert_close(keys, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.ones(2), rtol=0, atol=0)
        self.assertEqual(labels.tolist(), [1, 1])
        ordinary_second = torch.empty_like(second)
        distance, old_labels = backend.assign(
            rows, centers, norms, second_out=ordinary_second
        )
        torch.testing.assert_close(old_labels, labels, rtol=0, atol=0)
        self.assertEqual((ordinary_second - distance)[0].item(), 0)
        self.assertEqual((second - keys)[0].item(), 1)
        unit = torch.eye(2)
        a, ai = backend.assign(
            unit, unit, torch.zeros(2), True, second_out=second
        )
        b, bi = backend.assign(
            unit,
            unit,
            torch.zeros(2),
            True,
            second_out=ordinary_second,
            scores=True,
        )
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(ai, bi, rtol=0, atol=0)
        torch.testing.assert_close(second, ordinary_second, rtol=0, atol=0)

        if torch.cuda.is_available() and not torch.version.hip:
            cap = torch.cuda.get_device_capability()
            arms = [(TorchBackend(1), torch.float32)]
            if cap >= (8, 0):
                arms.append((TritonBackend(2), torch.bfloat16))
            if cap == (9, 0):
                arms.append((CuteBackend(2), torch.bfloat16))
            for backend, dtype in arms:
                with self.subTest(backend=backend.name):
                    x, c = geometry("cuda", dtype)
                    prepared, norms = backend.prepare_centroids(c, dtype)
                    second = torch.empty(2, device="cuda")
                    keys, ids = backend.assign(x, prepared, norms, second_out=second, scores=True)
                    torch.testing.assert_close(keys, torch.zeros_like(keys), rtol=0, atol=0)
                    torch.testing.assert_close(second, torch.ones_like(second), rtol=0, atol=0)
                    self.assertEqual(ids.tolist(), [1, 1])
                    prepared, norms = backend.prepare_centroids(c[1:], dtype)
                    backend.assign(x, prepared, norms, second_out=second, scores=True)
                    self.assertTrue(bool(torch.isposinf(second).all()))

    def test_genuine_observer_and_fixed_six_domains(self):
        rows, centers = geometry()
        norms, second = centers.square().sum(1), torch.empty(2)
        model = Observed(2, 2)
        keys, labels = model._assign_rows(
            rows, centers, norms, None, None, second, scores=True
        )
        self.assertEqual(model.seen, (6, {"scores": True}))
        torch.testing.assert_close(keys, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.ones(2), rtol=0, atol=0)
        self.assertEqual(labels.tolist(), [1, 1])
        legacy = FixedSix(2, 2)
        old, _ = legacy._assign_rows(rows, centers, norms, None, None, second)
        self.assertEqual(old[0].item(), 2**26)
        with self.assertRaisesRegex(TypeError, "scores"):
            legacy._assign_rows(
                rows, centers, norms, None, None, second, scores=True
            )

    def test_encoded_true_fails_before_provider(self):
        for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn):
            rows = torch.zeros((2, 2), dtype=dtype)
            centers = rows.clone()
            for backend in (TorchBackend(2), TritonBackend(2), CuteBackend(2)):
                with self.subTest(dtype=dtype, backend=backend.name):
                    with self.assertRaisesRegex(ValueError, "unencoded"):
                        backend.assign(rows, centers, scores=True)
            # Deferred FP8 can present FP32 rows but still has encoded operands.
            with self.assertRaisesRegex(ValueError, "unencoded"):
                CuteBackend(2).assign(
                    rows.float(),
                    SimpleNamespace(centroids=centers),
                    scores=True,
                )
        for q in (Int8Quantizer(2), Fp8Quantizer(2)):
            model = QuantizedEmeans(2, 2, quantizer=q)
            with self.assertRaisesRegex(ValueError, "unencoded"):
                model._assign_rows(
                    torch.zeros((2, 2)), torch.zeros((2, 2)), scores=True
                )
