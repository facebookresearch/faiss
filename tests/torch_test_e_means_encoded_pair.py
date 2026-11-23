# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Representative integer ordering and FP8 selected-distance contracts."""
import importlib.util
import unittest
from unittest import mock

import torch

CUDA = torch.cuda.is_available() and not torch.version.hip
CAP = torch.cuda.get_device_capability() if CUDA else None
TRITON = CUDA and importlib.util.find_spec("triton") is not None
INT8 = TRITON and CAP in ((8, 0), (9, 0), (10, 0))
FP8 = TRITON and CAP in ((9, 0), (10, 0), (10, 3))
CUTE = INT8 and CAP == (9, 0)


def pair_truth(rows, centers):
    distances = (rows.long()[:, None] - centers.long()[None]).square().sum(2)
    ids = distances.argsort(dim=1, stable=True)
    return distances.gather(1, ids[:, :2]), ids[:, :2]


def assert_pair(case, first, second, ids, other, rows, centers):
    distances, expected = pair_truth(rows.cpu(), centers.cpu())
    for actual, wanted in ((first, distances[:, 0]), (second, distances[:, 1])):
        torch.testing.assert_close(actual.cpu().double(), wanted.double(), rtol=0, atol=0)
    torch.testing.assert_close(ids.cpu(), expected[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(other.cpu(), expected[:, 1], rtol=0, atol=0)
    case.assertTrue(bool((ids != other).all()))


class TestEncodedPair(unittest.TestCase):
    @unittest.skipUnless(INT8, "qualified INT8 Triton CUDA required")
    def test_integer_pair_tails_ties_and_gap_above_fp32(self):
        from faiss.contrib.e_means.triton_assignment import assign_int8_rows, prepare_int8_centroids

        # Non-packed D and an exact tie in the last centroid tile.
        signed = torch.stack([torch.full((273,), x, dtype=torch.int8) for x in (-128, 0, 127)])
        centers = torch.full((259, 273), 50.)
        centers[128] = signed[0].float()
        centers[258] = centers[128]
        # Packed UINT8 distances differ by one above2**24: FP32 cannot rank them.
        unsigned = torch.full((1, 260), 255, dtype=torch.uint8)
        unsigned[0, 0] = 0
        near = torch.zeros((2, 260))
        near[1, 0] = 1
        for x, c in ((signed, centers), (unsigned, near)):
            with self.subTest(dtype=x.dtype):
                representation = "uint8" if x.dtype == torch.uint8 else "int8"
                x, c = x.cuda(), c.cuda()
                saved = x.clone()
                prepared, norms = prepare_int8_centroids(c, representation)
                first = torch.empty(len(x), dtype=torch.int64, device="cuda")
                second, other = torch.empty_like(first), torch.empty_like(first)
                returned, ids = assign_int8_rows(x, prepared, norms, representation,
                                                first, second_out=second, second_labels_out=other)
                self.assertIs(returned, first)
                assert_pair(self, first, second, ids, other, x, c)
                top1, top1_ids = assign_int8_rows(x, prepared, norms, representation)
                torch.testing.assert_close(top1, first.float(), rtol=0, atol=0)
                torch.testing.assert_close(top1_ids, ids, rtol=0, atol=0)
                torch.testing.assert_close(x, saved, rtol=0, atol=0)
                if x.dtype == torch.uint8:
                    self.assertGreater(first.item(), 2**24)
                    self.assertEqual((second - first).item(), 1)
                    self.assertEqual((second.float() - first.float()).item(), 0)

    @unittest.skipUnless(INT8, "qualified INT8 Triton CUDA required")
    def test_integer_second_validation(self):
        from faiss.contrib.e_means.triton_assignment import assign_int8_rows, prepare_int8_centroids

        rows = torch.zeros((3, 1), dtype=torch.int8, device="cuda")
        first = torch.full((3,), -123, dtype=torch.int64, device="cuda")
        prepared, norms = prepare_int8_centroids(torch.zeros((1, 1), device="cuda"), "int8")
        with self.assertRaisesRegex(ValueError, "at least two"):
            assign_int8_rows(rows, prepared, norms, "int8", first, second_out=first.clone())
        prepared, norms = prepare_int8_centroids(torch.zeros((2, 1), device="cuda"), "int8")
        with self.assertRaisesRegex(ValueError, "match the first"):
            assign_int8_rows(rows, prepared, norms, "int8", first,
                             second_out=torch.empty(3, device="cuda"))
        with self.assertRaisesRegex(ValueError, "second labels require"):
            assign_int8_rows(rows, prepared, norms, "int8", second_labels_out=first)
        torch.testing.assert_close(first, torch.full_like(first, -123), rtol=0, atol=0)

    @unittest.skipUnless(CUTE, "CuTe SM90 required")
    def test_cute_row_owned_and_tiled_pair_reuse(self):
        from faiss.contrib.e_means import cute

        # Signed row-owned and unsigned tiled, sharing the tie fixture.
        for dtype, k in ((torch.int8, 128), (torch.uint8, 129)):
            with self.subTest(dtype=dtype, k=k):
                x = (torch.arange(5 * 16).reshape(5, 16) % 5).cuda().to(dtype)
                c = torch.full((k, 16), 4., device="cuda")
                c[0] = c[-1] = x[0].float()
                saved = x.clone()
                prepared = cute._prepare_centroids(c, dtype=dtype)
                first = torch.empty(5, dtype=torch.int64, device="cuda")
                second = torch.empty_like(first)
                for _ in range(2):  # Both output planes must reset on reuse.
                    returned, ids = cute._assign_rows(x, prepared, distance_out=first,
                                                     second_out=second)
                    binding = prepared.bindings[cute._binding_key(5, torch.int64, True)]
                    self.assertIs(returned, first)
                    self.assertEqual(binding.launcher.__name__,
                                     "launch_row_owned" if k == 128 else "launch_tiled_gemm")
                    assert_pair(self, first, second, ids, binding.second_labels, x, c)
                top1, top1_ids = cute._assign_rows(x, prepared)
                torch.testing.assert_close(top1, first.float(), rtol=0, atol=0)
                torch.testing.assert_close(top1_ids, ids, rtol=0, atol=0)
                torch.testing.assert_close(x, saved, rtol=0, atol=0)

    @unittest.skipUnless(CUTE, "CuTe SM90 required")
    def test_private_integer_second_rejects_without_mutation(self):
        from faiss.contrib.e_means import cute_kernel
        from faiss.contrib.e_means.cute import _aligned_operand, _prepare_centroids
        from faiss.contrib.e_means.cute_argmin import stage_int8_rows

        rows = torch.zeros((3, 1), dtype=torch.int8, device="cuda")
        prepared = _prepare_centroids(torch.zeros((2, 1), device="cuda"), dtype=rows.dtype)
        workspace = _aligned_operand(3, 1, rows.device, torch.int8)
        norms = torch.empty(3, dtype=torch.int32, device="cuda")
        stage_int8_rows(rows, workspace, norms, "int8")
        outputs = [torch.full((3,), -123, dtype=torch.int64, device="cuda") for _ in range(3)]
        first, second, ids = outputs
        with mock.patch.object(cute_kernel, "_compile_ordered") as compile_kernel:
            with self.assertRaisesRegex(ValueError, "integer tiled SECOND requires second labels"):
                cute_kernel.prepare_tiled_gemm(workspace, prepared, ids, first, norms, second)
        compile_kernel.assert_not_called()
        for output in outputs:
            torch.testing.assert_close(output, torch.full_like(output, -123), rtol=0, atol=0)

    @unittest.skipUnless(FP8, "qualified FP8 Triton CUDA required")
    def test_fp8_second_completes_selected_ids_without_reranking(self):
        from faiss.contrib.e_means import triton_assignment as ta

        # FP32 ranking cancels: existing Top1 is0 although ID1 is closer.
        rows = torch.full((1, 128), 448., device="cuda")
        rows[:, -1] = 0
        centers = torch.full((2, 128), 448., device="cuda")
        centers[0, -1], centers[1, -1] = .25, .125
        x, c = rows.to(torch.float8_e4m3fn), centers.to(torch.float8_e4m3fn)
        norms = c.float().square().sum(1)
        second = torch.empty(1, device="cuda")
        other = torch.empty(1, dtype=torch.int64, device="cuda")
        first, ids = ta.assign_rows(x, c, norms, second_out=second, second_labels_out=other)
        old_first, old_ids = ta.assign_rows(x, c, norms)
        torch.testing.assert_close(first, old_first, rtol=0, atol=0)
        torch.testing.assert_close(ids, old_ids, rtol=0, atol=0)
        self.assertEqual((ids.item(), other.item()), (0, 1))
        self.assertEqual((first.item(), second.item()), (.0625, .015625))
        self.assertEqual((second - first).item(), -.046875)
        if CAP == (9, 0):
            from faiss.contrib.e_means import cute
            for k in (128, 129):  # Both independently completed FP8 epilogues.
                c = torch.full((k, 128), 448., device="cuda")
                c[:, -1] = 1
                c[0, -1], c[1, -1] = .25, .125
                prepared = cute._prepare_centroids(c, dtype=x.dtype)
                first, ids = cute._assign_rows(x, prepared, second_out=second)
                other = prepared.bindings[cute._binding_key(1, torch.float32, True)].second_labels
                self.assertEqual((ids.item(), other.item()), (0, 1))
                self.assertEqual((second - first).item(), -.046875)

    @unittest.skipUnless(INT8, "qualified INT8 CUDA required")
    def test_calibrated_raw_pair_and_public_distance_units(self):
        from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
        from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer

        values = torch.tensor([[10., 2.], [10.25, 2.5], [11., 3.]], device="cuda")
        q = Int8Quantizer(2)
        q.train(values)
        rows, initial = q.compute_codes(values), values[[0, 2]]
        for backend in ["triton"] + (["cute"] if CUTE else []):
            model = QuantizedEmeans(2, 2, quantizer=q, unused_threshold=.1,
                                    assignment_backend=backend)
            model.begin(3, initial)
            x = model._prepare_rows(model._prepare_input(rows))
            c, norms = model._prepare_torch_assignment(model._centroids)
            first = torch.empty(3, dtype=torch.int64, device="cuda")
            second = torch.empty_like(first)
            distance, ids = model._assign_rows(x, c, norms, first, None, second)
            truth, expected = pair_truth(rows, q.compute_codes(initial))
            torch.testing.assert_close(first, truth[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(second, truth[:, 1], rtol=0, atol=0)
            torch.testing.assert_close(ids, expected[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(distance, torch.ldexp(first.float(), q._distance_exponent),
                                       rtol=0, atol=0)
            ordinary, old_ids = model._assign_rows(x, c, norms)
            torch.testing.assert_close(distance, ordinary, rtol=0, atol=0)
            torch.testing.assert_close(ids, old_ids, rtol=0, atol=0)
