# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton integer pair foundation: independent exact represented-distance laws."""
import importlib.util
import unittest

import torch

QUALIFIED = (
    torch.cuda.is_available()
    and not torch.version.hip
    and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0))
    and importlib.util.find_spec("triton") is not None
)


@unittest.skipUnless(QUALIFIED, "qualified INT8 Triton CUDA required")
class TestTritonEncodedIntegerPair(unittest.TestCase):
    def check_pair(self, rows_cpu, centers_cpu):
        from faiss.contrib.e_means.triton_assignment import (
            assign_int8_rows,
            prepare_int8_centroids,
        )

        unsigned = rows_cpu.dtype == torch.uint8
        representation = "uint8" if unsigned else "int8"
        offset = 128 if unsigned else 0
        # Anchor the independent exact geometry before any preparation/assignment call.
        truth = (
            (
                (rows_cpu.long()[:, None] - offset)
                - (centers_cpu.long()[None] - offset)
            )
            .square()
            .sum(2)
        )
        order = truth.argsort(dim=1, stable=True)
        expected_first = truth.gather(1, order[:, :1]).flatten()
        expected_second = truth.gather(1, order[:, 1:2]).flatten()
        rows, centers = rows_cpu.cuda(), centers_cpu.cuda()
        saved_rows, saved_centers = rows.clone(), centers.clone()
        prepared, norms = prepare_int8_centroids(centers, representation)
        n = len(rows)
        first = torch.empty(n, dtype=torch.int64, device="cuda")
        second = torch.empty_like(first)
        alternative = torch.empty_like(first)
        returned, labels = assign_int8_rows(
            rows,
            prepared,
            norms,
            representation,
            first,
            second_out=second,
            second_labels_out=alternative,
        )
        self.assertIs(returned, first)
        torch.testing.assert_close(first.cpu(), expected_first, rtol=0, atol=0)
        torch.testing.assert_close(
            second.cpu(), expected_second, rtol=0, atol=0
        )
        torch.testing.assert_close(labels.cpu(), order[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(
            alternative.cpu(), order[:, 1], rtol=0, atol=0
        )
        self.assertTrue(
            bool(
                (
                    (alternative >= 0)
                    & (alternative < len(centers))
                    & (alternative != labels)
                ).all()
            )
        )
        self.assertTrue(bool((second < torch.iinfo(torch.int64).max).all()))
        top1, top1_labels = assign_int8_rows(
            rows, prepared, norms, representation
        )
        torch.testing.assert_close(top1, first.float(), rtol=0, atol=0)
        torch.testing.assert_close(top1_labels, labels, rtol=0, atol=0)
        formatted_second = torch.empty(n, dtype=torch.float32, device="cuda")
        paired_first, paired_labels = assign_int8_rows(
            rows, prepared, norms, representation, second_out=formatted_second
        )
        torch.testing.assert_close(paired_first, top1, rtol=0, atol=0)
        torch.testing.assert_close(paired_labels, labels, rtol=0, atol=0)
        torch.testing.assert_close(
            formatted_second, second.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
        torch.testing.assert_close(centers, saved_centers, rtol=0, atol=0)
        return first, second

    def test_exact_pair_ties_tiles_and_row_tails(self):
        signed = (torch.arange(129 * 3).reshape(129, 3) % 15 - 7).to(torch.int8)
        centers = torch.tensor(
            [
                [-7.0, -6.0, -5.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [5.0, 6.0, 7.0],
                [2.0, -2.0, 1.0],
            ]
        )
        self.check_pair(signed, centers)
        unsigned = (torch.arange(5 * 17).reshape(5, 17) % 256).to(torch.uint8)
        centers = (torch.arange(129 * 17).reshape(129, 17) % 256).float()
        centers[0] = unsigned[0].float()
        centers[128] = centers[
            0
        ]  # True second crosses a centroid tile boundary.
        self.check_pair(unsigned, centers)
        signed = torch.tensor([[-127], [0], [127]], dtype=torch.int8)
        centers = (torch.arange(257) % 255 - 127).float()[:, None]
        self.check_pair(signed, centers)

    def test_integer_gap_one_survives_above_float32_exactness(self):
        # Two exact integer distances differ by1, but both round to the same FP32.
        rows = torch.full((1, 260), 255, dtype=torch.uint8)
        rows[0, 0] = 0
        centers = torch.zeros((2, 260))
        centers[1, 0] = 1
        first, second = self.check_pair(rows, centers)
        self.assertGreater(first.item(), 2**24)
        self.assertEqual((second - first).item(), 1)
        self.assertEqual((second.float() - first.float()).item(), 0)

    def test_nonpacked_multiple_tiles_with_exact_tie(self):
        rows = torch.stack(
            [
                torch.full((273,), value, dtype=torch.int8)
                for value in (-128, 0, 127)
            ]
        )
        centers = torch.full((259, 273), 50.0)
        centers[128] = rows[0].float()
        centers[258] = centers[128]
        # Physical d288 selects non-packed scoring; the tied second is in another K tile.
        self.check_pair(rows, centers)

    def test_uint8_unstaged_descriptor_layout(self):
        rows = (torch.arange(5 * 272).reshape(5, 272) % 256).to(torch.uint8)
        centers = torch.full((257, 272), 128.0)
        centers[0] = rows[0].float()
        centers[256] = centers[0]
        # d272 is already16-aligned: on SM90 x_desc uses the in-loop UINT8 -128.
        self.check_pair(rows, centers)

    def test_pair_validation_and_existing_bf16_k1(self):
        from faiss.contrib.e_means.triton_assignment import (
            assign_int8_rows,
            assign_rows,
            prepare_int8_centroids,
        )

        rows = torch.zeros((3, 1), dtype=torch.int8, device="cuda")
        prepared, norms = prepare_int8_centroids(
            torch.zeros((1, 1), device="cuda"), "int8"
        )
        first = torch.empty(3, dtype=torch.int64, device="cuda")
        second = torch.empty_like(first)
        distance, ids = assign_int8_rows(rows, prepared, norms, "int8")
        torch.testing.assert_close(
            distance, torch.zeros_like(distance), rtol=0, atol=0
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            assign_int8_rows(
                rows, prepared, norms, "int8", first, second_out=second
            )
        prepared, norms = prepare_int8_centroids(
            torch.zeros((2, 1), device="cuda"), "int8"
        )
        with self.assertRaisesRegex(ValueError, "match the first"):
            assign_int8_rows(
                rows,
                prepared,
                norms,
                "int8",
                first,
                second_out=torch.empty(3, device="cuda"),
            )
        with self.assertRaisesRegex(ValueError, "second labels require"):
            assign_int8_rows(
                rows, prepared, norms, "int8", second_labels_out=second
            )
        floating = torch.zeros((3, 1), dtype=torch.bfloat16, device="cuda")
        other = torch.empty(3, dtype=torch.float32, device="cuda")
        assign_rows(
            floating,
            floating[:1],
            torch.zeros(1, device="cuda"),
            second_out=other,
        )
        self.assertTrue(bool(torch.isposinf(other).all()))

    def test_legacy_five_argument_top1_helper(self):
        from unittest import mock
        from faiss.contrib.e_means import e_means_quantized as qmodule

        assign, prepare = qmodule._get_triton_int8_functions()
        calls = []

        def legacy(rows, centers, norms, representation, distance_out=None):
            calls.append(representation)
            return assign(rows, centers, norms, representation, distance_out)

        rows = torch.tensor([[0, 1], [2, 1]], dtype=torch.int8, device="cuda")
        initial = torch.tensor([[0.0, 0.0], [2.0, 2.0]], device="cuda")
        saved_rows, saved_initial = rows.clone(), initial.clone()
        model = qmodule.QuantizedEmeans(
            2,
            2,
            quantizer=None,
            unused_threshold=0.1,
            assignment_backend="triton",
        )
        model.begin(2, initial)
        with mock.patch.object(
            qmodule,
            "_get_triton_int8_functions",
            return_value=(legacy, prepare),
        ):
            actual = model._prepare_rows(model._prepare_input(rows))
            centers, norms = model._prepare_torch_assignment(model._centroids)
            distance, labels = model._assign_rows(actual, centers, norms)
        truth = (
            (saved_rows.long()[:, None] - saved_initial.long()[None])
            .square()
            .sum(2)
        )
        expected, ids = truth.min(1)
        self.assertEqual(calls, ["int8"])
        torch.testing.assert_close(distance, expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(labels, ids, rtol=0, atol=0)
        torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
        torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)


@unittest.skipUnless(
    QUALIFIED and torch.cuda.get_device_capability() == (9, 0),
    "CuTe SM90 required",
)
class TestCuTeTiledEncodedIntegerPair(unittest.TestCase):
    def check_pair(self, rows_cpu, centers_cpu):
        from faiss.contrib.e_means.cute import (
            _aligned_operand,
            _prepare_centroids,
        )
        from faiss.contrib.e_means.cute_argmin import stage_int8_rows
        from faiss.contrib.e_means.cute_kernel import (
            prepare_tiled_gemm,
            launch_tiled_gemm,
        )

        offset = 128 if rows_cpu.dtype == torch.uint8 else 0
        truth = (
            (
                (rows_cpu.long()[:, None] - offset)
                - (centers_cpu.long()[None] - offset)
            )
            .square()
            .sum(2)
        )
        order = truth.argsort(dim=1, stable=True)
        rows, centers = rows_cpu.cuda(), centers_cpu.cuda()
        saved_rows, saved_centers = rows.clone(), centers.clone()
        prepared = _prepare_centroids(centers, dtype=rows.dtype)
        workspace = _aligned_operand(
            len(rows), rows.shape[1], rows.device, torch.int8
        )
        row_norms = torch.empty(len(rows), dtype=torch.int32, device="cuda")
        stage_int8_rows(rows, workspace, row_norms, prepared.representation)
        first, second = [
            torch.empty(len(rows), dtype=torch.int64, device="cuda")
            for _ in range(2)
        ]
        ids, alternative = torch.empty_like(first), torch.empty_like(first)
        pair = prepare_tiled_gemm(
            workspace, prepared, ids, first, row_norms, second, alternative
        )
        launch_tiled_gemm(pair)
        torch.testing.assert_close(
            first.cpu(), truth.gather(1, order[:, :1]).flatten(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            second.cpu(),
            truth.gather(1, order[:, 1:2]).flatten(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(ids.cpu(), order[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(
            alternative.cpu(), order[:, 1], rtol=0, atol=0
        )
        top1 = torch.empty(len(rows), dtype=torch.float32, device="cuda")
        top1_ids = torch.empty_like(ids)
        single = prepare_tiled_gemm(
            workspace, prepared, top1_ids, top1, row_norms
        )
        launch_tiled_gemm(single)
        torch.testing.assert_close(top1, first.float(), rtol=0, atol=0)
        torch.testing.assert_close(top1_ids, ids, rtol=0, atol=0)
        # Decode/reset and reuse both planes, not just the first.
        launch_tiled_gemm(pair)
        torch.testing.assert_close(
            alternative.cpu(), order[:, 1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            second.cpu(),
            truth.gather(1, order[:, 1:2]).flatten(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
        torch.testing.assert_close(centers, saved_centers, rtol=0, atol=0)

    def test_tiled_signed_unsigned_pair_identity(self):
        rows = torch.tensor(
            [[-7.0, 0.0, 3.0], [0.0, 0.0, 0.0], [7.0, 2.0, -3.0]]
        ).to(torch.int8)
        centers = torch.tensor(
            [
                [-7.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [7.0, 2.0, -3.0],
            ]
        )
        self.check_pair(rows, centers)
        rows = (torch.arange(5 * 17).reshape(5, 17) % 256).to(torch.uint8)
        centers = torch.full((197, 17), 128.0)
        centers[0], centers[196] = rows[0].float(), rows[0].float()
        self.check_pair(rows, centers)

    def test_tiled_encoded_k1_rejects_before_pair(self):
        from faiss.contrib.e_means.cute import (
            _aligned_operand,
            _prepare_centroids,
        )
        from faiss.contrib.e_means.cute_argmin import stage_int8_rows
        from faiss.contrib.e_means.cute_kernel import (
            prepare_tiled_gemm,
            launch_tiled_gemm,
        )

        rows = torch.zeros((3, 1), dtype=torch.int8, device="cuda")
        prepared = _prepare_centroids(
            torch.zeros((1, 1), device="cuda"), dtype=rows.dtype
        )
        workspace = _aligned_operand(3, 1, rows.device, torch.int8)
        norms = torch.empty(3, dtype=torch.int32, device="cuda")
        stage_int8_rows(rows, workspace, norms, "int8")
        first = torch.empty(3, dtype=torch.int64, device="cuda")
        ids, second = torch.empty_like(first), torch.empty_like(first)
        with self.assertRaisesRegex(ValueError, "at least two"):
            prepare_tiled_gemm(workspace, prepared, ids, first, norms, second)
        launch_tiled_gemm(
            prepare_tiled_gemm(workspace, prepared, ids, first, norms)
        )
        torch.testing.assert_close(
            first, torch.zeros_like(first), rtol=0, atol=0
        )
        floating = _aligned_operand(3, 1, rows.device, torch.bfloat16)
        floating.zero_()
        prepared = _prepare_centroids(
            torch.zeros((1, 1), device="cuda"), dtype=torch.bfloat16
        )
        distance = torch.empty(3, device="cuda")
        other = torch.empty_like(distance)
        launch_tiled_gemm(
            prepare_tiled_gemm(
                floating, prepared, ids, distance, second_scores=other
            )
        )
        self.assertTrue(bool(torch.isposinf(other).all()))

    def test_tiled_multimultistage_integer_ties(self):
        # Two M tiles, six N tiles, two128-byte K stages; ties span complete tiles.
        for dtype, offset in ((torch.int8, -126), (torch.uint8, 0)):
            with self.subTest(dtype=dtype):
                modulus = 253 if dtype == torch.int8 else 256
                rows = (
                    torch.arange(129 * 129).reshape(129, 129) % modulus + offset
                ).to(dtype)
                centers = torch.full(
                    (1001, 129), 0.0 if dtype == torch.int8 else 128.0
                )
                centers[:192] = rows[0].float()
                centers[768:960] = rows[
                    0
                ].float()  # Within/across lanes and N tiles.
                centers[1000] = rows[
                    1
                ].float()  # Best in the partial final N tile.
                self.check_pair(rows, centers)

    def test_tiled_bf16_int32_primary_second_compatibility(self):
        from faiss.contrib.e_means.cute import (
            _aligned_operand,
            _prepare_centroids,
        )
        from faiss.contrib.e_means.cute_kernel import (
            prepare_tiled_gemm,
            launch_tiled_gemm,
        )

        rows = _aligned_operand(3, 1, torch.device("cuda"), torch.bfloat16)
        rows.copy_(torch.tensor([[0.0], [1.0], [2.0]], device="cuda"))
        prepared = _prepare_centroids(
            torch.tensor([[0.0], [4.0]], device="cuda"), dtype=torch.bfloat16
        )
        primary = torch.empty(
            3, dtype=torch.int32, device="cuda"
        )  # Existing helper domain.
        first, second = torch.empty(3, device="cuda"), torch.empty(
            3, device="cuda"
        )
        pair = prepare_tiled_gemm(
            rows, prepared, primary, first, second_scores=second
        )
        self.assertEqual(pair.second_decode.codes.dtype, torch.int64)
        launch_tiled_gemm(pair)
        torch.testing.assert_close(
            primary, torch.zeros_like(primary), rtol=0, atol=0
        )
        torch.testing.assert_close(
            first, torch.zeros_like(first), rtol=0, atol=0
        )
        torch.testing.assert_close(
            second,
            torch.tensor([16.0, 8.0, 0.0], device="cuda"),
            rtol=0,
            atol=0,
        )


FP8_QUALIFIED = (
    torch.cuda.is_available()
    and not torch.version.hip
    and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3))
    and importlib.util.find_spec("triton") is not None
)


@unittest.skipUnless(FP8_QUALIFIED, "qualified FP8 Triton CUDA required")
class TestTritonFP8Pair(unittest.TestCase):
    def check_selected_completion(self, rows, centers):
        from faiss.contrib.e_means.triton_assignment import assign_rows

        saved_rows, saved_centers = (
            rows.float().clone(),
            centers.float().clone(),
        )
        norms = saved_centers.square().sum(1)
        second = torch.empty(len(rows), device=rows.device)
        alternative = torch.empty(
            len(rows), dtype=torch.int64, device=rows.device
        )
        first, ids = assign_rows(
            rows,
            centers,
            norms,
            second_out=second,
            second_labels_out=alternative,
        )
        old_first, old_ids = assign_rows(rows, centers, norms)
        torch.testing.assert_close(first, old_first, rtol=0, atol=0)
        torch.testing.assert_close(ids, old_ids, rtol=0, atol=0)
        self.assertTrue(
            bool(
                (
                    (alternative >= 0)
                    & (alternative < len(centers))
                    & (alternative != ids)
                ).all()
            )
        )
        # Completion oracle follows ACTUAL selected IDs, never an exact-rank rescan.
        expected_first = (
            (saved_rows.double() - saved_centers[ids].double())
            .square()
            .sum(1)
            .float()
        )
        expected_second = (
            (saved_rows.double() - saved_centers[alternative].double())
            .square()
            .sum(1)
            .float()
        )
        torch.testing.assert_close(first, expected_first, rtol=1e-5, atol=0)
        torch.testing.assert_close(second, expected_second, rtol=1e-5, atol=0)
        torch.testing.assert_close(rows.float(), saved_rows, rtol=0, atol=0)
        torch.testing.assert_close(
            centers.float(), saved_centers, rtol=0, atol=0
        )
        print(
            "FP8 selected-completion negative margins",
            int((second < first).sum()),
            "B/K/D",
            len(rows),
            len(centers),
            rows.shape[1],
        )
        return first, second, ids, alternative

    def test_nonlattice_completion_and_negative_margin(self):
        rows = torch.full((1, 128), 448.0, device="cuda")
        rows[:, -1] = 0
        centers = torch.full((2, 128), 448.0, device="cuda")
        centers[0, -1], centers[1, -1] = 0.25, 0.125
        first, second, ids, alternative = self.check_selected_completion(
            rows.to(torch.float8_e4m3fn), centers.to(torch.float8_e4m3fn)
        )
        # Matching FP32 norms/ranking cancel; old Top1 keeps ID0 despite the closer ID1.
        self.assertEqual((ids.item(), alternative.item()), (0, 1))
        self.assertEqual((first.item(), second.item()), (0.0625, 0.015625))
        self.assertEqual((second - first).item(), -0.046875)
        generator = torch.Generator().manual_seed(19)
        for k in (
            257,
            2048,
        ):  # Naturally selected direct-descriptor and split-K routes.
            with self.subTest(k=k):
                rows = (
                    (torch.randn((3, 512), generator=generator) * 16)
                    .cuda()
                    .to(torch.float8_e4m3fn)
                )
                centers = (
                    (torch.randn((k, 512), generator=generator) * 16)
                    .cuda()
                    .to(torch.float8_e4m3fn)
                )
                self.check_selected_completion(rows, centers)

    def check_pair(self, n, k, d):
        from faiss.contrib.e_means.triton_assignment import assign_rows

        # Small integer E4M3 lattice gives an independent exact rank/distance oracle.
        rows_cpu = (torch.arange(n * d).reshape(n, d) % 9 - 4).float()
        centers_cpu = torch.full((k, d), 4.0)
        centers_cpu[0] = rows_cpu[0]
        centers_cpu[-1] = rows_cpu[0]
        truth = (
            (rows_cpu.long()[:, None] - centers_cpu.long()[None])
            .square()
            .sum(2)
        )
        order = truth.argsort(dim=1, stable=True)
        rows = rows_cpu.cuda().to(torch.float8_e4m3fn)
        centers = centers_cpu.cuda().to(torch.float8_e4m3fn)
        saved_rows, saved_centers = (
            rows.view(torch.uint8).clone(),
            centers.view(torch.uint8).clone(),
        )
        norms = centers.float().square().sum(1)
        second = torch.empty(n, dtype=torch.float32, device="cuda")
        alternative = torch.empty(n, dtype=torch.int64, device="cuda")
        first, ids = assign_rows(
            rows,
            centers,
            norms,
            second_out=second,
            second_labels_out=alternative,
        )
        top1, top1_ids = assign_rows(rows, centers, norms)
        torch.testing.assert_close(first, top1, rtol=0, atol=0)
        torch.testing.assert_close(ids, top1_ids, rtol=0, atol=0)
        torch.testing.assert_close(ids.cpu(), order[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(
            alternative.cpu(), order[:, 1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            first.cpu().double(),
            truth.gather(1, order[:, :1]).flatten().double(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            second.cpu().double(),
            truth.gather(1, order[:, 1:2]).flatten().double(),
            rtol=0,
            atol=0,
        )
        self.assertTrue(bool((ids != alternative).all()))
        torch.testing.assert_close(
            rows.view(torch.uint8), saved_rows, rtol=0, atol=0
        )
        torch.testing.assert_close(
            centers.view(torch.uint8), saved_centers, rtol=0, atol=0
        )

    def test_fp8_direct_resident_descriptor_and_split_pair(self):
        from faiss.contrib.e_means import triton_assignment as ta

        for n, k, d in (
            (129, 17, 64),
            (7, 257, 128),
            (3, 257, ta._FULL_D_MAX_D + 64),
            (1, 2048, 64),
        ):
            with self.subTest(n=n, k=k, d=d):
                self.check_pair(n, k, d)

    def test_fp8_encoded_k1_and_calibrated_second_units(self):
        from faiss.contrib.e_means.triton_assignment import assign_rows
        from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
        from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer

        rows = torch.zeros((3, 64), dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            assign_rows(
                rows,
                rows[:1],
                torch.zeros(1, device="cuda"),
                second_out=torch.empty(3, device="cuda"),
            )
        for centered in (False, True):
            with self.subTest(centered=centered):
                q = Fp8Quantizer(1)
                calibration = (
                    torch.tensor(
                        [[100.0], [101.0], [102.0], [103.0]], device="cuda"
                    )
                    if centered
                    else torch.tensor(
                        [[-1.0], [0.0], [1.0]], device="cuda"
                    ).repeat(1024, 1)
                )
                q.train(calibration)
                self.assertEqual(q.shift is not None, centered)
                initial = torch.tensor(
                    [[100.0], [103.0]] if centered else [[-1.0], [1.0]],
                    device="cuda",
                )
                values = torch.tensor(
                    (
                        [[100.25], [100.75], [102.25]]
                        if centered
                        else [[-0.75], [0.0], [0.75]]
                    ),
                    device="cuda",
                )
                saved_initial, saved_values = initial.clone(), values.clone()
                decoded = q.decode(q.compute_codes(values)).double()
                decoded_centers = q.decode(q.compute_codes(initial)).double()
                truth = (
                    (decoded[:, None] - decoded_centers[None]).square().sum(2)
                )
                order = truth.argsort(dim=1, stable=True)
                model = QuantizedEmeans(
                    1,
                    2,
                    quantizer=q,
                    unused_threshold=0.1,
                    assignment_backend="triton",
                    nepochs=1,
                    batch_size=3,
                    alpha0=1,
                )
                model.begin(3, initial)
                prepared_rows = model._prepare_rows(
                    model._prepare_input(values)
                )
                centers, norms = model._prepare_torch_assignment(
                    model._centroids
                )
                second = torch.empty(3, device="cuda")
                first, ids = model._assign_rows(
                    prepared_rows, centers, norms, None, None, second
                )
                torch.testing.assert_close(
                    first.double(),
                    truth.gather(1, order[:, :1]).flatten(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    second.double(),
                    truth.gather(1, order[:, 1:2]).flatten(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    initial, saved_initial, rtol=0, atol=0
                )
                torch.testing.assert_close(values, saved_values, rtol=0, atol=0)


@unittest.skipUnless(
    FP8_QUALIFIED and torch.cuda.get_device_capability() == (9, 0),
    "CuTe SM90 required",
)
class TestCuTeEncodedPairBindings(unittest.TestCase):
    def test_row_owned_multiple_blocks_and_m_tail(self):
        from faiss.contrib.e_means import cute

        for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                base = (torch.arange(129 * 16).reshape(129, 16) % 5).float()
                if dtype != torch.uint8:
                    base -= 2
                initial_cpu = torch.full((384, 16), 4.0)
                initial_cpu[3] = base[0]  # Unique first in block0.
                other = base[0].clone()
                other[0] += 1
                initial_cpu[130], initial_cpu[260] = (
                    other,
                    other,
                )  # Tied second across blocks.
                initial_cpu[160:192] = base[
                    1
                ]  # Tie candidates span lane-owned fragments.
                truth = (
                    (base.long()[:, None] - initial_cpu.long()[None])
                    .square()
                    .sum(2)
                )
                order = truth.argsort(dim=1, stable=True)
                rows, initial = base.cuda().to(dtype), initial_cpu.cuda()
                prepared = cute._prepare_centroids(initial, dtype=dtype)
                raw_dtype = (
                    torch.float32
                    if dtype == torch.float8_e4m3fn
                    else torch.int64
                )
                first = torch.empty(129, dtype=raw_dtype, device="cuda")
                second = torch.empty_like(first)
                distance, ids = cute._assign_rows(
                    rows, prepared, first, second_out=second
                )
                binding = prepared.bindings[
                    cute._binding_key(129, raw_dtype, True)
                ]
                self.assertEqual(binding.launcher.__name__, "launch_row_owned")
                torch.testing.assert_close(
                    ids.cpu(), order[:, 0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    binding.second_labels.cpu(), order[:, 1], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    first.cpu().double(),
                    truth.gather(1, order[:, :1]).flatten().double(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    second.cpu().double(),
                    truth.gather(1, order[:, 1:2]).flatten().double(),
                    rtol=0,
                    atol=0,
                )

    def test_current_encoded_row_owned_and_tiled_routes(self):
        from faiss.contrib.e_means import cute

        for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn):
            for k in (
                128,
                129,
            ):  # Existing legal row-owned shape and tiled shape.
                with self.subTest(dtype=dtype, k=k):
                    base = (torch.arange(5 * 16).reshape(5, 16) % 5).float()
                    if dtype != torch.uint8:
                        base -= 2
                    centers = torch.full((k, 16), 4.0)
                    centers[0] = base[0]
                    centers[-1] = base[0]
                    truth = (
                        (base.long()[:, None] - centers.long()[None])
                        .square()
                        .sum(2)
                    )
                    order = truth.argsort(dim=1, stable=True)
                    rows = base.cuda().to(dtype)
                    initial = centers.cuda()
                    saved_rows = rows.view(torch.uint8).clone()
                    saved_initial = initial.clone()
                    prepared = cute._prepare_centroids(initial, dtype=dtype)
                    raw_dtype = (
                        torch.float32
                        if dtype == torch.float8_e4m3fn
                        else torch.int64
                    )
                    first = torch.empty(5, dtype=raw_dtype, device="cuda")
                    second = torch.empty_like(first)
                    returned, ids = cute._assign_rows(
                        rows, prepared, distance_out=first, second_out=second
                    )
                    binding = prepared.bindings[
                        cute._binding_key(5, raw_dtype, True)
                    ]
                    self.assertEqual(
                        binding.launcher.__name__,
                        "launch_row_owned" if k == 128 else "launch_tiled_gemm",
                    )
                    torch.testing.assert_close(
                        ids.cpu(), order[:, 0], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        binding.second_labels.cpu(), order[:, 1], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        first.cpu().double(),
                        truth.gather(1, order[:, :1]).flatten().double(),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        second.cpu().double(),
                        truth.gather(1, order[:, 1:2]).flatten().double(),
                        rtol=0,
                        atol=0,
                    )
                    top1, top1_ids = cute._assign_rows(rows, prepared)
                    torch.testing.assert_close(
                        top1, first.float(), rtol=0, atol=0
                    )
                    torch.testing.assert_close(top1_ids, ids, rtol=0, atol=0)
                    torch.testing.assert_close(
                        rows.view(torch.uint8), saved_rows, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        initial, saved_initial, rtol=0, atol=0
                    )

    def test_cute_fp8_true_completed_negative_pair(self):
        from faiss.contrib.e_means import cute

        for k in (128, 129):
            with self.subTest(k=k):
                base = torch.full((1, 128), 448.0, device="cuda")
                base[:, -1] = 0
                centers = torch.full((k, 128), 448.0, device="cuda")
                centers[:, -1] = 1
                centers[0, -1], centers[1, -1] = 0.25, 0.125
                rows = base.to(torch.float8_e4m3fn)
                prepared = cute._prepare_centroids(centers, dtype=rows.dtype)
                second = torch.empty(1, device="cuda")
                first, ids = cute._assign_rows(
                    rows, prepared, second_out=second
                )
                old_first, old_ids = cute._assign_rows(rows, prepared)
                torch.testing.assert_close(first, old_first, rtol=0, atol=0)
                torch.testing.assert_close(ids, old_ids, rtol=0, atol=0)
                alternative = prepared.bindings[
                    cute._binding_key(1, torch.float32, True)
                ].second_labels
                expected = (
                    (rows.float() - centers[alternative].to(rows.dtype).float())
                    .square()
                    .sum(1)
                )
                torch.testing.assert_close(second, expected, rtol=0, atol=0)
                self.assertEqual((ids.item(), alternative.item()), (0, 1))
                self.assertEqual((second - first).item(), -0.046875)

    def test_calibrated_integer_raw_pair_and_returned_units(self):
        from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
        from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer

        class Observed(QuantizedEmeans):
            def _assign_rows(self, *args, **kwargs):
                self.assign_calls += 1
                return super()._assign_rows(*args, **kwargs)

        base = torch.tensor(
            [[10.0, 2.0], [10.25, 2.5], [11.0, 3.0]], device="cuda"
        )
        q = Int8Quantizer(2)
        q.train(base)
        initial = base[[0, 2]].clone()
        rows = q.compute_codes(base)
        codes_c = q.compute_codes(initial)
        truth = (rows.long()[:, None] - codes_c.long()[None]).square().sum(2)
        order = truth.argsort(dim=1, stable=True)
        results = []
        for backend in ("triton", "cute"):
            model = Observed(
                2,
                2,
                quantizer=q,
                unused_threshold=0.1,
                assignment_backend=backend,
            )
            model.assign_calls = 0
            model.begin(3, initial)
            actual = model._prepare_rows(model._prepare_input(rows))
            centers, norms = model._prepare_torch_assignment(model._centroids)
            first = torch.empty(3, dtype=torch.int64, device="cuda")
            second = torch.empty_like(first)
            distance, labels = model._assign_rows(
                actual, centers, norms, first, None, second
            )
            self.assertEqual(model.assign_calls, 1)
            expected = torch.ldexp(
                truth.gather(1, order[:, :1]).flatten().float(),
                q._distance_exponent,
            )
            torch.testing.assert_close(
                first, truth.gather(1, order[:, :1]).flatten(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                second, truth.gather(1, order[:, 1:2]).flatten(), rtol=0, atol=0
            )
            torch.testing.assert_close(distance, expected, rtol=0, atol=0)
            ordinary, ordinary_ids = model._assign_rows(actual, centers, norms)
            self.assertEqual(model.assign_calls, 2)
            torch.testing.assert_close(distance, ordinary, rtol=0, atol=0)
            torch.testing.assert_close(labels, ordinary_ids, rtol=0, atol=0)
            results.append(distance)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_fp8_statistics_from_producing_mode_and_shape(self):
        from faiss.contrib.e_means.e_means_quantized import (
            QuantizedEmeans,
            statistics_codes,
        )
        from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer

        q = Fp8Quantizer(1)
        q.train(
            torch.tensor([[-1.0], [0.0], [1.0]], device="cuda").repeat(1024, 1)
        )
        model = QuantizedEmeans(
            1, 2, quantizer=q, unused_threshold=0.1, assignment_backend="cute"
        )
        model.begin(8, torch.tensor([[-1.0], [1.0]], device="cuda"))
        for n, pair, offset in (
            (3, False, 0.0),
            (3, True, 0.25),
            (5, True, -0.25),
            (2, False, 0.5),
            (3, True, -0.5),
        ):
            rows = (torch.arange(n, device="cuda").float() * 0.125 + offset)[
                :, None
            ]
            saved_rows = rows.clone()
            expected_codes = q.compute_codes(saved_rows)
            actual = model._prepare_rows(model._prepare_input(rows))
            centers, norms = model._prepare_torch_assignment(model._centroids)
            second = torch.empty(n, device="cuda") if pair else None
            model._assign_rows(actual, centers, norms, None, None, second)
            codes = statistics_codes(
                actual,
                centers,
                "cute",
                q,
                second=pair,
                distance_dtype=torch.float32,
            )
            torch.testing.assert_close(
                codes.view(torch.uint8),
                expected_codes.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                q.decode(codes), q.decode(expected_codes), rtol=0, atol=0
            )
            torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
