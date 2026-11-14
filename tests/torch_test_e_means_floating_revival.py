# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Local floating revival, decoded statistics and caller batch contracts."""

import unittest
from unittest import mock

import torch

from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.e_means_torch import Emeans
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.torch_backend import CuteBackend, TritonBackend


def _reference(codes, values, centroids, axes, dtype, quantizer, spherical=False):
    """Direct FP64 represented-distance and decoded Oja reference."""
    represented = (quantizer.compute_codes(centroids) if quantizer is not None and quantizer.center
                   else centroids.to(dtype)).cpu().double()
    rows = codes.cpu().double()
    distance = (-2 * rows @ represented.T if spherical else
                (rows[:, None] - represented[None]).square().sum(2))
    order = torch.argsort(distance, dim=1, stable=True)
    labels = order[:, 0]
    margin = distance.gather(1, order[:, 1:2]) - distance.gather(1, order[:, :1])
    y = values.cpu().double() - centroids.cpu().double()[labels]
    p = (y * axes.cpu().double()[labels]).sum(1, keepdim=True)
    sums = torch.zeros_like(centroids, device="cpu", dtype=torch.float64)
    sums.index_add_(0, labels, values.cpu().double())
    counts = torch.bincount(labels, minlength=len(centroids)).double()
    removal = torch.zeros((len(centroids), centroids.shape[1] + 1), dtype=torch.float64)
    removal.index_add_(0, labels, torch.cat((p * y, margin), dim=1))
    return sums, counts, removal, labels


class _DecodedOffset(CastQuantizer):
    def decode(self, codes):
        return super().decode(codes) + codes.new_tensor([.5, -.25, .125], dtype=torch.float32)


class TestFloatingRevivalCPU(unittest.TestCase):
    def test_stock_cast_reuse_preserves_decoder_and_matrix_overrides(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                codes = torch.tensor([[.25, .5], [1., 2.]], dtype=dtype)
                q = CastQuantizer(2, dtype)
                model = QuantizedEmeans(2, 2, quantizer=q)
                reused = model._statistics_rows(codes)
                torch.testing.assert_close(reused.float(), codes.float(), rtol=0, atol=0)
                self.assertEqual(reused.data_ptr(), codes.data_ptr())
                seen = []

                def custom_decode(values):
                    seen.append("decode")
                    return values.float() + 1

                with mock.patch.object(q, "decode", custom_decode):
                    torch.testing.assert_close(model._statistics_rows(codes), codes.float() + 1, rtol=0, atol=0)
                matrix = q._matrix

                def custom_matrix(values, **kwargs):
                    seen.append("matrix")
                    return matrix(values, **kwargs) + 1

                with mock.patch.object(q, "_matrix", custom_matrix):
                    torch.testing.assert_close(model._statistics_rows(codes), codes.float() + 1, rtol=0, atol=0)
                    self.assertFalse(model._replay_representation_eligible())
                self.assertEqual(seen, ["decode", "matrix"])

    def test_native_cpu_partial_batches_reference_and_natural_split(self):
        # Torch CPU only: no native Faiss arithmetic or CUDA codec claim.
        initial = torch.tensor([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.], [12., 0., 0.]])
        ids = torch.arange(25)
        rows = initial[2].expand(25, -1) + torch.stack((.125 + .125 * (ids % 3),
                                                      .125 * (ids % 2), -.125 * (ids % 3)), 1)
        kwargs = dict(nepochs=1, batch_size=5, alpha0=.5, seed=7, assignment_backend="native",
                      assignment_chunk_size=3)
        actual = QuantizedEmeans(3, 4, quantizer=None, **kwargs)
        reference = Emeans(3, 4, **kwargs)
        for model in (actual, reference):
            model.begin(25, initial)
            model.accumulate(rows[:0])
            self.assertEqual(model._input_dtype, torch.float32)
            self.assertIsNotNone(model._removal_stats)
            self.assertIsNone(model.recipe["unused_threshold"])
        observed = {}
        apply = actual._apply_splits

        def observed_split(source, target):
            observed.update(source=source.clone(), target=target.clone(),
                            centroids=actual._centroids.clone(), usage=actual._usage.clone(),
                            removal=actual._removal_stats.clone(), axes=actual._axes.clone())
            return apply(source, target)

        with mock.patch.object(actual, "_apply_splits", observed_split):
            for step, begin in enumerate(range(0, 25, 5)):
                batch = rows[begin:begin + 5]
                previous = actual._removal_stats.clone()
                c, a = actual._centroids.clone(), actual._axes.clone()
                sums, counts, removal, _ = _reference(batch, batch, c, a, torch.float32, None)
                partial = actual.accumulate(batch[:1])
                self.assertIsNone(partial["sums"])
                self.assertEqual(actual.accumulate(batch[:0])["n"], 1)
                torch.testing.assert_close(actual._centroids, c, rtol=0, atol=0)
                torch.testing.assert_close(actual._axes, a, rtol=0, atol=0)
                stats = actual.accumulate(batch[1:])
                torch.testing.assert_close(stats["sums"].double(), sums, rtol=0, atol=0)
                torch.testing.assert_close(stats["counts"].double(), counts, rtol=0, atol=0)
                torch.testing.assert_close(stats["removal"].double(), removal, rtol=1e-5, atol=1e-5)
                from faiss.contrib.e_means.base import _scheduled_alpha
                alpha = _scheduled_alpha(actual.recipe["alpha0"], step, actual.recipe["total_steps"])
                expected = previous.double() * (1 - alpha) + removal * alpha
                actual.update(stats)
                reference.update(reference.accumulate(batch))
                # The fifth update performs the real standard revival check;
                # compare the pre-split EMA before verifying its transfer.
                checked = observed["removal"] if step == 4 else actual._removal_stats
                torch.testing.assert_close(checked.double(), expected, rtol=1e-5, atol=1e-5)
                self.assertEqual(actual._step, step + 1)
                self.assertFalse(bool(actual._batch_removal.any()))
                for name in ("_centroids", "_usage", "_removal_stats", "_axes", "_moved", "_pass_sums", "_pass_counts"):
                    torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=1e-5, atol=1e-5)
        self.assertTrue(observed)
        self.assertGreater(actual.n_splits, 0)
        source, target = observed["source"], observed["target"]
        expected = observed["removal"].clone()
        expected[source] /= 2
        expected[target] = expected[source]
        torch.testing.assert_close(actual._removal_stats, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual._axes[target], observed["axes"][source], rtol=0, atol=0)
        torch.testing.assert_close(actual._usage[target], observed["usage"][source] / 2, rtol=0, atol=0)
        self.assertTrue(bool(actual._moved[source].all() & actual._moved[target].all()))
        for model in (actual, reference):
            model.end_epoch()
            model.finalize()
        torch.testing.assert_close(actual.centroids, reference.centroids, rtol=1e-5, atol=1e-5)
        centered = CastQuantizer(3, torch.bfloat16, center=True)
        centered.train(initial)
        with self.assertRaisesRegex(ValueError, "CPU E-means requires quantizer=None"):
            QuantizedEmeans(3, 4, quantizer=centered).begin(25, initial)


@unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip, "NVIDIA CUDA required")
class TestFloatingRevivalCUDA(unittest.TestCase):
    def test_decoded_logical_batches_match_independent_statistics(self):
        initial = torch.tensor([[-4., 0., 0.], [0., 1., 0.], [4., -1., 0.], [12., 0., 1.]], device="cuda")
        ids = torch.arange(13, device="cuda")
        rows = initial[ids % 4] + torch.stack((.125 + .125 * (ids % 3),
                                              .125 * (ids % 2), -.125 * (ids % 3)), 1)
        cases = [("native", dtype, kind) for dtype, kind in
                 ((torch.float32, "native"), (torch.float16, "native"), (torch.bfloat16, "native"),
                  (torch.float16, "cast"), (torch.bfloat16, "cast"),
                  (torch.float16, "centered"), (torch.bfloat16, "centered"),
                  (torch.bfloat16, "custom"), (torch.bfloat16, "spherical"))]
        cases += [("triton", dtype, kind) for _, dtype, kind in cases]
        if torch.cuda.get_device_capability() == (9, 0):
            cases += [("cute", torch.bfloat16, kind) for kind in ("cast", "centered", "custom", "spherical")]
        for backend, dtype, kind in cases:
            with self.subTest(backend=backend, dtype=dtype, kind=kind):
                if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
                    self.skipTest("BF16 requires NVIDIA SM80 or later")
                q = None if kind == "native" else (
                    _DecodedOffset(3, dtype) if kind == "custom" else CastQuantizer(3, dtype, center=kind == "centered"))
                source, seeds = rows, initial
                if kind == "centered":
                    shift = initial.new_tensor([100., -50., 30.])
                    q.train(torch.stack((shift - 1, shift + 1)))
                    source, seeds = rows + shift, initial + shift
                if kind == "spherical":
                    source = torch.nn.functional.normalize(source, dim=1)
                    seeds = torch.nn.functional.normalize(seeds, dim=1)
                codes = source.to(dtype) if q is None else q.compute_codes(source)
                values = codes.float() if q is None else q.decode(codes)
                models = [QuantizedEmeans(3, 4, nepochs=1, batch_size=5, alpha0=.6, seed=7,
                                         quantizer=q, assignment_backend=backend, assignment_chunk_size=3,
                                         spherical=kind == "spherical") for _ in range(2)]
                for model in models:
                    model.begin(13, seeds)
                    model.accumulate(codes[:0])  # Typed empty input resolves native policy too.
                    self.assertIsNotNone(model._removal_stats)
                    self.assertIsNone(model.recipe["unused_threshold"])
                    self.assertEqual(model._input_dtype, dtype)
                    model._axes.zero_()
                    model._axes[:, 0] = 1
                    model._removal_stats.fill_(.25)
                for step, begin in enumerate((0, 5, 10)):
                    end = min(begin + 5, 13)
                    c, a = models[0]._readout().clone(), models[0]._axes.clone()
                    reference = _reference(codes[begin:end], values[begin:end], c, a, dtype, q, kind == "spherical")
                    previous = models[0]._removal_stats.clone()
                    stats = []
                    for model, partial in zip(models, (False, True)):
                        unchanged = (model._centroids.clone(), model._axes.clone())
                        if partial:
                            first = model.accumulate(source[begin:begin + 1] if q is not None else codes[begin:begin + 1])
                            self.assertIsNone(first["sums"])
                            self.assertEqual(model.accumulate(codes[begin:begin])["n"], 1)
                            for name, expected in zip(("_centroids", "_axes"), unchanged):
                                torch.testing.assert_close(getattr(model, name), expected, rtol=0, atol=0)
                            batch = model.accumulate(codes[begin + 1:end])
                        else:
                            batch = model.accumulate(source[begin:end] if q is not None else codes[begin:end])
                        sums, counts, removal, _ = reference
                        torch.testing.assert_close(batch["sums"].cpu().double(), sums, rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(batch["counts"].cpu().double(), counts, rtol=0, atol=0)
                        torch.testing.assert_close(batch["removal"].cpu().double(), removal, rtol=1e-5, atol=1e-5)
                        stats.append(batch)
                    from faiss.contrib.e_means.base import _scheduled_alpha
                    alpha = _scheduled_alpha(models[0].recipe["alpha0"], step, models[0].recipe["total_steps"])
                    expected = previous.cpu().double() * (1 - alpha) + alpha * reference[2]
                    for model, batch in zip(models, stats):
                        model.update(batch)
                        self.assertEqual(model._step, step + 1)
                        torch.testing.assert_close(model._removal_stats.cpu().double(), expected, rtol=1e-5, atol=1e-5)
                        direction = expected[:, :-1]
                        direction = direction / direction.norm(dim=1, keepdim=True)
                        torch.testing.assert_close(model._axes.cpu().double(), direction, rtol=1e-5, atol=1e-5)
                        self.assertFalse(bool(model._batch_removal.any()))
                    for name in ("_centroids", "_usage", "_removal_stats", "_axes", "_pass_sums", "_pass_counts"):
                        torch.testing.assert_close(getattr(models[0], name), getattr(models[1], name), rtol=1e-5, atol=1e-5)
                for model in models:
                    model.end_epoch()
                    model.finalize()

    def test_live_split_hooks_are_honored_after_device_admission(self):
        initial = torch.tensor([[-4., 0.], [0., 0.], [4., 0.], [12., 0.]], device="cuda")
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
                    self.skipTest("BF16 requires NVIDIA SM80 or later")
                model = QuantizedEmeans(2, 4, nepochs=1, batch_size=4, quantizer=None,
                                        assignment_backend="triton")
                model.begin(20, initial)
                model.accumulate(initial[:0].to(dtype))
                self.assertTrue(model._device_check())
                model._removal_stats.copy_(initial.new_tensor([[0., 0., 0.], [3., 0., 10.],
                                                               [0., 0., 0.], [2., 0., 10.]]))
                model._usage.fill_(1)
                calls = []
                apply, separate = model._apply_splits, model._separate_split_values
                state = model._execution_backend.removal_split

                def observed_apply(source, target):
                    calls.append("apply")
                    return apply(source, target)

                def observed_separate(donor, child):
                    calls.append("separate")
                    return separate(donor, child)

                def observed_state(*args):
                    calls.append("state")
                    return state(*args)

                with mock.patch.object(model, "_apply_splits", observed_apply), \
                     mock.patch.object(model, "_separate_split_values", observed_separate), \
                     mock.patch.object(model._execution_backend, "removal_split", observed_state):
                    self.assertFalse(model._device_check())
                    self.assertEqual(model._revive_unused(model._split_rng), 2)
                self.assertEqual(calls, ["apply", "separate", "state"])
                self.assertEqual(model.n_splits, 2)
                self.assertEqual(model._moved.tolist(), [True] * 4)
                torch.testing.assert_close(model._usage, initial.new_full((4,), .5), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
                         "BF16 requires NVIDIA SM80 or later")
    def test_old_owner_and_update_hooks_receive_decoded_statistics(self):
        initial = torch.tensor([[100., -50.], [104., -50.], [112., -50.]], device="cuda")
        rows = initial[[0, 1, 2, 0, 1]] + .25
        q = CastQuantizer(2, torch.bfloat16, center=True)
        q.train(torch.tensor([[99., -51.], [101., -49.]], device="cuda"))
        codes = q.compute_codes(rows)
        seen = []

        class OldOwner(TritonBackend):
            def accumulate_removal(self, values, labels, stats, distance, second, axes, centroids, removal):
                seen.append(values.clone())
                return super().accumulate_removal(values, labels, stats, distance, second, axes, centroids, removal)

        class UpdateObserved(QuantizedEmeans):
            def _update_removal_statistics(self, alpha, decay):
                self.updates += 1
                return super()._update_removal_statistics(alpha, decay)

        model = UpdateObserved(2, 3, nepochs=1, batch_size=5, alpha0=.5, quantizer=q,
                               assignment_backend="triton", assignment_chunk_size=5)
        model.updates = 0
        model.begin(10, initial)
        model.accumulate(codes[:0])
        model._execution_backend = OldOwner(model.centroid_chunk_size)
        expected = _reference(codes, q.decode(codes), model._centroids, model._axes, q.dtype, q)
        # This is the same fold-capable private call used by graph capture;
        # the actual update hook must still run exactly once at update().
        batch = model._accumulate_core(codes, ema=(model._removal_stats, .5, .5))
        self.assertFalse(model._removal_folded)
        self.assertFalse(model._capture_operations_match())
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], q.decode(codes), rtol=0, atol=0)
        torch.testing.assert_close(batch["removal"].cpu().double(), expected[2], rtol=1e-5, atol=1e-5)
        model.update(batch)
        self.assertEqual(model.updates, 1)
        torch.testing.assert_close(model._removal_stats.cpu().double(), expected[2] * .5, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
                         "BF16 requires NVIDIA SM80 or later")
    def test_ordinary_accumulation_owner_receives_decoded_rows(self):
        initial = torch.tensor([[100., -50.], [104., -50.], [112., -50.]], device="cuda")
        rows = initial[[0, 1, 2, 0, 1]] + .25
        q = CastQuantizer(2, torch.bfloat16, center=True)
        q.train(torch.tensor([[99., -51.], [101., -49.]], device="cuda"))
        codes = q.compute_codes(rows)
        base = CuteBackend if torch.cuda.get_device_capability() == (9, 0) else TritonBackend
        seen = []

        class SumOwner(base):
            def accumulate(self, values, labels, stats, batch_size, indexed=False):
                seen.append(values.clone())
                return super().accumulate(values, labels, stats, batch_size, indexed)

        model = QuantizedEmeans(2, 3, nepochs=1, batch_size=5, quantizer=q,
                                assignment_backend=base.name, assignment_chunk_size=5)
        model.begin(10, initial)
        model.accumulate(codes[:0])
        model._execution_backend = SumOwner(model.centroid_chunk_size)
        expected = _reference(codes, q.decode(codes), model._centroids, model._axes, q.dtype, q)
        batch = model.accumulate(codes)
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], q.decode(codes), rtol=0, atol=0)
        torch.testing.assert_close(batch["sums"].cpu().double(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(batch["removal"].cpu().double(), expected[2], rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
                         "CuTe BF16 requires NVIDIA SM90")
    def test_old_cute_owner_keeps_completed_second_on_decoded_rows(self):
        # One represented coordinate isolates row-basis/completion ownership;
        # the broader floating references cover norm-plus-dot rounding.
        initial = torch.tensor([[100.], [104.], [112.]], device="cuda")
        rows = initial[[0, 1, 2, 0, 1]] + .25
        q = CastQuantizer(1, torch.bfloat16, center=True)
        q.train(torch.tensor([[99.], [101.]], device="cuda"))
        codes = q.compute_codes(rows)
        seen = []

        class OldOwner(CuteBackend):
            def accumulate_removal(self, values, labels, stats, distance, second, axes, centroids, removal):
                seen.append((values.clone(), second.clone()))
                return super().accumulate_removal(values, labels, stats, distance, second, axes, centroids, removal)

        model = QuantizedEmeans(1, 3, nepochs=1, batch_size=5, quantizer=q,
                                assignment_backend="cute", assignment_chunk_size=5)
        model.begin(10, initial)
        model.accumulate(codes[:0])
        model._execution_backend = OldOwner(model.centroid_chunk_size)
        expected = _reference(codes, q.decode(codes), model._centroids, model._axes, q.dtype, q)
        batch = model.accumulate(codes)
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0][0], q.decode(codes), rtol=0, atol=0)
        direct = (codes.double()[:, None] - q.compute_codes(initial).double()[None]).square().sum(2)
        second = direct.sort(dim=1, stable=True).values[:, 1]
        torch.testing.assert_close(seen[0][1].double(), second, rtol=0, atol=0)
        torch.testing.assert_close(batch["sums"].cpu().double(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(batch["counts"].cpu().double(), expected[1], rtol=0, atol=0)
        torch.testing.assert_close(batch["removal"].cpu().double(), expected[2], rtol=1e-5, atol=1e-5)

    def test_fp16_range_and_sums_only_external_update_remain_explicit(self):
        initial = torch.full((2, 1), 65504., device="cuda")
        for backend in ("native", "triton"):
            for quantizer in (None, CastQuantizer(1, torch.float16)):
                with self.subTest(backend=backend, quantizer=quantizer):
                    model = QuantizedEmeans(1, 2, nepochs=1, batch_size=2, quantizer=quantizer,
                                            assignment_backend=backend)
                    model.begin(4, initial)
                    model.accumulate(initial[:0].half())
                    self.assertIsNotNone(model._removal_stats)
                    self.assertFalse(model._device_check())
                    before = model._centroids.clone(), model._usage.clone()
                    with self.assertRaisesRegex(ValueError, "65504"):
                        model._apply_splits(torch.tensor([0], device="cuda"), torch.tensor([1], device="cuda"))
                    torch.testing.assert_close(model._centroids, before[0], rtol=0, atol=0)
                    torch.testing.assert_close(model._usage, before[1], rtol=0, atol=0)
                    self.assertEqual(model.n_splits, 0)
                    with self.assertRaisesRegex(ValueError, "from this model's accumulate"):
                        model.update(dict(sums=torch.zeros_like(initial), counts=initial.new_zeros(2),
                                          objective=initial.new_zeros((), dtype=torch.float64), n=2))

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
                         "BF16 requires NVIDIA SM80 or later")
    def test_centered_closing_handover_uses_decoded_rows(self):
        # The actual closing loop forces one physical split at the normal
        # fifth update. This is a retention/row-basis witness, not a claim
        # about which donors the revival algorithm should select.
        initial = torch.tensor([[100.25, -49.5], [104.25, -49.5], [112.25, -49.5]], device="cuda")
        original_labels = torch.arange(10, device="cuda") % 3
        rows = initial[original_labels]
        q = CastQuantizer(2, torch.bfloat16, center=True)
        q.train(torch.tensor([[99., -51.], [101., -49.]], device="cuda"))
        codes = q.compute_codes(rows)
        # Independent decoded reference: all fixture values are exactly
        # represented dyadics, and the fitted centre is [100, -50].
        decoded = codes.float() + torch.tensor([100., -50.], device="cuda")
        torch.testing.assert_close(decoded, rows, rtol=0, atol=0)

        class ForcedClosingMove(QuantizedEmeans):
            def _revive_unused(self, rng):
                return self._apply_splits(
                    torch.tensor([2], device="cuda"), torch.tensor([1], device="cuda"))

            def _handover_finish(self, order, count):
                self.before_labels = self._handover["labels"].clone()
                self.target_times = self._handover["moved_at"].clone()
                self.before_sums = self._pass_sums.clone()
                self.before_counts = self._pass_counts.clone()
                self.split_positions = self._centroids.clone()
                self.moved_slots = self._moved.clone()
                super()._handover_finish(order, count)
                self.after_sums = self._pass_sums.clone()
                self.after_counts = self._pass_counts.clone()

        backends = ("native", "triton", "cute") if torch.cuda.get_device_capability() == (9, 0) else ("native", "triton")
        for backend in backends:
            with self.subTest(backend=backend):
                model = ForcedClosingMove(2, 3, nepochs=1, batch_size=2, alpha0=.5,
                                          assignment_chunk_size=2, assignment_backend=backend,
                                          seed=7, quantizer=q)
                centroids, labels = model.train(rows, initial)
                self.assertEqual(model.recipe["total_steps"], 5)
                self.assertEqual(model.n_splits, 1)
                self.assertEqual(model.recipe["closing_handover"], "search")
                self.assertEqual(model.closing_handover_stats["searched"], 3)
                self.assertEqual(model.target_times.tolist(), [-1, 10, -1])
                self.assertEqual(model.moved_slots.tolist(), [False, True, True])
                torch.testing.assert_close(model.before_labels, original_labels, rtol=0, atol=0)
                expected_labels = torch.where(original_labels == 1, 0, original_labels)
                torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)

                sums = model.before_sums.clone()
                sums[0] += decoded[original_labels == 1].sum(0)
                counts = model.before_counts.clone()
                counts[0] += 3
                torch.testing.assert_close(model.after_sums, sums, rtol=0, atol=0)
                torch.testing.assert_close(model.after_counts, counts, rtol=0, atol=0)
                self.assertEqual(model.after_counts.sum().item(), 13)
                expected = model.split_positions.clone()
                expected[0] = sums[0] / counts[0]
                torch.testing.assert_close(centroids, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
