# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Encoded closing against a scalar search/twin oracle in statistics units."""

import importlib.util
import unittest

import torch

from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer


def _scalar_labels(
    labels, order, moved, events, values, codes, centers, represented
):
    """Independent row/event walk; stale searches are never twin-routed."""
    out, searched = labels.clone(), 0
    for rank, row in enumerate(order.tolist()):
        label = int(labels[row])
        if rank < int(moved[label]):
            distances = (
                (codes[row].double() - represented.double()).square().sum(1)
            )
            out[row] = distances.argmin()
            searched += 1
            continue
        later = [
            (birth, target)
            for source, target, birth in events
            if source == label and birth > rank
        ]
        if not later:
            continue
        birth, child = min(later)
        if int(moved[child]) != birth:
            continue
        own = (values[row].float() - centers[label]).double().square().sum()
        twin = (values[row].float() - centers[child]).double().square().sum()
        if twin < own or (twin == own and child < label):
            out[row] = child
    return out, searched


CUDA = torch.cuda.is_available() and torch.version.hip is None
TRITON = importlib.util.find_spec("triton") is not None


class TestCastClosingCPU(unittest.TestCase):
    def test_plain_cast_closing_uses_training_decode_and_handles_bf16(self):
        class OffsetCast(CastQuantizer):
            def decode(self, codes):
                self.calls = getattr(self, 'calls', 0) + 1
                return super().decode(codes).float() + .25

        class CPUCastOwner(QuantizedEmeans):
            def _assign_rows(self, rows, centroids, norm=None, first=None,
                             labels=None, second=None, second_labels=None, **kwargs):
                # The stock low-precision addmm(out_dtype=FP32) is CUDA-only.
                # This genuine caller hook provides FP32 CPU assignment over
                # the same represented codes; production has no fallback.
                return self._execution_backend.assign(
                    rows.float(), centroids.float(), norm, self.spherical,
                    first, labels, second, second_labels, **kwargs)

        rows = torch.tensor([0., 100., 1., 101., 2., 102.,
                             3., 103., 4., 104., 2., 102.])[:, None]
        initial = torch.tensor([[0.], [100.], [110.]])
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                codec = OffsetCast(1, dtype)
                model = CPUCastOwner(1, 3, quantizer=codec, nepochs=1,
                                     batch_size=2, alpha0=.5, seed=7,
                                     assignment_backend='native')
                model._choose_splits = lambda rng: (torch.tensor([0]), torch.tensor([1]))
                finish = model._handover_finish
                recorded = {}

                def observe(order, n, *, model=model, codec=codec, finish=finish,
                            recorded=recorded, dtype=dtype):
                    h = model._handover
                    events = [(int(source[0]), int(target[0]), birth)
                              for target, source, count, birth in h['events']]
                    values = rows.to(dtype).float() + .25
                    expected, searched = _scalar_labels(
                        h['labels'], order[:n], h['moved_at'], events, values,
                        rows.to(dtype), model._centroids,
                        model._centroids.to(dtype))
                    calls = codec.calls
                    result = finish(order, n)
                    self.assertGreater(codec.calls, calls)
                    recorded.update(labels=expected, searched=searched,
                                    sums=model._pass_sums.clone(),
                                    counts=model._pass_counts.clone())
                    return result

                model._handover_finish = observe
                centers, labels = model.train(rows, initial)
                self.assertEqual(model.n_splits, 1)
                self.assertEqual(model.recipe['closing_handover'], 'search')
                self.assertEqual(model.closing_handover_stats,
                                 {'searched': recorded['searched']})
                torch.testing.assert_close(labels, recorded['labels'], rtol=0, atol=0)
                values = rows.to(dtype).float() + .25
                sums = torch.zeros_like(initial).index_add_(0, labels, values)
                counts = torch.bincount(labels, minlength=3).double()
                torch.testing.assert_close(recorded['sums'], sums, rtol=0, atol=0)
                torch.testing.assert_close(recorded['counts'], counts, rtol=0, atol=0)
                populated = counts > 0
                torch.testing.assert_close(
                    centers[populated],
                    (sums[populated].double() / counts[populated, None]).float(),
                    rtol=0, atol=0)


class TestIntegerClosingReduction(unittest.TestCase):
    @unittest.skipUnless(CUDA and TRITON, "NVIDIA CUDA and Triton required")
    def test_float64_closing_sums_keep_values_beyond_float32_precision(self):
        from faiss.contrib.e_means.triton_assignment import closing_sums

        rows = torch.tensor([2**40 + 1, -(2**40), 2**40 + 2, -(2**40)],
                            dtype=torch.float64, device="cuda")[:, None]
        sums = torch.zeros(1, 1, dtype=torch.float64, device="cuda")
        counts = torch.zeros(1, dtype=torch.float64, device="cuda")
        closing_sums(rows, torch.arange(4, device="cuda"),
                     torch.tensor([0, 4], device="cuda"), sums, counts, 2)
        torch.testing.assert_close(sums, rows.new_tensor([[3.]]), rtol=0, atol=0)
        torch.testing.assert_close(counts, rows.new_tensor([4.]), rtol=0, atol=0)

    @unittest.skipUnless(
        CUDA and TRITON and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3)),
        "existing admitted FP8 CUDA device and Triton required")
    def test_fp8_closing_sums_widen_exactly_through_float32(self):
        from faiss.contrib.e_means.triton_assignment import closing_sums

        rows = torch.tensor([0., -0., 448., -448., 2**-9, -(2**-9),
                             1.125, -3.5, 2., 16., -16., .5, .25],
                            device="cuda").to(torch.float8_e4m3fn)[:, None]
        index = torch.tensor([3, 0, 5, 6, 2, 1, 4, 8, 7, 12, 9, 11, 10], device="cuda")
        sums = torch.zeros(2, 1, dtype=torch.float64, device="cuda")
        counts = torch.zeros(2, dtype=torch.float64, device="cuda")
        closing_sums(rows, index, torch.tensor([0, 5, 13], device="cuda"),
                     sums, counts, 4)
        values = rows.float().double()
        expected = torch.stack((values[index[:5]].sum(0), values[index[5:]].sum(0)))
        torch.testing.assert_close(sums, expected, rtol=0, atol=0)
        torch.testing.assert_close(counts, counts.new_tensor([5., 8.]), rtol=0, atol=0)

    @unittest.skipUnless(CUDA and TRITON, "NVIDIA CUDA and Triton required")
    def test_raw_byte_closing_sums_are_exact_above_float32_integer_range(self):
        from faiss.contrib.e_means.triton_assignment import closing_sums

        n = 2**18 + 13
        for dtype, value in ((torch.int8, 127), (torch.uint8, 255)):
            with self.subTest(dtype=dtype):
                rows = torch.full((n, 1), value, dtype=dtype, device="cuda")
                index = torch.arange(n, device="cuda")
                sums = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
                counts = torch.zeros(1, dtype=torch.float64, device="cuda")
                closing_sums(
                    rows,
                    index,
                    torch.tensor([0, n], device="cuda"),
                    sums,
                    counts,
                    65536,
                )
                torch.testing.assert_close(
                    sums,
                    torch.tensor([[n * value]], device="cuda"),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    counts,
                    torch.tensor(
                        [float(n)], dtype=torch.float64, device="cuda"
                    ),
                    rtol=0,
                    atol=0,
                )


class OffsetCodesCast(CastQuantizer):
    def compute_codes(self, rows):
        return super().compute_codes(rows) + .25


class InheritedCast(CastQuantizer):
    pass


@unittest.skipUnless(CUDA and TRITON and torch.cuda.get_device_capability() >= (8, 0),
                     "NVIDIA SM80+ and Triton required")
class TestEncodedClosingEvents(unittest.TestCase):
    def check_events(self, family, chunk=None):
        from unittest import mock
        from faiss.contrib.e_means import e_means_torch

        centers = torch.tensor([[0., 0.], [4., 0.], [8., 0.], [12., 0.],
                                [2., 0.], [20., 0.]], device="cuda")
        base = torch.tensor([[2., 0.], [0., 0.], [8., 0.], [4., 0.], [8., 0.],
                             [3., 0.], [8., 0.], [12., 0.], [2., 0.], [7., 0.]], device="cuda")
        order = torch.tensor([7, 2, 9, 0, 8, 1, 4, 3, 6], device="cuda")
        visited = base.clone()
        base[order], base[5] = visited[:9], visited[9]
        original = torch.full((10,), -1, dtype=torch.int64, device="cuda")
        original[order] = torch.tensor([0, 1, 2, 0, 1, 0, 1, 3, 4], device="cuda")
        moved = torch.tensor([-1, 3, 6, -1, 9, -1], device="cuda")
        events = [(0, 1, 3), (1, 2, 6), (0, 4, 5), (3, 4, 9)]
        q = None
        if family == "int8":
            base, centers = (base * 4).to(torch.int8), centers * 4
        elif family == "fp8":
            base = base.to(torch.float8_e4m3fn)
        elif family == "int8_cal":
            q = Int8Quantizer(2)
            q.train(torch.cat((base, centers)))
            base = q.compute_codes(base)
        else:
            cls = OffsetCodesCast if family == "override" else InheritedCast
            q = cls(2, torch.bfloat16, center=family == "center_bf16")
            if family == "center_bf16":
                base, centers = base + 100, centers + 100
            q.train(torch.cat((base, centers)))
        model = QuantizedEmeans(2, 6, quantizer=q, assignment_backend="triton",
                                nepochs=1, batch_size=8, assignment_chunk_size=8)
        model.begin(10, centers)
        model._prepare_input(base)
        codes = base if q is None or base.dtype == q.dtype else q.compute_codes(base)
        values = codes.float() if q is None else q.decode(codes)
        represented = centers.to(base.dtype) if q is None else q.compute_codes(centers)
        expected, searched = _scalar_labels(original.cpu(), order.cpu(), moved.cpu(), events,
                                            values.cpu(), codes.cpu(), centers.cpu(), represented.cpu())
        labels = original.clone()
        model._handover_start(labels, base, second=False)
        model._handover["moved_at"].copy_(moved)
        model._handover["events"] = [(torch.tensor([target], device="cuda"),
                                       torch.tensor([source], device="cuda"), None, birth)
                                      for source, target, birth in events]
        sums = torch.zeros_like(model._pass_sums)
        sums.index_add_(0, original[order], values[order].to(sums.dtype))
        model._pass_sums.copy_(sums)
        model._pass_counts.copy_(torch.bincount(original[order], minlength=6))
        with mock.patch.object(e_means_torch, "_HANDOVER_CHUNK", chunk or e_means_torch._HANDOVER_CHUNK):
            model._handover_finish(order, len(order))
        torch.testing.assert_close(labels.cpu(), expected, rtol=0, atol=0)
        self.assertEqual(model.closing_handover_stats, {"searched": searched})
        sums.zero_().index_add_(0, labels[order], values[order].to(sums.dtype))
        counts = torch.bincount(labels[order], minlength=6).to(model._pass_counts.dtype)
        self.assertEqual(model._pass_sums.dtype, torch.int64 if family == "int8" else torch.float32)
        torch.testing.assert_close(model._pass_sums, sums, rtol=0, atol=0)
        torch.testing.assert_close(model._pass_counts, counts, rtol=0, atol=0)
        final, populated = model.finalize(), counts > 0
        torch.testing.assert_close(final[populated],
                                   (sums[populated].double() / counts[populated, None]).float(),
                                   rtol=0, atol=0)
        torch.testing.assert_close(final[~populated], centers[~populated], rtol=0, atol=0)

    def test_scalar_search_twins_and_decoded_means(self):
        cap = torch.cuda.get_device_capability()
        for family in ("int8", "fp8", "int8_cal", "cast_bf16", "center_bf16", "override"):
            legal = (cap in ((8, 0), (9, 0), (10, 0)) if family in ("int8", "int8_cal")
                     else cap in ((9, 0), (10, 0), (10, 3)) if family == "fp8" else True)
            if not legal:
                print("ENCODED_CLOSING_UNAVAILABLE", family, cap, flush=True)
                continue  # Real rejection is covered by the compact admission table.
            with self.subTest(family=family):
                self.check_events(family)

    def test_direct_and_staged_chunk_boundary(self):
        # Nonzero begin crosses both the split birth and stale-row boundary.
        for family in ("cast_bf16", "center_bf16"):
            with self.subTest(family=family):
                self.check_events(family, chunk=4)
