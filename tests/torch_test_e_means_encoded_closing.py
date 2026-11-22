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


@unittest.skipUnless(
    CUDA and TRITON and torch.cuda.get_device_capability() >= (8, 0),
    "encoded closing requires NVIDIA SM80 or later and Triton",
)
class TestEncodedClosingEvents(unittest.TestCase):
    def test_scalar_search_twins_repeated_moves_boundaries_and_fractional_tail(
        self,
    ):
        centers = torch.tensor(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [8.0, 0.0],
                [12.0, 0.0],
                [2.0, 0.0],
                [20.0, 0.0],
            ],
            device="cuda",
        )
        base = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 0.0],
                [8.0, 0.0],
                [4.0, 0.0],
                [8.0, 0.0],
                [3.0, 0.0],
                [8.0, 0.0],
                [12.0, 0.0],
                [2.0, 0.0],
                [7.0, 0.0],
            ],
            device="cuda",
        )
        order = torch.tensor([7, 2, 9, 0, 8, 1, 4, 3, 6], device="cuda")
        visit_rows = base.clone()
        base[order] = visit_rows[:9]
        base[5] = visit_rows[9]
        original = torch.full((10,), -1, dtype=torch.int64, device="cuda")
        original[order] = torch.tensor(
            [0, 1, 2, 0, 1, 0, 1, 3, 4], device="cuda"
        )
        moved = torch.tensor([-1, 3, 6, -1, 9, -1], device="cuda")
        events = [(0, 1, 3), (1, 2, 6), (0, 4, 5), (3, 4, 9)]
        families = (
            "int8",
            "uint8",
            "fp8",
            "int8_cal",
            "fp8_plain",
            "fp8_center",
            "cast_fp16",
            "cast_bf16",
            "center_fp16",
            "center_bf16",
        )
        for family in families:
            rows, current, codec = base.clone(), centers.clone(), None
            if family in ("int8", "uint8"):
                dtype = torch.int8 if family == "int8" else torch.uint8
                offset = 128 if dtype == torch.uint8 else 0
                rows = (rows * 4 + offset).to(dtype)
                current = current * 4 + offset
            elif family == "fp8":
                rows = rows.to(torch.float8_e4m3fn)
            elif family == "int8_cal":
                codec = Int8Quantizer(2)
                codec.train(torch.cat((current, rows)))
                rows = codec.compute_codes(rows)
            else:
                if family.startswith("fp8"):
                    codec = Fp8Quantizer(2)
                else:
                    codec = CastQuantizer(
                        2,
                        (
                            torch.float16
                            if family.endswith("fp16")
                            else torch.bfloat16
                        ),
                        center=family.startswith("center"),
                    )
                if family == "fp8_center" or family.startswith("center"):
                    rows += 100
                    current += 100
                sample = (
                    torch.tensor(
                        [[-32.0, -16.0], [0.0, 0.0], [32.0, 16.0]],
                        device="cuda",
                    ).repeat(1024, 1)
                    if family == "fp8_plain"
                    else torch.cat((current, rows))
                )
                codec.train(sample)
            backends = ["triton"]
            if importlib.util.find_spec("cutlass") is not None:
                backends.append("cute")
            capability = torch.cuda.get_device_capability()
            for backend in backends:
                with self.subTest(family=family, backend=backend):
                    legal = ((capability == (9, 0) and not family.endswith('fp16'))
                             if backend == 'cute' else
                             capability in ((8, 0), (9, 0), (10, 0))
                             if family in ('int8', 'uint8', 'int8_cal') else
                             capability in ((9, 0), (10, 0), (10, 3))
                             if family.startswith('fp8') else capability >= (8, 0))
                    if not legal:
                        probe = QuantizedEmeans(2, 6, quantizer=codec,
                                                assignment_backend=backend,
                                                nepochs=1, batch_size=8)
                        if backend == 'cute':
                            error = ValueError if capability == (9, 0) else RuntimeError
                            message = 'CuTe operands' if capability == (9, 0) else 'exactly SM90'
                        elif family in ('int8', 'uint8', 'int8_cal'):
                            error, message = RuntimeError, 'SM80, SM90, or SM100'
                        else:
                            from triton.compiler.errors import CompilationError
                            error, message = CompilationError, 'fp8e4nv not supported'
                        with self.assertRaisesRegex(error, message):
                            probe.begin(10, current)
                            probe.accumulate(rows[:8])
                        print('ENCODED_CLOSING_UNAVAILABLE', family, backend, capability, flush=True)
                        continue
                    model = QuantizedEmeans(
                        2,
                        6,
                        quantizer=codec,
                        assignment_backend=backend,
                        nepochs=1,
                        batch_size=8,
                        assignment_chunk_size=8,
                    )
                    model.begin(10, current)
                    model._prepare_input(rows)
                    codes = (
                        rows
                        if codec is None or rows.dtype == codec.dtype
                        else codec.compute_codes(rows)
                    )
                    values = (
                        codes.float() if codec is None else codec.decode(codes)
                    )
                    represented = (
                        current.to(rows.dtype)
                        if codec is None
                        else codec.compute_codes(current)
                    )
                    if family in ("int8", "uint8"):
                        lo, hi = (
                            (-128, 127)
                            if rows.dtype == torch.int8
                            else (0, 255)
                        )
                        represented = (
                            current.round().clamp(lo, hi).to(rows.dtype)
                        )
                    expected, searched = _scalar_labels(
                        original.cpu(),
                        order.cpu(),
                        moved.cpu(),
                        events,
                        values.cpu(),
                        codes.cpu(),
                        current.cpu(),
                        represented.cpu(),
                    )
                    labels = original.clone()
                    model._handover_start(labels, rows, second=False)
                    model._handover["moved_at"].copy_(moved)
                    model._handover["events"] = [
                        (
                            torch.tensor([target], device="cuda"),
                            torch.tensor([source], device="cuda"),
                            None,
                            birth,
                        )
                        for source, target, birth in events
                    ]
                    sums = torch.zeros_like(model._pass_sums)
                    sums.index_add_(
                        0, original[order], values[order].to(sums.dtype)
                    )
                    model._pass_sums.copy_(sums)
                    model._pass_counts.copy_(
                        torch.bincount(original[order], minlength=6)
                    )
                    model._handover_finish(order, len(order))
                    torch.testing.assert_close(
                        labels.cpu(), expected, rtol=0, atol=0
                    )
                    self.assertEqual(
                        model.closing_handover_stats, {"searched": searched}
                    )
                    sums.zero_()
                    sums.index_add_(
                        0, labels[order], values[order].to(sums.dtype)
                    )
                    # Raw byte runs select INT64 tables; decoded/floating
                    # runs retain the baseline FP32 sums and FP64 counts.
                    raw_bytes = codec is None and family in ("int8", "uint8")
                    self.assertEqual(
                        model._pass_sums.dtype,
                        torch.int64 if raw_bytes else torch.float32,
                    )
                    counts = torch.bincount(labels[order], minlength=6).to(
                        torch.int64 if raw_bytes else torch.float64
                    )
                    torch.testing.assert_close(
                        model._pass_sums, sums, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        model._pass_counts, counts, rtol=0, atol=0
                    )
                    final = model.finalize()
                    populated = counts > 0
                    torch.testing.assert_close(
                        final[populated],
                        (
                            sums[populated].double() / counts[populated, None]
                        ).float(),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        final[~populated], current[~populated], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
