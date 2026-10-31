# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile
import importlib.util
import datetime
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import torch
import numpy as np

from faiss.contrib.e_means import Emeans, DistributedEmeans
from faiss.contrib.e_means.e_means import _epoch_seed
from faiss.contrib.e_means.e_means_quantizer import CastQuantizer
from faiss.contrib.e_means.e_means_fp8 import Fp8Quantizer
from faiss.contrib.e_means.e_means_int8 import Int8Quantizer
from faiss.contrib.e_means import e_means
from e_means_test_utils import EmeansReferenceContracts, highest_fp32, known_torch_centroids

TRITON_CUDA = (torch.cuda.is_available() and not torch.version.hip
               and torch.cuda.get_device_capability() >= (8, 0)
               and importlib.util.find_spec("triton") is not None)
FP8_CUDA = (TRITON_CUDA and torch.cuda.get_device_capability()
            in ((9, 0), (10, 0), (10, 3)))
INT8_CUDA = (TRITON_CUDA and torch.cuda.get_device_capability()
             in ((8, 0), (9, 0), (10, 0)))


def _cpu_gloo_replica(rank, rendezvous):
    import torch.distributed as dist
    from faiss.contrib.e_means.e_means_distributed import _combine

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    try:
        check = unittest.TestCase()
        for numpy_state in (True, False):
            def array(values, _numpy_state=numpy_state):
                return (
                    np.asarray(values, dtype="float32")
                    if _numpy_state
                    else torch.tensor(values, dtype=torch.float32)
                )

            def host(value, _numpy_state=numpy_state):
                return value if _numpy_state else value.detach().numpy()
            for transport in ("dense", "sparse", "auto"):
                initial = array([[0.], [2.], [100.], [200.]])
                caller_initial = initial + rank * 1000
                saved_initial = host(caller_initial).copy()
                rows = array(np.array([0., .25, 2., 2.25])[np.arange(10) % 4, None])
                args = dict(niter=2, batch_size=4, alpha0=0.5, seed=7,
                            capture_assignments=True)
                model = DistributedEmeans(1, 4, group=dist.group.WORLD, transport=transport, **args)
                reference = Emeans(1, 4, **args)
                with check.assertRaisesRegex(ValueError, "divisible"):
                    model.begin(9, caller_initial)
                with check.assertRaises(RuntimeError):
                    model.accumulate(rows[:1])
                with check.assertRaises(NotImplementedError):
                    model.train(rows)
                model.begin(len(rows), caller_initial)
                reference.begin(len(rows), initial)
                np.testing.assert_array_equal(host(model.current_centroids()), host(initial))
                np.testing.assert_array_equal(host(caller_initial), saved_initial)
                if rank:
                    model._split_rng.seed(123456) if numpy_state else model._split_rng.manual_seed(123456)
                for epoch in range(2):
                    for begin in range(0, len(rows), 4):
                        n = min(4, len(rows) - begin)
                        start = begin + rank * (n // 2)
                        local = rows[start:start + n // 2]
                        if epoch == begin == 0:
                            # External LOCAL statistics without calling accumulate:
                            # Initial labels and distances have exact dyadic truth.
                            labels = (host(local)[:, 0] >= 2).astype("int64")
                            sums = array(np.bincount(labels, weights=host(local)[:, 0], minlength=4)[:, None])
                            counts = array(np.bincount(labels, minlength=4))
                            objective = np.float64(((host(local)[:, 0] - 2 * labels) ** 2).sum())
                            if numpy_state:
                                sums = sums[::-1].copy()[::-1]  # Valid negative stride.
                                sums.flags.writeable = counts.flags.writeable = False
                            else:
                                objective = torch.tensor(objective, dtype=torch.float64)
                            stats = dict(sums=sums, counts=counts, objective=objective, n=len(local))
                        else:
                            before = host(model.current_centroids()).copy()
                            if rank == 0 and len(local) > 1:
                                partial = model.accumulate(local[:1])
                                assert partial["n"] == 1 and partial["sums"] is None
                                assert partial["counts"] is None
                                with check.assertRaisesRegex(ValueError, "complete the pending"):
                                    model.update({})
                                np.testing.assert_array_equal(host(model.current_centroids()), before)
                                stats = model.accumulate(local[1:])
                            else:
                                stats = model.accumulate(local)
                            if epoch == 1:
                                supplied = local[1:] if rank == 0 and len(local) > 1 else local
                                labels = ((host(supplied)[:, None, :] - before[None, :, :]) ** 2).sum(2).argmin(1)
                                np.testing.assert_array_equal(host(stats["labels"]), labels)
                            stats = {name: value.copy() if isinstance(value, np.ndarray)
                                     else value.clone() if isinstance(value, torch.Tensor) else value
                                     for name, value in stats.items()}
                        saved = {name: host(stats[name]).copy() for name in ("sums", "counts")}
                        with check.assertRaisesRegex(ValueError, "prescribed batch size"):
                            model.update(dict(stats, n=n))
                        model.update(stats)
                        assert model._epoch_rows == begin + n
                        for name, value in saved.items():
                            np.testing.assert_array_equal(host(stats[name]), value)
                        reference.update(reference.accumulate(rows[begin:begin + n]))
                        np.testing.assert_allclose(host(model.current_centroids()), host(reference.current_centroids()), rtol=1e-6, atol=1e-6)
                        np.testing.assert_array_equal(host(model._pass_counts), host(reference._pass_counts))
                        if model._pass_sums is not None:
                            np.testing.assert_allclose(host(model._pass_sums), host(reference._pass_sums), rtol=1e-6, atol=1e-6)
                    model.end_epoch()
                    reference.end_epoch()
                    np.testing.assert_allclose(float(model._objective), float(reference._objective), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(host(model.finalize()), host(reference.finalize()), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(host(model.obj), host(reference.obj), rtol=1e-6, atol=1e-6)
                assert model.assignments is None  # Caller chunks own labels, not a dataset result.
                assert model.n_splits == reference.n_splits and model.n_splits >= 2
                final = model.centroids
                saved_final = host(final).copy()
                model.reset()
                np.testing.assert_array_equal(host(final), saved_final)
            # Fail after real collective execution, then recover only via begin.
            class CollectiveFailure(BaseException):
                pass
            model = DistributedEmeans(1, 1, group=dist.group.WORLD,
                                      transport="dense", niter=1,
                                      batch_size=2, alpha0=0.5)
            initial = array([[0.]])
            broadcast = dist.broadcast
            def failed_broadcast(*args, _broadcast=broadcast, **kwargs):
                _broadcast(*args, **kwargs)
                raise CollectiveFailure()
            with mock.patch.object(dist, "broadcast", side_effect=failed_broadcast):
                with check.assertRaises(CollectiveFailure):
                    model.begin(2, initial)
            with check.assertRaises(RuntimeError):
                model.current_centroids()
            model.begin(2, initial)
            stats = model.accumulate(array([[.25]]))
            with check.assertRaisesRegex(ValueError, "prescribed batch size"):
                model.update(dict(stats, n=2))
            np.testing.assert_array_equal(host(model.current_centroids()), host(initial))
            def failed_combine(*args, **kwargs):
                _combine(*args, **kwargs)
                raise CollectiveFailure()
            with mock.patch("faiss.contrib.e_means.e_means_distributed._combine", side_effect=failed_combine):
                with check.assertRaises(CollectiveFailure):
                    model.update(stats)
            for call in (
                lambda _model=model: _model.accumulate(array([[.25]])),
                lambda _model=model, _stats=stats: _model.update(_stats),
                model.end_epoch,
                model.finalize,
                model.current_centroids,
            ):
                with check.assertRaises(RuntimeError):
                    call()
            model.begin(2, initial)
            model.update(model.accumulate(array([[.25]])))
            model.end_epoch()
            np.testing.assert_array_equal(host(model.finalize()), [[.25]])
            if numpy_state:
                # A reversed singleton can report C-contiguous yet retain a
                # negative stride. Exercise a real one-donor broadcast too.
                args = dict(niter=1, batch_size=2, alpha0=0.5, seed=7)
                model = DistributedEmeans(1, 2, group=dist.group.WORLD,
                                          transport="dense", **args)
                reference = Emeans(1, 2, **args)
                initial = array([[0.], [100.]])
                model.begin(10, initial + rank * 1000)
                reference.begin(10, initial)
                for _ in range(5):
                    model.update(model.accumulate(array([[.25]])))
                    reference.update(reference.accumulate(array([[.25], [.25]])))
                assert model.n_splits == reference.n_splits == 1
                model.end_epoch()
                reference.end_epoch()
                np.testing.assert_allclose(model.finalize(), reference.finalize(), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(model.obj, reference.obj, rtol=1e-6, atol=1e-6)
            # Packing edge cases independent of row-count validation: a rank
            # with no occupied IDs, and then all ranks with no occupied IDs.
            for all_empty in (False, True):
                counts = array([0., 0.] if rank or all_empty else [2., 0.])
                sums = array([[0.], [0.]])
                workspace = np.empty((2, 2), dtype="float32") if numpy_state else torch.empty((2, 2))
                ids, combined_sums, combined_counts = _combine(sums, counts, workspace, dist.group.WORLD, "sparse", 4)
                assert isinstance(ids, np.ndarray if numpy_state else torch.Tensor)
                assert isinstance(combined_sums, np.ndarray if numpy_state else torch.Tensor)
                np.testing.assert_array_equal(host(ids), [] if all_empty else [0])
                np.testing.assert_array_equal(host(combined_counts), [] if all_empty else [2.])
                assert combined_sums.shape == (0 if all_empty else 1, 1)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestTorchEmeans(EmeansReferenceContracts, unittest.TestCase):
    def test_exact_distance_rejects_floating_and_calibrated_inputs(self):
        output = torch.full((1,), -123, dtype=torch.int64)
        for quantizer, dtype in ((None, torch.float32), (Int8Quantizer(2), torch.int8)):
            model = Emeans(2, 1, quantizer=quantizer, assignment_backend="triton")
            model._input_dtype = dtype
            with self.assertRaisesRegex(ValueError, "quantizer=None"):
                model._assign_rows(torch.zeros((1, 2), dtype=dtype), torch.zeros((1, 2)),
                                   distance_out=output)
            self.assertEqual(output.item(), -123)

    @unittest.skipUnless(TRITON_CUDA and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0)),
                         "qualified INT8 CUDA required")
    def test_native_byte_exact_distance_output(self):
        from faiss.contrib.e_means.e_means_triton import prepare_int8_centroids, assign_int8_rows
        for dtype, offset, representation in ((torch.int8, 0, "int8"), (torch.uint8, 128, "uint8")):
            rows = torch.full((1, 8192), offset, dtype=dtype, device="cuda")
            c = torch.full((1, 8192), float(offset), device="cuda")
            c[:, :1024] -= 128
            c[:, 1024] += 1
            prepared = prepare_int8_centroids(c, representation, validate=False)
            normal, labels = assign_int8_rows(rows, *prepared, representation)
            output = torch.full((3,), -123, dtype=torch.int64, device="cuda")
            exact, exact_labels = assign_int8_rows(rows, *prepared, representation, output[1:-1])
            self.assertEqual(normal.dtype, torch.float32)
            self.assertEqual((normal.item(), exact.item()), (2**24, 2**24 + 1))
            self.assertEqual(exact.data_ptr(), output[1:-1].data_ptr())
            self.assertEqual(output[[0, -1]].tolist(), [-123, -123])
            torch.testing.assert_close(labels, exact_labels)
            saved = exact.clone()
            assign_int8_rows(rows, *prepared, representation)
            torch.testing.assert_close(exact, saved, rtol=0, atol=0)
            invalid = torch.full((1,), -123, dtype=torch.float16, device="cuda")
            with self.assertRaises(ValueError):
                assign_int8_rows(rows, *prepared, representation, invalid)
            self.assertEqual(invalid.item(), -123)

    @unittest.skipUnless(torch.distributed.is_available() and
                         torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_cpu_gloo_caller_owned_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(
                _cpu_gloo_replica, args=("file://" + directory + "/group",),
                nprocs=2, join=True, start_method="spawn"
            )

    def test_explicit_int8_quantizer_and_scope(self):
        x = torch.tensor([[10., 100.], [12., 104.], [14., 108.], [16., 112.]])
        q = Int8Quantizer(2)
        with self.assertRaises(RuntimeError):
            q.compute_codes(x)
        q.train(x)
        self.assertEqual((q.code_size, q.scale), (2, 16.))
        self.assertIs(q.to("cpu"), q)
        expected = torch.tensor([[-48, -96], [-16, -32], [16, 32], [48, 96]], dtype=torch.int8)
        torch.testing.assert_close(q.compute_codes(x), expected)
        torch.testing.assert_close(q.decode(expected), x)
        torch.testing.assert_close(q.compute_codes(torch.tensor([[1e6, -1e6]])),
                                   torch.tensor([[127, -128]], dtype=torch.int8))
        q.validate(x)
        with self.assertRaises(ValueError):
            q.validate(torch.full((1, 2), float("nan")))
        self.assertEqual(q.compute_codes(x[:0]).shape, (0, 2))
        with self.assertRaisesRegex(ValueError, "requires Triton"):
            Emeans(2, 2, quantizer=q, assignment_backend="native")
        with self.assertRaisesRegex(TypeError, "no JIT INT8"):
            Emeans(2, 2, quantizer=q, assignment_backend="triton").train(x)
        with self.assertRaisesRegex(ValueError, "d <= 43919"):
            Emeans(43920, 2, quantizer=Int8Quantizer(43920), assignment_backend="triton")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q.train(x.to(dtype))
                torch.testing.assert_close(q.shift, torch.tensor([13., 106.]), rtol=0, atol=0)
                torch.testing.assert_close(q.compute_codes(x.to(dtype)), expected, rtol=0, atol=0)
                torch.testing.assert_close(q.decode(expected), x, rtol=0, atol=0)
                q.validate(torch.tensor([[1.e6, -1.e6]]))  # Finite outliers clip.
                self.assertEqual(q.decode(expected[:0]).shape, (0, 2))
                for method in (q.compute_codes, q.decode):
                    with self.assertRaises(ValueError):
                        method(torch.zeros(1, 3))
        q.train(torch.zeros(2, 2))
        self.assertEqual((q.scale, q.exponent), (1., 0))
        for bad in (torch.full((1, 2), float('nan')), torch.full((1, 2), float('inf'))):
            with self.assertRaises(ValueError):
                q.train(bad)
            self.assertFalse(q.is_trained)
            with self.assertRaises(RuntimeError):
                q.decode(expected)

    def test_native_bytes_explicit_scope(self):
        for dtype in (torch.int8, torch.uint8):
            with self.assertRaises(ValueError):
                Emeans(1, 1).train(torch.zeros(2, 1, dtype=dtype))
        if TRITON_CUDA:
            for capability in ((7, 5), (8, 9), (10, 3)):
                with mock.patch.object(torch.cuda, 'get_device_capability', return_value=capability):
                    with self.assertRaisesRegex(RuntimeError, 'SM80|SM90|SM100'):
                        Emeans(1, 1, assignment_backend='triton').train(torch.zeros(2, 1, dtype=torch.int8, device='cuda'))
            with mock.patch.object(torch.version, 'hip', 'rejection-only probe'):
                with self.assertRaises(RuntimeError):
                    Emeans(1, 1, assignment_backend='triton').train(torch.zeros(2, 1, dtype=torch.int8, device='cuda'))

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0))
        and importlib.util.find_spec("triton") is not None, "qualified INT8 CUDA required"
    )
    def test_native_int8_uint8_and_decoded_training(self):
        base = torch.tensor([[-4.], [-2.], [4.], [6.]], device="cuda")
        q = Int8Quantizer(1)
        q.train(base)
        for dtype, offset, quantizer in ((torch.int8, 0, None),
                                         (torch.uint8, 128, None), (torch.int8, 0, q)):
            with self.subTest(dtype=dtype, calibrated=quantizer is not None):
                codes = (base + offset).to(dtype) if quantizer is None else q.compute_codes(base)
                initial = base[[0, 2]] + offset
                km = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, quantizer=quantizer,
                            assignment_backend="triton", assignment_chunk_size=1)
                centroids, captured_labels = km.train(codes, initial)
                self.assertEqual(km.obj[-1], 8.)
                self.assertIs(centroids, km.centroids)
                self.assertIs(captured_labels, km.assignments)
                torch.testing.assert_close(
                    km.centroids,
                    torch.tensor([[-3. + offset], [5. + offset]], device="cuda"),
                )
                distance, labels = km.assign(codes)
                torch.testing.assert_close(distance, torch.ones(4, device="cuda"))
                torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], device="cuda"))
                self.assertEqual((km.centroids.dtype, distance.dtype, labels.dtype),
                                 (torch.float32, torch.float32, torch.int64))
                self.assertEqual(km.recipe["representation"], str(dtype).removeprefix("torch."))
        with self.assertRaisesRegex(TypeError, "no JIT INT8"):
            Emeans(1, 2, quantizer=q, assignment_backend="triton").train(base)

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0))
        and importlib.util.find_spec("triton") is not None, "qualified INT8 CUDA required"
    )
    def test_raw_integer_statistics_dtype_and_totals(self):
        for dtype, value, cap in ((torch.int8, 127, 131072), (torch.uint8, 255, 65793)):
            for n in (0, 7, cap, cap + 1, 262145):
                with self.subTest(dtype=dtype, n=n):
                    model = Emeans(1, 1, niter=1, batch_size=max(n, 1), alpha0=.5,
                                   assignment_backend='triton', assignment_chunk_size=70001)
                    model.begin(max(n, 1), torch.tensor([[float(value)]], device='cuda'))
                    rows = torch.full((n, 1), value, dtype=dtype, device='cuda')
                    stats = model.accumulate(rows)
                    self.assertEqual((stats['sums'].dtype, stats['counts'].dtype),
                                     (torch.int64, torch.int64))
                    self.assertEqual(stats['objective'].dtype, torch.float64)
                    self.assertEqual((stats['sums'].item(), stats['counts'].item(), stats['n']),
                                     (value * n, n, n))
                    model.reset()
            # Same run: full batch followed by one-row kept tail stays INT64.
            model = Emeans(1, 1, niter=1, batch_size=cap + 1, alpha0=.5,
                           assignment_backend='triton', assignment_chunk_size=70001)
            model.begin(cap + 2, torch.tensor([[float(value)]], device='cuda'))
            for n in (cap + 1, 1):
                stats = model.accumulate(torch.full((n, 1), value, dtype=dtype, device='cuda'))
                self.assertEqual((stats['sums'].dtype, stats['counts'].dtype), (torch.int64, torch.int64))
                self.assertEqual((stats['sums'].item(), stats['counts'].item()), (value * n, n))
                model.update(stats)
            model.end_epoch()
            self.assertEqual(model.finalize().item(), float(value))

        previous, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            for deterministic in (False, True):
                torch.use_deterministic_algorithms(deterministic)
                rows = torch.empty(33, 6, dtype=torch.int8, device='cuda')[:, ::2]
                rows[::2] = -128
                rows[1::2] = 127
                model = Emeans(3, 1, niter=1, batch_size=33, alpha0=.5,
                               assignment_backend='triton', assignment_chunk_size=7)
                model.begin(33, torch.zeros(1, 3, device='cuda'))
                stats = model.accumulate(rows)
                truth = rows.cpu().long()
                torch.testing.assert_close(stats['sums'].cpu(), truth.sum(0, keepdim=True), rtol=0, atol=0)
                self.assertEqual(stats['counts'].item(), 33)
                self.assertEqual(stats['objective'].item(), truth.square().sum().item())
                model.update(stats)
                model.end_epoch()
                model.finalize()
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn)

    @unittest.skipUnless(TRITON_CUDA, 'qualified Triton CUDA required')
    def test_raw_integer_full_population_accumulation_boundaries(self):
        # FP16 is supported on floating targets even when byte assignment is not.
        cases = [(torch.float16, 1, 2**24 + 2)]
        if INT8_CUDA:
            cases += [(torch.int8, -128, 2**24), (torch.uint8, 0, 2**24),
                      (torch.int8, -128, 2**24 + 1),
                      (torch.uint8, 255, 2**31 // 255),
                      (torch.uint8, 255, 2**31 // 255 + 1)]
        for dtype, value, n in cases:
            with self.subTest(dtype=dtype, n=n):
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                rows = torch.full((n, 1), value, dtype=dtype, device='cuda')
                model = Emeans(1, 1, niter=1, batch_size=n, alpha0=.5,
                               assignment_backend='triton', assignment_chunk_size=65536)
                model.begin(n, torch.tensor([[float(value)]], device='cuda'))
                # Both halves fit the small-call bound. Scratch/shift selection
                # must still protect the whole prescribed population.
                partial = model.accumulate(rows[:n // 2])
                self.assertEqual(partial['n'], n // 2)
                self.assertIsNone(partial['sums'])
                self.assertIsNone(partial['counts'])
                stats = model.accumulate(rows[n // 2:])
                expected_dtype = torch.float32 if dtype == torch.float16 else torch.int64
                self.assertEqual((stats['sums'].dtype, stats['counts'].dtype), (expected_dtype, expected_dtype))
                self.assertEqual((stats['sums'].item(), stats['counts'].item()), (value*n, n))
                self.assertEqual(stats['objective'].item(), 0.)
                empty = model.accumulate(rows[:0])
                self.assertEqual((empty['sums'].item(), empty['counts'].item(), empty['n']), (value*n, n, n))
                print('raw boundary correctness cost', str(dtype), n,
                      time.monotonic()-start, 'seconds, peak', torch.cuda.max_memory_allocated(), 'bytes')
                model.reset()
                del rows, stats, model

    @unittest.skipUnless(INT8_CUDA, 'qualified INT8 CUDA required')
    def test_raw_integer_nonpower_two_closing_division(self):
        n = 1048577
        for total in (266199830, 132023843, -134153956):
            with self.subTest(total=total):
                model = Emeans(1, 1, niter=1, batch_size=n, alpha0=.5, assignment_backend='triton')
                model.begin(n, torch.zeros(1, 1, device='cuda'))
                stats = dict(sums=torch.tensor([[total]], dtype=torch.int64, device='cuda'),
                             counts=torch.tensor([n], dtype=torch.int64, device='cuda'),
                             objective=torch.zeros((), dtype=torch.float64, device='cuda'), n=n)
                model.update(stats)
                model.end_epoch()
                expected = np.float32(np.float64(total) / np.float64(n))
                self.assertNotEqual(expected, np.float32(np.float32(total) / np.float32(n)))
                self.assertNotEqual(expected, np.float32(np.float64(np.float32(total)) / np.float64(n)))
                self.assertEqual(model.finalize().item(), float(expected))

    @unittest.skipUnless(INT8_CUDA, 'qualified INT8 CUDA required')
    def test_native_bytes_integer_ranking_clipping_and_large_distance(self):
        for dtype, offset in ((torch.int8, 0), (torch.uint8, 128)):
            for n, k, d in ((2, 3, 17), (3, 257, 32), (65, 513, 129), (2, 3, 257), (1, 2, 43919)):
                with self.subTest(dtype=dtype, n=n, k=k, d=d):
                    g = torch.Generator(device='cuda').manual_seed(n+k+d)
                    signed = torch.randint(-128, 128, (n, d), device='cuda', dtype=torch.int8, generator=g)
                    c = torch.randint(-128, 128, (k, d), device='cuda', generator=g).float()
                    if d == 43919:
                        signed.fill_(127)
                        c.fill_(-128)
                    rows = (signed.int()+offset).to(dtype)
                    model = self._known_centroids(c+offset, assignment_backend='triton')
                    distance, labels = model.assign(rows)
                    truth = (rows.cpu().long()[:, None] - (c+offset).cpu().long()[None]).square().sum(2)
                    torch.testing.assert_close(labels.cpu(), truth.argmin(1), rtol=0, atol=0)
                    torch.testing.assert_close(distance.cpu(), truth.min(1).values.float(), rtol=0, atol=0)
                    if d == 43919:
                        self.assertGreater(truth[0, 0].item(), 2**31)
                    model.centroids.fill_(offset)
                    tied_distance, tied_labels = model.assign(rows)
                    self.assertTrue(bool((tied_labels == 0).all()))
                    torch.testing.assert_close(
                        tied_distance.cpu(),
                        (rows.cpu().long() - offset).square().sum(1).float(),
                        rtol=0,
                        atol=0,
                    )
            # Begin with actual raw rows, then mutate the exposed closing table
            # for a finite rounding/clipping query, not impossible raw totals.
            model = Emeans(257, 1, niter=1, batch_size=2, alpha0=.5, assignment_backend='triton')
            model.train(torch.full((2, 257), offset, device='cuda', dtype=dtype))
            model.centroids[0, :6] = torch.tensor([-1.5, -.5, .5, 1.5, 200., -200.], device='cuda')+offset
            wanted = torch.zeros(1, 257, device='cuda', dtype=torch.int32)
            wanted[0, :6] = torch.tensor([-2, 0, 0, 2, 127, -128], device='cuda')
            distance, labels = model.assign((wanted+offset).to(dtype))
            self.assertEqual((distance.item(), labels.item()), (0., 0))

    @unittest.skipUnless(INT8_CUDA, 'qualified INT8 CUDA required')
    def test_native_bytes_typed_tail_lifecycle_and_seed_ownership(self):
        x = torch.tensor([[.1, -.2], [2.2, .3], [10.3, .4], [12.4, .5], [11.8, .25]], device='cuda')
        q = Int8Quantizer(2)
        q.train(x)
        for dtype, offset, selected_q in ((torch.int8, 0, None), (torch.uint8, 128, None), (torch.int8, 0, q)):
            codes = q.compute_codes(x) if selected_q else (x.round()+offset).to(dtype)
            initial = x[[0, 2]] if selected_q else codes[[0, 2]].float()
            args = dict(niter=2, batch_size=2, alpha0=.5, seed=7, quantizer=selected_q,
                        assignment_backend='triton', assignment_chunk_size=1, capture_assignments=True)
            matrix, caller = Emeans(2, 2, **args), Emeans(2, 2, **args)
            saved = codes.clone()
            matrix.train(codes, initial)
            caller.begin(len(codes), initial)
            for epoch in range(2):
                order = e_means._randperm(len(codes), _epoch_seed(7, epoch), True, x.device)
                for start in range(0, len(codes), 2):
                    batch = codes[order[start:start+2]]
                    before = caller.current_centroids().clone()
                    c = (q.compute_codes(before) if selected_q else
                         before.round().clamp(offset-128, offset+127).to(dtype))
                    oracle = (batch.cpu().long()[:, None]-c.cpu().long()[None]).square().sum(2)
                    labels = oracle.argmin(1).to(x.device)
                    distances = oracle.min(1).values.float().to(x.device)
                    if selected_q:
                        distances = torch.ldexp(distances, q._distance_exponent)
                    for position in range(len(batch)):
                        stats = caller.accumulate(batch[position:position + 1])
                        torch.testing.assert_close(caller.current_centroids(), before, rtol=0, atol=0)
                        if position + 1 < len(batch):
                            self.assertIsNone(stats['sums'])
                            self.assertIsNone(stats['counts'])
                        if epoch == caller.niter - 1:
                            torch.testing.assert_close(stats['labels'], labels[position:position + 1], rtol=0, atol=0)
                    expected_dtype = torch.float32 if selected_q else torch.int64
                    self.assertEqual((stats['sums'].dtype, stats['counts'].dtype), (expected_dtype, expected_dtype))
                    sums = torch.zeros_like(stats['sums'])
                    sums.index_add_(0, labels, q.decode(batch) if selected_q else batch.long())
                    torch.testing.assert_close(stats['sums'], sums, rtol=0, atol=0)
                    torch.testing.assert_close(stats['counts'], torch.bincount(labels, minlength=2).to(expected_dtype), rtol=0, atol=0)
                    torch.testing.assert_close(stats['objective'], distances.double().sum(), rtol=0, atol=0)
                    caller.update(stats)
                caller.end_epoch()
            caller.finalize()
            torch.testing.assert_close(caller.centroids, matrix.centroids, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(caller.obj, matrix.obj, rtol=0, atol=0)
            indices = torch.randperm(len(codes), device=x.device, generator=torch.Generator(device=x.device).manual_seed(7))[:2]
            initial = q.decode(codes[indices]) if selected_q else codes[indices].float()
            default, explicit = Emeans(2, 2, **args), Emeans(2, 2, **args)
            default.train(codes)
            explicit.train(codes, initial)
            torch.testing.assert_close(default.centroids, explicit.centroids, rtol=0, atol=0)
            torch.testing.assert_close(default.obj, explicit.obj, rtol=0, atol=0)
            torch.testing.assert_close(codes, saved, rtol=0, atol=0)
            output, before = default.centroids, default.centroids.clone()
            default.reset()
            torch.testing.assert_close(output, before, rtol=0, atol=0)

    @unittest.skipUnless(INT8_CUDA, 'qualified INT8 CUDA required')
    def test_native_bytes_scaling_revival_and_finite_failure(self):
        from faiss.contrib.e_means.e_means_int8 import separate_splits
        q = Int8Quantizer(2)
        q.train(torch.tensor([[8., 8.], [12., 12.]], device='cuda'))
        base = torch.tensor([[0., 0.], [3., 2.], [-128., 1.], [127., 1.]], device='cuda')
        for offset, selected_q in ((0, None), (128, None), (0, q)):
            representation = 'uint8' if offset else 'int8'
            donor = q.decode(base) if selected_q else base+offset
            child = donor.clone()
            untouched = child[:, 1:].clone()
            separate_splits(donor, child, representation, selected_q)
            encoded = q.compute_codes(child) if selected_q else (child-offset).round().to(torch.int8)
            torch.testing.assert_close(encoded[:, 0], torch.tensor([1, 4, -127, 126], dtype=torch.int8, device='cuda'), rtol=0, atol=0)
            torch.testing.assert_close(child[:, 1:], untouched, rtol=0, atol=0)
            ordinary = donor + (8/q.scale if selected_q else 8)
            saved = ordinary.clone()
            separate_splits(donor, ordinary, representation, selected_q)
            torch.testing.assert_close(ordinary, saved, rtol=0, atol=0)
            initial = torch.tensor([[8., 8.], [20., 20.]], device='cuda')+offset
            model = Emeans(2, 2, niter=1, batch_size=10, alpha0=.5, seed=7,
                           assignment_backend='triton', quantizer=selected_q)
            model.begin(10, initial)
            dtype = torch.int8 if selected_q or not offset else torch.uint8
            # Isolate split geometry without pinning update-check cadence. The
            # empty public contribution binds the representation and stays pending
            # until a legitimate full external contribution consumes it below.
            empty_stats = model.accumulate(torch.empty(0, 2, dtype=dtype, device='cuda'))
            self.assertEqual(empty_stats['n'], 0)
            model._usage.copy_(torch.tensor([10., 0.], device='cuda'))
            origin, mass = model._centroids[0].clone(), model._usage[0].clone()
            model._apply_splits(torch.tensor([0], device='cuda'), torch.tensor([1], device='cuda'))
            self.assertEqual(model.n_splits, 1)
            epsilon = model.recipe['split_epsilon']
            offset_vector = torch.tensor([epsilon, -epsilon], device='cuda')
            torch.testing.assert_close(model._centroids[0], origin*(1-offset_vector), rtol=0, atol=0)
            torch.testing.assert_close(model._centroids[1, 1:], (origin*(1+offset_vector))[1:], rtol=0, atol=0)
            torch.testing.assert_close(model._usage, (mass/2).expand(2), rtol=0, atol=0)
            represented = (q.compute_codes(model._centroids) if selected_q else model._centroids.round())
            self.assertNotEqual(represented[0, 0].item(), represented[1, 0].item())
            stats_dtype = torch.float32 if selected_q else torch.int64
            sums = torch.zeros(2, 2, dtype=stats_dtype, device='cuda')
            sums[0] = initial[0].to(stats_dtype)*10  # Ten represented copies at label0.
            model.update(dict(sums=sums, counts=torch.tensor([10, 0], dtype=stats_dtype, device='cuda'),
                              objective=torch.zeros((), dtype=torch.float64, device='cuda'), n=10))
            model.end_epoch()
            empty = model._centroids[1].clone()
            final = model.finalize()
            torch.testing.assert_close(final[1], empty, rtol=0, atol=0)
        for exponent, encoded in ((75, 2), (-100, 0)):
            q = Int8Quantizer(1)
            peak = 2. ** (6-exponent)
            q.train(torch.tensor([[-peak], [peak]], device='cuda'))
            self.assertEqual(q.exponent, exponent)
            model = Emeans(1, 1, niter=1, batch_size=1, alpha0=.5, quantizer=q, assignment_backend='triton')
            model.begin(1, torch.zeros(1, 1, device='cuda'))
            codes = torch.tensor([[encoded]], dtype=torch.int8, device='cuda')
            stats = model.accumulate(codes)
            self.assertEqual(stats['objective'].item(), 2.**-148 if exponent == 75 else 0.)
            model.update(stats)
            model.end_epoch()
            model.finalize()
            self.assertEqual(model.assign(codes)[0].item(), 0.)
        maximum = torch.finfo(torch.float32).max
        q = Int8Quantizer(1)
        q.train(torch.tensor([[-maximum], [maximum]], device='cuda'))
        self.assertEqual(q.exponent, -122)
        donor = torch.tensor([[.99*maximum]], device='cuda')*(1-1.e-4)
        child = torch.tensor([[.99*maximum]], device='cuda')*(1+1.e-4)
        saved = child.clone()
        torch.testing.assert_close(q.compute_codes(donor), q.compute_codes(child), rtol=0, atol=0)
        separate_splits(donor, child, 'int8', q)
        torch.testing.assert_close(child, saved, rtol=0, atol=0)
        # A generated split beyond FP32 must fail before either centroid or
        # mass is written. Public training cannot isolate this finite limit;
        # this minimal existing split primitive probe makes no cache/API claim.
        model = Emeans(1, 2, niter=1, batch_size=2, alpha0=.5, assignment_backend='triton')
        model.begin(2, torch.full((2, 1), maximum, device='cuda'))
        model.accumulate(torch.empty(0, 1, dtype=torch.int8, device='cuda'))
        saved_centroids, saved_usage = model._centroids.clone(), model._usage.clone()
        with self.assertRaisesRegex(ValueError, 'finite'):
            model._apply_splits(torch.tensor([0], device='cuda'), torch.tensor([1], device='cuda'))
        torch.testing.assert_close(model._centroids, saved_centroids, rtol=0, atol=0)
        torch.testing.assert_close(model._usage, saved_usage, rtol=0, atol=0)
        model.reset()

    @unittest.skipUnless(INT8_CUDA, 'qualified INT8 CUDA required')
    def test_native_bytes_current_stream_decode_and_schema(self):
        current = torch.cuda.current_device()
        # One-GPU execution still checks caller streams, but only >=2 GPUs
        # exercise a genuinely noncurrent tensor device.
        device = torch.device('cuda', 1 if torch.cuda.device_count() >= 2 and current == 0 else 0)
        q = Int8Quantizer(2)
        q.train(torch.tensor([[-1., -1.], [1., 1.]]))
        q.to(device)
        untrained = Emeans(2, 1, quantizer=Int8Quantizer(2), assignment_backend='triton')
        with self.assertRaises(RuntimeError):
            untrained.begin(2, torch.zeros(1, 2, device=device))
        on_cpu = Int8Quantizer(2)
        on_cpu.train(torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            Emeans(2, 1, quantizer=on_cpu, assignment_backend='triton').begin(2, torch.zeros(1, 2, device=device))
        for selected_q in (None, q):
            dtype = torch.int8 if selected_q else torch.uint8
            known = torch.tensor([[0, 0], [64, 64]], dtype=dtype, device=device) if selected_q else torch.tensor([[128, 128], [129, 129]], dtype=dtype, device=device)
            initial = q.decode(known) if selected_q else known.float()
            model = Emeans(2, 2, niter=1, batch_size=2, alpha0=.5, quantizer=selected_q, assignment_backend='triton')
            model.train(known, initial)
            model.assign(known)
            torch.cuda.synchronize(device)
            rows = torch.empty(2, 4, dtype=dtype, device=device)[:, ::2]
            producer, consumer = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
            with torch.cuda.stream(producer):
                torch.cuda._sleep(50_000_000)
                rows.copy_(known)
            consumer.wait_stream(producer)
            with torch.cuda.stream(consumer):
                distance, labels = model.assign(rows)
                decoded = q.decode(rows) if selected_q else rows.float()
            torch.cuda.current_stream(device).wait_stream(consumer)
            torch.testing.assert_close(distance, torch.zeros(2, device=device), rtol=0, atol=0)
            torch.testing.assert_close(labels, torch.tensor([0, 1], device=device), rtol=0, atol=0)
            torch.testing.assert_close(decoded, initial, rtol=0, atol=0)
            torch.testing.assert_close(rows, known, rtol=0, atol=0)
            model.begin(2, initial)
            previous = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode('error')
                model.update(model.accumulate(known))
                if selected_q:
                    q.decode(known)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            model.end_epoch()
            model.finalize()
            self.assertEqual(torch.cuda.current_device(), current)
            if selected_q:
                for bad in (known.float(), known.half(), known.to(torch.uint8)):
                    with self.assertRaises(TypeError):
                        model.assign(bad)
                model.centroids[0, 0] = float('nan')
                with self.assertRaisesRegex(ValueError, 'finite'):
                    model.assign(known)
        constant = Int8Quantizer(1)
        constant.train(torch.full((2, 1), 255., device=device))
        model = Emeans(1, 1, niter=1, batch_size=2, alpha0=.5,
                       quantizer=constant, assignment_backend='triton')
        model.begin(2, torch.full((1, 1), 255., device=device))
        stats = model.accumulate(constant.compute_codes(torch.full((2, 1), 255., device=device)))
        self.assertEqual((stats['sums'].dtype, stats['counts'].dtype), (torch.float32, torch.float32))
        self.assertEqual((stats['sums'].item(), stats['counts'].item()), (510., 2.))
        model.update(stats)
        model.end_epoch()
        self.assertEqual(model.finalize().item(), 255.)
        extreme = Int8Quantizer(1)
        extreme.train(torch.tensor([[-2.**-69], [2.**-69]]))
        extreme.to(device)
        model = self._known_centroids(torch.zeros(1, 1, device=device), quantizer=extreme, assignment_backend='triton')
        distance, _ = model.assign(torch.tensor([[2]], dtype=torch.int8, device=device))
        self.assertEqual((distance.item(), distance.device, torch.cuda.current_device()), (2.**-148, device, current))

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0))
        and importlib.util.find_spec("triton") is not None, "qualified INT8 CUDA required"
    )
    def test_raw_integer_closing_partition_and_assign(self):
        for dtype, value in ((torch.int8, 127), (torch.uint8, 255)):
            for batch in (65795, 131073):
                with self.subTest(dtype=dtype, batch=batch):
                    rows = torch.full((131603, 1), value, dtype=dtype, device='cuda')
                    model = Emeans(1, 1, niter=2, batch_size=batch, alpha0=.5,
                                   assignment_backend='triton', assignment_chunk_size=70001)
                    model.train(rows, torch.tensor([[float(value)]], device='cuda'))
                    self.assertEqual(model.centroids.dtype, torch.float32)
                    self.assertEqual(model.centroids.item(), float(value))
                    self.assertIs(model.finalize(), model.centroids)
        # Default matrix training and caller batches share exact raw closing.
        for backend in ('triton',):
            for dtype in (torch.int8, torch.uint8):
                rows = (torch.arange(2049, device='cuda') % 256).reshape(-1, 1)
                rows = (rows - 128 if dtype == torch.int8 else rows).to(dtype)
                saved = rows.clone()
                expected = float(np.float32(np.float64(rows.cpu().long().sum().item()) / len(rows)))
                for niter in (1, 2, 3):
                    with self.subTest(raw_closing=dtype, niter=niter, backend=backend):
                        initial = torch.zeros(1, 1, device='cuda')
                        model = Emeans(1, 1, niter=niter, seed=7, assignment_backend=backend, capture_assignments=True)
                        model.train(rows, initial)
                        torch.testing.assert_close(model.assignments, torch.zeros(len(rows), dtype=torch.int64, device='cuda'), rtol=0, atol=0)
                        self.assertEqual(model.centroids.item(), expected)
                        output = model.centroids
                        self.assertIs(model.finalize(), output)
                        model.reset()
                        model.begin(len(rows), initial)
                        self.assertEqual(output.item(), expected)
                        model.reset()
                        caller = Emeans(1, 1, niter=niter, seed=7, assignment_backend=backend, capture_assignments=True)
                        caller.begin(len(rows), initial)
                        for epoch in range(niter):
                            for start in range(0, len(rows), caller.recipe['batch_size']):
                                piece = rows[start:start + caller.recipe['batch_size']]
                                stats = caller.accumulate(piece[:1])
                                if len(piece) > 1:
                                    self.assertIsNone(stats['sums'])
                                    self.assertIsNone(stats['counts'])
                                    stats = caller.accumulate(piece[1:])
                                self.assertEqual((stats['sums'].dtype, stats['counts'].dtype),
                                                 (torch.int64, torch.int64))
                                if epoch == niter - 1:
                                    torch.testing.assert_close(stats['labels'], torch.zeros_like(stats['labels']), rtol=0, atol=0)
                                caller.update(stats)
                            caller.end_epoch()
                        self.assertEqual(caller.finalize().item(), expected)
                        torch.testing.assert_close(rows, saved, rtol=0, atol=0)
                        torch.testing.assert_close(initial, torch.zeros_like(initial), rtol=0, atol=0)
        # An even-population balanced mean must be254.5 despite odd partitions.
        rows = torch.full((300002, 1), 255, dtype=torch.uint8, device='cuda')
        rows[::2] = 254
        model = Emeans(1, 1, niter=1, batch_size=131073, alpha0=.5,
                       assignment_backend='triton', assignment_chunk_size=70001)
        model.train(rows, torch.tensor([[255.]], device='cuda'))
        self.assertEqual(model.centroids.item(), 254.5)
        query = torch.tensor([[255]], dtype=torch.uint8, device='cuda')
        # After finalize, raw assignment must neither need running epoch fields
        # nor change finalization.254.5 rounds RN-even to represented254.
        distance, labels = model.assign(query)
        self.assertEqual((distance.item(), labels.item()), (1., 0))
        self.assertIs(model.finalize(), model.centroids)
        # Untrained assignment remains unsupported; do not override centroids.
        pair = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, assignment_backend='triton')
        with self.assertRaises(RuntimeError):
            pair.assign(query)
        pair.begin(4, torch.cat((model.centroids, torch.tensor([[255.]], device='cuda'))))
        # Caller-supplied integer totals produce the intended closing table.
        pair.update(dict(sums=torch.tensor([[int(2 * model.centroids.item())], [510]],
                                           dtype=torch.int64, device='cuda'),
                         counts=torch.tensor([2, 2], dtype=torch.int64, device='cuda'),
                         objective=torch.zeros((), dtype=torch.float64, device='cuda'), n=4))
        pair.end_epoch()
        torch.testing.assert_close(pair.finalize(), torch.tensor([[254.5], [255.]], device='cuda'),
                                   rtol=0, atol=0)
        distance, labels = pair.assign(query)
        self.assertEqual((distance.item(), labels.item()), (0., 1))

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0))
        and importlib.util.find_spec("triton") is not None, "qualified INT8 CUDA required"
    )
    def test_raw_integer_external_schema_empty_and_reset(self):
        def supplied(dtype, n=2):
            return dict(sums=torch.tensor([[255 * n]], dtype=dtype, device='cuda'),
                        counts=torch.tensor([n], dtype=dtype, device='cuda'),
                        objective=torch.zeros((), dtype=torch.float64, device='cuda'), n=n)
        args = dict(niter=1, batch_size=2, alpha0=.5, assignment_backend='triton')
        model = Emeans(1, 1, **args)
        initial = torch.tensor([[255.]], device='cuda')
        model.begin(4, initial)
        current = model.current_centroids()
        self.assertEqual((current.dtype, current.device), (torch.float32, initial.device))
        torch.testing.assert_close(current, initial, rtol=0, atol=0)
        # External-first selects integer schema. Ordinary copies remain valid.
        copied = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in supplied(torch.int64).items()}
        model.update(copied)
        for mixed in (
            supplied(torch.float32),
            dict(
                supplied(torch.int64),
                counts=torch.tensor([2.], device='cuda'),
            ),
        ):
            with self.assertRaises((TypeError, ValueError)):
                model.update(mixed)
        stats = model.accumulate(torch.full((2, 1), 255, dtype=torch.uint8, device='cuda'))
        self.assertEqual((stats['sums'].dtype, stats['counts'].dtype), (torch.int64, torch.int64))
        model.update(stats)
        model.end_epoch()
        self.assertEqual(model.finalize().item(), 255.)
        self.assertEqual(copied['sums'].item(), 510)
        model.reset()

        model.begin(2, initial)
        empty = model.accumulate(torch.empty((0, 1), dtype=torch.uint8, device='cuda'))
        self.assertEqual((empty['sums'].dtype, empty['counts'].dtype), (torch.int64, torch.int64))
        self.assertEqual((empty['sums'].item(), empty['counts'].item(), empty['n']), (0, 0, 0))
        with self.assertRaises(ValueError):
            model.update(empty)
        # Empty selection consumed no prescribed rows; a full external update fits.
        model.update(supplied(torch.int64))
        model.end_epoch()
        self.assertEqual(model.finalize().item(), 255.)
        model.reset()

        model.begin(4, initial)
        model.update(supplied(torch.float32))
        with self.assertRaises((TypeError, ValueError)):
            stats = model.accumulate(torch.full((2, 1), 255, dtype=torch.uint8, device='cuda'))
            model.update(stats)  # Either production or consumption may reject.
        model.reset()  # No new rollback guarantee; subsequent run must be valid.
        model.begin(2, initial)
        model.update(supplied(torch.float32))
        model.end_epoch()
        self.assertEqual(model.finalize().item(), 255.)

        # Declared-N check only: never allocate the giant population or permutation.
        model.reset()
        model.begin(2**53 // 255 + 1, initial)
        with self.assertRaises((TypeError, ValueError)):
            model.accumulate(torch.empty((0, 1), dtype=torch.uint8, device='cuda'))
        model.reset()

        # Actual schema-allocation failure poisons; host schema checks above do not.
        model.begin(2, initial)
        allocate = torch.Tensor.new_empty
        def failed_allocation(tensor, *args, **kwargs):
            allocate(tensor, *args, **kwargs)
            raise RuntimeError('injected statistics allocation failure')
        with mock.patch.object(torch.Tensor, 'new_empty', failed_allocation):
            with self.assertRaisesRegex(RuntimeError, 'injected statistics allocation'):
                model.accumulate(torch.empty((0, 1), dtype=torch.uint8, device='cuda'))
        with self.assertRaises(RuntimeError):
            model.current_centroids()
        model.begin(2, initial)
        model.update(model.accumulate(torch.full((2, 1), 255, dtype=torch.uint8, device='cuda')))
        model.end_epoch()
        self.assertEqual(model.finalize().item(), 255.)

    def test_explicit_fp8_calibration_and_scope(self):
        q = Fp8Quantizer(1)
        with self.assertRaises(RuntimeError):
            q.compute_codes(torch.zeros((1, 1)))
        sample = torch.tensor([[0.], [2.], [10.], [12.]])
        q.train(sample)
        self.assertTrue(q.is_trained)
        self.assertEqual(q.code_size, 1)
        self.assertEqual((q.scale, q.decision), (8., "small_sample_centered"))
        self.assertIs(q.to("cpu"), q)
        torch.testing.assert_close(q.decode(q.compute_codes(sample)), sample)
        self.assertEqual(q.compute_codes(sample[:0]).shape, (0, 1))
        with self.assertRaises(ValueError):
            q.validate(torch.tensor([[float("nan")]]))
        with self.assertRaisesRegex(ValueError, "FP8 requires Triton"):
            Emeans(1, 2, quantizer=q, assignment_backend="native")
        with self.assertRaises(ValueError):
            Emeans(1, 2, quantizer=q, assignment_backend="triton").train(sample)

        for n in (17, 2047, 2048, 3072):
            with self.subTest(rows=n):
                zero = torch.zeros(n, 3)
                fitted = Fp8Quantizer(3)
                fitted.train(zero)
                self.assertEqual(fitted.scale, 1.)
                self.assertEqual(fitted.code_size, 3)
                self.assertIsInstance(fitted.fit_rows, int)
                self.assertTrue(0 < fitted.fit_rows <= n)
                self.assertEqual(fitted.scale, 2. ** fitted.exponent)
                if n < 2048:
                    self.assertEqual(fitted.decision, "small_sample_centered")
                    self.assertIsNone(fitted.plain_excess)
                    self.assertIsNone(fitted.centered_excess)
                else:
                    self.assertEqual(fitted.decision, "uncentered_near_parity")
                    self.assertEqual((fitted.plain_excess, fitted.centered_excess), (0., 0.))
                self.assertEqual(fitted.decode(fitted.compute_codes(zero[:0])).shape, (0, 3))
                with self.assertRaises(ValueError):
                    fitted.validate(torch.zeros(1, 2))
                with self.assertRaises(TypeError):
                    fitted.train(torch.zeros(1, 3, dtype=torch.int32))
                self.assertFalse(fitted.is_trained)
                with self.assertRaises(RuntimeError):
                    fitted.decode(torch.zeros(1, 3))
        for bad in (torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
                    torch.tensor([[1.e-40], [-1.e-40]]), torch.empty(0, 1)):
            q.train(sample)
            with self.assertRaises(ValueError):
                q.train(bad)
            self.assertFalse(q.is_trained)

    def test_fp8_zero_reference_sse_selection(self):
        # The first2048 rows are queries and the last1024 are probe centroids.
        # Plain E4M3 collapses both values to32 and incurs SSE1/64; centering
        # preserves them as -32/+32 with zero SSE.  A zero denominator must
        # therefore distinguish the positive-error candidate from parity.
        values = torch.tensor([1., 257. / 256.])
        collapse = values.repeat(1536).reshape(-1, 1)
        selected = Fp8Quantizer(1)
        selected.train(collapse)
        self.assertEqual((selected.centered, selected.decision),
                         (True, "centered_nonpositive_excess"))
        self.assertEqual((selected.plain_excess, selected.centered_excess),
                         (1., 0.))

        parity = Fp8Quantizer(3)
        parity.train(torch.zeros(3072, 3))
        self.assertEqual((parity.centered, parity.decision),
                         (False, "uncentered_near_parity"))
        self.assertEqual((parity.plain_excess, parity.centered_excess),
                         (0., 0.))

        positive = Fp8Quantizer(4)
        positive.train(torch.randn(
            2048, 4, generator=torch.Generator().manual_seed(7)))
        self.assertEqual((positive.centered, positive.decision),
                         (False, "uncentered_insufficient_gain;both_above_quality_gate"))
        self.assertGreater(positive.plain_excess, 0.)
        self.assertGreater(positive.centered_excess, 0.)

    def test_fp8_positive_reference_selector_and_explicit_validation(self):
        x = torch.randn(2048, 4, generator=torch.Generator().manual_seed(7))
        quantizers = []
        for scale in (1., 2. ** -60):
            q = Fp8Quantizer(4)
            q.train(x * scale)
            quantizers.append(q)
            self.assertTrue(0 < q.fit_rows <= len(x))
            self.assertTrue(np.isfinite(q.sample_peak))
            self.assertGreater(q.scale, 0.)
            q.validate(x * scale)
            with self.assertRaises(ValueError):
                q.validate(torch.full((1, 4), float("inf")))
        a, b = quantizers
        self.assertEqual(a.decision, b.decision)
        self.assertEqual(a.centered, b.centered)
        self.assertEqual(a.plain_excess, b.plain_excess)
        self.assertEqual(a.centered_excess, b.centered_excess)
        self.assertEqual(b.exponent - a.exponent, 60)
        shifted = Fp8Quantizer(4)
        shifted.train(x + torch.tensor([1.e4, -2.e4, 3.e4, -4.e4]))
        self.assertTrue(shifted.centered)
        plain = Fp8Quantizer(4)
        plain.train(torch.randn(2048, 4, generator=torch.Generator().manual_seed(4)))
        self.assertFalse(plain.centered)
        with self.assertRaises(ValueError):
            plain.validate(torch.full((1, 4), 1.e20))
        with mock.patch.object(torch, "float8_e4m3fn", None):
            with self.assertRaisesRegex(RuntimeError, "float8_e4m3fn"):
                Fp8Quantizer(1).train(torch.zeros(2, 1))

    def test_fp8_explicit_unsupported_model_scope(self):
        rows = torch.zeros(2, 1).to(torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            Emeans(1, 1).train(rows)
        with self.assertRaises(ValueError):
            Emeans(1, 1, assignment_backend="triton").train(rows)
        if TRITON_CUDA and not FP8_CUDA:
            with self.assertRaisesRegex(RuntimeError, "SM90|SM100|SM103"):
                Emeans(1, 1, assignment_backend="triton").train(rows.cuda())
        if FP8_CUDA:
            q = Fp8Quantizer(1)
            q.train(torch.zeros(2, 1, device="cuda"))
            for backend in ('triton',):
                for dtype in (torch.int8, torch.uint8):
                    with self.subTest(backend=backend, unsupported_fp8=dtype):
                        byte_rows = torch.zeros(2, 1, dtype=dtype, device="cuda")
                        initial = torch.zeros(1, 1, device="cuda")
                        for supplied_initial in (None, initial):
                            model = Emeans(1, 1, quantizer=q, assignment_backend=backend)
                            with self.assertRaises(ValueError):
                                model.train(byte_rows, supplied_initial)
                            if supplied_initial is None:
                                self.assertIsNone(model.centroids)
                                self.assertIsNone(model.recipe)  # Default seeds reject before begin.
                        model = Emeans(1, 1, niter=1, batch_size=2, alpha0=1.,
                                       quantizer=q, assignment_backend=backend)
                        model.begin(2, initial)
                        with self.assertRaises(ValueError):
                            model.accumulate(byte_rows)
                        floating_rows = torch.zeros(2, 1, device="cuda")
                        model.update(model.accumulate(floating_rows))
                        model.end_epoch()
                        model.finalize()
                        with self.assertRaises(ValueError):
                            model.assign(byte_rows)
                        distance, labels = model.assign(q.compute_codes(floating_rows))
                        torch.testing.assert_close(distance, torch.zeros(2, device="cuda"), rtol=0, atol=0)
                        self.assertTrue(torch.equal(labels, torch.zeros(2, dtype=torch.int64, device="cuda")))
        if TRITON_CUDA:
            for capability in ((8, 0), (8, 9)):
                with mock.patch.object(torch.cuda, "get_device_capability", return_value=capability):
                    with self.assertRaisesRegex(RuntimeError, "SM90|SM100|SM103"):
                        Emeans(1, 1, assignment_backend="triton").train(rows.cuda())
            with mock.patch.object(torch.version, "hip", "scope-only probe"):
                with self.assertRaises(RuntimeError):
                    Emeans(1, 1, assignment_backend="triton").train(rows.cuda())


    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3))
        and importlib.util.find_spec("triton") is not None, "qualified FP8 CUDA required"
    )
    def test_fp8_native_jit_decoded_lifecycle(self):
        x = torch.tensor([[0.1], [2.2], [10.3], [12.4]], device="cuda")
        q = Fp8Quantizer(1)
        q.train(x)
        codes = q.compute_codes(x)
        represented = q.decode(codes)
        self.assertFalse(torch.equal(x, represented))
        init = torch.tensor([[0.], [10.]], device="cuda")
        args = dict(niter=1, batch_size=4, alpha0=0.5, quantizer=q, capture_assignments=True,
                    assignment_backend="triton", assignment_chunk_size=2)
        jit, native, steps = (Emeans(1, 2, **args) for _ in range(3))
        jit.train(x, init)
        native.train(codes, init)
        steps.begin(len(x), init)
        current = steps.current_centroids()
        self.assertEqual((current.dtype, current.device), (torch.float32, init.device))
        torch.testing.assert_close(current, init, rtol=0, atol=0)
        partial = steps.accumulate(codes[:1])
        self.assertIsNone(partial['sums'])
        self.assertIsNone(partial['counts'])
        torch.testing.assert_close(steps.current_centroids(), init, rtol=0, atol=0)
        stats = steps.accumulate(codes[1:])
        torch.testing.assert_close(stats['labels'], torch.tensor([0, 1, 1], device="cuda"), rtol=0, atol=0)
        steps.update(stats)
        steps.end_epoch()
        steps.finalize()
        expected = represented.reshape(2, 2, 1).mean(dim=1)
        for model in (jit, native, steps):
            torch.testing.assert_close(model.centroids, expected)
            torch.testing.assert_close(model.obj, jit.obj, atol=0, rtol=0)
            distance, labels = model.assign(codes)
            operands = q.decode(q.compute_codes(expected))
            oracle = (represented[:, None] - operands[None, :]).square().sum(2)
            torch.testing.assert_close(distance, oracle.min(1).values, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(labels, oracle.argmin(1))
        self.assertEqual((jit.centroids.dtype, distance.dtype, labels.dtype),
                         (torch.float32, torch.float32, torch.int64))
        raw = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5, capture_assignments=True,
                     assignment_backend="triton")
        raw.train(codes, codes[[0, 2]].float())
        torch.testing.assert_close(raw.centroids, codes.float().reshape(2, 2, 1).mean(1))
        torch.testing.assert_close(raw.assignments, native.assignments, rtol=0, atol=0)


    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3))
        and importlib.util.find_spec("triton") is not None, "qualified FP8 CUDA required"
    )
    def test_fp8_long_positive_selected_distances(self):
        d = 4096
        base = (torch.arange(d, device='cuda') % 251).float()
        x = torch.stack((base, base + 32, base + 40))
        q = Fp8Quantizer(d)
        q.train(torch.stack((torch.full_like(base, -128), torch.full_like(base, 128))))
        codes = q.compute_codes(x)
        for k in (2, 1024):
            initial = (base + 96).repeat(k, 1)
            initial[:2] = x[:2]
            operands = q.decode(q.compute_codes(initial)).double()
            oracle = (q.decode(codes).double()[:, None] - operands[None]).square().sum(2)
            for rows in (codes, x):
                with self.subTest(k=k, native=rows is codes):
                    model = self._known_centroids(
                        initial, quantizer=q, assignment_backend='triton'
                    )
                    distance, labels = model.assign(rows)
                    torch.testing.assert_close(labels, oracle.argmin(1))
                    chosen = oracle.gather(1, labels[:, None]).flatten()
                    torch.testing.assert_close(distance.double(), chosen, rtol=2e-6, atol=1e-3)
                    self.assertEqual(distance[:2].tolist(), [0., 0.])

    @unittest.skipUnless(FP8_CUDA, "qualified FP8 CUDA required")
    def test_fp8_matched_norm_ragged_integer_and_allowed_ties(self):
        rows = torch.tensor([[-8.5485334, .5754005, 5.9468751]], device="cuda")
        centroids = torch.tensor([[1.3439447, -6.8812032, 5.9468989],
                                 [1.6238794, -.7164297, -.9630429]], device="cuda")
        xq, cq = rows.to(torch.float8_e4m3fn).float(), centroids.to(torch.float8_e4m3fn).float()
        self.assertEqual(((xq[:, None] - cq).square().sum(2)).argmin(1).item(), 1)
        wrong = centroids.square().sum(1)[None] - 2 * xq @ cq.T
        self.assertEqual(wrong.argmin(1).item(), 0)
        model = self._known_centroids(centroids, assignment_backend="triton")
        self.assertEqual(model.assign(xq.to(torch.float8_e4m3fn))[1].item(), 1)
        for n, k, d in ((3, 7, 17), (17, 8065, 65), (65, 513, 129), (3, 4232, 512)):
            with self.subTest(n=n, k=k, d=d):
                g = torch.Generator(device="cuda").manual_seed(n + k + d)
                c = torch.randint(-4, 5, (k, d), generator=g, device="cuda").float()
                x = torch.randint(-4, 5, (n, d), generator=g, device="cuda").float()
                # Integer products/sums here are far below the FP32 exact bound.
                oracle = (x.double()[:, None] - c.double()).square().sum(2)
                m = self._known_centroids(c, assignment_backend="triton")
                distance, labels = m.assign(x.to(torch.float8_e4m3fn))
                torch.testing.assert_close(labels, oracle.argmin(1), rtol=0, atol=0)
                torch.testing.assert_close(distance.double(), oracle.min(1).values, rtol=0, atol=0)
                tied = self._known_centroids(torch.zeros_like(c), assignment_backend="triton")
                self.assertTrue(torch.equal(tied.assign(torch.zeros_like(x).to(torch.float8_e4m3fn))[1],
                                            torch.zeros(n, dtype=torch.int64, device="cuda")))
        d = 4096
        x = (torch.arange(d, device="cuda") % 31 + 128).to(torch.float8_e4m3fn)
        c = x.float().repeat(3, 1)
        c[0, 0] += 16
        c[1, 1] += 16
        c[2] += 64
        m = self._known_centroids(c, assignment_backend="triton")
        distance, labels = m.assign(x[None])
        self.assertIn(labels.item(), (0, 1))
        selected = (x.double() - c.to(torch.float8_e4m3fn).double()[labels.item()]).square().sum()
        torch.testing.assert_close(distance.double()[0], selected, rtol=2e-6, atol=1e-6)

    @unittest.skipUnless(FP8_CUDA, "qualified FP8 CUDA required")
    def test_fp8_decoded_tail_statistics_and_exact_code_lattice(self):
        x = torch.tensor([[.1, -.2], [2.2, .3], [10.3, .4], [12.4, .5], [11.8, .25]], device="cuda")
        q = Fp8Quantizer(2)
        q.train(x)
        initial = torch.tensor([[0., 0.], [10., 0.]], device="cuda")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                incoming = x.to(dtype)
                codes = q.compute_codes(incoming)
                decoded = q.decode(codes)
                m = Emeans(2, 2, niter=2, batch_size=2, alpha0=.5, seed=7, quantizer=q,
                           assignment_backend="triton", assignment_chunk_size=1, capture_assignments=True)
                m.begin(len(x), initial)
                for epoch in range(2):
                    order = e_means._randperm(len(x), _epoch_seed(7, epoch), True, x.device)
                    for start in range(0, len(x), 2):
                        indices = order[start:start + 2]
                        batch = codes[indices]
                        before = m.current_centroids().clone()
                        cq = q.compute_codes(before).float()
                        squared = (batch.float()[:, None] - cq[None]).square().sum(2)
                        labels = squared.argmin(1)
                        sums = torch.zeros_like(before)
                        sums.index_add_(0, labels, decoded[indices])
                        counts = torch.bincount(labels, minlength=2).float()
                        for offset in range(len(indices)):
                            stats = m.accumulate(incoming[indices[offset:offset + 1]])
                            torch.testing.assert_close(m.current_centroids(), before, rtol=0, atol=0)
                            if offset + 1 < len(indices):
                                self.assertIsNone(stats['sums'])
                                self.assertIsNone(stats['counts'])
                            if epoch == m.niter - 1:
                                torch.testing.assert_close(stats['labels'], labels[offset:offset + 1], rtol=0, atol=0)
                        self.assertEqual((stats['sums'].dtype, stats['counts'].dtype,
                                          stats['objective'].dtype), (torch.float32, torch.float32, torch.float64))
                        torch.testing.assert_close(stats['sums'], sums, rtol=0, atol=0)
                        torch.testing.assert_close(stats['counts'], counts, rtol=0, atol=0)
                        expected = torch.ldexp(squared.min(1).values, torch.tensor(-2 * q.exponent, device="cuda")).double().sum()
                        torch.testing.assert_close(stats['objective'], expected, rtol=1e-6, atol=3e-3)
                        m.update(stats)
                    m.end_epoch()
                m.finalize()
                reference = Emeans(2, 2, niter=2, batch_size=2, alpha0=.5, seed=7, quantizer=q,
                                   assignment_backend="triton", assignment_chunk_size=1)
                reference.train(codes, initial)
                torch.testing.assert_close(m.centroids, reference.centroids, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(m.obj, reference.obj, rtol=1e-6, atol=3e-5)
        # Abs 2**-9 lattice units per coordinate <2**24; bounded norm/dot
        # intermediates and normal power-of-two decode make these totals exact.
        q = Fp8Quantizer(2)
        q.train(torch.tensor([[-8., -8.], [8., 8.]], device="cuda").repeat(1024, 1))
        self.assertFalse(q.centered)
        codes = torch.tensor([[1., 2.], [-1., 0.], [2. ** -9, -2. ** -9]], device="cuda").to(torch.float8_e4m3fn)
        self.assertTrue(bool((codes.float().abs().sum(0) * 512 < 2 ** 24).all()))
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            for deterministic in (False, True):
                with self.subTest(code_lattice_deterministic=deterministic):
                    torch.use_deterministic_algorithms(deterministic)
                    m = Emeans(2, 1, niter=1, batch_size=3, alpha0=1., quantizer=q,
                               assignment_backend="triton", assignment_chunk_size=1, capture_assignments=True)
                    m.begin(3, torch.zeros(1, 2, device="cuda"))
                    partial = m.accumulate(codes[:1])
                    self.assertEqual(partial['n'], 1)
                    self.assertIsNone(partial['sums'])
                    self.assertIsNone(partial['counts'])
                    empty = m.accumulate(codes[:0])
                    self.assertEqual(empty['n'], 1)
                    self.assertEqual(empty['labels'].numel(), 0)
                    # Changing helper eligibility must not mix decoded and
                    # code-unit contributions within one frozen batch.
                    torch.use_deterministic_algorithms(not deterministic)
                    stats = m.accumulate(codes[1:])
                    expected = q.decode(codes).sum(0, keepdim=True)
                    torch.testing.assert_close(stats['sums'], expected, rtol=0, atol=0)
                    torch.testing.assert_close(stats['counts'], torch.tensor([3.], device="cuda"), rtol=0, atol=0)
                    saved = stats['sums'].clone()
                    torch.testing.assert_close(m.accumulate(codes[:0])['sums'], saved, rtol=0, atol=0)
                    m.update(stats)
                    m.end_epoch()
                    torch.testing.assert_close(m.finalize(), expected / 3, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous)

    @unittest.skipUnless(FP8_CUDA, "qualified FP8 CUDA required")
    def test_fp8_ldexp_scaling_and_finite_decoded_neighbor(self):
        from faiss.contrib.e_means.e_means_fp8 import separate_splits
        for exponent, expected in ((75, 2. ** -148), (-100, 0.)):
            peak = 2. ** (5 - exponent)
            q = Fp8Quantizer(1)
            q.train(torch.tensor([[-peak], [peak]], device="cuda"))
            self.assertEqual(q.exponent, exponent)
            encoded = 0. if exponent == -100 else 2.
            codes = torch.tensor([[encoded]], device="cuda").to(torch.float8_e4m3fn)
            m = Emeans(1, 1, niter=1, batch_size=1, alpha0=1., quantizer=q, assignment_backend="triton")
            m.begin(1, torch.zeros(1, 1, device="cuda"))
            stats = m.accumulate(codes)
            self.assertEqual(stats['objective'].item(), expected)
            m.update(stats)
            m.end_epoch()
            m.finalize()
            self.assertEqual(m.assign(codes)[0].item(), 0.)
        donor = torch.tensor([[0., 0.], [1., 2.], [-1., 3.], [448., 1.]], device="cuda")
        child = donor.clone()
        saved = donor.clone()
        separate_splits(donor, child, None)
        torch.testing.assert_close(child[:, 0], torch.tensor([2. ** -9, 1.125, -.9375, 416.], device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(child[:, 1:], saved[:, 1:], rtol=0, atol=0)
        ordinary = saved + 10
        separate_splits(donor, ordinary, None)
        torch.testing.assert_close(ordinary, saved + 10, rtol=0, atol=0)
        q = Fp8Quantizer(2)
        q.train(torch.tensor([[0., -2.], [2., 2.]], device="cuda"))
        donor = saved / q.scale + q.shift
        child = donor.clone()
        untouched = child[:, 1:].clone()
        separate_splits(donor, child, q)
        torch.testing.assert_close(q.compute_codes(child).float()[:, 0],
                                   torch.tensor([2. ** -9, 1.125, -.9375, 416.], device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(child[:, 1:], untouched, rtol=0, atol=0)
        maximum = torch.finfo(torch.float32).max
        q = Fp8Quantizer(1)
        q.train(torch.tensor([[-.97 * maximum], [.97 * maximum]], device="cuda"))
        self.assertEqual(q.exponent, -123)
        donor = torch.tensor([[.97 * maximum * (1 - 1.e-4)]], device="cuda")
        child = torch.tensor([[.97 * maximum * (1 + 1.e-4)]], device="cuda")
        saved = child.clone()
        torch.testing.assert_close(q.compute_codes(donor).float(), q.compute_codes(child).float(), rtol=0, atol=0)
        separate_splits(donor, child, q)
        torch.testing.assert_close(child, saved, rtol=0, atol=0)

        # Compare actual model writes with the stock represented-row primitive.
        code_rows = torch.tensor([[0., -0.], [0., 1.0625], [-1., 3.], [448., 1.]], device='cuda')
        for kind in ('raw', 'plain', 'centered', 'tiny'):
            q = None
            if kind != 'raw':
                sample = torch.tensor([[-.5, -.5], [.5, .5]])
                if kind == 'centered':
                    sample = sample + 10
                elif kind == 'tiny':
                    sample = sample * 2. ** -121
                q = Fp8Quantizer(2)
                q.train(sample if kind == 'centered' else sample.repeat(1024, 1))
                q.to('cuda')
                self.assertEqual(q.centered, kind == 'centered')
            donors = code_rows if q is None else q.decode(code_rows)
            for backend in ('triton',):
                with self.subTest(model_writer=kind, backend=backend):
                    initial = torch.cat((donors, donors, donors[:1]))
                    model = Emeans(2, 9, niter=1, batch_size=9, alpha0=.5,
                                   quantizer=q, assignment_backend=backend)
                    model.begin(9, initial)
                    model.accumulate(code_rows[:0].to(torch.float8_e4m3fn))
                    model._usage.fill_(8)
                    offset = donors.new_tensor([1., -1.]) * model.recipe['split_epsilon']
                    low, high = donors * (1. - offset), donors * (1. + offset)
                    separate_splits(low, high, q)
                    expected = torch.cat((low, high, donors[:1]))
                    source = torch.arange(4, device='cuda', dtype=torch.int64)
                    self.assertEqual(model._apply_splits(source, source + 4), 4)
                    torch.testing.assert_close(model._centroids.view(torch.int32),
                                               expected.view(torch.int32), rtol=0, atol=0)
                    torch.testing.assert_close(model._usage,
                                               donors.new_tensor([4.] * 8 + [8.]), rtol=0, atol=0)
                    model.reset()

    @unittest.skipUnless(FP8_CUDA, "qualified FP8 CUDA required")
    def test_fp8_metadata_default_codes_and_output_ownership(self):
        initial = torch.zeros(1, 1, device="cuda")
        q = Fp8Quantizer(1)
        m = Emeans(1, 1, niter=1, batch_size=1, alpha0=.5,
                   quantizer=q, assignment_backend="triton")
        with self.assertRaisesRegex(RuntimeError, "trained"):
            m.begin(1, initial)
        q.train(torch.zeros(2, 1))
        with self.assertRaisesRegex(ValueError, "same device"):
            m.begin(1, initial)
        q.to(initial.device)
        m.begin(1, initial)
        m.update(dict(sums=initial.clone(), counts=torch.ones(1, device="cuda"),
                      objective=torch.zeros((), device="cuda", dtype=torch.float64), n=1))
        m.end_epoch()
        torch.testing.assert_close(m.finalize(), initial, rtol=0, atol=0)
        x = torch.randn(7, 3, device="cuda", generator=torch.Generator(device="cuda").manual_seed(123))
        q = Fp8Quantizer(3)
        q.train(x)
        codes = q.compute_codes(x)
        before = codes.view(torch.uint8).clone()
        indices = torch.randperm(7, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))[:2]
        for calibrated in (False, True):
            with self.subTest(calibrated=calibrated):
                selected = q if calibrated else None
                args = dict(niter=1, batch_size=3, alpha0=.5, seed=7,
                            assignment_chunk_size=1, quantizer=selected, assignment_backend="triton")
                default, explicit = Emeans(3, 2, **args), Emeans(3, 2, **args)
                chosen = codes.float()[indices]
                initial = q.decode(chosen) if calibrated else chosen
                explicit.train(codes, initial)
                default.train(codes)
                torch.testing.assert_close(default.centroids, explicit.centroids, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(default.obj, explicit.obj, rtol=1e-6, atol=3e-5)
                torch.testing.assert_close(codes.view(torch.uint8), before, rtol=0, atol=0)
                output, saved = default.centroids, default.centroids.clone()
                default.reset()
                torch.testing.assert_close(output, saved, rtol=0, atol=0)
                raw = Emeans(3, 2, **args)
                raw.begin(7, initial)
                empty = raw.accumulate(codes[:0])
                self.assertEqual(empty['n'], 0)
                # Consume the empty pending contribution using a prescribed
                # three-copy contribution at its represented first centroid.
                supplied = torch.zeros(2, 3, device="cuda")
                supplied[0] = initial[0] * 3
                raw.update(dict(sums=supplied,
                                counts=torch.tensor([3., 0.], device="cuda"),
                                objective=torch.zeros((), device="cuda", dtype=torch.float64), n=3))
                if not calibrated:
                    with self.assertRaises(ValueError):
                        raw.accumulate(x[:1])
                else:
                    contribution = raw.accumulate(x[:1])
                    self.assertEqual(contribution['n'], 1)
                    self.assertIsNone(contribution['sums'])
                    self.assertIsNone(contribution['counts'])
                    contribution = raw.accumulate(x[1:3])
                    self.assertEqual((contribution['sums'].dtype, contribution['counts'].dtype),
                                     (torch.float32, torch.float32))
                    self.assertEqual(contribution['counts'].sum().item(), 3.)

    @unittest.skipUnless(FP8_CUDA, "qualified FP8 CUDA required")
    def test_fp8_finite_warm_updates_and_real_range_rejection(self):
        from faiss.contrib.e_means.e_means_fp8 import separate_splits
        x = torch.tensor([[-1., -1.], [1., 1.]], device="cuda")
        q = Fp8Quantizer(2)
        q.train(x)
        for route in ("raw", "codes", "floating"):
            with self.subTest(route=route):
                selected = None if route == "raw" else q
                batch = x if route == "floating" else q.compute_codes(x)
                initial = q.compute_codes(x).float() if route == "raw" else x
                m = Emeans(2, 2, niter=1, batch_size=2, alpha0=.5,
                           quantizer=selected, assignment_backend="triton")
                m.begin(2, initial)
                m.update(m.accumulate(batch))
                m.begin(2, initial)
                donor = torch.zeros(1, 2, device="cuda")
                child = donor.clone()
                torch.cuda.synchronize()
                previous = torch.cuda.get_sync_debug_mode()
                try:
                    torch.cuda.set_sync_debug_mode("error")
                    m.update(m.accumulate(batch))
                    separate_splits(donor, child, selected)
                finally:
                    torch.cuda.set_sync_debug_mode(previous)
                self.assertTrue(bool((child != donor).any()))
        stable = torch.full((2, 1), 1000., device="cuda")
        q = Fp8Quantizer(1)
        q.train(stable)
        m = Emeans(1, 2, niter=3, batch_size=2, alpha0=.5, quantizer=q, assignment_backend="triton")
        m.train(stable, stable)
        self.assertTrue(bool(torch.isfinite(m.centroids).all()))
        large = torch.full((10, 1), 1.e7, device="cuda")
        q.train(large)
        m = Emeans(1, 2, niter=1, batch_size=2, alpha0=.05, quantizer=q, assignment_backend="triton")
        with self.assertRaisesRegex(ValueError, "representable range"):
            m.train(large, torch.tensor([[1.e7], [1.e7 + 400]], device="cuda"))
        self.assertEqual(m.n_splits, 1)
        for calibrated in (False, True):
            codec = Fp8Quantizer(1) if calibrated else None
            if codec is not None:
                codec.train(torch.tensor([[-1.], [1.]], device="cuda"))
            model = self._known_centroids(torch.ones(1, 1, device="cuda"),
                                          quantizer=codec, assignment_backend="triton")
            model.centroids.fill_(17. if calibrated else 512.)
            query = torch.zeros(1, 1, device="cuda").to(torch.float8_e4m3fn)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "centroids.*representable range"):
                    model.assign(query)
            model.centroids.fill_(1.)
            for _ in range(2):
                distance, labels = model.assign(query)
                torch.testing.assert_close(distance, torch.ones_like(distance), rtol=0, atol=0)
                self.assertEqual(labels.item(), 0)

    @unittest.skipUnless(FP8_CUDA and torch.cuda.device_count() >= 2,
                         "FP8 and two CUDA devices required")
    def test_fp8_noncurrent_device_decode_assign_and_split(self):
        from faiss.contrib.e_means.e_means_fp8 import separate_splits
        current = torch.cuda.current_device()
        other = torch.device("cuda", 1 if current == 0 else 0)
        for exponent in (6, 75):
            peak = 2. ** (5 - exponent)
            q = Fp8Quantizer(1)
            q.train(torch.tensor([[-peak], [peak]]))
            q.to(other)
            codes = torch.tensor([[2.]], device=other).to(torch.float8_e4m3fn)
            m = self._known_centroids(
                torch.zeros(1, 1, device=other),
                quantizer=q,
                assignment_backend="triton",
            )
            distance, labels = m.assign(codes)
            expected = torch.ldexp(torch.tensor([4.]), torch.tensor(-2 * exponent))
            torch.testing.assert_close(distance.cpu(), expected, rtol=0, atol=0)
            self.assertEqual((distance.device, labels.device), (other, other))
            if exponent == 6:
                torch.testing.assert_close(q.decode(codes).cpu(), torch.tensor([[2. ** -5]]), rtol=0, atol=0)
                donor = torch.zeros(1, 1, device=other)
                child = donor.clone()
                separate_splits(donor, child, q)
                torch.testing.assert_close(child.cpu(), torch.tensor([[2. ** -15]]), rtol=0, atol=0)
            self.assertEqual(torch.cuda.current_device(), current)

    @unittest.skipUnless(FP8_CUDA, "qualified FP8 CUDA required")
    def test_fp8_conditioned_positive_distances_and_ranking_margin(self):
        for d, offset in ((384, 128), (4096, 0)):
            with self.subTest(d=d, offset=offset):
                g = torch.Generator(device="cuda").manual_seed(1818)
                c = (torch.randint(0, 32, (384, d), device="cuda", generator=g).float() + offset).to(torch.float8_e4m3fn).float()
                x = (torch.randint(0, 32, (8, d), device="cuda", generator=g).float() + offset).to(torch.float8_e4m3fn)
                x[0].copy_(c[7])
                m = self._known_centroids(c, assignment_backend="triton")
                distance, labels = m.assign(x)
                selected = (x.double() - c[labels].double()).square().sum(1)
                torch.testing.assert_close(distance.double(), selected, rtol=2e-6, atol=1e-6)
                self.assertEqual((labels[0].item(), distance[0].item()), (7, 0.))
                saved = distance.clone(), labels.clone()
                m.assign(x[1:])
                torch.testing.assert_close(distance, saved[0], rtol=0, atol=0)
                torch.testing.assert_close(labels, saved[1], rtol=0, atol=0)
        for d in (384, 4096):
            g = torch.Generator(device="cuda").manual_seed(20261016)
            population = torch.randint(0, 256, (4096 + 128, d), device="cuda", dtype=torch.uint8, generator=g)
            population = population.float().mul_(.125).to(torch.float8_e4m3fn)
            c, x = population[:4096].float(), population[4096:].clone()
            x[33].copy_(population[1052])
            m = self._known_centroids(c, assignment_backend="triton")
            _, labels = m.assign(x)
            minimum = torch.full((len(x),), float("inf"), device="cuda", dtype=torch.float64)
            for start in range(0, len(c), 32):
                direct = (x.double()[:, None] - c[start:start + 32].double()).square().sum(2)
                minimum = torch.minimum(minimum, direct.min(1).values)
            selected = (x.double() - c[labels].double()).square().sum(1)
            # This retained separated-positive fixture permits near ties; it is
            # not a universal FP64 argmin or low-precision error bound.
            self.assertTrue(bool((selected - minimum <= minimum * 2e-4 + 1e-6).all()))


    def test_backend_selection_and_cpu_rejection(self):
        self.assertEqual(Emeans(1, 1).assignment_backend, "auto")
        with self.assertRaises(ValueError):
            Emeans(1, 1, assignment_backend="unknown")
        with self.assertRaisesRegex(ValueError, "Triton assignment"):
            Emeans(1, 1, assignment_backend="triton").train(torch.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "Triton assignment"):
            Emeans(1, 1, assignment_backend="triton").train(np.zeros((2, 1), dtype="float32"))
        args = dict(niter=2, batch_size=7, alpha0=0.2, seed=9)
        rows = torch.arange(68, dtype=torch.float32).reshape(17, 4)
        default, explicit = Emeans(4, 3, **args), Emeans(4, 3, assignment_backend="native", **args)
        default.train(rows)
        explicit.train(rows)
        torch.testing.assert_close(default.centroids, explicit.centroids, rtol=0, atol=0)
        torch.testing.assert_close(default.obj, explicit.obj, rtol=0, atol=0)
        self.assertEqual(Emeans(1, 2**31-256, assignment_backend="triton").k, 2**31-256)
        with self.assertRaisesRegex(ValueError, r"k <= 2\*\*31 - 256"):
            Emeans(1, 2**31-255, assignment_backend="triton")

    def test_optional_triton_dependency_isolation(self):
        # A fresh interpreter avoids accidentally reusing an already imported
        # optional backend. Native execution must not require Triton.
        script = '''
import importlib.abc, importlib.util, sys, pathlib, torch
from faiss.contrib.e_means import Emeans, e_means
assert pathlib.Path(e_means.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve()
class NoTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ImportError("test blocks optional Triton")
available = importlib.util.find_spec("triton") is not None
cuda_supported = (torch.cuda.is_available() and not torch.version.hip
                  and torch.cuda.get_device_capability() >= (8, 0))
sys.meta_path.insert(0, NoTriton())
assert "triton" not in sys.modules
model = Emeans(1, 1, niter=1, batch_size=2, alpha0=0.5)
model.train(torch.zeros((2, 1)))
assert "triton" not in sys.modules
if cuda_supported:
    tiled = Emeans(1, 1, assignment_backend="triton", niter=1, batch_size=2, alpha0=0.5)
    try:
        tiled.train(torch.zeros((2, 1), device="cuda"))
    except ImportError as error:
        assert "triton package" in str(error)
    else:
        raise AssertionError("missing Triton did not reject the requested backend")
    sys.meta_path.pop(0)
    if available:
        tiled.reset()
        tiled.train(torch.zeros((2, 1), device="cuda"))
        assert float(tiled.centroids[0, 0]) == 0
'''
        subprocess.run([sys.executable, "-c", script, str(Path(e_means.__file__).resolve())],
                       check=True, timeout=60)

    def _floating_backends(self):
        return ("native", "triton") if TRITON_CUDA else ("native",)

    @unittest.skipUnless(TRITON_CUDA, "Triton NVIDIA SM80 required")
    def test_triton_planted_ragged_and_long_assignments(self):
        # Large searches use separated planted winners, not another expanded
        # matmul as a supposedly independent ranking oracle.
        cases = ((torch.float32, 1, 1, 1), (torch.float32, 17, 19, 17),
                 (torch.float16, 17, 19, 17), (torch.bfloat16, 17, 19, 17),
                 (torch.float32, 2, 1025, 129), (torch.float32, 1, 17, 3000),
                 (torch.bfloat16, 2, 8193, 511), (torch.float16, 2, 4097, 512),
                 (torch.bfloat16, 2, 8193, 513), (torch.float16, 1025, 257, 65),
                 (torch.bfloat16, 3, 257, 128))
        with highest_fp32("cuda"):
            for dtype, b, k, d in cases:
                with self.subTest(dtype=dtype, shape=(b, k, d)):
                    generator = torch.Generator(device="cuda").manual_seed(1000+b+k+d)
                    centroids = torch.randn(k, d, generator=generator, device="cuda")/4
                    expected = (torch.arange(b, device="cuda")*17+3) % k
                    rows = centroids[expected].clone()
                    rows[:, 0] += 1/32
                    saved = rows.clone()
                    model = self._known_centroids(centroids, assignment_backend="triton")
                    distance, labels = model.assign(rows.to(dtype))
                    direct = (rows.to(dtype).float()-centroids[expected].to(dtype).float()).square().sum(1)
                    torch.testing.assert_close(labels, expected, rtol=0, atol=0)
                    # Retained bounded factorized fixtures' tolerance, not a
                    # dimension-wide accuracy guarantee for arbitrary inputs.
                    torch.testing.assert_close(distance, direct, atol=max(2e-4, d*1e-5), rtol=2e-5)
                    torch.testing.assert_close(rows, saved, rtol=0, atol=0)
                    self.assertEqual((distance.dtype, labels.dtype), (torch.float32, torch.int64))

    @unittest.skipUnless(TRITON_CUDA, "Triton NVIDIA SM80 required")
    def test_triton_global_ties_ieee_and_large_half_norms(self):
        for dtype, b, k, d, first, second in (
            (torch.bfloat16, 1025, 257, 65, 3, 130),
            (torch.float16, 2, 3969, 128, 255, 256),
            (torch.float32, 1, 257, 17, 124, 125),
        ):
            with self.subTest(dtype=dtype, shape=(b, k, d)):
                rows = torch.ones((b, d), device="cuda", dtype=dtype)
                centroids = torch.full((k, d), 10., device="cuda")
                centroids[[first, second]] = 1
                distance, labels = self._known_centroids(centroids, assignment_backend="triton").assign(rows)
                torch.testing.assert_close(labels, torch.full((b,), first, device="cuda", dtype=torch.int64), rtol=0, atol=0)
                torch.testing.assert_close(distance, torch.zeros(b, device="cuda"), rtol=0, atol=0)
        for d, k, value in ((1024, 2, 10.), (512, 3969, 12.)):
            rows = torch.full((2, d), value, device="cuda", dtype=torch.float16)
            centroids = torch.full((k, d), value+1, device="cuda")
            centroids[0] = value
            self.assertGreater(float(centroids[0].square().sum()), torch.finfo(torch.float16).max)
            distance, labels = self._known_centroids(centroids, assignment_backend="triton").assign(rows)
            torch.testing.assert_close(distance, torch.zeros(2, device="cuda"), rtol=0, atol=0)
            torch.testing.assert_close(labels, torch.zeros(2, device="cuda", dtype=torch.int64), rtol=0, atol=0)
        with highest_fp32("cuda"):
            rows = torch.tensor([[473.1382141113281]], device="cuda")
            centroids = torch.tensor([[477.6440734863281], [469.184814453125]], device="cuda")
            norm = centroids.square().sum(1)
            ieee = norm-2*(rows*centroids).sum(1)
            # Deliberately truncate the low 13 mantissa bits to exhibit why
            # this named fixture needs IEEE FP32 rather than TF32 ranking.
            rounded_rows = (rows.view(torch.int32) & -8192).view(torch.float32)
            rounded_centroids = (centroids.view(torch.int32) & -8192).view(torch.float32)
            tf32 = norm-2*(rounded_rows*rounded_centroids).sum(1)
            self.assertEqual(int(ieee.argmin()), 1)
            self.assertEqual(int(tf32.argmin()), 0)
            _, labels = self._known_centroids(centroids, assignment_backend="triton").assign(rows)
            self.assertEqual(int(labels[0]), 1)

    @unittest.skipUnless(TRITON_CUDA, "Triton NVIDIA SM80 required")
    def test_triton_strided_current_stream_and_tensor_device(self):
        current = torch.cuda.current_device()
        device = torch.device("cuda", (current+1) % torch.cuda.device_count())
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), highest_fp32("cuda"):
                # Pitched rows and transposed/offset centroid views exercise
                # real public layouts, not synthetic enormous address strides.
                storage = torch.zeros((7, 9), dtype=dtype, device=device)
                rows = storage[1:6, 1:7:2]
                centroids = torch.arange(15, device=device, dtype=torch.float32).reshape(3, 5).t()[1:4]
                model = self._known_centroids(centroids, assignment_backend="triton")
                warm = model.assign(rows)
                torch.cuda.synchronize(device)
                del warm
                producer = torch.cuda.Stream(device=device)
                consumer = torch.cuda.Stream(device=device)
                with torch.cuda.stream(producer):
                    # Test-only delay keeps writes pending after the already
                    # warmed assignment is queued on its consumer stream.
                    torch.cuda._sleep(50_000_000)
                    rows.copy_(centroids[[0, 1, 2, 0, 1]].to(dtype))
                consumer.wait_stream(producer)
                with torch.cuda.stream(consumer):
                    saved = rows.clone()
                    distance, labels = model.assign(rows)
                torch.cuda.current_stream(device).wait_stream(consumer)
                direct = (rows.float()[:, None]-centroids.to(dtype).float()[None]).square().sum(2)
                torch.testing.assert_close(labels, direct.argmin(1), rtol=0, atol=0)
                torch.testing.assert_close(distance, direct.min(1).values, rtol=2e-5, atol=2e-4)
                torch.testing.assert_close(rows, saved, rtol=0, atol=0)
                self.assertEqual(distance.device, device)
                self.assertEqual(labels.device, device)
                self.assertEqual(torch.cuda.current_device(), current)
                with self.assertRaisesRegex(ValueError, "device"):
                    wrong = torch.device("cuda", current) if torch.cuda.device_count() >= 2 else torch.device("cpu")
                    model.assign(rows.to(wrong))
                if dtype == torch.float32:
                    # The exported wrapper also accepts FP32 rows with reduced
                    # centroid operands; model.assign's preparation is distinct.
                    from faiss.contrib.e_means.e_means_triton import assign_rows
                    backing = torch.zeros((5, 4), dtype=torch.bfloat16, device=device)
                    reduced = backing[1:4, :3].t()
                    reduced.copy_(centroids.bfloat16())
                    saved_centroids = reduced.clone()
                    norm = reduced.float().square().sum(1).contiguous()
                    distance, labels = assign_rows(rows, reduced, norm)
                    direct = (rows[:, None]-reduced.float()[None]).square().sum(2)
                    torch.testing.assert_close(labels, direct.argmin(1), rtol=0, atol=0)
                    torch.testing.assert_close(distance, direct.min(1).values, rtol=2e-5, atol=2e-4)
                    torch.testing.assert_close(reduced, saved_centroids, rtol=0, atol=0)
                    self.assertEqual((distance.device, labels.device), (device, device))
                    self.assertEqual(torch.cuda.current_device(), current)

    @unittest.skipUnless(TRITON_CUDA, "Triton NVIDIA SM80 required")
    def test_triton_exact_half_lattice_statistics(self):
        # Every partial sum is representable: per-slot absolute 2**-24
        # lattice units are 2**23 and 8, both <=2**24, in any atomic order.
        rows = torch.tensor([[1/16 if i % 2 else -1/16, 2**-24] for i in range(8)],
                            dtype=torch.float16, device="cuda")
        saved = rows.clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            for mode in (False, True):
                torch.use_deterministic_algorithms(mode)
                model = Emeans(2, 1, niter=1, batch_size=8, alpha0=1,
                               assignment_backend="triton", capture_assignments=True)
                model.begin(8, torch.zeros((1, 2), device="cuda"))
                partial = model.accumulate(rows[:3])
                self.assertEqual(partial["n"], 3)
                self.assertIsNone(partial["sums"])
                self.assertIsNone(partial["counts"])
                stats = model.accumulate(rows[3:])
                torch.testing.assert_close(stats["labels"], torch.zeros(5, dtype=torch.int64, device="cuda"), rtol=0, atol=0)
                torch.testing.assert_close(stats["sums"], torch.tensor([[0., 2**-21]], device="cuda"), rtol=0, atol=0)
                torch.testing.assert_close(stats["counts"], torch.tensor([8.], device="cuda"), rtol=0, atol=0)
                self.assertEqual((stats["sums"].dtype, stats["counts"].dtype, stats["objective"].dtype),
                                 (torch.float32, torch.float32, torch.float64))
                model.update(stats)
                model.end_epoch()
                torch.testing.assert_close(model.finalize(), torch.tensor([[0., 2**-24]], device="cuda"), rtol=0, atol=0)
                torch.testing.assert_close(rows, saved, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    @unittest.skipUnless(TRITON_CUDA, "Triton NVIDIA SM80 required")
    def test_triton_external_statistics_rounding_and_revival(self):
        objective = torch.zeros((), device="cuda", dtype=torch.float64)
        for alpha, zero_mass, value in (
            (0.5, False, 3.),
            (1., False, 1e8),
            (0.5, True, 1e8),
        ):
            for strided in (False, True):
                with self.subTest(alpha=alpha, zero_mass=zero_mass, strided=strided):
                    initial = torch.tensor([[value, value], [8.25, 8.25]], device="cuda")
                    models = [Emeans(2, 2, niter=2, batch_size=2, alpha0=alpha,
                                     assignment_backend=backend) for backend in ("native", "triton")]
                    backing = torch.zeros((2, 4 if strided else 2), device="cuda")
                    sums = backing[:, ::2] if strided else backing
                    sums[0] = 2
                    counts = torch.tensor([2., 0.], device="cuda")
                    saved = sums.clone()
                    for model in models:
                        model.begin(2, initial)
                        if zero_mass:
                            model._usage[0] = 0  # Isolate documented zero-mass reentry.
                        model.update(dict(sums=sums, counts=counts, objective=objective, n=2))
                    torch.testing.assert_close(models[0]._centroids, models[1]._centroids, rtol=0, atol=0)
                    torch.testing.assert_close(models[0]._usage, models[1]._usage, rtol=0, atol=0)
                    mass = 0 if zero_mass else 2**-16
                    weight = alpha*2/((1-alpha)*mass+alpha*2)
                    expected = 1 if weight == 1 else value+weight*(1-value)
                    torch.testing.assert_close(models[1]._centroids[0], torch.full((2,), expected, device="cuda", dtype=torch.float32),
                                               rtol=2e-7, atol=0)
                    torch.testing.assert_close(sums, saved, rtol=0, atol=0)
                    for model in models:
                        model.reset()
        models = [Emeans(1, 2, niter=1, batch_size=1, alpha0=0.9,
                         assignment_backend=backend) for backend in ("native", "triton")]
        for model in models:
            model.begin(400, torch.tensor([[0.], [8.25]], device="cuda"))
        mass, position = 2**-16, 0.
        saw_subnormal = False
        # Revival has separate actual controls; this probe isolates RN decay,
        # including subnormals. Identical RN operations on identical external
        # inputs justify equality here, not on arbitrary assignment trajectories.
        with mock.patch.object(models[0], "_revive_unused", return_value=0), \
             mock.patch.object(models[1], "_revive_unused", return_value=0):
            for step in range(400):
                value = (step % 7-3)/8
                alpha = e_means._scheduled_alpha(0.9, step, 400)
                mass = (1-alpha)*mass+alpha
                position += alpha/mass*(value-position)
                stats = dict(sums=torch.tensor([[value], [0.]], device="cuda"),
                             counts=torch.tensor([1., 0.], device="cuda"), objective=objective, n=1)
                for model in models:
                    model.update(stats)
                torch.testing.assert_close(models[0]._centroids, models[1]._centroids, rtol=0, atol=0)
                torch.testing.assert_close(models[0]._usage, models[1]._usage, rtol=0, atol=0)
                np.testing.assert_allclose(float(models[1]._centroids[0, 0]), position, rtol=3e-6, atol=3e-7)
                saw_subnormal |= 0 < float(models[1]._usage[1]) < torch.finfo(torch.float32).tiny
        self.assertTrue(saw_subnormal)
        self.assertLessEqual(float(models[1]._usage[1]), torch.finfo(torch.float32).tiny)
        for model in models:
            model.end_epoch()
            model.finalize()
        # Same bounded donor/seed/represented arithmetic on one device; this
        # compares the real split writer, not an arbitrary floating trajectory.
        for dtype in (None, torch.bfloat16):
            with self.subTest(writer_dtype=dtype):
                models = [Emeans(1, 3, niter=1, batch_size=10, alpha0=0.5, seed=7,
                                 assignment_backend=backend) for backend in ("native", "triton")]
                for model in models:
                    model.begin(50, torch.tensor([[0.], [10.], [30.]], device="cuda"))
                    if dtype is not None:
                        model.accumulate(torch.empty((0, 1), device="cuda", dtype=dtype))
                    for _ in range(5):
                        model.update(dict(sums=torch.tensor([[0.], [0.], [300.]], device="cuda"),
                                          counts=torch.tensor([0., 0., 10.], device="cuda"), objective=objective, n=10))
                    self.assertGreater(model.n_splits, 0)
                torch.testing.assert_close(models[0]._centroids, models[1]._centroids, rtol=0, atol=0)
                torch.testing.assert_close(models[0]._usage, models[1]._usage, rtol=0, atol=0)
                for model in models:
                    model.end_epoch()
                    model.finalize()

    @unittest.skipUnless(TRITON_CUDA, "Triton NVIDIA SM80 required")
    def test_triton_warmed_streaming_workspace(self):
        for k, seed in ((4097, 321), (16384, 654)):
            with self.subTest(k=k):
                generator = torch.Generator(device="cuda").manual_seed(seed)
                rows = torch.randn(512, 17, generator=generator, device="cuda").bfloat16()
                centroids = torch.randn(k, 17, generator=generator, device="cuda")
                model = self._known_centroids(centroids, assignment_backend="triton")
                warm = model.assign(rows)  # Inputs, conversion and compile outside measurement.
                torch.cuda.synchronize()
                del warm
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                distance, labels = model.assign(rows)
                torch.cuda.synchronize()
                extra = torch.cuda.max_memory_allocated()-before
                self.assertEqual(distance.shape, (512,))
                self.assertEqual(labels.shape, (512,))
                self.assertLess(extra, 512*k*4//4)

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() >= (8, 0)
        and importlib.util.find_spec("triton") is not None, "Triton CUDA required"
    )
    def test_triton_matches_represented_training(self):
        initial = torch.tensor([[0.], [3.]], device="cuda")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[0.125], [0.25], [3.125], [3.25], [3.5]],
                                 dtype=dtype, device="cuda")
                args = dict(niter=2, batch_size=2, alpha0=0.5, seed=7)
                native = Emeans(1, 2, **args)
                tiled = Emeans(1, 2, assignment_backend="triton", **args)
                native.train(x, initial)
                tiled.train(x, initial)
                torch.testing.assert_close(tiled.centroids, native.centroids, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(tiled.obj, native.obj, atol=1e-5, rtol=2e-6)
                torch.testing.assert_close(tiled.assign(x)[1], native.assign(x)[1])
                self.assertEqual(tiled.recipe["assignment_backend"], "triton")

    def test_stateless_cast_and_cpu_scope(self):
        for bad_dimension, bad_dtype in ((0, torch.float16), (1, torch.float32)):
            with self.assertRaises(ValueError):
                CastQuantizer(bad_dimension, bad_dtype)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = CastQuantizer(2, dtype)
                q.train()
                self.assertTrue(q.is_trained)
                self.assertEqual(q.code_size, 4)
                self.assertIs(q.to("cpu"), q)
                x = torch.tensor([[1.001, 9., 3.001, 9.],
                                  [2.001, 9., 4.001, 9.]], requires_grad=True)[:, ::2]
                saved = x.detach().clone()
                codes = q.compute_codes(x)
                self.assertTrue(codes.is_contiguous())
                self.assertFalse(codes.requires_grad)
                torch.testing.assert_close(q.decode(codes), x.to(dtype).float(), rtol=0, atol=0)
                torch.testing.assert_close(x, saved, rtol=0, atol=0)
                q.validate(x)
                for bad in (x[0], x[:, :1], x.detach().to_sparse()):
                    with self.assertRaises(ValueError):
                        q.compute_codes(bad)
                with self.assertRaises(TypeError):
                    q.compute_codes(x.to(torch.int32))
                with self.assertRaises(TypeError):
                    q.decode(x)
                with self.assertRaises(ValueError):
                    q.to("meta")
                for value in (float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        q.validate(torch.full((1, 2), value))
                for rows in (x, saved.numpy()):
                    with self.assertRaisesRegex(ValueError, "quantizer=None"):
                        Emeans(2, 1, quantizer=q).train(rows)
                with self.assertRaises(ValueError):
                    Emeans(3, 1, quantizer=q)
                # Native CPU low-precision acceptance begins at P03.
                model = Emeans(2, 1, niter=1, batch_size=2, alpha0=0.5)
                model.train(codes)
                torch.testing.assert_close(model.centroids, codes.float().mean(0, keepdim=True))
                self.assertEqual(model.centroids.dtype, torch.float32)
                model.begin(2, codes[:1].float())
                partial = model.accumulate(codes[:1])
                self.assertIsNone(partial["sums"])
                self.assertIsNone(partial["counts"])
                with self.assertRaisesRegex(ValueError, "complete the pending"):
                    model.update({})
                stats = model.accumulate(codes[1:])
                model.update(stats)
                model.end_epoch()
                torch.testing.assert_close(model.finalize(), codes.float().mean(0, keepdim=True))
        half = CastQuantizer(1, torch.float16)
        half.validate(torch.tensor([[65504.]]))
        with self.assertRaises(ValueError):
            half.validate(torch.tensor([[65505.]]))
        # Unchecked conversion follows Torch; it is not a validating API.
        torch.testing.assert_close(half.compute_codes(torch.tensor([[65520.]])),
                                   torch.tensor([[65520.]]).half(), rtol=0, atol=0)

    def _known_centroids(self, centroids, **kwargs):
        return known_torch_centroids(centroids, **kwargs)

    @unittest.skipUnless(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
                         and torch.cuda.get_device_capability() >= (8, 0), "NVIDIA SM80 required")
    def test_represented_assignment_norms_ties_and_high_finite(self):
        for backend in self._floating_backends():
            for dtype in (torch.float16, torch.bfloat16):
                for n, k, d in ((17, 7, 3), (31, 13, 65)):
                    with self.subTest(backend=backend, dtype=dtype, shape=(n, k, d)):
                        rng = torch.Generator(device="cuda").manual_seed(123+d)
                        rows = torch.randn(n, d, generator=rng, device="cuda")/4
                        centroids = torch.randn(k, d, generator=rng, device="cuda")/4
                        reference = (
                            rows.to(dtype).float()[:, None]
                            - centroids.to(dtype).float()[None]
                        ).square().sum(2)
                        expected_distance, expected_labels = reference.min(1)
                        for quantizer in (None, CastQuantizer(d, dtype)):
                            model = self._known_centroids(
                                centroids,
                                assignment_backend=backend,
                                quantizer=quantizer,
                                assignment_chunk_size=5,
                                centroid_chunk_size=2,
                            )
                            distance, labels = model.assign(rows.to(dtype) if quantizer is None else rows)
                            torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
                            torch.testing.assert_close(distance, expected_distance, atol=2e-4, rtol=2e-5)
                            self.assertEqual((distance.dtype, labels.dtype), (torch.float32, torch.int64))
                small = (1.0001, 1.0004) if dtype == torch.float16 else (1.001, 1.002)
                centroids = torch.tensor([[small[0], 0.], [small[1], 0.], [3., 0.]], device="cuda")
                rows = centroids[1:2].clone()
                self.assertEqual(int(torch.cdist(rows, centroids).argmin(1)), 1)
                torch.testing.assert_close(centroids[0].to(dtype), centroids[1].to(dtype), rtol=0, atol=0)
                model = self._known_centroids(centroids, assignment_backend=backend, centroid_chunk_size=1)
                distance, labels = model.assign(rows.to(dtype))
                self.assertEqual(int(labels[0]), 0)
                self.assertEqual(float(distance[0]), 0)
            rows = torch.tensor([[-0.4678138, 0.03789843, -0.6960376]], device="cuda")
            centroids = torch.tensor([[-0.8843690, 1.0697185, -0.7981653],
                                      [0.4984003, 0.5734861, -0.5365824]], device="cuda")
            represented = (
                rows.bfloat16().float()[:, None]
                - centroids.bfloat16().float()[None]
            ).square().sum(2)
            wrong_norm = centroids.square().sum(1) - 2 * (
                rows.bfloat16().float()[:, None]
                * centroids.bfloat16().float()[None]
            ).sum(2)
            self.assertEqual(int(represented.argmin(1)), 0)
            self.assertEqual(int(wrong_norm.argmin(1)), 1)
            model = self._known_centroids(centroids, assignment_backend=backend, centroid_chunk_size=1)
            distance, labels = model.assign(rows.bfloat16())
            self.assertEqual(int(labels[0]), 0)
            torch.testing.assert_close(distance, represented[:, 0], atol=1e-6, rtol=0)
            centroids = torch.tensor([[300.], [302.]], device="cuda")
            rows = torch.tensor([[300.], [301.]], device="cuda", dtype=torch.float16)
            self.assertGreater(float(centroids[0].square().sum()), torch.finfo(torch.float16).max)
            distance, labels = self._known_centroids(centroids, assignment_backend=backend, centroid_chunk_size=1).assign(rows)
            torch.testing.assert_close(labels, torch.zeros(2, dtype=torch.int64, device="cuda"), rtol=0, atol=0)
            torch.testing.assert_close(distance, torch.tensor([0., 1.], device="cuda"), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
                         and torch.cuda.get_device_capability() >= (8, 0), "NVIDIA SM80 required")
    def test_represented_empty_metadata_split_and_range(self):
        for backend in self._floating_backends():
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(backend=backend, dtype=dtype):
                    model = Emeans(1, 2, assignment_backend=backend, niter=1, batch_size=2, alpha0=0.5, capture_assignments=True)
                    model.begin(4, torch.tensor([[0.], [10.]], device="cuda"))
                    empty = model.accumulate(torch.empty((0, 1), dtype=dtype, device="cuda"))
                    self.assertEqual(empty["n"], 0)
                    with self.assertRaisesRegex(ValueError, "dtype must be consistent"):
                        model.accumulate(torch.empty((0, 1), device="cuda"))
                    model.update(dict(sums=torch.tensor([[2.], [0.]], device="cuda"),
                                      counts=torch.tensor([2., 0.], device="cuda"),
                                      objective=torch.zeros((), dtype=torch.float64, device="cuda"), n=2))
                    rows = torch.ones((2, 1), dtype=dtype, device="cuda")
                    before = model.current_centroids().clone()
                    partial = model.accumulate(rows[:1])
                    self.assertEqual(partial["n"], 1)
                    self.assertIsNone(partial["sums"])
                    self.assertIsNone(partial["counts"])
                    with self.assertRaisesRegex(ValueError, "dtype must be consistent"):
                        model.accumulate(rows[1:].float())
                    torch.testing.assert_close(model.current_centroids(), before, rtol=0, atol=0)
                    stats = model.accumulate(rows[1:])
                    torch.testing.assert_close(stats["labels"], torch.zeros(1, dtype=torch.int64, device="cuda"), rtol=0, atol=0)
                    self.assertEqual(stats["sums"].dtype, torch.float32)
                    self.assertEqual(stats["counts"].dtype, torch.float32)
                    model.update(stats)
                    model.end_epoch()
                    torch.testing.assert_close(model.finalize(), torch.tensor([[1.], [10.]], device="cuda"), rtol=0, atol=0)
                    maximum = torch.finfo(dtype).max
                    donor = torch.tensor([[0., 0.], [1., 1.], [1., 2.], [maximum, 0.]], device="cuda")
                    child = torch.tensor([[0., 0.], [1.0001, 0.9999], [2., 1.], [maximum, 0.]], device="cuda")
                    ordinary = child[2].clone()
                    previous = torch.cuda.get_sync_debug_mode()
                    try:
                        torch.cuda.set_sync_debug_mode("error")
                        e_means._separate_splits(donor, child, dtype)
                    finally:
                        torch.cuda.set_sync_debug_mode(previous)
                    self.assertFalse(bool((donor.to(dtype) == child.to(dtype)).all(1).any()))
                    torch.testing.assert_close(child[2], ordinary, rtol=0, atol=0)
                    self.assertLess(float(child[3, 0]), maximum)
                    torch.testing.assert_close(child[:2, 1], torch.tensor([0., 0.9999], device="cuda"), rtol=0, atol=0)
            for dtype, value in ((torch.float16, 65504.), (torch.bfloat16, 70000.)):
                rows = torch.full((2, 1), value, device="cuda", dtype=dtype)
                model = Emeans(1, 1, assignment_backend=backend, niter=1, batch_size=2, alpha0=0.5)
                model.train(rows, rows[:1].float())
                torch.testing.assert_close(model.centroids, rows[:1].float(), rtol=0, atol=0)
                torch.testing.assert_close(model.assign(rows)[0], torch.zeros(2, device="cuda"), rtol=0, atol=0)
            model = Emeans(1, 2, assignment_backend=backend, niter=1, batch_size=2, alpha0=0.5)
            rows = torch.full((10, 1), 65504., device="cuda", dtype=torch.float16)
            before_split = []
            apply_splits = model._apply_splits

            def checked_split(
                *args,
                _model=model,
                _apply_splits=apply_splits,
                _before_split=before_split,
            ):
                centroids, usage = _model._centroids.clone(), _model._usage.clone()
                try:
                    return _apply_splits(*args)
                except ValueError:
                    _before_split.append(True)
                    torch.testing.assert_close(_model._centroids, centroids, rtol=0, atol=0)
                    torch.testing.assert_close(_model._usage, usage, rtol=0, atol=0)
                    raise

            with mock.patch.object(model, "_apply_splits", checked_split), self.assertRaisesRegex(ValueError, "65504"):
                model.train(rows, rows[:2].float())
            self.assertTrue(before_split)
            self.assertEqual(model.n_splits, 0)
            rows = torch.zeros((2, 2), device="cuda", dtype=torch.bfloat16)
            with mock.patch.object(torch.version, "hip", "test"), self.assertRaisesRegex(RuntimeError, "SM80"):
                Emeans(2, 2, assignment_backend=backend).train(rows)
            with mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), self.assertRaisesRegex(RuntimeError, "SM80"):
                Emeans(2, 2, assignment_backend=backend).train(rows)


    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() >= (8, 0), "NVIDIA SM80 required"
    )
    def test_native_and_cast_represented_means(self):
        x = torch.tensor([[1.001], [1.002], [3.001], [3.002]], device="cuda")
        mean_initial = torch.tensor([[1.0], [3.0]], device="cuda")
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = CastQuantizer(1, dtype)
                codes = q.compute_codes(x)
                native = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)
                cast = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5, quantizer=q)
                native.train(codes, mean_initial)
                cast.train(x, mean_initial)
                expected = codes.float().reshape(2, 2, 1).mean(dim=1)
                torch.testing.assert_close(native.centroids, expected, atol=0, rtol=0)
                torch.testing.assert_close(cast.centroids, expected, atol=0, rtol=0)
                torch.testing.assert_close(native.obj, cast.obj, atol=0, rtol=0)
                dn, ln = native.assign(codes)
                dc, lc = cast.assign(x)
                torch.testing.assert_close(dn, dc, atol=0, rtol=0)
                torch.testing.assert_close(ln, lc)
                self.assertEqual((native.centroids.dtype, dn.dtype),
                                 (torch.float32, torch.float32))
                # Retained decimal T2/N17/B4 fixture includes the final row.
                rows = torch.arange(34, device="cuda", dtype=torch.float32).reshape(17, 2)/10
                represented = CastQuantizer(2, dtype).compute_codes(rows)
                initial = represented[[0, 5, 10, 16]].float()
                saved_rows, saved_initial = rows.clone(), initial.clone()
                args = dict(niter=2, batch_size=4, alpha0=0.5, seed=7)
                reference = Emeans(2, 4, assignment_chunk_size=2, centroid_chunk_size=2, **args)
                reference.train(represented, initial)
                q = CastQuantizer(2, dtype)
                cast = Emeans(2, 4, quantizer=q, assignment_chunk_size=2, centroid_chunk_size=2, **args)
                cast.train(rows, initial)
                unblocked = Emeans(2, 4, assignment_chunk_size=64, centroid_chunk_size=16, **args)
                unblocked.train(represented, initial)
                caller = Emeans(2, 4, quantizer=q, assignment_chunk_size=2, centroid_chunk_size=2, **args)
                caller_initial = initial.clone()
                caller.begin(len(rows), caller_initial)
                caller_initial.fill_(-7)
                current = caller.current_centroids()
                self.assertEqual((current.dtype, current.device), (torch.float32, initial.device))
                torch.testing.assert_close(current, initial, rtol=0, atol=0)
                for epoch in range(2):
                    generator = torch.Generator(device="cuda").manual_seed(_epoch_seed(7, epoch))
                    order = torch.randperm(len(rows), generator=generator, device="cuda")
                    for begin in range(0, len(rows), 4):
                        batch = rows[order[begin:begin+4]]
                        before = caller.current_centroids().clone()
                        for start in range(0, len(batch), 2):
                            stats = caller.accumulate(batch[start:start+2])
                            if start + 2 < len(batch):
                                self.assertIsNone(stats["sums"])
                                self.assertIsNone(stats["counts"])
                                torch.testing.assert_close(caller.current_centroids(), before, rtol=0, atol=0)
                        self.assertEqual(stats["sums"].dtype, torch.float32)
                        self.assertEqual(stats["counts"].dtype, torch.float32)
                        self.assertEqual(stats["objective"].dtype, torch.float64)
                        caller.update(stats)
                    caller.end_epoch()
                result = caller.finalize()
                for other in (cast, unblocked, caller):
                    torch.testing.assert_close(other.centroids, reference.centroids, atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(other.obj, reference.obj, atol=3e-5, rtol=1e-6)
                    for field in ("nsplit", "imbalance_factor"):
                        self.assertEqual([s[field] for s in other.iteration_stats],
                                         [s[field] for s in reference.iteration_stats])
                torch.testing.assert_close(cast.assign(rows)[0], reference.assign(represented)[0], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
                torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)
                saved_result = result.clone()
                caller.reset()
                caller.train(rows, initial)
                torch.testing.assert_close(caller.centroids, saved_result, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(result, saved_result, rtol=0, atol=0)

    def test_local_update_has_no_group_argument(self):
        model = Emeans(1, 1, niter=1, batch_size=2, alpha0=0.5)
        model.begin(4, torch.zeros((1, 1)))
        stats = model.accumulate(torch.zeros((2, 1)))
        with self.assertRaises(TypeError):
            model.update(stats, group=object())
        model.update(stats)
        stats = model.accumulate(torch.ones((2, 1)))
        with self.assertRaises(TypeError):
            model.update(stats, group=object())
        model.update(stats)
        model.end_epoch()
        torch.testing.assert_close(model.finalize(), torch.tensor([[0.5]]))


    def reference_arrays(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        return [(device, lambda values, dtype="float32", device=device:
                 torch.tensor(values, dtype=getattr(torch, dtype), device=device))
                for device in devices]

    def test_fractional_native_and_numpy_training(self):
        rng = np.random.RandomState(123)
        centers = (10*rng.randn(4, 6)).astype("float32")
        rows = centers[np.arange(80) % 4] + (0.1*rng.randn(80, 6)).astype("float32")
        args = dict(niter=3, batch_size=len(rows), alpha0=0.2, seed=7)
        reference = Emeans(6, 4, **args)
        reference.train(rows, centers)
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                model = Emeans(6, 4, **args)
                model.train(array(rows), array(centers))
                np.testing.assert_allclose(model.centroids.cpu().numpy(), reference.centroids,
                                           atol=2e-5, rtol=1e-6)
                np.testing.assert_allclose(model.obj.cpu().numpy(), reference.obj,
                                           atol=3e-3, rtol=1e-3)
                self.assertEqual(model.n_splits, reference.n_splits)

    def test_cpu_rejects_nonfinite_input(self):
        model = Emeans(1, 2, niter=1, batch_size=2, alpha0=0.5)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.train(torch.full((2, 1), value))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_local_updates_do_not_synchronize(self):
        model = Emeans(1, 2, niter=1, batch_size=2, alpha0=0.5)
        model.begin(4, torch.tensor([[0.0], [10.0]], device="cuda"))
        rows = torch.tensor([[0.0], [1.0]], device="cuda")
        donor = torch.zeros((2, 1), device="cuda")
        child = donor.clone()
        model.update(model.accumulate(rows))  # Preparation outside sync-debug.
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            model.update(model.accumulate(rows))
            self.assertEqual(model.current_centroids().device, rows.device)
            # Split-separation has no standalone public operation; this
            # bounded private probe checks its no-host-sync repair path.
            e_means._separate_splits(donor, child)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        model.end_epoch()
        torch.testing.assert_close(child, torch.nextafter(donor, torch.full_like(donor, float("inf"))),
                                   rtol=0, atol=0)
        torch.testing.assert_close(model.finalize(), torch.tensor([[0.5], [10.0]], device="cuda"),
                                   rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_noncurrent_device_and_mismatch_recovery(self):
        current = torch.cuda.current_device()
        device = torch.device("cuda", (current+1) % torch.cuda.device_count())
        rows = torch.tensor([[0.0], [2.0], [10.0], [12.0]], device=device)
        initial = rows[[0, 2]].clone()
        model = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)
        model.begin(4, initial)
        self.assertEqual(model.current_centroids().device, device)
        self.assertFalse(model.current_centroids().requires_grad)
        with self.assertRaises(ValueError):
            model.accumulate(rows.to(torch.device("cuda", current)))
        model.update(model.accumulate(rows))
        model.end_epoch()
        result = model.finalize()
        self.assertEqual(result.device, device)
        self.assertEqual(torch.cuda.current_device(), current)
        torch.testing.assert_close(result, torch.tensor([[1.0], [11.0]], device=device), rtol=0, atol=0)

    def test_large_cpu_batch_preserves_sum_and_count(self):
        n = 2**24 + 2
        x = torch.full((n, 1), 0.25)
        km = Emeans(1, 1, niter=1, batch_size=n, alpha0=1.0)
        km.begin(n, torch.zeros((1, 1)))
        partial = km.accumulate(x[:2**24])
        self.assertEqual(partial["n"], 2**24)
        self.assertIsNone(partial["sums"])
        self.assertIsNone(partial["counts"])
        stats = km.accumulate(x[2**24:])
        self.assertEqual(stats["sums"].dtype, torch.float32)
        self.assertEqual(stats["counts"].dtype, torch.float32)
        self.assertEqual(stats["sums"][0, 0].item(), n * 0.25)
        self.assertEqual(stats["counts"][0].item(), n)
        km.update(stats)
        km.end_epoch()
        self.assertEqual(km.finalize()[0, 0].item(), 0.25)

    def test_caller_batches_match_matrix_training(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for overrides in (
                dict(batch_size=4, alpha0=0.2,
                     assignment_chunk_size=2, centroid_chunk_size=2),
                {},
            ):
                with self.subTest(device=device, overrides=overrides):
                    # Binary values keep this exact fixture's FP32 sums independent
                    # of CUDA atomic addition order.
                    x = torch.arange(51, device=device, dtype=torch.float32)
                    x = x.reshape(17, 3) / 8
                    init = x[[0, 5, 10, 16]].clone()
                    args = dict(niter=3, seed=7, **overrides)
                    matrix, caller = Emeans(3, 4, **args), Emeans(3, 4, **args)
                    matrix.train(x, init_centroids=init)
                    caller.begin(len(x), init)
                    batch_size = caller.recipe["batch_size"]
                    for epoch in range(caller.niter):
                        generator = torch.Generator(device=device).manual_seed(
                            _epoch_seed(caller.seed, epoch)
                        )
                        order = torch.randperm(
                            len(x), generator=generator, device=device
                        )
                        for begin in range(0, len(x), batch_size):
                            stats = caller.accumulate(x[order[begin:begin + batch_size]])
                            caller.update(stats)
                        caller.end_epoch()
                    torch.testing.assert_close(
                        caller.finalize(), matrix.centroids, rtol=0, atol=0
                    )
                    torch.testing.assert_close(caller.obj, matrix.obj, rtol=0, atol=0)
                    self.assertEqual(caller.n_splits, matrix.n_splits)

    def test_closing_mean_and_assign(self):
        x = torch.tensor(
            [[0.0], [2.0], [10.0], [12.0]], requires_grad=True
        )
        init = torch.tensor([[0.0], [10.0]], requires_grad=True)
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)

        centroids, captured_labels = km.train(x, init_centroids=init)

        torch.testing.assert_close(km.centroids, torch.tensor([[1.0], [11.0]]))
        self.assertEqual(km.obj[-1], 8.0)
        self.assertIs(centroids, km.centroids)
        self.assertIs(captured_labels, km.assignments)
        self.assertFalse(km.centroids.requires_grad)
        km.centroids.requires_grad_()
        distance, labels = km.assign(x)
        torch.testing.assert_close(distance, torch.ones(4))
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1]))
        self.assertFalse(distance.requires_grad)

    def test_alpha_one_preserves_an_empty_cluster(self):
        x = torch.tensor([[99.0], [0.0], [1.0], [100.0]])
        init = torch.tensor([[0.0], [100.0]])
        km = Emeans(1, 2, niter=1, batch_size=2, alpha0=1.0)

        km.train(x, init_centroids=init)

        torch.testing.assert_close(
            km.centroids, torch.tensor([[0.5], [99.5]])
        )

    def test_closing_mean_uses_only_the_final_pass(self):
        x = torch.tensor([[3.0], [4.0], [4.0], [6.0], [12.0]])
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=2, batch_size=5, alpha0=1.0)

        km.train(x, init_centroids=init)

        torch.testing.assert_close(
            km.centroids, torch.tensor([[4.25], [12.0]])
        )

        # A private FP64 closing-scan table must be divided before narrowing.
        # Use the same non-power-of-two integer ratio for both FP64 and INT64
        # inputs: early FP32 conversion changes the correctly rounded result.
        n, total = 1048577, 266199830
        expected = np.float32(np.float64(total) / np.float64(n))
        early = np.float32(np.float32(total) / np.float32(n))
        self.assertNotEqual(expected, early)
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for dtype in (torch.float64, torch.int64):
                with self.subTest(closing_dtype=dtype, device=device):
                    model = Emeans(1, 2, niter=1, batch_size=n, alpha0=0.5)
                    initial = torch.tensor([[254.], [9.]], device=device)
                    model.begin(n, initial)
                    model.update({
                        "sums": torch.tensor([[total], [0.]], dtype=torch.float32,
                                             device=device),
                        "counts": torch.tensor([n, 0.], dtype=torch.float32,
                                               device=device),
                        "objective": torch.zeros((), dtype=torch.float64, device=device),
                        "n": n,
                    })
                    model.end_epoch()
                    retained = model.current_centroids()[1].clone()
                    # The optional sharded scan supplies this private table;
                    # no new public FP64 statistics schema is being exercised.
                    model._pass_sums = torch.tensor([[total], [0]], dtype=dtype,
                                                    device=device)
                    closed = model.finalize()
                    self.assertEqual(closed.dtype, torch.float32)
                    self.assertEqual(closed.device, initial.device)
                    self.assertEqual(closed[0, 0].item(), float(expected))
                    torch.testing.assert_close(closed[1], retained, rtol=0, atol=0)
                    self.assertIs(model.finalize(), closed)
                    self.assertIsNone(model._pass_sums)
                    with self.assertRaises(RuntimeError):
                        model.current_centroids()

    def test_blocked_assignment_matches_cdist(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(23, 7, generator=generator)
        centroids = torch.randn(11, 7, generator=generator)
        km = Emeans(7, 11, niter=1, batch_size=11, alpha0=0.5,
                    centroid_chunk_size=3)
        km.train(centroids, init_centroids=centroids)
        self.assertEqual(km.recipe["centroid_chunk_size"], 3)

        distance, labels = km.assign(x)
        reference = torch.cdist(x, centroids).square()
        expected_distance, expected_labels = reference.min(dim=1)

        torch.testing.assert_close(
            distance, expected_distance, atol=2e-5, rtol=1e-5
        )
        torch.testing.assert_close(labels, expected_labels)

    def test_blocked_assignment_breaks_ties_by_index(self):
        x = torch.zeros((1, 2))
        centroids = torch.ones((5, 2))
        km = Emeans(2, 5, niter=1, batch_size=5, alpha0=0.5,
                    centroid_chunk_size=2)
        km.train(centroids, init_centroids=centroids)

        _, labels = km.assign(x)

        self.assertEqual(labels.item(), 0)

    def test_revival_occurs_during_training(self):
        x = torch.zeros((10, 1))
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=5, batch_size=10, alpha0=0.5)

        km.train(x, init_centroids=init)

        self.assertEqual(km.n_splits, 1)

    def test_chunking_does_not_change_updates(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(41, 5, generator=generator)
        init = x[:4].clone()
        args = {"niter": 3, "batch_size": 13, "alpha0": 0.2, "seed": 7}
        a = Emeans(
            5,
            4,
            assignment_chunk_size=4,
            centroid_chunk_size=2,
            **args,
        )
        b = Emeans(
            5,
            4,
            assignment_chunk_size=64,
            centroid_chunk_size=8,
            **args,
        )

        a.train(x, init_centroids=init)
        b.train(x, init_centroids=init)

        torch.testing.assert_close(
            a.centroids, b.centroids, atol=2e-6, rtol=1e-6
        )
        torch.testing.assert_close(a.obj, b.obj, atol=2e-5, rtol=1e-6)

    def test_training_does_not_advance_global_rng(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            with self.subTest(device=device):
                x = torch.arange(80, dtype=torch.float32, device=device).reshape(20, 4)
                cpu_state = torch.random.get_rng_state()
                state = torch.cuda.get_rng_state() if device == "cuda" else cpu_state
                km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)
                km.train(x)
                after = torch.cuda.get_rng_state() if device == "cuda" else torch.random.get_rng_state()
                self.assertTrue(torch.equal(state, after))
                self.assertTrue(torch.equal(cpu_state, torch.random.get_rng_state()))

    def test_input_types_match(self):
        x = torch.randn(20, 4)
        init = x[:3].clone()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        with self.assertRaises(TypeError):
            km.train(x.numpy(), init_centroids=init)
        with self.assertRaises(TypeError):
            km.train(x, init_centroids=init.numpy())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_replay_failure_drains_and_model_recovers(self):
        cases = [(torch.float32, None)]
        if not torch.version.hip and torch.cuda.get_device_capability() >= (8, 0):
            cases.extend(((torch.bfloat16, None),
                          (torch.float32, CastQuantizer(3, torch.bfloat16))))
        for dtype, quantizer in cases:
            with self.subTest(dtype=dtype, quantizer=quantizer is not None):
                self._check_cuda_replay_failure(dtype, quantizer)
        if "triton" in self._floating_backends():
            self._check_cuda_replay_failure(torch.float32, None, "triton")
            self._check_cuda_replay_failure(torch.bfloat16, None, "triton")
        if FP8_CUDA:
            self._check_cuda_replay_failure(torch.float32, Fp8Quantizer(3), "triton")
        if INT8_CUDA:
            self._check_cuda_replay_failure(torch.uint8, None, "triton")
            self._check_cuda_replay_failure(torch.float32, Int8Quantizer(3), "triton")
        from torch_test_e_means_cute import _ready
        if _ready():
            self._check_cuda_replay_failure(torch.bfloat16, None, "cute")
            self._check_cuda_replay_failure(torch.float32, CastQuantizer(3, torch.bfloat16), "cute")

    def _check_cuda_replay_failure(self, dtype, quantizer, backend="native"):
        x = (torch.arange(93, device="cuda", dtype=torch.float32) / 8).reshape(31, 3)
        initial = x[[0, 5, 10, 16]].clone()
        x = x.to(dtype)
        if isinstance(quantizer, Fp8Quantizer):
            quantizer.train(x)
        if isinstance(quantizer, Int8Quantizer):
            quantizer.train(x)
            x = quantizer.compute_codes(x)
        saved_input, saved_initial = x.clone(), initial.clone()
        model = Emeans(3, 4, niter=3, seed=7, quantizer=quantizer, assignment_backend=backend, capture_assignments=True)
        replay, reset = torch.cuda.CUDAGraph.replay, torch.cuda.CUDAGraph.reset

        class InjectedFailure(BaseException):
            pass

        for error_type in (RuntimeError, InjectedFailure):
            with self.subTest(error_type=error_type.__name__):
                events, retired = [], []

                def fail_after_replay(
                    graph, _events=events, _error_type=error_type
                ):
                    # Torch's test-only delay is needed to prove work remains
                    # queued behind the delay at the host raise.
                    torch.cuda._sleep(50_000_000)
                    replay(graph)
                    done = torch.cuda.Event()
                    done.record()
                    _events.append(done)
                    self.assertFalse(done.query())
                    raise _error_type("injected replay failure")

                def checked_reset(graph, _events=events, _retired=retired):
                    if _events:
                        complete = _events[-1].query()
                        _retired.append(complete)
                        self.assertTrue(complete)  # BEFORE pool retirement.
                    reset(graph)

                with mock.patch.object(torch.cuda.CUDAGraph, "replay", fail_after_replay), \
                     mock.patch.object(torch.cuda.CUDAGraph, "reset", checked_reset):
                    with self.assertRaisesRegex(error_type, "injected replay failure"):
                        model.train(x, initial)
                self.assertEqual(len(events), 1)  # No retry.
                self.assertTrue(retired)
                self.assertIsNone(model.assignments)
                model.reset()
                recovered = []

                def counted_replay(graph):
                    recovered.append(True)  # noqa: B023 - rebound for closing replay
                    replay(graph)

                with mock.patch.object(torch.cuda.CUDAGraph, "replay", counted_replay):
                    model.train(x, initial)
                self.assertTrue(recovered)
                reference = Emeans(3, 4, niter=3, seed=7, assignment_backend=backend, assignment_chunk_size=2,
                                   quantizer=quantizer, capture_assignments=True)
                # The explicit override keeps this reference eager.
                reference.train(x, initial)
                torch.testing.assert_close(model.centroids, reference.centroids, rtol=0, atol=0)
                torch.testing.assert_close(model.obj, reference.obj, rtol=0, atol=0)
                torch.testing.assert_close(model.assignments, reference.assignments, rtol=0, atol=0)
                self.assertEqual(model.n_splits, reference.n_splits)
                torch.testing.assert_close(x, saved_input, rtol=0, atol=0)
                torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)

        # A one-epoch capture already owns closing sums at capture time.
        recovered = []
        single = Emeans(3, 4, niter=1, seed=7, quantizer=quantizer,
                        assignment_backend=backend, capture_assignments=True)
        with mock.patch.object(torch.cuda.CUDAGraph, "replay", counted_replay):
            single.train(x, initial)
        self.assertTrue(recovered)
        eager = Emeans(3, 4, niter=1, seed=7, assignment_chunk_size=2,
                       quantizer=quantizer, assignment_backend=backend, capture_assignments=True)
        eager.train(x, initial)
        torch.testing.assert_close(single.centroids, eager.centroids, rtol=0, atol=0)
        torch.testing.assert_close(single.obj, eager.obj, rtol=0, atol=0)
        torch.testing.assert_close(single.assignments, eager.assignments, rtol=0, atol=0)

    def test_assign_resolves_independently_without_changing_training_recipe(self):
        model = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5)
        rows = torch.tensor([[0.], [2.], [10.], [12.]])
        centroids, captured = model.train(rows, rows[[0, 2]])
        recipe = dict(model.recipe)
        model._input_dtype = torch.bfloat16  # Represents a completed BF16 run.
        with mock.patch.object(e_means, '_get_triton_assign_rows', side_effect=AssertionError("CPU probed Triton")), \
                mock.patch.object(e_means, '_get_cute_functions', side_effect=AssertionError("CPU probed CuTe")):
            distance, labels = model.assign(rows)
        torch.testing.assert_close(distance, torch.ones(4), rtol=0, atol=0)
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1]), rtol=0, atol=0)
        self.assertIs(model.centroids, centroids)
        self.assertIs(model.assignments, captured)
        self.assertEqual(model.recipe, recipe)
        self.assertEqual(model._input_dtype, torch.bfloat16)
        self.assertEqual(model.assignment_recipe['representation'], 'float32')
        self.assertEqual(model.assignment_recipe['assignment_backend'], 'native')

    def test_assign_runtime_failure_is_not_retried(self):
        model = Emeans(1, 1)
        model.centroids = torch.zeros((1, 1))
        with mock.patch.object(model, '_assign_rows', side_effect=RuntimeError('selected runtime fault')) as selected:
            with self.assertRaisesRegex(RuntimeError, 'selected runtime fault'):
                model.assign(torch.zeros((2, 1)))
        self.assertEqual(selected.call_count, 1)
        self.assertIsNone(model.assignment_recipe)

    def test_auto_new_begin_and_typed_empty_fix_run_metadata(self):
        model = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5)
        model._resolved_assignment_backend = 'cute'  # Prior run on a different device.
        model.begin(4, torch.tensor([[0.], [10.]]))
        self.assertIsNone(model._resolved_assignment_backend)
        model.accumulate(torch.empty((0, 1)))
        self.assertEqual(model._assignment_backend, 'native')
        self.assertEqual(model.recipe['assignment_backend'], 'native')
        self.assertEqual(model.recipe['assignment_backend_requested'], 'auto')
        self.assertEqual(model._input_dtype, torch.float32)
        self.assertEqual(model._step, 0)

    def test_auto_failed_train_preflight_preserves_active_run(self):
        rows = torch.tensor([[0.], [2.], [10.], [12.]])
        model = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5)
        model.begin(4, rows[[0, 2]])
        model.accumulate(rows[:1])
        backend, dtype, recipe = model._assignment_backend, model._input_dtype, dict(model.recipe)
        window, accumulation = model._batch_centroids, model._batch_accumulation

        def new_candidate(instance, x):
            instance._resolved_assignment_backend = 'triton'

        with mock.patch.object(Emeans, '_resolve_assignment_backend', new_candidate):
            with self.assertRaisesRegex(ValueError, 'shape'):
                model.train(rows, rows[:1])
        self.assertEqual(model._assignment_backend, backend)
        self.assertEqual(model._input_dtype, dtype)
        self.assertEqual(model.recipe, recipe)
        self.assertIs(model._batch_centroids, window)
        self.assertIs(model._batch_accumulation, accumulation)
        self.assertFalse(model._failed)
        stats = model.accumulate(rows[1:])
        torch.testing.assert_close(stats['counts'], torch.tensor([2., 2.]), rtol=0, atol=0)
        torch.testing.assert_close(stats['sums'], torch.tensor([[2.], [22.]]), rtol=0, atol=0)

    def test_auto_metadata_preference_and_no_encoded_fallback(self):
        from types import ModuleType, SimpleNamespace
        import faiss.contrib.e_means as package

        cute = ModuleType('faiss.contrib.e_means.e_means_cute')
        cute._check_device = mock.Mock()
        cute._check_extent = mock.Mock()
        cute._check_padded_span = mock.Mock()
        cute._padded_extent = lambda value, alignment: (value + alignment - 1) // alignment * alignment
        with mock.patch.object(package, 'e_means_cute', cute, create=True), \
                mock.patch.dict(sys.modules, {cute.__name__: cute}), \
                mock.patch.object(e_means, '_is_torch_tensor', return_value=True), \
                mock.patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
                mock.patch.object(e_means, '_get_triton_assign_rows', return_value=object()) as triton:
            for dtype, quantizer, expected in (
                    (torch.bfloat16, None, 'cute'), (torch.float32, None, 'triton'),
                    (torch.float16, None, 'triton'),
                    (torch.float32, CastQuantizer(1, torch.bfloat16), 'cute')):
                with self.subTest(dtype=dtype, explicit_codec=quantizer is not None):
                    model = Emeans(1, 2, quantizer=quantizer)
                    rows = SimpleNamespace(device=torch.device('cuda:7'), dtype=dtype)
                    model._resolve_assignment_backend(rows)
                    self.assertEqual(model._assignment_backend, expected)
                    self.assertEqual(model.assignment_backend, 'auto')
                    calls = triton.call_count
                    model._resolve_assignment_backend(rows)
                    self.assertEqual(triton.call_count, calls)  # Fixed within the run.
            cute._check_device.side_effect = RuntimeError('metadata incompatibility')
            model = Emeans(1, 2)
            model._resolve_assignment_backend(SimpleNamespace(device=torch.device('cuda:7'), dtype=torch.bfloat16))
            self.assertEqual(model._assignment_backend, 'triton')
            triton.side_effect = ImportError('optional dependency unavailable')
            for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn):
                with self.subTest(encoded=dtype), self.assertRaisesRegex(RuntimeError, 'no compatible backend'):
                    Emeans(1, 2)._resolve_assignment_backend(
                        SimpleNamespace(device=torch.device('cuda:7'), dtype=dtype))

    @unittest.skipUnless((torch.cuda.is_available() and not torch.version.hip
                         and torch.cuda.get_device_capability() >= (8, 0)
                         and importlib.util.find_spec("triton") is not None), "NVIDIA CUDA and Triton required")
    def test_auto_bf16_train_then_fp32_assign(self):
        rows = torch.ones((17, 3), dtype=torch.bfloat16, device='cuda')
        initial = torch.tensor([[1., 1., 1.], [0., 0., 0.]], device='cuda')
        model = Emeans(3, 2, niter=1, batch_size=17, alpha0=.5)
        centroids, captured = model.train(rows, initial)
        train_recipe = dict(model.recipe)
        distance, labels = model.assign(rows.float())
        self.assertEqual(model.assignment_recipe['representation'], 'float32')
        self.assertEqual(model.assignment_recipe['assignment_backend'], 'triton')
        self.assertEqual(model.recipe, train_recipe)
        self.assertIs(model.centroids, centroids)
        self.assertIs(model.assignments, captured)
        torch.testing.assert_close(distance, torch.zeros(17, device='cuda'), rtol=0, atol=0)
        torch.testing.assert_close(labels, torch.zeros(17, dtype=torch.int64, device='cuda'), rtol=0, atol=0)

    @unittest.skipUnless((torch.cuda.is_available() and not torch.version.hip
                         and torch.cuda.get_device_capability() >= (8, 0)
                         and importlib.util.find_spec("triton") is not None), "NVIDIA CUDA and Triton required")
    def test_auto_gpu_failed_train_preflight_preserves_snapshot(self):
        rows = torch.ones((4, 3), dtype=torch.bfloat16, device='cuda')
        initial = torch.tensor([[1., 1., 1.], [0., 0., 0.]], device='cuda')
        model = Emeans(3, 2, niter=1, batch_size=4, alpha0=.5)
        model.begin(4, initial)
        model.accumulate(rows[:1])
        backend, dtype, recipe = model._assignment_backend, model._input_dtype, dict(model.recipe)
        window, accumulation = model._batch_centroids, model._batch_accumulation
        with self.assertRaisesRegex(ValueError, 'shape'):
            model.train(rows.float(), initial[:1])
        self.assertEqual(model._assignment_backend, backend)
        self.assertEqual(model._input_dtype, dtype)
        self.assertEqual(model.recipe, recipe)
        self.assertIs(model._batch_centroids, window)
        self.assertIs(model._batch_accumulation, accumulation)
        self.assertFalse(model._failed)
        stats = model.accumulate(rows[1:])
        torch.testing.assert_close(stats['counts'], torch.tensor([4., 0.], device='cuda'), rtol=0, atol=0)
        torch.testing.assert_close(stats['sums'], initial * 4, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
