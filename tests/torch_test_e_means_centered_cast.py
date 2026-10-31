# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit centered casts: represented rows, original state, unchanged defaults.

CPU tests qualify codec/metadata only; CPU E-means still rejects quantizers.
GPU tests are small synthetic contracts, not accuracy or performance claims.
"""
import datetime
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest import mock

import torch

from faiss.contrib.e_means import Emeans, DistributedEmeans
from faiss.contrib.e_means.e_means import _resolve_recipe, _separate_splits
from faiss.contrib.e_means.e_means_quantizer import CastQuantizer
from e_means_test_utils import known_torch_centroids

_DISTRIBUTED_EVIDENCE = []


def _cuda_ready():
    return (torch.cuda.is_available() and not torch.version.hip
            and torch.cuda.get_device_capability() >= (8, 0))


def _cute_ready():
    if not _cuda_ready() or torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        return (importlib.metadata.version("quack-kernels") == "0.6.5"
                and importlib.metadata.version("nvidia-cutlass-dsl") == "4.7.1"
                and importlib.metadata.version("cuda-python").startswith("13."))
    except importlib.metadata.PackageNotFoundError:
        return False


def _fit(shift, dtype, device="cpu"):
    shift = torch.as_tensor(shift, dtype=torch.float32)
    q = CastQuantizer(shift.numel(), dtype, center=True)
    q.train(torch.stack((shift - 1, shift + 1)))
    return q.to(device)


class TestCenteredCastCPU(unittest.TestCase):
    def test_fit_and_frozen_metadata(self):
        for dtype in (torch.float16, torch.bfloat16):
            q = CastQuantizer(2, dtype, center=True)
            self.assertFalse(q.is_trained)
            with self.assertRaisesRegex(RuntimeError, "trained explicitly"):
                q.compute_codes(torch.zeros(1, 2))
            with self.assertRaisesRegex(RuntimeError, "trained explicitly"):
                q.to("cpu")
            sample = torch.tensor([[99., -51.], [101., -49.]], requires_grad=True)
            saved = sample.detach().clone()
            with torch.inference_mode():
                self.assertIsNone(q.train(sample))
            self.assertTrue(q.is_trained)
            self.assertFalse(q.shift.is_inference())
            self.assertFalse(q.shift.requires_grad)
            torch.testing.assert_close(q.shift, torch.tensor([100., -50.]), rtol=0, atol=0)
            token = q._fit_token()
            self.assertIs(q.to("cpu"), q)
            self.assertEqual(q._fit_token(), token)
            with torch.no_grad():
                sample.add_(10)
            torch.testing.assert_close(q.shift, saved.mean(0), rtol=0, atol=0)
            with self.assertRaises(AttributeError):
                q.center = False
            with self.assertRaises(AttributeError):
                q.shift = torch.zeros(2)
            self.assertIsNone(q.train(saved + 2))
            self.assertNotEqual(q._fit_token(), token)
            q.shift.add_(1)
            with self.assertRaisesRegex(RuntimeError, "frozen"):
                q.decode(torch.zeros(1, 2, dtype=dtype))

    def test_fit_validation(self):
        with self.assertRaises(TypeError):
            CastQuantizer(1, torch.float16, center=torch.tensor(True))
        for sample, error in ((None, TypeError), (torch.zeros(0, 2), ValueError),
                              (torch.zeros(2), ValueError), (torch.zeros(2, 1), ValueError),
                              (torch.zeros(1, 2, dtype=torch.int32), TypeError),
                              (torch.zeros(1, 2, dtype=torch.float64), TypeError)):
            q = CastQuantizer(2, torch.float16, center=True)
            with self.assertRaises(error):
                q.train(sample)
            self.assertFalse(q.is_trained)
        for value in (float("nan"), float("inf")):
            q = CastQuantizer(2, torch.float16, center=True)
            with mock.patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("value sync")), \
                 mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("value sync")):
                q.train(torch.full((1, 2), value))
            self.assertTrue(q.is_trained)  # Unchecked fit follows Torch.
            with self.assertRaises(ValueError):
                q.validate(torch.zeros(1, 2, dtype=torch.float16))
        q = _fit([100000.], torch.float16)
        q.validate(torch.tensor([[100001.]]))
        q.validate(torch.tensor([[1.]], dtype=torch.float16))  # Matching codes.
        with self.assertRaises(ValueError):
            q.validate(torch.tensor([[200000.]]))
        with self.assertRaises(ValueError):
            q.validate(torch.tensor([[float("inf")]], dtype=torch.float16))
        with self.assertRaises(ValueError):
            q.to("meta")
        # Finite BF16 residuals can round up, then overflow FP32 add-back.
        maximum = torch.finfo(torch.float32).max
        q = CastQuantizer(1, torch.bfloat16, center=True)
        q.train(torch.tensor([[maximum * .25]], dtype=torch.float32))
        original = torch.tensor([[maximum]], dtype=torch.float32)
        codes = q.compute_codes(original)
        self.assertTrue(bool(torch.isfinite(codes).all()))
        self.assertTrue(bool(torch.isinf(q.decode(codes)).all()))
        for value in (original, codes):
            with self.assertRaisesRegex(ValueError, "decode to finite"):
                q.validate(value)
        safe = original * .5
        q.validate(safe)
        q.validate(q.compute_codes(safe))

    def test_original_encoding_and_preencoded_model_convention(self):
        for dtype in (torch.float16, torch.bfloat16):
            q = _fit([100., -50.], dtype)
            x = torch.tensor([[100.25, 0., -49.5, 0.],
                              [99.75, 0., -50.5, 0.]], requires_grad=True)[:, ::2]
            saved = x.detach().clone()
            codes = q.compute_codes(x)
            self.assertTrue(codes.is_contiguous())
            self.assertFalse(codes.requires_grad)
            torch.testing.assert_close(codes.float(), x.detach() - q.shift, rtol=0, atol=0)
            torch.testing.assert_close(q.decode(codes), saved, rtol=0, atol=0)
            torch.testing.assert_close(x, saved, rtol=0, atol=0)
            model = Emeans(2, 1, quantizer=q)
            torch.testing.assert_close(model._prepare_rows(codes), codes, rtol=0, atol=0)
            # compute_codes always encodes original rows, even matching dtype.
            original_rounded = torch.tensor([[100., -50.]], dtype=dtype)
            torch.testing.assert_close(q.compute_codes(original_rounded),
                                       torch.zeros(1, 2, dtype=dtype), rtol=0, atol=0)
            torch.testing.assert_close(q.decode(model._prepare_rows(original_rounded)),
                                       original_rounded.float() + q.shift, rtol=0, atol=0)
            with self.assertRaises(TypeError):
                q.decode(saved)  # Explicit FP32 state is never decoded as codes.
            with self.assertRaisesRegex(ValueError, "quantizer=None"):
                model.begin(2, saved[:1])
            model._cast_fit_token = q._fit_token()
            q.train(saved + 1)
            with self.assertRaisesRegex(RuntimeError, "fit changed"):
                model._prepare_rows(saved)
            # A new run can seed with the new fit without changing old binding.
            torch.testing.assert_close(model._prepare_rows(saved, new_run=True),
                                       q.compute_codes(saved), rtol=0, atol=0)

    def test_hot_conversion_has_no_tensor_value_read(self):
        q = _fit([100., -50.], torch.bfloat16)
        rows = torch.tensor([[100.25, -49.5]])
        with mock.patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("value sync")), \
             mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("value sync")):
            codes = q.compute_codes(rows)
            decoded = q.decode(codes)
            q._fit_token()
            q.to("cpu")
        torch.testing.assert_close(decoded, rows, rtol=0, atol=0)

    def test_spherical_guard_split_frame_and_decoded_row_rounding(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "spherical"):
                Emeans(1, 1, spherical=True,
                       quantizer=CastQuantizer(1, dtype, center=True))
            Emeans(1, 1, spherical=True, quantizer=CastQuantizer(1, dtype))
            for shift in (0., 3., 50., 1000., -1000., float(2**39), -float(2**39)):
                q = _fit([shift], dtype)
                donor = torch.tensor([[shift]], dtype=torch.float32)
                child = donor.clone()
                with mock.patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("value sync")), \
                     mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("value sync")):
                    _separate_splits(donor, child, dtype, q)
                self.assertFalse(torch.equal(q.compute_codes(donor), q.compute_codes(child)))
                q.validate(child)
            # At 2**40, even the closest FP32 neighbour exceeds FP16 range.
            q = _fit([float(2**40)], dtype)
            donor = torch.tensor([[float(2**40)]], dtype=torch.float32)
            child = donor.clone()
            _separate_splits(donor, child, dtype, q)
            if dtype == torch.float16:
                torch.testing.assert_close(child, donor, rtol=0, atol=0)
                for direction in (float("inf"), float("-inf")):
                    adjacent = torch.nextafter(donor, torch.full_like(donor, direction))
                    self.assertGreater((adjacent - donor).abs().item(), torch.finfo(dtype).max)
                    self.assertFalse(bool(torch.isfinite(q.compute_codes(adjacent)).all()))
            else:
                self.assertFalse(torch.equal(q.compute_codes(donor), q.compute_codes(child)))
                q.validate(child)
            tiny = CastQuantizer(1, dtype, center=True)
            tiny.train(torch.tensor([[2**-24]]))
            decoded = tiny.decode(torch.ones(3, 1, dtype=dtype))
            torch.testing.assert_close(decoded, torch.ones(3, 1), rtol=0, atol=0)
            self.assertEqual(decoded.sum().item(), 3.)
            self.assertNotEqual((torch.tensor(3.) + 3 * tiny.shift[0]).item(), 3.)


class _GPUContracts:
    def arms(self):
        backends = ["native"]
        if importlib.util.find_spec("triton") is not None:
            backends.append("triton")
        return [(backend, dtype) for backend in backends
                for dtype in (torch.float16, torch.bfloat16)]

    def test_chunk_statistics_state_and_jit_codes(self):
        for backend, dtype in self.arms():
            with self.subTest(backend=backend, dtype=dtype):
                q = _fit([100., -50.], dtype, "cuda")
                initial = torch.tensor([[100., -50.], [104., -50.]], device="cuda")
                x = initial[[0, 1, 0, 1, 0]] + torch.tensor(
                    [[.25, .5], [-.25, -.5], [-.25, -.5], [.25, .5], [0., 0.]], device="cuda")
                codes, expected_labels = q.compute_codes(x), torch.tensor([0, 1, 0, 1, 0], device="cuda")
                decoded = q.decode(codes)
                model = Emeans(2, 2, niter=1, batch_size=3, alpha0=1, quantizer=q,
                               assignment_backend=backend, assignment_chunk_size=2,
                               capture_assignments=True)
                model.begin(5, initial)
                handle = model.current_centroids()
                self.assertEqual(handle.data_ptr(), model._centroids.data_ptr())
                model.accumulate(x[:0])
                partial = model.accumulate(x[:1])
                self.assertIsNone(partial["sums"])
                torch.testing.assert_close(handle, initial, rtol=0, atol=0)
                all_sums = torch.zeros_like(initial)
                all_counts = initial.new_zeros(2)
                for start, end, stats in ((0, 3, model.accumulate(codes[1:3])),
                                          (3, 5, None)):
                    if stats is None:
                        stats = model.accumulate(x[start:end])
                    sums, counts = torch.zeros_like(initial), initial.new_zeros(2)
                    sums.index_add_(0, expected_labels[start:end], decoded[start:end])
                    counts.index_add_(0, expected_labels[start:end], counts.new_ones(end-start))
                    torch.testing.assert_close(stats["sums"], sums, rtol=0, atol=0)
                    torch.testing.assert_close(stats["counts"], counts, rtol=0, atol=0)
                    self.assertFalse(model._batch_code_units)
                    model.update(stats)
                    all_sums += sums
                    all_counts += counts
                model.end_epoch()
                torch.testing.assert_close(model._pass_sums, all_sums, rtol=0, atol=0)
                final = model.finalize()
                torch.testing.assert_close(final, all_sums / all_counts[:, None], rtol=0, atol=0)
                self.assertEqual(final.dtype, torch.float32)
                distance, labels = model.assign(x)
                code_distance, code_labels = model.assign(codes)
                torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
                torch.testing.assert_close(labels, code_labels, rtol=0, atol=0)
                torch.testing.assert_close(distance, code_distance, rtol=0, atol=0)
                oracle = (codes.double()[:, None] - q.compute_codes(final).double()[None]).square().sum(2)
                torch.testing.assert_close(labels, oracle.argmin(1), rtol=0, atol=0)
                torch.testing.assert_close(distance.double(), oracle.min(1).values, atol=2e-4, rtol=2e-5)

    def test_decode_before_statistics_and_fit_guards(self):
        for backend, dtype in self.arms():
            with self.subTest(backend=backend, dtype=dtype):
                q = CastQuantizer(1, dtype, center=True)
                initial = torch.ones(1, 1, device="cuda")
                model = Emeans(1, 1, niter=1, batch_size=3, alpha0=1,
                               quantizer=q, assignment_backend=backend)
                with self.assertRaisesRegex(RuntimeError, "trained explicitly"):
                    model.begin(3, initial)
                q.train(torch.tensor([[2**-24]]))
                with self.assertRaisesRegex(ValueError, "same device"):
                    model.begin(3, initial)
                q.to("cuda")
                model.begin(3, initial)
                codes = torch.ones(3, 1, device="cuda", dtype=dtype)
                model.accumulate(codes[:1])
                stats = model.accumulate(codes[1:])
                self.assertEqual(stats["sums"].item(), 3.)
                self.assertEqual(stats["counts"].item(), 3.)
                model.update(stats)
                model.end_epoch()
                self.assertEqual(model.finalize().item(), 1.)
                q.train(torch.tensor([[2**-23]], device="cuda"))
                with self.assertRaisesRegex(RuntimeError, "fit changed"):
                    model.assign(codes)
                with self.assertRaisesRegex(RuntimeError, "fit changed"):
                    model.finalize()
                # Tuple results and implicit seed decoding survive refit.
                centroids, labels = model.train(codes)
                self.assertIs(centroids, model.centroids)
                self.assertIs(labels, model.assignments)
                self.assertEqual(model.obj[-1].ndim, 0)
                self.assertEqual(model.centroids.item(), q.decode(codes)[0].item())
                model.begin(3, initial)
                q.shift.add_(1)
                with self.assertRaisesRegex(RuntimeError, "frozen"):
                    model.accumulate(codes)

    def test_resident_replay_and_seeded_encoding_parity(self):
        for backend, dtype in self.arms():
            with self.subTest(backend=backend, dtype=dtype):
                q = _fit([100., -40., 30.], dtype, "cuda")
                ids = torch.arange(129, device="cuda")
                x = q.shift + torch.stack(((ids % 4).float(), (ids % 3).float() / 4,
                                           (ids % 5).float() / 8), dim=1)
                args = dict(niter=2, seed=10, quantizer=q, assignment_backend=backend,
                            capture_assignments=True)
                batch, _ = _resolve_recipe(129, 4, 2, "throughput", None, None)
                eager = Emeans(3, 4, assignment_chunk_size=batch, **args)
                eager.train(q.compute_codes(x))
                replay = Emeans(3, 4, **args)
                calls = [0]
                original = torch.cuda.CUDAGraph.replay
                capture = Emeans._capture_step
                owners = []
                graphs, pools, bindings = [], [], []

                def retained(
                    model,
                    *args,
                    _capture=capture,
                    _backend=backend,
                    _bindings=bindings,
                    _graphs=graphs,
                    _owners=owners,
                    _pools=pools,
                    _q=q,
                    **kwargs,
                ):
                    result = _capture(model, *args, **kwargs)
                    _graphs.append(weakref.ref(result[0]))
                    _pools.append(
                        weakref.ref(result[1] if len(result) == 2 else result[2])
                    )
                    if _backend == "cute":
                        _owners.append(weakref.ref(result[1]))
                        _bindings.extend(
                            weakref.ref(binding)
                            for binding in result[1].bindings.values()
                        )
                        self.assertIs(result[1].transform_bias, _q.shift)
                    return result

                def counted(
                    graph,
                    _calls=calls,
                    _bindings=bindings,
                    _graphs=graphs,
                    _owners=owners,
                    _pools=pools,
                    _original=original,
                ):
                    _calls[0] += 1
                    self.assertTrue(
                        all(
                            owner() is not None
                            for owner in _owners + _graphs + _pools + _bindings
                        )
                    )
                    return _original(graph)

                with mock.patch.object(torch.cuda.CUDAGraph, "replay", counted), \
                     mock.patch.object(Emeans, "_capture_step", retained):
                    centroids, labels = replay.train(x)
                self.assertGreater(calls[0], 0)
                self.assertIs(centroids, replay.centroids)
                self.assertIs(labels, replay.assignments)
                torch.testing.assert_close(replay.centroids, eager.centroids, atol=2e-5, rtol=2e-6)
                torch.testing.assert_close(replay.assignments, eager.assignments, rtol=0, atol=0)
                torch.testing.assert_close(replay.obj, eager.obj, atol=2e-4, rtol=2e-5)
                gc.collect()
                # Train retires its graph/pool; the reusable model cache stays
                # owned until reset, as in the original CuTe lifetime contract.
                self.assertTrue(graphs and pools)
                self.assertTrue(all(owner() is None for owner in graphs + pools))
                if backend == "cute":
                    self.assertTrue(owners and bindings)
                    self.assertIs(replay._cute_assignment_cache, owners[-1]())
                    self.assertIs(replay._cute_assignment_cache.transform_bias, q.shift)
                replay.reset()
                gc.collect()
                self.assertIsNone(replay._cute_assignment_cache)
                self.assertTrue(all(owner() is None for owner in owners + bindings))


@unittest.skipUnless(_cuda_ready(), "NVIDIA SM80 or later required")
class TestCenteredCastCUDA(_GPUContracts, unittest.TestCase):
    pass


@unittest.skipUnless(_cute_ready(), "pinned CuTe BF16 stack on SM90 required")
class TestCenteredCastCute(_GPUContracts, unittest.TestCase):
    def arms(self):
        return [("cute", torch.bfloat16)]

    def test_prepared_identity_refresh(self):
        q = _fit([100.] * 65, torch.bfloat16, "cuda")
        c = q.shift + torch.arange(257, device="cuda").float()[:, None].expand(-1, 65) / 8
        rows = q.compute_codes(c[:17] + .03125)
        model = known_torch_centroids(c, quantizer=q, assignment_backend="cute")
        allocations = []
        center_rows = q._center_rows

        def measured(value):
            before = torch.cuda.memory_allocated()
            result = center_rows(value)
            allocations.append(dict(shape=list(result.shape), device=str(result.device),
                                    dtype=str(result.dtype), logical_bytes=result.numel() * result.element_size(),
                                    allocated_delta=torch.cuda.memory_allocated() - before))
            return result

        def check_result(result):
            distance, labels = result
            oracle = (rows.double()[:, None] - q.compute_codes(model.centroids).double()[None]).square().sum(2)
            torch.testing.assert_close(labels, oracle.argmin(1), rtol=0, atol=0)
            torch.testing.assert_close(distance.double(), oracle.min(1).values, atol=2e-4, rtol=2e-5)

        with mock.patch.object(q, "_center_rows", measured):
            check_result(model.assign(rows))
            cached = model._cute_assignment_cache
            pointers = (cached.centroids.data_ptr(), cached.norms.data_ptr())
            torch.testing.assert_close(cached.centroids[:, :65], q.compute_codes(c), rtol=0, atol=0)
            check_result(model.assign(rows))
            self.assertIs(model._cute_assignment_cache, cached)
            model.centroids.add_(.25)
            check_result(model.assign(rows))
            self.assertIs(model._cute_assignment_cache, cached)
            self.assertEqual((cached.centroids.data_ptr(), cached.norms.data_ptr()), pointers)
            torch.testing.assert_close(cached.centroids[:, :65], q.compute_codes(model.centroids), rtol=0, atol=0)
            self.assertIs(cached.transform_bias, q.shift)
        print("CENTERED_CAST_MIRROR_TEMP " + json.dumps(allocations, sort_keys=True), flush=True)


def _distributed_worker(rank, rendezvous):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.cuda.reset_peak_memory_stats(device)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for transport in ("dense", "sparse"):
                q = _fit([100.], dtype, device)
                initial = torch.tensor([[100.], [104.]], device=device)
                model = DistributedEmeans(1, 2, niter=1, batch_size=8, alpha0=1,
                                          quantizer=q, group=dist.group.WORLD, transport=transport)
                model.begin(10, initial)
                expected_sums = torch.zeros_like(initial)
                expected_counts = initial.new_zeros(2)
                for n in (4, 1):
                    # Both ranks contribute distinct dyadic rows, stable labels.
                    x = initial[rank:rank+1].expand(n, -1) + (-.25 if rank else .25)
                    codes = q.compute_codes(x)
                    model.accumulate(codes[:0])
                    if n > 1:
                        model.accumulate(codes[:1])
                    stats = model.accumulate(x[1:] if n > 1 else x)
                    sums, counts = torch.zeros_like(initial), initial.new_zeros(2)
                    sums[rank] = q.decode(codes).sum(0)
                    counts[rank] = n
                    torch.testing.assert_close(stats["sums"], sums, rtol=0, atol=0)
                    torch.testing.assert_close(stats["counts"], counts, rtol=0, atol=0)
                    external = {key: value.clone() if isinstance(value, torch.Tensor) else value
                                for key, value in stats.items()}
                    model.update(external)
                    torch.testing.assert_close(external["sums"], sums, rtol=0, atol=0)
                    dist.all_reduce(sums)
                    dist.all_reduce(counts)
                    expected_sums += sums
                    expected_counts += counts
                    torch.testing.assert_close(model._pass_sums, expected_sums, rtol=0, atol=0)
                model.end_epoch()
                torch.testing.assert_close(model.current_centroids(), expected_sums / expected_counts[:, None], rtol=0, atol=0)
                torch.testing.assert_close(model.finalize(), expected_sums / expected_counts[:, None], rtol=0, atol=0)
                dist.barrier()
        peak = torch.cuda.max_memory_allocated(device)
        assert peak <= 2**30, "child Torch allocated peak exceeds 1 GiB"
        record = dict(rank=rank, pid=os.getpid(), device=str(device),
                      torch_peak_allocated_bytes=peak, source=__file__,
                      gpu_name=torch.cuda.get_device_name(device),
                      gpu_sm=list(torch.cuda.get_device_capability(device)))
        path = Path(rendezvous.removeprefix("file://")).parent / ("rank_%d.json" % rank)
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_cuda_ready() and torch.cuda.device_count() >= 2,
                     "two NVIDIA SM80 or later devices required")
class TestCenteredCastDistributed(unittest.TestCase):
    def test_two_rank_original_statistics_dense_sparse(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(_distributed_worker,
                                        args=("file://" + folder + "/rendezvous",), nprocs=2, join=True)
            records = [json.loads((Path(folder) / ("rank_%d.json" % rank)).read_text()) for rank in range(2)]
            _DISTRIBUTED_EVIDENCE.extend(records)
            print("CENTERED_CAST_DISTRIBUTED_PEAK " + json.dumps(records, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
