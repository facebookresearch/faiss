# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded reductions for DINO query scoring and sharded revival."""

import datetime
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from faiss.contrib.e_means.sharded_centroids import _removal_scalar_stats

_SPEC = importlib.util.spec_from_file_location(
    "dino_row_norms",
    Path(__file__).resolve().parents[1] / "benchs/e_means/dino_data.py",
)
_DINO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DINO)


def _query_mse_worker(rank, rendezvous):
    from faiss.contrib.e_means import EmeansShardedCentroids
    # Reuse the demo's independent held-out MSE and global-tie oracles.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchs/e_means"))
    from demo_sharded_centroids import self_test_queries

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        model = EmeansShardedCentroids(
            17, 8, group=dist.group.WORLD, assignment_backend="native",
            quantizer=None, unused_threshold=0,
        )
        generator = torch.Generator(device=device).manual_seed(19 + rank)
        model.centroids = torch.randn((4, 17), device=device, generator=generator)
        self_test_queries(model, device)
    finally:
        dist.destroy_process_group()


class TestRowNorms(unittest.TestCase):
    def _equivalence(self, device):
        generator = torch.Generator(device=device).manual_seed(1234)
        for d in (1, 7, 1024):
            state = torch.randn((83, d + 1), generator=generator, device=device)
            usage = torch.rand(83, generator=generator, device=device)
            for dtype in (torch.float32, torch.bfloat16):
                # Non-contiguous rows, ragged chunks and exact BF16 rounding.
                rows = state[:, :-1].to(dtype)
                before = rows.clone()
                expected = rows.float().square().sum(1)
                for chunk in (1, 13, 128, None):
                    result = _DINO._squared_row_norms(rows, chunk_rows=chunk)
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual(result.device, rows.device)
                    # CUDA changes its reduction tree for different row grids;
                    # chunking preserves FP32 arithmetic, not bitwise rounding.
                    rtol = 4 * torch.finfo(torch.float32).eps if rows.is_cuda else 0
                    torch.testing.assert_close(result, expected, rtol=rtol, atol=0)
                torch.testing.assert_close(rows, before, rtol=0, atol=0)
            # Reproduce the strided axis slice of the packed (k, d+1) state.
            before = state.clone()
            expected = torch.stack((
                state[:, -1].double(), usage.double(),
                torch.linalg.vector_norm(state[:, :-1], dim=1, dtype=torch.float64),
            ), dim=1)
            for chunk in (1, 13, 128, None):
                result = _removal_scalar_stats(state, usage, chunk_rows=chunk)
                self.assertEqual(result.dtype, torch.float64)
                self.assertEqual(result.device, state.device)
                rtol = 4 * torch.finfo(torch.float64).eps if state.is_cuda else 0
                torch.testing.assert_close(result, expected, rtol=rtol, atol=0)
            torch.testing.assert_close(state, before, rtol=0, atol=0)

    def test_cpu_equivalence(self):
        self._equivalence(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_equivalence(self):
        self._equivalence(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
                         "requires two CUDA GPUs and NCCL")
    def test_query_mse_global_ties_and_unchanged_state(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _query_mse_worker, args=("file://" + directory + "/group",),
                nprocs=2, join=True,
            )

    def test_empty_and_invalid_chunks(self):
        self.assertEqual(_DINO._squared_row_norms(torch.empty(0, 7)).shape, (0,))
        self.assertEqual(
            _removal_scalar_stats(torch.empty(0, 8), torch.empty(0)).shape,
            (0, 3),
        )
        for chunk in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive"):
                _DINO._squared_row_norms(torch.empty(2, 7), chunk_rows=chunk)
            with self.assertRaisesRegex(ValueError, "positive"):
                _removal_scalar_stats(
                    torch.empty(2, 8), torch.empty(2), chunk_rows=chunk
                )

    def test_special_values(self):
        rows = torch.tensor([[0., 0.], [float("inf"), 1.],
                             [float("nan"), 1.], [1e20, 1.]])
        torch.testing.assert_close(
            _DINO._squared_row_norms(rows, chunk_rows=1),
            rows.square().sum(1), equal_nan=True, rtol=0, atol=0,
        )
        state = torch.cat((rows, torch.ones(4, 1)), dim=1)
        torch.testing.assert_close(
            _removal_scalar_stats(state, torch.ones(4), chunk_rows=1)[:, 2],
            torch.linalg.vector_norm(rows, dim=1, dtype=torch.float64),
            equal_nan=True, rtol=0, atol=0,
        )

    def test_default_workspace_is_bounded(self):
        # Cross the default 8192-row boundary at DINO's d=1024.
        state = torch.ones(8193, 1025)
        squares, norms = [], []
        square = torch.Tensor.square
        vector_norm = torch.linalg.vector_norm

        def bounded_square(rows, *args, **kwargs):
            squares.append((rows.shape[0], rows.numel() * rows.element_size()))
            return square(rows, *args, **kwargs)

        def bounded_norm(rows, *args, **kwargs):
            norms.append((rows.shape[0], rows.numel() * 8, kwargs["dtype"]))
            return vector_norm(rows, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "square", bounded_square):
            result = _DINO._squared_row_norms(state[:, :-1].bfloat16())
        self.assertEqual([n for n, _ in squares], [8192, 1])
        self.assertLessEqual(max(size for _, size in squares), 32 << 20)
        torch.testing.assert_close(result, torch.full((8193,), 1024.))
        with mock.patch.object(torch.linalg, "vector_norm", bounded_norm):
            result = _removal_scalar_stats(state, torch.ones(8193))
        self.assertEqual([n for n, _, _ in norms], [8192, 1])
        self.assertTrue(all(dtype == torch.float64 for _, _, dtype in norms))
        self.assertLessEqual(max(size for _, size, _ in norms), 64 << 20)
        torch.testing.assert_close(result[:, 2], torch.full((8193,), 32., dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
