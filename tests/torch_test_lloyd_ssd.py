# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lloyd epoch updates and the SSD demo's exact/IVF sharded integration."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from faiss.contrib.e_means import QuantizedEmeans
from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids
from faiss.contrib.e_means.sharded_centroids_ivf import EmeansShardedCentroidsIVF

_SPEC = importlib.util.spec_from_file_location(
    "ssd_lloyd", Path(__file__).resolve().parents[1] / "benchs/e_means/lloyd.py"
)
_LLOYD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LLOYD)


class _CpuLloyd(_LLOYD._LloydEpoch, QuantizedEmeans):
    """Exercise the epoch policy with the native CPU assignment engine."""

    def __init__(self, d, k, **kwargs):
        super().__init__(d, k, **kwargs)
        self.global_k = k
        self._group_size = 1
        self._group = dist.group.WORLD


def _fixture(device):
    centers = torch.arange(6, device=device)[:, None] * 20.0
    centers = centers + torch.arange(16, device=device)[None, :] / 32
    offsets = torch.arange(36, device=device).remainder(3) / 8 - 0.125
    rows = centers[torch.arange(36, device=device).remainder(6)] + offsets[:, None]
    initial = torch.cat((centers, torch.full((2, 16), 10000.0, device=device)))
    return rows, initial


def _check_epochs(model, rows, initial, rank=0, ranks=1):
    n = len(rows)
    local_n, local_k = n // ranks, len(initial) // ranks
    model.begin(n, initial[rank * local_k:(rank + 1) * local_k].clone())
    expected = initial.clone()
    for budget in model.recipe["epoch_rows"]:
        local_rows = rows[rank * local_n:rank * local_n + budget // ranks]
        frozen = model.current_centroids().clone()
        for batch in local_rows.split(model.recipe["batch_size"] // ranks):
            stats = model.accumulate(batch)
            model.update(stats)
            torch.testing.assert_close(model.current_centroids(), frozen, rtol=0, atol=0)
        selected = torch.cat([
            rows[worker * local_n:worker * local_n + budget // ranks]
            for worker in range(ranks)
        ])
        if model.recipe.get("fine_dtype") == "bf16":
            represented = expected.to(torch.bfloat16).float()
            queries = selected.to(torch.bfloat16).float()
            labels = (represented.square().sum(1)[None, :]
                      - 2 * queries @ represented.t()).argmin(1)
        else:
            labels = (selected[:, None, :] - expected[None, :, :]).square().sum(2).argmin(1)
        for cluster in range(len(initial)):
            assigned = selected[labels == cluster]
            if len(assigned):
                expected[cluster] = assigned.double().mean(0).float()
        mse = (selected - expected[labels]).square().sum(1).mean().item()
        model.end_epoch()
        torch.testing.assert_close(
            model.current_centroids(), expected[rank * local_k:(rank + 1) * local_k],
            rtol=2e-6, atol=2e-5,
        )
        assert abs(model.iteration_stats[-1]["assignment_mse"] - mse) < 2e-4
        assert model.n_splits == 0
    result = model.finalize()
    assert model.finalize() is result
    torch.testing.assert_close(result, expected[rank * local_k:(rank + 1) * local_k])
    assert "alpha0" not in model.recipe


def _gpu_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        rows, initial = _fixture(torch.device("cuda", rank))
        for cls, parent, extra in (
            (_LLOYD.LloydShardedCentroids, EmeansShardedCentroids, {}),
            (_LLOYD.LloydShardedCentroidsIVF, EmeansShardedCentroidsIVF,
             {"nlist": 4, "nprobe": 4, "coarse_niter": 2}),
        ):
            for epochs in (0.5, 2.5):
                model = cls(16, 8, nepochs=epochs, batch_size=10, alpha0=1,
                            assignment_backend="triton", group=dist.group.WORLD, **extra)
                calls = []
                prepare = parent._prepare_assignment

                def counted(self):
                    calls.append(self._epoch)
                    return prepare(self)

                with mock.patch.object(parent, "_prepare_assignment", counted):
                    _check_epochs(model, rows, initial, rank, 2)
                assert calls == list(range(len(model.recipe["epoch_rows"])))
        # BF16 merges two distinct centroids on different owners. Assignment
        # must pick the lowest global ID; means must retain unrounded FP32 rows.
        initial = torch.zeros((4, 17), device=rows.device)
        initial[:, 0] = initial.new_tensor([1.0001, 4., 1.0021, 9.])
        rows = torch.zeros((8, 17), device=rows.device)
        rows[:, 0] = rows.new_tensor([1.0021, 4.12345, 9.12345, 1.0021,
                                     4.12345, 9.12345, 1.0021, 4.12345])
        rows[:, 1] = 0.000123
        model = _LLOYD.LloydShardedCentroids(
            17, 4, nepochs=1, batch_size=4, alpha0=1,
            assignment_backend="triton", group=dist.group.WORLD,
        )
        _check_epochs(model, rows, initial, rank, 2)
        assert model.recipe["representation"] == "float32"
        assert model.recipe["fine_dtype"] == "bf16"
        if rank == 0:
            assert abs(model.centroids[0, 0].item() - 1.0021) < 1e-6
    finally:
        dist.destroy_process_group()


class TestLloydSSD(unittest.TestCase):
    def test_epoch_means_and_fractional_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("gloo", init_method="file://" + directory + "/group",
                                    rank=0, world_size=1)
            try:
                rows, initial = _fixture(torch.device("cpu"))
                for epochs in (0.5, 2.5):
                    model = _CpuLloyd(16, 8, nepochs=epochs, batch_size=10, alpha0=1)
                    _check_epochs(model, rows, initial)
                    self.assertEqual(model.recipe["epoch_rows"],
                                     [18] if epochs == 0.5 else [36, 36, 18])
            finally:
                dist.destroy_process_group()

    def test_incomplete_epoch_keeps_centroids(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("gloo", init_method="file://" + directory + "/group",
                                    rank=0, world_size=1)
            try:
                rows, initial = _fixture(torch.device("cpu"))
                model = _CpuLloyd(16, 8, nepochs=1, batch_size=10, alpha0=1)
                model.begin(len(rows), initial)
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    model.end_epoch()
                torch.testing.assert_close(model.current_centroids(), initial)
                with self.assertRaisesRegex(RuntimeError, "finish all epochs"):
                    model.finalize()
            finally:
                dist.destroy_process_group()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_sharded_exact_and_ivf(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(_gpu_worker, nprocs=2,
                                        args=("file://" + directory + "/group",), join=True)


if __name__ == "__main__":
    unittest.main()
