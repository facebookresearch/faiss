# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent pair-law references and actual two-owner current-row identity."""

import datetime
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from faiss.contrib.e_means.base import _protect_candidates
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids, _protect_removal_candidates


def _expected(candidates, nearest, same):
    selected = []
    for position, candidate in enumerate(candidates):
        if all(nearest[prior] != candidate and
               (nearest[position] != candidates[prior] or same[position]) for prior in selected):
            selected.append(position)
    return [candidates[position] for position in selected]


class TestPairLaw(unittest.TestCase):
    def test_exact_identity_backward_exception_and_sentinels(self):
        sentinel = torch.iinfo(torch.int64).max
        cases = [([0, 2], [1, 0], [False, False], [0]),
                 ([0, 2], [1, 0], [True, True], [0, 2]),
                 ([0, 1, 2], [1, 0, 0], [True] * 3, [0, 2]),
                 ([0, 2], [1, sentinel], [False] * 2, [0, 2]),
                 ([0, 2], [1, -1], [False] * 2, [0, 2]),
                 ([2**62, 2**62 + 2], [2**62 + 1, 2**62], [False] * 2, [2**62])]
        for ids, nearest, same, known in cases:
            with self.subTest(ids=ids, nearest=nearest, same=same):
                self.assertEqual(_expected(ids, nearest, same), known)
                actual = _protect_candidates(ids, nearest, lambda position: same[position])
                self.assertEqual(actual.tolist(), known)
                tensors = [torch.tensor(values) for values in (ids, nearest, same)]
                self.assertEqual(_protect_removal_candidates(*tensors).tolist(), known)
                if torch.cuda.is_available():
                    self.assertEqual(_protect_removal_candidates(*(tensor.cuda() for tensor in tensors)).tolist(), known)

    def test_local_lazy_readout_equality_matches_direct_edge_oracle(self):
        cases = [(np.array([0., -.25, 1., 10.]), [False, False]),
                 (np.zeros(4), [True, True]),
                 (np.array([1., 1., np.nextafter(np.float32(1), np.float32(2)), 10.]), [True, False]),
                 (np.array([0., 0., -0., 10.]), [True, True]),
                 (np.array([np.inf, np.inf, np.inf, 10.]), [True, True]),
                 (np.array([np.nan, np.nan, np.nan, 10.]), [False, False])]
        for rows, same in cases:
            with self.subTest(rows=rows):
                expected = _expected([0, 2], [1, 0], same)
                actual = _protect_candidates([0, 2], [1, 0], lambda position: (
                    rows[[0, 2][position]] == rows[[1, 0][position]]))
                self.assertEqual(actual.tolist(), expected)


def _owner_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120))
    check = unittest.TestCase()
    try:
        eps = torch.nextafter(torch.tensor(1., dtype=torch.float32), torch.tensor(2.)).item()
        modes = [("unequal", [0., -.25, 1., 10.], [-2., 10., -1., 10.], None, None, [2], [0]),
                 ("duplicates", [0., 0., 0., 10.], [-2., 10., -1., 10.], None, None, [1, 3], [0, 2]),
                 ("near_duplicate", [1., 1., eps, 10.], [-2., 10., -1., 10.], None, None, [2], [0]),
                 ("sentinel", [0., -.25, 1., 10.], [-2., 10., -1., 10.], [1, 2**63 - 1], [1, 2], [1, 3], [0, 2]),
                 ("false_waiver", [0., -.25, 1., 10.], [-2., 10., -1., 10.], [1, 0], [1, 2], [2], [0]),
                 ("missed_waiver", [0., 10., 0., 0.], [-1., 10., -2., 10.], [3, 2], [3, 1], [1, 3], [2, 0])]
        for name, values, costs, root_ids, other_ids, donors, victims in modes:
            full = torch.tensor(values, dtype=torch.float32, device=device)[:, None]
            model = EmeansShardedCentroids(1, 4, group=dist.group.WORLD, nepochs=1, batch_size=4,
                                           alpha0=.5, seed=7, quantizer=CastQuantizer(1, torch.bfloat16),
                                           assignment_backend="triton")
            shard = slice(rank * 2, rank * 2 + 2)
            model.begin(20, full[shard])
            model._usage.fill_(1)
            state = full.new_tensor([[0., costs[0]], [3., costs[1]], [5., costs[2]], [1., costs[3]]])
            model._removal_stats.copy_(state[shard])
            model._axes.fill_(1)
            original = model._nearest_other
            returned = []

            def legacy(ids, *, return_tensor=False):
                result = original(ids, return_tensor=return_tensor)
                if root_ids is not None:
                    result = torch.tensor(root_ids if rank == 0 else other_ids, dtype=torch.int64, device=device)
                returned.append((result, result.clone()))
                return result

            with mock.patch.object(model, "_nearest_other", legacy):
                source, target = model._choose_sharded_removal_splits()
                if rank == 0:
                    check.assertEqual(source.tolist(), donors, name)
                    check.assertEqual(target.tolist(), victims, name)
                else:
                    check.assertIsNone(source)
                    check.assertIsNone(target)
                # Real owner transfer still uses the same independent known pairs.
                check.assertEqual(model._revive_unused(model._split_rng), len(donors), name)
            for result, saved in returned:
                torch.testing.assert_close(result, saved, rtol=0, atol=0)
            expected = state.clone()
            expected[donors] /= 2
            expected[victims] = expected[donors]
            torch.testing.assert_close(model._removal_stats, expected[shard], rtol=0, atol=0)
            usage = full.new_ones(4)
            usage[donors] = usage[victims] = .5
            torch.testing.assert_close(model._usage, usage[shard], rtol=0, atol=0)
            check.assertEqual(model.n_splits, len(donors), name)
            moved = torch.zeros(4, dtype=torch.bool, device=device)
            moved[donors + victims] = True
            torch.testing.assert_close(model._moved, moved[shard], rtol=0, atol=0)

        # Identity uses exact current rows, not score-zero, and ignores invalid IDs.
        for values, partners, expected in [([0., -0., np.inf, np.inf], [1, 0, 3, 2], [1, 1, 1, 1]),
                                           ([np.nan, np.nan, 1., eps], [1, 0, 3, 2], [0, 0, 0, 0]),
                                           ([0., -.25, 1., 10.], [1, 0, -1, 2**63 - 1], [0, 0, 0, 0])]:
            full = torch.tensor(values, dtype=torch.float32, device=device)[:, None]
            model._centroids.copy_(full[shard])
            ids = torch.arange(4, dtype=torch.int64, device=device)
            nearest = torch.tensor(partners, dtype=torch.int64, device=device)
            same = model._same_readout_pairs(ids, nearest)
            check.assertEqual(same.tolist(), expected)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2
                     and not torch.version.hip and dist.is_nccl_available(), "two NVIDIA GPUs/NCCL required")
class TestShardedPairNCCL(unittest.TestCase):
    def test_actual_owner_identity_callbacks_and_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(_owner_worker, args=("file://" + folder + "/rendezvous",), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
