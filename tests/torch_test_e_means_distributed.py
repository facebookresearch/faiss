# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Small caller-driven NCCL contracts; invoke explicitly on two or three GPUs."""
import datetime
import tempfile
import unittest

import torch
import torch.distributed as dist

from faiss.contrib.e_means import DistributedEmeans, Emeans


def _replicas(value, group):
    copies = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
    dist.all_gather(copies, value, group=group)
    for other in copies:
        torch.testing.assert_close(value, other, rtol=0, atol=0)


def _detached(model):
    # Running centroid/usage and pass totals have no public result until finalize.
    for value in (model._centroids, model._usage, model._batch_stats,
                  model._pass_counts, model._pass_sums, model._objective):
        if value is not None:
            assert not value.requires_grad and value.grad_fn is None


def _global_totals(group, transport, device):
    """Known labels and exact dyadic totals, independent of a second model."""
    assert dist.get_world_size(group) == 2
    rank = dist.get_rank(group)
    initial = torch.tensor([[0.], [2.], [4.]], device=device)
    caller_initial = (initial + 1000 * rank).requires_grad_()
    saved_initial = caller_initial.detach().clone()
    model = DistributedEmeans(1, 3, group=group, transport=transport,
                              niter=1, batch_size=8, alpha0=1, seed=7)
    model.begin(12, caller_initial)
    torch.testing.assert_close(model._centroids, initial, rtol=0, atol=0)
    torch.testing.assert_close(caller_initial, saved_initial, rtol=0, atol=0)
    _replicas(model._centroids, group)
    expected_counts = torch.zeros(3, device=device, dtype=torch.float64)
    expected_sums = torch.zeros((3, 1), device=device)
    for step, local_values in enumerate((
        ([0., 0., 0., 0.] if rank == 0 else [0., 2.25, 1.75, 2.]),
        ([0., 0.] if rank == 0 else [2., 2.]),
    )):
        local = torch.tensor(local_values, device=device)[:, None].requires_grad_()
        saved_local = local.detach().clone()
        before = model.current_centroids().clone()
        if rank == 0 and len(local) > 1:
            partial = model.accumulate(local[:1])
            assert partial["sums"] is None and partial["counts"] is None
            with unittest.TestCase().assertRaisesRegex(ValueError, "complete the pending"):
                model.update({})
            torch.testing.assert_close(model.current_centroids(), before, rtol=0, atol=0)
            stats = model.accumulate(local[1:])
        else:
            stats = model.accumulate(local)
        assert stats["n"] == len(local)
        assert stats["sums"].dtype == stats["counts"].dtype == torch.float32
        assert stats["objective"].dtype == torch.float64
        if step == 0:
            assert int((stats["counts"] > 0).sum()) == (1 if rank == 0 else 2)
        # External copied statistics may require gradients, but update detaches
        # them and must not mutate the caller's copies while packing collectives.
        external = {name: value.detach().clone().requires_grad_(
                        value.is_floating_point() or value.is_complex())
                    if isinstance(value, torch.Tensor) else value
                    for name, value in stats.items()}
        saved_stats = {name: value.detach().clone() for name, value in external.items()
                       if isinstance(value, torch.Tensor)}
        model.update(external)
        assert model._epoch_rows == (8 if step == 0 else 12)
        expected_counts += torch.tensor([5, 3, 0] if step == 0 else [2, 2, 0],
                                        device=device, dtype=torch.float64)
        expected_sums += torch.tensor(
            [[0.], [6. if step == 0 else 4.], [0.]], device=device
        )
        assert model._pass_counts.dtype == torch.float64
        assert model._pass_sums.dtype == torch.float32
        torch.testing.assert_close(model._pass_counts, expected_counts, rtol=0, atol=0)
        torch.testing.assert_close(model._pass_sums, expected_sums, rtol=0, atol=0)
        torch.testing.assert_close(model._objective, torch.tensor(
            0. if rank == 0 else 1 / 8, dtype=torch.float64, device=device), rtol=0, atol=0)
        for name, value in saved_stats.items():
            assert external[name].requires_grad == (value.is_floating_point() or value.is_complex())
            torch.testing.assert_close(external[name], value, rtol=0, atol=0)
        torch.testing.assert_close(local, saved_local, rtol=0, atol=0)
        _detached(model)
        _replicas(model._centroids, group)
        _replicas(model._usage, group)
    model.end_epoch()
    torch.testing.assert_close(model._objective, torch.tensor(
        1 / 8, dtype=torch.float64, device=device), rtol=0, atol=0)
    final = model.finalize()
    torch.testing.assert_close(final, initial, rtol=0, atol=0)
    _replicas(final, group)
    _detached(model)
    saved_final = final.clone()
    model.reset()
    torch.testing.assert_close(final, saved_final, rtol=0, atol=0)


def _rows(begin, end, device):
    return ((torch.arange(begin, end, device=device) % 3).float()[:, None]
            .expand(-1, 2).contiguous() / 8)


def _training(group, transport, device, alpha=0.5):
    """Replica/EMA/revival agreement; floating tolerance, not a universal oracle."""
    rank, ranks = dist.get_rank(group), dist.get_world_size(group)
    batch, population = 2 * ranks, 11 * ranks
    initial = torch.tensor([[0., 0.], [5., 5.], [100., 100.]], device=device)
    args = dict(niter=2, batch_size=batch, alpha0=alpha, seed=7,
                assignment_chunk_size=2, centroid_chunk_size=2)
    model = DistributedEmeans(2, 3, group=group, transport=transport, **args)
    reference = Emeans(2, 3, **args)
    caller_initial = initial + 1000 * rank
    saved_initial = caller_initial.clone()
    model.begin(population, caller_initial)
    reference.begin(population, initial)
    torch.testing.assert_close(caller_initial, saved_initial, rtol=0, atol=0)
    torch.testing.assert_close(model._centroids, initial, rtol=0, atol=0)
    with unittest.TestCase().assertRaises(NotImplementedError):
        model.train(_rows(0, batch, device))
    if rank != 0:
        # Noncoordinator split draws must not affect the broadcast revival.
        model._split_rng.manual_seed(123456)
    for _ in range(2):
        for begin in range(0, population, batch):
            n = min(batch, population - begin)
            local_n = n // ranks
            start = begin + rank * local_n
            local = _rows(start, start + local_n, device)
            saved_local = local.clone()
            model.update(model.accumulate(local))
            reference.update(reference.accumulate(_rows(begin, begin + n, device)))
            torch.testing.assert_close(local, saved_local, rtol=0, atol=0)
            _detached(model)
            _replicas(model._centroids, group)
            _replicas(model._usage, group)
            torch.testing.assert_close(model._centroids, reference._centroids,
                                       rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(model._usage, reference._usage, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(model._pass_counts, reference._pass_counts,
                                       rtol=0, atol=0)
            if model._pass_sums is not None:
                torch.testing.assert_close(model._pass_sums, reference._pass_sums,
                                           rtol=0, atol=0)
        model.end_epoch()
        reference.end_epoch()
        for field in ("imbalance_factor", "nsplit"):
            assert model.iteration_stats[-1][field] == reference.iteration_stats[-1][field]
    assert model.n_splits == reference.n_splits and model.n_splits > 0
    final = model.finalize()
    torch.testing.assert_close(final, reference.finalize(), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(model.obj, reference.obj, rtol=1e-6, atol=1e-5)
    _replicas(final, group)
    saved_final = final.clone()
    model.reset()
    torch.testing.assert_close(final, saved_final, rtol=0, atol=0)


def _invalid_sizes(group, transport, device):
    check = unittest.TestCase()
    initial = torch.tensor([[0., 0.], [5., 5.]], device=device)
    for n, batch in ((8, 5), (9, 4)):
        model = DistributedEmeans(2, 2, group=group, transport=transport,
                                  niter=1, batch_size=batch, alpha0=0.5)
        with check.assertRaisesRegex(ValueError, "divisible"):
            model.begin(n, initial)
        for call in (
            lambda _model=model: _model.accumulate(_rows(0, 1, device)),
            lambda _model=model: _model.update({}),
            model.end_epoch,
            model.finalize,
        ):
            with check.assertRaises(RuntimeError):
                call()
        # N below nominal B prescribes N, not an unusable nondivisible B.
        model.begin(4, initial)
        model.update(model.accumulate(_rows(0, 2, device)))
        model.end_epoch()
        torch.testing.assert_close(model.finalize(), torch.tensor(
            [[1 / 16, 1 / 16], [5., 5.]], device=device), rtol=0, atol=0)
    model = DistributedEmeans(2, 2, group=group, transport=transport,
                              niter=1, batch_size=4, alpha0=0.5)
    model.begin(6, initial)
    stats = model.accumulate(_rows(0, 2, device))
    for wrong_n in (0, 1, 3):
        with check.assertRaisesRegex(ValueError, "prescribed batch size"):
            model.update(dict(stats, n=wrong_n))
    model.update(stats)
    stats = model.accumulate(_rows(0, 1, device))
    with check.assertRaisesRegex(ValueError, "prescribed batch size"):
        model.update(dict(stats, n=2))
    model.update(stats)
    model.end_epoch()
    torch.testing.assert_close(model.finalize(), torch.tensor(
        [[1 / 24, 1 / 24], [5., 5.]], device=device), rtol=0, atol=0)


def _nccl_worker(rank, world_size, rendezvous, subgroup_only):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    subgroup = None
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=datetime.timedelta(seconds=60))
    try:
        if subgroup_only:
            # Every world rank creates the group; global rank0 is not its root
            # and never participates in any of its training collectives.
            subgroup = dist.new_group([1, 2], timeout=datetime.timedelta(seconds=60))
            if rank in (1, 2):
                for transport in ("dense", "sparse"):
                    _global_totals(subgroup, transport, device)
                    _training(subgroup, transport, device)
            else:
                with unittest.TestCase().assertRaisesRegex(ValueError, "group must contain the caller"):
                    DistributedEmeans(1, 2, group=subgroup)
            dist.barrier()
        else:
            for transport in ("dense", "sparse"):
                _global_totals(dist.group.WORLD, transport, device)
                _invalid_sizes(dist.group.WORLD, transport, device)
                _training(dist.group.WORLD, transport, device)
            _global_totals(dist.group.WORLD, "auto", device)
            _training(dist.group.WORLD, "dense", device, alpha=1)
    finally:
        if subgroup_only and rank in (1, 2) and subgroup is not None:
            dist.destroy_process_group(subgroup)
        dist.destroy_process_group()


class TestDistributedEmeans(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2, "NCCL and two CUDA GPUs required")
    def test_two_gpu_caller_owned_transports(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _nccl_worker,
                args=(2, "file://" + directory + "/group", False),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 3, "NCCL and three CUDA GPUs required")
    def test_three_gpu_nonzero_global_root(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _nccl_worker,
                args=(3, "file://" + directory + "/group", True),
                nprocs=3,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
