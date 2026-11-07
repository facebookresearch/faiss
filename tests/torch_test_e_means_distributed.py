# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Small caller-driven NCCL contracts; invoke explicitly on two or three GPUs."""
import datetime
import tempfile
import unittest
import importlib.util
from unittest import mock

import torch
import torch.distributed as dist

from faiss.contrib.e_means import DistributedEmeans
from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
from faiss.contrib.e_means.quantizer import CastQuantizer
from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer


def _replicas(value, group):
    copies = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
    dist.all_gather(copies, value, group=group)
    for other in copies:
        torch.testing.assert_close(value, other, rtol=0, atol=0)


def _nvidia_sm80_triton():
    """The BF16 default representation needs an NVIDIA SM80 or later GPU, and
    the IVF coarse assignment, the splits and the shared codecs need Triton."""
    return (torch.cuda.is_available()
            and getattr(torch.version, "hip", None) is None
            and importlib.util.find_spec("triton") is not None
            and torch.cuda.get_device_capability() >= (8, 0))


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
                              nepochs=1, batch_size=8, alpha0=1, seed=7, quantizer=None)
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
    del model
    torch.testing.assert_close(final, saved_final, rtol=0, atol=0)


def _rows(begin, end, device):
    return ((torch.arange(begin, end, device=device) % 3).float()[:, None]
            .expand(-1, 2).contiguous() / 8)


def _training(group, transport, device, alpha=0.5, quantizer=None, backend="native", byte_dtype=None):
    """Replica/EMA/revival agreement; floating tolerance, not a universal oracle."""
    rank, ranks = dist.get_rank(group), dist.get_world_size(group)
    batch, population = 2 * ranks, 11 * ranks
    initial = torch.tensor([[0., 0.], [5., 5.], [100., 100.]], device=device)
    def represented(rows):
        if byte_dtype is not None:
            return (8 * rows + (128 if byte_dtype == torch.uint8 else 0)).to(byte_dtype)
        return quantizer.compute_codes(rows) if isinstance(quantizer, Int8Quantizer) else rows
    if byte_dtype is not None:
        initial = 8 * initial + (128 if byte_dtype == torch.uint8 else 0)
    args = dict(nepochs=2, batch_size=batch, alpha0=alpha, seed=7,
                assignment_chunk_size=2, centroid_chunk_size=2, quantizer=quantizer)
    model = DistributedEmeans(
        2,
        3,
        group=group,
        transport=transport,
        assignment_backend=backend,
        **args
    )
    model_class = QuantizedEmeans
    reference = model_class(
        2,
        3,
        assignment_backend=(
            backend
            if byte_dtype is not None or isinstance(quantizer, Int8Quantizer)
            else "native"
        ),
        **args
    )
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
            local = represented(_rows(start, start + local_n, device))
            saved_local = local.clone()
            before = model.current_centroids().clone()
            if rank == 0 and len(local) > 1:
                partial = model.accumulate(local[:1])
                assert partial['sums'] is None and partial['counts'] is None
                with unittest.TestCase().assertRaisesRegex(ValueError, 'complete the pending'):
                    model.update({})
                torch.testing.assert_close(model.current_centroids(), before, rtol=0, atol=0)
                stats = model.accumulate(local[1:])
            else:
                stats = model.accumulate(local)
            expected_dtype = torch.int64 if byte_dtype is not None else torch.float32
            assert stats["sums"].dtype == stats["counts"].dtype == expected_dtype
            model.update(stats)
            assert model._epoch_rows == begin + n
            reference.update(reference.accumulate(represented(_rows(begin, begin + n, device))))
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
    del model
    torch.testing.assert_close(final, saved_final, rtol=0, atol=0)


def _integer_external_totals(group, transport, device):
    """Odd counts above FP32 exactness discriminate typed transport by values."""
    c, rank = 2**24 + 1, dist.get_rank(group)
    model = DistributedEmeans(
        1,
        3,
        group=group,
        transport=transport,
        assignment_backend="triton",
        nepochs=1,
        batch_size=2 * c,
        alpha0=1,
        seed=7,
    )
    model.begin(2*c, torch.tensor([[254.], [255.], [4.]], device=device))
    counts = torch.tensor([c, 0, 0] if rank == 0 else [c-1, 1, 0],
                          device=device, dtype=torch.int64)
    sums = (counts * torch.tensor([254, 255, 0], device=device))[:, None]
    objective = torch.zeros((), device=device, dtype=torch.float64, requires_grad=True)
    saved_sums, saved_counts = sums.clone(), counts.clone()
    model.update(dict(sums=sums, counts=counts, objective=objective, n=c))
    expected_counts = torch.tensor([2*c-1, 1, 0], device=device, dtype=torch.int64)
    expected_sums = (expected_counts * torch.tensor([254, 255, 0], device=device))[:, None]
    assert model._pass_counts.dtype == model._pass_sums.dtype == torch.int64
    torch.testing.assert_close(model._pass_counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(model._pass_sums, expected_sums, rtol=0, atol=0)
    torch.testing.assert_close(sums, saved_sums, rtol=0, atol=0)
    torch.testing.assert_close(counts, saved_counts, rtol=0, atol=0)
    _detached(model)
    model.end_epoch()
    # Revival may change an empty cluster; closing retains that current value.
    current_empty = model._centroids[2].clone()
    final = model.finalize()
    torch.testing.assert_close(final[:2], torch.tensor([[254.], [255.]], device=device), rtol=0, atol=0)
    torch.testing.assert_close(final[2], current_empty, rtol=0, atol=0)
    _replicas(final, group)


def _invalid_sizes(group, transport, device):
    check = unittest.TestCase()
    initial = torch.tensor([[0., 0.], [5., 5.]], device=device)
    for n, batch in ((8, 5), (9, 4)):
        model = DistributedEmeans(2, 2, group=group, transport=transport,
                                  nepochs=1, batch_size=batch, alpha0=0.5)
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
        model = DistributedEmeans(2, 2, group=group, transport=transport,
                                  nepochs=1, batch_size=batch, alpha0=0.5)
        model.begin(4, initial)
        model.update(model.accumulate(_rows(0, 2, device)))
        model.end_epoch()
        torch.testing.assert_close(model.finalize(), torch.tensor(
            [[1 / 16, 1 / 16], [5., 5.]], device=device), rtol=0, atol=0)
    model = DistributedEmeans(2, 2, group=group, transport=transport,
                              nepochs=1, batch_size=4, alpha0=0.5)
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


def _sharder_contracts(group, device):
    from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids
    from torch_test_e_means_cute import _ready

    rank = dist.get_rank(group)
    check = unittest.TestCase()
    initial = torch.tensor([[0.0], [2.0], [4.0], [9.0]], device=device)
    expected = torch.tensor([[0.25], [1.75], [4.25], [9.0]], device=device)
    local = torch.tensor([[.25], [1.75], [4.25]] * 2, device=device)
    saved_rows = local.clone()
    shard = slice(2 * rank, 2 * rank + 2)
    args = dict(group=group, nepochs=2, batch_size=8, alpha0=1,
                assignment_chunk_size=3, capture_assignments=True, quantizer=None)
    model = EmeansShardedCentroids(1, 4, **args)
    with check.assertRaisesRegex(ValueError, "divisible"):
        model.begin(11, initial[shard])
    model = EmeansShardedCentroids(1, 4, **args)
    model.begin(12, initial[shard])
    torch.testing.assert_close(model.current_centroids(), initial[shard], rtol=0, atol=0)
    with check.assertRaises(RuntimeError):
        model.final_closing_mean([local])
    retained_labels = []
    for epoch in range(2):
        for begin in (0, 4):
            rows = local[begin:begin + 4]
            parts = [torch.empty_like(model.current_centroids()) for _ in range(2)]
            dist.all_gather(parts, model.current_centroids(), group=group)
            frozen = torch.cat(parts)
            global_rows = torch.cat((rows, rows))
            distances = (global_rows.double()[:, None] - frozen.double()).square().sum(2)
            labels = distances.argmin(1)
            counts = torch.bincount(labels, minlength=4)
            sums = torch.zeros((4, 1), dtype=torch.float64, device=device)
            sums.index_add_(0, labels, global_rows.double())
            before = model.current_centroids().clone()
            partial = model.accumulate(rows[:1])
            assert partial["n"] == 2 and partial["sums"] is None and partial["counts"] is None
            torch.testing.assert_close(model.current_centroids(), before, rtol=0, atol=0)
            stats = model.accumulate(rows[1:])
            assert stats["n"] == 2 * len(rows)
            torch.testing.assert_close(stats["sums"], sums[shard].float(), rtol=0, atol=0)
            torch.testing.assert_close(stats["counts"], counts[shard].float(), rtol=0, atol=0)
            owned = (labels >= 2 * rank) & (labels < 2 * rank + 2)
            torch.testing.assert_close(stats["objective"], distances[owned, labels[owned]].sum(),
                                       rtol=0, atol=0)
            if epoch == 1:
                for contribution, oracle in ((partial, labels[:1]), (stats, labels[1:len(rows)])):
                    torch.testing.assert_close(contribution["labels"], oracle, rtol=0, atol=0)
                    retained_labels.append((contribution["labels"], oracle.clone()))
            else:
                assert "labels" not in partial and "labels" not in stats
            model.update(stats)
        model.end_epoch()
    torch.testing.assert_close(model.current_centroids(), expected[shard], rtol=0, atol=0)
    closed = model.final_closing_mean([local[:2], local[2:]])
    torch.testing.assert_close(closed, expected[shard], rtol=0, atol=0)
    assert closed.dtype == torch.float32 and not closed.requires_grad
    torch.testing.assert_close(local, saved_rows, rtol=0, atol=0)
    torch.testing.assert_close(initial, torch.tensor([[0.], [2.], [4.], [9.]], device=device), rtol=0, atol=0)
    for labels, oracle in retained_labels:
        torch.testing.assert_close(labels, oracle, rtol=0, atol=0)
    saved_closed = closed.clone()
    model = EmeansShardedCentroids(1, 4, **args)
    torch.testing.assert_close(closed, saved_closed, rtol=0, atol=0)
    resident = EmeansShardedCentroids(1, 4, **args)
    resident.train(local, init_centroids=initial[shard])
    torch.testing.assert_close(resident.centroids, expected[shard], rtol=0, atol=0)
    torch.testing.assert_close(resident.assignments, torch.tensor([0, 1, 2, 0, 1, 2], device=device), rtol=0, atol=0)
    retained = (resident.centroids, resident.obj, resident.assignments)
    saved = tuple(value.clone() for value in retained)
    with check.assertRaises(RuntimeError):
        resident.begin(12, initial[shard])
    check.assertFalse(resident._failed)
    for actual, original, snapshot in zip(
            (resident.centroids, resident.obj, resident.assignments), retained, saved):
        check.assertIs(actual, original)
        torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)

    # Wrong closing row count poisons the run; reset permits real same-model recovery.
    failed = EmeansShardedCentroids(1, 2, group=group, nepochs=1, batch_size=4, alpha0=1)
    ones = torch.ones((2, 1), device=device)
    own_initial = torch.tensor([[1. if rank == 0 else 9.]], device=device)
    failed.begin(4, own_initial)
    failed.update(failed.accumulate(ones))
    failed.end_epoch()
    with check.assertRaisesRegex(ValueError, "cover global_n"):
        failed.final_closing_mean([ones[:1]])
    with check.assertRaises(RuntimeError):
        failed.accumulate(ones)
    failed = EmeansShardedCentroids(
        1, 2, group=group, nepochs=1, batch_size=4, alpha0=1
    )
    failed.begin(4, own_initial)
    failed.update(failed.accumulate(ones))
    failed.end_epoch()
    torch.testing.assert_close(failed.final_closing_mean([ones]), own_initial, rtol=0, atol=0)

    raw_supported = (not torch.version.hip and torch.cuda.get_device_capability(device)
                     in ((8, 0), (9, 0), (10, 0)))
    if importlib.util.find_spec("triton") is None:
        print("sharder raw checks unavailable: Triton not installed", flush=True)
    elif not raw_supported:
        rejected = EmeansShardedCentroids(1, 2, group=group, assignment_backend="triton")
        rejected.begin(4, own_initial)
        with check.assertRaisesRegex(RuntimeError, "INT8 assignment requires"):
            rejected.accumulate(ones.to(torch.int8))
    else:
        for backend in (("triton", "cute") if _ready() else ("triton",)):
            for dtype, offset in ((torch.int8, 0), (torch.uint8, 128)):
                for tie in (False, True):
                    d = 8192
                    rows = torch.full((2, d), offset, dtype=dtype, device=device)
                    centers = torch.full((2, d), float(offset), device=device)
                    centers[:, :1024] -= 128
                    if not tie:
                        centers[0, 1024] += 1
                    distances = (centers.long() - rows[0].long()).square().sum(1)
                    oracle = int(distances.argmin())  # Exact ties choose the lowest global ID.
                    assert distances.tolist() == ([2**24, 2**24] if tie else [2**24 + 1, 2**24])
                    raw = EmeansShardedCentroids(
                        d,
                        2,
                        group=group,
                        nepochs=1,
                        batch_size=4,
                        alpha0=1,
                        capture_assignments=True,
                        assignment_backend=backend,
                    )
                    raw.begin(4, centers[rank:rank + 1])
                    stats = raw.accumulate(rows)
                    torch.testing.assert_close(stats["labels"], torch.full((2,), oracle, device=device), rtol=0, atol=0)
                    count = 4 if rank == oracle else 0
                    torch.testing.assert_close(stats["sums"], torch.full((1, d), count * offset, dtype=torch.int64, device=device), rtol=0, atol=0)
                    assert stats["counts"].dtype == torch.int64 and stats["counts"].item() == count
                    saved_labels = stats["labels"].clone()
                    raw.update(stats)
                    raw.end_epoch()
                    frozen = raw.current_centroids().clone()
                    result = raw.finalize()
                    expected_raw = rows[:1].float() if rank == oracle else frozen
                    torch.testing.assert_close(result, expected_raw, rtol=0, atol=0)
                    torch.testing.assert_close(stats["labels"], saved_labels, rtol=0, atol=0)
                    # External INT64 stats fix the schema, not signed/unsigned
                    # row metadata. The first raw closing rows must prepare the
                    # matching centroid operand without private priming.
                    raw = EmeansShardedCentroids(
                        d,
                        2,
                        group=group,
                        nepochs=1,
                        batch_size=4,
                        alpha0=1,
                        capture_assignments=True,
                        assignment_backend=backend,
                    )
                    raw.begin(4, centers[rank:rank + 1])
                    raw.update({
                        "sums": torch.full((1, d), count * offset, dtype=torch.int64, device=device),
                        "counts": torch.tensor([count], dtype=torch.int64, device=device),
                        "objective": distances[oracle].double() * count,
                        "n": 4,
                    })
                    raw.end_epoch()
                    scanned = raw.final_closing_mean([rows[:1], rows[1:]])
                    torch.testing.assert_close(scanned, expected_raw, rtol=0, atol=0)
                    print("sharder raw exact: %s %s tie=%s" % (backend, dtype, tie), flush=True)


def _saved_closing_contracts(group, device, quantizer=None, backend='native'):
    from faiss.contrib.e_means import EmeansShardedCentroids

    rank = dist.get_rank(group)
    check = unittest.TestCase()
    initial = torch.tensor([[0.], [4.], [100.], [200.]], device=device)
    local = torch.tensor(([0., 3., 6.] if rank == 0 else [1., 2., 5.]), device=device)[:, None]
    if quantizer is not None:
        local = local + 2**-10
        if rank == 1:
            local = quantizer.compute_codes(local)
    model = None
    shard = slice(2 * rank, 2 * rank + 2)

    def train():
        nonlocal model
        model = EmeansShardedCentroids(
            1, 4, group=group, nepochs=1, batch_size=4, alpha0=1,
            assignment_chunk_size=1, capture_assignments=True,
            quantizer=quantizer, assignment_backend=backend,
        )
        model.begin(6, initial[shard])
        saved = []
        for begin in (0, 2):
            stats = model.accumulate(local[begin:begin + 2])
            saved.append(stats["labels"].cpu().to(torch.int32))
            model.update(stats)
        model.end_epoch()
        return torch.cat(saved)

    codes = train()
    labels = [torch.empty(3, device=device, dtype=torch.int32) for _ in range(2)]
    represented = local if quantizer is None else quantizer.decode(
        local if local.dtype == quantizer.dtype else quantizer.compute_codes(local))
    rows = [torch.empty_like(represented) for _ in range(2)]
    dist.all_gather(labels, codes.to(device), group=group)
    dist.all_gather(rows, represented, group=group)
    labels, rows = torch.cat(labels).long(), torch.cat(rows)
    sums = torch.zeros_like(initial)
    sums.index_add_(0, labels, rows)
    counts = torch.bincount(labels, minlength=4)
    # Deliberately move the completed snapshot: means must still use old codes.
    model._centroids.add_(1000)
    expected = torch.where(
        (counts[shard] > 0)[:, None],
        sums[shard] / counts[shard].clamp_min(1)[:, None], model._centroids,
    )
    def no_search(*args, **kwargs):
        raise AssertionError("saved closing must not run assignment")
    model._prepare_assignment = model._local_assignment_batch = model._reduce_winners = no_search
    result = model.final_closing_mean(
        [local[:1], local[1:]], assignments=codes, chunk_rows=1,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    global_mean = sums / counts.clamp_min(1)[:, None]
    mse = (rows - global_mean[labels]).square().sum().item() / len(rows)
    check.assertAlmostEqual(model.closing_stats["assignment_mse"], mse, places=6)
    assert model.closing_stats["rows"] == 6 and model.closing_stats["saved_assignments"]
    # Rank 1 owns only empty clusters and must survive zero-length receives.
    assert counts[2:].sum().item() == 0

    del model._prepare_assignment, model._local_assignment_batch, model._reduce_winners
    codes = train()
    with check.assertRaisesRegex(ValueError, "counts differ"):
        model.final_closing_mean([local], assignments=torch.zeros_like(codes), chunk_rows=1)
    assert model._failed and model.centroids is None

    # A short final epoch leaves uncoded rows; all ranks still route empty blocks.
    model = EmeansShardedCentroids(
        1, 4, group=group, nepochs=0.5, batch_size=4, alpha0=1,
        capture_assignments=True,
        quantizer=quantizer, assignment_backend=backend,
    )
    local = torch.tensor([0., 4., 50., 60.], device=device)[:, None]
    if quantizer is not None:
        local = local + 2**-10
        if rank == 1:
            local = quantizer.compute_codes(local)
    model.begin(8, initial[shard])
    stats = model.accumulate(local[:2])
    codes = torch.full((4,), -1, dtype=torch.int32)
    codes[:2] = stats["labels"].cpu().to(torch.int32)
    model.update(stats)
    model.end_epoch()
    expected = model.current_centroids().clone()
    model._prepare_assignment = model._local_assignment_batch = model._reduce_winners = no_search
    torch.testing.assert_close(
        model.final_closing_mean([local], assignments=codes, chunk_rows=1),
        expected, rtol=0, atol=0,
    )
    assert model.closing_stats["rows"] == 4


def _all_gather_contracts(group, device):
    from unittest.mock import patch
    from faiss.contrib.e_means import EmeansShardedCentroids, EmeansShardedCentroidsIVF

    rank = dist.get_rank(group)
    generator = torch.Generator(device=device).manual_seed(37)
    # INT8 assignment needs an NVIDIA SM80, SM90 or SM100 GPU.
    modes = ["float", "ivf", "spherical"]
    if torch.cuda.get_device_capability(device) in ((8, 0), (9, 0), (10, 0)):
        modes += ["int8", "uint8"]
    for mode in modes:
        data = torch.randn((34, 8), device=device, generator=generator)
        if mode in ("int8", "uint8"):
            data = (data * 3).round().to(torch.int8)
            if mode == "uint8":
                data = (data.int() + 128).to(torch.uint8)
        initial = data[:8].float().clone()
        initial[1] = initial[0]  # Exact ties must retain lowest global-ID semantics.
        local = data[rank * 17:(rank + 1) * 17]
        model_type = EmeansShardedCentroidsIVF if mode == "ivf" else EmeansShardedCentroids
        extra = dict(nlist=4, nprobe=2, coarse_niter=2) if mode == "ivf" else {}
        models = [model_type(
            8, 8, group=group, nepochs=2, batch_size=12, alpha0=.3,
            assignment_chunk_size=2, capture_assignments=True,
            spherical=mode == "spherical",
            assignment_backend="triton" if mode in ("int8", "uint8") else "native",
            quantizer=None,
            **extra,
        ) for _ in range(2)]
        if mode == "ivf":
            # The oracle gathers first and lets every owner route the whole
            # batch, as before. Compare that with routing each reader's rows.
            models[1]._gather_assignment_rows = (
                EmeansShardedCentroids._gather_assignment_rows.__get__(models[1])
            )
        for model in models:
            model.begin(34, initial[rank * 4:(rank + 1) * 4])
            model._prepare_input(local)
        retained = []
        for epoch in range(2):
            order = torch.randperm(17, device=device, generator=generator)
            for begin in range(0, 17, 6):
                index = order[begin:begin + 6]
                outputs = []
                for i, model in enumerate(models):
                    # Compare partial with whole indexed accumulation, including tails.
                    partial = model._accumulate(local, index[:2]) if i == 0 else None
                    stats = model._accumulate(local, index[2:] if i == 0 else index)
                    outputs.append({key: stats[key].clone() for key in ("sums", "counts", "objective")})
                    if epoch == 1:
                        labels = (torch.cat((partial["labels"], stats["labels"]))
                                  if partial is not None else stats["labels"])
                        outputs[-1]["labels"] = labels
                        retained.append((stats["labels"], stats["labels"].clone()))
                    model.update(stats)
                for key in outputs[0]:
                    torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(models[0].current_centroids(), models[1].current_centroids(),
                                           rtol=2e-5, atol=2e-5)
            for model in models:
                model.end_epoch()
        torch.testing.assert_close(models[0].finalize(), models[1].finalize(), rtol=2e-5, atol=2e-5)
        for labels, expected in retained:
            torch.testing.assert_close(labels, expected, rtol=0, atol=0)

    # Isolate row/winner collectives from coarse preparation and revival.
    model = EmeansShardedCentroids(
        8, 8, group=group, nepochs=1, batch_size=12, alpha0=.3, assignment_chunk_size=2,
        assignment_backend="native", quantizer=None,
    )
    rows = torch.randn((6, 8), device=device, generator=generator)
    model.begin(12, rows[rank:rank + 4].clone())
    gather_name = "all_gather_single" if getattr(dist, "all_gather_single", None) else "all_gather_into_tensor"
    real_gather, real_reduce = getattr(dist, gather_name), dist.all_reduce
    calls = {"gather": 0, "reduce": 0}

    def gather(*args, **kwargs):
        calls["gather"] += 1
        return real_gather(*args, **kwargs)

    def reduce(*args, **kwargs):
        calls["reduce"] += 1
        return real_reduce(*args, **kwargs)

    def no_broadcast(*args, **kwargs):
        raise AssertionError("batch broadcast in all-gather path")

    with patch.object(dist, gather_name, new=gather), \
         patch.object(dist, "all_reduce", new=reduce), \
         patch.object(dist, "broadcast", new=no_broadcast):
        model.accumulate(rows)
    assert calls["gather"] == 1 and calls["reduce"] == 2

    # Every reader queries only lists owned by the other centroid shard.
    # Routing must use global occupancy, not discard locally empty lists.
    from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

    initial = torch.tensor(
        [[0., 0.], [1., 0.], [10., 0.], [11., 0.],
         [20., 0.], [21., 0.], [30., 0.], [31., 0.]], device=device,
    )
    local = initial[(1 - rank) * 4:(2 - rank) * 4].clone()
    model = EmeansShardedCentroidsIVF(
        2, 8, group=group, nepochs=1, batch_size=8, alpha0=.3,
        nlist=4, nprobe=1, assignment_backend="native",
    )
    model.begin(8, initial[rank * 4:(rank + 1) * 4])
    model._ivf = IndexIVFFlat(initial[::2].clone(), nprobe=1)
    model._prepare_input(local)
    prepared = model._prepare_assignment()
    full = model._gather_training_rows(local)
    expected_routing = prepared.route(full).to(torch.int32)
    distance, labels = prepared.search_1nn(full)
    expected_winner = model._reduce_winners(distance, labels)
    real_route, real_search = prepared.route, prepared.search_1nn
    calls = {"gather": 0, "reduce": 0, "route": 0, "route_rows": 0}
    routing = None

    def route(rows, *args, **kwargs):
        calls["route"] += 1
        calls["route_rows"] = len(rows)
        return real_route(rows, *args, **kwargs)

    def search(*args, **kwargs):
        nonlocal routing
        routing = kwargs["coarse_idx"].detach().clone()
        return real_search(*args, **kwargs)

    with patch.object(prepared, "route", new=route), \
         patch.object(prepared, "search_1nn", new=search), \
         patch.object(dist, gather_name, new=gather), \
         patch.object(dist, "all_reduce", new=reduce):
        stats = model.accumulate(local)
    assert calls["route"] == 1 and calls["route_rows"] == len(local)
    assert calls["gather"] == 2 and calls["reduce"] == 3
    assert routing.dtype == torch.int32 and routing.shape == (8, 1)
    torch.testing.assert_close(routing, expected_routing, rtol=0, atol=0)
    torch.testing.assert_close(stats["labels"], expected_winner[rank * 4:(rank + 1) * 4])
    torch.testing.assert_close(stats["counts"], torch.ones(4, device=device))
    torch.testing.assert_close(stats["sums"], initial[rank * 4:(rank + 1) * 4])


def _revival_worker(rank, world_size, rendezvous, backend):
    from unittest import mock
    from faiss.contrib.e_means import EmeansShardedCentroids
    from faiss.contrib.e_means.torch_backend import TorchBackend, TritonBackend

    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    dist.init_process_group(backend, init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=datetime.timedelta(seconds=60))
    try:
        # Unique sources/targets cross owners in both directions. The first
        # and last chunks have empty senders and empty receivers; the middle
        # chunk exercises mixed ownership and a partial tail for chunk_rows=4.
        source = torch.tensor([0, 2, 3, 10, 11, 14], device=device)
        target = torch.tensor([8, 9, 15, 1, 5, 6], device=device)
        initial = torch.tensor([8., 0.001, -2., 1.], device=device).repeat(16, 1)
        initial += torch.arange(16, device=device)[:, None] * 1e-5
        usage = torch.arange(1, 17, device=device, dtype=torch.float32)
        shard = slice(rank * 8, (rank + 1) * 8)
        cases = [(dtype, None, TorchBackend) for dtype in (None, torch.float16, torch.bfloat16)]
        if backend == "nccl":
            from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer

            q = Fp8Quantizer(4)
            q.train(torch.cat((initial - 1, initial + 1)))
            cases.append((q.dtype, q, TritonBackend))
        real_exchange = dist.all_to_all_single
        for dtype, quantizer, owner in cases:
            # An independent full-table split is the arithmetic oracle. In
            # particular, low-precision collapsed pairs must stay separated.
            oracle = QuantizedEmeans(4, 16, quantizer=quantizer, assignment_backend=owner.name)
            oracle._execution_backend = owner(oracle.centroid_chunk_size)
            oracle._centroids, oracle._usage = initial.clone(), usage.clone()
            oracle._input_dtype = dtype
            oracle._apply_splits(source, target)
            for chunk_rows in ((1, 2, 4, 100) if quantizer is None else (2,)):
                model = EmeansShardedCentroids(4, 16, group=dist.group.WORLD,
                                               quantizer=quantizer, assignment_backend=owner.name)
                model._execution_backend = owner(model.centroid_chunk_size)
                model._centroids, model._usage = initial[shard].clone(), usage[shard].clone()
                model._revival_dtype = dtype
                exchanges = [0]

                def counted(output, rows, *args, **kwargs):
                    if rows.ndim == 2:
                        assert len(rows) <= chunk_rows and len(output) <= chunk_rows
                    exchanges[0] += 1
                    return real_exchange(output, rows, *args, **kwargs)

                with mock.patch.object(model, "_choose_splits", return_value=(source, target)) as choose, \
                        mock.patch.object(dist, "all_to_all_single", new=counted):
                    assert model._revive_unused(torch.Generator(), chunk_rows=chunk_rows) == len(source)
                assert choose.call_count == (1 if rank == 0 else 0)
                assert model.n_splits == len(source)
                torch.testing.assert_close(model._centroids, oracle._centroids[shard], rtol=0, atol=0)
                torch.testing.assert_close(model._usage, oracle._usage[shard], rtol=0, atol=0)
                assert exchanges[0] == 3 * ((len(source) + chunk_rows - 1) // chunk_rows)
                # A later empty revival must leave the completed split alone.
                model._usage.fill_(1)
                before = model._centroids.clone()
                with mock.patch.object(dist, "all_to_all_single", new=counted):
                    assert model._revive_unused(torch.Generator(), chunk_rows=chunk_rows) == 0
                assert exchanges[0] == 3 * ((len(source) + chunk_rows - 1) // chunk_rows)
                assert model.n_splits == len(source)
                torch.testing.assert_close(model._centroids, before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _shared_codec_contracts(group, device):
    """Exercise the new owned-code wire; local/replica codec laws live elsewhere."""
    from faiss.contrib.e_means import EmeansShardedCentroids
    from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer

    check, rank = unittest.TestCase(), dist.get_rank(group)
    base = torch.tensor([1+2**-10, 1+2**-9, 3+2**-10, 3+2**-9], device=device)[:, None].repeat(3, 1)
    anchors = torch.tensor([1., 3., 4., 8.], device=device)[:, None]
    cases = [('default', 'native'), ('default_mixed', 'native'),
             ('fp32', 'native'), ('fp16', 'native'), ('bf16', 'native'),
             ('cast_fp16', 'native'), ('cast_bf16', 'native'),
             ('center_fp16', 'native'), ('center_bf16', 'native'),
             ('fp8', 'triton'), ('fp8', 'cute'), ('native_fp8', 'triton'),
             ('int8', 'triton'), ('raw_int8', 'triton'), ('raw_uint8', 'triton')]
    for name, backend in cases:
        with check.subTest(codec=name, backend=backend):
            rows, initial, q = base.clone(), anchors.clone(), None
            if name.startswith(('cast_', 'center_')):
                dtype = torch.float16 if name.endswith('fp16') else torch.bfloat16
                q = CastQuantizer(1, dtype, center=name.startswith('center_'))
                if q.center:
                    rows, initial = rows + 100, initial + 100
                    q.train(torch.full((2, 1), 100., device=device))
            elif name in ('fp8', 'int8'):
                q = Fp8Quantizer(1) if name == 'fp8' else Int8Quantizer(1)
                q.train(torch.cat((rows, initial)))
                if name == 'int8' or rank == 1:
                    rows = q.compute_codes(rows)
            elif name in ('raw_int8', 'raw_uint8'):
                dtype = torch.int8 if name == 'raw_int8' else torch.uint8
                offset = 0 if name == 'raw_int8' else 128
                rows, initial = (rows + offset).to(dtype), initial + offset
            elif not name.startswith('default'):
                rows = rows.to({'fp32': torch.float32, 'fp16': torch.float16,
                                'bf16': torch.bfloat16, 'native_fp8': torch.float8_e4m3fn}[name])
            options = {} if name.startswith('default') else {'quantizer': q}
            if name == 'default_mixed' and rank == 1:
                rows = rows.bfloat16()
            model = EmeansShardedCentroids(1, 4, group=group, nepochs=2, batch_size=8,
                                           alpha0=.5, seed=7, assignment_chunk_size=6,
                                           assignment_backend=backend, **options)
            model.begin(24, initial[2*rank:2*rank+2])
            model.accumulate(rows[:0])
            values_seen = []
            original = model._add_owned

            def observed(values, winner, sums, counts, objective, *extra):
                values_seen.append((values.clone(), winner.clone(), model._centroids.clone()))  # noqa: B023
                return original(values, winner, sums, counts, objective, *extra)  # noqa: B023

            def totals(dtype):
                sums = torch.zeros((2, 1), dtype=dtype, device=device)
                counts = torch.zeros(2, dtype=dtype, device=device)
                objective = torch.zeros((), dtype=torch.float64, device=device)
                for values, winner, snapshot in values_seen:  # noqa: B023
                    check.assertTrue(bool(((winner >= 0) & (winner < 4)).all()))
                    for local in range(2):
                        selected = values[winner == 2*rank+local].to(dtype)
                        sums[local] += selected.sum(0)
                        counts[local] += len(selected)
                        objective += (selected.float()-snapshot[local]).square().sum(dtype=torch.float64)
                return sums, counts, objective

            def represented(source):
                if q is not None:  # noqa: B023
                    return q.decode(source if source.dtype == q.dtype else q.compute_codes(source))  # noqa: B023
                return source.bfloat16().float() if name.startswith('default') else source.float()  # noqa: B023

            with mock.patch.object(model, '_add_owned', new=observed):
                for epoch in range(2):
                    for begin in (0, 4, 8):
                        values_seen.clear()
                        source = rows[begin:begin+4]
                        if name == 'default_mixed' and begin:
                            source = source.to(torch.float16 if rank == 0 else torch.float32)
                        partial = model.accumulate(source[:1])
                        check.assertIsNone(partial['sums'])
                        stats = model.accumulate(source[1:])
                        dtype = torch.int64 if name.startswith('raw_') else torch.float32
                        check.assertEqual(stats['n'], 8)
                        check.assertEqual(stats['sums'].dtype, dtype)
                        expected = totals(dtype)
                        for field, wanted in zip(('sums', 'counts', 'objective'), expected):
                            torch.testing.assert_close(stats[field], wanted, rtol=0, atol=0)
                        # Widen native FP8 BEFORE sorting; Torch cannot sort FP8.
                        consumed = torch.cat([v for v, _, _ in values_seen]).to(dtype).flatten().sort().values
                        wanted = represented(source).to(dtype).repeat(2, 1).flatten().sort().values
                        torch.testing.assert_close(consumed, wanted, rtol=0, atol=0)
                        held = None if epoch == 0 else stats['labels'].clone()
                        model.update(stats)
                        if held is not None:
                            torch.testing.assert_close(stats['labels'], held, rtol=0, atol=0)
                        _detached(model)
                    model.end_epoch()
                check.assertGreater(model.n_splits, 0)
                frozen = model.current_centroids().clone()
                values_seen.clear()
                final = model.final_closing_mean([rows])
                sums, counts, _ = totals(dtype)
                mean = (sums.double()/counts.clamp_min(1)[:, None]).float() if dtype is torch.int64 else sums/counts.clamp_min(1)[:, None]
                wanted = torch.where((counts > 0)[:, None], mean, frozen)
                torch.testing.assert_close(final, wanted, rtol=0, atol=0)
                consumed = torch.cat([v for v, _, _ in values_seen]).to(dtype).flatten().sort().values
                wanted = represented(rows).to(dtype).repeat(2, 1).flatten().sort().values
                torch.testing.assert_close(consumed, wanted, rtol=0, atol=0)
                check.assertEqual(model.closing_stats['rows'], 24)
                if name == 'center_bf16':
                    saved_final, saved_closing = final.clone(), dict(model.closing_stats)
                    # Used-run rejection precedes any explicit-fit communication.
                    with mock.patch('faiss.contrib.e_means.sharded_centroids.check_codec_fit',
                                    side_effect=AssertionError('repeat begin entered fit agreement')):
                        with check.assertRaises(RuntimeError):
                            model.begin(24, initial[2*rank:2*rank+2])
                    check.assertFalse(model._failed)
                    check.assertIs(model.centroids, final)
                    check.assertEqual(model.closing_stats, saved_closing)
                    torch.testing.assert_close(final, saved_final, rtol=0, atol=0)
    # Extend the existing no-search/-1/empty-receiver closing fixture, not a
    # second private model simulator or full codec/replica Cartesian product.
    centered = CastQuantizer(1, torch.bfloat16, center=True)
    centered.train(torch.full((2, 1), 100., device=device))
    fp8 = Fp8Quantizer(1)
    fp8.train(torch.tensor([[0.], [4.], [100.], [200.]], device=device))
    for codec, backend in ((CastQuantizer(1, torch.bfloat16), 'native'),
                           (centered, 'native'), (fp8, 'cute')):
        _saved_closing_contracts(group, device, codec, backend)
    unequal = CastQuantizer(1, torch.bfloat16, center=True)
    unequal.train(torch.full((2, 1), float(rank), device=device))
    model = EmeansShardedCentroids(1, 4, group=group, quantizer=unequal, nepochs=1, batch_size=8)
    with check.assertRaises(ValueError):
        model.begin(24, anchors[2*rank:2*rank+2])
    check.assertTrue(model._failed)
    # Generated seeds must not touch a bad local codec before fit agreement.
    codec = Fp8Quantizer(1) if rank == 0 else fp8
    model = EmeansShardedCentroids(1, 4, group=group, quantizer=codec,
                                   assignment_backend='triton', nepochs=1, batch_size=8)
    with check.assertRaises(ValueError):
        model.train(base)  # Omitted initialization; same existing rows/fit.
    check.assertTrue(model._failed)
    model = EmeansShardedCentroids(1, 4, group=group, nepochs=1, batch_size=8)
    model.begin(24, anchors[2*rank:2*rank+2])
    with check.assertRaises(ValueError):
        model.accumulate(base[:0].to(torch.float16 if rank == 0 else torch.bfloat16))
    check.assertTrue(model._failed)
    with mock.patch.object(model, '_gather_local',
                           side_effect=AssertionError('failed train entered first-input collective')):
        with check.assertRaises(RuntimeError):
            model.train(base, init_centroids=anchors[2*rank:2*rank+2])
    check.assertTrue(model._failed)


def _replica_default_contracts(group, device):
    """Representative changed default; parent replica native/bytes laws remain."""
    rows = torch.tensor([1+2**-10, 1+2**-9, 3+2**-10, 3+2**-9], device=device)[:, None]
    initial = torch.tensor([[1.], [3.]], device=device)
    for options in ({}, {'quantizer': CastQuantizer(1, torch.bfloat16)}, {'quantizer': None}):
        args = dict(nepochs=1, batch_size=8, alpha0=.5, assignment_backend='native', **options)
        model = DistributedEmeans(1, 2, group=group, transport='dense', **args)
        reference = QuantizedEmeans(1, 2, **args)
        for candidate in (model, reference):
            candidate.begin(8, initial)
        partial = model.accumulate(rows[:1])
        assert partial['sums'] is None
        model.update(model.accumulate(rows[1:]))
        reference.update(reference.accumulate(torch.cat((rows, rows))))
        model.end_epoch()
        reference.end_epoch()
        final = model.finalize()
        torch.testing.assert_close(final, reference.finalize(), rtol=2e-6, atol=2e-6)
        native = 'quantizer' in options and options['quantizer'] is None
        represented = rows if native else rows.bfloat16().float()
        wanted = torch.stack((represented[:2].mean(0), represented[2:].mean(0)))
        torch.testing.assert_close(final, wanted, rtol=0, atol=0)
        torch.testing.assert_close(model.obj, reference.obj, rtol=1e-6, atol=1e-5)
    centre = CastQuantizer(1, torch.bfloat16, center=True)
    centre.train(initial)  # Common semantic fit; existing provided GPU rows.
    if dist.get_rank(group) == 0:
        centre.shift.add_(0)  # Same values, but this local frozen-fit token is stale.
    model = DistributedEmeans(1, 2, group=group, transport='dense', quantizer=centre,
                              nepochs=1, batch_size=8, alpha0=.5, assignment_backend='native')
    with unittest.TestCase().assertRaises(ValueError):
        model.begin(8, initial)
    assert model._failed and model.assignments is None


def _nccl_worker(rank, world_size, rendezvous, subgroup_only, sharder_only=False):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    subgroup = None
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=datetime.timedelta(seconds=60))
    try:
        if sharder_only == 'codecs':
            _shared_codec_contracts(dist.group.WORLD, device)
            _replica_default_contracts(dist.group.WORLD, device)
            return
        if sharder_only == "transport":
            _all_gather_contracts(dist.group.WORLD, device)
            return
        if sharder_only == "saved":
            _saved_closing_contracts(dist.group.WORLD, device)
            return
        if sharder_only:
            _sharder_contracts(dist.group.WORLD, device)
        elif subgroup_only:
            # Every world rank creates the group; global rank0 is not its root
            # and never participates in any of its training collectives.
            subgroup = dist.new_group([1, 2], timeout=datetime.timedelta(seconds=60))
            if rank in (1, 2):
                for transport in ("dense", "sparse"):
                    _global_totals(subgroup, transport, device)
                    _training(subgroup, transport, device)
                if torch.cuda.get_device_capability(device) in ((8, 0), (9, 0), (10, 0)):
                    _training(subgroup, "sparse", device, backend="triton", byte_dtype=torch.uint8)
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
            # One represented family through both packing/merge paths; avoid
            # repeating the assignment-backend/dtype Cartesian product.
            for transport in ("dense", "sparse"):
                _training(dist.group.WORLD, transport, device,
                          quantizer=CastQuantizer(2, torch.bfloat16))
                _training(dist.group.WORLD, transport, device, backend="triton")
                if torch.cuda.get_device_capability(device) in ((8, 0), (9, 0), (10, 0)):
                    _integer_external_totals(dist.group.WORLD, transport, device)
                    for dtype in (torch.int8, torch.uint8):
                        _training(dist.group.WORLD, transport, device,
                                  backend="triton", byte_dtype=dtype)
                    q = Int8Quantizer(2)
                    q.train(_rows(0, 24, device))
                    _training(dist.group.WORLD, transport, device,
                              backend="triton", quantizer=q)
    finally:
        if subgroup_only and rank in (1, 2) and subgroup is not None:
            dist.destroy_process_group(subgroup)
        dist.destroy_process_group()


class TestDistributedEmeans(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_two_cpu_sharded_revival(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _revival_worker, args=(2, "file://" + directory + "/group", "gloo"),
                nprocs=2, join=True,
            )

    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2, "NCCL and two CUDA GPUs required")
    @unittest.skipUnless(_nvidia_sm80_triton(), "NVIDIA SM80 or later GPU and Triton required")
    def test_two_gpu_sharded_revival(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _revival_worker, args=(2, "file://" + directory + "/group", "nccl"),
                nprocs=2, join=True,
            )

    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2, "NCCL and two CUDA GPUs required")
    @unittest.skipUnless(_nvidia_sm80_triton(), "NVIDIA SM80 or later GPU and Triton required")
    def test_two_gpu_shared_codec_values_and_closing(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(_nccl_worker,
                                        args=(2, 'file://' + directory + '/group', False, 'codecs'), nprocs=2, join=True)

    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2, "NCCL and two CUDA GPUs required")
    @unittest.skipUnless(_nvidia_sm80_triton(), "NVIDIA SM80 or later GPU and Triton required")
    def test_two_gpu_all_gather_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _nccl_worker,
                args=(2, "file://" + directory + "/group", False, "transport"),
                nprocs=2, join=True,
            )

    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2, "NCCL and two CUDA GPUs required")
    def test_two_gpu_saved_closing_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _nccl_worker,
                args=(2, "file://" + directory + "/group", False, "saved"),
                nprocs=2, join=True,
            )

    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2, "NCCL and two CUDA GPUs required")
    @unittest.skipUnless(_nvidia_sm80_triton(), "NVIDIA SM80 or later GPU and Triton required")
    def test_two_gpu_sharded_centroids(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _nccl_worker,
                args=(2, "file://" + directory + "/group", False, True),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(dist.is_available() and dist.is_nccl_available()
                         and torch.cuda.device_count() >= 2, "NCCL and two CUDA GPUs required")
    @unittest.skipUnless(_nvidia_sm80_triton(), "NVIDIA SM80 or later GPU and Triton required")
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
    @unittest.skipUnless(_nvidia_sm80_triton(), "NVIDIA SM80 or later GPU and Triton required")
    def test_three_gpu_nonzero_global_root(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _nccl_worker,
                args=(3, "file://" + directory + "/group", True),
                nprocs=3,
                join=True,
            )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and (getattr(torch.version, "hip", None) is not None
             or torch.cuda.get_device_capability() < (8, 0)),
        "a GPU other than NVIDIA SM80 or later required",
    )
    def test_bf16_without_sm80_names_the_cause(self):
        with self.assertRaisesRegex(RuntimeError, "SM80"):
            QuantizedEmeans(2, 1, quantizer=CastQuantizer(2, torch.bfloat16)).train(
                torch.zeros((2, 2), device="cuda"))

    def test_auto_sharder_chunk_bound_is_independent_of_compute_backend(self):
        from types import SimpleNamespace
        from faiss.contrib.e_means import Emeans, EmeansShardedCentroids
        from faiss.contrib.e_means.torch_backend import TorchBackend, TritonBackend, CuteBackend

        state = SimpleNamespace(device=torch.device("cuda"), is_cuda=True, shape=(4096, 128))
        native = Emeans(128, 4096, assignment_backend="native")
        native._centroids = state
        model = EmeansShardedCentroids.__new__(EmeansShardedCentroids)
        model.d, model.k, model.centroid_chunk_size = native.d, native.k, native.centroid_chunk_size
        model.assignment_chunk_size = native.assignment_chunk_size
        model._centroids, model.centroids = state, None
        model.assignment_backend = "auto"
        expected_native_bound = native._get_assignment_chunk_size()
        for cls in (TorchBackend, TritonBackend, CuteBackend):
            model._execution_backend = cls(model.centroid_chunk_size)
            self.assertEqual(model._get_assignment_chunk_size(), expected_native_bound)
        for requested, cls in (("native", TorchBackend), ("triton", TritonBackend), ("cute", CuteBackend)):
            control = Emeans(native.d, native.k, assignment_backend=requested,
                             centroid_chunk_size=native.centroid_chunk_size)
            control._centroids = state
            control._execution_backend = cls(control.centroid_chunk_size)
            model.assignment_backend = requested
            model._execution_backend = cls(model.centroid_chunk_size)
            self.assertEqual(model._get_assignment_chunk_size(), control._get_assignment_chunk_size())
        model.assignment_backend, model.assignment_chunk_size = "auto", 37
        self.assertEqual(model._get_assignment_chunk_size(), 37)


if __name__ == "__main__":
    unittest.main()
