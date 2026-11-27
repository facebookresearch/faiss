# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-process balanced E-means: the global capacity cap equals one process.

The CPU cases spawn Gloo processes and run anywhere; the GPU cases need two
or more CUDA devices and NCCL.
"""
import datetime
import os
import tempfile
import unittest

import torch
import torch.distributed as dist

from faiss.contrib.e_means.e_means_balanced import EmeansBalanced
from faiss.contrib.e_means.e_means_balanced_distributed import (
    DistributedEmeansBalanced, cap_rounds, reduce_top2)
from faiss.contrib.e_means.e_means_balanced_sharded import EmeansBalancedSharded

N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
NCCL = dist.is_available() and dist.is_nccl_available() and N_GPUS >= 2


def _problem(seed, B, k, d, skew, device):
    """Rows, centroids and a bias that piles rows onto a few clusters."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(k, d, generator=g)
    x = torch.randn(B, d, generator=g)
    bias = torch.zeros(k)
    bias[:max(1, k // 16)] = skew  # a few greedy clusters swallow rows
    if seed >= 100:  # half-integer grid: many exactly tied margins and scores
        x, c = torch.round(x * 2) / 2, torch.round(c * 2) / 2
    return x.to(device), c.to(device), bias.to(device)


def _skew(seed):
    return 30.0 if seed % 2 == 0 else 1.5  # swallowers, or mild pile-ups


def _scores(x, c, bias):
    return (x[:, None, :] - c[None]).square().sum(2) - bias  # (n, k)


def _top(x, c, bias, m):
    v, i = torch.sort(_scores(x, c, bias), dim=1, stable=True)  # ties: lower id
    return v[:, :m], i[:, :m]


def _masked(c, bias):
    def assign(rows, full):
        s = _scores(rows, c, bias)
        s[:, full] = float("inf")
        v, i = torch.sort(s, dim=1, stable=True)
        return i[:, 0], v[:, 0]
    return assign


def _independent_rounds(x, codes, margin, second, second_val, cap, k, n_rounds, assign):
    """An independent implementation of the rounds cap (the research code's), for
    distinct margins and scores."""
    b = len(codes)
    first = codes.clone()
    cl = codes.clone()
    lo = margin.double().min()
    span = margin.double().max() - lo + 1.0
    order = torch.argsort(cl.double() * (2.0 * span) - (margin.double() - lo))
    cs = cl[order]
    rank = torch.arange(b) - torch.searchsorted(cs, cs)
    move = rank >= cap
    codes = codes.clone()
    codes[order] = torch.where(move, second[order], codes[order])
    prio = torch.where(codes == first, 0.0, 1.0).double()
    score = second_val.double().clone()
    for _ in range(n_rounds):
        cl = codes.clone()
        cnt = torch.bincount(cl, minlength=k)
        if int(cnt.max()) <= cap:
            break
        lo = score.min()
        span = score.max() - lo + 1.0
        key = cl.double() * (4.0 * span) + prio * (2.0 * span) + (score - lo)
        order = torch.argsort(key)
        cs = cl[order]
        rank = torch.arange(b) - torch.searchsorted(cs, cs)
        evict = order[rank >= cap]
        if evict.numel() == 0:
            break
        kept = torch.bincount(cl[order[rank < cap]], minlength=k)
        out, val = assign(x[evict], kept >= cap)
        codes[evict] = out
        prio[evict] = 1.0
        score[evict] = val.double()
    return codes


def _rounds_case(seed, B, k, d, cap, n_rounds, device, group):
    x, c, bias = _problem(seed, B, k, d, _skew(seed), device)
    v, i = _top(x, c, bias, 2)
    world = 1 if group is None else dist.get_world_size(group)
    rank = 0 if group is None else dist.get_rank(group)
    b = B // world
    sl = slice(rank * b, (rank + 1) * b)
    stats = {}
    out = cap_rounds(x[sl], i[sl, 0], v[sl, 1] - v[sl, 0], i[sl, 1], v[sl, 1], cap, k,
                     n_rounds, _masked(c, bias), group, stats)
    return out, stats


CASES = [  # seed, B, k, d, cap, rounds
    (0, 4096, 64, 8, 80, 2),
    (1, 4096, 256, 8, 20, 2),
    (2, 2048, 512, 4, 5, 4),
    (3, 4096, 1024, 4, 8, 3),
    (100, 4096, 128, 2, 40, 2),
    (101, 2048, 256, 3, 10, 3),
]


def _worker(rank, world, rendezvous, backend):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    group = dist.group.WORLD
    try:
        for seed, B, k, d, cap, extra in CASES:
            got, _ = _rounds_case(seed, B, k, d, cap, extra, device, group)
            ref, _ = _rounds_case(seed, B, k, d, cap, extra, device, None)
            b = B // world
            assert torch.equal(got, ref[rank * b:(rank + 1) * b]), (seed, "rounds")
        # sharded: per-shard top-2 reduced to the global top-2
        x, c, bias = _problem(5, 1024, 64, 8, 3.0, device)
        x = torch.round(x * 2) / 2  # ties between shards
        c = torch.round(c * 2) / 2
        kl = 64 // world
        begin = rank * kl
        v, i = _top(x, c[begin:begin + kl], bias[begin:begin + kl], 2)
        best, win, sv, si = reduce_top2(v[:, 0], i[:, 0], v[:, 1], i[:, 1], begin, kl, group)
        rv, ri = _top(x, c, bias, 2)
        assert torch.equal(best, rv[:, 0]) and torch.equal(win, ri[:, 0])
        assert torch.equal(sv, rv[:, 1]) and torch.equal(si, ri[:, 1])
    finally:
        dist.destroy_process_group()


def _integer_problem(n, k, d, seed, spread=6):
    """Small integer rows, centroids and bias: every score is exact in FP32."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-spread, spread + 1, (n, d), generator=g).float()
    c = torch.randint(-spread, spread + 1, (k, d), generator=g).float()
    b = torch.randint(-8, 9, (k,), generator=g).float() * 4
    return x, c, b


def _blobs(n, d, k, seed):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(k, d, generator=g) * 4
    weights = torch.arange(1, k + 1, dtype=torch.float64) ** -1.0
    comp = torch.multinomial(weights, n, replacement=True, generator=g)
    return means[comp] + torch.randn(n, d, generator=g)


def _gather(t, group):
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, t.contiguous(), group=group)
    return parts


def _start(model, x, c, b):
    model.begin(len(x) if not hasattr(model, "_group") else len(x), c)
    model._resolve_input_dtype(x)
    model._resolve_assignment_backend(x)
    model.bias.copy_(b)
    model._mu.copy_(b.clamp(min=0))
    model._lam.copy_((-b).clamp(min=0))


def _step(model, x, index, step):
    """One eager step of the single-process class at global step ``step``."""
    training, scale, alpha, cap = model._schedule(step)
    return model._train_step(x, index, training, scale, alpha, cap, model._revives(step), None)


def _steps_case(group, device, sharded=False, dtype=torch.float32, **extra):
    """Several capped steps (training, a revival, the polish) from identical
    integer state: R ranks take exactly the decisions of one process. On CUDA
    the data-parallel class runs the fused global cap, whose search breaks ties
    to the lower index; so does the Triton assignment the reference uses there
    (PyTorch's argmin does not promise which tied index it returns)."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    n, k, d = 2048, 64, 4
    x, c, b = (t.to(device) for t in _integer_problem(n, k, d, 7))
    x = x.to(dtype)  # small integers: exact in BF16
    kw = dict(nepochs=8, polish_epochs=2, batch_size=n, alpha0=0.5, seed=0, cap_factor=1.0,
              **dict(dict(assignment_backend="triton" if device.type == "cuda" else "native"),
                     **extra))
    ref = EmeansBalanced(d, k, **kw)
    cls = EmeansBalancedSharded if sharded else DistributedEmeansBalanced
    ours = cls(d, k, group=group, **kw)
    _start(ref, x, c, b)
    _start(ours, x, c, b)
    block = slice(rank * (k // world), (rank + 1) * (k // world)) if sharded else slice(0, k)
    nl = n // world
    local = torch.arange(rank * nl, (rank + 1) * nl, device=device)
    every = torch.arange(n, device=device)
    for step in range(8):  # 6 training steps (revival after step 4), 2 polish steps
        want = _step(ref, x, every, step)
        got = torch.cat(_gather(ours._step_at(x, local, step), group))
        assert torch.equal(got, want), ("codes", step, int((got != want).sum()))
        for name in ("_usage", "bias", "_lam", "_mu", "_centroids", "_distance_scale"):
            a, r = getattr(ours, name), getattr(ref, name)
            if name == "_centroids":
                r = r[block]
            torch.testing.assert_close(a, r, rtol=1e-6, atol=1e-6, msg=lambda m: name + m)
        assert torch.equal(ours._pass_counts, ref._pass_counts)
    if ref.cap_mode == "rounds":
        assert ref.n_splits + ref.n_inplace > 0, "the case must exercise revival"
        assert ref.cap_stats["evicted"] > 0, "the case must exercise the repair rounds"
    if ref.inplace_quantile is None:
        assert ref.n_splits > 0, "the case must exercise the donors"
    assert ours.n_splits == ref.n_splits and ours.n_inplace == ref.n_inplace
    if device.type == "cuda" and not sharded:
        assert ours._dp_cap is not None, "the fused global cap must have run"


def _drive_like_ranks(model, x, world, init):
    """One process run in the batch order R ranks use (rank-order concatenation
    of every rank's own shuffle), through the class's own step."""
    from faiss.contrib.e_means.base import _epoch_seed
    from faiss.contrib.e_means.e_means_torch import _torch_generator

    n = len(x)
    nl = n // world
    model._resolve_input_dtype(x)
    model._resolve_assignment_backend(x)
    model.begin(n, init)
    x = model._prepare_input(x)
    b = model.recipe["batch_size"] // world
    step = 0
    for epoch, rows in enumerate(model.recipe["epoch_rows"]):
        generator = _torch_generator(x.device, _epoch_seed(model.seed, epoch))
        order = torch.randperm(nl, generator=generator, device=x.device)
        for begin in range(0, rows // world, b):
            index = torch.cat([r * nl + order[begin:begin + b] for r in range(world)])
            _step(model, x, index, step)
            step += 1
            model._step = step
            model._epoch_steps += 1
            model._epoch_rows += len(index)
        model.end_epoch()
    model.finalize()
    return model.centroids


def _metrics(model, x):
    dist_, labels = model.assign(x)
    k = model.k
    counts = torch.bincount(labels, minlength=k).double()
    imbalance = float((counts ** 2).mean() / (len(x) / k) ** 2)
    return float(dist_.double().mean()), imbalance


def _full_run_case(group, device, backend, sharded=False, k=256, device_path=False, **extra):
    """A whole run on R ranks against one process on the same batch order."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    n, d = 65536, 32
    x = _blobs(n, d, 100, 11).to(device)
    nl = n // world
    kw = dict(nepochs=6, polish_epochs=2, batch_size=8192, seed=3, assignment_backend=backend,
              **extra)
    cls = EmeansBalancedSharded if sharded else DistributedEmeansBalanced
    ours = cls(d, k, group=group, **kw)
    ours._allow_device_path = device_path  # CUDA graphs with the collectives captured, or eager
    centroids, labels = ours.train(x[rank * nl:(rank + 1) * nl])
    if sharded:
        centroids = ours.gather_centroids()
    for t in (centroids, ours.bias):
        for other in _gather(t, group):
            assert torch.equal(t, other), "replicas diverged"
    ref = EmeansBalanced(d, k, **kw)
    ref._resolve_input_dtype(x)  # the start is rounded like the rows, as in train()
    init = ref._initial_centroid_values(
        x[torch.randperm(n, generator=torch.Generator(device=device).manual_seed(3)
                         if device.type == "cuda" else torch.Generator().manual_seed(3),
                         device=device)[:k]])
    _drive_like_ranks(ref, x, world, init)
    mse, imb = _metrics(ours, x)
    mse_ref, imb_ref = _metrics(ref, x)
    assert abs(mse / mse_ref - 1) < 0.01, (mse, mse_ref)
    assert abs(imb - imb_ref) < 0.05 * imb_ref, (imb, imb_ref)
    return mse, mse_ref, imb, imb_ref


def _class_worker(rank, world, rendezvous, backend, cases):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=300))
    group = dist.group.WORLD
    try:
        if "steps" in cases:
            _steps_case(group, device)  # in-place revivals
            _steps_case(group, device, inplace_quantile=None)  # every revival a split
            _steps_case(group, device, inplace_quantile=None, donor_rule="proportional")
        if "fused" in cases:  # 16-bit rows: the fused cap searches them in place
            _steps_case(group, device, dtype=torch.bfloat16)
            _steps_case(group, device, dtype=torch.bfloat16, inplace_quantile=None)
        if "sharded" in cases:
            _steps_case(group, device, sharded=True)
            _steps_case(group, device, sharded=True, inplace_quantile=None)
            _steps_case(group, device, sharded=True, inplace_quantile=None,
                        donor_rule="proportional")
            _steps_case(group, device, sharded=True, cap_mode=None)
        for engine in ("native", "triton", "cute"):
            if engine in cases:
                runs = [(False, {}), (True, {}), (False, dict(device_path=True)),
                        (False, dict(device_path=True, donor_rule="proportional")),
                        (True, dict(donor_rule="proportional"))]
                for sharded, extra in runs:
                    out = _full_run_case(group, device, engine, sharded, **extra)
                    if rank == 0:
                        tag = (engine, " sharded" if sharded else "",
                               " " + ",".join(sorted(extra)) if extra else "", world)
                        print("full run %s%s%s on %d ranks: mse %.6f vs %.6f, "
                              "imbalance %.4f vs %.4f" % (tag + out), flush=True)
    finally:
        dist.destroy_process_group()


class TestCapPrimitives(unittest.TestCase):

    def test_matches_independent_implementation(self):
        # The single-process cap is the independent one's (distinct margins).
        for seed, B, k, d, cap, n_rounds in CASES:
            if seed >= 100:
                continue  # its float64 key breaks ties arbitrarily
            x, c, bias = _problem(seed, B, k, d, _skew(seed), "cpu")
            v, i = _top(x, c, bias, 2)
            args = (x, i[:, 0], v[:, 1] - v[:, 0], i[:, 1], v[:, 1], cap, k, n_rounds,
                    _masked(c, bias))
            stats = {}
            ours = cap_rounds(*args, stats=stats)
            self.assertGreater(int(stats["moved"]), 0)
            self.assertTrue(torch.equal(ours, _independent_rounds(*args)), seed)

    def _spawn(self, world, backend):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _worker, args=(world, "file://" + os.path.join(directory, "g"), backend),
                nprocs=world, join=True)

    def _spawn_class(self, world, backend, cases):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _class_worker,
                args=(world, "file://" + os.path.join(directory, "g"), backend, cases),
                nprocs=world, join=True)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_class_steps_cpu(self):
        self._spawn_class(2, "gloo", ("steps", "sharded"))
        self._spawn_class(4, "gloo", ("steps", "sharded"))

    @unittest.skipUnless(NCCL, "two CUDA GPUs and NCCL required")
    def test_class_gpus(self):
        cases = ["steps", "fused", "sharded", "triton"]
        if torch.cuda.get_device_capability() == (9, 0):
            cases.append("cute")
        for world in (2, 4, 8):
            if world <= N_GPUS:
                self._spawn_class(world, "nccl", tuple(cases))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_two_and_four_cpu_processes(self):
        self._spawn(2, "gloo")
        self._spawn(4, "gloo")

    @unittest.skipUnless(NCCL, "two CUDA GPUs and NCCL required")
    def test_gpus(self):
        for world in (2, 4, 8):
            if world <= N_GPUS:
                self._spawn(world, "nccl")


if __name__ == "__main__":
    unittest.main()
