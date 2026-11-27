# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Balanced E-means on tensors: the recipe pieces and the class."""

import itertools
import unittest

import torch

from faiss.contrib.e_means.e_means_balanced import (
    EmeansBalanced,
    band_step,
    select_inplace,
    cap_first_pass,
    cap_reference,
    cap_round_evict,
    cap_rows,
    draw_donors,
    draw_donors_fixed,
    largest_donors,
    largest_donors_fixed,
    split_rows,
)


def _integer_problem(n, k, d, seed, spread=6):
    """Rows, centroids and a bias with small integer values: every score is exact in FP32."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-spread, spread + 1, (n, d), generator=g).float()
    c = torch.randint(-spread, spread + 1, (k, d), generator=g).float()
    b = torch.randint(-8, 9, (k,), generator=g).float()
    return x, c, b


def _scores(x, c, b):
    return ((x[:, None, :] - c[None]) ** 2).sum(-1).double() - b.double()[None]


def _cap_with_functions(scores, cap, rounds):
    """The library's cap on a dense score matrix: first pass, then masked rounds."""
    n, k = scores.shape
    order = torch.argsort(scores, dim=1, stable=True)
    best, second = order[:, 0], order[:, 1]
    rows = torch.arange(n)
    d1, d2 = scores[rows, best].float(), scores[rows, second].float()
    codes, moved = cap_first_pass(best, d2 - d1, second, cap, k)
    score, arrived = d2.clone(), moved.clone()
    for _ in range(rounds):
        evict, full = cap_round_evict(codes, arrived, score, cap, k)
        ids = evict.nonzero().flatten()
        if len(ids) == 0:
            break
        masked = torch.where(full[None, :], float("inf"), scores[ids])
        new = torch.argmin(masked, dim=1)
        codes[ids] = new
        score[ids] = scores[ids, new].float()
        arrived[ids] = True
    return codes


class TestCapPieces(unittest.TestCase):

    def test_cap_rows(self):
        self.assertEqual(cap_rows(1000, 100, 2.0, 1.0), 20)
        self.assertEqual(cap_rows(1000, 100, 2.0, 0.5), 40)
        self.assertEqual(cap_rows(1000, 100, 2.0, 0.5, anneal=False), 20)
        self.assertEqual(cap_rows(1000, 100, 2.0, 0.0), 2 * 10 ** 7)
        self.assertEqual(cap_rows(7, 100, 2.0, 1.0), 1)

    def test_first_pass_matches_sequential_reference(self):
        for seed in range(12):
            g = torch.Generator().manual_seed(seed)
            n, k = 300, 9
            scores = torch.rand(n, k, generator=g, dtype=torch.float64)
            # Crowd two clusters so the cap binds.
            scores[: n // 2, 0] -= 1.0
            scores[n // 2: 3 * n // 4, 1] -= 1.0
            cap = int(torch.randint(5, 60, (1,), generator=g))
            self.assertTrue(torch.equal(_cap_with_functions(scores, cap, 0),
                                        cap_reference(scores, cap, 0)))

    def test_rounds_match_sequential_reference(self):
        for seed in range(12):
            x, c, b = _integer_problem(400, 13, 3, seed)
            scores = _scores(x, c, b)
            for cap in (int(400 / 13 * 1.5) + 1, int(400 / 13 * 2) + 1, 70):
                for rounds in (1, 2, 4):
                    self.assertTrue(torch.equal(_cap_with_functions(scores, cap, rounds),
                                                cap_reference(scores, cap, rounds)),
                                    (seed, cap, rounds))

    def test_ties_go_to_lower_position(self):
        codes = torch.zeros(6, dtype=torch.int64)
        second = torch.arange(6) % 2 + 1
        margin = torch.ones(6)
        new, moved = cap_first_pass(codes, margin, second, 2, 3)
        self.assertEqual(moved.tolist(), [False, False, True, True, True, True])
        self.assertEqual(new.tolist(), [0, 0, 1, 2, 1, 2])
        arrived = torch.tensor([True, False, True, True, False, True])
        evict, full = cap_round_evict(codes, arrived, torch.zeros(6), 3, 3)
        # Own rows (1, 4) first, then arrivals by position: row 0 stays, 2, 3, 5 go.
        self.assertEqual(evict.tolist(), [False, False, True, True, False, True])
        self.assertEqual(full.tolist(), [True, False, False])

    def test_margin_order_and_negative_scores(self):
        codes = torch.zeros(5, dtype=torch.int64)
        margin = torch.tensor([0.5, 3.0, -0.0, 2.0, 1e-30])
        new, moved = cap_first_pass(codes, margin, torch.ones(5, dtype=torch.int64), 2, 2)
        self.assertEqual(moved.tolist(), [True, False, True, False, True])
        score = torch.tensor([-3.0, 2.0, -1e9, 0.0, 5.0])
        evict, _ = cap_round_evict(codes, torch.ones(5, dtype=torch.bool), score, 2, 2)
        self.assertEqual(evict.tolist(), [False, True, False, True, True])


class TestRecipePieces(unittest.TestCase):

    def test_band_step(self):
        counts = torch.tensor([0.0, 2.0, 4.0, 6.0])  # share 3
        lam, mu, bias = torch.zeros(4), torch.zeros(4), torch.zeros(4)
        band_step(lam, mu, bias, counts, 0.6)
        r = (counts - 3) / 3
        self.assertTrue(torch.allclose(lam, (0.6 * r).clamp(min=0)))
        self.assertTrue(torch.allclose(mu, (-0.6 * r).clamp(min=0)))
        self.assertTrue(torch.equal(bias, mu - lam))
        band_step(lam, mu, bias, torch.full((4,), 3.0), torch.tensor(1.0))
        self.assertTrue(torch.equal(bias, mu - lam))

    def test_inplace_reference_is_the_lower_median(self):
        bias = torch.tensor([5.0, 1.0, 3.0, -2.0, 0.0, 4.0])
        dead = torch.tensor([True, False, False, True, False, False])
        inplace, ref = select_inplace(dead, bias)
        self.assertEqual(float(ref), float(bias[~dead].median()))
        self.assertEqual(inplace.tolist(), [False, False, False, True, False, False])
        _, ref = select_inplace(torch.ones(6, dtype=torch.bool), bias)
        self.assertEqual(float(ref), 0.0)

    def test_donors(self):
        g = torch.Generator().manual_seed(0)
        w = torch.tensor([1.0, 0.0, 2.0, 3.0, 0.5, 0.0])
        dead = torch.tensor([False, True, False, False, False, True])
        source, target = draw_donors(w, dead, g)
        self.assertEqual(target.tolist(), [1, 5])
        self.assertEqual(len(set(source.tolist())), 2)
        self.assertFalse(bool(dead[source].any()))
        self.assertTrue(bool((w[source] > 0).all()))
        none = draw_donors(torch.zeros(6), dead, g)
        self.assertEqual(none, (None, None))
        # Proportional to the weights.
        hits = torch.zeros(3)
        w = torch.tensor([1.0, 2.0, 5.0, 0.0])
        dead = torch.tensor([False, False, False, True])
        for _ in range(4000):
            hits[draw_donors(w, dead, g)[0]] += 1
        self.assertTrue(torch.allclose(hits / 4000, w[:3] / 8, atol=0.03))

    def test_largest_donors(self):
        w = torch.tensor([3.0, 8.0, 0.01, 8.0, 5.0, 0.01, 9.0, 0.01, 1.0, 2.0])
        receivers = torch.zeros(10, dtype=torch.bool)
        receivers[[2, 5, 7]] = True
        # Largest usage first, the tie 1 / 3 to the lower index; receivers in index order.
        source, target = largest_donors(w, receivers)
        self.assertEqual(source.tolist(), [6, 1, 3])
        self.assertEqual(target.tolist(), [2, 5, 7])
        # Clusters found dead (revived in place) do not donate either.
        dead = receivers.clone()
        dead[6] = True
        source, target = largest_donors(w, receivers, dead)
        self.assertEqual(source.tolist(), [1, 3, 4])
        # More receivers than live clusters with usage: the first receivers only.
        w = torch.tensor([0.0, 4.0, 0.0, 0.0, 0.0, 2.0])
        receivers = torch.tensor([True, False, True, True, True, False])
        source, target = largest_donors(w, receivers)
        self.assertEqual((source.tolist(), target.tolist()), ([1, 5], [0, 2]))
        self.assertEqual(largest_donors(torch.zeros(6), receivers), (None, None))

    def test_fixed_donors_match_the_host_donors(self):
        for seed in range(50):
            g = torch.Generator().manual_seed(seed)
            k = int(torch.randint(2, 40, (1,), generator=g))
            # Coarse weights: frequent ties, and some zeros (not eligible).
            w = torch.randint(0, 5, (k,), generator=g).float()
            receivers = torch.rand(k, generator=g) < 0.3
            dead = receivers | (torch.rand(k, generator=g) < 0.1)
            for host, fixed in ((largest_donors(w, receivers, dead),
                                 largest_donors_fixed(w, receivers, dead)),
                                (draw_donors(w, receivers, torch.Generator().manual_seed(seed)),
                                 draw_donors_fixed(w, receivers,
                                                   torch.Generator().manual_seed(seed)))):
                source, target, count = fixed
                count = int(count)
                self.assertEqual(sorted(source.tolist()), list(range(k)))
                self.assertEqual(sorted(target.tolist()), list(range(k)))
                if host[0] is None:
                    self.assertEqual(count, 0)
                    continue
                self.assertEqual(source[:count].tolist(), host[0].tolist(), seed)
                self.assertEqual(target[:count].tolist(), host[1].tolist(), seed)

    def test_split_rows(self):
        donor, child = split_rows(torch.tensor([[2.0, 4.0, -8.0]]), 0.5)
        self.assertEqual(donor.tolist(), [[1.0, 6.0, -4.0]])
        self.assertEqual(child.tolist(), [[3.0, 2.0, -12.0]])


def _blobs(n, d, k, seed, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(k, d, generator=g) * 4
    weights = torch.arange(1, k + 1, dtype=torch.float64) ** -1.0
    comp = torch.multinomial(weights, n, replacement=True, generator=g)
    return (means[comp] + torch.randn(n, d, generator=g)).to(device)


def _donor_model(rule, usage, start, device, quantile=None, bias=None):
    """A model in its training state with the given usage (clusters below 0.1x the mean die)."""
    k, d = start.shape
    model = EmeansBalanced(d, k, nepochs=1, polish_epochs=0, batch_size=1000, seed=0,
                           donor_rule=rule, inplace_quantile=quantile)
    model.begin(1000, start.to(device))
    model._resolve_run_recipe(1000)
    model._usage.copy_(usage)
    if bias is not None:
        model.bias.copy_(bias)
        model._lam.copy_((-bias).clamp(min=0))
        model._mu.copy_(bias.clamp(min=0))
    # The counters the fixed-shape revival adds to (the device path's).
    model._inplace_device = torch.zeros((), dtype=torch.int64, device=device)
    model._check_splits = torch.zeros((), dtype=torch.int64, device=device)
    return model


class TestEmeansBalancedCPU(unittest.TestCase):

    def test_train_assign_and_determinism(self):
        x = _blobs(12000, 8, 40, 0)
        results = []
        for _ in range(2):
            model = EmeansBalanced(8, 64, nepochs=5, polish_epochs=2, seed=3)
            centroids, labels = model.train(x)
            results.append((centroids, labels, model.bias))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertTrue(torch.equal(results[0][2], results[1][2]))
        self.assertTrue(bool((labels >= 0).all()))
        recipe = model.recipe
        self.assertEqual(recipe["train_steps"], 3 * (12000 // recipe["batch_size"]))
        self.assertEqual(recipe["tail_policy"], "drop")
        # Biased and unbiased assignment against brute force.
        bias = model.bias
        dist, biased = model.assign(x)
        plain_dist, plain = model.assign(x, biased=False)
        full = ((x[:, None, :] - centroids[None]) ** 2).sum(-1)
        self.assertGreater(float((biased == torch.argmin(full - bias[None], 1)).float().mean()),
                           0.999)
        self.assertGreater(float((plain == torch.argmin(full, 1)).float().mean()), 0.999)
        rows = torch.arange(len(x))
        self.assertTrue(torch.allclose(dist, full[rows, biased], rtol=1e-4, atol=1e-3))
        self.assertTrue(torch.allclose(plain_dist, full[rows, plain], rtol=1e-4, atol=1e-3))
        # Balance: the biased routing is far more even than the plain one.
        imbalance = [float((torch.bincount(v, minlength=64).double() ** 2).mean()
                           / (len(x) / 64) ** 2) for v in (biased, plain)]
        self.assertLess(imbalance[0], imbalance[1])
        self.assertLess(imbalance[0], 1.1)

    def test_seed_changes_result(self):
        x = _blobs(6000, 4, 20, 1)
        a = EmeansBalanced(4, 32, nepochs=3, polish_epochs=1, seed=1).train(x)[0]
        b = EmeansBalanced(4, 32, nepochs=3, polish_epochs=1, seed=2).train(x)[0]
        self.assertFalse(torch.equal(a, b))

    def test_cap_off_and_no_polish(self):
        x = _blobs(6000, 4, 20, 2)
        model = EmeansBalanced(4, 32, nepochs=2, polish_epochs=0, cap_mode=None, seed=1)
        model.train(x)
        self.assertEqual(model.cap_stats["steps"], 0)
        self.assertEqual(len(model.iteration_stats), 2)

    def test_donor_rules_in_the_class(self):
        """The revival draws donors by usage by default, or splits the largest live clusters."""
        usage = torch.tensor([3.0, 8.0, 0.01, 8.0, 5.0, 0.01, 9.0, 0.01, 1.0, 2.0])
        start = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        for rule in ("largest", "proportional"):
            for fixed in (False, True):
                model = _donor_model(rule, usage, start, "cpu")
                generator = torch.Generator().set_state(model._split_rng.get_state())
                model._revive_fixed() if fixed else model._revive()
                if rule == "largest":
                    source = [6, 1, 3]
                else:
                    source = draw_donors(usage, usage < 0.1 * usage.mean(), generator)[0].tolist()
                donor, child = split_rows(start[source], model.split_eps)
                self.assertTrue(torch.equal(model._centroids[[2, 5, 7]], child), (rule, fixed))
                self.assertTrue(torch.equal(model._centroids[source], donor), (rule, fixed))
                self.assertTrue(torch.equal(model._usage[[2, 5, 7]], usage[source] / 2))
                self.assertEqual(model.n_splits, 3)
        self.assertEqual(model.recipe["donor_rule"], "proportional")
        self.assertEqual(EmeansBalanced(2, 10).donor_rule, "proportional")
        with self.assertRaises(ValueError):
            EmeansBalanced(2, 10, donor_rule="random")

    def test_numpy_class_takes_the_same_donors(self):
        try:
            from faiss.contrib.e_means.e_means_balanced_numpy import NumpyEmeansBalanced
        except ImportError:
            self.skipTest("the NumPy class is not available")
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            usage = torch.randint(1, 6, (24,), generator=g).float()
            usage[torch.rand(24, generator=g) < 0.3] = 0.01
            start = torch.randn(24, 3, generator=g)
            bias = torch.randint(-3, 4, (24,), generator=g).float()
            for quantile in (0.5, None):
                model = _donor_model("largest", usage, start, "cpu", quantile, bias)
                model._revive()
                ref = NumpyEmeansBalanced(3, 24, nepochs=1, polish_epochs=0, batch_size=1000,
                                          inplace_quantile=quantile, donor_rule="largest")
                ref.begin(1000, start.numpy())
                ref._usage[:] = usage.numpy()
                ref.bias[:] = bias.numpy()
                ref._lam[:] = (-bias).clamp(min=0).numpy()
                ref._mu[:] = bias.clamp(min=0).numpy()
                ref._revive()
                self.assertEqual((model.n_splits, model.n_inplace), (ref.n_splits, ref.n_inplace))
                self.assertTrue(torch.equal(model._centroids, torch.from_numpy(ref._centroids)))
                # The in-place mean usage is summed in another order: equal up to rounding.
                self.assertTrue(torch.allclose(model._usage, torch.from_numpy(ref._usage),
                                               rtol=1e-6))
                self.assertTrue(torch.equal(model.bias, torch.from_numpy(ref.bias)))

    def test_class_cap_matches_reference(self):
        """One capped step of the class against the sequential reference (exact scores)."""
        for seed in range(4):
            x, c, b = _integer_problem(512, 16, 4, seed)
            model = EmeansBalanced(4, 16, nepochs=1, polish_epochs=0, batch_size=512,
                                   alpha0=0.5, seed=seed)
            model.begin(512, c)
            model._resolve_input_dtype(x)
            model._resolve_assignment_backend(x)
            model.bias.copy_(b)
            cap = cap_rows(512, 16, 2.0, 1.0)
            prepared, norm, shift = model._biased_assignment(model._centroids, model.bias)
            d1, l1, d2, l2 = model._assign_chunk(x, prepared, norm)
            index = torch.arange(512)
            codes, _ = model._cap_rounds_step(x, index, x, l1, d1, d2, l2.long(),
                                              model._row_norms(x), cap, prepared, norm, shift,
                                              model.bias, None)
            # The class ranks arrivals by the kernel score (row norm dropped).
            scores = _scores(x, c, b) - (x.double() ** 2).sum(1)[:, None]
            self.assertTrue(torch.equal(codes, cap_reference(scores, cap, 2)), seed)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestEmeansBalancedGPU(unittest.TestCase):

    def _backends(self):
        names = ["native", "triton"]
        if torch.cuda.get_device_capability() == (9, 0):
            names.append("cute")
        return names

    def test_backends_agree_on_biased_assignment(self):
        x = _blobs(20000, 64, 100, 3, "cuda")
        model = EmeansBalanced(64, 256, nepochs=3, polish_epochs=1, seed=0,
                               assignment_backend="triton")
        centroids, _ = model.train(x)
        results = {}
        for name in self._backends():
            model.assignment_backend = name
            model._query_backend = None
            results[name] = (model.assign(x), model.assign(x, biased=False))
        base = results["triton"]
        for name, ((d, lb), (du, lu)) in results.items():
            self.assertGreater(float((lb == base[0][1]).float().mean()), 0.999, name)
            self.assertGreater(float((lu == base[1][1]).float().mean()), 0.999, name)
            self.assertTrue(torch.allclose(d, base[0][0], rtol=1e-2, atol=1e-2), name)

    def test_cap_step_on_every_backend(self):
        """From one state, each backend's capped codes respect the cap after the rounds."""
        x = _blobs(32768, 32, 50, 4, "cuda").bfloat16()
        for name in self._backends():
            model = EmeansBalanced(32, 512, nepochs=2, polish_epochs=1, batch_size=8192, seed=1,
                                   assignment_backend=name)
            centroids, labels = model.train(x)
            self.assertEqual(model.recipe["assignment_backend"], name)
            self.assertTrue(bool((labels >= 0).all()), name)
            self.assertGreater(model.cap_stats["steps"], 0, name)

    def test_device_rounds_match_host_rounds(self):
        """The sync-free repair rounds (Triton, count on the device) equal the host ones."""
        x = _blobs(16384, 64, 30, 6, "cuda").bfloat16()
        for name in self._backends():
            model = EmeansBalanced(64, 256, nepochs=1, polish_epochs=0, batch_size=16384,
                                   alpha0=0.5, seed=0, assignment_backend=name)
            init = x[torch.randperm(len(x), device="cuda")[:256]].float()
            model.begin(len(x), init)
            model._resolve_input_dtype(x)
            model._resolve_assignment_backend(x)
            model.bias.copy_(torch.randn(256, device="cuda") * 2)
            prepared, norm, shift = model._biased_assignment(model._centroids, model.bias)
            d1, l1, d2, l2 = model._assign_chunk(x, prepared, norm)
            index = torch.arange(len(x), device="cuda")
            rn = model._row_norms(x)
            cap = cap_rows(len(x), 256, 2.0, 1.0)
            host, _ = model._cap_rounds_step(x, index, x, l1.clone(), d1.clone(), d2, l2.long(),
                                             rn, cap, prepared, norm, shift, model.bias, None)
            state = model._device_state(len(x), "cuda")
            state["rows"].copy_(x)
            device, _ = model._cap_rounds_step(x, index, x, l1.clone(), d1.clone(), d2, l2.long(),
                                               rn, torch.tensor(cap, device="cuda"), prepared,
                                               norm, shift, model.bias, state)
            agree = float((host == device).float().mean())
            self.assertGreater(agree, 0.999, name)
            # The fused cap keeps each cluster's row count up to date through every move.
            held = state["fused"].held
            self.assertTrue(torch.equal(held.long(), torch.bincount(device, minlength=256)))
            over = torch.bincount(device, minlength=256).max()
            self.assertLessEqual(int(over), int(torch.bincount(host, minlength=256).max()) + 2)
            # Radix-sort ranking: the same decisions as the pairwise ranking, bit for bit;
            # a CuTe bucket for the masked search: the same up to near-ties.
            plans = [dict(first="sort", round=["sort", "sort"], bucket=[0, 0])]
            if name == "cute":
                plans.append(dict(first="pair", round=["pair", "sort"], bucket=[1024, 4096]))
                plans.append(dict(first="sort", round=["pair", "pair"], bucket=[-1, -1]))
            for plan in plans:
                model._cap_plan = plan
                other, _ = model._cap_rounds_step(
                    x, index, x, l1.clone(), d1.clone(), d2, l2.long(), rn,
                    torch.tensor(cap, device="cuda"), prepared, norm, shift, model.bias, state)
                if plan["bucket"][0] == 0:
                    self.assertTrue(torch.equal(other, device), (name, plan))
                else:
                    self.assertGreater(float((other == device).float().mean()), 0.999)
                self.assertTrue(torch.equal(held.long(), torch.bincount(other, minlength=256)))
            # Without runner-up labels the first pass searches the moved rows' runner-ups:
            # the same decisions up to near-ties between two kernels.
            model._cap_plan = dict(first="pair", round=["pair", "pair"], bucket=[0, 0])
            lazy, _ = model._cap_rounds_step(
                x, index, x, l1.clone(), d1.clone(), d2, None, rn,
                torch.tensor(cap, device="cuda"), prepared, norm, shift, model.bias, state)
            self.assertGreater(float((lazy == device).float().mean()), 0.999, name)
            self.assertTrue(torch.equal(held.long(), torch.bincount(lazy, minlength=256)))

    def test_device_cap_equals_host_cap_on_exact_scores(self):
        """With exact scores (small integers), the fused cap takes the host cap's decisions
        bit for bit: pairwise or sorted ranking, runner-up labels or searched runner-ups."""
        n, k, d = 2048, 64, 4
        x, c, b = (t.cuda() for t in _integer_problem(n, k, d, 7))
        x = x.bfloat16()
        model = EmeansBalanced(d, k, nepochs=1, polish_epochs=0, batch_size=n, alpha0=0.5,
                               seed=0, assignment_backend="triton")
        model.begin(n, c)
        model._resolve_input_dtype(x)
        model._resolve_assignment_backend(x)
        model.bias.copy_(b)
        prepared, norm, shift = model._biased_assignment(model._centroids, model.bias)
        index = torch.arange(n, device="cuda")
        rows = model._prepare_rows(x)
        l1, d1, d2, l2, rn = model._assign_batch(x, index, rows, prepared, norm, True)
        for cap in (20, 32, 50):
            host, _ = model._cap_rounds_step(x, index, rows, l1.clone(), d1.clone(), d2, l2,
                                             rn, cap, prepared, norm, shift, model.bias, None)
            self.assertGreater(int(model.cap_stats["evicted"]), 0)
            state = model._device_state(n, "cuda")
            state["rows"].copy_(rows)
            for first, second in itertools.product(("pair", "sort"), ("labels", "lazy")):
                model._cap_plan = dict(first=first, round=[first, "pair"], bucket=[0, 0])
                device, _ = model._cap_rounds_step(
                    x, index, rows, l1.clone(), d1.clone(), d2,
                    l2 if second == "labels" else None, rn, torch.tensor(cap, device="cuda"),
                    prepared, norm, shift, model.bias, state)
                self.assertTrue(torch.equal(device, host), (cap, first, second))

    def test_fixed_revival_matches_host_revival(self):
        """The graph-safe revival takes the host revival's decisions, for both donor rules."""
        for seed in range(5):
            g = torch.Generator().manual_seed(seed)
            usage = torch.randint(1, 6, (4096,), generator=g).float()
            usage[torch.rand(4096, generator=g) < 0.05] = 0.01
            start = torch.randn(4096, 16, generator=g)
            bias = torch.randn(4096, generator=g)
            for rule in ("largest", "proportional"):
                runs = []
                for fixed in (False, True):
                    model = _donor_model(rule, usage.cuda(), start, "cuda", 0.5, bias.cuda())
                    model._revive_fixed() if fixed else model._revive()
                    runs.append((model._centroids, model._usage, model.bias, model.n_splits,
                                 model.n_inplace))
                (c1, u1, b1, s1, i1), (c2, u2, b2, s2, i2) = runs
                self.assertGreater(s1, 0)
                self.assertEqual((s1, i1), (s2, i2), (seed, rule))
                self.assertTrue(torch.equal(c1, c2), (seed, rule))
                self.assertTrue(torch.equal(u1, u2), (seed, rule))
                self.assertTrue(torch.equal(b1, b2), (seed, rule))

    def test_fused_bias_step_matches_band_step(self):
        from faiss.contrib.e_means.balanced_triton import FusedBias

        g = torch.Generator(device="cuda").manual_seed(0)
        n, k = 4096, 300
        codes = torch.randint(0, k, (n,), device="cuda", generator=g)
        bias = torch.randn(k, device="cuda", generator=g)
        dist = torch.rand(n, device="cuda", generator=g) * 10 + bias.max() - bias[codes]
        shift = bias.max()
        lam = torch.rand(k, device="cuda", generator=g)
        mu = torch.rand(k, device="cuda", generator=g)
        fused = FusedBias(n, k, "cuda")
        epoch = torch.zeros((), dtype=torch.float64, device="cuda")
        fused.distances(dist, codes, bias, shift, epoch, True)
        counts = torch.bincount(codes, minlength=k).float()
        self.assertTrue(torch.equal(fused.counts, counts))
        true = dist - (shift - bias[codes])
        self.assertTrue(torch.allclose(fused.batch_sum, true.sum(dtype=torch.float64), rtol=1e-6))
        for first in (True, False):
            D = torch.tensor(3.0, device="cuda")
            scale = torch.tensor(0.7, device="cuda")
            l1, m1, b1 = lam.clone(), mu.clone(), bias.clone()
            fused.band(counts, l1, m1, b1, D, scale, 0.02, first, 0.1)
            mean = (true.sum(dtype=torch.float64) / n).float()
            D_ref = mean if first else torch.tensor(3.0, device="cuda") * 0.9 + 0.1 * mean
            self.assertTrue(torch.allclose(D, D_ref, rtol=1e-6))
            l2, m2, b2 = lam.clone(), mu.clone(), bias.clone()
            band_step(l2, m2, b2, counts, 0.02 * D_ref * scale)
            self.assertTrue(torch.allclose(l1, l2, rtol=1e-5, atol=1e-6))
            self.assertTrue(torch.allclose(m1, m2, rtol=1e-5, atol=1e-6))
            self.assertTrue(torch.allclose(b1, b2, rtol=1e-5, atol=1e-6))

    def test_graphs_match_eager(self):
        x = _blobs(60000, 32, 64, 7, "cuda")
        results = []
        for graphs in (True, False):
            model = EmeansBalanced(32, 512, nepochs=5, polish_epochs=2, seed=4,
                                   assignment_backend="triton", cuda_graphs=graphs)
            centroids, labels = model.train(x)
            results.append((centroids, labels, model.bias, model.n_splits, model.n_inplace))
        (c1, l1, b1, s1, i1), (c2, l2, b2, s2, i2) = results
        self.assertEqual((s1, i1), (s2, i2))
        self.assertGreater(float((l1 == l2).float().mean()), 0.99)
        self.assertLess(float((c1 - c2).abs().max() / c2.abs().max()), 1e-2)

    def test_train_on_every_backend(self):
        x = _blobs(40000, 32, 64, 5, "cuda")
        mse = {}
        for name in self._backends():
            model = EmeansBalanced(32, 256, nepochs=6, polish_epochs=2, seed=2,
                                   assignment_backend=name)
            centroids, labels = model.train(x)
            d, lb = model.assign(x)
            counts = torch.bincount(lb, minlength=256).double()
            mse[name] = float(d.mean())
            self.assertLess(float((counts ** 2).mean() / counts.mean() ** 2), 1.1, name)
        values = list(mse.values())
        self.assertLess(max(values) / min(values), 1.02, mse)


if __name__ == "__main__":
    unittest.main()
