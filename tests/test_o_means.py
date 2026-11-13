# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""o-means training schedule. No GPU or PyTorch needed; the training itself
is tested in torch_test_o_means.py."""

import math
import unittest

from faiss.contrib.e_means.base import (
    _CHECK_UNUSED_EVERY,
    _epoch_rows,
    _resolve_recipe,
    _scheduled_alpha,
)
from faiss.contrib.o_means.schedule import make_schedule, num_cells, num_probes


class TestOmeansSchedule(unittest.TestCase):
    def test_cells_and_probes(self):
        self.assertEqual(num_cells(1024), 16)
        self.assertEqual(num_cells(16384), 128)
        self.assertEqual(num_cells(20000), 160)
        self.assertEqual(num_cells(65536), 512)
        self.assertEqual(num_probes(96, 16384), 16)
        self.assertEqual(num_probes(768, 16384), 12)
        self.assertEqual(num_probes(96, 65536), 24)
        self.assertEqual(num_probes(768, 65536), 24)
        s = make_schedule(10**6, 96, 16384, 1)
        self.assertEqual((s.nlist, s.nprobe), (128, 16))
        s = make_schedule(1000, 8, 12, 1)
        self.assertEqual((s.nlist, s.nprobe), (12, 12))
        s = make_schedule(10**6, 96, 16384, 1, nlist=64, nprobe=4)
        self.assertEqual((s.nlist, s.nprobe), (64, 4))

    def test_recipe_is_e_means(self):
        for n, k, law in ((10**6, 16384, "throughput"), (3 * 10**6, 65536, "quality")):
            for T in (1, 1.6, 3):
                s = make_schedule(n, 96, k, T)
                self.assertEqual(s.batch_law, law)
                self.assertEqual((s.batch_size, s.alpha0),
                                 _resolve_recipe(n, k, T, law, None, None))
                self.assertEqual(s.epoch_rows, tuple(_epoch_rows(n, T)))
                self.assertEqual(len(s.steps), sum(-(-r // s.batch_size) for r in s.epoch_rows))
                self.assertEqual([t for t, step in enumerate(s.steps) if step.check],
                                 [t for t in range(len(s.steps))
                                  if (t + 1) % _CHECK_UNUSED_EVERY == 0])
                for t in (0, len(s.steps) // 2, len(s.steps) - 1):
                    self.assertEqual(s.steps[t].alpha,
                                     _scheduled_alpha(s.alpha0, t, len(s.steps)))
        self.assertEqual(make_schedule(10**6, 96, 16384, 1, batch_law="quality").batch_law,
                         "quality")

    def test_below_one_epoch_keeps_one_epoch_recipe(self):
        s = make_schedule(10**6, 96, 16384, 0.8)
        self.assertEqual((s.batch_size, s.alpha0),
                         _resolve_recipe(10**6, 16384, 1.0, "throughput", None, None))
        self.assertEqual(s.epoch_rows, (800_000,))

    def test_revival_off_in_two_dimensions(self):
        s = make_schedule(10**5, 2, 64, 3)
        self.assertEqual(s.unused_threshold, 0)
        self.assertFalse(any(step.check for step in s.steps))

    def test_closing_weights(self):
        n = 10**6
        s = make_schedule(n, 96, 16384, 1.6)
        self.assertTrue(s.recency_close)
        total = sum(s.epoch_rows)
        done = 0
        for step in s.steps:
            done += step.b1 - step.b0
            self.assertAlmostEqual(step.omega, math.exp(-(total - done) / n), places=12)
        self.assertEqual(s.steps[-1].omega, 1.0)
        s = make_schedule(n, 96, 16384, 2)
        self.assertFalse(s.recency_close)
        self.assertTrue(all(step.omega == (1.0 if step.epoch == 1 else None)
                            for step in s.steps))

    def test_rebuild_cadence(self):
        s = make_schedule(10**6, 96, 16384, 3)
        steps_per_pass = -(-10**6 // s.batch_size)
        self.assertEqual(s.rebuild_every, max(_CHECK_UNUSED_EVERY, -(-steps_per_pass // 2)))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            make_schedule(1000, 8, 16, 0)
        with self.assertRaises(ValueError):
            make_schedule(10, 8, 16, 1)
        with self.assertRaises(ValueError):
            make_schedule(1000, 8, 16, 1, batch_law="fast")
        with self.assertRaises(ValueError):
            make_schedule(1000, 8, 16, 1, nlist=32)
        with self.assertRaises(ValueError):
            make_schedule(1000, 8, 16, 1, nlist=8, nprobe=9)


if __name__ == "__main__":
    unittest.main()
