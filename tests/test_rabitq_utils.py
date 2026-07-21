# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import faiss

import numpy as np


def round_half_away(x):
    # Matches C++ std::round (round half away from zero), unlike Python round()
    # which rounds half to even.
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def round_clamped_ref(x, max_code):
    # Mirrors std::round (half away from zero) then clamp to [0, max_code].
    y = round_half_away(x)
    y = max(0.0, min(y, float(max_code)))
    return int(y)


def check_linear_transform_batch_match(test, have_bias):
    d_in = 7
    d_out = 5
    transform = faiss.LinearTransform(d_in, d_out, have_bias)
    transform.is_trained = True

    a = np.sin(np.arange(d_in * d_out, dtype=np.float32) * np.float32(0.37))
    faiss.copy_array_to_vector(a, transform.A)

    if have_bias:
        b = np.arange(d_out, dtype=np.float32)
        b = np.float32(0.25) * b - np.float32(0.5)
        faiss.copy_array_to_vector(b, transform.b)

    x = np.cos(np.arange(d_in, dtype=np.float32) * np.float32(0.19))

    one_out = np.empty(d_out, dtype="float32")
    transform.apply_noalloc(1, faiss.swig_ptr(x), faiss.swig_ptr(one_out))

    batch_x = np.tile(x, 4)
    batch_out = np.empty(4 * d_out, dtype="float32")
    transform.apply_noalloc(
        4, faiss.swig_ptr(batch_x), faiss.swig_ptr(batch_out)
    )

    for i in range(d_out):
        test.assertAlmostEqual(one_out[i], batch_out[i], delta=1e-5)


class RaBitQUtilsTest(unittest.TestCase):
    def test_round_nonnegative_to_uint8(self):
        self.assertEqual(faiss.round_nonnegative_to_uint8(0.0), 0)
        self.assertEqual(faiss.round_nonnegative_to_uint8(0.49), 0)
        self.assertEqual(faiss.round_nonnegative_to_uint8(0.5), 1)
        self.assertEqual(faiss.round_nonnegative_to_uint8(1.5), 2)
        self.assertEqual(faiss.round_nonnegative_to_uint8(254.5), 255)
        self.assertEqual(faiss.round_nonnegative_to_uint8(255.0), 255)

        below_half = float(np.nextafter(np.float32(0.5), np.float32(0.0)))
        self.assertLessEqual(
            abs(
                int(faiss.round_nonnegative_to_uint8(below_half))
                - round_half_away(below_half)
            ),
            1,
        )

        rng = np.random.RandomState(123)
        for x in rng.uniform(0.0, 255.49, size=10000).astype(np.float32):
            x = float(x)
            self.assertLessEqual(
                abs(
                    int(faiss.round_nonnegative_to_uint8(x))
                    - round_half_away(x)
                ),
                1,
            )

    def test_round_nonnegative_to_uint16(self):
        self.assertEqual(faiss.round_nonnegative_to_uint16(0.0), 0)
        self.assertEqual(faiss.round_nonnegative_to_uint16(0.5), 1)
        self.assertEqual(faiss.round_nonnegative_to_uint16(65534.5), 65535)
        self.assertEqual(faiss.round_nonnegative_to_uint16(65535.0), 65535)

    def test_round_clamped_to_uint8(self):
        self.assertEqual(faiss.round_clamped_to_uint8(-1.0, 15), 0)
        self.assertEqual(faiss.round_clamped_to_uint8(0.49, 15), 0)
        self.assertEqual(faiss.round_clamped_to_uint8(0.5, 15), 1)
        self.assertEqual(faiss.round_clamped_to_uint8(14.5, 15), 15)
        self.assertEqual(faiss.round_clamped_to_uint8(15.4, 15), 15)
        self.assertEqual(faiss.round_clamped_to_uint8(16.0, 15), 15)

        rng = np.random.RandomState(456)
        for x in rng.uniform(-4.0, 260.0, size=10000).astype(np.float32):
            x = float(x)
            self.assertLessEqual(
                abs(
                    int(faiss.round_clamped_to_uint8(x, 255))
                    - round_clamped_ref(x, 255)
                ),
                1,
            )


class LinearTransformTest(unittest.TestCase):
    def test_single_vector_matches_batch_path(self):
        check_linear_transform_batch_match(self, False)
        check_linear_transform_batch_match(self, True)
