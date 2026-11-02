# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from faiss.contrib.e_means import e_means_cute as backend
from torch_test_e_means_cute import _ready


class TestBf16Pitch(unittest.TestCase):
    def test_large_meta_allocations_keep_legal_layouts(self):
        limit = 2**31
        for rows, dim in (
            (2, 100),
            (20_000_000, 100),
            (limit // 16 - 1, 1),
            (limit // 16, 1),
            (limit // 16 + 1, 1),
        ):
            with self.subTest(rows=rows, dim=dim):
                backend._check_padded_span('rows', rows, dim)
                value = backend._aligned_bf16(rows, dim, 'meta')
                pitch = value.stride(0)
                self.assertEqual(value.device.type, 'meta')
                self.assertEqual(tuple(value.shape), (rows, dim))
                self.assertEqual(value.dtype, torch.bfloat16)
                self.assertEqual(value.stride(1), 1)
                self.assertGreaterEqual(pitch, dim)
                self.assertEqual(pitch * value.element_size() % 16, 0)
                span = rows * pitch
                self.assertLess(span, limit)
                capacity = value.untyped_storage().nbytes() // value.element_size()
                self.assertGreaterEqual(capacity - value.storage_offset(), span)
        with self.assertRaisesRegex(ValueError, 'signed int32'):
            backend._check_padded_span('rows', limit // 8, 1)

    def test_small_storage_suffix_and_other_dtypes(self):
        cases = [(torch.bfloat16, dim) for dim in (1, 8, 9, 16, 17, 24, 25, 96, 100, 256)]
        for dtype, dim in cases:
            with self.subTest(dtype=dtype, dim=dim):
                rows = backend._aligned_bf16(3, dim, 'cpu')
                pitch = rows.stride(0)
                self.assertEqual(tuple(rows.shape), (3, dim))
                self.assertEqual(rows.dtype, dtype)
                self.assertEqual(rows.stride(1), 1)
                self.assertGreaterEqual(pitch, dim)
                self.assertEqual(pitch * rows.element_size() % 16, 0)
                self.assertEqual(rows.data_ptr() % 16, 0)
                capacity = rows.untyped_storage().nbytes() // rows.element_size()
                self.assertGreaterEqual(capacity - rows.storage_offset(), 3 * pitch)
                physical = rows.as_strided((3, pitch), (pitch, 1))
                self.assertEqual(torch.count_nonzero(physical[:, dim:].view(torch.uint8)).item(), 0)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_mixed_pitches_native_and_grouped(self):
        from faiss.contrib.e_means.e_means_cute_kernel import (
            prepare_tiled_gemm, launch_tiled_gemm)
        from faiss.contrib.e_means.e_means_cute_row_owned import (
            prepare_row_owned, launch_row_owned)
        from faiss.contrib.e_means.e_means_cute_grouped import (
            prepare_grouped, launch_grouped)

        # Reuse the n33/k512/d96 integer/tie fixture, with four zero coordinates.
        n, k, original_dim, dim = 33, 512, 96, 100
        generator = torch.Generator(device='cuda').manual_seed(n + k + original_dim)
        original_rows = torch.randint(-2, 3, (n, original_dim),
                                      generator=generator, device='cuda').bfloat16()
        original_centroids = torch.randint(-2, 3, (k, original_dim),
                                           generator=generator, device='cuda').float()
        original_centroids[1].copy_(original_centroids[0])
        rows_source = torch.nn.functional.pad(original_rows, (0, dim - original_dim))
        centroids_source = torch.nn.functional.pad(original_centroids, (0, dim - original_dim))
        for row_pitch, centroid_pitch in ((104, 112), (112, 104)):
            with self.subTest(row_pitch=row_pitch, centroid_pitch=centroid_pitch):
                row_storage = torch.zeros((n, row_pitch), dtype=torch.bfloat16, device='cuda')
                centroid_storage = torch.zeros((k, centroid_pitch), dtype=torch.bfloat16, device='cuda')
                rows, centroids = row_storage[:, :dim], centroid_storage[:, :dim]
                rows.copy_(rows_source)
                centroids.copy_(centroids_source)
                prepared = backend._prepare_centroids(centroids_source)
                prepared.centroids = centroids
                keys = prepared.norms.double()[None] - 2 * rows.double() @ centroids.double().T
                expected_labels = keys.argmin(1)
                expected_keys = keys.gather(1, expected_labels[:, None]).flatten().float()
                for name, prepare, launch, plan in (
                    ('tiled', prepare_tiled_gemm, launch_tiled_gemm, None),
                    ('row_owned', prepare_row_owned, launch_row_owned, (128, 128, 3)),
                    ('grouped', prepare_grouped, launch_grouped, (112, 256, 3)),
                ):
                    with self.subTest(consumer=name):
                        values = torch.full((n + 2,), -123., device='cuda')
                        labels = torch.full((n + 2,), -123, dtype=torch.int64, device='cuda')
                        if plan is None:
                            operation = prepare(rows, prepared, labels[1:-1], values[1:-1])
                        else:
                            operation = prepare(rows, prepared, values[1:-1], labels[1:-1], plan)
                        launch(operation)
                        torch.testing.assert_close(labels[1:-1], expected_labels, rtol=0, atol=0)
                        torch.testing.assert_close(values[1:-1].view(torch.int32),
                                                   expected_keys.view(torch.int32), rtol=0, atol=0)
                        self.assertEqual(labels[[0, -1]].tolist(), [-123, -123])
                        self.assertEqual(values[[0, -1]].tolist(), [-123., -123.])
                torch.testing.assert_close(rows.view(torch.int16),
                                           rows_source.view(torch.int16), rtol=0, atol=0)
                torch.testing.assert_close(centroids.view(torch.int16),
                                           centroids_source.bfloat16().view(torch.int16), rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(row_storage[:, dim:].view(torch.int16)).item(), 0)
                self.assertEqual(torch.count_nonzero(centroid_storage[:, dim:].view(torch.int16)).item(), 0)


if __name__ == '__main__':
    unittest.main()
