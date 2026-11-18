# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from faiss.contrib.e_means import Emeans, QuantizedEmeans
from e_means_test_utils import highest_fp32, known_torch_centroids


class TestEmeansAutocast(unittest.TestCase):
    def check_assignment(self, device, dtype=torch.float32):
        for spherical in (False, True):
            if spherical:
                centroids = torch.eye(2, device=device)
                rows = torch.tensor(
                    [[1.003, 1.004], [1.004, 1.003]], device=device
                )
                rows = torch.nn.functional.normalize(rows, dim=1)
                expected_labels = [1, 0]
            else:
                rows = torch.tensor([[1.003, 0.], [1.004, 0.]], device=device)
                centroids = rows.clone()
                expected_labels = [0, 1]
            rows = rows.to(dtype)
            classes = (Emeans, QuantizedEmeans) if device == "cpu" else (
                QuantizedEmeans,
            )
            for cls in classes:
                for chunk_size in (1, 2):
                    with self.subTest(
                        device=device, dtype=dtype, cls=cls.__name__,
                        spherical=spherical, chunk_size=chunk_size,
                    ):
                        model = known_torch_centroids(
                            centroids, model_class=cls,
                            assignment_backend="native",
                            spherical=spherical,
                            centroid_chunk_size=chunk_size,
                        )
                        expected = model.assign(rows)
                        if dtype == torch.float32:
                            self.assertEqual(
                                expected[1].tolist(), expected_labels
                            )
                        before = torch.is_autocast_enabled(device)
                        with torch.autocast(device, dtype=torch.bfloat16):
                            actual = model.assign(rows)
                            self.assertTrue(torch.is_autocast_enabled(device))
                            self.assertEqual(
                                (rows @ rows.t()).dtype, torch.bfloat16
                            )
                        self.assertEqual(
                            torch.is_autocast_enabled(device), before
                        )
                        for result, reference in zip(actual, expected):
                            torch.testing.assert_close(
                                result, reference, rtol=0, atol=0
                            )
                        self.assertEqual(
                            model.assignment_recipe["representation"],
                            str(dtype).removeprefix("torch."),
                        )

    def test_cpu_assignment(self):
        self.check_assignment("cpu")

    def test_cpu_training(self):
        generator = torch.Generator().manual_seed(123)
        rows = torch.randn(97, 8, generator=generator)
        for cls in (Emeans, QuantizedEmeans):
            for spherical in (False, True):
                with self.subTest(cls=cls.__name__, spherical=spherical):
                    data = (torch.nn.functional.normalize(rows, dim=1)
                            if spherical else rows)
                    options = dict(
                        nepochs=2, batch_size=16, alpha0=0.2,
                        unused_threshold=0, assignment_backend="native",
                        assignment_chunk_size=7, centroid_chunk_size=3,
                        spherical=spherical,
                    )
                    if cls is QuantizedEmeans:
                        options["quantizer"] = None
                    reference = cls(8, 5, **options)
                    model = cls(8, 5, **options)
                    reference.train(data, data[:5])
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        model.train(data, data[:5])
                        self.assertTrue(torch.is_autocast_enabled("cpu"))
                    for name in ("centroids", "assignments", "_usage", "obj"):
                        torch.testing.assert_close(
                            getattr(model, name), getattr(reference, name),
                            rtol=0, atol=0,
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_assignment(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 autocast")
        with highest_fp32("cuda"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                self.check_assignment("cuda", dtype)


if __name__ == "__main__":
    unittest.main()
