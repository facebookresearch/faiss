# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for benchmark metrics and dataset provenance, without a native build."""

import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


BENCH_PATH = Path(__file__).resolve().parents[1] / "benchs" / "bench_rht.py"


def load_benchmark():
    if not BENCH_PATH.exists():
        raise AssertionError("bench_rht.py is not implemented")
    spec = importlib.util.spec_from_file_location("bench_rht", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRHTBenchmark(unittest.TestCase):
    def test_set_recall_ignores_duplicate_and_missing_predictions(self):
        bench = load_benchmark()
        truth = np.array([[1, 2, 3], [4, 5, 6]])
        predicted = np.array([[1, 1, -1], [6, 4, 99]])
        np.testing.assert_allclose(
            bench.per_query_recall(truth, predicted), [1 / 3, 2 / 3]
        )
        with self.assertRaises(ValueError):
            bench.per_query_recall(truth, predicted[:1])

    def test_paired_bootstrap_preserves_pairing_and_is_reproducible(self):
        bench = load_benchmark()
        baseline = np.arange(12).reshape(3, 4) / 20
        treatment = baseline + 0.125
        result = bench.paired_bootstrap(treatment, baseline, 200, seed=7)
        self.assertAlmostEqual(result["delta"], 0.125)
        np.testing.assert_allclose(result["ci95"], [0.125, 0.125])
        self.assertEqual(
            result, bench.paired_bootstrap(treatment, baseline, 200, seed=7)
        )
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["query_count"], 4)

    def test_synthetic_data_are_fixed_across_transform_seeds(self):
        bench = load_benchmark()
        for distribution in ("gaussian", "student_t", "sparse32"):
            args = bench.make_parser().parse_args(
                [
                    "--synthetic",
                    distribution,
                    "--nt",
                    "20",
                    "--nb",
                    "30",
                    "--nq",
                    "4",
                    "--d",
                    "64",
                    "--data-seed",
                    "11",
                ]
            )
            xt, xb, xq, provenance = bench.load_data(args)
            self.assertEqual(
                (xt.shape, xb.shape, xq.shape), ((20, 64), (30, 64), (4, 64))
            )
            self.assertEqual(xt.dtype, np.dtype("float32"))
            np.testing.assert_array_equal(xb, bench.load_data(args)[1])
            self.assertEqual(provenance["data_seed"], 11)
            self.assertIn("independent", provenance["training_base_disjointness"])
            if distribution == "sparse32":
                np.testing.assert_array_equal(np.count_nonzero(xb, axis=1), 32)

    def test_npz_without_training_reserves_source_rows(self):
        bench = load_benchmark()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vectors.npz"
            original = np.arange(60, dtype="float32").reshape(15, 4)
            np.savez(path, xb=original, xq=original[:2] + 0.5)
            args = bench.make_parser().parse_args(
                ["--npz", str(path), "--nt", "5", "--nb", "7", "--nq", "2",]
            )
            xt, xb, _, provenance = bench.load_data(args)
            np.testing.assert_array_equal(xt, original[:5])
            np.testing.assert_array_equal(xb, original[5:12])
            self.assertEqual(provenance["training_source_rows"], [0, 5])
            self.assertEqual(provenance["base_source_rows"], [5, 12])
            self.assertIn(
                "disjoint source rows", provenance["training_base_disjointness"]
            )

    def test_explicit_training_overlap_is_reported_when_checked(self):
        bench = load_benchmark()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vectors.npz"
            x = np.arange(48, dtype="float32").reshape(12, 4)
            np.savez(path, xt=x[:5], xb=x[3:10], xq=x[:2] + 0.5)
            args = bench.make_parser().parse_args(
                [
                    "--npz",
                    str(path),
                    "--nt",
                    "5",
                    "--nb",
                    "7",
                    "--nq",
                    "2",
                    "--check-disjoint",
                    "--hash-files",
                ]
            )
            _, _, _, provenance = bench.load_data(args)
            self.assertEqual(provenance["exact_training_base_overlap_rows"], 2)
            self.assertEqual(len(provenance["files"][0]["sha256"]), 64)

    def test_fvecs_dimension_headers_are_validated(self):
        bench = load_benchmark()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "base.fvecs"
            values = np.arange(12, dtype="<f4").reshape(3, 4)
            rows = np.empty((3, 5), dtype="<i4")
            rows[:, 0] = 4
            rows[:, 1:] = values.view("<i4")
            rows.tofile(path)
            np.testing.assert_array_equal(bench.read_fvecs(path), values)
            rows[1, 0] = 3
            rows.tofile(path)
            with self.assertRaisesRegex(ValueError, "dimension"):
                bench.read_fvecs(path)

    def test_haar_matrix_is_seeded_and_orthogonal(self):
        bench = load_benchmark()
        matrix = bench.haar_matrix(12, seed=7)
        self.assertEqual(matrix.dtype, np.dtype("float32"))
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(12), atol=2e-7)
        np.testing.assert_array_equal(matrix, bench.haar_matrix(12, seed=7))
        self.assertFalse(np.array_equal(matrix, bench.haar_matrix(12, seed=8)))

    def test_batch_search_covers_remainder_and_preserves_order(self):
        bench = load_benchmark()

        class ExactIndex:
            def __init__(self):
                self.batch_lengths = []

            def search(self, xq, k):
                self.batch_lengths.append(len(xq))
                return np.zeros((len(xq), k)), xq[:, :k].astype("int64")

        index = ExactIndex()
        queries = np.arange(14).reshape(7, 2).astype("float32")
        predictions = bench.search_batches(index, queries, 2, batch_size=3)
        np.testing.assert_array_equal(predictions, queries)
        self.assertEqual(index.batch_lengths, [3, 3, 1])

    def test_summary_pairs_each_seed_with_matching_1rht(self):
        bench = load_benchmark()
        rows = []
        for seed in (9, 2):
            for mode, recalls in (("rht1", [0.1, 0.2]), ("rht2", [0.2, 0.3])):
                rows.append(
                    dict(
                        index="EDEN",
                        bits=2,
                        batch_size=32,
                        mode=mode,
                        seed=seed,
                        recall_by_query=recalls,
                        search_seconds_median=0.01,
                        qps=200,
                        train_seconds=0.1,
                        add_seconds=0.1,
                        serialized_size_bytes=500,
                        code_size_bytes=12,
                        d_out=32,
                    )
                )
        summary = bench.summarize_rows(rows, bootstrap=100, bootstrap_seed=3)
        treatment = next(row for row in summary if row["mode"] == "rht2")
        paired = treatment["recall_delta_vs_1rht"]
        np.testing.assert_allclose(paired["ci95"], [0.1, 0.1])
        self.assertEqual(paired["seeds"], [2, 9])

    def test_checkpoint_retries_transient_windows_replace_failure(self):
        bench = load_benchmark()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "result.json"
            real_replace = Path.replace
            attempts = []

            def replace(path, destination):
                attempts.append(path)
                if len(attempts) == 1:
                    error = PermissionError("file held by sync client")
                    error.winerror = 5
                    raise error
                return real_replace(path, destination)

            with mock.patch.object(Path, "replace", replace):
                bench.write_results(target, {"rows": [1]})
            self.assertEqual(json.loads(target.read_text()), {"rows": [1]})
            self.assertEqual(len(attempts), 2)

    def test_nmse_distinguishes_pooled_and_per_vector_normalization(self):
        bench = load_benchmark()
        x = np.array([[0, 0], [1, 0], [2, 0]], dtype="float32")
        reconstructed = np.zeros_like(x)
        reconstructed[2, 0] = 1
        metrics = bench.reconstruction_metrics(x, reconstructed, np.zeros(2))
        self.assertAlmostEqual(metrics["nmse"], 2 / 5)
        self.assertAlmostEqual(metrics["vnmse"], (1 + 0.25) / 2)
        self.assertEqual(metrics["zero_residual_rows"], 1)
        self.assertEqual(metrics["zero_residual_with_error_rows"], 0)

    def test_run_limits_actual_blas_threads_and_restores_the_caller(self):
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError:
            self.skipTest("install the optional benchmark dependency threadpoolctl")

        bench = load_benchmark()
        args = bench.make_parser().parse_args(["--threads", "1"])

        def inspect_limits(unused_args):
            np.ones((8, 8)) @ np.ones((8, 8))
            pools = [pool for pool in threadpool_info() if pool["user_api"] == "blas"]
            self.assertTrue(pools)
            self.assertTrue(all(pool["num_threads"] == 1 for pool in pools))
            return "inspected"

        self.assertTrue(hasattr(bench, "run_with_thread_limits"))
        with threadpool_limits(limits=2, user_api="blas"):
            with mock.patch.object(bench, "run_with_thread_limits", inspect_limits):
                self.assertEqual(bench.run(args), "inspected")
            self.assertTrue(
                all(
                    pool["num_threads"] == 2
                    for pool in threadpool_info()
                    if pool["user_api"] == "blas"
                )
            )

    def test_wrong_host_fails_before_loading_data(self):
        bench = load_benchmark()
        args = bench.make_parser().parse_args(["--require-host", "expected-host"])
        with mock.patch.object(bench.platform, "node", return_value="other-host"):
            with self.assertRaisesRegex(ValueError, "required host"):
                bench.run(args)


class TestNativeRHTBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("FAISS_RHT_NATIVE_TESTS") != "1":
            raise unittest.SkipTest(
                "set FAISS_RHT_NATIVE_TESTS=1 to test the built extension"
            )
        import faiss

        cls.faiss = faiss
        cls.faiss.omp_set_num_threads(1)

    def test_tiny_end_to_end_checkpoint_contains_complete_metrics(self):
        bench = load_benchmark()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "tiny_correctness.json"
            args = bench.make_parser().parse_args(
                [
                    "--nt",
                    "16",
                    "--nb",
                    "12",
                    "--nq",
                    "3",
                    "--d",
                    "32",
                    "--k",
                    "3",
                    "--bits",
                    "2",
                    "--seeds",
                    "7",
                    "--batch-sizes",
                    "2",
                    "--repeats",
                    "1",
                    "--warmups",
                    "0",
                    "--bootstrap",
                    "5",
                    "--nmse-max-vectors",
                    "5",
                    "--output",
                    str(output),
                ]
            )
            with redirect_stdout(io.StringIO()):
                bench.run(args)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["rows"]), 10)
            self.assertEqual(len(report["summaries"]), 10)
            for row in report["rows"]:
                self.assertEqual(len(row["recall_by_query"]), 3)
                self.assertEqual(row["reconstruction"]["vector_count"], 5)
                self.assertTrue(np.isfinite(row["nmse"]))
                self.assertTrue(np.isfinite(row["vnmse"]))
            self.assertTrue(
                all(
                    pool["num_threads"] == 1
                    for pool in report["metadata"]["threadpools"]
                )
            )

    def test_all_modes_train_search_and_serialize(self):
        bench = load_benchmark()
        rng = np.random.default_rng(42)
        xt = rng.standard_normal((20, 48), dtype=np.float32)
        xb = rng.standard_normal((30, 48), dtype=np.float32)
        xq = rng.standard_normal((3, 48), dtype=np.float32)
        for family in ("EDEN", "RaBitQ"):
            for mode in bench.MODES:
                for pad in (False, True):
                    with self.subTest(family=family, mode=mode, pad=pad):
                        index, _, d_out = bench.make_index(
                            self.faiss, 48, family, 2, mode, 7, 4, pad
                        )
                        self.assertEqual(
                            d_out, 64 if pad or mode.startswith("rht") else 48
                        )
                        index.train(xt)
                        index.add(xb)
                        expected = index.search(xq, 5)
                        restored = self.faiss.deserialize_index(
                            self.faiss.serialize_index(index)
                        )
                        actual = restored.search(xq, 5)
                        np.testing.assert_array_equal(actual[1], expected[1])
                        np.testing.assert_array_equal(actual[0], expected[0])

    def test_eden_nmse_uses_full_native_decoder(self):
        bench = load_benchmark()
        rng = np.random.default_rng(6)
        xb = rng.standard_normal((20, 32), dtype=np.float32)
        index = self.faiss.index_factory(32, "EDEN2")
        index.train(xb + 0.1)
        index.add(xb)
        reconstructed = bench.reconstruct_leaf(self.faiss, index, 0, len(xb))
        np.testing.assert_array_equal(
            reconstructed, index.sa_decode(index.sa_encode(xb))
        )

    def test_multibit_rabitq_reconstruction_matches_full_native_scores(self):
        bench = load_benchmark()
        rng = np.random.default_rng(9)
        xb = rng.standard_normal((20, 32), dtype=np.float32)
        queries = rng.standard_normal((3, 32), dtype=np.float32)
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                index = self.faiss.index_factory(32, f"RaBitQ{bits}")
                index.qb = 0
                index.train(xb + 0.1)
                index.add(xb)
                reconstructed = bench.reconstruct_leaf(self.faiss, index, 0, len(xb))
                self.assertGreater(
                    np.linalg.norm(
                        reconstructed - index.sa_decode(index.sa_encode(xb))
                    ),
                    0.01,
                )
                center = self.faiss.vector_to_array(index.center)
                residuals, qr = xb - center, queries - center
                expected = (qr * qr).sum(axis=1)[:, None] + (residuals * residuals).sum(
                    axis=1
                )[None, :]
                expected -= 2 * qr @ (reconstructed - center).T
                expected = np.maximum(expected, 0)
                distances, ids = index.search(queries, len(xb))
                np.testing.assert_allclose(
                    distances,
                    np.take_along_axis(expected, ids, axis=1),
                    atol=3e-5,
                    rtol=2e-6,
                )


if __name__ == "__main__":
    unittest.main()
