# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the mmap-backed DINO loaders; no GPU needed."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


try:
    import dino_data as _module
except ImportError:
    _path = Path(__file__).resolve().parents[1] / "benchs/e_means/dino_data.py"
    _spec = importlib.util.spec_from_file_location("dino_data", _path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
DinoBvecs = _module.DinoBvecs
DinoQueries = _module.DinoQueries
DinoTrainset = _module.DinoTrainset


class TestDinoBvecs(unittest.TestCase):
    def test_cpu_import_keeps_faiss_and_gpu_dependencies_lazy(self):
        import builtins

        original_import = builtins.__import__

        def data_import(name, *args, **kwargs):
            if name.split(".")[0] in ("faiss", "triton", "cutlass", "quack"):
                raise AssertionError("data loader imported " + name)
            return original_import(name, *args, **kwargs)

        from unittest import mock
        with mock.patch.object(builtins, "__import__", side_effect=data_import):
            spec = importlib.util.spec_from_file_location("dino_data_cpu", _module.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertNotIn("Emeans", vars(module))
        self.assertIs(module.torch, __import__("torch"))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.chunks = self.root / "chunked_base_10B"
        self.chunks.mkdir()
        self.data = np.arange(65, dtype=np.uint8).reshape(13, 5)
        self.write("00.bvecs", self.data[:4])
        self.write("01.bvecs", self.data[4:])

    def write(self, name, rows):
        with (self.chunks / name).open("wb") as stream:
            for row in rows:
                stream.write(len(row).to_bytes(4, "little", signed=True))
                stream.write(row.tobytes())

    def test_boundary_prefix_and_owned_output(self):
        source = DinoBvecs(self.root, n=11)
        self.assertEqual(source.n, 11)
        self.assertEqual(source.ends[-1], 13)
        self.assertEqual(len(source.maps), 2)
        self.assertTrue(all(isinstance(mapping, np.memmap) for mapping in source.maps))
        self.assertTrue(all(not mapping.flags.writeable for mapping in source.maps))
        np.testing.assert_array_equal(source.read(2, 7), self.data[2:7])
        output = source.read(2, 7)
        self.assertTrue(output.flags.owndata)
        output.fill(0)
        np.testing.assert_array_equal(source.read(2, 7), self.data[2:7])
        self.assertEqual(source.read(11, 11).shape, (0, 5))
        with self.assertRaises(IndexError):
            source.read(10, 12)

    def test_contiguous_reads_cover_prefix_without_aliasing(self):
        source = DinoBvecs(self.root, n=13)
        parts = [source.read(a, min(a + 3, source.n)) for a in range(0, source.n, 3)]
        self.assertEqual([len(part) for part in parts], [3, 3, 3, 3, 1])
        self.assertTrue(all(part.flags.owndata for part in parts))
        np.testing.assert_array_equal(np.concatenate(parts), self.data)
        parts[1].fill(0)
        np.testing.assert_array_equal(source.read(3, 6), self.data[3:6])

    def test_malformed_file_and_first_dimension(self):
        (self.chunks / "01.bvecs").write_bytes(b"\x05\x00")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoBvecs(self.root, n=13)
        (self.chunks / "01.bvecs").write_bytes(b"\x05\x00\x00\x00\x01")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoBvecs(self.root, n=13)
        (self.chunks / "01.bvecs").write_bytes(b"\x05\x00\x00\x00\x01\x02\x03\x04\x05")
        # A whole row is still valid; the requested population exceeds capacity.
        with self.assertRaisesRegex(ValueError, "corpus contains"):
            DinoBvecs(self.root, n=13)
        with (self.chunks / "00.bvecs").open("r+b") as stream:
            stream.write((0).to_bytes(4, "little", signed=True))
        with self.assertRaisesRegex(ValueError, "invalid bvecs dimension"):
            DinoBvecs(self.root, n=4)

    def test_population_and_range_validation(self):
        for n in (0, -1, 14):
            with self.assertRaises(ValueError):
                DinoBvecs(self.root, n=n)
        source = DinoBvecs(self.root, n=np.int64(13))
        for begin, end in ((-1, 1), (2, 1), (0, 14)):
            with self.assertRaises(IndexError):
                source.read(begin, end)

    def test_only_required_prefix_is_mapped_and_validated(self):
        source = DinoBvecs(self.root, n=4)
        self.assertEqual(len(source.maps), 1)
        self.assertEqual(source.ends[-1], 4)
        del source
        # Current bounded loader maps only chunks needed by the requested prefix.
        (self.chunks / "02.bvecs").write_bytes(b"\x05\x00")
        np.testing.assert_array_equal(DinoBvecs(self.root, n=4).read(0, 4), self.data[:4])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoBvecs(self.root, n=14)
        (self.chunks / "01.bvecs").write_bytes(b"\x05\x00")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoBvecs(self.root, n=5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
    def test_read_and_resident_staging_boundary(self):
        import torch

        rows = np.arange(65539 * 5, dtype=np.uint32).astype(np.uint8).reshape(-1, 5)
        self.write("00.bvecs", rows)
        source = DinoBvecs(self.root, n=len(rows) + 9)
        np.testing.assert_array_equal(source.read(0, len(rows)), rows)
        expected = np.concatenate((rows[-3:], self.data[4:7]))
        np.testing.assert_array_equal(source.read(len(rows) - 3, len(rows) + 3), expected)
        resident = source.resident(1, source.n - 1, "cpu")
        self.assertEqual(resident.dtype, torch.float32)
        self.assertEqual(resident.device.type, "cpu")
        expected = np.concatenate((rows[1:], self.data[4:-1])).astype(np.float32)
        np.testing.assert_array_equal(resident.numpy(), expected)

    def test_unfinished_download_rejects_required_prefix(self):
        marker = self.chunks / "01.bvecs.parts.json"
        marker.write_text("{}")
        # Preallocated size and valid headers do not establish completion.
        with self.assertRaisesRegex(ValueError, "unfinished"):
            DinoBvecs(self.root, n=13)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            DinoBvecs(self.root, n=5)
        prefix = DinoBvecs(self.root, n=4)
        self.assertEqual(len(prefix.maps), 1)
        np.testing.assert_array_equal(prefix.read(0, 4), self.data[:4])
        marker.unlink()
        # Completion is a constructor gate; original mapped files stay immutable.
        (self.chunks / "00.bvecs.parts.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unfinished"):
            DinoBvecs(self.root, n=4)

    def test_numbered_chunk_gap_is_not_skipped(self):
        (self.chunks / "00.bvecs").rename(self.chunks / "chunk_0000.bvecs")
        (self.chunks / "01.bvecs").rename(self.chunks / "chunk_0002.bvecs")
        with self.assertRaisesRegex(ValueError, "missing chunk"):
            DinoBvecs(self.root, n=13)
        with self.assertRaisesRegex(ValueError, "missing chunk"):
            DinoBvecs(self.root, n=5)
        source = DinoBvecs(self.root, n=4)
        np.testing.assert_array_equal(source.read(0, 4), self.data[:4])
        del source
        (self.chunks / "chunk_0000.bvecs").rename(self.chunks / "chunk_0001.bvecs")
        with self.assertRaisesRegex(ValueError, "missing chunk"):
            DinoBvecs(self.root, n=4)

    def write_queries(self, rows):
        packed = np.empty((len(rows), rows.shape[1] + 4), dtype=np.uint8)
        packed[:, :4] = np.frombuffer(rows.shape[1].to_bytes(4, "little"), dtype=np.uint8)
        packed[:, 4:] = rows
        path = self.root / "queries_clean.bvecs"
        packed.tofile(path)
        return path, packed

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
    def test_query_batches_transform_tails_and_owned_storage(self):
        import torch

        self.write_queries(self.data)
        queries = DinoQueries(self.root, d=5, n=len(self.data))
        mean = torch.arange(5, dtype=torch.float32)
        indices = np.random.RandomState(1234).permutation(len(self.data))[6:]
        expected = (torch.from_numpy(self.data[indices]).float() - mean) / 2.0
        self.assertEqual(queries.n, 7)
        for batch_size in (4, 7, 20):
            parts = list(queries.batches(mean, 2.0, batch_size))
            self.assertEqual([len(x) for x in parts],
                             [min(batch_size, len(expected) - a)
                              for a in range(0, len(expected), batch_size)])
            self.assertTrue(all(x.dtype is torch.float32 for x in parts))
            torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)
            parts[0].fill_(-1)
            torch.testing.assert_close(torch.cat(list(queries.batches(mean, 2.0, batch_size))),
                                       expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            list(queries.batches(mean, 2.0, 0))

    def test_query_file_size_and_every_header_validation(self):
        with self.assertRaises(FileNotFoundError):
            DinoQueries(self.root, d=5, n=len(self.data))
        path, packed = self.write_queries(self.data)
        with self.assertRaisesRegex(ValueError, "expected"):
            DinoQueries(self.root, d=5, n=len(self.data) + 1)
        packed[-1, :4] = np.frombuffer((4).to_bytes(4, "little"), dtype=np.uint8)
        packed.tofile(path)
        with self.assertRaisesRegex(ValueError, "headers"):
            DinoQueries(self.root, d=5, n=len(self.data))


class TestDinoTrainset(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "train_queries_99M.bvecs"
        self.data = np.arange(65, dtype=np.uint8).reshape(13, 5)
        self.packed = np.empty((13, 9), dtype=np.uint8)
        self.packed[:, :4] = np.frombuffer((5).to_bytes(4, "little"), dtype=np.uint8)
        self.packed[:, 4:] = self.data
        self.packed.tofile(self.path)

    def test_prefix_transformed_batches_tails_and_owned_storage(self):
        import torch

        source = DinoTrainset(self.root, n=11)
        self.assertIsInstance(source.vectors, np.memmap)
        self.assertFalse(source.vectors.flags.writeable)
        mean = torch.arange(5, dtype=torch.float32)
        expected = (torch.from_numpy(self.data[:11]).float() - mean) / 2
        for size in (4, 11, 20):
            parts = list(source.batches(mean, 2, size))
            self.assertEqual([len(x) for x in parts],
                             [min(size, 11 - begin) for begin in range(0, 11, size)])
            torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)
            parts[0].fill_(-1)
            torch.testing.assert_close(torch.cat(list(source.batches(mean, 2, size))), expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            list(source.batches(mean, 2, 0))

    def test_population_size_and_required_headers(self):
        import torch

        for n in (0, -1, 14):
            with self.assertRaises(ValueError):
                DinoTrainset(self.root, n)
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoTrainset(self.root, 4)
        self.packed.tofile(self.path)
        self.packed[7, :4] = np.frombuffer((4).to_bytes(4, "little"), dtype=np.uint8)
        self.packed.tofile(self.path)
        with self.assertRaisesRegex(ValueError, "headers"):
            list(DinoTrainset(self.root, 8).batches(torch.zeros(5), 1, 4))
        self.packed[0, :4] = 0
        self.packed.tofile(self.path)
        with self.assertRaisesRegex(ValueError, "dimension"):
            DinoTrainset(self.root, 4)

    def test_bounded_row_norms_preserve_values_and_reject_zero_chunk(self):
        import torch

        rows = torch.from_numpy(self.data)
        for dtype in (torch.uint8, torch.float32, torch.bfloat16):
            values = rows.to(dtype)
            expected = values.float().square().sum(1)
            for size in (None, 1, 4, 20):
                actual = _module._squared_row_norms(values, chunk_rows=size)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            _module._squared_row_norms(rows, chunk_rows=0)
        self.assertEqual(_module._squared_row_norms(rows[:0]).shape, (0,))


if __name__ == "__main__":
    unittest.main()
