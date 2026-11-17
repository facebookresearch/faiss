# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA ownership and read-ahead checks for the faithful SSD prefetcher."""

from contextlib import closing
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


def load(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    path = Path(__file__).resolve().parents[1] / "benchs/e_means" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DinoBvecs = load("dino_data").DinoBvecs
_ssd = load("ssd_streaming")
SSDRowSource, SSDPrefetch = _ssd.SSDRowSource, _ssd.SSDPrefetch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDinoStreaming(unittest.TestCase):
    def source(self, data):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        chunks = root / "chunked_base_10B"
        chunks.mkdir()
        split = len(data) // 3
        for name, rows in (("00", data[:split]), ("01", data[split:])):
            packed = np.empty((len(rows), data.shape[1] + 4), dtype=np.uint8)
            packed[:, :4] = np.frombuffer(data.shape[1].to_bytes(4, "little"), dtype=np.uint8)
            packed[:, 4:] = rows
            packed.tofile(chunks / (name + ".bvecs"))
        reader = DinoBvecs(root, n=len(data))
        source = SSDRowSource(reader.paths, reader.d, 0, reader.n, header_bytes=4)
        self.addCleanup(source.close)
        return source

    def loader(self, source, batch_size):
        return SSDPrefetch(
            source, "cuda", batch_size, n_prefetch=2,
            chunk_bytes=batch_size * source.row_bytes,
            ram_budget_bytes=4 * batch_size * source.d,
            read_threads=1, extract_threads=1,
        )

    def test_nondefault_stream_order_dtype_and_logical_tails(self):
        data = np.random.RandomState(8).randint(0, 256, (37, 8), dtype=np.uint8)
        source = self.source(data)
        consumer = torch.cuda.Stream()
        mean = torch.from_numpy(np.arange(8, dtype=np.float32)).to("cuda")
        consumer.wait_stream(torch.cuda.current_stream())
        for sequential in (False, True):
            loader = self.loader(source, 10)
            order = loader.order(9)
            outputs = []
            with torch.cuda.stream(consumer):
                with closing(loader.batches(
                    order, sequential=sequential,
                    mean=mean if sequential else None,
                    scale=2.0 if sequential else None,
                )) as iterator:
                    for rows in iterator:
                        self.assertEqual(rows.dtype, torch.float32)
                        outputs.append(rows.clone())
            consumer.synchronize()
            self.assertEqual([len(x) for x in outputs], [10, 10, 10, 7])
            expected = torch.from_numpy(data if sequential else data[order]).float()
            if sequential:
                expected = (expected - mean.cpu()) / 2.0
            torch.testing.assert_close(torch.cat(outputs).cpu(), expected, rtol=0, atol=0)

    def test_early_close_preserves_owned_output(self):
        data = np.arange(37 * 8, dtype=np.uint16).astype(np.uint8).reshape(37, 8)
        source = self.source(data)
        loader = self.loader(source, 10)
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            iterator = loader.batches(sequential=True)
            first = next(iterator)
            copied = first.clone()
            iterator.close()
            retained = first.clone()
        consumer.synchronize()
        torch.testing.assert_close(copied.cpu(), torch.from_numpy(data[:10]).float())
        torch.testing.assert_close(retained, copied)
        np.testing.assert_array_equal(source.read(10, 14), data[10:14])

    def test_read_ahead_exception_propagates(self):
        data = np.zeros((37, 8), dtype=np.uint8)
        source = self.source(data)
        loader = self.loader(source, 10)
        original = source.read

        def fail(begin, end):
            if begin >= loader.window_rows:
                raise OSError("injected read failure")
            return original(begin, end)

        source.read = fail
        with self.assertRaisesRegex(OSError, "injected read failure"):
            with closing(loader.batches(sequential=True)) as iterator:
                for rows in iterator:
                    rows.sum()


if __name__ == "__main__":
    unittest.main()
