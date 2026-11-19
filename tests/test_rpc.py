# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import pickle
import unittest

from faiss.contrib.rpc import FileSock, RestrictedUnpickler


class MemorySocket:
    """A finite byte stream that fails promptly if a reader loops at EOF."""

    def __init__(self, data):
        self.stream = io.BytesIO(data)
        self.remaining_calls = len(data) + 1

    def recv(self, size):
        if self.remaining_calls <= 0:
            raise AssertionError("reader continued after EOF")
        self.remaining_calls -= 1
        return self.stream.read(size)


class TestFileSock(unittest.TestCase):

    def test_readline_stops_at_newline(self):
        stream = FileSock(MemorySocket(b"first\nsecond\n"))
        self.assertEqual(stream.readline(), b"first\n")
        self.assertEqual(stream.readline(), b"second\n")
        self.assertEqual(stream.readline(), b"")

    def test_readline_stops_at_eof(self):
        for data in [b"", b"unterminated"]:
            with self.subTest(data=data):
                self.assertEqual(FileSock(MemorySocket(data)).readline(), data)

    def test_protocol_zero_pickle(self):
        value = ("hello", 42, [1, 2, 3])
        data = pickle.dumps(value, protocol=0)
        stream = FileSock(MemorySocket(data))
        self.assertEqual(RestrictedUnpickler(stream).load(), value)


if __name__ == "__main__":
    unittest.main()
