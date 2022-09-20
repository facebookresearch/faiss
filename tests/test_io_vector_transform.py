import os
import tempfile
import unittest

import numpy as np
import faiss


class Test_IO_VectorTransform(unittest.TestCase):
    """
    test write_VectorTransform using IOWriter Pointer
    and read_VectorTransform using file name
    """
    def test_write_vector_transform(self):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype('float32')
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFSpectralHash(quantizer, d, n, 8, 1.0)
        index.train(x)
        index.add(x)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:

            writer = faiss.FileIOWriter(fname)
            faiss.write_VectorTransform(index.vt, writer)
            del writer

            vt = faiss.read_VectorTransform(fname)

            assert vt.d_in == index.vt.d_in
            assert vt.d_out == index.vt.d_out
            assert vt.is_trained

        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    """
    test write_VectorTransform using file name
    and read_VectorTransform using IOWriter Pointer
    """
    def test_read_vector_transform(self):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype('float32')
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFSpectralHash(quantizer, d, n, 8, 1.0)
        index.train(x)
        index.add(x)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:

            faiss.write_VectorTransform(index.vt, fname)

            reader = faiss.FileIOReader(fname)
            vt = faiss.read_VectorTransform(reader)
            del reader

            assert vt.d_in == index.vt.d_in
            assert vt.d_out == index.vt.d_out
            assert vt.is_trained

        finally:
            if os.path.exists(fname):
                os.unlink(fname)
