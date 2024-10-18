# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
import faiss
import tempfile
import os
import io
import sys
import pickle
from multiprocessing.pool import ThreadPool


class TestIOVariants(unittest.TestCase):

    def test_io_error(self):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype('float32')
        index = faiss.IndexFlatL2(d)
        index.add(x)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)

            # should be fine
            faiss.read_index(fname)

            with open(fname, 'rb') as f:
                data = f.read()
            # now damage file
            with open(fname, 'wb') as f:
                f.write(data[:int(len(data) / 2)])

            # should make a nice readable exception that mentions the filename
            try:
                faiss.read_index(fname)
            except RuntimeError as e:
                if fname not in str(e):
                    raise
            else:
                raise

        finally:
            if os.path.exists(fname):
                os.unlink(fname)


class TestCallbacks(unittest.TestCase):

    def do_write_callback(self, bsz):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype('float32')
        index = faiss.IndexFlatL2(d)
        index.add(x)

        f = io.BytesIO()
        # test with small block size
        writer = faiss.PyCallbackIOWriter(f.write, 1234)

        if bsz > 0:
            writer = faiss.BufferedIOWriter(writer, bsz)

        faiss.write_index(index, writer)
        del writer   # make sure all writes committed

        if sys.version_info[0] < 3:
            buf = f.getvalue()
        else:
            buf = f.getbuffer()

        index2 = faiss.deserialize_index(np.frombuffer(buf, dtype='uint8'))

        self.assertEqual(index.d, index2.d)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.codes),
            faiss.vector_to_array(index2.codes)
        )

        # This is not a callable function: should raise an exception
        writer = faiss.PyCallbackIOWriter("blabla")
        self.assertRaises(
            Exception,
            faiss.write_index, index, writer
        )

    def test_buf_read(self):
        x = np.random.uniform(size=20)

        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            x.tofile(fname)

            with open(fname, 'rb') as f:
                reader = faiss.PyCallbackIOReader(f.read, 1234)

                bsz = 123
                reader = faiss.BufferedIOReader(reader, bsz)

                y = np.zeros_like(x)
                reader(faiss.swig_ptr(y), y.nbytes, 1)

            np.testing.assert_array_equal(x, y)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def do_read_callback(self, bsz):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype('float32')
        index = faiss.IndexFlatL2(d)
        index.add(x)

        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)

            with open(fname, 'rb') as f:
                reader = faiss.PyCallbackIOReader(f.read, 1234)

                if bsz > 0:
                    reader = faiss.BufferedIOReader(reader, bsz)

                index2 = faiss.read_index(reader)

            self.assertEqual(index.d, index2.d)
            np.testing.assert_array_equal(
                faiss.vector_to_array(index.codes),
                faiss.vector_to_array(index2.codes)
            )

            # This is not a callable function: should raise an exception
            reader = faiss.PyCallbackIOReader("blabla")
            self.assertRaises(
                Exception,
                faiss.read_index, reader
            )
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_write_callback(self):
        self.do_write_callback(0)

    def test_write_buffer(self):
        self.do_write_callback(123)
        self.do_write_callback(2345)

    def test_read_callback(self):
        self.do_read_callback(0)

    def test_read_callback_buffered(self):
        self.do_read_callback(123)
        self.do_read_callback(12345)

    def test_read_buffer(self):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype('float32')
        index = faiss.IndexFlatL2(d)
        index.add(x)

        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)

            reader = faiss.BufferedIOReader(
                faiss.FileIOReader(fname), 1234)

            index2 = faiss.read_index(reader)

            self.assertEqual(index.d, index2.d)
            np.testing.assert_array_equal(
                faiss.vector_to_array(index.codes),
                faiss.vector_to_array(index2.codes)
            )

        finally:
            del reader
            if os.path.exists(fname):
                os.unlink(fname)


    def test_transfer_pipe(self):
        """ transfer an index through a Unix pipe """

        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype('float32')
        index = faiss.IndexFlatL2(d)
        index.add(x)
        Dref, Iref = index.search(x, 10)

        rf, wf = os.pipe()

        # start thread that will decompress the index

        def index_from_pipe():
            reader = faiss.PyCallbackIOReader(lambda size: os.read(rf, size))
            return faiss.read_index(reader)

        with ThreadPool(1) as pool:
            fut = pool.apply_async(index_from_pipe, ())

            # write to pipe
            writer = faiss.PyCallbackIOWriter(lambda b: os.write(wf, b))
            faiss.write_index(index, writer)

            index2 = fut.get()

            # closing is not really useful but it does not hurt
            os.close(wf)
            os.close(rf)

        Dnew, Inew = index2.search(x, 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)


class PyOndiskInvertedLists:
    """ wraps an OnDisk object for use from C++ """

    def __init__(self, oil):
        self.oil = oil

    def list_size(self, list_no):
        return self.oil.list_size(list_no)

    def get_codes(self, list_no):
        oil = self.oil
        assert 0 <= list_no < oil.lists.size()
        l = oil.lists.at(list_no)
        with open(oil.filename, 'rb') as f:
            f.seek(l.offset)
            return f.read(l.size * oil.code_size)

    def get_ids(self, list_no):
        oil = self.oil
        assert 0 <= list_no < oil.lists.size()
        l = oil.lists.at(list_no)
        with open(oil.filename, 'rb') as f:
            f.seek(l.offset + l.capacity * oil.code_size)
            return f.read(l.size * 8)


class TestPickle(unittest.TestCase):

    def dump_load_factory(self, fs):
        xq = faiss.randn((25, 10), 123)
        xb = faiss.randn((25, 10), 124)

        index = faiss.index_factory(10, fs)
        index.train(xb)
        index.add(xb)
        Dref, Iref = index.search(xq, 4)

        buf = io.BytesIO()
        pickle.dump(index, buf)
        buf.seek(0)
        index2 = pickle.load(buf)

        Dnew, Inew = index2.search(xq, 4)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

    def test_flat(self):
        self.dump_load_factory("Flat")

    def test_hnsw(self):
        self.dump_load_factory("HNSW32")

    def test_ivf(self):
        self.dump_load_factory("IVF5,Flat")


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


class TestIVFPQRead(unittest.TestCase):
    def test_reader(self):
        d, n = 32, 1000
        xq = np.random.uniform(size=(n, d)).astype('float32')
        xb = np.random.uniform(size=(n, d)).astype('float32')

        index = faiss.index_factory(32, "IVF32,PQ16np", faiss.METRIC_L2)
        index.train(xb)
        index.add(xb)
        fd, fname = tempfile.mkstemp()
        os.close(fd)

        try:
            faiss.write_index(index, fname)

            index_a = faiss.read_index(fname)
            index_b = faiss.read_index(fname, faiss.IO_FLAG_SKIP_PRECOMPUTE_TABLE)

            Da, Ia = index_a.search(xq, 10)
            Db, Ib = index_b.search(xq, 10)
            np.testing.assert_array_equal(Ia, Ib)
            np.testing.assert_almost_equal(Da, Db, decimal=5)

            codes_a = index_a.sa_encode(xq)
            codes_b = index_b.sa_encode(xq)
            np.testing.assert_array_equal(codes_a, codes_b)

        finally:
            if os.path.exists(fname):
                os.unlink(fname)
