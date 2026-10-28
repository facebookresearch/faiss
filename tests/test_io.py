# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pickle
import platform
import tempfile
import unittest
from multiprocessing.pool import ThreadPool

import faiss
import numpy as np
from common_faiss_tests import get_dataset_2
from faiss.contrib.evaluation import check_ref_knn_with_draws


d = 32
nt = 2000
nb = 1000
nq = 200


class TestIOVariants(unittest.TestCase):

    def test_io_error(self):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype("float32")
        index = faiss.IndexFlatL2(d)
        index.add(x)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)

            with open(fname, "rb") as f:
                data = f.read()
            with open(fname, "wb") as f:
                f.write(data[: int(len(data) / 2)])

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
        x = np.random.uniform(size=(n, d)).astype("float32")
        index = faiss.IndexFlatL2(d)
        index.add(x)

        f = io.BytesIO()
        # test with small block size
        writer = faiss.PyCallbackIOWriter(f.write, 1234)

        if bsz > 0:
            writer = faiss.BufferedIOWriter(writer, bsz)

        faiss.write_index(index, writer)
        del writer  # make sure all writes committed

        buf = f.getbuffer()

        index2 = faiss.deserialize_index(np.frombuffer(buf, dtype="uint8"))

        self.assertEqual(index.d, index2.d)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.codes),
            faiss.vector_to_array(index2.codes),
        )

        writer = faiss.PyCallbackIOWriter("blabla")
        self.assertRaises(Exception, faiss.write_index, index, writer)

    def do_read_callback(self, bsz):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype("float32")
        index = faiss.IndexFlatL2(d)
        index.add(x)

        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, fname)

            with open(fname, "rb") as f:
                reader = faiss.PyCallbackIOReader(f.read, 1234)

                if bsz > 0:
                    reader = faiss.BufferedIOReader(reader, bsz)

                index2 = faiss.read_index(reader)

            self.assertEqual(index.d, index2.d)
            np.testing.assert_array_equal(
                faiss.vector_to_array(index.codes),
                faiss.vector_to_array(index2.codes),
            )

            reader = faiss.PyCallbackIOReader("blabla")
            self.assertRaises(Exception, faiss.read_index, reader)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_write_callback(self):
        self.do_write_callback(0)

    def test_write_buffer(self):
        self.do_write_callback(123)

    def test_read_callback(self):
        self.do_read_callback(0)

    def test_read_callback_buffered(self):
        self.do_read_callback(123)

    def test_transfer_pipe(self):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype("float32")
        index = faiss.IndexFlatL2(d)
        index.add(x)
        Dref, Iref = index.search(x, 10)

        rf, wf = os.pipe()

        def index_from_pipe():
            reader = faiss.PyCallbackIOReader(lambda size: os.read(rf, size))
            return faiss.read_index(reader)

        with ThreadPool(1) as pool:
            fut = pool.apply_async(index_from_pipe, ())

            writer = faiss.PyCallbackIOWriter(lambda b: os.write(wf, b))
            faiss.write_index(index, writer)

            index2 = fut.get()

            os.close(wf)
            os.close(rf)

        Dnew, Inew = index2.search(x, 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)


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

        # Verify deserialized index is serializable again
        buf2 = io.BytesIO()
        pickle.dump(index2, buf2)
        buf2.seek(0)
        index3 = pickle.load(buf2)
        Dnew3, Inew3 = index3.search(xq, 4)
        np.testing.assert_array_equal(Iref, Inew3)
        np.testing.assert_array_equal(Dref, Dnew3)

    def test_ivf(self):
        self.dump_load_factory("IVF5,Flat")


class Test_IO_VectorTransform(unittest.TestCase):

    def test_write_vector_transform(self):
        d, n = 32, 1000
        x = np.random.uniform(size=(n, d)).astype("float32")
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

            # Verify deserialized VectorTransform is serializable again
            fd2, fname2 = tempfile.mkstemp()
            os.close(fd2)
            try:
                faiss.write_VectorTransform(vt, fname2)
                vt2 = faiss.read_VectorTransform(fname2)
                assert vt2.d_in == index.vt.d_in
                assert vt2.d_out == index.vt.d_out
                assert vt2.is_trained
            finally:
                if os.path.exists(fname2):
                    os.unlink(fname2)

        finally:
            if os.path.exists(fname):
                os.unlink(fname)


class Test_IO_PQ(unittest.TestCase):

    def test_io_pq(self):
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)
        index = faiss.IndexPQ(d, 4, 4)
        index.train(xt)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        try:
            faiss.write_ProductQuantizer(index.pq, fname)
            read_pq = faiss.read_ProductQuantizer(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)
        self.assertEqual(index.pq.M, read_pq.M)
        self.assertEqual(index.pq.nbits, read_pq.nbits)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.pq.centroids),
            faiss.vector_to_array(read_pq.centroids),
        )

        # Verify deserialized PQ is serializable again
        fd2, fname2 = tempfile.mkstemp()
        os.close(fd2)
        try:
            faiss.write_ProductQuantizer(read_pq, fname2)
            read_pq2 = faiss.read_ProductQuantizer(fname2)
        finally:
            if os.path.exists(fname2):
                os.unlink(fname2)
        self.assertEqual(index.pq.M, read_pq2.M)
        self.assertEqual(index.pq.nbits, read_pq2.nbits)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.pq.centroids),
            faiss.vector_to_array(read_pq2.centroids),
        )


class Test_IO_IndexLSH(unittest.TestCase):

    def test_io_lsh(self):
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)
        index_lsh = faiss.IndexLSH(d, 32, True, True)
        index_lsh.train(xt)
        index_lsh.add(xb)
        D, I = index_lsh.search(xq, 10)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        try:
            faiss.write_index(index_lsh, fname)

            reader = faiss.BufferedIOReader(faiss.FileIOReader(fname), 1234)
            read_index_lsh = faiss.read_index(reader)
            # Delete reader to prevent [WinError 32] The process cannot
            # access the file because it is being used by another process
            del reader

            np.testing.assert_array_equal(
                faiss.vector_to_array(index_lsh.codes),
                faiss.vector_to_array(read_index_lsh.codes),
            )
            D_read, I_read = read_index_lsh.search(xq, 10)

            np.testing.assert_array_equal(D, D_read)
            np.testing.assert_array_equal(I, I_read)

            # Verify deserialized index is serializable again
            data3 = faiss.serialize_index(read_index_lsh)
            index3 = faiss.deserialize_index(data3)
            D3, I3 = index3.search(xq, 10)
            np.testing.assert_array_equal(D, D3)
            np.testing.assert_array_equal(I, I3)

        finally:
            if os.path.exists(fname):
                os.unlink(fname)


class Test_IO_IndexIVFSpectralHash(unittest.TestCase):

    def test_io_ivf_spectral_hash(self):
        nlist = 1000
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFSpectralHash(quantizer, d, nlist, 8, 1.0)
        index.train(xt)
        index.add(xb)
        D, I = index.search(xq, 10)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        try:
            faiss.write_index(index, fname)

            reader = faiss.BufferedIOReader(faiss.FileIOReader(fname), 1234)
            read_index = faiss.read_index(reader)
            del reader

            D_read, I_read = read_index.search(xq, 10)
            np.testing.assert_array_equal(D, D_read)
            np.testing.assert_array_equal(I, I_read)

            # Verify deserialized index is serializable again
            data3 = faiss.serialize_index(read_index)
            index3 = faiss.deserialize_index(data3)
            D3, I3 = index3.search(xq, 10)
            np.testing.assert_array_equal(D, D3)
            np.testing.assert_array_equal(I, I3)

        finally:
            if os.path.exists(fname):
                os.unlink(fname)


class TestIVFPQRead(unittest.TestCase):
    def test_reader(self):
        d, n = 32, 1000
        rs = np.random.RandomState(123)
        xq = rs.uniform(size=(n, d)).astype("float32")
        xb = rs.uniform(size=(n, d)).astype("float32")

        index = faiss.index_factory(32, "IVF32,PQ16np", faiss.METRIC_L2)
        index.train(xb)
        index.add(xb)
        fd, fname = tempfile.mkstemp()
        os.close(fd)

        try:
            faiss.write_index(index, fname)

            index_a = faiss.read_index(fname)
            index_b = faiss.read_index(
                fname, faiss.IO_FLAG_SKIP_PRECOMPUTE_TABLE
            )

            Da, Ia = index_a.search(xq, 10)
            Db, Ib = index_b.search(xq, 10)
            # The precomputed table changes the order of float operations, so
            # near-tied neighbors can swap.
            check_ref_knn_with_draws(Da, Ia, Db, Ib)

            # Verify deserialized indexes are serializable again
            data3 = faiss.serialize_index(index_a)
            index3 = faiss.deserialize_index(data3)
            D3, I3 = index3.search(xq, 10)
            np.testing.assert_array_equal(Da, D3)
            np.testing.assert_array_equal(Ia, I3)

        finally:
            if os.path.exists(fname):
                os.unlink(fname)


class TestIOFlatMMap(unittest.TestCase):
    @unittest.skipIf(
        platform.system() not in ["Windows", "Linux", "Darwin"],
        "supported OSes only",
    )
    def test_mmap_ivf(self):
        d, nlist = 32, 64
        xt, xb, xq = get_dataset_2(d, 2000, 5000, 50)
        index = faiss.index_factory(d, f"IVF{nlist},Flat")
        index.train(xt)
        index.add(xb)
        index.nprobe = 8
        Dref, Iref = index.search(xq, 10)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        index2 = None
        try:
            faiss.write_index(index, fname)
            index2 = faiss.read_index(fname, faiss.IO_FLAG_MMAP_IFC)
            index2.nprobe = 8
            Dnew, Inew = index2.search(xq, 10)
            np.testing.assert_array_equal(Iref, Inew)
            np.testing.assert_array_equal(Dref, Dnew)
        finally:
            del index2
            if os.path.exists(fname):
                # On Windows, index2 holds the file handle until it is
                # garbage collected, so the file may not be deletable yet
                try:
                    os.unlink(fname)
                except Exception:
                    pass

    @unittest.skipIf(
        platform.system() not in ["Windows", "Linux", "Darwin"],
        "supported OSes only",
    )
    def test_mmap_reset(self):
        xt, xb, xq = get_dataset_2(32, 0, 100, 50)
        index = faiss.IndexFlatL2(32)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        index2 = None
        try:
            faiss.write_index(index, fname)
            index2 = faiss.read_index(fname, faiss.IO_FLAG_MMAP_IFC)
            self.assertFalse(index2.codes.is_owned)
            index2.reset()
            self.assertEqual(index2.ntotal, 0)
            index2.add(xb)
            Dnew, Inew = index2.search(xq, 10)
            np.testing.assert_array_equal(Iref, Inew)
            np.testing.assert_array_equal(Dref, Dnew)
        finally:
            del index2
            if os.path.exists(fname):
                try:
                    os.unlink(fname)
                except Exception:
                    pass


class TestIORoundTrip(unittest.TestCase):
    """Round-trip serialize/deserialize tests covering index types
    and sub-objects not covered by other Python tests."""

    def test_index_pq(self):
        xt, xb, xq = get_dataset_2(d, nt, nb, nq)
        index = faiss.index_factory(d, "PQ4np")
        index.train(xt)
        index.add(xb)
        Dref, Iref = index.search(xq, 5)

        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        D2, I2 = index2.search(xq, 5)
        np.testing.assert_array_equal(Iref, I2)
        np.testing.assert_array_equal(Dref, D2)

    def test_vector_transform_pca(self):
        xt, _, xq = get_dataset_2(d, nt, nb, nq)
        pca = faiss.PCAMatrix(d, 16)
        pca.train(xt)

        writer = faiss.VectorIOWriter()
        faiss.write_VectorTransform(pca, writer)

        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(
            np.array(faiss.vector_to_array(writer.data)), reader.data
        )
        pca2 = faiss.read_VectorTransform(reader)

        ref = pca.apply(xq)
        out = pca2.apply(xq)
        np.testing.assert_array_equal(ref, out)

    def test_vector_transform_hadamard_rotation(self):
        xt, _, xq = get_dataset_2(d, nt, nb, nq)
        fr = faiss.HadamardRotation(d, 42)

        writer = faiss.VectorIOWriter()
        faiss.write_VectorTransform(fr, writer)

        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(
            np.array(faiss.vector_to_array(writer.data)), reader.data
        )
        fr2 = faiss.read_VectorTransform(reader)

        ref = fr.apply(xq)
        out = fr2.apply(xq)
        np.testing.assert_array_equal(ref, out)

    def test_null_index(self):
        """Serializing None / null index round-trips to None."""
        writer = faiss.VectorIOWriter()
        faiss.write_index(None, writer)

        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(
            np.array(faiss.vector_to_array(writer.data)), reader.data
        )
        index2 = faiss.read_index(reader)

        self.assertIsNone(index2)


class Test_IO_HNSW(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        self.xt, self.xb, self.xq = get_dataset_2(d, nt, nb, nq)

    def _test_roundtrip(self, index):
        for _trips in range(2):
            read_index = faiss.deserialize_index(faiss.serialize_index(index))
            self.assertEqual(index.hnsw.efSearch, read_index.hnsw.efSearch)
            self.assertEqual(
                index.hnsw.efConstruction, read_index.hnsw.efConstruction
            )
            D, I = index.search(self.xq, 10)
            D_read, I_read = read_index.search(self.xq, 10)
            np.testing.assert_array_equal(D, D_read)
            np.testing.assert_array_equal(I, I_read)
            index = read_index  # test re-serialization

    def test_hnsw_prq(self):
        """Exercise HNSW with aritrary storage"""
        storage = faiss.IndexProductResidualQuantizer(
            d, 8, 2, 4, faiss.METRIC_L2, faiss.AdditiveQuantizer.ST_norm_qint8
        )
        index = faiss.IndexHNSW(storage, 32)
        del storage
        index.train(self.xt)
        index.add(self.xb)
        self._test_roundtrip(index)

    def test_hnsw_noadd(self):
        """Exercise HNSW with a swapped-out storage"""
        index = faiss.IndexHNSWFlat(d, 16)
        index.train(self.xt)
        index.add(self.xb)
        # Build the graph with Flat storage, then verify that swapping in an
        # equivalent RaBitQ storage preserves useful search results.
        I_flat = index.storage.assign(self.xq, 10)
        I_hnsw_flat = index.assign(self.xq, 10)
        index.storage = faiss.IndexRaBitQ(d)
        index.storage.train(self.xb)
        index.storage.add(self.xb)
        I_hnsw_rabitq = index.assign(self.xq, 10)
        recall_hnsw_flat = faiss.eval_intersection(I_hnsw_flat, I_flat)
        recall_hnsw_rabitq = faiss.eval_intersection(I_hnsw_rabitq, I_flat)
        self.assertGreater(recall_hnsw_flat, 0.999)
        self.assertGreater(recall_hnsw_rabitq, 0.999)
        self._test_roundtrip(index)
