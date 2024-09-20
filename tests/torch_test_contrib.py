# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch  # usort: skip
import unittest   # usort: skip
import numpy as np   # usort: skip

import faiss   # usort: skip
import faiss.contrib.torch_utils  # usort: skip
from faiss.contrib import datasets
from faiss.contrib.torch import clustering



class TestTorchUtilsCPU(unittest.TestCase):
    # tests add, search
    def test_lookup(self):
        d = 128
        index = faiss.IndexFlatL2(d)

        # Add to CPU index with torch CPU
        xb_torch = torch.rand(10000, d)
        index.add(xb_torch)

        # Test reconstruct
        y_torch = index.reconstruct(10)
        self.assertTrue(torch.equal(y_torch, xb_torch[10]))

        # Add to CPU index with numpy CPU
        xb_np = torch.rand(500, d).numpy()
        index.add(xb_np)
        self.assertEqual(index.ntotal, 10500)

        y_np = np.zeros(d, dtype=np.float32)
        index.reconstruct(10100, y_np)
        self.assertTrue(np.array_equal(y_np, xb_np[100]))

        # Search with np cpu
        xq_torch = torch.rand(10, d, dtype=torch.float32)
        d_np, I_np = index.search(xq_torch.numpy(), 5)

        # Search with torch cpu
        d_torch, I_torch = index.search(xq_torch, 5)

        # The two should be equivalent
        self.assertTrue(np.array_equal(d_np, d_torch.numpy()))
        self.assertTrue(np.array_equal(I_np, I_torch.numpy()))

        # Search with np cpu using pre-allocated arrays
        d_np_input = np.zeros((10, 5), dtype=np.float32)
        I_np_input = np.zeros((10, 5), dtype=np.int64)
        index.search(xq_torch.numpy(), 5, d_np_input, I_np_input)

        self.assertTrue(np.array_equal(d_np, d_np_input))
        self.assertTrue(np.array_equal(I_np, I_np_input))

        # Search with torch cpu using pre-allocated arrays
        d_torch_input = torch.zeros(10, 5, dtype=torch.float32)
        I_torch_input = torch.zeros(10, 5, dtype=torch.int64)
        index.search(xq_torch, 5, d_torch_input, I_torch_input)

        self.assertTrue(np.array_equal(d_torch_input.numpy(), d_np))
        self.assertTrue(np.array_equal(I_torch_input.numpy(), I_np))

    # tests train, add_with_ids
    def test_train_add_with_ids(self):
        d = 32
        nlist = 5

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        xb = torch.rand(1000, d, dtype=torch.float32)
        index.train(xb)

        # Test add_with_ids with torch cpu
        ids = torch.arange(1000, 1000 + xb.shape[0], dtype=torch.int64)
        index.add_with_ids(xb, ids)
        _, I = index.search(xb[10:20], 1)
        self.assertTrue(torch.equal(I.view(10), ids[10:20]))

        # Test add_with_ids with numpy
        index.reset()
        index.train(xb.numpy())
        index.add_with_ids(xb.numpy(), ids.numpy())
        _, I = index.search(xb.numpy()[10:20], 1)
        self.assertTrue(np.array_equal(I.reshape(10), ids.numpy()[10:20]))

    # tests reconstruct, reconstruct_n
    def test_reconstruct(self):
        d = 32
        index = faiss.IndexFlatL2(d)

        xb = torch.rand(100, d, dtype=torch.float32)
        index.add(xb)

        # Test reconstruct with torch cpu (native return)
        y = index.reconstruct(7)
        self.assertTrue(torch.equal(xb[7], y))

        # Test reconstruct with numpy output provided
        y = np.empty(d, dtype=np.float32)
        index.reconstruct(11, y)
        self.assertTrue(np.array_equal(xb.numpy()[11], y))

        # Test reconstruct with torch cpu output providesd
        y = torch.empty(d, dtype=torch.float32)
        index.reconstruct(12, y)
        self.assertTrue(torch.equal(xb[12], y))

        # Test reconstruct_n with torch cpu (native return)
        y = index.reconstruct_n(10, 10)
        self.assertTrue(torch.equal(xb[10:20], y))

        # Test reconstruct with numpy output provided
        y = np.empty((10, d), dtype=np.float32)
        index.reconstruct_n(20, 10, y)
        self.assertTrue(np.array_equal(xb.cpu().numpy()[20:30], y))

        # Test reconstruct_n with torch cpu output provided
        y = torch.empty(10, d, dtype=torch.float32)
        index.reconstruct_n(40, 10, y)
        self.assertTrue(torch.equal(xb[40:50].cpu(), y))

    # tests assign
    def test_assign(self):
        d = 32
        index = faiss.IndexFlatL2(d)
        xb = torch.rand(1000, d, dtype=torch.float32)
        index.add(xb)

        index_ref = faiss.IndexFlatL2(d)
        index_ref.add(xb.numpy())

        # Test assign with native cpu output
        xq = torch.rand(10, d, dtype=torch.float32)
        labels = index.assign(xq, 5)
        labels_ref = index_ref.assign(xq.cpu(), 5)

        self.assertTrue(torch.equal(labels, labels_ref))

        # Test assign with np input
        labels = index.assign(xq.numpy(), 5)
        labels_ref = index_ref.assign(xq.numpy(), 5)
        self.assertTrue(np.array_equal(labels, labels_ref))

        # Test assign with numpy output provided
        labels = np.empty((xq.shape[0], 5), dtype='int64')
        index.assign(xq.numpy(), 5, labels)
        self.assertTrue(np.array_equal(labels, labels_ref))

        # Test assign with torch cpu output provided
        labels = torch.empty(xq.shape[0], 5, dtype=torch.int64)
        index.assign(xq, 5, labels)
        labels_ref = index_ref.assign(xq, 5)
        self.assertTrue(torch.equal(labels, labels_ref))

    # tests remove_ids
    def test_remove_ids(self):
        # only implemented for cpu index + numpy at the moment
        d = 32
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 5)
        index.make_direct_map()
        index.set_direct_map_type(faiss.DirectMap.Hashtable)

        xb = torch.rand(1000, d, dtype=torch.float32)
        ids = torch.arange(1000, 1000 + xb.shape[0], dtype=torch.int64)
        index.train(xb)
        index.add_with_ids(xb, ids)

        ids_remove = np.array([1010], dtype=np.int64)
        index.remove_ids(ids_remove)

        # We should find this
        y = index.reconstruct(1011)
        self.assertTrue(np.array_equal(xb[11].numpy(), y))

        # We should not find this
        with self.assertRaises(RuntimeError):
            y = index.reconstruct(1010)

        # Torch not yet supported
        ids_remove = torch.tensor([1012], dtype=torch.int64)
        with self.assertRaises(AssertionError):
            index.remove_ids(ids_remove)

    # tests update_vectors
    def test_update_vectors(self):
        d = 32
        quantizer_np = faiss.IndexFlatL2(d)
        index_np = faiss.IndexIVFFlat(quantizer_np, d, 5)
        index_np.make_direct_map()
        index_np.set_direct_map_type(faiss.DirectMap.Hashtable)

        quantizer_torch = faiss.IndexFlatL2(d)
        index_torch = faiss.IndexIVFFlat(quantizer_torch, d, 5)
        index_torch.make_direct_map()
        index_torch.set_direct_map_type(faiss.DirectMap.Hashtable)

        xb = torch.rand(1000, d, dtype=torch.float32)
        ids = torch.arange(1000, 1000 + xb.shape[0], dtype=torch.int64)

        index_np.train(xb.numpy())
        index_np.add_with_ids(xb.numpy(), ids.numpy())

        index_torch.train(xb)
        index_torch.add_with_ids(xb, ids)

        xb_up = torch.rand(10, d, dtype=torch.float32)
        ids_up = ids[0:10]

        index_np.update_vectors(ids_up.numpy(), xb_up.numpy())
        index_torch.update_vectors(ids_up, xb_up)

        xq = torch.rand(10, d, dtype=torch.float32)

        D_np, I_np = index_np.search(xq.numpy(), 5)
        D_torch, I_torch = index_torch.search(xq, 5)

        self.assertTrue(np.array_equal(D_np, D_torch.numpy()))
        self.assertTrue(np.array_equal(I_np, I_torch.numpy()))

    # tests range_search
    def test_range_search(self):
        torch.manual_seed(10)
        d = 32
        index = faiss.IndexFlatL2(d)
        xb = torch.rand(100, d, dtype=torch.float32)
        index.add(xb)

        # torch cpu as ground truth
        thresh = 2.9
        xq = torch.rand(10, d, dtype=torch.float32)
        lims, D, I = index.range_search(xq, thresh)

        # compare against np
        lims_np, D_np, I_np = index.range_search(xq.numpy(), thresh)

        self.assertTrue(np.array_equal(lims.numpy(), lims_np))
        self.assertTrue(np.array_equal(D.numpy(), D_np))
        self.assertTrue(np.array_equal(I.numpy(), I_np))

    # tests search_and_reconstruct
    def test_search_and_reconstruct(self):
        d = 32
        nlist = 10
        M = 4
        k = 5
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 4)

        xb = torch.rand(1000, d, dtype=torch.float32)
        index.train(xb)

        # different set
        xb = torch.rand(500, d, dtype=torch.float32)
        index.add(xb)

        # torch cpu as ground truth
        xq = torch.rand(10, d, dtype=torch.float32)
        D, I, R = index.search_and_reconstruct(xq, k)

        # compare against numpy
        D_np, I_np, R_np = index.search_and_reconstruct(xq.numpy(), k)

        self.assertTrue(np.array_equal(D.numpy(), D_np))
        self.assertTrue(np.array_equal(I.numpy(), I_np))
        self.assertTrue(np.array_equal(R.numpy(), R_np))

        # numpy input values
        D_input = np.zeros((xq.shape[0], k), dtype=np.float32)
        I_input = np.zeros((xq.shape[0], k), dtype=np.int64)
        R_input = np.zeros((xq.shape[0], k, d), dtype=np.float32)

        index.search_and_reconstruct(xq.numpy(), k, D_input, I_input, R_input)

        self.assertTrue(np.array_equal(D.numpy(), D_input))
        self.assertTrue(np.array_equal(I.numpy(), I_input))
        self.assertTrue(np.array_equal(R.numpy(), R_input))

        # torch input values
        D_input = torch.zeros(xq.shape[0], k, dtype=torch.float32)
        I_input = torch.zeros(xq.shape[0], k, dtype=torch.int64)
        R_input = torch.zeros(xq.shape[0], k, d, dtype=torch.float32)

        index.search_and_reconstruct(xq, k, D_input, I_input, R_input)

        self.assertTrue(torch.equal(D, D_input))
        self.assertTrue(torch.equal(I, I_input))
        self.assertTrue(torch.equal(R, R_input))

    # tests sa_encode, sa_decode
    def test_sa_encode_decode(self):
        d = 16
        index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)

        xb = torch.rand(1000, d, dtype=torch.float32)
        index.train(xb)

        # torch cpu as ground truth
        nq = 10
        xq = torch.rand(nq, d, dtype=torch.float32)
        encoded_torch = index.sa_encode(xq)

        # numpy cpu
        encoded_np = index.sa_encode(xq.numpy())

        self.assertTrue(np.array_equal(encoded_torch.numpy(), encoded_np))

        decoded_torch = index.sa_decode(encoded_torch)
        decoded_np = index.sa_decode(encoded_np)

        self.assertTrue(torch.equal(decoded_torch, torch.from_numpy(decoded_np)))

        # torch cpu as output parameter
        encoded_torch_param = torch.zeros(nq, d, dtype=torch.uint8)
        index.sa_encode(xq, encoded_torch_param)

        self.assertTrue(torch.equal(encoded_torch, encoded_torch))

        decoded_torch_param = torch.zeros(nq, d, dtype=torch.float32)
        index.sa_decode(encoded_torch, decoded_torch_param)

        self.assertTrue(torch.equal(decoded_torch, decoded_torch_param))

        # np as output parameter
        encoded_np_param = np.zeros((nq, d), dtype=np.uint8)
        index.sa_encode(xq.numpy(), encoded_np_param)

        self.assertTrue(np.array_equal(encoded_torch.numpy(), encoded_np_param))

        decoded_np_param = np.zeros((nq, d), dtype=np.float32)
        index.sa_decode(encoded_np_param, decoded_np_param)

        self.assertTrue(np.array_equal(decoded_np, decoded_np_param))

    def test_non_contiguous(self):
        d = 128
        index = faiss.IndexFlatL2(d)

        xb = torch.rand(d, 100).transpose(0, 1)

        with self.assertRaises(AssertionError):
            index.add(xb)

        # disabled since we now accept non-contiguous arrays
        # with self.assertRaises(ValueError):
        #    index.add(xb.numpy())


class TestClustering(unittest.TestCase):

    def test_python_kmeans(self):
        """ Test the python implementation of kmeans """
        ds = datasets.SyntheticDataset(32, 10000, 0, 0)
        x = ds.get_train()

        # bad distribution to stress-test split code
        xt = x[:10000].copy()
        xt[:5000] = x[0]

        km_ref = faiss.Kmeans(ds.d, 100, niter=10)
        km_ref.train(xt)
        err = faiss.knn(xt, km_ref.centroids, 1)[0].sum()

        xt_torch = torch.from_numpy(xt)
        data = clustering.DatasetAssign(xt_torch)
        centroids = clustering.kmeans(100, data, 10)
        centroids = centroids.numpy()
        err2 = faiss.knn(xt, centroids, 1)[0].sum()

        # 33498.332 33380.477
        # print(err, err2)        1/0
        self.assertLess(err2, err * 1.1)
