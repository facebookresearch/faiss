# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest
import numpy as np
import faiss
import faiss.contrib.torch_utils

from faiss.contrib import datasets
from faiss.contrib.torch import clustering


def to_column_major_torch(x):
    if hasattr(torch, 'contiguous_format'):
        return x.t().clone(memory_format=torch.contiguous_format).t()
    else:
        # was default setting before memory_format was introduced
        return x.t().clone().t()

def to_column_major_numpy(x):
    return x.T.copy().T

class TestTorchUtilsGPU(unittest.TestCase):
    # tests add, search
    def test_lookup(self):
        cpu_index = faiss.IndexFlatL2(128)

        # Add to CPU index with np
        xb_torch = torch.rand(10000, 128)
        cpu_index.add(xb_torch.numpy())

        # Add to CPU index with torch GPU (should fail)
        xb_torch_gpu = torch.rand(10000, 128, device=torch.device('cuda', 0), dtype=torch.float32)
        with self.assertRaises(AssertionError):
            cpu_index.add(xb_torch_gpu)

        # Add to GPU with torch GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatL2(res, 128)
        gpu_index.add(xb_torch.cuda())

        # Search with torch CPU
        xq_torch_cpu = torch.rand(10, 128, dtype=torch.float32)
        d_torch_cpu, i_torch_cpu = gpu_index.search(xq_torch_cpu, 10)

        # Search with torch GPU
        xq_torch_gpu = xq_torch_cpu.cuda()
        d_torch_gpu, i_torch_gpu = gpu_index.search(xq_torch_gpu, 10)
        self.assertTrue(d_torch_gpu.is_cuda)
        self.assertTrue(i_torch_gpu.is_cuda)

        # Should be equivalent
        self.assertTrue(torch.equal(d_torch_cpu.cuda(), d_torch_gpu))
        self.assertTrue(torch.equal(i_torch_cpu.cuda(), i_torch_gpu))

        # Search with torch GPU using pre-allocated arrays
        new_d_torch_gpu = torch.zeros(10, 10, device=torch.device('cuda', 0), dtype=torch.float32)
        new_i_torch_gpu = torch.zeros(10, 10, device=torch.device('cuda', 0), dtype=torch.int64)
        gpu_index.search(xq_torch_gpu, 10, new_d_torch_gpu, new_i_torch_gpu)

        self.assertTrue(torch.equal(d_torch_cpu.cuda(), new_d_torch_gpu))
        self.assertTrue(torch.equal(i_torch_cpu.cuda(), new_i_torch_gpu))

        # Search with numpy CPU
        xq_np_cpu = xq_torch_cpu.numpy()
        d_np_cpu, i_np_cpu = gpu_index.search(xq_np_cpu, 10)
        self.assertEqual(type(d_np_cpu), np.ndarray)
        self.assertEqual(type(i_np_cpu), np.ndarray)

        self.assertTrue(np.array_equal(d_torch_cpu.numpy(), d_np_cpu))
        self.assertTrue(np.array_equal(i_torch_cpu.numpy(), i_np_cpu))

    # tests train, add_with_ids
    def test_train_add_with_ids(self):
        d = 32
        nlist = 5
        res = faiss.StandardGpuResources()
        res.noTempMemory()

        index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
        xb = torch.rand(1000, d, device=torch.device('cuda', 0), dtype=torch.float32)
        index.train(xb)

        ids = torch.arange(1000, 1000 + xb.shape[0], device=torch.device('cuda', 0), dtype=torch.int64)

        # Test add_with_ids with torch gpu
        index.add_with_ids(xb, ids)
        _, I = index.search(xb[10:20], 1)
        self.assertTrue(torch.equal(I.view(10), ids[10:20]))

        # Test add_with_ids with torch cpu
        index.reset()
        xb_cpu = xb.cpu()
        ids_cpu = ids.cpu()

        index.train(xb_cpu)
        index.add_with_ids(xb_cpu, ids_cpu)
        _, I = index.search(xb_cpu[10:20], 1)
        self.assertTrue(torch.equal(I.view(10), ids_cpu[10:20]))

        # Test add_with_ids with numpy
        index.reset()
        xb_np = xb.cpu().numpy()
        ids_np = ids.cpu().numpy()

        index.train(xb_np)
        index.add_with_ids(xb_np, ids_np)
        _, I = index.search(xb_np[10:20], 1)
        self.assertTrue(np.array_equal(I.reshape(10), ids_np[10:20]))

    # tests reconstruct, reconstruct_n
    def test_flat_reconstruct(self):
        d = 32
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        index = faiss.GpuIndexFlatL2(res, d)

        xb = torch.rand(100, d, device=torch.device('cuda', 0), dtype=torch.float32)
        index.add(xb)

        # Test reconstruct with torch gpu (native return)
        y = index.reconstruct(7)
        self.assertTrue(y.is_cuda)
        self.assertTrue(torch.equal(xb[7], y))

        # Test reconstruct with numpy output provided
        y = np.empty(d, dtype='float32')
        index.reconstruct(11, y)
        self.assertTrue(np.array_equal(xb.cpu().numpy()[11], y))

        # Test reconstruct with torch cpu output providesd
        y = torch.empty(d, dtype=torch.float32)
        index.reconstruct(12, y)
        self.assertTrue(torch.equal(xb[12].cpu(), y))

        # Test reconstruct with torch gpu output providesd
        y = torch.empty(d, device=torch.device('cuda', 0), dtype=torch.float32)
        index.reconstruct(13, y)
        self.assertTrue(torch.equal(xb[13], y))

        # Test reconstruct_n with torch gpu (native return)
        y = index.reconstruct_n(10, 10)
        self.assertTrue(y.is_cuda)
        self.assertTrue(torch.equal(xb[10:20], y))

        # Test reconstruct with numpy output provided
        y = np.empty((10, d), dtype='float32')
        index.reconstruct_n(20, 10, y)
        self.assertTrue(np.array_equal(xb.cpu().numpy()[20:30], y))

        # Test reconstruct_n with torch cpu output provided
        y = torch.empty(10, d, dtype=torch.float32)
        index.reconstruct_n(40, 10, y)
        self.assertTrue(torch.equal(xb[40:50].cpu(), y))

        # Test reconstruct_n with torch gpu output provided
        y = torch.empty(10, d, device=torch.device('cuda', 0), dtype=torch.float32)
        index.reconstruct_n(50, 10, y)
        self.assertTrue(torch.equal(xb[50:60], y))

    def test_ivfflat_reconstruct(self):
        d = 32
        nlist = 5
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        config = faiss.GpuIndexIVFFlatConfig()
        config.use_raft = False

        index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)

        xb = torch.rand(100, d, device=torch.device('cuda', 0), dtype=torch.float32)
        index.train(xb)
        index.add(xb)

        # Test reconstruct_n with torch gpu (native return)
        y = index.reconstruct_n(10, 10)
        self.assertTrue(y.is_cuda)
        self.assertTrue(torch.equal(xb[10:20], y))

        # Test reconstruct with numpy output provided
        y = np.empty((10, d), dtype='float32')
        index.reconstruct_n(20, 10, y)
        self.assertTrue(np.array_equal(xb.cpu().numpy()[20:30], y))

        # Test reconstruct_n with torch cpu output provided
        y = torch.empty(10, d, dtype=torch.float32)
        index.reconstruct_n(40, 10, y)
        self.assertTrue(torch.equal(xb[40:50].cpu(), y))

        # Test reconstruct_n with torch gpu output provided
        y = torch.empty(10, d, device=torch.device('cuda', 0), dtype=torch.float32)
        index.reconstruct_n(50, 10, y)
        self.assertTrue(torch.equal(xb[50:60], y))

    # tests assign
    def test_assign(self):
        d = 32
        res = faiss.StandardGpuResources()
        res.noTempMemory()

        index = faiss.GpuIndexFlatL2(res, d)
        xb = torch.rand(10000, d, device=torch.device('cuda', 0), dtype=torch.float32)
        index.add(xb)

        index_cpu = faiss.IndexFlatL2(d)
        index.copyTo(index_cpu)

        # Test assign with native gpu output
        # both input as gpu torch and input as cpu torch
        xq = torch.rand(10, d, device=torch.device('cuda', 0), dtype=torch.float32)

        labels = index.assign(xq, 5)
        labels_cpu = index_cpu.assign(xq.cpu(), 5)
        self.assertTrue(torch.equal(labels.cpu(), labels_cpu))

        # Test assign with np input
        labels = index.assign(xq.cpu().numpy(), 5)
        labels_cpu = index_cpu.assign(xq.cpu().numpy(), 5)
        self.assertTrue(np.array_equal(labels, labels_cpu))

        # Test assign with numpy output provided
        labels = np.empty((xq.shape[0], 5), dtype='int64')
        index.assign(xq.cpu().numpy(), 5, labels)
        self.assertTrue(np.array_equal(labels, labels_cpu))

        # Test assign with torch cpu output provided
        labels = torch.empty(xq.shape[0], 5, dtype=torch.int64)
        index.assign(xq.cpu(), 5, labels)
        labels_cpu = index_cpu.assign(xq.cpu(), 5)
        self.assertTrue(torch.equal(labels, labels_cpu))

    # tests remove_ids
    def test_remove_ids(self):
        # This is not currently implemented on GPU indices
        return

    # tests range_search
    def test_range_search(self):
        # This is not currently implemented on GPU indices
        return

    # tests search_and_reconstruct
    def test_search_and_reconstruct(self):
        # This is not currently implemented on GPU indices
        return

    # tests sa_encode, sa_decode
    def test_sa_encode_decode(self):
        # This is not currently implemented on GPU indices
        return

class TestTorchUtilsKnnGpu(unittest.TestCase):
    def test_knn_gpu(self, use_raft=False):
        torch.manual_seed(10)
        d = 32
        nb = 1024
        nq = 10
        k = 10
        res = faiss.StandardGpuResources()

        # make GT on torch cpu and test using IndexFlatL2
        xb = torch.rand(nb, d, dtype=torch.float32)
        xq = torch.rand(nq, d, dtype=torch.float32)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        gt_D, gt_I = index.search(xq, k)

        # for the GPU, we'll use a non-default stream
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            # test numpy inputs
            xb_np = xb.numpy()
            xq_np = xq.numpy()

            for xq_row_major in True, False:
                for xb_row_major in True, False:
                    if not xq_row_major:
                        xq_c = to_column_major_numpy(xq_np)
                        assert not xq_c.flags.contiguous
                    else:
                        xq_c = xq_np

                    if not xb_row_major:
                        xb_c = to_column_major_numpy(xb_np)
                        assert not xb_c.flags.contiguous
                    else:
                        xb_c = xb_np

                    D, I = faiss.knn_gpu(res, xq_c, xb_c, k, use_raft=use_raft)

                    self.assertTrue(torch.equal(torch.from_numpy(I), gt_I))
                    self.assertLess((torch.from_numpy(D) - gt_D).abs().max(), 1e-4)

            # test torch (cpu, gpu) inputs
            for is_cuda in True, False:
                for xq_row_major in True, False:
                    for xb_row_major in True, False:

                        if is_cuda:
                            xq_c = xq.cuda()
                            xb_c = xb.cuda()
                        else:
                            # also test torch cpu tensors
                            xq_c = xq
                            xb_c = xb

                        if not xq_row_major:
                            xq_c = to_column_major_torch(xq)
                            assert not xq_c.is_contiguous()

                        if not xb_row_major:
                            xb_c = to_column_major_torch(xb)
                            assert not xb_c.is_contiguous()

                        D, I = faiss.knn_gpu(res, xq_c, xb_c, k, use_raft=use_raft)

                        self.assertTrue(torch.equal(I.cpu(), gt_I))
                        self.assertLess((D.cpu() - gt_D).abs().max(), 1e-4)

                        # test on subset
                        try:
                            # This internally uses the current pytorch stream
                            D, I = faiss.knn_gpu(res, xq_c[6:8], xb_c, k, use_raft=use_raft)
                        except TypeError:
                            if not xq_row_major:
                                # then it is expected
                                continue
                            # otherwise it is an error
                            raise

                        self.assertTrue(torch.equal(I.cpu(), gt_I[6:8]))
                        self.assertLess((D.cpu() - gt_D[6:8]).abs().max(), 1e-4)

    @unittest.skipUnless(
        "RAFT" in faiss.get_compile_options(),
        "only if RAFT is compiled in")
    def test_knn_gpu_raft(self):
        self.test_knn_gpu(use_raft=True)

    def test_knn_gpu_datatypes(self, use_raft=False):
        torch.manual_seed(10)
        d = 10
        nb = 1024
        nq = 5
        k = 10
        res = faiss.StandardGpuResources()

        # make GT on torch cpu and test using IndexFlatL2
        xb = torch.rand(nb, d, dtype=torch.float32)
        xq = torch.rand(nq, d, dtype=torch.float32)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        gt_D, gt_I = index.search(xq, k)

        xb_c = xb.cuda().half()
        xq_c = xq.cuda().half()

        # use i32 output indices
        D = torch.zeros(nq, k, device=xb_c.device, dtype=torch.float32)
        I = torch.zeros(nq, k, device=xb_c.device, dtype=torch.int32)

        faiss.knn_gpu(res, xq_c, xb_c, k, D, I, use_raft=use_raft)

        self.assertTrue(torch.equal(I.long().cpu(), gt_I))
        self.assertLess((D.float().cpu() - gt_D).abs().max(), 1.5e-3)

        # Test using numpy
        D = np.zeros((nq, k), dtype=np.float32)
        I = np.zeros((nq, k), dtype=np.int32)

        xb_c = xb.half().numpy()
        xq_c = xq.half().numpy()

        faiss.knn_gpu(res, xq_c, xb_c, k, D, I, use_raft=use_raft)

        self.assertTrue(torch.equal(torch.from_numpy(I).long(), gt_I))
        self.assertLess((torch.from_numpy(D) - gt_D).abs().max(), 1.5e-3)


class TestTorchUtilsPairwiseDistanceGpu(unittest.TestCase):
    def test_pairwise_distance_gpu(self):
        torch.manual_seed(10)
        d = 32
        k = 100
        # To compare against IndexFlat, use nb == k
        nb = k
        nq = 10
        res = faiss.StandardGpuResources()

        # make GT on torch cpu and test using IndexFlatL2
        xb = torch.rand(nb, d, dtype=torch.float32)
        xq = torch.rand(nq, d, dtype=torch.float32)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        gt_D, _ = index.search(xq, k)

        # for the GPU, we'll use a non-default stream
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            # test numpy inputs
            xb_np = xb.numpy()
            xq_np = xq.numpy()

            for xq_row_major in True, False:
                for xb_row_major in True, False:
                    if not xq_row_major:
                        xq_c = to_column_major_numpy(xq_np)
                        assert not xq_c.flags.contiguous
                    else:
                        xq_c = xq_np

                    if not xb_row_major:
                        xb_c = to_column_major_numpy(xb_np)
                        assert not xb_c.flags.contiguous
                    else:
                        xb_c = xb_np

                    D = faiss.pairwise_distance_gpu(res, xq_c, xb_c)

                    # IndexFlat will sort the results, so we need to
                    # do the same on our end
                    D = np.sort(D, axis=1)

                    self.assertLess((torch.from_numpy(D) - gt_D).abs().max(), 1e-4)

            # test torch (cpu, gpu) inputs
            for is_cuda in True, False:
                for xq_row_major in True, False:
                    for xb_row_major in True, False:

                        if is_cuda:
                            xq_c = xq.cuda()
                            xb_c = xb.cuda()
                        else:
                            # also test torch cpu tensors
                            xq_c = xq
                            xb_c = xb

                        if not xq_row_major:
                            xq_c = to_column_major_torch(xq)
                            assert not xq_c.is_contiguous()

                        if not xb_row_major:
                            xb_c = to_column_major_torch(xb)
                            assert not xb_c.is_contiguous()

                        D = faiss.pairwise_distance_gpu(res, xq_c, xb_c)

                        # IndexFlat will sort the results, so we need to
                        # do the same on our end
                        D, _ = torch.sort(D, dim=1)

                        self.assertLess((D.cpu() - gt_D).abs().max(), 1e-4)

                        # test on subset
                        try:
                            # This internally uses the current pytorch stream
                            D = faiss.pairwise_distance_gpu(res, xq_c[4:8], xb_c)
                        except TypeError:
                            if not xq_row_major:
                                # then it is expected
                                continue
                            # otherwise it is an error
                            raise

                        # IndexFlat will sort the results, so we need to
                        # do the same on our end
                        print(D)
                        D, _ = torch.sort(D, dim=1)

                        self.assertLess((D.cpu() - gt_D[4:8]).abs().max(), 1e-4)


class TestClustering(unittest.TestCase):

    def test_python_kmeans(self):
        """ Test the python implementation of kmeans """
        ds = datasets.SyntheticDataset(32, 10000, 0, 0)
        x = ds.get_train()

        # bad distribution to stress-test split code
        xt = x[:10000].copy()
        xt[:5000] = x[0]

        # CPU baseline
        km_ref = faiss.Kmeans(ds.d, 100, niter=10)
        km_ref.train(xt)
        err = faiss.knn(xt, km_ref.centroids, 1)[0].sum()

        xt_torch = torch.from_numpy(xt).to("cuda:0")
        res = faiss.StandardGpuResources()
        data = clustering.DatasetAssignGPU(res, xt_torch)
        centroids = clustering.kmeans(100, data, 10)
        centroids = centroids.cpu().numpy()
        err2 = faiss.knn(xt, centroids, 1)[0].sum()

        # 33498.332 33380.477
        print(err, err2)
        self.assertLess(err2, err * 1.1)
