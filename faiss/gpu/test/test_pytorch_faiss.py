# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest
import faiss
import torch

from faiss.contrib.pytorch_tensors import search_index_pytorch, search_raw_array_pytorch, using_stream

def to_column_major(x):
    if hasattr(torch, 'contiguous_format'):
        return x.t().clone(memory_format=torch.contiguous_format).t()
    else:
        # was default setting before memory_format was introduced
        return x.t().clone().t()

class PytorchFaissInterop(unittest.TestCase):

    def test_interop(self):
        d = 128
        nq = 100
        nb = 1000
        k = 10

        xq = faiss.randn(nq * d, 1234).reshape(nq, d)
        xb = faiss.randn(nb * d, 1235).reshape(nb, d)

        res = faiss.StandardGpuResources()

        # Let's run on a non-default stream
        s = torch.cuda.Stream()

        # Torch will run on this stream
        with torch.cuda.stream(s):
            # query is pytorch tensor (CPU and GPU)
            xq_torch_cpu = torch.FloatTensor(xq)
            xq_torch_gpu = xq_torch_cpu.cuda()

            index = faiss.GpuIndexFlatIP(res, d)
            index.add(xb)

            # Query with GPU tensor (this will be done on the current pytorch stream)
            D2, I2 = search_index_pytorch(res, index, xq_torch_gpu, k)
            Dref, Iref = index.search(xq, k)

            assert np.all(Iref == I2.cpu().numpy())

            # Query with CPU tensor
            D3, I3 = search_index_pytorch(res, index, xq_torch_cpu, k)

            assert np.all(Iref == I3.numpy())

    def test_raw_array_search(self):
        d = 32
        nb = 1024
        nq = 128
        k = 10

        # make GT on Faiss CPU

        xq = faiss.randn(nq * d, 1234).reshape(nq, d)
        xb = faiss.randn(nb * d, 1235).reshape(nb, d)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        gt_D, gt_I = index.search(xq, k)

        # resource object, can be re-used over calls
        res = faiss.StandardGpuResources()

        # Let's have pytorch use a non-default stream
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for xq_row_major in True, False:
                for xb_row_major in True, False:

                    # move to pytorch & GPU
                    xq_t = torch.from_numpy(xq).cuda()
                    xb_t = torch.from_numpy(xb).cuda()

                    if not xq_row_major:
                        xq_t = to_column_major(xq_t)
                        assert not xq_t.is_contiguous()

                    if not xb_row_major:
                        xb_t = to_column_major(xb_t)
                        assert not xb_t.is_contiguous()

                    D, I = search_raw_array_pytorch(res, xb_t, xq_t, k)

                    # back to CPU for verification
                    D = D.cpu().numpy()
                    I = I.cpu().numpy()

                    assert np.all(I == gt_I)
                    assert np.all(np.abs(D - gt_D).max() < 1e-4)

                    # test on subset
                    try:
                        # This internally uses the current pytorch stream
                        D, I = search_raw_array_pytorch(res, xb_t, xq_t[60:80], k)
                    except TypeError:
                        if not xq_row_major:
                            # then it is expected
                            continue
                        # otherwise it is an error
                        raise

                    # back to CPU for verification
                    D = D.cpu().numpy()
                    I = I.cpu().numpy()

                    assert np.all(I == gt_I[60:80])
                    assert np.all(np.abs(D - gt_D[60:80]).max() < 1e-4)


if __name__ == '__main__':
    unittest.main()
