# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

import numpy as np
import unittest
import faiss
import torch


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        if x.is_cuda:
            D = torch.cuda.FloatTensor(n, k)
        else:
            D = torch.FloatTensor(n, k)
    else:
        assert D.__class__ in (torch.FloatTensor, torch.cuda.FloatTensor)
        assert D.size() == (n, k)
        assert D.is_contiguous()

    if I is None:
        if x.is_cuda:
            I = torch.cuda.LongTensor(n, k)
        else:
            I = torch.LongTensor(n, k)
    else:
        assert I.__class__ in (torch.LongTensor, torch.cuda.LongTensor)
        assert I.size() == (n, k)
        assert I.is_contiguous()
    torch.cuda.synchronize()
    xptr = x.storage().data_ptr()
    Iptr = I.storage().data_ptr()
    Dptr = D.storage().data_ptr()
    index.search_c(n, faiss.cast_integer_to_float_ptr(xptr),
                   k, faiss.cast_integer_to_float_ptr(Dptr),
                   faiss.cast_integer_to_long_ptr(Iptr))
    torch.cuda.synchronize()
    return D, I


class PytorchFaissInterop(unittest.TestCase):

    def test_interop(self):

        d = 16
        nq = 5
        nb = 20

        xq = faiss.randn(nq * d, 1234).reshape(nq, d)
        xb = faiss.randn(nb * d, 1235).reshape(nb, d)

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)
        index.add(xb)

        # reference CPU result
        Dref, Iref = index.search(xq, 5)

        # query is pytorch tensor (CPU)
        xq_torch = torch.FloatTensor(xq)

        D2, I2 = search_index_pytorch(index, xq_torch, 5)

        assert np.all(Iref == I2.numpy())

        # query is pytorch tensor (GPU)
        xq_torch = xq_torch.cuda()
        # no need for a sync here

        D3, I3 = search_index_pytorch(index, xq_torch, 5)

        # D3 and I3 are on torch tensors on GPU as well.
        # this does a sync, which is useful because faiss and
        # pytorch use different Cuda streams.
        res.syncDefaultStreamCurrentDevice()

        assert np.all(Iref == I3.cpu().numpy())


if __name__ == '__main__':
    unittest.main()
