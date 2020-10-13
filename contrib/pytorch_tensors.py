# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import torch
import contextlib

def swig_ptr_from_FloatTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch trensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch trensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

@contextlib.contextmanager
def using_stream(res, pytorch_stream=None):
    r""" Creates a scoping object to make Faiss GPU use the same stream
         as pytorch, based on torch.cuda.current_stream().
         Or, a specific pytorch stream can be passed in as a second
         argument, in which case we will use that stream.
    """

    if pytorch_stream is None:
        pytorch_stream = torch.cuda.current_stream()

    # This is the cudaStream_t that we wish to use
    cuda_stream_s = faiss.cast_integer_to_cudastream_t(pytorch_stream.cuda_stream)

    # So we can revert GpuResources stream state upon exit
    prior_dev = torch.cuda.current_device()
    prior_stream = res.getDefaultStream(torch.cuda.current_device())

    res.setDefaultStream(torch.cuda.current_device(), cuda_stream_s)

    # Do the user work
    try:
        yield
    finally:
        res.setDefaultStream(prior_dev, prior_stream)

def search_index_pytorch(res, index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)

    with using_stream(res):
        xptr = swig_ptr_from_FloatTensor(x)
        Iptr = swig_ptr_from_LongTensor(I)
        Dptr = swig_ptr_from_FloatTensor(D)
        index.search_c(n, xptr, k, Dptr, Iptr)

    return D, I


def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    """search xq in xb, without building an index"""
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = k
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr

    with using_stream(res):
        faiss.bfKnn(res, args)

    return D, I
