# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""

This is a set of function wrappers that override the default numpy versions.

Interoperability functions for pytorch and Faiss: Importing this will allow
pytorch Tensors (CPU or GPU) to be used as arguments to Faiss indexes and
other functions. Torch GPU tensors can only be used with Faiss GPU indexes.
If this is imported with a package that supports Faiss GPU, the necessary
stream synchronization with the current pytorch stream will be automatically
performed.

Numpy ndarrays can continue to be used in the Faiss python interface after
importing this file. All arguments must be uniformly either numpy ndarrays
or Torch tensors; no mixing is allowed.

"""


import faiss
import torch
import contextlib
import inspect
import sys
import numpy as np

##################################################################
# Equivalent of swig_ptr for Torch tensors
##################################################################

def swig_ptr_from_UInt8Tensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.uint8
    return faiss.cast_integer_to_uint8_ptr(
        x.untyped_storage().data_ptr() + x.storage_offset())


def swig_ptr_from_HalfTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.float16
    # no canonical half type in C/C++
    return faiss.cast_integer_to_void_ptr(
        x.untyped_storage().data_ptr() + x.storage_offset() * 2)


def swig_ptr_from_FloatTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.untyped_storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_IntTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.int32, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_int_ptr(
        x.untyped_storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_IndicesTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_idx_t_ptr(
        x.untyped_storage().data_ptr() + x.storage_offset() * 8)

##################################################################
# utilities
##################################################################

@contextlib.contextmanager
def using_stream(res, pytorch_stream=None):
    """ Creates a scoping object to make Faiss GPU use the same stream
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

def torch_replace_method(the_class, name, replacement,
                         ignore_missing=False, ignore_no_base=False):
    try:
        orig_method = getattr(the_class, name)
    except AttributeError:
        if ignore_missing:
            return
        raise
    if orig_method.__name__ == 'torch_replacement_' + name:
        # replacement was done in parent class
        return

    # We should already have the numpy replacement methods patched
    assert ignore_no_base or (orig_method.__name__ == 'replacement_' + name)
    setattr(the_class, name + '_numpy', orig_method)
    setattr(the_class, name, replacement)

##################################################################
# Setup wrappers
##################################################################

def handle_torch_Index(the_class):
    def torch_replacement_add(self, x):
        if type(x) is np.ndarray:
            # forward to faiss __init__.py base method
            return self.add_numpy(x)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d
        x_ptr = swig_ptr_from_FloatTensor(x)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.add_c(n, x_ptr)
        else:
            # CPU torch
            self.add_c(n, x_ptr)

    def torch_replacement_add_with_ids(self, x, ids):
        if type(x) is np.ndarray:
            # forward to faiss __init__.py base method
            return self.add_with_ids_numpy(x, ids)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d
        x_ptr = swig_ptr_from_FloatTensor(x)

        assert type(ids) is torch.Tensor
        assert ids.shape == (n, ), 'not same number of vectors as ids'
        ids_ptr = swig_ptr_from_IndicesTensor(ids)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.add_with_ids_c(n, x_ptr, ids_ptr)
        else:
            # CPU torch
            self.add_with_ids_c(n, x_ptr, ids_ptr)

    def torch_replacement_assign(self, x, k, labels=None):
        if type(x) is np.ndarray:
            # forward to faiss __init__.py base method
            return self.assign_numpy(x, k, labels)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d
        x_ptr = swig_ptr_from_FloatTensor(x)

        if labels is None:
            labels = torch.empty(n, k, device=x.device, dtype=torch.int64)
        else:
            assert type(labels) is torch.Tensor
            assert labels.shape == (n, k)
        L_ptr = swig_ptr_from_IndicesTensor(labels)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.assign_c(n, x_ptr, L_ptr, k)
        else:
            # CPU torch
            self.assign_c(n, x_ptr, L_ptr, k)

        return labels

    def torch_replacement_train(self, x):
        if type(x) is np.ndarray:
            # forward to faiss __init__.py base method
            return self.train_numpy(x)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d
        x_ptr = swig_ptr_from_FloatTensor(x)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.train_c(n, x_ptr)
        else:
            # CPU torch
            self.train_c(n, x_ptr)

    def search_methods_common(x, k, D, I):
        n, d = x.shape
        x_ptr = swig_ptr_from_FloatTensor(x)

        if D is None:
            D = torch.empty(n, k, device=x.device, dtype=torch.float32)
        else:
            assert type(D) is torch.Tensor
            assert D.shape == (n, k)
        D_ptr = swig_ptr_from_FloatTensor(D)

        if I is None:
            I = torch.empty(n, k, device=x.device, dtype=torch.int64)
        else:
            assert type(I) is torch.Tensor
            assert I.shape == (n, k)
        I_ptr = swig_ptr_from_IndicesTensor(I)

        return x_ptr, D_ptr, I_ptr, D, I

    def torch_replacement_search(self, x, k, D=None, I=None):
        if type(x) is np.ndarray:
            # forward to faiss __init__.py base method
            return self.search_numpy(x, k, D=D, I=I)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d

        x_ptr, D_ptr, I_ptr, D, I = search_methods_common(x, k, D, I)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.search_c(n, x_ptr, k, D_ptr, I_ptr)
        else:
            # CPU torch
            self.search_c(n, x_ptr, k, D_ptr, I_ptr)

        return D, I

    def torch_replacement_search_and_reconstruct(self, x, k, D=None, I=None, R=None):
        if type(x) is np.ndarray:
            # Forward to faiss __init__.py base method
            return self.search_and_reconstruct_numpy(x, k, D=D, I=I, R=R)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d

        x_ptr, D_ptr, I_ptr, D, I = search_methods_common(x, k, D, I)

        if R is None:
            R = torch.empty(n, k, d, device=x.device, dtype=torch.float32)
        else:
            assert type(R) is torch.Tensor
            assert R.shape == (n, k, d)
        R_ptr = swig_ptr_from_FloatTensor(R)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.search_and_reconstruct_c(n, x_ptr, k, D_ptr, I_ptr, R_ptr)
        else:
            # CPU torch
            self.search_and_reconstruct_c(n, x_ptr, k, D_ptr, I_ptr, R_ptr)

        return D, I, R

    def torch_replacement_search_preassigned(self, x, k, Iq, Dq, *, D=None, I=None):
        if type(x) is np.ndarray:
            # forward to faiss __init__.py base method
            return self.search_preassigned_numpy(x, k, Iq, Dq, D=D, I=I)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d

        x_ptr, D_ptr, I_ptr, D, I = search_methods_common(x, k, D, I)

        assert Iq.shape == (n, self.nprobe)
        Iq = Iq.contiguous()
        Iq_ptr = swig_ptr_from_IndicesTensor(Iq)

        if Dq is not None:
            Dq = Dq.contiguous()
            assert Dq.shape == Iq.shape
            Dq_ptr = swig_ptr_from_FloatTensor(Dq)
        else:
            Dq_ptr = None

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.search_preassigned_c(n, x_ptr, k, Iq_ptr, Dq_ptr, D_ptr, I_ptr, False)
        else:
            # CPU torch
            self.search_preassigned_c(n, x_ptr, k, Iq_ptr, Dq_ptr, D_ptr, I_ptr, False)

        return D, I

    def torch_replacement_remove_ids(self, x):
        # Not yet implemented
        assert type(x) is not torch.Tensor, 'remove_ids not yet implemented for torch'
        return self.remove_ids_numpy(x)

    def torch_replacement_reconstruct(self, key, x=None):
        # No tensor inputs are required, but with importing this module, we
        # assume that the default should be torch tensors. If we are passed a
        # numpy array, however, assume that the user is overriding this default
        if (x is not None) and (type(x) is np.ndarray):
            # Forward to faiss __init__.py base method
            return self.reconstruct_numpy(key, x)

        # If the index is a CPU index, the default device is CPU, otherwise we
        # produce a GPU tensor
        device = torch.device('cpu')
        if hasattr(self, 'getDevice'):
            # same device as the index
            device = torch.device('cuda', self.getDevice())

        if x is None:
            x = torch.empty(self.d, device=device, dtype=torch.float32)
        else:
            assert type(x) is torch.Tensor
            assert x.shape == (self.d, )
        x_ptr = swig_ptr_from_FloatTensor(x)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.reconstruct_c(key, x_ptr)
        else:
            # CPU torch
            self.reconstruct_c(key, x_ptr)

        return x

    def torch_replacement_reconstruct_n(self, n0=0, ni=-1, x=None):
        if ni == -1:
            ni = self.ntotal

        # No tensor inputs are required, but with importing this module, we
        # assume that the default should be torch tensors. If we are passed a
        # numpy array, however, assume that the user is overriding this default
        if (x is not None) and (type(x) is np.ndarray):
            # Forward to faiss __init__.py base method
            return self.reconstruct_n_numpy(n0, ni, x)

        # If the index is a CPU index, the default device is CPU, otherwise we
        # produce a GPU tensor
        device = torch.device('cpu')
        if hasattr(self, 'getDevice'):
            # same device as the index
            device = torch.device('cuda', self.getDevice())

        if x is None:
            x = torch.empty(ni, self.d, device=device, dtype=torch.float32)
        else:
            assert type(x) is torch.Tensor
            assert x.shape == (ni, self.d)
        x_ptr = swig_ptr_from_FloatTensor(x)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.reconstruct_n_c(n0, ni, x_ptr)
        else:
            # CPU torch
            self.reconstruct_n_c(n0, ni, x_ptr)

        return x

    def torch_replacement_update_vectors(self, keys, x):
        if type(keys) is np.ndarray:
            # Forward to faiss __init__.py base method
            return self.update_vectors_numpy(keys, x)

        assert type(keys) is torch.Tensor
        (n, ) = keys.shape
        keys_ptr = swig_ptr_from_IndicesTensor(keys)

        assert type(x) is torch.Tensor
        assert x.shape == (n, self.d)
        x_ptr = swig_ptr_from_FloatTensor(x)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.update_vectors_c(n, keys_ptr, x_ptr)
        else:
            # CPU torch
            self.update_vectors_c(n, keys_ptr, x_ptr)

    # Until the GPU version is implemented, we do not support pre-allocated
    # output buffers
    def torch_replacement_range_search(self, x, thresh):
        if type(x) is np.ndarray:
            # Forward to faiss __init__.py base method
            return self.range_search_numpy(x, thresh)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d
        x_ptr = swig_ptr_from_FloatTensor(x)

        assert not x.is_cuda, 'Range search using GPU tensor not yet implemented'
        assert not hasattr(self, 'getDevice'), 'Range search on GPU index not yet implemented'

        res = faiss.RangeSearchResult(n)
        self.range_search_c(n, x_ptr, thresh, res)

        # get pointers and copy them
        # FIXME: no rev_swig_ptr equivalent for torch.Tensor, just convert
        # np to torch
        # NOTE: torch does not support np.uint64, just np.int64
        lims = torch.from_numpy(faiss.rev_swig_ptr(res.lims, n + 1).copy().astype('int64'))
        nd = int(lims[-1])
        D = torch.from_numpy(faiss.rev_swig_ptr(res.distances, nd).copy())
        I = torch.from_numpy(faiss.rev_swig_ptr(res.labels, nd).copy())

        return lims, D, I

    def torch_replacement_sa_encode(self, x, codes=None):
        if type(x) is np.ndarray:
            # Forward to faiss __init__.py base method
            return self.sa_encode_numpy(x, codes)

        assert type(x) is torch.Tensor
        n, d = x.shape
        assert d == self.d
        x_ptr = swig_ptr_from_FloatTensor(x)

        if codes is None:
            codes = torch.empty(n, self.sa_code_size(), dtype=torch.uint8)
        else:
            assert codes.shape == (n, self.sa_code_size())
        codes_ptr = swig_ptr_from_UInt8Tensor(codes)

        if x.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.sa_encode_c(n, x_ptr, codes_ptr)
        else:
            # CPU torch
            self.sa_encode_c(n, x_ptr, codes_ptr)

        return codes

    def torch_replacement_sa_decode(self, codes, x=None):
        if type(codes) is np.ndarray:
            # Forward to faiss __init__.py base method
            return self.sa_decode_numpy(codes, x)

        assert type(codes) is torch.Tensor
        n, cs = codes.shape
        assert cs == self.sa_code_size()
        codes_ptr = swig_ptr_from_UInt8Tensor(codes)

        if x is None:
            x = torch.empty(n, self.d, dtype=torch.float32)
        else:
            assert type(x) is torch.Tensor
            assert x.shape == (n, self.d)
        x_ptr = swig_ptr_from_FloatTensor(x)

        if codes.is_cuda:
            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'

            # On the GPU, use proper stream ordering
            with using_stream(self.getResources()):
                self.sa_decode_c(n, codes_ptr, x_ptr)
        else:
            # CPU torch
            self.sa_decode_c(n, codes_ptr, x_ptr)

        return x


    torch_replace_method(the_class, 'add', torch_replacement_add)
    torch_replace_method(the_class, 'add_with_ids', torch_replacement_add_with_ids)
    torch_replace_method(the_class, 'assign', torch_replacement_assign)
    torch_replace_method(the_class, 'train', torch_replacement_train)
    torch_replace_method(the_class, 'search', torch_replacement_search)
    torch_replace_method(the_class, 'remove_ids', torch_replacement_remove_ids)
    torch_replace_method(the_class, 'reconstruct', torch_replacement_reconstruct)
    torch_replace_method(the_class, 'reconstruct_n', torch_replacement_reconstruct_n)
    torch_replace_method(the_class, 'range_search', torch_replacement_range_search)
    torch_replace_method(the_class, 'update_vectors', torch_replacement_update_vectors,
                         ignore_missing=True)
    torch_replace_method(the_class, 'search_and_reconstruct',
                         torch_replacement_search_and_reconstruct, ignore_missing=True)
    torch_replace_method(the_class, 'search_preassigned',
                        torch_replacement_search_preassigned, ignore_missing=True)
    torch_replace_method(the_class, 'sa_encode', torch_replacement_sa_encode)
    torch_replace_method(the_class, 'sa_decode', torch_replacement_sa_decode)

faiss_module = sys.modules['faiss']

# Re-patch anything that inherits from faiss.Index to add the torch bindings
for symbol in dir(faiss_module):
    obj = getattr(faiss_module, symbol)
    if inspect.isclass(obj):
        the_class = obj
        if issubclass(the_class, faiss.Index):
            handle_torch_Index(the_class)


# allows torch tensor usage with knn
def torch_replacement_knn(xq, xb, k, metric=faiss.METRIC_L2, metric_arg=0):
    if type(xb) is np.ndarray:
        # Forward to faiss __init__.py base method
        return faiss.knn_numpy(xq, xb, k, metric=metric, metric_arg=metric_arg)

    nb, d = xb.size()
    assert xb.is_contiguous()
    assert xb.dtype == torch.float32
    assert not xb.is_cuda, "use knn_gpu for GPU tensors"

    nq, d2 = xq.size()
    assert d2 == d
    assert xq.is_contiguous()
    assert xq.dtype == torch.float32
    assert not xq.is_cuda, "use knn_gpu for GPU tensors"

    D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    I_ptr = swig_ptr_from_IndicesTensor(I)
    D_ptr = swig_ptr_from_FloatTensor(D)
    xb_ptr = swig_ptr_from_FloatTensor(xb)
    xq_ptr = swig_ptr_from_FloatTensor(xq)

    if metric == faiss.METRIC_L2:
        faiss.knn_L2sqr(
            xq_ptr, xb_ptr,
            d, nq, nb, k, D_ptr, I_ptr
        )
    elif metric == faiss.METRIC_INNER_PRODUCT:
        faiss.knn_inner_product(
            xq_ptr, xb_ptr,
            d, nq, nb, k, D_ptr, I_ptr
        )
    else:
        faiss.knn_extra_metrics(
            xq_ptr, xb_ptr,
            d, nq, nb, metric, metric_arg, k, D_ptr, I_ptr
        )

    return D, I


torch_replace_method(faiss_module, 'knn', torch_replacement_knn, True, True)


# allows torch tensor usage with bfKnn
def torch_replacement_knn_gpu(res, xq, xb, k, D=None, I=None, metric=faiss.METRIC_L2, device=-1, use_raft=False):
    if type(xb) is np.ndarray:
        # Forward to faiss __init__.py base method
        return faiss.knn_gpu_numpy(res, xq, xb, k, D, I, metric, device)

    nb, d = xb.size()
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    if xb.dtype == torch.float32:
        xb_type = faiss.DistanceDataType_F32
        xb_ptr = swig_ptr_from_FloatTensor(xb)
    elif xb.dtype == torch.float16:
        xb_type = faiss.DistanceDataType_F16
        xb_ptr = swig_ptr_from_HalfTensor(xb)
    else:
        raise TypeError('xb must be f32 or f16')

    nq, d2 = xq.size()
    assert d2 == d
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    if xq.dtype == torch.float32:
        xq_type = faiss.DistanceDataType_F32
        xq_ptr = swig_ptr_from_FloatTensor(xq)
    elif xq.dtype == torch.float16:
        xq_type = faiss.DistanceDataType_F16
        xq_ptr = swig_ptr_from_HalfTensor(xq)
    else:
        raise TypeError('xq must be f32 or f16')

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        # interface takes void*, we need to check this
        assert (D.dtype == torch.float32)

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)

    if I.dtype == torch.int64:
        I_type = faiss.IndicesDataType_I64
        I_ptr = swig_ptr_from_IndicesTensor(I)
    elif I.dtype == I.dtype == torch.int32:
        I_type = faiss.IndicesDataType_I32
        I_ptr = swig_ptr_from_IntTensor(I)
    else:
        raise TypeError('I must be i64 or i32')

    D_ptr = swig_ptr_from_FloatTensor(D)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = k
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.vectorType = xb_type
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type
    args.device = device
    args.use_raft = use_raft

    with using_stream(res):
        faiss.bfKnn(res, args)

    return D, I

torch_replace_method(faiss_module, 'knn_gpu', torch_replacement_knn_gpu, True, True)

# allows torch tensor usage with bfKnn for all pairwise distances
def torch_replacement_pairwise_distance_gpu(res, xq, xb, D=None, metric=faiss.METRIC_L2, device=-1):
    if type(xb) is np.ndarray:
        # Forward to faiss __init__.py base method
        return faiss.pairwise_distance_gpu_numpy(res, xq, xb, D, metric)

    nb, d = xb.size()
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('xb matrix should be row or column-major')

    if xb.dtype == torch.float32:
        xb_type = faiss.DistanceDataType_F32
        xb_ptr = swig_ptr_from_FloatTensor(xb)
    elif xb.dtype == torch.float16:
        xb_type = faiss.DistanceDataType_F16
        xb_ptr = swig_ptr_from_HalfTensor(xb)
    else:
        raise TypeError('xb must be float32 or float16')

    nq, d2 = xq.size()
    assert d2 == d
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()
        xq_row_major = False
    else:
        raise TypeError('xq matrix should be row or column-major')

    if xq.dtype == torch.float32:
        xq_type = faiss.DistanceDataType_F32
        xq_ptr = swig_ptr_from_FloatTensor(xq)
    elif xq.dtype == torch.float16:
        xq_type = faiss.DistanceDataType_F16
        xq_ptr = swig_ptr_from_HalfTensor(xq)
    else:
        raise TypeError('xq must be float32 or float16')

    if D is None:
        D = torch.empty(nq, nb, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, nb)
        # interface takes void*, we need to check this
        assert (D.dtype == torch.float32)

    D_ptr = swig_ptr_from_FloatTensor(D)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = -1 # selects all pairwise distance
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.vectorType = xb_type
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.device = device

    with using_stream(res):
        faiss.bfKnn(res, args)

    return D

torch_replace_method(faiss_module, 'pairwise_distance_gpu', torch_replacement_pairwise_distance_gpu, True, True)
