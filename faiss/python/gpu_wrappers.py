# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

# not linting this file because it imports * from swigfaiss, which
# causes a ton of useless warnings.

import numpy as np

from faiss.loader import *


###########################################
# GPU functions
###########################################


def index_cpu_to_gpu_multiple_py(resources, index, co=None, gpus=None):
    """ builds the C++ vectors for the GPU indices and the
    resources. Handles the case where the resources are assigned to
    the list of GPUs """
    if gpus is None:
        gpus = range(len(resources))
    vres = GpuResourcesVector()
    vdev = Int32Vector()
    for i, res in zip(gpus, resources):
        vdev.push_back(i)
        vres.push_back(res)
    if isinstance(index, IndexBinary):
        return index_binary_cpu_to_gpu_multiple(vres, vdev, index, co)
    else:
        return index_cpu_to_gpu_multiple(vres, vdev, index, co)


def index_cpu_to_all_gpus(index, co=None, ngpu=-1):
    index_gpu = index_cpu_to_gpus_list(index, co=co, gpus=None, ngpu=ngpu)
    return index_gpu


def index_cpu_to_gpus_list(index, co=None, gpus=None, ngpu=-1):
    """ Here we can pass list of GPU ids as a parameter or ngpu to
    use first n GPU's. gpus mut be a list or None.
    co is a GpuMultipleClonerOptions
    """
    if (gpus is None) and (ngpu == -1):  # All blank
        gpus = range(get_num_gpus())
    elif (gpus is None) and (ngpu != -1):  # Get number of GPU's only
        gpus = range(ngpu)
    res = [StandardGpuResources() for _ in gpus]
    index_gpu = index_cpu_to_gpu_multiple_py(res, index, co, gpus)
    return index_gpu

# allows numpy ndarray usage with bfKnn


def knn_gpu(res, xq, xb, k, D=None, I=None, metric=METRIC_L2, device=-1, use_cuvs=False, vectorsMemoryLimit=0, queriesMemoryLimit=0):
    """
    Compute the k nearest neighbors of a vector on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    xq : array_like
        Query vectors, shape (nq, d) where d is appropriate for the index.
        `dtype` must be float32.
    xb : array_like
        Database vectors, shape (nb, d) where d is appropriate for the index.
        `dtype` must be float32.
    k : int
        Number of nearest neighbors.
    D : array_like, optional
        Output array for distances of the nearest neighbors, shape (nq, k)
    I : array_like, optional
        Output array for the nearest neighbors, shape (nq, k)
    metric : MetricType, optional
        Distance measure to use (either METRIC_L2 or METRIC_INNER_PRODUCT)
    device: int, optional
        Which CUDA device in the system to run the search on. -1 indicates that
        the current thread-local device state (via cudaGetDevice) should be used
        (can also be set via torch.cuda.set_device in PyTorch)
        Otherwise, an integer 0 <= device < numDevices indicates the GPU on which
        the computation should be run
    vectorsMemoryLimit: int, optional
    queriesMemoryLimit: int, optional
        Memory limits for vectors and queries.
        If not 0, the GPU will use at most this amount of memory
        for vectors and queries respectively.
        Vectors are broken up into chunks of size vectorsMemoryLimit,
        and queries are broken up into chunks of size queriesMemoryLimit,
        including the memory required for the results.

    Returns
    -------
    D : array_like
        Distances of the nearest neighbors, shape (nq, k)
    I : array_like
        Labels of the nearest neighbors, shape (nq, k)
    """
    nq, d = xq.shape
    if xq.flags.c_contiguous:
        xq_row_major = True
    elif xq.flags.f_contiguous:
        xq = xq.T
        xq_row_major = False
    else:
        xq = np.ascontiguousarray(xq, dtype='float32')
        xq_row_major = True

    xq_ptr = swig_ptr(xq)

    if xq.dtype == np.float32:
        xq_type = DistanceDataType_F32
    elif xq.dtype == np.float16:
        xq_type = DistanceDataType_F16
    else:
        raise TypeError('xq must be f32 or f16')

    nb, d2 = xb.shape
    assert d2 == d
    if xb.flags.c_contiguous:
        xb_row_major = True
    elif xb.flags.f_contiguous:
        xb = xb.T
        xb_row_major = False
    else:
        xb = np.ascontiguousarray(xb, dtype='float32')
        xb_row_major = True

    xb_ptr = swig_ptr(xb)

    if xb.dtype == np.float32:
        xb_type = DistanceDataType_F32
    elif xb.dtype == np.float16:
        xb_type = DistanceDataType_F16
    else:
        raise TypeError('xb must be float32 or float16')

    if D is None:
        D = np.empty((nq, k), dtype=np.float32)
    else:
        assert D.shape == (nq, k)
        # interface takes void*, we need to check this
        assert D.dtype == np.float32

    D_ptr = swig_ptr(D)

    if I is None:
        I = np.empty((nq, k), dtype=np.int64)
    else:
        assert I.shape == (nq, k)

    I_ptr = swig_ptr(I)

    if I.dtype == np.int64:
        I_type = IndicesDataType_I64
    elif I.dtype == I.dtype == np.int32:
        I_type = IndicesDataType_I32
    else:
        raise TypeError('I must be i64 or i32')

    args = GpuDistanceParams()
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
    args.use_cuvs = use_cuvs

    # no stream synchronization needed, inputs and outputs are guaranteed to
    # be on the CPU (numpy arrays)
    if vectorsMemoryLimit > 0 or queriesMemoryLimit > 0:
        bfKnn_tiling(res, args, vectorsMemoryLimit, queriesMemoryLimit)
    else:
        bfKnn(res, args)

    return D, I

# allows numpy ndarray usage with bfKnn for all pairwise distances


def pairwise_distance_gpu(res, xq, xb, D=None, metric=METRIC_L2, device=-1):
    """
    Compute all pairwise distances between xq and xb on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    xq : array_like
        Query vectors, shape (nq, d) where d is appropriate for the index.
        `dtype` must be float32.
    xb : array_like
        Database vectors, shape (nb, d) where d is appropriate for the index.
        `dtype` must be float32.
    D : array_like, optional
        Output array for all pairwise distances, shape (nq, nb)
    metric : MetricType, optional
        Distance measure to use (either METRIC_L2 or METRIC_INNER_PRODUCT)
    device: int, optional
        Which CUDA device in the system to run the search on. -1 indicates that
        the current thread-local device state (via cudaGetDevice) should be used
        (can also be set via torch.cuda.set_device in PyTorch)
        Otherwise, an integer 0 <= device < numDevices indicates the GPU on which
        the computation should be run

    Returns
    -------
    D : array_like
        All pairwise distances, shape (nq, nb)
    """
    nq, d = xq.shape
    if xq.flags.c_contiguous:
        xq_row_major = True
    elif xq.flags.f_contiguous:
        xq = xq.T
        xq_row_major = False
    else:
        raise TypeError(
            'xq matrix should be row (C) or column-major (Fortran)')

    xq_ptr = swig_ptr(xq)

    if xq.dtype == np.float32:
        xq_type = DistanceDataType_F32
    elif xq.dtype == np.float16:
        xq_type = DistanceDataType_F16
    else:
        xq = np.ascontiguousarray(xb, dtype='float32')
        xq_row_major = True

    nb, d2 = xb.shape
    assert d2 == d
    if xb.flags.c_contiguous:
        xb_row_major = True
    elif xb.flags.f_contiguous:
        xb = xb.T
        xb_row_major = False
    else:
        xb = np.ascontiguousarray(xb, dtype='float32')
        xb_row_major = True

    xb_ptr = swig_ptr(xb)

    if xb.dtype == np.float32:
        xb_type = DistanceDataType_F32
    elif xb.dtype == np.float16:
        xb_type = DistanceDataType_F16
    else:
        raise TypeError('xb must be float32 or float16')

    if D is None:
        D = np.empty((nq, nb), dtype=np.float32)
    else:
        assert D.shape == (nq, nb)
        # interface takes void*, we need to check this
        assert D.dtype == np.float32

    D_ptr = swig_ptr(D)

    args = GpuDistanceParams()
    args.metric = metric
    args.k = -1  # selects all pairwise distances
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

    # no stream synchronization needed, inputs and outputs are guaranteed to
    # be on the CPU (numpy arrays)
    bfKnn(res, args)

    return D
