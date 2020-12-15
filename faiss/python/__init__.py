# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#@nolint

# not linting this file because it imports * form swigfaiss, which
# causes a ton of useless warnings.

import numpy as np
import sys
import inspect
import array

# We import * so that the symbol foo can be accessed as faiss.foo.
from .loader import *


__version__ = "%d.%d.%d" % (FAISS_VERSION_MAJOR,
                            FAISS_VERSION_MINOR,
                            FAISS_VERSION_PATCH)

##################################################################
# The functions below add or replace some methods for classes
# this is to be able to pass in numpy arrays directly
# The C++ version of the classnames will be suffixed with _c
##################################################################


def replace_method(the_class, name, replacement, ignore_missing=False):
    try:
        orig_method = getattr(the_class, name)
    except AttributeError:
        if ignore_missing:
            return
        raise
    if orig_method.__name__ == 'replacement_' + name:
        # replacement was done in parent class
        return
    setattr(the_class, name + '_c', orig_method)
    setattr(the_class, name, replacement)

def handle_Clustering():
    def replacement_train(self, x, index, weights=None):
        n, d = x.shape
        assert d == self.d
        if weights is not None:
            assert weights.shape == (n, )
            self.train_c(n, swig_ptr(x), index, swig_ptr(weights))
        else:
            self.train_c(n, swig_ptr(x), index)
    def replacement_train_encoded(self, x, codec, index, weights=None):
        n, d = x.shape
        assert d == codec.sa_code_size()
        assert codec.d == index.d
        if weights is not None:
            assert weights.shape == (n, )
            self.train_encoded_c(n, swig_ptr(x), codec, index, swig_ptr(weights))
        else:
            self.train_encoded_c(n, swig_ptr(x), codec, index)
    replace_method(Clustering, 'train', replacement_train)
    replace_method(Clustering, 'train_encoded', replacement_train_encoded)


handle_Clustering()


def handle_Quantizer(the_class):

    def replacement_train(self, x):
        n, d = x.shape
        assert d == self.d
        self.train_c(n, swig_ptr(x))

    def replacement_compute_codes(self, x):
        n, d = x.shape
        assert d == self.d
        codes = np.empty((n, self.code_size), dtype='uint8')
        self.compute_codes_c(swig_ptr(x), swig_ptr(codes), n)
        return codes

    def replacement_decode(self, codes):
        n, cs = codes.shape
        assert cs == self.code_size
        x = np.empty((n, self.d), dtype='float32')
        self.decode_c(swig_ptr(codes), swig_ptr(x), n)
        return x

    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'compute_codes', replacement_compute_codes)
    replace_method(the_class, 'decode', replacement_decode)


handle_Quantizer(ProductQuantizer)
handle_Quantizer(ScalarQuantizer)


def handle_Index(the_class):

    def replacement_add(self, x):
        n, d = x.shape
        assert d == self.d
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        n, d = x.shape
        assert d == self.d

        assert ids.shape == (n, ), 'not same nb of vectors as ids'
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_assign(self, x, k, labels=None):
        n, d = x.shape
        assert d == self.d

        if labels is None:
            labels = np.empty((n, k), dtype=np.int64)
        else:
            assert labels.shape == (n, k)

        self.assign_c(n, swig_ptr(x), swig_ptr(labels), k)
        return labels

    def replacement_train(self, x):
        n, d = x.shape
        assert d == self.d
        self.train_c(n, swig_ptr(x))

    def replacement_search(self, x, k, D=None, I=None):
        n, d = x.shape
        assert d == self.d

        if D is None:
            D = np.empty((n, k), dtype=np.float32)
        else:
            assert D.shape == (n, k)

        if I is None:
            I = np.empty((n, k), dtype=np.int64)
        else:
            assert I.shape == (n, k)

        self.search_c(n, swig_ptr(x), k, swig_ptr(D), swig_ptr(I))
        return D, I

    def replacement_search_and_reconstruct(self, x, k, D=None, I=None, R=None):
        n, d = x.shape
        assert d == self.d

        if D is None:
            D = np.empty((n, k), dtype=np.float32)
        else:
            assert D.shape == (n, k)

        if I is None:
            I = np.empty((n, k), dtype=np.int64)
        else:
            assert I.shape == (n, k)

        if R is None:
            R = np.empty((n, k, d), dtype=np.float32)
        else:
            assert R.shape == (n, k, d)

        self.search_and_reconstruct_c(n, swig_ptr(x),
                                      k, swig_ptr(D),
                                      swig_ptr(I),
                                      swig_ptr(R))
        return D, I, R

    def replacement_remove_ids(self, x):
        if isinstance(x, IDSelector):
            sel = x
        else:
            assert x.ndim == 1
            index_ivf = try_extract_index_ivf (self)
            if index_ivf and index_ivf.direct_map.type == DirectMap.Hashtable:
                sel = IDSelectorArray(x.size, swig_ptr(x))
            else:
                sel = IDSelectorBatch(x.size, swig_ptr(x))
        return self.remove_ids_c(sel)

    def replacement_reconstruct(self, key, x=None):
        if x is None:
            x = np.empty(self.d, dtype=np.float32)
        else:
            assert x.shape == (self.d, )

        self.reconstruct_c(key, swig_ptr(x))
        return x

    def replacement_reconstruct_n(self, n0, ni, x=None):
        if x is None:
            x = np.empty((ni, self.d), dtype=np.float32)
        else:
            assert x.shape == (ni, self.d)

        self.reconstruct_n_c(n0, ni, swig_ptr(x))
        return x

    def replacement_update_vectors(self, keys, x):
        n = keys.size
        assert keys.shape == (n, )
        assert x.shape == (n, self.d)

        self.update_vectors_c(n, swig_ptr(keys), swig_ptr(x))

    # The CPU does not support passed-in output buffers
    def replacement_range_search(self, x, thresh):
        n, d = x.shape
        assert d == self.d

        res = RangeSearchResult(n)
        self.range_search_c(n, swig_ptr(x), thresh, res)
        # get pointers and copy them
        lims = rev_swig_ptr(res.lims, n + 1).copy()
        nd = int(lims[-1])
        D = rev_swig_ptr(res.distances, nd).copy()
        I = rev_swig_ptr(res.labels, nd).copy()
        return lims, D, I

    def replacement_sa_encode(self, x, codes=None):
        n, d = x.shape
        assert d == self.d

        if codes is None:
            codes = np.empty((n, self.sa_code_size()), dtype=np.uint8)
        else:
            assert codes.shape == (n, self.sa_code_size())

        self.sa_encode_c(n, swig_ptr(x), swig_ptr(codes))
        return codes

    def replacement_sa_decode(self, codes, x=None):
        n, cs = codes.shape
        assert cs == self.sa_code_size()

        if x is None:
            x = np.empty((n, self.d), dtype=np.float32)
        else:
            assert x.shape == (n, self.d)

        self.sa_decode_c(n, swig_ptr(codes), swig_ptr(x))
        return x

    replace_method(the_class, 'add', replacement_add)
    replace_method(the_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(the_class, 'assign', replacement_assign)
    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'search', replacement_search)
    replace_method(the_class, 'remove_ids', replacement_remove_ids)
    replace_method(the_class, 'reconstruct', replacement_reconstruct)
    replace_method(the_class, 'reconstruct_n', replacement_reconstruct_n)
    replace_method(the_class, 'range_search', replacement_range_search)
    replace_method(the_class, 'update_vectors', replacement_update_vectors,
                   ignore_missing=True)
    replace_method(the_class, 'search_and_reconstruct',
                   replacement_search_and_reconstruct, ignore_missing=True)
    replace_method(the_class, 'sa_encode', replacement_sa_encode)
    replace_method(the_class, 'sa_decode', replacement_sa_decode)

def handle_IndexBinary(the_class):

    def replacement_add(self, x):
        n, d = x.shape
        assert d * 8 == self.d
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        n, d = x.shape
        assert d * 8 == self.d
        assert ids.shape == (n, ), 'not same nb of vectors as ids'
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_train(self, x):
        n, d = x.shape
        assert d * 8 == self.d
        self.train_c(n, swig_ptr(x))

    def replacement_reconstruct(self, key):
        x = np.empty(self.d // 8, dtype=np.uint8)
        self.reconstruct_c(key, swig_ptr(x))
        return x

    def replacement_search(self, x, k):
        n, d = x.shape
        assert d * 8 == self.d
        distances = np.empty((n, k), dtype=np.int32)
        labels = np.empty((n, k), dtype=np.int64)
        self.search_c(n, swig_ptr(x),
                      k, swig_ptr(distances),
                      swig_ptr(labels))
        return distances, labels

    def replacement_range_search(self, x, thresh):
        n, d = x.shape
        assert d * 8 == self.d
        res = RangeSearchResult(n)
        self.range_search_c(n, swig_ptr(x), thresh, res)
        # get pointers and copy them
        lims = rev_swig_ptr(res.lims, n + 1).copy()
        nd = int(lims[-1])
        D = rev_swig_ptr(res.distances, nd).copy()
        I = rev_swig_ptr(res.labels, nd).copy()
        return lims, D, I

    def replacement_remove_ids(self, x):
        if isinstance(x, IDSelector):
            sel = x
        else:
            assert x.ndim == 1
            sel = IDSelectorBatch(x.size, swig_ptr(x))
        return self.remove_ids_c(sel)

    replace_method(the_class, 'add', replacement_add)
    replace_method(the_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'search', replacement_search)
    replace_method(the_class, 'range_search', replacement_range_search)
    replace_method(the_class, 'reconstruct', replacement_reconstruct)
    replace_method(the_class, 'remove_ids', replacement_remove_ids)


def handle_VectorTransform(the_class):

    def apply_method(self, x):
        n, d = x.shape
        assert d == self.d_in
        y = np.empty((n, self.d_out), dtype=np.float32)
        self.apply_noalloc(n, swig_ptr(x), swig_ptr(y))
        return y

    def replacement_reverse_transform(self, x):
        n, d = x.shape
        assert d == self.d_out
        y = np.empty((n, self.d_in), dtype=np.float32)
        self.reverse_transform_c(n, swig_ptr(x), swig_ptr(y))
        return y

    def replacement_vt_train(self, x):
        n, d = x.shape
        assert d == self.d_in
        self.train_c(n, swig_ptr(x))

    replace_method(the_class, 'train', replacement_vt_train)
    # apply is reserved in Pyton...
    the_class.apply_py = apply_method
    replace_method(the_class, 'reverse_transform',
                   replacement_reverse_transform)


def handle_AutoTuneCriterion(the_class):
    def replacement_set_groundtruth(self, D, I):
        if D:
            assert I.shape == D.shape
        self.nq, self.gt_nnn = I.shape
        self.set_groundtruth_c(
            self.gt_nnn, swig_ptr(D) if D else None, swig_ptr(I))

    def replacement_evaluate(self, D, I):
        assert I.shape == D.shape
        assert I.shape == (self.nq, self.nnn)
        return self.evaluate_c(swig_ptr(D), swig_ptr(I))

    replace_method(the_class, 'set_groundtruth', replacement_set_groundtruth)
    replace_method(the_class, 'evaluate', replacement_evaluate)


def handle_ParameterSpace(the_class):
    def replacement_explore(self, index, xq, crit):
        assert xq.shape == (crit.nq, index.d)
        ops = OperatingPoints()
        self.explore_c(index, crit.nq, swig_ptr(xq),
                       crit, ops)
        return ops
    replace_method(the_class, 'explore', replacement_explore)


def handle_MatrixStats(the_class):
    original_init = the_class.__init__

    def replacement_init(self, m):
        assert len(m.shape) == 2
        original_init(self, m.shape[0], m.shape[1], swig_ptr(m))

    the_class.__init__ = replacement_init

handle_MatrixStats(MatrixStats)

def handle_IOWriter(the_class):

    def write_bytes(self, b):
        return self(swig_ptr(b), 1, len(b))

    the_class.write_bytes = write_bytes

handle_IOWriter(IOWriter)

def handle_IOReader(the_class):

    def read_bytes(self, totsz):
        buf = bytearray(totsz)
        was_read = self(swig_ptr(buf), 1, len(buf))
        return bytes(buf[:was_read])

    the_class.read_bytes = read_bytes

handle_IOReader(IOReader)

this_module = sys.modules[__name__]


for symbol in dir(this_module):
    obj = getattr(this_module, symbol)
    # print symbol, isinstance(obj, (type, types.ClassType))
    if inspect.isclass(obj):
        the_class = obj
        if issubclass(the_class, Index):
            handle_Index(the_class)

        if issubclass(the_class, IndexBinary):
            handle_IndexBinary(the_class)

        if issubclass(the_class, VectorTransform):
            handle_VectorTransform(the_class)

        if issubclass(the_class, AutoTuneCriterion):
            handle_AutoTuneCriterion(the_class)

        if issubclass(the_class, ParameterSpace):
            handle_ParameterSpace(the_class)


###########################################
# Add Python references to objects
# we do this at the Python class wrapper level.
###########################################

def add_ref_in_constructor(the_class, parameter_no):
    # adds a reference to parameter parameter_no in self
    # so that that parameter does not get deallocated before self
    original_init = the_class.__init__

    def replacement_init(self, *args):
        original_init(self, *args)
        self.referenced_objects = [args[parameter_no]]

    def replacement_init_multiple(self, *args):
        original_init(self, *args)
        pset = parameter_no[len(args)]
        self.referenced_objects = [args[no] for no in pset]

    if type(parameter_no) == dict:
        # a list of parameters to keep, depending on the number of arguments
        the_class.__init__ = replacement_init_multiple
    else:
        the_class.__init__ = replacement_init


def add_ref_in_method(the_class, method_name, parameter_no):
    original_method = getattr(the_class, method_name)
    def replacement_method(self, *args):
        ref = args[parameter_no]
        if not hasattr(self, 'referenced_objects'):
            self.referenced_objects = [ref]
        else:
            self.referenced_objects.append(ref)
        return original_method(self, *args)
    setattr(the_class, method_name, replacement_method)

def add_ref_in_function(function_name, parameter_no):
    # assumes the function returns an object
    original_function = getattr(this_module, function_name)
    def replacement_function(*args):
        result = original_function(*args)
        ref = args[parameter_no]
        result.referenced_objects = [ref]
        return result
    setattr(this_module, function_name, replacement_function)

add_ref_in_constructor(IndexIVFFlat, 0)
add_ref_in_constructor(IndexIVFFlatDedup, 0)
add_ref_in_constructor(IndexPreTransform, {2: [0, 1], 1: [0]})
add_ref_in_method(IndexPreTransform, 'prepend_transform', 0)
add_ref_in_constructor(IndexIVFPQ, 0)
add_ref_in_constructor(IndexIVFPQR, 0)
add_ref_in_constructor(IndexIVFPQFastScan, 0)
add_ref_in_constructor(Index2Layer, 0)
add_ref_in_constructor(Level1Quantizer, 0)
add_ref_in_constructor(IndexIVFScalarQuantizer, 0)
add_ref_in_constructor(IndexIDMap, 0)
add_ref_in_constructor(IndexIDMap2, 0)
add_ref_in_constructor(IndexHNSW, 0)
add_ref_in_method(IndexShards, 'add_shard', 0)
add_ref_in_method(IndexBinaryShards, 'add_shard', 0)
add_ref_in_constructor(IndexRefineFlat, {2:[0], 1:[0]})
add_ref_in_constructor(IndexRefine, {2:[0, 1]})

add_ref_in_constructor(IndexBinaryIVF, 0)
add_ref_in_constructor(IndexBinaryFromFloat, 0)
add_ref_in_constructor(IndexBinaryIDMap, 0)
add_ref_in_constructor(IndexBinaryIDMap2, 0)

add_ref_in_method(IndexReplicas, 'addIndex', 0)
add_ref_in_method(IndexBinaryReplicas, 'addIndex', 0)

add_ref_in_constructor(BufferedIOWriter, 0)
add_ref_in_constructor(BufferedIOReader, 0)

# seems really marginal...
# remove_ref_from_method(IndexReplicas, 'removeIndex', 0)

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
    vdev = IntVector()
    for i, res in zip(gpus, resources):
        vdev.push_back(i)
        vres.push_back(res)
    index = index_cpu_to_gpu_multiple(vres, vdev, index, co)
    return index


def index_cpu_to_all_gpus(index, co=None, ngpu=-1):
    index_gpu = index_cpu_to_gpus_list(index, co=co, gpus=None, ngpu=ngpu)
    return index_gpu


def index_cpu_to_gpus_list(index, co=None, gpus=None, ngpu=-1):
    """ Here we can pass list of GPU ids as a parameter or ngpu to
    use first n GPU's. gpus mut be a list or None"""
    if (gpus is None) and (ngpu == -1):  # All blank
        gpus = range(get_num_gpus())
    elif (gpus is None) and (ngpu != -1):  # Get number of GPU's only
        gpus = range(ngpu)
    res = [StandardGpuResources() for _ in gpus]
    index_gpu = index_cpu_to_gpu_multiple_py(res, index, co, gpus)
    return index_gpu

# allows numpy ndarray usage with bfKnn
def knn_gpu(res, xb, xq, k, D=None, I=None, metric=METRIC_L2):
    nq, d = xq.shape
    if xq.flags.c_contiguous:
        xq_row_major = True
    elif xq.flags.f_contiguous:
        xq = xq.T
        xq_row_major = False
    else:
        raise TypeError('matrix should be row (C) or column-major (Fortran)')

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
        raise TypeError('matrix should be row (C) or column-major (Fortran)')

    xb_ptr = swig_ptr(xb)

    if xb.dtype == np.float32:
        xb_type = DistanceDataType_F32
    elif xb.dtype == np.float16:
        xb_type = DistanceDataType_F16
    else:
        raise TypeError('xb must be f32 or f16')

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

    # no stream synchronization needed, inputs and outputs are guaranteed to
    # be on the CPU (numpy arrays)
    bfKnn(res, args)

    return D, I

###########################################
# numpy array / std::vector conversions
###########################################

# mapping from vector names in swigfaiss.swig and the numpy dtype names
vector_name_map = {
    'Float': 'float32',
    'Byte': 'uint8',
    'Char': 'int8',
    'Uint64': 'uint64',
    'LongLong': 'int64',
    'Int': 'int32',
    'Double': 'float64'
}

sizeof_long = array.array('l').itemsize
if sizeof_long == 4:
    vector_name_map["Long"] = 'int32'
elif sizeof_long == 8:
    vector_name_map["Long"] = 'int64'



def vector_to_array(v):
    """ convert a C++ vector to a numpy array """
    classname = v.__class__.__name__
    assert classname.endswith('Vector')
    dtype = np.dtype(vector_name_map[classname[:-6]])
    a = np.empty(v.size(), dtype=dtype)
    if v.size() > 0:
        memcpy(swig_ptr(a), v.data(), a.nbytes)
    return a


def vector_float_to_array(v):
    return vector_to_array(v)


def copy_array_to_vector(a, v):
    """ copy a numpy array to a vector """
    n, = a.shape
    classname = v.__class__.__name__
    assert classname.endswith('Vector')
    dtype = np.dtype(vector_name_map[classname[:-6]])
    assert dtype == a.dtype, (
        'cannot copy a %s array to a %s (should be %s)' % (
            a.dtype, classname, dtype))
    v.resize(n)
    if n > 0:
        memcpy(v.data(), swig_ptr(a), a.nbytes)

# same for AlignedTable

def copy_array_to_AlignedTable(a, v):
    n, = a.shape
    # TODO check class name
    assert v.itemsize() == a.itemsize
    v.resize(n)
    if n > 0:
        memcpy(v.get(), swig_ptr(a), a.nbytes)

def array_to_AlignedTable(a):
    if a.dtype == 'uint16':
        v = AlignedTableUint16(a.size)
    elif a.dtype == 'uint8':
        v = AlignedTableUint8(a.size)
    else:
        assert False
    copy_array_to_AlignedTable(a, v)
    return v

def AlignedTable_to_array(v):
    """ convert an AlignedTable to a numpy array """
    classname = v.__class__.__name__
    assert classname.startswith('AlignedTable')
    dtype = classname[12:].lower()
    a = np.empty(v.size(), dtype=dtype)
    if a.size > 0:
        memcpy(swig_ptr(a), v.data(), a.nbytes)
    return a

###########################################
# Wrapper for a few functions
###########################################

def kmin(array, k):
    """return k smallest values (and their indices) of the lines of a
    float32 array"""
    m, n = array.shape
    I = np.zeros((m, k), dtype='int64')
    D = np.zeros((m, k), dtype='float32')
    ha = float_maxheap_array_t()
    ha.ids = swig_ptr(I)
    ha.val = swig_ptr(D)
    ha.nh = m
    ha.k = k
    ha.heapify()
    ha.addn(n, swig_ptr(array))
    ha.reorder()
    return D, I


def kmax(array, k):
    """return k largest values (and their indices) of the lines of a
    float32 array"""
    m, n = array.shape
    I = np.zeros((m, k), dtype='int64')
    D = np.zeros((m, k), dtype='float32')
    ha = float_minheap_array_t()
    ha.ids = swig_ptr(I)
    ha.val = swig_ptr(D)
    ha.nh = m
    ha.k = k
    ha.heapify()
    ha.addn(n, swig_ptr(array))
    ha.reorder()
    return D, I


def pairwise_distances(xq, xb, mt=METRIC_L2, metric_arg=0):
    """compute the whole pairwise distance matrix between two sets of
    vectors"""
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2
    dis = np.empty((nq, nb), dtype='float32')
    if mt == METRIC_L2:
        pairwise_L2sqr(
            d, nq, swig_ptr(xq),
            nb, swig_ptr(xb),
            swig_ptr(dis))
    else:
        pairwise_extra_distances(
            d, nq, swig_ptr(xq),
            nb, swig_ptr(xb),
            mt, metric_arg,
            swig_ptr(dis))
    return dis




def rand(n, seed=12345):
    res = np.empty(n, dtype='float32')
    float_rand(swig_ptr(res), res.size, seed)
    return res


def randint(n, seed=12345, vmax=None):
    res = np.empty(n, dtype='int64')
    if vmax is None:
        int64_rand(swig_ptr(res), res.size, seed)
    else:
        int64_rand_max(swig_ptr(res), res.size, vmax, seed)
    return res

lrand = randint

def randn(n, seed=12345):
    res = np.empty(n, dtype='float32')
    float_randn(swig_ptr(res), res.size, seed)
    return res


def eval_intersection(I1, I2):
    """ size of intersection between each line of two result tables"""
    n = I1.shape[0]
    assert I2.shape[0] == n
    k1, k2 = I1.shape[1], I2.shape[1]
    ninter = 0
    for i in range(n):
        ninter += ranklist_intersection_size(
            k1, swig_ptr(I1[i]), k2, swig_ptr(I2[i]))
    return ninter


def normalize_L2(x):
    fvec_renorm_L2(x.shape[1], x.shape[0], swig_ptr(x))

######################################################
# MapLong2Long interface
######################################################

def replacement_map_add(self, keys, vals):
    n, = keys.shape
    assert (n,) == keys.shape
    self.add_c(n, swig_ptr(keys), swig_ptr(vals))

def replacement_map_search_multiple(self, keys):
    n, = keys.shape
    vals = np.empty(n, dtype='int64')
    self.search_multiple_c(n, swig_ptr(keys), swig_ptr(vals))
    return vals

replace_method(MapLong2Long, 'add', replacement_map_add)
replace_method(MapLong2Long, 'search_multiple', replacement_map_search_multiple)

######################################################
# search_with_parameters interface
######################################################

search_with_parameters_c = search_with_parameters

def search_with_parameters(index, x, k, params=None, output_stats=False):
    n, d = x.shape
    assert d == index.d
    if not params:
        # if not provided use the ones set in the IVF object
        params = IVFSearchParameters()
        index_ivf = extract_index_ivf(index)
        params.nprobe = index_ivf.nprobe
        params.max_codes = index_ivf.max_codes
    nb_dis = np.empty(1, 'uint64')
    ms_per_stage = np.empty(3, 'float64')
    distances = np.empty((n, k), dtype=np.float32)
    labels = np.empty((n, k), dtype=np.int64)
    search_with_parameters_c(
        index, n, swig_ptr(x),
        k, swig_ptr(distances),
        swig_ptr(labels),
        params, swig_ptr(nb_dis), swig_ptr(ms_per_stage)
    )
    if not output_stats:
        return distances, labels
    else:
        stats = {
            'ndis': nb_dis[0],
            'pre_transform_ms': ms_per_stage[0],
            'coarse_quantizer_ms': ms_per_stage[1],
            'invlist_scan_ms': ms_per_stage[2],
        }
        return distances, labels, stats

range_search_with_parameters_c = range_search_with_parameters

def range_search_with_parameters(index, x, radius, params=None, output_stats=False):
    n, d = x.shape
    assert d == index.d
    if not params:
        # if not provided use the ones set in the IVF object
        params = IVFSearchParameters()
        index_ivf = extract_index_ivf(index)
        params.nprobe = index_ivf.nprobe
        params.max_codes = index_ivf.max_codes
    nb_dis = np.empty(1, 'uint64')
    ms_per_stage = np.empty(3, 'float64')
    res = RangeSearchResult(n)
    range_search_with_parameters_c(
        index, n, swig_ptr(x),
        radius, res,
        params, swig_ptr(nb_dis), swig_ptr(ms_per_stage)
    )
    lims = rev_swig_ptr(res.lims, n + 1).copy()
    nd = int(lims[-1])
    Dout = rev_swig_ptr(res.distances, nd).copy()
    Iout = rev_swig_ptr(res.labels, nd).copy()
    if not output_stats:
        return lims, Dout, Iout
    else:
        stats = {
            'ndis': nb_dis[0],
            'pre_transform_ms': ms_per_stage[0],
            'coarse_quantizer_ms': ms_per_stage[1],
            'invlist_scan_ms': ms_per_stage[2],
        }
        return lims, Dout, Iout, stats


######################################################
# KNN function
######################################################

def knn(xq, xb, k, distance_type=METRIC_L2):
    """ wrapper around the faiss knn functions without index """
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2

    I = np.empty((nq, k), dtype='int64')
    D = np.empty((nq, k), dtype='float32')

    if distance_type == METRIC_L2:
        heaps = float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = swig_ptr(D)
        heaps.ids = swig_ptr(I)
        knn_L2sqr(
            swig_ptr(xq), swig_ptr(xb),
            d, nq, nb, heaps
        )
    elif distance_type == METRIC_INNER_PRODUCT:
        heaps = float_minheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = swig_ptr(D)
        heaps.ids = swig_ptr(I)
        knn_inner_product(
            swig_ptr(xq), swig_ptr(xb),
            d, nq, nb, heaps
        )
    return D, I


###########################################
# Kmeans object
###########################################


class Kmeans:
    """shallow wrapper around the Clustering object. The important method
    is train()."""

    def __init__(self, d, k, **kwargs):
        """d: input dimension, k: nb of centroids. Additional
         parameters are passed on the ClusteringParameters object,
         including niter=25, verbose=False, spherical = False
        """
        self.d = d
        self.k = k
        self.gpu = False
        self.cp = ClusteringParameters()
        for k, v in kwargs.items():
            if k == 'gpu':
                self.gpu = v
            else:
                # if this raises an exception, it means that it is a non-existent field
                getattr(self.cp, k)
                setattr(self.cp, k, v)
        self.centroids = None

    def train(self, x, weights=None, init_centroids=None):
        n, d = x.shape
        assert d == self.d
        clus = Clustering(d, self.k, self.cp)
        if init_centroids is not None:
            nc, d2 = init_centroids.shape
            assert d2 == d
            copy_array_to_vector(init_centroids.ravel(), clus.centroids)
        if self.cp.spherical:
            self.index = IndexFlatIP(d)
        else:
            self.index = IndexFlatL2(d)
        if self.gpu:
            if self.gpu == True:
                ngpu = -1
            else:
                ngpu = self.gpu
            self.index = index_cpu_to_all_gpus(self.index, ngpu=ngpu)
        clus.train(x, self.index, weights)
        centroids = vector_float_to_array(clus.centroids)
        self.centroids = centroids.reshape(self.k, d)
        stats = clus.iteration_stats
        stats = [stats.at(i) for i in range(stats.size())]
        self.obj = np.array([st.obj for st in stats])
        # copy all the iteration_stats objects to a python array
        stat_fields = 'obj time time_search imbalance_factor nsplit'.split()
        self.iteration_stats = [
            {field: getattr(st, field) for field in stat_fields}
            for st in stats
        ]
        return self.obj[-1] if self.obj.size > 0 else 0.0

    def assign(self, x):
        assert self.centroids is not None, "should train before assigning"
        self.index.reset()
        self.index.add(self.centroids)
        D, I = self.index.search(x, 1)
        return D.ravel(), I.ravel()

# IndexProxy was renamed to IndexReplicas, remap the old name for any old code
# people may have
IndexProxy = IndexReplicas
ConcatenatedInvertedLists = HStackInvertedLists

###########################################
# serialization of indexes to byte arrays
###########################################

def serialize_index(index):
    """ convert an index to a numpy uint8 array  """
    writer = VectorIOWriter()
    write_index(index, writer)
    return vector_to_array(writer.data)

def deserialize_index(data):
    reader = VectorIOReader()
    copy_array_to_vector(data, reader.data)
    return read_index(reader)

def serialize_index_binary(index):
    """ convert an index to a numpy uint8 array  """
    writer = VectorIOWriter()
    write_index_binary(index, writer)
    return vector_to_array(writer.data)

def deserialize_index_binary(data):
    reader = VectorIOReader()
    copy_array_to_vector(data, reader.data)
    return read_index_binary(reader)


###########################################
# ResultHeap
###########################################

class ResultHeap:
    """Accumulate query results from a sliced dataset. The final result will
    be in self.D, self.I."""

    def __init__(self, nq, k):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        heaps = float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = swig_ptr(self.D)
        heaps.ids = swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_result(self, D, I):
        """D, I do not need to be in a particular order (heap or sorted)"""
        assert D.shape == (self.nq, self.k)
        assert I.shape == (self.nq, self.k)
        self.heaps.addn_with_ids(
            self.k, swig_ptr(D),
            swig_ptr(I), self.k)

    def finalize(self):
        self.heaps.reorder()
