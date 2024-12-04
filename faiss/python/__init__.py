# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

# not linting this file because it imports * from swigfaiss, which
# causes a ton of useless warnings.

import numpy as np
import sys
import inspect

# We import * so that the symbol foo can be accessed as faiss.foo.
from .loader import *

# additional wrappers
from faiss import class_wrappers
from faiss.gpu_wrappers import *
from faiss.array_conversions import *
from faiss.extra_wrappers import kmin, kmax, pairwise_distances, rand, randint, \
    lrand, randn, rand_smooth_vectors, eval_intersection, normalize_L2, \
    ResultHeap, knn, Kmeans, checksum, matrix_bucket_sort_inplace, bucket_sort, \
    merge_knn_results, MapInt64ToInt64, knn_hamming, \
    pack_bitstrings, unpack_bitstrings


__version__ = "%d.%d.%d" % (FAISS_VERSION_MAJOR,
                            FAISS_VERSION_MINOR,
                            FAISS_VERSION_PATCH)

class_wrappers.handle_Clustering(Clustering)
class_wrappers.handle_Clustering1D(Clustering1D)
class_wrappers.handle_MatrixStats(MatrixStats)
class_wrappers.handle_IOWriter(IOWriter)
class_wrappers.handle_IOReader(IOReader)
class_wrappers.handle_AutoTuneCriterion(AutoTuneCriterion)
class_wrappers.handle_ParameterSpace(ParameterSpace)
class_wrappers.handle_NSG(IndexNSG)
class_wrappers.handle_MapLong2Long(MapLong2Long)
class_wrappers.handle_IDSelectorSubset(IDSelectorBatch, class_owns=True)
class_wrappers.handle_IDSelectorSubset(IDSelectorArray, class_owns=False)
class_wrappers.handle_IDSelectorSubset(IDSelectorBitmap, class_owns=False, force_int64=False)
class_wrappers.handle_CodeSet(CodeSet)

class_wrappers.handle_Tensor2D(Tensor2D)
class_wrappers.handle_Tensor2D(Int32Tensor2D)
class_wrappers.handle_Embedding(Embedding)
class_wrappers.handle_Linear(Linear)
class_wrappers.handle_QINCo(QINCo)
class_wrappers.handle_QINCoStep(QINCoStep)


this_module = sys.modules[__name__]

# handle sub-classes
for symbol in dir(this_module):
    obj = getattr(this_module, symbol)
    # print symbol, isinstance(obj, (type, types.ClassType))
    if inspect.isclass(obj):
        the_class = obj
        if issubclass(the_class, Index):
            class_wrappers.handle_Index(the_class)

        if issubclass(the_class, IndexBinary):
            class_wrappers.handle_IndexBinary(the_class)

        if issubclass(the_class, VectorTransform):
            class_wrappers.handle_VectorTransform(the_class)

        if issubclass(the_class, Quantizer):
            class_wrappers.handle_Quantizer(the_class)

        if issubclass(the_class, IndexRowwiseMinMax) or \
                issubclass(the_class, IndexRowwiseMinMaxFP16):
            class_wrappers.handle_IndexRowwiseMinMax(the_class)

        if issubclass(the_class, SearchParameters):
            class_wrappers.handle_SearchParameters(the_class)

        if issubclass(the_class, CodePacker):
            class_wrappers.handle_CodePacker(the_class)

##############################################################################
# For some classes (IndexIVF, IDSelector), the object holds a reference to
# a C++ object (eg. the quantizer object of IndexIVF). We don't transfer the
# ownership to the C++ object (ie. set own_quantizer=true), but instead we add
# a reference in the Python class wrapper instead. This is done via an
# additional referenced_objects field.
#
# Since the semantics of ownership in the C++ classes are sometimes irregular,
# these references are added manually using the functions below.
##############################################################################


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

def add_to_referenced_objects(self, ref):
    if not hasattr(self, 'referenced_objects'):
        self.referenced_objects = [ref]
    else:
        self.referenced_objects.append(ref)


def add_ref_in_method(the_class, method_name, parameter_no):
    original_method = getattr(the_class, method_name)

    def replacement_method(self, *args):
        ref = args[parameter_no]
        add_to_referenced_objects(self, ref)
        return original_method(self, *args)
    setattr(the_class, method_name, replacement_method)


def add_ref_in_method_explicit_own(the_class, method_name):
    # for methods of format set_XXX(object, own)
    original_method = getattr(the_class, method_name)

    def replacement_method(self, ref, own=False):
        if not own:
            if not hasattr(self, 'referenced_objects'):
                self.referenced_objects = [ref]
            else:
                self.referenced_objects.append(ref)
        else:
            # transfer ownership to C++ class
            ref.this.disown()
        return original_method(self, ref, own)
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


try:
    from swigfaiss_gpu import GpuIndexIVFFlat, GpuIndexBinaryFlat, GpuIndexFlat, GpuIndexIVFPQ, GpuIndexIVFScalarQuantizer
    add_ref_in_constructor(GpuIndexIVFFlat, 1)
    add_ref_in_constructor(GpuIndexBinaryFlat, 1)
    add_ref_in_constructor(GpuIndexFlat, 1)
    add_ref_in_constructor(GpuIndexIVFPQ, 1)
    add_ref_in_constructor(GpuIndexIVFScalarQuantizer, 1)
except ImportError as e:
    print("Failed to load GPU Faiss: %s. Will not load constructor refs for GPU indexes." % e.args[0])

add_ref_in_constructor(IndexIVFFlat, 0)
add_ref_in_constructor(IndexIVFFlatDedup, 0)
add_ref_in_constructor(IndexPreTransform, {2: [0, 1], 1: [0]})
add_ref_in_method(IndexPreTransform, 'prepend_transform', 0)
add_ref_in_constructor(IndexIVFPQ, 0)
add_ref_in_constructor(IndexIVFPQR, 0)
add_ref_in_constructor(IndexIVFPQFastScan, 0)
add_ref_in_constructor(IndexIVFResidualQuantizer, 0)
add_ref_in_constructor(IndexIVFLocalSearchQuantizer, 0)
add_ref_in_constructor(IndexIVFResidualQuantizerFastScan, 0)
add_ref_in_constructor(IndexIVFLocalSearchQuantizerFastScan, 0)
add_ref_in_constructor(IndexIVFSpectralHash, 0)
add_ref_in_method_explicit_own(IndexIVFSpectralHash, "replace_vt")

add_ref_in_constructor(Index2Layer, 0)
add_ref_in_constructor(Level1Quantizer, 0)
add_ref_in_constructor(IndexIVFScalarQuantizer, 0)
add_ref_in_constructor(IndexRowwiseMinMax, 0)
add_ref_in_constructor(IndexRowwiseMinMaxFP16, 0)
add_ref_in_constructor(IndexIDMap, 0)
add_ref_in_constructor(IndexIDMap2, 0)
add_ref_in_constructor(IndexHNSW, 0)
add_ref_in_method(IndexShards, 'add_shard', 0)
add_ref_in_method(IndexBinaryShards, 'add_shard', 0)
add_ref_in_constructor(IndexRefineFlat, {2: [0], 1: [0]})
add_ref_in_constructor(IndexRefine, {2: [0, 1]})

add_ref_in_constructor(IndexBinaryIVF, 0)
add_ref_in_constructor(IndexBinaryFromFloat, 0)
add_ref_in_constructor(IndexBinaryIDMap, 0)
add_ref_in_constructor(IndexBinaryIDMap2, 0)

add_ref_in_method(IndexReplicas, 'addIndex', 0)
add_ref_in_method(IndexBinaryReplicas, 'addIndex', 0)

add_ref_in_constructor(BufferedIOWriter, 0)
add_ref_in_constructor(BufferedIOReader, 0)

add_ref_in_constructor(IDSelectorNot, 0)
add_ref_in_constructor(IDSelectorAnd, slice(2))
add_ref_in_constructor(IDSelectorOr, slice(2))
add_ref_in_constructor(IDSelectorXOr, slice(2))
add_ref_in_constructor(IDSelectorTranslated, slice(2))

add_ref_in_constructor(IDSelectorXOr, slice(2))
add_ref_in_constructor(IndexIVFIndependentQuantizer, slice(3))

# seems really marginal...
# remove_ref_from_method(IndexReplicas, 'removeIndex', 0)


######################################################
# search_with_parameters interface
######################################################

search_with_parameters_c = search_with_parameters


def search_with_parameters(index, x, k, params=None, output_stats=False):
    x = np.ascontiguousarray(x, dtype='float32')
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
    x = np.ascontiguousarray(x, dtype='float32')
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


# IndexProxy was renamed to IndexReplicas, remap the old name for any old code
# people may have
IndexProxy = IndexReplicas
ConcatenatedInvertedLists = HStackInvertedLists
IndexResidual = IndexResidualQuantizer

IVFSearchParameters = SearchParametersIVF

###########################################
# serialization of indexes to byte arrays
###########################################


def serialize_index(index, io_flags=0):
    """ convert an index to a numpy uint8 array  """
    writer = VectorIOWriter()
    write_index(index, writer, io_flags)
    return vector_to_array(writer.data)


def deserialize_index(data, io_flags=0):
    reader = VectorIOReader()
    copy_array_to_vector(data, reader.data)
    return read_index(reader, io_flags)


def serialize_index_binary(index):
    """ convert an index to a numpy uint8 array  """
    writer = VectorIOWriter()
    write_index_binary(index, writer)
    return vector_to_array(writer.data)


def deserialize_index_binary(data):
    reader = VectorIOReader()
    copy_array_to_vector(data, reader.data)
    return read_index_binary(reader)


class TimeoutGuard:
    def __init__(self, timeout_in_seconds: float):
        self.timeout = timeout_in_seconds

    def __enter__(self):
        TimeoutCallback.reset(self.timeout)

    def __exit__(self, exc_type, exc_value, traceback):
        PythonInterruptCallback.reset()
