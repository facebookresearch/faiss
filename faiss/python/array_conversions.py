# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

# not linting this file because it imports * from swigfaiss, which
# causes a ton of useless warnings.

import numpy as np
import array
import warnings

from faiss.loader import *

###########################################
# Utility to add a deprecation warning to
# classes from the SWIG interface
###########################################

def _make_deprecated_swig_class(deprecated_name, base_name):
    """
    Dynamically construct deprecated classes as wrappers around renamed ones

    The deprecation warning added in their __new__-method will trigger upon
    construction of an instance of the class, but only once per session.

    We do this here (in __init__.py) because the base classes are defined in
    the SWIG interface, making it cumbersome to add the deprecation there.

    Parameters
    ----------
    deprecated_name : string
        Name of the class to be deprecated; _not_ present in SWIG interface.
    base_name : string
        Name of the class that is replacing deprecated_name; must already be
        imported into the current namespace.

    Returns
    -------
    None
        However, the deprecated class gets added to the faiss namespace
    """
    base_class = globals()[base_name]

    def new_meth(cls, *args, **kwargs):
        msg = f"The class faiss.{deprecated_name} is deprecated in favour of faiss.{base_name}!"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        instance = super(base_class, cls).__new__(cls, *args, **kwargs)
        return instance

    # three-argument version of "type" uses (name, tuple-of-bases, dict-of-attributes)
    klazz = type(deprecated_name, (base_class,), {"__new__": new_meth})

    # this ends up adding the class to the "faiss" namespace, in a way that it
    # is available both through "import faiss" and "from faiss import *"
    globals()[deprecated_name] = klazz


###########################################
# numpy array / std::vector conversions
###########################################

sizeof_long = array.array('l').itemsize
deprecated_name_map = {
    # deprecated: replacement
    'Float': 'Float32',
    'Double': 'Float64',
    'Char': 'Int8',
    'Int': 'Int32',
    'Long': 'Int32' if sizeof_long == 4 else 'Int64',
    'LongLong': 'Int64',
    'Byte': 'UInt8',
    # previously misspelled variant
    'Uint64': 'UInt64',
}

for depr_prefix, base_prefix in deprecated_name_map.items():
    _make_deprecated_swig_class(depr_prefix + "Vector", base_prefix + "Vector")

    # same for the three legacy *VectorVector classes
    if depr_prefix in ['Float', 'Long', 'Byte']:
        _make_deprecated_swig_class(depr_prefix + "VectorVector",
                                    base_prefix + "VectorVector")

# mapping from vector names in swigfaiss.swig and the numpy dtype names
# TODO: once deprecated classes are removed, remove the dict and just use .lower() below
vector_name_map = {
    'Float32': 'float32',
    'Float64': 'float64',
    'Int8': 'int8',
    'Int16': 'int16',
    'Int32': 'int32',
    'Int64': 'int64',
    'UInt8': 'uint8',
    'UInt16': 'uint16',
    'UInt32': 'uint32',
    'UInt64': 'uint64',
    **{k: v.lower() for k, v in deprecated_name_map.items()}
}


def vector_to_array(v):
    """ convert a C++ vector to a numpy array """
    classname = v.__class__.__name__
    if classname.startswith('AlignedTable'):
        return AlignedTable_to_array(v)
    if classname.startswith('MaybeOwnedVector'):
        dtype = np.dtype(vector_name_map[classname[16:]])
        a = np.empty(v.size(), dtype=dtype)
        if v.size() > 0:
            memcpy(swig_ptr(a), v.data(), a.nbytes)
        return a

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
    if classname.startswith('MaybeOwnedVector'):
        assert v.is_owned, 'cannot copy to an non-owned MaybeOwnedVector'
        dtype = np.dtype(vector_name_map[classname[16:]])
        assert dtype == a.dtype, (
            'cannot copy a %s array to a %s (should be %s)' % (
                a.dtype, classname, dtype))
        v.resize(n)
        if n > 0:
            memcpy(v.data(), swig_ptr(a), a.nbytes)
        return

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
