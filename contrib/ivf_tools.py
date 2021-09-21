# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss

def add_preassigned(index_ivf, x, a, ids=None):
    """
    Add elements to an IVF index, where the assignment is already computed
    """
    n, d = x.shape
    assert a.shape == (n, )
    if isinstance(index_ivf, faiss.IndexBinaryIVF):
        d *= 8
    assert d == index_ivf.d
    if ids is not None:
        assert ids.shape == (n, )
        ids = faiss.swig_ptr(ids)
    index_ivf.add_core(
        n, faiss.swig_ptr(x), ids, faiss.swig_ptr(a)
    )


def search_preassigned(index_ivf, xq, k, list_nos, coarse_dis=None):
    """
    Perform a search in the IVF index, with predefined lists to search into
    """
    n, d = xq.shape
    if isinstance(index_ivf, faiss.IndexBinaryIVF):
        d *= 8
        dis_type = "int32"
    else:
        dis_type = "float32"

    assert d == index_ivf.d
    assert list_nos.shape == (n, index_ivf.nprobe)

    # the coarse distances are used in IVFPQ with L2 distance and by_residual=True
    # otherwise we provide dummy coarse_dis
    if coarse_dis is None:
        coarse_dis = np.zeros((n, index_ivf.nprobe), dtype=dis_type)
    else:
        assert coarse_dis.shape == (n, index_ivf.nprobe)

    D = np.empty((n, k), dtype=dis_type)
    I = np.empty((n, k), dtype='int64')

    sp = faiss.swig_ptr
    index_ivf.search_preassigned(
        n, sp(xq), k,
        sp(list_nos), sp(coarse_dis), sp(D), sp(I), False)
    return D, I


def range_search_preassigned(index_ivf, x, radius, list_nos, coarse_dis=None):
    """
    Perform a range search in the IVF index, with predefined lists to search into
    """
    n, d = x.shape
    if isinstance(index_ivf, faiss.IndexBinaryIVF):
        d *= 8
        dis_type = "int32"
    else:
        dis_type = "float32"

    # the coarse distances are used in IVFPQ with L2 distance and by_residual=True
    # otherwise we provide dummy coarse_dis
    if coarse_dis is None:
        coarse_dis = np.empty((n, index_ivf.nprobe), dtype=dis_type)
    else:
        assert coarse_dis.shape == (n, index_ivf.nprobe)

    assert d == index_ivf.d
    assert list_nos.shape == (n, index_ivf.nprobe)

    res = faiss.RangeSearchResult(n)
    sp = faiss.swig_ptr

    index_ivf.range_search_preassigned(
        n, sp(x), radius,
        sp(list_nos), sp(coarse_dis),
        res
    )
    # get pointers and copy them
    lims = faiss.rev_swig_ptr(res.lims, n + 1).copy()
    num_results = int(lims[-1])
    dist = faiss.rev_swig_ptr(res.distances, num_results).copy()
    indices = faiss.rev_swig_ptr(res.labels, num_results).copy()
    return lims, dist, indices
