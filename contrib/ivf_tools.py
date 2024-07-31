# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss

from faiss.contrib.inspect_tools import get_invlist_sizes


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
    Perform a search in the IVF index, with predefined lists to search into.
    Supports indexes with pretransforms (as opposed to the
    IndexIVF.search_preassigned, that cannot be applied with pretransform).
    """
    if isinstance(index_ivf, faiss.IndexPreTransform):
        assert index_ivf.chain.size() == 1, "chain must have only one component"
        transform = faiss.downcast_VectorTransform(index_ivf.chain.at(0))
        xq = transform.apply(xq)
        index_ivf = faiss.downcast_index(index_ivf.index)
    n, d = xq.shape
    if isinstance(index_ivf, faiss.IndexBinaryIVF):
        d *= 8
        dis_type = "int32"
    else:
        dis_type = "float32"

    assert d == index_ivf.d
    assert list_nos.shape == (n, index_ivf.nprobe)

    # the coarse distances are used in IVFPQ with L2 distance and
    # by_residual=True otherwise we provide dummy coarse_dis
    if coarse_dis is None:
        coarse_dis = np.zeros((n, index_ivf.nprobe), dtype=dis_type)
    else:
        assert coarse_dis.shape == (n, index_ivf.nprobe)

    return index_ivf.search_preassigned(xq, k, list_nos, coarse_dis)


def range_search_preassigned(index_ivf, x, radius, list_nos, coarse_dis=None):
    """
    Perform a range search in the IVF index, with predefined lists to
    search into
    """
    n, d = x.shape
    if isinstance(index_ivf, faiss.IndexBinaryIVF):
        d *= 8
        dis_type = "int32"
    else:
        dis_type = "float32"

    # the coarse distances are used in IVFPQ with L2 distance and
    # by_residual=True otherwise we provide dummy coarse_dis
    if coarse_dis is None:
        coarse_dis = np.empty((n, index_ivf.nprobe), dtype=dis_type)
    else:
        assert coarse_dis.shape == (n, index_ivf.nprobe)

    assert d == index_ivf.d
    assert list_nos.shape == (n, index_ivf.nprobe)

    res = faiss.RangeSearchResult(n)
    sp = faiss.swig_ptr

    index_ivf.range_search_preassigned_c(
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


def replace_ivf_quantizer(index_ivf, new_quantizer):
    """ replace the IVF quantizer with a flat quantizer and return the
    old quantizer"""
    if new_quantizer.ntotal == 0:
        centroids = index_ivf.quantizer.reconstruct_n()
        new_quantizer.train(centroids)
        new_quantizer.add(centroids)
    else:
        assert new_quantizer.ntotal == index_ivf.nlist

    # cleanly dealloc old quantizer
    old_own = index_ivf.own_fields
    index_ivf.own_fields = False
    old_quantizer = faiss.downcast_index(index_ivf.quantizer)
    old_quantizer.this.own(old_own)
    index_ivf.quantizer = new_quantizer

    if hasattr(index_ivf, "referenced_objects"):
        index_ivf.referenced_objects.append(new_quantizer)
    else:
        index_ivf.referenced_objects = [new_quantizer]
    return old_quantizer


def permute_invlists(index_ivf, perm):
    """ Apply some permutation to the inverted lists, and modify the quantizer
    entries accordingly.
    Perm is an array of size nlist, where old_index = perm[new_index]
    """
    nlist, = perm.shape
    assert index_ivf.nlist == nlist
    quantizer = faiss.downcast_index(index_ivf.quantizer)
    assert quantizer.ntotal == index_ivf.nlist
    perm = np.ascontiguousarray(perm, dtype='int64')

    # just make sure it's a permutation...
    bc = np.bincount(perm, minlength=nlist)
    assert np.all(bc == np.ones(nlist, dtype=int))

    # handle quantizer
    quantizer.permute_entries(perm)

    # handle inverted lists
    invlists = faiss.downcast_InvertedLists(index_ivf.invlists)
    invlists.permute_invlists(faiss.swig_ptr(perm))


def sort_invlists_by_size(index_ivf):
    invlist_sizes = get_invlist_sizes(index_ivf.invlists)
    perm = np.argsort(invlist_sizes)
    permute_invlists(index_ivf, perm)
