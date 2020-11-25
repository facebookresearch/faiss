# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss

def get_invlist(invlists, l):
    """ returns the inverted lists content as a pair of (list_ids, list_codes).
    The codes are reshaped to a proper size
    """
    invlists = faiss.downcast_InvertedLists(invlists)
    ls = invlists.list_size(l)
    list_ids = np.zeros(ls, dtype='int64')
    ids = codes = None
    try:
        ids = invlists.get_ids(l)
        faiss.memcpy(faiss.swig_ptr(list_ids), ids, list_ids.nbytes)
        codes = invlists.get_codes(l)
        if invlists.code_size != faiss.InvertedLists.INVALID_CODE_SIZE:
            list_codes = np.zeros((ls, invlists.code_size), dtype='uint8')
        else:
            # it's a BlockInvertedLists
            npb = invlists.n_per_block
            bs = invlists.block_size
            ls_round = (ls + npb - 1) // npb
            list_codes = np.zeros((ls_round, bs // npb, npb), dtype='uint8')
        faiss.memcpy(faiss.swig_ptr(list_codes), codes, list_codes.nbytes)
    finally:
        if ids is not None:
            invlists.release_ids(l, ids)
        if codes is not None:
            invlists.release_codes(l, codes)
    return list_ids, list_codes


def get_invlist_sizes(invlists):
    """ return the array of sizes of the inverted lists """
    return np.array([
        invlists.list_size(i)
        for i in range(invlists.nlist)
    ], dtype='int64')


def print_object_fields(obj):
    """ list values all fields of an object known to SWIG """

    for name in obj.__class__.__swig_getmethods__:
        print(f"{name} = {getattr(obj, name)}")


def get_pq_centroids(pq):
    """ return the PQ centroids as an array """
    cen = faiss.vector_to_array(pq.centroids)
    return cen.reshape(pq.M, pq.ksub, pq.dsub)


def get_LinearTransform_matrix(pca):
    """ extract matrix + bias from the PCA object
    works for any linear transform (OPQ, random rotation, etc.)
    """
    b = faiss.vector_to_array(pca.b)
    A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
    return A, b
