# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss

def get_invlist(invlists, l):
    """ returns the inverted lists content. """
    ls = invlists.list_size(l)
    list_ids = np.zeros(ls, dtype='int64')
    ids = codes = None
    try:
        ids = invlists.get_ids(l)
        faiss.memcpy(faiss.swig_ptr(list_ids), ids, list_ids.nbytes)
        codes = invlists.get_codes(l)
        list_codes = np.zeros((ls, invlists.code_size), dtype='uint8')
        faiss.memcpy(faiss.swig_ptr(list_codes), codes, list_codes.nbytes)
    finally:
        if ids is not None:
            invlists.release_ids(l, ids)
        if codes is not None:
            invlists.release_codes(l, codes)
    return list_ids, list_codes
