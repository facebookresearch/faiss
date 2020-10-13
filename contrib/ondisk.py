# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import faiss
import logging

LOG = logging.getLogger(__name__)

def merge_ondisk(trained_index: faiss.Index,
                 shard_fnames: List[str],
                 ivfdata_fname: str) -> None:
    """ Add the contents of the indexes stored in shard_fnames into the index
    trained_index. The on-disk data is stored in ivfdata_fname """
    # merge the images into an on-disk index
    # first load the inverted lists
    ivfs = []
    for fname in shard_fnames:
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        LOG.info("read " + fname)
        index = faiss.read_index(fname, faiss.IO_FLAG_MMAP)
        index_ivf = faiss.extract_index_ivf(index)
        ivfs.append(index_ivf.invlists)

        # avoid that the invlists get deallocated with the index
        index_ivf.own_invlists = False

    # construct the output index
    index = trained_index
    index_ivf = faiss.extract_index_ivf(index)

    assert index.ntotal == 0, "works only on empty index"

    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    invlists = faiss.OnDiskInvertedLists(
        index_ivf.nlist, index_ivf.code_size,
        ivfdata_fname)

    # merge all the inverted lists
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in ivfs:
        ivf_vector.push_back(ivf)

    LOG.info("merge %d inverted lists " % ivf_vector.size())
    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

    # now replace the inverted lists in the output index
    index.ntotal = index_ivf.ntotal = ntotal
    index_ivf.replace_invlists(invlists, True)
    invlists.this.disown()
