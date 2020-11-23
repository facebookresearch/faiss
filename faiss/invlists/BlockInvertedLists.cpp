/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/BlockInvertedLists.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

BlockInvertedLists::BlockInvertedLists (
        size_t nlist, size_t n_per_block,
        size_t block_size):
    InvertedLists (nlist, (size_t)(-1)),
    n_per_block(n_per_block), block_size(block_size)
{
    ids.resize (nlist);
    codes.resize (nlist);
}

size_t BlockInvertedLists::add_entries (
           size_t list_no, size_t n_entry,
           const idx_t* ids_in, const uint8_t *code)
{
    if (n_entry == 0) return 0;
    assert (list_no < nlist);
    size_t o = ids [list_no].size();
    assert(o == 0); // not clear how we should handle subsequent adds
    ids [list_no].resize (o + n_entry);
    memcpy (&ids[list_no][o], ids_in, sizeof (ids_in[0]) * n_entry);

    size_t n_block = (n_entry + n_per_block - 1) / n_per_block;
    codes [list_no].resize (n_block * block_size);
    memcpy (&codes[list_no][o * code_size], code, n_block * block_size);
    return o;
}

size_t BlockInvertedLists::list_size(size_t list_no) const
{
    assert (list_no < nlist);
    return ids[list_no].size();
}

const uint8_t * BlockInvertedLists::get_codes (size_t list_no) const
{
    assert (list_no < nlist);
    return codes[list_no].get();
}

const InvertedLists::idx_t * BlockInvertedLists::get_ids (size_t list_no) const
{
    assert (list_no < nlist);
    return ids[list_no].data();
}

void BlockInvertedLists::resize (size_t list_no, size_t new_size)
{
    ids[list_no].resize (new_size);
    size_t prev_nbytes = codes[list_no].size();
    size_t n_block = (new_size + n_per_block - 1) / n_per_block;
    size_t new_nbytes = n_block * block_size;
    codes[list_no].resize (new_nbytes);
    if (prev_nbytes < new_nbytes) {
        // set new elements to 0
        memset(
            codes[list_no].data() + prev_nbytes, 0,
            new_nbytes - prev_nbytes
        );
    }
}

void BlockInvertedLists::update_entries (
      size_t list_no, size_t offset, size_t n_entry,
      const idx_t *ids_in, const uint8_t *codes_in)
{
    FAISS_THROW_MSG("not impemented");
    /*
    assert (list_no < nlist);
    assert (n_entry + offset <= ids[list_no].size());
    memcpy (&ids[list_no][offset], ids_in, sizeof(ids_in[0]) * n_entry);
    memcpy (&codes[list_no][offset * code_size], codes_in, code_size * n_entry);
    */
}


BlockInvertedLists::~BlockInvertedLists ()
{}



} // namespace faiss