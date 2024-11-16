/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/io.h>
#include <faiss/invlists/InvertedLists.h>
#include <string>

namespace faiss {

/** Callbacks to handle other types of InvertedList objects.
 *
 * The callbacks should be registered with add_callback before calling
 * read_index or read_InvertedLists. The callbacks for
 * OnDiskInvertedLists are registrered by default. The invlist type is
 * identified by:
 *
 * - the key (a fourcc) at read time
 * - the class name (as given by typeid.name) at write time
 */
struct InvertedListsIOHook {
    const std::string key;       ///< string version of the fourcc
    const std::string classname; ///< typeid.name

    InvertedListsIOHook(const std::string& key, const std::string& classname);

    /// write the index to the IOWriter (including the fourcc)
    virtual void write(const InvertedLists* ils, IOWriter* f) const = 0;

    /// called when the fourcc matches this class's fourcc
    virtual InvertedLists* read(IOReader* f, int io_flags) const = 0;

    /** read from a ArrayInvertedLists into this invertedlist type.
     * For this to work, the callback has to be enabled and the io_flag has to
     * be set to IO_FLAG_SKIP_IVF_DATA | (16 upper bits of the fourcc)
     *
     * (default implementation fails)
     */
    virtual InvertedLists* read_ArrayInvertedLists(
            IOReader* f,
            int io_flags,
            size_t nlist,
            size_t code_size,
            const std::vector<size_t>& sizes) const;

    virtual ~InvertedListsIOHook() {}

    /**************************** Manage the set of callbacks ******/

    // transfers ownership
    static void add_callback(InvertedListsIOHook*);
    static void print_callbacks();
    static InvertedListsIOHook* lookup(int h);
    static InvertedListsIOHook* lookup_classname(const std::string& classname);
};

} // namespace faiss
