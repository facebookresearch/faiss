/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/InvertedListsIOHook.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

#include <faiss/invlists/BlockInvertedLists.h>

#ifndef _WIN32
#include <faiss/invlists/OnDiskInvertedLists.h>
#endif // !_WIN32

namespace faiss {

/**********************************************************
 * InvertedListIOHook's
 **********************************************************/

InvertedListsIOHook::InvertedListsIOHook(
        const std::string& key,
        const std::string& classname)
        : key(key), classname(classname) {}

namespace {

/// std::vector that deletes its contents
struct IOHookTable : std::vector<InvertedListsIOHook*> {
    IOHookTable() {
#ifndef _WIN32
        push_back(new OnDiskInvertedListsIOHook());
#endif
        push_back(new BlockInvertedListsIOHook());
    }

    ~IOHookTable() {
        for (auto x : *this) {
            delete x;
        }
    }
};

static IOHookTable InvertedListsIOHook_table;

} // namespace

InvertedListsIOHook* InvertedListsIOHook::lookup(int h) {
    for (const auto& callback : InvertedListsIOHook_table) {
        if (h == fourcc(callback->key)) {
            return callback;
        }
    }
    FAISS_THROW_FMT(
            "read_InvertedLists: could not load ArrayInvertedLists as "
            "%08x (\"%s\")",
            h,
            fourcc_inv_printable(h).c_str());
}

InvertedListsIOHook* InvertedListsIOHook::lookup_classname(
        const std::string& classname) {
    for (const auto& callback : InvertedListsIOHook_table) {
        if (callback->classname == classname) {
            return callback;
        }
    }
    FAISS_THROW_FMT(
            "read_InvertedLists: could not find classname %s",
            classname.c_str());
}

void InvertedListsIOHook::add_callback(InvertedListsIOHook* cb) {
    InvertedListsIOHook_table.push_back(cb);
}

void InvertedListsIOHook::print_callbacks() {
    printf("registered %zd InvertedListsIOHooks:\n",
           InvertedListsIOHook_table.size());
    for (const auto& cb : InvertedListsIOHook_table) {
        printf("%08x %s %s\n",
               fourcc(cb->key.c_str()),
               cb->key.c_str(),
               cb->classname.c_str());
    }
}

InvertedLists* InvertedListsIOHook::read_ArrayInvertedLists(
        IOReader*,
        int,
        size_t,
        size_t,
        const std::vector<size_t>&) const {
    FAISS_THROW_FMT("read to array not implemented for %s", classname.c_str());
}

} // namespace faiss
