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

#ifdef FAISS_ENABLE_DISK_IVF
#include <faiss/invlists/RaBitQSSDInvertedLists.h>
#endif

namespace faiss {

/**********************************************************
 * InvertedListIOHook's
 **********************************************************/

InvertedListsIOHook::InvertedListsIOHook(
        const std::string& key_in,
        const std::string& classname_in)
        : key(key_in), classname(classname_in) {}

namespace {

/// std::vector that deletes its contents
struct IOHookTable : std::vector<InvertedListsIOHook*> {
    IOHookTable() {
#ifndef _WIN32
        push_back(new OnDiskInvertedListsIOHook());
#endif
#ifdef FAISS_ENABLE_DISK_IVF
        push_back(new RaBitQSSDInvertedListsIOHook());
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
        if (static_cast<uint32_t>(h) == fourcc(callback->key)) {
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

std::string ondisk_same_dir_filename(IOReader* f, const std::string& filename) {
    FileIOReader* reader = dynamic_cast<FileIOReader*>(f);
    FAISS_THROW_IF_NOT_MSG(
            reader,
            "IO_FLAG_ONDISK_SAME_DIR only supported "
            "when reading from file");
    std::string dirname = "./";
    size_t slash = reader->name.find_last_of('/');
    if (slash != std::string::npos) {
        dirname = reader->name.substr(0, slash + 1);
    }
    std::string basename = filename;
    slash = basename.find_last_of('/');
    if (slash != std::string::npos) {
        basename = basename.substr(slash + 1);
    }
    return dirname + basename;
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
