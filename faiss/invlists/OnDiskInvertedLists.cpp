/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/invlists/OnDiskInvertedLists.h>

#include <pthread.h>

#include <unordered_set>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

namespace faiss {

/**********************************************
 * LockLevels
 **********************************************/

struct LockLevels {
    /* There n times lock1(n), one lock2 and one lock3
     * Invariants:
     *    a single thread can hold one lock1(n) for some n
     *    a single thread can hold lock2, if it holds lock1(n) for some n
     *    a single thread can hold lock3, if it holds lock1(n) for some n
     *       AND lock2 AND no other thread holds lock1(m) for m != n
     */
    pthread_mutex_t mutex1;
    pthread_cond_t level1_cv;
    pthread_cond_t level2_cv;
    pthread_cond_t level3_cv;

    std::unordered_set<int> level1_holders; // which level1 locks are held
    int n_level2;                           // nb threads that wait on level2
    bool level3_in_use;                     // a threads waits on level3
    bool level2_in_use;

    LockLevels() {
        pthread_mutex_init(&mutex1, nullptr);
        pthread_cond_init(&level1_cv, nullptr);
        pthread_cond_init(&level2_cv, nullptr);
        pthread_cond_init(&level3_cv, nullptr);
        n_level2 = 0;
        level2_in_use = false;
        level3_in_use = false;
    }

    ~LockLevels() {
        pthread_cond_destroy(&level1_cv);
        pthread_cond_destroy(&level2_cv);
        pthread_cond_destroy(&level3_cv);
        pthread_mutex_destroy(&mutex1);
    }

    void lock_1(int no) {
        pthread_mutex_lock(&mutex1);
        while (level3_in_use || level1_holders.contains(no)) {
            pthread_cond_wait(&level1_cv, &mutex1);
        }
        level1_holders.insert(no);
        pthread_mutex_unlock(&mutex1);
    }

    void unlock_1(int no) {
        pthread_mutex_lock(&mutex1);
        assert(level1_holders.contains(no));
        level1_holders.erase(no);
        if (level3_in_use) { // a writer is waiting
            pthread_cond_signal(&level3_cv);
        } else {
            pthread_cond_broadcast(&level1_cv);
        }
        pthread_mutex_unlock(&mutex1);
    }

    void lock_2() {
        pthread_mutex_lock(&mutex1);
        n_level2++;
        if (level3_in_use) { // tell waiting level3 that we are blocked
            pthread_cond_signal(&level3_cv);
        }
        while (level2_in_use) {
            pthread_cond_wait(&level2_cv, &mutex1);
        }
        level2_in_use = true;
        pthread_mutex_unlock(&mutex1);
    }

    void unlock_2() {
        pthread_mutex_lock(&mutex1);
        level2_in_use = false;
        n_level2--;
        pthread_cond_signal(&level2_cv);
        pthread_mutex_unlock(&mutex1);
    }

    void lock_3() {
        pthread_mutex_lock(&mutex1);
        level3_in_use = true;
        // wait until there are no level1 holders anymore except the
        // ones that are waiting on level2 (we are holding lock2)
        while (level1_holders.size() > n_level2) {
            pthread_cond_wait(&level3_cv, &mutex1);
        }
        // don't release the lock!
    }

    void unlock_3() {
        level3_in_use = false;
        // wake up all level1_holders
        pthread_cond_broadcast(&level1_cv);
        pthread_mutex_unlock(&mutex1);
    }

    void print() {
        pthread_mutex_lock(&mutex1);
        printf("State: level3_in_use=%d n_level2=%d level1_holders: [",
               int(level3_in_use),
               n_level2);
        for (int k : level1_holders) {
            printf("%d ", k);
        }
        printf("]\n");
        pthread_mutex_unlock(&mutex1);
    }
};

/**********************************************
 * OngoingPrefetch
 **********************************************/

struct OnDiskInvertedLists::OngoingPrefetch {
    struct Thread {
        pthread_t pth;
        OngoingPrefetch* pf;

        bool one_list() {
            idx_t list_no = pf->get_next_list();
            if (list_no == -1)
                return false;
            const OnDiskInvertedLists* od = pf->od;
            od->locks->lock_1(list_no);
            size_t n = od->list_size(list_no);
            const idx_t* idx = od->get_ids(list_no);
            const uint8_t* codes = od->get_codes(list_no);
            int cs = 0;
            for (size_t i = 0; i < n; i++) {
                cs += idx[i];
            }
            const idx_t* codes8 = (const idx_t*)codes;
            idx_t n8 = n * od->code_size / 8;

            for (size_t i = 0; i < n8; i++) {
                cs += codes8[i];
            }
            od->locks->unlock_1(list_no);

            global_cs += cs & 1;
            return true;
        }
    };

    std::vector<Thread> threads;

    pthread_mutex_t list_ids_mutex;
    std::vector<idx_t> list_ids;
    int cur_list;

    // mutex for the list of tasks
    pthread_mutex_t mutex;

    // pretext to avoid code below to be optimized out
    static int global_cs;

    const OnDiskInvertedLists* od;

    explicit OngoingPrefetch(const OnDiskInvertedLists* od) : od(od) {
        pthread_mutex_init(&mutex, nullptr);
        pthread_mutex_init(&list_ids_mutex, nullptr);
        cur_list = 0;
    }

    static void* prefetch_list(void* arg) {
        Thread* th = static_cast<Thread*>(arg);

        while (th->one_list())
            ;

        return nullptr;
    }

    idx_t get_next_list() {
        idx_t list_no = -1;
        pthread_mutex_lock(&list_ids_mutex);
        if (cur_list >= 0 && cur_list < list_ids.size()) {
            list_no = list_ids[cur_list++];
        }
        pthread_mutex_unlock(&list_ids_mutex);
        return list_no;
    }

    void prefetch_lists(const idx_t* list_nos, int n) {
        pthread_mutex_lock(&mutex);
        pthread_mutex_lock(&list_ids_mutex);
        list_ids.clear();
        pthread_mutex_unlock(&list_ids_mutex);
        for (auto& th : threads) {
            pthread_join(th.pth, nullptr);
        }

        threads.resize(0);
        cur_list = 0;
        int nt = std::min(n, od->prefetch_nthread);

        if (nt > 0) {
            // prepare tasks
            for (int i = 0; i < n; i++) {
                idx_t list_no = list_nos[i];
                if (list_no >= 0 && od->list_size(list_no) > 0) {
                    list_ids.push_back(list_no);
                }
            }
            // prepare threads
            threads.resize(nt);
            for (Thread& th : threads) {
                th.pf = this;
                pthread_create(&th.pth, nullptr, prefetch_list, &th);
            }
        }
        pthread_mutex_unlock(&mutex);
    }

    ~OngoingPrefetch() {
        pthread_mutex_lock(&mutex);
        for (auto& th : threads) {
            pthread_join(th.pth, nullptr);
        }
        pthread_mutex_unlock(&mutex);
        pthread_mutex_destroy(&mutex);
        pthread_mutex_destroy(&list_ids_mutex);
    }
};

int OnDiskInvertedLists::OngoingPrefetch::global_cs = 0;

void OnDiskInvertedLists::prefetch_lists(const idx_t* list_nos, int n) const {
    pf->prefetch_lists(list_nos, n);
}

/**********************************************
 * OnDiskInvertedLists: mmapping
 **********************************************/

void OnDiskInvertedLists::do_mmap() {
    const char* rw_flags = read_only ? "r" : "r+";
    int prot = read_only ? PROT_READ : PROT_WRITE | PROT_READ;
    FILE* f = fopen(filename.c_str(), rw_flags);
    FAISS_THROW_IF_NOT_FMT(
            f,
            "could not open %s in mode %s: %s",
            filename.c_str(),
            rw_flags,
            strerror(errno));

    uint8_t* ptro =
            (uint8_t*)mmap(nullptr, totsize, prot, MAP_SHARED, fileno(f), 0);

    fclose(f);

    FAISS_THROW_IF_NOT_FMT(
            ptro != MAP_FAILED,
            "could not mmap %s: %s",
            filename.c_str(),
            strerror(errno));
    ptr = ptro;
}

void OnDiskInvertedLists::update_totsize(size_t new_size) {
    // unmap file
    if (ptr != nullptr) {
        int err = munmap(ptr, totsize);
        FAISS_THROW_IF_NOT_FMT(err == 0, "munmap error: %s", strerror(errno));
    }
    if (totsize == 0) {
        // must create file before truncating it
        FILE* f = fopen(filename.c_str(), "w");
        FAISS_THROW_IF_NOT_FMT(
                f,
                "could not open %s in mode W: %s",
                filename.c_str(),
                strerror(errno));
        fclose(f);
    }

    if (new_size > totsize) {
        if (!slots.empty() &&
            slots.back().offset + slots.back().capacity == totsize) {
            slots.back().capacity += new_size - totsize;
        } else {
            slots.push_back(Slot(totsize, new_size - totsize));
        }
    } else {
        assert(!"not implemented");
    }

    totsize = new_size;

    // create file
    printf("resizing %s to %zd bytes\n", filename.c_str(), totsize);

    int err = truncate(filename.c_str(), totsize);

    FAISS_THROW_IF_NOT_FMT(
            err == 0,
            "truncate %s to %ld: %s",
            filename.c_str(),
            totsize,
            strerror(errno));
    do_mmap();
}

/**********************************************
 * OnDiskInvertedLists
 **********************************************/

#define INVALID_OFFSET (size_t)(-1)

OnDiskOneList::OnDiskOneList() : size(0), capacity(0), offset(INVALID_OFFSET) {}

OnDiskInvertedLists::Slot::Slot(size_t offset, size_t capacity)
        : offset(offset), capacity(capacity) {}

OnDiskInvertedLists::Slot::Slot() : offset(0), capacity(0) {}

OnDiskInvertedLists::OnDiskInvertedLists(
        size_t nlist,
        size_t code_size,
        const char* filename)
        : InvertedLists(nlist, code_size),
          filename(filename),
          totsize(0),
          ptr(nullptr),
          read_only(false),
          locks(new LockLevels()),
          pf(new OngoingPrefetch(this)),
          prefetch_nthread(32) {
    lists.resize(nlist);

    // slots starts empty
}

OnDiskInvertedLists::OnDiskInvertedLists() : OnDiskInvertedLists(0, 0, "") {}

OnDiskInvertedLists::~OnDiskInvertedLists() {
    delete pf;

    // unmap all lists
    if (ptr != nullptr) {
        int err = munmap(ptr, totsize);
        if (err != 0) {
            fprintf(stderr, "mumap error: %s", strerror(errno));
        }
    }
    delete locks;
}

size_t OnDiskInvertedLists::list_size(size_t list_no) const {
    return lists[list_no].size;
}

const uint8_t* OnDiskInvertedLists::get_codes(size_t list_no) const {
    if (lists[list_no].offset == INVALID_OFFSET) {
        return nullptr;
    }

    return ptr + lists[list_no].offset;
}

const idx_t* OnDiskInvertedLists::get_ids(size_t list_no) const {
    if (lists[list_no].offset == INVALID_OFFSET) {
        return nullptr;
    }

    return (const idx_t*)(ptr + lists[list_no].offset +
                          code_size * lists[list_no].capacity);
}

void OnDiskInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* codes_in) {
    FAISS_THROW_IF_NOT(!read_only);
    if (n_entry == 0)
        return;
    [[maybe_unused]] const List& l = lists[list_no];
    assert(n_entry + offset <= l.size);
    idx_t* ids = const_cast<idx_t*>(get_ids(list_no));
    memcpy(ids + offset, ids_in, sizeof(ids_in[0]) * n_entry);
    uint8_t* codes = const_cast<uint8_t*>(get_codes(list_no));
    memcpy(codes + offset * code_size, codes_in, code_size * n_entry);
}

size_t OnDiskInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) {
    FAISS_THROW_IF_NOT(!read_only);
    locks->lock_1(list_no);
    size_t o = list_size(list_no);
    resize_locked(list_no, n_entry + o);
    update_entries(list_no, o, n_entry, ids, code);
    locks->unlock_1(list_no);
    return o;
}

void OnDiskInvertedLists::resize(size_t list_no, size_t new_size) {
    FAISS_THROW_IF_NOT(!read_only);
    locks->lock_1(list_no);
    resize_locked(list_no, new_size);
    locks->unlock_1(list_no);
}

void OnDiskInvertedLists::resize_locked(size_t list_no, size_t new_size) {
    List& l = lists[list_no];

    if (new_size <= l.capacity && new_size > l.capacity / 2) {
        l.size = new_size;
        return;
    }

    // otherwise we release the current slot, and find a new one

    locks->lock_2();
    free_slot(l.offset, l.capacity);

    List new_l;

    if (new_size == 0) {
        new_l = List();
    } else {
        new_l.size = new_size;
        new_l.capacity = 1;
        while (new_l.capacity < new_size) {
            new_l.capacity *= 2;
        }
        new_l.offset =
                allocate_slot(new_l.capacity * (sizeof(idx_t) + code_size));
    }

    // copy common data
    if (l.offset != new_l.offset) {
        size_t n = std::min(new_size, l.size);
        if (n > 0) {
            memcpy(ptr + new_l.offset, get_codes(list_no), n * code_size);
            memcpy(ptr + new_l.offset + new_l.capacity * code_size,
                   get_ids(list_no),
                   n * sizeof(idx_t));
        }
    }

    lists[list_no] = new_l;
    locks->unlock_2();
}

size_t OnDiskInvertedLists::allocate_slot(size_t capacity) {
    // should hold lock2

    auto it = slots.begin();
    while (it != slots.end() && it->capacity < capacity) {
        it++;
    }

    if (it == slots.end()) {
        // not enough capacity
        size_t new_size = totsize == 0 ? 32 : totsize * 2;
        while (new_size - totsize < capacity) {
            new_size *= 2;
        }
        locks->lock_3();
        update_totsize(new_size);
        locks->unlock_3();
        it = slots.begin();
        while (it != slots.end() && it->capacity < capacity) {
            it++;
        }
        assert(it != slots.end());
    }

    size_t o = it->offset;
    if (it->capacity == capacity) {
        slots.erase(it);
    } else {
        // take from beginning of slot
        it->capacity -= capacity;
        it->offset += capacity;
    }

    return o;
}

void OnDiskInvertedLists::free_slot(size_t offset, size_t capacity) {
    // should hold lock2
    if (capacity == 0)
        return;

    auto it = slots.begin();
    while (it != slots.end() && it->offset <= offset) {
        it++;
    }

    size_t inf = ((size_t)1) << 60;

    size_t end_prev = inf;
    if (it != slots.begin()) {
        auto prev = it;
        prev--;
        end_prev = prev->offset + prev->capacity;
    }

    size_t begin_next = ((size_t)1) << 60;
    if (it != slots.end()) {
        begin_next = it->offset;
    }

    assert(end_prev == inf || offset >= end_prev);
    assert(offset + capacity <= begin_next);

    if (offset == end_prev) {
        auto prev = it;
        prev--;
        if (offset + capacity == begin_next) {
            prev->capacity += capacity + it->capacity;
            slots.erase(it);
        } else {
            prev->capacity += capacity;
        }
    } else {
        if (offset + capacity == begin_next) {
            it->offset -= capacity;
            it->capacity += capacity;
        } else {
            slots.insert(it, Slot(offset, capacity));
        }
    }

    // TODO shrink global storage if needed
}

/*****************************************
 * Compact form
 *****************************************/
size_t OnDiskInvertedLists::merge_from_multiple(
        const InvertedLists** ils,
        int n_il,
        bool shift_ids,
        bool verbose) {
    FAISS_THROW_IF_NOT_MSG(
            totsize == 0, "works only on an empty InvertedLists");

    std::vector<size_t> sizes(nlist);
    std::vector<size_t> shift_id_offsets(n_il);
    for (int i = 0; i < n_il; i++) {
        const InvertedLists* il = ils[i];
        FAISS_THROW_IF_NOT(il->nlist == nlist && il->code_size == code_size);

        for (size_t j = 0; j < nlist; j++) {
            sizes[j] += il->list_size(j);
        }

        size_t il_totsize = il->compute_ntotal();
        shift_id_offsets[i] =
                (shift_ids && i > 0) ? shift_id_offsets[i - 1] + il_totsize : 0;
    }

    size_t cums = 0;
    size_t ntotal = 0;
    for (size_t j = 0; j < nlist; j++) {
        ntotal += sizes[j];
        lists[j].size = 0;
        lists[j].capacity = sizes[j];
        lists[j].offset = cums;
        cums += lists[j].capacity * (sizeof(idx_t) + code_size);
    }

    update_totsize(cums);

    size_t nmerged = 0;
    double t0 = getmillisecs(), last_t = t0;

#pragma omp parallel for
    for (size_t j = 0; j < nlist; j++) {
        List& l = lists[j];
        for (int i = 0; i < n_il; i++) {
            const InvertedLists* il = ils[i];
            size_t n_entry = il->list_size(j);
            l.size += n_entry;
            ScopedIds scope_ids(il, j);
            const idx_t* scope_ids_data = scope_ids.get();
            std::vector<idx_t> new_ids;
            if (shift_ids) {
                new_ids.resize(n_entry);
                for (size_t k = 0; k < n_entry; k++) {
                    new_ids[k] = scope_ids[k] + shift_id_offsets[i];
                }
                scope_ids_data = new_ids.data();
            }
            update_entries(
                    j,
                    l.size - n_entry,
                    n_entry,
                    scope_ids_data,
                    ScopedCodes(il, j).get());
        }
        assert(l.size == l.capacity);
        if (verbose) {
#pragma omp critical
            {
                nmerged++;
                double t1 = getmillisecs();
                if (t1 - last_t > 500) {
                    printf("merged %zd lists in %.3f s\r",
                           nmerged,
                           (t1 - t0) / 1000.0);
                    fflush(stdout);
                    last_t = t1;
                }
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    return ntotal;
}

size_t OnDiskInvertedLists::merge_from_1(
        const InvertedLists* ils,
        bool verbose) {
    return merge_from_multiple(&ils, 1, verbose);
}

void OnDiskInvertedLists::crop_invlists(size_t l0, size_t l1) {
    FAISS_THROW_IF_NOT(0 <= l0 && l0 <= l1 && l1 <= nlist);

    std::vector<List> new_lists(l1 - l0);
    memcpy(new_lists.data(), &lists[l0], (l1 - l0) * sizeof(List));

    lists.swap(new_lists);

    nlist = l1 - l0;
}

void OnDiskInvertedLists::set_all_lists_sizes(const size_t* sizes) {
    size_t ofs = 0;
    for (size_t i = 0; i < nlist; i++) {
        lists[i].offset = ofs;
        lists[i].capacity = lists[i].size = sizes[i];
        ofs += sizes[i] * (sizeof(idx_t) + code_size);
    }
}

/*******************************************************
 * I/O support via callbacks
 *******************************************************/

OnDiskInvertedListsIOHook::OnDiskInvertedListsIOHook()
        : InvertedListsIOHook("ilod", typeid(OnDiskInvertedLists).name()) {}

void OnDiskInvertedListsIOHook::write(const InvertedLists* ils, IOWriter* f)
        const {
    uint32_t h = fourcc("ilod");
    WRITE1(h);
    WRITE1(ils->nlist);
    WRITE1(ils->code_size);
    const OnDiskInvertedLists* od =
            dynamic_cast<const OnDiskInvertedLists*>(ils);
    // this is a POD object
    WRITEVECTOR(od->lists);

    {
        std::vector<OnDiskInvertedLists::Slot> v(
                od->slots.begin(), od->slots.end());
        WRITEVECTOR(v);
    }
    {
        std::vector<char> x(od->filename.begin(), od->filename.end());
        WRITEVECTOR(x);
    }
    WRITE1(od->totsize);
}

InvertedLists* OnDiskInvertedListsIOHook::read(IOReader* f, int io_flags)
        const {
    OnDiskInvertedLists* od = new OnDiskInvertedLists();
    od->read_only = io_flags & IO_FLAG_READ_ONLY;
    READ1(od->nlist);
    READ1(od->code_size);
    // this is a POD object
    READVECTOR(od->lists);
    {
        std::vector<OnDiskInvertedLists::Slot> v;
        READVECTOR(v);
        od->slots.assign(v.begin(), v.end());
    }
    {
        std::vector<char> x;
        READVECTOR(x);
        od->filename.assign(x.begin(), x.end());

        if (io_flags & IO_FLAG_ONDISK_SAME_DIR) {
            FileIOReader* reader = dynamic_cast<FileIOReader*>(f);
            FAISS_THROW_IF_NOT_MSG(
                    reader,
                    "IO_FLAG_ONDISK_SAME_DIR only supported "
                    "when reading from file");
            std::string indexname = reader->name;
            std::string dirname = "./";
            size_t slash = indexname.find_last_of('/');
            if (slash != std::string::npos) {
                dirname = indexname.substr(0, slash + 1);
            }
            std::string filename = od->filename;
            slash = filename.find_last_of('/');
            if (slash != std::string::npos) {
                filename = filename.substr(slash + 1);
            }
            filename = dirname + filename;
            printf("IO_FLAG_ONDISK_SAME_DIR: "
                   "updating ondisk filename from %s to %s\n",
                   od->filename.c_str(),
                   filename.c_str());
            od->filename = filename;
        }
    }
    READ1(od->totsize);
    if (!(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        od->do_mmap();
    }
    return od;
}

/** read from a ArrayInvertedLists into this invertedlist type */
InvertedLists* OnDiskInvertedListsIOHook::read_ArrayInvertedLists(
        IOReader* f,
        int /* io_flags */,
        size_t nlist,
        size_t code_size,
        const std::vector<size_t>& sizes) const {
    auto ails = new OnDiskInvertedLists();
    ails->nlist = nlist;
    ails->code_size = code_size;
    ails->read_only = true;
    ails->lists.resize(nlist);

    FileIOReader* reader = dynamic_cast<FileIOReader*>(f);
    FAISS_THROW_IF_NOT_MSG(reader, "mmap only supported for File objects");
    FILE* fdesc = reader->f;
    size_t o0 = ftell(fdesc);
    size_t o = o0;
    { // do the mmap
        struct stat buf;
        int ret = fstat(fileno(fdesc), &buf);
        FAISS_THROW_IF_NOT_FMT(ret == 0, "fstat failed: %s", strerror(errno));
        ails->totsize = buf.st_size;
        ails->ptr = (uint8_t*)mmap(
                nullptr,
                ails->totsize,
                PROT_READ,
                MAP_SHARED,
                fileno(fdesc),
                0);
        FAISS_THROW_IF_NOT_FMT(
                ails->ptr != MAP_FAILED, "could not mmap: %s", strerror(errno));
    }

    FAISS_THROW_IF_NOT(o <= ails->totsize);

    for (size_t i = 0; i < ails->nlist; i++) {
        OnDiskInvertedLists::List& l = ails->lists[i];
        l.size = l.capacity = sizes[i];
        l.offset = o;
        o += l.size * (sizeof(idx_t) + ails->code_size);
    }
    // resume normal reading of file
    fseek(fdesc, o, SEEK_SET);

    return ails;
}

} // namespace faiss
