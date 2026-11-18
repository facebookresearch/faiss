/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/DiskIOBackend.h>

#include <libaio.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <vector>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

DiskIOBackend::DiskIOBackend(int max_events_in) {
    FAISS_THROW_IF_NOT(max_events_in > 0);
    const int ret = io_setup(max_events_in, &ctx);
    FAISS_THROW_IF_NOT_FMT(
            ret == 0,
            "io_setup failed: %s%s",
            strerror(-ret),
            ret == -EAGAIN ? " (too many libaio contexts, see fs.aio-max-nr)"
                           : "");
    pid = getpid();
    max_events = max_events_in;
    iocbs.resize(max_events);
    iocb_ptrs.resize(max_events);
    events.resize(max_events);
}

DiskIOBackend::~DiskIOBackend() {
    // blocks until the reads in flight are done, so the caller can free their
    // buffers afterwards. The context of a parent process is not ours to
    // destroy.
    if (owned_by_this_process()) {
        io_destroy(ctx);
    }
}

void DiskIOBackend::stage(
        int fd,
        void* buf,
        size_t len,
        size_t offset,
        void* user_data) {
    FAISS_THROW_IF_NOT_MSG(
            n_staged < max_events, "DiskIOBackend: too many reads staged");
    iocb* cb = &iocbs[n_staged];
    io_prep_pread(cb, fd, buf, len, static_cast<long long>(offset));
    cb->data = user_data;
    iocb_ptrs[n_staged] = cb;
    n_staged++;
}

void DiskIOBackend::submit() {
    int done = 0;
    while (done < n_staged) {
        const int ret =
                io_submit(ctx, n_staged - done, iocb_ptrs.data() + done);
        if (ret == -EINTR || ret == -EAGAIN) {
            continue; // transient, the queue is sized for all reads in flight
        }
        FAISS_THROW_IF_NOT_FMT(
                ret > 0,
                "io_submit failed: %s",
                ret < 0 ? strerror(-ret) : "no read accepted");
        done += ret;
        n_in_flight += ret;
    }
    n_staged = 0;
}

int DiskIOBackend::reap(void** user_data, long* res, int max, int min_nr) {
    max = std::min(max, max_events);
    min_nr = std::min(min_nr, std::min(max, n_in_flight));
    struct timespec no_wait = {0, 0};
    const int ret = io_getevents(
            ctx, min_nr, max, events.data(), min_nr > 0 ? nullptr : &no_wait);
    if (ret == -EINTR) {
        return 0;
    }
    FAISS_THROW_IF_NOT_FMT(ret >= 0, "io_getevents failed: %s", strerror(-ret));
    n_in_flight -= ret;
    for (int i = 0; i < ret; i++) {
        user_data[i] = events[i].data;
        // bytes read or a negative errno, in an unsigned field
        res[i] = static_cast<long>(events[i].res);
    }
    return ret;
}

bool DiskIOBackend::wait_all() noexcept {
    n_staged = 0;
    while (n_in_flight > 0) {
        const int ret = io_getevents(
                ctx, n_in_flight, max_events, events.data(), nullptr);
        if (ret == -EINTR) {
            continue;
        }
        if (ret < 0) {
            return false;
        }
        n_in_flight -= ret;
    }
    return true;
}

bool DiskIOBackend::owned_by_this_process() const {
    return pid == getpid();
}

} // namespace faiss
