/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/DiskIOBackend.h>

#if !FAISS_HAS_LIBAIO
#error "DiskIOBackend needs Linux and libaio"
#endif

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

struct DiskIOBackend::Impl {
    io_context_t ctx = nullptr;
    pid_t pid = 0; // process that created ctx
    int max_events = 0;
    int n_staged = 0;
    int n_in_flight = 0; // submitted and not reaped
    std::vector<struct iocb> iocbs;
    std::vector<struct iocb*> iocb_ptrs;
    std::vector<struct io_event> events;
};

DiskIOBackend::DiskIOBackend(int max_events) : impl(new Impl) {
    FAISS_THROW_IF_NOT(max_events > 0);
    const int ret = io_setup(max_events, &impl->ctx);
    FAISS_THROW_IF_NOT_FMT(
            ret == 0,
            "io_setup failed: %s%s",
            strerror(-ret),
            ret == -EAGAIN ? " (too many libaio contexts, see fs.aio-max-nr)"
                           : "");
    impl->pid = getpid();
    impl->max_events = max_events;
    impl->iocbs.resize(max_events);
    impl->iocb_ptrs.resize(max_events);
    impl->events.resize(max_events);
}

DiskIOBackend::~DiskIOBackend() {
    // blocks until the reads in flight are done, so the caller can free their
    // buffers afterwards. The context of a parent process is not ours to
    // destroy.
    if (owned_by_this_process()) {
        io_destroy(impl->ctx);
    }
}

void DiskIOBackend::stage(
        int fd,
        void* buf,
        size_t len,
        size_t offset,
        void* user_data) {
    FAISS_THROW_IF_NOT_MSG(
            impl->n_staged < impl->max_events,
            "DiskIOBackend: too many reads staged");
    struct iocb* cb = &impl->iocbs[impl->n_staged];
    io_prep_pread(cb, fd, buf, len, static_cast<long long>(offset));
    cb->data = user_data;
    impl->iocb_ptrs[impl->n_staged] = cb;
    impl->n_staged++;
}

void DiskIOBackend::submit() {
    int done = 0;
    while (done < impl->n_staged) {
        const int ret = io_submit(
                impl->ctx,
                impl->n_staged - done,
                impl->iocb_ptrs.data() + done);
        if (ret == -EINTR || ret == -EAGAIN) {
            continue; // transient, the queue is sized for all reads in flight
        }
        FAISS_THROW_IF_NOT_FMT(
                ret > 0,
                "io_submit failed: %s",
                ret < 0 ? strerror(-ret) : "no read accepted");
        done += ret;
        impl->n_in_flight += ret;
    }
    impl->n_staged = 0;
}

int DiskIOBackend::reap(void** user_data, long* res, int max) {
    max = std::min(max, impl->max_events);
    struct timespec no_wait = {0, 0};
    const int ret =
            io_getevents(impl->ctx, 0, max, impl->events.data(), &no_wait);
    if (ret == -EINTR) {
        return 0;
    }
    FAISS_THROW_IF_NOT_FMT(ret >= 0, "io_getevents failed: %s", strerror(-ret));
    impl->n_in_flight -= ret;
    for (int i = 0; i < ret; i++) {
        user_data[i] = impl->events[i].data;
        // bytes read or a negative errno, in an unsigned field
        res[i] = static_cast<long>(impl->events[i].res);
    }
    return ret;
}

bool DiskIOBackend::wait_all() noexcept {
    impl->n_staged = 0;
    while (impl->n_in_flight > 0) {
        const int ret = io_getevents(
                impl->ctx,
                impl->n_in_flight,
                impl->max_events,
                impl->events.data(),
                nullptr);
        if (ret == -EINTR) {
            continue;
        }
        if (ret < 0) {
            return false;
        }
        impl->n_in_flight -= ret;
    }
    return true;
}

int DiskIOBackend::max_events() const {
    return impl->max_events;
}

bool DiskIOBackend::owned_by_this_process() const {
    return impl->pid == getpid();
}

} // namespace faiss
