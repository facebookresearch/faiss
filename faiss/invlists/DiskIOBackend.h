/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libaio.h>
#include <sys/types.h>

#include <cstddef>
#include <vector>

namespace faiss {

/** Asynchronous reads with Linux libaio, for one thread.
 *
 * Reads are staged with stage(), sent to the kernel together with submit(),
 * and collected with reap(). With a file opened with
 * O_DIRECT, the buffer address, offset and length of every read must be
 * multiples of the device sector size.
 */
struct DiskIOBackend {
    io_context_t ctx = nullptr;
    pid_t pid = 0;       ///< process that created ctx
    int max_events = 0;  ///< at most max_events reads can be in flight
    int n_staged = 0;    ///< reads staged for the next submit()
    int n_in_flight = 0; ///< submitted and not reaped
    std::vector<iocb> iocbs;
    std::vector<iocb*> iocb_ptrs;
    std::vector<io_event> events;

    explicit DiskIOBackend(int max_events);
    ~DiskIOBackend();

    DiskIOBackend(const DiskIOBackend&) = delete;
    DiskIOBackend& operator=(const DiskIOBackend&) = delete;

    /// add a read of len bytes at offset into buf to the next submit();
    /// user_data identifies it in reap()
    void stage(int fd, void* buf, size_t len, size_t offset, void* user_data);

    /// send the staged reads to the kernel
    void submit();

    /** Collect up to max completed reads, waiting until at least min_nr
     * have completed (without waiting if min_nr is 0). min_nr is capped by
     * max and by the reads in flight, and a signal can end the wait early.
     *
     * @param user_data  output, the user_data of each completed read
     * @param res        output, the bytes read, or a negative errno
     * @return           number of completed reads
     */
    int reap(void** user_data, long* res, int max, int min_nr = 0);

    /** Drop the staged reads and wait for the submitted ones to complete,
     * ignoring their results, so that their buffers can be freed and the
     * context reused.
     *
     * @return false if waiting failed: reads may still be in flight
     */
    bool wait_all() noexcept;

    /// false in a process forked after the context was created: the context
    /// belongs to the parent process and cannot be used
    bool owned_by_this_process() const;
};

} // namespace faiss
