/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <variant>

namespace faiss {

class SpinLock {
   private:
    std::atomic<bool> locked_{false};
    static constexpr int INIT_BACKOFF = 1;
    static constexpr int YIELD_THRESHOLD = 64;
    static constexpr int MAX_BACKOFF = 1024;

    inline void pause_briefly() {
#if defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause" ::: "memory");
#elif defined(__arm__) || defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }

   public:
    bool try_lock() {
        return !locked_.exchange(true, std::memory_order_acquire);
    }

    void lock() {
        int backoff = INIT_BACKOFF;

        while (!try_lock()) {
            while (locked_.load(std::memory_order_relaxed)) {
                if (backoff < YIELD_THRESHOLD) {
                    for (int i = 0; i < backoff; ++i) {
                        pause_briefly();
                    }
                } else {
                    std::this_thread::yield();
                }

                if (backoff < MAX_BACKOFF) {
                    backoff <<= 1;
                }
            }
        }

        backoff = INIT_BACKOFF;
    }

    inline void unlock() {
        locked_.store(false, std::memory_order_release);
    }
};

class AdaptiveLock {
   private:
    std::variant<std::monostate, std::mutex, SpinLock> impl_;
    bool use_spinlock_;
    mutable std::mutex config_mutex_;

   public:
    AdaptiveLock(bool use_spinlock = true) : use_spinlock_(use_spinlock) {
        if (use_spinlock) {
            impl_.emplace<SpinLock>();
        } else {
            impl_.emplace<std::mutex>();
        }
    }

    void set_use_spinlock(bool use_spinlock) {
        std::lock_guard<std::mutex> guard(config_mutex_);
        use_spinlock_ = use_spinlock;
        if (use_spinlock) {
            impl_.emplace<SpinLock>();
        } else {
            impl_.emplace<std::mutex>();
        }
    }

    bool get_use_spinlock() const {
        std::lock_guard<std::mutex> guard(config_mutex_);
        return use_spinlock_;
    }

    void lock() {
        std::visit(
                [](auto& m) {
                    if constexpr (!std::is_same_v<
                                          std::decay_t<decltype(m)>,
                                          std::monostate>) {
                        m.lock();
                    }
                },
                impl_);
    }

    void unlock() {
        std::visit(
                [](auto& m) {
                    if constexpr (!std::is_same_v<
                                          std::decay_t<decltype(m)>,
                                          std::monostate>) {
                        m.unlock();
                    }
                },
                impl_);
    }
};

} // namespace faiss
