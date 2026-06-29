/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_EXCEPTION_INCLUDED
#define FAISS_EXCEPTION_INCLUDED

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace faiss {

/// Base class for Faiss exceptions
class FaissException : public std::exception {
   public:
    explicit FaissException(const std::string& msg);

    FaissException(
            const std::string& msg,
            const char* funcName,
            const char* file,
            int line);

    /// from std::exception
    const char* what() const noexcept override;

    std::string msg;
};

/// Handle multiple exceptions from worker threads, throwing an appropriate
/// exception that aggregates the information
/// The pair int is the thread that generated the exception
void handleExceptions(
        std::vector<std::pair<int, std::exception_ptr>>& exceptions);

/** RAII object for a set of possibly transformed vectors (deallocated only if
 * they are indeed transformed)
 */
struct TransformedVectors {
    const float* x;
    bool own_x;
    TransformedVectors(const float* x_orig, const float* x_in)
            : x(x_in), own_x(x_orig != x_in) {}

    ~TransformedVectors() {
        if (own_x) {
            delete[] x;
        }
    }
};

/// make typeids more readable
std::string demangle_cpp_symbol(const char* name);

/// Capture the current exception into `ex` if no prior exception has been
/// recorded.  Call from a catch block inside an OpenMP parallel region.
/// Uses `#pragma omp critical` to serialize access to `ex`.
///
/// The optional `cleanup` callable runs inside the critical section
/// alongside the exception capture, so that side-effects visible to
/// other threads (e.g. setting an interrupt flag) are serialized with
/// the exception_ptr write.
///
/// Usage:
///   std::exception_ptr ex;
///   bool interrupt = false;
///   #pragma omp parallel
///   {
///       try { ... } catch (...) {
///           omp_capture_exception(ex, [&] { interrupt = true; });
///       }
///   }
///   omp_rethrow_if_exception(ex);
inline void omp_capture_exception(std::exception_ptr& ex) {
#pragma omp critical(faiss_omp_exception)
    {
        if (!ex) {
            ex = std::current_exception();
        }
    }
}

/// Overload with cleanup that runs inside the critical section.
template <typename Cleanup>
inline void omp_capture_exception(std::exception_ptr& ex, Cleanup&& cleanup) {
#pragma omp critical(faiss_omp_exception)
    {
        cleanup();
        if (!ex) {
            ex = std::current_exception();
        }
    }
}

/// Rethrow the captured exception, if any.  Call on the main thread
/// after the parallel region completes.
inline void omp_rethrow_if_exception(std::exception_ptr& ex) {
    if (ex) {
        std::rethrow_exception(ex);
    }
}

} // namespace faiss

#endif
