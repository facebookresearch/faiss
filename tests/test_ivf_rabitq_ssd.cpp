/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The tests that the Python bindings cannot run: fork() and DiskIOBackend.
// The other IndexIVFRaBitQSSD tests are in test_ivf_rabitq_ssd.py.

#include <fcntl.h>
#include <libaio.h>
#include <omp.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQSSD.h>
#include <faiss/invlists/DiskIOBackend.h>

using faiss::idx_t;
using faiss::IndexIVFRaBitQSSD;
using faiss::MetricType;

namespace {

/// a temporary file name, removed at destruction
struct TempFile {
    std::string name;

    explicit TempFile(const char* suffix = "") {
        char tmpl[] = "/tmp/faiss_rabitq_ssd_XXXXXX";
        int fd = mkstemp(tmpl);
        EXPECT_GE(fd, 0);
        close(fd);
        unlink(tmpl);
        name = std::string(tmpl) + suffix;
    }

    ~TempFile() {
        unlink(name.c_str());
    }

    const char* c_str() const {
        return name.c_str();
    }
};

/// n vectors around ncenters Gaussian centers
std::vector<float> make_data(size_t n, size_t d, size_t ncenters, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> centers(ncenters * d);
    for (auto& v : centers) {
        v = 2 * normal(rng);
    }
    std::vector<float> x(n * d);
    for (size_t i = 0; i < n; i++) {
        const float* c = centers.data() + (i % ncenters) * d;
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] = c[j] + normal(rng);
        }
    }
    return x;
}

struct Config {
    size_t d = 128;
    size_t mem_dim = 64;
    int nb_bits = 5;
    MetricType metric = faiss::METRIC_L2;
    MetricType quantizer_metric = faiss::METRIC_L2;
    bool use_hacc = true;
    size_t nlist = 8;
    size_t nb = 1000;
};

struct TestIndex {
    TempFile file;
    std::unique_ptr<faiss::Index> quantizer;
    std::unique_ptr<IndexIVFRaBitQSSD> index;
    std::vector<float> xb;

    explicit TestIndex(const Config& cfg, int seed = 123) {
        xb = make_data(cfg.nb, cfg.d, 3 * cfg.nlist, seed);
        quantizer =
                std::make_unique<faiss::IndexFlat>(cfg.d, cfg.quantizer_metric);
        index = std::make_unique<IndexIVFRaBitQSSD>(
                quantizer.get(),
                cfg.d,
                cfg.nlist,
                cfg.mem_dim,
                cfg.nb_bits,
                cfg.metric);
        index->attach_file(file.c_str());
        index->use_hacc = cfg.use_hacc;
        index->train(cfg.nb, xb.data());
        index->add(cfg.nb, xb.data());
    }
};

/// Number of results of the second search that differ from the first: an id
/// missing from the same query of the first search, or a distance more than
/// tol away. The order of equal distances can differ between searches.
size_t n_different_results(
        const std::vector<float>& D1,
        const std::vector<idx_t>& I1,
        const std::vector<float>& D2,
        const std::vector<idx_t>& I2,
        size_t nq,
        size_t k,
        float tol = 0) {
    size_t n_diff = 0;
    for (size_t q = 0; q < nq; q++) {
        std::map<idx_t, float> dis;
        for (size_t j = 0; j < k; j++) {
            dis[I1[q * k + j]] = D1[q * k + j];
        }
        for (size_t j = 0; j < k; j++) {
            auto it = dis.find(I2[q * k + j]);
            n_diff += it == dis.end() ||
                    std::abs(it->second - D2[q * k + j]) >
                            tol * std::max(1.0f, std::abs(it->second));
        }
    }
    return n_diff;
}

/// Exit status of a child process, -2 if it did not exit normally, or -1 if
/// it is still running after timeout_s seconds (it is then killed).
int wait_child(pid_t pid, int timeout_s) {
    for (int i = 0; i < timeout_s * 10; i++) {
        int wstatus = 0;
        if (waitpid(pid, &wstatus, WNOHANG) == pid) {
            return WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -2;
        }
        usleep(100000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    return -1;
}

} // namespace

/// The tests write their files to /tmp, which must support O_DIRECT.
class IVFRaBitQSSD : public ::testing::Test {
   protected:
    void SetUp() override {
        TempFile file;
        const int fd = open(
                file.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            GTEST_SKIP() << "O_DIRECT not supported in /tmp";
        }
        close(fd);
    }
};

// A process forked after a search cannot use the libaio contexts that its
// parent keeps: it creates its own.
TEST_F(IVFRaBitQSSD, search_after_fork) {
    Config cfg;
    TestIndex ti(cfg);
    const size_t nq = 5;
    // with k = nb nothing is pruned
    const idx_t k = cfg.nb;
    std::vector<float> xq = make_data(nq, cfg.d, 5, 789);
    ti.index->nprobe = 4;
    std::vector<float> D_ref(nq * k), D(nq * k);
    std::vector<idx_t> I_ref(nq * k), I(nq * k);
    ti.index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());

    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        // single-threaded: OpenMP parallel regions can hang after fork()
        omp_set_num_threads(1);
        int status = 2;
        try {
            ti.index->search(nq, xq.data(), k, D.data(), I.data());
            if (n_different_results(D_ref, I_ref, D, I, nq, k) == 0) {
                status = 0;
            }
        } catch (...) {
            status = 3;
        }
        _exit(status);
    }
    // 2: different results, 3: the search threw
    EXPECT_EQ(wait_child(pid, 30), 0);

    // a child that destroys the index drops the contexts of the parent
    pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        ti.index.reset();
        _exit(0);
    }
    // -1: the child hangs
    EXPECT_EQ(wait_child(pid, 30), 0);

    ti.index->search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_EQ(n_different_results(D_ref, I_ref, D, I, nq, k), 0);
}

// reap with min_nr waits for a request in flight instead of returning
// nothing. The request is a poll of an empty pipe rather than a read, as a
// read can complete inside io_submit: the poll completes only when another
// thread writes to the pipe, 100 ms after reap is called.
TEST(DiskIOBackend, reap_waits) {
    int fds[2];
    ASSERT_EQ(pipe(fds), 0);
    faiss::DiskIOBackend io(4);
    void* user_data = nullptr;
    long res = 0;
    // nothing in flight: min_nr is capped and reap does not wait
    EXPECT_EQ(io.reap(&user_data, &res, 4, 1), 0);

    iocb cb;
    io_prep_poll(&cb, fds[0], POLLIN);
    int tag = 0;
    cb.data = &tag;
    iocb* cbs[] = {&cb};
    ASSERT_EQ(io_submit(io.ctx, 1, cbs), 1);
    io.n_in_flight++;
    std::thread writer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EXPECT_EQ(write(fds[1], "x", 1), 1);
    });
    EXPECT_EQ(io.reap(&user_data, &res, 4, 1), 1);
    EXPECT_EQ(user_data, &tag);
    EXPECT_TRUE(res & POLLIN);
    writer.join();
    close(fds[0]);
    close(fds[1]);
}
