/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFRaBitQSSD.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/RaBitQSSDInvertedLists.h>
#include <faiss/utils/distances.h>

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

/// queries near some of the vectors of xb
std::vector<float> make_queries(
        const std::vector<float>& xb,
        size_t d,
        size_t nq,
        int seed) {
    const size_t nb = xb.size() / d;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> xq(nq * d);
    for (size_t q = 0; q < nq; q++) {
        const float* x = xb.data() + (q * 37 % nb) * d;
        for (size_t j = 0; j < d; j++) {
            xq[q * d + j] = x[j] + 0.3f * normal(rng);
        }
    }
    return xq;
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
                file.c_str(),
                cfg.metric);
        index->use_hacc = cfg.use_hacc;
        index->train(cfg.nb, xb.data());
        index->add(cfg.nb, xb.data());
    }
};

/// Step of the look-up table of the in-memory scan: all the entries, the
/// sums of q over subsets of 4 consecutive dimensions, are quantized with one
/// offset and one step.
double lut_step(const float* q, size_t mem_dim, bool use_hacc) {
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    for (size_t t = 0; t < mem_dim / 4; t++) {
        for (int c = 0; c < 16; c++) {
            double v = 0;
            for (int j = 0; j < 4; j++) {
                if (c & (1 << j)) {
                    v += q[4 * t + j];
                }
            }
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    return (hi - lo) / (use_hacc ? 65535 : 255);
}

/// Reference estimate of the distance (similarity for IP) between q and
/// database vector o, computed in double from the RaBitQuantizer code:
///     <q - c, r> ~ |r|^2 <xb, q - c> / <xb, r>
/// Also returns, in tol, a bound on the difference with the index's float
/// computation, which quantizes <x_u, q> over [0, mem_dim) with the
/// look-up table.
double reference_estimate(
        const IndexIVFRaBitQSSD& index,
        const float* q,
        const float* o,
        idx_t list_no,
        bool use_hacc,
        double& tol) {
    const size_t d = index.d;
    const size_t nb_bits = index.rabitq.nb_bits;
    const size_t ex_bits = nb_bits - 1;
    std::vector<float> c(d);
    index.quantizer->reconstruct(list_no, c.data());
    faiss::RaBitQuantizer rq(d, index.metric_type, nb_bits);
    std::vector<uint8_t> code(rq.code_size);
    rq.compute_codes_core(o, code.data(), 1, c.data());
    const uint8_t* ex_code = code.data() + d / 8 +
            sizeof(faiss::rabitq_utils::SignBitFactorsWithError);

    double ip_xb_qc = 0, ip_xb_r = 0, r2 = 0, ip_r_c = 0, ip_q_c = 0, qc2 = 0,
           xb_abs_max = 0;
    for (size_t i = 0; i < d; i++) {
        const int bit = (code[i / 8] >> (i % 8)) & 1;
        const int ex =
                faiss::rabitq_utils::extract_code_inline(ex_code, i, ex_bits);
        const double xb = (bit << ex_bits) + ex - ((1 << ex_bits) - 0.5);
        const double r = double(o[i]) - c[i];
        ip_xb_qc += xb * (double(q[i]) - c[i]);
        ip_xb_r += xb * r;
        r2 += r * r;
        ip_r_c += r * c[i];
        ip_q_c += double(q[i]) * c[i];
        qc2 += (double(q[i]) - c[i]) * (double(q[i]) - c[i]);
        xb_abs_max = std::max(xb_abs_max, std::abs(xb));
    }
    const double ip_qc_r = ip_xb_r != 0 ? r2 * ip_xb_qc / ip_xb_r : 0;

    // each look-up table entry is rounded to the nearest step
    const double lut_error =
            index.mem_dim / 4 * lut_step(q, index.mem_dim, use_hacc) / 2;
    const double scale = ip_xb_r != 0 ? r2 / std::abs(ip_xb_r) : 0;
    // float rounding: relative to the magnitude of the terms
    const double magnitude = qc2 + r2 + std::abs(ip_qc_r) + std::abs(ip_q_c) +
            std::abs(ip_r_c) +
            scale * xb_abs_max * std::sqrt(double(d)) *
                    std::sqrt(qc2 + ip_q_c * ip_q_c / (qc2 + 1e-30) + 1);
    const double k = index.metric_type == faiss::METRIC_L2 ? 2 : 1;
    tol = k * scale * (1 << ex_bits) * lut_error + 2e-4 * magnitude;
    if (index.metric_type == faiss::METRIC_L2) {
        return qc2 + r2 - 2 * ip_qc_r;
    } else {
        return ip_q_c + ip_r_c + ip_qc_r;
    }
}

/// Search with k = nb: no vector is pruned, all are re-ranked, and their
/// distances must match the reference estimates.
void check_estimates(const Config& cfg) {
    SCOPED_TRACE(
            "d=" + std::to_string(cfg.d) +
            " mem_dim=" + std::to_string(cfg.mem_dim) +
            " nb_bits=" + std::to_string(cfg.nb_bits) +
            " metric=" + std::to_string(cfg.metric) +
            " quantizer_metric=" + std::to_string(cfg.quantizer_metric) +
            " use_hacc=" + std::to_string(cfg.use_hacc));
    TestIndex ti(cfg);
    const IndexIVFRaBitQSSD& index = *ti.index;
    ASSERT_EQ(index.ntotal, idx_t(cfg.nb));

    std::vector<idx_t> assign(cfg.nb);
    index.quantizer->assign(cfg.nb, ti.xb.data(), assign.data());

    const size_t nq = 4;
    std::vector<float> xq = make_data(nq, cfg.d, 5, 456);
    faiss::IVFRaBitQSSDSearchParameters params;
    params.nprobe = cfg.nlist;
    params.use_hacc = cfg.use_hacc;
    const idx_t k = cfg.nb;
    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    index.search(nq, xq.data(), k, D.data(), I.data(), &params);

    const bool is_ip = cfg.metric == faiss::METRIC_INNER_PRODUCT;
    size_t n_bad = 0;
    for (size_t q = 0; q < nq; q++) {
        std::set<idx_t> seen;
        for (idx_t j = 0; j < k; j++) {
            const idx_t id = I[q * k + j];
            ASSERT_GE(id, 0);
            ASSERT_LT(id, idx_t(cfg.nb));
            ASSERT_TRUE(seen.insert(id).second) << "duplicate id " << id;
            if (j > 0) {
                if (is_ip) {
                    ASSERT_LE(D[q * k + j], D[q * k + j - 1]);
                } else {
                    ASSERT_GE(D[q * k + j], D[q * k + j - 1]);
                }
            }
            double tol;
            double expected = reference_estimate(
                    index,
                    xq.data() + q * cfg.d,
                    ti.xb.data() + id * cfg.d,
                    assign[id],
                    cfg.use_hacc,
                    tol);
            if (std::abs(D[q * k + j] - expected) > tol) {
                if (n_bad++ < 5) {
                    ADD_FAILURE() << "query " << q << " id " << id << ": "
                                  << D[q * k + j] << " != " << expected
                                  << " (tolerance " << tol << ")";
                }
            }
        }
    }
    EXPECT_EQ(n_bad, 0);
}

double recall_at(
        const std::vector<idx_t>& I,
        const std::vector<idx_t>& gt,
        size_t nq,
        size_t k) {
    size_t n_found = 0;
    for (size_t q = 0; q < nq; q++) {
        std::set<idx_t> ref(gt.begin() + q * k, gt.begin() + (q + 1) * k);
        for (size_t j = 0; j < k; j++) {
            n_found += ref.count(I[q * k + j]);
        }
    }
    return double(n_found) / (nq * k);
}

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

/// libaio contexts of the process: each maps its ring as "[aio]"
int count_aio_contexts() {
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) {
        return -1;
    }
    char line[4096];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        n += strstr(line, "[aio]") != nullptr;
    }
    fclose(f);
    return n;
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

/// accepts all ids, and throws after max_calls calls
struct ThrowingSelector : faiss::IDSelector {
    int max_calls;
    mutable std::atomic<int> n_calls{0};

    explicit ThrowingSelector(int max_calls_in) : max_calls(max_calls_in) {}

    bool is_member(idx_t) const override {
        if (++n_calls > max_calls) {
            FAISS_THROW_MSG("ThrowingSelector: too many calls");
        }
        return true;
    }
};

} // namespace

TEST(IVFRaBitQSSD, layout) {
    TempFile file;
    faiss::RaBitQSSDInvertedLists il(4, 192, 64, 3, file.c_str());
    EXPECT_EQ(il.batch_bytes, 32 * 64 / 8 + 32 * 3 * 4);
    // 16 sign bytes, 72 bytes of ex code, 2 factors, id
    EXPECT_EQ(il.record_size, 16 + 72 + 8 + 8);
    EXPECT_EQ(il.page_size, 4096);
    EXPECT_EQ(il.records_per_page, 4096 / 104);

    // a record larger than a page
    TempFile file2;
    faiss::RaBitQSSDInvertedLists il2(4, 4096, 2048, 8, file2.c_str());
    EXPECT_EQ(il2.record_size, 256 + 4096 + 8 + 8);
    EXPECT_EQ(il2.page_size, 8192);
    EXPECT_EQ(il2.records_per_page, 1);

    TempFile file3;
    for (size_t d : {100, 120}) { // d must be a multiple of 64
        EXPECT_THROW(
                faiss::RaBitQSSDInvertedLists(4, d, 64, 1, file3.c_str()),
                faiss::FaissException);
    }
    // mem_dim must be a multiple of 64 in [64, d]
    for (size_t mem_dim : {0, 32, 96, 192}) {
        EXPECT_THROW(
                faiss::RaBitQSSDInvertedLists(
                        4, 128, mem_dim, 1, file3.c_str()),
                faiss::FaissException);
    }
    for (size_t ex_bits : {0, 9}) {
        EXPECT_THROW(
                faiss::RaBitQSSDInvertedLists(
                        4, 128, 64, ex_bits, file3.c_str()),
                faiss::FaissException);
    }
}

TEST(IVFRaBitQSSD, estimates_l2) {
    for (int nb_bits : {2, 5, 9}) {
        for (size_t mem_dim : {64, 128}) {
            Config cfg;
            cfg.nb_bits = nb_bits;
            cfg.mem_dim = mem_dim;
            check_estimates(cfg);
        }
    }
}

TEST(IVFRaBitQSSD, estimates_ip) {
    for (int nb_bits : {2, 4, 9}) {
        for (size_t mem_dim : {64, 128}) {
            Config cfg;
            cfg.metric = faiss::METRIC_INNER_PRODUCT;
            cfg.quantizer_metric = faiss::METRIC_INNER_PRODUCT;
            cfg.nb_bits = nb_bits;
            cfg.mem_dim = mem_dim;
            check_estimates(cfg);
        }
    }
}

// the coarse distances come from a quantizer with the other metric
TEST(IVFRaBitQSSD, estimates_cross_metric) {
    Config cfg;
    cfg.quantizer_metric = faiss::METRIC_INNER_PRODUCT;
    check_estimates(cfg);
    cfg.metric = faiss::METRIC_INNER_PRODUCT;
    cfg.quantizer_metric = faiss::METRIC_L2;
    check_estimates(cfg);
}

TEST(IVFRaBitQSSD, estimates_8bit_lut) {
    Config cfg;
    cfg.use_hacc = false;
    check_estimates(cfg);
    cfg.metric = faiss::METRIC_INNER_PRODUCT;
    cfg.quantizer_metric = faiss::METRIC_INNER_PRODUCT;
    check_estimates(cfg);
}

// several 8-bit scan segments, records larger than a page
TEST(IVFRaBitQSSD, estimates_shapes) {
    Config cfg;
    cfg.nb = 300;
    cfg.nlist = 4;

    cfg.d = 1536;
    cfg.mem_dim = 1280;
    cfg.nb_bits = 3;
    check_estimates(cfg);
    cfg.use_hacc = false;
    check_estimates(cfg);
    cfg.use_hacc = true;

    cfg.d = 4096;
    cfg.mem_dim = 256;
    cfg.nb_bits = 9;
    check_estimates(cfg);

    cfg.d = 192;
    cfg.mem_dim = 192;
    cfg.nb_bits = 2;
    check_estimates(cfg);
}

// The in-memory lower bound decides whether a page is read. A list holds one
// full page, then its farthest vector from the centroid alone on a second
// page. With one read in flight and k = 1, the second page is considered
// after the first is re-ranked, and is read iff the lower bound of that
// vector does not exceed the best distance on the first page.
TEST(IVFRaBitQSSD, bound_decides_reads) {
    for (MetricType metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        const size_t d = 128, mem_dim = 64;
        const int nb_bits = 5;
        TempFile file;
        faiss::IndexFlat quantizer(d, metric);
        std::vector<float> centroid(d, 0.5f);
        quantizer.add(1, centroid.data());
        IndexIVFRaBitQSSD index(
                &quantizer, d, 1, mem_dim, nb_bits, file.c_str(), metric);
        auto* il = index.get_ssd_invlists();
        const size_t nb = il->records_per_page + 1;
        std::vector<float> xb = make_data(nb, d, 1, 321);
        index.add(nb, xb.data());
        ASSERT_EQ(il->list_pages(0), 2);

        // the vector alone on page 1 is the farthest from the centroid
        size_t last = 0;
        double max_dis = -std::numeric_limits<double>::max();
        for (size_t i = 0; i < nb; i++) {
            double dis = metric == faiss::METRIC_L2
                    ? faiss::fvec_L2sqr(xb.data() + i * d, centroid.data(), d)
                    : -faiss::fvec_inner_product(
                              xb.data() + i * d, centroid.data(), d);
            if (dis > max_dis) {
                max_dis = dis;
                last = i;
            }
        }
        ASSERT_EQ(il->get_single_id(0, nb - 1), idx_t(last));

        std::mt19937 rng(5);
        std::normal_distribution<float> normal;
        int n_read = 0, n_pruned = 0;
        for (int trial = 0; trial < 200; trial++) {
            std::vector<float> q(d);
            for (auto& v : q) {
                v = 0.5f + 0.8f * normal(rng);
            }
            // best distance on page 0 (the search minimizes -<q, o> for IP)
            const double sign = metric == faiss::METRIC_L2 ? 1 : -1;
            double distk = std::numeric_limits<double>::infinity();
            double tol = 0;
            for (size_t i = 0; i < nb; i++) {
                if (i == last) {
                    continue;
                }
                double tol_i;
                double dis = sign *
                        reference_estimate(
                                     index,
                                     q.data(),
                                     xb.data() + i * d,
                                     0,
                                     true,
                                     tol_i);
                if (dis < distk) {
                    distk = dis;
                    tol = tol_i;
                }
            }
            // in-memory lower bound of the last vector
            const float* o = xb.data() + last * d;
            faiss::RaBitQuantizer rq(d, metric, nb_bits);
            std::vector<uint8_t> code(rq.code_size);
            rq.compute_codes_core(o, code.data(), 1, centroid.data());
            double ip_x1_qc = 0, ip_x1_r = 0, r2 = 0, qc2 = 0, ip_r_c = 0,
                   ip_q_c = 0;
            for (size_t i = 0; i < d; i++) {
                const double x1 = ((code[i / 8] >> (i % 8)) & 1) - 0.5;
                const double r = double(o[i]) - centroid[i];
                const double qc = double(q[i]) - centroid[i];
                if (i < mem_dim) {
                    ip_x1_qc += x1 * qc;
                }
                ip_x1_r += x1 * r;
                r2 += r * r;
                qc2 += qc * qc;
                ip_r_c += r * centroid[i];
                ip_q_c += double(q[i]) * centroid[i];
            }
            const double ip_qc_r =
                    r2 * (double(d) / mem_dim) * ip_x1_qc / ip_x1_r;
            const double k_metric = metric == faiss::METRIC_L2 ? 2 : 1;
            const double est = metric == faiss::METRIC_L2
                    ? qc2 + r2 - k_metric * ip_qc_r
                    : -(ip_q_c + ip_r_c + ip_qc_r);
            const double low = est -
                    k_metric * 1.9 * std::sqrt(r2 * qc2) /
                            std::sqrt(double(mem_dim));
            // margin for the look-up table and float rounding
            const double margin = 2 * tol + 1e-3 * std::abs(distk);
            if (std::abs(low - distk) < margin) {
                continue;
            }
            const bool expect_read = low <= distk;

            faiss::IVFRaBitQSSDSearchParameters params;
            params.nprobe = 1;
            params.qd_per_thread = 1;
            float D;
            idx_t I;
            faiss::indexIVFRaBitQSSD_stats.reset();
            index.search(1, q.data(), 1, &D, &I, &params);
            EXPECT_EQ(
                    faiss::indexIVFRaBitQSSD_stats.n_pages_read,
                    expect_read ? 2 : 1)
                    << "metric " << metric << " low " << low << " distk "
                    << distk;
            (expect_read ? n_read : n_pruned)++;
        }
        // both cases occur
        EXPECT_GT(n_read, 5);
        EXPECT_GT(n_pruned, 5);
    }
}

// Pruning keeps the results of an exhaustive re-ranking and reads few pages.
TEST(IVFRaBitQSSD, pruning) {
    for (MetricType metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        for (bool use_hacc : {false, true}) {
            Config cfg;
            cfg.nb_bits = 9;
            cfg.metric = metric;
            cfg.quantizer_metric = metric;
            cfg.nb = 5000;
            cfg.nlist = 16;
            cfg.use_hacc = use_hacc;
            TestIndex ti(cfg);

            const size_t nq = 50;
            const idx_t k = 10;
            std::vector<float> xq = make_queries(ti.xb, cfg.d, nq, 789);
            faiss::IVFRaBitQSSDSearchParameters params;
            params.nprobe = cfg.nlist;
            params.use_hacc = use_hacc;

            // exhaustive: with k = nb nothing is pruned
            const idx_t k_all = cfg.nb;
            std::vector<float> D_all(nq * k_all);
            std::vector<idx_t> I_all(nq * k_all);
            ti.index->search(
                    nq, xq.data(), k_all, D_all.data(), I_all.data(), &params);
            std::vector<idx_t> I_ref(nq * k);
            for (size_t q = 0; q < nq; q++) {
                std::copy(
                        I_all.begin() + q * k_all,
                        I_all.begin() + q * k_all + k,
                        I_ref.begin() + q * k);
            }
            size_t total_pages = 0;
            auto* il = ti.index->get_ssd_invlists();
            for (size_t l = 0; l < il->nlist; l++) {
                total_pages += il->list_pages(l);
            }

            std::vector<float> D(nq * k);
            std::vector<idx_t> I(nq * k);
            faiss::indexIVFRaBitQSSD_stats.reset();
            ti.index->search(nq, xq.data(), k, D.data(), I.data(), &params);
            const size_t pages = faiss::indexIVFRaBitQSSD_stats.n_pages_read;

            // the bound holds with high probability: the same results, from
            // a fraction of the pages
            EXPECT_GE(recall_at(I, I_ref, nq, k), 0.99)
                    << "metric " << metric << " use_hacc " << use_hacc;
            EXPECT_LT(pages, nq * total_pages / 4);
            EXPECT_GT(pages, 0);
        }
    }
}

TEST(IVFRaBitQSSD, recall) {
    Config cfg;
    cfg.nb_bits = 9;
    cfg.nb = 5000;
    cfg.nlist = 16;
    TestIndex ti(cfg);

    const size_t nq = 50;
    const idx_t k = 10;
    std::vector<float> xq = make_data(nq, cfg.d, 20, 789);
    faiss::IndexFlatL2 flat(cfg.d);
    flat.add(cfg.nb, ti.xb.data());
    std::vector<float> D_gt(nq * k), D(nq * k);
    std::vector<idx_t> I_gt(nq * k), I(nq * k);
    flat.search(nq, xq.data(), k, D_gt.data(), I_gt.data());

    ti.index->nprobe = cfg.nlist;
    ti.index->search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_GE(recall_at(I, I_gt, nq, k), 0.95);
}

TEST(IVFRaBitQSSD, serialization) {
    Config cfg;
    TestIndex ti(cfg);
    ti.index->nprobe = 4;
    ti.index->drain_alpha = 0.8f;
    ti.index->qd_per_thread = 8;

    // with k = nb nothing is pruned, so the results do not depend on the
    // order in which the reads complete
    const size_t nq = 3;
    const idx_t k = cfg.nb;
    std::vector<float> xq = make_data(nq, cfg.d, 5, 456);
    std::vector<float> D_ref(nq * k), D(nq * k);
    std::vector<idx_t> I_ref(nq * k), I(nq * k);
    ti.index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());

    TempFile index_file(".faissindex");
    faiss::write_index(ti.index.get(), index_file.c_str());

    for (int io_flags : {faiss::IO_FLAG_SKIP_IVF_DATA, faiss::IO_FLAG_MMAP}) {
        EXPECT_THROW(
                delete faiss::read_index(index_file.c_str(), io_flags),
                faiss::FaissException);
    }

    for (int io_flags : {0, faiss::IO_FLAG_READ_ONLY}) {
        std::unique_ptr<faiss::Index> loaded(
                faiss::read_index(index_file.c_str(), io_flags));
        auto* index2 = dynamic_cast<IndexIVFRaBitQSSD*>(loaded.get());
        ASSERT_TRUE(index2);
        EXPECT_EQ(index2->ntotal, ti.index->ntotal);
        EXPECT_EQ(index2->mem_dim, cfg.mem_dim);
        EXPECT_EQ(index2->drain_alpha, 0.8f);
        EXPECT_EQ(index2->qd_per_thread, 8);
        EXPECT_EQ(index2->nprobe, 4);
        index2->search(nq, xq.data(), k, D.data(), I.data());
        // equal distances can come out in either order
        EXPECT_EQ(D, D_ref);
        EXPECT_EQ(n_different_results(D_ref, I_ref, D, I, nq, k), 0);
        if (io_flags == faiss::IO_FLAG_READ_ONLY) {
            EXPECT_THROW(
                    index2->add(cfg.nb, ti.xb.data()), faiss::FaissException);
        }
    }

    // the ids are read back from the file
    auto* il = ti.index->get_ssd_invlists();
    size_t n = 0;
    for (size_t l = 0; l < il->nlist; l++) {
        faiss::InvertedLists::ScopedIds ids(il, l);
        for (size_t j = 0; j < il->list_size(l); j++) {
            EXPECT_EQ(il->get_single_id(l, j), ids[j]);
            n++;
        }
    }
    EXPECT_EQ(n, cfg.nb);

    // resetting and refilling the SSD invalidates the saved index
    ti.index->reset();
    ti.index->add(cfg.nb, ti.xb.data());
    EXPECT_THROW(
            delete faiss::read_index(index_file.c_str()),
            faiss::FaissException);
    faiss::write_index(ti.index.get(), index_file.c_str());

    // a new index on the same file invalidates the old index file
    TestIndex ti2(cfg);
    {
        IndexIVFRaBitQSSD other(
                ti2.quantizer.get(),
                cfg.d,
                cfg.nlist,
                cfg.mem_dim,
                cfg.nb_bits,
                ti.file.c_str());
    }
    EXPECT_THROW(
            delete faiss::read_index(index_file.c_str()),
            faiss::FaissException);
}

TEST(IVFRaBitQSSD, retrain_centroid_norms) {
    Config cfg;
    cfg.quantizer_metric = faiss::METRIC_INNER_PRODUCT;
    TestIndex ti(cfg, 123), fresh(cfg, 456);
    ti.index->reset();
    EXPECT_TRUE(ti.index->centroid_norms.empty());
    ti.quantizer->reset();
    ti.index->train(cfg.nb, fresh.xb.data());
    EXPECT_EQ(ti.index->centroid_norms, fresh.index->centroid_norms);
    ti.index->add(cfg.nb, fresh.xb.data());

    const size_t nq = 3;
    const idx_t k = cfg.nb;
    std::vector<float> xq = make_queries(fresh.xb, cfg.d, nq, 789);
    std::vector<float> D(nq * k), D_ref(nq * k);
    std::vector<idx_t> I(nq * k), I_ref(nq * k);
    ti.index->nprobe = fresh.index->nprobe = cfg.nlist;
    ti.index->search(nq, xq.data(), k, D.data(), I.data());
    fresh.index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());
    EXPECT_EQ(D, D_ref);
    EXPECT_EQ(n_different_results(D_ref, I_ref, D, I, nq, k), 0);
}

TEST(IVFRaBitQSSD, selector_and_store_pairs) {
    Config cfg;
    TestIndex ti(cfg);
    const size_t nq = 10;
    const idx_t k = 20;
    std::vector<float> xq = make_data(nq, cfg.d, 5, 456);
    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);

    faiss::IDSelectorRange sel(100, 300);
    faiss::SearchParametersIVF params;
    params.nprobe = cfg.nlist;
    params.sel = &sel;
    ti.index->search(nq, xq.data(), k, D.data(), I.data(), &params);
    for (idx_t id : I) {
        EXPECT_TRUE(id >= 100 && id < 300) << id;
    }

    // store_pairs: the labels are (list, offset) pairs. The reads complete in
    // an order that varies between runs, so a vector whose bound fails can
    // be pruned in one run and not in another: compare with k = nb, where
    // nothing is pruned.
    const idx_t k_all = cfg.nb;
    std::vector<float> D_ref(nq * k_all), D2(nq * k_all),
            coarse_dis(nq * cfg.nlist);
    std::vector<idx_t> I_ref(nq * k_all), I2(nq * k_all),
            assign(nq * cfg.nlist);
    params.sel = nullptr;
    ti.index->search(nq, xq.data(), k_all, D_ref.data(), I_ref.data(), &params);
    ti.index->quantizer->search(
            nq, xq.data(), cfg.nlist, coarse_dis.data(), assign.data());
    ti.index->search_preassigned(
            nq,
            xq.data(),
            k_all,
            assign.data(),
            coarse_dis.data(),
            D2.data(),
            I2.data(),
            true,
            &params);
    EXPECT_EQ(D2, D_ref);
    auto* il = ti.index->get_ssd_invlists();
    std::vector<idx_t> ids2(nq * k_all, -1);
    for (size_t j = 0; j < nq * k_all; j++) {
        if (I2[j] >= 0) {
            ids2[j] = il->get_single_id(
                    faiss::lo_listno(I2[j]), faiss::lo_offset(I2[j]));
        }
    }
    // equal distances can come out in either order
    EXPECT_EQ(n_different_results(D_ref, I_ref, D2, ids2, nq, k_all), 0);
}

TEST(IVFRaBitQSSD, fewer_results_than_k) {
    for (MetricType metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        Config cfg;
        cfg.nb = 50;
        cfg.nlist = 4;
        cfg.metric = metric;
        cfg.quantizer_metric = metric;
        TestIndex ti(cfg);
        const idx_t k = 80;
        std::vector<float> xq = make_data(1, cfg.d, 1, 456);
        std::vector<float> D(k);
        std::vector<idx_t> I(k);
        ti.index->nprobe = cfg.nlist;
        ti.index->search(1, xq.data(), k, D.data(), I.data());
        for (idx_t j = 0; j < k; j++) {
            if (j < 50) {
                EXPECT_GE(I[j], 0);
            } else {
                EXPECT_EQ(I[j], -1);
                EXPECT_EQ(
                        D[j],
                        metric == faiss::METRIC_L2
                                ? std::numeric_limits<float>::max()
                                : std::numeric_limits<float>::lowest());
            }
        }
    }
}

// adding in several batches gives the same distances as a single add
TEST(IVFRaBitQSSD, incremental_add) {
    Config cfg;
    TestIndex ti(cfg);

    TempFile file;
    faiss::IndexFlatL2 quantizer(cfg.d);
    IndexIVFRaBitQSSD index2(
            &quantizer,
            cfg.d,
            cfg.nlist,
            cfg.mem_dim,
            cfg.nb_bits,
            file.c_str());
    index2.train(cfg.nb, ti.xb.data());
    for (size_t i0 = 0; i0 < cfg.nb; i0 += 70) {
        size_t n = std::min(size_t(70), cfg.nb - i0);
        index2.add(n, ti.xb.data() + i0 * cfg.d);
    }
    EXPECT_EQ(index2.ntotal, idx_t(cfg.nb));
    // several extents per list
    auto* il = index2.get_ssd_invlists();
    size_t max_extents = 0;
    for (const auto& e : il->extents) {
        max_extents = std::max(max_extents, e.size());
    }
    EXPECT_GT(max_extents, 1);

    const idx_t k = cfg.nb;
    std::vector<float> xq = make_data(1, cfg.d, 1, 456);
    std::vector<float> D1(k), D2(k);
    std::vector<idx_t> I1(k), I2(k);
    ti.index->nprobe = cfg.nlist;
    index2.nprobe = cfg.nlist;
    ti.index->search(1, xq.data(), k, D1.data(), I1.data());
    index2.search(1, xq.data(), k, D2.data(), I2.data());
    std::map<idx_t, float> dis1;
    for (idx_t j = 0; j < k; j++) {
        dis1[I1[j]] = D1[j];
    }
    for (idx_t j = 0; j < k; j++) {
        ASSERT_EQ(dis1.count(I2[j]), 1);
        EXPECT_EQ(dis1[I2[j]], D2[j]);
    }

    ti.index->reset();
    EXPECT_EQ(ti.index->ntotal, 0);
    EXPECT_EQ(ti.index->get_ssd_invlists()->total_pages, 0);
    std::vector<idx_t> ids(cfg.nb);
    for (size_t i = 0; i < cfg.nb; i++) {
        ids[i] = 1000 + i;
    }
    ti.index->add_with_ids(cfg.nb, ti.xb.data(), ids.data());
    std::vector<float> D(1);
    std::vector<idx_t> I(1);
    ti.index->search(1, ti.xb.data() + 7 * cfg.d, 1, D.data(), I.data());
    EXPECT_EQ(I[0], 1007);
}

TEST(IVFRaBitQSSD, hnsw_quantizer) {
    Config cfg;
    cfg.nb = 3000;
    cfg.nlist = 32;
    std::vector<float> xb = make_data(cfg.nb, cfg.d, 60, 123);
    faiss::IndexHNSWFlat quantizer(cfg.d, 16);
    TempFile file;
    IndexIVFRaBitQSSD index(
            &quantizer,
            cfg.d,
            cfg.nlist,
            cfg.mem_dim,
            cfg.nb_bits,
            file.c_str());
    index.train(cfg.nb, xb.data());
    index.add(cfg.nb, xb.data());
    index.nprobe = 8;
    const idx_t k = 1;
    std::vector<float> D(20 * k);
    std::vector<idx_t> I(20 * k);
    index.search(20, xb.data(), k, D.data(), I.data());
    int n_found = 0;
    for (int i = 0; i < 20; i++) {
        n_found += I[i] == i;
    }
    EXPECT_GE(n_found, 18);
}

// search() runs the coarse quantizer in the search threads: the same results
// as the coarse quantizer on all the queries, then search_preassigned
TEST(IVFRaBitQSSD, search_matches_preassigned) {
    for (MetricType metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        Config cfg;
        cfg.metric = metric;
        cfg.quantizer_metric = metric;
        TestIndex ti(cfg);
        // the coarse distances of 30 queries are computed with BLAS, those of
        // one query are not: they differ in the last bits
        const size_t nq = 30;
        // with k = nb nothing is pruned, so the results do not depend on the
        // order in which the reads complete
        const idx_t k = cfg.nb;
        std::vector<float> xq = make_queries(ti.xb, cfg.d, nq, 789);
        faiss::IVFRaBitQSSDSearchParameters params;
        params.nprobe = cfg.nlist;
        std::vector<float> D(nq * k), D2(nq * k), coarse_dis(nq * cfg.nlist);
        std::vector<idx_t> I(nq * k), I2(nq * k), assign(nq * cfg.nlist);

        faiss::indexIVF_stats.reset();
        ti.index->search(nq, xq.data(), k, D.data(), I.data(), &params);
        EXPECT_EQ(faiss::indexIVF_stats.nq, nq);
        EXPECT_EQ(faiss::indexIVF_stats.nlist, nq * cfg.nlist);

        ti.index->quantizer->search(
                nq, xq.data(), cfg.nlist, coarse_dis.data(), assign.data());
        ti.index->search_preassigned(
                nq,
                xq.data(),
                k,
                assign.data(),
                coarse_dis.data(),
                D2.data(),
                I2.data(),
                false,
                &params);
        EXPECT_EQ(n_different_results(D, I, D2, I2, nq, k, 1e-5), 0)
                << "metric " << metric;
    }
}

// The libaio contexts of the searches are kept between searches and destroyed
// with the index.
TEST(IVFRaBitQSSD, io_contexts_kept) {
    const int n0 = count_aio_contexts();
    ASSERT_GE(n0, 0);
    {
        Config cfg;
        TestIndex ti(cfg);
        EXPECT_EQ(count_aio_contexts(), n0);
        const size_t nq = 50;
        const idx_t k = 10;
        std::vector<float> xq = make_queries(ti.xb, cfg.d, nq, 789);
        std::vector<float> D(nq * k);
        std::vector<idx_t> I(nq * k);
        ti.index->search(nq, xq.data(), k, D.data(), I.data());
        const int n1 = count_aio_contexts();
        EXPECT_GT(n1, n0);
        EXPECT_LE(n1, n0 + omp_get_max_threads());
        for (int rep = 0; rep < 3; rep++) {
            ti.index->search(nq, xq.data(), k, D.data(), I.data());
            EXPECT_EQ(count_aio_contexts(), n1);
        }
    }
    EXPECT_EQ(count_aio_contexts(), n0);
}

// Searches from several threads at once, on two indexes, give the same
// results as one search at a time.
TEST(IVFRaBitQSSD, concurrent_searches) {
    Config cfg;
    TestIndex ti1(cfg, 123), ti2(cfg, 456);
    IndexIVFRaBitQSSD* indexes[2] = {ti1.index.get(), ti2.index.get()};
    const size_t nq = 10;
    // with k = nb nothing is pruned
    const idx_t k = cfg.nb;
    std::vector<float> xq = make_data(nq, cfg.d, 5, 789);
    std::vector<float> D_ref[2];
    std::vector<idx_t> I_ref[2];
    for (int j = 0; j < 2; j++) {
        indexes[j]->nprobe = 4;
        D_ref[j].resize(nq * k);
        I_ref[j].resize(nq * k);
        indexes[j]->search(nq, xq.data(), k, D_ref[j].data(), I_ref[j].data());
    }

    const int n_threads = 4;
    std::vector<size_t> n_diff(n_threads, 0);
    std::vector<int> n_errors(n_threads, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            omp_set_num_threads(2);
            for (int rep = 0; rep < 4; rep++) {
                const int j = (t + rep) % 2;
                std::vector<float> D(nq * k);
                std::vector<idx_t> I(nq * k);
                try {
                    indexes[j]->search(nq, xq.data(), k, D.data(), I.data());
                } catch (const std::exception&) {
                    n_errors[t]++;
                    continue;
                }
                n_diff[t] +=
                        n_different_results(D_ref[j], I_ref[j], D, I, nq, k);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (int t = 0; t < n_threads; t++) {
        EXPECT_EQ(n_errors[t], 0) << "thread " << t;
        EXPECT_EQ(n_diff[t], 0) << "thread " << t;
    }
}

// A search that fails with reads in flight leaves its libaio contexts ready
// for the next searches.
TEST(IVFRaBitQSSD, exception_during_search) {
    Config cfg;
    cfg.nb = 3000;
    TestIndex ti(cfg);
    const size_t nq = 20;
    // with k = nb nothing is pruned
    const idx_t k = cfg.nb;
    std::vector<float> xq = make_data(nq, cfg.d, 5, 789);
    ti.index->nprobe = cfg.nlist;
    std::vector<float> D_ref(nq * k), D(nq * k);
    std::vector<idx_t> I_ref(nq * k), I(nq * k);
    ti.index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());

    // the selector is called when re-ranking, while other reads are in
    // flight
    ThrowingSelector sel(50);
    faiss::IVFRaBitQSSDSearchParameters params;
    params.nprobe = cfg.nlist;
    params.sel = &sel;
    params.qd_per_thread = 64;
    for (int rep = 0; rep < 3; rep++) {
        sel.n_calls = 0;
        EXPECT_THROW(
                ti.index->search(nq, xq.data(), k, D.data(), I.data(), &params),
                faiss::FaissException);
        ti.index->search(nq, xq.data(), k, D.data(), I.data());
        EXPECT_EQ(n_different_results(D_ref, I_ref, D, I, nq, k), 0);
    }
}

// A process forked after a search cannot use the libaio contexts that its
// parent keeps: it creates its own.
TEST(IVFRaBitQSSD, search_after_fork) {
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

// index_factory creates the index without inverted lists: they are attached
// with replace_invlists()
TEST(IVFRaBitQSSD, index_factory) {
    Config cfg;
    TestIndex ti(cfg);

    const std::string description = "IVF" + std::to_string(cfg.nlist) +
            ",RaBitQSSD" + std::to_string(cfg.nb_bits) + "_" +
            std::to_string(cfg.mem_dim);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(cfg.d, description.c_str(), cfg.metric));
    auto* ivf = dynamic_cast<IndexIVFRaBitQSSD*>(index.get());
    ASSERT_TRUE(ivf);
    EXPECT_EQ(ivf->nlist, cfg.nlist);
    EXPECT_EQ(ivf->mem_dim, cfg.mem_dim);
    EXPECT_EQ(int(ivf->rabitq.nb_bits), cfg.nb_bits);
    EXPECT_EQ(ivf->invlists, nullptr);

    // without inverted lists: training works, adding does not
    ivf->train(cfg.nb, ti.xb.data());
    EXPECT_THROW(ivf->add(cfg.nb, ti.xb.data()), faiss::FaissException);
    ivf->reset();
    EXPECT_EQ(ivf->ntotal, 0);

    // and the index can be written and read back
    TempFile index_file(".faissindex");
    faiss::write_index(ivf, index_file.c_str());
    std::unique_ptr<faiss::Index> loaded(faiss::read_index(index_file.c_str()));
    auto* ivf2 = dynamic_cast<IndexIVFRaBitQSSD*>(loaded.get());
    ASSERT_TRUE(ivf2);
    EXPECT_TRUE(ivf2->is_trained);
    EXPECT_EQ(ivf2->invlists, nullptr);
    EXPECT_EQ(ivf2->mem_dim, cfg.mem_dim);

    // with inverted lists attached, the same results as an index created
    // with a file
    TempFile file;
    ivf2->replace_invlists(
            new faiss::RaBitQSSDInvertedLists(
                    cfg.nlist,
                    cfg.d,
                    cfg.mem_dim,
                    cfg.nb_bits - 1,
                    file.c_str()),
            true);
    ivf2->add(cfg.nb, ti.xb.data());
    const size_t nq = 10;
    // with k = nb nothing is pruned
    const idx_t k = cfg.nb;
    std::vector<float> xq = make_queries(ti.xb, cfg.d, nq, 789);
    std::vector<float> D(nq * k), D_ref(nq * k);
    std::vector<idx_t> I(nq * k), I_ref(nq * k);
    ti.index->nprobe = cfg.nlist;
    ivf2->nprobe = cfg.nlist;
    ti.index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());
    ivf2->search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_EQ(n_different_results(D_ref, I_ref, D, I, nq, k), 0);

    // nb_bits = 1, and mem_dim not a multiple of 64
    for (const char* bad : {"IVF8,RaBitQSSD1_64", "IVF8,RaBitQSSD5_100"}) {
        EXPECT_THROW(
                delete faiss::index_factory(cfg.d, bad), faiss::FaissException)
                << bad;
    }
}

TEST(IVFRaBitQSSD, unsupported) {
    Config cfg;
    cfg.nb = 100;
    TestIndex ti(cfg);
    IndexIVFRaBitQSSD& index = *ti.index;
    faiss::IDSelectorRange sel(0, 10);
    EXPECT_THROW(index.remove_ids(sel), faiss::FaissException);
    std::vector<idx_t> ids = {1};
    EXPECT_THROW(
            index.update_vectors(1, ids.data(), ti.xb.data()),
            faiss::FaissException);
    // both reconstruction entry points reject the operation
    std::vector<float> recons(2 * cfg.d);
    for (int i = 0; i < 2; i++) {
        try {
            if (i == 0) {
                index.reconstruct(0, recons.data());
            } else {
                index.reconstruct_n(0, 2, recons.data());
            }
            ADD_FAILURE() << "reconstruct did not throw";
        } catch (const faiss::FaissException& e) {
            EXPECT_NE(
                    std::string(e.what()).find("reconstruct not supported"),
                    std::string::npos)
                    << e.what();
        }
    }
    std::vector<uint8_t> codes(index.sa_code_size());
    EXPECT_THROW(
            index.sa_encode(1, ti.xb.data(), codes.data()),
            faiss::FaissException);
    faiss::RangeSearchResult res(1);
    faiss::indexIVFRaBitQSSD_stats.reset();
    EXPECT_THROW(
            index.range_search(1, ti.xb.data(), 1.0, &res),
            faiss::FaissException);
    std::vector<float> D(1);
    std::vector<idx_t> I(1);
    EXPECT_THROW(
            index.search_and_reconstruct(
                    1, ti.xb.data(), 1, D.data(), I.data(), recons.data()),
            faiss::FaissException);
    EXPECT_THROW(
            index.search_and_return_codes(
                    1, ti.xb.data(), 1, D.data(), I.data(), codes.data()),
            faiss::FaissException);
    // all three throw before searching
    EXPECT_EQ(faiss::indexIVFRaBitQSSD_stats.n_pages_read, 0);
    EXPECT_EQ(faiss::indexIVFRaBitQSSD_stats.n_reranked, 0);
    faiss::SearchParametersIVF params;
    params.max_codes = 100;
    EXPECT_THROW(
            index.search(1, ti.xb.data(), 1, D.data(), I.data(), &params),
            faiss::FaissException);
    params.max_codes = 0;
    params.max_lists_num = 1;
    EXPECT_THROW(
            index.search(1, ti.xb.data(), 1, D.data(), I.data(), &params),
            faiss::FaissException);
    params.max_lists_num = 0;
    params.ensure_topk_full = true;
    EXPECT_THROW(
            index.search(1, ti.xb.data(), 1, D.data(), I.data(), &params),
            faiss::FaissException);
    TestIndex ti2(cfg);
    EXPECT_THROW(index.merge_from(*ti2.index, 0), faiss::FaissException);

    // nb_bits = 1: no extra bits to refine the estimate with
    TempFile file;
    faiss::IndexFlatL2 quantizer(cfg.d);
    for (size_t mem_dim : {64, 128}) {
        EXPECT_THROW(
                IndexIVFRaBitQSSD(
                        &quantizer, cfg.d, 4, mem_dim, 1, file.c_str()),
                faiss::FaissException);
    }
}
