/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sys/stat.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace benchmarks {

/// Read a .fvecs file. Returns data in row-major float array.
inline std::vector<float> fvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "benchmarks: could not open %s\n", fname);
        *d_out = 0;
        *n_out = 0;
        return {};
    }
    int d;
    size_t nr = fread(&d, sizeof(int), 1, f);
    if (nr != 1 || d <= 0 || d > 1000000) {
        fprintf(stderr,
                "benchmarks: unreasonable dimension %d in %s\n",
                d,
                fname);
        fclose(f);
        *d_out = 0;
        *n_out = 0;
        return {};
    }
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t row_size = ((size_t)d + 1) * sizeof(float);
    if (sz % row_size != 0) {
        fprintf(stderr, "benchmarks: weird file size for %s\n", fname);
        fclose(f);
        *d_out = 0;
        *n_out = 0;
        return {};
    }
    size_t n = sz / row_size;

    *d_out = d;
    *n_out = n;

    // Read raw data including row headers
    std::vector<float> raw(n * (d + 1));
    nr = fread(raw.data(), sizeof(float), n * (d + 1), f);
    fclose(f);
    if (nr != n * (d + 1)) {
        fprintf(stderr, "benchmarks: short read on %s\n", fname);
        *d_out = 0;
        *n_out = 0;
        return {};
    }

    // Strip row headers (first int in each row is the dimension)
    std::vector<float> result(n * d);
    for (size_t i = 0; i < n; i++) {
        memcpy(result.data() + i * d,
               raw.data() + i * (d + 1) + 1,
               d * sizeof(float));
    }
    return result;
}

/// Read a .ivecs file. Returns data in row-major int32 array.
inline std::vector<int32_t> ivecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out) {
    // ivecs has the same format as fvecs (int32 instead of float32)
    auto raw = fvecs_read(fname, d_out, n_out);
    std::vector<int32_t> result(raw.size());
    memcpy(result.data(), raw.data(), raw.size() * sizeof(float));
    return result;
}

/// Read a .bvecs file (BIGANN base/query/learn vectors). The framing is the
/// same as .?vecs — each row is an int32 dimension header followed by `d`
/// values — but the vector body is stored as uint8 rather than float32. The
/// returned array is row-major float32 (each byte widened to float), so the
/// rest of the harness can treat BIGANN like any other float dataset.
inline std::vector<float> bvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "benchmarks: could not open %s\n", fname);
        *d_out = 0;
        *n_out = 0;
        return {};
    }
    int d;
    size_t nr = fread(&d, sizeof(int), 1, f);
    if (nr != 1 || d <= 0 || d > 1000000) {
        fprintf(stderr,
                "benchmarks: unreasonable dimension %d in %s\n",
                d,
                fname);
        fclose(f);
        *d_out = 0;
        *n_out = 0;
        return {};
    }
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    // Each row: 4-byte int header + d uint8 values.
    size_t row_size = sizeof(int) + (size_t)d;
    if (sz % row_size != 0) {
        fprintf(stderr, "benchmarks: weird file size for %s\n", fname);
        fclose(f);
        *d_out = 0;
        *n_out = 0;
        return {};
    }
    size_t n = sz / row_size;
    *d_out = d;
    *n_out = n;

    std::vector<uint8_t> raw(n * row_size);
    nr = fread(raw.data(), 1, n * row_size, f);
    fclose(f);
    if (nr != n * row_size) {
        fprintf(stderr, "benchmarks: short read on %s\n", fname);
        *d_out = 0;
        *n_out = 0;
        return {};
    }

    // Strip the per-row int header and widen each uint8 to float32.
    std::vector<float> result(n * (size_t)d);
    for (size_t i = 0; i < n; i++) {
        const uint8_t* row = raw.data() + i * row_size + sizeof(int);
        float* out = result.data() + i * (size_t)d;
        for (int j = 0; j < d; j++) {
            out[j] = (float)row[j];
        }
    }
    return result;
}

/// Read a vector file, picking the reader from the filename extension:
/// `.bvecs` is uint8 framing (widened to float32), anything else is treated
/// as `.fvecs` float32 framing. Lets the same loader serve SIFT-style float
/// datasets and BIGANN-style byte datasets transparently.
inline std::vector<float> vecs_read_auto(
        const char* fname,
        size_t* d_out,
        size_t* n_out) {
    std::string s(fname);
    if (s.size() >= 6 && s.compare(s.size() - 6, 6, ".bvecs") == 0) {
        return bvecs_read(fname, d_out, n_out);
    }
    return fvecs_read(fname, d_out, n_out);
}

/// A {train, base, query, groundtruth} dataset loaded from four files in a
/// directory. Defaults to the SIFT1M layout (128-dim, 1M base, 10K queries,
/// 100K train) but every filename is overridable, so any dataset that ships
/// the same .fvecs/.bvecs + .ivecs framing (cohere, deep, gist, bigann, ...)
/// can be loaded by naming its files.
struct DatasetSIFT1M {
    size_t d = 0;
    size_t nb = 0;
    size_t nq = 0;
    size_t nt = 0;
    size_t gt_k = 0;

    std::vector<float> xb;   // base vectors [nb x d]
    std::vector<float> xq;   // query vectors [nq x d]
    std::vector<float> xt;   // train vectors [nt x d]
    std::vector<int32_t> gt; // ground truth [nq x gt_k]

    bool loaded = false;

    /// Load from a directory containing four dataset files. The filenames
    /// default to the standard SIFT1M layout but can each be overridden to
    /// load any dataset with the same framing (train/base/query as
    /// .fvecs/.bvecs, ground truth as .ivecs). The train/base/query reader is
    /// chosen per-file from the extension (see vecs_read_auto); ground truth
    /// is always .ivecs. Returns true on success.
    bool load(
            const std::string& dir = "sift1M",
            const std::string& train_file = "sift_learn.fvecs",
            const std::string& base_file = "sift_base.fvecs",
            const std::string& query_file = "sift_query.fvecs",
            const std::string& gt_file = "sift_groundtruth.ivecs") {
        size_t d1, d2, d3;

        xt = vecs_read_auto((dir + "/" + train_file).c_str(), &d, &nt);
        if (nt == 0)
            return false;

        xb = vecs_read_auto((dir + "/" + base_file).c_str(), &d1, &nb);
        if (nb == 0 || d1 != d)
            return false;

        xq = vecs_read_auto((dir + "/" + query_file).c_str(), &d2, &nq);
        if (nq == 0 || d2 != d)
            return false;

        // Read the GT row count into a separate variable: it must match the
        // query count, not overwrite it.
        size_t gt_n;
        gt = ivecs_read((dir + "/" + gt_file).c_str(), &d3, &gt_n);
        if (gt_n != nq)
            return false;
        gt_k = d3;

        loaded = true;
        return true;
    }
};

/// Generic {train, base, query, groundtruth} dataset loaded from a
/// directory by name. This supports on-disk .?vecs-framed datasets:
/// sift1M, bigann*, deep*, glove and music-100. If the expected
/// files are absent the loader returns false and callers fall back to
/// synthetic data (exactly as bench_index_ivf_factory.cpp already does for
/// SIFT1M).
///
/// Filename conventions under <data_dir>:
///   sift1M / sift  : sift_learn.fvecs, sift_base.fvecs, sift_query.fvecs,
///                    sift_groundtruth.ivecs           (float32 vectors)
///   bigann         : bigann_learn.bvecs, bigann_base.bvecs,
///                    bigann_query.bvecs, gnd/idx_*.ivecs (uint8 vectors)
///                    -> here we look for bigann_groundtruth.ivecs
///   deep1M / deep* : <name>_learn.fvecs, <name>_base.fvecs,
///                    <name>_query.fvecs, <name>_groundtruth.ivecs
///   gist           : gist_learn.fvecs, gist_base.fvecs, gist_query.fvecs,
///                    gist_groundtruth.ivecs
/// Any dataset whose name starts with "bigann" is read as .bvecs (uint8),
/// everything else as .fvecs (float32).
struct GenericDataset {
    size_t d = 0;
    size_t nb = 0;
    size_t nq = 0;
    size_t nt = 0;
    size_t gt_k = 0;

    std::vector<float> xb;   // base vectors [nb x d]
    std::vector<float> xq;   // query vectors [nq x d]
    std::vector<float> xt;   // train vectors [nt x d]
    std::vector<int32_t> gt; // ground truth [nq x gt_k]

    bool loaded = false;

    /// Best-effort load. `dir` is the data directory, `name` the dataset
    /// name (e.g. "sift1M", "bigann", "deep1M", "gist"). Returns true only
    /// if all four component files loaded consistently.
    bool load(const std::string& dir, const std::string& name) {
        bool is_bvecs = name.rfind("bigann", 0) == 0;
        // Sift files are prefixed "sift"; keep that convention while
        // letting other datasets use their own name as the file prefix.
        std::string prefix =
                (name == "sift1M" || name == "sift") ? "sift" : name;
        std::string ext = is_bvecs ? ".bvecs" : ".fvecs";
        auto vread = is_bvecs ? &bvecs_read : &fvecs_read;

        size_t d1, d2, d3;
        xt = vread((dir + "/" + prefix + "_learn" + ext).c_str(), &d, &nt);
        if (nt == 0)
            return false;
        xb = vread((dir + "/" + prefix + "_base" + ext).c_str(), &d1, &nb);
        if (nb == 0 || d1 != d)
            return false;
        xq = vread((dir + "/" + prefix + "_query" + ext).c_str(), &d2, &nq);
        if (nq == 0 || d2 != d)
            return false;
        // Read the GT row count into a separate variable: it must match the
        // query count, not overwrite it.
        size_t gt_n;
        gt = ivecs_read(
                (dir + "/" + prefix + "_groundtruth.ivecs").c_str(),
                &d3,
                &gt_n);
        if (gt_n != nq)
            return false;
        gt_k = d3;
        loaded = true;
        return true;
    }
};

/// Derive a short dataset tag from a filename, e.g.
/// "cohere1M_base.fvecs" -> "cohere1M". Strips any directory prefix, the
/// vector extension, and a trailing _base/_learn/_query/_train suffix. Used
/// to label benchmarks loaded via user-specified filenames.
inline std::string dataset_tag_from_file(const std::string& file) {
    std::string s = file;
    size_t slash = s.find_last_of('/');
    if (slash != std::string::npos) {
        s = s.substr(slash + 1);
    }
    size_t dot = s.find_last_of('.');
    if (dot != std::string::npos) {
        s = s.substr(0, dot);
    }
    for (const char* suffix : {"_base", "_learn", "_query", "_train"}) {
        size_t len = strlen(suffix);
        if (s.size() > len && s.compare(s.size() - len, len, suffix) == 0) {
            s = s.substr(0, s.size() - len);
            break;
        }
    }
    return s.empty() ? std::string("dataset") : s;
}

/// Benchmark-name label for a dataset loaded through DatasetSIFT1M. Preserves
/// the historical "sift1m" label when the default SIFT base filename is in use
/// (so existing benchmark names stay stable for regression tracking), and
/// otherwise derives a tag from the user-specified base filename.
inline std::string dataset_label(const std::string& base_file) {
    if (base_file == "sift_base.fvecs") {
        return "sift1m";
    }
    return dataset_tag_from_file(base_file);
}

/// Check if a dataset directory exists and is readable
inline bool dataset_available(const std::string& dir) {
    struct stat st;
    return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

/// Compute recall@k: average fraction of true k-NN found in search results.
/// @param labels  search result IDs [nq x k]
/// @param gt      ground truth IDs [nq x gt_k]
/// @param nq      number of queries
/// @param k       number of results per query
/// @param gt_k    number of ground truth neighbors per query
/// @return recall in [0, 1]
inline double compute_recall_at(
        const int64_t* labels,
        const int32_t* gt,
        size_t nq,
        int k,
        size_t gt_k) {
    int eval_k = (size_t)k < gt_k ? k : (int)gt_k;
    size_t n_found = 0;
    for (size_t i = 0; i < nq; i++) {
        for (int j = 0; j < eval_k; j++) {
            int32_t gt_id = gt[i * gt_k + j];
            for (int l = 0; l < k; l++) {
                if (labels[i * k + l] == gt_id) {
                    n_found++;
                    break;
                }
            }
        }
    }
    return (double)n_found / (nq * eval_k);
}

} // namespace benchmarks
