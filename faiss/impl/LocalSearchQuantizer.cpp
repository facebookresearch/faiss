/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/LocalSearchQuantizer.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

#include <algorithm>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h> // BitstringWriter
#include <faiss/utils/utils.h>

extern "C" {
// LU decomoposition of a general matrix
void sgetrf_(
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        FINTEGER* ipiv,
        FINTEGER* info);

// generate inverse of a matrix given its LU decomposition
void sgetri_(
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        FINTEGER* ipiv,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

// general matrix multiplication
int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);

// LU decomoposition of a general matrix
void dgetrf_(
        FINTEGER* m,
        FINTEGER* n,
        double* a,
        FINTEGER* lda,
        FINTEGER* ipiv,
        FINTEGER* info);

// generate inverse of a matrix given its LU decomposition
void dgetri_(
        FINTEGER* n,
        double* a,
        FINTEGER* lda,
        FINTEGER* ipiv,
        double* work,
        FINTEGER* lwork,
        FINTEGER* info);

// general matrix multiplication
int dgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const double* alpha,
        const double* a,
        FINTEGER* lda,
        const double* b,
        FINTEGER* ldb,
        double* beta,
        double* c,
        FINTEGER* ldc);
}

namespace {

void fmat_inverse(float* a, int n) {
    int info;
    int lwork = n * n;
    std::vector<int> ipiv(n);
    std::vector<float> workspace(lwork);

    sgetrf_(&n, &n, a, &n, ipiv.data(), &info);
    FAISS_THROW_IF_NOT(info == 0);
    sgetri_(&n, a, &n, ipiv.data(), workspace.data(), &lwork, &info);
    FAISS_THROW_IF_NOT(info == 0);
}

// c and a and b can overlap
void dfvec_add(size_t d, const double* a, const float* b, double* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

void dmat_inverse(double* a, int n) {
    int info;
    int lwork = n * n;
    std::vector<int> ipiv(n);
    std::vector<double> workspace(lwork);

    dgetrf_(&n, &n, a, &n, ipiv.data(), &info);
    FAISS_THROW_IF_NOT(info == 0);
    dgetri_(&n, a, &n, ipiv.data(), workspace.data(), &lwork, &info);
    FAISS_THROW_IF_NOT(info == 0);
}

void random_int32(
        std::vector<int32_t>& x,
        int32_t min,
        int32_t max,
        std::mt19937& gen) {
    std::uniform_int_distribution<int32_t> distrib(min, max);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = distrib(gen);
    }
}

} // anonymous namespace

namespace faiss {

lsq::LSQTimer lsq_timer;
using lsq::LSQTimerScope;

LocalSearchQuantizer::LocalSearchQuantizer(
        size_t d,
        size_t M,
        size_t nbits,
        Search_type_t search_type)
        : AdditiveQuantizer(d, std::vector<size_t>(M, nbits), search_type) {
    K = (1 << nbits);

    train_iters = 25;
    train_ils_iters = 8;
    icm_iters = 4;

    encode_ils_iters = 16;

    p = 0.5f;
    lambd = 1e-2f;

    chunk_size = 10000;
    nperts = 4;

    random_seed = 0x12345;
    std::srand(random_seed);

    icm_encoder_factory = nullptr;
}

LocalSearchQuantizer::~LocalSearchQuantizer() {
    delete icm_encoder_factory;
}

LocalSearchQuantizer::LocalSearchQuantizer() : LocalSearchQuantizer(0, 0, 0) {}

void LocalSearchQuantizer::train(size_t n, const float* x) {
    FAISS_THROW_IF_NOT(K == (1 << nbits[0]));
    nperts = std::min(nperts, M);

    lsq_timer.reset();
    LSQTimerScope scope(&lsq_timer, "train");
    if (verbose) {
        printf("Training LSQ, with %zd subcodes on %zd %zdD vectors\n",
               M,
               n,
               d);
    }

    // allocate memory for codebooks, size [M, K, d]
    codebooks.resize(M * K * d);

    // randomly intialize codes
    std::mt19937 gen(random_seed);
    std::vector<int32_t> codes(n * M); // [n, M]
    random_int32(codes, 0, K - 1, gen);

    // compute standard derivations of each dimension
    std::vector<float> stddev(d, 0);

#pragma omp parallel for
    for (int64_t i = 0; i < d; i++) {
        float mean = 0;
        for (size_t j = 0; j < n; j++) {
            mean += x[j * d + i];
        }
        mean = mean / n;

        float sum = 0;
        for (size_t j = 0; j < n; j++) {
            float xi = x[j * d + i] - mean;
            sum += xi * xi;
        }
        stddev[i] = sqrtf(sum / n);
    }

    if (verbose) {
        float obj = evaluate(codes.data(), x, n);
        printf("Before training: obj = %lf\n", obj);
    }

    for (size_t i = 0; i < train_iters; i++) {
        // 1. update codebooks given x and codes
        // 2. add perturbation to codebooks (SR-D)
        // 3. refine codes given x and codebooks using icm

        // update codebooks
        update_codebooks(x, codes.data(), n);

        if (verbose) {
            float obj = evaluate(codes.data(), x, n);
            printf("iter %zd:\n", i);
            printf("\tafter updating codebooks: obj = %lf\n", obj);
        }

        // SR-D: perturb codebooks
        float T = pow((1.0f - (i + 1.0f) / train_iters), p);
        perturb_codebooks(T, stddev, gen);

        if (verbose) {
            float obj = evaluate(codes.data(), x, n);
            printf("\tafter perturbing codebooks: obj = %lf\n", obj);
        }

        // refine codes
        icm_encode(codes.data(), x, n, train_ils_iters, gen);

        if (verbose) {
            float obj = evaluate(codes.data(), x, n);
            printf("\tafter updating codes: obj = %lf\n", obj);
        }
    }

    is_trained = true;
    {
        std::vector<float> x_recons(n * d);
        std::vector<float> norms(n);
        decode_unpacked(codes.data(), x_recons.data(), n);
        fvec_norms_L2sqr(norms.data(), x_recons.data(), d, n);

        train_norm(n, norms.data());
    }

    if (verbose) {
        float obj = evaluate(codes.data(), x, n);
        scope.finish();
        printf("After training: obj = %lf\n", obj);

        printf("Time statistic:\n");
        for (const auto& it : lsq_timer.t) {
            printf("\t%s time: %lf s\n", it.first.data(), it.second / 1000);
        }
    }
}

void LocalSearchQuantizer::perturb_codebooks(
        float T,
        const std::vector<float>& stddev,
        std::mt19937& gen) {
    LSQTimerScope scope(&lsq_timer, "perturb_codebooks");

    std::vector<std::normal_distribution<float>> distribs;
    for (size_t i = 0; i < d; i++) {
        distribs.emplace_back(0.0f, stddev[i]);
    }

    for (size_t m = 0; m < M; m++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t i = 0; i < d; i++) {
                codebooks[m * K * d + k * d + i] += T * distribs[i](gen) / M;
            }
        }
    }
}

void LocalSearchQuantizer::compute_codes_add_centroids(
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "LSQ is not trained yet.");

    lsq_timer.reset();
    LSQTimerScope scope(&lsq_timer, "encode");
    if (verbose) {
        printf("Encoding %zd vectors...\n", n);
    }

    std::vector<int32_t> codes(n * M);
    std::mt19937 gen(random_seed);
    random_int32(codes, 0, K - 1, gen);

    icm_encode(codes.data(), x, n, encode_ils_iters, gen);
    pack_codes(n, codes.data(), codes_out, -1, nullptr, centroids);

    if (verbose) {
        scope.finish();
        printf("Time statistic:\n");
        for (const auto& it : lsq_timer.t) {
            printf("\t%s time: %lf s\n", it.first.data(), it.second / 1000);
        }
    }
}

/** update codebooks given x and codes
 *
 * Let B denote the sparse matrix of codes, size [n, M * K].
 * Let C denote the codebooks, size [M * K, d].
 * Let X denote the training vectors, size [n, d]
 *
 * objective function:
 *     L = (X - BC)^2
 *
 * To minimize L, we have:
 *     C = (B'B)^(-1)B'X
 * where ' denote transposed
 *
 * Add a regularization term to make B'B inversible:
 *     C = (B'B + lambd * I)^(-1)B'X
 */
void LocalSearchQuantizer::update_codebooks(
        const float* x,
        const int32_t* codes,
        size_t n) {
    LSQTimerScope scope(&lsq_timer, "update_codebooks");

    if (!update_codebooks_with_double) {
        // allocate memory
        // bb = B'B, bx = BX
        std::vector<float> bb(M * K * M * K, 0.0f); // [M * K, M * K]
        std::vector<float> bx(M * K * d, 0.0f);     // [M * K, d]

        // compute B'B
        for (size_t i = 0; i < n; i++) {
            for (size_t m = 0; m < M; m++) {
                int32_t code1 = codes[i * M + m];
                int32_t idx1 = m * K + code1;
                bb[idx1 * M * K + idx1] += 1;

                for (size_t m2 = m + 1; m2 < M; m2++) {
                    int32_t code2 = codes[i * M + m2];
                    int32_t idx2 = m2 * K + code2;
                    bb[idx1 * M * K + idx2] += 1;
                    bb[idx2 * M * K + idx1] += 1;
                }
            }
        }

        // add a regularization term to B'B
        for (int64_t i = 0; i < M * K; i++) {
            bb[i * (M * K) + i] += lambd;
        }

        // compute (B'B)^(-1)
        fmat_inverse(bb.data(), M * K); // [M*K, M*K]

        // compute BX
        for (size_t i = 0; i < n; i++) {
            for (size_t m = 0; m < M; m++) {
                int32_t code = codes[i * M + m];
                float* data = bx.data() + (m * K + code) * d;
                fvec_add(d, data, x + i * d, data);
            }
        }

        // compute C = (B'B)^(-1) @ BX
        //
        // NOTE: LAPACK use column major order
        // out = alpha * op(A) * op(B) + beta * C
        FINTEGER nrows_A = d;
        FINTEGER ncols_A = M * K;

        FINTEGER nrows_B = M * K;
        FINTEGER ncols_B = M * K;

        float alpha = 1.0f;
        float beta = 0.0f;
        sgemm_("Not Transposed",
               "Not Transposed",
               &nrows_A, // nrows of op(A)
               &ncols_B, // ncols of op(B)
               &ncols_A, // ncols of op(A)
               &alpha,
               bx.data(),
               &nrows_A, // nrows of A
               bb.data(),
               &nrows_B, // nrows of B
               &beta,
               codebooks.data(),
               &nrows_A); // nrows of output

    } else {
        // allocate memory
        // bb = B'B, bx = BX
        std::vector<double> bb(M * K * M * K, 0.0f); // [M * K, M * K]
        std::vector<double> bx(M * K * d, 0.0f);     // [M * K, d]

        // compute B'B
        for (size_t i = 0; i < n; i++) {
            for (size_t m = 0; m < M; m++) {
                int32_t code1 = codes[i * M + m];
                int32_t idx1 = m * K + code1;
                bb[idx1 * M * K + idx1] += 1;

                for (size_t m2 = m + 1; m2 < M; m2++) {
                    int32_t code2 = codes[i * M + m2];
                    int32_t idx2 = m2 * K + code2;
                    bb[idx1 * M * K + idx2] += 1;
                    bb[idx2 * M * K + idx1] += 1;
                }
            }
        }

        // add a regularization term to B'B
        for (int64_t i = 0; i < M * K; i++) {
            bb[i * (M * K) + i] += lambd;
        }

        // compute (B'B)^(-1)
        dmat_inverse(bb.data(), M * K); // [M*K, M*K]

        // compute BX
        for (size_t i = 0; i < n; i++) {
            for (size_t m = 0; m < M; m++) {
                int32_t code = codes[i * M + m];
                double* data = bx.data() + (m * K + code) * d;
                dfvec_add(d, data, x + i * d, data);
            }
        }

        // compute C = (B'B)^(-1) @ BX
        //
        // NOTE: LAPACK use column major order
        // out = alpha * op(A) * op(B) + beta * C
        FINTEGER nrows_A = d;
        FINTEGER ncols_A = M * K;

        FINTEGER nrows_B = M * K;
        FINTEGER ncols_B = M * K;

        std::vector<double> d_codebooks(M * K * d);

        double alpha = 1.0f;
        double beta = 0.0f;
        dgemm_("Not Transposed",
               "Not Transposed",
               &nrows_A, // nrows of op(A)
               &ncols_B, // ncols of op(B)
               &ncols_A, // ncols of op(A)
               &alpha,
               bx.data(),
               &nrows_A, // nrows of A
               bb.data(),
               &nrows_B, // nrows of B
               &beta,
               d_codebooks.data(),
               &nrows_A); // nrows of output

        for (size_t i = 0; i < M * K * d; i++) {
            codebooks[i] = (float)d_codebooks[i];
        }
    }
}

/** encode using iterative conditional mode
 *
 * iterative conditional mode:
 *     For every subcode ci (i = 1, ..., M) of a vector, we fix the other
 *     subcodes cj (j != i) and then find the optimal value of ci such
 *     that minimizing the objective function.

 * objective function:
 *     L = (X - \sum cj)^2, j = 1, ..., M
 *     L = X^2 - 2X * \sum cj + (\sum cj)^2
 *
 * X^2 is negligable since it is the same for all possible value
 * k of the m-th subcode.
 *
 * 2X * \sum cj is the unary term
 * (\sum cj)^2 is the binary term
 * These two terms can be precomputed and store in a look up table.
 */
void LocalSearchQuantizer::icm_encode(
        int32_t* codes,
        const float* x,
        size_t n,
        size_t ils_iters,
        std::mt19937& gen) const {
    LSQTimerScope scope(&lsq_timer, "icm_encode");

    auto factory = icm_encoder_factory;
    std::unique_ptr<lsq::IcmEncoder> icm_encoder;
    if (factory == nullptr) {
        icm_encoder.reset(lsq::IcmEncoderFactory().get(this));
    } else {
        icm_encoder.reset(factory->get(this));
    }

    // precompute binary terms for all chunks
    icm_encoder->set_binary_term();

    const size_t n_chunks = (n + chunk_size - 1) / chunk_size;
    for (size_t i = 0; i < n_chunks; i++) {
        size_t ni = std::min(chunk_size, n - i * chunk_size);

        if (verbose) {
            printf("\r\ticm encoding %zd/%zd ...", i * chunk_size + ni, n);
            fflush(stdout);
            if (i == n_chunks - 1 || i == 0) {
                printf("\n");
            }
        }

        const float* xi = x + i * chunk_size * d;
        int32_t* codesi = codes + i * chunk_size * M;
        icm_encoder->verbose = (verbose && i == 0);
        icm_encoder->encode(codesi, xi, gen, ni, ils_iters);
    }
}

void LocalSearchQuantizer::icm_encode_impl(
        int32_t* codes,
        const float* x,
        const float* binaries,
        std::mt19937& gen,
        size_t n,
        size_t ils_iters,
        bool verbose) const {
    std::vector<float> unaries(n * M * K); // [M, n, K]
    compute_unary_terms(x, unaries.data(), n);

    std::vector<int32_t> best_codes;
    best_codes.assign(codes, codes + n * M);

    std::vector<float> best_objs(n, 0.0f);
    evaluate(codes, x, n, best_objs.data());

    FAISS_THROW_IF_NOT(nperts <= M);
    for (size_t iter1 = 0; iter1 < ils_iters; iter1++) {
        // add perturbation to codes
        perturb_codes(codes, n, gen);

        icm_encode_step(codes, unaries.data(), binaries, n, icm_iters);

        std::vector<float> icm_objs(n, 0.0f);
        evaluate(codes, x, n, icm_objs.data());
        size_t n_betters = 0;
        float mean_obj = 0.0f;

        // select the best code for every vector xi
#pragma omp parallel for reduction(+ : n_betters, mean_obj)
        for (int64_t i = 0; i < n; i++) {
            if (icm_objs[i] < best_objs[i]) {
                best_objs[i] = icm_objs[i];
                memcpy(best_codes.data() + i * M,
                       codes + i * M,
                       sizeof(int32_t) * M);
                n_betters += 1;
            }
            mean_obj += best_objs[i];
        }
        mean_obj /= n;

        memcpy(codes, best_codes.data(), sizeof(int32_t) * n * M);

        if (verbose) {
            printf("\tils_iter %zd: obj = %lf, n_betters/n = %zd/%zd\n",
                   iter1,
                   mean_obj,
                   n_betters,
                   n);
        }
    } // loop ils_iters
}

void LocalSearchQuantizer::icm_encode_step(
        int32_t* codes,
        const float* unaries,
        const float* binaries,
        size_t n,
        size_t n_iters) const {
    FAISS_THROW_IF_NOT(M != 0 && K != 0);
    FAISS_THROW_IF_NOT(binaries != nullptr);

    for (size_t iter = 0; iter < n_iters; iter++) {
        // condition on the m-th subcode
        for (size_t m = 0; m < M; m++) {
            std::vector<float> objs(n * K);
#pragma omp parallel for
            for (int64_t i = 0; i < n; i++) {
                auto u = unaries + m * n * K + i * K;
                memcpy(objs.data() + i * K, u, sizeof(float) * K);
            }

            // compute objective function by adding unary
            // and binary terms together
            for (size_t other_m = 0; other_m < M; other_m++) {
                if (other_m == m) {
                    continue;
                }

#pragma omp parallel for
                for (int64_t i = 0; i < n; i++) {
                    for (int32_t code = 0; code < K; code++) {
                        int32_t code2 = codes[i * M + other_m];
                        size_t binary_idx = m * M * K * K + other_m * K * K +
                                code * K + code2;
                        // binaries[m, other_m, code, code2]
                        objs[i * K + code] += binaries[binary_idx];
                    }
                }
            }

            // find the optimal value of the m-th subcode
#pragma omp parallel for
            for (int64_t i = 0; i < n; i++) {
                float best_obj = HUGE_VALF;
                int32_t best_code = 0;
                for (size_t code = 0; code < K; code++) {
                    float obj = objs[i * K + code];
                    if (obj < best_obj) {
                        best_obj = obj;
                        best_code = code;
                    }
                }
                codes[i * M + m] = best_code;
            }

        } // loop M
    }
}

void LocalSearchQuantizer::perturb_codes(
        int32_t* codes,
        size_t n,
        std::mt19937& gen) const {
    LSQTimerScope scope(&lsq_timer, "perturb_codes");

    std::uniform_int_distribution<size_t> m_distrib(0, M - 1);
    std::uniform_int_distribution<int32_t> k_distrib(0, K - 1);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < nperts; j++) {
            size_t m = m_distrib(gen);
            codes[i * M + m] = k_distrib(gen);
        }
    }
}

void LocalSearchQuantizer::compute_binary_terms(float* binaries) const {
    LSQTimerScope scope(&lsq_timer, "compute_binary_terms");

#pragma omp parallel for
    for (int64_t m12 = 0; m12 < M * M; m12++) {
        size_t m1 = m12 / M;
        size_t m2 = m12 % M;

        for (size_t code1 = 0; code1 < K; code1++) {
            for (size_t code2 = 0; code2 < K; code2++) {
                const float* c1 = codebooks.data() + m1 * K * d + code1 * d;
                const float* c2 = codebooks.data() + m2 * K * d + code2 * d;
                float ip = fvec_inner_product(c1, c2, d);
                // binaries[m1, m2, code1, code2] = ip * 2
                binaries[m1 * M * K * K + m2 * K * K + code1 * K + code2] =
                        ip * 2;
            }
        }
    }
}

void LocalSearchQuantizer::compute_unary_terms(
        const float* x,
        float* unaries, // [M, n, K]
        size_t n) const {
    LSQTimerScope scope(&lsq_timer, "compute_unary_terms");

    // compute x * codebook^T for each codebook
    //
    // NOTE: LAPACK use column major order
    // out = alpha * op(A) * op(B) + beta * C

    for (size_t m = 0; m < M; m++) {
        FINTEGER nrows_A = K;
        FINTEGER ncols_A = d;

        FINTEGER nrows_B = d;
        FINTEGER ncols_B = n;

        float alpha = -2.0f;
        float beta = 0.0f;
        sgemm_("Transposed",
               "Not Transposed",
               &nrows_A, // nrows of op(A)
               &ncols_B, // ncols of op(B)
               &ncols_A, // ncols of op(A)
               &alpha,
               codebooks.data() + m * K * d,
               &ncols_A, // nrows of A
               x,
               &nrows_B, // nrows of B
               &beta,
               unaries + m * n * K,
               &nrows_A); // nrows of output
    }

    std::vector<float> norms(M * K);
    fvec_norms_L2sqr(norms.data(), codebooks.data(), d, M * K);

#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            float* u = unaries + m * n * K + i * K;
            fvec_add(K, u, norms.data() + m * K, u);
        }
    }
}

float LocalSearchQuantizer::evaluate(
        const int32_t* codes,
        const float* x,
        size_t n,
        float* objs) const {
    LSQTimerScope scope(&lsq_timer, "evaluate");

    // decode
    std::vector<float> decoded_x(n * d, 0.0f);
    float obj = 0.0f;

#pragma omp parallel for reduction(+ : obj)
    for (int64_t i = 0; i < n; i++) {
        const auto code = codes + i * M;
        const auto decoded_i = decoded_x.data() + i * d;
        for (size_t m = 0; m < M; m++) {
            // c = codebooks[m, code[m]]
            const auto c = codebooks.data() + m * K * d + code[m] * d;
            fvec_add(d, decoded_i, c, decoded_i);
        }

        float err = faiss::fvec_L2sqr(x + i * d, decoded_i, d);
        obj += err;

        if (objs) {
            objs[i] = err;
        }
    }

    obj = obj / n;
    return obj;
}

namespace lsq {

IcmEncoder::IcmEncoder(const LocalSearchQuantizer* lsq)
        : verbose(false), lsq(lsq) {}

void IcmEncoder::set_binary_term() {
    auto M = lsq->M;
    auto K = lsq->K;
    binaries.resize(M * M * K * K);
    lsq->compute_binary_terms(binaries.data());
}

void IcmEncoder::encode(
        int32_t* codes,
        const float* x,
        std::mt19937& gen,
        size_t n,
        size_t ils_iters) const {
    lsq->icm_encode_impl(codes, x, binaries.data(), gen, n, ils_iters, verbose);
}

double LSQTimer::get(const std::string& name) {
    if (t.count(name) == 0) {
        return 0.0;
    } else {
        return t[name];
    }
}

void LSQTimer::add(const std::string& name, double delta) {
    if (t.count(name) == 0) {
        t[name] = delta;
    } else {
        t[name] += delta;
    }
}

void LSQTimer::reset() {
    t.clear();
}

LSQTimerScope::LSQTimerScope(LSQTimer* timer, std::string name)
        : timer(timer), name(name), finished(false) {
    t0 = getmillisecs();
}

void LSQTimerScope::finish() {
    if (!finished) {
        auto delta = getmillisecs() - t0;
        timer->add(name, delta);
        finished = true;
    }
}

LSQTimerScope::~LSQTimerScope() {
    finish();
}

} // namespace lsq

} // namespace faiss
