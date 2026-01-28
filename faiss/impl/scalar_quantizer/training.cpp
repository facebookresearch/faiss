/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/scalar_quantizer/training.h>

#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cmath>

namespace faiss {

namespace scalar_quantizer {
/*******************************************************************
 * Quantizer range training
 */

static float sqr(float x) {
    return x * x;
}

void train_Uniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int k,
        const float* x,
        std::vector<float>& trained) {
    trained.resize(2);
    float& vmin = trained[0];
    float& vmax = trained[1];

    if (rs == ScalarQuantizer::RS_minmax) {
        vmin = HUGE_VAL;
        vmax = -HUGE_VAL;
        for (size_t i = 0; i < n; i++) {
            if (x[i] < vmin) {
                vmin = x[i];
            }
            if (x[i] > vmax) {
                vmax = x[i];
            }
        }
        float vexp = (vmax - vmin) * rs_arg;
        vmin -= vexp;
        vmax += vexp;
    } else if (rs == ScalarQuantizer::RS_meanstd) {
        double sum = 0, sum2 = 0;
        for (size_t i = 0; i < n; i++) {
            sum += x[i];
            sum2 += x[i] * x[i];
        }
        float mean = sum / n;
        float var = sum2 / n - mean * mean;
        float std = var <= 0 ? 1.0 : std::sqrt(var);

        vmin = mean - std * rs_arg;
        vmax = mean + std * rs_arg;
    } else if (rs == ScalarQuantizer::RS_quantiles) {
        std::vector<float> x_copy(n);
        memcpy(x_copy.data(), x, n * sizeof(*x));
        // TODO just do a quickselect
        std::sort(x_copy.begin(), x_copy.end());
        int o = int(rs_arg * n);
        if (o < 0) {
            o = 0;
        }
        if (o > n - o) {
            o = n / 2;
        }
        vmin = x_copy[o];
        vmax = x_copy[n - 1 - o];

    } else if (rs == ScalarQuantizer::RS_optim) {
        float a, b;
        float sx = 0;
        {
            vmin = HUGE_VAL, vmax = -HUGE_VAL;
            for (size_t i = 0; i < n; i++) {
                if (x[i] < vmin) {
                    vmin = x[i];
                }
                if (x[i] > vmax) {
                    vmax = x[i];
                }
                sx += x[i];
            }
            b = vmin;
            a = (vmax - vmin) / (k - 1);
        }
        int verbose = false;
        int niter = 2000;
        float last_err = -1;
        int iter_last_err = 0;
        for (int it = 0; it < niter; it++) {
            float sn = 0, sn2 = 0, sxn = 0, err1 = 0;

            for (idx_t i = 0; i < n; i++) {
                float xi = x[i];
                float ni = floor((xi - b) / a + 0.5);
                if (ni < 0) {
                    ni = 0;
                }
                if (ni >= k) {
                    ni = k - 1;
                }
                err1 += sqr(xi - (ni * a + b));
                sn += ni;
                sn2 += ni * ni;
                sxn += ni * xi;
            }

            if (err1 == last_err) {
                iter_last_err++;
                if (iter_last_err == 16) {
                    break;
                }
            } else {
                last_err = err1;
                iter_last_err = 0;
            }

            float det = sqr(sn) - sn2 * n;

            b = (sn * sxn - sn2 * sx) / det;
            a = (sn * sx - n * sxn) / det;
            if (verbose) {
                printf("it %d, err1=%g            \r", it, err1);
                fflush(stdout);
            }
        }
        if (verbose) {
            printf("\n");
        }

        vmin = b;
        vmax = b + a * (k - 1);

    } else {
        FAISS_THROW_MSG("Invalid qtype");
    }
    vmax -= vmin;
}

void train_NonUniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int d,
        int k,
        const float* x,
        std::vector<float>& trained) {
    trained.resize(static_cast<size_t>(2) * d);
    float* vmin = trained.data();
    float* vmax = trained.data() + d;
    if (rs == ScalarQuantizer::RS_minmax) {
        memcpy(vmin, x, sizeof(*x) * d);
        memcpy(vmax, x, sizeof(*x) * d);
        for (size_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                if (xi[j] < vmin[j]) {
                    vmin[j] = xi[j];
                }
                if (xi[j] > vmax[j]) {
                    vmax[j] = xi[j];
                }
            }
        }
        float* vdiff = vmax;
        for (size_t j = 0; j < d; j++) {
            float vexp = (vmax[j] - vmin[j]) * rs_arg;
            vmin[j] -= vexp;
            vmax[j] += vexp;
            vdiff[j] = vmax[j] - vmin[j];
        }
    } else {
        // transpose
        std::vector<float> xt(n * d);
        for (size_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                xt[j * n + i] = xi[j];
            }
        }
        std::vector<float> trained_d(2);
#pragma omp parallel for
        for (int j = 0; j < d; j++) {
            train_Uniform(rs, rs_arg, n, k, xt.data() + j * n, trained_d);
            vmin[j] = trained_d[0];
            vmax[j] = trained_d[1];
        }
    }
}

} // namespace scalar_quantizer

} // namespace faiss
