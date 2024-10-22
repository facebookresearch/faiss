/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/quantize_lut.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

namespace quantize_lut {

/******************************************************
 * Quantize look-up tables
 ******************************************************/

namespace {

// there can be NaNs in tables, they should be ignored
float tab_min(const float* tab, size_t n) {
    float min = HUGE_VAL;
    for (int i = 0; i < n; i++) {
        if (tab[i] < min)
            min = tab[i];
    }
    return min;
}

float tab_max(const float* tab, size_t n) {
    float max = -HUGE_VAL;
    for (int i = 0; i < n; i++) {
        if (tab[i] > max)
            max = tab[i];
    }
    return max;
}

void round_tab(float* tab, size_t n, float a, float bi) {
    for (int i = 0; i < n; i++) {
        tab[i] = floorf((tab[i] - bi) * a + 0.5);
    }
}

template <typename T>
void round_tab(const float* tab, size_t n, float a, float bi, T* tab_out) {
    for (int i = 0; i < n; i++) {
        tab_out[i] = (T)floorf((tab[i] - bi) * a + 0.5);
    }
}

} // anonymous namespace

void round_uint8_per_column(
        float* tab,
        size_t n,
        size_t d,
        float* a_out,
        float* b_out) {
    float max_span = 0;
    std::vector<float> mins(n);
    for (int i = 0; i < n; i++) {
        mins[i] = tab_min(tab + i * d, d);
        float span = tab_max(tab + i * d, d) - mins[i];
        if (span > max_span) {
            max_span = span;
        }
    }
    float a = 255 / max_span;
    float b = 0;
    for (int i = 0; i < n; i++) {
        b += mins[i];
        round_tab(tab + i * d, d, a, mins[i]);
    }
    if (a_out)
        *a_out = a;
    if (b_out)
        *b_out = b;
}

void round_uint8_per_column_multi(
        float* tab,
        size_t m,
        size_t n,
        size_t d,
        float* a_out,
        float* b_out) {
    float max_span = 0;
    std::vector<float> mins(n);
    for (int i = 0; i < n; i++) {
        float min_i = HUGE_VAL;
        float max_i = -HUGE_VAL;
        for (int j = 0; j < m; j++) {
            min_i = std::min(min_i, tab_min(tab + (j * n + i) * d, d));
            max_i = std::max(max_i, tab_max(tab + (j * n + i) * d, d));
        }
        mins[i] = min_i;
        float span = max_i - min_i;
        if (span > max_span) {
            max_span = span;
        }
    }
    float a = 255 / max_span;
    float b = 0;
    for (int i = 0; i < n; i++) {
        b += mins[i];
        for (int j = 0; j < m; j++) {
            round_tab(tab + (j * n + i) * d, d, a, mins[i]);
        }
    }
    if (a_out)
        *a_out = a;
    if (b_out)
        *b_out = b;
}

// translation of
// https://github.com/fairinternal/faiss_improvements/blob/7122c3cc6ddb0a371d8aa6f1309cd8bcf2335e61/LUT_quantization.ipynb
void quantize_LUT_and_bias(
        size_t nprobe,
        size_t M,
        size_t ksub,
        bool lut_is_3d,
        const float* LUT,
        const float* bias,
        uint8_t* LUTq,
        size_t M2,
        uint16_t* biasq,
        float* a_out,
        float* b_out) {
    float a, b;
    if (!bias) {
        FAISS_THROW_IF_NOT(!lut_is_3d);
        std::vector<float> mins(M);
        float max_span_LUT = -HUGE_VAL, max_span_dis = 0;
        b = 0;
        for (int i = 0; i < M; i++) {
            mins[i] = tab_min(LUT + i * ksub, ksub);
            float span = tab_max(LUT + i * ksub, ksub) - mins[i];
            max_span_LUT = std::max(max_span_LUT, span);
            max_span_dis += span;
            b += mins[i];
        }
        a = std::min(255 / max_span_LUT, 65535 / max_span_dis);

        for (int i = 0; i < M; i++) {
            round_tab(LUT + i * ksub, ksub, a, mins[i], LUTq + i * ksub);
        }
        memset(LUTq + M * ksub, 0, ksub * (M2 - M));
    } else if (!lut_is_3d) {
        std::vector<float> mins(M);
        float max_span_LUT = -HUGE_VAL, max_span_dis;
        float bias_min = tab_min(bias, nprobe);
        float bias_max = tab_max(bias, nprobe);
        max_span_dis = bias_max - bias_min;
        b = 0;
        for (int i = 0; i < M; i++) {
            mins[i] = tab_min(LUT + i * ksub, ksub);
            float span = tab_max(LUT + i * ksub, ksub) - mins[i];
            max_span_LUT = std::max(max_span_LUT, span);
            max_span_dis += span;
            b += mins[i];
        }
        a = std::min(255 / max_span_LUT, 65535 / max_span_dis);
        b += bias_min;

        for (int i = 0; i < M; i++) {
            round_tab(LUT + i * ksub, ksub, a, mins[i], LUTq + i * ksub);
        }
        memset(LUTq + M * ksub, 0, ksub * (M2 - M));
        round_tab(bias, nprobe, a, bias_min, biasq);

    } else if (biasq) {
        // LUT is 3D
        std::vector<float> mins(nprobe * M);
        std::vector<float> bias2(nprobe);
        float bias_min = tab_min(bias, nprobe);
        float max_span_LUT = -HUGE_VAL, max_span_dis = -HUGE_VAL;

        b = HUGE_VAL;
        size_t ij = 0;
        for (int j = 0; j < nprobe; j++) {
            float max_span_dis_j = bias[j] - bias_min;
            float b2j = bias[j];
            for (int i = 0; i < M; i++) {
                mins[ij] = tab_min(LUT + ij * ksub, ksub);
                float span = tab_max(LUT + ij * ksub, ksub) - mins[ij];
                max_span_LUT = std::max(max_span_LUT, span);
                max_span_dis_j += span;
                b2j += mins[ij];
                ij++;
            }
            max_span_dis = std::max(max_span_dis, max_span_dis_j);
            bias2[j] = b2j;
            b = std::min(b, b2j);
        }

        a = std::min(255 / max_span_LUT, 65535 / max_span_dis);

        ij = 0;
        size_t ij_2 = 0;
        for (int j = 0; j < nprobe; j++) {
            for (int i = 0; i < M; i++) {
                round_tab(
                        LUT + ij * ksub, ksub, a, mins[ij], LUTq + ij_2 * ksub);
                ij++;
                ij_2++;
            }
            memset(LUTq + ij_2 * ksub, 0, ksub * (M2 - M));
            ij_2 += M2 - M;
        }

        round_tab(bias2.data(), nprobe, a, b, biasq);

    } else { // !biasq
        // then we integrate the bias into the LUTs
        std::vector<float> LUT2_storage(nprobe * M * ksub);
        float* LUT2 = LUT2_storage.data();
        size_t ijc = 0;
        for (int j = 0; j < nprobe; j++) {
            float bias_j = bias[j] / M;
            for (int i = 0; i < M; i++) {
                for (int c = 0; c < ksub; c++) {
                    LUT2[ijc] = LUT[ijc] + bias_j;
                    ijc++;
                }
            }
        }
        std::vector<float> mins(M, HUGE_VAL), maxs(M, -HUGE_VAL);
        size_t ij = 0;
        for (int j = 0; j < nprobe; j++) {
            for (int i = 0; i < M; i++) {
                mins[i] = std::min(mins[i], tab_min(LUT2 + ij * ksub, ksub));
                maxs[i] = std::max(maxs[i], tab_max(LUT2 + ij * ksub, ksub));
                ij++;
            }
        }

        float max_span = -HUGE_VAL;
        b = 0;
        for (int i = 0; i < M; i++) {
            float span = maxs[i] - mins[i];
            max_span = std::max(max_span, span);
            b += mins[i];
        }
        a = 255 / max_span;
        ij = 0;
        size_t ij_2 = 0;
        for (int j = 0; j < nprobe; j++) {
            for (int i = 0; i < M; i++) {
                round_tab(
                        LUT2 + ij * ksub, ksub, a, mins[i], LUTq + ij_2 * ksub);
                ij++;
                ij_2++;
            }
            memset(LUTq + ij_2 * ksub, 0, ksub * (M2 - M));
            ij_2 += M2 - M;
        }
    }
    if (a_out)
        *a_out = a;
    if (b_out)
        *b_out = b;
}

void aq_quantize_LUT_and_bias(
        size_t nprobe,
        size_t M,
        size_t ksub,
        const float* LUT,
        const float* bias,
        size_t M_norm,
        int norm_scale,
        uint8_t* LUTq,
        size_t M2,
        uint16_t* biasq,
        float* a_out,
        float* b_out) {
    float a, b;
    std::vector<float> mins(M);
    float max_span_LUT = -HUGE_VAL, max_span_dis;
    float bias_min = tab_min(bias, nprobe);
    float bias_max = tab_max(bias, nprobe);
    max_span_dis = bias_max - bias_min;
    b = 0;
    for (int i = 0; i < M; i++) {
        mins[i] = tab_min(LUT + i * ksub, ksub);
        float span = tab_max(LUT + i * ksub, ksub) - mins[i];
        max_span_LUT = std::max(max_span_LUT, span);
        max_span_dis += (i >= M - M_norm ? span * norm_scale : span);
        b += mins[i];
    }
    a = std::min(255 / max_span_LUT, 65535 / max_span_dis);
    b += bias_min;

    for (int i = 0; i < M; i++) {
        round_tab(LUT + i * ksub, ksub, a, mins[i], LUTq + i * ksub);
    }
    memset(LUTq + M * ksub, 0, ksub * (M2 - M));
    round_tab(bias, nprobe, a, bias_min, biasq);

    *a_out = a;
    *b_out = b;
}

float aq_estimate_norm_scale(
        size_t M,
        size_t ksub,
        size_t M_norm,
        const float* LUT) {
    float max_span_LUT = -HUGE_VAL;
    for (int i = 0; i < M - M_norm; i++) {
        float min = tab_min(LUT + i * ksub, ksub);
        float span = tab_max(LUT + i * ksub, ksub) - min;
        max_span_LUT = std::max(max_span_LUT, span);
    }

    float max_span_LUT_norm = -HUGE_VAL;
    for (int i = M - M_norm; i < M; i++) {
        float min = tab_min(LUT + i * ksub, ksub);
        float span = tab_max(LUT + i * ksub, ksub) - min;
        max_span_LUT_norm = std::max(max_span_LUT_norm, span);
    }

    return max_span_LUT_norm / max_span_LUT;
}

} // namespace quantize_lut

} // namespace faiss
