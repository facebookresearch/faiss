/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <cstring>
#include <memory>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>

#include <faiss/impl/scalar_quantizer/training.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>

#include <faiss/impl/scalar_quantizer/scanners.h>

#define THE_LEVEL_TO_DISPATCH SIMDLevel::NONE
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

namespace faiss {

namespace {

// Gaussian Lloyd-Max optimal quantizer centroids and boundaries for N(0,1).
// clang-format off
const float kLloydMaxCentroids1[] = {
    -0.797884560802865f, 0.797884560802865f
};
const float kLloydMaxBoundaries1[] = {
    0.000000000000000f
};
const float kLloydMaxCentroids2[] = {
    -1.510417608499078f, -0.452780034636484f,
     0.452780034636483f,  1.510417608499078f
};
const float kLloydMaxBoundaries2[] = {
    -0.981598821567781f, 0.000000000000000f, 0.981598821567781f
};
const float kLloydMaxCentroids3[] = {
    -2.151945704536914f, -1.343909278504930f,
    -0.756005281205826f, -0.245094178944203f,
     0.245094178944203f,  0.756005281205825f,
     1.343909278504930f,  2.151945704536914f
};
const float kLloydMaxBoundaries3[] = {
    -1.747927491520922f, -1.049957279855378f,
    -0.500549730075014f,  0.000000000000000f,
     0.500549730075014f,  1.049957279855378f,
     1.747927491520922f
};
const float kLloydMaxCentroids4[] = {
    -2.732589570994957f, -2.069017226531159f,
    -1.618046386021649f, -1.256231197346957f,
    -0.942340456486774f, -0.656759118532318f,
    -0.388048299490198f, -0.128395029851116f,
     0.128395029851116f,  0.388048299490198f,
     0.656759118532318f,  0.942340456486773f,
     1.256231197346959f,  1.618046386021649f,
     2.069017226531160f,  2.732589570994943f
};
const float kLloydMaxBoundaries4[] = {
    -2.400803398763058f, -1.843531806276404f,
    -1.437138791684303f, -1.099285826916865f,
    -0.799549787509546f, -0.522403709011258f,
    -0.258221664670657f,  0.000000000000000f,
     0.258221664670657f,  0.522403709011258f,
     0.799549787509546f,  1.099285826916866f,
     1.437138791684304f,  1.843531806276404f,
     2.400803398763051f
};
const float kLloydMaxCentroids8[] = {
    -4.2734901319f, -3.8270895246f, -3.5457169520f, -3.3354593381f,
    -3.1655721017f, -3.0219515320f, -2.8969009924f, -2.7857394515f,
    -2.6853990170f, -2.5937556343f, -2.5092755166f, -2.4308135619f,
    -2.3574913691f, -2.2886197969f, -2.2236478246f, -2.1621276457f,
    -2.1036901632f, -2.0480273642f, -1.9948793740f, -1.9440247677f,
    -1.8952732015f, -1.8484597247f, -1.8034403315f, -1.7600884415f,
    -1.7182920846f, -1.6779516274f, -1.6389779215f, -1.6012907825f,
    -1.5648177311f, -1.5294929453f, -1.4952563823f, -1.4620530375f,
    -1.4298323186f, -1.3985475108f, -1.3681553217f, -1.3386154890f,
    -1.3098904444f, -1.2819450217f, -1.2547462051f, -1.2282629097f,
    -1.2024657910f, -1.1773270781f, -1.1528204287f, -1.1289208010f,
    -1.1056043421f, -1.0828482901f, -1.0606308873f, -1.0389313043f,
    -1.0177295729f, -0.9970065268f, -0.9767437492f, -0.9569235264f,
    -0.9375288069f, -0.9185431646f, -0.8999507663f, -0.8817363426f,
    -0.8638851621f, -0.8463830081f, -0.8292161569f, -0.8123713596f,
    -0.7958358242f, -0.7795971999f, -0.7636435625f, -0.7479634007f,
    -0.7325456038f, -0.7173794494f, -0.7024545929f, -0.6877610560f,
    -0.6732892172f, -0.6590298016f, -0.6449738716f, -0.6311128174f,
    -0.6174383481f, -0.6039424829f, -0.5906175419f, -0.5774561379f,
    -0.5644511676f, -0.5515958029f, -0.5388834832f, -0.5263079060f,
    -0.5138630194f, -0.5015430136f, -0.4893423125f, -0.4772555660f,
    -0.4652776416f, -0.4534036165f, -0.4416287701f, -0.4299485757f,
    -0.4183586932f, -0.4068549615f, -0.3954333909f, -0.3840901561f,
    -0.3728215889f, -0.3616241712f, -0.3504945283f, -0.3394294221f,
    -0.3284257446f, -0.3174805116f, -0.3065908567f, -0.2957540250f,
    -0.2849673675f, -0.2742283355f, -0.2635344752f, -0.2528834222f,
    -0.2422728967f, -0.2317006985f, -0.2211647022f, -0.2106628526f,
    -0.2001931607f, -0.1897536989f, -0.1793425974f, -0.1689580400f,
    -0.1585982605f, -0.1482615390f, -0.1379461985f, -0.1276506012f,
    -0.1173731457f, -0.1071122637f, -0.0968664166f, -0.0866340933f,
    -0.0764138065f, -0.0662040909f, -0.0560034994f, -0.0458106014f,
    -0.0356239797f, -0.0254422284f, -0.0152639496f, -0.0050877521f,
     0.0050877521f,  0.0152639496f,  0.0254422284f,  0.0356239797f,
     0.0458106014f,  0.0560034994f,  0.0662040909f,  0.0764138065f,
     0.0866340933f,  0.0968664166f,  0.1071122637f,  0.1173731457f,
     0.1276506012f,  0.1379461985f,  0.1482615390f,  0.1585982605f,
     0.1689580400f,  0.1793425974f,  0.1897536989f,  0.2001931607f,
     0.2106628526f,  0.2211647022f,  0.2317006985f,  0.2422728967f,
     0.2528834222f,  0.2635344752f,  0.2742283355f,  0.2849673675f,
     0.2957540250f,  0.3065908567f,  0.3174805116f,  0.3284257446f,
     0.3394294221f,  0.3504945283f,  0.3616241712f,  0.3728215889f,
     0.3840901561f,  0.3954333909f,  0.4068549615f,  0.4183586932f,
     0.4299485757f,  0.4416287701f,  0.4534036165f,  0.4652776416f,
     0.4772555660f,  0.4893423125f,  0.5015430136f,  0.5138630194f,
     0.5263079060f,  0.5388834832f,  0.5515958029f,  0.5644511676f,
     0.5774561379f,  0.5906175419f,  0.6039424829f,  0.6174383481f,
     0.6311128174f,  0.6449738716f,  0.6590298016f,  0.6732892172f,
     0.6877610560f,  0.7024545929f,  0.7173794494f,  0.7325456038f,
     0.7479634007f,  0.7636435625f,  0.7795971999f,  0.7958358242f,
     0.8123713596f,  0.8292161569f,  0.8463830081f,  0.8638851621f,
     0.8817363426f,  0.8999507663f,  0.9185431646f,  0.9375288069f,
     0.9569235264f,  0.9767437492f,  0.9970065268f,  1.0177295729f,
     1.0389313043f,  1.0606308873f,  1.0828482901f,  1.1056043421f,
     1.1289208010f,  1.1528204287f,  1.1773270781f,  1.2024657910f,
     1.2282629097f,  1.2547462051f,  1.2819450217f,  1.3098904444f,
     1.3386154890f,  1.3681553217f,  1.3985475108f,  1.4298323186f,
     1.4620530375f,  1.4952563823f,  1.5294929453f,  1.5648177311f,
     1.6012907825f,  1.6389779215f,  1.6779516274f,  1.7182920846f,
     1.7600884415f,  1.8034403315f,  1.8484597247f,  1.8952732015f,
     1.9440247677f,  1.9948793740f,  2.0480273642f,  2.1036901632f,
     2.1621276457f,  2.2236478246f,  2.2886197969f,  2.3574913691f,
     2.4308135619f,  2.5092755166f,  2.5937556343f,  2.6853990170f,
     2.7857394515f,  2.8969009924f,  3.0219515320f,  3.1655721017f,
     3.3354593381f,  3.5457169520f,  3.8270895246f,  4.2734901319f
};
const float kLloydMaxBoundaries8[] = {
    -4.0502898282f, -3.6864032383f, -3.4405881450f, -3.2505157199f,
    -3.0937618168f, -2.9594262622f, -2.8413202220f, -2.7355692343f,
    -2.6395773257f, -2.5515155755f, -2.4700445392f, -2.3941524655f,
    -2.3230555830f, -2.2561338107f, -2.1928877352f, -2.1329089044f,
    -2.0758587637f, -2.0214533691f, -1.9694520708f, -1.9196489846f,
    -1.8718664631f, -1.8259500281f, -1.7817643865f, -1.7391902630f,
    -1.6981218560f, -1.6584647744f, -1.6201343520f, -1.5830542568f,
    -1.5471553382f, -1.5123746638f, -1.4786547099f, -1.4459426781f,
    -1.4141899147f, -1.3833514163f, -1.3533854053f, -1.3242529667f,
    -1.2959177331f, -1.2683456134f, -1.2415045574f, -1.2153643503f,
    -1.1898964346f, -1.1650737534f, -1.1408706148f, -1.1172625715f,
    -1.0942263161f, -1.0717395887f, -1.0497810958f, -1.0283304386f,
    -1.0073680499f, -0.9868751380f, -0.9668336378f, -0.9472261667f,
    -0.9280359858f, -0.9092469654f, -0.8908435544f, -0.8728107524f,
    -0.8551340851f, -0.8377995825f, -0.8207937582f, -0.8041035919f,
    -0.7877165121f, -0.7716203812f, -0.7558034816f, -0.7402545023f,
    -0.7249625266f, -0.7099170212f, -0.6951078244f, -0.6805251366f,
    -0.6661595094f, -0.6520018366f, -0.6380433445f, -0.6242755828f,
    -0.6106904155f, -0.5972800124f, -0.5840368399f, -0.5709536527f,
    -0.5580234853f, -0.5452396431f, -0.5325956946f, -0.5200854627f,
    -0.5077030165f, -0.4954426631f, -0.4832989393f, -0.4712666038f,
    -0.4593406291f, -0.4475161933f, -0.4357886729f, -0.4241536345f,
    -0.4126068274f, -0.4011441762f, -0.3897617735f, -0.3784558725f,
    -0.3672228800f, -0.3560593498f, -0.3449619752f, -0.3339275834f,
    -0.3229531281f, -0.3120356842f, -0.3011724408f, -0.2903606962f,
    -0.2795978515f, -0.2688814053f, -0.2582089487f, -0.2475781595f,
    -0.2369867976f, -0.2264327004f, -0.2159137774f, -0.2054280067f,
    -0.1949734298f, -0.1845481481f, -0.1741503187f, -0.1637781502f,
    -0.1534298998f, -0.1431038688f, -0.1327983999f, -0.1225118735f,
    -0.1122427047f, -0.1019893401f, -0.0917502549f, -0.0815239499f,
    -0.0713089487f, -0.0611037951f, -0.0509070504f, -0.0407172906f,
    -0.0305331041f, -0.0203530890f, -0.0101758509f,  0.0000000000f,
     0.0101758509f,  0.0203530890f,  0.0305331041f,  0.0407172906f,
     0.0509070504f,  0.0611037951f,  0.0713089487f,  0.0815239499f,
     0.0917502549f,  0.1019893401f,  0.1122427047f,  0.1225118735f,
     0.1327983999f,  0.1431038688f,  0.1534298998f,  0.1637781502f,
     0.1741503187f,  0.1845481481f,  0.1949734298f,  0.2054280067f,
     0.2159137774f,  0.2264327004f,  0.2369867976f,  0.2475781595f,
     0.2582089487f,  0.2688814053f,  0.2795978515f,  0.2903606962f,
     0.3011724408f,  0.3120356842f,  0.3229531281f,  0.3339275834f,
     0.3449619752f,  0.3560593498f,  0.3672228800f,  0.3784558725f,
     0.3897617735f,  0.4011441762f,  0.4126068274f,  0.4241536345f,
     0.4357886729f,  0.4475161933f,  0.4593406291f,  0.4712666038f,
     0.4832989393f,  0.4954426631f,  0.5077030165f,  0.5200854627f,
     0.5325956946f,  0.5452396431f,  0.5580234853f,  0.5709536527f,
     0.5840368399f,  0.5972800124f,  0.6106904155f,  0.6242755828f,
     0.6380433445f,  0.6520018366f,  0.6661595094f,  0.6805251366f,
     0.6951078244f,  0.7099170212f,  0.7249625266f,  0.7402545023f,
     0.7558034816f,  0.7716203812f,  0.7877165121f,  0.8041035919f,
     0.8207937582f,  0.8377995825f,  0.8551340851f,  0.8728107524f,
     0.8908435544f,  0.9092469654f,  0.9280359858f,  0.9472261667f,
     0.9668336378f,  0.9868751380f,  1.0073680499f,  1.0283304386f,
     1.0497810958f,  1.0717395887f,  1.0942263161f,  1.1172625715f,
     1.1408706148f,  1.1650737534f,  1.1898964346f,  1.2153643503f,
     1.2415045574f,  1.2683456134f,  1.2959177331f,  1.3242529667f,
     1.3533854053f,  1.3833514163f,  1.4141899147f,  1.4459426781f,
     1.4786547099f,  1.5123746638f,  1.5471553382f,  1.5830542568f,
     1.6201343520f,  1.6584647744f,  1.6981218560f,  1.7391902630f,
     1.7817643865f,  1.8259500281f,  1.8718664631f,  1.9196489846f,
     1.9694520708f,  2.0214533691f,  2.0758587637f,  2.1329089044f,
     2.1928877352f,  2.2561338107f,  2.3230555830f,  2.3941524655f,
     2.4700445392f,  2.5515155755f,  2.6395773257f,  2.7355692343f,
     2.8413202220f,  2.9594262622f,  3.0937618168f,  3.2505157199f,
     3.4405881450f,  3.6864032383f,  4.0502898282f
};
// clang-format on

struct LloydMaxTable {
    const float* centroids;
    const float* boundaries;
};

const LloydMaxTable kLloydMaxTables[] = {
        {nullptr, nullptr},                          // 0
        {kLloydMaxCentroids1, kLloydMaxBoundaries1}, // 1
        {kLloydMaxCentroids2, kLloydMaxBoundaries2}, // 2
        {kLloydMaxCentroids3, kLloydMaxBoundaries3}, // 3
        {kLloydMaxCentroids4, kLloydMaxBoundaries4}, // 4
        {nullptr, nullptr},                          // 5 (unused)
        {nullptr, nullptr},                          // 6 (unused)
        {nullptr, nullptr},                          // 7 (unused)
        {kLloydMaxCentroids8, kLloydMaxBoundaries8}, // 8
};

void populate_lloyd_max_trained(size_t mse_bits, std::vector<float>& trained) {
    FAISS_THROW_IF_NOT(mse_bits >= 1 && mse_bits <= 8);
    FAISS_THROW_IF_NOT(kLloydMaxTables[mse_bits].centroids != nullptr);
    size_t k = size_t(1) << mse_bits;
    const auto& t = kLloydMaxTables[mse_bits];
    trained.resize(k + (k - 1));
    std::copy(t.centroids, t.centroids + k, trained.begin());
    std::copy(t.boundaries, t.boundaries + k - 1, trained.begin() + k);
}

} // namespace

/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer(size_t d_in, QuantizerType qtype_in)
        : Quantizer(d_in), qtype(qtype_in) {
    set_derived_sizes();
}

ScalarQuantizer::ScalarQuantizer() {}

void ScalarQuantizer::set_derived_sizes() {
    switch (qtype) {
        case QT_1bit_tqmse:
            code_size = (d + 7) / 8;
            bits = 1;
            break;
        case QT_2bit_tqmse:
            code_size = (d * 2 + 7) / 8;
            bits = 2;
            break;
        case QT_3bit_tqmse:
            code_size = (d * 3 + 7) / 8;
            bits = 3;
            break;
        case QT_8bit:
        case QT_8bit_uniform:
        case QT_8bit_direct:
        case QT_8bit_direct_signed:
        case QT_8bit_tqmse:
            code_size = d;
            bits = 8;
            break;
        case QT_4bit:
        case QT_4bit_uniform:
        case QT_4bit_tqmse:
            code_size = (d + 1) / 2;
            bits = 4;
            break;
        case QT_6bit:
            code_size = (d * 6 + 7) / 8;
            bits = 6;
            break;
        case QT_fp16:
            code_size = d * 2;
            bits = 16;
            break;
        case QT_bf16:
            code_size = d * 2;
            bits = 16;
            break;
        case QT_0bit:
            code_size = 0;
            bits = 0;
            break;
        case QT_2bit_tq:
        case QT_3bit_tq:
        case QT_4bit_tq:
        case QT_5bit_tq: {
            size_t nb_bits = (qtype == QT_2bit_tq) ? 2
                    : (qtype == QT_3bit_tq)        ? 3
                    : (qtype == QT_4bit_tq)        ? 4
                    : (qtype == QT_5bit_tq)        ? 5
                                                   : 0;
            FAISS_THROW_IF_NOT_MSG(nb_bits > 0, "unexpected TurboQ qtype");
            size_t mse_bits = nb_bits - 1;
            size_t mse_bytes = mse_bits * ((d + 7) / 8);
            size_t qjl_bytes = (d + 7) / 8;
            code_size = mse_bytes + qjl_bytes +
                    sizeof(scalar_quantizer::SQTurboQFactors);
            bits = nb_bits;
            break;
        }
        default:
            break;
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
    using scalar_quantizer::train_NonUniform;
    using scalar_quantizer::train_Uniform;

    if (qtype == QT_0bit) {
        return; // nothing to train for centroid-only mode
    }

    int bit_per_dim = qtype == QT_4bit_uniform ? 4
            : qtype == QT_4bit                 ? 4
            : qtype == QT_6bit                 ? 6
            : qtype == QT_8bit_uniform         ? 8
            : qtype == QT_8bit                 ? 8
                                               : -1;

    switch (qtype) {
        case QT_4bit_uniform:
        case QT_8bit_uniform:
            FAISS_THROW_IF_NOT(n > 0);
            FAISS_THROW_IF_NOT(x != nullptr);
            train_Uniform(
                    rangestat,
                    rangestat_arg,
                    n * d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_4bit:
        case QT_8bit:
        case QT_6bit:
            FAISS_THROW_IF_NOT(n > 0);
            FAISS_THROW_IF_NOT(x != nullptr);
            train_NonUniform(
                    rangestat,
                    rangestat_arg,
                    n,
                    int(d),
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_fp16:
        case QT_8bit_direct:
        case QT_bf16:
        case QT_8bit_direct_signed:
            // no training necessary
            break;
        case QT_1bit_tqmse:
            populate_lloyd_max_trained(1, trained);
            break;
        case QT_2bit_tqmse:
            populate_lloyd_max_trained(2, trained);
            break;
        case QT_3bit_tqmse:
            populate_lloyd_max_trained(3, trained);
            break;
        case QT_4bit_tqmse:
            populate_lloyd_max_trained(4, trained);
            break;
        case QT_8bit_tqmse:
            populate_lloyd_max_trained(8, trained);
            break;
        case QT_2bit_tq:
        case QT_3bit_tq:
        case QT_4bit_tq:
        case QT_5bit_tq: {
            size_t mse_bits = bits - 1;
            populate_lloyd_max_trained(mse_bits, trained);
            // Pack seed and qjl_type at end of trained for dispatch
            float seed_f[2];
            TurboQuantRefine::pack_seed(turboq_refine.seed, seed_f);
            trained.push_back(seed_f[0]);
            trained.push_back(seed_f[1]);
            trained.push_back(static_cast<float>(turboq_refine.qjl_type));
            turboq_refine.init_projection(d);
            break;
        }
        default:
            break;
    }
}

void ScalarQuantizer::TurboQuantRefine::init_projection(size_t d) {
    if (use_fwht()) {
        padded_d = 1;
        while (padded_d < d) {
            padded_d <<= 1;
        }
        fwht_signs.resize(padded_d);
        RandomGenerator rng(seed);
        for (size_t i = 0; i < padded_d; i++) {
            fwht_signs[i] = (rng.rand_int(2) == 0) ? 1.0f : -1.0f;
        }
    } else {
        rr_matrix.resize(d * d);
        float_randn(rr_matrix.data(), d * d, seed);
        matrix_qr(static_cast<int>(d), static_cast<int>(d), rr_matrix.data());
    }
}

ScalarQuantizer::SQuantizer* ScalarQuantizer::select_quantizer() const {
    return with_simd_level_spr([&]<SIMDLevel SL>() -> SQuantizer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* q = scalar_quantizer::sq_select_quantizer<SL>(
                    qtype, d, trained);
            if (q) {
                return q;
            }
        }
        return scalar_quantizer::sq_select_quantizer<SIMDLevel::NONE>(
                qtype, d, trained);
    });
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    if (code_size == 0) {
        return; // QT_0bit: nothing to encode
    }
    std::unique_ptr<SQuantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        squant->encode_vector(x + i * d, codes + i * code_size);
    }
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    if (code_size == 0) {
        memset(x, 0, sizeof(float) * d * n);
        return; // QT_0bit: no per-vector data, zero-fill
    }
    std::unique_ptr<SQuantizer> squant(select_quantizer());

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        squant->decode_vector(codes + i * code_size, x + i * d);
    }
}

ScalarQuantizer::SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
    return with_simd_level_spr([&]<SIMDLevel SL>() -> SQDistanceComputer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* dc = scalar_quantizer::sq_select_distance_computer<SL>(
                    metric, qtype, d, trained);
            if (dc) {
                return dc;
            }
        }
        return scalar_quantizer::sq_select_distance_computer<SIMDLevel::NONE>(
                metric, qtype, d, trained);
    });
}

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
    return with_simd_level_spr([&]<SIMDLevel SL>() -> InvertedListScanner* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* s = scalar_quantizer::sq_select_InvertedListScanner<SL>(
                    qtype,
                    mt,
                    d,
                    code_size,
                    trained,
                    quantizer,
                    store_pairs,
                    sel,
                    by_residual);
            if (s) {
                return s;
            }
        }
        return scalar_quantizer::sq_select_InvertedListScanner<SIMDLevel::NONE>(
                qtype,
                mt,
                d,
                code_size,
                trained,
                quantizer,
                store_pairs,
                sel,
                by_residual);
    });
}

} // namespace faiss
