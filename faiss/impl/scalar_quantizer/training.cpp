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

// Positive Lloyd-Max reproduction levels for a standard normal source. The
// negative half is filled by symmetry in make_lloyd_max_normal_centroids().
constexpr float kLloydMaxNormalCentroids1[] = {0.7978845608028654f};
constexpr float kLloydMaxNormalCentroids2[] = {
        0.4527800398860679f,
        1.5104176087114887f};
constexpr float kLloydMaxNormalCentroids3[] = {
        0.24509416307340598f,
        0.7560052489539643f,
        1.3439092613750225f,
        2.151945669890335f};
constexpr float kLloydMaxNormalCentroids4[] = {
        0.12839501671105813f,
        0.38804823445328507f,
        0.6567589957631145f,
        0.9423402689122875f,
        1.2562309480263467f,
        1.6180460517130526f,
        2.069016730231837f,
        2.732588804065177f};
constexpr float kLloydMaxNormalCentroids5[] = {
        0.06588962234909321f,
        0.1980516892038791f,
        0.3313780514298761f,
        0.4666991751197207f,
        0.6049331689395434f,
        0.7471351317890572f,
        0.89456439585444f,
        1.0487823813655852f,
        1.2118032120324f,
        1.3863389353626248f,
        1.576226389073775f,
        1.7872312118858462f,
        2.0287259913633036f,
        2.3177364021261493f,
        2.69111557955431f,
        3.260726295605043f};
constexpr float kLloydMaxNormalCentroids6[] = {
        0.0334094558802581f,  0.1002781217139195f, 0.16729660990171974f,
        0.23456656976873475f, 0.3021922894403614f, 0.37028193328115516f,
        0.4389488009177737f,  0.5083127587538033f, 0.5785018460645791f,
        0.6496542452315348f,  0.7219204720694183f, 0.7954660529025513f,
        0.870474868055092f,   0.9471530930156288f, 1.0257343133937524f,
        1.1064859596918581f,  1.1897175711327463f, 1.2757916223519965f,
        1.3651378971823598f,  1.458272959944728f,  1.5558274659528346f,
        1.6585847114298427f,  1.7675371481292605f, 1.8839718992293555f,
        2.009604894545278f,   2.146803022259123f,  2.2989727412973995f,
        2.471294740528467f,   2.6722617014102585f, 2.91739146530985f,
        3.2404166403241677f,  3.7440690236964755f};
constexpr float kLloydMaxNormalCentroids7[] = {
        0.016828143177728235f, 0.05049075396896167f, 0.08417241989671888f,
        0.11788596825032507f,  0.1516442630131618f,  0.18546025708680833f,
        0.21934708340331643f,  0.25331807190834565f, 0.2873868062260947f,
        0.32156710392315796f,  0.355873075050329f,   0.39031926330596733f,
        0.4249205523979007f,   0.4596922300454219f,  0.49465018161031576f,
        0.5298108436256188f,   0.565191195643323f,   0.600808970989236f,
        0.6366826613981411f,   0.6728315674936343f,  0.7092759460939766f,
        0.746037126679468f,    0.7831375375631398f,  0.8206007832455021f,
        0.858451939611374f,    0.896717615963322f,   0.9354260757626341f,
        0.9746074842160436f,   1.0142940678300427f,  1.054520418037026f,
        1.0953237719213182f,   1.1367442623434032f,  1.1788252655205043f,
        1.2216138763870124f,   1.26516137869917f,    1.309523700469555f,
        1.3547621051156036f,   1.4009441065262136f,  1.448144252238147f,
        1.4964451375010575f,   1.5459387008934842f,  1.596727786313424f,
        1.6489283062238074f,   1.7026711624156725f,  1.7581051606756466f,
        1.8154009933798645f,   1.8747553268072956f,  1.9363967204122827f,
        2.0005932433837565f,   2.0676621538384503f,  2.1379832427349696f,
        2.212016460501213f,    2.2903268704925304f,  2.3736203164211713f,
        2.4627959084523208f,   2.5590234991374485f,  2.663867022558051f,
        2.7794919110540777f,   2.909021527386642f,   3.0572161028423737f,
        3.231896182843021f,    3.4473810105937095f,  3.7348571053691555f,
        4.1895219330235225f};
constexpr float kLloydMaxNormalCentroids8[] = {
        0.008445974137017219f, 0.025338726226901278f, 0.042233889994651476f,
        0.05913307399220878f,  0.07603788791797023f,  0.09294994306815242f,
        0.10987089037069565f,  0.12680234584461386f,  0.1437459285205906f,
        0.16070326074968388f,  0.1776760066764216f,   0.19466583496246115f,
        0.21167441946986007f,  0.22870343946322488f,  0.24575458029044564f,
        0.2628295721769575f,   0.2799301528634766f,   0.29705806782573063f,
        0.3142150709211129f,   0.3314029639954903f,   0.34862355883476864f,
        0.3658786774238477f,   0.3831701926964899f,   0.40049998943716425f,
        0.4178699650069057f,   0.4352820704086704f,   0.45273827097956804f,
        0.4702405882876f,      0.48779106011037887f,  0.505391740756901f,
        0.5230447441905988f,   0.5407522460590347f,   0.558516486141511f,
        0.5763396823538222f,   0.5942241184949506f,   0.6121721459546814f,
        0.6301861414640443f,   0.6482685527755422f,   0.6664219019236218f,
        0.684648787627676f,    0.7029517931200633f,   0.7213336286470308f,
        0.7397970881081071f,   0.7583450032075904f,   0.7769802937007926f,
        0.7957059197645721f,   0.8145249861674053f,   0.8334407494351099f,
        0.8524564651728141f,   0.8715754936480047f,   0.8908013031010308f,
        0.9101374749919184f,   0.9295877653215154f,   0.9491559977740125f,
        0.9688461234581733f,   0.9886622867721733f,   1.0086087121824747f,
        1.028689768268861f,    1.0489101021225093f,   1.0692743940997251f,
        1.0897875553561465f,   1.1104547388972044f,   1.1312812154370708f,
        1.1522725891384287f,   1.173434599389649f,    1.1947731980672593f,
        1.2162947131430126f,   1.238005717146854f,    1.2599130381874064f,
        1.2820237696510286f,   1.304345369166531f,    1.3268857708606756f,
        1.349653145284911f,    1.3726560932224416f,   1.3959037693197867f,
        1.419405726021264f,    1.4431719292973744f,   1.4672129964566984f,
        1.4915401336751468f,   1.5161650628244996f,   1.541100284490976f,
        1.5663591473033147f,   1.5919556551358922f,   1.6179046397057497f,
        1.6442219553485078f,   1.6709244249695359f,   1.6980300628044107f,
        1.7255580190748743f,   1.7535288357430767f,   1.7819645728459763f,
        1.81088895442524f,     1.8403273195729115f,   1.870306964218662f,
        1.9008577747790962f,   1.9320118435829472f,   1.9638039107009146f,
        1.9962716117712092f,   2.0294560760505993f,   2.0634026367482017f,
        2.0981611002741527f,   2.133785932225919f,    2.170336784741086f,
        2.2078803102947337f,   2.2464908293749546f,   2.286250990303635f,
        2.327254033532845f,    2.369604977942217f,    2.4134218838650208f,
        2.458840003415269f,    2.506014300608167f,    2.5551242195294983f,
        2.6063787537827645f,   2.660023038604595f,    2.716347847697055f,
        2.7757011083910723f,   2.838504606698991f,    2.9052776685316117f,
        2.976670770545963f,    3.0535115393558603f,   3.136880130166507f,
        3.2282236667414654f,   3.3295406612081644f,   3.443713971315384f,
        3.5751595986789093f,   3.7311414987004117f,   3.9249650523739246f,
        4.185630113705256f,    4.601871059539151f};

const float* lloyd_max_normal_half_centroids(size_t nbits) {
    switch (nbits) {
        case 1:
            return kLloydMaxNormalCentroids1;
        case 2:
            return kLloydMaxNormalCentroids2;
        case 3:
            return kLloydMaxNormalCentroids3;
        case 4:
            return kLloydMaxNormalCentroids4;
        case 5:
            return kLloydMaxNormalCentroids5;
        case 6:
            return kLloydMaxNormalCentroids6;
        case 7:
            return kLloydMaxNormalCentroids7;
        case 8:
            return kLloydMaxNormalCentroids8;
        default:
            FAISS_THROW_MSG("Lloyd-Max normal nbits must be in [1, 8]");
    }
}

std::vector<float> make_lloyd_max_normal_half_boundaries(size_t nbits) {
    const size_t count = size_t{1} << (nbits - 1);
    const float* centroids = lloyd_max_normal_half_centroids(nbits);
    std::vector<float> boundaries(count > 0 ? count - 1 : 0);
    for (size_t i = 0; i + 1 < count; i++) {
        boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
    }
    return boundaries;
}

std::vector<float> make_lloyd_max_normal_centroids(size_t nbits) {
    const size_t half_count = size_t{1} << (nbits - 1);
    const float* centroids = lloyd_max_normal_half_centroids(nbits);
    std::vector<float> full(half_count * 2);
    for (size_t i = 0; i < half_count; i++) {
        full[i] = -centroids[half_count - 1 - i];
        full[half_count + i] = centroids[i];
    }
    return full;
}

const std::vector<float>& lloyd_max_normal_centroids(size_t nbits) {
    static const std::vector<std::vector<float>> tables = {
            {},
            make_lloyd_max_normal_centroids(1),
            make_lloyd_max_normal_centroids(2),
            make_lloyd_max_normal_centroids(3),
            make_lloyd_max_normal_centroids(4),
            make_lloyd_max_normal_centroids(5),
            make_lloyd_max_normal_centroids(6),
            make_lloyd_max_normal_centroids(7),
            make_lloyd_max_normal_centroids(8),
    };
    FAISS_THROW_IF_NOT_MSG(
            nbits >= 1 && nbits <= 8,
            "Lloyd-Max normal nbits must be in [1, 8]");
    return tables[nbits];
}

const std::vector<float>& lloyd_max_normal_half_boundaries(size_t nbits) {
    static const std::vector<std::vector<float>> tables = {
            {},
            make_lloyd_max_normal_half_boundaries(1),
            make_lloyd_max_normal_half_boundaries(2),
            make_lloyd_max_normal_half_boundaries(3),
            make_lloyd_max_normal_half_boundaries(4),
            make_lloyd_max_normal_half_boundaries(5),
            make_lloyd_max_normal_half_boundaries(6),
            make_lloyd_max_normal_half_boundaries(7),
            make_lloyd_max_normal_half_boundaries(8),
    };
    FAISS_THROW_IF_NOT_MSG(
            nbits >= 1 && nbits <= 8,
            "Lloyd-Max normal nbits must be in [1, 8]");
    return tables[nbits];
}

constexpr size_t kTurboQuantMaxBits = 8;
// TurboQuant builds a 1-D optimal scalar quantizer analytically. We approximate
// the target density on a uniform grid over [-1, 1]; the grid is kept dense
// enough both in absolute terms and per output centroid.
constexpr size_t kTurboQuantGridMin = 1 << 15;
constexpr size_t kTurboQuantGridPerCentroid = 512;
constexpr int kTurboQuantMaxIter = 100;
constexpr double kTurboQuantTol = 1e-8;

void build_TurboQuantMSECodebook(
        size_t d,
        size_t nbits,
        std::vector<float>& centroids,
        std::vector<float>& boundaries) {
    FAISS_THROW_IF_NOT_FMT(
            nbits <= kTurboQuantMaxBits,
            "invalid TurboQuant nbits %zu (must be in [0, %zu])",
            nbits,
            kTurboQuantMaxBits);

    if (nbits == 0) {
        centroids.clear();
        boundaries.clear();
        return;
    }

    const size_t k = size_t(1) << nbits;

    if (d == 1) {
        // In 1-D, a unit vector can only be -1 or +1, so the marginal
        // distribution collapses to two atoms. The TurboQuant codebook is
        // therefore a repeated pair of endpoint centroids.
        centroids.resize(k);
        for (size_t i = 0; i < k; i++) {
            centroids[i] = i < k / 2 ? -1.0f : 1.0f;
        }
        boundaries.resize(k - 1);
        for (size_t i = 0; i + 1 < k; i++) {
            boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
        }
        return;
    }

    // For d > 1, TurboQuant uses the marginal distribution of one coordinate of
    // a random unit vector in R^d. On [-1, 1], this density is proportional to
    // (1 - x^2)^((d - 3) / 2), which is a symmetric beta-law after a change of
    // variables. The code below discretizes that density.
    const size_t ngrid =
            std::max(kTurboQuantGridMin, k * kTurboQuantGridPerCentroid);
    const double step = 2.0 / ngrid;
    const double alpha = 0.5 * (double(d) - 3.0);

    std::vector<double> xs(ngrid);
    // prefix_w stores the cumulative mass of the discretized density and
    // prefix_wx stores its cumulative first moment, so interval means can be
    // recovered in O(1).
    std::vector<double> prefix_w(ngrid + 1, 0.0);
    std::vector<double> prefix_wx(ngrid + 1, 0.0);

    for (size_t i = 0; i < ngrid; i++) {
        const double x = -1.0 + (i + 0.5) * step;
        const double one_minus_x2 = std::max(0.0, 1.0 - x * x);
        double w;
        if (alpha == 0.0) { // when d == 3
            w = 1.0;
        } else {
            // (1-x^2)^((d-3)/2)
            w = std::pow(one_minus_x2, alpha);
        }
        if (!std::isfinite(w) || w < 0.0) {
            w = 0.0;
        }
        xs[i] = x;
        prefix_w[i + 1] = prefix_w[i] + w;
        prefix_wx[i + 1] = prefix_wx[i] + w * x;
    }

    auto range_mean = [&](size_t i0, size_t i1, double fallback) {
        const double w = prefix_w[i1] - prefix_w[i0];
        if (w <= 0.0) {
            return fallback;
        }
        return (prefix_wx[i1] - prefix_wx[i0]) / w;
    };

    const double total_w = prefix_w.back();
    std::vector<size_t> cuts(k + 1, 0);
    cuts[k] = ngrid;

    // Initialize with k equal-mass cells under the target density. This gives
    // a stable starting point before the Lloyd refinements below.
    for (size_t i = 1; i < k; i++) {
        const double target = total_w * i / k;
        cuts[i] = std::lower_bound(prefix_w.begin(), prefix_w.end(), target) -
                prefix_w.begin();
        cuts[i] = std::min(cuts[i], ngrid);
    }

    std::vector<double> centroids_d(k);
    for (size_t i = 0; i < k; i++) {
        const double left = -1.0 + 2.0 * i / k;
        const double right = -1.0 + 2.0 * (i + 1) / k;
        // First estimate of each centroid: the conditional mean of its initial
        // equal-mass cell, with a uniform-cell midpoint as a fallback.
        centroids_d[i] = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
    }

    std::vector<double> boundaries_d(k > 0 ? k - 1 : 0);

    // Refine the 1-D codebook with a weighted Lloyd iteration over the
    // discretized marginal density on [-1, 1]:
    // 1. boundaries_d are the Voronoi separators implied by neighboring
    //    centroids.
    // 2. cuts map each boundary interval back to a contiguous range of the
    //    integration grid xs[].
    // 3. each centroid becomes the weighted mean of the samples currently in
    //    its cell, clipped to stay within its neighboring boundaries.
    //
    // The loop stops once the largest centroid update is below kTurboQuantTol.
    for (int iter = 0; iter < kTurboQuantMaxIter; iter++) {
        // Midpoints between adjacent centroids define the current Voronoi
        // partition of [-1, 1].
        for (size_t i = 0; i + 1 < k; i++) {
            boundaries_d[i] = 0.5 * (centroids_d[i] + centroids_d[i + 1]);
        }

        cuts[0] = 0;
        cuts[k] = ngrid;
        // Reassign the discretized density samples to the Voronoi cell induced
        // by each boundary. Because xs is sorted, the reassignment reduces to
        // finding the first grid point strictly greater than each boundary.
        for (size_t i = 1; i < k; i++) {
            cuts[i] = std::upper_bound(
                              xs.begin(), xs.end(), boundaries_d[i - 1]) -
                    xs.begin();
        }

        double max_delta = 0.0;
        for (size_t i = 0; i < k; i++) {
            const double left = i == 0 ? -1.0 : boundaries_d[i - 1];
            const double right = i + 1 == k ? 1.0 : boundaries_d[i];
            // Lloyd update: replace the centroid with the weighted average of
            // the mass assigned to its cell. Empty cells fall back to the cell
            // midpoint, and we clamp to [left, right] to preserve ordering.
            double c = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
            c = std::min(std::max(c, left), right);
            max_delta = std::max(max_delta, std::abs(c - centroids_d[i]));
            centroids_d[i] = c;
        }

        if (max_delta < kTurboQuantTol) {
            break;
        }
    }

    std::sort(centroids_d.begin(), centroids_d.end());

    centroids.resize(k);
    boundaries.resize(k - 1);
    for (size_t i = 0; i < k; i++) {
        centroids[i] = centroids_d[i];
    }
    for (size_t i = 0; i + 1 < k; i++) {
        boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
    }
}

void train_TurboQuantMSE(size_t d, size_t nbits, std::vector<float>& trained) {
    FAISS_THROW_IF_NOT_FMT(
            nbits > 0, "invalid TurboQuant SQ nbits %zu (must be > 0)", nbits);
    std::vector<float> centroids;
    std::vector<float> boundaries;
    build_TurboQuantMSECodebook(d, nbits, centroids, boundaries);
    const size_t k = centroids.size();

    trained.resize(k + (k - 1));
    for (size_t i = 0; i < k; i++) {
        trained[i] = centroids[i];
    }
    for (size_t i = 0; i + 1 < k; i++) {
        trained[k + i] = boundaries[i];
    }
}

void train_Uniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int k,
        const float* x,
        std::vector<float>& trained) {
    FAISS_THROW_IF_NOT(n > 0);
    trained.resize(2);
    float& vmin = trained[0];
    float& vmax = trained[1];

    if (rs == ScalarQuantizer::RS_minmax) {
        vmin = HUGE_VAL;
        vmax = -HUGE_VAL;
        for (idx_t i = 0; i < n; i++) {
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
        for (idx_t i = 0; i < n; i++) {
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
        idx_t o = static_cast<idx_t>(rs_arg * n);
        if (o < 0) {
            o = 0;
        }
        if (o > n - o) {
            o = n / 2;
        }
        std::nth_element(x_copy.begin(), x_copy.begin() + o, x_copy.end());
        vmin = x_copy[o];
        std::nth_element(
                x_copy.begin(), x_copy.begin() + (n - 1 - o), x_copy.end());
        vmax = x_copy[n - 1 - o];

    } else if (rs == ScalarQuantizer::RS_optim) {
        float a, b;
        float sx = 0;
        {
            vmin = HUGE_VAL, vmax = -HUGE_VAL;
            for (idx_t i = 0; i < n; i++) {
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
        for (idx_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (int j = 0; j < d; j++) {
                if (xi[j] < vmin[j]) {
                    vmin[j] = xi[j];
                }
                if (xi[j] > vmax[j]) {
                    vmax[j] = xi[j];
                }
            }
        }
        float* vdiff = vmax;
        for (int j = 0; j < d; j++) {
            float vexp = (vmax[j] - vmin[j]) * rs_arg;
            vmin[j] -= vexp;
            vmax[j] += vexp;
            vdiff[j] = vmax[j] - vmin[j];
        }
    } else {
        // transpose
        std::vector<float> xt(n * d);
        for (idx_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (int j = 0; j < d; j++) {
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
