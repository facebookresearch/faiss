/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * implementation of the index_factory function. Lots of regex parsing code.
 */

#include <faiss/index_factory.h>

#include <map>

#include <regex>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/random.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexLattice.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexRowwiseMinMax.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/VectorTransform.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexBinaryIVF.h>
#include <string>

namespace faiss {

/***************************************************************
 * index_factory
 ***************************************************************/

int index_factory_verbose = 0;

namespace {

/***************************************************************
 * Small functions
 */

bool re_match(const std::string& s, const std::string& pat, std::smatch& sm) {
    return std::regex_match(s, sm, std::regex(pat));
}

// find first pair of matching parentheses
void find_matching_parentheses(
        const std::string& s,
        int& i0,
        int& i1,
        int begin = 0) {
    int st = 0;
    i0 = i1 = 0;
    for (int i = begin; i < s.length(); i++) {
        if (s[i] == '(') {
            if (st == 0) {
                i0 = i;
            }
            st++;
        }

        if (s[i] == ')') {
            st--;
            if (st == 0) {
                i1 = i;
                return;
            }
            if (st < 0) {
                FAISS_THROW_FMT(
                        "factory string %s: unbalanced parentheses", s.c_str());
            }
        }
    }
    FAISS_THROW_FMT(
            "factory string %s: unbalanced parentheses st=%d", s.c_str(), st);
}

/// what kind of training does this coarse quantizer require?
char get_trains_alone(const Index* coarse_quantizer) {
    if (dynamic_cast<const IndexFlat*>(coarse_quantizer)) {
        return 0;
    }
    // multi index just needs to be quantized
    if (dynamic_cast<const MultiIndexQuantizer*>(coarse_quantizer) ||
        dynamic_cast<const ResidualCoarseQuantizer*>(coarse_quantizer)) {
        return 1;
    }
    if (dynamic_cast<const IndexHNSWFlat*>(coarse_quantizer)) {
        return 2;
    }
    return 2; // for complicated indexes, we assume they can't be used as a
              // kmeans index
}

// set the fields for factory-constructed IVF structures
IndexIVF* fix_ivf_fields(IndexIVF* index_ivf) {
    index_ivf->quantizer_trains_alone = get_trains_alone(index_ivf->quantizer);
    index_ivf->cp.spherical = index_ivf->metric_type == METRIC_INNER_PRODUCT;
    index_ivf->own_fields = true;
    return index_ivf;
}

int mres_to_int(const std::ssub_match& mr, int deflt = -1, int begin = 0) {
    if (mr.length() == 0) {
        return deflt;
    }
    return std::stoi(mr.str().substr(begin));
}

std::map<std::string, ScalarQuantizer::QuantizerType> sq_types = {
        {"SQ8", ScalarQuantizer::QT_8bit},
        {"SQ4", ScalarQuantizer::QT_4bit},
        {"SQ6", ScalarQuantizer::QT_6bit},
        {"SQfp16", ScalarQuantizer::QT_fp16},
        {"SQbf16", ScalarQuantizer::QT_bf16},
        {"SQ8_direct_signed", ScalarQuantizer::QT_8bit_direct_signed},
        {"SQ8_direct", ScalarQuantizer::QT_8bit_direct},
};
const std::string sq_pattern =
        "(SQ4|SQ8|SQ6|SQfp16|SQbf16|SQ8_direct_signed|SQ8_direct)";

std::map<std::string, AdditiveQuantizer::Search_type_t> aq_search_type = {
        {"_Nfloat", AdditiveQuantizer::ST_norm_float},
        {"_Nnone", AdditiveQuantizer::ST_LUT_nonorm},
        {"_Nqint8", AdditiveQuantizer::ST_norm_qint8},
        {"_Nqint4", AdditiveQuantizer::ST_norm_qint4},
        {"_Ncqint8", AdditiveQuantizer::ST_norm_cqint8},
        {"_Ncqint4", AdditiveQuantizer::ST_norm_cqint4},
        {"_Nlsq2x4", AdditiveQuantizer::ST_norm_lsq2x4},
        {"_Nrq2x4", AdditiveQuantizer::ST_norm_rq2x4},
};

const std::string aq_def_pattern = "[0-9]+x[0-9]+(_[0-9]+x[0-9]+)*";
const std::string aq_norm_pattern =
        "(|_Nnone|_Nfloat|_Nqint8|_Nqint4|_Ncqint8|_Ncqint4|_Nlsq2x4|_Nrq2x4)";

const std::string paq_def_pattern = "([0-9]+)x([0-9]+)x([0-9]+)";

AdditiveQuantizer::Search_type_t aq_parse_search_type(
        const std::string& stok,
        MetricType metric) {
    if (stok == "") {
        return metric == METRIC_L2 ? AdditiveQuantizer::ST_decompress
                                   : AdditiveQuantizer::ST_LUT_nonorm;
    }
    int pos = stok.rfind("_");
    return aq_search_type[stok.substr(pos)];
}

std::vector<size_t> aq_parse_nbits(std::string stok) {
    std::vector<size_t> nbits;
    std::smatch sm;
    while (std::regex_search(stok, sm, std::regex("[^q]([0-9]+)x([0-9]+)"))) {
        int M = std::stoi(sm[1].str());
        int nbit = std::stoi(sm[2].str());
        nbits.resize(nbits.size() + M, nbit);
        stok = sm.suffix();
    }
    return nbits;
}

/***************************************************************
 * Parse VectorTransform
 */

VectorTransform* parse_VectorTransform(const std::string& description, int d) {
    std::smatch sm;
    auto match = [&sm, description](std::string pattern) {
        return re_match(description, pattern, sm);
    };
    if (match("PCA(W?)(R?)([0-9]+)")) {
        bool white = sm[1].length() > 0;
        bool rot = sm[2].length() > 0;
        return new PCAMatrix(d, std::stoi(sm[3].str()), white ? -0.5 : 0, rot);
    }
    if (match("L2[nN]orm")) {
        return new NormalizationTransform(d, 2.0);
    }
    if (match("RR([0-9]+)?")) {
        return new RandomRotationMatrix(d, mres_to_int(sm[1], d));
    }
    if (match("ITQ([0-9]+)?")) {
        return new ITQTransform(d, mres_to_int(sm[1], d), sm[1].length() > 0);
    }
    if (match("OPQ([0-9]+)(_[0-9]+)?")) {
        int M = std::stoi(sm[1].str());
        int d_out = mres_to_int(sm[2], d, 1);
        return new OPQMatrix(d, M, d_out);
    }
    if (match("Pad([0-9]+)")) {
        int d_out = std::stoi(sm[1].str());
        return new RemapDimensionsTransform(d, std::max(d_out, d), false);
    }
    return nullptr;
}

/***************************************************************
 * Parse IndexIVF
 */

size_t parse_nlist(std::string s) {
    size_t multiplier = 1;
    if (s.back() == 'k') {
        s.pop_back();
        multiplier = 1024;
    }
    if (s.back() == 'M') {
        s.pop_back();
        multiplier = 1024 * 1024;
    }
    return std::stoi(s) * multiplier;
}

// parsing guard + function
Index* parse_coarse_quantizer(
        const std::string& description,
        int d,
        MetricType mt,
        std::vector<std::unique_ptr<Index>>& parenthesis_indexes,
        size_t& nlist,
        bool& use_2layer) {
    std::smatch sm;
    auto match = [&sm, description](std::string pattern) {
        return re_match(description, pattern, sm);
    };
    use_2layer = false;

    if (match("IVF([0-9]+[kM]?)")) {
        nlist = parse_nlist(sm[1].str());
        return new IndexFlat(d, mt);
    }
    if (match("IMI2x([0-9]+)")) {
        int nbit = std::stoi(sm[1].str());
        FAISS_THROW_IF_NOT_MSG(
                mt == METRIC_L2,
                "MultiIndex not implemented for inner prod search");
        nlist = (size_t)1 << (2 * nbit);
        return new MultiIndexQuantizer(d, 2, nbit);
    }
    if (match("IVF([0-9]+[kM]?)_HNSW([0-9]*)")) {
        nlist = parse_nlist(sm[1].str());
        int hnsw_M = sm[2].length() > 0 ? std::stoi(sm[2]) : 32;
        return new IndexHNSWFlat(d, hnsw_M, mt);
    }
    if (match("IVF([0-9]+[kM]?)_NSG([0-9]+)")) {
        nlist = parse_nlist(sm[1].str());
        int R = std::stoi(sm[2]);
        return new IndexNSGFlat(d, R, mt);
    }
    if (match("IVF([0-9]+[kM]?)\\(Index([0-9])\\)")) {
        nlist = parse_nlist(sm[1].str());
        int no = std::stoi(sm[2].str());
        FAISS_ASSERT(no >= 0 && no < parenthesis_indexes.size());
        return parenthesis_indexes[no].release();
    }

    // these two generate Index2Layer's not IndexIVF's
    if (match("Residual([0-9]+)x([0-9]+)")) {
        FAISS_THROW_IF_NOT_MSG(
                mt == METRIC_L2,
                "MultiIndex not implemented for inner prod search");
        int M = mres_to_int(sm[1]), nbit = mres_to_int(sm[2]);
        nlist = (size_t)1 << (M * nbit);
        use_2layer = true;
        return new MultiIndexQuantizer(d, M, nbit);
    }
    if (match("Residual([0-9]+)")) {
        FAISS_THROW_IF_NOT_MSG(
                mt == METRIC_L2,
                "Residual not implemented for inner prod search");
        use_2layer = true;
        nlist = mres_to_int(sm[1]);
        return new IndexFlatL2(d);
    }
    return nullptr;
}

// parse the code description part of the IVF description

IndexIVF* parse_IndexIVF(
        const std::string& code_string,
        std::unique_ptr<Index>& quantizer,
        size_t nlist,
        MetricType mt) {
    std::smatch sm;
    auto match = [&sm, &code_string](const std::string pattern) {
        return re_match(code_string, pattern, sm);
    };
    auto get_q = [&quantizer] { return quantizer.release(); };
    int d = quantizer->d;

    if (match("Flat")) {
        return new IndexIVFFlat(get_q(), d, nlist, mt);
    }
    if (match("FlatDedup")) {
        return new IndexIVFFlatDedup(get_q(), d, nlist, mt);
    }
    if (match(sq_pattern)) {
        return new IndexIVFScalarQuantizer(
                get_q(), d, nlist, sq_types[sm[1].str()], mt);
    }
    if (match("PQ([0-9]+)(x[0-9]+)?(np)?")) {
        int M = mres_to_int(sm[1]), nbit = mres_to_int(sm[2], 8, 1);
        IndexIVFPQ* index_ivf = new IndexIVFPQ(get_q(), d, nlist, M, nbit, mt);
        index_ivf->do_polysemous_training = sm[3].str() != "np";
        return index_ivf;
    }
    if (match("PQ([0-9]+)\\+([0-9]+)")) {
        FAISS_THROW_IF_NOT_MSG(
                mt == METRIC_L2,
                "IVFPQR not implemented for inner product search");
        int M1 = mres_to_int(sm[1]), M2 = mres_to_int(sm[2]);
        return new IndexIVFPQR(get_q(), d, nlist, M1, 8, M2, 8);
    }
    if (match("PQ([0-9]+)x4fs(r?)(_[0-9]+)?")) {
        int M = mres_to_int(sm[1]);
        int bbs = mres_to_int(sm[3], 32, 1);
        IndexIVFPQFastScan* index_ivf =
                new IndexIVFPQFastScan(get_q(), d, nlist, M, 4, mt, bbs);
        index_ivf->by_residual = sm[2].str() == "r";
        return index_ivf;
    }
    if (match("(RQ|LSQ)" + aq_def_pattern + aq_norm_pattern)) {
        std::vector<size_t> nbits = aq_parse_nbits(sm.str());
        AdditiveQuantizer::Search_type_t st =
                aq_parse_search_type(sm[sm.size() - 1].str(), mt);
        IndexIVF* index_ivf;
        if (sm[1].str() == "RQ") {
            index_ivf = new IndexIVFResidualQuantizer(
                    get_q(), d, nlist, nbits, mt, st);
        } else {
            FAISS_THROW_IF_NOT(nbits.size() > 0);
            index_ivf = new IndexIVFLocalSearchQuantizer(
                    get_q(), d, nlist, nbits.size(), nbits[0], mt, st);
        }
        return index_ivf;
    }
    if (match("(PRQ|PLSQ)" + paq_def_pattern + aq_norm_pattern)) {
        int nsplits = mres_to_int(sm[2]);
        int Msub = mres_to_int(sm[3]);
        int nbit = mres_to_int(sm[4]);
        auto st = aq_parse_search_type(sm[sm.size() - 1].str(), mt);
        IndexIVF* index_ivf;
        if (sm[1].str() == "PRQ") {
            index_ivf = new IndexIVFProductResidualQuantizer(
                    get_q(), d, nlist, nsplits, Msub, nbit, mt, st);
        } else {
            index_ivf = new IndexIVFProductLocalSearchQuantizer(
                    get_q(), d, nlist, nsplits, Msub, nbit, mt, st);
        }
        return index_ivf;
    }
    if (match("(RQ|LSQ)([0-9]+)x4fs(r?)(_[0-9]+)?" + aq_norm_pattern)) {
        int M = std::stoi(sm[2].str());
        int bbs = mres_to_int(sm[4], 32, 1);
        auto st = aq_parse_search_type(sm[sm.size() - 1].str(), mt);
        IndexIVFAdditiveQuantizerFastScan* index_ivf;
        if (sm[1].str() == "RQ") {
            index_ivf = new IndexIVFResidualQuantizerFastScan(
                    get_q(), d, nlist, M, 4, mt, st, bbs);
        } else {
            index_ivf = new IndexIVFLocalSearchQuantizerFastScan(
                    get_q(), d, nlist, M, 4, mt, st, bbs);
        }
        index_ivf->by_residual = (sm[3].str() == "r");
        return index_ivf;
    }
    if (match("(PRQ|PLSQ)([0-9]+)x([0-9]+)x4fs(r?)(_[0-9]+)?" +
              aq_norm_pattern)) {
        int nsplits = std::stoi(sm[2].str());
        int Msub = std::stoi(sm[3].str());
        int bbs = mres_to_int(sm[5], 32, 1);
        auto st = aq_parse_search_type(sm[sm.size() - 1].str(), mt);
        IndexIVFAdditiveQuantizerFastScan* index_ivf;
        if (sm[1].str() == "PRQ") {
            index_ivf = new IndexIVFProductResidualQuantizerFastScan(
                    get_q(), d, nlist, nsplits, Msub, 4, mt, st, bbs);
        } else {
            index_ivf = new IndexIVFProductLocalSearchQuantizerFastScan(
                    get_q(), d, nlist, nsplits, Msub, 4, mt, st, bbs);
        }
        index_ivf->by_residual = (sm[4].str() == "r");
        return index_ivf;
    }
    if (match("(ITQ|PCA|PCAR)([0-9]+)?,SH([-0-9.e]+)?([gcm])?")) {
        int outdim = mres_to_int(sm[2], d); // is also the number of bits
        std::unique_ptr<VectorTransform> vt;
        if (sm[1] == "ITQ") {
            vt.reset(new ITQTransform(d, outdim, d != outdim));
        } else if (sm[1] == "PCA") {
            vt.reset(new PCAMatrix(d, outdim));
        } else if (sm[1] == "PCAR") {
            vt.reset(new PCAMatrix(d, outdim, 0, true));
        }
        // the rationale for -1e10 is that this corresponds to simple
        // thresholding
        float period = sm[3].length() > 0 ? std::stof(sm[3]) : -1e10;
        IndexIVFSpectralHash* index_ivf =
                new IndexIVFSpectralHash(get_q(), d, nlist, outdim, period);
        index_ivf->replace_vt(vt.release(), true);
        if (sm[4].length()) {
            std::string s = sm[4].str();
            index_ivf->threshold_type = s == "g"
                    ? IndexIVFSpectralHash::Thresh_global
                    : s == "c"
                    ? IndexIVFSpectralHash::Thresh_centroid
                    :
                    /* s == "m" ? */ IndexIVFSpectralHash::Thresh_median;
        }
        return index_ivf;
    }
    return nullptr;
}

/***************************************************************
 * Parse IndexHNSW
 */

IndexHNSW* parse_IndexHNSW(
        const std::string code_string,
        int d,
        MetricType mt,
        int hnsw_M) {
    std::smatch sm;
    auto match = [&sm, &code_string](const std::string& pattern) {
        return re_match(code_string, pattern, sm);
    };

    if (match("Flat|")) {
        return new IndexHNSWFlat(d, hnsw_M, mt);
    }

    if (match("PQ([0-9]+)(x[0-9]+)?(np)?")) {
        int M = std::stoi(sm[1].str());
        int nbit = mres_to_int(sm[2], 8, 1);
        IndexHNSWPQ* ipq = new IndexHNSWPQ(d, M, hnsw_M, nbit);
        dynamic_cast<IndexPQ*>(ipq->storage)->do_polysemous_training =
                sm[3].str() != "np";
        return ipq;
    }
    if (match(sq_pattern)) {
        return new IndexHNSWSQ(d, sq_types[sm[1].str()], hnsw_M, mt);
    }
    if (match("([0-9]+)\\+PQ([0-9]+)?")) {
        int ncent = mres_to_int(sm[1]);
        int pq_m = mres_to_int(sm[2]);
        IndexHNSW2Level* hidx2l =
                new IndexHNSW2Level(new IndexFlatL2(d), ncent, pq_m, hnsw_M);
        dynamic_cast<Index2Layer*>(hidx2l->storage)->q1.own_fields = true;
        return hidx2l;
    }
    if (match("2x([0-9]+)\\+PQ([0-9]+)?")) {
        int nbit = mres_to_int(sm[1]);
        int pq_m = mres_to_int(sm[2]);
        Index* quant = new MultiIndexQuantizer(d, 2, nbit);
        IndexHNSW2Level* hidx2l = new IndexHNSW2Level(
                quant, (size_t)1 << (2 * nbit), pq_m, hnsw_M);
        Index2Layer* idx2l = dynamic_cast<Index2Layer*>(hidx2l->storage);
        idx2l->q1.own_fields = true;
        idx2l->q1.quantizer_trains_alone = 1;
        return hidx2l;
    }

    return nullptr;
}

/***************************************************************
 * Parse IndexNSG
 */

IndexNSG* parse_IndexNSG(
        const std::string code_string,
        int d,
        MetricType mt,
        int nsg_R) {
    std::smatch sm;
    auto match = [&sm, &code_string](const std::string& pattern) {
        return re_match(code_string, pattern, sm);
    };

    if (match("Flat|")) {
        return new IndexNSGFlat(d, nsg_R, mt);
    }
    if (match("PQ([0-9]+)(x[0-9]+)?(np)?")) {
        int M = std::stoi(sm[1].str());
        int nbit = mres_to_int(sm[2], 8, 1);
        IndexNSGPQ* ipq = new IndexNSGPQ(d, M, nsg_R, nbit);
        dynamic_cast<IndexPQ*>(ipq->storage)->do_polysemous_training =
                sm[3].str() != "np";
        return ipq;
    }
    if (match(sq_pattern)) {
        return new IndexNSGSQ(d, sq_types[sm[1].str()], nsg_R, mt);
    }

    return nullptr;
}

/***************************************************************
 * Parse basic indexes
 */

Index* parse_other_indexes(
        const std::string& description,
        int d,
        MetricType metric) {
    std::smatch sm;
    auto match = [&sm, description](const std::string& pattern) {
        return re_match(description, pattern, sm);
    };

    // IndexFlat
    if (description == "Flat") {
        return new IndexFlat(d, metric);
    }

    // IndexLSH
    if (match("LSH([0-9]*)(r?)(t?)")) {
        int nbits = sm[1].length() > 0 ? std::stoi(sm[1].str()) : d;
        bool rotate_data = sm[2].length() > 0;
        bool train_thresholds = sm[3].length() > 0;
        FAISS_THROW_IF_NOT(metric == METRIC_L2);
        return new IndexLSH(d, nbits, rotate_data, train_thresholds);
    }

    // IndexLattice
    if (match("ZnLattice([0-9]+)x([0-9]+)_([0-9]+)")) {
        int M = std::stoi(sm[1].str()), r2 = std::stoi(sm[2].str());
        int nbit = std::stoi(sm[3].str());
        return new IndexLattice(d, M, nbit, r2);
    }

    // IndexScalarQuantizer
    if (match(sq_pattern)) {
        return new IndexScalarQuantizer(d, sq_types[description], metric);
    }

    // IndexPQ
    if (match("PQ([0-9]+)(x[0-9]+)?(np)?")) {
        int M = std::stoi(sm[1].str());
        int nbit = mres_to_int(sm[2], 8, 1);
        IndexPQ* index_pq = new IndexPQ(d, M, nbit, metric);
        index_pq->do_polysemous_training = sm[3].str() != "np";
        return index_pq;
    }

    // IndexPQFastScan
    if (match("PQ([0-9]+)x4fs(_[0-9]+)?")) {
        int M = std::stoi(sm[1].str());
        int bbs = mres_to_int(sm[2], 32, 1);
        return new IndexPQFastScan(d, M, 4, metric, bbs);
    }

    // IndexResidualCoarseQuantizer and IndexResidualQuantizer
    std::string pattern = "(RQ|RCQ)" + aq_def_pattern + aq_norm_pattern;
    if (match(pattern)) {
        std::vector<size_t> nbits = aq_parse_nbits(description);
        if (sm[1].str() == "RCQ") {
            return new ResidualCoarseQuantizer(d, nbits, metric);
        }
        AdditiveQuantizer::Search_type_t st =
                aq_parse_search_type(sm[sm.size() - 1].str(), metric);
        return new IndexResidualQuantizer(d, nbits, metric, st);
    }

    // LocalSearchCoarseQuantizer and IndexLocalSearchQuantizer
    if (match("(LSQ|LSCQ)([0-9]+)x([0-9]+)" + aq_norm_pattern)) {
        std::vector<size_t> nbits = aq_parse_nbits(description);
        int M = mres_to_int(sm[2]);
        int nbit = mres_to_int(sm[3]);
        if (sm[1].str() == "LSCQ") {
            return new LocalSearchCoarseQuantizer(d, M, nbit, metric);
        }
        AdditiveQuantizer::Search_type_t st =
                aq_parse_search_type(sm[sm.size() - 1].str(), metric);
        return new IndexLocalSearchQuantizer(d, M, nbit, metric, st);
    }

    // IndexProductResidualQuantizer
    if (match("PRQ" + paq_def_pattern + aq_norm_pattern)) {
        int nsplits = mres_to_int(sm[1]);
        int Msub = mres_to_int(sm[2]);
        int nbit = mres_to_int(sm[3]);
        auto st = aq_parse_search_type(sm[sm.size() - 1].str(), metric);
        return new IndexProductResidualQuantizer(
                d, nsplits, Msub, nbit, metric, st);
    }

    // IndexProductLocalSearchQuantizer
    if (match("PLSQ" + paq_def_pattern + aq_norm_pattern)) {
        int nsplits = mres_to_int(sm[1]);
        int Msub = mres_to_int(sm[2]);
        int nbit = mres_to_int(sm[3]);
        auto st = aq_parse_search_type(sm[sm.size() - 1].str(), metric);
        return new IndexProductLocalSearchQuantizer(
                d, nsplits, Msub, nbit, metric, st);
    }

    // IndexAdditiveQuantizerFastScan
    // RQ{M}x4fs_{bbs}_{search_type}
    pattern = "(LSQ|RQ)([0-9]+)x4fs(_[0-9]+)?" + aq_norm_pattern;
    if (match(pattern)) {
        int M = std::stoi(sm[2].str());
        int bbs = mres_to_int(sm[3], 32, 1);
        auto st = aq_parse_search_type(sm[sm.size() - 1].str(), metric);

        if (sm[1].str() == "RQ") {
            return new IndexResidualQuantizerFastScan(d, M, 4, metric, st, bbs);
        } else if (sm[1].str() == "LSQ") {
            return new IndexLocalSearchQuantizerFastScan(
                    d, M, 4, metric, st, bbs);
        }
    }

    // IndexProductAdditiveQuantizerFastScan
    // PRQ{nsplits}x{Msub}x4fs_{bbs}_{search_type}
    pattern = "(PLSQ|PRQ)([0-9]+)x([0-9]+)x4fs(_[0-9]+)?" + aq_norm_pattern;
    if (match(pattern)) {
        int nsplits = std::stoi(sm[2].str());
        int Msub = std::stoi(sm[3].str());
        int bbs = mres_to_int(sm[4], 32, 1);
        auto st = aq_parse_search_type(sm[sm.size() - 1].str(), metric);

        if (sm[1].str() == "PRQ") {
            return new IndexProductResidualQuantizerFastScan(
                    d, nsplits, Msub, 4, metric, st, bbs);
        } else if (sm[1].str() == "PLSQ") {
            return new IndexProductLocalSearchQuantizerFastScan(
                    d, nsplits, Msub, 4, metric, st, bbs);
        }
    }

    return nullptr;
}

/***************************************************************
 * Driver function
 */
std::unique_ptr<Index> index_factory_sub(
        int d,
        std::string description,
        MetricType metric) {
    // handle composite indexes

    bool verbose = index_factory_verbose;

    if (verbose) {
        printf("begin parse VectorTransforms: %s \n", description.c_str());
    }

    // for the current match
    std::smatch sm;

    // IndexIDMap -- it turns out is was used both as a prefix and a suffix, so
    // support both
    if (re_match(description, "(.+),IDMap2", sm) ||
        re_match(description, "IDMap2,(.+)", sm)) {
        IndexIDMap2* idmap2 = new IndexIDMap2(
                index_factory_sub(d, sm[1].str(), metric).release());
        idmap2->own_fields = true;
        return std::unique_ptr<Index>(idmap2);
    }

    if (re_match(description, "(.+),IDMap", sm) ||
        re_match(description, "IDMap,(.+)", sm)) {
        IndexIDMap* idmap = new IndexIDMap(
                index_factory_sub(d, sm[1].str(), metric).release());
        idmap->own_fields = true;
        return std::unique_ptr<Index>(idmap);
    }

    // handle refines
    if (re_match(description, "(.+),RFlat", sm) ||
        re_match(description, "(.+),Refine\\((.+)\\)", sm)) {
        std::unique_ptr<Index> filter_index =
                index_factory_sub(d, sm[1].str(), metric);

        IndexRefine* index_rf = nullptr;
        if (sm.size() == 3) { // Refine
            std::unique_ptr<Index> refine_index =
                    index_factory_sub(d, sm[2].str(), metric);
            index_rf = new IndexRefine(
                    filter_index.release(), refine_index.release());
            index_rf->own_refine_index = true;
        } else { // RFlat
            index_rf = new IndexRefineFlat(filter_index.release(), nullptr);
        }
        FAISS_ASSERT(index_rf != nullptr);
        index_rf->own_fields = true;
        return std::unique_ptr<Index>(index_rf);
    }

    // IndexPreTransform
    // should handle this first (even before parentheses) because it changes d
    std::vector<std::unique_ptr<VectorTransform>> vts;
    VectorTransform* vt = nullptr;
    while (re_match(description, "([^,]+),(.*)", sm) &&
           (vt = parse_VectorTransform(sm[1], d))) {
        // reset loop
        description = sm[sm.size() - 1];
        vts.emplace_back(vt);
        d = vts.back()->d_out;
    }

    if (vts.size() > 0) {
        std::unique_ptr<Index> sub_index =
                index_factory_sub(d, description, metric);
        IndexPreTransform* index_pt = new IndexPreTransform(sub_index.get());
        std::unique_ptr<Index> ret(index_pt);
        index_pt->own_fields = true;
        sub_index.release();
        while (vts.size() > 0) {
            if (verbose) {
                printf("prepend trans %d -> %d\n",
                       vts.back()->d_in,
                       vts.back()->d_out);
            }
            index_pt->prepend_transform(vts.back().release());
            vts.pop_back();
        }
        return ret;
    }

    // what we got from the parentheses
    std::vector<std::unique_ptr<Index>> parenthesis_indexes;

    int begin = 0;
    while (description.find('(', begin) != std::string::npos) {
        // replace indexes in () with Index0, Index1, etc.
        int i0, i1;
        find_matching_parentheses(description, i0, i1, begin);
        std::string sub_description = description.substr(i0 + 1, i1 - i0 - 1);
        int no = parenthesis_indexes.size();
        parenthesis_indexes.push_back(
                index_factory_sub(d, sub_description, metric));
        description = description.substr(0, i0 + 1) + "Index" +
                std::to_string(no) + description.substr(i1);
        begin = i1 + 1;
    }

    if (verbose) {
        printf("after () normalization: %s %zd parenthesis indexes d=%d\n",
               description.c_str(),
               parenthesis_indexes.size(),
               d);
    }

    { // handle basic index types
        Index* index = parse_other_indexes(description, d, metric);
        if (index) {
            return std::unique_ptr<Index>(index);
        }
    }

    // HNSW variants (it was unclear in the old version that the separator was a
    // "," so we support both "_" and ",")
    if (re_match(description, "HNSW([0-9]*)([,_].*)?", sm)) {
        int hnsw_M = mres_to_int(sm[1], 32);
        // We also accept empty code string (synonym of Flat)
        std::string code_string =
                sm[2].length() > 0 ? sm[2].str().substr(1) : "";
        if (verbose) {
            printf("parsing HNSW string %s code_string=%s hnsw_M=%d\n",
                   description.c_str(),
                   code_string.c_str(),
                   hnsw_M);
        }

        IndexHNSW* index = parse_IndexHNSW(code_string, d, metric, hnsw_M);
        FAISS_THROW_IF_NOT_FMT(
                index,
                "could not parse HNSW code description %s in %s",
                code_string.c_str(),
                description.c_str());
        return std::unique_ptr<Index>(index);
    }

    // NSG variants (it was unclear in the old version that the separator was a
    // "," so we support both "_" and ",")
    if (re_match(description, "NSG([0-9]*)([,_].*)?", sm)) {
        int nsg_R = mres_to_int(sm[1], 32);
        // We also accept empty code string (synonym of Flat)
        std::string code_string =
                sm[2].length() > 0 ? sm[2].str().substr(1) : "";
        if (verbose) {
            printf("parsing NSG string %s code_string=%s nsg_R=%d\n",
                   description.c_str(),
                   code_string.c_str(),
                   nsg_R);
        }

        IndexNSG* index = parse_IndexNSG(code_string, d, metric, nsg_R);
        FAISS_THROW_IF_NOT_FMT(
                index,
                "could not parse NSG code description %s in %s",
                code_string.c_str(),
                description.c_str());
        return std::unique_ptr<Index>(index);
    }

    // IndexRowwiseMinMax, fp32 version
    if (description.compare(0, 7, "MinMax,") == 0) {
        size_t comma = description.find(",");
        std::string sub_index_string = description.substr(comma + 1);
        auto sub_index = index_factory_sub(d, sub_index_string, metric);

        auto index = new IndexRowwiseMinMax(sub_index.release());
        index->own_fields = true;

        return std::unique_ptr<Index>(index);
    }

    // IndexRowwiseMinMax, fp16 version
    if (description.compare(0, 11, "MinMaxFP16,") == 0) {
        size_t comma = description.find(",");
        std::string sub_index_string = description.substr(comma + 1);
        auto sub_index = index_factory_sub(d, sub_index_string, metric);

        auto index = new IndexRowwiseMinMaxFP16(sub_index.release());
        index->own_fields = true;

        return std::unique_ptr<Index>(index);
    }

    // IndexIVF
    {
        size_t nlist;
        bool use_2layer;
        size_t comma = description.find(",");
        std::string coarse_string = description.substr(0, comma);
        // Match coarse quantizer part first
        std::unique_ptr<Index> quantizer(parse_coarse_quantizer(
                description.substr(0, comma),
                d,
                metric,
                parenthesis_indexes,
                nlist,
                use_2layer));

        if (comma != std::string::npos && quantizer.get()) {
            std::string code_description = description.substr(comma + 1);
            if (use_2layer) {
                bool ok =
                        re_match(code_description, "PQ([0-9]+)(x[0-9]+)?", sm);
                FAISS_THROW_IF_NOT_FMT(
                        ok,
                        "could not parse 2 layer code description %s in %s",
                        code_description.c_str(),
                        description.c_str());
                int M = std::stoi(sm[1].str()), nbit = mres_to_int(sm[2], 8, 1);
                Index2Layer* index_2l =
                        new Index2Layer(quantizer.release(), nlist, M, nbit);
                index_2l->q1.own_fields = true;
                index_2l->q1.quantizer_trains_alone =
                        get_trains_alone(index_2l->q1.quantizer);
                return std::unique_ptr<Index>(index_2l);
            }

            IndexIVF* index_ivf =
                    parse_IndexIVF(code_description, quantizer, nlist, metric);

            FAISS_THROW_IF_NOT_FMT(
                    index_ivf,
                    "could not parse code description %s in %s",
                    code_description.c_str(),
                    description.c_str());
            return std::unique_ptr<Index>(fix_ivf_fields(index_ivf));
        }
    }
    FAISS_THROW_FMT("could not parse index string %s", description.c_str());
    return nullptr;
}

} // anonymous namespace

Index* index_factory(int d, const char* description, MetricType metric) {
    return index_factory_sub(d, description, metric).release();
}

IndexBinary* index_binary_factory(int d, const char* description) {
    IndexBinary* index = nullptr;

    int ncentroids = -1;
    int M, nhash, b;

    if (sscanf(description, "BIVF%d_HNSW%d", &ncentroids, &M) == 2) {
        IndexBinaryIVF* index_ivf =
                new IndexBinaryIVF(new IndexBinaryHNSW(d, M), d, ncentroids);
        index_ivf->own_fields = true;
        index = index_ivf;

    } else if (sscanf(description, "BIVF%d", &ncentroids) == 1) {
        IndexBinaryIVF* index_ivf =
                new IndexBinaryIVF(new IndexBinaryFlat(d), d, ncentroids);
        index_ivf->own_fields = true;
        index = index_ivf;

    } else if (sscanf(description, "BHNSW%d", &M) == 1) {
        IndexBinaryHNSW* index_hnsw = new IndexBinaryHNSW(d, M);
        index = index_hnsw;

    } else if (sscanf(description, "BHash%dx%d", &nhash, &b) == 2) {
        index = new IndexBinaryMultiHash(d, nhash, b);

    } else if (sscanf(description, "BHash%d", &b) == 1) {
        index = new IndexBinaryHash(d, b);

    } else if (std::string(description) == "BFlat") {
        index = new IndexBinaryFlat(d);

    } else {
        FAISS_THROW_IF_NOT_FMT(
                index, "description %s did not generate an index", description);
    }

    return index;
}

} // namespace faiss
