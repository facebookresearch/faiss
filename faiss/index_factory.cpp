/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/*
 * implementation of Hyper-parameter auto-tuning
 */

#include <faiss/index_factory.h>

#include <faiss/AutoTune.h>

#include <cinttypes>
#include <cmath>

#include <regex>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexLattice.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexResidual.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/VectorTransform.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexBinaryIVF.h>

namespace faiss {

/***************************************************************
 * index_factory
 ***************************************************************/

namespace {

struct VTChain {
    std::vector<VectorTransform*> chain;
    ~VTChain() {
        for (int i = 0; i < chain.size(); i++) {
            delete chain[i];
        }
    }
};

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

bool str_ends_with(const std::string& s, const std::string& suffix) {
    return s.rfind(suffix) == std::abs(int(s.size() - suffix.size()));
}

// check if ends with suffix followed by digits
bool str_ends_with_digits(const std::string& s, const std::string& suffix) {
    int i;
    for (i = s.length() - 1; i >= 0; i--) {
        if (!isdigit(s[i]))
            break;
    }
    return str_ends_with(s.substr(0, i + 1), suffix);
}

void find_matching_parentheses(const std::string& s, int& i0, int& i1) {
    int st = 0;
    i0 = i1 = 0;
    for (int i = 0; i < s.length(); i++) {
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

} // anonymous namespace

Index* index_factory(int d, const char* description_in, MetricType metric) {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
    VTChain vts;
    Index* coarse_quantizer = nullptr;
    std::string parenthesis_ivf, parenthesis_refine;
    Index* index = nullptr;
    bool add_idmap = false;
    int d_in = d;

    ScopeDeleter1<Index> del_coarse_quantizer, del_index;

    std::string description(description_in);
    char* ptr;

    // handle indexes in parentheses
    while (description.find('(') != std::string::npos) {
        // then we make a sub-index and remove the () from the description
        int i0, i1;
        find_matching_parentheses(description, i0, i1);

        std::string sub_description = description.substr(i0 + 1, i1 - i0 - 1);

        if (str_ends_with_digits(description.substr(0, i0), "IVF")) {
            parenthesis_ivf = sub_description;
        } else if (str_ends_with(description.substr(0, i0), "Refine")) {
            parenthesis_refine = sub_description;
        } else {
            FAISS_THROW_MSG("don't know what to do with parenthesis index");
        }
        description = description.erase(i0, i1 - i0 + 1);
    }

    int64_t ncentroids = -1;
    bool use_2layer = false;
    int hnsw_M = -1;
    int nsg_R = -1;

    for (char* tok = strtok_r(&description[0], " ,", &ptr); tok;
         tok = strtok_r(nullptr, " ,", &ptr)) {
        int d_out, opq_M, nbit, M, M2, pq_m, ncent, r2, R;
        std::string stok(tok);
        nbit = 8;
        int bbs = -1;
        char c;

        // to avoid mem leaks with exceptions:
        // do all tests before any instanciation

        VectorTransform* vt_1 = nullptr;
        Index* coarse_quantizer_1 = nullptr;
        Index* index_1 = nullptr;

        // VectorTransforms
        if (sscanf(tok, "PCA%d", &d_out) == 1) {
            vt_1 = new PCAMatrix(d, d_out);
            d = d_out;
        } else if (sscanf(tok, "PCAR%d", &d_out) == 1) {
            vt_1 = new PCAMatrix(d, d_out, 0, true);
            d = d_out;
        } else if (sscanf(tok, "RR%d", &d_out) == 1) {
            vt_1 = new RandomRotationMatrix(d, d_out);
            d = d_out;
        } else if (sscanf(tok, "PCAW%d", &d_out) == 1) {
            vt_1 = new PCAMatrix(d, d_out, -0.5, false);
            d = d_out;
        } else if (sscanf(tok, "PCAWR%d", &d_out) == 1) {
            vt_1 = new PCAMatrix(d, d_out, -0.5, true);
            d = d_out;
        } else if (sscanf(tok, "OPQ%d_%d", &opq_M, &d_out) == 2) {
            vt_1 = new OPQMatrix(d, opq_M, d_out);
            d = d_out;
        } else if (sscanf(tok, "OPQ%d", &opq_M) == 1) {
            vt_1 = new OPQMatrix(d, opq_M);
        } else if (sscanf(tok, "ITQ%d", &d_out) == 1) {
            vt_1 = new ITQTransform(d, d_out, true);
            d = d_out;
        } else if (stok == "ITQ") {
            vt_1 = new ITQTransform(d, d, false);
        } else if (sscanf(tok, "Pad%d", &d_out) == 1) {
            if (d_out > d) {
                vt_1 = new RemapDimensionsTransform(d, d_out, false);
                d = d_out;
            }
        } else if (stok == "L2norm") {
            vt_1 = new NormalizationTransform(d, 2.0);

            // coarse quantizers
        } else if (
                !coarse_quantizer &&
                sscanf(tok, "IVF%" PRId64 "_HNSW%d", &ncentroids, &M) == 2) {
            coarse_quantizer_1 = new IndexHNSWFlat(d, M, metric);

        } else if (
                !coarse_quantizer &&
                sscanf(tok, "IVF%" PRId64 "_NSG%d", &ncentroids, &R) == 2) {
            coarse_quantizer_1 = new IndexNSGFlat(d, R, metric);

        } else if (
                !coarse_quantizer &&
                sscanf(tok, "IVF%" PRId64, &ncentroids) == 1) {
            if (!parenthesis_ivf.empty()) {
                coarse_quantizer_1 =
                        index_factory(d, parenthesis_ivf.c_str(), metric);
            } else if (metric == METRIC_L2) {
                coarse_quantizer_1 = new IndexFlatL2(d);
            } else {
                coarse_quantizer_1 = new IndexFlatIP(d);
            }

        } else if (!coarse_quantizer && sscanf(tok, "IMI2x%d", &nbit) == 1) {
            FAISS_THROW_IF_NOT_MSG(
                    metric == METRIC_L2,
                    "MultiIndex not implemented for inner prod search");
            coarse_quantizer_1 = new MultiIndexQuantizer(d, 2, nbit);
            ncentroids = 1 << (2 * nbit);

        } else if (
                !coarse_quantizer &&
                sscanf(tok, "Residual%dx%d", &M, &nbit) == 2) {
            FAISS_THROW_IF_NOT_MSG(
                    metric == METRIC_L2,
                    "MultiIndex not implemented for inner prod search");
            coarse_quantizer_1 = new MultiIndexQuantizer(d, M, nbit);
            ncentroids = int64_t(1) << (M * nbit);
            use_2layer = true;

        } else if (std::regex_match(
                           stok,
                           std::regex(
                                   "(RQ|RCQ)[0-9]+x[0-9]+(_[0-9]+x[0-9]+)*"))) {
            std::vector<size_t> nbits;
            std::smatch sm;
            bool is_RCQ = stok.find("RCQ") == 0;
            while (std::regex_search(
                    stok, sm, std::regex("([0-9]+)x([0-9]+)"))) {
                int M = std::stoi(sm[1].str());
                int nbit = std::stoi(sm[2].str());
                nbits.resize(nbits.size() + M, nbit);
                stok = sm.suffix();
            }
            if (!is_RCQ) {
                index_1 = new IndexResidual(d, nbits, metric);
            } else {
                index_1 = new ResidualCoarseQuantizer(d, nbits, metric);
            }
        } else if (
                !coarse_quantizer &&
                sscanf(tok, "Residual%" PRId64, &ncentroids) == 1) {
            coarse_quantizer_1 = new IndexFlatL2(d);
            use_2layer = true;

        } else if (stok == "IDMap") {
            add_idmap = true;

            // IVFs
        } else if (!index && (stok == "Flat" || stok == "FlatDedup")) {
            if (coarse_quantizer) {
                // if there was an IVF in front, then it is an IVFFlat
                IndexIVF* index_ivf = stok == "Flat"
                        ? new IndexIVFFlat(
                                  coarse_quantizer, d, ncentroids, metric)
                        : new IndexIVFFlatDedup(
                                  coarse_quantizer, d, ncentroids, metric);
                index_ivf->quantizer_trains_alone =
                        get_trains_alone(coarse_quantizer);
                index_ivf->cp.spherical = metric == METRIC_INNER_PRODUCT;
                del_coarse_quantizer.release();
                index_ivf->own_fields = true;
                index_1 = index_ivf;
            } else if (hnsw_M > 0) {
                index_1 = new IndexHNSWFlat(d, hnsw_M, metric);
            } else if (nsg_R > 0) {
                index_1 = new IndexNSGFlat(d, nsg_R, metric);
            } else {
                FAISS_THROW_IF_NOT_MSG(
                        stok != "FlatDedup",
                        "dedup supported only for IVFFlat");
                index_1 = new IndexFlat(d, metric);
            }
        } else if (
                !index &&
                (stok == "SQ8" || stok == "SQ4" || stok == "SQ6" ||
                 stok == "SQfp16")) {
            ScalarQuantizer::QuantizerType qt = stok == "SQ8"
                    ? ScalarQuantizer::QT_8bit
                    : stok == "SQ6"    ? ScalarQuantizer::QT_6bit
                    : stok == "SQ4"    ? ScalarQuantizer::QT_4bit
                    : stok == "SQfp16" ? ScalarQuantizer::QT_fp16
                                       : ScalarQuantizer::QT_4bit;
            if (coarse_quantizer) {
                FAISS_THROW_IF_NOT(!use_2layer);
                IndexIVFScalarQuantizer* index_ivf =
                        new IndexIVFScalarQuantizer(
                                coarse_quantizer, d, ncentroids, qt, metric);
                index_ivf->quantizer_trains_alone =
                        get_trains_alone(coarse_quantizer);
                del_coarse_quantizer.release();
                index_ivf->own_fields = true;
                index_1 = index_ivf;
            } else if (hnsw_M > 0) {
                index_1 = new IndexHNSWSQ(d, qt, hnsw_M, metric);
            } else {
                index_1 = new IndexScalarQuantizer(d, qt, metric);
            }
        } else if (!index && sscanf(tok, "PQ%d+%d", &M, &M2) == 2) {
            FAISS_THROW_IF_NOT_MSG(
                    coarse_quantizer, "PQ with + works only with an IVF");
            FAISS_THROW_IF_NOT_MSG(
                    metric == METRIC_L2,
                    "IVFPQR not implemented for inner product search");
            IndexIVFPQR* index_ivf = new IndexIVFPQR(
                    coarse_quantizer, d, ncentroids, M, 8, M2, 8);
            index_ivf->quantizer_trains_alone =
                    get_trains_alone(coarse_quantizer);
            del_coarse_quantizer.release();
            index_ivf->own_fields = true;
            index_1 = index_ivf;
        } else if (
                !index &&
                (sscanf(tok, "PQ%dx4fs_%d", &M, &bbs) == 2 ||
                 (sscanf(tok, "PQ%dx4f%c", &M, &c) == 2 && c == 's') ||
                 (sscanf(tok, "PQ%dx4fs%c", &M, &c) == 2 && c == 'r'))) {
            if (bbs == -1) {
                bbs = 32;
            }
            bool by_residual = str_ends_with(stok, "fsr");
            if (coarse_quantizer) {
                IndexIVFPQFastScan* index_ivf = new IndexIVFPQFastScan(
                        coarse_quantizer, d, ncentroids, M, 4, metric, bbs);
                index_ivf->quantizer_trains_alone =
                        get_trains_alone(coarse_quantizer);
                index_ivf->metric_type = metric;
                index_ivf->by_residual = by_residual;
                index_ivf->cp.spherical = metric == METRIC_INNER_PRODUCT;
                del_coarse_quantizer.release();
                index_ivf->own_fields = true;
                index_1 = index_ivf;
            } else {
                IndexPQFastScan* index_pq =
                        new IndexPQFastScan(d, M, 4, metric, bbs);
                index_1 = index_pq;
            }
        } else if (
                !index &&
                (sscanf(tok, "PQ%dx%d", &M, &nbit) == 2 ||
                 sscanf(tok, "PQ%d", &M) == 1 ||
                 sscanf(tok, "PQ%dnp", &M) == 1)) {
            bool do_polysemous_training = stok.find("np") == std::string::npos;
            if (coarse_quantizer) {
                if (!use_2layer) {
                    IndexIVFPQ* index_ivf = new IndexIVFPQ(
                            coarse_quantizer, d, ncentroids, M, nbit);
                    index_ivf->quantizer_trains_alone =
                            get_trains_alone(coarse_quantizer);
                    index_ivf->metric_type = metric;
                    index_ivf->cp.spherical = metric == METRIC_INNER_PRODUCT;
                    del_coarse_quantizer.release();
                    index_ivf->own_fields = true;
                    index_ivf->do_polysemous_training = do_polysemous_training;
                    index_1 = index_ivf;
                } else {
                    Index2Layer* index_2l = new Index2Layer(
                            coarse_quantizer, ncentroids, M, nbit);
                    index_2l->q1.quantizer_trains_alone =
                            get_trains_alone(coarse_quantizer);
                    index_2l->q1.own_fields = true;
                    index_1 = index_2l;
                }
            } else if (hnsw_M > 0) {
                IndexHNSWPQ* ipq = new IndexHNSWPQ(d, M, hnsw_M);
                dynamic_cast<IndexPQ*>(ipq->storage)->do_polysemous_training =
                        do_polysemous_training;
                index_1 = ipq;
            } else {
                IndexPQ* index_pq = new IndexPQ(d, M, nbit, metric);
                index_pq->do_polysemous_training = do_polysemous_training;
                index_1 = index_pq;
            }
        } else if (
                !index &&
                sscanf(tok, "HNSW%d_%d+PQ%d", &M, &ncent, &pq_m) == 3) {
            Index* quant = new IndexFlatL2(d);
            IndexHNSW2Level* hidx2l =
                    new IndexHNSW2Level(quant, ncent, pq_m, M);
            Index2Layer* idx2l = dynamic_cast<Index2Layer*>(hidx2l->storage);
            idx2l->q1.own_fields = true;
            index_1 = hidx2l;
        } else if (
                !index &&
                sscanf(tok, "HNSW%d_2x%d+PQ%d", &M, &nbit, &pq_m) == 3) {
            Index* quant = new MultiIndexQuantizer(d, 2, nbit);
            IndexHNSW2Level* hidx2l =
                    new IndexHNSW2Level(quant, 1 << (2 * nbit), pq_m, M);
            Index2Layer* idx2l = dynamic_cast<Index2Layer*>(hidx2l->storage);
            idx2l->q1.own_fields = true;
            idx2l->q1.quantizer_trains_alone = 1;
            index_1 = hidx2l;
        } else if (!index && sscanf(tok, "HNSW%d_PQ%d", &M, &pq_m) == 2) {
            index_1 = new IndexHNSWPQ(d, pq_m, M);
        } else if (
                !index && sscanf(tok, "HNSW%d_SQ%d", &M, &pq_m) == 2 &&
                pq_m == 8) {
            index_1 = new IndexHNSWSQ(d, ScalarQuantizer::QT_8bit, M);
        } else if (!index && sscanf(tok, "HNSW%d", &M) == 1) {
            hnsw_M = M;
            // here it is unclear what we want: HNSW flat or HNSWx,Y ?
        } else if (!index && sscanf(tok, "NSG%d", &R) == 1) {
            nsg_R = R;
        } else if (
                !index &&
                (stok == "LSH" || stok == "LSHr" || stok == "LSHrt" ||
                 stok == "LSHt")) {
            bool rotate_data = strstr(tok, "r") != nullptr;
            bool train_thresholds = strstr(tok, "t") != nullptr;
            index_1 = new IndexLSH(d, d, rotate_data, train_thresholds);
        } else if (
                !index &&
                sscanf(tok, "ZnLattice%dx%d_%d", &M, &r2, &nbit) == 3) {
            FAISS_THROW_IF_NOT(!coarse_quantizer);
            index_1 = new IndexLattice(d, M, nbit, r2);
        } else if (stok == "RFlat") {
            parenthesis_refine = "Flat";
        } else if (stok == "Refine") {
            FAISS_THROW_IF_NOT_MSG(
                    !parenthesis_refine.empty(),
                    "Refine index should be provided in parentheses");
        } else {
            FAISS_THROW_FMT(
                    "could not parse token \"%s\" in %s\n",
                    tok,
                    description_in);
        }

        if (index_1 && add_idmap) {
            IndexIDMap* idmap = new IndexIDMap(index_1);
            del_index.set(idmap);
            idmap->own_fields = true;
            index_1 = idmap;
            add_idmap = false;
        }

        if (vt_1) {
            vts.chain.push_back(vt_1);
        }

        if (coarse_quantizer_1) {
            coarse_quantizer = coarse_quantizer_1;
            del_coarse_quantizer.set(coarse_quantizer);
        }

        if (index_1) {
            index = index_1;
            del_index.set(index);
        }
    }

    if (!index && hnsw_M > 0) {
        index = new IndexHNSWFlat(d, hnsw_M, metric);
        del_index.set(index);
    } else if (!index && nsg_R > 0) {
        index = new IndexNSGFlat(d, nsg_R, metric);
        del_index.set(index);
    }

    FAISS_THROW_IF_NOT_FMT(
            index, "description %s did not generate an index", description_in);

    // nothing can go wrong now
    del_index.release();
    del_coarse_quantizer.release();

    if (add_idmap) {
        fprintf(stderr,
                "index_factory: WARNING: "
                "IDMap option not used\n");
    }

    if (vts.chain.size() > 0) {
        IndexPreTransform* index_pt = new IndexPreTransform(index);
        index_pt->own_fields = true;
        // add from back
        while (vts.chain.size() > 0) {
            index_pt->prepend_transform(vts.chain.back());
            vts.chain.pop_back();
        }
        index = index_pt;
    }

    if (!parenthesis_refine.empty()) {
        Index* refine_index =
                index_factory(d_in, parenthesis_refine.c_str(), metric);
        IndexRefine* index_rf = new IndexRefine(index, refine_index);
        index_rf->own_refine_index = true;
        index_rf->own_fields = true;
        index = index_rf;
    }

    return index;
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
