/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <climits>
#include <cstdio>
#include <memory>

#include <faiss/IVFlib.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/MetricType.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

/* This demo file shows how to:
 * - use a DistanceComputer to compute distances with encoded vectors
 * - in the context of an IVF, how to split an additive quantizer into an
 * AdditiveCoarseQuantizer and a ResidualQuantizer, in two different ways, with
 * and without storing the prefix.
 */

int main() {
    /******************************************
     * Generate a test dataset
     ******************************************/
    using idx_t = faiss::idx_t;
    size_t d = 128;
    size_t nt = 10000;
    size_t nb = 10000;
    size_t nq = 100;
    double t0 = faiss::getmillisecs();

    auto tic = [t0]() {
        printf("[%.3f s] ", (faiss::getmillisecs() - t0) / 1000);
    };

    tic();
    printf("samping dataset of %zd dim vectors, Q %zd B %zd T %zd\n",
           d,
           nq,
           nb,
           nt);

    std::vector<float> buf(d * (nq + nt + nb));
    faiss::rand_smooth_vectors(nq + nt + nb, d, buf.data(), 1234);
    const float* xt = buf.data();
    const float* xb = buf.data() + nt * d;
    const float* xq = buf.data() + (nt + nb) * d;

    idx_t k = 10;
    std::vector<idx_t> gt(k * nq);
    std::vector<float> unused(k * nq);
    tic();
    printf("compute ground truth, k=%zd\n", k);
    faiss::knn_L2sqr(xq, xb, d, nq, nb, k, unused.data(), gt.data());

    // a function to compute the accuracy
    auto accuracy = [&](const idx_t* I) {
        idx_t accu = 0;
        for (idx_t q = 0; q < nq; q++) {
            accu += faiss::ranklist_intersection_size(
                    k, gt.data() + q * k, k, I + q * k);
        }
        return double(accu) / (k * nq);
    };

    /******************************************
     * Prepare the residual quantizer
     ******************************************/

    faiss::ResidualQuantizer rq(
            d, 7, 6, faiss::AdditiveQuantizer::ST_norm_qint8);
    // do cheap an inaccurate training
    rq.cp.niter = 5;
    rq.max_beam_size = 5;
    rq.train_type = 0;
    tic();
    printf("training the residual quantizer beam_size=%d\n", rq.max_beam_size);
    rq.train(nt, xt);

    tic();
    printf("encoding the database, code_size=%zd\n", rq.code_size);
    size_t code_size = rq.code_size;
    std::vector<uint8_t> raw_codes(nb * code_size);
    rq.compute_codes(xb, raw_codes.data(), nb);

    /****************************************************************
     * Make an index that uses that residual quantizer
     * Verify that a distance computer gives the same distances
     ****************************************************************/
    {
        faiss::IndexResidualQuantizer index(
                rq.d, rq.nbits, faiss::METRIC_L2, rq.search_type);

        // override trained index
        index.rq = rq;
        index.is_trained = true;

        // override vectors
        index.codes = faiss::MaybeOwnedVector<uint8_t>(raw_codes);
        index.ntotal = nb;

        tic();
        printf("IndexResidualQuantizer ready, searching\n");

        std::vector<float> D(k * nq);
        std::vector<idx_t> I(k * nq);
        index.search(nq, xq, k, D.data(), I.data());

        tic();
        printf("Accuracy (intersection @ %zd): %.3f\n", k, accuracy(I.data()));
        std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
                index.get_FlatCodesDistanceComputer());

        float max_diff12 = 0, max_diff13 = 0;

        for (idx_t q = 0; q < nq; q++) {
            const float* query = xq + q * d;
            dc->set_query(query);
            for (int i = 0; i < k; i++) {
                // 3 ways of computing the same distance

                // distance returned by the index
                float dis1 = D[q * k + i];

                // distance returned by the DistanceComputer that accesses the
                // index
                idx_t db_index = I[q * k + i];
                float dis2 = (*dc)(db_index);

                // distance computer from a code that does not belong to the
                // index
                const uint8_t* code = raw_codes.data() + code_size * db_index;
                float dis3 = dc->distance_to_code(code);

                max_diff12 = std::max(std::abs(dis1 - dis2), max_diff12);
                max_diff13 = std::max(std::abs(dis1 - dis3), max_diff13);
            }
        }
        tic();
        printf("Max DistanceComputer discrepancy 1-2: %g 1-3: %g\n",
               max_diff12,
               max_diff13);
    }

    /****************************************************************
     * Make an IVF index that uses the first 2 levels as a coarse quantizer
     * The IVF codes contain the full code (ie. redundant with the coarse
     *quantizer code)
     ****************************************************************/
    {
        // build a coarse quantizer from the 2 first levels of the RQ
        std::vector<size_t> nbits(2);
        std::copy(rq.nbits.begin(), rq.nbits.begin() + 2, nbits.begin());
        faiss::ResidualCoarseQuantizer rcq(rq.d, nbits);

        // set the coarse quantizer from the 2 first quantizers
        rcq.rq.initialize_from(rq);
        rcq.is_trained = true;
        rcq.ntotal = (idx_t)1 << rcq.rq.tot_bits;

        // settings for exhaustive search in RCQ
        rcq.centroid_norms.resize(rcq.ntotal);
        rcq.aq->compute_centroid_norms(rcq.centroid_norms.data());
        rcq.beam_factor = -1.0; // use exact search
        size_t nlist = rcq.ntotal;
        tic();
        printf("RCQ nlist = %zd tot_bits=%zd\n", nlist, rcq.rq.tot_bits);

        // build a IVFResidualQuantizer from that
        faiss::IndexIVFResidualQuantizer index(
                &rcq, rcq.d, nlist, rq.nbits, faiss::METRIC_L2, rq.search_type);
        index.by_residual = false;
        index.rq = rq;
        index.is_trained = true;

        // there are 3 ways of filling up the index...
        for (std::string filled_with : {"add", "manual", "derived"}) {
            tic();
            printf("filling up the index with %s, code_size=%zd\n",
                   filled_with.c_str(),
                   index.code_size);

            index.reset();

            if (filled_with == "add") {
                // standard add method
                index.add(nb, xb);
            } else if (filled_with == "manual") {
                // compute inverted lists and add elements manually
                // fill in the inverted index manually
                faiss::InvertedLists& invlists = *index.invlists;

                // assign vectors to inverted lists
                std::vector<idx_t> listnos(nb);
                std::vector<float> unused(nb);
                rcq.search(nb, xb, 1, unused.data(), listnos.data());

                // populate inverted lists
                for (idx_t i = 0; i < nb; i++) {
                    invlists.add_entry(
                            listnos[i], i, &raw_codes[i * code_size]);
                }

                index.ntotal = nb;
            } else if (filled_with == "derived") {
                // Since we have the raw codes precomputed, their prefix is the
                // inverted list index, so let's use that.
                faiss::InvertedLists& invlists = *index.invlists;

                // populate inverted lists
                for (idx_t i = 0; i < nb; i++) {
                    const uint8_t* code = &raw_codes[i * code_size];
                    faiss::BitstringReader rd(code, code_size);
                    idx_t list_no =
                            rd.read(rcq.rq.tot_bits); // read the list number
                    invlists.add_entry(list_no, i, code);
                }

                index.ntotal = nb;
            }

            tic();
            printf("Index filled in\n");

            for (int nprobe : {1, 4, 16, 64, int(nlist)}) {
                printf("setting nprobe=%-4d", nprobe);

                index.nprobe = nprobe;
                std::vector<float> D(k * nq);
                std::vector<idx_t> I(k * nq);
                index.search(nq, xq, k, D.data(), I.data());

                tic();
                printf("Accuracy (intersection @ %zd): %.3f\n",
                       k,
                       accuracy(I.data()));
            }
        }
    }

    /****************************************************************
     * Make an IVF index that uses the first 2 levels as a coarse
     * quantizer, but this time does not store the code prefix from the index
     ****************************************************************/

    {
        // build a coarse quantizer from the 2 first levels of the RQ
        int nlevel = 2;

        std::unique_ptr<faiss::IndexIVFResidualQuantizer> index(
                faiss::ivflib::ivf_residual_from_quantizer(rq, nlevel));

        // there are 2 ways of filling up the index...
        for (std::string filled_with : {"add", "derived"}) {
            tic();
            printf("filling up the IVF index with %s, code_size=%zd\n",
                   filled_with.c_str(),
                   index->code_size);

            index->reset();

            if (filled_with == "add") {
                // standard add method
                index->add(nb, xb);
            } else if (filled_with == "derived") {
                faiss::ivflib::ivf_residual_add_from_flat_codes(
                        index.get(), nb, raw_codes.data(), rq.code_size);
            }

            tic();
            printf("Index filled in\n");

            for (int nprobe : {1, 4, 16, 64, int(index->nlist)}) {
                printf("setting nprobe=%-4d", nprobe);

                index->nprobe = nprobe;
                std::vector<float> D(k * nq);
                std::vector<idx_t> I(k * nq);
                index->search(nq, xq, k, D.data(), I.data());

                tic();
                printf("Accuracy (intersection @ %zd): %.3f\n",
                       k,
                       accuracy(I.data()));
            }
        }
    }

    return 0;
}
