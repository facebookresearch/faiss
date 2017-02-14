
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright 2004-present Facebook. All Rights Reserved.
   implementation of Hyper-parameter auto-tuning
*/

#include "AutoTune.h"

#include "FaissAssert.h"
#include "utils.h"

#include "IndexFlat.h"
#include "VectorTransform.h"
#include "IndexLSH.h"
#include "IndexPQ.h"
#include "IndexIVF.h"
#include "IndexIVFPQ.h"
#include "MetaIndexes.h"



namespace faiss {


AutoTuneCriterion::AutoTuneCriterion (idx_t nq, idx_t nnn):
    nq (nq), nnn (nnn), gt_nnn (0)
{}


void AutoTuneCriterion::set_groundtruth (
     int gt_nnn, const float *gt_D_in, const idx_t *gt_I_in)
{
    this->gt_nnn = gt_nnn;
    if (gt_D_in) { // allow null for this, as it is often not used
        gt_D.resize (nq * gt_nnn);
        memcpy (gt_D.data(), gt_D_in, sizeof (gt_D[0]) * nq * gt_nnn);
    }
    gt_I.resize (nq * gt_nnn);
    memcpy (gt_I.data(), gt_I_in, sizeof (gt_I[0]) * nq * gt_nnn);
}



OneRecallAtRCriterion::OneRecallAtRCriterion (idx_t nq, idx_t R):
    AutoTuneCriterion(nq, R), R(R)
{}

double OneRecallAtRCriterion::evaluate (const float *D, const idx_t *I) const
{
    FAISS_ASSERT ((gt_I.size() == gt_nnn * nq && gt_nnn >= 1 && nnn >= R) ||
                  !"gound truth not initialized");
    idx_t n_ok = 0;
    for (idx_t q = 0; q < nq; q++) {
        idx_t gt_nn = gt_I [q * gt_nnn];
        const idx_t *I_line = I + q * nnn;
        for (int i = 0; i < R; i++) {
            if (I_line[i] == gt_nn) {
                n_ok++;
                break;
            }
        }
    }
    return n_ok / double (nq);
}


IntersectionCriterion::IntersectionCriterion (idx_t nq, idx_t R):
    AutoTuneCriterion(nq, R), R(R)
{}

double IntersectionCriterion::evaluate (const float *D, const idx_t *I) const
{
    FAISS_ASSERT ((gt_I.size() == gt_nnn * nq && gt_nnn >= R && nnn >= R) ||
                  !"gound truth not initialized");
    long n_ok = 0;
#pragma omp parallel for reduction(+: n_ok)
    for (idx_t q = 0; q < nq; q++) {
        n_ok += ranklist_intersection_size (
             R, &gt_I [q * gt_nnn],
             R, I + q * nnn);
    }
    return n_ok / double (nq * R);
}

/***************************************************************
 * OperatingPoints
 ***************************************************************/

OperatingPoints::OperatingPoints ()
{
    clear();
}

void OperatingPoints::clear ()
{
    all_pts.clear();
    optimal_pts.clear();
    /// default point: doing nothing gives 0 performance and takes 0 time
    OperatingPoint op = {0, 0, "", -1};
    optimal_pts.push_back(op);
}

/// add a performance measure
bool OperatingPoints::add (double perf, double t, const std::string & key,
                           size_t cno)
{
    OperatingPoint op = {perf, t, key, long(cno)};
    all_pts.push_back (op);
    if (perf == 0) {
        return false;  // no method for 0 accuracy is faster than doing nothing
    }
    std::vector<OperatingPoint> & a = optimal_pts;
    if (perf > a.back().perf) {
        // keep unconditionally
        a.push_back (op);
    } else if (perf == a.back().perf) {
        if (t < a.back ().t) {
            a.back() = op;
        } else {
            return false;
        }
    } else {
        int i;
        // stricto sensu this should be a bissection
        for (i = 0; i < a.size(); i++) {
            if (a[i].perf >= perf) break;
        }
        assert (i < a.size());
        if (t < a[i].t) {
            if (a[i].perf == perf) {
                a[i] = op;
            } else {
                a.insert (a.begin() + i, op);
            }
        } else {
            return false;
        }
    }
    { // remove non-optimal points from array
        int i = a.size() - 1;
        while (i > 0) {
            if (a[i].t < a[i - 1].t)
                a.erase (a.begin() + (i - 1));
            i--;
        }
    }
    return true;
}


int OperatingPoints::merge_with (const OperatingPoints &other,
                                 const std::string & prefix)
{
    int n_add = 0;
    for (int i = 0; i < other.all_pts.size(); i++) {
        const OperatingPoint & op = other.all_pts[i];
        if (add (op.perf, op.t, prefix + op.key, op.cno))
            n_add++;
    }
    return n_add;
}



/// get time required to obtain a given performance measure
double OperatingPoints::t_for_perf (double perf) const
{
    const std::vector<OperatingPoint> & a = optimal_pts;
    if (perf > a.back().perf) return 1e50;
    int i0 = -1, i1 = a.size() - 1;
    while (i0 + 1 < i1) {
        int imed = (i0 + i1 + 1) / 2;
        if (a[imed].perf < perf) i0 = imed;
        else                     i1 = imed;
    }
    return a[i1].t;
}


void OperatingPoints::all_to_gnuplot (const char *fname) const
{
    FILE *f = fopen(fname, "w");
    if (!f) {
        fprintf (stderr, "cannot open %s", fname);
        perror("");
        abort();
    }
    for (int i = 0; i < all_pts.size(); i++) {
        const OperatingPoint & op = all_pts[i];
        fprintf (f, "%g %g %s\n", op.perf, op.t, op.key.c_str());
    }
    fclose(f);
}

void OperatingPoints::optimal_to_gnuplot (const char *fname) const
{
    FILE *f = fopen(fname, "w");
    if (!f) {
        fprintf (stderr, "cannot open %s", fname);
        perror("");
        abort();
    }
    double prev_perf = 0.0;
    for (int i = 0; i < optimal_pts.size(); i++) {
        const OperatingPoint & op = optimal_pts[i];
        fprintf (f, "%g %g\n", prev_perf, op.t);
        fprintf (f, "%g %g %s\n", op.perf, op.t, op.key.c_str());
        prev_perf = op.perf;
    }
    fclose(f);
}

void OperatingPoints::display (bool only_optimal) const
{
    const std::vector<OperatingPoint> &pts =
        only_optimal ? optimal_pts : all_pts;
    printf("Tested %ld operating points, %ld ones are optimal:\n",
           all_pts.size(), optimal_pts.size());

    for (int i = 0; i < pts.size(); i++) {
        const OperatingPoint & op = pts[i];
        const char *star = "";
        if (!only_optimal) {
            for (int j = 0; j < optimal_pts.size(); j++) {
                if (op.cno == optimal_pts[j].cno) {
                    star = "*";
                    break;
                }
            }
        }
        printf ("cno=%ld key=%s perf=%.4f t=%.3f %s\n",
                op.cno, op.key.c_str(), op.perf, op.t, star);
    }

}

/***************************************************************
 * ParameterSpace
 ***************************************************************/

ParameterSpace::ParameterSpace ():
    verbose (1), n_experiments (500),
    batchsize (1<<30), thread_over_batches (false)
{
}

/* not keeping this constructor as inheritors will call the parent
   initialize()
 */

#if 0
ParameterSpace::ParameterSpace (Index *index):
    verbose (1), n_experiments (500),
    batchsize (1<<30), thread_over_batches (false)
{
    initialize(index);
}
#endif

size_t ParameterSpace::n_combinations () const
{
    size_t n = 1;
    for (int i = 0; i < parameter_ranges.size(); i++)
        n *= parameter_ranges[i].values.size();
    return n;
}

/// get string representation of the combination
std::string ParameterSpace::combination_name (size_t cno) const {
    char buf[1000], *wp = buf;
    *wp = 0;
    for (int i = 0; i < parameter_ranges.size(); i++) {
        const ParameterRange & pr = parameter_ranges[i];
        size_t j = cno % pr.values.size();
        cno /= pr.values.size();
        wp += snprintf (
              wp, buf + 1000 - wp, "%s%s=%g", i == 0 ? "" : ",",
              pr.name.c_str(), pr.values[j]);
    }
    return std::string (buf);
}


bool ParameterSpace::combination_ge (size_t c1, size_t c2) const
{
    for (int i = 0; i < parameter_ranges.size(); i++) {
        int nval = parameter_ranges[i].values.size();
        size_t j1 = c1 % nval;
        size_t j2 = c2 % nval;
        if (!(j1 >= j2)) return false;
        c1 /= nval;
        c2 /= nval;
    }
    return true;
}



#define DC(classname) \
    const classname *ix = dynamic_cast<const classname *>(index)

static void init_pq_ParameterRange (const ProductQuantizer & pq,
                                    ParameterRange & pr)
{
    if (pq.code_size % 4 == 0) {
        // Polysemous not supported for code sizes that are not a
        // multiple of 4
        for (int i = 2; i <= pq.code_size * 8 / 2; i+= 2)
            pr.values.push_back(i);
    }
    pr.values.push_back (pq.code_size * 8);
}

ParameterRange &ParameterSpace::add_range(const char * name)
{
    parameter_ranges.push_back (ParameterRange ());
    parameter_ranges.back ().name = name;
    return parameter_ranges.back ();
}


/// initialize with reasonable parameters for the index
void ParameterSpace::initialize (const Index * index)
{
    if (DC (IndexPreTransform)) {
        index = ix->index;
    }
    if (DC (IndexRefineFlat)) {
        ParameterRange & pr = add_range("k_factor_rf");
        for (int i = 0; i <= 6; i++) {
            pr.values.push_back (1 << i);
        }
        index = ix->base_index;
    }
    if (DC (IndexPreTransform)) {
        index = ix->index;
    }

    if (DC (IndexIVF)) {
        ParameterRange & pr = add_range("nprobe");
        for (int i = 0; i < 13; i++) {
            size_t nprobe = 1 << i;
            if (nprobe >= ix->nlist) break;
            pr.values.push_back (nprobe);
        }
    }
    if (DC (IndexPQ)) {
        ParameterRange & pr = add_range("ht");
        init_pq_ParameterRange (ix->pq, pr);
    }
    if (DC (IndexIVFPQ)) {
        ParameterRange & pr = add_range("ht");
        init_pq_ParameterRange (ix->pq, pr);

        const MultiIndexQuantizer *miq =
            dynamic_cast<const MultiIndexQuantizer *> (ix->quantizer);
        if (miq) {
            ParameterRange & pr_max_codes = add_range("max_codes");
            for (int i = 8; i < 20; i++) {
                pr_max_codes.values.push_back (1 << i);
            }
            pr_max_codes.values.push_back (1.0 / 0.0);
        }
    }
    if (DC (IndexIVFPQR)) {
        assert (ix);
        ParameterRange & pr = add_range("k_factor");
        for (int i = 0; i <= 6; i++) {
            pr.values.push_back (1 << i);
        }
    }
}

#undef DC

// non-const version
#define DC(classname) classname *ix = dynamic_cast<classname *>(index)


/// set a combination of parameters on an index
void ParameterSpace::set_index_parameters (Index *index, size_t cno) const
{

    for (int i = 0; i < parameter_ranges.size(); i++) {
        const ParameterRange & pr = parameter_ranges[i];
        size_t j = cno % pr.values.size();
        cno /= pr.values.size();
        double val = pr.values [j];
        set_index_parameter (index, pr.name, val);
    }
}

/// set a combination of parameters on an index
void ParameterSpace::set_index_parameters (
     Index *index, const char *description_in) const
{
    char description[strlen(description_in) + 1];
    char *ptr;
    memcpy (description, description_in, strlen(description_in) + 1);

    for (char *tok = strtok_r (description, " ,", &ptr);
         tok;
         tok = strtok_r (nullptr, " ,", &ptr)) {
        char name[100];
        double val;
        FAISS_ASSERT (sscanf (tok, "%100[^=]=%lf", name, &val) == 2);
        set_index_parameter (index, name, val);
    }

}

void ParameterSpace::set_index_parameter (
        Index * index, const std::string & name, double val) const
{
    if (verbose > 1)
        printf("    set %s=%g\n", name.c_str(), val);

    if (name == "verbose") {
        index->verbose = int(val);
    }
    if (DC (IndexPreTransform)) {
        index = ix->index;
    }
    if (name == "verbose") {
        index->verbose = int(val);
    }
    if (DC (IndexRefineFlat)) {
        if (name == "k_factor_rf") {
            ix->k_factor = int(val);
            return;
        }
        index = ix->base_index;
    }
    if (DC (IndexPreTransform)) {
        index = ix->index;
    }
    if (name == "verbose") {
        index->verbose = int(val);
        return; // last verbose that we could find
    }
    if (name == "nprobe") {
        DC(IndexIVF);
        ix->nprobe = int(val);
    } else if (name == "ht") {
        if (DC (IndexPQ)) {
            if (val >= ix->pq.code_size * 8) {
                ix->search_type = IndexPQ::ST_PQ;
            } else {
                ix->search_type = IndexPQ::ST_polysemous;
                ix->polysemous_ht = int(val);
            }
        } else if (DC (IndexIVFPQ)) {
            if (val >= ix->pq.code_size * 8) {
                ix->polysemous_ht = 0;
            } else {
                ix->polysemous_ht = int(val);
            }
        }
    } else if (name == "k_factor") {
        DC (IndexIVFPQR);
        ix->k_factor = val;
    } else if (name == "max_codes") {
        DC (IndexIVFPQ);
        ix->max_codes = finite(val) ? size_t(val) : 0;
    } else {
        fprintf(stderr,
                "ParameterSpace::set_index_parameter:"
                "could not set parameter %s\n",
                name.c_str());
    }
}

void ParameterSpace::display () const
{
    printf ("ParameterSpace, %ld parameters, %ld combinations:\n",
            parameter_ranges.size (), n_combinations ());
    for (int i = 0; i < parameter_ranges.size(); i++) {
        const ParameterRange & pr = parameter_ranges[i];
        printf ("   %s: ", pr.name.c_str ());
        char sep = '[';
        for (int j = 0; j < pr.values.size(); j++) {
            printf ("%c %g", sep, pr.values [j]);
            sep = ',';
        }
        printf ("]\n");
    }
}



void ParameterSpace::update_bounds (size_t cno, const OperatingPoint & op,
                                    double *upper_bound_perf,
                                    double *lower_bound_t) const
{
    if (combination_ge (cno, op.cno)) {
        if (op.t > *lower_bound_t) *lower_bound_t = op.t;
    }
    if (combination_ge (op.cno, cno)) {
        if (op.perf < *upper_bound_perf) *upper_bound_perf = op.perf;
    }
}



void ParameterSpace::explore (Index *index,
                              size_t nq, const float *xq,
                              const AutoTuneCriterion & crit,
                              OperatingPoints * ops) const
{
    FAISS_ASSERT (nq == crit.nq ||
                  !"criterion does not have the same nb of queries");

    size_t n_comb = n_combinations ();

    if (n_experiments == 0) {

        for (size_t cno = 0; cno < n_comb; cno++) {
            set_index_parameters (index, cno);
            std::vector<Index::idx_t> I(nq * crit.nnn);
            std::vector<float> D(nq * crit.nnn);

            double t0 = getmillisecs ();
            index->search (nq, xq, crit.nnn, D.data(), I.data());
            double t_search = (getmillisecs() - t0) / 1e3;

            double perf = crit.evaluate (D.data(), I.data());

            bool keep = ops->add (perf, t_search, combination_name (cno), cno);

            if (verbose)
                printf("  %ld/%ld: %s perf=%.3f t=%.3f s %s\n", cno, n_comb,
                       combination_name (cno).c_str(), perf, t_search,
                       keep ? "*" : "");
        }
        return;
    }

    int n_exp = n_experiments;

    if (n_exp > n_comb) n_exp = n_comb;
    FAISS_ASSERT (n_comb == 1 || n_exp > 2);
    std::vector<int> perm (n_comb);
    // make sure the slowest and fastest experiment are run
    perm[0] = 0;
    if (n_comb > 1) {
        perm[1] = n_comb - 1;
        rand_perm (&perm[2], n_comb - 2, 1234);
        for (int i = 2; i < perm.size(); i++) perm[i] ++;
    }

    for (size_t xp = 0; xp < n_exp; xp++) {
        size_t cno = perm[xp];

        if (verbose)
            printf("  %ld/%d: cno=%ld %s ", xp, n_exp, cno,
                   combination_name (cno).c_str());

        {
            double lower_bound_t = 0.0;
            double upper_bound_perf = 1.0;
            for (int i = 0; i < ops->all_pts.size(); i++) {
                update_bounds (cno, ops->all_pts[i],
                               &upper_bound_perf, &lower_bound_t);
            }
            double best_t = ops->t_for_perf (upper_bound_perf);
            if (verbose)
                printf ("bounds [perf<=%.3f t>=%.3f] %s",
                        upper_bound_perf, lower_bound_t,
                        best_t <= lower_bound_t ? "skip\n" : "");
            if (best_t <= lower_bound_t) continue;
        }

        set_index_parameters (index, cno);
        std::vector<Index::idx_t> I(nq * crit.nnn);
        std::vector<float> D(nq * crit.nnn);

        double t0 = getmillisecs ();

        if (thread_over_batches) {
#pragma omp parallel for
            for (size_t q0 = 0; q0 < nq; q0 += batchsize) {
                size_t q1 = q0 + batchsize;
                if (q1 > nq) q1 = nq;
                index->search (q1 - q0, xq + q0 * index->d,
                               crit.nnn,
                               D.data() + q0 * crit.nnn,
                               I.data() + q0 * crit.nnn);
            }
        } else {
            for (size_t q0 = 0; q0 < nq; q0 += batchsize) {
                size_t q1 = q0 + batchsize;
                if (q1 > nq) q1 = nq;
                index->search (q1 - q0, xq + q0 * index->d,
                               crit.nnn,
                               D.data() + q0 * crit.nnn,
                               I.data() + q0 * crit.nnn);
            }
        }

        double t_search = (getmillisecs() - t0) / 1e3;

        double perf = crit.evaluate (D.data(), I.data());

        bool keep = ops->add (perf, t_search, combination_name (cno), cno);

        if (verbose)
            printf(" perf %.3f t %.3f %s\n", perf, t_search,
                   keep ? "*" : "");
    }
}

/***************************************************************
 * index_factory
 ***************************************************************/

Index *index_factory (int d, const char *description_in, MetricType metric)
{
    VectorTransform *vt = nullptr;
    Index *coarse_quantizer = nullptr;
    Index *index = nullptr;
    bool add_idmap = false;
    bool make_IndexRefineFlat = false;

    char description[strlen(description_in) + 1];
    char *ptr;
    memcpy (description, description_in, strlen(description_in) + 1);

    int ncentroids = -1;

    for (char *tok = strtok_r (description, " ,", &ptr);
         tok;
         tok = strtok_r (nullptr, " ,", &ptr)) {
        int d_out, opq_M, nbit, M, M2;
        VectorTransform *vt_1 = nullptr;
        Index *coarse_quantizer_1 = nullptr;
        Index *index_1 = nullptr;

        // VectorTransforms
        if (sscanf (tok, "PCA%d", &d_out) == 1) {
            vt_1 = new PCAMatrix (d, d_out);
            d = d_out;
        } else if (sscanf (tok, "PCAR%d", &d_out) == 1) {
            vt_1 = new PCAMatrix (d, d_out, 0, true);
            d = d_out;
        } else if (sscanf (tok, "OPQ%d_%d", &opq_M, &d_out) == 2) {
            vt_1 = new OPQMatrix (d, opq_M, d_out);
            d = d_out;
        } else if (sscanf (tok, "OPQ%d", &opq_M) == 1) {
            vt_1 = new OPQMatrix (d, opq_M);
            // coarse quantizers
        } else if (sscanf (tok, "IVF%d", &ncentroids) == 1) {
            if (metric == METRIC_L2) {
                coarse_quantizer_1 = new IndexFlatL2 (d);
            } else { // if (metric == METRIC_IP)
                coarse_quantizer_1 = new IndexFlatIP (d);
            }
        } else if (sscanf (tok, "IMI2x%d", &nbit) == 1) {
            FAISS_ASSERT(metric == METRIC_L2 ||
                         !"MultiIndex not implemented for inner prod search");
            coarse_quantizer_1 = new MultiIndexQuantizer (d, 2, nbit);
            ncentroids = 1 << (2 * nbit);
        } else if (strcmp(tok, "IDMap") == 0) {
            add_idmap = true;

            // IVFs
        } else if (strcmp (tok, "Flat") == 0) {
            if (coarse_quantizer) {
                // if there was an IVF in front, then it is an IVFFlat
                IndexIVF *index_ivf = new IndexIVFFlat (
                    coarse_quantizer, d, ncentroids, metric);
                index_ivf->quantizer_trains_alone =
                    dynamic_cast<MultiIndexQuantizer*>(coarse_quantizer)
                    != nullptr;
                index_ivf->cp.spherical = metric == METRIC_INNER_PRODUCT;
                index_ivf->own_fields = true;
                index_1 = index_ivf;
            } else {
                index_1 = new IndexFlat (d, metric);
                if (add_idmap) {
                    IndexIDMap *idmap = new IndexIDMap(index_1);
                    idmap->own_fields = true;
                    index_1 = idmap;
                    add_idmap = false;
                }
            }
        } else if (sscanf (tok, "PQ%d+%d", &M, &M2) == 2) {
            FAISS_ASSERT(coarse_quantizer ||
                         !"PQ with + works only with an IVF");
            FAISS_ASSERT(metric == METRIC_L2 ||
                         !"IVFPQR not implemented for inner product search");
            IndexIVFPQR *index_ivf = new IndexIVFPQR (
                  coarse_quantizer, d, ncentroids, M, 8, M2, 8);
            index_ivf->quantizer_trains_alone =
                dynamic_cast<MultiIndexQuantizer*>(coarse_quantizer)
                != nullptr;
            index_ivf->own_fields = true;
            index_1 = index_ivf;
        } else if (sscanf (tok, "PQ%d", &M) == 1) {
            if (coarse_quantizer) {
                IndexIVFPQ *index_ivf = new IndexIVFPQ (
                    coarse_quantizer, d, ncentroids, M, 8);
                index_ivf->quantizer_trains_alone =
                    dynamic_cast<MultiIndexQuantizer*>(coarse_quantizer)
                    != nullptr;
                index_ivf->metric_type = metric;
                index_ivf->cp.spherical = metric == METRIC_INNER_PRODUCT;
                index_ivf->own_fields = true;
                index_ivf->do_polysemous_training = true;
                index_1 = index_ivf;
            } else {
                IndexPQ *index_pq = new IndexPQ (d, M, 8, metric);
                index_pq->do_polysemous_training = true;
                index_1 = index_pq;
                if (add_idmap) {
                    IndexIDMap *idmap = new IndexIDMap(index_1);
                    idmap->own_fields = true;
                    index_1 = idmap;
                    add_idmap = false;
                }
            }
        } else if (strcmp (tok, "RFlat") == 0) {
            make_IndexRefineFlat = true;
        } else {
            fprintf (stderr, "could not parse token \"%s\" in %s\n",
                     tok, description_in);
            FAISS_ASSERT (!"parse error");
        }

        if (vt_1)  {
            FAISS_ASSERT (!vt || !"cannot apply two VectorTransforms");
            vt = vt_1;
        }

        if (coarse_quantizer_1) {
            FAISS_ASSERT (!coarse_quantizer ||
                          !"cannot have 2 coarse quantizers");
            coarse_quantizer = coarse_quantizer_1;
        }

        if (index_1) {
            FAISS_ASSERT (!index || !"cannot have 2 indexes");
            index = index_1;
        }
    }

    if (add_idmap) {
        fprintf(stderr, "index_factory: WARNING: "
                "IDMap option not used\n");
    }

    if (vt) {
        IndexPreTransform *index_pt = new IndexPreTransform (vt, index);
        index_pt->own_fields = true;
        index = index_pt;
    }

    if (make_IndexRefineFlat) {
        IndexRefineFlat *index_rf = new IndexRefineFlat (index);
        index_rf->own_fields = true;
        index = index_rf;
    }

    return index;
}




}; // namespace faiss
