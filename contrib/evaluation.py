# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
import time
import faiss

from multiprocessing.pool import ThreadPool

###############################################################
# Simple functions to evaluate knn results

def knn_intersection_measure(I1, I2):
    """ computes the intersection measure of two result tables
    """
    nq, rank = I1.shape
    assert I2.shape == (nq, rank)
    ninter = sum(
        np.intersect1d(I1[i], I2[i]).size
        for i in range(nq)
    )
    return ninter / I1.size

###############################################################
# Range search results can be compared with Precision-Recall

def filter_range_results(lims, D, I, thresh):
    """ select a set of results """
    nq = lims.size - 1
    mask = D < thresh
    new_lims = np.zeros_like(lims)
    for i in range(nq):
        new_lims[i + 1] = new_lims[i] + mask[lims[i] : lims[i + 1]].sum()
    return new_lims, D[mask], I[mask]


def range_PR(lims_ref, Iref, lims_new, Inew, mode="overall"):
    """compute the precision and recall of range search results. The
    function does not take the distances into account. """

    def ref_result_for(i):
        return Iref[lims_ref[i]:lims_ref[i + 1]]

    def new_result_for(i):
        return Inew[lims_new[i]:lims_new[i + 1]]

    nq = lims_ref.size - 1
    assert lims_new.size - 1 == nq

    ninter = np.zeros(nq, dtype="int64")

    def compute_PR_for(q):

        # ground truth results for this query
        gt_ids = ref_result_for(q)

        # results for this query
        new_ids = new_result_for(q)

        # there are no set functions in numpy so let's do this
        inter = np.intersect1d(gt_ids, new_ids)

        ninter[q] = len(inter)

    # run in a thread pool, which helps in spite of the GIL
    pool = ThreadPool(20)
    pool.map(compute_PR_for, range(nq))

    return counts_to_PR(
        lims_ref[1:] - lims_ref[:-1],
        lims_new[1:] - lims_new[:-1],
        ninter,
        mode=mode
    )


def counts_to_PR(ngt, nres, ninter, mode="overall"):
    """ computes a  precision-recall for a ser of queries.
    ngt = nb of GT results per query
    nres = nb of found results per query
    ninter = nb of correct results per query (smaller than nres of course)
    """

    if mode == "overall":
        ngt, nres, ninter = ngt.sum(), nres.sum(), ninter.sum()

        if nres > 0:
            precision = ninter / nres
        else:
            precision = 1.0

        if ngt > 0:
            recall = ninter / ngt
        elif nres == 0:
            recall = 1.0
        else:
            recall = 0.0

        return precision, recall

    elif mode == "average":
        # average precision and recall over queries

        mask = ngt == 0
        ngt[mask] = 1

        recalls = ninter / ngt
        recalls[mask] = (nres[mask] == 0).astype(float)

        # avoid division by 0
        mask = nres == 0
        assert np.all(ninter[mask] == 0)
        ninter[mask] = 1
        nres[mask] = 1

        precisions = ninter / nres

        return precisions.mean(), recalls.mean()

    else:
        raise AssertionError()

def sort_range_res_2(lims, D, I):
    """ sort 2 arrays using the first as key """
    I2 = np.empty_like(I)
    D2 = np.empty_like(D)
    nq = len(lims) - 1
    for i in range(nq):
        l0, l1 = lims[i], lims[i + 1]
        ii = I[l0:l1]
        di = D[l0:l1]
        o = di.argsort()
        I2[l0:l1] = ii[o]
        D2[l0:l1] = di[o]
    return I2, D2


def sort_range_res_1(lims, I):
    I2 = np.empty_like(I)
    nq = len(lims) - 1
    for i in range(nq):
        l0, l1 = lims[i], lims[i + 1]
        I2[l0:l1] = I[l0:l1]
        I2[l0:l1].sort()
    return I2


def range_PR_multiple_thresholds(
            lims_ref, Iref,
            lims_new, Dnew, Inew,
            thresholds,
            mode="overall", do_sort="ref,new"
    ):
    """ compute precision-recall values for range search results
    for several thresholds on the "new" results.
    This is to plot PR curves
    """
    # ref should be sorted by ids
    if "ref" in do_sort:
        Iref = sort_range_res_1(lims_ref, Iref)

    # new should be sorted by distances
    if "new" in do_sort:
        Inew, Dnew = sort_range_res_2(lims_new, Dnew, Inew)

    def ref_result_for(i):
        return Iref[lims_ref[i]:lims_ref[i + 1]]

    def new_result_for(i):
        l0, l1 = lims_new[i], lims_new[i + 1]
        return Inew[l0:l1], Dnew[l0:l1]

    nq = lims_ref.size - 1
    assert lims_new.size - 1 == nq

    nt = len(thresholds)
    counts = np.zeros((nq, nt, 3), dtype="int64")

    def compute_PR_for(q):
        gt_ids = ref_result_for(q)
        res_ids, res_dis = new_result_for(q)

        counts[q, :, 0] = len(gt_ids)

        if res_dis.size == 0:
            # the rest remains at 0
            return

        # which offsets we are interested in
        nres= np.searchsorted(res_dis, thresholds)
        counts[q, :, 1] = nres

        if gt_ids.size == 0:
            return

        # find number of TPs at each stage in the result list
        ii = np.searchsorted(gt_ids, res_ids)
        ii[ii == len(gt_ids)] = -1
        n_ok = np.cumsum(gt_ids[ii] == res_ids)

        # focus on threshold points
        n_ok = np.hstack(([0], n_ok))
        counts[q, :, 2] = n_ok[nres]

    pool = ThreadPool(20)
    pool.map(compute_PR_for, range(nq))
    # print(counts.transpose(2, 1, 0))

    precisions = np.zeros(nt)
    recalls = np.zeros(nt)
    for t in range(nt):
        p, r = counts_to_PR(
                counts[:, t, 0], counts[:, t, 1], counts[:, t, 2],
                mode=mode
        )
        precisions[t] = p
        recalls[t] = r

    return precisions, recalls


###############################################################
# Functions that compare search results with a reference result.
# They are intended for use in tests

def _cluster_tables_with_tolerance(tab1, tab2, thr):
    """ for two tables, cluster them by merging values closer than thr.
    Returns the cluster ids for each table element """
    tab = np.hstack([tab1, tab2])
    tab.sort()
    n = len(tab)
    diffs = np.ones(n)
    diffs[1:] = tab[1:] - tab[:-1]
    unique_vals = tab[diffs > thr]
    idx1 = np.searchsorted(unique_vals, tab1, side='right') - 1
    idx2 = np.searchsorted(unique_vals, tab2, side='right') - 1
    return idx1, idx2


def check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, rtol=1e-5):
    """ test that knn search results are identical, with possible ties.
    Raise if not. """
    np.testing.assert_allclose(Dref, Dnew, rtol=rtol)
    # here we have to be careful because of draws
    testcase = unittest.TestCase()   # because it makes nice error messages
    for i in range(len(Iref)):
        if np.all(Iref[i] == Inew[i]): # easy case
            continue

        # otherwise collect elements per distance
        r = rtol * Dref[i].max()

        DrefC, DnewC = _cluster_tables_with_tolerance(Dref[i], Dnew[i], r)

        for dis in np.unique(DrefC):
            if dis == DrefC[-1]:
                continue
            mask = DrefC == dis
            testcase.assertEqual(set(Iref[i, mask]), set(Inew[i, mask]))


def check_ref_range_results(Lref, Dref, Iref,
                            Lnew, Dnew, Inew):
    """ compare range search results wrt. a reference result,
    throw if it fails """
    np.testing.assert_array_equal(Lref, Lnew)
    nq = len(Lref) - 1
    for i in range(nq):
        l0, l1 = Lref[i], Lref[i + 1]
        Ii_ref = Iref[l0:l1]
        Ii_new = Inew[l0:l1]
        Di_ref = Dref[l0:l1]
        Di_new = Dnew[l0:l1]
        if np.all(Ii_ref == Ii_new): # easy
            pass
        else:
            def sort_by_ids(I, D):
                o = I.argsort()
                return I[o], D[o]
            # sort both
            (Ii_ref, Di_ref) = sort_by_ids(Ii_ref, Di_ref)
            (Ii_new, Di_new) = sort_by_ids(Ii_new, Di_new)
            np.testing.assert_array_equal(Ii_ref, Ii_new)
        np.testing.assert_array_almost_equal(Di_ref, Di_new, decimal=5)


###############################################################
# OperatingPoints functions
# this is the Python version of the AutoTune object in C++

class OperatingPoints:
    """
    Manages a set of search parameters with associated performance and time.
    Keeps the Pareto optimal points.
    """

    def __init__(self):
        # list of (key, perf, t)
        self.operating_points = [
            #  (self.do_nothing_key(), 0.0, 0.0)
        ]
        self.suboptimal_points = []

    def compare_keys(self, k1, k2):
        """ return -1 if k1 > k2, 1 if k2 > k1, 0 otherwise """
        raise NotImplemented

    def do_nothing_key(self):
        """ parameters to say we do noting, takes 0 time and has 0 performance"""
        raise NotImplemented

    def is_pareto_optimal(self, perf_new, t_new):
        for _, perf, t in self.operating_points:
            if perf >= perf_new and t <= t_new:
                return False
        return True

    def predict_bounds(self, key):
        """ predicts the bound on time and performance """
        min_time = 0.0
        max_perf = 1.0
        for key2, perf, t in self.operating_points + self.suboptimal_points:
            cmp = self.compare_keys(key, key2)
            if cmp > 0: # key2 > key
                if t > min_time:
                    min_time = t
            if cmp < 0: # key2 < key
                if perf < max_perf:
                    max_perf = perf
        return max_perf, min_time

    def should_run_experiment(self, key):
        (max_perf, min_time) = self.predict_bounds(key)
        return self.is_pareto_optimal(max_perf, min_time)

    def add_operating_point(self, key, perf, t):
        if self.is_pareto_optimal(perf, t):
            i = 0
            # maybe it shadows some other operating point completely?
            while i < len(self.operating_points):
                op_Ls, perf2, t2 = self.operating_points[i]
                if perf >= perf2 and t < t2:
                    self.suboptimal_points.append(
                        self.operating_points.pop(i))
                else:
                    i += 1
            self.operating_points.append((key, perf, t))
            return True
        else:
            self.suboptimal_points.append((key, perf, t))
            return False


class OperatingPointsWithRanges(OperatingPoints):
    """
    Set of parameters that are each picked from a discrete range of values.
    An increase of each parameter is assumed to make the operation slower
    and more accurate.
    A key = int array of indices in the ordered set of parameters.
    """

    def __init__(self):
        OperatingPoints.__init__(self)
        # list of (name, values)
        self.ranges = []

    def add_range(self, name, values):
        self.ranges.append((name, values))

    def compare_keys(self, k1, k2):
        if np.all(k1 >= k2):
            return 1
        if np.all(k2 >= k1):
            return -1
        return 0

    def do_nothing_key(self):
        return np.zeros(len(self.ranges), dtype=int)

    def num_experiments(self):
        return int(np.prod([len(values) for name, values in self.ranges]))

    def sample_experiments(self, n_autotune, rs=np.random):
        """ sample a set of experiments of max size n_autotune
        (run all experiments in random order if n_autotune is 0)
        """
        assert n_autotune == 0 or n_autotune >= 2
        totex = self.num_experiments()
        rs = np.random.RandomState(123)
        if n_autotune == 0 or totex < n_autotune:
            experiments = rs.permutation(totex - 2)
        else:
            experiments = rs.choice(
                totex - 2, size=n_autotune - 2, replace=False)

        experiments = [0, totex - 1] + [int(cno) + 1 for cno in experiments]
        return experiments

    def cno_to_key(self, cno):
        """Convert a sequential experiment number to a key"""
        k = np.zeros(len(self.ranges), dtype=int)
        for i, (name, values) in enumerate(self.ranges):
            k[i] = cno % len(values)
            cno //= len(values)
        assert cno == 0
        return k

    def get_parameters(self, k):
        """Convert a key to a dictionary with parameter values"""
        return {
            name: values[k[i]]
            for i, (name, values) in enumerate(self.ranges)
        }

    def restrict_range(self, name, max_val):
        """ remove too large values from a range"""
        for name2, values in self.ranges:
            if name == name2:
                val2 = [v for v in values if v < max_val]
                values[:] = val2
                return
        raise RuntimeError(f"parameter {name} not found")


###############################################################
# Timer object

class TimerIter:
    def __init__(self, timer):
        self.ts = []
        self.runs = timer.runs
        self.timer = timer
        if timer.nt >= 0:
            faiss.omp_set_num_threads(timer.nt)

    def __next__(self):
        timer = self.timer
        self.runs -= 1
        self.ts.append(time.time())
        total_time = self.ts[-1] - self.ts[0] if len(self.ts) >= 2 else 0
        if self.runs == -1 or total_time > timer.max_secs:
            if timer.nt >= 0:
                faiss.omp_set_num_threads(timer.remember_nt)
            ts = np.array(self.ts)
            times = ts[1:] - ts[:-1]
            if len(times) == timer.runs:
                timer.times = times[timer.warmup :]
            else:
                # if timeout, we use all the runs
                timer.times = times[:]
            raise StopIteration

class RepeatTimer:
    """
    This is yet another timer object. It is adapted to Faiss by
    taking a number of openmp threads to set on input. It should be called
    in an explicit loop as:

    timer = RepeatTimer(warmup=1, nt=1, runs=6)

    for _ in timer:
        # perform operation

    print(f"time={timer.get_ms():.1f} ± {timer.get_ms_std():.1f} ms")

    the same timer can be re-used. In that case it is reset each time it
    enters a loop. It focuses on ms-scale times because for second scale
    it's usually less relevant to repeat the operation.
    """
    def __init__(self, warmup=0, nt=-1, runs=1, max_secs=np.inf):
        assert warmup < runs
        self.warmup = warmup
        self.nt = nt
        self.runs = runs
        self.max_secs = max_secs
        self.remember_nt = faiss.omp_get_max_threads()

    def __iter__(self):
        return TimerIter(self)

    def ms(self):
        return np.mean(self.times) * 1000

    def ms_std(self):
        return np.std(self.times) * 1000 if len(self.times) > 1 else 0.0

    def nruns(self):
        """ effective number of runs (may be lower than runs - warmup due to timeout)"""
        return len(self.times)
