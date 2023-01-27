# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest

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

def test_ref_knn_with_draws(Dref, Iref, Dnew, Inew):
    """ test that knn search results are identical, raise if not """
    np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)
    # here we have to be careful because of draws
    testcase = unittest.TestCase()   # because it makes nice error messages
    for i in range(len(Iref)):
        if np.all(Iref[i] == Inew[i]): # easy case
            continue
        # we can deduce nothing about the latest line
        skip_dis = Dref[i, -1]
        for dis in np.unique(Dref):
            if dis == skip_dis:
                continue
            mask = Dref[i, :] == dis
            testcase.assertEqual(set(Iref[i, mask]), set(Inew[i, mask]))


def test_ref_range_results(lims_ref, Dref, Iref,
                           lims_new, Dnew, Inew):
    """ compare range search results wrt. a reference result,
    throw if it fails """
    np.testing.assert_array_equal(lims_ref, lims_new)
    nq = len(lims_ref) - 1
    for i in range(nq):
        l0, l1 = lims_ref[i], lims_ref[i + 1]
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
