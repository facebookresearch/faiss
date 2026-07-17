# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def squaresig(n, period):
    if n == period:
        return np.zeros(n, dtype=bool)
    ret = np.zeros(n, dtype=bool).reshape(-1, 2, period)
    ret[:, 1, :] = True
    return ret.ravel()


def bitonic_merge_2(tab, step, stepk):
    tr = tab.reshape(-1, 2, step // 2)
    tc = tr.copy()
    # print(tc)
    mask = tr[:, 0, :] < tr[:, 1, :]
    flip_mask = squaresig(mask.shape[0], stepk // step)
    mask ^= flip_mask[:, None]
    # print(mask)
    # print(mask)
    # print(f"{n=} {step=} {stepk=}, {flip_mask.astype(int)}")
    tr[:, 0, :] = np.where(mask, tc[:, 0, :], tc[:, 1, :])
    tr[:, 1, :] = np.where(mask, tc[:, 1, :], tc[:, 0, :])
    # print(tr)


def bitonic_merge_rev(tab, res, step, stepk):
    n = len(tab)
    inv_tab = np.array([tab[i ^ (step // 2)] for i in range(n)])
    mask = np.ones(n, dtype=int)
    mask[:] = 2
    mask[res == tab] = 0
    mask[res == inv_tab] = 1
    cmp = tab > inv_tab
    mask ^= cmp.astype(int)
    flip_mask = np.array(
        [((i // (step // 2)) & 1) ^ (i // (stepk) & 1) for i in range(n)]
    )
    mask ^= flip_mask.astype(int)
    # flip_mask = np.array([(i % (stepk * 2) < stepk) for i in range(n)])
    # mask ^= flip_mask
    print(f"{n=:2} {step=:2} {stepk=:2}, A {mask}")
    # print(f"{n=:2} {step=:2} {stepk=:2}, B {flip_mask.astype(int)}")


def bitonic_merge_3(tab, step, stepk):
    n = len(tab)
    inv_tab = np.array([tab[i ^ (step // 2)] for i in range(n)])
    mask = tab > inv_tab
    flip_mask = np.array(
        [((i // (step // 2)) & 1) ^ (i // (stepk) & 1) for i in range(n)]
    )
    mask ^= flip_mask.astype(bool)
    print(f"{n=:2} {step=:2} {stepk=:2}, {mask.astype(int)}")
    tab[:] = np.where(mask, inv_tab, tab)


log_n = 4
rs = np.random.RandomState(1234)
tab = rs.permutation(1 << log_n)
print(tab)

for stepk in range(1, log_n + 1):
    for step in range(stepk, 0, -1):
        bitonic_merge_3(tab, 2**step, 2**stepk)
        print(tab)

if False:
    log_n = 5
    rs = np.random.RandomState(1234)

    for _ in range(10):
        tab = rs.permutation(1 << log_n)
        print(tab)

        for stepk in range(1, log_n + 1):
            for step in range(stepk, 0, -1):
                tab0 = tab.copy()
                bitonic_merge_3(tab, 2**step, 2**stepk)
                bitonic_merge_rev(tab0, tab, 2**step, 2**stepk)
