# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import faiss
import numpy as np

def do_partition(n, qin, maxval=65536, seed=123, id_type='int64'):
    print(
        f"n={n} qin={qin} maxval={maxval} id_type={id_type}  ",
        end="\t", flush=True
    )

    # print("seed=", seed)
    rs = np.random.RandomState(seed)
    vals = rs.randint(maxval, size=n).astype('uint16')
    ids = (rs.permutation(n) + 12345).astype(id_type)

    sp = faiss.swig_ptr

    tab_a = faiss.AlignedTableUint16()
    faiss.copy_array_to_AlignedTable(vals, tab_a)

    nrun = 2000

    times = []
    nerr = 0
    stats = faiss.cvar.partition_stats
    stats.reset()
    for _run in range(nrun):
        faiss.copy_array_to_AlignedTable(vals, tab_a)
        t0 = time.time()
        # print("tab a type", tab_a.get())
        if type(qin) == int:
            q = qin
            faiss.CMax_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n, q, q, None)
        else:
            q_min, q_max = qin
            q = np.array([-1], dtype='uint64')
            faiss.CMax_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n,
                q_min, q_max, sp(q)
            )
            q = q[0]

            if not (q_min <= q <= q_max):
                nerr += 1

        t1 = time.time()

        times.append(t1 - t0)

    times = np.array(times[100:]) * 1000000


    print(
        f"times {times.mean():.3f} µs (± {times.std():.4f} µs) nerr={nerr} "
        f"bissect {stats.bissect_cycles / 1e6:.3f} Mcy "
        f"compress {stats.compress_cycles / 1e6:.3f} Mcy"
    )

do_partition(200, (100, 100))
do_partition(200, (100, 150))
do_partition(2000, (1000, 1000))
do_partition(2000, (1000, 1500))
do_partition(20000, (10000, 10000))
do_partition(20000, (10000, 15000))


do_partition(200, (100, 100), id_type='int32')
do_partition(200, (100, 150), id_type='int32')
do_partition(2000, (1000, 1000), id_type='int32')
do_partition(2000, (1000, 1500), id_type='int32')
do_partition(20000, (10000, 10000), id_type='int32')
do_partition(20000, (10000, 15000), id_type='int32')
