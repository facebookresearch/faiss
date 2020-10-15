
import time
import os
import numpy as np
import faiss

from faiss.contrib.datasets import SynteticDataset


os.system("cat /proc/cpuinfo | grep 'model name' | head -1")

faiss.omp_set_num_threads(1)

for d in 16, 32, 64, 128:

    print("========== d=", d)


    nq = 100
    nb = 10000

    # d = 32

    ds = SynteticDataset(d, 0, nb, nq)

    print(ds)

    index = faiss.IndexFlatL2(d)

    index.add(ds.get_database())

    nrun = 10
    for k in 1, 10, 100:
        times = []
        for run in range(500):
            t0 = time.time()
            index.search(ds.get_queries(), k)
            t1 = time.time()
            if run > 50: # the rest is considered warmup
                times.append((t1 - t0) * 1000)
        print("search k=%3d t=%.3f ms (± %.4f)" % (
            k, np.mean(times), np.std(times)))

