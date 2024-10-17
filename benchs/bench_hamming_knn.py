# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
import faiss

if __name__ == "__main__":
    faiss.omp_set_num_threads(1)

    for d in 4, 8, 16, 13:
        nq = 10000
        nb = 30000
        print('Bits per vector = 8 *', d)
        xq = faiss.randint((nq, d // 4), seed=1234, vmax=256**4).view('uint8')
        xb = faiss.randint((nb, d // 4), seed=1234, vmax=256**4).view('uint8')
        for variant in "hc", "mc":
            print(f"{variant=:}", end="\t")
            for k in 1, 4, 16, 64, 256:
                times = []
                for _run in range(5):
                    t0 = time.time()
                    D, I = faiss.knn_hamming(xq, xb, k, variant=variant)
                    t1 = time.time()
                    times.append(t1 - t0)
                print(f'| {k=:} t={np.mean(times):.3f} s ± {np.std(times):.3f} ', flush=True, end="")
            print()
