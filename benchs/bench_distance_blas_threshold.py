#!/usr/bin/env python3
import argparse
import time

import numpy as np
import faiss

from faiss.contrib.datasets import SyntheticDataset


def measure(index, xq, k, nrun, nwarmup):
    times = []
    for i in range(nrun + nwarmup):
        t0 = time.perf_counter()
        index.search(xq, k)
        t1 = time.perf_counter()
        if i >= nwarmup:
            times.append(t1 - t0)
    arr = np.array(times) * 1000.0
    return float(arr.mean()), float(arr.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--nb", type=int, default=10000)
    parser.add_argument(
        "--nq", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument(
        "--thresholds", type=int, nargs="+", default=[1, 5, 10, 20, 40, 80]
    )
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument("--nrun", type=int, default=10)
    parser.add_argument("--nwarmup", type=int, default=3)
    parser.add_argument("--nthread", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--qbs", type=int, default=0)
    parser.add_argument("--dbs", type=int, default=0)
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.nthread)
    if args.qbs > 0:
        faiss.cvar.distance_compute_blas_query_bs = args.qbs
    if args.dbs > 0:
        faiss.cvar.distance_compute_blas_database_bs = args.dbs

    max_nq = max(args.nq)
    print("threshold,d,nq,k,mean_ms,std_ms")

    for threshold in args.thresholds:
        faiss.cvar.distance_compute_blas_threshold = threshold
        for d in args.d:
            ds = SyntheticDataset(d, 0, args.nb, max_nq)
            xb = ds.get_database()
            xq_all = ds.get_queries()

            if args.metric == "l2":
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)

            index.add(xb)

            for nq in args.nq:
                xq = xq_all[:nq]
                for k in args.k:
                    mean_ms, std_ms = measure(
                        index, xq, k, args.nrun, args.nwarmup
                    )
                    print(
                        f"{threshold},{d},{nq},{k},{mean_ms:.6f},{std_ms:.6f}"
                    )


if __name__ == "__main__":
    main()
