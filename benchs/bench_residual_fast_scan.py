import sys
import numpy as np

import faiss
# Manifold way too slow
# import faiss.contrib.datasets_fb as datasets

# cd /data/users/matthijs/simsearch_datasets
# manifold getr faiss_index_store/tree/simsearch/sift1M

# buck run @mode/opt //faiss/benchs/:bench_residual_fast_scan -- pq rq

from faiss.contrib import datasets
datasets.dataset_basedir = "/data/users/matthijs/simsearch_datasets/"

import time

t0 = time.time()
ds = datasets.DatasetDeep1B(nb=10**6)

def get_database_small(self):
    return datasets.sanitize(datasets.fvecs_mmap(self.basedir + "base10M.fvecs")[:self.nb])

ds.get_database = lambda: get_database_small(ds)

print("get data")
xq = ds.get_queries()
xb = ds.get_database()
# import pdb; pdb.set_trace()
xt = ds.get_train(maxtrain=100_000)

gt = ds.get_groundtruth()

for todo in sys.argv[1:]:
    print("running experiment with", todo)

    metric = (
        faiss.METRIC_INNER_PRODUCT if "ip" in todo else
        faiss.METRIC_L2 if "l2" in todo else
        1/0
    )

    if "pq" in todo:

        if "fs" in todo:
            index_pq = faiss.index_factory(ds.d, "PQ48x4fs", metric)
        else:
            index_pq = faiss.index_factory(ds.d, "PQ48x4", metric)

        print(f"[{time.time()-t0:.3f} s] train")
        index_pq.train(xt)
        print(f"[{time.time()-t0:.3f} s] add")
        index_pq.add(xb)
        print(f"[{time.time()-t0:.3f} s] search")
        D, I = index_pq.search(xq, 100)
        print(f"[{time.time()-t0:.3f} s] done")

        for rank in 1, 10, 100:
            recall = (I[:, :rank] == gt[:, :1]).sum() / ds.nq
            print(f"1-recall @ {rank}: {recall:.4f}")

    if "rq" in todo:
        if "rqnorm8" in todo:
            index_rq = faiss.index_factory(ds.d, "RQ46x4_Nqint8", metric)
        else:
            index_rq = faiss.index_factory(ds.d, "RQ48x4", metric)
        print("beam size", index_rq.rq.max_beam_size)
        print(f"[{time.time()-t0:.3f} s] train")
        index_rq.train(xt)
        print(f"[{time.time()-t0:.3f} s] add")
        index_rq.add(xb)
        if "fs" in todo:
            print(f"[{time.time()-t0:.3f} s] convert to fast-scan")
            index_rq = faiss.IndexResidualQuantizerFastScan(index_rq)
        print(f"[{time.time()-t0:.3f} s] search")
        D, I = index_rq.search(xq, 100)
        print(f"[{time.time()-t0:.3f} s] done")

        for rank in 1, 10, 100:
            recall = (I[:, :rank] == gt[:, :1]).sum() / ds.nq
            print(f"1-recall @ {rank}: {recall:.4f}")
