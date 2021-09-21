# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import os
import argparse

import numpy as np


def eval_recalls(name, I, gt, times):
    k = I.shape[1]
    s = "%-40s recall" % name
    nq = len(gt)
    for rank in 1, 10, 100, 1000:
        if rank > k:
            break
        recall = (I[:, :rank] == gt[:, :1]).sum() / nq
        s += "@%d: %.4f " % (rank, recall)
    s += "time: %.4f s (± %.4f)" % (np.mean(times), np.std(times))
    print(s)

def eval_inters(name, I, gt, times):
    k = I.shape[1]
    s = "%-40s inter" % name
    nq = len(gt)
    for rank in 1, 10, 100, 1000:
        if rank > k:
            break
        ninter = 0
        for i in range(nq):
            ninter += np.intersect1d(I[i, :rank], gt[i, :rank]).size
        inter = ninter / (nq * rank)
        s += "@%d: %.4f " % (rank, inter)
    s += "time: %.4f s (± %.4f)" % (np.mean(times), np.std(times))
    print(s)


def main():

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')

    aa('--db', default='deep1M', help='dataset')
    aa('--measure', default="1-recall",
        help="perf measure to use: 1-recall or inter")
    aa('--download', default=False, action="store_true")
    aa('--lib', default='faiss', help='library to use (faiss or scann)')
    aa('--thenscann', default=False, action="store_true")
    aa('--base_dir', default='/checkpoint/matthijs/faiss_improvements/cmp_ivf_scan_2')

    group = parser.add_argument_group('searching')
    aa('--k', default=10, type=int, help='nb of nearest neighbors')
    aa('--pre_reorder_k', default="0,10,100,1000", help='values for reorder_k')
    aa('--nprobe', default="1,2,5,10,20,50,100,200", help='values for nprobe')
    aa('--nrun', default=5, type=int, help='nb of runs to perform')
    args = parser.parse_args()

    print("args:", args)
    pre_reorder_k_tab = [int(x) for x in args.pre_reorder_k.split(',')]
    nprobe_tab = [int(x) for x in args.nprobe.split(',')]

    os.system('echo -n "nb processors "; '
            'cat /proc/cpuinfo | grep ^processor | wc -l; '
            'cat /proc/cpuinfo | grep ^"model name" | tail -1')

    cache_dir = args.base_dir + "/" + args.db + "/"
    k = args.k
    nrun = args.nrun

    if args.lib == "faiss":
        # prepare cache
        import faiss
        from datasets import load_dataset

        ds = load_dataset(args.db, download=args.download)
        print(ds)
        if not os.path.exists(cache_dir + "xb.npy"):
            # store for SCANN
            os.system(f"rm -rf {cache_dir}; mkdir -p {cache_dir}")
            tosave = dict(
                # xt = ds.get_train(10),
                xb = ds.get_database(),
                xq = ds.get_queries(),
                gt = ds.get_groundtruth()
            )
            for name, v in tosave.items():
                fname = cache_dir + "/" + name + ".npy"
                print("save", fname)
                np.save(fname, v)

            open(cache_dir + "metric", "w").write(ds.metric)

        name1_to_metric = {
            "IP": faiss.METRIC_INNER_PRODUCT,
            "L2": faiss.METRIC_L2
        }

        index_fname = cache_dir + "index.faiss"
        if not os.path.exists(index_fname):
            index = faiss_make_index(
                ds.get_database(), name1_to_metric[ds.metric], index_fname)
        else:
            index = faiss.read_index(index_fname)

        xb = ds.get_database()
        xq = ds.get_queries()
        gt = ds.get_groundtruth()

        faiss_eval_search(
                index, xq, xb, nprobe_tab, pre_reorder_k_tab, k, gt,
                nrun, args.measure
        )

    if args.lib == "scann":
        from scann.scann_ops.py import scann_ops_pybind

        dataset = {}
        for kn in "xb xq gt".split():
            fname = cache_dir + "/" + kn + ".npy"
            print("load", fname)
            dataset[kn] = np.load(fname)
        name1_to_name2 = {
            "IP": "dot_product",
            "L2": "squared_l2"
        }
        distance_measure = name1_to_name2[open(cache_dir + "metric").read()]

        xb = dataset["xb"]
        xq = dataset["xq"]
        gt = dataset["gt"]

        scann_dir = cache_dir + "/scann1.1.1_serialized"
        if os.path.exists(scann_dir + "/scann_config.pb"):
            searcher = scann_ops_pybind.load_searcher(scann_dir)
        else:
            searcher = scann_make_index(xb, distance_measure, scann_dir, 0)

        scann_dir = cache_dir + "/scann1.1.1_serialized_reorder"
        if os.path.exists(scann_dir + "/scann_config.pb"):
            searcher_reo = scann_ops_pybind.load_searcher(scann_dir)
        else:
            searcher_reo = scann_make_index(xb, distance_measure, scann_dir, 100)

        scann_eval_search(
            searcher, searcher_reo,
            xq, xb, nprobe_tab, pre_reorder_k_tab, k, gt,
            nrun, args.measure
        )

    if args.lib != "scann" and args.thenscann:
        # just append --lib scann, that will override the previous cmdline
        # options
        cmdline = " ".join(sys.argv) + " --lib scann"
        cmdline = (
            ". ~/anaconda3/etc/profile.d/conda.sh ; " +
            "conda activate scann_1.1.1; "
            "python -u " + cmdline)

        print("running", cmdline)

        os.system(cmdline)


###############################################################
# SCANN
###############################################################

def scann_make_index(xb, distance_measure, scann_dir, reorder_k):
    import scann

    print("build index")

    if distance_measure == "dot_product":
        thr = 0.2
    else:
        thr = 0
    k = 10
    sb = scann.scann_ops_pybind.builder(xb, k, distance_measure)
    sb = sb.tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
    sb = sb.score_ah(2, anisotropic_quantization_threshold=thr)

    if reorder_k > 0:
        sb = sb.reorder(reorder_k)

    searcher = sb.build()

    print("done")

    print("write index to", scann_dir)

    os.system(f"rm -rf {scann_dir}; mkdir -p {scann_dir}")
    # os.mkdir(scann_dir)
    searcher.serialize(scann_dir)
    return searcher

def scann_eval_search(
        searcher, searcher_reo,
        xq, xb, nprobe_tab, pre_reorder_k_tab, k, gt,
        nrun, measure):

    # warmup
    for _run in range(5):
        searcher.search_batched(xq)

    for nprobe in nprobe_tab:

        for pre_reorder_k in pre_reorder_k_tab:

            times = []
            for _run in range(nrun):
                if pre_reorder_k == 0:
                    t0 = time.time()
                    I, D = searcher.search_batched(
                        xq, leaves_to_search=nprobe, final_num_neighbors=k
                    )
                    t1 = time.time()
                else:
                    t0 = time.time()
                    I, D = searcher_reo.search_batched(
                        xq, leaves_to_search=nprobe, final_num_neighbors=k,
                        pre_reorder_num_neighbors=pre_reorder_k
                    )
                    t1 = time.time()

                times.append(t1 - t0)
            header = "SCANN nprobe=%4d reo=%4d" % (nprobe, pre_reorder_k)
            if measure == "1-recall":
                eval_recalls(header, I, gt, times)
            else:
                eval_inters(header, I, gt, times)




###############################################################
# Faiss
###############################################################


def faiss_make_index(xb, metric_type, fname):
    import faiss

    d = xb.shape[1]
    M = d // 2
    index = faiss.index_factory(d, f"IVF2000,PQ{M}x4fs", metric_type)
    # if not by_residual:
    #    print("setting no residual")
    #    index.by_residual = False

    print("train")
    # index.train(ds.get_train())
    index.train(xb[:250000])
    print("add")
    index.add(xb)
    print("write index", fname)
    faiss.write_index(index, fname)

    return index

def faiss_eval_search(
            index, xq, xb, nprobe_tab, pre_reorder_k_tab,
            k, gt, nrun, measure
    ):
    import faiss

    print("use precomputed table=", index.use_precomputed_table,
          "by residual=", index.by_residual)

    print("adding a refine index")
    index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(xb))

    print("set single thread")
    faiss.omp_set_num_threads(1)

    print("warmup")
    for _run in range(5):
        index.search(xq, k)

    print("run timing")
    for nprobe in nprobe_tab:
        for pre_reorder_k in pre_reorder_k_tab:
            index.nprobe = nprobe
            times = []
            for _run in range(nrun):
                if pre_reorder_k == 0:
                    t0 = time.time()
                    D, I = index.search(xq, k)
                    t1 = time.time()
                else:
                    index_refine.k_factor = pre_reorder_k / k
                    t0 = time.time()
                    D, I = index_refine.search(xq, k)
                    t1 = time.time()

                times.append(t1 - t0)

            header = "Faiss nprobe=%4d reo=%4d" % (nprobe, pre_reorder_k)
            if measure == "1-recall":
                eval_recalls(header, I, gt, times)
            else:
                eval_inters(header, I, gt, times)


if __name__ == "__main__":
    main()
