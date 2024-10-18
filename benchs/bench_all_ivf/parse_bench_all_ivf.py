# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from collections import defaultdict
from matplotlib import pyplot

import re

from argparse import Namespace

from faiss.contrib.factory_tools import get_code_size as unitsize


def dbsize_from_name(dbname):
    sufs = {
        '1B': 10**9,
        '100M': 10**8,
        '10M': 10**7,
        '1M': 10**6,
    }
    for s in sufs:
        if dbname.endswith(s):
            return sufs[s]
    else:
        assert False


def keep_latest_stdout(fnames):
    fnames = [fname for fname in fnames if fname.endswith('.stdout')]
    fnames.sort()
    n = len(fnames)
    fnames2 = []
    for i, fname in enumerate(fnames):
        if i + 1 < n and fnames[i + 1][:-8] == fname[:-8]:
            continue
        fnames2.append(fname)
    return fnames2


def parse_result_file(fname):
    # print fname
    st = 0
    res = []
    keys = []
    stats = {}
    stats['run_version'] = fname[-8]
    indexkey = None
    for l in open(fname):
        if l.startswith("srun:"):
            # looks like a crash...
            if indexkey is None:
                raise RuntimeError("instant crash")
            break
        elif st == 0:
            if l.startswith("dataset in dimension"):
                fi = l.split()
                stats["d"] = int(fi[3][:-1])
                stats["nq"] = int(fi[9])
                stats["nb"] = int(fi[11])
                stats["nt"] = int(fi[13])
            if l.startswith('index size on disk:'):
                stats['index_size'] = int(l.split()[-1])
            if l.startswith('current RSS:'):
                stats['RSS'] = int(l.split()[-1])
            if l.startswith('precomputed tables size:'):
                stats['tables_size'] = int(l.split()[-1])
            if l.startswith('Setting nb of threads to'):
                stats['n_threads'] = int(l.split()[-1])
            if l.startswith('  add in'):
                stats['add_time'] = float(l.split()[-2])
            if l.startswith("vector code_size"):
                stats['code_size'] = float(l.split()[-1])
            if l.startswith('args:'):
                args = eval(l[l.find(' '):])
                indexkey = args.indexkey
            elif "time(ms/q)" in l:
                # result header
                if 'R@1   R@10  R@100' in l:
                    stats["measure"] = "recall"
                    stats["ranks"] = [1, 10, 100]
                elif 'I@1   I@10  I@100' in l:
                    stats["measure"] = "inter"
                    stats["ranks"] = [1, 10, 100]
                elif 'inter@' in l:
                    stats["measure"] = "inter"
                    fi = l.split()
                    if fi[1] == "inter@":
                        rank = int(fi[2])
                    else:
                        rank = int(fi[1][len("inter@"):])
                    stats["ranks"] = [rank]

                else:
                    assert False
                st = 1
            elif 'index size on disk:' in l:
                stats["index_size"] = int(l.split()[-1])
        elif st == 1:
            st = 2
        elif st == 2:
            fi = l.split()
            if l[0] == " ":
                # means there are 0 parameters
                fi = [""] + fi
            keys.append(fi[0])
            res.append([float(x) for x in fi[1:]])
    return indexkey, np.array(res), keys, stats

# the directory used in run_on_cluster.bash
basedir = "/checkpoint/matthijs/bench_all_ivf/"
logdir = basedir + 'logs/'


def collect_results_for(db='deep1M', prefix="autotune."):
    # run parsing
    allres = {}
    allstats = {}
    missing = []

    fnames = keep_latest_stdout(os.listdir(logdir))
    # print fnames
    # filenames are in the form <key>.x.stdout
    # where x is a version number (from a to z)
    # keep only latest version of each name

    for fname in fnames:
        if not (
                'db' + db in fname and
                fname.startswith(prefix) and
                fname.endswith('.stdout')
            ):
            continue
        print("parse", fname, end="   ", flush=True)
        try:
            indexkey, res, _, stats = parse_result_file(logdir + fname)
        except RuntimeError as e:
            print("FAIL %s" % e)
            res = np.zeros((2, 0))
        except Exception as e:
            print("PARSE ERROR " + e)
            res = np.zeros((2, 0))
        else:
            print(len(res), "results")
        if res.size == 0:
            missing.append(fname)
        else:
            if indexkey in allres:
                if allstats[indexkey]['run_version'] > stats['run_version']:
                    # don't use this run
                    continue

            allres[indexkey] = res
            allstats[indexkey] = stats

    return allres, allstats

def extract_pareto_optimal(allres, keys, recall_idx=0, times_idx=3):
    bigtab = []
    for i, k in enumerate(keys):
        v = allres[k]
        perf = v[:, recall_idx]
        times = v[:, times_idx]
        bigtab.append(
            np.vstack((
                np.ones(times.size) * i,
                perf, times
            ))
        )
    if bigtab == []:
        return [], np.zeros((3, 0))

    bigtab = np.hstack(bigtab)

    # sort by perf
    perm = np.argsort(bigtab[1, :])
    bigtab_sorted = bigtab[:, perm]
    best_times = np.minimum.accumulate(bigtab_sorted[2, ::-1])[::-1]
    selection, = np.where(bigtab_sorted[2, :] == best_times)
    selected_keys = [
        keys[i] for i in
        np.unique(bigtab_sorted[0, selection].astype(int))
    ]
    ops = bigtab_sorted[:, selection]

    return selected_keys, ops

def plot_subset(
    allres, allstats, selected_methods, recall_idx, times_idx=3,
    report=["overhead", "build time"]):

    # important methods
    for k in selected_methods:
        v = allres[k]

        stats = allstats[k]
        d = stats["d"]
        dbsize = stats["nb"]
        if "index_size" in stats and "tables_size" in stats:
            tot_size = stats['index_size'] + stats['tables_size']
        else:
            tot_size = -1
        id_size = 8 # 64 bit

        addt = ''
        if 'add_time' in stats:
            add_time = stats['add_time']
            if add_time > 7200:
                add_min = add_time / 60
                addt = ', %dh%02d' % (add_min / 60, add_min % 60)
            else:
                add_sec = int(add_time)
                addt = ', %dm%02d' % (add_sec / 60, add_sec % 60)

        code_size = unitsize(d, k)

        label = k

        if "code_size" in report:
            label += " %d bytes" % code_size

        tight_size = (code_size + id_size) * dbsize

        if tot_size < 0 or "overhead" not in report:
            pass # don't know what the index size is
        elif tot_size > 10 * tight_size:
            label += " overhead x%.1f" % (tot_size / tight_size)
        else:
            label += " overhead+%.1f%%" % (
                tot_size / tight_size * 100 - 100)

        if "build time" in report:
            label += " " + addt

        linestyle = (':' if 'Refine' in k or 'RFlat' in k else
                     '-.' if 'SQ' in k else
                     '-' if '4fs' in k else
                     '-')
        print(k, linestyle)
        pyplot.semilogy(v[:, recall_idx], 1000 / v[:, times_idx], label=label,
                        linestyle=linestyle,
                        marker='o' if '4fs' in k else '+')

    recall_rank = stats["ranks"][recall_idx]
    if stats["measure"] == "recall":
        pyplot.xlabel('1-recall at %d' % recall_rank)
    elif stats["measure"] == "inter":
        pyplot.xlabel('inter @ %d' % recall_rank)
    else:
        assert False
    pyplot.ylabel('QPS (%d threads)' % stats["n_threads"])


def plot_tradeoffs(db, allres, allstats, code_size, recall_rank):
    stat0 = next(iter(allstats.values()))
    d = stat0["d"]
    n_threads = stat0["n_threads"]
    recall_idx = stat0["ranks"].index(recall_rank)
    # times come after the perf measure
    times_idx = len(stat0["ranks"])

    if type(code_size) == int:
        if code_size == 0:
            code_size = [0, 1e50]
            code_size_name = "any code size"
        else:
            code_size_name = "code_size=%d" % code_size
            code_size = [code_size, code_size]
    elif type(code_size) == tuple:
        code_size_name = "code_size in [%d, %d]" % code_size
    else:
        assert False

    names_maxperf = []

    for k in sorted(allres):
        v = allres[k]
        if v.ndim != 2: continue
        us = unitsize(d, k)
        if not code_size[0] <= us <= code_size[1]: continue
        names_maxperf.append((v[-1, recall_idx], k))

    # sort from lowest to highest topline accuracy
    names_maxperf.sort()
    names = [name for mp, name in names_maxperf]

    selected_methods, optimal_points =  \
        extract_pareto_optimal(allres, names, recall_idx, times_idx)

    not_selected = list(set(names) - set(selected_methods))

    print("methods without an optimal OP: ", not_selected)

    pyplot.title('database ' + db + ' ' + code_size_name)

    # grayed out lines

    for k in not_selected:
        v = allres[k]
        if v.ndim != 2: continue
        us = unitsize(d, k)
        if not code_size[0] <= us <= code_size[1]: continue

        linestyle = (':' if 'PQ' in k else
                     '-.' if 'SQ4' in k else
                     '--' if 'SQ8' in k else '-')

        pyplot.semilogy(v[:, recall_idx], 1000 / v[:, times_idx], label=None,
                        linestyle=linestyle,
                        marker='o' if 'HNSW' in k else '+',
                        color='#cccccc', linewidth=0.2)

    plot_subset(allres, allstats, selected_methods, recall_idx, times_idx)


    if len(not_selected) == 0:
        om = ''
    else:
        om = '\nomitted:'
        nc = len(om)
        for m in not_selected:
            if nc > 80:
                om += '\n'
                nc = 0
            om += ' ' + m
            nc += len(m) + 1

    # pyplot.semilogy(optimal_points[1, :], optimal_points[2, :], marker="s")
    # print(optimal_points[0, :])
    pyplot.xlabel('1-recall at %d %s' % (recall_rank, om) )
    pyplot.ylabel('QPS (%d threads)' % n_threads)
    pyplot.legend()
    pyplot.grid()
    return selected_methods, not_selected



if __name__ == "__main__xx":
    # tests on centroids indexing (v1)

    for k in 1, 32, 128:
        pyplot.gcf().set_size_inches(15, 10)
        i = 1
        for ncent in 65536, 262144, 1048576, 4194304:
            db = f'deep_centroids_{ncent}.k{k}.'
            allres, allstats = collect_results_for(
                db=db, prefix="cent_index.")

            pyplot.subplot(2, 2, i)
            plot_subset(
                allres, allstats, list(allres.keys()),
                recall_idx=0,
                times_idx=1,
                report=["code_size"]
            )
            i += 1
            pyplot.title(f"{ncent} centroids")
            pyplot.legend()
            pyplot.xlim([0.95, 1])
            pyplot.grid()

        pyplot.savefig('figs/deep1B_centroids_k%d.png' % k)


if __name__ == "__main__xx":
    # centroids plot per k

    pyplot.gcf().set_size_inches(15, 10)

    i=1
    for ncent in 65536, 262144, 1048576, 4194304:

        xyd = defaultdict(list)

        for k in 1, 4, 8, 16, 32, 64, 128, 256:

            db = f'deep_centroids_{ncent}.k{k}.'
            allres, allstats = collect_results_for(db=db, prefix="cent_index.")

            for indexkey, res in allres.items():
                idx, = np.where(res[:, 0] >= 0.99)
                if idx.size > 0:
                    xyd[indexkey].append((k, 1000 / res[idx[0], 1]))

        pyplot.subplot(2, 2, i)
        i += 1
        for indexkey, xy in xyd.items():
            xy = np.array(xy)
            pyplot.loglog(xy[:, 0], xy[:, 1], 'o-', label=indexkey)

        pyplot.title(f"{ncent} centroids")
        pyplot.xlabel("k")
        xt = 2**np.arange(9)
        pyplot.xticks(xt, ["%d" % x for x in xt])
        pyplot.ylabel("QPS (32 threads)")
        pyplot.legend()
        pyplot.grid()

    pyplot.savefig('../plots/deep1B_centroids_min99.png')





if __name__ == "__main__xx":
    # main indexing plots

    i = 0
    for db in 'bigann10M', 'deep10M', 'bigann100M', 'deep100M', 'deep1B', 'bigann1B':
        allres, allstats = collect_results_for(
            db=db, prefix="autotune.")

        for cs in 8, 16, 32, 64:
            pyplot.figure(i)
            i += 1
            pyplot.gcf().set_size_inches(15, 10)

            cs_range = (
                (0, 8) if cs == 8 else (cs // 2 + 1, cs)
            )

            plot_tradeoffs(
                db, allres, allstats, code_size=cs_range, recall_rank=1)
            pyplot.savefig('../plots/tradeoffs_%s_cs%d_r1.png' % (
                   db, cs))


if __name__ == "__main__":
    # 1M indexes
    i = 0
    for db in "glove", "music-100":
        pyplot.figure(i)
        pyplot.gcf().set_size_inches(15, 10)
        i += 1
        allres, allstats = collect_results_for(db=db, prefix="autotune.")
        plot_tradeoffs(db, allres, allstats, code_size=0, recall_rank=1)
        pyplot.savefig('../plots/1M_tradeoffs_' + db + ".png")

    for db in "sift1M", "deep1M":
        allres, allstats = collect_results_for(db=db, prefix="autotune.")
        pyplot.figure(i)
        pyplot.gcf().set_size_inches(15, 10)
        i += 1
        plot_tradeoffs(db, allres, allstats, code_size=(0, 64), recall_rank=1)
        pyplot.savefig('../plots/1M_tradeoffs_' + db + "_small.png")

        pyplot.figure(i)
        pyplot.gcf().set_size_inches(15, 10)
        i += 1
        plot_tradeoffs(db, allres, allstats, code_size=(65, 10000), recall_rank=1)
        pyplot.savefig('../plots/1M_tradeoffs_' + db + "_large.png")



if __name__ == "__main__xx":
    db = 'sift1M'
    allres, allstats = collect_results_for(db=db, prefix="autotune.")
    pyplot.gcf().set_size_inches(15, 10)

    keys = [
        "IVF1024,PQ32x8",
        "IVF1024,PQ64x4",
        "IVF1024,PQ64x4fs",
        "IVF1024,PQ64x4fsr",
        "IVF1024,SQ4",
        "IVF1024,SQ8"
    ]

    plot_subset(allres, allstats, keys, recall_idx=0, report=["code_size"])

    pyplot.legend()
    pyplot.title(db)
    pyplot.xlabel("1-recall@1")
    pyplot.ylabel("QPS (32 threads)")
    pyplot.grid()

    pyplot.savefig('../plots/ivf1024_variants.png')

    pyplot.figure(2)
    pyplot.gcf().set_size_inches(15, 10)

    keys = [
        "HNSW32",
        "IVF1024,PQ64x4fs",
        "IVF1024,PQ64x4fsr",
        "IVF1024,PQ64x4fs,RFlat",
        "IVF1024,PQ64x4fs,Refine(SQfp16)",
        "IVF1024,PQ64x4fs,Refine(SQ8)",
    ]

    plot_subset(allres, allstats, keys, recall_idx=0, report=["code_size"])

    pyplot.legend()
    pyplot.title(db)
    pyplot.xlabel("1-recall@1")
    pyplot.ylabel("QPS (32 threads)")
    pyplot.grid()

    pyplot.savefig('../plots/ivf1024_rerank.png')
