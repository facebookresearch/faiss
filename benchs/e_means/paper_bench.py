# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Reproduce the e-means paper's Table 3, Figure 4(b) and Figure 3 with faiss.contrib.e_means.

Pick a dataset, seeds and budgets; every run is timed the paper's way (paper_xb.py: a fresh
process, an untimed identical warm-up, both methods on the same GPU, one exact scorer) and
written as one JSON file under --out (default ./results). Existing files are kept, so an
interrupted command resumes where it stopped.

    # any budgets, both methods
    python paper_bench.py run --dataset deep1b-10M --seeds 10 11 --epochs 1 2 --flash-iters 25

    # Table 3's measurement for one dataset: the fewest epochs (1.0, 1.2, ...) at which e-means
    # reaches Flash-KMeans' test error after 25 iterations, and the speedup there
    python paper_bench.py crossing --dataset fb-ssnpp-1M --seeds 10 11 12
    python paper_bench.py crossing --dataset fb-ssnpp-1M --target paper   # no Flash run

    # whole artifacts (one GPU; run a dataset per job to parallelize), then print or plot
    python paper_bench.py table3 --scale 1M          python paper_bench.py table3 --print
    python paper_bench.py figure4 --scale 10M        python paper_bench.py figure4 --plot
    python paper_bench.py figure3 --method emeans    python paper_bench.py figure3 --plot

--target paper reads paper_targets.json: Flash-KMeans' test error and time after 25 iterations
for every (dataset, seed) of the paper, so a crossing can be checked without running Flash.
Speedups are only printed against a Flash run timed in the same job (--target measure, the
default). figure4 --plot draws Flash's curve from the same file when --out holds no Flash runs.

--input cast|bf16 chooses how e-means' input reaches bfloat16 (see paper_xb.py). The two routes
must agree; `compare` prints them side by side when both were run.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as data  # noqa: E402
import paper_xb as xb  # noqa: E402

SEEDS = list(range(10, 20))
LADDER = [round(1.0 + 0.2 * i, 1) for i in range(13)]       # 1.0, 1.2, ..., 3.4 epochs
TABLE3 = {"1M": ["deep1b-1M", "dino10b-1M-div4096", "spacev-1M", "fb-ssnpp-1M"],
          "10M": ["deep1b-10M", "dino10b-10M-div4096", "spacev-10M", "fb-ssnpp-10M"],
          "100M": ["deep1b-100M", "spacev-100M", "fb-ssnpp-100M"]}
TABLE3_SEEDS = {"1M": SEEDS, "10M": SEEDS, "100M": SEEDS}   # the paper: 10 seeds at every scale
FIGURE4 = {"10M": ["deep1b-10M", "spacev-10M", "fb-ssnpp-10M", "sift1b-10M-div512"],
           "100M": ["deep1b-100M", "spacev-100M", "fb-ssnpp-100M", "sift1b-100M-div512"]}
FIG4_EPOCHS = [0.125, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 17, 25]
FIG4_ITERATIONS = [1, 2, 3, 5, 8, 12, 17, 25, 40, 60, 80, 100]
FIGURE3 = {"deep1b-1M": "DEEP* 1M", "imagenet-clip-512-normalized": "ImageNet-CLIP 1M",
           "spacev-10M": "SPACEV 10M", "sift1b-10M-normalized": "SIFT 10M"}
FIG3_BUDGETS = [1, 2, 3, 5, 8, 12, 17, 25]
TITLES = {"deep1b": "DEEP*", "spacev": "SPACEV", "fb-ssnpp": "SimSearchNet++",
          "sift1b": "SIFT/512", "dino10b": "DINOv3/4096"}
TARGETS = os.path.join(HERE, "paper_targets.json")


def title(name):
    return TITLES[name.split("-")[0] if not name.startswith("fb-ssnpp") else "fb-ssnpp"]


def route_for(name, route):
    """The input route a dataset runs with: the one asked for, else paper_xb's default for its
    shape (cast where a float32 copy fits one GPU)."""
    if route:
        return route
    fp = json.load(open(os.path.join(data.SPLITS, "fingerprints.json")))[name]
    return xb.default_input(fp["n_train"], fp["dim"])


def save(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    json.dump(record, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def timed_run(out, spec):
    """One timed run, cached as a JSON file named after everything that varies."""
    if spec["method"] == "emeans":
        spec = dict(spec, input=route_for(spec["dataset"], spec.get("input")))
    budget = ("e%g_%s" % (spec["epochs"], spec["input"])
              if spec["method"] == "emeans" else "i%d" % spec["iterations"])
    path = os.path.join(out, "runs", spec["dataset"], "%s_%s_s%d.json" % (
        spec["method"], budget, spec["seed"]))
    if os.path.exists(path):
        return json.load(open(path))
    record = xb.isolated(spec)
    save(record, path)
    if spec["method"] == "emeans":
        print("%s seed %d: e-means %g epochs (%s): test %.6g in %.4f s" % (
            spec["dataset"], spec["seed"], spec["epochs"], record["input"], record["test_mse"],
            record["seconds"]), flush=True)
    else:
        print("%s seed %d: Flash-KMeans %d iterations: test %.6g in %.4f s" % (
            spec["dataset"], spec["seed"], spec["iterations"], record["test_mse"],
            record["seconds"]), flush=True)
    return record


def paper_target(name, seed):
    targets = json.load(open(TARGETS))
    return (targets["table3"].get(name) or targets["figure4"][name])["per_seed"][str(seed)]


def crossing(out, name, seeds, target="measure", route=None, ladder=LADDER):
    """Per seed: Flash@25's test error (measured here, or the paper's), then e-means at 1.0,
    1.2, ... epochs until it is at or below it. Writes <out>/crossing/<dataset>_<input>_s<seed>."""
    rows = []
    for seed in seeds:
        if target == "measure":
            flash = timed_run(out, dict(method="flash", dataset=name, seed=seed, iterations=25))
            goal = flash["test_mse"]
        else:
            flash, goal = None, paper_target(name, seed)["test_mse"]
        runs = []
        for epochs in ladder:
            runs.append(timed_run(out, dict(method="emeans", dataset=name, seed=seed,
                                            epochs=epochs, input=route)))
            if runs[-1]["test_mse"] <= goal:
                break
        hit = runs[-1] if runs[-1]["test_mse"] <= goal else None
        record = dict(dataset=name, seed=seed, target=goal, target_source=target, flash=flash,
                      emeans=runs, crossing=hit["epochs"] if hit else None,
                      speedup=flash["seconds"] / hit["seconds"] if hit and flash else None)
        save(record, os.path.join(out, "crossing", "%s_%s_s%d.json" % (
            name, runs[-1]["input"], seed)))
        rows.append(record)
        print("%s seed %d: crossing %s epochs%s" % (
            name, seed, record["crossing"],
            ", speedup %.1fx" % record["speedup"] if record["speedup"] else ""), flush=True)
    return rows


def cell(out, name, route=None):
    """Table 3 cell from the crossing files: upper median of the crossings, median of the
    per-seed speedups, and the seeds behind them."""
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(
        out, "crossing", "%s_%s_s*.json" % (name, route_for(name, route)))))]
    recs = [r for r in recs if r["target_source"] == "measure"]
    hits = [r for r in recs if r["crossing"] is not None]
    if not hits:
        return None
    cross = sorted(r["crossing"] for r in hits)
    return dict(seeds=len(recs), crossed=len(hits), crossing=cross[len(cross) // 2],
                speedup=float(np.median([r["speedup"] for r in hits])), crossings=cross)


def print_table3(out, route):
    for scale, names in TABLE3.items():
        print("%s rows" % scale)
        for name in names:
            c = cell(out, name, route)
            print("  %-20s %s" % (name, "no runs" if c is None else
                                  "%2d seeds (%d crossed): %.1f epochs, %5.1fx   crossings %s" % (
                                      c["seeds"], c["crossed"], c["crossing"], c["speedup"],
                                      c["crossings"])))


def figure4_runs(out, name, seeds, route):
    for seed in seeds:
        for iterations in FIG4_ITERATIONS:
            timed_run(out, dict(method="flash", dataset=name, seed=seed, iterations=iterations))
        for epochs in FIG4_EPOCHS:
            timed_run(out, dict(method="emeans", dataset=name, seed=seed, epochs=epochs,
                                input=route))


def curves(out, name, route):
    """(base, {"flash": [(budget, seconds, % above base)], "emeans": [...]}, seeds): medians
    over seeds, base being Flash's median test error after 25 iterations."""
    folder = os.path.join(out, "runs", name)
    flash = [json.load(open(p)) for p in glob.glob(os.path.join(folder, "flash_i*_s*.json"))]
    if not flash and os.path.exists(TARGETS):          # no Flash runs here: the paper's curve
        curve = json.load(open(TARGETS))["figure4"].get(name, {}).get("flash_curve", {})
        flash = [dict(iterations=int(it), seed=int(seed), **v)
                 for it, seeds in curve.items() for seed, v in seeds.items()]
    em = [json.load(open(p)) for p in glob.glob(os.path.join(folder, "emeans_e*_s*.json"))]
    em = [r for r in em if r["input"] == route_for(name, route) and r["epochs"] in FIG4_EPOCHS]
    base = [r["test_mse"] for r in flash if r["iterations"] == 25]
    if not base or not em:
        return None, {}, 0
    base = float(np.median(base))
    pts = {}
    for method, recs, key in (("flash", flash, "iterations"), ("emeans", em, "epochs")):
        pts[method] = [(b, float(np.median([r["seconds"] for r in recs if r[key] == b])),
                        float(np.median([100.0 * (r["test_mse"] / base - 1.0) for r in recs
                                         if r[key] == b])))
                       for b in sorted({r[key] for r in recs})]
    return base, pts, len({r["seed"] for r in em})


def plot_figure4(out, route):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = [s for s in FIGURE4 if any(curves(out, n, route)[1] for n in FIGURE4[s])]
    fig, axes = plt.subplots(len(scales), 4, figsize=(13.6, 3.4 * len(scales)), squeeze=False)
    for row, scale in zip(axes, scales):
        for ax, name in zip(row, FIGURE4[scale]):
            base, pts, seeds = curves(out, name, route)
            if not pts:
                ax.set_title("%s %s (no runs)" % (title(name), scale))
                continue
            flash25 = [s for b, s, _ in pts["flash"] if b == 25][0]
            cross = next(((b, s) for b, s, q in pts["emeans"] if q <= 0), None)
            print("%-20s %2d seeds: first e-means budget at or below Flash@25: %s, "
                  "25 epochs %+.2f%%" % (name, seeds, "%g epochs (%.1fx)" % (
                      cross[0], flash25 / cross[1]) if cross else "none", pts["emeans"][-1][2]))
            for method, colour, marker, label in (("emeans", "#1b6ca8", "o", "e-means"),
                                                  ("flash", "#c1121f", "s", "Flash-KMeans")):
                ax.plot([s for _, s, _ in pts[method]], [q for _, _, q in pts[method]],
                        color=colour, marker=marker, ms=3.5, lw=1.4, label=label)
            ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_ylim(min(q for m in pts.values() for _, _, q in m) - 0.2, 3.0)
            ax.set_title("%s %s, %d seeds" % (title(name), scale, seeds))
            ax.set_xlabel("seconds (median over seeds)")
            ax.grid(True, lw=0.3, alpha=0.5)
        row[0].set_ylabel("test MSE, % above Flash@25")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure4.pdf"))
    print("wrote %s" % os.path.join(out, "figure4.pdf"))


def figure3_runs(out, name, method, seeds, route):
    """Figure 3: training and test error against epochs, e-means (GPU) against Faiss k-means
    (CPU, 25 Lloyd iterations read off after 1, 2, 3, 5, ...). Both start from the paper's
    starting rows; the e-means arm draws them on the GPU and saves them for the Faiss arm, so run
    it first. Training error is each method's error on its OWN last-pass assignment over every
    training row, so neither spends an extra pass on scoring."""
    import faiss
    faiss.omp_set_num_threads(len(os.sched_getaffinity(0)))
    train, test = data.load(name)
    x = data.rows(train, name)                                # (n, d) float32 in memory
    n, d = x.shape
    k = xb.k_for(n)
    for seed in seeds:
        path = os.path.join(out, "figure3", "%s_%s_s%d.json" % (
            name, method if method == "faiss" else "emeans_" + (route or "cast"), seed))
        if os.path.exists(path):
            continue
        start_path = os.path.join(data.CACHE, name, "starts_s%d.npy" % seed)
        if not os.path.exists(start_path):
            if method == "faiss":
                sys.exit("%s seed %d: run the e-means arm first (it draws the starting rows on "
                         "the GPU)" % (name, seed))
            np.save(start_path, xb.starts(n, k, seed))
        c0 = x[np.load(start_path)]                           # (k, d) the shared start
        points = (fig3_emeans if method == "emeans" else fig3_faiss)(x, test, k, seed, c0, route)
        save(dict(dataset=name, method=method, input=route, seed=seed, k=k, points=points), path)


def _train_mse(x, centroids, codes, chunk=1 << 20):
    total = 0.0
    for s in range(0, len(x), chunk):
        diff = x[s:s + chunk] - centroids[codes[s:s + chunk]]
        total += float(np.einsum("ij,ij->", diff, diff, dtype=np.float64))
    return total / len(x)


def fig3_emeans(x, test, k, seed, c0, route):
    import torch
    from faiss.contrib.e_means import Emeans, QuantizedEmeans
    from faiss.contrib.e_means.quantizer import CastQuantizer
    n, d = x.shape
    route = route or "cast"
    xg = torch.from_numpy(x).cuda()
    model = Emeans
    kw = dict(seed=seed, batch_law=xb.batch_law(n), assignment_backend="cute",
              capture_assignments=True)
    if route == "cast":
        model = QuantizedEmeans
        kw["quantizer"] = CastQuantizer(d, torch.bfloat16)
    else:
        xg = xg.to(torch.bfloat16)
    c0g = torch.from_numpy(c0).cuda()
    out = []
    for epochs in FIG3_BUDGETS:
        km = model(d, k, nepochs=epochs, **kw)
        km.train(xg, init_centroids=c0g)
        c = km.centroids.cpu().numpy()
        out.append(dict(budget=epochs, train=_train_mse(x, c, km.assignments.cpu().numpy()),
                        test=xb.test_error(c, test)))
        print("seed %d e-means %2d epochs: train %.6g test %.6g" % (
            seed, epochs, out[-1]["train"], out[-1]["test"]), flush=True)
    return out


def fig3_faiss(x, test, k, seed, c0, route):
    import faiss
    n, d = x.shape
    centroids, out = c0, []
    for iteration in range(1, max(FIG3_BUDGETS) + 1):
        if iteration in FIG3_BUDGETS:      # the assignment this iteration is about to make
            _, codes = faiss.knn(x, centroids, 1)
        km = faiss.Kmeans(d, k, niter=1, nredo=1, seed=seed, max_points_per_centroid=10**9,
                          min_points_per_centroid=1, verbose=False)
        km.train(x, init_centroids=centroids)                 # one Lloyd iteration
        centroids = km.centroids
        if iteration in FIG3_BUDGETS:
            _, nearest = faiss.knn(test, centroids, 1)
            diff = test - centroids[nearest[:, 0]]
            out.append(dict(budget=iteration, train=_train_mse(x, centroids, codes[:, 0]),
                            test=float(np.einsum("ij,ij->", diff, diff, dtype=np.float64)
                                       / len(test))))
            print("seed %d Faiss %2d iterations: train %.6g test %.6g" % (
                seed, iteration, out[-1]["train"], out[-1]["test"]), flush=True)
    return out


def plot_figure3(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(FIGURE3), figsize=(13.6, 6.0), squeeze=False)
    for j, (name, label) in enumerate(FIGURE3.items()):
        recs = [json.load(open(p)) for p in glob.glob(os.path.join(
            out, "figure3", "%s_*_s*.json" % name))]
        faiss_recs = [r for r in recs if r["method"] == "faiss"]
        if not faiss_recs:
            continue
        for i, what in enumerate(("train", "test")):
            ref = np.mean([r["points"][-1][what] for r in faiss_recs])
            for method, colour in (("emeans", "#1b6ca8"), ("faiss", "#555555")):
                rr = [r for r in recs if r["method"] == method]
                if not rr:
                    continue
                budgets = [p["budget"] for p in rr[0]["points"]]
                ys = [np.mean([r["points"][b][what] for r in rr]) / ref
                      for b in range(len(budgets))]
                axes[i][j].plot(budgets, ys, marker="o", ms=3, color=colour,
                                label="%s (%d seeds)" % (method, len(rr)))
            axes[i][j].set_xscale("log")
            axes[i][j].axhline(1.0, color="0.6", lw=0.8, ls="--")
            axes[i][j].set_title("%s, %s error" % (label, what))
            axes[i][j].grid(True, lw=0.3, alpha=0.5)
    axes[0][0].legend(frameon=False)
    axes[1][0].set_xlabel("epochs")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure3.pdf"))
    print("wrote %s" % os.path.join(out, "figure3.pdf"))


def build_targets(out):
    """paper_targets.json from a finished campaign: every Flash@25 test error and time, per
    (dataset, seed), for Table 3 and Figure 4(b); Flash's whole Figure 4(b) curve (1 to 100
    iterations) per seed; and Faiss@25 for Figure 3."""
    import platform
    result = dict(note="Flash-KMeans after 25 iterations (Table 3, Figure 4(b)) and its whole "
                       "Figure 4(b) curve (flash_curve: iterations -> seed -> test_mse, seconds); "
                       "Faiss k-means after 25 Lloyd iterations (Figure 3, which has no Flash "
                       "arm). Test error per row, exact scorer of paper_xb.test_error; seconds "
                       "are one H100, median of the timed calls (paper_xb.timed).",
                  host=platform.node(), table3={}, figure4={}, figure3={})
    for key, names in (("table3", sum(TABLE3.values(), [])),
                       ("figure4", sum(FIGURE4.values(), []))):
        for name in names:
            recs = [json.load(open(p)) for p in glob.glob(os.path.join(
                out, "runs", name, "flash_i25_s*.json"))]
            if recs:
                r0 = recs[0]
                result[key][name] = dict(
                    flash_api=r0["flash_api"], flash_norms=r0["flash_norms"], gpu=r0["gpu"],
                    torch=r0["torch"], triton=r0["triton"],
                    median_test_mse=float(np.median([r["test_mse"] for r in recs])),
                    per_seed={str(r["seed"]): dict(test_mse=r["test_mse"], seconds=r["seconds"])
                              for r in sorted(recs, key=lambda r: r["seed"])})
                if key == "figure4":                   # Flash's whole curve, for --plot
                    curve = {}
                    for it in FIG4_ITERATIONS:
                        for p in sorted(glob.glob(os.path.join(out, "runs", name,
                                                               "flash_i%d_s*.json" % it))):
                            r = json.load(open(p))
                            curve.setdefault(str(it), {})[str(r["seed"])] = dict(
                                test_mse=r["test_mse"], seconds=r["seconds"])
                    result[key][name]["flash_curve"] = curve
    for name in FIGURE3:
        recs = [json.load(open(p)) for p in glob.glob(os.path.join(
            out, "figure3", "%s_faiss_s*.json" % name))]
        if recs:
            result["figure3"][name] = dict(
                reference="faiss k-means, 25 Lloyd iterations, CPU",
                median_test_mse=float(np.median([r["points"][-1]["test"] for r in recs])),
                per_seed={str(r["seed"]): dict(test_mse=r["points"][-1]["test"],
                                               train_mse=r["points"][-1]["train"])
                          for r in sorted(recs, key=lambda r: r["seed"])})
    save(result, TARGETS)
    print("wrote %s" % TARGETS)


def compare_inputs(out):
    """The two input routes side by side, per (dataset, budget, seed): they must agree."""
    for folder in sorted(glob.glob(os.path.join(out, "runs", "*"))):
        name = os.path.basename(folder)
        recs = {}
        for p in glob.glob(os.path.join(folder, "emeans_e*_s*.json")):
            r = json.load(open(p))
            recs.setdefault((r["epochs"], r["seed"]), {})[r["input"]] = r
        pairs = [v for v in recs.values() if len(v) == 2]
        if not pairs:
            continue
        err = [100.0 * (v["cast"]["test_mse"] / v["bf16"]["test_mse"] - 1.0) for v in pairs]
        time_ratio = [v["cast"]["seconds"] / v["bf16"]["seconds"] for v in pairs]
        print("%-20s %3d pairs: cast vs bf16 test error %+.4f%% median (max |%.4f|%%), "
              "time ratio %.3f median (%.3f-%.3f)" % (
                  name, len(pairs), np.median(err), np.max(np.abs(err)), np.median(time_ratio),
                  min(time_ratio), max(time_ratio)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "crossing", "table3", "figure4", "figure3",
                                        "targets", "compare"])
    ap.add_argument("--dataset", choices=sorted(data.DATASETS))
    ap.add_argument("--scale", choices=["1M", "10M", "100M"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--method", choices=["emeans", "flash", "both", "faiss"], default="both")
    ap.add_argument("--epochs", type=float, nargs="+", default=[])
    ap.add_argument("--flash-iters", type=int, nargs="+", default=[])
    ap.add_argument("--input", choices=["cast", "bf16"],
                    help="how e-means' input reaches bfloat16 (default: cast where a float32 "
                         "copy fits one GPU, else bf16)")
    ap.add_argument("--target", choices=["measure", "paper"], default="measure")
    ap.add_argument("--out", default="results")
    ap.add_argument("--print", action="store_true", help="table3: print the rows")
    ap.add_argument("--plot", action="store_true", help="figure3/figure4: draw the figure")
    a = ap.parse_args()
    if a.command == "run":
        for seed in a.seeds or [10]:
            if a.method in ("flash", "both"):
                for it in a.flash_iters:
                    timed_run(a.out, dict(method="flash", dataset=a.dataset, seed=seed,
                                          iterations=it))
            if a.method in ("emeans", "both"):
                for ep in a.epochs:
                    timed_run(a.out, dict(method="emeans", dataset=a.dataset, seed=seed,
                                          epochs=ep, input=a.input))
    elif a.command == "crossing":
        crossing(a.out, a.dataset, a.seeds or SEEDS, a.target, a.input)
    elif a.command == "table3":
        if a.print:
            print_table3(a.out, a.input)
            return
        names = [a.dataset] if a.dataset else TABLE3[a.scale]
        for name in names:
            scale = next(s for s, v in TABLE3.items() if name in v)
            crossing(a.out, name, a.seeds or TABLE3_SEEDS[scale], "measure", a.input)
    elif a.command == "figure4":
        if a.plot:
            plot_figure4(a.out, a.input)
            return
        for name in [a.dataset] if a.dataset else FIGURE4[a.scale]:
            figure4_runs(a.out, name, a.seeds or SEEDS, a.input)
    elif a.command == "figure3":
        if a.plot:
            plot_figure3(a.out)
            return
        for name in [a.dataset] if a.dataset else list(FIGURE3):
            figure3_runs(a.out, name, "faiss" if a.method == "faiss" else "emeans",
                         a.seeds or SEEDS, a.input)
    elif a.command == "targets":
        build_targets(a.out)
    else:
        compare_inputs(a.out)


if __name__ == "__main__":
    main()
