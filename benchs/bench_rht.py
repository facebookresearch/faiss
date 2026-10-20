#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Reproducible L2 search benchmark for one to three native RHT rounds.

See README_rht.md for data splits, score semantics, and interpretation of CIs.
Helpers can be tested without importing the native Faiss extension.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# The CLI starts numerical runtimes with the requested limit. Runtime contexts
# below also cover programmatic callers that imported NumPy before this module.
if __name__ == "__main__":
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument("--threads", type=int, default=1)
    early_args, _ = early_parser.parse_known_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(early_args.threads)

import numpy as np


MODES = ("none", "urr", "rht1", "rht2", "rht3")
MODE_LABELS = {
    "none": "none",
    "urr": "URR",
    "rht1": "1RHT",
    "rht2": "2RHT",
    "rht3": "3RHT",
}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--npz", type=Path, help="NPZ containing xb, xq, and optionally xt"
    )
    source.add_argument(
        "--data-dir", type=Path, help="Directory containing fvecs files"
    )
    source.add_argument(
        "--synthetic", choices=("gaussian", "student_t", "sparse32"), default="gaussian"
    )
    parser.add_argument("--base", default="base.fvecs")
    parser.add_argument("--queries", default="query.fvecs")
    parser.add_argument(
        "--train", help="Optional training fvecs; otherwise reserve base rows"
    )
    parser.add_argument("--nt", type=int, default=512)
    parser.add_argument("--nb", type=int, default=1000)
    parser.add_argument("--nq", type=int, default=64)
    parser.add_argument("--d", type=int, default=64, help="Synthetic dimension")
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument(
        "--check-disjoint",
        action="store_true",
        help="Check exact float32 vector duplicates between training and base",
    )
    parser.add_argument(
        "--hash-files", action="store_true", help="SHA-256 all input files"
    )
    parser.add_argument(
        "--indexes", nargs="+", choices=("RaBitQ", "EDEN"), default=["RaBitQ", "EDEN"]
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", choices=(2, 3, 4), default=[2, 3, 4]
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(12345, 12350)),
        help="Transform seeds; five or more recommended",
    )
    parser.add_argument("--qb", type=int, default=4, help="RaBitQ query bits")
    parser.add_argument("--pad-baselines", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--require-host", help="Refuse to run unless the short hostname matches"
    )
    parser.add_argument(
        "--nmse-max-vectors",
        type=int,
        default=10000,
        help="First base vectors for reconstruction diagnostics; 0 means all",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=9876)
    parser.add_argument("--output", type=Path, default=Path("rht_results.json"))
    return parser


def read_fvecs(path):
    """Read little-endian fvecs, checking every dimension header."""
    words = np.memmap(path, dtype="<i4", mode="r")
    if words.size == 0 or words[0] <= 0:
        raise ValueError(f"empty fvecs file or invalid dimension: {path}")
    dimension = int(words[0])
    if words.size % (dimension + 1):
        raise ValueError(
            f"fvecs file size does not match dimension {dimension}: {path}"
        )
    rows = words.reshape(-1, dimension + 1)
    if not np.all(rows[:, 0] == dimension):
        raise ValueError(f"inconsistent fvecs dimension headers: {path}")
    return rows[:, 1:].view("<f4")


def file_record(path, hash_file):
    path = Path(path).resolve()
    result = {"path": str(path), "size_bytes": path.stat().st_size}
    if hash_file:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def load_data(args):
    """Load one fixed dataset, independent of the transform seed sweep."""
    if min(args.nt, args.nb, args.nq) <= 0:
        raise ValueError("nt, nb, and nq must be positive")
    provenance = {"files": []}
    if args.npz:
        provenance["source"] = "npz"
        provenance["files"] = [file_record(args.npz, args.hash_files)]
        with np.load(args.npz, allow_pickle=False) as data:
            xb, xq = data["xb"], data["xq"]
            xt = data["xt"] if "xt" in data else None
    elif args.data_dir:
        provenance["source"] = "fvecs"
        paths = [args.data_dir / args.base, args.data_dir / args.queries]
        xb, xq = [read_fvecs(path) for path in paths]
        xt = None
        if args.train:
            paths.append(args.data_dir / args.train)
            xt = read_fvecs(paths[-1])
        provenance["files"] = [file_record(path, args.hash_files) for path in paths]
    else:
        if args.d <= 0:
            raise ValueError("d must be positive")
        provenance.update(
            source="synthetic", distribution=args.synthetic, data_seed=args.data_seed
        )
        rng = np.random.default_rng(args.data_seed)

        def draw(n):
            if args.synthetic == "student_t":
                # t(3) has variance 3; normalize to unit coordinate variance.
                return (rng.standard_t(3, (n, args.d)) / np.sqrt(3)).astype("float32")
            if args.synthetic == "sparse32":
                if args.d < 32:
                    raise ValueError("sparse32 requires d >= 32")
                x = np.zeros((n, args.d), dtype="float32")
                for row in x:
                    row[rng.choice(args.d, 32, replace=False)] = rng.standard_normal(32)
                return x
            return rng.standard_normal((n, args.d), dtype=np.float32)

        xt, xb, xq = draw(args.nt), draw(args.nb), draw(args.nq)
        provenance[
            "training_base_disjointness"
        ] = "independent generated samples; exact vector duplicates not checked"
    if xt is None:
        if len(xb) < args.nt + args.nb:
            raise ValueError(
                "base needs at least nt + nb rows when training is held out"
            )
        xt, xb = xb[: args.nt], xb[args.nt : args.nt + args.nb]
        provenance.update(
            training_source_rows=[0, args.nt],
            base_source_rows=[args.nt, args.nt + args.nb],
            training_base_disjointness="disjoint source rows; exact vector duplicates not checked",
        )
    elif "training_base_disjointness" not in provenance:
        provenance[
            "training_base_disjointness"
        ] = "separate supplied arrays/files; vector disjointness not checked"
    selected = []
    for name, values, count in (
        ("xt", xt, args.nt),
        ("xb", xb, args.nb),
        ("xq", xq, args.nq),
    ):
        if values.ndim != 2 or len(values) < count or values.shape[1] <= 0:
            raise ValueError(f"{name} must be 2D and contain at least {count} rows")
        values = np.ascontiguousarray(values[:count], dtype="float32")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values")
        selected.append(values)
    xt, xb, xq = selected
    if xt.shape[1] != xb.shape[1] or xq.shape[1] != xb.shape[1]:
        raise ValueError("training, base, and query dimensions must match")
    provenance["selected_shapes"] = {
        name: list(x.shape) for name, x in zip(("xt", "xb", "xq"), selected)
    }
    if args.check_disjoint:
        # Byte keys use the benchmark's float32 representation. Normalize signed
        # zeros so numerically identical zero vectors compare equal.
        training = set()
        for row in xt:
            training.add(np.where(row == 0, 0, row).astype("float32").tobytes())
        overlap = sum(
            np.where(row == 0, 0, row).astype("float32").tobytes() in training
            for row in xb
        )
        provenance["exact_training_base_overlap_rows"] = overlap
        provenance[
            "training_base_disjointness"
        ] = f"checked exact float32 values: {overlap} base rows also occur in training"
    return xt, xb, xq, provenance


def per_query_recall(truth, predicted):
    """Set Recall@k, so duplicate IDs and missing (-1) predictions earn no credit."""
    if truth.ndim != 2 or truth.shape != predicted.shape or truth.shape[1] == 0:
        raise ValueError(
            "truth and predicted must have matching nonempty (nq, k) shapes"
        )
    k = truth.shape[1]
    return np.array(
        [
            len(set(gt).intersection(int(i) for i in pred if i >= 0)) / k
            for gt, pred in zip(truth, predicted)
        ],
        dtype="float64",
    )


def paired_bootstrap(treatment, baseline, samples, seed):
    """Paired two-way bootstrap over transform seeds and shared query indices."""
    treatment, baseline = np.asarray(treatment), np.asarray(baseline)
    if treatment.ndim != 2 or treatment.shape != baseline.shape or not treatment.size:
        raise ValueError(
            "paired recall arrays must have matching (seeds, queries) shapes"
        )
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    differences = treatment - baseline
    ns, nq = differences.shape
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        seeds = rng.integers(ns, size=ns)
        queries = rng.integers(nq, size=nq)
        draws[i] = differences[np.ix_(seeds, queries)].mean()
    return {
        "delta": float(differences.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "seed_count": ns,
        "query_count": nq,
        "bootstrap_samples": samples,
        "method": "paired two-way seed/query percentile bootstrap",
        "conditional_on": "fixed training/base dataset and supplied query population",
        "limitation": "few transform seeds give imprecise seed uncertainty"
        if ns < 5
        else None,
    }


def haar_matrix(d, seed):
    """Haar orthogonal matrix from Gaussian QR with diagonal sign correction."""
    gaussian = np.random.default_rng(seed).standard_normal((d, d))
    q, r = np.linalg.qr(gaussian)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return np.ascontiguousarray(q * signs, dtype="float32")


def make_index(faiss, d, family, bits, mode, seed, qb, pad_baselines):
    """Build a native IndexPreTransform and return its leaf and dimensions."""
    padded_d = 1 << (d - 1).bit_length()
    d_out = padded_d if mode.startswith("rht") or pad_baselines else d
    transform = None
    if mode.startswith("rht"):
        transform = faiss.HadamardRotation(d, seed, int(mode[-1]))
    elif mode == "urr":
        transform = faiss.LinearTransform(d, d_out)
        matrix = haar_matrix(d_out, seed)[:, :d]
        faiss.copy_array_to_vector(matrix.ravel(), transform.A)
        transform.is_trained = True
        transform.is_orthonormal = True
    elif mode == "none" and d_out != d:
        transform = faiss.RemapDimensionsTransform(d, d_out, False)
    elif mode != "none":
        raise ValueError(f"unsupported mode: {mode}")
    leaf = faiss.index_factory(d_out, f"{family}{bits}", faiss.METRIC_L2)
    if family == "RaBitQ":
        leaf.qb = qb
    index = leaf if transform is None else faiss.IndexPreTransform(transform, leaf)
    return index, leaf, d_out


def search_batches(index, xq, k, batch_size):
    """Complete search including native query transforms, allocation, and batching."""
    predictions = np.empty((len(xq), k), dtype="int64")
    for start in range(0, len(xq), batch_size):
        end = min(start + batch_size, len(xq))
        _, predictions[start:end] = index.search(xq[start:end], k)
    return predictions


def apply_transform(faiss, index, values):
    if isinstance(index, faiss.IndexPreTransform):
        for position in range(index.chain.size()):
            transform = faiss.downcast_VectorTransform(index.chain.at(position))
            values = transform.apply(values)
    return values


def reconstruct_leaf(faiss, leaf, start, count):
    """Decode full stored multibit RaBitQ values, or use EDEN's native decoder.

    RaBitQ's native sa_decode/reconstruct_n exposes only its sign-bit baseline.
    The extended code's IP factor gives the full reconstruction vector used
    in its approximate L2 cross term. This diagnostic is independent of qb.
    """
    if not hasattr(leaf, "rabitq") or int(leaf.rabitq.nb_bits) == 1:
        return leaf.reconstruct_n(start, count)
    d, bits = int(leaf.d), int(leaf.rabitq.nb_bits)
    ex_bits = bits - 1
    binary_size = (d + 7) // 8
    ex_code_offset = binary_size + 12  # three float32 sign-only factors
    ex_factors_offset = ex_code_offset + (d * ex_bits + 7) // 8
    all_codes = faiss.rev_swig_ptr(leaf.codes.data(), leaf.codes.size()).reshape(
        -1, int(leaf.code_size)
    )
    codes = all_codes[start : start + count]
    signs = np.unpackbits(codes[:, :binary_size], axis=1, bitorder="little")[:, :d]
    planes = np.unpackbits(
        codes[:, ex_code_offset:ex_factors_offset], axis=1, bitorder="little"
    )[:, : d * ex_bits].reshape(count, d, ex_bits)
    extra = np.sum(planes * (1 << np.arange(ex_bits)), axis=2, dtype="float32")
    factors = (
        codes[:, ex_factors_offset : ex_factors_offset + 8]
        .copy()
        .view("float32")
        .reshape(count, 2)
    )
    code_values = (
        signs.astype("float32") * (1 << ex_bits) + extra - ((1 << ex_bits) - 0.5)
    )
    scales = -0.5 * factors[:, 1]
    center = faiss.vector_to_array(leaf.center)
    return center + scales[:, None] * code_values


def reconstruction_metrics(values, reconstructed, center):
    """Transformed-domain errors relative to centered original residuals."""
    error = np.sum((values.astype("float64") - reconstructed) ** 2, axis=1)
    norm = np.sum((values.astype("float64") - center) ** 2, axis=1)
    nonzero = norm > 0
    return {
        "nmse": float(error.sum() / norm.sum())
        if norm.sum() > 0
        else (0.0 if error.sum() == 0 else None),
        "vnmse": float(np.mean(error[nonzero] / norm[nonzero]))
        if nonzero.any()
        else None,
        "squared_error_sum": float(error.sum()),
        "residual_squared_norm_sum": float(norm.sum()),
        "zero_residual_rows": int((~nonzero).sum()),
        "zero_residual_with_error_rows": int(
            np.count_nonzero((~nonzero) & (error > 0))
        ),
        "vector_count": len(values),
        "domain": "transformed coordinates relative to native trained center",
    }


def summarize_rows(rows, bootstrap, bootstrap_seed):
    groups = {}
    for row in rows:
        key = row["index"], row["bits"], row["batch_size"], row["mode"]
        groups.setdefault(key, {})[row["seed"]] = row
    summaries = []
    for (family, bits, batch_size, mode), by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        values = [by_seed[seed] for seed in seeds]
        summary = {
            "index": family,
            "bits": bits,
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "batch_size": batch_size,
            "seeds": seeds,
            "recall_mean": float(np.mean([row["recall_by_query"] for row in values])),
            "qps_median_across_seeds": float(np.median([row["qps"] for row in values])),
            "train_seconds_median": float(
                np.median([row["train_seconds"] for row in values])
            ),
            "add_seconds_median": float(
                np.median([row["add_seconds"] for row in values])
            ),
            "d_out": values[0]["d_out"],
            "code_size_bytes": values[0]["code_size_bytes"],
            "serialized_size_bytes_minmax": [
                min(row["serialized_size_bytes"] for row in values),
                max(row["serialized_size_bytes"] for row in values),
            ],
        }
        for metric in ("nmse", "vnmse"):
            if all(metric in row and row[metric] is not None for row in values):
                summary[metric + "_mean_across_seeds"] = float(
                    np.mean([row[metric] for row in values])
                )
                summary[metric + "_by_seed"] = [row[metric] for row in values]
        baseline = groups.get((family, bits, batch_size, "rht1"))
        if baseline is not None and mode != "rht1":
            paired_seeds = sorted(set(seeds).intersection(baseline))
            recall = np.array(
                [by_seed[seed]["recall_by_query"] for seed in paired_seeds]
            )
            recall_ref = np.array(
                [baseline[seed]["recall_by_query"] for seed in paired_seeds]
            )
            paired = paired_bootstrap(recall, recall_ref, bootstrap, bootstrap_seed)
            paired["seeds"] = paired_seeds
            paired["delta_by_seed"] = (recall - recall_ref).mean(axis=1).tolist()
            summary["recall_delta_vs_1rht"] = paired
            summary["search_speedup_vs_1rht_by_seed"] = [
                baseline[seed]["search_seconds_median"]
                / by_seed[seed]["search_seconds_median"]
                for seed in paired_seeds
            ]
            for metric in ("nmse", "vnmse"):
                if all(
                    by_seed[seed].get(metric) is not None
                    and baseline[seed].get(metric) is not None
                    for seed in paired_seeds
                ):
                    treatment = [[by_seed[seed][metric]] for seed in paired_seeds]
                    reference = [[baseline[seed][metric]] for seed in paired_seeds]
                    paired_metric = paired_bootstrap(
                        treatment, reference, bootstrap, bootstrap_seed
                    )
                    paired_metric.pop("query_count")
                    paired_metric.update(
                        method="paired seed percentile bootstrap",
                        conditional_on="fixed training/base data and reconstruction vector sample",
                    )
                    summary[metric + "_delta_vs_1rht"] = paired_metric
        summaries.append(summary)
    return summaries


def metadata(faiss, args):
    from threadpoolctl import threadpool_info

    native_paths = {
        name: module.__file__
        for name, module in sys.modules.copy().items()
        if name.startswith("faiss.")
        and getattr(module, "__file__", "").endswith((".pyd", ".so"))
    }
    result = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "faiss_module_path": faiss.__file__,
        "native_module_paths": native_paths,
        "compile_options": faiss.get_compile_options(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "threads_requested": args.threads,
        "faiss_omp_threads": faiss.omp_get_max_threads(),
        "thread_control": "runtime threadpoolctl limits for BLAS/OpenMP plus Faiss OpenMP setting",
        "threadpools": threadpool_info(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",)
        },
    }
    root = Path(__file__).resolve().parents[1]
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        result["git_revision"] = None
    return result


def write_results(path, report):
    """Checkpoint completed cases, keeping the prior checkpoint on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 7:
                raise
            # Windows sync/indexing clients may briefly open the destination
            # without delete sharing; keep the prior complete checkpoint.
            time.sleep(0.01 * 2 ** attempt)


def run(args):
    if (
        args.require_host
        and platform.node().split(".")[0].lower() != args.require_host.lower()
    ):
        raise ValueError(
            f"required host {args.require_host!r}; actual host is {platform.node()!r}"
        )
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise ValueError(
            "threadpoolctl is required; install it with python -m pip install threadpoolctl"
        ) from error
    with threadpool_limits(limits=args.threads):
        return run_with_thread_limits(args)


def run_with_thread_limits(args):
    from threadpoolctl import threadpool_limits
    import faiss

    # Faiss can load an additional BLAS/OpenMP runtime. Discover and limit it
    # after import too, and restore the caller's limits when the run exits.
    with threadpool_limits(limits=args.threads):
        return run_native(args, faiss)


def run_native(args, faiss):
    if min(args.k, args.repeats, args.threads, args.bootstrap, *args.batch_sizes) <= 0:
        raise ValueError(
            "k, repeats, threads, bootstrap, and batch sizes must be positive"
        )
    if args.warmups < 0 or not 0 <= args.qb <= 8:
        raise ValueError("warmups must be nonnegative and qb must be in [0, 8]")
    if args.nmse_max_vectors < 0:
        raise ValueError("nmse-max-vectors must be nonnegative")
    for name in ("seeds", "batch_sizes", "modes", "bits", "indexes"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {name} would invalidate pairing")
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        raise ValueError("transform seeds must be uint32 values")
    faiss.omp_set_num_threads(args.threads)
    xt, xb, xq, provenance = load_data(args)
    if args.k > len(xb):
        raise ValueError("k cannot exceed the selected base size")
    d = xb.shape[1]
    started = time.perf_counter()
    exact = faiss.IndexFlatL2(d)
    exact.add(xb)
    _, truth = exact.search(xq, args.k)
    groundtruth_seconds = time.perf_counter() - started
    del exact
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "schema_version": 1,
        "status": "running",
        "metadata": metadata(faiss, args),
        "config": config,
        "data": provenance,
        "metric": "set Recall@k against exact original-space IndexFlatL2",
        "score_semantics": "native approximate squared-L2 scores, including native zero clamp",
        "training_semantics": "native leaf training mean from the same transformed training rows in every mode",
        "dimension_control": "padded-baselines"
        if args.pad_baselines
        else "native-size-baselines",
        "timing_semantics": "full batched native index.search including query transform, Python batching, and output allocation",
        "groundtruth_seconds": groundtruth_seconds,
        "limitations": [
            "fixed training/base dataset; bootstrap does not quantify dataset-to-dataset uncertainty",
            "five transform seeds are a minimum suggestion, not a guarantee of precise uncertainty",
            "timing is CPU/build/thread/batch dependent; QR initialization is separately recorded",
            "reconstruction uses full multibit RaBitQ codes, not its sign-only native sa_decode",
            "NMSE is in transformed coordinates; for padding, this includes error outside the input subspace",
            "vNMSE excludes zero residual rows; counts and pooled NMSE retain their error contributions",
            "no universal VQ or ANN improvement theorem is implied",
        ],
        "rows": [],
    }
    cases = [
        (family, bits, seed, mode)
        for family in args.indexes
        for bits in args.bits
        for seed in args.seeds
        for mode in args.modes
    ]
    # Reproducibly interleave modes to reduce fixed-order timing bias.
    np.random.default_rng(args.bootstrap_seed).shuffle(cases)
    report["case_order"] = [list(case) for case in cases]
    write_results(args.output, report)
    for case_number, (family, bits, seed, mode) in enumerate(cases, 1):
        started = time.perf_counter()
        index, leaf, d_out = make_index(
            faiss, d, family, bits, mode, seed, args.qb, args.pad_baselines
        )
        init_seconds = time.perf_counter() - started
        started = time.perf_counter()
        index.train(xt)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        index.add(xb)
        add_seconds = time.perf_counter() - started
        serialized_size = int(faiss.serialize_index(index).nbytes)
        nmse_count = min(len(xb), args.nmse_max_vectors or len(xb))
        diagnostic_values = apply_transform(faiss, index, xb[:nmse_count])
        reconstructed = reconstruct_leaf(faiss, leaf, 0, nmse_count)
        center = faiss.vector_to_array(leaf.center)
        reconstruction = reconstruction_metrics(
            diagnostic_values, reconstructed, center
        )
        rounds = int(mode[-1]) if mode.startswith("rht") else 0
        label = f"{family}{bits}/{MODE_LABELS[mode]}/qb{args.qb if family == 'RaBitQ' else 'NA'}/{report['dimension_control']}"
        for batch_size in args.batch_sizes:
            for _ in range(args.warmups):
                search_batches(index, xq, args.k, batch_size)
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                predicted = search_batches(index, xq, args.k, batch_size)
                samples.append(time.perf_counter() - started)
            recalls = per_query_recall(truth, predicted)
            median_seconds = float(np.median(samples))
            # This separate diagnostic cannot be subtracted from full search
            # timing: native search may have different allocation/cache effects.
            transform_samples = []
            if isinstance(index, faiss.IndexPreTransform):
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    for position in range(0, len(xq), batch_size):
                        apply_transform(
                            faiss, index, xq[position : position + batch_size]
                        )
                    transform_samples.append(time.perf_counter() - started)
            row = {
                "case_label": label,
                "index": family,
                "factory": f"{family}{bits}",
                "bits": bits,
                "mode": mode,
                "seed": seed,
                "qb": args.qb if family == "RaBitQ" else None,
                "batch_size": batch_size,
                "k": args.k,
                "d_in": d,
                "d_out": d_out,
                "code_size_bytes": int(leaf.code_size),
                "code_storage_bytes": int(leaf.code_size) * len(xb),
                "id_storage_bytes": 0,
                "id_semantics": "implicit sequential IDs in native flat leaf",
                "serialized_size_bytes": serialized_size,
                "init_seconds": init_seconds,
                "train_seconds": train_seconds,
                "add_seconds": add_seconds,
                "prepare_seconds": init_seconds + train_seconds + add_seconds,
                "search_seconds_samples": samples,
                "search_seconds_median": median_seconds,
                "search_seconds_min": min(samples),
                "search_seconds_max": max(samples),
                "search_seconds_std": float(np.std(samples, ddof=1))
                if len(samples) > 1
                else 0.0,
                "qps": len(xq) / median_seconds,
                "query_transform_seconds_samples": transform_samples,
                "query_transform_seconds_median": float(np.median(transform_samples))
                if transform_samples
                else 0.0,
                "recall": float(recalls.mean()),
                "recall_by_query": recalls.tolist(),
                "depth_counts": {str(rounds): len(xb)},
                "depth_fractions": {str(rounds): 1.0},
                "mean_rounds": rounds,
                "query_rounds": rounds,
                "nmse": reconstruction["nmse"],
                "vnmse": reconstruction["vnmse"],
                "reconstruction": reconstruction,
            }
            report["rows"].append(row)
            print(
                f"[{case_number}/{len(cases)}] {label} seed={seed} batch={batch_size} "
                f"Recall@{args.k}={row['recall']:.5f} NMSE={row['nmse']} "
                f"QPS={row['qps']:.1f}",
                flush=True,
            )
        write_results(args.output, report)
        del index, leaf
    report["summaries"] = summarize_rows(
        report["rows"], args.bootstrap, args.bootstrap_seed
    )
    report["status"] = "complete"
    write_results(args.output, report)
    print(f"Saved raw results and paired comparisons to {args.output.resolve()}")
    return report


if __name__ == "__main__":
    parser = make_parser()
    try:
        run(parser.parse_args())
    except ValueError as error:
        parser.error(str(error))
