#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare EDEN, RaBitQ, and TurboQuant at the same bit budget.

The default workload uses faiss.contrib.datasets.SyntheticDataset, matching the
standard synthetic dataset helper used by Faiss benchmarks and tests. Additional
stress distributions are available for comparing behavior under heavier-tailed
or anisotropic data.

The rotation mode is an explicit benchmark axis:

* none: the raw Faiss factory string.
* HR: Faiss HadamardRotation pre-transform, implemented with a fast WHT.
* RR: Faiss dense RandomRotationMatrix pre-transform.

For pre-transformed indexes, reconstruction MSE is computed in the quantizer
coordinate system after the transform. HR and RR preserve squared L2 distances,
so this is the quantization error that the index actually searches over.
"""

import argparse
import time
from dataclasses import dataclass

import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset


DEFAULT_DISTRIBUTIONS = ("synthetic",)
DISTRIBUTION_CHOICES = (
    "synthetic",
    "gaussian",
    "laplace",
    "student_t3",
    "anisotropic",
)
DEFAULT_BITS = (1, 2, 4)
DEFAULT_ROTATIONS = ("none", "HR", "RR")
DEFAULT_INDEXES = ("RaBitQ", "TurboQuant", "EDEN")
TURBOQUANT_BITS = {1, 2, 3, 4, 8}

ROTATION_LABELS = {
    "none": "none",
    "HR": "HR (fast WHT)",
    "RR": "RR (dense)",
}


@dataclass
class MetricResult:
    recall: float
    recall_by_query: np.ndarray
    recall_ci: tuple
    mse: float
    mse_by_vector: np.ndarray
    search_ms: float


@dataclass
class Result:
    distribution: str
    rotation: str
    bits: int
    metrics: dict
    recall_delta_cis: dict
    mse_reduction_cis: dict


def make_dataset(distribution, nt, nb, nq, d, seed):
    if distribution == "synthetic":
        ds = SyntheticDataset(d, nt, nb, nq, metric="L2", seed=seed)
        return (
            ds.get_train().astype("float32"),
            ds.get_database().astype("float32"),
            ds.get_queries().astype("float32"),
            ds,
        )

    rs = np.random.RandomState(seed)

    if distribution == "gaussian":
        xt = rs.randn(nt, d)
        xb = rs.randn(nb, d)
        xq = rs.randn(nq, d)
    elif distribution == "laplace":
        xt = rs.laplace(size=(nt, d))
        xb = rs.laplace(size=(nb, d))
        xq = rs.laplace(size=(nq, d))
    elif distribution == "student_t3":
        xt = rs.standard_t(3, size=(nt, d))
        xb = rs.standard_t(3, size=(nb, d))
        xq = rs.standard_t(3, size=(nq, d))
    elif distribution == "anisotropic":
        scales = np.exp(np.linspace(0.0, 2.0, d))
        xt = rs.randn(nt, d) * scales
        xb = rs.randn(nb, d) * scales
        xq = rs.randn(nq, d) * scales
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    return (
        xt.astype("float32"),
        xb.astype("float32"),
        xq.astype("float32"),
        None,
    )


def factory_string(index_name, bits, rotation):
    if index_name == "RaBitQ":
        base = "RaBitQ" if bits == 1 else f"RaBitQ{bits}"
    elif index_name == "TurboQuant":
        if bits not in TURBOQUANT_BITS:
            raise ValueError(
                f"TurboQuant factory strings exist only for bits "
                f"{sorted(TURBOQUANT_BITS)}"
            )
        base = f"SQtqmse{bits}"
    elif index_name == "EDEN":
        base = "EDEN" if bits == 1 else f"EDEN{bits}"
    else:
        raise ValueError(f"unknown index name: {index_name}")

    return base if rotation == "none" else f"{rotation},{base}"


def compute_groundtruth(xb, xq, k):
    gt = faiss.IndexFlatL2(xb.shape[1])
    gt.add(xb)
    _, gt_i = gt.search(xq, k)
    return gt_i


def dataset_groundtruth(dataset, xb, xq, k):
    if dataset is not None:
        return dataset.get_groundtruth(k)
    return compute_groundtruth(xb, xq, k)


def per_query_recall(gt_i, pred_i):
    recalls = np.empty(gt_i.shape[0], dtype="float32")
    for i, (gt_row, pred_row) in enumerate(zip(gt_i, pred_i)):
        recalls[i] = len(set(gt_row).intersection(pred_row)) / float(
            gt_i.shape[1]
        )
    return recalls


def leaf_index(index):
    if isinstance(index, faiss.IndexPreTransform):
        return faiss.downcast_index(index.index)
    return index


def set_query_bits(index, qb):
    leaf = leaf_index(index)
    if hasattr(leaf, "qb"):
        leaf.qb = qb


def apply_pretransform(index, x):
    transformed = x
    for i in range(index.chain.size()):
        vt = faiss.downcast_VectorTransform(index.chain.at(i))
        transformed = vt.apply_py(transformed)
    return transformed


def rabitq_reconstruction_errors(index, xb):
    nb_bits = int(index.rabitq.nb_bits)
    if nb_bits == 1:
        reconstructed = np.empty_like(xb)
        index.reconstruct_n(0, xb.shape[0], reconstructed)
        return np.sum((xb - reconstructed) ** 2, axis=1)

    if index.metric_type != faiss.METRIC_L2:
        raise ValueError("RaBitQ reconstruction MSE is implemented for L2 only")

    # IndexRaBitQ::sa_decode intentionally decodes only the sign-bit baseline.
    # For multi-bit quantization MSE, reconstruct the vector implied by the
    # refinement code used by the full multi-bit distance formula.
    d = xb.shape[1]
    ex_bits = nb_bits - 1
    binary_size = (d + 7) // 8
    sign_factors_size = 12
    ex_code_size = (d * ex_bits + 7) // 8
    ex_code_offset = binary_size + sign_factors_size
    ex_factors_offset = ex_code_offset + ex_code_size

    codes = faiss.vector_to_array(index.codes).reshape(
        xb.shape[0], int(index.code_size)
    )
    center = faiss.vector_to_array(index.center).astype("float32")
    if center.size == 0:
        center = np.zeros(d, dtype="float32")

    cb = -(float(1 << ex_bits) - 0.5)
    signs = np.unpackbits(
        codes[:, :binary_size], axis=1, bitorder="little"
    )[:, :d].astype("float32")
    ex_bitplanes = np.unpackbits(
        codes[:, ex_code_offset:ex_factors_offset],
        axis=1,
        bitorder="little",
    )[:, : d * ex_bits].reshape(xb.shape[0], d, ex_bits)
    ex_codes = np.zeros((xb.shape[0], d), dtype="float32")
    for bit in range(ex_bits):
        ex_codes += ex_bitplanes[:, :, bit].astype("float32") * float(1 << bit)

    ex_factors = (
        codes[:, ex_factors_offset : ex_factors_offset + 8]
        .copy()
        .view("float32")
        .reshape(xb.shape[0], 2)
    )
    scales = (-0.5 * ex_factors[:, 1]).astype("float32")

    code_values = signs * float(1 << ex_bits) + ex_codes + cb
    reconstructed = center + scales[:, None] * code_values

    return np.sum((xb - reconstructed) ** 2, axis=1)


def reconstruction_errors(index, xb):
    if isinstance(index, faiss.IndexPreTransform):
        comparable_xb = apply_pretransform(index, xb)
        comparable_index = leaf_index(index)
    else:
        comparable_xb = xb
        comparable_index = index

    if hasattr(comparable_index, "rabitq"):
        return rabitq_reconstruction_errors(comparable_index, comparable_xb)

    reconstructed = np.empty_like(comparable_xb)
    comparable_index.reconstruct_n(0, comparable_xb.shape[0], reconstructed)
    return np.sum((comparable_xb - reconstructed) ** 2, axis=1)


def bootstrap_mean_ci(values, num_samples, ci_level, rng):
    if num_samples <= 0:
        return (np.nan, np.nan)

    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return (np.nan, np.nan)

    bootstrap_means = np.empty(num_samples, dtype="float64")
    # Keep memory bounded when users increase nb/nq to tighten intervals.
    batch_size = max(1, min(100, num_samples))
    for start in range(0, num_samples, batch_size):
        end = min(num_samples, start + batch_size)
        sample_ids = rng.randint(
            0, values.size, size=(end - start, values.size)
        )
        bootstrap_means[start:end] = values[sample_ids].mean(axis=1)

    alpha = (1.0 - ci_level) * 0.5
    lo, hi = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def evaluate(factory, xt, xb, xq, gt_i, k, qb, ci_samples, ci_level, ci_rng):
    index = faiss.index_factory(xb.shape[1], factory)
    set_query_bits(index, qb)

    index.train(xt)
    index.add(xb)

    _, pred_i = index.search(xq[:1], k)
    del pred_i
    t2 = time.perf_counter()
    _, pred_i = index.search(xq, k)
    t3 = time.perf_counter()
    recall_by_query = per_query_recall(gt_i, pred_i)
    reconstruction_error = reconstruction_errors(index, xb)

    return MetricResult(
        recall=float(np.mean(recall_by_query)),
        recall_by_query=recall_by_query,
        recall_ci=bootstrap_mean_ci(
            recall_by_query, ci_samples, ci_level, ci_rng
        ),
        mse=float(np.mean(reconstruction_error)),
        mse_by_vector=reconstruction_error,
        search_ms=(t3 - t2) * 1000.0,
    )


def format_ci(ci):
    lo, hi = ci
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"[{lo:+.4f}, {hi:+.4f}]"


def format_float(value, digits=4, sign=False):
    if value is None or not np.isfinite(value):
        return "n/a"
    sign_flag = "+" if sign else ""
    return f"{value:{sign_flag}.{digits}f}"


def result_delta(result, index_name, baseline_name, field):
    metrics = result.metrics
    if index_name not in metrics or baseline_name not in metrics:
        return None
    return getattr(metrics[index_name], field) - getattr(
        metrics[baseline_name], field
    )


def pair_key(index_name, baseline_name):
    return f"{index_name}_vs_{baseline_name}"


def print_markdown_tables(
    results,
    indexes,
    include_speed,
    include_ci,
    ci_label,
    include_recall_delta_ci,
):
    grouped = {}
    for result in results:
        grouped.setdefault((result.distribution, result.rotation), []).append(
            result
        )

    baselines = [name for name in indexes if name != "EDEN"]

    for (distribution, rotation), rows in grouped.items():
        print(f"\n### {distribution} / rotation: {ROTATION_LABELS[rotation]}\n")

        header = "| bits / dim |"
        align = "|---:|"
        for index_name in indexes:
            header += f" {index_name} recall |"
            align += "---:|"
            if include_ci:
                header += f" {index_name} recall {ci_label} |"
                align += "---:|"
        for baseline in baselines:
            header += f" EDEN - {baseline} recall |"
            align += "---:|"
            if include_ci and include_recall_delta_ci:
                header += f" EDEN - {baseline} recall {ci_label} |"
                align += "---:|"
        for index_name in indexes:
            header += f" {index_name} MSE |"
            align += "---:|"
        for baseline in baselines:
            header += f" MSE reduction vs {baseline} |"
            align += "---:|"
            if include_ci:
                header += f" MSE reduction vs {baseline} {ci_label} |"
                align += "---:|"
        if include_speed:
            for index_name in indexes:
                header += f" {index_name} search ms |"
                align += "---:|"

        print(header)
        print(align)

        for row in rows:
            values = f"| {row.bits} |"
            for index_name in indexes:
                metric = row.metrics.get(index_name)
                values += f" {format_float(metric.recall if metric else None)} |"
                if include_ci:
                    values += f" {format_ci(metric.recall_ci) if metric else ''} |"
            for baseline in baselines:
                recall_delta = result_delta(row, "EDEN", baseline, "recall")
                values += f" {format_float(recall_delta, sign=True)} |"
                if include_ci and include_recall_delta_ci:
                    key = pair_key("EDEN", baseline)
                    ci = row.recall_delta_cis.get(key, (np.nan, np.nan))
                    values += f" {format_ci(ci)} |"
            for index_name in indexes:
                metric = row.metrics.get(index_name)
                values += f" {format_float(metric.mse if metric else None)} |"
            for baseline in baselines:
                mse_reduction = None
                if "EDEN" in row.metrics and baseline in row.metrics:
                    mse_reduction = row.metrics[baseline].mse - row.metrics[
                        "EDEN"
                    ].mse
                values += f" {format_float(mse_reduction)} |"
                if include_ci:
                    key = pair_key("EDEN", baseline)
                    ci = row.mse_reduction_cis.get(key, (np.nan, np.nan))
                    values += f" {format_ci(ci)} |"
            if include_speed:
                for index_name in indexes:
                    metric = row.metrics.get(index_name)
                    search_ms = metric.search_ms if metric else None
                    values += f" {format_float(search_ms, digits=1)} |"
            print(values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--nt", type=int, default=4000)
    parser.add_argument("--nb", type=int, default=4000)
    parser.add_argument("--nq", type=int, default=100)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--qb", type=int, default=0)
    parser.add_argument("--ci-samples", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--include-recall-delta-ci",
        "--include-recall-ci",
        dest="include_recall_delta_ci",
        action="store_true",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=DEFAULT_BITS,
        choices=range(1, 9),
    )
    parser.add_argument(
        "--rotations",
        nargs="+",
        default=DEFAULT_ROTATIONS,
        choices=DEFAULT_ROTATIONS,
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=DEFAULT_DISTRIBUTIONS,
        choices=DISTRIBUTION_CHOICES,
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=DEFAULT_INDEXES,
        choices=DEFAULT_INDEXES,
    )
    parser.add_argument("--include-speed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    results = []

    print(
        "EDEN, RaBitQ, and TurboQuant L2 benchmark: "
        f"d={args.d}, nt={args.nt}, nb={args.nb}, nq={args.nq}, "
        f"k={args.k}, qb={args.qb}, rotations={','.join(args.rotations)}, "
        f"indexes={','.join(args.indexes)}"
    )
    print(
        "TurboQuant uses current Faiss factory strings SQtqmse{bits}; "
        "SyntheticDataset is the default workload."
    )
    print(
        "MSE is measured after any pre-transform, in the quantizer coordinate "
        "system."
    )
    if args.ci_samples > 0:
        print(
            "CIs use bootstrap resampling over queries for recall means, "
            "paired bootstrap resampling over queries for recall deltas when "
            "--include-recall-delta-ci is set, and over database vectors "
            "for MSE reductions: "
            f"samples={args.ci_samples}, level={args.ci_level:.2f}."
        )

    ci_rng = np.random.RandomState(args.seed + 1000003)

    for dist_id, distribution in enumerate(args.distributions):
        xt, xb, xq, dataset = make_dataset(
            distribution,
            args.nt,
            args.nb,
            args.nq,
            args.d,
            args.seed + dist_id,
        )
        gt_i = dataset_groundtruth(dataset, xb, xq, args.k)

        for rotation in args.rotations:
            for bits in args.bits:
                metrics = {}
                for index_name in args.indexes:
                    try:
                        factory = factory_string(index_name, bits, rotation)
                    except ValueError:
                        continue
                    metrics[index_name] = evaluate(
                        factory,
                        xt,
                        xb,
                        xq,
                        gt_i,
                        args.k,
                        args.qb,
                        args.ci_samples,
                        args.ci_level,
                        ci_rng,
                    )

                recall_delta_cis = {}
                mse_reduction_cis = {}
                if "EDEN" in metrics:
                    for baseline in args.indexes:
                        if baseline == "EDEN" or baseline not in metrics:
                            continue
                        key = pair_key("EDEN", baseline)
                        recall_delta = (
                            metrics["EDEN"].recall_by_query
                            - metrics[baseline].recall_by_query
                        )
                        mse_reduction = (
                            metrics[baseline].mse_by_vector
                            - metrics["EDEN"].mse_by_vector
                        )
                        recall_delta_cis[key] = bootstrap_mean_ci(
                            recall_delta,
                            args.ci_samples,
                            args.ci_level,
                            ci_rng,
                        )
                        mse_reduction_cis[key] = bootstrap_mean_ci(
                            mse_reduction,
                            args.ci_samples,
                            args.ci_level,
                            ci_rng,
                        )

                results.append(
                    Result(
                        distribution=distribution,
                        rotation=rotation,
                        bits=bits,
                        metrics=metrics,
                        recall_delta_cis=recall_delta_cis,
                        mse_reduction_cis=mse_reduction_cis,
                    )
                )

    ci_label = f"{int(args.ci_level * 100)}% CI"
    print_markdown_tables(
        results,
        args.indexes,
        args.include_speed,
        args.ci_samples > 0,
        ci_label,
        args.include_recall_delta_ci,
    )


if __name__ == "__main__":
    main()
