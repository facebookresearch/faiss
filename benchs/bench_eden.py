#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark L2 quantizer indexes at matched bit budgets.

CSV rows are written to stdout. Run metadata is written to stderr.
"""

import argparse
import csv
import sys
import time

import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset


def display_name(index_name):
    return {
        "EDEN": "EDEN-Unbiased",
        "EDENBiased": "EDEN-Biased",
    }.get(index_name, index_name)


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
        if bits not in {1, 2, 3, 4, 8}:
            return None
        base = f"SQtqmse{bits}"
    elif index_name == "EDEN":
        base = "EDEN" if bits == 1 else f"EDEN{bits}"
    elif index_name == "EDENBiased":
        base = "EDENBIASED" if bits == 1 else f"EDEN{bits}BIASED"
    else:
        raise ValueError(f"unknown index name: {index_name}")

    return base if rotation == "none" else f"{rotation},{base}"


def comparison_target(indexes):
    if "EDENBiased" in indexes:
        return "EDENBiased"
    if "EDEN" in indexes:
        return "EDEN"
    return None


def dataset_groundtruth(dataset, xb, xq, k):
    if dataset is not None:
        return dataset.get_groundtruth(k)
    gt = faiss.IndexFlatL2(xb.shape[1])
    gt.add(xb)
    return gt.search(xq, k)[1]


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


def reconstruction_errors(index, xb):
    if isinstance(index, faiss.IndexPreTransform):
        xb = apply_pretransform(index, xb)
        index = leaf_index(index)

    if hasattr(index, "rabitq") and int(index.rabitq.nb_bits) > 1:
        if index.metric_type != faiss.METRIC_L2:
            raise ValueError("RaBitQ reconstruction MSE is implemented for L2 only")

        # IndexRaBitQ::sa_decode decodes only the sign-bit baseline. For
        # multi-bit MSE, reconstruct the vector implied by the distance formula.
        d = xb.shape[1]
        ex_bits = int(index.rabitq.nb_bits) - 1
        binary_size = (d + 7) // 8
        ex_code_offset = binary_size + 12
        ex_factors_offset = ex_code_offset + (d * ex_bits + 7) // 8
        codes = faiss.vector_to_array(index.codes).reshape(
            xb.shape[0], int(index.code_size)
        )
        center = faiss.vector_to_array(index.center).astype("float32")
        if center.size == 0:
            center = np.zeros(d, dtype="float32")

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
            ex_codes += ex_bitplanes[:, :, bit] * float(1 << bit)

        ex_factors = (
            codes[:, ex_factors_offset : ex_factors_offset + 8]
            .copy()
            .view("float32")
            .reshape(xb.shape[0], 2)
        )
        code_values = signs * float(1 << ex_bits) + ex_codes - (
            float(1 << ex_bits) - 0.5
        )
        reconstructed = center + (-0.5 * ex_factors[:, 1])[:, None] * code_values
        return np.sum((xb - reconstructed) ** 2, axis=1)

    reconstructed = np.empty_like(xb)
    index.reconstruct_n(0, xb.shape[0], reconstructed)
    return np.sum((xb - reconstructed) ** 2, axis=1)


def bootstrap_mean_ci(values, num_samples, ci_level, rng):
    if num_samples <= 0:
        return (np.nan, np.nan)

    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return (np.nan, np.nan)

    means = np.empty(num_samples, dtype="float64")
    for start in range(0, num_samples, 100):
        end = min(num_samples, start + 100)
        sample_ids = rng.randint(0, values.size, size=(end - start, values.size))
        means[start:end] = values[sample_ids].mean(axis=1)

    alpha = (1.0 - ci_level) * 0.5
    return tuple(float(x) for x in np.quantile(means, [alpha, 1.0 - alpha]))


def evaluate(factory, xt, xb, xq, gt_i, k, qb, ci_samples, ci_level, ci_rng):
    index = faiss.index_factory(xb.shape[1], factory)
    set_query_bits(index, qb)

    index.train(xt)
    index.add(xb)

    index.search(xq[:1], k)
    t0 = time.perf_counter()
    _, pred_i = index.search(xq, k)
    search_ms = (time.perf_counter() - t0) * 1000.0

    recall_values = per_query_recall(gt_i, pred_i)
    mse_values = reconstruction_errors(index, xb)
    return {
        "recall": float(recall_values.mean()),
        "recall_values": recall_values,
        "recall_ci": bootstrap_mean_ci(
            recall_values, ci_samples, ci_level, ci_rng
        ),
        "mse": float(mse_values.mean()),
        "mse_values": mse_values,
        "search_ms": search_ms,
    }


def format_ci(ci):
    lo, hi = ci
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"[{lo:+.4f}, {hi:+.4f}]"


def format_float(value, digits=4, sign=False):
    if value is None or not np.isfinite(value):
        return "n/a"
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    return f"{value:{'+' if sign else ''}.{digits}f}"


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
        "--bits", type=int, nargs="+", default=(1, 2, 4), choices=range(1, 9)
    )
    parser.add_argument(
        "--rotations",
        nargs="+",
        default=("none", "HR", "RR"),
        choices=("none", "HR", "RR"),
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=("synthetic",),
        choices=("synthetic", "gaussian", "laplace", "student_t3", "anisotropic"),
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=("EDEN", "RaBitQ", "TurboQuant"),
        choices=("EDEN", "EDENBiased", "RaBitQ", "TurboQuant"),
    )
    parser.add_argument("--include-speed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        "L2 quantizer benchmark: "
        f"d={args.d}, nt={args.nt}, nb={args.nb}, nq={args.nq}, "
        f"k={args.k}, qb={args.qb}, rotations={','.join(args.rotations)}, "
        f"indexes={','.join(display_name(index) for index in args.indexes)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "CSV output uses SQtqmse{bits} for TurboQuant. MSE is measured after "
        "any pre-transform, in the quantizer coordinate system.",
        file=sys.stderr,
        flush=True,
    )

    ci_rng = np.random.RandomState(args.seed + 1000003)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    target = comparison_target(args.indexes)
    baselines = [name for name in args.indexes if name != target]
    header = ["distribution", "rotation", "bits / dim"]
    for index_name in args.indexes:
        header.append(f"{display_name(index_name)} recall")
        if args.ci_samples > 0:
            header.append(f"{display_name(index_name)} recall CI")
    if target is not None:
        for baseline in baselines:
            header.append(f"{display_name(target)} - {display_name(baseline)} recall")
            if args.ci_samples > 0 and args.include_recall_delta_ci:
                header.append(
                    f"{display_name(target)} - {display_name(baseline)} recall CI"
                )
    for index_name in args.indexes:
        header.append(f"{display_name(index_name)} MSE")
    if target is not None:
        for baseline in baselines:
            header.append(f"MSE reduction vs {display_name(baseline)}")
            if args.ci_samples > 0:
                header.append(f"MSE reduction vs {display_name(baseline)} CI")
    if args.include_speed:
        for index_name in args.indexes:
            header.append(f"{display_name(index_name)} search ms")
    writer.writerow(header)

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
                    factory = factory_string(index_name, bits, rotation)
                    if factory is not None:
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

                row = [distribution, rotation, bits]
                for index_name in args.indexes:
                    metric = metrics.get(index_name)
                    row.append(format_float(metric["recall"] if metric else None))
                    if args.ci_samples > 0:
                        row.append(format_ci(metric["recall_ci"]) if metric else "")
                if target is not None:
                    for baseline in baselines:
                        delta = None
                        ci = (np.nan, np.nan)
                        if target in metrics and baseline in metrics:
                            delta_values = (
                                metrics[target]["recall_values"]
                                - metrics[baseline]["recall_values"]
                            )
                            delta = float(delta_values.mean())
                            ci = bootstrap_mean_ci(
                                delta_values,
                                args.ci_samples,
                                args.ci_level,
                                ci_rng,
                            )
                        row.append(format_float(delta, sign=True))
                        if args.ci_samples > 0 and args.include_recall_delta_ci:
                            row.append(format_ci(ci))

                for index_name in args.indexes:
                    metric = metrics.get(index_name)
                    row.append(format_float(metric["mse"] if metric else None))
                if target is not None:
                    for baseline in baselines:
                        reduction = None
                        ci = (np.nan, np.nan)
                        if target in metrics and baseline in metrics:
                            reduction_values = (
                                metrics[baseline]["mse_values"]
                                - metrics[target]["mse_values"]
                            )
                            reduction = float(reduction_values.mean())
                            ci = bootstrap_mean_ci(
                                reduction_values,
                                args.ci_samples,
                                args.ci_level,
                                ci_rng,
                            )
                        row.append(format_float(reduction))
                        if args.ci_samples > 0:
                            row.append(format_ci(ci))

                if args.include_speed:
                    for index_name in args.indexes:
                        metric = metrics.get(index_name)
                        value = (
                            format_float(metric["search_ms"], digits=1)
                            if metric
                            else ""
                        )
                        row.append(value)

                writer.writerow(row)


if __name__ == "__main__":
    main()
