# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run the DEEP1M e-means quality experiment."""

import argparse
import os

import numpy as np
import torch

from faiss.contrib import datasets
from faiss.contrib.e_means import Emeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        help="directory containing deep1b/base.fvecs and its query file",
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    if args.data_dir:
        datasets.set_dataset_basedir(os.path.join(args.data_dir, ""))

    dataset = datasets.DatasetDeep1B(nb=1_000_000)
    train = torch.from_numpy(dataset.get_database()).to(args.device)
    queries = dataset.get_queries()
    # Evaluate on a deterministic half of the query set.
    order = np.random.RandomState(1234).permutation(len(queries))
    test = torch.from_numpy(queries[order[len(queries) // 2 :]]).to(args.device)

    mse = []
    for seed in range(10):
        kmeans = Emeans(
            dataset.d,
            16_384,
            niter=3,
            batch_law="throughput",
            seed=seed,
            verbose=True,
        )
        kmeans.train(train)
        distance, _ = kmeans.assign(test)
        mse.append(distance.mean().item())
        print("seed %d test MSE %.6f" % (seed, mse[-1]))

    print("median test MSE %.6f" % np.median(mse))


if __name__ == "__main__":
    main()
