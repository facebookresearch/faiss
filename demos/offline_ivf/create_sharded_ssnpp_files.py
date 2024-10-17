# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
import os


def xbin_mmap(fname, dtype, maxn=-1):
    """
    Code from
    https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/dataset_io.py#L94
    mmap the competition file format for a given type of items
    """
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * np.dtype(dtype).itemsize
    if maxn > 0:
        n = min(n, maxn)
    return np.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(n, d))


def main(args: argparse.Namespace):
    ssnpp_data = xbin_mmap(fname=args.filepath, dtype="uint8")
    num_batches = ssnpp_data.shape[0] // args.data_batch
    assert (
        ssnpp_data.shape[0] % args.data_batch == 0
    ), "num of embeddings per file should divide total num of embeddings"
    for i in range(num_batches):
        xb_batch = ssnpp_data[
            i * args.data_batch:(i + 1) * args.data_batch, :
        ]
        filename = args.output_dir + f"/ssnpp_{(i):010}.npy"
        np.save(filename, xb_batch)
        print(f"File {filename} is saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_batch",
        dest="data_batch",
        type=int,
        default=50000000,
        help="Number of embeddings per file, should be a divisor of 1B",
    )
    parser.add_argument(
        "--filepath",
        dest="filepath",
        type=str,
        default="/datasets01/big-ann-challenge-data/FB_ssnpp/FB_ssnpp_database.u8bin",
        help="path of 1B ssnpp database vectors' original file",
    )
    parser.add_argument(
        "--filepath",
        dest="output_dir",
        type=str,
        default="/checkpoint/marialomeli/ssnpp_data",
        help="path to put sharded files",
    )

    args = parser.parse_args()
    main(args)
