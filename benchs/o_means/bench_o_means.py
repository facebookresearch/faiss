# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train o-means and e-means on one dataset and print time and error.

    python bench_o_means.py                                  # synthetic, 1M x 96, k = 16384
    python bench_o_means.py --data base.fvecs --k 16384 --nepochs 1 2 3

--data takes a .npy (float32, n x d) or .fvecs file. The last 10,000 rows are
held out to measure the error, with exact float32 assignment. Both methods
start from the same centroids and use the same batch law; e-means converts
the float32 data to BF16 as it does by default. Times include everything the
training call does, after one untimed warm-up call.

The two are compared at equal epochs. The speedups in
contrib/o_means/README.md compare instead the time each needs to reach the
same error, over a range of epochs.
"""

import argparse
import time

import numpy as np
import torch

from faiss.contrib.e_means import Emeans
from faiss.contrib.o_means import Omeans
from faiss.contrib.o_means.schedule import resolve_batch_law


def load(path, n_synthetic, d_synthetic):
    if path is None:
        # A mixture with more modes than clusters, so that longer training
        # keeps lowering the held-out error.
        rng = np.random.default_rng(0)
        modes = rng.standard_normal((65536, d_synthetic), dtype=np.float32)
        x = modes[rng.integers(0, 65536, n_synthetic + 10_000)]
        return x + 0.3 * rng.standard_normal(x.shape, dtype=np.float32)
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    if path.endswith(".fvecs"):
        from faiss.contrib.vecs_io import fvecs_mmap
        return fvecs_mmap(path)
    raise ValueError("--data must be a .npy or .fvecs file")


def error(x, centroids):
    c = centroids.float()
    norms = c.square().sum(1)
    total = 0.0
    for b0 in range(0, len(x), 8192):
        xb = x[b0:b0 + 8192]
        nearest = torch.addmm(norms[None, :], xb, c.t(), alpha=-2.0).argmin(1)
        total += float((xb.double() - c[nearest].double()).square().sum())
    return total / len(x)


def timed(train):
    train()
    torch.cuda.synchronize()
    start = time.perf_counter()
    centroids = train()
    torch.cuda.synchronize()
    return time.perf_counter() - start, centroids


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data")
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--nepochs", type=float, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--synthetic-n", type=int, default=1_000_000)
    parser.add_argument("--synthetic-d", type=int, default=96)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    data = load(args.data, args.synthetic_n, args.synthetic_d)
    x = torch.from_numpy(np.ascontiguousarray(data[:-10_000], dtype=np.float32)).cuda()
    held_out = torch.from_numpy(np.ascontiguousarray(data[-10_000:], dtype=np.float32)).cuda()
    n, d = x.shape
    k = args.k
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    init = x[torch.randperm(n, generator=generator, device="cuda")[:k]].clone()
    law = resolve_batch_law(n, "auto")
    print("n=%d d=%d k=%d, %s batch law, %s" % (n, d, k, law, torch.cuda.get_device_name()))

    for nepochs in args.nepochs:
        def train_e():
            em = Emeans(d, k, nepochs=nepochs, batch_law=law, seed=args.seed)
            return em.train(x, init_centroids=init)[0]

        def train_o():
            om = Omeans(d, k, nepochs=nepochs, batch_law=law, seed=args.seed,
                        capture_assignments=False)
            return om.train(x, init)[0]

        te, ce = timed(train_e)
        to, co = timed(train_o)
        ee, eo = error(held_out, ce), error(held_out, co)
        print("%5.2g epochs   e-means %8.3f s  error %.6g   o-means %8.3f s  error %.6g "
              "(%+.2f%%)   time ratio %.2f"
              % (nepochs, te, ee, to, eo, 100 * (eo / ee - 1), te / to))


if __name__ == "__main__":
    main()
