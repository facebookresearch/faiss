# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time

import faiss

import numpy as np

from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.ivf_tools import big_batch_search

parser = argparse.ArgumentParser()


def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)


group = parser.add_argument_group('dataset options')
aa('--dim', type=int, default=64)
aa('--size', default="S")

group = parser.add_argument_group('index options')
aa('--nlist', type=int, default=100)
aa('--factory_string', default="", help="overrides nlist")
aa('--k', type=int, default=10)
aa('--nprobe', type=int, default=5)
aa('--nt', type=int, default=-1, help="nb search threads")
aa('--method', default="pairwise_distances", help="")

args = parser.parse_args()
print("args:", args)

if args.size == "S":
    ds = SyntheticDataset(32, 2000, 4000, 1000)
elif args.size == "M":
    ds = SyntheticDataset(32, 20000, 40000, 10000)
elif args.size == "L":
    ds = SyntheticDataset(32, 200000, 400000, 100000)
else:
    raise RuntimeError(f"dataset size {args.size} not supported")

nlist = args.nlist
nprobe = args.nprobe
k = args.k


def tic(name):
    global tictoc
    tictoc = (name, time.time())
    print(name, end="\r", flush=True)


def toc():
    global tictoc
    name, t0 = tictoc
    dt = time.time() - t0
    print(f"{name}: {dt:.3f} s")
    return dt


print(f"dataset {ds}, {nlist=:} {nprobe=:} {k=:}")

if args.factory_string == "":
    factory_string = f"IVF{nlist},Flat"
else:
    factory_string = args.factory_string

print(f"instanciate {factory_string}")
index = faiss.index_factory(ds.d, factory_string)

if args.factory_string != "":
    nlist = index.nlist

print("nlist", nlist)

tic("train")
index.train(ds.get_train())
toc()

tic("add")
index.add(ds.get_database())
toc()

if args.nt != -1:
    print("setting nb of threads to", args.nt)
    faiss.omp_set_num_threads(args.nt)

tic("reference search")
index.nprobe
index.nprobe = nprobe
Dref, Iref = index.search(ds.get_queries(), k)
t_ref = toc()

tic("block search")
Dnew, Inew = big_batch_search(
    index, ds.get_queries(),
    k, method=args.method, verbose=10
)
t_tot = toc()

assert (Inew != Iref).sum() / Iref.size < 1e-4
np.testing.assert_almost_equal(Dnew, Dref, decimal=4)

print(f"total block search time {t_tot:.3f} s, speedup {t_ref / t_tot:.3f}x")
