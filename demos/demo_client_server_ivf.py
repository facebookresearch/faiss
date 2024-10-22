#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import faiss

from faiss.contrib.client_server import run_index_server, ClientIndex


#################################################################
# Small I/O functions
#################################################################


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#################################################################
#  Main program
#################################################################

stage = int(sys.argv[1])

tmpdir = '/tmp/'

if stage == 0:
    # train the index
    xt = fvecs_read("sift1M/sift_learn.fvecs")
    index = faiss.index_factory(xt.shape[1], "IVF4096,Flat")
    print("training index")
    index.train(xt)
    print("write " + tmpdir + "trained.index")
    faiss.write_index(index, tmpdir + "trained.index")


if 1 <= stage <= 4:
    # add 1/4 of the database to 4 independent indexes
    bno = stage - 1
    xb = fvecs_read("sift1M/sift_base.fvecs")
    i0, i1 = int(bno * xb.shape[0] / 4), int((bno + 1) * xb.shape[0] / 4)
    index = faiss.read_index(tmpdir + "trained.index")
    print("adding vectors %d:%d" % (i0, i1))
    index.add_with_ids(xb[i0:i1], np.arange(i0, i1))
    print("write " + tmpdir + "block_%d.index" % bno)
    faiss.write_index(index, tmpdir + "block_%d.index" % bno)


machine_ports = [
    ('localhost', 12010),
    ('localhost', 12011),
    ('localhost', 12012),
    ('localhost', 12013),
]
v6 = False

if 5 <= stage <= 8:
    # load an index slice and launch index
    bno = stage - 5

    fname = tmpdir + "block_%d.index" % bno
    print("read " + fname)
    index = faiss.read_index(fname)

    port = machine_ports[bno][1]
    run_index_server(index, port, v6=v6)


if stage == 9:
    client_index = ClientIndex(machine_ports)
    print('index size:', client_index.ntotal)
    client_index.set_nprobe(16)

    # load query vectors and ground-truth
    xq = fvecs_read("sift1M/sift_query.fvecs")
    gt = ivecs_read("sift1M/sift_groundtruth.ivecs")

    D, I = client_index.search(xq, 5)

    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(xq.shape[0])
    print("recall@1: %.3f" % recall_at_1)
