# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Common functions to load datasets and compute their ground-truth
"""

import time
import numpy as np
import faiss

from faiss.contrib import datasets as faiss_datasets

print("path:", faiss_datasets.__file__)

faiss_datasets.dataset_basedir = '/checkpoint/matthijs/simsearch/'

def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')


#################################################################
# Dataset
#################################################################

class DatasetCentroids(faiss_datasets.Dataset):

    def __init__(self, ds, indexfile):
        self.d = ds.d
        self.metric = ds.metric
        self.nq = ds.nq
        self.xq = ds.get_queries()

        # get the xb set
        src_index = faiss.read_index(indexfile)
        src_quant = faiss.downcast_index(src_index.quantizer)
        centroids = faiss.vector_to_array(src_quant.xb)
        self.xb = centroids.reshape(-1, self.d)
        self.nb = self.nt = len(self.xb)

    def get_queries(self):
        return self.xq

    def get_database(self):
        return self.xb

    def get_train(self, maxtrain=None):
        return self.xb

    def get_groundtruth(self, k=100):
        return faiss.knn(
            self.xq, self.xb, k,
            faiss.METRIC_L2 if self.metric == 'L2' else faiss.METRIC_INNER_PRODUCT
        )[1]






def load_dataset(dataset='deep1M', compute_gt=False, download=False):

    print("load data", dataset)

    if dataset == 'sift1M':
        return faiss_datasets.DatasetSIFT1M()

    elif dataset.startswith('bigann'):

        dbsize = 1000 if dataset == "bigann1B" else int(dataset[6:-1])

        return faiss_datasets.DatasetBigANN(nb_M=dbsize)

    elif dataset.startswith("deep_centroids_"):
        ncent = int(dataset[len("deep_centroids_"):])
        centdir = "/checkpoint/matthijs/bench_all_ivf/precomputed_clusters"
        return DatasetCentroids(
            faiss_datasets.DatasetDeep1B(nb=1000000),
            f"{centdir}/clustering.dbdeep1M.IVF{ncent}.faissindex"
        )

    elif dataset.startswith("deep"):

        szsuf = dataset[4:]
        if szsuf[-1] == 'M':
            dbsize = 10 ** 6 * int(szsuf[:-1])
        elif szsuf == '1B':
            dbsize = 10 ** 9
        elif szsuf[-1] == 'k':
            dbsize = 1000 * int(szsuf[:-1])
        else:
            assert False, "did not recognize suffix " + szsuf
        return faiss_datasets.DatasetDeep1B(nb=dbsize)

    elif dataset == "music-100":
        return faiss_datasets.DatasetMusic100()

    elif dataset == "glove":
        return faiss_datasets.DatasetGlove(download=download)

    else:
        assert False


#################################################################
# Evaluation
#################################################################


def evaluate_DI(D, I, gt):
    nq = gt.shape[0]
    k = I.shape[1]
    rank = 1
    while rank <= k:
        recall = (I[:, :rank] == gt[:, :1]).sum() / float(nq)
        print("R@%d: %.4f" % (rank, recall), end=' ')
        rank *= 10


def evaluate(xq, gt, index, k=100, endl=True):
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    nq = xq.shape[0]
    print("\t %8.4f ms per query, " % (
        (t1 - t0) * 1000.0 / nq), end=' ')
    rank = 1
    while rank <= k:
        recall = (I[:, :rank] == gt[:, :1]).sum() / float(nq)
        print("R@%d: %.4f" % (rank, recall), end=' ')
        rank *= 10
    if endl:
        print()
    return D, I
