# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import faiss
import getpass


from .vecs_io import fvecs_read, ivecs_read, bvecs_mmap, fvecs_mmap
from .exhaustive_search import knn

class Dataset:
    """ Generic abstract class for a test dataset """

    def __init__(self):
        """ the constructor should set the following fields: """
        self.d = -1
        self.metric = 'L2'   # or IP
        self.nq = -1
        self.nb = -1
        self.nt = -1

    def get_queries(self):
        """ return the queries as a (nq, d) array """
        raise NotImplementedError()

    def get_train(self, maxtrain=None):
        """ return the queries as a (nt, d) array """
        raise NotImplementedError()

    def get_database(self):
        """ return the queries as a (nb, d) array """
        raise NotImplementedError()

    def database_iterator(self, bs=128, split=(1, 0)):
        """returns an iterator on database vectors.
        bs is the number of vectors per batch
        split = (nsplit, rank) means the dataset is split in nsplit
        shards and we want shard number rank
        The default implementation just iterates over the full matrix
        returned by get_dataset.
        """
        xb = self.get_database()
        nsplit, rank = split
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        for j0 in range(i0, i1, bs):
            yield xb[j0: min(j0 + bs, i1)]

    def get_groundtruth(self, k=None):
        """ return the ground truth for k-nearest neighbor search """
        raise NotImplementedError()

    def get_groundtruth_range(self, thresh=None):
        """ return the ground truth for range search """
        raise NotImplementedError()

    def __str__(self):
        return (f"dataset in dimension {self.d}, with metric {self.metric}, "
                f"size: Q {self.nq} B {self.nb} T {self.nt}")

    def check_sizes(self):
        """ runs the previous and checks the sizes of the matrices """
        assert self.get_queries().shape == (self.nq, self.d)
        if self.nt > 0:
            xt = self.get_train(maxtrain=123)
            assert xt.shape == (123, self.d), "shape=%s" % (xt.shape, )
        assert self.get_database().shape == (self.nb, self.d)
        assert self.get_groundtruth(k=13).shape == (self.nq, 13)


class SyntheticDataset(Dataset):
    """A dataset that is not completely random but still challenging to
    index
    """

    def __init__(self, d, nt, nb, nq, metric='L2', seed=1338):
        Dataset.__init__(self)
        self.d, self.nt, self.nb, self.nq = d, nt, nb, nq
        d1 = 10     # intrinsic dimension (more or less)
        n = nb + nt + nq
        rs = np.random.RandomState(seed)
        x = rs.normal(size=(n, d1))
        x = np.dot(x, rs.rand(d1, d))
        # now we have a d1-dim ellipsoid in d-dimensional space
        # higher factor (>4) -> higher frequency -> less linear
        x = x * (rs.rand(d) * 4 + 0.1)
        x = np.sin(x)
        x = x.astype('float32')
        self.metric = metric
        self.xt = x[:nt]
        self.xb = x[nt:nt + nb]
        self.xq = x[nt + nb:]

    def get_queries(self):
        return self.xq

    def get_train(self, maxtrain=None):
        maxtrain = maxtrain if maxtrain is not None else self.nt
        return self.xt[:maxtrain]

    def get_database(self):
        return self.xb

    def get_groundtruth(self, k=100):
        return knn(
            self.xq, self.xb, k,
            faiss.METRIC_L2 if self.metric == 'L2' else faiss.METRIC_INNER_PRODUCT
        )[1]


############################################################################
# The following datasets are a few standard open-source datasets
# they should be stored in a directory, and we start by guessing where
# that directory is
############################################################################

username = getpass.getuser()

for dataset_basedir in (
        '/datasets01/simsearch/041218/',
        '/mnt/vol/gfsai-flash3-east/ai-group/datasets/simsearch/',
        f'/home/{username}/simsearch/data/'):
    if os.path.exists(dataset_basedir):
        break
else:
    # users can link their data directory to `./data`
    dataset_basedir = 'data/'


def set_dataset_basedir(path):
    global dataset_basedir
    dataset_basedir = path


class DatasetSIFT1M(Dataset):
    """
    The original dataset is available at: http://corpus-texmex.irisa.fr/
    (ANN_SIFT1M)
    """

    def __init__(self):
        Dataset.__init__(self)
        self.d, self.nt, self.nb, self.nq = 128, 100000, 1000000, 10000
        self.basedir = dataset_basedir + 'sift1M/'

    def get_queries(self):
        return fvecs_read(self.basedir + "sift_query.fvecs")

    def get_train(self, maxtrain=None):
        maxtrain = maxtrain if maxtrain is not None else self.nt
        return fvecs_read(self.basedir + "sift_learn.fvecs")[:maxtrain]

    def get_database(self):
        return fvecs_read(self.basedir + "sift_base.fvecs")

    def get_groundtruth(self, k=None):
        gt = ivecs_read(self.basedir + "sift_groundtruth.ivecs")
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt


def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')


class DatasetBigANN(Dataset):
    """
    The original dataset is available at: http://corpus-texmex.irisa.fr/
    (ANN_SIFT1B)
    """

    def __init__(self, nb_M=1000):
        Dataset.__init__(self)
        assert nb_M in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
        self.nb_M = nb_M
        nb = nb_M * 10**6
        self.d, self.nt, self.nb, self.nq = 128, 10**8, nb, 10000
        self.basedir = dataset_basedir + 'bigann/'

    def get_queries(self):
        return sanitize(bvecs_mmap(self.basedir + 'bigann_query.bvecs')[:])

    def get_train(self, maxtrain=None):
        maxtrain = maxtrain if maxtrain is not None else self.nt
        return sanitize(bvecs_mmap(self.basedir + 'bigann_learn.bvecs')[:maxtrain])

    def get_groundtruth(self, k=None):
        gt = ivecs_read(self.basedir + 'gnd/idx_%dM.ivecs' % self.nb_M)
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt

    def get_database(self):
        assert self.nb_M < 100, "dataset too large, use iterator"
        return sanitize(bvecs_mmap(self.basedir + 'bigann_base.bvecs')[:self.nb])

    def database_iterator(self, bs=128, split=(1, 0)):
        xb = bvecs_mmap(self.basedir + 'bigann_base.bvecs')
        nsplit, rank = split
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        for j0 in range(i0, i1, bs):
            yield sanitize(xb[j0: min(j0 + bs, i1)])


class DatasetDeep1B(Dataset):
    """
    See
    https://github.com/facebookresearch/faiss/tree/main/benchs#getting-deep1b
    on how to get the data
    """

    def __init__(self, nb=10**9):
        Dataset.__init__(self)
        nb_to_name = {
            10**5: '100k',
            10**6: '1M',
            10**7: '10M',
            10**8: '100M',
            10**9: '1B'
        }
        assert nb in nb_to_name
        self.d, self.nt, self.nb, self.nq = 96, 358480000, nb, 10000
        self.basedir = dataset_basedir + 'deep1b/'
        self.gt_fname = "%sdeep%s_groundtruth.ivecs" % (
            self.basedir, nb_to_name[self.nb])

    def get_queries(self):
        return sanitize(fvecs_read(self.basedir + "deep1B_queries.fvecs"))

    def get_train(self, maxtrain=None):
        maxtrain = maxtrain if maxtrain is not None else self.nt
        return sanitize(fvecs_mmap(self.basedir + "learn.fvecs")[:maxtrain])

    def get_groundtruth(self, k=None):
        gt = ivecs_read(self.gt_fname)
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt

    def get_database(self):
        assert self.nb <= 10**8, "dataset too large, use iterator"
        return sanitize(fvecs_mmap(self.basedir + "base.fvecs")[:self.nb])

    def database_iterator(self, bs=128, split=(1, 0)):
        xb = fvecs_mmap(self.basedir + "base.fvecs")
        nsplit, rank = split
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        for j0 in range(i0, i1, bs):
            yield sanitize(xb[j0: min(j0 + bs, i1)])


class DatasetGlove(Dataset):
    """
    Data from http://ann-benchmarks.com/glove-100-angular.hdf5
    """

    def __init__(self, loc=None, download=False):
        import h5py
        assert not download, "not implemented"
        if not loc:
            loc = dataset_basedir + 'glove/glove-100-angular.hdf5'
        self.glove_h5py = h5py.File(loc, 'r')
        # IP and L2 are equivalent in this case, but it is traditionally seen as an IP dataset
        self.metric = 'IP'
        self.d, self.nt = 100, 0
        self.nb = self.glove_h5py['train'].shape[0]
        self.nq = self.glove_h5py['test'].shape[0]

    def get_queries(self):
        xq = np.array(self.glove_h5py['test'])
        faiss.normalize_L2(xq)
        return xq

    def get_database(self):
        xb = np.array(self.glove_h5py['train'])
        faiss.normalize_L2(xb)
        return xb

    def get_groundtruth(self, k=None):
        gt = self.glove_h5py['neighbors']
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt


class DatasetMusic100(Dataset):
    """
    get dataset from
    https://github.com/stanis-morozov/ip-nsw#dataset
    """

    def __init__(self):
        Dataset.__init__(self)
        self.d, self.nt, self.nb, self.nq = 100, 0, 10**6, 10000
        self.metric = 'IP'
        self.basedir = dataset_basedir + 'music-100/'

    def get_queries(self):
        xq = np.fromfile(self.basedir + 'query_music100.bin', dtype='float32')
        xq = xq.reshape(-1, 100)
        return xq

    def get_database(self):
        xb = np.fromfile(self.basedir + 'database_music100.bin', dtype='float32')
        xb = xb.reshape(-1, 100)
        return xb

    def get_groundtruth(self, k=None):
        gt = np.load(self.basedir + 'gt.npy')
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt

class DatasetGIST1M(Dataset):
    """
    The original dataset is available at: http://corpus-texmex.irisa.fr/
    (ANN_SIFT1M)
    """

    def __init__(self):
        Dataset.__init__(self)
        self.d, self.nt, self.nb, self.nq = 960, 100000, 1000000, 10000
        self.basedir = dataset_basedir + 'gist1M/'

    def get_queries(self):
        return fvecs_read(self.basedir + "gist_query.fvecs")

    def get_train(self, maxtrain=None):
        maxtrain = maxtrain if maxtrain is not None else self.nt
        return fvecs_read(self.basedir + "gist_learn.fvecs")[:maxtrain]

    def get_database(self):
        return fvecs_read(self.basedir + "gist_base.fvecs")

    def get_groundtruth(self, k=None):
        gt = ivecs_read(self.basedir + "gist_groundtruth.ivecs")
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt


def dataset_from_name(dataset='deep1M', download=False):
    """ converts a string describing a dataset to a Dataset object
    Supports sift1M, bigann1M..bigann1B, deep1M..deep1B, music-100 and glove
    """

    if dataset == 'sift1M':
        return DatasetSIFT1M()

    elif dataset == 'gist1M':
        return DatasetGIST1M()

    elif dataset.startswith('bigann'):
        dbsize = 1000 if dataset == "bigann1B" else int(dataset[6:-1])
        return DatasetBigANN(nb_M=dbsize)

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
        return DatasetDeep1B(nb=dbsize)

    elif dataset == "music-100":
        return DatasetMusic100()

    elif dataset == "glove":
        return DatasetGlove(download=download)

    else:
        raise RuntimeError("unknown dataset " + dataset)
