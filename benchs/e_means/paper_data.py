# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The datasets of the e-means paper's Figure 3, Figure 4 and Table 3, split as in the paper.

The files are public and live under the folder named by the EMEANS_DATA environment variable
(default ./data, file list in README.md). For each dataset:

  train  the rows the paper trained on: the first n rows of the corpus (ImageNet-CLIP: a fixed
         one-million-row subset of its training split)
  test   the paper's test rows: every test error in the paper is measured on these rows

Where the test rows come from. Each dataset has a frozen evaluation set, cut in half by one fixed
permutation (seed 1234): the first half is validation (unused here), the second half is test.
  DEEP* and SIFT, 1M and 10M   the 10,000 rows that follow the training rows in the same file
  DEEP* and SIFT, 100M         the corpus's public query file (10,000 rows)
  DINOv3                       the corpus's public query file (100,000 rows)
  SPACEV 100M                  the public query file of that separate 100M sample
  SPACEV 1M/10M, SimSearchNet++, ImageNet-CLIP
                               20,000 rows held out of the file, listed in paper_splits/ (read
                               from heldout_rows.npy beside the corpus when paper_download.py
                               saved them there, so the whole corpus need not be downloaded)

Training rows of a prefix corpus stay in the type the file stores (float32, uint8 or int8) and
are memory-mapped, so a 100M corpus is never copied. rows(train, name, ...) converts a slice to
float32 with the dataset's rescaling (a power of two, so exact).

load() compares each dataset once with the paper's own arrays (paper_splits/fingerprints.json):
both evaluation halves byte for byte, and 1,000 evenly spaced training rows. A different file, a
wrong row range or a wrong rescaling fails loudly instead of producing plausible numbers.
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.environ.get("EMEANS_DATA", "data")
CACHE = os.environ.get("EMEANS_CACHE", "cache")
SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_splits")

DEEP = ("deep1b/base.1B.fbin", "deep1b/base.fvecs")           # either file works
DEEP_QUERIES = ("deep1b/query.public.10K.fbin", "deep1b/deep1B_queries.fvecs")
SIFT = ("bigann/base.1B.u8bin", "bigann/bigann_base.bvecs")
SIFT_QUERIES = ("bigann/query.public.10K.u8bin", "bigann/bigann_query.bvecs")
SPACEV = ("spacev1b/base.1.4B.i8bin",)
SPACEV100M = ("spacev100m/base.100M.i8bin",)                 # a separate 100M sample
SPACEV100M_QUERIES = ("spacev100m/query.30K.i8bin",)
SSNPP = ("fb_ssnpp/FB_ssnpp_database.u8bin",)
DINO = ("dino_vitl_10B/chunked_base_10B/chunk_0000.bvecs",)
DINO_QUERIES = ("dino_vitl_10B/queries_clean.bvecs",)
IMAGENET = ("vibe/imagenet-clip-512-normalized.hdf5",)
# The held-out rows of SPACEV and SimSearchNet++ sit near the end of their files; paper_download.py
# saves them in this small file beside the corpus, so only the corpus's prefix has to be downloaded.
HELDOUT = "heldout_rows.npy"

# name: base file, rows used for training, where the evaluation rows are, and a rescaling.
#   eval="next"        the 10,000 rows right after the training rows
#   eval=<files>       a query file
#   held_out=<.npy>    row numbers of the base file (paper_splits/)
DATASETS = {
    "deep1b-1M": dict(base=DEEP, n=10**6, eval="next"),
    "deep1b-10M": dict(base=DEEP, n=10**7, eval="next"),
    "deep1b-100M": dict(base=DEEP, n=10**8, eval=DEEP_QUERIES),
    "sift1b-10M-normalized": dict(base=SIFT, n=10**7, eval="next", normalize=True),
    "sift1b-10M-div512": dict(base=SIFT, n=10**7, eval="next", divide=512.0),
    "sift1b-100M-div512": dict(base=SIFT, n=10**8, eval=SIFT_QUERIES, divide=512.0),
    "spacev-1M": dict(base=SPACEV, n=10**6, held_out="spacev_holdout_rows.npy"),
    "spacev-10M": dict(base=SPACEV, n=10**7, held_out="spacev_holdout_rows.npy"),
    "spacev-100M": dict(base=SPACEV100M, n=10**8, eval=SPACEV100M_QUERIES),
    "fb-ssnpp-1M": dict(base=SSNPP, n=10**6, held_out="fb_ssnpp_holdout_rows.npy"),
    "fb-ssnpp-10M": dict(base=SSNPP, n=10**7, held_out="fb_ssnpp_holdout_rows.npy"),
    "fb-ssnpp-100M": dict(base=SSNPP, n=10**8, held_out="fb_ssnpp_holdout_rows.npy"),
    "dino10b-1M-div4096": dict(base=DINO, n=10**6, eval=DINO_QUERIES, divide=4096.0),
    "dino10b-10M-div4096": dict(base=DINO, n=10**7, eval=DINO_QUERIES, divide=4096.0),
    "imagenet-clip-512-normalized": dict(base=IMAGENET, n=10**6,
                                         held_out="imagenet_clip_holdout_rows.npy"),
}


def find(names):
    """The first of the candidate file names present under ROOT."""
    for name in names:
        if os.path.exists(os.path.join(ROOT, name)):
            return os.path.join(ROOT, name)
    raise FileNotFoundError("none of %s under %s (see README.md)" % (list(names), ROOT))


def matrix(path):
    """A vector file as a read-only memory map of shape (n, d), in the type it stores.

    .fvecs/.bvecs store a 4-byte dimension before every row; .fbin/.u8bin/.i8bin store one
    8-byte header (rows, dimension) and then the rows. The row count comes from the file
    size, so a partially downloaded file works as long as it holds the rows asked for."""
    ext = os.path.splitext(path)[1]
    size = os.path.getsize(path)
    if ext in (".fvecs", ".bvecs"):
        dtype = np.dtype(np.float32 if ext == ".fvecs" else np.uint8)
        d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
        width = 4 + d * dtype.itemsize                           # bytes per row
        raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(size // width, width))
        return raw[:, 4:].view(dtype)                            # (n, d)
    dtype = np.dtype({".fbin": np.float32, ".u8bin": np.uint8, ".i8bin": np.int8}[ext])
    n, d = (int(v) for v in np.fromfile(path, dtype=np.int32, count=2))
    n = min(n, (size - 8) // (d * dtype.itemsize))
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def _rescale(arr, spec):
    if spec.get("normalize"):                   # unit rows in float32, zero rows stay zero
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr /= norms
    if spec.get("divide"):                      # a power of two: exact in float32
        arr /= spec["divide"]
    return arr


def rows(train, name, start=0, stop=None, index=None):
    """float32 copy of training rows (a range, or the listed row numbers), rescaled as the paper
    rescaled them. `train` is what load(name) returned; cached training rows are already
    rescaled."""
    part = np.array(train[start:stop] if index is None else train[index], dtype=np.float32)
    return part if _cached_train(name) else _rescale(part, DATASETS[name])


def halves(evaluation):
    """The paper's validation and test halves of an evaluation set (fixed seed 1234)."""
    perm = np.random.RandomState(1234).permutation(evaluation.shape[0])
    n_val = int(round(0.5 * evaluation.shape[0]))
    return evaluation[perm[:n_val]], evaluation[perm[n_val:]]


def sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def _cached_train(name):
    """The two datasets that are not a prefix of a file keep float32 training rows in the cache."""
    spec = DATASETS[name]
    return bool(spec.get("normalize")) or spec["base"] == IMAGENET


def _imagenet(spec):
    """ImageNet-CLIP: the held-out rows come out of its training split, and Figure 3 trains on
    one fixed one-million-row subset of what is left."""
    import h5py
    corpus = np.asarray(h5py.File(find(spec["base"]), "r")["train"], dtype=np.float32)
    held = np.load(os.path.join(SPLITS, spec["held_out"]))
    keep = np.ones(corpus.shape[0], dtype=bool)
    keep[held] = False
    train, evaluation = corpus[keep], corpus[held]
    subset = np.sort(np.random.RandomState(1234).choice(train.shape[0], spec["n"],
                                                        replace=False))
    return train[subset], evaluation


def _prefix(name):
    spec = DATASETS[name]
    mat = matrix(find(spec["base"]))
    assert mat.shape[0] >= spec["n"], "%s holds %d rows, %d needed" % (
        find(spec["base"]), mat.shape[0], spec["n"])
    return mat[:spec["n"]]


def _evaluation(name):
    spec = DATASETS[name]
    if "held_out" in spec:
        index = np.load(os.path.join(SPLITS, spec["held_out"]))
        small = os.path.join(os.path.dirname(find(spec["base"])), HELDOUT)
        if os.path.exists(small):              # written by paper_download.py, in index order
            evaluation = np.load(small).astype(np.float32)
        else:
            evaluation = np.array(matrix(find(spec["base"]))[index], dtype=np.float32)
    elif spec["eval"] == "next":
        evaluation = np.array(matrix(find(spec["base"]))[spec["n"]:spec["n"] + 10_000],
                              dtype=np.float32)
    else:
        evaluation = np.array(matrix(find(spec["eval"])), dtype=np.float32)
    return _rescale(evaluation, spec)


def load(name):
    """(train, test) for one timed run.

    The first call checks the dataset against the paper's fingerprints, then caches the test rows
    (and, for the two non-prefix datasets, the float32 training rows) under $EMEANS_CACHE
    (default ./cache) with a stamp saying the check passed. Later calls, one per timed run since
    every run is a fresh process, only map the files. Cached files are written under a temporary
    name and renamed, so concurrent jobs never read a partial file."""
    spec = DATASETS[name]
    folder = os.path.join(CACHE, name)
    os.makedirs(folder, exist_ok=True)
    train_path, test_path = (os.path.join(folder, part + ".npy") for part in ("train", "test"))
    stamp = os.path.join(folder, "checked")
    own = _cached_train(name)
    if not os.path.exists(stamp):
        if spec["base"] == IMAGENET:
            train, evaluation = _imagenet(spec)
        else:
            train, evaluation = _prefix(name), _evaluation(name)
            if own:
                train = _rescale(np.array(train, dtype=np.float32), spec)
        val, test = halves(evaluation)
        want = json.load(open(os.path.join(SPLITS, "fingerprints.json")))[name]
        sample = np.linspace(0, train.shape[0] - 1, 1000).astype(np.int64)
        sampled = (np.asarray(train[sample]) if own
                   else _rescale(np.array(train[sample], dtype=np.float32), spec))
        for what, arr in (("val", val), ("test", test), ("train_rows", sampled)):
            assert sha256(arr) == want[what + "_sha256"], (
                "%s: the %s rows differ from the paper's; check the file and README.md" % (
                    name, what))
        assert tuple(train.shape) == (want["n_train"], want["dim"]), (name, train.shape)
        for path, arr in ((train_path, train), (test_path, test)) if own else ((test_path, test),):
            tmp = "%s.tmp%d.npy" % (path[:-4], os.getpid())
            np.save(tmp, arr)
            os.replace(tmp, path)
        open(stamp, "w").write("checked against paper_splits/fingerprints.json\n")
    train = np.load(train_path, mmap_mode="r") if own else _prefix(name)
    return train, np.load(test_path)
