# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import yaml

# with ssnpp sharded data
root = "/checkpoint/marialomeli/ssnpp_data"
file_names = [f"ssnpp_{i:010}.npy" for i in range(20)]
d = 256
dt = np.dtype(np.uint8)


def read_embeddings(fp):
    fl = os.path.getsize(fp)
    nb = fl // d // dt.itemsize
    print(nb)
    if fl == d * dt.itemsize * nb:  # no header
        return ("raw", np.memmap(fp, shape=(nb, d), dtype=dt, mode="r"))
    else:  # assume npy
        vecs = np.load(fp, mmap_mode="r")
        assert vecs.shape[1] == d
        assert vecs.dtype == dt
        return ("npy", vecs)


cfg = {}
files = []
size = 0
for fn in file_names:
    fp = f"{root}/{fn}"
    assert os.path.exists(fp), f"{fp} is missing"
    ft, xb = read_embeddings(fp)
    files.append(
        {"name": fn, "size": xb.shape[0], "dtype": dt.name, "format": ft}
    )
    size += xb.shape[0]

cfg["size"] = size
cfg["root"] = root
cfg["d"] = d
cfg["files"] = files
print(yaml.dump(cfg))
