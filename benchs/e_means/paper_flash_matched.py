# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build $EMEANS_CACHE/flash_matched/flash_kmeans: Flash-KMeans 0.3.1 with squared norms in
float32.

Flash-KMeans works in float16, whose largest value is 65,504. SimSearchNet++ rows have squared
norms near 4.3e6, and SPACEV's reach 6.2e4, so the paper ran a copy of Flash-KMeans that computes
squared norms in float32 on those two corpora. This script makes that copy from the installed
package by applying the paper's eight one-line edits, and refuses to run if any edit does not
match exactly. paper_xb.py puts the copy first on the path for SPACEV and SimSearchNet++
only.

    python paper_flash_matched.py
"""
import importlib.metadata
import importlib.util
import os
import shutil

TARGET = os.path.join(os.environ.get("EMEANS_CACHE", "cache"), "flash_matched",
                      "flash_kmeans")
# file: [(stock text, patched text, occurrences)]
EDITS = {
    "interface.py": [
        ("(x_b[:, i:i + _CHUNK] ** 2).sum(dim=-1)",
         "(x_b[:, i:i + _CHUNK].float() ** 2).sum(dim=-1)", 1)],
    "kmeans_large.py": [
        ("(centroids_g[g] ** 2).sum(dim=-1)", "(centroids_g[g].float() ** 2).sum(dim=-1)", 2),
        ("(x_block ** 2).sum(dim=-1)", "(x_block.float() ** 2).sum(dim=-1)", 2)],
    "kmeans_triton_impl.py": [
        ("(x[:, i:i + XSQ_CHUNK] ** 2).sum(dim=-1)",
         "(x[:, i:i + XSQ_CHUNK].float() ** 2).sum(dim=-1)", 1)],
    "torch_fallback.py": [
        ("(centroids ** 2).sum(dim=-1)", "(centroids.float() ** 2).sum(dim=-1)", 1),
        ("(x ** 2).sum(dim=-1)", "(x.float() ** 2).sum(dim=-1)", 1)],
}


def main():
    version = importlib.metadata.version("flash-kmeans")
    assert version == "0.3.1", "the paper used flash-kmeans 0.3.1, found %s" % version
    source = os.path.dirname(importlib.util.find_spec("flash_kmeans").origin)
    target = TARGET
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    for name, edits in EDITS.items():
        path = os.path.join(target, name)
        text = open(path).read()
        for stock, patched, count in edits:
            assert text.count(stock) == count, "%s: expected %d of %r, found %d" % (
                name, count, stock, text.count(stock))
            text = text.replace(stock, patched)
        open(path, "w").write(text)
    with open(os.path.join(os.path.dirname(target), "VERSION"), "w") as fh:
        fh.write(version + "\n")              # paper_xb.py checks it before every run
    print("wrote %s (flash-kmeans %s, 8 edits)" % (target, version))


if __name__ == "__main__":
    main()
