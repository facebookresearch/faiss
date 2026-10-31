

Demos for a few Faiss functionalities
=====================================


demo_auto_tune.py
-----------------

Demonstrates the auto-tuning functionality of Faiss


demo_e_means.py
---------------

Runs the DEEP1M e-means quality experiment on a PyTorch device and reports the
median held-out MSE over ten seeds. Download Deep1B as described in
`../benchs/README.md` and pass the directory containing `deep1b/`:

    python demos/demo_e_means.py --data-dir /path/to/data

The final line reports the measured median test MSE for that run.

demo_e_means_distributed.py
---------------------------

Runs synchronous row-distributed CPU/GPU training on generated rank-local batches,
with replicated centroids and the same public numerical lifecycle:

    torchrun --standalone --nproc-per-node=2 demos/demo_e_means_distributed.py

Use `--transport dense` or `--transport sparse` to override the conservative
automatic choice. No dataset download is required. This is an integration
example, not a disk-loading or performance measurement.


demo_ondisk_ivf.py
------------------

Shows how to construct a Faiss index that stores the inverted file
data on disk, eg. when it does not fit in RAM. The script works on a
small dataset (sift1M) for demonstration and proceeds in stages:

0: train on the dataset

1-4: build 4 indexes, each containing 1/4 of the dataset. This can be
done in parallel on several machines

5: merge the 4 indexes into one that is written directly to disk
(needs not to fit in RAM)

6: load and test the index
