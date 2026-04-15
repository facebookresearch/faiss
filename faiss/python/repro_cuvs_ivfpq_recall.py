# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Reproducer for cuVS IVF-PQ non-monotonic recall regression but for
open source datasets.

Observed with: ~85M vectors, d=100, M=100 (d_sub=1), nlist=16384, metric=IP.
The recall dip of about 3% occurs when going from nprobe 32 -> 64.
However, not reproducible on all datasets.

Requirements (conda):

conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.14.1 numpy h5py

Usage (synthetic data — fully self-contained):
    python repro_cuvs_ivfpq_recall.py

    # Larger scale (closer to production repro):
    python repro_cuvs_ivfpq_recall.py --nb 10000000 --nlist 4096

    # Full production-scale repro (requires ~32 GB RAM + large GPU):
    python repro_cuvs_ivfpq_recall.py --nb 85000000 --nlist 16384

Usage (VIBE benchmark datasets — auto-downloaded from HuggingFace):
    # Run on two small datasets (auto-downloads ~600 MB):
    python repro_cuvs_ivfpq_recall.py --vibe

    # Specific VIBE datasets:
    python repro_cuvs_ivfpq_recall.py --vibe \
        --vibe_datasets llama-128-ip yi-128-ip glove-200-cosine

    # All VIBE datasets:
    python repro_cuvs_ivfpq_recall.py --vibe --vibe_all

    # Custom cache directory:
    python repro_cuvs_ivfpq_recall.py --vibe --vibe_dir /path/to/cache

VIBE datasets (from https://huggingface.co/datasets/vector-index-bench/vibe):
    HDF5 files are auto-downloaded and cached locally. Available datasets:
        llama-128-ip                    d=128,  ~350 MB, IP
        yi-128-ip                       d=128,  ~250 MB, IP
        glove-200-cosine                d=200,  ~900 MB, cosine/IP
        yandex-200-cosine               d=200,  ~1.9 GB, cosine/IP
        yahoo-minilm-384-normalized     d=384,  ~1.0 GB, IP
        imagenet-clip-512-normalized    d=512,  ~2.5 GB, IP
        laion-clip-512-normalized       d=512,  ~4.2 GB, IP
        arxiv-nomic-768-normalized      d=768,  ~3.9 GB, IP
        codesearchnet-jina-768-cosine   d=768,  ~4.0 GB, IP
        landmark-dino-768-cosine        d=768,  ~2.2 GB, IP
        agnews-mxbai-1024-euclidean     d=1024, ~3.0 GB, L2
        celeba-resnet-2048-cosine       d=2048, ~1.6 GB, IP
        (and more — see VIBE_DATASETS list below)
"""

import argparse
import os
import sys
import time
import urllib.request

import faiss
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import rmm
    HAS_RMM = True
except ImportError:
    HAS_RMM = False

# ---------------------------------------------------------------------------
# VIBE dataset catalog (HuggingFace: vector-index-bench/vibe)
# ---------------------------------------------------------------------------

HF_BASE_URL = (
    "https://huggingface.co/datasets/vector-index-bench/vibe"
    "/resolve/main"
)

# Dataset name -> HDF5 filename on HuggingFace
VIBE_DATASETS = {
    "agnews-mxbai-1024-euclidean": "agnews-mxbai-1024-euclidean.hdf5",
    "arxiv-nomic-768-normalized": "arxiv-nomic-768-normalized.hdf5",
    "ccnews-nomic-768-normalized": "ccnews-nomic-768-normalized.hdf5",
    "celeba-resnet-2048-cosine": "celeba-resnet-2048-cosine.hdf5",
    "coco-nomic-768-normalized": "coco-nomic-768-normalized.hdf5",
    "codesearchnet-jina-768-cosine": "codesearchnet-jina-768-cosine.hdf5",
    "glove-200-cosine": "glove-200-cosine.hdf5",
    "gooaq-distilroberta-768-normalized": "gooaq-distilroberta-768-normalized.hdf5",
    "imagenet-align-640-normalized": "imagenet-align-640-normalized.hdf5",
    "imagenet-clip-512-normalized": "imagenet-clip-512-normalized.hdf5",
    "laion-clip-512-normalized": "laion-clip-512-normalized.hdf5",
    "landmark-dino-768-cosine": "landmark-dino-768-cosine.hdf5",
    "landmark-nomic-768-normalized": "landmark-nomic-768-normalized.hdf5",
    "llama-128-ip": "llama-128-ip.hdf5",
    "simplewiki-openai-3072-normalized": "simplewiki-openai-3072-normalized.hdf5",
    "yahoo-minilm-384-normalized": "yahoo-minilm-384-normalized.hdf5",
    "yandex-200-cosine": "yandex-200-cosine.hdf5",
    "yi-128-ip": "yi-128-ip.hdf5",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_rmm():
    if not HAS_RMM:
        print("WARNING: rmm not available. cuVS will use default allocator.")
        print("  For best performance: pip install rmm-cuXX")
        return
    mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
    rmm.mr.set_current_device_resource(mr)
    print("RMM memory pool initialized.")


def download_vibe_dataset(name, cache_dir):
    """Download a VIBE HDF5 file from HuggingFace if not cached locally."""
    if name not in VIBE_DATASETS:
        print(f"ERROR: Unknown dataset '{name}'.")
        print(f"  Available: {', '.join(sorted(VIBE_DATASETS.keys()))}")
        sys.exit(1)

    fname = VIBE_DATASETS[name]
    local_path = os.path.join(cache_dir, fname)

    if os.path.exists(local_path):
        return local_path

    url = f"{HF_BASE_URL}/{fname}"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  Downloading {name} from HuggingFace...")
    print(f"    URL: {url}")
    print(f"    -> {local_path}")

    tmp_path = local_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path, _download_progress)
        os.rename(tmp_path, local_path)
        print()
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"\n  ERROR downloading {name}: {e}")
        sys.exit(1)

    return local_path


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r    {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="",
              flush=True)


def load_vibe_hdf5(path):
    """Load a VIBE HDF5 dataset. Returns (xb, xq, gt, metric, metric_name)."""
    if not HAS_H5PY:
        print("ERROR: h5py is required for VIBE datasets.")
        print("  pip install h5py")
        sys.exit(1)

    with h5py.File(path, "r") as f:
        distance = f.attrs.get("distance", "euclidean")
        if isinstance(distance, bytes):
            distance = distance.decode()

        xb = np.array(f["train"], dtype="float32")
        xq = np.array(f["test"], dtype="float32")
        gt = np.array(f["neighbors"], dtype="int64")

    if distance == "euclidean":
        faiss_metric = faiss.METRIC_L2
        metric_name = "L2"
    else:
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        metric_name = "IP"

    return xb, xq, gt, faiss_metric, metric_name


def pick_m_values(d):
    """Pick PQ M values targeting d_sub in {1, 2, 4, 8}."""
    m_values = set()
    for dsub in [1, 2, 4, 8]:
        m = d // dsub
        if m > 0 and d % m == 0:
            m_values.add(m)
    return sorted(m_values)


def compute_recall(gt, predictions, k):
    """Compute recall@k via set intersection."""
    n = gt.shape[0]
    gt_k = gt[:, :k]
    pred_k = predictions[:, :k]
    correct = sum(
        len(np.intersect1d(gt_k[i], pred_k[i])) for i in range(n)
    )
    return correct / (n * k)


# ---------------------------------------------------------------------------
# Synthetic data generation (from faiss.contrib.datasets.SyntheticDataset)
# ---------------------------------------------------------------------------

def generate_synthetic_data(d, nt, nb, nq, seed=1338):
    """Generate synthetic data with intrinsic low-dimensional structure."""
    d1 = 10  # intrinsic dimension
    rs = np.random.RandomState(seed)
    proj = rs.rand(d1, d).astype("float32")
    scale = rs.rand(d).astype("float32") * 4 + 0.1

    def make(n, label):
        print(f"  Generating {label} ({n:,} x {d})...")
        t0 = time.time()
        x = rs.normal(size=(n, d1)).astype("float32")
        x = np.dot(x, proj)
        x *= scale
        np.sin(x, out=x)
        print(f"    {time.time() - t0:.1f}s")
        return x

    xt = make(nt, "train")
    xq = make(nq, "queries")
    xb = make(nb, "database")
    return xt, xb, xq


def load_or_generate_data(data_dir, d, nt, nb, nq, seed):
    """Load cached synthetic data or generate and save."""
    xb_path = os.path.join(data_dir, "xb.npy")
    xt_path = os.path.join(data_dir, "xt.npy")
    xq_path = os.path.join(data_dir, "xq.npy")

    if all(os.path.exists(p) for p in [xb_path, xt_path, xq_path]):
        print(f"Loading cached data from {data_dir}...")
        t0 = time.time()
        xt = np.load(xt_path)
        xq = np.load(xq_path)
        xb = np.load(xb_path, mmap_mode="r")
        print(f"  xt: {xt.shape}, xq: {xq.shape}, xb: {xb.shape}")
        print(f"  Loaded in {time.time() - t0:.1f}s")
        return xt, xb, xq

    print("Generating synthetic data...")
    t0 = time.time()
    xt, xb, xq = generate_synthetic_data(d, nt, nb, nq, seed=seed)
    print(f"  Total: {time.time() - t0:.1f}s")

    os.makedirs(data_dir, exist_ok=True)
    print(f"Saving to {data_dir}...")
    np.save(xt_path, xt)
    np.save(xq_path, xq)
    np.save(xb_path, xb)

    return xt, xb, xq


def compute_or_load_groundtruth(data_dir, xb, xq, k, metric):
    """Load cached ground truth or compute with CPU flat index."""
    suffix = "_ip" if metric == faiss.METRIC_INNER_PRODUCT else "_l2"
    gt_path = os.path.join(data_dir, f"gt_k{k}{suffix}.npy")

    if os.path.exists(gt_path):
        print(f"Loading ground truth from {gt_path}...")
        t0 = time.time()
        gt = np.load(gt_path)
        print(f"  gt: {gt.shape}, {time.time() - t0:.1f}s")
        return gt

    metric_name = "IP" if metric == faiss.METRIC_INNER_PRODUCT else "L2"
    print(f"Computing ground truth (CPU flat {metric_name}, k={k})...")
    t0 = time.time()
    _, gt = faiss.knn(xq, np.ascontiguousarray(xb), k, metric=metric)
    print(f"  {time.time() - t0:.1f}s")

    np.save(gt_path, gt)
    return gt


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_index(index_gpu, xq, gt, k, nprobe_values, nlist, label):
    """Search with varying nprobe, return list of result dicts."""
    nq = xq.shape[0]
    results = []

    # Warmup
    index_gpu.nprobe = nprobe_values[0]
    index_gpu.search(xq[:min(100, nq)], k)

    header = (f"{'nprobe':>8}  {'Recall@' + str(k):>12}  "
              f"{'QPS':>12}  {'Latency(ms)':>14}")
    print(f"\n{header}")
    print("-" * len(header))

    prev_recall = -1.0
    for nprobe in nprobe_values:
        if nprobe > nlist:
            continue
        index_gpu.nprobe = nprobe

        t0 = time.time()
        _, I = index_gpu.search(xq, k)
        elapsed = time.time() - t0

        recall = compute_recall(gt, I, k)
        qps = nq / elapsed
        latency_ms = elapsed * 1000 / nq

        flag = ""
        if recall < prev_recall - 1e-6:
            flag = "  <-- RECALL DROP"
        prev_recall = recall

        print(f"{nprobe:>8}  {recall:>12.4f}  {qps:>12.1f}  "
              f"{latency_ms:>14.3f}{flag}")
        results.append({
            "index": label, "nprobe": nprobe,
            "recall": recall, "qps": qps,
        })

    return results


def run_synthetic_benchmark(args):
    """Run benchmark on synthetic data."""
    d = args.d
    nb = args.nb
    nq = args.nq
    nt = args.nt
    k = args.k
    nlist = args.nlist
    M_values = args.M
    nprobe_values = sorted(args.nprobe)
    data_dir = args.data_dir

    if args.metric == "IP":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    else:
        faiss_metric = faiss.METRIC_L2

    # Validate M values
    for M in M_values:
        if d % M != 0:
            factors = sorted(i for i in range(1, d + 1) if d % i == 0)
            print(f"ERROR: d={d} not divisible by M={M}. Valid: {factors}")
            sys.exit(1)

    print(f"Config: d={d}, nb={nb:,}, nq={nq:,}, k={k}")
    print(f"  M={M_values}, nlist={nlist}, metric={args.metric}")
    print(f"  nprobe sweep: {nprobe_values}")
    print()

    # Data
    xt, xb, xq = load_or_generate_data(data_dir, d, nt, nb, nq, args.seed)
    if not xb.flags["C_CONTIGUOUS"] or xb.base is not None:
        print("Loading xb into RAM...")
        t0 = time.time()
        xb = np.array(xb)
        print(f"  {time.time() - t0:.1f}s")
    print()

    # Ground truth
    gt = compute_or_load_groundtruth(data_dir, xb, xq, k, faiss_metric)
    print()

    res = faiss.StandardGpuResources()
    all_results = []

    for M in M_values:
        dsub = d // M
        if faiss_metric == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(d)
        else:
            quantizer = faiss.IndexFlatL2(d)

        print(f"\n{'='*60}")
        print(f"IVF{nlist},PQ{M} (d_sub={dsub}), metric={args.metric}")
        print(f"{'='*60}")

        index_cpu = faiss.IndexIVFPQ(
            quantizer, d, nlist, M, 8, faiss_metric
        )
        print("Training...")
        t0 = time.time()
        index_cpu.train(xt)
        print(f"  {time.time() - t0:.1f}s")

        print("Adding vectors...")
        t0 = time.time()
        index_cpu.add(xb)
        print(f"  {time.time() - t0:.1f}s")

        # --- cuVS ---
        print("Cloning to GPU (cuVS=True)...")
        co = faiss.GpuClonerOptions()
        co.use_cuvs = True
        t0 = time.time()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)
        print(f"  {time.time() - t0:.1f}s")

        results = bench_index(
            index_gpu, xq, gt, k, nprobe_values, nlist, f"cuVS PQ{M}"
        )
        all_results.extend(results)
        del index_gpu

        # --- Classic Faiss GPU ---
        if args.compare_classic:
            print("\nCloning to GPU (cuVS=False, classic Faiss GPU)...")
            co = faiss.GpuClonerOptions()
            co.use_cuvs = False
            t0 = time.time()
            index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)
            print(f"  {time.time() - t0:.1f}s")

            results = bench_index(
                index_gpu, xq, gt, k, nprobe_values, nlist,
                f"classic PQ{M}"
            )
            all_results.extend(results)
            del index_gpu

        del index_cpu

    print_summary(all_results, k, args.metric)
    check_monotonicity(all_results, k)


def run_vibe_benchmark(args):
    """Run benchmark on VIBE datasets (auto-downloaded from HuggingFace)."""
    vibe_dir = args.vibe_dir
    nprobe_values = sorted(args.nprobe)
    k = args.k
    nlist = args.nlist

    if args.vibe_all:
        datasets = list(VIBE_DATASETS.keys())
    elif args.vibe_datasets:
        datasets = args.vibe_datasets
    else:
        # Default to two small IP datasets that are fast to benchmark
        datasets = ["llama-128-ip", "yi-128-ip"]

    print(f"VIBE datasets to run: {datasets}")
    print(f"Cache dir: {vibe_dir}")
    print()

    grand_results = []

    for name in datasets:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {name}")
        print(f"{'#'*70}")

        # Download from HuggingFace if needed
        hdf5_path = download_vibe_dataset(name, vibe_dir)

        # Load HDF5
        print(f"Loading {hdf5_path}...")
        t0 = time.time()
        xb, xq, gt, faiss_metric, metric_name = load_vibe_hdf5(hdf5_path)
        nb, d = xb.shape
        nq = xq.shape[0]
        gt_k = gt.shape[1]
        print(f"  xb: {xb.shape}, xq: {xq.shape}, gt: {gt.shape}")
        print(f"  Metric: {metric_name}, d={d}, nb={nb:,}, loaded in "
              f"{time.time() - t0:.1f}s")

        use_k = min(k, gt_k)

        # Pick M values
        if args.M:
            m_values = [m for m in args.M if d % m == 0]
            if len(m_values) < len(args.M):
                skipped = [m for m in args.M if d % m != 0]
                print(f"  Skipping M={skipped} (d={d} not divisible)")
        else:
            m_values = pick_m_values(d)

        if not m_values:
            print(f"  No valid M values for d={d}, skipping.")
            continue

        print(f"  M values: {m_values}, nlist: {nlist}")

        # Training data
        nt = min(1_000_000, nb)
        if nt < nb:
            rng = np.random.RandomState(1338)
            xt = xb[rng.choice(nb, nt, replace=False)].copy()
        else:
            xt = xb.copy()

        res = faiss.StandardGpuResources()
        ds_results = []

        for M in m_values:
            dsub = d // M
            if faiss_metric == faiss.METRIC_INNER_PRODUCT:
                quantizer = faiss.IndexFlatIP(d)
            else:
                quantizer = faiss.IndexFlatL2(d)

            print(f"\n{'='*60}")
            print(f"IVF{nlist},PQ{M} (d_sub={dsub}), metric={metric_name}")
            print(f"{'='*60}")

            index_cpu = faiss.IndexIVFPQ(
                quantizer, d, nlist, M, 8, faiss_metric
            )
            print("Training...")
            t0 = time.time()
            index_cpu.train(xt)
            print(f"  {time.time() - t0:.1f}s")

            print("Adding vectors...")
            t0 = time.time()
            index_cpu.add(xb)
            print(f"  {time.time() - t0:.1f}s")

            # cuVS
            print("Cloning to GPU (cuVS=True)...")
            co = faiss.GpuClonerOptions()
            co.use_cuvs = True
            t0 = time.time()
            index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)
            print(f"  {time.time() - t0:.1f}s")

            results = bench_index(
                index_gpu, xq, gt, use_k, nprobe_values, nlist,
                f"cuVS PQ{M}"
            )
            ds_results.extend(results)
            del index_gpu

            # Classic
            if args.compare_classic:
                print("\nCloning to GPU (cuVS=False, classic)...")
                co = faiss.GpuClonerOptions()
                co.use_cuvs = False
                t0 = time.time()
                index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)
                print(f"  {time.time() - t0:.1f}s")

                results = bench_index(
                    index_gpu, xq, gt, use_k, nprobe_values, nlist,
                    f"classic PQ{M}"
                )
                ds_results.extend(results)
                del index_gpu

            del index_cpu

        # Per-dataset summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {name} (metric={metric_name}, d={d}, nb={nb:,})")
        print(f"{'='*60}")
        print(f"{'index':>20}  {'nprobe':>8}  {'Recall@' + str(use_k):>12}  "
              f"{'QPS':>12}")
        print("-" * 58)
        for r in ds_results:
            print(f"{r['index']:>20}  {r['nprobe']:>8}  "
                  f"{r['recall']:>12.4f}  {r['qps']:>12.1f}")

        grand_results.extend(
            {**r, "dataset": name} for r in ds_results
        )
        check_monotonicity(ds_results, use_k)

    return grand_results


def print_summary(results, k, metric):
    """Print final summary table."""
    print(f"\n\n{'='*60}")
    print(f"SUMMARY (metric={metric})")
    print(f"{'='*60}")
    print(f"{'index':>20}  {'nprobe':>8}  {'Recall@' + str(k):>12}  {'QPS':>12}")
    print("-" * 58)
    for r in results:
        print(f"{r['index']:>20}  {r['nprobe']:>8}  "
              f"{r['recall']:>12.4f}  {r['qps']:>12.1f}")


def check_monotonicity(results, k):
    """Check for non-monotonic recall and flag it."""
    by_index = {}
    for r in results:
        by_index.setdefault(r["index"], []).append(r)

    found_issue = False
    for label, runs in by_index.items():
        runs_sorted = sorted(runs, key=lambda r: r["nprobe"])
        for i in range(1, len(runs_sorted)):
            prev = runs_sorted[i - 1]
            curr = runs_sorted[i]
            drop = prev["recall"] - curr["recall"]
            if drop > 1e-6:
                if not found_issue:
                    print(f"\n*** NON-MONOTONIC RECALL DETECTED ***")
                    found_issue = True
                print(
                    f"  {label}: nprobe {prev['nprobe']}->{curr['nprobe']}  "
                    f"recall {prev['recall']:.4f}->{curr['recall']:.4f}  "
                    f"(drop={drop:.4f})"
                )

    if not found_issue:
        print(f"\nNo recall regression detected at this scale.")
        print("Try increasing --nb and --nlist for larger-scale repro.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproducer for cuVS IVF-PQ non-monotonic recall drop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument("--vibe", action="store_true",
                        help="Use VIBE benchmark datasets (auto-downloaded from HuggingFace)")
    parser.add_argument("--vibe_dir", type=str, default="./vibe_datasets",
                        help="Local cache directory for downloaded VIBE HDF5 files")
    parser.add_argument("--vibe_datasets", nargs="+", default=None,
                        help="VIBE dataset names to run (default: llama-128-ip yi-128-ip)")
    parser.add_argument("--vibe_all", action="store_true",
                        help="Run all VIBE datasets")

    # Synthetic data options
    parser.add_argument("--d", type=int, default=100,
                        help="Vector dimension for synthetic data")
    parser.add_argument("--nb", type=int, default=1_000_000,
                        help="Database size (use 85000000 for full repro)")
    parser.add_argument("--nt", type=int, default=500_000,
                        help="Training set size")
    parser.add_argument("--nq", type=int, default=10_000,
                        help="Number of queries")
    parser.add_argument("--seed", type=int, default=1338)
    parser.add_argument("--data_dir", type=str, default="./cuvs_repro_data",
                        help="Directory to cache generated data")

    # Index options
    parser.add_argument("--M", type=int, nargs="+", default=None,
                        help="PQ M values (default: auto-select for d)")
    parser.add_argument("--nlist", type=int, default=1024,
                        help="IVF centroids (use 16384 for full repro)")
    parser.add_argument("--nprobe", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256],
                        help="nprobe values to sweep")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbors")
    parser.add_argument("--metric", type=str, default="IP",
                        choices=["L2", "IP"],
                        help="Distance metric")

    # Comparison
    parser.add_argument("--compare_classic", action="store_true",
                        help="Also run classic (non-cuVS) Faiss GPU for comparison")

    args = parser.parse_args()

    # Auto-select M values if not specified
    if args.M is None:
        if args.vibe:
            args.M = []  # auto per-dataset
        else:
            args.M = pick_m_values(args.d)

    print("=" * 60)
    print("cuVS IVF-PQ Recall Regression Reproducer")
    print("=" * 60)
    print(f"  faiss version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    print(f"  GPU count: {faiss.get_num_gpus()}")
    print()

    setup_rmm()
    print()

    if args.vibe:
        run_vibe_benchmark(args)
    else:
        run_synthetic_benchmark(args)


if __name__ == "__main__":
    main()
