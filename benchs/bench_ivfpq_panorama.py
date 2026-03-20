# Quick 10% verification of IVFPQPanorama (with index caching)

import multiprocessing as mp
import os
import time

import faiss
import numpy as np

print("Compile options:", faiss.get_compile_options(), flush=True)


def fvecs_read(fname):
    a = np.fromfile(fname, dtype="float32")
    d = a[0].view("int32")
    return a.reshape(-1, d + 1)[:, 1:].copy()


GIST_DIR = "/datasets/PCA_init"
CACHE_DIR = "/home/akash/faiss-panorama/index_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

IVFPQ_CACHE = os.path.join(CACHE_DIR, "ivfpq_10pct.index")
IVFPQ_TRAINED_CACHE = os.path.join(CACHE_DIR, "ivfpq_trained_10pct.index")
IVFPQ_PANO_CACHE = os.path.join(CACHE_DIR, "ivfpq_pano_10pct.index")

print("Loading GIST1M data (10% subset)...", flush=True)
xb_full = fvecs_read(os.path.join(GIST_DIR, "gist1m_base.fvecs"))
xq = fvecs_read(os.path.join(GIST_DIR, "gist1m_query.fvecs"))

nb_full, d = xb_full.shape
nb = nb_full // 10  # 10% = 100000
xb = xb_full[:nb].copy()
del xb_full

nq = xq.shape[0]
print(f"Database: {nb} x {d}, Queries: {nq} x {d}", flush=True)

xt = xb[:50000].copy()

k = 10
M = 960
nbits = 8
nlist = 64
n_levels = 8
batch_size = 128

GT_PATH = os.path.join(CACHE_DIR, "gt_10pct.npy")
if os.path.exists(GT_PATH):
    gt_I = np.load(GT_PATH)
    print(f"Loaded cached ground truth: {gt_I.shape}", flush=True)
else:
    print("Computing ground truth on 10% subset...", flush=True)
    flat = faiss.IndexFlatL2(d)
    flat.add(xb)
    _, gt_I = flat.search(xq, k)
    np.save(GT_PATH, gt_I)
    print("Ground truth computed and cached.", flush=True)


def eval_recall(index, nprobe_val):
    faiss.cvar.indexPanorama_stats.reset()
    t0 = time.time()
    _, I = index.search(xq, k=k)
    t = time.time() - t0
    speed = t * 1000 / nq
    qps = 1000 / speed
    corrects = sum(len(set(gt_I[i]) & set(I[i])) for i in range(nq))
    recall = corrects / (nq * k)
    stats = faiss.cvar.indexPanorama_stats
    pct_active = stats.ratio_dims_scanned * 100
    print(
        f"\tnprobe {nprobe_val:3d}, Recall@{k}: "
        f"{recall:.6f}, speed: {speed:.6f} ms/query, QPS: {qps:.1f}, "
        f"active: {pct_active:.1f}%",
        flush=True,
    )
    return recall, qps


faiss.omp_set_num_threads(mp.cpu_count())

# # --- IVFPQ baseline (cached) ---
# if os.path.exists(IVFPQ_CACHE):
#     print(f"\nLoading cached IVFPQ from {IVFPQ_CACHE}...", flush=True)
#     t0 = time.time()
#     ivfpq = faiss.read_index(IVFPQ_CACHE)
#     print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)
# else:
#     print(f"\nBuilding IVFPQ: nlist={nlist}, M={M}, nbits={nbits}", flush=True)
#     quantizer = faiss.IndexFlatL2(d)
#     ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
#     t0 = time.time()
#     ivfpq.train(xt)
#     print(f"  Training took {time.time() - t0:.1f}s", flush=True)

#     print(f"  Saving trained state to {IVFPQ_TRAINED_CACHE}...", flush=True)
#     faiss.write_index(ivfpq, IVFPQ_TRAINED_CACHE)

#     t0 = time.time()
#     ivfpq.add(xb)
#     print(f"  Adding took {time.time() - t0:.1f}s", flush=True)

#     print(f"  Saving full index to {IVFPQ_CACHE}...", flush=True)
#     faiss.write_index(ivfpq, IVFPQ_CACHE)

# faiss.omp_set_num_threads(1)
# print("\n====== IVFPQ baseline", flush=True)
# for nprobe in [1, 2, 4, 8, 16]:
#     ivfpq.nprobe = nprobe
#     eval_recall(ivfpq, nprobe)

# --- IVFPQPanorama (cached) ---
faiss.omp_set_num_threads(mp.cpu_count())

if os.path.exists(IVFPQ_PANO_CACHE):
    print(f"\nLoading cached IVFPQPanorama from {IVFPQ_PANO_CACHE}...", flush=True)
    t0 = time.time()
    ivfpq_pano = faiss.read_index(IVFPQ_PANO_CACHE)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)
else:
    def build_panorama_from_trained(trained_index):
        quantizer2 = trained_index.quantizer
        trained_index.own_fields = False

        pano = faiss.IndexIVFPQPanorama(
            quantizer2, d, nlist, M, nbits, n_levels, batch_size
        )
        centroids = faiss.vector_to_array(trained_index.pq.centroids)
        faiss.copy_array_to_vector(centroids, pano.pq.centroids)
        pano.is_trained = True
        pano.use_precomputed_table = 1
        pano.precompute_table()
        return pano

    if os.path.exists(IVFPQ_TRAINED_CACHE):
        print(
            f"\nLoading trained IVFPQ for Panorama from {IVFPQ_TRAINED_CACHE}...",
            flush=True,
        )
        trained = faiss.read_index(IVFPQ_TRAINED_CACHE)
        ivfpq_pano = build_panorama_from_trained(trained)
        print("  Reused trained PQ (skipped training).", flush=True)
    else:
        print(
            f"\nTraining IVFPQ for Panorama from scratch: nlist={nlist}, M={M}, nbits={nbits}",
            flush=True,
        )
        quantizer2 = faiss.IndexFlatL2(d)
        trained = faiss.IndexIVFPQ(quantizer2, d, nlist, M, nbits)
        t0 = time.time()
        trained.train(xt)
        print(f"  Training took {time.time() - t0:.1f}s", flush=True)

        print(f"  Saving trained state to {IVFPQ_TRAINED_CACHE}...", flush=True)
        faiss.write_index(trained, IVFPQ_TRAINED_CACHE)

        ivfpq_pano = build_panorama_from_trained(trained)

    t0 = time.time()
    ivfpq_pano.add(xb)
    print(f"  Adding took {time.time() - t0:.1f}s", flush=True)

    print(f"  Saving IVFPQPanorama to {IVFPQ_PANO_CACHE}...", flush=True)
    faiss.write_index(ivfpq_pano, IVFPQ_PANO_CACHE)

faiss.omp_set_num_threads(1)
print("\n====== IVFPQPanorama", flush=True)
for nprobe in [1, 2, 4, 8, 16]:
    ivfpq_pano.nprobe = nprobe
    eval_recall(ivfpq_pano, nprobe)

print("\nVerification complete!", flush=True)
