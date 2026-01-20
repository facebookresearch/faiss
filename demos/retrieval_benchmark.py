import faiss
import numpy as np
import time
import json
import os

def run_benchmark():
    # Configuration
    d = 128          # dimension
    nb = 10000       # database size
    nq = 1000        # nb of queries
    nlist = 100      # num of clusters for IVF
    M = 32           # HNSW params
    
    # Generate data
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    results = {
        "dataset": {
            "d": d,
            "nb": nb,
            "nq": nq
        },
        "benchmarks": []
    }
    
    print(f"Generating data: d={d}, nb={nb}, nq={nq}")

    # 1. IndexFlatL2 (Exact Search)
    print("\nBenchmarking IndexFlatL2 (Exact Search)...")
    start_time = time.time()
    index_l2 = faiss.IndexFlatL2(d)
    index_l2.add(xb)
    build_time = time.time() - start_time
    
    start_time = time.time()
    D, I = index_l2.search(xq, k=5)
    search_time = time.time() - start_time
    
    results["benchmarks"].append({
        "index_type": "IndexFlatL2",
        "build_time_sec": round(build_time, 4),
        "search_time_sec": round(search_time, 4),
        "notes": "Baseline exact search"
    })
    
    # Ground truth for recall calculation using FlatL2
    gt_I = I

    # 2. IndexIVFFlat (Inverted File)
    print("Benchmarking IndexIVFFlat...")
    quantizer = faiss.IndexFlatL2(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    start_time = time.time()
    index_ivf.train(xb)
    index_ivf.add(xb)
    build_time = time.time() - start_time
    
    start_time = time.time()
    index_ivf.nprobe = 10  # default nprobe
    D_ivf, I_ivf = index_ivf.search(xq, k=5)
    search_time = time.time() - start_time

    # Calculate Recall@5
    recall = (I_ivf == gt_I).sum() / I_ivf.size
    
    results["benchmarks"].append({
        "index_type": "IndexIVFFlat",
        "build_time_sec": round(build_time, 4),
        "search_time_sec": round(search_time, 4),
        "recall_at_5": round(recall, 4),
        "params": {"nlist": nlist, "nprobe": 10}
    })

    # 3. IndexHNSWFlat (Graph-based)
    print("Benchmarking IndexHNSWFlat...")
    start_time = time.time()
    index_hnsw = faiss.IndexHNSWFlat(d, M)
    index_hnsw.add(xb)
    build_time = time.time() - start_time
    
    start_time = time.time()
    D_hnsw, I_hnsw = index_hnsw.search(xq, k=5)
    search_time = time.time() - start_time
    
    # Calculate Recall@5
    recall_hnsw = (I_hnsw == gt_I).sum() / I_hnsw.size

    results["benchmarks"].append({
        "index_type": "IndexHNSWFlat",
        "build_time_sec": round(build_time, 4),
        "search_time_sec": round(search_time, 4),
        "recall_at_5": round(recall_hnsw, 4),
        "params": {"M": M}
    })
    
    # Save results
    output_file = "metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nBenchmark completed. Results saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_benchmark()
