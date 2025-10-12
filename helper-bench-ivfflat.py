import argparse
import time
import numpy as np
import faiss
import os

def read_fvecs(filename, limit=None):
    """Read fvecs format file."""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                break
            dim = int.from_bytes(dim_bytes, byteorder='little')
            
            vector_bytes = f.read(dim * 4)
            if len(vector_bytes) != dim * 4:
                break
            
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            vectors.append(vector)
            
            if limit is not None and len(vectors) >= limit:
                break
        
        return np.array(vectors, dtype=np.float32)

def benchmark_index(index, xb, xq, k, name):
    """Benchmark an index and return resu lts."""
    print(f"\n{'='*50}")
    print(f"Benchmarking {name}")
    print(f"{'='*50}")
    
    # Add data
    start = time.perf_counter()
    index.add(xb)
    add_time = time.perf_counter() - start
    print(f"Add time: {add_time:.3f}s")
    
    # Search
    start = time.perf_counter()
    D, I = index.search(xq, k)
    search_time = time.perf_counter() - start
    print(f"QPS: {len(xq)/search_time:.1f}")
    
    return D, I, search_time

def main():
    parser = argparse.ArgumentParser(description="Compare IVFFlat vs IVFFlatPanorama")
    parser.add_argument("--base-data", type=str, required=True, help="Path to base vectors (fvecs)")
    parser.add_argument("--query-data", type=str, required=True, help="Path to query vectors (fvecs)")
    parser.add_argument("--nb", type=int, default=None, help="Limit number of base vectors")
    parser.add_argument("--nq", type=int, default=1000, help="Limit number of query vectors")
    parser.add_argument("--nlist", type=int, default=10, help="Number of clusters")
    parser.add_argument("--nprobe", type=int, default=8, help="Number of clusters to search")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--nlevels", type=int, default=8, help="Panorama levels")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading base vectors from {args.base_data}...")
    xb = read_fvecs(args.base_data, args.nb)
    print(f"Loaded {len(xb)} base vectors, dimension {xb.shape[1]}")
    
    print(f"Loading query vectors from {args.query_data}...")
    xq = read_fvecs(args.query_data, args.nq)
    print(f"Loaded {len(xq)} query vectors")
    
    d = xb.shape[1]
    print(f"\nParameters: nlist={args.nlist}, nprobe={args.nprobe}, k={args.k}")
    
    # Ground truth
    print("\nComputing ground truth...")
    faiss.omp_set_num_threads(64)
    index_gt = faiss.IndexFlatL2(d)
    index_gt.add(xb)
    D_gt, I_gt = index_gt.search(xq, args.k)
    faiss.omp_set_num_threads(1)
    
    # IVFFlat
    quantizer1 = faiss.IndexFlatL2(d)
    index1 = faiss.IndexIVFFlat(quantizer1, d, args.nlist)
    index1.train(xb)
    index1.nprobe = args.nprobe
    D1, I1, search_time1 = benchmark_index(index1, xb, xq, args.k, "IVFFlat")
    
    # IVFFlatPanorama
    quantizer2 = faiss.IndexFlatL2(d)
    index2 = faiss.IndexIVFFlatPanorama(quantizer2, d, args.nlist, args.nlevels)
    index2.train(xb)
    index2.nprobe = args.nprobe
    D2, I2, search_time2 = benchmark_index(index2, xb, xq, args.k, "IVFFlatPanorama")
    
    # Calculate recalls
    recall1 = np.mean([len(set(I_gt[i]) & set(I1[i])) / args.k for i in range(len(I_gt))])
    recall2 = np.mean([len(set(I_gt[i]) & set(I2[i])) / args.k for i in range(len(I_gt))])
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"\nIVFFlat:")
    print(f"  Search time: {search_time1:.3f}s")
    print(f"  QPS:         {len(xq)/search_time1:.1f}")
    print(f"  Recall:      {recall1:.4f}")
    
    print(f"\nIVFFlatPanorama:")
    print(f"  Search time: {search_time2:.3f}s")
    print(f"  QPS:         {len(xq)/search_time2:.1f}")
    print(f"  Recall:      {recall2:.4f}")
    
    print(f"\nSpeedup:")
    print(f"  Search: {search_time1/search_time2:.2f}x")
    print(f"  QPS:    {(len(xq)/search_time2)/(len(xq)/search_time1):.2f}x")

if __name__ == "__main__":
    main()