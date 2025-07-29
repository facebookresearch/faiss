#!/usr/bin/env python

import numpy as np
import faiss

print("Starting binary CAGRA to HNSW debug test...")

# Test parameters
d = 128 * 8  # 128 bytes = 1024 bits
n = 1000     # Small dataset for debugging
k = 32
nq = 10

print(f"Parameters: d={d}, n={n}, k={k}, nq={nq}")

# Create GPU resources
res = faiss.StandardGpuResources()

# Create and train GpuIndexBinaryCagra
print("\n1. Creating GpuIndexBinaryCagra...")
index_gpu = faiss.GpuIndexBinaryCagra(res, d)

# Generate random binary data
print(f"2. Generating {n} random binary vectors...")
xb = np.random.randint(low=0, high=256, size=(n, d // 8), dtype=np.uint8)

print("3. Training GpuIndexBinaryCagra...")
index_gpu.train(xb)
print("   Training completed successfully")

# Generate query vectors
xq = np.random.randint(low=0, high=256, size=(nq, d // 8), dtype=np.uint8)

# Search with GPU index
print("\n4. Searching with GPU index...")
D_gpu, I_gpu = index_gpu.search(xq, k)
print(f"   Search completed. Shape: D={D_gpu.shape}, I={I_gpu.shape}")

# Test copyTo HNSW
print("\n5. Testing copyTo from GpuIndexBinaryCagra to IndexBinaryHNSW...")
print("   Creating empty IndexBinaryHNSW...")
index_hnsw = faiss.IndexBinaryHNSW(d, 32)  # M=32

print("   Calling index_gpu.copyTo(index_hnsw)...")
try:
    index_gpu.copyTo(index_hnsw)
    print("   copyTo completed successfully!")
    
    # Test search on copied index
    print("\n6. Testing search on copied HNSW index...")
    D_hnsw, I_hnsw = index_hnsw.search(xq, k)
    print(f"   Search completed. Shape: D={D_hnsw.shape}, I={I_hnsw.shape}")
    
    # Calculate recall
    print("\n7. Calculating recall...")
    recall_sum = 0
    for i in range(nq):
        gpu_set = set(I_gpu[i])
        hnsw_set = set(I_hnsw[i])
        recall_sum += len(gpu_set.intersection(hnsw_set)) / k
    recall = recall_sum / nq
    print(f"   Recall@{k}: {recall:.4f}")
    
except Exception as e:
    print(f"   ERROR during copyTo: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug test completed.") 