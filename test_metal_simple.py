#!/usr/bin/env python3
"""Simple test to verify Metal backend implementation"""

import numpy as np
import faiss
import time


def test_metal_indexflat():
    """Test MetalIndexFlat if available"""
    d = 128  # dimension
    nb = 1000  # database size
    nq = 10  # number of queries

    # Generate random data
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    print("Testing CPU IndexFlat...")
    # Create CPU index
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(xb)

    # Search
    k = 4
    start = time.time()
    D_cpu, I_cpu = cpu_index.search(xq, k)
    cpu_time = time.time() - start
    print(f"CPU search time: {cpu_time:.4f}s")

    # Verify results
    print(f"CPU results shape: D={D_cpu.shape}, I={I_cpu.shape}")
    print(f"First query neighbors: {I_cpu[0]}")
    print(f"First query distances: {D_cpu[0]}")

    # Try to test Metal if available
    try:
        # Check if Metal backend is available
        if hasattr(faiss, "MetalIndexFlat"):
            print("\nTesting Metal IndexFlat...")
            metal_index = faiss.MetalIndexFlat(d)
            metal_index.add(xb)

            start = time.time()
            D_metal, I_metal = metal_index.search(xq, k)
            metal_time = time.time() - start
            print(f"Metal search time: {metal_time:.4f}s")

            # Compare results
            print(f"Metal results shape: D={D_metal.shape}, I={I_metal.shape}")
            print(f"First query neighbors: {I_metal[0]}")
            print(f"First query distances: {D_metal[0]}")

            # Check if results are similar
            common_neighbors = 0
            for i in range(nq):
                for j in range(k):
                    if I_cpu[i, j] in I_metal[i]:
                        common_neighbors += 1

            overlap = common_neighbors / (nq * k)
            print(f"\nResult overlap: {overlap:.2%}")

            if overlap > 0.9:
                print("✓ Metal implementation produces similar results to CPU")
            else:
                print("✗ Metal implementation results differ significantly from CPU")
        else:
            print("\n✗ Metal backend not available in this build")

    except Exception as e:
        print(f"\n✗ Error testing Metal backend: {e}")


def test_metal_indexhnsw():
    """Test MetalIndexHNSW if available"""
    d = 64  # dimension
    nb = 500  # database size
    nq = 5  # number of queries
    M = 16  # HNSW parameter

    # Generate random data
    np.random.seed(5678)
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    print("\n\nTesting CPU IndexHNSW...")
    # Create CPU index
    cpu_index = faiss.IndexHNSWFlat(d, M)
    cpu_index.add(xb)

    # Search
    k = 10
    start = time.time()
    D_cpu, I_cpu = cpu_index.search(xq, k)
    cpu_time = time.time() - start
    print(f"CPU HNSW search time: {cpu_time:.4f}s")

    # Verify results
    print(f"CPU results shape: D={D_cpu.shape}, I={I_cpu.shape}")
    print(f"First query neighbors: {I_cpu[0]}")

    # Try to test Metal if available
    try:
        if hasattr(faiss, "MetalIndexHNSW"):
            print("\nTesting Metal IndexHNSW...")
            metal_index = faiss.MetalIndexHNSW(d, M)
            metal_index.add(xb)

            start = time.time()
            D_metal, I_metal = metal_index.search(xq, k)
            metal_time = time.time() - start
            print(f"Metal HNSW search time: {metal_time:.4f}s")

            # Compare results
            print(f"Metal results shape: D={D_metal.shape}, I={I_metal.shape}")
            print(f"First query neighbors: {I_metal[0]}")

            # Check overlap (HNSW is approximate, so we expect some variation)
            common_neighbors = 0
            for i in range(nq):
                for j in range(k):
                    if I_cpu[i, j] in I_metal[i]:
                        common_neighbors += 1

            overlap = common_neighbors / (nq * k)
            print(f"\nResult overlap: {overlap:.2%}")

            if overlap > 0.5:  # Lower threshold for approximate search
                print("✓ Metal HNSW implementation produces reasonable results")
            else:
                print("✗ Metal HNSW implementation results differ too much from CPU")
        else:
            print("\n✗ Metal HNSW backend not available in this build")

    except Exception as e:
        print(f"\n✗ Error testing Metal HNSW backend: {e}")


if __name__ == "__main__":
    print("=== Faiss Metal Backend Test ===")
    test_metal_indexflat()
    test_metal_indexhnsw()
    print("\n=== Test Complete ===")
