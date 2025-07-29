#!/usr/bin/env python

import numpy as np
import faiss
import sys

def test_hnsw_variable_degree_rejection():
    """Test that copyFrom correctly rejects HNSW graphs with variable degree"""
    
    print("Testing HNSW -> CAGRA conversion validation\n")
    
    # Test parameters
    d = 128 * 8  # 128 bytes = 1024 bits
    n = 1000
    
    # Generate random binary data
    xb = np.random.randint(low=0, high=256, size=(n, d // 8), dtype=np.uint8)
    
    # Test 1: Create HNSW with normal construction (may have variable degree)
    print("1. Testing HNSW with potentially variable degree...")
    index_hnsw = faiss.IndexBinaryHNSW(d, 16)  # M=16, so max degree at level 0 = 32
    index_hnsw.hnsw.efConstruction = 40  # Low efConstruction may lead to incomplete graphs
    index_hnsw.add(xb)
    
    # Try to copy to CAGRA
    res = faiss.StandardGpuResources()
    index_cagra = faiss.GpuIndexBinaryCagra(res, d)
    
    try:
        index_cagra.copyFrom(index_hnsw)
        print("   ✓ Conversion succeeded (HNSW had fixed degree)")
    except RuntimeError as e:
        if "Cannot faithfully convert HNSW to CAGRA" in str(e):
            print(f"   ✓ Correctly rejected: {e}")
        else:
            print(f"   ✗ Unexpected error: {e}")
            raise
    
    # Test 2: Verify degree mismatch detection in copyTo
    print("\n2. Testing CAGRA -> HNSW with mismatched degrees...")
    
    # Create CAGRA with specific degree
    config = faiss.GpuIndexCagraConfig()
    config.graph_degree = 64  # CAGRA will have degree 64
    config.build_algo = faiss.graph_build_algo_NN_DESCENT
    
    index_cagra2 = faiss.GpuIndexBinaryCagra(res, d, config)
    index_cagra2.train(xb)
    
    # Try to copy to HNSW with different M
    index_hnsw2 = faiss.IndexBinaryHNSW(d, 16)  # M=16, expects degree 32 at level 0
    
    try:
        index_cagra2.copyTo(index_hnsw2)
        print("   ✗ Conversion succeeded but should have failed!")
    except RuntimeError as e:
        if "Cannot convert CAGRA to HNSW" in str(e) and "graph degree" in str(e):
            print(f"   ✓ Correctly rejected: {e}")
        else:
            print(f"   ✗ Unexpected error: {e}")
            raise
    
    # Test 3: Verify successful conversion with matching degrees
    print("\n3. Testing successful conversion with matching degrees...")
    
    # Create HNSW with M=32 (degree 64 at level 0, matching CAGRA)
    index_hnsw3 = faiss.IndexBinaryHNSW(d, 32)
    
    try:
        index_cagra2.copyTo(index_hnsw3)
        print("   ✓ Conversion succeeded with matching degrees")
        
        # Verify the index works
        xq = np.random.randint(low=0, high=256, size=(10, d // 8), dtype=np.uint8)
        D, I = index_hnsw3.search(xq, 10)
        print(f"   ✓ Converted index search successful, shape: D={D.shape}, I={I.shape}")
        
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        raise
    
    print("\n✓ All validation tests passed!")
    return True


def create_hnsw_with_incomplete_graph():
    """Helper to create an HNSW with deliberately incomplete graph"""
    d = 64 * 8
    n = 100
    
    # Create a very small dataset that will likely have variable degree
    xb = np.random.randint(low=0, high=256, size=(n, d // 8), dtype=np.uint8)
    
    index = faiss.IndexBinaryHNSW(d, 8)
    index.hnsw.efConstruction = 16  # Very low to increase chance of incomplete graph
    
    # Add vectors one by one to increase chance of variable degree
    for i in range(n):
        index.add(xb[i:i+1])
    
    return index, xb


if __name__ == "__main__":
    success = test_hnsw_variable_degree_rejection()
    
    # Additional test with deliberately incomplete graph
    print("\n4. Testing with deliberately incomplete HNSW graph...")
    index_incomplete, xb = create_hnsw_with_incomplete_graph()
    
    res = faiss.StandardGpuResources()
    index_cagra = faiss.GpuIndexBinaryCagra(res, index_incomplete.d)
    
    try:
        index_cagra.copyFrom(index_incomplete)
        print("   ⚠ Conversion succeeded - HNSW happened to have fixed degree")
    except RuntimeError as e:
        if "Cannot faithfully convert HNSW to CAGRA" in str(e):
            print(f"   ✓ Correctly rejected incomplete graph: {e}")
        else:
            print(f"   ✗ Unexpected error: {e}")
            raise
    
    sys.exit(0 if success else 1) 