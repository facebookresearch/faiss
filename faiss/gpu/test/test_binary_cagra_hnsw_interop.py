#!/usr/bin/env python

import numpy as np
import faiss
import sys

def test_binary_cagra_hnsw_interop():
    """Comprehensive test for binary CAGRA and HNSW interoperability"""
    
    print("Testing Binary CAGRA <-> HNSW Interoperability\n")
    
    # Test parameters
    d = 128 * 8  # 128 bytes = 1024 bits
    n_train = 10000
    n_query = 100
    k = 32
    
    # Generate random binary data
    print(f"1. Generating {n_train} random binary vectors of dimension {d} bits...")
    xb = np.random.randint(low=0, high=256, size=(n_train, d // 8), dtype=np.uint8)
    xq = np.random.randint(low=0, high=256, size=(n_query, d // 8), dtype=np.uint8)
    
    # Test 1: CAGRA -> HNSW (copyTo)
    print("\n2. Test CAGRA -> HNSW (copyTo)")
    print("   Creating and training GpuIndexBinaryCagra...")
    res = faiss.StandardGpuResources()
    
    config = faiss.GpuIndexCagraConfig()
    config.graph_degree = 64
    config.intermediate_graph_degree = 128
    config.build_algo = faiss.graph_build_algo_NN_DESCENT
    config.nn_descent_niter = 20
    
    index_cagra = faiss.GpuIndexBinaryCagra(res, d, config)
    index_cagra.train(xb)
    
    print("   Searching with CAGRA...")
    D_cagra, I_cagra = index_cagra.search(xq, k)
    
    print("   Creating IndexBinaryHNSW and copying from CAGRA...")
    index_hnsw = faiss.IndexBinaryHNSW(d, 32)
    
    try:
        index_cagra.copyTo(index_hnsw)
        print("   ✓ copyTo completed successfully!")
        
        # Verify the copy
        print("   Verifying copied index...")
        assert index_hnsw.ntotal == n_train, f"Expected {n_train} vectors, got {index_hnsw.ntotal}"
        assert index_hnsw.d == d, f"Expected dimension {d}, got {index_hnsw.d}"
        
        # Search with copied HNSW
        print("   Searching with copied HNSW...")
        D_hnsw_copy, I_hnsw_copy = index_hnsw.search(xq, k)
        
        # Calculate recall
        recall = calculate_recall(I_cagra, I_hnsw_copy, k)
        print(f"   Recall@{k}: {recall:.4f}")
        
        if recall < 0.7:
            print(f"   ⚠ Warning: Low recall ({recall:.4f}), may indicate graph structure issues")
        else:
            print(f"   ✓ Good recall ({recall:.4f})")
            
    except Exception as e:
        print(f"   ✗ ERROR during copyTo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: HNSW -> CAGRA (copyFrom)
    print("\n3. Test HNSW -> CAGRA (copyFrom)")
    print("   Creating and training IndexBinaryHNSW...")
    index_hnsw2 = faiss.IndexBinaryHNSW(d, 32)
    index_hnsw2.hnsw.efConstruction = 200
    index_hnsw2.add(xb)
    
    print("   Searching with HNSW...")
    D_hnsw2, I_hnsw2 = index_hnsw2.search(xq, k)
    
    print("   Creating GpuIndexBinaryCagra and copying from HNSW...")
    index_cagra2 = faiss.GpuIndexBinaryCagra(res, d)
    
    try:
        index_cagra2.copyFrom(index_hnsw2)
        print("   ✓ copyFrom completed successfully!")
        
        # Verify the copy
        print("   Verifying copied index...")
        assert index_cagra2.ntotal == n_train, f"Expected {n_train} vectors, got {index_cagra2.ntotal}"
        assert index_cagra2.d == d, f"Expected dimension {d}, got {index_cagra2.d}"
        
        # Search with copied CAGRA
        print("   Searching with copied CAGRA...")
        D_cagra_copy, I_cagra_copy = index_cagra2.search(xq, k)
        
        # Calculate recall
        recall = calculate_recall(I_hnsw2, I_cagra_copy, k)
        print(f"   Recall@{k}: {recall:.4f}")
        
        if recall < 0.7:
            print(f"   ⚠ Warning: Low recall ({recall:.4f}), may indicate graph structure issues")
        else:
            print(f"   ✓ Good recall ({recall:.4f})")
            
    except Exception as e:
        print(f"   ✗ ERROR during copyFrom: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Round-trip conversion
    print("\n4. Test Round-trip conversion (CAGRA -> HNSW -> CAGRA)")
    try:
        # CAGRA -> HNSW
        index_hnsw_rt = faiss.IndexBinaryHNSW(d, 32)
        index_cagra.copyTo(index_hnsw_rt)
        
        # HNSW -> CAGRA
        index_cagra_rt = faiss.GpuIndexBinaryCagra(res, d)
        index_cagra_rt.copyFrom(index_hnsw_rt)
        
        # Search with round-trip index
        D_rt, I_rt = index_cagra_rt.search(xq, k)
        
        # Compare with original
        recall = calculate_recall(I_cagra, I_rt, k)
        print(f"   Round-trip recall@{k}: {recall:.4f}")
        
        if recall < 0.9:
            print(f"   ⚠ Warning: Significant information loss in round-trip ({recall:.4f})")
        else:
            print(f"   ✓ Good round-trip fidelity ({recall:.4f})")
            
    except Exception as e:
        print(f"   ✗ ERROR during round-trip: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests completed successfully!")
    return True


def calculate_recall(I_ref, I_test, k):
    """Calculate recall@k between two sets of search results"""
    n_queries = I_ref.shape[0]
    recall_sum = 0
    
    for i in range(n_queries):
        ref_set = set(I_ref[i])
        test_set = set(I_test[i])
        recall_sum += len(ref_set.intersection(test_set)) / k
    
    return recall_sum / n_queries


if __name__ == "__main__":
    success = test_binary_cagra_hnsw_interop()
    sys.exit(0 if success else 1) 