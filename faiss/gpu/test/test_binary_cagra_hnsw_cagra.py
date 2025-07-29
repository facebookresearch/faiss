#!/usr/bin/env python

import numpy as np
import faiss
import sys

def test_binary_hnsw_cagra_interop():
    """Test interoperability between GpuIndexBinaryCagra and IndexBinaryHNSWCagra"""
    
    print("Testing Binary CAGRA <-> HNSWCagra Interoperability\n")
    
    # Test parameters
    d = 128 * 8  # 128 bytes = 1024 bits
    n_train = 10000
    n_query = 100
    k = 32
    
    # Generate random binary data
    print(f"1. Generating {n_train} random binary vectors of dimension {d} bits...")
    xb = np.random.randint(low=0, high=256, size=(n_train, d // 8), dtype=np.uint8)
    xq = np.random.randint(low=0, high=256, size=(n_query, d // 8), dtype=np.uint8)
    
    # Test 1: CAGRA -> HNSWCagra
    print("\n2. Test CAGRA -> HNSWCagra")
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
    
    print("   Creating IndexBinaryHNSWCagra...")
    # M should be graph_degree / 2 = 32
    index_hnsw_cagra = faiss.IndexBinaryHNSWCagra(d, 32)
    
    print("   Copying from CAGRA to HNSWCagra...")
    try:
        index_cagra.copyTo(index_hnsw_cagra)
        print("   ✓ copyTo completed successfully!")
        
        # Verify the index
        assert index_hnsw_cagra.ntotal == n_train
        assert index_hnsw_cagra.d == d
        print(f"   ✓ Index has correct size: {index_hnsw_cagra.ntotal} vectors")
        
        # Verify fixed degree
        assert index_hnsw_cagra.has_fixed_degree()
        fixed_degree = index_hnsw_cagra.get_fixed_degree()
        print(f"   ✓ Index has fixed degree: {fixed_degree}")
        
        # Search with HNSWCagra
        print("   Searching with HNSWCagra...")
        D_hnsw, I_hnsw = index_hnsw_cagra.search(xq, k)
        
        # Calculate recall
        recall = calculate_recall(I_cagra, I_hnsw, k)
        print(f"   Recall@{k}: {recall:.4f}")
        
        # With base_level_only, we expect perfect recall since it's the same graph
        if recall < 0.95:
            print(f"   ⚠ Warning: Lower than expected recall ({recall:.4f})")
        else:
            print(f"   ✓ Excellent recall ({recall:.4f})")
            
    except Exception as e:
        print(f"   ✗ ERROR during copyTo: {e}")
        raise
    
    # Test 2: HNSWCagra -> CAGRA
    print("\n3. Test HNSWCagra -> CAGRA")
    print("   Creating IndexBinaryHNSWCagra with fixed degree...")
    index_hnsw2 = faiss.IndexBinaryHNSWCagra(d, 32)
    index_hnsw2.add(xb)
    
    # Ensure fixed degree
    index_hnsw2.ensure_fixed_degree()
    assert index_hnsw2.has_fixed_degree()
    
    print("   Creating new GpuIndexBinaryCagra...")
    index_cagra2 = faiss.GpuIndexBinaryCagra(res, d)
    
    print("   Copying from HNSWCagra to CAGRA...")
    try:
        index_cagra2.copyFrom(index_hnsw2)
        print("   ✓ copyFrom completed successfully!")
        
        # Search with copied CAGRA
        D_cagra2, I_cagra2 = index_cagra2.search(xq, k)
        
        # Search with original HNSWCagra
        D_hnsw2, I_hnsw2 = index_hnsw2.search(xq, k)
        
        # Calculate recall
        recall = calculate_recall(I_hnsw2, I_cagra2, k)
        print(f"   Recall@{k}: {recall:.4f}")
        
        if recall < 0.90:
            print(f"   ⚠ Warning: Lower than expected recall ({recall:.4f})")
        else:
            print(f"   ✓ Good recall ({recall:.4f})")
            
    except Exception as e:
        print(f"   ✗ ERROR during copyFrom: {e}")
        raise
    
    # Test 3: Round-trip with base_level_only search
    print("\n4. Test base_level_only search mode")
    index_hnsw_base = faiss.IndexBinaryHNSWCagra(d, 32)
    index_cagra.copyTo(index_hnsw_base)
    
    # Enable base_level_only for search
    index_hnsw_base.base_level_only = True
    
    print("   Searching with base_level_only mode...")
    D_base, I_base = index_hnsw_base.search(xq, k)
    
    # Compare with original CAGRA results
    recall_base = calculate_recall(I_cagra, I_base, k)
    print(f"   Recall@{k} (base level only): {recall_base:.4f}")
    
    if recall_base > 0.98:
        print(f"   ✓ Near-perfect recall with base_level_only search")
    
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
    success = test_binary_hnsw_cagra_interop()
    sys.exit(0 if success else 1) 