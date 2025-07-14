#!/usr/bin/env python3
"""Test Metal backend functionality from Python."""

import numpy as np
import time
import sys

# Try to import faiss
try:
    import faiss
except ImportError:
    print("Error: Could not import faiss. Make sure it's installed.")
    print("Try: pip install faiss-cpu or build from source")
    sys.exit(1)


def test_metal_availability():
    """Test if Metal backend is available."""
    print("Testing Metal availability...")

    # Check if Metal is enabled in the build
    if not hasattr(faiss, "MetalIndexFlat"):
        print("✗ Metal support not compiled into Faiss")
        return False

    try:
        # Try to create a simple Metal index
        index = faiss.MetalIndexFlat(64, faiss.METRIC_L2)
        print("✓ Metal support is available")
        return True
    except Exception as e:
        print(f"✗ Metal initialization failed: {e}")
        return False


def benchmark_metal_vs_cpu():
    """Benchmark Metal vs CPU performance."""
    print("\nBenchmarking Metal vs CPU...")

    # Parameters
    d = 128  # dimension
    nb = 10000  # database size
    nq = 1000  # number of queries
    k = 10  # k nearest neighbors

    # Generate random data
    np.random.seed(42)
    database = np.random.randn(nb, d).astype("float32")
    queries = np.random.randn(nq, d).astype("float32")

    # CPU Index
    print(f"\nCreating CPU index (d={d}, nb={nb})...")
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(database)

    # CPU search
    start = time.time()
    cpu_distances, cpu_labels = cpu_index.search(queries, k)
    cpu_time = time.time() - start
    print(f"CPU search time: {cpu_time*1000:.2f} ms ({nq/cpu_time:.0f} queries/sec)")

    # Metal Index
    if hasattr(faiss, "MetalIndexFlat"):
        print(f"\nCreating Metal index (d={d}, nb={nb})...")
        metal_index = faiss.MetalIndexFlat(d, faiss.METRIC_L2)
        metal_index.add(database)

        # Metal search
        start = time.time()
        metal_distances, metal_labels = metal_index.search(queries, k)
        metal_time = time.time() - start
        print(
            f"Metal search time: {metal_time*1000:.2f} ms ({nq/metal_time:.0f} queries/sec)"
        )

        # Verify accuracy
        accuracy = np.mean(cpu_labels == metal_labels) * 100
        print(f"\nAccuracy: {accuracy:.1f}%")

        # Calculate speedup
        speedup = cpu_time / metal_time
        print(f"Speedup: {speedup:.2f}x")

        # Show sample results
        print("\nSample results (first query):")
        print(f"CPU:   {cpu_labels[0][:5]} (distances: {cpu_distances[0][:5]})")
        print(f"Metal: {metal_labels[0][:5]} (distances: {metal_distances[0][:5]})")


def test_metal_hnsw():
    """Test Metal HNSW implementation."""
    print("\n\nTesting Metal HNSW...")

    if not hasattr(faiss, "MetalIndexHNSW"):
        print("✗ MetalIndexHNSW not available")
        return

    # Parameters
    d = 128
    nb = 1000
    nq = 10
    k = 5
    M = 32

    # Generate data
    np.random.seed(42)
    database = np.random.randn(nb, d).astype("float32")
    queries = np.random.randn(nq, d).astype("float32")

    # Create CPU HNSW
    print("Creating CPU HNSW index...")
    cpu_quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexHNSW(cpu_quantizer, d, M)
    cpu_index.hnsw.efConstruction = 200
    cpu_index.hnsw.efSearch = 64
    cpu_index.add(database)

    # Create Metal HNSW
    print("Creating Metal HNSW index...")
    metal_index = faiss.MetalIndexHNSW(d, M, faiss.METRIC_L2)
    metal_index.hnsw.efConstruction = 200
    metal_index.hnsw.efSearch = 64
    metal_index.add(database)

    # Search
    print("Searching...")
    cpu_distances, cpu_labels = cpu_index.search(queries, k)
    metal_distances, metal_labels = metal_index.search(queries, k)

    # Verify accuracy
    accuracy = np.mean(cpu_labels == metal_labels) * 100
    print(f"Accuracy: {accuracy:.1f}%")

    if accuracy >= 90:
        print("✓ Metal HNSW test PASSED")
    else:
        print("✗ Metal HNSW test FAILED")


def main():
    """Main test function."""
    print("=== Faiss Metal Backend Tests ===\n")

    # Test Metal availability
    if not test_metal_availability():
        print("\nMetal backend not available. Exiting.")
        return

    # Run benchmarks
    benchmark_metal_vs_cpu()

    # Test HNSW
    test_metal_hnsw()

    print("\n=== Tests completed ===")


if __name__ == "__main__":
    main()
