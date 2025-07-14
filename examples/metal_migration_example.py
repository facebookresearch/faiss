#!/usr/bin/env python3
"""
Example showing how to migrate from CPU to Metal GPU indexes in Faiss.
This script demonstrates side-by-side comparison of CPU vs Metal performance.
"""

import numpy as np
import time
import sys

try:
    import faiss
except ImportError:
    print("Error: faiss not found. Please install with: pip install faiss-cpu")
    sys.exit(1)


def print_separator():
    print("-" * 60)


def format_time(seconds):
    """Format time in appropriate units"""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


class CPUSearch:
    """Traditional CPU-based Faiss search"""

    def __init__(self, d, metric="L2"):
        self.d = d
        self.metric = metric
        if metric == "L2":
            self.index = faiss.IndexFlatL2(d)
        else:
            self.index = faiss.IndexFlatIP(d)
        self.name = "CPU"

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, queries, k):
        return self.index.search(queries, k)


class MetalSearch:
    """Metal GPU-accelerated Faiss search"""

    def __init__(self, d, metric="L2"):
        self.d = d
        self.metric = metric
        try:
            self.res = faiss.StandardMetalResources()
            if metric == "L2":
                self.index = faiss.MetalIndexFlatL2(self.res, d)
            else:
                self.index = faiss.MetalIndexFlatIP(self.res, d)
            self.name = "Metal"
            self.available = True
        except AttributeError:
            print("Warning: Metal support not available in this Faiss build")
            self.available = False
            # Fallback to CPU
            if metric == "L2":
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = faiss.IndexFlatIP(d)
            self.name = "CPU (Metal unavailable)"

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, queries, k):
        return self.index.search(queries, k)


def benchmark_index(index_class, dimension, nb, nq, k, metric="L2"):
    """Benchmark an index class"""
    print(f"\nBenchmarking {index_class.__name__} ({metric} distance)...")

    # Generate random data
    np.random.seed(1234)
    database = np.random.random((nb, dimension)).astype("float32")
    queries = np.random.random((nq, dimension)).astype("float32")

    # Normalize for inner product
    if metric == "IP":
        faiss.normalize_L2(database)
        faiss.normalize_L2(queries)

    # Create index
    index = index_class(dimension, metric)

    # Time adding vectors
    start_time = time.time()
    index.add(database)
    add_time = time.time() - start_time
    print(f"  Add time: {format_time(add_time)} for {nb:,} vectors")

    # Warmup search
    index.search(queries[:10], k)

    # Time search
    start_time = time.time()
    distances, indices = index.search(queries, k)
    search_time = time.time() - start_time

    queries_per_second = nq / search_time
    print(f"  Search time: {format_time(search_time)} for {nq:,} queries")
    print(f"  Throughput: {queries_per_second:,.0f} queries/second")

    return {
        "name": index.name,
        "add_time": add_time,
        "search_time": search_time,
        "qps": queries_per_second,
        "distances": distances,
        "indices": indices,
    }


def verify_results(cpu_results, metal_results, tolerance=1e-4):
    """Verify that CPU and Metal results match"""
    print("\nVerifying results match...")

    # Check indices match
    indices_match = np.allclose(cpu_results["indices"], metal_results["indices"])

    # Check distances are close (some floating point differences expected)
    distances_close = np.allclose(
        cpu_results["distances"], metal_results["distances"], rtol=tolerance
    )

    if indices_match and distances_close:
        print("✓ Results match between CPU and Metal!")
    else:
        print("✗ Results differ between CPU and Metal")
        if not indices_match:
            print("  - Indices don't match")
        if not distances_close:
            print("  - Distances differ beyond tolerance")

    return indices_match and distances_close


def migration_example():
    """Show a complete migration example"""
    print("\n" + "=" * 60)
    print("MIGRATION EXAMPLE: CPU to Metal")
    print("=" * 60)

    # Original CPU code
    print("\n1. Original CPU code:")
    print("-" * 30)
    print(
        """
# Original implementation
import faiss
import numpy as np

d = 128
index = faiss.IndexFlatL2(d)

# Add vectors
vectors = np.random.random((10000, d)).astype('float32')
index.add(vectors)

# Search
queries = np.random.random((100, d)).astype('float32')
D, I = index.search(queries, k=10)
    """
    )

    # Metal version
    print("\n2. Migrated Metal code:")
    print("-" * 30)
    print(
        """
# Metal implementation
import faiss
import numpy as np

d = 128
res = faiss.StandardMetalResources()  # Add this line
index = faiss.MetalIndexFlatL2(res, d)  # Change index type

# Add vectors (unchanged)
vectors = np.random.random((10000, d)).astype('float32')
index.add(vectors)

# Search (unchanged)
queries = np.random.random((100, d)).astype('float32')
D, I = index.search(queries, k=10)
    """
    )

    print("\n3. Key changes:")
    print("-" * 30)
    print("   • Add: res = faiss.StandardMetalResources()")
    print("   • Change: IndexFlatL2 → MetalIndexFlatL2")
    print("   • Pass 'res' as first parameter to Metal index")
    print("   • Everything else remains the same!")


def main():
    """Main benchmark and migration demo"""
    print("Faiss CPU to Metal Migration Demo")
    print("=" * 60)

    # Parameters
    dimension = 128
    nb = 100000  # Database size
    nq = 1000  # Number of queries
    k = 10  # Number of neighbors

    print(f"\nConfiguration:")
    print(f"  Dimension: {dimension}")
    print(f"  Database size: {nb:,}")
    print(f"  Query count: {nq:,}")
    print(f"  k (neighbors): {k}")

    # Run benchmarks
    print_separator()

    # L2 distance
    cpu_results_l2 = benchmark_index(CPUSearch, dimension, nb, nq, k, "L2")
    metal_results_l2 = benchmark_index(MetalSearch, dimension, nb, nq, k, "L2")

    # Inner product
    cpu_results_ip = benchmark_index(CPUSearch, dimension, nb, nq, k, "IP")
    metal_results_ip = benchmark_index(MetalSearch, dimension, nb, nq, k, "IP")

    # Show speedup
    print_separator()
    print("\nPerformance Comparison:")

    if metal_results_l2["name"] == "Metal":
        l2_speedup = cpu_results_l2["search_time"] / metal_results_l2["search_time"]
        ip_speedup = cpu_results_ip["search_time"] / metal_results_ip["search_time"]

        print(f"  L2 Distance:")
        print(f"    CPU: {cpu_results_l2['qps']:,.0f} queries/sec")
        print(f"    Metal: {metal_results_l2['qps']:,.0f} queries/sec")
        print(f"    Speedup: {l2_speedup:.2f}x")

        print(f"  Inner Product:")
        print(f"    CPU: {cpu_results_ip['qps']:,.0f} queries/sec")
        print(f"    Metal: {metal_results_ip['qps']:,.0f} queries/sec")
        print(f"    Speedup: {ip_speedup:.2f}x")

        # Verify results
        verify_results(cpu_results_l2, metal_results_l2)
    else:
        print("  Metal support not available - cannot compare performance")

    # Show migration example
    migration_example()

    # Additional tips
    print("\n" + "=" * 60)
    print("TIPS FOR BEST PERFORMANCE")
    print("=" * 60)
    print("1. Use batch operations - search multiple queries at once")
    print("2. Use dimensions that are multiples of 4 for SIMD optimization")
    print("3. For large datasets, consider approximate methods (IVF, HNSW)")
    print("4. Metal performs better with larger batch sizes (32-1024 queries)")

    print("\nFor more details, see METAL_PYTHON_USAGE.md")


if __name__ == "__main__":
    main()
