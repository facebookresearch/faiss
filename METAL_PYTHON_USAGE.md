# Using Faiss Metal Backend from Python

This guide explains how to port your existing Python Faiss code to use the Metal GPU backend on Apple Silicon Macs.

## Prerequisites

1. Apple Silicon Mac (M1, M2, M3, etc.)
2. macOS 11.0 or later
3. Python 3.8+
4. Faiss compiled with Metal support

## Building Faiss with Metal Support

First, ensure Faiss is built with Metal support enabled:

```bash
# Clone the repository
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Configure with Metal support
cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_METAL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE=$(which python3)

# Build
cmake --build build -j

# Install Python bindings
cd build/faiss/python
python setup.py install
```

## Basic Usage Pattern

The Metal backend follows the same API as CUDA GPU indexes, but uses Metal-specific classes:

### 1. Simple Flat Index Example

**CPU Version:**
```python
import faiss
import numpy as np

# Parameters
d = 128  # dimension
nb = 100000  # database size
nq = 1000  # number of queries

# Generate random data
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# CPU index
index = faiss.IndexFlatL2(d)
index.add(xb)
k = 10
D, I = index.search(xq, k)
```

**Metal Version:**
```python
import faiss
import numpy as np

# Parameters
d = 128  # dimension
nb = 100000  # database size
nq = 1000  # number of queries

# Generate random data
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Metal index
res = faiss.StandardMetalResources()
index = faiss.MetalIndexFlatL2(res, d)
index.add(xb)
k = 10
D, I = index.search(xq, k)
```

### 2. HNSW Index Example

**CPU Version:**
```python
import faiss

# Create HNSW index
M = 32
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200
index.add(xb)
index.hnsw.efSearch = 64
D, I = index.search(xq, k)
```

**Metal Version:**
```python
import faiss

# Create Metal HNSW index
M = 32
res = faiss.StandardMetalResources()
index = faiss.MetalIndexHNSW(res, d, M)
index.hnsw.efConstruction = 200
index.add(xb)
index.hnsw.efSearch = 64
D, I = index.search(xq, k)
```

### 3. IVF Index Example

**CPU Version:**
```python
import faiss

# Create IVF index
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(xb)
index.add(xb)
index.nprobe = 10
D, I = index.search(xq, k)
```

**Metal Version:**
```python
import faiss

# Create Metal IVF index
nlist = 100
res = faiss.StandardMetalResources()
quantizer = faiss.IndexFlatL2(d)
index = faiss.MetalIndexIVFFlat(res, quantizer, d, nlist)
index.train(xb)
index.add(xb)
index.nprobe = 10
D, I = index.search(xq, k)
```

## Migration Guide

### Step 1: Identify Your Index Type

Look at your current Faiss index creation:
- `faiss.IndexFlatL2` → `faiss.MetalIndexFlatL2`
- `faiss.IndexFlatIP` → `faiss.MetalIndexFlatIP`
- `faiss.IndexHNSWFlat` → `faiss.MetalIndexHNSW`
- `faiss.IndexIVFFlat` → `faiss.MetalIndexIVFFlat`
- `faiss.IndexIVFPQ` → `faiss.MetalIndexIVFPQ`

### Step 2: Create Metal Resources

Add resource creation before index creation:
```python
# Create Metal resources (do this once)
res = faiss.StandardMetalResources()
```

### Step 3: Update Index Creation

Replace your index creation with the Metal equivalent:

**Before:**
```python
index = faiss.IndexFlatL2(d)
```

**After:**
```python
res = faiss.StandardMetalResources()
index = faiss.MetalIndexFlatL2(res, d)
```

### Step 4: Keep Everything Else the Same

The `add()`, `search()`, and other methods work identically:
```python
# These remain unchanged
index.add(vectors)
distances, indices = index.search(queries, k)
```

## Complete Migration Example

Here's a complete example showing how to migrate a similarity search application:

**Original CPU Code:**
```python
import faiss
import numpy as np
import time

class SimilaritySearch:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_vectors(self, vectors):
        self.index.add(vectors)
    
    def search(self, queries, k=10):
        return self.index.search(queries, k)

# Usage
d = 128
searcher = SimilaritySearch(d)

# Add vectors
data = np.random.random((100000, d)).astype('float32')
searcher.add_vectors(data)

# Search
queries = np.random.random((1000, d)).astype('float32')
start = time.time()
D, I = searcher.search(queries, k=10)
print(f"Search time: {time.time() - start:.3f}s")
```

**Migrated Metal Code:**
```python
import faiss
import numpy as np
import time

class MetalSimilaritySearch:
    def __init__(self, dimension):
        self.dimension = dimension
        self.res = faiss.StandardMetalResources()
        self.index = faiss.MetalIndexFlatL2(self.res, dimension)
    
    def add_vectors(self, vectors):
        self.index.add(vectors)
    
    def search(self, queries, k=10):
        return self.index.search(queries, k)

# Usage
d = 128
searcher = MetalSimilaritySearch(d)

# Add vectors
data = np.random.random((100000, d)).astype('float32')
searcher.add_vectors(data)

# Search
queries = np.random.random((1000, d)).astype('float32')
start = time.time()
D, I = searcher.search(queries, k=10)
print(f"Search time: {time.time() - start:.3f}s")
```

## Performance Considerations

1. **Batch Size**: Metal performs better with larger batch sizes. Try to search with at least 32-64 queries at once.

2. **Vector Dimensions**: Performance is best with dimensions that are multiples of 4 (for SIMD optimization).

3. **Index Type Selection**:
   - Use `MetalIndexFlatL2/IP` for exact search with small datasets (<1M vectors)
   - Use `MetalIndexHNSW` for approximate search with good recall
   - Use `MetalIndexIVFFlat` for large-scale approximate search

4. **Memory Management**: Metal uses unified memory on Apple Silicon, so data transfer overhead is minimal.

## Debugging Tips

1. **Check Metal Support**:
```python
# Verify Metal is available
try:
    res = faiss.StandardMetalResources()
    print("Metal support is available")
except:
    print("Metal support not available")
```

2. **Fallback Pattern**:
```python
def create_index(dimension, use_gpu=True):
    try:
        if use_gpu:
            res = faiss.StandardMetalResources()
            return faiss.MetalIndexFlatL2(res, dimension)
    except:
        print("Falling back to CPU")
    return faiss.IndexFlatL2(dimension)
```

3. **Performance Monitoring**:
```python
import time

def benchmark_search(index, queries, k=10):
    # Warm up
    index.search(queries[:10], k)
    
    # Benchmark
    start = time.time()
    D, I = index.search(queries, k)
    metal_time = time.time() - start
    
    print(f"Search time: {metal_time:.3f}s")
    print(f"Queries per second: {len(queries)/metal_time:.0f}")
    return D, I
```

## Advanced Usage

### Using Multiple Metal Devices (Future)
```python
# Currently only single device is supported
res = faiss.StandardMetalResources()
# Future: res.setDevice(0)  # Use specific Metal device
```

### Custom Resource Configuration
```python
# Set custom resource limits (when implemented)
res = faiss.StandardMetalResources()
# Future: res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
```

### Mixing CPU and Metal Indexes
```python
# You can use both CPU and Metal indexes in the same application
cpu_index = faiss.IndexFlatL2(d)
metal_index = faiss.MetalIndexFlatL2(res, d)

# Add same data to both
cpu_index.add(xb)
metal_index.add(xb)

# Compare performance
import time

start = time.time()
D_cpu, I_cpu = cpu_index.search(xq, k)
cpu_time = time.time() - start

start = time.time()
D_metal, I_metal = metal_index.search(xq, k)
metal_time = time.time() - start

print(f"CPU time: {cpu_time:.3f}s")
print(f"Metal time: {metal_time:.3f}s")
print(f"Speedup: {cpu_time/metal_time:.2f}x")
```

## Troubleshooting

### Import Error
If you get `AttributeError: module 'faiss' has no attribute 'MetalIndexFlatL2'`:
- Make sure Faiss was built with `-DFAISS_ENABLE_METAL=ON`
- Reinstall the Python bindings after building

### Performance Issues
If Metal is slower than CPU:
- Ensure you're using batch searches (multiple queries at once)
- Check that you're using appropriate index types for your data size
- Verify you're on Apple Silicon (not Intel Mac)

### Memory Errors
If you get Metal memory allocation errors:
- Reduce batch size
- Check available memory with Activity Monitor
- Consider using approximate indexes (IVF, HNSW) for large datasets

## Example: Real-World Migration

Here's how to migrate a recommendation system:

**Original Code:**
```python
class ProductRecommender:
    def __init__(self, product_embeddings):
        self.d = product_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.d)  # Inner product for cosine similarity
        faiss.normalize_L2(product_embeddings)
        self.index.add(product_embeddings)
    
    def find_similar(self, query_embedding, k=20):
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, k)
        return I[0], D[0]  # indices and similarities
```

**Metal Version:**
```python
class MetalProductRecommender:
    def __init__(self, product_embeddings):
        self.d = product_embeddings.shape[1]
        self.res = faiss.StandardMetalResources()
        self.index = faiss.MetalIndexFlatIP(self.res, self.d)  # Metal inner product
        faiss.normalize_L2(product_embeddings)
        self.index.add(product_embeddings)
    
    def find_similar(self, query_embedding, k=20):
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, k)
        return I[0], D[0]  # indices and similarities
```

The migration is straightforward - just add resource creation and use Metal index classes!