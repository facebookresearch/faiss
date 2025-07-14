# Faiss Metal Backend - Quick Reference

## Installation

```bash
cmake -B build -DFAISS_ENABLE_METAL=ON
cmake --build build -j
cd build/faiss/python && python setup.py install
```

## Basic Pattern

```python
# CPU Version
import faiss
index = faiss.IndexFlatL2(d)

# Metal Version
import faiss
res = faiss.StandardMetalResources()
index = faiss.MetalIndexFlatL2(res, d)
```

## Index Type Mapping

| CPU Index | Metal Index | Usage |
|-----------|-------------|-------|
| `IndexFlatL2` | `MetalIndexFlatL2` | Exact L2 search |
| `IndexFlatIP` | `MetalIndexFlatIP` | Exact inner product |
| `IndexHNSWFlat` | `MetalIndexHNSW` | Graph-based ANN |
| `IndexIVFFlat` | `MetalIndexIVFFlat` | Inverted file index |
| `IndexIVFPQ` | `MetalIndexIVFPQ` | Product quantization |

## Complete Example

```python
import faiss
import numpy as np

# 1. Create Metal resources
res = faiss.StandardMetalResources()

# 2. Create index (L2 distance)
d = 128  # dimension
index = faiss.MetalIndexFlatL2(res, d)

# 3. Add vectors (same as CPU)
vectors = np.random.random((100000, d)).astype('float32')
index.add(vectors)

# 4. Search (same as CPU)
queries = np.random.random((100, d)).astype('float32')
k = 10
D, I = index.search(queries, k)
```

## Migration Checklist

- [ ] Add `res = faiss.StandardMetalResources()`
- [ ] Change index type: `IndexFlatL2` → `MetalIndexFlatL2`
- [ ] Pass `res` as first parameter to Metal index constructor
- [ ] Keep all other code unchanged

## Performance Tips

1. **Batch Size**: Use at least 32-64 queries per search
2. **Dimensions**: Use multiples of 4 (128, 256, 512)
3. **Large Datasets**: Consider approximate indexes (IVF, HNSW)

## Error Handling

```python
def create_index(d, use_gpu=True):
    try:
        if use_gpu:
            res = faiss.StandardMetalResources()
            return faiss.MetalIndexFlatL2(res, d)
    except:
        pass
    return faiss.IndexFlatL2(d)  # Fallback to CPU
```

## Common Issues

**Import Error**: `AttributeError: 'MetalIndexFlatL2'`
- Solution: Rebuild with `-DFAISS_ENABLE_METAL=ON`

**Slower than CPU**: 
- Solution: Increase batch size, use appropriate index type

**Memory Error**:
- Solution: Use approximate indexes for large datasets