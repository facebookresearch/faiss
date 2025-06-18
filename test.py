import faiss
import numpy as np

# Step 1: Create the CAGRA index config
config = faiss.GpuIndexCagraConfig()
config.graph_degree = 32
config.intermediate_graph_degree = 64

# Step 2: Initialize the CAGRA index
res = faiss.StandardGpuResources()
gpu_cagra_index = faiss.GpuIndexCagra(res, 96, faiss.METRIC_L2, config)

# Step 3: Add the 1M vectors to the index
n = 1000000
data = np.random.random((n, 96)).astype('float32')
gpu_cagra_index.train(data)

# Step 4: Search the index for top 10 neighbors for each query.
xq = np.random.random((10000, 96)).astype('float32')
D, I = gpu_cagra_index.search(xq,10)

# Step 1: Create the HNSW index object.
d = 96
M = 16
cpu_hnsw_index_1 = faiss.IndexHNSWCagra(d, M, faiss.METRIC_L2)

# Step 2: Initializes the HNSW base layer with the CAGRA graph. 
# Option 1: The resultant HNSW graph is immutable.
cpu_hnsw_index_1.base_level_only=True
gpu_cagra_index.copyTo(cpu_hnsw_index_1)

# Option 2: Initialize the base layer, add the original vectors to the hierarchy. This can make use of OpenMP.
cpu_hnsw_index_2 = faiss.IndexHNSWCagra(d, M, faiss.METRIC_L2)
cpu_hnsw_index_2.base_level_only=False
faiss.omp_set_num_threads(32)
gpu_cagra_index.copyTo(cpu_hnsw_index_2)
# It is possible to add new vectors to the hierarchy
newVecs = np.random.random((100000, 96)).astype('float32')
cpu_hnsw_index_2.add(newVecs)
