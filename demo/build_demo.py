import faiss
import numpy as np

M = 32
efSearch = 32  # number of entry points (neighbors) we use on each layer
efConstruction = 32  # number of entry points used on each layer
                     # during construction


# now define a function to read the fvecs file format of Sift1M dataset
def read_fvecs(fp):
    a = np.fromfile(fp, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')
SIFT_DIR="/home/ubuntu/Power-RAG/HNSW/HNSWfaiss/sift"
# 1M samples
xb = read_fvecs(SIFT_DIR + '/sift_base.fvecs')
# queries
xq = read_fvecs(SIFT_DIR + '/sift_query.fvecs')[0].reshape(1, -1)
xq_full = read_fvecs(SIFT_DIR + '/sift_query.fvecs')
print(xb.shape)

print('building index')
index = faiss.IndexHNSWFlat(xb.shape[1], M)
index.hnsw.efConstruction = efConstruction
index.add(xb)

print('searching')
index.hnsw.efSearch = efSearch
D, I = index.search(xq, 10)
print(I)

levels = faiss.vector_to_array(index.hnsw.levels)
print(levels)

cum_nneighbor_per_level = faiss.vector_to_array(index.hnsw.cum_nneighbor_per_level)
print(cum_nneighbor_per_level)

offsets = faiss.vector_to_array(index.hnsw.offsets)
print(offsets)
