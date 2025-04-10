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

SIFT_DIR = "/home/ubuntu/Power-RAG/HNSW/HNSWfaiss/sift"
# 1M samples
xb = read_fvecs(SIFT_DIR + '/sift_base.fvecs')
# queries
xq = read_fvecs(SIFT_DIR + '/sift_query.fvecs')[0].reshape(1, -1)
xq_full = read_fvecs(SIFT_DIR + '/sift_query.fvecs')
xq_1 = xq[0].reshape(1, xq.shape[1])
print(xb.shape)


# recall_idx = []

index_flat = faiss.IndexFlatL2(xb.shape[1])
index_flat.add(xb)
D_flat, recall_idx_flat = index_flat.search(xq_full[:1000], k=3)

print(recall_idx_flat)

print('building index')
index = faiss.IndexHNSWFlat(xb.shape[1], 64, faiss.METRIC_L2, 64)
index.hnsw.efConstruction = efConstruction
index.add(xb)
import time

for efSearch in [2, 4, 8, 16, 32, 64]:
    index.efSearch = efSearch
    # print('searching')
    index.hnsw.efSearch = efSearch
    # calculate the time of searching
    start_time = time.time()
    D, I = index.search(xq_full[:1000], 3)
    end_time = time.time()
    print(f'time: {end_time - start_time}')
    # print(I)

    # calculate the recall using the flat index the formula:
    # recall = sum(recall_idx == recall_idx_flat) / len(recall_idx)
    recall=[]
    for i in range(len(I)):
        acc = 0
        for j in range(len(I[i])):
            if I[i][j] in recall_idx_flat[i]:
                acc += 1
        recall.append(acc / len(I[i]))
    recall = sum(recall) / len(recall)
    print(f'efSearch: {efSearch}')
    print(f'recall: {recall}')

# print('searching q1')
# D, I = index.search(xq_1, 10)
# print(I)

# levels = faiss.vector_to_array(index.hnsw.levels)
# print(levels)

# cum_nneighbor_per_level = faiss.vector_to_array(index.hnsw.cum_nneighbor_per_level)
# print(cum_nneighbor_per_level)

# offsets = faiss.vector_to_array(index.hnsw.offsets)
# print(offsets)
