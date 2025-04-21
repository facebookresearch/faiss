import time
import faiss
import numpy as np
import pickle
import os

M = 32
efSearch = 32  # number of entry points (neighbors) we use on each layer
efConstruction = 256  # number of entry points used on each layer
                     # during construction


EMBEDDING_FILE = "/powerrag/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki_1M/1-shards/passages_00.pkl"
# 1M samples
print(f"Loading embeddings from {EMBEDDING_FILE}...")
with open(EMBEDDING_FILE, 'rb') as f:
    data = pickle.load(f)

xb = data[1]
print(f"Original dtype: {xb.dtype}")

if xb.dtype != np.float32:
    print("Converting embeddings to float32.")
    xb = xb.astype(np.float32)
else:
    print("Embeddings are already float32.")
print(f"Loaded database embeddings (xb), shape: {xb.shape}")
d = xb.shape[1] # Get dimension

# 1000 queries
num_queries = 1000
if num_queries > xb.shape[0]:
    print(f"Warning: Requested {num_queries} queries, but only {xb.shape[0]} vectors available. Using all vectors as queries.")
    num_queries = xb.shape[0]

print(f"Using the first {num_queries} vectors from the database as queries (xq_full).")
xq_full = xb[:num_queries]
print(f"Query embeddings (xq_full), shape: {xq_full.shape}")


# recall_idx = []

index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
D_flat, recall_idx_flat = index_flat.search(xq_full[:1000], k=3)

print(recall_idx_flat)

print('building index')
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2, 32)
index.hnsw.efConstruction = efConstruction
index.add(xb)

for efSearch in [2, 4, 8, 16, 32, 64]:
    index.efSearch = efSearch
    # print('searching')
    index.hnsw.efSearch = efSearch
    # calculate the time of searching
    start_time = time.time()
    D, I = index.timech(xq_full[:1000], 3)
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