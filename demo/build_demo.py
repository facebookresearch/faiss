import os
import pickle
import faiss
import numpy as np
import time

EMBEDDING_FILE = "/opt/dlami/nvme/scaling_out/embeddings/facebook/contriever-msmarco/rpj_wiki/1-shards/passages_00.pkl"
INDEX_OUTPUT_DIR = "/opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/hnsw" # 保存索引的目录
M_VALUES_FOR_L2 = [30, 60] # M values for L2
EF_CONSTRUCTION_FOR_L2 = 128 # fixed efConstruction

if not os.path.exists(INDEX_OUTPUT_DIR):
    print(f"Creating index directory: {INDEX_OUTPUT_DIR}")
    os.makedirs(INDEX_OUTPUT_DIR)

print(f"Loading embeddings from {EMBEDDING_FILE}...")
with open(EMBEDDING_FILE, 'rb') as f:
    data = pickle.load(f)
# Directly assume data is a tuple and the second element is embeddings
embeddings = data[1]

print(f"Converting embeddings from {embeddings.dtype} to float32.")
embeddings = embeddings.astype(np.float32)
print(f"Loaded embeddings, shape: {embeddings.shape}")
dim = embeddings.shape[1]

# --- Build HNSW L2 index ---
print("\n--- Build HNSW L2 index ---")

# Loop through M values
for HNSW_M in M_VALUES_FOR_L2:
    efConstruction = EF_CONSTRUCTION_FOR_L2

    print(f"\nBuilding HNSW L2 index: M={HNSW_M}, efConstruction={efConstruction}...")

    # Define the filename and path for the L2 index
    hnsw_filename = f"hnsw_IP_M{HNSW_M}_efC{efConstruction}.index"
    hnsw_filepath = os.path.join(INDEX_OUTPUT_DIR, hnsw_filename)

    # Note: No longer check if the file exists, it will be overwritten if it exists

    # Create HNSW L2 index
    index_hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = efConstruction

    index_hnsw.verbose = True

    print(f"Adding {embeddings.shape[0]} vectors to HNSW L2 (M={HNSW_M}) index...")
    start_time_build = time.time()

    index_hnsw.add(embeddings) 

    end_time_build = time.time()
    build_time_s = end_time_build - start_time_build
    print(f"HNSW L2 build time: {build_time_s:.4f} seconds")

    # Save L2 index (direct operation, no try-except)
    print(f"Saving HNSW L2 index to {hnsw_filepath}")
    faiss.write_index(index_hnsw, hnsw_filepath)
    # Do not check storage size or handle save errors

    print(f"Index {hnsw_filename} saved.")

    del index_hnsw

print("\n--- HNSW L2 index build completed ---")
print("\nScript ended.")