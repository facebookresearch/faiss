import os
import pickle
import numpy as np
import time
import math
from pathlib import Path

# --- Configuration ---
DOMAIN_NAME = "rpj_wiki" # Domain name used for finding passages
EMBEDDER_NAME = "facebook/contriever-msmarco" # Used in paths
ORIGINAL_EMBEDDING_SHARD_ID = 0 # The shard ID of the embedding file we are loading

# Define the base directory
SCALING_OUT_DIR = Path("/powerrag/scaling_out").resolve()

# Original Data Paths (using functions similar to your utils)
# Assuming embeddings for rpj_wiki are in a single file despite passage sharding
# Adjust NUM_SHARDS_EMBEDDING if embeddings are also sharded
NUM_SHARDS_EMBEDDING = 1
ORIGINAL_EMBEDDING_FILE_TEMPLATE = (
    SCALING_OUT_DIR
    / "embeddings/{embedder_name}/{domain_name}/{total_shards}-shards/passages_{shard_id:02d}.pkl"
)
ORIGINAL_EMBEDDING_FILE = str(ORIGINAL_EMBEDDING_FILE_TEMPLATE).format(
    embedder_name=EMBEDDER_NAME,
    domain_name=DOMAIN_NAME,
    total_shards=NUM_SHARDS_EMBEDDING,
    shard_id=ORIGINAL_EMBEDDING_SHARD_ID,
)

# Passage Paths
NUM_SHARDS_PASSAGE = 8 # As specified in your original utils (NUM_SHARDS['rpj_wiki'])
ORIGINAL_PASSAGE_FILE_TEMPLATE = (
    SCALING_OUT_DIR
    / "passages/{domain_name}/{total_shards}-shards/raw_passages-{shard_id}-of-{total_shards}.pkl"
)

# New identifier for the sampled dataset
NEW_DATASET_NAME = "rpj_wiki_1M"

# Fraction to sample (1/60)
SAMPLE_FRACTION = 1 / 60

# Output Paths for the new sampled dataset
OUTPUT_EMBEDDING_DIR = SCALING_OUT_DIR / "embeddings" / EMBEDDER_NAME / NEW_DATASET_NAME / "1-shards"
OUTPUT_PASSAGE_DIR = SCALING_OUT_DIR / "passages" / NEW_DATASET_NAME / "1-shards"

OUTPUT_EMBEDDING_FILE = OUTPUT_EMBEDDING_DIR / f"passages_{ORIGINAL_EMBEDDING_SHARD_ID:02d}.pkl"
# The new passage file represents the *single* shard of the sampled data
OUTPUT_PASSAGE_FILE = OUTPUT_PASSAGE_DIR / f"raw_passages-0-of-1.pkl"

# --- Directory Setup ---
print("Creating output directories if they don't exist...")
OUTPUT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PASSAGE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Embeddings output dir: {OUTPUT_EMBEDDING_DIR}")
print(f"Passages output dir: {OUTPUT_PASSAGE_DIR}")


# --- Helper Function to Load Passages ---
def load_all_passages(domain_name, num_shards, template):
    """Loads all passage shards and creates an ID-to-content map."""
    all_passages_list = []
    passage_id_to_content_map = {}
    print(f"Loading passages for domain '{domain_name}' from {num_shards} shards...")
    total_loaded = 0
    start_time = time.time()

    for shard_id in range(num_shards):
        shard_path_str = str(template).format(
            domain_name=domain_name,
            total_shards=num_shards,
            shard_id=shard_id,
        )
        shard_path = Path(shard_path_str)

        if not shard_path.exists():
            print(f"Warning: Passage shard file not found, skipping: {shard_path}")
            continue

        try:
            print(f"  Loading shard {shard_id} from {shard_path}...")
            with open(shard_path, 'rb') as f:
                shard_passages = pickle.load(f) # Expected: list of dicts
                if not isinstance(shard_passages, list):
                     print(f"Warning: Shard {shard_id} data is not a list.")
                     continue

                all_passages_list.extend(shard_passages)
                # Build the map, ensuring IDs are strings for consistent lookup
                for passage_dict in shard_passages:
                    if 'id' in passage_dict:
                        passage_id_to_content_map[str(passage_dict['id'])] = passage_dict
                    else:
                        print(f"Warning: Passage dict in shard {shard_id} missing 'id' key.")
                print(f"  Loaded {len(shard_passages)} passages from shard {shard_id}.")
                total_loaded += len(shard_passages)

        except Exception as e:
            print(f"Error loading passage shard {shard_id} from {shard_path}: {e}")

    load_time = time.time() - start_time
    print(f"Finished loading passages. Total passages loaded: {total_loaded} in {load_time:.2f} seconds.")
    print(f"Total unique passages mapped by ID: {len(passage_id_to_content_map)}")
    return all_passages_list, passage_id_to_content_map


# --- Load Original Embeddings ---
print(f"\nLoading original embeddings from {ORIGINAL_EMBEDDING_FILE}...")
start_load_time = time.time()
try:
    with open(ORIGINAL_EMBEDDING_FILE, 'rb') as f:
        original_embedding_data = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Original embedding file not found at {ORIGINAL_EMBEDDING_FILE}")
    exit(1)
except Exception as e:
    print(f"Error loading embedding pickle file: {e}")
    exit(1)
load_time = time.time() - start_load_time
print(f"Loaded original embeddings data in {load_time:.2f} seconds.")

# --- Extract and Validate Embeddings ---
try:
    if not isinstance(original_embedding_data, (list, tuple)) or len(original_embedding_data) != 2:
        raise TypeError("Expected embedding data to be a list or tuple of length 2 (ids, embeddings)")

    original_embedding_ids = original_embedding_data[0] # Should be a list/iterable of IDs
    original_embeddings = original_embedding_data[1] # Should be a NumPy array

    # Ensure IDs are in a list for easier indexing later if they aren't already
    if not isinstance(original_embedding_ids, list):
        print("Converting embedding IDs to list...")
        original_embedding_ids = list(original_embedding_ids)

    if not isinstance(original_embeddings, np.ndarray):
        raise TypeError("Expected second element of embedding data to be a NumPy array")

    print(f"Original data contains {len(original_embedding_ids)} embedding IDs.")
    print(f"Original embeddings shape: {original_embeddings.shape}, dtype: {original_embeddings.dtype}")

    if len(original_embedding_ids) != original_embeddings.shape[0]:
        raise ValueError(f"Mismatch! Number of embedding IDs ({len(original_embedding_ids)}) does not match number of embeddings ({original_embeddings.shape[0]})")

except (TypeError, ValueError, IndexError) as e:
    print(f"Error processing loaded embedding data: {e}")
    print("Please ensure the embedding pickle file contains: (list_of_passage_ids, numpy_embedding_array)")
    exit(1)

total_embeddings = original_embeddings.shape[0]

# --- Load Original Passages ---
# This might take time and memory depending on the dataset size
_, passage_id_to_content_map = load_all_passages(
    DOMAIN_NAME, NUM_SHARDS_PASSAGE, ORIGINAL_PASSAGE_FILE_TEMPLATE
)

if not passage_id_to_content_map:
    print("Error: No passages were loaded. Cannot proceed with sampling.")
    exit(1)

# --- Calculate Sample Size ---
num_samples = math.ceil(total_embeddings * SAMPLE_FRACTION) # Use ceil to get at least 1/60th
print(f"\nTotal original embeddings: {total_embeddings}")
print(f"Sampling fraction: {SAMPLE_FRACTION:.6f} (1/60)")
print(f"Target number of samples: {num_samples}")

if num_samples > total_embeddings:
    print("Warning: Calculated sample size exceeds total embeddings. Using all embeddings.")
    num_samples = total_embeddings
elif num_samples <= 0:
    print("Error: Calculated sample size is zero or negative.")
    exit(1)

# --- Perform Random Sampling (Based on Embeddings) ---
print("\nPerforming random sampling based on embeddings...")
start_sample_time = time.time()

# Set a seed for reproducibility if needed
# np.random.seed(42)

# Generate unique random indices from the embeddings list
sampled_indices = np.random.choice(total_embeddings, size=num_samples, replace=False)

# Retrieve the corresponding IDs and embeddings using the sampled indices
sampled_embedding_ids = [original_embedding_ids[i] for i in sampled_indices]
sampled_embeddings = original_embeddings[sampled_indices]

sample_time = time.time() - start_sample_time
print(f"Sampling completed in {sample_time:.2f} seconds.")
print(f"Sampled {len(sampled_embedding_ids)} IDs and embeddings.")
print(f"Sampled embeddings shape: {sampled_embeddings.shape}")

# --- Retrieve Corresponding Passages ---
print("\nRetrieving corresponding passages for sampled IDs...")
start_passage_retrieval_time = time.time()
sampled_passages = []
missing_ids_count = 0
for i, pid in enumerate(sampled_embedding_ids):
    # Convert pid to string for lookup in the map
    pid_str = str(pid)
    if pid_str in passage_id_to_content_map:
        sampled_passages.append(passage_id_to_content_map[pid_str])
    else:
        # This indicates an inconsistency between embedding IDs and passage IDs
        print(f"Warning: Passage ID '{pid_str}' (from embedding index {sampled_indices[i]}) not found in passage map.")
        missing_ids_count += 1

passage_retrieval_time = time.time() - start_passage_retrieval_time
print(f"Retrieved {len(sampled_passages)} passages in {passage_retrieval_time:.2f} seconds.")
if missing_ids_count > 0:
    print(f"Warning: Could not find passages for {missing_ids_count} sampled IDs.")

if not sampled_passages:
      print("Error: No corresponding passages found for the sampled embeddings. Check ID matching.")
      exit(1)

# --- Prepare Output Data ---
# Embeddings output format: tuple(list_of_ids, numpy_array_of_embeddings)
output_embedding_data = (sampled_embedding_ids, sampled_embeddings)
# Passages output format: list[dict] (matching input shard format)
output_passage_data = sampled_passages

# --- Save Sampled Embeddings ---
print(f"\nSaving sampled embeddings to {OUTPUT_EMBEDDING_FILE}...")
start_save_time = time.time()
try:
    with open(OUTPUT_EMBEDDING_FILE, 'wb') as f:
        pickle.dump(output_embedding_data, f, protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print(f"Error saving sampled embeddings: {e}")
    # Continue to try saving passages if desired, or exit(1)
save_time = time.time() - start_save_time
print(f"Saved sampled embeddings in {save_time:.2f} seconds.")

# --- Save Sampled Passages ---
print(f"\nSaving sampled passages to {OUTPUT_PASSAGE_FILE}...")
start_save_time = time.time()
try:
    with open(OUTPUT_PASSAGE_FILE, 'wb') as f:
        pickle.dump(output_passage_data, f, protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print(f"Error saving sampled passages: {e}")
    exit(1)
save_time = time.time() - start_save_time
print(f"Saved sampled passages in {save_time:.2f} seconds.")

print(f"\nScript finished successfully.")
print(f"Sampled embeddings saved to: {OUTPUT_EMBEDDING_FILE}")
print(f"Sampled passages saved to:   {OUTPUT_PASSAGE_FILE}")