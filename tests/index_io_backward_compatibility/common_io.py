# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Common utilities for cross-build file I/O testing.
Provides shared functions for writing, reading, and verifying FAISS indexes.
"""

import os
import json
import numpy as np
import faiss
import traceback

d = 32
nb = 1000

INDEX_TYPES = [
    # Basic flat indexes
    "Flat",
    # PQ indexes
    "PQ4",
    "PQ4np",
    # Scalar quantizer
    "SQ8",
    # IVF indexes
    "IVF10,PQ4",
    "IVF10,PQ4np",
    "IVF10,FlatDedup",
    # Pre-transforms with indexes
    "PCAR8,IVF10,PQ4",
    "OPQ16,Flat",
    "OPQ16_64,Flat",
    "PCA30,IVF256(PQ15),Flat",
    # RaBitQ indexes
    "RaBitQ",
    "IVF256,RaBitQ",
    "IVF256,RaBitQ4",  # multibit
    "RaBitQfs",
    "RaBitQfs_64",  # batch size 64
    "IVF256,RaBitQfs",
    "IVF256,RaBitQfs_64",  # batch size 64
    # HNSW indexes
    "HNSW32",
    "HNSW32_SQ8",
    "HNSW32_PQ4",
    "HNSW32,Flat",
    "HNSW32,SQ8",
    "HNSW,PQ4",
    "HNSW32,PQ4np",
    "HNSW32,PQ4x6np",
    # NSG indexes
    "NSG64",
    "NSG64,Flat",
    "NSG64,PQ4x6",
    "IVF200_NSG64,Flat",
    "IVF200_NSG64,PQ2x8",
    # LSH indexes
    "LSHrt",
    "LSH16rt",
    # FastScan indexes
    "PQ32x4fs",
    "PQ32x4fs_64",
    "IVF50,PQ32x4fs_64",
    "IVF50,PQ32x4fsr_64",
    "PQ32x4fs,RFlat",
    # Parenthesized quantizers
    "IVF256(PQ16),Flat",
    # Refine indexes
    "IVF32,Flat,Refine(PQ16x4)",
    "PCA32,IVF32,Flat,Refine(PQ16x4)",
    "IVF1000(IVF20,SQ4,Refine(SQ8)),Flat",
    "Flat,RFlat",
    "LSHrt,Refine(Flat)",
    "IDMap,PQ4x4fs,RFlat",
    # Residual quantization
    "IVF1000,PQ32x4fsr",
    # Pre-transforms
    "PCAR16,L2Norm,PCAW8,LSHr",
    "PCA10,Flat",
    "ITQ8,LSHt",
    # IVF variants
    "IVF456,Flat",
    "IVF100_HNSW,Flat",
    "IVF100(LSHr),Flat",
    # IDMap
    "Flat,IDMap",
    "Flat,IDMap2",
    "IDMap2,Flat",
    # Additive quantizers
    "IVF256(RCQ2x4),RQ3x4",
    "RQ2x4_2x4_6x4",
    "RQ8x8_Nqint8",
    # IVFSpectralHash
    "IVF256,ITQ16,SH1.2",
    # Panorama
    "IVF256,FlatPanorama8",
    "FlatL2Panorama8",
    "FlatL2Panorama8_256",
]

INDEX_BINARY_TYPES = [
    "BIVF10",
    "BFlat",
    "BHNSW32",
    "BIVF128_BHNSW32",
    "BHash12",
    "BHash5x6",
    "IDMap2,BFlat",
    "BFlat,IDMap2",
]


def sanitize_index_key(index_key: str) -> str:
    return (
        index_key.replace(",", "__").replace("(", "--").replace(")", "--")
    )


def generate_vectors(d: int, nb: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    xb = np.random.random((nb, d)).astype("float32")
    xb[:, 0] += np.arange(nb) / 1000.0
    return xb


def write_vectors(
    output_dir: str, vectors: np.ndarray, filename: str
) -> None:
    path = os.path.join(output_dir, filename)
    print(f"Writing vectors to: {path}")
    np.save(path, vectors)


def create_and_populate_index(
    index_key: str, xb: np.ndarray
) -> faiss.Index:
    d = xb.shape[1]
    index = faiss.index_factory(d, index_key)
    print(f"Created index type: {type(index).__name__}")

    if not index.is_trained:
        print("Training index...")
        index.train(xb)

    print(f"Adding {xb.shape[0]} vectors to index")
    # IDMap indexes require add_with_ids() instead of add()
    if "IDMap" in index_key:
        ids = np.arange(xb.shape[0], dtype=np.int64)
        print(
            f"Using add_with_ids() with IDs from 0 to {xb.shape[0] - 1}"
        )
        index.add_with_ids(xb, ids)
    else:
        index.add(xb)

    print(f"Index now contains {index.ntotal} vectors")
    return index


def create_and_populate_binary_index(
    index_key: str, xb: np.ndarray
) -> faiss.IndexBinary:
    d = xb.shape[1] * 8
    index = faiss.index_binary_factory(d, index_key)
    print(f"Created binary index type: {type(index).__name__}")

    if not index.is_trained:
        print("Training binary index...")
        index.train(xb)

    print(f"Adding {xb.shape[0]} vectors to binary index")
    # IDMap indexes require add_with_ids() instead of add()
    if "IDMap" in index_key:
        ids = np.arange(xb.shape[0], dtype=np.int64)
        print(
            f"Using add_with_ids() with IDs from 0 to {xb.shape[0] - 1}"
        )
        index.add_with_ids(xb, ids)
    else:
        index.add(xb)

    print(f"Binary index now contains {index.ntotal} vectors")
    return index


def write_index(
    index: faiss.Index, output_dir: str, filename: str
) -> None:
    index_path = os.path.join(output_dir, filename)

    # Delete existing file if present
    if os.path.exists(index_path):
        print(f"Deleting existing index file: {index_path}")
        os.remove(index_path)

    print(f"Writing index to: {index_path}")
    faiss.write_index(index, index_path)


def write_binary_index(
    index: faiss.IndexBinary, output_dir: str, filename: str
) -> None:
    index_path = os.path.join(output_dir, filename)

    # Delete existing file if present
    if os.path.exists(index_path):
        print(f"Deleting existing binary index file: {index_path}")
        os.remove(index_path)

    print(f"Writing binary index to: {index_path}")
    faiss.write_index_binary(index, index_path)


def read_index(input_dir: str, filename: str) -> faiss.Index:
    index_path = os.path.join(input_dir, filename)
    print(f"Loading index from: {index_path}")
    return faiss.read_index(index_path)


def read_binary_index(input_dir: str, filename: str) -> faiss.IndexBinary:
    index_path = os.path.join(input_dir, filename)
    print(f"Loading binary index from: {index_path}")
    return faiss.read_index_binary(index_path)


def test_index_file(index_info: dict, input_dir: str) -> dict:
    index_key = index_info["index_key"]
    filename = index_info["filename"]

    try:
        print(f"Testing: {index_key}")

        index = read_index(input_dir, filename)
        print(f"Index type: {type(index).__name__}")
        print(f"Index contains {index.ntotal} vectors")

        # Delete index file after successful verification
        index_path = os.path.join(input_dir, filename)
        print(f"Deleting index file: {index_path}")
        os.remove(index_path)

        return {
            "index_key": index_key,
            "index_type": type(index).__name__,
            "status": "success",
            "ntotal": int(index.ntotal),
        }

    except Exception as e:
        print(f"ERROR: Failed to process {index_key}: {e}")
        traceback.print_exc()
        return {
            "index_key": index_key,
            "status": "failed",
            "error": str(e),
        }


def test_binary_index_file(index_info: dict, input_dir: str) -> dict:
    index_key = index_info["index_key"]
    filename = index_info["filename"]

    try:
        print(f"Testing binary: {index_key}")

        index = read_binary_index(input_dir, filename)
        print(f"Binary index type: {type(index).__name__}")
        print(f"Binary index contains {index.ntotal} vectors")

        # Delete index file after successful verification
        index_path = os.path.join(input_dir, filename)
        print(f"Deleting binary index file: {index_path}")
        os.remove(index_path)

        return {
            "index_key": index_key,
            "index_type": type(index).__name__,
            "status": "success",
            "ntotal": int(index.ntotal),
        }

    except Exception as e:
        print(f"ERROR: Failed to process binary {index_key}: {e}")
        traceback.print_exc()
        return {
            "index_key": index_key,
            "status": "failed",
            "error": str(e),
        }


def test_write_index_type(
    index_key: str, output_dir: str, xb: np.ndarray, writer: str
) -> dict:
    try:
        print(f"Testing: {index_key}")

        index = create_and_populate_index(index_key, xb)

        safe_key = sanitize_index_key(index_key)
        filename = f"{writer}_{safe_key}.faissindex"
        write_index(index, output_dir, filename)

        return {
            "index_key": index_key,
            "index_type": type(index).__name__,
            "status": "success",
            "ntotal": int(index.ntotal),
            "filename": filename,
        }

    except Exception as e:
        print(f"ERROR: Failed to process {index_key}: {e}")
        traceback.print_exc()
        return {
            "index_key": index_key,
            "status": "failed",
            "error": str(e),
        }


def test_write_binary_index_type(
    index_key: str, output_dir: str, xb: np.ndarray, writer: str
) -> dict:
    try:
        print(f"Testing binary: {index_key}")

        index = create_and_populate_binary_index(index_key, xb)

        safe_key = sanitize_index_key(index_key)
        filename = f"{writer}_{safe_key}.faissindex"
        write_binary_index(index, output_dir, filename)

        return {
            "index_key": index_key,
            "index_type": type(index).__name__,
            "status": "success",
            "ntotal": int(index.ntotal),
            "filename": filename,
        }

    except Exception as e:
        print(f"ERROR: Failed to process binary {index_key}: {e}")
        traceback.print_exc()
        return {
            "index_key": index_key,
            "status": "failed",
            "error": str(e),
        }


def write_test_all_files(
    writer: str, output_dir: str, seed: int
) -> int:
    print(f"{writer.capitalize()} Writer: Testing index serialization")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating shared vectors: {nb} database vectors, dimension {d}")
    xb = generate_vectors(d, nb, seed=seed)
    write_vectors(output_dir, xb, f"vectors_db_{writer}.npy")

    results = []
    for index_key in INDEX_TYPES:
        result = test_write_index_type(index_key, output_dir, xb, writer)
        results.append(result)

    write_metadata(
        output_dir, f"{writer}_metadata.json", writer, d, nb, results
    )

    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count

    print(f"{writer.capitalize()} Writer: Summary")
    print(f"Total index types tested: {len(INDEX_TYPES)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    if fail_count > 0:
        print("\nSome index types failed to serialize")
        for result in results:
            if result["status"] == "failed":
                print(
                    f"  - {result['index_key']}: "
                    f"{result.get('error', 'Unknown error')}"
                )
        return 1

    print("\nAll index types serialized successfully")

    # Now write binary indexes
    print(
        f"\n{writer.capitalize()} Writer: "
        "Testing binary index serialization"
    )
    print(f"\nGenerating binary vectors: {nb} database vectors, dimension {d}")
    assert d % 8 == 0
    np.random.seed(seed)
    xb_binary = np.random.randint(256, size=(nb, int(d / 8))).astype('uint8')
    write_vectors(output_dir, xb_binary, f"vectors_db_binary_{writer}.npy")

    binary_results = []
    for index_key in INDEX_BINARY_TYPES:
        result = test_write_binary_index_type(
            index_key, output_dir, xb_binary, writer
        )
        binary_results.append(result)

    write_metadata(
        output_dir,
        f"{writer}_binary_metadata.json",
        writer,
        d,
        nb,
        binary_results,
    )

    binary_success_count = sum(
        1 for r in binary_results if r["status"] == "success"
    )
    binary_fail_count = len(binary_results) - binary_success_count

    print(f"{writer.capitalize()} Writer: Binary Summary")
    print(f"Total binary index types tested: {len(INDEX_BINARY_TYPES)}")
    print(f"Successful: {binary_success_count}")
    print(f"Failed: {binary_fail_count}")

    if binary_fail_count > 0:
        print("\nSome binary index types failed to serialize")
        for result in binary_results:
            if result["status"] == "failed":
                print(
                    f"  - {result['index_key']}: "
                    f"{result.get('error', 'Unknown error')}"
                )
        return 1

    print("\nAll binary index types serialized successfully")
    return 0


def read_test_all_files(reader: str, writer: str, input_dir: str) -> int:
    print(f"{reader} Reader: Testing index deserialization")
    metadata = read_metadata(input_dir, f"{writer}_metadata.json")

    print(f"Metadata from {writer} build:")
    print_metadata(metadata)

    results = []
    success_count = 0
    fail_count = 0
    for index_info in metadata["results"]:
        if index_info["status"] != "success":
            print(
                f"\nSkipping {index_info['index_key']} "
                f"(failed during {writer} writing)"
            )
            continue

        result = test_index_file(index_info, input_dir)
        results.append(result)

        if result["status"] == "success":
            success_count += 1
        else:
            fail_count += 1

    print(f"{reader} Reader: Summary")
    print(f"Total index types tested: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    # Clean up remaining files
    cleanup_files(
        [
            os.path.join(input_dir, f"{writer}_metadata.json"),
            os.path.join(input_dir, f"vectors_db_{writer}.npy"),
        ]
    )

    if fail_count > 0:
        print(
            "\nSome index types failed to deserialize/search. "
            "Malformed fourcc indicates extra non-backward-compatible bytes "
            "being written."
        )
        for result in results:
            if result["status"] == "failed":
                print(
                    f"  - {result['index_key']}: "
                    f"{result.get('error', 'Unknown error')}"
                )
        return 1

    print("\nAll index types deserialized and searched successfully")
    print(f"Index serialization compatibility verified: {writer} → {reader}")

    # Now read binary indexes
    print(f"\n{reader} Reader: Testing binary index deserialization")
    binary_metadata = read_metadata(
        input_dir, f"{writer}_binary_metadata.json"
    )

    print(f"Binary metadata from {writer} build:")
    print_metadata(binary_metadata)

    binary_results = []
    binary_success_count = 0
    binary_fail_count = 0
    for index_info in binary_metadata["results"]:
        if index_info["status"] != "success":
            print(
                f"\nSkipping binary {index_info['index_key']} "
                f"(failed during {writer} writing)"
            )
            continue

        result = test_binary_index_file(index_info, input_dir)
        binary_results.append(result)

        if result["status"] == "success":
            binary_success_count += 1
        else:
            binary_fail_count += 1

    print(f"{reader} Reader: Binary Summary")
    print(f"Total binary index types tested: {len(binary_results)}")
    print(f"Successful: {binary_success_count}")
    print(f"Failed: {binary_fail_count}")

    # Clean up remaining files
    cleanup_files(
        [
            os.path.join(input_dir, f"{writer}_binary_metadata.json"),
            os.path.join(input_dir, f"vectors_db_binary_{writer}.npy"),
        ]
    )

    if binary_fail_count > 0:
        print(
            "\nSome binary index types failed to deserialize/search. "
            "Malformed fourcc indicates extra non-backward-compatible bytes "
            "being written."
        )
        for result in binary_results:
            if result["status"] == "failed":
                print(
                    f"  - {result['index_key']}: "
                    f"{result.get('error', 'Unknown error')}"
                )
        return 1

    print("\nAll binary index types deserialized and searched successfully")
    print(
        f"Binary index serialization compatibility verified: "
        f"{writer} → {reader}"
    )
    return 0


def write_metadata(
    output_dir: str,
    filename: str,
    build_type: str,
    d: int,
    nb: int,
    results: list,
) -> None:
    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count

    metadata = {
        "build_type": build_type,
        "dimension": d,
        "num_db_vectors": nb,
        "faiss_version": faiss.__version__,
        "total_tests": len(INDEX_TYPES),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
    }

    metadata_path = os.path.join(output_dir, filename)
    print(f"\nWriting metadata to: {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def read_metadata(input_dir: str, filename: str) -> dict:
    metadata_path = os.path.join(input_dir, filename)
    print(f"\nReading metadata from: {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def print_metadata(metadata) -> None:
    print(f"  Build type: {metadata['build_type']}")
    print(f"  Dimension: {metadata['dimension']}")
    print(f"  Database vectors: {metadata['num_db_vectors']}")
    print(f"  Faiss version: {metadata['faiss_version']}")
    print(f"  Total tests: {metadata['total_tests']}")


def cleanup_files(file_paths: list[str]) -> None:
    print("\nCleaning up test artifacts...")
    for path in file_paths:
        if os.path.exists(path):
            print(f"Deleting: {path}")
            os.remove(path)
