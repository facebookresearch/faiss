#!/usr/bin/env -S grimaldi --kernel bento_kernel_faiss
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# fmt: off
# flake8: noqa


""":md
# Serializing codes separately, with IndexLSH and IndexPQ

Let's say, for example, you have a few vector embeddings per user
and want to shard a flat index by user so you can re-use the same LSH or PQ method
 for all users but store each user's codes independently.


"""

""":py"""
import faiss
import numpy as np

""":py"""
d = 768
n = 1_000
ids = np.arange(n).astype('int64')
training_data = np.random.rand(n, d).astype('float32')

""":py"""
def read_ids_codes():
    try:
        return np.load("/tmp/ids.npy"), np.load("/tmp/codes.npy")
    except FileNotFoundError:
        return None, None


def write_ids_codes(ids, codes):
    np.save("/tmp/ids.npy", ids)
    np.save("/tmp/codes.npy", codes.reshape(len(ids), -1))


def write_template_index(template_index):
    faiss.write_index(template_index, "/tmp/template.index")


def read_template_index_instance():
    return faiss.read_index("/tmp/template.index")

""":md
## IndexLSH: separate codes

The first half of this notebook demonstrates how to store LSH codes. Unlike PQ, LSH does not require training. In fact, it's compression method, a random projections matrix, is deterministic on construction based on a random seed value that's [hardcoded](https://github.com/facebookresearch/faiss/blob/2c961cc308ade8a85b3aa10a550728ce3387f625/faiss/IndexLSH.cpp#L35).
"""

""":py"""
nbits = 1536

""":py"""
# demonstrating encoding is deterministic

codes = []
database_vector_float32 = np.random.rand(1, d).astype(np.float32)
for i in range(10):
    index = faiss.IndexIDMap2(faiss.IndexLSH(d, nbits))
    code = index.index.sa_encode(database_vector_float32)
    codes.append(code)

for i in range(1, 10):
    assert np.array_equal(codes[0], codes[i])

""":py"""
# new database vector

ids, codes = read_ids_codes()
database_vector_id, database_vector_float32 = max(ids) + 1 if ids is not None else 1, np.random.rand(1, d).astype(np.float32)
index = faiss.IndexIDMap2(faiss.IndexLSH(d, nbits))

code = index.index.sa_encode(database_vector_float32)

if ids is not None and codes is not None:
    ids = np.concatenate((ids, [database_vector_id]))
    codes = np.vstack((codes, code))
else:
    ids = np.array([database_vector_id])
    codes = np.array([code])

write_ids_codes(ids, codes)

""":py '2840581589434841'"""
# then at query time

query_vector_float32 = np.random.rand(1, d).astype(np.float32)
index = faiss.IndexIDMap2(faiss.IndexLSH(d, nbits))
ids, codes = read_ids_codes()

index.add_sa_codes(codes, ids)

index.search(query_vector_float32, k=5)

""":py"""
!rm /tmp/ids.npy /tmp/codes.npy

""":md
## IndexPQ: separate codes from codebook

The second half of this notebook demonstrates how to separate serializing and deserializing the PQ codebook
 (via faiss.write_index for IndexPQ) independently of the vector codes. For example, in the case
 where you have a few vector embeddings per user and want to shard the flat index by user you 
 can re-use the same PQ method for all users but store each user's codes independently. 

"""

""":py"""
M = d//8
nbits = 8

""":py"""
# at train time
template_index = faiss.index_factory(d, f"IDMap2,PQ{M}x{nbits}")
template_index.train(training_data)
write_template_index(template_index)

""":py"""
# New database vector

index = read_template_index_instance()
ids, codes = read_ids_codes()
database_vector_id, database_vector_float32 = max(ids) + 1 if ids is not None else 1, np.random.rand(1, d).astype(np.float32)

code = index.index.sa_encode(database_vector_float32)

if ids is not None and codes is not None:
    ids = np.concatenate((ids, [database_vector_id]))
    codes = np.vstack((codes, code))
else:
    ids = np.array([database_vector_id])
    codes = np.array([code])

write_ids_codes(ids, codes)

""":py '1858280061369209'"""
# then at query time
query_vector_float32 = np.random.rand(1, d).astype(np.float32)
id_wrapper_index = read_template_index_instance()
ids, codes = read_ids_codes()

id_wrapper_index.add_sa_codes(codes, ids)

id_wrapper_index.search(query_vector_float32, k=5)

""":py"""
!rm /tmp/ids.npy /tmp/codes.npy /tmp/template.index

""":md
## Comparing these methods

- methods: Flat, LSH, PQ
- vary cost: nbits, M for 1x, 2x, 4x, 8x, 16x, 32x compression
- measure: recall@1

We don't measure latency as the number of vectors per user shard is insignificant.

"""

""":py '2898032417027201'"""
n, d

""":py"""
database_vector_ids, database_vector_float32s = np.arange(n), np.random.rand(n, d).astype(np.float32)
query_vector_float32s = np.random.rand(n, d).astype(np.float32)

""":py"""
index = faiss.index_factory(d, "IDMap2,Flat")
index.add_with_ids(database_vector_float32s, database_vector_ids)
_, ground_truth_result_ids= index.search(query_vector_float32s, k=1)

""":py '857475336204238'"""
from dataclasses import dataclass

pq_m_nbits = (
    # 96 bytes
    (96, 8),
    (192, 4),
    # 192 bytes
    (192, 8),
    (384, 4),
    # 384 bytes
    (384, 8),
    (768, 4),
)
lsh_nbits = (768, 1536, 3072, 6144, 12288, 24576)


@dataclass
class Record:
    type_: str
    index: faiss.Index
    args: tuple
    recall: float


results = []

for m, nbits in pq_m_nbits:
    print("pq", m, nbits)
    index = faiss.index_factory(d, f"IDMap2,PQ{m}x{nbits}")
    index.train(training_data)
    index.add_with_ids(database_vector_float32s, database_vector_ids)
    _, result_ids = index.search(query_vector_float32s, k=1)
    recall = sum(result_ids == ground_truth_result_ids)
    results.append(Record("pq", index, (m, nbits), recall))

for nbits in lsh_nbits:
    print("lsh", nbits)
    index = faiss.IndexIDMap2(faiss.IndexLSH(d, nbits))
    index.add_with_ids(database_vector_float32s, database_vector_ids)
    _, result_ids = index.search(query_vector_float32s, k=1)
    recall = sum(result_ids == ground_truth_result_ids)
    results.append(Record("lsh", index, (nbits,), recall))

""":py '556918346720794'"""
import matplotlib.pyplot as plt
import numpy as np

def create_grouped_bar_chart(x_values, y_values_list, labels_list, xlabel, ylabel, title):
    num_bars_per_group = len(x_values)

    plt.figure(figsize=(12, 6))

    for x, y_values, labels in zip(x_values, y_values_list, labels_list):
        num_bars = len(y_values)
        bar_width = 0.08 * x
        bar_positions = np.arange(num_bars) * bar_width - (num_bars - 1) * bar_width / 2 + x

        bars = plt.bar(bar_positions, y_values, width=bar_width)

        for bar, label in zip(bars, labels):
            height = bar.get_height()
            plt.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom'
            )

    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x_values, labels=[str(x) for x in x_values])
    plt.tight_layout()
    plt.show()

# # Example usage:
# x_values = [1, 2, 4, 8, 16, 32]
# y_values_list = [
#     [2.5, 3.6, 1.8],
#     [3.0, 2.8],
#     [2.5, 3.5, 4.0, 1.0],
#     [4.2],
#     [3.0, 5.5, 2.2],
#     [6.0, 4.5]
# ]
# labels_list = [
#     ['A1', 'B1', 'C1'],
#     ['A2', 'B2'],
#     ['A3', 'B3', 'C3', 'D3'],
#     ['A4'],
#     ['A5', 'B5', 'C5'],
#     ['A6', 'B6']
# ]

# create_grouped_bar_chart(x_values, y_values_list, labels_list, "x axis", "y axis", "title")

""":py '1630106834206134'"""
# x-axis: compression ratio
# y-axis: recall@1

from collections import defaultdict

x = defaultdict(list)
x[1].append(("flat", 1.00))
for r in results:
    y_value = r.recall[0] / n
    x_value = int(d * 4 / r.index.sa_code_size())
    label = None
    if r.type_ == "pq":
        label = f"PQ{r.args[0]}x{r.args[1]}"
    if r.type_ == "lsh":
        label = f"LSH{r.args[0]}"
    x[x_value].append((label, y_value))

x_values = sorted(list(x.keys()))
create_grouped_bar_chart(
    x_values,
    [[e[1] for e in x[x_value]] for x_value in x_values],
    [[e[0] for e in x[x_value]] for x_value in x_values],
    "compression ratio",
    "recall@1  q=1,000 queries",
    "recall@1 for a database of n=1,000 d=768 vectors",
)
