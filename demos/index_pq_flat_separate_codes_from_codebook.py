#!/usr/bin/env -S grimaldi --kernel bento_kernel_faiss
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off
# flake8: noqa


""":md
# IndexPQ: separate codes from codebook

This notebook demonstrates how to separate serializing and deserializing the PQ codebook
 (via faiss.write_index for IndexPQ) independently of the vector codes. For example, in the case
 where you have a few vector embeddings per user and want to shard the flat index by user you 
 can re-use the same PQ method for all users but store each user's codes independently. 

"""

""":py"""
import faiss
import numpy as np

""":py"""
d = 768
n = 10000
ids = np.arange(n).astype('int64')
training_data = np.random.rand(n, d).astype('float32')
M = d//8
nbits = 8

""":py"""
def read_ids_codes():
    try:
        return np.load("/tmp/ids.npy"), np.load("/tmp/codes.npy")
    except FileNotFoundError:
        return None, None


def write_ids_codes(ids, codes):
    # print(ids, codes)
    np.save("/tmp/ids.npy", ids)
    np.save("/tmp/codes.npy", codes.reshape(len(ids), -1))


def write_template_index(template_index):
    faiss.write_index(template_index, "/tmp/template.index")


def read_template_index_instance():
    pq_index = faiss.read_index("/tmp/template.index")
    return pq_index, faiss.IndexIDMap2(pq_index)

""":py"""
# at train time

template_index = faiss.IndexPQ(d, M, nbits)
template_index.train(training_data)
write_template_index(template_index)

""":py"""
# New database vector

template_instance_index, id_wrapper_index = read_template_index_instance()
database_vector_id, database_vector_float32 = np.int64(
    np.random.rand() * 10000
), np.random.rand(1, d).astype("float32")
ids, codes = read_ids_codes()
# print(ids, codes)
code = template_instance_index.sa_encode(database_vector_float32)
if ids is not None and codes is not None:
    ids = np.concatenate((ids, [database_vector_id]))
    codes = np.vstack((codes, code))
else:
    ids = np.array([database_vector_id])
    codes = np.array([code])
write_ids_codes(ids, codes)

""":py '1545041403561975'"""
# then at query time
query_vector_float32 = np.random.rand(1, d).astype("float32")
template_index_instance, id_wrapper_index = read_template_index_instance()
ids, codes = read_ids_codes()

for code in codes:
    for c in code:
        template_index_instance.codes.push_back(int(c))
template_index_instance.ntotal = len(codes)
for i in ids:
    id_wrapper_index.id_map.push_back(int(i))

id_wrapper_index.search(query_vector_float32, k=5)

""":py"""
!rm /tmp/ids.npy /tmp/codes.npy /tmp/template.index
