# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import re


def get_code_size(d, indexkey):
    """ size of one vector in an index in dimension d
    constructed with factory string indexkey"""

    if indexkey == "Flat":
        return d * 4

    if indexkey.endswith(",RFlat"):
        return d * 4 + get_code_size(d, indexkey[:-len(",RFlat")])

    mo = re.match("IVF\\d+(_HNSW32)?,(.*)$", indexkey)
    if mo:
        return get_code_size(d, mo.group(2))

    mo = re.match("IVF\\d+\\(.*\\)?,(.*)$", indexkey)
    if mo:
        return get_code_size(d, mo.group(1))

    mo = re.match("IMI\\d+x2,(.*)$", indexkey)
    if mo:
        return get_code_size(d, mo.group(1))

    mo = re.match("(.*),Refine\\((.*)\\)$", indexkey)
    if mo:
        return get_code_size(d, mo.group(1)) + get_code_size(d, mo.group(2))

    mo = re.match('PQ(\\d+)x(\\d+)(fs|fsr)?$', indexkey)
    if mo:
        return (int(mo.group(1)) * int(mo.group(2)) + 7) // 8

    mo = re.match('PQ(\\d+)\\+(\\d+)$', indexkey)
    if mo:
        return (int(mo.group(1)) + int(mo.group(2)))

    mo = re.match('PQ(\\d+)$', indexkey)
    if mo:
        return int(mo.group(1))

    if indexkey == "HNSW32" or indexkey == "HNSW32,Flat":
        return d * 4 + 64 * 4 # roughly

    if indexkey == 'SQ8':
        return d
    elif indexkey == 'SQ4':
        return (d + 1) // 2
    elif indexkey == 'SQ6':
        return (d * 6 + 7) // 8
    elif indexkey == 'SQfp16':
        return d * 2

    mo = re.match('PCAR?(\\d+),(.*)$', indexkey)
    if mo:
        return get_code_size(int(mo.group(1)), mo.group(2))
    mo = re.match('OPQ\\d+_(\\d+),(.*)$', indexkey)
    if mo:
        return get_code_size(int(mo.group(1)), mo.group(2))
    mo = re.match('OPQ\\d+,(.*)$', indexkey)
    if mo:
        return get_code_size(d, mo.group(1))
    mo = re.match('RR(\\d+),(.*)$', indexkey)
    if mo:
        return get_code_size(int(mo.group(1)), mo.group(2))
    raise RuntimeError("cannot parse " + indexkey)


def get_hnsw_M(index):
    return index.hnsw.cum_nneighbor_per_level.at(1) // 2


def reverse_index_factory(index):
    """
    attempts to get the factory string the index was built with
    """
    index = faiss.downcast_index(index)
    if isinstance(index, faiss.IndexFlat):
        return "Flat"
    elif isinstance(index, faiss.IndexIVF):
        quantizer = faiss.downcast_index(index.quantizer)

        if isinstance(quantizer, faiss.IndexFlat):
            prefix = f"IVF{index.nlist}"
        elif isinstance(quantizer, faiss.MultiIndexQuantizer):
            prefix = f"IMI{quantizer.pq.M}x{quantizer.pq.nbits}"
        elif isinstance(quantizer, faiss.IndexHNSW):
            prefix = f"IVF{index.nlist}_HNSW{get_hnsw_M(quantizer)}"
        else:
            prefix = f"IVF{index.nlist}({reverse_index_factory(quantizer)})"

        if isinstance(index, faiss.IndexIVFFlat):
            return prefix + ",Flat"
        if isinstance(index, faiss.IndexIVFScalarQuantizer):
            return prefix + ",SQ8"
        if isinstance(index, faiss.IndexIVFPQ):
            return prefix + f",PQ{index.pq.M}x{index.pq.nbits}"
        if isinstance(index, faiss.IndexIVFPQFastScan):
            return prefix + f",PQ{index.pq.M}x{index.pq.nbits}fs"

    elif isinstance(index, faiss.IndexPreTransform):
        if index.chain.size() != 1:
            raise NotImplementedError()
        vt = faiss.downcast_VectorTransform(index.chain.at(0))
        if isinstance(vt, faiss.OPQMatrix):
            prefix = f"OPQ{vt.M}_{vt.d_out}"
        elif isinstance(vt, faiss.ITQTransform):
            prefix = f"ITQ{vt.itq.d_out}"
        elif isinstance(vt, faiss.PCAMatrix):
            assert vt.eigen_power == 0
            prefix = "PCA" + ("R" if vt.random_rotation else "") + str(vt.d_out)
        else:
            raise NotImplementedError()
        return f"{prefix},{reverse_index_factory(index.index)}"

    elif isinstance(index, faiss.IndexHNSW):
        return f"HNSW{get_hnsw_M(index)}"

    elif isinstance(index, faiss.IndexRefine):
        return f"{reverse_index_factory(index.base_index)},Refine({reverse_index_factory(index.refine_index)})"

    elif isinstance(index, faiss.IndexPQFastScan):
        return f"PQ{index.pq.M}x{index.pq.nbits}fs"

    elif isinstance(index, faiss.IndexPQ):
        return f"PQ{index.pq.M}x{index.pq.nbits}"

    elif isinstance(index, faiss.IndexLSH):
        return "LSH" + ("r" if index.rotate_data else "") + ("t" if index.train_thresholds else "")

    elif isinstance(index, faiss.IndexScalarQuantizer):
        sqtypes = {
            faiss.ScalarQuantizer.QT_8bit: "8",
            faiss.ScalarQuantizer.QT_4bit: "4",
            faiss.ScalarQuantizer.QT_6bit: "6",
            faiss.ScalarQuantizer.QT_fp16: "fp16",
        }
        return f"SQ{sqtypes[index.sq.qtype]}"

    raise NotImplementedError()
