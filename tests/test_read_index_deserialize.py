# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deserialization validation tests for crafted and corrupted index
payloads."""

import contextlib
import struct
import sys
import unittest

import numpy as np

import faiss

# struct format codes for the C++ types that the faiss writer emits.
INT = "i"
INT64 = "q"
SIZE_T = "Q"
BOOL = "?"
FLOAT = "f"
DOUBLE = "d"
CHAR = "b"
UINT8 = "B"

SIZE_MAX = 2**64 - 1
INT64_MAX = 2**63 - 1

SVS_ENABLED = hasattr(faiss, "IndexSVSVamana")


# -----------------------------------------------------------------------
# Byte-buffer helpers that mirror the faiss writer layout.
# -----------------------------------------------------------------------


def push_val(buf, fmt, val):
    """Append a scalar value in little-endian format, matching WRITE1."""
    buf += struct.pack("<" + fmt, val)


def push_vector(buf, fmt, vec):
    """Append a WRITEVECTOR-formatted vector (size_t length prefix followed
    by raw element data)."""
    vec = list(vec)
    push_val(buf, SIZE_T, len(vec))
    buf += struct.pack("<%d%s" % (len(vec), fmt), *vec)


def push_fourcc(buf, s):
    """Append a fourcc string. The writer stores it as the uint32
    x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24, which is the same bytes in
    little-endian order."""
    assert len(s) == 4
    buf += s.encode("ascii")


def push_index_header(buf, d, ntotal, is_trained=True, metric_type=1):
    """Append write_index_header fields (metric_type 1 is L2)."""
    push_val(buf, INT, d)
    push_val(buf, INT64, ntotal)
    dummy = 1 << 20
    push_val(buf, INT64, dummy)
    push_val(buf, INT64, dummy)
    push_val(buf, BOOL, is_trained)
    push_val(buf, INT, metric_type)


def push_pq(buf, d, M, nbits, centroids=()):
    """Append write_ProductQuantizer fields (d, M, nbits, centroids vec)."""
    push_val(buf, SIZE_T, d)
    push_val(buf, SIZE_T, M)
    push_val(buf, SIZE_T, nbits)
    push_vector(buf, FLOAT, centroids)


def push_binary_index_header(buf, d, ntotal, is_trained=True, metric_type=1):
    """Append read_index_binary_header fields:
    int d, int code_size, idx_t ntotal, bool is_trained, int metric_type."""
    code_size = d // 8
    push_val(buf, INT, d)
    push_val(buf, INT, code_size)
    push_val(buf, INT64, ntotal)
    push_val(buf, BOOL, is_trained)
    push_val(buf, INT, metric_type)


def push_empty_direct_map(buf):
    """Append empty direct_map bytes (NoMap type + empty array)."""
    push_val(buf, CHAR, 0)  # DirectMap::NoMap
    push_val(buf, SIZE_T, 0)  # empty array


def push_null_invlists(buf):
    """Append null inverted lists ("il00" fourcc)."""
    push_fourcc(buf, "il00")


def push_minimal_binary_flat(buf, d):
    """Append a minimal IndexBinaryFlat ("IBxF") with ntotal=0."""
    push_fourcc(buf, "IBxF")
    push_binary_index_header(buf, d, 0)
    push_vector(buf, UINT8, [])  # empty xb


def push_additive_quantizer(buf, d, M, nbits, codebook_d=0):
    """Append a minimal AdditiveQuantizer (d, M, nbits, is_trained,
    codebooks, search_type=ST_decompress, norm_min, norm_max).
    Writes zero-filled codebooks of size codebook_d * total_codebook_size so
    that the deserialization codebooks-size validation passes.
    For standalone quantizers codebook_d == d. For ProductAdditiveQuantizer
    codebook_d == d / nsplits. Pass codebook_d = 0 (default) to use d."""
    if codebook_d == 0:
        codebook_d = d
    total_codebook_size = sum(1 << nb for nb in nbits)
    push_val(buf, SIZE_T, d)
    push_val(buf, SIZE_T, M)
    push_vector(buf, SIZE_T, nbits)
    push_val(buf, BOOL, True)  # is_trained
    push_vector(buf, FLOAT, [0.0] * (codebook_d * total_codebook_size))
    push_val(buf, INT, 0)  # search_type = ST_decompress
    push_val(buf, FLOAT, 0.0)  # norm_min
    push_val(buf, FLOAT, 1.0)  # norm_max


def push_residual_quantizer(buf, d, M, nbits):
    """Append a minimal ResidualQuantizer (AdditiveQuantizer + train_type +
    max_beam_size). Uses Skip_codebook_tables to avoid
    compute_codebook_tables on empty codebooks."""
    push_additive_quantizer(buf, d, M, nbits)
    push_val(buf, INT, 2048)  # train_type = Skip_codebook_tables
    push_val(buf, INT, 1)  # max_beam_size


def push_local_search_quantizer(buf, d, M, nbits):
    """Append a minimal LocalSearchQuantizer (AdditiveQuantizer + K +
    train_iters + encode_ils_iters + train_ils_iters + icm_iters + p + lambd
    + chunk_size + random_seed + nperts + update_codebooks_with_double)."""
    push_additive_quantizer(buf, d, M, nbits)
    push_val(buf, SIZE_T, 256)  # K
    push_val(buf, SIZE_T, 25)  # train_iters
    push_val(buf, SIZE_T, 8)  # encode_ils_iters
    push_val(buf, SIZE_T, 8)  # train_ils_iters
    push_val(buf, SIZE_T, 4)  # icm_iters
    push_val(buf, FLOAT, 0.5)  # p
    push_val(buf, FLOAT, 0.0)  # lambd
    push_val(buf, SIZE_T, 0)  # chunk_size
    push_val(buf, INT, 123)  # random_seed
    push_val(buf, SIZE_T, 4)  # nperts
    push_val(buf, BOOL, False)  # update_codebooks_with_double


def push_minimal_flat(buf, d, ntotal=0):
    """Append a minimal IndexFlatL2 ("IxF2"). Uses the WRITEXBVECTOR format
    (size_t num_floats prefix, then raw data)."""
    push_fourcc(buf, "IxF2")
    push_index_header(buf, d, ntotal)
    num_floats = ntotal * d
    push_val(buf, SIZE_T, num_floats)
    buf += bytes(num_floats * 4)


def push_minimal_hnsw(buf, ntotal):
    """Append a minimal valid HNSW structure. All nodes are at level 1 with
    zero neighbors, which passes validate_HNSW."""
    push_vector(buf, DOUBLE, [])  # assign_probas
    # cum_nneighbor_per_level: 0 cumulative neighbors at each level
    push_vector(buf, INT, [0, 0])
    # levels: one entry per node, all at level 1 (1-indexed in HNSW)
    push_vector(buf, INT, [1] * ntotal)
    push_vector(buf, SIZE_T, [0] * (ntotal + 1))  # offsets
    push_vector(buf, INT, [])  # neighbors
    push_val(buf, INT, 0 if ntotal > 0 else -1)  # entry_point
    push_val(buf, INT, 0)  # max_level
    push_val(buf, INT, 40)  # efConstruction
    push_val(buf, INT, 16)  # efSearch
    push_val(buf, INT, 1)  # upper_beam (deprecated, always 1)


def push_hnsw_with_neighbors(
    buf, ntotal, neighbors_per_node, neighbors, entry_point=0
):
    """Append an HNSW structure with explicit neighbor data. All nodes are
    at level 1, cum_nneighbor_per_level = {0, neighbors_per_node}. Callers
    supply the full neighbors vector (ntotal * neighbors_per_node entries,
    using -1 for empty slots)."""
    push_vector(buf, DOUBLE, [])  # assign_probas
    push_vector(buf, INT, [0, neighbors_per_node])
    push_vector(buf, INT, [1] * ntotal)  # levels
    # offsets: each node occupies neighbors_per_node slots
    offsets = [i * neighbors_per_node for i in range(ntotal + 1)]
    push_vector(buf, SIZE_T, offsets)
    push_vector(buf, INT, neighbors)
    push_val(buf, INT, entry_point)
    push_val(buf, INT, 0)  # max_level
    push_val(buf, INT, 40)  # efConstruction
    push_val(buf, INT, 16)  # efSearch
    push_val(buf, INT, 1)  # upper_beam (deprecated)


def push_nndescent(buf, ntotal, d, K, has_built, final_graph):
    """Append NNDescent fields."""
    push_val(buf, INT, ntotal)
    push_val(buf, INT, d)
    push_val(buf, INT, K)
    push_val(buf, INT, 10)  # S
    push_val(buf, INT, 10)  # R
    push_val(buf, INT, 10)  # L
    push_val(buf, INT, 1)  # iter
    push_val(buf, INT, 10)  # search_L
    push_val(buf, INT, 42)  # random_seed
    push_val(buf, BOOL, has_built)
    push_vector(buf, INT, final_graph)


def push_rabitq(buf, d):
    """Append a minimal RaBitQuantizer (single-bit format)."""
    push_val(buf, SIZE_T, d)
    push_val(buf, SIZE_T, 1)  # code_size
    push_val(buf, INT, 1)  # metric_type (L2)


def push_rabitq_multibit(buf, d):
    """Append a minimal RaBitQuantizer (multi-bit format)."""
    push_rabitq(buf, d)
    push_val(buf, SIZE_T, 1)  # nb_bits


def push_ivf_header(buf, d):
    """Append an IVF header (index_header + nlist + nprobe + flat quantizer
    + empty direct_map)."""
    push_index_header(buf, d, 0)
    push_val(buf, SIZE_T, 1)  # nlist
    push_val(buf, SIZE_T, 1)  # nprobe
    push_minimal_flat(buf, d)
    push_empty_direct_map(buf)


def pack_bits(nbytes, fields):
    """Pack (value, nbit) fields with faiss.BitstringWriter."""
    bitbuf = np.zeros(nbytes, dtype="uint8")
    wr = faiss.BitstringWriter(faiss.swig_ptr(bitbuf), bitbuf.size)
    for value, nbit in fields:
        wr.write(value, nbit)
    return bitbuf.tolist()


def make_idmap_payload(fourcc, outer_ntotal, inner_ntotal, id_map_size):
    """Serialized IxMp/IxM2 payload. The inner IxF2 carries codes sized to
    its own ntotal, so it passes the inner codes-vs-ntotal check and the
    IDMap-vs-inner-ntotal check is exercised in isolation."""
    d = 4
    buf = bytearray()
    push_fourcc(buf, fourcc)
    push_index_header(buf, d, outer_ntotal)
    push_fourcc(buf, "IxF2")
    push_index_header(buf, d, inner_ntotal)
    # IndexFlat codes use WRITEXBVECTOR: a size_t prefix of
    # (byte_count / 4) followed by byte_count bytes.
    byte_count = inner_ntotal * d * 4
    push_val(buf, SIZE_T, byte_count // 4)
    buf += bytes(byte_count)
    push_vector(buf, INT64, [0] * id_map_size)
    return buf


def push_flat_panorama(buf, d, n_levels, batch_size, ntotal):
    """"IxFP" format: fourcc + d(int) + n_levels + batch_size +
    ntotal(int64) + is_trained(bool) + empty codes vec + empty cum_sums
    vec."""
    push_fourcc(buf, "IxFP")
    push_val(buf, INT, d)
    push_val(buf, SIZE_T, n_levels)
    push_val(buf, SIZE_T, batch_size)
    push_val(buf, INT64, ntotal)
    push_val(buf, BOOL, True)
    push_vector(buf, UINT8, [])  # codes
    push_vector(buf, FLOAT, [])  # cum_sums


# -----------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------


def make_reader(buf):
    reader = faiss.VectorIOReader()
    faiss.copy_array_to_vector(
        np.frombuffer(bytes(buf), dtype="uint8"), reader.data
    )
    return reader


def read_index(buf):
    return faiss.deserialize_index(np.frombuffer(bytes(buf), dtype="uint8"))


def read_index_binary(buf):
    return faiss.deserialize_index_binary(
        np.frombuffer(bytes(buf), dtype="uint8")
    )


def read_vector_transform(buf):
    return faiss.read_VectorTransform(make_reader(buf))


def read_invlists(buf):
    return faiss.read_InvertedLists(make_reader(buf))


def serialize_index(index):
    return bytearray(faiss.serialize_index(index).tobytes())


@contextlib.contextmanager
def vector_byte_limit(value):
    old_limit = faiss.get_deserialization_vector_byte_limit()
    faiss.set_deserialization_vector_byte_limit(value)
    try:
        yield
    finally:
        faiss.set_deserialization_vector_byte_limit(old_limit)


@contextlib.contextmanager
def lattice_r2_limit(value):
    old_limit = faiss.get_deserialization_lattice_r2_limit()
    faiss.set_deserialization_lattice_r2_limit(value)
    try:
        yield
    finally:
        faiss.set_deserialization_lattice_r2_limit(old_limit)


def swig_default_construct(cls):
    """Call the swig default constructor of cls.

    For IndexIVFFlat and IndexBinaryIVF, cls() raises IndexError in the
    Python wrapper: add_ref_in_constructor(cls, 0) reads args[0], which the
    default constructor does not have.

    The low-level module of swigfaiss, swigfaiss_avx2, swigfaiss_sve, ... is
    _swigfaiss, _swigfaiss_avx2, _swigfaiss_sve, ..."""
    module = sys.modules[cls.__module__]
    swig_module = getattr(module, "_" + module.__name__.rpartition(".")[2])
    obj = cls.__new__(cls)
    name = cls.__name__
    getattr(swig_module, name + "_swiginit")(
        obj, getattr(swig_module, "new_" + name)()
    )
    return obj


def make_random_data(d, n, seed=42):
    return np.random.RandomState(seed).rand(n, d).astype("float32")


class FaissTestCase(unittest.TestCase):
    @contextlib.contextmanager
    def assertRaisesFaiss(self, expected_substr=None):
        """Python form of EXPECT_THROW(..., faiss::FaissException).

        swig maps faiss::FaissException to RuntimeError with e.what() as the
        message, std::bad_alloc to MemoryError, and any other std::exception
        to RuntimeError with a "C++ exception " prefix. The prefix check
        rejects a std::exception that is not a FaissException."""
        with self.assertRaises(RuntimeError) as cm:
            yield
        msg = str(cm.exception)
        self.assertFalse(msg.startswith("C++ exception "), msg)
        if expected_substr is not None:
            self.assertIn(expected_substr, msg)

    def expect_read_throws(self, buf, expected_substr=None):
        with self.assertRaisesFaiss(expected_substr):
            read_index(buf)

    def expect_binary_read_throws(self, buf, expected_substr=None):
        with self.assertRaisesFaiss(expected_substr):
            read_index_binary(buf)

    def expect_vt_read_throws(self, buf, expected_substr=None):
        with self.assertRaisesFaiss(expected_substr):
            read_vector_transform(buf)

    def expect_invlists_read_throws(self, buf, expected_substr=None):
        with self.assertRaisesFaiss(expected_substr):
            read_invlists(buf)


class TestReadIndexDeserialize(FaissTestCase):
    def test_pq_with_m_zero_divide_by_zero(self):
        # ProductQuantizer with M=0 causes divide-by-zero in
        # set_derived_values(). read_ProductQuantizer validates M > 0.
        buf = bytearray()
        push_fourcc(buf, "Imiq")
        push_index_header(buf, 4, 0)
        push_pq(buf, 4, 0, 8)
        self.expect_read_throws(buf)

    def test_index_header_invalid_bool_encoding_is_trained(self):
        # A non-{0,1} byte read into a bool is UB; READ1_BOOL must reject it.
        buf = bytearray()
        push_fourcc(buf, "IxF2")
        push_index_header(buf, 4, 0)
        # is_trained sits at offset 4 (fourcc) + 4 (d) + 8 (ntotal) +
        # 8 (dummy) + 8 (dummy) = 32.
        is_trained_offset = 32
        buf[is_trained_offset] = 0x02
        self.expect_read_throws(buf, "invalid bool encoding")

    def test_additive_quantizer_nbits_size_mismatch(self):
        # AdditiveQuantizer with nbits.size() != M causes out-of-bounds
        # access on nbits in set_derived_values().
        buf = bytearray()
        push_fourcc(buf, "IPRf")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 10)  # M = 10
        push_vector(buf, SIZE_T, [8])  # nbits: 1 element, should be 10
        push_val(buf, BOOL, True)  # is_trained
        push_vector(buf, FLOAT, [])  # codebooks
        push_val(buf, INT, 0)  # search_type = ST_decompress
        push_val(buf, FLOAT, 0.0)  # norm_min
        push_val(buf, FLOAT, 1.0)  # norm_max
        self.expect_read_throws(buf, "nbits size")

    def test_residual_quantizer_old_nbits_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxRQ")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 5)  # M = 5
        push_vector(buf, SIZE_T, [8, 8])  # nbits: 2 elements, should be 5
        self.expect_read_throws(buf, "nbits size")

    def test_pq_centroids_overflow(self):
        # d * (1 << 24) overflows size_t; mul_no_overflow detects it.
        buf = bytearray()
        push_fourcc(buf, "Imiq")
        huge_d = 1 << 48
        # The header d is (int)huge_d: the low 32 bits, which are 0.
        push_index_header(buf, huge_d & 0xFFFFFFFF, 0)
        push_pq(buf, huge_d, 1, 24)  # M=1 so d % M == 0
        self.expect_read_throws(buf)

    def test_readvector_byte_limit(self):
        # READVECTOR rejects a vector whose byte size reaches the
        # deserialization byte limit. "LTra" (LinearTransform) format:
        # fourcc + have_bias + READVECTOR(A) + READVECTOR(b) + d_in + d_out
        # + is_trained. A contains 1024 floats = 4096 bytes.
        # READVECTOR check: size < limit / sizeof(float)
        d_in = 32
        d_out = 32
        n_elements = d_in * d_out
        buf = bytearray()
        push_fourcc(buf, "LTra")
        push_val(buf, BOOL, False)  # have_bias
        push_vector(buf, FLOAT, [0.0] * n_elements)  # A
        push_vector(buf, FLOAT, [])  # b
        push_val(buf, INT, d_in)
        push_val(buf, INT, d_out)
        push_val(buf, BOOL, True)  # is_trained

        # Exactly at the boundary: the check is strict less-than.
        with vector_byte_limit(n_elements * 4):
            with self.assertRaisesFaiss():
                read_vector_transform(buf)
        # One element above the boundary passes.
        with vector_byte_limit((n_elements + 1) * 4):
            read_vector_transform(buf)

    def test_pq_centroids_byte_limit(self):
        # d=8, M=1, nbits=8 -> ksub=256, centroids = 8*256 = 2048 floats.
        # PQ check: n < limit / sizeof(float), where n = d * ksub = 2048.
        d = 8
        M = 1
        nbits = 8
        n_elements = d * (1 << nbits)
        buf = bytearray()
        push_fourcc(buf, "Imiq")
        push_index_header(buf, d, 0)
        push_pq(buf, d, M, nbits, [0.0] * n_elements)

        with vector_byte_limit(n_elements * 4):
            with self.assertRaisesFaiss():
                read_index(buf)
        with vector_byte_limit((n_elements + 1) * 4):
            read_index(buf)

    def test_index_lattice_nsq_zero_divide_by_zero(self):
        # nsq=0 causes divide-by-zero in the constructor (dsq = d / nsq).
        # "IxLa": fourcc + d + nsq + scale_nbit + r2 + index_header
        #         + READVECTOR(trained)
        buf = bytearray()
        push_fourcc(buf, "IxLa")
        push_val(buf, INT, 16)  # d
        push_val(buf, INT, 0)  # nsq = 0
        push_val(buf, INT, 4)  # scale_nbit
        push_val(buf, INT, 14)  # r2
        self.expect_read_throws(buf)

    def test_index_lattice_d_not_divisible_by_nsq(self):
        buf = bytearray()
        push_fourcc(buf, "IxLa")
        push_val(buf, INT, 17)  # d = 17 (not divisible by nsq=4)
        push_val(buf, INT, 4)  # nsq
        push_val(buf, INT, 4)  # scale_nbit
        push_val(buf, INT, 14)  # r2
        self.expect_read_throws(buf, "divisible by nsq")

    def test_index_lattice_r2_zero(self):
        # r2=0 causes heap-buffer-overflow in the ZnSphereCodecRec ctor.
        buf = bytearray()
        push_fourcc(buf, "IxLa")
        push_val(buf, INT, 4)  # d
        push_val(buf, INT, 1)  # nsq
        push_val(buf, INT, 4)  # scale_nbit
        push_val(buf, INT, 0)  # r2 = 0
        self.expect_read_throws(buf, "r2")

    def test_index_lattice_dsq_not_power_of_2(self):
        # d/nsq that is not a power of 2 >= 2 causes heap-buffer-overflow
        # in the ZnSphereCodecRec ctor. d=3, nsq=1 -> dsq=3.
        buf = bytearray()
        push_fourcc(buf, "IxLa")
        push_val(buf, INT, 3)  # d
        push_val(buf, INT, 1)  # nsq
        push_val(buf, INT, 4)  # scale_nbit
        push_val(buf, INT, 1)  # r2
        self.expect_read_throws(buf, "power of 2")

    def test_index_lattice_dsq_one(self):
        # d=1, nsq=1 -> dsq=1 (too small, causes cache_level=-1)
        buf = bytearray()
        push_fourcc(buf, "IxLa")
        push_val(buf, INT, 1)  # d
        push_val(buf, INT, 1)  # nsq
        push_val(buf, INT, 4)  # scale_nbit
        push_val(buf, INT, 1)  # r2
        self.expect_read_throws(buf, "power of 2")

    def test_index_lattice_r2_too_large(self):
        # r2 too large causes a timeout in the ZnSphereCodecRec ctor
        # (exponential codeword count in the decode cache). dsq=16
        # (log2_dim=4, cache_level=3) with r2=73 would take well over a
        # minute of CPU time if not rejected.
        buf = bytearray()
        push_fourcc(buf, "IxLa")
        push_val(buf, INT, 16)  # d
        push_val(buf, INT, 1)  # nsq
        push_val(buf, INT, 4)  # scale_nbit
        push_val(buf, INT, 73)  # r2
        self.expect_read_throws(buf, "decode cache")

    def test_index_lattice_r2_configurable_limit(self):
        # deserialization_lattice_r2_limit caps the ZnSphereCodecRec
        # decode-cache build cost for untrusted payloads, separately from
        # the in-codec memory cap. d=4, nsq=1, scale_nbit=4 (dsq=4) passes
        # the structural checks. The cap is "r2 > limit".
        d = 4

        def build(r2):
            buf = bytearray()
            push_fourcc(buf, "IxLa")
            push_val(buf, INT, d)
            push_val(buf, INT, 1)  # nsq
            push_val(buf, INT, 4)  # scale_nbit
            push_val(buf, INT, r2)
            push_index_header(buf, d, 0)
            push_vector(buf, FLOAT, [])  # untrained: empty trained vector
            return buf

        with lattice_r2_limit(2):
            # r2 == limit: the ZnSphereCodecRec ctor runs to completion.
            read_index(build(2))
            # r2 == limit + 1: rejected with the limit in the message.
            self.expect_read_throws(
                build(3), "deserialization_lattice_r2_limit"
            )

    def test_binary_unrecognized_fourcc(self):
        buf = bytearray()
        push_fourcc(buf, "ZZZZ")
        self.expect_binary_read_throws(buf)

    def test_binary_flat_xb_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IBxF")
        push_binary_index_header(buf, 16, 1)
        push_vector(buf, UINT8, [])  # empty xb, ntotal=1 expects 2 bytes
        self.expect_binary_read_throws(buf, "xb.size()")

    def test_binary_multi_hash_storage_cast_failure(self):
        buf = bytearray()
        push_fourcc(buf, "IBHm")
        push_binary_index_header(buf, 16, 99)
        push_minimal_binary_flat(buf, 16)  # ntotal=0, mismatches 99
        self.expect_binary_read_throws(buf)

    def test_binary_ivf_inv_list_nlist_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IBwF")
        push_binary_index_header(buf, 16, 0)
        push_val(buf, SIZE_T, 10)  # nlist = 10
        push_val(buf, SIZE_T, 1)  # nprobe
        push_minimal_binary_flat(buf, 16)
        push_empty_direct_map(buf)
        # ArrayInvertedLists: "ilar" + nlist + code_size + sizes
        push_fourcc(buf, "ilar")
        push_val(buf, SIZE_T, 5)  # nlist = 5 (mismatches IVF nlist=10)
        push_val(buf, SIZE_T, 2)  # code_size = d/8
        for _ in range(5):
            push_val(buf, SIZE_T, 0)
        self.expect_binary_read_throws(buf)

    def test_binary_hash_empty_invlist_buffer(self):
        # An empty hash invlist buffer with a non-zero entry count would be
        # dereferenced as null in read_binary_hash_invlists.
        buf = bytearray()
        push_fourcc(buf, "IBHh")
        push_binary_index_header(buf, 16, 0)
        push_val(buf, INT, 4)  # b
        push_val(buf, INT, 0)  # nflip
        push_val(buf, SIZE_T, 1)  # sz = 1 (non-zero)
        push_val(buf, INT, 8)  # il_nbit
        push_vector(buf, UINT8, [])  # empty buffer
        self.expect_binary_read_throws(buf, "binary hash invlists")

    def test_nsg_negative_r(self):
        buf = bytearray()
        push_fourcc(buf, "INSf")
        push_index_header(buf, 4, 0)
        push_val(buf, INT, 0)  # GK
        push_val(buf, CHAR, 0)  # build_type
        push_val(buf, INT, 10)  # nndescent_S
        push_val(buf, INT, 10)  # nndescent_R
        push_val(buf, INT, 10)  # nndescent_L
        push_val(buf, INT, 1)  # nndescent_iter
        push_val(buf, INT, 0)  # ntotal
        push_val(buf, INT, -1)  # R = -1 (invalid)
        self.expect_read_throws(buf, "invalid NSG R")

    def test_scalar_quantizer_invalid_qtype(self):
        buf = bytearray()
        push_fourcc(buf, "IxSQ")
        push_index_header(buf, 4, 0)
        push_val(buf, INT, 99)  # qtype = 99 (out of range)
        self.expect_read_throws(buf, "qtype")

    def test_product_additive_quantizer_zero_nsplits(self):
        buf = bytearray()
        push_fourcc(buf, "IxPR")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 1)  # M
        push_vector(buf, SIZE_T, [8])  # nbits
        push_val(buf, BOOL, True)  # is_trained
        # codebooks: d * total_codebook_size = 4 * 256 floats
        push_vector(buf, FLOAT, [0.0] * (4 * 256))
        push_val(buf, INT, 0)  # search_type = ST_decompress
        push_val(buf, FLOAT, 0.0)  # norm_min
        push_val(buf, FLOAT, 1.0)  # norm_max
        push_val(buf, SIZE_T, 0)  # nsplits = 0 (invalid)
        self.expect_read_throws(buf, "nsplits")

    def test_pre_transform_negative_chain_length(self):
        buf = bytearray()
        push_fourcc(buf, "IxPT")
        push_index_header(buf, 4, 0)
        push_val(buf, INT, -1)  # nt = -1 (invalid)
        self.expect_read_throws(buf, "chain length")

    def test_binary_multi_hash_zero_nhash(self):
        buf = bytearray()
        push_fourcc(buf, "IBHm")
        push_binary_index_header(buf, 16, 0)
        push_minimal_binary_flat(buf, 16)
        push_val(buf, INT, 4)  # b
        push_val(buf, INT, 0)  # nhash = 0 (invalid)
        self.expect_binary_read_throws(buf, "nhash")

    def test_binary_hash_b_zero(self):
        # Without the b > 0 check, BitstringReader::read(0) would silently
        # produce garbage hash values on every inverted-list entry.
        buf = bytearray()
        push_fourcc(buf, "IBHh")
        push_binary_index_header(buf, 16, 0)
        push_val(buf, INT, 0)  # b = 0 (invalid)
        self.expect_binary_read_throws(buf, "IndexBinaryHash b=")

    def test_binary_hash_negative_il_nbit(self):
        # A negative il_nbit would wrap to a huge size_t in the
        # bits-per-entry calculation and read out of bounds in
        # BitstringReader.
        buf = bytearray()
        push_fourcc(buf, "IBHh")
        push_binary_index_header(buf, 16, 0)
        push_val(buf, INT, 4)  # b
        push_val(buf, INT, 0)  # nflip
        push_val(buf, SIZE_T, 0)  # sz = 0
        push_val(buf, INT, -1)  # il_nbit = -1 (invalid)
        self.expect_binary_read_throws(buf, "il_nbit=")

    def test_binary_hash_il_nbit_zero_with_entries(self):
        # With il_nbit=0 and sz > 0, every inverted-list size would read as
        # 0 and corrupt the deserialized index.
        buf = bytearray()
        push_fourcc(buf, "IBHh")
        push_binary_index_header(buf, 16, 0)
        push_val(buf, INT, 4)  # b
        push_val(buf, INT, 0)  # nflip
        push_val(buf, SIZE_T, 1)  # sz = 1 (non-zero)
        push_val(buf, INT, 0)  # il_nbit = 0 (invalid when sz > 0)
        self.expect_binary_read_throws(buf, "il_nbit=")

    def test_binary_hash_b_exceeds_code_size(self):
        # A b larger than code_size*8 makes BitstringReader::read() access
        # past the code buffer (heap-buffer-overflow).
        buf = bytearray()
        push_fourcc(buf, "IBHh")
        push_binary_index_header(buf, 16, 0)
        # d=16 -> code_size=2 -> 16 bits available
        push_val(buf, INT, 17)  # b = 17
        self.expect_binary_read_throws(buf, "IndexBinaryHash b=")

    def test_binary_multi_hash_map_ilsz_exceeds_ntotal(self):
        # ilsz values that sum to more than ntotal would make the inner
        # loop read past the end of the BitstringReader buffer.
        # ntotal=4 -> id_bits=2, b=4, sz=2 entries.
        # Total bits = (b + id_bits) * sz + ntotal * id_bits = 20 -> 3 bytes
        # Entry 0: hash=0, ilsz=3 (3 ids follow)
        # Entry 1: hash=1, ilsz=3 -> total ilsz 6 > ntotal 4, throws here.
        b = 4
        id_bits = 2
        ntotal = 4
        sz = 2
        nbit = (b + id_bits) * sz + ntotal * id_bits
        bitbuf = pack_bits(
            (nbit + 7) // 8,
            [
                (0, b),
                (3, id_bits),
                (0, id_bits),
                (0, id_bits),
                (0, id_bits),
                (1, b),
                (3, id_bits),
            ],
        )

        buf = bytearray()
        push_fourcc(buf, "IBHm")
        push_binary_index_header(buf, 16, ntotal)
        # Nested IBxF storage with matching ntotal (code_size = 2).
        push_fourcc(buf, "IBxF")
        push_binary_index_header(buf, 16, ntotal)
        push_vector(buf, UINT8, [0] * (ntotal * 2))
        push_val(buf, INT, b)
        push_val(buf, INT, 1)  # nhash = 1
        push_val(buf, INT, 0)  # nflip
        # Multi hash map fields (1 map):
        push_val(buf, INT, id_bits)
        push_val(buf, SIZE_T, sz)
        push_vector(buf, UINT8, bitbuf)
        self.expect_binary_read_throws(buf, "would exceed ntotal")

    def test_binary_id_map_id_map_size_mismatch(self):
        # IBMp with id_map size != ntotal. Without the check,
        # construct_rev_map (IBM2) or search would use an inconsistent ID
        # mapping. Outer ntotal=2, id_map has 5 entries.
        buf = bytearray()
        push_fourcc(buf, "IBMp")
        push_binary_index_header(buf, 16, 2)
        push_fourcc(buf, "IBxF")
        push_binary_index_header(buf, 16, 2)
        push_vector(buf, UINT8, [0] * (2 * 2))  # ntotal * code_size
        push_vector(buf, INT64, [10, 20, 30, 40, 50])
        self.expect_binary_read_throws(buf, "id_map")

    def test_index_id_map2_id_map_size_mismatch(self):
        # IxM2 with id_map size != ntotal. Without the check,
        # construct_rev_map() reads past id_map bounds.
        buf = bytearray()
        push_fourcc(buf, "IxM2")
        push_index_header(buf, 4, 2)
        push_fourcc(buf, "IxF2")
        push_index_header(buf, 4, 2)
        # WRITEXBVECTOR: ntotal * d floats
        num_floats = 2 * 4
        push_val(buf, SIZE_T, num_floats)
        for _ in range(num_floats):
            push_val(buf, FLOAT, 0.0)
        push_vector(buf, INT64, [10, 20, 30, 40, 50])
        self.expect_read_throws(buf, "id_map")

    def test_block_inverted_lists_n_per_block_zero(self):
        # n_per_block=0 causes divide-by-zero in resize() and add_entries().
        buf = bytearray()
        push_fourcc(buf, "ilbl")
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 32)  # code_size
        push_val(buf, SIZE_T, 0)  # n_per_block = 0 (invalid)
        push_val(buf, SIZE_T, 128)  # block_size
        self.expect_invlists_read_throws(buf, "n_per_block")

    def test_block_inverted_lists_block_size_zero(self):
        # block_size=0 causes divide-by-zero in add_entries().
        buf = bytearray()
        push_fourcc(buf, "ilbl")
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 32)  # code_size
        push_val(buf, SIZE_T, 32)  # n_per_block
        push_val(buf, SIZE_T, 0)  # block_size = 0 (invalid)
        self.expect_invlists_read_throws(buf, "block_size")

    def test_block_inverted_lists_codes_size_mismatch(self):
        # A codes size inconsistent with ids size, n_per_block and
        # block_size would make search or add access past the codes buffer.
        buf = bytearray()
        push_fourcc(buf, "ilbl")
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 32)  # code_size
        push_val(buf, SIZE_T, 32)  # n_per_block
        push_val(buf, SIZE_T, 128)  # block_size
        # ids: 10 entries -> ceil(10/32) = 1 block -> codes = 128 bytes
        push_vector(buf, INT64, [0] * 10)
        push_vector(buf, UINT8, [0] * 64)  # wrong, should be 128
        self.expect_invlists_read_throws(buf, "codes size")

    def test_block_inverted_lists_codes_overflow(self):
        # n_block * block_size overflow. Without mul_no_overflow the product
        # wraps and the size comparison passes spuriously.
        buf = bytearray()
        push_fourcc(buf, "ilbl")
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 32)  # code_size
        push_val(buf, SIZE_T, 1)  # n_per_block
        push_val(buf, SIZE_T, SIZE_MAX)  # block_size
        # ids: 2 entries -> n_block = 2 -> 2 * SIZE_MAX overflows
        push_vector(buf, INT64, [0] * 2)
        push_vector(buf, UINT8, [])  # overflow is caught first
        self.expect_invlists_read_throws(buf, "overflow")

    def test_pq_centroids_size_mismatch(self):
        # PQ: d=4, M=2, nbits=8 -> ksub=256, expected centroids = 1024
        buf = bytearray()
        push_fourcc(buf, "Imiq")
        push_index_header(buf, 4, 0)
        push_pq(buf, 4, 2, 8, [0.0] * 100)
        self.expect_read_throws(buf, "centroids size")

    def test_sq_trained_size_mismatch(self):
        # For QT_4bit, expected trained.size() = 2 * d.
        buf = bytearray()
        push_fourcc(buf, "IxSQ")
        push_index_header(buf, 4, 0)
        push_val(buf, INT, faiss.ScalarQuantizer.QT_4bit)
        push_val(buf, INT, 0)  # rangestat
        push_val(buf, FLOAT, 0.0)  # rangestat_arg
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, FLOAT, [1.0, 2.0, 3.0])  # 3 floats, expected 8
        self.expect_read_throws(buf, "ScalarQuantizer trained size")

    def test_index_pq_codes_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxPq")
        push_index_header(buf, 4, 2)
        # PQ: d=4, M=2, nbits=8 -> code_size=2, ksub=256
        push_pq(buf, 4, 2, 8, [0.0] * (4 * 256))
        # codes should be ntotal * code_size = 4 bytes; provide 10
        push_vector(buf, UINT8, [0] * 10)
        push_val(buf, INT, 0)  # search_type
        push_val(buf, INT, 0)  # encode_signs
        push_val(buf, INT, 0)  # polysemous_ht
        self.expect_read_throws(buf, "codes.size()")

    def test_index_residual_quantizer_codes_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxRq")
        push_index_header(buf, 4, 2)
        push_residual_quantizer(buf, 4, 1, [8])
        push_val(buf, SIZE_T, 1)  # code_size (M=1, nbits=8)
        push_vector(buf, UINT8, [0] * 10)  # should be 2 bytes
        self.expect_read_throws(buf, "codes.size()")

    def test_index_local_search_quantizer_codes_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxLS")
        push_index_header(buf, 4, 2)
        push_local_search_quantizer(buf, 4, 1, [8])
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, UINT8, [0] * 10)  # should be 2 bytes
        self.expect_read_throws(buf, "codes.size()")

    def test_index_product_residual_quantizer_codes_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxPR")
        push_index_header(buf, 4, 2)
        push_additive_quantizer(buf, 4, 1, [8])
        push_val(buf, SIZE_T, 1)  # nsplits = 1
        push_residual_quantizer(buf, 4, 1, [8])  # d_sub = d / nsplits
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, UINT8, [0] * 10)  # should be 2 bytes
        self.expect_read_throws(buf, "codes.size()")

    def test_index_product_local_search_quantizer_codes_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxPL")
        push_index_header(buf, 4, 2)
        push_additive_quantizer(buf, 4, 1, [8])
        push_val(buf, SIZE_T, 1)  # nsplits = 1
        push_local_search_quantizer(buf, 4, 1, [8])
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, UINT8, [0] * 10)  # should be 2 bytes
        self.expect_read_throws(buf, "codes.size()")

    def test_nn_descent_graph_size_mismatch(self):
        # NNDescent: ntotal=2, K=3 -> expected graph size 6, provide 10.
        buf = bytearray()
        push_fourcc(buf, "INNf")
        push_index_header(buf, 4, 2)
        push_nndescent(buf, 2, 4, 3, True, [0] * 10)
        self.expect_read_throws(buf, "NNDescent final_graph size")

    def test_nn_descent_graph_invalid_id(self):
        # NNDescent: ntotal=2, K=2, graph size 4. ID 99 is out of [-1, 2).
        buf = bytearray()
        push_fourcc(buf, "INNf")
        push_index_header(buf, 4, 2)
        push_nndescent(buf, 2, 4, 2, True, [0, 1, 99, 0])
        self.expect_read_throws(buf, "out of range")

    def test_hnsw_levels_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IHNf")
        push_index_header(buf, 4, 5)
        push_minimal_hnsw(buf, 3)  # 3 entries, mismatches ntotal=5
        self.expect_read_throws(buf, "HNSW levels size")

    def test_hnsw_storage_ntotal_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IHNf")
        push_index_header(buf, 4, 3)
        push_minimal_hnsw(buf, 3)
        push_minimal_flat(buf, 4, 99)  # storage ntotal=99 (mismatch)
        self.expect_read_throws(buf, "HNSW storage ntotal")

    def test_hnsw_neighbor_negative_id(self):
        # validate_HNSW must reject a negative neighbor ID other than -1.
        ntotal, npn = 3, 2
        # Node 0: {1, -5} (-5 is invalid), node 1: {0, -1}, node 2: {0, -1}
        neighbors = [1, -5, 0, -1, 0, -1]
        buf = bytearray()
        push_fourcc(buf, "IHNf")
        push_index_header(buf, 4, ntotal)
        push_hnsw_with_neighbors(buf, ntotal, npn, neighbors)
        push_minimal_flat(buf, 4, ntotal)
        self.expect_read_throws(buf, "HNSW neighbors")

    def test_hnsw_neighbor_id_too_large(self):
        ntotal, npn = 3, 2
        neighbors = [1, 99, 0, -1, 0, -1]  # 99 >= ntotal
        buf = bytearray()
        push_fourcc(buf, "IHNf")
        push_index_header(buf, 4, ntotal)
        push_hnsw_with_neighbors(buf, ntotal, npn, neighbors)
        push_minimal_flat(buf, 4, ntotal)
        self.expect_read_throws(buf, "HNSW neighbors")

    def test_hnsw_entry_point_out_of_range(self):
        ntotal, npn = 3, 2
        neighbors = [1, -1, 0, -1, 0, -1]
        buf = bytearray()
        push_fourcc(buf, "IHNf")
        push_index_header(buf, 4, ntotal)
        push_hnsw_with_neighbors(buf, ntotal, npn, neighbors, entry_point=99)
        push_minimal_flat(buf, 4, ntotal)
        self.expect_read_throws(buf, "HNSW entry_point")

    def test_hnsw2_level_wrong_storage_type(self):
        # IndexHNSW2Level storage must be Index2Layer or IndexIVFPQ, else a
        # failed dynamic_cast leads to a null deref.
        buf = bytearray()
        push_fourcc(buf, "IHN2")
        push_index_header(buf, 4, 0)
        push_minimal_hnsw(buf, 0)
        push_minimal_flat(buf, 4, 0)  # IndexFlat storage: wrong type
        self.expect_read_throws(buf, "Index2Layer or IndexIVFPQ")

    def test_index2_layer_null_quantizer_ownership(self):
        # Index2Layer with a null quantizer ("null" fourcc) still sets
        # own_fields=true. The read is truncated after nlist, so the next
        # READ1 fails and triggers cleanup; with quantizer=nullptr the
        # destructor must not crash.
        buf = bytearray()
        push_fourcc(buf, "Ix2L")
        push_index_header(buf, 4, 0)
        push_fourcc(buf, "null")  # null sub-quantizer
        push_val(buf, SIZE_T, 1)  # nlist
        self.expect_read_throws(buf)

    def test_hnsw2_level_null_storage(self):
        buf = bytearray()
        push_fourcc(buf, "IHN2")
        push_index_header(buf, 4, 0)
        push_minimal_hnsw(buf, 0)
        push_fourcc(buf, "null")  # null storage
        self.expect_read_throws(buf, "non-null storage")

    def test_nsg_ntotal_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "INSf")
        push_index_header(buf, 4, 5)
        push_val(buf, INT, 0)  # GK
        push_val(buf, CHAR, 0)  # build_type (char, not int)
        push_val(buf, INT, 10)  # nndescent_S
        push_val(buf, INT, 10)  # nndescent_R
        push_val(buf, INT, 10)  # nndescent_L
        push_val(buf, INT, 1)  # nndescent_iter
        push_val(buf, INT, 3)  # NSG ntotal=3 (mismatch with 5)
        push_val(buf, INT, 2)  # R
        push_val(buf, INT, 10)  # L
        push_val(buf, INT, 10)  # C
        push_val(buf, INT, 10)  # search_L
        push_val(buf, INT, 0)  # enterpoint
        push_val(buf, BOOL, True)  # is_built
        # Graph: 3 nodes, R=2, all empty (EMPTY_ID terminates each)
        for _ in range(3):
            push_val(buf, INT, -1)
        self.expect_read_throws(buf, "NSG ntotal")

    def test_nn_descent_ntotal_mismatch(self):
        # NNDescent ntotal=3 (mismatch with index ntotal=5). The graph is
        # internally valid: K=2, 6 entries all in range.
        buf = bytearray()
        push_fourcc(buf, "INNf")
        push_index_header(buf, 4, 5)
        push_nndescent(buf, 3, 4, 2, True, [0, 1, 0, 2, 1, 2])
        self.expect_read_throws(buf, "NNDescent ntotal")

    def test_hnsw_cagra_empty_index_search(self):
        # base_level_only search on an empty index must not access
        # uninitialized graph nodes.
        idx = faiss.IndexHNSWCagra(4, 16)
        idx.base_level_only = True
        _, labels = idx.search(np.ones((1, 4), dtype="float32"), 1)
        self.assertEqual(labels[0, 0], -1)

    def test_index_ivf_null_invlists_search(self):
        # Search with null invlists (e.g. loaded with IO_FLAG_SKIP_IVF_DATA)
        # throws instead of crashing.
        quantizer = faiss.IndexFlatL2(4)
        idx = faiss.IndexIVFFlat(quantizer, 4, 10)
        idx.own_fields = False
        idx.is_trained = True
        idx.replace_invlists(None, False)
        with self.assertRaisesFaiss("inverted lists"):
            idx.search(np.ones((1, 4), dtype="float32"), 1)

    def test_index_ivf_null_invlists_add(self):
        quantizer = faiss.IndexFlatL2(4)
        idx = faiss.IndexIVFFlat(quantizer, 4, 10)
        idx.own_fields = False
        idx.is_trained = True
        idx.replace_invlists(None, False)
        with self.assertRaisesFaiss("inverted lists"):
            idx.add(np.ones((1, 4), dtype="float32"))

    # IVF quantizer ntotal / nlist acceptance tests. The quantizer may
    # legitimately have ntotal != nlist (sharded indexes, custom inverted
    # list management, untrained quantizers).

    def _ivf_flat_with_quantizer_ntotal(self, nlist, quantizer_ntotal):
        buf = bytearray()
        push_fourcc(buf, "IwFl")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, nlist)
        push_val(buf, SIZE_T, 1)  # nprobe
        push_minimal_flat(buf, 4, quantizer_ntotal)
        push_empty_direct_map(buf)
        push_null_invlists(buf)
        return buf

    def test_ivf_quantizer_ntotal_not_nlist_accepted(self):
        # shard_ivf_index_centroids() produces ntotal > nlist (all centroids
        # spread across shards) and 0 < ntotal < nlist (one shard's subset).
        # The search-time key < nlist check prevents OOB access.
        for nlist, ntotal in [(2, 5), (10, 3), (10, 0)]:
            with self.subTest(nlist=nlist, ntotal=ntotal):
                read_index(self._ivf_flat_with_quantizer_ntotal(nlist, ntotal))

    def test_ivf_scalar_quantizer_untrained_search_rejected(self):
        # Empty trained vector with is_trained=false deserializes (a
        # legitimate untrained index), but searching it throws because
        # IndexIVF checks is_trained.
        buf = bytearray()
        push_fourcc(buf, "IwSq")
        push_index_header(buf, 4, 0, is_trained=False)
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 1)  # nprobe
        push_minimal_flat(buf, 4)
        push_empty_direct_map(buf)
        push_val(buf, INT, 0)  # qtype = QT_8bit
        push_val(buf, INT, 0)  # rangestat
        push_val(buf, FLOAT, 0.0)  # rangestat_arg
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 4)  # code_size
        push_vector(buf, FLOAT, [])  # trained (empty: untrained)
        push_val(buf, SIZE_T, 4)  # IwSq code_size
        push_val(buf, BOOL, False)  # by_residual
        push_null_invlists(buf)

        idx = read_index(buf)
        self.assertFalse(idx.is_trained)

        xq = np.zeros((1, 4), dtype="float32")
        with self.assertRaisesFaiss():
            idx.search(xq, 1)
        with self.assertRaisesFaiss():
            idx.range_search(xq, 1.0)

        key = np.zeros((1, 1), dtype="int64")
        coarse_dis = np.zeros((1, 1), dtype="float32")
        with self.assertRaisesFaiss():
            idx.search_preassigned(xq, 1, key, coarse_dis)
        with self.assertRaisesFaiss():
            idx.range_search_preassigned(xq, 1.0, key, coarse_dis)

    def test_ivfpq_null_quantizer_precompute_table_rejected(self):
        # initialize_IVFPQ_precomputed_table rejects a null quantizer, which
        # corrupt data produces with a "null" quantizer fourcc.
        pq = faiss.ProductQuantizer(4, 1, 8)
        use_precomputed_table = np.zeros(1, dtype="int32")
        with self.assertRaisesFaiss():
            faiss.initialize_IVFPQ_precomputed_table(
                faiss.swig_ptr(use_precomputed_table),
                None,
                pq,
                faiss.AlignedTableFloat32(),
                True,  # by_residual
                False,  # verbose
            )

    def _ivf_flat_with_null_quantizer(self):
        ivf = swig_default_construct(faiss.IndexIVFFlat)
        ivf.quantizer = None
        return ivf

    def test_ivf_null_quantizer_search_rejected(self):
        ivf = self._ivf_flat_with_null_quantizer()
        ivf.is_trained = True
        with self.assertRaisesFaiss():
            ivf.search(np.zeros((1, ivf.d), dtype="float32"), 1)

    def test_ivf_null_quantizer_range_search_rejected(self):
        ivf = self._ivf_flat_with_null_quantizer()
        ivf.is_trained = True
        with self.assertRaisesFaiss():
            ivf.range_search(np.zeros((1, ivf.d), dtype="float32"), 1.0)

    def test_ivf_null_quantizer_add_rejected(self):
        ivf = self._ivf_flat_with_null_quantizer()
        with self.assertRaisesFaiss():
            ivf.add(np.zeros((1, ivf.d), dtype="float32"))

    def test_ivf_null_quantizer_train_rejected(self):
        ivf = self._ivf_flat_with_null_quantizer()
        with self.assertRaisesFaiss():
            ivf.train(np.zeros((1, ivf.d), dtype="float32"))

    def test_normalization_transform_din_dout_mismatch(self):
        # d_in != d_out makes memcpy overflow the output buffer in
        # apply_noalloc.
        buf = bytearray()
        push_fourcc(buf, "VNrm")
        push_val(buf, FLOAT, 2.0)  # norm (L2)
        push_val(buf, INT, 16)  # d_in
        push_val(buf, INT, 8)  # d_out (mismatch)
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(buf, "d_in == d_out")

    def test_centering_transform_din_dout_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "VCnt")
        push_vector(buf, FLOAT, [0.0] * 16)  # mean
        push_val(buf, INT, 16)  # d_in
        push_val(buf, INT, 8)  # d_out (mismatch)
        push_val(buf, BOOL, True)
        self.expect_vt_read_throws(buf, "d_in == d_out")

    def test_pre_transform_chain_dimension_mismatch(self):
        # A NormalizationTransform (d_in=d_out=4) followed by a sub-index
        # with d=8: the chain's d_out (4) != sub-index d (8).
        buf = bytearray()
        push_fourcc(buf, "IxPT")
        push_index_header(buf, 4, 0)
        push_val(buf, INT, 1)  # nt = 1 transform
        push_fourcc(buf, "VNrm")
        push_val(buf, FLOAT, 2.0)
        push_val(buf, INT, 4)  # d_in
        push_val(buf, INT, 4)  # d_out
        push_val(buf, BOOL, True)
        push_minimal_flat(buf, 8)
        self.expect_read_throws(buf, "d_out=4")

    def test_pre_transform_empty_chain_dimension_mismatch(self):
        # With no transform, reconstruct() hands the caller's buffer, sized
        # from the outer d, straight to the sub-index, so a larger sub-index
        # d is a heap overflow (T287094917).
        buf = bytearray()
        push_fourcc(buf, "IxPT")
        push_index_header(buf, 4, 0)
        push_val(buf, INT, 0)  # nt = 0, empty chain
        push_minimal_flat(buf, 8)
        self.expect_read_throws(buf, "IndexPreTransform empty chain")

    def test_hadamard_rotation_invalid_dout(self):
        buf = bytearray()
        push_fourcc(buf, "HRot")
        push_val(buf, INT, 42)  # seed
        push_val(buf, INT, 16)  # d_in
        push_val(buf, INT, 1 << 21)  # d_out (not the power of 2 for d_in)
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(buf, "d_out must be the smallest power of 2")

    def test_hadamard_rotation_din_zero(self):
        buf = bytearray()
        push_fourcc(buf, "HRot")
        push_val(buf, INT, 42)  # seed
        push_val(buf, INT, 0)  # d_in = 0 (invalid)
        push_val(buf, INT, 1)  # d_out
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(buf, "HadamardRotation d_in=")

    def test_hadamard_rotation_hr64_invalid_dout(self):
        # HR64: as HRot, with d_out the smallest multiple of 64 >= d_in
        buf = bytearray()
        push_fourcc(buf, "HR64")
        push_val(buf, INT, 42)  # seed
        push_val(buf, INT, 100)  # d_in
        push_val(buf, INT, 256)  # d_out (should be 128)
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(
            buf, "d_out must be the smallest multiple of 64 >= d_in")

    def test_hadamard_rotation_hr64_din_zero(self):
        buf = bytearray()
        push_fourcc(buf, "HR64")
        push_val(buf, INT, 42)  # seed
        push_val(buf, INT, 0)  # d_in = 0 (invalid)
        push_val(buf, INT, 64)  # d_out
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(buf, "HadamardRotation d_in=")

    def test_remap_dimensions_transform_map_too_small(self):
        buf = bytearray()
        push_fourcc(buf, "RmDT")
        push_vector(buf, INT, [0, 1])  # map with 2 entries
        push_val(buf, INT, 4)  # d_in
        push_val(buf, INT, 10)  # d_out = 10 > map.size()=2
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(buf, "RemapDimensionsTransform map size")

    def test_centering_transform_mean_too_small(self):
        buf = bytearray()
        push_fourcc(buf, "VCnt")
        push_vector(buf, FLOAT, [1.0, 2.0])  # mean with 2 entries
        push_val(buf, INT, 8)  # d_in = 8 > mean.size()=2
        push_val(buf, INT, 8)  # d_out
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(buf, "CenteringTransform mean size")

    def test_itq_transform_mean_too_small(self):
        buf = bytearray()
        push_fourcc(buf, "Viqt")
        push_vector(buf, FLOAT, [1.0])  # mean with 1 entry
        push_val(buf, BOOL, False)  # do_pca
        # Sub-VT 1: ITQMatrix ("Viqm")
        push_fourcc(buf, "Viqm")
        push_val(buf, INT, 10)  # max_iter
        push_val(buf, INT, 42)  # seed
        push_val(buf, BOOL, False)  # have_bias
        push_vector(buf, FLOAT, [])  # A
        push_vector(buf, FLOAT, [])  # b
        push_val(buf, INT, 0)  # d_in
        push_val(buf, INT, 0)  # d_out
        push_val(buf, BOOL, True)  # is_trained
        # Sub-VT 2: LinearTransform ("LTra")
        push_fourcc(buf, "LTra")
        push_val(buf, BOOL, False)  # have_bias
        push_vector(buf, FLOAT, [])  # A
        push_vector(buf, FLOAT, [])  # b
        push_val(buf, INT, 0)  # d_in
        push_val(buf, INT, 0)  # d_out
        push_val(buf, BOOL, True)  # is_trained
        # Outer Viqt footer
        push_val(buf, INT, 4)  # d_in = 4 > mean.size()=1
        push_val(buf, INT, 4)  # d_out
        push_val(buf, BOOL, True)  # is_trained
        self.expect_vt_read_throws(buf, "ITQTransform mean size")

    def test_inverted_list_oversized_entry(self):
        # An inverted list entry over the byte limit is rejected. Corrupt
        # data would otherwise cause OOM through an unchecked
        # vector::resize in read_InvertedLists_up.
        with vector_byte_limit(1024):  # 1 KB limit
            buf = bytearray()
            push_fourcc(buf, "IwFl")
            push_ivf_header(buf, 4)
            # "ilar" inverted lists with 1 list, code_size=16
            push_fourcc(buf, "ilar")
            push_val(buf, SIZE_T, 1)  # nlist
            push_val(buf, SIZE_T, 16)  # code_size
            push_fourcc(buf, "full")  # "full" list type
            push_val(buf, SIZE_T, 1)  # sizes vector length
            push_val(buf, SIZE_T, 1 << 40)  # sizes[0] = 1 TB
            self.expect_read_throws(buf, "deserialization byte limit")

    @unittest.skipIf(
        sys.platform == "win32",
        "OnDiskInvertedLists is unsupported on Windows.",
    )
    def test_on_disk_inverted_lists_short_lists_vector_rejected(self):
        buf = bytearray()
        push_fourcc(buf, "IwFl")
        push_ivf_header(buf, 4)
        push_fourcc(buf, "ilod")
        push_val(buf, SIZE_T, 1)  # nlist = 1
        push_val(buf, SIZE_T, 16)  # code_size
        push_val(buf, SIZE_T, 0)  # lists: empty, inconsistent with nlist
        self.expect_read_throws(buf, "read 0 lists for nlist 1")

    # RaBitQ qb validation. qb=0 is valid for Ixrq / Ixrr / Iwrq / Iwrr: it
    # disables query quantization and uses raw fp32 values (see the qb
    # comment in IndexRaBitQ.h / IndexIVFRaBitQ.h and
    # RaBitQDistanceComputerNotQ). qb=0 is not supported for the FastScan
    # variants Irfn / Iwrn, which need quantized queries for SIMD.

    def _rabitq_flat_buf(self, fourcc, multibit, qb):
        buf = bytearray()
        push_fourcc(buf, fourcc)
        push_index_header(buf, 4, 0)
        if multibit:
            push_rabitq_multibit(buf, 4)
        else:
            push_rabitq(buf, 4)
        push_vector(buf, UINT8, [])  # codes
        push_vector(buf, FLOAT, [0, 0, 0, 0])  # center
        push_val(buf, UINT8, qb)
        return buf

    def test_rabitq_qb_too_large_ixrq(self):
        buf = self._rabitq_flat_buf("Ixrq", False, 9)  # qb = 9 > 8
        self.expect_read_throws(buf, "RaBitQ qb=")

    def test_rabitq_qb_zero_accepted_ixrq(self):
        read_index(self._rabitq_flat_buf("Ixrq", False, 0))

    def test_rabitq_qb_too_large_ixrr(self):
        buf = self._rabitq_flat_buf("Ixrr", True, 9)
        self.expect_read_throws(buf, "RaBitQ qb=")

    def test_rabitq_qb_zero_accepted_ixrr(self):
        read_index(self._rabitq_flat_buf("Ixrr", True, 0))

    def _rabitq_fastscan_buf(self, qb):
        buf = bytearray()
        push_fourcc(buf, "Irfn")
        push_index_header(buf, 4, 0)
        push_rabitq_multibit(buf, 4)
        push_vector(buf, FLOAT, [0, 0, 0, 0])  # center
        push_val(buf, UINT8, qb)
        return buf

    def test_rabitq_qb_too_large_irfn(self):
        self.expect_read_throws(self._rabitq_fastscan_buf(9), "RaBitQ qb=")

    def test_rabitq_qb_zero_rejected_irfn(self):
        self.expect_read_throws(self._rabitq_fastscan_buf(0), "RaBitQ qb=")

    def _ivf_rabitq_buf(self, fourcc, multibit, qb, with_invlists):
        buf = bytearray()
        push_fourcc(buf, fourcc)
        push_ivf_header(buf, 4)
        if multibit:
            push_rabitq_multibit(buf, 4)
        else:
            push_rabitq(buf, 4)
        push_val(buf, SIZE_T, 1)  # code_size
        push_val(buf, BOOL, False)  # by_residual
        push_val(buf, UINT8, qb)
        if with_invlists:
            push_null_invlists(buf)
        return buf

    def test_rabitq_qb_too_large_iwrq(self):
        buf = self._ivf_rabitq_buf("Iwrq", False, 9, False)
        self.expect_read_throws(buf, "RaBitQ qb=")

    def test_rabitq_qb_zero_accepted_iwrq(self):
        read_index(self._ivf_rabitq_buf("Iwrq", False, 0, True))

    def test_rabitq_qb_too_large_iwrr(self):
        buf = self._ivf_rabitq_buf("Iwrr", True, 9, False)
        self.expect_read_throws(buf, "RaBitQ qb=")

    def test_rabitq_qb_zero_accepted_iwrr(self):
        read_index(self._ivf_rabitq_buf("Iwrr", True, 0, True))

    @unittest.skipUnless(
        hasattr(faiss, "IndexIVFRaBitQSSD"), "requires disk IVF")
    def test_ivf_rabitq_ssd_unsupported_metric(self):
        buf = bytearray()
        push_fourcc(buf, "IwDr")
        push_index_header(buf, 64, 0, metric_type=faiss.METRIC_L1)
        push_val(buf, FLOAT, 0.0)  # metric_arg for metric_type > 1
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 1)  # nprobe
        push_minimal_flat(buf, 64)
        push_empty_direct_map(buf)
        push_val(buf, SIZE_T, 64)  # rabitq.d
        push_val(buf, SIZE_T, 1)  # rabitq.code_size
        push_val(buf, INT, faiss.METRIC_L1)  # rabitq.metric_type
        push_val(buf, SIZE_T, 2)  # rabitq.nb_bits
        self.expect_read_throws(buf, "IndexIVFRaBitQSSD: unsupported metric")

    def _ivf_rabitq_fastscan_buf(self, qb):
        buf = bytearray()
        push_fourcc(buf, "Iwrn")
        push_ivf_header(buf, 4)
        push_rabitq_multibit(buf, 4)
        push_val(buf, BOOL, False)  # by_residual
        push_val(buf, SIZE_T, 1)  # code_size
        push_val(buf, INT, 32)  # bbs
        push_val(buf, SIZE_T, 0)  # qbs2
        push_val(buf, SIZE_T, 1)  # M2
        push_val(buf, INT, 0)  # implem
        push_val(buf, UINT8, qb)
        return buf

    def test_rabitq_qb_too_large_iwrn(self):
        self.expect_read_throws(self._ivf_rabitq_fastscan_buf(9), "RaBitQ qb=")

    def test_rabitq_qb_zero_rejected_iwrn(self):
        self.expect_read_throws(self._ivf_rabitq_fastscan_buf(0), "RaBitQ qb=")

    def test_binary_header_dimension_not_multiple_of8(self):
        buf = bytearray()
        push_fourcc(buf, "IBxF")
        push_val(buf, INT, 17)  # d = 17 (not multiple of 8)
        push_val(buf, INT, 2)  # code_size
        push_val(buf, INT64, 0)  # ntotal
        push_val(buf, BOOL, True)  # is_trained
        push_val(buf, INT, 1)  # metric_type
        self.expect_binary_read_throws(buf, "multiple of 8")

    def test_binary_header_code_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IBxF")
        push_val(buf, INT, 16)  # d = 16
        push_val(buf, INT, 5)  # code_size = 5 (should be 2)
        push_val(buf, INT64, 0)  # ntotal
        push_val(buf, BOOL, True)  # is_trained
        push_val(buf, INT, 1)  # metric_type
        self.expect_binary_read_throws(buf, "code_size")

    def test_binary_hnsw_levels_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IBHf")
        push_binary_index_header(buf, 16, 5)
        push_minimal_hnsw(buf, 3)  # 3 != 5
        self.expect_binary_read_throws(buf, "HNSW levels size")

    def test_binary_hnsw_storage_ntotal_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IBHf")
        push_binary_index_header(buf, 16, 3)
        push_minimal_hnsw(buf, 3)
        push_minimal_binary_flat(buf, 16)  # ntotal=0 != 3
        self.expect_binary_read_throws(buf, "storage ntotal")

    def _push_binary_ivf_storage(self, buf):
        # Nest an IBwF (IVF) instead of IBxF (flat)
        push_fourcc(buf, "IBwF")
        push_binary_index_header(buf, 16, 0)
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 1)  # nprobe
        push_minimal_binary_flat(buf, 16)  # quantizer
        push_empty_direct_map(buf)
        push_null_invlists(buf)

    def test_binary_hnsw_storage_not_flat(self):
        buf = bytearray()
        push_fourcc(buf, "IBHf")
        push_binary_index_header(buf, 16, 0)
        push_minimal_hnsw(buf, 0)
        self._push_binary_ivf_storage(buf)
        self.expect_binary_read_throws(buf, "IndexBinaryFlat storage")

    def _push_binary_hnsw_cagra_prefix(self, buf, ntotal):
        push_fourcc(buf, "IBHc")
        push_binary_index_header(buf, 16, ntotal)
        push_val(buf, BOOL, True)  # keep_max_size_level0
        push_val(buf, BOOL, False)  # base_level_only
        push_val(buf, INT, 10)  # num_base_level_search_entrypoints

    def test_binary_hnsw_cagra_levels_size_mismatch(self):
        buf = bytearray()
        self._push_binary_hnsw_cagra_prefix(buf, 5)
        push_minimal_hnsw(buf, 3)  # 3 != 5
        self.expect_binary_read_throws(buf, "HNSW levels size")

    def test_binary_hnsw_cagra_storage_not_flat(self):
        buf = bytearray()
        self._push_binary_hnsw_cagra_prefix(buf, 0)
        push_minimal_hnsw(buf, 0)
        self._push_binary_ivf_storage(buf)
        self.expect_binary_read_throws(buf, "IndexBinaryFlat storage")

    def test_binary_hnsw_cagra_storage_ntotal_mismatch(self):
        buf = bytearray()
        self._push_binary_hnsw_cagra_prefix(buf, 3)
        push_minimal_hnsw(buf, 3)
        push_minimal_binary_flat(buf, 16)  # ntotal=0 != 3
        self.expect_binary_read_throws(buf, "storage ntotal")

    def test_binary_hash_invlists_vecs_size_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IBHh")
        push_binary_index_header(buf, 16, 0)
        push_val(buf, INT, 4)  # b
        push_val(buf, INT, 0)  # nflip
        # hash invlists with 1 entry
        push_val(buf, SIZE_T, 1)  # sz = 1
        push_val(buf, INT, 8)  # il_nbit = 8
        # Bitstring: 1 entry needs (b + il_nbit) = 12 bits -> 2 bytes
        # hash = 0, ilsz = 1
        push_vector(buf, UINT8, pack_bits(2, [(0, 4), (1, 8)]))
        push_vector(buf, INT64, [0])  # ids: 1 entry
        push_vector(buf, UINT8, [0, 0, 0])  # vecs: 3 bytes, code_size=2
        self.expect_binary_read_throws(buf, "binary hash invlists: vecs size")

    def test_binary_multi_hash_map_id_out_of_range(self):
        # IBHm with ntotal=1, one map containing id=1 (>= ntotal).
        b = 4
        id_bits = 1
        ntotal = 1
        sz = 1
        # bits: (b + id_bits) * sz + 1 * id_bits = 6 bits -> 1 byte
        nbit = (b + id_bits) * sz + 1 * id_bits
        # hash = 0, ilsz = 1, id = 1 (>= ntotal=1, invalid)
        bitbuf = pack_bits(
            (nbit + 7) // 8, [(0, b), (1, id_bits), (1, id_bits)]
        )

        buf = bytearray()
        push_fourcc(buf, "IBHm")
        push_binary_index_header(buf, 16, ntotal)
        push_fourcc(buf, "IBxF")
        push_binary_index_header(buf, 16, ntotal)
        push_vector(buf, UINT8, [0] * (ntotal * 2))  # code_size=2
        push_val(buf, INT, b)
        push_val(buf, INT, 1)  # nhash = 1
        push_val(buf, INT, 0)  # nflip
        # Multi hash map fields (1 map):
        push_val(buf, INT, id_bits)
        push_val(buf, SIZE_T, sz)
        push_vector(buf, UINT8, bitbuf)
        self.expect_binary_read_throws(buf, "multi hash map: id=")

    def test_binary_multi_hash_b_zero(self):
        # b must be positive: BitstringReader::read(0) produces garbage hash
        # values and b is the bit width for hash extraction.
        buf = bytearray()
        push_fourcc(buf, "IBHm")
        push_binary_index_header(buf, 16, 0)
        push_minimal_binary_flat(buf, 16)
        push_val(buf, INT, 0)  # b = 0 (invalid)
        self.expect_binary_read_throws(buf, "IndexBinaryMultiHash b=")

    def test_binary_multi_hash_nhash_times_b_exceeds_code_size(self):
        # nhash * b must not exceed code_size * 8, otherwise
        # BitstringReader overflows the query buffer during search.
        buf = bytearray()
        push_fourcc(buf, "IBHm")
        push_binary_index_header(buf, 16, 0)
        push_minimal_binary_flat(buf, 16)
        push_val(buf, INT, 12)  # b = 12
        # nhash = 2 => nhash*b = 24 > code_size*8 = 16 (code_size = d/8)
        push_val(buf, INT, 2)
        self.expect_binary_read_throws(buf, "exceeds code_size")

    def test_binary_hnsw_get_distance_computer_non_flat_throws(self):
        idx = faiss.IndexBinaryHNSW()
        idx.storage = None
        idx.own_fields = False
        with self.assertRaisesFaiss("IndexBinaryFlat"):
            idx.get_distance_computer()

    def test_binary_hnsw_cagra_empty_index_search(self):
        idx = faiss.IndexBinaryHNSWCagra(16, 4)
        idx.base_level_only = True
        query = np.zeros((1, 2), dtype="uint8")  # d=16 -> code_size=2
        with self.assertRaisesFaiss("empty index"):
            idx.search(query, 1)

    def test_binary_hnsw_cagra_zero_entrypoints(self):
        idx = faiss.IndexBinaryHNSWCagra(16, 4)
        # add requires base_level_only == false
        idx.add(np.full((1, 2), 0xFF, dtype="uint8"))
        idx.base_level_only = True
        idx.num_base_level_search_entrypoints = 0
        query = np.zeros((1, 2), dtype="uint8")
        with self.assertRaisesFaiss("num_base_level_search_entrypoints"):
            idx.search(query, 1)

    def test_residual_coarse_quantizer_huge_ntotal(self):
        # Huge ntotal triggers the centroid_norms byte limit check (guard on
        # ntotal vs byte_limit/sizeof(float)).
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 1 << 60)  # would cause OOM in search()
        push_residual_quantizer(buf, 4, 2, [4, 4])
        push_val(buf, FLOAT, -1.0)  # beam_factor = -1 (skip tables)
        self.expect_read_throws(buf, "centroid norms allocation")

    def test_residual_coarse_quantizer_ntotal_times_m(self):
        # ntotal * M exceeds the byte limit even when ntotal alone passes
        # the centroid_norms check: 200000 < 1MB/4 = 262144, but
        # ntotal*M*sizeof(int32_t) = 200000*2*4 = 1600000 > 1MB.
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 200000)
        push_residual_quantizer(buf, 4, 2, [4, 4])
        push_val(buf, FLOAT, -1.0)  # beam_factor = -1 (skip tables)
        with vector_byte_limit(1 << 20):  # 1 MB
            self.expect_read_throws(buf, "deserialization vector byte limit")

    def test_residual_coarse_quantizer_huge_beam_factor(self):
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 16)
        push_residual_quantizer(buf, 4, 2, [4, 4])
        push_val(buf, FLOAT, 1e10)  # beam_factor = 1e10 (way too large)
        self.expect_read_throws(buf, "beam_factor")

    def test_residual_quantizer_max_beam_size_zero(self):
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 16)
        push_additive_quantizer(buf, 4, 2, [4, 4])
        push_val(buf, INT, 2048)  # train_type = Skip_codebook_tables
        push_val(buf, INT, 0)  # max_beam_size = 0 (invalid)
        self.expect_read_throws(buf, "max_beam_size")

    def test_residual_quantizer_huge_max_beam_size(self):
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 16)
        push_additive_quantizer(buf, 4, 2, [4, 4])
        push_val(buf, INT, 2048)  # train_type = Skip_codebook_tables
        push_val(buf, INT, 1 << 30)  # max_beam_size = 2^30 (huge)
        with vector_byte_limit(1 << 20):  # 1 MB
            self.expect_read_throws(buf, "deserialization vector byte limit")

    def test_residual_coarse_quantizer_negative_ntotal(self):
        # read_index_header rejects a negative ntotal.
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, -1)
        push_residual_quantizer(buf, 4, 2, [4, 4])
        push_val(buf, FLOAT, -1.0)  # beam_factor
        self.expect_read_throws(buf, "invalid ntotal")

    def test_additive_quantizer_m_zero(self):
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 16)
        push_additive_quantizer(buf, 4, 0, [])  # M=0
        push_val(buf, INT, 2048)  # train_type = Skip_codebook_tables
        push_val(buf, INT, 1)  # max_beam_size
        self.expect_read_throws(buf, "invalid AdditiveQuantizer M")

    def test_residual_coarse_quantizer_ntotal_zero(self):
        # An ntotal inconsistent with the codebooks is rejected before the
        # centroid norms are computed.
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 0)
        push_residual_quantizer(buf, 4, 2, [4, 4])
        push_val(buf, FLOAT, -1.0)  # beam_factor = -1 (skip tables)
        self.expect_read_throws(buf, "inconsistent with 2^tot_bits")

    def test_binary_ivf_null_invlists_search(self):
        idx = swig_default_construct(faiss.IndexBinaryIVF)
        idx.d = 16
        idx.code_size = 2
        idx.invlists = None
        idx.own_invlists = False
        query = np.zeros((1, 2), dtype="uint8")
        with self.assertRaisesFaiss("inverted lists"):
            idx.search(query, 1)

    def _vnrm_buf(self, d_in, d_out):
        buf = bytearray()
        push_fourcc(buf, "VNrm")
        push_val(buf, FLOAT, 2.0)
        push_val(buf, INT, d_in)
        push_val(buf, INT, d_out)
        push_val(buf, BOOL, True)
        return buf

    def test_vector_transform_dim_exceeds_byte_limit(self):
        # d_in=1024, d_out=1024 => d_in*d_out = 1M floats = 4 MB > 1 MB.
        buf = self._vnrm_buf(1024, 1024)
        with vector_byte_limit(1 << 20):
            self.expect_vt_read_throws(buf)

    def test_vector_transform_dim_within_byte_limit(self):
        # d_in=16, d_out=16 => 256 floats = 1 KB < 1 MB.
        buf = self._vnrm_buf(16, 16)
        with vector_byte_limit(1 << 20):
            read_vector_transform(buf)

    def _corrupt_iwiq_vt_dimension(self, field_offset):
        # Serialize a valid IndexIVFIndependentQuantizer, locate the VNrm
        # fourcc and overwrite one VT dimension with d + 1.
        # VNrm layout: fourcc(4) + norm(4) + d_in(4) + d_out(4) +
        # is_trained(1)
        d = 16
        nlist = 4
        quantizer = faiss.IndexFlat(d)
        ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
        ivf.own_fields = False
        vt = faiss.NormalizationTransform(d)
        indep = faiss.IndexIVFIndependentQuantizer(quantizer, ivf, vt)
        indep.own_fields = False

        data = serialize_index(indep)
        vnrm_pos = data.find(b"VNrm")
        self.assertGreaterEqual(vnrm_pos, 0)
        corrupted = bytearray(data)
        struct.pack_into("<i", corrupted, vnrm_pos + field_offset, d + 1)
        return corrupted

    def test_iw_iq_vt_din_mismatch(self):
        # A VT d_in that does not match the outer index dimension.
        buf = self._corrupt_iwiq_vt_dimension(4 + 4)  # after fourcc + norm
        self.expect_read_throws(buf)

    def _build_index_pq_fast_scan_buf(self, bbs=32, M2=2, qbs=0):
        # Minimal valid IndexPQFastScan ("IPfs"). PQ: d=4, M=2, nbits=4 ->
        # ksub=16, centroids size = d*ksub = 64 floats.
        buf = bytearray()
        push_fourcc(buf, "IPfs")
        push_index_header(buf, 4, 0)
        push_pq(buf, 4, 2, 4, [0.0] * 64)
        push_val(buf, INT, 0)  # implem
        push_val(buf, INT, bbs)
        push_val(buf, INT, qbs)
        push_val(buf, SIZE_T, 0)  # ntotal2
        push_val(buf, SIZE_T, M2)
        push_vector(buf, UINT8, [])  # codes
        return buf

    def _build_aq_fast_scan_buf(self, fastscan_M=3, fastscan_ksub=16, qbs=0):
        # Minimal valid IndexResidualQuantizerFastScan ("IRfs"):
        # AdditiveQuantizer + ResidualQuantizer fields, then FastScan fields.
        buf = bytearray()
        push_fourcc(buf, "IRfs")
        push_index_header(buf, 4, 0)
        # AdditiveQuantizer fields:
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 1)  # M (AQ M, not FastScan M)
        push_vector(buf, SIZE_T, [4])  # nbits
        push_val(buf, BOOL, False)  # is_trained
        push_vector(buf, FLOAT, [0.0] * 64)  # codebooks (d*16)
        push_val(buf, INT, 0)  # search_type = ST_decompress
        push_val(buf, FLOAT, 0.0)  # norm_min
        push_val(buf, FLOAT, 1.0)  # norm_max
        # ResidualQuantizer extras:
        push_val(buf, INT, 2048)  # train_type = Skip_codebook_tables
        push_val(buf, INT, 1)  # max_beam_size
        # FastScan fields (IndexAdditiveQuantizerFastScan):
        push_val(buf, INT, 0)  # implem
        push_val(buf, INT, 32)  # bbs
        push_val(buf, INT, qbs)
        push_val(buf, SIZE_T, fastscan_M)  # M
        push_val(buf, SIZE_T, 4)  # nbits
        push_val(buf, SIZE_T, fastscan_ksub)  # ksub
        push_val(buf, SIZE_T, 2)  # code_size
        push_val(buf, SIZE_T, 0)  # ntotal2
        # M2 = roundup(M, 2) in size_t arithmetic (wraps to 0 for
        # M = SIZE_MAX).
        push_val(buf, SIZE_T, ((fastscan_M + 1) & ~1) & SIZE_MAX)
        push_val(buf, BOOL, True)  # rescale_norm
        push_val(buf, INT, 1)  # norm_scale
        push_val(buf, SIZE_T, 48)  # max_train_points
        push_vector(buf, UINT8, [])  # codes
        return buf

    def test_index_pq_fast_scan_bbs_zero(self):
        # Without this check, search() would divide by zero in ntotal2/bbs.
        buf = self._build_index_pq_fast_scan_buf(bbs=0)
        self.expect_read_throws(buf, "invalid bbs")

    def test_index_pq_fast_scan_bbs_not_aligned(self):
        # bbs not a multiple of 32.
        buf = self._build_index_pq_fast_scan_buf(bbs=33)
        self.expect_read_throws(buf, "invalid bbs")

    def test_index_pq_fast_scan_m2_zero(self):
        # M2 must equal roundup(M, 2). M2=0 while M>0 makes
        # compute_quantized_LUT write M*ksub bytes into a buffer sized for
        # M2*ksub=0 bytes.
        buf = self._build_index_pq_fast_scan_buf(bbs=32, M2=0)
        self.expect_read_throws(buf, "invalid M2")

    def test_aq_fast_scan_m_zero(self):
        # Without this check, search() would crash in compute_float_LUT.
        buf = self._build_aq_fast_scan_buf(fastscan_M=0)
        self.expect_read_throws(buf, "invalid quantizer state")

    def test_aq_fast_scan_ksub_zero(self):
        buf = self._build_aq_fast_scan_buf(fastscan_M=3, fastscan_ksub=0)
        self.expect_read_throws(buf, "invalid quantizer state")

    def test_aq_fast_scan_ksub_m_overflow(self):
        buf = self._build_aq_fast_scan_buf(
            fastscan_M=SIZE_MAX, fastscan_ksub=16
        )
        self.expect_read_throws(buf, "overflow")

    def test_rabitq_fast_scan_bbs_zero(self):
        # Serialize a valid IndexRaBitQFastScan and patch its bbs field. The
        # format has qb (uint8_t = 8) immediately followed by bbs (int = 32).
        idx = faiss.IndexRaBitQFastScan(4, faiss.METRIC_L2, 32, 1)
        buf = serialize_index(idx)
        pos = buf.find(struct.pack("<Bi", 8, 32))
        self.assertGreaterEqual(
            pos, 0, "Could not find bbs field in serialized data"
        )
        struct.pack_into("<i", buf, pos + 1, 0)
        self.expect_read_throws(buf, "invalid bbs")

    def test_additive_quantizer_codebooks_too_small(self):
        # read_AdditiveQuantizer rejects codebooks smaller than
        # d * total_codebook_size. IxRq with d=4, M=1, nbits={8} and empty
        # codebooks (0 < d*256 = 1024).
        buf = bytearray()
        push_fourcc(buf, "IxRq")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 1)  # M
        push_vector(buf, SIZE_T, [8])  # nbits
        push_val(buf, BOOL, True)  # is_trained
        push_vector(buf, FLOAT, [])  # codebooks (empty, triggers check)
        push_val(buf, INT, 0)  # search_type = ST_decompress
        push_val(buf, FLOAT, 0.0)  # norm_min
        push_val(buf, FLOAT, 1.0)  # norm_max
        push_val(buf, INT, 2048)  # train_type = Skip_codebook_tables
        push_val(buf, INT, 1)  # max_beam_size
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, UINT8, [])  # codes (matches ntotal=0)
        self.expect_read_throws(buf, "not a positive multiple")

    def test_additive_quantizer_d_zero(self):
        buf = bytearray()
        push_fourcc(buf, "IxRq")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, 0)  # d = 0 (invalid)
        push_val(buf, SIZE_T, 1)  # M
        push_vector(buf, SIZE_T, [8])  # nbits
        push_val(buf, BOOL, True)  # is_trained
        push_vector(buf, FLOAT, [])  # codebooks
        push_val(buf, INT, 0)  # search_type = ST_decompress
        push_val(buf, FLOAT, 0.0)  # norm_min
        push_val(buf, FLOAT, 1.0)  # norm_max
        self.expect_read_throws(buf, "invalid AdditiveQuantizer d")

    def test_additive_quantizer_codebooks_not_multiple(self):
        # d=4, M=1, nbits={8} -> total_codebook_size=256; provide 257 floats.
        buf = bytearray()
        push_fourcc(buf, "IxRq")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, 4)  # d
        push_val(buf, SIZE_T, 1)  # M
        push_vector(buf, SIZE_T, [8])  # nbits
        push_val(buf, BOOL, True)  # is_trained
        push_vector(buf, FLOAT, [0.0] * 257)  # not a multiple
        push_val(buf, INT, 0)  # search_type = ST_decompress
        push_val(buf, FLOAT, 0.0)  # norm_min
        push_val(buf, FLOAT, 1.0)  # norm_max
        self.expect_read_throws(buf, "not a positive multiple")

    def test_product_additive_quantizer_d_not_divisible(self):
        # d=5, nsplits=2 -> d % nsplits != 0. The codebooks are sized for d
        # because the divisibility check fires before the codebooks size
        # check.
        buf = bytearray()
        push_fourcc(buf, "IxPR")
        push_index_header(buf, 5, 0)
        push_val(buf, SIZE_T, 5)  # d
        push_val(buf, SIZE_T, 2)  # M
        push_vector(buf, SIZE_T, [4, 4])  # nbits
        push_val(buf, BOOL, True)  # is_trained
        push_vector(buf, FLOAT, [0.0] * (5 * 32))  # codebooks
        push_val(buf, INT, 0)  # search_type = ST_decompress
        push_val(buf, FLOAT, 0.0)  # norm_min
        push_val(buf, FLOAT, 1.0)  # norm_max
        push_val(buf, SIZE_T, 2)  # nsplits = 2
        self.expect_read_throws(buf, "not divisible by nsplits")

    def test_product_residual_quantizer_sub_d_mismatch(self):
        # d=8, nsplits=2 -> d_sub should be 4, the sub-quantizer has d=6.
        buf = bytearray()
        push_fourcc(buf, "IxPR")
        push_index_header(buf, 8, 0)
        push_additive_quantizer(buf, 8, 2, [4, 4], codebook_d=4)
        push_val(buf, SIZE_T, 2)  # nsplits = 2
        # First sub-quantizer: d=6. The read throws before the second.
        push_residual_quantizer(buf, 6, 1, [4])
        self.expect_read_throws(buf, "sub-quantizer")

    def test_product_local_search_quantizer_sub_d_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxPL")
        push_index_header(buf, 8, 0)
        push_additive_quantizer(buf, 8, 2, [4, 4], codebook_d=4)
        push_val(buf, SIZE_T, 2)  # nsplits = 2
        push_local_search_quantizer(buf, 6, 1, [4])
        self.expect_read_throws(buf, "sub-quantizer")

    def test_product_residual_quantizer_multi_split_valid(self):
        # nsplits > 1 round-trips (codebooks sized for d_sub, not d).
        # d=8, nsplits=2, each sub-RQ has d_sub=4, M=1, nbits={4}
        buf = bytearray()
        push_fourcc(buf, "IxPR")
        push_index_header(buf, 8, 0)
        push_additive_quantizer(buf, 8, 2, [4, 4], codebook_d=4)
        push_val(buf, SIZE_T, 2)  # nsplits = 2
        push_residual_quantizer(buf, 4, 1, [4])
        push_residual_quantizer(buf, 4, 1, [4])
        push_val(buf, SIZE_T, 1)  # code_size (8 bits -> 1 byte)
        push_vector(buf, UINT8, [])  # codes (ntotal=0)
        idx = read_index(buf)
        self.assertEqual(idx.ntotal, 0)
        self.assertEqual(idx.d, 8)

    def test_product_local_search_quantizer_multi_split_valid(self):
        buf = bytearray()
        push_fourcc(buf, "IxPL")
        push_index_header(buf, 8, 0)
        push_additive_quantizer(buf, 8, 2, [4, 4], codebook_d=4)
        push_val(buf, SIZE_T, 2)  # nsplits = 2
        push_local_search_quantizer(buf, 4, 1, [4])
        push_local_search_quantizer(buf, 4, 1, [4])
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, UINT8, [])  # codes (ntotal=0)
        idx = read_index(buf)
        self.assertEqual(idx.ntotal, 0)
        self.assertEqual(idx.d, 8)

    # A negative qbs must be rejected at deserialization time to prevent
    # infinite loops in pq4_qbs_to_nq (arithmetic right shift preserves the
    # sign bit, so the loop never terminates for negative values).

    def test_index_pq_fast_scan_negative_qbs(self):
        buf = self._build_index_pq_fast_scan_buf(bbs=32, M2=2, qbs=-1)
        self.expect_read_throws(buf, "qbs must be non-negative")

    def test_aq_fast_scan_negative_qbs(self):
        buf = self._build_aq_fast_scan_buf(
            fastscan_M=3, fastscan_ksub=16, qbs=-1
        )
        self.expect_read_throws(buf, "qbs must be non-negative")

    def test_ivf_aq_fast_scan_negative_qbs(self):
        # Serialize a valid IndexIVFResidualQuantizerFastScan, then patch
        # the qbs field. qbs (int) immediately follows bbs (int = 32) in the
        # IVRf format; a distinctive sentinel qbs makes the pattern unique.
        coarse_quantizer = faiss.IndexFlat(4, faiss.METRIC_L2)
        idx = faiss.IndexIVFResidualQuantizerFastScan(
            coarse_quantizer, 4, 1, 1, 4
        )
        train_data = np.arange(4 * 64, dtype="float32").reshape(64, 4)
        idx.train(train_data)
        sentinel_qbs = 0x1234
        idx.qbs = sentinel_qbs

        buf = serialize_index(idx)
        pos = buf.find(struct.pack("<ii", 32, sentinel_qbs))
        self.assertGreaterEqual(
            pos, 0, "Could not find qbs field in serialized data"
        )
        struct.pack_into("<i", buf, pos + 4, -1)
        self.expect_read_throws(buf, "qbs must be non-negative")

    def _sq_buf(self, qtype, sq_d, trained, is_trained=True):
        buf = bytearray()
        push_fourcc(buf, "IxSQ")
        push_index_header(buf, 4, 0, is_trained=is_trained)
        push_val(buf, INT, qtype)
        push_val(buf, INT, 0)  # rangestat
        push_val(buf, FLOAT, 0.0)  # rangestat_arg
        push_val(buf, SIZE_T, sq_d)
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, FLOAT, trained)
        push_vector(buf, UINT8, [])  # codes (ntotal=0)
        return buf

    def test_sq_empty_trained_non_uniform(self):
        # QT_8bit (NON_UNIFORM) expects trained.size() == 2*d == 8.
        buf = self._sq_buf(faiss.ScalarQuantizer.QT_8bit, 4, [])
        self.expect_read_throws(buf, "ScalarQuantizer trained size")

    def test_sq_empty_trained_uniform(self):
        # QT_8bit_uniform (UNIFORM) expects trained.size() == 2.
        buf = self._sq_buf(faiss.ScalarQuantizer.QT_8bit_uniform, 4, [])
        self.expect_read_throws(buf, "ScalarQuantizer trained size")

    def test_sq_dimension_mismatch(self):
        # ScalarQuantizer d = 8 mismatches header d = 4.
        buf = self._sq_buf(faiss.ScalarQuantizer.QT_8bit, 8, [0.0] * 16)
        self.expect_read_throws(buf, "ScalarQuantizer d")

    def test_sq_is_trained_mismatch(self):
        # Correctly sized trained data (2*d=8 floats) but is_trained=false.
        buf = self._sq_buf(
            faiss.ScalarQuantizer.QT_8bit, 4, [0.0] * 8, is_trained=False
        )
        self.expect_read_throws(buf, "is_trained")

    def test_index_residual_quantizer_aq_dimension_mismatch(self):
        # Index header d=4, AQ d=8.
        buf = bytearray()
        push_fourcc(buf, "IxRq")
        push_index_header(buf, 4, 0)
        push_residual_quantizer(buf, 8, 1, [4])
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, UINT8, [])  # codes
        self.expect_read_throws(buf, "does not match index d")

    def test_index_local_search_quantizer_aq_dimension_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IxLS")
        push_index_header(buf, 4, 0)
        push_local_search_quantizer(buf, 8, 1, [4])
        push_val(buf, SIZE_T, 1)  # code_size
        push_vector(buf, UINT8, [])  # codes
        self.expect_read_throws(buf, "does not match index d")

    def test_residual_coarse_quantizer_aq_dimension_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "ImRQ")
        push_index_header(buf, 4, 0)
        push_residual_quantizer(buf, 8, 1, [4])
        push_val(buf, FLOAT, -1.0)  # beam_factor
        self.expect_read_throws(buf, "does not match index d")

    def test_index_rq_fast_scan_aq_dimension_mismatch(self):
        buf = bytearray()
        push_fourcc(buf, "IRfs")
        push_index_header(buf, 4, 0)
        push_residual_quantizer(buf, 8, 1, [4])
        # The early validation fires before these FastScan fields.
        push_val(buf, INT, 0)  # implem
        push_val(buf, INT, 32)  # bbs
        push_val(buf, INT, 0)  # qbs
        self.expect_read_throws(buf, "does not match index d")

    # A corrupt code_size passes the codes.size() == ntotal * code_size
    # check when ntotal == 0 and can cause excessive allocations in
    # GenericFlatCodesDistanceComputer during search.

    def _find_code_size_offset(self, index, attr, real_cs):
        # Locate the byte offset of code_size by diffing two
        # serializations: one with the real code_size and one with a probe
        # value. This avoids false matches from other fields that happen to
        # share the same numeric value.
        w1 = serialize_index(index)
        probe_cs = real_cs ^ 0xDEAD
        setattr(index, attr, probe_cs)
        w2 = serialize_index(index)
        setattr(index, attr, real_cs)

        self.assertEqual(len(w1), len(w2))
        offset = -1
        i = 0
        while i + 8 <= len(w1):
            if w1[i] != w2[i]:
                (v1,) = struct.unpack_from("<Q", w1, i)
                (v2,) = struct.unpack_from("<Q", w2, i)
                if v1 == real_cs and v2 == probe_cs:
                    offset = i
                i += 8
            else:
                i += 1
        return offset

    def _corrupt_code_size_and_expect_throw(self, index, attr, expected_substr):
        real_cs = getattr(index, attr)
        offset = self._find_code_size_offset(index, attr, real_cs)
        self.assertGreaterEqual(
            offset,
            0,
            "could not locate code_size field in serialized data via "
            "double-serialization diff",
        )
        buf = serialize_index(index)
        struct.pack_into("<Q", buf, offset, real_cs + 999)
        self.expect_read_throws(buf, expected_substr)

    def test_residual_quantizer_code_size_mismatch(self):
        d, nb = 8, 256
        idx = faiss.IndexResidualQuantizer(d, 2, 4)
        idx.train(make_random_data(d, nb))
        self._corrupt_code_size_and_expect_throw(
            idx, "code_size", "code_size mismatch"
        )

    def test_local_search_quantizer_code_size_mismatch(self):
        d, nb = 8, 256
        idx = faiss.IndexLocalSearchQuantizer(d, 2, 4)
        idx.train(make_random_data(d, nb))
        self._corrupt_code_size_and_expect_throw(
            idx, "code_size", "code_size mismatch"
        )

    def test_product_residual_quantizer_code_size_mismatch(self):
        d, nb = 16, 512
        idx = faiss.IndexProductResidualQuantizer(d, 2, 4, 8)
        idx.train(make_random_data(d, nb))
        self._corrupt_code_size_and_expect_throw(
            idx, "code_size", "code_size mismatch"
        )

    def test_product_local_search_quantizer_code_size_mismatch(self):
        d, nb = 8, 256
        idx = faiss.IndexProductLocalSearchQuantizer(d, 2, 2, 4)
        idx.train(make_random_data(d, nb))
        self._corrupt_code_size_and_expect_throw(
            idx, "code_size", "code_size mismatch"
        )

    def test_lsh_code_size_mismatch(self):
        # IndexLSH code_size is validated against (nbits + 7) / 8.
        buf = bytearray()
        push_fourcc(buf, "IxHe")  # new format IndexLSH
        push_index_header(buf, 8, 0)
        push_val(buf, INT, 64)  # nbits
        push_val(buf, BOOL, False)  # rotate_data
        push_val(buf, BOOL, False)  # train_thresholds
        push_vector(buf, FLOAT, [])  # thresholds
        push_val(buf, INT, 99)  # code_size = 99 (should be 8)
        # rrot: RandomRotationMatrix
        push_fourcc(buf, "rrot")
        push_val(buf, INT, 8)  # d_in
        push_val(buf, INT, 64)  # d_out
        push_val(buf, BOOL, False)  # is_trained
        push_val(buf, BOOL, False)  # have_bias
        push_vector(buf, FLOAT, [])  # A
        push_vector(buf, FLOAT, [])  # b
        push_vector(buf, UINT8, [])  # codes
        self.expect_read_throws(buf, "code_size mismatch")

    def test_ivf_scalar_quantizer_code_size_mismatch(self):
        # QT_fp16 gives code_size = d * 2 = 16, distinctive enough. The
        # corrupt code_size is caught either by validate_code_size_match or
        # by the InvertedLists code_size consistency check.
        d, nb, nlist = 8, 256, 4
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16
        )
        idx.own_fields = False
        idx.train(make_random_data(d, nb))
        self._corrupt_code_size_and_expect_throw(idx, "code_size", "code_size")

    def test_ivf_additive_quantizer_code_size_mismatch(self):
        d, nb, nlist = 16, 512, 4
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFResidualQuantizer(quantizer, d, nlist, 2, 8)
        idx.own_fields = False
        idx.train(make_random_data(d, nb))
        self._corrupt_code_size_and_expect_throw(idx, "code_size", "code_size")

    def _trained_index2_layer(self):
        d, nb, nlist = 8, 256, 4
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.Index2Layer(quantizer, nlist, 4)
        xb = make_random_data(d, nb)
        idx.train(xb)
        idx.add(xb)
        return idx

    def test_index2_layer_code_size2_mismatch(self):
        self._corrupt_code_size_and_expect_throw(
            self._trained_index2_layer(), "code_size_2", "code_size mismatch"
        )

    def test_index2_layer_code_size_sum_mismatch(self):
        self._corrupt_code_size_and_expect_throw(
            self._trained_index2_layer(), "code_size", "code_size mismatch"
        )

    def test_index_lattice_code_size_too_large(self):
        # IndexLattice code_size derives from scale_nbit, lattice_nbit and
        # nsq. A corrupt (negative) scale_nbit overflows the
        # total_nbit -> code_size computation and gives a huge code_size.
        buf = bytearray()
        push_fourcc(buf, "IxLa")
        push_val(buf, INT, 4)  # d
        push_val(buf, INT, 2)  # nsq (dsq = 2, power of 2 >= 2)
        push_val(buf, INT, -100)  # scale_nbit (corrupt)
        push_val(buf, INT, 1)  # r2
        self.expect_read_throws(buf, "code_size")

    def test_product_quantizer_m_ksub_exceeds_byte_limit(self):
        # ProductQuantizer::set_derived_values only validates d % M == 0,
        # which any M satisfies when d == 0, so a crafted PQ header could
        # carry an enormous M with an empty centroids vector. Without the
        # read-side bound, allocations sized M*ksub (e.g. IVFPQ residual
        # norms in initialize_IVFPQ_precomputed_table) reach
        # std::vector::max_size() and abort via std::length_error.
        with vector_byte_limit(1 << 20):  # 1 MB
            buf = bytearray()
            push_fourcc(buf, "IxPq")
            push_index_header(buf, 0, 0)
            # PQ: d=0, M=1<<20, nbits=8 -> ksub=256.
            # M * ksub = 2^28 floats = 1 GB, far above the 1 MB limit.
            push_pq(buf, 0, 1 << 20, 8, [])
            self.expect_read_throws(buf, "M*ksub")

    def _saturating_pq(self):
        # M*ksub saturates most of size_t. set_derived_values (which would
        # resize centroids) is skipped: the tests exercise the arithmetic
        # guard in initialize_IVFPQ_precomputed_table, not PQ training.
        pq = faiss.ProductQuantizer()
        pq.d = 0
        pq.M = 1 << 40
        pq.nbits = 24
        pq.ksub = 1 << pq.nbits
        pq.dsub = 0
        pq.code_size = 0
        return pq

    def test_initialize_ivfpq_precomputed_table_user_set_flag_overflow_rejected(
        self,
    ):
        # The overflowing M*ksub is rejected even when the caller pre-sets
        # use_precomputed_table to 1, which bypasses the size-class branch.
        # Protects in-memory IVFPQ callers (training, IndexHNSW,
        # IndexIVFPQFastScan) and the IndexIVFIndependentQuantizer read path
        # that reads use_precomputed_table directly from the stream.
        quantizer = faiss.IndexFlatL2(0)
        quantizer.ntotal = 1
        use_table = np.array([1], dtype="int32")
        with self.assertRaisesFaiss():
            faiss.initialize_IVFPQ_precomputed_table(
                faiss.swig_ptr(use_table),
                quantizer,
                self._saturating_pq(),
                faiss.AlignedTableFloat32(),
                True,  # by_residual
                False,  # verbose
            )

    def _expect_precomputed_table_limit(self, index, nlist, M, nbits, limit):
        # Read with a limit below the recomputed table size must throw from
        # the precompute-table guard (the message pins it, so no other
        # byte-limit check can satisfy the test). Just above the table size
        # the index still loads (positive control).
        xt = (
            np.random.RandomState(12345)
            .uniform(-1.0, 1.0, size=(10000, index.d))
            .astype("float32")
        )
        index.train(xt)
        index.add(xt)
        data = serialize_index(index)
        table_elems = nlist * M * (1 << nbits)
        with vector_byte_limit(limit):
            self.expect_read_throws(data, "precomputed_table")
        with vector_byte_limit((table_elems + 1) * 4):
            read_index(data)

    def test_ivfpq_precomputed_table_exceeds_byte_limit(self):
        # read_ivfpq bounds the derived (not stored) IVFPQ precomputed table
        # by the byte limit before it recomputes it. The table is
        # quantizer->ntotal * pq.M * pq.ksub floats from attacker-controlled
        # header fields, so a small crafted index can drive a large
        # allocation (memory-amplification DoS reachable from read_index
        # and the binary index fuzzer).
        d, nlist, M, nbits = 16, 256, 8, 8  # ksub = 256
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        # Recomputed table = 256 * 8 * 256 = 524288 floats (2 MB). A 256 KB
        # limit (65536 floats) is below the table but above the smaller
        # derived reads (PQ centroids ~16 KB, inverted lists).
        self._expect_precomputed_table_limit(index, nlist, M, nbits, 1 << 18)

    def test_ivfpq_fast_scan_precomputed_table_exceeds_byte_limit(self):
        # The same bound guards the IVFPQFastScan (IwPf) read path, whose
        # precompute_table() call is unconditional. nbits=4 -> ksub=16, so
        # the table = 256 * 8 * 16 = 32768 floats (128 KB). 64 KB (16384
        # floats) is below the table but above the smaller derived reads.
        d, nlist, M, nbits = 16, 256, 8, 4
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQFastScan(quantizer, d, nlist, M, nbits)
        self._expect_precomputed_table_limit(index, nlist, M, nbits, 1 << 16)

    def test_ivfpq_null_quantizer_read_rejected(self):
        # read_ivf_header reads the coarse quantizer via read_index, which
        # returns nullptr for a crafted "null" fourcc. The null check right
        # after read_ivf_header must throw before the precompute-table guard
        # (or precompute_table) dereferences quantizer->ntotal.
        buf = bytearray()
        push_fourcc(buf, "IwPQ")
        push_index_header(buf, 4, 0, is_trained=True)
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 1)  # nprobe
        push_fourcc(buf, "null")  # coarse quantizer -> nullptr
        push_empty_direct_map(buf)
        push_val(buf, BOOL, True)  # by_residual
        push_val(buf, SIZE_T, 1)  # code_size
        # Valid PQ: d=4, M=1, nbits=8 -> centroids = d*ksub = 1024.
        push_pq(buf, 4, 1, 8, [0.0] * (4 * 256))
        push_null_invlists(buf)
        self.expect_read_throws(buf, "IVFPQ coarse quantizer is null")

    def test_ivfpq_multi_index_quantizer_precomputed_table_exceeds_byte_limit(
        self,
    ):
        # The type-2 table (MultiIndexQuantizer coarse quantizer) is
        # cpq.ksub * pq.M * pq.ksub floats, decoupled from quantizer->ntotal.
        d = 8
        buf = bytearray()
        push_fourcc(buf, "IwPQ")
        push_index_header(buf, d, 0, is_trained=True)
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 1)  # nprobe
        # Coarse quantizer: MultiIndexQuantizer ("Imiq") with ntotal=1 but
        # cpq.nbits=10 -> cpq.ksub=1024.
        push_fourcc(buf, "Imiq")
        push_index_header(buf, d, 1, is_trained=True)
        push_pq(buf, d, 1, 10, [0.0] * (d * 1024))
        push_empty_direct_map(buf)
        push_val(buf, BOOL, True)  # by_residual
        push_val(buf, SIZE_T, 1)  # code_size
        # Outer PQ: M=2, nbits=8 -> ksub=256, m_ksub=512, pq.M % cpq.M == 0.
        push_pq(buf, d, 2, 8, [0.0] * (d * 256))
        push_null_invlists(buf)
        # type-1 bound: 1 * 2 * 256 = 512 floats (an ntotal-only guard would
        # pass). type-2 table: 1024 * 2 * 256 = 524288 floats.
        with vector_byte_limit(1 << 16):  # 64 KB
            self.expect_read_throws(buf, "precomputed_table")

    def test_ivfpq_fast_scan_null_quantizer_read_rejected(self):
        # The IVFPQFastScan (IwPf) read path also rejects a null coarse
        # quantizer right after read_ivf_header.
        buf = bytearray()
        push_fourcc(buf, "IwPf")
        push_index_header(buf, 4, 0, is_trained=True)
        push_val(buf, SIZE_T, 1)  # nlist
        push_val(buf, SIZE_T, 1)  # nprobe
        push_fourcc(buf, "null")  # coarse quantizer -> nullptr
        push_empty_direct_map(buf)
        self.expect_read_throws(buf, "IVFPQFastScan coarse quantizer is null")

    # SVS fourcc rejection and deserialization safety (T262015608).

    def _push_svs_vamana_prefix(self, buf, storage_kind):
        # IndexSVSVamana on-disk fields up to and including storage_kind.
        push_index_header(buf, 8, 0)
        push_val(buf, SIZE_T, 32)  # graph_max_degree
        push_val(buf, FLOAT, 1.2)  # alpha
        push_val(buf, SIZE_T, 10)  # search_window_size
        push_val(buf, SIZE_T, 10)  # search_buffer_capacity
        push_val(buf, SIZE_T, 64)  # construction_window_size
        push_val(buf, SIZE_T, 750)  # max_candidate_pool_size
        push_val(buf, SIZE_T, 28)  # prune_to
        push_val(buf, BOOL, False)  # use_full_search_history
        push_val(buf, INT, storage_kind)

    def _push_svs_ivf_prefix(self, buf, storage_kind):
        # IndexSVSIVF on-disk fields up to and including storage_kind, for
        # the validation guard at the ISIQ/ISIL/ISID read sites.
        push_index_header(buf, 8, 0)
        push_val(buf, SIZE_T, 8)  # num_centroids
        push_val(buf, SIZE_T, 64)  # minibatch_size
        push_val(buf, SIZE_T, 1)  # num_iterations
        push_val(buf, BOOL, False)  # is_hierarchical
        push_val(buf, FLOAT, 0.1)  # training_fraction
        push_val(buf, SIZE_T, 0)  # hierarchical_level1_clusters
        push_val(buf, SIZE_T, 0)  # seed
        push_val(buf, SIZE_T, 1)  # n_probes
        push_val(buf, FLOAT, 1.0)  # k_reorder (float, not size_t)
        push_val(buf, SIZE_T, 1)  # num_threads
        push_val(buf, SIZE_T, 1)  # intra_query_threads
        push_val(buf, INT, storage_kind)

    @unittest.skipUnless(SVS_ENABLED, "requires SVS")
    def test_svs_vamana_invalid_storage_kind(self):
        # An invalid storage_kind is rejected at deserialization time, not
        # aborted via FAISS_ASSERT in to_svs_storage_kind().
        buf = bytearray()
        push_fourcc(buf, "ISVD")
        self._push_svs_vamana_prefix(buf, faiss.SVS_count)
        push_val(buf, BOOL, True)  # initialized
        self.expect_read_throws(buf, "storage_kind")

    @unittest.skipUnless(SVS_ENABLED, "requires SVS")
    def test_svs_vamana_negative_storage_kind(self):
        # The shared validator takes a signed int because READ1 reads four
        # bytes that could be either sign.
        buf = bytearray()
        push_fourcc(buf, "ISVD")
        self._push_svs_vamana_prefix(buf, -1)
        push_val(buf, BOOL, True)
        self.expect_read_throws(buf, "storage_kind")

    # ISIQ/ISIL/ISID share one storage_kind read site.

    @unittest.skipUnless(SVS_ENABLED, "requires SVS")
    def test_svsivf_invalid_storage_kind(self):
        buf = bytearray()
        push_fourcc(buf, "ISID")
        self._push_svs_ivf_prefix(buf, faiss.SVS_count)
        push_val(buf, BOOL, True)  # is_static
        push_val(buf, BOOL, True)  # initialized
        self.expect_read_throws(buf, "storage_kind")

    @unittest.skipUnless(SVS_ENABLED, "requires SVS")
    def test_svsivflvq_invalid_storage_kind(self):
        buf = bytearray()
        push_fourcc(buf, "ISIQ")
        self._push_svs_ivf_prefix(buf, -1)
        push_val(buf, BOOL, True)
        push_val(buf, BOOL, True)
        self.expect_read_throws(buf, "storage_kind")

    @unittest.skipUnless(SVS_ENABLED, "requires SVS")
    def test_svsivf_lean_vec_invalid_storage_kind(self):
        buf = bytearray()
        push_fourcc(buf, "ISIL")
        self._push_svs_ivf_prefix(buf, faiss.SVS_count + 7)
        push_val(buf, BOOL, True)  # is_static
        push_val(buf, SIZE_T, 8)  # leanvec_d (only on the ISIL path)
        push_val(buf, BOOL, True)  # initialized
        self.expect_read_throws(buf, "storage_kind")

    @unittest.skipUnless(SVS_ENABLED, "requires SVS")
    def test_svs_vamana_invalid_stream_throws(self):
        # Invalid SVS stream data throws from the SVS runtime load failure
        # rather than a null-pointer dereference (deserialize_impl called
        # impl->load() on a null impl pointer).
        buf = bytearray()
        push_fourcc(buf, "ISVD")
        self._push_svs_vamana_prefix(buf, 0)  # storage_kind = SVS_Float16
        push_val(buf, BOOL, False)  # is_static
        push_val(buf, BOOL, True)  # initialized -> deserialize_impl
        buf += bytes(256)  # garbage SVS stream data
        self.expect_read_throws(buf)

    @unittest.skipUnless(SVS_ENABLED, "requires SVS")
    def test_svs_flat_invalid_stream_throws(self):
        buf = bytearray()
        push_fourcc(buf, "ISVF")
        push_index_header(buf, 8, 0)
        push_val(buf, BOOL, True)  # initialized -> deserialize_impl
        buf += bytes(256)  # garbage SVS stream data
        self.expect_read_throws(buf)

    # k_factor must be finite and in [1, 1000].

    def _index_refine_buf(self, fourcc, k_factor):
        d = 4
        buf = bytearray()
        push_fourcc(buf, fourcc)
        push_index_header(buf, d, 0)
        push_minimal_flat(buf, d)  # base_index
        push_minimal_flat(buf, d)  # refine_index
        push_val(buf, FLOAT, k_factor)
        return buf

    def test_index_refine_k_factor_valid(self):
        read_index(self._index_refine_buf("IxRF", 1.0))

    def test_index_refine_k_factor_max(self):
        read_index(self._index_refine_buf("IxRF", 1000.0))

    def test_index_refine_k_factor_too_large(self):
        buf = self._index_refine_buf("IxRF", 1001.0)
        self.expect_read_throws(buf, "k_factor")

    def test_index_refine_k_factor_zero(self):
        self.expect_read_throws(self._index_refine_buf("IxRF", 0.0), "k_factor")

    def test_index_refine_k_factor_nan(self):
        buf = self._index_refine_buf("IxRF", float("nan"))
        self.expect_read_throws(buf, "k_factor")

    def _make_ivfpqr_bytes(self, k_factor):
        # nbits=4 (16 centroids) so that training succeeds with few vectors.
        d = 8
        quantizer = faiss.IndexFlatL2(d)
        ivfpqr = faiss.IndexIVFPQR(quantizer, d, 1, 2, 4, 2, 4)
        ntrain = 64
        train_data = (
            np.arange(d * ntrain, dtype="float32") / np.float32(d * ntrain)
        ).reshape(ntrain, d)
        ivfpqr.train(train_data)
        ivfpqr.k_factor = k_factor
        return serialize_index(ivfpqr)

    def _patch_last_float(self, buf, new_val):
        # k_factor is the last field written for IVFPQR.
        struct.pack_into("<f", buf, len(buf) - 4, new_val)

    def test_index_ivfpqr_k_factor_too_large(self):
        buf = self._make_ivfpqr_bytes(4.0)
        self._patch_last_float(buf, 1e10)
        self.expect_read_throws(buf, "k_factor")

    def test_index_ivfpqr_k_factor_negative(self):
        buf = self._make_ivfpqr_bytes(4.0)
        self._patch_last_float(buf, -1.0)
        self.expect_read_throws(buf, "k_factor")

    def test_index_ivfpqr_k_factor_valid(self):
        buf = self._make_ivfpqr_bytes(64.0)  # AutoTune max
        self.assertEqual(read_index(buf).k_factor, 64.0)

    @unittest.skipIf(SVS_ENABLED, "SVS fourccs are readable with SVS")
    def test_svs_vamana_fourcc_rejected(self):
        buf = bytearray()
        push_fourcc(buf, "ISVD")
        push_index_header(buf, 8, 0)
        buf += bytes(128)
        self.expect_read_throws(buf, "not recognized")

    # IndexIDMap invariant: id_map.size(), idxmap.ntotal and the inner index
    # ntotal must agree. The outer IDMap header and the inner index header
    # carry independent ntotal fields on disk.

    def test_index_id_map_inner_ntotal_mismatch_rejected(self):
        # Outer ntotal=0 agrees with id_map.size()=0, inner ntotal=3.
        buf = make_idmap_payload("IxMp", 0, 3, 0)
        self.expect_read_throws(buf, "inner index ntotal")

    def test_index_binary_id_map_inner_ntotal_mismatch_rejected(self):
        buf = bytearray()
        push_fourcc(buf, "IBMp")
        push_binary_index_header(buf, 8, 0)
        push_fourcc(buf, "IBxF")
        push_binary_index_header(buf, 8, 3)
        push_vector(buf, UINT8, [0] * 3)  # ntotal * code_size
        push_vector(buf, INT64, [])  # id_map
        self.expect_binary_read_throws(buf, "inner index ntotal")

    # IndexIDMap with a "null" sub-index fourcc. read_index returns nullptr
    # for it, so the inner-ntotal cross-check would dereference
    # idxmap->index without the "inner index is null" guard.
    # The binary IBMp/IBM2 path is reached indirectly (an embedded float
    # IxMp/IxM2 inside a binary wrapper such as IndexBinaryFromFloat) and is
    # covered by the same float-side guard.

    def test_index_id_map_null_inner_rejected(self):
        buf = bytearray()
        push_fourcc(buf, "IxMp")
        push_index_header(buf, 4, 0)
        push_fourcc(buf, "null")
        push_vector(buf, INT64, [])  # id_map
        self.expect_read_throws(buf, "inner index is null")

    def test_index_binary_from_float_null_inner_rejected(self):
        # A null inner index would null-deref on the first search or add.
        d = 8
        buf = bytearray()
        push_fourcc(buf, "IBFf")
        push_binary_index_header(buf, d, 0)
        push_fourcc(buf, "null")
        self.expect_binary_read_throws(buf, "inner index is null")

    def test_index_id_map_matching_ntotals_accepted(self):
        read_index(make_idmap_payload("IxMp", 0, 0, 0))

    # Panorama deserialization and reconstruct bounds (T287094917).
    # The IxFP branch does not go through read_index_header, so its own
    # range checks are all that guard d and ntotal.

    def test_flat_panorama_dimension_out_of_range(self):
        # A d large enough to make code_size = d * 4 overflow into
        # gigabytes is rejected, as it is for every other index type.
        buf = bytearray()
        push_flat_panorama(buf, 1280725833, 1, 1024, 0)
        self.expect_read_throws(buf, "out of range")

    def test_flat_panorama_negative_ntotal(self):
        buf = bytearray()
        push_flat_panorama(buf, 8, 1, 4, -1)
        self.expect_read_throws(buf, "invalid ntotal")

    def test_flat_panorama_zero_batch_size_rejected(self):
        buf = bytearray()
        push_flat_panorama(buf, 8, 1, 0, 2)
        self.expect_read_throws(buf, "invalid IxFP batch_size")

    def test_flat_panorama_codes_size_overflow_rejected(self):
        # batch_size=SIZE_MAX, ntotal=2: num_slots = 1 * SIZE_MAX does not
        # overflow, the later num_slots * code_size multiplication does.
        buf = bytearray()
        push_flat_panorama(buf, 8, 1, SIZE_MAX, 2)
        self.expect_read_throws(
            buf, "integer overflow in IndexFlatPanorama codes"
        )

    def test_flat_panorama_codes_size_mismatch(self):
        buf = bytearray()
        push_flat_panorama(buf, 8, 1, 4, 2)
        self.expect_read_throws(buf, "IndexFlatPanorama codes size mismatch")

    def test_pre_transform_over_empty_flat_panorama_reconstruct(self):
        # The fuzzer crash shape: an IndexPreTransform with an empty chain
        # around an empty IndexFlatPanorama. Everything deserializes, and
        # reconstruct() must reject the out-of-range key instead of copying
        # code_size bytes out of a zero-length codes buffer.
        d = 8
        buf = bytearray()
        push_fourcc(buf, "IxPT")
        push_index_header(buf, d, 0)
        push_val(buf, INT, 0)  # nt = 0, empty chain
        push_flat_panorama(buf, d, 1, 4, 0)
        idx = read_index(buf)
        with self.assertRaisesFaiss():
            idx.reconstruct(0)

    def test_hnsw_flat_panorama_cum_sums_size_mismatch(self):
        # "IHfP": fourcc + index_header + n_levels + cum_sums vec + HNSW
        # graph + storage index. cum_sums must hold ntotal * (n_levels + 1)
        # floats; a short vector is an out-of-bounds read on search.
        buf = bytearray()
        push_fourcc(buf, "IHfP")
        push_index_header(buf, 4, 1)
        push_val(buf, SIZE_T, 1)  # n_levels
        push_vector(buf, FLOAT, [0.0])  # cum_sums: 1 float, needs 2
        self.expect_read_throws(buf, "IndexHNSWFlatPanorama cum_sums")

    def test_hnsw_flat_panorama_zero_levels(self):
        buf = bytearray()
        push_fourcc(buf, "IHfP")
        push_index_header(buf, 4, 0)
        push_val(buf, SIZE_T, 0)  # n_levels = 0
        self.expect_read_throws(buf, "invalid IHfP n_levels")


class TestIndexIDMapSafety(FaissTestCase):
    def test_search_remaps_out_of_range_labels_to_sentinel(self):
        # IndexIDMap::search_ex maps each inner label through id_map. When
        # inner.ntotal exceeds id_map.size() (mid-construction, or after a
        # recoverable failure in add_with_ids_ex / add_sa_codes /
        # merge_from), inner labels can land outside id_map and must map to
        # the -1 "no result" sentinel.
        d = 4
        nb = 3
        xb = np.zeros((nb, d), dtype="float32")
        xb[:, 0] = np.arange(nb)

        inner = faiss.IndexFlat(d, faiss.METRIC_L2)
        idmap = faiss.IndexIDMap(inner)
        inner.add(xb)  # inner.ntotal == nb, idmap.id_map is empty

        _, labels = idmap.search(np.zeros((1, d), dtype="float32"), 1)
        self.assertEqual(labels[0, 0], -1)


class TestIndexFlatPanoramaSafety(FaissTestCase):
    def test_reconstruct_rejects_out_of_range_key(self):
        # IndexFlatPanorama overrides reconstruct()/reconstruct_n() and does
        # not inherit the range guards of IndexFlat / IndexFlatCodes. The
        # level-major layout means an out-of-range key reads whole batch
        # strides past the end of codes.
        # reconstruct_n_c is the swig method: the Python reconstruct_n
        # wrapper maps ni=-1 to ntotal - n0 and allocates (ni, d), so it
        # cannot pass the raw arguments.
        d = 8
        index = faiss.IndexFlatL2Panorama(d, 2, 4)
        recons = np.zeros(2 * d, dtype="float32")
        recons_ptr = faiss.swig_ptr(recons)

        index.add(np.ones((2, d), dtype="float32"))

        index.reconstruct(1)
        with self.assertRaisesFaiss():
            index.reconstruct(2)
        with self.assertRaisesFaiss():
            index.reconstruct(-1)
        index.reconstruct_n_c(0, 2, recons_ptr)
        bad_ranges = [(-1, 1), (0, -1), (1, INT64_MAX), (1, 2), (3, 0)]
        for n0, ni in bad_ranges:
            with self.assertRaisesFaiss():
                index.reconstruct_n_c(n0, ni, recons_ptr)


# -----------------------------------------------------------------------
# IndexPiPNN
# -----------------------------------------------------------------------

PIPNN_DIM = 4

# A valid unbuilt, empty index.
EMPTY_PIPNN = {
    "ntotal": 0,
    "entry_point": -1,
    "graph": (),
    "storage_ntotal": 0,
}


def pipnn_payload(
    metric_type=1,
    R=2,
    pipnn_ntotal=None,
    ntotal=2,
    entry_point=0,
    graph=(1, -1, 0, -1),
    next_copy=(),
    storage_ntotal=2,
    storage_d=PIPNN_DIM,
    panorama_storage=False,
):
    buf = bytearray()
    push_fourcc(buf, "IPNf")
    push_index_header(buf, PIPNN_DIM, ntotal, True, metric_type)
    push_val(buf, BOOL, False)  # defer_build
    push_val(buf, INT, R)
    # leaf_k, hash_bits, reservoir_size, c_max, c_min, leader_cap, replicas
    for value in (2, 12, 64, 1024, 128, 1000, 1):
        push_val(buf, INT, value)
    push_val(buf, FLOAT, 0.005)  # leader_fraction
    push_val(buf, FLOAT, 0.0)  # alpha
    push_vector(buf, INT, (10, 3))  # fanout
    push_val(buf, INT64, 1234)  # seed
    push_val(buf, BOOL, False)  # ip_partition_by_angle
    push_val(buf, INT, 64)  # search_L
    push_val(buf, INT64, ntotal if pipnn_ntotal is None else pipnn_ntotal)
    push_val(buf, INT, entry_point)
    for neighbor in graph:
        push_val(buf, INT, neighbor)
    push_vector(buf, INT, next_copy)
    if panorama_storage:
        push_flat_panorama(buf, PIPNN_DIM, 1, 1024, storage_ntotal)
    else:
        push_minimal_flat(buf, storage_d, storage_ntotal)
    return buf


class TestPiPNNDeserialize(FaissTestCase):

    def test_valid_payloads(self):
        for fields in ({}, EMPTY_PIPNN):
            self.assertTrue(read_index(pipnn_payload(**fields)).own_fields)

    def test_rejects_malformed_payloads(self):
        cases = [
            ({"R": 0}, "R must be >= 1"),
            ({"ntotal": 2**31}, "exceeds INT32_MAX"),
            ({"pipnn_ntotal": 1}, "PiPNN ntotal 1 != index ntotal"),
            ({"graph": (1, 2, 0, -1)}, "out of range"),
            ({"entry_point": 2}, "entry_point"),
            ({"entry_point": -1}, "entry_point"),
            ({"next_copy": (1,)}, "next_copy.size()"),
            ({**EMPTY_PIPNN, "panorama_storage": True}, "unsupported storage"),
            ({"storage_ntotal": 3}, "PiPNN storage ntotal"),
            ({"storage_d": 2 * PIPNN_DIM}, "PiPNN storage d"),
            ({"metric_type": 0}, "PiPNN storage metric"),
        ]
        for fields, msg in cases:
            with self.subTest(**fields):
                self.expect_read_throws(pipnn_payload(**fields), msg)

    def test_corrupt_graph_view_search_stays_in_bounds(self):
        # views skip the neighbour scan on read; search must still stay in
        # bounds on corrupt neighbour and copy ids, above ntotal and negative
        data = np.array(
            pipnn_payload(graph=(1, 99, -5, -1), next_copy=(99, -1)),
            dtype="uint8",
        )
        index = faiss.read_index(
            faiss.ZeroCopyIOReader(faiss.swig_ptr(data), data.size)
        )
        _, labels = index.search(np.zeros((1, PIPNN_DIM), dtype="float32"), 3)
        self.assertTrue(np.all((labels >= -1) & (labels <= 1)), labels)
