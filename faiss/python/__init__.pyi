# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Faiss Python API type stubs
# Provides type information for IDE autocompletion and static type checkers
# (mypy, pyright, Pyre, etc.)

from __future__ import annotations

from typing import Any, Callable, overload

import numpy as np
import numpy.typing as npt

try:
    import torch
except ImportError:
    pass

idx_t = int
MetricType = int
METRIC_INNER_PRODUCT: int
METRIC_L2: int
METRIC_L1: int
METRIC_Linf: int
METRIC_Lp: int
METRIC_Canberra: int
METRIC_BrayCurtis: int
METRIC_JensenShannon: int
METRIC_Jaccard: int
METRIC_NaNEuclidean: int
METRIC_GOWER: int

# I/O flag constants for reading/writing indexes
IO_FLAG_SKIP_STORAGE: int  # skip the storage for graph-based indexes
IO_FLAG_READ_ONLY: int  # read-only mode
IO_FLAG_ONDISK_SAME_DIR: int  # strip directory component from ondisk filename
IO_FLAG_SKIP_IVF_DATA: int  # don't load IVF data to RAM, only list sizes
IO_FLAG_SKIP_PRECOMPUTE_TABLE: int  # don't initialize precomputed table after loading
IO_FLAG_PQ_SKIP_SDC_TABLE: int  # don't compute the sdc table for PQ-based indices
IO_FLAG_MMAP: int  # try to memmap data (useful to load as OnDiskInvertedLists)
IO_FLAG_MMAP_IFC: int  # mmap for IndexFlatCodes-derived indices and HNSW

# Numeric type enum
Float32: int
Float16: int
UInt8: int
Int8: int

def get_numeric_type_size(numeric_type: int) -> int: ...
def normalize_L2(x: npt.NDArray[np.float32]) -> None: ...
def real_to_binary(d: int, x_in: Any, x_out: Any) -> None: ...
def bucket_sort(
    tab: npt.NDArray[np.int64], nbucket: int | None = None, nt: int = 0
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
def matrix_bucket_sort_inplace(
    tab: npt.NDArray[np.int32 | np.int64], nbucket: int | None = None, nt: int = 0
) -> npt.NDArray[np.int64]: ...
def eval_intersection(I1: npt.NDArray[np.int64], I2: npt.NDArray[np.int64]) -> int: ...
def checksum(a: npt.NDArray[np.uint8]) -> int | npt.NDArray[np.uint64]: ...
def rand_smooth_vectors(
    n: int, d: int, seed: int = 1234
) -> npt.NDArray[np.float32]: ...
def merge_knn_results(
    Dall: npt.NDArray[np.float32], Iall: npt.NDArray[np.int64], keep_max: bool = False
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
def knn(
    xq: npt.NDArray[np.float32],
    xb: npt.NDArray[np.float32],
    k: int,
    metric: int = METRIC_L2,
    metric_arg: float = 0.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
def knn_hamming(
    xq: npt.NDArray[np.uint8],
    xb: npt.NDArray[np.uint8],
    k: int,
    variant: str = "hc",
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
def pack_bitstrings(
    a: npt.NDArray[np.int32], nbit: int | npt.NDArray[np.int32]
) -> npt.NDArray[np.uint8]: ...
def unpack_bitstrings(
    b: npt.NDArray[np.uint8],
    M_or_nbits: int | npt.NDArray[np.int32],
    nbit: int | None = None,
) -> npt.NDArray[np.int32]: ...

# Version information
FAISS_VERSION_MAJOR: int
FAISS_VERSION_MINOR: int
FAISS_VERSION_PATCH: int

# Vector types (std::vector templates)
class Float32Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: float) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...  # float*
    def size(self) -> int: ...
    def at(self, n: int) -> float: ...
    def __getitem__(self, n: int) -> float: ...
    def __setitem__(self, n: int, val: float) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: Float32Vector) -> None: ...

class Float64Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: float) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> float: ...
    def __getitem__(self, n: int) -> float: ...
    def __setitem__(self, n: int, val: float) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: Float64Vector) -> None: ...

class Int8Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...

class Int16Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: Int16Vector) -> None: ...

class Int32Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: Int32Vector) -> None: ...

class Int64Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: Int64Vector) -> None: ...

class UInt8Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: UInt8Vector) -> None: ...

class UInt16Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: UInt16Vector) -> None: ...

class UInt32Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: UInt32Vector) -> None: ...

class UInt64Vector:
    def __init__(self) -> None: ...
    def push_back(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def data(self) -> Any: ...
    def size(self) -> int: ...
    def at(self, n: int) -> int: ...
    def __getitem__(self, n: int) -> int: ...
    def __setitem__(self, n: int, val: int) -> None: ...
    def resize(self, n: int) -> None: ...
    def swap(self, other: UInt64Vector) -> None: ...

# Forward declarations
class IDSelector: ...
class DistanceComputer: ...

# Range search result structure
class RangeSearchResult:
    nq: int  # number of queries
    lims: Any  # size_t* - size (nq + 1) array
    labels: Any  # idx_t* - result for query i is labels[lims[i]:lims[i+1]]
    distances: Any  # float* - corresponding distances (not sorted)
    buffer_size: int  # size of the result buffers used

    def __init__(self, nq: int, alloc_lims: bool = True) -> None: ...
    def do_allocation(self) -> None: ...

# IDSelector implementations
class IDSelectorTranslated(IDSelector):
    def __init__(
        self,
        id_map: npt.NDArray[np.int64] | list[int],
        sel: IDSelector,
    ) -> None: ...
    def is_member(self, id: int) -> bool: ...

# Search parameters
class SearchParameters:
    sel: IDSelector | None
    def __init__(self) -> None: ...

class SearchParametersPreTransform(SearchParameters):
    index_params: SearchParameters | None
    def __init__(self) -> None: ...

# Base Index class
class Index:
    d: int  # vector dimension
    ntotal: int  # total number of indexed vectors
    verbose: bool  # verbosity level
    is_trained: bool  # whether the index is trained
    metric_type: MetricType  # metric type for search
    metric_arg: float  # metric argument

    def __init__(self, d: int = 0, metric: MetricType = METRIC_L2) -> None: ...
    # Python wrapper interface (what users actually see)
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.float32]) -> None: ...
    @overload
    def add(self, x: torch.Tensor) -> None: ...
    @overload
    def add(self, x: npt.NDArray[np.float32]) -> None: ...
    @overload
    def add_with_ids(self, x: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_with_ids(
        self, x: npt.NDArray[np.float32], ids: npt.NDArray[np.int64]
    ) -> None: ...
    @overload
    def search(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search(
        self,
        x: torch.Tensor,
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search(
        self,
        x: npt.NDArray[np.float32],
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.int64]
    ]: ...
    @overload
    def assign(self, x: torch.Tensor, k: int = 1) -> torch.Tensor: ...
    @overload
    def assign(
        self, x: npt.NDArray[np.float32], k: int = 1
    ) -> npt.NDArray[np.int64]: ...
    def reset(self) -> None: ...
    def remove_ids(self, sel: IDSelector) -> int: ...
    @overload
    def reconstruct(self, key: int, x: torch.Tensor | None = None) -> torch.Tensor: ...
    @overload
    def reconstruct(
        self, key: int, x: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def reconstruct_batch(
        self, keys: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def reconstruct_batch(
        self,
        keys: npt.NDArray[np.int64],
        x: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def reconstruct_n(
        self, n0: int = 0, ni: int = -1, x: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def reconstruct_n(
        self, n0: int = 0, ni: int = -1, x: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def search_and_reconstruct(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
        R: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def search_and_reconstruct(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
        R: npt.NDArray[np.float32] | None = None,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.float32]
    ]: ...
    def compute_residual(
        self, x: npt.NDArray[np.float32], residual: npt.NDArray[np.float32], key: int
    ) -> None: ...
    def compute_residual_n(
        self,
        n: int,
        xs: npt.NDArray[np.float32],
        residuals: npt.NDArray[np.float32],
        keys: npt.NDArray[np.int64],
    ) -> None: ...
    def get_distance_computer(self) -> DistanceComputer: ...

    # Standalone codec interface with tensor overloads
    def sa_code_size(self) -> int: ...
    @overload
    def sa_encode(
        self, x: torch.Tensor, codes: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def sa_encode(
        self,
        x: npt.NDArray[np.float32],
        codes: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    def sa_decode(
        self, codes: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def sa_decode(
        self,
        codes: npt.NDArray[np.uint8],
        x: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def add_sa_codes(
        self, codes: torch.Tensor, ids: torch.Tensor | None = None
    ) -> None: ...
    @overload
    def add_sa_codes(
        self, codes: npt.NDArray[np.uint8], ids: npt.NDArray[np.int64] | None = None
    ) -> None: ...
    def merge_from(self, other_index: Index, add_id: int = 0) -> None: ...
    def check_compatible_for_merge(self, other_index: Index) -> None: ...
    @overload
    def search_and_return_codes(
        self,
        x: torch.Tensor,
        k: int,
        include_listnos: bool = False,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
        codes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def search_and_return_codes(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        include_listnos: bool = False,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
        codes: npt.NDArray[np.uint8] | None = None,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.uint8]
    ]: ...
    @overload
    def update_vectors(self, keys: torch.Tensor, x: torch.Tensor) -> None: ...
    @overload
    def update_vectors(
        self, keys: npt.NDArray[np.int64], x: npt.NDArray[np.float32]
    ) -> None: ...
    @overload
    def permute_entries(self, perm: torch.Tensor) -> None: ...
    @overload
    def permute_entries(self, perm: npt.NDArray[np.int64]) -> None: ...

# Vector transform classes
class VectorTransform:
    d_in: int  # input dimension
    d_out: int  # output dimension
    is_trained: bool

    def __init__(self, d_in: int = 0, d_out: int = 0) -> None: ...
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.float32]) -> None: ...
    @overload
    def apply(self, x: torch.Tensor) -> torch.Tensor: ...
    @overload
    def apply(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    @overload
    def apply_py(self, x: torch.Tensor) -> torch.Tensor: ...
    @overload
    def apply_py(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    @overload
    def reverse_transform(
        self, xt: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def reverse_transform(
        self,
        xt: npt.NDArray[np.float32],
        x: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]: ...
    def check_identical(self, other: VectorTransform) -> None: ...

class LinearTransform(VectorTransform):
    have_bias: bool
    is_orthonormal: bool
    A: Float32Vector  # transformation matrix
    b: Float32Vector  # bias vector
    verbose: bool

    def __init__(
        self, d_in: int = 0, d_out: int = 0, have_bias: bool = False
    ) -> None: ...
    def set_is_orthonormal(self) -> None: ...
    def check_identical(self, other: VectorTransform) -> None: ...

class RandomRotationMatrix(LinearTransform):
    def __init__(self, d_in: int, d_out: int) -> None: ...
    def init(self, seed: int) -> None: ...

class PCAMatrix(LinearTransform):
    eigen_power: float
    epsilon: float
    random_rotation: bool
    max_points_per_d: int
    balanced_bins: int
    mean: Float32Vector
    eigenvalues: Float32Vector
    PCAMat: Float32Vector

    def __init__(
        self,
        d_in: int = 0,
        d_out: int = 0,
        eigen_power: float = 0,
        random_rotation: bool = False,
    ) -> None: ...
    def copy_from(self, other: PCAMatrix) -> None: ...
    def prepare_Ab(self) -> None: ...
    # Explicit overloads for apply method to ensure torch.Tensor support
    @overload
    def apply(self, x: torch.Tensor) -> torch.Tensor: ...
    @overload
    def apply(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class ITQMatrix(LinearTransform):
    max_iter: int
    seed: int
    init_rotation: Float64Vector

    def __init__(self, d: int = 0) -> None: ...

class ITQTransform(VectorTransform):
    mean: Float32Vector
    do_pca: bool
    itq: ITQMatrix
    max_train_per_dim: int
    pca_then_itq: LinearTransform

    def __init__(self, d_in: int = 0, d_out: int = 0, do_pca: bool = False) -> None: ...
    def check_identical(self, other: VectorTransform) -> None: ...

class OPQMatrix(LinearTransform):
    M: int  # number of subquantizers
    niter: int
    niter_pq: int
    niter_pq_0: int
    max_train_points: int
    verbose: bool
    pq: ProductQuantizer | None

    def __init__(self, d: int = 0, M: int = 1, d2: int = -1) -> None: ...

class RemapDimensionsTransform(VectorTransform):
    map: Int32Vector

    def __init__(
        self,
        d_in: int,
        d_out: int,
        map: npt.NDArray[np.int32] | None = None,
        uniform: bool = True,
    ) -> None: ...
    def check_identical(self, other: VectorTransform) -> None: ...

class NormalizationTransform(VectorTransform):
    norm: float

    def __init__(self, d: int, norm: float = 2.0) -> None: ...
    def check_identical(self, other: VectorTransform) -> None: ...

class CenteringTransform(VectorTransform):
    mean: Float32Vector

    def __init__(self, d: int = 0) -> None: ...
    def check_identical(self, other: VectorTransform) -> None: ...

# Specific Index implementations
class IndexFlat(Index):
    def __init__(self, d: int, metric: MetricType = METRIC_L2) -> None: ...
    @overload
    def search(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search(
        self,
        x: torch.Tensor,
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search(
        self,
        x: npt.NDArray[np.float32],
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.int64]
    ]: ...
    @overload
    def reconstruct(
        self, key: int, recons: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def reconstruct(
        self, key: int, recons: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]: ...
    def get_xb(self) -> npt.NDArray[np.float32]: ...

class IndexFlatIP(IndexFlat):
    def __init__(self, d: int) -> None: ...

class IndexFlatL2(IndexFlat):
    cached_l2norms: Float32Vector

    def __init__(self, d: int) -> None: ...
    def sync_l2norms(self) -> None: ...
    def clear_l2norms(self) -> None: ...

class IndexFlat1D(IndexFlatL2):
    continuous_update: bool
    perm: Int64Vector

    def __init__(self, continuous_update: bool = True) -> None: ...
    def update_permutation(self) -> None: ...
    def reset(self) -> None: ...

class IndexPreTransform(Index):
    chain: list[VectorTransform]  # std::vector<VectorTransform*> chain
    index: Index
    own_fields: bool

    @overload
    def __init__(self, index: Index) -> None: ...
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, ltrans: VectorTransform, index: Index) -> None: ...
    def prepend_transform(self, ltrans: VectorTransform) -> None: ...
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.float32]) -> None: ...
    @overload
    def add(self, x: torch.Tensor) -> None: ...
    @overload
    def add(self, x: npt.NDArray[np.float32]) -> None: ...
    @overload
    def add_with_ids(self, x: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_with_ids(
        self, x: npt.NDArray[np.float32], ids: npt.NDArray[np.int64]
    ) -> None: ...
    def reset(self) -> None: ...
    def remove_ids(self, sel: IDSelector) -> int: ...
    @overload
    def search(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search(
        self,
        x: torch.Tensor,
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search(
        self,
        x: npt.NDArray[np.float32],
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.int64]
    ]: ...
    @overload
    def reconstruct(self, key: int, x: torch.Tensor | None = None) -> torch.Tensor: ...
    @overload
    def reconstruct(
        self, key: int, x: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def reconstruct_n(
        self, n0: int = 0, ni: int = -1, x: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def reconstruct_n(
        self, n0: int = 0, ni: int = -1, x: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def search_and_reconstruct(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
        R: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def search_and_reconstruct(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
        R: npt.NDArray[np.float32] | None = None,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.float32]
    ]: ...
    def apply_chain(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def reverse_chain(
        self, xt: npt.NDArray[np.float32], x: npt.NDArray[np.float32]
    ) -> None: ...
    def get_distance_computer(self) -> DistanceComputer: ...
    def sa_code_size(self) -> int: ...
    @overload
    def sa_encode(
        self, x: torch.Tensor, codes: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def sa_encode(
        self,
        x: npt.NDArray[np.float32],
        codes: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    def sa_decode(
        self, codes: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def sa_decode(
        self,
        codes: npt.NDArray[np.uint8],
        x: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]: ...
    def merge_from(self, other_index: Index, add_id: int = 0) -> None: ...
    def check_compatible_for_merge(self, other_index: Index) -> None: ...

# Quantizer classes
class Quantizer:
    d: int
    code_size: int
    is_trained: bool

    def __init__(self, d: int = 0, code_size: int = 0) -> None: ...
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.float32]) -> None: ...
    @overload
    def compute_codes(self, x: torch.Tensor) -> torch.Tensor: ...
    @overload
    def compute_codes(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]: ...
    @overload
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...
    @overload
    def decode(self, codes: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]: ...

class ProductQuantizer(Quantizer):
    # Core attributes from C++ struct
    M: int  # number of subquantizers
    nbits: int  # bits per subquantizer

    # Derived values
    dsub: int  # dimensionality of each subvector
    ksub: int  # number of centroids for each subquantizer

    verbose: bool  # verbose during training?

    # Training configuration
    train_type: int  # enum train_type_t
    Train_default: int
    Train_hot_start: int
    Train_shared: int
    Train_hypercube: int
    Train_hypercube_pca: int

    cp: ClusteringParameters
    assign_index: Index | None  # optional index for assignment

    # Centroid storage
    centroids: Float32Vector  # M * ksub * dsub layout
    transposed_centroids: Float32Vector  # dsub * M * ksub layout
    centroids_sq_lengths: Float32Vector  # M * ksub layout

    # Symmetric Distance Table
    sdc_table: Float32Vector

    def __init__(self, d: int = 0, M: int = 0, nbits: int = 0) -> None: ...

    # Basic encoding/decoding
    def compute_code(
        self, x: npt.NDArray[np.float32], code: npt.NDArray[np.uint8]
    ) -> None: ...
    def compute_codes_with_assign_index(
        self, x: npt.NDArray[np.float32], codes: npt.NDArray[np.uint8], n: int
    ) -> None: ...
    def decode(
        self, code: npt.NDArray[np.uint8], x: npt.NDArray[np.float32]
    ) -> None: ...

    # Distance computation methods
    def compute_distance_table(
        self, x: npt.NDArray[np.float32], dis_table: npt.NDArray[np.float32]
    ) -> None: ...
    def compute_inner_prod_table(
        self, x: npt.NDArray[np.float32], dis_table: npt.NDArray[np.float32]
    ) -> None: ...
    def compute_distance_tables(
        self, nx: int, x: npt.NDArray[np.float32], dis_tables: npt.NDArray[np.float32]
    ) -> None: ...
    def compute_inner_prod_tables(
        self, nx: int, x: npt.NDArray[np.float32], dis_tables: npt.NDArray[np.float32]
    ) -> None: ...

    # Advanced encoding methods
    def compute_code_from_distance_table(
        self, tab: npt.NDArray[np.float32], code: npt.NDArray[np.uint8]
    ) -> None: ...

    # Search methods
    def search(
        self,
        x: npt.NDArray[np.float32],
        nx: int,
        codes: npt.NDArray[np.uint8],
        ncodes: int,
        res: Any,  # float_maxheap_array_t*
        init_finalize_heap: bool = True,
    ) -> None: ...
    def search_ip(
        self,
        x: npt.NDArray[np.float32],
        nx: int,
        codes: npt.NDArray[np.uint8],
        ncodes: int,
        res: Any,  # float_minheap_array_t*
        init_finalize_heap: bool = True,
    ) -> None: ...
    def search_sdc(
        self,
        qcodes: npt.NDArray[np.uint8],
        nq: int,
        bcodes: npt.NDArray[np.uint8],
        ncodes: int,
        res: Any,  # float_maxheap_array_t*
        init_finalize_heap: bool = True,
    ) -> None: ...

    # Centroid management
    def set_derived_values(self) -> None: ...
    def set_params(self, centroids: npt.NDArray[np.float32], m: int) -> None: ...
    def get_centroids(self, m: int, i: int) -> npt.NDArray[np.float32]: ...
    def sync_transposed_centroids(self) -> None: ...
    def clear_transposed_centroids(self) -> None: ...
    def compute_sdc_table(self) -> None: ...

class AdditiveQuantizer(Quantizer):
    # Core attributes from C++ struct
    M: int  # number of codebooks
    nbits: Int32Vector  # bits for each step (variable length)
    codebooks: Float32Vector  # codebooks

    # Derived values
    codebook_offsets: Int64Vector  # codebook offsets
    tot_bits: int  # total number of bits
    norm_bits: int  # bits allocated for norms
    total_codebook_size: int  # size of codebook in vectors
    only_8bit: bool  # are all nbits = 8 (use faster decoder)

    verbose: bool  # verbose during training
    is_trained: bool  # is trained or not

    # Auxiliary data for special search types
    norm_tabs: Float32Vector  # norms of codebook entries for 4-bit fastscan
    qnorm: IndexFlat1D  # store and search norms
    centroid_norms: Float32Vector  # norms of all codebook entries
    codebook_cross_products: Float32Vector  # dot products with previous codebooks
    max_mem_distances: int  # memory limit for beam search

    # Search type configuration
    search_type: int  # Search_type_t enum value
    norm_min: float  # min for quantization of norms
    norm_max: float  # max for quantization of norms

    # Search type constants
    ST_decompress: int
    ST_LUT_nonorm: int
    ST_norm_from_LUT: int
    ST_norm_float: int
    ST_norm_qint8: int
    ST_norm_qint4: int
    ST_norm_cqint8: int
    ST_norm_cqint4: int
    ST_norm_lsq2x4: int
    ST_norm_rq2x4: int

    @overload
    def __init__(self, d: int, nbits: list[int], search_type: int = 0) -> None: ...
    @overload
    def __init__(self) -> None: ...
    def set_derived_values(self) -> None: ...
    @overload
    def train_norm(self, n: int, norms: torch.Tensor) -> None: ...
    @overload
    def train_norm(self, n: int, norms: npt.NDArray[np.float32]) -> None: ...
    @overload
    def compute_codes_add_centroids(
        self,
        x: torch.Tensor,
        codes: torch.Tensor,
        n: int,
        centroids: torch.Tensor | None = None,
    ) -> None: ...
    @overload
    def compute_codes_add_centroids(
        self,
        x: npt.NDArray[np.float32],
        codes: npt.NDArray[np.uint8],
        n: int,
        centroids: npt.NDArray[np.float32] | None = None,
    ) -> None: ...
    def pack_codes(
        self,
        n: int,
        codes: npt.NDArray[np.int32],
        packed_codes: npt.NDArray[np.uint8],
        ld_codes: int = -1,
        norms: npt.NDArray[np.float32] | None = None,
        centroids: npt.NDArray[np.float32] | None = None,
    ) -> None: ...
    @overload
    def decode_unpacked(
        self,
        codes: npt.NDArray[np.int32],
        x: npt.NDArray[np.float32],
        n: int,
        ld_codes: int = -1,
    ) -> None: ...
    def decode_64bit(self, n: int, x: npt.NDArray[np.float32]) -> None: ...
    @overload
    def compute_LUT(
        self,
        n: int,
        xq: npt.NDArray[np.float32],
        LUT: npt.NDArray[np.float32],
        alpha: float = 1.0,
        ld_lut: int = -1,
    ) -> None: ...
    def knn_centroids_inner_product(
        self,
        n: int,
        xq: npt.NDArray[np.float32],
        k: int,
        distances: npt.NDArray[np.float32],
        labels: npt.NDArray[np.int64],
    ) -> None: ...
    def compute_centroid_norms(self, norms: npt.NDArray[np.float32]) -> None: ...
    def knn_centroids_L2(
        self,
        n: int,
        xq: npt.NDArray[np.float32],
        k: int,
        distances: npt.NDArray[np.float32],
        labels: npt.NDArray[np.int64],
        centroid_norms: npt.NDArray[np.float32],
    ) -> None: ...
    def encode_norm(self, norm: float) -> int: ...
    def encode_qcint(self, x: float) -> int: ...
    def decode_qcint(self, c: int) -> float: ...

class ResidualQuantizer(AdditiveQuantizer):
    # Training configuration
    train_type: int  # train_type_t enum value

    # Training type constants
    Train_default: int
    Train_progressive_dim: int
    Train_refine_codebook: int
    Train_top_beam: int
    Skip_codebook_tables: int

    niter_codebook_refine: int  # iterations for codebook refinement
    max_beam_size: int  # beam size for training and encoding
    use_beam_LUT: int  # use LUT for beam search
    approx_topk_mode: int  # ApproxTopK_mode_t enum value

    # Clustering parameters
    cp: ProgressiveDimClusteringParameters
    assign_index_factory: ProgressiveDimIndexFactory

    @overload
    def __init__(self, d: int, nbits: list[int], search_type: int = 0) -> None: ...
    @overload
    def __init__(self, d: int, M: int, nbits: int, search_type: int = 0) -> None: ...
    @overload
    def __init__(self) -> None: ...
    def initialize_from(self, other: ResidualQuantizer, skip_M: int = 0) -> None: ...
    @overload
    def retrain_AQ_codebook(self, n: int, x: torch.Tensor) -> float: ...
    @overload
    def retrain_AQ_codebook(self, n: int, x: npt.NDArray[np.float32]) -> float: ...
    def refine_beam(
        self,
        n: int,
        beam_size: int,
        residuals: npt.NDArray[np.float32],
        new_beam_size: int,
        new_codes: npt.NDArray[np.int32],
        new_residuals: npt.NDArray[np.float32] | None = None,
        new_distances: npt.NDArray[np.float32] | None = None,
    ) -> None: ...
    def refine_beam_LUT(
        self,
        n: int,
        query_norms: npt.NDArray[np.float32],
        query_cp: npt.NDArray[np.float32],
        new_beam_size: int,
        new_codes: npt.NDArray[np.int32],
        new_distances: npt.NDArray[np.float32] | None = None,
    ) -> None: ...
    def memory_per_point(self, beam_size: int = -1) -> int: ...
    def train_type_to_str(self, train_type: int) -> str: ...

class IcmEncoder:
    """ICM (Iterated Conditional Modes) encoder for LSQ"""
    def __init__(self, lsq: LocalSearchQuantizer) -> None: ...

class IcmEncoderFactory:
    """Factory class for ICM (Iterated Conditional Modes) encoders in LSQ"""
    def __init__(self) -> None: ...
    def get(self, lsq: LocalSearchQuantizer) -> IcmEncoder: ...

class LocalSearchQuantizer(AdditiveQuantizer):
    K: int  # number of codes per codebook

    # Training parameters
    train_iters: int  # iterations in training
    encode_ils_iters: int  # iterations of local search in encoding
    train_ils_iters: int  # iterations of local search in training
    icm_iters: int  # iterations in ICM

    # Algorithm parameters
    p: float  # temperature factor
    lambd: float  # regularization factor

    # Processing parameters
    chunk_size: int  # vectors to encode at a time
    random_seed: int  # seed for random generator
    nperts: int  # number of perturbations in each code

    # Encoder configuration
    icm_encoder_factory: IcmEncoderFactory | None  # lsq::IcmEncoderFactory*
    update_codebooks_with_double: bool

    @overload
    def __init__(self, d: int, M: int, nbits: int, search_type: int = 0) -> None: ...
    @overload
    def __init__(self) -> None: ...
    @overload
    def update_codebooks(
        self, x: torch.Tensor, codes: npt.NDArray[np.int32], n: int
    ) -> None: ...
    @overload
    def update_codebooks(
        self, x: npt.NDArray[np.float32], codes: npt.NDArray[np.int32], n: int
    ) -> None: ...
    def icm_encode(
        self,
        codes: npt.NDArray[np.int32],
        x: npt.NDArray[np.float32],
        n: int,
        ils_iters: int,
        gen: Any,  # std::mt19937&
    ) -> None: ...
    def icm_encode_impl(
        self,
        codes: npt.NDArray[np.int32],
        x: npt.NDArray[np.float32],
        unaries: npt.NDArray[np.float32],
        gen: Any,  # std::mt19937&
        n: int,
        ils_iters: int,
        verbose: bool,
    ) -> None: ...
    def icm_encode_step(
        self,
        codes: npt.NDArray[np.int32],
        unaries: npt.NDArray[np.float32],
        binaries: npt.NDArray[np.float32],
        n: int,
        n_iters: int,
    ) -> None: ...
    def perturb_codes(
        self,
        codes: npt.NDArray[np.int32],
        n: int,
        gen: Any,  # std::mt19937&
    ) -> None: ...
    def perturb_codebooks(
        self,
        T: float,
        stddev: list[float],
        gen: Any,  # std::mt19937&
    ) -> None: ...
    def compute_binary_terms(self, binaries: npt.NDArray[np.float32]) -> None: ...
    @overload
    def compute_unary_terms(
        self, x: torch.Tensor, unaries: npt.NDArray[np.float32], n: int
    ) -> None: ...
    @overload
    def compute_unary_terms(
        self, x: npt.NDArray[np.float32], unaries: npt.NDArray[np.float32], n: int
    ) -> None: ...
    @overload
    def evaluate(
        self,
        codes: npt.NDArray[np.int32],
        x: torch.Tensor,
        n: int,
        objs: npt.NDArray[np.float32] | None = None,
    ) -> float: ...
    @overload
    def evaluate(
        self,
        codes: npt.NDArray[np.int32],
        x: npt.NDArray[np.float32],
        n: int,
        objs: npt.NDArray[np.float32] | None = None,
    ) -> float: ...

class RaBitQuantizer(Quantizer):
    # Core attributes
    centroid: Any  # float* - pointer to centroid (not serialized)
    metric_type: MetricType  # metric type for the quantizer

    def __init__(self, d: int = 0, metric: MetricType = METRIC_L2) -> None: ...
    @overload
    def compute_codes_core(
        self,
        x: torch.Tensor,
        codes: torch.Tensor,
        n: int,
        centroid_in: torch.Tensor,
    ) -> None: ...
    @overload
    def compute_codes_core(
        self,
        x: npt.NDArray[np.float32],
        codes: npt.NDArray[np.uint8],
        n: int,
        centroid_in: npt.NDArray[np.float32],
    ) -> None: ...
    @overload
    def decode_core(
        self,
        codes: torch.Tensor,
        x: torch.Tensor,
        n: int,
        centroid_in: torch.Tensor,
    ) -> None: ...
    @overload
    def decode_core(
        self,
        codes: npt.NDArray[np.uint8],
        x: npt.NDArray[np.float32],
        n: int,
        centroid_in: npt.NDArray[np.float32],
    ) -> None: ...
    def get_distance_computer(
        self,
        qb: int,
        centroid_in: npt.NDArray[np.float32] | None = None,
    ) -> Any: ...  # FlatCodesDistanceComputer*

class ProductAdditiveQuantizer(AdditiveQuantizer):
    nsplits: int  # number of sub-vectors we split a vector into
    quantizers: list[AdditiveQuantizer]  # sub-additive quantizers

    @overload
    def __init__(
        self,
        d: int,
        aqs: list[AdditiveQuantizer],
        search_type: int = 0,
    ) -> None: ...
    @overload
    def __init__(self) -> None: ...
    def init(
        self,
        d: int,
        aqs: list[AdditiveQuantizer],
        search_type: int,
    ) -> None: ...
    def subquantizer(self, m: int) -> AdditiveQuantizer: ...
    @overload
    def compute_unpacked_codes(
        self,
        x: torch.Tensor,
        codes: npt.NDArray[np.int32],
        n: int,
        centroids: torch.Tensor | None = None,
    ) -> None: ...
    @overload
    def compute_unpacked_codes(
        self,
        x: npt.NDArray[np.float32],
        codes: npt.NDArray[np.int32],
        n: int,
        centroids: npt.NDArray[np.float32] | None = None,
    ) -> None: ...

class ProductLocalSearchQuantizer(ProductAdditiveQuantizer):
    @overload
    def __init__(
        self,
        d: int,
        nsplits: int,
        Msub: int,
        nbits: int,
        search_type: int = 0,
    ) -> None: ...
    @overload
    def __init__(self) -> None: ...

class ProductResidualQuantizer(ProductAdditiveQuantizer):
    @overload
    def __init__(
        self,
        d: int,
        nsplits: int,
        Msub: int,
        nbits: int,
        search_type: int = 0,
    ) -> None: ...
    @overload
    def __init__(self) -> None: ...

class ScalarQuantizer(Quantizer):
    qtype: int
    rangestat: int
    rangestat_arg: float
    d: int
    code_size: int
    trained: Float32Vector

    def __init__(self, d: int = 0, qtype: int = 0) -> None: ...

    # ScalarQuantizer quantization type constants (as class attributes)
    QT_8bit: int
    QT_4bit: int
    QT_8bit_uniform: int
    QT_4bit_uniform: int
    QT_fp16: int
    QT_8bit_direct: int
    QT_6bit: int
    QT_bf16: int
    QT_8bit_direct_signed: int

    # RangeStat constants (as class attributes)
    RS_minmax: int
    RS_meanstd: int
    RS_quantiles: int
    RS_optim: int

# LSH index
class IndexLSH(Index):
    nbits: int
    bytes_per_vec: int
    rrot: RandomRotationMatrix
    bytes: UInt8Vector
    train_thresholds: bool
    sign_bit: float

    def __init__(
        self,
        d: int,
        nbits: int,
        rotate_data: bool = True,
        train_thresholds: bool = True,
    ) -> None: ...

# PQ index
class IndexPQ(Index):
    pq: ProductQuantizer
    codes: UInt8Vector

    def __init__(
        self, d: int, M: int, nbits: int, metric: MetricType = METRIC_L2
    ) -> None: ...

# Scalar Quantizer index
class IndexScalarQuantizer(Index):
    sq: ScalarQuantizer
    codes: UInt8Vector

    def __init__(self, d: int, qtype: int, metric: MetricType = METRIC_L2) -> None: ...

# IVF classes
class InvertedLists:
    nlist: int
    code_size: int

    def __init__(self, nlist: int, code_size: int) -> None: ...
    def list_size(self, list_no: int) -> int: ...
    def get_codes(self, list_no: int) -> npt.NDArray[np.uint8]: ...
    def get_ids(self, list_no: int) -> npt.NDArray[np.int64]: ...
    def add_entries(
        self,
        list_no: int,
        n_entry: int,
        ids: npt.NDArray[np.int64],
        codes: npt.NDArray[np.uint8],
    ) -> int: ...
    def update_entries(
        self,
        list_no: int,
        offset: int,
        n_entry: int,
        ids: npt.NDArray[np.int64],
        codes: npt.NDArray[np.uint8],
    ) -> None: ...
    def resize(self, list_no: int, new_size: int) -> None: ...
    def merge_from(self, other: InvertedLists, add_id: int = 0) -> None: ...

class ArrayInvertedLists(InvertedLists):
    def __init__(self, nlist: int, code_size: int) -> None: ...

# InvertedListsIOHook base class for I/O operations
class InvertedListsIOHook:
    key: str  # fourcc key for identification
    classname: str  # typeid.name for the associated class

    def __init__(self, key: str, classname: str) -> None: ...
    def write(self, ils: InvertedLists, f: IOWriter) -> None: ...
    def read(self, f: IOReader, io_flags: int) -> InvertedLists: ...
    def read_ArrayInvertedLists(
        self,
        f: IOReader,
        io_flags: int,
        nlist: int,
        code_size: int,
        sizes: list[int],
    ) -> InvertedLists: ...

    # Static methods for managing callbacks
    @staticmethod
    def add_callback(cb: InvertedListsIOHook) -> None: ...
    @staticmethod
    def print_callbacks() -> None: ...
    @staticmethod
    def lookup(h: int) -> InvertedListsIOHook: ...
    @staticmethod
    def lookup_classname(classname: str) -> InvertedListsIOHook: ...

# OnDiskOneList structure
class OnDiskOneList:
    size: int  # size of inverted list (entries)
    capacity: int  # allocated size (entries)
    offset: int  # offset in buffer (bytes)

    def __init__(self) -> None: ...

# OnDiskInvertedLists Slot structure
class OnDiskSlot:
    offset: int  # offset in bytes
    capacity: int  # capacity in bytes

    def __init__(self, offset: int = 0, capacity: int = 0) -> None: ...

# OnDiskInvertedLists class for memory-mapped inverted lists
class OnDiskInvertedLists(InvertedLists):
    # Core attributes
    lists: list[OnDiskOneList]  # vector of OnDiskOneList
    slots: Any  # std::list<Slot> - available slots sorted by size
    filename: str  # filename of the mmapped file
    totsize: int  # total size of mmapped region
    ptr: Any  # uint8_t* - mmap base pointer
    read_only: bool  # are inverted lists mapped read-only

    # Slot management (private in C++ but accessible in Python)
    locks: Any  # LockLevels* - thread synchronization
    pf: Any  # OngoingPrefetch* - prefetch management
    prefetch_nthread: int  # number of prefetch threads

    @overload
    def __init__(self, nlist: int, code_size: int, filename: str) -> None: ...
    @overload
    def __init__(self) -> None: ...  # empty constructor for I/O functions

    # Override base InvertedLists methods
    def list_size(self, list_no: int) -> int: ...
    def get_codes(self, list_no: int) -> npt.NDArray[np.uint8]: ...
    def get_ids(self, list_no: int) -> npt.NDArray[np.int64]: ...
    def add_entries(
        self,
        list_no: int,
        n_entry: int,
        ids: npt.NDArray[np.int64],
        codes: npt.NDArray[np.uint8],
    ) -> int: ...
    def update_entries(
        self,
        list_no: int,
        offset: int,
        n_entry: int,
        ids: npt.NDArray[np.int64],
        codes: npt.NDArray[np.uint8],
    ) -> None: ...
    def resize(self, list_no: int, new_size: int) -> None: ...

    # OnDiskInvertedLists specific methods
    def merge_from_multiple(
        self,
        ils: Any,  # const InvertedLists** (C pointer array)
        n_il: int,  # number of InvertedLists
        shift_ids: bool = False,
        verbose: bool = False,
    ) -> int: ...
    def merge_from_1(self, il: InvertedLists, verbose: bool = False) -> int: ...
    def crop_invlists(self, l0: int, l1: int) -> None: ...
    def prefetch_lists(self, list_nos: npt.NDArray[np.int64], nlist: int) -> None: ...

    # Memory management methods
    def do_mmap(self) -> None: ...
    def update_totsize(self, new_totsize: int) -> None: ...
    def resize_locked(self, list_no: int, new_size: int) -> None: ...
    def allocate_slot(self, capacity: int) -> int: ...
    def free_slot(self, offset: int, capacity: int) -> None: ...
    def set_all_lists_sizes(self, sizes: npt.NDArray[np.int64]) -> None: ...

# OnDiskInvertedListsIOHook for I/O operations
class OnDiskInvertedListsIOHook(InvertedListsIOHook):
    def __init__(self) -> None: ...
    def write(self, ils: InvertedLists, f: IOWriter) -> None: ...
    def read(self, f: IOReader, io_flags: int) -> InvertedLists: ...
    def read_ArrayInvertedLists(
        self,
        f: IOReader,
        io_flags: int,
        nlist: int,
        code_size: int,
        sizes: list[int],
    ) -> InvertedLists: ...

class IndexIVF(Index):
    # Core attributes from C++ struct
    invlists: InvertedLists | None
    own_invlists: bool
    code_size: int
    parallel_mode: int
    PARALLEL_MODE_NO_HEAP_INIT: int = 1024
    direct_map: Any  # DirectMap type
    by_residual: bool

    # Additional Python wrapper attributes (legacy compatibility)
    quantizer: Index
    nprobe: int
    nlist: int
    quantizer_trains_alone: str  # char in C++ -> single character string in Python
    own_fields: bool
    cp: ClusteringParameters
    clustering_index: Index
    max_codes: int

    def __init__(
        self, quantizer: Index, d: int, nlist: int, metric: MetricType = METRIC_L2
    ) -> None: ...

    # Core IndexIVF methods from C++
    def add_core(
        self,
        n: int,
        x: npt.NDArray[np.float32],
        xids: npt.NDArray[np.int64] | None = None,
        precomputed_idx: npt.NDArray[np.int64] | None = None,
        inverted_list_context: Any | None = None,
    ) -> None: ...
    def search_preassigned(
        self,
        n: int,
        x: npt.NDArray[np.float32],
        k: int,
        assign: npt.NDArray[np.int64],
        centroid_dis: npt.NDArray[np.float32],
        distances: npt.NDArray[np.float32],
        labels: npt.NDArray[np.int64],
        store_pairs: bool,
        params: IVFSearchParameters | None = None,
        stats: IndexIVFStats | None = None,
    ) -> None: ...
    def range_search_preassigned(
        self,
        nx: int,
        x: npt.NDArray[np.float32],
        radius: float,
        keys: npt.NDArray[np.int64],
        coarse_dis: npt.NDArray[np.float32],
        result: RangeSearchResult,
        store_pairs: bool = False,
        params: Any | None = None,  # IVFSearchParameters
        stats: Any | None = None,  # IndexIVFStats
    ) -> None: ...
    def set_beam_factor(self, beam_factor: float) -> None: ...
    def encode_vectors(
        self,
        n: int,
        x: npt.NDArray[np.float32],
        list_nos: npt.NDArray[np.int64],
        codes: npt.NDArray[np.uint8],
        include_listno: bool = False,
    ) -> None: ...
    def decode_vectors(
        self,
        n: int,
        codes: npt.NDArray[np.uint8],
        list_nos: npt.NDArray[np.int64],
        x: npt.NDArray[np.float32],
    ) -> None: ...
    def train_encoder(
        self,
        n: int,
        x: npt.NDArray[np.float32],
        assign: npt.NDArray[np.int64] | None = None,
    ) -> None: ...
    def train_encoder_num_vectors(self) -> int: ...
    def get_InvertedListScanner(
        self,
        store_pairs: bool = False,
        sel: IDSelector | None = None,
        params: Any | None = None,  # IVFSearchParameters
    ) -> Any: ...  # InvertedListScanner*
    def update_vectors(
        self,
        nv: int,
        idx: npt.NDArray[np.int64],
        v: npt.NDArray[np.float32],
    ) -> None: ...
    def reconstruct_from_offset(
        self, list_no: int, offset: int, recons: npt.NDArray[np.float32]
    ) -> None: ...
    def get_CodePacker(self) -> Any: ...  # CodePacker*
    def copy_subset_to(
        self,
        other: IndexIVF,
        subset_type: int,  # InvertedLists::subset_type_t
        a1: int,
        a2: int,
    ) -> None: ...
    def get_list_size(self, list_no: int) -> int: ...
    def check_ids_sorted(self) -> bool: ...
    def make_direct_map(self, new_maintain_direct_map: bool = True) -> None: ...
    def set_direct_map_type(self, type: int) -> None: ...  # DirectMap::Type
    def replace_invlists(self, invlists: InvertedLists, own: bool = False) -> None: ...
    @overload
    def search_preassigned(
        self,
        x: torch.Tensor,
        k: int,
        Iq: torch.Tensor,
        Dq: torch.Tensor | None,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search_preassigned(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        Iq: npt.NDArray[np.int64],
        Dq: npt.NDArray[np.float32] | None,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

class IndexIVFFlat(IndexIVF):
    def __init__(
        self, quantizer: Index, d: int, nlist: int, metric: MetricType = METRIC_L2
    ) -> None: ...

class IndexIVFPQ(IndexIVF):
    pq: ProductQuantizer
    code_size: int
    by_residual: bool
    use_precomputed_table: int
    polysemous_ht: int

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
    ) -> None: ...
    def precompute_table(self) -> None: ...

class IndexIVFScalarQuantizer(IndexIVF):
    sq: ScalarQuantizer
    code_size: int
    by_residual: bool

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        qtype: int,
        metric: MetricType = METRIC_L2,
        encode_residual: bool = True,
    ) -> None: ...

# IVF FastScan base class
class IndexIVFFastScan(IndexIVF):
    M: int
    nbits: int
    ksub: int
    M2: int
    bbs: int
    qbs: int
    implem: int
    skip: int
    orig_invlists: InvertedLists | None

    def __init__(
        self,
        quantizer: Index | None = None,
        d: int = 0,
        nlist: int = 0,
        code_size: int = 0,
        metric: MetricType = METRIC_L2,
        own_invlists: bool = True,
    ) -> None: ...
    def init_fastscan(
        self,
        fine_quantizer: Quantizer,
        M: int,
        nbits: int,
        nlist: int,
        metric: MetricType,
        bbs: int = 32,
        own_invlists: bool = True,
    ) -> None: ...
    def init_code_packer(self) -> None: ...
    def get_CodePacker(self) -> Any: ...  # CodePacker*
    @overload
    def permute_entries(self, perm: torch.Tensor) -> None: ...
    @overload
    def permute_entries(self, perm: npt.NDArray[np.int64]) -> None: ...
    def reconstruct_orig_invlists(self) -> None: ...

class IndexIVFPQFastScan(IndexIVFFastScan):
    pq: ProductQuantizer
    use_precomputed_table: int
    precomputed_table: AlignedTableFloat32

    def __init__(
        self,
        quantizer: Index | None = None,
        d: int = 0,
        nlist: int = 0,
        M: int = 0,
        nbits: int = 0,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
        own_invlists: bool = True,
    ) -> None: ...
    def train_encoder_num_vectors(self) -> int: ...
    def precompute_table(self) -> None: ...
    def lookup_table_is_3d(self) -> bool: ...

# HNSW classes
class HNSWStats:
    n1: int
    n2: int
    n3: int
    ndis: int
    nreorder: int

class HNSW:
    max_level: int
    entry_point: int
    efConstruction: int
    efSearch: int
    hnsw_stats: HNSWStats
    assign_probas: Float32Vector
    cum_nneighbor_per_level: Int32Vector
    levels: Int32Vector
    graph: Int32Vector

    def __init__(self, M: int = 32) -> None: ...
    def reset(self) -> None: ...

class IndexHNSW(Index):
    hnsw: HNSW
    storage: Index
    own_fields: bool
    reconstruct_from_neighbors: Callable[
        [int, npt.NDArray[np.float32], npt.NDArray[np.float32]], None
    ]

    def __init__(self, d: int, M: int, metric: MetricType = METRIC_L2) -> None: ...

class IndexHNSWFlat(IndexHNSW):
    def __init__(self, d: int, M: int, metric: MetricType = METRIC_L2) -> None: ...

class IndexHNSWPQ(IndexHNSW):
    pq: ProductQuantizer

    def __init__(
        self, d: int, pq: ProductQuantizer, M: int, metric: MetricType = METRIC_L2
    ) -> None: ...

class IndexHNSWSQ(IndexHNSW):
    sq: ScalarQuantizer

    def __init__(
        self, d: int, sq: ScalarQuantizer, M: int, metric: MetricType = METRIC_L2
    ) -> None: ...

class IndexHNSW2Level(IndexHNSW):
    q1: Any  # Level1Quantizer
    pq: ProductQuantizer  # from Index2Layer

    def __init__(
        self,
        d: int,
        q1: Quantizer,
        nlist: int,
        M: int,
        cu: int,
        metric: MetricType = METRIC_L2,
    ) -> None: ...

# Binary Index classes
class IndexBinary:
    d: int
    code_size: int
    ntotal: int
    verbose: bool
    is_trained: bool

    def __init__(self, d: int) -> None: ...
    @overload
    def add(self, x: torch.Tensor) -> None: ...
    @overload
    def add(self, x: npt.NDArray[np.uint8]) -> None: ...
    @overload
    def add_with_ids(self, x: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_with_ids(
        self, x: npt.NDArray[np.uint8], ids: npt.NDArray[np.int64]
    ) -> None: ...
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.uint8]) -> None: ...
    @overload
    def search(
        self,
        x: torch.Tensor,
        k: int,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search(
        self,
        x: npt.NDArray[np.uint8],
        k: int,
        params: SearchParameters | None = None,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search(
        self,
        x: torch.Tensor,
        thresh: int,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search(
        self,
        x: npt.NDArray[np.uint8],
        thresh: int,
        params: SearchParameters | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    @overload
    def reconstruct(self, key: int) -> torch.Tensor: ...
    @overload
    def reconstruct(self, key: int) -> npt.NDArray[np.uint8]: ...
    @overload
    def reconstruct_n(
        self, n0: int = 0, ni: int = -1, x: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def reconstruct_n(
        self, n0: int = 0, ni: int = -1, x: npt.NDArray[np.uint8] | None = None
    ) -> npt.NDArray[np.uint8]: ...
    def reset(self) -> None: ...
    @overload
    def remove_ids(self, x: torch.Tensor) -> int: ...
    @overload
    def remove_ids(self, x: npt.NDArray[np.int64]) -> int: ...
    @overload
    def remove_ids(self, sel: IDSelector) -> int: ...
    @overload
    def assign(self, x: torch.Tensor, k: int = 1) -> torch.Tensor: ...
    @overload
    def assign(self, x: npt.NDArray[np.uint8], k: int = 1) -> npt.NDArray[np.int64]: ...
    @overload
    def search_preassigned(
        self,
        x: torch.Tensor,
        k: int,
        Iq: torch.Tensor,
        Dq: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search_preassigned(
        self,
        x: npt.NDArray[np.uint8],
        k: int,
        Iq: npt.NDArray[np.int64],
        Dq: npt.NDArray[np.int32] | torch.Tensor | None,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search_preassigned(
        self,
        x: torch.Tensor,
        thresh: int,
        Iq: torch.Tensor,
        Dq: torch.Tensor | None,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search_preassigned(
        self,
        x: npt.NDArray[np.uint8],
        thresh: int,
        Iq: npt.NDArray[np.int64],
        Dq: npt.NDArray[np.int32] | None,
        params: SearchParameters | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...

class IndexBinaryFlat(IndexBinary):
    def __init__(self, d: int) -> None: ...

class IndexBinaryIVF(IndexBinary):
    nlist: int
    nprobe: int
    quantizer: IndexBinary
    invlists: InvertedLists
    own_fields: bool
    cp: Any  # ClusteringParameters
    clustering_index: Index | None  # to override index used during clustering
    max_codes: int
    use_heap: bool
    per_invlist_search: bool
    direct_map: Any  # DirectMap type

    def __init__(self, quantizer: IndexBinary, d: int, nlist: int) -> None: ...
    @overload
    def search_preassigned(
        self,
        x: torch.Tensor,
        k: int,
        Iq: torch.Tensor,
        Dq: torch.Tensor | None,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search_preassigned(
        self,
        x: npt.NDArray[np.uint8],
        k: int,
        Iq: npt.NDArray[np.int64],
        Dq: npt.NDArray[np.int32] | None,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.int32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search_preassigned(
        self,
        x: torch.Tensor,
        thresh: int,
        Iq: torch.Tensor,
        Dq: torch.Tensor | None = None,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search_preassigned(
        self,
        x: npt.NDArray[np.uint8],
        thresh: int,
        Iq: npt.NDArray[np.int64],
        Dq: npt.NDArray[np.int32] | None = None,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    def reconstruct_from_offset(
        self, list_no: int, offset: int, key: npt.NDArray[np.uint8] | None = None
    ) -> npt.NDArray[np.uint8]: ...
    def make_direct_map(self, new_maintain_direct_map: bool = True) -> None: ...
    def set_direct_map_type(self, type: int) -> None: ...
    def replace_invlists(self, invlists: InvertedLists, own: bool = False) -> None: ...
    def get_list_size(self, list_no: int) -> int: ...
    def merge_from(self, other: IndexBinaryIVF, add_id: int = 0) -> None: ...
    def check_compatible_for_merge(self, other: IndexBinaryIVF) -> None: ...

class IndexBinaryFromFloat(IndexBinary):
    index: Index

    def __init__(self, index: Index) -> None: ...

class IndexBinaryHNSW(IndexBinary):
    hnsw: Any
    storage: IndexBinary
    own_fields: bool

    def __init__(self, d: int, M: int) -> None: ...

class IndexBinaryHash(IndexBinary):
    nflip: int

    def __init__(self, d: int, nflip: int) -> None: ...

class IndexBinaryMultiHash(IndexBinaryHash):
    maps: UInt8VectorVector

    def __init__(self, d: int, nhash: int, nflip: int) -> None: ...

# Index refinement classes
class IndexRefineSearchParameters(SearchParameters):
    k_factor: float
    base_index_params: SearchParameters | None

class IndexRefine(Index):
    base_index: Index
    refine_index: Index
    own_fields: bool
    own_refine_index: bool
    k_factor: float

    def __init__(
        self, base_index: Index | None = None, refine_index: Index | None = None
    ) -> None: ...
    def reset(self) -> None: ...
    def sa_code_size(self) -> int: ...

class IndexRefineFlat(IndexRefine):
    def __init__(
        self, base_index: Index | None = None, xb: npt.NDArray[np.float32] | None = None
    ) -> None: ...

# FastScan base class
class IndexFastScan(Index):
    M: int
    nbits: int
    ksub: int
    M2: int
    bbs: int
    qbs: int
    implem: int
    skip: int
    quantizer: Quantizer
    codes: AlignedTableUint8

    def __init__(self) -> None: ...

# FastScan PQ index
class IndexPQFastScan(IndexFastScan):
    pq: ProductQuantizer

    def __init__(
        self,
        d: int = 0,
        M: int = 0,
        nbits: int = 0,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
    ) -> None: ...

# Index factory and related functions
def downcast_index(index: Index) -> Index: ...
def downcast_IndexBinary(index: IndexBinary) -> IndexBinary: ...
def downcast_VectorTransform(vt: VectorTransform) -> VectorTransform: ...
def downcast_InvertedLists(il: InvertedLists) -> InvertedLists: ...

# IO Writer/Reader classes - for serialization
class IOWriter:
    def __init__(self) -> None: ...

class IOReader:
    def __init__(self) -> None: ...

class VectorIOWriter(IOWriter):
    data: UInt8Vector
    def __init__(self) -> None: ...

class VectorIOReader(IOReader):
    data: UInt8Vector
    def __init__(self) -> None: ...

class BufferedIOWriter(IOWriter):
    def __init__(self, writer: IOWriter) -> None: ...

class BufferedIOReader(IOReader):
    def __init__(self, reader: IOReader) -> None: ...

class FileIOWriter(IOWriter):
    def __init__(self, fname: str) -> None: ...

class FileIOReader(IOReader):
    def __init__(self, fname: str) -> None: ...

# Index serialization functions (THE MAIN ONES FROM __init__.py)
def serialize_index(index: Index, io_flags: int = 0) -> npt.NDArray[np.uint8]: ...
def deserialize_index(data: npt.NDArray[np.uint8], io_flags: int = 0) -> Index: ...
def serialize_index_binary(index: IndexBinary) -> npt.NDArray[np.uint8]: ...
def deserialize_index_binary(data: npt.NDArray[np.uint8]) -> IndexBinary: ...

# Index file I/O functions with overloads
@overload
def write_index(index: Index, fname: str) -> None: ...
@overload
def write_index(index: Index, writer: IOWriter, io_flags: int = 0) -> None: ...
@overload
def read_index(fname: str, io_flags: int = 0) -> Index: ...
@overload
def read_index(reader: IOReader, io_flags: int = 0) -> Index: ...
@overload
def write_index_binary(index: IndexBinary, fname: str) -> None: ...
@overload
def write_index_binary(index: IndexBinary, writer: IOWriter) -> None: ...
@overload
def read_index_binary(fname: str, io_flags: int = 0) -> IndexBinary: ...
@overload
def read_index_binary(reader: IOReader) -> IndexBinary: ...
def write_VectorTransform(vt: VectorTransform, fname: str) -> None: ...
def read_VectorTransform(fname: str) -> VectorTransform: ...

# InvertedLists I/O functions
def write_InvertedLists(invlists: InvertedLists, writer: IOWriter) -> None: ...
def read_InvertedLists(reader: IOReader, io_flags: int = 0) -> InvertedLists: ...

# Search with parameters functions
@overload
def search_with_parameters(
    index: Index,
    x: torch.Tensor,
    k: int,
    params: SearchParameters | None = None,
    output_stats: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
): ...
@overload
def search_with_parameters(
    index: Index,
    x: npt.NDArray[np.float32],
    k: int,
    params: SearchParameters | None = None,
    output_stats: bool = False,
) -> (
    tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]
    | tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], dict[str, Any]]
): ...
@overload
def range_search_with_parameters(
    index: Index,
    x: torch.Tensor,
    radius: float,
    params: SearchParameters | None = None,
    output_stats: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
): ...
@overload
def range_search_with_parameters(
    index: Index,
    x: npt.NDArray[np.float32],
    radius: float,
    params: SearchParameters | None = None,
    output_stats: bool = False,
) -> (
    tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.int64]]
    | tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.float32],
        npt.NDArray[np.int64],
        dict[str, Any],
    ]
): ...

# IVF Search Parameters
class IVFSearchParameters(SearchParameters):
    nprobe: int
    max_codes: int
    sel: IDSelector | None
    def __init__(self) -> None: ...

# IVF Statistics tracking
class IndexIVFStats:
    nq: int  # number of queries run
    nlist: int  # number of inverted lists scanned
    ndis: int  # number of distances computed
    nheap_updates: int  # number of times the heap was updated
    quantization_time: float  # time spent quantizing vectors (in ms)
    search_time: float  # time spent searching lists (in ms)

    def __init__(self) -> None: ...
    def reset(self) -> None: ...
    def add(self, other: IndexIVFStats) -> None: ...

# Legacy aliases and remapped classes
IndexProxy = IndexReplicas
ConcatenatedInvertedLists = HStackInvertedLists
IndexResidual = IndexResidualQuantizer
SearchParametersIVF = IVFSearchParameters

# TimeoutGuard class for managing timeouts
class TimeoutGuard:
    timeout: float
    def __init__(self, timeout_in_seconds: float) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

# Callback classes for timeout handling
class TimeoutCallback:
    @staticmethod
    def reset(timeout: float) -> None: ...

class PythonInterruptCallback:
    @staticmethod
    def reset() -> None: ...

# Array conversion utilities
def vector_to_array(v: Any) -> npt.NDArray[Any]: ...
def vector_float_to_array(v: Float32Vector) -> npt.NDArray[np.float32]: ...
@overload
def copy_array_to_vector(a: torch.Tensor, v: Any) -> None: ...
@overload
def copy_array_to_vector(a: npt.NDArray[Any], v: Any) -> None: ...
@overload
def copy_array_to_AlignedTable(a: torch.Tensor, v: Any) -> None: ...
@overload
def copy_array_to_AlignedTable(a: npt.NDArray[Any], v: Any) -> None: ...

# Pointer conversion utilities
def swig_ptr(a: npt.NDArray[Any]) -> Any: ...
def rev_swig_ptr(ptr: Any, n: int) -> npt.NDArray[Any]: ...
def cast_integer_to_float_ptr(x: int) -> Any: ...
def cast_integer_to_idx_t_ptr(x: int) -> Any: ...
def cast_integer_to_int_ptr(x: int) -> Any: ...
def cast_integer_to_void_ptr(x: int) -> Any: ...

# Additional vector types not covered yet
class UInt8VectorVector:
    def __init__(self) -> None: ...
    def size(self) -> int: ...
    def at(self, n: int) -> UInt8Vector: ...
    def push_back(self, v: UInt8Vector) -> None: ...
    def resize(self, n: int) -> None: ...

class ParameterRangeVector:
    def __init__(self) -> None: ...
    def size(self) -> int: ...
    def at(self, n: int) -> ParameterRange: ...
    def push_back(self, v: ParameterRange) -> None: ...
    def resize(self, n: int) -> None: ...

class OperatingPointVector:
    def __init__(self) -> None: ...
    def size(self) -> int: ...
    def at(self, n: int) -> OperatingPoint: ...
    def push_back(self, v: OperatingPoint) -> None: ...
    def resize(self, n: int) -> None: ...

# Missing index classes from loader imports
class IndexShards(Index):
    own_fields: bool
    threaded: bool
    successive_ids: bool

    def __init__(self, d: int, threaded: bool = False) -> None: ...
    def add_shard(self, index: Index) -> None: ...
    def remove_shard(self, index: Index) -> None: ...

class IndexReplicas(Index):
    own_fields: bool

    @overload
    def __init__(self, threaded: bool = True) -> None: ...
    @overload
    def __init__(self, d: int, threaded: bool = True) -> None: ...
    def addIndex(self, index: Index) -> None: ...
    def removeIndex(self, index: Index) -> None: ...

class IndexBinaryShards(IndexBinary):
    own_fields: bool
    threaded: bool
    successive_ids: bool

    def __init__(self, d: int, threaded: bool = False) -> None: ...
    def add_shard(self, index: IndexBinary) -> None: ...
    def remove_shard(self, index: IndexBinary) -> None: ...

class IndexBinaryReplicas(IndexBinary):
    own_fields: bool

    def __init__(self, d: int) -> None: ...
    def addIndex(self, index: IndexBinary) -> None: ...
    def removeIndex(self, index: IndexBinary) -> None: ...

class HStackInvertedLists(InvertedLists):
    def __init__(self, nil: int, invlists: list[InvertedLists]) -> None: ...

# Additional classes mentioned in __init__.py
# Clustering parameters base class
class ClusteringParameters:
    niter: int
    nredo: int
    verbose: bool
    spherical: bool
    int_centroids: bool
    update_index: bool
    frozen_centroids: bool
    min_points_per_centroid: int
    max_points_per_centroid: int
    seed: int
    decode_block_size: int
    check_input_data_for_NaNs: bool
    use_faster_subsampling: bool

    def __init__(self) -> None: ...

class ClusteringIterationStats:
    obj: float
    time: float
    time_search: float
    imbalance_factor: float
    nsplit: int

    def __init__(self) -> None: ...

# K-means clustering class
class Clustering(ClusteringParameters):
    d: int
    k: int
    centroids: Float32Vector
    iteration_stats: list[ClusteringIterationStats]

    @overload
    def __init__(self, d: int, k: int) -> None: ...
    @overload
    def __init__(self, d: int, k: int, cp: ClusteringParameters) -> None: ...
    @overload
    def train(
        self,
        x: torch.Tensor,
        index: Index,
        x_weights: torch.Tensor | None = None,
    ) -> None: ...
    @overload
    def train(
        self,
        x: npt.NDArray[np.float32],
        index: Index,
        x_weights: npt.NDArray[np.float32] | None = None,
    ) -> None: ...
    @overload
    def train_encoded(
        self,
        x: torch.Tensor,
        codec: Index,
        index: Index,
        weights: torch.Tensor | None = None,
    ) -> None: ...
    @overload
    def train_encoded(
        self,
        x: npt.NDArray[np.uint8],
        codec: Index,
        index: Index,
        weights: npt.NDArray[np.float32] | None = None,
    ) -> None: ...
    def post_process_centroids(self) -> None: ...

class Clustering1D(Clustering):
    @overload
    def __init__(self, k: int) -> None: ...
    @overload
    def __init__(self, k: int, cp: ClusteringParameters) -> None: ...
    @overload
    def train_exact(self, x: torch.Tensor) -> None: ...
    @overload
    def train_exact(self, x: npt.NDArray[np.float32]) -> None: ...

class ProgressiveDimClusteringParameters(ClusteringParameters):
    progressive_dim_steps: int  # number of incremental steps
    apply_pca: bool  # apply PCA on input

    def __init__(self) -> None: ...

class ProgressiveDimIndexFactory:
    """Generates an index suitable for clustering when called"""

    def __call__(self, dim: int) -> Index: ...
    # Note: ownership transferred to caller

class ProgressiveDimClustering(ProgressiveDimClusteringParameters):
    """K-means clustering with progressive dimensions used

    The clustering first happens in dim 1, then with exponentially increasing
    dimension until d (I steps). This is typically applied after a PCA
    transformation (optional). Reference:

    "Improved Residual Vector Quantization for High-dimensional Approximate
    Nearest Neighbor Search"

    Shicong Liu, Hongtao Lu, Junru Shao, AAAI'15

    https://arxiv.org/abs/1509.05195
    """

    d: int  # dimension of the vectors
    k: int  # nb of centroids
    centroids: Float32Vector  # centroids (k * d)
    iteration_stats: list[
        ClusteringIterationStats
    ]  # stats at every iteration of clustering

    @overload
    def __init__(self, d: int, k: int) -> None: ...
    @overload
    def __init__(
        self, d: int, k: int, cp: ProgressiveDimClusteringParameters
    ) -> None: ...
    @overload
    def train(self, x: torch.Tensor, factory: ProgressiveDimIndexFactory) -> None: ...
    @overload
    def train(
        self, x: npt.NDArray[np.float32], factory: ProgressiveDimIndexFactory
    ) -> None: ...

# Standalone k-means clustering function
@overload
def kmeans_clustering(
    d: int,
    n: int,
    k: int,
    x: torch.Tensor,
    centroids: torch.Tensor | None = None,
) -> float: ...
@overload
def kmeans_clustering(
    d: int,
    n: int,
    k: int,
    x: npt.NDArray[np.float32],
    centroids: npt.NDArray[np.float32] | None = None,
) -> float: ...

class MatrixStats:
    comments: str
    n: int
    d: int
    n_collision: int
    hash_value: int

    @overload
    def __init__(self, x: torch.Tensor) -> None: ...
    @overload
    def __init__(self, x: npt.NDArray[np.float32]) -> None: ...
    def reset(self) -> None: ...

# AutoTune related classes (from AutoTune.h)
class AutoTuneCriterion:
    nq: int
    nnn: int
    gt_nnn: int
    gt_D: Float32Vector
    gt_I: Int64Vector

    def __init__(self, nq: int, nnn: int) -> None: ...
    @overload
    def set_groundtruth(
        self,
        gt_nnn: int,
        gt_D_in: torch.Tensor,
        gt_I_in: torch.Tensor,
    ) -> None: ...
    @overload
    def set_groundtruth(
        self,
        gt_nnn: int,
        gt_D_in: npt.NDArray[np.float32],
        gt_I_in: npt.NDArray[np.int64],
    ) -> None: ...
    @overload
    def evaluate(self, D: torch.Tensor, I: torch.Tensor) -> float: ...
    @overload
    def evaluate(
        self, D: npt.NDArray[np.float32], I: npt.NDArray[np.int64]
    ) -> float: ...

class OneRecallAtRCriterion(AutoTuneCriterion):
    R: int

    def __init__(self, nq: int, R: int) -> None: ...

class IntersectionCriterion(AutoTuneCriterion):
    R: int

    def __init__(self, nq: int, R: int) -> None: ...

# Neural Network classes from NeuralNet.h
class Tensor2D:
    shape: tuple[int, int]
    v: Float32Vector

    @overload
    def __init__(
        self, n0: int, n1: int, data: npt.NDArray[np.float32] | None = None
    ) -> None: ...
    @overload
    def __init__(self, array: npt.NDArray[np.float32]) -> None: ...
    def numel(self) -> int: ...
    def data(self) -> npt.NDArray[np.float32]: ...
    def numpy(self) -> npt.NDArray[np.float32]: ...
    def column(self, j: int) -> Tensor2D: ...
    def __iadd__(self, other: Tensor2D) -> Tensor2D: ...

class Int32Tensor2D:
    shape: tuple[int, int]
    v: Int32Vector

    @overload
    def __init__(
        self, n0: int, n1: int, data: npt.NDArray[np.int32] | None = None
    ) -> None: ...
    @overload
    def __init__(self, array: npt.NDArray[np.int32]) -> None: ...
    def numel(self) -> int: ...
    def data(self) -> npt.NDArray[np.int32]: ...
    def numpy(self) -> npt.NDArray[np.int32]: ...
    def column(self, j: int) -> Int32Tensor2D: ...
    def __iadd__(self, other: Int32Tensor2D) -> Int32Tensor2D: ...

# Neural network layer classes from NeuralNet.h
class Linear:
    in_features: int
    out_features: int
    weight: Float32Vector
    bias: Float32Vector

    @overload
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None: ...
    @overload
    def __init__(self, torch_linear: Any) -> None: ...  # torch.nn.Linear
    def __call__(self, x: Tensor2D) -> Tensor2D: ...
    def from_torch(self, linear: Any) -> None: ...  # torch.nn.Linear
    def from_array(
        self,
        array: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32] | None = None,
    ) -> None: ...

class Embedding:
    num_embeddings: int
    embedding_dim: int
    weight: Float32Vector

    @overload
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    @overload
    def __init__(self, torch_embedding: Any) -> None: ...  # torch.nn.Embedding
    def __call__(self, indices: Int32Tensor2D) -> Tensor2D: ...
    def data(self) -> npt.NDArray[np.float32]: ...
    def from_torch(self, emb: Any) -> None: ...  # torch.nn.Embedding
    def from_array(self, array: npt.NDArray[np.float32]) -> None: ...

class FFN:
    linear1: Linear
    linear2: Linear

    def __init__(self, d: int, h: int) -> None: ...
    def __call__(self, x: Tensor2D) -> Tensor2D: ...

# QINCo neural net codec classes from NeuralNet.h
class QINCoStep:
    d: int
    K: int
    L: int
    h: int
    codebook: Embedding
    MLPconcat: Linear
    residual_blocks: list[FFN]

    @overload
    def __init__(self, d: int, K: int, L: int, h: int) -> None: ...
    @overload
    def __init__(self, torch_qinco_step: Any) -> None: ...  # torch QINCoStep
    def get_residual_block(self, i: int) -> FFN: ...
    def encode(
        self, xhat: Tensor2D, x: Tensor2D, residuals: Tensor2D | None = None
    ) -> Int32Tensor2D: ...
    def decode(self, xhat: Tensor2D, codes: Int32Tensor2D) -> Tensor2D: ...
    def from_torch(self, step: Any) -> None: ...  # torch QINCoStep

class NeuralNetCodec:
    d: int
    M: int

    def __init__(self, d: int, M: int) -> None: ...
    def decode(self, codes: Int32Tensor2D) -> Tensor2D: ...
    def encode(self, x: Tensor2D) -> Int32Tensor2D: ...

class QINCo(NeuralNetCodec):
    K: int
    L: int
    h: int
    codebook0: Embedding
    steps: list[QINCoStep]

    @overload
    def __init__(self, d: int, K: int, L: int, M: int, h: int) -> None: ...
    @overload
    def __init__(self, torch_qinco: Any) -> None: ...  # torch QINCo
    def get_step(self, i: int) -> QINCoStep: ...
    def decode(self, codes: Int32Tensor2D) -> Tensor2D: ...
    def encode(self, x: Tensor2D) -> Int32Tensor2D: ...
    def from_torch(self, qinco: Any) -> None: ...  # torch QINCo

# Code set utility from utils.h
class CodeSet:
    d: int

    def __init__(self, d: int) -> None: ...
    @overload
    def insert(
        self, codes: torch.Tensor, inserted: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @overload
    def insert(
        self,
        codes: npt.NDArray[np.uint8],
        inserted: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.bool_]: ...

# ID selector implementations (from impl/IDSelector.h) with correct C++ signatures
class IDSelectorRange(IDSelector):
    imin: int
    imax: int
    assume_sorted: bool

    def __init__(self, imin: int, imax: int, assume_sorted: bool = False) -> None: ...
    def is_member(self, id: int) -> bool: ...
    def find_sorted_ids_bounds(
        self,
        list_size: int,
        ids: npt.NDArray[np.int64],
    ) -> tuple[int, int]: ...

class IDSelectorArray(IDSelector):
    n: int
    # Note: C++ uses const idx_t* ids (raw pointer), Python wrapper handles this
    @overload
    def __init__(self, n: int, ids: npt.NDArray[np.int64]) -> None: ...
    @overload
    def __init__(
        self, ids: npt.NDArray[np.int64]
    ) -> None: ...  # Python wrapper convenience
    def is_member(self, id: int) -> bool: ...

class IDSelectorBatch(IDSelector):
    # C++ has std::unordered_set<idx_t> set and bloom filter internals

    @overload
    def __init__(self, n: int, indices: npt.NDArray[np.int64]) -> None: ...
    @overload
    def __init__(
        self, indices: npt.NDArray[np.int64]
    ) -> None: ...  # Python wrapper convenience
    def is_member(self, id: int) -> bool: ...

class IDSelectorBitmap(IDSelector):
    n: int
    # Note: C++ uses const uint8_t* bitmap (raw pointer), Python wrapper handles this
    def __init__(self, n: int, bitmap: npt.NDArray[np.uint8]) -> None: ...
    def is_member(self, id: int) -> bool: ...

class IDSelectorNot(IDSelector):
    sel: IDSelector

    def __init__(self, sel: IDSelector) -> None: ...
    def is_member(self, id: int) -> bool: ...

class IDSelectorAll(IDSelector):
    def __init__(self) -> None: ...
    def is_member(self, id: int) -> bool: ...

class IDSelectorAnd(IDSelector):
    lhs: IDSelector
    rhs: IDSelector

    def __init__(self, lhs: IDSelector, rhs: IDSelector) -> None: ...
    def is_member(self, id: int) -> bool: ...

class IDSelectorOr(IDSelector):
    lhs: IDSelector
    rhs: IDSelector

    def __init__(self, lhs: IDSelector, rhs: IDSelector) -> None: ...
    def is_member(self, id: int) -> bool: ...

class IDSelectorXOr(IDSelector):
    lhs: IDSelector
    rhs: IDSelector

    def __init__(self, lhs: IDSelector, rhs: IDSelector) -> None: ...
    def is_member(self, id: int) -> bool: ...

# CodePacker classes (from impl/CodePacker.h) with correct C++ API
class CodePacker:
    code_size: int
    nvec: int
    block_size: int

    # Abstract base class - no direct constructor
    def pack_1(
        self,
        flat_code: npt.NDArray[np.uint8],
        offset: int,
        block: npt.NDArray[np.uint8],
    ) -> None: ...
    def unpack_1(
        self,
        block: npt.NDArray[np.uint8],
        offset: int,
        flat_code: npt.NDArray[np.uint8],
    ) -> None: ...
    def pack_all(
        self,
        flat_codes: npt.NDArray[np.uint8],
        block: npt.NDArray[np.uint8],
    ) -> None: ...
    def unpack_all(
        self,
        block: npt.NDArray[np.uint8],
        flat_codes: npt.NDArray[np.uint8],
    ) -> None: ...

class CodePackerFlat(CodePacker):
    def __init__(self, code_size: int) -> None: ...

# Utility functions
def get_mem_usage_kb() -> int: ...
def get_compile_options() -> str: ...
def check_openmp() -> bool: ...
def shard_ivf_index_centroids(
    index: IndexIVF,
    shard_count: int = 20,
    filename_template: str = "shard.%d.index",
    sharding_function: ShardingFunction | None = None,
    generate_ids: bool = False,
) -> None: ...
def shard_binary_ivf_index_centroids(
    index: IndexBinaryIVF,
    shard_count: int = 20,
    filename_template: str = "shard.%d.index",
    sharding_function: ShardingFunction | None = None,
    generate_ids: bool = False,
) -> None: ...

# IVF extraction utility functions (from IVFlib.h)
def extract_index_ivf(index: Index) -> IndexIVF: ...
def try_extract_index_ivf(index: Index) -> IndexIVF | None: ...

# IVFlib utility functions and classes (from IVFlib.h)
def check_compatible_for_merge(index1: Index, index2: Index) -> None: ...
def merge_into(index0: Index, index1: Index, shift_ids: bool) -> None: ...
def search_centroid(
    index: Index,
    x: npt.NDArray[np.float32],
    n: int,
    centroid_ids: npt.NDArray[np.int64],
) -> None: ...
def search_and_return_centroids(
    index: Index,
    n: int,
    xin: npt.NDArray[np.float32],
    k: int,
    distances: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    query_centroid_ids: npt.NDArray[np.int64],
    result_centroid_ids: npt.NDArray[np.int64],
) -> None: ...
def get_invlist_range(index: Index, i0: int, i1: int) -> ArrayInvertedLists: ...
def set_invlist_range(
    index: Index, i0: int, i1: int, src: ArrayInvertedLists
) -> None: ...
def search_with_parameters(
    index: Index,
    n: int,
    x: npt.NDArray[np.float32],
    k: int,
    distances: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    params: IVFSearchParameters,
    nb_dis: npt.NDArray[np.int64] | None = None,
    ms_per_stage: npt.NDArray[np.float64] | None = None,
) -> None: ...
def range_search_with_parameters(
    index: Index,
    n: int,
    x: npt.NDArray[np.float32],
    radius: float,
    result: RangeSearchResult,
    params: IVFSearchParameters,
    nb_dis: npt.NDArray[np.int64] | None = None,
    ms_per_stage: npt.NDArray[np.float64] | None = None,
) -> None: ...
def ivf_residual_from_quantizer(
    rq: ResidualQuantizer, nlevel: int
) -> IndexIVFResidualQuantizer: ...
def ivf_residual_add_from_flat_codes(
    ivfrq: IndexIVFResidualQuantizer,
    ncode: int,
    codes: npt.NDArray[np.uint8],
    code_size: int = -1,
) -> None: ...

class SlidingIndexWindow:
    index: Index
    ils: ArrayInvertedLists
    n_slice: int
    nlist: int
    sizes: list[list[int]]

    def __init__(self, index: Index) -> None: ...
    def step(self, sub_index: Index | None, remove_oldest: bool) -> None: ...

class ShardingFunction:
    def __call__(self, i: int, shard_count: int) -> int: ...

class DefaultShardingFunction(ShardingFunction):
    def __call__(self, i: int, shard_count: int) -> int: ...

# Version information
__version__: str

# Logger
logger: Any

# Additional missing index types from SWIG includes

# IDMap index classes - for mapping external IDs to internal indices
class IndexIDMap(Index):
    index: Index
    own_fields: bool
    id_map: Int64Vector

    def __init__(self, index: Index) -> None: ...
    # Override methods that don't use add() directly
    @overload
    def add_with_ids(self, x: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_with_ids(
        self, x: npt.NDArray[np.float32], ids: npt.NDArray[np.int64]
    ) -> None: ...
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.float32]) -> None: ...
    def reset(self) -> None: ...
    def remove_ids(self, sel: IDSelector) -> int: ...
    @overload
    def search(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
        D: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search(
        self,
        x: npt.NDArray[np.float32],
        k: int,
        *,
        params: SearchParameters | None = None,
        D: npt.NDArray[np.float32] | None = None,
        I: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search(
        self,
        x: torch.Tensor,
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search(
        self,
        x: npt.NDArray[np.float32],
        thresh: float,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.int64]
    ]: ...
    def sa_code_size(self) -> int: ...
    @overload
    def add_sa_codes(self, codes: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_sa_codes(
        self, codes: npt.NDArray[np.uint8], ids: npt.NDArray[np.int64]
    ) -> None: ...
    def merge_from(self, other_index: Index, add_id: int = 0) -> None: ...
    def check_compatible_for_merge(self, other_index: Index) -> None: ...

class IndexBinaryIDMap(IndexBinary):
    index: IndexBinary
    own_fields: bool
    id_map: Int64Vector

    def __init__(self, index: IndexBinary) -> None: ...
    # Override methods that don't use add() directly
    @overload
    def add_with_ids(self, x: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_with_ids(
        self, x: npt.NDArray[np.uint8], ids: npt.NDArray[np.int64]
    ) -> None: ...
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.uint8]) -> None: ...
    def reset(self) -> None: ...
    def remove_ids(self, sel: IDSelector) -> int: ...
    @overload
    def search(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search(
        self,
        x: npt.NDArray[np.uint8],
        k: int,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    @overload
    def range_search(
        self,
        x: torch.Tensor,
        thresh: int,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @overload
    def range_search(
        self,
        x: npt.NDArray[np.uint8],
        thresh: int,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    @overload
    def reconstruct(self, key: int) -> torch.Tensor: ...
    @overload
    def reconstruct(self, key: int) -> npt.NDArray[np.uint8]: ...

class IndexIDMap2(IndexIDMap):
    rev_map: Any  # std::unordered_map<idx_t, idx_t>

    def __init__(self, index: Index) -> None: ...
    @overload
    def add_with_ids(self, x: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_with_ids(
        self, x: npt.NDArray[np.float32], ids: npt.NDArray[np.int64]
    ) -> None: ...
    def check_consistency(self) -> None: ...
    def construct_rev_map(self) -> None: ...
    def merge_from(self, other_index: Index, add_id: int = 0) -> None: ...
    def remove_ids(self, sel: IDSelector) -> int: ...
    @overload
    def reconstruct(self, key: int, x: torch.Tensor | None = None) -> torch.Tensor: ...
    @overload
    def reconstruct(
        self, key: int, x: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]: ...

class IndexBinaryIDMap2(IndexBinaryIDMap):
    rev_map: Any  # std::unordered_map<idx_t, idx_t>

    def __init__(self, index: IndexBinary) -> None: ...
    @overload
    def add_with_ids(self, x: torch.Tensor, ids: torch.Tensor) -> None: ...
    @overload
    def add_with_ids(
        self, x: npt.NDArray[np.uint8], ids: npt.NDArray[np.int64]
    ) -> None: ...
    def check_consistency(self) -> None: ...
    def construct_rev_map(self) -> None: ...
    def merge_from(self, other_index: IndexBinary, add_id: int = 0) -> None: ...
    def remove_ids(self, sel: IDSelector) -> int: ...
    @overload
    def reconstruct(self, key: int) -> torch.Tensor: ...
    @overload
    def reconstruct(self, key: int) -> npt.NDArray[np.uint8]: ...

# IndexNSG (Neural Sparse Graph)
class IndexNSG(Index):
    nsg: Any  # faiss::nsg::Graph
    base_index: Index
    GK: int
    build_type: int
    verbose: bool

    def __init__(self, d: int, R: int, metric: MetricType = METRIC_L2) -> None: ...
    @overload
    def build(self, x: torch.Tensor, graph: torch.Tensor) -> None: ...
    @overload
    def build(
        self, x: npt.NDArray[np.float32], graph: npt.NDArray[np.int64]
    ) -> None: ...

class IndexNSGFlat(IndexNSG):
    def __init__(self, d: int, R: int) -> None: ...

class IndexNSGPQ(IndexNSG):
    pq: ProductQuantizer

    def __init__(self, d: int, pq: ProductQuantizer, R: int) -> None: ...

class IndexNSGSQ(IndexNSG):
    sq: ScalarQuantizer

    def __init__(self, d: int, sq: ScalarQuantizer, R: int) -> None: ...

# IndexNNDescent
class IndexNNDescent(Index):
    nndescent: Any  # faiss::nndescent::NNDescent
    storage: Index
    own_fields: bool

    def __init__(self, d: int, K: int, metric: MetricType = METRIC_L2) -> None: ...

class IndexNNDescentFlat(IndexNNDescent):
    def __init__(self, d: int, K: int, metric: MetricType = METRIC_L2) -> None: ...

# Index2Layer
class Index2Layer(Index):
    q1: Quantizer
    pq: ProductQuantizer
    code_size_1: int
    code_size_2: int
    code_size: int
    codes: UInt8Vector

    def __init__(
        self,
        quantizer: Index,
        nlist: int,
        M: int,
        nbit: int = 8,
        metric: MetricType = METRIC_L2,
    ) -> None: ...
    def transfer_to_IVFPQ(self, other: IndexIVFPQ) -> None: ...

# IndexIVFPQR (with Refine)
class IndexIVFPQR(IndexIVFPQ):
    k_factor: float
    refine_index: Index
    own_refine_index: bool

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbits: int,
        M_refine: int,
        nbits_refine: int,
        metric: MetricType = METRIC_L2,
    ) -> None: ...

# IndexLattice
class IndexLattice(Index):
    d: int
    dsub: int
    scale_nbit: int
    r2: float

    def __init__(self, d: int, dsub: int = 0, dsuper: int = 0) -> None: ...

# IndexRowwiseMinMax
class IndexRowwiseMinMax(IndexFlat):
    def __init__(self, d: int, metric: MetricType = METRIC_L2) -> None: ...

class IndexRowwiseMinMaxFP16(Index):
    def __init__(self, d: int, metric: MetricType = METRIC_L2) -> None: ...

# IndexRandom for testing
class IndexRandom(Index):
    d: int

    def __init__(self, d: int, seed: int = 1234) -> None: ...

# IndexShards and IndexReplicas (already templated in SWIG)
class IndexShardsIVF(Index):
    own_fields: bool
    threaded: bool
    successive_ids: bool

    def __init__(
        self, quantizer: Index, d: int, nlist: int, threaded: bool = False
    ) -> None: ...
    def add_shard(self, index: Index) -> None: ...

# Missing IVF Fast Scan variants
class IndexIVFResidualQuantizer(IndexIVF):
    rq: ResidualQuantizer
    code_size: int
    by_residual: bool

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        encode_residual: bool = True,
    ) -> None: ...

class IndexIVFLocalSearchQuantizer(IndexIVF):
    lsq: LocalSearchQuantizer
    code_size: int
    by_residual: bool

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        encode_residual: bool = True,
    ) -> None: ...

# FastScan variants for IVF
class IndexIVFResidualQuantizerFastScan(IndexIVFFastScan):
    rq: ResidualQuantizer

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
    ) -> None: ...

class IndexIVFLocalSearchQuantizerFastScan(IndexIVFFastScan):
    lsq: LocalSearchQuantizer

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
    ) -> None: ...

# Product variants
class IndexProductResidualQuantizer(Index):
    rq: ResidualQuantizer
    codes: UInt8Vector

    def __init__(
        self, d: int, M: int, nbits: int, metric: MetricType = METRIC_L2
    ) -> None: ...

class IndexProductLocalSearchQuantizer(Index):
    lsq: LocalSearchQuantizer
    codes: UInt8Vector

    def __init__(
        self, d: int, M: int, nbits: int, metric: MetricType = METRIC_L2
    ) -> None: ...

# IVF Product variants
class IndexIVFProductResidualQuantizer(IndexIVF):
    pq: ProductQuantizer
    rq: ResidualQuantizer
    code_size: int
    by_residual: bool

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        nsplits: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        encode_residual: bool = True,
    ) -> None: ...

class IndexIVFProductLocalSearchQuantizer(IndexIVF):
    pq: ProductQuantizer
    lsq: LocalSearchQuantizer
    code_size: int
    by_residual: bool

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        nsplits: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        encode_residual: bool = True,
    ) -> None: ...

# FastScan product variants
class IndexProductResidualQuantizerFastScan(IndexFastScan):
    prq: ProductResidualQuantizer

    def __init__(
        self,
        d: int,
        nsplits: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
    ) -> None: ...

class IndexProductLocalSearchQuantizerFastScan(IndexFastScan):
    plsq: ProductLocalSearchQuantizer

    def __init__(
        self,
        d: int,
        nsplits: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
    ) -> None: ...

class IndexIVFProductResidualQuantizerFastScan(IndexIVFFastScan):
    prq: ProductResidualQuantizer

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        nsplits: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
    ) -> None: ...

class IndexIVFProductLocalSearchQuantizerFastScan(IndexIVFFastScan):
    plsq: ProductLocalSearchQuantizer

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        nsplits: int,
        M: int,
        nbits: int,
        metric: MetricType = METRIC_L2,
        bbs: int = 32,
    ) -> None: ...

# Additional missing classes
class MultiIndexQuantizer(Index):
    pq: ProductQuantizer

    def __init__(self, d: int, M: int, nbits: int) -> None: ...

class ResidualCoarseQuantizer(Index):
    rq: ResidualQuantizer

    def __init__(self, d: int, M: int, nbits: int) -> None: ...

class LocalSearchCoarseQuantizer(Index):
    lsq: LocalSearchQuantizer

    def __init__(self, d: int, M: int, nbits: int) -> None: ...

# NeuralNet index
class IndexNeuralNetCodec(Index):
    d: int

    def __init__(self, d: int, filename: str) -> None: ...

# RaBitQ indices
class IndexRaBitQ(Index):
    rq: RaBitQuantizer
    codes: UInt8Vector

    def __init__(
        self,
        d: int,
        M: int,
        nbit: int = 1,
        metric: MetricType = METRIC_L2,
        trained: bool = True,
    ) -> None: ...

class IndexIVFRaBitQ(IndexIVF):
    rq: RaBitQuantizer
    code_size: int
    by_residual: bool

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbit: int = 1,
        metric: MetricType = METRIC_L2,
        encode_residual: bool = True,
    ) -> None: ...

# Independent quantizer
class IndexIVFIndependentQuantizer(Index):
    quantizer: Index
    index_ivf: Index
    own_fields: bool

    def __init__(self, quantizer: Index, index_ivf: Index) -> None: ...

# IVF Spectral Hash
class IndexIVFSpectralHash(IndexIVF):
    vt: VectorTransform
    threshold_type: int
    period: int
    trained: UInt8Vector

    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        nbit: int,
        metric: MetricType = METRIC_L2,
        period: int = 64,
    ) -> None: ...
    def replace_vt(self, vt: VectorTransform) -> None: ...

# Missing Binary index variants
class IndexBinaryHNSWCagra(IndexBinaryHNSW):
    def __init__(self, d: int, M: int = 32) -> None: ...

# AutoTune related classes
class ParameterRange:
    name: str
    values: Float64Vector

    def __init__(self, name: str = "") -> None: ...

class OperatingPoint:
    perf: float
    t: float
    key: str
    cno: int

    def __init__(self) -> None: ...

# ParameterSpace from AutoTune.h with complete API
# IndexIVFInterface from IndexIVF.h
class IndexIVFInterface:
    nprobe: int  # number of probes at query time
    max_codes: int  # max nb of codes to visit to do a query

    def __init__(self, quantizer: Index | None = None, nlist: int = 0) -> None: ...
    def search_preassigned(
        self,
        n: int,
        x: npt.NDArray[np.float32],
        k: int,
        assign: npt.NDArray[np.int64],
        centroid_dis: npt.NDArray[np.float32],
        distances: npt.NDArray[np.float32],
        labels: npt.NDArray[np.int64],
        store_pairs: bool,
        params: IVFSearchParameters | None = None,
        stats: IndexIVFStats | None = None,
    ) -> None: ...
    def range_search_preassigned(
        self,
        nx: int,
        x: npt.NDArray[np.float32],
        radius: float,
        keys: npt.NDArray[np.int64],
        coarse_dis: npt.NDArray[np.float32],
        result: RangeSearchResult,
        store_pairs: bool = False,
        params: IVFSearchParameters | None = None,
        stats: IndexIVFStats | None = None,
    ) -> None: ...

class ParameterSpace:
    parameter_ranges: ParameterRangeVector
    verbose: int
    n_experiments: int
    batchsize: int
    thread_over_batches: bool
    min_test_duration: float

    def __init__(self) -> None: ...
    def add_range(self, name: str) -> ParameterRange: ...
    def n_combinations(self) -> int: ...
    def combination_ge(self, c1: int, c2: int) -> bool: ...
    def combination_name(self, cno: int) -> str: ...
    def display(self) -> None: ...
    def initialize(self, index: Index) -> None: ...
    def set_index_parameters(self, index: Index, cno: int) -> None: ...
    def set_index_parameters(self, index: Index, param_string: str) -> None: ...
    def set_index_parameter(self, index: Index, name: str, val: float) -> None: ...
    def update_bounds(
        self,
        cno: int,
        op: OperatingPoint,
        upper_bound_perf: float,
        lower_bound_t: float,
    ) -> tuple[float, float]: ...
    @overload
    def explore(
        self,
        index: Index,
        nq: int,
        xq: torch.Tensor,
        crit: AutoTuneCriterion,
        ops: OperatingPoints,
    ) -> None: ...
    @overload
    def explore(
        self,
        index: Index,
        nq: int,
        xq: npt.NDArray[np.float32],
        crit: AutoTuneCriterion,
        ops: OperatingPoints,
    ) -> None: ...

# GPU IVF base classes
class GpuIndex(Index):
    def getDevice(self) -> int: ...
    def getResources(self) -> StandardGpuResources: ...
    def setMinPagingSize(self, size: int) -> None: ...
    def getMinPagingSize(self) -> int: ...
    def copyFrom(self, index: Index) -> None: ...
    def copyTo(self, index: Index) -> None: ...

class GpuIndexIVF(GpuIndex, IndexIVFInterface): ...

class GpuIndexIVFConfig(GpuIndexConfig):
    """Configuration for GPU IVF indices"""
    def __init__(self) -> None: ...

class GpuIndexIVFPQConfig(GpuIndexIVFConfig):
    """Configuration for GPU IVFPQ index"""

    useFloat16LookupTables: bool
    usePrecomputedTables: bool
    interleavedLayout: bool
    useMMCodeDistance: bool

    def __init__(self) -> None: ...

class GpuIndexIVFPQ(GpuIndexIVF):
    """GPU IVFPQ index implementation"""

    pq: ProductQuantizer

    @overload
    def __init__(
        self,
        provider: StandardGpuResources,
        index: IndexIVFPQ,
        config: GpuIndexIVFPQConfig | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        provider: StandardGpuResources,
        dims: int,
        nlist: int,
        subQuantizers: int,
        bitsPerCode: int,
        metric: MetricType = METRIC_L2,
        config: GpuIndexIVFPQConfig | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        provider: StandardGpuResources,
        coarseQuantizer: Index,
        dims: int,
        nlist: int,
        subQuantizers: int,
        bitsPerCode: int,
        metric: MetricType = METRIC_L2,
        config: GpuIndexIVFPQConfig | None = None,
    ) -> None: ...
    def copyFrom(self, index: IndexIVFPQ) -> None: ...
    def copyTo(self, index: IndexIVFPQ) -> None: ...
    def reserveMemory(self, numVecs: int) -> None: ...
    def setPrecomputedCodes(self, enable: bool) -> None: ...
    def getPrecomputedCodes(self) -> bool: ...
    def getNumSubQuantizers(self) -> int: ...
    def getBitsPerCode(self) -> int: ...
    def getCentroidsPerSubQuantizer(self) -> int: ...
    def reclaimMemory(self) -> int: ...
    def reset(self) -> None: ...
    def updateQuantizer(self) -> None: ...
    @overload
    def train(self, x: torch.Tensor) -> None: ...
    @overload
    def train(self, x: npt.NDArray[np.float32]) -> None: ...

# GPU functions with comprehensive overloaded signatures for tensor support

class GpuParameterSpace(ParameterSpace):
    """GPU-specific parameter space for auto-tuning"""

    def initialize(self, index: Index) -> None: ...
    def set_index_parameter(self, index: Index, name: str, val: float) -> None: ...

class OperatingPoints:
    all_pts: OperatingPointVector
    optimal_pts: OperatingPointVector

    def __init__(self) -> None: ...
    def merge_with(self, other: OperatingPoints, prefix: str = "") -> int: ...
    def clear(self) -> None: ...
    def add(self, perf: float, t: float, key: str, cno: int = 0) -> bool: ...
    def t_for_perf(self, perf: float) -> float: ...
    def display(self, only_optimal: bool = True) -> None: ...
    def all_to_gnuplot(self, fname: str) -> None: ...
    def optimal_to_gnuplot(self, fname: str) -> None: ...

# Threading support
class IndexSplitVectors(Index):
    sub_indexes: list[Index]
    own_fields: bool
    threaded: bool

    def __init__(self, d: int, threaded: bool = True) -> None: ...

# Distance functions
def fvec_L2sqr(
    x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], d: int
) -> float: ...
def fvec_inner_product(
    x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], d: int
) -> float: ...
def fvec_L1(
    x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], d: int
) -> float: ...
def fvec_Linf(
    x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], d: int
) -> float: ...
def fvec_norm_L2sqr(x: npt.NDArray[np.float32], d: int) -> float: ...
def pairwise_L2sqr(
    d: int,
    nq: int,
    xq: npt.NDArray[np.float32],
    nb: int,
    xb: npt.NDArray[np.float32],
    dis: npt.NDArray[np.float32],
    ldq: int = -1,
    ldb: int = -1,
    ldd: int = -1,
) -> None: ...
def knn_inner_product(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    d: int,
    nx: int,
    ny: int,
    k: int,
    distances: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    sel: IDSelector | None = None,
) -> None: ...
def knn_L2sqr(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    d: int,
    nx: int,
    ny: int,
    k: int,
    distances: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    y_norm2: npt.NDArray[np.float32] | None = None,
    sel: IDSelector | None = None,
) -> None: ...
def range_search_L2sqr(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    d: int,
    nx: int,
    ny: int,
    radius: float,
    result: RangeSearchResult,
    sel: IDSelector | None = None,
) -> None: ...
def range_search_inner_product(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    d: int,
    nx: int,
    ny: int,
    radius: float,
    result: RangeSearchResult,
    sel: IDSelector | None = None,
) -> None: ...

# Index factory functions
def index_factory(
    d: int, description: str, metric: MetricType = METRIC_L2, own_invlists: bool = True
) -> Index: ...
def index_binary_factory(
    d: int, description: str, own_invlists: bool = True
) -> IndexBinary: ...

# Cloner classes
class Cloner:
    def clone_Index(self, index: Index) -> Index: ...

# GpuResourcesProvider base class
class GpuResourcesProvider:
    def __init__(self) -> None: ...

# I/O functions
def read_index(fname: str) -> Index: ...
def write_index(index: Index, fname: str) -> None: ...
def clone_index(index: Index) -> Index: ...

# Utility functions
def omp_set_num_threads(num_threads: int) -> None: ...
def omp_get_max_threads() -> int: ...

# Pointer conversion utilities
def swig_ptr(a: np.ndarray) -> Any: ...
def cast_integer_to_float_ptr(x: int) -> Any: ...
def cast_integer_to_idx_t_ptr(x: int) -> Any: ...
def cast_integer_to_int_ptr(x: int) -> Any: ...
def cast_integer_to_void_ptr(x: int) -> Any: ...

# Version utilities
def swig_version() -> int: ...

# Hash table implementation
class MapLong2Long:
    def __init__(self) -> None: ...
    def add(self, n: int, keys: np.ndarray, vals: np.ndarray) -> None: ...
    def search(self, key: int) -> int: ...
    def search_multiple(self, n: int, keys: np.ndarray, vals: np.ndarray) -> None: ...

# Additional utilities
def get_num_gpus() -> int: ...
def gpu_profiler_start() -> None: ...
def gpu_profiler_stop() -> None: ...
def gpu_sync_all_devices() -> None: ...

# Distance computation globals
distance_compute_blas_threshold: int
distance_compute_blas_query_bs: int
distance_compute_blas_database_bs: int
distance_compute_min_k_reservoir: int

# Index factory verbose flag
index_factory_verbose: int

# GPU-specific types and functions
class GpuDistanceParams:
    metric: MetricType
    k: int
    dims: int
    vectors: Any
    vectorsRowMajor: bool
    vectorType: int
    numVectors: int
    queries: Any
    queriesRowMajor: bool
    queryType: int
    numQueries: int
    outDistances: Any
    outIndices: Any
    outIndicesType: int
    device: int
    use_cuvs: bool

    def __init__(self) -> None: ...

class StandardGpuResourcesImpl:
    """Standard implementation of the GpuResources object that provides for a temporary memory manager"""

    def __init__(self) -> None: ...
    def supportsBFloat16(self, device: int) -> bool: ...
    def noTempMemory(self) -> None: ...
    def setTempMemory(self, size: int) -> None: ...
    def setPinnedMemory(self, size: int) -> None: ...
    def setDefaultStream(self, device: int, stream: Any) -> None: ...  # cudaStream_t
    def revertDefaultStream(self, device: int) -> None: ...
    def getDefaultStream(self, device: int) -> Any: ...  # cudaStream_t
    def setDefaultNullStreamAllDevices(self) -> None: ...
    def setLogMemoryAllocations(self, enable: bool) -> None: ...
    def initializeForDevice(self, device: int) -> None: ...
    def getBlasHandle(self, device: int) -> Any: ...  # cublasHandle_t
    def getAlternateStreams(self, device: int) -> list[Any]: ...  # vector<cudaStream_t>
    def allocMemory(self, req: Any) -> Any: ...  # AllocRequest -> void*
    def deallocMemory(self, device: int, ptr: Any) -> None: ...
    def getTempMemoryAvailable(self, device: int) -> int: ...
    def getMemoryInfo(self) -> dict[int, dict[str, tuple[int, int]]]: ...
    def getPinnedMemory(self) -> tuple[Any, int]: ...  # (void*, size_t)
    def getAsyncCopyStream(self, device: int) -> Any: ...  # cudaStream_t

class StandardGpuResources:
    """Default implementation of GpuResources that allocates a cuBLAS stream and 2 streams for use, as well as temporary memory."""

    def __init__(self) -> None: ...
    def getResources(self) -> Any: ...  # shared_ptr<GpuResources>
    def supportsBFloat16(self, device: int) -> bool: ...
    def supportsBFloat16CurrentDevice(self) -> bool: ...
    def noTempMemory(self) -> None: ...
    def setTempMemory(self, size: int) -> None: ...
    def setPinnedMemory(self, size: int) -> None: ...
    def setDefaultStream(self, device: int, stream: Any) -> None: ...  # cudaStream_t
    def revertDefaultStream(self, device: int) -> None: ...
    def setDefaultNullStreamAllDevices(self) -> None: ...
    def getMemoryInfo(self) -> dict[int, dict[str, tuple[int, int]]]: ...
    def getDefaultStream(self, device: int) -> Any: ...  # cudaStream_t
    def getTempMemoryAvailable(self, device: int) -> int: ...
    def syncDefaultStreamCurrentDevice(self) -> None: ...
    def setLogMemoryAllocations(self, enable: bool) -> None: ...

class GpuIndexBinaryFlatConfig(GpuIndexConfig):
    def __init__(self) -> None: ...

class GpuIndexBinaryFlat(IndexBinary):
    @overload
    def __init__(
        self,
        resources: StandardGpuResources,
        index: IndexBinaryFlat,
        config: GpuIndexBinaryFlatConfig = GpuIndexBinaryFlatConfig(),
    ) -> None: ...
    @overload
    def __init__(
        self,
        resources: StandardGpuResources,
        dims: int,
        config: GpuIndexBinaryFlatConfig = GpuIndexBinaryFlatConfig(),
    ) -> None: ...
    def getDevice(self) -> int: ...
    def getResources(self) -> StandardGpuResources: ...
    def copyFrom(self, index: IndexBinaryFlat) -> None: ...
    def copyTo(self, index: IndexBinaryFlat) -> None: ...
    @overload
    def add(self, x: torch.Tensor) -> None: ...
    @overload
    def add(self, x: npt.NDArray[np.uint8]) -> None: ...
    def reset(self) -> None: ...
    @overload
    def search(
        self,
        x: torch.Tensor,
        k: int,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def search(
        self,
        x: npt.NDArray[np.uint8],
        k: int,
        *,
        params: SearchParameters | None = None,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    @overload
    def reconstruct(self, key: int) -> torch.Tensor: ...
    @overload
    def reconstruct(self, key: int) -> npt.NDArray[np.uint8]: ...

class GpuResourcesVector:
    def __init__(self) -> None: ...
    def push_back(self, res: StandardGpuResources) -> None: ...
    def size(self) -> int: ...

# GPU Cloner Options (from gpu/GpuClonerOptions.h)
class GpuClonerOptions:
    indicesOptions: int  # IndicesOptions enum
    useFloat16CoarseQuantizer: bool
    useFloat16: bool
    usePrecomputed: bool
    reserveVecs: int
    storeTransposed: bool
    verbose: bool
    use_cuvs: bool
    allowCpuCoarseQuantizer: bool

    def __init__(self) -> None: ...

class GpuMultipleClonerOptions(GpuClonerOptions):
    shard: bool
    shard_type: int
    common_ivf_quantizer: bool

    def __init__(self) -> None: ...

# GPU Cloner classes (from gpu/GpuCloner.h)
class ToCPUCloner(Cloner):
    def merge_index(self, dst: Index, src: Index, successive_ids: bool) -> None: ...
    def clone_Index(self, index: Index) -> Index: ...

class ToGpuCloner(Cloner, GpuClonerOptions):
    provider: GpuResourcesProvider
    device: int

    def __init__(
        self,
        provider: GpuResourcesProvider,
        device: int,
        options: GpuClonerOptions,
    ) -> None: ...
    def clone_Index(self, index: Index) -> Index: ...

class ToGpuClonerMultiple(Cloner, GpuMultipleClonerOptions):
    sub_cloners: list[ToGpuCloner]

    @overload
    def __init__(
        self,
        providers: list[GpuResourcesProvider],
        devices: list[int],
        options: GpuMultipleClonerOptions,
    ) -> None: ...
    @overload
    def __init__(
        self,
        sub_cloners: list[ToGpuCloner],
        options: GpuMultipleClonerOptions,
    ) -> None: ...
    def copy_ivf_shard(
        self,
        index_ivf: IndexIVF,
        idx2: IndexIVF,
        n: int,
        i: int,
    ) -> None: ...
    def clone_Index_to_shards(self, index: Index) -> Index: ...
    def clone_Index(self, index: Index) -> Index: ...

class GpuProgressiveDimIndexFactory(ProgressiveDimIndexFactory):
    options: GpuMultipleClonerOptions
    vres: list[GpuResourcesProvider]
    devices: list[int]
    ncall: int

    def __init__(self, ngpu: int) -> None: ...
    def __call__(self, dim: int) -> Index: ...

# GPU Cloner functions (from gpu/GpuCloner.h)
def index_gpu_to_cpu(gpu_index: Index) -> Index: ...
def index_binary_gpu_to_cpu(gpu_index: IndexBinary) -> IndexBinary: ...
def index_binary_cpu_to_gpu(
    provider: GpuResourcesProvider,
    device: int,
    index: IndexBinary,
    options: GpuClonerOptions | None = None,
) -> IndexBinary: ...

# GPU distance data types
DistanceDataType_F32: int
DistanceDataType_F16: int

# GPU indices data types
IndicesDataType_I64: int
IndicesDataType_I32: int

# GPU functions
def bfKnn(res: StandardGpuResources, params: GpuDistanceParams) -> None: ...
def bfKnn_tiling(
    res: StandardGpuResources,
    params: GpuDistanceParams,
    vectorsMemoryLimit: int,
    queriesMemoryLimit: int,
) -> None: ...
def index_cpu_to_gpu(
    provider: StandardGpuResources,
    device: int,
    index: Index,
    options: GpuClonerOptions | None = None,
) -> GpuIndex: ...
def index_cpu_to_gpu_multiple(
    resources: GpuResourcesVector,
    devices: Int32Vector,
    index: Index,
    co: GpuMultipleClonerOptions | None = None,
) -> GpuIndex: ...
def index_binary_cpu_to_gpu_multiple(
    resources: GpuResourcesVector,
    devices: Int32Vector,
    index: IndexBinary,
    co: GpuMultipleClonerOptions | None = None,
) -> IndexBinary: ...
def index_cpu_to_gpu_multiple_py(
    resources: list[StandardGpuResources],
    index: Index,
    co: GpuMultipleClonerOptions | None = None,
    gpus: list[int] | None = None,
) -> GpuIndex: ...
def index_cpu_to_all_gpus(
    index: Index, co: GpuClonerOptions | None = None, ngpu: int = -1
) -> GpuIndex: ...
def index_cpu_to_gpus_list(
    index: Index,
    co: GpuClonerOptions | None = None,
    gpus: list[int] | None = None,
    ngpu: int = -1,
) -> GpuIndex: ...

class GpuIndexConfig:
    device: int
    memorySpace: Any  # MemorySpace enum
    use_cuvs: bool

    def __init__(self) -> None: ...

class GpuIndexFlatConfig(GpuIndexConfig):
    useFloat16: bool

    def __init__(self) -> None: ...

class GpuIndexFlat(Index):
    def __init__(
        self,
        provider: StandardGpuResources,
        d: int,
        metric: MetricType = METRIC_L2,
        config: GpuIndexFlatConfig | None = None,
    ) -> None: ...

class GpuIndexFlatL2(GpuIndexFlat):
    @overload
    def __init__(
        self,
        provider: StandardGpuResources,
        d: int,
        config: GpuIndexFlatConfig | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        provider: StandardGpuResources,
        index: IndexFlatL2,
        config: GpuIndexFlatConfig | None = None,
    ) -> None: ...

class GpuIndexFlatIP(GpuIndexFlat):
    @overload
    def __init__(
        self,
        provider: StandardGpuResources,
        d: int,
        config: GpuIndexFlatConfig | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        provider: StandardGpuResources,
        index: Any,  # IndexFlatIP*
        config: GpuIndexFlatConfig | None = None,
    ) -> None: ...

# GPU functions with comprehensive overloaded signatures for tensor support

# knn_gpu overloads: Precise return types based on input types
@overload
def knn_gpu(
    res: StandardGpuResources,
    xq: torch.Tensor,
    xb: torch.Tensor,
    k: int,
    D: torch.Tensor | None = None,
    I: torch.Tensor | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
    use_cuvs: bool = False,
    vectorsMemoryLimit: int = 0,
    queriesMemoryLimit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def knn_gpu(
    res: StandardGpuResources,
    xq: npt.NDArray[np.float32],
    xb: npt.NDArray[np.float32],
    k: int,
    D: npt.NDArray[np.float32] | None = None,
    I: npt.NDArray[np.int64] | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
    use_cuvs: bool = False,
    vectorsMemoryLimit: int = 0,
    queriesMemoryLimit: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
@overload
def knn_gpu(
    res: StandardGpuResources,
    xq: torch.Tensor,
    xb: npt.NDArray[np.float32],
    k: int,
    D: torch.Tensor | npt.NDArray[np.float32] | None = None,
    I: torch.Tensor | npt.NDArray[np.int64] | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
    use_cuvs: bool = False,
    vectorsMemoryLimit: int = 0,
    queriesMemoryLimit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def knn_gpu(
    res: StandardGpuResources,
    xq: npt.NDArray[np.float32],
    xb: torch.Tensor,
    k: int,
    D: torch.Tensor | npt.NDArray[np.float32] | None = None,
    I: torch.Tensor | npt.NDArray[np.int64] | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
    use_cuvs: bool = False,
    vectorsMemoryLimit: int = 0,
    queriesMemoryLimit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]: ...

# pairwise_distance_gpu overloads: Precise return types based on input types
@overload
def pairwise_distance_gpu(
    res: StandardGpuResources,
    xq: torch.Tensor,
    xb: torch.Tensor,
    D: torch.Tensor | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
) -> torch.Tensor: ...
@overload
def pairwise_distance_gpu(
    res: StandardGpuResources,
    xq: npt.NDArray[np.float32],
    xb: npt.NDArray[np.float32],
    D: npt.NDArray[np.float32] | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
) -> npt.NDArray[np.float32]: ...
@overload
def pairwise_distance_gpu(
    res: StandardGpuResources,
    xq: torch.Tensor,
    xb: npt.NDArray[np.float32],
    D: torch.Tensor | npt.NDArray[np.float32] | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
) -> torch.Tensor: ...
@overload
def pairwise_distance_gpu(
    res: StandardGpuResources,
    xq: npt.NDArray[np.float32],
    xb: torch.Tensor,
    D: torch.Tensor | npt.NDArray[np.float32] | None = None,
    metric: MetricType = METRIC_L2,
    device: int = -1,
) -> torch.Tensor: ...

# Additional utility functions with tensor support

# Utility functions from extra_wrappers.py with tensor overloads
@overload
def kmin(array: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def kmin(
    array: npt.NDArray[np.float32], k: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
@overload
def kmax(array: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def kmax(
    array: npt.NDArray[np.float32], k: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
@overload
def pairwise_distances(
    xq: torch.Tensor,
    xb: torch.Tensor,
    metric: MetricType = METRIC_L2,
    metric_arg: float = 0,
) -> torch.Tensor: ...
@overload
def pairwise_distances(
    xq: npt.NDArray[np.float32],
    xb: npt.NDArray[np.float32],
    metric: MetricType = METRIC_L2,
    metric_arg: float = 0,
) -> npt.NDArray[np.float32]: ...
def rand(
    n: int | tuple[int, ...] | list[int], seed: int = 12345
) -> npt.NDArray[np.float32]: ...
def randint(
    n: int, seed: int = 12345, vmax: int | None = None
) -> npt.NDArray[np.int64]: ...

# Alias for randint
lrand = randint

def randn(
    n: int | tuple[int, ...] | list[int], seed: int = 12345
) -> npt.NDArray[np.float32]: ...
@overload
def checksum(a: torch.Tensor) -> Any: ...
@overload
def checksum(a: npt.NDArray[Any]) -> Any: ...
def rand_smooth_vectors(
    n: int, d: int, seed: int = 1234
) -> npt.NDArray[np.float32]: ...
@overload
def eval_intersection(I1: torch.Tensor, I2: torch.Tensor) -> int: ...
@overload
def eval_intersection(I1: npt.NDArray[np.int64], I2: npt.NDArray[np.int64]) -> int: ...
@overload
def normalize_L2(x: torch.Tensor) -> None: ...
@overload
def normalize_L2(x: npt.NDArray[np.float32]) -> None: ...
@overload
def bucket_sort(
    tab: torch.Tensor, nbucket: int | None = None, nt: int = 0
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def bucket_sort(
    tab: npt.NDArray[np.int64], nbucket: int | None = None, nt: int = 0
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
@overload
def matrix_bucket_sort_inplace(
    tab: torch.Tensor, nbucket: int | None = None, nt: int = 0
) -> torch.Tensor: ...
@overload
def matrix_bucket_sort_inplace(
    tab: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    nbucket: int | None = None,
    nt: int = 0,
) -> npt.NDArray[np.int64]: ...
@overload
def knn(
    xq: torch.Tensor,
    xb: torch.Tensor,
    k: int,
    metric: MetricType = METRIC_L2,
    metric_arg: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def knn(
    xq: npt.NDArray[np.float32],
    xb: npt.NDArray[np.float32],
    k: int,
    metric: MetricType = METRIC_L2,
    metric_arg: float = 0.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
@overload
def knn_hamming(
    xq: torch.Tensor, xb: torch.Tensor, k: int, variant: str = "hc"
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def knn_hamming(
    xq: npt.NDArray[np.uint8], xb: npt.NDArray[np.uint8], k: int, variant: str = "hc"
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
@overload
def merge_knn_results(
    Dall: torch.Tensor, Iall: torch.Tensor, keep_max: bool = False
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def merge_knn_results(
    Dall: npt.NDArray[np.float32], Iall: npt.NDArray[np.int64], keep_max: bool = False
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

# Pack/unpack bitstring functions from extra_wrappers.py
@overload
def pack_bitstrings(a: npt.NDArray[np.int32], nbit: int) -> npt.NDArray[np.uint8]: ...
@overload
def pack_bitstrings(
    a: npt.NDArray[np.int32], nbit: npt.NDArray[np.int32]
) -> npt.NDArray[np.uint8]: ...
@overload
def unpack_bitstrings(
    b: npt.NDArray[np.uint8], M: int, nbit: int
) -> npt.NDArray[np.int32]: ...
@overload
def unpack_bitstrings(
    b: npt.NDArray[np.uint8], nbits: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]: ...

# Array conversion utilities with tensor support
@overload
def vector_to_array(v: Any) -> npt.NDArray[Any]: ...
@overload
def vector_float_to_array(v: Float32Vector) -> npt.NDArray[np.float32]: ...
@overload
def copy_array_to_vector(a: torch.Tensor, v: Any) -> None: ...
@overload
def copy_array_to_vector(a: npt.NDArray[Any], v: Any) -> None: ...
@overload
def copy_array_to_AlignedTable(a: torch.Tensor, v: Any) -> None: ...
@overload
def copy_array_to_AlignedTable(a: npt.NDArray[Any], v: Any) -> None: ...
@overload
def array_to_AlignedTable(a: torch.Tensor) -> Any: ...
@overload
def array_to_AlignedTable(a: npt.NDArray[Any]) -> Any: ...
@overload
def AlignedTable_to_array(v: Any) -> npt.NDArray[Any]: ...

# AlignedTable types
class AlignedTableUint8:
    def __init__(self, n: int) -> None: ...
    def size(self) -> int: ...
    def itemsize(self) -> int: ...
    def get(self) -> Any: ...
    def data(self) -> Any: ...
    def resize(self, n: int) -> None: ...

class AlignedTableUint16:
    def __init__(self, n: int) -> None: ...
    def size(self) -> int: ...
    def itemsize(self) -> int: ...
    def get(self) -> Any: ...
    def data(self) -> Any: ...
    def resize(self, n: int) -> None: ...

class AlignedTableFloat32:
    def __init__(self, n: int) -> None: ...
    def size(self) -> int: ...
    def itemsize(self) -> int: ...
    def get(self) -> Any: ...
    def data(self) -> Any: ...
    def resize(self, n: int) -> None: ...

# MaybeOwnedVector types
class MaybeOwnedVectorUInt8:
    is_owned: bool

    def __init__(self) -> None: ...
    def size(self) -> int: ...
    def data(self) -> Any: ...
    def resize(self, n: int) -> None: ...

class MaybeOwnedVectorInt32:
    is_owned: bool

    def __init__(self) -> None: ...
    def size(self) -> int: ...
    def data(self) -> Any: ...
    def resize(self, n: int) -> None: ...

class MaybeOwnedVectorFloat32:
    is_owned: bool

    def __init__(self) -> None: ...
    def size(self) -> int: ...
    def data(self) -> Any: ...
    def resize(self, n: int) -> None: ...

# Memory utilities
def memcpy(dest: Any, src: Any, n: int) -> Any: ...

# Heap implementations
class float_minheap_array_t:
    k: int
    nh: int
    val: Any
    ids: Any

    def __init__(self) -> None: ...
    def heapify(self) -> None: ...
    def addn(self, n: int, vals: Any) -> None: ...
    def addn_with_ids(self, n: int, vals: Any, ids: Any, id_stride: int) -> None: ...
    def addn_query_subset_with_ids(
        self, nsubset: int, subset: Any, n: int, vals: Any, ids: Any, id_stride: int
    ) -> None: ...
    def reorder(self) -> None: ...

class float_maxheap_array_t:
    k: int
    nh: int
    val: Any
    ids: Any

    def __init__(self) -> None: ...
    def heapify(self) -> None: ...
    def addn(self, n: int, vals: Any) -> None: ...
    def addn_with_ids(self, n: int, vals: Any, ids: Any, id_stride: int) -> None: ...
    def addn_query_subset_with_ids(
        self, nsubset: int, subset: Any, n: int, vals: Any, ids: Any, id_stride: int
    ) -> None: ...
    def reorder(self) -> None: ...

class int_minheap_array_t:
    k: int
    nh: int
    val: Any
    ids: Any

    def __init__(self) -> None: ...
    def heapify(self) -> None: ...
    def addn(self, n: int, vals: Any) -> None: ...
    def reorder(self) -> None: ...

class int_maxheap_array_t:
    k: int
    nh: int
    val: Any
    ids: Any

    def __init__(self) -> None: ...
    def heapify(self) -> None: ...
    def addn(self, n: int, vals: Any) -> None: ...
    def reorder(self) -> None: ...

# Hamming distance functions
def hammings_knn_hc(
    heap: int_maxheap_array_t,
    xq: npt.NDArray[np.uint8],
    xb: npt.NDArray[np.uint8],
    nb: int,
    ncodes: int,
    ordered: int,
) -> None: ...
def hammings_knn_mc(
    xq: npt.NDArray[np.uint8],
    xb: npt.NDArray[np.uint8],
    nq: int,
    nb: int,
    k: int,
    ncodes: int,
    distances: npt.NDArray[np.int32],
    labels: npt.NDArray[np.int64],
) -> None: ...

# Supported instruction sets utility
def supported_instruction_sets() -> set[str]: ...

# Additional merge functions for different types
def merge_knn_results_CMin(
    n: int,
    k: int,
    nshard: int,
    all_distances: Any,
    all_labels: Any,
    distances: Any,
    labels: Any,
) -> None: ...
def merge_knn_results_CMax(
    n: int,
    k: int,
    nshard: int,
    all_distances: Any,
    all_labels: Any,
    distances: Any,
    labels: Any,
) -> None: ...

# Efficient ID to ID map class from extra_wrappers.py
class MapInt64ToInt64:
    log2_capacity: int
    capacity: int
    tab: npt.NDArray[np.int64]

    def __init__(self, capacity: int) -> None: ...
    def add(self, keys: npt.NDArray[np.int64], vals: npt.NDArray[np.int64]) -> None: ...
    def lookup(self, keys: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]: ...

# Additional hash table functions
def hashtable_int64_to_int64_init(log2_capacity: int, tab: Any) -> None: ...
def hashtable_int64_to_int64_add(
    log2_capacity: int, tab: Any, n: int, keys: Any, vals: Any
) -> None: ...
def hashtable_int64_to_int64_lookup(
    log2_capacity: int, tab: Any, n: int, keys: Any, vals: Any
) -> None: ...

# ResultHeap utility class from extra_wrappers.py
class ResultHeap:
    I: npt.NDArray[np.int64]
    D: npt.NDArray[np.float32]
    nq: int
    k: int
    heaps: Any

    def __init__(self, nq: int, k: int, keep_max: bool = False) -> None: ...
    @overload
    def add_result(self, D: torch.Tensor, I: torch.Tensor) -> None: ...
    @overload
    def add_result(
        self, D: npt.NDArray[np.float32], I: npt.NDArray[np.int64]
    ) -> None: ...
    @overload
    def add_result_subset(
        self,
        subset: torch.Tensor,
        D: torch.Tensor,
        I: torch.Tensor,
    ) -> None: ...
    @overload
    def add_result_subset(
        self,
        subset: npt.NDArray[np.int64],
        D: npt.NDArray[np.float32],
        I: npt.NDArray[np.int64],
    ) -> None: ...
    def finalize(self) -> None: ...

# Kmeans utility class
class Kmeans:
    d: int
    k: int
    centroids: npt.NDArray[np.float32] | None
    obj: npt.NDArray[np.float32] | None
    iteration_stats: list[dict[str, Any]] | None
    cp: ClusteringParameters
    index: Index
    gpu: Any
    fac: Any

    def __init__(self, d: int, k: int, **kwargs: Any) -> None: ...
    def set_index(self) -> None: ...
    def reset(self, k: int | None = None) -> None: ...
    def train(
        self,
        x: npt.NDArray[np.float32],
        weights: npt.NDArray[np.float32] | None = None,
        init_centroids: npt.NDArray[np.float32] | None = None,
    ) -> float: ...
    def assign(
        self, x: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
