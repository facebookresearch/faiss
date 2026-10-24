# Randomized Hadamard rotation

`HadamardRotation` preprocesses vectors using a composition of randomized
Hadamard transforms (RHTs). Its default is **three successive RHTs**.
The `rounds` argument provides a **two-RHT alternative** for scalar
quantization, as well as a one-round option.

## What each round does

Let `p = d_out` be the smallest power of two at least as large as `d_in`,
and let `H` be the orthonormal `p`-dimensional Walsh-Hadamard matrix.
For each round `i`, `D_i` is a diagonal matrix of independently generated
random signs, each equally likely to be -1 or +1. Using column-vector notation:

```text
1 RHT:   R_1 x = H D_1 x
2 RHTs:  R_2 x = H D_2 H D_1 x
3 RHTs:  R_3 x = H D_3 H D_2 H D_1 x    (default)
```

Faiss zero-pads the input to `p` coordinates once, before the first round.
Each round multiplies by its sign vector and applies an unnormalized fast
Walsh-Hadamard transform (FWHT). A final factor `p^(-rounds/2)` supplies the
normalization in the formula above. This preserves Euclidean norms, inner
products, and L2 distances before quantization, up to floating-point error.
The transform does not normalize each input to unit length.

The sign vectors are generated once from `seed` and shared across every
vector processed by that object. Successive rounds use different sign
vectors: this is not repeatedly applying the same randomized matrix.
Use the same transform for training vectors, database vectors, and queries.
`IndexPreTransform` handles this automatically.

For `r` rounds the transform costs `O(r p log p)`. Selecting two rounds
instead of three removes one sign/FWHT pass. This is one third fewer such
passes, not a claim of a one-third reduction in end-to-end search time.

## Why two RHTs suffice for scalar quantization

[Ben-Basat et al. (2026)][paper], Section 3, prove that for every fixed
nonzero input and every fixed coordinate, `sqrt(p) * (R_2 x)_j / ||x||_2`
approaches a standard Gaussian with `O(p^(-1/2))` error in both Kolmogorov
and 1-Wasserstein distance, over the independent random signs. Thus **two
RHTs suffice for the scalar marginal guarantee** used by scalar quantizers.
Sections 4-5 derive asymptotically URR-matching guarantees for DRIVE and
QUIC-FL. One RHT can have non-Gaussian marginals on sparse inputs.

Scalar quantizers encode coordinates separately. Vector quantizers encode
blocks jointly, so their behavior also depends on the joint distribution.
Section 6 of [the same paper][paper] shows that two RHTs can retain block
dependencies, whereas three RHTs recover the Gaussian/URR expected-error
limit for a fixed finite codebook and fixed block dimension. This explains
the distinction between the two-round scalar option and the three-round
default; it is not a claim that Faiss originally chose its default based on
this paper.

These are asymptotic statements under the paper's assumptions. They do not
say that all Faiss scalar quantizers have identical finite-dimensional error
or ANN recall with two and three rounds. Data-dependent codebooks, scaling,
and search configuration still matter. See the
[RHT benchmark guide](../benchs/README_rht.md) for matched comparisons.

## Python usage

```python
import faiss
import numpy as np

d = 128
seed = 42
rng = np.random.RandomState(123)
xb = rng.randn(1000, d).astype("float32")
xq = rng.randn(10, d).astype("float32")

# Explicit transform: seed is the second argument, rounds is the third.
rotation = faiss.HadamardRotation(d, seed, 2)
rotated = rotation.apply(xb)
restored = rotation.reverse_transform(rotated)  # d is a power of two
np.testing.assert_allclose(restored, xb, rtol=1e-5, atol=1e-5)

# Apply the same two-RHT transform throughout a scalar-quantized index.
index = faiss.index_factory(d, "HR42r2,SQ8")
index.train(xb)
index.add(xb)
distances, labels = index.search(xq, 5)

# Existing constructor and factory forms still use three rounds.
default_rotation = faiss.HadamardRotation(d, seed)
default_index = faiss.index_factory(d, "HR42,SQ8")
```

Here `SQ8` illustrates composition with a Faiss scalar quantizer; the example
does not assert that its learned quantizer is one of the paper's analyzed
schemes. An explicit equivalent to the two-round factory construction is:

```python
rotation = faiss.HadamardRotation(d, seed, 2)
sq = faiss.IndexScalarQuantizer(rotation.d_out, faiss.ScalarQuantizer.QT_8bit)
index = faiss.IndexPreTransform(rotation, sq)
```

Use `rotation.d_out` for the underlying index dimension when padding is
needed. For example, input dimension 100 becomes 128. Padding can increase
the stored quantized code size. `reverse_transform` currently requires
`d_in == d_out`, so it rejects padded inputs.

## C++ usage

```cpp
#include <faiss/VectorTransform.h>

faiss::HadamardRotation two_rht(128, 42, 2);
faiss::HadamardRotation three_rht(128, 42); // default: three rounds
// For n contiguous input vectors (128 floats each) and an output buffer:
// two_rht.apply_noalloc(n, input, output);
```

## Factory syntax and compatibility

| Prefix | Seed | RHT rounds |
| --- | --- | --- |
| `HR` | 12345 | 3 |
| `HR42` | 42 | 3 |
| `HRr2` | 12345 | 2 |
| `HR42r2` | 42 | 2 |
| `HRr1` | 12345 | 1 |
| `HRr3` | 12345 | 3 |

The number immediately following `HR` is always the **seed**: `HR2` means
three rounds with seed 2, not two rounds. Append `r2` to select two rounds.
Round counts outside 1-3 are rejected. Choose the round count before
training or adding vectors and keep it fixed for the lifetime of the index.

The default preserves the original three-round sign stream and arithmetic.
For equal seeds, the one- and two-round variants use the corresponding
prefixes of the three-round sign sequence. Three-round serialization keeps
the existing `HRot` format, and old files load as three rounds. One- and
two-round transforms use the new `HRnd` format and require a reader that
supports it. The count and seed are preserved when serializing an index.

## Reference

Ran Ben-Basat, William Kuszmaul, Michael Mitzenmacher, Amit Portnoy, and
Shay Vargaftik. **Quantizing With Randomized Hadamard Transforms: Efficient
Heuristic Now Proven.** 2026. [arXiv:2605.06014][paper].

[paper]: https://arxiv.org/abs/2605.06014
