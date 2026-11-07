# Reproducing the e-means paper: Table 3, Figure 4(b) and Figure 3

The `paper_*` files in this folder re-measure the timed results of the e-means paper with the
e-means that ships in faiss (`faiss.contrib.e_means`), against the baseline the paper used for
each result, with the paper's timing protocol. You choose how far to go (1M, 10M or 100M rows),
download only the data that choice needs, and run one command per result.

| Paper result | What it shows | Compared with | Scales |
|---|---|---|---|
| Table 3 | epochs e-means needs to reach Flash-KMeans' error after 25 iterations, and how much faster it gets there | Flash-KMeans 0.3.1 | 1M, 10M, 100M rows |
| Figure 4(b) | test error against wall-clock time | Flash-KMeans 0.3.1 | 10M, 100M rows |
| Figure 3 | training and test error against epochs (no timing) | Faiss k-means on the CPU | 1M, 10M rows |

**Not covered:** Table 3's 1B row and its DINOv3 cell at 100M rows (several GPUs, or more memory
than one GPU holds), the Faiss-GPU and cuML curves of Figure 4(b), and the k-means++ Faiss, Sculley
and Staleness-Reduction curves of Figure 3.

## 1. What you need

- **One NVIDIA H100 (80 GB).** The CuTe kernels are tuned for it, and the timings are only
  comparable to the paper's on it. The 100M results put 20 to 51 GB of data on the card. Figure
  3's Faiss arm runs on CPU cores.
- **Software.** `faiss.contrib.e_means` is in the faiss repository but not yet in a faiss
  release, so install faiss from PyPI for its compiled core and add the library from a checkout of
  the repository:

  ```bash
  python -m venv emeans && source emeans/bin/activate
  pip install torch faiss-cpu
  pip install quack-kernels==0.6.5 nvidia-cutlass-dsl==4.7.1 "cuda-python>=13,<14"   # CuTe kernels
  pip install flash-kmeans==0.3.1 h5py matplotlib   # the baseline (exactly 0.3.1), ImageNet-CLIP, plots
  git clone https://github.com/facebookresearch/faiss.git
  cp -r faiss/contrib/e_means "$(python -c 'import faiss, os; print(os.path.dirname(faiss.__file__))')/contrib/"
  cd faiss/benchs/e_means
  ```

  These exact commands were run from scratch on an H100 (they installed PyTorch 2.15.0, Triton
  3.9.0 and faiss-cpu 1.15.1): every Table 3 seed tested crossed at the same budget as in the
  paper, with the same test errors, and the e-means test suite passed. The paper's numbers were
  measured with PyTorch 2.14 and Triton 3.8. Flash-KMeans is written in Triton, so its times can
  move by a few percent with the Triton version.

## 2. Choose your scale and download the data

Each scale includes the ones below it. Larger scales read a longer prefix of the large corpora,
so **the download grows about fivefold per step**:

| Choice | Download | Time at 25 MB/s (200 Mbit/s) | Time at 250 MB/s | Results you can reproduce |
|---|---|---|---|---|
| `--results 1M` | 4.5 GB | about 3 min | under a minute | Table 3, 1M row. Figure 3, DEEP* 1M and ImageNet-CLIP 1M |
| `--results 10M` | 21.7 GB | about 14 min | about 1 min | the above, plus Table 3, 10M row. Figure 4(b), upper row. Figure 3, SPACEV 10M and SIFT 10M |
| `--results 100M` | 100.8 GB | about 67 min | about 7 min | the above, plus Table 3, 100M row. Figure 4(b), lower row |

```bash
export EMEANS_DATA=/path/to/data           # where the files go
export EMEANS_CACHE=/path/to/cache         # small files written by the scripts
python paper_download.py --results 1M --dry-run   # prints the table above and the file list
python paper_download.py --results 1M             # asks for confirmation, then downloads
```

Measured from our cluster (16 parallel requests, 270 to 400 MB/s): `--results 1M` took 82 s, moving up to `10M` 58 s more, and up to `100M` 305 s more.

`paper_download.py` fetches only the bytes each result reads, with parallel HTTP range requests:
a prefix of each corpus, and, for SPACEV and SimSearchNet++, the 20,000 held-out test rows, saved
in a small `heldout_rows.npy` beside the corpus (they sit near the end of files of 140 and 256 GB,
which therefore need not be downloaded). An interrupted download resumes where it stopped, and
moving to a larger scale later only fetches what is missing. At the end it checks every dataset
against fingerprints of the paper's own arrays and stops on any difference, so a wrong or corrupted
file cannot produce plausible-looking numbers. Then build the Flash-KMeans copy the paper used on
two corpora, once:

```bash
python paper_flash_matched.py
```

## 3. Run

One GPU per command. Every timed run is written once to its own JSON file under `--out` (default
`./results`), named after everything that varies, so an interrupted command resumes and several
commands can run in parallel on different GPUs (pass `--dataset` to split the work).

```bash
# Table 3: one row at a time, then print it
python paper_bench.py table3 --scale 1M       # also --scale 10M, --scale 100M
python paper_bench.py table3 --print

# Figure 4(b): one row at a time, then draw it
python paper_bench.py figure4 --scale 10M     # also --scale 100M
python paper_bench.py figure4 --plot          # figure4.pdf

# Figure 3: e-means first (it draws the starting rows on the GPU), then Faiss on the CPU
python paper_bench.py figure3 --method emeans
python paper_bench.py figure3 --method faiss
python paper_bench.py figure3 --plot          # figure3.pdf

# any setting: a dataset, seeds, e-means budgets in epochs, Flash budgets in iterations
python paper_bench.py run --dataset deep1b-10M --seeds 10 11 --epochs 1 2 --flash-iters 25
```

## 4. Compare with the paper without re-running the baseline

`paper_targets.json` holds the paper's baseline measurements, per dataset and seed (10 to 19):
Flash-KMeans' test error and time after 25 iterations, Flash-KMeans' whole Figure 4(b) curve (1
to 100 iterations), and Faiss's error after 25 iterations for Figure 3. You can therefore time
e-means alone and compare it with them:

```bash
python paper_bench.py crossing --dataset fb-ssnpp-1M --seeds 10 11 12 --target paper
python paper_bench.py figure4 --plot          # with no local Flash runs, Flash is drawn from the file
```

Test errors transfer to any hardware. Times transfer only to an H100.

## 5. How the numbers are measured

- **One timed run** is a fresh process. It loads the data, takes the paper's starting centroids
  (the first k rows of a permutation drawn by a seeded CUDA generator, identical for both
  methods), makes one untimed identical warm-up call (two iterations for Flash-KMeans), then times
  one call between two GPU synchronizations, or the median of three when it took under 2 s.
  Loading, starting centroids and scoring are not timed. Both methods of a dataset and seed run in
  the same job, on the same GPU.
- **The test error** is the mean over the test rows of the squared distance to the nearest
  centroid: nearest by an IEEE float32 product (TF32 off), distance recomputed by direct difference
  in float64. The test rows are the second half of each dataset's frozen evaluation set.
- **e-means** runs the library's default recipe (the "throughput" batch at 1M rows, the "quality"
  batch above), bfloat16 assignment with its CuTe kernels. Float32 data goes through
  `CastQuantizer(d, torch.bfloat16)` (through `QuantizedEmeans`). SimSearchNet++ at 100M rows, whose float32 copy does not fit
  one GPU, is cast to bfloat16 once (exact for its uint8 values). `--input cast|bf16` forces either.
  A budget of T epochs is its own run, whose schedule anneals over T epochs.
- **Flash-KMeans** must be release 0.3.1 (any other is refused). It runs its standard function
  `batch_kmeans_Euclid` in float16 on data held on the GPU, every iteration forced. On SPACEV it uses
  a copy with squared norms in float32, and on SimSearchNet++, whose squared norms overflow
  float16, that copy's large-N function.
- **Table 3:** per seed, Flash-KMeans' error after 25 iterations is the target, and e-means runs 1.0,
  1.2, 1.4, ... epochs and the first budget at or below the target is the seed's crossing. The
  seed's speedup is Flash's time over e-means' time there. A cell is the upper median of the
  crossings and the median of the per-seed speedups.
- **Figure 4(b):** each point is the median over seeds of the run time, against the median over
  seeds of the error in percent above Flash's median error after 25 iterations (the zero line).
- **Figure 3:** training error is each method's error on its own last-pass assignment over every
  training row. Each curve is the mean over seeds, divided by Faiss after 25 iterations.

## 6. The files

| File | What it is |
|---|---|
| `paper_download.py` | Downloads the data for the scale you choose and checks it |
| `paper_bench.py` | The command line: `run`, `crossing`, `table3`, `figure4`, `figure3`, `targets`, `compare` |
| `paper_xb.py` | The timing protocol, how each method is called, and the test-error scorer |
| `paper_data.py` | The datasets and the paper's evaluation split, checked against the paper's arrays |
| `paper_flash_matched.py` | Builds the Flash-KMeans 0.3.1 copy with float32 squared norms |
| `paper_targets.json` | The paper's baseline measurements, per dataset and seed |
| `paper_splits/` | Held-out row numbers, and fingerprints of the paper's arrays |

## 7. Where the data comes from

| Dataset | Source |
|---|---|
| DEEP* | Deep1B (`base.1B.fbin`, `query.public.10K.fbin`), from the Hugging Face mirror `suchun/deep1b-mirror` |
| SIFT | BigANN, `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/` |
| SPACEV 1M, 10M | SPACEV1B (Microsoft), the original parts `vectors.bin/vectors_N.bin`, from the Hugging Face mirror `jkhe/spacev1b` |
| SPACEV 100M | a 100M sample of SPACEV (not a prefix of the 1B file), Hugging Face `unum-cloud/ann-spacev-100m` |
| SimSearchNet++ | `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_database.u8bin` |
| DINOv3 | `https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/` (first chunk, and `queries_clean.bvecs`) |
| ImageNet-CLIP | the VIBE benchmark (Jääsaari et al., 2025), Hugging Face `vector-index-bench/vibe` |

Each mirror was checked to hold the same bytes as the copies the paper used.

## 8. Our run

Run on H100s with `faiss.contrib.e_means`, PyTorch 2.14, Triton 3.8, CUTLASS DSL 4.7.1 and Flash-KMeans 0.3.1 in the same environment, seeds 10 to 19, every rule above. The campaign ran on the code before it was merged into faiss. Re-running Table 3 at 1M and 10M and Figure 3 on faiss `main` gave the same crossing on every seed and the same test errors.

These timings include a fix to the CuTe argmin decode cache, which is being made separately. Without it, a budget with a partial epoch (1.2, 1.4, ... epochs) recompiles a kernel on every call when the vectors are wide, which adds about 0.3 s per call on SimSearchNet++ and DINOv3 at 1M rows, so their Table 3 speedups come out far lower. Test errors and crossings are unaffected.

**Table 3.** Epochs, then speedup over Flash-KMeans at 25 iterations (10 seeds per cell). "Research code" is the paper's earlier re-timing of the authors' implementation, with Flash under PyTorch 2.8 and Triton 3.4.

| Dataset | `--input cast` (default) | `--input bf16` | Research code |
|---|---|---|---|
| DEEP* 1M | 2.0, 12.8x | 2.0, 14.3x | 2.0, 10.6x |
| DINOv3/4096 1M | 2.0, 12.7x | 2.0, 14.3x | 2.0, 18.2x |
| SPACEV 1M | 2.4, 9.5x | 2.4, 10.5x | 2.4, 11.4x |
| SimSearchNet++ 1M | 1.4, 12.9x | 1.4, 14.3x | 1.4, 12.8x |
| DEEP* 10M | 2.0, 27.8x | 2.0, 28.8x | 2.0, 26.0x |
| DINOv3/4096 10M | 1.4, 23.3x | 1.4, 24.0x | 1.6, 26.3x |
| SPACEV 10M | 2.0, 19.2x | 2.0, 20.6x | 2.2, 24.0x |
| SimSearchNet++ 10M | 1.6, 23.8x | 1.6, 24.9x | 1.6, 18.3x |
| DEEP* 100M | 2.0, 27.0x | 2.0, 27.0x | 2.0, 30.0x |
| SPACEV 100M | 1.4, 35.9x | 1.4, 36.1x | 1.4, 37.8x |
| SimSearchNet++ 100M | -- | 2.6, 19.9x | 2.6, 11.6x |

The two input routes cross at the same budget on every seed and give bit-identical test errors on every run. Pre-casting is faster by the cost of the per-batch conversion, about 10% at k=16,384, 4% at k=65,536 and 0.6% at k=262,144. SimSearchNet++ at 100M runs pre-cast only (its float32 copy does not fit one GPU).

**Figure 4(b).** First e-means budget at or below Flash@25 (speedup), and e-means after 25 epochs relative to Flash@25, medians over seeds 10 to 19.

| Panel | First budget at or below Flash@25 | 25 epochs |
|---|---|---|
| DEEP* 10M | 2 epochs (31.0x) | -0.64% |
| SPACEV 10M | 2 epochs (19.9x) | -1.04% |
| SimSearchNet++ 10M | 2 epochs (21.5x) | -0.60% |
| SIFT/512 10M | 2 epochs (29.4x) | -0.52% |
| DEEP* 100M | 3 epochs (17.1x) | -0.54% |
| SPACEV 100M | 2 epochs (25.5x) | -4.20% |
| SimSearchNet++ 100M | 3 epochs (17.3x) | -0.39% |
| SIFT/512 100M | 2 epochs (24.3x) | -0.43% |

**Figure 3.** e-means after 25 epochs, test error relative to Faiss after 25 iterations (mean over seeds 10 to 19): DEEP* 1M 0.9909, ImageNet-CLIP 1M 0.9889, SPACEV 10M 0.9898, SIFT 10M 0.9951.

Every Flash-KMeans target behind these numbers is in `paper_targets.json`.

**Cost on one H100** (wall time of our jobs, fresh processes and loading included, both input
routes): Table 3 took 8 to 29 minutes per 1M dataset and 35 to 49 minutes per 10M dataset (10
seeds each), and about 70 minutes per 100M dataset and seed including that seed's Figure 4
e-means curve. A Figure 4 seed at 10M rows took about 13 minutes. At 100M, Flash's curve is the
long part: 100 iterations alone took 30 minutes (DEEP*) to 57 minutes (SimSearchNet++). Figure 3's
e-means arm took 4 to 46 minutes per dataset (10 seeds), and its Faiss arm about 2 minutes per
seed at 1M rows and 50 at 10M on 32 CPU cores.
