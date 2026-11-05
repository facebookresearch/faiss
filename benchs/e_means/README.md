# Sharded e-means on DINO

These demos show sharded e-means training on the public DINO10B dataset in 
a scalable way. 

Most of the training code itself in the e_means contrib directory. The code here 
handles the data format and I/O to disk.

The code is primarlily intended to run on a single multi-GPU machine.
Every worker holds one contiguous centroid shard and an equal row contribution
per global batch. `k`, `n`, and the resolved global batch size must divide by
the number of workers. 

## Download DINO10B

The public corpus is at
https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/.
All training demos need the first base chunk and the held-out test queries:

- `chunked_base_10B/chunk_0000.bvecs`: 200,000,000 vectors,
  205,600,000,000 bytes (about 191.5 GiB).
- `queries_clean.bvecs`: 100,000 held-out queries,
  102,800,000 bytes (about 98 MiB).

Both files contain 1,024 uint8 coordinates per vector, preceded by a four-byte
little-endian dimension header. Use an SSD-backed location with enough free
space. The commands below use `/tmp/DINO10B`; replace that root throughout
and pass `--data PATH` to the demos if you use another location.

Download the public instructions and both data files:

```bash
mkdir -p /tmp/DINO10B/chunked_base_10B
# minimal test
wget -c -O /tmp/DINO10B/README.md \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/README.md
# queries are used as a held-out test set
wget -c -O /tmp/DINO10B/queries_clean.bvecs \
    https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/queries_clean.bvecs
for i in {0000..0049}; do
  wget -c -O /tmp/DINO10B/chunked_base_10B/chunk_$i.bvecs \
      https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/chunked_base_10B/chunk_$i.bvecs
done
```

The complete corpus takes about 10.28 TB but if you can download only a subset
for smaller scale tests, eg. chunk_0000.bvecs, ..., chunk_0004.bvecs is enough
for tests at 1B scale. These demos do not need `train_queries_99M.bvecs` or nearest-neighbor
ground truth.

### Held-out MSE

Both training demos evaluate every test query after each epoch and again
after finalization, including any final closing-mean update. `test_mse` is
the average squared L2 distance to the nearest centroid.

The shipped 100,000 queries are permuted with Numpy
`RandomState(1234)`; the first half is reserved for validation and the second
half is evaluated, matching the reference's test set. 


## Reproducing the DINO10B e-means+IVF results 

The results of the scaling plot of Figure 1 are from tables 20 and 21 (for n=10B vectors and n=1B vectors 
respectively. 
They can be reproduced with the code below.

### 1B scale repro 

This needs 4 GPUs (H100 GPUs for timing). The following command reproduces the setting with 
3 epochs, IVF1024 and L_IVF=4: 

```bash 
python demo_training_from_ssd.py \
     --launch 4 --data /tmp/DINO10B   \
     --n 1000_000_000 --k $((1<<21)) \
     --epochs 3  --nlist 1024 --nprobe 4 \
     --seed 1  --final-closing-mean \
     --batch-size $((1<<21)) --ssd-ram-gib 100
```
The options are: 

- `--launch 4` means running on 4 GPUs. The command can also be launched with torchrun or slurm.

- `--final-closing-mean` does a final pass to compute the final centroids.
The alternative is to accumulate them during the last epoch, but this uses more GPU memory.

- `--batch-size` overrides the batch size. In general, use the largest possible that fits on the GPU

- `--ssd-ram-gib` sets the batch prefetch cache size. Higer can be faster when the ssd I/O is slow.

The output should look like: [this](https://gist.github.com/mdouze/020709077a6ef8decfdab07119049301).
It runs in 948.52s and obtains a test MSE 147074.75, which is close to the result in table 21 (MSE=147120 in 1.0h = 900s/machine).

### 10B scale repro

Similarly, the 10B command is: 
```bash
python demo_training_from_ssd.py \
    --launch 8 --n 10_000_000_000 --k $((1<<24)) \
    --epochs 3 --ivf --nlist 4096 --nprobe 4 \
    --batch-size $((1<<21)) --ssd-ram-gib 120  \
    --final-closing-mean
```

The ouput looks like [this gist](https://gist.github.com/mdouze/84949f331c780a60015b30571ab2f52d). 
The run is 9762.65s and test MSE = 129865.47, which is close to the result reported in tab 20: MSE=129830 and 23.2h = 10440s / machine. 
