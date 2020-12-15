set -e

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

# This script launches the experiments on a cluster
# It assumes two shell functions are defined:
#
#    run_on_1machine: runs a command on one (full) machine on a cluster
#
#    run_on_8gpu: runs a command on one machine with 8 GPUs
#
# the two functions are called as:
#
#    run_on_1machine <name> <command>
#
# the stdout of the command should be stored in $logdir/<name>.stdout


function run_on ()
{
    sys="$1"
    shift
    name="$1"
    shift
    script="$logdir/$name.sh"

    if [ -e "$script" ]; then
        echo script "$script" exists
        return
    fi

    # srun handles special characters fine, but the shell interpreter
    # does not
    escaped_cmd=$( printf "%q " "$@" )

    cat > $script <<EOF
#! /bin/bash
srun $escaped_cmd
EOF

    echo -n "$logdir/$name.stdout "
    sbatch -n1 -J "$name" \
           $sys \
            --comment='priority is the only one that works'  \
           --output="$logdir/$name.stdout" \
           "$script"

}


function run_on_1machine {
    run_on "--cpus-per-task=80 --gres=gpu:0 --mem=500G --time=70:00:00 --partition=priority" "$@"
}

function run_on_1machine_1h {
    run_on "--cpus-per-task=80 --gres=gpu:2 --mem=100G --time=1:00:00 --partition=priority" "$@"
}

function run_on_1machine_3h {
    run_on "--cpus-per-task=80 --gres=gpu:2 --mem=100G --time=3:00:00 --partition=priority" "$@"
}

function run_on_4gpu_3h {
    run_on "--cpus-per-task=40 --gres=gpu:4 --mem=100G --time=3:00:00 --partition=priority" "$@"
}

function run_on_8gpu () {
    run_on "--cpus-per-task=80 --gres=gpu:8 --mem=100G --time=70:00:00 --partition=priority" "$@"
}


# prepare output directories
# set to some directory where all indexes, can be written.
basedir=/checkpoint/matthijs/bench_all_ivf

logdir=$basedir/logs
indexdir=$basedir/indexes
centdir=$basedir/precomputed_clusters

mkdir -p $logdir $indexdir


# adds an option to use a pretrained quantizer
function add_precomputed_quantizer () {
    local db="$1"
    local coarse="$2"

    case $db in
        bigann*) rname=bigann ;;
        deep*)   rname=deep ;;
        sift1M) return;;
        music-100) return ;;
        glove) return ;;
        *) echo "bad db"; exit 1;;
    esac

    case $coarse in
        IVF65536*)
            cname=clustering.db${rname}1M.IVF65536.faissindex
            copt="--get_centroids_from $centdir/$cname"
            ;;
        IVF262144*)
            cname=clustering.db${rname}1M.IVF262144.faissindex
            copt="--get_centroids_from $centdir/$cname"
            ;;
        IVF1048576*)
            cname=clustering.db${rname}1M.IVF1048576.faissindex
            copt="--get_centroids_from $centdir/$cname"
            ;;
        IVF4194304*)
            cname=clustering.db${rname}1M.IVF4194304.faissindex
            copt="--get_centroids_from $centdir/$cname"
            ;;
        *)
        copt="" ;;
    esac

    echo $copt
}

function get_db_dim () {
    local db="$1"
    case $db in
        sift1M) dim=128;;
        bigann*) dim=128;;
        deep*) dim=96;;
        music-100) dim=100;;
        glove) dim=100;;
        *) echo "bad db"; exit 1;;
    esac
    echo $dim
}


# replace HD = half dim with the half of the dimension we need to handle
# relying that variables are global by default...
function replace_coarse_PQHD () {
    local coarse="$1"
    local dim=$2


    coarseD=${coarse//PQHD/PQ$((dim/2))}
    coarse16=${coarse//PQHD/PQ8}
    coarse32=${coarse//PQHD/PQ16}
    coarse64=${coarse//PQHD/PQ32}
    coarse128=${coarse//PQHD/PQ64}
    coarse256=${coarse//PQHD/PQ128}
    coarse112=${coarse//PQHD/PQ56}

}



if false; then



###############################################
# comparison with SCANN

for db in sift1M deep1M glove music-100
do
    opt=""
    if [ $db == glove ]; then
        opt="--measure inter"
    fi

    run_on_1machine_1h cmp_with_scann.$db.c \
        python -u cmp_with_scann.py --db $db \
        --lib faiss $opt --thenscann

done




############################### Preliminary SIFT1M experiment


for db in sift1M  ; do

    for coarse in  IVF1024
    do
        indexkeys="
            HNSW32
            $coarse,SQfp16
            $coarse,SQ4
            $coarse,SQ8
            $coarse,PQ32x8
            $coarse,PQ64x4
            $coarse,PQ64x4fs
            $coarse,PQ64x4fs,RFlat
            $coarse,PQ64x4fs,Refine(SQfp16)
            $coarse,PQ64x4fs,Refine(SQ8)
            OPQ64,$coarse,PQ64x4fs
            OPQ64,$coarse,PQ64x4fs,RFlat
        "
        indexkeys="
            $coarse,PQ64x4fsr
            $coarse,PQ64x4fsr,RFlat
        "

        # OPQ actually degrades the results on SIFT1M, so let's ignore

        for indexkey in $indexkeys
        do
            # escape nasty characters
            key="autotune.db$db.${indexkey//,/_}"
            key="${key//(/_}"
            key="${key//)/_}"
            run_on_1machine_1h $key.a \
                 python -u bench_all_ivf.py \
                    --db $db \
                    --indexkey "$indexkey" \
                    --maxtrain 0  \
                    --indexfile $indexdir/$key.faissindex \
                    --searchthreads 32
        done
    done
done




############################### 1M experiments

fi
# for db in sift1M deep1M music-100 glove; do

for db in glove music-100; do

    dim=$( get_db_dim $db )

    for coarse in IVF1024 IVF4096_HNSW32
    do

        replace_coarse_PQHD "$coarse" $dim

        indexkeys="
            $coarseD,PQ$((dim/2))x4fs
            $coarseD,PQ$((dim/2))x4fsr

            OPQ8_64,$coarse64,PQ8
            PCAR16,$coarse16,SQ4
            OPQ16_64,$coarse64,PQ16x4fs
            OPQ16_64,$coarse64,PQ16x4fsr

            OPQ16_64,$coarse64,PQ16
            PCAR16,$coarse16,SQ8
            PCAR32,$coarse32,SQ4
            OPQ32_64,$coarse64,PQ32x4fs
            OPQ32_64,$coarse64,PQ32x4fsr

            OPQ32_128,$coarse128,PQ32
            PCAR32,$coarse32,SQ8
            PCAR64,$coarse64,SQ4
            PCAR16,$coarse16,SQfp16
            OPQ64_128,$coarse128,PQ64x4fs
            OPQ64_128,$coarse128,PQ64x4fsr

            OPQ64_128,$coarse128,PQ64
            PCAR64,$coarse64,SQ8
            PCAR32,$coarse32,SQfp16
            PCAR128,$coarse128,SQ4
            OPQ128_256,$coarse256,PQ128x4fs
            OPQ128_256,$coarse256,PQ128x4fsr
            OPQ16_64,$coarse64,PQ16x4fs,Refine(OPQ56_112,PQ56)
            OPQ16_64,$coarse64,PQ16x4fs,Refine(PCAR72,SQ6)
            OPQ32_64,$coarse64,PQ16x4fs,Refine(PCAR64,SQ6)
            OPQ32_64,$coarse64,PQ32x4fs,Refine(OPQ48_96,PQ48)
            OPQ64_128,$coarse,PQ64x12

            OPQ64_128,$coarse,PQ64x4fs,RFlat
            OPQ64_128,$coarse,PQ64x4fs,Refine(SQfp16)
            OPQ64_128,$coarse,PQ64x4fs,Refine(SQ8)
            OPQ64_128,$coarse,PQ64x4fs,Refine(SQ6)
            OPQ64_128,$coarse,PQ64x4fs,Refine(SQ4)
            OPQ32_64,$coarse,PQ32x4fs,Refine(SQfp16)
            OPQ32_64,$coarse,PQ32x4fs,Refine(SQ8)
            OPQ32_64,$coarse,PQ32x4fs,Refine(SQ6)
            OPQ32_64,$coarse,PQ32x4fs,Refine(SQ4)

        "

        indexkeys="
            $coarseD,PQ$((dim/2))x4fs
            $coarseD,PQ$((dim/2))x4fsr
            $coarseD,PQ$((dim/2))x4fsr,RFlat
            $coarseD,PQ$((dim/2))x4fsr,Refine(SQfp16)
            $coarseD,PQ$((dim/2))x4fsr,Refine(SQ8)
            $coarseD,PQ$((dim/4))x4fs
            $coarseD,PQ$((dim/4))x4fsr
            $coarseD,PQ$((dim/4))x4fsr,RFlat
            $coarseD,PQ$((dim/4))x4fsr,Refine(SQfp16)
            $coarseD,PQ$((dim/4))x4fsr,Refine(SQ8)
            $coarseD,PQ$((dim/2))
            $coarseD,PQ$((dim/4))
            HNSW32,Flat
        "

        indexkeys="HNSW32,Flat"

        for indexkey in $indexkeys
        do
            key=autotune.db$db.${indexkey//,/_}
            key="${key//(/_}"
            key="${key//)/_}"
            run_on_1machine_3h $key.q \
              python -u bench_all_ivf.py \
                    --db $db \
                    --indexkey "$indexkey" \
                    --maxtrain 0  \
                    --indexfile "$indexdir/$key.faissindex" \
                    $( add_precomputed_quantizer $db $coarse ) \
                    --searchthreads 32 \
                    --min_test_duration 3
        done


    done
done

if false; then

############################################
# precompute centroids on GPU for large vocabularies

for db in deep1M bigann1M; do

    for ncent in 262144 65536 1048576 4194304; do

        key=clustering.db$db.IVF$ncent
        run_on_4gpu_3h $key.e \
            python -u bench_all_ivf.py \
                --db $db \
                --indexkey IVF$ncent,SQ8 \
                --maxtrain 100000000  \
                --indexfile $centdir/$key.faissindex \
                --searchthreads 32 \
                --min_test_duration 3 \
                --add_bs 1000000 \
                --train_on_gpu

    done
done

###############################
## coarse quantizer experiments on the centroids of deep1B


for k in 4 8 16 64 256; do

    for ncent in 65536 262144 1048576 4194304; do
        db=deep_centroids_$ncent

        # compute square root of ncent...
        for(( ls=0; ncent > (1 << (2 * ls)); ls++)); do
            echo -n
        done
        sncent=$(( 1 << ls ))

        indexkeys="
            IVF$((sncent/2)),PQ48x4fs,RFlat
            IVF$((sncent*2)),PQ48x4fs,RFlat
            HNSW32
            PQ48x4fs
            PQ48x4fs,RFlat
            IVF$sncent,PQ48x4fs,RFlat
        "

        for indexkey in $indexkeys; do
            key="cent_index.db$db.k$k.$indexkey"
            run_on_1machine_1h "$key.b" \
                    python -u bench_all_ivf.py \
                    --db $db \
                    --indexkey "$indexkey" \
                    --maxtrain 0  \
                    --inter \
                    --searchthreads 32 \
                    --k $k
        done

    done
done


############################### 10M experiments


for db in deep10M bigann10M; do

    coarses="
        IVF65536(IVF256,PQHDx4fs,RFlat)
        IVF16384_HNSW32
        IVF65536_HNSW32
        IVF262144_HNSW32
        IVF262144(IVF512,PQHDx4fs,RFlat)
    "

    dim=$( get_db_dim $db )

    for coarse in $coarses
    do

        replace_coarse_PQHD "$coarse" $dim

        indexkeys="
            $coarseD,PQ$((dim/2))x4fs

            OPQ8_64,$coarse64,PQ8
            PCAR16,$coarse16,SQ4
            OPQ16_64,$coarse64,PQ16x4fs
            OPQ16_64,$coarse64,PQ16x4fsr

            OPQ16_64,$coarse64,PQ16
            PCAR16,$coarse16,SQ8
            PCAR32,$coarse32,SQ4
            OPQ32_64,$coarse64,PQ32x4fs
            OPQ32_64,$coarse64,PQ32x4fsr

            OPQ32_128,$coarse128,PQ32
            PCAR32,$coarse32,SQ8
            PCAR64,$coarse64,SQ4
            PCAR16,$coarse16,SQfp16
            OPQ64_128,$coarse128,PQ64x4fs
            OPQ64_128,$coarse128,PQ64x4fsr

            OPQ64_128,$coarse128,PQ64
            PCAR64,$coarse64,SQ8
            PCAR32,$coarse32,SQfp16
            PCAR128,$coarse128,SQ4
            OPQ128_256,$coarse256,PQ128x4fs
            OPQ128_256,$coarse256,PQ128x4fsr
            OPQ56_112,$coarse112,PQ7+56
            OPQ16_64,$coarse64,PQ16x4fs,Refine(OPQ56_112,PQ56)
            OPQ16_64,$coarse64,PQ16x4fs,Refine(PCAR72,SQ6)
            OPQ32_64,$coarse64,PQ16x4fs,Refine(PCAR64,SQ6)
            OPQ32_64,$coarse64,PQ32x4fs,Refine(OPQ48_96,PQ48)
        "

        indexkeys="
            OPQ16_64,$coarse64,PQ16x4fsr
            OPQ32_64,$coarse64,PQ32x4fsr
            OPQ64_128,$coarse128,PQ64x4fsr
            OPQ128_256,$coarse256,PQ128x4fsr
        "


        for indexkey in $indexkeys
        do
            key=autotune.db$db.${indexkey//,/_}
            key="${key//(/_}"
            key="${key//)/_}"
            run_on_1machine_3h $key.l \
              python -u bench_all_ivf.py \
                    --db $db \
                    --indexkey "$indexkey" \
                    --maxtrain 0  \
                    --indexfile "$indexdir/$key.faissindex" \
                    $( add_precomputed_quantizer $db $coarse ) \
                    --searchthreads 32 \
                    --min_test_duration 3 \
                    --autotune_max nprobe:2000
        done
    done
done


############################### 100M experiments

for db in deep100M bigann100M; do
    coarses="
        IVF65536_HNSW32
        IVF262144_HNSW32
        IVF262144(IVF512,PQHDx4fs,RFlat)
        IVF1048576_HNSW32
        IVF1048576(IVF1024,PQHDx4fs,RFlat)
    "
    dim=$( get_db_dim $db )

    for coarse in $coarses
    do
        replace_coarse_PQHD "$coarse" $dim

        indexkeys="
            OPQ8_64,$coarse64,PQ8
            OPQ16_64,$coarse64,PQ16x4fs

            PCAR32,$coarse32,SQ4
            OPQ16_64,$coarse64,PQ16
            OPQ32_64,$coarse64,PQ32x4fs

            OPQ32_128,$coarse128,PQ32
            PCAR64,$coarse64,SQ4
            PCAR32,$coarse32,SQ8
            OPQ64_128,$coarse128,PQ64x4fs

            PCAR128,$coarse128,SQ4
            OPQ64_128,$coarse128,PQ64

            PCAR32,$coarse32,SQfp16
            PCAR64,$coarse64,SQ8
            OPQ128_256,$coarse256,PQ128x4fs

            OPQ56_112,$coarse112,PQ7+56
            OPQ16_64,$coarse64,PQ16x4fs,Refine(OPQ56_112,PQ56)

            $coarseD,PQ$((dim/2))x4fs
        "

        indexkeys="
            OPQ128_256,$coarse256,PQ128x4fsr
            OPQ64_128,$coarse128,PQ64x4fsr
            OPQ32_64,$coarse64,PQ32x4fsr
            OPQ16_64,$coarse64,PQ16x4fsr
            OPQ16_64,$coarse64,PQ16x4fsr,Refine(OPQ56_112,PQ56)
        "

        for indexkey in $indexkeys
        do
            key=autotune.db$db.${indexkey//,/_}
            key="${key//(/_}"
            key="${key//)/_}"
            run_on_1machine $key.e \
                 python -u bench_all_ivf.py \
                    --db $db \
                    --indexkey "$indexkey" \
                    --maxtrain 0  \
                    --indexfile $indexdir/$key.faissindex \
                    --searchthreads 32 \
                    --min_test_duration 3 \
                    $( add_precomputed_quantizer $db $coarse ) \
                    --add_bs 1000000 \
                    --autotune_max nprobe:2000

        done
    done
done


#################################
# 1B-scale experiment



for db in deep1B bigann1B; do
    coarses="
        IVF1048576_HNSW32
        IVF4194304_HNSW32
        IVF4194304(IVF1024,PQHDx4fs,RFlat)
    "
    dim=$( get_db_dim $db )

    for coarse in $coarses; do

        replace_coarse_PQHD "$coarse" $dim


        indexkeys="
            OPQ8_64,$coarse64,PQ8
            OPQ16_64,$coarse64,PQ16x4fsr

            OPQ16_64,$coarse64,PQ16
            OPQ32_64,$coarse64,PQ32x4fsr

            OPQ32_128,$coarse128,PQ32
            OPQ64_128,$coarse128,PQ64x4fsr

            OPQ64_128,$coarse128,PQ64
            OPQ128_256,$coarse256,PQ128x4fsr
            OPQ56_112,$coarse112,PQ7+56
            OPQ16_64,$coarse64,PQ16x4fs,Refine(OPQ56_112,PQ56)

            $coarseD,PQ$((dim/2))x4fs
        "

        for indexkey in $indexkeys
        do
            key=autotune.db$db.${indexkey//,/_}
            key="${key//(/_}"
            key="${key//)/_}"
            run_on_1machine $key.d \
                 python -u bench_all_ivf.py \
                    --db $db \
                    --indexkey "$indexkey" \
                    --maxtrain 0  \
                    --indexfile $indexdir/$key.faissindex \
                    --searchthreads 32 \
                    --min_test_duration 3 \
                    $( add_precomputed_quantizer $db $coarse ) \
                    --add_bs 1000000 \
                    --autotune_max nprobe:3000
        done
    done

done

fi