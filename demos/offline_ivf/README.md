
# Offline IVF

This folder contains the code for the offline ivf algorithm powered by faiss big batch search.

Create a conda env:

`conda create --name oivf python=3.10`

`conda activate oivf`

`conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.7.4`

`conda install tqdm`

`conda install pyyaml`

`conda install -c conda-forge submitit`


## Run book

1. Optionally shard your dataset (see create_sharded_dataset.py) and create the corresponding yaml file `config_ssnpp.yaml`. You can use `generate_config.py` by specifying the root directory of your dataset and the files with the data shards

`python generate_config`

2. Run the train index command

`python run.py --command train_index --config config_ssnpp.yaml --xb ssnpp_1B`


3. Run the index-shard command so it produces sharded indexes, required for the search step

`python run.py --command index_shard --config config_ssnpp.yaml --xb ssnpp_1B`


6. Send jobs to the cluster to run search

`python run.py  --command search --config config_ssnpp.yaml --xb ssnpp_1B  --cluster_run --partition <PARTITION-NAME>`


Remarks about the `search` command: it is assumed that the database vectors are the query vectors when performing the search step.
a. If the query vectors are different than the database vectors, it should be passed in the xq argument
b. A new dataset needs to be prepared (step 1) before passing it to the query vectors argument `–xq`

`python run.py --command search --config config_ssnpp.yaml --xb ssnpp_1B --xq <QUERIES_DATASET_NAME>`


6. We can always run the consistency-check for sanity checks!

`python run.py  --command consistency_check--config config_ssnpp.yaml --xb ssnpp_1B`

