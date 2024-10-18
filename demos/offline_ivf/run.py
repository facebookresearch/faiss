# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from utils import (
    load_config,
    add_group_args,
)
from offline_ivf import OfflineIVF
import faiss
from typing import List, Callable, Dict
import submitit


def join_lists_in_dict(poss: List[str]) -> List[str]:
    """
    Joins two lists of prod and non-prod values, checking if the prod value is already included.
    If there is no non-prod list, it returns the prod list.
    """
    if "non-prod" in poss.keys():
        all_poss = poss["non-prod"]
        if poss["prod"][-1] not in poss["non-prod"]:
            all_poss += poss["prod"]
        return all_poss
    else:
        return poss["prod"]


def main(
    args: argparse.Namespace,
    cfg: Dict[str, str],
    nprobe: int,
    index_factory_str: str,
) -> None:
    oivf = OfflineIVF(cfg, args, nprobe, index_factory_str)
    eval(f"oivf.{args.command}()")


def process_options_and_run_jobs(args: argparse.Namespace) -> None:
    """
    If "--cluster_run", it launches an array of jobs to the cluster using the submitit library for all the index strings. In
    the case of evaluate, it launches a job for each index string and nprobe pair. Otherwise, it launches a single job
    that is ran locally with the prod values for index string and nprobe.
    """

    cfg = load_config(args.config)
    index_strings = cfg["index"]
    nprobes = cfg["nprobe"]
    if args.command == "evaluate":
        if args.cluster_run:
            all_nprobes = join_lists_in_dict(nprobes)
            all_index_strings = join_lists_in_dict(index_strings)
            for index_factory_str in all_index_strings:
                for nprobe in all_nprobes:
                    launch_job(main, args, cfg, nprobe, index_factory_str)
        else:
            launch_job(
                main, args, cfg, nprobes["prod"][-1], index_strings["prod"][-1]
            )
    else:
        if args.cluster_run:
            all_index_strings = join_lists_in_dict(index_strings)
            for index_factory_str in all_index_strings:
                launch_job(
                    main, args, cfg, nprobes["prod"][-1], index_factory_str
                )
        else:
            launch_job(
                main, args, cfg, nprobes["prod"][-1], index_strings["prod"][-1]
            )


def launch_job(
    func: Callable,
    args: argparse.Namespace,
    cfg: Dict[str, str],
    n_probe: int,
    index_str: str,
) -> None:
    """
    Launches an array of slurm jobs to the cluster using the submitit library.
    """

    if args.cluster_run:
        assert args.num_nodes >= 1
        executor = submitit.AutoExecutor(folder=args.logs_dir)

        executor.update_parameters(
            nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
            cpus_per_task=args.cpus_per_task,
            tasks_per_node=args.tasks_per_node,
            name=args.job_name,
            slurm_partition=args.partition,
            slurm_time=70 * 60,
        )
        if args.slurm_constraint:
            executor.update_parameters(slurm_constraint=args.slurm_constrain)

        job = executor.submit(func, args, cfg, n_probe, index_str)
        print(f"Job id: {job.job_id}")
    else:
        func(args, cfg, n_probe, index_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("general")

    add_group_args(group, "--command", required=True, help="command to run")
    add_group_args(
        group,
        "--config",
        required=True,
        help="config yaml with the dataset specs",
    )
    add_group_args(
        group, "--nt", type=int, default=96, help="nb search threads"
    )
    add_group_args(
        group,
        "--no_residuals",
        action="store_false",
        help="set index.by_residual to False during train index.",
    )

    group = parser.add_argument_group("slurm_job")

    add_group_args(
        group,
        "--cluster_run",
        action="store_true",
        help=" if True, runs in cluster",
    )
    add_group_args(
        group,
        "--job_name",
        type=str,
        default="oivf",
        help="cluster job name",
    )
    add_group_args(
        group,
        "--num_nodes",
        type=str,
        default=1,
        help="num of nodes per job",
    )
    add_group_args(
        group,
        "--tasks_per_node",
        type=int,
        default=1,
        help="tasks per job",
    )

    add_group_args(
        group,
        "--gpus_per_node",
        type=int,
        default=8,
        help="cluster job name",
    )
    add_group_args(
        group,
        "--cpus_per_task",
        type=int,
        default=80,
        help="cluster job name",
    )

    add_group_args(
        group,
        "--logs_dir",
        type=str,
        default="/checkpoint/marialomeli/offline_faiss/logs",
        help="cluster job name",
    )

    add_group_args(
        group,
        "--slurm_constraint",
        type=str,
        default=None,
        help="can be volta32gb for the fair cluster",
    )

    add_group_args(
        group,
        "--partition",
        type=str,
        default="learnlab",
        help="specify which partition to use if ran on cluster with job arrays",
        choices=[
            "learnfair",
            "devlab",
            "scavenge",
            "learnlab",
            "nllb",
            "seamless",
            "seamless_medium",
            "learnaccel",
            "onellm_low",
            "learn",
            "scavenge",
        ],
    )

    group = parser.add_argument_group("dataset")

    add_group_args(group, "--xb", required=True, help="database vectors")
    add_group_args(group, "--xq", help="query vectors")

    args = parser.parse_args()
    print("args:", args)
    faiss.omp_set_num_threads(args.nt)
    process_options_and_run_jobs(args=args)
