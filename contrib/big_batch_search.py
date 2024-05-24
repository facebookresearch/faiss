# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import pickle
import os
import logging
from multiprocessing.pool import ThreadPool
import threading
import _thread
from queue import Queue
import traceback
import datetime

import numpy as np
import faiss

from faiss.contrib.inspect_tools import get_invlist


class BigBatchSearcher:
    """
    Object that manages all the data related to the computation
    except the actual within-bucket matching and the organization of the
    computation (parallel or not)
    """

    def __init__(
            self,
            index, xq, k,
            verbose=0,
            use_float16=False):

        # verbosity
        self.verbose = verbose
        self.tictoc = []

        self.xq = xq
        self.index = index
        self.use_float16 = use_float16
        keep_max = faiss.is_similarity_metric(index.metric_type)
        self.rh = faiss.ResultHeap(len(xq), k, keep_max=keep_max)
        self.t_accu = [0] * 6
        self.t_display = self.t0 = time.time()

    def start_t_accu(self):
        self.t_accu_t0 = time.time()

    def stop_t_accu(self, n):
        self.t_accu[n] += time.time() - self.t_accu_t0

    def tic(self, name):
        self.tictoc = (name, time.time())
        if self.verbose > 0:
            print(name, end="\r", flush=True)

    def toc(self):
        name, t0 = self.tictoc
        dt = time.time() - t0
        if self.verbose > 0:
            print(f"{name}: {dt:.3f} s")
        return dt

    def report(self, l):
        if self.verbose == 1 or (
            self.verbose == 2 and (
                l > 1000 and time.time() < self.t_display + 1.0
            )
        ):
            return
        t = time.time() - self.t0
        print(
            f"[{t:.1f} s] list {l}/{self.index.nlist} "
            f"times prep q {self.t_accu[0]:.3f} prep b {self.t_accu[1]:.3f} "
            f"comp {self.t_accu[2]:.3f} res {self.t_accu[3]:.3f} "
            f"wait in {self.t_accu[4]:.3f} "
            f"wait out {self.t_accu[5]:.3f} "
            f"eta {datetime.timedelta(seconds=t*self.index.nlist/(l+1)-t)} "
            f"mem {faiss.get_mem_usage_kb()}",
             end="\r" if self.verbose <= 2 else "\n",
             flush=True,
        )
        self.t_display = time.time()

    def coarse_quantization(self):
        self.tic("coarse quantization")
        bs = 65536
        nq = len(self.xq)
        q_assign = np.empty((nq, self.index.nprobe), dtype='int32')
        for i0 in range(0, nq, bs):
            i1 = min(nq, i0 + bs)
            q_dis_i, q_assign_i = self.index.quantizer.search(
                self.xq[i0:i1], self.index.nprobe)
            # q_dis[i0:i1] = q_dis_i
            q_assign[i0:i1] = q_assign_i
        self.toc()
        self.q_assign = q_assign

    def reorder_assign(self):
        self.tic("bucket sort")
        q_assign = self.q_assign
        q_assign += 1   # move -1 -> 0
        self.bucket_lims = faiss.matrix_bucket_sort_inplace(
            self.q_assign, nbucket=self.index.nlist + 1, nt=16)
        self.query_ids = self.q_assign.ravel()
        if self.verbose > 0:
            print('  number of -1s:', self.bucket_lims[1])
        self.bucket_lims = self.bucket_lims[1:]  # shift back to ignore -1s
        del self.q_assign   # inplace so let's forget about the old version...
        self.toc()

    def prepare_bucket(self, l):
        """ prepare the queries and database items for bucket l"""
        t0 = time.time()
        index = self.index
        # prepare queries
        i0, i1 = self.bucket_lims[l], self.bucket_lims[l + 1]
        q_subset = self.query_ids[i0:i1]
        xq_l = self.xq[q_subset]
        if self.by_residual:
            xq_l = xq_l - index.quantizer.reconstruct(l)
        t1 = time.time()
        # prepare database side
        list_ids, xb_l = get_invlist(index.invlists, l)

        if self.decode_func is None:
            xb_l = xb_l.ravel()
        else:
            xb_l = self.decode_func(xb_l)

        if self.use_float16:
            xb_l = xb_l.astype('float16')
            xq_l = xq_l.astype('float16')

        t2 = time.time()
        self.t_accu[0] += t1 - t0
        self.t_accu[1] += t2 - t1
        return q_subset, xq_l, list_ids, xb_l

    def add_results_to_heap(self, q_subset, D, list_ids, I):
        """add the bucket results to the heap structure"""
        if D is None:
            return
        t0 = time.time()
        if I is None:
            I = list_ids
        else:
            I = list_ids[I]
        self.rh.add_result_subset(q_subset, D, I)
        self.t_accu[3] += time.time() - t0

    def sizes_in_checkpoint(self):
        return (self.xq.shape, self.index.nprobe, self.index.nlist)

    def write_checkpoint(self, fname, completed):
        # write to temp file then move to final file
        tmpname = fname + ".tmp"
        with open(tmpname, "wb") as f:
            pickle.dump(
                {
                    "sizes": self.sizes_in_checkpoint(),
                    "completed": completed,
                    "rh": (self.rh.D, self.rh.I),
                }, f, -1)
        os.replace(tmpname, fname)

    def read_checkpoint(self, fname):
        with open(fname, "rb") as f:
            ckp = pickle.load(f)
        assert ckp["sizes"] == self.sizes_in_checkpoint()
        self.rh.D[:] = ckp["rh"][0]
        self.rh.I[:] = ckp["rh"][1]
        return ckp["completed"]


class BlockComputer:
    """ computation within one bucket """

    def __init__(
            self,
            index,
            method="knn_function",
            pairwise_distances=faiss.pairwise_distances,
            knn=faiss.knn):

        self.index = index
        if index.__class__ == faiss.IndexIVFFlat:
            index_help = faiss.IndexFlat(index.d, index.metric_type)
            decode_func = lambda x: x.view("float32")
            by_residual = False
        elif index.__class__ == faiss.IndexIVFPQ:
            index_help = faiss.IndexPQ(
                index.d, index.pq.M, index.pq.nbits, index.metric_type)
            index_help.pq = index.pq
            decode_func = index_help.pq.decode
            index_help.is_trained = True
            by_residual = index.by_residual
        elif index.__class__ == faiss.IndexIVFScalarQuantizer:
            index_help = faiss.IndexScalarQuantizer(
                index.d, index.sq.qtype, index.metric_type)
            index_help.sq = index.sq
            decode_func = index_help.sq.decode
            index_help.is_trained = True
            by_residual = index.by_residual
        else:
            raise RuntimeError(f"index type {index.__class__} not supported")
        self.index_help = index_help
        self.decode_func = None if method == "index" else decode_func
        self.by_residual = by_residual
        self.method = method
        self.pairwise_distances = pairwise_distances
        self.knn = knn

    def block_search(self, xq_l, xb_l, list_ids, k, **extra_args):
        metric_type = self.index.metric_type
        if xq_l.size == 0 or xb_l.size == 0:
            D = I = None
        elif self.method == "index":
            faiss.copy_array_to_vector(xb_l, self.index_help.codes)
            self.index_help.ntotal = len(list_ids)
            D, I = self.index_help.search(xq_l, k)
        elif self.method == "pairwise_distances":
            # TODO implement blockwise to avoid mem blowup
            D = self.pairwise_distances(xq_l, xb_l, metric=metric_type)
            I = None
        elif self.method == "knn_function":
            D, I = self.knn(xq_l, xb_l, k, metric=metric_type, **extra_args)

        return D, I


def big_batch_search(
        index, xq, k,
        method="knn_function",
        pairwise_distances=faiss.pairwise_distances,
        knn=faiss.knn,
        verbose=0,
        threaded=0,
        use_float16=False,
        prefetch_threads=1,
        computation_threads=1,
        q_assign=None,
        checkpoint=None,
        checkpoint_freq=7200,
        start_list=0,
        end_list=None,
        crash_at=-1
        ):
    """
    Search queries xq in the IVF index, with a search function that collects
    batches of query vectors per inverted list. This can be faster than the
    regular search indexes.
    Supports IVFFlat, IVFPQ and IVFScalarQuantizer.

    Supports three computation methods:
    method = "index":
        build a flat index and populate it separately for each index
    method = "pairwise_distances":
        decompress codes and compute all pairwise distances for the queries
        and index and add result to heap
    method = "knn_function":
        decompress codes and compute knn results for the queries

    threaded=0: sequential execution
    threaded=1: prefetch next bucket while computing the current one
    threaded=2: prefetch prefetch_threads buckets at a time.

    compute_threads>1: the knn function will get an additional thread_no that
        tells which worker should handle this.

    In threaded mode, the computation is tiled with the bucket perparation and
    the writeback of results (useful to maximize GPU utilization).

    use_float16: convert all matrices to float16 (faster for GPU gemm)

    q_assign: override coarse assignment, should be a matrix of size nq * nprobe

    checkpointing (only for threaded > 1):
    checkpoint: file where the checkpoints are stored
    checkpoint_freq: when to perform checkpoinging. Should be a multiple of threaded

    start_list, end_list: process only a subset of invlists
    """
    nprobe = index.nprobe

    assert method in ("index", "pairwise_distances", "knn_function")

    mem_queries = xq.nbytes
    mem_assign = len(xq) * nprobe * np.dtype('int32').itemsize
    mem_res = len(xq) * k * (
        np.dtype('int64').itemsize
        + np.dtype('float32').itemsize
    )
    mem_tot = mem_queries + mem_assign + mem_res
    if verbose > 0:
        logging.info(
            f"memory: queries {mem_queries} assign {mem_assign} "
            f"result {mem_res} total {mem_tot} = {mem_tot / (1<<30):.3f} GiB"
        )

    bbs = BigBatchSearcher(
        index, xq, k,
        verbose=verbose,
        use_float16=use_float16
    )

    comp = BlockComputer(
        index,
        method=method,
        pairwise_distances=pairwise_distances,
        knn=knn
    )

    bbs.decode_func = comp.decode_func

    bbs.by_residual = comp.by_residual
    if q_assign is None:
        bbs.coarse_quantization()
    else:
        bbs.q_assign = q_assign
    bbs.reorder_assign()

    if end_list is None:
        end_list = index.nlist

    completed = set()
    if checkpoint is not None:
        assert (start_list, end_list) == (0, index.nlist)
        if os.path.exists(checkpoint):
            logging.info(f"recovering checkpoint: {checkpoint}")
            completed = bbs.read_checkpoint(checkpoint)
            logging.info(f"   already completed: {len(completed)}")
        else:
            logging.info("no checkpoint: starting from scratch")

    if threaded == 0:
        # simple sequential version

        for l in range(start_list, end_list):
            bbs.report(l)
            q_subset, xq_l, list_ids, xb_l = bbs.prepare_bucket(l)
            t0i = time.time()
            D, I = comp.block_search(xq_l, xb_l, list_ids, k)
            bbs.t_accu[2] += time.time() - t0i
            bbs.add_results_to_heap(q_subset, D, list_ids, I)

    elif threaded == 1:

        # parallel version with granularity 1

        def add_results_and_prefetch(to_add, l):
            """ perform the addition for the previous bucket and
            prefetch the next (if applicable) """
            if to_add is not None:
                bbs.add_results_to_heap(*to_add)
            if l < index.nlist:
                return bbs.prepare_bucket(l)

        prefetched_bucket = bbs.prepare_bucket(start_list)
        to_add = None
        pool = ThreadPool(1)

        for l in range(start_list, end_list):
            bbs.report(l)
            prefetched_bucket_a = pool.apply_async(
                add_results_and_prefetch, (to_add, l + 1))
            q_subset, xq_l, list_ids, xb_l = prefetched_bucket
            bbs.start_t_accu()
            D, I = comp.block_search(xq_l, xb_l, list_ids, k)
            bbs.stop_t_accu(2)
            to_add = q_subset, D, list_ids, I
            bbs.start_t_accu()
            prefetched_bucket = prefetched_bucket_a.get()
            bbs.stop_t_accu(4)

        bbs.add_results_to_heap(*to_add)
        pool.close()
    else:

        def task_manager_thread(
            task,
            pool_size,
            start_task,
            end_task,
            completed,
            output_queue,
            input_queue,
        ):
            try:
                with ThreadPool(pool_size) as pool:
                    res = [pool.apply_async(
                        task,
                        args=(i, output_queue, input_queue))
                        for i in range(start_task, end_task)
                        if i not in completed]
                    for r in res:
                        r.get()
                    pool.close()
                    pool.join()
                output_queue.put(None)
            except:
                traceback.print_exc()
                _thread.interrupt_main()
                raise

        def task_manager(*args):
            task_manager = threading.Thread(
                target=task_manager_thread,
                args=args,
            )
            task_manager.daemon = True
            task_manager.start()
            return task_manager

        def prepare_task(task_id, output_queue, input_queue=None):
            try:
                logging.info(f"Prepare start: {task_id}")
                q_subset, xq_l, list_ids, xb_l = bbs.prepare_bucket(task_id)
                output_queue.put((task_id, q_subset, xq_l, list_ids, xb_l))
                logging.info(f"Prepare end: {task_id}")
            except:
                traceback.print_exc()
                _thread.interrupt_main()
                raise

        def compute_task(task_id, output_queue, input_queue):
            try:
                logging.info(f"Compute start: {task_id}")
                t_wait_out = 0
                while True:
                    t0 = time.time()
                    logging.info(f'Compute input: task {task_id}')
                    input_value = input_queue.get()
                    t_wait_in = time.time() - t0
                    if input_value is None:
                        # signal for other compute tasks
                        input_queue.put(None)
                        break
                    centroid, q_subset, xq_l, list_ids, xb_l = input_value
                    logging.info(f'Compute work: task {task_id}, centroid {centroid}')
                    t0 = time.time()
                    if computation_threads > 1:
                        D, I = comp.block_search(
                            xq_l, xb_l, list_ids, k, thread_id=task_id
                        )
                    else:
                        D, I = comp.block_search(xq_l, xb_l, list_ids, k)
                    t_compute = time.time() - t0
                    logging.info(f'Compute output: task {task_id}, centroid {centroid}')
                    t0 = time.time()
                    output_queue.put(
                        (centroid, t_wait_in, t_wait_out, t_compute, q_subset, D, list_ids, I)
                    )
                    t_wait_out = time.time() - t0
                logging.info(f"Compute end: {task_id}")
            except:
                traceback.print_exc()
                _thread.interrupt_main()
                raise

        prepare_to_compute_queue = Queue(2)
        compute_to_main_queue = Queue(2)
        compute_task_manager = task_manager(
            compute_task,
            computation_threads,
            0,
            computation_threads,
            set(),
            compute_to_main_queue,
            prepare_to_compute_queue,
        )
        prepare_task_manager = task_manager(
            prepare_task,
            prefetch_threads,
            start_list,
            end_list,
            completed,
            prepare_to_compute_queue,
            None,
        )

        t_checkpoint = time.time()
        while True:
            logging.info("Waiting for result")
            value = compute_to_main_queue.get()
            if not value:
                break
            centroid, t_wait_in, t_wait_out, t_compute, q_subset, D, list_ids, I = value
            # to test checkpointing
            if centroid == crash_at:
                1 / 0
            bbs.t_accu[2] += t_compute
            bbs.t_accu[4] += t_wait_in
            bbs.t_accu[5] += t_wait_out
            logging.info(f"Adding to heap start: centroid {centroid}")
            bbs.add_results_to_heap(q_subset, D, list_ids, I)
            logging.info(f"Adding to heap end: centroid {centroid}")
            completed.add(centroid)
            bbs.report(centroid)
            if checkpoint is not None:
                if time.time() - t_checkpoint > checkpoint_freq:
                    logging.info("writing checkpoint")
                    bbs.write_checkpoint(checkpoint, completed)
                    t_checkpoint = time.time()

        prepare_task_manager.join()
        compute_task_manager.join()

    bbs.tic("finalize heap")
    bbs.rh.finalize()
    bbs.toc()

    return bbs.rh.D, bbs.rh.I
