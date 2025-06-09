import faiss
from faiss.contrib import datasets, evaluation
import numpy as np
import time

n_train = 0
n_base = 10000
n_query = 500

n_dim = 32
k = 12
metric = faiss.METRIC_L2


def check_correctness(Dref, Iref, Dnew, Inew):
    try:
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)
    except Exception as e:
        print("Wrong correctness:", e)
    else:
        print("Correct! check_ref_knn_with_draws passed successfully.\n\n")


def test_train_search_fp16(res, cagraIndexConfig, fp16_database, fp16_queries):
    print("==================== Testing Cagra Train and Search in FP16 ====================")
    print("Running Cagra Train and Search in FP16")
    index = faiss.GpuIndexCagra(res, n_dim, metric, cagraIndexConfig)
    index.train(fp16_database, faiss.Float16)
    Dnew, Inew = index.search(fp16_queries, k, numeric_type=faiss.Float16)

    return Dnew, Inew
    
    # # The following is here just to check that check_ref_knn_with_draws will
    # properly raises an AssertionError when wrong indices and distances are given
    # false_I = np.zeros((n_query, k), dtype=np.int64)
    # false_D = np.random.uniform(
    #     low=0.0,
    #     high=100.0,
    #     size=(n_query, k)
    # ).astype(np.float32)
    # check_correctness(Dref, Iref, Dnew, Inew)


def test_train_search_fp32(res, cagraIndexConfig, fp32_database, fp32_queries):
    print("==================== Testing Cagra Train and Search in FP32 ====================")
    print("Running Cagra Train and Search in FP32")
    index = faiss.GpuIndexCagra(res, n_dim, metric, cagraIndexConfig)
    index.train(fp32_database)
    Dnew, Inew = index.search(fp32_queries, k)

    return Dnew, Inew


def gpu_to_cpu_fp16(res, cagraIndexConfig, fp16_database, fp32_queries):
    print("==================== Testing Cagra GPU Train FP16 -> CPU Search FP32 ====================")
    print("Running Cagra Train in FP16")
    index = faiss.GpuIndexCagra(res, n_dim, metric, cagraIndexConfig)
    index.train(fp16_database, faiss.Float16)

    print("Copying trained GPU index (GpuIndexCagra) to CPU index")
    copied_cpu_index = faiss.index_gpu_to_cpu(index)

    print("Searching copied CPU index with FP32")
    search_params = faiss.SearchParametersHNSW()
    search_params.efSearch = 2*k
    Dnew, Inew = copied_cpu_index.search(fp32_queries, k, params=search_params)
    return Dnew, Inew


def gpu_to_cpu_fp32(res, cagraIndexConfig, fp32_database, fp32_queries):
    print("==================== Testing Cagra GPU Train FP32 -> CPU Search FP32 ====================")
    print("Running Cagra Train in FP32")
    index = faiss.GpuIndexCagra(res, n_dim, metric, cagraIndexConfig)
    index.train(fp32_database)

    print("Copying trained GPU index (GpuIndexCagra) to CPU index")
    copied_cpu_index = faiss.index_gpu_to_cpu(index)

    print("Searching copied CPU index with FP32")
    search_params = faiss.SearchParametersHNSW()
    search_params.efSearch = 2*k
    Dnew, Inew = copied_cpu_index.search(fp32_queries, k, params=search_params)
    return Dnew, Inew


def gpu_to_cpu_to_gpu_fp16(res, cagraIndexConfig, fp16_database, fp16_queries):
    print("==================== Testing Cagra GPU Train FP16 -> CPU -> GPU Search FP16 ====================")
    # IndexHNSWCagra is not directly exposed in python, so we check copyFrom (CPU ->GPU copy)
    # by doing (GPU -> CPU -> GPU) copy
    print("Running Cagra Train in FP16")
    index = faiss.GpuIndexCagra(res, n_dim, metric, cagraIndexConfig)
    index.train(fp16_database, faiss.Float16)

    print("Copying trained GPU index (GpuIndexCagra) to CPU index")
    copied_cpu_index = faiss.index_gpu_to_cpu(index)

    print("Copying CPU index to GPU index again")
    copied_gpu_index = faiss.index_cpu_to_gpu(res, 0, copied_cpu_index)

    print("Searching copied GPU index with FP16")
    Dnew, Inew = copied_gpu_index.search(fp16_queries, k, numeric_type=faiss.Float16)
    return Dnew, Inew


def gpu_to_cpu_to_gpu_fp32(res, cagraIndexConfig, fp32_database, fp32_queries):
    print("==================== Testing Cagra GPU Train FP32 -> CPU -> GPU Search FP32 ====================")
    # IndexHNSWCagra is not directly exposed in python, so we check copyFrom (CPU ->GPU copy)
    # by doing (GPU -> CPU -> GPU) copy
    print("Running Cagra Train in FP32")
    index = faiss.GpuIndexCagra(res, n_dim, metric, cagraIndexConfig)
    index.train(fp32_database)

    print("Copying trained GPU index (GpuIndexCagra) to CPU index")
    copied_cpu_index = faiss.index_gpu_to_cpu(index)

    print("Copying CPU index to GPU index again")
    copied_gpu_index = faiss.index_cpu_to_gpu(res, 0, copied_cpu_index)

    print("Searching copied GPU index with FP32")
    Dnew, Inew = copied_gpu_index.search(fp32_queries, k)
    return Dnew, Inew


if __name__ == "__main__":
    res = faiss.StandardGpuResources()
    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.build_algo = faiss.graph_build_algo_NN_DESCENT

    print("Generate dataset")
    dataset = datasets.SyntheticDataset(n_dim, n_train, n_base, n_query)
    half_base_data = dataset.get_database().astype(np.float16)

    print("Running knn for reference")
    Dref, Iref = faiss.knn(dataset.get_queries(), dataset.get_database(), k, metric)

    Dnew, Inew = test_train_search_fp16(res, 
                                        cagraIndexConfig, 
                                        dataset.get_database().astype(np.float16),
                                        dataset.get_queries().astype(np.float16))
    check_correctness(Dref, Iref, Dnew, Inew)

    Dnew, Inew = test_train_search_fp32(res, 
                                        cagraIndexConfig, 
                                        dataset.get_database(),
                                        dataset.get_queries())
    check_correctness(Dref, Iref, Dnew, Inew)

    Dnew, Inew = gpu_to_cpu_fp16(res, 
                                cagraIndexConfig, 
                                dataset.get_database().astype(np.float16),
                                dataset.get_queries())
    check_correctness(Dref, Iref, Dnew, Inew)

    Dnew, Inew = gpu_to_cpu_fp32(res, 
                                cagraIndexConfig, 
                                dataset.get_database(),
                                dataset.get_queries())
    check_correctness(Dref, Iref, Dnew, Inew)

    Dnew, Inew = gpu_to_cpu_to_gpu_fp16(res, 
                                        cagraIndexConfig, 
                                        dataset.get_database().astype(np.float16),
                                        dataset.get_queries().astype(np.float16))
    check_correctness(Dref, Iref, Dnew, Inew)

    Dnew, Inew = gpu_to_cpu_to_gpu_fp32(res, 
                                        cagraIndexConfig, 
                                        dataset.get_database(),
                                        dataset.get_queries())
    check_correctness(Dref, Iref, Dnew, Inew)
