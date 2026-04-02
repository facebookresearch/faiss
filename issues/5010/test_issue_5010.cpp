#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/invlists/OnDiskInvertedLists.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 小规模测试
    int d = 128, nb = 10000, nq = 100, k = 10;
    int nlist = 100, nprobe = 10;
    
    std::cout << "Building test index..." << std::endl;
    
    // 生成随机数据
    std::vector<float> xb(nb * d), xq(nq * d);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0, 1);
    for (auto& v : xb) v = dist(rng);
    for (auto& v : xq) v = dist(rng);
    
    // 创建索引
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    index.train(nb, xb.data());
    
    // 转换为磁盘索引
    faiss::OnDiskInvertedLists* invlists = 
        new faiss::OnDiskInvertedLists(nlist, index.code_size, "/tmp/test_invlists.dat");
    index.replace_invlists(invlists, true);
    index.add(nb, xb.data());
    
    // 保存并重新加载
    faiss::write_index(&index, "/tmp/test_index.bin");
    auto* reloaded = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index("/tmp/test_index.bin"));
    reloaded->nprobe = nprobe;
    
    // 测试
    std::vector<float> dist_out(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    std::cout << "\n=== Performance Test ===" << std::endl;
    
    for (int nt : {0, 4, 32}) {
        auto start = std::chrono::high_resolution_clock::now();
        reloaded->search(nq, xq.data(), k, dist_out.data(), labels.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Threads=" << nt << ": " << ms << " ms" << std::endl;
    }
    
    delete reloaded;
    return 0;
}
