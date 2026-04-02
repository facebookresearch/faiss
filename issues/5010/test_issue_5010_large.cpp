#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/invlists/OnDiskInvertedLists.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    // 增加数据规模以匹配原始 Issue
    int d = 384;      // all-MiniLM-L6-v2 维度
    int nb = 200000;  // 20万向量（平衡测试时间和数据规模）
    int nq = 1000;    // 1000个查询
    int k = 10;
    int nlist = 1024; // 增加到1024
    int nprobe = 64;  // 增加到64
    
    std::cout << "=== Faiss Issue #5010 - Large Scale Test ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  - Dimensions: " << d << std::endl;
    std::cout << "  - Vectors: " << nb << std::endl;
    std::cout << "  - Queries: " << nq << std::endl;
    std::cout << "  - nlist: " << nlist << std::endl;
    std::cout << "  - nprobe: " << nprobe << std::endl;
    std::cout << std::endl;
    
    std::cout << "Building test index (this may take a minute)..." << std::endl;
    
    // 生成随机数据
    std::vector<float> xb(nb * d), xq(nq * d);
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(0, 1);
    for (auto& v : xb) v = dist(rng);
    for (auto& v : xq) v = dist(rng);
    
    // 创建索引
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    
    std::cout << "Training index..." << std::endl;
    index.train(nb, xb.data());
    
    // 转换为磁盘索引
    std::cout << "Converting to on-disk index..." << std::endl;
    faiss::OnDiskInvertedLists* invlists = 
        new faiss::OnDiskInvertedLists(nlist, index.code_size, "/tmp/test_invlists_large.dat");
    index.replace_invlists(invlists, true);
    
    std::cout << "Adding vectors..." << std::endl;
    index.add(nb, xb.data());
    
    // 保存并重新加载
    std::cout << "Saving index..." << std::endl;
    faiss::write_index(&index, "/tmp/test_index_large.bin");
    
    std::cout << "Reloading index..." << std::endl;
    auto* reloaded = dynamic_cast<faiss::IndexIVFFlat*>(
        faiss::read_index("/tmp/test_index_large.bin"));
    reloaded->nprobe = nprobe;
    
    // 获取 OnDiskInvertedLists 指针
    auto* ondisk = dynamic_cast<faiss::OnDiskInvertedLists*>(reloaded->invlists);
    if (!ondisk) {
        std::cerr << "Error: Not an OnDiskInvertedLists!" << std::endl;
        return 1;
    }
    
    // 测试不同预取配置
    std::vector<int> configs = {0, 1, 4, 8, 16, 32};
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    std::cout << "\n=== Performance Test (20万向量) ===" << std::endl;
    std::cout << "| Prefetch Threads | Latency (ms) | Change     |" << std::endl;
    std::cout << "|------------------|--------------|------------|" << std::endl;
    
    double baseline = 0;
    for (int nt : configs) {
        // 设置预取线程数
        ondisk->prefetch_nthread = nt;
        
        // 预热（3次）
        for (int i = 0; i < 3; i++) {
            reloaded->search(10, xq.data(), k, distances.data(), labels.data());
        }
        
        // 实际测试（3次取平均，减少测试时间）
        const int num_runs = 3;
        double total_ms = 0;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            reloaded->search(nq, xq.data(), k, distances.data(), labels.data());
            auto end = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_ms += ms;
        }
        
        double avg_ms = total_ms / num_runs;
        
        if (nt == 0) {
            baseline = avg_ms;
            std::cout << "| No prefetch      | " 
                     << std::fixed << std::setprecision(2) << std::setw(9) << avg_ms 
                     << "  | baseline   |" << std::endl;
        } else {
            double pct = ((avg_ms - baseline) / baseline) * 100;
            std::cout << "| " << std::setw(2) << nt << " threads       | " 
                     << std::fixed << std::setprecision(2) << std::setw(9) << avg_ms 
                     << "  | " << std::showpos << std::setw(6) << pct << std::noshowpos << "% |" 
                     << std::endl;
        }
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    delete reloaded;
    return 0;
}
