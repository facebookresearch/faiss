#!/usr/bin/env python3
"""
复现 Faiss Issue #5010（大规模测试）
根据 Issue 描述，问题在于：
- 当 prefetch_nthread > 0 时，所有线程都在等待单个读操作完成
- 期望：多个列表并行预取
- 实际：串行等待，导致性能下降
"""

import faiss
import numpy as np
import time
import os
import tempfile

def test_large_scale():
    """大规模测试 - 更接近真实场景"""
    
    # 更大的参数（Issue 中使用的是百万级别）
    d = 128           # 维度
    nb = 1_000_000    # 100万向量
    nq = 1000         # 1000个查询
    k = 100           # top-100
    nlist = 4096      # 4096个聚类中心（更真实）
    nprobe = 64       # 搜索64个聚类
    
    print("=" * 60)
    print("Faiss Issue #5010 大规模复现测试")
    print("=" * 60)
    print(f"配置: d={d}, nb={nb:,}, nq={nq}, k={k}, nlist={nlist}, nprobe={nprobe}")
    print(f"数据大小: ~{nb * d * 4 / 1024 / 1024:.1f} MB")
    print()
    
    # 生成随机数据
    print("生成数据...")
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    # 创建索引
    print("1. 构建 IndexIVFFlat 索引...")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    # 训练
    print("   训练中...")
    index.train(xb[:100000])  # 用10万样本训练
    
    # 转换为磁盘倒排列表
    temp_dir = tempfile.mkdtemp()
    invlists_file = os.path.join(temp_dir, "large_invlists.dat")
    print(f"2. 转换为 OnDiskInvertedLists: {invlists_file}")
    
    ondisk_lists = faiss.OnDiskInvertedLists(nlist, index.code_size, invlists_file)
    index.replace_invlists(ondisk_lists)
    
    # 添加数据
    print("3. 添加数据...")
    batch_size = 100000
    for i in range(0, nb, batch_size):
        end = min(i + batch_size, nb)
        index.add(xb[i:end])
        print(f"   已添加 {end:,}/{nb:,} 向量")
    
    # 保存并重新加载
    index_file = os.path.join(temp_dir, "large_index.bin")
    print(f"4. 保存并重新加载索引: {index_file}")
    faiss.write_index(index, index_file)
    
    print("   重新加载...")
    reloaded_index = faiss.read_index(index_file)
    reloaded_index.nprobe = nprobe
    
    # 检查磁盘文件大小
    disk_size_mb = os.path.getsize(invlists_file) / 1024 / 1024
    print(f"   磁盘倒排列表大小: {disk_size_mb:.1f} MB")
    
    print()
    print("=" * 60)
    print("性能测试（大规模）")
    print("=" * 60)
    
    # 清理系统缓存（在 macOS 上）
    print("清理系统缓存...")
    os.system("sync")
    os.system("purge 2>/dev/null")
    time.sleep(2)
    
    results = []
    for prefetch_threads in [0, 4, 16, 32]:
        # 设置预取线程数
        if hasattr(reloaded_index.invlists, 'prefetch_nthread'):
            reloaded_index.invlists.prefetch_nthread = prefetch_threads
        
        print(f"\n测试 prefetch_threads={prefetch_threads}...")
        
        # 热身（少量查询）
        _ = reloaded_index.search(xq[:10], k)
        
        # 多次测试取平均
        times = []
        for run in range(3):
            start = time.time()
            distances, labels = reloaded_index.search(xq, k)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
            print(f"  运行 {run+1}: {elapsed_ms:.2f} ms")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results.append((prefetch_threads, avg_time, std_time))
        print(f"  平均: {avg_time:.2f} ± {std_time:.2f} ms")
    
    # 分析结果
    print()
    print("=" * 60)
    print("结果分析（大规模）")
    print("=" * 60)
    
    baseline_time = results[0][1]
    print(f"基准时间 (threads=0): {baseline_time:.2f} ms")
    print()
    
    issue_detected = False
    for threads, avg_time, std_time in results[1:]:
        speedup = baseline_time / avg_time
        if speedup < 1.0:
            slowdown_pct = (avg_time / baseline_time - 1) * 100
            print(f"🔴 threads={threads:2d}: 性能下降 {slowdown_pct:+.1f}% - Issue #5010!")
            issue_detected = True
        elif speedup < 1.05:  # 提升不到5%也可能有问题
            print(f"⚠️  threads={threads:2d}: 微小提升 {(speedup-1)*100:+.1f}% (预期更高)")
        else:
            print(f"✅ threads={threads:2d}: 性能提升 {(speedup-1)*100:+.1f}%")
    
    print()
    if issue_detected:
        print("🔴 Issue #5010 已复现：多线程预取导致性能下降")
        print("   根据 Issue 描述，这是因为所有线程在等待单个磁盘读操作")
    else:
        print("✅ 未检测到 Issue #5010 症状（可能已修复或环境不同）")
    
    # 清理
    print()
    print(f"清理临时文件: {temp_dir}")
    os.remove(invlists_file)
    os.remove(index_file)
    os.rmdir(temp_dir)
    
    return results

if __name__ == "__main__":
    results = test_large_scale()
