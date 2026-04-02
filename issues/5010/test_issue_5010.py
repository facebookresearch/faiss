#!/usr/bin/env python3
"""
复现 Faiss Issue #5010：OnDiskInvertedLists 在多线程环境下性能降低
"""

import faiss
import numpy as np
import time
import os
import tempfile

def test_ondisk_prefetch_issue():
    """测试磁盘倒排列表预取性能"""
    
    # 小规模测试参数
    d = 128          # 向量维度
    nb = 10000       # 数据库大小
    nq = 100         # 查询数量
    k = 10           # 返回top-k
    nlist = 100      # 聚类中心数
    nprobe = 10      # 搜索的聚类数
    
    print("=" * 60)
    print("Faiss Issue #5010 复现测试")
    print("=" * 60)
    print(f"配置: d={d}, nb={nb}, nq={nq}, k={k}, nlist={nlist}, nprobe={nprobe}")
    print()
    
    # 生成随机数据
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    # 创建索引
    print("1. 构建 IndexIVFFlat 索引...")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    # 训练
    index.train(xb)
    
    # 转换为磁盘倒排列表
    temp_dir = tempfile.mkdtemp()
    invlists_file = os.path.join(temp_dir, "invlists.dat")
    print(f"2. 转换为 OnDiskInvertedLists: {invlists_file}")
    
    ondisk_lists = faiss.OnDiskInvertedLists(nlist, index.code_size, invlists_file)
    index.replace_invlists(ondisk_lists)
    
    # 添加数据
    print("3. 添加数据...")
    index.add(xb)
    
    # 保存并重新加载（模拟生产场景）
    index_file = os.path.join(temp_dir, "index.bin")
    print(f"4. 保存并重新加载索引: {index_file}")
    faiss.write_index(index, index_file)
    reloaded_index = faiss.read_index(index_file)
    
    # 设置搜索参数
    reloaded_index.nprobe = nprobe
    
    print()
    print("=" * 60)
    print("性能测试（Issue #5010 复现）")
    print("=" * 60)
    
    # 测试不同预取线程数
    results = []
    for prefetch_threads in [0, 4, 32]:
        # 设置预取线程数
        if hasattr(reloaded_index.invlists, 'prefetch_nthread'):
            reloaded_index.invlists.prefetch_nthread = prefetch_threads
        
        # 热身
        _ = reloaded_index.search(xq[:5], k)
        
        # 计时测试
        start = time.time()
        distances, labels = reloaded_index.search(xq, k)
        elapsed_ms = (time.time() - start) * 1000
        
        results.append((prefetch_threads, elapsed_ms))
        print(f"prefetch_threads={prefetch_threads:2d}: {elapsed_ms:7.2f} ms")
    
    # 分析结果
    print()
    print("=" * 60)
    print("结果分析")
    print("=" * 60)
    
    baseline_time = results[0][1]  # prefetch_threads=0 的时间
    print(f"基准时间 (threads=0): {baseline_time:.2f} ms")
    print()
    
    issue_detected = False
    for threads, elapsed in results[1:]:
        speedup = baseline_time / elapsed
        if speedup < 1.0:
            slowdown_pct = (elapsed / baseline_time - 1) * 100
            print(f"⚠️  threads={threads}: 性能下降 {slowdown_pct:.1f}% (Issue #5010 症状)")
            issue_detected = True
        else:
            speedup_pct = (speedup - 1) * 100
            print(f"✅ threads={threads}: 性能提升 {speedup_pct:.1f}%")
    
    print()
    if issue_detected:
        print("🔴 Issue #5010 已复现：多线程预取导致性能下降")
    else:
        print("✅ 未检测到 Issue #5010 症状")
    
    # 清理
    print()
    print(f"清理临时文件: {temp_dir}")
    os.remove(invlists_file)
    os.remove(index_file)
    os.rmdir(temp_dir)
    
    return results

if __name__ == "__main__":
    results = test_ondisk_prefetch_issue()
