#!/usr/bin/env python3
"""
复现 Faiss Issue #5010 + 统计 page fault 次数
通过 resource.getrusage 在每次搜索前后采样 minor/major page fault，
观察不同 prefetch_nthread 设置下 page fault 的变化。
"""

import faiss
import numpy as np
import time
import os
import sys
import resource
import tempfile


def get_page_faults():
    """获取当前进程的 page fault 计数"""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_minflt, ru.ru_majflt


def drop_page_cache(filepath):
    """
    尝试清除文件的 page cache，让后续访问产生真实的 page fault。
    macOS: purge（需要 sudo）
    Linux: 写 /proc/sys/vm/drop_caches
    """
    os.system("sync")
    if sys.platform == "darwin":
        # macOS: purge 需要 root 权限，失败也无妨
        ret = os.system("sudo purge 2>/dev/null")
        if ret != 0:
            print("  [警告] purge 失败（需要 sudo 权限），page cache 可能未清除")
            # 备选：用 dd 读一块大内存来挤掉 cache
    else:
        # Linux
        ret = os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null")
        if ret != 0:
            print("  [警告] drop_caches 失败（需要 sudo 权限）")


def run_search_with_pagefault_stats(index, xq, k, label=""):
    """执行搜索并统计 page fault"""
    minflt_before, majflt_before = get_page_faults()
    t0 = time.time()

    distances, labels = index.search(xq, k)

    elapsed_ms = (time.time() - t0) * 1000
    minflt_after, majflt_after = get_page_faults()

    delta_minflt = minflt_after - minflt_before
    delta_majflt = majflt_after - majflt_before

    return {
        "label": label,
        "time_ms": elapsed_ms,
        "minor_faults": delta_minflt,
        "major_faults": delta_majflt,
        "total_faults": delta_minflt + delta_majflt,
        "distances": distances,
        "labels": labels,
    }


def test_ondisk_prefetch_pagefault():
    """测试磁盘倒排列表预取性能 + page fault 统计"""

    # 参数（用小规模先验证逻辑，可按需放大）
    d = 128           # 向量维度
    nb = 100_000      # 数据库大小
    nq = 200          # 查询数量
    k = 10            # 返回 top-k
    nlist = 256       # 聚类中心数
    nprobe = 32       # 搜索的聚类数

    print("=" * 70)
    print("Faiss Issue #5010 复现 + Page Fault 统计")
    print("=" * 70)
    print(f"配置: d={d}, nb={nb:,}, nq={nq}, k={k}, nlist={nlist}, nprobe={nprobe}")
    print(f"平台: {sys.platform}, 页大小: {resource.getpagesize()} bytes")
    print()

    # ---------- 构建索引 ----------
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    print("1. 构建 IndexIVFFlat 索引 + 训练...")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(xb)

    # 转换为 OnDiskInvertedLists
    temp_dir = tempfile.mkdtemp()
    invlists_file = os.path.join(temp_dir, "invlists.dat")
    print(f"2. OnDiskInvertedLists → {invlists_file}")

    ondisk_lists = faiss.OnDiskInvertedLists(nlist, index.code_size, invlists_file)
    index.replace_invlists(ondisk_lists)

    print("3. 添加数据...")
    index.add(xb)

    # 保存 → 重新加载（模拟生产场景，mmap 读取）
    index_file = os.path.join(temp_dir, "index.bin")
    print(f"4. 保存并重新加载索引...")
    faiss.write_index(index, index_file)

    # 释放原始索引
    del index, ondisk_lists

    reloaded_index = faiss.read_index(index_file)
    reloaded_index.nprobe = nprobe

    disk_size_mb = os.path.getsize(invlists_file) / 1024 / 1024
    print(f"   磁盘倒排列表大小: {disk_size_mb:.1f} MB")

    # ---------- 性能测试 ----------
    print()
    print("=" * 70)
    print(f"{'threads':>8} | {'run':>4} | {'time_ms':>10} | {'minor_flt':>10} | "
          f"{'major_flt':>10} | {'total_flt':>10}")
    print("-" * 70)

    all_results = {}
    prefetch_thread_configs = [0, 4, 16, 32]
    num_runs = 3

    for prefetch_threads in prefetch_thread_configs:
        # 获取底层 OnDiskInvertedLists 并设置预取线程数
        # read_index 返回的 invlists 类型是 InvertedLists*，
        # 需要通过 downcast 获取 OnDiskInvertedLists
        ondisk_il = faiss.downcast_InvertedLists(reloaded_index.invlists)
        if hasattr(ondisk_il, "prefetch_nthread"):
            ondisk_il.prefetch_nthread = prefetch_threads

        run_results = []
        for run_idx in range(num_runs):
            # 每次运行前清除 page cache（尽力而为）
            drop_page_cache(invlists_file)
            time.sleep(0.5)  # 等待 cache 清除生效

            result = run_search_with_pagefault_stats(
                reloaded_index, xq, k,
                label=f"threads={prefetch_threads}, run={run_idx+1}"
            )
            run_results.append(result)

            print(f"{prefetch_threads:>8} | {run_idx+1:>4} | "
                  f"{result['time_ms']:>10.2f} | {result['minor_faults']:>10,} | "
                  f"{result['major_faults']:>10,} | {result['total_faults']:>10,}")

        all_results[prefetch_threads] = run_results

    # ---------- 汇总分析 ----------
    print()
    print("=" * 70)
    print("汇总（取 3 次运行的平均值）")
    print("=" * 70)
    print(f"{'threads':>8} | {'avg_ms':>10} | {'avg_minor':>10} | "
          f"{'avg_major':>10} | {'avg_total':>10} | {'vs baseline':>12}")
    print("-" * 70)

    baseline_ms = None
    for pt in prefetch_thread_configs:
        runs = all_results[pt]
        avg_ms = np.mean([r["time_ms"] for r in runs])
        avg_minor = np.mean([r["minor_faults"] for r in runs])
        avg_major = np.mean([r["major_faults"] for r in runs])
        avg_total = np.mean([r["total_faults"] for r in runs])

        if baseline_ms is None:
            baseline_ms = avg_ms
            vs = "baseline"
        else:
            ratio = avg_ms / baseline_ms
            if ratio > 1.0:
                vs = f"🔴 +{(ratio-1)*100:.1f}%"
            elif ratio > 0.95:
                vs = f"⚠️  {(ratio-1)*100:+.1f}%"
            else:
                vs = f"✅ {(ratio-1)*100:+.1f}%"

        print(f"{pt:>8} | {avg_ms:>10.2f} | {avg_minor:>10,.0f} | "
              f"{avg_major:>10,.0f} | {avg_total:>10,.0f} | {vs:>12}")

    # ---------- 清理 ----------
    print()
    print(f"清理临时文件: {temp_dir}")
    del reloaded_index
    try:
        os.remove(invlists_file)
        os.remove(index_file)
        os.rmdir(temp_dir)
    except OSError as e:
        print(f"  清理失败: {e}")

    return all_results


if __name__ == "__main__":
    results = test_ondisk_prefetch_pagefault()
