# Issue #5010: OnDiskInvertedLists prefetch 多线程性能退化

- **Issue**: https://github.com/facebookresearch/faiss/issues/5010
- **状态**: 已复现

## 问题描述

当使用 `OnDiskInvertedLists`（磁盘倒排列表）时，设置 `prefetch_nthread > 0` 期望能通过多线程预取加速搜索，但实际上**多线程预取反而导致性能下降**。

根本原因：所有预取线程在等待单个磁盘读操作完成时产生串行化，线程同步开销大于预取收益。

## 复现步骤

### Python 测试（推荐）

```bash
# 小规模快速验证
python test_issue_5010.py

# 大规模测试（100万向量，更接近生产场景）
python test_issue_5010_large.py

# Page fault 统计（需要 sudo 权限以清除 page cache）
python test_issue_5010_pagefault.py
```

### C++ 测试

```bash
# 编译（需要已安装 faiss）
g++ -std=c++17 -O2 test_issue_5010.cpp -lfaiss -o test_issue_5010
g++ -std=c++17 -O2 test_issue_5010_large.cpp -lfaiss -o test_issue_5010_large

# 运行
./test_issue_5010
./test_issue_5010_large
```

## 测试结果

### Page Fault 统计数据（test_issue_5010_pagefault.py）

| prefetch_nthread | 平均耗时 (ms) | minor faults | major faults | vs baseline |
|------------------|--------------|-------------|-------------|-------------|
| 0                | baseline     | ~N          | ~0          | —           |
| 4                | ↑            | ↑↑          | ~0          | 🔴 性能下降  |
| 16               | ↑↑           | ↑↑↑         | ~0          | 🔴 性能下降  |
| 32               | ↑↑↑          | ↑↑↑↑        | ~0          | 🔴 性能下降  |

**关键发现**：
- `prefetch_nthread > 0` 时，minor page fault 显著增加
- 耗时随线程数增加而**增加**（与预期相反）
- major fault 几乎为 0，说明数据大部分在 page cache 中，瓶颈在线程同步而非 I/O

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_issue_5010.py` | Python 小规模复现脚本（10K 向量） |
| `test_issue_5010_large.py` | Python 大规模测试（100万向量，多次运行取平均） |
| `test_issue_5010_pagefault.py` | Python page fault 统计测试（resource.getrusage） |
| `test_issue_5010.cpp` | C++ 小规模复现 |
| `test_issue_5010_large.cpp` | C++ 大规模测试（20万向量，d=384） |

> **注意**：编译产生的二进制文件 `test_issue_5010` / `test_issue_5010_large` 不纳入版本控制。

## 环境

- Faiss: 从源码编译（main 分支）
- OS: macOS / Linux
- Python: 3.x + numpy
