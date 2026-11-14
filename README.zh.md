# Faiss

[English](README.md) | [简体中文](README.zh.md)

Faiss 是一个用于稠密向量高效相似性检索与聚类的代码库。它包含可对任意规模向量集进行检索的算法，即使向量集大到无法全部装入内存（RAM）也同样支持。此外，它还提供了用于性能评估和参数调优的配套支持代码。Faiss 采用 C++ 编写，并提供完整的 Python/numpy 封装。其中部分最实用的算法已在 GPU 上实现。该项目主要由 Meta 的[基础人工智能研究团队（Fundamental AI Research）](https://ai.facebook.com/)开发。

## 最新动态

关于最新特性的详细信息，请参阅 [CHANGELOG.md](CHANGELOG.md)。

## 简介

Faiss 包含多种用于相似性检索的方法。它假定样本实例表示为向量并由整数唯一标识，且向量之间可通过 L2（欧氏）距离或点积进行比对。与查询向量相似的向量是指与该查询向量具有最小 L2 距离或最大点积的向量。Faiss 同时支持余弦相似度，因为其等价于归一化向量上的点积。

部分检索方法（如基于二值向量和紧凑量化编码的方法）仅使用向量的压缩表示，而无需保留原始向量。这通常会带来一定的检索精度损失，但使得这些方法能够在单台服务器的主存中扩展至数十亿级向量规模。另一些方法（如 HNSW 和 NSG）则在原始向量之上构建索引结构，从而显著提高检索效率。

GPU 实现既可以接收来自 CPU 内存的输入，也可以接收来自 GPU 显存的输入。在配备 GPU 的服务器上，GPU 索引可以直接替换 CPU 索引（例如，将 `IndexFlatL2` 替换为 `GpuIndexFlatL2`），并在 CPU 内存与 GPU 显存之间自动处理数据拷贝。若输入和输出均常驻于 GPU 显存中，检索速度将进一步提升。Faiss 同时支持单 GPU 和多 GPU 并行使用。

## 安装

Faiss 为 Python 提供了适用于 Anaconda 的预编译库，详见 [faiss-cpu](https://anaconda.org/pytorch/faiss-cpu)、[faiss-gpu](https://anaconda.org/pytorch/faiss-gpu) 以及 [faiss-gpu-cuvs](https://anaconda.org/pytorch/faiss-gpu-cuvs)。该库主要基于 C++ 实现，唯一的外部依赖是 [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) 实现。可选的 GPU 支持通过 CUDA 或 AMD ROCm 提供，Python 接口同样为可选组件。此外，还可以按需启用 NVIDIA [cuVS](https://github.com/rapidsai/cuvs) 后端 GPU 实现。项目采用 CMake 构建，详情请参阅 [INSTALL.md](INSTALL.md)。

## Faiss 的工作原理

Faiss 围绕着“索引类型”构建，该类型用于存储一组向量，并提供利用 L2 和/或点积比对在其中进行搜索的函数。部分索引类型属于基础基准方案，例如精确搜索。大多数可用的索引结构分别在以下维度权衡取舍：

- 检索耗时（search time）
- 检索质量（search quality）
- 每个索引向量占用的内存（memory used per index vector）
- 训练耗时（training time）
- 向量添加耗时（adding time）
- 无监督训练对外部数据的依赖程度（need for external data for unsupervised training）

可选的 GPU 实现提供了（截至 2017 年 3 月）高维向量已知最快的精确与近似（压缩域）近邻搜索实现、最快的 Lloyd k-means 聚类以及最快的小 k 选择算法。[具体实现细节详见此处论文](https://arxiv.org/abs/1702.08734)。

## 完整文档

以下是查阅 Faiss 文档的入口：

- 完整文档可查阅 [Wiki 页面](https://github.com/facebookresearch/faiss/wiki)，包含[新手入门教程](https://github.com/facebookresearch/faiss/wiki/Getting-started)、[常见问题解答 (FAQ)](https://github.com/facebookresearch/faiss/wiki/FAQ) 以及[疑难排查指南](https://github.com/facebookresearch/faiss/wiki/Troubleshooting)
- [Doxygen 文档](https://faiss.ai/) 提供了从源代码注释提取的各类别详细信息
- 如需复现我们研究论文中的实验结果：关于 [Polysemous codes](https://arxiv.org/abs/1609.01882) 与 [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)，请参阅 [基准测试 README](benchs/README.md)；关于 [Link and code: Fast indexing with graphs and compact regression codes](https://arxiv.org/abs/1804.09996)，请参阅 [link_and_code README](benchs/link_and_code)

## 作者

Faiss 的主要作者包括：
- [Hervé Jégou](https://github.com/jegou)：Faiss 项目的发起者并编写了最初的实现版本
- [Matthijs Douze](https://github.com/mdouze)：实现了 Faiss CPU 版本的大部分代码
- [Jeff Johnson](https://github.com/wickedfoo)：实现了 Faiss GPU 版本的全部代码
- [Lucas Hosseini](https://github.com/beauby)：实现了二值索引与构建系统
- [Chengqi Deng](https://github.com/KinglittleQ)：实现了 NSG、NNdescent 以及大量加性量化代码
- [Alexandr Guzhva](https://github.com/alexanderguzhva)：完成了诸多优化工作，包括 SIMD 指令优化、内存分配与布局优化、向量编解码器的快速解码核函数等
- [Gergely Szilvasy](https://github.com/algoriddle)：构建系统与基准测试评估框架

## 引用参考

在学术研究论文中使用 Faiss 时，请引用以下文献：
```
@article{douze2024faiss,
      title={The Faiss library},
      author={Matthijs Douze and Alexandr Guzhva and Chengqi Deng and Jeff Johnson and Gergely Szilvasy and Pierre-Emmanuel Mazaré and Maria Lomeli and Lucas Hosseini and Hervé Jégou},
      year={2024},
      eprint={2401.08281},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
对于 Faiss 的 GPU 版本，请引用：
```
@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```

## 加入 Faiss 社区

有关 Faiss 的公开讨论或疑问咨询，欢迎访问 https://github.com/facebookresearch/faiss/discussions 。

我们持续关注本仓库的 [Issues 页面](https://github.com/facebookresearch/faiss/issues)。
欢迎在上面提交缺陷报告、咨询技术问题等。

## 许可声明

Faiss 采用 MIT 许可证，详情请参阅仓库根目录下的 [LICENSE 文件](https://github.com/facebookresearch/faiss/blob/main/LICENSE)。

Copyright © Meta Platforms, Inc.
