---
title: "SQLite-is-Enough-Lexical-Semantic-and-Hybrid-Search-with-scr"
source: https://arxiv.org/pdf/2608.24060v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:56:06"
---

# 论文速读：SQLite-is-Enough-Lexical-Semantic-and-Hybrid-Search-with-scr

## 一句话总结
论文提出了 scrydb，一个基于 SQLite 的轻量级 Python 库，将词法搜索（FTS5/BM25）、多精度语义搜索（float32/int8/二值化向量）与混合搜索（RRF 融合/重排序）封装进单个 SQLite 文件，实现了无需外部索引或专用向量数据库的单文件可复现检索流水线。

## 研究问题与动机
- 现有 IR 研究通常依赖多组件系统（分离的倒排索引、向量存储、配置文件），难以共享、归档与复现，导致实验资源碎片化。
- 专用向量数据库与 ANN 索引（如 Faiss、HNSW）虽能支撑大规模并发检索，但为中小型知识库部署全套重型架构属于过度设计，带来不必要的工程负担与计算能耗。
- 神经双编码器检索已成为主流，但全量 float32 向量比较成本高昂，社区呼吁在保障检索效果的同时探索更高效、低计算足迹的落地方案。
- SQLite 被美国国会图书馆列为推荐存档格式，具备单文件自包含与长期可归档特性；若能结合 FTS5 与 vec0 扩展，可自然满足“单文件复现”与“轻量化检索”的双重诉求。

## 核心贡献（创新点）
1. **单文件全栈检索流水线封装**：将 BM25 词法打分、多精度语义向量检索、第二阶段重排序与 RRF 融合统一集成于单一 SQLite 数据库，无需跨系统编排；与现有需组合多组件或依赖外部服务的检索框架本质不同，直接消解了索引同步与部署碎片化问题。
2. **三精度向量化存储与按需检索**：在同一库内并行维护 float32、int8 标量量化与二值化（1-bit）三种精度嵌入，支持任意精度间自由切换或级联重排；区别于仅支持单一精度或需独立向量库的方案，该设计让用户在存储体积（压缩 32 倍）与查询精度间做显式权衡。
3. **单文件学术复现与资源托管**：提供预构建的 scrydb 数据库（含 Qwen3-Embedding-8B 嵌入与最终检索 run 文件），使整个 IR 实验可作为单一数据集文件共享；与仅开源代码或 checkpoint 的工作相比，强调检索资源本身的长期可归档性与可继承性。
4. **二进制优先的性价比检索策略验证**：实验证明基于 Hamming 距离的二值化粗排已能将高优先级文档保留在 Top-1000 候选集中，后续用 int8/float 重排可达到与全量扫描相当的前 10 位效果且延迟大幅降低；这与追求极致精度的 ANN 贪心策略形成对比，重新定义了中小规模检索的成本-效果边界。

## 方法详解
- **词法搜索**：委托 SQLite 内置 FTS5 模块，利用倒排索引与 BM25 打分函数，通过 `MATCH` 运算符查询；同时暴露 `snippet()` 与 `highlight()` 辅助函数，无需应用层后处理即可返回关键词上下文片段。
- **语义搜索三精度机制**：
  - **全精度 float32**：直接计算余弦相似度 $\mathrm{sim}_{\mathrm{cos}}(\mathbf{e}_Q, \mathbf{e}_D) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_D}{\|\mathbf{e}_Q\| \|\mathbf{e}_D\|}$。
  - **二值化 (Binary)**：通过 Heaviside 阶跃函数逐分量截断 $b_i = H(e_i)$，将 $d$ 维向量压缩为 $d/8$ 字节的 bit vector，搜索使用汉明距离 $d_H(\mathbf{b}_Q, \mathbf{b}_D) = \mathrm{popcount}(\mathbf{b}_Q \oplus \mathbf{b}_D)$，存储体积缩小 32 倍。
  - **标量量化 int8**：对 L2 归一化后的嵌入进行仿射映射 $q_i = \mathrm{trunc}(255(e_i+1)/2 - 128)$，保留维度幅度信息，仍用余弦相似度打分，体积缩小 4 倍。
- **检索流水线设计**：`search()` 支持 `lexical` / `semantic` / `hybrid` 模式，`precision` 可选 `binary` / `int8` / `float`，`rerank` 可选在粗排 Top-K 候选上施加更高精度重排。混合搜索采用 Reciprocal Rank Fusion (RRF) 融合词法与语义排名。
- **索引结构**：每个集合按精度生成独立的 `vec0` 虚拟表，文档与查询文本、原始嵌入、二值化/量化嵌入共存于同一 SQLite 文件，所有变换均在 SQLite 内部完成，无需额外格式转换。

## 实验与结果
- **数据集**：BEIR 基准的 8 个公开检索数据集：ArguAna (8.67K)、FiQA (57K)、NFCorpus (3.6K)、Quora (523K)、SciDocs (25K)、SciFact (5K)、Touché (382K)、TREC-COVID (171K)。
- **评估基线**：MTEB 官方报告的 Qwen3-Embedding-8B 全精度 float32 检索结果；作者独立使用相同模型重算所有嵌入以排除 pipeline 差异。
- **主要结果**：
  - 所有二值化语义检索在 nDCG@10 上均超过 BM25，且多数差距显著，证明二值化向量保留了足够的语义判别信号。
  - **最强结果与提升**：`Hamming + cos_int8` 在 Fi
