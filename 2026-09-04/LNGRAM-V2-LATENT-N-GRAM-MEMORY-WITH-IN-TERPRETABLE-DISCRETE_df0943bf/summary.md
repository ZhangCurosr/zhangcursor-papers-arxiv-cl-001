---
title: "LNGRAM-V2-LATENT-N-GRAM-MEMORY-WITH-IN-TERPRETABLE-DISCRETE"
source: https://arxiv.org/pdf/2609.03426v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:06:13"
field: "高效记忆机制与大模型可解释性"
keywords: ["latent n-gram memory", "conditional memory", "interpretable discrete representations", "GQA readout", "surrogate gradient", "vision-language models", "model interpretability"]
innovations: ["解耦路由数、内存维度与骨干宽度的可扩展条件记忆设计", "反事实代理梯度在硬离散寻址中的稳定训练方法", "仅凭离散 ID 即可保留并读出连续隐状态大部分语义的机制"]
benchmarks: ["OCRBench", "MathVista", "MM-Vet", "HallusionBench", "MM-Star", "MMMU", "AI2D", "MMBench-EN", "MMBench-CN", "Benchmark V2.1"]
---

# 论文速读：LNGRAM-V2-LATENT-N-GRAM-MEMORY-WITH-IN-TERPRETABLE-DISCRETE

## 一句话总结
Lngram v2 提出一种可解释的离散隐式条件记忆机制，解耦路由数、内存维度与骨干网络宽度，通过上下文感知 GQA 读取与反事实代理梯度实现高效可扩展的 n-gram 精确查表，并在 2B 到 30B 参数视觉语言模型上持续取得性能提升。

## 研究问题与动机
- Transformer 缺乏原生查找机制，对局部静态模式的重用依赖密集计算，消耗原本可用于组合推理的参数容量。
- Lngram v1 的内存路由数与骨干网络宽度耦合，导致内存规模与参数量、激活开销随模型维度同步增长，限制可扩展性。
- 离散 ID 是否保留连续隐状态的语义结构尚不明确；若保留，则可同时作为记忆寻址与内部表示分析的离散接口。
- 现有条件记忆方法依赖分词器 ID 与固定哈希，对 tokenization 边界敏感且难以统一推广到视觉等多模态输入。

## 核心贡献（创新点）
- 解耦设计：将路由数 R、每路由比特数 M 与骨干网络宽度 d 解耦，使内存容量可独立扩展而不再受限于骨干维度。
- 上下文感知 GQA 读取：将不同 n-gram 阶次检索出的记忆组织为路由级 memory tokens，并用当前隐状态进行分组查询注意力选择，支持可靠检索的自适应抑制。
- 零值 Sink 与反事实代理梯度：引入固定零值 Sink 降低不可靠检索的概率质量，并通过反事实内存查表提供保留硬离散寻址的优化信号。
- 语义可读性验证：证明离散 ID 自身即可恢复连续隐状态 65.77%–84.27% 的超额语义读出能力，并提供可复现的路线–概念关联。
- 可扩展至大模型：在 Keye2B 与 Keye30B 上一致提升，且相比 v1 在 R=16 配置下将模块总参数降低 82.6%、每 token 激活参数降低 95.2%。

## 方法详解
- 多路由离散寻址：对层输出 H 做 RMSNorm 后投影得到 Z=UW_q，按位阈值得到比特 b_{t,r,j}=I[z_{t,r,j}>0]，打包为离散符号 a_{t,r}∈{0,…,K−1}，其中 K=2^M，共 R 条并行路由。
- 精确 n-gram 检索：为每个阶次 n 维护表 E^{(n)}∈R^{RK^n×d_m}，按路由与历史符号构造地址 g_{t,r}^{(n)} 并直接查表 m_{t,r}^{(n)}=E^{(n)}[g_{t,r}^{(n)}]；无效前缀与跨边界 n-gram 被掩码。
- 多阶拼接为 memory tokens：x_{t,r}=Concat_{n∈N} m_{t,r}^{(n)}，每个位置产生 R 个 memory tokens。
- 上下文感知 GQA 读取：使用当前 h_t 生成 Q_t，对 memory tokens 生成 K_{t,r},V_{t,r}，按 head 分组共享 KV；引入固定 logit b_0 的零值 Sink，使 α_{t,a,r}=exp(s_{t,a,r})/(exp(b_0)+Σ exp(s_{t,a,r'}))，从而允许模型抑制不可靠记忆。
- 输出投影与因果卷积精炼：拼接多头输出后接深度可分离因果卷积（kernel=4），再以残差注入骨干 H^{(ℓ)}←H^{(ℓ)}+Y。
- 反事实代理梯度：前向保持硬离散寻址；反向时对单个符号枚举反事实检索并计算条件期望梯度，或在大规模时采用一位翻转近似 ∂L/∂z_j≈λτp_j(1−p_j)⟨g,E_j^{(1)}−E_j^{(0)}⟩，使路由学习兼顾地址切换引起的内存内容变化。

## 实验与结果
- 数据集/基准：OCRBench、MathVista、MM-Vet、HallusionBench、MM-Star、MMMU、AI2D、MMBench-EN/_CN、Benchmark V2.1。
- 模型规模：Keye2B 与 Keye30B，基线在同一数据与优化设置下继续训练约 5B tokens；Lngram v2 插入浅层与中间层，使用 4-bit 代码与 2/3-gram 内存。
- Keye2B 主要结果：平均得分从 47.25±0.15 提升至 47.93±0.18（均方增益 +0.68）；PKM 基线仅得 47.24，参数匹配的 Sparse FFN 仅得 45.86，说明增益不可归因于参数或稀疏容量本身。
- Keye30B 主要结果：十基准均值从 78.23 提升至 79.71（+1.48），配对 95% CI 为 [0.86, 2.11]；AI2D 与 MMBench-EN 经 Holm 校正显著，MMBench-CN 经 BH 校正显著。
- 效率对比：在 r64/m4/KV8 配置下，序列长度 1024 时 prefill 延迟增加约 6.1%，decode 延迟增加约 8.0%，显存增量有限。
- v1 对比：R=16 配置下 Lngram v2 总参数较 v1 减少 82.6%，每 token 激活参数减少 95.2%，验证损失仍略优于 v1（2.8584 vs 2.8609）。
- 语义读出：Layer 1 保留 82.21%–84.27% 超额 AP，Layer 13 保留 65.77%–73.84%；9220 条可复现关联，局部存在率复现率高达 83%–85%。

## 相关工作脉络
- Conditional Memory / Engram：基于分词器 ID 的条件记忆，依赖词汇边界与固定哈希，难以统一到视觉等多模态；Lngram 使用隐空间离散符号实现跨模态通用寻址。
- Product Key Memory (PKM)：参数匹配的强基线之一，但在相同预算下表现不及 Lngram v2，说明精确 n-gram 结构与语义保留的重要性。
- Sparse FFN / MoE：参数匹配的稀疏前馈作为控制组，同样无法复现 Lngram v2 的增益，排除了“仅靠稀疏容量”的解释。
- N-gram 语言建模：传统 Brants 等的大规模 n-gram 统计建模被扩展到神经网络隐表示空间，实现可学习的条件记忆。
- 表征可解释性/探针：Alain & Bengio、Huben 等依赖连续激活与学习字典；本文证明仅凭离散 ID 即可重建大部分可读语义，提供结构化接口。
- GQA 与高效注意力：Ainslie 等的分组查询注意力被引入记忆读取，降低 KV 头数量同时保持多查询能力，支持独立扩展路由与内存维度。

## 局限性与未来方向
- 对骨干表征稳定性的依赖：在跨模态适配阶段表征仍在变化时，离散地址不稳定会降低记忆形成效率，建议在中后期插入而非从头训练。
- 超参敏感性与容量权衡：极大路由与内存维度带来显著长序列显存开销与收益递减，需在实践中平衡性能与部署成本。
- 语义读出仍非完全恢复：离散 ID 仅保留连续表示的 65%–84% 超额语义，部分信息在量化过程中丢失。
- 单步/一比特近似可能损失梯度精度：大规模训练采用一位反事实近似，与精确枚举相比存在近似误差。
- 当前主要在 VLM 上验证，更通用的预训练与下游微调策略仍需进一步检验。

## 研究启发与可借鉴点
- 解耦设计思路：将路由数、内存维度和骨干宽度独立控制，可在不改变主干的前提下按需扩展记忆容量，适合多尺度模型迭代。
- 零值 Sink 的可迁移性：为条件记忆/查表模块引入固定零值选项，可有效抑制噪声检索，其他查表型模块可直接借鉴。
- 反事实代理梯度的训练技巧：将反向信号与候选地址对应的内存内容变化对齐，比纯 STE 更稳定；可推广到其它硬离散路由场景。
- 仅凭离散 ID 做语义读出的分析范式：无需访问连续激活即可评估表征结构，可成为模型解释与诊断的新工具。
- 实验对比策略：参数匹配 Sparse FFN 与 PKM 双基线共同排除“参数/稀疏容量”解释，方法论上可作为同类工作的参考模板。

## 关键术语表
**Lngram v2**：隐空间离散 n-gram 条件记忆模块的第二版，解耦路由与骨干维度并支持多模态精确查表。
**GQA（Grouped-Query Attention）**：多查询头共享少量 KV 头的注意力变体，用于高效记忆读取。
**Zero-value Sink**：注意力分布中引入的固定零值槽位，允许模型将概率质量分配给“无记忆”选项。
**Counterfactual Surrogate Gradient**：在反向传播中基于候选离散地址的反事实检索结果估算路由梯度的代理方法。
**Discrete Route Code**：每条路由上的 K=2^M 个离散符号之一，用作内存地址的一部分。
**ID-KNN Semantic Reader**：仅依赖离散 ID 的汉明距离最近邻读出器，用于评估离散表征的语义保留程度。
**Exact N-gram Retrieval**：按路由与历史符号直接构造整数地址并进行查表，避免哈希碰撞与近似搜索。
**Active Parameters per Token**：每个 token 在推理/训练时实际参与计算的模块参数规模。

## 可复现要素
- 数据集/模型：Keye2B、Keye30B、COCO 2017、FineWeb-Edu；评测基准均为公开基准。论文未声明单独开源数据集。
- 代码/权重：论文未明确声明代码与模型权重是否开源。
- 关键超参：Keye2B 使用 R=121、M=4、KV heads=16、n-gram=2/3、memory dim=128；Keye30B 使用 R=256、M=4、KV heads=16、memory dim=256；Sink 初始有效路由质量 μ_0=0.5；短卷积 kernel=4、dilation=3、零初始化输出投影；学习率与 weight decay 见论文 Table 13。
- 训练设置：Keye30B 使用 64 张 H800，TP=1、PP=4、CP=16、EP=8，序列长度 32768，BF16，全局 batch=32；Keye2B 配置见 Appendix E。
