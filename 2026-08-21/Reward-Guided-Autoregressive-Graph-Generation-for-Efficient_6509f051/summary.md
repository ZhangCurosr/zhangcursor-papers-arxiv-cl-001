---
title: "Reward-Guided-Autoregressive-Graph-Generation-for-Efficient"
source: https://arxiv.org/pdf/2608.20099v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:34:05"
---

# 论文速读：Reward-Guided-Autoregressive-Graph-Generation-for-Efficient

## 一句话总结
本文提出 RGA-Designer，将 RLHF 范式迁移至多智能体通信拓扑的自回归图生成任务，通过训练融合任务正确性与结构紧凑性的图级奖励模型，引导预训练生成器微调；在六个基准上保持 ARG-Designer 同等准确率的同时，平均降低 20.5% 的推理 Token 消耗。

## 研究问题与动机
- LLM 多智能体系统（MAS）通过角色分工与协作显著提升复杂推理性能，但随之带来极高的推理 Token 成本。
- 现有自回归图生成方法（如 ARG-Designer）仅通过最大化训练集拓扑的条件对数似然进行监督训练，缺乏鼓励生成稀疏、高效结构的显式动机。
- 静态模板剪枝类方法（如 Agent-Prune、AgentDropout）受限于预定义结构起点，无法生成训练集中未见过的全新拓扑。
- 亟需一种既能兼顾任务完成质量、又能显式优化通信结构紧凑性，且支持零样本泛化新角色的自动化拓扑设计框架。

## 核心贡献（创新点）
- **首次将奖励引导训练引入自回归 MAS 拓扑生成**：突破最大似然目标的局限，为图生成模型提供明确的稀疏性激励。
- **设计图级奖励模型联合量化任务正确性与结构紧凑性**：利用可编程验证的 MAS 执行结果替代人工偏好标注，契合 RLVR 范式。
- **实现跨基准奖励模型复用**：基于 embedding 空间的打分机制使单一全局奖励模型可直接迁移至新数据集与新角色，无需扩展生成器的角色分类头。
- **显著降低推理开销**：在六个基准上平均减少 20.5% 的 Token 消耗，并通过 Best-of-N 选择策略进一步优化精度-效率权衡。

## 方法详解
- **整体 pipeline**：继承 ARG-Designer 的预训练自回归图生成器，依次经历数据集收集、奖励模型训练、策略优化（GRPO）三个阶段；推理时采用 Best-of-N 机制筛选最终拓扑。
- **数据集构建与奖励计算**：使用不同采样温度从预训练生成器采样候选图，在 MAS 中执行并记录 pass/fail；合并 ARG 训练阶段的冷启动与精调数据。ground-truth 奖励由三段加权构成：$r = 0.6c + 0.3\cdot\text{size\_norm} + 0.1\cdot\text{edge\_norm}$，任务正确性权重最高，确保稀疏性不以牺牲正确性为代价。按奖励差值 $\delta=0.05$ 过滤构建偏好对。
- **奖励模型架构**：采用 2 层 GraphSAGE（含残差连接与层归一化）编码器。节点特征拼接角色 embedding、任务 query embedding（均来自 `all-MiniLM-L6-v2`，维度 384）及 5 维结构特征（图大小、边数、生成顺序、入度、出度）。全局均值池化后接两层 MLP 输出标量奖励。使用 Bradley-Terry 成对排名损失训练，pass/pass 对的权重设为 $\lambda_{p,p}=0.1$ 以弱化冗余信号。
- **策略优化（GRPO）**：采用组相对策略优化，组内归一化计算优势值 $\hat{A}_i = (\hat{r}_i - \mu_r)/\max(\sigma_r, 0.01)$。目标函数为 $\mathcal{L}_\pi = -\hat{A}_i \log \pi_\theta(\mathcal{G}|q) + \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$，KL 系数 $\beta=0.2$，每组采样 $G=4$ 个图；因每次更新均重采样，重要性比率恒为 1，省略裁剪项。
- **推理 Best-of-N 筛选**：拓扑设计器独立采样 $N=5$ 个候选图并由奖励模型打分，选取最高分作为最终拓扑，开销远低于额外 LLM 调用。

## 实验与结果
- **数据集与基线**：GSM8K、AQuA、MultiArith、SVAMP、HumanEval、MMLU；基线包括 Vanilla、G-Designer、AgentPrune、AgentDropout、ARG-Designer。底层 LLM 为 Qwen3-4B，单卡 V100 运行，每项实验重复 10 次取均值。
- **准确率表现**：RGA-Designer 平均准确率达 87.80%，与 ARG-Designer（87.55%）及其他 MAS 基线相比无统计学显著下降（Welch's t-test $p>0.05$），仅在 GSM8K 和 HumanEval 上有微小提升。
- **Token 消耗**：平均 Token 使用量降至 3032，较 ARG-Designer（3815）减少约 20.5%；除 MultiArith（模板化题目冗余空间小，$p=0.673$）外，其余 5 个基准的 Token 缩减均具统计学显著性（$p<0.001$ 或 $p<0.05$）。
- **消融实验**：移除奖励模型（w/o RM）或 Best-of-N（w/o BoN）均会导致 Token 消耗上升或精度波动；完整方法在 GSM8K 和 HumanEval 上同时实现最低 Token 与最高/持平精度，MMLU 上 w/o BoN 精度略高但 Token 多 30.5%。

## 相关工作脉络
- **静态/模板化拓扑设计**：早期 MAS 依赖链式、星型等固定结构；Agent-Prune、AgentDropout 从预定义模板出发进行边/节点剪枝，无法生成未见过结构；G-Designer 虽
