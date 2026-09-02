---
title: "DiaRelay-Relaying-Dialogue-Context-with-a-Constant-Size-Memo"
source: https://arxiv.org/pdf/2608.22745v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:48:55"
field: "对话情绪识别"
keywords: ["Emotion Recognition in Conversation", "Large Language Models", "Parameter-Efficient Fine-Tuning", "Memory-Augmented Modeling", "Dialogue Context", "Low-Rank Adaptation"]
innovations: ["提出恒定大小中继记忆替代无限增长序列历史，实现跨话语情感线索持久传播", "设计误差修正写入与双轴记忆读取机制，使低秩适配具备上下文条件化能力", "建立先读后写因果范式，在不扩展上下文窗口的前提下突破局部窗口限制"]
benchmarks: ["IEMOCAP", "MELD"]
---

# 论文速读：DiaRelay-Relaying-Dialogue-Context-with-a-Constant-Size-Memo

## 一句话总结
论文提出了 **DiaRelay**，一种轻量级记忆增强适配器，使大语言模型能够在固定上下文窗口之外，通过恒定大小的对话级中继记忆持续传递历史情感线索，从而在对话情绪识别（ERC）任务中实现更准确的情绪预测。

## 研究问题与动机
- **固定上下文窗口的局限性**：现有方法将当前话语与固定数量的前序话语拼接作为输入，窗口过短会丢弃潜在的长距离情感证据，窗口过大则重复编码重叠话语、增加计算/内存开销，并可能引入无关上下文。
- **LoRA 等 PEFT 方法无法维护对话状态**：LoRA 引入的低秩变换在特征空间中是固定的，跨话语共享，不显式维护对话级别的状态，也无法根据不断演化的对话上下文条件化其变换。
- **对话历史信息的持久性缺失**：当某话语离开显式上下文窗口后，标准 PEFT 技术无法保留该话语的前置情感线索，也无法将后续变换条件化于对话历史。
- **需要零样本推理效率**：希望避免测试时的梯度更新，同时保持低秩适配的参数效率优势。

## 核心贡献（创新点）
1. **提出 DiaRelay 适配器**：在 LoRA 基础上，为每个适配 Transformer 层引入独立的恒定大小中继记忆，使历史信息能够跨话语传播，无需扩展输入上下文长度或进行测试时参数更新。
2. **设计选择性中继记忆转换（SRMT）**：通过门控误差修正操作，将当前话语表征选择性地融入有界中继记忆，使记忆在对话过程中持续演化而不增加存储开销；利用残差写信号（当前值减去记忆已恢复的部分）避免冗余累积。
3. **设计双轴中继记忆读取（DaRMR）**：从传播的记忆中沿两个方向检索信息，分别生成 query 侧和 output 侧的低秩修正，实现上下文依赖的特征自适应，而非静态低秩变换。
4. **建立"先读后写"范式**：预测第 t 个话语时仅访问前 t-1 轮的记忆，预测完成后再将当前话语写入记忆，防止当前信息泄露到自身预测中。
5. **在 MELD 上达到 SOTA weighted F1**：仅增加约 7.1M 可训练参数（约 Qwen3-8B 的 0.09%），在 IEMOCAP 和 MELD 上均取得竞争力或最优结果。

## 方法详解
**问题设定**：给定包含 T 个话语的对话 $\mathcal{D} = \{(x_t, s_t)\}_{t=1}^{T}$，目标是根据对话上下文 $\mathcal{C}_t$ 预测每个话语 $x_t$ 的情绪标签 $y_t$。

**整体架构**：
- 在 LoRA 基础上，为每个适配的 Transformer 层 $\ell$ 配备独立的中继记忆 $\mathbf{R}_t^{(\ell)} \in \mathbb{R}^{r \times r}$，初始化为零矩阵。
- 目标话语的层级别表征 $\mathbf{u}_t^{(\ell)}$ 通过对目标话语 token 跨度内的隐藏状态进行平均池化得到。

**Selective Relay Memory Transition (SRMT)**：
1. **中继空间投影**：将高维话语表征 $\mathbf{u}_t$ 投影到低维中继向量：
   - $\mathbf{q}_t^m = \text{Norm}(\tanh(\mathbf{W}_q^m \mathbf{u}_t))$
   - $\mathbf{k}_t^m = \text{Norm}(\tanh(\mathbf{W}_k^m \mathbf{u}_t))$
   - $\mathbf{v}_t^m = \mathbf{W}_v^m \mathbf{u}_t$
2. **选择性记忆中继**：通过门控误差修正操作更新记忆：
   - 更新门：$\boldsymbol{\beta}_t = \sigma(\mathbf{W}_\beta \mathbf{u}_t + \mathbf{b}_\beta)$
   - 保留门：$\boldsymbol{\lambda}_t = 1 - \boldsymbol{\beta}_t$
   - 误差修正信号：$\mathbf{e}_t^m = \mathbf{v}_t^m - \mathbf{R}_{t-1} \mathbf{k}_t^m$
   - 记忆更新：$\mathbf{R}_t = \text{Diag}(\boldsymbol{\lambda}_t) \mathbf{R}_{t-1} + \text{Diag}(\boldsymbol{\beta}_t) \mathbf{e}_t^m (\mathbf{k}_t^m)^\top$

**Dual-axis Relay Memory Read (DaRMR)**：
1. **双轴记忆读取**：
   - Query 侧修正：$\Delta \mathbf{q}_t^m = \frac{\alpha_q}{r} \mathbf{P}_q \mathbf{R}_{t-1}^\top \text{Norm}(\tanh(\mathbf{v}_t^m))$
   - Output 侧修正：$\Delta \mathbf{o}_t^m = \frac{\alpha_o}{r} \mathbf{P}_o \mathbf{R}_{t-1} \mathbf{q}_t^m$
2. **记忆条件化注意力修正**：
   - 修正后 Query：$\tilde{\mathbf{q}}_{t,i} = \mathbf{q}_{t,i}^0 + \Delta \mathbf{q}_{t,i}^L + \Delta \mathbf{q}_t^m$
   - 修正后 Output：$\tilde{\mathbf{o}}_{t,i} = \mathbf{o}_{t,i}^0 + \Delta \mathbf{o}_{t,i}^L + \Delta \mathbf{o}_t^m$
3. 修正仅应用于目标话语 token 位置，显式历史上下文的表征保持不变。

**学习目标**：保留 LLM 的自回归负对数似然损失：
$$\mathcal{L}_{\text{ERC}} = -\sum_{t=1}^{T} \sum_{j=1}^{M_t} \log p_\theta(y_{t,j} | y_{t,<j}, \mathcal{C}_t, \{\mathbf{R}_{t-1}^{(\ell)}\}_{\ell \in \mathcal{L}_D})$$

**"先读后写"流程**：
1. 初始化所有适配层记忆为零
2. 构建显式上下文（当前话语 + 最多 4 个前序话语）
3. DaRMR 读取前一时刻记忆，生成 query/output 修正
4. Backbone 预测当前话语情绪
5. 平均池化目标话语表征并投影到中继空间
6. SRMT 更新记忆
7. 将更新后的记忆传递给下一个话语预测；对话结束时重置记忆

## 实验与结果
**数据集**：
- **IEMOCAP**：对话式情感识别数据集，训练+验证 5,810 话语/120 对话，测试 1,623 话语/31 对话
- **MELD**：多说话人对话数据集（源自《老友记》），训练+验证 11,098 话语/1,152 对话，测试 2,610 话语/280 对话

**评估指标**：主要使用加权 F1（W-F1）和准确率（Acc），消融实验中额外报告 Macro F1（M-F1）。

**基线方法**：包括传统神经网络方法（DialogueRNN、DialogueGCN）、预训练语言模型方法（COSMIC）、LLM 基线方法（MMLA、InstructERC、BiosERC、MSG-LLM、LaERC-S、SpeechCueLLM、Causal-ERC、PRC-Emo）以及多模态基线（MSE-Adapter）。

**主要结果**：
- **MELD（Qwen3-8B）**：W-F1 = **70.06%**，Acc = **71.15%**，达到 SOTA
- **IEMOCAP（Qwen3-8B）**：W-F1 = **70.01%**，Acc = **69.93%**，具有竞争力
- **相比 LoRA-only 提升**：MELD 上 W-F1 提升 3.05%，IEMOCAP 上提升 2.58%
- **相比更小基座模型最佳结果**：在 ≤6B 参数量级的 backbone 上，IEMOCAP 提升 1.80%/2.09%（W-F1/Acc），MELD 提升 0.32/1.33

**消融实验关键发现**：
- 完整 DiaRelay 相比 LoRA-only 在 MELD 上 W-F1 提升 3.05%，M-F1 提升 4.84%，Acc 提升 2.41%
- 对话级记忆传播（Full vs Window-local）带来 W-F1 提升 1.11%
- 误差修正更新策略相比完整值更新带来 M-F1 提升 2.37%
- 双轴读取的两个修正路径均有益，output-only 独立效果优于 query-only
- 交叉话语梯度跨度 K=2 达到最优，相比 K=1 提升 1.24% W-F1
- 中继维度 r=8 达到最优，相比 r=4 提升 1.59% W-F1

**案例研究**：展示了 DiaRelay 能够利用超出显式四话语窗口的历史线索（如 u_{t-6} 和 u_{t-12}）正确预测情绪，而 LoRA 和 Window-local 变体则预测错误。

## 相关工作脉络
1. **传统 ERC 方法**（DialogueRNN、DialogueGCN、DialogueCRN）：依赖任务特定的循环推理或显式构建话语级图结构，不适用于大语言模型的参数高效适配。
2. **记忆机制相关 ERC 方法**（DialogXL、CoMPM）：DialogXL 依赖 XLNet 的内部循环架构，难以迁移到通用 LoRA 适配的 LLM；CoMPM 需要额外的预训练记忆提取路径，而 DiaRelay 维护在线有界中继记忆。
3. **LLM-based ERC 方法**（InstructERC、LaERC-S、CoE、PRC-Emo）：通过更丰富的输入构造、额外知识、辅助监督或多阶段训练提升性能，但仍未显式维护持续演化的紧凑对话状态，且引入额外数据构建或训练复杂性。
4. **参数高效微调方法**（LoRA 及其变体）：引入可训练的低秩变换但跨话语共享，不显式携带累积对话信息；DiaRelay 通过传播有界中继记忆并使用其条件化低秩变换，正交补充了此局限。
5. **快速权重编程**（Schlag et al., 2021）：Delta 规则启发了 SRMT 的误差修正写入信号设计。
6. **多模态情感分析适配器**（MSE-Adapter）：处理 TAV 多模态输入并使用未来/完整对话信息，而 DiaRelay 聚焦文本 ERC 的历史-only 设置。

## 局限性与未来方向
- **仅评估文本模态**：当前框架主要在纯文本 ERC 设置下验证，对多模态对话理解及更通用对话任务的适用性有待探索。
- **缺乏可解释性**：当前框架专注于分类性能，未显式识别与每次预测相关的历史证据，也未提供预测情绪的自然语言解释。
- **IEMOCAP 训练规模限制**：IEMOCAP 训练数据规模较小，可能不足以充分优化新引入的基于注意力的记忆交互，导致在该数据集上提升幅度相对有限。
- **未来方向**：扩展到多模态和通用对话设置；集成解释生成机制，联合产生情绪预测和基于中继对话记忆的忠实、人类可读的推理依据。

## 研究启发与可借鉴点
1. **恒定大小记忆用于序列建模**：通过有界中继记忆替代无限增长的序列历史，可在不增加显式上下文长度的前提下保留长距离依赖，这一设计可迁移到长对话摘要、多轮对话管理等任务。
2. **误差修正写入策略**：利用残差信号（当前值减去记忆已恢复的部分）进行选择性更新，避免冗余累积，该思想可用于设计更高效的记忆网络写入机制。
3. **双轴记忆读取设计**：同时从 query 侧（重加权可见证据）和 output 侧（直接补充历史信息）进行记忆条件化，形成互补机制，可借鉴到其他需要上下文感知的适配器设计中。
4. **"先读后写"因果保证**：在序列预测中严格执行先读取前序记忆再写入当前信息的顺序，确保无信息泄露，这一范式可推广到其他需要因果性保证的记忆增强 LLM 应用。
5. **与 LoRA 的正交结合**：在冻结的 backbone 和低秩适配之上叠加对话级状态记忆，参数开销极小（仅 0.09%），为其他 PEFT 方法的扩展提供了参考框架。

## 关键术语表
**Emotion Recognition in Conversation (ERC)**：对话情绪识别，任务目标是根据对话上下文识别每个话语表达的情绪类别。
**Parameter-Efficient Fine-Tuning (PEFT)**：参数高效微调，通过少量 trainable 参数适配预训练模型，避免全参数微调的计算开销。
**Low-Rank Adaptation (LoRA)**：低秩适配，通过低秩矩阵分解引入可训练参数，冻结原始权重以实现高效微调。
**Selective Relay Memory Transition (SRMT)**：选择性中继记忆转换，通过门控误差修正操作将当前话语信息选择性融入有界中继记忆的组件。
**Dual-axis Relay Memory Read (DaRMR)**：双轴中继记忆读取，从传播的记忆中沿 query 侧和 output 侧两个方向检索信息以修正自注意力的组件。
**Read-before-Write Paradigm**：先读后写范式，预测当前话语时仅访问前序记忆，预测完成后再将当前信息写入，防止信息泄露。
**Error-Corrective Update**：误差修正更新，利用当前值与记忆已恢复值的残差作为写入信号，避免重复累积已有信息。
**Relay Memory**：中继记忆，恒定大小的对话级状态矩阵，跨话语传播以保留历史情感线索。

## 可复现要素
- **数据集**：IEMOCAP 和 MELD（公开数据集，官方分割）
- **代码**：论文未明确提及代码开源状态，需查阅 arXiv 页面
- **权重**：使用 Qwen3-4B 和 Qwen3-8B 作为 backbone（需从官方渠道获取）
- **关键超参**：
  - 显式上下文窗口：最多 4 个前序话语 + 当前目标话语
  - 中继维度 r：8
  - LoRA rank/alpha/dropout：32/128/0.05
  - LoRA 学习率：3×10⁻⁴
  - DiaRelay 学习率：2×10⁻⁴
  - 交叉话语梯度跨度 K：2
  - 中继缩放因子：16
  -  backbone 量化：4-bit NF4 + bfloat16 计算
  - 优化器：AdamW
  - 最大序列长度：2048 tokens
- **硬件**：单张 NVIDIA RTX 3090 GPU
