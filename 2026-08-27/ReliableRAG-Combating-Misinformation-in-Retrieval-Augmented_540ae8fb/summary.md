---
title: "ReliableRAG-Combating-Misinformation-in-Retrieval-Augmented"
source: https://arxiv.org/pdf/2608.25487v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:42:34"
field: "检索增强生成与多跳推理"
keywords: ["Retrieval-Augmented Generation", "Multi-hop QA", "Misinformation Robustness", "Reasoning Chains", "Triple Extraction", "Reliability Quantification"]
innovations: ["首个在三元组粒度进行细粒度可靠性感知的RAG框架，结合语义相关性与可信度双因子量化", "动态构建推理链过程中每链独立查询+束搜索+置信度累积，有效防止欺骗性信息传播"]
benchmarks: ["HotPotQA", "2WikiMultiHopQA", "MuSiQue"]
---

# 论文速读：ReliableRAG-Combating-Misinformation-in-Retrieval-Augmented

## 一句话总结
论文提出 ReliableRAG，首个基于可靠性的 RAG 框架，通过在**三元组粒度**上动态量化语义相关性+可信度，过滤欺骗性错误信息，并自回归构建健壮的多跳推理链，显著提升 RAG 系统在虚假信息干扰下的事实可靠性。

---

## 研究问题与动机

- **多跳 QA 中欺骗性信息的传播问题**：现有 RAG 在多跳问答中，即使少量含欺骗性错误信息的段落也能通过推理链级联传播，导致整个推理过程被污染，最终生成错误答案。
- **细粒度可靠性感知缺失**：现有显式调节方法（如文档级可信度、知识图谱结构）无法识别"语义上高度相关但事实上不正确"的细粒度欺骗性三元组，易被误导。
- **隐式对齐方法的局限**：训练型方法（Self-RAG、CAG、Knowledge-R1）依赖大量高质量训练数据和计算资源，且存在领域泛化差距，难以适应现实场景。
- **长文本中的中间丢失现象（Lost-in-the-middle）**：直接将粗粒度文档输入 LLM 时，关键证据常被忽略，多跳 QA 中证据分散在多文档，更易出现问题。

---

## 核心贡献（创新点）

1. **首次在三元组粒度进行细粒度可靠性感知**：通过双因子感知机制（查询-三元组语义相关性 + 三元组可信度）动态量化信息可靠性，本质区别于现有文档级或句子级的粗粒度方法。
2. **提出 Triple Evaluation 模块**：每个推理跳步中，为每条现有推理链动态构造唯一查询，检索并筛选 top-K 最可靠且非冗余的三元组，形成高可信搜索空间。
3. **提出可靠性引导的 Chain Construction 模块**：采用束搜索策略自回归构建多跳推理链，支持提前终止，并基于选择概率累积更新链置信度，有效防止欺骗性信息传播。
4. **无训练、免微调的即插即用框架**：ReliableRAG 不依赖任何模型微调，可直接集成到现有 RAG 系统中，增强其对抗欺骗性信息的能力。

---

## 方法详解

### 整体框架（三模块两阶段）

ReliableRAG 包含三个核心模块：**Triple Extraction（离线）**、**Triple Evaluation（在线）**、**Chain Construction（在线）**。

---

### 1. Triple Extraction（三元组提取，离线阶段）

- 将每篇文档按句号分句，利用 LLM 的 ICL 能力，结合文档标题和 3 个上下文示例，提取结构化三元组 $\mathcal{T} = \{(s_j, p_j, o_j)\}_{j=1}^{M}$。
- 三元组形式：$\langle s_k; p_k; o_k \rangle$，解决长文本中的 lost-in-the-middle 问题。

---

### 2. Triple Evaluation（三元组评估，在线阶段）

#### 2.1 链定义与查询构造
- 第 $i$ 跳维护推理链集合 $C^i = \{u_j^i\}$。
- 第 1 跳从问题 $Q$ 生成初始查询 $q^1$；后续跳 $i \in \{2,\dots,L\}$ 中，为每条已有链 $u_j^{i-1}$ 构造动态查询：$q_j^i = [u_j^{i-1} \oplus Q]$。

#### 2.2 细粒度可靠性量化（双因子感知机制）

**因子 1：语义相关性**（Bi-encoder 编码后余弦相似度）
$$
\Phi(h_j^i, h_k) = \frac{h_j^i \cdot h_k}{\|h_j^i\|\|h_k\|}
$$

**因子 2：三元组可信度 $\eta_k$**（LLM-based Evaluator 三步分析）：
- Step 1：实体可信度分析（subject/object 是否为真实实体）
- Step 2：关系可信度分析（predicate 描述的关系是否真实一致）
- Step 3：综合量化（输出 0–10 的可信度分数）

**细粒度可靠性评分**（平衡系数 $\alpha \in [0,1]$）：
$$
R_{j,k}^i = \alpha \cdot \Phi(h_j^i, h_k) + (1 - \alpha) \cdot \eta_k
$$

- 对已存在于推理链中的三元组赋予 $-\infty$ 惩罚，避免重复选择。
- 每个查询选取 top-K 最可靠三元组作为当前跳的搜索空间 $S_j^i$。

---

### 3. Chain Construction（推理链构建，在线阶段）

- 对每个查询 $q_j^i$ 构造推理 prompt $I_j^i = \Psi(\text{Instr}, \text{Exemp}, q_j^i, S_j^i)$，包含任务指令 + 3 个上下文示例 + 选项列表（A=终止，B/C/D...=可选三元组）。
- LLM-based Selector $M_{sel}$ 输出各选项概率，选取 top-B 高概率选项：
$$
\mathcal{E}_j^i = \text{Top-}B_{e_k}\big(P_{M_{sel}}(e_k \mid I_j^i)\big)
$$
- 根据选中选项分支推理链：若选 A 则终止，否则拼接对应三元组：
$$
u_m^i = \begin{cases} u_j^{i-1}, & e_k \text{为选项A或链已终止} \\ u_j^{i-1} \oplus t_k, & \text{否则} \end{cases}
$$
- 累积置信度：$\omega_m^i = \omega_j^{i-1} \cdot \rho_{j,k}^i$，保留 top-F 最置信链进入下一跳。
- 最大跳数 $L$ 后输出 $C_{final}$，通过 TBS 或 SBS 策略合成支撑上下文 $Z$，与 $Q$ 拼接后输入 Generator。

---

## 实验与结果

### 数据集
- **HotPotQA**（2–4 跳）、**2WikiMultiHopQA**（2–4 跳）、**MuSiQue**（4 跳），每问 10/10/20 篇 Wikipedia 文档，各采样 1000 测试题。
- 注入 LLM 生成的欺骗性低可信度文档（每问 1–3 篇）。

### 评估基线
- Fine-tuning：CAG-7B、Self-RAG-7B、Knowledge-R1-7B
- Non-fine-tuning：Naive LLM、Vanilla RAG、Prompt-Based、Exclusion、CrAM、TruthfulRAG

### 主要结果（Ideal Setting，Llama3-8B 生成器）

| 方法 | HotPotQA EM | 2WikiMultiHopQA EM | MuSiQue EM |
|------|-------------|---------------------|------------|
| Vanilla RAG | 20.10 | 14.50 | 10.40 |
| TruthfulRAG | 36.70 | 25.70 | 19.50 |
| CrAM | 42.70 | 27.30 | 20.40 |
| **ReliableRAG-TBS** | **52.20** | **45.80** | **30.30** |
| **ReliableRAG-SBS** | **53.20** | **43.10** | **31.60** |

- **ReliableRAG-TBS 最佳**：HotPotQA EM 55.1%、2WikiMultiHopQA EM 45.4%、MuSiQue EM 31.6%（含低可信文档设置）。
- 相比 Vanilla RAG 提升约 **35–40 个 EM 点**，显著优于所有对比方法。
- 在 Evaluator-Generated Setting 下同样表现最优。

### 鲁棒性分析
- 随低可信文档比例增加，ReliableRAG 性能下降最小；在 HotPotQA 上 ReliableRAG-SBS 甚至略有上升（证明能从低可信文档中挖掘正确三元组）。

### 消融实验
- **w/o Triple Extraction**：HotPotQA EM 下降 34.1%（最大影响）
- **w/o Triple Evaluation**：HotPotQA EM 下降 15.6%
- **w/o Chain Construction**：Top-25 Triples 时 EM 最高仅 47.0%，说明盲目堆叠三元组无效

### 超参敏感性
- 最优平衡系数 $\alpha^* = 0.4$（语义相关性权重 40%，可信度权重 60%）
- 最大推理深度 $L > 4$ 时性能趋于饱和甚至轻微下降

---

## 相关工作脉络

1. **Self-RAG（Asai et al., ICLR'24）**：通过反思 token 自适应判断检索必要性；但需微调，且依赖内部决策策略而非细粒度可靠性感知。
2. **CAG（Pan et al., EMNLP'24）**：将可信度编码进指令微调；存在领域泛化差距，且仅作用于文档级。
3. **Knowledge-R1（Lin et al., arxiv'25）**：RL 平衡参数知识与外部上下文；需大量训练数据，现实部署成本高。
4. **CrAM（Deng et al., AAAI'25）**：基于文档级可信度动态调整 attention 权重；无法感知细粒度欺骗性信息。
5. **TruthfulRAG（Liu et al., AAAI'26）**：基于知识图谱和熵过滤解决事实冲突；但三元组检索仅依赖语义相似度，无法区分语义相关但事实错误的欺骗性信息。
6. **TRACE（Fang et al., EMNLP'24）**：自回归构建推理链的方法；但未考虑信息可靠性，易受欺骗性信息误导。

---

## 局限性与未来方向

- **计算开销**：多跳推理 + 三元组评估 + LLM 调用链构建，推理延迟高于 Vanilla RAG；TBS 策略输入更短，SBS 策略保留更多语义但开销更大。
- **评估器依赖**：可信度评分依赖 LLM Evaluator（如 GLM-4-Flash），更高精度评估器可进一步提升性能，但成本增加。
- **三元组提取质量**：抽取效果受 Extractor LLM 能力影响，但目前实验显示差异不大（模型无关性）。
- **未来方向**：探索更轻量的细粒度可信度评估模型；将框架扩展至更复杂的知识密集型任务（如开放域对话、代码生成）；研究动态调整 $\alpha$ 的策略。

---

## 研究启发与可借鉴点

1. **细粒度可靠性感知范式**：将信息可靠性从文档级下沉到三元组级，结合语义相关性 + 可信度双因子量化，可迁移至其他需要对抗欺骗性信息的 RAG 应用场景。
2. **动态查询构造**：每个推理链独立构造查询以检索互补信息，实现"一链一查"的精准检索，避免多链共享查询导致的噪声扩散。
3. **置信度累积的束搜索策略**：用选择概率的连乘更新链置信度，结合提前终止机制，可在保证推理深度的同时控制搜索空间，适用于多步决策场景。
4. **上下文合成策略分离**：TBS（三元组直接拼接，信息密度高、开销低）与 SBS（映射回原文，语义丰富但开销大）提供两种不同权衡，可根据场景灵活选择。
5. **离线预计算优化**：三元组提取和可信度评分可在离线阶段完成，在线阶段仅需查询和推理，降低实时延迟。

---

## 关键术语表

- **ReliableRAG**：首个基于可靠性的 RAG 框架，通过三元组级细粒度可靠性感知构建健壮推理链，防止欺骗性信息传播。
- **Triple Extraction**：利用 LLM 的 ICL 能力将文档内容提取为结构化三元组（主词-谓词-宾词），以缓解长文本中的中间丢失现象。
- **Triple Evaluation**：在线模块，通过双因子感知机制（语义相关性 + 可信度）动态量化每个三元组的细粒度可靠性，筛选出高可信搜索空间。
- **Chain Construction**：采用束搜索策略自回归构建多跳推理链，支持提前终止，并基于选择概率累积更新链置信度。
- **Dual-Factor Perception Mechanism**：将查询-三元组余弦相似度与三元组可信度评分加权融合，平衡语义匹配与信息可靠性。
- **TBS（Triple-based Synthesis）**：上下文合成策略，将推理链中的三元组直接拼接为自然语言文本，信息密集且输入开销低。
- **SBS（Source-based Synthesis）**：上下文合成策略，将三元组映射回原始文档，通过频率投票保留丰富语义，但输入开销较大。
- **Indirect Effect（IE）**：通过对比不同平衡系数下的生成概率，量化双因子机制对推理正确性的正向贡献。

---

## 可复现要素

- **数据集**：HotPotQA、2WikiMultiHopQA、MuSiQue（公开，论文使用每数据集 1000 测试题 + 100 开发题调参）
- **代码**：论文声明源代码匿名开源（在线可获取）
- **生成器**：Llama3-8B-Instruct（默认），支持 Llama-2-7B、Qwen2.5-7B-Instruct 等
- **Bi-encoder**：E5-Mistral-7B-Instruct（默认），对比 DRAGON+、E5
- **Extractor/Evaluator/Selector**：Llama3-8B-Instruct（默认）
- **关键超参**：平衡系数 $\alpha^* = 0.4$、最大跳数 $L = 4$、beam size $B = ?$、每链保留三元组数 $K = 5$、最终保留链数 $F = 5$
- **低可信文档注入**：每问注入 1–3 篇由 GLM-4-Flash 生成的欺骗性文档

---
