---
title: "CONSOLIDATING-RLVR-CAPABILITIES-ACROSS-DOMAINS-A-DEEP-DIVE-I"
source: https://arxiv.org/pdf/2608.27409v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:28:05"
field: "多领域大模型强化学习后训练"
keywords: ["RLVR", "Model Merging", "Multi-Domain LLM", "On-Policy Distillation", "Task Vector", "Cross-Domain Transfer"]
innovations: ["首个在统一框架下系统比较Merge/Mix RL/MOPD三种RLVR融合范式", "揭示task vector几何结构与跨域行为迁移的一致性", "界定融合仅reweight而非expand solution coverage的边界"]
benchmarks: ["AIME 2025/26", "GPQA-Diamond", "LiveCodeBench v5/v6", "IFEval", "IFBench", "BFCL v3", "SimpleQA-Verified", "AA-LCR"]
---

# 论文速读：CONSOLIDATING-RLVR-CAPABILITIES-ACROSS-DOMAINS-A-DEEP-DIVE-I

## 一句话总结
论文系统性地比较了三种将多领域RLVR专家整合为统一模型的融合范式（Merge、Mix RL、MOPD），揭示了领域间行为关系与任务向量几何结构的一致性，并提供了基于成本、专家可用性和域间关系的选型指南。

## 研究问题与动机
1. **多领域RLVR的整合难题**：RLVR通常针对单一能力（数学/代码）训练，扩展至多能力需并行服务多个专家或路由，成本高且部署复杂。
2. **三种范式缺乏统一比较**：Merge（合并任务向量）、Mix RL（池化数据集联合训练）、MOPD（多教师在线策略蒸馏）各自独立发展，缺乏在相同backbone、相同数据和相同benchmark下的公平对比。
3. **域间关系不明**：不同领域间是否存在正/负迁移？这种关系在参数空间和行为规范上是否一致？
4. **融合的真实收益与边界**：融合仅提升单样本准确率，还是扩大了可解决问题集合？是否损害未见领域能力？

## 核心贡献（创新点）
1. **首个统一框架下的三范式控制实验**：在Qwen3-4B/8B上训练5个领域专家，以共享expert和数据源对比Merge、Mix RL、MOPD，消除了以往研究因backbone和benchmark不同导致的不可比问题。
2. **揭示跨域关系的几何-行为一致性**：发现数学/科学/代码三个推理密集型领域在行为迁移和task vector余弦相似度上高度对齐，而指令跟随和智能体使用与其他领域几乎正交。
3. **界定融合的增益边界**：证明三种方法均提升pass@1但未扩大solution coverage（pass@32时与base无显著差异），且未损害SimpleQA-Verified（事实记忆）和AA-LCR（长上下文推理）等held-out能力。
4. **提供基于成本-性能的选型指南**：明确各范式在GPU-h、专家前提、收敛速度、教师天花板等维度上的权衡，给出实操性建议。

## 方法详解
**训练设置（基础）**：以Qwen3-4B-Instruct-2507和Qwen3-8B（non-thinking模式）为backbone，用GRPO算法在5个领域分别训练full-parameter和LoRA（rank=32）专家，每领域生成task vector $\tau_i = \theta_i - \theta_0$。

**Merge（任务向量合并）**：
- 不训练，直接将5个expert的task vector按$\theta_{\text{merge}} = \theta_0 + \lambda \sum_i \tau_i$折叠回base weights。
- 主实验取$\lambda = 0.6$（避免thinking-mode泄漏），比较了Average、TA、TIES、DARE-TA、SCE及LoRA的Concat/SVD等方法。
- 对LoRA expert可直接在$A_i, B_i$层面操作以保持rank不变。

**Mix RL（混合数据联合训练）**：
- 跳过单独专家训练，从$\theta_0$出发在一个混合 corpus（87,699条，比例：数学25%、科学22%、代码22%、IF 19%、Agent 12%）上执行单次GRPO。
- 每个prompt按domain分配verifier $v_i$打分，共享策略在同一优化轨迹中接收多领域reward信号。
- 域比例$p_i$直接决定各domain的更新方向和收敛速度。

**MOPD（多教师在线策略蒸馏）**：
- 要求学生模型在自身rollout上最小化反向KL：$\mathcal{L}(\theta) = \mathbb{E}_{q,o \sim \pi_\theta}[\sum_t D_{\text{KL}}(\pi_\theta(\cdot|s_t) \| \pi_{\theta_i}(\cdot|s_t))]$，其中$s_t=(q,o_{<t})$，teacher为对应domain的expert。
- 无需verifier reward，学生仅学习复现teacher行为，不提供超越teacher的信号。
- 收敛最快（100步已达70%收益），但存在教师性能天花板。

## 实验与结果
**数据集与Benchmark**：
- 训练数据：Math（Polaris hard, 38k）、Science（OpenScienceReasoning-2 hard, 50k）、Code（CodeContests+Open-R1, 19k）、IF（WildChat-1M+Open-Instruct, 17k）、Agent（WorkplaceAssistant, 10k）。
- 评估：AIME 2025/26、GPQA-Diamond、LiveCodeBench v5/v6、IFEval、IFBench、BFCL v3；Held-out：SimpleQA-Verified、AA-LCR。采样16 outputs/report mean@16。

**主要结果（Table 1）**：
- **Qwen3-4B**：Base avg=57.0；Per-domain RL全参数avg=63.9、LoRA avg=63.7；Merge(Full)=63.7、Merge(LoRA)=62.7；Mix RL=62.3；MOPD=63.3。
- **Qwen3-8B**：Base avg=42.0；Per-domain RL全参数avg=53.8、LoRA avg=53.5；Merge(Full)=53.5、Merge(LoRA)=53.7；Mix RL=52.8、65.6（注意表格中有两行Mix RL，应为Full和另一变体）；MOPD=53.2。
- **关键数字**：三种融合avg差距≤1.4分，但单benchmark差距达8.6分（IFBench）。Mix RL在8B AIME上超越数学expert 6.5/7.3分（最大margin）。MOPD不超越任何教师。

**收敛与成本（Table 2）**：
- Per-domain RL总成本：4B=5,220 GPU-h、8B=4,670 GPU-h，部署5个模型。
- Merge：融合成本≈0，总成本=专家训练成本（1.00×），部署1模型。
- Mix RL：融合成本3,042/3,138 GPU-h，总成本0.58×/0.67×，无需expert。
- MOPD：融合成本741/899 GPU-h（<0.2×），但总成本1.14×/1.19×（因需先训5个teacher）。

**Solution Coverage（Figure 5）**：
- pass@1：融合较base提升4.5-6.9分；pass@32时三范式与base无显著差异。
- Held-out能力：SimpleQA-Verified和AA-LCR上所有范式均不低于base。

**几何-行为一致性（Figure 3）**：
- 数学-科学余弦相似度：4B=21.7×10⁻³、8B=50.7×10⁻³；数学-代码：10.3/8.2×10⁻³；涉及IF/Agent的配对≤2.3×10⁻³。
- layer-16余弦相似度与行为迁移量呈强正相关。

## 相关工作脉络
1. **Model Merging（Ilharco et al., 2023; Wortsman et al., 2022; Yadav et al., 2023）**：Merge范式的基础，本文将其扩展至RLVR expert融合，并系统比较了TA、TIES、DARE-TA、SCE等多种合并规则。
2. **Mixed Multi-Domain RLVR（Huang et al., 2026; Wang et al., 2026）**：Mix RL的先驱工作，本文首次在统一框架下对比Mix RL与expert-based融合，并量化了数据比例对收敛的影响。
3. **On-Policy Distillation（Lu & Lab, 2025; Li et al., 2026; Yang et al., 2026）**：MOPD基于OPD框架，本文揭示了其在多领域场景下的教师天花板限制，并指出reward extrapolation等变体尚未在多教师融合中验证。
4. **Cross-Domain Transfer in Multi-Task Learning（Yu et al., 2020; Wu et al., 2026）**：本文的行为迁移观察与多任务学习中的gradient interference理论呼应，但首次通过task vector几何在RLVR setting中验证。
5. **RLVR for LLMs（Jaech et al., 2024; Guo et al., 2025; Shao et al., 2024）**：作为融合范式的前提，GRPO等group-based RLVR算法支撑了5个领域expert的训练。

## 局限性与未来方向
1. **领域数量有限**：仅评估5个领域，未覆盖更广泛的capability组合（如多语言、创意写作）。
2. **MOPD未突破教师天花板**：标准reverse-KL目标限制了学生超越teacher，未来需探索reward extrapolation或多教师冲突消解机制。
3. **LoRA expert未充分探索**：论文指出LoRA的cheap部署和跨backbone transfer潜力，但未在融合范式外验证其routing/mixture方案的实际收益。
4. **Scaling behavior未知**：仅在4B/8B上实验，未探究更大模型（如32B+）上融合范式的相对表现变化。
5. **Thinking-mode干扰**：8B合并时在较大λ下出现thinking-mode泄漏（回答长度从2,042增至7,334 tokens），影响评估公平性。

## 研究启发与可借鉴点
1. **几何-行为一致性分析框架**：可通过task vector余弦相似度预测领域间迁移方向，为多领域训练的数据配比提供先验指导（如正相关领域可高比例混合，正交领域需单独保护）。
2. **pass@k覆盖度评估**：证明融合主要reweight而非expand solution set，提示未来工作应关注如何引入新pattern（如weak-to-strong generalization）。
3. **Held-out能力监控**：SimpleQA-Verified和AA-LCR作为held-out benchmark的选取方式，可作为RLVR融合论文的标配评估，确保能力无损。
4. **成本透明化报告**：将融合阶段成本与端到端成本分开报告（如Table 2），为工业界部署决策提供清晰依据。
5. **LoRA merging直接操作适配器**：在LoRA场景下直接在$A_i, B_i$矩阵上应用权重/稀疏化规则（如SVD、Concat），避免full-parameter merge的显存开销。

## 关键术语表
**RLVR**：Reinforcement Learning with Verifiable Rewards，通过可验证规则（如代码执行、数学答案）打分并驱动策略梯度更新的LLM后训练范式。
**Task Vector**：$\tau_i = \theta_i - \theta_0$，expert相对于base model的参数位移，表征该领域训练引入的权重更新。
**Merge**：无需额外训练，将多个expert的task vector按加权规则求和后折叠回base weights的参数空间融合方法。
**Mix RL**：跳过单独expert训练，将所有domain数据混合后在单一策略上执行RLVR，依赖数据比例调控域间交互。
**MOPD**：Multi-Teacher On-Policy Distillation，学生模型在自身rollout上通过反向KL匹配domain-specific teacher的logits分布。
**pass@k**：在k次采样中至少1次正确的概率度量，用于评估solution coverage而非单样本准确率。
**GRPO**：Group Relative Policy Optimization，Group-based RLVR算法，通过组内相对优势估计$\hat{A}_j = r_j - \frac{1}{G}\sum_k r_k$更新策略。
**Held-out Capability**：训练过程中完全未涉及的模型能力（如事实记忆、长上下文推理），用于验证融合是否导致能力退化。

## 可复现要素
- **数据集**：Polaris（Math）、OpenScienceReasoning-2（Science）、CodeContests+Open-R1（Code）、WildChat-1M+Open-Instruct（IF）、WorkplaceAssistant（Agent）；部分数据经难度过滤/随机子采样。
- **代码开源**：论文声明GitHub和Hugging Face链接（具体URL在原文中未完整列出，需访问arxiv页面获取）。
- **框架**：基于verl（Sheng et al., 2025）实现。
- **关键超参**：LoRA rank=32、α=64；学习率1e-6（math域1e-5）；rollout batch=128（MOPD=264）；rollout n=16（MOPD=4）；训练步数Per-domain/MOPD=300、Mix RL=800；max prompt/response length=5120/16384；温度=1.0；32张NVIDIA H20 GPU。
- **权重开源**：论文未明确声明base model及expert权重开源状态，需查阅附属仓库。
