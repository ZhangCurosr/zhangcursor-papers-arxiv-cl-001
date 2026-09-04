---
title: "CONSOLIDATING-RLVR-CAPABILITIES-ACROSS-DOMAINS-A-DEEP-DIVE-I"
source: https://arxiv.org/pdf/2608.27409v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:28:12"
field: "大语言模型多域强化学习后训练"
keywords: ["RLVR", "multi-domain fusion", "task arithmetic", "on-policy distillation", "model merging", "cross-domain transfer", "GRPO"]
innovations: ["统一框架下系统比较 Merge/Mix RL/MOPD 三种多域 RLVR 融合范式", "揭示跨域关系在行为与任务向量几何中的双重结构表征", "证明融合仅重加权已有解分布而不扩展解覆盖且不退化 Held-out 能力"]
benchmarks: ["AIME 2025/26", "GPQA-Diamond", "LiveCodeBench v5/v6", "IFEval", "IFBench", "BFCL v3", "SimpleQA-Verified", "AA-LCR"]
---

# 论文速读：CONSOLIDATING RLVR CAPABILITIES ACROSS DOMAINS: A DEEP DIVE INTO FUSION PARADIGMS

## 一句话总结
本文系统比较了将多域 RLVR 专家能力整合到单一模型的三种融合范式（Merge、Mix RL、MOPD），揭示了跨域关系如何影响融合效果，并给出了基于成本与先验条件的实践选型指南。

## 研究问题与动机
1. **RLVR 收益的跨域扩展难题**：RLVR 通常在单一能力域（如数学、代码）上取得显著提升，但扩展至多域需要训练多个领域专家，现有方法缺乏统一比较框架。
2. **三种融合范式孤立研究**：Merge（合并任务向量）、Mix RL（池化数据集联合训练）、MOPD（多教师在线蒸馏）长期在不同骨干网络与基准上独立评估，结果不可比。
3. **跨域相互作用机制不明确**：不同领域间是互补还是干扰、任务向量几何结构如何决定融合后各域保留程度，尚无系统性刻画。
4. **实际部署中的成本-性能权衡缺失**：三种范式对先验专家、监督信号、推理部署模型数量的要求差异显著，缺乏统一量化分析。

## 核心贡献（创新点）
1. **统一框架下的三范式受控比较**：在相同骨干（Qwen3-4B/8B）、相同专家与数据、相同多域基准套件下进行对比，填补了此前孤立评估的空白。
2. **揭示跨域关系的结构与几何表征**：通过行为轨迹与任务向量余弦相似度两个独立视角，发现推理密集型域（数学/科学/代码）相互正交增强，而指令遵循与 Agent 域与其他域几乎正交。
3. **刻画融合的收益边界**：证明三种范式均仅对基座模型已可访问的解分布进行重加权（pass@k 优势随 k 增大衰减至与基座无差异），且未对 Held-out 事实回忆（SimpleQA-Verified）与长上下文推理（AA-LCR）造成可测量退化。
4. **提供基于成本的实践选型指南**：量化三种范式的 GPU 成本、先验要求与训练动态，形成明确的使用建议矩阵。

## 方法详解
1. **Merge（任务向量合并）**：从基座权重 $\theta_0$ 出发，将每个专家的任务向量 $\tau_i = \theta_i - \theta_0$ 通过组合规则 $\mathcal{F}$ 合并为单次更新：$\theta_{\text{merge}} = \theta_0 + \mathcal{F}(\tau_1, \ldots, \tau_N)$。主要方法包括 Task Arithmetic（TA，$\mathcal{F} = \lambda \sum_i \tau_i$，本文取 $\lambda=0.6$）、TIES、DARE-TA、SCE；对 LoRA 专家还支持 Concat 与 SVD 合并。无需额外训练，仅需分钟级算术运算。

2. **Mix RL（混合数据强化学习）**：将五个领域的训练数据集合并为单一语料库（87,699 条样本，比例：数学 25%、科学 22%、代码 22%、指令遵循 19%、Agent 12%），从 $\theta_0$ 出发执行单次 GRPO 训练，每个 prompt 由对应域验证器打分，跨域交互发生在优化过程中而非事后。

3. **MOPD（多教师在线蒸馏）**：将五个全参数专家分别作为各域教师，学生模型在自身采样轨迹上最小化反向 KL 散度：
$$\mathcal{L}(\theta) = \mathbb{E}_{q, o \sim \pi_\theta} \left[ \sum_{t=1}^{|o|} D_{\text{KL}} \left( \pi_\theta(\cdot|s_t) \| \pi_{\theta_i}(\cdot|s_t) \right) \right]$$
学生在每个 visited state 接收教师 log-prob 提供的稠密监督信号，收敛最快但不超越教师性能上限。

## 实验与结果
- **数据集**：五个训练域——Polaris（数学，38,131 条 hard 样本）、OpenScienceReasoning-2（科学，50,000 条）、CodeContests + Open-R1（代码，19,169 条）、WildChat-1M + Open-Instruct（指令遵循，16,575 条）、WorkplaceAssistant（Agent，10,229 条）。
- **评估基准**：AIME 2025/26、GPQA-Diamond、LiveCodeBench v5/v6、IFEval、IFBench、BFCL v3，以及 Held-out 的 SimpleQA-Verified 和 AA-LCR。
- **骨干模型**：Qwen3-4B-Instruct-2507 与 Qwen3-8B（non-thinking 模式）。
- **主要结果（4B 全参融合，Avg. over 8 benchmarks）**：
  - Base：57.0 → Per-domain RL Full：63.9 → Merge（Full）：**63.7** → Mix RL（Full）：62.3 → MOPD（Full）：63.3
  - **最大单榜差距达 8.6 分**（IFBench：Mix RL 39.8 vs. Per-domain RL IF 专家 49.0）
  - 8B 上 Mix RL 在 AIME 2025 上超过数学专家 **6.5 分**（45.7 vs. 39.2），为表中最大超越幅度
- **最强结果**：4B Merge（Full）以 Avg. 63.7 与 Per-domain RL（63.9）几乎持平；8B Mix RL 在 AIME 2025 达 45.7，超越所有其他融合方法
- **Cost 对比**：Merge 融合阶段 ≈0 GPU-h；Mix RL 总成本 0.58×~0.67×参考；MOPD 融合阶段 <0.2×参考但端到端 1.14×~1.19×参考（因需训练教师）

## 相关工作脉络
1. **Task Arithmetic / Model Merging**（Ilharco et al., 2023; Wortsman et al., 2022; Yadav et al., 2023）：Merge 范式的基础方法，本文系统扩展至多域 RLVR 专家场景并对比替代方案。
2. **Mixed Multi-domain RLVR**（Huang et al., 2026; Wang et al., 2026）：Mix RL 的前置工作，本文在其基础上提供与 Merge/MOPD 的同框架比较。
3. **On-Policy Distillation**（Lu & Lab, 2025; Xiao et al., 2026）：MOPD 的理论基础，本文揭示其在多域场景中的教师边界约束。
4. **Cross-domain interaction in Multi-task Learning**（Yu et al., 2020; Wu et al., 2026; Li et al., 2025）：多任务学习中的交叉域干扰/增强研究，本文为 RLVR 场景提供了行为与参数空间的双重证据。
5. **RLVR post-training**（Jaech et al., 2024; Guo et al., 2025; Shao et al., 2024）：GRPO 与可验证奖励强化学习的基础框架，本文在此之上解决多域整合问题。

## 局限性与未来方向
1. **专家参数的选择未充分探讨**：本文用全参数与 LoRA（rank=32）两类专家做对比，但未系统研究不同 rank 或冻结策略对融合效果的影响。
2. **MOPD 的教师边界未突破**：标准反向 KL 目标限制学生无法超越教师，虽有最近工作尝试用奖励外推突破此上限（Yang et al., 2026），但在多域场景下的可行性尚待验证。
3. **混合比例的敏感性分析有限**：Mix RL 高度依赖数据配比，但本文仅评估了一组固定比例（25/22/22/19/12），未做系统化 sweep。
4. **推理时的 think-mode 泄漏风险**：Merge 中 $\lambda$ 过大导致 Qwen3-8B（hybrid-thinking 模型）触发思考模式泄漏（约 66% 响应包含额外推理 trace），需额外约束。
5. **更多域与更大模型尺度未覆盖**：当前仅五域、两模型规模（4B/8B），跨更多域和更大规模的泛化性需进一步验证。

## 研究启发与可借鉴点
1. **任务向量几何可作为跨域关系的预测指标**：余弦相似度与行为层跨域迁移量的强相关性（图 3(d)）为"哪些域适合联合训练"提供了低成本的事前判断工具。
2. **pass@k 衰减分析揭示融合的本质**：通过 pass@k 曲线区分"解覆盖扩展"与"解重加权"，是一个简洁而有判别力的分析框架，可推广至其他融合场景的能力评估。
3. **Held-out 能力退化测试应成为标配**：本文验证了三类融合方法在 SimpleQA-Verified 和 AA-LCR 上均无退化，这种"无损失保证"的评估设计值得在多域 RLVR 研究中常规化。
4. **LoRA 专家可用于低成本消融**：LoRA 专家（rank=32）在多数基准上与全参数专家差距 ≤1.4 分，为快速验证融合方法提供了低成本的替代方案。
5. **训练动态双坐标轴可视化**：同时以"优化步数"和"该域 prompt 采样数"为横轴绘制收敛曲线，有效分离了方法效率与数据暴露量两个因素，是一个值得复用的分析技巧。

## 关键术语表
**RLVR（Reinforcement Learning with Verifiable Rewards）**：利用可验证奖励信号（如数学答案正确性）对 LLM 进行强化学习微调，代表工作有 GRPO、DeepSeek-R1 等。
**Task Vector（任务向量）**：专家模型权重与基座模型权重之差 $\tau_i = \theta_i - \theta_0$，表示训练施加的参数位移，是 Merge 范式的基本操作单元。
**Merge（任务向量合并）**：无需额外训练，直接将多个专家的任务向量线性组合后加回基座权重，实现多域能力融合。
**Mix RL（混合强化学习）**：将多域训练数据合并为单一语料库，从基座模型出发执行一次联合 RLVR 训练，使跨域交互发生在优化过程中。
**MOPD（Multi-Teacher On-Policy Distillation）**：多教师在线蒸馏，将各领域专家作为教师，在学生自采样轨迹上以反向 KL 散度最小化学生与对应教师 log-prob 的差异。
**GRPO（Group Relative Policy Optimization）**：一种基于组的 RLVR 算法，通过组内相对优势 $\hat{A}_j = r_j - \frac{1}{G}\sum_k r_k$ 估计每个响应的策略梯度。
**Pass@k**：在 k 次采样中至少有一次正确的解题成功率，用于衡量模型求解覆盖范围而非单样本准确率。
**Cross-domain Transfer（跨域迁移）**：在一个域上训练的专家对其他域基准的性能影响，正值为增强，负值为干扰。

## 可复现要素
- **数据集**：Polaris、OpenScienceReasoning-2、CodeContests、Open-R1、WildChat-1M、Open-Instruct、WorkplaceAssistant、AIME 2025/26、GPQA-Diamond、LiveCodeBench v5/v6、IFEval、IFBench、BFCL v3、SimpleQA-Verified、AA-LCR（各数据集均有公开来源）
- **代码/权重**：论文提供了 GitHub 与 Hugging Face 链接（具体地址在原文中，此处略）；实现基于开源框架 verl；全参数与 LoRA 专家权重公开
- **关键超参**：LoRA rank=32，$\alpha=64$；GRPO rollout batch=128，rollout n=16；MOPD rollout batch=264，rollout n=4；学习率 $1\times10^{-6}$（数学域 $1\times10^{-5}$）；训练步数 300（Mix RL 800 步）；Merge 缩放系数 $\lambda=0.6$；所有实验在 32×NVIDIA H20 GPU 上运行
