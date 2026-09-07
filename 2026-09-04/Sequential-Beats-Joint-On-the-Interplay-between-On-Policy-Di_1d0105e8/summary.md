---
title: "Sequential-Beats-Joint-On-the-Interplay-between-On-Policy-Di"
source: https://arxiv.org/pdf/2609.04108v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:22:52"
field: "大语言模型推理能力训练"
keywords: ["on-policy distillation", "RLVR", "reinforcement learning", "knowledge distillation", "reasoning LLM", "post-training"]
innovations: ["提出OPD-then-RL两阶段顺序方案，超越所有联合优化基线", "从pass@k、学习动态和参数更新三视角系统解释顺序训练有效性", "揭示联合优化中信号干扰机制并提出切换时机选择策略"]
benchmarks: ["Reasoning Gym (K&K, Zebra, Countdown)", "MATH-500", "AMC23", "AIME24", "AIME25"]
---

# 论文速读：Sequential-Beats-Joint-On-the-Interplay-between-On-Policy-Distillation-and-RLVR

## 一句话总结
本文系统研究了 On-Policy Distillation (OPD) 与 RLVR 的组合方式，发现简单的两阶段顺序方案 OPD-then-RL 在逻辑与数学推理任务上一致优于纯 OPD、纯 RLVR 以及所有已知的联合优化基线；通过 pass@k 分析、学习动态和参数更新三个视角揭示了其机制：OPD 负责扩展教师支持解的覆盖范围，RL 负责在该范围内锐化，而联合优化会导致两种信号相互干扰。

## 研究问题与动机
- **现有方法的盲区**：当前融合 OPD 与 RLVR 的工作均采用单步联合优化策略，分为"加权加法"（将 RL 优势与 OPD 优势直接相加）和"教师调制"（用教师信号缩放 RL 优势的幅度）两类范式，但二者均未系统比较顺序训练方案。
- **两种方法的天然互补性**：RLVR 提供任务目标对齐的稀疏结果级奖励，但探索成本高；OPD 提供密集的 token 级监督，能快速提升学习效率，但仅优化行为代理而非真实任务性能。
- **教师性能的天花板效应**：纯 OPD 和联合方法在逻辑推理任务上普遍被教师模型性能所限，无法突破教师表现上限。
- **冷启动选择问题**：RL 训练前的冷启动阶段（如 SFT 与 OPD）对后续 RL 效果的影响尚未被充分研究。

## 核心贡献（创新点）
- **提出并验证 OPD-then-RL 顺序方案**：将 OPD 与 RLVR 分两个独立阶段训练，而非单步融合，在多个逻辑与数学推理基准上超越所有联合优化基线，逻辑推理任务最高领先 26.7 pass@1。
- **统一形式化现有 OPD-RLVR 组合方法**：在 token 级策略梯度视角下，将所有已有方法归为加权加法和教师调制两类范式，揭示它们仅在 token 级优势函数构造上的差异，为后续分析提供统一框架。
- **从三个角度系统解释 OPD-then-RL 的有效性**：pass@k 分析证明 OPD 扩展能力边界而 RL 在其中锐化；学习动态分析揭示联合方法受限于教师性能天花板而顺序训练可突破；参数更新分析表明联合方法的符号冲突会牺牲 OPD 的能力扩展方向。

## 方法详解
- **统一 token 级策略梯度形式**：RL（GRPO）和 OPD 的优化目标均可写为 $\nabla_\theta \mathcal{I}^{\text{alg}}(\theta) = \mathbb{E}[\sum_{i,t} A_t^{\text{alg},(i)} \nabla_\theta \log \pi_\theta(y_t^{(i)} | h_t^{(i)})]$，其中 RL 的 token 级优势 $A_t^{\text{GRPO},(i)} = \hat{A}^{(i)}$（基于组内归一化的结果奖励），OPD 的 token 级优势 $A_t^{\text{OPD},(i)} = d_t^{(i)} = \log \pi_T(y_t^{(i)}|h_t^{(i)}) - \log \pi_\theta(y_t^{(i)}|h_t^{(i)})$。
- **OPD-then-RL 两阶段训练**：第一阶段执行 OPD 优化 $\mathcal{J}_{\text{OPD}}$ 共 $S$ 步，第二阶段切换为纯 GRPO 优化 $\mathcal{J}_{\text{GRPO}}$；优势函数按训练步数硬切换，不引入任何软调度。
- **加权加法范式**：将教师信号作为独立加项叠加到 RL 优势上，形式为 $A_t^{\text{add},(i)} = w_R^{(i,t)} \hat{A}^{(i)} + w_T^{(i,t)} d_t^{(i)}$，代表方法包括 KDRL、KDRL-mask、SRPO、HDPO。
- **教师调制范式**：用教师信号仅调节 RL 优势的幅度，保持符号由 verifiable reward 决定，形式为 $A_t^{\text{mod},(i)} = m(d_t^{(i)}) \cdot \hat{A}^{(i)}$，代表方法包括 TRRD（将教师信号折叠入 importance ratio）和 RLSD（用 clipped teacher factor 直接缩放优势）。
- **切换点选择策略**：OPD 阶段的验证分数是切换时机的关键信号；在 OPD 快速改善阶段结束后（如 step 60）切换可获得最佳性能，而验证分数饱和后继续 OPD 的收益递减。

## 实验与结果
- **模型与任务**：教师模型 Qwen3-8B，学生模型 Qwen3-1.7B-Base（补充实验使用 Qwen3-0.6B-Base）；逻辑推理任务包括 Knights & Knaves（K&K）、Zebra Puzzles、Countdown；数学推理任务使用 DeepMath-103K 训练，在 MATH-500、AMC23、AIME24、AIME25 上评估。
- **主要结果（逻辑推理）**：OPD-then-RL 在三个逻辑任务上的平均 pass@1 为 80.6，远超其他方法（如 KDRL 62.8、SRPO 56.0、GRPO 49.4），最高领先幅度达 26.7（K&K 任务 92.6 vs. KDRL 53.7）。
- **主要结果（数学推理）**：OPD-then-RL 平均 pass@1 为 31.8，与最强基线（TRRD 30.1、SRPO 31.6）差距较小但在统计上显著优于六个基线；pass@32 无方法显著优于 OPD-then-RL。
- **跨模型泛化**：使用 Qwen3-0.6B-Base 学生和 OLMo-3.1-32B-Instruct 教师时，OPD-then-RL 同样 dominating，表明结果不受学生规模或模型族限制。
- **关键对比**：OPD-then-RL（硬切换）优于 KDRL-Annealing（软调度），验证了信号干扰问题；OPD 冷启动优于 SFT 冷启动，前者在 GRPO 后平均提升 7.2 pass@32，而后者几乎无收益。

## 相关工作脉络
- **RLVR 方法**：GRPO（Shao et al., 2024）、DeepSeek-R1（Guo et al., 2025）、Kimi-K1.5（Team et al., 2025）等工作探索了稀疏奖励下推理能力的激发，但 RLVR 单独使用探索效率低。
- **On-Policy Distillation**：Lu and Lab（2025）、Agarwal et al.（2024）、Gu et al.（2024）提出 OPD 概念，近期被 Qwen3（Yang et al., 2025）、GLM-5（GLM-5-Team, 2026）、DeepSeek-V4（DeepSeek-AI, 2026）等工业管线广泛采用。
- **SFT-then-RL 顺序训练**：Limozin et al.（2026）、Hu et al.（2026）等工作探索了 off-policy SFT 与 RL 的顺序组合，本文与之定位不同，聚焦于 on-policy 蒸馏与 RL 的组合。
- **加权加法融合**：KDRL（Xu et al., 2025）、SRPO（Li et al., 2026a）、HDPO（Ding, 2026）等方法将教师信号作为独立项加入 RL 优势，本文指出这类方法会因信号冲突牺牲 pass@k 覆盖能力。
- **教师调制融合**：TRRD（Zhang et al., 2026b）、RLSD（Yang et al., 2026）等方法用教师信号调制 RL 优势幅度，保留符号一致性，但本文证明其仍无法避免联合优化带来的干扰。
- **Self-distilled RLVR**：Zhao et al.（2026）、Hübotter et al.（2026）提出的 SRPO 和 RLSD 原本针对 self-distillation 场景设计，本文将其机制移植到外部教师设置下进行公平对比。

## 局限性与未来方向
- **教师-学生配置范围有限**：研究聚焦于外部教师强于学生的常见设置，未探索任务特定教师、多教师场景以及 on-policy self-distillation（教师为带额外信息的自身增强版本）。
- **冻结教师假设**：采用固定教师以保持监督信号稳定，未比较"先用 RLVR 改进教师再用蒸馏转移给学生"的方案（后者在工业管线中更为常见）。
- **OPD 目标函数的选择**：仅使用 reverse-KL 目标，未探索全 next-token 分布上的 reverse-KL、forward-KL 或 Jensen-Shannon divergence 等其他目标可能对冷启动产生的不同影响。
- **数学推理提升相对有限**：在数学推理任务上，OPD-then-RL 仅以小幅优势领先最强基线，可能因教师本身已在该领域高度优化，信号干扰较小而顺序分离的额外收益有限。

## 研究启发与可借鉴点
- **顺序分离优于联合融合的通用原则**：当两种训练信号承担不同功能（如能力扩展 vs. 锐化），顺序执行比单步融合更能避免信号干扰，该原则可推广至其他多目标训练场景。
- **pass@k 分析作为能力扩展诊断工具**：通过观察不同 k 下的性能变化轨迹，可精确识别 OPD 的扩展效应与 RL 的锐化效应，为训练策略选择提供可量化依据。
- **验证分数作为切换时机的实用信号**：OPD 阶段的验证性能饱和点是切换至 RL 的可靠指示器，无需依赖固定步数，可直接应用于实际训练管线。
- **OPD 作为 RL 冷启动的价值**：相比传统 SFT，OPD 能保留更多 on-policy 特性，为后续 RL 提供更强的初始分布，这一发现可指导推理模型的训练流水线设计。
- **符号冲突率（SCR）作为方法评估指标**：通过度量联合方法与纯 OPD 参数更新方向的冲突程度，可预测方法是否会牺牲能力扩展，为组合方法的设计提供新的分析维度。

## 关键术语表
- **On-Policy Distillation (OPD)**：在学生自采样的 rollout 上使用 token 级教师信号进行蒸馏，最小化学生与教师分布间的 reverse-KL 散度，相比 off-policy SFT 避免 exposure bias。
- **Reinforcement Learning with Verifiable Rewards (RLVR)**：基于可验证的结果奖励（如数学答案正确性）对模型进行策略梯度优化，典型算法包括 GRPO 和 PPO。
- **pass@k**：对每个问题采样 k 个响应，至少有一个正确的概率；pass@1 衡量单次生成的准确率，pass@32 或 pass@128 衡量多次采样的覆盖能力。
- **加权加法范式 (Weighted-Additive)**：将 RL 优势与 OPD 优势直接线性相加作为最终 token 级优势，可能导致教师信号覆盖 verifiable reward 信号。
- **教师调制范式 (Teacher-Modulated)**：仅用教师信号缩放 RL 优势的幅度而不改变其符号，保持 verifiable reward 的决定性作用。
- **OPD-then-RL**：先执行固定步数的 OPD 训练，再切换为纯 GRPO 训练的两阶段顺序方案，避免两种信号的瞬时冲突。
- **符号冲突率 (Sign-Conflict Rate, SCR)**：衡量某方法的参数更新方向与纯 OPD 更新方向在 magnitude 最大的参数子空间上的冲突比例，用于评估方法对 OPD 更新结构的破坏程度。

## 可复现要素
- **数据集**：Reasoning Gym（Knights & Knaves、Zebra Puzzles、Countdown），DeepMath-103K，MATH-500，AMC23，AIME24，AIME25；Reasoning Gym 和 DeepMath-103K 为公开数据集。
- **代码**：论文使用 VERL 框架实现所有方法，但未提供官方开源代码链接（论文未提及代码仓库 URL）。
- **模型**：教师模型 Qwen3-8B（non-thinking 模式），学生模型 Qwen3-1.7B-Base；模型权重可通过 HuggingFace 获取。
- **关键超参**：学习率 $1 \times 10^{-6}$，梯度裁剪 1.0，PPO clip ratio 0.2，batch size 128，rollout 数 8，OPD 阶段 60 步，总训练步数逻辑任务 150 步/数学任务 120 步，temperature 1.0，max response length 逻辑 4096/数学 8192。
