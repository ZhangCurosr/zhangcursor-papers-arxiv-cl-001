---
title: "Sequential-Beats-Joint-On-the-Interplay-between-On-Policy-Di"
source: https://arxiv.org/pdf/2609.04108v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:22:33"
field: "推理型大模型后训练"
keywords: ["on-policy distillation", "RLVR", "post-training", "reasoning LLM", "GRPO", "knowledge distillation", "sequential training"]
innovations: ["提出并验证OPD-then-RL两阶段方案，解耦OPD能力扩展与RL sharpening，超越所有联合优化方法", "统一token-level policy-gradient视角将现有OPD-RLVR混合方法归纳为weighted-additive与teacher-modulated两类范式", "从pass@k动态、学习动力学与参数sign-conflict三个角度一致解释解耦优于联合的机制"]
benchmarks: ["Reasoning Gym (Knights & Knaves, Zebra Puzzles, Countdown)", "DeepMath-103K", "MATH-500", "AMC23", "AIME24", "AIME25"]
---

# 论文速读：Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

## 一句话总结
本文发现并系统地验证了一个简单的两阶段训练方案 **OPD-then-RL**（先用 on-policy distillation 扩展学生模型的推理覆盖范围，再用 RLVR 在其内 sharpening），该方案在逻辑和数学推理任务上稳定超越纯 OPD、纯 RLVR 以及所有已知的联合优化方法（加权相加式和教师调制式），揭示了两种信号联合优化时的本质干扰机制。

## 研究问题与动机
1. **核心问题**：后训练推理 LLM 时，dense token-level 监督（OPD）与 sparse outcome reward（RLVR）应如何组合？现有方法均在单步内融合两类信号，但其有效性和副作用未被系统理解。
2. **纯方法的局限**：RLVR 奖励稀疏，需大量 trial-and-error 才能探索到高奖励行为；OPD 提供密集 token 级监督、消除探索负担，但仅优化行为代理（模仿教师），不保证任务性能最大化。
3. **联合优化的潜在干扰**：加权相加式可能因教师信号翻转优势符号而破坏 RL 更新方向；教师调制式虽保留符号一致性，但两信号在同一步骤内仍会相互掣肘。
4. **动机**：通过系统化对比与机制分析，寻找能避免干扰、充分发挥两者互补性的训练策略。

## 核心贡献（创新点）
1. **提出并验证 OPD-then-RL 两阶段方案**：首次系统论证了"先 OPD 扩展能力边界、后 RL 在范围内 sharpening"的解耦训练优于任何联合优化方式，在逻辑推理上最高领先 26.7 pass@1 点。
2. **统一 token-level 视角归纳现有方法**：将所有 OPD-RLVR 混合方法抽象为共享 PPO clipped surrogate 但 advantage 不同的框架，清晰划分为 weighted-additive 与 teacher-modulated 两类范式。
3. **三项互补的机制解释**：从 pass@k 行为（OPD 扩围、RL sharpening）、学习动态（打破教师性能天花板、保持分布支持内收敛）、参数更新（最低 sign-conflict rate）三个角度一致解释为何解耦优于联合。
4. **给出实用操作指南**：确认 OPD 验证分数是切换至 RL 的关键信号；证明 OPD 作为 RL cold start 显著优于传统 SFT。

## 方法详解
- **统一 token-level policy-gradient 形式**：RL（GRPO）与 OPD 均可写成 $\nabla_\theta \mathcal{I} = \mathbb{E}[\sum_{i,t} A_t^{(i)} \nabla_\theta \log \pi_\theta(y_t^{(i)}|h_t^{(i)})]$，区别仅在于 token-level advantage：$A_t^{\text{GRPO}} = \hat{A}^{(i)}$（group-normalized outcome reward），$A_t^{\text{OPD}} = d_t^{(i)} = \log \pi_T(y_t^{(i)}|h_t^{(i)}) - \log \pi_\theta(y_t^{(i)}|h_t^{(i)})$。
- **两类联合范式**：
  - *Weighted-Additive*：$A_t^{\text{add}} = w_R \hat{A} + w_T d_t$，如 KDRL、SRPO、HDPO；优势符号可能被教师项翻转。
  - *Teacher-Modulated*：$A_t^{\text{mod}} = m(d_t) \cdot \hat{A}$，如 TRRD、RLSD；教师仅缩放幅度，不改变符号。
- **OPD-then-RL 两阶段**：前 $S$ 步仅用 OPD 目标优化 $\pi_\theta$，之后切换为纯 GRPO 目标；关键超参为切换步数 $S$。
- **EOS 位置掩码**：由于学生与教师 tokenizer 的 EOS token 不同，教师信号在 EOS 位置会产生人为大惩罚，所有方法均掩码掉 EOS 位置。

## 实验与结果
- **数据集**：逻辑推理（Reasoning Gym：Knights & Knaves、Zebra Puzzles、Countdown）；数学推理（DeepMath-103K 训练集，MATH-500、AMC23、AIME24、AIME25 评测）。
- **模型配置**：Qwen3-8B（非 thinking 模式）为教师；Qwen3-1.7B-Base（主实验）/ 0.6B-Base（消融）为学生；VERL + vLLM + FSDP，bf16，AdamW lr=1e-6，PPO clip=0.2。
- **核心结果**（Table 2）：
  - *逻辑推理 avg pass@1*：OPD-then-RL = **80.6**，显著领先 OPD（53.9）、GRPO（49.4）及所有联合方法；最高领先 KDRL（62.8）达 **+17.8 点**，在 K&K 上领先 92.6 vs 53.7（+38.9）。
  - *数学推理 avg pass@1*：OPD-then-RL = **31.8**，略优于 OPD（31.0）和 GRPO（28.4），bootstrap 检验显著领先 6/9 竞争方法。
  - *pass@32*：OPD-then-RL 在两类任务上均为最优或并列第一，KDRL-Annealing 次之。
- **跨模型泛化**：在 Qwen3-0.6B 学生（Table 5）及 OLMo-3.1-32B teacher / OLMo-3-7B student（Table 6）上均复现相同排序。

## 相关工作脉络
1. **RLVR for Reasoning**：GRPO（Shao et al., 2024）、DeepSeek-R1（Guo et al., 2025）、Kimi-K1.5（Team et al., 2025）等，利用 verifiable reward 激发长链推理；但 sparse reward 导致探索效率低。
2. **On-Policy Distillation (OPD)**：Lu & Lab（2025）、Agarwal et al.（2024）、Gu et al.（2024）；后续被工业界广泛采用（Qwen3、GLM-5、DeepSeek-V4）。本文揭示其 capability expansion 角色。
3. **OPD-RLVR 联合优化**：KDRL（Xu et al., 2025）、SRPO（Li et al., 2026a）、HDPO（Ding, 2026）、TRRD（Zhang et al., 2026b）、RLSD（Yang et al., 2026）；本文指出这些方法在两信号融合时存在内在干扰。
4. **SFT-then-RL 序列方案**：Guo et al.（2025）、Limozin et al.（2026）等探索 off-policy SFT 后接 RL；本文证明 OPD（on-policy）作为 cold start 比 SFT 更优（Table 8）。
5. **Off-policy vs. On-policy 蒸馏 debate**：Muennighoff et al.（2025）、Jin et al.（2026）讨论 exposure bias 与 KL 目标选择；本文聚焦 on-policy 场景下的信号组合策略。

## 局限性与未来方向
1. **教师-学生配置范围受限**：仅研究 external teacher 强于 student 的设定；task-specific teacher、multi-teacher、on-policy self-distillation 等场景待探索。
2. **教师固定冻结**：未考虑先对教师做 RLVR 再蒸馏给学生的"教师优先"范式（行业常见，如 GLM-5、DeepSeek-V4）。
3. **OPD 目标单一**：仅使用 reverse-KL on student rollouts；forward-KL、JSD、full distribution/top-K 等替代目标的冷启动效果未知。
4. **切换点 $S$ 的泛化性**：当前使用固定步数（60），更鲁棒的动态切换准则（如基于 validation score 阈值）仍需研究。
5. **仅覆盖逻辑与数学推理**：其他领域（代码生成、agent planning）的结论是否成立未验证。

## 研究启发与可借鉴点
1. **"解耦优于耦合"的设计哲学**：对于多信号后训练场景，优先考虑时序分离而非单步融合，避免信号间 interference；该原则可迁移至 reward modeling + distillation、multi-task RL 等方向。
2. **pass@k 曲线分析作为诊断工具**：通过 pass@k vs. k 和 pass@k vs. difficulty 分解模型能力的"覆盖度"与"集中度"，为方法比较提供更丰富洞察，值得引入日常评测流程。
3. **Sign-Conflict Rate (SCR) 作为更新兼容性度量**：定量衡量不同目标对关键参数的更新方向一致性，可直接用于筛选或诊断混合训练方案，避免"看似合理实则冲突"的信号组合。
4. **OPD 作为 RL cold start 的实证优势**：建议将 OPD 纳入 RL 训练 pipeline 的标准起始阶段，替代传统 SFT，尤其在教师模型能力显著高于学生时增益更明显。
5. **EOS token mismatch 的工程细节**：跨 tokenizer 蒸馏时需注意 EOS 位置的伪惩罚，掩码处理是可复用的工程技巧。

## 关键术语表
**On-Policy Distillation (OPD)**：在学生自身采样轨迹上计算 token-level 教师-学生分布差异并进行反向 KL 最小化的训练范式，提供密集监督信号。
**Reinforcement Learning with Verifiable Rewards (RLVR)**：利用可验证的结局奖励（如数学答案正确性）通过策略梯度（如 GRPO）优化推理模型的后训练方法。
**Pass@k**：对每道题目采样 k 次回答，至少一次正确的概率；pass@1 衡量单次生成质量，pass@k（k 大）衡量搜索空间覆盖广度。
**Weighted-Additive Paradigm**：将 RL 优势与 OPD 优势线性相加作为最终 token-level 优势的联合优化范式，可能存在符号冲突。
**Teacher-Modulated Paradigm**：用教师信号仅缩放 RL 优势幅度（不改符号）的联合优化范式，试图保留教师的方向指引而不颠覆任务目标。
**Sign-Conflict Rate (SCR)**：某方法在 OPD 更新幅度最大的参数子空间上与 OPD 更新方向符号相反的参数比例，衡量两种信号的参数级冲突程度。
**Cold Start**：RL 训练前的初始模型状态；本文比较 OPD 微调与 SFT 微调作为 GRPO 的冷启动效果。
**Distributional Support**：教师分布赋予显著概率的生成区域；OPD-then-RL 的 RL 阶段始终在此区域内集中概率，而非漂移至教师支持外。

## 可复现要素
- **数据集**：Reasoning Gym（Knights & Knaves, Zebra, Countdown）训练各 20K 题；DeepMath-103K（约 103K 数学题）；评测集 MATH-500、AMC23、AIME24/25 均为公开基准。训练/评测脚本遵循 Reasoning Gym 与 DeepMath 官方实现。
- **代码**：基于 VERL 框架 + vLLM rollouts + FSDP，开源仓库未明确声明；论文未提供独立代码链接。
- **权重**：教师 Qwen3-8B、学生 Qwen3-1.7B-Base / 0.6B-Base 均为公开模型；OPD-then-RL 最终权重未公开。
- **关键超参**：lr=1e-6、batch=128、rollouts=8、PPO clip=0.2、temperature=1.0、max response length=4096（逻辑）/ 8192（数学）；OPD 阶段步数 $S=60$；$\beta$ 按任务 tuned（逻辑 0.2/0.02，数学 0.002）。
