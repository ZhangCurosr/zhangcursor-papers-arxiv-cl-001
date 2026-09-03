---
title: "HOW-TO-TRAIN-A-CRITIC-STABLY-AND-EFFICIENTLY"
source: https://arxiv.org/pdf/2608.23566v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:58:09"
field: "大语言模型强化学习"
keywords: ["critic-based RL", "LLM reinforcement learning", "PPO", "GRPO", "value function stabilization", "privileged critic", "GAE", "RLHF"]
innovations: ["BPCO配方：DPPO+有界价值头+解耦GAE+无归一化优势度+长度自适应GAE的五项整合", "首次将参考答案/rubric作为训练时critic的特权输入", "通过可控sanity test逐一隔离并诊断critic训练不稳定根因"]
benchmarks: ["AIME 2025 avg@32", "DeepScaleR", "DAPO-Math-17k", "OpenRubrics"]
---

# 论文速读：HOW-TO-TRAIN-A-CRITIC-STABLY-AND-EFFICIENTLY

## 一句话总结
本文针对大语言模型（LLM）强化学习中 critic-based 训练不稳定的问题，提出 **BPCO（Best-Practice Critic Optimization）** 训练配方，通过五项关键设计（DPPO裁剪、有界价值预测、无偏蒙特卡洛价值目标、无归一化优势度、长度自适应GAE）使得单响应critic训练既稳定又高效，在数学推理任务上可匹配甚至超越组方法（group-based），同时首次将"特权信息"用于训练时critic。

## 研究问题与动机
1. **组方法效率低**：GRPO等方法需为每个prompt采样G个响应再计算优势度，无法构建token级精确的信用分配。
2. **标准critic配方不稳定**：PPO的ratio clipping对高低概率token处理不均匀；bootstrapped价值目标继承critic误差；固定λ使长短响应中terminal reward权重差异大。
3. **线性价值头越界**：即使已知回报有界（如[0,1]），标准线性头仍会预测范围外数值，导致训练崩溃。
4. **批量优势归一化有害**：策略接近最优时优势度方差趋于0，归一化会将噪声放大为训练信号，阻碍更新自然收敛。
5. **critic可获特权信息**：critic仅在训练中使用，可暴露参考答案/评分标准等仅由prompt决定的信息，降低价值函数近似难度。

## 核心贡献（创新点）
1. **系统性提出BPCO配方**：整合DPPO、有界价值头、解耦GAE（λ_π<1, λ_V=1）、移除优势归一化、长度自适应GAE五项设计，形成单响应actor-critic的完整训练流程。
2. **首次将特权信息引入critic训练**：证明参考答案/官方解法/rubric可作为critic的额外输入，在不改变策略输入的前提下加速critic学习。
3. **通过可控 sanity test 逐一隔离五项设计的独立效应**，揭示标准配方中"看似合理"的组件（线性头、batch归一化、固定λ）为何破坏稳定性。
4. **在1.5B~30B-A3B多尺度模型、数学推理与rubric奖励任务上验证**，BPCO持续优于critic基线，且单响应下匹配或超越组方法。

## 方法详解
BPCO在 verl 默认 critic-based 基线上进行五项增量修改：

### Step 1：用 DPPO 替换 PPO
- PPO 使用固定 ε 裁剪策略比率，等价于对采样token施加相对概率变化约束，导致低概率token变化被过度抑制、高概率token允许大跳跃。
- DPPO 将裁剪边界改为 ε/μ(y_t|s_t)，等价于约束 **|π_θ(y_t|s_t) − μ(y_t|s_t)| ≤ ε**，即统一的绝对概率变化阈值。

### Step 2：价值预测有界于奖励范围 [R_min, R_max]
- 标准线性头 z_φ(s_t) ∈ ℝ 无界，即便回报已知有界也会越界。
- BPCO 采用缩放反正切映射：
  > V_φ(s_t) = R_min + (R_max − R_min)·(1/2 + (1/π)·arctan(z_φ(s_t)))
- 保证任意有限输入均映射到 (R_min, R_max) 开区间内。

### Step 3：解耦 GAE，critic 目标使用无偏蒙特卡洛
- 保留 λ_π = 0.99 用于策略优势估计（方差缩减）。
- 设 λ_V = 1，γ=1 时价值目标退化为观测回报：
  > V̂_t = R(x, y)
- 标准自引用目标 V̂_t(λ)=Â_GAE(λ)+V_φ_old(s_t) 在 λ<1 时容易拟合但偏离真实回报，导致"高解释方差+低实际价值"的假象。

### Step 4：移除批量优势归一化
- 标准做法将优势度标准化：Ã_t = (Â_t − Ā)/σ_A，使每个batch的 Advantage 方差为1。
- BPCO 直接使用原始 GAE 优势度 Â_t。
- 原因：策略接近最优时 Â_t → 0，归一化将微小估计噪声放大为大幅策略更新，且减去均值可能翻转小正优势的符号，损害探索。

### Step 5：长度自适应 GAE（LA-GAE）
- 固定 λ_π = 0.99 对长响应过早依赖 bootstrapped 残差，系统性critic误差主导策略信号。
- BPCO 采用：
  > λ_π(L) = 1 − 1/(αL)，其中 α > 0
- 短响应 λ_π 较小（更多bootstrapping），长响应 λ_π → 1（更接近MC估计），缓解长尾token估计偏差。

### 特权信息输入（Privileged Critic）
- 对数学题：将参考答案（answer/solution）作为critic额外输入；对开放任务：将评分rubric作为额外输入。
- 因 q(x) 由prompt固定，不改变最优价值函数，仅降低近似难度。
- 策略输入保持不变，推理时无额外开销。

## 实验与结果
**实验设置**：
- 数据集：DeepScaleR（40.3K数学题）、DAPO-Math-17k、OpenRubrics（rubric奖励）
- 模型：DeepSeek-R1-Distill-Qwen-1.5B、Qwen3-30B-A3B-Base/MoE、Qwen3-4B-Base
- 评估：AIME 2025 avg@32（每题采样32次取均值准确率）
- 基线：Dr. GRPO（组大小G=16）、标准critic基线（含 decoupled GAE + LA-GAE，但保留线性头和batch归一化）

**主要结果**：
1. **1.5B + DeepScaleR**：BPCO 在训练奖励、验证性能和解释方差上持续优于两组基线；移除价值有界性降低训练效率和AIME性能；特权答案显著提升训练速度。
2. **30B-A3B + DAPO-Math-17k**：critic基线在100步后AIME停止提升（训练不稳定），BPCO在两种模型上均大幅领先；对比Dr. GRPO在30B-A3B上胜出、在Base上持平。
3. **4B + OpenRubrics（rubric奖励）**：BPCO无特权信息版本优于两组基线，证明核心价值设计在无特权场景下同样有效；特权信息在此简单任务上无额外收益（过拟合风险）。

**结论**：BPCO 在全部三个尺度/任务上稳定改进 critic 基线，并在单响应条件下匹配或超越组方法。

## 相关工作脉络
1. **PPO（Schulman et al., 2017）**：LLM RL 基础策略优化算法，BPCO 以其为起点，指出 ratio clipping 在LLM词汇空间中的不公平性，用 DPPO 改进。
2. **GRPO / Dr. GRPO（Shao et al., 2024; Liu et al., 2025）**：无critic的组方法代表，BPCO 以它们为效率对照，证明单响应critic可达同等甚至更优效果。
3. **DPPO（Qi et al., 2026b）**：已被作者团队提出，BPCO 将其作为策略更新的标准组件纳入统一配方。
4. **Decoupled GAE（VC-PPO, Yuan et al., 2025）**：BPCO 借鉴其"policy与value使用不同λ"思想，将 λ_V 严格设为1以获得无偏MC目标。
5. **LA-GAE（VAPO/SAO, Yue et al., 2025; Hou et al., 2026）**：BPCO 采用并验证了长度自适应GAE在 critic 场景下的有效性。
6. **Privileged Critic 理念**：类比多智能体RL中的"centralized training with decentralized execution"（Vinyals et al., 2019; Wang et al., 2021），首次系统引入LLM RL critic训练。

## 局限性与未来方向
1. **任务覆盖有限**：证据仅来自数学推理和rubric奖励，需验证在代码生成、对话、多模态等场景的泛化性。
2. **依赖已知奖励范围**：BPCO 假设 R_min/R_max 已知（如二值reward为[0,1]），对于复杂连续奖励分布需扩展。
3. **特权信息非普适**：需存在可暴露的奖励定义信息（答案/rubric），在纯无监督或匿名评分场景中不适用；小数据下特权信息易导致过拟合。
4. **计算/内存开销未计入**：critic 训练增加额外计算和显存消耗，与 trajectory-matched 的组方法比较时未做成本权衡分析。
5. **未来方向**：拓展至开放式写作/对话评估、探索动态奖励范围的自适应边界、研究特权信息的选择性注入策略。

## 研究启发与可借鉴点
1. **价值有界映射作为通用组件**：缩放 arctan 或 sigmoid 边界化是可复用的稳定训练技巧，适用于任何回报有界的 RLHF 场景。
2. **解耦 policy/value 的 GAE 参数**：λ_π < 1 降方差、λ_V = 1 去偏的分离策略可迁移至所有 critic-based LLM RL 管线。
3. **移除 batch-wise 优势归一化**：这是一个反直觉但有效的设计，建议在团队新 pipeline 中默认禁用。
4. **特权信息利用范式**："训练时向critic暴露仅由prompt决定的奖励定义信息"思路可推广至代码评测（传入标准答案）、科学推理（传入公式/定理）等任务。
5. **逐步消融 sanity test 方法**：用小可解数据集（本论文用1460题，让初始模型近乎100%可解）做增量消融，是诊断RL训练bug的高效策略，值得在团队中推广。

## 关键术语表
**BPCO（Best-Practice Critic Optimization）**：本文提出的critic稳定高效训练配方，整合五项关键技术。

**DPPO（Divergence Proximal Policy Optimization）**：以绝对概率变化而非相对比率定义裁剪边界，解决PPO在LLM大词汇空间中的不公平问题。

**解耦GAE（Decoupled GAE）**：policy 和 critic 使用不同 λ 参数的 GAE 变体，本文设 λ_π=0.99、λ_V=1。

**LA-GAE（Length-Adaptive GAE）**：根据响应长度 L 自适应调整 λ_π(L)=1−1/(αL)，长响应更接近MC估计。

**Privileged Critic**：训练时向critic暴露仅在prompt层面可得的奖励定义信息（如答案/rubric），推理时策略不感知。

**Advantage Normalization**：对每个batch的估计优势度做减均值除标准差的归一化，本文证明其破坏接近最优时的自然收敛。

**蒙特卡洛价值目标**：λ_V=1 时 critic 目标直接等于最终观测回报 R(x,y)，消除 bootstrapping 偏差。

**AIME 2025 avg@32**：评估指标，每题采样32个响应取平均准确率，衡量模型数学推理能力。

## 可复现要素
- **数据集**：DeepScaleR（40.3K数学题）、DAPO-Math-17k、OpenRubrics；论文未明确说明各数据集是否完全公开，DeepScaleR和DAPO-Math可参考原论文。
- **代码**：已开源，https://github.com/QPHutu/golden_critic
- **权重**：未提及
- **关键超参**：政策学习率 1e-6，critic学习率 1e-5，minibatch 256，迭代 1500 次（sanity test）/更多（主实验），α=0.4（LA-GAE），ε=0.2（DPPO裁剪），λ_π=0.99，λ_V=1，γ=1，组大小 G=16（Dr. GRPO基线）
