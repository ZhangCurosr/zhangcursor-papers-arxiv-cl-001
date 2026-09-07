---
title: "Rethinking-On-Policy-Distillation-of-Large-Language-Models-I"
source: https://arxiv.org/pdf/2609.04172v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:22:02"
field: "大语言模型后训练与知识蒸馏"
keywords: ["on-policy distillation", "one-shot training", "data efficiency", "state coverage", "multi-teacher OPD", "LLM post-training"]
innovations: ["提出One-Shot OPD极端实验揭示数据效率极限", "定义状态覆盖率度量训练数据对全量状态空间的覆盖程度", "揭示OPD'数据过剩但算法饥饿'的双重机制"]
benchmarks: ["MATH-500", "AMC 2023", "AIME 2025", "LiveCodeBench v6", "Multi-IF", "BFCL v3"]
---

# 论文速读：Rethinking-On-Policy-Distillation-of-Large-Language-Models-II

## 一句话总结
本文通过将训练数据缩减至极限——仅用单个查询进行 On-Policy Distillation (OPD)，揭示大模型后训练中"数据过剩但算法饥饿"的本质：一个查询即可覆盖全量数据 71.5% 的状态空间，16 个语义多样的查询即可匹配全量 OPD 性能。

## 研究问题与动机
1. **现有研究的空白**：OPD 的算法行为已被系统研究，但训练数据在其中的作用尚未被探索，数据与算法的交互机制不清。
2. **核心疑问**：为何 OPD 在仅使用单个查询的情况下，仍能持续学习数百步并实现显著改进？
3. **现象来源**：受 One-shot RLVR（单样本强化学习验证）的启发，作者将相同实验视角引入 OPD，发现两者在学习曲线形态上高度相似。
4. **研究价值**：解耦数据供给与算法吸收效率，理解 OPD 数据效率的底层机制。

## 核心贡献（创新点）
1. **提出 One-Shot OPD 极端实验设置**：首次系统研究 OPD 在单查询训练下的行为，揭示其数据效率的极限潜力，与 RLVR 形成对照。
2. **定义并量化"状态覆盖率"(State Coverage)**：通过语义聚类度量训练数据覆盖的全量状态空间比例，证明单查询可达 71.5%，16 查询达 98.9%。
3. **揭示 OPD"数据过剩、算法饥饿"的双重机制**：从数据侧（状态覆盖充分）和算法侧（吸收率持续下降）解释为何少量数据即可产生大量增益。
4. **拓展至多教师 OPD (MOPD)**：证明在每个领域仅需 16 个语义多样化的查询即可匹配全量 MOPD 性能。
5. **挑战"任务内容必要性"假设**：无内容模板和域外 WildChat 查询同样能驱动有效 OPD，说明输入的核心价值在于诱导推理状态而非内容本身。

## 方法详解
1. **OPD 基础框架**：
   - 学生模型 $\pi_\theta$ 采样轨迹，教师模型 $\pi_T$ 在每个访问状态 $s_i = (x, y_{<i})$ 提供完整 next-token 分布作为稠密监督信号。
   - 损失函数为 per-token KL 散度：$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}[\sum_{i=1}^{L} \text{KL}(\pi_\theta(\cdot|s_i) || \pi_T(\cdot|s_i))]$
   - 两种优势估计：采样 token 优势（Eq. 2）和 top-k 优势（Eq. 3，取学生最高概率 k 个 token 加权平均）。

2. **状态覆盖率度量**：
   - 状态表示：教师最终层隐向量 $h_T(s)$
   - 参考空间：全量 OPD 在 DAPO-Math-17K 上访问的状态池（留出部分用于评估）
   - 聚类：PCA 降维后 $\kappa$-means 分为 $K=200$ 个簇
   - 覆盖率公式：$\text{Cov}(S) = \frac{1}{K}|\{c(s): s \in S\}|$

3. **对齐动态度量**：
   - 距离 $d_t$：教师-学生 per-token 优势绝对值的平均
   - 吸收率 $\nu_t = \frac{d_t - d_{t+1}}{d_t}$：每次更新消除的剩余差距比例
   - 归一化距离 $d_t / d_{30}$：消除初始距离差异，比较收敛曲线形状

4. **实验配置**：
   - 四领域：数学推理、代码生成、指令遵循、智能体工具使用
   - 三模型族：DeepSeek-R1-Distill-Qwen、Llama、OLMo
   - 超参：batch size 64, lr $10^{-6}$, rollout temperature 1.0, gradient clip norm 1.0

## 实验与结果
1. **One-Shot OPD 主要结果**：
   - 数学推理：单查询在 300 步时恢复 69% 教师-学生差距，87% 全量 OPD 增益；1000 步时恢复 72% 全量增益（68.4 vs 72.1）
   - 跨模型族：R1-Distill-1.5B、Llama-3B-It、OLMo-7B-It-DPO 均验证该效应
   - 跨领域：代码生成恢复 73%、指令遵循 66%、智能体工具使用 64%

2. **状态覆盖率结果**：
   - 单查询：71.5%（前 100 步已达 65.9%）
   - 16 查询：98.9%，匹配全量训练
   - 多样化 vs 数量：16 个单簇查询仅达 76.8% 覆盖率，远低于 16 个跨簇查询的 98.9%

3. **对齐动力学结果**：
   - 吸收率随训练持续下降，且与查询数量无关（1/4/16/全量均在相似速率下降）
   - 固定状态实验：off-policy 训练（固定 64 条轨迹）同样需数百步才收敛
   - 学习率仅缩放时间轴，不改变收敛曲线形状

4. **多教师 OPD (MOPD)**：
   - 16-shot MOPD 平均准确率达 52.9%，超过 full-data MOPD 的 52.8（101% 恢复）
   - 各领域分别恢复：数学 93%、代码 136%、指令 109%

5. **极端输入实验**：
   - 无内容模板和域外 WildChat 查询逼近真实查询基线（差距约 1 分）
   - 对比 RLVR：OPD 1000 步增益是 RLVR 的两倍以上，因稠密 token 级监督在解题后仍持续存在

## 相关工作脉络
1. **MiniLLM [Gu et al., 2024] / GKD [Agarwal et al., 2024]**：早期 OPD 算法奠基工作，关注目标函数与优化几何，未涉及数据效率问题。
2. **One-shot RLVR [Wang et al., 2026a]**：同构思想源头，证明单样本在稀疏奖励 RL 中的有效性，本文将其拓展至稠密监督 setting。
3. **OPD 动力学分析 [Li et al., 2026, Fu et al., 2026, Cai et al., 2026]**：算法侧解释 OPD 为何有效，但假设训练集给定，未解耦数据角色。
4. **数据高效推理后训练 [LIMR, LIMA, LIMO, s1]**：关注少样本 SFT/RLVR 的有效性，但监督仍依赖精心策划的任务，与 OPD 的稠密状态级监督不同。
5. **合成/无监督后训练数据 [Magpie, Zero, Self-Instruct]**：减少人工数据依赖，本文从"状态生成器"视角进一步弱化输入内容必要性。
6. **Token Teachability [Wang et al., 2026b]**：研究哪些 token 可学，与本文"状态覆盖"视角互补：前者关注监督内容质量，后者关注监督空间广度。

## 局限性与未来方向
1. **状态覆盖率是语义代理指标**：基于全量 OPD 参考空间构建，未考虑各簇访问频率和教师信号强度差异。
2. **吸收率决定机制未明**：虽观察到吸收率下降，但其根源（参数空间几何、优化动力学等）仍需进一步解释。
3. **MOPD 扩展有限**：仅验证 3 个领域 3 个教师，更多教师/领域的推广性待检验。
4. **规模限制**：实验限于 1.5B-7B 模型，大模型行为可能不同。
5. **未来方向**：(1) 基于查询自身估算状态覆盖率作为选择标准；(2) 通过 trust region 或多 epoch 复用提升步效率；(3) 扩展至更大模型、更多教师、长上下文场景。

## 研究启发与可借鉴点
1. **极端实验方法论**：将数据缩减至极限（单样本）可有效解耦数据与算法的贡献，值得在其他后训练方法中推广。
2. **状态覆盖率为数据选择提供新标准**：从"收集多少问题"转向"诱导哪些状态"，可设计基于状态多样性的自动数据筛选 pipeline。
3. **固定状态 ablation 设计**：off-policy 固定轨迹实验有效分离"状态新鲜度"与"算法吸收率"的贡献，是分析 on-policy 方法的经典对照。
4. **输入作为"状态生成器"的视角**：弱化任务内容必要性，启发利用通用对话数据或模板进行高效 OPD 训练。
5. **吸收率作为训练效率瓶颈的度量**：为后续研究 OPD 步效率提供直接诊断指标，可指导算法改进。

## 关键术语表
- **On-Policy Distillation (OPD)**：学生模型采样自身轨迹，教师在每个访问状态提供稠密 token 级分布监督的对齐方法。
- **One-Shot OPD**：仅使用单个查询进行 OPD 训练的极端设置，用于隔离数据角色的控制实验。
- **State Coverage（状态覆盖率）**：训练数据 rollouts 覆盖的全量 OPD 状态空间的语义簇比例。
- **Absorption Rate（吸收率）**：每次参数更新消除的教师-学生差距比例，衡量算法学习效率。
- **Teacher-Student Gap Recovery**：学生当前性能相对初始差距的恢复百分比，标准化跨设置比较。
- **Multi-Teacher OPD (MOPD)**：单一学生模型在多领域训练，每个查询路由至对应领域教师的 OPD 变体。
- **Token-level Advantage**：教师与学生 log-probability 差异，作为 per-token 监督信号替代 RLVR 的轨迹级奖励。
- **Content-light Template**：无实质任务内容的输入模板（如仅含 `<|thought|>`），用于检验状态诱导而非内容的重要性。

## 可复现要素
- **数据集**：DAPO-Math-17K、Open-R1 Codeforces、UltraData-SFT-2605 子集、xLAM-function-calling-60K、WildChat
- **代码开源**：是（https://github.com/Thinking-Space/One-Shot-OPD）
- **模型权重**：使用公开模型（DeepSeek-R1-Distill-Qwen-1.5B、Llama-3.2-3B-Instruct、OLMo-3-7B-Instruct-DPO 等）
- **关键超参**：batch size=64, lr=$10^{-6}$, temperature=1.0, gradient clip norm=1.0, KL coefficient=0.0, top-k=16（数学）/0（其他）
