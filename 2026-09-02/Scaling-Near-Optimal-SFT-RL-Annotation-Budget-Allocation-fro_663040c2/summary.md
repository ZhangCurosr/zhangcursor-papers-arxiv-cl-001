---
title: "Scaling-Near-Optimal-SFT-RL-Annotation-Budget-Allocation-fro"
source: https://arxiv.org/pdf/2609.01573v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:59:55"
field: "LLM后训练与对齐"
keywords: ["SFT-RL预算分配", "近优区域", "跨尺度迁移", "DPO", "GRPO", "LLM后训练", "数据混合优化", "scaling laws"]
innovations: ["提出近优区域概念替代单一最优比率，刻画SFT-RL预算分配的宽平台特性", "证明近优区域从小型代理模型可靠迁移至大型目标模型（ε=5%时94.3%命中率）", "建立成本不对称性框架，揭示SFT标注成本上升时分配灵活性增强的规律"]
benchmarks: ["GSM8K", "IFEval", "Reddit TL;DR", "HelpSteer", "Reward Model评分"]
---

# 论文速读：Scaling-Near-Optimal-SFT-RL-Annotation-Budget-Allocation-fro

## 一句话总结
本文研究了LLM后训练中固定标注预算在SFT与RL两阶段间的分配问题，提出以"近优区域"（near-optimal region）而非单一最优比率来刻画预算分配空间。实验发现该区域在小容忍度下即相当宽（可达可行空间的55%–75%），且随模型规模扩大进一步增宽，并能从小型代理模型可靠地迁移至大型目标模型，从而提供了一种低成本、高效的跨尺度预算分配指导策略。

## 研究问题与动机
1. **核心问题**：在LLM后训练阶段，给定固定的标注样本预算 $B$，应如何将其划分为SFT阶段（$rB$）和RL阶段（$(1-r)B$），以最大化下游任务性能？
2. **现有工作的不足**：
   - Raghavendra et al. (2025) 仅刻画了SFT在低数据区间主导、偏好优化在大规模下更有利的粗略趋势，未建立系统的分配框架。
   - 现有数据混合优化工作聚焦单阶段场景（预训练数据混合、指令微调混合等），不涉及多阶段SFT→RL的预算分配问题。
   - 最优比率是否可在不同模型规模间迁移从未被系统检验；精确最优比率因后训练对方法、任务、指标敏感，在不同尺度上可能不可靠转移。
3. **实际操作困难**：调整SFT–RL预算配比通常需要从头重训或重跑后续阶段，在全模型尺度上进行穷举网格搜索的成本极高。
4. **现实背景**：标注成本占据后训练总成本的主导地位（比计算成本高3–5个数量级），因此以标注样本数作为预算轴具有实际意义。

## 核心贡献（创新点）
1. **近优区域分析框架**：将问题从寻找单一最优分配比率重新定义为刻画"近优区域"（在指定容忍度 $\varepsilon$ 内达到峰值性能 $\geq (1-\varepsilon)\mathcal{P}^*$ 的所有分配集合），揭示了后训练性能在宽泛分配范围内对预算比例的选择并不敏感。
2. **跨尺度近优区域的可迁移性**：证明近优区域从小型代理模型到大型目标模型的迁移可靠性显著高于精确最优比率——在 $\varepsilon=5\%$ 和 $10\%$ 时，代理模型识别的区域在目标模型上分别满足94.3%和97.1%的命中率。
3. **尺度依赖的区域扩展现象**：在固定容忍度下，近优区域随模型规模增大而一致增宽；对于Llama和Qwen系列，8B模型的近优区域明显宽于1B/2B代理模型。
4. **成本不对称性的扩展分析**：当SFT标注成本相对RL更高时（如人工标注 vs. 偏好对标注，$\rho=c_\text{SFT}/c_\text{DPO} \in \{1,2,5,10\}$），近优区域进一步增宽，分配选择更加灵活。
5. **方法论验证的充分性**：通过9点网格验证、多种子验证、绝对容忍度对比、训练超参数扰动等大量消融实验，确保结论稳健。

## 方法详解
**形式化建模**：给定预训练模型 $\mathcal{M}_N$（参数量 $N$）和总标注预算 $B$，分配比率 $r \in [0,1]$ 将 $rB$ 分配给SFT、$(1-r)B$ 分配给RL阶段。目标函数为任务特定的非负性能度量 $\mathcal{P}(N, r, B; \tau)$。

**近优区域的定义**：
$$
\mathcal{R}_\varepsilon(N, B) = \left\{ r \in [0, 1] : \mathcal{P}(N, r, B) \geq (1-\varepsilon) \max_{r' \in [0,1]} \mathcal{P}(N, r', B) \right\}
$$
其中 $\varepsilon \in [0,1]$ 为相对性能容忍度（如 $\varepsilon=5\%$ 允许任何保留至少95%峰值性能的分配）。

**宽度度量**：
- **范围宽度** $w_\varepsilon^\text{rng}(N,B) = \max \widehat{\mathcal{R}}_\varepsilon - \min \widehat{\mathcal{R}}_\varepsilon$：衡量近优区域在 $[0,1]$ 上的跨度。
- **计数宽度** $w_\varepsilon^\text{cnt}(N,B) = |\widehat{\mathcal{R}}_\varepsilon| / |\mathcal{G}|$：衡量网格中近优点的比例。

**跨尺度迁移率**：
$$
T_\varepsilon(N_s \to N_t; B) = \frac{|\widehat{\mathcal{R}}_\varepsilon(N_s, B) \cap \widehat{\mathcal{R}}_\varepsilon(N_t, B)|}{|\widehat{\mathcal{R}}_\varepsilon(N_s, B)|}
$$
该指标衡量从小型代理模型 $M_{N_s}$ 识别的近优区域中，有多少比例在大型目标模型 $M_{N_t}$ 上仍然成立。

**代理分配三步流程**：
1. 使用目标家族中最小模型作为代理，在5点网格 $\mathcal{G} = \{0, 0.25, 0.5, 0.75, 1\}$ 上评估目标预算 $B$。
2. 选择容忍度 $\varepsilon \in [5\%, 10\%]$，构建近优区域 $\widehat{\mathcal{R}}_\varepsilon(N_s, B)$。
3. 取近优区域的中心点作为最终分配比率；若必须使用网格点则取最近邻。

**成本不对称扩展**：引入货币预算 $B$ 和SFT-to-DPO成本比 $\rho = c_\text{SFT}/c_\text{DPO}$，每个阶段获得样本数 $n_\text{SFT} = rB/c_\text{SFT}$ 和 $n_\text{DPO} = (1-r)B/c_\text{DPO}$。

## 实验与结果
**实验设置**：
- **模型家族**：Llama 3（1B/3B/8B）、Qwen 2.5（1.5B/7B/14B）、Qwen 3（1.7B/4B/8B）
- **任务**：数学推理（GSM8K）、指令遵循（IFEval）、摘要生成（Reddit TL;DR，ROUGE-L）、有用性（HelpSteer，Reward Model评分）
- **RL方法**：DPO（离线偏好优化）、GRPO（在线策略优化）
- **预算网格**：$\mathcal{B} \subset [0, 15\text{k}]$ 样本数；分析聚焦 $B \geq 5\text{k}$ 的稳定区间
- **训练配置**：LoRA（rank=32, α=32），学习率 SFT=$10^{-5}$、RL=$10^{-6}$，有效batch size=16，cosine schedule，adamw_8bit

**关键结果**：
- **近优区域宽度**（图1）：在 $\varepsilon=10\%$ 容忍度下，大多数任务的近优比率覆盖55%–75%的可行分配空间，显著支持了"性能平台"假设。
- **跨尺度扩展**（图2–4）：在固定容忍度下，近优区域宽度随模型规模一致增大；1B/2B代理的近优区域可可靠指导8B+目标模型。
- **迁移率量化**（Sec 3.3）：
  - 精确最优（$\varepsilon=0$）在Qwen 2.5家族的迁移表现不一致。
  - 允许5%–10%容忍度后，迁移率显著提升并保持一致。
  - 三步代理流程：在94.3%（$\varepsilon=5\%$）和97.1%（$\varepsilon=10\%$）的案例中找到满足目标模型近优区域的条件比率。
- **成本不对称性**（图6）：随着 $\rho$ 从1增至10（SFT标注成本上升），在相同容忍度下近优区域进一步增宽，分配灵活性增强。
- **GRPO扩展**（图5）：在线策略方法（GRPO）呈现与DPO相同的定性趋势，但波动略大（符合在线训练的方差特性）。
- **对比固定比率**（Tab.6）：与固定 $r=0.5$ 相比，代理方法在 $\varepsilon=5\%$ 时命中率0.90–0.95 vs. 0.20–0.60；5点网格搜索仅额外花费约100 GPU分钟（≈\$2.1），远低于在全规模上穷举的成本。

## 相关工作脉络
1. **Raghavendra et al. (2025)**：最直接相关工作，研究了SFT与偏好优化在固定标注预算下的权衡，观察到"低数据下SFT主导、大规模下偏好优化更有利"的粗略趋势。本文与其本质区别在于：不从"单一最优比率"出发，而是刻画整个近优区域并研究其跨尺度迁移性。
2. **Li et al. (2026)**：研究在固定SFT数据预算下何时停止SFT以获得最佳后续RL性能，提出自适应早停损失（AESL）。本文研究的是SFT预算相对于RL预算的比例分配，两者互补而非重叠。
3. **数据混合优化（单阶段）**：包括预训练阶段的DoReMi（Xie et al., 2023）、DOGE（Fan et al., 2024）、RegMix（Liu et al., 2025）等，以及微调阶段的Tülu配方（Ivison et al., 2023; Lambert et al., 2025）、DaMo（Shi et al., 2026）、AutoMixAlign（Corrado et al., 2025）等。这些工作均假设在同一损失面上调整混合比例，而SFT→RL跨阶段的重新分配需要从头重训后续阶段，问题本质不同。
4. **Scaling Laws研究**：Kaplan et al. (2020)、Hoffmann et al. (2022)建立了预训练阶段的scaling laws预测范式。后训练场景中scaling行为更脆弱（broken/inverse scaling），本文通过近优区域框架提供了一种更鲁棒的跨尺度外推机制。
5. **直接对齐的scaling研究**：Rafailov et al. (2024)研究了reward model过优化的scaling laws，Shen et al. (2025)研究了RLHF数据扩展现象。本文与其共同构成了后训练阶段系统性的scaling分析图谱。

## 局限性与未来方向
1. **训练设置局限于同分布**：所有实验均在任务特定数据与评估数据同分布的条件下进行；OOD（分布外）场景下性能缩放更不可靠，近优区域的迁移性在OOD设定下仍是开放问题。
2. **模型和数据规模上限**：最大分析至14B参数（排除了Llama 3 70B和Qwen 2.5 32B–72B），部分任务的预算已接近公开数据集上限（如GSM8K仅~7.5K SFT样本），前沿规模的验证需更多数据和计算资源。
3. **受限的RL算法范围**：仅测试了DPO和GRPO，未涉及PPO、SimPO等其他流行算法。
4. **成本模型简化**：仅考虑标注成本，未纳入训练计算、GPU时间和实验搜索成本的联合建模。
5. **仅限两阶段SFT→RL流水线**：未扩展到更复杂的多阶段或交错调度场景，也未考虑标注成本模型不确定或自适应的情况。

## 研究启发与可借鉴点
1. **从"精确最优"转向"近优区域"**：后训练性能曲面通常在峰值附近形成宽平台而非尖锐峰，采用近优区域而非点最优作为优化目标是更稳健的科学视角，可迁移到其他超参数/数据混合优化问题。
2. **代理模型的跨尺度指导策略**：用最小可用模型做5点快速扫描（仅需约100 GPU分钟/$2.1），即可为大型目标模型提供>90%命中率的可靠分配建议，大幅降低全规模搜索成本，可作为标准工程实践推广。
3. **容忍度的双重作用**：$\varepsilon \in [5\%, 10\%]$ 不仅提供了分配灵活性，更是跨尺度迁移可靠性的关键保障——精确最优比率在不同尺度上可能不一致，但近优区域具有良好的结构稳定性。
4. **成本不对称性的定量分析框架**：引入 $\rho = c_\text{SFT}/c_\text{RL}$ 的通用成本比框架，可灵活适配不同标注来源（人工标注、LLM蒸馏、合成数据）的实际成本差异，为工程决策提供量化依据。
5. **多粒度验证方法**：通过5点vs.9点网格验证、范围宽度vs.计数宽度双指标、相对vs.绝对容忍度对比、多种子验证、训练超参扰动等系统性消融，确保了结论的可靠性，这套验证方法论值得借鉴。

## 关键术语表
- **近优区域（Near-optimal Region）** $\mathcal{R}_\varepsilon$：在性能度量上不低于峰值 $(1-\varepsilon)$ 倍的所有分配比率构成的集合，替代单一最优比率的更稳健分析对象。
- **SFT（Supervised Fine-Tuning）**：在标注的 (prompt, response) 对上进行下一词元预测损失的微调，为后续RL提供初始策略。
- **DPO（Direct Preference Optimization）**：无需显式 reward model 的离线偏好优化方法，通过在静态偏好数据集上直接优化策略与参考模型的对比概率比来实现对齐。
- **GRPO（Group Relative Policy Optimization）**：在线策略RL方法，对每个prompt生成一组响应并通过组内归一化奖励计算相对优势，使用PPO风格的裁剪目标更新策略。
- **跨尺度迁移率（Cross-scale Transferability）** $T_\varepsilon$：衡量从小型代理模型的近优区域中识别出的分配比率，在大型目标模型上仍保持近优的比例。
- **分配比率（Allocation Ratio）** $r$：总标注预算 $B$ 中分配给SFT阶段的比例，$(1-r)$ 分配给RL阶段。
- **成本不对称性（Cost Asymmetry）**：SFT数据与RL阶段数据的单位标注成本不相等的情况，用 $\rho = c_\text{SFT}/c_\text{RL}$ 量化。
- **Range Width vs. Count Width**：两种近优区域宽度度量——前者计算区域在 $[0,1]$ 上的跨度，后者计算网格中近优点的比例，两者定性趋势一致。

## 可复现要素
- **数据集**：全部公开可用——GSM8K（SFT）、GSM8K preferences（RL）、Tülu3 Instruction Following（SFT/DPO）、GRPO-IFeval（GRPO）、Reddit TL;DR（SFT）、Reddit comparison dataset（RL）、HelpSteer1/2（SFT/RL）、IFEval（评测）、HelpSteer Reward Model（评测）。具体HuggingFace链接见论文Table 2。
- **代码**：论文未明确声明代码开源情况。
- **模型**：Llama 3、Qwen 2.5、Qwen 3 系列，均为公开预训练模型。
- **关键超参数**：
  - LoRA: rank=32, α=32, dropout=0, 目标注意力模块
  - SFT: lr=$10^{-5}$, batch=16, epochs=1 (IF=2), warmup=0.1, cosine scheduler, adamw_8bit, weight_decay=0.01
  - DPO: lr=$10^{-6}$, batch=16, epochs=1 (IF=2), warmup=0.1, cosine, adamw_8bit, weight_decay=0.01, β=0.1
  - GRPO: lr=$10^{-6}$, batch=16, epochs=1 (IF=2), warmup=0.1, cosine, adamw_8bit, weight_decay=0.01, 4 generations, temperature=0.9, top-p=1.0
- **硬件**：L40s和H200 GPU
- **随机种子**：主要实验使用seed=42；多种子验证使用3个seed
