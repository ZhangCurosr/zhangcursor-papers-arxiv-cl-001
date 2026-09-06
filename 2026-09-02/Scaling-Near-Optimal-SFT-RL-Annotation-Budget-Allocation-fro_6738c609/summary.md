---
title: "Scaling-Near-Optimal-SFT-RL-Annotation-Budget-Allocation-fro"
source: https://arxiv.org/pdf/2609.01573v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:19:45"
field: "大语言模型后训练与对齐"
keywords: ["SFT-RL预算分配", "近优区域", "跨尺度迁移", "后训练", "DPO", "GRPO", "Scaling Law"]
innovations: ["提出近优区域概念替代单点最优比例，刻画SFT-RL预算分配的稳健解空间", "证明近优区域随模型规模扩展且可从小代理可靠迁移至大目标模型", "提出三步代理分配流程，以1B小模型低成本搜索替代大规模网格搜索"]
benchmarks: ["GSM8K", "IFEval", "Reddit TL;DR", "HelpSteer2"]
---

# 论文速读：Scaling-Near-Optimal-SFT-RL-Annotation-Budget-Allocation-fro

## 一句话总结
本文提出了**近优区域（near-optimal region）**概念，用于刻画LLM后训练中SFT与RL标注预算的最优分配范围，发现该区域在容忍度2%–10%内相当宽泛，随模型规模扩大而进一步扩展，且能从小型代理模型可靠地迁移至大型目标模型，从而提供了一种低成本、无需大规模网格搜索的预算分配策略。

## 研究问题与动机
- **核心问题**：给定固定的标注预算 B，应如何在监督微调（SFT）和强化学习（RL）两个阶段之间分配？
- **现有方法不足**：Raghavendra et al. (2025) 仅刻画了粗粒度的趋势（低数据量时SFT占优，大规模时偏好优化更优），未识别精确最优比例，也未验证跨模型规模的迁移性；其他数据混合优化工作仅针对单阶段训练。
- **挑战一**：后训练对微调方法、任务、评估指标高度敏感，且最优SFT–RL比例可能随模型规模变化而不稳定（scaling behavior fragile）。
- **挑战二**：每次调整SFT–RL分配比例通常需从头重训练或重新运行后续阶段，全规模穷举网格搜索计算成本极高。

## 核心贡献（创新点）
- **近优区域分析**：将问题从"寻找唯一最优比例"转化为"刻画近优区域"，证明即使容忍度很小（如5%），性能曲线在峰值附近也形成宽平台而非尖峰。
- **规模依赖的区域扩展**：在固定容忍度下，近优区域宽度随模型规模一般性地增大；小模型上识别的区域可可靠迁移到大模型，避免了大规模穷举搜索。
- **跨设置泛化性**：该现象在多个任务（数学推理、指令遵循、摘要、有用性）、多个模型族（Llama 3、Qwen 2.5、Qwen 3）以及两种RL方法（离策略DPO、在策略GRPO）上均一致成立，并进一步扩展到SFT与RL标注成本不对称的场景。
- **实用代理分配流程**：提出了三步代理法——用小模型做5点网格搜索→选择ε（推荐5%–10%）→取近优区域中点作为最终比例，该方法在94.3%（ε=5%）和97.1%（ε=10%）的情况下命中目标模型近优区域。

## 方法详解
- **问题形式化**：给定预训练模型 $\mathcal{M}_N$ 和总标注预算 B，分配比例 $r \in [0,1]$ 将 $rB$ 分配给SFT、$(1-r)B$ 分配给RL，评估函数为 $\mathcal{P}(N, r, B; \tau)$。
- **近优区域定义**：$\varepsilon$-近优区域 $\mathcal{R}_\varepsilon(N, B) = \{ r \in [0,1] : \mathcal{P}(N, r, B) \geq (1-\varepsilon) \max_{r'} \mathcal{P}(N, r', B) \}$，即性能不低于峰值 $(1-\varepsilon)$ 的所有分配比例集合。
- **宽度度量**：采用范围宽度 $w_\varepsilon^{\text{rng}} = \max \widehat{\mathcal{R}}_\varepsilon - \min \widehat{\mathcal{R}}_\varepsilon$ 和计数宽度 $w_\varepsilon^{\text{cnt}} = |\widehat{\mathcal{R}}_\varepsilon| / |\mathcal{G}|$ 两种估计器。
- **跨尺度迁移性度量**：$T_\varepsilon(N_s \to N_t; B) = |\widehat{\mathcal{R}}_\varepsilon(N_s, B) \cap \widehat{\mathcal{R}}_\varepsilon(N_t, B)| / |\widehat{\mathcal{R}}_\varepsilon(N_s, B)|$，衡量小代理模型近优区域中有多少比例在大目标模型上仍为近优。
- **成本不对称扩展**：引入SFT与RL标注成本比 $\rho = c_{\text{SFT}} / c_{\text{DPO}}$，分析 $\rho \in \{1, 2, 5, 10\}$ 下近优区域的变化规律。
- **实验设置**：使用LoRA（rank=32）进行参数高效微调，网格 $\mathcal{G} = \{0, 0.25, 0.5, 0.75, 1.0\}$，预算 $B \in [0, 15\text{k}]$，聚焦 $B \geq 5\text{k}$ 的稳定训练区间。

## 实验与结果
- **数据集与任务**：数学（GSM8K）、指令遵循（Tülu3 Persona IF + IFEval评测）、摘要（Reddit TL;DR + ROUGE-L）、有用性（HelpSteer/HelpSteer2 + Reward Model），覆盖Llama 3（1B–8B）、Qwen 2.5（1.5B–14B）、Qwen 3（1.7B–8B）三族模型。
- **基线对比**：主要对比 Raghavendra et al. (2025) 的趋势性结论，以及固定 $r=0.5$ 的简单启发式策略。
- **关键结果**：
  - **区域宽度**：10%容忍度下，近优区域覆盖55%–75%的可行分配空间；2%–10%容忍度内区域均显著宽于单点最优。
  - **规模扩展**：固定容忍度下，近优区域宽度随模型规模一般性地增大（如Llama Math 1B→8B，$\varepsilon=10\%$ 从窄区域扩展到覆盖约半空间）。
  - **跨尺度迁移**：在 $\varepsilon=5\%$ 时代理迁移命中率94.3%，$\varepsilon=10\%$ 时达97.1%；相比之下，精确最优比例（$\varepsilon=0$）的迁移一致性较差，尤其在Qwen 2.5族上。
  - **成本不对称**：随着 $\rho$ 增大（SFT相对更贵），近优区域进一步扩展，说明分配灵活性更高。
  - **代理 vs 固定比例**：代理法在 $\varepsilon=5\%$ 下比固定 $r=0.5$ 显著提升命中率（如Llama HelpSteer：0.95 vs 0.20）；且代理搜索（1B五点搜索+8B单次运行，~200 GPU-min）比全规模五点网格搜索（~500 GPU-min）节省2.5×计算量。
  - **GRPO扩展**：在策略方法（GRPO）下定性规律与DPO一致，但波动略强。

## 相关工作脉络
- **Raghavendra et al. (2025)**：最相关的先验工作，研究SFT与偏好优化的预算权衡，发现SFT在低数据量占优的趋势；本文在其基础上引入近优区域概念和跨尺度迁移性分析，填补了"最优比例是否存在且可迁移"的空白。
- **Li et al. (2026)**：研究给定固定SFT预算下的SFT早停问题（AESL方法），决策变量是SFT停止步数；本文关注SFT预算份额相对RL的分配，两者互补。
- **数据混合优化（Pretraining）**：如DoReMi (Xie et al., 2023)、Scaling-law外推 (Liu et al., 2025; Ye et al., 2025) 等，仅针对单阶段数据混合优化，不涉及跨阶段SFT–RL分配。
- **数据混合优化（Fine-tuning）**：如Tülu recipes (Ivison et al., 2023; Lambert et al., 2025)、AutoMixAlign (Corrado et al., 2025)、DaMo (Shi et al., 2026) 等，均在单一后训练阶段内进行数据组成优化，不处理SFT→RL两阶段预算分配。
- **Pretraining-Finetuning权衡**：Bai et al. (2021)、Kang et al. (2023) 等研究预训练与微调之间的预算 trade-off；本文聚焦于后训练内部SFT与RL两阶段的分配。
- **Scaling Laws 跨尺度迁移**：Yang et al. (2021) 证明在maximal-update parameterization下超参可零样本迁移；本文通过实证发现后训练中近优区域而非精确最优比例更具跨尺度稳健性。

## 局限性与未来方向
- **仅限两阶段SFT→RL流水线**：未扩展到更复杂的多阶段或交替调度方案。
- **分布内训练设定**：所有实验在SFT和RL数据与评估数据同分布的条件下进行，OOD（分布外）场景下近优区域是否同样宽泛且可迁移仍是开放问题。
- **模型与数据规模上限**：最大仅到14B（70B、32B–72B未测试），部分数据集的预算已接近公开训练集上限。
- **RL算法覆盖有限**：仅测试DPO和GRPO，未涉及PPO、SimPO等其他主流算法。
- **成本模型简化**：仅考虑标注成本，未纳入训练计算成本、GPU时间和实验搜索成本。

## 研究启发与可借鉴点
- **近优区域作为稳健优化目标**：将优化目标从"寻找精确最优"转向"刻画近优区域"，可有效规避后训练性能敏感性和非单调性问题，这一思路可迁移到其他超参/数据混合优化场景。
- **小代理模型指导大模型配置**：利用1B–2B小模型进行低成本网格搜索，即可为8B–14B目标模型提供可靠的分配建议，显著提升大规模实验效率；该策略可推广到学习率、LoRA rank等其他超参的跨尺度迁移。
- **成本不对称分析框架**：引入标注成本比 $\rho$ 统一刻画SFT与RL的数据成本差异，为实际工程中的预算决策提供了更贴近现实的建模方式。
- **实验设计借鉴**：采用范围宽度和计数宽度双估计器交叉验证近优区域的连续性；通过heatmap可视化per-ratio hit rate，直观展示分配比例的稳健性分布。
- **与团队方向结合机会**：可探索将近优区域概念应用于多任务混合配比、RLHF vs DPO vs GRPO的方法选择、以及跨任务共享代理搜索策略等方向。

## 关键术语表
- **近优区域（Near-optimal Region）**：性能不低于峰值 $(1-\varepsilon)$ 的所有分配比例构成的集合，代替单一最优比例作为优化目标。
- **SFT（Supervised Fine-Tuning）**：监督微调，使用标注的(prompt, response)对通过next-token prediction损失适配预训练模型。
- **DPO（Direct Preference Optimization）**：直接偏好优化，一种无需显式奖励模型的离策略RL方法，在静态偏好对上优化策略。
- **GRPO（Group Relative Policy Optimization）**：群体相对策略优化，一种在策略RL方法，对每个prompt生成多个response后通过组内归一化奖励计算优势进行PPO式更新。
- **跨尺度迁移性（Cross-scale Transferability）**：小代理模型上识别的近优区域在大目标模型上仍保持有效的程度，用交集比例 $T_\varepsilon$ 量化。
- **成本不对称（Cost Asymmetry）**：SFT数据与RL数据在标注成本上的差异，用比率 $\rho = c_{\text{SFT}} / c_{\text{RL}}$ 表征。
- **Range Width / Count Width**：近优区域宽度的两种估计量，前者衡量连续跨度，后者衡量网格点占比。
- **Proxy Sweep**：用小模型在离散分配比例网格上进行低成本实验搜索，以指导大模型的预算分配决策。

## 可复现要素
- **数据集**：GSM8K、Tülu3 Persona IF、Tülu3 RLVR IFeval、Reddit TL;DR、HelpSteer/HelpSteer2，均为公开数据集（HuggingFace链接见论文Table 2）。
- **代码**：论文未提及开源代码仓库。
- **权重**：使用Llama 3、Qwen 2.5、Qwen 3公开预训练权重；实验使用LoRA（rank=32, alpha=32）微调，未单独发布适配器权重。
- **关键超参**：SFT学习率 $1 \times 10^{-5}$，DPO/GRPO学习率 $1 \times 10^{-6}$，batch size=16，epochs=1（指令遵循2），warmup=0.1，cosine scheduler，adamw_8bit，weight decay=0.01，DPO的$\beta=0.1$，GRPO的group size=4、temperature=0.9、top-p=1.0。
- **硬件**：L40s和H200 GPU。
