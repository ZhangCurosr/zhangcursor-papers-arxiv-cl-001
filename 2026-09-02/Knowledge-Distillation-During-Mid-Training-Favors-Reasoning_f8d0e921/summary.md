---
title: "Knowledge-Distillation-During-Mid-Training-Favors-Reasoning"
source: https://arxiv.org/pdf/2609.01532v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:20:13"
field: "大模型训练方法与目标设计"
keywords: ["knowledge distillation", "mid-training", "reasoning", "factual recall", "language models", "reverse KL", "teacher entropy"]
innovations: ["发现mid-training下标准KD存在推理-事实记忆权衡现象", "提出基于教师预测熵的路由蒸馏目标SWITCH DISTILLATION，仅在低熵token上执行蒸馏", "建立教师置信度不对称与学生知识状态交互的解释框架"]
benchmarks: ["OLMES", "GSM8K", "MMLU", "TriviaQA", "MATH", "DROP", "BBH"]
---

# 论文速读：Knowledge-Distillation-During-Mid-Training-Favors-Reasoning

## 一句话总结
本文发现标准前向KL知识蒸馏（FKD）在中间训练阶段（mid-training）存在"推理-事实记忆权衡"：虽能提升推理能力，但会延缓事实性知识的习得。为此提出 **SWITCH DISTILLATION**，以教师预测熵作为轻量路由信号，仅在教师高置信度token上执行蒸馏，其余回退到交叉熵，在中训练阶段显著提升推理与知识性能并保留事实记忆。

## 研究问题与动机
- **Mid-training 的KD行为未被充分研究**：现代大模型训练通常分为 pre-training → mid-training → post-training 三个阶段，现有KD研究几乎全部集中于 pre-training 和 post-training，缺乏对 mid-training 阶段KD行为的系统性分析。
- **发现阶段依赖的推理-事实记忆权衡**：通过控制实验发现，标准前向KL蒸馏在 pre-training 阶段能同时改善推理和事实记忆，但在 mid-training 阶段却只提升推理而牺牲事实记忆，二者表现出现质的差异。
- **原因归因于教师置信度不对称与学生知识状态交互**：教师在程序性数据（数学、指令跟随）上熵更低（更自信），在学生尚未掌握的事实性知识上熵反而更高；KD在教师高熵区域弱化了来自语料库的真实监督信号，导致事实记忆习得更慢。
- **缺乏同时兼顾推理与事实记忆的mid-training目标**：不同教师规模、KL方向、插值系数下，没有任何现有KD目标能在mid-training中Pareto优于纯NTP。

## 核心贡献（创新点）
1. **揭示了mid-training下独特的推理-事实记忆权衡现象**：通过系统性实验发现标准FKD/RKD在mid-training阶段均无法同时改善推理和事实记忆，而在pre-training阶段可Pareto优于NTP，这是此前未被报道的行为差异。
2. **建立了教师置信度-学生知识状态-优化动力学的三重交互解释框架**：从教师域间熵不对称、事实习得的时间分层、以及KD梯度弱化三个层面定量解释了tradeoff的成因，为后续设计阶段感知目标提供了理论依据。
3. **提出SWITCH DISTILLATION，基于教师预测熵的路由蒸馏目标**：将每个token按教师熵分为低熵（路由至反向KL蒸馏）和高熵（使用交叉熵）两部分，无需额外参数和前向传播，计算开销可忽略；相比FKD/RKD/TRKD显著缓解了权衡并保留推理增益。

## 方法详解
**SWITCH DISTILLATION** 的核心设计：

1. **教师预测熵计算**（作为路由信号）：对每个token位置 $n$，计算教师分布 $p_{T,n}^{(\tau)}$ 的熵：
$$H_n = -\sum_{v \in \mathcal{V}} p_{T,n}^{(\tau)}(v) \log p_{T,n}^{(\tau)}(v)$$
其中 $\tau$ 为温度（论文取 $\tau = 2$）。

2. **按分位数路由**：设 $S_q = \{n : H_n \leq \text{Quantile}_q(\{H_n\})\}$ 为batch内熵最低的前 $q\%$ token，分配给蒸馏；其余token $\bar{S}_q$ 使用标准交叉熵。

3. **整体目标函数**：
$$\mathcal{L}^{\text{SWITCH}} = \tau^2 \cdot \frac{1}{|S_q|}\sum_{n \in S_q} \text{RKL}(p_{S,n}^{(\tau)} \| p_{T,n}^{(\tau)}) + \frac{1}{|\bar{S}_q|}\sum_{n \notin S_q} \mathcal{L}_{\text{CE},n}$$
关键选择是**使用反向KL（RKD）而非前向KL**进行蒸馏：RKD具有"模式搜索"性质，更利于强化教师低熵分布下的偏好续写。

4. **路由阈值 $q$**：默认 $q=20\%$（即对batch中熵最低的20% token执行蒸馏），实验表明 $q=20\%$ 到 $30\%$ 性能稳定，对具体取值不敏感。

5. **计算效率**：路由信号从教师logits直接派生（教师logits原本就需计算），无需额外前向传播或参数。

## 实验与结果
- **实验设置**：OLMo-2 1B学生模型，在Dolmino Mix 1124语料上分别进行100B token pre-training和60B token mid-training；教师模型为OLMo-2 1B/7B/13B Instruct。评估使用OLMES基准，包含REASONING（GSM8K、GSM-S、GSM+、BBH、DROP、MATH）、FACTUAL RECALL（TQA、NQ、SQA）、KNOWLEDGE & COMMONSENSE（MMLU、MMLU-Pro、ARC-C、OBQA、Wino、AGI）。
- **与基线对比（Table 1）**：7B教师下，SWITCH DISTILLATION推理宏平均达44.7%（NTP为26.1%，提升约**71%**），知识&常识51.6%，事实记忆仅下降1个百分点（29.3% vs NTP的30.3%）；13B教师下推理42.1%，同样接近NTP的事实记忆水平。
- **Post-training后收益持续（Table 2）**：经过完整四阶段post-training（SFT→DPO→RLVR1→RLVR2），SWITCH DISTILLATION推理相对NTP仍保持+32%，知识任务+20%，且事实记忆差距完全关闭；7B教师最终推理50.6%（vs NTP 46.8%），13B教师48.0%。
- **关键提升幅度**：相对标准NTP，mid-training后推理性能提升**1.61–1.71×**，知识与常识提升**1.13–1.19×**，同时保留96.7–96.8%的事实记忆；post-training后仍保持推理1.25–1.32×和知识1.13–1.20×的提升。
- **消融实验（Table 3）**：替换为FKL损失导致推理下降2.9%；使用教师正确性/领域知识/随机路由均显著劣于教师熵路由；始终使用CE则无增益。
- **其他发现**：7B教师整体优于13B教师，印证"教师-学生容量差距过大会降低蒸馏效果"；FKL与CE的梯度余弦相似度高于RKD，解释了二者在优化几何上的系统性差异。

## 相关工作脉络
1. **Token-Routing KD (TRKD, Goyal et al., 2026)**：针对**pre-training**阶段，对高熵token应用前向KL蒸馏以缓解对上下文学习能力的负面影响；本文定位不同——TRKD关注pre-training的上下文学习能力保护，SWITCH DISTILLATION解决mid-training的推理-事实记忆权衡，且两者路由方向相反（TRKD路由高熵，SWITCH路由低熵）。
2. **Entropy-Aware On-Policy Distillation (EOPD, Jin et al., 2026)**：在**post-training**阶段利用教师熵自适应混合前向/反向KL；本文与其概念最接近但训练阶段不同（self-supervised mid-training vs. on-policy post-training），且EOPD始终使用教师蒸馏仅调整KL方向，SWITCH则用熵决定**是否使用教师还是语料库监督**。
3. **MiniLLM (Gu et al., 2024)**：推动在LLM蒸馏中使用反向KL以增强生成能力；本文在此基础上进一步设计了熵感知的token级选择性蒸馏策略。
4. **Distillation Scaling Laws (Busbridge et al., 2025)**：刻画pre-training蒸馏的计算最优教师-学生配置；本文扩展了KD研究视角到mid-training阶段，揭示不同阶段的定性差异。
5. **On-Policy Distillation (Agarwal et al., 2024)**：基于学生自采样轨迹的对策蒸馏；本文不涉及RL，而是基于固定语料库的自我监督训练。
6. **Generative Distillation Precision-Recall Tradeoff (Cha & Cho, 2025)**：揭示蒸馏中低熵教师产生高精确但低覆盖学生的现象；本文的工作独立发现但聚焦于自监督mid-training阶段的推理-事实权衡。

## 局限性与未来方向
- 主要实验集中在OLMo-2生态，虽在SmolLM2和跨模型家族教师上验证了现象的一般性，但缺乏对其他模型架构（如MoE）的广泛验证。
- 未充分探索更好的选择性蒸馏策略：当前基于固定分位数的熵路由是最简实现，作者建议未来可探索学习式路由器（如MoE风格）。
- 仅验证了小参数学生（1B）下超Chinchilla最优的极端场景，对更大尺度学生或未超训练的情形推广性存疑。
- 作者推测SWITCH DISTILLATION也可能适用于晚期预训练（此时学生已掌握大部分易迁移知识），但未经实验验证。
- 需要中间检查点和完整训练配方才能复现各阶段对比实验，限制了其他工作组的验证广度。

## 研究启发与可借鉴点
1. **训练目标应具有阶段感知性（Stage-aware Objective Design）**：同一蒸馏目标在不同训练阶段可能产生截然不同的行为，设计training objective时不应"一刀切"，应结合学生知识状态和教师置信度进行动态调整——这一原则可扩展到continued pre-training、instruction tuning等阶段。
2. **教师预测熵作为轻量路由信号可直接迁移**：熵的计算仅需教师logits（蒸馏本身已需计算），无需额外开销；可借鉴到self-distillation、teacher-free蒸馏、或其他需要token级自适应监督强度的场景中。
3. **Gold-token梯度分析是诊断KD效果的有力工具**：论文通过推导NTP/FKD/RKD下gold-answer梯度的闭式解，量化比较了不同目标对正确token的强化力度，这一分析方法可用于诊断其他蒸馏变体的优劣势。
4. **"推理-记忆权衡"现象的可迁移性**：除mid-training外，在instruction tuning、RLVR等阶段也可能出现类似现象（教师强推理能力但弱事实锚定），可将SWITCH的设计思想延伸到这些场景中。
5. **实验验证跨教师规模的稳健性值得借鉴**：论文系统比较了1B/7B/13B三种教师规模，结论高度一致，且注意到适中容量差距更优；这对设计蒸馏实验时选择合适的教师规模有参考价值。

## 关键术语表
**Mid-training**：介于pre-training和post-training之间的自监督训练阶段，在精选的高质量语料上继续训练语言模型以激发推理、事实、代码等特定能力。
**Forward KL Distillation (FKD)**：最小化教师分布到学生分布的前向KL散度，鼓励学生覆盖教师概率质量集中的所有区域（覆盖模式）。
**Reverse KL Distillation (RKD)**：最小化学生分布到教师分布的反向KL散度，引导学生集中在教师概率最高的区域（模式搜索）。
**Predictive Entropy**：教师模型对某token位置的预测分布的熵，衡量该位置教师监督的"不确定性"或"集中程度"。
**Reasoning-Recall Tradeoff**：在mid-training下，标准KD提升推理能力但同时减缓事实性知识习得的现象，二者呈现此消彼长的关系。
**Teacher-Student Capacity Gap**：教师与学生在参数量/能力上的差距；论文发现差距过大时蒸馏效果反而下降。
**OLMES**：标准化的语言模型评估框架，将benchmark任务归类为Reasoning、Factual Recall、Knowledge & Commonsense、Instruction Following四大组别。
**Routing Quantile (q)**：SWITCH DISTILLATION中用于决定多少比例token被路由到蒸馏的分位数阈值，默认设为20%。

## 可复现要素
- **数据集**：Dolmino Mix 1124（DCLM web text、FLAN、Dolmino Math、peS2o、Wikipedia、Stack Exchange），已开源公开。
- **代码**：已开源，https://github.com/facebookresearch/midtraining-distillation
- **权重**：OLMo-2模型族为开源模型，checkpoint可在HuggingFace获取。
- **关键超参**：distillation temperature $\tau = 2$，路由阈值 $q = 20\%$，$\alpha = 0.5$（FKD/RKD基线），global batch size 2,097,152 tokens/step，pre-training LR peak $4\times10^{-4}$，mid-training LR peak $7.45\times10^{-5}$。
- **训练平台**：32× NVIDIA H200 Tensor Core GPUs（4节点）。
- **教师量化**：7B/13B教师使用FP8量化（论文称对熵排序无实质影响）。
