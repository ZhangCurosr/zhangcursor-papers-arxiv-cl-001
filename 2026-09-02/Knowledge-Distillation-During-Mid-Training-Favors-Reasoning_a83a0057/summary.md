---
title: "Knowledge-Distillation-During-Mid-Training-Favors-Reasoning"
source: https://arxiv.org/pdf/2609.01532v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:19:47"
field: "大语言模型训练优化"
keywords: ["Knowledge Distillation", "Mid-training", "Reasoning", "Factual Recall", "Teacher Entropy", "Token Routing"]
innovations: ["发现 mid-training 阶段知识蒸馏的推理-回忆权衡现象", "提出基于教师预测熵的路由机制 SwITCH DISTILLATION", "揭示教师置信度不对称性与学生学习状态的交互机制"]
benchmarks: ["OLMES", "GSM8K", "BBH", "MMLU", "TriviaQA", "Natural Questions"]
---

# 论文速读：Knowledge-Distillation-During-Mid-Training-Favors-Reasoning

## 一句话总结
本文发现知识蒸馏在 mid-training 阶段存在“推理-回忆权衡”（reasoning-recall tradeoff），并提出 SwITCH DISTILLATION 方法，利用教师预测熵作为路由信号，仅在教师置信度高的 token 上进行蒸馏，其余使用交叉熵，从而在大幅提升推理能力的同时保留事实回忆。

## 研究问题与动机
1. **mid-training 阶段知识蒸馏行为尚不明确**：现代 LLM 训练包含 pre-training、mid-training 和 post-training 三个阶段，但知识蒸馏（KD）的研究主要集中在 pre-training 和 post-training，对 mid-training 阶段的影响缺乏系统理解。
2. **现有 KD 方法在 mid-training 中存在权衡问题**：标准前向 KL 蒸馏在 pre-training 中可同时提升推理和事实回忆，但在 mid-training 中却以牺牲事实回忆为代价换取推理提升，这一现象未被充分揭示。
3. **训练效率需求**：mid-training 使用的 token 数量远少于 pre-training，如何从每个 token 中提取更多学习信号变得尤为重要，而 KD 提供了增强监督信号的天然途径。

## 核心贡献（创新点）
1. **发现 mid-training 中独特的推理-回忆权衡现象**：通过控制实验揭示标准 KD 在 mid-training 阶段的行为与 pre-training 截然不同，推理提升伴随事实回忆下降。
2. **提供机制解释**：将权衡归因于教师置信度的不对称性（程序性数据 vs 知识密集型数据）与学生知识状态的交互作用，高熵事实因未学而在 mid-training 初期受到较弱的教师监督。
3. **提出 SwITCH DISTILLATION 方法**：设计一个基于教师预测熵的路由机制，将低熵 token 路由到反向 KL 蒸馏，高熵 token 保持交叉熵，该方法计算开销极小且可显著改善权衡。

## 方法详解
**SwITCH DISTILLATION** 的核心设计：

1. **教师预测熵计算**：对每个 token 位置 $n$，计算教师分布的熵：
   $$H_n = -\sum_{v \in \mathcal{V}} p_T^{(\tau)}(v|x_{<n}) \log p_T^{(\tau)}(v|x_{<n})$$

2. **基于熵的路由策略**：选取 batch 中熵最低的 $q\%$ tokens 构成集合 $S_q$，这些 token 被路由到蒸馏目标；其余 tokens 保留交叉熵监督。

3. **损失函数**：
   $$\mathcal{L}^{\text{SWITCH}} = \tau^2 \frac{1}{|S_q|} \sum_{n \in S_q} \text{RKL}(p_{S,n}^{(\tau)} \| p_{T,n}^{(\tau)}) + \frac{1}{|\bar{S}_q|} \sum_{n \notin S_q} \mathcal{L}_{\text{CE},n}$$

4. **设计要点**：
   - 使用反向 KL（RKL）而非前向 KL，因为 RKL 的 mode-seeking 特性更适合低熵、尖锐的教师分布
   - 路由信号来自教师 logits，无需额外前向传播或参数
   - 默认路由比例 $q = 20\%$，在不同 teacher 规模下表现稳定

## 实验与结果
**实验设置**：
- 学生模型：OLMo-2 1B
- 教师模型：OLMo-2 7B Instruct、13B Instruct
- 数据：Dolmino Mix 1124（包含 web text、instruction-following、math、Wikipedia 等）
- 评估基准：OLMES 标准评测套件，涵盖 REASONING、FACTUAL RECALL、KNOWLEDGE & COMMONSENSE 三大类

**主要结果**（mid-training 后）：
| 方法 | Reasoning (avg) | Factual Recall (avg) | Knowledge & Commonsense (avg) |
|------|-----------------|---------------------|------------------------------|
| NTP | 26.1% | 30.3% | 43.6% |
| FKD (α=0.5, 7B teacher) | 30.6% | 29.8% | 49.5% |
| RKD (α=0.5, 7B teacher) | 31.6% | 29.4% | 50.1% |
| **SwITCH DISTILLATION (7B)** | **44.7%** | **29.3%** | **49.3%** |
| **SwITCH DISTILLATION (13B)** | **42.1%** | **29.3%** | **46.5%** |

- **推理性能**：相对 NTP 提升 **71%**（7B teacher）和 **61%**（13B teacher）
- **事实回忆**：仅下降约 1 个百分点，显著优于其他 KD 方法
- **后训练持续性**：经过完整 post-training pipeline（SFT + DPO + RLVR）后，推理仍保持 32% 提升，知识类任务提升 20%，事实回忆差距完全消除

## 相关工作脉络
1. **标准知识蒸馏（Hinton et al., 2015）**：本文聚焦 logit-based KD，区别于 sequence-level hard distillation（Kim & Rush, 2016）
2. **预训练中的 KD 研究**：Busbridge et al. (2025) 推导了 LLM 蒸馏的 scaling laws；Goyal et al. (2026) 提出 TRKD 缓解 in-context learning 退化，但针对 pre-training 阶段
3. **后训练中的 on-policy 蒸馏**：Agarwal et al. (2024) 的 on-policy distillation 和 Jin et al. (2026) 的 EOPD 均使用教师熵，但 EOPD 始终应用教师监督，仅在高低熵间切换 KL 方向；本文在 mid-training 阶段通过熵决定"是否使用教师"
4. **Mid-training 研究**：Liu et al. (2026)、Tan et al. (2026) 等强调 mid-training 对后续 RL 的促进作用，但未系统研究 KD 在此阶段的效果
5. **数据效率方法**：Xie et al. (2023) 的 Doremi、Maini et al. (2024) 等方法优化训练数据质量，本文从算法角度提升单 token 学习效率

## 局限性与未来方向
1. **模型规模限制**：学生模型仅为 1B 参数，虽在更大规模上验证了现象，但需进一步扩展验证
2. **实验生态单一**：控制实验主要基于 OLMo-2，尽管补充了 SmolLM2 和跨家族教师分析，但完整 stack（模型权重+中间 checkpoint+训练配方）的获取限制了更广泛的验证
3. **路由策略简单**：当前使用固定阈值路由，未来可探索基于 MoE 架构的可学习路由网络
4. **未探索 late-stage pre-training**：作者假设该方法可能在 late-stage pre-training 中同样有益，但未系统验证

## 研究启发与可借鉴点
1. **Stage-aware 优化理念**：不同训练阶段应采用不同的优化目标，而非 stage-agnostic 的通用方法，这一原则可推广至其他训练阶段
2. **熵作为路由信号的有效性**：利用教师预测熵进行 token-level 路由是一种简单高效的机制，可迁移至其他需要选择性监督的场景
3. **梯度分析揭示现象本质**：通过 gold-token gradient 分析揭示 KD 在不同熵 quintile 上的监督强度差异，为理解训练动态提供了可复用的分析方法
4. **实验设计的对照性**：严格控制 pre-training 和 mid-training 的实验条件，使用同一学生模型的中间 checkpoint 进行对比，为阶段效应研究提供了良好的实验范式

## 关键术语表
**Mid-training**：位于 pre-training 和 post-training 之间的训练阶段，在高质量精选语料上继续进行自监督 next-token prediction，以增强特定能力
**Reasoning-recall tradeoff**：在 mid-training 中，标准 KD 提升推理能力但减缓事实回忆学习的现象
**Teacher predictive entropy**：教师模型对每个 token 预测的不确定性度量，低熵表示高置信度，用于路由决策
**Forward KL (FKL)**：$\text{KL}(p_T \| p_S)$，要求学生分布覆盖教师分布的所有模式，倾向于 spreading mass
**Reverse KL (RKL)**：$\text{KL}(p_S \| p_T)$，要求学生分布集中在教师分布的高概率区域，具有 mode-seeking 特性
**Token-routing KD (TRKD)**：Goyal et al. (2026) 提出的方法，对高熵 token 应用 FKL 蒸馏，与本文的"低熵路由到蒸馏"策略相反
**OLMES**：标准化的 LLM 评估 harness（Gu et al., 2025），将 benchmark 任务分为 REASONING、FACTUAL RECALL 等类别

## 可复现要素
- **数据集**：Dolmino Mix 1124（公开），评估基准使用 OLMES harness 公开任务
- **代码**：开源，位于 https://github.com/facebookresearch/midtraining-distillation
- **模型权重**：OLMo-2 系列模型公开可用
- **关键超参**：
  - 蒸馏温度 $\tau = 2$
  - 路由比例 $q = 20\%$（默认，在 10%-30% 范围内稳定）
  - 学生初始化：OLMo-2 1B Stage 1 checkpoint（4T tokens pre-trained）
  - Mid-training token 数：60B
  - Teacher 量化：7B/13B 使用 FP8，1B 使用 BF16
