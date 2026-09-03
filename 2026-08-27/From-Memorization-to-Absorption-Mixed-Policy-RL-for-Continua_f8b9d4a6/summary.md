---
title: "From-Memorization-to-Absorption-Mixed-Policy-RL-for-Continua"
source: https://arxiv.org/pdf/2608.25243v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:33:09"
---

# 论文速读：From-Memorization-to-Absorption-Mixed-Policy-RL-for-Continua

## 一句话总结
提出 GRIN 框架与 Golden-GRPO 混合策略强化学习算法，通过三阶段自学习流程将新知识从“表面记忆”转化为“参数化内化”，在保持基础事实召回的同时，显著提升 LLM 在多源检索、推理与反事实信念覆盖上的泛化能力。

## 研究问题与动机
- **SFT 记忆的格式绑定缺陷**：现有持续知识注入方法主要依赖监督微调，模型仅在训练问答对的表面句式上记忆事实，遇到语义改写、多段拼接或需要组合推理的提问时性能急剧下降。
- **旧有参数化信念的干扰未显式建模**：SFT 未处理模型已持有的过时或错误先验，新事实注入时旧知识易在推理阶段再次浮现，导致覆盖不可靠。
- **纯 on-policy RL 在无先验知识时梯度消失**：当基础模型尚未掌握某条事实时，其 rollout 几乎全错，优势函数趋近于零，RL 阶段无法获得有效的学习信号。
- **现有混合策略 RL 的离策略梯度坍缩**：LUFFY、GOLF 等方法依赖重要性采样比率放大离策略轨迹贡献，但在知识注入场景下 base policy 对黄金答案的概率极低，比率直接 collapse，梯度无法驱动参数更新。

## 核心贡献（创新点）
1. **提出 GRIN 三阶段自监督知识注入框架**。**本质区别**：完全摒弃外部教师模型与人工标注，仅凭基础模型自身完成原子事实提取、语料条件化问题采样与强化学习闭环，区别于依赖人工 SFT 数据或外部引导信号的前续注入工作。
2. **设计 Golden-GRPO 混合策略 RL 算法**。**本质区别**：以“优势值缩放直接监督梯度”替换重要性采样离策略分支，从根本上消除了传统混合策略 RL 在未知事实上因 base policy 概率极低而导致的梯度坍缩问题。
3. **构建 BLANK 与 COUNTER 正交双基准及三层评估协议**。**本质区别**：首次将“新事实获取”与“反事实信念覆盖”解耦评估，并引入 fail@k 量化先验泄露，弥补了以往工作仅报告聚合准确率而掩盖“记忆 vs 吸收”差异的评估盲区。
4. **实证确立 RL 吸收优于 SFT 记忆的认知规律**。**本质区别**：在严格匹配计算预算（基线延长至 50 epoch）与消融验证下证明性能提升源于奖励目标与梯度设计而非算力堆叠，驳斥了训练轮数导致的性能假象。

## 方法详解
- **Stage 1: 自提取事实注入（SFT）**
  将目标语料 $D$ 按句子切分为 $S(d_i)$，用固定提示 $P_{ext}$ 引导基础模型 $\pi_\theta$ 生成原子事实问答对 $F_{i,j}$，聚合后以标准交叉熵损失微调：
  $\mathcal{L}_1(\theta) = -\mathbb{E}_{(q,a)\sim F_1} \log \pi_\theta(a|q)$。此阶段为模型提供参数的初始事实接入，但仅产生脆弱的格式绑定记忆。
- **Stage 2: 语料条件化采样**
  在每个语料 $d_i$ 前拼接系统提示与预查询 token $T_{query}$ 采样问题，再拼预回答 token $T_{ans}$ 生成黄金答案 $a^\star$：
  $q \sim \pi_\theta(T_{query} \oplus d_i),\ a^\star \sim \pi_\theta(T_{ans} \oplus d_i \oplus q)$。优先保证跨句覆盖度与表述多样性，容忍一定噪声，因为 RL 奖励函数会过滤劣质样本。
- **Stage 3: Golden-GRPO 训练**
  - **Rollout 构建**：对每个 $(q, a^\star)$，组合 $N_{on}$ 个 on-policy 轨迹 $\tau_i \sim \pi_{\theta_{old}}(\cdot|q)$ 与 1 条 off-policy
