---
title: "Boosting-LLM-Exploration-via-Weak-Model-Guidance-in-RLVR"
source: https://arxiv.org/pdf/2608.27420v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:50:51"
field: "大语言模型推理增强"
keywords: ["RLVR", "GRPO", "entropy collapse", "pass@k", "prefix guidance", "exploration", "reinforcement learning"]
innovations: ["通过弱模型前缀引导RLVR训练中的探索空间扩展", "基于熵动态的自适应前缀截断策略", "揭示前缀质量与扰动效用的解耦现象"]
benchmarks: ["MATH", "AIME 2024", "AIME 2025", "AMC 2023", "Minerva", "Olympiad Bench"]
---

# 论文速读：Boosting-LLM-Exploration-via-Weak-Model-Guidance-in-RLVR

## 一句话总结
本文提出了一种基于弱模型前缀引导的RLVR框架，通过将较小模型的局部推理轨迹作为前缀注入训练过程，强制目标大模型探索偏离其高置信度分布的推理路径，有效缓解了GRPO训练中的策略熵坍缩，显著提升了pass@k（尤其是大k值）性能。

## 研究问题与动机
1. **RLVR训练中的熵坍缩问题**：GRPO训练初期策略熵急剧下降，模型对少数高奖励推理路径过度自信，导致推理覆盖范围收窄，pass@k性能随k增大反而劣化。
2. **现有方法忽视跨模型多样性**：当前缓解entropy collapse的方案（熵正则化、梯度校准、奖励设计）均在目标模型自身搜索空间内运作，未利用跨模型生成多样性这一正交信号。
3. **评估指标片面**：主流研究聚焦pass@1准确率，忽视推理覆盖率（reasoning coverage），导致RLVR训练后模型在多次采样场景下表现下降。
4. **前缀质量与探索效用解耦**：实验发现弱模型生成的前缀多数"无实质指导"或"误导"，但目标模型仍能从中恢复并扩展探索空间，提示前缀的价值在于扰动而非正确性。

## 核心贡献（创新点）
1. **提出Prefix-Completion RLVR框架**：通过辅助弱模型生成部分推理前缀，引导目标模型在陌生起点上完成推理，将跨模型生成多样性转化为非参数化探索信号；与知识蒸馏的本质区别在于使用弱模型而非强模型，且不追求前缀的准确性。
2. **设计熵自适应前缀截断策略**：基于目标基础模型对辅助前缀的逐步熵动态，定位熵值急剧下降的转折点（Equation 5），自动确定最优前缀长度以最大化早期不确定性；区别于随机截断或固定长度截断。
3. **揭示前缀质量与扰动效用的解耦现象**：发现Gemma-2-2B虽生成低质量前缀（多数为"无指导"或"误导"），但因跨模型分布差异大而探索增益更大；同源模型（Qwen系列）前缀语义质量更高却因分布相似而扰动效用低。
4. **提供系统的RLVR探索动力学分析**：从策略熵、奖励信号稀疏性、零奖励比例等多维度刻画prefix-guided方法的训练动态，表明前缀注入可稀释GRPO组内全对/全错导致的梯度信号匮乏。

## 方法详解
1. **Prefix-Completion RLVR目标函数**：给定问题q，辅助模型生成L步前缀$\tilde{r}$，目标模型条件生成续段$r_{\text{suf}}$，优化目标为最大化完整轨迹$(\tilde{r} \circ r_{\text{suf}})$的verifier奖励期望（Equation 4）。
2. **熵自适应截断**：将辅助轨迹按换行/句号分割为步骤$s_j$，计算目标基础模型$\theta_0$在每步的条件熵$\bar{H}_{\theta_0}(s_j)$（Equation 2），取相邻步熵降最大处作为截断点$L^*$（Equation 5），保留高熵区域$\tilde{r} = \{\tilde{s}_1, \cdots, \tilde{s}_{L^*}\}$（Equation 6）。
3. **混合训练策略**：以概率$p=0.2$使用prefix-augmented输入，以概率$1-p$使用原始问题，目标函数为加权期望（Equation 7），平衡探索与部署一致性。
4. **GRPO核心机制**：组内采样G个响应，计算组归一化优势$\hat{A}_i = (R_i - \text{mean}(R)) / \text{std}(R)$（Equation 1），采用PPO风格的clip surrogate loss加KL惩罚（Equation 2），但实验中移除KL项以稳定训练。

## 实验与结果
- **训练数据**：MATH训练集（7500个唯一问题-答案对）
- **评估基准**：AIME 2024、AIME 2025、AMC 2023、MATH 500、Minerva、Olympiad Bench，报告平均pass@k
- **目标模型**：Qwen2.5-7B、Qwen2.5-Math-7B
- **辅助模型**：Gemma-2-2B、LLaMA-3.2-1B（前缀生成，temperature=0.4）
- **最强结果**：Qwen2.5-7B + Gemma-2-2B prefix在pass@128上达70.71%（基线67.76%），提升+2.95pp；在pass@1上达39.01%（基线38.29%），提升+0.72pp
- **关键趋势**：所有设置下pass@k越大提升越显著；LLaMA-3.2-1B前缀在pass@128上达69.06%（vs基线67.76%，+1.30pp）；p=0.2优于p=0.5和p=1.0；熵截断优于随机截断；同源模型（Qwen2.5-1.5B/7B）前缀无显著增益

## 相关工作脉络
1. **Cui et al. (2025) The entropy mechanism of RLVR**：从算法层面引入熵正则化缓解over-confidence；本文从数据层面提供正交通路，不改内部目标函数。
2. **Peng et al. (2025) Simko / Pass@k training**：通过梯度重新分配控制探索-利用平衡；本文不修改梯度计算，仅改变输入分布。
3. **Yan et al. (2026) Learning to reason under off-policy guidance**：依赖强教师蒸馏信号稳定学习；本文刻意使用弱模型，利用分布差异而非知识迁移。
4. **Dong et al. (2025) RL-Plus**：通过混合策略优化应对能力边界坍缩；本文聚焦推理轨迹层面的前缀扰动而非策略混合。
5. **Wu et al. (2025b) Thought-augmented policy optimization**：整合外部思维模式增强推理；本文前缀来自不同模型而非同一模型的规划模块。

## 局限性与未来方向
1. 超参数敏感：前缀注入概率p需精细调优（p=1.0导致训练-评估不一致，性能下降）。
2. 任务域受限：当前仅验证于数学推理，逻辑推理、代码生成等未探索。
3. 前缀生成依赖额外小模型推理，增加训练阶段的计算开销。
4. 未来方向：与熵正则化、advantage function设计等算法级改进结合，探索数据级扰动与目标级正则化的协同效应。

## 研究启发与可借鉴点
1. **跨模型多样性作为探索信号**：利用不同架构/预训练数据的模型生成多样化前缀，可扩展至其他RLVR变体（如REINFORCE、PPO），为pass@k优化提供低成本正交通路。
2. **熵动态驱动的自动化超参**：熵截断策略通过单点计算即可确定前缀长度，避免网格搜索，可迁移至其他需要控制探索深度的序列生成任务。
3. **"质量-扰动"解耦洞察**：前缀不需要正确，只需要"陌生"；这挑战了传统"蒸馏需高质量信号"的假设，启示在RL探索中重新评估辅助信号的来源标准。
4. **混合训练比例设计**：p=0.2的轻度注入策略既可维持pass@1不降，又能显著提升pass@k，为探索-利用权衡提供了实用参考点。

## 关键术语表
**RLVR**：Reinforcement Learning with Verifiable Rewards，利用可验证奖励信号（如数学答案正确性）进行序列级强化学习训练的方法。
**GRPO**：Group Relative Policy Optimization，PPO的变体，通过组内多采样响应的奖励比较估计优势函数，无需独立价值模型。
**Pass@k**：在k次独立采样中至少获得一个正确回答的概率，衡量模型推理覆盖范围而非单次准确率。
**Entropy Collapse**：RLVR训练中策略分布熵快速下降的现象，导致模型过度集中于少数推理路径，丧失探索能力。
**Prefix-Completion**：在给定部分推理轨迹前缀的条件下，让模型继续生成剩余推理步骤的框架。
**Non-parametric Steering**：不依赖可训练参数的引导机制，此处指利用弱模型生成的文本前缀作为外部扰动信号。

## 可复现要素
- **数据集**：MATH训练集（7500个问题）公开；测试集（AIME 2024/2025, AMC 2023, MATH 500, Minerva, Olympiad Bench）均为公开基准
- **代码**：基于VeRL框架实现，论文未声明代码是否开源
- **关键超参**：学习率$10^{-6}$，无warmup；batch size 1024，minibatch 256；每组采样G=8个响应，temperature=1.0；前缀注入概率p=0.2；前缀生成temperature=0.4；评估时temperature=0.6，top-p=0.95，max length=4096；pass@k采样数：MATH 500/Minerva/Olympiad Bench为128次，AIME/AMC为200次
