---
title: "Is-Discrete-Difficulty-Sufficient-Leveraging-Continuous-Diff"
source: https://arxiv.org/pdf/2608.24590v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 10:42:56"
field: "大模型推理效率优化"
keywords: ["Self-Consistency", "Test-Time Scaling", "Difficulty Estimation", "Output Entropy", "Efficient Decoding", "Adaptive Sampling"]
innovations: ["以输出熵为连续难度信号替代离散难度分级，实现更精细的推理资源分配", "提出轻量线性探针从 LLM 最后 token embedding 预测输出熵，训练开销极低", "FSC 框架在多项基准上相比 SC 最高节约 76.7% token 且精度持平或提升"]
benchmarks: ["MATH500", "AMC23", "AIME2024", "AIME2025", "GPQA-Diamond", "MMLU-Pro"]
---

# 论文速读：Is-Discrete-Difficulty-Sufficient-Leveraging-Continuous-Diff

## 一句话总结
本文提出 Flexible Self-Consistency（FSC），利用轻量级线性探针预测 LLM 输出熵作为**连续难度信号**，动态调整各问题的采样预算（推理路径数），在保持与 Self-Consistency（SC）相当准确率的同时实现高达 76% 的 token 节约。

## 研究问题与动机
- **SC 的计算效率瓶颈**：Self-Consistency 需为每个问题生成多条推理路径，大量问题实际只需少量路径即可收敛到正确答案，固定预算策略浪费严重。
- **现有难度自适应方法粒度不足**：AC、ESC、DSC 等前作将难度划分为少数离散类别（如 easy/hard），同一类别内题目间推理需求差异被忽略，导致资源分配不精确——简单题可能过度计算，困难题可能探索不足。
- **模型内在不确定性缺乏连续度量**：现有困难度估计工作依赖 LLM 自评或离散分类，未能充分捕捉模型对输入问题的**连续不确定性感知**。
- **核心问题**：能否将题目难度建模为连续信号而非离散类别，从而实现更精细的推理资源分配？

## 核心贡献（创新点）
- **提出 FSC 框架，以输出熵为连续难度信号指导自适应采样预算**：区别于 DSC 等使用离散难度分级的方案，FSC 通过训练线性探针预测每个输入的连续熵值，据此动态分配推理路径数。
- **揭示并验证输出熵与题目难度之间的单调连续关系**：在 MATH 数据集上实证表明，随难度等级上升，模型输出熵逐渐增大；高难度区间（Level 5）所需推理链数量急剧增加，且错误答案的多样性显著高于正确答案。
- **设计零额外推理开销的轻量探针训练流程**：基于 MATH 训练集（7,500 样本）构造合成数据，以最后 token 的 hidden embedding 为输入、生成的答案分布熵为监督信号，训练无非线性激活的线性回归模型。
- **在多个基准和模型规模上验证高效性**：FSC 在 Qwen2.5 系列（3B/7B/14B）和 Gemma-3-4B-it 上，相比 SC 最高减少 76.7% token 消耗且精度持平甚至略有提升。

## 方法详解
FSC 包含两个阶段：

**阶段一：训练轻量线性探针**
- 对训练集每个问题 $q$，用 LLM 生成 $N$ 条推理轨迹，提取最终答案构成集合 $\mathcal{A}_q$，统计唯一答案的频率分布 $p_q(a)$，计算输出熵：
$$H_q = -\sum_{a \in \mathcal{U}_q} p_q(a) \log_2 p_q(a)$$
- 以 LLM 对问题 $q$ 的最后 token hidden embedding 为输入，预测 $\hat{H}_q$，使用 MSE 损失训练线性回归探针：
$$\mathcal{L} = \mathbb{E}_{(q, H_q) \sim \tilde{D}} \left[ (H_q - \hat{H}_q)^2 \right]$$

**阶段二：熵引导的自适应 Self-Consistency**
- 对新的输入问题，探针预测输出熵 $\hat{H}_q$，并进行裁剪防止越界：$\hat{H}_q \leftarrow \mathrm{clip}(\hat{H}_q, 0, \log_2 N)$。
- 将预测熵归一化为 [0,1] 范围内的相对难度分数，计算自适应采样预算：
$$N_{\mathrm{adj}} = \left\lceil 1 + (N - 1) \cdot \frac{\hat{H}_q}{\log_2 N} \right\rceil$$
- 当 $\hat{H}_q = 0$（极简单题）时分配 1 条推理路径；当 $\hat{H}_q = \log_2 N$（极困难题）时分配全部 $N$ 条路径。中间难度按线性比例分配，实现连续精细调节。

## 实验与结果
- **数据集**：MATH500、AMC23、AIME2024、AIME2025、GPQA-Diamond（推理评估）；MMLU-Pro（OOD 泛化评估）；探针训练使用 MATH 训练集（7,500 样本）。
- **模型**：Qwen2.5-Instruct（3B/7B/14B）、Gemma-3-4B-it，zero-shot 设置。
- **基线**：SC、AC（Aggarwal et al., 2023）、ESC（Li et al., 2024）、DSC（Wang et al., 2025）。最大采样预算统一设为 $N=40$。
- **主要结果**：
  - **Qwen2.5-14B / MATH500**：FSC 准确率 82.2%（优于 SC 的 81.6%），token 消耗仅 6.1×10³（较 SC 减少 75.1%）。
  - **Qwen2.5-14B / GPQA-Diamond**：FSC 准确率 46.7%（vs SC 46.2%），token 减少 68.2%。
  - **Qwen2.5-7B / AMC23**：FSC 准确率 65.0%（超越 SC 的 62.5%），token 减少 56.5%。
  - **Gemma-3-4B / MATH500**：FSC 准确率 78.8%（vs SC 79.2%），token 减少 76.7%（最大节省幅度）。
  - **MMLU-Pro OOD 评估**：FSC 在大多数学科领域实现最高效率，跨分布表现稳定。
- **对比结论**：AC/ESC 在某些设置下 token 消耗甚至高于 SC；FSC 在所有模型-数据集组合上均稳定降低 token 消耗，且精度与基线持平或更优。

## 相关工作脉络
- **Self-Consistency (SC, Wang et al., 2022)**：多路径采样+多数投票，是本文的性能上限基线，但计算开销大，FSC 在此基础上实现更高效调度。
- **Adaptive Consistency (AC, Aggarwal et al., 2023)**：顺序生成推理链、检测一致性后提前终止；在部分场景下 token 消耗高于 SC，FSC 通过连续信号避免此类不稳定。
- **Early-Stopping SC (ESC, Li et al., 2024)**：固定窗口内答案收敛则停止；属于启发式早停策略，不涉及难度感知。
- **Difficulty-Adaptive SC (DSC, Wang et al., 2025)**：首次引入离散难度估计进行自适应预算分配；FSC 的关键区别在于将难度从离散等级升级为连续熵信号，实现更细粒度分配。
- **内部难度估计 (Lee et al., 2025; Zhu et al., 2025)**：利用 LLM 内部表征预测题目难度；FSC 继承"使用 hidden embedding"的思路，但将目标从离散难度标签改为连续输出熵。
- **Test-Time Scaling (TTS, Snell et al., 2024; Muennighoff et al., 2025)**：本文属 TTS 范式下的推理效率优化分支，聚焦于如何精确定位"每道题需要多少计算"。

## 局限性与未来方向
- **模型规模限制**：实验仅覆盖 ≤14B 参数模型，探针预测的熵值随模型规模和架构变化，尚未验证更大模型（如 70B+）的效果。
- **依赖模型内部隐藏表示**：探针需访问 LLM 最后 token 的 hidden embedding，无法应用于 GPT 等闭源模型，限制了通用性。
- **探针训练数据领域局限**：探针仅在数学数据集（MATH）上训练，跨领域泛化能力待进一步验证（MMLU-Pro 实验提供初步证据，但系统性评估缺失）。
- **未来方向**：开发不依赖内部表征的更高效自洽方法；在更多样化数据集（如代码、知识问答）上训练探针以提升跨领域鲁棒性；探索适用于闭源模型的替代方案。

## 研究启发与可借鉴点
- **输出熵作为连续难度代理**：将模型输出分布的不确定性直接映射为连续难度信号，这一思路可迁移至其他自适应解码策略（如 adaptive sampling、early exit）中。
- **轻量探针 + 线性回归的简洁性**：无需复杂架构，仅利用最后 token embedding 做 MSE 回归即可捕捉细粒度难度，为后续工作提供低开销、易实现的探针范式。
- **线性比例预算分配公式**：公式 $N_{\mathrm{adj}} = \lceil 1 + (N-1) \cdot \hat{H}_q / \log_2 N \rceil$ 简洁且可解释，可直接复用于其他需要连续预算调度的场景。
- **与团队方向的结合机会**：可将 FSC 的连续难度信号思想引入团队在多模态推理/代码生成上的自适应计算分配研究，探索跨模态的熵-难度映射关系。
- **探针训练数据的低成本生成方式**：用同一 LLM 生成 40 条路径并统计熵值作为伪标签，无需人工标注，为同类探针研究提供可复用的数据构造模板。

## 关键术语表
- **Self-Consistency (SC)**：对同一问题生成多条推理路径，通过多数投票选择最终答案的解码策略。
- **Flexible Self-Consistency (FSC)**：本文提出的框架，利用预测输出熵作为连续难度信号，动态调整各问题的采样预算。
- **Output Entropy**：模型对某输入生成的答案分布的信息熵，衡量模型对该问题的不确定性程度，本文作为连续难度信号。
- **Test-Time Scaling (TTS)**：在推理阶段额外分配计算资源以提升模型性能的范式，与扩展训练时计算（scaling laws）相对。
- **Linear Probe**：冻结预训练 LLM 参数、在其输出层后接轻量线性层的评估工具，用于探测模型内部表征所蕴含的信息。
- **Adaptive Consistency (AC)**：顺序生成推理链、检测中间结果一致性后提前终止采样的效率优化方法。
- **Early-Stopping SC (ESC)**：在固定滑动窗口内检测答案收敛性并据此提前终止采样的方法。
- **Difficulty-Adaptive SC (DSC)**：基于 LLM 自评将题目难度离散分类（easy/hard），并据此分配不同推理路径数的方法。

## 可复现要素
- **探针训练数据**：MATH 训练集（7,500 样本），MIT License，开源。
- **评测数据集**：MATH500（MIT）、AMC23（Apache 2.0）、AIME2024/2025（CC BY-NC-SA 4.0）、GPQA-Diamond（MIT）、MMLU-Pro（MIT），均公开可用。
- **代码开源情况**：论文未提及代码仓库链接。
- **关键超参**：推理时温度=0.7、top-p=0.95；探针训练时温度=1.0、top-p=1.0、每条问题生成 40 条路径；AC 阈值=0.95；ESC 窗口大小=5；DSC judge 窗口=24；FSC 最大采样预算 $N=40$。
