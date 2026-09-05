---
title: "Evaluating-and-Improving-LLM-Self-Modeling"
source: https://arxiv.org/pdf/2608.30980v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-09-05 06:41:50"
---

# 论文速读：Evaluating-and-Improving-LLM-Self-Modeling

## 一句话总结
本文构建了首个系统性的 LLM 行为级自我建模评估基准（9 类任务、4 大领域），并提出自动化扰动合成与多任务 RL 训练管道；实验证明自我建模是可测量且可通过 RL 跨模型族提升的行为能力，但当前前沿模型在该能力上仍有显著差距，且训练提升主要源于格式合规与行为泛化而非特权内省访问。

## 研究问题与动机
- **系统性自我建模失败**：前沿模型在反事实翻转预测、自身准确率估计、扰动响应评估等任务上存在稳定且可复现的错误模式（如将显式指令误判为因果核心组件、过度自信）。
- **缺乏统一的行为级基准**：既有工作多聚焦隐状态干预或单一内省任务，缺少覆盖多答案格式、多领域、可量化的综合评测协议。
- **训练提升机制不明**：RL post-training 能否真正增强模型的“自我认知”？还是仅学会了行为层面的模式匹配？跨模型迁移路径与归因尚未厘清。
- **评估指标偏差**：不同任务天然翻转率差异大，raw accuracy 不可比，需引入去基线化的标准化指标。

## 核心贡献（创新点）
- **构建首个多维度行为级自我建模基准**：整合 9 类任务（E1–E10）覆盖数学、编程、安全、公平性，严格限定于外部可观测的行为预测，不依赖内部状态访问。与以往侧重 SAE 特征提取或单一忠实度任务的工作本质不同。
- **提出 Skill Score 评估指标**：引入扣除虚拟预测器基线的净分（$skill_t = s_t - b_t$，范围 [-1, 1]），消除因任务行为翻转率不同导致的评价偏差。与单纯 report accuracy 的评测方式本质不同。
- **设计自动化扰动数据生成管道**：六阶段 pipeline（SEARCH-FETCH-EXTRACT-MAP-VALIDATE）结合 generate-verify-revise 与 fork-verify-revise 循环，自动产出带翻转率标签的 ~20K 训练三元组。与人工标注或静态数据集构建方式本质不同。
- **揭示 RL 训练的解耦提升路径**：通过 Shapley 分解证明 RL 收益来自“格式合规修复”与“自我建模质量提升”两条独立路径，并量化不同模型族的贡献占比差异。与以往将 RL 收益笼统归因于对齐或指令遵循的研究本质不同。

## 方法详解
- **基准任务设计**：9 类任务（FLIP-DECISION、OUTPUT-PREDICTION、FLIP-RATE、SELF-ACCURACY、CONFIDENCE-RECALL、PERTURBATION-CHOICE、COMPONENT-ATTRIBUTION、EDIT-PROPOSAL、FEATURE-RATE，开源模型另含 E10 LOGIT-ESTIMATION）。按行为变化单位分为 Label mode（最终答案变化）与 Feature mode（verifier 谓词变化）。答案格式涵盖二分类、多选、标量 [0,1]、自由文本。
- **经验翻转率估计**：$g(x) = \frac{1}{K^2}\sum_{i=1}^K\sum_{j=1}^K \mathbb{1}[a_b^{(i)}(x) \neq a_\ell^{(j)}(x)]$，通过 K 重采样估计扰动后答案/特征的变化概率。
- **自动化数据管道**：
  - **数据集发现**：查询 KNOWN_REPORT_URLS（14 条）→ DuckDuckGo + Haiku 回退 → 难度分层（>95%/80-95%/<80% frontier）。
  - **过滤规则**：排除非文本、eval reserve list（GSM8K/HumanEval/BBQ/WildGuardTest）、样本<10、平均长度<20 字、HF 下载<50、超限 250 个、schema 缺失 question/answer 字段的数据集。
  - **扰动生成**：DECOMPOSE 识别 5–8 个结构元素（content/format/context/assumption/constraint/implicit_premise，必须含至少一个 implicit_premise）；GENERATE 三类扰动：flip_inducing（目标翻转率 60–80%）、non_flip（0–10%）、boundary（30–50%），共 17 种 wording-level perturbations。
- **RL 训练框架（RLVR）**：
  - 架构：LoRA（rank=32）+ REINFORCE with unclipped importance sampling（Tinker loss）。
  - Advantage：组内均值中心化 $A_k = R_k - \bar{R}$，无 σ 除、无 KL penalty、无 ratio clip。
  - Reward：格式 bonus 0.5 + 任务分 [0,1]。
  - 超参：batch=64 prompts/step（K=16 → 1024 trajectories），rollout temp=0.7，max_output_tokens=2048，max_seq_len=4096，AdamW（β₁=0.9, β₂=0.95, wd=0.01），lr=2e−5（50 步 warmup + cosine decay），训练 200 步。
- **评估协议**：四层平均（组内→组间 macro 等权→种子内任务平均→跨种子平均）；Base vs Instruct 严格对比；Prompt 敏感性验证（6 前沿模型 × 3 变体 × 100 示例，p=0.014）。

## 实验与结果
- **数据集与基线**：GSM8K、HumanEval、BBQ、WildGuardTest；基线包括 no-FT、SFT（6 种 JSON 格式）、DAPO、单任务 RL、多任务 MTL RL；跨模型族测试 Llama-3.1-8B、Qwen3-8B、GPT-OSS-20B。
- **最强结果**：18 模型 Leaderboard 中，**DeepSeek-V3.1 获得最高聚合 skill +0.147**，距理论天花板 +1 仍有较大空间；Base checkpoint 得分显著低于 Instruct（主因 JSON 格式遵从失败）。
- **RL 提升与迁移**：多任务 HF RL 在三个模型族上均优于 no-FT；BLOOM-only RL 对 Qwen3-8B 产生负迁移。跨模型 explainer 互换实验中，own-model explainer 通常更优（Qwen→Qwen 0.129 vs GPT→Qwen 0.067），但强解释器可在弱模型行为标签上胜出。
- **数据规模非单调规律**：小数据（1,275 行）峰值在 epoch 8/9（skill +
