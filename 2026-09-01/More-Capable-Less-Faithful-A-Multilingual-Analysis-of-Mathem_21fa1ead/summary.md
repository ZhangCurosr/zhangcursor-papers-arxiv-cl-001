---
title: "More-Capable-Less-Faithful-A-Multilingual-Analysis-of-Mathem"
source: https://arxiv.org/pdf/2608.30463v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 13:51:47"
---

# 论文速读：More-Capable-Less-Faithful-A-Multilingual-Analysis-of-Mathem

## 一句话总结
本文构建首个支持法语与希腊语的可解/不可解数学题配对基准，从行为、表征与忠实度三维分析多语言 LLM 的数学不可解性检测能力，发现模型内部的 Solvability Belief（SB）具有跨语言普遍性，但高资源语言（英语）下模型“解题能力越强，对外表达却越不忠实”。

## 研究问题与动机
- 现有不可解性检测研究几乎全限于英语，无法判断多语言失败源于内部知识差异还是语言表达/转述能力的差异。
- 语言资源强弱是否系统性影响 LLM 将内部可解性判断“如实输出”的忠实度（faithfulness）？
- 针对特定语言微调（如 Llama-Krikri / French-Alpaca）能否有效改善低资源语言下的拒绝行为，还是会引发误拒率飙升的副作用？
- 单语言探针能否直接迁移至多语言场景，是否需要为每种语言单独建模 SB 编码？

## 核心贡献（创新点）
- **首个多语言配对基准**：将 ReliableMath 完整翻译并人工校验至法语与希腊语，覆盖313道可解题与1102道不可解题，填补非英语数学可靠性评测空白。
- **跨语言通用 SB 探针验证**：首次在多语言设置下证明 Solvability Belief 编码于高度通用的轻度语言特定子空间中，跨语言池化训练的通用探针性能等于或优于单语言探针。
- **揭示“高能力-低忠实度”的语言反转**：原生多语言模型（Gemma、Qwen）在英语下内部 SB 编码最清晰，但行为输出忠实度反而最低；法语/希腊语下虽解题准确率下降，模型对内部判断的表达却更加忠实。
- **清晰刻画 Aware Prompting 与语言特化的行为效应**：显式允许声明“不可解”可显著提升低资源语言忠实度；语言特化微调会全局性降低目标语言的拒绝阈值，导致真拒与误拒同步上升。

## 方法详解
- **基准翻译与校验**：使用 Claude Sonnet 5 将 ReliableMath 翻译至法语与希腊语；不可解题通过对原题删除或矛盾关键信息构造。两名作者分别对100道法/希题目进行内容、数值与可解性类型的双盲人工校验，一致率分别达97.0%与95.0%。
- **CoT 生成与隐状态提取**：为三种语言分别设计 Standard 与 Aware 提示模板；从各层提取隐状态，选取探针验证 AUC 最高的中间层（Llama 系列第16层，Qwen3-30B/Gemma-4 第36层），每序列均匀采样20个 token 向量。
- **SB 探针训练**：以题目真实可解性标签作为模型内部信念代理，训练 L1 正则化逻辑回归探针；分别训练 Per-language 探针与跨语言池化的 Universal Probe，以验证集 AUC 衡量 SB 编码强度。
- **文本裁决（LLM-as-a-Judge）**：鉴于 CoT 推理过程中模型判断可能动态漂移，放弃硬性关键词匹配，改用 Llama-3.3-70B-Instruct 作为裁判推断推理轨迹的主导裁决；在100条样本上与人工标注对比，一致率达93.0%。
- **忠实度度量**：将 Universal SB 探针预测与 LLM Judge 裁决逐样本比对，一致性比例即为 Faithfulness Rate；对不可解题子集额外对比 Standard→Aware 提示切换前后的忠实度提升幅度。

## 实验与结果
- **模型与数据**：测试 Qwen3-30B-A3B-Instruct-2507、Qwen3-4B-Instruct-2507、Llama-3.1-8B-Instruct、Gemma4-31B-IT，以及语言特化版 Llama-Krikri-8B（希腊语）与 French-Alpaca-8B（法语）。数据源自 MATH、MinervaMath、AIME24/AMC。
- **解题能力分层**：所有模型准确率呈现英>法>希的稳定层级；Gemma-4-31B-IT（英61.3%）与 Qwen3 系列显著领先，Llama-3.1-8B 最低（英23.3%）（Table 1）。
- **SB 表征强度**：各模型 Universal Probe AUC 均随语言资源递减；French-Alpaca 与 Llama-Krikri 即使经过目标语言微调，主导预训练语言（英语）的编码优势仍压倒微调语言。Universal Probe 在绝大多数模型-语言组合上显著优于或等于 Per-language 探针（bootstrap 95% CI 验证）。
- **拒绝行为（Abstention）**：Standard 提示下整体拒绝率较低，仅 Qwen3 系列约25%；改为 Aware 提示后拒绝率骤升。语言特化模型在目标语言
