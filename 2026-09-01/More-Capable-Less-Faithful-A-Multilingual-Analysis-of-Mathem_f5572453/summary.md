---
title: "More-Capable-Less-Faithful-A-Multilingual-Analysis-of-Mathem"
source: https://arxiv.org/pdf/2608.30463v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 13:51:27"
field: "多语言大语言模型能力评估"
keywords: ["可解性检测", "多语言LLM", "忠实度分析", "内部表征探针", "数学推理", "拒绝行为"]
innovations: ["首个多语言配对可解/不可解数学问题基准（法语+希腊语）", "揭示高资源语言能力强但可解性检测忠实度低的反常现象", "证明可解性信念以跨语言通用的方式编码于隐藏状态"]
benchmarks: ["ReliableMath", "MATH", "MinervaMath", "AIME24/AMC"]
---

# 论文速读：More-Capable-Less-Faithful-A-Multilingual-Analysis-of-Mathem

## 一句话总结
本文首次构建了多语言版本的数学可解性检测基准（将 ReliableMath 扩展至法语和希腊语），从行为、表征和忠实度三个维度系统分析了 LLM 在多语言环境下的数学不可解性检测能力，发现高资源语言（如英语）虽然数学推理性能更强，但其可解性检测的忠实度反而更低。

## 研究问题与动机
- **现有研究局限**：此前关于 LLM 数学可解性检测的研究几乎全部局限于英语，不清楚多语言失败是源于内部可解性信念的差异，还是语言依赖性的表达失败。
- **RQ1**：LLM 在数学推理和不可解性检测方面的性能如何随语言变化？
- **RQ2**：语言如何影响内部可解性表征与文本可解性判断之间的忠实度？
- **动机**：填补多语言可解性检测研究的空白，理解模型能力与忠实表达之间的关系。

## 核心贡献（创新点）
- **首个多语言配对可解/不可解数学问题基准**：将 ReliableMath 翻译为法语和希腊语，保留原始问题的数学内容和可解性标签，代码和数据将开源。
- **行为分析**：在六个前沿 LLM 上系统评估了英语、法语、希腊语三种语言下的数学推理准确性和拒绝率。
- **首个多语言可解性信念表征分析**：证明可解性信念（Solability Belief）在隐藏状态中以 largely universal、language-agnostic 的方式编码，但高资源语言的忠实度反而更低——能力与忠实度呈现反向关系。

## 方法详解
- **基准翻译**：使用 Claude Sonnet 5 将 ReliableMath（含 313 个可解问题和 1,102 个不可解变体）翻译为法语和希腊语，并进行人工验证（法语 97%，希腊语 95% 一致）。
- **思维链生成与隐藏状态提取**：对每种语言使用标准 prompt 和 aware prompt（允许模型显式声明问题不可解），从 probing 性能最高的层提取隐藏状态（每序列采样 20 个 token 向量）。
- **可解性信念探针**：用 ground-truth 可解性标签训练 L1 正则化逻辑回归探针，分别用 per-language 探针和 pooled universal 探针（跨语言联合训练），universal 探针性能匹配或超越 per-language 探针。
- **文本可解性判断标注**：采用 LLM-as-a-judge（Llama-3.3-70B-Instruct）方法，由裁判模型推断推理 trace 中占主导地位的可解性判断（与人工标注一致率达 93%）。
- **忠实度定义**：比较 universal SB 探针预测与文本裁判判断的一致性比率。

## 实验与结果
- **模型**：Gemma-4-31B-it、Qwen3-4B、Qwen3-30B、Llama-3.1-8B、Llama-Krikri-8B（希腊语适配）、French-Alpaca-8B（法语适配）。
- **数据集**：ReliableMath 三语言版本，含 313 个可解问题（来自 MATH、MinervaMath、AIME24/AMC）和 1,102 个不可解变体。
- **能力表现**：模型在所有三种语言中均以英语表现最佳、希腊语最差（如 Gemma-4-31B：En 61.3% > Fr 55.6% ≈ Gr 56.5%；Llama-3.1-8B：En 23.3% > Fr 15.0% > Gr 14.4%）。
- **SB 编码强度**：AUC 从英语到法语到希腊语单调下降，universal 探针跨语言表现优异，表明 SB 编码为共享的、轻度语言特定的子空间。
- **拒绝率**：aware prompt 显著提升所有模型的拒绝率；语言适配模型在目标语言中真拒绝率上升但假拒绝率也同步激增（阈值整体降低）。
- **忠实度逆转**：多语言原生模型（Gemma、Qwen）在非英语语言中忠实度更高（如 Qwen3-30B：En 0.712 < Fr 0.730 < Gr 0.723）；少数语言模型 Llama-3.1-8B 相反（En 0.544 > Fr 0.529 > Gr 0.498）。aware prompt 对不可解问题的忠实度提升在非英语中更显著（如 Krikri-8B：Gr 0.273→0.556）。

## 相关工作脉络
- **Xue et al. (2025) ReliableMath**：本文基准的基础，提供英语配对可解/不可解数学问题，本文将其扩展至法语和希腊语。
- **Xiros et al. (2026)**：分解 LLM 中可解性信念与语言化表达为两个几何解耦的方向，本文为此框架补充多语言视角。
- **Sanyal et al. (2025)**：区分信心与能力，本文为不可解性检测提供多语言表征层面的验证。
- **Liu et al. (2026)**：发现 LLM 内部常识别不可解问题但无法拒绝，本文从多语言忠实度角度深化此现象的解释。
- **Civelli et al. (2026)**：研究多语言模型中问题难度编码的共享几何结构，与本文的 universal SB 探针发现相呼应。
- **Shi et al. (2023)**：多语言 CoT 推理的开创性工作，本文在其基础上聚焦于可解性检测这一特定能力。

## 局限性与未来方向
- 仅覆盖三种语言（英语、法语、希腊语），需扩展至高资源语言（如中文）和类型学多样的低资源语言。
- 未提出对齐内部 SB 与文本判断的方法，未来可基于此发展多语言忠实可解性检测技术。
- 翻译依赖 LLM，可能存在细微语义偏差（虽经人工验证）。
- 未深入分析忠实度差异的成因（如输出分布、培训数据语言比例等）。

## 研究启发与可借鉴点
- **Universal Probe 设计**：跨语言 pooled 探针优于 per-language 探针，验证了内部表征的语言无关性，此策略可迁移至其他多语言表征分析任务。
- **忠实度与能力反向关系**：高资源语言能力强但忠实度低，提示在关键应用中需对高资源语言额外校准输出可靠性。
- **Aware Prompt 效果**：显式允许模型声明不可解可大幅提升忠实度，尤其在非英语中增益更大，可作为多语言数学推理的标准实践。
- **LLM-as-Judge 验证**：用 LLM 标注文本可解性判断并与人工对照（93% 一致率），为自动评估方法提供可靠范式。
- **语言适配的双刃剑**：适配模型在目标语言中真拒绝提升但假拒绝同步增加，提醒fine-tuning 可能引入校准偏差。

## 关键术语表
- **Solability Belief (SB)**：模型对问题是否可解的内部表征/信念，通过隐藏状态探针测量。
- **Faithfulness**：内部 SB 表征与文本可解性判断之间的一致性比率，衡量模型"知行合一"程度。
- **Abstention（拒绝）**：模型正确识别问题不可解并声明不可解的能力，区分 true abstention 和 false abstention。
- **Aware Prompt**：允许模型显式输出"unsolvable"的 prompt 设置，区别于标准求解 prompt。
- **Universal Probe**：跨多种语言 pooled 训练的可解性信念探针，验证 SB 编码的跨语言通用性。
- **LLM-as-Judge**：用大型语言模型作为裁判推断推理 trace 的可解性判断，替代严格字符串匹配。

## 可复现要素
- **数据集**：ReliableMath 法语/希腊语翻译版，论文声明代码和数据将以 Apache 2.0 许可开源。
- **模型**：Qwen3-30B-A3B、Qwen3-4B、Llama-3.1-8B、Gemma-4-31B、Llama-Krikri-8B、French-Alpaca-8B。
- **关键超参**：hidden state 采样 20 个 token 向量；L1 正则化逻辑回归探针；judge 模型为 Llama-3.3-70B-Instruct。
- **探针层选择**：Gemma-4-31B（layer 36）、Qwen3-4B（layer 18）、Qwen3-30B（layer 36）、Llama-3.1-8B 系列（layer 16）。
- **统计检验**：problem-level bootstrap（1000 resamples, 95% CI）用于 SB 分析；two-proportion z-test 用于忠实度比较。
