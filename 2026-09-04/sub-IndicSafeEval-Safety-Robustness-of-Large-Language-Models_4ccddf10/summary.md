---
title: "sub-IndicSafeEval-Safety-Robustness-of-Large-Language-Models"
source: https://arxiv.org/pdf/2609.03781v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-07 23:16:56"
---

# 论文速读：sub-IndicSafeEval-Safety-Robustness-of-Large-Language-Models

## 一句话总结
本文提出 IndicSafeEval 基准，通过构建覆盖 4 种印度语言、10 类风险与 6 种说服策略的 9,000 条多语言牢笼突破提示，系统评估主流开源 LLM 在低资源语言环境下的安全鲁棒性，揭示当前以英语为中心的安全对齐评估存在显著盲区与语言偏差。

## 研究问题与动机
- 现有 LLM 安全评估高度依赖英语，无法真实反映低资源与文化多元语言中的对齐失败。
- 安全鲁棒性显著受语言背景与请求措辞风格（如说服性社会工程学技巧）影响，但缺乏系统性多语言评测框架。
- 不同风险类别（如 Gov. Decision Making vs. Hate Speech）对牢笼攻击的脆弱性差异巨大，现有工作未进行细粒度量化。
- 模型在高流利度语言（如 Hindi/English）上的低 ASR 可能仅源于语言生成能力不足，而非真正的安全对齐更强，现有指标易产生误导性结论。

## 核心贡献（创新点）
1. **提出 IndicSafeEval 多语言牢笼突破基准**：首个针对印度语言、融合说服策略与 10 类风险的多维安全评测框架；与已有英语单语言基准的本质区别在于将“语言多样性”与“社会工程学说服路径”纳入统一评测维度。
2. **设计三阶段自动化数据生成流水线**：通过 GPT-5.1 few-shot 改写 + 策略对齐 + LaBSE 跨语言语义校验规模化产出 9,000 条高质量对抗提示；与以往人工编写或简单回译方式的区别在于实现了从 seed 到多语言对抗样本的结构化、可追溯生成。
3. **系统量化策略-语言-风险的交互效应**：首次完整披露 6 种说服策略在不同语言与风险类别上的 ASR 差异；与现有仅报告整体 ASR 的工作的区别在于提供细粒度诊断信号，明确“最强攻击策略”“最抗攻击风险”及“训练语料偏向”规律。

## 方法详解
- **数据生成流水线（三阶段）**：
  1. **Stage I**：用 GPT-5.1 为每个风险类别×策略对生成 2 条 few-shot 示例（共 120 条），人工核验语义与策略对齐。
  2. **Stage II**：以 few-shot 示例引导 GPT-5.1 将 300 条 seed（Shen et al., 2024b）改写为 1,800 条英文对抗提示。
  3. **Stage III**：Google Translate 翻译至 Hindi/Bengali/Marathi/Punjabi，并用 **LaBSE** 计算跨语言余弦相似度进行一致性校验（均值 0.84–0.88），附录 A.3/A.4 含人工分析。
- **提示策略分类**：基于 Zeng et al. (2024) 说服分类学，采用 6 种策略：Logical Appeal、Authority Endorsement、Misrepresentation、Anchoring、Priming、Confirmation Bias，分别模拟不同社会工程学说服路径。
- **评测设置**：黑盒 single-pass；主指标为 ASR，辅以 SRR、HLR、PCR；自动评判采用 Gemini-2.5-Flash，等级 4–5 计为成功突破；额外开展“响应语言强制为英语”控制实验以隔离语言生成能力与安全对齐能力的混杂效应；更大/闭源模型（Llama-3.3-70B-Instruct、GPT-4o-mini）仅在 Physical Harm 类别上
