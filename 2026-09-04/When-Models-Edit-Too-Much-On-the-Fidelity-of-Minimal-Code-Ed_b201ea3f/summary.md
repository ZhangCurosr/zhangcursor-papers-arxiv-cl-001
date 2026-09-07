---
title: "When-Models-Edit-Too-Much-On-the-Fidelity-of-Minimal-Code-Ed"
source: https://arxiv.org/pdf/2609.04061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 23:14:55"
---

# 论文速读：When Models Edit Too Much: On the Fidelity of Minimal Code Edits

## 一句话总结
本文揭示大模型在代码修复时普遍存在“过度编辑（over-editing）”现象：补丁虽能通过测试，却会不必要地重写大量无关逻辑；通过构建以已知最小补丁为 ground truth 的受控基准，证明显式保留指令可显著缓解该行为，且强化学习（RL）比监督微调（SFT/DPO）更能学会可迁移的最小化编辑偏好，同时不损害通用编码能力。

## 研究问题与动机
- 现有代码评测基准（如 HumanEvalFix、SWE-bench、CanItEdit）主要依赖 Pass@1/k 或 issue 解决率，无法量化模型在 brownfield 维护场景中是否保留了原始实现意图与局部性。
- 前沿 LLM 默认倾向于“产出健壮代码”而非“最小局部修复”：例如 GPT-5.5 仅修一行 off-by-one 却插入 60 行输入校验与 NaN 掩码，通过测试但大幅加重审查负担。
- 推理增强（reasoning）与模型规模放大并未单调改善编辑保真度，打破“更大/更会思考就能更精准”的直觉假设。
- 研究需明确：过度编辑是能力缺陷还是任务框架错位？能否通过 prompting 或后训练有效抑制，且训练后的模型仍保持通用编码能力？

## 核心贡献（创新点）
- **提出独立的编辑保真度评估轴**：构建首个以已知最小补丁为 ground truth 的受控基准（400 个 BigCodeBench 任务 + 注入 AST 级局部篡改），首次将 excess Levenshtein distance 与 added cognitive complexity 作为可量化、可验证的独立指标。
- **系统揭示前端模型的过度编辑默认行为**：评测 20+ 前沿模型，证明高 Pass@1 与大规模超额编辑可长期共存；单条显式保留指令即可使 aggregate excess Lev 从 0.195 降至 0.131，Pass@1 提升 2.3 分。
- **澄清推理与规模的非单调作用**：对比 reasoning/non-reasoning 变体与 Qwen2.5-Coder 0.5B→32B 缩放曲线，证实更强的推理预算或参数量不能保证更小的局部补丁。
- **确立 RL 为学习最小化编辑偏好的最优后训练范式**：对比 SFT/rSFT/DPO/RL，SFT 严重过拟合训练篡改族（OOD Pass@1 从 0.932 跌至 0.458）并损害 LiveCodeBench（-14.9 pts），而 RL 在 OOD 上取得 Pass@1 0.782、excess Lev 0.050、added CC 0.185，且 LCB +0.6 pts。
- **验证低秩适配与跨语言迁移可行性**：LoRA rank 64 的 RL 几乎追平全参数表现；在真实 Java 缺陷集 Defects4J 上，RL 微调保持通过率不变的同时显著压缩 token edits，证明最小化编辑偏好可跨语言迁移。

## 方法详解
- **基准构建**：从 BigCodeBench 采样 400 个 Python 函数，基于 14 类预定义 AST 篡改族（比较运算符、范围边界、切片索引、条件反转、布尔/数值常量翻转等）注入 1-2 个局部错误，仅保留篡改后程序无法通过原始可执行测试的案例。目标最小补丁为篡改的严格反向操作：50.2% 仅需单 token 修改，91.8% ≤2 tokens，100% 不超过 2 行。
- **核心评估指标**：
  - **Pass@1**：通过全部测试的模型输出比例。
  - **Excess Levenshtein Distance**：$E_{Lev}(M) = d(M, C) - d(G, C)$，其中 $d(\cdot
