---
title: "Debias-SparseGPT-Bias-Aware-Pruning-for-Large-Language-Model"
source: https://arxiv.org/pdf/2609.02496v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:45:44"
field: "大语言模型压缩与公平性"
keywords: ["LLM pruning", "bias mitigation", "model compression", "SparseGPT", "fairness", "post-training pruning", "Hessian-based compression"]
innovations: ["在 SparseGPT 二阶剪枝框架中引入成对输入差分项构建偏置感知 Hessian，首次实现压缩时去偏", "通过 bias-aware saliency 同时优化 mask 选择和权重补偿，保持 O(d³) 复杂度", "揭示校准数据多样性对极端结构化稀疏下公平-性能权衡的关键作用"]
benchmarks: ["UnQover", "BBQ", "CrowS-Pairs", "MMLU", "HellaSwag", "RealToxicityPrompts", "HarmBench"]
---

# 论文速读：Debias-SparseGPT: Bias-Aware Pruning for Large Language Models

## 一句话总结
本文提出了 **Debias-SparseGPT**，一种在 SparseGPT 框架基础上引入表征去偏机制的后训练剪枝方法，通过在 Hessian 中叠加成对对立刻板印象输入的差分项，显著缓解大语言模型压缩过程中的偏见放大问题，同时保持下游性能与 perplexity 不变。

## 研究问题与动机
- **剪枝会放大 LLM 预存偏见**：现有压缩方法（如 SparseGPT）在保持整体准确率的同时，会显著加剧模型对不同人口统计群体的表现差异（如 UnQover/BBQ 基准上的 Not Stated 预测率下降）。
- **现有方法缺乏剪枝时去偏能力**：尽管有研究显示剪枝可降低毒性，但尚无方法在压缩过程中直接优化公平性目标，仅 SparseGPT/Wanda 等二阶剪枝方法被广泛使用。
- **校准数据多样性不足影响极端稀疏设置**：在 2:4 等高稀疏比下，StereoSet 等小规模校准集导致的性能退化问题尚未被系统研究。
- **压缩-公平性权衡缺乏统一度量**：现有研究多单独报告偏见分数，缺乏类似 DTO（Distance-to-Optimum）等综合指标来评估性能-公平性的联合优化效果。

## 核心贡献（创新点）
1. **首次将公平性目标形式化嵌入 SparseGPT 二阶重建框架**：通过引入成对输入差分项，推导出偏置感知 Hessian，从理论上证明压缩目标可显式惩罚表征差异的放大。
2. **设计 bias-aware saliency 与权重更新规则**：新 Hessian 同时影响剪枝 mask 选择（salience 计算）和二阶权重补偿，保持了 SparseGPT 的 $O(d_{hidden}^3)$ 计算复杂度。
3. **系统性验证了跨九类 LLM 的去偏效果**：在 UnQover、BBQ、CrowS-Pairs、DTO 等多个基准上，Debias-SparseGPT 在所有模型族和稀疏度（25%/50%/2:4）下均优于 SparseGPT。
4. **揭示了校准数据多样性对极端稀疏剪枝的关键作用**：在 2:4 设置下，用 UltraChat 扩充校准集可将 UnQover 准确率提升 +22.32%（相对 StereoSet-only），表明长上下文校准对保真压缩至关重要。

## 方法详解
- **目标函数（Eq. 2）**：在 SparseGPT 的层输出重建误差基础上，额外加入 $\|\mathbf{W}\Delta\mathbf{X} - \tilde{\mathbf{W}}\Delta\mathbf{X}\|_2^2$ 项，其中 $\Delta\mathbf{X} = \mathbf{X}_0 - \mathbf{X}_1$ 为成对 pro-/anti-刻板印象输入的表示差，惩罚剪枝前后输入差异的表征变化。
- **偏置感知 Hessian（Eq. 4）**：$\mathbf{H} = \mathbf{X}_0\mathbf{X}_0^\top + \mathbf{X}_1\mathbf{X}_1^\top + 2\Delta\mathbf{X}\Delta\mathbf{X}^\top$，相比 SparseGPT 的标准 Hessian，多出一项 $2\Delta\mathbf{X}\Delta\mathbf{X}^\top$，使得 Hessian 对输入对比敏感。
- **重要性度量（Eq. 5）**：$\varepsilon_p = \frac{w_p^2}{2[\mathbf{H}_\mathbf{w}^{-1}]_{pp}}$，使用修正后的 Hessian 计算 saliency，指导 mask 构建。
- **权重更新（Eq. 3）**：基于约束二次优化的拉格朗日解法，$\Delta\mathbf{w}^* = -\frac{w_p}{[\mathbf{H}_\mathbf{w}^{-1}]_{pp}}\mathbf{H}_\mathbf{w}^{-1}\mathbf{e}_p$，结构与 SparseGPT 一致，仅 Hessian 不同。
- **算法流程（Algorithm 1）**：块级剪枝，Cholesky 分解 $\mathbf{C}^\top\mathbf{C}=\mathbf{H}^{-1}$ 保证数值稳定性，lazy batched 传播补偿误差，整体复杂度 $O(d_{hidden}^3)$ 与 SparseGPT 相同。
- **实现细节**：基于 LLM-Compressor 框架，block size=128，Hessian 阻尼系数 0.01，校准数据使用 StereoSet 开发集（4212 对）；UltraChat 仅贡献 $\mathbf{X}_0\mathbf{X}_0^\top + \mathbf{X}_1\mathbf{X}_1^\top$ 部分（无 $\Delta\mathbf{X}$ 项）。

## 实验与结果
- **模型**：9 个 LLM（LLaMA-3.1-8B-IT、Vicuna-7B、Qwen-2.5-7B、Mistral-7B、Aya-Expanse-8B、Phi-4-Mini、Gemma-9B、DeepSeek-7B、Qwen-3-8B），涵盖指令微调与 base 模型。
- **基线**：Magnitude Pruning、Wanda、SparseGPT、Dense 原始模型。
- **稀疏设置**：非结构化 25%/50%、半结构化 1:4/2:4。
- **评估基准**：
  - 性能：WikiText-2 Perplexity、MMLU（zero-shot）、HellaSwag
  - 公平性：UnQover（Accuracy of N/S）、BBQ（Accuracy of N/S）、CrowS-Pairs（likelihood diff）
  - 综合：DTO（MMLU + UnQover 归一化空间的欧氏距离）
- **主要结果（Table 2，1:4 稀疏，Qwen-2.5-7B-IT）**：
  - UnQover：SparseGPT 67.35 → Debias-SparseGPT 74.41（+7.06，$p<0.01$）
  - BBQ：65.80 → 66.70（+0.90）
  - DTO：0.311 → 0.291（最优）
- **最强提升（Table 2，LLaMA-3.1-8B-IT，1:4）**：
  - UnQover：SparseGPT 35.60% → Debias-SparseGPT 60.46%（**+24.86pp**，超越所有基线，远超 Wanda 的 41.98%）
  - BBQ：67.10% → 70.70%（+3.60pp）
  - DTO：0.539 → 0.399（最优）
- **极端稀疏（2:4，Table 3/4）**：
  - StereoSet-only：UnQover 28.45%（SparseGPT）→ 24.94%（Debias），性能下降；DTO 0.616 vs 0.645
  - **+UltraChat（256条）**：Debias-SparseGPT UnQover 24.94% → **47.26%（+22.32pp）**，MMLU 48.16% → 54.17%（+6.01pp），DTO 0.645 → **0.494**，超越 SparseGPT+UltraChat（DTO 0.522）
- **效率（Table 5，Qwen-2.5-7B，2:4）**：吞吐量 73.05 tok/s（与 SparseGPT 相同），CO₂ 0.0376 kg/Mtok，DTO 0.494 < SparseGPT 0.522。
- **安全性（Table 16，Appendix F）**：Debias-SparseGPT 在 RealToxicityPrompts 和 HarmBench 上均低于 SparseGPT，不引入额外安全风险。
- **关键结论**：去偏效果在初始预测不确定性较高的模型（如 LLaMA，正确预测熵 0.94 vs Qwen 0.11）上更显著；宗教类校准数据的去偏效果最佳（DTO 0.453，Table 15）。

## 相关工作脉络
1. **SparseGPT**（Frantar & Alistarh, 2023）：本文的核心基础，基于 OBS 的二阶剪枝方法；Debias-SparseGPT 在其 Hessian 中引入配对输入差分项，而 SparseGPT 仅考虑重建误差。
2. **Wanda**（Sun et al., 2024a）：magnitude-activation 乘积剪枝，无需 Hessian 但依赖校准数据；本文方法在二阶精度和去偏目标上与之形成对比。
3. **Ramesh et al. (2023)、Hong et al. (2024)**：实证揭示了剪枝会恶化公平性，但未提出压缩时去偏方法；本文填补了这一空白。
4. **CrowS-Pairs / BBQ / UnQover**（Nangia et al., 2020; Parrish et al., 2022; Li et al., 2020）：本文采用的偏见评估基准，用于量化去偏效果。
5. **DTO 公平-性能权衡度量**（Han et al., 2022）：本文沿用其 Distance-to-Optimum 指标，将 MMLU 与 UnQover 归一化后计算欧氏距离。
6. **BiasEdit / Self-Debias / INLP**（Xu et al., 2025; Meade et al., 2022）：预训练或推理时去偏方法，非压缩阶段；本文首次将去偏目标嵌入后训练剪枝过程。

## 局限性与未来方向
- **仅单语言（英语）验证**：校准数据和评测基准均为英语，未验证多语言泛化能力。
- **未全面覆盖安全维度**：主要评估 representational bias，对 toxicity、robustness 仅做初步验证（Appendix F），未系统分析。
- **未深入分析稀疏模式**：对 bias-aware Hessian 如何影响逐层/逐列剪枝决策的分析较浅（Appendix G 仅提供初步结果）。
- **极端稀疏下 perplexity 退化严重**：2:4 设置下 WikiText-2 perplexity 从 7.14 升至 16.17，需要更丰富的校准数据弥补。
- **校准数据敏感度缺乏理论分析**：当前关于校准集大小/组成对 Hessian 影响的结论仅基于实验，未建立理论界 bound。
- **未来方向**：扩展至量化/蒸馏等其他压缩技术、探索自适应稀疏训练、参数高效微调（PEFT）、建立校准数据集的理论分析框架。

## 研究启发与可借鉴点
1. **二阶 Hessian 扩展可直接移植到其他模型压缩方法**：本文的 bias-aware Hessian 思路（$\mathbf{H} \leftarrow \mathbf{H} + 2\Delta\mathbf{X}\Delta\mathbf{X}^\top$）可自然推广至 quantization（如 Q-LORA）、distillation 等场景，为"压缩时同时优化公平性"提供通用框架。
2. **配对输入构造是低成本高收益的去偏信号**：仅需 ~4k 条 minimal pair 即可有效引导剪枝决策，说明校准数据的质量（结构化对比）比数量更重要，可作为后续研究的设计范式。
3. **DTO 指标可有效刻画压缩-公平性权衡**：建议团队在评估压缩方法时同步报告 DTO，而非仅关注单一 bias benchmark，以全面衡量性能-公平性 trade-off。
4. **预测熵可作为去偏效果的代理指标**：LLaMA 因 dense 模型不确定性更高而从去偏中获益更多，提示可基于初始不确定性进行分层去偏策略设计。
5. **半结构化剪枝（2:4）与长文本校准结合有巨大潜力**：UltraChat 实验证明扩充多样性校准数据可大幅缓解结构化稀疏的性能退化，为高效部署场景提供可行路径。

## 关键术语表
- **SparseGPT**：基于 Optimal Brain Surgeon 的单次一剪后训练剪枝方法，利用校准数据近似 Hessian 实现二阶精度，是当前 LLM 剪枝的主流基线。
- **Bias-aware Hessian**：在标准二阶 Hessian 基础上叠加成对输入差分项（$2\Delta\mathbf{X}\Delta\mathbf{X}^\top$），使剪枝过程显式感知并惩罚表征偏差放大。
- **UnQover**：由 Li et al. (2020) 构建的生成式偏见基准，通过信息不足的问题测试模型是否依赖刻板印象给出人口统计推断，核心指标为"Not Stated"预测准确率。
- **DTO (Distance-to-Optimum)**：归一化空间中到理想点（最大性能+最大公平性）的欧氏距离，越低表示性能-公平性权衡越优。
- **Calibration Data**：后训练压缩中用于近似 Hessian 的小规模校准数据集，本文使用 StereoSet（4212 对）及 UltraChat（256 条）作为校准源。
- **Saliency（重要性分数）**：$\varepsilon_p = w_p^2 / (2[\mathbf{H}^{-1}]_{pp})$，衡量移除某权重对目标函数的二阶影响，用于排序并选择剪枝位置。
- **Semi-structured Sparsity (N:M)**：每 M 个连续权重中保留 N 个，其余置零的模式，2:4 即每 4 个权重保留 2 个，支持 Tensor Core 硬件加速。
- **Representational Bias**：模型在生成中对特定人口统计属性（性别、宗教等）的系统性刻板印象倾向，本文重点关注的去偏目标。

## 可复现要素
- **数据集**：StereoSet（开发集，4212 对）、UltraChat（256 条示例）、UnQover、BBQ、CrowS-Pairs、WikiText-2、MMLU、HellaSwag、RealToxicityPrompts、HarmBench；均为公开基准，论文未提及自建数据集。
- **代码**：已开源，集成于 LLM-Compressor 包（Apache-2.0 许可证），论文声明 "We make the implementation of Debias-SparseGPT openly available"。
- **模型权重**：使用的 9 个 LLM 均通过 HuggingFace 公开获取（见 Table 7），论文未发布新模型权重。
- **关键超参**：block size = 128，Hessian 阻尼系数 = 0.01，saliency stride $B_s = M$（半结构化），配对输入 token 数对齐。
