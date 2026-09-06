---
title: "Debias-SparseGPT-Bias-Aware-Pruning-for-Large-Language-Model"
source: https://arxiv.org/pdf/2609.02496v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:45:45"
field: "大语言模型压缩与公平性"
keywords: ["模型压缩", "大语言模型剪枝", "公平性去偏", "二阶优化", "SparseGPT", "Bias-Aware Pruning"]
innovations: ["提出偏置感知 Hessian 在 SparseGPT 二阶剪枝中融入公平性目标", "系统性验证剪枝引发的偏见放大及缓解方法", "揭示校准数据对高稀疏度下公平性-性能权衡的影响机制"]
benchmarks: ["UnQover", "BBQ-HELM", "CrowS-Pairs", "MMLU", "HellaSwag", "RealToxicityPrompts", "HarmBench"]
---

# 论文速读：Debias-SparseGPT: Bias-Aware Pruning for Large Language Models

## 一句话总结
本文提出 Debias-SparseGPT，一种在稀疏化压缩过程中融入公平性约束的后训练剪枝方法，通过引入基于人口统计对比输入的 Hessian 二阶项，在不损失模型性能的前提下有效缓解模型压缩引发的偏见放大问题。

## 研究问题与动机
- **核心问题**：LLM 的权重稀疏化方法（如 SparseGPT）虽能提升部署效率，但会放大模型已有的偏见，导致输出随 prompt 中的 persona 线索产生显著差异。
- **现有方法不足**：已有研究仅报告剪枝会加剧偏见，但缺乏在压缩过程中直接缓解偏见的方法；现有剪枝方法（Magnitude、Wanda、SparseGPT）均无公平性目标。
- **评估缺口**：模型压缩的主流评估聚焦于 perplexity 和下游准确率，安全相关维度（毒性、偏见、公平性）受到较少关注。
- **实证动机**：图 3 展示在 UnQover 基准上，随着稀疏度增加，模型从"Not stated"回答向特定群体标签偏移，错误率上升。

## 核心贡献（创新点）
1. **首次提出剪枝时的去偏方法**：与已有工作仅报告剪枝有害不同，本文提出 Debias-SparseGPT，在压缩过程中直接干预偏见的放大。
2. **推导了偏置感知压缩公式**：通过将 Hessian 从标准形式扩展为包含配对输入差项 $\Delta\mathbf{X}\Delta\mathbf{X}^\top$，使掩码构建和二阶权重重建均受公平性目标影响，同时保持 SparseGPT 的计算复杂度。
3. **系统性实验验证**：在 9 个 LLM（包括 7 个指令微调模型和 2 个基础模型）、多种稀疏度（25%、50%、1:4、2:4）下验证，一致证明该方法在保持 perplexity 和零样本准确率的同时降低剪枝引发的偏见。
4. **发现校准数据的影响机制**：在高稀疏度（2:4）下，通过补充长上下文、内容丰富的 UltraChat 校准数据可显著提升公平性-性能权衡。

## 方法详解
- **目标函数设计**：将剪枝建模为二阶重构问题，在标准 SparseGPT 目标基础上增加一项惩罚配对输入表征差异的变化：
  $$\hat{\mathbf{W}} = \arg\min_{\mathbf{M}, \mathbf{W}'} \left[\frac{1}{2}\sum_{i\in\{0,1\}}\|\mathbf{W}\mathbf{X}_i - \tilde{\mathbf{W}}\mathbf{X}_i\|_2^2 + \|\mathbf{W}\Delta\mathbf{X} - \tilde{\mathbf{W}}\Delta\mathbf{X}\|_2^2\right]$$
  其中 $\mathbf{X}_0$ 和 $\mathbf{X}_1$ 为成对的 pro-/anti-stereotypical 输入，$\Delta\mathbf{X} = \mathbf{X}_0 - \mathbf{X}_1$。
- **偏置感知 Hessian**：输入空间 Hessian 扩展为 $\mathbf{H} = \mathbf{X}_0\mathbf{X}_0^\top + \mathbf{X}_1\mathbf{X}_1^\top + 2\Delta\mathbf{X}\Delta\mathbf{X}^\top$，第三项专门捕获配对输入差异的方差。
- **权重更新规则**：基于 OBC/OBS 框架推导得 $\Delta\mathbf{w}^\star = -\frac{w_p}{[\mathbf{H}_\mathbf{w}^{-1}]_{pp}}\mathbf{H}_\mathbf{w}^{-1}\mathbf{e}_p$，重要性度量（saliency）为 $\varepsilon_p = \frac{w_p^2}{2[\mathbf{H}_\mathbf{w}^{-1}]_{pp}}$。
- **算法实现**：采用块级剪枝策略，按列分批处理权重矩阵，使用 Cholesky 分解提高数值稳定性；对 Semi-structured N:M 稀疏设置，stride $B_s = M$，每块选择 saliency 最低的 $N$ 个权重进行剪枝。

## 实验与结果
- **模型与数据集**：9 个 LLM（LLaMA-3.1-8B-IT、Vicuna-7B-v1.5-IT、Qwen-2.5-7B-IT、Mistral-7B-v0.3-IT、Aya-Expanse-8B-IT、Phi-4-4B-Mini-IT、Gemma-9B-IT、Qwen-3-8B、DeepSeek-8B）；校准数据为 StereoSet 开发集（4212 对例句）。
- **评估基准**：偏见评估用 UnQover、BBQ、CrowS-Pairs；性能评估用 WikiText-2 perplexity、MMLU、HellaSwag；综合指标为 DTO（Distance-to-Optimum）。
- **核心结果（1:4 稀疏）**：
  - LLaMA：UnQover 从 35.60%（SparseGPT）提升至 60.46%（Debias-SparseGPT），DTO 从 0.539 降至 0.399。
  - Qwen-2.5-7B：UnQover 从 70.60% 提升至 74.41%，MMLU 保持 67.73%，DTO 从 0.311 降至 0.291。
  - Vicuna：UnQover 从 17.82% 提升至 21.54%，DTO 从 0.695 降至 0.674。
- **稀疏度影响**：25% 和 50% 非结构化稀疏下，Debias-SparseGPT 在 UnQover 上均显著优于 SparseGPT；2:4 半结构化稀疏下性能下降较大，但加入 UltraChat 校准数据后 DTO 从 0.645 降至 0.494。
- **效率**：内存复杂度与 SparseGPT 相同 $O(d^2 + nd)$，吞吐量一致（27.54 → 73.05 tok/s）。

## 相关工作脉络
- **SparseGPT**（Frantar & Alistarh, 2023）：本文的直接基线，采用二阶重建框架进行单层剪枝，但不考虑公平性。
- **Wanda**（Sun et al., 2024a）：基于 magnitude 和 activation norm 的剪枝方法，使用校准数据但不进行权重更新。
- **OPTIMAL BRAIN DAMAGE/SURGEON**（LeCun et al., 1989; Hassibi & Stork, 1992）：二阶剪枝的理论基础，本文在此基础上扩展公平性目标。
- **StereoSet/UnQover/BBQ**：用于测量语言模型中刻板印象偏见的经典基准，本文将其用于压缩后的公平性评估。
- **CrowS-Pairs**（Nangia et al., 2020）：最小对立句对数据集，用于测量偏好偏向，但本文发现其对剪枝去偏效果较不敏感。
- **DTO 指标**（Han et al., 2022）：用于量化公平性-性能权衡的单一综合指标，本文用于跨方法比较。

## 局限性与未来方向
- **单语言限制**：实验仅在英语上进行，未验证多语言场景下的泛化能力。
- **安全维度覆盖不足**：主要关注表征偏见，未系统评估毒性、有害生成等其他安全维度（附录 F 提供初步分析）。
- **稀疏模式分析不够深入**：未对学到的稀疏模式进行全面的层/列级分析。
- **高稀疏度下的校准敏感性**：2:4 等激进稀疏设定下模型质量显著下降，需要更丰富的校准数据。
- **理论界未建立**：校准集大小和组成对偏置感知 Hessian 估计的变异缺乏理论边界分析。
- **未来方向**：可扩展至量化、蒸馏等其他压缩方法；结合 sparse training 或 parameter-efficient fine-tuning 进一步提升效果。

## 研究启发与可借鉴点
1. **二阶剪枝 + 公平性目标的结合思路**：将配对输入差异项加入 Hessian 是一个简洁有效的去偏正则化策略，可迁移到其他二阶优化方法中。
2. **校准数据对压缩后公平性的影响机制**：发现 UltraChat 等长上下文数据可显著改善高稀疏度下的公平性-性能权衡，提示校准数据的选择对压缩后模型的公平性至关重要。
3. **预测不确定性解释改进幅度差异**：LLaMA 在密集模型上预测熵更高，剪枝后改进更大，这一发现为理解不同模型的去偏效果差异提供了新视角。
4. **DTO 作为单一综合评估指标**：将性能（MMLU）和公平性（UnQover）纳入统一框架，便于跨方法比较，值得在后续研究中推广。
5. **实现兼容性**：基于 LLM-Compressor 框架实现，可直接与 quantization 等其他压缩技术组合使用。

## 关键术语表
- **Debias-SparseGPT**：本文提出的后训练剪枝方法，通过在 SparseGPT 的二阶目标中加入配对输入差异项来减少剪枝引发的偏见。
- **Bias-aware Hessian**：扩展的二阶 Hessian 矩阵，包含 $\mathbf{X}_0\mathbf{X}_0^\top + \mathbf{X}_1\mathbf{X}_1^\top + 2\Delta\mathbf{X}\Delta\mathbf{X}^\top$，用于捕捉成对输入差异对权重的敏感度。
- **Pro-/anti-stereotypical pairs**：成对的促进和反对刻板印象的输入文本，用于构建去偏目标中的差分项。
- **DTO (Distance-to-Optimum)**：公平性-性能权衡的综合指标，计算模型在归一化空间中距"乌托邦点"的欧氏距离。
- **UnQover**：衡量语言模型在语境不足时是否会产生群体偏见回答的生成式基准测试。
- **Semi-structured N:M sparsity**：每个包含 M 个权重的块中强制设置 N 个为零的半结构化稀疏模式（如 2:4）。

## 可复现要素
- **代码**：已开源，集成于 llm-compressor 包，兼容 Hugging Face transformers（论文未提供具体 GitHub 链接，声明将于发表后以 Apache-2.0 开源）。
- **模型**：9 个公开 LLM，许可证各异（LLaMA Community License、Apache-2.0、CC BY-NC 等），均在各自官方仓库提供。
- **数据集**：
  - 校准数据：StereoSet 开发集（4212 对例句）；UltraChat（256 个示例，约 15k tokens）。
  - 评估基准：UnQover、BBQ-HELM、CrowS-Pairs、WikiText-2、MMLU、HellaSwag、RealToxicityPrompts、HarmBench。
- **关键超参**：block size = 128，Hessian damping fraction = 0.01，默认应用于每层的 7 个权重矩阵（q/k/v/o_proj 和 gate/up/down_proj）。
- **硬件环境**：2× NVIDIA A100 GPU（80 GB 显存）。
