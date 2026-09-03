---
title: "What-Does-Activation-Steering-Control-Attribution-Across-Ans"
source: https://arxiv.org/pdf/2608.22985v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:04:22"
field: "大语言模型可解释性与对齐"
keywords: ["activation steering", "contrastive activation addition", "interpretability", "representation engineering", "cross-encoding evaluation", "extraction-index following"]
innovations: ["提出固定干预+反事实重编码的交叉评估框架以分离语义/索引/行跟随", "揭示 CAA 效果集中在低秩输出敏感子空间（15.4%能量保留96.3%效应）"]
benchmarks: ["NormBank", "MNLI", "Social Chemistry 101 (SC101)"]
---

# 论文速读：What-Does-Activation-Steering-Control-Attribution-Across-Ans

## 一句话总结
论文提出**交叉编码评估框架**（Cross-Encoding Steering Evaluation），通过固定干预方向并反事实地重新编码答案选项，分离出激活引导效果的三种可能归因——语义标签跟随、提取索引跟随、提取行跟随；在 NormBank 上发现 CAA 方向主要跟随**提取索引**而非语义标签，且该效果集中在后期层的一个低秩输出敏感子空间中（15.4% 能量保留 96.3% 效果）。

## 研究问题与动机
- **核心归因问题**：当激活引导提高了报告分数，这个增益真正控制的是什么？是目标语义类别、选项标识符的提取位置（如 C），还是其显示行？现有文献中该基本问题从未被明确分离。
- **评估偏差风险**：现有方法（CAA、ITI 等）通常在构建方向的同一 answer encoding 下进行评估，报告的增益可能仅反映与答案标识符（如 A/B/C）的兼容性，而非真正的语义控制。
- **缺乏反事实归因工具**：已有工作无法在固定干预条件下区分"追踪语义"与"追踪索引"，导致 steering gain 本身不足以识别干预的实际控制目标。
- **多粒度行为结论不一致**：多项选择题（MCQ）评分与开放生成行为可能给出矛盾的结论，现有评估体系未系统考察这一分歧。

## 核心贡献（创新点）
1. **提出固定干预+反事实重编码的交叉评估框架**：与已有 work 仅在单一编码下报告 gain 的做法不同，本文冻结方向，系统改变标签-标识符映射、词汇表、行顺序，实现三类跟随形式的因果分离。
2. **定义并量化提取索引跟随（extraction-index following）**：首次将"方向是否追踪提取时的选项索引"作为一个独立可测量的效应，并给出严格的因子审计设计（6 映射 × 3 词汇 × 6 行序 = 108 条件）。
3. **揭示效果在输出敏感子空间中的高度集中**：基于 logit 梯度 SVD 构造局部标识符读出子空间，发现仅含 15.4% 方向能量的投影分量保留了 96.3% 的提取索引效果，而残差分量几乎无效。
4. **跨方法/任务/模型的归因异质性图谱**：ITI 复现了 NormBank 上的提取索引偏好；MNLI 总体支持提取索引跟随，但 Qwen 反向；SC101 则总体偏好语义标签跟随，证明归因必须实证测量而非假设。
5. **MCQ 与开放生成结论分离合规范**：在公开 CAA 协议下，幻觉和谄媚在两种评估中一致正向，但拒绝行为在 MCQ 中有大幅增益却在开放生成中无显著变化，指出单一评估粒度的不足。

## 方法详解
- **CAA 方向构建**：对对比对 $(x_i^+, x_i^-)$ 提取预答案位置 $p_{\mathrm{ext}}$ 的残差流差分均值 $\mathbf{v}_{\mathrm{raw}} = \frac{1}{N_c}\sum_i (\mathbf{h}(x_i^+) - \mathbf{h}(x_i^-))$，在推理时注入 $\mathbf{h} \leftarrow \mathbf{h} + \alpha \mathbf{v}_{\mathrm{raw}}$。
- **交叉编码评估**：固定 $\mathbf{v}_{\mathrm{raw}}$、层、位置、剂量，仅改变 answer encoding（标签→标识符映射、词汇表 A/B/C vs X/Y/Z vs 1/2/3、行顺序），在同一编码内比较目标-源 margin 变化 $\Delta m_f$，不同编码间不比较绝对值。
- **三类效应定义**：语义标签效应 $\Delta m_{\mathrm{sem},\pi}$ 跟踪当前映射下的目标标签；提取索引效应 $\Delta m_{\mathrm{idx},\pi}$ 跟踪提取时目标的标识符索引（对齐 A/X/1、B/Y/2、C/Z/3）；提取行效应跟踪提取时目标的显示行。三者之差构成提取索引优势 $A_{\mathrm{idx}} = \Delta m_{\mathrm{idx}} - \Delta m_{\mathrm{sem}}$。
- **因子审计**：对每个冻结方向遍历 108 种条件（6 映射 × 3 词汇 × 6 行序），分别计算三类 margin 变化，通过 $\Delta m_{\mathrm{idx}} - \Delta m_{\mathrm{row}}$ 分离标识符索引与显示行。
- **输出敏感子空间定位**：对训练 prompt 计算相邻 token logit 差对 hidden state 的梯度 $\mathbf{g}_{x,a,b} = \nabla_{\mathbf{h}}[z(o_a) - z(o_b)]$，堆叠后做 SVD，选取最小秩 $r$ 使验证梯度平方范数捕获率≥90%；将 $\mathbf{v}_{\mathrm{raw}}$ 正交分解为投影 $\mathbf{v}_{\parallel}$ 和残差 $\mathbf{v}_{\perp}$，分别测试其提取索引效应与能量占比。
- **L2 匹配随机控制**：使用与 CAA 方向同范数的 Gaussian 向量作为扰动控制，减去随机方向效应得到随机调整估计； group-cluster bootstrap 联合重采样完整 setting–behavior 组。

## 实验与结果
- **数据集**：NormBank（主要审计，T/N/E 三标签，上下文匹配）、MNLI（entailment/neutral/contradiction，相同前提配对）、SC101（bad/ok/good，行动级配对）。
- **模型**：Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Gemma-2-9B-IT。
- **NormBank 核心结果**：在 5 种 A/B/C 重映射下，提取索引效应（3.00–4.13）均显著超过语义标签效应，提取索引优势达 1.95–4.04（Table 1）；因子审计确认效果来自索引而非行位置（Llama/Mistral/Gemma 支持索引跟随，Qwen 例外 favor 语义标签）。
- **层级定位**：归一化匹配后，提取索引优势在 50% 深度为负（−0.061），在 75% 和 87.5% 深度转正（+0.236 / +0.208），效果集中于后期层（Figure 3）。
- **位置定位**：从预答案位置改到场景末尾提取，提取索引概率 margin 效应从 0.218 骤降至 0.004（降幅 98.3%）。
- **子空间集中**：投影分量含 15.4% 方向能量但保留 96.3% 提取索引效应；残差含 84.6% 能量仅保留 1.0%（Table 22）。跨词汇表 A/B/C > X/Y/Z > 1/2/3 衰减，但投影始终优于残差。
- **ITI 复现**：在 3 个保留模型上，提取索引优势在全部 5 种映射下为正，语义标签效应仅在 2/5 映射下为正（Table 2）。
- **跨任务**：MNLI 总体支持提取索引跟随（5/5 映射优势为正，Qwen 反向）；SC101 反转，总体偏好语义标签跟随（0/5 映射优势为正）。
- **开放生成分歧**：幻觉 MCQ +3.31 ↔ 开放 +1.60（一致正向）；拒绝 MCQ +3.28 ↔ 开放 −0.13（显著分歧）；谄媚 MCQ −0.48 ↔ 开放 +1.11（方向相反）。

## 相关工作脉络
- **Rimsky et al. [2024] CAA / Zou et al. [2023] Representation Engineering**：本文在其疗效报告基础上追问"gain 到底控制什么"，而 prior work 未在固定方向下做跨编码归因。
- **Li et al. [2023] ITI**：本文用 ITI 复现 NormBank 排序，证明提取索引跟随不依赖 CAA 的特定构造方式，具有方法通用性。
- **Shao et al. [2026] DecodeShare**：同样利用投影–干预逻辑，但 DecodeShare 追踪 task-shared decode-time subspace，本文聚焦 identifier-readout 局部子空间，归因目标不同。
- **Tan et al. [2024] / Pres et al. [2024]**：prior 工作指出 steering 对 input 敏感、需 context-matched 评估；本文在此基础上进一步冻结干预、反事实改变编码，实现更严格的因果归因。
- **Fraile Navarro et al. [2026]**：发现 output-stage scaffold features 可主导 decision logits；本文与之互补，证明 CAA 的 extraction-index following 正是此类输出敏感依赖的机制之一。
- **Ye et al. [2026]**：指出 steering efficacy 强依赖于 activation source selection；本文的层级扫描和位置定位与其发现一致并深化——效果集中在 pre-answer 后期层。

## 局限性与未来方向
- **覆盖范围有限**：六映射审计仅覆盖三个三标签任务和四个 instruction-tuned 模型族，完整因子审计仅在 NormBank 上进行。
- **子空间仅功能定位**：梯度定义的标识符读出子空间在功能上定位了效果，但未识别产生该依赖的完整 circuit。
- **层级归一化局限**：层间 $L_2$ 匹配仅等价方向幅度，不等价各层模型敏感度；位置定位仅用单一注入位点。
- **开放生成与 MCQ 结论不同步**：本文揭示了这一分歧，但尚未建立统一的跨粒度评估理论。
- **未解决的非identifiability**：Venkatesh & Kurapath [2026] 指出行为等价的 steering vectors 存在非唯一性，本文通过固定方向做归因而非解决该问题。

## 研究启发与可借鉴点
- **交叉编码框架可直接迁移**：任何已发布的 steering direction（价值观、道德、文化适应等）均可套用此框架检验其真实归因，无需重新训练。
- **因子审计设计具有通用性**：6 映射 × 词汇表 × 行顺序的正交设计可推广至任意三标签或多标签分类任务的 steering 归因。
- **输出敏感子空间定位可作为标准诊断工具**：基于 logit 梯度 SVD 的投影–残差分解成本低、解释力强，建议纳入 steering 论文的默认实验。
- **多层级评估协议值得借鉴**：MCQ margin + 因子审计 + 开放生成 judge 的组合可防止单一指标误导，建议作为后续工作的 baseline pipeline。
- **Mapping-balanced direction 作为对照**：平均全部六种映射提取的方向可大幅降低提取索引依赖，为"如何构造更语义忠于方向的 steering vector"提供实证依据。

## 关键术语表
- **Activation Steering**：在推理时向 LLM 隐藏状态注入人工构造的向量以改变其行为，无需更新模型权重。
- **Contrastive Activation Addition (CAA)**：对正负样本对的隐藏状态求差分并取均值，得到 steering direction，在推理时线性叠加注入。
- **Semantic-label Following**：steering 效果追踪当前测试编码下被赋予目标语义的标识符（即"做什么"）。
- **Extraction-index Following**：steering 效果追踪提取方向时的标识符索引位置（即"第几个选项"），与当前语义分配无关。
- **Cross-Encoding Steering Evaluation**：本文提出的归因框架，固定干预方向，反事实地改变答案编码以分离语义/索引/行跟随。
- **Output-sensitive Subspace**：由 option-logit 梯度张成的低维子空间，捕获隐藏状态对最终选项标识符 logits 的一阶敏感性。
- **Extraction-index Advantage ($A_{\mathrm{idx}}$)**：提取索引效应与语义标签效应的差值，正值表示方向更跟随索引而非语义。
- **Factorial Audit**：交叉语义映射（6）、标识符词汇表（3）、显示行顺序（6）共 108 种条件的完整归因审计设计。

## 可复现要素
- **数据集**：NormBank（公开）、MNLI（公开）、SC101（公开）；论文提供了严格的 train/val/test 划分，pair ID、context group、endpoint 跨 split 无重叠（Table 7）。
- **代码/权重**：论文声明配套代码与评估脚本随论文发布；方向提取、因子审计、子空间分解均有详细附录说明。
- **关键超参**：CAA 剂量 $\alpha = 0.8$（NormBank/MNLI）或 $\alpha = 1.0$（SC101）；层位置为各模型 75% 深度附近（Qwen 20、Llama 23、Mistral 23、Gemma 31，零基索引）；随机控制数量 $K=5$（主实验）或 $K=10$（子空间分析）；bootstrap 重采样 5,000 次。
