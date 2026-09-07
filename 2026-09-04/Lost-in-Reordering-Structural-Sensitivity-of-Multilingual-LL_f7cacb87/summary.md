---
title: "Lost-in-Reordering-Structural-Sensitivity-of-Multilingual-LL"
source: https://arxiv.org/pdf/2609.03511v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:19:45"
field: "多语言大语言模型鲁棒性评测"
keywords: ["multilingual LLM", "structural robustness", "activation patching", "GSM8K", "Hindi", "Malayalam", "compositional reasoning", "DO RA"]
innovations: ["构建IndicReStruct多语言结构扰动数据集（重排+语态转换）", "发现多语言LLM在语义保持结构扰动下普遍存在显著性能退化", "首次用残差流激活补丁定位多语言推理失败的关键因果组件（Entity-Quantity Combinations）"]
benchmarks: ["GSM8K-Reordered", "GSM8K-Voice", "ARC-Challenge-Indic"]
---

# 论文速读：Lost-in-Reordering-Structural-Sensitivity-of-Multilingual-LL

## 一句话总结
本文构建了 **IndicReStruct** 数据集（GSM8K-Reordered 与 GSM8K-Voice），通过成分重排和主动-被动态转换等**语义保持的结构扰动**，系统评估了多语言 LLM 在印地语和马来语数学推理任务中的结构性敏感度；发现所有评测模型均出现**显著且一致的性能下降**，且轻量微调无法根本改善该问题。

---

## 研究问题与动机
1. **核心问题**：多语言 LLM 在面对"语义等价但句法结构不同"的输入时，是否具备真正的组合式语义推理能力？现有工作主要关注词汇级扰动，对**控制性结构扰动**的研究严重不足。
2. **现有方法不足**：当前多语言 LLM 主要依赖**表面统计模式**而非深层语义理解；且绝大多数预训练数据为英语，非英语语言代表性不足，导致其在自由语序语言（如印地语、马来语）上的鲁棒性存疑。
3. **语言类型学动机**：印地语和马来语属于**相对自由的 SOV 语序语言**，日常口语中频繁通过成分重排实现强调或语用目的，是检验模型结构性敏感度的理想场景。
4. **可解释性缺口**：即便观察到性能下降，尚不清楚模型内部的哪些计算组件（层级、词类）对推理失败或恢复起关键因果作用。

---

## 核心贡献（创新点）
1. **IndicReStruct 数据集**：构建了两个面向多语言数学推理的结构扰动基准（GSM8K-Reordered 和 GSM8K-Voice），覆盖印地语与马来语，**语义保持性经余弦相似度和 LLM-as-judge 双重验证**。与已有扰动数据集的本质区别在于：扰动操作受语言学约束（语义凝聚词组 + 边界保护），而非随机词级噪声。
2. **系统性结构敏感度评测**：在 6 个 LLM × 4 种提示策略下全面评测，揭示所有模型在重排和语态转换下均出现显著退化，证明**表面句法敏感性是跨模型、跨语言、跨提示的共性问题**。
3. **机制可解释性分析**：首次在多语言数学推理的结构扰动场景下，使用**残差流激活补丁（residual-stream activation patching）**定位因果组件，发现中间层（Layer 5–25）和 Entity-Quantity Combinations 词类对推理恢复贡献最大，为结构性鲁棒性的表征研究提供新视角。

---

## 方法详解

### 1. 结构化数据扰动方法
**（1）GSM8K-Reordered（成分重排）**
- **语义凝聚词组划分**：采用 **Samanvaya 分组原则**（Dangarikar et al., 2024），将语义上不可分割的词组作为移动单位（如形容词修饰名词、复合动词、数量短语等），使用 Gemini-2.5-Pro 生成分组，避免语义破坏性重排（如 "बच्ा मैदान में" → 合法重排单元为 [बच्ा] [मैदान] [में]，而非 [ब] [च्ा] 等字符级拆分）。
- **约束性伪随机置换**：对每个句子的词组序列应用伪随机排列 π(Gi)，约束条件包括：
  - 边界标记（括号、标点）作为不可跨越屏障
  - 连接词集合（如 hindi 的 **और, लेकिन, क्योंकि, अगर** 等约 40 个）作为话语锚点，禁止跨越
  - 生成 5 组扰动数据（seed 不同），选取中位数性能对应的 seed=42 用于评测
- **生成 5 个重排版本**以确保统计稳健性

**（2）GSM8K-Voice（语态转换）**
- 使用 Gemini-2.5-Pro 将主动语态与被动语态相互转换，同时严格保持命题意义、时态、数字和命名实体不变。

### 2. 数据质量验证
- **余弦相似度对比**（Table 1）：Proposed 方法 vs. Fully Random 重排在多个多语言句子嵌入模型下对比：
  - Hindi: MiniLM 98.22% / LaBSE 98.50% / DistilUSE 98.46% / IndicBERTv2 98.56%
  - Malayalam: MiniLM 93.15% / LaBSE 89.37% / DistilUSE 99.64% / IndicBERTv2 99.24%
  - 随机重排相似度高至 90%+，但 Proposed 显著更高（尤其 Hindi），证明语义保持性
- **LLM-as-judge 质量评分**（Gemini-2.5-Flash）：三档评级（Excellent/Good/Reject），Excellent 占 41.9%，Good 占 48.7%，Reject 仅 9.4%
- **人工验证**：60 个样本人工标注，Llama 判定趋于保守

### 3. 评估设置
- **6 个 LLM**：Gemma-2-9B-it、Gemma-2-27B-it、GPT-OSS-20B、Llama-3.1-8B-it、Param-2-17B-A2.4B（MoE）、Qwen3-30B-A3B（MoE）
- **4 种提示策略**：Zero-shot CoT、One-shot CoT、Three-shot CoT、Plan & Solve
- **指标**：GSM8K 标准答案精确匹配（##### 分隔符后数值）
- **微调实验**：Gemma-2-9B-it + DoRA/LoRA，r=16, α=32, dropout=0.05，80/20 划分

### 4. 机制可解释性（激活补丁）
- **设置**：选取原问题模型答对、重排问题答错的样本（Hindi 65 题，Malayalam 67 题）
- **方法**：Denoising activation patching——将 clean run（原问题正确推理）的残差流激活替换到 perturbed run（重排问题错误推理）的对应位置，检验是否正确恢复
- **词类分组**（9 类）：Subject、Object、Verb、Connectors、Operation words、Number、Unit、Entity-Quantity Combinations、Punctuation
- **工具**：TransformerLens 库

---

## 实验与结果

### 主实验（Table 2）：重排扰动
| 语言 | 最强模型（Original） | 重排后最差下降 | 关键发现 |
|------|---------------------|---------------|---------|
| Hindi | GPT-OSS-20B: 84.00% (ZS-CoT) | Llama-3.1-8B-it: ↓31.21% (56.63→25.42) | 所有模型均显著下降 |
| Malayalam | GPT-OSS-20B: 84.46% (ZS-CoT) | Param-2: ↓33.08% (66.49→33.41) | 下降幅度普遍在 6–33% |

**关键结论**：
- **GPT-OSS-20B** 在两类语言中结构性鲁棒性相对最强（Hindi ↓10.36%，Malayalam ↓9.50%）
- **Llama-3.1-8B-it** 和 **Param-2** 对重排极度敏感
- 多-shot 提示**未能缓解**性能下降，反而有时加剧（如 Hindi ZS-CoT 76.30→61.78 vs. 3-shot 74.75→58.12）

### ARC-Challenge-Indic 泛化（Table 3）
- Gemma-2-27B-it: Hindi 重排下降 1.91–3.56%
- Gemma-4-12B-it: Hindi 重排下降 3.05–3.91%
- 表明**结构性敏感度不限于 GSM8K，在科学 QA 任务同样存在**

### 语态转换扰动（Table 4）
- Hindi 下降幅度相对较小（↓0.36% ~ ↓29.41%）
- Malayalam 下降更显著（↓0.49% ~ ↓23.28%）
- Llama-3.1-8B-it 在 Malayalam Voice 下从 19.86% 暴跌至 7.13%

### 微调实验（Table 5, 6）
| Setting | GSM8K-Reordered | GSM8K-Voice |
|---------|----------------|-------------|
| Zero-shot CoT | 61.78% | 76.30% |
| LoRA | 41.59% | 40.94% |
| DoRA | 46.13% | 51.25% |

- **DoRA > LoRA**，但两者均**远低于零样本提示基线**
- 归因：微调语料仅约 7.5k 样本，模型可能过拟合扰动表面模式而非学习深层语义不变性
- **核心结论**：结构性鲁棒性无法通过提示或轻量微调解决，需要更根本的训练/架构干预

### 错误分类（Figure 2, Table 8）
9 类错误：Overthinking、Incorrect Math Operation、Skipped Step、CoT Language Mismatch、Semantics Drift、**Entity/Quantity Misalignment**（最高频）、Answer Extraction Error、Irrelevant Reasoning、Others

### 激活补丁分析（Figure 3, 4）
- **最高恢复效率层**：中间层（Layer 5–25）
- **最高恢复率词类**：Entity-Quantity Combinations（与错误分类中最高频错误一致）
- Number、Verb、Punctuation 也表现较好
- **关键发现**：推理失败不仅是计算错误，更是模型**内部组织跨句实体-数量关联能力**的破坏

---

## 相关工作脉络
1. ** Compositional Relation Reasoning（Zhao & Zhang, 2024）**：指出 LLM 常依赖捷径统计规律而非系统性组合推理——本文在其基础上将研究扩展到**多语言自由语序场景**的结构扰动。
2. **Structural/Lexical Sensitivity（Cheng et al., 2024; Wang et al., 2025; Kostić et al., 2026）**：证明句子结构变化影响模型预测——本文与之区别在于：①关注**语义保持**的控制性扰动而非任意噪声；②聚焦**非英语多语言**场景。
3. **Free Word Order 语言学（Sabel & Saito, 2005; Kulkarni et al., 2015）**：论述印地语等语言的自由语序特性——本文为首次将该语言学特性系统引入 **LLM 推理鲁棒性评测**。
4. **Mechanistic Interpretability（Meng et al., 2022; Zhang & Nanda, 2024）**：激活补丁已用于事实记忆定位——本文首次将其应用于**多语言结构扰动下的推理失败归因**。
5. **Multilingual Math Reasoning（Kamath et al., 2025; Sarvam AI, 2025b）**：构建印地语/马来语 GSM8K 翻译——本文在其基础上进一步施加**结构化扰动**以探测深层鲁棒性。
6. **PEFT for Reasoning（Liu et al., 2024 DoRA; Hu et al., 2022 LoRA）**：DoRA 在数学推理上优于 LoRA——本文发现即使最优 PEFT 仍无法修复结构性敏感度，指向更深层问题。

---

## 局限性与未来方向
1. **语言覆盖有限**：仅评测印地语和马来语两种印度语言，结论可能无法直接推广至其他类型学语言（如土耳其语、日语、藏语等自由语序语言）。
2. **任务范围有限**：仅涉及数学推理（GSM8K）和科学 QA（ARC-Challenge），未探索**常识推理、符号推理、长上下文多跳推理**等场景下的结构性敏感度。
3. **微调数据规模不足**：~7.5k 样本可能不足以让模型学会深层语义不变性，需更大规模、更多样化的结构性扰动训练数据。
4. **MoE 模型可解释性受限**：激活补丁仅在 Dense 模型（Gemma-2-9B-it）上实施，MoE 架构下的路由机制与结构性敏感度的关系尚不明确。
5. **未来方向**：①扩展至更多自由语序语言；②探索跨任务的结构性鲁棒性；③设计面向结构性不变性的训练方法（如对抗训练、数据增强、架构修改）。

---

## 研究启发与可借鉴点
1. **语义凝聚词组 + 约束性重排**的数据扰动方案：可迁移至其他自由语序语言（藏语、泰米尔语等）的鲁棒性评测，也可用于构建**多语言结构扰动基准**。
2. **激活补丁用于故障归因**的范式：将推理失败分析与内部表征因果定位结合，可作为**多语言 LLM 可靠性诊断**的标准工具链。
3. **Entity-Quantity Combinations 作为关键诊断信号**：该词类的恢复率最高且错误占比最大，提示**实体-数量对齐**是多语言数学推理的脆弱环节，可在训练数据构造中针对性增强。
4. **DoRA vs. LoRA 在多语言微调中的表现对比**：可为团队多语言 PEFT 策略选择提供参考依据。
5. **LLM-as-judge 质量验证框架**：三档语义保持性评分（Excellent/Good/Reject）+ 人工抽查，可作为**任何自动数据扰动流水线**的质量控制模板。

---

## 关键术语表
- **IndicReStruct**：本文构建的多语言结构扰动数据集，包含 GSM8K-Reordered 和 GSM8K-Voice 两个变体，覆盖印地语和马来语。
- **Semantic-preserving structural perturbation**：保持语义不变的结构扰动，本文指成分重排和主动-被动态转换两种操作。
- **Samanvaya grouping principle**：源自印度语言学的语义凝聚词组分组原则，将语法/语义上不可分割的词单元作为移动块。
- **Constrained constituent reordering**：在保护边界标记和连接词的前提下，对语义词组进行伪随机置换的数据增强方法。
- **Activation patching（残差流补丁）**：机制可解释性技术，将 clean run 的中间激活替换到 perturbed run 的对应位置，以定位因果影响组件。
- **Entity-Quantity Combinations**：描述特定项目数量的完整名词短语（如"तीन कप चिकन फीड"），是本文发现对推理恢复最关键的一类词。
- **DoRA（Weight-decomposed Low-Rank Adaptation）**：将预训练权重分解为幅值和方向独立组件的 PEFT 方法，比 LoRA 在推理任务上表现更优。
- **Compositional semantic reasoning**：模型通过系统性地组合各语言单元语义来推导整体意义，并对表层句法变化保持不变的推理能力。

---

## 可复现要素
| 要素 | 详情 |
|------|------|
| **数据集** | GSM8K-Hi（Kamath et al., 2025）、GSM8K-Malayalam（Sarvam AI, 2025b）、ARC-Challenge-Indic（Sarvam AI, 2025a）——均公开可获取 |
| **IndicReStruct 数据集** | 论文声称开源（脚注 2 标注），但未提供具体链接；重排种子=42 的 5 组数据可从论文方法复现 |
| **代码** | 论文未提供官方代码仓库链接 |
| **模型** | Gemma-2-9B-it、Gemma-2-27B-it、GPT-OSS-20B、Llama-3.1-8B-it、Param-2-17B-A2.4B、Qwen3-30B-A3B——均公开权重可获取 |
| **关键超参** | PEFT: r=16, α=32, dropout=0.05; 训练: max 20 epochs, patience=5, weight decay=0.01, gradient clip=0.3, cosine LR warmup=0.1, batch=8 (per-device 2, accum 4), seed=42 |
| **激活补丁工具** | TransformerLens（Nanda & Bloom, 2022） |

---
