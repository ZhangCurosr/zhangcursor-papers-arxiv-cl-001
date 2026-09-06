---
title: "From-Terminology-to-Diagrams-Visual-Instruction-Generation-f"
source: https://arxiv.org/pdf/2609.00948v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:18"
field: "多模态科学理解"
keywords: ["视觉语言模型", "科学图表理解", "指令微调", "数据集构建", "多模态QA", "LLaVA"]
innovations: ["术语驱动的六阶段视觉指令生成框架", "SciGram数据集：194K图表+1.4M指令支撑科学图表理解", "LLaVA-SciGram OV在TQA图表题建立新SOTA"]
benchmarks: ["TQA", "ScienceQA", "AI2D"]
---

# 论文速读：From-Terminology-to-Diagrams-Visual-Instruction-Generation-f

## 一句话总结
本文提出了一种基于科学课程术语驱动的视觉指令生成框架，构建了包含194K科学图表和1.4M多模态指令的SciGram数据集，并在TQA、SQA、AI2D等科学图表问答基准上实现了新SOTA或匹配现有最优模型。

## 研究问题与动机
- **科学图表理解仍是开放问题**：尽管VLM在自然图像上表现优异，但科学图表具有符号化、抽象性和结构多样性，现有模型难以有效理解。
- **训练数据稀缺**：现代VLM的训练数据中科学图表覆盖率极低，而TQA、AI2D、ScienceQA等基准本身提供的训练数据量不足以充分培养图表推理能力。
- **现有数据集不匹配目标任务**：MMMU、VQA Abstract Scenes、SciVerse等数据集包含图表类图像，但与科学图表理解基准的视觉特征存在差异。
- **通用指令数据缺乏领域特异性**：LLaVA OneVision、PixMo等通用数据集以覆盖广度优先，科学图表相关样本稀疏，难以有效支撑专业领域Fine-tuning。

## 核心贡献（创新点）
1. **术语驱动的六阶段数据集构建框架**：从中小学科学课程术语出发，经原子事实生成→图表检索→指令合成，形成端到端的数据生成流水线，与纯自由文本或随机网页数据方法形成本质区别。
2. **SciGram数据集**：构建194K+科学图表、1.4M条多模态指令的大规模数据集，包含Caption对齐（Align）、MCQA推理（VIT）、高质量微调（M³）三个子集，以"覆盖优先于精确"为设计原则，显著小于通用数据集但针对性更强。
3. **LLaVA-SciGram模型系列**：在LLaVA架构上微调的两个7B模型（从零训练和基于LLaVA OV微调），在TQA图表题上超越所有基线，建立新SOTA。
4. **系统化的数据集质量评估**：通过人类专家评估验证88%的caption与图表对齐、89%的MCQ视觉 grounding，揭示61%问题可能用先验知识回答等局限。
5. **模型无关的方法论推广性**：术语→事实→检索→生成的范式可迁移至工程图纸、医学插图等其他结构化视觉知识领域。

## 方法详解
**整体流程**：术语提取 → 原子事实生成 → 图表检索与过滤 → 指令数据合成（Caption + MCQ）→ 高质量子集整合。

**1) 术语提取（Terminology Extraction）**
- 基于Kembhavi et al. (2017)教科书，按主题提取名词短语候选集$\hat{T}_d$
- 使用weirdness index（Ahmad et al., 1999）筛选领域特征术语，阈值$t=2$，对比BNC通用语料库
- RoBERTa-base嵌入后进行聚类，以欧氏距离剔除偏离质心超过1个标准差的术语
- 最终获得4,820个语义一致的术语

**2) 原子事实生成（Atomic Fact Generation）**
- 对每个主题$d$，取其术语幂集$C_d = \mathcal{P}(T_d) \setminus \emptyset$的所有非空组合
- 用LLaMA3-8B-Instruct为每个组合$c \in C_d$生成最多50条中学水平的事实陈述
- 去重后获得5,508,218条独特原子事实

**3) 图表检索（Diagram Retrieval）**
- 以原子事实+"diagram"作为DuckDuckGo查询词，收集Top5图片结果
- 过滤策略：仅保留至少关联5个原子事实的图片；通过感知哈希去重；剔除无效文件
- 保留一定噪声以换取规模覆盖，最终获得255,657张唯一图片

**4) 指令数据生成（Instruction Generation）**
- **Caption生成**：用Qwen2-VL-7B对每个图表生成三段描述性caption（温度0.7），平均normalized Levenshtein相似度0.42，保证多样性；形成SciGram-Align（582,213条）
- **MCQ生成**：对每个图表生成4选1的多选题（中学水平，答案平衡分布），去重5.7%；形成SciGram-VIT（737,887条）
- **M³子集整合**：从LLaVA OV训练混合中取SQA、AI2D、TQA及文本QA（ARC-Easy/Challenge、OpenBookQA），共47,506条，答案选项打乱以减少过拟合

**训练阶段**（LLaVA-SciGram OV）：
- 阶段1（Align）：冻结ViT和LLM，仅训练mm projector，lr=1e-3，1 epoch
- 阶段2（VIT）：LoRA微调（r=128, alpha=256），lr=1e-5，1 epoch
- 阶段3（M³）：合并后继续LoRA微调，lr=1e-5，多epoch，batch=1，grad_acc=16

## 实验与结果
**数据集与基准**：
- SciGram：194,071张图表，1.4M条指令（Align/VIT/M³）
- 评测基准：TQA（含Text-only/True-False/Diagram）、ScienceQA（9个子类）、AI2D（Opaque/Transparent标签）
- 已排除基准测试图片防止数据污染

**主要结果**：

| 模型 | TQA Overall | SQA Overall | AI2D Overall |
|------|-------------|-------------|---------------|
| LLaVA OV 7B | 82.70 | 85.83 | 84.70 |
| **LLaVA-SciGram OV 7B** | **85.57** | **96.04** | **87.94** |
| LLaVA-SciGram 7B (from scratch) | 83.66 | 95.33 | 85.07 |

- **TQA**：LLaVA-SciGram OV在Diagram MC上达80.12%，超越所有基线，包括GPT4o（77.32%）；整体超越LLaVA OV +2.87pp
- **SQA**：在Visual Support问题上达97.62%（OV），超越上一SOTA约0.54pp；整体96.04%，接近T-SciQ（96.18%）
- **AI2D**：Opaque标签83.45%，超越MOLMo 7B-D（82.40%）1.05pp

**Ablation**：三阶段全部使用SciGram子集效果最佳（Table 6），单独移除任一阶段均导致性能下降；加入文本QA（OpenBookQA、ARC）带来小幅提升

**细粒度分析**：
- 在Structure、Teleology/Purpose、Algebraic、Spatial/Kinematic、Visual Labeling等深度图表理解类型上获>5pp提升
- 对需视觉支持的问题（94题）提升近10pp，对仅需语言先验的问题持平，表明增益来自真正的图表理解而非语言捷径

## 相关工作脉络
1. **早期科学图表理解**：Kembhavi et al. (2017) TQA开创性工作，结合MRC、VQA和Diagram Parser；本文在此基础上解决VLM时代的数据稀缺问题
2. **ISAAQ**（Gómez-Pérez & Ortega, 2020）：同团队前作，用跨模态注意力处理图表QA；本文扩展到大规模VLM训练数据构建
3. **通用VLM**（LLaVA OneVision、MOLMo、Pixtral、Qwen2-VL）：依赖通用指令数据，科学图表覆盖稀疏；本文针对性补足此短板
4. **领域专用VLM**（LLaVA-Med、LLaVA-Chef、LLaVA-Ultra）：证明垂直领域微调的有效性；本文将此思路系统化推广到科学图表
5. **数据集对比**（MMMU、SciVerse、VQA Abstract Scenes）：虽含图表图像，但与中小学科学图表基准（AI2D/TQA/SQA）在视觉风格和语义粒度上存在差异
6. **对比学习方法**（CLIP、SIGLIP）：构成VLM视觉编码基础；本文在此基础上构建针对性的指令微调数据

## 局限性与未来方向
- **噪声可控但不可消除**：人类评估显示24%检索图片非真正图表；61% MCQ可脱离图表用先验知识回答
- **标注一致性缺陷**：16% MCQ存在标签不一致（正确选项被标为错误）
- **过程/因果推理仍有不足**：Figure 3显示"Processes & Causal"和"Causal/Explanation"类型未获显著改善
- **未来方向**：①更精确的图表/非图表分类器；②自动化一致性验证模型；③更强Teacher模型提升数据质量；④扩展至工程图纸、医学插图等领域

## 研究启发与可借鉴点
1. **术语→事实→检索→指令的级联生成范式**：从结构化课程知识出发，经LLM生成中间表示再映射到视觉数据，可复用于其他领域（法律、地理、历史图谱等）的数据构建
2. **"覆盖优先于精确"的大规模数据策略**：接受一定噪声换取规模，配合多阶段Fine-tuning仍可显著提升性能；对资源受限团队有启发意义
3. **三阶段分离训练设计**（Align→VIT→M³）：对齐阶段冻结参数、推理阶段LoRA微调、精调阶段整合高质量数据，层次分明且可复现
4. **答案选项打乱平衡**：针对AI2D/ARC等原数据存在选项分布不平衡问题，通过shuffle降低overfitting风险，是实用的数据工程技巧
5. **可视化 grounding 验证方法**：通过"需视觉支持"vs"仅需语言先验"的子集对比分析，可验证模型是否真正利用了图像信息，值得纳入评测规范

## 关键术语表
**SciGram**：本文构建的科学图表多模态数据集，含194K图表和1.4M指令，分为Align/VIT/M³三个子集
**weirdness index**：通过对比目标语料与通用语料（BNC）的术语频率，量化术语的领域特异性，阈值过滤通用词汇
**atomic fact**：由术语组合生成的简洁科学性陈述，作为图表检索的查询锚点
**LLaVA-SciGram OV**：基于LLaVA OneVision 7B架构，经SciGram三阶段微调后的视觉语言模型
**Visual Instruction Tuning (VIT)**：SciGram子集之一，包含737K条图表驱动的MCQA指令，用于模型推理能力训练
**TQA (Textbook Question Answering)**：中小学科学课本问答基准，含文本/图表两种形式的问题
**ScienceQA (SQA)**：覆盖更广科学领域的多模态QA基准，含自然/社会科学、文本/图像/无支持等多种类型
**AI2D**：小学科学图表理解基准，特点是图表标签被遮蔽（opaque）或可见（transparent）

## 可复现要素
- **数据集**：SciGram发布在HuggingFace（https://huggingface.co/collections/expertailab/scigram），许可证CC BY 4.0；图片以URL形式提供，未托管原图
- **代码**：开源（https://github.com/expertailab/scigram）
- **模型权重**：LLaVA-SciGram 7B和LLaVA-SciGram OV 7B均已发布
- **关键超参**：Alignment lr=1e-3（1 epoch, projector only）；VIT lr=1e-5, LoRA r=128 alpha=256（1 epoch）；M³ lr=1e-5, LoRA alpha=256；batch=1, grad_acc=16；max_length=32768
- **限制因素**：图片URL可能随时间失效（link rot）；评估指标部分依赖外部API；需要至少2×A100 GPU完成训练（~450 GPU-hours/模型）
