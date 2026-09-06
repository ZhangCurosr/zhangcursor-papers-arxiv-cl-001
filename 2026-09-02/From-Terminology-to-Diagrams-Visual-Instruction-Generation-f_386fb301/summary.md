---
title: "From-Terminology-to-Diagrams-Visual-Instruction-Generation-f"
source: https://arxiv.org/pdf/2609.00948v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:21"
field: "科学图表多模态理解"
keywords: ["科学图表理解", "视觉指令生成", "多模态大模型", "数据合成", "术语驱动"]
innovations: ["术语驱动的科学图表指令数据生成框架（从课程术语到原子事实到图表检索到指令合成）", "构建194K图表/1.4M指令的SciGram数据集并开源", "LLaVA-SciGram OV在TQA Diagram上建立新SOTA（80.12%）"]
benchmarks: ["TQA", "ScienceQA", "AI2D"]
---

# 论文速读：From-Terminology-to-Diagrams-Visual-Instruction-Generation-f

## 一句话总结
本文提出了一种从科学课程术语出发生成大规模图表指令数据的框架（SciGram），构建了包含19.4万张科学图表和140万条多模态指令的数据集，基于此微调的LLaVA模型在TQA、ScienceQA和AI2D等科学图表理解基准上达到SOTA或与之相当的性能。

## 研究问题与动机
- **科学图表理解仍是开放难题**：尽管VLM在自然图像VQA上表现优异，但科学图表（具有符号性、抽象性、结构多样性，传达概念与关系而非真实场景）的理解能力仍然薄弱。
- **现有训练数据中科学图表严重稀缺**：主流VLM（如LLaVA OneVision、MOLMo）依赖通用指令数据集，科学图表覆盖稀疏；现有基准（AI2D、TQA、SQA）本身提供的训练数据不足。
- **高质量人工标注成本高昂**：领域专用VLM（如LLaVA-Med）验证了领域数据微调的价值，但人工构建规模化的科学图表指令数据代价巨大。
- **"噪声数据也能有效"的假设待验证**：大规模Web抓取数据天然含噪声，论文验证了即使存在约24%的非图表图像，结构化术语驱动的生成管线仍可提供有效的多模态监督信号。

## 核心贡献（创新点）
1. **术语驱动的科学图表指令数据生成框架**：以中学科学课程术语为语义骨架，逐层生成原子事实→检索图表→合成指令；与基于自由文本Web数据的方法本质不同，后者缺乏对科学概念的结构性覆盖。
2. **SciGram大规模数据集（194K图表、1.4M指令）**：覆盖生物/地球/物理三大学科，包含Align、VIT、M³三个子集，强调"覆盖优先于精度"；与现有数据集（如MMMUs、SciVerse）不同，SciGram专门针对AI2D/TQA/SQA这类具象科学概念图表构建。
3. **LLaVA-SciGram模型系列**：在LLaVA架构上实现从无到有训练（7B）和基于LLaVA OV微调两种范式；LLaVA-SciGram OV 7B在TQA Diagram上超越所有基线，建立新SOTA。
4. **证明提升来源于视觉推理而非文本先验**：消融分析显示SciGram微调使需要视觉支持的问题准确率提升近10个百分点，而纯文本问题增益有限，说明图表理解能力确有实质增强。

## 方法详解
**整体管线（六阶段，图1）：** 术语提取 → 原子事实生成 → 图表检索 → SciGram子集构建（Align、VIT、M³）

**3.1 术语提取（4,820个术语）：**
- **分词与名词短语识别**：对TQA教材按主题分段，提取名词短语候选集合 $\hat{T_d}$。
- **显著术语筛选**：计算**怪异度指数（weirdness index）** $w(t) = \frac{P(t|\text{TQA})}{P(t|\text{BNC})}$，阈值 $t=2$，过滤通用词汇。
- **嵌入聚类过滤**：使用RoBERTa-base对每个术语在所有共现句中获取上下文表示后取平均 $e_i$，计算与类别中心 $c_d$ 的欧氏距离 $\delta_i = |\mathbf{e}_i - \mathbf{c}_d|_2$，剔除超过1个标准差的术语。

**3.2 原子事实生成：**
- 对每个主题的所有非空术语组合 $C_d = \mathcal{P}(T_d) \setminus \emptyset$，使用**LLaMA3-8B-Instruct**生成包含该组合的中学校顿原子事实（每条≤12 tokens，每组合最多50条）。
- 去重后得到**5,508,218条**唯一原子事实。

**3.3 图表检索：**
- 每条原子事实追加后缀"diagram"，通过**DuckDuckGo**搜索，收集Top 5图片URL及元数据（双云实例，21天）。
- 轻量过滤：仅保留被≥5条原子事实关联的图片；感知哈希去重；剔除无效文件。最终获**255,657张**唯一图片。

**3.4 指令生成（三个子集）：**
- **SciGram-Align**（582,213条）：用**Qwen2-VL-7B**对每张图生成3次描述性caption（avg Levenshtein相似度=0.4196，多样性良好），格式化为图像-文本对齐指令。
- **SciGram-VIT**（737,887条）：对每张图生成多选项问答题（MCQs），确保基于视觉内容、初中难度、选项四选一且正确率均匀分布；去重率5.7%。
- **SciGram-M³**（47,506条）：整合已有领域QA数据集（TQA 14,050、SQA 12,726、OpenBookQA 4,957、ARC-Easy/Challenge 3,370），答案选项打乱以减少过拟合偏差。

**训练管线：** 分三阶段——① 投影矩阵对齐（SciGram-Align，lr=1e-3，1 epoch）→ ② LoRA指令微调（SciGram-VIT，lr=1e-5，1 epoch）→ ③ LoRA进一步微调（SciGram-M³，lr=1e-5，3 epochs）。

## 实验与结果
**基准：** TQA（文本/判断/图表三部分）、ScienceQA（9种子类型）、AI2D（不透明/透明标签两变体）。所有测试图表均从SciGram中排除以防数据泄露。

**Table 1（与LLaVA OV基线对比）：**
- LLaVA-SciGram OV在TQA Diagram上+3.04（77.08→80.12），SQA Visual Support +10.31（87.31→97.62），AI2D Opaque +3.95（79.50→83.45）。
- SQA整体：+10.21（85.83→96.04）；TQA整体：+2.87。

**Table 2（TQA vs 多模型）：**
- LLaVA-SciGram OV 7B在Diagram MC上**80.12%**，超越所有基线（包括GPT4o的77.32%），建立新SOTA。
- LLaVA-SciGram 7B从scratch训练整体83.66%，略低于OV微调版（85.57%）。

**Table 3（SQA）：**
- LLaVA-SciGram OV在IMG子类型**97.62%**，超越次优T-SciQ的94.70%；整体96.04%，与T-SciQ的96.18%几乎持平。

**Table 4（AI2D）：**
- LLaVA-SciGram OV在Opaque Labels上**83.45%**，超越前SOTA MOLMo 7B-D的82.40%（+1.05%）。

**消融（Table 5/6）：** 三阶段子集均有贡献；完整管线（Align+VIT+M³）取得最佳效果；加入文本-only数据集带来小幅持续增益。

**图表理解分析（Figure 3/4）：** SciGram在Structure、Teleology/Purpose、Algebraic、Spatial/Kinematic、Visual Labeling等深度理解类型上提升超5个百分点；在需要视觉支持的问题上比基线高近10分，而在可凭文本先验作答的问题上两者相当。

## 相关工作脉络
- **早期图表QA方法**（Kembhavi et al., 2017; ISAAQ）：探索机器阅读理解与图结构建模，但未利用预训练VLM，且专注于自然图像多模态预训练（VL-BERT、LXMERT）并未覆盖科学图表领域。
- **CLIP/SIGLIP类对比学习**：奠定了现代VLM（LLaVA、MOLMo）基础，但依赖通用指令数据（如LLaVA OV的7.8M指令），科学图表覆盖稀疏。
- **领域VLM微调范式**：LLaVA-Med、LLaVA-Chef等证明了垂直领域微调的有效性；SciGram沿此路线但聚焦K-12科学图表这一此前未被充分覆盖的子领域。
- **科学图表基准数据集**（AI2D、TQA、SQA）：提供了评测标准但训练数据量不足；SciGram首次以自动化方式从这些基准的"知识空间"反推大规模训练数据。
- **LLaVA OV与PixMo**：代表通用视觉指令调优的最前沿；本文在相同架构上仅用1.4M vs 7.8M指令即实现更强科学图表能力，体现了"质量/针对性>规模"的思路。

## 局限性与未来方向
- **约24%的检索图像非真正图表**（含自然图、图表等噪声），依赖后续 caption/MCQ 的容错能力而非预处理消除。
- **16%的MCQ存在标签不一致**（正确/错误选项标注颠倒），未来需引入自动一致性验证模型。
- **61%的MCQ可能仅凭先验知识即可作答**，图表依赖度有待提高，需更强的"必须看图"约束。
- **Web链接存在失效风险（link rot）**，模型仅发布URL而非图片本体，长期可复现性受限。
- **过程与因果类问题（Processes & Causal）仍有提升空间**，Figure 3显示该类型未见显著增益。
- **硬件依赖较高**：训练需双A100 GPU约450 GPU-hours，限制了低资源场景的复现。

## 研究启发与可借鉴点
1. **术语驱动的数据生成范式可迁移**：从结构化术语出发（而非自由文本）引导事实生成→视觉检索的链路，可扩展至工程图纸、医学插图等其他结构化视觉知识领域。
2. **"覆盖优先于精度"的大规模合成策略**：接受24%噪声换取19.4K规模，在科学图表等人工标注稀缺的领域是一种务实可行的路线。
3. **三阶段分层训练设计（Align→VIT→M³）值得借鉴**：每阶段对应不同能力（视觉对齐→多模态推理→领域精化），且M³中答案选项打乱以减小过拟合偏差，实验设计严谨。
4. **怪异度指数（weirdness index）+ 欧氏距离聚类**的术语筛选方法，为领域词汇库构建提供了可复用的技术方案。
5. **视觉 grounding 验证思路**：通过"需视觉支持"vs"可文本先验作答"两类问题的差异化增益来论证模型真正学到了图表理解能力，而非跟随文本线索——这一评估设计可直接复用于其他多模态工作。

## 关键术语表
**Scientific Diagram（科学图表）**：用于传达科学概念、关系或过程的符号化、结构化图像，与自然场景图像有本质区别。
**Weirdness Index（怪异度指数）**：衡量某术语在目标语料中相对通用语料出现频率的指标，用于筛选领域特有词汇。
**SciGram**：本文构建的多模态数据集，包含194K科学图表和1.4M指令，分为Align、VIT、M³三个子集。
**Atomic Fact（原子事实）**：包含若干术语组合的最小粒度科学陈述（≤12 tokens），作为图表检索的查询锚点。
**Visual Instruction Tuning（VIT）**：通过多模态问答指令微调视觉语言模型的过程，对应SciGram的第二个训练阶段。
**TQA / SQA / AI2D**：三个主流科学图表理解基准，分别侧重课本问答、多模态科学QA和图表多项选择。
**LLaVA-SciGram OV**：基于LLaVA OneVision 7B使用SciGram微调的模型，在多个科学图表基准上建立新SOTA。
**Gwet's AC1**：一种对类别不平衡鲁棒的评分者一致性度量，本文用于评估人工标注质量。

## 可复现要素
- **数据集**：SciGram已公开（Hugging Face: https://huggingface.co/collections/expertailab/scigram），代码已开源（https://github.com/expertailab/scigram）；图片以URL形式提供（非直接托管），存在link rot风险。
- **模型权重**：LLaVA-SciGram 7B与LLaVA-SciGram OV 7B均已开源。
- **关键超参**：对齐阶段lr=1e-3、1 epoch；VIT阶段LoRA r=128/alpha=256、lr=1e-5、1 epoch；M³阶段LoRA alpha=256、lr=1e-5、3 epochs；图像分辨率策略（anyres）包含[(384,768),(768,384),(768,768),(1152,384),(384,1152)]；训练硬件为2×NVIDIA A100。
- **评估设置**：答案选项打乱；图片测试集与训练集无交集。
