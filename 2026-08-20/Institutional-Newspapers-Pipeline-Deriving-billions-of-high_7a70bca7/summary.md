---
title: "Institutional-Newspapers-Pipeline-Deriving-billions-of-high"
source: https://arxiv.org/pdf/2608.18972v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:41:18"
field: "历史文献数字人文与多模态信息抽取"
keywords: ["历史报纸数字化", "文档分割", "多模态OCR", "VLM", "低开销流水线", "结构化数据集", "阅读顺序检测", "实体识别"]
innovations: ["分割与分类解耦的双分支crop类型融合分类器", "Tesseract+轻量VLM双路OCR互补策略", "面向工作站硬件的15步模块化低开销流水线设计"]
benchmarks: ["mAP@50=0.955（分割）", "Kendall'sτ=0.959（阅读顺序）", "Crop类型分类准确率≈91-92%"]
---

# 论文速读：Institutional Newspapers Pipeline: Deriving billions of high quality tokens from historical newspapers

## 一句话总结
本文提出了一套面向历史报纸扫描件的多模态处理流水线（Institutional Newspapers Pipeline），以可解释、可定制、低计算开销的方式，从波士顿公共图书馆147万余份公版报纸扫描件中，自动化提取出8310万个高分辨率图像块（crop），生成总计163亿（o200k_base）token的高质量OCR文本及丰富的结构化标注数据。

## 研究问题与动机
- **历史报纸的计算访问门槛高**：历史报纸版式密集且不规则，包含多栏排版、装饰性标题和表格，通用工具难以有效处理，导致大量馆藏内容难以被计算方法挖掘。
- **现有OCR方案在批量场景下成本与质量难以兼顾**：直接使用前沿VLM对整页扫描进行OCR成本极高（估算需25–80万美元），且复杂版式下VLM容易出现重复循环与幻觉；传统Tesseract对栏结构和装饰文本的处理也不理想。
- **大模型高质量训练数据不足**：LLM预训练极度依赖数据规模与质量，历史报纸作为记录公共生活的高价值文献，其内容尚未被充分解构为结构化数据进入AI的"数字饮食"。
- **现有工作多为联合检测+分类，类不平衡问题严重**：类似 Newspaper Navigator 等方案将检测与分类联合训练，但在真实报纸数据中文本类（Content）与广告类（Advertisement）占绝对主导，导致分类精度下降。

## 核心贡献（创新点）
1. **分割与分类解耦的模块化流水线设计**：将版面分割（crop detection）与后续分类/标注步骤严格分离，避免联合训练带来的类别不平衡问题，同时保证每一步均可独立评估与替换，适配不同机构馆藏。
- 与已有工作（如 Newspaper Navigator 的端到端检测+分类）的本质区别在于：先输出类型无关（type-agnostic）的矩形边界框，再分别用图像和文本两个分类器做判断，最后融合。

2. **双路OCR策略（Tesseract + 轻量VLM）**：在同一流水线中同时运行 Tesseract 5（提供可兼容现有图书馆发现系统的词级bbox）与 dots.mocr（3B参数多模态VLM，支持表格解析），两者互补——Tesseract 幻觉率低，VLM 能恢复破损文本并产出结构化输出。
- 本质区别：不同于以往只用单一OCR引擎的工作，本文论证了双路交叉的价值，尤其在字符数（57.4亿 vs 50.1亿）和tokenizability上VLM更优，而Tesseract在独特词召回上表现更好。

3. **图像+文本双分支_crop类型分类器及融合决策机制_**：分别训练基于 YOLO26m-cls（11.6M参数）的图像分类器和基于 Model2Vec fine-tuned potion-base-32M 的文本分类器，通过优先级规则融合两者信号（图像分类器在"Photograph/Illustration"和"Empty"上优先）。
- 与以往单一模态分类器的本质区别：两种信号在难以判别的区域（如插图、卡通）显著互补，两分类器仅90.55%一致性，但融合后能获得更鲁棒的最终标签。

4. **面向工作站级硬件的低开销推理架构**：整套15步流水线可在单机8×L40S工作站上运行，总成本约2.5万美元（或1650小时），远低于使用商业VLM的25–80万美元预算。
- 与同类"重模型、高算力"方案的本质区别：全文强调 frugal computing，所有核心模型均控制在数千万至数亿参数以内，预处理与推理的I/O瓶颈也通过并行化+磁盘缓存解决。

5. **开源数据集与生产级工具链**：开放包含163亿token、8310万crop的结构化数据集（Parquet格式，字段覆盖bbox、OCR文本/标注、语言、NER实体、主题分类、embedding等15类），以及配套模型和 Agent Skill 文件。
- 与已有开放语料（如 The Pile、Dolma）的定位差异：本文聚焦于历史报纸这一特定领域的细粒度结构化数据（每页多个独立crop、附带阅读顺序、多模态嵌入），而非单纯的纯文本集合。

## 方法详解
流水线按顺序执行15个步骤，可分为6个逻辑层：

**① 分割（Segmentation）**
- 模型：YOLO26x（55.7M参数），输入分辨率960px，训练时划分15%验证、15%测试。
- 标注集：1020份随机抽样扫描，47,485个边界框（9,028手工标注，38,457由中间模型预标注后人工校验）。
- 推理过滤：置信度阈值 0.6，NMS 的 IoU 阈值 0.15（较低以保留重叠框）。
- 效果：mAP@50 = 0.955，mAP@50-95 = 0.901，F1 = 0.918。

**② Crop级OCR**
- Tesseract 5 + tessdata\_best：单crop缩放至最大1.5MP，使用 tesserocr 多线程推理，输出词级 bbox 以便兼容 AltoXML。
- dots.mocr（3B参数 VLM）：通过 vLLM 在8×L40S上部署，使用双缓冲机制（准备下一批时同时推理当前批），crop图像缩放到0.25–1MP之间，启用 aggressive batching 与 KV cache。

**③ 语言检测与文本分析**
- 语言检测：Lingua（短文本鲁棒），置信度低于0.5或字数<30时使用 issue-level 元数据替代（共替换3.04%的crop）。
- 文本分析指标：字符数、词/句数（总及唯一）、tokenizability 分数（词数与 token 数之比，基准值~1.25）、Markdown 标记与表格检测。
- Flatten 处理：去除HTML标签、连字符断行合并、零宽空格去除、换行替换为空格。

**④ Crop类型分类（双分支融合）**
- 图像分类器：YOLO26m-cls，11.6M参数，7类（含 Empty），输入768px，训练集188,477条。
- 文本分类器：potion-base-32M（Model2Vec提取静态embedding）微调，6类（无 Empty），训练集185,900条。
- 自动标注：使用 Qwen3-VL-30B-A3B-Thinking-FP8 做 auto-annotation，人工抽查600条准确率达91%（严格）/94.5%（宽松）。
- 融合规则：默认取置信度最高者；但若图像分类器输出"Photograph/Illustration"或"Empty"，或文本判定为 Empty，则以图像分类器为准。

**⑤ 扫描级阅读顺序检测**
- 基于 HDBSCAN 的非神经网络方案：
  1. 根据相对栏宽将 crop 分为 narrow/wide；
  2. 对 narrow Content crop 按 x-center 聚类为列；
  3. 非内容 narrow crop 分配至最近列；wide crop 分配至上方内容最多的列；
  4. 列间左→右、列内上→下排序；
  5. 后处理：按左边缘 bucketing 微调顺序；视觉元素按顶部边而非中心排序，确保在正文前。
- 评估：723份人工标注扫描，Kendall's τ = 0.959（micro），位置准确率 80.8%（micro）。

**⑥ 丰富标注层（NER / 主题 / Embedding / Thesauri匹配）**
- NER：Flair base ner-fast 模型，置信度阈值0.85，仅保留 PER/LOC/ORG，去重后保留最高置信度。
- 主题分类：ModernBERT-large-zeroshot-v2.0 做 zero-shot 分类，保留完整排序和置信度（实验性输出）。
- Embedding：图像用 DINOv2-small（384维，22M参数）；文本用 potion-multilingual-128M（256维，静态模型）。
- Chronicling America Thesauri 匹配：对种族、族裔、移民、公民权关键词列表做朴素匹配（实验性）。

## 实验与结果
**数据集规模**
- 扫描总数：1,473,635 份（1795–1930年，美国公版报纸，主要来自波士顿公共图书馆）。
- Crop 总数：83,147,041 个，平均每份扫描约56.4个 crop。
- OCR Token 总量：dots.mocr 输出 163亿 o200k\_base tokens；Tesseract 输出 147亿 tokens。
- 语言覆盖：73种语言代码，其中英语占97.61%，其余以意第绪语（1.18%）、德语（0.56%）、瑞典语（0.46%）、法语（0.09%）为主。

**关键性能数字**
| 步骤 | 指标 | 数值 |
|---|---|---|
| 分割 | mAP@50 | 0.955 |
| 分割 | F1 | 0.918 |
| Crop类型分类（图像） | Overall accuracy | 0.913（Top-1）|
| Crop类型分类（文本） | Overall accuracy | 0.92 |
| 双分类器一致率 | — | 90.55% |
| 阅读顺序（Kendall's τ micro） | — | 0.959 |
| 阅读顺序位置准确率（micro） | — | 80.8% |
| NER | LOC 提及数 | 1.556亿 |
| NER | PER 提及数 | 1.422亿 |
| NER | ORG 提及数 | 4910万 |
| Embedding总存储 | 图像+文本 | ~213 GB |

**成本与效率**
- 单批次（200期）平均耗时：86.69分钟，其中 OCR 占约一半（Tesseract 16.48min + dots.mocr 28.62min）。
- 租用同等硬件总成本估算：约2.5万美元（1650小时 × $15/小时）。

## 相关工作脉络
1. **Newspaper Navigator（Lee et al., 2020）**：使用 CNN 对 Chronicling America 的1600万页报纸进行版面元素检测与分类，本文与其同源但定位不同——本文追求"原子化"（crop-level）的细粒度结构化数据，而非页面级 headlines/visuals 的粗粒度标注。
2. **American Stories（Dell et al., 2023）**：同样基于历史报纸的大规模结构化文本数据集，但本文强调多模态（图像+文本）与细粒度 crop 级别的丰富标注（NER、主题、embedding），覆盖颗粒度更细。
3. **LayoutReader（Wang et al., 2021）**：基于 Transformer 的阅读顺序检测模型，本文认为其在大规模部署时计算开销过大，且操作粒度为 span 而非 crop，故采用 HDBSCAN 规则方案作为替代。
4. **The Pile（Gao et al., 2020）与 Dolma（Soldaini et al., 2024）**：开源预训练语料，本文定位为其补充——提供来自历史报纸的高结构化、多模态子集，而非纯文本集合。
5. **Talkie-LM（Levine et al., 2026）**：在1931年前数据上训练的LLM，已部分使用本团队之前的 Institutional Books 数据集，本文是其报纸领域的对应延伸，共同服务于"让LLM更好理解历史材料"的目标。
6. **olmOCR（Poznanski et al., 2025）与 CHURRO（Semnani et al., 2025）**：前沿VLM用于PDF/历史文档OCR的工作，本文借鉴了其质量优势但通过 crop 粒度和本地部署规避了成本与幻觉问题。

## 局限性与未来方向
- **模型泛化性受限**：当前模型在波士顿公共图书馆馆藏上训练与评估，面对版式差异较大的其他馆藏（如非栏式布局、非英文报纸）可能表现下降，需在多个机构数据上持续微调。
- **阅读顺序检测的结构性假设**：HDBSCAN 方案依赖"栏式布局"假设，对复杂或非线性排版的报纸不适用，未来考虑引入 Transformer-based 方案。
- **VLM OCR仍是计算瓶颈**：dots.mocr 虽已优化，但 VLM 推理在规模上仍是最耗时的步骤；未来方向是训练针对历史报纸 crop OCR 的专用更小模型，以减少 token 循环等问题。
- **NER与主题分类为实验性输出**：Flair ner-fast 未针对历史文本和OCR噪声做专门微调，且存在有害语言的识别风险；zero-shot 主题分类置信度跨度大，不宜直接用于推断文本性质。
- **多语言覆盖有限**：73种语言中绝大部分为英语，小语种样本极少，Lingua 对部分语言（如意第绪语）依赖 issue-level 元数据回退。

## 研究启发与可借鉴点
1. **"分割-分类解耦"设计范式**：将版面分割与类型分类分离，既缓解了类不平衡问题，又允许后续用更轻量的分类器迭代更新，不重复训练检测模型——该思路可迁移至任何复杂版式的文档数字化场景（如期刊、档案）。
2. **双模态分类融合决策**：图像+文本双分支分类器在不同类别上各有优势（视觉类图像强、文字类文本强），通过规则融合的决策机制简单有效，避免了训练单一多模态大模型的复杂性和成本，值得在其他多模态分类任务中借鉴。
3. **Crop 粒度的 OCR 策略**：将整页划分为 crop 再分别 OCR，既降低了 VLM 的 token 循环风险，又保留了词级 bbox 和结构化输出（Markdown/表格），这套"先分割再逐块处理"的范式对历史文献数字化具有通用价值。
4. **工作站级 frugal pipeline 的工程实践**：并行预处理、vLLM 双缓冲 batching、磁盘缓存 overlap I/O 与计算等工程优化策略，展示了如何在有限硬件上以低成本完成十亿级 token 的批处理任务。
5. **与图书馆系统无缝对接**：Tesseract 输出的词级 bbox 可直接转为 AltoXML 格式，兼容现有图书馆发现系统，同时输出 VLM Markdown 版本，兼顾学术研究与工程落地双重需求。

## 关键术语表
**Crop**：从报纸扫描中分割出的单个矩形内容块（包含连续的文本或视觉元素），是流水线处理的最小原子单元。
**o200k\_base tokenizer**：OpenAI 的 BPE 分词器，本文用它来量化 OCR 输出的 token 数量（与 LLM 预训练尺度对齐）。
**Eigen-CAM**：基于主成分分析的类激活可视化方法，本文用于解释分割模型关注版面的哪些结构特征（如栏线和标题）。
**Flair ner-fast**：Flair 框架中的轻量级 NER 模型，本文用作实验性实体识别模块，未针对历史文本做专门微调。
**HDBSCAN**：层次密度聚类算法，本文用于将 narrow crop 按 x-center 聚类为列，进而推断阅读顺序。
**Lingua**：Python 短文本语言检测库，本文用于 crop 级语言识别，低置信度时回退到 issue-level 元数据。
**Model2Vec**：将上下文 embedding 模型（如 sentence-transformer）转化为轻量级静态 embedding 的工具，本文用于构建高效的 crop 文本分类器和文本 embedding。
**Zero-shot classification**：无需微调即可对新类别进行分类的方法，本文用 ModernBERT 对12类主题做 zero-shot 标注，置信度作为辅助信号而非确定性标签。

## 可复现要素
- **数据集**：来自波士顿公共图书馆的1,473,635份公版报纸扫描件（1795–1930年）；数据集已开源。
- **代码**：GitHub https://github.com/institutional/institutional-newspapers-pipeline
- **模型**：在 Hugging Face 开源（https://huggingface.co/collections/institutional/institutional-newspapers）
- **关键超参**：
  - 分割模型输入分辨率：960px；置信度阈值：0.6；NMS IoU：0.15
  - Tesseract 单 crop 最大尺寸：1.5MP
  - dots.mocr 单 crop 尺寸范围：0.25–1MP；推理设备：8×NVIDIA L40S，vLLM
  - 图像分类器输入：768px；训练集：188,477条（70/15/15切分）
  - 文本分类器训练集：185,900条
  - NER 置信度阈值：0.85；语言检测回退阈值：置信度<0.5或字数<30
- **计算环境**：8×NVIDIA L40S / 256 CPU核 / 768GB RAM
