---
title: "Industrial-Instruction-An-End-to-End-Framework-for-Building"
source: https://arxiv.org/pdf/2608.22817v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:09:16"
field: "工业领域语言模型与数据集构建"
keywords: ["Industrial Instruction", "Instruction Tuning", "RAG", "Synthetic Data Generation", "Industrial QA", "Small Language Models", "Catastrophic Forgetting"]
innovations: ["提出五场景检索-回答关系建模的工业QA数据集构建范式", "首次公开基于真实工业PDF的双版本指令微调数据集并开源全链路管道", "在同构流水线下系统比较开源与闭源模型作为数据生成器的质量-成本-遗忘权衡"]
benchmarks: ["Panasonic Industrial Dataset", "FailureSensorIQ", "MMLU"]
---

# 论文速读：Industrial-Instruction-An-End-to-End-Framework-for-Building

## 一句话总结
论文提出 **Industrial-Instruction** 端到端框架，从工业技术报告（以松下906份公开PDF为案例）自动化构建多轮选择题指令微调数据集与基准；同时发布两套并行数据（分别由开源 Qwen3-30B-A3B-Instruct 与闭源 Claude-Opus-4.6 生成），系统比较 open-weight 与 frontier-model 作为数据生成器的质量-成本权衡，并验证其对 <10B 小型模型的有效微调收益。

## 研究问题与动机
- **工业文档的高价值与难处理并存**：技术报告承载维护、故障排查、产品工程等专家知识，但结构高度异构（密集 prose + 规格表格 + 图示），传统检索/问答管线难以有效索引与推理。
- **小模型工业知识缺口显著**：通用 LLM 在工业基准（如 FailureSensorIQ）平均仅 ~53.5% 准确率，LoRA 微调后的开源小模型（如 HotpotQA 微调后）在工业基准仅 ~29%，且小型模型（<10B）因部署成本与可落地性被严重低估。
- **缺乏公开可用的高质量工业指令微调与基准数据集**：现有研究多聚焦 GPT-4/ChatGPT 等大模型，缺少面向小型开源模型、含真实表格与散点技术知识的公开数据集。
- **标准评测指标不适配工业问答**：Exact-Match/Accuracy 对集合型答案顺序敏感，工业场景需要更贴合逻辑正确性的集合相似度度量。

## 核心贡献（创新点）
- **双版本 Panasonic 工业 QA 数据集**：同一端到端管道分别用 Qwen3-30B-A3B-Instruct 与 Claude-Opus-4.6 生成，均含 ~13.6k/25.3k 筛选后样本与 1,000 题 held-out benchmark 划分；**本质区别**在于首次公开以真实工业 PDF 为源、显式建模五种检索–回答关系的开源数据集，而非仅评估大模型或依赖合成/二手语料。
- **端到端可复现构建流水线**：集成 Dots.OCR 布局感知提取 → EmbeddingGemma+FAISS 语义索引 → 五场景 RAG 生成 → 规则过滤；**本质区别**是将前人多场景 RAG 范式迁移到异构工业 PDF（表格+文本+散点知识）并全链路开源。
- **Open-weight vs. Frontier-model 数据生成对照实验**：在同构 pipeline 下对比两套数据集的原始质量、下游微调增益、成本与通用知识保留（MMLU）；**本质区别**在于直接量化“生成器质量 × 成本”的 trade-off，为资源受限工业落地提供实证指南。

## 方法详解
- **PDF 信息提取**：采用视觉语言模型 Dots.OCR，单页输入生成（1）含页眉页脚的 Markdown、（2）去页眉页脚 Markdown、（3）带区域标注的 JPG、（4）结构化 JSON（region 类型+文本）；优于 MinerU/Mathpix 在 OmniDocBench/XDocParse 的 OveralEdit 指标。
- **知识库构建**：仅保留文本与 Markdown 化表格，剔除图片（评估纯文本模型能力）；使用 EmbeddingGemma（300M 参数，768→128 可降维）+ FAISS 构建语义索引；去除空白/低内容/重复页，共删 370 页，剩余 906 文档 7,525 页。
- **五场景检索关系建模**（Figure 3 / Table 18）：
  - r0 **Useless Document**：文档与问题相关但无法提供有用信息，抑制幻觉。
  - r1 **Single-Doc Support**：单文档给线索但无直接答案。
  - r2 **Multi-Doc Support**：多文档联合给线索。
  - r3 **Single-Doc Answer**：单文档直接给出完整答案。
  - r4 **Multi-Doc Answer**：多文档整合完成多步推理。
- **数据生成**：以 SensorIQ 多轮题作为 instruction simulator，每场景检索 1/≥2 篇文档后由生成模型产出 q*, a*, options*（固定 5 选 1–5），输出严格 JSON。
- **三级规则预处理**：① 选项完整性（需标识符+解释文本）；② 排除选项被埋入题干；③ 答案字段为标准字符列表；Qwen 版本初筛 23.9k → 13.6k（丢弃 43%），Claude 版本 26.4k → 26.3k（丢弃 0.5%）。

## 实验与结果
- **数据集规模**：Qwen 版 13,557 样本（训练 12,557 + benchmark 1,000）；Claude 版 26,252 样本（训练 25,252 + benchmark 1,000）。
- **评估指标**：Set-Match Accuracy、F1-Score、Jaccard Similarity；另在 FailureSensorIQ 上使用 AccOrgIBM、AccPerIBM、F1-Macro/Micro。
- **小模型基线（Panasonic）**：Qwen-4B-Instruct Set-Match Acc. 28.5% / F1 46.65%；Phi-3-mini 17.5% / 31.27%；RAG-Instruct-Llama3-8B 仅 0.7% / 0.9%。
- **Qwen 数据微调（Qwen-4B）**：全量微调后 Set-Match Acc. 28.5% → 42.0%，F1 46.6% → 63.5%（有/无 RAG 均稳定提升）；LoRA 各 rank 配置无实质增益。
- **Claude 数据微调（Qwen-4B）**：Set-Match Acc. 40.9% → 56.4%，F1 58.55% → 72.66%；提升幅度大于 Qwen 数据。
- **通用知识保留（MMLU）**：Claude 微调仅 -0.05 分（几乎无遗忘），Qwen 微调 -1.26 分（ Humanities 类 moral_scenarios 单科目 -10.72 分）。
- **FailureSensorIQ 跨 benchmark**：Qwen 微调使 AccOrgIBM 34%→27%、F1-Macro 40%→43%；Claude 微调使 AccOrgIBM 34.0%→49.6%、F1-Macro 40.0%→33.5%；**两类微调在扰动问题 AccPerIBM 上均为 0%**，RAG-Instruct-Llama3-8B 在扰动问题上 33%。

## 相关工作脉络
- **FailureSensorIQ**（Constantinides et al., 2025）：专家标注的工业多轮 QA 基准，评估模型对故障模式-传感器数据的映射；本文在其上与 Panasonic benchmark 双轨验证，发现小模型在扰动问题上的脆弱性。
- **Min et al. (2024)** 表格转文本方法研究：比较 Markdown/模板/BART/LLM 四类策略；本文沿袭"Markdown 保留表格语义 + 与检索向量化兼容"路线，采用 Dots.OCR 一体化解析。
- **QuRE 数据集**（Femmer et al., 2025）：奔驰汽车需求工程数据，关注自然语言需求的复杂度与歧义标注；本文聚焦技术手册/规格表的检索问答而非需求工程，任务形态不同。
- **Semikong**（Nguyen et al., 2024）：半导体垂直领域大模型从头预训练路线；本文走"小型开源基座 + 工业 PDF 指令微调"低资源落地路径，目标部署成本量级不同。
- **HotpotQA**（Yang et al., 2018）：多跳解释性问答；本文取其多文档推理思想，但扩展为 5 种检索–回答关系（含无用文档与支撑类），更贴合真实 RAG 噪声场景。

## 局限性与未来方向
- **扰动鲁棒性缺失**：所有模型在 FailureSensorIQ 重述题上 AccPerIBM=0%，pipeline 未生成 paraphrase/adversarial 变体。
- **单源单厂商**：仅用松下文档，跨企业/跨行业泛化未验证。
- **弃用视觉模态**：移除全部图片损失了示意图/结构图信息，实际工业文档高度图文耦合。
- **基础 RAG**：仅单层语义检索，未探索 HyDE、多路检索、重排序等进阶架构。
- **规模有限**：~13k–26k 样本对复杂工业推理可能不足；需扩展到多源、更大规模。
- **未来方向**：多模态工业文档解析、跨领域知识库扩充、先进 RAG 架构、扰动/对抗变体显式生成以提升鲁棒性、更大参数模型与更多架构的泛化评估。

## 研究启发与可借鉴点
- **五场景检索关系设计**可直接迁移到医疗/法律/金融等技术文档构建流程，尤其 r0（无用文档）对抑制幻觉有价值。
- **同管道 Open vs. Frontier 对照实验范式**可作为团队后续"数据生成器选型"的标准评估模板（质量 × 成本 × 遗忘）。
- **三级规则预处理管线**（选项完整性 / 题干内嵌答案 / 答案格式标准化）是低成本、高复用的数据质量控制模板。
- **小模型全量微调显著优于 LoRA**在工业 QA 上的表现，提示"领域深度知识"需要较大参数更新；团队在类似垂直任务上可优先考虑 full FT 或更大 rank。
- **集合相似度指标（Set-Match / Jaccard）**比 Exact-Match 更适合多轮/多答案场景，建议作为同类任务的默认评估集。

## 关键术语表
- **Industrial-Instruction**：面向工业技术报告的端到端指令微调与基准构建框架，输出多轮选择题 QA 对及对应源文档。
- **Set-Match Accuracy**：将预测答案与参考答案视为无序集合后判等的准确率，忽略顺序与格式差异。
- **RAG（Retrieval-Augmented Generation）**：检索增强生成，通过从外部知识库召回证据文段辅助 LLM 生成答案。
- **Dots.OCR**：多语言文档布局解析视觉语言模型，支持同时识别文本/表格/图像区域并输出 Markdown。
- **EmbeddingGemma**：基于 Gemma3 的轻量嵌入模型（300M 参数），用于语义检索向量构建。
- **FailureSensorIQ**：工业领域多轮选择题基准，评估模型对资产故障模式与传感器信号关系的理解与鲁棒性。
- **Catastrophic Forgetting**：微调/继续训练导致预训练通用知识显著退化的现象。
- **LoRA**：低秩自适应参数高效微调方法，通过在权重更新中注入低秩矩阵减少训练参数量。

## 可复现要素
- **数据集**：Panasonic Dataset，Hugging Face 公开（DOI: 10.57967/hf/10098），含 Qwen/Claude 双版本与 benchmark 划分。
- **代码**：https://github.com/parssky/industrial-instruction（pipeline、生成脚本、评测脚本均已开源）。
- **生成模型**：Qwen3-30B-A3B-Instruct（开源权重）；Claude-Opus-4.6（API）。
- **提取模型**：Dots.OCR。
- **嵌入模型**：EmbeddingGemma（300M）。
- **检索引擎**：FAISS。
- **训练硬件**：2 × NVIDIA RTX 5090。
- **训练时长**：约 12 小时 3 分钟。
- **关键超参**：effective batch size=64（per-device=2，gradient accumulation=32）；论文未详细列出学习率/权重衰减/优化器细节。
