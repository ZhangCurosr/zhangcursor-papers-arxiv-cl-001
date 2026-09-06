---
title: "LeakageBench-Document-Level-Leakage-Risk-for-Redacting-Perso"
source: https://arxiv.org/pdf/2609.02207v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 21:02:37"
field: "文档隐私与PII删除安全评估"
keywords: ["PII redaction", "document-level leakage", "LeakageBench", "privacy benchmark", "vision-language model", "OCR-free"]
innovations: ["提出LeakageBench文档图像PII删除安全基准与文档级泄露指标DocLeak", "统一OCR依赖与OCR-free系统的图像空间预测接口与诊断评估框架", "发现最强系统仍无法使大多数页面达到安全发布标准（关键泄露率96.8%）"]
benchmarks: ["LeakageBench"]
---

# 论文速读：LeakageBench-Document-Level-Leakage-Risk-for-Redacting-Perso

## 一句话总结
本文提出 LeakageBench，一个包含 500 张文档图像、11,954 个 GDPR 对齐 PII 标注的挑战集，用于评估文档图像中个人身份信息删除的文档级泄露风险。实验表明，即便最强的基线系统，关键页面级泄露率仍高达 96.8%，说明现有 PII 删除系统距离实际安全发布仍有显著差距。

## 研究问题与动机
1. **现有基准缺乏文档级风险度量**：多数 PII 基准仅测量文本级 span 精召率，未评估"页面仍有任意标识符可见则整页不安全"的删除安全性。
2. **文档图像存在额外噪声**：OCR 错误、复杂版式、重复标识符和视觉噪声使识别和空间定位远比干净文本困难。
3. **删除存在不对称风险**：假阳性过度遮蔽内容，而单个假阴性即可导致页面不可发布，因此召回导向的文档级指标至关重要。
4. **缺乏统一诊断基准**：现有隐私评估工作（如 PRvL、ProPILE）未提供带空间标注和页面级发布标准的文档图像评测集。

## 核心贡献（创新点）
1. **LeakageBench 挑战集**：包含 500 张文档页、11,954 个空间定位 PII 标注，覆盖异构布局、密集表格、扫描件和混合渠道，与现有文本基准形成互补。
2. **GDPR 对齐的三组隐私分类架构**：将标注分为直接标识符、链接键和上下文重新识别面，反映真实运营文档中的可识别性来源。
3. **文档级泄露指标 DocLeak**：以页面为单位度量残余泄露，作为实体级 F1 的召回导向补充，揭示平均性能掩盖的安全盲区。
4. **统一图像空间预测接口**：支持 OCR 依赖型管道与 OCR-free 视觉语言模型在同一框架下比较，暴露两类架构的差异化失败模式。
5. **细粒度误差归因分析**：将泄露拆解为界面失败、零 IoU、低 IoU 框、匹配冲突等类别，指出视觉语言模型仍以空间定位为主瓶颈而非接口问题。

## 方法详解
- **任务定义**：给定单页文档图像，系统需输出 PII 区域的空间边界框、类型标签及可选文本值；预测以图像坐标为准，跨页无上下文。
- **预测格式**：$\hat{y} = \{(\hat{b}_j, \hat{t}_j, \hat{v}_j, \hat{s}_j)\}_{j=1}^{M}$，其中 $\hat{b}_j$ 为像素级轴对齐框，$\hat{t}_j$ 为 PII 类型，$\hat{v}_j$ 为原文转录值，$\hat{s}_j \in [0,1]$ 为置信度。
- **实例匹配**：页面内一对一贪心 IoU 匹配，阈值 $\tau = 0.75$；分 headline（类型感知）与 type-agnostic（仅位置）两种评估口径。
- **实体级指标**：$\mathrm{F1}_{\mathrm{loc}}$ 为类型无关的位置 F1，$\mathrm{F1}_{\mathrm{type}}$ 需同时满足 IoU 阈值与类型一致。
- **文档级指标**：
  $$\mathrm{DocLeak} = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \mathcal{H}[d \text{ 泄露}]$$
  分别报告 $\mathrm{DocLeak}_{\mathrm{all}}$（所有 PII）与 $\mathrm{DocLeak}_{\mathrm{crit}}$（Direct + Linkage）。
- **基线分类**：
  - OCR 依赖型：Presidio（Tesseract）、Textract + GLiNER 系列 / Amazon Comprehend / Google DLP 等。
  - OCR-free VLM：Qwen3-VL-32B、InternVL3-38B、GPT-5.4/5.5（含 Code Interpreter 工具辅助）。
- **IoU 敏感度分析**：额外报告 $\tau \in \{0.50, 0.75, 0.95\}$ 下的结果以验证结论稳健性。

## 实验与结果
- **数据集**：LeakageBench，500 页、291 份文档，来自 OCR-IDL、VRDU Ad-Buy Forms、FCC 公开档案三源。
- **主要结果（Table 3, τ=0.75）**：
  - 最佳 $\mathrm{F1}_{\mathrm{loc}}$：Amazon Comprehend PII = 0.304。
  - 最佳 $\mathrm{F1}_{\mathrm{type}}$：GPT-5.5 + Code Interpreter = 0.119。
  - 最佳 $\mathrm{DocLeak}_{\mathrm{crit}}$：Amazon Comprehend PII / GPT-5.5+Code Interpreter = 0.968。
  - GPT-5.5 Code Interpreter 将 $\mathrm{F1}_{\mathrm{loc}}$ 从 0.090 提升至 0.249，$\mathrm{F1}_{\mathrm{type}}$ 从 0.050 提升至 0.119，但仍无法显著降低关键泄露率。
- **误差归因（Table 5）**：OCR-free 模型中界面失败占比 0–36.9%，而残留泄露中以低 IoU 框失败为主（GPT-5.5 系 96.1–98.9%）。
- **结论**：更强的检测和工具辅助改善了定位精度，但并未让大多数页面达到安全发布标准；受限类型系统和 OCR 对齐失败共同构成残余泄露。

## 相关工作脉络
1. **Text-based PII benchmarks**（TAB、PII detection on clinical/legal text）：仅评估 span 级精度，不含空间定位和页面安全度量。
2. **Document-image understanding**（FUNSD、SROIE、CORD、DocVQA、BuDDIE）：面向特定提取任务而非全面 PII 定位与删除安全。
3. **OCR-based/OCR-free models**（LayoutLM 系列、Donut）：提供文档理解基础模型，但非以泄露安全为目标设计。
4. **Leakage-oriented privacy evaluation**（PRvL、ProPILE、RedactBuster）：关注 LLM 隐私风险或重识别攻击，但缺乏带空间标注的文档图像基准。
5. **本文定位**：填补文档图像级 PII 删除安全的空白，强调页面级风险而非文本级指标，并提供统一的可复现诊断平台。

## 局限性与未来方向
1. 数据集仅 500 页，覆盖公共商业文档，未包含手写、医疗、法律发现、多语言或非美文档。
2. 评估为单页局部推断，未测量多页上下文推理能力。
3. 轴对齐框无法准确刻画倾斜、手写或不规则形状区域。
4. 缺少文档级过度删除/可用性指标，仅从安全侧评估。
5. 部分 OCR 依赖分析的日志不可用（如 Google DLP 的 vendor OCR 未保存），限制归因完整性。
6. 未来方向：引入多页上下文、扩展到更多语言与手写场景、结合可用性与安全的联合指标、探索训练型 LayoutLM 基线。

## 研究启发与可借鉴点
1. **双指标评估范式**：实体级 F1 与文档级 DocLeak 并行报告，能有效避免平均性能对安全风险的掩盖，适合任何高风险识别任务。
2. **统一图像空间接口**：将 OCR 依赖与 OCR-free 系统映射到同一预测格式，便于横向对比；可迁移到文档解析、版面分析等任务的统一评测设计。
3. **IoU 敏感度 sweeps**：对不同 τ 的报告可作为鲁棒性验证的标准化做法。
4. **误差归因框架**：将残余失败拆分为界面/零重叠/低重叠/冲突四类，有助于定位系统瓶颈并提出针对性改进。
5. **Code Interpreter 等工具辅助**：证明多模态大模型在视觉 grounding 任务中可通过工具调用显著提升精度，但仍不足以解决全部安全问题，提示需结合专用视觉定位头或后处理策略。

## 关键术语表
- **LeakageBench**：论文提出的文档图像 PII 删除安全基准，包含 500 页与 11,954 个空间标注实例。
- **DocLeak**：文档级泄露指标，页面中存在至少一个未匹配 PII 即计为泄露。
- **Direct identifiers**：直接标识符，如姓名、地址、邮箱、电话等直接揭示身份的信息。
- **Linkage keys**：链接键，如保单号、客户 ID、发票号等通过 Lookup 关联到个人的稳定标识。
- **Contextual identifiers**：上下文标识符，如日期、组织名等单独未必是个人信息，但与其他字段结合会提升可识别性。
- **F1_loc**：类型无关的页面级宏平均定位 F1，衡量边界框回归质量。
- **F1_type**：同时满足 IoU 阈值与正确 PII 类型的宏观平均 F1。
- **Type-agnostic / Type-aware matching**：前者仅按位置匹配，后者要求预测类型与金标准一致，用于 headline 与诊断两种评估口径。

## 可复现要素
- **数据集**：LeakageBench（500 页），论文声明将通过隐私和数据授权协议（DUA）向研究使用提供，未完全公开。
- **代码/权重**：论文未提供官方开源代码或权重，附录 B 给出了各基线的详细配置以便复现。
- **关键超参**：IoU 匹配阈值 τ = 0.75（主结果），另报告 τ ∈ {0.50, 0.95} 的敏感性分析。
- **基线配置**：Presidio 使用 Tesseract；其他 OCR 依赖基线共享 AWS Textract；VLM 基线使用官方模型并附统一 prompt 模板（附录 C）。
