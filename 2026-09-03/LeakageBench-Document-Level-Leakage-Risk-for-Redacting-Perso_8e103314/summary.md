---
title: "LeakageBench-Document-Level-Leakage-Risk-for-Redacting-Perso"
source: https://arxiv.org/pdf/2609.02207v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 21:02:43"
field: "文档理解与隐私安全"
keywords: ["PII 隐藏", "文档图像理解", "LeakageBench", "文档级泄露", "OCR-free VLM", "隐私保护"]
innovations: ["提出页面级 DocLeak 指标，以任一关键标识符未匹配触发页面级安全判定", "构建 GDpR/CCPA 对齐的 Direct/Linkage/Contextual 三组 schema 与统一图像空间预测接口", "建立 OCR-miss / detector-miss / box-alignment / interface-failure 四维误差归因框架，揭示最强系统仍无法实现页面安全释放的根因"]
benchmarks: ["LeakageBench", "OCR-IDL", "VRDU Ad-Buy Forms", "PRvL", "ProPILE", "RedactBuster"]
---

# 论文速读：LeakageBench-Document-Level-Leakage-Risk-for-Redacting-Perso

## 一句话总结
本文提出 LeakageBench，一个面向文档图像 PII 隐藏的文档级泄露风险基准，包含 500 页图像与 11,954 个 GDPR 对齐标注；评测发现即使最强系统（GPT-5.5 + Code Interpreter）仍将关键 PII 泄露在 96.8% 的页面上，凸显"页面安全释放"与"实体级定位 F1"之间的巨大鸿沟。

## 研究问题与动机
- **现有基准的文本中心化缺陷**：主流 PII 基准以纯文本为输入，只报告 span 级 precision/recall/F1，不评估文档图像中的空间定位与页面级释放安全。
- **文档级安全指标的缺失**：真实业务场景中，只要一页上有一个 in-scope 标识符未被红删，该页即视为不安全；当前评估无法量化这一"发布前最终安全"状态。
- **隐藏的不对称运营权衡**：FN 代价极高（直接导致页面不可发布），FP 代价相对较低（仅降低文档可用性），因此召回导向的 DocLeak 是比平均 F1 更贴近部署安全的指标。
- **OCR 依赖与 OCR-free 系统的共性瓶颈**：两者在复杂布局、密集表格、OCR 片段化、重复标识符等真实文档图像条件下，都存在难以克服的空间定位与 schema 覆盖差距。

## 核心贡献（创新点）
- **首个面向文档图像 PII 隐藏的密集挑战集**：500 页、11,954 个标注，包含 Direct / Linkage / Contextual 三组隐私对齐标签，区别于以往仅关注单一字段提取的数据集。
- **文档级泄露指标 DocLeak**：以页面为粒度、以"任一关键标识符未匹配"为触发条件，弥补 F1 类指标掩盖单页风险的盲区。
- **统一图像空间预测接口**：OCR 依赖与 OCR-free 系统通过同一 `(bbox, type, value, score)` 格式对齐，使不同架构间的对比真正公平可比。
- **多维度诊断归因框架**：将失败分解为 schema 覆盖、OCR 可恢复性、检测失败、界面有效性与空间 grounding，揭示"为什么最强系统仍无法让页面安全"。
- **GDPR/CCPA/NIST/HIPAA 对齐的 identifiability-first 模式设计**：将 Linkage keys（保单号、客户 ID 等）视为与直接标识符同等重要的安全目标，反映真实监管触发条件。

## 方法详解
- **任务定义**：给定单页文档图像，系统输出预测集合 $\hat{y} = \{(\hat{b}_j, \hat{t}_j, \hat{v}_j, \hat{s}_j)\}_{j=1}^M$，其中 $\hat{b}_j$ 为像素坐标轴对齐包围盒、$\hat{t}_j$ 为 PII 类型、$\hat{v}_j$ 为可选原文值、$\hat{s}_j$ 为置信度。
- **实例匹配**：页面内一一匹配，IoU ≥ τ（主结果 τ = 0.75）的预测–金标对视为 localization match；type-aware 匹配还要求类型一致。
- **实体级指标**：
  - $\mathrm{F1}_{\mathrm{loc}}$：页面宏观平均、type-agnostic 的定位 F1。
  - $\mathrm{F1}_{\mathrm{type}}$：IoU ≥ τ 且类型一致的 macro F1。
- **文档级指标**：
  - $\mathrm{DocLeak}_{\mathrm{all}} = \frac{1}{|\mathcal{D}|}\sum_{d\in\mathcal{D}}\mathcal{H}[d \text{ leaks}]$，覆盖所有 PII。
  - $\mathrm{DocLeak}_{\mathrm{crit}}$：仅统计 Direct + Linkage 的关键泄露比例。
- **误差归因**：将每种泄漏实例分解为 OCR miss / detector miss（零 IoU）/ box-alignment failure（正 IoU < τ）/ assignment conflict，用于定位瓶颈。
- **支撑性评估**：Supported-type leakage（过滤为系统支持类型）、IoU 阈值扫测（0.50 / 0.75 / 0.95）、空间 PII 热图（14×14 像素网格）。

## 实验与结果
- **数据集**：500 页来自 3 个源（OCR-IDL 113 页、VRDU Ad-Buy 222 页、FCC 非结构化 165 页），共 11,954 个实例，平均 23.9 个/页；关键标识符（Direct+Linkage）8.63 个/页。
- **评测基线**：9 种 OCR 依赖管线（Presidio / GLiNER-base/multi-PII/NVIDIA-GLiNER / GLiNER2 / Amazon Comprehend PII / Google DLP / OpenAI Privacy Filter / Presidio-best）、4 种 OCR-free VLM（Qwen3-VL-32B / InternVL3-38B / GPT-5.4 / GPT-5.5 / GPT-5.5+Code Interpreter）。
- **核心数字**：
  - 最强 $\mathrm{F1}_{\mathrm{loc}}$ = 0.304（Amazon Comprehend PII）；最强 $\mathrm{F1}_{\mathrm{type}}$ = 0.119（GPT-5.5 + CI）。
  - 最强 $\mathrm{DocLeak}_{\mathrm{crit}}$ = 0.968（Amazon Comprehend PII / GPT-5.5 + CI），几乎所有系统 ≥ 0.97。
  - GPT-5.5 + CI 相比 GPT-5.5 单遍：$\mathrm{F1}_{\mathrm{loc}}$ 0.090 → 0.249，$\mathrm{F1}_{\mathrm{type}}$ 0.050 → 0.119；但 Crit 泄露仅从 0.990 降至 0.968。
  - 典型泄漏页面平均仍有 5.3–8.7 个关键标识符未被定位（median 4，P90 11–18）。
- **结论**：更强的检测器与工具辅助能提升定位 F1，但无法使大多数页面达到"可发布"安全水平；schema 覆盖不全与 OCR 对齐失败是主要根因，即使全 schema 覆盖的系统在 Linkage 组上泄露率仍接近 100%。

## 相关工作脉络
- **Text-based PII 基准**（TAB / SpEdAC / PRvL 等）：以纯文本为输入，报告 span F1，未覆盖文档图像中的空间定位与页面级释放安全。
- **文档理解基准**（FUNSD / SROIE / CORD / DocVQA / BuDDIE）：面向表单理解、OCR 或 key 字段抽取，目标是任务级准确率而非 exhaustive PII 隐藏安全。
- **OCR-dependent 模型**（LayoutLM / LayoutLMv3）：将 OCR 文本与二维布局结合，但会传播 OCR 错误与 span-to-box 对齐误差；LeakageBench 用统一接口对比其与 OCR-free 方案的相对劣势。
- **OCR-free VLM**（Donut / Qwen3-VL / InternVL3 / GPT 系列）：直接处理图像，但受限于空间 grounding 精度与输出格式稳定性；本文首次将其与 OCR 管线在同一文档级泄露标准下对比。
- **隐私泄露评估**（PRvL / ProPILE / RedactBuster）：关注 LLM 的隐式 PII 暴露或下游推理再识别，但未提供带局部图像空间标注的页面级 release criterion 基准。
- **LeakageBench 的定位差异**：从"文本 span 识别"转向"页面能否安全发布"，从"平均性能"转向"最坏情况（任一遗漏即失败）"，从"单一指标"转向"F1 + DocLeak + 多粒度归因"的诊断组合。

## 局限性与未来方向
- **数据集规模与代表性有限**：仅 500 页，集中于美国业务文档（发票、邮件、表格），未覆盖手写文档、医疗记录、法律文书、多语言与非英语文档。
- **单页局部推理**：评估限制为单页独立处理，未检验跨页上下文关联（如同一人在多页反复出现）下的红删安全性。
- **轴对齐框与不规则区域的偏差**：IoU 与矩形框难以准确度量倾斜、手写、签名等不规则 PII 区域；缺少像素级覆盖评估。
- **缺少过隐藏（over-redaction）/ 可用性度量**：DocLeak 只惩罚 FN，不惩罚 FP；无法反映"过度红删损害文档可用性"的另一面权衡。
- **归因数据的 artifacts 限制**：35 个关键实例缺少 verbatim 值、span 级对齐日志不可用、Google DLP 的 vendor OCR 未存储，导致部分 error attribution 只能近似。
- **未来方向**：扩展至多语言 / 多领域；引入跨页推理与身份链接追踪；加入 utility / over-redaction 联合指标；探索 tool-augmented VLM 的最佳实践（如 Code Interpreter 裁剪、滚动、放大）的系统化设计。

## 研究启发与可借鉴点
- **文档级安全指标的移植性**：DocLeak 的"任一遗漏即失败"思想可直接迁移至其他文档理解安全任务（如合同条款遮盖、医学影像诊断区遮蔽）。
- **误差归因框架**：将泄漏分解为 OCR miss / detection miss / box alignment / interface failure 的四路归因，可作为通用文档 AI 系统的诊断模板复用。
- **IoU 阈值扫测 + Supported-type leakage 的组合评估**：同时报告 τ ∈ {0.5, 0.75, 0.95} 与按 schema 过滤的组级泄露，能帮助研究者区分"模型能力不足"与"接口/schema 定义问题"。
- **工具辅助 VLM 的系统化评估**：Code Interpreter 显著提升 GPT-5.5 的 F1（0.09 → 0.25）但仍无法突破 DocLeak 瓶颈，提示未来工作应研究"何时工具调用真正改善空间 grounding"的条件与策略。
- **identifiability-first 的 schema 设计**：将 Linkage keys（保单号、员工 ID、参考号等）与 Direct 并列，而非仅关注姓名/地址，反映了真实合规触发的关键路径，值得同类隐私基准借鉴。

## 关键术语表
- **LeakageBench**：面向文档图像 PII 隐藏的文档级泄露基准，含 500 页图像与 11,954 个标注。
- **DocLeak**：页面级泄露指标，定义为"至少有一个 in-scope PII 未被匹配"的页面比例。
- **Direct identifiers**：直接揭示身份或联系方式的 PII（姓名、地址、邮箱、电话）。
- **Linkage keys**：可通过组织系统反向映射到个人的稳定键值（保单号、客户 ID、发票号等）。
- **Contextual identifiers**：单独不足以识别个人，但与其他 PII 共现时提升再识别风险的上下文字段（日期、位置、组织名）。
- **Type-aware / type-agnostic matching**：前者要求 IoU 与类型均一致才计入保护；后者仅看 IoU，用于诊断"类型错误但位置正确"的空间覆盖率。
- **Supported-type leakage**：将金标过滤为系统支持的类型后计算的泄露率，用于隔离 schema 覆盖不足与检测定位失败。
- **OCR-free vision-language model**：不经过显式 OCR，直接将文档图像输入 VLM 以预测 PII 位置与类型的模型（如 GPT-5.5、Qwen3-VL、InternVL3）。

## 可复现要素
- **数据集**：LeakageBench（500 页、3 个公开源混合）；论文声明将通过隐私导向的 Data Use Agreement（DUA）提供图像与标注，供研究使用。
- **代码**：论文未明确声明开源仓库，但提供了完整的基线配置（Appendix B）、prompt 模板（Appendix C）、基准清单（Table 6）与 IoU 扫测结果（Appendix G），可据此复现。
- **关键超参**：IoU 阈值 τ = 0.75（主结果）；扫测 τ ∈ {0.50, 0.75, 0.95}；VLM 坐标归一化至 [0, 999]；GPT-5.5 使用 high image detail + medium reasoning effort；Code Interpreter 可裁剪/缩放图像。
- **基线版本**：Microsoft Presidio (2026)、Amazon Textract / Comprehend (2026)、Google DLP (2026)、OpenAI Privacy Filter (2026)、NVIDIA GLiNER PII (2025)、GLiNER / GLiNER2 (2024/2025)、Qwen3-VL (2025)、InternVL3 (2025)。
