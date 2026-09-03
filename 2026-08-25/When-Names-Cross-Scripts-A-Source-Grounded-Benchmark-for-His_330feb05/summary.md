---
title: "When-Names-Cross-Scripts-A-Source-Grounded-Benchmark-for-His"
source: https://arxiv.org/pdf/2608.23507v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:22:35"
field: "历史 NLP / 跨语言实体消歧"
keywords: ["historical entity reconciliation", "source-grounded benchmark", "cross-script names", "entity linking", "abstention", "provenance control", "Mongol world NLP"]
innovations: ["提出来源可控的 MHER 基准，以 mention×source 粒度约束历史证据的关联方式，防止上下文泄漏", "设计配对干预与打乱上下文证伪控制，分离名称表面证据与来源锚定证据对身份判断的贡献", "在历史实体消歧任务中系统评估弃权（abstention）作为独立能力维度，并验证协议有效性"]
benchmarks: ["MHER (Mongol-world Historical Entity Reconciliation)", "Name-only core (396 pairs)", "Source-grounded paired subset (160 pairs)", "Expert-unresolved challenge (10 cases)"]
---

# 论文速读：When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World

## 一句话总结
本文提出了 **MHER**（Mongol-world Historical Entity Reconciliation），一个来源可控的历史实体消歧配对基准，旨在评估多语言/多文字系统中历史人名身份归并能力。研究表明，在提供来源锚定的历史证据后，五个生成模型的身份判断准确率相比纯名称输入提升 **12.96–94.44 个百分点**，同时揭示了"名字是身份的证据，但名字本身不等于身份关系"这一核心论断。

---

## 研究问题与动机

1. **跨语言/跨文字的历史命名复杂性**：同一历史人物在不同语言、文字、转写传统、学者罗马化体系中可能出现形式差异极大的记录；而不同人物又可能拥有高度相似甚至完全相同的名字，使得简单的字符串匹配无法解决身份归并问题。
2. **现有实体链接范式的局限**：现代实体链接将提及映射到预定义知识库（KB）中的规范实体，但历史场景下 KB 往往遗漏长尾人物、史料残缺或不一致，且现代参考资源中的归并本身可能是学术争议的产物。
3. **上下文引入的评估难题**：为模型提供"背景信息"存在泄漏风险——共享上下文可能无意中透露答案，或通用主题相似度成为非意图的标签线索；因此**来源可靠性（provenance）** 本身是语义有效性的组成部分，而非可选的元数据。
4. **身份判断需区分"证据不足"与"错误判断"**：强制二元决策混淆了两类不同的失败模式，需要在评估框架中允许明确的弃权（abstention）动作，以区分不正确解析与对证据充分性的正确识别。

---

## 核心贡献（创新点）

1. **将历史实体消歧形式化为证据条件任务**：直接对两个来源锚定的人名提及进行配对身份判断（$m_a \overset{?}{\equiv} m_b$），区别于传统的提及→KB 映射范式，并支持 AMBIGUOUS 弃权动作。
2. **提出来源可控的 MHER 基准**：构建 84 位主要历史人物、396 对 Name-only 配对（DEV/TEST 按实体分离），以及更严格的 160 对 Source-grounded 子集（基于 mention×source 级别的独立证据）。
3. **设计泄漏感知的证据干预与证伪机制**：以 mention×source 粒度构建上下文，要求 same 配对两侧提供独立来源证据；配套 Context-only 消融和三种确定性打乱上下文控制，区分"有额外文本"与"正确来源锚定"的信息。
4. **超越强制二元精度的评估框架**：结合词法/转写/多语言嵌入/专有模型/开源模型五类系统，评估 abstention 率、选择性精度、概率校准质量、表面易混淆挑战案例、重复实体鲁棒性及专家未决挑战。

---

## 方法详解

### 任务形式化
- 人名提及表示为 $m_i = (s_i, \ell_i, \sigma_i)$，其中 $s_i$ 为表面形式，$\ell_i$ 为语言，$\sigma_i$ 为文字/转写系统。
- 身份关系：$y(m_a, m_b) \in \{\text{SAME, DIFFERENT}\}$（二元黄金标准）。
- 模型动作集合：$\mathcal{A} = \{\text{SAME, DIFFERENT, AMBIGUOUS}\}$，其中 AMBIGUOUS 为弃权动作，而非第三类身份。

### 证据条件划分
- **Name-only 条件**：$x_i^{\text{name}} = (m_a, m_b)$，仅输入两个提及及语言/文字元数据。
- **Source-grounded 条件**：$x_i^{\text{source}} = ((m_a, e_a), (m_b, e_b))$，附带来源锚定的历史证据（年代、职官、亲属关系、地理、事件参与等）。
- **Context-only 条件**：$x_i^{\text{context}} = (e_a, e_b)$，移除表面名称，仅保留历史描述，用于检验上下文本身的身份信息量。
- **Shuffled context 条件**：$x_{i,r}^{\text{shuffle}} = ((m_a, \tilde{e}_{a,r}), (m_b, \tilde{e}_{b,r}))$，三种确定性失配控制，保留上下文池但破坏 mention×source 对齐，显式告知模型证据已被打乱。

### 基准构建约束
- 来源证据不得明确陈述"两人是同一人"或列出目标的注册别名（防止标签泄漏）。
- same 配对两侧须有独立来源证据；若无法满足，则该对仅保留在 Name-only 核心中。
- DEV/TEST 按规范实体分离（17/67），同一实体的变体不会跨越划分边界。

### 评估指标
- 普通准确率（accuracy）、宏平均 F1（macro-F1）、弃权率（abstention rate）、条件准确率（selective accuracy）。
- 置信度质量：黄金关系概率均值、多分类 Brier score、log loss。
- 配对统计：每模型错误→正确 / 正确→错误转移计数，McNemar 精确检验，Bonferroni 校正；实体簇 bootstrap、leave-one-entity-out 鲁棒性检验。

### 基线系统
- **词法/转写基线**：精确匹配、Unicode 规范化、Levenshtein、Jaro-Winkler、字符 2–5-gram TF-IDF、Unidecode 转换后编辑距离。
- **上下文基线**：TF-IDF、token Jaccard、序列相似度、六特征逻辑回归分类器。
- **多语言嵌入基线**：Google gemini-embedding-2。
- **生成系统**：GPT-5.6 Terra/Luna/Sol（OpenAI API）、Gemini 3.7 Flash（Google API）、Qwen3-8B（本地冻结量化部署）。

---

## 实验与结果

### 数据集规模
- **84** 位主要历史人物（14 头部 / 41 中部 / 29 尾部）
- **396** 对 Name-only 配对（DEV 80 + TEST 316，SAME/DIFFERENT 各半）
- **160** 对 Source-grounded 配对（DEV 52 + TEST 108）
- **10** 例专家未决挑战

### 非生成基线（TEST 结果）
| 证据类型 | 方法 | 准确率 | Macro-F1 |
|---|---|---|---|
| Name-only | Gemini Embedding 2 | 64.87% | .625 |
| Name-only | Unidecode + Levenshtein | 59.81% | .598 |
| Context | Token Jaccard | 76.85% | .758 |
| Context | Gemini Embedding 2 | **86.11%** | **.861** |

### 主要配对干预结果（108 对 TEST）

| 模型 | Name-only | Source-grounded | 提升(pp) | 95% CI | W→C | C→W |
|---|---|---|---|---|---|---|
| GPT-5.6 Terra | 32.41% | **100.00%** | **+67.59** | [58.33, 75.93] | 73 | 0 |
| GPT-5.6 Luna | 6.48% | 99.07% | **+92.59** | [87.04, 97.22] | 100 | 0 |
| GPT-5.6 Sol | 4.63% | 99.07% | **+94.44** | [89.81, 98.15] | 102 | 0 |
| Gemini 3.7 Flash | 20.37% | **100.00%** | **+79.63** | [72.22, 87.04] | 86 | 0 |
| Qwen3-8B | 75.00% | 87.96% | +12.96 | [2.78, 23.15] | 23 | 9 |

- **最强结果**：GPT-5.6 Sol 提升最大（+94.44pp），四款商业 API 模型在 Source-grounded 条件下均接近天花板（87.96%–100%）。
- **表面易混淆挑战**：5 个同名不同人案例，Name-only 条件下全部模型得分为 **0/25**；Source-grounded 条件下为 **24/25** 正确，剩余 1 例为弃权。

### 消融与控制结果
- **Context-only**：四款 API 模型仍保持 80.56%–94.44% 准确率，表明历史描述本身携带大量身份信息。
- **Qwen3-8B 的反向效应**：Context-only（94.44%）> Source-grounded（87.96%），恢复名称后 10 例正确区分被转为错误合并（surface-form interference）。
- **Shuffled context**：Source-grounded 与三种打乱控制的差距：Terra +85.19pp、Luna +82.41pp、Sol +96.30pp、Gemini +50.31pp、Qwen3-8B +28.09pp。
- **专家未决挑战**：五款模型在 10 例中全部输出 AMBIGUOUS（50/50 弃权），验证协议有效性。

---

## 相关工作脉络

1. **多语言实体链接（mGENRE、MOLEMAN）**：mGENRE（De Cao et al., 2022）以自回归方式生成多语言实体名并映射到 KB 库存；MOLEMAN（FitzGerald et al., 2021）学习多语言实体提及的上下文表示后检索同类标注提及。MHER 与其结构不同：不假设完整 KB 已知，直接对两个来源提及做配对身份判断。
2. **跨文字名字与转写（ParaNames、Khakhmovich et al.）**：ParaNames 包含 400+ 语言约 1.4 亿条名字记录，支持名字翻译/转写/NER/实体链接任务。MHER 将表面对应视为一种证据来源而非目标关系本身。
3. **历史 NLP 资源（KE-MHISTO、Mahanama、MHEL-LLaMo）**：KE-MHISTO（Graciotti et al., 2025）聚焦多语言历史知识提取的长尾问题；Mahanama（Sarkar et al., 2025）含 10.9 万提及/5,500 实体，展示文学实体消歧需远超表面匹配的推理；MHEL-LLaMo（Santini et al., 2026）将多语言 bi-encoder 检索与 LLM 置信度选择应用于欧洲六语历史 EL。MHER 的定位差异在于：不做提及→候选/KB 映射，而是操纵附属于两个提及的证据并观察身份判断的变化。
4. **记录链接（Fellegi-Sunter、Abramitzky et al.）**：经典框架将身份匹配视为从多个部分 informative 记录字段的推断；历史记录的链接研究表明姓名非唯一且易出错，补充地理/家庭关系等信息可显著提升匹配可靠性。MHER 共享此证据组合观，但研究对象是源锚定多语言历史人名及可见叙事证据，而非表格人口普查记录。
5. **来源锚定与证伪（ALCE、ExpertQA）**：ALCE（Gao et al., 2023）评估模型回答及其引用；ExpertQA（Malaviya et al., 2024）使用专家策展问题测试事实性与归属。MHER 将类似关注从输出端前移至输入端——上下文只有在与目标提及正确关联时才构成有效历史证据。

---

## 局限性与未来方向

1. **领域与任务范围受限**：仅覆盖蒙古世界人名，未涵盖其他历史时期/地区/文献传统，也未评估 NER、候选检索、文档检索或组织/地点/事件等其他实体类型。
2. **证据可及性选择的天花板效应**：Source-grounded 子集仅纳入两侧均有独立来源证据的配对，近天花板性能不代表历史档案中总存在充分证据；稀疏、损坏、冲突或索引不良的史料将大幅削弱效果。
3. **上下文简短且人工精简**：当前历史描述刻意紧凑，未来需在更长、含噪声、部分相关、存在冲突证词的原始文档上进行评估。
4. **未决案例仅测试 Name-only 条件**：因现有学术不确定性注释含明确泄漏语言，未构建 Source-grounded 未决条件；更强的设计需中立证据不含不确定性暗示。
5. **基准规模较小（84 人物、396 对）**：反映来源锚定证据的手动审计成本；更大规模将提升跨实体/跨域泛化的估计力度。
6. **模型身份与部署可复现性**：商业 API 存在版本迭代风险，Qwen3-8B 结果仅对应单一冻结部署；精确执行记录保留在冷冻项目状态中但未公开于本版。

---

## 研究启发与可借鉴点

1. **来源控制的基准设计范式可迁移**：MHER 将 provenance 作为评估输入语义的一部分（而非事后附加元数据），这一原则可直接应用于其他需要利用历史/文献上下文的 NLP 任务（如历史文献引用、跨源知识整合）。
2. **泄漏感知的上下文构造方法**：mention×source 粒度约束、same 配对需独立来源证据、预评估审计流程——这套设计可复用至任何涉及外部上下文注入的基准建设中，防止巧合共现成为捷径。
3. **配对干预（paired intervention）作为因果性更强的评估策略**：通过固定身份问题仅改变证据条件，直接测量"证据增量"对模型判断的影响，比跨模型排名更有解释力；适用于评估 RAG、上下文增强、工具调用等场景的效果度量。
4. **弃权作为独立能力维度的评估价值**：区分"错误判断"与"对证据不足的合理识别"，配合 expert-signaled unresolved 协议验证，为 selective prediction 在历史 NLP 中的应用提供了可操作模板。
5. **打乱上下文控制作为 informed stress test**：保存上下文池但破坏 mention×source 对齐，并显式告知模型——这种方法比完全盲态测试更具诊断价值，可用于检验模型是否真正依赖来源锚定而非文本风格/长度等表面特征。

---

## 关键术语表

**Historical Entity Reconciliation（历史实体消歧）**：对两个来源锚定的人名提及进行配对判断，确定它们是否指向同一历史人物；区别于传统的提及→KB 映射。

**Source-grounded（来源锚定）**：历史证据与具体提及在 mention×source 级别精确关联，而非从共享规范实体描述中复制；证据来源的可追溯性是上下文有效性的前提。

**Abstention / AMBIGUOUS（弃权/拒绝作答）**：模型在证据不足以支持二元判断时选择放弃作答，而非强制输出 SAME 或 DIFFERENT；用于区分错误判断与对不确定性的正确识别。

**Paired Intervention（配对干预）**：对同一组历史身份问题，在 Name-only 和 Source-grounded 两种证据条件下测量模型判断的变化，以量化证据增量的因果效应。

**Shuffled-context Control（打乱上下文控制）**：保留相同历史上下文池但人为破坏 mention×source 对齐，并显式告知模型证据已被打乱；用于检验正确来源锚定是否为高性能的必要条件。

**Surface-form Interference（表面形式干扰）**：当强历史上下文已足够区分不同人物时，恢复名称表面形式反而触发不支持的身份假设，导致原本正确的 DIFFERENT 判断被转为错误的 SAME——本文 Qwen3-8B 的典型错误模式。

**Entity-disjoint Split（实体分离划分）**：按规范实体而非提及配对划分 DEV/TEST，确保同一人的不同变体不会出现在对立划分中，防止间接泄漏。

**Provenance-controlled Benchmark（来源可控基准）**：将来源可靠性作为基准构造的核心约束而非可选元数据，确保评估输入的语义有效性依赖于证据的正确归因。

---

## 可复现要素

- **数据集**：MHER 基准数据及项目级材料**未公开**，计划于正式发表后发布（受来源重新分发限制）；可见历史证据将以项目撰写 paraphrase + 书目引注形式发布。
- **代码**：未公开，计划于发表后开源；当前 arXiv 版本不包含代码。
- **权重**：Qwen3-8B 使用冻结的本地量化部署；精确 artifact 身份、运行时和解码来源保留在冷冻项目记录中，本版未公开。
- **关键超参**：API 模型的解码参数和重试策略在正式 TEST 前冻结；非生成基线的阈值在 DEV 上选择后锁定；具体数值在论文中未详细列出。
- **统计方法**：McNemar 精确检验（配对）、200,000 次 paired item bootstrap、Bonferroni 校正、实体簇 bootstrap、leave-one-entity-out 分析；具体实现细节见论文 Section 5.6。

---
