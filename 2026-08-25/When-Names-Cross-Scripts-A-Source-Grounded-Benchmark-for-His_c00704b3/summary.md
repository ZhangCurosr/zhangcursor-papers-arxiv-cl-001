---
title: "When-Names-Cross-Scripts-A-Source-Grounded-Benchmark-for-His"
source: https://arxiv.org/pdf/2608.23507v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:22:53"
field: "历史自然语言处理 / 实体链接"
keywords: ["Historical Entity Reconciliation", "Source-Grounded Benchmark", "Cross-Script Names", "Abstention Evaluation", "Mongol World History", "Provenance Control"]
innovations: ["提出基于来源 grounding 的历史实体成对消歧基准 MHER，严格分离名称证据与来源证据", "设计防泄漏上下文构建与确定性打乱对照实验，验证模型对来源对齐的依赖性", "揭示名称恢复在强上下文下的表面形式干扰效应，挑战多源融合的单调性假设"]
benchmarks: ["MHER (Name-only core: 396 pairs, Source-grounded subset: 160 pairs)", "Gemini Embedding 2 baseline", "GPT-5.6 Terra/Luna/Sol", "Gemini 3.7 Flash", "Qwen3-8B"]
---

# 论文速读：When-Names-Cross-Scripts-A-Source-Grounded-Benchmark-for-His

## 一句话总结
本文提出了 **MHER**，一个针对蒙古世界历史人物实体的来源可控（provenance-controlled）基准，通过严格区分“仅名称输入”与“附带独立来源证据输入”，证明历史实体消解不仅依赖表面名称对应，更高度依赖于与提及严格绑定的历史证据，并在此基础上揭示了名称恢复可能引发的表面形式干扰现象。

## 研究问题与动机
1. **名称与身份的不对等性**：历史人物在不同语言、文字及转录传统中呈现极大拼写差异，而不同人物又可能共享完全相同或高度相似的名称，导致单纯的字符串匹配或转写无法解决身份归属问题。
2. **现有实体链接范式的局限**：主流工作将提及映射到预定义知识库，但历史场景中知识库往往缺失长尾人物，且现代参考资源中的身份裁决本身可能是学者的一种假设而非定论。
3. **上下文证据的来源陷阱**：为模型提供“上下文”虽能提升性能，但如果上下文是从规范实体描述中复制而来，极易意外泄露答案；若来源未与具体提及绑定，则非有效证据。因此，**来源（provenance）是任务语义的一部分，而非可选元数据**。

## 核心贡献（创新点）
1. **将历史实体消解形式化为证据条件任务**：不同于传统 mention-to-KB 链接，本文直接评估两个来源提及是否指向同一历史个体，并显式引入 **ABSTENTION（弃权）** 动作，允许系统在证据不足时拒绝做出二分类判断。
2. **构建 MHER 基准**：包含 84 位核心历史人物、396 对名称对的平衡测试集，以及更严格的 160 对来源验证子集（108 对 TEST），采用实体独立的 DEV/TEST 划分以防止别名泄漏。
3. **设计防泄漏的证据干预与证伪控制**：首次系统化地分离“名称证据”与“来源证据”，构造 Context-only 消融、三种确定性打乱上下文（Shuffled context）对照以及专家未决案例，以验证模型是否真正依赖正确的来源对齐。
4. **揭示名称恢复的干扰效应**：发现对于开放权重模型 Qwen3-8B，在已有强上下文证据时恢复名称表面形式，反而会导致 10 例原本正确的“不同人”判断被错误合并为“同一人”，证明名称并非单调增益特征。

## 方法详解
### 任务形式化
给定两个带有语言 $\ell_i$、文字/转录系统 $\sigma_i$ 和表面形式 $s_i$ 的姓名提及 $m_a, m_b$，目标是推断二元关系 $y(m_a, m_b) \in \{\text{SAME, DIFFERENT}\}$。评估系统可输出三种动作 $\mathcal{A} = \{\text{SAME, DIFFERENT, AMBIGUOUS}\}$，其中 AMBIGUOUS 代表弃权。

### 数据集构建约束
- **来源独立性**：对于 SAME 对，两侧的上下文证据必须来自独立 attestations，禁止从共享的规范实体描述中复制。
- **防泄漏审计**：排除任何直接陈述“同一人”、“等同于”或泄露注册别名的上下文；最终审计未发现重复上下文或明确等价线索。
- **固定划分**：17 个实体用于 DEV，67 个用于 TEST，确保同一历史人物不出现在两侧。

### 评估条件
对 108 对来源验证的 TEST 项，构造以下五种条件：
1. **Name-only**：仅输入 $(m_a, m_b)$ 及语言/文字元数据。
2. **Context-only**：移除名称，仅输入正确绑定的历史证据 $(e_a, e_b)$。
3. **Source-grounded**：输入 $(m_a, e_a), (m_b, e_b)$ 的组合。
4. **Shuffled context (A/B/C)**：名称保持不变，但上下文被确定性重排（排除自关联和固定点），并向模型明示上下文已错位。
5. **Expert-unresolved**：10 个学者未决案例，指令明确允许弃权。

### 模型与输出
评估五个生成系统：GPT-5.6 Terra/Luna/Sol、Gemini 3.7 Flash、Qwen3-8B。每个模型输出三个动作的概率分布 $\mathbf{p}_i = (p^{\text{same}}, p^{\text{different}}, p^{\text{ambiguous}})$，并以最大概率作为预测动作。

## 实验与结果
### 非生成基线
在 316 对 Name-only TEST 上，最佳非生成方法为 Gemini Embedding 2（64.87% accuracy）；在 108 对 Context 子集上，Token Jaccard 达 76.85%，Gemini Embedding 2 达 86.11%。

### 生成模型主结果
- **Name-only 表现高度依赖策略**：API 模型普遍高弃权（Terra 35.13% acc / 63.61% abst；Luna/Sol >93% abst），而 Qwen3-8B 零弃权但准确率 67.09%。
- **Source-grounded 大幅提升**：在 108 对配对测试中，所有模型显著提升：
  - GPT-5.6 Sol：+94.44 pp（4.63% → 99.07%）
  - GPT-5.6 Luna：+92.59 pp（6.48% → 99.07%）
  - Gemini 3.7 Flash：+79.63 pp（20.37% → 100%）
  - GPT-5.6 Terra：+67.59 pp（32.41% → 100%）
  - Qwen3-8B：+12.96 pp（75.00% → 87.96%）
- **相同表面不同人挑战**：5 对完全同名不同人，Name-only 全错（0/25 model-item），Source-grounded 全对（24/25 正确，1 弃权）。

### 消融与证伪
- **Context-only 保持高分**（80.56%–94.44%），表明历史描述本身蕴含强身份信号。
- **打乱上下文后性能骤降**：例如 Terra 从 100% 降至 13.89%–16.67%，证明模型依赖正确的来源对齐而非泛文本特征。
- **Qwen3-8B 的表面形式干扰**：13 处 Source-grounded 错误中，10 例为 Context-only 正确但恢复名称后变为错误合并（unsupported alias / transliteration 或 decision-rationale inconsistency）。

## 相关工作脉络
1. **多语言实体链接**（Pan et al., 2017; Botha et al., 2020; mGENRE）：将提及映射到 KB，MHER 不依赖完整 KB，直接做 pairwise 消解，更适配历史长尾场景。
2. **跨文字名称与转写**（ParaNames, Upadhyay et al., 2018）：关注名称跨文字对应，MHER 认为表面对应仅是证据之一，不能替代身份推断。
3. **历史 NLP/实体链接**（Blouin et al., 2024; KE-MHISTO; Mahanama; MHEL-LLaMo）：处理历史文本的实体识别与链接，MHER 聚焦于证据组合与来源 grounding，而非单纯链接到候选。
4. **记录链接**（Fellegi–Sunter, 1969; Abramitzky et al., 2021）：统计框架下多源证据匹配，MHER 借鉴其思想但针对非表格化多语言历史文本。
5. **检索增强与归因生成**（RAG, ALCE, ExpertQA）：强调输出引用支持，MHER 在输入端强调来源对齐，区分“有上下文”与“有正确证据”。

## 局限性与未来方向
- **领域与任务范围有限**：仅覆盖蒙古世界历史人物，不涉及 NER、候选检索或端到端管线；未评估组织、地点等其他实体类型。
- **证据选择偏差**：Source-grounded 子集仅包含证据充足且独立的案例，近天花板结果不代表稀疏/冲突/残缺档案下的泛化能力。
- **未决案例规模小**：仅 10 个专家未决案例，且仅测试 Name-only 条件，缺乏 Source-grounded 未决评估。
- **未来方向**：扩展至其他历史时期与地区；处理更长、噪声更大、证据冲突的文档；设计盲测对照（不告知模型上下文已打乱）；探索模型自主检测证据不足的能力。

## 研究启发与可借鉴点
1. **来源控制（Provenance Control）作为基准设计核心**：MHER 严格审计上下文是否与提及一一对应，防止数据泄漏，这一原则可迁移至任何依赖外部证据的 NLP 任务。
2. **证伪对照实验设计**：使用确定性打乱上下文作为 informed misgrounding 压力测试，能有效区分模型对来源对齐的依赖与对泛文本模式的利用。
3. **弃权机制的显式评估**：将 AMBIGUOUS 作为独立动作并报告 abstention rate 与 selective accuracy，有助于区分“能力不足”与“策略保守”，对高风险决策任务具有参考价值。
4. **多源融合的单调性假设需验证**：Qwen3-8B 的名称干扰现象提示，在已有强上下文时额外输入名称可能引入错误先验，后续研究需在多模态/多源融合中检验各特征的边际增益。

## 关键术语表
**Historical Entity Reconciliation**：针对历史人物姓名的成对身份消解任务，判断两个来源提及是否指向同一历史个体，不依赖预定义知识库。
**Source-grounded**：模型输入中的历史证据（如生平描述）与特定的提及 × 来源记录严格绑定，确保证据具有可追溯的来源依据。
**Abstention (AMBIGUOUS)**：评估时允许模型选择弃权，表示当前证据不足以做出 SAME/DIFFERENT 判断，区别于错误分类。
**Shuffled context**：将测试集中的上下文与提及错误匹配的对照组，用于检验模型对来源对齐的敏感性，同时保持上下文池的统计特性不变。
**Provenance**：证据的来源追溯，在 MHER 中指历史描述必须源自支持该提及的具体文献或记录，而非泛化或复制的实体摘要。
**Surface-form interference**：指在已有充分上下文证据时，恢复原始名称表面形式反而导致模型错误合并不同历史人物的现象。

## 可复现要素
- **数据集**：MHER，当前 arXiv 版本未公开，计划于论文发表后发布（受限于来源重新分发限制）。
- **代码/权重**：冻结的项目状态保留执行记录与评估脚本，但未公开；Qwen3-8B 使用固定量化本地部署。
- **关键超参**：未明确列出，但强调 prompt、评估规则、阈值及解码参数均已冻结并在 DEV 上选定。
