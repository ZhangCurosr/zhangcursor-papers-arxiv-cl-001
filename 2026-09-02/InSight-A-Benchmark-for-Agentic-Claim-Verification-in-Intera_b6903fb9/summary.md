---
title: "InSight-A-Benchmark-for-Agentic-Claim-Verification-in-Intera"
source: https://arxiv.org/pdf/2609.01383v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:27:51"
field: "多模态代理与可视化理解"
keywords: ["interactive visualization", "claim verification", "vision-language model", "agentic benchmark", "fact-checking", "interaction efficiency score", "Vega-Lite"]
innovations: ["提出交互式可视化声明验证任务，将事实核查的认识论要求扩展至代理式多模态环境", "构建 InSight 基准（21,349 条声明），源自人类专家分析笔记本并通过 NLI 验证的受控变异生成 False/NEI 声明", "引入 IES 指标将推理质量评估从准确率扩展到交互效率，惩罚无 grounding 的正确回答"]
benchmarks: ["InSight"]
---

# 论文速读：InSight-A-Benchmark-for-Agentic-Claim-Verification-in-Intera

## 一句话总结
本文提出了 **InSight**，一个用于交互式可视化中代理式声明验证的基准测试，包含 21,349 个源自人类作者分析笔记本的真实声明；要求 VLM 代理通过主动导航、悬停、点击等交互行为来获取分布在不同视图中的视觉证据，判断声明为 True/False/NEI，并引入 IES 指标评估交互推理质量。

## 研究问题与动机
- **静态评估的生态效度缺陷**：现有 VLM 图表理解基准（如 ChartQA、PlotQA）均为静态图像 + 一次性问答，无法反映真实数据分析中证据被遮挡、跨视图分布、需主动探索的交互式场景。
- **缺乏对代理式证据获取能力的评估**：现有 fact-checking 数据集（FEVER、NewsCLIPpings）依赖静态多模态输入，无法捕捉模型如何主动搜寻、筛选和综合视觉证据的推理过程。
- **交互轨迹作为推理代理的必要性**：在传统 QA 范式中模型推理过程不透明，而 InSight 将交互动作序列（hover、click、scroll）作为显式的 reasoning proxy，可审计模型是否真正接触到了支持/反驳证据。
- **NEI 标注的现实需求**：真实分析场景中大量声明因数据缺失而无法验证，现有基准缺少对"Not Enough Information"这一认识论状态的显式建模。

## 核心贡献（创新点）
1. **提出交互式可视化声明验证任务**：将事实核查的认识论要求扩展至代理式多模态环境，要求模型在部分可观察的交互式 Vega-Lite 环境中通过主动交互获取证据。
2. **构建 InSight 基准数据集**：源自 297 本人类专家撰写的分析笔记本，经四阶段流水线（片段提取→声明分解→受控变异→人工验证）生成 21,349 条声明（True/False/NEI），相比合成数据更具生态效度。
3. **引入 IES（Interaction Efficiency Score）指标**：将答案准确性与有效交互比例相乘，惩罚仅凭参数知识"猜对"的代理，迫使评估关注推理质量而非单一准确率。
4. **系统性评估 SOTA VLM 在交互验证中的表现**：揭示交互验证仍是难题（最佳 GPT-5.5 仅 57.2% 准确率），并发现 falsification 比 verification 更难，小模型普遍缺乏主动探索能力。
5. **提供交互轨迹的行为分析框架**：分析不同模型的探索策略差异（如 Gemini 广泛探索 vs. GPT-5.5 经济高效），暴露 premature commitment、unfocused exploration 等系统性失败模式。

## 方法详解
**数据集构建流水线（四阶段）：**
- **Stage 1 Span Extraction**：采用 Lundgard & Satyanarayan 四层语义模型，仅保留 Level 2（统计洞察）和 Level 3（视觉/感知洞察）的声明片段；对每段文本进行 3 次独立 LLM 提取（Gemini 2.5 Flash），通过 trigram overlap 多阶段一致性程序整合，仅保留 ≥2 次提取的 span。
- **Stage 2 Claim Decomposition**：将复合 span 分解为原子化、去上下文化的可验证命题；通过 self-verification（Weng et al., 2023）确认分解声明被原文蕴含；语义标签经 3 轮多数投票分配。
- **Stage 3 Claim Mutation**：对 TRUE 声明进行三种受控变异生成 FALSE/NEI：① **Antonym substitution**（反义词替换，如 increased→decreased）→ FALSE；② **In-lexicon argument substitution**（同数据集参数替换）→ FALSE；③ **Out-of-lexicon argument substitution**（替换为数据集中不存在参数）→ NEI；所有变异均通过 NLI 模型（DeBERTaV3）验证语义方向（FALSE 双向 contradiction > 0.9，NEI entailment/contradiction < 0.2）。
- **Stage 4 Human Validation**：13 位专家标注员盲评 475 条声明，与数据集标签原始一致率 81.3%；Dawid-Skene 贝叶斯模型恢复的分类比例与地面真实值误差 < 3pp，MAP 准确率 77.6%。

**评估环境设置：**
- 使用 Playwright + Chromium headless browser 渲染 Vega-Lite 交互式网页，固定视口小于实际页面尺寸，强制模型通过 scroll 发现多视图。
- 仅提供 RGB 截图，**无 DOM/HTML/accessibility tree**，迫使模型依赖视觉渲染进行推理。

**Agent 动作空间：**
- Navigation：`scroll(dx,dy)`、`page_up(n)`、`arrow_right(n)` 等
- Mouse Interaction：`click(x,y)`、`hover(x,y)`、`drag((x1,y1),(x2,y2))`、`shift_click(x,y)`
- Termination：`answer(label)`，label ∈ {True, False, NEI}

**IES 指标设计：**
$$E_i = \sum_{t=1}^{T_i} \mathbf{1}[s_{t+1} \neq s_t]$$
$$\mathrm{IES}_i = A_i \cdot \frac{E_i}{T_i}$$
其中 $A_i$ 为答案正确性（0/1），$E_i/T_i$ 为状态改变比率（SCR）。**首次交互即答对的 Episode 强制 IES=0**，因为初始截图无法验证任何声明，正确回答只能归因于参数知识或猜测。

## 实验与结果
**数据集统计：** 21,349 条声明，41.6% True / 45.0% False / 13.4% NEI；平均长度 18 tokens；56.8% 为 Level 2（统计/数值），43.2% 为 Level 3（视觉/感知）；每个 notebook 平均含 4.5 个视图、3.7 个可视化规范。

**评估子集：** 500 条声明（167 True / 167 False / 166 NEI），交互预算 $T_{max}=10$。

| 模型 | 准确率 | Acc_False | Acc_NEI | Acc_True | E/T | IES |
|------|--------|-----------|---------|----------|-----|-----|
| **GPT-5.5** | **57.2%** | 59.6% | 59.0% | 53.0% | 45.4% | **26.98%** |
| Gemini 3.5 Flash (T=25) | 50.0% | 54.8% | 50.6% | 44.6% | 56.7% | 31.0% |
| Gemma 4 31B | 47.2% | 40.4% | 63.9% | 37.5% | 43.8% | 20.57% |
| Qwen 3.5 27B | 45.0% | 38.0% | 66.3% | 31.0% | 17.7% | 7.57% |
| Qwen 3.5 0.8B | 34.0% | 21.7% | 54.8% | 25.6% | 0.0% | 0.00% |

**关键发现：**
- **GPT-5.5 综合最优**：57.2% 准确率 + 26.98% IES，交互行为最为经济高效（平均仅 3.13 步）。
- **Gemini 3.5 Flash 呈非单调趋势**：$T_{max}=1$ 时 44.2% → $T=10$ 时 41.6%（被中途强制终止拖累）→ $T=25$ 时 50.0%（充分探索后提升）。
- **小模型失效严重**：Gemma 4 E2B（31.2%）、Qwen 3.5 0.8B（34.0%）接近随机基线（33.3%），IES 趋近 0。
- **NEI 易、False 难**：多数模型在 NEI 上 IES 最高，而在 False（falsification）上最低——需要主动寻找反驳证据，认知负荷最大。
- **Hover 是核心交互**：InSight 中最常见的证据揭示方式是 hover tooltip，但小模型普遍缺失 hover 动作。

## 相关工作脉络
1. **Agentic VLM 基准**（OSWorld、MiniWoB++、VisualWebArena、Browser-Gym）：聚焦通用任务完成，成功定义为是否到达目标状态；InSight 聚焦单一严格任务（声明验证），将交互轨迹作为推理过程本身进行评估。
2. **图表理解 QA 基准**（FigureQA、DVQA、PlotQA、ChartQA）：均为静态图像 + 一次性 QA，依赖合成或模板化图表；InSight 采用真实分析师撰写的 Vega-Lite 交互式可视化，证据分布多视图且条件可见。
3. **图表声明验证基准**（ChartCheck）：虽引入 claim verification 范式，但仍基于静态图像和为基准专门构造的声明；InSight 的声明源自人类分析笔记本，保留真实分析工作流的上下文。
4. **文本事实核查基准**（FEVER、SciFact、FEVEROUS）：引入 NEI 标签和证据检索框架；InSight 将其扩展至交互式视觉环境，证据需通过主动交互而非静态检索获取。
5. **多模态事实核查基准**（NewsCLIPpings、AVerImaTeC）：评估图文一致性，但为 closed-world 静态输入；InSight 支持 open-world 探索和部分可观察环境。
6. **GUI 代理基准**：多数基于 DOM/accessibility tree 作为输入；InSight 刻意仅用 RGB 截图，迫使模型依赖视觉 grounding 而非结构化解析。

## 局限性与未来方向
- **动作空间抽象**：固定高层动作集合（click/hover/scroll）无法捕捉复杂手势或语义快捷操作，可能限制交互策略的多样性评估。
- **交互轨迹 ≠ 完整推理证据**：模型可能执行正确动作但出于错误理由，或内部推理合理但交互轨迹无法捕捉；轨迹应视为补充证据而非决定性证据。
- **IES 无法完全排除"正确但未 grounding"**：状态改变动作即使未用于最终推理也会被计入有效交互，未来需引入声明级证据标注以精细化评估。
- **未进行大规模 fine-tuning 评估**：当前仅评测 closed/open-weight 预训练模型行为，交互验证可能需要专门的训练信号（如 trajectory supervised fine-tuning）。
- **数据集规模与领域覆盖**：297 个笔记本虽经人工审核高质量，但领域多样性有限；可扩展至更多学科和分析场景。

## 研究启发与可借鉴点
1. **IES 评估范式可迁移**：将"交互效率"纳入评估的设计思路适用于任何需要主动证据获取的多模态任务（如科学文献挖掘、医疗影像诊断代理），防止模型"走过场式探索"。
2. **NLI 验证 + 受控变异的声明构造流水线**：反义词替换（生成 FALSE）与词外参数替换（生成 NEI）相结合，并通过 NLI 阈值过滤，这一 pipeline 可复用于其他 claim verification 数据集构建。
3. **"初始视图不可解"的设计原则**：强制要求模型必须交互才能作答，避免了参数知识泄漏导致的虚假高分，这一设计对交互式代理基准具有普遍参考价值。
4. **交互动作类型与性能关联分析**：论文揭示小模型缺失 hover 动作导致无法访问 tooltip 证据，提示未来工作中应将 action profile 分析纳入模型能力诊断。
5. **与视觉分析交叉**：InSight 的数据源自可视化分析师笔记本，可与 VisIA（可视化交互分析）社区合作，将用户行为日志（真实分析师如何探索）作为 agent 探索策略的 ground truth 对比基准。

## 关键术语表
**Interactive Claim Verification**：要求代理在交互式可视化环境中主动获取证据，判断自然语言声明为 True/False/NEI 的任务。
**IES (Interaction Efficiency Score)**：准确率 × 有效交互比例的综合指标，惩罚无交互或低效交互的正确回答。
**NEI (Not Enough Information)**：视觉证据不足以支持或反驳声明的认识论标签，体现真实场景中的信息不完全性。
**Vega-Lite**： declarative visualization grammar，InSight 中用于构建交互式多视图环境的核心技术栈。
**Semantic Level 2/3**：Lundgard 语义模型中 Level 2 为统计数据洞察（数值/极值），Level 3 为视觉感知洞察（趋势/模式）。
**Antonym/Argument Substitution**：通过反义词替换或参数替换生成 FALSE/NEI 声明的受控变异方法。
**Effective Action**：导致环境视觉状态发生可观测变化的交互动作，用于计算 IES。
**Partially Observable Environment**：环境信息分处于不同视图或通过交互条件揭示，初始截图无法获取完整证据的场景设定。

## 可复现要素
- **数据集**：InSight 已公开，地址 https://github.com/maevehutch/insight
- **代码**：论文未提供统一代码仓库，但注明完整实现细节在 Appendix A
- **评估环境**：Playwright + Chromium headless browser，固定视口
- **交互预算**：主实验 $T_{max}=10$，消融实验 $T_{max} \in \{1, 10, 25\}$
- **NLI 验证模型**：DeBERTaV3 (He et al., 2023)
- **Span Extraction 模型**：Gemini 2.5 Flash（三次独立运行）
- **人工标注**：13 位专家标注员，Dawid-Skene 贝叶斯模型
