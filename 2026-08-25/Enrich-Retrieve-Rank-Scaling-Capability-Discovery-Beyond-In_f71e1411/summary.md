---
title: "Enrich-Retrieve-Rank-Scaling-Capability-Discovery-Beyond-In"
source: https://arxiv.org/pdf/2608.22695v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:50:01"
field: "多Agent系统中的工具/能力检索与路由"
keywords: ["capability discovery", "tool retrieval", "retrieval-augmented LLM", "agent routing", "information retrieval", "in-context learning"]
innovations: ["将大规模Agent能力发现重构为信息检索流水线（Enrich-Retrieve-Rank），解耦检索与重排", "系统绘制N=10→7,278全程退化曲线，定位in-context路由与retrieve-then-rank交叉点N≈500", "离线丰富稀疏元数据，在name-only场景下Match@1提升25.6pp；生产四信号组合提升4.5pp"]
benchmarks: ["ToolRet", "AppWorld", "MCP"]
---

# 论文速读：Enrich-Retrieve-Rank: Scaling Capability Discovery Beyond In-Context Routing

## 一句话总结
论文提出 **Enrich-Retrieve-Rank** 流水线，将大规模 Agent 生态中的能力发现（capability discovery）重构为信息检索问题：离线将稀疏元数据丰富为结构化配置，在线通过 BM25 + 稠密检索召回 top-k 候选，再用单次 LLM 调用重排，实现**不在线调用任何候选**的 ranked shortlist 输出。在 ToolRet（7,278 工具、7,961 查询）上 Match@1 达 **0.397**，超越 Search&Pick 6.5 pp，且成本仅为 Full-Ctx 的 **1/70**。

## 研究问题与动机
1. **规模扩展下的路由崩溃**：现代 Agent 系统的 MATS（Models, Agents, Tools, Skills）组件注册表已达数千规模，in-context routing（将注册表片段放入 prompt 让 LLM 选择）随 N 增大性能急剧下降，无法扩展。
2. **检索与重排的解耦价值未量化**：已有工作（ToolRet、AnyTool 等）指出工具选择是主要失败模式，但未分解检索召回率与重排条件准确率，也未系统扫描 N=10→7,278 的全程退化曲线。
3. **稀疏元数据的利用不足**：公开 benchmark 中的注册表元数据质量较高，但生产环境中大量 MATS 条目仅含名称或单句描述，缺乏结构化表达；现有方法未系统研究离线丰富（enrichment）在稀疏场景下的收益。
4. **成本-精度权衡缺失**：in-context 方法（Full-Ctx、Search&Pick、Trial&Err）将 token 消耗与精度强耦合，缺乏在不调用候选的前提下同时优化精度与成本的系统性评测。

## 核心贡献（创新点）
1. **提出 Enrich-Retrieve-Rank 三阶段流水线**：将能力发现形式化为信息检索问题，离线丰富稀疏元数据 → 在线双路召回 → 单次 LLM 重排，与 Full-Ctx/Search&Pick 等 in-context 基线解耦精度与成本。
2. **首次系统绘制 N=10→7,278 的全程退化曲线并定位交叉点**：揭示 in-context routing 在 N≈500 处崩盘（Match@1 0.85→0.12），而 retrieve-then-rank 平缓退化（0.81→0.39），交叉点约在 N=500（Nova Micro 设定下）。
3. **分解检索召回率与重排条件准确率**：发现固定 k=15 的重排器条件准确率稳定在 0.70–0.87，大型注册表中约 **70% 的失败发生在第一阶段检索**，为后续改进指明方向。
4. **跨类型零调参通用配置**：同一套丰富模板、检索栈和评分权重（BM25:BGE:LLM:Quality:Intent 固定比例）直接应用于 Tool、Agent、Skill 三类注册表，无需按类型微调。
5. **生产部署实证**：流水线已作为大规模多 Agent 平台（覆盖 4 类 MATS）的默认能力发现层上线，在内部注册表上 Quality+Intent 信号带来 **+4.5 pp Match@1**（p=0.031）。

## 方法详解
**3.1 问题形式化**
给定查询 $q$ 和能力注册表 $C = \{c_1, \ldots, c_N\}$，在不在线调用任何 $c_i$ 的前提下，输出有序列表 $\{(c, s_c)\}$，供 orchestrator 选取 top-k。

**3.2 离线丰富（Offline Enrichment）**
在注册时（每条能力调用一次 LLM，非每次查询）将稀疏元数据改写为结构化 profile，包含五个类型化字段：
- **Capability Summary**：能力摘要
- **Action-verb-led Description**：以行为动词开头的描述
- **Differentiating Keywords**：区分性关键词
- **Positive Usage Examples**：正面示例
- **Negative Usage Examples**：负面示例
- 生产环境中额外添加 **Trust Score**（可信度分数）和 **Self-reported Capability Tags**（类型标签）

这些字段同时喂给 BM25 索引（词法）和 BGE/Titan 稠密编码器（语义），重排器直接读取原文评分。与 HyDE/doc2query 不同，丰富工作在注册时一次性完成，而非查询时生成。

**3.3 在线检索-重排（Online Retrieve-then-Rank）**
- **召回阶段**：使用 BM25、BGE-large-en-v1.5 或 Amazon Titan Embed V2（1024d）分别召回 top-k（k=15 或 25），混合检索时取并集。
- **重排阶段**：单次 Nova Micro API 调用，融合四个归一化至 [0,1] 的分数：
  - **LLM 分数**（权重 0.50）：单次 LLM call 产生的 listwise 重排分
  - **BM25 分数**（权重 0.05）：词法匹配分
  - **Quality 分数**（权重 0.30）：离线 trust score 归一化（生产专用）
  - **Intent 分数**（权重 0.15）：查询推断类型与候选 self-reported type tag 匹配（生产专用）
- 缺失字段时对应信号自动 drop，剩余权重 renormalize。公开 benchmark 仅使用 LLM+BM25（固定 10:1 比例）。

## 实验与结果
**数据集与基线**
- **ToolRet**：N=7,278 工具，7,961 查询（最大规模）
- **AppWorld (Tools)**：N=332，147 查询
- **AppWorld (Agents)**：N=8，147 查询
- **MCP Skills**：N=859，1,627 查询
- 基线：Regex、Full-Ctx（全注册表 in-context）、Search&Pick（LLM+搜索工具）、Trial&Err（迭代试错）、BM25-only、BGE-only

**主要结果（Match@1，Table 2）**

| 数据集 | Full-Ctx | Search&Pick | Ours+BM25 | Ours+Titan |
|--------|----------|-------------|-----------|------------|
| Tools-ToolRet (7,278) | 0.166 | **0.332** | 0.388 | **0.397** |
| Tools-AppWorld (332) | **0.762** | 0.476 | 0.741 | 0.741 |
| Agents-ToolRet (5,885) | 0.148 | 0.341 | **0.418** | 0.397 |
| Agents-AppWorld (8) | 0.986 | 0.973 | 1.000 | 1.000 |
| Skills-MCP (859) | **0.968** | 0.829 | 0.807 | **0.942** |

**关键结论**
- **最强结果**：Tools-ToolRet 上 Ours+Titan Match@1=**0.397**，领先 Search&Pick **+6.5 pp**，Token 消耗约为其 **1/2**，成本约为 Full-Ctx 的 **1/70**（$0.066 vs $4.48 / 1k queries）。
- **规模退化**：N=10→7,278，Full-Ctx Match@1 从 0.85 暴跌至 0.12；Pipeline 从 0.81 缓降至 0.39；交叉点 **N≈500**。
- **检索瓶颈**：~70% 失败发生在召回阶段（BM25 top-15 未覆盖 GT），重排器条件准确率稳定在 0.70–0.87。
- **Enrichment 效果**：在高质量公开数据上中性/轻微负向（ToolRet ±0.1 pp，MCP −4.4 pp）；在受控稀疏测试中（从完整描述退化至 name-only），Match@1 提升 **+5.8 / +9.1 / +25.6 pp**（p<10⁻⁶）。
- **跨模型鲁棒**：Nova Micro > Nova Lite ≈ Claude 3.5 Haiku（截断问题）< Claude Sonnet 4；Qwen3 Next/Mistral Large 下 Pipeline 仍全面超越 Search&Pick（+5.0–7.7 pp）。

## 相关工作脉络
1. **Tool Retrieval（ToolRet、CRAFT、EasyTool、PLUTO）**：ToolRet 证明通用 IR 模型在工具检索上表现差；CRAFT/EasyTool 做 API 文档改写；PLUTO 最接近本文思路（规划+描述编辑），但未量化检索-重排分解，也未研究规模效应。
2. **AnyTool / Meta-Tool**：AnyTool 用层次化 Agent 处理 16k API，但未分离检索与 MATS 调用；Meta-Tool 确认工具选择是主要失败模式，但未提出 retrieve-then-rank 管线。
3. **In-context Routing（AppWorld、HuggingGPT、Visual ChatGPT）**：这些方法假设注册表可放入上下文（N≤8–500），本文 Nova Micro sweep 显示 N≈500 后全崩溃，明确划清了边界。
4. **Dense Retrieval（BGE、Titan、E5-Mistral、GTE）**：本文使用 BGE-large-en-v1.5 和 Amazon Titan Embed V2 作为稠密召回器，E5-Mistral/GTE 留作未来工作；ToolBench-IR（BERT-base 微调）未带来端到端提升。
5. **Listwise Reranking（RankGPT、Tool-DE、Multi-Field Tool Retrieval）**：RankGPT 证明 LLM listwise 重排接近监督 cross-encoder；Tool-DE 和 Multi-Field Tool Retrieval 同期独立提出 enrich-then-retrieve，但未研究规模变化下的性能曲线。
6. **Capability Discovery 标准（MCP、Agent2Agent、GPT Store）**：这些协议标准化了枚举和调用，但假设客户端已知道目标 server，本文解决"找到哪个 server/tool"的预路由问题。

## 局限性与未来方向
1. **Agent/Skill 跨类型结论受限**：Agents-ToolRet 与 Tools-ToolRet 共享 89% 注册表和查询集，非独立证据；Skills-MCP 查询本质上是描述的 paraphrase，Lexical retriever 占优；Agent 和 Skill 的独立大规模 benchmark（N>500，真实用户意图，multi-ground-truth 轨迹标注）尚未存在。
2. **Enrichment 在高质量注册表上无效**：公开 benchmark 元数据已较完善，丰富步骤收益中性甚至负向（MCP 上 −4.4 pp，因关键词稀释了已有强词法重叠）；稀疏元数据的增益需人工构造退化实验验证，未反映真实分布。
3. **检索召回仍是瓶颈**：Titan Embed V2 Recall@15=0.625（Tools-ToolRet），距饱和仍有差距；现有领域微调检索器（ToolBench-IR）未能突破；未来需更强的一阶段检索器。
4. **Model Routing 未覆盖**：生产注册表将模型与 Tool/Agent/Skill 统一索引，但实验未报告模型检索结果，依赖现有 model-routing 文献。
5. **成本估算偏后验**：Pipeline 行的 token/cost/latency 为 post-hoc 估计（±5–10%），非实测，与 baseline 的 real 数据存在不对称。

## 研究启发与可借鉴点
1. **检索-重排分解是规模扩展的通用范式**：本文的核心洞察——将"单次大选择"拆解为"召回+重排"两个解耦阶段，且重排器条件准确率对规模不敏感——可直接迁移到文档检索、代码搜索、RAG 系统等任何候选集随规模增长的场景。
2. **离线丰富（offline enrichment）在稀疏元数据场景下价值显著**：在注册时一次性用 LLM 将名称+单句描述扩展为五字段结构化 profile，可使 name-only 场景 Recall@15 从 0.134 提升至 0.467（+25.6 pp）；这对内部系统、新兴 API 平台、开源工具库注册表极具参考价值。
3. **Quality/Intent 信号（trust score + type tag 匹配）是生产级增量**：在具有信任分层和类型标注的内部注册表上，四信号组合比两信号提升 +4.5 pp；可借鉴到多租户 Agent 平台、企业级 tool catalog 建设中。
4. **跨模型鲁棒性验证值得复用**：本文用 4 种 reranker（Nova Micro/Lite、Claude 3.5 Haiku/Sonnet 4）和 3 种 LLM（Nova Micro、Qwen3 Next、Mistral Large 3）交叉验证，确认 pipeline 优势不依赖单一模型，可作为方法论模板。
5. **规模扫描（size sweep）是评估路由方法的必要实验**：仅报告单一规模下的数字（如 N=8 或 N=16k）可能误导结论；本文 N=10→7,278 的 sweep 清晰刻画了 crossover 点，建议后续工作强制包含规模变量。

## 关键术语表
**MATS**：Models, Agents, Tools, Skills 四类组件的统称，本文统一 schema 索引的核心对象。
**In-context Routing**：将注册表片段（名称/描述）放入 LLM prompt，由 LLM 直接选择候选的传统方式，随规模扩大性能急剧退化。
**Match@k**：主评估指标，查询的前 k 个候选中包含至少一个 ground-truth 能力的比例。
**MRR（Mean Reciprocal Rank）**：第一个正确候选排名的倒数均值，衡量排序质量。
**Enrichment（离线丰富）**：在注册时调用 LLM 将稀疏元数据改写为五字段结构化 profile（summary、action description、keywords、正负示例），一次性完成，非查询时生成。
**Conditional Accuracy（条件准确率）**：在检索器已召回正确候选的前提下，重排器将其排在首位的概率，本文稳定在 0.70–0.87。
**HyDE（Hypothetical Document Embedding）**：查询时生成假设文档再检索的方法，本文与之对比，强调离线丰富（注册时一次性）优于查询时生成。
**Recall@k**：前 k 个候选中覆盖的 ground-truth 能力比例，适用于 one-to-many 查询场景。

## 可复现要素
- **数据集**：ToolRet（公开，7,278 tools）、AppWorld（公开，332 tools / 8 agents）、MCP Skills（公开，859 skills）；内部基准（N=221，未公开）
- **代码/权重**：论文未提供开源链接；使用组件为开源模型（BGE-large-en-v1.5、Amazon Titan Embed V2、Nova Micro/Claude 系列）
- **关键超参**：k=15（生产）、k=25（实验）；BM25:LLM:Quality:Intent 权重 0.05:0.50:0.30:0.15（公开数据 renormalized 为 0.1:0.9）；Nova Micro 上下文预算 480k characters（Full-Ctx 截断阈值）
