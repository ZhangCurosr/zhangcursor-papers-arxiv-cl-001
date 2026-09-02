---
title: "Enrich-Retrieve-Rank-Scaling-Capability-Discovery-Beyond-In"
source: https://arxiv.org/pdf/2608.22695v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:49:39"
field: "多Agent系统中的能力检索与路由"
keywords: ["capability discovery", "tool retrieval", "retrieve-then-rank", "agent routing", "LLM reranking", "scalable registry"]
innovations: ["检索-排序解耦流水线架构，固定k排序器条件准确率稳定在0.70-0.87", "全规模（N=10至7,278）性能退化曲线刻画，定位N=500交叉点", "单一配置跨Tool/Agent/Skill三类注册表通用，已在生产部署"]
benchmarks: ["ToolRet (7,278 tools / 5,885 agents)", "AppWorld (332 tools / 8 agents)", "MCP Skills (859 skills)"]
---

# 论文速读：Enrich-Retrieve-Rank: Scaling Capability Discovery Beyond In-Context Routing

## 一句话总结
本文提出了一种离线丰富化+在线检索-排序流水线，将多Agent系统中的能力发现从上下文路由重构为信息检索问题；在7,278个能力的注册表规模下，相比现有最优上下文基线Search&Pick提升6.5个百分点Match@1，同时将成本降低约70倍。

## 研究问题与动机
1. **规模扩展导致上下文路由失效**：现有Agent系统依赖LLM在上下文中阅读注册表并直接选择候选能力（in-context routing），但当注册表从N=10扩大到N=7,278时，Match@1从0.85骤降至0.12。
2. **成本与精度耦合**：每次错误调用不仅消耗token和延迟，还可能调用不可信端点来"试探"其功能，且全量上下文输入的成本随规模线性增长。
3. **缺乏阶段分解认知**：既往工作仅在单一规模报告退化，未分解检索召回率与排序条件准确率各自的责任占比。
4. **生产级跨类型统一发现的缺失**：现有方法多为特定类型或特定规模设计，缺少一套统一配置可同时在Agent/Tool/Skill注册表中工作的规模化方案。

## 核心贡献（创新点）
1. **检索-排序解耦架构**：将能力发现拆分为离线丰富化→在线检索→LLM排序三阶段流水线，使固定k的排序器能稳定保持0.70–0.87的条件准确率，而不像上下文路由那样随规模坍塌。
2. **全规模退化曲线与交叉点定位**：首次在N=10至7,278的完整范围内刻画性能曲线，定位出Nova Micro上下文大小的交叉点约为N=500，揭示了检索阶段是大规模miss的主要来源（约70%）。
3. **单一配置跨类型通用**：同一套丰富化模板、检索栈和打分权重适用于Tool、Agent、Skill三类注册表，无需按类型调参。
4. **生产部署验证**：已在AWS Lambda上以Serverless形态部署为某大型多Agent平台的默认能力发现层，并实测了真实成本、延迟和精度。

## 方法详解
- **离线丰富化（Enrichment）**：在能力注册时，由LLM将稀疏元数据改写为五字段结构化档案——能力摘要（summary）、动作动词开头的描述（action-led description）、区分性关键词（differentiating keywords）、正例和反例用法（positive/negative examples）。同时附加trust score和self-reported capability tags（生产元数据提供时）。该过程一次性执行，不在查询时重复。
- **在线检索（Retrieve）**：采用BM25词汇检索、BGE-large-en-v1.5或Amazon Titan Embed V2密集检索，或两者混合；从丰富化后的索引中返回Top-k（生产取k=15）候选列表。
- **在线排序（Rank）**：单次LLM调用对k个候选进行列表式（listwise）重排，综合四类信号得分归一化至[0,1]：LLM评分（权重0.50）、BM25（0.05）、Quality信任分（0.30）、Intent类型匹配（0.15）。缺少某类字段时自动降权并重新归一化（公共基准仅使用LLM+BM25，固定10:1比例）。

## 实验与结果
- **数据集**：ToolRet（7,278工具 / 5,885 Agent，7,961查询）、AppWorld（332工具 / 8 Agent，147查询）、MCP Skills（859技能，1,627查询）。
- **最强结果（Tools-ToolRet, N=7,278）**：Ours+Titan达到Match@1=0.397，领先Search&Pick（0.332）+6.5 pp；在MRR上达0.467，Recall@15达0.625。
- **规模交叉点**：Nova Micro下交叉点约N=500；N<500时Full-Ctx仍占优（如Tools-AppWorld中0.762 vs 0.741）。
- **成本对比**：Ours+BM25每千查询成本$0.066，约为Search&Pick（$0.117）的一半，约为Full-Ctx（$4.48）的1/70；p50延迟约1,100–1,500ms。
- **丰富化消融**：在元数据稀疏的受控退化测试（全描述→首句提示→仅名称）中，Match@1分别提升+5.8、+9.1、+25.6 pp；在完整公开数据上增益微乎其微（±0.1 pp），说明丰富化价值与元数据稀疏度正相关。
- **检索器分析**：BGE/Titan相比BM25提升Recall@15达+4.7–7.6 pp，但端到端Match@1因排序器条件准确率饱和而未显著放大（差距<1 pp）。
- **跨模型稳健性**：Claude Sonnet 4 > Nova Micro ≈ Nova Lite ≫ Claude 3.5 Haiku；Claude 3.5 Haiku因输出截断在满规模下崩溃（0.496→0.280）。

## 相关工作脉络
1. **ToolRetrieval（CRAFT/EasyTool/PLUTO）**：类似方向做丰富化，但未量化检索vs排序的分解，也未研究规模扩展效应；本文明确分离两阶段并给出分解归因。
2. **AnyTool / Meta-Tool**：指出工具选择是主要失败模式，但未提出检索-排序分离架构；本文将其形式化为可工程化的IR流水线。
3. **doc2query / HyDE**：早期文档扩展/假设文档方法，本文借鉴思路但改为注册时一次性丰富化，且不依赖查询时生成。
4. **RankGPT**：证明列表式LLM重排器可比肩监督交叉编码器；本文复用该范式并验证其在能力发现场景的普适性。
5. **ToolBench-IR**：领域微调的BERT检索器，本文测试后发现其Recall@15低于通用密集编码器，说明领域微调未能突破检索瓶颈。
6. **Knowledge-graph approach (Mulang' et al., 2026)**：知识图谱过滤在269工具上有效，但在更大规模或未结构化菜单上效果未验证；本文强调通用IR方案的扩展性。

## 局限性与未来方向
1. **丰富化在干净数据上无效甚至有害**：公共基准元数据已很完善，丰富化增益为零或负（MCP上-4.4 pp因关键词稀释）。
2. **Agent和Skill基准规模不足**：当前最大Agent/Skill注册表（8/859）远未达到交叉点，无法独立验证跨类型扩展结论。
3. **检索是最终瓶颈**：剩余gap需要更强的一阶段检索器，现有领域微调（ToolBench-IR）未能解决问题。
4. **未报告Model作为能力类型的检索结果**：生产注册表含Model，但论文未给出该类型的独立评测数据。
5. **未来方向**：探索更强的一阶段检索器、在大规模真实Agent/Skill注册表上验证、以及结合多ground-truth轨迹标签的评估。

## 研究启发与可借鉴点
1. **检索-排序分解思维**：将端到端性能分解为召回率和条件准确率，有助于识别真正的瓶颈并指导资源分配（本文锁定检索而非排序）。
2. **离线丰富化适配稀疏场景**：在元数据稀疏的生产注册表中，一次性LLM改写价值显著；可作为通用ETL策略迁移至其他"稀疏描述→可检索档案"的场景。
3. **固定k排序器的成本-精度trade-off**：通过限制排序器视野（k=15）保证条件准确率稳定，换取大规模下的可扩展性；适用于任何候选集远超上下文预算的排序任务。
4. **单配置跨类型泛化**：用统一模板+同权重处理不同实体类型，减少了调参复杂度，可为多模态/多类型系统集成提供参考。
5. **生产部署的完整指标体系**：同时报告Match@k、MRR、Recall、token数、成本和延迟，为工业界评估RAG/检索类系统提供了可复用的度量框架。

## 关键术语表
**In-context routing**：LLM直接在prompt上下文中阅读注册表并从中选择候选能力的传统路由方式。

**Match@k**：查询结果中任意一个ground-truth能力出现在Top-k列表中的比例，衡量候选列表的"命中"质量。

**Retrieve-then-rank pipeline**：先通过检索器召回Top-k候选，再由排序器重排的流水线架构，与单阶段全量选择相对。

**Offline enrichment**：在注册时一次性将稀疏元数据改写为结构化丰富档案的过程，不在查询时重复执行。

**BM25**：基于词频的经典词汇检索排序函数，本文作为lexical baseline使用。

**BGE-large-en-v1.5**：大规模预训练的密集文本嵌入模型，本文用于生成query/capability向量进行相似度检索。

**Conditional accuracy**：给定ground-truth已进入Top-k候选的前提下，排序器将其排在首位的概率。

**Crossover point**：两种方法性能相等的规模阈值，本文测定约为N=500。

## 可复现要素
- **数据集**：ToolRet、AppWorld、MCP均为公开基准；ToolRet含7,278工具和5,885 Agent。
- **代码/权重开源**：论文未明确声明代码开源；使用了开源模型BGE-large-en-v1.5、Amazon Titan Embed V2、以及Amazon Nova Micro/Lite和Claude系列。
- **关键超参**：k=15（生产）、weights=[LLM:0.50, BM25:0.05, Quality:0.30, Intent:0.15]；丰富化模板五字段结构；检索器含BM25+BGE/Titan混合。
- **训练/微调**：无额外微调，使用预训练检索器和reranker直接推理。
