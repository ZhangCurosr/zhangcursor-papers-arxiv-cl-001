---
title: "Token-Eficient-Data-Reasoning-Agents-via-Adaptive-Structurin"
source: https://arxiv.org/pdf/2608.31082v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:46:30"
field: "Agent推理效率优化"
keywords: ["agentic reasoning", "data cracking", "token efficiency", "adaptive structuring", "unstructured data", "KV-cache reuse"]
innovations: ["将数据库cracking思想引入AI推理，查询驱动自适应提取可复用结构化知识", "推测式结构提取：cracking分支超越当前问题语义预取未来可能需要的实体/关系", "双重局部性利用：语义局部性+KV-cache共享前缀复用协同降低prefill成本"]
benchmarks: ["FanOutQA", "Cinema Case Study (Hitchcock)"]
---

# 论文速读：Token-Efficient-Data-Reasoning-Agents-via-Adaptive-Structurin

## 一句话总结
论文提出**Agentic Data Cracking**方法，借鉴数据库cracking思想，在LLM Agent推理过程中自适应地从非结构化文档中提取可复用结构，使后续相关查询可通过结构化读取替代重复文档加载，在FanOutQA上将Agent推理成本降低53%同时保持准确率。

## 研究问题与动机
- **核心问题**：LLM Agent进行数据推理（data reasoning）时，需反复打开大文档提取分散证据，单次问题可消耗高达百万tokens、成本近1美元，成本过高阻碍规模化应用。
- **预结构化不现实**：文档蕴含的实体/属性/关系数量远超任何工作负载所需，预先完整提取是无界decode且不可行；有用结构只有查询到来时才可知。
- **查询驱动的价值发现**：相关查询会重复打开相同文档，语义局部性是常态（同一用户持续追问、同一文档承载多类相关事实），工作负载本身揭示哪些结构值得提取。
- **KV-cache复用机会**：文档已被加载后其prefill已计算，cracking分支通过共享前缀KV-cache Fork生成，仅增加有限decode开销，即可提取服务于未来查询的结构化知识。

## 核心贡献（创新点）
1. **Agentic Data Cracking框架**：将数据库cracking的"查询驱动自适应组织"思想引入AI推理，但目标是从头创建而非重组已有结构。与已有工作的本质区别在于：不是先建库再查询，而是在推理过程中边用边建，由观测到的查询指导结构提取。
2. **推测式（speculative）结构提取**：Cracking子Agent不仅提取当前问题所需事实，还基于语义推理推测周边实体/属性/关系，提取服务于相关但不同查询的潜在有价值结构。区别于传统缓存（仅命中相同问题），这是跨查询的语义预取。
3. **基于RDF扩展的数据模型**：定义带基数(cardinality)、单位(unit)、证据(evidence)注解的cracked对象（`c = ⟨s, r, o, κ, u, ε⟩`），支持单值与列表关系，并通过catalogue逻辑视图帮助Agent在结构化读取前决策。
4. **双重局部性利用**：同时利用查询间语义局部性（相关查询重叠文档和实体集）和推理级KV-cache局部性（cracking复用已加载文档的prefix），实现token与API成本的显著降低。

## 方法详解
- **双模式行为设计**：
  - **Read/Reasoning Behaviour**：主Agent除原有文件系统工具外，新增受限tool-call支持从cracked-object store按subject/relation进行结构化读取；若命中则返回紧凑值+证据，跳过文档prefill。
  - **Write/Cracking Behaviour**：当结构化读取miss时，主Agent打开原始文档；文档进入上下文后，cracking sub-agent从推理分支fork，利用共享prefix KV-cache以低边际成本执行bounded decode，生成schema约束JSON。
- **数据模型**：
  - Cracked对象公式：`c = ⟨s, r, o, κ, u, ε⟩`，其中`s ∈ E`（实体）、`r ∈ R`（关系）、`o ∈ E ∪ V`（实体或标记标量）、`κ ∈ {singular, list}`（基数）、`u`（单位）、`ε = ⟨d, ρ⟩`（证据：来源文档d + 支持区域ρ）。
  - Value类型支持string、integer、date三类tagged scalar。
  - Catalogue定义：`Cat(d) ≜ π_{s,r,κ}(σ_{doc(ε)=d}(C))`，列出每文档可用的subject-relation对及其基数。
- **关键设计约束**：
  - 系统永不为提取结构而主动打开文档，cracking仅作为推理过程的副产物；
  - Cracking在答案依赖路径之外并行执行，不阻塞当前查询；
  - 固定decode budget（实验设为4K tokens/question），生成后后处理验证、归一化并展开list为独立边；
  - 严格fallback机制：结构化读取miss时回退到原始文档，保障鲁棒性。

## 实验与结果
- **数据集**：FanOutQA（Wikipedia多跳问答），以及Hitchcock电影案例研究（模拟持续调查场景）。
- **基线**：Claude-Haiku-4.5 + FanOutQA原Agent设置（RAG baselines同样使用Haiku）。
- **实验设计要点**：为模拟真实工作负载的查询局部性，对每个测试问题生成1个相关但不同属性的问题（由claude-opus-5生成、人工验证），cracked-object store完全由前置相关问题填充（warm store评估）。
- **主要结果**：
  - **Token消耗**：FanOutQA平均prefill从189K降至87K tokens；案例研究从565K降至161K tokens；decode均<3K。
  - **API成本**：FanOutQA从$0.26降至$0.12（降幅53%）；案例研究从$0.81降至$0.27。
  - **准确率**：String accuracy微降2个百分点（主要因答案规范化），LLM-judge accuracy 42% vs 43%（p=0.39，无统计显著差异）。
  - **分位改善**：中位数成本比3.4×；第10百分位成本降低9×；第90百分位成本为baseline的1.24×（无复用场景下cracking overhead）。
  - **理想上界**：人工构建的完美结构化store使推理便宜28倍（FanOutQA均7文档），gap随fan-out增大。
- **Cracking开销**：4K budget每问题约12% overhead，产生约150个cracked objects；更大budget产出更多但复用率更低的object。

## 相关工作脉络
1. **Deep-research/推理Agent**（如Synthesizing Scientific Literature [1], AgentIR [4]）：本文聚焦其高成本瓶颈，提供查询驱动的自适应结构积累机制作为互补。
2. **OpenIE/KG构建**（[6][7][28][33]）：这些方法在查询前预先构建知识库；本文与之相反，结构由工作负载按需推测提取，避免无界提取。
3. **Database Cracking**（[11]）：经典数据系统中由查询驱动物理重组；本文将其语义推广至从零创建结构，而非重组已有关系。
4. **Query-driven Document Analytics**（[14]）：仅提取当前查询所需值；本文cracking超出当前查询，推测未来相关结构。
5. **Semantic Cache**（[2]）：仅命中近似重复问题；Cracked objects跨不同问题复用，且为紧凑evidence-backed结构化数据。
6. **Agent Memory系统**（MemGPT [20], Letta [13], Mem0 [5], Zep [24], A-Mem [31], MemoryBank [35]）：此类系统记忆用户偏好与对话历史；本文memory针对语料库推测结构，并具备speculation/prefetching能力。
7. **KV-cache Serving**（[8][12][17][23][32]）：缓存状态绑定单一模型、体积庞大；Cracking读取cached prefix但产出纯文本结构，跨模型可迁移、形成持久化组织资产。

## 局限性与未来方向
- **工具接口受限**：当前仅暴露受限read函数而非完整SQL，不支持count/aggregation/join/direct multi-hop查询，安全暴露这些能力是未来方向。
- **静态语料假设**：系统假设语料静态，动态编辑需通过丢弃受影响cracked objects或增量 refine 处理，尚未实现。
- **Benchmark局限性**：当前测试使用公开训练的Wikipedia内容，企业私有文档（未见于模型训练、跨用户重复查询）是更有价值的 workload，但缺乏对应benchmark。
- **Budget分配策略简单**：4K token budget均匀分配，更智能的per-query/per-document分配仍是未来工作。
- **更激进的reuse策略**：当前设计优先fallback鲁棒性；更激进的policy可进一步降本但需权衡准确率。

## 研究启发与可借鉴点
1. **"推理即提取"范式**：将结构提取作为推理过程的副产物（byproduct）而非前置步骤，避免无界extract开销，这一思路可迁移至任何需反复读取大型非结构化数据的Agent场景。
2. **双重局部性利用**：同时挖掘语义局部性（查询间重叠实体）与系统级局部性（KV-cache prefix复用），为Agent系统的成本优化提供了可组合的优化层次框架。
3. **Catalogue作为中介视图**：通过catalogue抽象暴露可用subject-relation对，让Agent在open document前决策，这一"先探查后访问"模式可降低无效prefill。
4. **跨模型持久化资产**：Cracked objects作为纯文本结构化数据可跨模型迁移，区别于绑定模型的KV-cache，形成了可复用的组织知识资产——这一思想适用于企业级AI基础设施设计。
5. **Warm-store评估方法论**：通过前置相关问题填充store再评估测试问题的做法，为类似"自适应结构系统"的评估提供了可复现的实验协议。

## 关键术语表
- **Data Reasoning（数据推理）**：Agent从非结构化文档中提取证据、构建中间数据结构并逐层传递的推理模式，答案来源于构造的中间数据而非单次检索。
- **Agentic Data Cracking（Agent数据 cracking）**：借鉴数据库cracking思想的自适应结构提取方法，由观测到的查询驱动，在推理过程中推测并提取可复用结构。
- **Cracked Object（Cracked对象）**：由cracking sub-agent提取的结构化事实单元，扩展RDF边形式，携带基数、单位与证据注解。
- **Catalogue（目录）**：针对每文档的逻辑视图，列出可用的subject-relation对及其基数，辅助Agent决策结构化读取。
- **Prefill-intensive（Prefill密集型）**：数据推理的推理特征，agent反复加载大文档进行prefill，而最终答案通常简短。
- **Semantic Locality（语义局部性）**：相关查询重复打开相同文档、共享重叠实体集的分布特性，是cracking机制有效性的前提。
- **Shared-prefix KV-cache Reuse（共享前缀KV-cache复用）**：Cracking分支复用已加载文档的KV-cache prefix，避免重复prefill开销。
- **Speculative Structuring（推测式结构化）**：Cracking不只提取当前问题所需结构，还基于语义推理推测服务于未来相关查询的结构。

## 可复现要素
- **数据集**：FanOutQA（公开基准，见[36]）；Cinema case study（Hitchcock相关问题，论文未声明公开）。
- **代码/权重**：论文未声明代码开源；使用Claude-Haiku-4.5模型与FanOutQA官方prompts/tools。
- **关键超参**：Cracking decode budget = 4K tokens/question；prompt caching启用；相关工作负载：FanOutQA每题配1个生成相关问题；案例研究20个前置问题+10个测试问题。
- **评估指标**：Prefill/decode tokens、API cost（美元）、string accuracy、LLM-judge accuracy（GPT-as-judge）。
