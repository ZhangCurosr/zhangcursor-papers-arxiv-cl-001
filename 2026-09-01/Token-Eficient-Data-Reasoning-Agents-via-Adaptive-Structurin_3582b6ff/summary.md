---
title: "Token-Eficient-Data-Reasoning-Agents-via-Adaptive-Structurin"
source: https://arxiv.org/pdf/2608.31082v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:46:17"
field: "Agent推理效率优化"
keywords: ["Agentic Data Cracking", "Token Efficiency", "Adaptive Structuring", "RAG", "LLM Reasoning", "KV Cache Reuse"]
innovations: ["提出Agentic Data Cracking：在推理过程中自适应、推测性地从非结构化文档提取结构化数据，降低重复prefill成本", "设计扩展RDF数据模型与Catalog视图，支持structured read/write并保留robust fallback", "实证验证：FanOutQA上成本降低53%，10th percentile节省9×，准确率无显著下降"]
benchmarks: ["FanOutQA", "Cinema Case Study (Hitchcock)"]
---

# 论文速读：Token-Eficient-Data-Reasoning-Agents-via-Adaptive-Structurin

## 一句话总结
本文提出 **Agentic Data Cracking** 方法，在 Agent 打开文档回答问题的同时，以边际成本派生一个 cracking 子 Agent，从已加载的上下文中推测性提取潜在有用的结构化数据（entities/attributes/relations）。未来查询可直接复用这些 cracked objects，避免重复打开文档，从而大幅降低 Token 消耗——在 FanOutQA 基准上将成本降低 53%，且保持准确率不变。

---

## 研究问题与动机

- **问题**：基于 LLM 的 Agentic 推理在处理非结构化数据（网页、报告、合同等）时，需要反复打开大文档以提取分散证据，导致推理成本极高（单个问题可达 100 万 token，近 1 美元）。
- **现有方法不足**：预先对所有文档进行结构化提取不可行——文档包含的实体、属性、关系远超任何实际查询所需，且有用结构直到查询到达才可知；现有 KV-cache 复用仅限于相同输入，无法跨查询复用。
- **关键洞察**：相关工作查询会重复打开相同文档（语义局部性），且文档的 prefill/KV-cache 已经存在——此时派生一个 cracking 分支只需支付少量 decode 成本，即可提取可复用的结构。
- **理想差距**：若数据已完美结构化，同一任务仅需 Text2SQL 查询即可完成，成本大幅降低（FanOutQA 上理想存储便宜 28 倍）。

---

## 核心贡献（创新点）

- **引入 Agentic Data Cracking**：将数据库 cracking 的"查询驱动自适应"理念迁移到 AI 推理领域，但目标是"从无到有创建结构"而非重排现有关系。
  *本质区别*：相比预构建知识图谱/固定 schema 抽取，本方法仅在推理过程中按需、推测性地提取结构化数据。

- **KV-cache 共享前缀复用 + cracking 子 Agent 派生机制**：利用 shared-prefix serving / API prompt caching，cracking 分支在已加载文档的 KV-cache 之上以极低成本派生，仅产生有界 decode。
  *本质区别*：不同于传统答案缓存（仅匹配完全相同问题），本方法推测并提取未来相关查询可能需要的结构，复用范围更广。

- **扩展 RDF 数据模型 + Catalog 视图**：设计含基数、单位、证据注解的 cracked objects 模型，并提供 logical view（catalog）帮助 Agent 判断结构化读是否可用。
  *本质区别*：相比 MemGPT/Letta 等会话记忆系统，本方法记忆的是语料中的实体关系结构，且具有推测/预取能力。

- **实证验证成本 - 准确率权衡**：在 FanOutQA 和 Cinema Case Study 上证明，仅添加一个相关查询即可将成本降低 53%，10th 分位数节省达 9×，且准确率无显著下降。
  *本质区别*：首次系统量化了"自适应结构化"在 Agent 推理中的收益空间与理想存储之间的差距。

---

## 方法详解

### 整体框架
系统包含两条并行路径：
1. **Answer Agent（主 Agent）**：负责回答问题，使用文件系统工具搜索、打开文档、提取事实。
2. **Cracking Sub-Agent（派生 Agent）**：每当 Answer Agent 打开一个文档后，从已加载的上下文派生一个 cracking 分支，推测性提取结构化的 entities/attributes/relations。

### Cracked Objects 数据模型
每个 cracked object 为扩展 RDF-style 边：
$$c = \langle s, r, o, \kappa, u, \varepsilon \rangle$$
- $s$（subject）、$o$（object）：实体或带标签的标量值（string/int/date）
- $r$（relation）：关系标签
- $\kappa \in \{\text{singular}, \text{list}\}$：区分单值关系与列表关系
- $u$：可选规范单位
- $\varepsilon = \langle d, \rho \rangle$：证据来源（文档 ID + 支撑区域）

### Read / Reasoning Behaviour（读取行为）
- Agent 可通过 constrained tool-calls 从 cracked-object store 中按 subject、relation 进行结构化查询
- **Catalog 视图**：对每个文档 $d$，列出可用的 subject-relation 对及其规范标签和基数：
  $$\operatorname{Cat}(d) \triangleq \pi_{s,r,\kappa}\left(\sigma_{\text{doc}(\varepsilon)=d}(C)\right)$$
- 若 catalogue 命中，直接结构化读取；否则 fallback 到打开原始文档（prefill 成本）

### Write / Cracking Behaviour（写入行为）
- **触发条件**：仅当 Answer Agent 因 miss 打开文档时才 fork cracking 子 Agent，绝不单独为提取而打开文档
- **机制**：利用 shared-prefix KV-cache 复用，在相同上下文上以不同 prompt（cracking instruction）进行 bounded decode
- **输出约束**：在固定 token 预算（如 4K）内输出 schema-constrained JSON；完整 list-valued relation 才输出，否则省略
- **后处理**：验证、规范化（数字/单位/日期）、将 list 展开为独立 edge，仅插入 grounded objects
- **独立性**：cracking 在 answer 路径之外并行执行，不阻塞当前答案；若 cracking 失败，Agent 仍可使用原始文档

---

## 实验与结果

- **数据集**：
  - **FanOutQA**：Wikipedia 上的多跳多文档 QA；每道题扩展一个由 LLM 生成、人工验证的相关问题（不同属性，重叠实体集）
  - **Cinema Case Study**：围绕 Alfred Hitchcock 的 20 个相关问题 + 10 个测试问题，模拟更长运行时间的调查场景

- **评估设置**：基于 Claude-Haiku-4.5，开启 API prompt caching；对比基线为纯 Agentic Reasoning（无 cracking）

- **主要结果**（Fig. 3）：
  - **FanOutQA**：平均 prefill 从 189K → 87K tokens；平均成本从 \$0.26 → \$0.12；准确率无显著差异（LLM-judge 42% vs 43%，p=0.39）
  - **Case Study**：prefill 从 565K → 161K tokens；成本从 \$0.81 → \$0.27
  - **Per-question 分布**：median cost \$0.246 → \$0.072（3.4×）；10th percentile 节省达 **9×**；约 3/4 的问题获益
  - **Cracking 开销**：4K token 预算带来 ~12% 额外开销，产生约 150 个 cracked objects
  - **理想存储差距**：FanOutQA 上理想预结构化存储便宜 **28×**，当前方法正在缩小这一差距

- **核心结论**：在保留 Agentic 推理准确率的同时，将 API 成本降低约 53%（FanOutQA），10th percentile 节省达 9 倍；随着工作负载运行更久、reuse 累积，成本收敛至理想存储。

---

## 相关工作脉络

- **Database Cracking（Idreos et al., 2007）**：查询驱动的增量索引重建；本文借鉴"查询决定组织方式"理念，但目标是"从无到有创建结构"而非重排已有数据。
- **Semantic Cache / Answer Cache**（GPTCache 等）：缓存近似重复问题的答案；本文的 cracked objects 跨不同查询复用，复用范围更广。
- **Knowledge Base Construction / Graph RAG**（Wikidata, GraphRAG）：预先构建全局知识图谱；本文不做全量抽取，仅按需提取相关结构，且可跨模型持久化。
- **Query-Driven Document Analytics**（Lin et al., 2025）：仅提取当前查询所需值；本文通过 speculation 提取超出当前查询的结构。
- **Agent Memory Systems**（MemGPT, Letta, Mem0, Zep, A-Mem）：记忆会话历史和偏好；本文记忆的是语料本身的实体/关系结构，且具有预取能力。
- **KV-Cache Serving Systems**（PagedAttention, CacheGen, Mooncake, CacheBlend）：复用 GPU/host 内存中的 KV 状态；本文在此基础上进一步将 cached prefix 转化为可跨模型复用的文本结构。

---

## 局限性与未来方向

- **Schema 受限**：当前仅支持 string/int/date 三种标量类型和预定义 cardinality，缺乏 SQL 级别的计数、聚合、join、多跳查询能力。
- **静态语料假设**：未处理文档编辑/更新场景；未来需支持受影响 cracked objects 的删除或增量修正。
- **Benchmark 局限**：实验基于公开 Wikipedia 内容（模型可能已见过）；真实企业私有文档的跨查询复用价值更高，但缺乏对应 benchmark。
- **Cracking 预算分配**：当前使用固定 4K token 预算 per question；更智能的 per-query/per-document 动态预算分配是未来方向。
- **长尾问题**：约 10% 的问题（90th percentile）因无 reuse 而亏损（成本为基线的 1.24×）。

---

## 研究启发与可借鉴点

- **KV-cache + 元数据双轨复用**：利用 shared-prefix serving 让 cracking 分支以极低成本派生，是"推理时元数据提取"的有效范式，可迁移到其他需要频繁打开文档的场景。
- **推测性预提取（Speculative Prefetching）**：不局限于当前查询所需字段，而是提取"可能被相关查询复用的相邻结构"——这种基于语义局部性的预取策略可推广到 RAG、Agent Memory 等方向。
- **Catalog 作为结构 discovery 接口**：在 document search result 之后、open 决策之前暴露可用 subject-relation 对，避免无效的 document open，设计简洁且高效。
- **跨模型持久化结构**：Cracked objects 以纯文本形式存储，不受模型版本绑定，可作为组织级资产跨 Agent/用户积累——为"数据护城河"提供了新思路。
- **Agentic Data Reasoning Benchmark 设计**：FanOutQA 本身缺乏 query locality，作者通过"每道测试题扩展一个相关题"的方式人工注入语义局部性，可作为后续工作的评估范式。

---

## 关键术语表

- **Agentic Data Cracking (AdC)**：当 Agent 打开文档回答问题时，派生一个子 Agent 以边际成本推测性提取结构化数据的方法。
- **Cracked Object**：从非结构化文档中提取的、带证据标注的结构化三元组（实体-关系-值），扩展自 RDF 边。
- **Catalog**：对 cracked-object store 的逻辑视图，列出每个文档可用的 subject-relation 对及其规范标签，用于指导结构化查询。
- **Prefill-intensive**：推理中 prefill（key-value cache 构建）占主导的成本特征，区别于 decode-intensive 的数学推理。
- **Shared-prefix KV-cache**：利用 LLM 推理框架的 prefix caching，使多个请求共享相同前缀的 KV 状态以跳过重复 prefill。
- **Semantic Locality**：相关查询在语义层面重复访问重叠文档和实体集的规律，是 AdC 收益的来源。
- **Data Reasoning**：需要从非结构化文档中提取证据、构造中间数据结构、并通过多步推理得出答案的任务模式。
- **Bounded Decode Budget**：cracking 子 Agent 在固定 token 上限内生成结构化 JSON 输出的设计约束。

---

## 可复现要素

- **数据集**：FanOutQA（公开）、Cinema Case Study（自构建，论文未提供代码）
- **代码/权重**：论文未提及开源；基于 Claude-Haiku-4.5 API
- **关键超参**：Cracking decode budget = 4K tokens per question；模型：Claude-Haiku-4.5；Prompt caching：开启；FanOutQA 扩展问题由 claude-opus-5 生成 + 人工验证

---
