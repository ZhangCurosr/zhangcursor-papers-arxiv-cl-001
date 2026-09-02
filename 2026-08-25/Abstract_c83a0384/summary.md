---
title: "Abstract"
source: https://arxiv.org/pdf/2608.23265v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:35:58"
field: "跨会议知识演进问答"
keywords: ["Cross-Meeting QA", "Knowledge Evolution", "State Overwriting", "Structured Memory", "RAG", "Long-Context Reasoning"]
innovations: ["提出Build-Read非对称架构，将状态冲突解决前置到写入时", "设计State-Overwrite协议维护实体版本链与Active/Overturned标签", "构建CrossMeet双语基准覆盖事实一致性与时序多跳推理"]
benchmarks: ["CrossMeet-EN", "CrossMeet-ZH", "MeetingQA", "ELITR-Bench", "LongMemEval", "MuSiQue"]
---

# 论文速读：EvoWiki

## 一句话总结

EvoWiki 提出了一种增量问答架构，通过离线 Build-在线 Read 的非对称设计，显式建模跨会议决策的生命周期与状态覆写，实现可追溯的动态知识演进问答。

## 研究问题与动机

- **跨会议状态演进的建模缺失**：实际协作中决策、风险、负责人等状态随会议推进被反复修订、推翻和替换，现有方法缺乏对会议内决策过程和知识库生命周期的显式建模。
- **长上下文模型难以保证证据可靠使用**：单纯扩展上下文窗口长度无法保证模型能够可靠地识别和使用时序相关的证据，易出现"中间丢失"（Lost-in-the-Middle）现象。
- **传统 RAG 按语义相关性排序而非有效性**：现有检索增强方法按语义相关性排序证据，无法区分过时状态与当前有效状态，导致陈旧答案的返回。
- **LLM-Wiki 等结构化记忆方法未建模状态生命周期**：已有结构化记忆方法将知识组织为静态或仅追加的事实，缺乏对跨会议决策替换关系的建模，旧状态与新状态可能同时存在引发混淆。

## 核心贡献（创新点）

1. **提出 EvoWiki 的 Build-Read 非对称架构**：将会议内微演进解析、实体版本链维护与状态覆写协议置于写入时，读取时仅访问已结构化的 Wiki，实现歧义与冲突解决的写入时集中化。与已有工作本质区别在于将状态生命周期管理从读取时前置到写入时。

2. **设计细粒度 State-Overwrite 协议**：每个实体-属性对维护有序版本链，通过 Active/Overturned 标签和版本替换边显式区分当前有效状态与历史状态，保持最多一个活跃版本。与追加式知识库的本质区别在于支持状态的显式覆写与撤销。

3. **构建 CrossMeet 双语基准**：包含 100 个项目、500 场会议、2,000 个 QA 对，平均上下文长度超过 26K token，涵盖事实一致性（50%）、时序推理（30%）和多跳推理（20%）三类任务。与已有会议理解数据集的本质区别在于专为长时序状态演进设计。

4. **实现可追溯的证据链恢复机制**：每个状态绑定来源会议和证据 span，通过 Trace 函数联合返回支持状态集合与版本替换边，实现答案的可验证溯源。与模糊检索返回的本质区别在于确定性实体寻址与结构化证据链。

## 方法详解

**任务形式化**：给定按时间排序的会议序列 $\mathcal{D} = \{M_1, M_2, \ldots, M_T\}$，处理完第 T 场会议后回答查询 q。原子会议知识表示为 $s_i = \langle e_i, r_i, v_i, \tau_i, \ell_i, p_i \rangle$，其中 $p_i = (m_i, \text{span}_i)$ 为来源会议与证据 span。

**Build-Read 架构**：
- **离线 Build 阶段**执行四步操作：
  1. **Temporal Alignment (Resolve)**：规范化会议时间线，解析会议内提案→讨论→决策的微演进序列
  2. **Write-time Coreference Resolution (Coref)**：将当前会议与 $\mathcal{W}_{t-1}$ 中的实体索引结合，解析指代
  3. **Information Extraction (Extract)**：生成候选更新集 $\mathcal{U}_t$
  4. **Entity Routing + State-Overwrite Protocol**：将更新绑定到实体-属性槽位，执行状态覆写

- **在线 Read 阶段**：将完整 Wiki 序列化为唯一上下文 $C_T$，通过确定性实体寻址、跨实体多跳推理和时序聚合生成答案，不访问原始会议记录。

**Structured Wiki 表示**：
$$\mathcal{W}_t = (\mathcal{E}_t, \mathcal{H}_t, \mathcal{A}_t, \mathcal{P}_t)$$
其中 $\mathcal{H}_{e,r}^{(t)}$ 为实体-属性对的有序版本链。当新状态替换当前活跃状态时，旧记录不物理删除，而是关闭有效区间、标记为 Overturned，新状态标记为 Active，创建版本替换边。约束条件：
$$\sum_{s \in \mathcal{H}_{e,r}^{(t)}} \mathbf{1}[\ell(s) = \text{Active}] \leq 1$$

**可追溯性机制**：
$$\operatorname{Trace}(\hat{y}) = \bigcup_{s \in \hat{S}_q} p(s) \cup \operatorname{VersionEdges}(\hat{S}_q)$$

## 实验与结果

**数据集**：
- **CrossMeet-EN / CrossMeet-ZH**：自行构建的双语基准，100 项目 × 500 会议 × 2000 QA，平均上下文长度英文 26,861 / 中文 29,480 tokens
- **MeetingQA**：基于 AMI 场景会议的人类记录转录
- **ELITR-Bench**：基于真实项目会议 ASR 转录的长会议理解基准
- **LongMemEval**：跨会话长期记忆与知识更新测试
- **MuSiQue**：静态组合式多跳推理基准

**评估基线**（9 个）：Direct LLM、VanillaRAG (BM25/Dense/Hybrid)、RAPTOR、GraphRAG、LightRAG、HippoRAG 2、LogicRAG

**读者模型**：DeepSeek-V4-Flash、Qwen3.5-397B-A17B

**主要结果**：
- 使用 DeepSeek-V4-Flash 时，EvoWiki 宏观平均 Judge Accuracy 达 **60.09%**，超越最强基线 LogicRAG (50.37%) **9.72 个百分点**
- 使用 Qwen3.5-397B-A17B 时，EvoWiki 达 **63.02%**，超越 LogicRAG (53.02%) **10.00 个百分点**
- 在 LongMemEval 上领先最强基线分别达 **13.10** 和 **14.10 个百分点**
- CrossModel Judge Robustness 实验中，三种独立 Judge (Gemini 3.1 Pro, Claude Opus 4.7, DeepSeek-V4-Pro) 均将 EvoWiki 排在第一

**消融实验关键数字**：
- w/o Overwrite：平均 51.46（↓8.63）
- w/o Coref：平均 50.93（↓9.16）
- w/o Entity：平均 53.73（↓6.36）
- w/o Tags：平均 49.79（↓10.30）

## 相关工作脉络

1. **长上下文 QA 与 RAG**：Direct LLM 和 VanillaRAG 依赖语义相关性检索，无法区分新旧状态；EvoWiki 将冲突解决前置到写入时，读取时直接访问已清理的结构化知识。

2. **结构化知识组织方法**：GraphRAG、LightRAG、HippoRAG 2、LogicRAG 等改进知识组织形式，但未显式建模跨会议决策生命周期和状态替换关系；EvoWiki 通过版本链和覆写协议显式追踪状态演化。

3. **LLM-Wiki 与结构化记忆**：WiCER 等方法将来源编译为持久记忆，但知识组织为静态或仅追加形式；EvoWiki 在写入时维护 Active/Overturned 标签和版本替换边。

4. ** temporal retrieval 与参数化编辑**：FreshLLaMA、RASTeR 等方法处理时间敏感知识，但难以保持会议级溯源；EvoWiki 将当前状态与历史状态共同绑定到证据链中。

5. **会议理解与 QA**：MeetingQA、ELITR-Bench 聚焦单次会议理解；CrossMeet 专为跨会议状态演进设计，涵盖角色绑定、版本替换和多跳证据链。

## 局限性与未来方向

- **Build 阶段提取完整性限制**：Wiki-onlly Read 受限于离线提取的完整性，特别是在 ASR 噪声和非正式 speech 场景下可能遗漏关键事实。
- **错误归因偏向写入侧**：错误分析显示 54.2% 的错误来自 Build 阶段（提取遗漏 27.4%、覆写失败 15.1%），说明写入保真度是当前瓶颈。
- **未来方向**：探索不确定性感知写入（uncertainty-aware writing）、选择性来源验证（selective source verification）、扩展至更广泛语言和领域、支持更长的时间跨度。

## 研究启发与可借鉴点

1. **写入时冲突解决架构**：将歧义消解和状态冲突处理前置到离线构建阶段，使在线读取简化为确定性寻址，这一非对称设计值得迁移到其他动态知识场景。

2. **显式版本链与标签体系**：通过 Active/Overturned 标签和版本替换边维护状态演化历史，为知识演进追踪提供了可复用的结构化表示方案。

3. **CrossMeet 基准设计思路**：通过受控的项目蓝图→会议序列→跨会议 QA 生成管道，结合多模型交叉验证和人工审计，为合成基准构建提供了高质量评估范式。

4. **消融实验的精细设计**：分别移除状态覆写、核心指代解析、实体中心化组织和结构标签，定量揭示了各组件的贡献，消融策略值得借鉴。

5. **多 Judge 交叉验证**：使用三个独立模型作为 Judge 验证结果稳健性，避免单一 Judge 偏好导致的评估偏差。

## 关键术语表

**EvoWiki**：一种增量问答架构，通过离线 Build-在线 Read 的非对称设计，显式建模跨会议知识的生命周期与状态覆写。

**State-Overwrite Protocol**：状态覆写协议，当新状态替换旧状态时，不物理删除旧记录，而是关闭其有效区间并标记为 Overturned，同时创建版本替换边。

**Entity Version Chain**：实体版本链，为每个实体-属性对维护的时间有序状态序列，确保最多只有一个 Active 版本。

**CrossMeet**：双语跨会议问答基准，包含 100 项目、500 会议、2,000 QA 对，覆盖事实一致性、时序推理和多跳推理三类任务。

**Write-time Coreference Resolution**：写入时核心指代解析，将会议中的指代表达映射到 Wiki 中的规范实体，避免实体碎片化。

**Deterministic Entity Addressing**：确定性实体寻址，读取时通过实体、属性和生命周期标签直接定位有效状态，而非基于语义相关性的 Top-k 检索。

**Provenance Anchor**：溯源锚点，每个状态绑定来源会议 ID 和证据 span，支持证据链恢复。

**Judge Accuracy**：评估指标，采用 LLM-as-a-Judge 范式，由 Gemini 3.1 Pro 对答案正确性进行 0/0.5/1 三分值打分。

## 可复现要素

- **数据集**：CrossMeet 中英文版本，论文声明将在接受后开源
- **代码/权重**：论文声明代码和实现将在接受后发布
- **关键超参**：Chunk size 根据数据集不同设置为 200/512 tokens，overlap 为 25/64 tokens，Top-k 检索为 3/10；温度设为 0.0，最大生成长度 1024 tokens
- **读者模型**：DeepSeek-V4-Flash、Qwen3.5-397B-A17B
- **Build 模型**：DeepSeek-V4-Flash

---
