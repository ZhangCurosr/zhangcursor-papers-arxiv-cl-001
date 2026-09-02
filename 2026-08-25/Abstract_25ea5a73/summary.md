---
title: "Abstract"
source: https://arxiv.org/pdf/2608.23265v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:36:14"
field: "跨会议知识演进问答"
keywords: ["跨会议问答", "状态演进", "RAG", "实体版本链", "结构化记忆", "长上下文"]
innovations: ["提出不对称 Build-Read 架构，将状态冲突解决移至写入阶段", "设计实体版本链与细粒度状态覆写协议，显式区分有效状态与历史版本", "构建双语高保真基准 CrossMeet，覆盖事实一致性、时间推理与多跳证据链"]
benchmarks: ["CrossMeet-EN", "CrossMeet-ZH", "MeetingQA", "ELITR-Bench", "LongMemEval", "MuSiQue"]
---

# 论文速读：Abstract

## 一句话总结
论文提出 **EvoWiki**，一种面向动态跨会议长文本的增量问答架构，通过将离线**增量构建（Build）**与在线**结构化读取（Read）**解耦，利用**实体版本链**与**细粒度状态覆写协议**显式区分当前有效状态与被取代的历史记录，同时保留会议级溯源锚点；并构建了双语基准 **CrossMeet** 验证其效果。

## 研究问题与动机
1. **长期协作会议的状态演进特性**：在跨多次会议的场景中，事实状态（如决策、风险、负责人）会不断被修订、推翻和替换，而现有长上下文方法通常堆叠完整历史，无法有效处理状态更新。
2. **现有RAG/Lwiki方法的静态性局限**：传统RAG、LLM-Wiki及结构化记忆方法将知识组织为静态或仅追加的事实，依赖语义相关性检索；在缺乏明确建模会议内决策过程和知识生命周期时，可能同时保留冲突的旧状态与新状态，或在更新快照时丢弃历史，导致检索混乱且答案难以验证。
3. **缺乏对跨会议状态演进的显式建模**：现有任务在角色绑定、版本替换以及跨会议多跳证据合成方面覆盖不足，难以应对因状态覆盖引起的时效性冲突和证据溯源需求。

## 核心贡献（创新点）
1. **提出 EvoWiki 的不对称 Build–Read 架构**：将离线增量构建（含时间对齐、微演化解析、写时消歧、实体路由与状态覆写）与在线仅基于 Wiki 的结构化读取解耦，把歧义与冲突解决移至写入阶段，使在线查询无需访问原始会议记录。
2. **设计实体版本链与细粒度状态覆写协议（State-Overwrite Protocol）**：显式区分当前有效状态与被取代的历史状态，同时保留会议级溯源锚点（provenance anchors），保证版本历史可审计，并满足任意实体–属性对在同一时刻至多只有一个 Active 状态的约束。
3. **构建双语高保真基准 CrossMeet**：涵盖事实一致性、时间推理与跨会议多跳问答，每个语言包含 100 个项目、500 场高保真模拟会议与 2,000 个 QA 对，平均上下文长度超 26K tokens，并提供问题类型、推理跳数与跨会议证据链标注。
4. **实现可追溯且低幻觉风险的问答**：通过确定性实体寻址与跨实体多跳聚合，结合版本替换边与溯源锚点，生成具有证据链回溯能力的结构化答案，显著减少由于冲突状态共存导致的过时检索。

## 方法详解
- **任务形式化**：给定按时间排序的会议序列 $\mathcal{D} = \{ M_1, M_2, \ldots, M_T \}$，在处理到第 $T$ 次会议后回答查询 $q$。原子会议知识表示为 $s_i = \langle e_i, r_i, v_i, \tau_i, \ell_i, p_i \rangle$，其中 $p_i=(m_i, \text{span}_i)$ 为来源会议与证据跨度。Wiki 在因果约束下更新：$\mathcal{W}_t = F_{\text{Build}}(\mathcal{W}_{t-1}, M_t)$。
- **Build–Read 架构**：离线 Build 执行时间对齐（Resolve）、会议内微演化解析、写时核心共指消解（Coref）、信息抽取（Extract）、实体路由（Route）与状态覆写；在线 Read 以完整 Wiki 为唯一上下文，通过确定性实体寻址、跨实体多跳推理与时序聚合生成答案，不回退访问原始会议记录。
- **状态覆写与结构化 Wiki**：Wiki 表示为 $\mathcal{W}_t = (\mathcal{E}_t, \mathcal{H}_t, \mathcal{A}_t, \mathcal{P}_t)$。对每个实体–属性对维护按时间有序的版本链 $\mathcal{H}_{e,r}^{(t)} = \langle s_{e,r}^1, s_{e,r}^2, \ldots, s_{e,r}^n \rangle$。当新状态替换当前活跃状态时，旧记录不物理删除，而是闭合有效期、标记为 Overturned，追加新状态为 Active，并建立版本替换边；保证 $\sum_{s \in \mathcal{H}_{e,r}^{(t)}} \mathbf{1}[\ell(s) = \text{Active}] \leq 1$。
- **Wiki-Only 在线读取**：将完整 Wiki 序列化 $C_T = \text{Serialize}(\mathcal{W}_T)$ 作为唯一上下文，生成答案 $\hat{y}$ 与支持状态集 $\hat{S}_q$；$\hat{S}_q$ 保留为溯源元数据，不直接参与评分。
- **可追溯性与成本**：通过证据链 $\text{Trace}(\hat{y}) = \bigcup_{s \in \hat{S}_q} p(s) \cup \text{VersionEdges}(\hat{S}_q)$ 定位支持答案的会议证据并追溯状态替换原因；存储复杂度为 $O(U)$（$U$ 为状态更新总数），在线输入上下文大小为 $O(S_W)$，无外部检索调用。

## 实验与结果
- **数据集**：自建双语基准 CrossMeet（CM-EN 与 CM-ZH），各含 100 项目、500 会议、2,000 QA 对，平均上下文长度分别为 26,861 与 29,480 tokens；另在 4 个公开基准（MeetingQA、ELITR-Bench、LongMemEval、MuSiQue）上评测。
- **基线**：包括 Direct LLM、VanillaRAG（BM25/Dense/Hybrid）、RAPTOR、GraphRAG、LightRAG、HippoRAG 2、LogicRAG 等 9 种代表性方法。
- **阅读器模型**：DeepSeek-V4-Flash 与 Qwen3.5-397B-A17B。
- **主要结果**：在六个数据集上，EvoWiki 的宏观平均 Judge Accuracy 相对于最强基线 LogicRAG 分别提升 **9.72** 与 **10.00** 个百分点（DeepSeek-V4-Flash 达到 60.09，Qwen3.5-397B-A17B 达到 63.02）。在 LongMemEval 上分别领先最强基线 13.10 与 14.10 个百分点，证明状态覆写有效缓解长期知识冲突。
- **消融实验**：移除状态覆写（w/o Overwrite）导致平均准确率下降 8.63 个百分点；移除写时共指消解（w/o Coref）下降 9.16 个百分点；移除实体中心化组织（w/o Entity）下降 6.36 个百分点；移除所有结构化标签（w/o Tags）下降最多，达 10.30 个百分点，表明显式结构表征对实体、关系、时序、状态与溯源边界的关键引导作用。
- **稳定性与跨模型评估**：EvoWiki 在五轮不同解码种子与温度设置下保持最高均值与最低标准差（0.59）；在三个独立 Judge 模型（Gemini 3.1 Pro、Claude Opus 4.7、DeepSeek-V4-Pro）交叉评估中均排名第一，与主 Judge 一致性 κ 达 0.887–0.901。
- **错误分析与效率**：212 个错误中 Build 侧占 54.2%，Read 侧占 45.8%；主要失败模式为抽取遗漏（27.4%）、覆写失败（15.1%）、过时状态读取（13.2%）等。对比 Direct LLM，EvoWiki 降低推理延迟 45.0%，减少 Read Token 使用 35.0%。
- **状态翻转敏感性**：随着实体状态翻转次数从 0–1 增至 4，EvoWiki 仅从 71.03% 降至 66.90%（下降 4.13 个百分点），而 VanillaRAG、GraphRAG、LogicRAG 下降幅度更大，在高冲突子集上优势显著（领先 5.70–24.10 个百分点）。

## 相关工作脉络
1. **长上下文 QA 与 RAG**：传统 RAG（Lewis et al. 2020）及 Self-RAG、Corrective RAG、Adaptive-RAG 等侧重检索控制，但缺乏对跨会议决策生命周期的建模；EvoWiki 在写入时完成版本验证，区别于仅依赖语义相关性的检索范式。
2. **结构化记忆与图谱方法**：RAPTOR、GraphRAG/LightRAG、HippoRAG 2、LogicRAG 等方法采用层级摘要、实体图谱或关联推理，但未显式维护状态替换关系；EvoWiki 通过版本链与覆写协议同时保留当前有效视图与可追溯历史。
3. ** temporal 知识与事件溯源**：时间感知检索（Vu et al. 2024; Schumacher et al. 2025）处理时间敏感知识，参数化编辑研究（Menk et al. 2022; Wang et al. 2024）关注事实定位与多跳一致性，但难以保留会议级溯源；EvoWiki 在写入时将当前与已推翻状态绑定至证据，形成版本化账本。
4. **会议理解与 QA 基准**：MeetingQA（Prasad et al. 2023）、ELITR-Bench（Thonet et al. 2025）、LongMemEval（Wu et al. 2024）等聚焦单次会议或通用记忆；CrossMeet 专门针对双语跨会议状态演进，覆盖角色绑定、版本替换与多跳证据链。

## 局限性与未来方向
- **构建提取完整性限制**：Wiki-Only 读取受限于 Build 阶段的提取完备性，尤其在 ASR 噪声与非正式 Speech 场景下，关键事实可能未被完整捕获或准确写入。
- **线性增长的输入上下文**：在线查询需读取完整 Wiki（长度 $S_W$ 随会议累积线性增长），虽消除了检索遗漏风险，但在超大规模场景下面临计算与延迟压力。
- **未来方向**：作者提及将探索不确定性感知写入（uncertainty-aware writing）、选择性源验证（selective source verification）以及扩展至更广泛的语言、领域与时间跨度。

## 研究启发与可借鉴点
1. **写入时消歧与冲突解决的不对称设计**：将复杂的时空对齐、实体消歧与状态冲突化解移至离线 Build 阶段，可使在线 Read 简化为确定性寻址，这一思路可迁移至其他需要频繁状态更新的长对话或多文档推理系统。
2. **实体版本链与覆写协议的结构化表达**：通过显式版本链、生命周期标签（Active/Overturned）与版本替换边，实现“当前有效状态唯一性”约束，可有效防止冲突状态共存导致的过时检索，适用于金融、法律、项目管理等状态频繁变更的知识密集型领域。
3. **高保真模拟基准构建方法**：CrossMeet 采用“项目蓝图→会议序列→跨会议 QA”可控生成流水线，并以多模型交叉验证与独立人工评估（Fleiss’ κ > 0.90）保障质量，为领域特定的长上下文基准建设提供了可复用的工程范式。
4. **可追溯证据链生成机制**：答案生成时同步输出支持状态集 $\hat{S}_q$ 与溯源锚点，实现证据链回溯（Trace），有助于降低幻觉风险并提升答案可解释性，可结合审计需求应用于合规性较强的应用场景。

## 关键术语表
- **EvoWiki**：一种面向动态长文本的增量问答架构，通过异步的 Build–Read 设计与实体版本链实现状态演进的知识组织。
- **Build–Read 架构**：将离线增量构建（写入时段）与在线结构化读取（查询时段）解耦，把歧义与冲突解决前置到写入阶段。
- **实体版本链（Entity Version Chain）**：针对每个实体–属性对维护按时间排序的状态序列，显式记录状态替换关系。
- **State-Overwrite Protocol**：状态覆写协议，在写入时闭合旧状态有效期、标记为 Overturned，并追加 Active 新状态，保证同一时刻至多一个有效状态。
- **Provenance Anchors**：溯源锚点，记录每个状态对应的来源会议与证据跨度，支持答案的证据链回溯。
- **CrossMeet**：作者构建的双语跨会议基准，包含事实一致性、时间推理与多跳 QA，用于评估状态演进处理能力。
- **Judge Accuracy**：基于 LLM-as-a-Judge 的评估指标，由 Gemini 3.1 Pro 以 0/0.5/1 分制对答案正确性进行评分。
- **Deterministic Entity Addressing**：确定性实体寻址，Read 阶段通过实体、属性与生命周期标签直接定位 Wiki 中的有效状态，而非语义相似度检索。

## 可复现要素
- **数据集**：CrossMeet（英文与中文版本）论文声明将在接受后发布；四个公开基准 MeetingQA、ELITR-Bench、LongMemEval、MuSiQue 均为已发布数据集。
- **代码与权重**：论文声明实现将在接受后发布；阅读器模型使用 DeepSeek-V4-Flash 与 Qwen3.5-397B-A17B，Embedding 模型使用 Qwen3-Embedding-8B。
- **关键超参**：检索基线采用统一长度分桶配置（短上下文 chunk=200, overlap=25, Top-3；长上下文 chunk=512, overlap=64, Top-10）；生成长度上限 1,024 tokens；温度设定 0.0（稳定性实验 varied {0.1, 0.3, 0.5, 0.7, 0.9}）。
- **评估设置**：每方法运行五次取均值，以 Gemini 3.1 Pro 为主 Judge，交叉评估使用 Claude Opus 4.7 与 DeepSeek-V4-Pro；人工验证采样 300 实例（每语言 150），三位标注员独立评估。
