---
title: "Same-Agent-Diferent-Answers"
source: https://arxiv.org/pdf/2608.22856v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:00:06"
field: "检索增强生成系统的评估与可靠性"
keywords: ["retrieval-augmented generation", "answer churn", "RAG evaluation", "backward compatibility", "LLM evaluation", "corpus scaling", "snapshot audit"]
innovations: ["提出 Repeat-Aware Snapshot Compatibility Audit，通过扣除同快照重复噪声估计超额答案漂移 D-hat", "定义 repeat-stable semantic flip 诊断，定量归因答案漂移的构成", "揭示 accuracy-blind answer churn 现象：聚合准确率不变时答案级行为已发生显著不兼容"]
benchmarks: ["Natural Questions", "TriviaQA"]
---

# 论文速读：Same-Agent-Different-Answers

## 一句话总结
本文提出 **Snapshot Compatibility Audit** 方法，通过扣除同快照重复噪声来量化 RAG 系统在不同语料规模下的"超额答案漂移（excess answer churn）"；实验发现 FineWeb 索引从 1 shard 扩至 7 shard 时，NQ 语义超额漂移达 **10.25 pp**，而 EM 仅变化 −1.50 pp，说明聚合准确率会掩盖大量行为层面的兼容性问题。

## 研究问题与动机
- **核心问题：** 当 RAG 系统的检索语料扩展后，即使模型、prompt、检索策略、证据深度和生成控制全部固定，系统返回的答案是否仍然与旧版本"行为兼容"？
- **现有评估的盲区：** 现有工作（包括 corpus-scaling 研究）主要关注平均质量/聚合效用（如 EM、F1），但 gains 和 losses 相互抵消会让准确率仪表盘保持绿色，同时掩盖答案级层面的实质性变化。
- **随机性干扰：** RAG 系统本身具有生成随机性，一次调用前后的答案不同无法直接归因于语料更新，需要建立"同状态重复噪声基线"。
- **下游影响：** RAG 系统的输出常被用于缓存、回归测试、自动化工作流甚至人类决策，行为不兼容可能破坏这些依赖链路。

## 核心贡献（创新点）
1. **识别并命名"accuracy-blind answer churn"**：首次系统揭示索引扩展可以在聚合效用几乎不变的情况下导致 RAG 系统行为不兼容，与已有 RAG 评估工作（RAGAS、ARES 等）关注组件级指标的定位不同。
2. **提出 Snapshot Compatibility Audit 框架**：通过在同一语料快照上两次独立调用估计生成噪声基线，再用跨快照相似度与之相减得到超额漂移估计量 $\widehat{D}$，区别于以往仅比较单次输出或做 raw "answer changed" 指示的做法。
3. **引入 repeat-stable semantic flip 诊断**：定义严格语义翻转问题（Equation 7），实现"8/400 NQ 问题的语义翻转贡献了 10.00/10.25 pp 的语义超额漂移"的定量归因，填补了"谁在变"的缺失。
4. **设计预注册的四阶段审计协议 + 跨家族 LLM judge 鲁棒性检验**：blind semantic judge、delayed gold unlock、outcome-blind V4-Pro 复现，保证审计结论不受结果泄露影响，与既有 LLM judge 工作（MT-Bench 等）对比而言强调"去偏"而非"增强判别力"。

## 方法详解
- **测试平台：** 单轮 retriever-generator，固定 deepseek-v4-flash 模型、固定 prompt、固定 top-8 检索、固定渲染格式，仅改变可访问的 FineWeb 前缀规模（n0、n1、n3、n7）。
- **核心估计量：** 对问题 $i$，在低快照 $L$ 和高快照 $H$ 各生成两次独立回答 $L_{i,a}, L_{i,b}$ 和 $H_{i,a}, H_{i,b}$，定义相似度核 $k \in [0,1]$：
  - 同快照内平均相似度：$w_i = \frac{1}{2}[k(L_{i,a}, L_{i,b}) + k(H_{i,a}, H_{i,b})]$
  - 跨快照平均相似度：$c_i = \frac{1}{4}\sum_{r,s \in \{a,b\}} k(L_{i,r}, H_{i,s})$
  - 超额漂移估计：$\widehat{D} = \frac{1}{N}\sum_{i=1}^{N}(w_i - c_i)$
  - 等价形式：$\widehat{D} = (1 - \bar{c}) - (1 - \bar{w})$，即"跨快照差异 − 同快照噪声基线"。
- **两种核函数：**
  - **Normalized-exact kernel：** 基准格式化后字符串完全相等为 1，否则为 0；理论上有精确的 MMD 解释（Equation 6）。
  - **Blinded semantic kernel：** 双盲 LLM judge 判断两答案是否语义等价（允许合理别名），不暴露 scale/evidence/gold。
- **推断：** 整题 bootstrap，50,000 次重采样，NQ 采用交叉_union 决策门（normalized-exact $\widehat{D} \geq 3$ pp + 单边 95% LCB > 0 + 语义 LCB > 0）。
- **稳定翻转诊断（post-hoc）：** 若 $k(L_a, L_b) = k(H_a, H_b) = 1$ 且所有跨快照对 $k(L_r, H_s) = 0$，则记为 repeat-stable semantic flip，每个问题贡献最大单位值 1 给 $\widehat{D}$。
- **审计协议四阶段：** (1) 冻结对比条件；(2) 每题至少两次独立输出并 counterbalance；(3) 盲态相似性比较 + 跨家族 judge 验证；(4) 结合 flipped questions、EM 转移矩阵、检索重叠度进行分级。

## 实验与结果
- **数据集：** NQ 400 题（confirmatory，预注册）+ TriviaQA 200 题（supportive，预注册）+ 盲选 100 题 V4-Pro 复现。
- **基线与设置：** 语料规模 n0/n1/n3/n7；DeepSeek v4-flash（主）与 v4-pro（复现）；检索 top-8，每文档最多 1200 token；n1→n7 为主要对比。
- **NQ 主结果（Table 2）：**
  - Normalized-exact：within 25.25% → cross 18.81%，$\widehat{D} = 6.44$ pp，95% LCB = 4.56 pp，EM 变化 −1.50 pp。
  - Blind semantic：within 89.13% → cross 78.88%，$\widehat{D} = 10.25$ pp，LCB = 7.69 pp。
- **TriviaQA 支持性结果：** 精确 $\widehat{D} = 3.00$ pp（LCB 0.50）、语义 $\widehat{D} = 2.13$ pp（LCB 0.375），但 EM 反而上升 +1.25 pp（方向与 NQ 相反）。
- **V4-Pro 复现：** 精确 $\widehat{D} = 7.25$ pp，语义 $\widehat{D} = 8.75$ pp，EM 上升 +3.00 pp。
- **EM 掩盖效应（Figure 3 / Section 5.3）：** 46/800 从 match→nonmatch，34/800 反向，净 −1.50 pp，但总流动 80/800 = 10.00 pp；另有 155/800 = 19.38% 在 nonmatch 类内部发生语义转变，对 EM 完全不可见。
- **稳定语义翻转：** NQ 40/400 题（10.00%），TriviaQA 5/200 题（2.50%），V4-Pro 100 题中 6 题。
- **检索重叠度：** n1 与 n7 top-8 平均仅重叠 1.19/8（NQ）和 1.09/8（TriviaQA），零重叠率 27.75%/29.50%，说明大多数漂移发生在证据高度不同的条件下。
- **跨家族 judge 一致性：** DeepSeek 与 OpenAI gpt-5.6-sol 在 1,400 对标签上一致率 95.71%，Cohen's κ = 0.855；NQ 语义 $\widehat{D}$ 原始 judge 9.50 pp vs 跨家族 judge 11.00 pp。

## 相关工作脉络
1. **Corpus/数据缩放研究**（如 Scaling Retrieval-Based LM、Atlas）：主要报告聚合效用随 scale 的提升，本文聚焦**答案级兼容性**而非平均质量。
2. **RAG 评估工具**（RAGAS、ARES、RAGChecker、RAGTruth）：评估相关性、faithfulness、幻觉等组件级指标；本文引入**repeat-aware 噪声校正**的漂移估计量。
3. **Consistency/Robustness 工作**（Factual Consistency、Con-RAG、Stable-RAG）：关注同问改写、retrieval 排列、冲突证据下的输出稳定性；本文改变的是**语料可访问规模**，证据变化是 endogenous 而非人工构造。
4. **Prediction churn 与向后兼容更新**（Launch and Iterate、Backward-Compatible Prediction Updates、MUSCLE）：研究模型权重更新导致的预测漂移；本文改变的是**检索语料**而非模型参数，属于不同的漂移源。
5. **Dynamic corpora / 动态知识**（StreamingQA、RealTime QA、FreshLLMs、SituatedQA）：关注时间敏感知识；本文使用**冻结的 FineWeb 索引**进行嵌套前缀扩展，并非时间更新。
6. **LLM Judge 偏见研究**（LLM Evaluators Favor Own Generations、MT-Bench）：本文通过 blind 双盲 judge、cross-family 复现、delayed gold 解锁降低 bias，是该方向的实际应用。

## 局限性与未来方向
- **单一固定路径：** 仅测试 FineWeb 一条嵌套前缀路径（n0 ⊂ n1 ⊂ n3 ⊂ n7），不能推广为通用"缩放定律"或预测任意索引更新的行为；未随机化 shard 子集。
- **单次重复的限制：** 每态仅两次独立调用能估计噪声基线，但对条件输出分布的刻画较粗糙，更多重复可降低方差。
- **仅限 DeepSeek 模型与英文 QA：** V4-Pro 复现与 v4-flash 主实验是不同模型+接口同时变化，无法分离模型效应；未见非英文/非 QA 场景。
- **Single-turn 限制：** 未覆盖 agent 多轮交互、工具调用、迭代检索等复杂场景，结论对 trajectory-level 兼容性而言既不是上界也不是下界。
- **Judge 非人类标注：** cross-family judge 仅 50 题 1,400 对标签，且未保留原始执行 trace；语义等价判定仍依赖 LLM。
- **机制未识别：** 已知答案漂移幅度大但无法定位具体哪条文档导致哪个问题的翻转；overlap-bin 分析非单调且 Fisher 检验不显著。
- **Future directions：** 扩展到多模型族、多检索器、不同任务类型、随机语料路径、时间动态索引更新；探索 trajectory-level 兼容性度量；开发更丰富的 flip 类型分类；建立应用相关的阈值标准。

## 研究启发与可借鉴点
1. **Repeat-aware 噪声校正范式可直接迁移：** 对任何涉及"相同系统配置在不同时间点/版本下输出一致性"的评测（模型更新、API 升级、prompt 变更），均可套用 $w_i - c_i$ 框架扣除基线噪声，避免夸大变化幅度。
2. **Blind semantic judge + delayed gold unlock 协议设计值得借鉴：** 四阶段 locked 流程（retrieval lock → answer lock → blind judgment lock → gold unlock）是生成式评测中减少结果泄露的有效实践，适用于构建高可信 benchmark。
3. **EM-transition matrix 可视化辅助解读：** 将 $\Delta$EM 拆解为 match→nonmatch / nonmatch→match / nonmatch→nonmatch 三类流动，比单一 delta 更能解释"为什么准确率没变但答案变了"，可移植到任何 QA 评测报告。
4. **与团队方向结合机会：** 若团队关注 RAG 生产系统的版本管理/灰度发布/回归测试，可将 Snapshot Compatibility Audit 集成到 CI pipeline，作为"检索语料更新"后的兼容性门禁；进一步可与 team 现有的 RAG 评测工具链（如 RAGAS 相关指标）串联使用。
5. **Stable-flip 诊断的精细化扩展：** 当前仅识别"全四对跨快照均不同"的严格翻转，团队可在此基础上细分：partial flip（部分重复对跨越）、EM-blinding flip（EM 相同但语义不同）等更细粒度类型，形成一套 RAG 漂移分类体系。

## 关键术语表
- **Accuracy-blind answer churn：** 聚合准确率几乎不变（gains/losses 抵消），但答案级行为已发生显著改变的 RAG 系统更新现象。
- **Snapshot Compatibility Audit：** 一种 repeat-aware 审计协议，通过比较同快照重复相似度与跨快照相似度来估计超额答案漂移 $\widehat{D}$。
- **Excess answer churn ($\widehat{D}$)：** 跨快照平均相似度与同快照重复平均相似度之差，衡量超出正常生成噪声的额外答案变化幅度（单位为百分比点 pp）。
- **Repeat-stable semantic flip：** 同一问题在两快照各自两次独立调用均保持语义一致，但跨快照间所有对均为语义不同的严格问题级翻转。
- **Normalized-exact kernel：** 对答案进行基准格式化归一后，字符串完全相同得 1、否则得 0 的二值相似度核；对应精确的 MMD² 解释。
- **Blinded semantic kernel：** 由 LLM judge 在不暴露 scale/evidence/gold 的情况下判定的两答案是否语义等价的相似度核。
- **Nested FineWeb prefix：** 将 FineWeb 数据集按 shard 编号形成层级包含关系（n1 ⊂ n3 ⊂ n7）的语料可访问路径，用于构造"固定其他条件仅扩展语料"的自然对照。
- **Intersection-union decision gate：** NQ 主研究的三重决策规则：normalized-exact $\widehat{D} \geq 3$ pp、exact 单边 95% LCB > 0、semantic 单边 95% LCB > 0，三者同时满足才判定显著。

## 可复现要素
- **数据集：** Natural Questions（NQ）test split，3,610 题中排除 2,400 开发题后按 SHA-256 选取 400 题；TriviaQA RC-Web validation 中选取 200 题。**未公开发布题目文本和 gold aliases**（仅发布 ID manifests 与哈希）。
- **代码/权重开源情况：** 匿名 artifact 包含 ID-only manifests、sanitize 答案与 judge 单元格、terminal locks、analysis code、bootstrap streams，以及 zero-network verifier（可重新生成所有 drift/transition/stability/overlap/cross-judge 指标）；不含 question text、gold aliases、retrieved passages、provider traces。**论文未提及代码仓库公开链接**。
- **关键超参：** 检索 top-8 文档；每文档最多 1200 token；答案最长 512 UTF-8 bytes；temperature/top-p 未设置（使用 provider default，未记录数值）；bootstrap 50,000 draws；V4-Pro 复现中启用 thinking 并 low reasoning effort。
- **基础设施：** DeepResearchGym [14]、FineWeb [56]；generator deepseek-v4-flash（主）、deepseek-v4-pro（复现）；cross-family judge 使用 OpenAI gpt-5.6-sol。
