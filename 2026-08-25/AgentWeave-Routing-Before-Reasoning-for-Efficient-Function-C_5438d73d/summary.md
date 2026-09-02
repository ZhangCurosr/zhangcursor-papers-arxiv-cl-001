---
title: "AgentWeave-Routing-Before-Reasoning-for-Efficient-Function-C"
source: https://arxiv.org/pdf/2608.23078v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:37:09"
---

# 论文速读：AgentWeave: Routing Before Reasoning for Efficient Function Calling in Tool-Rich Language Models

## 一句话总结
提出 AgentWeave，一种确定性预推理路由层，通过在语言模型推理前按需构建有界候选函数集，使固定模型在 BFCL V4 多函数任务上的原生成功率从无提升至 12.5%，同时显著削减可见工具数、输入 token 与本地推理延迟。

## 研究问题与动机
- 随着工具目录膨胀，function-calling 模型需处理更多 schema、消耗更多 prompt token，并在高度相似或无关的候选中艰难区分；现有评估 pipeline 通常将完整候选目录作为固定输入，忽视了“推理前筛选可见工具”这一系统变量。
- 主流优化路径聚焦于模型自身改进（如数据合成、函数掩码、检索训练），但未将候选空间构建作为独立于推理的阶段进行控制实验。
- 简单的上下文压缩策略（如随机裁剪或纯语义 top-K）虽能降低 token 与延迟，却可能破坏候选集内部的竞争结构，导致下游原生成功率归零。
- 现有研究缺乏对“路由召回质量”与“下游决策质量”之间解耦关系的系统验证，难以判断高召回是否等价于高成功。

## 核心贡献（创新点）
- 形式化“推理前工具路由”为 function-calling 的独立系统阶段，提出候选空间构建与语言模型推理解耦的架构视角。与现有工作本质区别：不改模型权重与解码逻辑，仅通过前置编排层改变模型输入的候选上下文。
- 设计 AgentWeave 确定性路由层，融合策略/作用域硬过滤、需求感知路由与能力分组，显式约束模型可见候选集规模。与 Hammer/ToolACE 等强化下游模型能力的路线相比，本文保持模型冻结，聚焦输入侧上下文构造。
- 构建并冻结 48 题 BFCL V4 多函数任务的对比实验协议，严格控制下游模型与生成设置不变，隔离候选呈现效应。与一般基准评测相比，本文强调“路由压力研究”而非官方榜单冲刺，并提供可审计的冻结 artifact。
- 揭示“候选召回率 ≠ 原生成功率”的现象，证明路由后的候选集竞争结构与分布几何比单纯的高召回更重要。与常规检索论文只报告 Recall@K 不同，本文同步报告下游 native success 并指出召回优势的误导性。

## 方法详解
- **问题形式化**：给定自然语言请求 $x$、可用工具目录 $\mathrm{F}=\{f_1,...,f_n\}$、固定 function-calling 模型 $M$，传统做法为 $y=M(x,\mathrm{F})$；AgentWeave 引入路由函数 $R$，使预测变为 $y=M(x, \mathrm{F}')$，其中 $\mathrm{F}'=R(x,\mathrm{F})\subseteq\mathrm{F}$，且 $|\mathrm{F}'|$ 受有界预算约束。
- **确定性范围与策略过滤**：优先基于角色、租户、权限、部署环境、管辖权、产品面或能力标签等硬约束剔除绝对不可用的候选，无需语义解释；若过滤后目录已足够小，可跳过后续语义路由。
- **需求感知路由**：对剩余候选，从请求中提取需求信号，匹配候选描述、能力分组与提供商归属，并在有界预算内保留最相关子集；路由接口与具体模型无关，可适配本地/宿主模型或 Provider-neutral 适配器。
- **阶段分离与溯源设计**：将候选空间构建、模型推理、执行授权划分为独立阶段，路由层记录来源身份、配置、候选计数、被滤工具、模型可见集、选定函数与执行结果；明确路由仅优化模型视野，安全授权须在模型选择后、执行前以 fail-closed 机制独立实现。
- **路由评估诊断**：除 native success 外，额外报告 mean original-candidate recall 与 all-original-candidates-retained rate，用于刻画上游召回质量与压缩程度的解耦关系。

## 实验与结果
- **数据集与协议**：48 个 BFCL V4 multiple-function 任务（与前期 12 题 v5 pilot 零重叠），每个任务置于固定的 16-tool pressure 环境中。
- **评估基线**：All tools（全量 16 工具）、Random top-8（确定性随机裁剪）、Semantic top-8（嵌入/相似度裁剪）。
- **下游模型**：MadeAgents/Hammer2.1-1.5b，所有条件下权重、生成设置与评估器完全冻结。
- **主要结果**：
  - AgentWeave 原生成功率 6/48（12.5%），其余三组基线均为 0/48；配对优势 +12.5 pp，10,000 次重采样 paired bootstrap 95% CI 为 [+4.17, +22.92] pp，exact McNemar p=0.03125。
  - 效率收益：相对 All tools，AgentWeave 平均可见工具从 16.00 降至 4.77（-70.18%），输入 token 从 124,821 降至 47,805（-61.70%），本地推理延迟从 55.92s 降至 27.43s（-50.95%）。
  - 路由诊断悖论：Semantic top-8 召回更强（原始候选召回 86.81%，全保留率 66.67%），但 native success 仍为 0；AgentWeave 召回略低（76.91% / 54.17%）却唯一成功，证明单纯高召回无法解释下游表现。
- **结论**：在冻结压力下，改变候选集构造能显著改变固定模型的行为；压缩幅度与候选竞争结构的共同作用决定最终成功率。

## 相关工作脉络
- **Toolformer [1] / ReAct [2]**：奠定 LLM 与外部工具协同的基础范式；本文不改变推理-行动交织结构，而是将候选集选择前置为独立路由阶段。
- **Gorilla [3] / ToolLLM [5]**：侧重大规模 API 检索与神经检索训练；本文主张检索/路由与最终选择解耦，强调路由层无需微调下游模型即可生效。
- **Hammer [6] / ToolACE [7]**：通过函数掩码、合成数据与训练增强提升模型鲁棒性；本文固定模型，仅修改输入侧候选呈现，属系统编排策略而非模型增强路线。
- **BFCL [8]/V4**：提供可执行与 AST 级评测框架；本文复用其多函数类别与原生评估器，但注入 16 工具压力并冻结评估快照，定位为路由压力研究而非官方榜单成绩。
- **API-Bank [4]**：关注规划、检索与调用的端到端评估；本文进一步拆解为“召回诊断 vs 原生成功”，指出高召回不一定转化为正确调用。

## 局限性与未来方向
- **规模与类别局限**：仅 48 题单一多函数类别，不足以表征广义 function-calling 性能，需更大零重叠 replication。
- **单模型依赖**：仅使用 1.5B 轻量模型，该模型可能对候选压力更敏感；大模型或不同训练目标的模型收益未知。
- **路由非 SOTA 检索**：Semantic top-8 在召回指标上优于当前路由，说明 AgentWeave 路由器本身仍有较大优化空间。
- **因果分解不足**：未单独剥离压缩幅度、干扰项几何、候选排序等路由特征的真实贡献，需 ablation 与候选组成实验。
- **未来方向**：高召回混合路由（确定性过滤+稠密/稀疏检索+重排序）、K 预算曲线 sweep、多模型/多类别验证、标准 BFCL 原生数据集成、对抗目录与授权安全评估、级联失败分解（P(R)×P(S|R)×P(A|S,R)×P(E|A,S,R)）。

## 研究启发与可借鉴点
- **“路由-推理”解耦范式**可直接迁移至多 Agent 编排、RAG 路由、长上下文截断等场景，将候选集构建作为独立可优化、可审计的系统组件。
- **冻结对比与分阶段验证设计**极具参考价值：先以小样本 pilot 生成假设，再冻结协议做零重叠 replication，有效避免对基准的隐式过拟合与后验调参。
- **双轨诊断指标**（召回质量 + 下游决策质量）值得引入团队评测体系，避免以 Recall@K 或工具数量单一指标替代端到端 success。
- **成本-性能三轴量化框架**（可见工具数、输入 token、推理延迟）可作为后续工具调用系统的标准化性能基线。
- **失败分类学**（Missing required candidate / Partial coverage / Schema competition / Argument error / Model failure / Over-compression）可直接复用于 Agent 工具调用链路的定位与归因。

## 关键术语表
- **AgentWeave**：确定性预推理路由层，负责在 LLM 推理前按需过滤、分组并裁剪工具候选集。
- **Routing-pressure protocol**：在 BFCL 任务基础上强制注入固定数量（16）的竞争工具，以放大不同路由策略的差异效应。
- **Native BFCL success**：经 BFCL 原生可执行/AST 评估器判定的端到端函数调用正确率。
- **Candidate composition**：模型实际可见候选工具集合的构成与竞争结构，区别于单纯的数量压缩。
- **Recall@K / All-retained rate**：路由后保留的原始基准候选比例与全部原始候选均保留的任务比例，用于衡量上游召回质量。
- **Schema competition**：多个描述/参数相似的候选并存时导致的模型误选或干扰现象。
- **Frozen evidence / anti-tuning control**：首次评分后即锁定实验配置与样本，任何修改需启用新标识与新样本，防止后验调参污染结论。
- **Stage-level provenance**：记录路由配置、候选计数、过滤与选定结果的可追溯元数据，支持安全边界划分与失败定位。

## 可复现要素
- **数据集**：BFCL V4 multiple-function 任务子集（48 题），与前期 v5 零重叠；BFCL/Gorilla 仓库与评估器公开，论文未提及额外闭源数据。
- **代码/权重**：AgentWeave 源码与 artifact 开源（Apache-2.0），地址 github.com/sauravsingla/agentweave；下游模型 MadeAgents/Hammer2.1-1.5b 公开可下载。
- **关键超参**：压力候选集大小 16；All tools=16，Random/Semantic top-8=8，AgentWeave 动态路由预算；下游模型与生成设置完全冻结；统计采用 10,000 次配对 bootstrap 与精确 McNemar 检验。
- **版本锁定**：BFCL/Gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`，Scored source head `ca6ff084da4fe5c670421b99e7ad413650e60c33`，Workflow run `31983991285`，Artifact `9275082744`，外部 API 支出 $0。

<!--META
{"keywords": ["Function Calling", "Tool Routing", "Agent Orchestration
