---
title: "BTS-AgentBench-A-Deterministic-Replayable-Pipeline-from-Read"
source: https://arxiv.org/pdf/2608.27334v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:48:35"
field: "智能体评测基准与工业时序数据"
keywords: ["agent benchmark", "deterministic replay", "tool-use evaluation", "building telemetry", "read-only data", "contract-based scoring", "industrial IoT"]
innovations: ["从只读遥测到多轮 Agent episode 的确定性编译流水线", "带分层相位契约与证据归因的可审计 benchmark", "基于无学习控制器的数据硬化与排除机制"]
benchmarks: ["BTS-AgentBench", "XAI4HEAT"]
---

# 论文速读：BTS-AgentBench-A-Deterministic-Replayable-Pipeline-from-Read

## 一句话总结
本文提出了一种确定性的“遥测日志→多轮智能体评测”自动化流水线，将只读的工业楼宇时间序列数据（BTS、XAI4HEAT）编译为含澄清、目标修订、时间戳策略、质量门控和证据归因的标准化 Agent Benchmark；同一份输入与契约可逐行重放并生成完全一致的 532 行评测集。

## 研究问题与动机
- 工业现场留存大量只读遥测/传感器日志，但现有 Agent 基准缺乏将其系统转化为可执行多轮工具调用任务的构造方法。
- 手工标注或自由文本式构造无法扩展：仅 BTS 即可产生约 2.19M 日窗口候选与 5.99M 配对候选，且通用 Agent 数据缺少设施本地词汇、点位、设备与流标识的落地 grounding。
- 现有工具使用或工业基准多关注对话可运行性/状态依赖，却未从原始时序库出发，将静态计算、证据链、相位化交互约束与确定性评分统一编译出来。
- 需要一种可重放的程序化构造路径，使源计算、 splits、tool-derived 金答案与证据保持不变，同时为后续微调、对照评测提供统一底物。

## 核心贡献（创新点）
- 提出从规范化原始遥测到静态可执行任务、分阶段交互契约、算子面向 episodes、证据支撑验证器的确定性构建流水线。与现有工作相比，本文以“源库→工具仓库→任务→交互协议”的顺序化编译为核心，而非直接构造对话。
- 发布 BTS-AgentBench：532 行、九大家庭（点位消歧/日均值/24h相对均值/窗口均值/窗口对比/窗口排名/时间戳精确查询/最近时间戳查询/质量门控报告）的可评测基准，并内建澄清、目标修订、时间戳策略、质量承诺与证据归因等约束。
- 设计带预检与构造排除控制的验证/审计框架：编码式 contract preflight 零发现，确定性 controller 在构造完成后对 532 行全部未命中（0/532），并提供独立两次 raw-to-episode 重放与所有逻辑导出一致。
- 在 XAI4HEAT 上完成可移植性研究：经语料适配后同样路径产出 204 episodes，其 41 行 held-out 测试集中 controller 仍 0/41 完成，GPT-5.5 保留执行 41/41 达成。

## 方法详解
- **静态可执行任务层**：将 BTS 元数据与原始流归档匹配，归一化 site/point/equipment/location，筛选可落到原始历史的 point 作为 tool-ready；构建 DuckDB 支持的只读 tool store（原始流索引、逐流质量统计、日/周/月聚合、日历特征、流预览）。
- **家族化任务生成与保留规则**：对每类任务施加过滤——点位消歧要求同站同类至少两流且目标元组唯一；聚合候选剔除低于 corpus 10th 覆盖率上限与高于 site/class 99.5th 绝对均值上限者；对比需正差且≥组中位数；排名需≥3 流且顶次差≥组中位数；时间戳任务限定内点或典型相邻间隔；质量任务采用基于 corpus 分位数的固定周覆盖率与间隙规则。
- **从静态任务到 Agent Episodes**：通过模式分支（direct-answer/clarify-time/implicit-nearest/quality-rationale 等）隐藏必要字段形成澄清槽，并按有限交互语法组装澄清、初始回答、目标修订、时间戳策略、质量承诺、理由说明与证据归因相位；各相位由类型化模板渲染为算子面向询问，但所有工具调用与金答案均来自只读执行而非语言模型生成。
- **类型化契约表示与级联修订**：每个 episode 记为 $C=(Q,\Phi,A,E,V)$，其中 $\phi_i=(f_i,g_i,R_i)$；各阶段按固定顺序应用谓词 $P_k$ 与可执行更新 $U_k$（公式 2），保证源计算保留、执行接地、话语对齐三耦合条件。
- **审计与重放**：stage 历史、contract preflight、phase/cardinality/ground-truth/时序窗口/证据一致性校验构成多重门禁；相同 raw archive、选行 identity、predicates 与模板下二次/三次构建可逐对象重现 released train/dev/test 行。
- **确定性控制器与接受规则**：由有界解析器与显式站/流状态构成的无学习 controller 完成全协议探查；若 controller 能完成某行则该行被修复或删除，最终 release 中 controller 完成 0/532（0/89 test），以此作为构造排除信号。

## 实验与结果
- **数据集与拆分**：BTS-AgentBench 共 532 行，split 为 train 356 / dev 87 / test 89；XAI4HEAT 适配后共 204 行，split 为 train 132 / dev 31 / test 41。
- **评估基线**：三款前沿模型 GPT-5.5、Gemini 3.1 Pro、Claude Opus 4.7；统一 deterministic user simulator、只读工具、单工具每轮、固定停止协议与无 LLM judge 的打分器。
- **BTS 主要结果（test 89 行）**：GPT-5.5 79/89（88.8%）、Gemini 3.1 Pro 71/89（79.8%）、Claude Opus 4.7 58/89（65.2%）；controller 在 test 上 0/89 完成。高精度集中在直接查找与聚合（日均值、相对 24h 均值、最近时间戳、质量门控多为 90–100%），交互敏感型较弱：pairwise 6/10、5/10、5/10；rank 8/10、5/10、4/10；point disambiguation 8/10、5/10、6/10。
- **诊断得分（test）**：GPT-5.5 Final 0.978、Evidence 0.955、Phase 0.949、Task 0.965、Protocol 86/89；Gemini 3.1 Pro Final 0.921、Evidence 0.955、Phase 0.939、Task 0.957、Protocol 81/89；Opus 4.7 Final 0.903、Evidence 0.933、Phase 0.875、Task 0.927、Protocol 81/89。
- **可移植性**：XAI4HEAT 41 行 test 上 GPT-5.5 保留执行 41/41 达成；controller 0/41 完成；五大家庭复用（day mean 60、relative 24h mean 60、timestamp value 35、timestamp nearest 35、window mean 14）。

## 相关工作脉络
- **API-Bank / ToolSandbox / τ-bench / ACEBench**：侧重可运行对话、状态依赖或细粒度多轮错误维度；本文起点为已采集的只读遥测语料，同时编译静态任务与算子交互层，并以证据与契约对齐为评测核心。
- **ITBench / AssetOpsBench / ReAct Meets Industrial IoT**：面向结构化运维流程或语言代理访问工业遥测；本文聚焦“只读日志→边界明确、可确定性计分”的 episode 编译与 hardening。
- **BTS / Brick**：提供建筑时序与可移植元数据基底；本文在此基础上派生可执行的分析任务与 agent episodes。
- **SUPER**：从研究仓库编译可执行完成任务；本文同样采用“可执行工件出发”的思路，但把起点改为遥测与元数据，并通过确定性控制器排除易被机械完成的数据行。

## 局限性与未来方向
- 评测范围限定于只读建筑遥测的搜索、聚合、对比、排名、时间戳报告性与质量感知报告，未覆盖写入控制、安全关键执行、维护规划与长程排障。
- 算子询问为确定性有界模板，对话多样性与语言噪声被刻意压缩，难以完全模拟真实运营人员交互。
- 最终 532 行未做系统性领域专家审计；所报告验证主要针对遥测落地性与可执行契约闭合。
- 可扩展至更广泛工具使用域需要新增任务家族、语料特定交互契约与新控制器审计套件；对事件日志/状态转移日志等结构差异更大的语料仍需额外适配。

## 研究启发与可借鉴点
- **可重放基准的构造理念**：固定 raw archive + 选行 identity + 确定性子过程，可实现跨次重建的 JSON-level 完全一致；适合需要严格复现的评测发布流程。
- **分相位契约评分**：将“最终答案正确”拆成澄清-初始回答-目标修订-时间戳/质量策略-承诺-证据链条的全相位核查，能暴露仅看 final score 掩盖的错误（如缺失 stream grounding、证据链断裂、承诺不匹配）。
- **控制器排除法作为数据硬化信号**：用无学习、有界 deterministic controller 探查协议可完成性，并将可被机械绕过的行剔除或修复，提升基准区分度。
- **语料适配只停留在上游映射**：BTS→XAI4HEAT 的案例表明，只要完成 site/stream/point/equipment/timestamp/value 的结构化映射，下游 builder、auditor、scorer 可字面复用。
- **工具仓库物化思路**：把点库存、质量统计、日/周/月聚合、日历特征与预览统一沉淀为 read-only store，可作为多家族任务的共享后端，降低重复检索与不一致风险。

## 关键术语表
- **BTS-AgentBench**：基于 BTS 建筑时序数据构建的 532 行多轮 Agent 基准，覆盖九类遥测分析任务与多种交互约束。
- **Deterministic replay**：在固定原始归档、元数据契约与构造阶段顺序下，跨独立构建可逐字段复现发布的 benchmark artifact。
- **Tool store**：基于 DuckDB 的只读后端，包含流索引、质量统计、多尺度聚合与日历特征，供任务与 episode 统一使用。
- **Static executable task**：携带来源计算、工具调用集合、金答案、证据与验证器的离线可执行任务行。
- **Canonical episode schema**：把任务转化为若干类型化相位（澄清、初始回答、修订、时间戳策略、质量承诺、理由/证据）的有界交互协议。
- **Contract preflight**：发布前对相位/回合数、证据标识、runtime 对齐、rendered gold 可接受性等进行编码检查的零发现门禁。
- **Construction-exclusion controller**：无学习、基于有界解析与显式状态的全协议确定性探查器，用于排除/修复可被机械完成的行。
- **Grounding / Evidence attribution**：要求模型最终答案与推理绑定到具体流/点位/时间戳/聚合证据，并在后续 evidence follow-up 中被核验。

## 可复现要素
- **数据集**：BTS（公开多年度建筑时间序列，带标准化元数据）；XAI4HEAT（公开区域供热 SCADA 子站数据）。
- **代码/权重**：代码、artifacts 与重放报告已在 https://github.com/kjy7567/BTS-AgentBench 开源；单命令 wrapper 仅绑定与验证发布阶段的构造步骤，不含任务生成与打分逻辑。
- **关键超参/环境**：论文未明确列出多数学习超参（本工作以规则化编译为主）；运行环境声明为 Python 3.11.11、DuckDB 1.5.0、NumPy 1.26.4、pandas 3.0.1、PyArrow 23.0.1、RDFLib 7.6.0。
- **评估设置**：温度 0、每轮单工具、12 基础回合上限、180s 超时；不同模型通过 OpenAI 直连/OpenRouter 接入，output cap 512 或 1536、seed 支持程度各异。
- **拆分与保留**：BTS train/dev/test 为 356/87/89；XAI4HEAT 为 132/31/41。
