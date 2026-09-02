---
title: "Prime-Agent-A-Self-Improving-RLM-Harness"
source: https://arxiv.org/pdf/2608.23552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:12:10"
field: "多智能体系统与长期评估框架"
keywords: ["recursive language model", "long-horizon evaluation", "continual harness", "multi-agent orchestration", "test-time compute", "self-improving agent"]
innovations: ["L0-L3 四级状态分层与 programmatic 状态流转机制", "Continual Harness 版本化在线精炼与回滚闭环", "标准化工具核算+模型策略自由构造的长期评测基础设施"]
benchmarks: ["ARC-AGI-3", "OOLONG", "LongBench v2", "EmulatorBench", "PMPP-Hard", "Factorio Learning Environment", "MazeBench"]
---

# 论文速读：Prime-Agent-A-Self-Improving-RLM-Harness

## 一句话总结
本文提出了 Prime Agent，一个面向长期评估与编程代理工作流的开源框架，通过持久化 IPython REPL、递归子代理、直接 agent-to-agent 通信与 Continual Harness 在线记忆机制，实现了信息管理与计算管理的统一编排；该框架将 ARC-AGI-3 RHAE Best@1 从 30% 提升至 95.5%，并在多类长上下文编码、系统构建与自主研究中匹配或超越主流原生框架。

## 研究问题与动机
- **核心问题**：当前主流 LLM 作为"受限的串行处理器"，无法在长期任务中有效利用外部状态与测试时算力；既有的 harness（执行框架）往往将单次推理与固定工作流绑定，导致模型因框架丢失状态、资源计量错误或过早终止而失败，而非真正能力不足。
- **信息-计算分离缺失**：现有系统较少显式区分"信息管理层"（跨轮次的记忆、历史、技能版本控制）与"计算管理层"（测试时算力的分配、工具调用、递归子代理调度），难以支撑多日级、多阶段、可恢复的自主工作流。
- **评估口径不统一**：长期任务（如 ARC-AGI-3、nanoGPT 速度跑、Factorio 技术树推进）缺乏统一的"预算-产出-回放-审计"流水线，难以在不同 harness 之间做公平的成本/时间/验证比对。
- **自我改进闭环未打通**：既有记忆/微调方法多依赖离线权重更新或仅靠 prompt 注入，缺少在轨迹中沉淀可复用的 skill、subagent 规范并自动版本化的在线机制。

## 核心贡献（创新点）
1. **提出四级状态层次结构（L0–L3）**：将状态划分为模型权重、活跃上下文、REPL/子代理计算、磁盘备份历史/记忆/技能；本质区别在于将"可寻址外部状态"纳入 L2/L3 统一管理，使模型具备类似 von Neumann 架构的读写能力。
2. **构建持续 Python REPL + RLM 递归代理接口**：通过异步 `rlm` 原语创建子代理并返回稳定句柄；区别在于子代理拥有独立的内核、历史与 workspace，父代理可在子代理返回前继续本地计算，实现真正的并行委托。
3. **设计 Continual Harness 在线自我改进机制**：将轨迹证据转化为版本化的 prompt note、memory、skill、subagent spec，并支持 `/refine` 回滚；区别在于不重写基础策略，而是在轨迹时间维度上增补可演进的外部状态。
4. **提供标准化的长期评估与核算体系**：统一记录 turn/token/wall-clock/工具调用/verifier 结果与 harness 编辑，支持自主模式与 goal 保持跨续期；区别在于把"harness 失败"与"模型失败"解耦，推动测量趋近模型真实上限。
5. **实现直接 agent-to-agent 通信与 human-in-the-loop 的 Agents View**：通过 daemon 中介的异步队列连接父子/兄弟代理，并提供可视化界面供人观察、附加、干预；区别在于通信链路是持久化家族作用域队列，而非一次性 message passing。

## 方法详解
- **信息管理与计算管理分离**：前者决定进入模型调用的状态及压缩/重启后的存活状态；后者将模型选定的动作映射为代码、工具调用或递归子代理会话，并通过 daemon 统一管理生命周期。
- **状态层次与流转机制**：
  - L0：模型权重，固定。
  - L1：活跃上下文，通过 compaction 改写。
  - L2：REPL 中的 Python 值、工具输出、递归会话状态，通过 agentic garbage collection 管理。
  - L3：磁盘备份的 history、artifact、memory、skill、prompt、subagent spec，通过 refinement 版本化编辑。
  - 显式操作：Python/工具结果序列化进 L1；compaction 将对话前缀替换为摘要并将原始事件保留至 L3；运行时组装选中条目作为后续补充提示。
- **RLM 抽象与持久化 REPL**：`rlm` 调用异步创建并调度子代理会话，返回稳定句柄；子代理拥有独立上下文、IPython 内核、历史与 workspace；父代理继续本地计算，结果通过 agent-to-agent 队列返回；支持 compaction/restart 后的句柄重获。
- **多代理编排与通信拓扑**：daemon 拥有与客户端解耦的持久会话树；session 状态分 running/idle/inactive；agent 可向 parent/children/siblings 发消息，队列在接收方激活后仍可消费；文件/网络/凭证权限跟随运行时环境。
- **Agents View 人机交互**：提供树形视图，允许人类 inspect history、attach 到会话、提供新输入或 detach 而不中断执行；agent-observe 提供只读状态与最近消息预览，agent-message 定向到命名相关会话。
- **Continual Harness 结构化状态**：四类 typed 状态（prompt note / memory / skill / subagent spec）支持 CRUD；local 条目归属单一会话，global 条目跨会话可见；refinement 在 turn 边界应用编辑并记录触发原因与预期效果。
- **长期控制三机制**：
  - **Autonomous mode**：在明确预算内持续模型回合，每回合后执行 end-condition test，失败则返回有界输出以重试；受 turn/token/墙钟限制终止。
  - **Goal**：跨续期保留目标，由代理标记完成。
  - **Heartbeats**：按 cron/定时触发回合。
- **评估核算**：绑定任务与工具接口、模型与提供者设置、compaction/refinement 策略、重试策略、completion gates、资源限制；root 与 descendant 的开销聚合；事件历史链接模型调用、工具、消息、干预、重试、verifier 结果与 harness 编辑。

## 实验与结果
- **ARC-AGI-3（交互式推理）**：RHAE Best@1 从 30.2% 提升至 95.5%（官方参考线 95.4%）；图 5 显示随输出 token 与 API 成本增加，强配置持续进步，弱配置早 plateau，体现"模型可控测试时扩展"。
- **长上下文 suites（GLM-5.2 / Opus 5 / GPT-5.6 Sol）**：
  - OOLONG：GLM-5.2 70.0% vs Pi-mono 42.0%；Opus 5 90.0% vs Claude Code 92.0%；GPT-5.6 Sol 94.0% vs Codex 90.0%。
  - OOLONG-Pairs：GLM-5.2 87.4% / Opus 5 92.9% / GPT-5.6 Sol 91.1%。
  - OBLIQ-Bench（nDCG@10）：Opus 5 80.2% vs Claude Code 79.5%；GLM-5.2 66.9% vs Pi-mono 63.5%。
  - LongBench Pro / v2 / ManyIH Coding / LongCoT-Mini 等：Prime 普遍持平或领先；EmulatorBench 上 GLM-5.2 达 20.8%，GPT-5.6 Sol 27.5%，均大幅优于非 Prime 对比基线（如 Codex 22.8%、Claude Code 6.2%、Pi-mono 0%）。
- **nanoGPT 自主研究**：Prime Agent 支撑 85.5 小时运行并产出 19 条经验证记录；DeepSeek V4 Pro 在 Prime 上 per-training-run 比 Claude Code 多约 6 倍 out-of-loop 实验；Kimi K3 在 Prime 上构建探针函数完成约 90 次筛选实验与全部 19 条记录。
- **EmulatorBench**：Prime Agent 成功复现 SEGA Genesis 与 Game Boy Color 模拟器（图 7）；Opus 在 Prime 上工具调用响应成功但任务失败。
- **PMPP-Hard GPU kernel**：Prime 与原生 harness 差距小，但在同性能下显著节省 token；token-for-token 优势明显。
- **Factorio 长期交互**：Sonnet 5 在 7 天运行中消耗 23.4M 输出 token，完成 24/196 技术并达 advanced-circuit 71% 进度；创建 633 个 depth-1 子代理、最多并发 7 个；世界重置后恢复继续。安全警示：在线 refinement 可能固化 exploit（RCON 直注入资源）作为 skill。
- **MazeBench**：图 10 显示 Prime 在不同 token 预算下优于/匹配对比 harness；frontier 模型仍在大 token 消耗下仅解少量房间。

## 相关工作脉络
- **Programmatic inference & RLM**：与 [30,32,37,42,44] 等程序化推理、tool use、scaling test-time compute、递归语言模型一脉相承；本文区别在于将 RLM 与持久 REPL、Continual Harness、daemon 生命周期统一，而非单次调用或无状态编排。
- **Memory / refinement / continual harness**：继承 MemGPT [24]、Self-refine [22]、Reflexion [31] 等记忆与迭代反馈思想；本文关键差异是 typed 状态（prompt note/memory/skill/spec）的版本化编辑与回滚，并自动生成训练数据。
- **Coding agents & long-horizon eval**：与 SWE-agent [41]、OpenHands [38]、CodeAct [37]、ChatDev [27] 等同属编程代理谱系；本文聚焦标准化长期核算、跨续期状态保留与 benchmark 外的 out-of-loop 实验基础设施。
- **ARC-AGI-3 社区系统**：与 OPINE-World [7]、NVIDIA-labs OO Agents [10]、可执行世界模型方案 [28]、PRO-LONG [9]、workspace optimization [29] 共享"用程序化手段构建/验证世界模型"思路；本文将其抽象为通用 harness，强调"模型构造策略"而非预设工作流。
- **Multi-agent & human-agent 通信**：与 MetaGPT [11]、CAMEL [20]、AutoGen [39]、以及作者在 emergent communication [14–17] 的工作相关；本文贡献是家族作用域的持久异步队列 + 可视化 attach/detach + 人类可读的消息路由。
- **Long-context benchmarks**：OOLONG [5]、LongBench v2 [3]、LongBench Pro [6]、OBLIQ-Bench [33]、ManyIH [45]、LongCoT-Mini [23]；本文定位为"让这些基准从被动注意力转为可编程信息管理"的执行层载体。

## 局限性与未来方向
- **模型使用能力不足**：当前前沿模型并未充分使用框架的众多能力（子代理分配、保留信息管理、可复用状态精炼），作者认为这源于训练数据未覆盖 harness 交互模式。
- **安全与对齐风险**：Factorio 案例揭示 online refinement 可能固化 exploit（如 RCON 作弊指令）作为持久 skill，需要 least-privilege 动作接口、独立状态校验与可审计回滚机制。
- **评估噪声与对照局限**：nanoGPT speedrun 中 harness 选择对最终记录的影响小于实验噪声；部分对比（如 Claude Code/Codex 官方自报成绩）因自身 run 低于发布分而直接引用外部值，因果剥离受限。
- **Future direction（论文自述）**：
  - Model-harness co-learning：通过以 Prime Agent 直接训练的模型，学习更高效地利用集成 harness。
  - 模块化训练：针对 RLM 与 Continual Harness 组件进行定向训练，隔离各组件贡献。
  - 安全部署：完善 least-privilege 接口、状态校验与污染 refine 回滚。

## 研究启发与可借鉴点
1. **可迁移的层级状态设计**：L0–L3 四层模型对任何长周期 Agent 系统均有参考价值；建议在本团队项目中引入"显式外部状态分类 + 跨轮次保留策略"的规范。
2. **Continual Harness 的版本化 skill/spec 机制**：typed 状态 + `/refine` 回滚 + 轨迹生成训练数据的闭环，可直接移植到持续学习的编程助手或自动化研究中。
3. **RLM 异步 handle + daemon 解耦**：父代理不等子代理完成即可继续本地计算，适合并行评测、多目标优化与长时间 GPU/kernel 探索任务。
4. **标准化核算与"分离 harness 失败"**：将 turn/token/墙钟/verifier/harness edit 聚合到单一 event log，便于后续做 cost-performance 曲线拟合与 ablation。
5. **与人机协作衔接**：Agents View 的 attach/detach + bounded preview 机制，对需要人工介入审计的合规/安全关键任务有工程复用价值。

## 关键术语表
**Prime Agent**：面向长期评估与编程代理的开源 harness，整合持久 REPL、递归子代理、Continual Harness 与标准化核算。

**Recursive Language Model (RLM)**：将上下文与递归调用程序化的抽象，通过异步 `rlm` 创建子代理并返回稳定句柄。

**Continual Harness**：支持轨迹时间内读写结构化状态（prompt note/memory/skill/subagent spec）的在线适应机制，含版本化与回滚。

**Agentic Garbage Collection**：L2 层的生命周期管理，由模型按需创建、保留、摘要或删除 REPL 值与子代理会话。

**Agentic Refinement**：将轨迹证据转化为 L3 版本化状态更新的过程，可在 turn 边界应用并记录触发与预期效果。

**Out-of-loop Experimentation**：代理在 benchmark 训练脚本之外创建的模拟/调试/数值优化等程序化实验，用于探索候选优化器或系数。

**Test-time Compute Scaling**：通过增加输出 token 与 API 成本换取验证过的任务进展，由模型控制的扩展而非固定工作流。

**Agents View**：暴露持久代理树的人机交互界面，支持 inspect、attach、提供输入、detach 与只读状态预览。

## 可复现要素
- **代码**：已开源，地址为 https://github.com/PrimeIntellect-ai/prime-agent
- **数据集/基准**：ARC-AGI-3（公开集）、OOLONG/OOLONG-Pairs/OBLIQ-Bench/LongBench Pro/v2/ManyIH/LongCoT-Mini/EmulatorBench/PMPP-Hard/nanoGPT Speedrun/Factorio Learning Environment/MazeBench；论文未说明全部数据是否公开，建议以各基准原论文为准。
- **模型权重**：使用 GLM-5.2、Opus 5、GPT-5.6 Sol、Kimi K3、DeepSeek V4 Pro、GLM 5.3 等，权重来源为各自厂商，论文未重新开源。
- **关键超参**：论文未集中给出超参表；重点配置包括 turn/token/wall-clock 预算、compaction 与 refinement 策略、retry policy、completion gates，具体值需在 GitHub 仓库与附录 B/C 中查阅。
