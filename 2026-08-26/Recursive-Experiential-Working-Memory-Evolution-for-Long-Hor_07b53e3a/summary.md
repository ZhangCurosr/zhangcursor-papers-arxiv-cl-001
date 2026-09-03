---
title: "Recursive-Experiential-Working-Memory-Evolution-for-Long-Hor"
source: https://arxiv.org/pdf/2608.24876v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:53:48"
field: "LLM Agent 记忆与自改进"
keywords: ["long-horizon agents", "recursive self-improvement", "experiential memory", "working memory", "skill invocation", "agent harness", "failure localization", "state-grounded retrieval"]
innovations: ["EM-WM耦合将技能调用锚定到验证型工作记忆而非历史全文", "结构化执行轨迹支撑组件级失败归因，定位率64.8%", "有界递归与验证门控的 Skill Memory 跨模型迁移"]
benchmarks: ["tau2-Retail", "tau2-Airline", "SkillFlow", "Terminal-Bench 2.1"]
---

# 论文速读：Recursive-Experiential-Working-Memory-Evolution-for-Long-Horizon-Agent-Harnesses

## 一句话总结
本文提出 Recuris，一种递归式经验–工作记忆架构，通过经验记忆（EM）与工作记忆（WM）的耦合实现长程 Agent 的任务状态对齐，并在此基础上以结构化执行轨迹为依据进行组件级故障定位与验证门控的递归记忆演化，从而在冻结底层模型的前提下实现可复用、可转移的长期自改进。

## 研究问题与动机
1. **长程任务的 RSI 困境**：在长程 Agent 任务中，交互历史不断增长导致任务状态被噪声淹没，已存储的技能与当前执行需求之间缺乏可靠对齐机制，使递归自改进（RSI）难以落地。
2. **现有经验记忆检索的失效**：既有方法多从初始指令或完整对话历史中检索技能，随着执行推进，已完成步骤、过期信息与执行噪声混合，检索越来越不可靠。
3. **记忆更新缺乏可定位性**：现有系统主要从最终成败信号中学习，无法区分失败源于技能缺失、状态追踪错误、调用时机不当还是验证器偏差，导致更新粗粒度、易引入回归。
4. **长程失效模式未分解**：长交互中的典型失败并非单一成因，现有工作缺少将失败归因到具体记忆组件并实施定向修复的证据基础。

## 核心贡献（创新点）
1. **提出 EM–WM 耦合架构**：用结构化的工作记忆持续追踪当前任务进展与未完成任务，使经验技能的调用从“基于历史”转向“基于当前状态”，这是本文对长程 RSI 的关键重构。
2. **构建组件级故障定位机制**：通过结构化执行轨迹记录工作状态、调用决策、动作与观察之间的对应关系，使 Meta-Agent 能将失败归因到四个记忆组件之一，定位准确率由仅看结果的 13.0% 提升至 64.8%。
3. **提出验证门控的有界递归记忆演化**：跨任务的演化不是直接重写整个 Agent，而是对受归因组件进行定向补丁，并通过固定验证集上的双重检验（修复源任务且不退化锚点任务）才纳入正式记忆。
4. **在四个长程基准与十个模型上系统性验证**：Recuris 在 37 组模型–基准对中有 35 组提升，将前沿模型带上 SOTA 水平，且优势随交互时域变长而扩大，常见长程失效降幅最高达 80%。

## 方法详解
- **Skill Memory 的四元组表示**：`M_k = (E_k, W_k, ρ_k, C_k)`，其中 E 为可复用技能库，W 为工作记忆规范，ρ 为调用策略，C 为检查器集合；每一轮演化只改动被归因到的组件，其余组件原样复制。
- **结构化工作记忆**：每个目标项记录内容、状态（pending/done/blocked）、支持证据与可选阻断原因；`w_0` 由初始任务状态初始化，`w_t` 显式呈现已完成、未完成与未验证的内容。
- **状态 grounding 的技能调用**：`E_t = ρ_k(x, w_t, e_t, E_k)`，在定义的执行事件（如撰写改写型 tool call 时）触发检索，而非一次性注入全部技能；本文实例化两种 deliverer：Call-time invocation（在改写类调用前触发，将合成 not-executed 结果暂代返回，使技能先于实际执行进入上下文）与 Boundary invocation（在回合边界触发）。
- **证据 grounding 的状态更新**：`w_{t+1} = K(w_t, w̃_{t+1}, c_t)`，其中 `w̃_{t+1}` 由 `U_{W_k}` 提议，`c_t = C_k(...)` 由检查器评估支持度；目标只在 `C_{k,g}=1` 时才由 pending 转为 done，验证器拒绝的假完成也会被记录。
- **轨迹记录的有界递归演化**：每轮生成 `Γ_k = (x_k, {w_t, E_t, a_t, o_t, w̃_{t+1}, c_t, w_{t+1}}_t, y_k)`；Meta-Agent 基于轨迹做失败归因 `D_k = A_fixed(Γ_k, M_k)`，对每个归因组件生成补丁 `Δm_z`，并通过验证门控 `G_fixed` 决定是否接受；外层 LLM、工具集、Meta-Agent 与除 M_k 外的所有机制始终保持固定。
- **测试时单任务适配模式**：同一 Meta-Agent 在同一任务上最多运行 k 次，每次失败后仅获知一条零/一比特反馈而不接触隐藏验证器，据此在经验记忆中写入定向补丁并重试直到成功。

## 实验与结果
- **基准**：τ²-Bench（τ²-Retail 114 题、τ²-Airline 50 题）、SkillFlow（166 题，20 个家族）、Terminal-Bench 2.1（87 题）。
- **模型范围**：从 3B 开源（Granite-4.1-3B、Qwen3.5-4B/9B、GPT-OSS-20B、Qwen3.6-27B/35B）到前沿模型（Gemini 3.7 Flash、GPT-5.6 Sol、Claude Opus 5），部署模型为 doubao-seed-2-0-pro；所有模型均冻结，无权重更新。
- **总体结果**：37 对模型–基准中有 35 对提升；τ²-Retail 上对 GPT-5.6 Sol +17.8pp（58.3→76.1）、Claude Opus 5 +15.6pp（72.4→87.9）；SkillFlow 上 Qwen3.6-27B +16.6pp（42.2→58.7）、Qwen3.6-35B +13.5pp（35.3→48.8）；部署模型提升 +23.3pp 至 81.4%。
- **时域扩展收益**：在最长任务四分位上优势扩大到 +32.2pp；读操作召回在所有模型与长度上稳定在 88–98%，核心差距集中在写操作召回。
- **组件归因对比**：仅看结局 13.0%、原始轨迹 37.0%、结构化轨迹 Γ 64.8%。
- **门控安全性**：4/42 被接受补丁造成锚点任务下降（9.5%），同等条件下重跑同一包则 25.9% 下降；p=0.013。
- **跨模型迁移**：同一包在 GPT-5.6 Sol / Claude Opus 5 / Gemini 3.7 Flash 上分别 +17.8 / +15.6 / +4.8pp；最弱模型 Granite-4.1-3B 在 τ²-Retail 上仍 +13.4pp。
- **单任务适配**：Terminal-Bench 2.1 上尝试预算带来的基线 +26.4pp，而在预算匹配条件下学习额外 +2.3pp；56 题中 learned 子集 avg@4 提升 +4.5pp。

## 相关工作脉络
1. **Experiential memory / skill 类**：Voyager、AWM、ExpeL、Buffer of Thoughts、ReasonFlux、Dynamic Cheatsheet 主要解决“存什么”，对“何时调用”关注不足；本文以 EM–WM 耦合将调用时机绑定到当前验证状态，解决检索与执行的脱节。
2. **State-grounded retrieval 类**：AutoGuide、SGDR 以相似性匹配环境观察来检索，仍以原始观测为条件；本文改为以工作记忆中的进展与阻塞信息为检索键，更贴合执行阶段。
3. **Working memory / state tracking 类**：StateAct、ReflAct、Magentic-One、StateFlow、StructAgent 提供显式状态表示，但状态更新多依赖模型自述或手写规则；本文通过 checker 将状态推进与实际工具返回绑定，并在 StructAgent 的基础上进一步将 verified state 反向用于技能调用。
4. **Memory-evolving / self-improving agent 类**：Memento、EvolveMem、AgentSkill 等以最终评分或自我评判接受更新，归因与补丁单位往往不一致；本文把递归限制在外部化的记忆控制层，归因粒度精细到四个组件之一并施加交叉验证门控。
5. **Skill evolution 相关**：SkillOpt、SkillComposer、MetaSkill-Evolve 聚焦技能本身的学习与组合；本文强调“纪律”（何时调用、如何验证）比“技能内容”更能决定长程性能，且在跨模型迁移实验中证明这一点。
6. **Gated improvement 类**：PACE、SAGE、SSGM 提出新颖性检验、成对显著性测试等更严格采纳标准；本文在它们之上补充了“组件级归因”这一环节，使 gate 不仅判断是否改进，还明确改进哪个部件。

## 局限性与未来方向
1. **跨任务结构的依赖性**：Terminal-Bench 2.1 因无共享结构与工具，13 轮演化均未通过门控，说明跨任务递归在弱结构任务上难以运转，需探索单任务内更高效的定位与更新方式。
2. **验证门控的保守性**：dev 集很小（12 题）导致区间极宽，部分有效补丁被误拒，随后在 86 题 held-out 集上才显现增益，提示 gate 的统计功效依赖划分规模与尝试次数。
3. **单一 Meta-Agent 实现**：论文使用固定 LLM Agent 作为归因与补丁生成器，其语言理解能力可能成为瓶颈；不同 Prompt/编排可能引入系统误差。
4. **SkillFlow 无 held-out 分裂**：该基准上只能验证跨模型迁移而无法验证跨任务泛化，限制了对其 procedural 包真实泛化边界的判断。
5. **未显式讨论安全风险**：尽管论文强调有界递归避免直接改模型，但未展开讨论在开放环境中递归演化的对齐与安全问题。
6. **单模型演化、单源记忆**：虽然增强了可移植性论证，但也意味着不同模型的特性未被充分利用，可能留下跨模型差异带来的未优化空间。

## 研究启发与可借鉴点
1. **状态驱动的调用时机是长程 Agent 的核心杠杆**：论文证明"invocation control > skill content"，将技能注入全部上下文反而更差；可借鉴到本团队的技能库管理中，设计事件驱动的动态检索而非批量 prompt 注入。
2. **结构化轨迹是故障归因的基础设施**：记录 `(w_t, E_t, a_t, o_t, w̃_{t+1}, c_t, w_{t+1})` 使得定位准确率逼近 65%；建议团队在 Agent 系统中内置可审计的执行轨迹以支撑后续 debug 与改进。
3. **验证门控的保守性是安全的必要代价**：dev 集过小时门控会拒正确补丁，但也避免了退化；可在团队项目中采用分层门控——先小样本快筛再大样本精审。
4. **跨模型单源演化具有工程性价比**：一次演化部署到多个模型仍有效，提示可复用“一次学习、多处调用”的记忆资产建设路线。
5. **领域关键组件存在双解离现象**：Airline 域的关键是 write review，Retail 域的关键是 status board；说明架构选择不应是固定的，应在演化过程中根据证据动态调整各组件权重。

## 关键术语表
- **Experiential Memory (EM)**：存储可复用技能的经验记忆，以 agent-skill 格式呈现，作为跨任务积累的持久化知识库。
- **Working Memory (WM)**：在当前任务内动态维护的结构化状态，记录各目标进展与验证证据，作为技能调用的依据。
- **EM–WM Coupling**：工作记忆决定“现在需要什么”，经验记忆提供“对应的可用经验”，两者闭环使调用与验证基于当前状态而非整个历史。
- **Skill Memory M_k**：由 `(E_k, W_k, ρ_k, C_k)` 四个组件构成的可演化记忆控制层，是递归自改进的作用面。
- **State-Grounded Invocation**：在特定执行事件处依据当前工作状态触发检索，而非一次性注入全部技能。
- **Evidence-Grounded State Update**：仅在工具/环境返回支持证据时才将目标标记为完成，拒绝仅凭模型自述的状态推进。
- **Bounded Recursive Evolution**：外层 LLM 与 Meta-Agent 固定，仅在记忆控制层内进行有界的迭代更新，确保改进可追溯、可回滚。
- **Validation-Gated Patch Admission**：补丁须同时通过源任务修复与保留集不退化两重检验才能被采纳。

## 可复现要素
- **数据集**：τ²-Bench（τ²-Retail/τ²-Airline）、SkillFlow、Terminal-Bench 2.1；论文提供了 splits 与开源链接。
- **代码与权重**：代码开源，见 https://github.com/Gen-Verse/Recuris；未提及公开检查点权重。
- **关键超参**：尝试次数 k=4（部分实验 k=16），步数上限 200，连续工具错误上限 10，温度 0；τ²-Retail evolve/dev/test 分 16/12/86，τ²-Airline 分别为 10/15/25 与 11/10/29，SkillFlow 无 held-out。
- **其他**：部署模型 doubao-seed-2-0-pro；Meta-Agent 基于 Claude Code 与 DeepSeek Harness 两套独立实现。
