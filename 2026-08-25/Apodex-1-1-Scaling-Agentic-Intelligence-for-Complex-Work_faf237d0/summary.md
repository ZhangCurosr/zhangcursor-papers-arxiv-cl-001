---
title: "Apodex-1-1-Scaling-Agentic-Intelligence-for-Complex-Work"
source: https://arxiv.org/pdf/2608.23283v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:37:43"
field: "智能体系统与长程任务"
keywords: ["agentic intelligence", "environment scaling", "multi-agent coordination", "reinforcement learning", "long-horizon tasks", "verification", "working capability"]
innovations: ["提出环境扩展与智能体协调扩展两个互补缩放面，将工作完成能力作为智能体基本单位", "设计非对称验证与显式任务板，提升长周期多智能体协作的可追溯性与可修正性", "开发 PIVOT-RL 通过回溯定位关键决策点实现高效长轨迹强化学习"]
benchmarks: ["FrontierFinance", "FrontierScience-Research", "APEX-Agents", "GDPVal", "Humanity's Last Exam", "DeepSearchQA", "SWE-bench Verified", "Terminal-Bench 2.1", "BioMysteryBench", "FrontierSearchBench", "FrontierResearchBench", "HDS6"]
---

# 论文速读：Apodex 1.1: Scaling Agentic Intelligence for Complex Work

## 一句话总结
Apodex 1.1 提出通过"环境扩展"和"智能体协调扩展"两个互补维度来训练智能体在复杂长周期任务中的"工作能力"，使模型能够在文件、搜索、代码和并行协作环境中可持续地推进任务、恢复失败并交付可验证成果；在专业工作、金融、科研、数学推理和编码等主流基准上达到前沿性能带，且 35B 参数 Mini 版本仍能保持竞争力。

## 研究问题与动机
1. **推理与执行的鸿沟**：通用语言模型推理能力迅速提升，但在真实长周期任务中仍需持续与文件、信息源、可执行代码交互，维护状态、恢复失败并交付可验收产物，现有工作仅优化孤立响应质量。
2. **环境覆盖率不足**：现有智能体训练多将工具作为静态文本描述附加于 prompt，缺乏多样化、高保真、可验证的文件/搜索/代码世界，导致模型在真实交互中频繁失败。
3. **多智能体协调的规模变量不明确**：既有系统通过对话协调或角色分工获得增益，但核心缩放变量应是"随着目标演化可协调的有用工作量"而非单纯增加智能体数量或采样数。
4. **评估仅观测结果投影**：多数 benchmark 只给出最终答案得分，难以反映状态连贯性、证据忠实度、失败恢复和交付完整性等过程维度。

## 核心贡献（创新点）
1. **将"工作完成能力"定义为智能体智能的基本单位**：提出任务合约形式化与工作轨迹评价，将"完成的工件"而非"一次性回答"作为智能能力的衡量单元，与仅优化 answer quality 的 prior work 形成根本差异。
2. **提出 Environment Scaling 作为独立缩放面**：系统构建并扩充文件、搜索、代码三类可执行世界，强调"forward cheap, inverse expensive"的不对称构造与独立 verifier，不同于以往仅增加 prompt 数量的做法。
3. **提出 Agentic Coordination Scaling 并通过 Agent Team 1.1 实现**：将分解、委派、异步结果整合、动态重规划训练为模型策略行为，以显式任务板和 staged return 替代隐式上下文传递，区别于 AutoGen/MetaGPT 等固定角色对话方案。
4. **设计 AgentOS 作为统一执行基座**：提供持久化工作区、三层命名空间、外部任务板、异步干预队列与受控交付合约，将环境交互与多智能体协调收敛到同一执行契约下，弥补既有系统接口碎片化问题。
5. **开发 PIVOT-RL 训练方法与 HDS6 过程评测框架**：PIVOT-RL 通过回溯定位关键决策点并构造局部修正任务，提升长轨迹信用分配效率；HDS6 从六类过程能力对执行轨迹打分，突破仅看最终分数的评估瓶颈。

## 方法详解
1. **任务合约形式化**：定义 $\mathcal{E} = (\mathcal{W}, W_0, q, \mathcal{A}, \mathcal{T}, \Omega, \mathbf{B}, D, V_D)$，其中 $q$ 为目标、$D$ 为交付合约、$V_D$ 为任务级 verifier；模型在每个决策步从 $a_t \in \mathcal{A}$ 中选择动作，环境返回观察 $o_{t+1}$，轨迹结束由 $S_D = V_D(W_0, W_H, \tau_H)$ 判定交付结果。
2. **环境扩展三大家族**：
   - **File worlds**：围绕权威关系与转换，跨嵌套目录与异构格式重建业务逻辑，verifier 基于可独立推导的数值或记录 provenance。
   - **Search worlds**：围绕发现与证据对齐，要求查询重构、来源分诊、跨源 reconcile 和停止纪律，gold 对象包含源集合、claim-evidence alignment 与显式不确定性。
   - **Code worlds**：围绕可执行转换与沙箱验证，复用基础镜像 + 任务级仓库装配，要求 fail-to-pass/pass-to-pass 双测约束与 verifier hardening。
   - 难度坐标：文件/搜索任务采用 $\rho_{\text{acq}}(\mathcal{E}) = \frac{N_{\text{cand}} + N_{\text{hop}}}{B_{\text{tool}}}$ 衡量获取压力；代码任务关注依赖深度、状态转移深度、测试可观测性与失败到验证信号的距离。
3. **Agent Team 1.1 核心设计**：
   - **显式任务板**：主代理将分解写入外部 task board，条目含目标、依赖、解决状态与返回结果；board 由协调器语义更新，运行时异步 job 状态独立维护，避免 plan 在 context 压缩后丢失。
   - **异步人类干预**：执行期间接受用户消息 $u_t$，策略学习区分"保留 $(q,D)$ 的澄清/优先级变更"与"实质性改变目标的续约任务"，保持因果连续性。
   - **非对称验证**：verifier 仅接收特定 claim、支撑证据与交付约束进行针对性攻击（反例搜索、独立源三角校验、格式合规），输出可操作的修正信号，避免完整重做造成的上下文污染。
   - **自适应 Max Team Effort**：仅对薄弱、争议或承重 claim 追加独立调查波次，缩放变量为"有用的协调工作量"。
   - **证据驱动的综合阶段**：双 pass——先构建 claim–evidence graph 并标注承重/ corroborated/disputed/unresolved，再由 writer agent 在相同工具与交付约束下生成最终产物。
4. **AgentOS 执行基座**：
   - 工作区状态 $W_t = (F_t, Q_t, C_t, I_t, G_t, K_t)$，包含文件、证据、可执行状态、artifact 索引、依赖图与协调控制状态。
   - 三层命名空间：`/inputs`(只读)、`/workspace`(计算与中间产物)、`/outputs`(交付根)，外加可选 `/shares`(用户文档库只读挂载)。
   - 受控交付：单一出版 lease、精确 manifest、scoped write 策略与 baseline reconciliation，防止并发/过期/不完整交付。
   - 上下文压缩：按推理端 token 报告触发 tiered compaction，软/硬预算边界与 bounded report recovery。
5. **训练流程**：
   - **统一 SFT**：覆盖推理、工具、搜索、文件、编程、数学、科研金融、专业交付与多智能体协调，过滤无效交互、状态不一致与未交付轨迹。
   - **PIVOT-RL**：回顾式轨迹定位识别 consequential pivot（偏离生产力策略、证据不足、工具误用、假设未修正），保留有效前缀并构造含简短 hint 的局部续任务；hint 不进入推理时输入。异步优化处理异构轨迹耗时差异。

## 实验与结果
- **专业工作**：APEX-Agents 38.5（Agent Team），GDPVal 78.8% win rate；ReAct 基线分别为 34.4 与 69.5。
- **金融**：FrontierFinance 54.3（Agent Team，本表最强），YC-Bench 表现与 DeepSeek V4-Pro 接近。
- **科学研究**：FrontierScience-Research 63.3（Agent Team，本表最强），BioMysteryBench Human-difficult 35.3。
- **通用推理与深度搜索**：HLE 56.1，DeepSearchQA F1 92.4。
- **数学**：IMO 2025 得分 36.5（超过金牌阈值 35），IMO 2026 30.5（超阈值 29），USAMO 2026 26.5（超阈值 25）；ProofBench Basic 96.7%、Advanced 63.3%。
- **代码**：SWE-bench Verified 77.7，Terminal-Bench 2.1 70.8。
- **内部搜索**：FrontierSearchBench 平均 69.1（Agent Team，超越最强对比）。
- **内部科研交付**：FrontierResearchBench Pass Rate 12.4%（Agent Team），所有对比系统均低于 21%。
- **35B Mini 效率**：FrontierFinance 50.2、FrontierScience-Research 51.7、APEX-Agents 27.7，达到选定前沿系统性能带；相较 Apodex 1.0 mini 分别提升 10.2、6.7、3.5 分。
- **HDS6 过程分析**：Initial Decomposition +1.3、Final Verification +0.8 为最大单项目增益，证据忠实度与假设管理能力整体提升。

## 相关工作脉络
1. **ReAct / Toolformer**：奠定 reasoning-action-observation 循环与自监督工具调用学习基础；本文在此基础上将工具调用嵌入长期状态化工作流，并以 verifier 约束交付。
2. **WebArena / WorkArena / OSWorld / APEX-Agents**：提供可执行环境评估范式；本文与之同源但将环境同时用作训练合成、回放与评估的统一尺度，强调跨 family 的可验证交付。
3. **SWE-bench / SWE-agent / Terminal-Bench**：指向仓库与命令行层面的软件工程师具；本文扩展至更广义的"可执行转换 + 测试 + 沙箱"代码世界，并与科研/金融场景统一训练。
4. **AutoGen / MetaGPT / ChatDev / AgentVerse**：多智能体对话与角色组织先验；本文强调以显式任务板、非对称验证与自适应 team effort 取代固定对话图。
5. **Search-R1 / RAGEN / ROLL**：智能体 RL 与异步训练基础设施；本文 PIVOT-RL 进一步将信用分配聚焦于 consequential pivot，结合环境扩展提供训练信号来源。
6. **Chain-of-Verification / process reward / self-refine**：验证与过程监督相关；本文将其嵌入 persistent execution state 并与 artifact provenance、sandbox execution 联合使用，形成三层验证体系。

## 局限性与未来方向
1. **内部 benchmark 未完全开源**：FrontierSearchBench 与 FrontierResearchBench 作为关键能力验证尚未公开，限制社区独立复现与横向对比。
2. **复杂科研端到端交付通过率仍低**：FrontierResearchBench 最高仅 12.4%，说明当前系统在跨域可执行科研流水线上的可靠性仍有较大提升空间。
3. **AgentOS 的持久化边界**：当前 task board 与 Agent Bus 为 run-scoped，进程重启恢复与文件系统历史回滚未被契约覆盖，尚需联合版本化支持。
4. **非对称验证依赖 verifier 质量**：尽管比完整重做更窄，若目标 claim 本身涉及高度主观或不可检验命题，仍存在验证信号不足的风险。
5. **未来方向**：扩展环境覆盖与保真度、深化长程协调策略、改进 hierarchical trace 上的信用分配、建立真实失败→任务构造→训练→评估的闭环。

## 研究启发与可借鉴点
1. **"forward cheap, inverse expensive"构造原则**：可用 latent state/reference program 廉价生成任务世界，再用不可见约束逼迫模型恢复；这一不对称性值得迁移到工具使用与代码生成的数据合成管线。
2. **非对称验证的工程化落地**：将验证任务限定为"攻击特定 claim + 支撑证据 + 交付约束"，可获得比完整交叉检查更强的可操作反馈；建议在本团队研究中对 factuality-critical 子任务引入同类 verifier。
3. **显式任务板替代隐式上下文**：把分解决策写入共享 board、分离 coordinator 语义字段与 runtime 异步 job 状态，可有效缓解长轨迹 context 压缩导致的 plan 丢失；可直接借鉴到多步骤 agent 系统中。
4. **PIVOT-RL 的信用分配思路**：通过 hindsight 定位 pivotal failure/recovery 点并构造短 horizon 续任务，显著降低长轨迹 RL 的计算浪费；可复用于本团队的长程 tool-use 后训练。
5. **过程维度的评测框架**：HDS6 从状态连贯、证据忠实、假设管理、边界与失败推理、工具执行、自我修正六类打分，提示团队在面向科研/专业工作的 agent 评估中不应只盯 final score。

## 关键术语表
- **Working Capability**：智能体在长周期内通过工具交互、状态维护和失败恢复实现可验证交付的能力。
- **Environment Scaling**：通过扩充文件、搜索、代码三类可执行世界的多样性、保真度和可验证性来扩大策略的学习分布。
- **Agentic Coordination Scaling**：将任务分解、委派、异步整合与动态重规划训练为模型策略行为，并以 Agent Team 在运行时实现。
- **AgentOS**：为单/多智能体提供持久化工作区、命名空间隔离、异步干预队列与受控交付合约的执行基座。
- **Task Board**：协调器维护的外部显式任务板，记录子目标、依赖、状态与返回结果，脱离 LLM 上下文独立存在。
- **Asymmetric Verification**：验证任务比生成任务更窄，仅针对具体声明及其证据进行检查，避免完整重做造成的上下文污染。
- **PIVOT-RL**：基于 hindsight 定位轨迹中的 consequential decision 点，构造带 hint 的局部续任务进行强化学习的训练方法。
- **HDS6**：评估智能体执行过程的六类能力框架，涵盖状态连贯、证据忠实、假设管理、边界与失败推理、工具执行与自我修正。

## 可复现要素
- **数据集**：公开基准包括 FrontierFinance、FrontierScience-Research、APEX-Agents、BioMysteryBench、Humanity's Last Exam、DeepSearchQA、SWE-bench Verified、Terminal-Bench 2.1；内部 FrontierSearchBench、FrontierResearchBench 及多专业领域文件世界未公开。
- **代码**：GitHub https://github.com/ApodexAI/FrontierAgent（论文声明）。
- **模型权重**：HuggingFace https://huggingface.co/collections/apodex/apodex-11（论文声明）。
- **关键超参**：论文未详细披露 PIVOT-RL 的学习率、epoch、batch size 等训练细节。
