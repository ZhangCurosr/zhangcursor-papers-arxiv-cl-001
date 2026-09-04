---
title: "JIT-Agent"
source: https://arxiv.org/pdf/2608.25593v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:46:36"
field: "LLM Agent 系统架构"
keywords: ["Agent Harness", "Just-in-Time Generation", "Harness Intelligence", "Program Synthesis", "Multi-objective RL", "Test-time Adaptation"]
innovations: ["提出Harness Intelligence概念，将agent harness构造/修复/进化定义为可训练的独立能力维度", "四模块协议h=(M,P,A,F)将异构agent harness统一为可组合、可协议验证的结构化程序表示", "Evo-GDPO多目标解耦进化算法，同时优化reward/latency/cost三通道并推动archive frontier持续前移"]
benchmarks: ["DeepSearchQA", "xBench-DS", "AgentIF-Oneday", "PinchBench", "DeepPlanning-Shopping", "DeepPlanning-Travel", "OfficeBench", "OdysseyBench", "BrowseComp-Plus"]
---

# 论文速读：JIT-Agent

## 一句话总结
JIT-Agent 提出了一种"Model-as-a-Harness"新范式，训练一个轻量元代理模型在推理时为任意现成 LLM 实时合成任务适配的 Agent harness（内存-规划-动作-能力编排四模块协议），并通过定制、修复与进化三阶段训练实现"harness 智能"，使弱 backbone（如 DeepSeek-V4-Flash）在多项基准上超越更强模型（GPT-5.6）。

## 研究问题与动机
1. **Agent 能力由模型与 harness 共同决定**：基础模型的推理/行动能力与 harness（内存管理、规划策略、动作协议、工具编排）紧密耦合；错误 harness 可使强模型失败，强 harness 需模型能理解执行才能生效。
2. **AOT（前置时间）harness 优化的局限**：现有方法在部署前预编译一个持久性 harness，假设其在未来任务中泛化；面对异构任务（宽搜索、终端交互、深度研究、代码编写）难以用单一 harness 覆盖。
3. **harness 设计本质上是实例依赖的**：不同任务结构需要不同先验（DAG 并行、ReAct 串行、工作内存等），手动设计无法扩展，亟需可学习的 JIT 合成能力。
4. **训练这样的系统面临三大挑战**：适应性（匹配生成 harness 到任务）、可靠性（保证可执行并能在合成失败时修复）、可进化性（将执行反馈转化为更强的未来 harness）。

## 核心贡献（创新点）
1. **提出"Harness Intelligence"概念并形式化**：首次将 harness 构造、修复与进化定义为可训练的独立能力维度，与模型权重 scaling 正交。
2. **引入四模块可组合 harness 协议 h = (M, P, A, F)**：将异构 agent harness 统一分解为 Memory/Planning/Action/Capability 四个协议兼容模块，使 harness 生成从自由文本合成变为结构化程序组装，大幅压缩搜索空间。
3. **设计三阶段训练流水线（Customize → Repair → Evolve）**：Stage I 基于教师蒸馏学习任务条件化定制；Stage II 将失败轨迹转化为有界修复监督（最多 2 轮）；Stage III 提出 Evo-GDPO 在线进化算法，同时优化 reward、latency、cost 三个解耦目标。
4. **构建 HarnessFactory 代码库与 seed bank B₀**：基于统一协议重新实现 13 种代表性 agent 框架（ReAct, Plan-and-Execute, Flash-Searcher, ROMA 等），为训练提供异构源材料。
5. **大量实验验证 harness 智能的通用增益**：JIT-Agent + DeepSeek-V4-Flash 在 DeepSearchQA 超越 GPT-5.6 (+9.1)，GLM-5.2 提升 +20.2；JIT-generated harness 在 token 消耗与 API 成本上全面优于 Claude Code、OpenCode 等成熟固定 harness。

## 方法详解

### 4 模块协议框架
每个 harness $\mathbf{h} = (\mathbf{M}, \mathbf{P}, \mathbf{A}, \mathbf{F}) \in \mathfrak{M} \times \mathfrak{P} \times \mathfrak{A} \times \mathfrak{F}$，运行时依赖顺序为 $\mathbf{M} \to \mathbf{P} \to \mathbf{F} \to \mathbf{A}$：
- **Memory**：将不可变事件历史 $\pmb{\xi}_{<t}$ 与可变控制器状态 $\mathbf{s}_t$ 压缩为视图 $\mathbf{v}_t = \mathbf{M}(\pmb{\xi}_{<t}, \mathbf{s}_t) \in \mathcal{V}$
- **Planning**：基于任务 $\tau$、状态、视图生成局部指令 $\mathbf{d}_t = \mathbf{P}(\tau, \mathbf{s}_t, \mathbf{v}_t) \in \mathcal{D}_{\mathrm{dir}}$（无显式规划器时返回空指令 $\mathbf{d}_\varnothing$）
- **Capability Orchestration**：从能力注册表 $C_\tau$ 中筛选子集 $C_t = \mathbf{F}(C_\tau, \mathbf{s}_t, \mathbf{v}_t, \mathbf{d}_t) \subseteq C_\tau$
- **Action**：更新状态并执行动作 $(\mathbf{s}_{t+1}, e_t) = \mathbf{A}(\mathbf{s}_t, \tau, \mathbf{v}_t, \mathbf{d}_t, C_t)$

### Stage I：任务条件化定制
- 数据准备：用冻结强教师 $q_\phi$ 在 seed bank $\mathcal{B}_0$ 中采样 3 个参考 harness 生成合规示例 $\mathbf{h}^{\mathrm{teach}}$
- 联合优化两个目标：
  - **SFT 生成损失** $\mathcal{L}_\mathrm{I}^\mathrm{gen}$：标准自回归负对数似然，学习协议合规结构
  - **偏好损失** $\mathcal{L}_\mathrm{I}^\mathrm{pref}$：基于奖励严格改善且不劣化效率轴（latency/cost）的偏好对，参考 anchored RL 目标
  - 总损失：$\mathcal{L}_\mathrm{I} = \mathcal{L}_\mathrm{I}^\mathrm{gen} + \lambda_\mathrm{pref} \mathcal{L}_\mathrm{I}^\mathrm{pref}$

### Stage II：修复学习
- 将 Stage I 中协议/执行校验失败的 harness $\widetilde{\mathbf{h}}^{(0)}$ 及其诊断报告 $\mathbf{g}^{(0)}$ 纳入训练
- 教师模型在 patch 空间 $\mathcal{P}$ 中生成有界修订 $\Delta^{(k+1)}$，经 Apply 后重新校验，最多 2 轮内恢复可执行即保留
- 训练目标：模仿教师修复轨迹的 conditioned LM 损失 $\mathcal{L}_\mathrm{II}$，使模型学会使用执行反馈进行小步高杠杆修复

### Stage III：Evo-GDPO 进化学习
- **候选生成**：从当前策略 $p_{\theta_\mathrm{old}}$ 采样 $G$ 个候选 harness $\mathbf{h}_i$，每个经校验后执行
- **解耦奖励建模**（三通道独立归一化后合并）：
  - $R_i^\mathrm{rew} = r_i + \lambda_\mathrm{evo}[r_i - b_r]_+$（超越 incumbent 的奖励奖励）
  - $R_i^\mathrm{lat} = \mathbb{I}[r_i \geq b_r][b_\ell - \bar{\ell}_i]_+$（同等奖励下的 latency 节省）
  - $R_i^\mathrm{cost} = \mathbb{I}[r_i \geq b_r][b_\kappa - \bar{\kappa}_i]_+$（同等奖励下的成本节省）
  - 先 channel 级 Z-score 归一化，再 batch 级归一化：$\widehat{A}_i^\Sigma$
- **策略更新**：PPO-style clipped objective，以 $\widehat{A}_i^\Sigma$ 为优势信号驱动更新
- **Bank 保守更新**：候选仅当匹配或超越当前 reward frontier 且严格改进至少一个维度时才加入 $\mathcal{B}_n$

### 推理模式
- **Static Inference**：并行生成 $n$ 个 harness，选择最优一个执行（test-time scaling）
- **Streaming Inference**：跨任务序列携带经验，执行结果用于评估是否更新 bank，不注入当前 rollout 反馈，不更新模型参数

## 实验与结果

### 数据集与基准
9 个 benchmark 覆盖 4 类任务：
- **Deep Research**：BrowseComp-Plus、DeepSearchQA、xBench-DS
- **Daily Work**：AgentIF-Oneday、PinchBench
- **Planning**：DeepPlanning-Shopping、DeepPlanning-Travel
- **Workspace**：OfficeBench、OdysseyBench

### Backbone 模型
Qwen3.6-27B（训练基底）、GLM-5.2、DeepSeek-V4-Flash/Pro、Qwen3.6-Flash/Plus、Mimo-V2.5-Flash/Pro

### 基线
- **模型基线**：Qwen3.7-Plus、GPT-5.6、Gemini 3.1 Pro/3.5 Flash、Kimi K2.7 Code
- **Harness 基线**：Claude Code、Codex、OpenCode、Hermes、NanoBot

### 关键结果
| 配置 | DeepSearchQA | OdysseyBench | DeepPlanning-Travel |
|---|---|---|---|
| JIT-Agent + DeepSeek-V4-Flash | **85.1**（超 GPT-5.6 +9.1）| **73.0**（超 GPT-5.6 +4.3）| 61.3 |
| JIT-Agent + GLM-5.2 | **93.9** | 78.7 | **83.0**（+20.2 vs vanilla）|
| JIT-Agent + DeepSeek-V4-Flash（平均）| — | — | 9-benchmark avg +8.8 pts |
| JIT-Agent + GLM-5.2（平均）| — | — | 9-benchmark avg +7.7 pts |

- **对比固定 harness**（Table 4）：JIT-Agent 在所有 6 组设定中 token 消耗与 API 成本最低；DeepSeek-V4-Flash + JIT 在 DeepSearchQA 以 $0.066/case 达 85.1，相对 NanoBot（$0.131）成本降低 49.6% 且性能 +4.7 分
- **泛化性**：跨 3 个模型族（DeepSeek V4, Qwen 3.6, Mimo V2.5）共 24 组对比，JIT harness 平均超越 ReAct +7.6 分
- **流式进化**：Streaming JIT 在 DeepPlanning-Shopping/Travel 和 OfficeBench 上累积准确率均超过 Static JIT

## 相关工作脉络
1. **AOT Harness 搜索类**（AutoHarness, Meta-Harness, AHE）：在经验分布上预搜索持久 harness，无实例合成、无学习修复、无在线进化；JIT-Agent 将优化摊销到推理时按需生成。
2. **AOT + 测试时编辑类**（Adaptive AH, TTHE, RHI）：在预优化 harness 基础上做 test-time 编辑，但非模型化生成，无结构化修复学习；JIT-Agent 直接生成完整实例级 harness。
3. **Harness-R1**：具备 Learned repair 与 Online evolution，但仍基于 AOT 预编译框架做局部编辑；JIT-Agent 从根源上以协议约束的程序合成替代自由文本编辑。
4. **Harness 模块化工作**（HarnessX, Code-as-Agent-Harness）：提出模块分解思想，但聚焦人工工程而非可学习合成；本文在此基础上定义四模块协议并训练生成器。
5. **RLVR/代码 Agent 基线**（DeepSeek-V4, GLM-5.2 等）：以模型 scaling 为核心提升 agent 性能；本文证明 harness 智能可作为正交维度带来同等量级的能力增益。

## 局限性与未来方向
1. **四模块协议的表达力边界**：生产级 harness（如 Codex、Claude Code）包含更丰富的机制，本文刻意简化以证明核心命题，但可能丢失某些复杂任务所需的高阶编排能力。
2. **JIT-Agent-27B 自身规模**：作为 meta-agent 仍需 27B 参数，在资源受限场景下部署成本尚需评估。
3. **对 backbone 遵从性的隐含假设**：harness 有效性依赖于底层模型能正确理解和执行生成的协议代码，若 backbone 能力不足则增益受限。
4. **进化阶段的 online 依赖**：Stage III 需要在线执行反馈，部署时需要一定冷启动期积累 bank；对小样本/一次性的任务场景效果待验证。
5. **论文自述未来方向**：① 模型-harness 协同设计（foundation model 与执行结构联合训练）；② 生产系统可采用"稳定核心 + 可替换组件"的混合架构；③ 将 harness 合成/修复/进化能力内化到基础模型中。

## 研究启发与可借鉴点
1. **四模块协议化分解**：将 agent harness 形式化为 (M, P, A, F) 四元组并建立统一接口，是可将搜索空间从自由文本降维到结构化程序合成的有效思路，可迁移到任何需要 programmatic harness 的场景。
2. **修复学习（Repair Learning）范式**：将失败轨迹（含诊断报告）转化为有界多轮修复监督数据，而非丢弃或仅用 reject sampling，对任何代码/程序生成任务均有借鉴价值。
3. **Evo-GDPO 多目标解耦优化**：将 reward/latency/cost 三个异质信号独立归一化后再加权合并，避免数值尺度失衡；此技巧可推广至多目标 RL 或 preference learning 场景。
4. **Streaming Bank 经验传承**：静态生成（每任务独立）vs 流式累积（bank 持续增长）的对比设计清晰展示了"元知识积累"的价值，可在多轮 agent 部署或连续任务流中复用。
5. **Model-as-a-Harness 新范式**：将 harness 工程从人工 AOT 设计转变为可训练的 JIT 生成能力，为"agent 架构搜索"开辟了 model-based 替代路线，可与 NAS、prompt optimization 等领域交叉。

## 关键术语表
**Harness Intelligence**：指智能体运行 scaffold 的构造、修复与进化能力，是独立于模型权重的可训练、可迁移、可累积的能力维度。
**Just-in-Time (JIT) Harness**：在推理时为具体任务按需生成适配 harness，而非预先编译固定 artifact；与 AOT 范式相对。
**Four-Module Protocol h = (M, P, A, F)**：将 agent harness 统一分解为 Memory（历史压缩为视图）、Planning（视图生成指令）、Capability Orchestration（指令条件筛选能力）、Action（状态更新与动作发射）四个协议兼容模块。
**HarnessFactory**：基于统一四模块协议重新实现的 13 种代表性 agent 框架代码库，构成 seed bank $\mathcal{B}_0$ 供训练采样。
**Evo-GDPO（Evolutionary Group-Decoupled Policy Optimization）**：Stage III 的在线进化目标，对 reward/latency/cost 三通道分别归一化后合并为 PPO-style 优势信号，推动候选 harness 超越 archive frontier。
**Streaming Inference**：跨任务序列将执行反馈持续累积到 harness bank 的推理模式，不更新模型参数但不断进化参考经验池。
**Instance-dependent Harness**：harness 不仅取决于任务领域，还取决于具体任务实例的结构特征，是 JIT 范式区别于 AOT 的核心动机。

## 可复现要素
- **数据集**：9 个公开 benchmark（BrowseComp-Plus, DeepSearchQA, xBench-DS, AgentIF-Oneday, PinchBench, DeepPlanning, OfficeBench, OdysseyBench），均有公开发布
- **代码**：GitHub https://github.com/bingreeky/JIT（论文声明开源）
- **权重**：Hugging Face https://huggingface.co/JIT-Agent（论文声明开源）
- **网站**：https://bingreeky.github.io/JIT-site
- **训练基底模型**：Qwen3.6-27B
- **关键超参**：论文未详细给出（$\lambda_\mathrm{pref}, \beta_\mathrm{pref}, \lambda_\mathrm{evo}, w_\mathrm{rew}/w_\mathrm{lat}/w_\mathrm{cost}, \epsilon_\mathrm{clip}, G$ 等未列出具体数值）
