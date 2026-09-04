---
title: "JIT-Agent"
source: https://arxiv.org/pdf/2608.25593v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:47:33"
---

# 论文速读：JIT-Agent

## 一句话总结
JIT-Agent 是一个专用于即时生成任务自适应 agent harness 的元代理模型，通过三阶段训练（定制→修复→进化）学习可执行的 four-module 运行协议，使任意 off-the-shelf agentic LLM 在推理时获得任务特化的操作脚手架，以更低成本超越当前前沿模型。

## 研究问题与动机
- **Agent 能力由模型 + harness 共同决定**：底层 foundation model 产生推理与动作，harness 决定记忆组织、子目标规划、工具/技能暴露、验证与恢复机制；错误 harness 会浪费强模型能力。
- **AOT harness 优化难以覆盖异构任务分布**：现有方法（AutoHarness、Meta-Harness、AHE 等）采用 ahead-of-time 假设，预先搜索一个持久 artifact 以期泛化，但在 deep research、terminal、coding、workspace 等异构场景下需不同 harness prior。
- **Instance-dependent harness 设计尚未被有效探索**：同一模型在不同任务甚至不同 instance 上最优 harness 结构差异显著（如 DAG 规划 vs. 串行 ReAct vs. 递归委托），亟需学习化、可迁移的即时生成机制。
- **Harness engineering 尚未被纳入模型 scaling 体系**：当前 agent 能力提升仍主要依赖模型规模与推理 compute，harness 设计仍停留在手工/搜索层面，缺乏可训练、可累积的智能维度。

## 核心贡献（创新点）
- **提出 Harness Intelligence 新范式**：将 harness 能力形式化为可学习、可迁移、可累积的独立维度，与模型 weight scaling 正交，首次确立 "模型作为 harness 生成器" 的 AgentIMprovement 路线。
- **设计四模块协议化 harness 表示 h = (M, P, A, F)**：把异构 harness 统一为 memory/planning/action/capability-orchestration 四个互操作模块，保持最小可执行契约的同时保留内部扩展性，使 harness 成为 machine-generatable artifact。
- **引入三阶段训练流水线**：Stage I 用教师生成协议合规 harness 示例进行 SFT + 偏好优化；Stage II 将编译/运行时失败转化为有界 repair 轨迹进行 supervised learning；Stage III 提出 Evo-GDPO，在在线 archive frontier 竞争中联合优化 reward/latency/cost 三元目标。
- **实证 harness 智能可弥补模型 gap**：JIT-Agent + DeepSeek-V4-Flash 在 DeepSearchQA (+9.1)、PinchBench (+8.7)、OdysseyBench (+4.3) 超越 GPT-5.6；GLM-5.2 + JIT-Agent 在 DeepPlanning-Travel 提升 +20.2，证明 operational scaffold 优化可回收大部分原本依赖 backbone scaling 的收益。
- **开源完整代码、权重与 HarnessFactory**：GitHub 与 Hugging Face 同步开源，包含 13 种代表性 scaffolds 的统一实现与可复现训练脚本。

## 方法详解
### 1. 四模块协议化表示与执行语义
- **模块化分解**：每个 harness h ∈ H_Π 表示为 h = (M, P, A, F) ∈ M × P × A × F，四模块在共享 frozen executor π_ψ 下按 M→P→F→A 顺序依赖执行。
- **运行时方程**：
  - 视图构建：v_t = M(ξ_<t, s_t)
  - 局部指令：d_t = P(τ, s_t, v_t)
  - 能力编排：C_t = F(C_τ, s_t, v_t, d_t) ⊆ C_τ
  - 动作发射与状态更新：(s_{t+1}, e_t) = A(s_t, τ, v_t, d_t, C_t)
  - 内核解释：o_t = Exec(e_t; C_t)，ξ_{≤t} = ξ_<t ⊕ (s_t, e_t, o_t)
- **协议兼容性**：固定协议 Π 规定模块 schema、接口、生命周期、验证规则与共享执行语义，去除语言/运行时的偶然差异，保留区分不同 harness 的关键操作选择。

### 2. Stage I：任务条件化定制（Customization）
- **数据准备**：强教师 q_φ 接收 (τ, Π, C_τ, E_τ)，其中 E_τ 是从任务类型匹配的 seed bank B_0^(d(τ)) 采样 3 个参考 scaffold，生成协议合规 harness h^teach；仅保留通过 Valid_Π 与执行检查的样本入 D_I。
- **SFT 目标**：L_I^gen(θ) = −E[Σ log p_θ(y_j^teach | y_<j^teach, c_τ)]，教授从任务 + 参考到协议 compliant harness 的直接映射。
- **偏好优化目标**：比较同 backbone 下 (h^+, h^−) 的 (r, ℓ, κ)，定义严格 Pareto 偏好 h^+ ≻_τ h^− ⟺ r^+ > r^− ∧ ℓ^+ ≤ ℓ^− ∧ κ^+ ≤ κ^− ∧ (ℓ^+ < ℓ^− ∨ κ^+ < κ^−)，并用 Reference-anchored DPO-style 损失 L_I^pref 联合优化。

### 3. Stage II：有界修复学习（Repair）
- **失败转化**：Stage I 产出无效 harness h̃^(0) ∉ H_Π^exec，附带诊断报告 g^(0)（含编译错误、接口不匹配、tool-call failure、运行时异常）。
- **教师修补轨迹**：每步教师从 patch space P_z 提议 Δ^(k+1)，Apply 确定性地更新 h̃^(k+1) = Apply(h̃^(k), Δ^(k+1))；仅保留 K^* = min{k ∈ {1,2} : Valid_Π(h̃^(k)) = 1} 存在的有界修复轨迹。
- **监督目标**：L_II(θ) = −E[Σ_{k=0}^{K^*-1} log p_θ(Δ^(k+1) | c_τ, {(h̃^(j), g^(j))}_{j=0}^k}]，教会模型基于诊断报告进行高杠杆局部修改。

### 4. Stage III：Evo-GDPO 在线进化
- **上下文构建**：在线轮 n 采样 τ，从当前 bank B_n 检索 E_{τ,n}，与当前 incumbent (b_r, b_ℓ, b_κ) 组成 c_{τ,n}。
- **多目标奖励归一化**：
  - R_i^rew = r_i + λ_evo [r_i − b_r]_+
  - R_i^lat = 𝟙[r_i ≥ b_r][b_ℓ − ℓ̄_i]_+
  - R_i^cost = 𝟙[r_i ≥ b_r][b_κ − κ̄_i]_+
  - 三通道分别 group-level 标准化后加权合并，w_rew > w_lat + w_cost 确保 reward 主导。
- **PPO-style 更新**：L_III^Evo-GDPO(θ) = −E[(1/G) Σ (1/|h_i|) Σ min(ρ·Â, clip(ρ, 1−ε_clip, 1+ε_clip)·Â)]，结合 Stage II checkpoint 的 token-level KL 惩罚稳定训练。
- **保守存档**：仅当候选在 reward 上不劣于当前 frontier 且严格 Pareto 改进某一维度时才纳入 B_n。

## 实验与结果
- **评测基准**：9 个 benchmark 分属 4 类任务——
  - Deep Research：BrowseComp-Plus (accuracy)、DeepSearchQA (F1)、xBench-DS (accuracy)
  - Daily Work：AgentIF-Oneday (normalized rubric)、PinchBench (avg score)
  - Planning：DeepPlanning-Shop (cart match)、DeepPlanning-Travel (constraint score)
  - Workspace：OfficeBench、OdysseyBench (task success rate)
  所有指标均映射到 0–100 尺度。
- **Backbone**：GLM-5.2、DeepSeek-V4-Flash/Pro、Qwen3.6-Flash/Plus、Mimo-V2.5-Flash/Pro（训练基于 Qwen3.6-27B）。
- **基线**：GPT-5.6、Gemini 3.1/3.5 Pro/Flash、Kimi K2.7 Code、Claude Code、Codex、OpenCode、Hermes、NanoBot、Vanilla ReAct。
- **主要结果**：
  - **同 backbone 稳定增益**：GLM-5.2 + JIT 九基准平均 74.1 → 81.8 (+7.7)；DeepSeek-V4-Flash + JIT 六基准平均 66.7 → 75.5 (+8.8)；最大单项提升 DeepPlanning-Travel +20.2 (62.8 → 83.0)。
  - **超越前沿模型**：JIT + DeepSeek-V4-Flash 在 DeepSearchQA (+9.1)、PinchBench (+8.7)、OdysseyBench (+4.3) 超过 GPT-5.6；九基准中有八个列的 Top-1 由 JIT-equipped 模型摘得。
  - **对抗固定 harness**：控制 backbone 后，JIT-Agent 在 6 组设定中 4 组取得最高性能（DeepSeek-V4-Flash + DeepSearchQA 85.1 vs. NanoBot 80.4；x 63.8 vs. Claude Code 66.9 代价更低）；Token 消耗与 API 成本在所有 6 组最低，平均成本降低 36.0%。
  - **跨模型泛化**：24 组 model-pair × benchmark 中 JIT harness 全部优于 ReAct baseline，平均 +7.6 分（DeepSeek V4 +10.2、Qwen 3.6 +4.0、Mimo 2.5 +8.6）。
  - **Streaming evolution 有效**：跨任务流累积准确率与静态 JIT 拉开差距，且 cost/tool-call 轨迹不单调递增。
- **局限提示**：DeepPlanning-Travel 是唯一未被 JIT 模型夺魁的列（GLM-5.2 + JIT 83.0 vs. GPT-5.6 84.9，差 1.9 分）。

## 相关工作脉络
- **HarnessX / Code-as-Agent-Harness**：前者将 runtime 分解为 prompts/tools/memory/control-flow，后者组织为 interface/mechanism/multi-agent-scaling 三层；本文在此基础上进一步压缩为最小四模块协议 h = (M, P, A, F)，聚焦可学习性而非完备性。
- **AutoHarness / Meta-Harness / AHE (AOT search)**：三类方法均通过 offline 搜索或 test-time editing 优化持久 harness artifact，本文定位为 "instance synthesis + learned model + online evolution" 的互补路线，强调 inference-time 合成而非预先搜索。
- **HARNESS-R1 / TT HE / RHI (test-time editing)**：虽学习从失败轨迹 edit harness，但仍是 AOT 起点上的局部修补；本文 Stage II 直接学习有界 repair 轨迹，Stage III 实现 archive frontier 驱动的在线进化。
- **ROMA / AOrchestra / Recursive LMs**：递归多 agent 架构被本文的 F 模块 (delegation/routing) 与 A 模块 (recursive execution) 统一捕获，验证四模块表示对生产级架构的涵盖力。
- **GRPO / GDPO (multi-reward RL)**：Evo-GDPO 将 Group-Decoupled Policy Optimization 引入 harness generation，关键扩展在于引入 archive incumbent 竞争信号与在线 Pareto-frontier 更新，而非仅组内相对排名。

## 局限性与未来方向
- **协议表达力刻意简化**：四模块是对 Claude Code、Codex 等生产 harness 的有意识降维，无法覆盖复杂状态机、多代理协作等高级机制；未来需在 compactness 与 expressiveness 间重新权衡。
- **教师依赖瓶颈**：Stage I/SII 依赖强教师 q_φ 生成合规 harness 与修复轨迹，当教师质量不足或 task type 超出教师经验时可能出现系统性偏差。
- **仅 benchmark 验证**：所有实验在离线 benchmark 上进行，未展示 real-world production deployment 下的稳定性、安全性与边界情况。
- **单模型训练**：JIT-Agent-27B 基于单一 backbone (Qwen3.6-27B) 训练，跨 architect 泛化（如 MoE、不同 context length）未评估。
- **未来方向**：(1) Foundation model 与 operational scaffold co-design，将 harness 生成内化进底座模型；(2) Production 场景中允许模型仅替换/修订部分组件的稳定 core + adaptive overlay 混合范式；(3) 可验证的运行时修改 (verifiable runtime modification) 与 adaptive interface 研究。

## 研究启发与可借鉴点
- **结构化表示缩小搜索空间**：将 harness 编码为 protocol-compliant four-module tuple 而非自由文本，既保证可执行性又使 LLM 生成具备 composability，这一思路可直接迁移至 workflow graph、agent skill composition 等程序生成任务。
- **Failure-as-supervision 范式**：Stage II 把编译/运行时诊断转化为有界 repair 轨迹的 supervised learning，为代码生成、agent runtime、DSL 编译器训练提供了可复用的 "失败即数据" 框架。
- **Multi-objective archive-frontier 竞争机制**：Evo-GDPO 的 reward/latency/cost 解耦归一化 + 保守存档策略，可用于 neural architecture search、hyperparameter optimization、多目标 RL 中的任意 program 生成场景。
- **Test-time harness diversity scaling**：Static inference 并行生成 G 个候选 harness 后选优执行，以零额外环境 rollout 换取多样性，对 cost-sensitive 部署极具吸引力。
- **Streaming bank-based knowledge transfer**：跨任务通过 harness bank 的增量更新传递经验而非在线更新模型参数，实现了零参数修改的 task-to-task 适应，可借鉴至 few-shot agent、online meta-learning。

## 关键术语表
- **Harness**：Agent 的运行时操作层，组织记忆、规划、工具调用、动作执行与验证恢复，决定 foundation model 如何转化为 closed-loop agent。
- **JIT (Just-in-Time) Harness**：在推理时针对当前任务动态合成的 executable operational scaffold，区别于预先优化的 AOT artifact。
- **Harness Intelligence**：构造、修复、进化 agent 操作脚手架的 learned capability，具备 adaptivity、reliability、evolvability 三属性。
- **Evo-GDPO**：Evolutionary Group-Decoupled Policy Optimization，对候选 harness 组的多目标归一化 Advantage + PPO clipped loss，驱动在线进化。
- **Four-module protocol h = (M, P, A, F)**：将 harness 形式化为 Memory、Planning、Action、Capability-orchestration 四个协议兼容模块的紧凑表示。
- **Streaming inference**：跨任务序列将每次执行反馈通过 harness bank 的 Update_III 规则累积，支撑后续任务的参考检索与性能提升。
- **Archive frontier**：当前 bank 中 Pareto 最优的 (reward, latency, cost) 集合，新 harness 仅在被严格超越时才纳入。
- **AgentIF**：Agent Instruction Following benchmark，衡量 agent 在多步骤日常任务中遵循指令的综合性得分。

## 可复现要素
- **数据集**：9 个公开 benchmark（BrowseComp-Plus、DeepSearchQA、xBench-DS、AgentIF-Oneday、PinchBench、DeepPlanning-Shop/Travel、OfficeBench、OdysseyBench），均为公开学术资源。
- **代码/权重**：GitHub https://github.com/bingreeky/JIT、Hugging Face https://huggingface.co/JIT-Agent 已开源；训练基于 Qwen3.6-27B。
- **关键超参**：β_pref（偏好 sharpness）、λ_pref（偏好损失权重）、α_r/α_ℓ/α_κ（Stage I 效率权衡）、λ_evo（Stage III 奖励 bonus）、ε_clip（PPO clipping）、w_rew/w_ℓ/w_κ（Evo-GDPO 归一化权重）、G（group size）、K^* 最大修复轮数（固定为 2）；具体数值见附录（论文未在主文列出完整超参表，部分
