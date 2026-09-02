---
title: "Apodex-1-1-Scaling-Agentic-Intelligence-for-Complex-Work"
source: https://arxiv.org/pdf/2608.23283v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:38:09"
---

# 论文速读：Apodex-1-1-Scaling-Agentic-Intelligence-for-Complex-Work

## 一句话总结
Apodex 1.1 提出以“工作效能（Working Capability）”为核心目标的通用智能体系统，通过**环境缩放（Environment Scaling）**与**代理协调缩放（Agentic Coordination Scaling）**两个互补维度，训练模型在长周期、多工具、可失败、需交付的真实工作流中维持有效进展；在专业工作、金融、科研、数学、代码与深度搜索等任务上达到前沿性能带，且 35B 参数的 Mini 版本仍具备显著的效率优势。

## 研究问题与动机
- **答案质量与可靠执行的鸿沟**：现有通用语言模型在知识、推理与编码上快速进步，但复杂工作需要在长时间跨度内与异构文件、搜索源、可执行代码持续交互，并维持状态、从失败恢复、最终交付可验证产物；单纯提升推理能力无法解决此问题。
- **工具使用与执行环境的脱节**：Prior 方法常将工具视为静态文本描述或外挂接口，缺乏对真实状态转移、失败模式、资源预算与交付验收的系统性训练，导致“会答题但不会干活”。
- **多智能体协作停留在推理期包装**：已有系统多依赖固定角色对话图、多数投票或静态编排，未将任务分解、异步集成、依赖失效传播与动态重规划内化为模型策略，缩放变量仅是代理数量而非有效协调工作量。
- **评估体系无法刻画过程质量**：主流 benchmark 聚焦最终得分或单轮工具调用，缺乏对长轨迹状态一致性、证据溯源、假设管理与受控交付的过程级度量，难以指导面向真实工作的能力迭代。

## 核心贡献（创新点）
1. **环境缩放（Environment Scaling）作为独立能力面**：将文件、搜索、代码三类可执行世界统一纳入训练分布，强调状态转移保真度、失败模式覆盖与交付验证的结构深度；与既有工作仅扩充工具列表或提示数量的本质区别在于以“任务契约的生态多样性”而非“提示/工具数量”驱动策略泛化。
2. **代理协调缩放（Agentic Coordination Scaling）**：将分解、委派、异步结果集成与动态重规划训练为模型自身的 working policy，并在推理期通过 Agent Team 以自适应 fan-in/fan-out 实现；与 AutoGen/MetaGPT 等固定编排框架的区别在于以“随目标演进的有效协调工作量”为缩放变量，而非并行样本或角色数。
3. **AgentOS 统一执行基座**：提供持久化工作区、三区域命名空间、外部 Task Board 与依赖图、分层上下文压缩与受控交付租约，使环境交互与多智能体协调共享同一运行时契约；与孤立工具链/单次 ReAct 循环的本质区别在于跨长轨迹的状态保持与干预-恢复可组合性。
4. **PIVOT-RL 局部强化学习**：针对长轨迹信用分配难题，通过回溯定位 consequential decision points，在保留有效前缀的前提下注入方向性提示进行局部修正训练；与端到端 RL 的本质区别在于将全局 reward 转化为聚焦关键转折点的低成本定向优化。
5. **过程导向的评估范式（HDS6）**：从答案评分转向六维过程验证（状态一致性、证据保真、假设管理、边界/失败推理、工具执行、自我修正）；与主流 benchmark 仅看最终分数的区别在于以“已完成工作的可辩护路径”作为能力单元。

## 方法详解
- **任务契约形式化**：定义 $\mathcal{E} = (\mathcal{W}, W_0, q, \mathcal{A}, \mathcal{T}, \Omega, \mathbf{B}, D, V_D)$，其中 $\mathcal{W}$ 为工作区状态空间，$q$ 为规范化目标，$\mathcal{A}$ 为动作集合，$\mathcal{T}$ 为状态转移算子，$\Omega$ 为观测接口，$\mathbf{B}$ 为资源预算向量，$D$ 为交付契约，$V_D$ 为任务级验证器。模型轨迹 $\tau_H = (u_0, a_0, o_1, \dots)$ 的最终质量由 $S_D = V_D(W_0, W_H, \tau_H)$ 判定，而非自然语言答案本身。
- **三类环境家族**：
  - **File worlds**：以权威性与变换为核心，工作区包含嵌套目录、多格式历史版本与跨文件引用；验证锚定于独立可推导数值或已记录血缘。
  - **Search worlds**：以发现与证据对齐为核心，agent 需进行查询重构、来源分流、交叉溯源与冲突调解；gold 包含来源集合与不确定性标注。
  - **Code worlds**：以可执行变换与沙箱验证为核心，基于 Docker 镜像与仓库状态； harvested 任务要求 fail-to-pass/pass-to-pass 测试双重校验，synthesized 任务需通过对抗性 grader hardening 防 reward hacking。
- **难度校准**：对文件/搜索类任务引入获取压力坐标 $\rho_{\text{acq}}(\mathcal{E}) = \frac{N_{\text{cand}} + N_{\text{hop}}}{B_{\text{tool}}}$，衡量候选检验数、证据跳数与工具预算的比值；代码任务则依赖依赖深度、状态转移深度与测试可观测性。
- **Agent Team 1.1 协调机制**：
  - **显式 Task Board**：主 Agent 将分解结果写入外部任务板（目标/依赖/所有者/状态），替代 LLM 内部记忆；依赖图 $G_t$ 与工件索引 $I_t$ 支持单项失效仅影响后代而不破坏独立分支。
  - **异步人类干预**：执行中接收 $u_t$，若仅澄清/改优先级/补充证据则更新板项并失效相关后代；若实质性改变 $q$ 或 $D$ 则开启新契约并继承有效工作区状态。
  - **不对称验证**：Verifier 仅接收特定声明、支撑证据与交付约束，目标是寻找反例/核对原子细节/检验合规性，而非全题重做，降低上下文污染与冗余计算。
  - **自适应 Max Team Effort**：仅对弱、争议或承重假设追加独立调查分支，每轮结果返回后动态重分配预算。
  - **证据 grounded synthesis**：两阶段——先构建 claim-evidence graph 并生成写作大纲，再交由 Writer Agent 在相同工具/引用/交付约束下产出最终产物。
- **AgentOS 运行时**：三命名空间 `/inputs`（只读任务文件）、`/workspace`（中间产物）、`/outputs`（受控交付根）；可选挂载只读 `/shares` 持久参考库；分层上下文压缩（先驱逐旧观测正文，不足再 LLM 摘要）；软/硬双重预算与暂停/取消分离；单一发布者租约 + baseline 快照比对防伪造交付。
- **训练流程**：统一 SFT 混合（推理/工具/搜索/文件/代码/数学/专业交付/多智能体）→ PIVOT-RL 异步优化（定位 pivot → 保留有效前缀 + 短提示局部继续 + 与完整任务混合）；失败轨迹经 Task Pipeline 归类为能力缺口，驱动下一轮环境与任务构造。

## 实验与结果
- **公开基准**：APEX-Agents、GDPVal、FrontierFinance、YC-Bench、FrontierScience-Research、BioMysteryBench（Human-difficult）、HLE、DeepSearchQA、IMO/USAMO/ProofBench、Terminal-Bench 2.1、SWE-bench Verified。
- **主要结果**：
  - 专业工作：APEX-Agents ReAct 34.4 / Agent Team 38.5；GDPVal ReAct 69.5 / Agent Team 78.8。
  - 金融：FrontierFinance ReAct 48.7 / Agent Team 54.3（领先对比集）；YC-Bench 达 $1,038,255。
  - 科研：FrontierScience-Research ReAct 55.0 / Agent Team 63.3；BioMysteryBench 23.5% → 35.3%。
  - 推理与搜索：H
