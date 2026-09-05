---
title: "One-Policy-Is-Enough-Single-Agent-Reinforcement-Learning-Out"
source: https://arxiv.org/pdf/2608.30952v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:42:56"
field: "科学领域工具增强语言模型"
keywords: ["tool learning", "reinforcement learning", "chemistry agents", "GRPO", "programmatic reward", "single-agent"]
innovations: ["单策略RL替代多模型树搜索进行化学工具学习", "系统性程序化奖励框架支持多工具调用训练", "证明SFT热身对工具recall和RL对precision的互补作用"]
benchmarks: ["ChemToolBench"]
---

# 论文速读：One-Policy-Is-Enough-Single-Agent-Reinforcement-Learning-Out

## 一句话总结
本文证明在化学工具学习中，一个单策略模型通过强化学习即可超越基于树搜索的多模型系统；该方法在 ChemToolBench 上以单次模型调用超越 CheMatAgent 的 HE-MCTS，且在 Qwen-2.5-7B 上 Tool F1 提升 5.5%、Return F1 提升 9.6%。

## 研究问题与动机
- 化学问题需要精确计算和数据库查询，LLM 无法从参数中获取，必须调用外部工具。
- 工具学习面临三大挑战：从大规模工具池中选择正确工具、生成严格格式的结构化参数、处理长链依赖的工具调用。
- 现有 SOTA 方法 CheMatAgent 使用四级模型协作（策略模型+执行模型+PRM+ORM）及树搜索，训练和推理成本高昂。
- 核心疑问：是否真有必要使用如此复杂的搜索架构？单策略能否达到同等或更好效果？

## 核心贡献（创新点）
- **单策略 Rollout 协议**：一个策略 π_θ 交替生成推理块和工具调用，一次生成完成全部工具执行，摒弃了执行模型、搜索和 learned critic。与 CheMatAgent 多模型树搜索的本质区别在于推理阶段仅需一次模型调用。
- **系统性程序化奖励框架**：设计了包含答案值包含、工具集匹配、调用链匹配及混合奖励的多维奖励函数，无需 GPT judge 或 learned reward model。与 ToolRL 等工作的区别在于直接读取 gold call chain 而非仅匹配最终答案。
- **超越 SOTA 的实验性能**：在相同 backbone 上以单策略+RL 超越 CheMatAgent 最强搜索配置，Qwen-2.5-7B 上 Tool F1 达 95.76（+5.5%），Return F1 达 88.89（+9.6%）。
- **训练组件解耦分析**：量化了 SFT 热身 vs RL 各自的贡献，发现 SFT 对 recall 和 Pass Rate 更重要，RL 主要提升 precision。

## 方法详解
- **单代理 Rollout 协议（§4.1）**：模型在最多 T=16 轮对话中与工具服务器交互；每轮生成 `<thinking>` 推理块和单个 `<tool_call>` JSON；服务器执行后将 `<tool_result>` 追加到上下文；`<tool_result>` 跨度在训练中 mask 掉，仅在模型生成的 token 上计算 loss。
- **监督微调热身（§4.2）**：将 gold 调用链线性化为 rollout 轨迹（推理块+调用+返回），训练策略模型学习工具调用格式和正确的工具选择；SFT→RL 以此 checkpoint 开始 RL 训练。
- **程序化奖励设计（§4.3）**：所有奖励通过程序计算并映射到 [-1, 1]：
  - R_ans：答案中完整包含 gold return values 的比例
  - R_tool/R_tool^F1：工具名集合的 recall 或 F1
  - R_call/R_call^F1：工具名+参数的匹配，后者将调用视为列表以惩罚重复调用
  - R_hyb：工具奖励与答案奖励的等权混合
- **GRPO 强化学习（§4.4）**：每个 prompt 采样 n=5 条 rollout，组内标准化奖励得到 advantage；学习率 1×10^-6；使用 slime 训练框架，Megatron-LM 更新策略，SGLang server 生成多轮 rollout。

## 实验与结果
- **数据集**：ChemToolBench 多工具化学子集，200 条测试题，平均 3.15 次依赖调用（2-6 次）。
- **Backbone**：Qwen-2.5-7B-Instruct、Llama-3.1-8B-Instruct、Qwen3-4B。
- **主要结果（Table 2）**：
  - Qwen-2.5-7B：Tool F1 95.76 vs CheMatAgent-M3 的 90.80（+5.5%）；Return F1 88.89 vs 81.10（+9.6%）；Pass Rate 68.50 vs 67.32。
  - Llama-3.1-8B：Tool F1 95.93 vs 92.47（+3.7%）；Return F1 89.70 vs 86.36（+3.9%）；Pass Rate 64.00 vs 72.30（搜索仍领先）。
- **训练消融（Table 3）**：SFT-only 在 Tool F1 和 Pass Rate 上均超过 RL-only，表明 SFT 热身贡献不小于 RL 本身。
- **奖励设计消融（Table 4）**：仅奖励 recall 会导致 tool spamming（平均 18.33 次调用 vs gold 3.15 次，97% 未终止）；引入 F1/precision 后调用数恢复正常（~3.06 次）。

## 相关工作脉络
- **CheMatAgent (Wu et al., 2025)**：本文对比的 SOTA，使用 HE-MCTS 树搜索+两模型分工+PRM/ORM critic；本文在相同 benchmark 和工具池下仅改变训练方式即超越。
- **Toolformer / ToolLLM**：早期监督式工具学习工作，本文在其基础上引入在线 RL 和真实工具执行。
- **DeepSeek-R1 / GRPO**：Verifiable reward RL 的成功范式，本文将其迁移到多工具化学场景。
- **ToolRL (Qian et al., 2026)**：同样使用程序化分解奖励，但匹配工具 trace 而非 gold call chain；本文在 reward 设计上更精细（考虑 list vs set、repeat 惩罚）。
- **ChemCrow / CACTUS / ChemLLM**：化学领域的工具增强 LLM 工作，本文方法可直接应用于此类系统。
- **ReTool / Search-R1**：近期将 RL 用于代码执行和搜索的代理工作，本文延续此思路至化学工具领域。

## 局限性与未来方向
- **域单一**：仅在 ChemToolBench 化学/材料领域验证，未测试其他科学工具域。
- **Horizon 有限**：当前问题最多 6 次调用，更长 horizon 未评估。
- **模型规模**：仅 4-8B 参数，更大模型的效果未知。
- **Pass Rate 差距**：Llama-3.1-8B 上 Pass Rate 仍落后 CheMatAgent，说明纯工具级奖励对最终答案质量的间接优化有限。

## 研究启发与可借鉴点
- **简化架构的可行性**：复杂搜索系统可被单策略+RL 替代，降低工程复杂度；适用于其他需要多步工具调用的领域。
- **SFT 热身的重要性**：对于格式敏感的工具调用任务，短期 SFT 对稳定训练和提升 recall 至关重要，可与 RL 形成互补。
- **程序化奖励设计模式**：多维度奖励（工具选择+参数匹配+返回值）的组合及 F1 语义防止 reward hacking 的经验可迁移到其他 tool use 场景。
- **自修复机制**：真实工具执行产生的 error feedback 可被模型利用进行 self-repair，这是模拟执行无法提供的信号，值得在其他 agentic 系统中利用。
- **统一 rollout 协议评估**：训练和评估使用相同的 rollout 协议，确保性能提升归因于训练方法而非评估差异。

## 关键术语表
- **HE-MCTS**：Hierarchical Evolutionary Monte-Carlo Tree Search，CheMatAgent 使用的分层进化树搜索方法。
- **GRPO**：Group Relative Policy Optimization，DeepSeekMath 提出的无 critic 的 RL 算法，通过组内相对优势估计替代价值网络。
- **PRM / ORM**：Process Reward Model / Outcome Reward Model，分别对中间步骤和最终答案打分的学习型奖励模型。
- **Rollout Protocol**：单策略模型交替生成推理和工具调用的交互协议，工具结果由服务器实时返回。
- **Tool Spamming**：奖励设计缺陷导致模型过度调用工具的 pathological behavior。
- **Self-Repair**：模型根据工具执行的错误反馈自动修正后续调用参数的能力。
- **ChemToolBench**：包含 137 个化学工具的 benchmark，配有验证过的 gold 调用链。

## 可复现要素
- **数据集**：ChemToolBench，已公开（Wu et al., 2025）。
- **代码/权重**：论文未提及开源声明；backbone 为 Qwen-2.5-7B-Instruct、Llama-3.1-8B-Instruct、Qwen3-4B，均为开源模型。
- **关键超参**：SFT 学习率 10^-5、batch size 64、3 epochs；RL 学习率 1×10^-6、group size n=5、100 rollout steps；最大 turn T=16。
- **训练框架**：slime post-training stack + Megatron-LM + SGLang server。
