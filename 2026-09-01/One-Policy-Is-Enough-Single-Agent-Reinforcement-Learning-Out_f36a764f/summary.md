---
title: "One-Policy-Is-Enough-Single-Agent-Reinforcement-Learning-Out"
source: https://arxiv.org/pdf/2608.30952v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:43:27"
field: "化学工具学习/Agent RL"
keywords: ["tool learning", "single-agent RL", "GRPO", "programmatic reward", "ChemToolBench", "chemistry agents", "tree search replacement"]
innovations: ["单策略 Rollout 协议替代 CheMatAgent 四模型+树搜索管线", "面向多步调用链的程序化奖励框架（R_call^F1 等）避免 learned Critic", "SFT 提供召回、RL 提供精确率的两段式训练并揭示 self-repair 行为"]
benchmarks: ["ChemToolBench (chemistry multi-tool split, 200 questions)"]
---

# 论文速读：One-Policy-Is-Enough-Single-Agent-Reinforcement-Learning-Out

## 一句话总结
论文用单一策略模型配合 GRPO 强化学习与程序化奖励，替代了 CheMatAgent 的多模型树搜索管线，在 ChemToolBench 化学工具学习基准上以一次模型调用达到更强的 Tool F1（+5.5%）和 Return F1（+9.6%）。

## 研究问题与动机
1. 化学问题需要精确计算和数据库查询，语言模型参数无法内化这些信息，必须依赖外部工具调用。
2. 现有最强方法 CheMatAgent 使用策略模型 + 执行模型 + Process Reward Model（PRM）+ Outcome Reward Model（ORM）共四个模型，并在推理时对每个问题执行树搜索，成本和管线复杂度较高。
3. 当前树搜索方法的 PRM/ORM 之一还依赖 GPT-4o 评分混合，训练链路中存在 LLM judge，复现性与可控性受限。
4. 核心疑问：化学工具学习是否真的需要这样一层"重型"管线，还是单策略 RL 足以逼近甚至超越？

## 核心贡献（创新点）
1. **单策略 Rollout 协议**：一个策略 π_θ 在同一左到右生成中交替输出思考块 <think>... 与工具调用 <tool_call>，执行结果由工具服务器以 <tool_result> 回填，推理成本为一次模型调用；与 CheMatAgent 拆分为策略模型 + 执行模型 + 树搜索的本质不同在于"不再需要搜索与第二模型"。
2. **程序化奖励框架**：提出 R_ans、R_tool/R_tool^F1、R_call/R_call^F1、R_hyb/R_hyb^F1 等多维奖励变体，直接比对黄金调用链，无需训练任何 Reward Model 或引入 LLM judge；与 CheMatAgent 依赖 PRM/ORM（含 GPT 评分）的根本差异在于"可微分且确定性的规则信号取代学习型 Critic"。
3. **端到端 GRPO 优化**：对每组 n=5 采样做组内标准化 Advantage，学习率 1e-6，100 rollout steps 即可收敛；与 ToolRL/Nemotron 等同类 RL 工作相比，本文 reward 直接读取黄金链上的 tool+argument 匹配，而非仅对最终答案打分，更贴合多步依赖链场景。

## 方法详解
1. **Rollout 协议（§4.1）**：每个 episode 最多 T=16 轮；每轮先生成一小段 <think> 推理，再发出严格 JSON 的 <tool_call>{"name": "<lib>/<func>", "arguments": {...}} 请求；服务器将真实返回值包裹为 <tool_result>... 追加到上下文；episode 以单行 "Answer: ..." 或预算耗尽终止。<tool_result> 段在训练 loss 中 mask，只更新模型自生成 token。
2. **监督预热（§4.2）**：将黄金调用链 C 线性化为完整 rollout 轨迹，末尾补充完整的自然语言答案句；3 个 epoch、lr=1e-5、batch=64；使模型先掌握输出格式与工具选择初值。
3. **程序化奖励（§4.3）**：所有奖励均通过程序直接由黄金链 C 与模型生成结果 Ĉ, â 计算，映射到 [-1, 1]：
   - R_ans：答案中包含的归一化后黄金返回值的比例（逐字匹配，部分计入负奖励）。
   - R_tool / R_tool^F1：仅对工具名做 recall / F1（set 语义），惩罚无节制多调。
   - R_call / R_call^F1：工具名 + 参数同时匹配；^F1 采用 list 语义（重复调用扩大分母），进一步抑制 spam。
   - R_hyb：0.5·R_tool + 0.5·R_ans（或其 ^F1 版本）。
4. **GRPO 训练（§4.4）**：每组采样 n=5 条 rollout，组内标准化得 advantage，clip 策略梯度；lr=1e-6；由 slime 框架串联 Megatron-LM（策略更新）与 SGLang server（多轮 rollout 生成），通过 data buffer 通信。
5. **推理设置**：贪心解码 τ=0，确保表格数字不含采样方差；所有模型（包括未训练 backbone）均在同一条 rollout 协议下评估，分离"协议贡献"与"训练贡献"。

## 实验与结果
- **数据集**：ChemToolBench 化学多工具 split，200 条测试题（平均 3.15 步/题，跨度 2-6 步）；工具池 137 个（ChemCrow 8 / CACTUS 10 / chemlib 24 / pymatgen 82 / Chemistry Tools 13）。因部分工具调用在线服务导致历史黄金返回值过时，训练/开发各去掉 57/4 条，测试集不动。
- **Backbone**：Qwen-2.5-7B-Instruct、Llama-3.1-8B-Instruct（主实验），另加 Qwen3-4B 作规模检查。
- **评估指标**：Format、Tool P/R/F1、Param P/R/F1、Return P/R/F1、Pass Rate（GPT-4o-mini judge 判答案）。
- **最强结果**：
  - Qwen-2.5-7B：Tool F1=95.76（CheMatAgent-M3 的 90.80，↑5.5%）；Return F1=88.89（81.10，↑9.6%）；Pass Rate=68.50（67.32，↑1.2%）；Param F1=87.85。
  - Llama-3.1-8B：Tool F1=95.93（-M3 的 92.47，↑3.7%）；Return F1=89.70（85.41，↑3.9%）；Pass Rate=64.00（低于 -M2 的 72.30，但优于全部 CoT/SFT 基线）。
- **关键结论**：在 Qwen 上单策略在几乎所有可比列均领先；在 Llama 上搜索仍有 Pass Rate 优势，但工具/参数/返回 F1 全面被追平；RL-only 与 SFT-only 对比显示：SFT 主要贡献召回与最终答案质量，RL 主要贡献精确率。

## 相关工作脉络
1. **CheMatAgent (Wu et al., 2025)**：同 benchmark 的 SOTA，用 HE-MCTS 拆分策略/执行两模型并引入 PRM/ORM 搜索；本文保留其 benchmark 与工具池不变，仅替换训练方式，证明"同样数据下单策略更优"。
2. **Toolformer / ToolLLM**：早期工具学习以 SFT 为主；本文把 SFT 降级为"热身"，在线 RL 作为真正驱动力，路径由离线imitation 转向 online探索。
3. **DeepSeek-R1 (Guo et al., 2025) → GRPO**：借鉴"verifiable reward 激发单模型推理"的思想，从数学迁移到多步工具调用场景。
4. **ReTool (Feng et al., 2026) / Search-R1 (Jin et al., 2025)**：同属单模型 RL 范式，但分别针对代码执行与搜索；本文面向化学多工具长链调用，强调工具间参数传递与自修复行为。
5. **ToolRL (Qian et al., 2026) / MatchTIR (Qu et al., 2026)**：均做过细粒度 reward；本文与之的区别在于 reward 直接比对"整条黄金调用链（含参数）"而非仅答案级或 turn 级匹配，且明确分析 spamming 与非终止病态。

## 局限性与未来方向
1. **单一领域与基准**：仅在 ChemToolBench 化学多工具 split 上验证，跨域泛化未检验；是否适用于其他科学工具领域未知。
2. **未见更长 horizon 与大模型**：测试链长为 2-6 步、backbone 为 4-8B；更长的多步链与更大参数规模的模型上表现未知。
3. **Pass Rate 在非 Qwen 上落后搜索方案**：说明纯工具链奖励不能直接优化最终答案质量，可能需要融合 answer-level 信号。
4. **在线工具过时问题**：部分工具返回随时间变化导致少量黄金链失效（需剔除训练样本），在更开放环境中此类 drift 可能更严重。

## 研究启发与可借鉴点
1. **"SFT 热身 + GRPO 精细调优"的两段式在工具学习中极为有效**：SFT 负责召回/格式，RL 负责精确率；可迁移到其他函数调用/ Agent 场景作为默认训练管线。
2. **程序化 reward 替代 learned Critic 的可行性**：只要存在可验证的中间结构（调用链、参数、返回值），就能用规则奖励完全绕过 PRM/ORM，降低管线复杂度和训练噪声。
3. **List-F1 语义对抑制 tool spamming 的关键作用**：简单 set-recall 会导致模型调用所有工具刷分；把重复调用计入分母能有效阻断 reward hacking。
4. **同 rollout 协议评估 untrained backbone 以分离协议效应**：这一实验设计把"评估框架本身带来的提升"与"训练带来的提升"拆开，值得在多 agent 论文中推广。
5. **执行反馈触发的 self-repair 行为可被 RL 自发习得**：gold chain 只含成功调用，模型却能在 RL 中学到"出错重试"，提示在线执行比离线模拟更能激活此类能力。

## 关键术语表
**HE-MCTS**：CheMatAgent 使用的层次进化蒙特卡洛树搜索，拆分策略模型与执行模型并用 PRM/ORM 引导树搜索。
**GRPO**：Group Relative Policy Optimization，PPO 的去 critic 变体，通过组内归一化 reward 估计 advantage。
**PRM / ORM**：Process Reward Model（步骤级）与 Outcome Reward Model（结果级），CheMatAgent 的两类学习 critic。
**Programmatic Reward**：由程序直接比对黄金调用链与模型输出而得的确定性奖励，无需学习模型或 LLM judge。
**Rollout Protocol**：单策略交替生成 <think>/<tool_call> 并在 <tool_result> 中读取真实执行结果的交互协议。
**R_call^F1**：以 list 语义计算的工具名+参数联合 F1，重复调用会拉低 precision，从而抑制 tool spamming。
**Pass Rate**：由 GPT-4o-mini judge 判定的最终答案正确率。
**Self-repair from Execution Feedback**：模型在读到工具执行错误后主动更正参数并重试的能力，只能由在线执行触发。

## 可复现要素
- **数据集**：ChemToolBench（chemistry multi-tool split），原始发布随 CheMatAgent；论文已公开训练/开发剔除的过时样本数（训练 57、开发 4），测试集维持 200 题。
- **代码/权重**：论文未明确声明开源；工具池引用 CheMatAgent 的 137 工具与 prompt/rollout 细节已在附录披露。
- **关键超参**：SFT 3 epoch、lr=1e-5、batch=64；GRPO 100 rollout steps、组大小 n=5、lr=1e-6；推理 τ=0 贪心；最大轮次 T=16。
- **框架**：slime（Zhu et al., 2025）+ Megatron-LM + SGLang server。
