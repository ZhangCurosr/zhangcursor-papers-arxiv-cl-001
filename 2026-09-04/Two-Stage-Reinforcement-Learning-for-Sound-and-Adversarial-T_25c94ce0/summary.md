---
title: "Two-Stage-Reinforcement-Learning-for-Sound-and-Adversarial-T"
source: https://arxiv.org/pdf/2609.03955v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 23:12:59"
field: "代码LLM后训练与验证"
keywords: ["代码生成", "强化学习", "测试用例生成", "推理期扩展", "对抗性测试", " solver-verifier 协同"]
innovations: ["两阶段RL课程学习解耦soundness与adversariality训练目标", "policy-aligned rolling buffer实现solver-verifier协同进化", "pass-count选择理论的指数可靠性边界及margin分解"]
benchmarks: ["TACO", "LiveCodeBench"]
---

# 论文速读：Two-Stage-Reinforcement-Learning-for-Sound-and-Adversarial-T

## 一句话总结
论文提出了 Test Cases Scaling (TCS)，一种两阶段强化学习框架，通过先训练模型生成与参考答案一致的"可靠"测试（Stage 1），再训练其生成针对当前模型错误模式的对抗性反例测试（Stage 2），从而在代码生成任务中实现训练期 pass@1 提升与推理期测试驱动的答案筛选。

## 研究问题与动机
- **高质量测试用例稀缺**：代码生成任务中测试用例需要同时满足"对参考解正确"（soundness）和"能区分错误解"（discriminative/adversarial）两个相互矛盾的目标，而公开测试集往往不足或质量有限。
- **现有方法局限**：传统 RL 仅优化 solver 代码生成，忽略 verifier/测试生成角色；离线 SFT 训练测试生成器无法跟踪 solver 动态变化的失败模式；单纯优化对抗性奖励在训练初期过于稀疏，导致学习困难。
- **推理期选答依赖可执行证据**：代码生成改进的瓶颈从"生成合理代码"转向"验证哪个候选正确"，基于自生成测试的执行筛选需要测试本身既可靠又有针对性。
- **奖励设计挑战**：仅优化 soundness 会导致 trivial 测试，仅优化 counterexample 奖励极稀疏，需要解耦两阶段训练以稳定学习。

## 核心贡献（创新点）
- **将测试生成形式化为对抗性 RL 问题**：首次在同一执行可验证后训练设置下，明确区分 soundness control（通过参考答案验证）和 candidate-conditioned adversariality（针对当前 solver 失败模式）两个独立目标，这在之前的工作（如 Sol-Ver）中未被显式分离。
- **提出 TCS 两阶段课程学习框架**：Stage 1 用密集 soundness 奖励建立可靠测试生成能力，Stage 2 切换到对抗性奖励，这一课程设计解决了直接优化对抗奖励时稀疏反馈的问题，区别于静态离线数据集训练方式。
- **提出 policy-aligned rolling buffer 机制**：测试生成 prompt 来自 solver 在线生成的当前策略输出（而非固定数据集），使 verifier 能持续跟踪 solver 的动态错误分布，实现 solver-verifier 的协同进化，这与外部独立训练测试生成器（如 CodeRM-8B）有本质区别。
- **提供推理期测试扩展的理论保证与实证**：推导了 pass-count 选择的指数可靠性边界（Proposition 1），证明扩展效果取决于 margin $(\delta - \alpha)$，并在 TACO 和 LiveCodeBench 上验证了自生成测试在推理期选择中的实际价值。

## 方法详解
- **共享策略双角色设置**：同一个 LLM 同时作为 solver（生成通过所有测试的代码）和 verifier（生成对参考解正确且能击败候选错误代码的测试用例），prompt 模板与奖励函数按角色区分。
- **基于 GRPO 的联合 RL 训练**：采用 Group Relative Policy Optimization (GRPO)，从混合池 $\mathcal{D} \cup \mathcal{B}$（代码任务数据集 + 策略对齐 buffer）中采样 batch，solver rollout 用 $R^c$（标准通过测试奖励），verifier rollout 用阶段特定的 $R_1^t$ 或 $R_2^t$。
- **Policy-aligned Buffer（策略对齐滚动缓冲）**：动态收集 solver 当前步生成的可执行代码（Stage 1 包含所有可执行输出，Stage 2 仅保留至少失败一个数据集测试的错误代码），保留最近 $T_b$ 步数据，确保测试 prompt 与 solver 当前行为对齐。
- **Stage 1 奖励 $R_1^t$（Soundness 控制）**：
  $$R_1^t(I_g, O_g) = \begin{cases} 1, & \text{if Exec}(C^*, I_g)=O_g \text{ and } (I_g,O_g)\notin\mathcal{T}_{\text{example}} \\ 0, & \text{otherwise} \end{cases}$$
  仅检查生成测试与参考答案执行一致，并排除与题目示例测试完全相同的情况，确保密集反馈。
- **Stage 2 奖励 $R_2^t$（对抗性反例）**：
  $$R_2^t(I_g, O_g, C^*, C_{\text{wrong}}) = \begin{cases} 1, & \text{if Exec}(C^*, I_g)=O_g \land \text{Exec}(C_{\text{wrong}}, I_g)\neq O_g \land (I_g,O_g)\notin\mathcal{T}_{\text{example}} \\ 0, & \text{otherwise} \end{cases}$$
  要求测试通过参考答案但失败于从 buffer 采样的当前策略错误候选代码；对于 runtime error/timeout 也计为成功（鼓励发现健壮性缺陷）。
- **Stage 切换策略**：当 verifier 在训练 batch 上的测试生成准确率约达 0.75 时硬性切换（1.5B 模型约 200 步，7B 约 40 步），切换时清空 buffer 重新开始收集。
- **推理期扩展（Inference-time Scaling）**：对 N 个候选代码各生成 M 个测试（论文主实验 M=1），将 N×M 个测试 pooled 后执行每个候选，选 pass-count 最高的代码；公式：$C_{\text{selected}}=\arg\max_{C_i} \sum_{j=1}^{NM}\mathbb{I}[\text{Exec}(C_i,I_j)=O_j]$。
- **理论保证（Proposition 1）**：定义 soundness error $\alpha=\Pr[\text{Exec}(C^*,I)\neq O]$ 和 counterexample rate $\delta=\min_{C\in\mathcal{C}^-}\Pr[\text{Exec}(C^*,I)=O\wedge\text{Exec}(C,I)\neq O]$，在 $\delta>\alpha$ 条件下，选错概率上界为 $(N-1)\exp(-K(\delta-\alpha)^2/2)$，其中 $K=NM$。

## 实验与结果
- **数据集与设置**：训练使用 TACO 数据集（过滤后 6,318 个问题，每个含问题描述、参考答案和测试套件）；评估在 TACO val（1,000 题）和 LiveCodeBench（2024.8–2025.2）上进行；基线模型为 DeepSeek-R1-Distill-Qwen-1.5B 和 7B。
- **对比基线**：Base Model、+Reward Model（InternLM2-7B-reward）、+Self-Generated Test Cases（基础模型自生成）、Joint SFT 离线 baseline（Sol-Ver 风格）、Code-RL（仅代码 RL）、Test-RL（仅测试 RL）。
- **主要结果（TACO w/o pub，pass@1 + BoN 选择）**：
  - R1-Distill-Qwen-1.5B：Base 5.63 → SFT 9.43 → TCS 12.31；TCS + Self-Gen Test 达 20.52（+8.21 相对 TCS 无测试选择）。
  - R1-Distill-Qwen-7B：Base 14.36 → SFT 18.92 → TCS 24.09；TCS + Self-Gen Test 达 35.35（+11.26）。
- **LiveCodeBench 最强结果**：TCS-7B + Self-Gen Test + Reward Model 组合达到 54.75（w/ pub），相对 Base 提升 26.19 点；单独 Self-Gen Test 选择达 48.79（w/o pub），超越 CodeRM-8B 的 41.70。
- **关键发现**：① TCS 自生成测试在无公开测试时几乎追平 reward model 排名；② 联合 TCS 训练同时优于 Code-RL 和 Test-RL 单独训练，证明 solver-verifier 协同的必要性；③ 随候选数 N 增大，TCS 表现持续提升且稳定，而 reward model 在 N 大时反而退化（对 OOD 候选敏感）；④ 难度分层实验显示 TCS 在 HARD/V_HARD 子集上提升最显著。

## 相关工作脉络
- **Sol-Ver (Lin et al., 2025)**：离线 joint SFT 训练 code 和 test，使用 teacher (R1-Distill-Qwen-32B) 生成样本构造对抗训练数据；本文与之对比的关键在于：TCS 通过在线 RL + policy-aligned buffer 持续跟踪 solver 动态错误，而非依赖静态离线数据。
- **CodeRM-8B (Ma et al., 2025)**：外部 unit test 生成器（SFT on Llama-3.1-8B），在相同推理预算下测试生成能力弱于 TCS 自生成测试，归因于基座模型能力、SFT vs RL 差异及非策略对齐。
- **CodeT (Chen et al., 2023)**：早期代码+测试联合生成工作；本文沿袭其思路但引入 RL 阶段的 curriculum 设计和在线策略对齐机制。
- **PyTester (Takerngsaksiri et al., 2025)**：DRL 文本到测试用例生成；本文强调 solver-verifier 共享策略和对抗性奖励的课程设计是其独特贡献。
- **Inference-time scaling 相关工作**（Snell et al., 2025; Wu et al., 2025; Shinn et al., 2023）：本文聚焦代码场景下的执行验证型扩展，区别于通用 reward model  reranking。
- **RL 基础方法 GRPO (Shao et al., 2024)**：本文采用的核心优化器，避免 value model 开销，使用 group-relative advantage。

## 局限性与未来方向
- **单次推理仅生成一个测试**：当前实现每轮推理只生成一个测试用例，并行生成多个测试效率更高，但多测试奖励函数设计面临 reward hacking 风险（如计数正确答案数易被利用）。
- **硬性两阶段切换**：采用固定阈值触发 Stage 1→Stage 2 切换，探索软性切换（动态调整两阶段奖励比例）可能更优，但因 RL 计算成本高而未深入。
- **依赖参考答案假设**：训练阶段需要 ground-truth 解 $C^*$ 来验证生成测试的 soundness，限制了无参考答案场景的适用性；未来可扩展至 model-based 或 consistency-based 验证。
- **测试输出的严格格式依赖**：测试生成 prompt 要求特定 JSON 输出格式，模型需学习格式约束，可能影响生成灵活性。

## 研究启发与可借鉴点
- ** curriculum RL 解耦复合目标**：将多目标优化（soundness + adversariality）拆分为课程学习两阶段，先用密集奖励建立基础能力，再用稀疏对抗奖励精细化——此模式可迁移到其他需要多属性平衡的生成任务（如安全测试、红队提示生成）。
- **Policy-aligned online buffer 实现 self-play 协同**：用当前策略输出构建训练数据而非固定数据集，使 verifier 始终针对 solver 的"当下弱点"——这一思想可直接应用于 solver-verifier 交替训练的通用框架，也适用于 agent 场景中 skill 协同学习。
- **理论-bound 指导实践设计**：将推理期扩展性能绑定到可度量的 $(\delta - \alpha)$ margin，不仅解释了为何需要两阶段训练（Stage 1 降 $\alpha$、Stage 2 升 $\delta$），也为测试生成质量评估提供了统一度量，可推广到其他基于执行验证的选择任务。
- **实验设计的 decoupled ablation**：将联合训练拆分为 Code-RL 和 Test-RL 两个独立变体进行对比，清晰分离 solver 侧和 verifier 侧贡献，这一 ablation 策略值得在协同训练类工作中复用。
- **与外部强模型对比验证泛化性**：将 TCS 生成的测试用于筛选其他强 LLM（如 14B 模型）的输出，证明 learned failure modes 具有跨模型迁移价值，而非仅 self-play artifact。

## 关键术语表
- **TCS (Test Cases Scaling)**：论文提出的两阶段强化学习框架，联合训练代码生成 solver 和测试生成 verifier。
- **Soundness error ($\alpha$)**：生成测试对参考答案执行错误的概率，衡量测试可靠性。
- **Counterexample rate ($\delta$)**：生成测试在参考答案通过的前提下使错误候选失败的最低概率，衡量测试判别力。
- **Policy-aligned buffer (B)**：存储 solver 当前策略在线生成代码的滚动缓冲，用于构建与 solver 当前失败模式对齐的测试生成 prompt。
- **GRPO (Group Relative Policy Optimization)**：无需 value model 的 RL 算法，用同 prompt 多输出组内平均奖励作为 baseline 计算 advantage。
- **Pass-count selection**：推理期从 N 个候选中选 pass 测试数最多的代码作为最终答案的策略。
- **Inference-time scaling**：通过增加推理期计算（更多候选采样、更多测试生成）而非扩大模型参数来提升性能的方法。
- **Adversarial test case**：通过参考答案但使特定错误候选失败的测试用例，用于暴露候选代码的隐藏 bug。

## 可复现要素
- **数据集**：TACO 训练集（过滤后 6,318 题，已公开于 Hugging Face: TACO-Train）；评估集 TACO val（1,000 题）和 LiveCodeBench（2024.8–2025.2）。
- **代码/权重**：TCS-1.5B 和 TCS-7B checkpoint 已公开于 Hugging Face；代码和评估 pipeline 公开于 TCS GitHub 仓库。
- **关键超参**：batch size=128，PPO mini-batch=64，GRPO group size=16，temperature=0.8，max response length=8192，无 KL loss（使用 entropy loss 维持熵），Stage 1→2 切换阈值约 0.75 测试生成准确率，推理期 M=1（每候选生成 1 个测试）。
- **训练步骤**：1.5B 模型 Stage 1=200 步、Stage 2=250 步；7B 模型 Stage 1=40 步、Stage 2=160 步；Code-RL baseline 同为 450（1.5B）/200（7B）步。
- **硬件**：NVIDIA H100 80GB，1.5B 训练 720 GPU-hrs，7B 训练 960 GPU-hrs。
- **SFT baseline 配置**：使用 R1-Distill-Qwen-32B 生成 32 samples/题，3 epochs，默认 TRL 配置。
