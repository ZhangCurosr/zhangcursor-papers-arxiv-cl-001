---
title: "REPOLICY-REINFORCEMENT-LEARNING-FOR-SAFETY-POLICY-INVOCATION"
source: https://arxiv.org/pdf/2608.24275v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:37:14"
field: "LLM Agent Safety"
keywords: ["Agent Safety", "Reinforcement Learning", "Policy Invocation", "Guard Models", "Trajectory-level Safety"]
innovations: ["将安全策略调用建模为代理强化学习任务，通过GRPO直接优化策略调用与安全预测", "引入policy-context perturbation机制增强策略选择的鲁棒性", "构建PolicyTraj-20K数据集支持策略级监督"]
benchmarks: ["ATBench", "R-Judge", "OpenAgentSafety", "ASSEBench", "HINTBench", "AgentHazard"]
---

# 论文速读：REPOLICY: REINFORCEMENT LEARNING FOR SAFETY-POLICY INVOCATION IN AGENT SAFEGUARDS

## 一句话总结
论文提出了 RePolicy，一种将安全策略调用建模为代理强化学习任务的 Agent 安全守护方法，通过冷启动 SFT + GRPO 直接优化"策略调用→策略内容获取→安全推理→安全预测"的完整序列，并在构建的 PolicyTraj-20K 数据集上实现了 88.15 的 Overall Unsafe F1，显著优于现有基线。

## 研究问题与动机
- **现有方法不足**：当前 Agent 安全守护主要依赖提示工程或监督微调，无法充分适应未见过轨迹和新策略组合； prompting 不直接优化策略使用方式，SFT 能力受限于训练数据的轨迹-策略分布。
- **策略情境动态性**：同一动作在不同用户权限、组织要求、监管管辖下可能面临不同约束，且安全策略本身会随时间演化，固定策略集难以覆盖真实部署。
- **轨迹级安全评估需求**：Agent 安全风险已从单一交互中的有害内容扩展到完整执行轨迹的行为危害，需要轨迹级监控能力。
- **监督与部署不匹配**：缺乏同时包含多样化 Agent 轨迹和细粒度安全策略监督的数据资源，现有资源往往针对特定环境或风险类型。

## 核心贡献（创新点）
1. **将安全策略调用形式化为代理强化学习任务**：与已有工作将策略作为被动上下文不同，RePolicy 将策略视为可调用的能力，通过 RL 直接优化调用决策和安全预测。
2. **构建 PolicyTraj-20K 数据集**：包含 20K+ 条带安全策略标注的轨迹，按操作场景组织策略库，每条轨迹标注适用策略、相关条款、安全标签和决策依据，为 SFT 初始化和 RL 提供统一数据支撑。
3. **引入 policy-context perturbation 机制**：在候选策略集中注入无关和干扰策略，防止模型依赖表面相关性，强制模型根据轨迹语义而非候选集组成来判断策略适用性。
4. **设计可验证的规则奖励函数**：组合格式奖励、策略调用奖励和安全预测奖励，实现端到端的 RL 优化，策略调用准确率从 94.4% 提升至 98.5%。

## 方法详解
- **任务形式化**：给定 Agent 轨迹 τ 和动态策略库 P={p₁,...,pₖ}，首先调用适用策略 p̂∼π_θ(·|τ,P)，然后将策略内容 c(p̂) 作为上下文进行安全推理，生成基于策略的 rationale r̂ 和安全标签 ŷ∈{safe,unsafe}。
- **冷启动 SFT**：将适用策略与干扰策略集合并为局部策略库，用标准交叉熵损失优化两步生成：策略调用和 rationale+label 生成。
- **GRPO 强化学习**：基于冷启动模型，对每个输入采样多个 rollout，使用可验证规则奖励进行比较优化。
- **奖励函数设计**：R = λ_fmt R_fmt + λ_pol R_pol + λ_acc R_acc，其中 λ 系数分别为 0.1/0.3/0.6；R_pol=I[p̂=p*] 直接评估策略调用正确性；R_acc=I[ŷ=y] 评估安全预测准确性。
- **策略-上下文扰动**：RL 阶段重采样干扰策略并 shuffle 局部策略库顺序，干扰策略包含无关操作场景的合法策略和合成干扰策略，每个局部库平均含 9.2 个干扰策略。

## 实验与结果
- **数据集**：PolicyTraj-20K 含 20K+ 条轨迹，六个基准共 7,369 条标注轨迹（ATBench 1,000、R-Judge 1,242、OpenAgentSafety 289、ASSEBench 1,476、HINTBench 709、AgentHazard 2,653）。
- **基线对比**：19 个外部模型（闭源通用模型、开源通用模型、专用 Guard 模型）。
- **主要结果**：RePolicy-4B 在六个基准上平均 Unsafe F1 为 88.15，比最强外部基线（Claude Sonnet 4.6，84.17）高 3.98 分，比最强专用 Guard（AgentDoG 1.5，77.65）高 8.46 分；在 ATBench（99.40）、OpenAgentSafety（62.79）、ASSEBench（88.11）、AgentHazard（99.20）四个基准上排名第一。
- **RL 效果**：GRPO 使策略调用正确率从 94.4% 提升至 98.5%，平均每个轨迹调用策略数从 1.9 增至 3.2，但干扰策略选择率始终低于 1%。
- **消融**：冷启动 SFT 提升 Overall F1 28.48 分；显式策略调用是主要 RL 组件，在 AgentHazard 上提升 5.9 分、OpenAgentSafety 上提升 4.4 分；策略库大小敏感性实验显示 Macro F1 在 |P|=40 时饱和。

## 相关工作脉络
- **传统 Guard 模型**（WildGuard、ShieldGemma、Llama Guard）：主要针对提示/响应分类，基于预定义风险分类，不适合轨迹级监控。
- **轨迹级 Agent 守护**：GuardAgent 进行知识增强安全检查，AGrail 自适应构建安全检查，AgentDoG 提供细粒度轨迹诊断；本文与它们同属轨迹级设置，但额外显式识别适用策略。
- **策略感知安全守护**：ShieldAgent 推理可验证规则，DynaGuard 支持用户定义自由格式策略，LPG 在动态策略上下文中推理；本文区别在于将策略调用显式化，并从候选库中主动选择策略，同时优化调用和安全预测。
- **代理强化学习**：SearchR1、Retool 等将 RL 应用于工具使用；本文借鉴该思路，将安全策略调用作为可调用能力进行 RL 优化。

## 局限性与未来方向
- **策略库覆盖假设**：RePolicy 假设候选策略库包含适用且足够精确的策略，不完整覆盖、模糊范围、过时要求或策略冲突会影响决策；当前 formulation 将每条轨迹关联单一策略，但真实部署可能需要跨操作、组织、监管层面联合应用多条策略。
- **推理开销**：策略调用引入额外交互步骤，推理开销高于单遍 Guard。
- **评估局限**：仅在一个骨干模型规模、离线轨迹级基准和二元安全标签上评估，未充分捕捉多语言部署、持续演化环境、在线干预或 Guard 需决定何时中断执行 Agent 的场景。
- **理由忠实度**：奖励验证了策略调用和最终预测，但未保证生成理由中每句话都忠于被调用的策略。
- **未来方向**：多策略联合调用、在线干预机制、理由忠实度验证、更大规模和多语言评估。

## 研究启发与可借鉴点
- **RL 优化显式调用能力**：将"调用外部知识/工具"作为可优化能力，通过 RL 直接优化调用决策，而非仅依赖 SFT 模仿，对工具使用、知识检索等场景有可迁移价值。
- **干扰策略构造**：policy-context perturbation 通过注入合法但不相关策略 + 合成干扰策略来增强模型判别能力，该方法可迁移至其他需要上下文选择的场景（如文档检索、代码片段选择）。
- **可验证规则奖励设计**：结合格式验证、调用正确性和预测准确性三种奖励，实现端到端优化且避免 reward hacking，值得在需要多步推理的任务中借鉴。
- **场景中心化的策略组织**：按 Agent 操作场景而非最终风险类别组织策略，使单一策略能捕捉一类操作的多风险，对构建分层安全体系有参考价值。

## 关键术语表
- **RePolicy**：一种将安全策略调用建模为代理强化学习任务的 Agent 安全守护方法。
- **PolicyTraj-20K**：包含 20K+ 条带安全策略标注的轨迹数据集，按操作场景组织，每条轨迹标注适用策略、相关条款、安全标签和决策依据。
- **GRPO（Group Relative Policy Optimization）**：一种基于组相对策略优化的强化学习方法，通过比较同输入下的多个 rollout 来优化策略。
- **Policy-context perturbation**：在训练过程中对候选策略集进行重采样和排序打乱的增强技术，防止模型依赖表面相关性。
- **Unsafe F1**：以 unsafe 为正类的 F1 分数，用于评估安全检测性能。
- **Safety-policy invocation**：从候选策略库中选择适用于当前轨迹的安全策略的过程。
- **Agent safeguard**：监控 Agent 行为并防止潜在风险的守护系统。
- **Policy-grounded rationale**：基于被调用策略条款生成的安全推理依据。

## 可复现要素
- **数据集**：PolicyTraj-20K 训练集已公开，包含训练集划分；30 条策略/358 条款的策略库已发布。
- **代码**：完整实现已开源，https://github.com/jianghoucheng/RePolicy，包含训练/评估 pipeline、奖励函数、GRPO 脚本、基准归一化代码等。
- **权重**：RePolicy-4B checkpoint 已发布。
- **关键超参**：SFT 学习率 1e-5，batch size 128，epochs 2；GRPO 学习率 1e-6，batch size 64，rollouts per prompt 16，KL coefficient 0.001，reward weights 0.1/0.3/0.6；训练硬件 8× NVIDIA H20 (96GB)。
