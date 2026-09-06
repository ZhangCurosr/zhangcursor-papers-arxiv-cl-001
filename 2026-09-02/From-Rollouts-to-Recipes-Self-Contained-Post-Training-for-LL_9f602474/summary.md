---
title: "From-Rollouts-to-Recipes-Self-Contained-Post-Training-for-LL"
source: https://arxiv.org/pdf/2609.01422v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:11"
field: "大语言模型后训练与强化学习"
keywords: ["post-training", "GRPO", "on-policy self-distillation", "behavior-conditioned routing", "verifier-based RL", "LLM reasoning"]
innovations: ["提出 Self-Routing，用 rollout 正确率与置信度将样本路由到 GRPO/OPSD/REG/SKIP 四种配方", "构建自包含 on-policy 信号回路，无需外部教师与额外标注", "给出可解释的四分支启发式路由公式与批次内置信度校准机制"]
benchmarks: ["GSM8K", "MATH-500", "AIME24", "AIME25", "MMLU-Pro", "GPQA-diamond", "SATBench", "AutoLogi", "LiveCodeBench-v5"]
---

# 论文速读：From Rollouts to Recipes: Self-Contained Post-Training for LLMs

## 一句话总结
论文提出 **Self-Routing**，一种基于 rollouts 行为状态的条件化 post-training 框架：利用当前模型自身的 rollout 正确率与置信度，将每个样本路由到 GRPO、on-policy 自蒸馏（OPSD）、正则化（REG）或跳过（SKIP）四种训练配方之一，从而在不引入外部教师、额外标注或额外采样的前提下实现样本级自适应优化。

## 研究问题与动机
- **全局配方忽略样本异质性**：现有 verifier-based post-training 对数据集中所有样本使用统一的优化目标，未充分利用当前策略 rollouts 中蕴含的样本级学习状态信号。
- **已有利用 rollouts 的方法局限**：all-correct/all-wrong 过滤、熵/置信度投票等仅用于“选哪些样本”或“推理时选答案”，并未回答“给定一个样本及其 rollout，应当施加何种优化信号”。
- **自蒸馏的分布错位风险**：基于离线 CoT 或外部教师的蒸馏容易与当前策略访问的状态产生分布偏移；使用 on-policy 信号可规避该风险。
- **训练价值动态变化**：样本的训练价值并非静态属性，而是随模型当前 rollout 行为和所施加优化信号共同变化；同一 prompt 在不同训练阶段适合不同配方。

## 核心贡献（创新点）
1. **提出样本级行为条件路由框架**：将 rollout 生成的行为信号（准确率+置信度）直接转化为训练配方分配决策，而非仅用于样本筛选。
2. **自包含的 on-policy 信号回路**：路由与蒸馏信号全部来自模型自身 rollout 与当前策略，无需外部教师、人工标注或额外推理开销。
4. **可解释且轻量级的路由设计**：通过高斯平滑分解准确率并校准批次内置信度，构建可解读的 Four-Branch 分配规则（GRPO / OPSD / REG / SKIP）。
5. **系统验证与训练动力学刻画**：在 Qwen3/Qwen3.5 多尺度模型上验证显著增益，并展示路由分布在训练过程中的动态演变规律。

## 方法详解
- **Rollout 收集**：每个样本 $x$ 由当前策略 $\pi_\theta$ 采样 $G$ 条回复，经 verifier 得到二元正确性 $R(o_i,x) \in \{0,1\}$，计算经验准确率 $a_x = \frac{1}{G}\sum_i R(o_i,x)$。
- **行为信号构建**：
  - **准确率分解**：将 $a_x$ 映射为低/中/高三个高斯成员度 $[l,m,h]$（中心分别为 0、0.5、1，$\sigma_l=\sigma_h=0.18,\sigma_m=0.16$），归一化后得 $\omega(a_x)$。
  - **置信度校准**：以序列 token 平均熵衡量不确定性，做批次内归一化得到相对置信度 $c_x$ 与批量偏差 $\Delta c_x$，构造 $\varphi(c_x)=(\tilde{c},\tilde{c}^+,\tilde{c}^-,\widetilde{\Delta c})$。
- **Recipe Router**：由 $\omega(a_x)$ 与 $\varphi(c_x)$ 计算四条路由分数并 softmax：
  - $s_{GRPO}=m+h(1-\widetilde{\Delta c})+l(1-\tilde{c})\tilde{c}$
  - $s_{OPSD}=l(1-\tilde{c}^-)$
  - $s_{REG}=h\cdot\widetilde{\Delta c}\cdot\tilde{c}^+$
  - $s_{SKIP}=l\cdot\tilde{c}^-\cdot\tilde{c}$
  按 Categorical 采样分配每个样本到一个队列。
- **四分支更新**：
  - **GRPO**：标准组内相对优势 + clipped ratio；
  - **OPSD**：同模型以题目+正确答案为条件生成 teacher 轨迹（离线预处理一次），学生 token-level 模仿；
  - **REG**：KL 惩罚靠近参考策略 $\pi_{ref}$（初始/上一 checkpoint）；
  - **SKIP**：当期无梯度。最终损失按活跃队列大小加权聚合。

## 实验与结果
- **训练集**：DAPO-Math-17K；**评测集**：GSM8K、MATH-500、AIME24、AIME25（ID 数学）、MMLU-Pro、GPQA-diamond（OOD 推理）；**模型**：Qwen3（0.6B/1.7B/4B）、Qwen3.5（0.8B/2B/4B）。
- **最强结果**：Qwen3-4B 上 Self-Routing 六榜宏平均 **73.7**，相对 Naive-GRPO（66.8）+6.9、相对 Naive-OPSD（70.4）+3.3；Qwen3.5-4B 达到 **86.6**（vs Naive-GRPO 79.8 / Naive-OPSD 83.0）。
- **关键结论**：
  - 在所有 backbone 上均 consistently 优于统一 GRPO、统一 OPSD、固定比例混合与随机/仅准确率路由；
  - 模型越大收益越明显；OPSD 在强 backbone 上系统性优于 GRPO；
  - OOD 上 GPQA-diamond 持续提升，MMLU-Pro 虽有回落但 Self-Routing 退化最小（仅次于 Base）；
  - 效率分析（normalized FLOPs，G=8）：Naive-GRPO 64.0 > Self-Routing 34.7 > Naive-OPSD 24.0，主要收益来自“昂贵配方仅作用于高信号子集”。
  - 训练动力学：早期 OPSD 占比>50%，中期 GRPO 上升，后期 REG 成为主导分支。

## 相关工作脉络
- **Verifier-based RL/RLVR**（PPO/GRPO/DAPO 系）：本文目标并非改进单一 RL 目标本身，而是研究在不同 rollout 状态下应分配何种配方。
- **All-correct/all-wrong 过滤与课程学习**：前者关注“选哪些样本/顺序”，本文进一步追问“选定样本应使用何种优化信号”。
- **Self-consistency / 置信度投票（推理时）**：把 rollout 不确定性仅用于答案选择；本文将其延伸到训练时配方路由。
- **On-policy distillation / Self-distillation**：本文采用同类思想，但以自包含、行为条件化为特征，避免外部教师分布错位。
- **Reward shaping / KL regularization / 长度控制**：与本文正交——本文不改进单配方内部细节，而是跨配方调度。
- **PODS / DAPO-style RL（Appendix J）**：在更强基线上，Self-Routing 仍取得 ID Math 80.9 / OOD-V 71.1 / General 59.3 的最优。

## 局限性与未来方向
- 缺乏对“为何某类 rollout 模式更适配某优化信号”的机制性或理论性解释，当前路由更多依赖直觉与实验证据。
- 主要在数学推理上验证；虽有 SATBench/AutoLogi/LiveCodeBench-v5 上的转移结果，但在 agent 规划、开放指令遵循等异质任务上的泛化尚不明。
- 聚焦路由视角，未同步优化 GRPO/OPSD 内部细节（reward engineering、trajectory refinement 等），两者可组合但本文未探索。
- 路由阈值 $\sigma$ 等超参固定，跨数据集/跨任务自适应调节未研究。
- 当前实现并非以 wall-clock 加速为目标，FLOPs 高于 Naive-OPSD。

## 研究启发与可借鉴点
1. **将 rollout 信号从“选样”拓展到“选配方”**：可迁移到 RLHF、offline RL、多目标蒸馏等场景，形成行为感知的训练调度器。
2. **高斯平滑分解 + 批次内置信度校准**：轻计算、免额外采样，可作为通用“行为状态编码器”复用到其他 on-policy 框架。
3. **多分支+skip 的动态权衡**：REG/SKIP 对稳定样本提供保守约束，缓解过度微调；启发我们在多目标训练中引入“何时不更新”的判别机制。
4. **可解释的启发式路由设计**：分数公式可直接映射到行为语义，便于 ablation 与工程调参；可与其他可解释分配（如基于 uncertainty/margin/entropy）结合。
5. **训练动力学可视化**：定期统计各分支占比并与性能曲线对齐，为诊断和优化提供直观反馈。

## 关键术语表
- **Self-Routing**：本文提出的自包含行为条件路由框架，将 rollout 正确性与置信度映射到四类训练配方。
- **GRPO（Group Relative Policy Optimization）**：基于组内相对优势的近端策略优化变体，适用于稀疏 outcome reward。
- **OPSD（On-Policy Self-Distillation）**：以当前模型或延迟版本为 teacher、在同分布 rollout 上做 token-level 模仿的自蒸馏方法。
- **REG（KL Regularization）**：通过 KL 散度约束更新后策略贴近参考策略，防止已稳定样本被激进优化破坏。
- **SKIP**：当期不对该样本施加梯度，避免在低信息量样本上浪费优化。
- **准确率分解（$\omega(a_x)$）**：将单值准确率平滑映射为低/中/高三元高斯成员度。
- **置信度校准（$\varphi(c_x)$）**：基于 token 预测熵并做批次内归一化，得到相对置信度与偏离量。
- **DAPO-Math-17K**：本文训练数据集，包含 17K 道数学推理题与正确答案。

## 可复现要素
- **数据集**：DAPO-Math-17K（训练）；评测集 GSM8K/MATH-500/AIME24/AIME25/MMLU-Pro/GPQA-diamond + SATBench/AutoLogi/LiveCodeBench-v5；论文声明计划开源代码与数据。
- **代码/权重**：代码计划开源；模型使用公开 Qwen3/Qwen3.5 系列。
- **关键超参**：每样本 rollout 数 $G=8$；准确率高斯中心 0/0.5/1，$\sigma_l=\sigma_h=0.18$、$\sigma_m=0.16$；批次内熵归一化；参考策略取初始/旧策略之一；ms-swift 后端；8× NVIDIA A100。
- **OPSD teacher 轨迹**：离线一次性由基座模型以题目+ground-truth answer 条件生成后复用，不引入额外在线推理。
