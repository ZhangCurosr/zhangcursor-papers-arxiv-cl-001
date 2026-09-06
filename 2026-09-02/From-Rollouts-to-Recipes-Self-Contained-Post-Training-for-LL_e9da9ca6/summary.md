---
title: "From-Rollouts-to-Recipes-Self-Contained-Post-Training-for-LL"
source: https://arxiv.org/pdf/2609.01422v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:20"
field: "大语言模型后训练与强化学习"
keywords: ["post-training", "GRPO", "on-policy self-distillation", "recipe routing", "verifier-based RL", "mathematical reasoning"]
innovations: ["基于 rollout 正确性与置信度构建可解释行为状态，将样本动态路由至 GRPO/OPSD/REG/SKIP 四路", "揭示混合正确性样本受益 GRPO、低正确低置信样本受益 OPSD、稳定已解样本受益 REG 的匹配规律", "在 Qwen3/Qwen3.5 多尺度上统一验证行为条件化路由优于均匀配方与简单路由基线"]
benchmarks: ["DAPO-Math-17K", "GSM8K", "MATH-500", "AIME24", "AIME25", "MMLU-Pro", "GPQA-diamond", "SATBench", "AutoLogi", "LiveCodeBench-v5"]
---

# 论文速读：From-Rollouts-to-Recipes-Self-Contained-Post-Training-for-LL

## 一句话总结
论文提出 Self-Routing，一种自包含的行为条件化后训练框架，利用当前策略自身生成的 rollout 正确性与置信度信号，将每个样本动态路由到 GRPO、on-policy 自蒸馏（OPSD）、正则化或跳过四种训练配方中，无需外部教师、额外标注或额外采样，在数学推理任务上显著优于均匀应用单一配方的基线。

## 研究问题与动机
- **全局配方的结构性局限**：现有验证器驱动的后训练方法对数据集中所有样本应用相同的优化机制或固定混合目标，忽视了 rollout 本身已包含的样本级学习状态异质性。
- **既有信号的片面利用**：all-correct/all-wrong 过滤仅用于筛选样本（决定"是否训练"），置信度/熵投票多用于推理时答案选择，均未回答"给定样本及其 rollout，应施加何种优化信号"。
- **外部蒸馏的分布失配风险**：离线 CoT 轨迹或外部教师生成的目标与当前策略实际访问的状态存在分布偏移，可能引入次优学习目标。
- **样本训练价值的情境依赖性**：同一问题的训练价值并非静态属性，而取决于当前策略的 rollout 行为与所施加优化信号的组合——全错样本对 RL 梯度贡献有限但可受益于密集蒸馏信号，稳定已解样本应受保守约束，高置信失败样本则适合暂时跳过。

## 核心贡献（创新点）
1. **提出自包含的行为条件化路由机制**：将 rollout 正确性与 token 级预测熵置信度融合为行为状态，通过可解释的评分函数将样本分配至 GRPO/OPSD/REG/SKIP 四个不相交队列，实现样本级训练配方自适应。
2. **揭示 rollout 行为状态与优化信号的匹配规律**：通过 Qwen3-4B 上的预诊断实验表明，混合正确性样本适合 GRPO（利用组内奖励对比），低正确低置信样本适合 OPSD（密集修正信号），稳定已解样本适合 REG（防止过度漂移），高置信失败样本适合 SKIP（无有效训练信号）。
3. **构建统一后训练框架下的系统对比**：在 DAPO-Math-17K 上统一测试 Self-Routing 与 Naive-GRPO、Naive-OPSD、DAPO-style RL、PODS 及三种路由基线（随机、固定比例、仅准确率），证明行为条件化路由的通用优势。

## 方法详解
- **Rollout 收集**：当前策略 $\pi_\theta$ 为每个样本 $x$ 生成 $G$ 条 on-policy rollout，Verifier 返回二元结果 $R(o_i, x) \in \{0,1\}$，样本经验准确率 $a_x = \frac{1}{G}\sum_i R(o_i, x)$。
- **行为信号构造**：
  - **准确率分解**：将 $a_x$ 映射为三个平滑隶属度 $l_x, m_x, h_x$（高斯核，中心分别为 0/0.5/1，$\sigma_l=\sigma_h=0.18, \sigma_m=0.16$），归一化得 $\omega(a_x)=[l,m,h]$。
  - **置信度校准**：计算每条 rollout 的 token 级预测熵均值，转换为 batch 内归一化置信度 $c_x$，再计算相对偏差 $\Delta c_x = c_x - \bar{c}_B$，得到 $\varphi(c_x)=(\tilde{c}, \tilde{c}^+, \tilde{c}^-, \widetilde{\Delta c})$。
- **路由评分函数**：
  - $s_{GRPO} = m + h(1-\widetilde{\Delta c}) + l(1-\tilde{c})\cdot\tilde{c}$
  - $s_{OPSD} = l(1-\tilde{c}^-)$
  - $s_{REG} = h\cdot\widetilde{\Delta c}\cdot\tilde{c}^+$
  - $s_{SKIP} = l\cdot\tilde{c}^-\cdot\tilde{c}$
  - 经 softmax 归一化后按 Categorical 采样决定配方 $t_x$。
- **四路训练目标**：
  - **GRPO 分支**：标准 clipped PPO 风格优势估计，利用组内奖励方差构造相对 advantage。
  - **OPSD 分支**：以同一基模型在问题+答案条件下的离线生成轨迹为教师，对学生当前输出做 token 级 SFT 式模仿，teacher trajectory 预处理一次后复用。
  - **REG 分支**：KL 正则化约束当前策略贴近 reference policy（初始模型/旧策略/中间 checkpoint）。
  - **SKIP 分支**：本轮无梯度贡献。
- **最终损失**：仅聚合前三队列，按样本数加权 $\mathcal{L} = \frac{n_g\mathcal{L}_{GRPO} + n_o\mathcal{L}_{OPSD} + n_r\mathcal{L}_{REG}}{n_g+n_o+n_r}$。

## 实验与结果
- **数据集**：训练集 DAPO-Math-17K；ID 评测 GSM8K、MATH-500、AIME24、AIME25；OOD 评测 MMLU-Pro、GPQA-diamond。
- **模型**：Qwen3-0.6B/1.7B/4B、Qwen3.5-0.8B/2B/4B，共 6 个尺度。
- **最强结果**：
  - Qwen3-4B：Self-Routing 平均 73.7，相对 Base(61.0) 提升 +12.7，相对 Naive-GRPO(66.8) 提升 +6.9，相对 Naive-OPSD(70.4) 提升 +3.3。
  - Qwen3.5-4B：Self-Routing 平均 86.6，相对 Naive-GRPO(79.8) 提升 +6.8，相对 Naive-OPSD(83.0) 提升 +3.6。
  - OOD 泛化：Qwen3-4B 上 SATBench/AutoLogi/LiveCodeBench-v5 平均 71.1，优于 DAPO-style RL(69.1) 和 PODS(69.1)。
- **效率分析**：$G=8$ 时归一化 FLOPs 为 34.7（Naive-GRPO 为 64.0，Naive-OPSD 为 24.0）；虽非最快，但昂贵配方仅作用于有信号的子集。
- **路由动态**：训练早期 OPSD 占主导（>50%），中期 GRPO 上升达峰值，后期 REG 成为主力，SKIP 全程保持低位并持续下降。

## 相关工作脉络
- **Verifier-Based RL/RLVR**：GRPO（Shao et al., 2024 DeepSeekMath）、DAPO（Yu et al., 2026）、PPO/GRPO/DAPO 系列改进——本文不设计新目标，而是研究如何根据 rollout 状态在不同目标间自适应切换。
- **Data Selection & Curriculum Learning**：all-correct/all-wrong 过滤（Xu et al., 2025 PODS；Jiang et al., 2025 VCRL）——前者回答"是否选样本"，本文回答"选定样本后施加何种优化信号"，二者互补。
- **On-Policy Self-Distillation**：Hübotter et al. (2026) RLSVD、Zhao et al. (2026) Self-Distilled Reasoner——本文沿用 OPSD 作为 primitive，但将其纳入路由框架而非单独使用。
- **Self-Specialized Teacher**：Li et al. (2026b)——避免 replay buffer 依赖；本文与之正交，关注路由而非教师构建。
- **推理时不确定性利用**：Self-Consistency (Wang et al., 2022)、ToT (Yao et al., 2023)、MUR (Yan et al., 2026b)——本文首次将 rollout 行为信号从推理时适配扩展至训练时配方路由。
- **Reward/Advantage 工程**：KL 正则、length control、entropy bonus 等——本文不改进单算法内部设计，聚焦跨算法层面的样本级调度。

## 局限性与未来方向
- **机制解释不足**：路由行为主要由直觉与实证支撑，缺乏对"为何特定 rollout 模式与特定优化信号匹配"的严格理论或机制分析。
- **任务迁移边界待探**：实验集中于数学推理，对 agent planning、open-ended instruction following 等非结构化/开放域设定是否泛化尚不清楚。
- **单次离线 teacher 生成假设**：OPSD teacher trajectory 预处理一次后复用，未考虑训练过程中模型能力变化带来的目标漂移。
- **路由超参未调优**：$\sigma_l, \sigma_m, \sigma_h$ 等固定为默认值，跨数据集/跨尺度是否需适配未充分验证。
- **效率瓶颈**：当前实现保留多 rollout 行为估计且部分样本仍执行 GRPO 更新，wall-clock 成本高于 Naive-OPSD，仅在"精确分配"层面节省。

## 研究启发与可借鉴点
- ** rollout 行为信号可作为训练调度器输入**：正确率+置信度的双维度分解比单一 loss/准确率更丰富，可迁移至代码生成、agent RL 等其他 verifier-based 场景。
- **可解释路由评分函数设计**：基于行为状态先验手动构造 $s_{GRPO}/s_{OPSD}/s_{REG}/s_{SKIP}$ 而非端到端学习，便于诊断与调整，适合资源受限团队快速部署。
- **统一 backend 下的公平对比范式**：所有方法共享 ms-swift launcher、rollout 接口、verifier、tokenizer 与 checkpointing，排除实现差异干扰，值得复用。
- **路由动态可视化作为训练诊断工具**：Figure 4 展示四路占比随步数演变，可直接反映模型学习阶段与配方需求变化，建议纳入后续实验报告模板。
- **离线 teacher 预生成+复用策略**：一次性生成答案条件 trajectory 并在训练中重复使用，避免额外推理开销，可与延迟策略（delayed policy）结合进一步扩展。

## 关键术语表
- **Self-Routing**：本文提出的行为条件化后训练框架，根据 rollout 正确性与置信度动态分配样本至 GRPO/OPSD/REG/SKIP 四路。
- **GRPO**：Group Relative Policy Optimization，基于组内奖励方差构造相对 advantage 的 PPO 变体，适用于奖励信号有对比性的样本。
- **OPSD**：On-Policy Self-Distillation，以同一模型在问题+答案条件下的离线生成为教师，对学生当前输出做 token 级 SFT 式模仿。
- **Behavior State**：由准确率隶属度 $\omega(a_x)$ 与校准后置信度 $\varphi(c_x)$ 组成的二维状态向量，作为路由器的输入特征。
- **Verifier-based Post-Training**：利用最终答案正确性/测试通过等 outcome reward 进行 RL 训练，无需 CoT 标注或过程监督。
- **Routing Score**：由可解释公式计算的 GRPO/OPSD/REG/SKIP 四类偏好值，经 softmax 归一化后采样决定配方。
- **All-correct/all-wrong Filtering**：跳过组内全对或全错样本的常见筛选策略，仅提供二值信号，无法区分中间状态。
- **On-Policy Signal**：由当前策略实时生成而非来自离线数据集的 rollout/熵/置信度等训练动态信息。

## 可复现要素
- **数据集**：DAPO-Math-17K（训练）；GSM8K、MATH-500、AIME24、AIME25、MMLU-Pro、GPQA-diamond（评测）——论文声明计划开源代码与数据。
- **代码/权重**：论文 Ethics 部分声明 "We plan to release our code and data"，截至发表时未明确给出仓库链接；基座模型为 Qwen3/Qwen3.5 系列（HuggingFace 公开）。
- **关键超参**：$G=8$ 条 rollout/样本；$\sigma_l=\sigma_h=0.18, \sigma_m=0.16$；batch 内归一化熵→置信度；Categorical 采样路由；reference policy 取初始模型/旧策略/中间 checkpoint 之一。
- **训练环境**：ms-swift 框架，PyTorch + HuggingFace Transformers，单节点 8× NVIDIA A100，混合精度。
