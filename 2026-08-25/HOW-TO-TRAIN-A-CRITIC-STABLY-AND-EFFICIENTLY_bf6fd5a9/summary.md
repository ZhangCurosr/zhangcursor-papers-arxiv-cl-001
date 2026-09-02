---
title: "HOW-TO-TRAIN-A-CRITIC-STABLY-AND-EFFICIENTLY"
source: https://arxiv.org/pdf/2608.23566v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:51:16"
field: "大语言模型强化学习"
keywords: ["reinforcement learning", "critic training", "LLM alignment", "DPPO", "GAE", "privileged information"]
innovations: ["提出 BPCO 配方，整合有界值预测、蒙特卡洛目标、未归一化优势与长度自适应 GAE 实现稳定单 rollout critic 训练", "系统验证 privileged information 对 critic 加速学习的作用及过拟合权衡"]
benchmarks: ["AIME 2025 avg@32", "DeepScaleR", "DAPO-Math-17k", "OpenRubrics"]
---

# 论文速读：HOW-TO-TRAIN-A-CRITIC-STABLY-AND-EFFICIENTLY

## 一句话总结
本文针对大语言模型（LLM）强化学习中 critic-based 训练不稳定的问题，提出 **Best-Practice Critic Optimization (BPCO)** 配方，通过组合 DPPO、有界值预测、无偏蒙特卡洛目标、未归一化策略优势及长度自适应 GAE 等技术，实现单 rollout  critic 的稳定高效训练，并在数学推理与 rubric-based 奖励任务上匹配或超越 group-based 基线。

## 研究问题与动机
- **Critic-based RL 训练脆弱**：标准 PPO 的 ratio clipping 对高低概率 token 处理不对称；bootstrap value targets 会继承 critic 误差；固定 GAE 参数 $\lambda$ 使短长响应的终端奖励权重差异过大。
- **值头无界与归一化误导**：线性值头可能预测超出奖励范围的值；batch-wise advantage 归一化在策略接近最优时反而放大噪声，破坏更新自然衰减。
- **Privileged information 未被充分利用**：critic 仅在训练时使用，可向其输入奖励定义信息（如参考答案、评分标准），而不改变策略输入与部署需求，但以往工作未系统研究。
- **缺乏受控消融验证**：现有方法常混合多种技巧，难以辨识各设计对稳定性的贡献；本文通过逐步消融揭示关键失效机制。

## 核心贡献（创新点）
1. **提出 BPCO 集成配方**：首次系统整合 DPPO、有界值预测、蒙特卡洛 critic 目标、未归一化优势与长度自适应 GAE，形成一套单 rollout critic 训练的稳定流程。  
   *与已有工作的区别*：不同于 VC-PPO、VAPO、SAO 等仅关注单一改进，BPCO 强调各组件间的协同一致性（输出范围、目标、输入、策略信号对齐）。
2. **证明有界值预测（bounded value prediction）的必要性**：使用缩放反正切函数将 critic 输出约束到已知奖励区间 $[R_{\min}, R_{\max}]$，避免极端预测导致训练崩溃。  
   *与已有工作的区别*：标准实现普遍使用线性头；本文通过受控实验证明该约束对稳定性至关重要，且在大模型规模下仍有效。
3. **采用无偏蒙特卡洛 critic 目标（unbiased MC target）**：将 critic 训练目标设为观测到的最终回报 $R(x,y)$，与策略 advantage 估计解耦（$\lambda_V=1$ 而 $\lambda_\pi=0.99$）。  
   *与已有工作的区别*：不同于 VC-PPO 仅解耦优势与目标，本文明确证明 bootstrap target 会导致 explained variance 虚高且政策训练不稳定，必须回归真实回报。
4. **去除 batch-wise advantage 归一化**：直接使用原始 GAE 优势更新策略，避免小方差时噪声被放大及正 advantage 被减均值翻转符号。  
   *与已有工作的区别*：多数 PPO 实现默认归一化；本文证明该操作在接近最优时破坏策略更新的自然收缩，且移除后验证性能更优。
5. **探索 privileged information 辅助 critic**：向 critic 显式输入参考解答或评分 rubric，加速学习但不改变策略输入；揭示其收益与过拟合风险的权衡。  
   *与已有工作的区别*：类比多智能体集中训练-分布式执行，首次在 LLM RL 中系统分析 privileged input 对 critic 拟合与泛化的影响。

## 方法详解
BPCO recipe 的关键设计如下：

- **DPPO 策略更新**：采用 Divergence PPO（Equation 2），将 clipping boundary 设为 $\epsilon/\mu(y_t|s_t)$，等价于约束采样 token 的概率绝对变化不超过 $\epsilon$，替代标准 PPO 的相对比率裁剪，缓解高低概率 token 不对称问题。
- **有界值预测**：值头使用缩放反正切参数化（Equation 9）：
  $$
  V_\phi(s_t) = R_{\min} + (R_{\max}-R_{\min})\left(\frac{1}{2}+\frac{1}{\pi}\arctan(z_\phi(s_t))\right),
  $$
  确保输出严格落在奖励区间内。
- **解耦 GAE 与蒙特卡洛目标**：策略 advantage 使用 $\lambda_\pi<1$（如 0.99）以减少方差；critic 目标使用 $\lambda_V=1$，使得目标退化为最终回报 $R(x,y)$（Equation 11），消除 bootstrap 误差传播。
- **未归一化优势**：直接使用 $\widehat{A}_t^{\text{GAE}(\lambda_\pi)}$ 作为策略更新信号，不执行 Equation 12 的 batch 内标准化。
- **长度自适应 GAE**：$\lambda_\pi$ 随响应长度 $L$ 调整（Equation 14）：
  $$
  \lambda_\pi(L) = 1 - \frac{1}{\alpha L},
  $$
  短响应使用较大 $\lambda$（更多 bootstrap），长响应使用较小 $\lambda$（更接近 Monte Carlo），平衡偏差与方差。
- **Privileged information**：critic 额外接收与 prompt 绑定的奖励定义信息 $q(x)$（如参考答案、rubric），估计条件价值函数 $V_\phi^\mu(s_t, q(x))$，策略仍只输入 $x$ 与历史 token。

## 实验与结果
- **数据集**：
  - 受控 sanity test：1,460 道数学题（初始模型可解）。
  - DeepScaleR：40.3K 数学问题-答案对（约 7.3K 含官方解答）。
  - DAPO-Math-17k：用于 30B-A3B MoE 模型评估。
  - OpenRubrics：rubric-based 奖励任务。
- **模型**：DeepSeek-R1-Distill-Qwen-1.5B、Qwen3-4B-Base、Qwen3-30B-A3B-Base / Qwen3-30B-A3B。
- **基线**：
  - Group-based：Dr. GRPO（group size=16）。
  - Critic-based：带 decoupled GAE 与 length-adaptive GAE 的标准配方（保留无界值头与优势归一化）。
  - BPCO 变体：BPCO+Ans（附参考答案）、BPCO+Sol（附官方解答）。
- **主要结果**：
  - 在小数据 sanity test 中，BPCO 各组件依次验证稳定性提升（图 1–6）。
  - DeepScaleR 上，BPCO 持续优于 critic baseline 与 group baseline，explained variance 更高，AIME 2025 avg@32 显著提升（图 7–9）。
  - 30B-A3B MoE 模型在 DAPO-Math-17k 上，BPCO 大幅改善 critic baseline 的优化崩溃问题，并在 Qwen3-30B-A3B 上超越 Dr. GRPO（图 10）。
  - Rubric 奖励任务（图 11）：BPCO（无 privileged info）优于所有基线，证明有界值与未归一化优势普适有效；privileged info 加速学习但未带来最终性能提升（任务较简单）。
- **最强结果**：在 Qwen3-30B-A3B 模型上，BPCO 在 AIME 2025 上取得最高准确率，相对 critic baseline 提升显著，且与 group-based Dr. GRPO 相比持平或更优。

## 相关工作脉络
- **PPO / DPPO**：本文以 DPPO 为策略更新基础，改进其 trust region 定义，解决标准 PPO 对高低概率 token 裁剪不对称问题（Qi et al., 2026b）。
- **GRPO / Dr. GRPO**：Group-based 方法无需 critic 但需多次 rollout；本文证明单 rollout critic 经 BPCO 设计后可达到相当效率，避免组内归一化带来的 prompt 重加权。
- **VC-PPO**：Yuan et al. (2025) 提出 decoupled GAE 分离优势与目标；本文在此基础上明确 critic 目标必须使用 Monte Carlo 回报（$\lambda_V=1$），否则 explained variance 虚高。
- **VAPO / SAO**：Yue et al. (2025)、Hou et al. (2026) 引入 length-adaptive GAE；本文将其整合进 BPCO 并证明其与 $\lambda_\pi$ 固定值的权衡。
- **Privileged information 在 RL**：类比多智能体集中训练-分布式执行（Vinyals et al., 2019; Wang et al., 2021; Amato, 2024），本文首次将此类思想用于 LLM RL 的 critic 输入扩展。

## 局限性与未来方向
- **任务范围局限**：实验仅涵盖数学推理与 rubric-based 奖励，未验证于开放对话、代码生成等复杂领域。
- **奖励范围需已知**：BPCO 假设 $[R_{\min}, R_{\max}]$ 已知，对于动态或未知奖励分布需额外设计。
- **Privileged variants 依赖评估信息**：参考答案或 rubric 并非所有任务可得；且小数据下易过拟合。
- **计算与内存开销**：critic 训练增加额外参数与显存，本文未与 trajectory-matched 的 group 方法比较实际吞吐量。
- **未来方向**：扩展至多模态 RL、自动估计奖励范围、动态 privileged info 选择机制、结合长上下文推理场景的优化。

## 研究启发与可借鉴点
- **受控消融范式**：通过“应能拟合小规模可解数据集”的 sanity test 快速暴露优化缺陷，值得在 RLHF 算法研发中推广。
- **组件协同设计意识**：BPCO 各技巧（有界输出、MC 目标、未归一化优势、自适应 λ）共同保证 critic 输出、目标、输入与策略信号的一致性，提示后续工作需系统性对齐各模块假设。
- **Privileged information 思路迁移**：可将任务特定知识（如评分标准、中间验证信号）作为 critic 额外输入，在不改变策略部署的前提下加速训练，适用于有隐性评估维度的任务。
- **实验设计严谨性**：分别评估 scaling to larger dataset、larger models、不同奖励类型，层层验证通用性；每组均报告 explained variance 等过程指标，增强结论可信度。

## 关键术语表
- **DPPO（Divergence Proximal Policy Optimization）**：将 PPO 的 clipping boundary 由概率比率改为概率绝对变化，使高低概率 token 受到对称约束。
- **GAE（Generalized Advantage Estimation）**：通过指数加权求和 TD 残差估计 advantage，$\lambda$ 控制 bootstrap 程度。
- **Privileged information**：训练时提供给 critic 但推理时不可用的额外信息（如参考答案、评分 rubric）。
- **Bounded value prediction**：使用有界激活函数（如缩放 arctan）限制 critic 输出在已知奖励区间内。
- **Monte Carlo value target**：critic 训练目标直接使用观测到的最终回报，而非 bootstrap 估计。
- **Batch-wise advantage normalization**：对每个 minibatch 内的 advantage 做零均值单位方差标准化，本文证明其有害。
- **Length-adaptive GAE**：$\lambda_\pi$ 随响应长度动态调整，平衡偏差与方差。
- **Explained variance**：衡量 critic 对目标变量方差的解释比例，用于诊断价值函数拟合质量。

## 可复现要素
- **数据集**：
  - DeepScaleR：40.3K 数学题，官方未明确声明公开，但常见于开源社区。
  - DAPO-Math-17k：来自 DAPO 系统开源项目，可公开获取。
  - OpenRubrics：来自 ACL 2026 论文，数据集随论文开源。
  - Sanity test 1,460 题：未说明来源，可能为内部构造。
- **代码/权重**：代码已开源（https://github.com/QPHutu/golden_critic）；模型权重使用开源的 DeepSeek-R1-Distill-Qwen-1.5B、Qwen3 系列。
- **关键超参**：
  - 学习率：策略 $10^{-6}$，critic $10^{-5}$。
  - Batch size：1,024 trajectories，minibatch 256。
  - 迭代次数：sanity test 1,500 步。
  - $\lambda_\pi$：固定 0.99 或长度自适应（$\alpha=0.4$）。
  - $\lambda_V$：固定 1。
  - DPPO $\epsilon$：论文未明确数值，需在代码中确认。
  - 最大响应长度：24k tokens。
