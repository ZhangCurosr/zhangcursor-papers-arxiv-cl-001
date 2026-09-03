---
title: "A-Token-Level-Analysis-of-Sampled-Token-Reverse-KL-On-Policy"
source: https://arxiv.org/pdf/2608.25643v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:38:44"
---

# 论文速读：A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation

## 一句话总结
本文从梯度层面解析了 on-policy distillation（OPD）中 per-token K2 估计量对 student logits 的梯度范数，发现其可分解为 teacher–student log-probability gap 与学生侧 softmax 因子 $1-\pi_S$ 的乘积，并由此提出 Surprise-aware Reweighting（SuRe）作为轻量干预方法，在 Qwen3 数学蒸馏实验中取得了显著性能提升。

## 研究问题与动机
1. **核心问题**：OPD 以 per-token K2 估计量优化 reverse KL 时，梯度在各采样 token 位置上的分配机制尚不明确，哪些 token 获得大梯度、为何获得大梯度缺乏理论刻画。
2. **现有分析视角不足**：已有 token 级研究多使用 entropy 或 sampled-token probability 等诊断量分析 token 不确定性，但无法捕获 K2 损失的有向梯度信息；entropy/JSD 等无向量难以有效排名梯度贡献。
3. **动机**：理解梯度分配可为设计更有效的重加权/选择策略提供理论依据，从而提升 OPD 在推理能力迁移上的效率。

## 核心贡献（创新点）
1. **K2 估计量的精确梯度恒等式**：推导出 per-token K2 损失对 student logits 梯度的 $\ell_1$ 范数可因式分解为 $2|\Delta\log p_t|\cdot(1-\pi_S(y_t|c_t))$，揭示了低概率 token 梯度更大的几何来源。
2. **实证 token 级梯度分配非均匀性**：在 Qwen3 数学蒸馏设定中，低学生概率 token 贡献了梯度范数总和的异常大份额，且同样富集大的 teacher–student gap，证实这是联合经验模式而非仅由学生侧因子决定。
3. **SuRe 轻量重加权干预**：基于上述分析提出一个有界（$w_t\in[1,1+\alpha]$）、detach 的学生侧权重规则，仅需单一超参 $\alpha$ 即可平滑放大小概率 token 的梯度贡献，无需额外参考模型或前向开销。

## 方法详解
1. **K2 估计量的梯度分解**：令 $\Delta\log p_t = \log\pi_T(y_t|c_t) - \log\pi_S(y_t|c_t)$，per-token 损失 $L_t^{\mathrm{RKL}} = \frac{1}{2}(\Delta\log p_t)^2$。利用 softmax Jacobian 恒等式 $\nabla_z\log\pi_S(y_t|c_t)=e_{y_t}-\pi_S(\cdot|c_t)$，通过链式法则得 $\nabla_z L_t^{\mathrm{RKL}} = -\Delta\log p_t\cdot(e_{y_t}-\pi_S(\cdot|c_t))$，取 $\ell_1$ 范数即得 $2|\Delta\log p_t|\cdot(1-\pi_S(y_t|c_t))$。方向由 $\Delta\log p_t$ 符号决定（teacher 更看好的 token 被提升）。
2. **SuRe 权重设计**：定义 detached 学生概率 $\bar{\pi}_{S,t} = \mathrm{sg}(\pi_S(y_t|c_t))$，权重 $w_t = 1+\alpha(1-\bar{\pi}_{S,t})$，其中 $\alpha\geq0$ 为惊喜系数。当 $\alpha=0$ 时退化为 vanilla OPD；$w_t$ 仅放大梯度而不改变方向。
3. **目标函数**：$\mathcal{L}_{\mathrm{RKL-rw}} = \frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}} w_t L_t^{\mathrm{RKL}}$，分母保持未加权 token 均值，不重新归一化权重，使得整体 loss scale 随 $\alpha$ 增大而上升。

## 实验与结果
- **设置**：teacher=Qwen3-8B，student=Qwen3-1.7B-Base 与 Qwen3-4B-Base；训练数据为 DeepMath hard split（57K，难度≥6）；32×H20，lr=$10^{-6}$，batch=512，2 epochs。
- **基线**：Base、KD、SeqKD、Vanilla OPD，以及与 GRPO 的对比。
- **主要结果（Table 1）**：
  - Qwen3-1.7B：SuRe（α=1.0）相对 Vanilla OPD 在 AMC23 pass@8 提升 **+7.5pp**（67.5→75.0），AIME24 pass@8 提升 **+6.7pp**（16.67→23.33）。
  - Qwen3-4B：AMC23 pass@8 提升 **+5.0pp**（85.0→90.0），AIME24 pass@8 提升 **+6.7pp**（30.0→36.67）。
- **OOD 结果**：CRUX、IFEval、MMLU-Pro 上 SuRe 与 Vanilla OPD 大致持平，未见明确退化，也未出现通用 OOD 提升。
- **对比 GRPO（Table 8）**：SuRe 在 avg@k 上全面持平或领先，pass@k 在三个基准上与 GRPO 相当或优于。
- **消融**：α 扫描显示 α=1.0 为最优；方向控制验证 surprise-aligned 优于 High-reweight 与 Random-reweight。

## 相关工作脉络
1. **On-Policy Distillation（GKD、MiniLLM）**：Agarwal et al.（2024）提出 K2 估计量的基于 loss 的 OPD 框架；本文与其互补，不改变 rollout 源或散度类型，而是分析 K2 reverse-KL OPD 内部的 token 级梯度分配。
2. **Token 级熵/不确定性分析（Wang et al., 2025; Huang et al., 2026）**：这些工作指出 RLVR 梯度集中于高熵少数 token；本文则直接刻画 sampled-token reverse-KL OPD 的梯度范数，并证明有向残差 $|\Delta\log p_t|$ 比熵/JSD 更能有效排名梯度贡献。
3. **Entropy-aware OPD（Jin et al., 2026）**：聚焦熵感知 token 选择；本文方法基于梯度恒等式的 student-side softmax 几何因子，与熵视角正交。
4. **Probability-based failure filtering（Li et al., 2026）**：关注概率过滤；本文提出平滑有界重加权而非硬阈值，保留 surprise token 的同时避免截断风险。
5. **Reward-based OPD（Yang et al., 2026; Kimi K3）**：将 teacher log-ratio 解释为 dense KL 约束 RL reward；本文揭示 K2 梯度与 Kimi K3 detached reward 方向相同但前者无 clip，提供了更精细的梯度级理解。
6. **RLVR token 选择（Ko et al., 2026）**：松弛 OPD 用于推理；本文工作在 OPD 理论层面对 token 梯度的刻画可与其方法结合。

## 局限性与未来方向
1. 仅研究 sampled-token reverse-KL OPD，未系统探索 full-vocabulary distillation 或使用 JSD 作为优化目标的变体。
2. 实验局限于数学推理领域，推理 trace 长度较短，结论对更长生成过程和其他领域（如代码）的泛化性待验证。
3. 计算资源限制下仅评估了两个较小 student 规模（1.7B/4B），大尺度模型上的效果未知。
4. SuRe 的控制实验未能完全分离精确 surprise assignment 与更通用的非均匀加权收益，机制归属仍有模糊空间。

## 研究启发与可借鉴点
1. **梯度恒等式驱动的轻量化干预范式**：先推导理论恒等式再设计简单权重规则，避免了 trial-and-error 调参；此方法论可迁移至其他基于 K2/反向 KL 的 distillation 变体。
2. **Detached bounded reweighting 的工程价值**：SuRe 无需额外 forward pass 或 reference model，实现代价极低（仅需一行代码插入权重），适合工业化部署中的快速迭代。
3. **有向残差 $|\Delta\log p_t|$ 作为 token 优先级排序器**：相比 entropy/JSD，该量保留了更新方向且统计上更能覆盖梯度范数总和，可作为 token 筛选/自适应学习率设计的通用指标。
4. **可结合的方向**：SuRe 可与 GRPO 等 RL 信号组合（作者已暗示），也可与 RLVR 的高熵 token 选择互补，为混合监督信号提供统一的 token 级分析框架。
5. **α 非单调性启示**：消融发现过大 α（α=2.0）反而削弱小 k 表现，提示 over-amplification 可能引发优化不稳定，后续需探索动态或自适应 α 调度策略。

## 关键术语表
- **On-Policy Distillation（OPD）**：在 student 自身采样轨迹上进行知识蒸馏的训练范式，通过保留 on-policy 数据减少分布偏移。
- **K2 估计量**：Schulman（2020）提出的 reverse KL 的有偏估计，形式为 $\frac{1}{2}(\Delta\log p)^2$，其 realized-loss 梯度在期望意义下无偏。
- **Reverse KL**：$D_{\mathrm{KL}}(\pi_S\|\pi_T)$，mode-seeking 散度，偏好 student 在高概率区域匹配 teacher，而非覆盖 teacher 全部支持。
- **Teacher–Student Log-Probability Gap（$\Delta\log p_t$）**：teacher 与 student 在同一采样 token 上的对数概率之差，决定更新方向与幅度。
- **Surprise-aware Reweighting（SuRe）**：本文提出的有界重加权规则，按 $w_t=1+\alpha(1-\pi_S)$ 放大 student 低概率（surprise）token 的梯度贡献。
- **Checkpoint-shift 统计量**：比较训练前后同一 token 的 log-probability 变化，用于诊断模型行为偏移，与 per-token K2 梯度分析正交。
- **Jensen–Shannon Divergence（JSD）**：对称化 KL 散度，用于衡量 teacher 与 student 分布的整体差异，本文作为对比诊断量使用。
- **$\ell_1$ 梯度范数**：论文选用的梯度度量，具有局部 logit 空间灵敏度解释，且产生简洁的因式分解形式。

## 可复现要素
- **数据集**：DeepMath hard split（57K，难度≥6），论文声明数据集可获取；MATH-500、AIME2024/2025、AMC23 为标准基准；CRUX、IFEval、MMLU-Pro 为 OOD 基准。
- **代码/权重**：论文未声明代码开源；使用了 Qwen3 系列模型（huggingface 公开权重）。
- **关键超参**：lr=$1\times10^{-6}$，batch=512，epochs=2，生成 temperature=1.0（训练）/0.7（评估），top-p=1.0（训练）/0.9（评估），SuRe 权重系数 α=1.0（主实验），rollout max token/GPU=24576（1.7B）/16384（4B），FSDP actor size=8，seed=42（主）/43（第二种子验证）。
