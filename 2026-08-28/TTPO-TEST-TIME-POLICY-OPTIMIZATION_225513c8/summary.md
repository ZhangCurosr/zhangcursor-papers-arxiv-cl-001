---
title: "TTPO-TEST-TIME-POLICY-OPTIMIZATION"
source: https://arxiv.org/pdf/2608.27448v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:48:29"
field: "大语言模型推理增强"
keywords: ["测试时训练", "策略优化", "自蒸馏", "强化学习", "无标签学习", "数学推理"]
innovations: ["提出非对称目标将OPSD蒸馏与GRPO惩罚分别匹配正负样本，实现伪标签噪声鲁棒的TTT", "双分支token级选择机制：蒸馏分支下加权已收敛位置，RL分支掩码仅惩罚置信错误", "无标签TTPO在竞赛级数学基准上匹配/超越有标签OPSD，Qwen3-1.7B提升7.2个百分点"]
benchmarks: ["AIME 2025", "AIME 2026", "HMMT 2025", "HMMT 2026", "BRUMO 2025"]
---

# 论文速读：TTPO-TEST-TIME-POLICY-OPTIMIZATION

## 一句话总结
TTPO 提出了一种异步测试时策略优化方法，通过 Majority Vote 伪标签将 rollouts 分为正负样本，正样本用 OPSD 蒸馏、负样本用 GRPO 惩罚，结合 token 级选择机制，在无标签场景下实现了可比甚至优于有标签方法的数学推理能力提升。

## 研究问题与动机
- 现有 RL 和 OPSD 方法均依赖 ground-truth 标签，无法直接应用于无标签的测试时训练（TTT）场景。
- 用 Majority Vote 伪标签替代 ground truth 是自然思路，但伪标签错误时会污染 teacher 并误导所有 token；论文发现这种错误传播具有**非对称性**：即使伪标签错误，约 79% 的不一致 rollouts 也是真正错误的，因此惩罚不一致样本是安全的。
- 纯 RL 方法（如 TTRL）仅提供序列级标量奖励，缺乏 token 级密集监督；纯蒸馏方法对所有样本一视同仁，误差传播范围大。
- 需要一种既能利用正样本密集监督、又能安全惩罚负样本的**非对称目标函数**。

## 核心贡献（创新点）
1. **非对称目标设计**：对 majority vote 一致样本（P）用 OPSD 前向 KL 蒸馏，对不一致样本（N）用 GRPO 惩罚，将两种信号的可靠性与样本类型匹配。与已有工作的本质区别在于打破了"统一处理所有样本"的范式，利用错误传播的非对称性实现误差鲁棒。
2. **双分支 token 级选择机制**：蒸馏分支用学生熵与教师-学生散度联合加权，抑制已收敛位置；RL 分支用置信错误掩码，仅惩罚导致失败的低概率低熵 token。与已有工作的本质区别在于两个分支分别针对不同失败模式设计，形成互补。
3. **无标签匹配有标签性能**：在五个竞赛级数学基准上，TTPO（无标签）匹配或超越有标签 OPSD，Qwen3-1.7B TTT 从 38.0% 提升至 45.2%，关闭 thinking mode 后提升 +25.2% 至 +36.4%。与已有工作的本质区别在于证明了伪标签+异步设计可在零标签下达到甚至超过 Ground-Truth 监督效果。

## 方法详解
**整体框架**：对每个问题 x 采样 K=64 条轨迹，提取答案 a_k，按数学等价性聚类得到 majority vote 伪标签 â，将轨迹分为正样本集 P（答案=â）和负样本集 N（答案≠â）。

**教师构造**：同 OPSD，将同一模型以伪标签 â 作为特权信息 conditioning，生成教师分布 $q_t^{(\hat{a})} = \pi_\theta(\cdot | [x; \hat{a}]_{teacher}, y_{<t})$（无梯度），学生分布 $p_t = \pi_\theta(\cdot | x_{student}, y_{<t})$（有梯度）。

**正样本分支（OPSD）**：
$$\mathcal{L}_{OPSD}(k) = \frac{1}{T_k}\sum_{t=1}^{T_k} w(t) \cdot KL(q_t^{(\hat{a})} \| p_t)$$
Token 加权：$w(t) = \hat{H}(t) + \hat{\Delta}(t) - \hat{H}(t)\cdot\hat{\Delta}(t)$，其中 H 为学生熵、Δ 为教师-学生散度，经 per-sample min-max 归一化后以 Soft-OR 组合，仅在两处均收敛时权重趋零。

**负样本分支（GRPO）**：
$$\mathcal{L}_{GRPO}(k) = -\frac{A_k}{T_k}\sum_{t=1}^{T_k} m(t) \cdot \log\pi_\theta(y_k^{(t)} | x, y_k^{(<t)})$$
Token 掩码：$s(t) = -\log p_t(y_k^{(t)}) \cdot (1-\hat{H}(t))$，取 top-50% 高得分 token 构建二值掩码 m(t)，优先惩罚低概率且低熵的"置信错误"。

**统一目标**：$\mathcal{L}_{TTPO} = \frac{1}{|B|}\left(\sum_{k\in P}\mathcal{L}_{OPSD}(k) + \lambda\sum_{k\in N}\mathcal{L}_{GRPO}(k)\right)$，其中 λ=0.1 平衡两分支梯度量级。

## 实验与结果
- **数据集**：OpenThoughts 训练数据 + 五个竞赛级数学基准（AIME 2025/2026、HMMT 2025/2026、BRUMO 2025）。
- **基线**：GRPO†（有标签）、OPSD†（有标签）、TTRL（无标签）、OPSD-TTT（无标签）。
- **主要结果（Table 1，OpenThoughts 有标签训练）**：TTPO 在 Qwen3-1.7B/4B/8B 上均超越 OPSD†，平均准确率分别为 40.1 vs 39.7、58.6 vs 58.4、62.6 vs 61.7。
- **主要结果（Table 2，纯 TTT 无标签）**：Qwen3-1.7B 上 TTPO 达 45.2% 平均，较 base（38.0%）+7.2 绝对提升，领先 OPSD-TTT（41.9%）+3.3、领先 TTRL（40.2%）+5.4。关闭 thinking mode 后（Table 7）平均提升 +25.2%~+36.4%，数倍于 OPSD。
- **最强结果**：Qwen3-8B + TTPO 在 BRUMO25 达 73.1%，平均 62.6%；TTPO-4B（61.1%）已超越 Qwen3-8B base（60.7%）。
- **泛化**：在单一基准训练可提升其他两个基准（Figure 4），证明获得的是可迁移推理能力而非过拟合。

## 相关工作脉络
- **TTRL (Zuo et al., 2026)**：无标签 RL 通过 majority vote 伪奖励训练，但仅用序列级二元奖励，缺乏 token 级密集监督；TTPO 在此基础上引入 distillation 分支实现密集监督。
- **OPSD (Zhao et al., 2026)**：有标签 on-policy 自蒸馏，依赖 ground-truth 作为特权信息；TTPO 将其推广至无标签场景，用 majority vote 替代 ground truth 并设计非对称目标。
- **Hi-TTRL (Xu et al., 2026b) / SCRL (Yan et al., 2026)**：改进 TTRL 对共识质量的敏感性，但仍停留在纯 RL 框架；TTPO 引入 distillation 信号突破纯 RL 的信息瓶颈。
- **TIP (Xu et al., 2026c)**：token 重要性分析用于蒸馏，发现 10% token 即可匹配全 token 性能；TTPO 借鉴其思想设计双分支 token 级选择。
- **STAPO (Liu et al., 2026a)**：掩码 RL 中低概率低熵 token 以稳定训练；TTPO 将此思路迁移至负样本惩罚分支。
- **NLNL (Kim et al., 2019)**：负学习证明"告知样本不是什么"在噪声标签下仍可靠；TTPO 的理论动机与此同源——对负样本的惩罚不依赖伪标签具体内容。

## 局限性与未来方向
- **依赖 majority vote 质量**：当 K 过小或问题极难导致无正确 rollout 时，投票信号退化，两分支均受噪声干扰；可探索自适应正负比例或低共识回退到纯 RL。
- **领域限制**：目前仅验证于可验证答案的数学推理；扩展到代码生成（需执行验证）或开放式推理（需 learned reward model）尚未探索。
- **固定非对称目标**：训练过程中 pseudo-label 准确性逐渐提升，最优的蒸馏/RL 权重可能动态变化；可探索 curriculum 动态调整 λ 或正负比例。

## 研究启发与可借鉴点
1. **非对称信号匹配**：将不同可靠性级别的监督信号匹配到对应可靠性的样本子集，这一设计思路可迁移至其他有噪声标签的场景（如 noisy label learning、self-training）。
2. **双分支 token 级选择**：蒸馏分支"抑制已收敛位置"与 RL 分支"仅惩罚置信错误"形成互补，两类机制的组合策略值得在其他 RLVR/蒸馏工作中复用。
3. ** Majority Vote 的动态价值**：论文发现 vote-based routing 比 ground-truth routing 在难问题上更稳定（GT 导致 P 集接近空集，两分支均失效），这一现象对其他需要路由的工作有启示。
4. **Self-evolution 循环设计**：Avg@12 与 Maj@12 同步上升的自进化机制表明，训练信号质量可随模型能力提升而自动改善，这一动态监控指标可用于评估 TTT 方法的健康度。
5. **Thinking→Non-thinking 蒸馏效率**：TTPO 关闭 thinking mode 后仍获得巨大提升（+25~36 points），说明不对称目标能有效将 reasoning 能力转移到非 thinking 分布，这一 transfer 效率可作为评估蒸馏方法的新指标。

## 关键术语表
- **TTPO (Test-Time Policy Optimization)**：测试时策略优化，本文提出的无标签 TTT 方法，通过非对称目标结合蒸馏与 RL。
- **OPSD (On-Policy Self-Distillation)**：on-policy 自蒸馏，以模型自身在 privileged information（如 ground-truth/答案）conditioning 下的输出来指导当前 policy 训练。
- **GRPO (Group Relative Policy Optimization)**：Group Relative Policy Optimization，DeepSeekMath 提出的 RL 训练方法，通过组内相对 advantage 计算策略梯度。
- **TTT (Test-Time Training)**：测试时训练，在推理时对模型进行在线微调以适应当前输入分布。
- **Majority Vote Pseudo-label**：多数投票伪标签，对多条 rollout 的答案聚类取最多票作为伪真值。
- **Forward KL Distillation**：前向 KL 散度蒸馏，最小化 teacher 分布到 student 分布的 KL，鼓励 student 覆盖 teacher 的高概率区域。
- **Thinking Mode**：模型的一种推理模式，生成更长、更详细的 chain-of-thought；实验中 teacher 使用 thinking mode、student 使用非 thinking mode。
- **Avg@K**：对每个问题采样 K 次取平均正确率，本文中 K=12。

## 可复现要素
- **数据集**：OpenThoughts 训练数据 + AIME 2025/2026、HMMT 2025/2026、BRUMO 2025；代码已开源：https://github.com/ZJU-REAL/TTPO。
- **代码/权重**：代码已开源；模型权重使用 Qwen3-1.7B/4B/8B（公开模型）。
- **关键超参**：LoRA r=64, α=128；学习率 5e-6；每问题采样 K=64 条轨迹，max length=16,000 tokens；梯度更新取前 1,024 tokens；K_train=8（50%正/50%负）；λ=0.1；temperature=1.1，top-p=0.95，top-k=20；训练 100 steps，每 25 steps 保存 checkpoint。
