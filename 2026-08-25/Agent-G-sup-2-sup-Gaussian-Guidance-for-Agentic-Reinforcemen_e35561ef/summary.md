---
title: "Agent-G-sup-2-sup-Gaussian-Guidance-for-Agentic-Reinforcemen"
source: https://arxiv.org/pdf/2608.23318v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:36:50"
field: "智能体强化学习调度策略"
keywords: ["hint-based reinforcement learning", "guidance depth scheduling", "Gaussian distribution", "GRPO", "long-horizon agentic tasks", "reward sparsity"]
innovations: ["将guidance depth建模为per-task高斯邻域而非确定性标量，用rollout统计在线估计中心与方差", "global-local分解的自适应调度器：全局基线跟踪策略进度，per-cluster EMA按难度修正中心并拓宽覆盖", "无需probe rollout或辅助网络，同批rollout同时更新策略与调度参数"]
benchmarks: ["ALFWorld", "WebShop"]
---

# 论文速读：Agent-G²: Gaussian Guidance for Agentic Reinforcement Learning

## 一句话总结
论文提出 Agent-G²，一种面向长 horizon 智能体强化学习的高斯引导框架：将 hint-based RL 中的专家轨迹前缀长度建模为每个任务的高斯分布，其中心与方差通过已收集的 GRPO rollout 统计量在线估计，无需额外 probe rollouts 或学习深度预测器。在 ALFWorld 和 WebShop 上均取得最强非 probe 结果，且 rollout 成本仅为 per-sample probing 的不到三分之一。

## 研究问题与动机
- **Reward 稀疏与 advantage collapse**：长 horizon 智能体任务只有终态二值奖励，on-policy 从零状态探索几乎无法到达成功终点，导致 GRPO 风格训练的 advantage 坍缩。
- **既有 hint-based RL 对 guidance depth 的刻画有误**：现有方法把"前缀长度"当成确定性标量——scheduled 方法在一个 batch 内共享单一深度，忽略任务间难度异构；per-sample probing 方法则通过 O(log n) 额外 rollout 做二分/枚举搜索，成本高昂且在 GRPO 预算下仍受噪声影响。
- **有用深度的真实结构是"邻域带"**：诊断实验发现每个任务的 informative depth 分布在某个最优值 $d_i^\star$ 附近呈单峰、近似对称的高斯形态（σ≈0.22，R²=0.92），而非集中在唯一点；正确做法是覆盖这个邻域而非" pinpoint 一个点"。
- **智能体任务比数学推理更异构**：现有方法几乎全在数学推理上验证，而 ALFWorld/WebShop 等智能体场景内 batch 难度差异极大（如两步 Pick vs 二十步 Pick Two），需要 per-task 自适应的调度机制。

## 核心贡献（创新点）
1. **揭示指导深度的邻域结构**：通过系统诊断证明 useful depth 构成 band 而非点，统一解释了 shared-depth scheduler 与 per-sample probing 两类失败模式的结构性根源。
2. **提出 Agent-G² 高斯引导框架**：将每任务 depth 建模为 $\mathcal{N}(\mu_i, \sigma_i^2)$，中心由 global baseline 与 per-cluster 难度偏移叠加而成，方差由 cluster 内 rollout 成功率方差自适应控制，全程复用 GRPO 已采集 rollout 统计量，无额外 probe 开销。
3. **global-local 分解的在线自适应调度**：global $\mu_{\text{global}}$ 跟踪策略整体进展，per-cluster EMA $(A_k, V_k)$ 按困难度修正中心、扩宽覆盖，二者在每批 rollout 后同步更新，实现无监督深度调度器。
4. **在两个长 horizon 智能体基准上验证显著增益**：ALFWorld 上 1.5B/7B 分别达到 95.3%/98.4% success，超过最强非 probe hint-RL、最强 hint-free RL、最强 Aux-RL 基线；且 1.5B Agent-G² 已超越 7B BEACON，表明调度设计可部分替代 backbone 放大。

## 方法详解
- **任务与 rollout 设定**：每任务 $i$ 有自然语言指令 $q_i$ 与一条专家轨迹 $\tau_i^\star=(a_{i,1}^\star,\dots,a_{i,L_i}^\star)$。调度器选 guidance ratio $r_i \in [0,1]$，转为前缀长度 $n_i=\min(\lceil r_i L_i\rceil, L_i-1)$，执行前缀动作到达 post-prefix 状态后，策略 $\pi_\theta$ 在该状态下跑 $R$ 次 rollout，得到二值终态奖励 $y_{i,j}$ 与经验成功率 $\hat p_i$。
- **高斯调度核心公式**：
  - 全局基线：$\mu_{\text{global}}\leftarrow\text{clip}(\mu_{\text{global}}+\text{sign}(p_{\text{target}}-\text{acc}_B)\Delta, 0, 1)$，以 $p_{\text{target}}=0.5$ 为参考，batch 平均成功率低则加深、高则减浅。
  - Cluster EMA：对按专家轨迹长度离线划分的 $K$ 个 cluster $\mathcal{C}_k$，计算 batch 内 $\bar p_{B_k}$ 与 $v_{B_k}$，再经 EMA 更新 $A_k\leftarrow(1-\alpha)A_k+\alpha\bar p_{B_k}$、$V_k\leftarrow(1-\alpha)V_k+\alpha v_{B_k}$。
  - 每任务分布：$\mu_i=\text{clip}(\mu_{\text{global}}+\lambda(p_{\text{target}}-A_k), 0, 1)$，$\sigma_i=\max(\gamma V_k, \sigma_{\min})$。低成功率簇推深中心，高方差簇拓宽覆盖。
- **采样与 rollout**：每任务 $z_i\sim\mathcal{N}(\mu_i, \sigma_i^2)$，截断到 $[0,1]$ 得 $r_i$，生成唯一前缀长度 $n_i$ 供该任务 $R$ 次 rollout 共用；同 cluster 任务共享参数但独立采样，实现 batch 内 per-task 深度多样性而无额外 rollout。
- **联合损失**：$\mathcal{L}=\mathcal{L}_{\text{GRPO}}+\eta\mathcal{L}_{\text{aux}}$，其中 $\mathcal{L}_{\text{aux}}=-\sum_{i}\sum_{t=1}^{n_i}\log\pi_\theta(a_{i,t}^\star|s_{i,t},q_i)$ 是对采样前缀做 teacher-forced SFT，作为正则稳定器；主增益来自 prefix 之后状态的 GRPO 在线优化。
- **闭环**：同批 rollout 的终态奖励同时用于（a）更新策略 $\pi_\theta$、（b）刷新 $\mu_{\text{global}},A_k,V_k$，整个 schedule 不需额外网络或 probe 预算。

## 实验与结果
- **数据集与模型**：ALFWorld（文本嵌入式多步任务，6 种 task type 按 horizon 分 Short/Medium/Long）、WebShop（网页购物导航）；基座 Qwen2.5-1.5B-Instruct 与 Qwen2.5-7B-Instruct；5 次随机种子报告均值。
- **主要结果（ALFWorld success rate %）**：
  - 1.5B：Agent-G² **95.3%**（Short 96.8 / Medium 100 / Long 94.7），超过最强 hint-based RL（Enumeration 86.4，+9.3）、最强 hint-free RL（BEACON 83.1，+12.2）、最强 Aux-RL（RLVMR 87.9，+7.4）。
  - 7B：Agent-G² **98.4%**（全部 Short/Medium 达 100，Long 91.7），超过最强 hint-based RL（Enumeration 96.1，+2.3）与最强 Aux-RL（RLVMR 91.8，+6.6）。
  - 跨 scale：1.5B Agent-G² 已超过 7B BEACON（86.1），体现调度对 backbone 放大的替代效应。
- **WebShop**：两 scale 均获 reward score **92.3**，purchase success 1.5B=78.9%、7B=84.4%，均为最强非 probe 方法；1.5B 超出 Enumeration +2.2 reward-score 点。
- **效率**：Agent-G² 每梯度步 88s，约为 Binary Search（285s，3.24×）与 Enumeration（425s，4.83×）的 1/3；收敛步数约为 scheduled baselines 的一半。
- **消融要点**：去掉采样（固定 $d_i=\mu_i$）下降 5.5 点；替换为同方差均匀分布下降 7.0 点；去 cluster 分组（K=1）下降 6.2 点；去 $\mathcal{L}_{\text{aux}}$ 下降 8.6 点；去 $\mathcal{L}_{\text{GRPO}}$ 退化为 Sampled-Prefix SFT 仅 26.6%（↓68.7），确认主增益来自 prefix 之后的 RL 而非模仿。

## 相关工作脉络
1. **Schedule-based hint-RL**（Linear/Cosine/Step decay, Target-acc）：按训练步或 batch 反馈共享单一深度；Agent-G² 与其本质区别是从"共享标量"转向"每任务高斯邻域"，吸收因 batch 内难度异构带来的 mismatch 结构性误差。
2. **Per-sample probing**（Binary Search, Enumeration）：通过额外 rollout 估算最优深度；Agent-G² 用已有 rollout 的统计量构造高斯邻域，以 1/3  rollout 成本实现更低 mismatch。
3. **End-to-end hint-RL**（StepHint, TraPO）：学习深度预测器或端到端优化；Agent-G² 无需额外网络，调度由 rollout 统计显式驱动，可解释且零额外探针。
4. **Aux-RL**（ETO, RLVMR, BEACON）：引入 preference、价值头、过程奖励模型等辅助监督；Agent-G² 仅使用专家轨迹前缀作为 rollout 起点与轻量 $\mathcal{L}_{\text{aux}}$，不引入任何辅助模型或离线标注。
5. **RL without hints**（GRPO, GiGPO）：直接从初始状态探索；Agent-G² 在相同 GRPO 骨干上叠加高斯引导，解决 advantage collapse，且与 GiGPO/BEACON 正交可组合（论文仅在同等 backbone 上公平比较）。

## 局限性与未来方向
- **依赖完整专家轨迹**：每任务需一条专家 trajectory；若仅能获取次优演示或弱形式 hint，需扩展至弱监督版本。
- **高斯形式是经验拟合**：Section 2 在 ALFWorld/WebShop 上 R²≈0.9 支持高斯假设，但多峰或强偏斜深度分布的任务可能需更丰富的分布族；论文均匀分布消融表明"形状"不如"随机覆盖邻域"重要，但 richer family 仍有潜力。
- **聚类信号固定且粗粒度**：当前按专家轨迹长度离线分 K=3 簇，不随策略能力提升动态演化；未来可做在线 difficulty estimation 使 cluster 随训练自适应。
- **benchmark 局限**：仅在 ALFWorld 与 WebShop 验证，尚未扩展到 GUI 导航、移动 agent 等更复杂环境；通用化仍需更多实证。

## 研究启发与可借鉴点
1. **"邻域覆盖"思维替代"最优定点"**：在 RL 调度类问题中，把目标变量看作具有 informative band 而非单点最优，用分布采样替代确定值选择，可天然吸收噪声与 heterogeneity；该思想可迁移至 curriculum、temperature、rollout budget 等超参调度。
2. **Global-local 分解的在线调度器设计**：全局基线 + 局部 EMA 修正的结构，在无需额外网络的前提下实现 per-task 自适应，适合一切基于 rollout 统计可调的参数（如 entropy schedule、advantage 归一化窗口）。
3. **Rollout 一用多用**：同批 rollout 既更新策略又刷新调度器参数，消除 probe 开销——这是与 GRPO group-normalization 天然兼容的效率技巧，值得推广到其他 on-policy 变体。
4. **Cluster 划分应与 difficulty signal 对齐**：消融显示长度量化 $K{=}3{-}5$ 稳定，随机分配退化为 K=1；后续工作可按 task embedding 或初始成功率做 finer clustering。
5. **$\mathcal{L}_{\text{aux}}$ 定位为稳定器而非主驱动**：68.7 点消融对比表明 prefix SFT 只是辅助，主增益来自 prefix 后的在线 RL；设计类似 hint-based 方法时应避免把模仿权重设过高而压制探索。

## 关键术语表
- **Hint-based RL**：在每次 rollout 前注入专家轨迹前缀，使策略从更接近成功的状态开始探索，以缓解长 horizon 任务的 reward 稀疏与 advantage collapse。
- **Guidance depth**：保留的专家前缀长度（或比例），过短导致 under-guided、过长导致 over-guided，决定 rollout 落在 informative band 内的概率。
- **In-range band**：经验成功率落在 [0.4, 0.6] 区间的深度集合，该区间对应 Bernoulli 方差最大、训练信号最 informative。
- **GRPO (Group Relative Policy Optimization)**：对同一任务 $R$ 次 rollout 的终态奖励做组内归一化计算 advantage，无需 value network，适合 sparse 终态奖励。
- **EMA (Exponential Moving Average)**：对每 cluster 的 batch 成功率均值与方差做滑动平均，用于在线估计 $\mu_i$ 与 $\sigma_i$，$\alpha=0.2$。
- **Aux-RL**：在终态奖励之外引入额外监督信号（如 DPO、过程奖励模型、价值头蒸馏）的 RL 训练范式。
- **Mismatch ratio $\rho$**：分配深度落在任务 in-range band 之外的任务占比，用于量化调度器的错配程度。
- **Global-local decomposition**：$\mu_i$ 由全局基线 $\mu_{\text{global}}$ 与 cluster 偏移 $\lambda(p_{\text{target}}-A_k)$ 相加得到，兼顾整体进度与局部难度。

## 可复现要素
- **数据集**：ALFWorld（公开）、WebShop（公开），训练/验证/测试 split 继承 GiGPO 代码库；**公开**。
- **代码**：论文声明"Code & Project Page & Models"，MIT 许可，**开源**（发布时）。
- **基座模型**：Qwen2.5-1.5B-Instruct、Qwen2.5-7B-Instruct（公开权重）。
- **关键超参**：$\Delta{=}0.1,\ \alpha{=}0.2,\ \lambda{=}1.0,\ \gamma{=}1.0,\ \sigma_{\min}{=}0.1,\ \eta{=}0.5,\ p_{\text{target}}{=}0.5,\ \mu_{\text{global}}^{(0)}{=}0.8,\ K{=}3$；GRPO clip $\epsilon{=}0.2$、KL penalty $\beta_{\text{KL}}{=}0.01$、lr $1{\times}10^{-5}$ cosine+10% warmup、$R{=}16$ rollout/任务、batch $|B|=16$；ALFWorld 200 steps、WebShop 150 steps。**全部论文已给出**。
