---
title: "Agent-G-sup-2-sup-Gaussian-Guidance-for-Agentic-Reinforcemen"
source: https://arxiv.org/pdf/2608.23318v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:36:50"
---

# 论文速读：Agent-G²: Gaussian Guidance for Agentic Reinforcement Learning

## 一句话总结
论文针对长程智能体强化学习中的奖励稀疏问题，提出 Agent-G²，将专家轨迹的引导深度从单点标量扩展为任务级高斯分布，其均值与方差通过在线聚合已有 rollout 统计量自适应估计；该方法无需额外探针采样或辅助网络，在 ALFWorld 与 WebShop 上显著超越现有 hint-based、hint-free 与 Aux-RL 基线，且训练成本仅为逐样本探测方法的三分之一以下。

## 研究问题与动机
1. **核心问题**：Hint-based RL 通过在 rollout 前注入专家轨迹前缀缓解终态奖励稀疏，但其效果高度依赖“引导深度（guidance depth）”的选取。
2. **共享深度调度的结构性缺陷**：Schedule-based 方法为整批任务共享单一确定性深度，忽略 batch 内任务难度的异质性；诊断表明超 60% 的分配落在有效训练区间之外，多数为过度引导。
3. **逐样本探测的成本-精度困境**：Probe-based 方法通过二分搜索或枚举为每样本单独估计深度，虽降低共享假设的偏差，但需消耗 $O(\log n)$ 或数倍于 GRPO 的额外 rollout 预算，且在有限预算下估计噪声仍导致高错配率。
4. **认知前提偏差**：现有工作均隐含“每个任务存在唯一最优深度”的假设；本文诊断证明有效深度实际是以最优值为中心的带状邻域，训练信号沿相对深度呈近似单峰对称的高斯形态，引导深度应被“覆盖”而非“ pinpoint”。

## 核心贡献（创新点）
1. **揭示引导深度的邻域结构**：定量证明 informative depths 构成高斯型带状分布（拟合 σ=0.22，R²=0.92），从原理上统一解释了共享深度与逐样本探测两类方法的失效机制。
2. **提出零探针成本的自适应高斯调度**：将每任务的引导深度建模为 Gaussian，中心与方差由同一批 rollout 的成功率统计在线估计，消除对额外采样预算或深度学习器的依赖。
3. **全局-局部双尺度参数分解**：设计全局基线跟踪策略整体进度，结合按轨迹长度预分组的 EMA 统计量（$A_k$, $V_k$）修正中心位置与分布宽度，以极低开销匹配同 batch 内的难度异质性。
4. **强鲁棒性与高效率的实验验证**：在 ALFWorld 与 WebShop 上于 1.5B/7B 双尺度均取得最优或非探测类最优，且单步耗时仅为逐样本探测方法的 1/3 以下，验证了“调度设计可部分替代骨干模型scaling”。

## 方法详解
- **问题设定**：每任务 $i$ 有自然语言指令 $q_i$ 与专家轨迹 $\tau_i^\star$。调度器输出引导比率 $r_i \in [0,1]$，转换为前缀长度 $n_i = \min(\lceil r_i L_i \rceil, L_i-1)$；执行前缀后从剩余状态启动 $R$ 次 rollout，获得二元终态奖励 $y_{i,j}$。策略使用 GRPO 基于组内相对优势进行更新。
- **自适应高斯调度（Adaptive Gaussian Schedule）**：
  - **全局基线 $\mu_{\text{global}}$**：根据 batch 平均成功率 $\text{acc}_B$ 向目标 $p_{\text{target}}=0.5$ 的方向更新：$\mu_{\text{global}} \leftarrow \text{clip}(\mu_{\text{global}} + \text{sign}(0.5 - \text{acc}_B)\Delta, 0, 1)$。成功率低则加深前缀，反之缩短。
  - **聚类统计量**：训练集按专家轨迹长度离线划分为 $K$ 个难度簇 $\mathcal{C}_k$。对每簇维护成功率的 EMA 均值 $A_k$ 与方差 $V_k$，反映当前簇的难度水平与离散程度。
  - **任务级参数**：$\mu_i = \text{clip}(\mu_{\text{global}} + \lambda(0.5 - A_k), 0, 1)$，$\sigma_i = \max(\gamma V_k, \sigma_{\min})$。失败率低的簇加深中心，簇内差异大的展宽分布以覆盖更广的有效邻域。
- **逐任务采样与 Rollout**：$z_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$，截断至 $[0,1]$ 得 $r_i$ 并映射为 $n_i$。同簇任务共享 $(\mu_i, \sigma_i)$ 但独立采样，在同一 batch 内天然产生深度多样性，无需额外计算。
- **联合训练目标**：$\mathcal{L}(\mathcal{B}) = \mathcal{L}_{\text{GRPO}}(\mathcal{B}) + \eta \mathcal{L}_{\text{aux}}(\mathcal{B})$，其中 $\mathcal{L}_{\text{aux}} = -\sum_{i,t \le n_i} \log \pi_\theta(a_{i,t}^\star | s_{i,t}, q_i)$ 为对采样前缀的 teacher-forcing 损失，用于稳定模仿过程；终端奖励同时用于策略更新与刷新 $(\mu_{\text{global}}, A_k, V_k)$，形成无需
