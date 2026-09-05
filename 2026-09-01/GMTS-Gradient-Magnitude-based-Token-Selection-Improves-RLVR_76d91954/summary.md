---
title: "GMTS-Gradient-Magnitude-based-Token-Selection-Improves-RLVR"
source: https://arxiv.org/pdf/2608.30632v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:25:13"
field: "大语言模型强化学习微调"
keywords: ["RLVR", "Token Selection", "Gradient Magnitude", "GRPO", "DAPO", "Entropy-based Selection", "Large Language Model Reasoning"]
innovations: ["提出GMTS方法，用熵×梯度系数代理梯度幅度进行Token筛选，跨答案一致性优于纯熵方法", "形式化证明Token熵与log-probability梯度幅度的渐近线性关系（Proposition 3.1）", "在数学/代码/常识三个推理域及1.5B-8B多模型规模上系统性验证GMTS相对ETS的稳定优势"]
benchmarks: ["AIME2024", "AIME2025", "AMC23", "MATH-500", "Minerva", "OlympiadBench", "HumanEval", "MBPP", "LiveCodeBench", "BigCode-Bench", "CS-QA", "CS-QA2"]
---

# 论文速读：GMTS-Gradient-Magnitude-based-Token-Selection-Improves-RLVR

## 一句话总结
论文提出了 **GMTS（梯度幅度 Token 选择）** 方法，通过利用熵–梯度的近似关系在 RLVR 训练中更精细地筛选 Token，解决了仅凭 Token 熵筛选时跨答案无法一致反映重要性的问题，在数学、代码和常识推理三个领域及多模型规模上持续超越 ETS 基线。

---

## 研究问题与动机

- **现有 RLVR 方法对所有 Token 采用统一目标函数**，未充分区分各 Token 对策略更新的贡献差异；GRPO / DAPO 等将答案级 Advantage 均匀分配给同一答案内所有 Token。
- **已有 ETS（Entropy-based Token Selection）仅按熵排序选择 Top-20% Token**，在单答案内部高熵 Token 与梯度幅度正相关，但当不同答案的奖励信号 $A_{i,t}$ 存在差异时，相同熵值对应的真实梯度幅度可能截然不同。
- **PPO 风格裁剪机制与 KL 正则项进一步放大熵的偏差**：裁剪会抑制偏离比值过大的 Token 梯度，KL 校正项也影响有效梯度，这些效应无法被熵单独捕捉。
- 论文核心问题：**能否在不需要额外计算开销的前提下，得到比熵更稳定、跨答案一致的 Token 重要性度量？**

---

## 核心贡献（创新点）

1. **揭示并形式化"Token 熵–梯度幅度"的渐近线性关系（Proposition 3.1）**：证明当 $1 - \pi_\theta(o_t|q,o_{<t}) = \varepsilon \to 0$ 时，$\frac{\log G_t}{\log E_t} \to 1$，为"高熵 Token 更有训练价值"提供理论解释。

2. **提出 GMTS（Gradient Magnitude-based Token Selection）**：定义 $\delta_{i,t} = |E_{i,t} \cdot \omega_{i,t}(\theta)|$ 作为梯度幅度的可计算代理，其中 $\omega_{i,t}$ 为 RLVR 中 Token 级损失系数（含 Advantage、裁剪指示、KL 校正），从而在保留熵信号的同时修正跨答案梯度差异。

3. **建立 ETS vs GMTS 的三点本质区别**：（1）相同熵在不同答案中梯度幅度可不同；（2）裁剪会使高熵 Token 的梯度被无效化，GMTS 通过 $\mathbb{I}_{\epsilon_1,\epsilon_2}$ 自动降权；（3）KL 正则项校正项同样被 GMTS 捕捉，ETS 无法反映。

4. **在三个推理领域 × 多种模型规模上的系统性实验验证**：在 MATH、CODE、COMMONSENSE 三个域，1.5B / 7B / 8B 三档模型上，GMTS 相对于 ETS 平均提升 1–3 个百分点，最突出的 AIME2024 +5.21pp、AIME2025 +3.75pp。

---

## 方法详解

### 1. Token 梯度的分解

Token 级目标 $\ell_{i,t}(\theta)$ 的梯度可分解为：

$$
\nabla_\theta \ell_{i,t}(\theta) = \omega_{i,t}(\theta) \cdot \nabla_\theta \log \pi_\theta(o_{i,t})
$$

其中有效系数：

$$
\omega_{i,t}(\theta) = r_{i,t}(\theta) A_{i,t} \cdot \mathbb{I}_{\epsilon_1,\epsilon_2}(r_{i,t}(\theta), A_{i,t}) + \beta\left(\frac{\pi_{\text{ref}}(o_{i,t})}{\pi_\theta(o_{i,t})} - 1\right)
$$

- $r_{i,t} = \pi_\theta / \pi_{\text{old}}$ 为概率比；
- $\mathbb{I}_{\epsilon_1,\epsilon_2}$ 为 PPO 裁剪指示函数（裁剪区间外为 0）；
- KL 项在 DAPO 中被省略，此时 $\omega_{i,t} \approx A_{i,t}$。

### 2. 熵–梯度幅度的渐近线性（Proposition 3.1）

令 $G_t = \|\nabla_{z_t} \log \pi_\theta(o_t|q,o_{<t})\|_2$ 为对 logit 的梯度范数，$E_t$ 为预测分布熵，$\varepsilon = 1 - \pi_\theta(o_t|q,o_{<t})$，则有：

$$
\lim_{\varepsilon \to 0} \frac{\log G_t}{\log E_t} = 1
$$

即 $\log G_t \approx \log E_t$，两者在对数空间呈线性且斜率趋近 1。再经链式法则 $\nabla_\theta \log \pi_\theta = \frac{\partial z_t}{\partial \theta} \cdot \nabla_{z_t} \log \pi_\theta$，高熵 Token 诱导更大的参数梯度。

### 3. GMTS Score 与训练目标

GMTS 用熵近似 $\nabla_\theta \log \pi_\theta$ 的幅度方向，构造 Token 重要性分数：

$$
\delta_{i,t} = |E_{i,t} \cdot \omega_{i,t}(\theta)|
$$

GMTS 训练目标（相比 GRPO/DAPO 仅在 selected tokens 上平均）：

$$
\max_\theta \mathbb{E}_q\left[\frac{1}{S_\rho}\sum_{i=1}^G\sum_{t=1}^{|o_i|} \mathbb{I}[\delta_{i,t} \geq \tau_\rho] \cdot \ell_{i,t}(\theta)\right]
$$

- $\rho$ 为 Top 比例（实验默认 $\rho = 0.2$）；
- $\tau_\rho$ 为对应分位数阈值；
- $S_\rho$ 为选中 Token 数；
- 所有所需量（$E_{i,t}, r_{i,t}, A_{i,t}, \pi_{\text{ref}}, \pi_{\text{old}}$）在标准 RLVR 训练中均已具备，**额外开销极小**。

### 4. 与 ETS 的对比

| 维度 | ETS | GMTS |
|------|-----|------|
| 排序依据 | 仅熵 $E_{i,t}$ | 熵 × 梯度系数 $\delta_{i,t}=|E_{i,t}\cdot\omega_{i,t}|$ |
| 跨答案一致性 | 差（忽略 $A_{i,t}$ 差异） | 好（纳入 Advantage 与裁剪） |
| 裁剪处理 | 无 | 自动降权被裁剪的高熵 Token |
| KL 正则项 | 忽略 | 纳入 $\omega_{i,t}$ |

---

## 实验与结果

### 数据集与评估基准

- **MATH 域**：MATH-12K（1.5B/7B）、DAPO-MATH-17K（8B）；评测：AIME2024、AIME2025、AMC23、MATH-500、Minerva、OlympiadBench
- **CODE 域**：KodCode；评测：LiveCodeBench、HumanEval、MBPP、BigCode-Bench
- **CS 域**：CS-QA；评测：CS-QA、CS-QA2

### 关键结果（Top-20%，average@16 / greedy）

**MATH 域（Qwen2.5-math-1.5B，DAPO）**：GMTS 相对 ETS 平均 **+1.55pp**（41.56 vs 40.01），AIME2024 +0.41pp、AMC23 +2.19pp、Minerva +1.50pp。

**MATH 域（Qwen2.5-math-7B，DAPO）**：GMTS 相对 ETS 平均 **+1.33pp**（50.14 vs 48.81），AIME2024 达 **25.00%**（+5.00pp vs 20.00）。

**MATH 域（Qwen3-8B，DAPO）**：GMTS 相对 ETS 平均 **+1.85pp**（56.08 vs 54.23），**AIME2024 +5.21pp**（39.79 vs 34.58）、**AIME2025 +3.75pp**（30.00 vs 26.25）。

**CODE 域（Qwen2.5-coder-1.5B，DAPO）**：GMTS 相对 ETS 平均 **+1.87pp**（46.58 vs 44.71），HumanEval +1.21pp，BigCode-Bench +3.84pp。

**CS 域（Qwen2.5-base-1.5B，DAPO）**：GMTS 相对 ETS 平均 **+0.87pp**（66.33 vs 65.46）。

### 消融结论

- **Bottom 选择实验**：GMTS Bottom 显著劣于 ETS Bottom（-1.08pp/-1.28pp），说明低梯度幅度 Token 确实贡献较小，支持 GMTS 的上界选择逻辑。
- **选择比例敏感性**：在 10 种 $\rho \in \{0.1, 0.2, 0.5, 0.7, 0.9\} \times \{DAPO, GRPO\}$ 配置中，GMTS 胜 9 出 10，说明优势不依赖于特定阈值。

---

## 相关工作脉络

1. **ETS（Wang et al., 2025c）**：本文最直接的对标基线，基于 Top-20% 高熵 Token 训练；GMTS 在同框架下通过引入 $\omega_{i,t}$ 修正，精度更高。
2. **Token-level advantage estimation（Chen et al., 2025; Tan & Pan, 2025; Cheng et al., 2026）**：通过修改 Token 级 Advantage 实现不均匀学习；GMTS 不同在于**直接过滤而非重新加权**，更简洁。
3. **Objective function 设计（Wang et al., 2025b; Cui et al., 2025; Li et al., 2025）**：entropy-aware clipping、entropy collapse 预防等；GMTS 不涉及改目标函数，只改 Token 选择，**兼容性更强**。
4. **Low importance token filtering（Lv et al., 2026; Meng et al., 2026）**：自适应/稀疏 Token 选择；GMTS 定位与之相近但度量标准从"相对惊讶度"或"分布偏移"变为"梯度幅度代理"。
5. **Fipo（Ma et al., 2026）**：用 future KL 重新评估 Token 贡献；GMTS 与其思路相近但**无需额外 KL 估计**，计算代价更低。

---

## 局限性与未来方向

- **理论保证尚不完整**：Proposition 3.1 仅给出渐近线性，GMTS 相对于 ETS 优势的严格理论分析仍是开放问题。
- **更大模型未评估**：受计算资源限制，未在 14B / 32B 等规模上验证，扩展性待进一步检验。
- **选择比例 $\rho$ 的经验调优**：虽验证了多比例下 GMTS 均优于 ETS，但最优 $\rho$ 随任务/模型的潜在变化未系统研究。
- **仅测试了 GRPO / DAPO**：对 GSPO 等其他 RLVR 变体的适配性未验证（论文仅提及"可迁移"，未做实验）。

---

## 研究启发与可借鉴点

1. **梯度系数分解视角**：将 Token 梯度分解为"方向（$\nabla_\theta \log \pi_\theta$）× 标量系数（$\omega_{i,t}$）"是理解 Token 重要性的通用框架，可迁移至其他 RL 微调场景（如 RM 训练、AI feedback 微调）。
2. **熵–梯度渐近关系**：Proposition 3.1 提供了"熵可作为梯度代理"的严格条件，未来可推广到多轮对话、长 CoT 场景下对难 Token 的识别。
3. **无需额外计算开销的改进范式**：GMTS 完全复用训练中已有量（$E_{i,t}, r_{i,t}, A_{i,t}$），启示后续工作可在不改架构的前提下通过**后验重加权**提升效率。
4. **Bottom 选择实验的设计**：通过"反向验证"（Bottom 选择）佐证方法正确性，是可借鉴的消融策略——不仅证明 Top 选得好，更要证明 Bottom 选得差。
5. **可与本团队"难样本挖掘 / 重要性采样"方向结合**：GMTS 思想可拓展到 SFT 阶段或 DPO/ORPO 的 Token 级重要性建模，形成统一的"梯度幅度优先"训练策略。

---

## 关键术语表

- **RLVR（Reinforcement Learning with Verifiable Rewards）**：利用可验证reward（如数学答案正确性）对 LLM 进行强化学习训练，代表方法为 GRPO / DAPO。
- **GMTS（Gradient Magnitude-based Token Selection）**：本文提出的 Token 筛选方法，用 $|E_{i,t} \cdot \omega_{i,t}|$ 代理梯度幅度进行 Top-ρ 选择。
- **ETS（Entropy-based Token Selection）**：Wang et al. (2025c) 提出的仅按 Token 熵排序选取 Top-20% Token 的基线方法。
- **$\omega_{i,t}(\theta)$**：Token 级策略梯度中的有效标量系数，包含 Advantage、PPO 裁剪指示和 KL 正则项，决定该 Token 对参数更新的实际贡献强度。
- **PPO 裁剪指示函数 $\mathbb{I}_{\epsilon_1,\epsilon_2}$**：当概率比 $r_{i,t}$ 超出 $[1-\epsilon_1, 1+\epsilon_2]$ 时将对应梯度置零，防止策略更新过大。
- **average@16**：对每道题生成 16 个答案候选，取其中之一正确的比例，常用于 MATH / CS 等需要多路径推理的评测。
- **DAPO（Dynamic Sampling Policy Optimization）**：在 GRPO 基础上引入动态采样（丢弃全对/全错 group）、移除 KL 惩罚、增大上裁剪阈值的工程改进版 RLVR。
- **Token-level KL divergence $\mathbb{D}_{i,t}^{KL}$**：当前策略与参考策略在 Token $o_{i,t}$ 上的 KL 散度，用于约束策略漂移幅度。

---

## 可复现要素

- **数据集**：MATH-12K、DAPO-MATH-17K、KodCode、CS-QA、CS-QA2（均为公开数据集，论文已引用来源）。
- **代码开源**：standalone implementation 在论文第 3 脚注给出（github 链接，需访问原文）；基于 verl 框架实现。
- **关键超参**：
  - 选择比例 $\rho = 0.2$（Top-20%），消融实验覆盖 $0.1, 0.2, 0.5, 0.7, 0.9$；
  - 裁剪参数 $\epsilon_1 = 0.2,\ \epsilon_2 = 0.28$；
  - 学习率：1.5B/7B 模型 $\eta = 3 \times 10^{-5}$，8B 及以上 $\eta = 1 \times 10^{-6}$；
  - rollout 数 $G = 16$，温度 $T = 1.0$；
  - 最大响应长度：MATH/CS 域 2048–4096，CODE 域 32768；
  - 训练 seed = 0，8 × NVIDIA H100 80GB。
- **模型**：Qwen2.5-math-1.5B、Qwen2.5-math-7B、Qwen3-8B、Qwen2.5-coder-1.5B/7B、Qwen2.5-base-1.5B/7B。
