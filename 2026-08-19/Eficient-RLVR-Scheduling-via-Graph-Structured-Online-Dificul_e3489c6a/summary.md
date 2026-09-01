---
title: "Eficient-RLVR-Scheduling-via-Graph-Structured-Online-Dificul"
source: https://arxiv.org/pdf/2608.17941v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:51:16"
field: "大语言模型强化学习调度"
keywords: ["RLVR", "difficulty estimation", "graph-based inference", "curriculum learning", "rollout allocation", "variational inference", "Potts prior"]
innovations: ["首次将RLVR动态难度估计建模为图结构潜在变量推断问题，通过Potts先验在相关样本间传播rollout反馈", "设计在线均值场变分推断算法，以Beta-Binomial状态级模型聚合稀疏反馈，无需额外探测即可持续追踪难度变化", "提供即插即用的图结构难度估计器，可无缝集成到GVM/PCL/GRESO等调度器并统一提升下游性能"]
benchmarks: ["MATH500", "AIME 2024", "AIME 2025", "OlympiadBench"]
---

# 论文速读：Eficient-RLVR-Scheduling-via-Graph-Structured-Online-Dificulty-Estimation

## 一句话总结
本文提出了一种**即插即用的基于图的在线难度估计框架**，通过构建语义与推理相似度感知的样本图，利用 rollout 反馈在相关样本间传播信息，无需额外探测即可持续追踪 RLVR 训练中样本难度的动态变化，可无缝集成到 sample-selection 和 rollout-allocation 调度器中提升训练效率。

## 研究问题与动机
- **探索预算分配不均导致训练低效**：GRPO 等 RLVR 算法对所有样本分配相同的 rollout 次数，简单样本获得冗余探索，而困难但可学样本探索不足。
- **现有难度估计方法的局限性**：
  - 探测型方法（probing-based）需额外 rollout 开销大、统计噪声高；
  - 历史型方法（history-based）面临冷启动问题且历史反馈易过时；
  - 两类方法均忽略了样本之间的关联结构。
- **核心科学问题**：如何在 RLVR 训练过程中，以低成本持续、准确地估计所有训练样本的动态难度，以支持自适应调度。

## 核心贡献（创新点）
1. **首次将 RLVR 动态难度估计建模为图结构潜在变量推断问题**——此前工作均采用独立样本建模或纯历史聚合，本文首次引入图结构刻画样本间语义与推理相关性。
2. **提出基于 Potts 先验的图结构潜在难度模型**——通过谱聚类初始化 + Potts 图先验鼓励邻居共享状态，同时用 Beta-Binomial 状态级模型聚合 rollout 反馈，本质区别在于通过图传播缓解稀疏观测而非依赖独立样本估计。
3. **设计在线均值场变分推断算法**——支持每步训练后增量更新所有样本的难度后验，计算高效且理论保证 ELBO 单调收敛，区别于非序贯批量推断方法。
4. **提供可复用的即插即用组件**——可集成到 GVM、PCL、GRESO 等不同调度范式，统一提升下游性能。

## 方法详解

### 整体流程
框架包含四个模块：难度感知样本图构建 → 图结构潜在难度建模 → 在线均值场变分推断 → 面向自适应调度的难度估计输出。

### 3.3 难度感知样本图（Difficulty-Aware Sample Graph）
- 引入难度感知指令 $I_\pi$：要求编码模型同时捕获语义内容与难度特征。
- 用预训练 embedding 模型 $f_{\text{emb}}$ 编码样本：$\phi_i = \widetilde{\phi}_i / \|\widetilde{\phi}_i\|_2$。
- 计算余弦相似度 $c_{ij} = \phi_i^\top \phi_j$，构建互 KNN 稀疏无向图 $W$：仅保留互为 kNN 且相似度为正的对。

### 3.4 图结构潜在难度模型（Graph-Structured Latent Difficulty Model）
- **谱聚类初始化**：对图 $W$ 做谱聚类得到 $K$ 个静态簇标签 $c_i$，转化为软先验：$A_{ik} = 1-\epsilon$（若 $k=c_i$）否则 $\epsilon/(K-1)$。
- **Potts 图先验**：
$$p(\mathbf{z}_t \mid W, A) = \frac{1}{Z(W,A)} \left( \prod_{i=1}^N A_{i,z_{i,t}} \right) \exp\left[ \frac{\beta}{2} \sum_{i,j} W_{ij} \mathbb{I}(z_{i,t} = z_{j,t}) \right]$$
其中 $\beta$ 控制图传播强度。
- **状态级成功概率建模**：同一状态 $k$ 共享 $\theta_{k,t} \sim \text{Beta}(a_{k,t}^{\text{hist}}, b_{k,t}^{\text{hist}})$，rollout 结果 $(s_{i,t} \mid n_{i,t}, z_{i,t}=k, \theta_{k,t}) \sim \text{Binomial}(n_{i,t}, \theta_{k,t})$。
- **联合分布**：
$$p_t(\mathbf{s}_t, \mathbf{z}_t, \boldsymbol{\theta}_t \mid \mathbf{n}_t, W, A) = p(\mathbf{z}_t \mid W, A) \prod_{k=1}^K \text{Beta}(\theta_{k,t} \mid a_{k,t}^{\text{hist}}, b_{k,t}^{\text{hist}}) \prod_{i \in \mathcal{O}_t} \text{Binomial}(s_{i,t} \mid n_{i,t}, \theta_{z_{i,t},t})$$

### 3.5 在线均值场变分推断（Online Mean-Field Variational Inference）
- 变分近似：$q_t(\mathbf{z}_t, \boldsymbol{\theta}_t) \approx \prod_i q_i(z_{i,t}) \prod_k q(\theta_{k,t})$，其中 $q(\theta_{k,t}) = \text{Beta}(a_{k,t}, b_{k,t})$。
- **ELBO 优化**（坐标上升交替更新）：
  - **Assignment 更新**（Proposition 1）：
$$q_{ik,t}^\star = \text{Softmax}_k\left[ \log A_{ik} + \beta \sum_j W_{ij} q_{jk,t} + \mathbb{I}(i \in \mathcal{O}_t) \ell_{ik,t}^{\text{VB}} \right]$$
其中 $\ell_{ik,t}^{\text{VB}} = s_{i,t}[\psi(a_{k,t}) - \psi(a_{k,t}+b_{k,t})] + (n_{i,t}-s_{i,t})[\psi(b_{k,t}) - \psi(a_{k,t}+b_{k,t})]$。
  - **Beta 更新**（Proposition 2）：
$$a_{k,t} = a_{k,t}^{\text{hist}} + \sum_{i \in \mathcal{O}_t} q_{ik,t} s_{i,t}, \quad b_{k,t} = b_{k,t}^{\text{hist}} + \sum_{i \in \mathcal{O}_t} q_{ik,t}(n_{i,t}-s_{i,t})$$
- **在线策略**：每步用上一状态 $S_{t-1}$ 初始化，交替执行 E-step（assignment 更新）和 M-step（Beta 参数更新），直至 ELBO 增量低于阈值。

### 3.6 面向调度的难度估计
- 状态级后验均值：$\mu_{k,t-1} = a_{k,t-1}/(a_{k,t-1}+b_{k,t-1})$。
- 模型预测：$\widehat{p}_{i,t}^{\text{model}} = \sum_k q_{ik,t-1} \mu_{k,t-1}$。
- 最终估计（对有历史数据的样本加权融合）：$\widehat{p}_{i,t} = \gamma \widehat{p}_{i,t}^{\text{model}} + (1-\gamma)\bar{r}_{i,<t}$，难度 $d_{i,t} = 1 - \widehat{p}_{i,t}$。

## 实验与结果
- **基础模型**：Qwen 2.5 Math-1.5B、Llama 3 1B-Instruct。
- **训练数据集**：Numina-Math（约 150K 题目）。
- **评估基准**：MATH500、AIME 2024、AIME 2025、OlympiadBench。
- **基线方法**：GVM（rollout 分配）、PCL（课程学习）、GRESO（选择性 rollout）。
- **关键超参**：$K=320$，$k_{\text{nn}}=80$，$\gamma=0.5$，$\beta=2$。
- **主要结果**（Table 1，Average@8 准确率）：
  - **Qwen-2.5 Math-1.5B**：GVM + Ours 在 MATH500 上从 71.9→74.7，AIME24 从 11.3→16.7；PCL + Ours MATH500 从 59.7→61.6，AIME24 从 7.92→11.3；GRESO + Ours MATH500 从 66.8→68.2。
  - **Llama-3.2 1B-Instruct**：GVM + Ours MATH500 从 23.9→25.7，AIME24 从 0.42→3.30。
  - **综合提升率**：22 个非平局实验中 21 个为正（95.5%），符号检验 $p = 1.10 \times 10^{-5}$，统计显著。
- **难度估计精度**（Table 2）：早期阶段 Qwen 模型上 Pearson r=0.482（低开销方法最优），全程 r 升至 0.836；估算额外开销仅 ~0.12 A100·h（对比 SBR 的 ~29.6h、SBS 的 ~45.3h）。
- **消融实验**（Table 3）：去掉谱聚类初始化 MAE 从 0.183→0.218，去掉平滑先验 MAE 从 0.183→0.197。

## 相关工作脉络
1. **Curriculum Learning / Prompt Selection**：PCL（Gao et al., 2026）训练专用难度预测器，本文不依赖独立学习预测器，而是通过图传播复用已有 rollout 反馈。
2. **Rollout Allocation via Gradient Variance**：GVM（Yao et al., 2026）用梯度方差指导 rollout 分配，需实时难度信号；本文提供该信号的低成本来源。
3. **History-based Dynamic Estimation**：MoPPS（Qu et al., 2026）做流式贝叶斯推断但忽略样本间关联；本文通过图结构显式建模样本相关性，缓解稀疏观测。
4. **Probing-based Methods**：VIP（Nguyen et al., 2026a）用轻量预测器拟合近期 rollout；本文以概率图模型替代数据驱动预测器，理论更严谨。
5. **Reward Dynamics Modeling**：GRESO（Zheng et al., 2026）利用奖励时间规律性；本文方法与时序规律正交，可互补集成。
6. **Graph-based Semi-supervised Learning**：谱聚类与 Potts 先验源自图半监督学习传统（Zhu et al., 2003; Belkin et al., 2006），本文首次将其引入 RLVR 难度估计领域。

## 局限性与未来方向
- **静态图假设**：当前图基于离线 embedding 构建，未考虑训练过程中语义/推理结构的动态演化。
- **窗口化 vs 全历史的权衡**：Appendix G 表明窗口化变体虽降低 MAE 但削弱了 batch 级排序相关性（r 从 0.836 降至 0.712），在调度场景中稳定排序比逐样本校准更重要，故仍用累积版本。
- **仅验证数学推理领域**：代码生成实验（Appendix D）仅证明可迁移性，未展示显著优势，推广至其他领域尚需验证。
- **嵌入模型选择敏感**：移除难度感知指令后 MAE 大幅上升至 0.309，说明对 embedding 质量和指令设计有较强依赖。
- **未来方向**：动态图构建（随策略演化更新相似度）、自适应 $\gamma$ 权重机制、跨领域系统性评估。

## 研究启发与可借鉴点
1. **图传播 + 贝叶斯状态聚合的组合范式**：Potts 先验 + Beta-Binomial 状态级模型的联合框架，可有效利用稀疏观测，这一组合可直接迁移至其他需要样本级信心估计的场景（如主动学习、在线 Bandit）。
2. **谱聚类初始化 + 平滑先验的变分推断策略**：用谱聚类提供高质量初值规避局部最优，再用软先验允许后续反馈修正——此设计在任意 latent variable model 的 online inference 中均有借鉴价值。
3. **即插即用估计器架构**：将难度估计与调度策略解耦，作为通用组件替换原有 estimator，这一工程范式值得推广到其他 RLVR 优化方法的研究中。
4. **同步 Jacobi 加速实现**：Appendix A.4 给出与严格序列 Gauss-Seidel 相比误差 $<10^{-2}$ 但并行效率更高的同步实现，为大规模图变分推断提供了实用的工程参考。
5. **MRL 压缩提升近邻图质量**：发现 Matryoshka Representation Learning 压缩 embedding 维度可缓解高维距离失真，这一技巧可直接用于构建高质量样本图的其他研究。

## 关键术语表
**RLVR**（Reinforcement Learning with Verifiable Rewards）：利用可自动验证的奖励信号（如数学答案正确性、代码测试通过）对大语言模型进行强化学习的训练范式。
**GRPO**（Group Relative Policy Optimization）：一种组内相对优势的 RL 算法，对每个样本生成多条 rollout 并比较组内奖励以更新策略。
**Potts 先验**：源自统计物理的图结构正则化先验，鼓励相邻节点分配相同的离散标签，此处用于鼓励语义相近样本共享难度状态。
**Beta-Binomial 共轭模型**：以 Beta 分布为成功概率的先验、二项分布为似然的贝叶斯模型，支持滚动更新且解析可解，用于状态级成功率的在线估计。
**均值场变分推断**（Mean-Field Variational Inference）：将联合后验因子化为各变量独立分布乘积的近似推断方法，通过坐标上升最大化 ELBO。
**Average@8**：对每个 prompt 采样 8 次 rollout，取最高分的正确率，用于降低单次评估的随机性。
**PCL**（Prompt Curriculum Learning）：训练专用难度预测器以进行课程学习的调度方法（Gao et al., 2026）。
**GVM**（Gradient Variance Minimization）：基于梯度方差最小化原则进行 rollout 分配的调度方法（Yao et al., 2026）。

## 可复现要素
- **数据集**：Numina-Math（约 150K 训练题，从 860K 原始数据中提取）；评估集 MATH500、AIME 2024、AIME 2025、OlympiadBench 均为公开数据集。
- **代码**：论文未明确声明代码开源链接，但说明"integrate our framework as a plug-and-play component into the original code"，需在基线代码基础上修改。
- **权重**：使用 Qwen3-Embedding-0.6B 作为默认 embedding 模型（公开可用）。
- **关键超参**：$K=320$（潜在状态数）、$k_{\text{nn}}=80$（近邻数）、$\beta=2$（Potts 强度）、$\gamma=0.5$（历史融合权重）、embedding 维度 1024（MRL 截断）。
- **算力**：4×A100 80G GPU，额外估计开销约 0.12 A100·h（3 轮 epoch，150K 样本）。
