---
title: "Measuring-Optimal-Transport-in-Transformer-Depth"
source: https://arxiv.org/pdf/2609.00748v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:21:38"
field: "Transformer 可解释性与动力学分析"
keywords: ["optimal transport", "transformer interpretability", "layer dynamics", "Wasserstein distance", "Brenier map", "model efficiency"]
innovations: ["首次在实证层面检验训练 Transformer 层间移动是否遵循最优传输", "提出包含采样修正、校准和 sliced 下界的 OT 测量协议", "发现 OT 对齐随训练在出口层涌现而入口层退化"]
benchmarks: ["WikiText-103", "Pile"]
---

# 论文速读：Measuring-Optimal-Transport-in-Transformer-Depth

## 一句话总结
本文首次在实证层面检验了训练好的 Transformer 在层间转移 token 状态时是否遵循最优传输（Optimal Transport, OT）原理：最后层的 token 移动确实沿最优传输映射到达最优代价，而首层不遵循；该最优性随训练逐步涌现。

## 研究问题与动机
- **核心问题**：Transformer 每一层将 token 状态云从层 ℓ 移动到层 ℓ+1，这一移动是否在代价上达到最优，且每个 token 是否沿 Brenier 最优映射到达其最优目的地？
- **现有方法的不足**：
  - 将 OT 嵌入 attention 内部的工作（如 entropic OT attention）只关注层内机制，从未在层间对网络自身的耦合进行代价效率或映射对齐测试。
  - OT-Transformer、rectified flows 等工作显式施加 OT 正则化，本文则反向检验：无此类正则化时，训练后的网络是否自发接近最优耦合。
  - 层间动力学线性近似（如 linear-drift SDE、tuned lens）只能解释部分移动，未检验非线性最优传输成分。

## 核心贡献（创新点）
1. **首个实证检验**：首次在实际训练 Transformer 上测量层间耦合是否构成最优传输，同时评估代价效率和映射对齐（Spearman rank agreement）。
2. **一套可复用的 OT 测量协议**：包括采样误差修正（self-term correction / Sinkhorn divergence）、对已知最优耦合的校准、整体平移代价与 token 特异移动的分离、无需校准的下界（sliced Wasserstein）。
3. **涌现性发现**：证明最后层的 OT 对齐是训练学到的（初始化为 0.64→训练后 0.86），而首层反而远离最优传输，说明 OT 性质并非全局均匀形成。

## 方法详解
- **实验对象**：Pythia-160m（12 层）和 Pythia-410m（24 层），在 WikiText-103 和 Pile 子集上提取 token 状态云。
- **两种状态表示**：
  - **Mean flow**：每种 token 类型在所有出现位置上的平均残差流状态（25,268 种类型）。
  - **Raw states**：每个实际位置的 token 出现点的完整状态。
- **降维投影**：使用 16 维 PCA 相机，原因：经验 Wasserstein 距离的误差以 $n^{-1/d}$ 衰减，低维更易精确估计。
- **代价效率度量**：
  - 原始效率：$e_\ell = W_2^2(\mu_\ell, \mu_{\ell+1}) / \mathbb{E}_i\|x_i^{(\ell+1)} - x_i^{(\ell)}\|^2$。
  - 去除公共平移后：将位移 $d_i$ 分解为均值 $\Delta m_\ell$ 和 token 特异部分，定义 $P_\ell = \mathbb{E}_i\|d_i - \Delta m_\ell\|^2$（token 特异能量）、$A_\ell = \|\Delta m_\ell\|^2$（平移能量），token 特异效率 $\bar{e}_\ell$ 和整体效率 $\epsilon_\ell = (A_\ell + \bar{e}_\ell P_\ell)/(A_\ell + P_\ell)$。
- **精确赋权协议**：从层 ℓ 和 ℓ+1 各采样 4,000 个 token，构造代价矩阵 $C_{ij} = \|x_i - y_j\|^2$，用 POT 库的 network simplex 精确求解最优赋权，得到 $W_2^2$ 估计。
- **采样误差修正**：通过计算同层独立样本之间的赋权代价（Sinkhorn divergence 的精确版），从分子中减去自项，消除独立采样的有限样本偏差。
- **校准**：将同一管道用于两个已知最优的耦合（平移 $y=x+\Delta m_\ell$ 和半径匹配缩放），验证仪器精度在 10–15%。
- **无校准下界**：使用 sliced Wasserstein（2,000 个随机方向投影后排序计算），给出 $\bar{e}_\ell^{\text{sliced}} \leq \bar{e}_\ell$，无需采样、无需校准。
- **映射对齐度量**：Spearman rank agreement $\rho$（ pooled 分量）和 median cosine，与 shuffle 配对 null（$\rho \leq 0.08$）对比。
- **线性控制**：拟合高斯 Brenier 映射 $T_G$（仅用均值和协方差），评估线性变换能解释多少 token 特异移动，以及剩余部分是否仍对齐最优传输。

## 实验与结果
- **数据集**：WikiText-103（242,015 个 raw 位置，25,268 种 token 类型）和 Pile（46,542 条 mean flow 轨迹）。
- **主要结果**（16-d 相机，n=4,000，3 seeds）：
  - **Pythia-160m**：在 4 个可解析过渡中，3 个（entry 0→1、3→4、10→11）的 token 特异效率 $\bar{e}_\ell \approx 1.0$（仪器精度内最优）；唯一缺口在出口 11→12，$\bar{e}_\ell=0.86$（比最优代价高 14–27%）。映射对齐：入口 $\rho=0.41$，中间 0.38–0.62，出口 $\rho=0.89$，median cosine=0.95。
  - **Pythia-410m**：所有可解析过渡（0→1、5→6、6→7、22→23、23→24）均接近最优；出口 $\rho=0.88$，median cosine=0.95。
  - **块级测量**：160m 的 3→9（六层块）$\bar{e}=0.97$，8→11（三层块）$\bar{e}=0.96$；410m 的 4→20（十六层块）$\bar{e}=0.96$。
  - **训练动态**：160m 出口 $\rho$ 从初始化 0.64→step4000 的 0.67→全训练 0.86；入口反而从 0.52 降到 0.37。
  - **线性控制**：160m 出口线性映射仅解释 39% 的 token 特异移动，剩余 61% 仍以 $\rho=0.83$ 对齐最优传输；410m 出口线性解释 90%，剩余部分不可分辨。

## 相关工作脉络
- **Transformers as measure-to-measure maps**（Vuckovic et al., Furuya et al.）：理论推导层间映射性质，本文做实证测量；对象相同（token 状态云），但目标不同（检验是否最优）。
- **Attention and OT**（Sander et al., Wang & Wang）：将 OT 置于 attention 内部，用正则化；本文在层间放无正则化的精确 OT，并检验网络自身耦合的效率和对齐。
- **OT-regularized flows**（OT-Transformer, Kan et al.; Rectified flows, Liu et al.）：显式施加 OT 代价；本文反向检验无正则化时网络是否自发接近最优。
- **Depth as dynamics**（Sarfati et al., Belrose et al.）：用线性 SDE/线性 lens 描述深度；本文的线性基线 $T_G$ 是同族最优传输成员，检验网络是否超越线性映射。
- **Gromov-Wasserstein alignment**（Alvarez-Melis & Jaakkola）和 **Model fusion via OT**（Singh & Jaggi）：若层间映射确为 OT 型，则这些方法可带保证地应用——本文提供实证基础。
- **Neural OT**（Makkuva et al., Korotin et al.）：学习 OT 映射的神经网络；本文反过来检验已有网络是否自发呈现 OT 性质。

## 局限性与未来方向
- **分辨率限制**：每样本 4,000 token，仅能解析 token 特异移动大于采样间隔的过渡；160m 仅 4/12 层、410m 仅 5/24 层可解析，中层只能以块评估。
- **模型规模与数据集有限**：仅测试两个小模型（160m、410m）和两个小型语料库，结论的泛化性待验证。
- **Mean flow 的平均效应**：对每种 token 类型跨上下文取平均会抬高对齐度量，raw states 的对齐略低（出口 0.74 vs 0.89）。
- **中层线性假设未验证**：410m 中层可能接近线性映射，但需要更高分辨率才能判定。
- **尺度效应待查**：160m 出口的代价缺口（14–27%）在 410m 消失，需更大模型验证是否随尺度闭合。
- **泛化性不明**：OT 对齐是语言模型特有属性，还是训练 Transformer 的普遍现象，尚需更多实验。

## 研究启发与可借鉴点
- **OT 效率协议可直接迁移**：采样误差修正（自项减法）、已知最优耦合校准、sliced Wasserstein 下界这三步法可推广到其他架构的层间动力学分析。
- **平移与 token 特异的能量分离**（$A/P$ 比率）是一个简洁的诊断指标，可广泛用于分析网络层间移动的结构性，判断哪些层主要做全局平移、哪些做精细重排。
- **训练动态的 OT 对齐曲线**可作为新的训练监控信号：出口对齐随训练单调上升、入口下降，提示训练在"塑造"而非"均匀扰动"表征空间。
- **线性控制（Gaussian Brenier map $T_G$）作为基线**：分离线性解释部分与非线性剩余，验证非线性成分是否仍遵循 OT，这一思路可用于分析任何深度网络的层间映射结构。
- **可与本团队方向结合**：若研究模型压缩/剪枝/蒸馏，OT 效率可作为层间"浪费运动"的无标签度量；若研究多模型融合，本文的实证基础保证了 OT 对齐方法可安全应用于已训练模型。

## 关键术语表
- **Optimal Transport（最优传输）**：在两个概率分布间寻找最小代价的移动方案，本文使用 squared Euclidean 代价。
- **Wasserstein-2 distance（$W_2$）**：基于平方的最优传输距离，衡量两层 token 状态云之间的最小移动代价。
- **Brenier map**：最优传输的最优耦合由凸函数梯度给出，$y = \nabla\varphi(x)$，本文检验网络的实际移动是否匹配此映射。
- **Sliced Wasserstein**：沿随机方向投影后在一维上精确求解 OT，平均后给出真实 $W_2^2$ 的下界，无需采样校准。
- **Transport efficiency $\bar{e}_\ell$**：token 特异部分的实际代价与最优代价之比，1 表示完全最优。
- **Rank agreement $\rho$（Spearman）**：网络移动与 OT 移动在排序维度上的一致性，0.89 表示强正相关。
- **Self-term correction（Sinkhorn divergence）**：减去同层独立样本间的赋权代价，消除有限采样的系统偏差。
- **Gaussian Brenier map $T_G$**：仅用均值和协方差拟合的高斯最优传输映射，作为线性基线控制。

## 可复现要素
- **数据集**：WikiText-103 和 Pile（公开）。
- **模型**：Pythia-160m、Pythia-410m（开源，来自 Biderman et al., 2023）。
- **代码**：论文未提及代码是否开源。
- **关键超参**：PCA 相机 16 维；每样本 4,000 token；独立样本对；3 个 random seed；sliced Wasserstein 用 2,000 个随机方向；POT 库的 network simplex（Bonneel et al., 2011）求解精确赋权。
