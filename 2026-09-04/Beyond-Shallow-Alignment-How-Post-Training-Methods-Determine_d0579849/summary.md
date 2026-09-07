---
title: "Beyond-Shallow-Alignment-How-Post-Training-Methods-Determine"
source: https://arxiv.org/pdf/2609.03887v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 05:29:29"
field: "大语言模型安全对齐与机制解释"
keywords: ["mechanistic interpretability", "safety alignment", "post-training methods", "refusal circuits", "steering robustness", "alignment trilemma"]
innovations: ["首次跨训练目标（SFT/Ra-SFT/ORPO）的机制级对照比较", "揭示推理增强导致拒绝电路从注意力头主导转向 MLP 主导", "实证提出对齐三元悖论：分布式编码、安全/能力可分离、细粒度可纠正无法兼得"]
benchmarks: ["WildJailbreak", "StrongREJECT", "XSTest", "MMLU"]
---

# 论文速读：Beyond-Shallow-Alignment-How-Post-Training-Methods-Determine Refusal Circuits And Steering Robustness

## 一句话总结
本文首次在机制层面系统比较了 SFT、推理增强 SFT（Ra-SFT）和偏好优化（ORPO）三种后训练方法如何重塑大语言模型的“拒绝”内部计算电路，并揭示了三种方法在分布式编码、安全/能力可分离性、细粒度可纠正性三者之间无法同时满足的“对齐三元悖论”。

## 研究问题与动机
- **核心问题**：训练目标（而非仅训练数据）是否会影响模型内部实现“拒绝有害请求”的计算电路结构？
- **现有方法不足**：
  - 当前安全对齐研究多停留在行为层面评估（如 ASR、ORR），缺乏对内部机制（circuit）的系统性比较。
  - 已有机制工作仅刻画固定模型的拒绝方向或神经元，未分析不同训练目标如何重塑电路拓扑。
  - 后训练方法被政策制定者与开发者视为可靠的二值化安全屏障，但近期事件（如 Claude Code/GPT‑4.1 遭角色扮演/jailbreak 攻击）表明行为级对齐评估不足。
  - 推理增强与偏好优化两种主流范式在机制层面的对比与可解释性尚属空白。

## 核心贡献（创新点）
1. **跨范式受控比较**：在数据、超参、基础模型固定的条件下，仅改变训练目标（SFT vs Ra‑SFT vs ORPO），在 Llama‑3.1‑8B、Gemma‑2‑9B、Qwen3‑8B 三款架构迥异的模型上进行对照实验，首次揭示训练目标本身决定拒绝几何与电路拓扑。
2. **首次机制分析 Ra‑SFT 与 ORPO**：通过激活修补（Activation Patching）、归因修补（Attribution Patching）与推理时干预（ITI/ActAdd），首次刻画推理增强与偏好优化在内部如何组织拒绝执行。
3. **提出“对齐三元悖论”**：实证证明，在离线目标中，没有任何一种方法能同时实现分布式拒绝编码、安全/能力可分离性与细粒度可纠正性；任一维度的提升均以牺牲另一维度为代价。

## 方法详解
- **实验设计**：三模型（Llama‑3.1‑8B、Gemma‑2‑9B、Qwen3‑8B）× 三目标（SFT、Ra‑SFT、ORPO），使用匹配的 Alpaca（16k 良性）与 BeaverTails（4k 安全）数据，ORPO 额外使用 BeaverTails 原版配对数据共 11,179 条提示。
- **拒绝几何分析**：采用差异均值法（Difference‑in‑Means, DIM）从 256 对拒绝‑顺从提示中抽取各层拒绝方向向量 $\hat{\mathbf{r}}^{(l)}$，并按层内激活范数均值归一化，计算跨目标的余弦相似度以观察方向是否收敛/发散。
- **电路分析**：先进行逐层激活修补量化因果贡献，再对 top‑K（K=5）层做归因修补近似组件（MLP、注意力头）因果效应，最后对 top 组件执行精确激活修补验证。
- **干预测试**：在推理时施加拒绝方向向量进行激活加法（ActAdd，层级别）或针对特定注意力头的推理时干预（ITI），强度参数为 $\alpha$，观测对 ASR、ORR 与 MMLU 准确率的影响。

## 实验与结果
- **数据集与基线**：
  - 安全性：WildJailbreak（2k 对抗）、StrongREJECT（420 提示，7 类攻击）、XSTest（250 对抗良性）。
  - 能力：MMLU（200 提示，5 个领域）。
  - 基线：三款模型的 base 版本与各后训练 checkpoint。
- **主要结果**：
  - **行为安全**：ORPO 在 WildJailbreak 与 StrongREJECT 上 ASR 最低（Gemma‑ORPO 在 StrongREJECT 达 0.0%），但 Gemma‑ORPO 出现过拒绝率（ORR）31.6%；Ra‑SFT 在三项模型上 ASR 普遍最低，且 ORR 控制较好（Gemma 0.8%、Llama 42.8%、Qwen 24.0%）。
  - **几何重塑**：三种目标的拒绝方向均远离 base 模型，且彼此间余弦相似度低于 0.8；SFT 与 ORPO 方向较接近，Ra‑SFT 方向显著独立，体现推理监督带来的质变路径。
  - **电路拓扑**：
    - Llama：SFT → Ra‑SFT → ORPO 呈现从注意力头主导（Head‑25 抑制）向 MLP 主导的转变（Ra‑SFT 第 31 层 MLP +0.90）。
    - Gemma：SFT 与 ORPO 在全部组件上呈现均匀冗余编码（+0.24~+0.47），Ra‑SFT 则呈现不均匀 MLP 主导（第 39 层 MLP −0.44，第 37 层 MLP +0.25），跨架构一致。
    - Qwen：三类目标均 MLP 主导，但 Ra‑SFT 第 31 层 MLP 强抑制（−0.36）。
  - **干预鲁棒性**：
    - ActAdd 在 Llama 上无论何目标，α=10 时 MMLU 准确率降至 0%（安全‑能力严重重叠）。
    - Gemma 与 Qwen 上识别层（recognition）干预比执行层（execution）更有效：Gemma Ra‑SFT α=20 时识别层 ASR 下降 18.8pp，执行层仅 5.8pp；Qwen SFT 下降 28.2pp vs 10.8pp。
    - ITI 在单头干预上均失败：Llama 出现单 token 循环/重复问题；Gemma SFT 因“海德拉效应”无效；Gemma ORPO 因过度约束电路使头表征移出拒绝空间。
- **最强结果**：Ra‑SFT 在三项模型上均实现最低的 StrongREJECT ASR（Gemma 7.9%、Llama 6.7%、Qwen 4.3%），且 ORR 低于 ORPO 变体；Gemma ORPO 在 StrongREJECT 上达到 0.0% ASR 但伴随 31.6% ORR。

## 相关工作脉络
1. **拒绝表示几何**（Arditi et al., 2024; Du et al., 2025; Yeo et al., 2025; Wu et al., 2026）：将拒绝刻画为单一方向或双轴（识别‑执行），本文扩展至跨训练目标系统比较。
2. **浅层对齐与电路集中**（Qi et al., 2025; Huang et al., 2026; Chen et al., 2025; Kazemi et al., 2026）：指出安全机制高度集中于少量头或神经元，本文验证 Ra‑SFT 能打破集中但引入推理开销。
3. **后训练方法与对齐**（Vennemeyer et al., 2026; Janiak et al., 2026; Thakkar et al., 2025; Haldar et al., 2025）：关注行为‑效用权衡，本文首次从电路层面解释 ORPO 过度约束、Ra‑SFT 推理链分布等现象。
4. **SFT 与推理变体**（Jain et al., 2024; Hu et al., 2026）：行为或神经元级分析，本文补充组件级因果归因。
5. **干预可靠性**（Tan et al., 2024; Braun et al., 2025）：指出数据集与原始偏好影响 steering，本文揭示训练目标同样决定可 steering 性。

## 局限性与未来方向
- **规模局限**：仅考察 8B‑9B 密度模型，电路分析在更大规模上的泛化性待验证（现有机制研究多集中于 2.8B 以下且结论不一）。
- **未探索推理‑偏好交互**：Ra‑SFT 与 ORPO 的组合尚未测试，且推理链生成依赖 GPT‑4o one‑shot，存在风格/长度偏差。
- **数据偏差**：BeaverTails 标注者的安全/有害界定可能引入主观偏差。
- **评估稳定性**：边界案例分类存在运行间方差（如 7% 的 benign 提示在不同运行中翻转）。
- **未来方向**：追踪拒绝概念在训练过程中的时序演化、结合更多对齐标准（human agency 等）、探索推理‑偏好联合训练。

## 研究启发与可借鉴点
- **跨范式对照设计**：固定数据、超参与基础模型，仅切换训练目标，是解耦目标效应的干净范式，可迁移至任何机制对比研究。
- **识别‑执行分层干预策略**：通过分别定位“识别层”与“执行层”，可显著提升 steering 效率并减少能力损伤，为后续定向对齐编辑提供操作框架。
- **组件级因果稳定性评估**：结合 bootstrap 重采样与 Spearman ρ 排名稳定性分析，能可靠区分真正关键的组件与噪声，建议作为电路分析标配。
- **对齐三元悖论的工程警示**：在部署关键安全场景时，需明确当前后训练方法无法同时兼顾分布式、可分离、可纠正，应针对不同风险偏好选择目标（如侧重安全选 ORPO、侧重可控选 Ra‑SFT）。

## 关键术语表
- **Difference‑in‑Means (DIM)**：通过计算有害与良性提示激活均值之差来估计拒绝方向向量的几何分析方法。
- **Activation Patching**：在推理时替换某一层的源激活为目标激活，以测量该层对特定行为的因果贡献。
- **Attribution Patching**：基于泰勒展开的一阶近似，用单次反向传播高效估计各组件对目标函数的直接因果效应。
- **Recognition‑Execution Framework**：将拒绝分为早期“识别有害性”与后期“执行拒绝”两个阶段，对应不同网络层的功能分工。
- **Alignment Trilemma**：分布式拒绝编码、安全/能力可分离性、细粒度可纠正性三者无法在单一离线训练目标下同时实现。
- **Over‑Refusal Rate (ORR)**：模型将良性提示错误拒绝的比例，反映过度安全对齐带来的效用损失。
- **Inference‑Time Intervention (ITI)**：在推理阶段对特定注意力头的输入切片施加钩子（hook）干预，以测试单组件 steering 效果。
- **Hydra Effect**：因多组件因果效应均匀分布，干预单一组件会触发其他组件补偿性响应，导致 steering 失效或退化。

## 可复现要素
- **数据集**：Alpaca（16k 良性）、BeaverTails（4k 安全 + 原版配对）、Arditi et al. (2024) 256 对提示、WildJailbreak、StrongREJECT、XSTest、MMLU；论文未明确声明公开，但代码与模型已开源。
- **代码/权重**：https://github.com/hoangcuongnguyen2001/Beyond‑Shallow‑Alignment
- **关键超参**：学习率 1e‑5、batch size 1、梯度累积 128、有效 batch size 128、epochs 3、warmup ratio 0.1、weight decay 0.01、max sequence length 2048；ORPO β=0.1、max prompt length 1536；K=5（top‑K 层/头）；α 取值 0‑20。
- **硬件**：单张 A100 80GB GPU。
- **解码**：greedy decoding。
