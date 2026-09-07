---
title: "Train-What-You-Deploy-Closing-the-MLP-Reachability-Gap-in-Lo"
source: https://arxiv.org/pdf/2609.02006v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:31:38"
field: "大模型压缩与蒸馏"
keywords: ["Knowledge Distillation", "Model Compression", "Low-Rank Clone", "MLP Reachability", "Train-Serve Consistency", "LLM Compression"]
innovations: ["形式化并审计 LRC 蒸馏中的训练-部署可达集间隙", "提出 Train What You Deploy 原则及 Dense/CORE-LRC 两种零推理成本实现", "通过等参数量、基线、跨配方三重控制实验归因增益到可达集扩张本身"]
benchmarks: ["Avg9 (0-shot)", "MMLU", "LM Evaluation Harness"]
---

# 论文速读：Train-What-You-Deploy-Closing-the-MLP-Reachability-Gap-in-Lo

## 一句话总结
论文指出 Low-Rank Clone（LRC）蒸馏方法存在"训练-部署可达集间隙"——部署时训练全宽 MLP 矩阵，但训练过程仅能触及教师诱导的子空间，导致 62.5–81.4% 的已部署线性自由度无法训练。提出的 "Train What You Deploy" 原则从同一 LRC 热启动出发，以零推理成本训练完整部署矩阵，在三个教师上分别获得 +2.36/+2.71/+10.45 Avg9 的提升，最宽教师（Qwen）实现 2× token 效率。

## 研究问题与动机
- **核心问题**：LRC 压缩了 student 的隐藏维度 $d_{\text{model}}^{(S)}$，却继承了教师 MLP 中间宽度 $d_{\text{ff}}$，导致 student 部署的是一个 $d_{\text{ff}} \times r$ 的全宽矩阵，但训练仅写入教师列空间的一个 $d_{\text{model}}^{(T)}$ 维切片，形成"部署宽、训练窄"的可达集间隙。
- **现有方法不足**：LRC 的可达集 $\mathcal{R}_{\text{LRC}} = \{TZ^\top\}$ 是部署假设空间 $\mathbb{R}^{d_{\text{ff}} \times r}$ 的真子集，间隙由参数化结构确定性决定（与初始化、目标、优化器无关），却未带来任何部署节省（不同于真实剪枝删除维度，也不同于 LoRA 为保护预训练知识而接受低利用率）。
- **关键指标**：训练利用率 $u = d_{\text{model}}^{(T)} / d_{\text{ff}} = 1/\rho$，其中 $\rho = d_{\text{ff}}/d_{\text{model}}$ 在三个教师上分别为 2.67、3.50、5.375，对应被"搁浅"的自由度比例为 62.5%–81.4%。
- **动机验证路径**：通过干预打开被排除的正交补空间 $\text{col}(W)^\perp$，测量恢复的容量，而非仅依赖"初始化时很少使用"的观察性论证。

## 核心贡献（创新点）
- **形式化可达集间隙并使其可审计**：引入训练利用率 $u$ 作为量化指标，证明 LRC 的 MLP 矩阵存在确定性可达集间隙 $\mathcal{G} = \mathcal{H}_{\text{deploy}} \setminus \mathcal{R}_{\text{LRC}}$，区别于 RED 诊断的表示秩坍缩（那是优化器层面的问题，而非参数化结构约束）。
- **提出"Train What You Deploy"原则及两种可合并实现**：从相同 LRC 热启动出发，通过 Dense-LRC（标准稠密权重全参数训练）和 CORE-LRC（教师谱分解的补空间重参数化，零初始化后合并回同一部署权重）打开补空间，推理时 FLOPs 和参数量不变。
- **严格的因果归因控制实验**：通过等参数量控制（同训练量但约束在教师切片内，无增益）、基线控制（规范稠密、随机补全均在噪声内收敛）、跨配方控制（GPD 简化配方下 gap 依然存在）三重验证，证明增益来源于可达集扩张本身而非坐标系统或额外参数。

## 方法详解
- **Dense-LRC**：将每个 MLP 投影（gate/up/down）参数化为标准 $d_{\text{ff}} \times r$ 稠密权重，从合并的 plain-LRC 热启动开始训练所有条目，可达列空间为整个 $\mathbb{R}^{d_{\text{ff}}}$，推理时退化为单一矩阵，无额外参数或 FLOPs。
- **CORE-LRC（教师谱重参数化）**：利用教师权重的 SVD 分解，令 $T$ 为 $\text{col}(W)$ 的标准正交基，$U_\perp$ 为其正交补基，student 权重写为 $W_{\text{student}} = TZ_{\text{col}}^\top + U_\perp Z_\perp^\top$，其中 $Z_\perp$ 零初始化，训练从 plain-LRC 精确启动，合并后与 Dense-LRC 部署完全相同；两种实现在窄教师（Llama）上性能一致，在最宽教师（Qwen）上 CORE-LRC 显著更优。
- **训练目标**：温度缩放 KL 散度（logits 对齐，主导项）+ 逐层 hidden-state 与 attention MSE 对齐（$\lambda_{\text{aux}}=0.2$）+ next-token cross-entropy 正则化；所有控制实验保持相同配方。
- **合并性命题**：由于 $[T\ U_\perp]$ 构成 $\mathbb{R}^{d_{\text{ff}}}$ 的完备标准正交基，CORE-LRC 等价于在固定教师对齐坐标系下训练标准权重，函数上与 Dense-LRC 等价，仅优化器 AdamW 的坐标不同（AdamW 非旋转不变）。

## 实验与结果
- **数据集与配方**：~10B FineWeb-Edu + 0.35B OpenHermes 蒸馏 tokens，可选 0.62B tokens 短 SFT；评估使用 LM Evaluation Harness 的 9-task macro-average（Avg9，含 MMLU，排除 MathQA），全部 0-shot。
- **Llama3.2-3B→1.5B（$\rho=2.67$）**：Full-MLP LRC 达 65.40（PT 10B），较 plain-LRC 基线 +2.36；SFT 后 66.21 与教师 66.18 在评估噪声内无差异（+0.03）；MMLU +4.02。长预算控制（plain-LRC 继续至 50B tokens 或独立跑 50B）均不改善，证实为容量增益而非收敛加速。
- **Llama3.1-8B→2.7B（$\rho=3.50$）**：Dense-LRC 达 68.26（PT 10B），+2.71；SFT 后 68.93，超过 Meta 官方同系压缩 Llama3.2-3B（~9T tokens 训练）的 66.18，压缩阶段 tokens 减少约 900×（注：为 token 计数比较，非公平算力比较）。
- **Qwen2.5-3B→1.7B（$\rho=5.375$，最难压缩）**：CORE-LRC 达 63.44（PT 10B），+10.45，达到原始 LRC 配方 ~20B tokens 的精度（63.43），实现 2× token 效率；同一亲缘 Dense-LRC 控制臂 +6.39 为严格受控证据。
- **最强结果**：Qwen 教师上 CORE-LRC +10.45 Avg9，是最宽教师上差距最大的增益；1.5B student 以 ~10B 蒸馏 tokens 匹配 ~9T tokens 训练的教师的 9-task macro-average。

## 相关工作脉络
- **Low-Rank Clone（Hao et al. 2025）**：本文的骨干方法，联合软剪枝（低秩投影）与教师激活克隆；本文指出的可达集间隙是其结构性缺陷，修复后仍基于 LRC 热启动。
- **RED（He et al. 2026，同期工作）**：诊断同一投影蒸馏家族的另一种失败——隐藏表示的有效秩坍缩，通过激活感知通道选择初始化修复，但仍局限于 $\{TZ^\top\}$ 子空间内训练；两者正交可分（激活感知初始化在 Qwen 上仅 +0.68 Avg9，而打开可达集 +6.39 至 +10.45）。
- **结构化剪枝蒸馏（Minitron、Sheared LLaMA）**：从宽度方向删除中间维度（而非继承）；本文视 LRC 继承的未使用补空间为可填充储备，而非被删除的冗余。
- **Function-preserving growth（Samragh et al. 2024）**：从小模型扩展并依赖全训练激活未使用方向；本文设定为其压缩对偶。
- **PEFT/子空间适配器（PiSSA、DoRA、LoRA-Null 等）**：CORE-LRC 的谱分解机制在形式上类似约束 SVD 子空间更新，但目标不同——PEFT 方法保护预训练知识，本文在蒸馏预训练阶段添加可用容量。
- **FFN 谱缩放分析（Jha & Reagen 2025/2026）**：观察性论证 FFN 宽度利用率低且受优化器影响；本文为建设性对应，提供干预实验验证。

## 局限性与未来方向
- **无"零遗忘"保证**：单层正交性无法控制端到端 Jacobian，MMLU 相对教师仍有下降（残余 deficit）。
- **归因强度受限**：等参数量控制仅用单一固定矩阵 A 和一个 seed；2×2 配方实验仅在 2B tokens 上运行；Qwen  headline +10.45 为跨亲缘比较，严格受控依赖于同亲缘 Dense-LRC +6.39。
- **单设计轴**：仅针对 LRC 的隐藏维度压缩展开，未探索最优 $d_{\text{ff}}$；隐藏宽度扫描缺乏每宽度无补基线。
- **无 MoE 覆盖**：Per-expert 扩展比接近或低于 1，结构性补空间很小。
- **单 backbone、单一评估范围**：所有 student 使用 LRC 骨干，外部基线为已发表数值（仅参考）；评估仅限 0-shot 多选/QA，不含生成或长链推理套件。
- **未来方向**：将审计框架推广至其他剪枝-蒸馏骨干、探索更宽学生的 MMLU 恢复、多 seed 实验、跨配方公平比较。

## 研究启发与可借鉴点
- **"Train/Serve Consistency" 审计框架**：对任意压缩权重，计算训练利用率 $u = \dim(\mathcal{R}) / \dim(\mathcal{H}_{\text{deploy}})$，若 $u < 1$ 需有明确理由（如 LoRA 保护预训练、剪枝删除维度），否则应关闭间隙；此审计可迁移至其他蒸馏/压缩方法。
- **热启动+全矩阵训练的零成本增益模式**：从已有的低秩蒸馏热启动出发，仅将 MLP 投影的训练对象扩展至完整部署矩阵，推理 FLOPs 和参数量不变，是一种"免费回收搁浅容量"的工程策略。
- **严格的因果归因实验设计**：等参数量控制（同训练量但约束在子空间内）、基线控制（规范稠密 vs. 随机补全 vs. 教师谱基）、跨配方 2×2 因子实验，三层控制形成强归因链条，可作为压缩-蒸馏研究的实验设计范本。
- **教师谱分解初始化对病态目标的增益**：在最宽教师（$\rho=5.375$）上，CORE-LRC 的教师谱基实现显著优于规范稠密基（+10.45 vs. +6.39），提示在病态条件压缩中，保持教师对齐坐标系统有助于优化器收敛。
- **团队结合机会**：可将此框架与团队正在研究的 LLM 压缩/蒸馏方向结合，审计其他压缩方法（如 Minitron、Sheared LLaMA）的可达集间隙，或在 MoE 场景外探索补空间利用。

## 关键术语表
- **Reachability Gap（可达集间隙）**：部署权重矩阵的训练可达子空间 $\mathcal{R}$ 与完整部署假设空间 $\mathcal{H}_{\text{deploy}}$ 之间的确定性结构差，由参数化公式决定而非优化过程。
- **Training Utilization $u$**：训练可达维度与部署维度的比值，$u = d_{\text{model}}^{(T)} / d_{\text{ff}} = 1/\rho$；$u<1$ 且有收益才合理，否则为浪费。
- **Dense-LRC**：将 LRC 热启动后的 MLP 投影替换为标准稠密权重并全参数训练的实现方式。
- **CORE-LRC**：基于教师 SVD 分解的补空间重参数化实现，将权重拆分为教师列空间路径与正交补空间路径，补路径零初始化后合并回同一部署矩阵。
- **Avg9**：九项标准评测（ARC-E、ARC-C、LogiQA、CSQA、PIQA、WinoGrande、BoolQ、SciQ、MMLU）的 0-shot 准确率均值（排除 MathQA）。
- **Col(W)⊥（列空间正交补）**：教师权重列空间的正交补空间，维度为 $d_{\text{ff}} - d_{\text{model}}^{(T)}$，是 LRC 训练中被排除但部署时支付成本的结构补空间。
- **Mergeability（可合并性）**：CORE-LRC 的双路径设计保证在 $Z_\perp=0$ 时退化为 plain-LRC，合并后推理形状、参数量、FLOPs 均不变。

## 可复现要素
- **数据集**：FineWeb-Edu（≈10B tokens，教育分数≥4）+ OpenHermes（≈0.35B tokens）用于 PT；8-dataset 通用混合（≈0.62B tokens）用于 SFT；数据公开（FineWeb-Edu 已开源）。
- **代码/权重**：论文声明代码、配置和运行日志可向通讯作者申请获取（qmao@titanholdings.ai）；未声明在公共仓库开源。
- **关键超参**：AdamW，余弦学习率调度，10% warmup，grad-clip 1.0；PT 学习率 $1\times10^{-4}$，SFT $1\times10^{-5}$；有效 batch 32 条长度为 2048 的序列；bf16 精度；温度 $T=40$，$\lambda_{\text{aux}}=0.2$，$w_{\text{KL}}=w_{\text{NTP}}=1.0$；单 seed 运行。
- **初始化**：数据感知贪心 SVD（Qwen 额外保留 top-48 outlier 通道为恒等行）； ridge regression 逐层 zoom matrix 初始化。
