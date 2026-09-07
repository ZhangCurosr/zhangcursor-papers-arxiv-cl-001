---
title: "XMerge-Cross-Axis-Selection-and-Reconstructive-Layer-Merging"
source: https://arxiv.org/pdf/2609.02083v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:33:23"
field: "大模型压缩与加速"
keywords: ["depth compression", "layer pruning", "transformer compression", "LLM optimization", "post-training compression", "model efficiency"]
innovations: ["跨轴选择（RM+BI max融合）避免单轴盲区", "重构合并：通过本地激活匹配重拟合完整非线性块而非线性化或新增模块"]
benchmarks: ["CORE", "MMLU", "WikiText-2 PPL"]
---

# 论文速读：XMerge-Cross-Axis-Selection-and-Reconstructive-Layer-Merging

## 一句话总结
XMerge是一种后训练深度压缩方法，通过跨轴选择识别安全的Transformer层，并通过重构合并将相邻两层映射融合到保留层中，在无需任务标签、不引入架构变更的前提下，显著优于现有基线。

## 研究问题与动机
- **Transformer深度导致推理延迟**：解码过程在层间顺序执行，移除完整块可同时降低时延和KV cache流量。
- **现有深度压缩方法质量损失大且不可预测**：不同方法对"移除哪层"和"如何吸收"存在分歧，且性能退化在不同模型间差异大。
- **现有方法的两类失败模式需要分离分析**：选择重要层错误 vs. 未能保留安全层对与其邻居的组合映射。
- **服务架构兼容性要求**：深度压缩需保持原有hidden size、attention、vocabulary不变，以便在标准基础设施上运行。

## 核心贡献（创新点）
- **重构合并算子**：将相邻两块的映射蒸馏回一个现有标准块（非线性），而非引入新模块或线性化——这是唯一通过重新拟合完整标准块来实现重建的算子。
- **无参数跨轴选择器**：结合相对幅度（RM）和Block-Influence（BI）角域两个轴，以max融合避免单轴盲区，无超参权重。
- **系统评估框架**：在7个Llama/Qwen骨干网络、5个基线、3个压缩级别下验证，唯一实现零崩溃的算子。
- **服务接口保留**：压缩后仍为标准$L-k$层Transformer，无新增推理参数，兼容现有推理引擎。

## 方法详解
**整体流程**：$\mathcal{C} = \mathcal{M} \circ \mathcal{S}$，先选择待移除块$f_\ell$，再将相邻块对$(f_\ell, f_{\ell+1})$合并为单个保留块$\tilde{f}_\ell$。

**选择器（Cross-Axis Selection）**：
- **相对幅度轴（RM）**：衡量激活向量的残差位移大小
$$\mathrm{RM}_l = \mathbb{E}_{x,t}\left[\frac{\|h_{l+1}^{(x,t)} - h_l^{(x,t)}\|_2}{\|h_l^{(x,t)}\|_2 + \epsilon}\right]$$
- **角域轴（BI）**：衡量表示方向的旋转程度（来自ShortGPT的Block-Influence）
$$\mathrm{BI}_l = \mathbb{E}_{x,t}\left[1 - \frac{\langle h_l^{(x,t)}, h_{l+1}^{(x,t)}\rangle}{\|h_l^{(x,t)}\|_2 \|h_{l+1}^{(x,t)}\|_2 + \epsilon}\right]$$
- **跨轴融合**：标准化后取max $s_l = \max(z_l^{\mathrm{RM}}, z_l^{\mathrm{BI}})$，避免单轴高分被另一轴均值掩盖。

**重构算子（Reconstructive Merge）**：
- 选定块$f_\ell$后，缓存其输入$h_\ell(x)$和原输出$h_{\ell+2}(x) = f_{\ell+1}(f_\ell(h_\ell(x)))$
- 根据相邻激活修补冗余分数$S_{\mathrm{patch}}$选择保留侧
- 通过300步Adam优化（lr=$10^{-5}$，batch=16，128个WikiText-2序列）最小化：
$$\min_{\tilde{\theta}_l} \mathbb{E}_x[\|\tilde{f}_l(h_l(x);\tilde{\theta}_l) - f_{\ell+1}(f_\ell(h_l(x)))\|_2^2]$$
- 迭代压缩时选择顺序固定，重建目标逐次更新。

## 实验与结果
**数据集与模型**：7个Llama/Qwen骨干网络（0.5B–8B），使用WikiText-2作为校准数据。

**评估指标**：CORE（22任务聚合，主要指标）、MMLU（zero-shot）、WikiText-2 PPL。

**主要结果（k=4，最激进压缩）**：
- XMERGE在6/7骨干网络上CORE排名第一，6/7在MMLU排名第一，5/7同时领先两项。
- 相比最强基线：Llama-3.2-1B CORE +0.070，Llama-3.2-3B +0.033，Qwen3-8B +0.048（bootstrap置信区间排除零）。
- 唯一在14个(model, regime)单元格中零崩溃的算子（其他基线崩溃1–7次）。
- 校准方面：ECE退化最低（+0.010），仅为次优算子的一半。
- 相比LaCo/MKA等，避免了$k=4$时的极端PPL膨胀（如MKA在Llama-3.2-1B上PPL达8343）。

**成本效益**：构建时间1分钟至4.4小时（随$k$线性增长），但通过每token解码节省可在约1.9k–24k请求后收回成本。

## 相关工作脉络
- **ShortGPT**：基于BI的层丢弃方法；本文重用BI作为角域轴，但增加了RM轴进行交叉验证，避免单轴盲区。
- **LaCo/MKA/SWM/CoMe**：解析折叠或窗口平均类合并方法；XMERGE是唯一通过梯度拟合完整非线性块的重建算子，而非解析公式折叠或新增模块。
- **ReplaceMe**：将块线性化为线性映射并折叠到保留权重；XMERGE保持非线性激活结构，通过本地重建拟合。
- **LLM-Streamline**：学习替换模块；XMERGE无需新增模块，保持服务接口不变。
- **Prune&Comp**：研究移除层引发的隐藏状态幅度gap；RM轴的设计受到此类幅度分析的启发。

## 局限性与未来方向
- **单次压缩种子**：仅使用seed 42，未评估重建种子方差。
- **仅测试稠密Decoder-only模型**：未覆盖MoE、编码器-解码器、多模态或状态空间模型。
- **校准评估有限**：仅在Llama-3-8B上一个骨干网络上测试了ECE，无法建立通用安全性保证。
- **WikiText-2校准数据**：部分回归训练域，下游基准（CORE/MMLU）独立但PPL指标有轻微偏差。
- **未消融合并方向启发式**：$S_{\mathrm{patch}}$选择保留侧未做消融实验。
- **未评估安全性行为**：幻觉、拒绝等安全相关行为未测试，需指令微调变体。

## 研究启发与可借鉴点
- **双轴选择鲁棒性**：RM+BI的max融合思路可迁移到其他深度压缩场景，避免单指标盲区；可尝试扩展到更多正交轴（如梯度重要性、信息流分析）。
- **本地激活匹配重建范式**：label-free的本地重建方法无需任务标签，适用性广；可探索不同重建目标（如对抗扰动鲁棒性、分布对齐）。
- **Pareto前沿评估设计**：质量-时延Pareto图直接展示相同推理成本下的精度差异，实验设计清晰值得借鉴。
- **跨 regime 鲁棒性分析**：同时评估zero-shot和ICL regime崩溃情况，提供全面可靠性视角。
- **成本-收益分析框架**：构建时间与解码收益的盈亏平衡计算，为实际部署决策提供量化依据。

## 关键术语表
**Depth Compression**：移除Transformer完整层以缩短网络深度，保持推理接口不变。
**Cross-Axis Selection**：结合幅度（RM）和角域（BI）两个正交轴选择可安全移除的层。
**Reconstructive Merge**：通过本地激活匹配重拟合保留块，使其逼近被移除块对的原始输出映射。
**CORE**：Centered Objective Rank Evaluation，22任务聚合指标，覆盖zero-shot和ICL。
**Block-Influence (BI)**：测量层对表示方向的旋转程度，来自ShortGPT。
**Relative Magnitude (RM)**：测量层引起的隐藏状态残差位移大小。
**Setting A/B**：Setting A为纯算子比较（无恢复），Setting B为共享恢复预算下的对比。
**Collapse Threshold**：CORE < 0.10（接近随机水平）作为模型崩溃的判定标准。

## 可复现要素
- **数据集**：WikiText-2用于校准（训练集），测试集用于PPL评估；CORE使用nanochat bundle评测。
- **代码/权重**：论文未提及公开；提供了官方实现端口（baselines_*.py）。
- **关键超参**：重建步数=300，lr=$10^{-5}$，batch size=16，序列长度=512，校准序列数=128；压缩级别$k \in \{1, 2, 4\}$。
- **硬件**：NVIDIA A100-SXM4-80GB MIG切片。
- **种子**：单次压缩种子42。
