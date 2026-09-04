---
title: "Squeezing-More-from-Limited-Data-with-Recursive-Transformers"
source: https://arxiv.org/pdf/2608.26973v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:30:05"
field: "低资源语言模型预训练"
keywords: ["递归Transformer", "有限数据预训练", "因式分解嵌入", "BabyLM", "模型缩放", "计算效率"]
innovations: ["提出RecursiveGPT架构，通过递归权重共享将深度作为独立于参数量的计算扩展轴", "在10M和100M词预算下系统验证递归Transformer优于标准Transformer并与BabyLM 2025获奖者竞争", "揭示有限数据下标准Transformer的非单调缩放行为及最优规模对下游任务的依赖性"]
benchmarks: ["BLiMP", "COMPS", "LAMBADA pass@5", "EWoK", "Entity Tracking", "BabyLM Challenge 2025"]
---

# 论文速读：Squeezing-More-from-Limited-Data-with-Recursive-Transformers

## 一句话总结
本文研究了在固定有限数据预算（10M–100M词）但计算资源相对充足的预训练场景下，标准Transformer缩放效果不佳的根本原因，并提出RecursiveGPT架构——通过递归权重共享和解耦嵌入打破参数量与每token计算量的强耦合，从而在低数据预算下更有效地利用计算资源，性能优于标准Transformer并与BabyLM Challenge 2025获奖模型相当。

## 研究问题与动机
- **核心问题**：在数据受限（10M–100M词）、计算相对充裕的预训练设置中，如何有效扩展计算而非单纯增加参数量？
- **标准Transformer缩放瓶颈**：固定词汇表大小下，缩小隐藏维度时嵌入矩阵（$V \times H$）按线性减少，而Transformer块参数按平方减少，导致小模型中嵌入和LM头占据绝大部分参数预算，每token的计算量严重不足。
- **已有方法局限**：现有工作（如Kim et al., 2026）采用训练动态策略（超大权重衰减、集成训练）来缓解过拟合，但未从架构层面解耦容量与计算；而本文主张从架构设计出发，使模型更适配数据受限场景。
- **科学/实践动机**：该设置更接近人类语言学习的发育驱动数据预算（BabyLM Challenge），且随着高质量人类文本供给趋紧，有限数据预训练的实际价值日益凸显。

## 核心贡献（创新点）
1. **系统刻画了有限数据下标准Transformer的非单调缩放行为**：发现最优模型规模强烈依赖于数据预算和下游评测目标（BLiMP饱和早、COMPS持续受益于更大模型），揭示了参数规模本身在数据受限时是一种正则化选择。
2. **提出RecursiveGPT架构，将递归深度作为独立于参数量的计算扩展轴**：通过共享单个Transformer块 across 深度（递归步数R），在不显著增加参数的前提下提升每token计算量，解耦了表示容量与计算深度。
3. **引入因式分解嵌入（Factorized Embeddings）减轻小模型参数分配失衡**：用$V \times E$和$E \times H$两个矩阵替代原始$V \times H$嵌入矩阵（$E \ll H$），使词汇表映射参数从$O(VH)$降至$O(VE+EH)$，释放参数预算给计算层。
4. **系统性消融与对比实验验证了递归架构的有效性**：在10M和100M词预算下，RecursiveGPT均优于标准Transformer；100M词设置下404M参数的RecursiveGPT-Large在BLiMP、EWoK、COMPS和平均分数上超越1.22B标准模型，并与BabyLM 2025获奖者竞争。

## 方法详解
- **递归Transformer核心设计**：将标准R层独立参数Transformer替换为单次共享块$F_\theta$的递归展开：$h^{(0)} = \text{Embed}_{FE}(x_t)$，$h^{(r)} = F_\theta(h^{(r-1)}; \phi_r)$，$y_t = \text{Head}_{FE}(h^{(R)})$。其中$\theta$为跨深度共享的注意力与MLP权重，$\phi_r$为第r步的轻量级深度特定归一化参数和偏置（充当深度条件信号）。
- **因式分解嵌入（Factorized Embedding）**：输入侧：$\text{Embed}_{FE}(x_t) = x_t W_{emb} W_{proj}$，$W_{emb} \in \mathbb{R}^{V \times E}$，$W_{proj} \in \mathbb{R}^{E \times H}$；输出侧：$\text{Head}_{FE}(h_t) = h_t W_{proj} W_{unembed}$。当$E \ll H$时，每个词汇表映射仅需$O(VE+EH)$参数而非$O(VH)$。
- **架构细节**：使用预归一化残差结构、因果自注意力、RoPE位置编码、Query-Key归一化、融合QKV投影、注意力头大小64、MLP扩展因子16（远高于常规的4）、零初始化输出投影、每头sigmoid门控。词汇表大小32768，BPE tokenizer由rustbpe训练。
- **训练配置**：双组优化器——Transformer块参数用Muon优化器（学习率0.02、动量0.95、权重衰减0.1、neuronwise自适应学习率），嵌入/归一化参数用Adam（学习率0.005、$\beta=(0.9,0.95)$、权重衰减0.005）；均使用cautious weight decay；50步warmup、恒定阶段、最后20%线性冷却；梯度裁剪至范数2.0；bfloat16混合精度；有效全局批次大小32768 token。

## 实验与结果
- **数据集与设置**：两个语料库——BabyLM baseline corpus（GPT-BERT使用）和Nemotron-ClimbMix；数据预算10M、25M、50M、100M词（嵌套子集），每个预算训练10个epoch；模型规模从13M到1.2B参数。
- **评测基准**：BLiMP（67个最小对立对语法/形态/语义评测）、COMPS（概念属性知识）、LAMBADA pass@5（ discourse理解），以及BabyLM 2025的BLiMP Supplement、EWoK、Entity Tracking。
- **主要结果（Table 1）**：
  - 10M词：RecursiveGPT+FE (27.6M参数, R=16)平均45.80，优于Standard (41.1M, 44.93)和Standard+FE (28.7M, 45.07)。
  - 100M词：RecursiveGPT-Large (404.1M, R=24)平均63.93，略优于Standard (1.22B, 63.47)；124M参数的小RecursiveGPT+FE在BLiMP上80.06超过1.22B模型的79.71。
- **与BabyLM 2025对比（Table 2）**：10M词下RecursiveGPT在BLiMP (72.21)和EWoK (52.01)上最强；100M词下两个RecursiveGPT变体在所有五项指标上均超越GPT-BERT基线。
- **消融（Table 3）**：移除因式分解嵌入导致性能大幅下降（46.16→44.90/44.09），表明嵌入瓶颈对递归模型尤为关键；共享归一化参数影响较小。
- **计算匹配对比（Appendix F）**：将标准模型训练146 epoch（10M词）或50 epoch（100M词）以匹配RecursiveGPT的FLOPs，结果反而更差（平均41.47 vs 45.80；56.68 vs 63.93），证明递归深度是更有效的计算扩展轴。

## 相关工作脉络
- **BabyLM Challenge**：Hu et al. (2024)设立的发育驱动数据预算预训练竞赛，本文在其框架内评估，目标不是超越专用训练配方而是提供强架构基线。
- **ALBERT (Lan et al., 2020)**：最早提出因式分解嵌入和跨层参数共享用于BERT类编码器压缩；本文将其思想扩展到因果解码器并引入递归深度条件。
- **Universal Transformer (Dehghani et al., 2019)**：将自注意力转换函数递归应用于深度，支持自适应停止；本文采用固定步数简化设计，避免优化不稳定性。
- **Deep Equilibrium Models (Bai et al., 2019)**：求解无限深度网络的不动点；本文与之不同，采用有限显式递归步骤而非隐式求解。
- **近期递归Transformer工作**：Bae et al. (2025)用层 tying+轻量LoRA压缩预训练模型；Aleksandrov et al. (2025)和Geiping et al. (2026)研究循环展开用于推理时计算扩展；本文聚焦预训练阶段的数据效率。
- **小数据递归推理模型**：HRM和TRM (Wang et al., 2025; Jolicoeur-Martineau, 2025)在监督 puzzle 任务上展示递归模块从小数据中泛化的能力，与本文低数据原则一致但任务领域不同。

## 局限性与未来方向
- **超参搜索不充分**：学习率、优化器和架构细节仅在局部调优以获得稳定运行，未对每个语料库/数据预算/模型规模/递归深度的组合单独调优。
- **递归设计空间探索有限**：仅使用单共享块、固定递归步数、简单深度条件参数；未探索部分共享、更丰富的深度条件机制。
- **计算成本问题**：缺乏自适应计算时间(ACT)或深度调度机制，增加递归深度会显著增加训练和推理开销（尽管Appendix C显示约80% token在倒数第二步前已稳定）。
- **未来方向**：结合BabyLM中已验证的专用目标和训练配方；探索自适应停止、深度调度、部分权重共享；将递归架构用于测试时计算扩展。

## 研究启发与可借鉴点
- **解耦参数与计算的架构思路可迁移**：在数据受限、计算充裕的场景下，递归权重共享是一种通用的"计算扩展轴"设计，可应用于视觉、多模态或其他序列建模任务。
- **因式分解嵌入在小模型中的正则化价值**：不仅减少参数量，其瓶颈结构还充当正则器；可探索在其他参数敏感场景（如低资源NMT、小型代码模型）中的应用。
- **递归深度与预测稳定性的关系值得深入研究**：Appendix C的发现（多数token在中间深度稳定）提示可设计动态计算调度，这为高效推理提供了新思路。
- **非单调缩放曲线的分析方法**：本文展示了按下游任务分析缩放行为的必要性（不同任务的最优规模不同），这一评估框架可用于其他数据受限场景的模型选择。
- **训练效率对比实验设计**：Appendix F通过FLOPs匹配而非epochs匹配来公平比较递归vs标准架构，这一控制变量方法值得借鉴。

## 关键术语表
**Recursive Transformer**：通过共享单个Transformer块 across 深度（递归展开）来增加计算深度而不显著增加参数量的架构变体。
**Factorized Embedding**：将$V \times H$的嵌入矩阵分解为$V \times E$和$E \times H$两个小矩阵（$E \ll H$），降低词汇表映射的参数占比。
**RecursiveGPT**：本文提出的递归因果解码器家族，结合递归权重共享和因式分解嵌入，用于有限数据预训练。
**BabyLM Challenge**：由Hu et al. (2024)发起的竞赛，研究发育合理数据预算（10M/100M词）下的语言模型预训练。
**BLiMP**：Warstadt et al. (2020)提出的67个英语最小对立对基准，评测语法、形态和语义现象。
**COMPS**：Misra et al. (2023)提出的概念属性知识最小对立对基准，测试模型的概念继承和属性泛化能力。
**LAMBADA pass@5**：Paperno et al. (2016)的 discourse理解基准，要求参考词出现在模型 top-5 预测中。
**Muon Optimizer**：Jordan et al. (2024)提出的专为Transformer隐藏层设计的优化器，结合neuronwise自适应学习率使用。

## 可复现要素
- **数据集**：BabyLM baseline corpus（GPT-BERT使用）、Nemotron-ClimbMix；数据预算为嵌套子集，公开可得性见BabyLM Challenge和Nemotron项目。
- **代码/权重**：论文声明开放训练pipeline（脚注1），具体仓库链接未在本页给出；权重未明确说明是否公开。
- **关键超参**：词汇表大小32768、BPE tokenizer（rustbpe训练）、12层标准/递归深度R=16或24、隐藏维度H=640/1408/2560、因式分解维度E=192/768/2560、MLP扩展因子16、注意力头大小64、有效批次大小32768、10 epochs、学习率0.02（Muon）/0.005（Adam）、weight decay 0.1/0.005、cautious weight decay、梯度裁剪2.0、bfloat16、50步warmup、最后20%线性冷却。
