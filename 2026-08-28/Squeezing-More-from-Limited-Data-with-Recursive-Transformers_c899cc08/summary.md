---
title: "Squeezing-More-from-Limited-Data-with-Recursive-Transformers"
source: https://arxiv.org/pdf/2608.26973v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:30:29"
---

# 论文速读：Squeezing-More-from-Limited-Data-with-Recursive-Transformers

## 一句话总结
本文针对数据受限（10M–100M 词）但计算相对充裕的预训练场景，提出 RecursiveGPT 架构：通过共享单一 Transformer 块实现递归深度扩展，并结合因式分解嵌入（Factorized Embeddings）解耦词表参数与隐藏容量。实验表明该设计能有效绕过低数据下的过拟合瓶颈，在 10M 与 100M 词预算下均优于标准 Transformer，并与 BabyLM 2025 强基线保持竞争水平。

## 研究问题与动机
- **核心问题**：在固定数据预算、计算充裕的低资源预训练 regime 下，如何将额外的计算量转化为泛化收益，而非盲目增加参数量导致过拟合。
- **标准 Transformer 缩放失效**：缩小模型时固定词表大小使 embedding/LM head 占比过高；同时宽度、深度、passes 等计算扩展 knob 都会同步增加参数，二者强耦合，难以独立控制“计算深度”与“表示容量”。
- **Scaling Law 直觉不再适用**：Web-scale 下“更多参数+更多数据=更好性能”的规律在此失效，参数量本身退化为正则化手段，超过最优规模后下游性能呈非单调下降。
- **科学与实用动机**：该设置贴近儿童语言习得的数据预算（发展认知视角），且高质量人类文本数据日趋枯竭，亟需探索数据受限条件下的新 architecture-level scaling 路径。

## 核心贡献（创新点）
- **提出 RecursiveGPT 架构**：将单次因果解码块在深度上循环复用，把递归步数 $R$ 作为独立于参数规模的计算扩展轴。与 ALBERT 跨层共享的本质区别在于：明确用于低数据预训练的生成式解码场景，并将深度条件与归一化解耦以保持训练稳定。
- **因式分解嵌入的系统性引入**：将 $V \times H$ 的词表映射拆为 $V \times E$ 与 $E \times H$ 两级投影，使词表参数占比不再主导微小模型。与直接缩小 hidden size 的区别在于：保留 $H$ 以维持每层计算容量，仅压缩词表接口维度。
- **揭示低数据 optimal scale 的任务依赖性**：在双语料、多下游任务上验证最优模型尺寸强烈依赖数据量与评测目标（BLiMP 早饱和、COMPS 持续受益），打破“等比缩小 web-scale 模型”的惯性假设。
- **建立可复用的低数据递归训练配方**：开源完整训练流水线，并以 BabyLM 2025 挑战赛为基准完成与 GPT-BERT、AMLM hard decay 的正面对比，证明纯因果递归架构可作为数据高效方法的通用底座。

## 方法详解
- **递归权重共享（Recursive Weight Sharing）**：模型定义为 $h^{(0)} = \text{Embed}_{FE}(x_t)$，$h^{(r)} = F_\theta(h^{(r-1)}; \phi_r)$，$y_t = \text{Head}_{FE}(h^{(R)})$。$\theta$ 为跨步共享的注意力与 MLP 权重，$\phi_r$ 仅为每步独立的归一化参数与偏置，充当轻量级深度条件信号。
- **因式分解嵌入（Factorized Embeddings）**：输入映射 $\text{Emb}_{FE}(x_t) = x_t W_{emb
