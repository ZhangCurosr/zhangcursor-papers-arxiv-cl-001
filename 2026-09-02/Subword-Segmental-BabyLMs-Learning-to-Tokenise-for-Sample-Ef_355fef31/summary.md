---
title: "Subword-Segmental-BabyLMs-Learning-to-Tokenise-for-Sample-Ef"
source: https://arxiv.org/pdf/2609.01151v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:20:48"
field: "低资源语言模型预训练"
keywords: ["子词分段建模", "可学习分词", "样本效率", "BabyLM", "掩码语言建模"]
innovations: ["提出 SubSegGPT，首次将 SSLM 适配到 GPT-2 解码器架构", "提出 SubSegDeBERTa，首次将 SSLM 扩展到掩码语言建模场景", "在 BabyLM 挑战赛中验证可学习分词在小数据下的样本效率增益"]
benchmarks: ["BabyLM 2026 STRICT", "BabyLM 2026 STRICT-SMALL", "BLiMP", "SuperGLUE"]
---

# 论文速读：Subword-Segmental-BabyLMs-Learning-to-Tokenise-for-Sample-Ef

## 一句话总结
论文提出两种将子词分词作为可学习组件的端到端训练语言模型（SubSegGPT 和 SubSegDeBERTa），通过将分词与语言建模联合优化，在小数据场景下显著提升了预训练的样本效率，其中 SubSegDeBERTa 在 100M 词轨道上零样本性能超越 GPT-2 基线 3.16 分。

## 研究问题与动机
- **分词与建模割裂问题**：标准 NLP 管道将 subword 分词作为预处理固定步骤，模型无法根据训练目标优化分词策略；而人类语言习得是逐步学习分段过程的，并非天生拥有固定词汇表。
- **频率驱动分词的局限**：BPE/ULM 等算法基于频率目标学习子词边界，无法保证产生的单元对 LM 可学习性最优；大数据下模型鲁棒，但小数据下次优分词会加剧学习困难。
- **可学习分词的潜力未验证**：先前研究仅在低资源黏着语（Nguni 语族）上验证了 SSLM 的有效性，其在通用英语 BabyLM 场景下的样本效率增益仍未知。

## 核心贡献（创新点）
- **提出 SubSegGPT**：首个将 SSLM 框架适配到 GPT-2 风格解码器架构的工作，通过字符级历史编码器 + 词表/LSTM 混合子词打分器实现自回归分词学习，区别于先前 LSTM 基 SSLM，采用了更现代的架构约定。
- **提出 SubSegDeBERTa**：首次将 SSLM 扩展到掩码语言建模（MLM）场景，引入词上下文编码器解决 DeBERTa-v2 仅输出单一 [MASK] 表示的问题，使编码器也能联合学习分词。
- **系统性验证样本效率增益**：在 BabyLM 2026 STRICT（100M 词）和 STRICT-SMALL（10M 词）两个赛道上全面评估，证明可学习分词在数据受限场景下有效，SubSegGPT 在 STRICT-SMALL 上全面超越基线。

## 方法详解
**子词分段建模框架（SSLM）**：将分词视为潜变量，序列概率对所有候选分词方案边缘化：$p(S) = \sum_{T \in \pi(S)} p(T)$，通过动态规划高效计算。约束：子词最大长度 $L$，条件概率基于字符级上下文 $c_{<t_i}$。

- **SubSegGPT 架构**：
  - **字符级历史编码器**：用 GPT-2 backbone（去掉 LM head）编码前缀字符序列，取最后一个字符的隐藏状态 $\mathbf{h}_{<t_i}$ 代表历史。
  - **子词分段打分器**：混合模型 $p(t_i|c_{<t_i}) = \lambda p_{lex} + (1-\lambda)p_{char}$，其中 $p_{lex}$ 基于固定大小词表（V 个高频子词）直接预测，$p_{char}$ 用 1 层字符级 LSTM 逐字符生成并连乘概率；门控 $\lambda$ 由 sigmoid 线性投影动态计算。
  - 通过动态规划算法对所有候选分词方案边缘化，端到端最小化负对数似然。

- **SubSegDeBERTa 架构**：
  - **字符级上下文编码器**：DeBERTa-v2 处理字符序列，随机遮蔽单词为单个 [MASK] token，输出 $\mathbf{h}_{[MASK]}$ 编码双向上下文 $(w_{<i}, w_{>i})$。
  - **词上下文编码器**：为解决 [MASK] 处只有单一表示的问题，引入 1 层字符级 LSTM，初始隐藏状态为 $\mathbf{h}_{[MASK]}$ 的投影，每一步拼接 $\mathbf{h}_{[MASK]}$ 到输入，输出逐位置表示 $\mathbf{h}_k$。
  - **掩码词打分器**：同样使用混合模型，以 $\mathbf{h}_k$ 为条件计算子词概率，对遮蔽单词的所有候选分词方案边缘化，端到端训练。

## 实验与结果
- **数据集与赛道**：BabyLM 2026 挑战赛数据，STRICT（100M 词，10 epoch）和 STRICT-SMALL（10M 词）。
- **基线**：同等架构的 GPT-2 和 DeBERTa-v2 固定分词基线。
- **主要结果**：
  - **STRICT 轨道**：SubSegDeBERTa 零样本平均 53.74，超越 GPT-2（50.58）+3.16 分；BLiMP 78.22，MRPC 微调 90.54。
  - **STRICT-SMALL 轨道**：SubSegGPT 零样本平均 47.17，超越 GPT-2（46.52）+0.65 分；BLiMP +3.57，BLiMP Sup. +5.52。
  - **微调任务**：STRICT 下细分词模型优势缩小，STRICT-SMALL 下 SubSegGPT 仍以 66.56 平均超越 GPT-2（63.81）。
- **学习动力学**：两模型均经历初期快速变化后收敛， fertility（每词子词数）递增至高于 BPE 基线（1.23），形态边界 F1 高于随机插入（~11.6%）但低于 Morfessor（42.9%）。

## 相关工作脉络
- **Segmental LM（SLM）**（Sun & Deng, 2018）：LSTM 风格的片段语言模型，对字符序列做边缘化，目标是无监督词发现，非子词级别。
- **Masked SLM**（Downey et al., 2022）：基于 Transformer 的无监督词发现模型，仅适用于无空格语言（如中文），未扩展到子词级通用 LM。
- **原始 SSLM**（Meyer & Buys, 2022）：首个子词分段模型，但基于 LSTM、仅在 Nguni 语族上评估，未验证通用预训练场景。
- **Transformer SSLM**（Meyer & Buys, 2025）：将 SSLM 扩展到 Transformer，但仍限于低资源黏着语研究分词动力学。
- **BabyLM Challenge**：提供 100M/10M 词严格数据约束的预训练评测平台，本文首次将 SSLM 应用于此场景验证样本效率假设。

## 局限性与未来方向
- **计算开销大**：子词边缘化导致训练时间显著增加，SubSegDeBERTa 需约 5× DeBERTa 的 A100 GPU 小时，以计算效率换取样本效率。
- **子词长度约束**：最大子词长度限制为 5 字符，导致 "kitchen"/"little" 等无法保持完整；去除该约束在计算上不可行。
- **人机相似性不足**：在两轨道上，模型在人类相似性任务（阅读时间预测、词汇习得年龄）上均不及固定分词基线。
- **未来方向**：探索无界子词长度的可行近似算法；研究分词学习与人类语言习得路径的对齐机制。

## 研究启发与可借鉴点
- **混合打分器设计**：词表层（覆盖高频子词）+ 字符级 LSTM 层（覆盖任意子词）的混合架构，结合可学习门控 λ，可在保证覆盖率的同时兼顾效率，适合任何需要处理任意序列片段的模型。
- **分词-建模联合优化的范式**：将分词视为可学习潜变量而非预处理固定步骤，在样本效率敏感场景（低资源、多语言）下值得系统性探索。
- **动态规划边缘化的应用**：对所有候选分段方案边缘化序列概率的实现方式，可推广到其他需要隐式结构预测的序列建模任务。
- **BabyLM 评测适配**：通过 wrapper 将子词边缘化输出转化为标准 per-position log-probability，为评估非标准架构提供了参考。

## 关键术语表
**Subword Segmental Language Modelling (SSLM)**：将子词分词作为可学习潜变量的框架，通过对所有候选分词方案边缘化来联合优化分词与语言建模。
**Fertility**：平均每个单词被分割成的子词数量，反映子词粒度的粗细程度。
**Boundary Flip Rate**：相邻检查点之间子词边界发生变化的比例，衡量分词学习过程的稳定性。
**Morpheme Boundary F1**：学习到的子词边界与真实词素边界的重合度 F1 分数，评估形态对齐质量。
**Dynamic Programming Marginalisation**：利用动态规划高效计算所有候选分词方案概率之和的算法，避免穷举指数级分词。
**SubSegGPT / SubSegDeBERTa**：本文提出的两种子词分段模型，分别基于 GPT-2 解码器和 DeBERTa-v2 编码器架构。
**BabyLM Challenge**：面向低资源语言模型预训练的基准挑战赛，设 STRICT（100M 词）和 STRICT-SMALL（10M 词）两个数据约束赛道。

## 可复现要素
- **数据集**：BabyLM 2026 Challenge 官方文本数据集（纯文本，STRICT 100M 词 / STRICT-SMALL 10M 词），由组织方发布。
- **代码**：论文未提及代码开源；附录提供了超参数详情（Table 4/5）和评估 wrapper 实现细节。
- **关键超参**：最大子词长度 5 字符；词表大小 10k-40k；LSTM 隐藏维度 256；学习率 5e-4；masking ratio 0.3-0.4；batch size 16；序列长度 1024。
