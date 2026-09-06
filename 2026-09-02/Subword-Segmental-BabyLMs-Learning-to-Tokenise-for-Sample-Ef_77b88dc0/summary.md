---
title: "Subword-Segmental-BabyLMs-Learning-to-Tokenise-for-Sample-Ef"
source: https://arxiv.org/pdf/2609.01151v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:21:05"
field: "子词分词与低资源语言建模"
keywords: ["subword segmental language model", "tokenisation learning", "sample-efficient pretraining", "BabyLM", "morpheme boundary", "marginalization", "mixed subword scorer"]
innovations: ["SubSegGPT: 首个GPT-2式解码器子词分段模型，混合打分器平衡词汇层与字符级解码", "SubSegDeBERTa: 首个编码器式掩码子词分段模型，Word context encoder生成per-position双向上下文表示", "系统性验证可学习tokenisation在STRICT/STRICT-SMALL双赛道的样本效率增益及规模依赖性"]
benchmarks: ["BabyLM STRICT (100M words)", "BabyLM STRICT-SMALL (10M words)", "BLiMP", "BLiMP Supplement", "EWoK", "SuperGLUE finetuning", "Human-likeness (Read, AoA)"]
---

# 论文速读：Subword Segmental BabyLMs: Learning to Tokenise for Sample-Efficient Pretraining

## 一句话总结
论文提出两种可学习子词分词的语言模型（SubSegGPT 与 SubSegDeBERTa），将 tokenisation 从固定预处理步骤转化为端到端训练的隐变量，证明在样本受限场景下可学习的子词分词能显著提升预训练效率。

## 研究问题与动机
1. **固定分词的次优性**：标准 BPE/ULM 分词器基于频率目标学习子词边界，不保证对 LM 可学习性最优，尤其在数据稀缺时加剧表征学习难度。
2. **形态对齐缺失**：现有分词器产出的子词不可靠地对齐形态素边界（Batsuren et al., 2024），与语言习得的认知过程不符。
3. **发展合理性缺失**：人类并非天生具备固定词汇集，而是在语言习得过程中逐步学会将语音分割为有意义单位，固定分词范式与此过程相悖。
4. **现有 SSLM 的局限性**：先前 SSLM 仅在 Nguni 等低资源黏着语上验证，未知其能否推广至通用预训练样本效率提升。

## 核心贡献（创新点）
1. **SubSegGPT**：首个结合 GPT-2 骨干与子词分段建模的解码器式 SSLM，利用混合打分器（词汇层 + 字符 LSTM）动态计算子词概率。
   - 区别：先前的 SSLM 基于 LSTM，本文将其适配至现代 GPT 风格架构并验证于 BabyLM 通用预训练场景。
2. **SubSegDeBERTa**：首个编码器式掩码子词分段模型，通过 Word context encoder（1 层字符级 LSTM）为被掩码词每个字符位置生成带双向上下文的 per-position 表示。
   - 区别：首次将 SSL 框架扩展至 MLM 设定，解决单向自回归假设与双向上下文的冲突。
3. **系统性样本效率验证**：在 BabyLM Challenge STRICT（100M 词）与 STRICT-SMALL（10M 词）双赛道证明可学习分词提升零样本性能，SubSegDeBERTa 在 STRICT 零样本平均提升 +3.16 点，SubSegGPT 在 STRICT-SMALL 零样本 BLiMP +3.57 点、BLiMP Sup. +5.52 点。
4. **子词学习动力学分析**：首次量化 SSLM 训练过程中 fertility、边界翻转率、形态素边界 F1 的演化轨迹，揭示"快速变化→稳定收敛→高 fertility+弱形态对齐"的共性规律。

## 方法详解
1. **子词分段框架核心**：将 tokenisation T 视为隐变量，序列概率 $p(S) = \sum_{T \in \pi(S)} p(T)$，其中 $\pi(S)$ 为所有合法子词分割方案集合，$p(T)$ 仍按链式法则在 token 序列上计算。
2. **训练约束**：① 子词长度上限 $L$（本文设为 5 字符）；② 每个子词概率条件于未分词的字符级历史 $c_{<t_i}$，即 $p(t_i | c_{<t_i})$，近似丢弃分词历史但使条件计算 tractable。
3. **动态规划边际**：Meyer & Buys (2022) 的 DP 算法迭代计算 $p(S_{1:k})$，避免穷举指数级 tokenisation 组合。
4. **SubSegGPT 混合打分器**：$p(t_i | h_{<t_i}) = \lambda p_{\text{lex}}(t_i|h_{<t_i}) + (1-\lambda) p_{\text{char}}(t_i|h_{<t_i})$，其中 $p_{\text{lex}}$ 为固定词表语言模型头，$p_{\text{char}}$ 为 1 层字符级 LSTM（初始隐藏态 $h_{<t_i}$），$\lambda$ 由 sigmoid 投影动态学习。
5. **SubSegDeBERTa 字符位置编码**：DeBERTa  backbone 仅对被掩码词提供单一 $h_{[MASK]}$，引入 Word context encoder（1 层字符 LSTM，初始态为 $h_{[MASK]}$ 投影，每步拼接 $h_{[MASK]}$）生成 per-character 上下文表示 $h_k$，用于 conditioned 子词概率。

## 实验与结果
1. **数据集与设置**：2026 BabyLM Challenge 英文文本数据集，STRICT（100M 词，10 轮）、STRICT-SMALL（10M 词，10 轮），训练 10 个 epoch。
2. **基线**：等效大小的 GPT-2 BASE 与 DeBERTa-v2 BASE，同架构同数据，仅 difference 为固定分词 vs 可学习分词。
3. **零样本任务（Table 1）**：
   - STRICT：SubSegDeBERTa 平均 53.74 vs GPT-2 50.58（+3.16），BLiMP 78.22 vs 74.73（+3.49），BLiMP Sup. 68.73 vs 65.00（+3.73），EWoK 55.39 vs 54.37（+1.02），Entity 22.28 vs 16.91（+5.37）。
   - STRICT-SMALL：SubSegGPT 平均 47.17 vs GPT-2 46.52（+0.65），BLiMP 68.80 vs 65.23（+3.57），BLiMP Sup. 62.77 vs 57.25（+5.52）。
4. **微调任务（Table 2）**：STRICT 赛道优势消失，DeBERTa 微调平均 70.93 最高；STRICT-SMALL 赛道 SubSegGPT 平均 66.56 vs GPT-2 63.81（+2.75）。
5. **结论**：可学习分词的样本效率增益具有规模依赖性——SubSegGPT 在极端低资源（10M 词）更优，SubSegDeBERTa 在较大规模（100M 词）更可靠。

## 相关工作脉络
1. **Segmental Language Model (SLM)**：Sun & Deng (2018) 提出基于 LSTM 的分段语言模型，用于中文无监督分词，首次将分词作为隐变量 marginalise。
2. **Masked SLM**：Downey et al. (2022) 提出基于 transformer 的掩码分段语言模型，改进递归 SLM 的无监督分词性能。
3. **Subword SLM (SSLM)**：Meyer & Buys (2022) 首次提出子词分段建模，限制子词不跨越词边界，在 Nguni 黏着语上验证 SSLM 的 perplexity 和序列到序列优势。
4. **Transformer SSLM**：Meyer & Buys (2025) 提出 transformer 式 SSLM 并分析子词学习动力学，本文扩展至 GPT-2 与 DeBERTa-v2 骨干。
5. **定位差异**：本文首次将 SSLM 推广至通用英文预训练（BabyLM 设定），系统比较解码器式与编码器式 SSLM 的样本效率差异，并在严格控制的基线下隔离 tokenisation 学习贡献。

## 局限性与未来方向
1. **计算开销显著**：SubSegDeBERTa 训练需 ~5× A100 GPU 小时（STRICT 80h vs DeBERTa 17h），牺牲计算效率换取样本效率。
2. **人类相似性差**：两种模型在 AoA（年龄习得）和 Read（阅读时间）任务上与心理语言学数据相关性弱，甚至低于固定分词基线。
3. **子词长度硬约束**：最大 5 字符限制导致长词无法保留整词形式（如 "kitchen"、"little"），移除约束将边际计算不可行。
4. **生态效度有限**：仅验证于英文，对黏着语、高融合语的迁移性未知。
5. **未来方向**：放宽长度约束、探索更高效的边际近似、将 SSLM 与人类语言习得曲线对齐、扩展至多语言场景。

## 研究启发与可借鉴点
1. **混合打分器设计**：词汇层 + 字符解码器的组合有效兼顾常见子词概率准确性与 OOV 子词覆盖，该设计可迁移至任何需处理开放词表的生成模型。
2. **规模依赖的策略选择**：SubSegGPT 在 10M 词更优、SubSegDeBERTa 在 100M 词更优，提示团队在资源受限场景优先选择解码器式 SSLM，数据充足时可选编码器式。
3. **分词动力学量化指标**：fertility、边界翻转率、形态素边界 F1 可作为可学习 tokenisation 模型的通用诊断工具，辅助超参选择。
4. **端到端 tokenisation 范式**：将分词从预处理步骤释放为可学习组件的思路可扩展至分句、断词、音节切分等其他 NLP 任务。

## 关键术语表
- **Subword Segmental Language Model (SSLM)**：将子词分词作为隐变量 marginalise 的语言模型框架，tokenisation 与 LM 联合优化。
- **Fertility**：每个词的平均子词数量，反映分词粒度，越高表示子词越短。
- **Boundary flip rate**：连续训练检查点之间子词边界变化的比例，衡量分词学习稳定性。
- **Morpheme boundary F1**：学到子词边界与人工标注形态素边界的一致程度，衡量分词的形态学可解释性。
- **Dynamic programming marginal**：迭代计算序列边际概率 $p(S_{1:k})$ 的算法，避免穷举所有 tokenisation。
- **Mixture subword scorer**：$p(t|c) = \lambda p_{\text{lex}} + (1-\lambda) p_{\text{char}}$，动态平衡固定词表与字符级解码的贡献。
- **Word context encoder**：SubSegDeBERTa 引入的 1 层字符级 LSTM，为被掩码词每个字符位置生成 per-position 上下文表示。
- **BabyLM Challenge**：面向样本高效预训练的竞赛平台，设定 STRICT（100M 词）与 STRICT-SMALL（10M 词）双赛道。

## 可复现要素
- **数据集**：2026 BabyLM Challenge 官方英文文本数据集（由主办方发布，公开可用）。
- **代码**：论文未提供开源链接，附录提供超参与模型配置表（Tables 4、5），可据此复现。
- **权重**：未声明开源。
- **关键超参**：最大子词长度 5 字符、lexicon size 10k–40k、char vocab 416–744、hidden size 256（字符 LSTM）、学习率 5e-4、warmup ratio 0.1、batch size 16、10 epochs、sequence length 512/1024。
- **硬件**：A100 GPU，STRICT 训练 80–85 小时，STRICT-SMALL 训练 9–16 小时。
