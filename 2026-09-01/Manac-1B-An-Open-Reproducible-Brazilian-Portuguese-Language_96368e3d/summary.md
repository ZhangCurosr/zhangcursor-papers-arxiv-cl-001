---
title: "Manac-1B-An-Open-Reproducible-Brazilian-Portuguese-Language"
source: https://arxiv.org/pdf/2608.30114v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 13:50:37"
field: "低资源语言大模型与可复现评估"
keywords: ["Brazilian Portuguese", "reproducible training", "tokenizer fidelity", "paired evaluation", "LAMBADA-PT", "CALAME-PT", "open language model"]
innovations: ["发布完全容器化可复现的 1.72B 巴西葡语基础模型 Manacá-1B，公开全部训练与评估证据链", "提出 tokenizer 感知的配对评估协议，在四个基准上使用 paired McNemar 检验与标准误差", "发现并修复 SentencePiece→HuggingFace 转换中丢失 nmt_nfkc_cf normalizer 导致的静默评估下降（LAMBADA-PT -20.3pp）"]
benchmarks: ["CALAME-PT", "ARC-Challenge-PT", "HellaSwag-PT", "LAMBADA-PT"]
---

# 论文速读：Manac-1B-An-Open-Reproducible-Brazilian-Portuguese-Language

## 一句话总结
本文发布了 Manacá-1B（1.72B 参数），一个从头训练、完全容器化可复现的巴西葡萄牙语语言模型，并提出了一套带配对显著性检验的 tokenizer 感知评估协议，同时揭示并修复了 HuggingFace 转换中丢失 SentencePiece 大小写折叠 normalizer 的隐藏评估陷阱。

## 研究问题与动机
- **开源葡语模型缺乏可复现性**：Tucano、Sabiá、GlórIA 等巴西/欧洲葡语模型训练流水线端到端难以复现，配置和日志未完整公开。
- **模型比较缺乏不确定性度量**：已有工作通常只报告点估计，缺少标准误差和配对显著性检验，导致结果对比不可靠。
- **Tokenizer 被忽视且影响评估**：Tokenizer 默默决定模型如何解析文本，但几乎从不被审计；将 SentencePiece 转为 HuggingFace fast 格式会静默丢失 nmt_nfkc_cf 正常化，使评估分数严重失真（如 LAMBADA-PT 从 45.3 骤降至 25.0）而不易察觉。
- **小规模基础模型的诚实定位缺位**：已有大模型（如 Sabiá-7B）因训练量巨大而领先，但同类规模的国产模型缺乏统一基准下的严格对比。

## 核心贡献（创新点）
- **发布一个完全可复现的 1.72B 葡语基础模型**：Manacá-1B 训练于约 200 亿巴西葡语 token，容器化流水线、原始日志、逐样本预测向量全部公开，区别于仅有模型权重而无完整证据链的已有工作。
- **提出 tokenizer 感知的配对评估协议**：在四个巴西葡语基准上使用同一 harness、相同提示和配对 McNemar 检验，每条目均附二项标准误差/Bootstrap 置信区间，优于仅报告点估计的行业惯例。
- **发现并修复 SentencePiece→HuggingFace 转换中的正常化丢失陷阱**：量化了未加 normalizer 导致的静默性能下降（LAMBADA-PT -20.3pp，困惑度从 17.3 升至 10⁶ 量级），并开源一行修复代码。
- **提供严格的 harness 验证与缺陷透明度报告**：在已知公开数字上校准评估管道，同时明确标注 GlórIA-1b3 和 mGPT-1b3 的 CALAME-PT 分数因生成式协议被高估约 7–8pp。

## 方法详解
- **架构设计**：Llama-style decoder-only，24 层、隐藏维度 2048、FFN 宽度 8192（SwiGLU）、32 个注意力头 + 8 组 KV（GQA）、RoPE（θ=5×10⁵）、RMSNorm、所有线性层含 learned bias。
- **Tokenizer**：SentencePiece unigram，词表 64k（填充至 64,128），nmt_nfkc_cf 正常化（NFKC + 大小写折叠），character coverage 0.9995，启用 byte fallback 与数字分割，模型"天生小写"。
- **预训练数据**：约 201 亿 token、3370 万文档，来源为 GigaVerbo 子集（73%，Apache-2.0）、Ulysses Tesemõ 法律文献（24%，公共领域）、葡萄牙 Wikipedia（3%，CC BY-SA 4.0）；仅做源内 MinHash 去重，不做跨源全局去重；文档按固定顺序拼接后在固定 seed 下 shuffle。
- **训练配置**：LLM-jp fork 的 Megatron-LM，z-loss 正则化 + Adam（β₁=0.9, β₂=0.999, ε=10⁻⁸, weight decay 0.1, grad clip 1.0），学习率线性预热 2000 步至 3×10⁻⁴ 后 cosine 衰减至 3×10⁻⁵；总步数 20,000、全局 batch 512、seq len 4096，约 419 亿 update tokens（≈24 tok/param，接近 compute-optimal 比率）；DP=2、ZeRO-1、micro-batch=1+ 梯度累积、全 activation recompute、FlashAttention、bfloat16，两台 24GB GPU（PCIe）即可运行。
- **训练稳定性**：2000 次日志记录中 0 skipped/NaN 步骤；loss 从 11.41 单调下降至 2.48，validation loss 降至 2.07 nats（PPL=7.96）；出现数次梯度 norm 尖峰（最高 24.5）均由 z-loss 和 optimizer 自行恢复。
- **模型导出**：针对 Megatron→HuggingFace 定制转换器，显式保留所有 bias、解包 GQA 与 SwiGLU 融合矩阵、将 untied lm_head 独立拷贝；三级验证（键名完整性、葡萄牙语续写质量、与训练 tokenizer identifier 对齐）。
- **评估协议**：CALAME-PT（贪婪最后词生成，2075 条）；ARC-Challenge-PT / HellaSwag-PT（Okapi 机翻版本，log-likelihood 25-shot / 10-shot）；LAMBADA-PT（log-likelihood 0-shot）；使用 EleutherAI lm-evaluation-harness；配对比较采用 McNemar 检验 + 共享样本 bootstrap，优于边际置信区间重叠判断。

## 实验与结果
- **数据集**：CALAME-PT、ARC-Challenge-PT、HellaSwag-PT、LAMBADA-PT（均为巴西葡语或机翻葡语基准）。
- **基线**：TeenyTinyLlama-160m/460m、Tucano-160m/630m/1b1/2b4、GlórIA-1b3、mGPT-1b3、Sabiá-7B（共 9 个开源模型，span 0.16B–7B、三种 tokenizer 家族）。
- **主要结果（准确率 ± SE）**：

| 模型 | Params | CALAME-PT | ARC-Ch-PT | HellaSwag-PT | LAMBADA-PT |
|---|---|---|---|---|---|
| Manacá-1B | 1.72B | 60.63±1.07 | 27.18±1.30 | 41.61±0.51 | **45.31±0.69** |
| Tucano-1b1 | 1.10B | 59.08±1.08 | 29.66±1.34 | 44.23±0.52 | 31.50±0.65 |
| Tucano-2b4 | 2.40B | 59.57±1.08 | 30.85±1.35 | 48.63±0.52 | 34.35±0.66 |
| Sabiá-7B | 7.00B | **63.23±0.06** | **46.67±1.46** | **64.55±0.50** | **63.67±0.67** |

- **最强结果**：Manacá-1B 在 sub-7B 模型中 LAMBADA-PT 最佳（45.31），较第二名 GlórIA-1b3（35.30）高出 10pp，且配对 McNemar 检验对所有 sub-7B 基线均显著（p < 10⁻⁴）；CALAME-PT 为 60.63，与 Sabiá-7B 差距 2.6pp，但显著优于所有其他 sub-7B 模型。
- **重要对比**：
  - LAMBADA-PT：Manacá-1B 显著超越所有 sub-7B 基线；HellaSwag-PT 和 ARC-Challenge-PT 上因训练量仅为 Tucano 的约 1/10，落后于 Tucano-1b1 / 2b4，但显著优于同参数量的 GlórIA / mGPT（体现巴西葡语专项预训练价值）。
  - ARC-Challenge-PT：所有 sub-2B 模型均接近 25% 随机水平，无显著差异，属合理基线行为。
- **Tokenizer 修复效果**：LAMBADA-PT 准确率从 25.0 跃升至 45.3（+20.3pp），困惑度从 ~10⁶ 降至 17.3。
- **Harness 验证**：Tucano 系列 SentencePiece 模型在 CALAME-PT 上复现误差约 ±1pp；GlórIA/mGPT 因生成式协议被高估 7–8pp，作者明确标注而非隐藏。

## 相关工作脉络
- **Tucano 系列（Corrêa et al., 2024）**：同家族巴西葡语模型，但训练 token 超 10 倍于 Manacá-1B；本文在 sub-7B 规模上与 Tucano-1b1 / 2b4 公平比较，并强调训练预算差异而非架构劣势解释推理差距。
- **TeenTinyLlama（Corrêa et al., 2024）**：极小基线（160m/460m），用于构建尺寸-性能趋势线；本文沿用其数据与训练 recipe 思想但扩展至 1.7B 规模。
- **Sabiá（Pires et al., 2023）**：7B 级多Variety 葡语模型，是本文唯一显著领先的参照；本文将其定位为"大规模 vs 小尺度"对照组，凸显 compute-optimal 训练的诚实定位。
- **GlórIA（Lopes et al., 2024）**：欧洲葡语模型 + CALAME-PT 基准的提出者；本文揭示其 byte-level BPE 在生成式最后词协议下被高估 7–8pp，并提出统一 log-likelihood 基准的必要性。
- **mGPT（Shliazhko et al., 2022）**：多语言基础模型作为对照，凸显 1.7B 巴西葡语专项模型在同规模下的优势（LAMBADA-PT +7.9pp，HellaSwag-PT +16.3pp）。
- **LLM-jp 项目与 Megatron-LM（Shoeybi et al., 2019）**：本文继承 LLM-jp-3.1-1.8B 架构与训练配方，证明该配方在巴西葡语上的可迁移性与容器化复现价值。

## 局限性与未来方向
- **训练 token 预算有限**：约 42B update tokens（~2 轮），远低于 Tucano（>10 倍），推理差距主要源于数据量而非架构。
- **评估基准覆盖不全**：尚未纳入 Open Portuguese LLM Leaderboard 的原生巴西考试题与推理任务；LAMBADA-PT 使用 log-likelihood 近似而非原始 generation-based 协议。
- **四种基准评分方式不一致**：CALAME-PT 用生成式，其余三者用 log-likelihood，跨榜横向比较需谨慎。
- **仅有 base model**：未进行 instruction tuning 评估，下游指令遵循能力未知。
- **未来方向**：引入原生巴西基准与多语言大模型（Llama-3.2、Qwen2.5）外比较；开展葡语变体过滤的对照消融；将 token 预算推向过训练区分离数据与规模效应；训练 instruction-tuned 变体。

## 研究启发与可借鉴点
- **Tokenizer 审计应成为评估前置步骤**：任何涉及 SentencePiece normalizer / 大小写折叠 / 特殊预处理的模型，在转换到 HuggingFace 时必须逐 token identifier 核对，避免"静默下降"污染对比结果。
- **配对显著性检验优于边际置信区间**：同一共享样本集上的 McNemar 检验 + bootstrap 能捕捉误差相关性，提升对比统计效力；建议在新语言模型评测中成为标配。
- **Compute-optimal 小模型同样有社区价值**：无需追求最大 scale，1–2B 规模在专项语言建模上可做到 open baselines 中最强，并降低复现门槛与硬件要求。
- **容器化 + 版本锁定 + 原始日志公开构成可复现黄金标准**：provenance 文件记录 commit、超参、容器版本，配合逐样本预测向量，实现"任一键盘可重算任一数字"。
- **透明标注评估 protocol 差异的影响**：主动披露 GlórIA / mGPT 的分数 inflated，比隐藏差异更能建立学术可信度，并为后续工作修正评估协议提供参考。

## 关键术语表
- **Manacá-1B**：1.72B 参数、从头训练的巴西葡萄牙语 decoder-only 基础语言模型。
- **nmt_nfkc_cf**：SentencePiece 的正常化规则，先执行 NFKC 形态学标准化，再执行 case folding（大写→小写），使模型"天生小写"。
- **Paired McNemar 检验**：在共享样本上检验两模型错误分布是否独立的分类精度显著性检验，适用于配对比较。
- **Compute-optimal 训练比率**：Hoffmann et al. (2022) 提出的最优化训练公式，约 20 token/parameter 时损失下降效率最高；Manacá-1B 约 24 tok/param 接近该最优。
- **z-loss**：对输出 logits 施加的正则化损失（PaLM / LLM-jp 风格），防止 log-softmax 数值不稳定。
- **LM-Evaluation-Harness**：EleutherAI 开源的多任务大模型统一评估框架（version 0.4.5）。
- **CALAME-PT**：基于 native 葡萄牙语段落的最后词预测基准，由 GlórIA 论文引入。
- **Tokenized Fidelity**：评估时 tokenizer 的输出必须与训练时一致；本文的核心方法论主张，即 tokenizer 审计优先于任何分数报告。

## 可复现要素
- **数据集**：GigaVerbo 子集（Apache-2.0）、Ulysses Tesemõ（公共领域）、Wikipedia-pt（CC BY-SA 4.0）；全部可公开获取。
- **代码**：https://github.com/Instituto-IA-LNCC/manaca-1b-base（容器化训练/转换/评估 pipeline、脚本、原始日志、逐样本预测向量）。
- **模型权重与 corrected tokenizer**：https://huggingface.co/menezesbruno/manaca-1b-base（CC BY 4.0）。
- **关键超参**：20,000 steps、global batch 512、seq len 4096、lr 3e-4→3e-5 cosine、Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸, wd=0.1, grad clip=1.0)、seed=1234、init std=0.02、bfloat16、z-loss。
- **硬件**：2×24GB GPU (PCIe, no NVLink)。
