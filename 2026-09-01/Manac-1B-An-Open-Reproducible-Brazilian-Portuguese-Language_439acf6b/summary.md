---
title: "Manac-1B-An-Open-Reproducible-Brazilian-Portuguese-Language"
source: https://arxiv.org/pdf/2608.30114v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 13:50:36"
field: "低资源语言大模型"
keywords: ["Brazilian Portuguese LLM", "reproducible training", "tokenizer fidelity", "paired evaluation", "compute-optimal training"]
innovations: ["提出完全容器化可复现的巴西葡语1.72B模型Manacá-1B", "发现SentencePiece转HuggingFace丢失nmt_nfkc_cf归一化导致LAMBADA-PT得分暴跌20pp的评估陷阱", "在统一harness下对九个基线提供带配对McNemar检验的严格对比评估"]
benchmarks: ["CALAME-PT", "ARC-Challenge-PT", "HellaSwag-PT", "LAMBADA-PT"]
---

# 论文速读：Manacá-1B-An-Open-Reproducible-Brazilian-Portuguese-Language

## 一句话总结
论文提出了 Manacá-1B，一个 1.72B 参数的巴西葡萄牙语开源语言模型，采用完全容器化的可复现训练管道，并在四个葡萄牙语基准上提供了带不确定性估计的严格对比评估；同时揭示了一个普遍存在的评估陷阱——将 SentencePiece tokenizer 转为 HuggingFace fast 格式时会静默丢失大小写折叠归一化，导致 LAMBADA-PT 准确率从 45.3 暴跌至 25.0。

## 研究问题与动机
1. **巴西葡萄牙语开源模型缺乏可复现性**：现有模型（Tucano、Sabiá、GlórIA 等）的训练管道难以复现，代码和日志未完整公开。
2. **模型对比缺乏不确定性度量**：多数工作仅报告点估计值，未提供标准误差和配对显著性检验，导致结论不可靠。
3. **Tokenizer 审计被忽视**：Tokenizer 作为模型读取文本的关键组件，其格式转换中的归一化丢失问题往往被忽视，却会严重扭曲评估结果。

## 核心贡献（创新点）
1. **开源可复现的巴西葡萄牙语基础模型**：Manacá-1B 在容器化 Megatron-LM 管道中从零训练，所有配置、日志和 artifact 均公开；与已有工作本质区别在于完整复现路径而非单纯发布权重。
2. **严格的一体化评估协议**：在四个基准（CALAME-PT、ARC-Challenge-PT、HellaSwag-PT、LAMBADA-PT）上用单一 harness 对比九个开放基线，每指标附二项标准误差和配对 McNemar 检验；与已有工作区别在于引入配对统计检验和置信区间。
3. **发现并修复 Tokenizer 保真度陷阱**：揭示 HuggingFace 转换丢弃 SentencePiece nmt_nfkc_cf 归一化器的问题，量化其对 LAMBADA-PT 的致命影响（45.3→25.0），并提供一行修复方案；这是对模型评估基础设施的一次关键修正。

## 方法详解
**架构设计**：Decoder-only Llama-style transformer，24 层，dim=2048，FFN dim=8192（SwiGLU），32 个 attention head + 8 个 KV groups（GQA），RoPE（θ=5×10⁵），RMSNorm，所有线性层含 learnable bias。

**Tokenizer**：SentencePiece unigram，vocab=64k（pad 到 64,128），nmt_nfkc_cf 归一化（NFKC + case folding），byte fallback 启用，character coverage=0.9995。模型默认小写运行。

**预训练数据**：约 201 亿 tokens，3370 万文档，来源包括 GigaVerbo 子采样（73.1%，Apache-2.0）、Ulysses Tesemõ 法律文献（23.9%，public domain）、葡萄牙语 Wikipedia（3.1%，CC BY-SA 4.0）。无跨源全局去重，仅做源内 MinHash 去重。

**训练配置**：LLM-jp fork 的 Megatron-LM，bfloat16，global batch=512，seq_len=4096，20,000 步 ≈ 419 亿 update tokens（≈24 tok/param，接近 compute-optimal 点）。Adam（β₁=0.9, β₂=0.999, ε=10⁻⁸），weight decay=0.1，grad clip=1.0，lr 线性 warmup 2000 步至 3×10⁻⁴ 后余弦衰减至 3×10⁻⁵，PaLM-style z-loss 正则化。

**硬件与并行**：2×24GB GPU（PCIe 无 NVLink），纯数据并行（DP=2，ZeRO-1），full activation recompute，FlashAttention，micro-batch=1 + gradient accumulation。

**稳定性**：零跳过步、零 NaN；训练 loss 从 11.41 单调降至 2.48，validation loss 达 2.07 nats（PPL=7.96）；梯度范数在数次瞬态尖峰（最大 24.5）后自恢复。

**评估协议**：CALAME-PT 用贪心生成评分，其余三个用 log-likelihood；所有模型使用统一 harness；不确定性用二项标准误差（accuracy）和 harness 估计（log-likelihood）；配对比较用 McNemar 检验和 bootstrap CI。

## 实验与结果
**数据集**：CALAME-PT（2075 passages，last-word prediction）、ARC-Challenge-PT（25-shot）、HellaSwag-PT（10-shot）、LAMBADA-PT（0-shot，log-likelihood）。

**基线模型**：TeenyTinyLlama-160m/460m、Tucano-160m/630m/1b1/2b4、GlórIA-1b3、mGPT-1b3、Sabiá-7B，共九种。

**主要结果**：

| 模型 | Params | CALAME-PT | ARC-Ch-PT | HellaSwag-PT | LAMBADA-PT |
|------|--------|-----------|-----------|--------------|------------|
| Manacá-1B | 1.72B | 60.63±1.07 | 27.18±1.30 | 41.61±0.51 | **45.31±0.69** |
| Tucano-1b1 | 1.10B | 59.08±1.08 | 29.66±1.34 | 44.23±0.52 | 31.50±0.65 |
| Tucano-2b4 | 2.40B | 59.57±1.08 | **30.85±1.35** | **48.63±0.52** | 34.35±0.66 |
| Sabiá-7B | 7.00B | **63.23±0.06** | **46.67±1.46** | **64.55±0.50** | **63.67±0.67** |

**最强结果**：Manacá-1B 在 <7B 量级中 LAMBADA-PT 最高（45.31%），显著超过 Tucano-1b1（+13.82pp, p<10⁻⁴）、Tucano-2b4（+10.96pp, p<10⁻⁴）及 GlórIA/mGPT；但 HellaSwag-PT 和 ARC-Challenge-PT 因训练 token 预算差距落后于更大数据量的 Tucano 系列；ARC 任务所有 <2B 模型接近随机水平（~25%）。

**Token 转换陷阱量化**：未修正 tokenizer 使 LAMBADA-PT 从 45.3 降至 25.0，token-level perplexity 从 17.3 升至 ~10⁶；修正方式为注入 NFKC+Lowercase 归一化序列。

## 相关工作脉络
1. **Tucano 系列**（Corrêa et al., 2024）：训练数百亿 tokens，规模达 2.4B，数据丰富但训练管道不可复现；本文在同等参数预算下证明巴西葡语专属训练的价值。
2. **TeenyTinyLlama**（Corrêa et al., 2024）：极小模型（160m/460m）基准，用于验证小模型评估一致性。
3. **GlórIA**（Lopes et al., 2024）：欧洲葡萄牙语 decoder，引入 CALAME-PT 基准；本文揭示其 byte-level BPE tokenizer 在生成式评分下会被高估 7-8pp。
4. **Sabiá**（Pires et al., 2023）：7B 多尺度模型，本文唯一在全部四项任务上超越 Manacá-1B 的基线，但参数量为其四倍。
5. **LLM-jp 系列**（NII Japan）：Manacá-1B 直接继承其架构和训练 recipe（LLM-jp-3.1-1.8B），验证了跨语言迁移的可复现性。
6. **Okapi 翻译基准**（Lai et al., 2023）：ARC 和 HellaSwag 的葡萄牙语机器翻译来源，本文指出翻译引入的非原生词汇和干扰项影响了小模型的推理得分。

## 局限性与未来方向
1. **训练 token 预算有限**：约 42B update tokens（~2 epochs over 20B corpus），远低于 Tucano 系列（数百亿 tokens），导致推理任务表现接近随机。
2. **仅评估 base model**：未训练 instruction-tuned 变体，无法反映实际可用性。
3. **基准覆盖不足**：缺少 Open Portuguese LLM Leaderboard 的原生巴西考试和推理任务。
4. **评分协议不一致**：LAMBADA-PT 用 log-likelihood 而非原始 generation-based 协议，CALAME-PT 用 generation，跨任务比较需谨慎。
5. **未来方向**：扩展 token 预算至 over-trained  regime、进行巴西/欧洲葡语变体过滤 ablation、引入更多 multilingual baselines（Llama-3.2、Qwen2.5）、训练 instruction-tuned 版本。

## 研究启发与可借鉴点
1. **Token 归一化审计必须纳入评估流程**：任何经过非平凡归一化（如 case folding、NFKC）训练的 tokenizer，在格式转换后必须逐字符验证与训练 tokenizer 的一致性；建议团队在模型转换 pipeline 中加入 probe string 校验环节。
2. **配对显著性检验是公平对比的基础**：相同样本上的配对 McNemar 检验比边际置信区间重叠更准确，可复用到本团队的模型对比实验中。
3. **Near-compute-optimal 训练是资源受限下的合理选择**：24 tok/param 虽不如大 budget 模型，但证明了在可复现前提下的小规模高质量训练价值。
4. **全量 artifact 开源提升社区可信度**：raw training log、per-example prediction vectors、provenance 文件三位一体，值得在团队项目中推广。
5. **多评分协议并存的基准需谨慎跨任务比较**：当不同子任务使用不同评分方式（generation vs log-likelihood）时，应优先在同协议内比较。

## 关键术语表
**Manacá-1B**：本文提出的 1.72B 参数巴西葡萄牙语 decoder-only 语言模型，从零训练，完全开源可复现。

**nmt_nfkc_cf**：SentencePiece 归一化规则，依次执行 NFKC 规范化（Unicode 兼容分解再组合）和 case folding（转小写），使模型默认小写运行。

**Compute-optimal ratio**：Hoffmann et al. (2022) 提出的训练效率最优 token/参数比（约 20 tok/param），Manacá-1B 以 24 tok/param 接近该点。

**Paired McNemar test**：用于配对二分类结果的显著性检验，比较两个模型在同一组样本上的错误是否相关，比独立检验更适用于公平对比。

**Calame-PT**：由 GlórIA 团队引入的葡萄牙语 last-word prediction 基准，2075 条原生段落，贪心生成最后单词后计算准确率。

**z-loss**：PaLM/LLM-jp 风格的输出 logits 正则化项，惩罚过大的 logit 值以提升数值稳定性，防止训练溢出。

**Grouped-Query Attention (GQA)**：将 KV heads 分组共享的 attention 变体，Manacá-1B 使用 32 个 query head 和 8 个 KV group。

**Byte-fallback**：SentencePiece 机制，当 token 超出 vocab 时用字节级编码回退，未被正确归一化的大写单词会触发此回退导致分数暴跌。

## 可复现要素
- **数据集**：GigaVerbo（Apache-2.0）、Ulysses Tesemõ（public domain）、Wikipedia-pt（CC BY-SA 4.0）；均开源可访问
- **代码**：https://github.com/Instituto-IA-LNCC/manaca-1b-base
- **模型权重**：https://huggingface.co/menezesbruno/manaca-1b-base（CC BY 4.0）
- **关键超参**：steps=20000, global_batch=512, seq_len=4096, lr_peak=3e-4, lr_min=3e-5, β₁=0.9, β₂=0.999, weight_decay=0.1, grad_clip=1.0, seed=1234
- **其他公开**：raw training log、per-example prediction vectors、corrected tokenizer、评估 harness 版本锁定
