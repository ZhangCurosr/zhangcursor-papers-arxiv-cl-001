---
title: "MoganBert-TR-A-Turkish-Encoder-Foundation-Model-Trained-from"
source: https://arxiv.org/pdf/2608.25768v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:43:11"
field: "低资源语言预训练"
keywords: ["encoder language models", "Turkish NLP", "pretraining objectives", "CLM-MLM curriculum", "embedding models", "quality filtering", "tokenizer design"]
innovations: ["CLM→MLM两阶段课程学习在土耳其语编码器中的控制消融验证", "分支退火策略以4.3%额外成本超越模型汤", "教师蒸馏质量分类器管线解决低资源语言过滤缺口"]
benchmarks: ["TrGLUE", "TabiBench", "MTEB(Turkish)", "Turkish MS MARCO"]
---

# 论文速读：MoganBert-TR-A-Turkish-Encoder-Foundation-Model-Trained-from

## 一句话总结
本文从零开始训练了149M参数的土耳其语编码器基础模型MoganBert-TR，首次系统性验证了CLM→MLM两阶段课程学习对土耳其语（黏着语）的有效性，并提出分支退火策略与教师蒸馏质量过滤管线，最终模型在TrGLUE上取得78.41分（Turkish ModernBERT系列最佳），派生嵌入模型MoganBert-Embed以51倍参数差距达到教师模型99.5%的性能。

## 研究问题与动机
1. **预训练目标长期固定**：土耳其语编码器（BERTurk、TabiBERT、ModernBERT-TR）均采用纯MLM目标，未对训练目标本身进行系统性探索。
2. **CLM→MLM课程的有效性存疑**：英语尺度控制实验已证明该课程优于纯MLM，但黏着语（morphologically rich, agglutinative）在自有tokenizer与语料下是否成立尚未验证。
3. **长上下文退火策略缺乏规范**：mmBERT证明衰减阶段数据混合比例至关重要，但同阶段上下文长度（context length）的作用未被测量。
4. **低资源语言质量过滤缺口**：土耳其语缺乏现成质量分类器，需从 scratch 构建高效过滤管线。

## 核心贡献（创新点）
1. **CLM→MLM课程学习的土耳其语控制消融**：同等步骤预算下，CLM 25%+MLM 75%在Turkish MS MARCO检索上超越纯MLM 2.7–3.7×；机制解释为嵌入几何优化（纯MLM单一方向吸收28.1%方差 vs 课程11.9%）。
2. **分支退火（Branched annealing）策略**：共享前缀后将长上下文扩展与LR衰减分为两支，在1024上下文运行最终衰减阶段较模型汤（model soup）方案以仅4.3%额外成本提升TrGLUE 0.75分。
3. **149M参数MoganBert-TR达TrGLUE 78.41**：为所比较土耳其ModernBERT模型中最高分；TabiBench 77.73分位列单语模型第二，代码检索类别超TabiBERT +3.62分。
4. **MoganBert-Embed嵌入模型**：通过教师蒸馏+多信号对比微调，以149M参数达到7.57B参数教师模型99.5%性能，MTEB(Turkish)学生模型第一（68.30分）。
5. **语言特异性数据管线与tokenizer**：提出"教师蒸馏→fastText"两步质量分类器（94.4%一致率、90×加速）；50,048词表tokenizer在压缩率与 fertility 上超越所有对比土耳其tokenizer。

## 方法详解
**架构设计**：遵循ModernBERT-base配置（22层、hidden 768、局部窗口128、RoPE、GLU），共149.4M参数；重写注意力路径以支持CLM阶段（需处理因果局部窗口、边界感知packing、position-id重置、文档边界masking四个静默失败陷阱）。

**CLM→MLM课程学习**：总237.3B tokens，前36.0B（16.6%）使用CLM目标，后续180.7B使用MLM目标；转换发生在WSD调度稳定期且不动learning rate；掩码率从30%降至10%；注意力窗口从causal切换为bidirectional。

**分支退火设计**：共享前缀10.5B tokens（ctx 8192、mask 10%、LR上半段衰减）后分为两支：Branch A在8192上下文完成下半段衰减；Branch B在1024上下文完成同等衰减；最终选择anneal1k分支作为主模型。

**质量分类器蒸馏**：LLM标注→Turkish BERT微调（70.7% exact acc）→作为label generator生成480K标签→蒸馏至fastText（8,880 docs/sec，94.4% agreement）。

**Tokenizer设计**：SentencePiece Unigram，50,048词表（64的倍数）；保留代码缩进的预处理管道；特殊token布局避免[MASK]被字面文本污染。

## 实验与结果
**数据集**：
- 主语料：FineWeb2土耳其部分 + Common Crawl最近月份（HTTP Range GET增量获取）+ VLM提取的印刷/机构文档
- 混合比例：73%土耳其语、17%英语、10%代码
- 去重：MinHash模糊去重（n-gram 5, 14 buckets, 8 hashes）

**评估基准**：
- TrGLUE（5-seed协议，八任务平均）：MoganBert-TR **78.41±0.32**，超TabiBERT 77.83、ModernBERT-TR 77.64
- TabiBench（28任务，单seed）：77.73分，代码检索60.57分（超TabiBERT +3.62）
- MTEB(Turkish)（26任务）：MoganBert-Embed **68.30**，学生模型第一

**关键对比**：
| 任务 | Pure MLM | CLM→MLM | 提升倍数 |
|------|----------|---------|---------|
| MS MARCO R@1 | 2.96% | 10.86% | ×3.67 |
| MS MARCO MRR | 5.59% | 16.23% | ×2.90 |
| 嵌入方差首分量 | 28.1% | 11.9% | -57.7% |

## 相关工作脉络
1. **Gisserot-Boukhlef et al. [7]**：英语尺度38模型控制实验证明CLM→MLM课程优于纯MLM；本文将其验证于土耳其语黏着语场景。
2. **mmBERT [13]**：发现衰减阶段数据混合比例关键；本文补充测量上下文长度在同阶段的作用。
3. **TabiBERT [26]**：首个将ModernBERT架构带至土耳其语的工作；本文在其基础上探索预训练目标而非仅架构迁移。
4. **BERTurk [21]**：长期作为土耳其语事实标准；其CoLA/STS-B优势源于NSP损失对[CLS]的直接梯度，ModernBERT移除NSP后形成架构性劣势。
5. **Timkey & van Schijndel [25]**：提出"rogue dimension"概念解释嵌入各向异性；本文以此解释纯MLM检索性能劣势。
6. **Qwen3-Embedding-8B [20]**：作为7.57B参数教师模型用于蒸馏；本文学生模型仅149M参数（51×更小）达到其99.5%性能。

## 局限性与未来方向
1. **消融实验规模有限**：CLM→MLM消融仅10K步骤单seed，需全量预训练尺度验证；CLM比例（16.6%）偏离消融最优值（25%）但未验证。
2. **TabiBench单seed限制**：与ModernBERT-TR的0.19分差距未达统计显著性；参考分数来自不同环境，存在环境偏差。
3. **情感对比数据未经人工验证**：Phase 2使用的40,737条LLM生成数据缺乏人工校验。
4. **单一教师模型**：蒸馏仅使用Qwen3-Embedding-8B，未探索多教师集成。
5. **未来方向**：检索Gap可通过增加检索信号比例或独立soup组件弥补；计划扩展至~400M参数大模型及decoder变体；需分解代码检索优势的tokenizer/混合比例/目标三因素贡献。

## 研究启发与可借鉴点
1. **课程学习目标作为独立杠杆**：在固定架构与计算预算下，训练目标本身可产生质性差异；警告编码器比较不应仅依赖GLUE类benchmark。
2. **分支退火替代模型汤**：共享前缀后分叉只需~4.3%额外成本，比完整模型汤更经济且效果更好；为退火阶段设计提供新思路。
3. **嵌入几何诊断取代内部探针**：Fill-mask probe误导性地偏好model soup，而余弦检索揭示真实差距；提示small-sample内部指标不可靠。
4. **教师蒸馏构建质量过滤器**：无现成分类器的低资源语言可通过"慢但准的教师→快学生"蒸馏模式解决；90×加速经验可迁移。
5. **黏着语masking策略的上界分析**：通过语料统计（95%词型为多piece但仅41% token流涉及）论证WWM与random masking等价，避免无谓消融。

## 关键术语表
**CLM→MLM Curriculum**：先使用因果语言建模（左至右预测）训练部分步骤，再切换至掩码语言建模的阶段性训练策略。
**Rogue Dimension**：嵌入空间中单一方向吸收过高分方差（>25%），导致余弦相似性判别信号被淹没的几何病态。
**Branched Annealing**：在共享训练前缀后，将不同上下文长度（8192 vs 1024）的衰减阶段分成两支并行执行的退火策略。
**WSD Schedule**：Weight-Scheduled Decay，一种包含warmup、stable、decay阶段的训练调度方案。
**Fertility**：Tokenizer效率指标，定义为字符数/token数，越低表示压缩率越高。
**Model Soup**：对多个checkpoint的权重进行加权平均以融合多样性的集成技术。
**GOR (Global Orthogonal Regularization)**：全局正交正则化，用于缓解嵌入各向异性的正则化项。
**InfoNCE**：基于InfoMax原理的对比学习损失函数，用于训练嵌入模型。

## 可复现要素
- **数据集**：部分公开（Filtering pipeline code开源），主语料因 licensing restrictions 无法完整发布
- **代码/权重**：模型权重、tokenizer、embedding model、evaluation code均开源（https://huggingface.co/moganai）
- **关键超参**：149.4M参数、50,048词表、237.3B tokens、peak LR 8e-4（WSD）、global batch 2,097,152 tokens、4×H100、bf16、StableAdamW
- **硬件**：4× H100 (DDP)，throughput 836K tokens/sec，MFU 0.248
