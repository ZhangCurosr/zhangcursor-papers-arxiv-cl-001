---
title: "MoganBert-TR-A-Turkish-Encoder-Foundation-Model-Trained-from"
source: https://arxiv.org/pdf/2608.25768v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:43:16"
field: "低资源编码器语言模型"
keywords: ["encoder language model", "CLM MLM curriculum", "Turkish NLP", "embedding model", "pretraining objective", "model soup", "quality filtering"]
innovations: ["CLM→MLM课程训练在土耳其语上验证检索任务2.7-3.7倍提升", "分支退火以4.3%额外成本获得比模型汤更优的TrGLUE表现", "教师蒸馏fastText构建90倍加速的土耳其语质量分类器"]
benchmarks: ["TrGLUE", "TabiBench", "MTEB(Turkish)", "Turkish MS MARCO", "TR-MMLU", "FLORES-200"]
---

# 论文速读：MoganBert-TR-A-Turkish-Encoder-Foundation-Model-Trained-from

## 一句话总结
本文提出从头训练的土耳其语编码器基础模型 MoganBert-TR（149M 参数），采用 CLM→MLM 两阶段课程训练策略，在同等计算预算下较纯 MLM 在 Turkish MS MARCO 检索任务上提升 2.7–3.7×，同时配套产出嵌入模型 MoganBert-Embed 及高质量数据管道与分词器。

## 研究问题与动机
1. 现有土耳其语编码器（TabiBERT、ModernBERT-TR）仅迁移了 ModernBERT 架构，预训练目标仍固定为纯 MLM，从未系统验证训练目标对土耳其语的影响。
2. CLM→MLM 课程在英语规模已被验证有效，但土耳其语是形态丰富的黏着语，且拥有独立的分词器与语料库，该结论是否适用仍是开放问题。
3. 退火阶段（annealing phase）中，长上下文扩展与学习率衰减的交互机制缺乏度量，mmBERT 已证明低 LR 阶段数据混合比例关键，但上下文长度在此阶段的作用未被测量。
4. 土耳其语缺乏现成的文本质量分类器，难以直接复用 FineWeb 的英文质量过滤方案，需从头构建适应土耳其语的数据管道。

## 核心贡献（创新点）
1. **CLM→MLM 课程训练的土耳其语验证**：在相同架构与语料、等步数预算下，CLM→MLM 课程较纯 MLM 在 Turkish MS MARCO 检索上提升 2.7–3.7×，本质区别在于曲线任务揭示了嵌入几何的"rogue dimension"现象（纯 MLM 首主成分吸收 28.1% 方差，课程仅 11.9%）。
2. **分支退火（Branched Annealing）**：将长上下文扩展与学习率衰减在共享前缀后拆分为两条分支，最终 4,810 步以 1024 上下文运行衰减，较 8192 上下文分支在 TrGLUE 上提升 +0.49±0.26（p≈0.013），且比模型汤方案高 0.75 分，额外成本仅 ~4.3%。
3. **MoganBert-TR 基础模型**：149M 参数，TrGLUE 78.41（土耳其语 ModernBERT 系列最佳），TabiBench 77.73（单语土耳其语编码器第二，仅落后 ModernBERT-TR 0.19 分），在代码检索上领先 TabiBERT +3.62 分。
4. **MoganBert-Embed 嵌入模型**：通过教师蒸馏（Qwen3-Embedding-8B）与多信号对比微调产出，MTEB(Turkish) 总分 68.30，达 7.57B 教师模型的 99.5%，参数量仅为其 1/51。
5. **语言特异性数据管道与分词器**：构建蒸馏链将慢速但准确的 BERT 标签生成器转化为 fastText（94.4% 一致率，~90× 加速）；50,048 词元分词器在压缩率与生育率上超越所有对比土耳其语分词器，且在代码语料上实现 100% 无损往返。

## 方法详解
1. **CLM→MLM 课程训练**：总 237.3B 词元，前 36.0B（16.6%）使用因果语言建模（CLM），后 180.7B（76.2%）使用掩码语言建模（MLM，mask rate 30%），转换发生在 WSD（Whalen-Sidhu-Davis）调度稳定阶段，不触碰学习率；注意力窗口同步从因果模式切换为双向模式。
2. **分支退火设计**：共享前缀（10.5B 词元，上下文 8192，mask 10%，LR 上半段衰减）后拆为两分支：Branch A 在 8192 上下文继续衰减（10.1B 词元），Branch B 在 1024 上下文继续衰减（10.1B 词元）；LR 曲线（1-sqrt，8e-4 → 8.36e-6）按 9,810 步总计算，分支仅改变最后 4,810 步的上下文长度。
3. **数据管道**：三源整合——FineWeb2 土耳其语子集（重新通过土耳其语质量过滤器）、近期 Common Crawl 月抓取的字节范围 HTTP Range GET 提取、视觉语言模型从印刷/机构源（书籍、论文、法律文本）提取领域密集文本；单 Pass 完成语言识别（fastText lid.176.ftz）、Gopher/C4 启发式过滤、土耳其语 boilerplate 检查、PII 掩码与学习质量分类；MinHash 模糊去重（n-gram=5, 14 buckets, 8 hashes/bucket）。
4. **质量分类器蒸馏链**：fastText（off-the-shelf 词向量）失败（丢弃合法短文本）→ LLM 标注的 ~6,000 文档重训练 fastText（~54% 准确率）→ 土耳其语 BERT 微调达到 70.7% exact / 93.9% ±1 准确率但仅 ~96.5 doc/s → 用 BERT 作为标签生成器，在 ~480,000 条 keep/discard 决策上训练 fastText，达成 94.4% 一致率与 ~8,880 doc/s（~90× 加速）。
5. **分词器设计**：50,048 词元 SentencePiece Unigram 模型（22层×768 hidden）；代码分支保留缩进（allow_whitespace_only_pieces=True，转义换行为\t），避免不同程序坍缩为同一 token 序列；[MASK] 置于 control_symbols 而非 tokens，防止网页中文字"[MASK]"被误转为掩码标识符；64K 词表虽效率更高但嵌入参数增加 ~10.7M（占 150M 模型的 ~7%），故选择 50K 为基准。
6. **嵌入模型两阶段训练**：Phase 1 教师蒸馏（Qwen3-Embedding-8B，dim=3072）：L = 1.0·L_distill + 0.05·L_GOR，cos_raw 从 0.9841 降至 0.0851；Phase 2 多信号对比微调，v1 仅检索对，v2 引入显式负样本（contradiction hypothesis、低分句对、相反极性情感句）并添加 false-negative masking 与 anchor decay，v3 按信号类型分别处理（STS 用 CoSENT、NLI 设 contradiction 为 hard negative/neutral 为 soft negative、同类 mask）。
7. **模型汤（Model Soup）**：Phase1 + v3 + v4 加权平均（权重 0.4/1/1），Phase1 占比 ~16.7%，MTEB(Turkish) 总分 68.30。

## 实验与结果
- **数据集**：237.3B 词元土耳其语混合语料（~73% 土耳其语、~17% 英语、~10% 代码），包含 FineWeb2、Common Crawl 近期抓取、教育/代码/数学/法律/书籍等域；独立测试集含 TR-MMLU（6,200 题）、Turkish Wikipedia（30K 文档）、FLORES-200（2,009 句）、18 语言 540 文件代码语料。
- **评估基线**：TrGLUE（8 任务，五种子协议）、TabiBench（28 任务，8 类别，单种子）、MTEB(Turkish)（26 任务）、Turkish MS MARCO 检索（无对比训练，raw pretraining representations）。
- **核心结果**：
  - CLM→MLM vs 纯 MLM 消融：线性探针平均仅 +0.64 分（seed 噪声范围），但 Turkish MS MARCO R@1 从 2.96 → 10.86（×3.67），R@10 从 9.92 → 26.34（×2.66），MRR 从 5.59 → 16.23（×2.90）。
  - 分支退火：anneal1k（1024 上下文衰减）TrGLUE 78.41±0.32，较 anneal（8192）77.92±0.48 提升 +0.49±0.26（p≈0.013），较模型汤 77.66 高 0.75 分。
  - TrGLUE：MoganBert-TR 78.41（ModernBERT 系列土耳其语最佳），较 BERTurk（79.87）低 1.46 分，差距集中于 CoLA（37.63 vs 41.71）与 STS-B（69.59 vs 72.08）。
  - TabiBench：77.73，单语土耳其语编码器第二，代码检索 60.57 领先 TabiBERT +3.62。
  - MTEB(Turkish)：MoganBert-Embed 68.30，学生模型第一，达 7.57B 教师模型 68.66 的 99.5%（参数量 51× 更小）。
- **最强结果与提升**：MS MARCO 检索 R@1 提升 2.7–3.7×；TrGLUE 较同架构纯 MLM 消融不可见，但检索任务揭示显著差距；嵌入模型 cos_raw 从 0.9841 降至 0.0851，零样本 IR NDCG@10 从 0.2361 升至 0.5927。

## 相关工作脉络
1. **BERTurk [21]**：首个从头训练的土耳其语单语 BERT，长期作为事实标准，但未公开完整语料与预处理细节；本文 MoganBert-TR 在 TrGLUE 上落后 1.46 分，差距源于 NSP-free 架构的已知弱点（CoLA、STS-B）。
2. **TabiBERT [26] / ModernBERT-TR [1]**：首个将 ModernBERT 架构迁移至土耳其语的工作，使用纯 MLM 目标；本文定位差异在于系统验证了训练目标对土耳其语的影响，并证明目标选择可带来检索任务的质变。
3. **Gisserot-Boukhlef et al. [7]**：在英语规模（38 模型，210M–1B，15,000+ fine-tuning runs）证明 CLM→MLM 课程固定计算预算下优于纯 MLM；本文将其结论推广至形态学丰富的土耳其语，并揭示"rogue dimension"几何机制。
4. **mmBERT [13]**：证明退火阶段数据混合比例对多语言模型关键；本文延伸测量了上下文长度在衰减阶段的作用，提出分支退火。
5. **FineWeb/FineWeb2 [18,19]**、**CCNet [29]**：建立 Web 文本提取与质量过滤管道范式；本文适配土耳其语时需从头构建质量分类器（无 off-the-shelf 可用），提出教师蒸馏到 fastText 的链式方案。
6. **Timkey & van Schijndel [25]（rogue dimension）**：指出 MLM 预训练编码器表征各向异性问题；本文通过 CLM 阶段缓解该现象（方差吸收从 28.1% 降至 11.9%），并在嵌入模型中用 GOR 进一步修正。

## 局限性与未来方向
1. **目标消融单一种子**：CLM→MLM vs 纯 MLM 消融仅在 10K 步短跑中进行，全量预训练规模下的优势需单独验证；检索腿仅基于单一数据集（MS MARCO），两个探针任务存在饱和。
2. **超参数未完全验证**：全量预训练的 CLM 比例（16.6%）与消融验证的 25% 不一致，未做控制消融；"纯 1024 退火"分支也未被测量。
3. **模型汤过拟合风险**：汤组件为单种子运行，权重在 MTEB 上扫描，虽曲线平坦（0.10–0.25 区间）降低风险但未消除。
4. **TabiBench 单种子限制**：仅 MoganBert-TR 运行了完整 28 任务协议，参考分数来自不同环境；0.19 分差距（vs ModernBERT-TR）与 0.15 分差距（vs TabiBERT）不具统计显著性。
5. **嵌入模型检索弱点**：Phase 2 为平衡多信号降低检索信号占比，导致检索成为相对短板（仅 11 个检索任务中赢 3 个）。
6. **情感对比数据未经人工验证**：40,737 条情感对比三元组由 LLM 生成，未做人工校验。
7. **数据无法完整公开**：因许可限制，语料无法完整发布，仅公开管道代码与过滤决策。

## 研究启发与可借鉴点
1. **CLM→MLM 课程对低资源/形态丰富语言的可迁移性**：本文证明该策略在土耳其语（黏着语）上同样有效且机制可解释（嵌入几何改善），可推广至其他非英语低资源语言（如哈萨克语、阿塞拜疆语等突厥语族语言）。
2. **分支退火作为低成本探索退火设计的范式**：相较模型汤需完整独立训练，分支退火仅多计算 ~4.3% 即可比较不同上下文策略，为长上下文扩展的实验设计提供高效替代方案。
3. **教师蒸馏构建语言特异性质量分类器**：无 off-the-shelf 质量分类器时，用慢速高精度 BERT 作为标签生成器再蒸馏到 fastText 的链式方案，可复用于其他缺少数据过滤工具的语言。
4. **代码感知分词器设计**：保留缩进的代码分支（escape 换行为 \t，allow_whitespace_only_pieces=True）实现 100% 无损往返，该技巧可直接迁移至多语言代码-自然语言混合预训练场景。
5. **小样本内部指标的误导性警示**：模型汤在 fill-mask probe（50 句）上胜出但最终 TrGLUE 排名第五，提示选择检查点时应避免依赖小样本内部指标，需结合下游任务验证。

## 关键术语表
**CLM→MLM 课程训练**：先以因果语言建模（CLM）训练部分步数，再切换为掩码语言建模（MLM）完成剩余训练，切换发生在 WSD 调度稳定阶段。
**Rogue Dimension（异常维度）**：MLM 预训练编码器表征中某单一方向吸收大量方差（本文纯 MLM 为 28.1%），导致余弦相似度判别信号被淹没。
**Branched Annealing（分支退火）**：在共享前缀后将长上下文扩展与学习率衰减拆分为两条并行分支，仅最后阶段计算不同。
**WSD Schedule**：Whalen-Sidhu-Davis 调度，一种包含 warmup、stable、decay 三阶段的训练调度，课程切换发生在 stable 阶段。
**Cosine Retrieval**：直接使用预训练表征的原始余弦相似度进行检索，不经过对比微调，用于暴露嵌入几何缺陷。
**Model Soup（模型汤）**：对多个同起点但不同微调/退火的模型权重进行加权平均，以提升泛化性能。
**GOR（Global Orthogonal Regularization）**：全局正交正则化，用于修正嵌入空间各向异性，迫使表征分布更均匀。
**CoSENT Loss**：用于语义文本相似度的排序损失，基于余弦相似度对满足 s_i > s_j 的句对排序。

## 可复现要素
- **数据集**：语料因许可限制无法完整公开；管道代码与过滤决策将开源；独立测试集（TR-MMLU、Wikipedia、FLORES-200）为公开资源。
- **代码/权重**：模型权重、分词器、嵌入模型与评估代码将在预印本发表时于 https://huggingface.co/moganai 开源。
- **关键超参**：22 层、hidden 768、local attention window 128、vocab 50,048、peak LR 8e-4、global batch 2,097,152 tokens/step、optimizer StableAdamW（betas 0.90/0.98, eps 1e-6, weight decay 1e-5）、precision bf16、hardware 4×H100 (DDP)、MFU 0.248、total tokens 237.3B。
