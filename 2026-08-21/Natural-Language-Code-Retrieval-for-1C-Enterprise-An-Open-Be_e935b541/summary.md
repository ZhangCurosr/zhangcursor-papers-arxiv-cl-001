---
title: "Natural-Language-Code-Retrieval-for-1C-Enterprise-An-Open-Be"
source: https://arxiv.org/pdf/2608.19957v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:08:15"
---

# 论文速读：Natural-Language-Code-Retrieval-for-1C-Enterprise-An-Open-Benchmark-and-Efficient-Bi-Encoder

## 一句话总结
本文针对俄罗斯企业软件开发平台1C:Enterprise（使用BSL语言和俄语查询），构建了首个开源代码检索基准测试（3,413个真实查询-代码对）和高效的领域自适应双编码器模型；通过LLM生成78万合成训练三元组并进行微调，模型在balanced macro nDCG@10上达到0.5992，显著超越通用多语言嵌入基线。

## 研究问题与动机
1. **领域空白**：1C:Enterprise是俄罗斯主流企业软件平台，其BSL代码混合西里尔字母关键字、俄语标识符和业务术语，但开源代码检索基础设施几乎不存在。
2. **通用模型不足**：现有通用多语言嵌入模型（如deepvk/USER2-base、google/embeddinggemma-300m）未针对俄语查询到BSL代码的匹配进行优化，未经领域适配的性能未知。
3. **评测基准缺失**：现有1C相关评测资源（如1C Code Bench、PRISM/GenLab-1C）聚焦于代码生成任务，缺乏针对检索任务的开源基准。
4. **低资源困境**：缺乏标注的查询-代码对数据，需依赖合成数据生成方案。

## 核心贡献（创新点）
1. **首个1C代码检索开源基准**：构建了PruhaNLP/1C-Ebench（3,413个真实论坛和代码片段查询-代码对），填补了该领域的评测空白，与已有的1C生成类基准（如1C Code Bench）形成互补定位。
2. **大规模合成训练数据生成流水线**：使用google/gemma-4-26B-A4B-it从公开GitHub代码库生成784,057个(query, d+, d-)三元组，并结合FAISS硬负样本挖掘，解决了低资源场景下的训练数据短缺问题。
3. **领域自适应双编码器模型**：基于deepvk/USER2-base微调得到PruhaNLP/USER2-1C-code，引入隐私感知tokenization（PII占位符映射到特殊token），以+0.106 balanced macro nDCG@10的提升显著超越未适配基线。
4. **Matryoshka Representation Learning工程化**：首次在1C检索场景中应用MRL，支持768→32七档维度截断，256维时保留99.9%检索质量同时降低约3倍存储与计算开销。
5. **严格的污染审计协议**：采用exact match与13-gram双层重叠检测，证明结果不受训练-基准污染影响（clean-only macro仍达0.6011）。

## 方法详解
**任务形式化**：将1C代码检索定义为单阶段密集闭集检索问题。给定查询q和文档集D={d₁,...,dₙ}，双编码器独立编码：
- e_q = Enc(q; search_query)
- e_d = Enc(d; search_document)
- score(q,d) = cos(e_q, e_d)

**模型架构**：基于deepvk/USER2-base（149M参数，ModernBERT/RuModernBERT-base架构），支持8,192 token上下文，768维输出向量，mean pooling + L2归一化 + 余弦相似度。

**训练数据构建**：
- 源数据：leongl/1c_github（3,038,637行公共代码），过滤后保留784,058个唯一文档
- 合成查询：google/gemma-4-26B-A4B-it生成俄语query
- 硬负样本挖掘：d⁻ = argmax_{d∈Top-20(q)\{d+}} cos(e_q, e_d)，使用deepvk/USER2-small + FAISS IndexFlatIP
- PII脱敏：规则引擎替换[EMAIL]/[PHONE]/[PATH]/[PERSON]等，影响0.106%单元格

**训练目标**：MatryoshkaLoss(CachedMNRL) = Σ_{m∈{768,512,384,256,128,64,32}} InfoNCE(e^{(m)}, negatives)

**隐私感知Tokenization补丁**：[PATH]/[PERSON]映射到[unused0]/[unused1]；email/phone/IP占位符使用特殊USER2 token；新token嵌入从对应子token均值初始化。

**评测协议**：
- Balanced-subset macro = ½(s_forum + s_fastcode)，等权两个检索子场景
- Query-weighted micro按查询量加权（forum 84.5%，fastcode 15.5%）
- 主要指标：nDCG@10、Recall@10、MRR@10

**超参数**：batch_size=256，mini_batch=32，epochs=3，steps=9,189，lr=2e-5，5% cosine warmup，max_length=8192，FP16，A100，~7.5h。

## 实验与结果
**数据集**：PruhaNLP/1C-Ebench（3,413对）
- forum子集：2,883个（84.5%），长问题、上下文、错误描述、内联代码
- fastcode子集：530个（15.5%），短查询、即用型代码片段

**评估基线**：deepvk/USER2-base、google/embeddinggemma-300m、deepvk/USER-bge-m3、ibm-granite/granite-embedding-311m-multilingual-r2、microsoft/harrier-oss-v1-270m、intfloat/multilingual-e5-base、ai-forever/sbert_large_nlu_ru、BM25Okapi(k₁=1.2, b=0.75)、RRF(BM25+dense)。

**主要结果（nDCG@10）**：
| 排名 | 模型 | forum | fastcode | macro | micro |
|------|------|-------|----------|-------|-------|
| 1 | **PruhaNLP/USER2-1C-code** | 0.4617 | **0.7366** | **0.5992** | **0.5044** |
| 2 | google/embeddinggemma-300m | 0.3720 | 0.7080 | 0.5404 | 0.4242 |
| 3 | deepvk/USER2-base | 0.3670 | 0.6190 | 0.4932 | 0.4061 |
| - | BM25Okapi | 0.2865 | 0.3334 | 0.3099 | 0.2938 |

**最强结果与提升**：
- PruhaNLP/USER2-1C-code达最佳balanced macro 0.5992，比deepvk/USER2-base提升+0.106（paired bootstrap 95% CI [0.087, 0.125], P<0.001），比google/embeddinggemma-300m提升+0.0588（95% CI [0.042, 0.075], P<0.001）
- BM25Okapi显著落后，表明纯词汇重叠不足以应对俄语→BSL查询
- 未调优RRF融合反而降低性能（从0.5992降至0.4300），归因于\b+分词器对BSL标识符和错误字符串的分割效果差

**污染敏感性分析**：移除所有被exact/13-gram标记的样本后，clean-only macro=0.6011，clean-only fastcode=0.7369（略高于原始0.7366），证明结果不能由训练-基准污染解释。

**MRL截断效果**：256维保留99.9%检索质量，索引大小降至33%，理论3倍加速（乘加运算减少）。

## 相关工作脉络
1. **CodeSearchNet [12]**：首个大规模多语言代码搜索基准（6种主流语言），但未覆盖1C:Enterprise或俄语查询。
2. **CoSQA [10]**：面向真实搜索场景的代码检索基准（20,000+ Web查询），针对主流编程语言，与1C领域无交集。
3. **CoIR [18]**：覆盖text-to-code、code-to-code混合检索的综合基准，但不包含1C/BSL领域。
4. **1C Code Bench [8]**：评估LLM生成BSL代码的能力（编译正确性指标），与本文检索任务互补——前者问"能否写出代码"，后者问"能否找到相关片段"。
5. **PRISM/GenLab-1C [7]**：executable BSL evaluation，同样侧重于代码生成而非检索。
6. **Dense Passage Retrieval [14]**：密集检索奠基性工作，本文沿用其双编码器架构和硬负样本挖掘策略。
7. **Matryoshka Representation Learning [15]**：支持多维度截断的表示学习，本文首次将其应用于1C领域代码检索。

## 局限性与未来方向
**论文自述局限**：
1. **Single-Gold Qrels**：每个查询仅有一个gold文档，manual check发现10%案例中存在多个相关片段，可能低估实际性能。
2. **合成数据验证不足**：仅用LLM-as-a-judge在300个样本上评分（Relevance=4.33, Naturalness=4.17, Clarity=4.37），缺乏人类专家标注。
3. **PII脱敏不彻底**：规则引擎无精确率/召回率报告，未做残留审计。
4. **实验覆盖有限**：未评估rerankers、多种子方差、hard-vs-random negatives消融、ANN indexes、大规模延迟。

**未来方向**：
- 引入multi-gold qrels、human query judging、FN denoising、调优hybrids/rerankers、targeted ablations。

## 研究启发与可借鉴点
1. **合成数据生成流水线可迁移**：用LLM从公开代码库生成query-d⁺-d⁻三元组+FAISS硬负样本挖掘的方案，可直接迁移至其他低资源领域代码检索任务。
2. **MRL的工程部署价值**：多维度截断支持精度-成本灵活权衡，256维保留99.9%质量同时降3倍开销，适合资源受限的生产环境。
3. **污染审计标准化**：exact match + N-gram重叠的双层审计方法可作为代码检索基准的默认实践，本文clean-only分析有效排除了结果解释的质疑。
4. **隐私感知tokenization策略**：将PII占位符映射到特殊token并从子token均值初始化，可推广至其他含敏感信息的领域数据。
5. **Balanced-subset macro设计**：forum/fastcode等权平均的指标设计思路，适用于子集规模差异大且业务重要性相当的多场景评测。

## 关键术语表
**1C:Enterprise**：俄罗斯主流企业软件开发平台，广泛用于ERP、CRM等企业内部系统开发。
**BSL (Business Languages Script)**：1C:Enterprise平台专用脚本语言，混合西里尔字母关键字、俄语标识符和业务术语。
**Dense Retrieval**：将查询和文档映射为连续向量，通过相似度打分排序的检索方式，与BM25等稀疏检索相对。
**Bi-Encoder**：查询和文档分别通过独立编码器编码后计算相似度的检索架构，推理时可预计算文档向量，效率高。
**Matryoshka Representation Learning (MRL)**：支持多维度截断的表示学习，前m维嵌入可独立使用且性能递减平缓。
**Cached Multiple Negatives Ranking Loss (MNRL)**：对比学习损失函数，利用批次内负样本和硬负样本优化检索模型。
**nDCG@k**：归一化折损累计增益，衡量前k个结果的排序质量，考虑相关文档的位置加权。
**Balanced-subset Macro**：对各子集指标等权平均，适用于子集规模差异大但业务重要性相当的场景。
**PII (Personally Identifiable Information)**：个人可识别信息，包括邮箱、电话、IP地址、路径、人名等。

## 可复现要素
- **数据集**：PruhaNLP/1C-Ebench（3,413对）和PruhaNLP/1C-Code-Train（784,057三元组）均已公开于Hugging Face
- **代码**：PruhaNLP/1C-RB评测harness已开源于GitHub
- **模型**：PruhaNLP/USER2-1C-code（768维，支持MRL截断至768/512/384/256
