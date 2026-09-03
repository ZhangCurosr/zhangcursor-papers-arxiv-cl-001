---
title: "TSWAP-A-Multilingual-Retrieval-Augmented-Thai-Wellness-Advis"
source: https://arxiv.org/pdf/2608.22917v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:02:04"
field: "多语言健康RAG系统"
keywords: ["retrieval-augmented generation", "Thai NLP", "traditional medicine", "health chatbot", "multilingual LLM", "RAG safety"]
innovations: ["强制检索路由防止静默幻觉", "英文校准量化对泰语声调的破坏效应", "首个泰国传统医学检索基准"]
benchmarks: ["50-question Thai retrieval benchmark", "production QA campaign (429 cases)"]
---

# 论文速读：TSWAP-A-Multilingual-Retrieval-Augmented-Thai-Wellness-Advis

## 一句话总结
TSWAP 是一个已部署的多语言检索增强生成（RAG）泰国健康养生顾问系统，通过未微调的开源 LLM（Qwen3.6-35B-A3B）结合 30.6K chunk 的泰国传统医学知识库，为8种语言提供实体查询、养生建议和_provider推荐_服务，同时强制检索路由避免幻觉并内置安全过滤层。

## 研究问题与动机
- **信息碎片化问题**：泰国产值405亿美元的养生经济市场中，泰国传统医学和养生素材提供商信息分散在多渠道，缺乏可信、可验证、多语言的中心化来源，外国游客面临语言障碍和标准不确定。
- **领域空白**：泰国NLP领域已有多个通用大模型（Typhoon、OpenThaiGPT等），但**没有公开的泰国传统医学/草药检索基准或数据集**；唯一相关工作 AppHerb 是微调模型、无检索、无多语言、无部署。
- **安全与幻觉风险**：直接使用LLM进行健康咨询可能导致剂量建议（违规）、越界请求（如写代码、讨论政治）、幻觉性provider推荐等问题，需要强制检索+规则安全层。
- **低资源语言部署挑战**：英文校准的量化模型可能破坏泰语声调符号，隐式推理预算截断导致答案不完整。

## 核心贡献（创新点）
1. **首个泰国传统医学/养生检索基准**：发布50题泰语检索黄金集合（含gold document IDs，Recall@5=0.88），填补泰国草药领域基准空白。
2. **"检索优先"路由设计**：通过首轮查询分类器强制实体查询走工具调用（tool_choice=forced），防止短名词短语查询时模型静默跳过检索。
3. **生产级安全探针量化**：在71个真实问题上的消融实验证明：无安全提示时模型给出完整对乙酰氨基酚剂量方案（严重违规），有无KB时无法提供任何可验证provider推荐。
4. **两个可迁移的部署发现**：英文校准4-bit AWQ量化会破坏泰语元音/声调标记；强制检索路由对可靠grounding是必要的。

## 方法详解
**整体架构**：未微调的 Qwen3.6-35B-A3B（MoE，35B总参/3B激活）+ vLLM自托管 + Milvus向量库（2.6.4）+ OpenAI兼容端点。

**检索流水线**：
- **混合检索**：BAAI/bge-m3（1024-d多语言embedding）+ BM25稀疏检索，Reciprocal Rank Fusion融合，10候选→cross-encoder（Qwen/Qwen3-Reranker-8B）重排→top_k=3
- **结构化过滤**：省份、食品类型、价格范围schema规范化 + 地理空间过滤（WKT GEOMETRY字段+RTREE索引+ST_DWITHIN半径查询）
- **检索阈值**：top相似>0.85时跳过rerank

**强制检索路由**：
- 首轮LLM分类器将查询分为5类意图：wellness lookup / general advice / emergency / out-of-scope / small talk
- wellness lookup类强制tool_choice=rag_search，保证任何实体查找（草药、provider、spa、套餐等）先检索后回答
- system prompt约束：仅推荐检索结果中存在的项，不编造实体，说"no match found"而非虚构

**多语言策略**：
- 8种语言（泰、英、中、日、马来、俄、韩、印地语）使用单一多语言模型+单英文system prompt
- translate-then-retrieve：模型先将查询译为泰语再调用rag_search，然后用用户语言回答
- zero-shot服务，无per-language模板

**安全层**：
- **Scope限制**：仅限养生话题（泰国草药、营养、生活方式、睡眠、运动、压力、provider、套餐、养生旅游）
- **医疗安全**："你不是医生"——不诊断、不处方、不给药物剂量；引导咨询合格专业人士
- **紧急路由**：胸痛/呼吸困难/大出血/中风/昏厥→立即拨打1669；自残/自杀意念→1323热线
- **质量守卫**：post-generation Thai quality guard检测泰语损坏（combining-mark启发式：mark-to-consonant ratio>0.28触发重新采样）

## 实验与结果
**检索基准**：
- 50题泰语检索黄金集，Recall@5=0.88（44/50）
- 弱项：thai_herbs和packages集合~0.75

**部署评估**：
- UAT（n=120）：满意度87.2%，用户自评准确性86.5%，泰药知识准确性90.2%，provider信任度91.0%
- 负载测试：200并发用户0%错误，平台API延迟~527ms，LLM生成中位数14.1s/答案
- 安全评估：通过全部55个OWASP Top-10 (2025)用例

**生产QA活动**：
- 429案例回归测试（6组），第二轮259案例重测：91.1%复现pass（236/259），21 odd，2 fail
- 地理空间搜索组163/181 pass（占问题总数47/56=84%失败案例在此组）

**No-retrieval探针（71题）**：
- 无安全提示时：对剂量问题给出完整对乙酰氨基酚方案（含儿科剂量），3/22明显安全违规
- +安全提示后：0/22违规，剂量请求被拒绝并转介药剂师
- 紧急号码突出性：无提示时1669出现在第2354字符，有提示时在第141字符
- 无KB时：12个地理空间查询中**0个产生可验证provider推荐**，8/12退回知名连锁品牌（无地址/认证）

## 相关工作脉络
- **Typhoon/OpenThaiGPT/SEA-LION/Sailor/Chinda**：通用泰语/SEA LLM，未针对传统医学优化，TSWAP选择grounding而非预训练路线
- **AppHerb**：唯一泰国传统医学LLM（Gemma-2微调），但无检索、无多语言、无部署、无基准
- **Eir/泰国国家医师执照考试评估**：临床NLP方向，关注的是现代医学任务而非传统草药/养生
- **MTCMB/TCM-Ladder**：中医基准测试，作为泰国草药基准设计的参考模板
- **HealthBench/HEALTH-PARIKSHA**：多语言健康对话评估方法，启发TSWAP的答案质量评估协议

## 局限性与未来方向
- 多语言质量继承自预训练，次要语言（马来/俄/韩）可能存在退化风险
- 安全层无训练分类器或相互作用检查器，未做对抗性红队测试
- 检索在thai_herbs集合上最弱，地理空间搜索依赖口语化表达易失败
- 答案跨轮次非完全可复现（采样非确定性）
- 评估仅覆盖单一部署场景，缺少引擎侧消融和多语言答案质量对比
- 未来方向：完整答案质量基准、引擎侧消融、FP8量化迁移、thai_herbs检索改进、结构化红队测试

## 研究启发与可借鉴点
1. **强制检索路由设计**：对实体查找类查询使用tool_choice强制检索，可有效防止"静默幻觉"——这对所有工具调用型RAG系统均有参考价值。
2. **翻译-检索-回答策略**：对于知识库为单一语言（泰语）但服务多语言用户的场景，translate-then-retrieve比多语言embedding更有效，可迁移至其他低资源语言RAG系统。
3. **量化敏感度验证**：英文校准的AWQ量化破坏泰语声调，提醒多语言部署需验证量化对目标语言的token级别影响。
4. **生产级安全探针**：通过禁用grounding对比有/无安全提示的差异，量化安全层的实际贡献，为安全机制设计提供实证依据。
5. **地理空间检索的schema设计**：WKT GEOMETRY字段+RTREE索引+ST_DWITHIN半径查询的组合，为位置敏感的健康服务推荐提供了可复用架构。

## 关键术语表
- **RAG（Retrieval-Augmented Generation）**：检索增强生成，将外部知识库检索结果作为上下文输入LLM以减少幻觉
- **tool_choice强制路由**：通过query classifier将特定意图映射到强制工具调用，避免模型静默跳过检索
- **bge-m3**：BAAI的多语言embedding模型（1024维），支持密集检索
- **AWQ量化**：Activation-aware Weight Quantization，一种LLM量化方法，英文校准版本可能破坏泰语声调
- **recall@k**：检索命中率指标，K=5时达0.88表示50题中44题在top5检索结果中找到gold document
- **cross-encoder reranking**：使用Qwen3-Reranker-8B对粗排结果重新排序，相似度>0.85时跳过以节省计算
- **translate-then-retrieve**：将非泰语查询先翻译为泰语再检索，利用bge-m3的多语言能力提供跨语言容忍度

## 可复现要素
- **数据集**：50题泰语检索黄金集已公开（含gold document IDs）；429案例生产QA日志已发布；71题frontier-probe日志已发布
- **代码/权重**：模型Qwen3.6-35B-A3B为开源社区权重；检索基准harness已发布；知识库本身未公开
- **关键超参**：chunk size=512 tokens，overlap=50 tokens；BM25 drop_ratio_search=0.1；dense ANN cosine similarity；rerank阈值>0.85跳过；top_k=3；泰语质量守卫mark-to-consonant ratio阈值=0.28
