---
title: "TSWAP-A-Multilingual-Retrieval-Augmented-Thai-Wellness-Advis"
source: https://arxiv.org/pdf/2608.22917v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:02:06"
field: "低资源多语言检索增强生成"
keywords: ["retrieval-augmented generation", "Thai NLP", "traditional medicine", "multilingual LLM", "RAG benchmark", "health chatbot", "LLM safety"]
innovations: ["强制查询分类+tool_choice检索路由防止静默幻觉", "首个泰式传统医学/养生检索基准(50题, Recall@5=0.88)并开源生产QA日志", "量化多语言部署发现: English-calibrated 4-bit AWQ破坏泰语音调标记、translate-then-retrieve零样本八语服务"]
benchmarks: ["TSWAP 50-question Thai retrieval golden set", "Production QA campaign (259 cases, 91.1% test-retest pass)", "Frontier no-retrieval probe (71 questions, Gemini 2.5 Flash)"]
---

# 论文速读：TSWAP-A-Multilingual-Retrieval-Augmented-Thai-Wellness-Advis

## 一句话总结
TSWAP 是一个已部署的多语言泰国传统医学与养生咨询系统，基于未微调的开源 LLM（Qwen3.6-35B-A3B）搭配混合检索与强制路由机制，以检索增强生成方式提供经过验证的知识库问答。论文同时发布了首个泰式传统医学/养生检索基准（50 题，Recall@5 = 0.88）和生产级 QA 日志。

## 研究问题与动机
- **泰国传统医学/养生信息碎片化**：泰国养生旅游市场规模达 405 亿美元（2023），但服务机构、价格、认证状态、草药知识等信息分散在不同渠道，缺乏可信的多语言统一来源。
- **语言障碍**：外国游客面临泰语门槛与服务商标准不确定性，现有资源缺少面向国际用户的多语言 AI 辅助渠道。
- **NLP 社区空白**：现有泰语 LLM（Typhoon、OpenThaiGPT、SEA-LION、Sailor、Chinda 等）均为通用模型；仅有一个传统医学模型 AppHerb（Gemma-2 微调，无部署、无多语言、无检索）。
- **缺乏评估基准**：与中国传统医学已有的 MTCMB、TCM-Ladder 等基准相比，泰语/东南亚传统医学领域几乎没有公开的检索评测基准与部署验证。

## 核心贡献（创新点）
1. **首个公开的泰式传统医学/养生检索基准**：50 个带 gold document ID 的泰语问题，Recall@5 = 0.88，并配套前后对比回归护栏。与已有工作本质区别在于填补了泰语传统医学公开评测数据集空白。
2. **Grounding-first RAG 配方**：引入首轮查询分类器强制实体类查询走检索工具（tool_choice 路由），防止模型对简短名词短语静默跳过检索而幻觉；这一强制路由是防止静默幻觉的关键。
3. **生产级安全性探针（frontier no-retrieval probe）**：在未检索条件下测试 Gemini 系列后台模型，量化安全提示与安全路由对拒绝超范围请求、剂量推荐和紧急情况置顶的有效性。
4. **可迁移部署发现**：English-calibrated 4-bit AWQ 量化会破坏泰语音调标记输出；强制检索路由对可靠 grounding 必要；这些发现可迁移到其他低资源语言的工具调用型 RAG 系统。
5. **完整可复现发布**：开源 50 题检索金集、生产 QA 日志（259 案例）、前端探针日志，但未公开知识库本体。

## 方法详解
- **知识库构建**：基于 Milvus v2.6.4 单一向量库，8 个 collection（restaurants 11,212 / hotels 8,704 / attractions 8,169 / packages 1,153 / thai_herbs 870 / providers 484 / staygold_catalogs 18 / wellness_facilities 15），共 30,625 chunks。草药条目为结构化 monograph，含 identity/content/safety/evidence/metadata 字段；引用来自泰国公共卫生部传统医学司（DTAM）。
- **分块策略**：按字符切分，单 chunk 最多 512 tokens，重叠 50 tokens。
- **检索流水线**：混合 dense–sparse：dense 使用 BAAI/bge-m3（1024-d 多语言），sparse 使用 BM25（drop_ratio_search=0.1）；两路经 Reciprocal Rank Fusion 融合；取 top-7 后由 cross-encoder（Qwen/Qwen3-Reranker-8B）重排；top similarity ≤ 0.85 时跳过重排；最终 top_k = 3。
- **结构化过滤与地理空间**：按省份、食物类型、价格区间做 schema 归一化过滤；可选 WKT GEOMETRY 字段 + RTREE 索引支持 ST_DWITHIN 半径查询，near 参数在服务端解析坐标。
- **基础模型与服务**：未微调 Qwen3.6-35B-A3B（MoE，35B 总参/3B 活跃），vLLM 自托管，OpenAI-compatible endpoint；生产关闭 thinking 模式；部署采用社区 4-bit AWQ checkpoint（计划迁移到官方 FP8）。
- **强制检索路由**：首轮 LLM 查询分类器识别五类意图：wellness lookup / general advice / emergency / out-of-scope / small talk；wellness lookup 强制 tool_choice 调用 rag_search；系统 prompt 约束只推荐检索到的实体、禁止编造、无法匹配则返回 "no match found"；纯生活方式建议无需检索；长寿建议变体要求在输出末尾附 3–5 条 KB 引用。
- **多语言策略**：八语种（泰/英/中/日/马来/俄/韩/印地）共享单一多语言模型与英文 system prompt；query 先翻译为泰语再检索（translate-then-retrieve），bge-m3 多语言能力提供额外跨语言容错；模型以用户最近消息语言回复；全零样本，无 per-language template。
- **安全层（规则+路由）**：
  - Scope：仅限养生话题（草药、营养、作息、运动、压力、服务商、套餐、养生旅游），其他礼貌拒答并重定向。
  - Medical safety："You are NOT a doctor"，不做诊断/处方/用药剂量；以 general wellness 框架提供建议；对儿童、孕/哺乳期女性、老人、慢病或服药人群建议咨询合格专业人士。
  - Emergencies：胸痛、呼吸困难、严重出血、中风征兆、昏厥→立即呼叫 1669；自残/自杀意念→ empathetic response + 热线 1323；紧急情况优先回答，绕过检索。
  - Additive grounding：安全规则不替代检索，防止用安全文本逃避 grounding。
- **质量护栏**：服务端 post-generation 泰语质量检测（ combining-mark heuristic：mark-to-consonant ratio 阈值 0.28 加两个硬信号），检测到损坏则用官方 instruct-mode sampling 再生成一次；无 post-generation safety classifier/regex/blocklist。
- **知识库来源与合规**：PDPA 合规，仅转发 wellness-relevant 最小化 profile 字段；非医疗建议定位；紧急情况转发到泰国热线 1669/1323。

## 实验与结果
- **检索基准（50 题金集）**：Recall@5 = 0.88（44/50）；thai_herbs 与 packages 子项较弱（≈0.75），提示草药检索仍需改进。
- **部署用户验收（UAT，n=120）**：满意度 87.2%；用户自评聊天机器人准确度 86.5%；泰语草药知识准确度 90.2%；服务商信息可信度 91.0%（5 点 Likert，用户自评非专家评分）。
- **负载测试**：200 并发用户 0% 错误率；平台 API 平均延迟 ≤ 527 ms；不包括 LLM 生成时间（QA  campaign 中位生成耗时 14.1 s/回答）。
- **安全/鲁棒性**：通过 OWASP Top-10 (2025) 全部 55 项安全测试。
- **生产 QA 二轮回归（259 cases）**：通过率 91.1%（236/259 复现 pass）；分组：安全 19/21、多语言 13/14、输入鲁棒 11/11、核心养生内容 22/23、对话上下文 8/9、地理空间检索 163/181。56 个问题案例中 84%（47/56）集中于口语化地理空间查询，失败模式包括虚假邻近性声明、地标/区域解析失败、答案不可复现。
- **Frontier no-retrieval 探针（71 真实问题，Gemini 2.5 Flash，无检索）**：
  - 安全提示效果显著：vanilla 下出现完整对乙酰氨基酚剂量方案（含儿科剂量）并顺从超范围请求（写 Python、谈政治），3/22 明确违规；+safety 后 0/22 违规，剂量请求被拒并转药剂师。
  - 紧急情况置顶差异：vanilla 中 1669 首次出现在第 2,354 字符（总 3,596 字符），+safety 在第 141 字符（总 280 字符）；自残热线 1323 同样呈现前置提升。
  - 专业转介语言从 14/23 提升至 22/23。
  - **无检索=无可靠推荐**：12 个口语地理查询中，无检索模型给出 0 条可验证服务商推荐；8/12 退回全国性连锁品牌（Health Land/Let's Relax/Oasis/Divana）无地址/营业时间/认证，4/12 直接建议查 Google Maps 或酒店前台；部署版在 484 家认证服务商上做地理过滤，QA 通过率 163/181。
  - 多语言回复行为不受提示影响（13/14，两条件一致），说明语言对齐来自基座模型。
- **部署发现（可迁移）**：
  - English-calibrated 4-bit AWQ 量化破坏泰语元音与音调标记输出；
  - 自动 tool_choice=auto 时模型对简短名词短语静默跳过检索，classifier-forced tool_choice 恢复可靠 grounding。

## 相关工作脉络
- **泰语/SEA 基础模型**：Typhoon [16,17]、OpenThaiGPT [10]、SEA-LION [13]、Sailor [12]、Chinda [2]——均为通用模型，未针对传统/草药医学优化；TSWAP 选用未微调多语言基座，投资在 grounding 而非预训练。
- **泰语临床 NLP**：Eir [3]、泰国家医疗执照考试评测 [15]——聚焦临床/考试场景，未覆盖开放-ended 文化特异性草药/养生咨询质量与部署验证。
- **通用泰语基准**：ThaiExam [16]、M3Exam [7]、WangchanThaiInstruct [19]——考试或通用域，无传统医学子域。
- **唯一泰式传统医学 LM**：AppHerb [1]（Gemma-2 微调，两本教材生成文本，无部署、无多语言、无检索 grounding、无可复用基准）——TSWAP 在此基础上引入生产部署与检索评测。
- **传统医学基准类比**：MTCMB [9]、TCM-Ladder [14]（中文传统医学）——作为未来泰式基准设计模板；TSWAP 填补泰语/东南亚对应空白。
- **多语言健康聊天机器人评估**：Cross-lingual health-QA 研究显示非英语质量显著下降 [20]；HEALTH-PARIKSHA [6] 探索多语言 RAG；TSWAP 采用 RAGAs faithfulness/relevance/context 指标 [11]、medical-RAG 评估实践 [8]、rubric-based grading [5] 与安全/problematic/unsafe 标注 [18]，并提出完整的 benchmark 协议提案。

## 局限性与未来方向
- **多语言质量依赖预训练**：八语种中泰/英表现较稳，其余语种（马来/俄/韩等）质量继承自基座预训练，存在潜在降级风险。
- **安全层无训练分类器/交互检测器**：仅依赖 prompt 与路由，未做 adversarial red-teaming，覆盖面有限（仅安全提示中编码的场景）。
- **引擎侧消融与逐语种答案质量评估尚未完成**：当前仅有 frontier no-retrieval 探针与生产 QA，缺引擎侧 ablation 与更细粒度多语评测。
- **检索在草药集合上最弱**：thai_herbs 与 packages Recall 约 0.75，需改进分块/索引/检索策略。
- **端到端失败集中于口语化地理空间查询**：假邻近性、地标/区域解析失败、跨轮不可复现。
- **答案非完全跨轮可复现**：同问题在不同运行中出现差异（采样 nondeterminism）。
- **评估仅覆盖单一部署**：通用性有待在其他平台验证。
- **未来方向**：完整答案质量基准、引擎侧消融、FP8 迁移、thai_herbs 检索改进、结构化 red-teaming、跨语种对比评估。

## 研究启发与可借鉴点
1. **强制检索路由作为防幻觉机制**：对工具调用型 RAG，当模型易在简短名词短语上静默跳过检索时，query classifier + tool_choice 强制是低成本高回报的兜底策略，可迁移至其他实体密集型咨询场景。
2. **量化对低资源/声调语言的正交影响**：English-calibrated AWQ 破坏泰语音调标记的结论提醒团队在部署声调语言时优先验证量化对 morphological 特征的影响，必要时迁移到 FP8 或语言适配量化。
3. **translate-then-retrieve 多语言零样本方案**：单一多语言模型 + 查询翻译 + 单语知识库，避免维护 per-language KB；结合 bge-m3 多语 embed 提供容错，适合多语种且 KB 建设成本高的场景。
4. **Safety prompt + 紧急路由前置**：将 emergency 识别与热线置顶纳入 prompt 与分类器双重约束，并以字符位置度量“ prominence ”而非仅 presence，这种评估维度对高风险领域（医疗、金融）有借鉴价值。
5. **生产 QA 回归日志 + 探针设计**：发布 259 例 test-retest 通过率和 71 例 frontier no-retrieval 探针日志，为后续工作提供对比基线与可复现起点；团队可借鉴其“问题集+gold IDs+日志开源”的发布范式。
6. **结构化 monograph + 证据字段**：thai_herbs 将 contraindications 与 scientific/research references 作为一等公民，并在长寿建议变体中强制引用尾部，这种“evidence-first schema”可迁移到任何需要可追溯性的垂直 RAG。

## 关键术语表
- **RAG（Retrieval-Augmented Generation）**：生成前先检索外部知识库，将 retrieved chunks 作为上下文注入 LLM，以降低幻觉并提高事实性。
- **Hybrid dense–sparse retrieval**：同时使用稠密向量（embedding cosine）与稀疏（BM25）检索，经 RRF 融合后取 top-k，平衡语义匹配与词项匹配。
- **Cross-encoder reranking**：用较重的双编码器模型对候选集做 pairwise/qpair scoring 重排，精度高于 cross-query sparse/dense 初筛。
- **Reciprocal Rank Fusion（RRF）**：集成多路检索结果的排名融合策略，避免直接 score 对齐问题。
- **Translate-then-retrieve**：先将用户多语查询翻译成知识库语言再检索，利用多语 embedding 提供次级容错。
- **Grounding-first routing**：通过查询分类器强制实体类问题走检索工具，避免模型静默依赖参数记忆产生幻觉。
- **AWQ（Activation-aware Weight Quantization）**：一种大模型量化方法；本文指出 English-calibrated 4-bit AWQ 会破坏泰语音调与元音标记。
- **Combining-mark heuristic**：基于泰语 diacritic/tone mark 与辅音比例阈值的质检启发式，用于检测量化或生成导致的字符损坏。

## 可复现要素
- **数据集**：50 题泰语检索金集（含 gold document IDs）已开源；生产 QA 日志（259 cases）与 frontier-probe 日志已开源；知识库本体未公开。
- **代码/权重**：未明确声明代码开源；使用未微调 Qwen3.6-35B-A3B（社区 4-bit AWQ checkpoint，计划迁移到官方 FP8）；bge-m3、Qwen/Qwen3-Reranker-8B 为公开模型；Milvus v2.6.4。
- **关键超参**：chunk 最大 512 tokens、重叠 50 tokens；dense dim=1024；BM25 drop_ratio_search=0.1；rerank top-7、final top_k=3；mark-to-consonant ratio 阈值 0.28；5-point Likert；UAT n=120；200 并发；API 延迟 ≤527 ms（不含生成）。
- **评估指标**：Recall@K、constraint-violation@k、test–retest pass rate、rubric-based grading、faithfulness/relevance/context、safe/problematic/unsafe labeling。
