---
title: "Natural-Language-Code-Retrieval-for-1C-Enterprise-An-Open-Be"
source: https://arxiv.org/pdf/2608.19957v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:08:59"
field: "代码检索与嵌入式表示学习"
keywords: ["code retrieval", "1C:Enterprise", "BSL", "dense retrieval", "Matryoshka embeddings", "synthetic data", "domain adaptation"]
innovations: ["构建首个1C:Enterprise俄语-BSL代码检索开放基准与合成训练数据", "149M参数领域适配双编码器超越300M通用模型，验证中等规模+高质量合成数据的有效性", "MRL多尺度截断实现256维保留99.9%检索质量并降低3倍存储与计算开销"]
benchmarks: ["PruhaNLP/1C-Ebench"]
---

# 论文速读：Natural-Language-Code-Retrieval-for-1C-Enterprise-An-Open-Benchmark-and-Eficient-Bi-Encoder

## 一句话总结
本文构建了首个面向1C:Enterprise生态的自然语言代码检索开放基准（PruhaNLP/1C-Ebench，3,413个真实query-code对）与大规模合成训练数据（784,057个三元组），并在此基础上微调出领域适配双编码器 PruhaNLP/USER2-1C-code，在俄语查询→BSL/1C查询代码检索任务上显著超越通用多语言embedding基线。

## 研究问题与动机
1. **领域空白**：1C:Enterprise作为俄语区企业级软件平台，其代码（BSL语言+1C查询）混合西里尔关键词、俄语文档标识符与类SQL语法，但Open检索基础设施几乎为空。
2. **数据稀缺**：公开可用的监督式1C检索数据极少，现有1C评测资源（如1C Code Bench、PRISM/GenLab-1C）聚焦于代码生成而非检索。
3. **多语言/跨模态对齐难题**：查询以自然语言俄语描述错误、报表、会计操作等，文档为碎片化BSL代码片段，意图→代码的跨模态匹配不同于传统CLIR（非两种自然语言间翻译，而是语言意图→机器代码）。
4. **通用多语言Embedding是否足够？**：需实证检验强多语言embedding（如gemma、m3、e5）在1C领域是否足以胜任，或是否必须领域适应。

## 核心贡献（创新点）
1. **开源1C检索基准 PruhaNLP/1C-Ebench（3,413对真实query-code）**：填补1C领域开放评测空白，区分forum（84.5%，长上下文问题）与fastcode（15.5%，短查询+现成片段）两个子集，解决此前该生态无评测资源的结构性缺失。
2. **大规模合成训练数据 PruhaNLP/1C-Code-Train（784,057三元组）**：利用google/gemma-4-26B-A4B-it从公共GitHub代码库生成俄语查询+hard-negative对，辅以PII脱敏，首次构建可用于对比学习训练的1C检索监督信号。
3. **领域适配双编码器 PruhaNLP/USER2-1C-code**：以deepvk/USER2-base为底座，引入不对称prompt（search_query/search_document）、隐私感知tokenizer补丁、MRL多尺度截断，以149M参数实现显著性能跃升（+0.106 macro nDCG@10 vs base），证明中等规模模型+高质量领域数据可超越更大通用模型。
4. **可复现评估流水线 PruhaNLP/1C-RB**：统一支持dense/BM25/RRF融合与nDCG/Recall/MRR指标计算，并公开污染审计协议（exact+13-gram overlap），为社区提供标准化的复现基线。

## 方法详解
1. **Benchmark构建流程**：从Hugging Face数据集arefaste/1C_Forums（19,041条论坛记录）提取Markdown代码块，经精准过滤（仅保留BSL/1C查询片段）→精确去重→PII脱敏（[EMAIL]、[PHONE]、[PATH]、[PERSON]等占位符；问题侧保留原文，若含残余PII则整条剔除），最终产出3,413个query-code对（forum 2,883 / fastcode 530），每query对应单条gold文档（closed-set single-gold qrels）。

2. **合成训练数据生成**：基于leongl/1c_github（3,038,637行，过滤后784,058条唯一文档），使用google/gemma-4-26B-A4B-it以多样化prompt模板（长度/风格提示，另设API-name模式允许标识符直拷）生成俄语查询；硬负样本通过FAISS IndexFlatIP在top-20候选中按余弦相似度最大化选取（排除正样本）；PII处理变更约2,488处（≈0.106%）。

3. **模型架构与Prompt**：PruhaNLP/USER2-1C-code基于deepvk/USER2-base（ModernBERT/RuModernBERT架构，149M参数，max_len=8,192，mean pooling，cosine打分）。采用不对称role-specific prompt：查询端为`search_query: <q>`，文档端为`search_document: <d>`。隐私感知tokenizer补丁：[PATH]/[PERSON]映射到[unused0]/[unused1]，email/phone/IP使用专用USER2 tokens，新token权重初始化为对应子词均值。

4. **训练目标与超参**：损失函数为 `L = MatryoshkaLoss(CachedMNRL)`，即InfoNCE（1硬负+in-batch负）在各MRL前缀维度{768, 512, 384, 256, 128, 64, 32}上求和。超参：batch=256（MNRL mini-batch=32），epochs=3（steps=9,189），lr=2e-5，5%余弦warmup+decay，FP16，A100约7.5h。

5. **Matryoshka Representation Learning (MRL)截断**：推理时可将768维embedding前m维截断后L2归一化，实验表明256维保留99.9%检索质量，同时密集索引存储与精确相似度算术开销降至约1/3。

6. **污染审计协议**：对PruhaNLP/1C-Code-Train与1C-Ebench执行NFKC规范化+小写+空白折叠后，分别计算exact match与13-gram重叠；严格exact-pair层面零重复；fastcode 22.83%的dirty rate主要来自短查询模板与通用BSL样板代码（非语义泄露），移除所有dirty样本后模型仍保持0.6011 macro，排除数据污染解释。

## 实验与结果
- **数据集**：PruhaNLP/1C-Ebench（3,413对，forum 2,883 / fastcode 530，closed-set single-gold）。
- **评估指标**：主指标nDCG@10；报告balanced macro（两子集等权）、query-weighted micro、per-subset分数，辅以Recall@10、MRR@10。
- **基线模型**：deepvk/USER2-base、google/embeddinggemma-300m、deepvk/USER-bge-m3、ibm-granite/granite-embedding-311m-multilingual-r2、microsoft/harrier-oss-v1-270m、intfloat/multilingual-e5-base、ai-forever/sbert_large_nlu_ru、BM25Okapi（k1=1.2, b=0.75）、RRF(BM25+dense)融合（k_RRF=60）。
- **最优结果**：PruhaNLP/USER2-1C-code **macro=0.5992**，micro=0.5044，forum=0.4617，fastcode=0.7366；相对deepvk/USER2-base（macro=0.4932）提升 **+0.106**，相对google/embeddinggemma-300m（macro=0.5404）提升 **+0.0588**，paired bootstrap 95% CI均不为零（P<0.001）。
- **消融/稳健性**：
  - RRF(BM25+dense)在untuned设置下劣化dense结果（macro 0.4300 vs 0.5992），归因于`\w+`分词难以分离BSL标识符与错误串，非融合策略本身无效。
  - 去除13-gram污染样本后macro=0.6011，clean-only fastcode=0.7369，性能稳定，非数据泄露驱动。
  - MRL截断至256维保留99.9%质量，索引体积降至33%，理论计算量降3倍。
- **定性分析**：领域适配显著提升含1C平台术语（如动态列表模式、类型API口语表达）查询的召回；但对罕见COM API（如SAPI.SpVoice、Base64编码）仍失效，通用模型凭表面token反超。

## 相关工作脉络
1. **CodeSearchNet / CoSQA / CodeXGLUE / CoIR / CodeRAG-Bench**：覆盖主流英文编程语言检索评测，均未涉及1C/BSL或俄语查询，本文填补该细分生态空白。
2. **1C Code Bench (GigaCode/Sber AI)** 与 **PRISM/GenLab-1C**：评估LLM生成可编译BSL函数的能力（代码生成范式），与本文检索范式互补——前者问"能否写出正确代码"，后者问"能否从俄语描述找到已有片段"。
3. **Dense Passage Retrieval (Karpukhin et al.)** 与 **RocketQA**：奠定双编码器+InfoNCE硬负挖掘范式，本文沿用并适配至1C俄语-BSL域。
4. **BM25 lexical baseline**：证明纯词法重叠不足以捕捉俄语自然语言意图→BSL代码语义，尤其在非精确标识符匹配场景下差距显著。
5. **Matryoshka Representation Learning (Kusupati et al.)**：本文首次在1C代码检索中应用MRL实现多粒度部署权衡，验证256维即可保留近全量性能。
6. **Synthetic Query + LLM-as-a-Judge**：借鉴CoIR合成数据路线，但以gemma-4-26B生成查询、gemini-3.1-pro-preview抽检质量（300样本，相关性/自然度/清晰度均分≥4占比76-82%），作者审慎标注其为"近似可用信号而非专家标注"。

## 局限性与未来方向
1. **Single-Gold Qrels**：每query仅一条gold文档，论坛场景下常存在多个等价解决方案，实际检索质量被低估（手工抽样50条中10%存在多个相关片段）。
2. **合成数据缺乏人工校验**：300样本的LLM-judge评分为间接信号，未获开发者层面的俄语BSL专家验证，可能存在领域特有表达偏差。
3. **PII脱敏规则未量化**：基于正则的precision-first方案未报告精确率/召回率，也未做残留审计，存在隐私风险未知。
4. **实验覆盖不足**：未评估reranker、多seed方差、hard-vs-random负样本对比、tokenizer补丁消融、ANN索引与大规模延迟；RRF为untuned naive实现，不排除调优后有益。
5. **污染审计仅覆盖n-gram**：无法检测语义 paraphrase 泄露；train-benchmark领域鸿沟仍存在。
6. **未来方向**：多gold qrels构建、人工查询标注、FN去噪、调优hybrid/reranker、针对性消融实验、扩展至1C查询语言（SQL-like）与外部系统集成。

## 研究启发与可借鉴点
1. **中等规模底座+高质量领域合成数据 > 通用大模型**：149M参数的USER2-base经领域微调后超越300M级gemma，提示垂直领域应以数据质量与prompt工程为核心杠杆，而非盲目堆参。
2. **MRL多粒度截断的工程价值**：在部署受限场景（移动端/边缘检索、大规模在线服务）下，可复用"训练全维、推理按需截断"策略实现存储-精度-计算三角权衡。
3. **不对称role-specific prompt设计**：`search_query` vs `search_document`前缀对中文/多语检索任务同样适用，值得在团队中文代码检索（如Java/Python企业库）中迁移验证。
4. **污染审计标准化流程**：exact+13-gram两阶段审计+clean-only重算的稳健性报告范式，可作为合成数据训练论文的必备透明度指标纳入团队规范。
5. **Hard-negative mining的误判风险评估缺口**：论文明确未估计FN率，团队在做对比学习训练时，应引入人工抽检或多样性约束缓解假负样本污染。

## 关键术语表
**PruhaNLP/1C-Ebench**：面向1C:Enterprise的开放检索基准，含3,413个真实俄语问题-BSL/1C查询代码对，分为forum与fastcode两个子集。
**PruhaNLP/1C-Code-Train**：由gemma-4-26B生成的784,057个(Russian query, positive code, hard negative code)三元组训练集，已做PII脱敏。
**PruhaNLP/USER2-1C-code**：基于deepvk/USER2-base微调的领域适配双编码器，149M参数，支持MRL多尺度截断与不对称prompt。
**BSL (Business Processes and Languages)**：1C:Enterprise平台专有编程语言，混合西里尔标识符与类SQL语法，用于企业ERP/会计/报表开发。
**Matryoshka Representation Learning (MRL)**：使高维embedding的前m维自身构成低维优质表示的技术，支持推理时按资源需求灵活截断。
**Balanced macro vs query-weighted micro**：macro对两子集等权平均（估计"检索体制均衡"性能），micro按query数量加权（估计"随机提问"性能），二者互补反映不同部署视角。
**13-gram dirty flag**：保守的污染检测机制，将query或code文本切分为连续13元语素集合，任一交集即标记为dirty，可捕获短语级重复。
**Single-gold qrels**：每条query仅标注一条gold相关文档的评测协议，简单但可能低估检索器在有多等价解时的真实效用。

## 可复现要素
- **数据集**：PruhaNLP/1C-Ebench（Hugging Face）公开；PruhaNLP/1C-Code-Train公开；上游数据源arefaste/1C_Forums与leongl/1c_github均在Hugging Face。
- **代码/评估流水线**：PruhaNLP/1C-RB评估框架开源（GitHub/Hugging Face）。
- **模型权重**：PruhaNLP/USER2-1C-code模型权重开源。
- **关键超参**：batch=256，MNRL mini-batch=32，epochs=3（steps=9,189），lr=2e-5，5% cosine warmup，max_seq_len=8,192，FP16，A100 ~7.5h。
- **训练硬件/时间**：单卡A100，约7.5小时。
- **提示模板**：查询端 `search_query: <q>`，文档端 `search_document: <d>`（详见论文Table 4）。
