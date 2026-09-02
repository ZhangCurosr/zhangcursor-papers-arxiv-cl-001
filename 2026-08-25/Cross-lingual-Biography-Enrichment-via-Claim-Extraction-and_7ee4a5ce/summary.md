---
title: "Cross-lingual-Biography-Enrichment-via-Claim-Extraction-and"
source: https://arxiv.org/pdf/2608.23390v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-02 19:47:53"
field: "跨语言自然语言处理"
keywords: ["跨语言传记丰富化", "Claim 抽取", "跨语言对齐", "多语言 NLI", "维基百科补全", "事实性生成"]
innovations: ["提出跨语言传记丰富化任务，以 claim 级结构化证据驱动英语传记重写", "构建 CLAW-4L 三件套数据集（传记对/claim抽取/关系分类基准）", "设计 Tier-aware 分层采样策略与约束局部搜索实现多维分布对齐"]
benchmarks: ["CLAW-4L", "CLAW-4L-CX", "CLAW-4L-RC", "X-Claimify", "FactScore", "VeriScore"]
---

# 论文速读：Cross-lingual-Biography-Enrichment-via-Claim-Extraction-and

## 一句话总结
本文提出**跨语言传记丰富化**任务，通过将英/非英语维基百科传记共同映射到共享的英语 claim 空间、经 claim-pair 关系对齐筛选出非英侧特有补充信息，以结构化证据驱动英语传记重写；同时构建 CLAW-4L 系列三件套数据集（传记/claim抽取/关系分类），验证该方法可显著降低幻觉并提升 supported additions。

## 研究问题与动机
- **长尾人物知识不对称**：英语维基百科常被视为默认百科来源，但对非英语语境中的长尾人物（尤其女性），非英语版本可能包含更丰富的本地化事实。
- **直接翻译/原始输入易产生幻觉**：将非英语传记翻译后整段输入生成器（Translation 设置）或直接拼接原文（Raw 设置），相比显式抽取 + 对齐 novel evidence 会产生更多幻觉且支持度低。
- **缺乏跨语言 claim 级基准**：现有 claim 抽取与关系分类基准主要面向英语单语场景，缺少面向多语言传记的原子化 fact 抽取、配对关系标注与丰富化生成评测体系。
- **开源模型在 claim 级任务上已达可用水平**：Qwen3.5-9B 等开源 LLM 在 claim 抽取和 pair 关系分类上已接近 GPT-5.1，为低成本落地提供可能。

## 核心贡献（创新点）
1. **提出跨语言传记丰富化（Cross-lingual Biography Enrichment）任务**：以已有英语传记为目标侧，用非英语传记经 claim 提取+对齐筛选后的补充证据驱动重写，区别于从头生成或纯翻译输入的设置。
2. **构建 CLAW-4L 系列三件套数据集**：CLAW-4L（300 对英/法/中/阿塞拜疆女性传记）、CLAW-4L-CX（600 句手动标注的英语 claim 抽取基准）、CLAW-4L-RC（600 对跨语言 claim 关系标注基准），填补多语言传记事实级评测空白。
3. **设计 Tier-aware 分层采样策略**：基于语言特异性 token 膨胀因子 $b_L$ 估计内容丰富度，以规范化比率 $r^{\text{norm}}=r/b_L$ 进行三级分层筛选，并用约束局部搜索（加权失配目标 $\mathcal{L}(S)$）实现职业/tier/token 量等多维分布对齐。
4. **系统性评测五种跨语言 claim 抽取框架并推荐 X-Claimify**：对比 X-FactScore / X-VeriScore / X-DnDScore / X-FactCheck-GPT / X-Claimify，证明多阶段 multi-turn pipeline（verifiable selection → disambiguation → decomposition）在跨语言场景显著优于 Joint 单步方案。
5. **揭示非英语维基与英语维基的互补结构**：仅 13.9% 非英 claim 可与英语 claim 精确对齐，48.6% 完全无对应、26.2% 为补充性增补，合选率达 86.2%，证实跨语言 enrichment 的信息增益来源。

## 方法详解
- **共享 claim 空间映射**：将英语与非英语传记均通过 claim 提取器（X-Claimify）映射为英文结构化 claim，字段包括 subject / predicate / object / time / location / reason / manner / hedge，使双语内容进入同一表征空间。
- **语言校准的 token 膨胀因子**：基于 FLORES+ 平行数据与 `google/mt5-small` tokenizer 估计每对 (EN, 非EN) 句对的期望 token 比率 $b_L$；规范化比率 $r^{\text{norm}}=r/b_L$ 作为内容丰富度的启发式指标（EN→ZH=0.91，EN→FR=1.41，EN→AZ=1.36）。
- **三级分层筛选（Tier-aware Selection）**：Tier A ($r^{\text{norm}}\geq 1.20$) / Tier B ($1.00\leq r^{\text{norm}}<1.20$) / Tier C ($r^{\text{norm}}<1.00$)；优先从 Tier A/B 选取，Tier C 兜底；每个国家各选 300 实例，共 900 候选池。
- **约束局部搜索采样**：以职业桶、tier、$r^{\text{norm}}$、非英文 token 长度、国籍数量桶、infobox 存在性为匹配属性，定义加权失配目标 $\mathcal{L}(S)$（$\lambda_{\text{tier}}=4.0,\lambda_{\text{ratio}}=2.0,\lambda_{\text{tok}}=1.5,\lambda_{\text{nat}}=1.0,\lambda_{\text{info}}=0.8$），在 occupation bucket 内 swap 最小化目标。
- **CLAW-4L-CX 选句策略**：每传记只选一句，按 claim 数量分四桶（1/2/3/$\geq$4），三指标（LaBSE Coverage / Fact density / Rewrite cost）各自 biography 内归一化排名后等权聚合；全局约束每桶恰好 25 例/国。
- **Claim pair 关系分类**：六类标签——exact alignment (A=B)、EN-more (A>B)、target-add (B>A 或 A↔B)、contradicted (A⊥B)、not relevant (A⊣B)；用 Qwen3.5-9B 作为分类器（ARC 95.5），排除 Exact 和 EN-More 后纳入 enricher 候选池。
- **三种生成设置对比**：Raw（原始英+原始非英直接输入）/ Translation（非英先翻译再输入）/ Claims（经 claim 提取+对齐后的补充 claim 驱动重写）；Claims 设置以显式 novel evidence 显著降低幻觉并提升 supported additions。

## 实验与结果
- **Claim 抽取框架（GPT-5.1 骨干，Table 2/17）**：X-Claimify 全面最优，EN Exact-Aligned-F1 = 72.44，FR = 73.56，ZH = 71.12，AZ = 66.80；平均 A-F1 = 73.72，E-F1 = 71.18，显著超越 X-VeriScore（平均 E-F1 49.47）。
- **Claim pair 分类器（Table 5.1/15）**：开源最优 Qwen3.5-9B（ARC 95.5 / Align-FG 89.5 / Overall 93.2），闭源 GPT-5.1（ARC 95.9 / Align-FG 90.8 / Overall 93.3）；Infobox 增强对大模型无益甚至略降分。
- **翻译模型（Table 20）**：开源 LLM 全面优于专用 MT 模型；最终 AZ→EN 选 Gemma-4-31B-it，FR→EN 选 Mistral-3.2-24B-it，ZH→EN 选 Qwen3.6-27B。
- **非英→英 claim 覆盖统计（Table 3）**：仅 13.9% 非英 claim 与英语 exact-aligned；48.6% 完全无对应（Non-English-additive）；26.2% 为补充性增补（Target-Add）；合选率 86.2%，证实强互补性。
- **选句质量增益**：CLAW-4L-CX 选句后 EN 总分 +0.2722，non-EN 总分 +0.3179（Table 12）。
- **语义相似度验证（Table 14）**：Exact aligned 对 BGE 相似度最高（96.87）；Not relevant 对最低（BGE 51.31 / MPNet 35.86），与标注一致。
- **最强结果**：X-Claimify 在多语言 claim 抽取上 Exact-Aligned-F1 达 70–74，为所有评测框架中的绝对最优。

## 相关工作脉络
- **FactScore / DnDScore / VeriScore / FactCheck-GPT**：均为单语 claim 抽取或 verifiability 评估框架；本文将其适配为跨语言版本（X-前缀），并系统对比五者在英/法/中/阿四语上的表现，证明多阶段 pipeline（X-Claimify）优于各类 Joint 方案。
- **维基百科跨语言对齐工作**：以往工作多依赖 Wikidata 实体链接或 sentence-level 翻译对齐；本文聚焦 claim 级原子 fact 抽取与六类关系分类，粒度更细且面向生成任务下游。
- **传记/知识丰富化生成**：从头生成（从零构建）或结构化数据驱动的方法不依赖已有英语传记；本文独特定位是"以英语传记为基底、以非英 claim 为补充证据"的重写范式。
- **跨语言 NLI / 关系分类**：MiniCheck-7B / DeBERTa-v3-large-mnli 等 NLI 基线因标签体系不匹配无法覆盖六类关系，本文提出专用分类器（Qwen3.5-9B）填补空白。
- **CLAW-4L-CX 与 CLAW-4L-RC 作为独立基准**：前者面向多语言 claim 抽取，后者面向跨语言 claim 对关系分类，二者均为领域内首批面向女性长尾传记的公开基准。

## 局限性与未来方向
- **仅覆盖四种语言（EN/FR/ZH/AZ）与女性人物**：样本局限于特定性别与语种，泛化至其他人口统计学维度与其他语言尚待验证。
- **非英→英 claim 精确对齐率仅 13.9%**：虽互补性强，但大量 claim 无对应可能包含噪声或难以验证的事实，当前筛选策略未做事实核查。
- **X-Claimify 依赖 GPT-5.1 等大模型**：虽开源模型（Qwen3.5-9B）已接近，但高质量抽取仍以闭源模型为上限，部署成本较高。
- **claim pair 关系标注依赖人工校验 GPT-5.1 改写样本**：非正例通过结构化扰动+LLM 改写生成，可能存在人工校验疏漏。
- **未报告端到端丰富化生成的完整指标**：文中重点在抽取与对齐环节，最终的传记重写质量（如人工评测/事实一致性）需进一步展开。

## 研究启发与可借鉴点
- **Tier-aware 分层采样策略可迁移至其他多语言数据集构建**：基于语言膨胀因子做规范化比率、再以约束局部搜索做多维分布对齐，适用于任何跨语言平行数据筛选场景。
- **claim 级结构化输出（subject/predicate/object/time/location/hedge）可直接复用于知识图谱构建或 fact-grounded 生成任务**，无需重新设计 schema。
- **三指标选句（Coverage / Fact density / Rewrite cost）的归一化排名聚合方案**为句子级样本质量评估提供了可复用的度量框架。
- **开源 LLM（Qwen3.5-9B）在 claim 抽取与关系分类上已达 GPT-5.1 的 95%+ 水平**，提示后续工作可在低成本开源模型上开展规模化实验。
- **非英→英 86.2% 合选率**为多语言知识库补全任务提供了强有力的动机定量依据，可引用于跨语言 RAG 或 multilingual KB enrichment 的动机论述。

## 关键术语表
**Cross-lingual Biography Enrichment**：以已有英语传记为基底，利用同人物非英语维基百科传记经 claim 提取+对齐后的补充证据进行重写富化的任务。
**Claim**：传记中的原子化事实单元，结构化为 (subject, predicate, object, time, location, reason, manner, hedge) 元组，基本对应 fact。
**X-Claimify**：多阶段 multi-turn prompt pipeline，按 verifiable selection → disambiguation → decomposition 顺序抽取英文结构化 claim，为本工作中抽取性能最优的框架。
**Tier-aware Selection**：基于语言校准 token 膨胀因子 $r^{\text{norm}}$ 将候选样本划分为 A/B/C 三级并优先选取高丰富度样本的分层筛选策略。
**Aligned-F1 / Exact-Aligned-F1**：claim 抽取评估的两个核心指标，前者接受 partial 对齐判定，后者仅接受 exact 对齐判定。
**ARC（Aligned Relationship Classification）**：claim pair 六类关系分类的 macro-avg 准确率，衡量跨语言 claim 对齐关系的判定质量。
**CLAW-4L / CLAW-4L-CX / CLAW-4L-RC**：本文构建的三个 benchmark，分别为跨语言女性传记对（300 对）、句子级 claim 抽取（600 句）、claim 对关系分类（600 对）。
**Token 膨胀因子 $b_L$**：基于 FLORES+ 平行数据估计的每种语言相对于英语的平均 token 比率期望值，用于内容丰富度启发式估算。

## 可复现要素
- **数据集**：CLAW-4L / CLAW-4L-CX / CLAW-4L-RC；构建方法详细描述，公开状态论文未明确声明（需以 arxiv 附录或项目页为准）
- **代码**：论文未提及开源仓库
- **权重**：使用 GPT-5.1 作为主要 backbone；开源替代为 Qwen3.5-9B、Gemma-4-31B-it、Mistral-3.2-24B-it、Qwen3.6-27B
- **关键超参**：tier 阈值 1.00 / 1.20；采样权重 $\lambda_{\text{tier}}=4.0, \lambda_{\text{ratio}}=2.0, \lambda_{\text{tok}}=1.5, \lambda_{\text{nat}}=1.0, \lambda_{\text{info}}=0.8$；chunk 长度 ≤ 4096 tokens；cosine 阈值 ≥ 0.7；每桶 25 例/国
