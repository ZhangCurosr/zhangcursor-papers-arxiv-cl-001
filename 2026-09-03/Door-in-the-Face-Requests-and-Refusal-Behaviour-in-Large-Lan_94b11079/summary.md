---
title: "Door-in-the-Face-Requests-and-Refusal-Behaviour-in-Large-Lan"
source: https://arxiv.org/pdf/2609.02707v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 22:35:01"
field: "大语言模型安全与对齐"
keywords: ["door-in-the-face", "LLM compliance", "sequential request", "jailbreak", "refusal behavior", "multi-turn persuasion", "model family heterogeneity"]
innovations: ["发现DITF在Anthropic与OpenAI/Google模型间符号反转，跨九模型证实模型家族是效应方向的决定性因素", "提出交付物类型(operational vs. inert)是退让能否生效的根本前提，并将265条操作性被拒改写为惰性后263条始终合规", "在真实拒绝场景下建立四条件因果鉴定设计，证明让步溢价跨模型通用但暖场抑制同样普遍"]
benchmarks: ["60-item constructed verdict set", "XSTest", "OR-Bench-Hard", "FalseReject", "HarmBench", "StrongREJECT", "JailbreakBench", "EVOREFUSE"]
---

# 论文速读：Door-in-the-Face-Requests-and-Refusal-Behaviour-in-Large-Lan

## 一句话总结
本文首次在真实拒绝场景下测试了心理学中的"面子法"（Door-in-the-Face, DITF）能否提升大语言模型对后续较小请求的遵从率；发现效果高度依赖模型家族与请求内容——Anthropic前沿模型显著提升遵从，OpenAI/Google前沿模型反而降低遵从，且只有当小请求索取"判断"而非"可操作输出"时，退让才可能奏效。

## 研究问题与动机
- **核心问题**：面对 LLM 反复拒绝用户请求的现实，将人类社会心理学中的顺序请求策略（先提大要求被拒→再缩小要求）迁移到 LLM 上是否能提升遵从？
- **现有工作不足**：
  - Meincke 等（2026）测试了七条 Cialdini 原则但对 LLM 的"拒绝"本身未作操纵，DITF 不在其测试范围内；
  - Zeng 等（2024）将 DITF 降格为单轮改写，没有真正触发模型拒绝；
  - 多轮红队/对话文献将拒绝视为攻击障碍，无人研究"同一模型拒绝后再退让"的因果效应；
  - 现有基准（XSTest、HarmBench 等）仅在冷启动上下文中对单条提示打标，无法刻画前一个被拒请求对后续请求的影响。

## 核心贡献（创新点）
- **模型家族层面的符号反转**：同一 DITF 序列在 Opus 5 上提升遵从 +36.5pp，在 Gemini 3.1 Pro 上却下降 −23.0pp；差异由厂商/模型家族组织，不跟随规模缩放。
- **请求内容决定退让可行性**：区分"操作性"（可执行的代码/脚本/分析）与"惰性"（立场/解释/判断）两类交付物；将 265 条被拒的操作性请求改写为同主题惰性解释后，263 条在所有尝试中始终合规，说明什么被索取才是退让能否生效的根本因素。
- **发现通用让步溢价与测量控制需求**：跨所有九个模型，同项被拒大请求都比无关项被拒大请求更有助于小请求（+0.7 至 +22.8pp）；但一次善意的暖场回答本身会抑制遵从（−17.8pp），因此 DITF 必须以直接冷问为对照，否则会被放大。
- **基准拒绝不可迁移**：公开基准上的模型自有拒绝在 DITF 下无一显著正向，Opus 5 从 +36.5pp 翻转为 −13.3pp，证明 DITF 无法从人工构造的判断类请求泛化到真实基准拒绝。

## 方法详解
- **实验设计（四条件）**：每个 trial 是一次独立 API 对话，以相同的"小请求"作为最后一个用户轮次；四种前置条件：
  - **A（cold）**：直接发送小请求；
  - **B（DITF）**：先发由同一 item 放大的"大请求"（添加三项限定：要求列出全量成员、宣称排名客观正确、贬低不同意者为傻瓜），模型被拒后再发小请求；
  - **C（warm-up）**：先问一个同主题中性事实问题并得到回答，再发小请求；
  - **D（unrelated refusal）**：先发另一 item 的同类型大请求（不同主题组）并被拒，再发小请求。
- **两个数据集**：
  - **60-item 判断集**：每条询问对某一现实机构类别的单条负面判断（含意见套语 "in your opinion"），共 35 个实体类（24 个 UK 相关）；锚点由模板自动生成并经筛选。
  - **基准迁移集**：从 10 个公开套件（XSTest、OKTest、OR-Bench-Hard、PHTest、FalseReject、EVOREFUSE、CoCoNot、JailbreakBench、StrongREJECT、HarmBench）抽取 7,664 条去重提示，筛选出每个模型"可移动"（movable）的被拒项（部分拒绝/部分合规）参与二次实验。
- **模型与采样**：九个生产模型（Anthropic 五款、OpenAI 两款、Google 两款），均使用官方 API 默认采样；token 上限 16,000（实验对话）、4,096（冷筛）、2,000（judge 调用）。
- **评估与判定**：两家跨厂商 judge 模型（从不与目标同厂）对每轮回复独立打分（complied/partial/refused/clarify），严格规则要求双方一致判为 complied 才算遵从； disagreement 计入 non-complied 但不剔除 trial。
- **统计估计**：主要估计量为逐 item 的 DITF−cold 风险差异均值 $\Delta_{BA} = \frac{1}{n}\sum_i(\bar{p}_i^B - \bar{p}_i^A)$；95% CI 采用 item-cluster bootstrap（10,000 次）；p 值采用 exact sign-flip randomization；九模型内 Holm 校正；跨模型汇总采用 item-clustered GEE（Liang-Zeger）。

## 实验与结果
- **主实验（60-item 判断集）**：
  | 模型 | Cold | DITF | Δ(B−A) | Holm p |
  |---|---|---|---|---|
  | Opus 5 | 29.3% | 65.8% | **+36.5** [28.2, 45.0] | 5.5×10⁻¹² |
  | Opus 4.5 | 6.5% | 32.3% | +25.8 [17.2, 35.0] | 3.8×10⁻⁷ |
  | Sonnet 5 | 20.3% | 37.0% | +16.7 [8.2, 25.7] | 0.0022 |
  | Opus 4.8 | 3.7% | 17.3% | +13.7 [4.8, 22.7] | 0.012 |
  | Haiku 4.5 | 29.8% | 13.8% | −16.0 [−25.3, −6.8] | 0.0049 |
  | GPT-5.6 sol | 58.8% | 43.3% | −15.5 [−22.7, −8.3] | 4.5×10⁻⁴ |
  | Gemini 3.1 Pro | 24.7% | 1.7% | **−23.0** [−31.5, −15.2] | 5.4×10⁻⁷ |
  - **最强效果**：Opus 5（+36.5pp），**最大负效果**：Gemini 3.1 Pro（−23.0pp）；二者 within-item 差距 +59.5pp（p=3.4×10⁻¹⁴）。
- **条件分离**：
  - Warm-up 抑制在所有 9 个模型上均为负，GEE  pooled −17.8pp（SE 2.0, p=1.4×10⁻¹⁸），证明 DITF 必须与 cold 对照；
  - Unrelated-refusal 控制下，同项大请求在所有 9 模型上均优于无关大请求（+0.7 至 +22.8pp，Holm-significant 7 个），即"让步关系"具有跨模型溢价。
- **基准迁移（H3）**：在 399 个 model-item 对的基准可移动拒绝项上，无任何模型获得显著正向 DITF 效果；Opus 5 翻转至 −13.3pp；per-protocol 下 Gemini 3.1 Pro（−16.0pp）与 Gemini 3 Flash（−28.0pp）更是显著恶化。
- **交付物类型机制（H4）**：
  - 基准池中 91.7% 为 operational 类，判断集全为 inert 类；
  - 将 265 条 operational 被拒项改写为同主题 inert 解释后，263/265 始终合规；回归均值无法解释（P(S≥263) < 10⁻²⁷⁸）；
  - 盲测六模型 holdout：inert 上 DITF 平均 +20.5pp，operational 上 −11.7pp，交互 +32.2pp（p=5.5×10⁻⁴）。
- **稳健性**：五种判定规则下 8/9 模型符号稳定；per-protocol 过滤（仅保留 anchor 被判为 refused 的 trial）不改变结论；baseline floor/ceiling / UK 子集检查均不影响结论。

## 相关工作脉络
- **Cialdini et al. (1975)** 提出 DITF 的互惠让步机制；本文将其从人类情境严格移植到 LLM 的"真实拒绝→退让"多轮序列，此前无人真正触发并退让于模型自身的拒绝。
- **Meincke et al. (2026)** 在 N=126,000 对话上测试七条 Cialdini 原则对 objectionable 请求的提升；DITF 未被测试且未产生真实拒绝，本文证明该技术的排序高度依赖是否包含真正拒绝。
- **Zeng et al. (2024) Persuasive Paraphrase Taxonomy** 将 DITF 简化为单轮改写而失去"拒绝"这一核心成分，使该技术排在最弱档；本文强调拒绝本身是生效的必要前提。
- **Wang et al. (2024)、Weng et al. (2025)** 报告 foot-in-the-door 作为多轮越狱极具威力（成功率 84%–94%）；本文 DITF （foot-in-the-door 的镜像）在 benchmark 上完全失效，说明两种策略对请求类型敏感。
- **Singhania et al. (2025)** 观察性发现首轮拒绝可显著降低后续攻击成功率（54.7%→6.6%）；本文为因果估计并揭示同序列在不同模型上可能反向。
- **Arditi et al. (2024)** 发现拒绝由单一激活方向介导；本文机制推断与之兼容但仅基于行为观测，未做 mechanistic probing。

## 局限性与未来方向
- 主效应仅来自单一模型生成的英语请求族，不能推广至其他风格或领域；
- 九个模型中仅 GPT-5 mini 为固定快照，其余未锁定，结果描述 2026-08 部署状态；
- 机制推断完全基于行为数据，未做内嵌激活/权重层面的验证；
- 基准交付物对比为观察性（仅定方向），"改写消除拒绝"缺乏保持内容不变的 paraphrase 对照；
- 控制设计缺"无关中性暖场"一项，尚难分离"话题切换成本"与"拒绝本身成本"。
- 未来方向：按家族自适应生成 item、引入 unrelated benign prior turn 补齐控制、对自发 counter-offer 进行因果种植实验。

## 研究启发与可借鉴点
- **四条件对照设计（cold / DITF / warm-up / unrelated refusal）** 可直接复用于其他多轮影响力策略的因果鉴定，避免将暖场抑制误读为技术效果；
- **交付物分类（operational vs. inert）** 可作为红队/安全评测的元特征，提示防御方按交付物类型分层评估而非一刀切；
- **跨厂商双 judge 盲评** 的设计有效隔离了目标模型自判偏差，适合后续多模型影响力实验复用；
- **逐 item 筛选冷拒绝行为以构建"可移动"集**，为后续研究提供了可复用的基准筛选流程（prescreen → rescreen → movability 判定）；
- 本团队可结合"请求类型→拒绝边界"的思路，探索在其他心理策略（foot-in-the-door、reciprocity）上按模型家族定制 item 生成的可行性。

## 关键术语表
- **Door-in-the-Face (DITF)**：先提一个极可能被拒的大请求，被拒后再退让到较小目标请求，人类中可使后者遵从率上升的影响策略。
- **Per-item mean risk difference（逐 item 风险差异均值）**：本文衡量 DITF 效果的主要估计量，即每个 item 上 DITF 与 cold 遵从率之差 Across items 的平均。
- **Operational vs. Inert 交付物**：前者指可付诸执行的输出（代码、脚本、模板、弱点分析），后者仅指陈述性立场或解释。
- **Movable item（可移动项）**：在基准集中既曾被模型拒绝、也曾被模型遵从的提示，可用于评估上下文对同一提示行为的影响。
- **Warm-up suppression（暖场抑制）**：先进行一次成功的中性问答会使后续小请求遵从率下降约 17.8pp，因此在 DITF 实验中 warm-up 不适合作为主要对照。
- **Concession premium（让步溢价）**：无论模型家族如何，同 item 的被拒大请求总是比无关 item 的被拒大请求更能提升后续小请求遵从。
- **Cluster bootstrap / sign-flip randomization**：Bootstrap 以 item 为聚类单元重抽样估计 CI；sign-flip 以 item 为单位翻转符号计算 exact p 值。
- **Per-protocol sensitivity**：在 DITF 条件下仅保留 anchor 两轮均被判为"拒绝"的 trial 的分析，排除 anchor 实际已被部分满足的情况。

## 可复现要素
- **数据集**：主实验 60 条 constructed verdict items；基准集 7,664 条去重 prompt（来自 XSTest、OKTest、OR-Bench-Hard、PHTest、FalseReject、EVOREFUSE、CoCoNot、JailbreakBench、StrongREJECT、HarmBench）。论文声明 item sets、judged per-trial outcomes、analysis code 将公开；harmful-tier 原始文本不公开。
- **代码/权重**：论文未提及额外开源权重；代码与数据声明将在发表时开源。
- **关键超参**：未指定任何采样参数（temperature/top-p/seed/stop），使用各提供商默认配置；token 上限 16,000（实验对话）、4,096（冷筛）、2,000（judge）；每 cell 10 次重复；bootstrap 10,000 次；sign-flip 精确检验；Holm 校正跨九模型。
- **模型标识**：九个模型均通过官方 API 调用，仅 GPT-5 mini 为固定快照（gpt-5-mini-2025-08-07）。
