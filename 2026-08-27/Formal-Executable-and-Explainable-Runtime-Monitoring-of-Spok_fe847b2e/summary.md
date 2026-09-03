---
title: "Formal-Executable-and-Explainable-Runtime-Monitoring-of-Spok"
source: https://arxiv.org/pdf/2608.25926v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:32:33"
field: "空管运行时形式化验证"
keywords: ["runtime verification", "air traffic control", "temporal logic", "spoken procedure monitoring", "formal methods", "multi-source integration", "explainable AI"]
innovations: ["首个管制员/飞行员义务的度量时序逻辑形式化", "语音-监视多源融合的自动化轨迹构建流水线", "基于三值语义的可解释违规裁决框架"]
benchmarks: ["TartanAviation KAGC 3h ground-truth", "ATCO2 test set", "Uberlingen 2002事故重建", "Comair 5191 2006事故重建"]
---

# 论文速读：Formal-Executable-and-Explainable-Runtime-Monitoring-of-Spok

## 一句话总结
本文提出了一套运行时验证框架，将飞行员与管制员 spoken 无线电通信解析为与航空器实体绑定的操作事件，与监视/机载观测数据融合为带时间戳的执行轨迹，并通过 ICAO 导出的含显式时限的时间公式进行评估，最终以可解释的证据链报告每一起违规。

## 研究问题与动机
1. **核心问题**：空管程序合规性验证需要同时关联语音内容、航空器身份与状态、以及多重时间约束，而人工持续完成这一任务在流量增长与人员紧张的双重压力下已难以为继（2024 年全球定期航班 3700 万班次，欧洲航路延误创 2001 年以来新高）。
2. **现有方法不足**：既有工作仅覆盖部分监控需求——轨迹类方法缺少语音输入，语音类方法缺少多源融合与时序推理，形式化 RV 方法假设轨迹已知，LLM 提示类方法缺乏实体接地与时限语义。
3. **事故驱动**：Uberlingen（2002，71 人遇难）和 Comair 5191（2006，49 人遇难）两起事故的经验教训表明，程序偏差往往隐藏在单源数据中，必须跨语料、监视和机载系统联合推理才能发现。
4. **四项监控需求**：实体接地（R₁）、多源融合（R₂）、时序/时限推理（R₃）、基于证据的可解释裁决（R₄）需同时满足。

## 核心贡献（创新点）
1. **首个管制员/飞行员义务的时间逻辑形式化**：将 ICAO Doc 4444 规范编码为带度量时限的有限轨迹时序公式，此前的工作未对 ATC 双向义务进行完整的时序逻辑建模。
2. **自动化的异构操作数据融合流水线**：从语音识别 → 实体绑定 → 轨迹组装构建统一带时间戳执行轨迹，与之前仅用单一数据源（轨迹或录音）的工作形成本质区别。
3. **可解释的三层语义运行时验证器**：在开放轨迹上采用三值语义（true/false/inconclusive），违规裁决同时返回触发位置和支撑命题集合，区别于仅返回公式级证明的先前 RV 工作。
4. **在真实流量与历史事故上的实证评估**：真实交通 F1=0.85；1,495 个合成场景全部正确；Uberlingen 和 Comair 5191 事故重建中检测时间与调查结论一致且留有操作预警裕度。
5. **五种可组合的程序逻辑模式**：有界响应、不变量、优先权、待决响应、有界禁止，覆盖了读回确认、状态一致性、跑道入侵预防等 ATC 核心程序类型。

## 方法详解
**形式模型** $\mathcal{M} = \langle \mathcal{A}P, \Pi, \Phi, \models \rangle$：
- $\mathcal{A}P$：多排序一阶词汇集合（Aircraft、Level、Runway、Agent、Utterance 等），包括四类命题——语音命题（Says(atc, cmd_descend(a,ℓ)) 等）、空态命题（descending(a) 等）、地表状态命题、派生命题（读回匹配、授权保持、RA 重建等）。
- $\Pi$：有限带时间戳轨迹 $\pi = (t_1, L_1), \ldots, (t_n, L_n)$，其中 $t_i \in \mathbb{N}$（秒），$L_i \subseteq \mathcal{A}P$，时间戳单调非减。
- $\Phi$：基于五种模式实例化的时序公式集（见下表），采用 $\mathrm{LTL}_f$ + 度量算子 $\mathbf{F}_{\leq \delta}$。
- $\models$：三值运行时语义（true/false/inconclusive），时限到期后转为二值。

**五种公式模式与 ATC 家族**：
| 模式 | 形式 | ATC 家族示例 |
|---|---|---|
| 有界响应 | $\mathbf{G}(\alpha \to \mathbf{F}_{\leq\delta}\beta)$ | readback（30s 内读回）、state-consistency（状态一致性）、RA（5s 响应） |
| 不变量 | $\mathbf{G}(\alpha \to \beta)$ | runway-incursion（占跑道需授权）、advisory-priority（RA 期间禁新指令） |
| 优先权 | $\lnot\alpha \text{ W } \beta$ | clearance-precedence（降落/进近需预先许可） |
| 待决响应 | $\mathbf{G}(\alpha \to \lnot\mathbf{X}\lnot(\lnot\alpha \text{ W } \beta))$ | repeated-instructions（同一指令重复前须有读回） |
| 有界禁止 | $\mathbf{G}(\alpha \to \lnot\mathbf{F}_{\leq\delta}\beta)$ | advisory-direction（RA climb 后 δ 秒内禁 descend） |

**流水线架构**（四阶段）：
1. **采集（Acquisition）**：无线电音频 + ADS-B 监视流，模块化可接入更多信号源。
2. **原子提取（Atom Extraction）**：ASR（faster-whisper，ATC 微调）→ 词法语义解析（LLM，JSON schema 约束）；监视解码 → 空态/地表态原子。
3. **轨迹构建（Trace Construction）**：呼号解析（Callsign Resolution）绑定语音与航空器；按事件时间排序；增量扩展派生命题（读回匹配、授权传播、RA 重建）。
4. **验证（Verification）**：对每个家族筛选与当前轨迹中实体匹配的公式，按 $\models$ 求值；在线配置在两阶段求值之间插入虚拟空观测以检测过期时限。

**关键设计点**：
- 时限 $\delta$ 来源：RA 5s 响应来自 ICAO Doc 9863 §4.1.4.2；30s 读回窗口基于飞行员响应时间统计 [21]。
- 在线/离线双模式：离线用大模型（qwen3.8-27b），在线用小模型（qwen2.5-7b）保证亚秒延迟。
- 复杂度：最坏情况二次于轨迹长度（外层 G 遍历全轨迹，各时态算子扫描后缀）。

## 实验与结果
**数据集**：
- **ATCO2**：公开一小时测试集，七座欧美机场，含人工转录，AADSB 状态从 OpenSky Network 对齐获取。
- **TartanAviation**：KAGC（有塔台）同步无线电+ADS-B 轨迹；选取 2022 年三天的最忙时段共 3 小时，人工盲注 12 个违规（主要为读回错误），作为 ground truth。

**合成验证**：255 个触发事件扩展为 1,495 个合成场景（664 合规 + 831 违规），覆盖 50 种公式实例、5 个家族；**监测器在所有场景返回预期判决（100% 正确率）**。

**真实流量评测**（Table II）：
| 解析器 | Precision | Recall | F1 |
|---|---|---|---|
| qwen3.8-27b, NT | 0.79 | 0.92 | **0.85** |
| qwen3.8-27b, T | 0.78 | 0.58 | 0.67 |
| gemma4-26b-a4b, NT | 0.40 | 0.33 | 0.36 |
| gemma4-26b-a4b, T | 0.67 | 0.33 | 0.44 |
| qwen2.5-7b, NT | 0.57 | 0.67 | 0.62 |

**最强结果**：qwen3.8-27b no-thinking 达到 F1=0.85，Recall=0.92。

**计算成本**（Table III）：RTFx（音频/处理时长）32–48；大 LLM 解析中位 1.58s/utterance；公式求值中位 0.119s/event。

**历史事故重建**：
- **Uberlingen（2002）**：21:35:01 检测 advisory-direction 违规（距碰撞 31 秒前）；21:35:03 检测 advisory-priority 违规（距碰撞 29 秒前）；与 BFU 报告一致。
- **Comair 5191（2006）**：06:05:15 呼号错误；06:06:16 wrong-runway-takeoff 触发（距接地 19 秒前）；与 NTSB 报告一致。

**词汇覆盖率**：两个语料库共覆盖 37.6% 例行程序原子模式（排除紧急/入侵等稀有事件）。

## 相关工作脉络
1. **Reynolds & Hansman [41]**：将观测轨迹与活跃许可的预期轨迹对比；仅部分满足 R₃（有限时序），R₁/R₂/R₄ 未涉及；本文在其基础上增加了语音接地与可解释违规归因。
2. **Helmke et al. [23]**（HAAWAII）：基于共享本体检测读回错误；满足 R₁ 但缺少监视融合（R₂ 未满足）和违规义务归因（R₄ 部分满足）；本文的多源融合与义务链接超越其能力。
3. **Lin et al. [42]**：结合语音与 ADS-B 评估 maneuvers；满足 R₁/R₂，但指令不叠加覆盖（R₃ 部分满足），警告不含违规义务（R₄ 部分满足）；本文的时序逻辑形式化使 R₃/R₄ 得到完整支持。
4. **Lima et al. [43]**（MTL RV 可解释监控）：给出公式级证明但假设轨迹已知（R₁/R₂ 缺失），解释停留在公式层面而非操作观测（R₄ 部分满足）；本文端到端从语音到证据链，覆盖全部四项需求。
5. **Maierhofer et al. [44] / Krasowski et al. [45]**：在公路/海上领域用 MTL 形式化规则并评估轨迹；仅 R₃ 满足，无语音接地与多源融合；本文将其方法论移植至 ATC  spoken 场景并扩展为五模式家族。
6. **Lall & Liu [46]**：用 LLM prompt 评估海事训练合规性；仅部分满足 R₁（链接到 checklist item 而非实体），R₂/R₃ 缺失；本文的实体绑定与显式时限推理弥补这些缺陷。

## 局限性与未来方向
1. **词汇覆盖率有限**（37.6%）：当前语料仅覆盖例行程序，缺乏紧急、跑道入侵、碰撞规避等关键稀有场景的训练数据。
2. **呼号解析容错**：未确认呼号仅在 controller 与 pilot 一致使用保留，存在真实场景下不完整呼号的解析损失。
3. **时间戳对齐**：ASR 识别延迟已被处理，但 ADS-B 更新速率与语音采集系统的时钟同步未详细讨论。
4. **在线部署验证**：目前离线与在线架构相同，但实际空管工作流集成、误报呈现策略和 human-in-the-loop 校准尚待验证。
5. **论文自述未来方向**：扩展监控公式覆盖本地程序惯例；在更广泛的交通语料库上验证；探索其他运输领域（海事、公路）的适用性。

## 研究启发与可借鉴点
1. **五模式公式模板可作为领域迁移的蓝图**： bounded-response / invariant / precedence / pending-response / bounded-prohibition 五种模式具有高复用性，可直接适配铁路、公路、海运等具有指令-响应结构的运行监控场景。
2. **三值运行时语义 + 双阶段求值策略**值得借鉴：在线模式下比较"|=" 与"将未过期时限视为满足"两种求值结果，可避免过早误报，同时保证时限到期后给出确定裁决。
3. **呼号解析（实体接地）的严格/宽松双重策略**：对于 ASR 输出中不完整或缩写呼号，本文给出保留一致性判据，为低资源 NLP 后的实体链接提供了实用的工程方案。
4. **合成场景构造方法**：从真实触发事件出发，通过可控扰动（删响应、超时限 0.5s、无授权执行）生成正负样本，有效解决了安全关键场景中违规样本稀缺的问题。
5. **LLM 作为结构化提取器 + JSON Schema 约束**：采用 llama.cpp + greedy decoding (temperature=0) + 固定 seed，将非结构化语音转录转化为形式化原子命题，该范式可复用于其他需要 NLP→形式化映射的场景。

## 关键术语表
**Runtime Verification (RV)**：在系统执行过程中实时或离线检查观测轨迹是否满足形式化规范的验证方法，支持三值语义。
**LTL_f（有限轨迹线性时序逻辑）**：LTL 的有限轨迹版本，公式在有序观察序列的最后一个位置有明确的终止语义。
**Metric Temporal Logic (MTL)**：在时序逻辑中引入显式时间界限算子（如 $\mathbf{F}_{\leq\delta}$），用于表达限时义务。
**Readback（读回）**：飞行员复述管制指令的关键部分以确认理解，是 ATC 程序的核心安全机制。
**ADS-B（广播式自动相关监视）**：飞机自动广播其 GPS 定位、高度、速度等信息的地面接收监视技术。
**TCAS/RA（交通警戒与防撞系统/决断咨询）**：机载防撞系统独立于 ATC 发出的 climb/descend 指令，具有优先于管制指令的效力。
**Callsign Resolution（呼号解析）**：将语音中出现的呼号（可能为缩写或不完整形式）匹配到 ADS-B 标签中的完整航空器标识的过程。
**Three-valued Semantics（三值语义）**：在线监控中，公式可处于 true（必然满足）、false（必然违反）或 inconclusive（尚未确定）三种状态之一。

## 可复现要素
- **数据集**：ATCO2（公开）、TartanAviation（公开）、OpenSky Network ADS-B 数据（公开）；ground truth 标注基于 TartanAviation KAGC 三天数据手工标注，论文未公开具体标注文件。
- **代码**：论文未公开代码仓库；各组件基于公开工具（faster-whisper、llama.cpp、Shapely、OpenStreetMap）。
- **关键超参**：读回时限 δ=30s；RA 响应时限 δ=5s（ICAO 规定）；parser 温度=0，greedy decoding，固定 seed；在线配置使用 qwen2.5-7b，离线配置使用 qwen3.8-27b。
- **硬件**：NVIDIA RTX 4090，AMD Ryzen 9 7900，64GB RAM，Fedora Linux 44。
