---
title: "SocialReasonBench-A-Video-QA-Benchmark-for-Social-Reasoning"
source: https://arxiv.org/pdf/2608.30716v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:51:10"
field: "多模态社会推理评测"
keywords: ["video reasoning", "social reasoning benchmark", "counterfactual reasoning", "large multimodal models", "interactive narrative games", "diagnostic distractor", "causal antecedent"]
innovations: ["基于互动叙事游戏分支结构构建可验证的反事实与因果推理视频QA基准", "理论驱动的七维社会推理分类体系与五种诊断陷阱设计", "多智能体自动化策展流水线结合游戏状态信号接地"]
benchmarks: ["SocialReasonBench"]
---

# 论文速读：SocialReasonBench: A Video-QA Benchmark for Social Reasoning with Counterfactual Narrative Videos

## 一句话总结
论文提出了 **SocialReasonBench**，一个基于互动叙事游戏《Detroit: Become Human》的视频多选QA基准，用于评估大多模态模型（LMMs）在社会推理上的能力；它利用游戏分支剧情结构构建可验证的反事实与因果推理场景，并设计了7个细粒度推理维度的诊断性干扰项。

---

## 研究问题与动机

1. **现有视频社交推理基准的局限**：已有基准多基于单一轨迹的影视或短视频（如 TVQA、Social-IQ），无法判断模型是否真正理解社会动态，还是仅仅利用了重复的叙事模式。
2. **缺乏可验证的反事实社会推理评估**：静态视频只呈现一条实现的故事线，难以测试模型对"替代决策如何重塑社会后果"的推理能力（Pearl 因果层次中的第3层）。
3. ** ground truth 的可靠性问题**：现有基准依赖人工标注，成本高且易引入主观解释偏差；本文希望利用游戏自身状态信号（like affinity scores、flowchart）提供可核查的标签。
4. **多模态融合必要性未被充分检验**：许多模型可能仅依赖视觉捷径而非真正的多模态社会推理，需要设计能强制整合视听信息的评测场景。

---

## 核心贡献（创新点）

1. **提出 SocialReasonBench 基准**：基于互动叙事游戏分支结构构建的 532 个视频QA实例，首次系统性地覆盖反事实推理与因果前件推理两大高难度社会推理维度——与 TVQA/Social-IQ 等线性叙事基准的本质区别在于"替代结局可观察且可验证"。
2. **设计多智能体自动化策展流水线**：由 Director Agent（片段定位）、Tracker Agent（游戏信号接地）、Generator Agent（QA合成）协同完成从原始游戏视频到标注QA对的全流程生成——与纯人工标注基准相比，标注成本大幅降低且答案可由游戏内流图交叉验证。
3. **理论驱动的七维推理分类体系**：基于社会认知理论（Theory of Mind、Attribution Theory、Structural Causal Models 等）构建 $\mathcal{W} = \mathcal{I} \times \mathcal{R}$ 二维分类轴，每个维度均有明确心理学/哲学理论基础——与以往经验性分类相比，理论锚定使错误分析更具诊断价值。
4. **引入五种诊断性干扰项类型与 Trap Fall Rate (TFR) 指标**：Visual/Logic/Interpretation/Knowledge/Attribution 陷阱分别对应视觉启发、因果幻觉、社会过度解读、先验污染、根本归因错误——与仅报告准确率的工作相比，能揭示模型"为何犯错"而不仅是"是否犯错"。

---

## 方法详解

### 数据源与环境抽象
- 以 Quantic Dream 的互动叙事游戏 **Detroit: Become Human** 为基础，该游戏有 32 章、分支叙事图 $\mathcal{G} = (\mathcal{S}, \mathcal{A}, \mathcal{T})$，其中状态 $s_t$、行动 $a_t$ 与确定性转移函数 $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$ 构成可追踪的剧情因果链。
- 每条实现轨迹 $\tau = (s_1, a_1, \ldots, s_T)$ 伴随未实现的分支边，后者即为反事实推理的来源。

### 多智能体合成框架（图3）
1. **Director Agent（导演智能体）**：
   - 输入：章节级流程图 $\mathcal{G}$ + 全章通关视频与脚本。
   - 定位两类关键叙事节点：(i) 导致不同下游轨迹的分支选择点；(ii) 游戏变量显式变化的时刻（亲和力 shift、关系更新）。
   - 对每个候选节点分配二维标签 $(i, r) \in \mathcal{I} \times \mathcal{R}$，并在目标节点前后各填充 ±15s 上下文缓冲区，得到最终片段 $v_i$（中位时长 85s，IQR 70–115s）。
2. **Tracker Agent（追踪智能体）**：
   - 从两类信号源提取隐藏游戏状态 $z_i$：
     - **显式信号**：界面可见的 UI 更新（如 [Software Instability ↑]、[Probability of Success +7%]）。
     - **隐式信号**：通过比对已执行行动与流程图分支条件推导出的状态（如"某拒绝分支仅在低信任条件下可达 → 记录 low-trust"）。
   - 定义七类状态变量体系（Table 7）：system_metric、probability、relationship、life_state、approach_modifier、generic_relationship、other。
3. **Generator Agent（生成智能体）**：
   - 基于 $(v_i, z_i, i, r)$ 三元组合成 QA 对。
   - **实体匿名化**：构建 155 条替换规则（Table 5），将 "Connor → Agent Alpha"、"CyberLife → OmniCorp"、"android → synthetic agent" 等，防止模型利用 franchise 先验记忆。
   - 正确答案 $y_i$ 必须被 $z_i$ 蕴含；五个错误选项按五种诊断陷阱类型构造（Table 4）。
   - 选项初始映射固定（O_A 为正确，O_B–O_F 对应各陷阱），最终评估时通过 SHA-256 种子随机打乱顺序。

### 标签接地层次（Table 10 / Appendix J）
- **Tier A（Explicit UI）**：19.4%，直接从 clip 内可见 UI 数字/符号读取。
- **Tier B（Flowchart-documented）**：27.8%，依赖流程图分支前提，不直接可见于 UI。
- **Tier AB（Mixed）**：9.2%，UI + 流程图联合验证。
- **Tier C（Script/branch-derived）**：43.6%，需短程推导，含意图/情感推断等隐式状态。
- 综合：56.4% 可直接核查，43.6% 需推导。

### 人类验证
- 从 730 候选片段开始，剔除重复分支 replay 与高度相似 clip。
- 三项检查：(i) clip–taxonomy 对齐；(ii) video–QA 一致性；(iii) label validity。
- 最终 532 个 QA 实例。

---

## 实验与结果

### 评测模型（8个，zero-shot）
- 闭源：Gemini 3 Pro、Gemini 3 Flash、GPT-4o (2024-08)、Claude 3.5 Sonnet、GLM-4.6V-Flash
- 开源：Qwen3-Omni-30B、MiniCPM-o 2.6、ARC-Hunyuan-Video
- 输入方式差异：Gemini/Open-weight omni-video 使用原生视频+音频；GPT-4o/Claude/GLM 使用 32 等距采样关键帧（无音频）。

### 主要结果（Table 3）
| 维度 | HE（人工） | Gemini 3 Pro | GPT-4o | Claude 3.5 Sonnet | ARC-Hunyuan |
|---|---|---|---|---|---|
| Intent Recognition | 93.24% | 83.78% | 85.14% | 82.43% | 24.32% |
| Behavior Prediction | 88.46% | 66.67% | 74.36% | 71.79% | 41.03% |
| Emotional Empathy | 90.83% | 70.00% | 76.67% | 73.33% | 36.67% |
| Relationship Dynamics | 89.36% | 82.98% | 55.32% | 52.13% | 25.53% |
| Moral Dilemma (n=21) | 85.71% | 66.67% | 28.57% | 33.33% | 28.57% |
| Counterfactual Reasoning | 88.00% | 62.00% | 50.00% | 56.00% | 22.00% |
| Causal Antecedent | 86.32% | 70.53% | 55.79% | 58.95% | 30.53% |

- **最强模型**：Gemini 3 Pro 在多数维度领先，Intent Recognition 达 83.78%。
- **核心发现**：所有模型在基础感知维度（Intent Recognition / Behavior Prediction / Emotional Empathy）显著优于深层推理维度（Counterfactual / Causal Antecedent），经 Holm 校正后 6/8 模型差异显著（一侧 Fisher 精确检验最大 adjusted $p = 0.045$）。
- **GPT-4o 典型案例**：Intent Recognition 85.14% → Counterfactual 50.00%（降幅 35pp）→ Moral Dilemma 28.57%（降幅 57pp）。

### 模态消融（Figure 4）
- 对 Gemini 3 Pro 做音频消融（保留全视觉）：总体准确率下降约 **6%**。
- 最大降幅出现在 **Causal Antecedent** 与 **Emotional Empathy**，表明音频（语调、对话内容）对情感推断和因果归因至关重要。

### 诊断陷阱分析（Figure 5 + Table 4）
- 强闭源模型（GPT-4o、Claude 3.5 Sonnet）更易落入 **Knowledge Trap**（引入外部先验）与 **Visual Trap**（依赖显著视觉线索）。
- 小开源模型（MiniCPM-o 2.6）更多陷入 **Logic Trap**（时序依赖与分支状态追踪困难）。

### 人类基准
- 5 名作者盲评（每实例 1 次），Fleiss' κ = 0.73，平均准确率 85–93%，证明任务在给定视频上下文内原则上可答。

---

## 相关工作脉络

1. **Video Reasoning with LMMs**：Video-LLaMA、Video-ChatGPT、Video-LLaVA 扩展语言模型至视频；MVBench、Video-MME、EgoSchema、LongVideoBench 评估通用视频理解——本文定位差异：聚焦"社会情境深层推理"而非一般事件/动作识别。
2. **Multimodal Social Reasoning Evaluation**：TVQA、VLEP、VisualCOMET、Social-IQ / Social-IQ 2.0——本文定位差异：这些基准基于**固定线性叙事**，无法测试反事实与因果前件推理；SocialReasonBench 利用分支结构使替代结局可观测。
3. **Counterfactual Reasoning Evaluation**：CounterBench（语言）、ACQUIRED（真实生活视频+人工撰写反事实）——本文定位差异：ACQUIRED 的反事实为人工构想；本文的反事实来自**游戏中实际发生的替代分支**，可交叉验证。
4. **视频游戏作为 AI 评测源**：SIMA、VideoGameBench——本文定位差异：前者侧重任务完成/长程交互；本文聚焦**社交富含的分支叙事**与可核查的社会后果。
5. **Theory of Mind 评测**：MoMentS、Social Genome——本文定位差异：侧重单帧图片或短交互；本文覆盖**多步因果链与可验证的游戏状态信号**。

---

## 局限性与未来方向

1. **单一数据源**：仅基于《Detroit: Become Human》一款英文互动叙事游戏，存在文化、风格与叙事特征局限；android 主角与人类社交代理仍有一步之遥。未来可扩展至更多叙事游戏以丰富场景。
2. **策展管道模型家族偏置**：三个策展智能体均使用 Gemini 系列模型（同族系统也参与评测），可能引入 generator-family 风格优势；作者已验证排除 Gemini 后主结论不变，但未来可用多样化策展模型或纯人工子集缓解。
3. **道德困境子集过小**：仅 21 例，各模型在 Moral Dilemma 上的数字仅作参考，不足以得出关于道德推理能力的结论。
4. **输入模态不一致**：部分模型使用 32 帧无音频关键帧，另一部分使用原生视频，横向比较存在混杂——这不是基准本身的问题，而是评测协议限制。

---

## 研究启发与可借鉴点

1. **分支叙事图作为社会推理的可验证grounding源**：将游戏/模拟环境的因果图 $(\mathcal{S}, \mathcal{A}, \mathcal{T})$ 作为 label 验证依据，可有效规避人工标注的主观偏差；这一思路可迁移至任何具有状态机的交互式环境（视觉小说、文字冒险、仿真平台）。
2. **五类诊断陷阱的设计范式**：Visual/Logic/Interpretation/Knowledge/Attribution 陷阱分别对应认知科学的经典偏差（视觉启发、因果幻觉、根本归因错误等），可在其他社会推理基准中复用或扩展。
3. **实体匿名化防污染策略**：用通用占位符替换专有名词（角色名、组织名、物种概念），是应对"franchise prior contamination"的有效手段，值得在基于已知IP的评测中推广。
4. **音频消融作为多模态必要性检验**：为每个视频实例自动生成 audio-ablated 版本，使后续研究可标准化地检验音频对社会推理的贡献——这一资产构建思路可被其他多模态基准参考。
5. **理论锚定的二维分类轴**：将交互类型轴（$\mathcal{I}$）与推理能力轴（$\mathcal{R}$）正交组合，并逐一绑定社会认知理论——这种"理论先行的 taxonomy 设计"可为构建其他多维评测基准提供方法论参考。

---

## 关键术语表

**SocialReasonBench**：本文提出的视频多选QA基准，基于互动叙事游戏分支剧情评估 LMM 的七维度社会推理能力，共 532 个实例。

**Counterfactual Reasoning（反事实推理）**：推断"若采取不同行动会发生什么"的推理能力，属于 Pearl 因果层次结构的第三层（干预/反事实）。

**Causal Antecedent（因果前件）**：从已观察到的社会结果反向追溯使其成为可能的先前条件、决策或隐藏状态。

**Diagnostic Distractor（诊断性干扰项）**：针对特定认知失败模式设计的 plausible 但错误的选项，用于定位模型犯错的具体原因。

**Trap Fall Rate (TFR)**：衡量模型错误分布的指标，定义为在某陷阱类型上犯错的比例占所有错误比例的条件概率。

**Two-Axis Task Taxonomy ($\mathcal{W} = \mathcal{I} \times \mathcal{R}$)**：社交交互类型轴（亲社会/对抗/策略/规范）与推理能力轴（7种）正交构成的分类体系。

**Grounding Tier（接地层次）**：标签可从游戏证据中直接读取（Tier A/B/AB）还是需推导（Tier C）的四层分类，用于表征 label verifiability。

**Entity Anonymization（实体匿名化）**：将游戏专有名词替换为语义中性占位符（如 Connor → Agent Alpha），以防止模型利用 franchise 先验记忆走捷径。

---

## 可复现要素

- **数据集**：SocialReasonBench（532 实例）已公开，视频访问受 Dataset Terms of Use 约束；标注数据以 CC BY-NC-SA 4.0 发布。
- **代码**：多智能体策展管线 prompt、数据结构化 schema 及评估代码随论文发布（附录 L 完整 prompt 文本已开源）。
- **关键超参**：Director/Tracker Agent 使用 Gemini 3.1 Pro Preview（temperature 0.1–0.2）；Generator Agent 使用 Gemini 2.5 Pro（temperature 0.3）；评估时所有模型 temperature 设为 0.0；HTTP 429 退避策略为 30s/60s/120s/240s 最多 4 次重试。
- **输入处理**：关键帧模型使用 FFmpeg 从精确时间戳均匀采样 32 帧；原生视频模型接收完整 clip。

---
