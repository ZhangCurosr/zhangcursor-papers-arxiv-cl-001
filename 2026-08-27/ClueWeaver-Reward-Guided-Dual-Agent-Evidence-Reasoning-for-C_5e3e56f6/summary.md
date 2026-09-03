---
title: "ClueWeaver-Reward-Guided-Dual-Agent-Evidence-Reasoning-for-C"
source: https://arxiv.org/pdf/2608.25531v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:31:08"
field: "长叙事理解与可解释QA"
keywords: ["长上下文问答", "证据选择", "双Agent框架", "奖励引导强化学习", "紧凑语言模型", "可解释推理"]
innovations: ["将长叙事QA分解为Finder（证据选择）与Interpreter（接地解释）双Agent流水线，并显式化证据选择过程", "针对Finder与Interpreter分别设计任务特定规则奖励，使用GRPO进行奖励引导的强化学习训练", "在紧凑型本地模型（Qwen3-4B）上实现显著性能提升，超过更大规模本地模型并接近API模型，同时提供可追溯的段落引用推理链"]
benchmarks: ["DetectiveQA", "∞Bench", "LongBench v2", "NoCha"]
---

# 论文速读：ClueWeaver-Reward-Guided-Dual-Agent-Evidence-Reasoning-for-C

## 一句话总结
ClueWeaver 将长篇叙事问答拆分为“线索选取（Finder）”与“证据解释（Interpreter）”两个可独立训练的 agent，并通过奖励引导的强化学习（GRPO）优化两者，使紧凑型本地模型在长叙事 QA 任务上大幅超越直接端到端提示，且推理路径可追溯、可检查。

## 研究问题与动机
- **核心问题**：在人文学科与长叙事阅读中，答案线索往往稀疏且分散于数百段文本，紧凑型本地模型难以在单次长上下文窗口内可靠定位并利用这些线索。
- **现有方法不足**：
  1. 直接全量上下文提示将证据选择隐式化，模型易遗漏关键线索或被无关细节干扰。
  2. 传统 RAG 及其变体（如 IRCoT、Self-Ask）主要面向开放领域检索，而非在已给定叙事内显式选取与保留稀疏线索。
  3. 现有长上下文基准评估侧重最终答案，对线索发现与证据保留过程缺乏关注。
  4. 单阶段读者无法区分“证据遗漏”与“推理错误”，难以定位失败原因。

## 核心贡献（创新点）
1. **提出证据感知双 Agent 流水线**：将长叙事 QA 显式分解为证据选择与自我校准解释两个阶段，使证据定位与推理过程可检查、可追溯。
2. **奖励引导的强化学习训练框架**：针对 Finder 和 Interpreter 设计任务特定的规则奖励（高召回证据保留、忠实段落引用、正确且简洁的解释），使用 GRPO 优化两者，对齐中间证据决策与最终任务目标。
3. **在紧凑本地模型上实现显著性能提升**：基于 Qwen3-4B-Instruct 的 ClueWeaver 在多个长上下文叙事基准上优于更大规模的本地端到端模型（最高提升 +14.5 分），并接近商业 API 模型性能，同时提供可追溯的证据路径。

## 方法详解
- **整体流程**：给定叙事 $X = \{p_1, p_2, ..., p_m\}$ 与问题 $q$，首先通过检索引导分割生成候选片段集 $S$；Finder 逐段判断是否包含答案相关线索，输出 YES/NO 决策、引用的段落 ID 与简要理由；选中片段按原始叙事顺序打包为证据包 $E$；Interpreter 基于 $E$ 生成最终答案与 grounded 解释，对高风险问题触发内部自我校准（self-cal）进行一致性检查。
- **Finder 设计**：采用 lexical + dense（BGE-M3）检索定位锚点段落，扩展为局部窗口构成片段；目标是**高召回**，避免遗漏答案支撑证据；奖励 $R_F$ 包含格式、决策正确性、段落引用 F1、紧凑性、否定合理性等加权项，鼓励保留关键证据而非激进过滤。
- **Interpreter 设计**：输入为有序证据包 $E$，输出 XML 结构化解析理由与答案；奖励 $R_I$ 以答案正确性为主导，辅以段落引用奖励、解释简洁性、具体接地信号及“hedging”惩罚，鼓励保守、可追溯的推理。
- **训练方法**：两 Agent 均使用 GRPO（Group Relative Policy Optimization）进行强化学习优化，采样 $K$ 个输出后计算组内相对优势；不使用值模型，KL 正则化系数 $\beta=0.04$；训练数据来自基准训练集，按段监督 Finder，按问答对监督 Interpreter。
- **关键超参数**：Finder 与 Interpreter 共享 Qwen3-4B-Instruct 基座；LR 分别为 $5\times10^{-7}$ 与 $3\times10^{-7}$；训练样本数 Finder 12K、Interpreter 1K；max_input_len=9216，max_output_len=256。

## 实验与结果
- **数据集**：DetectiveQA（稀疏侦探情节线索）、∞Bench、LongBench v2（长上下文推理）、NoCha（小说长度主张验证）。
- **基线**：端到端本地模型（Qwen3-4B/8B/30B-A3B、Ministral-3-14B、GPT-OSS-20B、Gemma-4-31B-it）及 API 模型（Claude Haiku 4.5、GPT-5 nano）；Agent 基线包括 ReAct、IRCoT、Self-Ask、Chain-of-Agents、RAG-DDR。
- **主要结果**：
  - ClueWeaver（Qwen3-4B）**总体准确率 59.0%**，优于最强本地基线 IRCoT（52.6%）+6.4 分，比直接端到端 4B 模型（44.5%）提升 **+14.5 分**。
  - 在 DetectiveQA（55.8%）、∞Bench（63.8%）、LongBench v2（50.0%）、NoCha（61.3%）四项上均领先本地方法。
  - 与最强 API 模型（GPT-5 nano 61.9%、Claude Haiku 4.5 63.9%）相比仅落后 4.9 分，并在 LongBench v2 上反超 11.5 分。
- **消融实验**：
  - 移除 Interpreter_self-cal 导致准确率下降 4.8 分；移除 Finder 下降 5.8 分；两者皆移除退化为直接端到端（36.5%）。
  - 移除 Finder RL 损失最大（-6.8 分），说明 Finder 训练对证据保留至关重要；移除 Interpreter RL 损失 1.0 分。
- **效率与成本**：单 GPU（Qwen3-4B）推理耗时 8.6–9.8 秒/题（ vs. 直接阅读 2.8 秒）；仅需 24GB 显存（RTX 4090/5090），远低于 30B 级模型所需 80GB。

## 相关工作脉络
- **RAG 及其变体（IRCoT、Self-Ask、Chain-of-Agents、RAG-DDR）**：聚焦开放域检索或多跳 QA，依赖外部文档；ClueWeaver 面向**已给定叙事**，核心任务是保留稀疏线索而非引入新检索原语。
- **长上下文基准（LongBench、∞Bench、NarrativeQA、QuALITY）**：评估最终答案，忽视线索发现与证据保留过程；本文明确将**证据选择显式化、可检查**作为设计目标。
- **奖励引导强化学习（DeepSeekMath、GRPO）**：此前多用于数学推理或通用指令跟随；本文将其适配至**证据选取与段落引用**这一细粒度任务，设计了专用规则奖励。
- **自我校准机制**：类似 Self-RAG 的自我批判；本文将校准内置于 Interpreter，避免引入第三 Agent，保持轻量化。
- **定位差异**：ClueWeaver 不是端到端压缩或全量检索，而是**检索引导分割 → 高召回证据选取 → 证据接地推理**的三阶段流水线，强调中间过程的透明度与可干预性。

## 局限性与未来方向
- **局限**：
  1. 当答案依赖极远距离的多跳线索或表面重叠极弱的段落时，仍可能失效（见案例研究中的 Unresolved case）。
  2. 检索分割依赖 BGE-M3 与词法匹配，对非英文或领域差异较大的文本泛化性待验证。
  3. 自我校准仅在 Interpreter 内部触发，未探索跨 Agent 的交叉验证。
- **未来方向**：增强远距离多跳线索的检索与整合能力；扩展至更多样化的长叙事类型（如剧本、档案）；探索更轻量或自适应的校准触发机制。

## 研究启发与可借鉴点
1. **奖励引导的 GRPO 可迁移至证据选择任务**：设计细粒度规则奖励（格式、决策、引用、紧凑性）能有效对齐中间步骤与最终目标，适用于任何需显式选择依据的流水线。
2. **XML 结构化输出提升可追溯性**：强制要求模型输出段落 ID 引用，便于事后审计与错误定位，可作为类似“可解释推理”任务的通用设计。
3. **自我校准作为内部 pass 而非独立 Agent**：在高风险问题上复用同一 Interpreter 进行一致性检查，既提升可靠性又避免系统复杂化，适合资源受限场景。
4. **高召回优先的证据筛选策略**：Finder 奖励权重偏向保留证据而非过滤，符合“Interpreter 只能推理已保留证据”的流水线逻辑，启示我们在多阶段系统中应针对各阶段角色定制奖励。
5. **训练数据分布对齐测试分布**：Interpreter 训练集采用 27.5% 基座模型错题，集中奖励于难例，可提升训练效率，值得在 RL 微调中借鉴。

## 关键术语表
- **Finder**：双 Agent 中的证据选择模块，负责判定每个叙事片段是否包含答案相关线索，并输出 YES/NO 决策与段落引用。
- **Interpreter**：双 Agent 中的解释模块，基于 Finder 选出的证据包生成最终答案与接地推理，必要时进行自我校准。
- **GRPO（Group Relative Policy Optimization）**：基于组内相对优势的强化学习算法，无需值模型，通过奖励归一化计算优势并更新策略。
- **奖励引导强化学习（Reward-Guided RL）**：使用手工设计的规则奖励（非学习到的奖励模型）引导 RL 训练，使模型行为对齐任务特定目标。
- **自我校准（Self-Calibration）**：Interpreter 在检测到高风险问题时，使用同一证据包重新验证 provisional answer 的内部校准步骤。
- **段落 ID 引用（Paragraph-ID Referencing）**：要求模型在理由中显式标注证据来源的段落编号，确保推理可追溯至原文。
- **证据包（Evidence Packet）**：由 Finder 选中的片段按原始叙事顺序打包而成的紧凑证据集合，作为 Interpreter 的输入。
- **长叙事 QA（Long-Narrative QA）**：在小说、剧本、档案等长篇叙事文本上回答需要稀疏线索与因果推理的问题的任务。

## 可复现要素
- **数据集**：DetectiveQA、∞Bench、LongBench v2、NoCha 均为公开基准。
- **代码/权重**：论文未明确声明开源，但基座模型 Qwen3-4B-Instruct 已公开；BGE-M3 嵌入模型亦为开源。
- **关键超参**：GRPO β=0.04、采样数 K=8、max_input_len=9216、max_output_len=256；Finder LR=5e-7、样本 12K、步骤 150；Interpreter LR=3e-7、样本 1K、步骤 100；训练硬件 8×A100 GPU。
- **实现细节**：证据打包参数 N_E、P_r、P_w、B_c 因数据集而异（见 Appendix C.2）；检索使用 BGE-M3；输出强制 XML 格式（<reason> 与 <answer> 标签）。
