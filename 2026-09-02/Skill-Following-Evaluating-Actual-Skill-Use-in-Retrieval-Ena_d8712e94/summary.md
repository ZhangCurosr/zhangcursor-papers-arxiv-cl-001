---
title: "Skill-Following-Evaluating-Actual-Skill-Use-in-Retrieval-Ena"
source: https://arxiv.org/pdf/2609.00549v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:42:20"
field: "LLM Agent 评估与工具使用"
keywords: ["LLM Agent", "Tool Use Evaluation", "Retrieval-Augmented", "Skill Following", "Paired Evaluation", "Selection Bias"]
innovations: ["提出RAE指标，在检索触发的同任务配对比较中消除选择偏差，精确测量技能使用的真实因果效应", "揭示聚合检索收益与RAE之间普遍存在的符号反转悖论，证明强聚合指标可能掩盖负面实际效果"]
benchmarks: ["MBPP+", "HumanEval+", "Math500"]
---

# 论文速读：Skill-Following-Evaluating-Actual-Skill-Use-in-Retrieval-Ena

## 一句话总结
本文提出了 RAE（Retrieval-Invoked Actual-Use Effect）指标，通过在**同一任务**上严格配对技能启用（SE）与技能禁用（SD）执行，消除任务选择偏差，精确衡量 LLM 智能体在实际调用检索后是否真正从技能注入中受益。对 17 个 LLM 在编程和数学领域的评测揭示了一个普遍存在的"评估悖论"：大量模型呈现正向的聚合检索收益，却在同一任务的配对比较中出现负向的 RAE，表明现有聚合指标可能营造出工具使用能力的虚假繁荣。

## 研究问题与动机
1. **核心问题**：当 LLM agent 主动检索并注入技能后，该技能调用链是否真正改善了该具体任务的最终答案质量？现有评估无法回答这个"因果效用"问题。
2. **聚合指标的误导性**：整体基准平均准确率被大量未检索任务稀释，掩盖了检索-答案链条的真实效果；self-selected 的检索/非检索任务子集之间存在系统性难度差异，导致选择偏差。
3. **检索≠遵循≠利用**：LLM 能成功检索到技能文本，但后续还需正确解读、对齐、整合并格式化输出，现有评估缺少对这一完整链路的隔离测量。
4. **对工具能力高估的风险**：若仅凭聚合指标判断 agent 能力，可能导致对实际部署系统的性能过度乐观，进而误导工程决策与审计。

## 核心贡献（创新点）
1. **形式化 Skill Following（SF）链路并揭示评估缺口**：将技能整合定义为"检索→注入→解释→推理→综合"的完整认知链，数学化证明聚合指标因选择偏差无法隔离技能的真实任务级因果影响。
2. **提出 RAE 指标与配对评估协议**：仅以 SE 执行中实际检索并返回技能的子集 $S_{\text{call}}$ 为条件，计算相同任务上 SE vs SD 的二元结果差异，彻底消除跨任务难度混淆。
3. **跨 17 个 LLM 揭示普遍的"指标反转"现象**：在 MBPP+、HumanEval+ 和 Math500 三个基准上，多个模型呈现"聚合收益为正但 RAE 为负"的悖论（如 Gemini-2.5-Flash-lite 在 MBPP+ s42 上聚合 +6.2pp 而 RAE 为 −15.6pp），证明上下文暴露远不足以保障有效技能遵循。
4. **机制诊断与内容控制实验**：通过 GPT-5.5 行为标注（四种失败模式）和技能内容扰动（Normal/Schema-Empty/Filler-Dummy/Random-skills/Corrupted），证明负面 RAE 并非单纯由检索内容质量差导致，而是多阶段链条的综合失败。

## 方法详解
**1. 配对执行设计（Paired Execution Design）**
每个任务 $t$ 经历两条路径：SE（完整技能库 + `search_skills` 工具）与 SD（移除工具定义，其余 prompt、seed、解码配置完全一致）。二元结果 $y_t^+, y_t^- \in \{0,1\}$ 映射为四类转移：
- **Helpful flip (b)**：SD 错 → SE 对（技能带来收益）
- **Harmful flip (c)**：SD 对 → SE 错（技能带来损害）
- **Concordant pass/fail**：双方一致正确/错误

**2. 核心指标公式**
- **OAE（Overall Skill-Access Effect）**：全量任务集合 $S_{\text{all}}$ 上的配对差异均值：
$$\text{OAE} = \frac{1}{|S_{\text{all}}|}\sum_{t \in S_{\text{all}}}(y_t^+ - y_t^-) = \frac{\text{all}_b - \text{all}_c}{n_{\text{pairs}}}$$

- **RAE（Retrieval-Invoked Actual-Use Effect）**：仅以检索被触发的子集 $S_{\text{call}}$ 为条件：
$$\text{RAE} = \frac{1}{|S_{\text{call}}|}\sum_{t \in S_{\text{call}}}(y_t^+ - y_t^-) = \frac{\text{cond}_b - \text{cond}_c}{n_{\text{cond}}}$$

- **分解关系**：$\text{OAE} = \frac{n_{\text{cond}}}{n_{\text{pairs}}}\text{RAE} + \frac{n_{\text{skip}}}{n_{\text{pairs}}}\Delta_{\text{skip}}$，说明 OAE 是 RAE 与跳过子集效应的加权组合。

**3. 技能环境与检索机制**
- 编程技能池：9 个 procedural skills（binary-search-boundary, counting-hash, dfs-iterative, edge-case-guard, knapsack-01-dp, prefix-sum, rotate-left, sliding-window-longest, two-pointer-pair-sum）。
- 数学技能池：8 个 skills（algebraic-substitution, case-analysis, check-by-substitution, coordinate-distance, factor-then-solve, modular-arithmetic, sequence-formulae, systematic-enumeration）。
- 检索：BM25 索引，每次最多 3 次调用，返回 top-3 技能全文（Markdown SKILL.md 格式）。

**4. 诊断控制实验**
- **技能内容控制**：五种扰动（Normal / Schema-Empty / Filler-Dummy / Random-skills / Corrupted），隔离内容质量对 RAE 的影响。
- **遵循标注**：用 GPT-5.5 对有害翻转（harmful transitions）进行 5 类行为标注（appropriate adherence / ignored-or-independent / misapplied-or-overapplied / format-or-interface-failure / unclear），并由 9 位人工标注者进行分层审计验证。

## 实验与结果
**数据集**：MBPP+（80 任务，3 个 seed 分区）、HumanEval+（80 任务）、Math500（80 任务）；共 17 个 LLM（涵盖 8B 至 235B 参数，闭源 API 与开源权重）。

**主要结果（MBPP+ seed 42）**：

| 模型 | 聚合检索收益 | RAE | 反转 |
|---|---|---|---|
| DeepSeek-V3.2 | +31.6pp | −9.1pp | ✅ |
| Gemini-2.5-Flash-lite | +6.2pp | −15.6pp | ✅ |
| Llama-3.3-70B | +69.8pp | +4.2pp | ❌ |
| Qwen3-235B | −0.1pp | +7.0pp | ✅（方向相反） |
| Claude-4.5-H | +1.4pp | +1.4pp | ❌ |

- **MBPP+ 上 5/13 报告单元出现符号反转，HumanEval+ 上 3/13，Math500 上 Llama-3.3-70B 呈现极端反转（+14.2pp 聚合 vs −39.4pp RAE）。**
- 高检索覆盖率不保证正向 RAE：DeepSeek-V3.2 检索覆盖率 97.1%，但 RAE 为 −15.9pp（pooled over seeds）。
- 内容控制结果（Table 4）：两模型的 RAE 在所有非空返回条件下均保持负值（DeepSeek-V3.2：Normal −15.9, Filler −10.2, Random −15.1, Corrupted −9.1；Gemini：Normal −17.9, Filler −5.0, Random −15.4, Corrupted −17.2），证明负面 RAE 并非内容质量问题所致。
- 行为标注显示有害翻转中最大比例为"ignored/independent"（DeepSeek 64.4%，Gemini 72.5%），说明多数情况下检索到的技能被模型忽略。
- **最强 RAE 正向结果**：Claude-4.5-H 在部分 seed 下 RAE ≈ +2.7pp（MBPP+ s44）；Qwen3-235B 在 HumanEval+ s42 上 RAE = +15.5pp。

## 相关工作脉络
1. **SkillsBench / SWE-Skills-Bench / Skill-LearnBench**：同样采用配对设计，但效应仍聚合于全量评测集；本文 RAE 将其进一步限定于检索被实际触发的条件子集，实现更精确的因果隔离。
2. **AgentBench / AgentBoard / WebArena / OS-World / τ-bench / TheAgentCompany**：衡量端到端 agent 系统级行为（整体成功率/步骤诊断），不隔离"注入外部内容是否直接改变该任务最终结果"这一配对因果问题。
3. **Voyager / Agent KB / ExpeL / Reflexion / CRITIC**：展示 agent 自主使用和保留外部制品的能力，但缺乏将每次检索调用与对应 SD 基线严格配对的设计来测量净性能迁移。
4. **BFCL / RAGAS / IFEval**：诊断 tool calling、检索质量、grounding 等子阶段指标；RAE 与之互补而非替代，关注检索-答案全链路的端到端因果效应。
5. **Aggregate retrieval lift 的隐含问题**：现有工具使用评估常以检索 vs 非检索子集的均值差异为指标（非配对），本文通过 $\Delta_{\text{select}}$（SD 基线下的选择缺口）数学化揭示其选择偏差本质。

## 局限性与未来方向
1. **任务类型局限**：仅在单轮编码/数学任务上验证，结论不可直接推广至长 horizon agent、可执行代码库、自然语言标注类任务或带迭代记忆规划的复杂场景。
2. **条件因果的边界**：RAE 基于 SE 执行自选择的 $S_{\text{call}}$，并非预处理前的无偏因果估计，也不是模型级别的技能使用能力分数——它是协议条件性的配对结果信号。
3. **未做的消融**：缺少 oracle 检索、仅元数据检索、强制全技能获取、多技能池构建、人工裁定因果标签等更细粒度的因果分解。
4. **未来方向**：探索使检索更贴合任务难度的 skill placement 策略；研究如何改善模型对检索内容的整合遵循（adherence）能力；开发跨域稳定的 skill-use 能力排名指标。

## 研究启发与可借鉴点
1. **配对同任务评估范式可迁移**：任何涉及"外部知识/工具注入是否真正有用"的评估场景（RAG 质量、plugin 调用、fine-tuning 后工具适配）均可复用 SE/SD 配对 + 条件子集分析的设计，消除选择偏差。
2. **"检索覆盖率 ≠ 技能利用效能"的量化启示**：Stage-diagnostic 四个代理指标（Invocation / Retrieval / Adherence / Effective）构成一套层次化能力解剖框架，可直接移植到本团队 agent 评估流水线中作为分层诊断工具。
3. **内容扰动控制实验设计**：Schema-Empty / Filler-Dummy / Random-skills / Corrupted 五类扰动可复用于检验任何"外部知识注入"场景中的内容依赖程度，区分模型能力与内容质量的贡献。
4. **LLM-as-annotator 的有害翻转归因**：用 GPT-5.5 对 paired outcome transitions 进行行为标注（blinded to correctness）并辅以人工审计的策略，是本团队可参考的自动化诊断范式。
5. **可探索的创新机会**：将 RAE 思想引入多跳 RAG 评估——不仅看检索召回质量，更看注入后答案变化的配对因果效应；或在 tool learning 训练中引入 RAE 作为训练信号，直接优化检索→整合→输出的完整链路。

## 关键术语表
**Skill Following (SF)**：从检索触发到最终答案生成的完整认知链（检索→注入→解释→推理→综合），本文将其形式化为评估工具使用真实效用的目标。
**RAE（Retrieval-Invoked Actual-Use Effect）**：在 SE 执行中实际检索并返回技能的子集上，配对计算 SE vs SD 的二元结果差异均值，消除任务选择偏差的同任务因果指标。
**OAE（Overall Skill-Access Effect）**：在全量任务集上计算的 SE vs SD 配对结果差异均值，反映技能环境的系统级整体影响。
**Aggregate Retrieval Lift**：传统指标，比较 SE 执行中检索返回任务与非检索任务的均值准确率差异，存在严重的选择偏差。
**Helpful/Harmful Flip**：配对转移分类；helpful flip（b）指 SD 错误但 SE 正确，harmful flip（c）指 SD 正确但 SE 错误。
**Selection Gap（$\Delta_{\text{select}}$）**：用 SD 基线准确率衡量的检索任务子集与非检索任务子集之间的内在难度差异，量化选择偏差的程度。
**Adherence（遵循度）**：SE 执行最终答案中是否出现检索技能标识符或代码模式等结构性痕迹的代理指标。
**Effective Rate**：在检索被触发的任务中，同时满足 adherence 条件且最终答案正确的比例。

## 可复现要素
- **数据集**：MBPP+、HumanEval+、Math500（均为公开基准，论文已引用来源）。
- **代码/权重**：论文未明确声明代码开源仓库链接（附录含完整实验细节与 skill pool 列表），模型均通过 OpenAI-compatible API 接口访问。
- **关键超参**：temperature=0.7，max output length=8192，reasoning disabled，每任务最多 3 次 `search_skills` 调用，BM25 返回 top-3，seed=42/43/44。
- **评估脚本/annotator**：GPT-5.5 作为行为标注器（blinded），9 位人工标注者完成分层审计（50 个有害翻转样本）。
