---
title: "Skill-Following-Evaluating-Actual-Skill-Use-in-Retrieval-Ena"
source: https://arxiv.org/pdf/2609.00549v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 20:58:09"
---

# 论文速读：Skill-Following-Evaluating-Actual-Skill-Use-in-Retrieval-En

## 一句话总结
论文针对现有LLM Agent工具/技能评估中“检索提升”指标受任务自选择偏差干扰的问题，提出配对实验设计与严格限定在真实检索事件上的RAE指标，实证揭示当前主流大模型在主动检索技能后往往反而损害原任务正确率的普遍性“评估悖论”。

## 研究问题与动机
1. **聚合指标掩盖真实因果效应**：现有评估多比较“检索返回”与“未检索”任务的全局准确率差异，但这两类任务由模型自主选择，固有难度与提示复杂度不同，无法隔离技能检索对目标任务的真实贡献。
2. **“暴露≠整合”的认知断层**：检索调用仅是工具使用的起点，模型还需经历理解、接地、逻辑融合与格式化输出；现有协议缺乏对完整Skill Following链的配对约束评估。
3. **度量缺口需要形式化填补**：需要一种明确条件于实际检索事件、控制任务身份不变的同任务对比度量，才能判断检索-回答流水线是否真正拯救了更多结果而非引入干扰。

## 核心贡献（创新点）
1. **形式化Skill Following评估框架**：将技能集成定义为包含检索触发、内容注入、逻辑接地与答案合成的完整认知链，并从数学上证明标准聚合指标因自选择偏差会系统性掩盖真实任务级影响。
2. **提出RAE配对条件指标**：仅在agent实际调用检索并成功返回至少一个技能的任务子集 $S_{\mathrm{call}}$ 上，严格计算skill-enabled（SE）与skill-disabled（SD）配对执行的正确率差异，消除跨子集难度混杂。
3. **揭示跨17模型的指标反转悖论**：在编码与数学双域评测中发现，大量模型呈现“正向聚合检索提升 vs 负向RAE”的符号分歧，证明单纯暴露于检索上下文远不足以支撑有效技能利用。

## 方法详解
1. **配对执行设计**：每个任务 $t$ 并行执行两次路径：SE条件开放 `search_skills(query)` 工具，SD条件完全移除该工具定义；两者共享相同任务prompt、generation seed与解码配置，将结果方差严格归因于技能接入。
2. **四种转移分类矩阵**：根据二元正确性 $(y_t^+, y_t^-) \in \{0,1\}^2$ 将任务映射至 concordant pass、concordant fail、helpful flip（记为 $b$，SD错→SE对）与 harmful flip（记为 $c$，SD对→SE错）。
3. **OAE与RAE公式**：
   - $\mathrm{OAE} = \Delta_{\mathrm{all}} = (all_b - all_c)/n_{\mathrm{pairs}}$，覆盖全评测集 $S_{\mathrm{all}}$，反映系统级技能接入的宏观效用。
   - $\mathrm{RAE} = \Delta_{\mathrm{cond}} = (cond_b - cond_c)/n_{\mathrm{cond}}$，严格条件于 $S_{\mathrm{call}}$（SE实际检索并返回技能的子集）。
4. **指标分解关系**：$\mathrm{OAE} = \frac{n_{\mathrm{cond}}}{n_{\mathrm{pairs}}}\mathrm{RAE} + \frac{n_{\mathrm{skip}}}{n_{\mathrm{pairs}}}\Delta_{\mathrm{skip}}$，表明RAE并不取代OAE，而是剥离出“实际使用”成分的纯净信号。
5. **诊断控制与标注**：设计五类Skill-Content Control（Normal / Schema-Empty / Filler-Dummy / Random-skills / Corrupted）分离内容质量与模型整合效应；使用GPT-5.5盲标器对SE输出进行Adherence Annotation（appropriate / ignored / misapplied / format-interface failure / unclear），并辅以9人人工审计验证可靠性。

## 实验与结果
1. **数据集与技能库**：主实验采用MBPP+（80任务，seed 42/43/44）与HumanEval+；跨域验证采用Math500（80任务）。编码技能池含9个结构化Markdown技能（如 `binary-search-boundary`, `knapsack-01-dp`, `sliding-window-longest` 等），数学池含8个技能（如 `case-analysis`, `modular-arithmetic`, `sequence-formulae` 等），统一使用BM25检索Top-3全文注入上下文。
2. **评估基线**：对比Aggregate Retrieval Lift（非配对检索vs未检索子集均值差）、OAE与本文RAE；覆盖17个主流LLM（闭源API与开源权重混合）。
3. **主要数值结果**：
   - **指标反转普遍**：MBPP+ s42上13个可报告配置中5个出现正聚合提升但负RAE（如DeepSeek-V3.2: +31.6 / -9.1 pp；Gemini-2.5-Flash-lite: +6.2 / -15.6 pp）。
   - **选择偏差显著**：Figure 2显示 $\Delta_{\mathrm{select}} = \mathrm{Acc}^{-}(S_{\mathrm{ret}}) - \mathrm{Acc}^{-}(S_{\mathrm{noret}})$ 广泛偏离零，证实聚合指标对比的是难度不均的自选择人群。
   - **高覆盖率≠高有效利用**：部分模型检索覆盖率>95%，但结构Adherence仅~15-19%，Effective成功率仅~11%。
   - **内容控制验证**：表4显示DeepSeek-V3.2与Gemini-2.5-Flash-lite在Normal、Random-skills、Corrupted、Filler-Dummy条件下RAE均严格为负（如DeepSeek-V3.2 Normal RAE=-15.9 pp，Random=-15.1 pp），证明负向效果源于模型整合失败而非技能文本质量。
   - **失败主因**：表12显示harmful flip中64-72%标记为“ignored/independent”，即检索后答案未体现任何技能内容采纳，模型仍独立求解。
4. **最强表现**：Claude-4.5-H在多数分片与基准上RAE接近零或轻微正值（MBPP+ s42: +1.4 / +1.4 pp；HumanEval+ s42: +4.5 / +0.0 pp），是少数未出现显著反转的模型。

## 相关工作脉络
1. **Agent-Bench / WebArena / OS-World / TheAgentCompany**：聚焦端到端交互成功率与步骤级诊断，不隔离检索注入对同一任务最终输出的配对因果改变，本文弥补该粒度缺口。
2. **SkillsBench / SWE-Skills-Bench / Skill-LearnBench**：采用配对设计（skill-enabled vs no-skill），但报告效应仍聚合于全评测集，未限定在真实检索事件；RAE进一步剥离选择偏差，定位“实际调用”段的净效应。
3. **Voyager / Agent KB / ExpeL / Reflexion / CRITIC**：关注agent自主检索、记忆复用与自我纠错机制，但缺乏“检索事件→SD基线”的配对约束，本文聚焦SF链端到端净收益而非流程展示。
4. **BFCL / RAGAS / IFEval**：诊断工具调用、检索质量、指令接地等单阶段指标；RAE与之完全互补，评估完整检索-推理-合成链的系统级因果影响。

## 局限性与未来方向
1. **评测设定局限**：仅覆盖单轮编码与数学任务、固定检索接口与静态技能库，结论未必外推至长程多步agent、可执行代码库、自然语言标注任务或带迭代记忆/规划的系统。
2. **因果解释边界**：$S_{\mathrm{call}}$ 由SE执行自选择，RAE并非预干预任务总体的无偏因果效应，也不是独立的模型技能能力评分；更细粒度分解需Oracle检索、元数据仅检索、多次技能池构造等消融。
3. **未来方向**：可结合检索策略优化、技能结构化格式化、模型端技能接地（grounding）对齐训练，以及开发抑制“忽略/误用/格式泄漏”行为的数据集与微调方案。

## 研究启发与可借鉴点
1. **配对条件度量范式可迁移**：RAE的“条件于真实触发事件+同任务配对SD基线”设计可直接迁移至RAG、Function Calling、多工具调度评测，消除自选择偏差带来的虚假提升幻觉。
2. **转移矩阵+阶段代理指标联合诊断**：将 $b/c$ 计数与Adherence/Effective等过程标签交叉分析，能快速定位SF链断裂环节（如检索成功但落地失败），适合构建Agent评估的诊断仪表盘。
3. **五类内容扰动对照实验设计**：Schema-Empty / Filler-Dummy / Random-skills / Corrupted 的对照组合能有效剥离“内容质量”与“模型整合能力”的贡献，在工具滥用、hallucination mitigation研究中高度可复用。
4. **团队结合机会**：若本团队从事Agent技能库构建、RAG评估或工具调用对齐，可将RAE作为核心汇报指标之一，配合 $\Delta_{\mathrm{select}}$ 偏差量化与失败模式分层，显著提升方法论严谨性与论文评审说服力。

## 关键术语表
**Skill Following (SF)**：LLM Agent从调用检索接口、接收技能文本到将其逻辑融合并输出正确最终答案的完整认知执行链。
**RAE (Retrieval-Invoked Actual-Use Effect)**：仅在agent实际发起检索并返回至少一个技能的任务集合上，计算SE与SD配对执行的正确率差异（百分比点）。
**OAE (Overall Skill-Access Effect)**：在整个评测集上计算SE与SD配对执行的平均正确率差异，反映系统级技能接入的宏观效用。
**Helpful/Harmful Flip**：分别指SD错误而SE正确（$b$）与SD正确而SE错误（$c$）的任务转移计数。
**Δ_select (
