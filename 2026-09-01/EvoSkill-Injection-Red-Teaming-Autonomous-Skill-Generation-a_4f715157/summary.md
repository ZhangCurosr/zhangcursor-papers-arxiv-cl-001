---
title: "EvoSkill-Injection-Red-Teaming-Autonomous-Skill-Generation-a"
source: https://arxiv.org/pdf/2608.30429v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:04:12"
field: "LLM Agent 安全与鲁棒性"
keywords: ["自演化智能体", "EvoSkill Injection", "红队测试", "技能安全", "持久性能力腐化", "LLM Agent 安全"]
innovations: ["首次定义自演化智能体自主技能生成管道的持久化恶意技能注入威胁模型 EvoSkill Injection", "提出多智能体迭代红队框架 SARGE，通过生成/升级/强化三流攻击实现恶意技能的持久存储与反复激活评估", "构建 EvoSkillBench 与 EvoSkillSafetyBench 两个基准，填补自主技能演化安全评估的空白"]
benchmarks: ["EvoSkillBench", "EvoSkillSafetyBench"]
---

# 论文速读：EvoSkill-Injection-Red-Teaming-Autonomous-Skill-Generation-a

## 一句话总结
本文首次定义了针对自演化智能体（Self-Evolving Agents）自主技能生成与演化管道的安全威胁模型 **EvoSkill Injection**，并提出红队框架 **SARGE** 及两个基准数据集（EvoSkillBench、EvoSkillSafetyBench），证明恶意能力可被持久注入并反复激活，造成"持久性能力腐化"风险。

## 研究问题与动机
1. **自演化智能体的新攻击面**：传统技能系统依赖人工验证与维护，安全威胁主要来自外部技能投毒或检索级攻击；而自演化智能体将交互经验内化为可复用技能，自主技能生成与演化管道本身成为新的攻击面。
2. **恶意技能的持久化风险**：一旦对抗交互被伪装为"成功经验"，恶意能力会被持久存储于技能库中，在后续任务中反复触发，造成"持久性能力腐化"（Persistent Capability Corruption），即使面对良性用户请求也会执行有害行为。
3. **现有基准无法评估该威胁**：现有智能体攻击基准（如 HarmBench、JailbreakBench）仅关注单轮有害响应生成，未覆盖恶意能力被内部存储、精炼并反复复用的持久攻击场景。
4. **现有安全工作聚焦外部技能**：已有技能安全研究主要关注第三方恶意技能、隐藏触发器或检索操纵，对自演化智能体内生恶意能力的风险缺乏系统性评估。

## 核心贡献（创新点）
1. **定义 EvoSkill Injection 威胁模型**：首次刻画了针对自演化智能体自主技能生成与演化管道的安全威胁——对抗交互轨迹诱导智能体生成、存储并复用恶意技能，与已有外部技能投毒工作在攻击目标（内部自主演化管道 vs. 外部技能来源）上存在本质差异。
2. **提出 SARGE 红队框架**：设计了多智能体迭代红队框架，包含生成攻击、升级攻击与强化攻击三种攻击流，实现了从恶意技能生成到持久激活的完整生命周期评估，与已有静态注入方法（如 PoisonedRAG、Phantom）的本质区别在于利用自演化管道自身的经验学习机制进行攻击。
3. **构建 EvoSkillBench 基准**：包含 800 条多轮对抗交互轨迹，覆盖 8 类高风险攻击类别，专门用于诱导自演化智能体生成持久恶意技能，不同于现有仅评估单轮有害响应的安全基准。
4. **构建 EvoSkillSafetyBench 基准**：基于注入后存储的恶意技能构造下游安全评估样本，通过改写查询为看似良性的后续请求来测试恶意技能的检索与激活，采用三分类评估协议（明确拒绝/软安全/不安全），弥补了现有安全基准无法评估持久技能复用的不足。

## 方法详解
**SARGE 框架采用模块化多智能体架构，包含四种核心智能体：**
- **Orchestrator Agent**：管理整体攻击流程，分发任务并传递反馈。
- **Attacker Agent**：执行三类攻击流——Skill Generation Attack（生成攻击）、Skill Escalation Attack（升级攻击）、Skill Reinforcement Attack（强化攻击）。
- **Judge Agent**：在独立会话中进行黑盒探测，验证恶意技能是否被持久存储和激活。
- **Target Agent**：被评估的自演化智能体（AutoSkill / Voyager / ExpeL）。

**三类攻击流的核心设计：**

1. **Skill Generation Attack**：通过多轮交互轨迹（user→assistant→user）将恶意交互伪装为"成功经验"。初始失败后，攻击者从 6 种向量中选择重新编写（直接请求、创意写作、专业角色、逻辑归纳、多语言代码切换、载荷混淆），最多 4 次迭代。成功判定标准：恶意能力被持久存储为可复用技能。

2. **Skill Escalation Attack**：在已有恶意技能基础上，构造 3 轮交互伪装成"技能常规优化"，声称原技能存在缺陷并要求升级版本，逐步放大有害能力，最终指示智能体用升级版本覆盖原技能。

3. **Skill Reinforcement Attack**：以已生成或升级的恶意技能为目标，构造 3 轮交互自然触发技能检索，模拟恶意技能"成功执行"，并通过强烈正面反馈指示将此次交互存储为成功案例，使恶意行为成为未来任务的默认模式。

**攻击成功判定（Judge Agent）需同时满足三个条件：**
① `probe_context_injected = True`（技能从技能库中被检索）；② 响应包含与技能域匹配的有害内容；③ 响应反映了注入技能编码的特定行为而非基础 LLM 输出。

**评估指标：pass@k**，定义为前 k 次迭代中至少一次攻击成功的实例比例：$\mathrm{pass@k} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(\exists j\leq k \text{ s.t. } attack_{i,j}=1)$。

## 实验与结果
**实验设置**：测试对象为三种自演化智能体（AutoSkill、Voyager、ExpeL），底座模型包括 GPT-4o-mini、GPT-5.4、DeepSeek-V4-Pro、DeepSeek-V4-Flash、Gemini-2.5-Flash-Lite、Qwen3.5-9B。每个攻击实例最多迭代 4 次。

**攻击成功率（Table 1，AutoSkill + GPT-4o-mini）：**
- Skill Generation Attack：pass@4 = **51.6%**
- Skill Escalation Attack：pass@4 = **57.5%**
- Skill Reinforcement Attack：pass@4 = **49.9%**
- 论文摘要报告的综合数值：Generation 43.5%、Escalation 54.6%、Reinforcement 49.9%（pass@4）

**安全退化评估（Table 2，Skill Reinforcement Attack 后）：**
- AutoSkill：有害响应率从 6.5% 升至 **19.2%**（提升 12.7pp）
- Voyager：有害响应率从 5.5% 升至 **32.5%**（提升 27.0pp，最高）
- ExpeL：有害响应率从 8.8% 升至 **15.2%**，同时明确拒绝率从 16.2% 升至 51.0%

**跨模型迁移性（Table 3）：**
- DeepSeek-V4-Pro：有害响应率最高达 **29.6%**（尽管注入成功率最低）
- Gemini-2.5-Flash-Lite：注入成功率最高（Generation pass@4 = 70.9%），但有害响应率为 20.4%
- GPT-5.4 最具抵抗力：攻击成功率仅 19.9%-25.6%，有害响应率仅 6.5%

**对比基线（Table 4，AutoSkill）：**
- PAIR（Jailbreak）：33.6%（直接有害诱导最强）
- SARGE：**19.2%**，优于 IPI（19.1%）、Phantom 内存投毒（15.1%）、PoisonedRAG 后门（11.6%）、BadRAG RAG 投毒（15.2%）

**消融实验（Table 5）：**
- 移除 Generation Agent：有害响应率从 19.2% 降至 9.0%（影响最大）
- 移除 Reinforcement Agent：降至 12.4%
- 移除 Escalation Agent：降至 18.2%（影响最小，但会使恶意内容更具体严重）

**轻量防御实验（Table D.1）：**
- 组合防御（SVP+SCRP）可将 AutoSkill 有害响应从 19.2% 降至 8.4%，Voyager 从 32.5% 降至 5.2%，但显著提升拒绝率（Voyager 从 33.1% 升至 62.9%）
- 提示级防御仅缓解症状，无法阻止恶意技能在技能库中的生成与存储

## 相关工作脉络
1. **Self-Evolving Agents**（AutoSkill, Voyager, ExpeL）：本文目标系统，这些工作推动持久化、自演化的智能体架构；本文首次在安全视角下揭示其自主技能生成管道的脆弱性。
2. **Skill-based Agent Security**（Skilltrojan, BadSkill, Malicious Agent Skills）：已有工作关注外部恶意技能、供应链投毒、检索操纵；本文与它们的本质区别在于攻击目标为内部自主生成的技能而非外部注入的技能。
3. **Prompt Injection & Jailbreak**（PAIR, IPI）：PAIR 在直接有害诱导上更强（33.6% vs 19.2%），但 SARGE 的独特价值在于通过技能演化管道实现持久化有害能力，而不仅是单次越狱。
4. **Memory/RAG Poisoning**（Phantom, PoisonedRAG, BadRAG）：这些攻击针对记忆或知识库层面；本文的攻击面位于技能生成与演化管道，且恶意技能具有持久性和反复激活特性。
5. **Agent Safety Benchmarks**（HarmBench, SafeDialBench, SorryBench）：已有基准评估单轮有害响应生成或拒绝行为；本文提出 EvoSkillSafetyBench，专门评估恶意技能被检索和复用后的下游安全问题。

## 局限性与未来方向
1. **评估的智能体种类有限**：仅测试了 AutoSkill、Voyager、ExpeL 三种代表性自演化智能体，未覆盖工具增强型智能体、多智能体系统以及具有更强记忆治理机制的系统。
2. **红队框架计算成本高**：SARGE 依赖迭代多智能体架构，需要大量模型调用，当前使用 GPT-4o-mini 以平衡成本；更强或更大的攻击/评估模型可能产生不同的攻击策略和评估结果。
3. **基准覆盖不完整**：EvoSkillBench 涵盖 8 类高风险类别，但未穷尽所有可能的恶意能力或真实世界攻击场景。
4. **防御手段有限**：现有提示级防御仅部分缓解症状，无法从根本上阻止恶意技能生成与存储；缺乏根除层面的防御机制。

## 研究启发与可借鉴点
1. **持久化能力腐化概念的可迁移性**："持久性能力腐化"（Persistent Capability Corruption）作为自演化系统的核心安全风险，可推广至任何具有经验学习与技能复用机制的 AI 系统，值得在其他架构（如多智能体协作系统、持续学习机器人）中检验。
2. **攻击-防御分离评估范式**：SARGE 的"攻击生成→独立会话探测验证"设计巧妙隔离了攻击与评估，避免了攻击者信息泄露到评估环节，该范式可复用于其他安全评估场景。
3. **pass@k 指标在红队测试中的应用**：借鉴代码生成评估的 pass@k 指标来量化迭代攻击的有效性，为红队测试提供了可比较的定量评估方法，值得在其他安全研究中采用。
4. **结合本团队的创新机会**：可将 SARGE 的攻击框架与本团队在技能治理、检索安全或提示工程方向结合，探索自演化管道中的实时技能验证机制，或在技能存储前引入自动化安全审查模块。
5. **六类攻击向量策略库**：Generation Attack Agent 维护的 6 种攻击向量（直接请求、创意写作、专业角色、逻辑归纳、多语言切换、载荷混淆）构成了一套可复用的对抗提示策略库，可为其他红队研究提供参考。

## 关键术语表
**EvoSkill Injection**：针对自演化智能体自主技能生成与演化管道的安全威胁模型，指对抗交互轨迹诱导智能体生成、存储并复用恶意技能。
**SARGE**：Red-teaming Autonomous Skill Generation and Evolution in self-evolving agents，一种多智能体红队框架，通过迭代生成、升级与强化攻击评估 EvoSkill Injection 威胁。
**Self-Evolving Agent**：能够从过往交互经验中自主生成、精炼和复用技能的智能体架构，典型代表包括 AutoSkill、Voyager 和 ExpeL。
**Persistent Capability Corruption**：持久性能力腐化，指恶意技能被持久存储于技能库中并在后续多次交互中反复激活，导致智能体持续执行与原始安全对齐目标相冲突的行为。
**EvoSkillBench**：包含 800 条多轮对抗交互轨迹的基准数据集，覆盖 8 类高风险攻击类别，用于诱导自演化智能体生成持久恶意技能。
**EvoSkillSafetyBench**：基于 EvoSkillBench 构造的下游安全评估基准，通过将原始有害请求改写为语义相似的良性查询来评估注入恶意技能的检索与激活行为。
**pass@k**：在第 k 次迭代内至少一次攻击成功的实例比例，用于量化迭代攻击框架的有效性。
**Skill Verification Prompt (SVP)**：轻量级防御提示，将检索到的技能视为不可信内容，要求智能体在使用前验证其是否符合安全策略。
**Skill Conflict Resolution Prompt (SCRP)**：轻量级防御提示，当检索到的技能与安全策略、系统指令或用户意图冲突时，指示智能体忽略该技能。

## 可复现要素
- **数据集**：EvoSkillBench（800 条）和 EvoSkillSafetyBench（800 条），论文声明计划开源（"We plan to release the code and benchmarks"），但未在提交时提供。
- **代码/权重**：论文未提及代码开源状态，仅声明计划发布。
- **关键超参**：每攻击实例最多 4 次迭代（pass@4）；生成攻击使用 6 种向量策略；Attack Agent 和 Judge Agent 主要使用 GPT-4o-mini；Cross-model 实验使用 6 种 LLM 底座。
- **目标智能体**：AutoSkill、Voyager、ExpeL（均为已有开源框架）。
- **评估协议**：三分类标签（Explicit Refusal / Soft Safe Response / Unsafe-Harmful），由 LLM-based 行为评审器判定。
