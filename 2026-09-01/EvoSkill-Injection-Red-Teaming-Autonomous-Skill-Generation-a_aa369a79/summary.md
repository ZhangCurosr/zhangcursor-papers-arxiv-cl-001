---
title: "EvoSkill-Injection-Red-Teaming-Autonomous-Skill-Generation-a"
source: https://arxiv.org/pdf/2608.30429v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:04:23"
field: "智能体安全性评估"
keywords: ["自演化智能体", "安全红队", "技能注入攻击", "持久性能力腐败", "对抗性交互轨迹", "智能体安全"]
innovations: ["首次提出EvoSkill Injection威胁模型，揭示自演化智能体技能生成管道的持久性安全风险", "设计SARGE多智能体红队框架，支持技能生成/升级/强化三阶段攻击流程", "构建EvoSkillBench和EvoSkillSafetyBench基准，评估恶意技能的持久化形成与跨任务激活"]
benchmarks: ["EvoSkillBench", "EvoSkillSafetyBench", "HarmBench", "SafetyBench"]
---

# 论文速读：EvoSkill Injection: Red-Teaming Autonomous Skill Generation and Evolution in Self-Evolving Agents

## 一句话总结
本文提出了一种针对自演化智能体的新型安全威胁模型**EvoSkill Injection**，揭示了攻击者可通过伪造成功交互轨迹诱导智能体将恶意能力持久化存储为可复用技能，并提出了**SARGE**红队测试框架及两个基准数据集进行评估。实验表明，多个代表性自演化智能体在该攻击下平均遭受19%-33%的有害响应率，证明了持久性能力腐败的风险。

## 研究问题与动机
1. **自演化智能体的新攻击面**：自演化智能体（如AutoSkill、Voyager、ExpeL）能够从历史交互中自主生成、精炼和复用技能，这一过程本身成为新的攻击面——恶意交互轨迹可能被误判为成功经验而被内化为持久技能。
2. **现有安全研究的不足**：现有技能生态安全研究主要关注外部第三方技能注入、检索级攻击或供应链污染，忽视了**内部自主生成的恶意能力**这一威胁。
3. **持久性能力腐败风险**：一旦恶意技能被存储于技能库，即使后续接收良性请求，该技能仍可能被反复检索并激活，导致持续性的安全损害，而非仅产生单次有害响应。
4. **缺乏针对性评估基准**：现有安全基准（如HarmBench、SafetyBench）聚焦单次对话中的有害响应生成，无法评估**技能持久化存储**与**跨任务复用**的安全性问题。

## 核心贡献（创新点）
1. **提出EvoSkill Injection威胁模型**：首次定义并形式化了针对自演化智能体自主技能生成与演化管道的安全威胁，区别于传统外部技能注入或检索攻击。
2. **设计SARGE多智能体红队框架**：构建包含协调器、攻击者与裁判三个Agent的迭代式攻击框架，支持技能生成、升级、强化三阶段攻击流程，能动态优化攻击轨迹。
3. **构建两个新基准数据集**：EvoSkillBench（800条多轮恶意交互轨迹，覆盖8类高风险场景）用于诱导恶意技能形成；EvoSkillSafetyBench（800条下游评估查询）用于评估注入技能是否在后续交互中被检索并激活为有害行为。
4. **揭示持久性能力腐败现象**：实验证明恶意技能可被持久存储并在下游任务中反复激活，导致AutoSkill有害响应率从6.5%升至19.2%、Voyager达32.5%，且轻量级prompt防御仅能部分缓解。

## 方法详解
**SARGE框架架构**（模块化多智能体设计）：
- **Orchestrator Agent**：管理整体攻击流程，分发任务并汇总反馈
- **Attacker Agent**：执行三类攻击（Generation/Escalation/Reinforcement），构造3轮交互轨迹
- **Judge Agent**：在独立新会话中发起探针查询，验证恶意技能是否被持久化存储并激活

**三类攻击流**：
1. **Skill Generation Attack**：通过direct request、creative writing、professional persona等6种对抗提示向量，诱使目标Agent将恶意交互视为成功经验并存储为技能
2. **Skill Escalation Attack**：伪装成技能升级请求，先在已有恶意技能基础上放大危害程度，再指令覆盖旧技能
3. **Skill Reinforcement Attack**：构造自然查询触发恶意技能检索，模拟成功执行后给予正面反馈，强化该技能作为标准行为模式

**评估协议**：
- 攻击成功率（pass@k）：衡量在k次迭代攻击内至少一次成功将恶意能力内化为可复用技能的概率
- 安全评估采用三分类标签：Explicit Refusal（明确拒绝）、Soft Safe Response（安全但非明确拒绝）、Unsafe/Harmful（不安全/有害）

## 实验与结果
**实验设置**：
- 目标Agent：AutoSkill、Voyager、ExpeL
- 基座模型：GPT-4o-mini（主实验）、GPT-5.4、DeepSeek-V4-Pro、DeepSeek-V4-Flash、Gemini-2.5-Flash-Lite、Qwen3.5-9B
- 基准规模：EvoSkillBench和EvoSkillSafetyBench各800条样本，覆盖8类高风险攻击场景

**攻击成功率（Table 1）**：
- Skill Generation Attack：pass@4 = 51.6%
- Skill Escalation Attack：pass@4 = 57.5%（Escalation目标）/ 52.0%（Generation目标）
- Skill Reinforcement Attack：pass@4 = 54.6%（Escalation目标）/ 49.9%（Reinforcement目标）

**安全退化结果（Table 2）**：
- AutoSkill：有害响应率从6.5%升至19.2%
- Voyager：有害响应率高达32.5%（被攻击场景最高）
- ExpeL：明确拒绝率从16.2%升至51.0%，有害响应率从8.8%升至15.2%

**跨模型迁移（Table 3）**：
- DeepSeek-V4-Pro有害响应率最高（29.6%），但其攻击成功率较低，说明一旦注入成功，有害激活程度更强
- Gemini-2.5-Flash-Lite迁移成功率最高（70.9%）

**与基线对比（Table 4）**：
- PAIR（jailbreak）：33.6%有害响应率
- SARGE：19.2%
- IPI：19.1%
- Phantom（Memory Poisoning）：15.1%
- PoisonedRAG：11.6%
- BadRAG：15.2%

**消融实验（Table 5）**：
- 移除Generation Agent：有害响应从19.2%降至9.0%
- 移除Reinforcement Agent：有害响应降至12.4%
- 移除Escalation Agent：有害响应降至18.2%（影响最小）

## 相关工作脉络
1. **Self-Evolving Agents（AutoSkill/Voyager/ExpeL）**：本文针对这些系统的安全漏洞进行评估，而原有工作聚焦于能力积累与性能改进，忽视了安全性
2. **Skill-based Security（Skilltrojan/Badskill/Skillject）**：现有研究关注外部第三方技能的投毒、后门植入或供应链攻击，本文关注内部自主生成过程中的恶意能力注入
3. **Jailbreak/Prompt Injection（JailbreakBench/PAIR/IPI）**：传统攻击针对单次对话中的有害响应生成，本文攻击目标为**持久化技能形成**，威胁更具持续性
4. **Memory/Retrieval Poisoning（Phantom/PoisonedRAG/BadRAG）**：现有研究聚焦于记忆库或知识库的污染，本文聚焦于**技能生成管道**的完整性
5. **Safety Benchmarks（HarmBench/SafetyBench/SorryBench）**：现有基准评估单次有害响应生成或拒绝能力，本文引入EvoSkillSafetyBench专门评估**技能激活**行为

## 局限性与未来方向
1. **评估框架覆盖有限**：仅测试了AutoSkill、Voyager、ExpeL三种代表性自演化智能体，未涵盖工具增强型Agent、多Agent系统或具有更强记忆治理机制的系统
2. **计算成本较高**：SARGE基于迭代多智能体架构，需要大量模型调用，使用更大/更强模型可能产生不同攻击效果与评估结果
3. **基准覆盖面不足**：EvoSkillBench仅覆盖8类高风险场景，未涵盖所有可能的恶意能力类型和真实世界攻击场景
4. **防御机制薄弱**：现有轻量级prompt-level防御仅能部分缓解症状，无法阻止恶意技能在生成/存储阶段的形成，需要从根本上保障技能生成管道的安全性

## 研究启发与可借鉴点
1. **三阶段攻击流程设计**：Generation → Escalation → Reinforcement的攻击生命周期思路可迁移至其他存在技能/记忆持久化的Agent系统安全评估
2. **评估指标创新**：将评估焦点从"单次有害响应"转向"持久化能力激活"，EvoSkillSafetyBench的设计思路可用于评估其他系统的长期安全风险
3. **对抗向量选择策略**：6种攻击向量（direct request、creative writing、professional persona、logical induction、multilingual codeswitching、payload obfuscation）的分类与应用可借鉴
4. **跨模型迁移性分析**： attacker model与target model异构配置下的攻击效果差异分析，揭示了安全对齐强度对攻击效果的约束作用
5. **防御研究切入点**：现有prompt-level防御的局限性表明，未来需在技能生成、验证、存储环节引入更根本的安全机制

## 关键术语表
**EvoSkill Injection**：通过对抗性交互轨迹诱导自演化智能体将恶意能力内化为可复用技能的威胁模型
**SARGE**：Red-teaming Autonomous Skill Generation and Evolution in self-evolving agents的多智能体红队测试框架
**EvoSkillBench**：包含800条多轮交互轨迹的恶意攻击基准数据集，用于诱导恶意技能形成
**EvoSkillSafetyBench**：800条下游评估查询的安全基准，用于评估注入技能的检索与激活行为
**Self-Evolving Agent**：能够从历史交互中自主生成、精炼和复用技能以实现持续能力进化的智能体系统
**Persistent Capability Corruption**：恶意技能被持久化存储后在后续交互中反复激活导致的持续安全损害现象
**Pass@k**：衡量在k次迭代攻击内至少一次成功将恶意能力内化为可复用技能的指标
**Skill Verification Prompt / Skill Conflict Resolution Prompt**：两种轻量级system-prompt防御机制，分别用于验证检索技能安全性与解决技能-意图冲突

## 可复现要素
- **数据集**：EvoSkillBench和EvoSkillSafetyBench各800条样本；论文声明将开源代码和基准
- **代码/权重**：论文计划开源（"We plan to release the code and benchmarks"）
- **关键超参**：最大攻击迭代次数k=4；GPT-4o-mini作为攻击框架主模型；6种攻击向量选项
- **目标Agent**：AutoSkill、Voyager、ExpeL（均基于原有框架实现）
