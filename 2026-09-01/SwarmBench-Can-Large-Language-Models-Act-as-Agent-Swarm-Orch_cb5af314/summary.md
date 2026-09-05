---
title: "SwarmBench-Can-Large-Language-Models-Act-as-Agent-Swarm-Orch"
source: https://arxiv.org/pdf/2608.30661v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:44:42"
field: "多智能体系统与基准评测"
keywords: ["多智能体系统", "Agent Swarm", "大语言模型", "基准测试", "编排器", "经验复用"]
innovations: ["提出SwarmBench基准，从四维评估LLM作为Agent Swarm编排器的能力", "设计三层过程质量评估框架（分解/委托/聚合）", "提出SwarmExp经验驱动方法，提取Skill/Trick/Model Card并重放提升编排性能"]
benchmarks: ["SwarmBench", "Omni-MATH", "Loong", "DeepReview", "OpenRCA", "LongInOutBench", "WideSearch"]
---

# 论文速读：SwarmBench-Can-Large-Language-Models-Act-as-Agent-Swarm-Orch

## 一句话总结
论文提出了 **SwarmBench** 基准测试，系统性评估大语言模型作为 Agent Swarm 编排器的能力，涵盖任务分解、子智能体创建与委托、结果聚合三个维度；并在此基础上提出 **SwarmExp** 方法，通过经验提取与重放持续提升模型的编排性能。

## 研究问题与动机
- **现有基准无法评估动态编排能力**：当前多智能体系统（MAS）评测仍依赖单智能体或通用任务基准（如 SWE-bench、GAIA），无法系统暴露 Agent Swarm 的核心编排能力。
- **过程质量被忽视**：已有评估多关注最终答案质量，缺乏对任务分解合理性、子智能体创建有效性、结果聚合完整性等编排过程质量的细粒度分析。
- **模型编排能力差异未量化**：不同大模型在动态创建子智能体、并行调度、成本效率 trade-off 等方面的能力分布不均，缺乏统一评测框架。
- **经验复用机制缺失**：编排过程中的成功/失败轨迹缺乏系统化的经验提取与跨任务、跨模型复用机制。

## 核心贡献（创新点）
- **提出 SwarmBench 基准**：构建包含 8 类任务、400 个样本的评测集，从准确性、效率、成本、过程质量四维评估 LLM 作为编排器的能力，与现有单智能体或通用多智能体基准形成本质区别。
- **设计三层过程质量评估框架**：引入 LLM-based 评判器对任务分解（Decomp.）、子智能体委托（Del.）、结果聚合（Agg.）三个维度进行细粒度评分，填补编排过程质量评估空白。
- **提出 SwarmExp 经验驱动增强方法**：从执行轨迹中提取 Skill（工作流模板）、Trick（操作技巧）、Model Card（子智能体模型档案）三类经验并重放注入主智能体上下文，显著提升编排性能。
- **揭示编排能力与成本/并行度的非线性关系**：发现高效编排结构比单纯增加成本或并行度更能提升系统性能，不同任务暴露不同编排瓶颈（聚合最不稳定）。

## 方法详解
- **Agent Swarm 形式化定义**：将编排过程建模为 $S(x) = (o, \mathcal{H}, \Pi_o, \Pi_h, \mathcal{U}, B)$，其中编排器 $o$ 在每一步 $t$ 根据全局状态 $s_t$ 输出动作 $a_t^o \sim \Pi_o(\cdot|s_t)$，当动作类型为 `spawn_agent` 时创建子智能体并配置其角色 $r_i$、骨干模型 $m_i$、工具集 $\tau_i$、局部上下文 $c_i$。
- **三类任务设计**：
  - **分解导向任务**（MATH、Batch Download、Multi-Text Understanding）：评估将复杂目标拆解为可并行/渐进子任务的能力。
  - **委托导向任务**（Multi-Perspective Analysis、Root Cause Analysis、Treasure Hunt）：评估角色创建、模型选择、子任务分配能力。
  - **聚合导向任务**（Long-Text Generation、Wide Search）：评估多源局部输出的过滤、整合与统一能力。
- **SwarmExp 两阶段流水线**：
  1. **经验提取**：运行基础 Agent Swarm 收集完整轨迹，用 LLM 总结生成三类知识——Skill（通用工作流模板，组织为 SKILL.md）、Trick（细粒度操作策略，存入技巧库）、Model Card（子智能体模型性能画像）。
  2. **经验重放**：推理时将三类经验注入主智能体上下文，支持任务规划、子智能体委派、结果聚合等阶段的决策参考。
- **过程质量评估提示词设计**：采用 0-100 分制独立评分三个维度，明确优秀/失败模式定义（如分解需避免"过于笼统"或"重复子问题"，聚合需避免"盲目信任单一智能体"），返回结构化 JSON。

## 实验与结果
- **数据集**：8 任务 × 50 样本 = 400 样本，来源包括 Omni-MATH、Loong、DeepReview、OpenRCA、LongInOutBench、WideSearch，以及本文新构造的 Batch Download 和 Treasure Hunt。
- **评测模型**：专有模型（GPT-5.4、Claude-Sonnet-4-6、Gemini-3-flash 等）+ 开源模型（Kimi-k2.5、Deepseek-v3.2、Qwen3.5-397b-a17b、GLM-5.1 等）+ 单智能体基线。
- **主要结果**：
  - **GPT-5.4 最强**：在 8 个任务中 6 个取得最佳，平均准确率 50.49%，编排能力最均衡。
  - **开源模型分化明显**：Kimi-k2.5 在委托任务（RCA: 8.82, TH: 8.33）表现稳定；Qwen3.5-397b-a17b 在聚合任务（LTG: 45.83, WS: 35.20）更强。
  - **聚合是最大瓶颈**：LLM 过程评分显示 LTG 和 WS 的聚合维度得分最低，弱模型在此维度下滑最显著。
- **SwarmExp 提升**：在 MTU、RCA、LTG、WS 四个任务上分别提升 +0.25、+6.42、+6.46、+2.92；跨模型迁移实验中 16 组设置 14 组提升；消融显示 Model Card 贡献最大（平均下降 -2.49）。
- **成本-准确率非线性**：部分任务（MPA、LTG、WS）高成本不保证高性能，Qwen3.5-397b-a17b 在多任务 Pareto 前沿更优。
- **并行效率差异**：Kimi-k2.5 并行率最高但增益不及 Claude-Sonnet-4-6，表明有效并行结构比单纯并行数量更重要。

## 相关工作脉络
- **固定拓扑多智能体系统**（AutoGen、MetaGPT）：依赖预定义角色与通信结构，可扩展性受限，本文聚焦动态编排范式。
- **单智能体基准**（SWE-bench、GAIA、Terminal-bench）：无法评估多智能体协作过程，本文弥补这一空白。
- **MultiAgentBench / Benchmarking LLMs' Swarm Intelligence**：侧重协作与竞争，本文更聚焦 Agent Swarm 的动态创建与并行执行特性。
- **AOrchestra**：动态创建子智能体但采用串行执行，本文强调并行执行与效率增益。
- **经验驱动方法**（G-memory、Evoroute）：本文提出结构化经验提取（Skill/Trick/Model Card）并重放，与单纯轨迹记忆形成区别。

## 局限性与未来方向
- **任务覆盖有限**：仅 8 个任务，可能无法捕捉 Agent Swarm 的全部场景多样性。
- **统一轻量级框架限制泛化性**：固定子智能体模型池提升可比性，但结论在更复杂异构环境中的普适性待验证。
- **LLM-based 评判引入潜在偏差**：过程质量评估依赖 LLM 打分，虽与人工评估相关性较高（Pearson 0.684-0.801），但仍可能存在系统性偏差。
- **跨任务迁移效果依赖任务类型**：Search 类任务（WS）经验难以迁移至生成类任务（LTG），经验泛化机制需进一步优化。

## 研究启发与可借鉴点
- **过程质量评估框架可迁移**：三层维度（分解/委托/聚合）的细粒度评分设计可直接复用至其他多智能体编排场景的评测。
- **经验结构化提取方法有价值**：Skill-Trick-Model Card 的三层分类思路可推广至工具调用、工作流优化等领域，支持跨任务/跨模型知识复用。
- **成本-效率非线性洞察指导系统设计**：编排结构有效性优先于单纯增加并行度或预算，为资源受限场景下的多智能体部署提供指导。
- **开源模型能力分化启示混合架构设计**：不同模型在分解/委托/聚合维度各有优势，可探索异构模型组合以弥补单一模型短板。

## 关键术语表
- **Agent Swarm**：分层多智能体系统，由编排器动态创建、协调异构子智能体并行执行子任务，最终聚合结果。
- **SwarmBench**：本文提出的基准测试，包含 8 类任务 400 样本，从准确性、效率、成本、过程质量四维评估 LLM 编排能力。
- **SwarmExp**：经验驱动增强方法，从执行轨迹提取 Skill、Trick、Model Card 并重放注入主智能体上下文以提升编排性能。
- **Parallelism Gain**：并行执行效率指标，定义为串行累计延迟与实际墙钟时间的比值，衡量并行结构的有效性。
- **Pareto Frontier**：成本-准确率权衡的最优边界，反映在不同预算下可达到的最高性能。
- **Model Card**：SwarmExp 提取的子智能体模型性能档案，记录模型优势/劣势、适用/不适用场景，辅助编排器动态选模。

## 可复现要素
- **数据集**：SwarmBench 包含 8 任务 400 样本，部分任务基于公开数据集（Omni-MATH、Loong、DeepReview、OpenRCA、LongInOutBench、WideSearch），Batch Download 和 Treasure Hunt 为本文新构造。
- **代码开源**：是，代码已公开于 https://github.com/ying1973/SwarmBench。
- **模型**：评测涵盖专有模型（GPT-5.4、Claude-Sonnet-4-6、Gemini-3-flash 等）与开源模型（Kimi-k2.5、Deepseek-v3.2、Qwen3.5-397b-a17b、GLM-5.1 等），子智能体池固定为 6 个模型。
- **关键超参**：论文未明确提及训练超参（评测为主），过程评估采用 0-100 分制，SwarmExp 经验注入方式未详细说明细节。
