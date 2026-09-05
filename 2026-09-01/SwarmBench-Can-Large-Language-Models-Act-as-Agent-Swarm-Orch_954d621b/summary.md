---
title: "SwarmBench-Can-Large-Language-Models-Act-as-Agent-Swarm-Orch"
source: https://arxiv.org/pdf/2608.30661v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:45:01"
field: "多智能体编排与评估"
keywords: ["Agent Swarm", "Multi-Agent Systems", "LLM Benchmark", "Orchestration", "Experience Replay"]
innovations: ["提出SwarmBench基准系统评估LLM作为Agent Swarm编排者的能力", "提出SwarmExp方法通过经验提取与回放提升编排性能"]
benchmarks: ["SwarmBench"]
---

# 论文速读：SwarmBench-Can-Large-Language-Models-Act-as-Agent-Swarm-Orch

## 一句话总结
论文提出 **SwarmBench** 基准测试，系统评估大语言模型作为 Agent Swarm 编排者的能力，并从准确性、效率、成本和过程质量四维度揭示当前模型的编排瓶颈；在此基础上提出 **SwarmExp** 方法，通过经验提取与回放显著提升了编排性能。

## 研究问题与动机
- 现有 benchmark 主要针对单 agent 或通用 agent 设计，无法系统评估 Agent Swarm 这一新兴动态编排范式的核心能力
- 多 agent 系统正从固定拓扑协作（如 debate、pipeline、manager-worker）向动态编排演进，但缺乏专门针对"编排者能力"的评估标准
- 现有评估过度关注最终答案质量，忽视编排过程质量（任务分解合理性、子 agent 创建有效性、结果聚合完整性）
- 需要通过多维评估揭示不同模型的编排能力结构，并探索基于经验积累的方法提升编排性能

## 核心贡献（创新点）
1. **提出 SwarmBench 基准**：专为评估 LLM 作为 Agent Swarm 编排者能力而设计，覆盖任务分解、子 agent 委派、结果聚合三大能力维度，并提供准确性、效率、成本、过程质量四维度评估体系；与已有 MAS benchmark（如 MultiAgentBench）聚焦最终结果不同，本文强调过程质量的细粒度分析。
2. **构建 8 任务、400 样本的多样化数据集**：涵盖分解导向（MATH、BD、MTU）、委派导向（MPA、RCA、TH）和聚合导向（LTG、WS）三类任务，其中 BD 和 TH 为论文全新构建；与已有 benchmark 相比，任务设计更贴近真实动态编排场景。
3. **系统揭示当前模型的编排能力结构与瓶颈**：发现聚合是所有模型最薄弱的环节，且成本与性能并非线性关系，关键在于编排结构的有效性而非单纯增加计算开销。
4. **提出 SwarmExp 经验驱动增强方法**：从轨迹中提取 skill、trick、model card 三种经验并回放注入主 agent 上下文，实验表明可稳定提升多个任务的编排性能；该方法基于"经验可复用积累"的假设，区别于纯提示工程 approach。

## 方法详解
**Agent Swarm 定义**：分层多 agent 系统，由单个 orchestrator 协调动态创建的子 agent 集合完成任务，具备三个特征：(1) 层次化编排；(2) 运行时动态创建异构子 agent；(3) 子 agent 并行执行。

**形式化框架**：给定输入任务 $x$，Agent Swarm 过程表示为 $S(x) = (o, \mathcal{H}, \Pi_o, \Pi_h, \mathcal{U}, B)$，其中 $o$ 为编排者，$\mathcal{H}$ 为子 agent 配置集合，$\Pi_o$ 为编排策略，$\Pi_h$ 为子 agent 执行策略，$\mathcal{U}$ 为工具集，$B$ 为执行预算。

**编排动作**：全局步骤 $t$，编排者观察状态 $s_t$ 输出动作 $a_t^o \sim \Pi_o(\cdot|s_t)$；当 $a_t^o = \text{spawn agent}$ 时创建子 agent，指定配置 $\phi_i = (r_i, m_i, \tau_i, c_i)$（角色、骨干模型、工具集、局部上下文）。

**评估维度**：
- **准确性**：各任务自有评测标准（LLM judge / 规则评估）
- **效率**：Parallelism Rate（平均并发子 agent 数）和 Parallelism Gain（串行/实际时间比）
- **成本**：实际货币花费（按 token 价格计算）
- **过程质量**：LLM-based judge 评估分解、委派、聚合三个维度（0-100分）

**SwarmExp 流程**：
1. 运行初始 Agent Swarm 收集完整轨迹
2. 用 LLM 总结轨迹并提取三类经验：Skill（高层工作流模式）、Trick（细粒度操作技巧）、Model Card（子 agent 骨干模型的表现特征）
3. 推理时将经验注入主 agent 上下文，辅助规划、委派、聚合决策

## 实验与结果
**数据集**：SwarmBench，8 个任务 × 50 样本 = 400 样本，来源包括 Omni-MATH、Loong、DeepReview、OpenRCA、LongInOutBench、WideSearch，以及全新构建的 Batch Download 和 Treasure Hunt。

**评测模型**：商业模型（GPT-5.4、Claude-Sonnet-4-6、Gemini-3-flash、Doubao-seed-1-8、Claude-Haiku-4-5）和开源模型（Kimi-k2.5、Deepseek-v3.2、Qwen3.5-397b-a17b、GLM-5.1、Qwen3-30b-a3b），另有单 agent baseline。

**主要结果**：
- **最强整体编排者**：GPT-5.4，在 8 个任务中 6 个任务取得最佳/次佳成绩，平均分 50.49
- **开源最优**：Qwen3.5-397b-a17b（聚合任务 LTG/WS 表现突出）和 Kimi-k2.5（委派任务 RCA/TH 表现突出）
- **成本-准确性关系**：非线性，强编排者可用更低成本实现更好效果；MATH/BD 随成本提升而改善，而 MPA/LTG/RCA/WS 存在近最优低成本的点
- **效率发现**：并行率最高≠并行增益最高（Kimi-k2.5 并行率最高但 Gemini-3-flash 并行增益最高），关键在于有效并行结构而非并发数量
- **过程质量瓶颈**：聚合是普遍最弱维度，所有模型在 LTG/WS 上聚合得分最低

**SwarmExp 提升**（Table 2）：
- MTU: 48.82 → 49.07 (+0.25)
- RCA: 10.00 → 16.42 (+6.42)
- LTG: 48.19 → 54.65 (+6.46)
- WS: 38.67 → 41.59 (+2.92)

**消融实验**（Table 4）：移除 Model Card 影响最大（均降 2.49），Skill 和 Trick 移除也有显著下降，三者均必要。

**跨模型迁移**（Table 3）：从 GPT-5.4 提取的经验注入其他模型后，16 组中有 14 组提升，LTG 增益最大。

## 相关工作脉络
- **MultiAgentBench (Zhu et al., 2025)**：共享 multi-agent 评估思想，但侧重协作竞争而非动态编排，缺乏过程质量分析
- **AOrchestra (Ruan et al., 2026)**：同样动态创建子 agent，但采用串行执行，未探索并行编排与效率收益
- **Benchmark for single-agent/general agent (GaIA, SWE-bench, Terminal-bench)**：无法暴露 Agent Swarm 特有的编排能力瓶颈
- **Experience-driven agent methods (Evoroute, G-Memory)**：SwarmExp 借鉴经验提取思想但聚焦于编排知识（skill/trick/model card）而非任务规划经验
- **Heterogeneous Swarms (Feng et al., 2026)**：优化模型角色权重，但未系统性评估编排质量全过程

## 局限性与未来方向
- **任务覆盖面有限**：仅 8 个任务，可能无法捕获全部 Agent Swarm 场景
- **框架统一性限制泛化**：所有模型在统一轻量级 swarm scaffold + 固定子 agent 模型池下评估，结论推广性受限
- **LLM judging 偏差**：过程质量评估依赖 LLM judge，虽与人工评估相关系数较高（0.68-0.80）但仍可能存在偏见
- **未来方向**：扩展任务多样性、探索跨任务经验迁移（已有初步探索显示 WS 类任务迁移困难）、结合强化学习优化编排策略

## 研究启发与可借鉴点
1. **多维评估框架**：将准确性、效率、成本、过程质量统一纳入评估，可复用到其他 agent orchestration 研究
2. **经验提取三分类设计**：Skill（高层模式）/Trick（细粒度技巧）/Model Card（模型能力档案）的分工思路，可迁移至工具调用、GUI agent 等领域
3. **跨模型经验迁移验证**：用 GPT-5.4 提取的经验成功迁移到其他模型，证明经验的可复用性；团队可探索自有模型的经验提取与外部经验注入
4. **并行增益≠并行率的洞察**：提醒后续研究不能仅追求高并发，而应关注有效并行结构的设计，这对并行任务调度研究有启发

## 关键术语表
- **Agent Swarm**：由编排者动态创建、协调异构子 agent 并行执行任务的分层多 agent 系统范式
- **Orchestrator（编排者）**：负责任务分解、子 agent 创建委派、结果聚合的主 agent
- **Parallelism Rate**：执行过程中平均并发子 agent 数量，衡量并行度
- **Parallelism Gain**：串行执行累积延迟与实际 wall-clock 时间之比，衡量并行效率收益
- **SwarmExp**：通过提取轨迹中的 skill、trick、model card 经验并回放注入主 agent 上下文的编排增强方法
- **Model Card（在此语境）**：子 agent 骨干模型在具体任务上的实际表现特征与适用场景的经验化描述
- **Pareto Frontier（在此语境）**：成本-准确性权衡下的最优性能边界

## 可复现要素
- **数据集**：SwarmBench 公开可用（含 8 个任务、400 样本）
- **代码**：https://github.com/ying1973/SwarmBench 开源
- **权重**：使用商用模型 API（GPT-5.4 等）及开源模型，未提供专属微调权重
- **关键超参**：论文未提及特殊超参；子 agent 模型池固定为 6 个模型（GPT-5-mini、Claude-Haiku-4-5-thinking、Gemini-2.5-flash-lite、Qwen3.5-35b-a3b、GLM-4.5-air、Doubao-seed-1-6-flash）
