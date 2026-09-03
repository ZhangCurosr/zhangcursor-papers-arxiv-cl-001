---
title: "DEEPREPOQA-Code-Repository-Question-Answering-with-Deep-Agen"
source: https://arxiv.org/pdf/2608.24221v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:26:51"
field: "仓库级代码理解"
keywords: ["repository-level QA", "MCTS", "code understanding", "agent", "software engineering"]
innovations: ["基于 MCTS 的四智能体协作框架", "单步 LLM 价值估计替代多步 rollout", "结构-语义双索引的 Evidence Curation 流水线"]
benchmarks: ["SWE-QA", "Java Subset (30 QA pairs)"]
---

# 论文速读：DEEPREPOQA-Code-Repository-Question-Answering-with-Deep-Agen

## 一句话总结
论文提出 DEEPREPOQA，一种基于 MCTS（蒙特卡洛树搜索）的智能体问答框架，通过将仓库级代码问题转化为系统性规划问题，实现跨文件的深度多跳推理与证据综合，在 SWE-QA 基准上显著优于现有基线。

## 研究问题与动机
1. **核心问题**：开发者回答软件仓库问题时，需要跨多个文件追踪依赖链、综合架构知识，而现有方法多停留在函数级别或依赖浅层 RAG 检索，无法处理复杂的多跳推理。
2. **单一轨迹局限**：ReAct 风格智能体（如 OpenHands）采用单路径探索，容易因早期错误决策导致重要证据被遗漏，且缺乏纠错机制。
3. **检索偏差问题**：现有仓库理解方法倾向于局部片段检索，难以支撑长距离代码依赖的证据 grounding。
4. **探索与利用失衡**：在复杂仓库中，如何平衡广泛探索与精准利用仍是开放挑战。

## 核心贡献（创新点）
1. **将仓库 QA 形式化为 MCTS 规划问题**：区别于扁平检索管道，DEEPREPOQA 构建搜索树进行系统性探索与验证，支持多跳推理。
2. **四智能体协作架构**：引入 Perception、Planning、Execution、Evaluation 四个专业化智能体，分别负责状态感知、行动规划、证据执行与价值评估，形成闭环反馈。
3. **LLM 反馈驱动的价值估计**：用 Evaluation Agent 替代传统多步 rollout，通过 LLM 打分实时更新树节点价值，降低计算开销。
4. **结构-语义双索引表示**：结合 AST 精确查找（FindClass/Function）与语义向量检索（SemanticSearch），兼顾精确定位与概念匹配。
5. **可扩展的 action space**：定义 six 种原子操作（FindClass、FindFunction、FindCodeSnippet、ViewCode、SemanticSearch、Finish），覆盖从粗粒度搜索到细粒度验证的完整推理链。

## 方法详解
### 3.1 仓库解析与表示
- **AST 解析**：使用 Tree-sitter 生成语言无关的抽象语法树，提取类、函数、代码片段及其跨文件关系（调用链、继承、模块依赖）。
- **多层表示**：
  - 索引查找：基于目录结构和 AST 支持精确符号定位（FindClass、FindFunction、FindCodeSnippet）。
  - 语义检索：使用 voyage-code-3 将代码元素向量化，支持 SemanticSearch。

### 3.2 Action Space
| Action | 参数 | 是否嵌入 |
|--------|------|----------|
| FindClass | class-name, file-pattern | No |
| FindFunction | function-name, class-name, file-pattern | No |
| FindCodeSnippet | code-snippet, file-pattern | No |
| ViewCode | code-span-list, file-pattern | No |
| SemanticSearch | query, category, file-pattern | Yes |
| Finish | answer, finish-reason | No |

### 3.3 四智能体设计
1. **Perception Agent**：分析当前搜索状态，生成情境报告，识别冗余、盲点和候选符号，避免重复探索。
2. **Planning Agent**：基于情境报告为叶子节点生成候选行动（最多 max_expand=3 个子节点），选择信息增益最大的分支。
3. **Execution Agent**：执行规划行动，内部包含证据筛选器：按相关性排序、去重、折叠样板代码、提取精确行号范围。
4. **Evaluation Agent**：对行动-观察对打分（0-100 标量值），评估其对回答问题的效用，生成定性反馈，并将价值反向传播到根节点。

### 3.4 MCTS 流程
```
Algorithm 1: DEEPREPOQA Algorithm
Input: Q, R, N
Output: A
1  search_tree ← MCTSTree(Q, R)
2  while iteration_count < max_iterations do
3      node ← Select(search_tree.root)  // UCT 最高分节点
4      new_node ← Expand(node)
5      situational_report ← Perceive(new_node, parent, siblings)
6      new_node.action ← PlanNextAction(Q, report)
7      raw_result ← ExecuteAction(new_node.action)
8      new_node.observation ← FilterEvidence(Q, raw_result)
9      value ← Evaluate(new_node)
10     Backpropagate(new_node, value)
11     if new_node.action.type = "Finish" then break
12 end
13 T ← GetBestTrajectory(finished_nodes)
14 A ← ExtractAnswer(T)
```

**Selection 公式**：
$$\text{UCT}(s, a) = Q(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

**Simulation 简化**：单次四智能体循环代替传统 rollout，Evaluation Agent 直接输出 $V(s, a)$ 并赋值给 $Q(s, a)$。

**Back-propagation**：将 leaf 节点价值沿路径更新所有祖先节点的 visit count 和 Q 值。

**Answer Synthesis**：从完成节点中提取最佳轨迹，要求至少一个 code span 支撑答案，验证问题类型（what/where/how/why）。

## 实验与结果
### 数据集
- **SWE-QA**：15 个开源 Python 仓库，720 个 QA 对（原 576 + 新增 conan/streamlink/reflex 各 48）。
- **Java 子集**：30 个 QA 对（Strata, Fineract, Shiro）用于跨语言验证。

### 评估基线
- **Direct Prompting**：无上下文直接查询 LLM
- **Function Chunking RAG**：基于函数/类粒度的语义检索
- **Sliding Window RAG**：500 行滑动窗口检索
- **SWE-agent v1.0**：ReAct 风格智能体
- **OpenHands v1.1.0**：另一主流智能体
- **商业工具**：Tongyi Lingma、Cursor-agent

### 主要结果（Qwen3-Coder-480B-A35B-Instruct）
| 方法 | Correctness | Completeness | Reasoning | Overall |
|------|-------------|--------------|-----------|---------|
| Direct Prompting | 7.71 | 5.85 | 7.92 | 51.33 |
| SWE-agent | 8.98 | 8.49 | 11.02 | 57.35 |
| OpenHands | 10.30 | 9.57 | 12.96 | 62.33 |
| **DEEPREPOQA** | **10.97** | **10.17** | **13.28** | **64.43** |

- 最大提升：**+7.08 分**（vs SWE-agent，Qwen3-Coder 模型）
- GPT-5.1 上达 **70.06 分**，超越 Tongyi Lingma（69.12），接近 Cursor（70.71）

### 消融实验（Qwen3-Coder 基座）
| 变体 | Overall | Δ |
|------|---------|---|
| DEEPREPOQA (full) | 64.43 | - |
| w/o MCTS | 62.16 | -2.27 |
| w/o Perception Agent | 62.70 | -1.73 |
| **w/o Evaluation Agent** | **60.93** | **-3.50** |
| w/o Semantic Search | 63.36 | -1.07 |

- **Evaluation Agent 贡献最大**（-3.50 分），验证了价值估计的关键性。
- **MCTS 框架次之**（-2.27 分），证明系统性搜索优于贪婪单路径。

### 探索预算分析
| Max Nodes | Overall | Correctness | Reasoning |
|-----------|---------|-------------|-----------|
| 5 | 55.06 | 8.27 | 8.81 |
| 10 | 62.97 | 10.43 | 12.70 |
| 15 | 64.43 | 10.97 | 13.28 |
| 20 | 65.33 | 11.57 | 12.88 |
| 30 | 65.15 | 11.43 | 12.90 |

- 5→10 节点增益最大（+7.91），10→15 次之，15→20 趋于饱和。
- 建议在 15-20 节点间平衡性能与成本。

### 效率分析
- DEEPREPOQA 输入 token 数（平均 78,681）低于 SWE-agent（126,026）和 OpenHands（87,045）。
- 输出 token 略高（主因是 Perception/Evaluation Agent 的反馈输出）。
- **总体效率与 SOTA 智能体相当，但性能更优**。

### Judge 可靠性
- 三 LLM 裁判（GPT-5.4、Claude-Sonnet-4-6、Gemini-3.1-Pro）排名一致性 $\rho_{overall} = 1$。
- LLM 裁判与人类专家相关性 Pearson $r = 0.972$，偏好一致性 88.2%（Cohen's $\kappa = 0.725$）。

## 相关工作脉络
1. **RAG 方法**（Zhang et al., 2023; Wang et al., 2024b）：基于语义检索的静态上下文拼接，缺乏动态验证机制。DEEPREPOQA 通过 MCTS 实现多跳推理。
2. **Agent 方法**（Yang et al., 2024; Wang et al., 2025b）：SWE-agent/OpenHands 采用单路径 ReAct，DEEPREPOQA 引入树搜索避免单点故障。
3. **代码图方法**（Ouyang et al., 2024）：Repograph 构建仓库知识图谱，但需离线构建。DEEPREPOQA 在线探索，无需预计算图结构。
4. **长上下文压缩**（Shi et al., 2025）：LongCodeZip 压缩代码上下文，侧重生成任务。DEEPREPOQA 聚焦 QA 场景的多跳验证。
5. **仓库理解基准**（Peng et al., 2026; Chen et al., 2025a）：SWE-QA 和 CoreQA 定义了评测协议。本文是首个引入 MCTS 的仓库 QA 方法。

## 局限性与未来方向
1. **语义歧义导致的相邻概念混淆**：当问题表述模糊时（如 "structured container" 同时匹配 AppInfo 和 AppWrap），所有分支可能共享错误初始检索，MCTS 难以回溯。
2. **关键词搜索瓶颈**：SemanticSearch 失效时，若 Agent 缺乏结构导航回退策略（如 FollowImport、ViewDirectory），会陷入关键词变体循环。
3. **开放设计问题评估偏差**：参考解唯一且具体时，LLM-as-Judge 可能对合理但不同的设计扣分（F4 模式）。
4. **跨文件依赖深度限制**：超深调用链（如 >10 跳）可能导致记忆路径过长，增加 token 消耗与推理延迟。
5. **仅验证 Python 和 Java**：对其他语言（如 C++、JavaScript）的泛化需进一步验证。

## 研究启发与可借鉴点
1. **单步 Simulation 替代 Rollout**：用 Evaluation Agent 直接估计 Q(s,a) 替代多步 rollout，大幅降低计算开销，可迁移至其他 MCTS-based agent 系统。
2. **Perception Agent 的横向感知**：不仅关注当前节点，还总结兄弟分支的探索结果，避免重复尝试失败路径——这一设计可用于任何多路径搜索场景。
3. **Evidence Curation Pipeline**：Execution Agent 内部的 Ranking-Deduplication-Context-Stitching 流程可复用为通用证据整理模块。
4. **多裁判 LLM-as-Judge 协议**：三层裁判平均 + Spearman 一致性验证，为评测系统设计提供参考。
5. **Failure Mode 分类法**：F1-F4 四类失败模式（Wrong Location/Adjacent Concept/Keyword-only/Open-ended）可指导后续改进方向。

## 关键术语表
**MCTS（Monte-Carlo Tree Search）**：一种树搜索算法，通过选择、扩展、模拟、反向传播四阶段平衡探索与利用，此处用于代码仓库推理路径搜索。

**UCT（Upper Confidence Bound for Trees）**：MCTS 的选择策略公式，$Q(s,a) + c\sqrt{\ln N(s)/N(s,a)}$，平衡已知价值与不确定性。

**SWE-QA**：仓库级代码问答基准，包含 15 个 Python 仓库、720 个 QA 对，评估 Correctness/Completeness/Relevance/Clarity/Reasoning 五维度。

**Evidence Curation**：执行行动后对原始返回结果的过滤、排序、去重、上下文拼接过程，生成 citation-ready 的证据束。

**Value Estimation**：Evaluation Agent 对行动-观察对赋予 0-100 标量分，反映其对回答问题的预期效用，替代传统 rollout。

**SemanticSearch vs Structural Lookup**：前者基于向量嵌入的概念匹配（模糊），后者基于 AST/目录结构的精确符号定位（精确）。

**Multi-judge LLM-as-Judge**：使用三个不同 LLM（GPT-5.4、Claude-Sonnet-4-6、Gemini-3.1-Pro）作为裁判取平均，降低单裁判偏差。

**Failure Mode F1-F4**：四类失败模式——F1 错误位置、F2 相邻概念混淆、F3 关键词搜索停滞、F4 开放设计评估偏差。

## 可复现要素
- **数据集**：SWE-QA（https://doi.org/10.5281/zenodo.21063159），15 个 Python 仓库 + 3 个 Java 仓库子集，720 QA 对
- **代码**：已开源（Zenodo DOI: 10.5281/zenodo.21063159）
- **权重**：公开 LLM（GLM-4.6、Kimi K2、Qwen3-Coder-480B-A35B-Instruct、GPT-5.1），Embedding 模型 voyage-code-3
- **关键超参**：
  - max_expand = 3（每节点最大子节点数）
  - max_iterations = 15（主实验）
  - MCTS temperature = 0.7（探索阶段）
  - Answer generation temperature = 0
  - Top-p = default API value
