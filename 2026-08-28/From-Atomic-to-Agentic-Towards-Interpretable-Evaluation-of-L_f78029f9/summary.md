---
title: "From-Atomic-to-Agentic-Towards-Interpretable-Evaluation-of-L"
source: https://arxiv.org/pdf/2608.26950v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:26:14"
field: "大语言模型评测与智能体能力评估"
keywords: ["LLM评估", "智能体数学推理", "过程级评测", "原子能力", "AgenticBenchmark", "可解释评估"]
innovations: ["提出原子能力×智能体功能交叉矩阵的过程级评测框架AMB", "构建覆盖规划/行动/反馈三维的细粒度任务族与自动化数据流水线", "实证揭示端到端准确率与智能体能力分布的显著解耦现象"]
benchmarks: ["AgenticMathBench (AMB)", "MATH", "AIME25", "OlympiadBench", "Omni-MATH", "MiniF2F"]
---

# 论文速读：From-Atomic-to-Agentic-Towards-Interpretable-Evaluation-of-L

## 一句话总结
本文提出 **AgenticMathBench (AMB)**，一个面向大语言模型（LLM）智能体数学推理能力的面过程级评测基准，通过将数学原子能力与智能体核心功能（规划 Planning、行动 Action、反馈 Feedback）对齐，揭示端到端准确率相似的模型在智能体能力分布上存在显著差异。

## 研究问题与动机
- **现有数学基准仅评估最终答案**：大多数数学评测关注端到端正确性，无法诊断求解过程中的过程级失败或逻辑严谨性。
- **缺乏数学能力与智能体行为的对齐**：现有工作要么覆盖窄口径的数学技能，要么未建立数学原子能力与智能体原子行为之间的映射关系，难以定位能力瓶颈。
- **过程级评估缺失阻碍智能体发展**：随着 LLM 向 agentic reasoning 演进，需在部署复杂智能体系统前评估基础模型所具备的内在智能体能力。
- **数学推理与智能体行为具有结构同构性**：两者均可分解为原子化、可复用的单元，这一特性为过程级可解释评估提供了理论基础。

## 核心贡献（创新点）
- **提出首个面向智能体数学推理的面过程级基准 AMB**，超越最终准确率，支持对规划、行动、反馈三个维度的细粒度诊断。
- **构建数学原子能力分层分类体系并与智能体功能对齐**，定义三级能力（基础概念与计算、高级推理与应用、数学元认知），分别映射至 Action、Feedback 与 Planning。
- **设计覆盖文本与多模态的规划/行动/反馈任务族**，包含 9 类原子能力任务（如符号识别、概念理解、形式化、数学建模等）及 3 类智能体任务（规划、行动、反馈）。
- **开发自动化数据工程流水线**，集成多源数据收集、高质量轨迹合成（Math Agent 范式）、多级过滤（答案正确性→人工推理质量→多样性）与细粒度 LLM 标注。
- **实证揭示"相似 E2E 准确率≠相似智能体能力分布"**，如 GLM-4.7 在 MATH/AIME25 上接近满分但智能体能力垫底，GPT-5.2 与 Gemini-3-Pro E2E 分数相近但能力画像差异显著。

## 方法详解
- **原子能力分类体系（Atomic Capability System）**：分为三个层级——Level 1（符号识别 Symbol Recognition、概念理解 Concept Understanding、计算 Calculation）；Level 2（空间感知 Spatial Perception、形式化 Formalization、演绎与归纳推理 Deductive and Inductive Reason、数学建模 Mathematics Modeling）；Level 3（定理应用 Theorem Application、自我反思 Self-Reflection、新知识学习 New Knowledge Learning）。
- **智能体任务 formulation**：
  - **规划（Planning）**：将问题分解为有序能力序列 $\pi = [(a_1, g_1), \ldots, (a_T, g_T)]$，其中 $a_t$ 为原子能力、$g_t$ 为子目标，评估能力选择、方案生成与动态下一步规划。
  - **行动（Action）**：解耦评估单一原子能力的孤立执行，涵盖 8 类可执行任务（如符号识别输出标准 LaTeX、形式化输出 Lean4 定理声明等）。
  - **反馈（Feedback）**：定义为 $f: (\text{problem}, \tau_{\le t}) \to \{\text{status}, \text{type}, \text{sugg}\}$，评估正确性判断、错误定位与修复建议。
- **数据构建流水线**：从 150+ 数学基准中筛选 27 个数据集；单原子任务直接复用/规范化已有数据；组合原子任务（建模、推理）使用 GPT-4o 重写；轨迹合成采用 Math Agent 范式（规划→执行→验证→调整），经最终答案过滤（17.3% 留存率）、人工两步质量过滤与多样性过滤（至少 2 步、2 种原子能力）。
- **评估指标体系**：包括 P/R/F1、Exact Match、CAS Equivalence、Lean Compilation、LLM-as-Judge（Judge(Cov/Cons)）、语义相似度等；针对开放型任务采用 DeepSeek-V3 作为默认 Judge，并通过 10% 人工标注验证（F1=0.86）与 GPT-5-mini 交叉验证（跨 Judge 一致性 0.96）确保可靠性。
- **推理超参**：规划/反馈任务 temperature=0.0、max tokens 256–1024；行动任务 temperature=0.7、max tokens 2048；形式化与定理应用抽取 5 次采样取多数投票。

## 实验与结果
- **评测模型**：三大家庭——通用开源模型（Llama-4-Scout/Maverick-17B、DeepSeek-V3.2、Qwen3-32B/235B、GLM-4.7）、数学专用开源模型（Qwen2.5-Math-72B、DeepSeek-Math-V2）、商业模型（GPT-5.2、Claude-4.5-Sonnet-Thinking、Gemini-3-Pro）。
- **规划能力**：商业模型全面领先，Claude-4.5 在 Capability Planning 上 F1=78.8；Open-source 中 DeepSeek-V3.2 在 Solution Planning 上达到 76.1 接近 GPT-5.2 的 85.1；几乎所有模型在 Next-step Planning 上出现显著性能下滑（如 GPT-5.2 从 85.1 降至 41.0），暴露动态规划瓶颈。
- **反馈能力**：正确性判断整体中等（最高 Claude-4.5 46.3%，Gemini-3-Pro 52.8%）；错误定位中 DeepSeek-V3.2 的 Type Classification 达 66.8% 最优；修复建议整体最弱（GPT-5.2 34.7%、Claude-4.5 32.8%），模型能解释原因但难以转化为可执行修复步骤。
- **行动能力**：商业模型在形式化与建模上表现最强（GPT-5.2 Formalization Compile=100%，Modeling Var-F1=90.9%）；GLM-4.7 在形式化上仅 Compile=4.4% 极弱；DeepSeek-V3.2 在 Calculation 上 EM=84.7、Concept F1=52.4 全面优异。
- **E2E vs. Agentic 对比（Table 5）**：GLM-4.7 E2E 满分（MATH=98.8, AIME25=95.7）但 Agentic 全面垫底（Plan=29.0, Feed=29.7, Act=64.5）；Qwen2.5-Math-72B E2E 仅 85.9/20.0 但 Act=30.5 同样薄弱，印证"数学专项训练≠智能体能力"。
- **最强结果**：Claude-4.5 在 Capability Planning F1=78.8、Solution Planning Overall=84.0 为当前最高；DeepSeek-V3.2 在 Action 多项上突出（Calc EM=84.7, Formalization Compile=99.4, Theorem Pass@k=100）。

## 相关工作脉络
- **数学推理基准**：MATH、GSM8K、OlympiadBench、Omni-MATH 等以端到端准确率为核心，仅报告 aggregate score，缺乏过程级诊断；GAUSS 虽引入结构化技能维度但仍停留在问题级评估，未对齐智能体行为。
- **过程级评估**：Evaluating beyond accuracy (Xia et al., 2024)、process reward models (Lightman et al., 2024) 关注步骤级验证，但未建立与智能体功能的系统映射；本文首次将过程评估绑定到 Planning/Action/Feedback 三元组。
- **智能体数学推理系统**：ToRA、MathChat、MathAgent-PRER 等构建工具集成或多智能体系统，但评测仍依赖最终答案；MM-Agent 聚焦现实建模场景；本文聚焦基础模型的内在智能体能力而非部署后系统表现。
- **原子思维与能力分解**：Kuang et al. (2025) 定义三种基本数学原子能力；本文扩展为九维三层分类并系统对齐智能体模块。
- **形式化验证基准**：MiniF2F、FormL4、CriticLeanBench 等专注 Lean 自动定理证明；本文将其作为 Formalization 原子能力子集，纳入更广泛的智能体能力框架。
- **定位差异**：本文填补了"数学原子能力×智能体功能"交叉矩阵的空白，提供多维可解释诊断而非单一 accuracy 数字，强调解耦评估（intrinsic capability vs. deployed agent）。

## 局限性与未来方向
- **分类体系未覆盖所有数学智能体推理形态**：如长期记忆（long-horizon memory）与迭代新知识学习尚未纳入，因单轮测试难以直接评估。
- **多模态数据规模偏小**：含图表的高难度多模态数学问题稀缺，导致多模态轨迹仅 20 条（文本 217 条），限制了视觉推理评估的充分性。
- **非部署态评估**：AMB 测量的是内在智能体能力而非完整 agent 系统，未统计 token 消耗与延迟成本；附录 I 仅作定性讨论，缺乏成本感知的联合评估。
- **未来方向**：扩展至长期 horizon 场景（显式记忆扰动测试、工具交互循环）；扩充多模态轨迹；引入细粒度 token/latency 与能力增益的成本权衡分析；探索 multi-agent debate 与元认知评测的自然延伸。

## 研究启发与可借鉴点
- **"原子能力×智能体功能"交叉矩阵设计范式**：可将此框架迁移至其他领域（如代码生成、科学发现、机器人操作），通过解耦评估定位各模块瓶颈。
- **轨迹合成与多级过滤流水线**：Math Agent 范式（规划→执行→验证→调整）结合答案正确性→人工质量→多样性三步过滤，为合成高质量推理轨迹提供可复用模板。
- **LLM-as-Judge 的多维评分与鲁棒性验证**：引入 3–5 条专家撰写的评分准则、多维度（正确性/一致性/连贯性/效率）加权、10% 人工标注对齐验证（F1=0.86）及跨 Judge 交叉验证（一致性 0.96），为标准化工具性评测提供方法参考。
- **E2E 与过程级能力的解耦分析**：Table 5 的对比揭示"高分模型≠强智能体"现象，提示在模型训练/评测中需同时报告 E2E 与过程级指标，避免单一 accuracy 掩盖能力缺陷。
- **概念理解作为主要瓶颈的发现**：Figure 6 显示 Concept 是最弱 Action 维度，提示后续训练需在语义对齐层面加强，而非仅优化计算或形式化。

## 关键术语表
- **AgenticMathBench (AMB)**：本文提出的面向 LLM 智能体数学推理能力的面过程级评测基准，覆盖 Planning/Action/Feedback 三维。
- **Atomic Capability（原子能力）**：数学推理中最小的可复用认知单元，分为三级九维（如符号识别、形式化、数学建模等）。
- **Planning（规划）**：智能体核心功能之一，指全局方案生成与动态下一步决策能力，评估能力选择、有序分解与轨迹条件下的续步规划。
- **Action（行动）**：智能体核心功能之一，指执行单个原子数学子任务的能力，解耦评估以保证细粒度诊断。
- **Feedback（反馈）**：智能体核心功能之一，指对已执行轨迹的状态评估、错误定位与修复建议生成能力。
- **LLM-as-Judge**：使用大语言模型作为自动评判器，对开放型推理过程进行多维度评分，辅以人工对齐验证确保可靠性。
- **Trajectory Synthesis（轨迹合成）**：通过 Math Agent 范式自动生成符合"规划→执行→验证→调整"结构的数学推理轨迹，并经多级过滤保留高质量样本。
- **End-to-End (E2E) Accuracy**：传统数学基准评估方式，仅以最终答案正确性为指标，缺乏过程级诊断能力。

## 可复现要素
- **数据集**：基于 27 个公开数学基准（CROHME、NaturalProofs、LILA/AMPS、FormalGeo、CriticLeanBench、FormL4、MiniF2F、ProofNet、AMO-Bench、IMO-Bench、OlymMATH、Omni-MATH、MathArena、FIMO、OlympiadBench 等）二次构建；论文未声明完全开源，但提供数据源元信息与构建流程细节。
- **代码/权重**：论文未声明代码开源。
- **关键超参**：规划/反馈 temperature=0.0、top-p=1.0，max tokens 256（能力规划）/512（方案规划）/1024（下一步规划）；行动任务 temperature=0.7、max tokens 2048；形式化/定理应用 n=5 采样取多数投票；Judge 模型为 DeepSeek-V3（temperature=0）。
- **轨迹留存率**：17.3%（217 文本 + 20 多模态），过滤主因为答案错误（~76.8%）与步骤/能力多样性不足（~6%）。
