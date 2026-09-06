---
title: "HarnessDev-Can-LLMs-Create-and-Evolve-Their-Own-Agent-Harnes"
source: https://arxiv.org/pdf/2609.01437v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:40"
field: "Agent 系统与评测基准"
keywords: ["Agent Harness", "Benchmark", "LLM Agent", "Harness Evolution", "Self-Evolving Agents", "Creation and Evolution"]
innovations: ["提出 HarnessDev 基准，将评测单元从任务输出转向可复用可运行基础设施，覆盖 Creation 和 Evolution 两阶段", "设计弱种子 Harness H_seed 隔离评估模型设计执行系统的能力而非复现样板代码", "引入 Creator-Executor 分离的 Self-Eval/Unified-Eval 双视角协议及独立 Held-out 泛化评估"]
benchmarks: ["SWE-bench Pro", "Terminal-Bench 2.1", "MLE-bench", "EQ-Bench3", "BrowseComp"]
---

# 论文速读：HarnessDev-Can-LLMs-Create-and-Evolve-Their-Own-Agent-Harness

## 一句话总结
本文提出了 HarnessDev 基准，首次系统评估 LLM 从零构建可运行 Agent Harness（执行基础设施）并基于下游反馈持续迭代优化的能力，发现当前模型在 Creation 阶段可生成可用 Harness，但 Evolution 阶段的增益不稳定且泛化有限。

## 研究问题与动机
- 现有 Agent 评测将 Harness 视为实验配置而非开发对象，模型只需在既定 Harness 上完成下游任务，无法衡量模型自主构建和维护执行系统的能力。
- 随着 Agent 向部署工具演进，其能力越来越依赖模型外的执行基础设施（如 Claude Code、Codex CLI），相同模型权重在不同 Harness 下性能差异显著（如 GPT-5 在 Terminus 2 上解决 35.2% Terminal-Bench，而在 Codex CLI 上达 49.6%）。
- Harness 工程与普通代码编辑存在本质区别：修改 Harness 等于修改自身行为的执行基质，需要模型从执行轨迹中识别自身能力局限、诊断结构性瓶颈并做出累积性改进。
- 缺少一个分离 Creator 与 Executor、同时衡量可运行基础设施的 Capability 和 Efficiency 的统一评测基准。

## 核心贡献（创新点）
- 提出 HarnessDev 基准，将评测单元从"单次任务输出"转向"可复用的可运行基础设施"，覆盖 Creation（从零构建）和 Evolution（基于反馈迭代）两个阶段。
- 设计了弱种子 Harness（$H_{\text{seed}}$）的开发环境，提供完全可运行的兼容层但没有任何 agent 循环、工具策略或验证逻辑，确保任何非零得分均来自 Creator 新增的执行逻辑。
- 引入 Creator-Executor 分离协议（Self-Eval 与 Unified-Eval），隔离 Harness 质量与 Executor 能力的耦合，揭示跨模型可移植性的关键约束。
- 在 Evolution 阶段实现完整的双基准反馈集（SWE-Pro-100 + Terminal-Bench-89）与独立 Held-out 评估（SWE-Pro-630）的分离设计，精确区分在线适应与泛化能力。
- 系统报告了 6 种 Creator LLM × 4 领域 × 5 基准（共 2,207 个实例）的 Creation 结果，以及 9 条 Evolution 轨迹的完整行为分析。

## 方法详解
- **基本框架**：Creator LLM $L_C$ 在开发环境 D 中，以弱种子 Harness $H_{\text{seed}}$、任务族规范及 1-3 个开发用例为起点，构建完整可运行 Harness $H$（含执行循环 E、工具 T、上下文管理 C、状态 S、生命周期控制 L、验证 V 六个模块），冻结后由 Executor LLM $L_E$ 在下游任务 x 上执行，Evaluator J 打分。
- **弱种子设计**：$H_{\text{seed}}$ 仅提供 CLI 入口、LLM 配置解析、审计输出（result.json/trajectory.jsonl/response.md）、被动工具原语（读/写/搜索/命令执行）；无 agent 循环、无工具策略、无上下文管理、无状态持久化、无恢复逻辑，未修改时所有下游基准得分为零。
- **Creation 协议**：Creator 不可见人类实现和隐藏评测集，提交物为可运行的冻结 Harness，以 avg@3（三次独立创建均值）作为最终分数。
- **Evolution 协议**：以 RQ1 冻结 Harness $H_0$ 为起点，使用 100-task SWE-Pro + 89-task Terminal-Bench 作为反馈集；官方候选必须以双基准完整评估对提交，10 对预算；每次评估对之间最多 2 次探针（各跑 5 题）用于诊断；所有候选在冻结后额外评估 630-task 独立 Held-out 集。
- **评估指标**：Capability（下游任务成功率和原生指标得分）+ Efficiency（Executor 消耗的 tokens，排除 Creator 构建成本），两者共同衡量 Harness 质量。
- **约束合规审计**：禁止硬编码答案、从任务 ID 推导方案、访问隐藏测试/答案、替换运行时接口等，通过隔离计分路径（SWE-Pro 仅认可真实 repo diff、Terminal-Bench 仅认可最终环境状态）确保审计有效，全量审计结果为零违规。

## 实验与结果
- **数据集**：Creation 覆盖 4 领域 5 基准共 2,207 个独立下游实例（SWE-Pro 731 + Terminal-Bench 89 + MLE-bench 75 + EQ-Bench3 46 + BrowseComp 1,266）；Evolution 使用 SWE-Pro 100-task 反馈集 + 89-task Terminal-Bench 反馈集，Held-out 为 630-task SWE-Pro。
- **基线**：弱种子 Harness（零分基线）、成熟人工工程 Harness 参考（来自各基准公开排行榜），以及 6 种 Creator LLM（Opus 4.8、GPT-5.5、Gemini 3.1 Pro、DeepSeek V4 Pro、Qwen 3.7 Max、Seed 2.0 Pro）生成的 Harness。
- **Creation 主要结果（Self-Eval avg@3）**：Opus 4.8 综合得分最高（67.8/86.2），Writing（EQ-Bench3: 84.6 vs. 参考 83.7）和 MLE-bench（32.9 vs. 参考 24.0）接近或超越人工参考，Search（BrowseComp: 52.4 vs. 参考 92.2）和 Code（SWE-Pro: 69.3 vs. 参考 80.0）差距最大。MLE-bench 执行 token 消耗变化约 19 倍，高消耗不必然带来高分。
- **Unified-Eval（固定 Gemini 执行器）**：跨模型直接可比；Opus 的 SWE-Pro 从 Self-Eval 69.3 降至 33.0，暴露强 creator 协同效应；Qwen 的 BrowseComp 自 32.3 升至 49.9。
- **Evolution 主要结果**：Self-runtime 下 5 条轨迹均有 Held-out 提升（+1.43 ~ +4.44 分，平均 +3.11）；固定 Gemini 下仅 Opus 改善（+2.70），其余 3 条退化（-1.11 ~ -2.38）。64 次官方切换中仅 2/9 声明版本为 Held-out 最优，53.1% 切换反馈集与 Held-out 方向一致。
- **编辑特征**：中位声明版本修改 8 文件、+476/-38 行；58/64 切换修改执行/控制流，37/64 修改工具，仅 4/64 修改状态——反映 Evolution 更关注运行时行为而非数据层。

## 相关工作脉络
- **Agent Benchmarks（SWE-bench、GAIA、WebArena、τ-bench、AgentBench）**：评测框架假定 Harness 已固定，评测任务是给定 Harness 下的表现，与 HarnessDev 将 Harness 本身作为开发对象形成根本差异。
- **Meta-Agent Challenge [27]**：同样评估 Agent 的自主开发能力（在沙盒中迭代编程一个 agent 产物），但聚焦 Creation 单一阶段，且无 Creator-Executor 分离和执行成本衡量。
- **HarnessOpt-Bench [47]**：最接近 Evolution 概念，给定 seed harness + 分级反馈 + 固定评估预算，但仅研究优化已有 Harness，而非从零 Creation。
- **Evo-Bench [16]**：固定 runtime 模型下研究 Evolution，关注跨域最终版本质量；HarnessDev 将 Creation→Evolution 串联，且包含 self/fixed 双视角和 token 效率维度。
- **VeRO [46]、Meta-Harness [21]、Self-Harness [56]、HarnessFix [9]、HarnessCompass [58]、Harness-R1 [41]**：提出各种 Harness 优化方法或基础设施；HarnessDev 是首个评估通用前沿模型在统一 Creation+Evolution 协议下执行整个开发者角色的基准。
- **Harness Updating Is Not Harness Benefit [24]**：区分"产生有用更新"与"执行器能否利用该更新"，启发了本文 Creator-Executor 分离的 Self-Eval/Unified-Eval 设计。

## 局限性与未来方向
- **领域覆盖有限**：仅 4 个领域，Evolution 目前仅研究 Code 领域；覆盖范围不足以代表所有真实部署场景。
- **人工参考不对齐**：人类基线来自不同 Harness-模型组合（外部排行榜），非同执行器配对对照，比较存在混杂因素。
- **进化轨迹统计效力不足**：每个 Creator-运行时组合仅 1 条轨迹，无法进行显著性检验或置信区间估计。
- **Held-out 评估仅限 SWE-Pro**：Evolution 冻结后的泛化评估仅覆盖 Code 领域，缺乏跨领域的 Held-out 验证。
- **开发环境固定**：D 在整个两阶段中保持不变，未探索演化出的 Harness 能否反哺作为新一轮进化的开发环境。
- **仅衡量模型外学习**：明确说明不声称启发式学习可替代参数训练，测量边界受限。

## 研究启发与可借鉴点
- **弱种子设计范式**：提供完全可运行的框架空壳（接口完备、审计完备、零功能），强制模型从零设计执行逻辑——这一范式可迁移至任何"从基础架构开发到功能构建"的评测任务。
- **Creator-Executor 分离的两视角协议**：Self-Eval 衡量 co-design 质量，Unified-Eval 隔离 Harness 的可移植性——这一分离设计对评测任何"生成型系统"（prompt engineering、workflow 生成等）均有借鉴价值。
- **双集分离反馈-泛化评估**：Evolution 阶段用 100-task 反馈集 + 630-task 独立 Held-out 的分离设计，使研究者能精确区分在线适应与真实泛化，避免噪声驱动的过拟合评估。
- **可审计约束+审计验证机制**：通过隔离计分路径（不信任 Harness 自报告状态）实现强约束合规，结合全量轨迹审计，为后续安全相关基准设计提供了模板。
- **可与本团队方向结合的机会**：将 HarnessDev 的 Creator-Executor 分离思路应用于 prompt/workflow 自动生成评测；或将 Evolution 的 feedback-driven 迭代协议迁移到代码修复 Agent 的持续改进评测中。

## 关键术语表
**Agent Harness**：模型外部的执行基础设施，管理执行循环、工具使用、上下文管理、失败恢复和结果验证，是将模型输出转化为实际动作的系统。
**Creation（RQ1）**：Harness 开发的第一阶段，Creator LLM 从弱种子出发，基于任务规范和少量开发用例构建完整可运行 Harness。
**Evolution（RQ2）**：Harness 开发的第二阶段，Creator 基于下游执行反馈迭代改进自身已构建的 Harness，目标是提升基准性能。
**Self-Eval**：Creator 同时作为 Executor 评测自己构建的 Harness，反映 model-harness co-design 质量。
**Unified-Eval**：固定同一个 Executor（Gemini 3.1 Pro）评测所有 Creator 生成的 Harness，隔离 Harness 质量与 Executor 能力的耦合。
**Held-out 集**：在 Evolution 开发循环中从未向 Creator 展示的独立测试集（SWE-Pro-630），用于评估真实泛化而非过拟合。
**Weak Seed ($H_{\text{seed}}$)**：提供完全可运行但无任何 agent 逻辑的兼容层，未修改时在下游基准上得分为零，迫使 Creator 从零构建执行逻辑。
**Execution Token**：Harness 在下游任务执行过程中消耗的 Executor 模型 token 数，作为效率度量与任务成功率共同评估 Harness 质量。

## 可复现要素
- **数据集**：基于公开基准（SWE-bench Pro public split、Terminal-Bench 2.1、MLE-bench、EQ-Bench3、BrowseComp）；Held-out 630-task split 将与基准发布。
- **代码/权重**：项目页面 https://self-developing-agents.github.io/，审计脚本和 seed harness 实现骨架在 Appendix C.1 有代码列出；论文声明"release the audit artifacts"。
- **关键超参**：Creator LLM 运行于各自官方 API 或 OpenRouter；解码温度 0.6-1.0、top-p 0.70-0.95、top-k 20-64、max output 65K-131K tokens；MLE-bench 每任务 GPU 限制 NVIDIA A800-80GB、36,000s 总时限（agent 占 34,200s）、500-step cap；Code 基准 500-step cap + 7,200s 时限。
