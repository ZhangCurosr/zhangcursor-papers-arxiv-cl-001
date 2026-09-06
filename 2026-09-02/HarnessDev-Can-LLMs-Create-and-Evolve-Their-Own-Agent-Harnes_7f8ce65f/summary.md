---
title: "HarnessDev-Can-LLMs-Create-and-Evolve-Their-Own-Agent-Harnes"
source: https://arxiv.org/pdf/2609.01437v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:59"
field: "Agent 评测与自动化 harness 工程"
keywords: ["Agent Harness", "LLM Agent", "Benchmark", "Self-Development", "Harness Evolution", "Agent Infrastructure"]
innovations: ["提出 HarnessDev 基准，将评估单元从任务输出转向可运行 harness，区分 Creation 与 Evolution 两阶段", "设计 Weak Seed Harness 并提供 Self-Eval/Unified-Eval 双视角评测协议以隔离 harness 质量与 executor 能力", "建立能力-效率二维评估体系并建立约束合规审计机制"]
benchmarks: ["SWE-Pro", "Terminal-Bench 2.1", "MLE-bench", "EQ-Bench3", "BrowseComp"]
---

# 论文速读：HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?

## 一句话总结
论文提出 HarnessDev 基准，将 Agent 评估单元从"任务输出"转向"可运行基础设施（harness）"，测量模型从无到有构建执行系统（Creation）并利用反馈持续改进自身 harness（Evolution）的能力；实验覆盖六款前沿模型、四个领域、五项基准共 2,207 个下游实例，发现当前模型可在写作和 ML 实验领域匹配或超越人工基线，但在代码和搜索研究领域差距显著，进化增益不稳定且跨模型泛化有限。

## 研究问题与动机
- **核心问题**：LLM 能否从零构建一个可持久运行、可审计、可复用的 Agent 执行 harness，并基于下游执行反馈持续改进它？
- **现有评估的盲区**：当前 Agent 评测（SWE-bench、GAIA、WebArena 等）将 harness 视为实验配置的一部分，只报告在选定 harness 下的任务表现，忽略了 harness 本身是需要被设计和演化的"工件"。
- **harness 工程与常规编程的本质差异**：修改普通程序时目标是外部分配的、可局部验证的；修改自身 harness 是在编辑它自己赖以运行的"执行基底"，改变模型未来所有任务中的观察、规划和恢复方式，需要识别自身行为瓶颈、诊断结构性缺陷并作出可持续累积的改动。
- **工业界现实支撑**：Forward-Deployed Engineer（FDE）角色的高速增长反映了"有能力的模型 ≠ 可运行的系统"，当前鸿沟靠人工工程师填补，paper 探索模型能否部分承担这一角色。

## 核心贡献（创新点）
1. **提出 HarnessDev 基准，将评估单元从任务答案转向可运行 harness**：划分 Creation 和 Evolution 两阶段，前者要求模型从弱种子构建完整执行系统，后者要求基于下游反馈迭代改进；与 Meta-Agent Challenge 和 HarnessOpt-Bench 相比，HarnessDev 同时覆盖从0构建到持续演化全流程，并分离 creator 与 executor 模型角色。
2. **设计 Weak Seed Harness（弱种子 harness）**：提供仅含 CLI 解析、审计写人和被动原语的执行框架骨架，不含 agent 循环、工具策略、上下文管理、状态、恢复逻辑或停止规则；任何非零 Creation 得分必然来自创作者添加的执行逻辑，避免模型仅复现 benchmark boilerplate。
3. **引入 Self-Eval 与 Unified-Eval 双视角评测协议**：Self-Eval 让 creator 自身作为 executor 评估自建 harness（衡量 co-design），Unified-Eval 固定使用 Gemini 3.1 Pro 作为 executor（隔离 harness 质量与 executor 能力），可量化 harness 跨执行器的可移植性。
4. **建立能力-效率二维评估体系**：除下游任务成功率/奖牌率/准确率外，还报告 executor-model 消耗量（execution tokens），揭示"更高成本不必然带来更高分"的设计权衡。
5. **严格的约束合规审计机制**：评分路径与 harness 隔离（SWE-Pro 凭证仅来自真实 repo diff，Terminal-Bench 仅来自最终环境状态），harness 自报状态不参与评分；对所有提交 harness 源码和轨迹进行事后审计，本文报告中未发现任何违规得分案例。

## 方法详解
- **公式框架**：$(L_C, D) \to H, \quad (H, L_E, x) \to y \xrightarrow{J} \text{score}$，其中 $L_C$ 为 creator LLM，$D$ 为开发环境（Claude Code 2.1.177），$H$ 为产出的可运行 harness，$L_E$ 为冻结后的 executor LLM，$J$ 为 scorer。
- **Harness 抽象**：$H = \langle E, T, C, S, L, V \rangle$，分别对应执行循环、工具策略、上下文管理、状态与记忆、生命周期/恢复、结果验证六大模块。
- **Creation（RQ1）**：每个 creator 收到相同的 $H_\text{seed}$、任务族规范、工具约束、设计教程及 1-3 个 development cases；不得访问人类实现或隐藏评测集。每对 creator-benchmark 独立创建 3 个 harness，报告 avg@3。
- **Evolution（RQ2）**：以 RQ1 产出的冻结 harness $H_0$ 为起点，使用 100-task SWE-Pro 反馈集 + 全部 89-task Terminal-Bench 作为反馈；每轮完整评测需提交一对（SWE-Pro 100 + Terminal 89）共消耗 1 个预算单元，总额度 10 对；两次正式评测之间最多 2 次 probe（固定前 5 任务，仅诊断用）。
- **数据集构成**：四个领域、五项下游基准（SWE-Pro 731、Terminal-Bench 2.1 89、MLE-bench 75、EQ-Bench3 46、BrowseComp 1,266），合计 2,207 个唯一下游实例；Evolution 的 held-out 集为 630 个 SWE-Pro 实例（与 100-task 反馈集不相交）。
- **约束检查**：禁止硬编码实例级解法、从 task ID/文件名/已知答案推导 patch、访问隐藏测试/答案/评分器内部信息；通过分离评分路径与 harness 实现可检查而非 advisory 性质。

## 实验与结果
- **Creator 模型**：Opus 4.8、GPT-5.5、Gemini 3.1 Pro、DeepSeek V4 Pro、Qwen 3.7 Max、Seed 2.0 Pro。
- **Creation — Self-Eval 关键数字**（表3）：Opus 4.8 总均分最高（67.8），但仍低于人工基线（86.2）；Writing（EQ-Bench3）Opus 达 84.6 vs 人工 83.7，ML 实验（MLE-bench）Opus 32.9% 奖牌率 vs 人工 24.0%，Search（BrowseComp）差距最大（52.4% vs 92.2%）；Code（SWE-Pro）Opus 69.3% vs 人工 80.0%。
- **Creation — Unified-Eval**（表4）：固定 Gemini executor 后排名显著变化；Opus SWE-Pro 从 69.3 降至 33.0，DeepSeek/MLE-bench 从 19.6 降至 9.8，Qwen BrowseComp 从 32.3 升至 49.9；说明部分 harness 与原 executor 过拟合。
- **执行成本差异巨大**：MLE-bench 上 token 消耗相差约 19 倍（GPT-5.5 达 19.1% 奖牌率仅需 29.3M tokens，DeepSeek 达 19.6% 需 208.4M tokens），高成本不保证高分。
- **Code harness 编辑统计**（表5）：18 个 Code artifact 共净增 17,111 LOC；Gemini 仅加 1,006 行却获得最佳 Terminal-Bench 成绩（68.8），说明精准修改比代码量更重要。
- **架构证据缺口**（图5）：18 个 Code harness 中 13/18 实现执行循环、13/18 实现工具策略、15/18 实现验证；但仅 11/18 定义 State 类、仅 1 个暴露 state-saving 接口、仅 1 个实现定期 checkpointing（26,679 条轨迹中无任何 checkpoint 事件）。
- **Dead code 普遍**：108 个 Code 组件中 18 个从未触发；587 个 Writing 特性中 124 个确认为死代码；self-test 计数与下游分数的 Spearman 相关仅 0.13–0.26，而 revision 调用次数相关达 0.57（p ≤ 0.0005）。
- **Evolution — 反馈集增益**（表6）：五组 self-runtime 均在可见反馈集上提升，Opus 自 $H_0$ 到最终版本 SWE-Pro 平均对分 +3.0pp、Qwen +13.9pp、DeepSeek +13.4pp；固定 Gemini 下 Opus +9.7pp，Qwen 仅 +1.1pp，DeepSeek +6.5pp。
- **Evolution — Held-out 泛化**：五组 self-runtime 在 630 实例 held-out 上均有改善（+1.43 ~ +4.44pp，均值 +3.11pp）；固定 Gemini 下仅 Opus 改善（+2.70pp），Qwen（−1.11pp）和 DeepSeek（−2.38pp）回退。
- **不稳定现象**：64 次官方切换中 8 次双基准回退、27 次增益在重复运行噪声带内（±4.75pp）、同一次 commit 也可波动 ±4.75pp；169 个新增函数/类中 113 可达、31 仅通过死代码可达、25 无调用者。
- **版本选择不可靠**：9 条 lineage 中仅 2/9 声明的最终版本是 held-out 最优；反馈集分数与 held-out 分数仅 53.1% 同向变动，说明反复优化噪声分数易导致过拟合。

## 相关工作脉络
- **Agent 评测基准族**（SWE-bench、GAIA、WebArena、τ-bench、AgentBench）：将 harness 作为实验配置固定，评估模型在选定 harness 下的任务表现，与 HarnessDev 形成"评测 harness 下模型"vs"评测模型构建 harness 能力"的对比。
- **Harness-Bench [55]**：测量 harness 选择如何影响模型表现，关注 harness 对模型的作用；HarnessDev 更进一步评估模型自身构建和演化 harness 的能力，并测量执行成本。
- **Meta-Agent Challenge [27]**：meta-agent 在沙箱中迭代编程 agent artifact，与 Creation 最相近；但 HarnessDev 同时覆盖 Creation→Evolution 全链路，分离 creator/executor 角色，并测量跨 executor 泛化。
- **HarnessOpt-Bench [47]（同期工作）**：LLM optimizer 接收 seed harness 和分级反馈，在不可测测试集上优化；Focus 是优化已有 harness，而 HarnessDev 连接 from-scratch Creation 到 Evolution，并在相同冻结 artifact 下测试 self 和 fixed runtime。
- **Evo-Bench [16]**：evolver model 在固定 runtime 下改进共享 CodeAct seed，侧重跨领域最终版本质量；HarnessDev 连接 Creation+Evolution，含 self/fixed runtime 双视角、执行 token 成本和跨 executor 泛化，并在 trajectory 全程对 held-out 集评估每一冻结版本。
- **自动化 harness 改进方法**（VeRO、Meta-Harness、Self-Harness、HarnessFix、DemoEvolve、HarnessCompass、Harness-R1）：提出具体改进方法或基础设施；HarnessDev 不提出新方法，而是以统一 Creation+Evolution 协议评估通用前沿模型承担该开发者角色的能力上限。

## 局限性与未来方向
- **领域覆盖不全**：仅四个 domain，人类 baseline 不保证最优，Unified-Eval 虽减少但不能完全消除 executor-model 交互差异。
- **Evolution 样本有限**：每个 creator-runtime 组合仅一条 trajectory，一篇 main-runtime 未完成；post-freeze held-out 仅覆盖 SWE-Pro，无法支持不确定性估计或群体级比较。
- **开发环境固定**：D 在两个阶段均保持不变，evolved harness 自身能否作为下一代 development environment 有待未来探索。
- **不含参数训练**：HarnessDev 测量的是 model-external learning，不声称 heuristic learning 可替代参数训练。
- **未做 matched-search 对照**：进化可能只是 blind search，future work 需设置 matched-budget 比较。

## 研究启发与可借鉴点
1. **Weak Seed Harness 设计范式**：提供一个仅含 CLI 解析、审计写人和被动原语的"空壳"而非空仓库或成熟 agent，可分离"基础设施搭建"与"执行策略设计"两个变量，值得迁移至其他 harness/agent 自动化研究。
2. **Self-Eval vs Unified-Eval 双视角协议**：将"creator 与 executor 是否同构"作为实验变量，可有效诊断 harness 是否与特定 executor 过拟合，这一设计可复用于任何涉及"生成式执行系统"的评测。
3. **能力-效率二维评估**：同时报告任务分数和 executor tokens，揭示"更高成本 ≠ 更好结果"的 trade-off，为后续研究提供成本感知的 benchmark 设计范式。
4. **约束合规的可检查机制**：通过隔离评分路径与 harness（凭证仅来自真实 repo diff / 最终环境状态）实现 hard constraint 而非 advisory，审计脚本与 artifact 开源可复现，这一设计对其他 agent 安全/对齐评测具有参考价值。
5. **Evolution 轨迹分析指标体系**：edit 规模与性能提升相关性弱（edit size 不预测结果），但 revision 调用次数相关系数达 0.57，提示"迭代诊断-修改-验证"循环比单次大改更有效；这一发现可直接指导未来 harness evolution 的方法设计。

## 关键术语表
**Agent Harness**：包裹在 LLM 外部的执行基础设施，包含执行循环、工具策略、上下文管理、状态/记忆、生命周期/恢复和结果验证等模块，将模型输出转化为可执行动作。
**Creation（RQ1）**：HarnessDev 第一阶段的评估任务，模型从弱种子 harness 出发，利用任务规范和少量 development cases 构建完整的可运行 harness。
**Evolution（RQ2）**：HarnessDev 第二阶段的评估任务，模型基于下游执行反馈（100-task SWE-Pro + 89-task Terminal-Bench）迭代改进已创建的 harness，评估其在不相交 held-out 集上的泛化能力。
**Self-Eval**：Creator LLM 同时作为 executor 运行自建 harness 的评测模式，反映 creator-harness 协同设计的整体效果。
**Unified-Eval**：固定使用同一 executor（Gemini 3.1 Pro）运行所有 creator 产出的 harness 的评测模式，用于隔离 harness 质量与 executor 能力。
**Weak Seed Harness（$H_\text{seed}$）**：Benchmark 提供给所有 creator 的统一起始 harness，仅提供 CLI 解析、审计写人和被动工具原语，不含执行循环、工具策略、状态管理或恢复逻辑，未修改时下游得分为零。
**Held-out 集**：在 Creation 或 Evolution 阶段对 creator 完全不可见、仅在冻结后用于评估泛化性能的测试实例集合（SWE-Pro 630 实例）。
**Execution Token（executor tokens）**：harness 在冻结后运行下游任务时 executor LLM 消耗的 token 总量，不计入 creator 开发过程的 token，用于衡量 harness 的运行效率。

## 可复现要素
- **数据集**：Creation 使用五项公开基准（SWE-Pro、Terminal-Bench 2.1、MLE-bench、EQ-Bench3、BrowseComp），Evolution 的 held-out 集为 SWE-Pro 公共 split 中 630 个实例（与 100-task 反馈集不相交）；论文声明公开 benchmark 及审计 artifacts，具体 URL 见 Project Page: https://self-developing-agents.github.io/。
- **代码/权重**：基准实现、seed harness 骨架、审计脚本和 held-out 任务 split 随 benchmark 发布（论文声明；具体 GitHub 地址待项目页确认）。
- **关键超参**：Development env 固定为 Claude Code 2.1.177（GPT-5.5 用 Codex 0.144.3）；Self-Eval 每 creator-benchmark 独立创建 3 次取 avg@3；Evolution 预算为 10 对完整评测 + 每轮最多 2 次 probe（固定前 5 任务）；MLE-bench 每任务 500 步 / 34,200s harness 预算；Code/Data 每任务 500 步 / 7,200s 限制。
