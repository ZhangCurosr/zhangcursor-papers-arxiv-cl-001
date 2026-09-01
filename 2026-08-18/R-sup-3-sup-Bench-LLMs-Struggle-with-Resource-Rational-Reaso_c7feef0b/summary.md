---
title: "R-sup-3-sup-Bench-LLMs-Struggle-with-Resource-Rational-Reaso"
source: https://arxiv.org/pdf/2608.16033v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:44:11"
field: "LLM 推理与资源分配评测"
keywords: ["Resource Rationality", "Shared Budget", "LLM Benchmark", "Tool-Free Reasoning", "Agentic Evaluation", "Resource Allocation", "Response Curve Oracle"]
innovations: ["提出R³-BENCH首个共享预算suite级基准，通过matched single-problem response curve oracle量化已展示能力与共享预算下实现能力之间的gap", "设计模型特定预算校准与双基准诊断（oracle + equal-allocation replay）分离能力与分配质量", "系统归因pressure-dependent failure模式并发现轻量在线调度策略跨领域不通用"]
benchmarks: ["R³-BENCH", "Omni-MATH", "MathNet", "LiveCodeBench Pro", "Reasoning Gym"]
---

# 论文速读：R³-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets

## 一句话总结
本文提出 R³-BENCH，首次在共享预算（shared budget）设置下系统性评测 LLM 在数学、编程、抽象推理三领域的资源理性分配能力，发现当前 SOTA 模型在共享预算下的实际得分显著低于其自身单题能力的离线最优分配上界，揭示出"能力被展现"与"能力被实现"之间存在持续且广泛的差距。

## 研究问题与动机
- **核心问题**：当多个任务共享有限计算资源时，LLM 能否有效分配资源以实现单题已展示能力的最大化？即是否存在"资源理性分配能力"？
- **现有方法不足**：
  - 传统推理基准（如 MATH、GPQA）和 Agent 基准（如 SWE-bench、GAIA）均以独立每任务预算评估，模型可对每个问题花费全额预算，不产生跨任务机会成本。
  - 虽有预算感知方法（如 SelfBudgeter、BATS、USACOArena、CLEAR），但均聚焦单任务内预算自适应或跨模型/流水线调度，**未将多题竞争同一共享预算的行为与同模型单题能力对照**。
  - 已有工作不校准 suite 表现与模型自身单题能力之间的关系，无法分离"不会做"与"做错了分配"。

## 核心贡献（创新点）
1. **构建首个共享预算 suite 级基准 R³-BENCH**：在数学、编程、抽象推理三个领域各构造 50 个六题 contest，涵盖无工具（tool-free）与 agentic 两类设置，引入机会成本机制，使跨题分配成为推理问题本身的一部分。
2. **设计 matched single-problem response curve oracle 与 equal-allocation replay 作为离线诊断器**：通过单题响应曲线构建离线多项选择背包 oracle（每问题选择一个预算级别），以及等分重放（equal allocation replay），在不运行新轨迹的前提下量化"已展示能力 vs. 共享预算下实现能力"的 Gap，而非简单比较绝对分数。
3. **发现持续性分配差距并系统归因**：72 个主要单元格中 oracle 在 71 个严格高于 contest 得分；强压力下主要失败模式为"预算花在其他题上"，中等压力下为"在部分进展后放弃"；轻量在线调度策略（Strategy A/B）可部分缩小差距但跨领域不通用。
4. **证明 Gap Ratio 不能由综合能力指数解释**：在 ECI（综合 50+ 基准）相近的模型对之间，Gap Ratio 差异可达 60.6 个百分点，表明该 benchmark 捕捉到现有评测体系之外的独立维度。

## 方法详解
- **问题池与难度分层**：数学（Omni-MATH + MathNet）、编程（LiveCodeBench Pro）、抽象推理（Reasoning Gym），各 300 题；以三个参考模型（DeepSeek V4 Pro、GLM-5.2、GPT-5.5）的平均输出长度为难度指标：最短 150 题为 Easy，次 100 题为 Medium，剩余 50 题为 Hard。难度标签不展示给被测模型。
- **Contest 构造**：每领域 50 个六题 contest，固定配比（3 Easy + 2 Medium + 1 Hard），题目呈现顺序随机化。
- **模型特定预算校准**：以各模型无约束基准 $R_{m,d}^{\infty}$（仅统计成功完成的 contest）为基准，设置压力水平 $\rho \in \{0.2, 0.8\}$，即 $R_{m,d}^{\rho} = \rho \cdot R_{m,d}^{\infty}$，使不同模型在**相同相对压力**下测试，而非同等绝对预算。
- **Tool-free 设置**：输出 token 预算（max tokens），一次性自由格式完成，每题有独立答案格式（数学 `\boxed{...}`、编程 C++17 代码块、AR `<answer>...</answer>`）。
- **Agentic 设置**：基于 Harbor/Terminus-2 环境，action 预算（计数型工具调用），含 `focus problem <id>` / `shelve problem` 记账命令（免费），区分 counted action（python/g++/编译/测试等计算操作）与 free action（文件写入、环境检查等）。
- **Two-stage 协议（推理模型）**：Stage 1 为预算受限推理（含 thinking tokens），Stage 2 为 trace-only finalizer（无原始题目、无 thinking），仅从 trace 中提取答案，避免格式化混淆。
- **响应曲线与 Oracle**：每个问题在每个压力级别 $\rho \in \{0.05, 0.1, 0.2, 0.4, 0.8\}$ 上做 $K=5$ 次独立单题运行，得到响应率 $q_p(\ell)$；Oracle 求解多项选择背包：$\max \sum_{p,\ell} q_p(\ell) z_{p,\ell}$，s.t. $\sum b^\ell z_{p,\ell} \le B$，每问题仅选一级（含零级），共 $6^6 = 46,656$ 种枚举，精确求解。
- **Gap 度量**：$\Delta_{RR} = \text{Oracle} - \text{Contest}$，Gap Ratio $= \Delta_{RR} / \text{Oracle}$（Oracle > 0 时）。
- **轨迹诊断**：标注 oracle-selected miss 的主因（never attempted / attempted too late / stopped after partial progress / spent budget elsewhere / misread tool feedback / wrong format / genuinely unsolved）及在线适应率（difficulty observation → substantive strategy update → resource-rational update）。
- **在线调度干预**：Strategy A（强制 initial coverage：在每题花费第二次付费 step 前先对所有可见题做一次 probe）；Strategy B（A + cheap verification gate：稳定候选物最多一次额外付费 check）。

## 实验与结果
- **数据集**：Math（Omni-MATH + MathNet）、Code（LiveCodeBench Pro）、AR（Reasoning Gym），各 300 题，构造 50 个 contest/领域。
- **评测模型**：6 个旗舰模型（DeepSeek-V4-Pro、Qwen3.7-Max、GLM-5.2、Hy-3、GPT-5.5、Claude-Opus-4.8）+ 2 个基线（DeepSeek-Chat、DeepSeek-Reasoner），共 8 个。
- **关键数字**：
  - **72 个 model–setting–pressure–domain 单元格中，oracle 在 71 个严格高于 contest 得分**，揭示普遍性分配差距。
  - 工具自由（tool-free）平均 gap = **1.16 题/contest**（12 个 model–pressure 对）。
  - Agentic Gap Ratio 在 27/36 匹配比较中低于 tool-free；提高 ρ 从 0.2→0.8 在 23/36 中降低 Gap Ratio。
  - **强压力（ρ=0.2）下**：GLM（42%）、Hy（95%）、GPT（66%）、Opus（59%）的主要失败为"spent budget elsewhere"；Qwen 以"attempted too late"为主（40%）；DS-Pro 分散于 finalization（27%）和 misread feedback（24%）。
  - **中等压力（ρ=0.8）下**：DS-Pro（67%）、GLM（38%）、Hy（38%）、GPT（55%）主要为"stopped after partial progress"。
  - GPT-5.5 在 AR 强压力 tool-free Gap Ratio 达 **82.47%**（Contest 0.34 vs Oracle 1.94）。
  - Hy-3 agentic ρ=0.2 AR：Contest 1.22 vs Oracle 3.14，Gap Ratio **61.15%**。
  - Claude-Opus-4.8 表现最稳定：agentic ρ=0.8 下 Math Gap Ratio 仅 **3.39%**，Code **11.02%**，AR **4.12%**。
- **位置效应**：所有 12 个 model–pressure 系列中，Position 6 准确率低于 Position 1；强压力下 Position 1 消耗 20.5–52.9% 的 attributed output，中等压力为 16.9–23.5%。
- **上下文压力诊断**：Target-in-suite 实验显示额外 suite 上下文对目标题影响均值为 **1.1 个百分点**，最大 8.3 个百分点，非统一解释。
- **在线调度结果**（3 模型 × 3 领域）：A 或 B 在 6/9 单元格优于 contest reference，但无统一优势策略；Code 在所有模型上均改善，Math 和 AR 结果依赖模型。

## 相关工作脉络
- **Wang et al. (2024) / Han et al. (2025) / Li et al. (2025)**：Per-task 预算自适应方法，优化单题内的 token/采样/工具预算，不涉及多题共享同一预算的竞争。
- **Liu et al. (2025) / BATS**：预算感知 tool-use agent，但每个任务独立预算，不存在跨题机会成本。
- **USACOArena（Zhou et al., 2026）**：credit-budgeted 编程 arena，具有共享预算思想，但**未与同模型单题能力对照**，缺少 response-curve oracle 诊断框架。
- **CLEAR（Wan et al., 2026）**：用外部 utility model + shadow-price 优化 token 分配，属优化式调度，与本文实证诊断范式不同。
- **SelfBudgeter（Li et al., 2025）**：单题自适应 token 分配，不跨题。
- **General AgentBench / ZEBRA**：均为 per-task 或跨模型编排评估，不评测共享预算下多题竞争的资源分配能力。

## 局限性与未来方向
- **Response-curve oracle 为离线诊断器，非可执行策略**：受限于 5 个 grid level 和 K=5 次重复的有限采样，oracle 分数非理论性能上界。
- **Equal-allocation replay 基于 realized cost，非新施加硬上限的重运行**：无法保证重运行时一定复现相同成功。
- **在线调度干预仅测试了两种轻量 directive，未探索更复杂策略**：发现 Strategy B 在 Code 上劣于 A，说明directive 复杂性非单调改进。
- **模型接口限制**：GPT-5.5 和 Claude-Opus-4.8 因 API 不暴露 chain-of-thought，在 tool-free 设置中 thinking 被禁用，可能影响公平比较。
- **未来方向**：训练内在线资源分配 policy（通过 RL/微调，使模型能根据领域、进度、剩余预算动态决策 cover/continue/switch/verify/stop）。

## 研究启发与可借鉴点
1. **响应曲线 oracle + 等分重放的双基准诊断设计**值得复用：通过单题能力数据构建离线最优分配上界，可直接应用于其他需要分离"能力"与"分配质量"的研究场景（如 multi-task agent、multi-turn reasoning）。
2. **模型特定预算校准方法**（$R_{m,d}^{\rho} = \rho \cdot R_{m,d}^{\infty}$）解决了跨模型绝对预算不公平问题，可迁移至任何需要跨模型对比资源约束下表现的评测设计。
3. **轨迹级因果归因标注体系**（oracle-selected miss 主因分类 + 在线适应事件标注）提供了系统性的 failure mode 分析框架，可用于 agent 行为诊断研究。
4. **在线调度干预实验**（coverage guard + verification gate）展示了低成本外部约束可部分弥补模型内部调度缺陷，为后续 policy 训练提供了可行的干预起点和 ablation 设计思路。
5. **Gap Ratio 与 ECI 的低相关性**提醒团队：综合基准排名不能预测分配能力，需在评测设计中加入专门的"分配效率"维度，可作为创新评测设计的切入点。

## 关键术语表
- **Resource Rationality（资源理性）**：将认知视为在有限计算资源约束下最优利用资源的决策过程（Anderson, 1991；Lieder & Griffiths, 2020）。
- **Response-Curve Oracle（响应曲线 Oracle）**：基于单题在不同预算级别下的观测成功率，求解离线多项选择背包问题的最优分配器，作为已展示能力的上界参考。
- **Gap Ratio**：$(\text{Oracle} - \text{Contest}) / \text{Oracle}$，量化共享预算下已展示能力未实现的相对比例，越低越好。
- **Equal-Allocation Replay（等分重放）**：将 contest 预算均分为六份，检查每题是否在单题运行中曾出现 cost ≤ 1/6 预算的成功尝试，作为固定均匀分配的对照基线。
- **Oracle-Selected Miss（Oracle 选定遗漏题）**：被 oracle 判定应获得正预算分配但在实际 contest 中未解出的题目。
- **Resource-Rational Strategy Update（资源理性策略更新）**：agent 基于新观察到的困难度/反馈/剩余预算，对目标、方法、执行顺序、验证或资源分配做出实质性修订的行为。
- **Pressure Level ρ**：共享预算与模型无约束基准使用量的比值，ρ=0.2 为强压力（保留 20% 自然使用量），ρ=0.8 为中等压力。
- **Counted Action（计费操作）**：agentic 设置中消耗预算单位的工具调用（计算/编译/测试等），与免费 bookkeeping 操作相区分。

## 可复现要素
- **数据集**：Omni-MATH、MathNet、LiveCodeBench Pro、Reasoning Gym（均为公开数据集）；R³-BENCH 的 contest 构造代码及 prompts 随 benchmark artifact 开源。
- **代码/权重**：论文标注 Code + Dataset 开源，具体仓库见论文提供的链接。
- **关键超参**：压力水平 ρ ∈ {0.2, 0.8}；响应曲线 grid 级别 ρ ∈ {0.05, 0.1, 0.2, 0.4, 0.8}，每级别 K=5 次独立运行；每 contest 5 次独立重复；难度分层阈值基于三个参考模型的平均输出长度。
- **模型配置**：各模型 sampling 参数（temperature/top-p/reasoning-effort）遵循官方推荐，未在 R³-BENCH 上 fine-tune；GPT-5.5 和 Claude-Opus-4.8 因接口限制禁用 thinking（详见 Appendix I Table 13）。
