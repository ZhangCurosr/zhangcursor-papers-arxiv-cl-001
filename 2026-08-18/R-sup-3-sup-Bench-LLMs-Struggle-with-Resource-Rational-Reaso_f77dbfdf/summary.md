---
title: "R-sup-3-sup-Bench-LLMs-Struggle-with-Resource-Rational-Reaso"
source: https://arxiv.org/pdf/2608.16033v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:44:22"
field: "大语言模型评测与推理"
keywords: ["资源理性推理", "共享预算", "LLM评测", "任务调度", "Agent基准测试", "资源分配"]
innovations: ["提出首个多任务共享预算的 R³-BENCH 基准，同步覆盖 tool-free 和 agentic 两种设置", "构造基于单题响应曲线的离线 Oracle（多选背包）量化能力兑现缺口", "揭示压力依赖型失败模式（强压：预算耗尽他题；中压：半途而废）及调度干预的领域异质性"]
benchmarks: ["R³-BENCH", "Omni-MATH", "LiveCodeBench Pro", "Reasoning Gym", "MathNet"]
---

# 论文速读：R³-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets

## 一句话总结
本文提出了首个针对大语言模型**共享计算预算下资源理性推理能力**的评测基准 R³-BENCH，通过在数学、编程和抽象推理三个领域构建六题共用一预算的竞赛场景，揭示当前主流 LLM 在跨问题资源分配中存在持续的能力兑现缺口。

## 研究问题与动机
- **核心问题**：当多个任务共享有限计算预算时，LLM 能否有效分配资源以最大化整体收益？即模型能否将其在单题上已展现的能力充分实现在共享预算场景下？
- **现有方法不足**：
  - 推理基准（如 MATH、GPQA）和 Agent 基准（如 SWE-bench、GAIA）均采用**独立单任务预算**，无法观测跨任务的资源分配决策。
  - 已有预算感知方法仅在单个问题内部做采样/token/工具调用自适应，未涉及多问题间的资源共享与机会成本权衡。
  - 缺乏将"单题能力上限"与"共享预算下的实际表现"进行对照的诊断机制。

## 核心贡献（创新点）
- **提出 R³-BENCH 共享预算评测框架**：构建涵盖数学、竞技编程、抽象推理三领域的六题竞赛场景，每个竞赛共享统一预算，同时支持 tool-free 和 agentic 两种评估模式。与已有工作本质区别在于首次引入"多问题竞争单一预算"的设定，显式建模跨任务机会成本。
- **构造匹配单题响应曲线的离线实证 Oracle**：基于各模型在单题上的响应曲线构建最优分配离线 Oracle（形式为多选背包问题），用于量化"理论上可实现的最高得分"与"实际竞赛得分"之间的差距。与已有工作的区别在于该 Oracle 以模型自身单题能力为基准，不依赖外部标答或假设性能上界。
- **揭示"展示能力 vs. 共享预算实现"之间的持续性缺口**：在 72 个主表单元格中，Oracle 在全部 72 个单元格中均达到或超越竞赛均值，且仅在 1 个单元格持平。这一发现表明当前 LLM 的跨问题资源分配能力存在系统性不足，而非单纯能力差距所致。
- **压力依赖型失败模式诊断与在线调度干预实验**：通过轨迹诊断发现强压力下主要失败原因为"预算消耗于其他问题"，中等压力下主要为"半途而废"；轻量级在线调度策略（覆盖守卫+验证门控）在 Code 领域有效但在 Math/AR 领域效果有限，证明不同领域对调度信号的利用差异。

## 方法详解
- **任务套件构建**：
  - 从 Omni-MATH、MathNet、LiveCodeBench Pro、Reasoning Gym 各抽取 300 道题目，按三款参考模型（DeepSeek V4 Pro、GLM-5.2、GPT-5.5）的平均输出长度进行难度分层：最短 150 题为 Easy，次 100 题为 Medium，最长 50 题为 Hard。
  - 每个领域构建 50 套六题竞赛，固定难度分布（3 Easy + 2 Medium + 1 Hard），呈现顺序随机化。
- **预算校准**：
  - 预算基于各模型自身无约束基线消耗标定：$R_{m,d}^{\rho} = \rho \cdot R_{m,d}^{\infty}$，其中 $\rho \in \{0.2, 0.8\}$ 分别对应强/中等预算压力。
  - Tool-free 模式下预算单位为输出 token 数；Agentic 模式下预算单位为 counted action 数（计算类工具调用）。
- **两种评估设置**：
  - **Tool-free 推理**：模型接收提示后生成自由格式完成，预算限制为 max tokens。
  - **Agentic 推理**：在 Harbor/Terminus-2 环境中运行，模型可使用 bash、编译器、解释器等工具，每个 counted action 消耗 1 预算单位。添加 `focus problem` 和 `shelve problem` 命令用于问题级归因（不消耗预算）。
- **响应曲线 Oracle 构造**：
  - 在每个问题上进行 5 轮 × 5 个预算级别（$\rho \in \{0.05, 0.1, 0.2, 0.4, 0.8\}$ + 零选项）的独立单题运行，记录成功率 $q_p(\ell)$。
  - Oracle 求解多选背包问题：$\max \sum_{p,\ell} q_p(\ell) \cdot z_{p,\ell}$，约束 $\sum_{p,\ell} b_\ell \cdot z_{p,\ell} \leq B$，$\sum_\ell z_{p,\ell}=1$。
  - 由于每竞赛仅 6 题、每题 6 个选项，直接枚举 $6^6=46656$ 种分配方案精确求解。
- **关键指标**：
  - $\Delta_{RR} = \text{Oracle} - \text{Contest}$（资源理性缺口）
  - $\text{Gap Ratio} = \Delta_{RR} / \text{Oracle}$（归一化缺口）
- **轨迹诊断**：对 Oracle-selected miss（被 Oracle 选中但实际未解决的题目）标注主要原因：从未尝试、尝试过晚、半途而废、预算耗尽于他题、误读工具反馈、格式错误等；同时标注在线策略更新率和资源理性更新率。

## 实验与结果
- **评测模型**：DeepSeek-V4-Pro/Chat/Reasoner、Qwen3.7-Max、GLM-5.2、Hy-3、GPT-5.5、Claude-Opus-4.8 共 8 款前沿 LLM。
- **主要结果**（Table 2，6 款旗舰模型）：
  - **Tool-free 设置**：Oracle 在全部 12 个 model-pressure 配对中均高于 Contest 平均分，平均缺口为 1.16 题/套。例如 DeepSeek-V4-Pro 在 Math $\rho=0.2$ 时 Contest=2.34，Oracle=3.94，Gap Ratio=40.61%。
  - **Equal-allocation replay**：在 $\rho=0.8$ 下，4 款模型（DeepSeek-V4-Pro、Qwen3.7-Max、GLM-5.2、Claude-Opus-4.8）的均分 replay 超过实际 Contest 表现，暴露模型自身分配策略的次优性。
  - **Agentic 设置**：27/36 匹配比较中 Gap Ratio 低于 tool-free 设置；23/36 中 $\rho=0.8$ 低于 $\rho=0.2$，说明交互反馈和更宽松预算可缓解但无法消除缺口。
  - **最强结果**：Claude-Opus-4.8 在 agentic $\rho=0.8$ 下取得最低 Gap Ratio（Math 3.39%、Code 11.02%、AR 4.12%），DeepSeek-V4-Pro 在 agentic $\rho=0.2$ Math 上仅 15.15%。
- **失败模式诊断**（Figure 6）：
  - 强压力（$\rho=0.2$）下最大失败类别为"spent budget elsewhere"（GLM 42%、Hy 95%、GPT 66%、Opus 59%）。
  - 中等压力（$\rho=0.8$）下最大失败类别转为"stopped after partial progress"（DS-Pro 67%、GLM 38%、Hy 38%、GPT 55%）。
- **呈现位置效应**（Figure 4）：所有 12 个 model-pressure 系列中，第 6 题准确率均低于第 1 题，且预算在前几题高度集中（强压力下第 1 题吸收 20.5%–52.9% 输出）。
- **一般能力无法解释缺口**（Section 4.4）：ECI 综合 50+ 基准，相邻 ECI 分数模型间 Gap Ratio 差异可达 9.6–21.0 个百分点；部分高 ECI 模型反而有更大缺口（如 tool-free Math $\rho=0.2$ 中 ECI 最高的模型 Gap Ratio=43.75%）。
- **在线调度干预**（Table 3，agentic $\rho=0.2$）：
  - 策略 A（覆盖守卫）和 B（覆盖+验证门控）在 9 个 model-domain 单元格中 6 个优于基准，但无策略在所有领域占优。
  - Code 领域在三种模型上均改善，Math/AR 领域效果因模型而异，Hy-3 在某些 Cell 中被调度策略伤害。

## 相关工作脉络
- **Per-task reasoning/agent evaluation**：Hendrycks et al. (2021) MATH、Rein et al. (2023) GPQA、Zheng et al. (2025) LiveCodeBench Pro、Liu et al. (2024) AgentBench、Jimenez et al. (2024) SWE-bench——均为独立单任务评测，未建模跨任务资源竞争。
- **Budget-aware reasoning**：Wang et al. (2024)、Han et al. (2025) SelfBudgeter、Li et al. (2025)、Liu et al. (2025) BATS——仅在单题内部自适应 token/采样/工具预算，不涉及多题共享预算。
- **Resource allocation across models/pipelines**：Hamri & Talgam-Cohen (2026) ZEBRA——跨模型编排的资源分配，而非单模型多问题间分配。
- **USACOArena**（Zhou et al., 2026）： Credit-budgeted ICPC-style coding，涉及共享预算但仅覆盖编程领域且无单题能力对照。
- **CLEAR**（Wan et al., 2026）：基于外部 utility model 和 shadow-price 的 token 分配优化，但未以单题能力为基准测量分配质量。
- **定位差异**：R³-BENCH 是唯一同时具备"共享预算"、"三领域覆盖"、"tool-free+agentic 双设置"、"以模型自身单题响应曲线为 Oracle 对照"四大特征的基准。

## 局限性与未来方向
- **Oracle 为离线诊断而非可执行策略**：Oracle 使用名义预算而非实际消耗成本作为代价，且继承 $K=5$ 重复的采样误差，不代表模型可达性能上界。
- **Equal-allocation replay 的局限性**：仅衡量"是否观察到低成本成功"，不保证在新严格预算下可复现；极端压力（如 GPT-5.5 $\rho=0.2$ Code 仅 38 token）下 replay 分数趋近零。
- **领域异质性导致通用调度策略失效**：Code 因编译/测试反馈丰富而改善显著，Math/AR 因弱信号环境下浅层探测可能浪费预算。
- **未来方向**：论文建议通过训练内化资源理性调度策略，使模型能根据领域类型、当前进展质量和剩余预算动态决定继续/切换/验证/停止。

## 研究启发与可借鉴点
- **单题响应曲线 + 离线 Oracle 的设计范式**：可用于任何需要评估"能力兑现效率"的场景——先构建单任务性能响应面，再与多任务共享资源场景下的实际表现对比，量化分配损失。
- **模型自标定压力级别的预算设计**：以各模型自身无约束消耗为基准设定相对压力（$\rho=0.2, 0.8$），避免绝对预算导致的模型间不公平比较，此设计可直接迁移至其他资源受限评测。
- **轨迹级因果归因标注体系**：对 Oracle-selected miss 进行互斥主因标注（6 大类），以及 online adaptation 的四层标注（观察→策略更新→资源理性更新→仍失败），为后续分析提供细粒度诊断基础。
- **轻量级在线调度干预的可复现模板**：Strategy A/B 仅依赖运行时可见状态（已付出步骤数、剩余预算、候选稳定性），不依赖 oracle/标答/难度标签，可作为通用 agent 调度约束设计的参考。
- **团队可结合方向**：可探索将资源理性调度作为 RL 训练 reward 信号，或在模型推理时引入外部 scheduler（类似 Strategy A/B 但通过 fine-tuning 学习动态参数），尤其针对 Math/AR 等弱反馈领域改进。

## 关键术语表
- **Resource Rationality（资源理性）**：将认知视为在有限计算资源约束下最大化期望收益的最优分配过程（Lieder & Griffiths, 2020）。
- **Shared Budget（共享预算）**：多个任务竞争单一计算预算，跨任务存在显式机会成本，区别于传统的每任务独立预算设定。
- **Response-Curve Oracle（响应曲线 Oracle）**：基于单题多预算级别 empirical 成功率的离线最优分配求解器，形式为多选背包问题。
- **Gap Ratio**：$(Oracle - Contest) / Oracle$，衡量单题展示能力在共享预算场景下的兑现比例，越低越好。
- **Equal-Allocation Replay**：将竞赛预算均分为六等份后，统计各题是否存在已观测到的低成本成功尝试的回放诊断。
- **Oracle-Selected Miss**：被 Oracle 分配了正预算但未在 actual contest 中解决的题目，用于失败归因分析。
- **Resource-Rational Strategy Update**：轨迹中由新证据触发的、与资源分配/进度/期望价值/剩余预算相关的实质性策略调整。
- **Target-in-Suite Diagnostic**：控制实验，保持六题上下文但仅要求解决指定目标题，用于排除"上下文干扰"对缺口的贡献。

## 可复现要素
- **数据集**：问题池来自 Omni-MATH、MathNet、LiveCodeBench Pro、Reasoning Gym，共 1200 题；比赛构造方法明确（50 套/领域 × 6 题）。论文声明开源：Code Dataset（GitHub 链接在首页）。
- **代码/权重**：论文首页标注 "Code Dataset" 且 Appendices 提供完整 protocol、prompt templates、action-accounting rules；代码和评估脚本随 benchmark artifact 发布。
- **关键超参**：预算压力级别 $\rho \in \{0.2, 0.8\}$；响应曲线网格 $\Lambda = \{0.05, 0.1, 0.2, 0.4, 0.8\}$；每级别重复 $K=5$；50 套竞赛/领域；每题每模型 5 次独立运行。
- **模型接口配置**：Table 13 列出各模型的 temperature、top-p、reasoning configuration；Table 14 给出 sandbox 配置（内存 2048 MB、存储 10240 MB、超时 7200s）。
