---
title: "Skill-Issue-Are-Skills-Language-Invariant-in-LLMs"
source: https://arxiv.org/pdf/2608.25832v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:49:13"
field: "多语言大模型评估"
keywords: ["multilingual LLM", "cross-lingual skill inconsistency", "self-play evaluation", "agent gameplay", "language-conditioned reasoning", "multilingual benchmark"]
innovations: ["提出 multilingual self-play 范式正交量化 LLM 跨语言技能不一致性", "发现推理语言切换可部分恢复弱语言界面性能（最高 89.4%）", "构建 193 语言 65 游戏的多语言 TextArena 资源库并提供三级翻译验证管线"]
benchmarks: ["Multilingual TextArena", "Global-MMLU", "Belebele"]
---

# 论文速读：Skill-Issue-Are-Skills-Language-Invariant-in-LLMs

## 一句话总结
本文通过多语言自我对弈（multilingual self-play）框架，首次量化了同一 LLM 在不同语言接口下**技能表现的一致性差异**——即使知识库、规则与动作空间完全不变，模型在空间推理、策略执行、已知知识检索等环节仍会因语言而表现出显著的能力鸿沟。

## 研究问题与动机
1. **核心问题**：LLM 跨语言性能不均是已知现象，但现有研究多聚焦"知识可及性"（knowledge accessibility）的词汇/事实差异；本文追问：同一模型在互动决策场景中，其**推理、规划、决策技能**是否随语言接口而变化？
2. **现有方法不足**：静态多语言基准（如 Global-MMLU、Belebele）仅评估固定输入-输出映射，无法捕捉模型在** evolving interaction** 中表达等价策略的能力差异；交互类评测大多比较不同模型或提示方法，而非控制模型不变下的语言效应。
3. **隔离难度高**：语言效应与模型整体能力、训练数据分布、脚本类型等因素高度耦合，需要在游戏规则、状态空间、动作集严格固定的前提下，单独扰动"界面语言"才能识别技能层差异。
4. **应用动机**：若要构建真正公平的多语言智能体系统，必须确认模型在不同语言中能**同等调用**已有技能，否则"懂知识 ≠ 会做事"的语义鸿沟将在部署阶段暴露。

## 核心贡献（创新点）
1. **提出 multilingual self-play 评估范式**：同一模型的两个实例在同一游戏环境中以不同语言接口进行对抗，模型/对手/规则/动作空间完全固定，从而正交地隔离语言的技能效应。与已有工作相比，该设计将"语言效应"从知识可及性与整体能力中解耦，是首个在互动决策层面量化跨语言技能不一致性的方法。
2. **构建 Multilingual TextArena 资源库**：扩展 TextArena 至 193 种语言的 65 个游戏，建立三级（Tier A/B/C/E）自动+人工混合的翻译与验证管线；其中 Tier A 的 8 种语言由母语者逐句校对。规模与质量保证显著超越先前多语言交互基准。
3. **发现并刻画三类语言条件化技能差异**：（a）空间推理失败模式随语言变化——非拉丁脚本下模型更易漏检列/对角线威胁；（b）策略行为的风险偏好发生偏移——以 Kuhn Poker 为例，Qwen 以最小牌 bluff 的概率跨语言超过 2 倍；（c）已知最优策略的"存在但不可达"——Nim 的必胜策略在阿拉伯语/希伯来语界面下提及率和执行率骤降，但中文推理可部分恢复。
4. **揭示语言敏感性的多层根源**：通过"界面语言固定 + 推理语言切换"干预实验证明，约 60–89% 的性能差距可通过改用更强语言进行中间推理来恢复，说明语言效应同时作用于状态解释与推理两个阶段；同时，性能差异仅部分可由静态基准得分与网络文本量解释，脚本类型与模型特有关系构成剩余变量的关键来源。

## 方法详解
- **多语言自我对弈设置**：对每个游戏 $g$，构造语言对 $(\ell_0, \ell_1)$，Player 0 的观察指令为 $\ell_0$，Player 1 为 $\ell_1$，但棋盘符号、坐标、卡牌花色/点数、数值及动作语法（如 `[bet]`、`[cell]`）在所有语言中保持字面一致，确保技能差异的唯一来源是语言接口。
- **评价指标**：
  - **Role-pooled win–loss margin**（式 1）：$\Delta_{m,g}(A,B) = \frac{W_{m,g}(A,B) - L_{m,g}(A,B)}{N_{m,g}(A,B)}$，双向角色 assignment 合并后消除先手优势偏差，$\Delta(A,B) = -\Delta(B,A)$。
  - **Mean language margin**（式 2）：$\mu_{m,g}(A) = \frac{1}{|\mathcal{L}|-1}\sum_{B \neq A} \Delta_{m,g}(A,B)$，衡量语言 A 的平均对抗强度。
  - **Model-level mean**（式 3）：跨 6 个游戏 macro-average 得 $\bar{\mu}_m(A)$。
- **语言敏感性度量**：$max_\ell \mu_\ell - min_\lib \mu_\ell$，反映同一模型跨语言表现的极差。
- **恢复干预**：固定弱语言界面（如 de/de，$\mu=-0.22$），将中间推理语言切换为英语（de/en），测量恢复比例 $(\mu - \mu_{weak})/(\mu_{best} - \mu_{weak})$。
- **Prompt 设计**：默认 prompt 要求模型"reason in the language provided"；推理语言干预实验额外显式指定内部推理语言。
- **采样参数**：temperature=1.0、top_p=0.95、top_k=64，鼓励多样对弈轨迹。

## 实验与结果
- **模型**：Gemma-4-E4B-it、Qwen3-4B、Ministral3-3B-Instruct-2512（均≤4B）。
- **语言**：Tier A 共 8 种（en/ar/de/es/fr/he/ms/zh），经母语者逐条验证。
- **游戏**（6 个，覆盖多种技能）：TicTacToe（空间推理）、Nim（完美信息/算法策略）、Simple-Tak（空间连接）、Colonel Blotto（资源分配）、Kuhn Poker（不完全信息/ bluff）、Iterated Prisoner's Dilemma（重复社会互动）。
- **规模**：每模型-游戏组合 28,800 场自我对弈（$(\binom{8}{2}+8)\times 2\times 400$），总 518,400 场；GPU 时长约 6 H200 GPU-hours/主要实验跑。
- **主要发现**：
  - **跨语言技能差异显著**：英语在所有模型上平均最强，希伯来语最弱；Qwen3 语言层级最尖锐（Table 2 语言敏感性均值 0.54，Gemma 仅 0.28）。
  - **不同游戏敏感度不同**：Colonel Blotto 语言差距最大（三模型均≥1.07），Kuhn Poker 最稳定（均值 0.13）。
  - **空间推理失败模式**：Gemma 在 Arabic/Hebrew 界面下对角线/列损失占比达 45%/57%，而英语界面行列对角均衡（28.5%/34.8%/36.7%）。
  - **策略行为偏移**：Qwen 以最小牌 bluff 概率跨语言超过 2 倍（Table G.3）；Ministral 的 invalid action 率高达 9–21%。
  - **Nim 最优策略"存在但不可达"**：Qwen 在英语界面提及策略 150,005 次且执行率 80.8%，但在法语界面提及 159,675 次仅执行 24.6%；Arabic/Hebrew 提及极少且执行率低于 11%。
  - **强语言推理可部分恢复**：Gemma 在 TicTacToe 中以 de 界面 + en 推理恢复 89.4% 差距（Table 3）；SimpleTak 恢复 60.5%；Kuhn Poker 恢复有限（37.6–49.3%），说明语言效应作用于决策链不同阶段。
  - **外部因素解释力**：语言 margin 与 Global-MMLU 相关性 r=0.73–0.92，与 FineWeb-2 文本量 r≈0.79；但 Malay 异常高于数据量预测，Chinese 在 Qwen/Ministral 上远超英语数据比例，表明脚本迁移与模型特异性同样关键。

## 相关工作脉络
1. **跨语言知识不一致性研究**（Jiang et al., 2020; Qi et al., 2023; Ifergan et al., 2024; Goldman et al., 2025）：聚焦事实知识在不同语言间的 compartmentalization；本文扩展至技能层面，证明知识存在 ≠ 技能可调用。
2. **多语言静态基准**（Belebele, Global-MMLU, BenchMAX）：评估固定输入-输出准确率；本文用交互式对弈补充"动态技能表达"维度，填补静态基准无法捕捉的策略一致性缺口。
3. **TextArena / SmartPlay / GameBench / GTBench**（Guertler et al., 2025; Wu et al., 2024; Costarelli et al., 2024; Duan et al., 2024）：均在统一语言接口下比较模型或方法；本文首创"同模型多语言对弈"设计，将语言变量从基准对比中剥离。
4. **跨语言思维/自翻译方法**（Etxaniz et al., 2024; Zhu et al., 2024; Huang et al., 2023; Mondshine et al., 2025）：将非英语输入翻译为英语以提升性能；本文区分"界面语言"与"推理语言"，证明仅切换推理语言即可部分恢复，无需翻译环境观察。
5. **脚本与跨语言迁移**（Goldman et al., 2025; Ifergan et al., 2024; Malkin et al., 2022）：ECLeKTic 发现共享书写系统的语言间迁移更强；本文观察到 Malay（拉丁脚本）表现优于 Hebrew（虽数据量更少），印证脚本转移效应。

## 局限性与未来方向
- **闭源模型限制**：训练数据不可见，只能以 FineWeb-2 文本量为代理变量，无法精确溯源语言表现差异的成因。
- **模型规模局限**：仅评估 3B–4B 参数模型，3B 模型的 invalid action 率较高；大模型可能呈现不同的语言层级或更均匀表现，结论不可直接外推。
- **Apertus 等开放数据模型未能有效参与**：因产生过多 invalid moves 导致评分不可行，限制了可解释性分析。
- **Future direction**：（1）扩展至更大规模模型验证结论普适性；（2）探索推理语言切换作为轻量部署级修复策略的通用适用边界；（3）深入分析脚本类型、训练数据分布与模型架构对语言敏感性的交互效应。

## 研究启发与可借鉴点
1. **Multilingual self-play 设计可直接迁移**：适用于任何需要"技能而非知识"来判别的场景（工具使用、代码生成、对话协调），只需构造相同环境的多语言接口版本，即可正交地量化语言效应。
2. **界面语言/推理语言解耦实验**：Tab. 3 的恢复实验设计简洁有力——固定弱语言界面、切换推理语言，能快速定位语言敏感性的作用阶段（是状态理解问题还是推理问题），值得推广到更多 agent 任务。
3. **失效模式的细粒度分析框架**：将 defeat 按空间轴（行/列/对角）、卡牌条件（Bluff_J / Value_K / BadFold_K）、策略执行率分层统计，为跨语言不一致性提供可操作的诊断信号；可复用于其他多语言 agent 评测。
4. **三级翻译验证管线**：Tier A（母语者校对）→ Tier B（LLM 双轮验证）→ Tier C/E（NLLB+双模型 fidelity judge），为大规模多语言 benchmark 构建提供工程模板。
5. **可与本团队方向结合**：若团队关注多语言 agent 部署，可将此框架接入现有 RLHF/RLAIF 管线，在 reward model 阶段引入语言一致性正则项，或在 SFT 阶段加入跨语言 self-play 数据。

## 关键术语表
**Multilingual self-play**：同一模型在不同语言接口下与自身对抗的评估范式，控制模型/规则不变，单独检验语言对技能表达的影响。
**Role-pooled win–loss margin**：双向角色 assignment 合并后的归一化胜负差（[−1,1]），消除先手优势，量化语言对的相对强弱。
**Language sensitivity（语言敏感性）**：同一模型在某一游戏中最高与最低语言 margin 之差，反映跨语言表现的不一致性幅度。
**Reasoning language recovery（推理语言恢复）**：固定弱语言界面，将中间推理切换为强语言（如英语），测量由此恢复的性能比例。
**Knowledge-compartmentalization（知识区室化）**：多语言模型在不同语言中存储的知识相互隔离，一种语言触发的知识在另一种语言中可能不可检索。
**Nim-sum**：Nim 游戏的必胜策略核心——使各堆石子数的异或和为 0，本文用来检验模型是否在非英语界面仍能调用已知算法策略。
**Script transfer effect（脚本迁移效应）**：共享书写系统的语言之间（如马来语与英语同用拉丁字母）存在跨语言知识与技能的正向迁移。

## 可复现要素
- **数据集**：Multilingual TextArena（65 游戏，193 语言；实验用 6 游戏×8 语言 Tier A）；**开源**（MIT License，集成进 TextArena 仓库）。
- **代码**：Multilingual extension 已提交至 TextArena 开源项目（App. D 含 starter code）；pipeline 代码在独立 branch（论文未合并至主仓）。
- **权重**：Gemma-4-E4B-it、Qwen3-4B、Ministral3-3B-Instruct-2512 均为 open-weight，模型公开可复现。
- **关键超参**：temperature=1.0、top_p=0.95、top_k=64；每语言对每方向 400 场自对弈；vLLM + Ray 分布式 rollout。
- **翻译管线**：Claude Opus 4.8 / GPT-5.2（Tier A）；Llama-3.1-405B + Qwen2.5-72B（Tier B）；NLLB-200（Tier C/E）。
- **推理恢复实验**：固定界面语言，显式 prompt 指定推理语言，具体 prompt 见 App. A。
