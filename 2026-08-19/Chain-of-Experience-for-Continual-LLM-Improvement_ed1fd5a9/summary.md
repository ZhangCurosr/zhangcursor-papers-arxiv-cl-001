---
title: "Chain-of-Experience-for-Continual-LLM-Improvement"
source: https://arxiv.org/pdf/2608.18027v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:29:12"
field: "测试时学习与经验累积"
keywords: ["Chain-of-Experience", "Test-time Learning", "Feedback-driven Reasoning", "LLM Self-improvement", "Experience Accumulation", "Test-time Scaling"]
innovations: ["提出 Chain-of-Experience (CoE) 统一框架，将单问题求解扩展为经验轨迹迭代累积过程", "系统对比无反馈/模型自反馈/执行反馈/正确性反馈四类信号的效果、效率与互补性", "揭示基座模型能力与测试时改进潜力的正相关关系并给出归因分析与鲁棒性缓解策略"]
benchmarks: ["AIME 2025", "OmniMath", "LiveCodeBench (V6)", "LiveBench (Code)", "GPQA Diamond", "EvaLearn"]
---

# 论文速读：Chain-of-Experience-for-Continual-LLM-Improvement

## 一句话总结
论文提出 **Chain-of-Experience (CoE)** 框架，使大型语言模型（LLM）能在测试时通过与环境或自身反馈进行多轮迭代交互、累积经验轨迹，实现持续推理改进。在数学、编程、知识三类任务的六项基准上，仅使用自反馈（self‑feedback）的平均准确率即可从 62.9% 提升至 71.0%（+7–9% 绝对提升），并以 19% 更低的 API 成本获得整体 5.6% 的综合增益。

## 研究问题与动机
1. **传统评估的盲点**：现有 LLM 评测几乎完全基于零样本/少样本的一次性生成，忽略了模型在**推理阶段通过交互与反馈持续学习**的潜力。
2. **既有测试时策略的经验利用过于短暂**：多数测试时扩展方法（如多数投票、Tree‑of‑Thought、Self‑Refine/Reflexion 等）将经验临时固化于单条答案后即丢弃，无法形成跨步累积的“经验链”。
3. **缺乏统一、系统的反馈谱系对照**：不同反馈类型（无反馈、模型自反馈、代码执行反馈、正确性/裁判反馈）对迭代改进的贡献边界不清，效率与鲁棒性亦未在同一框架下量化比较。
4. **模型容量与“可从经验中学习”的能力关系不明**：尚未有工作系统检验更强基座模型的测试时迭代改进潜力是否随其初始能力单调提升。

## 核心贡献（创新点）
1. **提出 Chain‑of‑Experience (CoE) 统一框架**：将单次生成过程扩展为 $a_t \sim P(a_t \mid Q, e_0,\dots,e_{t-1})$ 的经验累积迭代过程，形式化定义并实证四种反馈类型。与以往“单轮自我修正”或“跨任务检索经验”的本质区别在于，CoE **将同一问题的完整求解轨迹视为可复用经验**，并在单问题内持续演化。
2. **在八款前沿推理模型、六项基准上的大规模对照实验**：首次系统量化 no‑feedback / self‑feedback / executor‑feedback / correctness‑feedback 四类信号的效果、效率与组合收益。相比 Dynamic CheatSheet（DC）与 Agentic Context Engineering（ACE）等经验蒸馏/检索基线，CoE 仅凭自反馈即以更高效率取得更优表现，且**组合互补反馈可进一步增益**。
3. **揭示“基座能力↔改进潜力”的正相关规律**：在五类任务上计算归一化改进率 $\Delta_\mathcal{M}=(S_{\max}-S_{\text{base}})/(1-S_{\text{base}})$ 与零样本精度，得到平均 Pearson 相关系数 +0.50（编程任务高达 0.83–0.97），表明**更强模型的测试时自我演化能力系统性更强**。
4. **提出并验证 Selective Majority Voting (SelMV‑n)**：在存在欺骗性（spurious）反馈的极端设定下，仅取前 $n$ 轮有效尝试进行多数投票即可恢复/超越单次自反馈表现（如 GPQA‑Diamond 上当错误反馈下的 SelMV 较纯模型反馈提升 0.9%），证明 CoE 在对抗噪声时的鲁棒性。
5. **细粒度归因分析**：基于 6,630 例错→对轨迹，利用 GPT‑5 自动归因并与人工标注达成 Cohen's $\kappa=0.768$；识别出 47.7% 的改进由反馈直接驱动、30.0% 来自规格/格式召回（编程任务尤为突出），且**自生成反馈的“反馈相关性改进”比例（58.7%）显著高于其他来源**，同时证明绝大多数性能增益集中在前 20 轮以内。

## 方法详解
- **迭代生成形式化**：给定问题 $Q$，第 $t$ 步回答 $a_t$ 由完整历史经验 $e_i=(a_i,f_i)$ 条件采样得到：
  $$a_t \sim P\bigl(a_t \mid Q,\; e_0, e_1, \dots, e_{t-1}\bigr)$$
  经验在上下文内持续累积，构成“经验链”。
- **四类反馈及其定义**：
  - **No feedback**：$f_i=\emptyset$，$e_i=(a_i,\emptyset)$，仅靠既往回答的反思推进。
  - **Execution feedback**：在可执行环境中运行 $a_i$，以测试用例通过/失败、报错、运行时日志作为 $f_i$（适用于编程任务）。
  - **Model feedback**：由辅助 LLM $\mathcal{M}_\text{fb}$ 作为裁判，输出文本批判、偏好分数或结构化评价。
  - **Correctness feedback**：由领域验证器给出二元信号 $f_i=\mathbf{1}\{a_i \text{ is correct}\}\in\{0,1\}$，作为性能上界参考。
- **双反馈组合**：在同一迭代中将模型反馈与正确性/执行器信号合并，使“语义层面的解释性纠正”与“硬信号层面的对/错判定”相互补充。
- **经验选择实验**：在 CoE 内部接入 Dynamic CheatSheet（DC）与 SimpleMem 做压缩式检索/摘要，发现**激进压缩会丢弃中间关键推理步骤**，反而劣于保留全量轨迹的纯迭代。
- **选择性多数投票（SelMV‑n）**：当反馈存在系统性偏差时，仅聚合前 $n$ 次“形式上合法”的尝试进行多数投票，以滤除后期被污染轨迹。
- **训练/成本效率分析**：统计总 token 消耗与 API 费用，证明 CoE 通过把计算重新分配到“反馈迭代”而非单纯拉长单轮生成长度，从而实现**更高 Accuracy-per-Token**。

## 实验与结果
- **数据集/基准**（各 2 个 × 3 类任务）：
  - 数学：AIME 2025（30 题）、OmniMath（抽样 200 题）
  - 编程：LiveCodeBench (V6)（175 题）、LiveBench (Code)（128 题）
  - 知识：GPQA Diamond（198 题）、EvaLearn（648 题）
- **评测模型**（8 款）：GPT‑5、GPT‑5‑mini、o4‑mini、o3、o3‑mini、Gemini‑2.5 Pro、Claude‑4.5 Sonnet、以及 OpenAI/Anthropic 的不同 reasoning effort 变体。
- **基线**：无反馈（w/o feedback）、low/high reasoning effort、few‑shot ICL、Dynamic CheatSheet（DC）、Agentic Context Engineering（ACE）。
- **主要结果（摘要）**：
  - **平均性能**（表 3）：self‑feedback 均分 71.0%；correctness/executor feedback 达 79.3%。ICL（62.1%）、ACE（64.0%）、DC（62.7%）均低于无反馈基线（66.8%）。
  - **编程任务**：executor feedback 在 LiveBench (Code) 与 LiveCodeBench (V6) 上平均从 66.4% 提至 75.0%（+8.6%）；自反馈亦达 +7.0%。
  - **数学/知识任务**：correctness feedback 作为上界最高；自反馈仍稳健提升（如 AIME 2025：75.1% > 67.1% > 62.5%）。
  - **效率**：self‑feedback 以 13.4% 更少 API 调用（如 \$70.7 vs \$81.6）获取大部分 executor 增益；整体**API 成本降低约 19%**；每 token 准确率显著优于 DC 等多轮方法。
  - **最佳单点**：Claude‑4.5 Sonnet 在 AIME 2025 上使用 correctness/executor 反馈达 89.05%（Table 3），为所有方法中最强结果。
  - **鲁棒性**：全“错误”反馈平均下降 7.6%（AIME 2025）/ 2.6%（GPQA），而 GPT‑5‑mini 仅降 2.5%/0.6%；引入 SelMV 后部分场景可反超。
  - **迭代收敛**：将迭代数从 20 延长至 50，AIME 2025 上前期贡献 16.7% 增益而后期仅 2.2%，表明**绝大多数改进在前 20 轮已完成**。
- **结论**：CoE 以更低代价、更少 token 实现更优且更稳定的跨域推理改进；反馈类型具互补性；经验压缩策略需审慎。

## 相关工作脉络
1. **Test‑time 无训练推理增强**：与 CoT、ToT、Self‑Consistency、MCTS* 等一脉相承，但上述方法依赖并行采样+事后筛选或单次搜索，不保留跨步经验；CoE 在单轮内通过迭代反馈实现**持续性自我演化**。
2. **从经验学习（Learning from Experience）**：Dynamic CheatSheet、ACE 等通过跨任务检索/蒸馏策略到外部记忆；本文指出其在相同任务内的**激进压缩会损失中间推理细节**，而保留全轨迹的 CoE 更优。
3. **自我修正/自调试**：Self‑Refine、Self‑Debug、Reflexion、Iteration‑of‑Thought、ReVeal 等。这些方法多为**单问题内的浅层修正**或依赖单一执行信号；CoE 将其推广为**多类型反馈的通用迭代循环**，并系统刻画不同反馈的贡献差异。
4. **训练时在线经验利用**：RLHF/DPO、online post‑training（如 Dapo）通过策略梯度把经验固化为参数；CoE 完全**不更新参数**，隔离出纯粹“测试时上下文内复用经验”的机制。
5. **Verifier‑based 重排序**：Lightman 等人的 step‑level、Zheng/Cai 等人的 output‑level 验证属于**后处理**；CoE 把验证/裁判信号**直接嵌入生成循环**，驱动下一轮条件采样。
6. **Agent 记忆与经验共享**：ReasoningBank、Agent KB、Experience Synthesis 等侧重多智能体间经验流转或离线经验重用；本文聚焦**单智能体在单任务上的在线经验累积与反馈驱动的闭环**。

## 局限性与未来方向
- **评测域有限**：主要集中于受控的数学、编程与知识问答；**长视野、高交互的真实场景**（如持续多步骤工具调用、实时 GUI/网页操控）尚未充分覆盖。
- **无参数更新**：改进完全源于上下文内的经验复用，**并非真正的在线学习**；未来需探索在保持上下文效率的同时将关键经验内化到权重中。
- **部分任务依赖强监督信号**：correctness/executor feedback 在真实世界中获取成本较高；模型自反馈虽廉价，但在知识类分布外任务（如 BrowseComp‑Plus）上可能出现性能下降。
- **反馈源质量敏感**：外部模型裁判需达到一定能力阈值才能媲美正确性信号；对较弱模型的自反馈效果有限。
- **经验压缩的权衡机制未彻底解决**：本文证明激进摘要有害，但尚未给出**何种条件下可安全降维**的理论/工程判据。

## 研究启发与可借鉴点
1. **测试时经验链条可作为低成本增效手段**：在不改参数的前提下，通过多轮“生成→反馈→再思考”即可显著提升推理精度与单位 token 收益，适合部署在 API 计费敏感的线上系统。
2. **反馈类型需按任务匹配**：编程类任务强烈受益于执行器信号；数学/知识类可引入正确性验证作为上界；自反馈则提供普适的廉价替代。可借鉴“多源互补+阶段性切换”的设计思路。
3. **早期迭代收益主导**：80% 以上增益在前 20 轮实现，提示工程与调度策略应优先保证**高质量前几轮的反馈清晰度**，而非盲目堆叠轮数。
4. **保留中间推理比压缩摘要更重要**：将 DC/SimpleMem 等记忆压缩方法直接接入 CoE 反而会劣化，启示后续工作应设计**保留关键中间步骤的细粒度经验选择机制**，而非粗暴 RAG 式检索。
5. **鲁棒性可通过选择性聚合缓解**：SelMV‑n 证明在遭遇系统性噪声反馈时，剔除后期污染轨迹并进行多数投票即可恢复表现；该机制可直接迁移到任何依赖多轮裁判的测试时系统中。

## 关键术语表
- **Chain‑of‑Experience (CoE)**：将 LLM 对单一问题的求解建模为累积历史经验 $(a_i,f_i)$ 的迭代生成过程，使模型在测试时持续自我演化。
- **Test‑time Scaling / Test‑time 增强**：不在训练阶段更新权重，而是在推理时通过更多计算（多轮采样、搜索、反馈迭代）换取更高正确率。
- **Self‑feedback / 模型自反馈**：由同一或另一 LLM 对当前回答进行文本级评判与纠错建议，作为下一轮生成的条件。
- **Executor Feedback / 执行反馈**：将程序提交至解释器/单元测试环境运行，返回通过/失败、日志或错误栈等信号。
- **Correctness Feedback / 正确性反馈**：基于 ground‑truth 或形式化验证器给出的二元对/错信号，作为理想化的强监督上界。
- **Dynamic CheatSheet (DC)**：基于过往解题经验蒸馏出可复用策略的测试时记忆方法；本文在其与 CoE 结合时证明其压缩会损失细节。
- **Selective Majority Voting (SelMV‑n)**：在含噪声/欺骗性反馈的场景中，仅对前 $n$ 次有效尝试进行多数投票以提升鲁棒性。
- **Improvement Attribution（改进归因）**：将模型从错到对的轨迹变化分解为反馈驱动、自我反思、规格召回、随机漂移四类，量化不同学习模式占比。

## 可复现要素
- **数据集**：AIME 2025、OmniMath、LiveCodeBench (V6)、LiveBench (Code)、GPQA Diamond、EvaLearn、BrowseComp‑Plus（均有公开来源）。
- **代码/权重开源**：论文未提及代码与实验权重的开源声明。
- **关键超参**：temperature 设为 OpenAI 默认 1.0、其余模型 0.2；推理强度通过 `reasoning_effort`（low/high）或 thinking budget（Claude 设为 10,000）控制；最大迭代轮数 20（扩展实验至 50）；检索 k 取值 1/5/8/12/15/20；重复实验 3 次取均值±标准差。

---
