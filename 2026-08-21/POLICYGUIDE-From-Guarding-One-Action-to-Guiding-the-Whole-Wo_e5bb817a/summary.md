---
title: "POLICYGUIDE-From-Guarding-One-Action-to-Guiding-the-Whole-Wo"
source: https://arxiv.org/pdf/2608.19861v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:10:14"
field: "LLM Agent 安全与策略合规"
keywords: ["LLM Agent", "Runtime Safeguard", "Policy Compliance", "Workflow Enforcement", "τ²-bench", "Action Guard", "Workflow Graph"]
innovations: ["将策略合规形式化为护栏+工作流联合问题并通过干预覆盖理论证明工作流级触发的必要性", "提出 POLICYGUIDE 外部主动验证器：将策略编译为工作流图、跨轮持久化状态并在用户对话边界返回定向修复指引", "在 τ²-bench 三个领域实现均值 PASS⁴ 20pp 提升且跨 agent 迁移，CRAFT 红队攻击成功率最低、有序轨迹合规率最高"]
benchmarks: ["τ²-bench (Airline/Retail/Telecom)", "CRAFT Red-Teaming", "Author-designed Telecom Ordered Trace Rubric"]
---

# 论文速读：POLICYGUIDE: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents

## 一句话总结
POLICYGUIDE 将领域策略编译为工作流图，并在每轮用户对话边界调用外部主动验证器追踪跨轮次状态，返回逐步修复指引，使通用 LLM Agent 在完成多步骤业务流程时始终遵守策略约束。在 τ²-bench 三个领域上，结合 GPT 5.4 将均值 PASS⁴ 从 0.42 提升至 0.62，其中工作流最密集的 Telecom 领域从 0.19 跃升至 0.61。

## 研究问题与动机
- **策略违规存在两类独立失败模式**：一是执行禁止性动作（如为不合格用户变更账户），二是遗漏或错序前置程序步骤（如身份核验、确认流程），后者即使最终动作本身合法，也无法被事后拦截撤销。
- **现有运行时护栏（runtime safeguard）多为动作局部检查**：仅在被拦截的 mutating tool call 处触发，无法覆盖诊断链、用户指令、分支选择等发生在"安全动作"之前的流程要求；Telecom 诊断流程甚至不含任何 agent 侧 mutating 动作，完全错过拦截点。
- **现有工作流/SOP 代理侧重流程忠实执行而非行为护栏**：其检查设计目标是流程完成度，而非防范通用 agent 违反策略的行为；且通常耦合于特定架构，难以跨 agent 复用。
- **源策略分析揭示流程需求的广泛性与结构性差异**：Airline 67.4% 为过程级要求（P）、Telecom 98.0% 且有序工作流要求达 54.0%，表明纯动作级守卫在 Telecom 这类"流程密集型"领域损失最大。

## 核心贡献（创新点）
1. **将策略合规形式化为"护栏 + 工作流执行"联合问题**，并用干预覆盖理论（Theorem 1）严格证明：只有当验证器的触发调度覆盖所有可达首次偏离时，才能完整保持程序合法性；动作触发调度仅在每次首次偏离本身触发检查时才满足该条件，而工作流级调度天然满足。
2. **提出 POLICYGUIDE 作为外部主动验证器**：离线将策略编译为冻结的工作流图（七类节点、schema 验证、子流程内联），在线在用户对话边界触发，维护持久化请求状态，从记录位置遍历至首个未满足节点并返回定向 remediation，将工作流合规检查与 agent 执行架构解耦。
3. **实现显著的跨领域 PASS⁴ 提升与跨 agent 迁移**：GPT 5.4 均值从 0.42→0.62（+20pp），Telecom 从 0.19→0.61（+42pp）；同一冻结图不经重写即迁移至 Claude Sonnet 4.6 和 Gemini 2.5 Pro 均获增益，证明运行时层与执行器的正交性。
4. **在对抗鲁棒性与程序轨迹合规两个补充评估上均最优**：CRAFT 红队攻击成功率最低（0.087 vs PolicyGuard 0.125、ReAct 0.200）；作者设计的 Telecom 有序轨迹审计中 process-valid rate 达 56.2%（ReAct 17.5%、PolicyGuard 13.1%）。

## 方法详解
- **离线策略编译管线**（六阶段，冻结后跨所有系统复用）：① 提取工具规格与 mutating 工具；② 推导请求类型、共享流程、有序子流程并做覆盖率审计；③ 人工审阅计划；④ 生成并 schema 校验子流程（一次修复重试 + 分支审阅）；⑤ 连接 intake 主干、分类器与子流程；⑥ 验证 schema 合规、工具清单、mutating 授权覆盖、图组合/边arity/可达性，审查策略到图映射并剪枝无用子流程。最终得到每领域一张扁平化工作流图。
- **节点类型与满足条件**：entry/exit（结构）、agent_action（非工具行为）、user_input（用户响应）、tool_call（只读工具）、tool_authorization（mutating 工具授权闸门）、decision（分支）、subflow（子流程调用，加载时内联）。每个节点标注 actor、action 及显式满足条件。
- **在线验证器接口**：`V_φ(π, T, G, H, S) = (d, R̂)`，其中 φ 为验证器 LLM；π/T/G 为缓存静态前缀（策略原文、工具规格、渲染后的图、判定规则、输出契约），H 为对话历史，S 为代码拥有的请求状态。
- **触发时机**：每轮用户输入后、agent 回复前触发一次；若 agent 尝试未经授权的 mutating tool call，则拦截并触发一次纠正性重调用；跳过的 tool-result 轮在下次触发时折叠进对话 delta。
- **Reconcile 与 Traversal（Algorithm 1）**：验证器将当前对话中的请求与 S 中持久化请求对齐——新请求在 entry 打开，续接请求保留记录位置，废弃/重复请求丢弃或合并；随后从记录位置沿图遍历，对每个节点判断其 satisfying condition：事实/资格以 TOOL_RESULT 为真，用户选择/同意从其消息中确立；停在首个不满足节点，其 required action 即为 remediation。单次生成可跨越多个已满足节点。
- **状态所有权**：由代码（非 LLM 对话记忆）持有，拒绝未知 node ID，按 mutating 工具清单过滤授权输出，重建当前可用工具集，持久化每请求位置与 memory。
- **干预机制**：评估配置为 advisory 模式——首个未经授权 mutating call 被一次性拦截后解除闸门以允许立即重试；其余流程动作通过 remediation 引导而非硬闸门。
- **理论框架（Appendix A）**：定义 $P_G = \text{Pref}(\mathcal{L}(G))$ 为合法前缀集合；可达首次偏离为 $(\tau, e)$ 满足 $\tau \in P_G, \tau e \notin P_G$；Theorem 1 证明验证器完整覆盖所有首次偏离是保持程序合法性的充要条件；Corollary 2 刻画实际评估触发调度（用户轮边界 + 一次纠正拦截）的覆盖条件。

## 实验与结果
- **数据集**：τ²-bench 三个领域——Airline（50 任务，24 PV/26 Mut）、Retail（114，10/104）、Telecom（114，43/71）；诊断变体与 FlowAgent 对比使用 held-out test split（Retail 40，Telecom 40）。Telecom test 中 PV 占比更高（52.5% vs base 37.7%）。固定用户模拟器为 GPT-4.1，工作流均由 GPT-5.4 离线编写并冻结。
- **基线**：ReAct（无守卫）、ToolGuard（静态代码，仅 Airline 可用）、PolicyGuard（GPT-5.4，动作级 checklist）、FlowAgent（PDL + API 控制）。
- **主结果（GPT-5.4，n=4，Table 1）**：POLICYGUIDE 在三个领域均取得最高 Overall PASS⁴——Airline 0.620、Retail 0.725、Telecom 0.675；Telecom 增益最大（+0.421 [0.316, 0.526]，p < 10⁻¹²）。PASSᵏ 曲线（Figure 4）显示随 k 增大优势持续保持。
- **消融（Table 2）**：POLICYGUIDE-RAW（用原始策略文本替代图）相对 POLICYGUIDE 在 Airline/Retail/Telecom 分别低 0.100/0.150/0.325；POLICYGUIDE-SELF（工作流放入 actor prompt、移除外部验证器）在 Telecom Mut 上仍为 0.053，证明外部追踪是关键。
- **跨 agent 迁移（Table 4，Airline 50 任务）**：Claude Sonnet 4.6 整体 0.780（+0.060 over PolicyGuard），Gemini 2.5 Pro 整体 0.680（+0.080）；Telecom 迁移未见但 Airline 图直接复用证明正交性。
- **CRAFT 对抗鲁棒性（Airline 20 任务，Figure 6）**：POLICYGUIDE 试验级 ASR = 0.087（PolicyGuard 0.125，ReAct 0.200），阻止 91.3% 测试攻击。
- **有序轨迹合规（Table 5，Telecom 作者设计 rubric）**：POLICYGUIDE process-valid rate = 56.2%（ReAct 17.5%，PolicyGuard 13.1%）。
- **成本（Table 14）**：平均每任务约 $0.34–0.56（Prompt 缓存 85.8%–88.1%，但 output tokens 占支出 67.5%–71%）；墙钟时间约为 ReAct 的 5.45×–5.78×（Table 15）。

## 相关工作脉络
- **PolicyGuard (Kang et al., 2026)**：最相近的动作级守卫，读取全对话对 mutating call 作 checklist 检查并返回 PASS/BLOCK + remediation，但仍是动作触发，不持久化工作流位置，无法覆盖前置偏离。POLICYGUIDE 将其扩展为跨轮图遍历。
- **ToolGuard / Solver-Aided (Winston et al., 2026)**：工具级静态守卫/约束求解，仅能观测单次调用，无法触及过程级要求。
- **FlowAgent (Shi et al., 2025b)**：最相近的工作流控制器，使用 PDL + API 依赖/重复调用守卫；但目标是"合规且灵活的工作流执行"，未以策略违规防护为目标评估，且需将策略写入 actor 内部而非外部 overlay。
- **SOP-Agent (Ye et al., 2025) / StateFlow (Wu et al., 2024) / FLAP (Roy et al., 2024)**：将 SOP/状态机编码为 agent 内部结构以驱动执行，关注流程忠实完成，缺乏对通用 agent 行为的护栏角色。
- **CRMArena-Pro (Huang et al., 2025) / Near-Miss (Rabinovich et al., 2026) / Call-NMR (Kang et al., 2026)**：分别评估保密合规、事后审计、条件化 near-miss；POLICYGUIDE 侧重事前主动引导 + 事中轨迹合规。
- **CRAFT (Nakash et al., 2025)**：提供对抗性用户注入虚假资格前提，用于评估护栏鲁棒性；本文用作红队测试基准。

## 局限性与未来方向
- **评估范围有限**：仅三个 English τ²-bench 领域，固定用户模拟器，每格 n=4；Retail 仅 10 个 PV 任务，总体增益统计不显著；未见其他政策体制、语言或真实用户。
- **CRAFT 跨域覆盖不足**：仅报告 Airline 20 任务 clean split，Retail/Telecom 官方 set 未对齐，无法断言跨域或自适应攻击鲁棒性。
- **工作流作者模型单一**：全部由 GPT-5.4 编写并冻结，未测试跨作者模型种子的工作流泛化；虽有手工验证 faithfulness，但不构成自动化保证。
- **触发调度非全覆盖**：仅在用户轮边界与一次纠正拦截后触发，Corollary 2 刻画其覆盖条件；更广泛的干预需更多调用，会增加成本。
- **概率性执行**：验证器提示 LLM 输出，合规为经验性而非形式化保证；fail-open 设计在高危场景需额外确定性监控器。
- **成本与延迟**：平均 $0.34–0.56/任务、墙钟 ~5.5× ReAct；输出 token 占比高，缩短 remediation 长度是优化方向。
- **未来方向**：（i）降低 guide 输出 token 以提升性价比；（ii）将触发调度扩展至更多 agent action 边界；（iii）接入确定性形式化检查器作为 hard guarantee 子集；（iv）自动化工作流编译减少人工审阅；（v）扩展到 live user 与跨语言场景。

## 研究启发与可借鉴点
- **"外部 overlay + 代码所有权状态"架构**：将验证器与 actor 分离、用代码而非 LLM 对话记忆持有关于请求状态与位置，这一模式可移植到其他需要跨轮状态追踪的 agent 护栏场景（如多轮审批、双人复核）。
- **策略 → 图编译管线的五类 schema 约束**：节点类型定义、authorize→verify 成对出现、gates 必须 outcome-framed（以实际条件成立为真而非"已检查"为真）、工具节点仅命名 agent 清单工具、shared subflow 抽取——这套规范可直接迁移至其他需要形式化流程执行的领域。
- **Passᵏ 可靠性曲线作为稳健指标**：除了传统均值，用 $\text{Pass}^k = \binom{c}{k}/\binom{n}{k}$ 刻画"至少 k 次成功"的可靠性，比单次通过更能反映生产可用性；建议在本团队评测中也采用。
- **理论化的"干预覆盖"视角**：Theorem 1 / Corollary 2 将验证器触发调度与可达首次偏离集合的联系形式化，可作为后续工作的设计准则——任何新触发策略都应回代验证其 $C_S(G) = D_G$。
- **作者自定义 rubric 弥补 benchmark 缺陷**：τ²-bench 仅评估终态与断言，不校验中间顺序；本文针对 Telecom 手工构建 Step-TCR / Trace-TCR / process-valid rate，提示我们在面对标准评测盲区时，可构造任务特化的过程合规审计器。

## 关键术语表
- **PASS⁴（Pass⁴）**：四折均值可靠性指标；对含 c 次成功的任务计算 $\binom{c}{4}/\binom{n}{4}$ 并跨任务平均，衡量"四次尝试全部成功"的概率。
- **Workflow Graph（工作流图）**：将策略编码为有向图，节点表示 actor/action，边表示满足条件下的转移，是策略合规交互的结构化表示。
- **Mutating Tool Call（变更工具调用）**：会改写后端数据库状态的 agent 工具调用，通常作为策略合规的关键拦截点。
- **Policy Adherence（策略遵从）**：agent 行为同时满足"不执行禁止动作"与"完成必要程序步骤"的双重约束。
- **Remediation（修复指引）**：验证器返回的下一节点所需 action 描述，以自然语言注入 agent prompt 引导其走回合规路径。
- **CRAFT Red-Teaming（CRAFT 红队攻击）**：由 adversarial user simulator 注入虚假资格前提诱导 forbidden mutation 的攻击基准，用于评估护栏鲁棒性。
- **Process-Level (P) vs Workflow-Level (W) Requirement**：P 要求读取对话/只读工具结果即可判定；W 额外要求按特定顺序执行步骤，错序本身即为违规。
- **Call-NMR（Near-Miss Mutation Rate）**：在 outcome-passing 轨迹中，成功执行的 mutating call 缺少某前置只读检查的比例，衡量程序前提的完整性。

## 可复现要素
- **数据集**：τ²-bench（Airline/Retail/Telecom 三个领域 base split + held-out test split），由 Barres et al. (2025) 发布，需申请访问。
- **代码/权重开源**：论文声明 "We will release the prompts, workflow schemas, and analysis artifacts needed for reproducibility"（伦理声明末尾）；截至论文发表时未给出正式 repo URL。Agent 侧（GPT-5.4 / Claude Sonnet 4.6 / Gemini 2.5 Pro）均为闭源商业模型。
- **关键超参**：验证器 temperature = 0；每任务 trials n = 4；Prompt 前缀缓存（静态部分 byte-stable）；verifier 与 agent 同模型族配对（GPT-5.4 域内统一）；工作流由 GPT-5.4 离线生成并冻结。
- **硬件/环境**：论文未提及具体 GPU 配置，验证器与服务端 agent 均为 API 调用。
