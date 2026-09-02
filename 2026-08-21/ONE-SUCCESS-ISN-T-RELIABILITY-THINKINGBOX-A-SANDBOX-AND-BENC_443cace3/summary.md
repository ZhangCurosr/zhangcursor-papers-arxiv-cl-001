---
title: "ONE-SUCCESS-ISN-T-RELIABILITY-THINKINGBOX-A-SANDBOX-AND-BENC"
source: https://arxiv.org/pdf/2608.19741v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:08:57"
field: "智能体评测与可靠执行"
keywords: ["agent benchmark", "stateful workflow", "executable evaluation", "tool use reliability", "side-effect checking", "business agent", "MCP"]
innovations: ["提出 Thinkingbox 沙盒统一承载业务工具-智能体-用户交互的可执行终态判定", "同时报告 pass@k 与 pasŝk 分离发现能力与可靠执行能力", "以侧效提取+合取判定代替轨迹匹配，揭示 80%+ 表观成功实为失败"]
benchmarks: ["THINKINGBOX-BENCH (507 tasks)", "τ-bench", "AppWorld", "MCP-Atlas", "SWE-bench", "BFCL"]
---

# 论文速读：ONE SUCCESS ISN'T RELIABILITY — THINKINGBOX, A SANDBOX AND BENCHMARK FOR AGENTS IN STATEFUL BUSINESS WORKFLOWS

## 一句话总结
论文提出了 **THINKINGBOX**——一个面向有状态业务工作流中工具智能体的可执行交互沙盒，以及基于该沙盒构建的 **THINKINGBOX-BENCH**（507 项多轮、多策略、含后端副作用检查的业务任务基准），揭示了"偶然成功"与"可靠完成"之间的巨大差距（pass@20=91.12% vs pasŝ20=25.25%）。

## 研究问题与动机
- **现有评测聚焦"看起来做对了"**：SWE-bench、BFCL、WebArena 等基准主要校验最终代码、工具调用语法或网页操作序列，无法捕捉业务场景中真正的完成质量。
- **业务工作流的三重复杂性**：多轮对话（用户信息不完整需追问）、有状态后端（记录必须转入正确终态）、副作用约束（不能误改无关记录或产生非法附加效果）。
- **"单次成功≠可靠"**：即使模型能在一两次尝试中凑出正确轨迹，其跨次重复成功率极低，评测指标缺乏可靠性维度。
- **端到端可执行判定的缺失**：现有方法缺少从初始状态到终态的确定性状态对比、策略合规性核查与对话完整性验证的统一框架。

## 核心贡献（创新点）
1. **THINKINGBOX 沙盒**：提供隔离 MCP 工具会话 + 可执行侧效提取 + 任务特定判定器的统一评估循环，可复用做评测、失败分析与训练奖励。
2. **THINKINGBOX-BENCH（507 项任务）**：覆盖零售、酒店、车险、新银行 IT、咨询 IT/HR 五个业务域，每个任务均以终态数据库对比为核心判定信号，30 项额外附加自然语言 rubric 检查。
3. **发现-可靠性分离指标**：同时报告 pass@k（k 次内至少一次成功）与 pasŝk（k 次全部成功），量化智能体的发现能力与可靠执行能力之差。
4. **可复现的失败诊断分类**：基于完整轨迹提取四类主导失败签名（工具使用失败、未执行状态变更、用户解析不完整、错误状态更新），提供细粒度可追溯分析。
5. **端到端强判定 vs 弱代理判定的显著差距**：80.88% 的失败轨迹表面上"干净终止+调用了写工具"，若仅依赖表面信号会大量误判为成功；可执行状态比对能揭露字段级错误。

## 方法详解
- **任务形式化**：每个任务表述为 $x = (b_0, g, \mathcal{T}, \mathcal{U}, \mathcal{C})$，其中 $b_0$ 是初始后端状态、$g$ 是用户目标、$\mathcal{T}$ 是域工具集合、$\mathcal{U}$ 是模拟用户策略、$\mathcal{C}=\{c_i\}$ 是一组隐藏的可执行检查。
- **诱导 POMDP**：$M_x=(S_x, A_x, \mathcal{O}_x, P_x, Z_x, R_x, \mu_x, H)$，隐状态 $s_t=(b_t, z_t, \ell_t, q_t)$ 包含后端状态、模拟用户私有状态、事件日志和回合状态；智能体仅基于可观测历史 $h_t$ 决策。
- **隔离与重置**：每次尝试将任务重置到 $b_0$，创建独立 MCP 会话；连续两次尝试不共享任何数据库行、缓存或副作用，确保 pass@k / pasŝk 可比较。
- **侧效提取与判定**：终端后提取 $e = \Delta_x(s_0, s_T, \rho)$（任务相关的状态变更与动作记录），最终判决 $V(x,\rho)=\prod_i c_i(s_T, e, \rho) \in \{0,1\}$，采用合取评分（所有条件须满足）。
- **模拟用户 $\mathcal{U}$**：固定由 GPT-5.4-mini 驱动，温度 0.3；严格遵循 USER CONTEXT 中的事实与行为规则，只回答被问到的信息，绝不主动提供未披露信息或编造实体；最多 10 轮跟进，保证对话闭合与可复现。
- **Rubric 任务**：30 项额外要求最终回复满足某种属性（如不泄露保密酒店信息、明确传达政策受限结果），由同一 GPT-5.4-mini 作判定器。
- **训练接口**：判定值 $V$ 可化为稀疏终端奖励；单条检查也可给出分量奖励 $r_i(\rho)=c_i(\cdot)$，同一沙盒可用于 RL 训练。

## 实验与结果
- **数据集**：507 项任务，分属 5 域（Retail 98 / Booking 104 / Insurance 100 / Neobank 104 / Consulting 101）；后端系统 47 个、数据库表 103 张共 638 行、工具 100 个读写；每项任务平均 4–9 步（Booking 最高 8.8 步）。
- **评测协议**：12 个模型（6 个闭源 + 6 个开源）在相同沙盒、相同系统提示下测试；每项任务 $N=20$ 次独立采样，按微平均报告 pass@1，并配 95% 任务聚类 Bootstrap CI。
- **最强闭源模型**：GPT-5.4 以 65.36% pass@1 居首；但 pasŝ20 仅 25.25%，而 pass@20 高达 91.12%，揭示"偶发可达"与"稳定可达"的巨大鸿沟。
- **开源最强**：DeepSeek-V4-Pro 43.26% pass@1；GLM-5.1 33.19%；Kimi-K2.6 37.66%；Qwen3.6-27B 32.94%；小型模型差距显著（Qwen3.5-9B 5.84%、Mistral-Large-3 4.66%）。
- **域难度分层**：Retail 最易（平均约 52% pass@1），Auto Insurance 最难（平均约 23%），Booking/Neobank/Consulting 居中（~27–35%）；GPT-5.4 与 Claude Sonnet 4.6 是唯一全域 >50% 的模型。
- **失败分类（跨模型平均）**：Tool Usage 77.5%、Wrong State Update 12.1%、Incomplete User Resolution 7.9%、No State-Changing Action 2.5%；GPT-5.4 在 Tool Usage 上占比最高（89.6%）。
- **弱代理判定消融**：84.86% 的失败轨迹满足"干净终止"、80.88% 额外满足"调用写工具"、67.24% 再满足"末工具响应无显式错误"；但真实可执行状态比对仍能检出 98.95% 的差异，证明弱信号远不够。

## 相关工作脉络
- **可执行评测先驱**：SWE-bench（代码修复）、AppWorld（应用 API+终态+副作用检测）、τ-bench（策略对话+终端 DB 状态+重复可靠性）——Thinkingbox 把终态判定与重复可靠性统一到一个业务沙盒里。
- **工具调用评测**：BFCL、API-Bank、ToolBench 等聚焦 API 选择与参数生成，不涉及持久化状态与副作用；Thinkingbox 强调"调对了工具≠做成了任务"。
- **交互式环境**：WebArena/OSWorld/WebShop/ALFWorld/AgentGym 多为开放世界导航或桌面控制；Thinkingbox 更贴近企业支持场景（多工具、多系统、策略条件、不可逆副作用）。
- **MCP 生态**：MCP-Bench、MCPMark、MCP-Atlas 测真实服务器的工具发现与 CRUD；Thinkingbox 在此基础上叠加用户对话模拟与任务级终态检查。
- **专业工作流评测**：WorkArena++、CRM Arena/Pro、AgentDojo 分别侧重企业软件、CRM 对话、安全注入；Thinkingbox 填补了"有状态业务+可执行终态+可靠重复"的交叉空白。
- **定位差异**：Thinkingbox 不追求最大任务数或最复杂单步推理，而是用人工审核的 507 项可执行任务，提供统一可复现的 sandbox，把评测、失败诊断、训练奖励合并在同一回路。

## 局限性与未来方向
- **判定覆盖不对称**：477/507 项仅看终态与副作用，不看回复内容；正确完成却表达糟糕的轨迹仍可被判成功。
- **任务来源与代表性**：源于非公开企业案例库，仅合成重构为基准；不代表任一公司真实工作流分布，也未声称统计代表。
- **单一模拟用户**：使用固定 GPT-5.4-mini，严格合作、不记错、不改目标、不拒绝配合；未测试应对难缠用户、目标漂移、隐瞒信息等现实难度。
- **固定黄金终态**：每个任务只有一个目标终态，排除了"多种合理路径均正确"的任务设计，可能低估某些模型的等效解探索能力。
- **同家族偏差**：模拟用户与响应判定器均为 GPT-5.4-mini，无法完全排除与最强 Agent（GPT-5.4）的交互风格亲和效应。
- **未来方向**：扩展 LLM-rubric 到所有任务、引入多样化/对抗型用户模拟器、增加可多解任务、扩大域覆盖、探索基于沙盒的 RL 训练。

## 研究启发与可借鉴点
- **终态比对优先于轨迹匹配**：用哈希/字段级数据库对比取代对工具调用序列的硬匹配，允许不同合理路径达成相同终态，显著提升评测鲁棒性。
- **副作用与"额外效果"检测**：同时拒收"遗漏必要变更"和"产生无关变更"两条反例，对业务场景尤其关键；可在多系统联动的评测中复用。
- **pass@k 与 pasŝk 并用**：将"一次试对"与"反复都做对"拆开报告，避免高 pass@k 掩盖低可靠性；这是评估生产可用性的必要组合。
- **失败签名分类+轨迹可视化**：基于确定规则的主失败类型标注，让评测不只是数字而是可诊断的归因，值得作为团队评测报告的标配。
- **沙盒化统一接口**：同一套 sandbox 承载评测/失败分析/训练奖励，便于后续直接把它作为 RL 环境的 reward oracle；开源实现可迁移到本团队内部工具评测。

## 关键术语表
- **THINKINGBOX**：面向有状态业务工作流的工具-智能体-用户交互沙盒，提供隔离会话、轨迹记录与可执行判定。
- **THINKINGBOX-BENCH**：构建于 Thinkingbox 之上的 507 项可执行任务基准，覆盖五个企业业务域。
- **pass@k**：k 次独立尝试中至少一次成功的概率估计，衡量"发现可行轨迹"的能力。
- **pasŝk**：k 次独立尝试全部成功的概率估计，衡量"可靠执行"的能力。
- **Side-effect extraction**：在交互结束后从终态与轨迹中提取与任务相关的状态变更集合 $e$。
- **Executable verdict**：基于 $V(x,\rho)=\prod_i c_i(s_T,e,\rho)$ 的合取判定，所有检查必须通过才视为完成。
- **Simulated user $\mathcal{U}$**：固定 GPT-5.4-mini 驱动的闭合世界对话代理，仅基于 USER CONTEXT 和可见历史作答。
- **Tool Usage failure**：调用工具出现错误/前置条件失败后未能恢复，是模型最主要失败模式（平均 77.5%）。

## 可复现要素
- **数据集**：THINKINGBOX-BENCH 已公开，含 507 个任务标识符与 JSONL 轨迹文件（见仓库）。
- **代码/权重**：框架与基准均开源，见 https://github.com/microsoft/thinkingbox。
- **推理参数**：Agent temperature=1.0、max_completion_tokens=4096、is_reasoning=True、reasoning_effort=medium、top-p=1.0、timeout=600s；模拟用户与判定器 temperature=0.3、4096 token、无推理；Qwen 系列本地 vLLM port=8000、data_parallel=8、tensor_parallel=1、max_model_length=65536。
- **协议要点**：每项任务 N=20 次独立采样；MCP 会话超 300s、请求超时 600s、HTTP 5xx 最多重试 5 次；对话上限 10 轮用户跟进。
