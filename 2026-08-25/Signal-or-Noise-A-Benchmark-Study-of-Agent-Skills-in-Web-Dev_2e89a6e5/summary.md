---
title: "Signal-or-Noise-A-Benchmark-Study-of-Agent-Skills-in-Web-Dev"
source: https://arxiv.org/pdf/2608.23067v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:00:58"
field: "Agent 评测与Skill效用分析"
keywords: ["Agent Skills", "WebDev benchmark", "prompt injection", "length-matched control", "slice ablation", "skill routing"]
innovations: ["提出WebDev-Skills-Bench四条件受控基准分离Skill内容与长度效应", "发现长度分心vs内容误导两种失败模式并量化跨模型分裂", "证实Skill效用是(Skill,project,model)三元组假设而非可移植资产"]
benchmarks: ["WebDev-Skills-Bench", "Web-Bench", "SkillsBench", "SWE-Skills-Bench"]
---

# 论文速读：Signal-or-Noise-A-Benchmark-Study-of-Agent-Skills-in-Web-Dev

## 一句话总结
论文提出 WebDev-Skills-Bench，对 31 个公开 Agent Skills 在 50 个 Web-Bench 项目、1,000 个顺序任务的条件下进行受控实验，发现 Skills 注入平均降低 Pass@2 1.3–4.2 pp、Token 成本激增 72–394%，且效用高度依赖 (Skill, project, model) 三元组而非单一 Skill 属性。

## 研究问题与动机
- **Skills 默认注入的隐性成本**：商业 Agent 将 Skill 作为持久化行为先验注入每个请求，Skill 越多 Prompt 越长，但从未有基准同时回答"Agent 能否解题"和"该 Skill 是否应当被注入"两个问题。
- **已有基准无法分离内容与长度效应**：Web-Bench、ArtifactsBench 等固定 Prompt 测能力，SkillsBench (+16.2 pp) 与 SWE-Skills-Bench (49/49 零提升) 结论矛盾，两者均未隔离 Skill 内容与其带来的 Prompt 长度增长。
- **跨模型 Skill 效用可正可负**：同一个 Skill 在 Sonnet 上可能 +33 pp，在 DeepSeek/Qwen 上却 −22 pp，缺乏 per-pair 审计导致静态排名不可迁移。
- **Easy-task 退化机制未被识别**：注入在模型已有强先验的简单早期任务上破坏重试灵活性，导致 chain 提前终止，而现有报告仅关注项目级均值。

## 核心贡献（创新点）
- **提出 WebDev-Skills-Bench**：基于 Web-Bench 的 50 项目/1,000 任务，覆盖 31 个第三方 Skill，首次在同一任务空间下比较 Skill 注入与不注入的边际收益。
- **引入四条件受控设计（C0/C1/C2/C3）**：C0 无 Skill 基线、C1 目标 Skill、C2 长度匹配的无关 Skill 控制、C3 leave-one-out 切片归因，使 ∆Length 与 ∆Content 可分离。
- **Workspace-aware 注入协议**：仅将 SKILL.md 写入 Prompt，auxiliary 文件挂载至文件系统，保证 C2 的字节匹配在所有多文件 Skill 下均可行。
- **发现两种失败模式**：Sonnet/Qwen 受长度分心（∆Length 占主导），GPT-5.1/DeepSeek 受内容误导（∆Content 占主导），同一均值得出相反根因。
- **证实 Skill 效用的三元组条件性**：跨模型 per-pair 相关系数 |r| ≤ 0.12，证明 Skill 是 (Skill, project, model) 假设而非可移植资产。

## 方法详解
- **任务语料**：Web-Bench 50 个项目、11 栈类别（React/Vue/Angular/Svelte、Express/Fastify、ORM/DB、CSS、Canvas/SVG/Three.js、bundler、DOM app），每项目 20 个顺序任务，共 1,000 任务，使用 Playwright 确定性测试而非 LLM-judge。
- **Skill 库**：31 个来自 Anthropic/Mindrally/Osmani/Vercel Labs/VoltAgent 的公开 Skill，覆盖 1.2K–22K 字符 SKILL.md；由两位具备 WebDev 经验的作者独立标注 1,550 个 (Skill, project) 对为 core/skip，Cohen's κ ≈ 0.74，最终 117 个 core pairs。
- **注入协议**：C0 空 Prompt、C1 注入目标 SKILL.md、C2 注入字节相近（±5%）的 skip-tier Skill、C3 移除 R_p/R_n/X 中任一切片；workspace 每次执行前 `git clean -fdx` 重置。
- **模型面板**：Claude Sonnet 4、GPT-5.1、DeepSeek-V4-flash、Qwen3-Coder-30B-A3B，greedy 解码、64k maxTokens，每 (model, condition, pair) 重复 N=3 次。
- **指标**：∆Pass@1、∆Pass@2（配对 bootstrap 95% CI）、∆Task Completion Depth（TCD，连续 Pass@2 前缀最长长度）、相对 Token 开销 ρ = (tokens_Cx − tokens_C0)/tokens_C0。

## 实验与结果
- **平均效应**：所有模型 ∆Pass@2 均为负，Sonnet −4.2 pp、Qwen −2.3 pp、DeepSeek −2.0 pp、GPT-5.1 −1.3 pp；仅 17%–36% (Skill, project) 对获益；ρ = +72% 至 +394%。
- **难度分布**：Easy-task 退化最显著且 CI 均排除 0（Qwen −10.7 pp、DeepSeek −7.3 pp、GPT-5.1 −4.0 pp、Sonnet −5.3 pp）；moderate/challenging 噪声大，DeepSeek 在 moderate 上 +11.7 pp。
- **长度/内容分解**：Sonnet ∆Length = −3.3、∆Content = −0.9（长度分心）；Qwen ∆Length = −3.5、∆Content = +1.2（长度分心）；GPT-5.1 ∆Length = −0.2、∆Content = −1.1（内容误导）；DeepSeek ∆Length = −0.6、∆Content = −1.4（内容误导）；Survival 比例 60%–95%。
- **跨模型相关**：Pearson |r| ≤ 0.12，74% pairs 至少一正一负，仅 1% 在四模型全正、4% 全负；典型案例 lowdb × database-optimizer 跨模型 swing 达 55 pp。
- **C3 切片归因**：在 5 个正向 tails 上 Whole Skill +5.1 pp；anti-patterns R_n +3.1 pp（McNemar p=0.008）最稳定；examples X 全局 null（−0.7 pp）但高度依赖模型：DeepSeek +8.3、Qwen +3.7、GPT-5.1 +0.7、Sonnet −15.3。

## 相关工作脉络
- **SkillsBench (Li et al., 2026)**：报告 +16.2 pp 增益，但未隔离 WebDev、未做长度控制；本文在同一任务空间下发现平均负效应，揭示其可能源于跨域异质性。
- **SWE-Skills-Bench (Han et al., 2026)**：49/49 Skill 零提升，但未限定 WebDev 且缺少 C2 控制；本文证明相同 Skill 在不同模型上可正可负，不能简单外推。
- **Web-Bench (Xu et al., 2025)**：固定 Prompt 测能力，未变 Skill 条件；本文在其 50 项目/1,000 任务上叠加 Skill 干预，正交扩展。
- **SWE-bench-style benchmarks**：通用工程任务评测；本文聚焦 WebDev 高流量领域并引入 chain-position 细粒度分析。
- **RAG / few-shot prompting**：Skill 本质是持久化 behavioral prior，不同于一次性检索或示例；本文将其视为需要 per-deployment 路由的假设。

## 局限性与未来方向
- **Seed 方差与 headline effect 量级相当**：Sonnet 三次 replicate 的 C0/C1 Pass@2 波动 4.4/3.6 pp，per-pair 估计需谨慎。
- **C2 测量未完全去重**：109 个唯一 C2 runs 覆盖 117 pairs，cluster bootstrap 与全 per-pair C2 去重待跟进。
- **Routing 偏保守**：仅评估 117 core pairs，off-target 部署行为未知；enterprise closed Skills / fine-tuned skill-routers 模式可能不同。
- **非在线 A/B**：Web-Bench 是 pre-deployment proxy，缺少真实用户流量、人工干预、产品验收标准。
- **评估维度局限**：Playwright 只测功能正确性，未覆盖视觉保真度、交互 UX、可访问性、developer review time。

## 研究启发与可借鉴点
- **长度匹配控制作为基准最低要求**：C2 设计可直接迁移至任何 Agent Skill / tool-use 评测，区分"更长"与"更错"两种退化源。
- **Chain-position 细分报告优于项目级均值**：Skill 效用集中在 early easy tasks，建议按任务顺序分段统计 ∆Pass 与 ∆TCD。
- **Slice ablation 揭示高价值组件**：C3 证明 anti-patterns 是 cost-effective 信号，examples 对强模型有害；可作为 Skill 设计与筛选的启发式。
- **Per-model routing 替代静态排名**：跨模型相关性接近零，marketplace 应提供 model-conditioned utility、prompt length、length-control survival。
- **Retry lock-in 机制可指导重试策略设计**：Skill 固定 structural choice 会剥夺简单重试的恢复空间；可在 agent loop 中加入"early-task bypass"逻辑。

## 关键术语表
- **Agent Skill**：以 Markdown 为主、可附带脚本/参考文件的持久化行为模块，在 session 内每个 Prompt 中注入，编码框架约定、反模式与工具。
- **WebDev-Skills-Bench**：本文提出的基准，在 Web-Bench 50 项目/1,000 任务上测试 31 个 Skill 的边际注入效应。
- **Pass@k**：k 次独立生成中至少一次通过 Playwright 测试的比例；本文主要报告 Pass@1 与 Pass@2。
- **Task Completion Depth (TCD)**：20 任务链中连续 Pass@2 的最长前缀长度，衡量 Skill 对 long-horizon resilience 的影响。
- **C0/C1/C2/C3**：四条件——无 Skill 基线、目标 Skill、长度匹配无关 Skill、leave-one-out 切片归因。
- **Length-distracted vs Content-misled**：前者指等效长度无关 Skill 复现大部分损失（Sonnet/Qwen），后者指长度中性但内容本身降分（GPT-5.1/DeepSeek）。
- **Survival**：C1 正增益对在经 C2 长度控制后仍保持正增益的比例，反映内容正效用的稳健度。
- **Retry lock-in**：Skill 固定的 structural choice 将可恢复的首次尝试错误转化为 chain-terminating 失败，破坏重试灵活性。

## 可复现要素
- **数据集**：Web-Bench（Xu et al., 2025）50 项目/1,000 任务，Playwright 测试套件；31 个第三方 Skill 来源明确。
- **代码/权重**：基准 harness、C0–C3 路由、per-pair/condition/output CSV、analysis pipeline 已开源至 https://anonymous.4open.science/r/webdev-skills-bench-1C32/；原始 per-task 报告与 base Web-Bench harness 从上游获取。
- **超参**：greedy 解码、temperature=0、maxTokens=64k、每 cell N=3 replicates、C2 字节匹配 ±5%、bootstrap 1,000 resamples。
