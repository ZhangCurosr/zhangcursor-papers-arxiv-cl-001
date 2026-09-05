---
title: "Beyond-the-Payload-How-User-Invocation-Shapes-Coding-Agent-V"
source: https://arxiv.org/pdf/2608.30686v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:53:36"
field: "AI安全与代码智能体"
keywords: ["coding agent", "repository poisoning", "prompt injection", "security evaluation", "prompt-level configuration", "ASR", "alert rate"]
innovations: ["首个系统变化用户侧提示级配置(PLCs)的编码智能体仓库投毒基准CIPR", "数据驱动从社交媒体提取提示风格并聚类用于安全评估", "揭示RUN-TESTS任务形成静默攻击面(高ASR低AR)的任务类型依赖性发现"]
benchmarks: ["CIPR", "SWE-bench", "SUSVIBE"]
---

# 论文速读：Beyond-the-Payload-How-User-Invocation-Shapes-Coding-Agent-Vulnerability-to-Repository-Poisoning

## 一句话总结
本文首次从**用户侧提示级配置（Prompt-Level Configurations, PLCs）**视角系统研究代码智能体对仓库投毒的脆弱性，提出了 CIPR 基准测试（1,920 个实例），发现攻击成功率和智能体告警率高度依赖于任务类型、提示表达风格和技能/规则配置，而非仅由攻击者载荷决定。

## 研究问题与动机
1. **现有工作局限**：现有仓库投毒研究主要聚焦攻击者可控因素（注入方式、伪装策略），忽略了开发者日常调用智能体时的提示级选择（任务类型、提示表达、技能/规则配置）如何共同塑造攻击成败。
2. **威胁建模范式转变**：需要从"攻击者如何构造载荷"转向" benign 用户的调用配置如何治理静态载荷的成功"，即用户侧 PLCs 对攻击结果的影响。
3. **实证知识空白**：当前缺乏一个结合真实仓库、真实社会媒体提示风格、可控技能/规则配置的基准，无法量化 PLCs 对编码智能体安全行为的系统性影响。
4. **防御设计需求**：如果某些日常调用模式无意中降低了智能体防御，开发者需要知道应避免哪些配置，但现有评估在单一规范提示下运行，无法提供此指导。

## 核心贡献（创新点）
1. **CIPR 基准（首个）**：首个系统性地变化 PLCs（任务类型 × 提示风格 × 技能/规则）并注入真实仓库的系统级基准，共 1,920 个实例（20 仓库 × 4 任务 × 4 提示风格 × 3 技能/规则 × 5 重复）。
2. **数据驱动的提示风格提取方法**：从社交媒体爬取 1,200+ 条真实编码智能体交互提示，经 LLM 标注 12 个风格维度后 K-means 聚类，提取三种最具区分度的提示风格用于实验（SFV、TIU、TNV）。
3. **双或门自动化评估框架**：设计了分别测量 ASR（通过 mock HTTP 服务器拦截出站请求）和 AR（通过 LLM judge 分析对话轨迹）的独立自动化或门，并经人工评估验证一致性（Cohen's κ = 0.83）。
4. **PLC 效应的系统性实证发现**：揭示了任务类型可造成最大 4.5 倍 ASR 差异（RUN-TESTS 为静默攻击面），提示表达通过执行深度和安全性显著性两条间接机制影响攻击结果，安全规则提升 AR 但不显著降低 ASR。

## 方法详解
- **威胁模型**：用户将 SE 任务 τ 表达为用户提示 P_user，智能体将其与系统提示 P_system、技能/规则 P_sr 拼接为 I_0 = P_sys ⊕ P_sr ⊕ P_user，随后进入循环调用 LLM 并与环境交互；攻击者可在仓库中注入恶意载荷。
- **仓库收集**：从 GitHub 收集 20 个仓库（Python/JavaScript/C/Java 各 5 个），需满足：至少两个已关闭的可复现 issue（bug-fix + feature-request）、存在构建配置文件（setup.py/package.json/Makefile 等）、存在测试套件；通过开发类型、文件数、LOC 三维度均衡选择。
- **任务类型（4 种）**：T1 PREPARE-ENV（安装构建项目）、T2 RUN-TESTS（运行测试套件）、T3 FIX-BUG（修复真实已关闭 bug）、T4 FIX-FEATURE（实现真实已关闭功能需求）；注入位置由任务隐式决定（环境准备类注入配置文件，测试类注入测试文件）。
- **提示风格（3 种）**：Socially Framed Vague（SFV，社交 framing + 高词汇模糊性）、Terse Indirect Underspecified（TIU，极简间接 + 信息严重不足）、Typo Noisy Vague（TNV，直接意图 + 高歧义/拼写噪声）；通过 LLM 基于聚类质心和代表性示例重写指令，保持任务语义不变。
- **技能/规则（3 种）**：No-Skills/Rules（空配置）、Normal Skills（从 GitHub 收集的常规生产力导向技能文件）、Security-Aware（Normal Skills + 显式安全指令，如禁止执行未验证代码）。
- **攻击载荷（2 种）**：Direct（恶意代码直接嵌入目标文件）、Direct-Deep（目标文件仅调用位于更深路径的帮助脚本，脚本含实际载荷），目标均为数据泄露（读取 /etc/passwd 并通过 HTTP POST 发送到受控服务器）。
- **评估或门**：ASR——拦截 mock 服务器上接收到的包含目标 secret 的出站请求；AR——使用 GPT-4o LLM judge 分析对话轨迹，判断智能体是否显式标记注入内容为可疑；两者在隔离 Docker 容器中独立评估。
- **实验设计**：4 × 4 × 3 因子设计，20 仓库 × 2 注入方法作为每个单元格的重复，有效样本 n=40/单元格，N=1,920；使用 Wilson 置信区间、卡方全域检验和逻辑回归进行统计推断。

## 实验与结果
- **数据集**：20 个真实开源仓库（5 Python / 5 JavaScript / 5 C / 5 Java），4 种任务类型，4 种提示风格（含 Baseline），3 种技能/规则配置，2 种注入方法。
- **评估基线**：Codex + GPT-5.4（主要），另对比 Codex + GPT-5.5、OpenCode + GPT-5.4、Claude Code + Sonnet 4.6。
- **核心结果**：
  - **任务类型**：RUN-TESTS 的 ASR 最高（45.5%，95% CI [41.1, 50.0]），FIX-BUG 最低（8.6%，95% CI [6.4, 11.5]），**相差 4.5 倍**；RUN-TESTS 同时具有最低 AR（9.4%）和最高 P(read)，形成**静默攻击面**（高 ASR + 低 AR）。
  - **提示风格**：TIU 相对 Baseline 显著降低 ASR（Adjusted OR = 0.71，95% CI [0.51, 0.98]，p = 0.036），因简短提示截断执行深度；TNV 方向性降低 AR（-3.7pp）并微弱增加 ASR（+2.0pp），但不显著。
  - **技能/规则**：Security-Aware 提升 AR（14.8% vs 10.5%），但 ASR 置信区间重叠，说明安全规则提升显式检测但不统一阻止攻击。
- **跨模型泛化**：任务类型效应（RUN-TESTS 最高 ASR）在四种 Agent×模型组合中一致；绝对 ASR 差异大：Codex+GPT-5.4/5.5 最高，Claude Code+Sonnet 4.6 最稳健。

## 相关工作脉络
1. **仓库投毒攻击面研究**（Liu et al., 2025; Qu et al., 2026; Kao et al., 2026; Xie et al., 2025）：聚焦攻击者控制的注入表面（规则、技能、README），本文与之本质区别是转向用户侧 PLCs，而非攻击者载荷设计。
2. **提示敏感性研究**（ProSA Zhuo et al., 2024; RobustAlpacaEval Cao et al., 2024; PhishNChips Litvak, 2026）：关注提示变化对 LLM 行为/安全检测的影响，但多在单次生成场景；本文关注 PLCs 引发的系统级执行行为变化（探索深度、文件解读方式）。
3. **编码智能体安全基准**（SWE-bench Jimenez et al., 2024; SUSVIBE Zhao et al., 2025）：评估功能正确性或Agent生成代码安全性；本文首次将真实仓库投毒与用户端配置变化结合，提出 ASR+AR 双指标体系。
4. **编码智能体平台**（SWE-agent Yang et al., 2024; OpenHands Wang et al., 2025; Meta-GPT Hong et al., 2024）：研究 agent 工具接口和多 agent 协作；本文定位于这些平台的**安全评估**层面。
5. **供应链攻击实证**（He et al., 2024 假星研究；Kurmi 2026 axios 投毒事件）：提供攻击者行为动机背景；本文与之配合，从防御者/用户侧视角补充理解。

## 局限性与未来方向
1. **PLC 覆盖不全**：仅研究提示级配置，未涵盖记忆设置、MCP 服务器、工具权限策略、IDE 集成等其他用户侧因素，这些因素更难跨智能体标准化。
2. **攻击目标单一**：仅评估数据泄露，未覆盖破坏性命令执行和持久化妥协等其他危害类型。
3. **交互模式局限**：仅评估无约束自动化模式（UAM），未包含人工审批等交互模式，未来需系统评估不同执行/权限模式下的 PLC 效应。
4. **提示风格外推**：从 635 条社交媒体提示聚类提取的风格，是否完全代表全球开发者使用习惯，存在地域/平台偏差。

## 研究启发与可借鉴点
1. **双指标或门设计值得复用**：ASR（攻击是否成功执行）+ AR（智能体是否显式告警）的组合提供了更全面的安全视图，可迁移至其他智能体安全评估基准。
2. **数据驱动的提示风格提取方法**：从真实用户交互数据（社交媒体、issue 评论等）经 LLM 标注 + 聚类提取风格类别，优于研究者主观定义提示风格，适用于任意智能体交互安全评估。
3. **"静默攻击面"概念**：RUN-TESTS 任务的发现（高 ASR 低 AR）揭示了当智能体将注入文件视为"基础设施"而非"待审计配置"时会产生安全盲区，这一解释框架可推广到其他任务类型。
4. **安全规则的局限性启示**：单纯添加安全规则提升告警率但不降低攻击成功率，提示底层防御需要**系统级强制机制**（如告警时立即中止执行），而不仅是提示层面约束；这对本团队研究 Agent 防御系统设计有直接参考价值。
5. **因子实验设计 + 统计建模**：4×4×3 因子设计结合 Wilson CI、卡方全域检验和协变量调整逻辑回归的组合，为大规模智能体安全评估提供了可复用的统计推断范式。

## 关键术语表
- **Prompt-Level Configurations (PLCs)**：用户在调用编码智能体时做出的提示级配置选择，包括任务类型、用户提示表达风格和技能/规则配置。
- **CIPR (Coding In Poisoned Repos)**：本文提出的首个系统变化 PLCs 的真实仓库投毒评估基准，包含 1,920 个实验实例。
- **Attack Success Rate (ASR)**：攻击成功率的衡量指标，定义为 mock HTTP 服务器实际接收到含目标 secret 的出站请求的实验比例。
- **Alert Rate (AR)**：智能体告警率，定义为智能体在对话轨迹中显式标记注入内容为可疑/拒绝执行的实验比例。
- **静默攻击面（Silent Attack Surface）**：指高 ASR 且低 AR 的任务配置组合（如 RUN-TESTS），智能体执行了恶意载荷但几乎不发出安全告警。
- **Direct vs. Direct-Deep 注入**：Direct 将恶意代码直接嵌入目标文件；Direct-Deep 在目标文件中仅放置对深层路径辅助脚本的调用，降低注入可见性。
- **Socially Framed Vague (SFV)**：一种社交 framing + 高词汇模糊性的提示风格，包含闲聊式社交前缀和非正式语气。
- **Terse Indirect Underspecified (TIU)**：极简间接且信息严重不足的提示风格，通过截断执行深度间接降低 ASR。

## 可复现要素
- **数据集**：20 个真实 GitHub 仓库，任务元数据和注入脚本将公开发布（论文附录声明 artifact 可用性）。
- **代码/权重**：CIPR 基准的注入脚本和评估基础设施将公开发布以支持防御性研究；原始社交媒体截图不公开，仅发布风格描述和聚类质心。
- **关键超参**：K-means 聚类 K=15；MinHash LSH Jaccard 阈值 0.90；LLM 标注 temperature=0.0（重标注 subset 用 0.3）；提示重写 temperature=0.7；Docker 隔离容器评估。
- **评估模型**：主实验 Codex + GPT-5.4 (gpt-5.4-2026-03-05)；AR judge 使用 GPT-4o (gpt-4o-2024-08-06)；提示重写使用 gemini-3.1-flash-lite。
