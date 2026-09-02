---
title: "LongWoF-Bench-Evaluating-EvoMap-Genes-for-Verifiable-Long-Wo"
source: https://arxiv.org/pdf/2608.23200v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:10:45"
---

# 论文速读：LongWoF-Bench-Evaluating-EvoMap-Genes-for-Verifiable-Long-Wo

## 一句话总结
本文提出 LongWoF-Bench 基准，系统评估 EvoMap 框架中经验证器确认的执行轨迹所构建的 Gene 与常规 Skill 在可验证长工作流任务上的表现。结果表明经验溯源（provenance）起决定性作用：Evolved Gene 在 7 个模型上均显著优于 Skill（提升 8.7–15.5 个百分点）且可跨模型家族复用，同时减少 9.9% 求解 Token 消耗；而仅靠参考蒸馏的 Gene 则无效甚至劣于 Skill。

## 研究问题与动机
- **核心问题**：LLM 完成复杂端到端工作流后，调试过程中积累的纠错策略、边界条件与失败防护通常随单次运行消失，如何将其外部化并跨模型复用？
- **现有 Skill 的局限**：Skill 主要编码静态程序知识（流程、接口规范、推荐实践），无法捕获验证器反馈驱动的“哪类看似合理的局部决策会导致全局失败”这类经验。
- **评测缺口**：现有 Agent 基准（如 SWE-bench、WebArena、OSWorld）侧重工具调用与开放环境能力，缺乏在统一严格机器验证下对比“过程性指导”与“验证型执行经验”的控制实验。
- **实际诉求**：代码实现、环境构建、规则推理等场景强依赖跨步骤约束一致性，单次试错成本高昂，亟需可复用、可追溯的经验资产机制。

## 核心贡献（创新点）
1. **提出 LongWoF-Bench 基准**：包含 778 个机器可验证任务，覆盖代码生成、智能体环境合成、数学推理与规则遵循四类长工作流。与已有基准的区别在于统一了严格端到端验证协议与公开/隐藏信息边界，聚焦约束依赖型任务而非单纯上下文长度。
2. **建立 No Context / Skill / EvoMap Gene 控制对照框架**：首次在同一任务集、运行环境与验证器下直接对比静态程序指导与验证型经验的价值。与 SkillsBench 等研究的区别在于将“经验溯源”作为核心变量，而非仅评估技能泛用性。
3. **揭示经验溯源对 Gene 有效性的决定性作用**：Evolved Gene 跨 7 模型稳定超越 Skill，而 Reference-distilled Gene 反而劣于 Skill，证明紧凑表征本身不足，必须以验证器确认的迭代轨迹为来源。
4. **证明 Gene 复用可同步提升完成率与推理效率**：对 Claude Opus 而言，Gene 比 Skill 多解决 39 个任务并减少 9.9% 求解期 Token；相比多轮发现流程节省 45.8% 成本，确立了经验摊销的实证基线。

## 方法详解
- **任务形式化**：将可验证长工作流任务定义为 $\mathcal{T} = (S, E, \mathcal{Y}, V)$，其中 $S$ 为公开规范，$E$ 为模型可见环境，$\mathcal{Y}$ 为产物空间，$V$ 为私有机器验证器。成功当且仅当 $V(S, E, y) = 1$。难点在于后期决策依赖前期状态/产物，局部合理选择易累积为接口违规、顺序错误或边界遗漏。
- **Gene 构建（Evolver 流程）**：Producer 模型（本文用 Claude Opus 4.8）在无验证器隐私信息条件下首轮尝试；若失败，下一轮接收前序解与脱敏反馈（sanitized verifier feedback）并迭代修订，直至通过验证。随后将验证通过的完整轨迹蒸馏为结构化 Gene，保留成功策略、前置检查、边界条件与失败修正记录。
- **Gene 复用（EvoMap 协议）**：Consumer 模型仅接收公开规范与对应 Gene，不可访问 Producer 轨迹或验证器反馈。支持同模型复用与跨模型家族复用（如 Opus 生成的 Gene 直接用于 Gemini/MiniMax/Qwen）。
- **Skill 对照构造**：Skill 通过参考侧教师信号静态蒸馏，侧重流程规范与接口约定，不含验证器确认的纠错轨迹，作为“纯程序性知识”的基线。
- **评估协议**：固定公开规范、运行时、解码配置与私有验证器，仅辅助指导不同。每任务-条件配对单次采样（one-shot），使用严格通过率（strict pass rate）、求解期 Token 数、调用次数为指标，配对差异采用确定性任务级 bootstrap 置信区间与 McNemar 检验。

## 实验与结果
- **数据集划分**：LongWoF-Bench 共 778 任务（代码生成 341、Agent 环境合成 127、数学推理 151、规则遵循 159）。主实验聚焦 Opus 成功生成 Evolved Gene 的 252 任务；526 任务为 Reference-distilled（Opus 未能在进化预算内验证通过）；180 任务为 Opus 与 Gemini 共同验证通过的双生产方子集。
- **评测模型**：Claude Opus 4.8 / Sonnet 4.6、Gemini 3.1 Flash-Lite Preview / Pro Preview、MiniMax M3、Qwen3-Coder-30B-A3B-Instruct、Qwen3.5-397B-A17B。
- **主要结果**：
  - 平均严格通过率：No Context 41.0% → Skill 51.2% → Gene 62.9%。Gene 较 Skill 在所有 7 模型上提升 8.7–15.5 个百分点（Opus 从 63.9% 增至 79.4%，+15.5pp）。
  - 溯源分析：Op
