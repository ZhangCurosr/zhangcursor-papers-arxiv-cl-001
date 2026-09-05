---
title: "Aspire-Can-Models-Self-Evolve-from-Vague-Goals"
source: https://arxiv.org/pdf/2608.31111v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:33:02"
field: "LLM自进化与自主后训练"
keywords: ["vague-goal self-evolution", "target operationalization", "LLM agent", "post-training benchmark", "hidden evaluation", "weight evolution", "harness evolution"]
innovations: ["形式化目标操作化为自主后训练的独立维度", "引入密封评估的最小交互Aspire基准支持权重与harness双重演化", "揭示完成训练循环与保留目标对齐增益之间的关键差距"]
benchmarks: ["Aspire (520-item hidden evaluation, 6 goals)", "PostTrainBench (vague-goal adaptation)"]
---

# 论文速读：Aspire-Can-Models-Self-Evolve-from-Vague-Goals

## 一句话总结
本文提出 Aspire 基准，用于评估大语言模型在仅有模糊能力目标（如"成为更好的物理学家"）且下游评估任务被隐藏的情况下，能否自主将目标操作化为可训练任务、选择数据与更新方法，并最终在隐藏专家评测集上实现能力增长；实验表明当前智能体虽能完成训练循环，但权重级增益稀疏不稳定，且最强演化后的 harness 仍低于人工工程参考。

## 研究问题与动机
1. **现有工作局限**：已有 LLM 自进化研究（如 PostTrainBench、LaMDAgent、SEAL 等）均以人类指定的具体任务、评估脚本和成功指标为前提，智能体仅负责搜索"如何优化"，无法决定"优化什么"。
2. **真实部署鸿沟**：工业界 forward-deployed engineer (FDE) 的核心工作正是将模糊的部署需求转化为可操作目标、约束与成功标准，并构建验证信号；当前基准隐藏了这一关键环节。
3. **目标未饱和问题**：当现有基准饱和或不再反映新兴能力缺口时，人类仍需发现新弱点并将其转化为可计算任务，阻碍了真正的递归自改进。
4. **核心问题**：给定模糊目标且无 agent 可见的任务规范或可分解奖励，智能体能否自主完成目标诊断、子目标分解、学习信号构造，并在保留基础能力的前提下实现真实能力增益？

## 核心贡献（创新点）
1. **形式化"目标操作化"**：将"将广义能力目标转化为可训练目标、学习信号与验证标准"定义为自主后训练的缺失维度，区别于现有工作仅搜索优化路径的做法。
2. **引入 Aspire 基准**：提供密封式隐藏评估（520 题、六大目标）与最小交互环境，支持模型权重演化与 agent harness 演化两种表面，评测时智能体不可见评估项与参考答案。
3. **结果与轨迹双重证据**：揭示"完成训练循环"与"保留目标对齐改进"之间的差距——智能体常训练在失配数据上、过度信任窄代理验证，局部增益无法迁移至隐藏评估，且持续搜索可能抹除已有改进。

## 方法详解
1. **任务定义与演化契约**：每个 campaign 由契约 $\Gamma = (G, \mathcal{E}_G, J, A, B, \Sigma)$ 定义，其中 $G$ 为模糊自然语言目标，$\mathcal{E}_G$ 为绑定该目标的版本化评估器（对 agent 隐藏），$J$ 为所需时的 judge，$A$ 为类型化动作契约，$B$ 为预算，$\Sigma$ 为预声明的终端选择规则。初始状态 $Y_r^{\text{in}} = (M_r, H_r, D_r)$ 包含模型权重、agent harness 与决策模型。
2. **两种演化表面**：
   - 权重演化（RQ1–RQ2）：固定 harness $H_0$ 与决策模型 $D_r$，优化模型权重 $M$。
   - Harness 演化（RQ3）：固定模型权重 $M_0$，优化 agent harness $H$（运行时指令、工具策略、工作流、记忆与验证逻辑）。
3. **隐藏评估集**：520 道专家原创题目，覆盖六大目标（科学学术推理 75 题、人文社科知识 110 题、健康医学推理 100 题、数学推理 126 题、逻辑可靠性与指令遵循 89 题、学术科学写作 20 题）。评估项与评分标准对 agent 完全隐藏，仅返回协议允许的聚合分数。
4. **最小交互环境**：暴露单一统一工具，支持搜索/下载/导入公开数据集、合成注册数据、启动 SFT 或 GRPO（含 LoRA）、查询任务状态、在自有验证数据上运行检查、分支或终止。控制器处理凭证、存储、分布式作业与恢复，agent 只需做战略决策。
5. **安全保留机制**：定义原始分数变化 $\Delta^{\text{raw}}(k) = \hat{s}_\Gamma(Y(k)) - \hat{s}_\Gamma(Y^{\text{in}})$，仅当最佳候选 $k^\star$ 满足 $\Delta^{\text{raw}}(k^\star) > 0$ 时才保留，否则回滚至初始状态，确保保留增益 $\Delta^{\text{ret}} \geq 0$。
6. **反馈协议**：
   - Final-only：无中间评分，仅提交一次终态检查点进行评估。
   - Adaptive-feedback：允许有限次查询同一目标的聚合分数，用于指导后续更新/分支/停止决策。

## 实验与结果
1. **RQ1（模糊目标 vs 显式任务）**：Claude Opus 4.8 在模糊目标下得分为 27.07，低于官方 PostTrainBench 参考 32.90；GPT-5.6 为 29.58 vs 36.23。模糊目标将搜索重心从"如何优化"转向"目标解释与操作化"，匹配运行中决策模型思考时间增加 2,109 秒、GPU 空闲时间增加 0.61 小时，活跃训练时间减少 1.27 小时。
2. **RQ2a（Final-only 权重演化，24 次运行）**：Qwen3.5-4B Self 在 6 个目标上 0/6 均值超过基线；Qwen3.5-9B Self 仅 1/6（科学学术推理从 45.33 提升至 48.00，+2.67 分）。24 次中仅 3 次单运行超过基线，21 次触发回滚。
3. **RQ2b（Adaptive-feedback 权重演化，30 个配置-目标单元）**：28/30 产生可评估检查点，21/30 满足合格标准。仅 2 个单元的最佳检查点分数超过基线：Self 科学目标从 44.00→45.33，Terra 优化 Qwen3.5-4B 数学从 17.86→20.10（唯一保留改进）。持续搜索不保证更好结果，Sol 产生 33 个检查点但无一超过基线。
4. **失败模式**：Self 在 30/32 次数据导入中使用 GSM8K 或 Hendrycks 数学数据（即使目标为科学/逻辑/写作）；5 个使用数值标签 MMLU SFT 的检查点输出全部坍缩为单个数字，得分 0–6.14。
5. **RQ3（Harness 演化）**：固定 Qwen3.5-4B，三个外部决策模型（Luna/Terra/Sol）生成后继 harness，均低于 Qwen-Agent 参考（28.64 task macro / 27.65 example micro）；Sol 最佳但仍有 1.42 分差距。Luna 因窄代理验证过早停止；Terra 遗漏最终答案不变量导致部分草稿被提交（占聚合下降约 70%）。

## 相关工作脉络
1. **Self-training / Self-evolution 方法**（STaR、ReST-EM、Self-Rewarding LM、SPIN、R-Zero、Absolute Zero、SEAL）：均依赖具体信号（正确性、奖励、自博弈结果或任务增益）定义进展；Aspire 不引入新训练算法，而是测试仅给模糊目标时模型能否构造有效学习过程。
2. **Automated post-training benchmarks**（PostTrainBench、LaMDAgent、RSIBench-Data、AI4AI-Bench、PAST-Bench）：大多从已操作化的任务/分数/评估器出发；RSIBench-Data 发现的"持续搜索多数低于自身最佳检查点"与本文"完成训练循环≠保留能力增益"相呼应。
3. **ML 实验自动化**（MLE-bench、MLAgentBench、RE-Bench、AIDE、DS-Agent、SELA）：测量 ML 工程与 AI R&D 的量化目标，但仍始于已指定任务。
4. **系统/代码改进**（ADAS、Darwin Gödel Machine、Hyperagents、Frontis-MA1、Red Queen Gödel Machine）：改进 agent 系统或代码，但同样依赖明确目标。
5. **Memory / Workflow / Skill evolution**（Evo-Memory、EvoFSM、SkillLearnBench、HELIX、Co-Harness）：研究记忆、工具与工作流的演化；RQ3 将此线与模糊目标学习连接，检验运行时权重固定时 harness 能否演化。
6. **开放研究能力评估**（Kirgis et al., 2026）：影子评估发现前沿 agent 可完成工程推进但未推动未发表研究问题，呼应本文关于"目标设定仍是人类瓶颈"的观点。

## 局限性与未来方向
1. **目标覆盖有限**：仅六个模糊目标，未来需更广泛覆盖。
2. **评估集质量与数量**：520 题Expert 原创，但单配置-目标仅一次运行，缺乏多种子重复估计。
3. **递归演化未测试**：当前所有运行固定初始决策模型，未测试递归决策模型替换或 harness 的递归自我改进（$H_1 \to H_2$）。
4. **无关能力保留未测量**：RQ2 仅评估单一目标，未衡量更新是否破坏指令调校已获得的其他能力。
5. **内容级轨迹分析受限**：可移植 artifact 不含原始模型消息与 per-item judge 推理，因果分析需受控访问。

## 研究启发与可借鉴点
1. **密封评估设计**：Aspire 的"科学仪器式"隐藏评估器设计值得借鉴——评估集对 agent 完全密封，仅返回聚合分数，可有效防止过拟合评估项同时测量真实能力迁移。
2. **目标操作化作为独立研究轴**：将"目标诊断→子目标分解→学习信号构造→验证标准确立"形式化为独立环节，为后续研究提供清晰的分析框架。
3. **安全保留机制**：$\Delta^{\text{ret}} \geq 0$ 的回滚策略确保不会因无效更新而退化，可作为自进化系统的通用安全原语。
4. **双重演化表面**：同时支持权重演化与 harness 演化，揭示了"模型能力"与"围绕模型的 agent 系统"可分别优化的研究视角。
5. **失败模式诊断价值**：数值标签 SFT 导致的输出格式坍缩、窄代理验证引起的过拟合、遗漏最终答案不变量导致的草稿提交等失败模式，为后续自进化工作提供具体调试方向。

## 关键术语表
**Vague-goal-driven self-evolution**：仅给定模糊能力目标（如"成为更好的物理学家"）且无 agent 可见任务规范时，模型自主决定学习什么、如何学习并验证进步的过程。
**Target operationalization**：将广义能力目标转化为可训练目标、学习信号与验证标准的转换过程，是自主后训练的缺失维度。
**Hidden evaluation set**：对 agent 完全密封的专家原创评估集（520 题），仅返回协议允许的聚合分数，用于测量真实能力迁移。
**Evolution surface**：被演化的组件层面，分为权重演化（优化 $M$）与 harness 演化（优化 $H$），另一方固定。
**Adaptive-feedback protocol**：允许 agent 在进化过程中获得有限次聚合分数查询，用于指导后续更新/分支/停止决策的协议。
**Final-only protocol**：无中间评分，agent 仅能提交一次终态检查点进行评估的开放循环协议。
**Retained improvement**：经安全回滚机制筛选后实际保留的能力增益，$\Delta^{\text{ret}} \geq 0$，区别于中间检查点的瞬时最高分。
**Agent harness**：围绕模型的运行时系统，包括提示指令、工具策略、工作流、记忆机制与验证逻辑。

## 可复现要素
- **数据集**：520 题隐藏评估集（六大目标），专家原创，不与 GPQA/MMLU-Pro/MedQA 重叠；论文未公开评估集内容（密封设计），但提供构造流程与质量控制的详细附录。
- **代码/权重**：项目页面 https://self-developing-agents.github.io/；论文未明确声明代码开源状态。
- **关键超参**：训练预算 40 GPU-hours/run（RQ2）；决策模型包括 Qwen3.5-4B Self、Qwen3.5-9B Self、GPT-5.6 Luna/Terra/Sol；基础模型 Qwen3.5-4B/9B；RQ3 固定 Qwen3.5-4B。
- **训练方法**：SFT、GRPO、LoRA/PEFT。
- **评估器**：确定性/结构化评分优先；开放-ended 响应使用 campaign 固定的 rubric-based judge（gemini-3.5-flash，8,192 token thinking budget，10% fail-closed threshold）。
