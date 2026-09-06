---
title: "DualStake-Dual-Path-Confidence-Calibration-in-Deep-Research"
source: https://arxiv.org/pdf/2609.00935v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:24:15"
field: "大语言模型可信推理"
keywords: ["置信度校准", "Deep Research Agent", "强化学习", "证据置信度", "stake reward", "GRPO"]
innovations: ["提出 DualStake 双路径校准方法，联合监督 E-Conf 和 A-Conf", "发现 E-Conf 比 A-Conf 提供更强的不确定性信号且 A-Conf 主要由 E-Conf 塑造", "在 stake reward 中引入 margin clipping 防止极端置信度优化损害准确率"]
benchmarks: ["NQ", "TriviaQA", "HotpotQA", "2WikiMultiHopQA", "MuSiQue", "Bamboogle", "SimpleQA", "PopQA"]
---

# 论文速读：DualStake-Dual-Path-Confidence-Calibration-in-Deep-Research

## 一句话总结
本文针对 Deep Research Agent 严重过度自信（overconfidence）问题，提出 DualStake 双路径置信度校准方法，通过在 GRPO 训练中联合监督检索后的证据置信度（E-Conf）与生成后的答案置信度（A-Conf），使用带边界的 stake reward 实现置信度与答案正确性对齐，在 8 个 QA 基准上显著改善校准指标且不牺牲准确率。

## 研究问题与动机
- **Deep Research Agent 存在严重过置信问题**：多轮检索+决策型生成过程中，模型反复获取外部证据并经历 RL 优化，导致表达置信度不可靠，影响用户信任与下游拒答（abstention）等应用。
- **现有校准方法仅关注最终答案**：已有方法（如 GRPO + Temperature Scaling、RLCR、MSCR）仅在生成答案后 elicitation 一次置信度，将推理视为单步事件，无法捕获多步检索中模型的认知状态演变轨迹。
- **A-Conf 并非独立反映答案正确性**：机制分析表明 A-Conf 的 logit 区分度极低，其表征几乎不编码答案正确性信息，而是主要被前序 E-Conf 所主导（activation patching 显示 E-Conf token 的因果影响峰值 Δ=+0.363，而答案 token 峰值仅 Δ=+0.073）。
- **直接优化 stake reward 可能导致极端置信度**：若不施加约束，模型可能通过极端化置信度（推向 0 或 1）来放大 reward 幅度，从而干扰答案正确性优化。

## 核心贡献（创新点）
1. **构建检索级逐步置信度 elicitation pipeline**：在 Search-R1 基础上，于每次检索后插入 `<confidence>` 查询，定义 E-Conf；保留原有的 `<final-confidence>` 为 A-Conf，揭示 E-Conf 提供比 A-Conf 更强的不确定性信号，且 A-Conf 主要由 E-Conf 塑造。
2. **提出 DualStake 双路径校准方法**：将带 margin clipping 的 stake reward 独立作用于 E-Conf 和 A-Conf 两条路径，通过 `(2q-1)·clip(c, m⁻, m⁺)` 使高置信度奖励正确、惩罚错误，同时防止极端置信度优化损害准确率。
3. **通过机制分析（logit 分布、线性探针、activation patching）验证 E-Conf 优势**：发现 A-Conf logit 条件熵标准差仅 0.0012（与答案正确性无关），而 E-Conf 为 0.28；A-Conf 在测试时 AUC 略优是因其偶尔分配极值"1"所致，并非真正具备区分能力。
4. **在 3 个模型 × 8 个 QA 基准上验证 SOTA 校准效果**：Qwen2.5-7B 上 ECE 从 0.518 降至 0.178（-65.6%），AUC 从 0.552 提升至 0.712（+29.0%），BS 从 0.497 降至 0.220（-55.7%），准确率保持不变。

## 方法详解
- **Pipeline 扩展**：在 Search-R1 框架下，每次检索步骤后输出 `<confidence>X</confidence>`（E-Conf），答案生成后输出 `<final-confidence>X</final-confidence>`（A-Conf），两者基于相同累积证据，仅在序列中的位置不同。
- **Stake Reward（赌注奖励）**：受 proper scoring rule 启发，将置信度 c 视为赌注，正确答案时获得 `+c` 奖励，错误时遭受 `-c` 惩罚：$\tilde{R}_s = (2q - 1) \cdot c$，其中 q 为 token-level F1 答案正确性，c ∈ [0,1] 为归一化置信度。
- **Margin Clipping**：对 stake reward 施加边界裁剪以稳定训练：$R_s = (2q-1) \cdot \min(\max(c, m^-), m^+)$，默认 $m^+=0.9$，$m^-=0.1$，防止模型通过极端置信度规避惩罚或过度放大 reward。
- **Dual-Path 联合监督总 reward**：$R_{\text{DualStake}} = R_{\text{corr}} + R_{\text{form}} + \alpha(t) \cdot R_s^{\text{evi}} + \beta(t) \cdot R_s^{\text{ans}}$，其中 $R_{\text{corr}}$ 为答案正确性 reward（F1），$R_{\text{form}}$ 为格式 reward（默认 0.1），α(t) 和 β(t) 为线性 warm-up 调度系数，默认 α=β=0.25。
- **训练框架**：基于 GRPO，每样本 5 次 rollout，检索使用 E5 encoder + 2018 Wikipedia 索引，每查询 top-3 passages。

## 实验与结果
- **数据集与模型**：训练集为 NQ + HotpotQA 合并；评测 8 个 QA 基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle、SimpleQA）； backbone 为 Qwen2.5-7B、Qwen2.5-7B-Instruct、Qwen3-4B。
- **评估指标**：ACC（Exact Match）、ECE↓、AUC↑、BS↓，另报告选择性预测 AURC。
- **主要结果（Qwen2.5-7B，Table 1）**：
  - vs. Vanilla GRPO：ECE 0.518 → 0.178（-65.6%），AUC 0.552 → 0.712（+29.0%），BS 0.497 → 0.220（-55.7%），ACC 0.370 → 0.375（持平）。
  - vs. MSCR：ECE 0.300 → 0.178（-40.7%），AUC 0.647 → 0.712（+10.1%）。
  - vs. RLCR：AUC 0.682 → 0.712，BS 0.258 → 0.220。
  - DualStake 在 AUC 和 BS 上均取得全部方法中的最优。
- **跨模型泛化（Table 2）**：Qwen2.5-7B-Instruct 上 AUC 0.742（最优），BS 0.213（最优）；Qwen3-4B 上 AUC 0.726（第二），BS 0.237（第二）。
- **选择性预测（Appendix A.6）**：DualStake 较 GRPO 在 4 个数据集上 AURC 平均降低 13.5%（0.600 → 0.516），低覆盖率（20%）下 selective accuracy 提升尤为显著（如 NQ 从 0.535 提升至 0.659）。
- **Ablation（Table 3）**：Margin clipping 在所有 α/β 配置下均有正向贡献；E-Conf-only 单独使用 ECE/BS 仍偏高（0.463/0.425），加入少量 A-Conf 权重后显著改善。

## 相关工作脉络
- **Search-R1（Jin et al., 2025）**：Deep Research agent 的代表性框架，本文以其为基线 pipeline，在其基础上扩展逐步置信度 elicitation。
- **RLCR（Damani et al., 2025）**：Training-time calibration 方法，结合正确性 reward 与 Brier-score-based 校准 reward；本文与其区别在于 RLCR 仅监督最终 A-Conf，而 DualStake 同时监督 E-Conf 和 A-Conf 两条路径。
- **MSCR（Xuan et al., 2026b）**：引入 margin-separated calibration reward 到 GRPO；本文与其区别在于 MSCRs 的 margin 分离的是正确/错误样本的校准 reward，而 DualStake 的 margin 是对置信度数值本身的 clipping 约束，且 DualStake 引入了 E-Conf 路径。
- **Rewarding Doubt（Bani-Harouni et al., 2026, ICLR 2026）**：在 GRPO 中基于 proper scoring rule 训练 LLM 输出校准置信度；本文沿袭其 RL 校准范式，但将其拓展至 Deep Research 的多步检索场景。
- **Agentic Confidence Calibration（Zhang et al., 2026）**：针对 agentic 场景的置信度校准工作，但仍将置信度视为 post-answer 信号；本文定位为首次系统性地研究 Deep Research 中多步检索阶段 E-Conf 与 A-Conf 的差异与协同。
- **Temperature Scaling / Sequence Probability（Guo et al., 2017; Zhao et al., 2023）**：Post-hoc calibration 方法，不修改模型参数；本文强调 training-time 校准的优势，post-hoc 方法虽可降低 ECE 但无法改善 AUC 和 BS。

## 局限性与未来方向
- **模型规模受限**：实验主要集中在 3B-7B 参数量的模型，未验证在更大规模模型（如 70B+）上的效果。
- **数据集覆盖有限**：仅使用 8 个英文 QA 数据集，未涉及多语言、代码生成或其他任务类型。
- **部分基线在个别数据集上表现更优**：如 ECE 指标上 RLCR 在 SimpleQA 等数据集上低于 DualStake，说明在特定任务上仍有优化空间。
- **E-Conf-only 监督不足**：虽然 E-Conf 提供更强的不确定性信号，但单独监督无法达到最佳校准效果，仍需 A-Conf 协同。

## 研究启发与可借鉴点
- **机制分析指导方法设计**：通过 logit 分布分析、线性探针（layer-wise linear probing）和 activation patching 三种互补手段，系统地揭示了 E-Conf 与 A-Conf 的差异根源，这种"先分析后设计"的研究范式值得借鉴。
- **Margin Clipping 的通用性**：stake reward 中引入边界约束以防止极端置信度优化的思路，可迁移到其他基于 confidence-dependent reward 的 RL 校准场景中。
- **双路径联合监督设计**：证明了"强信号源（E-Conf）+ 弱信号源（A-Conf）"的协同监督优于单一路径，对多步骤 agent 的中间状态校准具有启发意义。
- **训练调度策略**：α(t) 和 β(t) 的线性 warm-up 设计避免训练初期校准 reward 过大干扰正确性学习，是可复用的训练技巧。
- **团队结合机会**：与本团队在 RL-based agent 训练方向高度契合，可将 DualStake 的思路迁移至搜索增强推理（Search-R1 类）或工具调用 agent 的置信度校准研究中。

## 关键术语表
**Deep Research Agent**：通过多轮检索、证据聚合和决策型生成解决知识密集型任务的 AI agent 范式。
**Evidence Confidence（E-Conf）**：在最后一次检索步骤后 elicited 的置信度，反映模型基于当前累积证据的不确定性评估。
**Answer Confidence（A-Conf）**：在答案生成后 elicited 的置信度，即传统 post-answer 置信度。
**Stake Reward**：受 proper scoring rule 启发的 confidence-dependent reward，形式为 $(2q-1) \cdot c$，高置信度奖励正确、惩罚错误。
**Margin Clipping**：对 stake reward 中的置信度施加上下界约束 $\min(\max(c, m^-), m^+)$，防止极端置信度干扰训练。
**GRPO（Group Relative Policy Optimization）**：DeepSeekMath 提出的强化学习算法，通过组内相对优势估计策略梯度。
**ECE（Expected Calibration Error）**：将预测按置信度分箱后，计算各箱内平均置信度与平均准确率的加权偏差。
**AURC（Area Under Risk-Coverage Curve）**：选择性预测任务中的评估指标，值越低表示按置信度排序后风险-覆盖率曲线越优。

## 可复现要素
- **训练数据集**：NQ + HotpotQA 训练集合并（公开数据集，可复现）。
- **评测数据集**：8 个公开 QA 基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle、SimpleQA）。
- **代码开源**：https://github.com/FloXXXt/DualStake。
- **基础模型**：Qwen2.5-7B、Qwen2.5-7B-Instruct、Qwen3-4B（公开权重）。
- **关键超参**：α=0.25，β=0.25，m⁺=0.9，m⁻=0.1，每样本 5 rollout，E5 encoder + Wikipedia 2018 索引 top-3。
- **训练框架**：GRPO，基于 Search-R1 流程。
