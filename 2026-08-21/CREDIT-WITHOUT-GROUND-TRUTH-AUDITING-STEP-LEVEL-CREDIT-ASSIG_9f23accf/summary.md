---
title: "CREDIT-WITHOUT-GROUND-TRUTH-AUDITING-STEP-LEVEL-CREDIT-ASSIG"
source: https://arxiv.org/pdf/2608.19760v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-02 00:02:41"
---

# 论文速读：CREDIT-WITHOUT-GROUND-TRUTH-AUDITING-STEP-LEVEL-CREDIT-ASSIG

## 一句话总结
本文在 ALFWorld 单智能体工具环境中，通过执行回放构建因果 ground truth，系统审计了现有步骤级信用信号（LLM 裁判评分、结果条件化 log-prob、策略置信度），发现它们在识别“因果关键步骤”上均不优于随机水平；不同信用规则的训练性能差异主要由优化剂量而非信号机制本身解释。

## 研究问题与动机
- **核心问题**：LLM Agent 训练中广泛使用的步骤级信用分配信号，能否真正捕捉对最终结果产生因果贡献的关键决策步骤？
- **正确性≠贡献性**：现有评估将信用信号与“步骤标注正确性”对齐；本文指出正确步骤可能轨迹已确定（零贡献），错误步骤可能开启恢复状态（高贡献），训练循环实际需要的是贡献（contribution）而非正确性。
- **现有方法不足**：隐式信用、裁判评分、结果条件化等机制缺乏因果可验证性，且与策略自身流畅度（fluency）高度共线，易产生虚假相关性。
- **动机**：建立独立于被审计信号的因果 ground truth，并以前置注册协议严格评估各信用规则的实证有效性，为 Agent 信用分配提供可校准的理论标尺。

## 核心贡献（创新点）
1. **提出执行回放因果审计框架**：无需人工标注，通过在轨迹每个决策点重采样替代动作并多次回放至终端，直接计算回放优势作为步骤级因果 ground truth。
2. **揭露主流信用信号的随机水平表现**：LLM 裁判、结果条件化 log-prob、策略置信度在识别因果关键步骤上无一优于随机，且隐式信用实质是策略流畅度（fluency）的回声。
3. **确立“贡献优先于正确性”的训练范式**：证明步骤级信用的目标变量应从标注正确性转向因果贡献度，填补信用分配评估的理论缺口。
4. **揭示训练剂量混淆效应**：不同信用规则 checkpoint 间的性能差异完全可由优化器步数（dose）解释，此前常被误归因为信用机制的有效性。
5. **分离置信度路由的检测与成本功能**：证明低置信度转发路由以机会水平召回关键步骤，实为裁判调用成本削减机制而非因果检测器。

## 方法详解
- **因果 Ground Truth 构建（执行回放）**：在每条轨迹的每个决策点 $t$，冻结当前策略快照，从合法动作空间重采样 $K=4$ 个不同替代动作；每个替代动作回滚至决策点并独立执行至少 3 次至终端，记录终态奖励；回放优势定义为 $A_{\text{replay}}(t) = \text{mean}(\text{outcome}|\text{factual}) - \text{mean}(\text{outcome}|\text{alternative})$，完全独立于待审计信号。
- **审计模型与环境**：主审计模型 Qwen2.5-7B-Instruct，跨族复现使用 Llama-3.1-8B-Instruct；环境为 ALFWorld（支持状态回滚与单智能体工具调用）。
- **预注册审计协议**：阈值、排除规则、裁决顺序提前注册；采用边际匹配洗牌控制（marginal-matched shuffled control）为基准。三维度验证：秩保真度（rank fidelity）、逐步骤符号一致性（per-step sign consistency）、条件化 fluency 后的 partial correlation。
- **七臂预注册训练实验**：涵盖 outcome-only、implicit credit、judge credit 及其洗牌/反转对照、低分辨率 replay-truth 臂；使用共同随机数（CRN），保留 128 任务独立评估。
- **置信度路由规则**：冻结低置信度转发策略，阈值由策略自身置信度的 order statistic 确定；路由比例约 13.1%，每步节省裁判调用约 13.1%（每轨迹 14.0%）。

## 实验与结果
- **数据集与环境**：ALFWorld 单智能体工具环境；128 保留任务用于最终评估。
- **评估基线**：outcome-only、implicit credit、judge credit 及各类洗牌/反转对照臂，与未训练基线（0.422）对比。
- **关键数字与结论**：
  - 因果效应稀疏：仅 **30.5%**（Qwen）/ **38.3%**（Llama）的完整步骤存在可测量因果效应，其余为因果惰性；Llama 排除率（13.1%）显著低于 Qwen（26.8%，区间不重叠）。
  - 秩保真度：implicit credit（Qwen $\beta=0.0193$ [−0.109, 0.081]，Llama $\beta=-0.043$ [−0.125, −0.016]）均不显著；judge credit（Qwen $\beta=0.1142$ [0.027, 0.168]）为最强信号，但 precision-at-pivotal lift 仅 1.000 [0.935, 1.001]，触及机会线。
  - Fluency 混淆：implicit credit 与策略 fluency 中位秩相关 **+0.75**；条件化 fluency 后 credit 与因果增量 partial correlation 为 **−0.004**（p=0.87）；fluency 回归权重约为因果增量的 **2.33×**。
  - 训练性能：七臂均**无可靠超越**基线 0.422；implicit credit 低 2.3pp（p=0.43），低 outcome-only 7.0pp（p=0.91）；臂间 JS 散度（0.0520）为臂内（0.0198）的 **2.6×**（p=0.0001），但控制优化步数跨度（112→8 per round）后 partial Mantel $\rho=+0.078$（p=0.774），差异完全由剂量解释。
  - 路由效果：置信度路由召回关键步骤仅 **11.9%**，与路由比例一致，达**机会水平**。
- **最强结果**：Judge credit 秩保真度 $\beta=0.1142$ 为所有信号中最高，但即使在此最优条件下，precision-at-pivotal 仍未显著高于机会线，整体信用分配能力仍属随机水平。

## 相关工作脉络
- **LLM Agent 信用分配/学习**：传统 RLHF/DAPO 等方法依赖轨迹级奖励或正确性标注；本文指出步骤级信用需转向因果贡献评估，重塑评估标尺。
- **LLM 裁判（Judge LLM）评估**：广泛使用裁判模型进行步骤级打分；本文证明其符号
