---
title: "APEX-Distillation-of-Agent-Procedural-Experience-for-Adaptiv"
source: https://arxiv.org/pdf/2609.02253v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 20:59:14"
---

# 论文速读：APEX-Distillation-of-Agent-Procedural-Experience-for-Adaptiv

## 一句话总结
论文提出 APEX 框架，将深度研究智能体的交互历史组织为实例级轨迹记忆与类别级程序技能，通过三阶段交替 GRPO 训练与技能引导的测试时强化学习（TTRL），在无真实标签条件下实现执行-蒸馏-规划的闭环自主进化，在 7 个基准上达到 SOTA。

## 研究问题与动机
- 深度研究智能体需通过多轮工具交互与长程推理回答问题，但缺乏持续从历史经验中学习改进的机制，容易重复过往错误而非积累成功经验。
- 现有记忆方法存在两类缺陷：情景记忆直接检索完整轨迹会导致上下文冗长、任务特异性过强且噪声多；过程记忆虽抽象出技能，但通常由固定提示词生成，与下游策略优化脱耦，无法自适应指导推理。
- 经验驱动的改进应构建成闭环而非单向复用：长期交互经验需蒸馏为结构化技能，技能生成应由下游强化学习信号优化，且蒸馏后的技能应作为过程先验支持测试时在线策略适配，形成“更好轨迹→更强技能→更好规划”的正反馈。

## 核心贡献（创新点）
- **分层经验利用架构**：提出实例级轨迹记忆与类别级程序技能的层级结构，通过 Executor-Distiller-Planner 闭环实现经验到动作的转化，与仅做扁平记忆检索的方法本质不同。
- **三阶段交替 GRPO 训练范式**：对三个模块分别设计组合奖励并进行交替强化学习优化，使技能蒸馏成为奖励引导的学习过程而非固定提示生成，解决了跨模块信用分配不稳定问题。
- **技能引导的测试时自适应（Skill-guided TTRL）**：在推理阶段利用无真实标签的多Judge LLM-as-Judge 构建伪奖励，结合技能置信度加权的对齐正则进行在线 Planner 更新，形成无监督闭环进化，与依赖多数投票或需人工标注的测试时适应方法本质不同。
- **上下文高效验证与广泛基准 SOTA**：在 7 个图文/文本基准平均 64.4%，超越 GPT-5.4 14.7 分及最强记忆基线 MIA 3.0 分；仅用技能变体仍超 MIA 1.5 分，且较 MIA 减少 45.6% 记忆 token 注入与 36.8% 总 token 消耗。

## 方法详解
- **分层经验记忆组织**：实例级记忆 $m_i=(q_i, \xi_i, y_i, w_i)$ 按任务类别 $c$ 与模态 $d$ 分桶存储；类别级技能 $s_{c,d}=(P_{c,d}, F_{c,d}, w_{c,d}, n_{c,d}, \deg_{c,d})$ 由 Distiller 将同桶记忆综合为包含推荐步骤与常见失败模式的动态文档。检索时采用双通道相似度（问题嵌入+上下文嵌入）结合历史复用胜率 $w_i$ 进行 Top-K 排序，仅当技能置信度 $\gamma_{c,d}$ 超过阈值时才注入 Planner。
- **三阶段交替 GRPO 优化**：
  - **Executor 训练**：冻结其他模块，基于计划生成多轮工具轨迹。奖励 $R^{\mathrm{exec}} = \lambda_1 R_{\mathrm{acc}} + \lambda_2 R_{\mathrm{tool}} + \lambda_3 R_{\mathrm{fmt}}$，综合答案正确性、各执行段工具调用有效性与格式合规性。
  - **Distiller 训练**：基于当前记忆桶与旧技能输出操作类型（synthesize/refine/create/skip）及新技能。奖励 $R^{\mathrm{dist}} = \mu_1 R_{\mathrm{quality}} + \mu_2 R_{\mathrm{fmt}} + \mu_3 R_{\mathrm{evolve}}$，由 LLM Judge 评估技能结构化质量与演化逻辑合理性。
  - **Planner 训练**：遵循 plan-execute-evaluate-replan 循环，预测是否重规划。奖励 $R^{\mathrm{plan}} = \alpha_1 R_{\mathrm{acc}}(\hat{y}_2) + \alpha_2 R_{\mathrm{acc}}(\hat{y}_1) + \alpha_3 R_{\mathrm{fmt}} + \alpha_4 R_{\mathrm{dec}}$，鼓励首轮正确时停止、首轮失败时触发针对性重规划。
- **技能引导的测试时适应（TTRL）**：推理时无真实标签，采用三人多Judge机制计算无标签奖励 $R_{\mathrm{nogt}}$（涵盖推理一致性、证据支持度、结果完整性）。引入技能对齐奖励 $R_{\mathrm{align}}$ 防止策略漂移，最终奖励 $R_{\mathrm{final}} = (1-\lambda_s)R_{\mathrm{nogt}} + \lambda_s R_{\mathrm{align}}$，权重 $\lambda_s$ 由技能胜率与证据充分度动态调节。高置信度技能可触发 Skill-Gated Update 跳过参数更新；每批次结束后将轨迹合并入记忆库并迭代更新技能。

## 实验与结果
- **数据集与基线**：7 个基准（FVQA-test, SimpleVQA, LiveVQA, InfoSeek, MMSearch, HotpotQA, 2WikiMultiHopQA）；对比覆盖原始大模型（GPT-5.4, Gemini
