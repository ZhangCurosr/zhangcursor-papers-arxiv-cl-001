---
title: "APEX-Distillation-of-Agent-Procedural-Experience-for-Adaptiv"
source: https://arxiv.org/pdf/2609.02253v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:43:50"
field: "Agent 记忆与持续学习"
keywords: ["deep research agents", "procedural memory distillation", "test-time reinforcement learning", "hierarchical experience", "GRPO", "skill-guided adaptation"]
innovations: ["层次化经验记忆将实例级轨迹与类别级技能解耦并组织于类别-模态 bucket", "三阶段交替 GRPO 使技能蒸馏成为 reward-guided 过程而非固定 prompt 生成", "Skill-guided TTRL 利用技能置信度自适应加权无监督奖励与对齐正则实现在线适应"]
benchmarks: ["FVQA-test", "HotpotQA", "2WikiMultiHopQA", "SimpleVQA", "LiveVQA", "InfoSeek", "MMSearch"]
---

# 论文速读：APEX-Distillation-of-Agent-Procedural-Experience-for-Adaptiv

## 一句话总结
论文提出 APEX，一种层次化经验利用框架，将深度研究智能体的交互历史组织为实例级轨迹记忆与类别级程序技能，通过 Executor-Distiller-Planner 闭环与三阶段交替 GRPO 训练实现 reward-guided 技能蒸馏，并在测试时利用技能引导的 TTRL 进行无 ground-truth 的在线自适应，最终在 7 个基准上取得 SOTA 性能（平均 64.4%），超越 GPT-5.4 达 14.7 分。

## 研究问题与动机
1. **深度研究智能体（DRA）需要长期经验复用**：复杂长程问答涉及多步工具编排、迭代检索与多源综合，无经验复用会导致重复犯错。
2. **现有经验利用方法的不足**：
   - 情景记忆方法（如 Episodic Memory）检索到的轨迹冗长且任务特定，增加决策负担；
   - 反思记忆方法生成的摘要通常由固定 prompt 生成，未针对下游规划与执行效用优化；
   - 程序记忆方法将技能视为外部知识产物，而非直接优化 agent policy 的自适应先验。
3. **缺乏闭环经验驱动改进机制**：现有工作多为单向"存储-检索-复用"流程，未形成"执行→蒸馏→规划→适应"的闭环反馈。
4. **测试时无法利用无标注数据自我改进**：推理阶段缺乏 ground-truth 监督，难以在未知查询上实现持续优化。

## 核心贡献（创新点）
1. **层次化经验记忆架构**：将长期轨迹分为实例级（instance-level）与类别级（category-level）双层记忆，保留具体求解痕迹的同时抽象出可复用程序知识；与现有扁平化检索方法本质不同，后者仅做轨迹匹配而不做跨任务抽象。
2. **Executor-Distiller-Planner 经验到行动闭环**：三者通过 GRPO 交替训练，使技能蒸馏成为 reward-guided 过程而非固定 prompt 生成；区别于 ExpeL/Memento 等仅靠静态记忆增强的方案，本文强调跨模块信用分配与技能演化。
3. **Skill-guided Test-time Reinforcement Learning（TTRL）**：利用蒸馏技能作为程序先验对 Planner 进行在线适应，配合 no-ground-truth 多评委奖励与 skill-alignment 正则防止策略漂移；与已有 TTRL 依赖多数投票不同，本文引入技能对齐作为正则项。
4. **轻量上下文高效利用**：Skill-only 变体在减少注入记忆 token 45.6% 的同时仍优于最强基线 MIA，证明层次化抽象比扁平检索更节省上下文。

## 方法详解
1. **层次化经验记忆**：
   - 实例级记忆 $m_i = (q_i, \xi_i, y_i, w_i)$，按任务类别 $c$ 与输入模态 $d$ 组织成 $\mathcal{M}_{c,d}$，包含 workflow 摘要、正/负样本标签与重用胜率。
   - 类别级技能 $s_{c,d} = (P_{c,d}, F_{c,d}, w_{c,d}, n_{c,d}, \deg_{c,d})$，由 Distiller 对每个 bucket 内记忆进行 synthesize/refine/create/skip 操作进化。
   - 检索采用双通道相似度 $\sin(q,m_i) = \beta \cos(\phi_q(q),\phi_q(q_i)) + (1-\beta)\cos(\phi_c(q),\phi_c(m_i))$，再与历史胜率 $w_i$ 加权融合，分别 Top-K 检索成功与失败案例。

2. **三阶段交替 GRPO 训练**：
   - **Stage 1 Executor**：冻结 Distiller/Planner，优化 $\pi_\theta^{\mathrm{exec}}$，奖励 $R^{\mathrm{exec}} = 0.7 R_{\mathrm{acc}} + 0.2 R_{\mathrm{tool}} + 0.1 R_{\mathrm{fmt}}$，鼓励正确工具调用与格式合规。
   - **Stage 2 Distiller**：冻结 Executor/Planner，优化 $\pi_\phi^{\mathrm{dist}}$，奖励 $R^{\mathrm{dist}} = 0.5 R_{\mathrm{quality}} + 0.1 R_{\mathrm{fmt}} + 0.4 R_{\mathrm{evolve}}$，评估技能质量与操作合理性。
   - **Stage 3 Planner**：冻结前两者，在 plan-execute-evaluate-replan 循环中优化 $\pi_\psi^{\mathrm{plan}}$，奖励 $R^{\mathrm{plan}} = 0.7 R_{\mathrm{acc}}(\hat{y}_2) + 0.2 R_{\mathrm{acc}}(\hat{y}_1) + 0.05 R_{\mathrm{fmt}} + 0.05 R_{\mathrm{dec}}$，鼓励正确重规划决策。

3. **Skill-guided TTRL**：
   - 测试时无 ground-truth，采用三评委 LLM-as-Judge：$E_1$ 评推理一致性、$E_2$ 评信息溯源忠实度、$E_3$ 评结果完整性，经仲裁者聚合得 $R_{\mathrm{nogt}} = 0.9\mathcal{A} + 0.1 R_{\mathrm{fmt}}$。
   - 技能对齐奖励 $R_{\mathrm{align}} = \mathcal{I}_{\mathrm{align}}(p, P_{c,d}, F_{c,d})$ 判断计划是否遵循推荐步骤并避开已知陷阱。
   - 最终奖励 $R_{\mathrm{final}} = (1-\lambda_s) R_{\mathrm{nogt}} + \lambda_s R_{\mathrm{align}}$，其中 $\lambda_s = \lambda_{\mathrm{base}} \cdot w_{c,d} \cdot \min(n_{c,d}/N_{\mathrm{threshold}}, 1)$ 随技能置信度自适应。
   - 仅更新 Planner 参数 $\psi$，每批结束后更新记忆库并触发 Distiller 演化技能，形成闭环。

## 实验与结果
- **基准**：FVQA-test、SimpleVQA、LiveVQA、InfoSeek、MMSearch（图像-文本）、HotpotQA、2WikiMultiHopQA（纯文本），共 7 个。
- **最强结果**：APEX 平均 64.4%，超越 GPT-5.4（49.7）+14.7 分，超越最强记忆增强基线 MIA（61.4）+3.0 分；在 HotpotQA 上达 75.2%，在 2Wiki 上达 67.8%。
- **相对于 backbone 的提升**：以 Qwen2.5-VL-7B（17.6）为起点，APEX 提升 46.8 分绝对值，证明结构化技能可弥补模型规模差距。
- **Skill-only 变体**：去除实例级记忆后平均 67.7（4 数据集均值），仍超 MIA 1.5 分；相比完整 APEX 仅降 1.8 分。
- **上下文效率**：相比 MIA，APEX (Skill only) 精度 +1.7 分，注入 memory token 减少 45.6%，总 token 消耗降低 36.8%。
- **泛化性**：替换为闭源 Executor（GPT/Gemini 等）无需额外训练，APEX 相对 ReAct 在 HotpotQA 提升 11.0%，LiveVQA 提升 7.0%。
- **消融**：去除经验利用（w/o Both）下降 10.1 分；去除 Executor 训练（w/o Exec.）下降 8.4 分；去除 TTRL 下降 4.2 分；去除迭代蒸馏（w/o Iter.）下降 2.5 分。

## 相关工作脉络
1. **MIA（Memory Intelligence Agent）**：同样采用 Manager-Planner-Executor 架构与实例级记忆，但仅做扁平检索，缺乏类别级技能抽象与 reward-guided 蒸馏；APEX 在此基础上引入 Distiller 模块与 skill-alignment 正则。
2. **ExpeL**：LLM agents 作为经验学习者，通过反射记忆压缩轨迹，但摘要由固定 prompt 生成，未与下游 policy 联合优化；APEX 将技能蒸馏纳入 GRPO 训练。
3. **Memento / Memento-Skills**：微调 LLM agent 而不调参 LLM，侧重记忆存取策略学习；APEX 强调跨模块交替 RL 与测试时在线适应。
4. **SkillRL**：递归 skill-augmented RL 演进 agent，但技能作为外部知识库而非自适应先验；APEX 将技能直接作为 TTRL 的正则化锚点。
5. **Mem0 / A-Mem / Lightmem**：面向生产级 agent 的记忆系统，聚焦存储压缩与检索；APEX 的独特性在于将记忆抽象为可进化的程序技能并接入 RL 闭环。
6. **DeepResearcher / MMSearch-R1 / Deepeyes2**：强化工具搜索能力的搜索 agent，无跨任务记忆；APEX 在工具搜索基础上叠加层次化经验复用。

## 局限性与未来方向
1. **规模扩展未验证**：当前基于 7B/8B 小开源模型，未探索在更强 backbone 上的缩放行为。
2. **三阶段交替训练的非端到端性**：顺序优化 Executor-Distiller-Planner 可能未充分挖掘模块间协同，未来可探索端到端联合训练。
3. **TTRL 无监督限制**：当前 skill-guided TTRL 完全无 ground-truth 信号，引入少量监督信号有望进一步提升技能质量与闭环速度。
4. **工具生态单一**：目前仅使用离线文本搜索与图像搜索，尚未扩展到更广泛的工具环境。
5. **跨智能体技能迁移未探索**：当前技能在单 agent 内部演化，未来可扩展至多 agent 间的 skill transfer。

## 研究启发与可借鉴点
1. **Reward-guided 技能蒸馏范式**：将 Distiller 的训练目标与下游 GRPO advantage 耦合，而非固定 prompt 生成，可迁移至任何需"从轨迹中提取程序性知识"的场景。
2. **Skill-alignment 正则化在线适应**：用历史技能置信度动态加权 no-ground-truth 奖励与对齐奖励，可有效防止 TTRL 策略漂移；适用于任何无标注在线学习 setting。
3. **层次化记忆 + 双通道检索设计**：实例级（具体案例）与类别级（抽象技能）解耦，配合双通道语义+上下文相似度，可作为通用经验管理模块被其他 agent 框架复用。
4. **三模块交替 GRPO 的训练稳定性**：在 Executor/Distiller/Planner 之间轮流固定其余模块优化，避免了联合训练的 credit assignment 难题，该策略可推广至多模块 RL 系统。
5. **上下文效率权衡分析**：通过 Token-efficiency 曲线（精度 vs 注入 token vs 总成本）可视化对比不同 memory-augmented 方法，为后续工作提供可直接复用的评估维度。

## 关键术语表
**Deep Research Agents (DRAs)**：_augmented with external tools 以多轮推理回答复杂长程问题的智能体系统_
**Instance-level Trajectory Memory**：_保存具体任务求解轨迹（含 workflow 摘要、正负标签、重用胜率）的细粒度记忆_
**Category-level Skill**：_从同类任务记忆中蒸馏出的抽象程序性知识文档，包含推荐步骤、常见陷阱与置信度_
**Three-stage Alternating GRPO**：_Executor、Distiller、Planner 轮流在其余模块冻结条件下分别进行 Group Relative Policy Optimization 的训练范式_
**Skill-guided TTRL**：_测试时利用蒸馏技能作为正则先验，结合 no-ground-truth 多评委奖励对 Planner 进行在线 GRPO 适应的机制_
**No-ground-truth Reward**：_在推理时无真值可用的场景下，通过三评委 LLM-as-Judge 聚合推理一致性、信息溯源忠实度与结果完整性得出质量信号_
**Skill-alignment Regularization**：_用技能中的推荐步骤 $P_{c,d}$ 与已知陷阱 $F_{c,d}$ 评估生成计划，防止在线适应偏离已验证的程序知识_
**Confidence-weighted Retrieval**：_检索分数同时考虑双通道语义相似度与历史重用胜率（实例级）或胜率与证据数量（技能级），高置信度经验优先被注入_

## 可复现要素
- **数据集**：FVQA-train/test（公开）、HotpotQA（公开）、2WikiMultiHopQA（公开）、SimpleVQA（公开）、LiveVQA（公开版本）、InfoSeek/MMSearch（来自 MMSearch-R1 代码仓库）；全部公开可获取。
- **代码/权重**：代码开源，链接 https://github.com/J-Ding519/APEx；权重未明确说明，需查看仓库。
- **关键超参**：Executor batch=128、G=8、lr=1e-6、epochs=8；Distiller batch=64、G=8、lr=1e-6、epochs=8；Planner batch=48、G=4、lr=1e-6、epochs=5；TTRL lr=2e-7、G=4、batch=8、epochs=1；奖励权重 $\lambda_1=0.7,\lambda_2=0.2,\lambda_3=0.1$；$\mu_1=0.5,\mu_2=0.1,\mu_3=0.4$；$\alpha_1=0.7,\alpha_2=0.2,\alpha_3=0.05,\alpha_4=0.05$；$\lambda_{\mathrm{base}}=0.1,N_{\mathrm{threshold}}=10$；PPO clip $\epsilon=0.2$；KL $\beta=0.0$；Entropy=0.0。
- **训练基础设施**：单节点 8× NVIDIA A100，FSDP + 参数/优化器 offload，Qwen3-32B 作为 LLM Judge（vLLM tensor parallelism=2）。
