---
title: "S-sup-3-sup-Gym-Can-LLMs-Turn-Self-Testing-and-Self-Judging"
source: https://arxiv.org/pdf/2608.31100v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:29:25"
field: "自我改进型 Agent 评估"
keywords: ["Self-Improving Agents", "Self-Judging", "In-Context Learning", "Experience Integration", "Interactive Benchmarks", "LLM Agents"]
innovations: ["统一评估自我测试-自我评判-自我改进三阶段闭环的交互式基准 S³Gym", "在相同协议下系统对比 History ICL、Summary Memory、Parameter Training 三种经验整合路径", "显式测量并分析模型自我评判质量与后续行为改进之间的耦合关系"]
benchmarks: ["S³Gym", "Chess", "Minesweeper", "Nullify", "Tetris", "Snake", "Plants-vs-Zombies", "Trust Evolution"]
---

# 论文速读：S³Gym: Can LLMs Turn Self-Testing and Self-Judging into Self-Improvement?

## 一句话总结
本文提出 S³Gym，一个交互式基准测试，通过"自我测试→自我评判→自我改进"的闭环协议，系统评估 LLM 能否利用自身交互经验持续改进后续行为；实验表明自我改进并非自动或均匀发生，其效果高度依赖任务结构与经验整合路径的选择。

## 研究问题与动机
1. **现有基准将模型视为固定策略**：AgentBench、KORGym 等主流交互式基准主要回答"模型当前有多强"，无法反映模型能否从自身过往交互中提炼可迁移经验并改进未来决策。
2. **经验驱动学习需要三个耦合能力**：自我测试（探索策略并收集诊断证据）、自我评判（评估行为与结果的可复用性）、自我改进（将经验转化为行为变化），三者缺一不可，但现有工作通常只孤立研究其中一环。
3. **不同经验整合路径缺乏统一对比基准**：History ICL 保留完整细节但消耗上下文；Summary Memory 提供紧凑抽象但依赖评判质量；Parameter Training 实现持久内化但可能放大错误评判——尚无统一协议在相同条件下比较三者。
4. **自我评判可靠性可能是改进瓶颈**：不准确的评判会导致有害经验被保留或有效经验被丢弃，而现有工作很少显式测量模型自评与客观环境奖励之间的一致性。

## 核心贡献（创新点）
1. **统一的形式化体验驱动自我改进框架**：将自我改进定义为 Explore → Judge → Consolidate → Update → Evaluate 的五阶段迭代过程，并分离宽松探索与严格 Held-out 评估两个阶段；与已有工作相比，首次在同一协议下同时考察三类经验整合路径。
2. **三条经验整合路径的系统性对比**：在七个文本游戏中平等评估 History ICL、Summary Memory 和 Parameter Training，揭示"总结并非无条件优于原始历史"——任务结构决定最优路径。
3. **显式评估自我评判可靠性**：通过对比模型自评分 s_{x,i} 与环境验证器评分 r_{x,i}，测量事件一致性、NMAE、过度/低估自信，并分析评判质量与后续改进幅度的耦合关系（ρ≈0）。
4. **全面的诊断分析与失败模式刻画**：参数训练在 Trust Evolution 上大幅提升但 PvZ 上出现严重负迁移；自我评判在 Chess 和 Trust 上接近随机水平；总结在可压缩为规则的任务中有效，在依赖精确状态信息的任务中失效。

## 方法详解
**五阶段循环**（公式 1）：$\mathcal{R}_t^{(p)} = \text{Explore} \to \text{Judge} \to \text{Consolidate} \to \text{Update}_p \to \text{Evaluate}$，其中 $p \in \{\text{History, Memory, Training}\}$ 表示整合路径。

**交互协议**：每集（episode）从 $O_{x,0} = \text{Reset}(\text{seed}_x, c_x)$ 开始，每步模型联合输出动作与自评分数 $(a_{x,i}, s_{x,i}) = \pi_\theta(O_{x,i}, H_{x,i}; Z_t^{(p)})$，环境返回可观测反馈 $F_{x,i}$ 和验证器奖励 $r_{x,i}$（后者不对模型暴露）。

**三条经验整合路径**：
- **History ICL**（公式 8-14）：将带分数的探索轨迹直接序列化追加到后续 episode 上下文中，最大 100K tokens。
- **Summary Memory**（公式 9-10, 15）：将轨迹压缩为三类摘要 $(R^{\text{retain}}, R^{\text{avoid}}, D^{\text{next}})$——保留的策略、避免的错误、下一步方向——作为跨 episode 的记忆。
- **Parameter Training**（公式 16-17）：将探索轨迹与自评转换为 SFT 数据，保留高分动作、过滤/修正低分动作，对 Qwen3-8B 进行连续 20 个 checkpoint 的微调。

**评估设计**：每个游戏提供探索分布 $\mathcal{C}^{\text{exp}}_g$（宽松条件：更小状态空间/更弱惩罚/额外试错机会）和评估分布 $\mathcal{C}^{\text{eval}}_g$（严格条件），使用不相交随机种子。衡量指标为 Avg.（全程平均分）、Max.（最高分）、AUC⁺（高于初始基线的正向面积）。

## 实验与结果
**数据集与基线**：七个文本游戏（Chess, Minesweeper, Nullify, Tetris, Snake, PvZ, Trust Evolution），七款商业模型（GPT-4o/4.1/o3-mini, Gemini-2.5-Flash/Pro, GPT-5.5, Gemini-3.5-Flash），以及 Qwen3-8B 用于参数训练实验。

**核心结果**：
- **History ICL 最强表现**：GPT-5.5 在 PvZ 上 AUC⁺ 达 548.499（全场最高），Gemini-3.5-Flash 在 Chess 上 AUC⁺ 达 12.280、在 TrustEvo 上达 200.000。
- **Summary Memory 选择性有效**：Gemini-2.5-Flash 在 Minesweeper 上 AUC⁺ 从 0.000（ICL）提升至 7.794（Summary）；PvZ 从 24.402 提升至 238.501；但 GPT-5.5 的 PvZ AUC⁺ 从 548.499 骤降至 33.219。
- **参数训练（Qwen3-8B）**：Trust Evolution 从 0 升至最高 30，19 个 checkpoint 中 18 个保持正增益；Minesweeper/Nullify/Tetris 无提升；PvZ 从 23 降至 6（严重负迁移）；Snake 间歇性达到 1 后又归零。
- **自我评判可靠性**：PvZ/Snake/Tetris 事件一致性 >0.87，但 Chess 仅 0.496（接近随机）、Trust 仅 0.665；NMAE 在 Chess(0.141) 最低、PvZ(0.882) 最高；评判质量与后续改进的 Spearman 相关系数 ρ≈0。
- **最强结果**：GPT-5.5 在 PvZ History ICL 下 Max 分 48.000、AUC⁺ 548.499；参数训练中 Trust Evolution AUC⁺ 163.5。

## 相关工作脉络
1. **交互式 Agent 基准**（AgentBench [15], AgentBoard [16], TextWorld [5], ALFWorld [27], KORGym [25]）：侧重于静态任务完成能力评估，将模型视为固定策略；S³Gym 将焦点从"当前能力"转向"经验驱动的行为进化"。
2. **经验整合——上下文层**（Reflexion [26], Voyager [30]）：语言代理通过自我反思生成文本反馈指导后续决策；S³Gym 将其形式化为可量化的 History ICL 与 Summary Memory，并加入严格 held-out 评估检验迁移。
3. **经验整合——参数层**（STaR [37], ReST [11], Re-ReST [9]）：通过自生成推理轨迹进行监督微调；S³Gym 引入自我评判信号进行轨迹筛选，并显式测量错误评判导致的负迁移风险。
4. **自适应/自我改进基准**（PostTrainBench [22], SEA-Eval [13], SEAGym [40], PAST-Bench [34], ContinualSkillBench [10]）：多聚焦单一改进 artifact（参数/记忆/技能库）或长期演化结果；S³Gym 首次在相同环境、相同交互预算下统一比较三种整合路径，并显式评估自我评判环节。
5. **LLM 作为评判者的可靠性研究**（LLM-as-a-Judge [41], JudgeBench [28]）：揭示 LLM 评判存在偏差和校准问题；S³Gym 进一步将评判质量与后续行为改进做耦合分析，发现高评判一致性并不保证后续提升。

## 局限性与未来方向
1. **参数训练不稳定且存在负迁移风险**：当前 SFT 实现下，训练可能在部分任务上产生严重退化（如 PvZ 从 23 降至 6），需更好的轨迹筛选与防过拟合机制。
2. **自我评判与后续改进脱钩**：评判质量与改进幅度几乎无关（ρ≈0），说明仅有准确的局部评分不足以保证改进，还需将反馈转化为可执行的策略抽象。
3. **总结的记忆容量瓶颈**：Summary Memory 在依赖精确状态信息的任务（Minesweeper 部分模型、Snake、PvZ）中劣于 History ICL，亟需更精细的信息压缩与检索机制。
4. **当前仅评估了三个整合路径**，未探索混合路径（如先总结再检索原始片段）或元技能演化（如 MetaSkill-Evolve [33]）等更复杂的机制。
5. **实验限于七款文本游戏**，任务类型覆盖有限（虽涵盖规则归纳、约束满足、空间规划等，但未涉及工具使用或代码生成等更复杂场景）。

## 研究启发与可借鉴点
1. **探索-评估分离的设计范式**：将宽松探索与严格 held-out 评估解耦，既保证经验采集的多样性，又防止数据泄露，此协议可直接迁移至团队的其他 agent 进化研究方向。
2. **三重指标互补评估**：Avg.（整体表现）、Max.（突破性能力）、AUC⁺（持续改进趋势）结合使用，比单一指标更能刻画自我改进的动态特性，可作为后续实验设计的参考模板。
3. **自我评判显式测量**：通过对比模型自评 s 与环境真实奖励 r，量化事件一致性、NMAE、过度/低估自信等维度，为团队构建自评判模块提供直接的可复用评估体系。
4. **任务结构决定路径选择的发现**：可压缩为规则的任务适合 Summary Memory，依赖精确状态的任务适合 History ICL——这一结论提示团队在未来构建 agent 系统时应根据任务特性动态选择经验整合策略。
5. **负迁移的诊断价值**：参数训练在 PvZ 上的严重退化为"何时不该微调"提供了警示案例，可启发团队在后续研究中增加轨迹质量阈值检查与选择性更新机制。

## 关键术语表
**Self-Testing**：智能体在宽松环境中主动探索策略、收集诊断证据的行为，对应经验循环的 Explore 阶段。
**Self-Judging**：智能体对自身动作和结果的即时评分 $s_{x,i}$，用于识别有用/有害经验，是经验整合的前提。
**Self-Improvement**：将经过评判的经验转化为未来行为变化的能力，通过 History ICL、Summary Memory 或 Parameter Training 实现。
**AUC⁺**：Positive Area Above Baseline，衡量探索过程中持续改进的累积面积，公式为 $\int_0^{30} \max(0, y(x) - y_0) dx$。
**History ICL**：将带评分的完整交互轨迹序列化后直接追加到后续 episode 上下文中的经验整合方式。
**Summary Memory**：将长轨迹压缩为可复用的策略摘要（保留项/避免项/下一步方向），以紧凑形式注入后续决策。
**Negative Transfer**：参数训练后模型在新配置下的性能低于初始水平的现象，本文在 PvZ 中出现从 23 降至 6 的严重案例。
**NMAE**：Normalized Mean Absolute Error，对自评分数与环境奖励分别做 min-max 归一化后计算平均绝对误差，衡量校准精度。

## 可复现要素
- **数据集**：七个文本游戏（Chess, Minesweeper, Nullify, Tetris, Snake, PvZ, Trust Evolution），项目页面 https://self-developing-agents.github.io/，论文未明确声明代码是否开源。
- **代码/权重**：论文未提及代码或权重开源声明；附录 A 提供了完整的游戏 prompt 模板（`test/S3GYM/envs/*/prompt.py`）。
- **关键超参**：History ICL 输入上限 100,000 tokens；每 response 最多 14,000 new tokens；每 episode 最多 64 步；探索预算 30 个 episode；每 3 个探索 episode 评估一次；评估集每个 checkpoint 3 个严格模式 episode；评估 checkpoint 为 $x \in \{0, 3, 6, \ldots, 30\}$。
