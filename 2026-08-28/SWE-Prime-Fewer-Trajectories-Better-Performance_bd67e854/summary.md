---
title: "SWE-Prime-Fewer-Trajectories-Better-Performance"
source: https://arxiv.org/pdf/2608.27449v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:29:47"
field: "代码智能体训练数据选择"
keywords: ["software agent", "SFT data selection", "trajectory quality", "selective loss", "SWE-bench"]
innovations: ["多粒度两阶段轨迹与语义片段选择框架", "五维轨迹质量评估与聚类代表性采样", "选择性SFT损失仅在高质量片段上计算梯度"]
benchmarks: ["SWE-Bench Verified", "SWE-Bench Pro"]
---

# 论文速读：SWE-Prime: Fewer Trajectories, Better Performance

## 一句话总结
提出 SWE-Prime，一种多粒度两阶段 SFT 数据选择方法，通过轨迹级别筛选（过程质量、结果质量、数据代表性）和语义片段级别选择（贡献度、可学习性、行为风险），从成功轨迹中筛选高质量监督信号；仅用 10% 轨迹子集训练即在 SWE-Bench Verified 和 Pro 上分别获得最高 24.2% 和 12.2% 的相对性能提升，优于全量轨迹训练。

## 研究问题与动机
- 现有编码 Agent SFT 工作（如 SWE-Gym、SWE-smith、R2E-Gym）仅基于任务执行结果筛选"成功轨迹"，但成功不等于高质量监督：成功轨迹可能包含无效、冗余或有风险的行为步骤。
- 直接对全量成功轨迹做 SFT 会引入噪声监督，诱导模型模仿不期望的解题行为（如 Git hacking 捷径、连续冗余调用、过度修改文件）。
- 现有过滤机制（如 SWE-Lego 排除失败工具调用）仍不够细粒度，无法识别轨迹内部低价值语义片段。
- 研究空白：如何在成功轨迹池中进行**过程质量 + 结果质量 + 代表性**联合评估，并进一步在轨迹内做**语义片段级别**的价值筛选。

## 核心贡献（创新点）
- **多粒度两阶段数据选择框架**：先从轨迹维度筛选高质量且具代表性的子集，再从语义片段维度识别高价值行为，本质区别在于将 SFT 监督目标从"任务成功"细化到"过程+结果+代表性"三重评估。
- **轨迹级五维质量评估**：提出工作流 grounding（observe-edit-verify）、工具调用成功率、连续冗余检测、Git hacking 检测、结果最小化评分五维联合打分，首次将行为合法性与结果精炼度纳入轨迹选择。
- **语义片段级选择性 SFT 损失**：定义语义片段为行为连贯单元，用 LLM 评估其对最终修复的贡献、局部可学习性和行为风险，训练时仅选中片段的 assistant tokens 参与 loss，其余保留为上下文，而非传统全序列监督。
- **聚类增强的代表性采样**：使用 Qwen3-Embedding-8B + HDBSCAN 对 issue 描述进行语义聚类，在各簇内按轨迹分排序选择，避免纯质量排序导致的问题类型集中。
- **实证验证"少即是多"**：仅用 10% 轨迹子集训练，在三种 base model（GLM-4.7-Flash、Qwen3-30B-A3B-Instruct、Qwen3-Coder-30B-A3B-Instruct）和两个 benchmark（SWE-Bench Verified、Pro）上均显著优于全量成功轨迹 SFT。

## 方法详解
- **Stage 1 轨迹级筛选**：
  - 过程质量四信号：
    - **工作流 grounding** $S_{\text{workflow}}$：轨迹在执行前检查仓库、最终修改后通过测试/校验则得 1，否则 0。
    - **工具可靠性** $S_{\text{tool}}$：计算工具调用成功率，取在整个轨迹池中的百分位排名。
    - **连续冗余控制** $S_{\text{redundancy}}$：相邻步骤工具名和参数完全相同时置 0，否则 1。
    - **Git hacking 检测** $S_{\text{git}}$：若 agent 在修改代码前通过 Git 历史操作获取了参考 patch 内容，置 0，否则 1。
  - 结果质量评分：对比生成 patch 与 reference patch 的文件数和变更行数，定义 $S_{\text{file}}=\min(F_{\text{gold}}/F_{\text{model}}, 1)$，$S_{\text{line}}=\min(L_{\text{gold}}/L_{\text{model}}, 1)$，$S_{\text{result}}=S_{\text{file}}\times S_{\text{line}}$，乘法惩罚只有一方面聚焦的 patch。
  - 轨迹总分：$S_{\text{traj}}=\frac{1}{5}(S_{\text{workflow}}+S_{\text{tool}}+S_{\text{redundancy}}+S_{\text{git}}+S_{\text{result}})$。
  - 代表性采样：用 Qwen3-Embedding-8B 编码 issue 描述，HDBSCAN 聚类，各簇内按 $S_{\text{traj}}$ 排序，在预算内选取高分候选轨迹。
- **Stage 2 片段级筛选**：
  - **语义分块**：将连续步骤按共同意图分组为语义片段；采用边界感知滑动窗口处理（保留当前窗口末尾不完整的片段进入下一窗口重新评估），避免固定窗口割裂行为边界；窗口间传递任务目标、已建立事实和已识别风险的摘要。
  - **片段评分**：LLM 基于 issue 描述、轨迹上下文和 reference patch，从三个维度打分（0-10）：
    - 对最终修复的贡献（推进定位/上下文收集/修改/验证/错误恢复）。
    - 局部可学习性（是否提供清晰 evidence-action-feedback 模式）。
    - 行为风险（无关探索、无信息失败、不安全操作、硬编码解）。
  - **选择性 SFT 损失**：对片段 $s_{i,j}$ 设掩码 $m_{i,j}=\mathbb{I}[q_{i,j}\ge\delta]$，loss 仅作用于选中片段的 assistant tokens：
    $$\mathcal{L}_{\text{SFT}}=-\frac{1}{N_{\text{sel}}}\sum_i\sum_j m_{i,j}\sum_{t\in A(s_{i,j})}\log p_\theta(y_{i,t}\mid y_{i,<t})$$
    其中 $N_{\text{sel}}$ 为选中片段中 assistant token 总数，用于归一化。未被选中片段的 token 保留在上下文 $y_{i,<t}$ 中但不参与梯度更新。

## 实验与结果
- **训练数据**：SWE-rebench OpenHands Trajectories 中 32,161 条成功解决的轨迹（源自 Qwen3-Coder-480B-A35B-Instruct + OpenHands 在 SWE-rebench issues 上生成）。
- **模型**：GLM-4.7-Flash（30B）、Qwen3-30B-A3B-Instruct-2507（30.5B/3.3B）、Qwen3-Coder-30B-A3B-Instruct（30.5B/3.3B）。
- **评测基准**：SWE-Bench Verified（500 人工验证任务）、SWE-Bench Pro（731 任务公开 split）。
- **基线**：Raw Model、Resolved-Trajectory SFT（全量 32,161 条）、Random-10% SFT。
- **主要结果**：
  - SWE-Prime（10% 轨迹）在 SWE-Bench Verified 上相对 Raw Model 提升 18.8%–57.1%，在 Pro 上提升 17.6%–140.2%。
  - 相对全量 Resolved-Trajectory SFT，SWE-Prime 在 Verified 上最高相对提升 24.2%，在 Pro 上最高相对提升 12.2%。
  - Segment-level 选择贡献：以 Qwen3-Coder-30B 为例，去掉 Stage 2 后 Verified 从 53.2% 降至 50.2%，Pro 从 34.75% 降至 30.92%。
  - 跨语言泛化：在 Python/JS/TS/Go 子集上均取得最好结果。
- **超参敏感性**：轨迹保留率 10% 最佳（5%→46.2%，10%→50.2%，20%→49.6%，30%→49.2%）；片段分数阈值 $\delta=7$ 最佳（5→46.8%，7→53.2%，8→49.6%，9→47.0%）。选定配置在三种模型和两个基准上保持一致有效。
- **行为分析**：SWE-Prime 训练的模型相较 Resolved-Trajectory SFT 冗余降低最多 0.188、observe-before-edit 比率提升最多 9.6 个百分点、工具成功率提升最多 0.094，平均交互轮数减少 4.1–12.7 轮，说明性能增益伴随更可靠、更结构化的问题解决行为。

## 相关工作脉络
- **SWE-Gym / SWE-smith / R2E-Gym / SWE-Factory**：基于执行验证筛选成功轨迹做 SFT；本文定位差异：不仅看结果成功，还评估过程质量、结果精炼度和代表性，避免噪声监督。
- **SWE-Lego**：剔除工具调用失败的步骤；本文扩展：进一步识别无明确错误但低价值的片段（如冗余探索、不可学习行为）。
- **ATLAS / STECa / AgentPro**：关注 step-level 过程监督或自动过程提炼；本文定位差异：提出语义片段粒度（multi-step coherent unit）和 selective loss，兼顾上下文保持与监督聚焦。
- **SWE-Master**：通过后训练释放 SWE agent 潜力；本文定位差异：聚焦 SFT 数据选择阶段的质量控制，而非推理/规划层改进。
- **AlphaGenius / SemDeDup**：通用 LLM 数据去重与筛选思路；本文迁移到软件工程 agent 轨迹，引入领域特有信号（Git hacking、work flow grounding）。
- **SWE-rebench / SWE-Bench Pro**：评测基准相关；本文利用其公开轨迹数据做选择实验，验证"少而精"数据策略的有效性。

## 局限性与未来方向
- 轨迹级评分依赖启发式规则（如 workflow、Git hacking 检测），可能漏检复杂作弊或高隐蔽性低质量行为。
- 片段级 LLM 评估需调用外部模型，增加数据选择阶段的计算开销；阈值 $\delta$ 和保留比 10% 虽经敏感性验证，但不同轨迹池可能需要调整。
- 实验集中在 SWE-rebench 生成的成功轨迹，未涵盖失败轨迹或混合质量轨迹的利用潜力。
- 未探索多阶段迭代筛选（如加入 DPO/RL 阶段）与选择性 SFT 的结合。
- 代码/权重开源状态论文未明确声明，限制完全复现。

## 研究启发与可借鉴点
- **选择性 SFT 损失设计**：保留全部 token 作为上下文但仅对被选片段计算 loss，可在长 horizon agent 轨迹中直接复用，避免噪声监督同时维持上下文连贯性。
- **语义片段边界感知分块**：滑动窗口+尾部片段延续+摘要传递的机制，适用于任何需要行为连贯性评估的长轨迹分析任务。
- **五维轨迹质量信号**：workflow grounding、工具可靠性、连续冗余、Git hacking、结果最小化，可作为软件工程 agent 数据筛选的通用指标体系。
- **聚类代表性采样**：在质量排序基础上叠加 HDBSCAN 聚类分层选择，避免单一维度优化导致的分布偏移，可推广至其他 agent 训练数据选择。
- **行为-性能关联分析**：通过冗余、observe-before-edit、工具成功率、交互轮数等多维度行为指标解释性能增益，为后续 ablation 和行为诊断提供参考范式。

## 关键术语表
- **SWE-Prime**：本文提出的多粒度两阶段 SFT 数据选择框架，从轨迹和语义片段两级筛选高质量监督。
- **Semantic Segment（语义片段）**：由连续且意图一致的步骤组成的行为单元，用于细粒度质量评估。
- **Selectve SFT Loss**：仅对被选中语义片段的 assistant tokens 计算交叉熵损失，其余 token 仅作上下文。
- **Git Hacking**：Agent 通过仓库历史/版本记录获取目标修复信息从而走捷径的行为，被视为污染监督的信号。
- **Observe-Edit-Verify Workflow**：先观察/检查仓库、再编辑代码、最后通过测试或校验验证的规范问题解决流程。
- **HDBSCAN**：层次密度聚类算法，本文用于对 issue 描述嵌入进行语义聚类以保障数据代表性。
- **Resolved-Trajectory SFT**： baseline 方法，对所有执行验证成功的轨迹直接进行 SFT，不做额外质量过滤。
- **Relative Improvement**：相对于 Raw Model 的性能提升百分比，用于衡量 SFT 训练带来的增益。

## 可复现要素
- **数据集**：SWE-rebench OpenHands Trajectories（67,074 条轨迹，其中 32,161 条成功解决），来源 Nebius 发布；论文未明确声明是否另行开源筛选后的子集。
- **代码/权重**：论文未明确声明开源状态。
- **关键超参**：轨迹保留率 10%、片段评分阈值 $\delta=7$、最大序列长度 131,072、batch size 32、AdamW、余弦学习率调度、峰值学习率 $4\times10^{-6}$、评估时最多 100 轮交互。
