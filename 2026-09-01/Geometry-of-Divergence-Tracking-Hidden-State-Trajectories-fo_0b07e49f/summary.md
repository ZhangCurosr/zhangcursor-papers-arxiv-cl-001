---
title: "Geometry-of-Divergence-Tracking-Hidden-State-Trajectories-fo"
source: https://arxiv.org/pdf/2608.30650v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:05:04"
---

# 论文速读：Geometry-of-Divergence-Tracking-Hidden-State-Trajectories-for-Adaptive-Multi-Turn-Reasoning

## 一句话总结
本文提出通过追踪大语言模型隐藏状态轨迹的几何特征（时间曲率 κ 与方差斜率 β），在长多轮交互完成前提前识别推理错误，并以此作为在线控制信号触发自适应长思维链推理；在 τ-Bench 上将任务成功率从 24.1% 提升至 39.6%，同时 token 成本降低 11.2%。

## 研究问题与动机
1. **多轮上下文累积导致表示漂移**：随着交互轮次增加，LLM 对早期任务关键信息的内部表示逐渐模糊，构造性推理与表示漂移难以区分。
2. **现有上下文工程方法未解决目标错位**：规划、反思循环、记忆架构等显式方法仍作用于文本历史层面，未能弥合智能体目标与底层 LLM 逐轮最大化似然目标之间的 mismatch。
3. **固定推理强度引入语义噪声**：当前 Large Reasoning Models (LRMs) 在多轮对话中使用固定推理预算，不必要推理会向自回归历史注入无关反事实，污染已累积约束的内部表示。
4. **缺乏在线拐点检测机制**：需要一种可观测的、能精确定位隐藏状态开始偏离核心目标的时刻的指标，以实现及时干预。

## 核心贡献（创新点）
1. **提出时间曲率与方差斜率两个互补几何信号刻画多轮推理轨迹**，区别于 ICR Probe 等仅分析单轮响应的后验探测方法，首次将轨迹几何用于在线控制。
2. **揭示轨迹可分性的动作依赖性**，通过将 episode 分解为三动作链证明信息输出链（如 Respond×3）携带最强判别结构，而信息获取链判别力有限。
3. **设计无需训练的几何条件触发策略**，阈值通过一维扫描定位且具备单模态与跨任务可迁移性，在 τ-Bench 上取得优于训练型 Learned-HST 策略的性能。
4. **建立几何信号与任务失败前的表征退化之间的因果关联**，证明失败轨迹表现为方向反转（κ 更负）与过早收敛（β 更小），为多轮 agent 的可解释性诊断提供量化依据。

## 方法详解
- **隐藏状态轨迹构建**：在每轮用户输入 $q_t$ 的最后一个 token 位置，从固定中间探针层（Qwen3-14B 第 22 层、Qwen3-32B 第 42 层、Llama-3.1-8B 第 16 层）提取隐藏状态 $h_t \in \mathbb{R}^d$，构成序列 $\mathcal{H} = \{h_0, h_1, \ldots, h_{T-1}\}$；相邻状态差 $\mathbf{v}_t = h_t - h_{t-1}$ 表示该轮带来的表示更新。
- **时间曲率 Temporal Curvature**：$\kappa_t = \frac{\mathbf{v}_t \cdot \mathbf{v}_{t-1}}{\|\mathbf{v}_t\|_2 \|\mathbf{v}_{t-1}\|_2}$，取值 $[-1,1]$；接近 1 表示更新方向一致（建设性累积），接近 -1 表示方向急剧反转（漂移迹象），约为 0 表示正交更新。
- **方差斜率 Variance Slope**：先计算前缀离散度 $V(\lambda) = \frac{1}{\lambda}\sum_{i=0}^{\lambda}\|h_i - \bar{h}^{(\lambda)}\|_2^2$（无偏估计），再对 $\lambda$ 做普通最小二乘回归得到斜率 $\beta_t$；$\beta_t > 0$ 表示表示空间扩散（持续探索），$\beta_t < 0$ 表示收缩（过早收敛）。
- **轮次对齐 Turn Alignment**：所有 $h_t$ 统一取自用户输入末尾，以轮次为索引而非 token 数，消除失败 episode 更长带来的机械性曲率/方差混淆。
- **动作链分解 Action Chain Decomposition**：将助手动作分为 Read / Write / Respond / Transfer 四类，用长度为 3 的滑动窗口提取连续动作链，统计正确与失败 episode 中各类链的频率与几何信号判别力差异。
- **几何触发策略 Geometry-Conditioned Trigger**：在每轮 $t \ge 2$ 判断若 $\kappa_t < \tau_\kappa$ 或 $\beta_t > \tau_\beta$ 则切换至长 CoT 思考模式；阈值通过一维扫描在诊断统计（正确-失败 κ 差距）指导下选取，无需额外训练。

## 实验与结果
- **数据集**：Lost in Conversation 基准的 Math 与 Code 任务（sharded-prompt 框架模拟渐进错误假设）；τ-Bench 的 Airline 与 Retail 任务（带领域 API 工具的多轮对话，默认最多 50 轮）。
- **模型**：Qwen3-14B、Qwen3-32B（均可切换思考模式的 LRM）、Llama-3.1-8B；推理使用 vLLM，8×L40S + 4×H200 GPU。
- **基线**：Fixed policies（Never-thinking / Always-thinking / Warm-up 首两轮思考）、Action-conditioned policies（基于预测动作触发）、Learned-HST（训练分类器联合输入近期动作 one-hot 与 $(\kappa_t,\beta_t)$）。
- **轨迹几何分离结果**：
  - 时间曲率：所有 6 个 task-model 配对中失败组均值均低于成功组，差值（Correct−Incorrect）为 Math +0.047、Code +0.148、Retail(14B) +0.030、Airline(14B) +0.049、Retail(32B) +0.035、Airline(32B) +0.088；Code 正确组中位数 κ = +0.127。
  - 方差斜率：成功 episode 的 β 普遍高于失败 episode，Gap 在 Math/Code 上最大，Airline 两组区分度不显著（14B p=0.30，32B p=0.057）。
- **动作链判别分析**：Read 主导链（如 Read→Read→Read）判别力弱；Respond 主导链（如 Respond→Respond→Respond）在几何空间中呈现最强分割，正确 episode 在 PCA 空间中持续向外扩张，失败 episode 局限于局部。
- **在线控制结果**：
  - 最优几何触发：κ < −0.20 在 Retail+14B 上达 0.397 成功率，与 Learned-HST（0.398）持平且 token 成本更低（101.0k vs 104.4k）；κ < −0.15 在 Retail+32B 上达 0.442，超越 Learned-HST（0.405）约 27%。
  - 全局提升：四组 domain-model 平均成功率从 24.1% 升至 39.6%（+15.5 pp），平均 token 成本从 104.8k 降至 93.0k（−11.2%），最大轮次终止数显著下降（如 Retail 14B 从 75 降至 34）。
- **阈值可迁移性**：最优 $\tau_\kappa$ 随正确-失败 κ 差距单调偏移（Retail 14B +0.030、Airline 14B +0.049、Retail 32B +0.035、Airline 32B +0.088），可通过诊断统计预测，无需每部署重新扫描。

## 相关工作脉络
1. **ICR Probe (Zhang et al., 2025)**：追踪跨层隐藏状态动态用于幻觉检测，仅分析单轮响应；本文扩展至多轮轨迹并用于在线控制。
2. **Semantic Entropy (Farquhar et al., 2024; Kossen et al., 2024)**：基于多次采样近似不确定性，单生成时可从隐藏状态近似；跨任务泛化有限，本文信号具有动作依赖性且几何阈值可迁移。
3. **AdaptThink (Zhang et al., 2025a) 与 Thinkless (Fang et al., 2026)**：用强化学习在输入端选择思考/不思考模式，主要在单轮设置；本文基于累积内部状态触发，适应多轮动态。
4. **AdaCoT (Lou et al., 2025)**：Pareto-optimal chain-of-thought 触发，依赖输入问题；本文几何触发无需训练，阈值单模态易搜索。
5. **Lost in Conversation (Laban et al., 2025)**：揭示多轮对话中早期错误假设导致 39% 性能退化；本文在其基准上建立几何信号分析框架。
6. **τ-Bench / τ²-Bench (Yao et al., 2024; Barres et al., 2025)**：工具驱动多轮交互基准；本文在其上验证几何触发的在线控制效果与 token 效率权衡。

## 局限性与未来方向
- 方差斜率在开放度高、动作空间大的任务（如 Airline）上区分度下降甚至不显著，可能受环境噪声干扰。
- 时间曲率在信息获取链（Read 主导）中判别力有限，信号强度高度依赖动作组合类型。
- 探针层索引固定为单点（不同模型不同层），未探索多层联合或自适应层选择策略。
- 阈值虽可通过诊断统计指导，但仍需针对特定模型分布校准，跨模型零调优迁移性未验证。
- 论文自述未来方向：将轨迹几何用于多轮对抗攻击（jailbreak）内部检测，以及与表示漂移校正结合构建持续学习的 agent 系统。

## 研究启发与可借鉴点
1. **轻量级在线监测范式**：时间曲率与方差斜率仅需提取固定层隐藏状态并做简单向量运算，可无缝嵌入任何 transformer-based agent 框架，无需额外参数。
2. **动作链分解分析法**：将 episode 按滑动窗口拆分为连续动作组合，再评估各模式下的信号判别力，为理解“何种行为模式最易引发表示漂移”提供结构化诊断工具。
3. **单模态阈值搜索**：几何触发阈值随任务差距单调变化，一维扫描即可定位最优值，避免高维超参搜索，适合部署在资源受限环境。
4. **轨迹对齐设计**：以轮次而非 token 数为索引消除长度混淆，这一对齐原则对任何基于序列长度的表征分析均有参考价值。
5. **成本-成功率权衡可视化**：将 token 成本与最大轮次终止数作为副指标纳入比较，为 agent 系统的工程落地提供清晰的效率-效果前沿参考。

## 关键术语表
- **Temporal Curvature（时间曲率）**：连续两轮隐藏状态更新向量的余弦相似度，衡量语义漂移的方向一致性，负值指示方向反转。
- **Variance Slope（方差斜率）**：隐藏状态轨迹离散度随轮次增长的 OLS 回归斜率，正值表示探索空间扩张，负值表示过早收敛。
- **Hidden-State Trajectory（隐藏状态轨迹）**：多轮交互中每轮指定探针层最终 token 隐藏状态的有序序列，编码累积上下文的内部表示演化。
- **Action Chain（动作链）**：由连续三轮助手动作（Read / Write / Respond / Transfer）构成的滑动窗口模式，用于分析几何信号判别力的动作依赖性。
- **Geometry-Conditioned Trigger（几何条件触发）**：基于 κ 或 β 阈值动态决定是否启用长 CoT 思考模式的在线控制策略，无需额外训练。
- **τ-Bench**：评估工具型多轮对话智能体的基准，包含 Airline 与 Retail 两个模拟域，提供 ground-truth 答案与 token 成本记录。
- **Probe Layer（探针层）**：用于提取隐藏状态的固定中间层，本文取 Qwen3-14B 第 22 层、Qwen3-32B 第 42 层、Llama-3.1-8B 第 16 层。
- **Representation Drift（表示漂移）**：多轮上下文累积导致 LLM 内部任务相关表示逐渐模糊或偏离原目标的现象。

## 可复现要素
- **数据集**：Lost in Conversation（Math、Code）与 τ-Bench（Airline、Retail）；论文未声明是否重新发布，但引用来源应可公开访问。
- **代码/权重**：论文未提及代码开源情况；模型权重 Qwen3-14B / Qwen3-32B / Llama-3.1-8B 可从官方渠道获取。
- **关键超参**：探针层索引（22 / 42 / 16）、触发阈值 $\tau_\kappa$（一维扫描范围约 −0.15 至 −0.25）、滑动窗口长度 3 轮；vLLM 解码超参严格遵循各模型官方配置。

<!--META
{"keywords": ["Multi-turn Reasoning", "Hidden State Trajectory", "Temporal Curvature", "Variance Slope", "Adaptive Reasoning", "Agent Control", "Representation Drift", "τ-Bench"], "field": "多轮推理与智能体控制", "innovations": ["提出时间曲率与方差斜率两个互补几何信号追踪隐藏状态轨迹，区分正确与错误推理
