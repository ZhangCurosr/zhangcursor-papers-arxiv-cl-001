---
title: "Geometry-of-Divergence-Tracking-Hidden-State-Trajectories-fo"
source: https://arxiv.org/pdf/2608.30650v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:05:44"
field: "多轮Agent推理与表示健康监测"
keywords: ["multi-turn reasoning", "hidden-state trajectory", "temporal curvature", "variance slope", "adaptive reasoning", "LLM agent", "tau-Bench"]
innovations: ["用时间曲率和方差斜率两个几何信号刻画多轮隐藏状态轨迹并区分成败episode", "发现轨迹可分性具有动作依赖性，信息输出链判别力最强", "证明无训练单阈值几何触发策略可超越学习型触发器并实现token效率提升"]
benchmarks: ["tau-Bench", "Lost in Conversation (Math/Code)", "Qwen3-14B/32B", "Llama-3.1-8B"]
---

# 论文速读：Geometry-of-Divergence-Tracking-Hidden-State-Trajectories-for-Adaptive-Multi-Turn-Reasoning

## 一句话总结
本文提出从大语言模型隐藏状态的隐式轨迹中提取**时间曲率**和**方差斜率**两个几何信号，用于在多轮 Agent 交互中提前区分成功与失败推理路径，并据此触发自适应思考模式，在 τ-Bench 上将任务成功率从 24.1% 提升至 39.6%，同时降低 11.2% 的 token 消耗。

## 研究问题与动机
- **多轮推理中的表示漂移**：随着多轮上下文累积，底层 LLM 对早期任务信息的内部表征逐渐失稳，难以区分"有效推理更新"与"表示漂移"。
- **既有方法依赖显式上下文工程**：规划、反思循环、记忆架构等方法作用于文本历史层，未解决 Agent 目标与 LLM 训练目标之间的错位问题。
- **推理强度固定、缺乏自适应**：现有大型推理模型（LRM）在对话中以固定推理力度运行，无法根据每轮的内在不确定性动态调整，导致简单问题过度思考或复杂问题思考不足。
- **隐藏状态轨迹难以观测和诊断**：多轮交互中每次隐藏状态更新同时包含当前推理与早期轮次的累积影响，导致难以定位"何时开始偏离核心目标"的关键拐点。

## 核心贡献（创新点）
- **双信号轨迹表征**：将多轮推理形式化为可观测的隐藏状态轨迹，并用**时间曲率 κ**（表征相邻轮次方向一致性）和**方差斜率 β**（表征探索空间扩张/收缩）两个互补几何信号刻画轨迹形态，跨 4 类任务×3 个模型验证了失败轨迹的方向反转和过早收敛特征。
- **动作链分解揭示可分性差异**：将每轮 episode 分解为四种动作（Read/Write/Respond/Transfer）构成的三动作滑动窗口链，发现轨迹可分性具有**动作依赖性**——信息输出型链（尤其是 Respond→Respond→Respond）判别力最强，信息获取型链（如连续 Read）判别力较弱。
- **无训练几何触发策略实现在线控制**：证明几何信号可作为在线控制信号，在失败迫近轮次触发思考模式；基于 κ 的单阈值无训练策略即可达到或超越学习到的触发分类器（Learned-HST），并在 τ-Bench 上平均提升成功率 15.5 个百分点、减少 11.2% token 消耗。

## 方法详解
- **隐藏状态序列定义**：在第 t 轮用户输入 $q_t$ 的末位 token 处，从固定中间探针层（Qwen3-14B 取 layer 22，Qwen3-32B 取 layer 42，Llama-3.1-8B 取 layer 16）提取隐藏状态 $h_t \in \mathbb{R}^d$，构成轨迹 $\mathcal{H} = \{h_0, \ldots, h_{T-1}\}$。
- **残差过程建模**：轮间动态建模为 $h_t = h_{t-1} + \mathbf{v}_t$，其中位移向量 $\mathbf{v}_t = h_t - h_{t-1}$ 编码该轮相对历史上下文的表示更新。
- **时间曲率 $\kappa_t$**：定义为连续两个位移向量夹角的余弦值，$\kappa_t = \frac{\mathbf{v}_t \cdot \mathbf{v}_{t-1}}{\|\mathbf{v}_t\|_2 \|\mathbf{v}_{t-1}\|_2}$，取值 [−1, 1]；接近 1 表示方向一致累积，接近 −1 表示方向急剧反转，成功轨迹的 κ 中位数更接近 0 或为正，失败轨迹更负。
- **方差斜率 $\beta_t$**：先计算前缀 $\{h_0, \ldots, h_\lambda\}$ 的 dispersion $V(\lambda) = \frac{1}{\lambda}\sum_{i=0}^\lambda \|h_i - \bar{h}^{(\lambda)}\|_2^2$（无偏估计），再对 $\lambda=1,\ldots,t$ 的 dispersion 序列关于前缀长度做 OLS 回归，斜率即 $\beta_t$；正值表示表示逐步发散（探索空间扩张），负值表示收敛（探索收缩）。
- **轮对齐策略**：所有 $h_t$ 对齐到用户输入 $q_t$ 的末位 token 处（Agent 行动前），以消除"失败 episode 更长"这一混淆因素；κ 从第 2 轮起有定义，β 采用相同索引对齐。
- **动作链分类**：将 τ-Bench 中的动作分为 Write（改写数据库）、Read（只读查询）、Respond（回复用户）、Transfer（转人工）四类，用 3-步滑动窗口构造三动作链，统计各链在成功/失败 episode 中的占比及几何信号的可分性。
- **触发策略对比**：Fixed Policies（Never/Always/Warm-up）、Action-conditioned（基于预测动作触发）、Geometry-conditioned（κ<τκ 或 β>τβ 直接触发）、Learned-HST（训练分类器联合输入最近 3 轮动作 one-hot 与 (κ, β)）四类策略，在 τ-Bench 上对比。

## 实验与结果
- **数据集/基准**：Math、Code（来自 Lost in Conversation 的分块 prompt 框架）+ Airline、Retail（来自 τ-Bench），覆盖开放式推理与工具约束交互两类场景；使用 Qwen3-14B、Qwen3-32B、Llama-3.1-8B 三个模型。
- **曲率分析**：失败 episode 的 κ 均值在全部 6 组配对中均低于成功组，差值（Correct−Incorrect）：Math +0.047、Code +0.148、Retail(14B) +0.030、Airline(14B) +0.049、Retail(32B) +0.035、Airline(32B) +0.088；Code 成功组中位数达 +0.127，体现结构化任务中方向一致累积。
- **方差斜率分析**：成功 episode 的 β 在所有 6 组中均高于失败组；可分性在 Math/Code 上最显著，Airline 两组最不显著（14B: p=0.30, 32B: p=0.057），与任务开放度正相关。
- **动作链可分性**：信息输出型链（如 Respond×3，ID=2）在两类任务中均有最强的几何分离结构；信息获取型链（ID=1,4 以 Read 为主）判别力有限。
- **在线触发结果（Table 3）**：
  - **Retail/Qwen3-14B**：Never-think 0.286 → κ<−0.20 触发 0.397（提升 11.1pp），token 101.0k vs 112.6k（↓10.4%），max-turn 终止从 75 降至 34。
  - **Retail/Qwen3-32B**：κ<−0.15 触发达 0.442，超越 Learned-HST（0.405）达 +27%（相对 Never-think 0.347）。
  - **全局平均**：成功率从 24.1% 升至 39.6%（+15.5pp），平均 token 成本从 104.8k 降至 93.0k（↓11.2%）。
  - 几何阈值策略无需额外训练，单标量阈值即可达到或优于学习策略，且阈值可由 κ 的 Correct−Incorrect 差值指导校准。

## 相关工作脉络
- **Multi-Turn Degradation（Lost in Conversation, Laban et al. 2025；τ-Bench, Yao et al. 2024）**：已有工作揭示了多轮交互中性能衰减现象，但主要依赖显式上下文工程；本文从隐式隐藏状态轨迹切入，提供互补的诊断视角。
- **Hidden State Probing（ICR Probe, Zhang et al. 2025b；Semantic Entropy, Farquhar et al. 2024；INSIDE, Chen et al. 2024a）**：I CR Probe 追踪跨层隐藏状态更新用于幻觉检测，但仅分析单条响应内部；本文聚焦**跨轮间**轨迹几何，并将信号用于在线控制而非事后诊断。
- **Adaptive Reasoning Control（AdaptThink, Zhang et al. 2025a；Adacot, Lou et al. 2025；Thinkless, Fang et al. 2026）**：前述方法基于输入问题在单轮设置下训练 RL/Pareto 策略选择是否思考；本文策略以跨轮累积的隐藏状态几何为条件，不依赖额外训练数据。
- **Truth/ Hallucination Detection from Activations（Azaria & Mitchell 2023；Marks & Tegmark 2023；Orgad et al. 2025）**：已有工作证明 truthfulness 可从激活中线性解码或在特定 token 富集；本文进一步将表征几何与多轮失败类型（而非单一真假判断）关联。
- **Context Engineering Survey（Mei et al. 2025）**：综述了规划、反思、记忆等显式方法；本文定位为"不修改上下文结构，直接监控模型内部动力学"的轻量级补充。

## 局限性与未来方向
- **任务可分性不均匀**：Airline 任务的方差斜率可分性不显著（p=0.30/0.057），在高度开放的交互空间中几何信号的判别力下降，需要结合其他信号。
- **阈值仍需离线校准**：几何触发器虽无需训练，但阈值 τκ/τβ 的选取依赖对特定任务-模型配对的 κ 差值分布进行一维扫描，尚未实现完全自适应。
- **探针层固定**：当前使用固定中间层（layer 22/42/16）提取隐藏状态，未系统搜索最优探针层；不同层可能编码不同粒度的任务信息。
- **作者自述方向**：① 将几何信号应用于多轮对抗攻击/越狱检测；② 与持续学习 Agent 系统结合，实现长时程漂移自适应修正。

## 研究启发与可借鉴点
- **"轮对齐"消除轨迹长度混淆**：将隐藏状态对齐到"用户输入末位 token（Agent 行动前）"而非行动后，可有效剥离"失败 episode 更长"这一机械偏差，该对齐策略值得在多轮诊断任务中复用。
- **无训练阈值替代学习策略**：单标量 κ 阈值即可达到甚至超越 Learned-HST 分类器，为资源受限场景下的在线控制提供了零样本部署路径，可迁移到任何可访问隐藏状态的 Agent 系统。
- **动作链分解揭示信号来源**：通过三动作滑动窗口统计各链的成功/失败占比与几何可分性，精准定位"哪些行为模式携带判别信息"——这一分析方法可推广至其他多轮 benchmark（如 SWE-bench、WebShop）的诊断研究。
- **方向一致性与探索空间扩张作为通用健康指标**：κ 表征方向一致性、β 表征探索广度，两者分别捕捉"是否走偏"和"是否过早收敛"，可作为多轮 Agent 健康度监控的通用双指标框架。

## 关键术语表
- **Temporal Curvature (κ)**：连续两轮隐藏状态位移向量夹角的余弦值，刻画语义漂移方向的一致性，κ≈−1 表示方向反转（失败信号）。
- **Variance Slope (β)**：前缀隐藏状态 dispersion 序列关于前缀长度的 OLS 斜率，正斜率表示探索空间逐步扩张，负斜率表示过早收敛。
- **Hidden-State Trajectory**：多轮交互中按轮次排列的模型隐藏状态序列 $\mathcal{H}=\{h_0,\ldots,h_{T-1}\}$，是本文几何分析的对象。
- **Action Chain**：由连续三轮 Agent 动作构成的三元组（Read/Write/Respond/Transfer 任意组合），用于分解episode 并定位几何信号来源。
- **Geometry-Conditioned Trigger**：当 κ<τκ 或 β>τβ 时触发思考模式的在线干预策略，无需额外训练数据。
- **Learned-HST**：训练分类器联合输入最近 3 轮动作 one-hot 与 (κ, β) 预测失败迫近轮次的学习型触发策略。
- **τ-Bench**：模拟用户-Lang-Agent 动态对话的基准，包含 Airline 和 Retail 两个带领域 API 工具的任务域。
- **Representation Drift**：多轮上下文中 LLM 内部任务相关表示逐渐失稳、偏离初始目标的过程。

## 可复现要素
- **数据集**：Math、Code（Lost in Conversation 分块 prompt 框架）+ Airline、Retail（τ-Bench, Yao et al. 2024）；τ-Bench 公开可访问。
- **模型**：Qwen3-14B、Qwen3-32B（Qwen Team 2025）、Llama-3.1-8B（Grattafiori et al. 2024），均可公开获取。
- **代码/权重**：论文未明确声明代码开源仓库；模型权重可公开下载。
- **关键超参**：探针层（Qwen3-14B: layer 22, Qwen3-32B: layer 42, Llama-3.1-8B: layer 16）；触发阈值 τκ∈{−0.10, −0.15, −0.20, −0.25}；Warm-up 策略在前 2 轮强制思考；最大轮次 50。
- **推理框架**：vLLM，8× NVIDIA L40S + 4× NVIDIA H200。
