---
title: "StreamSoccer-Event-Driven-Memory-for-Streaming-Soccer-Commen"
source: https://arxiv.org/pdf/2608.19723v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:36:50"
---

# 论文速读：StreamSoccer: Event-Driven Memory for Streaming Soccer Commentary

## 一句话总结
本文提出 StreamSoccer，一种以事件记忆（Event Memory）为中间表示的流式足球解说系统，通过将连续视频流组织为固定预算的活跃状态与异步闭合的可检索历史记录，在严格因果截断约束下统一生成当前事件、近期窗口与历史记忆三轨解说，并在90分钟完整比赛的原始视频流中保持稳定的亚实时处理效率。

## 研究问题与动机
- 现有足球解说方法多依赖预定义视频片段、固定输出时间戳或离线全片输入，无法在直播流式场景下按时间顺序持续更新状态。
- 通用流式视频语言模型（如 StreamingVLM、TimeChat-Online）以帧、视觉 token 或 KV cache 为单位压缩历史，缺乏对“语义事件生命周期（激活-更新-闭合-检索）”的显式建模。
- 足球解说需覆盖不同时域：刚发生事件描述、近期多事件聚合、早期事件回溯，现有架构难以在同一系统内以有界计算/内存统一支撑这三种跨度。
- 长历史流式处理伴随计算与显存持续增长风险，需在不停累积无限历史的条件下实现稳定的实时推理。

## 核心贡献（创新点）
- 提出流式足球解说的三轨任务定义与严格因果评测协议，明确观察截断点（causal cutoff）与源事件溯源，彻底隔离未来信息泄露，并配套匹配离散（match-disjoint）的数据切分管线。
- 设计 StreamSoccer 事件驱动架构，以固定预算的事件记忆为中间表示，通过门控融合更新活跃状态、基于操作转换或时长兜底自动闭合生成完成事件记忆，并异步转为可检索的结构化文本历史记录。
- 构建统一的多上下文解说生成器与规则辅助调度器，共享底层记忆投影接口与冻结 LLM，在单一生成界面内协同当前、近期与历史三种语境配置，避免多任务独立部署的冗余。
- 提供从参考锚点质量、原始视频流效率到记忆表征机制的完整验证链条，证明事件记忆在压缩效率与跨时域语义保持上显著优于帧级/时间窗级基线，且 p95 RTF 不随比赛历史增长而恶化。

## 方法详解
- **Streaming Event Memory Encoder**：冻结 Qwen3-VL 视觉编码器，Clip Event Adapter（交叉注意力重采样）将每段 4s clip 压缩为 $K_m=9$ 个事件 token $E_t$。Transition Head 比较当前 clip 与上一活跃状态，预测操作转换信号 $b_t$；结合最大时长 $D_{\max}=24\text{s}$ 的安全兜底 $\rho_t$ 得到重置信号 $r_t = b_t \vee \rho_t$。活跃记忆通过槽位门控渐进融合：$\widetilde{M}_t = (1-G_t) \odot M_{t-1} + G_t \odot \mathcal{U}(M_{t-1}, E_t)$；当 $r_t=1$ 时，$M_{t-1}$ 冻结为完成事件记忆 $M_j^c$，新 clip 初始化下一活跃状态。
- **Memory-to-Record Consolidation**：完成记忆 $M_j^c$ 经投影器 $P_\theta$ 转为固定长度（$K_p=8$）软前缀，结合最小可用比赛上下文 $x_j^{\text{ctx}}$ 由语言模型自回归生成事件摘要 $c_j$，随后附加确定性元数据（时间、动作、类型、比分等）组装为文本记录 $\mathcal{R}_j$，异步存入历史库 $\mathcal{L}_t$ 供后续检索。
- **Multi-Context Commentary Generator**：三轨共用 $P_\theta$ 与 LLM。当前事件轨道仅投影 $M_j^c$；近期/历史轨道将活跃记忆 $M_t$ 与本地窗口内已完成记忆 $[\cdot; \mathcal{B}_t^{\text{sel}}]$ 拼接后投影为软前缀 $Z_t$；历史轨道额外将检索到的文本记录 $\mathcal{R}_t$ 作为 prompt 文本输入语言模型，实现隐状态与文本历史的统一解码。
- **Rule-Assisted Scheduler**：基于确定性规则（冷却时间、周期阈值、事件距离门控、优先级 history > event > recent）在 {silence, current-event, recent-window, historical-memory} 中决策；调度器不学习策略，仅负责 speaking cadence 与上下文配置选择。
- **Training Objectives**：三阶段训练。Stage 1 训练记忆编码器，含 clip 级稀疏多标签识别 loss（asymmetric loss）、转换预测 focal loss、完成事件动作集与类型分类 loss；Stage 2 训练投影器与 LoRA（r=8, α=16）生成记录摘要；Stage 3 联合优化生成 loss、记忆保持 loss 及路由/局部/检索排序 loss，前 5k 步使用 teacher-forced 上下文，后 5k 步逐步暴露预测上下文（比例从 0.10 升至 0.50）。

## 实验与结果
- **数据集**：基于 SoccerNet 动作标注与 MatchTime 解说构建的三轨流式足球解说数据集，共 27,639 样本（当前事件 15,189 / 近期窗口 7,127 / 历史记忆 5,323）；按比赛 ID 排序切分为训练（19,641）、验证（4,209）、测试（3,789），严格 match-disjoint。
- **评测协议**：区分 sample-level conditional、reference-anchored replay、free-scheduler replay、raw-video scaling 四类设置；主表采用参考锚点回放，使用 oracle 操作闭合与记录就绪干预。
- **主要结果**：在共享因果输出锚点评测中，StreamSoccer 当前事件 CIDEr 达 **38.62**（第一），近期窗口 **23.96**（第二，略低于 UniSoccer 的 27.85），历史记忆 **17.39**（第一）；BERTScore-F1 分别为 0.9734 / 0.9612 / 0.9636，全面领先或持平主流基线。
- **效率**：58 场源干净比赛的 174 次原始视频运行中，RTF p95 稳定在 **0.10–0.22**，不随比赛历史增长而持续上升；峰值 VRAM 约 17,996 MiB。对比组中 TimeChat-Online 约 55 分钟后触及上下文上限停止输出，VideoLLM-online 后期 RTF 逼近或超过实时阈值。
- **消融与机制验证**：循环事件记忆在记录摘要生成上稳定优于操作事件池化（CIDEr 37.40 vs 22.20）与固定时间池化（15.72）；完整三轨配置在所有轨道均最优，近期窗口增益最大（+6.01 CIDEr）；盲评 LLM（GPT-5.6-terra）审核胜率 **93.16%**。

## 相关工作脉络
- **足球解说与字幕基线**：UniSoccer、SoccerMaster、MatchAware、GameSight 等依赖预定义片段、外部统计或全片输入，缺乏流式状态维护；本文聚焦因果截断下的连续状态演进与事件生命周期管理。
- **通用流式视频语言模型**：StreamingVLM、TimeChat-Online、VideoLLM-online 等以帧/token/cache 压缩历史；本文将压缩原子升级为“事件”，更适合体育解说这类强过程连续性与语义闭合需求的场景。
- **长视频记忆与检索增强**：MART、MovieChat、VideoRAG、SoccerComment 多使用通用内部状态或预建外部语料；本文从当前流中实时衍生可检索事件记录，实现流内闭环与异步归档。
- **训练与评估范式**：采用三阶段逐步解冻与 curriculum 式上下文暴露策略；同时严格区分“可控锚点质量”“
