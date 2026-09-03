---
title: "Code-World-Model-Coding-Agent-as-World-Brain"
source: https://arxiv.org/pdf/2608.25927v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:31:28"
field: "视频生成与交互式世界模型"
keywords: ["world model", "coding agent", "proxy conditioning", "interactive video generation", "persistent world state", "code-driven simulation"]
innovations: ["以 coding agent + 可执行代码作为世界演化核心并与视频模型解耦", "提出代理分辨率逐帧时空条件（proxy/proxy video）作为世界状态到视觉生成的白色盒子接口", "构建游戏与真实世界对齐的 proxy-observation 数据流水线并验证可复用性"]
benchmarks: ["GTA V 游戏数据（自建 proxy–RGB 配对）", "KITTI-360 真实世界 proxy 合成 proof-of-concept"]
---

# 论文速读：Code-World-Model-Coding-Agent-as-World-Brain

## 一句话总结
本文提出 Code World Model (CWM)，将世界演化与视觉实现解耦：用 Coding Agent 通过可执行代码维持持久化世界状态与因果演化，结合 Proxy 接口将状态编译为逐帧时空约束视频，引导预训练视频模型生成高保真视觉观测，从而实现开放-ended、规则一致的世界演化。

## 研究问题与动机
- 现有 video-based world model 仅从视觉结果学习动态，无法显式捕捉支配世界演化的隐藏知识、规则与机制，难以维持跨时间段/离屏的持久化后果。
- 纯视频生成路线在长时程复杂交互（如玩家弑君后的政治连锁反应）中缺乏常识推理、规划与因果链保持能力。
- 视频模型上下文通常短于1分钟，而许多世界因果过程跨越更长时间尺度，靠视觉数据“间接推断”规则效率低且不优。
- 现有控制接口（文本、动作/相机信号、轻量几何）难以对实体位姿、轨迹、相机路径与空间关系提供逐帧细粒度、可直接追溯的时空控制。

## 核心贡献（创新点）
1. 提出以 Coding Agent 为“世界大脑”的 Code World Model，将世界状态拆分为可执行状态与视觉状态，通过 agent-code 联合转移驱动长期演化，并将视觉生成外包给视频模型；与现有视频世界模型仅把生成序列作为中心建模对象不同，本文把持久、可修改的可执行代码置于演化核心。
2. 设计 Proxy 接口：把实体位置/姿态/轨迹、空间关系与相机运动组织为粗粒度可编程表征并由确定性编译器渲染为 proxy video，为视频模型提供直接逐帧时空约束，区别于依赖完整3D资产管线或仅文本/低维信号的基线控制方式。
3. 构建游戏与真实世界两套 proxy–observation 对齐数据流水线：游戏侧从运行态同步抽取相机/实体/场景状态自动构造像素级实例映射，真实侧利用 KITTI-360 的标定相机、语义3D重建与对象标注离线合成 proxy，无需动作标注即可复用真实视频先验。

## 方法详解
- 世界状态分解：$S_t = (S_t^{\mathrm{exe}}, S_t^{\mathrm{vis}})$，其中 $S_t^{\mathrm{exe}}$ 含可执行世界程序、实体属性、规则、关系、事件历史等；$S_t^{\mathrm{vis}}$ 含需时间一致的外观/运动信息。
- 更新方程：
  - $S_{t+1}^{\mathrm{exe}} = \mathcal{T}_{\mathrm{AC}}(S_t^{\mathrm{exe}}, A_t)$，由 coding agent 读当前状态后调用/重写可复用代码，代码在高频、确定性、规则一致的前提下推进大量变量，agent 只在稀疏高复杂度决策、异常与机制变更时介入；
  - $S_{t+1}^{\mathrm{vis}} \sim G_\theta(S_t^{\mathrm{vis}}, S_{t+1}^{\mathrm{exe}})$，视频模型基于更新后执行状态与前一视觉状态生成新观测。
- Proxy 与条件带宽：proxy 不追求完整3D与外观细节，只提供当前观测必须遵循的最低充分时空状态；其控制信号全部可追溯至世界状态，保持 agent 可inspect/edit 的白色盒子路径。代理分辨率取目标视频的1/4（空间维度），额外 token 开销约为1/16。
- 与文本双通道协同：结构化文本传达语义意图、身份/属性/动作语义与外观细节；proxy video 提供相机轨迹、实体位置、遮挡与拓扑的帧级约束。agent 可根据任务决定是否启用/调整 proxy 粒度与帧率。
- 数据构造：
  - 游戏数据（如 GTA V）：同步录制 target RGB + 运行态记录（相机、位置/朝向、近似尺度、场景布局、交互关键状态），用可重用 primitive 集合程序化构造 proxy 并在相同相机轨迹下渲染；保留像素级实例图绑定 identity。
  - 真实数据（KITTI-360 原型）：利用标定相机位姿与累积语义3D重建离线合成 proxy，投影得到度量深度与法向；行人/车辆/植被/建筑按统一 primitive 词表表达，共享深度缓冲保持时序遮挡，无需动作/相机标注。
- 训练/推理配置：在 MiniMax-H3 Ref2VA 上 LoRA（rank-128，~596M 参数）微调；训练集约5.6小时、9420个5s clip，target 124帧 1344×768@24fps，proxy 124帧 336×192；优化 AdamW、batch=8、Warmup 100 + cosine 衰减。推理以 GPT-5.6 Sol 为 coding agent、GPT Image 2 生成首帧外观锚点，20-step 显式 Euler 采样；长视频采用124帧滑动窗口（34帧重叠）保证局部连续与全局身份稳定。

## 实验与结果
- 数据集/数据：
  - 游戏数据：约157段 GTA V 片段，采样为9420个5s clip（~5.6小时源视频）。
  - 真实数据：KITTI-360 用于离线 proxy 合成验证（未见定量主实验数字，属 proof-of-concept）。
- 基线与对比：与 LingBot-World 2.0、WorldPlay、Matrix-Game/Matrix-Game 3.0、Hunyuan GameCraft、YUME 1.5、Infinite-World 等视频/交互世界模型相比，主要定性对比 proxy 控制在角色运动、动作与相机跟随上的帧级精度与响应性；论文未给出统一数值排行榜，强调控制精度而非推理延迟。
- 主要结果（定性为主）：仅用5小时游戏数据 LoRA 微调后，视频模型在由 coding agent 构建的简单交互世界所录制的 proxy 条件下，能严格遵循 proxy 指定的实体位置/动作轨迹、场景布局与相机运动，同时保留丰富的外观细节与细粒度物理/动作动态；跨 diverse characters、environments、motions、camera trajectories 强泛化。
- 最强/提升：在“细粒度逐帧时空控制+高保真视觉”的组合指标上显著优于仅文本/动作/相机信号控制的基线；proxy 为视频模型提供比纯文本或低维信号更强的定位与轨迹约束，同时避免完整3D资产管线的成本与视觉上限。

## 相关工作脉络
1. 交互式视频世界模型（Genie/Genie 2/3、Diffusion-for-Atari、GameFactory、GameGen-X、Hunyuan GameCraft、YUME、PAND、Matrix-Game、Infinite-World、MinWM 等）以视觉生成与交互控制为核心，主要靠历史/记忆/动作信号维持一致性；CWM 的区别在于以 persistent executable code 承载状态与后果，视频模型只负责视觉渲染。
2. 长程一致性与记忆（WorldMem、VRAG、RELIC、Infinite-World、MosaicMem、VMem、ActWorld 等）通过几何历史、检索帧、持久视图/事件记忆缓解自回归累积误差；CWM 不依赖视觉记忆吸收长期因果，而是由 code 持续维护可被后续观察引用的显式状态。
3. 实时/流式生成（Oasis、Happy Oyster、The Matrix、Vid2World、LongLive、Rolling/Causal/Self Forcing 等）关注吞吐与延迟；CWM 当前为离线/窗口式生成，实时优化留作工程扩展。
4. 生成式3D世界（HY-World、WorldGen、HunyuanWorld、Matrix-3D、FlashWorld、WorldGrow、HOLODECK 等）强调可导航几何与资产；CWM 的 proxy 并非最终3D世界，而是面向视频模型的粗粒时空条件接口。
5. 代码驱动的世界模型（WorldCoder、Dainese 等 MC+LLM、Code world models for general game playing、ARC-AGI-3 可执行模型、FAIR CodeGen CWM 等）证明规则/转移可外化为可检查可重写的程序；CWM 进一步与视频模型对接，并把 agent-code 分工（稀疏推理 vs. 高频执行）纳入世界演化架构。
6. 开源/评测驱动的 agent 游戏构建（GameDevBench、OpenGame 等）以产出可玩游戏为目标；CWM 的 coding agent 目标是持续治理开放视觉世界的演化机制而非一次性生成游戏制品。

## 局限性与未来方向
- 训练规模受算力限制（仅5.6小时游戏数据、3 epochs），生成质量与泛化仍有提升空间。
- 未实现自回归实时生成；当前为离线/窗口拼接范式，延迟与帧率受限于视频模型推理。
- 现有 coding agent 仍难以从0可靠构建完整开放世界机制与复杂3D资产；原型依赖已有 AAA 模板场景与逻辑进行组合/改写。
- Proxy 带宽与可见性需在“可构造性 vs. 空间接地强度”之间权衡；当前为固定 proxy 设计，未包含 agent 自主开关/带宽调节能力。
- 真实世界数据仅以 KITTI-360 做离线 proof-of-concept，尚未扩展到大规模真实视频训练。

## 研究启发与可借鉴点
- **状态-渲染解耦的工程范式**：将“规则/状态演化”与“视觉合成”拆为 agent-code 与 video model 两条流水线，可作为通用 world-model 设计模板；适用于需要长期因果一致性的虚拟世界、仿真训练、agent 评估环境。
- **Proxy 作为可白盒化的视觉条件**：用低分辨率、可直接追踪至 world state 的逐帧时空视频替代纯文本/低维信号，兼顾控制精度与构建成本；在需要精细相机/角色轨迹复现的任务中值得复用。
- **游戏/真实数据对齐的 unified pipeline**：游戏侧复用运行态（相机+实体+实例图），真实侧用标定/3D重建离线合成，无需动作标注；为跨域预训练提供低成本、大规模 proxy-RGB 样本来源。
- **Agent-Code 分层调度**：稀疏高认知决策由 coding agent 完成、密集确定性子过程由代码高频执行；这一分工可直接迁移到需要长程规划+实时物理/数值更新的仿真与 embodied agent 训练场景。
- **身份绑定强化条件有效性**：pixel-wise instance map 将结构化文本中的 identity/外观/动作与 proxy 区域精确对齐，显著提升多主体场景下的生成准确性，值得在多图/多角色生成中推广。

## 关键术语表
- **Code World Model (CWM)**：以 coding agent 为“世界大脑”的框架，通过可执行代码驱动持久化状态演化，并由视频模型负责视觉实现。
- **Proxy / Proxy video**：将实体位置/姿态/轨迹、空间关系与相机运动编码为粗粒度可编程表征，并由确定性编译器渲染为视频，作为帧级时空条件输入视频模型。
- **Executable state vs. Visual state**：世界状态被拆为 $S^{\mathrm{exe}}$（规则、属性、事件历史等可被 code 直接操作的部分）与 $S^{\mathrm{vis}}$（视频模型生成的外观/运动一致性部分）。
- **Agent–code 分层**：coding agent 处理稀疏高复杂度推理与机制修订，代码在后台高频执行确定性更新，避免每次都调用大模型。
- **Condition bandwidth**：proxy 提供信息的粗细程度；越丰富控制越强但 agent 构建负担越大，需在可构造性与空间接地强度间权衡。
- **GPT-5.6 Sol / GPT Image 2 / MiniMax-H3 Ref2VA**：分别作为 coding agent、首帧外观锚点生成器与视频生成 backbone 的具体模型。
- **Auto/离线 proxy 构造**：游戏侧从运行时记录自动同步生成，真实侧利用 3D 重建与标定相机离线合成，均保持 RGB-proxy 时序对齐。

## 可复现要素
- 数据集：GTA V（约157段、~5.6小时，作者自建 proxy–RGB 配对）；KITTI-360（真实世界 proof-of-concept）。论文未明确外部公开链接。
- 代码/权重：使用 MiniMax-H3 Ref2VA 官方 checkpoint 与 LoRA 权重训练；coding agent 使用 GPT-5.6 Sol，首帧使用 GPT Image 2。项目页面为 https://buaacyw.github.io/cwm/，论文未给出独立开源仓库声明。
- 关键超参：LoRA rank=128、可训练参数约596M；batch size=8；AdamW、weight decay=0.01、梯度裁剪 1.0；BF16 + FlashAttention-3 + gradient checkpointing；学习率从 $2\times10^{-5}$ cosine 衰减至 $1\times10^{-6}$；warmup 100 steps，共3 epoch（3,534 steps）；proxy 分辨率 336×192（target 1344×768 的 1/4），采样 11 帧（偏移 0,12,…,120）；长视频窗口 124 帧、重叠 34 帧（步进 90 帧）；采样 20 steps 显式 Euler。
