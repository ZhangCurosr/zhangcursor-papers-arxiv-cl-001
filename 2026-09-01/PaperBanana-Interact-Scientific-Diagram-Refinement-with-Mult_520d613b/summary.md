---
title: "PaperBanana-Interact-Scientific-Diagram-Refinement-with-Mult"
source: https://arxiv.org/pdf/2608.30241v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:43:56"
field: "多轮交互科学可视化生成"
keywords: ["multi-turn diagram generation", "scientific illustration", "interactive refinement", "quality drift", "forgetting", "multi-agent system", "user simulator", "MTPaperBananaBench"]
innovations: ["提出MTPaperBananaBench首个多轮科学配图迭代基准并揭示质量漂移与遗忘两大失效模式", "设计PaperBanana-Interact多智能体critic-refine循环与多目标结构化评判", "引入文本化多图像历史摘要器以紧凑记忆替代全量历史并平衡质量与效率"]
benchmarks: ["MTPaperBananaBench", "PaperBananaBench"]
---

# 论文速读：PaperBanana-Interact: Scientific Diagram Refinement with Multi-Turn Human Feedback

## 一句话总结
本文首次系统研究科学论文配图的多轮人工反馈迭代生成任务，提出 MTPaperBananaBench 基准与 PaperBanana-Interact 多智能体交互系统，解决基线系统在多轮迭代中常见的质量漂移（quality drift）与遗忘（forgetting）两大失效模式，使多轮生成的配图质量与用户需求满足度均显著提升。

## 研究问题与动机
- **单轮生成无法满足作者的细节偏好**：作者对配图的内容表达、空间布局、视觉风格等往往需要在看到初稿后才能具体化，14 名参与者的探索性用户研究全部要求在初稿基础上进一步修订，且 86% 认为多轮迭代后更满意。
- **现有方法缺少多轮交互支持**：已有工作（如 AutoFigure、PaperBanana、Crafter）均聚焦单轮文本→图生成，缺乏处理迭代反馈的机制；直接沿用单轮模型的多次迭代会暴露系统性退化。
- **大规模人工交互评测成本过高**：人工参与多轮标注和评测不具可扩展性，无法标准化评估跨系统的多轮表现。
- **两类共性失效模式未被识别与缓解**：基线模型在多轮迭代中既会出现整体质量逐步下降（质量漂移），也会丢失前序已满足的用户需求（遗忘），两者均未被此前工作系统刻画。

## 核心贡献（创新点）
- **首个面向多轮科学配图迭代的基准 MTPaperBananaBench**：覆盖 292 张专家绘制配图与 3,518 条用户需求标注，引入用户模拟器以可重复方式逐步暴露需求。
- **系统性揭示两类多轮共性失效模式**：量化并分析质量漂移与遗忘的发生比例与维度拆解，为后续交互生成研究提供诊断基线。
- **提出 PaperBanana-Interact 多智能体精炼系统**：内部设置 critic-refiner 循环，并通过多目标 critic 同时校验当前请求、历史请求、原文忠实度与呈现质量，避免逐轮优化时的顾此失彼。
- **引入文本化多图像历史摘要器（Summarizer）**：将多轮交互历史压缩为紧凑文本记忆，在保留关键视觉细节的同时显著降低长上下文成本，避免直接喂入全量历史引发的推理退化。
- **在两项指标上全面超越基线**：相比最优单轮生成器的多轮迭代， PaperBanana-Interact 将质量得分提升 11.9–18.6 分，并将遗忘率降低 3.7–6.2 分，且用户调研中 81.3% 和 76.7% 样本分别优于 NanoBananaPro 与 PaperBanana-DirectRefine。

## 方法详解
- **任务设定**：给定初始源上下文 $x_0$（论文方法与图注），系统首先生成初稿 $I_0$；在第 $t$ 轮由用户输入自然语言反馈 $x_t$，系统基于交互历史 $[\mathbf{I}_{<t}, \mathbf{x}_{\le t}]$ 生成更新后配图 $I_t$，交互预算为 $T_{\max}$ 轮。
- **用户模拟器**：将 $N$ 条隐藏需求 $R=[r_1,\ldots,r_N]$ 按随机固定顺序逐轮揭示；每轮由 VLM 裁判 $E$ 评估当前图是否满足各需求，挑出前 $k$ 条失败需求并给出判断依据，再由 VLM 将其转化为口语化用户发言 $x_t$，直到所有需求满足或达到最大轮数。
- **数据集构建与标注体系**：需求按语义类别（内容/组织/视觉呈现）与严重等级（关键/沟通/风格）分类，经 Gemini-3.1-Pro 候选生成与三人独立人工校对，删除 0.5%、修订 24.7%；207 条子集上的 Cohen's $\kappa=0.767$，评估端与人类 Cohen's $\kappa=0.812$。
- **Summarizer**：以 Lead Visual Designer 角色按轮次输出 JSON 形式上下文笔记，描述每版图的核心内容、相对前一版的修改、与当轮用户请求的对应及剩余不满，供后续智能体复用而不重复处理原始多图像。
- **Critic-Refiner 循环**：设迭代预算 $\tau_{\max}=10$；第 $\tau$ 步 critic 基于当前图 $I^{(\tau-1)}_t$、所有用户发言 $\mathbf{x}_{\le t}$ 与摘要记忆 $M_t$，按四段顺序产出结构化 JSON 批评：`critique_latest_turn` / `critique_prior_turnN` / `critique_source_content` / `critique_presentation`；若仍需修改则由 refiner 综合批评生成修订提示词，再由 visualizer 重绘。
- **多目标 critic 规则**：第一轮的批评优先突出最新请求并在末轮综合验证；后续轮要求全面核查所有轮次约束且不得因"用户未再提及"就默认已满足；必须同时保留前序已实现的成功元素、校验原文忠实度、避免幻觉并检查排版清晰度。
- **直接精炼基线 PaperBanana-DirectRefine**：将最近一版图与提示词拼接，并完整输入用户发言序列，由 VLM 直接更新提示词后重绘，不引入独立 critic 与历史压缩。
- **评估指标**：主要指标为需求满足率 Req（全部需求平均二分得分）与配图质量 Qual（与人工参考图四维度对抗胜率：忠实度、简洁性、可读性、审美）；诊断指标为每轮需求满足率 Req$^{pt}$ 与遗忘率 Forget（曾经满足但后续转轮再次失败的占比）。

## 实验与结果
- **数据集规模**：MTPaperBananaBench 包含 292 张专家配图、3,518 条需求，平均每图 12 条需求；单/双 $k=1,3$ 两种用户模拟器设置。
- **生成器与底座模型**：单轮生成对比 NanoBananaPro、GPT-Image-2、AutoFigure、AutoFigure-Edit、PaperBanana、Crafter；多轮精炼使用 Gemini-3.1-Pro + NanoBananaPro 作为统一语言/图像底座，以保证公平。
- **主要结果（$k=1$）**：PaperBanana + PaperBanana-Interact 取得质量 61.2、需求满足 58.0、每轮满足 77.2、遗忘 12.6；相比 PaperBanana + PaperBanana-DirectRefine（47.1/52.6/68.6/18.8）质量提升 14.1、遗忘降低 6.2；相比 PaperBanana + NanoBananaPro（19.0/45.2/56.2/19.0）质量提升 42.2、需求提升 12.8、遗忘降低 6.4。
- **主要结果（$k=3$）**：PaperBanana + PaperBanana-Interact 取得质量 54.5、需求满足 94.2、每轮满足 79.6、遗忘 13.0；同样显著优于 DirectRefine（42.6/87.6/71.0/19.1）与 NanoBananaPro（18.3/76.9/59.8/22.9）。
- **质量漂移诊断**：NanoBananaPro 质量从 50.3 降至 19.0（$k=1$），其审美与结构变形在图 15/16 中集中体现；DirectRefine 则因强行接入最新请求导致布局失衡与风格漂移（图 6/17）。
- **遗忘诊断**：所有基线遗忘率集中在 14.1%–22.9%，典型场景为样式迭代覆盖掉前序已实现的标签或颜色（图 7）。
- **人工偏好验证**：150 样本盲评中 PaperBanana-Interact 以 81.3% 胜率优于 NanoBananaPro、以 76.7% 胜率优于 DirectRefine。
- **消融**：用朴素 critic 使每轮满足下降 4.9、质量下降 2.3；去掉摘要记忆遗忘从 12.6 升至 14.7；喂入全量历史虽略降遗忘却使每轮满足降至 73.5 且 token 增加 37.1%；迭代预算从 10 降至 1 时遗忘从 12.6 升至 16.8、质量从 61.2 降至 43.5。

## 相关工作脉络
- **Image generation and refinement**：InstructPix2Pix、Imagic、Prompt-to-Prompt 等提供单轮图像编辑基础，但未针对科学配图的结构性与事实性约束进行多轮校验。
- **Multi-turn text-to-image**：Hahn et al. (2025)、Nabati et al. (2024) 探索不确定条件下的交互生成，但焦点在通用场景而非科学可视化的严格忠实度要求。
- **Code-based scientific diagrams**：Automatikz、Detikzify、Tikzero 等以 TikZ/PythonPPTX 生成代码配图，优点是可复现，但编辑灵活性受限于代码空间，难以承接自然语言迭代。
- **LLM-optimized prompt generation**：Zhao et al. (2026)、Zhu et al. (2026a) 仅用 LLM 优化文字 prompt，视觉器本身不承担迭代纠错，因此多轮仍会累积漂移。
- **Agentic diagram systems**：AutoFigure、AutoFigure-Edit、Crafter、PaperBanana 等已实现端到端生成，但缺少内置 critic 与历史压缩，论文证明其在多轮下普遍劣化。
- **本文定位**：首次在科学配图领域提出多轮交互式基准、发现质量漂移/遗忘两大退化模式，并以多目标 critic+摘要记忆的组合打破"单轮强、多轮弱"的困境。

## 局限性与未来方向
- **用户模拟器仍与真实用户有差距**：当前模拟基于隐藏需求集合和 VLM 裁判，难以完全还原人类用户在交互过程中自发新增的需求与语义模糊性。
- **模型底座依赖商业系统**：实验统一使用 Gemini-3.1-Pro 与 NanoBananaPro 等，不同底座的多轮行为可能差异较大，泛化性需进一步验证。
- **迭代预算与计算开销**：$\tau_{\max}=10$ 的内部循环带来约 159.4K/11.2K token 的额外消耗（Gemini），低预算下的质量折损仍明显，自动化早停策略有待改进。
- **遗忘并未被完全消除**：最优遗忘率仍在 10%–13%，说明多目标 critic 的联合校验不足以覆盖所有历史需求的长期保留。
- **仅评测了 292 张专家配图**：数据集规模相对有限，且来自特定来源分布，跨学科/跨期刊的普适性需要扩展验证。
- **未来方向**：引入真实人机交互协议与大样本用户研究、开发轻量级多目标联合优化器、探索自进化式 critic、将方法推广到流程图/统计图/表格等多模态科研可视化工具链。

## 研究启发与可借鉴点
- **多目标 critic 的结构化输出范式**：按"最新请求→历史请求→源上下文→呈现质量"四段式拆解批评，避免单一轮次优化导致的顾此失彼，可迁移至任意长交互视觉生成任务。
- **历史压缩优于全量喂入**：摘要器以紧凑文本记忆代替原始多图像序列，在 Token 节省与推理质量之间取得更好权衡，适用于任何多轮图像/视频生成代理系统。
- **遗忘率作为多轮评测的必备诊断指标**：与每轮满足率、质量漂移曲线一起构成三层评估，可在其他多轮 Agent 场景（对话、编程、设计）中复用。
- **用户模拟器+VLM judge 的闭环评测方案**：在数据稀缺场景下通过需求库与裁判模型模拟真实交互，既保证可复现性又便于跨方法横向比较。
- **与团队结合的创新机会**：可将该框架接入团队当前面向科研助手的图文流水线，针对论文图表规范、期刊风格一致性、跨轮版本追溯等方向做定制化 critic 与评测扩展。

## 关键术语表
- **MTPaperBananaBench**：面向多轮科学配图迭代的基准，含 292 张配图与 3,518 条人工标注需求。
- **Quality drift（质量漂移）**：多轮迭代过程中配图整体质量随轮数增加而逐步退化的现象。
- **Forgetting（遗忘）**：前序已满足的用户需求在后续轮次中被意外覆盖或丢失的现象。
- **Summarizer（历史摘要器）**：将多轮交互历史压缩为紧凑文本记忆的组件，供后续智能体复用。
- **Multi-objective critic（多目标 critic）**：同时校验最新请求、历史请求、原文忠实度与呈现质量的结构化评价智能体。
- **Per-turn requirement satisfaction（每轮需求满足率）**：隔离单轮指令跟随能力的诊断指标，衡量当轮新提需求在该轮即被满足的比例。
- **User simulator（用户模拟器）**：基于隐藏需求库与 VLM judge 模拟真实用户逐步反馈的评测组件。
- **Req$^{pt}$**：每轮需求满足率的缩写，对应单轮指令跟随而非多轮累计表现。

## 可复现要素
- **数据集**：MTPaperBananaBench 标注数据与用户模拟器代码见项目页 https://shirley-wu.github.io/PaperBanana-Interact/；依赖既有 PaperBananaBench 的 292 张 expert 配图。
- **代码/权重**：论文附录 H 给出 Summarizer、Critic、Refiner、Visualizer、User Simulator、Evaluator 的完整 system prompt，但商业底座（NanoBananaPro、Gemini-3.1-Pro、GPT-Image-2）权重不可开源获取。
- **关键超参**：用户每轮揭示需求数 $k \in \{1,3\}$，最大交互轮数 $T_{\max}=5$，内部迭代预算 $\tau_{\max} \in \{1,3,5,10\}$；温度均为 1.0，Gemini 最大输出 50,000 tokens。
- **分辨率与比例**：NanoBananaPro 使用各数据点指定比例、1K 分辨率；GPT-Image-2 固定 1536×1024、高质量、PNG 输出；Crafter 采用 16:9、2K。
- **底座一致性**：所有对比系统统一使用 Gemini-3.1-Pro + NanoBananaPro，以确保公平。
