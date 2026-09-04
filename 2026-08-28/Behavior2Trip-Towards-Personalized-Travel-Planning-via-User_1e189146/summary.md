---
title: "Behavior2Trip-Towards-Personalized-Travel-Planning-via-User"
source: https://arxiv.org/pdf/2608.26807v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:49:56"
field: "旅行规划智能体与个性化偏好推断"
keywords: ["Travel Planning", "Reinforcement Learning", "User Preference Inference", "Agentic RL", "Implicit Instruction"]
innovations: ["提出从行为轨迹隐式推断偏好的行为感知旅行规划新任务", "设计结合外部工具调用与内部记忆管理的RL智能体B2T-Agent"]
benchmarks: ["Behavior2Trip", "TravelPlanner"]
---

# 论文速读：Behavior2Trip-Towards-Personalized-Travel-Planning-via-User

## 一句话总结
本文提出面向隐式用户行为的个性化旅行规划新任务，构建了基于真实OTA平台数据的1.1万规模基准Behavior2Trip，并设计了结合外部工具调用与内部记忆管理的强化学习智能体B2T-Agent，有效从行为轨迹中推断偏好并生成高质量旅行计划。

## 研究问题与动机
- 现有旅行规划智能体主要依赖**单轮显式指令**或**多轮澄清对话**来收集用户偏好，过度依赖用户的主动输入。
- 用户历史交互行为（点击、收藏、预订等）中天然蕴含丰富的隐性偏好信号，但现有方法普遍忽略这一信号来源。
- 过度依赖显式交互显著增加了用户沟通负担，且难以实现真正贴合个人习惯的深度个性化规划。
- 亟需一种无需显式指令、能直接从行为轨迹中隐式推断偏好并生成符合常识与个性化约束的旅行计划的新范式。

## 核心贡献（创新点）
1. **提出行为感知旅行规划新任务**：首次将用户历史行为轨迹作为偏好推断源，实现无显式指令的个性化计划生成，与依赖主动输入的现有范式形成本质区别。
2. **构建大规模真实数据基准Behavior2Trip**：基于头部OTA平台脱敏数据构建11,400条样本，覆盖平均39.8个行为动作、14个属性维度与5类偏好，填补了隐式偏好规划评测空白。
3. **设计B2T-Agent强化学习智能体**：引入结构化动作空间，联合外部工具调用与内部键值记忆管理，并通过多组件奖励函数端到端优化规划过程，摆脱了传统提示工程的手动工作流。
4. **验证强泛化能力**：B2T-Agent在Behavior2Trip上显著超越现有基线，并在TravelPlanner基准上以Qwen3-8B体量击败GPT-4.1，证明方法具备跨场景迁移价值。

## 方法详解
- **问题定义**：给定用户行为轨迹 $A=[a_1,...,a_t]$（隐式编码偏好 $P$）与自然语言请求 $I$，模型通过工具集 $\mathcal{T}$ 查询POI数据库 $\mathcal{D}$ 生成计划 $\mathcal{R}$，需同时满足常识约束（CC）与用户偏好约束（UPC）。
- **基准构建四阶段**：
  1. **沙盒环境搭建**：集成80.6k真实POI（景点/酒店/餐厅/娱乐）、9个细粒度搜索工具及32项约束。
  2. **偏好驱动行为生成**：按易/中/难三档生成用户画像，利用LLM角色扮演生成连贯的点击→收藏→预订序列，噪声POI仅允许点击动作。
  3. **Chain of Action标注**：以Open-Manus为 rollout 底座，通过注入Next Step Prompt引导生成含工具调用与中间决策的完整推理轨迹。
  4. **双阶段质量评估**：Deepseek-R1自动打分（仅保留满分样本）+ 3名独立标注员抽检，一致率94.6%。
- **B2T-Agent核心设计**：
  - **结构化动作序列**：$y=(a_1,...,a_k,R)$，动作类型包括外部工具调用`<tool_call>`、内部记忆读写`<memory>`、思维链推理`<think>`与最终结果`<answer>`。
  - **多组件奖励建模**：引入无效动作惩罚 $P_{action}=-1$；旅行计划奖励采用两级门控 $R_{plan}=R_{format}\cdot(R_{CC}+R_{UPC})$，总奖励 $R_{total}=P_{action}+R_{plan}$。
  - **策略优化**：基于GRPO目标函数优化策略模型，保留KL散度正则项防止策略偏离参考模型。
  - **Loss Mask机制**：屏蔽`<tool_response>`与`<memory_response>`等非模型生成token，仅对自生成token计算梯度，避免环境噪声干扰训练。

## 实验与结果
- **数据集与基线**：Behavior2Trip（9,000训练/2,400测试，分Easy/Medium/Hard）；基线包括ReAct、SFT、Qwen3-8/14/32B、DeepSeek-V3、GPT-4o、GPT-4.1。
- **评估指标**：Delivery Rate (DR)、CC/UPC的Micro PR与Macro PR、Final PR（全约束通过率）、LLM PR。
- **主要结果**：
  -  hardest任务上GPT-4.1的Final PR仅**0.5%**，凸显任务挑战性；UPC对齐是随难度上升的核心瓶颈。
  - B2T-Agent全面领先：Qwen3-8B版本平均Final PR达**4.7%**，超越GPT-4.1的**3.2%**；Qwen3-14B在Hard集上Final PR达**5.0%**，LLM PR达**74.3%**（GPT-4.1为28.2%）。
  - 奖励驱动显著优于SFT：Hard集UPC Macro PR上，B2T-14B为**5.0%**，而SFT-14B仅**0.3%**，证明RL能有效捕捉动态个性化偏好。
- **泛化验证**：在TravelPlanner上，Qwen3-8B-B2T-Agent的FPR达**9.0%**（GPT-4.1为1.5%），CMa达**25.0%**（GPT-4.1为5.7%），验证跨基准迁移能力。
- **消融实验**：移除Memory模块使Final/LLM PR下降约11%；移除Penalty使DR下降10.2%；移除Loss Mask导致DR/Final/LLM PR均暴跌约19-20%。

## 相关工作脉络
- **旅行规划基准**：TravelPlanner、ChinaTravel、Ask-before-plan等均采用显式指令或澄清对话范式，无法评估隐式偏好推断；Behavior2Trip填补了低交互负担与行为轨迹输入的评测空白。
- **端到端LLM智能体**：ReAct、TravelPlanner等依赖手工提示词驱动工具调用，缺乏跨多步交互的自适应能力；B2T-Agent通过RL学习完整的工具-记忆协同策略。
- **混合LLM-符号求解器**：如TRIP-PAL、ITINERA仅在最终规划阶段引入符号约束求解，检索与信息收集过程仍由LLM自由发挥；B2T-Agent将约束遵循融入全过程的奖励信号中。
- **用户行为推荐**：传统序列推荐（如Next POI推荐）聚焦短时交互预测，本文将其扩展至长视距、多约束、强时空耦合的旅行规划场景。
- **强化学习Agent训练**：继承Search-R1、Kimi K1.5等RL训练思路，但针对旅行规划特有的工具调用错误与上下文累积问题，设计了专用动作惩罚与Loss Mask策略。

## 局限性与未来方向
- **训练计算开销大**：GRPO需对每个查询采样多条rollout，显著增加训练成本。
- **推理首Token延迟高**：多轮工具调用与记忆交互导致响应延迟较大，影响实时用户体验。
- **上下文长度限制**：长轨迹与多轮交互易超20k token，模型在复杂上下文中提取有效信号的能力受限。
- **未来方向**：优化 rollout 采样效率、探索压缩记忆表示与并行工具调用机制，以在不牺牲规划质量的前提下降低计算与延迟开销。

## 研究启发与可借鉴点
1. **隐式偏好推断范式**：将用户历史行为序列作为规划输入，可大幅降低显式交互成本，该思路可迁移至购物清单生成、日程安排等个性化决策场景。
2. **结构化动作空间设计**：将工具调用、记忆读写与思维链统一为序列化动作，并结合Loss Mask屏蔽环境token，为长视距Agent的RL训练提供了稳定可靠的训练范式。
3. **多级门控奖励机制**：先校验格式合法性，再分项评估常识与个性化约束，有效避免了早期错误扩散，适用于其他强约束生成任务。
4. **沙盒化真实数据构建流程**：从真实平台脱敏数据出发，经偏好映射→行为模拟→CoA标注→双阶段质检的流水线，可复用于其他垂直领域Agent基准构建。
5. **跨基准泛化验证策略**：在同一模型上同步评测行为2Trip与TravelPlanner，证明方法不仅解决新问题，且能提升通用规划能力，增强了工作的说服力。

## 关键术语表
- **Behavior-Aware Travel Planning**：通过解析用户历史行为轨迹隐式推断偏好，无需显式指令即可生成个性化旅行计划的新任务。
- **Commonsense Constraint (CC)**：要求旅行计划满足格式规范、POI真实性、时空合理性与信息完整性的基础约束集合。
- **User Preference Constraint (UPC)**：要求生成的计划必须与从行为轨迹中推断出的用户个性化偏好（住宿、餐饮、交通等）保持一致的约束。
- **B2T-Agent**：基于GRPO强化学习训练的旅行规划智能体，支持外部工具调用与内部键值记忆管理的双模块架构。
- **Group Relative Policy Optimization (GRPO)**：无需独立价值网络的策略梯度算法，通过组内相对优势估计稳定更新策略模型。
- **Chain of Action (CoA)**：记录智能体完整规划过程的监督信号，包含多步工具调用、记忆读写与中间推理决策。
- **Loss Mask**：在训练损失计算中屏蔽外部工具响应与记忆读取结果等非模型生成token，仅对自生成部分回传梯度。
- **Delivery Rate (DR)**：衡量模型成功输出结构完整旅行计划的比例，反映基础可用性。

## 可复现要素
- **数据集**：Behavior2Trip（11,400实例，含9,000训练/2,400测试），论文未明确声明是否开源。
- **代码/权重**：论文未公开代码与模型权重；训练基于RLFactory框架，SFT基线基于LLaMAFactory。
- **关键超参**：未详细披露GRPO的 rollout 数量、学习率、$\beta$系数及训练步数；工具集含9个API，约束共32项。
- **基础模型**：Qwen3-8B/14B/32B、DeepSeek-V3、GPT-4o、GPT-4.1。
