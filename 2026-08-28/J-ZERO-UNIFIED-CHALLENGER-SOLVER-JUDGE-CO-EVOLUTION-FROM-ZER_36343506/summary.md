---
title: "J-ZERO-UNIFIED-CHALLENGER-SOLVER-JUDGE-CO-EVOLUTION-FROM-ZER"
source: https://arxiv.org/pdf/2608.26582v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:26:56"
field: "大语言模型自进化与对齐"
keywords: ["self-evolving LLM", "zero-data self-play", "challenger-solver-judge", "reward model co-adaptation", "preference optimization", "unverifiable domain"]
innovations: ["提出 Challenger-Solver-Judge 三方协同进化框架，同时覆盖验证域与不可验证域", "利用角色不对称与子任务放大两类结构偏好对训练 Judge，避免自强化偏差", "基于响应分散度的任务选择机制，适配连续 Judge 奖励信号"]
benchmarks: ["GSM8K", "MATH500", "AlpacaEval 2.0", "Arena-Hard-v2.0", "RM-Bench"]
---

# 论文速读：J-ZERO: UNIFIED CHALLENGER–SOLVER–JUDGE CO-EVOLUTION FROM ZERO DATA

## 一句话总结
提出 J-ZERO 框架，通过 Challenger–Solver 对抗博弈与 Judge 协同适应，在无外部数据条件下实现验证域与不可验证域的自进化；Judge 利用循环内构造的角色不对称对与子任务放大对进行 BT 更新，使评估信号持续提升，从而突破固定 Judge 的性能天花板，在多个基准上取得显著且稳定的迭代改进。

## 研究问题与动机
- **验证域之外的自进化缺失**：现有零数据自博弈方法（如 R-Zero）主要依赖可执行验证或多数投票奖励，难以推广到开放式、无单一正确答案的不可验证域。
- **固定 Judge 的性能天花板**：在不可验证域中使用静态奖励模型会限制 Solver 上限——Solver 一旦饱和 Judge 当前判别能力，训练信号便消失，导致迭代两三轮后退化。
- **循环内偏好标注的可信性难题**：若偏好对完全由 Judge 自身打分生成，容易强化既有偏差；需要不依赖 Judge 评分、仅在闭环内构造的可靠监督信号。
- **评估信号本身需共同进化**：为支持持续改进，评价模型应与策略模型同步适应最新能力前沿，而非停留在初始校准状态。

## 核心贡献（创新点）
- **统一框架**：提出 Challenger–Solver–Judge 三方协同进化的零数据自博弈框架，同时覆盖验证域与不可验证域。与已有工作在仅使用固定 Judge 或仅支持验证域方面的本质区别在于将评估信号也纳入迭代更新。
- **角色不对称偏好对**：利用 Solver 与 Challenger 训练目标的结构性差异，直接构造 $y^S \succ y^C$ 的偏好对，其标签不依赖 Judge 当前得分，可在 Judge 不确定区域重新注入判别信号。
- **子任务放大偏好对**：借鉴迭代放大思想，由 Challenger 分解任务、Solver 分步解答后再组合，构造 $y^{\text{amp}} \succ y^{S}$ 对，暴露 Judge 于超越 Solver 单步回答的质量前沿，避免 Judge 饱和于当前策略水平。
- **基于分散度的任务选择**：以 Solver 响应 Judge 评分的标准差筛选最具训练信息的任务，作为连续版“ informative band"，比 R-Zero 基于二值准确率的筛选更适配连续奖励信号。
- **持续多轮提升**：实验表明 J-ZERO 在至少 10 轮迭代中持续改进，而基线在第 2 轮后下降，验证了 Judge 协同进化对维持判别信号的关键作用。

## 方法详解
- **三角色设定**：Challenger $C_{\theta_c}$ 生成任务，Solver $S_{\theta_s}$ 作答，Judge $J_{\phi}$ 对 (任务, 回答) 打分并输出标量奖励 $r=\sigma(J(x,y))$。
- **Challenger–Solver 对抗博弈**：Challenger 最小化 Judge 对 Solver 响应的平均得分，Solver 最大化该得分；双方均使用 GRPO 进行策略更新，并加入 KL 正则项 $\beta \mathbb{D}_{KL}[\cdot\|\cdot_{\text{ref}}]$ 防止偏移过大。
- **Challenger 复合奖励**：任务难度奖励为 $1-\bar{r}_i$，辅以重复惩罚 $r_i^{\text{rep}}=\lambda |\mathcal{C}_k|/N$（基于 BLEU 聚类）和格式检查（必须含 `<question>` 标签），得到 $r_i^C=\max(0,1-\bar{r}_i-r_i^{\text{rep}})$ 等。
- **基于响应对比度的任务选择**：冻结 Challenger 后采样候选任务集，对每个任务让 Solver 生成 M 个回答并评分，取响应得分标准差 $s_i=\text{std}(\{r^S_{i,j}\})$ 最大的前 K 个任务用于 Solver 训练；理论依据为策略改进下界与奖励方差正相关。
- **Judge 协同适应**：每轮结束后基于两个偏好数据集更新 Judge：
  - **角色不对称对** $\mathcal{D}_{\text{role}}=\{(x,y^S,y^C)\}$，标签由角色职责决定：Solver 被优化来作答，Challenger 被优化来出题，因此 $y^S\succ y^C$。
  - **子任务放大对** $\mathcal{D}_{\text{amp}}=\{(x,y^{\text{amp}},y^S)\}$：Challenger 将 $x$ 分解为 3–5 个子任务，Solver 在原始任务语境下逐一回答，Challenger 再组合为完整回答 $y^{\text{amp}}$，因其基于更易的子任务求解，故 $y^{\text{amp}}\succ y^S$。
- **BT 损失更新**：$\mathcal{L}_J(\phi)=-\mathbb{E}[\log\sigma(J(x,y^+)-J(x,y^-))]$，两组偏好对等比例混合；训练集中在当前能力前沿最难区分的样本上。
- **迭代流程**：每轮依次训练 Challenger 5 步、Solver 15 步、Judge 8 步；当任一域平均分开始下降时停止并回退至最佳 checkpoint。

## 实验与结果
- **模型与基线**：基于 Qwen3-4B-Base 与 Qwen3-8B-Base；基线为 Base、R-Zero、G-Zero；Judge 使用 Skywork-Reward-V2-Llama-3.1-8B。
- **验证域基准**：GSM8K、MATH500、Minerva、OlympiadBench、AMC23、AIME24、AIME25、MMLU-Pro、SuperGPQA、BBH、IFEval。J-ZERO 在 Qwen3-4B 上平均提升 9.47 分（44.91→54.38），在 Qwen3-8B 上提升 7.88 分（50.67→58.55）；相对 R-Zero 分别提升 4.74 与 3.56 分。
- **不可验证域基准**：AlpacaEval 2.0、Arena-Hard-v2.0（H.P. 与 C.W. 子集）、EQ-Bench Creative Writing v3。J-ZERO 在 Qwen3-4B 上平均提升 11.23 分（9.58→20.81），在 Qwen3-8B 上提升 10.18 分（13.23→23.41）；AlpacaEval 2.0 从 6.22 跃升至 28.56（4B）和 12.93 跃升至 33.53（8B）。
- **最强结果**：4B 模型 Overall 验证域 54.38、不可验证域 20.81；8B 模型验证域 58.55、不可验证域 23.41。
- **迭代稳定性**：J-ZERO 在前 10 轮单调提升；R-Zero 与 G-Zero 均在第 2 轮达峰后下降。冻结 Judge 变体在第 3 轮后开始 plateau，分别落后完整方法 1.66（验证域）与 4.44（不可验证域）分。
- **偏好对可靠性**：Role-asymmetry 对胜率从 87.9% 缓慢降至约 66%，始终高于 60%；Subtask-amplification 对前 3 轮低于 50%（迭代 1 仅 21.1%），第 4 轮后回升至 70–80%，两曲线在中段交叉，保证全程有有效监督。
- **Judge 独立评测**：RM-Bench 平均准确率从 92.61 升至 93.95（+1.34），Hard 对准确率从 85.08 升至 89.85（+4.77）。

## 相关工作脉络
- **R-Zero**：数据驱动的零数据自博弈方法，用多数投票替代外部验证器，适用于验证域但奖励信号难以泛化到不可验证域；本文使用连续 Judge 评分与偏好学习扩展其适用范围。
- **G-Zero**：采用 Challenger 生成的提示辅助构建 Solver 响应间的偏好对，并用 DPO 训练 Solver，但不显式维护 Judge 的协同进化；本文强调 Judge 自身能力必须同步提升才能维持信号判别力。
- **Absolute Zero / Tool-R0 / Dr. Zero**：同样面向验证域，依赖执行器、工具调用或搜索结构获取硬反馈；本文聚焦无需任何外部资源且兼顾开放生成的统一场景。
- **Self-Rewarding 系列**（Yuan et al., 2024 等）：让模型既作策略又作裁判，存在自我强化偏差风险；本文偏好对的标签由结构不对称性决定而非 Judge 评分，规避该问题。
- **Kuba et al. (2025)**：首个将无数据自进化扩展到不可验证域的工作，但仍使用静态 Judge；本文通过协同适应打破其上限。
- **Iterated Amplification（Christiano et al., 2018）**：经典弱学习者逐步增强范式；本文将其思想工程化为子任务分解-回答-组合流水线，并作为 Judge 训练的偏好来源之一。

## 局限性与未来方向
- **规模与模型类型受限**：实验仅覆盖至 8B 基础模型，未测试后训练推理模型（尤其是输出长 CoT 的模型）及更大参数规模。
- **Judge 架构为判别式分类器**：当前 Judge 基于 Skywork-Reward-V2 初始化并使用 BT 损失，若采用生成式 Judge（LLM-as-judge）使单一基础模型担任三角色，可获得更丰富的反馈，但如何将其纳入闭环协同进化仍待探索。
- **计算约束**：仅使用 8B 参数的 Challenger/Solver 与 8B Judge，硬件 budget 限制了更长训练周期与更多rollout。
- **偏好对质量随难度上升而衰减**：Role-asymmetry 对胜率随迭代下降（87.9%→66%），反映前沿任务上两者能力趋近；虽不破坏整体效果，但提示极端难度下需补充更强的监督来源。

## 研究启发与可借鉴点
- **结构不对称优于模型自评分**：利用不同角色在目标函数中的结构性差异构造偏好标签，可避免自我强化循环；该思路可迁移至其他需要无外部监督信号的多 agent 系统。
- **迭代放大作为偏好源泉的工程化落地**：将 "decompose → solve → compose" 流程作为训练数据的自动生成器，既产出高质量回答，又提供明确的优劣关系，值得在其他生成任务中复用。
- **基于响应方差的任务筛选**：用 Solver 多组响应的评分标准差衡量任务信息量，能自然适配连续奖励设定；在 RLHF/DPO 等连续评分场景中可作为有价值的 curriculum 组件。
- **Judge 独立能力的正向溢出**：协同进化不仅服务于闭环训练，还能同步提升 RM-Bench 上的判别性能（尤其 Hard 对），说明评估模型在目标分布上的持续校准具有普适价值。
- **与团队方向的结合机会**：若团队关注长程推理或工具使用，可将子任务放大思想与 Tree-of-Thought / SWE-RL 等结构结合，构造分层偏好对以提升复杂任务下的 Judge 判别力。

## 关键术语表
- **J-ZERO**：Challenger–Solver–Judge 三方零数据协同进化框架，支持验证域与不可验证域的自改进。
- **GRPO（Group Relative Policy Optimization）**：以组内相对优势进行策略梯度的强化学习更新方法，本文用于 Challenger 与 Solver 的训练。
- **角色不对称对（Role-asymmetry pairs）**：由 Solver 与 Challenger 回答同一任务形成的偏好对，标签源自二者训练目标的结构性差异而非 Judge 打分。
- **子任务放大对（Subtask-amplification pairs）**：基于迭代放大思想，将难任务分解后由 Solver 分步解答再组合，与单步回答构成偏好对。
- **Bradley–Terry（BT）损失**：用于序贯偏好数据训练的负对数似然损失，驱动 Judge 参数更新。
- **Informative band / 响应分散度筛选**：以 Solver 响应的评分标准差筛选最有利于学习的任务，替代二值准确率的早期任务选择策略。
- **AlpacaEval 2.0**：长度控制的自动评测基准，衡量模型在开放指令任务上相对于 GPT-4-Turbo 的胜率。
- **Arena-Hard-v2.0**：来源于 LMSYS Chatbot Arena 的高难度评测集，包含 Hard Prompt 与 Creative Writing 子集。

## 可复现要素
- **数据集**：无外部训练数据；评测使用 GSM8K、MATH500、Minerva、OlympiadBench、AMC23、AIME24、AIME25、MMLU-Pro、SuperGPQA、BBH、IFEval、AlpacaEval 2.0、Arena-Hard-v2.0、EQ-Bench Creative Writing v3、RM-Bench。
- **代码**：论文提供 Project Page 与 GitHub/Hugging Face 链接（论文未给出具体 URL，但标注了资源位置）。
- **权重**：基于 Qwen3-4B-Base 与 Qwen3-8B-Base；Judge 使用 Skywork-Reward-V2-Llama-3.1-8B；论文未声明开源训练后权重。
- **关键超参**：Challenger 5 步/轮、batch=16；Solver 15 步/轮、batch=12（mini=16）；Judge 8 步/轮、batch=64；学习率 $1\times10^{-6}$（C/S）与 $5\times10^{-7}$（J）；KL 系数 0.01；rollout 温度 1.0、top-p 0.99；clip 范围 (0.20, 0.28)；偏好对两种类型等比例混合。
