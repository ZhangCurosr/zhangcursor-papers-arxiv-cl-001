---
title: "J-ZERO-UNIFIED-CHALLENGER-SOLVER-JUDGE-CO-EVOLUTION-FROM-ZER"
source: https://arxiv.org/pdf/2608.26582v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:26:40"
field: "大语言模型自进化与自我改进"
keywords: ["self-evolving LLM", "zero-data self-play", "reward model co-adaptation", "challenger-solver-judge", "unverifiable domain", "preference optimization"]
innovations: ["Challenger-Solver-Judge 三者协同进化的零数据统一框架", "基于角色不对称和子任务放大的循环内偏好对构造方法", "突破固定 Judge 性能天花板实现十轮以上持续改进"]
benchmarks: ["GSM8K", "MATH500", "AIME24", "MMLU-Pro", "AlpacaEval 2.0", "Arena-Hard-v2.0"]
---

# 论文速读：J-ZERO: UNIFIED CHALLENGER–SOLVER–JUDGE CO-EVOLUTION FROM ZERO DATA

## 一句话总结
论文提出了 J-ZERO，一种零数据自进化框架，通过 Challenger-Solver-Judge 三者的协同进化，在可验证和不可验证领域均实现了稳定、持续的自我改进，突破了固定 Judge 造成的性能天花板。

## 研究问题与动机
1. **人工监督成本高**：依赖人工标注员设计任务和提供标签是开发超越人类智能的 AI 系统的基本瓶颈。
2. **不可验证领域自进化探索不足**：现有自进化算法主要在可验证领域（有客观正确答案）取得进展，不可验证领域因缺乏单一正确答案而难以应用。
3. **固定 Judge 构成性能天花板**：冻结的 Judge 只能推动 Solver 向其已内化的偏好靠拢，一旦 Solver 达到 Judge 的判别极限，进一步训练便无法获得学习信号。
4. **零数据方法局限于可验证领域**：Absolute Zero、R-Zero 等无数据自进化方法主要依赖可执行反馈或多数投票，难以扩展到开放生成任务。

## 核心贡献（创新点）
1. **提出 Challenger-Solver-Judge 协同进化统一框架**：J-ZERO 允许 Judge 在自我对弈循环中同步适应，打破固定 Judge 的性能上限，使可验证和不可验证领域都能持续改进。
2. **构造两类循环内偏好对训练 Judge**：角色不对称对（Role-asymmetry pairs）和子任务放大对（Subtask-amplification pairs）的标签由过程结构确定而非 Judge 自身打分，避免偏差强化。
3. **实现多轮持续改进而非早期饱和**：J-ZERO 在至少十轮迭代中持续进步，而基线方法（R-Zero、G-Zero）在两轮后即出现性能下降。
4. **在不损害通用奖励建模能力的前提下提升 Judge 判别力**：协同进化的 Judge 在 RM-Bench 上 Hard 难度对的准确率提升 4.77 分，整体能力提升同时保持稳定性。

## 方法详解
**整体流程**：每轮迭代包含三个阶段——Challenger 生成更具挑战性的任务（最小化 Judge 对 Solver 的奖励）、Solver 学习回答这些任务（最大化 Judge 奖励）、Judge 通过偏好对更新（Brady-Terry loss）。

**Challenger 设计**：
- 利用 GRPO 优化，目标函数为最大化任务的"难度奖励" $1 - \bar{r}_i$（$\bar{r}_i$ 为 Solver 响应的平均 Judge 得分）。
- 加入重复惩罚 $r_i^{\text{rep}}$（基于 BLEU 距离聚类）和格式检查（需包含 `<question>` 标签），复合奖励为 $r_i^C = \max(0, 1 - \bar{r}_i - r_i^{\text{rep}})$。

**Solver 任务选择与优化**：
- 从候选任务中选择 Judge 评分离散度 $s_i = \text{std}(\{r_{i,j}^S\}_{j=1}^M)$ 最大的 top-K 任务，确保训练信号最丰富。
- 使用 GRPO 最大化 Judge 奖励：$\mathcal{R}_S = \mathbb{E}_{x \sim C_{\theta_c}} \mathbb{E}_{y \sim S_{\theta_s}(\cdot|x)}[\sigma(J_\phi(x,y))]$。

**Judge 自适应训练**：
- **角色不对称对** $\mathcal{D}_{\text{role}}$：对保留任务 $x$，chosen 来自 Solver $y^S$，rejected 来自 Challenger $y^C$（因 Solver 被优化回答问题而 Challenger 被优化制造困难）。
- **子任务放大对** $\mathcal{D}_{\text{amp}}$：Challenger 将 $x$ 分解为子任务 $\{q_k\}$，Solver 逐一回答后由 Challenger 组合为 $y^{\text{amp}}$，与 Solver 的单步回答 $y^S$ 构成偏好对（divide-and-conquer 更易准确）。
- Judge 通过 BT loss 更新：$\mathcal{L}_J = -\mathbb{E}[\log \sigma(J_\phi(x,y^+) - J_\phi(x,y^-))]$。

## 实验与结果
**实验设置**：基于 Qwen3-4B-Base 和 Qwen3-8B-Base，Judge 使用 Skywork-Reward-V2-Llama-3.1-8B，对比基线为 R-Zero 和 G-Zero。评测涵盖 11 个可验证基准（7 个数学推理 + 3 个通用推理 + IFEval）和 3 个不可验证基准（AlpacaEval 2.0、Arena-Hard-v2.0、EQ-Bench Creative Writing v3）。

**可验证领域结果**：
- J-ZERO（Qwen3-4B）在数学推理、通用推理、指令遵循上分别达 51.20、52.93、58.99，总体 54.38，较 Base 提升 9.47 分，较 R-Zero 提升 4.74 分。
- J-ZERO（Qwen3-8B）总体达 58.55，较 Base 提升 7.88 分，较 R-Zero 提升 3.56 分。
- AIME24/AIME25 提升尤为显著：4B 从 8.96/6.67 → 16.15/15.83。

**不可验证领域结果**：
- J-ZERO（4B）总体 20.81，较 Base 提升 11.23 分，较 R-Zero（12.66）提升 8.15 分。
- AlpacaEval 2.0 改善最大：4B 从 6.22 → 28.56，8B 从 12.93 → 33.53。
- G-Zero 在无 Judge 情况下几乎无提升（4B 仅 10.89），凸显 Judge 的重要性。

**迭代分析**：J-ZERO 在前 10 轮持续改进，而 R-Zero/G-Zero 在第 2 轮即达到峰值后下降；冻结 Judge 变体在第 3 轮后平台化，验证了 Judge 协同进化的关键作用。

## 相关工作脉络
1. **R-Zero (Huang et al., 2026b)**：零数据自对弈框架，利用多数投票获取可验证任务的奖励信号，但未扩展到不可验证领域。本文相对 R-Zero 的核心差异在于引入了 Judge 协同进化，使其能统一处理两类领域。
2. **G-Zero (Huang et al., 2026a)**：同样针对不可验证领域的零数据自对弈，但使用 Challenger 生成的提示构建偏好对仅训练 Solver，不更新 Judge。本文进一步让 Judge 直接随 Solver 能力提升而进化。
3. **Kuba et al. (2025) Language Self-Play**：首次将数据自由自进化扩展到不可验证领域，但依赖静态 Judge，本文通过 co-adaptation 突破其性能天花板。
4. **Self-rewarding methods (Yuan et al., 2024; Prasad et al., 2025)**：使用模型自身生成的最高/最低奖励响应作为偏好对，存在强化 Judge 自身偏差的风险；本文的偏好对标签由过程结构保证，不依赖 Judge 打分。
5. **Absolute Zero (Zhao et al., 2025)**：基于执行器的验证信号用于代码生成，属于严格的可验证领域方法，无法直接应用于开放式文本生成任务。

## 局限性与未来方向
1. **规模限制**：受计算约束，仅测试了 8B 参数的基座模型，尚未验证更大规模或经过 post-training 的推理模型（如支持长思维链的模型）。
2. **Judge 架构单一**：当前 Judge 为基于分类器的判别式奖励模型，与 Challenger/Solver 的生成式初始化不同；未来可探索生成式 Judge（如 LLM-as-a-judge）使三者共享同一基座模型。
3. **长期可持续性待验证**：当前实验仅观察到 10 轮迭代，更长周期的性能趋势尚不明确。

## 研究启发与可借鉴点
1. **偏好对标签的结构化构造思路**：利用角色分工（Challenger vs. Solver 的目标差异）和分解-组合（subtask amplification）的过程结构生成可靠偏好对，无需外部监督，这一设计可直接迁移至其他需要偏好信号的自进化场景。
2. **任务筛选的方差启发式**：选择 Solver 响应分散度（std）最高的任务作为训练样本，等价于选择位于当前能力边界的"最具学习价值"任务，这一连续化替代 R-Zero 的二值 accuracy 过滤的思路值得借鉴。
3. **多信号互补的时间动态设计**：Role-asymmetry 对在整个训练过程中可靠，而 subtask-amplification 对需 Solver 成熟后才有效（前 3 轮胜率 <50%，后续升至 70-80%），两者的切换覆盖了训练的各阶段，这种动态互补设计可用于其他多组件联合优化的场景。
4. **Judge 作为可训练组件的定位**：将评估器而非视为固定工具，而是与策略模型同步更新的观点，为奖励模型的自我改进提供了新范式，可启发后续的"元评估"研究。

## 关键术语表
**J-ZERO**：Judge co-adaptation from ZERO data，一种 Challenger-Solver-Judge 三者协同进化的零数据自改进框架。
**Challenger**：负责生成越来越困难的训练任务的策略模型。
**Solver**：负责学习回答 Challenger 生成任务的策略模型。
**Judge**：负责为任务-响应对打分并提供奖励信号的评估模型，在 J-ZERO 中与其他两模型同步更新。
**GRPO（Group Relative Policy Optimization）**：基于组内相对优势的强化学习策略优化算法，本文用于 Challenger 和 Solver 的策略更新。
**Role-asymmetry pairs**：利用 Challenger 和 Solver 目标差异构造的偏好对，Solver 响应天然优于 Challenger 响应。
**Subtask-amplification pairs**：通过 divide-and-conquer 策略将难题分解后求解再组合，所得响应质量高于 Solver 单步回答，用于训练 Judge 识别更高质量的输出。
**Bradley-Terry (BT) loss**：用于偏好对排序的对比损失函数，最大化 chosen 相对于 rejected 的得分差。

## 可复现要素
- **数据集**：零数据自生成，无外部训练数据依赖；评测使用公开基准（GSM8K、MATH500、Minerva、OlympiadBench、AMC23、AIME24/25、MMLU-Pro、SuperGPQA、BBH、IFEval、AlpacaEval 2.0、Arena-Hard-v2.0、EQ-Bench Creative Writing v3）。
- **代码/权重**：项目页面和 GitHub/Hugging Face 链接已在论文中标注（具体 URL 见原文），权重开源情况未明确说明。
- **关键超参**：每轮 Challenger 训练 5 步、Solver 15 步、Judge 8 步；batch size 均为 16；learning rate 为 $1 \times 10^{-6}$（Challenger/Solver）和 $5 \times 10^{-7}$（Judge）；KL 惩罚系数 $\beta = 0.01$；rollout temperature = 1.0，top-p = 0.99；clip ratio = (0.20, 0.28)；max length Prompt 1024/4096，Response 4096。训练硬件为 4×NVIDIA B200 + 4×NVIDIA H200。
