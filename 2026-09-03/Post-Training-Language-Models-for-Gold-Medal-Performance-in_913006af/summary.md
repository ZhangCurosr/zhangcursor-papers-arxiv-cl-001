---
title: "Post-Training-Language-Models-for-Gold-Medal-Performance-in"
source: https://arxiv.org/pdf/2609.02849v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 22:36:53"
field: "代码智能与算法编程"
keywords: ["competitive programming", "post-training", "supervised fine-tuning", "reinforcement learning", "test-time compute", "code generation", "IOI", "GRPO"]
innovations: ["提出端到端竞赛编程后训练流水线（SFT+RL+GenCorrect），在小参数模型上达到 IOI 金牌", "引入 GenCorrect 反馈驱动测试时计算策略，通过子任务累积评分实现多轮定向精炼", "在 IOI 2026 实时比赛中以 535.4/600 超越最高人类选手，首次有 AI 系统做到"]
benchmarks: ["IOI 2025", "IOI 2026", "ICPC 2025", "LiveCodeBench Pro"]
---

# 论文速读：Post-Training-Language-Models-for-Gold-Medal-Performance-in-Coding-Competitions

## 一句话总结
本文提出了一套端到端的竞赛编程专用后训练流水线，结合大规模数据筛选、合成推理轨迹、监督微调（SFT）和强化学习（RL），并引入 GenCorrect 这一反馈驱动的测试时计算策略；在 IOI 2025 和 IOI 2026 上分别斩获金牌成绩与超越最高人类选手的 535.4/600 分。

## 研究问题与动机
1. **核心问题**：如何通过后训练让通用大语言模型达到国际信息学奥林匹克（IOI）和 ICPC 竞赛的金牌水平？
2. **现有方法不足**：已有达到金牌的系统多为闭源、依赖专用模型或特定硬件，且训练数据、后训练方法、模型规模与推理时算力的贡献难以隔离分析；开源工作往往缺乏完整的端到端训练-推理闭环验证。
3. **评测缺口**：IOI 级别任务需要模型在时限和提交次数约束下实时解决问题，而多数代码基准（如 HumanEval、MBPP）偏重短函数生成，无法衡量算法综合推理能力。
4. **动机**：构建一个可复现、全开源的竞赛编程后训练流水线，系统量化 SFT、RL、模型规模和测试时计算各自的贡献，并在真实比赛中验证。

## 核心贡献（创新点）
1. **端到端竞赛专用后训练流水线**：结合 22,000 道题的筛选、120 万条合成推理轨迹、SFT 与可执行奖励 GRPO 的 RL，首次在小参数模型（Nano-CC，3B 活跃参数）上达到 IOI 金牌。
   *与已有工作的本质区别*：此前金牌系统（如 o1-ioi、Gemini）多为闭源且未披露完整训练细节；本文提供了从数据到推理的全链路开源可复现方案。
2. **GenCorrect 反馈驱动测试时计算策略**：在最多 50 次提交限制下，通过并行生成-聚类多样性选择-执行评估-子任务反馈累积的四步迭代循环，将 200 个候选解集中到 10 次有效提交。
   *与已有工作的本质区别*：区别于 AlphaCode/GenCluster 仅做一次性大规模采样，GenCorrect 利用部分分的子任务级反馈进行多轮定向修正，显著降低所需生成量。
3. **消融分析揭示各组件贡献**：系统量化了 SFT、RL 和测试时计算的各自增益——SFT 贡献最大（Nano-CC 从 130→280 分），RL 带来小幅稳定提升（280→291），GenCorrect 将最终分数推至 468（超金牌线）。
   *与已有工作的本质区别*：此前研究多报告单一系统结果，本文首次在同一模型族上对比 SFT-only vs SFT+RL vs SFT+RL+GenCorrect 的逐阶段进展。
4. **IOI 2026 实时竞赛验证**：基于 IOI 2025 作为开发集调优系统，在 IOI 2026 正式比赛期间、与人类选手同等时间/网络/提交限制下实时运行，以 535.4/600 超越最高人类选手（498.27），号称首次有 AI 系统在 IOI 题集上击败人类最高分。
   *与已有工作的本质区别*：此前所有竞赛评测均为事后离线测试，本文是在比赛进行中实时完成，具有严格的公平性约束。

## 方法详解
### 3.1 数据筛选（Data Curation）
- 从 16 个区域/国际竞赛家族（近 20 年）及在线编程平台收集 22,000 道题。
- 自动化管道将每道题封装为可执行评估环境（含题目描述、约束、测试用例、辅助文件、参考解）。
- 过滤条件：参考解与生成解判决一致；排除 IOI 2025、ICPC 2025、LiveCodeBench Pro 的题目并做去重；RL 阶段额外排除评估耗时超过 300 秒的题目，最终保留 3,219 题（2,847 训练 / 372 验证）。

### 3.2 监督微调（SFT）
- 使用 DeepSeek-V4-Flash 为 Nano（30B-A3B）生成 120 万条推理轨迹，为 Ultra（550B-A55B）生成 477,642 条。
- 轨迹包含自改进样本（teacher 在已有解基础上生成改进版本），使模型暴露于 GenCorrect 所用的迭代优化行为。
- Nano-CC：3 个 epoch，global batch size=64，序列打包至 262K tokens；Ultra-CC：1 个 epoch，同样设置，初始化自 RLVR-teacher checkpoint。
- 关键超参：Nano-CC 学习率 5×10⁻⁵（constant），Ultra-CC 1.5×10⁻⁵（cosine，warmup=0.1）；硬件为 64 块 NVIDIA GB300 GPU。

### 3.3 强化学习（RL）
- 仅对 Nano-CC 应用 RL，Ultra 因算力限制未做 RL。
- 使用 NeMo RL + GRPO 算法，每步 64 prompts × 16 rollouts = 1,024 rollouts，temperature=1.0，最大序列长度 262K tokens。
- 奖励机制：生成 C++17 代码后编译执行，获得满分得 1，否则得 0（二值终态奖励，无中间部分分奖励）。
- 目标函数：token-level clipped policy gradient，无 reference policy KL penalty；以留一法（leave-one-out）计算组内相对优势。
- 最终选择 step 39 checkpoint（在 held-out validation set 上最优）。

### 3.4 GenCorrect 测试时计算
每轮迭代含四个阶段：
1. **生成**：并行生成最多 200 个候选解（首轮只用题目；后续轮加入之前轮的解和反馈）。
2. **多样性选择**：过滤无效输出后，用分数无关的局部启发式 $Q(c)$ 初始化中心集 $C$，迭代选取与现有中心余弦相似度最小的候选作为新中心（公式 1）：
   $$c_{\text{next}} \in \arg\max_{c \notin C} \min_{z \in C}[1 - \sin(c, z)]$$
   选出最多 10 个中心后，将候选分配到最近中心，每簇选 $Q(c)$ 最高的代表。
3. **执行**：提交 10 个代表解，IOI 获得子任务分数，ICPC 获得二值通过/失败反馈。
4. **精炼**：累积每子任务历史最高分（公式 2）：
   $$A_r(t) = \max(A_{r-1}(t), \max_{c \in S_r} s_t(c)), \quad A_0(t) = 0$$
   下一轮以累积的子任务分数向量 $A_r$ 为条件，辅以三个互补参考解（最强整体解、最大子任务缺口解、广覆盖剩余缺口的解），引导模型定向攻克未解决子任务。

IOI 场景：最多 5 轮 × 10 次提交 = 50 次，匹配官方限制；ICPC 场景：持续至问题解决或性能饱和。

## 实验与结果
### 评测基准
- **IOI 2025**：6 题 × 100 分 = 600 分满分，金牌线 438.3 分
- **ICPC 2025 World Finals**：12 题，二元评分，5 小时限时
- **LiveCodeBench Pro**：持续更新的编程评测基准

### 主要结果（单样本 Score@1 / Pass@1）

| 模型 | IOI 2025 Score@1 | ICPC 2025 Pass@1 | LCB Pro Pass@1 |
|---|---|---|---|
| Nemotron-3-Nano-30B-A3B（基座） | 21.7% | 16.9% | 17.6% |
| Nemotron-3-Nano-CC（SFT+RL） | 48.5% | 51.0% | 71.6% |
| Nemotron-3-Ultra-550B-A55B（基座） | 45.5% | 54.0% | 72.6% |
| Nemotron-3-Ultra-CC（SFT only） | 50.7% | 57.4% | 74.5% |
| GLM-5.2 | 66.0% | 65.7% | 83.8% |
| DeepSeek-V4-Pro | 56.8% | 69.6% | 78.2% |

### GenCorrect 渐进提升（IOI 2025）
- Nano-CC：Score@1 = 291（SFT+RL），Score@200 = 461；GenCorrect 5 轮后均值 468.2，**超过金牌线 438.3**。
- Ultra-CC：Score@1 = 304（SFT only），GenCorrect 5 轮后均值 502.0，**第 3 轮即超金牌线**。

### IOI 2026 实时竞赛结果
- Competition Ultra-CC（NVFP4 量化 + 竞赛适配）：**535.4 / 600**，超金牌线 361.12 达 174.3 分，超最高人类选手 498.27 达 37.1 分。
- 通用 GenCorrect 管线事后均值 521.72（495.0–545.8），实时结果高出 13.68 分。

### 消融结论
- SFT 贡献最大：Nano-CC 经 3 epoch SFT 后 IOI 从 21.7% 升至 47.3%，ICPC 从 16.9% 升至 46.7%。
- RL 贡献较小但跨基准稳定：IOI 46.7% → 48.5%，主要因可执行奖励的二值稀疏性导致长时信分配困难。
- 模型规模 vs 训练强度：Ultra-CC 仅需 1 个 epoch SFT 即超越 Nano-CC 的全部三个基准，表明在算力允许时"强基座 + 轻量 SFT"优于"弱基座 + 重度后训练"。

## 相关工作脉络
1. **AlphaCode / AlphaCode 2**（Li et al., 2022; Leblond et al., 2023）：开创性地将大规模采样+行为聚类应用于竞赛编程；本文 GenCorrect 在聚类思想基础上引入多轮反馈精炼，显著减少所需生成量。
2. **OpenCodeReasoning / OpenCodeReasoning-II**（Ahmad et al., 2025a,b）：使用合成推理轨迹蒸馏提升编程能力；本文继承其合成数据范式，并扩展至 SFT+RL+GenCorrect 全链路。
3. **GenCluster**（Samadi et al., 2026）：首个用开源模型在 IOI 上获金牌的工作；本文 GenCorrect 借鉴其测试时计算思路，但增加子任务级反馈的累积精炼机制。
4. **DeepSeek-R1 / DAPO**（Guo et al., 2025; Yu et al., 2025）：可扩展的 RL 训练范式；本文沿用 GRPO+无 KL penalty 的配置，适配代码可执行奖励。
5. **o1-ioi / Gemini ICPC**（OpenAI et al., 2025; Lin & Cheng, 2025）：闭源系统在 IOI/ICPC 上获金牌；本文的核心差异在于提供完整开源训练管线与逐组件消融分析。
6. **Nemotron-Cascade 2**（Yang et al., 2026）：3B 活跃参数的 Cascade RL 方法；本文 Nano-CC 在相近参数规模上通过 SFT+RL+GenCorrect 达到更高 IOI 分数。

## 局限性与未来方向
1. **计算资源依赖**：系统需要大量训练和测试时算力（IOI 2026 实时推理峰值占用 760 块 GB300 GPU），限制了在小规模团队中的可复现性。
2. **RL 未扩展到 Ultra 规模**：因算力约束，Ultra-CC 仅使用 SFT，未做 RL 训练，无法量化 RL 在更大模型上的边际增益。
3. **泛化范围未知**：当前方法和发现主要围绕竞赛编程场景，是否适用于其他算法密集型领域（如数学证明、形式验证）有待验证。
4. **数据分发限制**：因第三方 redistribution 限制，无法公开完整训练语料，仅能提供详细的数据和训练流程说明。
5. **未来方向**：计划通过 NeMo-Skills 开源 checkpoint 和可运行推理/评估脚本；探索更高效的 RL 算法以缓解长轨迹稀疏奖励问题；将竞赛经验迁移至更广泛的代码智能场景。

## 研究启发与可借鉴点
1. **合成数据中的"自改进"轨迹设计**：在 SFT 数据中加入 teacher 对已有解的 refine 样本，可有效预训练模型的迭代修正能力，减少推理时 GenCorrect 的学习成本；可迁移至任何需要自我修正的代码/推理任务。
2. **子任务级累积反馈的多轮精炼范式**：GenCorrect 将部分分拆解为子任务维度并累积历史最优，指导后续生成定向攻克剩余 gap；该思路可推广至任何具有层次化评估指标的任务（如数学证明的 intermediate lemmas）。
3. **"强基座 + 轻量 SFT"优于"弱基座 + 重度后训练"的经验**：Ultra-CC 仅 1 epoch SFT 即全面超越 Nano-CC 的 3 epoch SFT + RL，提示在选择基座模型时应优先投资预训练质量而非后训练深度。
4. **NVFP4 量化与 MTP 加速的工程实践**：实时竞赛中用 NVFP4（FP8 KV cache, MTP=5）实现 3.7× 吞吐提升且仅损失 6.6pp Score@1，为长上下文代码生成场景的部署优化提供了可复用的量化配置参考。
5. **严格的评测污染防控**：在 SFT/RL 数据构建阶段主动排除所有评估集题目并做去重，IOI 2026 甚至采用赛前实时运行的前瞻性评测设计；这一做法可作为竞赛类 LLM 评测的黄金标准。

## 关键术语表
**GenCorrect**：一种反馈驱动的测试时计算策略，通过多轮"生成→多样性选择→执行评估→子任务反馈累积→精炼生成"的循环迭代，在受限提交次数内逐步逼近最优解。

**GRPO（Group Relative Policy Optimization）**：一种强化学习算法，通过对同一 prompt 的多条 rollout 计算组内相对优势（leave-one-out baseline），无需 critic 网络即可优化策略。

**Score@k**：在 IOI 评测中，生成 k 个独立解后取每子任务的最高分求和的平均值，衡量模型的并行采样能力。

**NVFP4**：NVIDIA 提出的混合精度量化格式，将权重量化为 4-bit，支持 FP8/BF16 KV cache，在保证精度的同时显著提升推理吞吐量。

**RLVR-teacher checkpoint**：强化学习验证与正则化（Reinforcement Learning Verification and Regularization）训练后的教师模型 checkpoint，包含更强的推理能力。

**SFT（Supervised Fine-Tuning）**：在有标注的推理轨迹数据上对预训练语言模型进行微调，使其适应特定任务风格的训练方式。

**LiveCodeBench Pro**：由竞赛奖牌得主评审的、防止数据污染的持续性代码能力评测基准，按时间分离训练/评测数据。

**MTP（Multi-Token Prediction）**：多步预测加速技术，在一次前向传播中同时预测多个 token，可显著提升推理吞吐量。

## 可复现要素
- **数据集**：22,000 道竞赛题用于 SFT，3,219 道用于 RL；论文未公开完整训练语料（第三方 redistribution 限制），但提供了详细的筛选流程和统计信息。
- **代码/权重**：计划通过 NeMo-Skills 开源 Competition Ultra-CC checkpoint 及推理/评估 recipe；NeMo RL 库已开源（github.com/NVIDIA-NeMo/RL）。
- **关键超参**：SFT 学习率 Nano=5×10⁻⁵（constant），Ultra=1.5×10⁻⁵（cosine）；RL 学习率 3×10⁻⁶（constant）；GRPO rollouts per step=1,024；序列长度 262K tokens；GenCorrect 每轮 200 生成/10 提交（最终轮扩至 1,000 生成）。
