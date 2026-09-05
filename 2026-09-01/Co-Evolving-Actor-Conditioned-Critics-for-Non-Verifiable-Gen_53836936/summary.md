---
title: "Co-Evolving-Actor-Conditioned-Critics-for-Non-Verifiable-Gen"
source: https://arxiv.org/pdf/2608.30397v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:02:58"
field: "语言模型对齐与反馈学习"
keywords: ["critique-guided refinement", "actor-conditioned supervision", "co-evolving critic-actor", "non-verifiable generation", "TAISCORE", "GRPO", "DPO"]
innovations: ["提出TAISCORE，以actor-conditioned方式将critique质量与目标actor执行结果绑定评估", "设计critic-actor共演化训练循环，使critic持续适配actor演化能力", "受控实验证明critic独立质量不等于对目标actor的实用性"]
benchmarks: ["WritingBench", "HelloBench", "DeepResearch-Gym"]
---

# 论文速读：Co-Evolving-Actor-Conditioned-Critics-for-Non-Verifiable-Gen

## 一句话总结
本文提出**TAISCORE**，一个将critique质量与其对目标actor的实际改善效果绑定的奖励信号，并通过actor-tailored critic训练（GRPO）与actor DPO交替更新的**共演化循环**，使critic能够适配actor不断变化的能力，在非可验证生成任务上显著超越静态大scale critic和仅基于独立critique质量/最终结果质量的训练方法。

---

## 研究问题与动机

1. **非可验证生成缺乏确定性验证器**：数学/代码等任务可通过答案或测试用例评估，而创意写作、深度研究等open-ended任务存在多解性，质量是多维的，单纯标量偏好评分无法指明具体缺陷及改进方向。
2. **现有critique评价标准存在归因偏差**：独立评判critique质量忽略了actor执行能力，仅看最终改进又无法证明改进来自critique而非actor自身能力提升，导致训练信号与"实用反馈"脱钩。
3. **Critic规模不等于对目标actor的有用性**：可控实验表明，更大scale critic能产出更高独立评分的critique，但并不必然提升目标actor的采纳度（adherence）与任务增益；相反，actor能力提升能显著增强critique执行力。
4. **静态critic会随actor进化而过时**：actor能力变化后，其失败模式与可执行性边界随之改变，原先有用的反馈可能变得冗余或过于困难，需要critic持续适配actor当前状态。

---

## 核心贡献（创新点）

1. **将critique重新定义为actor-conditioned的revision guidance**，指出其实用性取决于critique与actor的交互结果而非critique本身，并通过受控实验量化了critic scaling vs. actor scaling对有用性的不同影响。
2. **提出TAISCORE**，一个评估指令-初始响应-critique-修订四元组完整链路的奖励，同时考察critique是否指向真实缺陷、actor是否采纳、目标维度是否改善、以及是否忠于原指令。
3. **设计共演化critic-actor训练循环**：用TAISCORE通过GRPO训练actor-tailored critic，再用critique-guided refinement构造DPO偏好对更新actor，交替进行使critic持续匹配actor演化的能力与失败模式。

---

## 方法详解

### 4.1 TAISCORE（Targeted Actionable Improvement Score）
输入完整rollout $\tau = (x, y_0, c, y_1)$，judge首先输出四个诊断分：
- $q_{\text{qual}}$：critique validity——忠实、具体、重要、可执行
- $q_{\text{adh}}$：critique adherence——actor是否真正采纳critique
- $q_{\text{gain}}$：targeted improvement——修订是否在critique指向的维度上改善
- $q_{\text{faith}}$：faithfulness——critique与修订均忠于原始指令

随后基于诊断分输出最终标量 $T(\tau) \in [1, 10]$ 作为critic训练的GRPO reward。诊断分在推理时生成作为分析工具，最终分 $T(\tau)$ 用于训练。

### 4.2 Critic Update（GRPO）
对每个on-policy $y_0 \sim \pi_t$，critic采样 $N=4$ 条critique $\{c_i\}$，actor各自生成修订 $y_{1,i}$，得到rollout组 $\{\tau_i\}$，每条计算 $r_i = T(\tau_i)$，以组内归一化后的TAISCORE为advantage：
$$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$
$$\mathcal{I}_t(\kappa) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N A_i \log \kappa(c_i \mid x, y_0)\right]$$
同组内初始响应和actor固定，确保reward相对credit只来自critique差异。

### 4.3 Actor Update（DPO）
用适应后的critic $\kappa_{t+1}$ 为actor $\pi_t$ 生成critique-guided revisions，由critique-blind pairwise judge比较 $(y_0, y_1)$，选取 $y_1 \succ y_0$ 构造偏好对：
$$\mathcal{D}_t = \{(x_i, y_{1,i}, y_{0,i})\}_{i=1}^M$$
使用标准DPO损失更新actor，$\pi_{\text{ref}}$ 为更新前的冻结副本，$\beta=0.1$。

### 4.4 Co-Evolving Loop
交替执行两步共3轮：
1. **Critic adaptation**：固定 $\pi_t$，对2K queries做critic GRPO更新得 $\kappa_{t+1}$
2. **Actor update**：用 $\kappa_{t+1}$ 生成critiques，构造2K DPO对，更新得 $\pi_{t+1}$

---

## 实验与结果

- **数据集**：DeepWriting-20K（6K training prompts）、OpenScholar（6K deep research queries）
- **基准**：WritingBench、HelloBench（OEQA + HTG子集）、DeepResearch-Gym（KPR、KPC、Report Quality）
- **基线模型**：Qwen3-8B actor/critic；gpt-oss-120B zero-shot frozen critic；两种消融reward（outcome-gain、critique-quality）
- **主要结果（Qwen3-8B最终actor）**：

| 方法 | WritingBench | HelloBench HTG | DeepResearch-Gym KPR |
|---|---|---|---|
| Base Qwen3-8B | 72.33 | 39.14 | 71.93 |
| gpt-oss-120B critic | 75.41 | 50.93 | 73.46 |
| Outcome-gain reward | 75.63 | 45.14 | 74.37 |
| Critique-quality reward | 75.18 | 49.75 | 74.19 |
| **TAISCORE** | **75.96** | **53.78** | **75.21** |
| **TAISCORE + co-evolution** | **76.72** | **54.40** | **76.14** |

- TAISCORE超越gpt-oss-120B zero-shot critic（写作+3.31，HTG+3.47）；co-evolution进一步将WritingBench从75.96提升至76.72。
- Actor-Critic匹配实验：匹配actor的critic（Qwen3-8B）比适配更小的Llama-3.2-3B/Qwen3-4B分别高出1.34/0.88分。
- 直接修订（不经DPO）实验：TAISCORE critic使Qwen3-8B从72.33提升至75.11（+2.78），远超无critique（+0.97）和通用critique（+1.01）。

---

## 相关工作脉络

1. **Self-Refine / Reflexion**（Madaan et al., 2023; Shinn et al., 2023）：单模型test-time critique-refine循环；本文将其扩展为训练信号，并强调critique需actor-tailored。
2. **RLHF / LLM-as-a-Judge**（Ouyang et al., 2022; Zheng et al., 2023）：标量偏好建模传统路径；本文主张用自然语言critique作为比标量score richer的监督信号。
3. **Rubrics as Rewards / Open-Rubrics**（Gunjal et al., 2025; Liu et al., 2025）：将评估结构化为实例特定criteria；本文与其一致但更关注criteria如何转化为可执行的critique而非仅用于打分。
4. **DR-Tulu**（Shao et al., 2025）：动态维护rubric buffer以适配policy；本文用co-evolving critic-actor loop实现类似思想，但针对non-verifiable generation场景。
5. **Critique-RL**（Xi et al., 2025）：两阶段RL同时训练critique discriminability与actor refinement；本文通过TAISCORE一次性端到端评价完整修订过程，避免分离评价带来的归因误差。
6. **CGI**（Yang et al., 2025b）：interactive agent环境中联合训练actor和critic；本文聚焦open-ended text generation，critique形式为自然语言反馈而非action-level feedback。

---

## 局限性与未来方向

1. **领域覆盖有限**：仅在creative writing和deep research两个领域验证，未扩展至对话、多模态生成、长指令following等。
2. **模型族单一**：主训练实验集中于Qwen系列，跨模型family推广性待验证。
3. **固定轮次co-evolution**：当前策略固定3轮且各轮数据不重叠；未来可探索自适应调度（何时更新critic/actor）或continuous co-training。
4. **Judge依赖**：TAISCORE使用gpt-oss-120B作为judge，虽经Claude Opus 4.8交叉验证显示排名一致性高，但仍存在judge偏差风险。

---

## 研究启发与可借鉴点

1. **归因完整性设计**：TAISCORE的诊断分四元组（validity+adherence+gain+faithfulness）可迁移到任何需要评估"反馈→执行→改善"因果链的场景，如代码生成中的review-guided refinement。
2. **组内对比避免跨样本混杂**：GRPO中固定 $y_0$ 和actor、只变critique的设计，消除了初始质量和actor能力对reward的干扰，这一控制变量思路可用于其他feedback-driven训练。
3. **Co-evolution的解耦更新节奏**：先训critic再训actor的交替策略可借鉴到any两阶段对齐任务，例如先将LLM-as-a-judge对齐到特定actor的失败模式，再用其标注训练pair。
4. **Matched vs. shuffled critique控制实验**：Table 3b的shuffled critique控制条件证明了critique内容特异性的重要性，这种对照实验设计值得在评估critique system时复用。
5. **可与本团队方向结合**：若团队涉及"可验证+不可验证混合任务"，TAISCORE可作为不可验证分支的supervision bridge，链接verifiable reward（数学/代码）与非verifiable reward（写作）。

---

## 关键术语表

- **TAISCORE**：Targeted Actionable Improvement Score，评估critique是否指向真实缺陷、被actor采纳、并带来目标维度改善的四维judge奖励。
- **Actor-conditioned critique**：critique实用性取决于其能否被目标actor有效执行，而非critique本身的孤立质量。
- **Co-evolving critic-actor loop**：交替更新critic（GRPO）与actor（DPO）的训练循环，使critic始终适配actor当前能力。
- **GRPO（Group Relative Policy Optimization）**：相对策略梯度优化，用组内normalized reward作为advantage更新critic。
- **DPO（Direct Preference Optimization）**：直接使用偏好对 $(y^+, y^-)$ 优化actor，无需显式reward model。
- **Critique adherence**：衡量actor修订是否真正采纳critique内容，而非做无关修改。
- **Outcome-gain reward**：仅比较修订前后最终response质量差值的二值/三值奖励，忽略critique是否导致改善。
- **Non-verifiable generation**：无确定性验证器的开放生成任务，如创意写作、深度研究，多解且质量多维。

---

## 可复现要素

- **数据集**：DeepWriting-20K（写作训练prompt来源）、OpenScholar（深度研究查询）、WritingBench、HelloBench、DeepResearch-Gym——均为公开benchmark
- **代码开源**：论文未明确声明代码开源，仅提及"Appendix"补充细节
- **关键超参**：Critic GRPO学习率 $1\times10^{-6}$、KL系数0.02、N=4 critiques/prompt、batch size 8；Actor DPO学习率 $1\times10^{-6}$、$\beta=0.1$、epoch=1；共演化3轮，每轮2K queries
- **GPU**：4× NVIDIA RTX PRO 6000 Blackwell

---
