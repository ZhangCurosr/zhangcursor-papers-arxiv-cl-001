---
title: "DOES-ON-POLICY-DISTILLATION-REALLY-DISTILL-FROM-NOISY-TEACHE"
source: https://arxiv.org/pdf/2608.31046v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:35:10"
field: "大语言模型强化学习后训练"
keywords: ["在线策略蒸馏", "零监督强化学习", "token级信用分配", "大模型后训练", "策略自适应"]
innovations: ["发现OPD教师监督噪声率高且学生对其不敏感，揭示OPD增益源于低概率token抑制而非知识蒸馏", "提出OPSA：完全零外部监督的熵自适应负优势token级RL方法，重塑策略分布实现自我改进"]
benchmarks: ["AIME24", "AIME25", "HMMT25", "MBPP+", "GPQA-Diamond"]
---

# 论文速读：DOES-ON-POLICY-DISTILLATION-REALLY-DISTILL-FROM-NOISY-TEACHE

## 一句话总结
本文系统分析了在线策略蒸馏（OPD）的真正增益来源，发现教师监督噪声率高达30%~50%，但学生对其不敏感；OPD的提升本质上来自对低log-probability token的抑制，而非真正的知识蒸馏。基于此，作者提出了完全无需外部监督的在线策略自适应（OPSA），通过在低概率token上分配熵自适应的负优势来重塑策略分布，在数学推理任务上大幅超越现有方法。

## 研究问题与动机
- **OPD教师监督可靠性存疑**：OPD通过反向KL散度让教师在学生生成的轨迹（对学生是on-policy，对教师本质上是off-policy）上计算token级优势，但教师能否在学生轨迹上提供可靠监督尚未被量化验证。
- **现有方法依赖额外监督信号**：RLVR需要可验证奖励（稀疏且易导致同组内优势消失）；OPD需要共享词表和教师logits的白盒访问；OPSD需要通过hint（如参考答案）构建教师，仍存在信息泄漏风险——三者均依赖外部监督。
- **OPD增益的真实来源未被揭示**：即使去除或保留噪声监督，OPD训练结果收敛到相近性能，暗示其增益并非来自行为模仿，但具体机制尚不清楚。
- **缺乏真正零外部监督的token级密集信号方法**：现有label-free方法（如TTRL、EMPO）多为轨迹级粗粒度信号，难以支持细粒度的推理优化。

## 核心贡献（创新点）
1. **发现OPD中教师监督高度噪声化（噪声率30%~50%），且学生对此不敏感**：通过控制实验证明，仅用噪声轨迹训练与标准OPD、去除噪声后训练收敛到相近性能，揭示了OPD机制可能并非知识蒸馏。
2. **定位OPD增益的真正驱动因素：低log-probability token的抑制**：证明高log-probability token几乎不提供有效梯度；将所有OPD优势替换为单一固定负值即可复现大部分性能提升。
3. **提出OPSA（On-Policy Self-Adaptation），一种完全零外部监督的token级RL方法**：通过熵自适应负优势在最低20% log-probability token上更新策略，重新分配概率质量，同时提升低熵位置精度与高熵分叉位置的探索多样性——区别于OPD/OPSD的外部教师依赖和GRPO的轨迹级稀疏奖励。
4. **系统性消融与机制解释**：证明了fork token上的概率重分配是性能提升的主要来源，并通过Jaccard距离分析确认OPSA不会导致多样性坍缩。

## 方法详解
**OPSA核心设计：**
- **训练token选择**：仅对学生采样轨迹中log-probability最低的20% token进行更新（排除高log-probability的near-zero梯度区域）。
- **熵自适应负优势**：对选定的每个token $i$，动态计算优势值：
$$A_i^{\mathrm{dyn}} = -\frac{1}{2} - \frac{H_i - H_{\min}}{2(H_{\max} - H_{\min})}$$
其中 $H_i$ 为该token的熵，$H_{\min}$ 和 $H_{\max}$ 为同一条轨迹中最低20% log-probability位置的熵最小/最大值。$\delta=1$ 时，高熵token获得更大负优势幅度。
- **训练损失**：
$$\mathcal{L}_{\mathrm{OPSA}} = -\mathbb{E}\left[\frac{1}{|S_{\mathrm{lowest\ 20}}|}\sum_{i \in S_{\mathrm{lowest\ 20}}} A_i^{\mathrm{dyn}} \log \pi_\theta(y_i | x; y_{<i})\right]$$
- **机制效果**：
  - 在**高熵分叉位置**：负优势将概率质量从尾部token重新分配给头部token，同时均匀分布在多个头部token之间，保持多样性；
  - 在**低熵位置**：高置信token（前20%之外）被排除在训练之外，保留已有高精度预测；
  - 整体效果：增强反射性长链推理（如"wait"、"however"等token增多），引导模型进入更长、更有反思性的推理分支。

## 实验与结果
**数据集**：训练集DAPO-17k（无标签、无ground-truth）；测试集包括AIME24、AIME25、HMMT25（数学推理）、MBPP+（代码生成）、GPQA-Diamond（通用问答）。

**模型**：Qwen3-1.7B、Qwen3-4B、Qwen3.5-9B。

**基线**：GRPO、TTRL、OPD（Qwen3-4B-Instruct为教师）、OPSD、NSR。

**主要结果（Qwen3-1.7B）**：
| 方法 | AIME24 Avg@32 | AIME24 Pass@32 | AIME25 Avg@32 | HMMT25 Avg@32 |
|------|-------------|---------------|--------------|--------------|
| Base | 13.44 | 40.00 | 9.69 | 5.73 |
| +GRPO | 33.96 | 70.00 | 25.31 | 15.10 |
| +OPD | 32.08 | 73.33 | 20.52 | 13.85 |
| **+OPSA** | **48.85 (+35.41)** | **80.00 (+40.00)** | **35.31** | **23.33** |

- OPSA相对Base提升：AIME24 **+263%**、AIME25 **+264%**、HMMT25 **+307%**，Pass@32均翻倍以上。
- 超越最强基线（GRPO）平均**+11.04点Avg@32**和**+8.89点Pass@32**。
- **Qwen3.5-9B**上也有持续改善：AIME24从76.35→87.81（+15.0%）。
- 在相同token预算下，OPSA仍大幅超越GRPO和OPD，证明增益非单纯来自响应长度增加。
- OPSA可作为GRPO的冷启动，进一步带来约9点提升。

## 相关工作脉络
- **OPD（Lu & Lab, 2025）**：本文直接分析的对象，通过K1估计器实现token级反向KL蒸馏；本文发现其增益并非来自教师监督，而是策略自重塑。
- **OPSD（Zhao et al., 2026; Hubotter et al., 2026）**：用hint替代外部教师进行在线自蒸馏；本文指出OPSD仍依赖hint构建教师分布，且存在信息泄漏风险。
- **NSR（Zhu et al., 2026）**：仅从错误轨迹中提取负优势进行RL；本文OPSA进一步消除了对可验证奖励的依赖，实现完全零监督。
- **TTRL（Zuo et al., 2026）**：基于self-consistency的测试时训练；本文指出其在本地最优处锐化分布，显著降低Pass@k。
- **Intuitor（Zhao et al., 2025b）**：用轨迹级熵作为奖励；本文对比指出轨迹级信号过于粗糙，易导致错误模式过拟合。
- **GRPO（Shao et al., 2024）**：主流RLVR方法；本文OPSA在更少rollout采样下达到更好性能，且无需可验证奖励。

## 局限性与未来方向
- 实验仅覆盖≤9B参数的小模型，OPSA在更大模型或MoE架构上的扩展性尚待验证。
- OPSA通过重新分配现有概率质量发挥作用，对已深度后训练、输出分布已过度尖锐的低熵模型，提升可能有限。
- 作为plug-and-play方法，OPSA可能无法显著扩展策略的底层探索边界（体现在thinking-mode Pass@k提升有限）。
- 未来可与GRPO等RL方法结合，用OPSA做冷启动后再进行后续RL训练，或进一步与扩展探索的机制结合。

## 研究启发与可借鉴点
- **"解耦分析法"的示范价值**：通过逐步剔除OPD的各个组件（教师信号、正优势、高logp token），精确定位了真实增益来源——这一分析范式可迁移到其他RL算法的机制探究。
- **低log-probability token是关键学习信号**：高置信token几乎不提供有效梯度，这一发现对设计更高效的token选择策略有普遍参考价值。
- **熵自适应负优势的通用性**：用token熵调制学习信号强度的思路，可迁移到其他需要精细credit assignment的序列生成任务。
- **零外部监督下的密集信号构建**：OPSA完全不需要教师、reward或hint，为资源受限场景下的RL训练提供了新思路；可作为其他RL方法的免费冷启动模块。
- **Jaccard距离评估多样性**：通过4-gram Jaccard距离定量验证策略多样性不坍缩，该方法论可直接用于评估其他策略优化算法的探索能力。

## 关键术语表
- **On-Policy Distillation (OPD)**：学生策略在其自身采样轨迹上通过反向KL散度接受教师token级监督的在线蒸馏方法。
- **On-Policy Self-Adaptation (OPSA)**：本文提出的零外部监督方法，通过熵自适应负优势对低log-probability token进行策略更新，实现自我改进。
- **Reinforcement Learning with Verifiable Rewards (RLVR)**：利用可验证奖励（如数学答案正确性）进行轨迹级强化学习训练的方法，如GRPO、DAPO。
- **Token Entropy**：策略在某个token位置的概率分布熵，衡量该位置的不确定性程度，高熵对应多种合理候选token分散分布。
- **Negative Advantage**：负的优势值，在OPSA中用于抑制低概率token，将其概率质量重新分配给更高概率的token。
- **Fork Token**：思维链中引发不同推理分支的分叉token位置，通常具有高熵特征并与反射性语言相关。
- **K1 Estimator**：用于估计OPD中反向KL散度的无偏估计器，避免teacher模型需要对完整响应分布进行采样。
- **Pass@k / Avg@k**：在k次采样中至少一次回答正确的比例 / k次采样的平均正确率，是数学推理任务的核心评测指标。

## 可复现要素
- **数据集**：DAPO-17k（训练集）公开；AIME24/25、HMMT25、MBPP+、GPQA-Diamond均为公开benchmark。
- **代码/框架**：基于slime框架（https://github.com/THUDM/slime）实现；论文未提供独立开源代码仓库（仅标注"Hugging Face  Github"链接，但正文未给出具体URL）。
- **关键超参**：学习率1e-6、rollout batch size 64、训练仅使用最低20% log-probability token、temperature=1.0（训练解码）、8×H100/H200 GPU。
