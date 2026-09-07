---
title: "Legibility-is-Not-Interpretability-Comparing-Judged-and-Actu"
source: https://arxiv.org/pdf/2609.04194v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:07:30"
field: "大模型可解释性与推理分析"
keywords: ["Chain-of-Thought", "Interpretability", "Process Reward Modeling", "Advantage", "Faithfulness", "LLM Judges", "Reasoning Traces"]
innovations: ["将RL优势函数操作化为CoT步骤重要性度量并配以PELT变点检测pipeline", "构建split-half噪声天花板参照下的judge/critic可解码性评测框架", "揭示正确/错误回答间步骤重要性文本解码的强不对称性"]
benchmarks: ["AIME 24/25/26", "AMC 23", "MATH500", "GSM8K", "Scruples"]
---

# 论文速读：Legibility-is-Not-Interpretability-Comparing-Judged-and-Actu

## 一句话总结
本文提出以强化学习中的"优势（advantage）"来操作化定义CoT推理步骤的重要性，并通过蒙特卡洛采样+变点检测识别关键步骤；实验发现LLM裁判和微调批评家虽能在错误回答中较好解码步骤重要性，但在正确回答中recover的信号极少，说明推理文本的"可读性"不等于"可解释性"。

## 研究问题与动机
- CoT推理轨迹从文本上看似乎提供了模型如何得出答案的清晰窗口，越来越多工作用LLM裁判诊断错误、评估忠实度、通过过程奖励模型进行步级监督，但这些做法都隐含假设：推理步骤的文本编码了其功能重要性信息。
- 现有忠实度测试（如Turpin等、Chen等）多在响应层面做二元判定，未考察推理链中各步骤对最终答案的粒度级功能贡献。
- 过程奖励建模（如Math-Shepherd）依赖中间步骤的可评估性，若步骤重要性无法从文本解码，则此类方法的基础假设存疑。
- 需要一种与RL credit assignment相衔接的、可操作的步骤重要性度量，并实证检验其从文本中的可解码程度。

## 核心贡献（创新点）
- **将CoT步骤重要性操作化为"优势"**：定义A^π(s,a)=Q^π(s,a)−V^π(s)，以蒙特卡洛rollout估计，与反事实必要性/KL距离等方法在目标函数上本质不同——直接刻画步骤对目标答案方向的移动量。
- **提出基于PELT变点检测的关键步骤识别 pipeline**：将价值轨迹建模为分段常数，用Pruned Exact Linear Time算法+Beta后验效应量过滤（δ=0.1, P≥0.95）定位 consequential 步骤，比逐步z检验假阳性低三个数量级且 replication 率达97-99%。
- **构建judge/critic可解码性评测框架**：用PR-AUC和precision@k%在self-advantage和correctness-based两种reward下，系统比较zero-shot LLM裁判与fine-tuned critic的解码能力，并引入split-half噪声天花板作参照。
- **揭示正确/错误回答间的解码不对称性**：微调critic在错误回答上PR-AUC达0.28-0.30（接近保守天花板≈0.6的一半），但在正确回答上仅0.065-0.10，表明文本对关键步骤信息的编码是不完全的。
- **将优势概念用于补充cue-based忠实度测试**：在Scruples数据集上发现加cue后consequential步骤比例从58%降至15%，揭示cue常使推理过程"被确定性化"。

## 方法详解
- **优势定义**：给定语言模型策略π、状态s_t（prompt+前t-1步）、步骤a_t，价值函数V^π(s)=E_π[r|s]为从s出发获得reward=1的概率；Q值Q^π(s,a)=E_π[r|s,a]为执行a后的期望reward；优势A^π(s,a)=Q^π(s,a)−V^π(s)。本文主要使用self-advantage（r=1[最终答案与原轨迹一致]），次要使用correctness-based（r=1[答案正确]）。
- **蒙特卡洛估计**：在每个前缀s_t处采样N=50次续写，统计reward=1的比例作为V̂(s_t)和Q̂(s_t,a_t)，代入公式得Â(s_t,a_t)。
- **变点检测标记consequential步骤**：将V̂序列视作分段常数时间序列，用PELT（Killick et al., 2012）+精确二项代价函数检测变点；对每个候选变点放置Beta后验p_before~Beta(1+K_b,1+N_b−K_b)、p_after~Beta(1+K_a,1+N_a−K_a)，若P(|Δp|>δ)≥0.95且95%可信区间不含0（δ=0.1），则标记该步为consequential。最小段长≥2，定位要求置信集为单步。
- **Judge评测**：用Qwen3-1.7B/8B/32B和Qwen3.6-27B作zero-shot裁判，system prompt要求输出[−1,1]区间的优势评分（direct-importance模式），从\boxed{}解析。
- **Critic微调**：在四个backbone上用LoRA（r=16,α=32,dropout=0.05）加标量回归头，训练数据为80%五个数学数据集的MC估计标签，优化器AdamW(lr=2e−4,wd=0.01)训2 epoch；比较WMSE直接回归、Value BCE回归、Value BCE+rank辅助损失三种目标，默认用Value BCE。
- **评测指标**：PR-AUC（因consequential步骤仅占1.8-2.3%极不平衡）和precision@k%，并以split-half噪声天花板（conservative/optimistic）作性能上限参照。

## 实验与结果
- **数据集**：AIME 24/25/26、AMC 23、MATH500、GSM8K六个数学推理benchmark，每benchmark 30题×10回复=1800条/模型；Scruples 100条伦理判断样本用于忠实度补充实验。
- **生成模型**：Qwen3-1.7B/4B/8B（thinking off）和Qwen3-1.7B（thinking on）；Judge/Critic模型：Qwen3-1.7B/8B/32B、Qwen3.6-27B。
- **consequential步骤稀有性**：non-thinking Qwen3-1.7B上仅1.8%步骤为consequential（ID），thinking模式更低；难题（AIME）比简单题（GSM8K/MATH500）有更高比例的consequential响应。
- **推理轨迹模式**：performance提升主要来自"high throughout"比例增加（thinking: 24%→61%；scale 1.7B→8B: 25%→39%），说明规模/thinking优势源于更好的先验而非推理过程中发现答案。
- **Judge性能**：即使最强裁判Qwen3.6-27B在ID上PR-AUC仅0.065（self-advantage），距噪声天花板≈0.6差9×；value-derived模式在所有尺度均贴近chance（PR-AUC 0.018-0.027）。
- **Critic性能（核心结果）**：微调critic在错误回答ID上PR-AUC达0.28-0.30（≈保守天花板0.6的半数），precision@0.5%达0.55-0.60贴近天花板0.61-0.62；但在正确回答上PR-AUC仅0.065-0.10（3.5-5×chance），precision@0.5%仅0.08-0.16，远低于天花板0.72/0.56。
- **跨模型泛化**：Qwen3-8B生成器上replicate同样不对称（PR-AUC 0.18-0.28 ID）；critic scaling在self-advantage上基本平坦（1.7B≈Qwen3.6-27B）。
- **Cue-based忠实度实验**：BASE设置58%响应含consequential步骤，加expert cue后降至15%，说明cue使推理"去信息化"。

## 相关工作脉络
- **Turpin et al. (2023)、Chen et al. (2025)**：用hint cue扰动测试CoT忠实度，结论是模型常按prompt hint作答但不在CoT中提及；本文用per-step advantage在此setup上揭示更细粒度——即使提到cue的步骤也可能无consequential self-advantage。
- **Bogdan et al. (2025) Thought Anchors**：提出KL散度和反事实cosine-similarity过滤衡量步骤重要性；本文与之本质区别是advantage直接靶向目标答案方向，而KL无法区分朝向/远离目标且易被无关答案概率质量膨胀。
- **Wang et al. (2024) Math-Shepherd**：过程奖励建模，数学上potential等价于advantage，但N=8过小且completer与generator不同模型；本文N=50且自rollout，追求统计效力而非在线RL信号。
- **Gandhi et al. (2025)**：将CoT步骤分类为"cognitive行为"（uncertainty management、active computation等）；本文在其分类基础上量化各类别的consequential比例，发现thinking模式下uncertainty管理在正确回答中最常consequential。
- **Lanham et al. (2023)、Paul et al. (2024)**：扰动测试显示大模型CoT可能不忠实；本文指出这些方法多为响应级二元判定，缺少步骤级粒度。
- **Boppana et al. (2026) Reasoning Theater**：CoT可能是"performative"——模型早在锁定答案后才继续生成；本文的self-advantage可在同一response内量化每一步的实际价值变化。

## 局限性与未来方向
- **文本可解码信号有限**：正确回答中consequential步骤的信息几乎无法从文本 recover，意味着CoT文本对"真正推动成功的步骤"编码较弱。
- **MC估计方差**：N=50 rollout在高variance场景（模型能力边缘）有效，但对近确定性题目（如GSM8K正确回答）信号微弱，天花板本身较低。
- **piecewise-constant假设的违反**：约2.2-13%轨迹呈ramp状渐变而非阶跃，导致pipeline将 gradual climb 误标为单步consequential（主要为thinking模式）。
- **仅评估math推理**：general reasoning、code、多模态等域尚未验证，概念外推需谨慎。
- **仅用self-advantage为主**：correctness-based advantage更难解码（PR-AUC仅0.028-0.092），但论文自述正确回答中的 decode 更具解释学价值却更难实现。
- **未探索非textual信号**：hidden state、attention pattern、activation trace等或许承载更多信息，本文只探测文本本身。

## 研究启发与可借鉴点
- **将RL advantage引入CoT可解释性分析**：credit assignment视角为步骤重要性提供了形式化、可计算的度量，可迁移到任何需要解释中间推理过程的场景（如多步规划、代码生成）。
- **PELT变点检测+Beta后验过滤的pipeline设计**：相比逐步假设检验假阳性低三个数量级且split-half replication达97-99%，该方法可直接复用至其他时间序列式模型行为分析。
- **split-half噪声天花板作为硬上限参照**：鉴于gold label本身是MC估计存在方差，引入噪声天花板而非追求1.0 F1，使评测更诚实；这一思路可推广到其他带噪声label的解码任务。
- **correct vs. incorrect回答的不对称分析框架**：按最终答案正确性分层评测judge/critic性能，揭示不同情境下文本信号的可解码差异，值得作为标准分析维度纳入后续工作。
- **将self-advantage用于补充cue-based忠实度测试**：证明加expert cue会使58%→15%的响应丧失consequential步骤，为"模型是否真的在reasoning"提供了比二元faithfulness更细的实证判据。

## 关键术语表
- **Advantage（优势）**：强化学习中Q值与V值之差，本文操作化定义为"包含某推理步骤后期望reward的变化量"，即该步骤对最终答案的贡献方向与大小。
- **Self-advantage**：以"续写匹配原轨迹最终答案"为reward估计的优势，衡量步骤对模型实际行为的贡献；区别于correctness-based优势（以答案正确为reward）。
- **Consequential step（关键步骤）**：|A^π(s,a)|>δ=0.1且P≥0.95的推理步骤，即对该步骤前后价值产生实质性 jump 的步。
- **Process Reward Modeling（过程奖励建模）**：对CoT中间步骤赋予稠密奖励信号的方法（如Math-Shepherd），本文结果对其"步骤文本可评估"假设提出质疑。
- **Chain-of-Thought（CoT）**：模型在输出最终答案前生成的中间推理token序列，本文质疑其文本是否真正encode推理过程的功能结构。
- **PELT（Pruned Exact Linear Time）**：Killick et al. (2012)提出的分段常数时间序列变点检测算法，本文用于定位价值轨迹的阶跃位置。
- **Noise Ceiling（噪声天花板）**：用split-half MC估计计算的理论可达上限，反映gold label自身方差限制下的最优性能。
- **Faithfulness（忠实度）**：CoT文本是否真实反映模型内部推理过程的性质；本文结论支持"CoT legible但不interpret"的立场。

## 可复现要素
- **代码**：https://github.com/kdu4108/importance-advantage（已开源）
- **数据集**：https://hf.co/datasets/kducohere/MC-Math-Rollouts（已开源）
- **可视化Demo**：https://hf.co/spaces/kducohere/mc-math-rollouts-viewer（已开源）
- **关键超参**：MC rollout N=50、top-p=0.95、temperature=1.0、变点检测δ=0.1、PELT penalty λ∈{5,6,8}依数据集/模型校准、LoRA r=16/α=32/dropout=0.05、AdamW lr=2e−4 wd=0.01、2 epoch、seed=42
- **模型**：Qwen3-1.7B/4B/8B、Qwen3.6-27B（公开权重可用）
- **基准数据集**：AIME 24/25/26、AMC 23、MATH500、GSM8K、Scruples（公开）
