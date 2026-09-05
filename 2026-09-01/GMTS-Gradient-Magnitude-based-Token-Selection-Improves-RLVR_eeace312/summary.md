---
title: "GMTS-Gradient-Magnitude-based-Token-Selection-Improves-RLVR"
source: https://arxiv.org/pdf/2608.30632v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 13:44:53"
field: "大语言模型推理增强"
keywords: ["RLVR", "Token Selection", "Gradient Magnitude", "Entropy", "GRPO", "DAPO", "Reasoning"]
innovations: ["建立Token熵与梯度幅值的理论联系并证明线性关系", "提出GMTS方法通过熵-梯度代理量化跨答案Token重要性", "在三个推理域和多种模型规模上验证GMTS优于ETS的一致性优势"]
benchmarks: ["MATH-500", "AIME2024", "AIME2025", "AMC23", "Minerva", "OlympiadBench", "HumanEval", "MBPP", "LiveCodeBench", "Bigcode-Bench", "CS-QA", "CS-QA2"]
---

# 论文速读：GMTS-Gradient-Magnitude-based-Token-Selection-Improves-RLVR

## 一句话总结
本文提出基于梯度幅值的Token选择方法（GMTS），通过建立Token熵与梯度幅值的理论联系，解决了仅凭熵值无法跨不同答案一致反映Token重要性的问题，在数学推理、代码生成和常识推理三个领域均显著优于原有的高熵Token选择方法（ETS）。

## 研究问题与动机
- **现有RLVR方法忽略Token贡献不均衡**：GRPO、DAPO等主流方法对同一回答中的所有Token使用相同的response-level advantage，未考虑不同Token对策略更新的贡献差异。
- **高熵Token的重要性未被充分理解**：已有研究表明仅训练top 20%高熵Token可带来显著提升，但其内在机制（为何高熵Token有效）缺乏理论解释。
- **熵无法跨答案一致反映Token重要性**：不同答案的reward signal存在差异，且PPO-style clipping和KL正则化会进一步影响有效梯度，导致相同熵值的Token在不同答案中实际贡献可能差异显著。

## 核心贡献（创新点）
1. **建立Token熵与梯度幅值的理论联系**：通过命题3.1证明log-probability梯度幅值与Token熵在log尺度下呈线性关系（极限比值为1），解释了高熵Token有效的内在原因。
2. **提出GMTS方法量化Token重要性**：定义GMTS分数δ_i,t = |E_i,t · ω_i,t(θ)|，其中ω_i,t(θ)包含advantage、clipping indicator和KL正则化项，有效捕捉跨答案的Token贡献差异。
3. **广泛的实验验证**：在MATH、CODE、CS三个推理域，1.5B至8B多种模型规模上验证GMTS的一致性优势，平均提升1-3个百分点，在AIME2024等困难基准上提升达5.21个百分点。

## 方法详解
**Token梯度分解**：
- _token-level objective_的梯度可分解为：∇_θ ℓ_i,t(θ) = ω_i,t(θ) · ∇_θ log π_θ(o_i,t)
- 系数ω_i,t(θ)收集了RLVR中的有效学习信号：ω_i,t(θ) = r_i,t(θ)A_i,t · I_ε1,ε2(r_i,t(θ), A_i,t) + β·π_ref/π_θ - β
- 其中I_ε1,ε2为clipping indicator函数，r_i,t为probability ratio，A_i,t为advantage

**熵-梯度理论联系（Proposition 3.1）**：
- 设G_t为log π_θ(o_t|q, o_<t)关于logits z_t的ℓ2梯度范数，E_t为分布熵，ε = 1 - π_θ(o_t|q, o_<t)
- 当ε→0时，lim log(G_t)/log(E_t) = 1
- 这表明高熵Token通常诱导更大的parameter gradient

**GMTS分数与训练目标**：
- GMTS分数：δ_i,t = |E_i,t · ω_i,t(θ)|，利用熵近似梯度幅值排名
- 训练目标：max_θ E_q[1/S_ρ · Σ_i Σ_t I[δ_i,t ≥ τ_ρ] · ℓ_i,t(θ)]
- 仅保留batch中top ρ比例的Token计算梯度，S_ρ为选中Token数量
- 额外计算开销极小，因所需组件（advantage、probability ratio、entropy）在标准RLVR训练中已可用

**与ETS的本质区别**：
1. 相同熵值但不同advantage的Token，GMTS能区分其实际贡献
2. PPO clipping会抑制超出范围的policy gradient，GMTS通过clipping indicator自动降权
3. KL正则化项影响有效梯度，GMTS能捕捉这一影响

## 实验与结果
**实验设置**：
- 框架：verl，基线方法：GRPO、DAPO
- 模型：Qwen2.5-math/coder/base 1.5B/7B，Qwen3-8B
- 训练集：MATH-12K/DAPO-MATH-17K、KodCode、CS-QA
- 评估：average@16（MATH/CS域），greedy（CODE域）
- 默认选择比例：top 20%

**主要结果**：
| 模型 | 方法 | MATH Avg. | CODE Avg. | CS Avg. |
|------|------|-----------|-----------|---------|
| 1.5B | DAPO+ETS → DAPO+GMTS | 40.01→41.56 (+1.55) | 44.71→46.58 (+1.87) | 65.46→66.33 (+0.87) |
| 7B | DAPO+ETS → DAPO+GMTS | 48.81→50.14 (+1.33) | 54.76→55.50 (+0.74) | 74.77→76.30 (+1.53) |
| 8B | DAPO+ETS → DAPO+GMTS | 54.23→56.08 (+1.85) | - | - |

- **最强提升**：Qwen2.5-math-7B+GRPO在AIME2024上从19.17%提升至23.33%（+4.16pp），平均提升3.41pp
- **跨模型/域一致性**：GMTS在9/10种选择比例配置下均优于ETS
- **Bottom selection验证**：低梯度幅值Token性能低于低熵Token，表明高梯度幅值确实对应更高训练价值

## 相关工作脉络
- **Token-level advantage estimation**：Chen et al. (2025)利用entropy redistribution token-level advantages；Tan & Pan (2025)分配entropy-weighted rewards；Cheng et al. (2026)直接将entropy纳入advantage estimation；Ma et al. (2026)用token-level future KL重新评估贡献。GMTS定位：不修改advantage本身，而是筛选高贡献Token。
- **Objective function design**：Wang et al. (2025b)引入entropy-aware clipping；Cui et al. (2025)动态调整entropy regularization；Li et al. (2025)对高熵critical positions resample构建branched trajectories。GMTS定位：无需修改objective function，仅需选择Token子集。
- **Low importance token filtering**：Wang et al. (2025c)发现top 20%高熵Token驱动RLVR训练（ETS baseline）。GMTS定位：在相同filtering方向上，用gradient-magnitude proxy替代纯entropy作为选择标准。

## 局限性与未来方向
- **理论基础仍不充分**：仅提供了熵-梯度关系的初步理论分析，GMTS优于ETS的严格理论证明有待完善。
- **未验证更大模型**：受计算资源限制，未在14B、32B等更大规模模型上评估GMTS。
- **选择比例敏感性**：虽然实验显示GMTS在宽范围比例下均优于ETS，但最优ρ值仍需任务调优。

## 研究启发与可借鉴点
- **熵-梯度理论联系的可迁移性**：Proposition 3.1建立的log-entropy与log-gradient线性关系可作为其他Token选择/ weighting方法的理论基础。
- **GMTS与现有框架的无缝集成**：方法仅需利用训练中已有的advantage、probability ratio等信号，计算开销极小，可直接集成到GRPO/DAPO等框架。
- **跨域验证的设计思路**：在MATH、CODE、CS三个异构领域验证方法通用性，可为团队研究提供多域评估的参考范式。
- **Bottom selection对照实验**：通过对比低梯度/低熵Token的性能，反向验证了高梯度幅值Token的价值，这种对照设计值得借鉴。

## 关键术语表
**RLVR**：Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习，通过规则reward（如答案正确性）驱动LLM推理能力训练。
**GRPO**：Group Relative Policy Optimization，DeepSeek提出的无value model的PPO变体，通过group-level advantage estimation简化训练。
**DAPO**：Dynamic sAmpling Policy Optimization，基于GRPO的改进方法，移除KL penalty、采用dynamic sampling和token-level loss averaging。
**ETS**：Entropy-based Token Selection，仅按Token熵值选择top 20%进行RLVR训练的方法（Wang et al., 2025c）。
**GMTS**：Gradient Magnitude-based Token Selection，本文提出的方法，通过熵-梯度联系近似梯度幅值排名以选择高贡献Token。
**Token-level advantage**：分配到单个Token的advantage信号，用于衡量该Token对策略更新的贡献程度。
**Clipping indicator**：PPO-style clipping机制中的指示函数，当probability ratio超出[1-ε1, 1+ε2]范围时置零，防止过大更新。

## 可复现要素
- **数据集**：MATH-12K、DAPO-MATH-17K、KodCode、CS-QA/CS-QA2（公开数据集）
- **代码**：论文提供standalone implementation（链接见正文）
- **关键超参**：clip参数ε1=0.2、ε2=0.28；学习率1.5B/7B模型为3×10⁻⁵，8B模型为1×10⁻⁶；temperature=1.0；rollout数=16；最大响应长度根据任务设定（MATH/CS: 2048/4096，CODE: 32768）
- **训练环境**：8×NVIDIA H100 80GB GPU，训练seed固定为0
