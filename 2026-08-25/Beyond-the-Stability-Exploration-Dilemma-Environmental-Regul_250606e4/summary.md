---
title: "Beyond-the-Stability-Exploration-Dilemma-Environmental-Regul"
source: https://arxiv.org/pdf/2608.23311v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:39:56"
field: "大语言模型强化学习"
keywords: ["RLVR", "policy optimization", "stability-exploration dilemma", "Query-KL regularization", "environmental regularization", "LLM training"]
innovations: ["将正则化从action-side Policy-KL移至input-side Query-KL，解耦环境约束与探索预算", "证明QKL梯度仅通过query log-likelihood流动、与response score function正交的结构解耦性质", "基于pre-RL参考分布构造数据集静态per-query权重，降低高温度解码下的梯度方差"]
benchmarks: ["AIME24", "AIME25", "AMC", "MATH500", "Minerva", "OlympiadBench"]
---

# 论文速读：Beyond-the-Stability-Exploration-Dilemma-Environmental-Regularization-for-LLM-Policy-Optimization

## 一句话总结
本文提出ERPO（Environment-Regularized Policy Optimization），通过将正则化从action-side的Policy-KL移至input-side的Query-KL（QKL），在保持探索自由的同时有效约束训练过程中查询分布的漂移，在数学推理任务上实现更高精度与更强长程训练稳定性。

## 研究问题与动机
- **稳定性-探索困境**：现有LLM RLHF/RLVR方法依赖action-side的Policy-KL正则化来控制策略漂移，但该正则化直接约束响应行为并消耗探索预算；若移除则优化失去显式漂移控制，导致训练不稳定。
- **查询分布漂移未被显式建模**：即使训练集固定，随θ更新，模型对自身生成query的自回归似然分布ρ_θ会偏离pre-RL参考分布ρ_θ₀，形成输入侧的环境非平稳性，放大梯度方差。
- **已有方法局限**：主流RLHF PO关注action-side Policy-KL；少数处理prompt分布的工作（如EVA、Align-Pro、StablePrompt）未直接约束query分布漂移，实证显示GRPO训练时Query-KL持续上升而Policy-KL保持低位。
- **高温度解码下的脆弱性**：LLM在高温度采样时尤其敏感于解码分布长尾，查询分布漂移会加剧该阶段性能退化与reward hacking。

## 核心贡献（创新点）
1. **Query-KL（QKL）正则化**：将pre-RL模型诱导的query分布ρ_θ₀作为参考环境，用KL散度约束训练过程中query分布漂移，与现有Policy-KL形成本质区分——前者作用于输入侧、不占用action-side探索预算。
2. **结构解耦定理（Proposition 1）**：证明QKL梯度仅通过query log-likelihood ∇ℓ_θ(q)流动，完全不包含response score function ∇_θ log π_θ(o|q)，从而在正则化输入环境的同时保留响应侧自由探索。
3. **参考分布派生的per-query权重w(q)**：基于缓存的参考模型log-probability构造数据集静态权重，重要性加权使优化目标偏向ρ_θ₀下典型query，降低估计方差并改善高温度行为。
4. **Estimator-agnostic实现**：ERPO以插件形式嵌入GRPO/PPO/REINFORCE等策略梯度流水线，无需额外前向传递、架构改动或clip/baseline逻辑修改。
5. **多温度鲁棒评估范式**：提出固定训练步数、在温度0.1–1.5范围内聚合Avg@k/Pass@k的评估协议，揭示GRPO长程训练中的reward hacking与性能崩塌现象，并验证ERPO在此设定下的显著增益。

## 方法详解
- **查询似然定义**：对tokenized query q=(x₁,…,x_T)，定义序列对数似然ℓ_θ(q)=∑_t log P_θ(x_t|x_{<t})，参考模型ℓ_θ₀(q)在RL前一次性计算并缓存为常量表。
- **目标函数**：J_ERPO(θ)=J_RL(θ)−α·ℛ_query(θ)，其中ℛ_query(θ)=KL(ρ_θ‖ρ_θ₀)为query-level正则项，α控制强度。
- **QKL梯度结构**：∇_θℛ_query(θ)=E_{q~ρ_θ}[(ℓ_θ(q)−ℓ_θ₀(q))·∇_θℓ_θ(q)]，仅依赖query log-likelihood及其梯度，与response policy score function正交。
- **per-query权重推导**：理想重要性权重w*(q)=ρ_θ₀(q)/ρ_train(q)；在均匀采样假设下w*(q)∝ρ_θ₀(q)。实践中用单调代理w(q)∝ℓ_θ₀(q)，经负对数归一化与clip(0,2)截断后用于minibatch更新。
- **PG兼容损失**：L_PG(θ)=−(1/m)∑_{q∈B}[w_B(q)/K·∑_{o∈G(q)}u_θ(q,o)A*_θ(q,o)+α·ℛ̂_query(θ)]，不同estimator通过(u_θ,A*_θ)选择实例化为GRPO/PPO/REINFORCE。
- **零额外开销设计**：ℓ_θ₀表缓存复用；ℓ_θ(q)在每步PG前向中已计算，QKL与QW仅读取该值，无额外FLOPs。

## 实验与结果
- **数据集**：MATH Level 3–5共约8.5K训练样本；测试集含AIME24、AIME25、AMC、MATH500、Minerva、OlympiadBench六个数学推理基准。
- **模型与框架**：Qwen2.5-Math-7B与Qwen2.5-32B，基于EasyR1实现，最大序列长度3K，rollout batch=512、update batch=128、240步训练，默认KL系数0.01。
- **评估协议**：固定训练步数，在温度0.1–1.5聚合Avg@k/Pass@k/Pass@1；同时报告≤1.0与1.2–1.5子区间均值以分离短/长尾行为。
- **主要结果（Qwen-7B）**：ERPO vs GRPO在Avg@32上平均提升6.2%（0.336 vs 0.274），Pass@1提升5.69%（0.332 vs 0.275），Pass@32提升3.64%，各基准全面超越；最高单点增益达14.9%。
- **长程稳定性**：扩展至1K步时，GRPO在温度>1.0于约400步后出现性能骤降（典型reward hacking），ERPO虽有小幅衰退但整体退化幅度显著更低，且高温度区段性能反而提升。
- **Reward hacking量化**：GRPO训练-评估一致性缺口平均6.47%（Step-240 Eval骤降至58.4%而Train仍≈77%）；ERPO将其降至3.14%，降幅约51%。
- **跨算法推广**：ERPO+DAPO与ERPO+RLOO分别在≤1.0温度下获得+10.24%与+2.28%绝对增益。

## 相关工作脉络
- **RLVR与verifiable reward**：AlphaCode、DeepSeekMath等工作利用执行/答案可验证性提供自动reward，但高方差与reward hacking稳定性问题未解；PRM/工具增强推理亦面临类似挑战。
- **Policy-KL主导的正则化体系**：TRPO/PPO/RLHF/RLAIF/DPO等传统方法均以action-side KL或trust-region约束策略，本文指出其未覆盖输入侧环境漂移这一隐性不稳定源。
- **Prompt/数据分布建模**：EVA、Align-Pro尝试形式化prompt分布；StablePrompt、WPO进行数据重加权，但未直接约束policy-induced query分布相对pre-RL参考的KL漂移。
- **非平稳RL与robust RL**：初始状态分布或转移核漂移的经典设定（Iyengar 2005; Padakandla 2021）鼓励显式分布控制，本文为LLM场景引入同等思路。
- **Imitation learning covariate shift**：DAgger/AggreVaTe通过交互数据聚合缓解state分布漂移；本文类比提出query分布显式锚定。
- **高温度敏感性与文本退化**：Holtzman等（2020）揭示长文本生成中熵增退化现象；本文证实query分布漂移会加剧高温度解码的长尾崩溃。

## 局限性与未来方向
- **任务/模型泛化待验证**：实验集中于数学推理与Qwen家族，对指令遵循、对话、代码生成、多语言等场景的迁移性未充分检验。
- **查询似然估计成本**：依赖每步前向计算ℓ_θ(q)，对超大规模数据集或极长query的计算与内存开销可能随模型规模放大；论文未给出系统性的复杂度分析。
- **超参α未做充分搜索**：正则化强度α的选取经验性较强，缺乏系统敏感性分析与自动调优策略。
- **参考分布假设**：将ρ_θ₀视为稳定锚点隐含“pre-RL模型已具备合理query先验”的前提，对于从零预训练或domain-shift剧烈的场景可能不适用。
- **长期collapse仍未根除**：图5/6显示ERPO在极端长程下仍可能出现熵激增与采样能力丧失，仅延缓而非完全免疫。

## 研究启发与可借鉴点
- **环境-策略解耦思想**：将regularization拆分为input-side（环境）与output-side（动作）两个正交通道，为多模态/agent系统中observation distribution drift问题提供新思路。
- **零额外开销的插件化设计**：利用已有PG前向中的ℓ_θ(q)直接构造QKL估计，实现低成本高收益的性能提升，可推广至任何基于sequence likelihood的RL pipeline。
- **参考分布派生importance weight的构造技巧**：通过单调代理+clip截断平衡bias-variance，避免精确密度比估计的数值不稳定，对off-policy校正有借鉴价值。
- **多温度鲁棒评估协议**：将评估从单点温度扩展至温度区间聚合，能更稳定地反映模型长尾与泛化能力，建议作为后续RLVR工作的标准评测范式。
- **与团队现有方向结合点**：可将QKL正则化应用于代码生成（syntax tree likelihood）、对话系统（turn-distribution drift）等query-sensitive场景；亦可与process reward结合探索step-level的环境正则化。

## 关键术语表
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：利用自动可验证的ground truth（如数学答案、代码执行）替代人工偏好标注的RL训练范式。
- **Policy-KL / Query-KL**：前者约束当前policy与reference policy在response空间的对数概率差；后者约束policy-induced query分布与pre-RL reference query分布的对数概率差。
- **Policy-induced query distribution ρ_θ**：由模型自身自回归似然定义的query概率分布，区别于固定训练集分布ρ_train。
- **Environment drift / 环境漂移**：训练过程中因policy更新导致输入侧query分布偏离参考分布的现象，类比非平稳RL中的转移核漂移。
- **Estimator-agnostic**：方法不绑定特定策略梯度估计器，可无缝替换GRPO/PPO/REINFORCE内部的advantage与weight函数。
- **Reward hacking**：模型过度拟合训练reward信号而牺牲真实推理能力的现象，表现为训练集准确率高但评测集性能骤降。
- **Per-query weight w(q)**：基于reference模型log-probability构造的数据集静态权重，用于对advantage进行重要性重加权以降低方差。
- **Structural decoupling（结构解耦）**：QKL梯度仅含query log-likelihood导数项，与response score function正交，从而正则化不挤压探索预算。

## 可复现要素
- **数据集**：MATH Level 3–5（约8.5K），训练/测试集划分论文未明示；测试基准AIME24/25、AMC、MATH500、Minerva、OlympiadBench均为公开基准。
- **代码**：作者已开源，地址为 https://github.com/alibaba/ERPO。
- **模型权重**：使用Qwen2.5-Math-7B与Qwen2.5-32B（开源模型），论文未提供额外微调权重。
- **关键超参**：最大序列长度3K；rollout batch=512、update batch=128；训练步数240（主实验）/1000（长程实验）；rollout per query=8（主实验）/16（消融）；KL系数0.01；α默认值论文未明确给出（附录提及α=5×10⁻²与5×10⁻³的消融）；temperature采样范围0.1–1.5。
- **框架**：EasyR1（Zheng et al., 2025）。
