---
title: "DOES-ON-POLICY-DISTILLATION-REALLY-DISTILL-FROM-NOISY-TEACHE"
source: https://arxiv.org/pdf/2608.31046v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:35:38"
field: "大模型后训练与自我改进"
keywords: ["on-policy distillation", "self-distillation", "token-level reinforcement learning", "self-improvement", "entropy-adaptive advantage", "label-free training"]
innovations: ["揭示OPD教师监督高度噪声化且学生提升主要来自低logp token抑制而非知识蒸馏", "提出监督-free的OPSA方法通过熵自适应负advantage重塑策略分布", "在Qwen3-1.7B上AIME24 Avg@32提升263%并超越GRPO/OPD/OPSD等多基线"]
benchmarks: ["AIME24", "AIME25", "HMMT25", "MBPP+", "GPQA-Diamond"]
---

# 论文速读：DOES-ON-POLICY-DISTILLATION-REALLY-DISTILL-FROM-NOISY-TEACHE

## 一句话总结
本文发现现有On-Policy Distillation（OPD）中教师监督高度噪声化，学生提升实际来自对低log-probability token的抑制而非知识蒸馏；据此提出完全无需外部监督的On-Policy Self-Adaptation（OPSA），通过在最小20% logp token上施加熵自适应负advantage重塑策略分布，在Qwen3-1.7B上AIME24提升263%（+35.41 Avg@32）。

## 研究问题与动机
- RLVR（如GRPO/DAPO）仅提供回答级稀疏奖励，长推理路径内中间步骤无法获得细粒度指导，且同组内正确答案一致时归一化advantage趋零，训练不稳定。
- OPD用强教师通过反向KL提供token级优势信号，但要求共享词表与白盒logits，教师选择受限；且学生采样轨迹对学生自身on-policy、对教师天生off-policy，教师能否提供可靠监督存疑。
- 现有On-Policy Self-Distillation（OPSD）用hint（如参考答案）替代外部教师，但仍需额外采样/标注，且存在信息泄漏风险。
- 核心未知：OPD在学生采样轨迹上的"蒸馏"增益究竟来自何处？是否真的依赖教师行为匹配？

## 核心贡献（创新点）
- **揭示OPD教师监督高度噪声化**：噪声随教师规模递增（4B为30.6%，235B达50.6%），最大教师几乎对所有答案token给出负advantage，学生对此噪声不敏感。
- **定位OPD增益的真实来源**：有效学习集中在学生采样的最低20% log-probability token上；将所有OPD advantage替换为单一固定负值仍可达到相近性能，说明增益主要来自对低概率token的抑制而非知识蒸馏。
- **提出监督-free的OPSA方法**：仅用学生自身token熵构造熵自适应负advantage，无需教师/奖励/hint，通过在最低20% logp token上动态分配强度完成自我适应。
- **理论解释OPSA机制**：压制尾部低概率token并将质量重分配至头部token，低熵位置锐化置信预测、高熵分叉处保留多样性并促进反思型长推理。
- **系统性实验验证**：跨Qwen3/Qwen3.5多模型多任务验证，在数学推理上显著提升，同时保持长文多样性并超越GRPO/OPD/OPSD/TTRL。

## 方法详解
- **OPD回顾**：在学生对PREFIX的采样轨迹上计算反向KL，token级advantage $A_i = \log \frac{\pi_t(y_i|x;y_{<i})}{\pi_s(y_i|x;y_{<i})}$，损失 $\mathcal{L}_{OPD} = -E[\frac{1}{|y|}\sum_i A_i \log \pi_s(y_i|x;y_{<i})]$。由于教师需评分学生轨迹，大量off-policy导致噪声。
- **噪声定义**：针对$\boxed{}$内可验证答案token，若正确回答收到负advantage或错误回答收到正advantage，即视为噪声。
- **关键发现（§3.1）**：近零advantage集中在学生高logp token上；对这些top-logp token训练（即使替换为随机advantage）几乎无改善，梯度在该区间趋零。
- **关键发现（§3.2）**：仅用最低20% logp token训练即可复现完整OPD；固定负advantage（−0.5）稳定提升，固定正advantage导致策略坍塌（响应长度趋零、梯度范数爆炸）。
- **OPSA构造（§4.2）**：令$A_i^{fix} = -3/4$，$r_i = 2\frac{H_i - H_{min}}{H_{max} - H_{min}} - 1$，取$\delta=1$得$A_i^{dyn} = -\frac{1}{2} - \frac{H_i - H_{min}}{2(H_{max} - H_{min})}$，只在最低20% logp位置计算损失$\mathcal{L}_{OPSA}$。
- **机制（§4.3）**：在高位熵分叉处，负advantage将质量在竞争head token间均匀重分配而不进一步锐化到单一token；在低位熵处压制尾部采样避免进入错误分支；整体促使更长、更多反思词的推理轨迹。
- **效率**：省去教师前向计算，训练时长显著短于OPD与GRPO（§5.1、附录B.3）。

## 实验与结果
- **数据集**：训练DAPO-17k（仅问题，无标签）；评估AIME24/AIME25/HMMT25（数学I.D.）、MBPP+（代码O.O.D.）、GPQA-Diamond（Q&A O.O.D.）。
- **基线**：RLVR（GRPO）、TTRL、OPD（用Qwen3-4B-Instruct）、OPSD（需thinking启用制造分布错配）、NSR。
- **主结果（Qwen3-1.7B，表2/3）**：OPSA在AIME24上Avg@32从13.44→48.85（+35.41，相对↑263.5%），Pass@32从40→80（↑100%）；AIME25 +25.62（↑264.4%），HMMT25 +17.60（↑307.2%）；综合超越最佳基线GRPO/OPD约+11.04 Avg@32。
- **跨尺度（表2）**：Qwen3-4B AIME24从23.33→62.08（↑166%）；Qwen3.5-9B从76.35→87.81（↑15%），Pass@k亦有小幅提升；代码/Q&A O.O.D.上稳定改善。
- **OPSA vs OPD（表3）**：AIME24 Avg@32 48.85 vs 32.08，Pass@32 80.00 vs 73.33；同时训练与推理效率均更优（表6）。
- **消融**：最小10%训练表现差（过锐化），20%/30%/40%均可达↑45；掩码reflective fork位置后OPSA增益基本消失、响应长度坍塌（图8）。
- **多样性（§5.3.2）**：Jaccard distance分析显示长文本多样性与基线相当，未发生collapse；低总体entropy≠低有效探索。
- **冷启动（附录C.4）**：OPSA checkpoint作为GRPO初始化可再提升~9分且训练稳定。

## 相关工作脉络
- **GRPO/RLVR（Shao et al., 2024；Yu et al., 2026）**：回答级稀疏奖励，需要可验证reward；OPSA放弃该信号而获得更优结果，定位差异在token级无监督。
- **OPD（Agarwal et al., 2024；Gu et al., 2024；Lu & Lab, 2025）**：反向KL教师蒸馏；本文发现其增益主要来自低p token抑制，重新解读K1估计器的真实作用。
- **OPSD（Zhao et al., 2026；Hubotter et al., 2026；Shenfeld et al., 2026）**：用hint构造自教师；仍依赖额外标注/采样，本文证明无需hint可达同等甚至更优。
- **NSR（Zhu et al., 2026）**：负样本强化，但仍是轨迹级且需正确/错误划分；OPSA在token级无监督且按熵自适应。
- **TTRL（Zuo et al., 2026）**：test-time训练，易陷局部最优导致Pass@k下降；OPSA全程on-policy且多样性更好。
- **Intuitor/自我奖励（Yuan et al., 2024；Ding & Zhang, 2026；Zhao et al., 2025b）**：依赖初始能力与轨迹级奖励；若正确答案远离mode易坍塌，OPSA用细粒度token信号规避该风险。

## 局限性与未来方向
- 实验仅覆盖≤9B小模型，是否可扩展至更大参数或MoE架构未验证（§A）。
- 对已过度锐化、极低熵的重度post-trained模型，OPSA主要做质量重分配，增益可能有限。
- OPSA未显著扩展潜在探索前沿，thinking-mode下Pass@k提升相对保守。
- 未结合其他RL方法协同，作者建议将其细粒度负advantage与其他RL合并。
- OPD的K1估计器机制仍需更严格理论分析以厘清真实增益来源。

## 研究启发与可借鉴点
- **噪声敏感性实验范式**：按噪声轨迹分片训练对比，可量化任何"监督方法"的真实依赖度，值得移植到其他蒸馏/对齐工作中。
- **token-level梯度解剖**：利用logit梯度公式识别近零梯度区（高logp + 小advantage），避免无效优化区，可推广至SFT/RL混合训练。
- **熵自适应信号构造**：用模型自身不确定性动态调节学习强度，无需外部标注即可实现细粒度credit assignment，对label-free学习有通用启发。
- **fork-token聚焦**：识别反射词集合（wait/but/check等）并掩码消融，验证"分叉处重分配"假说，可迁移至CoT/反思行为分析。
- **冷启动价值**：OPSA作为"免费午餐"初始化可兼容GRPO等后续训练，提示无监督预适应可作为任何RL pipeline的插件前缀。

## 关键术语表
- **On-Policy Distillation（OPD）**：在学生自身采样轨迹上用反向KL从强教师获取token级advantage进行蒸馏训练。
- **On-Policy Self-Adaptation（OPSA）**：本文提出的无监督方法，仅用学生token熵构造自适应负advantage，无需外部教师/奖励/hint。
- **Reinforcement Learning with Verifiable Rewards（RLVR）**：基于可验证回答奖励的RL，如GRPO/DAPO，提供稀疏轨迹级信号。
- **On-Policy Self-Distillation（OPSD）**：用hint条件化自学生作为替代教师的蒸馏范式。
- **Negative Sample Reinforcement（NSR）**：对错误轨迹施加固定负advantage的RL变体。
- **Token Entropy（token熵）**：模型在给定上下文处输出分布的不确定性度量，本文用于决定负advantage强度。
- **K1 Estimator**：估计学生-教师反向KL的采样估计器，OPD的核心计算组件。
- **Fork Token（分叉token）**：CoT中引发不同推理路径的关键位置，常伴随反思词，OPSA在此处做均匀重分配。

## 可复现要素
- **数据集**：训练DAPO-17k（Yu et al., 2026）；评估AIME24/AIME25/HMMT25/MBPP+/GPQA-Diamond——公开可用。
- **代码**：基于slime框架（Zhu et al., 2025，GitHub: THUDM/slime），论文提供训练与评估超参（附录表4/5）。
- **模型**：Qwen3-1.7B/4B、Qwen3.5-9B——开源权重可从官方获取。
- **关键超参**：最低20% logp token训练；学习率1×10⁻⁶；rollout batch 64；training temperature 1.0/top-p 1.0/max 12000；eval temperature 0.7/top-k 20/top-p 0.8/max 32768；save/validate每20步；Checkpoint按val集Avg@4选取。
- **硬件**：8×NVIDIA H100/H200。
- **框架版本**：slime 0.2.4、Megatron 0.16.0rc0、SGLang 0.5.14。
