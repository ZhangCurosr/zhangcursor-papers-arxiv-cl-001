---
title: "Beyond-the-Stability-Exploration-Dilemma-Environmental-Regul"
source: https://arxiv.org/pdf/2608.23311v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:39:57"
field: "大语言模型强化学习"
keywords: ["环境正则化", "Query-KL", "RLVR", "策略优化", "训练稳定性"]
innovations: ["将正则化从动作侧迁移到输入侧解耦稳定性与探索性", "提出Query-KL正则化与参考派生查询加权机制", "零额外前向成本的估算器无关设计"]
benchmarks: ["AIME24", "AIME25", "AMC", "MATH500", "Minerva", "OlympiadBench"]
---

# 论文速读：Beyond-the-Stability-Exploration-Dilemma-Environmental-Regul

## 一句话总结
论文提出环境正则化策略优化（ERPO），通过将正则化从动作侧（Policy-KL）迁移到输入侧（Query-KL），解耦训练稳定性与探索性，在保持高温度解码能力与长程训练稳定性的同时显著提升数学推理准确率。

## 研究问题与动机
1. **稳定性-探索性困境**：现有LLM策略优化依赖动作侧Policy-KL约束，导致探索预算被消耗；若放弃Policy-KL则缺乏显式漂移控制。
2. **环境非平稳性**：训练过程中策略诱导的查询分布 $\rho_\theta$ 会偏离预强化学习参考分布 $\rho_{\theta_0}$，引发梯度方差放大与策略崩溃。
3. **已有方法局限**：RLHF主流方案聚焦动作空间正则化，忽略输入查询分布控制；EVA/Align-Pro等方法未直接约束Query-KL。
4. **温度敏感性**：高温度解码时LLM对解码分布长尾敏感，查询分布漂移会导致训练-推理性能差距扩大（奖励黑客现象）。

## 核心贡献（创新点）
1. **Query-KL正则化**：首次将环境漂移控制从动作侧迁移至输入侧，通过约束 $\text{KL}(\rho_\theta \| \rho_{\theta_0})$ 实现环境稳定性。
2. **结构解耦理论**：证明QKL梯度仅通过查询似然 $\nabla_\theta \ell_\theta(q)$ 传播，不与响应策略得分函数耦合，保留动作探索自由度。
3. **参考派生查询加权**：提出静态数据集权重 $w(q) \propto \ell_{\theta_0}(q)$ 降低梯度方差，改善高温度解码鲁棒性。
4. **估算器无关设计**：ERPO可无缝集成到GRPO/PPO/REINFORCE流水线，无需额外前向传播或架构修改。
5. **多温度评估框架**：建立跨温度范围（0.1-1.5）的Avg@k/Pass@k评估协议，揭示长程训练中的稳定性差异。

## 方法详解
**Query-KL正则化**：
- 定义策略诱导查询分布 $\rho_\theta(q) = P_\theta(q)$（自回归序列似然）
- 正则项 $\mathcal{R}_{\text{query}}(\theta) = \mathbb{E}_{q \sim \rho_\theta}[\log \rho_\theta(q) - \log \rho_{\theta_0}(q)]$
- 梯度闭式解：$\nabla_\theta \mathcal{R}_{\text{query}}(\theta) = \mathbb{E}_{q \sim \rho_\theta}[(\ell_\theta(q) - \ell_{\theta_0}(q)) \nabla_\theta \ell_\theta(q)]$

**查询重加权**：
- 理想重要性权重 $w^*(q) = \rho_{\theta_0}(q)/\rho_{\text{train}}(q)$
- 实用近似：$w(q) = \text{clip}(\bar{s}/s_i, 0, 2)$，其中 $s_i = -\log p_{\theta_0}(q_i)$
- 权重仅作用于外层查询循环，不进入目标函数

**损失函数**：
$$\widehat{L}_{\text{ERPO}}(\theta) = -\frac{1}{m}\sum_{q \in B} w_B(q) \bar{g}_\theta(q) + \alpha \widehat{\mathcal{R}}_{\text{query}}(\theta)$$

## 实验与结果
- **数据集**：MATH Level 3-5（~8.5K示例），测试集含AIME24/25、AMC、MATH500、Minerva、OlympiadBench
- **基线**：Vanilla GRPO、GRPO+Policy-KL、DAPO、RLOO
- **模型**：Qwen2.5-Math-7B/32B
- **关键结果**：
  - Avg@32平均提升6.2%（AIME24: +4.4%，MATH500: +14.9%）
  - Pass@1提升5.69%，Pass@32提升3.64%
  - 训练-推理一致性差距减少51%（6.47%→3.14%）
  - 1000步长程训练中，ERPO在高温度下性能下降幅度显著小于GRPO

## 相关工作脉络
1. **RLVR方法**（GRPO/DAPO/RLOO）：本文在动作侧正则化缺失下，从输入侧补充稳定性控制
2. **Prompt分布方法**（EVA/Align-Pro）：聚焦提示优化而非查询分布漂移约束
3. **数据重加权**（StablePrompt/WPO）：处理静态数据分布，未考虑策略诱导的动态漂移
4. **非平稳RL**：将经典初始状态分布控制思想迁移到LLM查询空间
5. ** imitation learning**（DAgger）：协变量偏移motivation相似，但本文针对LLM自回归特性设计

## 局限性与未来方向
1. **验证范围有限**：主要在数学推理 benchmark 和Qwen系列验证，指令遵循/代码生成/多语言场景待验证
2. **超参数搜索不充分**：正则化强度 $\alpha$ 未做系统调优
3. **估算成本**：依赖查询似然估计，计算开销可能随数据选择机制和模型规模变化
4. **理论假设**：A1假设（保持查询分布对齐利于泛化）需更多实证支持

## 研究启发与可借鉴点
1. **解耦设计范式**：将稳定性约束与探索自由度分离的思路可迁移到多模态RL训练
2. **温度敏感性评估**：多温度聚合评估协议可成为RLVR论文的标准化评测流程
3. **环境非平稳性分析**：Query-KL监控可作为RL训练稳定的诊断指标
4. **轻量化正则化**：零额外前向成本的设计原则适用于资源受限场景
5. **训练-推理一致性**：通过输入侧正则化缓解奖励黑客，对RLHF/RLAIF有普适价值

## 关键术语表
**Policy-induced query distribution**：策略 $\pi_\theta$ 诱导的查询分布 $\rho_\theta(q)=P_\theta(q)$，反映模型对训练查询的隐式偏好
**Query-KL (QKL)**：当前策略与参考策略在查询分布上的KL散度，用于约束环境漂移
**Environment regularization**：从输入侧而非动作侧施加正则化，解耦稳定性与探索性
**Reward hacking**：模型过度拟合训练分布特性导致推理性能下降的现象
**Policy-induced environment drift**：训练过程中 $\rho_\theta$ 偏离 $\rho_{\theta_0}$ 导致的训练环境非平稳性
**Estimator-agnostic**：方法独立于具体策略梯度估算器（GRPO/PPO/REINFORCE等）
**Pass@k / Avg@k**：采样k次答案中至少1个正确的概率 / 多次采样的平均正确率
**Temperature-sensitive stability**：模型在高分辨率解码温度下的性能保持能力

## 可复现要素
- **数据集**：MATH（Level 3-5），公开可用
- **代码**：https://github.com/alibaba/ERPO（已开源）
- **关键超参**：$\alpha=1\times10^{-2}$（默认），rollout batch=512，update batch=128，训练步数240/1000
- **模型**：Qwen2.5-Math-7B/32B，最大序列长度3K tokens
