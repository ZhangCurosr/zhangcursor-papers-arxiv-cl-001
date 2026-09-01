---
title: "From-Storage-to-Access-Verifiable-Activation-of-Parametric-K"
source: https://arxiv.org/pdf/2608.18581v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:52:39"
---

# 论文速读：From-Storage-to-Access-Verifiable-Activation-of-Parametric-K

## 一句话总结
提出VAKE（Verifiable Activation of Parametric KnowledgE）两阶段强化学习框架，通过显式“启动”(Priming)在稀疏检索子图上插入可验证的桥接三元组以激活LLM中“存储但不可访问”的参数化事实，再将激活能力迁移至隐式“推理”(Reasoning)阶段进行直接CoT问答；在7个多跳/单跳QA基准上持续超越推理时方法与训练时RL基线，且具备OOD泛化与通用能力无损特性。

## 研究问题与动机
1. **核心问题**：LLM预训练已将大量事实编码进参数，但后训练阶段常出现“存储但不可访问”(stored but inaccessible)的知识召回失败，导致多跳问答等任务性能触顶。
2. **现有方法混淆归因**：推理时方法（Self-Ask、RECITE、Step-Back等）与训练时RL方法（Unlock、标准GRPO等）均依赖自由文本生成，知识激活与答案推理过程纠缠，无法判定正确答案是来自参数知识激活还是上下文复制/数据集记忆。
3. **现有方法测量缺陷**：传统准确率评估混淆“编码失败”与“召回失败”；内部探针虽显示隐藏层含事实信号，但缺乏可控的外部干预与可观察的激活证据。
4. **动机**：借鉴认知科学的“启动效应”(priming effect)，假设通过在检索子图上引入独立的线索驱动激活阶段，可唤醒沉睡的参数知识，并通过冻结回答者的干预设计实现严格归因。

## 核心贡献（创新点）
1. **两阶段分离框架**：首次将参数知识激活与直接问答推理在训练上解耦，Priming显式外化可验证三元组，Reasoning将激活能力内化至隐式CoT，二者可串联亦可独立使用。
2. **可观察且可归因的激活表征**：以离散关系三元组作为激活输出，配合冻结回答者$M_0$的干预，使插入内容的答案增益可直接归因于参数知识激发，彻底消除上下文复制的测量混淆。
3. **与标准后训练管线无缝兼容**：作为RL增强模块可平滑接入现有SFT/GRPO流程；实验证明其在多跳QA上获得增益的同时，不影响数学、指令遵循、通用知识等基础能力，且能从HotpotQA直接迁移至OOD单跳/多跳数据集。

## 方法详解
- **问题形式化**：给定问题$q$、不充分检索子图$c=\mathcal{R}(q)$，目标是从参数中激发知识生成增量$\mathcal{I}$，满足$M_0(q,c)\neq a^\star$但$M_0(q,c\cup\mathcal{I})=a^\star$。$\mathcal{I}$由策略$\pi_\theta$生成而非外部检索，来源为参数。
- **Stage I: Priming (插入-后回答)**：$\mathcal{I}\sim\pi_\theta(\cdot|q,\mathcal{R}(q))$，冻结回答者$M_0$在$(q,\mathcal{R}(q)\cup\mathcal{I})$上生成$\hat{a}$。$M_0$与$\mathcal{R}(q)$固定，$\mathcal{I}$是唯一自由变量，奖励仅通过$\hat{a}$正确性反传至$\pi_\theta$。
- **Stage II: Reasoning (直接推理)**：以Stage I权重$\theta^{(1)}$初始化策略，直接在$(q,\mathcal{R}(q))$上生成CoT与答案$\hat{a}\sim\pi_\theta(\cdot|q,\mathcal{R}(q))$，使用GRPO优化，检验显式激活能力向隐式推理的迁移。
- **奖励函数**：$r^{(s)}(\hat{a},a^\star;o)=r_{\mathrm{qa}}+r_{\mathrm{fmt}}^{(s)}$。$r_{\mathrm{qa}}$软融合exact match(0.6)与token-level F1(0.4)，缓解硬0/1信号稀疏；$r_{\mathrm{fmt}}^{(s)}$为格式塑造项，符合三元组/推理格式给小加分，严重错
