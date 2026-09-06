---
title: "Post-Training-Science-for-Supervised-Fine-Tuning"
source: https://arxiv.org/pdf/2609.01244v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:42:04"
field: "大语言模型微调方法论"
keywords: ["Supervised Fine-Tuning", "LoRA", "Learning Rate Scaling", "Mixture-of-Experts", "Muon Optimizer", "SFT Scaling Laws"]
innovations: ["LoRA最优学习率10^-3在0.6-32B范围内完全平坦且跨家族可迁移", "MoE模型SFT性能由活跃参数与总参数的几何平均决定", "验证NLL在固定配方内可靠排序下游质量但flatness不辅助选择"]
benchmarks: ["IFEval", "customer-task-production-judges"]
---

# 论文速读：Post-Training-Science-for-Supervised-Fine-Tuning

## 一句话总结
本文通过系统性消融实验，测量了Qwen3和Llama系列模型在四个真实客户SFT数据集上的超参数影响，建立了可迁移的学习率规则、LoRA默认配置建议（rank=64, α=32），并揭示了MoE模型fine-tuning时的"密集等价尺寸"规律与epoch过拟合边界。

## 研究问题与动机
- SFT的决策链（学习率、batch size、LoRA vs FullFT、epoch数、优化器等）每次都要从零重新摸索，缺乏可迁移的经验法则
- 现有学习方法多基于预训练知识或他人在不同模型/数据集上获得的规则，未针对SFT场景系统测量
- 需要一套统一度量体系：在控制数据、split和seed不变的前提下，逐变量扫描以分离各杠杆的真实影响

## 核心贡献（创新点）
- **LoRA学习率平坦法则**：0.6–32B范围内最优LoRA LR恒定为10⁻³，跨家族迁移误差<0.004 nats，这是首次在大范围尺度上证实LoRA LR不随模型尺寸变化的发现
- **MoE密集等价尺寸的几何平均规律**：30B-A3B、80B-A3B、235B-A22B三个MoE模型在SFT损失曲线上均位于其活跃参数与总参数的几何平均处，优于单独使用活跃或总参数
- **Rank-64容量阈值与α=32全局最优**：rank>64后损失改善<0.001 nats但参数翻倍，α偏离32无论高低均损害性能
- **Epoch过拟合边界的分离度量**：验证NLL在两epoch后过拟合而任务judge分数不降，IFEval一般指令遵循能力持续侵蚀，提出"能力侵蚀"作为epoch数上限标准
- **Muon优化器在SFT中的有限迁移**：Muon在FullFT下达到略低NLL且最小值更平坦，但不提升任务质量，反而在IFEval上保留更多一般能力（+0.09）

## 方法详解
- **实验框架**：固定数据、split、seed，每次只变一个杠杆；970/1008个cell合格（排除全部为LoRA在LR=3×10⁻³失稳的情况）
- **学习率规则拟合**：LR = C · M_tuner · (2000/hidden_size)^(p+q_tuner)，通过cross-validation和bootstrap置信区间估计参数
- **MoE等效尺寸判定**：将MoE在默认配置下的损失映射到Qwen3密集模型损失曲线，找出最接近的密集等价尺寸
- **Flatness度量**：使用Fisher trace（单反向样本的二阶导数等价估计，公式Tr ∇²L = E[‖∇log pₓₜ‖²]），非完整Hessian但可在单反向中估算
- **数据生成**：四个数据集均采用iSFT（iterative SFT）生成，模型草稿→评价器打分→反馈→修订→通过，确保训练目标内部一致且下游judge与训练目标对齐

## 实验与结果
- **模型覆盖**：Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 32B)、Llama 3.1/3.2 (1B, 3B, 8B)，外加三个Qwen3 MoE (30B-A3B, 80B-A3B, 235B-A22B)
- **数据集**：security、leasing、support、docs四个匿名客户SFT数据集，辅助token差异18–23倍
- **核心结果**：
  - LoRA LR始终为10⁻³（95% CI涵盖0），FullFT最优约3×10⁻⁵，比值为33×
  - LoRA回收FullFT改进的98%（中位数），仅训练3.1–12.6%参数
  - Rank 64为容量饱和点，rank 32参数减半仅损失≤0.003 nats
  - α=32在四个cell中均为最优
  - MoE密集等价尺寸：30B-A3B≈8.5B（几何平均10.1B），80B和235B在token-heavy数据集上达到甚至超过32B密集模型
  - 验证NLL在固定(cell)内的Spearman ρₛ = −0.38至−0.88，可可靠排序下游质量
  - Muon最优LR约AdamW的1/3，Fisher trace低1.2–2.8倍，IFEval高0.09
  - 两epoch后NLL过拟合但任务分数不提升，四epoch时IFEval下降显著

## 相关工作脉络
- **Thinking Machines Lab (2025)** "LoRA without regret"：报告LoRA LR约为FullFT的10倍，本文扩展至全尺寸范围并发现LoRA LR实际平坦而非随尺寸变化
- **Zhang et al. (2024) ICLR** "When scaling meets LLM finetuning"：先前的finetuning scaling law，本文在其基础上引入MoE架构和更细粒度的超参数扫描
- **Liu et al. (2023) ICML** "Same pre-training loss, better downstream"：提出flat minima偏好理论，本文在SFT尺度检验发现flatness不辅助in-domain选择但track一般能力保留
- **Jordan et al. (2024) Muon**：结构感知优化器，本文首次系统测试其在SFT FullFT场景的迁移效果
- **Kalajdzievski (2023)**：指出LoRA的α/r缩放在大rank下梯度坍缩问题，本文验证α=32全局最优支持标准缩放
- **Muennighoff et al. (2023) NeurIPS**：数据受限语言模型scaling law，本文将其框架延伸至SFT并发现重复pass的边际收益低于新鲜数据

## 局限性与未来方向
- **数据与judge的非独立性**：训练数据由iSFT生成以通过judge，导致judge非独立于训练目标，可能高估loss作为选择指标的有效性
- **跨家族transfer受限**：验证NLL不能跨Qwen/Llama家族直接比较，需依赖improvement over baseline
- **Muon仅在FullFT测试**：未测试LoRA+Muon组合，Low-rank subspace下Muon的表现未知
- **示例数与token数混淆**：同一数据集内两者共变，无法分离"新增示例"vs"新增token"的真实效应
- **高学习率不稳定性边界**：稳定上限因模型而异（越大模型越敏感），未给出统一的稳定性判据
- **32B以上epoch扫面缺失**：大模型end-of-training评估超出单设备显存限制

## 研究启发与可借鉴点
- **iSFT数据生成范式**：迭代式"模型草稿→评价器打分→反馈→修订"可推广至其他需要高质量SFT数据的领域，确保训练目标与评估指标对齐
- **Fisher trace作为flatness代理**：单反向即可估算，可部署为训练监控指标，尤其在比较不同优化器或超参设置时捕捉隐式正则化差异
- **跨架构尺寸映射方法**：将MoE等混合架构的性能映射到密集模型的"等效参数轴"，为不同架构间的fair comparison提供可复用框架
- **Epoch选择的多指标分离**：区分"训练loss"、"任务judge"、"一般能力(IFEval)"三者的最优epoch可能不同，提出以"能力侵蚀"而非"loss过拟合"作为epoch上限标准
- **跨家族选择规则验证协议**：固定一个家族的超参规则，在另一家族blind测试并用regret（nats）量化transfer损失，可作为新规则的验证模板

## 关键术语表
- **LoRA (Low-Rank Adaptation)**：通过低秩分解矩阵增量适配LLM的轻量微调方法，训练参数极少但性能接近FullFT
- **iSFT (iterative Supervised Fine-Tuning)**：模型输出经迭代打分-反馈-修订循环生成的高质量SFT数据构建流程
- **Fisher Trace**：Fisher信息矩阵的迹，等价于收敛最小值处单样本梯度范数的期望平方，作为loss landscape flatness的代理
- **Muon Optimizer**：对2D权重矩阵进行牛顿-舒尔兹正交化的结构感知优化器，在pretrain中展现出计算效率优势
- **MoE (Mixture-of-Experts)**：包含大量专家层但每步只激活部分专家的稀疏架构，本文发现其SFT性能由活跃参数与总参数的几何平均决定
- **Geometric-mean Placement**：MoE模型在密集模型缩放律上的等效位置，等于√(active_params × total_params)
- **IFEval**：Instruction-Following Evaluation基准，测量LLM遵循通用指令模板的能力，用于检测SFT后的能力侵蚀
- **Cosine Decay Schedule**：学习率在训练过程中按余弦曲线衰减的策略，本文证明其显著优于constant LR尤其在较高LR时

## 可复现要素
- **数据集**：四个匿名客户SFT数据集，未公开（客户数据）
- **代码/权重**：模型为Qwen3和Llama开源家族，训练框架使用Megatron-swift；Muon实现来自NVIDIA Megatron-Core
- **关键超参**：LoRA默认lr=10⁻³, batch=8/16, rank=64, α=32；FullFT默认lr=3×10⁻⁵, batch=16；epoch上限建议≈2；所有实验使用cosine decay schedule
