---
title: "SelFusion-Self-distillation-for-Diffusion-Language-Models"
source: https://arxiv.org/pdf/2608.22898v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:00:19"
field: "语言模型压缩与加速"
keywords: ["扩散语言模型", "知识蒸馏", "自蒸馏", "双向蒸馏", "RMSNorm校准"]
innovations: ["提出双模式自蒸馏框架实现DLM内部知识迁移", "设计token级双向KD机制动态选择蒸馏方向", "引入轻量RMSNorm校准解决过自信问题"]
benchmarks: ["Dolly", "Self-Inst", "Vicuna", "S-NI", "UnNI", "SAMSum"]
---

# 论文速读：SelFusion-Self-distillation-for-Diffusion-Language-Models

## 一句话总结
论文提出SelFusion，一种面向扩散语言模型（DLMs）的新型自蒸馏框架，通过在同一模型内构建"easy mode"（低掩码）和"hard mode"（高掩码）两个模式，并引入基于token级别正确性的双向知识蒸馏，有效克服了传统外部教师蒸馏在DLMs上效果不佳的问题。

## 研究问题与动机
- **DLMs生成质量瓶颈**：扩散语言模型虽具备并行解码优势可实现快速推理，但因非自回归性质，其生成质量显著低于自回归大语言模型（AR LLMs），存在10-32%的perplexity差距。
- **传统KD在DLMs上失效**：论文实证发现，直接将logit-level KD或sequence-level KD从LLM或DLM教师迁移到DLM学生时，仅获得边际提升甚至导致性能下降。
- **分布不匹配问题**：AR模型对top-1 token赋予近100%概率，而DLM在50%掩码率下仅约60%，这种陡峭分布差异阻碍了有效的logit级知识转移。
- **DLM教师质量不足**：现有DLM教师模型本身生成质量有限，即使增加去噪步数也无法提供足够高质量的知识来源。

## 核心贡献（创新点）
- **双模式自蒸馏框架**：利用DLM的去噪过程在同一模型内构建easy和hard两种模式，无需外部教师即可实现知识蒸馏，本质区别在于打破了传统KD对外部强教师的依赖。
- **Token级双向KD机制**：根据每个token的正确性和置信度动态决定蒸馏方向，而非固定单向传递，解决了"easy模式并非总是更准确"的关键问题。
- **RMSNorm-based Logit校准**：仅增加1,280参数即在easy模式末尾插入RMSNorm层，选择性抑制错误预测的过度自信（从~40%降至~10%），同时保留正确预测的置信度。
- **完整的训练效率分析**：SelFusion消除教师训练阶段，总计算量较传统KD方法减少约2倍，同时保持竞争力性能。

## 方法详解
- **双模式掩码策略**：对每个样本采样两个噪声级别$t_h$和$t_e$（其中$t_e \sim U(0, t_h)$），确保easy模式掩码率始终低于hard模式，使easy模式可见更多上下文信息。
- **双向蒸馏方向判定**：对于每个token位置$i$，根据两个模式的正确性指示$c_h$和$c_e$动态确定蒸馏方向：两者都正确时选高概率模式为教师；两者都错误时选更接近ground-truth的模式；仅一个正确时直接使用该模式。
- **扩散损失函数**：对easy和hard模式分别计算重加权扩散token预测损失$\mathcal{L}_{\mathrm{diff}}^{(h)}$和$\mathcal{L}_{\mathrm{diff}}^{(e)}$，公式包含掩码概率$q_i^{(m)}$的逆权重。
- **双向KD损失**：使用温度缩放的KL散度$\mathcal{L}_{\mathrm{bkd}} = \frac{T^2}{|\mathcal{D}|}\sum_{i\in\mathcal{D}}\mathrm{KL}(P_{i,T}^{(j)}||P_{i,T}^{(k)})$，其中$(j,k)$由动态方向决定。
- **总目标函数**：$\mathcal{L}_{\mathrm{SelFusion}} = \mathcal{L}_{\mathrm{diff}}^{(h)} + \mathcal{L}_{\mathrm{diff}}^{(e)} + \mathcal{L}_{\mathrm{bkd}}$，两个模式共享参数，单次反向传播联合更新。
- **RMSNorm校准设计**：在easy模式的最终隐藏状态和LM head之间插入RMSNorm层，针对正确预测适度降温（90%→60%），对错误预测强力压制（40%→10%），约75%的过置信抑制率。

## 实验与结果
- **数据集**：训练集Databricks-Dolly-15K（12K训练/500验证），测试集包括Dolly、Self-Inst、Vicuna、S-NI和UnNI五个指令遵循基准。
- **模型配置**：使用SMDM架构，学生模型472M参数，教师模型1,476M参数（TinyLlama作为LLM教师），训练20轮。
- **最强结果**：SelFusion在16步设置下，Self-inst达到12.87（超越LLM教师的11.04），Vicuna达到17.08（超越LLM教师的14.94），Sinst达到27.07（超越LLM教师的23.21），Uinst达到27.86（超越LLM教师的27.13）。
- **相对提升**：相比最强基线LLM-to-DLM序列级KD提升2-4分（约16%相对改进），相比DLM SFT基线在所有五个基准上稳定提升1.5-3分。
- **泛化验证**：在32步扩散设置和SAMSum摘要任务上同样取得最佳性能，证明方法通用性。
- **推理速度**：即使32步扩散，DLM仍比AR模型快1.5倍；8步时加速达5.9倍。

## 相关工作脉络
- **MiniLLM**：通过反向KL散度和on-policy优化改进指令跟随任务的LLM蒸馏，本文将其思想扩展至DLM领域并解决分布不匹配问题。
- **GKD**：探索使用学生生成样本的on-policy蒸馏，本文采用内部双模式替代外部教师，消除教师训练成本。
- **自蒸馏研究（Hahn & Choi, 2019; Yang et al., 2024）**：先前工作聚焦AR模型，本文首次系统解决DLM特有的分布不匹配和过自信问题。
- **LLM-to-DLM蒸馏**：本文揭示AR教师与NAR学生间的top-k概率尺度和token重叠双重不匹配，指出直接logit匹配失效的根本原因。
- **DLM预训练蒸馏**：先前DLM蒸馏工作集中于预训练阶段，本文填补了post-training蒸馏研究的空白。
- **SMDM与LLaDA**：作为DLM基础架构，本文在其上验证蒸馏有效性，推动了DLM实用化进程。

## 局限性与未来方向
- **DLM骨干网络选择有限**：当前可用的DLM预训练模型较少，限制了在更广泛架构上的验证。
- **仅评估英语指令微调**：未测试多语言场景和其他任务类型（如代码生成、数学推理）。
- **掩码策略固定**：easy/hard模式的掩码比例关系固定，可能不是最优配置。
- **扩展性待验证**：在更大参数规模（如7B以上）和更长序列上的表现尚未探索。

## 研究启发与可借鉴点
- **自蒸馏替代外部教师**：对于难以获取高质量教师模型的场景，可利用模型自身不同配置构造互补知识源，降低训练成本和依赖。
- **动态蒸馏方向选择**：按样本/token级别的可靠性动态调整知识流向，比固定单向蒸馏更鲁棒，可迁移至其他蒸馏场景。
- **过自信抑制技术**：RMSNorm轻量校准方案仅需1,280参数即可显著改善蒸馏效果，为类似问题提供简洁解决方案。
- **DLM与AR分布差异分析**：揭示top-k概率尺度不匹配和token重叠度下降机制，为跨架构蒸馏提供理论指导。
- **训练效率权衡**：消除教师训练阶段可将总计算量减半，在资源受限场景下具有重要实用价值。

## 关键术语表
**Diffusion Language Models (DLMs)**：基于扩散过程的离散语言模型，通过逐步去噪生成文本，支持并行解码。
**Knowledge Distillation (KD)**：将教师模型的知识迁移到学生模型的技术，分为logit-level和sequence-level两类。
**Easy/Hard Mode**：SelFusion中的两种前向模式，easy模式掩码率低（上下文多），hard模式掩码率高（上下文少）。
**Bidirectional KD**：根据token级别正确性动态决定蒸馏方向的知识蒸馏机制。
**RMSNorm Logit Calibration**：在easy模式输出端插入的归一化层，用于抑制过度自信预测。
**Masking Ratio**：被替换为[MASK]标记的token比例，控制模型的可见上下文信息量。
**Classifier-Free Guidance (CFG)**：DLM推理时的条件调控参数，本文固定为1.0。

## 可复现要素
- **数据集**：Databricks-Dolly-15K（公开），评估基准Dolly/Self-Inst/Vicuna/S-NI/UnNI均公开；SAMSum摘要数据集公开。
- **代码**：官方代码已开源，链接https://github.com/scai-research/SelFusion_official。
- **模型**：基于SMDM架构（472M/1476M参数）和TinyLlama（1476M参数），均提供预训练权重。
- **关键超参**：学习率网格搜索{5e-5, 1e-4, 2e-4}，选定5e-5；全局batch size 512；warmup ratio 0.05；weight decay 0.1；Adam betas (0.9, 0.95)；gradient clip 1.0；训练20轮；评估使用seed {10, 20, 30}取平均。
