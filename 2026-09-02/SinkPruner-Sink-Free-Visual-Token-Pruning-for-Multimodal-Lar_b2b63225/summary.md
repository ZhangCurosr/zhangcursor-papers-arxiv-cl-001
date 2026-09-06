---
title: "SinkPruner-Sink-Free-Visual-Token-Pruning-for-Multimodal-Lar"
source: https://arxiv.org/pdf/2609.01004v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 20:57:54"
field: "多模态大语言模型效率优化"
keywords: ["视觉token剪枝", "多模态大语言模型", "注意力汇聚", "无需训练", "高效推理"]
innovations: ["发现高范数异常token的冗余性并量化其影响", "提出级联剪枝框架SinkPruner，先净化视觉流再文本引导剪枝", "设计可迁移的视觉净化器模块，作为即插即用组件增强现有方法"]
benchmarks: ["GQA", "MMBench", "MME", "POPE", "SQA", "VQAv2", "TextVQA", "MMStar", "MMMU", "AI2D", "MM-Vet", "MVBench", "SEED-Bench", "NextQA", "VideoMME"]
---

# 论文速读：SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
论文提出了一种无需训练的级联视觉token剪枝框架SinkPruner，通过预过滤高范数异常token净化视觉流，显著降低多模态大语言模型的推理计算开销，同时保持高准确率（在LLaVA-1.5上剪枝89%仅损失3.5%性能）。

## 研究问题与动机
- 多模态大语言模型（MLLMs）将图像/视频编码为超长视觉token序列，与文本拼接后经Transformer解码器处理，带来二次复杂度计算开销，阻碍资源受限环境部署。
- 现有视觉中心剪枝方法（如VisionZip）依赖CLS注意力选择重要token，但会被高范数异常token误导——这些token来自非语义背景区域，因异常大特征范数吸引过高注意力，挤占token预算。
- 文本引导剪枝方法（如FastV、SparseVLM）在LLM解码器利用文本-视觉注意力选择token，但受注意力汇聚（少数token吸引不成比例注意力）和注意力分散（跨模态注意力过度弥散）干扰，导致选择不可靠。
- 高范数异常token在空间维度和特征维度均高度冗余，且阻碍下游文本引导剪枝的可靠性，现有方法未系统解决该问题。

## 核心贡献（创新点）
1. **发现并量化高范数异常token的冗余性**：通过特征范数分布、空间邻居相似性和内部成对相似度分析，证明高范数token主要来自同质背景区域且特征坍塌，与现有剪枝方法依赖注意力的假设形成本质对比。
2. **提出无需训练的级联剪枝框架SinkPruner**：采用粗到细设计，先通过视觉净化器过滤高范数异常，再经文本引导剪枝器保留语义对齐token，区别于现有单一阶段的剪枝方法。
3. **设计视觉净化器缓解注意力汇聚与分散**：聚合高范数异常token为单个sink token，并结合显著性选择和多样性去重筛选低范数token，为下游文本引导剪枝提供纯净视觉输入。
4. **验证视觉净化器的强可迁移性**：作为即插即用模块集成到现有视觉中心剪枝框架（如VisionZip）中，在不同剪枝率下均能持续提升性能。

## 方法详解
- **高范数异常token识别**：计算每个视觉token的L2范数，按相对排名取top ρ fraction（默认ρ=1%）作为高范数集合X_high，剩余为低范数候选集X_low；该scale-free规则无需绝对阈值校准。
- **视觉净化器**：
  1. **高范数聚合**：将X_high中token平均池化为单个proxy sink token \(x_{sink} = \frac{1}{|X_{high}|}\sum_{x \in X_{high}} x\)，压缩冗余同时保留全局信息。
  2. **显著性选择**：从X_low中选取CLS注意力得分最高的top \(k_{res}\)个token构成\(X_{res}\)。
  3. **多样性选择**：对剩余候选R进行相似度去重，迭代选取与已选集合S余弦相似度最低的token加入\(X_{div}\)，避免特征坍塌。
  4. 最终净化视觉表示\(Z = [x_{sink}, X_{res}, X_{div}]\)。
- **文本引导剪枝器**：在LLM解码器早期层，计算每个视觉token \(z_j\)的全局相关性得分\(\tilde{p}_j = \frac{1}{L_t}\sum_{i=1}^{L_t} \text{Softmax}(\mathbf{Q}_{text} \cdot \mathbf{K}_{vis}^\top)_{i,j}\)，保留得分最高的top K token。

## 实验与结果
- **数据集**：12个图像语言基准（GQA、MMBench、MMB-CN、MME、POPE、SQA、VQAv2、TextVQA、MMStar、MMMU、AI2D、MM-Vet）和4个视频语言基准（MVBench、SEED-Bench、NextQA、VideoMME）。
- **评估基线**：ToMe、FastV、MustDrop、PDrop、VisionZip、SparseVLM、HoloV、ApET、VisPruner、DART、DivPrune、MMTok等。
- **主要结果**：
  - LLaVA-1.5-7B上保留64个token（剪枝率88.9%）：平均性能保持96.5%，超越VisionZip（+3.3%）和HoloV（+1.8%）；保留32个token（剪枝率94.4%）：保持91.2%，超越VisPruner（+4.0%）。
  - Qwen2.5-VL-7B上剪枝率66.7%、77.8%、88.9%时分别保持98.6%、96.3%、91.8%性能。
  - 视频任务（Qwen2.5-VL-7B，80%剪枝率）：平均性能保持98.0%，优于DART和DivPrune。
  - 难推理基准（MMStar、MMMU、AI2D、MM-Vet，32 token预算）：保持93.7%平均性能，显著优于VisionZip（88.7%）和FastV（85.9%）。
- **消融实验**：移除视觉净化器导致MMB性能下降10.2%；移除高范数过滤造成MME从1754降至1705；移除多样性去重影响较小。
- **可迁移性**：将高范数过滤器集成到VisionZip，在32/64 token预算下MMB和POPE性能提升0.9%-2.6%。

## 相关工作脉络
- **视觉中心剪枝**（如VisionZip、Faster-VLM）依赖CLS注意力，但未考虑高范数异常token对注意力的干扰，本文通过预过滤解决该问题。
- **文本引导剪枝**（如FastV、SparseVLM）直接利用LLM解码器注意力，受注意力汇聚和分散影响；本文先净化视觉流再引导，改善注意力可靠性。
- **渐进/层次化剪枝**（如MustDrop、PDrop、HiDrop、AutoPrune）关注动态调整保留策略，但未解决高范数冗余对剪枝决策的扭曲。
- **Token合并/聚类**（如ToMe、PruMerge）基于特征相似度压缩，而本文识别并聚合异常token而非普通冗余token。
- **定位差异**：SinkPruner首次系统性地将高范数异常过滤与文本引导剪枝级联，从上游净化视觉表征，使下游注意力更可信。

## 局限性与未来方向
- 当前评估仅针对离线推理（固定长度预记录输入），尚未处理实时流式视频等在线场景，其中历史动态更新且剪枝决策不可回溯。
- 高范数过滤对场景文本敏感：高分辨率分块编码中，小型字符patch可能具有高范数，与背景异常混合聚合，影响TextVQA等OCR任务性能（在LLaVA-NeXT上TextVQA表现略逊于VisPruner）。
- 未来方向包括：将净化器扩展至在线流式处理、设计文本感知的高范数豁免机制（如基于局部边缘密度）。

## 研究启发与可借鉴点
1. **可复用方法**：基于相对排名的scale-free top-ρ规则识别异常token，无需绝对阈值校准，可迁移至不同视觉编码器（如DINOv2）。
2. **实验设计借鉴**：通过注意力熵和sink比例等量化指标分析现有方法缺陷，为后续研究提供评估视角；消融实验设计清晰分离各模块贡献。
3. **创新机会**：高范数过滤思想可应用于其他多模态压缩任务（如长视频理解、3D场景表示），或与文本感知结合优化OCR敏感任务。

## 关键术语表
- **高范数异常token**：特征L2范数异常大的视觉token，通常来自非语义背景区域，在注意力机制中产生虚假高峰，导致剪枝决策失真。
- **视觉净化器**：SinkPruner中的预过滤模块，聚合高范数异常token并筛选低范数token，生成纯净且多样化的视觉表示。
- **注意力汇聚**：LLM解码器中少数token吸引不成比例大量注意力的现象，使剪枝方法错误保留非语义token。
- **注意力分散**：跨模态注意力过度弥散，导致模型难以形成对查询相关区域的自信排序。
- **scale-free top-ρ rule**：基于相对排名而非绝对值的异常token识别规则，适用于不同特征范数尺度的视觉编码器。
- **即插即用模块**：指视觉净化器可无缝集成到现有剪枝框架中，无需重新训练即可提升性能。

## 可复现要素
- **数据集**：公开的多模态基准数据集（GQA、MMBench、MME、POPE、VQAv2、TextVQA、MMStar、MMMU、AI2D、MM-Vet、MVBench、SEED-Bench、NextQA、VideoMME）。
- **代码/权重**：代码已开源（https://github.com/LaVi-Lab/SinkPruner），基于PyTorch 2.1.2和CUDA 12.1。
- **关键超参**：高范数比例ρ=1%；低范数选择池大小与目标token预算匹配；剪枝层为(2,6,15)；渐进保留调度见论文表9和表10。
