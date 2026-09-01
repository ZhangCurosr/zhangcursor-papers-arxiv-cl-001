---
title: "Understanding-Multilingual-Medical-ASR-Adaptation-Through-La"
source: https://arxiv.org/pdf/2608.18825v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:56:40"
---

# 论文速读：Understanding-Multilingual-Medical-ASR-Adaptation-Through-La

## 一句话总结
本文系统评测了零样本、单语微调、两阶段及直接多语言微调四种策略在多款 Whisper 模型上的医学语音识别性能，并结合逐层编码器分析揭示适配过程对内部表征的重塑规律；发现英语医学微调是驱动表征重构的主导力量，而多语言续训主要保留已适配空间，同时随着 WER 下降，编码器中对转录错误的线性可预测性显著减弱。

## 研究问题与动机
- **核心问题**：大型预训练 ASR 模型（如 Whisper）经过医学域与多语言微调后，其内部编码器表征发生了怎样的层次化重组？仅凭 WER 指标是否足以刻画适配效果？
- **现有方法不足**：
  1. **研究路线割裂**：领域适配（Domain Adaptation）与表征可解释性（Representation Analysis）长期独立发展，缺乏对医学多语言微调过程内部机制的系统性解读。
  2. **跨域泛化瓶颈**：通用 ASR 在医疗场景下表现差（如 Wav2Vec2 CTC 英语 WER >92%），急需针对性微调，但不同模型规模与微调路径的优劣缺乏统一基准与理论解释。
  3. **多语言评估不可比**：既有工作（如 MultiMed）的数据集、语言与解码设置差异大，难以横向对比；且两阶段微调（先英语后混合）与直接多语言微调的本质差异尚未被表征分析揭示。

## 核心贡献（创新点）
1. **系统性多规模多策略对比**：在四档 Whisper（Base/Small/Medium/Large-v3）上全面评测零样本
