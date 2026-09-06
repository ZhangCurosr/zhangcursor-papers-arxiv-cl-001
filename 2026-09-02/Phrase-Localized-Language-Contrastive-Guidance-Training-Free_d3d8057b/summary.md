---
title: "Phrase-Localized-Language-Contrastive-Guidance-Training-Free"
source: https://arxiv.org/pdf/2609.01016v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:18:30"
---

# 论文速读：Phrase-Localized Language-Contrastive Guidance: Training-Free Localized Accent Control for Code-Switching Text-to-Speech

## 一句话总结
本文提出一种无需训练的推理时介入框架（LCG），通过自注意力探测动态提取语码切换短语的声学帧掩码，并将局部语言对比引导强度与全局文本引导解耦，在多语言离散扩散TTS中精准恢复嵌入短语的原生口音，同时保持主句流畅度与说话人身份一致性。

## 研究问题与动机
- **核心问题**：零样本多语言TTS模型（如OmniVoice）在处理句内语码切换（Code-Switching, CS）时，嵌入的外语短语会被主语言口音严重同化（accent leakage），导致发音不自然、声学失真。
- **现有方法不足**：
  1. 传统CS TTS依赖双语音素后验、跨语言嵌入或多阶段微调，训练成本高且泛化受限；近期大模型（如X-Voice）虽引入双级语言注入架构，但仍需海量多语言语料微调，且作者自承未能充分优化句内密集短语切换。
  2. 标准分类器自由引导（CFG）在全序列全局统一缩放，缺乏空间分辨率，会强行抹平局部语言边界，加剧外语短语的口音塌陷。
  3. 现有推理时控制方法多针对文本或图像领域，尚未将帧级空间引导范式引入离散非自回归语音语言
