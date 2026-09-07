---
title: "Building-and-Evaluating-Fixed-Voice-Thai-TTS-from-Synthetic"
source: https://arxiv.org/pdf/2609.03502v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 05:30:31"
field: "低资源语音合成"
keywords: ["Thai TTS", "low-resource speech", "knowledge distillation", "synthetic data", "fixed-voice TTS", "code-switching", "prosody evaluation"]
innovations: ["教师到学生的合成数据蒸馏路线", "多维度评估框架（Keyword Accuracy + Pause Accuracy）", "拒采恢复策略保持困难样本覆盖"]
benchmarks: ["Challenge-Set (1,531 items)", "WangchanThaiInstruct test split", "Common Voice 17.0 Thai", "LibriTTS English"]
---

# 论文速读：Building-and-Evaluating-Fixed-Voice-Thai-TTS-from-Synthetic

## 一句话总结
本文提出了一种利用大型语音克隆教师模型（OmniVoice）作为可编程数据源，将短语音参考（15秒）转化为高质量合成语音库，进而训练紧凑固定音色学生模型（Wayu-Paxa-TTS-Edge）的方法，实现了无需参考音频的可端侧泰英双语TTS部署。

## 研究问题与动机
- **低资源场景部署困境**：现有TTS系统通常需要在"大型语音克隆模型（需GPU推理）"与"紧凑固定音色系统（需单说话人语料）"之间二选一，缺乏第三种可行路线。
- **合成数据质量瓶颈**：教师模型的生成错误会直接成为训练目标，而过滤失败样本可能降低困难文本的覆盖度，需要精细的流水线设计。
- **泰语特有挑战**：泰语正字法不标记词边界、存在词汇声调、不规则专名和借词、非正式拼写、数字口语化以及泰英代码转换等问题，使评估和训练更具复杂性。
- **评估指标单一性**：句子级CER无法捕捉关键表达误读和停顿位置错误，需要多维度评估框架。

## 核心贡献（创新点）
- **教师到学生的蒸馏路线**：首次系统研究用零样本语音克隆教师模型生成合成语料训练固定音色学生模型，区别于仅用少量真实目标说话人音频微调的前人工作（Joshi & Garera, 2023）。
- **多维度评估框架**：提出超越CER的评估体系，包含Challenge-Set Keyword Accuracy、Prosody Pause Accuracy、Speaker Similarity和Speaking Rate四个互补维度，揭示单指标无法捕获的错误模式。
- **拒采恢复策略**：提出对过滤失败样本进行最多4次重渲染的rejection sampling机制，在保持质量的同时恢复困难文本覆盖度，实现Keyword Accuracy +2.0点的提升。
- **前端设计消融**：系统比较单语/双语前端、预训练初始化、质量过滤等组件的独立与组合效应，揭示各设计选择对正确性、韵律和声音相似性的具体影响。
- **方言迁移验证**：证明15秒Isan方言参考即可将声音身份和方言特征迁移至固定音色学生模型，为低资源方言TTS提供可行路径。

## 方法详解
**三阶段流水线**（Figure 1）：

1. **文本准备与口语化**
   - 文本来源：WangchanThaiInstruct（宽泛泰语覆盖）+ LLM关键词合成管道（针对困难表达）+ LibriTTS英语文本
   - LLM口语化器（deepseek-ai/DeepSeek-V4-Flash）：将歧义数字和嵌入英语 span 改写为发音导向的泰语或Tinglish文本
   - 按句子级别分块

2. **教师采样与质量过滤**
   - 使用OmniVoice Voice Design模式从12个说话人规格生成seed reference
   - 使用OmniVoice cloning模式渲染语料文本，每个voice生成多条候选 utterance
   - **内容正确性过滤**：使用CTC-based泰语ASR模型（airesearch/wav2vec2-large-xlsr-53-th）转录，与目标在音素空间比较；hard token定义为OOV或TLTK字典生词且National Corpus低频
   - **韵律与时长过滤**：拒绝停顿位置违规（Section 3.2定义允许位置）、语速偏离speaker-specific范围（~10%）、hard token时长异常压缩的候选
   - **拒采恢复**：过滤失败文本最多重渲染4次，保留最优结果

3. **学生模型训练**
   -  backbone：82M参数Kokoro（基于StyleTTS2架构）
   - **音素前端**：集成TLTK作为泰语grapheme-to-phoneme组件；四/五泰语声调复用已有轮廓token，仅低需要新embedding
   - **双语前端策略**：Latin span保持原样，泰语/英语span分别通过TLTK/Misaki路由，组合到Kokoro共享音素词汇
   - **训练配置**：AdamW，8 epochs，lr=1e-4（主模型）/1e-5（PL-BERT），effective batch size=8
   - **初始化**：PL-BERT、BERT projection、prosody predictor、text encoder、decoder从Kokoro checkpoint加载；style encoder从StyleTTS2-LibriTTS checkpoint加载

## 实验与结果
**评估数据集**：
- Challenge Set：1,531条测试句子，5类别（泰英代码转换391、专名310、生词410、非正式拼写210、长句210）
- CER Set：500条（250 WangchanThaiInstruct test + 250 Common Voice 17.0）
- Pause评估：210条长句（有annotated pause mask）

**主要结果**（Table 8）：
| 系统 | 参数 | Keyword Acc. | Thai CER | Eng. CER | Pause Prec. | PPER | Intra-word | Speaker sim. |
|------|------|--------------|----------|----------|-------------|------|------------|--------------|
| Wayu-Paxa-TTS-Edge | 82M | 68.2% | 3.7% | 1.1% | 91.4% | 6.7% | 1.4% | 0.882 |
| OmniVoice teacher | 600M | 72.8% | 4.6% | 0.9% | 89.9% | 17.6% | 5.2% | 0.899 |
| Gemini 3.1 Flash TTS | ≥405B* | 79.8% | 3.3% | 0.8% | 96.4% | 13.8% | 1.9% | 0.816 |

- 学生模型pause precision超越教师（91.4% vs 89.9%），PPER和intra-word pause率最低
- Keyword Accuracy达Gemini 3.1的85.5%（68.2%/79.8%）

**关键消融**（Table 5）：
- Pause filtering：Keyword Acc. +2.1pt（67.5%→69.6%），CER -0.5pt
- Bilingual frontend：保留phrase boundary行为
- Resample rejected：Keyword Acc. +2.0pt，CER -0.5pt，Pause Prec. +7.4pt（85.4%→92.8%）

**预训练初始化效应**（Table 6）：
- Code-switch Acc. +42.7pt（22.8%→65.5%）
- Pause Prec. +10.8pt（82.0%→92.8%）

**Best-of-K教师采样**（Table 10）：
- K=118时exact accuracy达87.9%，揭示15.1pt可恢复空间
- 未见过关键词覆盖率66%，100-10k次出现词覆盖率98%

## 相关工作脉络
- **Joshi & Garera, 2023**：先用合成目标语言语音，再适配数小时真实目标说话人音频；本文无需真实目标说话人语料，仅需15秒参考。
- **Geng et al., 2025**：构建大规模泰语speech/text collection含explicit tone和pause annotation；本文通过合成+过滤而非人工标注实现类似目标。
- **Zhang et al., 2025; Hu et al., 2026; Boson AI, 2026; Zhu et al., 2026 (OmniVoice)**：现代多语种语音克隆模型，依赖大backbone和大规模multilingual speech collection；本文将其作为"可编程数据源"而非最终部署对象。
- **Kim et al., 2021 (VITS); Li et al., 2023b (StyleTTS2)**：compact固定音色TTS架构；本文采用Kokoro（82M参数）backbone作为学生模型基础。
- **Hexgrad, 2025 (Kokoro)**：开源82M参数TTS模型；本文首次系统研究其泰语适配与蒸馏训练。

## 局限性与未来方向
- **教师覆盖限制**：最难项集中于teacher training corpus中代表性不足的表达式，即使K=118仍有21%未见过关键词未解决。
- **评估指标依赖启发式**：停顿评估依赖Thai rule-based tokenization和phoneme conversion，segmentation与aligner误差bound scorer准确性。
- **方言自然度未验证**：Isan适应实验仅报告CER，未评估方言自然度或dialect fidelity。
- **与开放权重教师仍有差距**：Keyword Accuracy落后OmniVoice teacher 4.6pt，需改进sampling和selection策略。
- **未来方向**：结合neural和rule-based方法处理泰语context-dependent特性；构建comprehensive Isan evaluation set。

## 研究启发与可借鉴点
- **拒采恢复策略**：对过滤失败样本进行多次重渲染而非直接丢弃，可在保持质量同时恢复困难样本覆盖，适用于任何合成数据训练场景。
- **多维度评估设计**：Challenge-Set Keyword Accuracy分离出句子级CER掩盖的关键表达错误，Prosody Pause Accuracy揭示transcript正确但phrasing错误的情况，可作为泰语/低资源语言TTS评估模板。
- **前端改进无需重训**：Section 4.3证明inference-time LLM verbalization和G2P改进可在固定acoustic model权重下提升1.1pt Keyword Accuracy，为长期维护提供实用路径。
- **短参考方言迁移**：15秒参考即可指定固定音色并保留dialect forms，为其他低资源方言TTS提供可行方案。
- **教师采样分析**：best-of-K oracle实验量化teacher distribution recoverable headroom，指导后续采样策略设计。

## 关键术语表
- **OmniVoice**：Zhu et al. (2026)提出的多语种零样本语音克隆教师模型，本文用作合成数据源。
- **Kokoro**：Hexgrad (2025)开源的82M参数固定音色TTS backbone，基于StyleTTS2架构。
- **Challenge-Set Keyword Accuracy**：针对1,531条困难表达测试集的关键词精确匹配准确率，用于揭示CER无法捕获的局部发音错误。
- **Prosody Pause Accuracy**：评估停顿位置是否符合linguistically acceptable position的指标，含pause precision、PPER、intra-word pause rate三个子指标。
- **TLTK**：Thai Language Toolkit，用于泰语grapheme-to-phoneme转换的开源NLP工具包。
- **Rejection Sampling**：对质量过滤失败的样本进行最多4次重渲染，保留最优结果以维持困难文本覆盖。
- **Tinglish**：Thai-accented English发音的文本表示形式，用于code-switched表达的口语化。
- **Best-of-K Sampling**：对同一文本多次采样教师输出，选择最优 realization以量化teacher distribution可恢复空间。

## 可复现要素
- **数据集**：WangchanThaiInstruct（公开）、LibriTTS（公开）、Common Voice 17.0（公开）、Thai Dialect Isan Speech Corpus（部分公开）、自建Challenge Set（1,531项，随论文发布）
- **代码/权重**：Wayu-Paxa-TTS-Edge模型（82M参数）已开源；Evaluation framework开源
- **关键超参**：8 epochs训练，AdamW优化，lr=1e-4（主模型）/1e-5（PL-BERT），effective batch size=8，英语数据 capped at 30% of Thai corpus duration
- **模型依赖**：OmniVoice（需API访问）、Kokoro checkpoint（HuggingFace）、TLTK、DeepSeek-V4-Flash（LLM verbalization）
- **硬件要求**：学生训练需GPU（论文未明确指定），推理支持on-device部署
