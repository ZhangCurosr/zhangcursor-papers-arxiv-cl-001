---
title: "Scalable-Direction-Following-TTS-via-Voice-Impression-Guided"
source: https://arxiv.org/pdf/2609.02623v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 22:39:58"
field: "方向条件语音合成"
keywords: ["direction-following TTS", "pseudo triplet", "voice impression control", "rectified flow matching", "style refinement", "LLM-as-judge"]
innovations: ["提出方向跟随TTS任务并构建大规模伪三元组解决数据稀缺", "基于rectified flow matching的方向条件风格细化器实现说话人稳健的风格变换", "揭示伪数据与真实录音在说话人稳健性与方向对齐间的互补机制"]
benchmarks: ["UT-MOSv2", "ECAPA-TDNN说话人相似度", "LLM方向对齐评估", "HiFi-CAPTAIN test set"]
---

# 论文速读：Scalable-Direction-Following-TTS-via-Voice-Impression-Guided Pseudo Triplet Construction

## 一句话总结
本文提出了一种**方向跟随TTS**框架，通过**印象可控TTS模型 + LLM**构建大规模伪三元组数据，解决方向驱动语音风格修改任务中配对数据稀缺的问题；实验表明伪数据可稳定保持说话人身份，真实录音数据增强方向对齐，二者结合实现最佳平衡。

## 研究问题与动机
1. **任务定义新颖性**：方向跟随TTS要求系统根据自然语言指令，在保留说话人身份和语言学内容的前提下，对参考语音进行风格修改——这是一种**相对风格变换**而非绝对风格条件生成。
2. **数据稀缺瓶颈**：现有零样本TTS大规模语料库（如WenetSpeech4TTS、HiFiTTS-2）仅含每脚本单次朗读，无法捕捉相对风格变化；即使有多次录音的语料库（如ESD、SpeechCraft）也规模小且缺少方向文本标注。
3. **生成模型依赖大量数据**：基于diffusion/flow的生成模型对训练数据需求量大，数据稀缺进一步限制了说话人保持的稳健性。
4. **现有方法定位不足**：传统text-prompted TTS和speech editing系统关注**绝对风格条件**或prompt控制，而非建模pre-mod→post-mod的**相对变换**。

## 核心贡献（创新点）
1. **提出方向跟随TTS任务**：将导演指导演员重读场景形式化为给定脚本+参考语音+自然语言方向→生成修改后语音的任务，与绝对风格条件TTS形成本质区分。
2. **可扩展伪三元组构建流水线**：首次结合印象可控TTS（作为代理表演者）和LLM（从估计印象差异生成自然语言方向）自动化生成大规模(参考语音, 方向文本, 修改后语音)三元组，解决数据稀缺问题。
3. **基于rectified flow matching的方向条件风格细化器**：在speech embedding空间建模方向条件的随机向量场，避免确定性回归导致的保守更新，同时通过辅助损失鼓励方向一致性和幅度对齐。
4. **伪数据与真实数据的互补性验证**：系统对比Pseudo-all、Recorded、Full三种配置，揭示伪数据提升说话人稳健性、真实数据增强方向表达力的互补机制。

## 方法详解

**整体架构**（Fig. 1）：
- **固定骨干TTS**：基于FastSpeech2 + 冻结HuBERT语音编码器 + HiFi-GAN vocoder的印象可控零样本TTS模型。
- **方向条件风格细化器**：在speech embedding空间操作，预测$e_{post} - e_{pre}$的加法修改量，推理时加到pre-mod embedding上。

**伪三元组构建流程**（Fig. 2）：

1. **印象可控语音生成**：
   - 使用13维印象向量控制风格，维度包括：high–low pitched, masculine–feminine, clear–hoarse, calm–restless, powerful–weak, youthful–elderly, thick–thin, tense–relaxed, dark–bright, cold–warm, slow–fast, fluent–hesitant, emotional–neutral。
   - 随机选择3个性别相关系数<0.7的维度，在[-2, +2]范围内采样控制值，生成paired utterances。

2. **过滤与印象估计**：
   - 过滤条件：ECAPA-TDNN余弦相似度0.80–0.95（排除过相似和过漂移），语速比0.85–1.15。
   - 使用印象估计器预测pre-mod/post-mod的13维向量，计算$\Delta \mathbf{I} = \mathbf{I}_{post} - \mathbf{I}_{pre}$。

3. **LLM方向生成**：
   - 使用Qwen3-Next-80B-A3B-Instruct，prompt扮演声音导演，基于$\Delta \mathbf{I}$生成自然语言表演方向（非限制为显式轴级描述，可包含组合式/上下文相关指导）。
   - 每个语音对生成最多5条方向文本，过滤畸形文本后形成伪三元组。

**风格细化器训练**：
- 目标修改量：$\Delta = e_{post} - e_{pre}$（局部加法近似，不假设全局线性）。
- 采用rectified flow matching：采样$ x_0 \sim \mathcal{N}(0,1) $，构造中间状态$ x_t = (1-t)x_0 + t\Delta $。
- Refiner预测常速度场$ v_\theta(x_t, t, e_{pre}, e_{dir}) $，训练目标匹配$\Delta - x_0$。
- 方向编码使用ModernBERT-Ja-310M。
- 优化器Adam，lr=0.01，batch size=32，最多1M步。

## 实验与结果

**数据集**：
- **伪数据**：1,600说话人日语语音 → 16万初始语音对 → 筛选后74,619对 → 生成350,617个三元组（127.6小时）。训练/验证划分：346,488/4,129（30个 held-out 说话人）。
- **真实录音**：2位专业配音演员（一男一女），8.9小时，6,899语音对；方向文本经LLM准备并经Easy/Medium/Hard模板扩充。

**评估设置**：
- 4个说话人（2 seen, 2 unseen），每个生成15,000条方向条件语音。
- 使用固定骨干TTS生成pre-mod，50条固定方向（GPT-5.2生成）。
- 客观评估：ECAPA-TDNN说话人相似度、UT-MOSv2自然度、LLM方向对齐度（1-5分）。
- 主观评估：258/208名参与者，SMOS（说话人相似度）和AlignMOS（方向对齐度）。

**关键结果**：

| 条件 | Seen UTMOS | Seen LLM | Unseen UTMOS | Unseen LLM |
|------|-----------|---------|-------------|-----------|
| Pre-mod | 2.95 | - | 2.97 | - |
| **Full** | **2.96** | **3.79** | 2.97 | 3.24 |
| Recorded | 2.94 | 3.78 | 2.95 | **3.37** |
| Pseudo-all | 2.95 | 3.67 | **2.99** | 3.12 |

- **说话人相似度**（Fig. 3）：Pseudo条件对unseen说话人更稳健， Recorded方差大且常低于5th percentile（0.57）；Full在稳健性和变异性间取得平衡。
- **自然度**：所有条件下UT-MOSv2与source recorded相当（~2.95-3.00），方向细化未显著降低自然度。
- **方向对齐**（Table 1）：Full在seen说话人上LLM得分最高（3.79），Recorded在unseen上略优（3.37）；Pseudo-all更保守。
- **主观评估**（Table 2）：Pseudo-all SMOS最高（3.54 seen, 3.24 unseen），Recorded AlignMOS最高（3.50 seen, 3.48 unseen）；Full在SMOS（3.35/3.22）和AlignMOS（3.22/3.32）间取得最佳平衡。

**核心结论**：
- Pseudo-all alone可实现稳定身份保持的合理方向跟随。
- Recorded数据增强方向对齐但牺牲说话人稳健性。
- Full组合实现表达力与稳定性的最佳权衡。
- 伪数据F0变化（mean ln F0 abs diff: 0.05 vs 0.14）小于真实录音，解释其更保守的风格调制。

## 相关工作脉络

1. **印象可控TTS**（[11,12]）：本文印象估计器和可控TTS的基础，但 prior work 关注**绝对风格生成**，本文转向**相对变换建模**。
2. **指令引导TTS/语音编辑**（PromptTTS 2 [13], InstructSpeech [14], ControlSpeech [15], SpeechCraft [16], OV-InstructTTS [17], ISSE [18]）：这些方法聚焦**单 utterance 的绝对风格条件**或 prompt-based 编辑，本文强调 pre-mod→post-mod 的**配对相对修改**。
3. **语音转换与风格编辑**（PromptVC [19], FlexiVoice [20]）：类似 latent space 风格控制思路，但本文引入**rectified flow matching**建模方向条件的随机向量场，避免确定性回归的保守性问题。
4. **方向跟随TTS前作**（[4] Kanagawa et al.）：多交互TTS工作，本文在此基础上扩展到**零样本场景**并通过伪三元组解决数据稀缺。
5. **Flow matching for speech**（F5-TTS [10]）：本文采用 rectified flow matching 技术，但应用于**embedding 空间的风格细化**而非端到端语音生成。

## 局限性与未来方向

1. **伪数据的表达力上限**：伪数据的F0变化幅度（0.05）显著小于专业录音（0.14），导致风格调制更保守；当前伪数据难以完全复现专业配音演员的自然表达范围。
2. **方向文本质量依赖LLM**：伪方向由LLM生成，可能存在偏离真实导演意图的情况；真实录音虽质量好但规模有限。
3. **语言与说话人范围**：当前仅验证日语1,600说话人，跨语言泛化能力未知。
4. **评估指标的局限性**：LLM作为judge的客观评估需要进一步validate；impression difference representation是否充分捕捉方向语义仍有疑问。
5. **Future work自述**：如何更好地将人类表现力与伪数据结合。

## 研究启发与可借鉴点

1. **伪数据构建的通用范式**：印象可控TTS + LLM方向生成的流水线可迁移到其他**相对变换任务**（如情感转换、口音迁移），为数据稀缺的语音操控任务提供可扩展解决方案。
2. **Rectified flow matching用于embedding空间细化**：将flow matching应用于**局部风格修改**而非完整语音生成，是一种高效参数利用策略；可借鉴到语音编辑、voice conversion等任务。
3. **伪数据与真实数据的互补性分析框架**：系统对比Pseudo/Recorded/Full的配置，揭示"稳健性 vs 表达力"的trade-off，为后续多源数据融合提供实验设计参考。
4. **LLM-as-judge的方向对齐评估**：使用LLM比较印象差异与方向文本语义一致性，为**相对风格跟随任务**提供自动化评估指标，可推广到其他方向conditioned任务。
5. **13维印象向量的可扩展性**：基于反义词对的主观量表设计（11维 validated + 2维扩展）为语音风格建模提供细粒度连续控制接口，可与离散风格标签方法互补。

## 关键术语表

**Direction-following TTS**：给定脚本、参考语音和自然语言方向指令，生成保留说话人身份和语言学内容但体现风格修改的新语音的任务。

**Pseudo triplet**：由(参考语音, 方向文本, 修改后语音)组成的合成训练样本，通过印象可控TTS生成语音对并用LLM生成方向文本。

**Impression vector**：13维连续向量，表示基于反义词对的主观语音印象强度（如high–low pitched, calm–restless等）。

**Pre-mod / Post-mod utterance**：分别指方向修改前（参考）和修改后（目标）的语音对。

**Rectified flow matching**：一种生成建模技术，通过学习从噪声到目标的直线轨迹来建模数据分布；本文用于在embedding空间建模方向条件的随机变换。

**Style refiner**：方向条件风格细化器，在固定TTS骨干的speech embedding空间预测修改量并叠加到参考语音embedding上。

**LLM-as-judge**：利用大语言模型作为评估器，比较估计的印象差异与方向文本语义的一致性来打分。

**ECAPA-TDNN**：用于说话人嵌入提取的深度残差网络，本文用于计算说话人相似度（余弦相似度）。

## 可复现要素

- **数据集**：
  - 伪数据：基于NTT内部日语 speech data（1,600说话人），未公开。
  - 真实录音：2位专业配音演员，8.9小时，未公开。
  - HiFi-CAPTAIN test set用于unseen说话人选择（公开）。
- **代码/权重**：论文未明确声明开源；demo page提供音频示例和prompt示例。
- **关键超参**：
  - 印象维度控制：3维，范围[-2, +2]，相关性<0.7。
  - 过滤阈值：ECAPA-TDNN相似度0.80–0.95，语速比0.85–1.15。
  - Refiner优化：Adam, lr=0.01, batch size=32, 最多1M步。
  - 方向编码：ModernBERT-Ja-310M。
  - 每语音对最多5条方向文本（Qwen3-Next-80B-A3B-Instruct）。
