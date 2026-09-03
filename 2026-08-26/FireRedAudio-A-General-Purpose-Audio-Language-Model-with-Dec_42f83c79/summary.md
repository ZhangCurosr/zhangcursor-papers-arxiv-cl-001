---
title: "FireRedAudio-A-General-Purpose-Audio-Language-Model-with-Dec"
source: https://arxiv.org/pdf/2608.24168v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:36:09"
field: "多模态统一理解与生成"
keywords: ["统一音频语言模型", "解耦连续表示", "流匹配 DiT", "零样本 TTS", "语音编辑", "长时音频理解"]
innovations: ["输入侧解耦的连续理解/生成双路径设计", "确定性强编码器 RedAE 结合高层语义蒸馏", "flow-matching DiT 自回归生成连续声学潜变量"]
benchmarks: ["MMAU", "MMSU", "Seed-TTS-Eval", "InstructTTSEval", "Ming-Freeform-Audio-Edit", "FLEURS-102", "LibriSpeech"]
---

# 论文速读：FireRedAudio: A General-Purpose Audio Language Model with Decoupled Continuous Representations for Understanding and Generation

## 一句话总结
FireRedAudio 提出了一种**解耦连续输入表示**的统一音频语言模型：在同一个 9B 参数 LLM 中，理解任务走 Audio Encoder 路径，生成任务走 RedAE（确定性音频自编码器）路径，分别以 12.5 Hz 和 6.25 Hz 的帧率喂给共享 LLM；生成侧由 flow-matching DiT 直接产出连续声学潜变量，避免了量化码本的信息损失。模型在 ASR、音频理解、零样本/指令 TTS 及语义与声学语音编辑上均达到竞争性或领先结果，并支持长达 1 小时的音频理解与秒级时间戳精度的结构化组织。

## 研究问题与动机
- **单一表示无法兼顾理解与生成**：理解需要紧凑、任务导向的抽象表征以支持长上下文建模；生成需要波形可重构的高分辨率表征以保留音色、韵律、音高与风格等细粒度声学细节。强迫同一表征同时满足两者会导致性能折损（引用 [1]）。
- **已有统一模型的表征路径仍有局限**：LongCat-Next 输入输出均用离散 token；UALM/Audex 输入连续、输出离散；Kimi-Audio 将 Whisper 特征与离散语义 token 逐元素相加；Qwen3-Omni 采用 Thinker–Talker 分解；UniAudio 2.0 与 DualSpeechLM 虽分别分配语义/声学 token 角色，但仍依赖量化码本，存在信息损失与多码本开销。
- **纯连续路径也未能自动化解任务冲突**：Ming-UniAudio 共享 VAE 潜空间路径并在早期联合训练时冻结语义模块才能避免性能下降；Audio-Omni 冻结多模态 LLM 并用其提取语义再条件化 DiT；UAT 在同一双流骨干上同时做连续音频扩散与离散文本掩码扩散。表明**关键问题不在于是否使用连续表示，而在于能否让共享模型为不同任务分配合适的连续输入表示**。
- **长时音频结构化理解缺少精度保证**：现有工作对超过数分钟的录音缺乏稳定时间–内容对齐能力，本文目标将理解长度延伸至 1 小时并达到秒级时间戳精度。

## 核心贡献（创新点）
1. **首次在同一可训练自回归 LLM 中提供相互解耦的连续输入路径**：理解走 Audio Encoder + Adapter（12.5 Hz），生成走 RedAE Encoder + Patch Encoder（25 Hz → 6.25 Hz），两者不融合、路由到对应音频段的位置输入共享 LLM。与 UniAudio 2.0/DualSpeechLM 的本质区别在于：**理解与生成使用的连续表征来自不同预训练编码器，且在输入侧即解耦，而非共享同一潜空间后再分化**。
2. **确定性强化的连续自编码器 RedAE**：无变分后验与 KL 正则，仅靠重构损失 + GAN 目标训练；并引入来自冻结教师 Audio Encoder 的 MSE 语义蒸馏损失，使潜变量既保留波形可重构性又获得高层结构，优于 Ming-UniAudio 在重构潜空间之后外挂独立语义模块的设计。
3. **Flow-matching DiT 直接生成连续 RedAE 潜变量**：每 audio step 由 LLM 隐藏状态 + 前两步声学历史（12 帧）条件化，生成 4 帧（160 ms）潜变量并反馈 Patch Encoder 作为下一步 LLM 上下文，形成闭环自回归连续生成，避免了量化 token 的信息瓶颈。
4. **统一且可扩展的多任务能力**：在 9B 参数量级下同时覆盖 ASR、音频理解（最长 1 小时）、零样本 TTS、Instruct TTS 与语义/声学语音编辑；结构化长时音频组织实现秒级时间戳精度（content@1s 通过率 >96%）。
5. **五阶段渐进式训练与两层次采样平衡**：Adapter Alignment → Audio Encoder Adaptation → Unified Mid-training → Multitask Post-training → Long-context Extension；以幂律平滑的两层次采样（α_task=0.5, α_data=0.7）缓解大规模任务淹没小规模任务的优化偏置。

## 方法详解
### 架构组成
- **Audio Encoder**：初始化自 Whisper-large-v3 Encoder，含卷积前端 + Transformer；16 kHz 单声道波形 → log-Mel → 30s 非重叠窗口局部编码至 50 Hz → 拼接后经 Audio Adapter 下采样至 12.5 Hz 并投影到 LLM 输入嵌入空间。
- **RedAE（Red Audio Autoencoder）**：确定性因果自编码器，24 kHz 波形分块为 480 样本（50 Hz 帧），经两个级联 Qwen3 Transformer（前者上下文编码、后者时序聚合，每对相邻帧聚合），输出 64 维潜变量（25 Hz）；解码端做线性投影 + 通道到时间重排 + 因果 Qwen3 Transformer + iSTFT 头重构波形。
- **Patch Encoder**：局部 Transformer 将每 4 个连续 25 Hz 潜帧聚合为 1 个 RedAE-Patch（与 LLM 嵌入同维），将 LLM 侧帧率降至 6.25 Hz；波形成像与 DiT 生成仍在原生 25 Hz 潜空间进行。
- **共享 LLM**：Qwen3.5-9B，所有任务统一为 ChatML 对话形式，通过系统提示区分任务。
- **Flow-matching DiT**：以当前及前两个 audio step 的 LLM 隐状态（LLM conditioning）+ 前两步 12 帧声学历史（acoustic conditioning）为条件，对噪声与目标 4 帧潜变量之间的概率流积分，输出预测速度场；仅最终 4 个位置的输出计入 loss。

### 训练目标
$$\mathcal{L} = \lambda_{\mathrm{text}} \mathcal{L}_{\mathrm{text}} + \lambda_{\mathrm{flow}} \mathcal{L}_{\mathrm{flow}}, \quad \lambda_{\mathrm{text}} = \lambda_{\mathrm{flow}} = 1$$
- **$\mathcal{L}_{\mathrm{text}}$**：下一 token 交叉熵，覆盖普通文本 token、audio-start/audio-end token 及 audio-step token；其中 audio-step 因数量庞大赋予 $w_a = 0.01$ 权重（分子分母同时加权），避免其主导优化。
- **$\mathcal{L}_{\mathrm{flow}}$**：flow-matching MSE 损失，平均 over 4 帧 × 64 维；训练时随机 drop LLM conditioning 以实现 classifier-free guidance，声学 conditioning 始终保留。
- **RedAE 预训练损失**（仅在独立预训练阶段使用）：
$$\mathcal{L}_G = \lambda_{\mathrm{rec}} \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{adv}} \mathcal{L}_{\mathrm{adv}} + \lambda_{\mathrm{fm}} \mathcal{L}_{\mathrm{fm}} + \lambda_{\mathrm{distill}} \mathcal{L}_{\mathrm{distill}}$$
其中 $\mathcal{L}_{\mathrm{distill}}$ 为 RedAE 潜变量与冻结教师 Audio Encoder 特征间的 MSE 对齐损失。

### 五阶段渐进训练
| 阶段 | 数据焦点 | 体量 | 最大序列长度 | 关键设置 |
|---|---|---|---|---|
| Adapter Alignment | 中英 ASR + 多语 ASR | 180B tokens | 8k | 冻结 Audio Encoder 与 LLM，仅训 Adapter，LR 线性 warmup 至 $2 \times 10^{-4}$ |
| Audio Encoder Adaptation | ASR + 语音/音频/音乐理解 | 390B | 8k | 解冻 Audio Encoder 与 Adapter 联合优化，LLM 冻结，LR 峰值 $2 \times 10^{-4}$ |
| Unified Mid-training | 文本 + ASR + 理解 + 零样本 TTS + 音文交错 | 990B | 8k | 全部模块联合训练；LLM LR $3 \times 10^{-5}$，音频模块 LR $2 \times 10^{-4}$ |
| Multitask Post-training | 保留上文 + Instruct TTS + 语音编辑 + CoT 监督 | 511B | 8k | LLM LR 保持 $3 \times 10^{-5}$，音频模块降至 $1 \times 10^{-4}$ |
| Long-context Extension | 1 小时长音频理解 + 保留多任务数据 | 591B | 200k | LLM LR $1 \times 10^{-5}$，音频模块 $3 \times 10^{-5}$ |

### 推理流程（以零样本 TTS 为例）
参考语音经 RedAE Encoder 编码后，其最后两个 audio step 的潜帧填充到声学历史 buffer；LLM 自回归输出 audio-start → audio-step 序列；每步 LLM 隐状态 + 声学 buffer（12 帧）驱动 DiT 生成 4 帧新潜变量，更新 buffer 并送入 Patch Encoder 反馈 LLM；遇到 audio-end 终止，完整潜序列由 RedAE Decoder 重构为 24 kHz 波形。

## 实验与结果
### 音频理解
- **MMAU**：test-mini 82.0%（最优），test 80.9%（最优）；超越 Qwen3.5-Omni-Plus（81.4*/79.9*）与 Gemini 3.1 Pro（80.7*/78.8*）。
- **MMSU**：83.3%（最优），超越 Qwen3.5-Omni-Plus（80.7*）与 Gemini 3.1 Pro（82.7*）。
- 结论：专用 Audio Encoder 路径未因生成任务的分流而牺牲理解能力。

### 自动语音识别（ASR）
- **LibriSpeech**：test-clean 0.67%（最优）、test-other 2.91%。
- **FLEURS**：English 2.53%（最优）、Chinese 3.14%；FLEURS-102 宏平均 14.94%（最优）。
- **AISHELL-1** 0.71%、**WenetSpeech Test_Net** 5.18%/Test_Meeting 5.33%、**KeSpeech** 4.82%、**Opencpop** 1.63%。
- 结论：在多语言、方言、网络/会议、歌唱语音上均保持竞争力或领先。

### 零样本 TTS（Seed-TTS-Eval）
- **Seed-ZH**：CER 0.83%（最优）、SIM 0.74。
- **Seed-EN**：WER 1.56%（第二）、SIM 0.68。
- **平均内容误差** 1.20%（所有对比系统最低）；平均 SIM 0.71（统一模型中最高）。
- 关键优势：无需专用 speaker embedding，直接以参考语音的 RedAE 表征作条件，仍获强音色保持。

### 指令 TTS（InstructTTSEval）
- **中文**：APS 86.0%、DSD 84.1%、RP 70.1%（全优）。
- **英文**：APS 81.1%、DSD 83.6%、RP 70.3%（全优）；RP 较次优提升 6.1pp，显示从角色/场景推断未明示语音风格的能力突出。
- CoT 监督可能促成 LLM 在条件化 DiT 前先推理出合适说话风格。

### 语音编辑（Ming-Freeform-Audio-Edit）
- **语义编辑**（删除/插入/替换，basic & open）：整体 WER 与 no-edit WER 全面下降，edit-region ACC 持平或提升；open 删除中文 WER 从 22.92% 降至 10.49%。
- **声学编辑**：速度（ZH WER 5.88→2.00，EN 17.53→4.43）、音高（ZH 7.45→2.00，EN 13.37→3.04）、音量（ZH RAE 14.90→2.39，EN 11.70→3.74）各指标全优；SIM 同步提升，证明语言学内容与说话人特征未被破坏。
- 结论：LLM + CoT 解释编辑意图、DiT–RedAE 合成编辑后波形，实现语义改动与未改动区域保真的双重目标。

### 长时结构化音频组织（5–50 分钟，1 小时输入）
- **strict@0**：总通过率 73.6%，较 Qwen3.5-Omni-Plus（56.6%）提升 17.0pp。
- **content@0.5s**：总 96.1%；**content@1.0s**：总 96.7%，50 分钟组仍达 94.4%。
- 归因：长时训练数据由精确标注 chunk 合并、chunk-relative 时间戳转为录音级时间戳、跨 chunk 说话人标签全局一致化；时间戳转换直接监督时间–内容对齐。

## 相关工作脉络
1. **LongCat-Next**：输入输出均离散 token，理解与生成用同一离散表示；FireRedAudio 在输入侧即解耦为两条连续路径。
2. **UALM / Audex**：输入连续、输出离散；FireRedAudio 输出亦为连续潜变量，避免码本信息损失与多码本开销。
3. **Kimi-Audio**：将连续 Whisper 特征与离散语义 token 逐元素相加融合；FireRedAudio 不做输入融合，按段路由到对应路径。
4. **Qwen3-Omni（Thinker–Talker）**：Thinker 处理连续理解、Talker 预测离散语音 token；FireRedAudio 共用同一 LLM，生成侧用 DiT 输出连续潜变量。
5. **UniAudio 2.0 / DualSpeechLM**：分配不同 token 角色但仍是量化目标；消融显示仅用重建 token 理解更弱；FireRedAudio 从输入侧即用不同编码器产出不同性质的连续表征。
6. **Ming-UniAudio**：共享 VAE 潜空间路径，早期联合训练需冻结语义模块以避免性能下降；FireRedAudio 不共享，RedAE 预训练阶段即注入高层语义蒸馏，无需推理时外挂语义模块。
7. **LatentLM / DiTAR / VibeVoice**：连续潜变量自回归生成；FireRedAudio 独特之处在于将此类生成与长时理解统一在同一 9B 模型，并以解耦输入表示化解任务冲突。
8. **Audio-Omni / UAT**：前者冻结 LLM 仅用其提取语义再条件化 DiT，后者双流骨干；FireRedAudio 的 LLM 全程参与理解与生成联合优化。

## 局限性与未来方向
- **参数规模相对有限**：9B 主干在面向超大模型的多模态竞赛中并非最大；在极低资源语言、复杂声学环境下的 ASR 与理解仍有提升空间（如 FLEURS 部分语言 WER 仍高于 Gemini 3.1 Pro）。
- **Speaker similarity 未达最强专用 TTS**：尽管在统一模型中表现最佳，但绝对 SIM 低于 Seed-TTS、CosyVoice 3 等专属系统，推测源于未显式建模 speaker embedding。
- **长时理解的严格时间边界仍有误差**：strict@0 通过率 ~73%，说明精确切割点仍需改进；50 分钟组 content@0.5s 降至 93.9%。
- **RedAE 训练数据配比**：50% 干净语音 + 25% 带噪语音 + 10% 音效 + 15% 音乐，对环境音与音乐的泛化能力未充分评估。
- **推理延迟**：每 160 ms 音频需一次 LLM step + 一次 DiT 推理，实时流式场景需进一步优化。

## 研究启发与可借鉴点
1. **输入侧解耦表征的可迁移性**：不仅适用于音频，对视频/多模态统一模型同样有参考价值——理解分支可沿用预训练视觉/音频编码器，生成分支使用自编码器路径，避免共享表征的任务冲突。
2. **确定性强编码器 + 高层语义蒸馏**：RedAE 的训练策略（无 KL + GAN + 教师蒸馏）可作为通用连续自编码器范式，适用于音乐、环境音、语音等多域统一表示学习。
3. **Flow-matching DiT 与 LLM 闭环自回归**：每一 audio step 将新生成片段经 Patch Encoder 反馈 LLM 作为上下文，形成自洽的生成回路；该设计可直接迁移至视频帧序列生成、时序信号建模等任务。
4. **两层次幂律采样**：先采任务族（α_task=0.5）再采数据集（α_data=0.7），对多任务长尾分布的缓解效果显著，可作为统一模型训练的通用数据调度策略。
5. **CoT 监督在指令生成任务中的价值**：Instruct TTS 与语音编辑中引入显式链式思维，提升模型对风格推断与编辑意图解释的稳定性；可推广至其他可控生成任务（如音乐生成、语音变换）。

## 关键术语表
- **FireRedAudio**：快手/小红书团队提出的 9B 参数通用音频语言模型，支持理解与生成统一接口。
- **Decoupled Continuous Representations**：理解与生成使用由不同编码器产出的独立连续表征，在输入侧即解耦而不融合。
- **RedAE（Red Audio Autoencoder）**：确定性强化的连续音频自编码器，25 Hz、64 维潜变量，支持波形可重构与高层语义对齐。
- **RedAE-Patch**：Patch Encoder 将 4 个连续 RedAE 帧聚合为 1 个 LLM 兼容表示，使 LLM 侧帧率降至 6.25 Hz。
- **Flow-matching DiT**：以 LLM 隐状态与声学历史为条件、通过流匹配学习从噪声到 RedAE 潜变量的条件速度场的 Diffusion Transformer。
- **Audio Adapter**：连接 Audio Encoder 与 LLM 的轻量投影模块，完成时序压缩与跨模态对齐。
- **Classifier-free Guidance**：训练时随机 drop LLM conditioning，推理时通过条件/无条件输出的组合增强生成质量。
- **Multimodal Token**：进入 LLM 的文本 token 或连续音频表示的统称。

## 可复现要素
- **代码**：https://github.com/FireRedTeam/FireRedAudio（论文声明已开源）。
- **权重**：论文未明确说明 9B 主模型权重是否公开，仅给出代码链接；RedAE、Audio Encoder（Whisper-large-v3）、LLM（Qwen3.5-9B）均为开源组件。
- **数据集**：训练数据混合未完全公开细节；评估基准（MMAU、MMSU、Seed-TTS-Eval、InstructTTSEval、Ming-Freeform-Audio-Edit、AISHELL-1/2、WenetSpeech、LibriSpeech、FLEURS、KeSpeech、Opencpop）均为公开或可申请的评测集。
- **关键超参**：α_task=0.5、α_data=0.7；LLM 与音频模块学习率在 5 个阶段分别不同（见 Table 2）；audio step = 4 帧（160 ms）；RedAE 潜维 64；加权系数 λ_text=λ_flow=1、w_a=0.01；训练总量约 2.67B tokens（180+390+990+511+591）。
