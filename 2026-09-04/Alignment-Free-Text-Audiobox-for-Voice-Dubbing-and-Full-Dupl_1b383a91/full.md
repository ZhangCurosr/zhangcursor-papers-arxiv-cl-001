# Alignment-Free Text-Audiobox for Voice Dubbing and Full-Duplex Dialogue Synthesis

Sanyuan Chen<sup>∗</sup>, Min-Jae Hwang<sup>∗</sup>, Sho Inoue<sup>∗</sup>, Anna Sun, Bokai Yu, David Kant, Dongmin Hyun, Dorian Desblancs, Gregory Antonovsky, Oleg Repin, Peng-Jen Chen, Xutai Ma, Zehai Tu, Juan Pino, Wei-Ning Hsu<sup>†</sup>

FAIR at Meta

<sup>∗</sup>Equal contribution, <sup>†</sup>Senior author

We present Alignment-Free Text-Audiobox (Text-AB), a unified framework for high-quality voice dubbing and full-duplex dialogue synthesis. Building on a Difusion Transformer backbone trained with a flow-matching objective, Text-AB departs from prior Audiobox system along three key dimensions. First, it operates in a latent difusion framework using DAC-VAE features that encode 48 kHz waveforms into a 25 Hz low-rate latent sequence, providing over 10× higher compression than previous EnCodec representations while improving resynthesis quality. Second, Text-AB is alignment-free: it consumes raw text via an of-the-shelf text encoder and learns text–speech alignment through cross-attention, removing the need for forced alignment and explicit duration prediction. Third, we scale the model and data substantially, pretraining a 3B-parameter model on 480k hours of monolingual speech, followed by supervised fine-tuning on three downstream tasks—cross-lingual voice dubbing, full-duplex dialogue synthesis, and emotional full-duplex dialogue synthesis. At inference time, Text-AB supports both one-shot generation for up to ∼1 min speech and arbitrarily long-form generation via a multi-difusion scheme. We further incorporate a multi-stage reranking strategy to efectively enhance generation quality based on automated metrics. On a real-world dubbing benchmark, Text-AB delivers a step-change improvement over the latest internal dubbing system, with large gains in prosody similarity, voice similarity, naturalness, and shareability. For full-duplex dialogue synthesis, Text-AB approaches human recordings on short-form conversations and substantially outperforms the latest internal model on long-form human-likeness and expressivity, while natively modeling turn-taking, back-channeling, and emotional dynamics. For emotional dialogue synthesis, Text-AB with emotion conditioning significantly improves emotion alignment and emotional interaction quality over the baseline without emotion conditioning.

Date: September 4, 2026 Correspondence: mjhwang@meta.com

∞Meta

## 1 Introduction

Text-to-speech (TTS) aims to generate high-quality speech from text with high clarity and intelligibility. Driven by recent advances in deep learning, TTS systems have made remarkable progress in naturalness and robustness (Shen et al., 2018; Li et al., 2019; Ren et al., 2019). More recently, zero-shot TTS has attracted increasing attention, where the model is required to synthesize speech for previously unseen speakers given only a short enrollment utterance at inference time. A variety of approaches have been proposed, including language-model-based methods (Chen et al., 2025a), difusion-based methods (Ju et al., 2024), and flow-matching-based methods (Le et al., 2024; Vyas et al., 2023), with some systems achieving near human-parity performance under certain conditions (Chen et al., 2024).

Despite these advances in monolingual single-speaker TTS, more complex settings such as multilingual generation for voice dubbing and full-duplex dialogue synthesis remain highly challenging. Voice dubbing aims to translate speech into a diferent language while preserving the speaker’s voice, emotion, and expressiveness from a source audio prompt. A typical industrial solution is a cascaded pipeline: an ASR model transcribes the source speech, a machine translation model produces the target-language transcript, and a TTS model finally generates the target speech (Yang et al., 2020; Federico et al., 2020; Artioli et al., 2025). In this work we focus on the last component. From a modeling perspective, voice dubbing can be viewed as a cross-lingual zero-shot TTS problem: the input speech serves as an audio prompt, and the model aims to generate target-language speech conditioned on a given target transcript while preserving the source speaker’s characteristics. A major challenge is that large-scale training data are typically monolingual, whereas the use case is cross-lingual, so the model must learn to transfer speaker attributes across languages despite this training–inference mismatch.

On the other hand, full-duplex dialogue synthesis, whose goal is to generate a natural conversation between two speakers, has become an important problem with the rise of full-duplex speech language models (FD-SLMs) (Défossez et al., 2024; Cui et al., 2025). Training FD-SLM purely on real full-duplex recordings (two-channel conversations from real speakers) satisfies stringent audio-quality requirements, but ofers limited control over content, personality, style, and identity. Moreover, increasing the proportion of such data in the training mix of FD-SLM can also degrade factuality and lead to identity confusion. To address this, prior work has relied on synthetic pipelines by generating single-turn monologues using a TTS model, then algorithmically stitching them into a dialogue (Guo et al., 2021; Xue et al., 2023; Lee et al., 2023; Hu et al., 2024; Liu et al., 2024; Xie et al., 2025). Although this approach is straightforward, it fundamentally limits the ability to create highly fluid conversations because it lacks joint modeling of conversational context and turn dynamics. Recent work has attempted to generate full-duplex dialogue in a more integrated manner (Peng et al., 2025; Zhang et al., 2025; Zhu et al., 2025). However, these approaches either lack zero-shot voice prompting capabilities or are trained on limited data with modest model scales, leaving room for improvement in both controllability and quality of the synthesized dialogues.

In this work, we present Alignment-Free Text-Audiobox (Text-AB), a unified framework with two variants, Text-AB-Mono and Text-AB-Stereo, for end-to-end single- and two-channel speech synthesis, respectively. Text-AB builds on Audiobox (Vyas et al., 2023), which also uses a Difusion Transformer (DiT) backbone (Peebles and Xie, 2023) and a flow-matching objective (Lipman et al., 2022), but introduces several key improvements. First, we adopt a latent difusion framework (Rombach et al., 2022) with DAC-VAE audio features (Polyak et al., 2024) that encode 48 kHz waveforms into a low-rate (25 Hz) latent sequence. Compared to the EnCodec features used in Audiobox, these representations ofer much higher compression (1920× vs. 160×–320×), higher audio resolution (48 kHz vs. 16–24 kHz), and better resynthesis quality. Second, Text-AB is an alignment-free, end-to-end model that avoids explicit token-duration prediction, which brings three major benefits: (a) prior work typically relies on regression-based duration predictors that tend to underfit and limit expressivity, while autoregressive or difusion-based duration models are often less stable in practice; (b) training a duration predictor requires a forced aligner to estimate token durations, and forced-alignment errors can significantly hurt performance, especially on noisy or highly conversational speech; and (c) Text-AB uses a single generative component, which can be optimized end-to-end and scaled more easily. Third, Text-AB is substantially larger, uses stronger audio representations, and is trained on more and higher-quality data with improved training strategies.

Our main model has 3B parameters (∼10× larger than Audiobox) and uses a multi-stage training pipeline: (i) large-scale pretraining on 480k hours of monolingual data (∼3× the Audiobox scale) to learn general acoustic and prosodic characteristics, (ii) fine-tuning on 2k hours of high-quality monolingual and cross-lingual data for voice dubbing, (iii) task-specific fine-tuning on 28k hours of two-channel dialogue data for full-duplex dialogue synthesis, or with additional emotion conditioning for emotional dialogue synthesis. At inference time, we extend the model to long-form generation via multi-difusion, which enables arbitrarily long speech generation and removes the efective length constraints imposed by the training data duration distribution. Moreover, we dramatically increase generation performance by incorporating multi-stage reranking strategy, which automatically selects the best output among multiple candidates based on the word error rate and speaker similarity scores.

We evaluate Text-AB on three downstream tasks—voice dubbing, full-duplex dialogue synthesis, and emotional full-duplex dialogue synthesis. For voice dubbing, Text-AB delivers a step-change improvement over the latest internal dubbing model in human evaluations using a [−3, 3] mean opinion score (MOS) scale: we observe large gains not only in overall shareability (+0.39), but also across fine-grained aspects, including prosody similarity (+0.34), voice similarity (+0.32), and voice naturalness (+0.42). For full-duplex dialogue synthesis, Text-AB can generate up to 1 minute of audio in a single one-shot pass and supports long-form generation beyond 10 minutes via a multi-difusion scheme. Text-AB closely matches real recordings on short-form human-written scripts (only −0.09 in overall human-likeness on a 5-point MOS scale) and substantially outperforms the previous dialogue synthesis system on long-form synthetic scripts (+0.86 in overall human-likeness). Text-AB learns turn dynamics, back-channeling, and emotional evolution directly from data, and therefore does not rely on ad-hoc turn stitching, heuristic back-channel insertion, or explicit emotion tagging used in previous systems. For emotional full-duplex dialogue synthesis, Text-AB additionally incorporates explicit turn-level emotion style embeddings together with turn-level transcriptions as conditioning inputs, allowing the generated emotion to difer from the prompt speaking style.

## 2 Text-AB

## 2.1 Flow-Matching

Following prior work on Audiobox (Vyas et al., 2023), we train Text-AB model using the flow-matching loss (Lipman et al., 2022). Flow matching generates samples from a target data distribution by iteratively transforming samples drawn from a simple prior distribution, $\mathrm { e . g . } _ { \mathrm { } } . _ { }$ , a standard Gaussian.

During training, given an audio sample in the latent space $\mathbf { X } _ { 1 }$ , we sample a time step $t \in [ 0 , 1 ]$ from a logit-normal distribution and a noise sample $\mathbf { X } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Then, we use the linear interpolation or the optimal-transport path (Lipman et al., 2022) to construct the noised sample ${ \bf X } _ { t } = t { \bf X } _ { 1 } + \left( 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t \right) { \bf X } _ { 0 }$ where $\sigma _ { \mathrm { m i n } } = 1 0 ^ { - 5 }$ . The model is trained to predict the velocity $\begin{array} { r } { \mathbf { V } _ { t } = \frac { d \mathbf { X } _ { t } } { d t } = \mathbf { X } _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) \mathbf { X } _ { 0 } , } \end{array}$ , which determines how to move $\mathbf { X } _ { t }$ toward the target sample $\mathbf { X } _ { 1 }$ . Let θ denote the model parameters and C the conditions. We denote the model’s predicted velocity by $u ( \mathbf { X } _ { t } , \mathbf { C } , t ; \theta )$ . The training objective is the mean squared error between the ground-truth and predicted velocities:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { t , \mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } , \mathbf { C } } [ | | u ( \mathbf { X } _ { t } , \mathbf { C } , t ; \theta ) - \mathbf { V } _ { t } | | _ { 2 } ^ { 2 } ] . } \end{array}\tag{1}
$$

At inference time, we first sample $\mathbf { X } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and then solve the corresponding ordinary diferential equation (ODE) to obtain $\mathbf { X } _ { 1 }$ using the model’s estimate of $d \mathbf { X } _ { t } / d t$ . We employ a simple first-order Euler ODE solver with a fixed set of N time steps tailored to our model.

## 2.2 Architecture

![](images/5f90b801ca8c4d5b1c3b2f9a25f99b6025f3d4e7489ba3b6b9dced63e136b770.jpg)  
Figure 1 The architectures of Text-AB: (a) Text-AB-Mono and (b) Text-AB-Stereo.

Figure 1 illustrates the architectures of Text-AB which includes the two variants—Text-AB-Mono and Text-AB-Stereo for single and two channel speech generation, respectively.

DiT Backbone. The model is based on a DiT backbone (Peebles and Xie, 2023). In each Transformer block (Vaswani et al., 2017), the flow-time embedding modulates (i) the outputs of normalization layers via scale and bias, and (ii) the outputs of self-attention and feed-forward layers via scale. A multi-layer perceptron (MLP) takes the flow-time embedding as input and predicts six modulation parameters (four scales and two biases). Similar to Polyak et al. (2024), the MLP is shared across all layers and only layer-dependent biases are added to the MLP outputs, which saves parameters without sacrificing performance.

Audio Latent. We adopt a latent difusion framework (Rombach et al., 2022) in which audio is represented as compact latent features. Specifically, we use the DAC-VAE model (Polyak et al., 2024) to produce latent features at a 25 Hz frame rate and 128 dimensions for 48 kHz input audio. Following Audiobox, we take partially masked noised audio features as input and additionally use an audio context to enable audio prompting. The audio context is also represented as the DAC-VAE features of the audio and replaced with zero vectors in the masked region. To perform audio generation without any audio context, we simply input an all-zero sequence as the audio context. We concatenate it with the noised audio latent along the channel dimension frame by frame.

Text and Language Condition. In contrast to Audiobox, which relies on force-aligned text tokens obtained from a force-aligner, our model directly consumes raw text and implicitly learns text–speech alignment using cross-attention. Concretely, we obtain high-level text embeddings using an mT5 text encoder (Xue et al., 2021) and feed these embeddings into cross-attention layers within the Transformer encoder. To support multilingual speech generation, we additionally condition on frame-level language IDs. We first obtain frame-level language labels, then convert them to embedding vectors using a simple embedding layer. These embeddings are concatenated with the audio context and noised audio features along the channel dimension.

Context Zero-Out. Note that we optimize the predicted velocity only within the masked region following Audiobox, which leads to a discrepancy between training and inference. During training, the context region of the noised audio feature is obtained via interpolation between ground-truth speech and random noise, as detailed in Section 2.1. However, during inference, it is derived from model predictions generated in previous ODE steps. Since the model is not optimized to predict velocity for the context region, these predictions may be inaccurate, leading to error accumulation in the noised audio features across subsequent ODE steps. To mitigate this mismatch, we mask the context region with zero values during both phases. Notably, this does not hinder performance, as the necessary contextual information is already provided via the auxiliary audio context.

Similarly, the mismatch would also occur in the language-ID embeddings. More specifically, in the monolingual training data, the context region of the language-ID embedding always matches the target region. During inference, however, the language in the context region may difer from the target region (e.g., Spanish in the context and English in the target). To avoid this issue, we likewise ignore the context region of the language-ID embeddings in both training and inference by masking them with zero-valued tensors, as illustrated in Figure 1.

Stereo Speech Modeling. To support stereo speech generation in Text-AB-Stereo, we concatenate the audio context, noised audio features, and language-ID embeddings from the two channels along the channel dimension, and let the model jointly predict the velocities for both channels. For the text input, we construct a single transcript by temporally ordering and concatenating the two speaker transcripts, inserting special tokens to denote the speaker ID of each segment, and feed this unified transcript to the text encoder.

## 3 Training

## 3.1 Pre-Training

We first conduct large-scale pretraining on Text-AB-Mono using multilingual monologue data and a speechinfilling task, following Audiobox. More specifically, we randomly mask DAC-VAE features and train the DiT model to predict the velocity of the features distribution in the masked regions using the flow-matching loss. We use 480k hours of speech data consisting of 380k hours of English (En) data and 100k hours of Spanish (Es) data for this pretraining stage. Starting from this pretrained Text-AB-Mono model, we first perform dubbing SFT (Section 3.2). Then, we branch into dialogue SFT (Section 3.3) and emotional dialogue SFT (Section 3.4) for full-duplex dialogue synthesis and emotional dialogue synthesis, respectively.

## 3.2 Dubbing SFT

Sentence-level masking. Unlike pretraining, which applies random masking within a single sentence, the inference setting requires sentence-level masking, where source and target sentences are concatenated along the time dimension and only the target sentence is masked. To bridge this gap, we design a sentence-level masking strategy. Specifically, we constructed a 2k hours of multi-sentence SFT dataset containing multiple complete sentences for both En and Es (1k hours for each language). During training, we randomly sample two sets of consecutive sentences from this dataset and use them as the left and right parts of the model input. For the loss computation, we apply masks only to the right part, closely mimicking the inference scenario where only the target sentence is masked.

Cross-lingual SFT data synthesis. Dubbing systems structurally require cross-lingual TTS, where the audio prompts and target text are in diferent languages. To reflect this environment during training, we adopt a Unit-VoiceBox (Barrault et al., 2023b) as a cross-lingual voice conversion task, artificially generating a 50 hours of voice and style-aligned cross-lingual SFT data between En and Es. We mixed the cross-lingual SFT data with multi-sentence SFT data, resulting in a total of 2.1k hours of SFT data.

## 3.3 Dialogue SFT

Given the pretrained Text-AB-Mono model, we perform Dialogue SFT on stereo full-duplex dialogue data to obtain the full-duplex dialogue synthesis model. To initialize Text-AB-Stereo from Text-AB-Mono, we modify only the input and output projection layers by duplicating both projection matrices to accommodate the two-channel input and output. In particular, the input projection weights are divided by 2 so that the distribution of the projected features remains consistent with pretraining. We fine-tune on 28k hours of English two-channel full-duplex dialogue data, consisting of real conversational recordings between two speakers captured on separate channels. For the speech conditioning, we concatenate the speech features of both speakers along the channel dimension. For the text conditioning, in contrast, we concatenate the transcripts of both speakers in temporal order with special speaker-ID tokens delimiting each turn, as described in Section 2.2. This unified transcript allows the model to attend to the full conversational context when generating each frame.

## 3.4 Emotional dialogue SFT

Following the same approach as dialogue SFT, we fine-tune Text-AB-Mono to support explicit turn-level emotion control. We incorporate emotion style embeddings derived from valence-arousal-dominance (VAD) representations (Russell and Mehrabian, 1977), which capture emotion along three continuous dimensions: valence, arousal, and dominance. For each turn in the training data, we extract a turn-level VAD vector from the corresponding speech segment using a pretrained wav2vec2-based emotion recognizer (Baevski et al., 2020)<sup>1</sup>, and project it into the model’s hidden dimension via a learned linear layer. The resulting emotion embedding is concatenated with the text encoder output along the sequence dimension and fed into the cross-attention layers of the DiT backbone. Note that this design decouples emotional expression from the audio prompt’s speaking style, enabling the model to generate speech whose emotion difers from the reference speaker’s tone while preserving the speaker’s voice identity.

## 4 Inference

At inference time, Text-AB supports two modes: (i) one-shot inference, which generates up to about one minute of speech in a single chunk (beyond which quality degrades notably in our experiments), and (ii) long-form inference with a multi-difusion mechanism, which can generate arbitrarily long speech. In this section, we use Text-AB-Stereo as the example to illustrate both inference modes without loss of generality, as Text-AB-Mono can be viewed as a special case where the number of channels is reduced from two to one.

(b) Mono Prompt  
![](images/c9bf5f1d46402defa61f7a4c9b1c1fc44b7963be66957878363f3a141c52260d.jpg)  
(a) Empty Prompt

![](images/7466a17f20372a58a9bd95b605233a241480a5f3de7df385bce8109799bbc508.jpg)

![](images/d459d537f10b4b7e5d0c114b5623454f33da6326049430b95096b1b836bcd064.jpg)  
(c) Stereo Prompt

Figure 2 One-Shot inference with diferent audio prompts.

One-Shot Inference. Figure 2 illustrates one-shot inference under diferent audio-prompt configurations. In the empty prompt setting, no audio prompt is provided. We initialize from random noise and solve the ODE by running the Text-AB-Stereo model forward for a fixed number of ODE steps (denoted by ode\_steps), obtaining a two-channel audio with randomly sampled voices on both channels. In the mono prompt setting, the input audio prompt contains speech only on one channel. The model then generates speech for both channels, reusing the voice in the prompted channel and automatically creating a random voice for the silent channel. In the stereo prompt setting, the input audio prompt contains speech on both channels, corresponding to two speakers, and the model generates a dialogue with two channels that remain consistent with the respective speaker identities.

Long-Form Inference. To generate longer conversations, simply concatenating multiple one-minute dialogues produced by Text-AB-Stereo leads to audible artifacts and discontinuities around the chunk boundaries. To address this, we adopt the multi-difusion inference scheme from Movie Gen Audio (Polyak et al., 2024) and extend it to variable-length chunking, as illustrated in Figure 3. Given a target sequence and an overlapping chunk partition, each chunk is processed independently by Text-AB-Stereo, and at every ODE step the predicted flows in the overlapping frames are merged via a weighted average. This yields a temporally consistent long-form dialogue without artifacts at chunk boundaries. Multi-difusion is only efective with stereo prompts. When using a mono or empty prompt, we observe that the speaker identity can become inconsistent across chunks for any channel whose speaker is not fixed by the prompt. A simple remedy is to first generate an initial chunk in which both speakers speak, and then use this chunk as the stereo audio prompt for subsequent multi-difusion inference.

Multi-Stage Reranking Motivated by the observation that flow-matching objective enables diverse generations across random seeds, we adopt an objective-metric-based reranking strategy (Chen et al., 2024, 2025a) to automatically select the best sample among multiple candidates for both voice dubbing and full-duplex dialogue synthesis. More specifically, we first generate multiple audio samples using diferent random seeds. Then, for each sample we compute speaker similarity (SpkSim) between the generated speech and the audio prompt and word error rate (WER) between the ASR transcript of the output speech and the target text, using the WavLM speaker verification model (Chen et al., 2022) and Whisper-Large-V3 (Radford et al., 2023), respectively. Next, we retain only those samples whose SpkSim is at least p% of the maximum SpkSim observed among all candidates, ensuring that the selected samples preserve the source speaker’s voice. Finally, we pick the sample with the lowest WER among the remaining candidates.

![](images/0cb788147a185c9dbb10f21fa3b93213f7e251ae69eddafc58e672b71e833124.jpg)  
Figure 3 Illustration of multi-difusion. A target audio of 13 frames is split into 6, 7, 5 frames, with 2 and 3 frames overlapping. The overlapping frames (highlighted with yellow background at output) are consolidated with weighted average from contributing chunks.

For emotional dialog synthesis, we additionally evaluate the turn alignment score and emotion accuracy. We assign the highest importance to the turn alignment score to reflect its primary role, and we weight emotion accuracy and WER equally.

## 5 Experiment

## 5.1 Setups

Training Setup. We use a 3B-parameter model as the default configuration and follow the multi-stage training pipeline described in Section 3. In the first pretraining stage, we train on roughly 480k hours of English and Spanish monolingual data with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ for 800k steps on 256 A100 GPUs. In the dubbing SFT stage, we initialize from the pretrained model and fine-tune on about 2k hours of English and Spanish monolingual data plus 50 hours of En→Es and Es→En data, using a linearly decaying learning rate starting from $1 \times 1 0 ^ { - 4 }$ for 200k steps on 256 A100 GPUs. In the dialogue SFT stage, we again initialize from the pretrained model and fine-tune on about 28k hours of English full-duplex dialogue data, with a linearly decaying learning rate starting from $1 \times 1 0 ^ { - 4 }$ for 1M steps on 256 A100 GPUs. For the emotion dialogue SFT, we follow the same training configuration as the dialogue SFT but additionally annotate the data with turn-level valence-arousal-dominance (VAD) labels using an automatic extractor and condition the model on these embeddings.

Evaluation Sets. For voice dubbing, we use an internal real-world competitive benchmarking dataset, containing 100 En→Es and 100 Es→En samples. For full-duplex dialogue synthesis, we use two test sets: (i) a short-form set with average duration of about 30 s drawn from a held-out subset of the training data, used for comparison against ground-truth audio, and (ii) a long-form set with 1–2 min text prompts, focusing more on factuality and general helpfulness, used to compare our model with the latest internal system. For emotional full-duplex dialogue synthesis, we generate 200 multi-turn text dialogues with turn-level emotion labels using an internal text-LLM, spanning four emotion categories (angry, happy, neutral, sad). Then, we condition them on real fully-duplex speech prompts which are held out from training to generate dialogue data for evaluation.

Objective Metrics. We evaluate synthetic speech using the following objective metrics. WER measures the content accuracy of the generated speech as the word error rate between the ASR transcript of the generated speech and the target text, using Whisper-Large-V3 (Radford et al., 2023); lower is better. Note that ASR may still make recognition errors, especially for speech with strong accents, low recording quality, or noisy conditions. SpkSim measures the speaker similarity of the generated speech with respect to the audio prompt, using a WavLM-based speaker verification model (Chen et al., 2022); higher is better. Aes measures the overall audio quality of the generated speech using the Audiobox-Aesthetics-PQ model (Tjandra et al., 2025); higher is better. For full-duplex dialogue synthesis, we report each metric averaged over the two channels.

For emotional full-duplex dialogue synthesis, we additionally report turn alignment, emotion accuracy, and an LLM-based dialog naturalness score. More specifically, we define the turn alignment score to quantify how well the ordering of conversational turns in a generated dialogue matches the intended sequence. After segmenting speech into turns, we compare the synthesized turns with the reference turns using Kendall’s tau correlation coeficient. To evaluate emotion accuracy, we apply Qwen2-Audio<sup>2</sup> (Chu et al., 2024) as a speech emotion recognizer to classify the emotion of each generated turn after turn segmentation, and compute emotion accuracy against the conditioned emotion labels. To evaluate the dialog naturalness score, we prompt Gemini 2.5 Pro<sup>3</sup> (Comanici et al., 2025) with the generated audio to produce a numeric naturalness score on a 1–5 scale, assessing whether the conversation sounds human-like in terms of turn-taking, backchannels, and interjection timing while ignoring conversational content.

Subjective Metrics. For voice dubbing, we conduct side-by-side comparisons and collect human ratings on translation accuracy, audio degeneration, prosody similarity, voice similarity, voice naturalness, and shareability. Raters assign a score in the range [−3, 3] for each metric, where a positive score indicates that Text-AB is preferred over the comparison system. For full-duplex dialogue synthesis, we design a human evaluation protocol based on multi-turn conversations with pairwise comparisons between two variants. Labelers are presented with two conversations side by side, with randomized left/right ordering. The evaluation covers overall human-likeness as well as more fine-grained aspects, including intonation, pacing, expressive intensity, expressive correctness, and use of non-speech vocalizations (NSVs) and fillers. For long-duration samples, we randomly crop each conversation into 1-minute segments to avoid labeler fatigue. We found that using a single randomly sampled segment per conversation is suficient and therefore do not require annotating all segments from the same sample.

For emotional full-duplex dialogue synthesis, we additionally report MOS scores (scale: 1.0-5.0) for dialog naturalness, emotional interaction smoothness, and emotion alignment. For dialog naturalness, annotators judge whether the generated dialogs resemble human conversations. For emotional interaction smoothness, annotators assess the naturalness of emotional transitions across turns, where higher scores indicate coherent transitions and lower scores indicate abrupt or unjustified changes. For emotion alignment, annotators evaluate how well the expressed emotion in turn-level speech matches the provided emotion label.

Inference Setup. For voice dubbing evaluation, we use 32 ODE steps and a reranking size of 32 by default. For full-duplex dialogue synthesis evaluation, we use 32 ODE steps and a reranking size of 4. For the SpkSim filtering, we retained the samples whose SpkSim is at least 75% of the maximum SpkSim score. For long-form generation with multi-difusion, we use a 30 s chunk size and a 20 s chunk overlap as the default setup. For emotional full-duplex dialogue synthesis evaluation, we use 32 ODE steps and a reranking size of 64. To control emotion, the user provides a target emotion label per turn, which is converted to a VAD vector via a predefined mapping, and the model generates emotionally expressive dialogue accordingly.

Table 1 Human evaluation results for voice dubbing compared to the latest internal dubbing model. Positive scores indicate that Text-AB is preferred over the baseline.
<table><tr><td></td><td>Es→En</td><td>En→Es</td></tr><tr><td>Translation Accuracy ↑</td><td>0.02</td><td>0.03</td></tr><tr><td>Audio Degeneration ↑</td><td>0.16</td><td>0.06</td></tr><tr><td>Prosody Similarity ↑</td><td>0.33</td><td>0.34</td></tr><tr><td>Voice Similarity ↑</td><td>0.29</td><td>0.36</td></tr><tr><td>Voice Naturalness ↑</td><td>0.39</td><td>0.45</td></tr><tr><td>Shareability ↑</td><td>0.40</td><td>0.38</td></tr></table>

## 5.2 Voice Dubbing Evaluation

Main Results. As summarized in Table 1, Text-AB significantly outperforms the latest internal dubbing model (+0.40 for Es→En and +0.38 for En→Es) in shareability by a large margin. In particular, Text-AB improves over the internal dubbing model on all fine-grained aspects, including translation accuracy, audio degeneration, prosody similarity, voice similarity, and voice naturalness.

Table 2 Ablation on model size for voice dubbing.
<table><tr><td>Model size</td><td>WER (%) ↓</td><td>SpkSim ↑</td><td>Aes ↑</td></tr><tr><td>300M</td><td>44.45</td><td>0.64</td><td>6.03</td></tr><tr><td>1B</td><td>22.97</td><td>0.72</td><td>6.26</td></tr><tr><td>3B</td><td>13.98</td><td>0.76</td><td>6.29</td></tr></table>

Ablation Study. Table 2 shows an ablation on model size for the voice dubbing task. We observe strong scalability—performance of Text-AB consistently improves when increasing the model size from 300M to 1B to 3B parameters. Specifically, compared to the 300M model, the 3B model achieves relative improvements of 69%, 19% and 4% on content accuracy, speaker similarity and audio quality, respectively.

Table 3 Ablation on dubbing SFT.

<table><tr><td></td><td>WER (%) ↓</td><td>SpkSim ↑</td><td>Aes ↑</td></tr><tr><td>Pre-Training</td><td>5.93</td><td>0.69</td><td>6.72</td></tr><tr><td>mono-lingual SFT</td><td>3.16</td><td>0.73</td><td>6.75</td></tr><tr><td>mono+cross-lingual SFT</td><td>2.72</td><td>0.60</td><td>6.82</td></tr></table>

Table 3 compares the pretraining-only model with models further fine-tuned using diferent SFT data configurations. SFT substantially improves all objective metrics, especially WER, indicating much better content accuracy. More specifically, adding cross-lingual data on top of monolingual SFT yields further gains in content accuracy and audio quality, at the cost of a reduction in speaker similarity.

Figure 4 demonstrates that voice dubbing performance is significantly enhanced by the multi-stage reranking strategy, with gains scaling consistently as the number of reranking candidates increases. For instance, using 32 candidates at inference improves the WER from 4.05% to 2.20% and the SpkSim from 0.66 to 0.74, compared to a baseline without reranking.

## 5.3 Full-Duplex Dialogue Synthesis Evaluation

Main Results. Table 4 reports human evaluation results for full-duplex dialogue synthesis compared to the latest internal dialogue synthesis model and to ground-truth results. On the short-form evaluation set,

![](images/8c02e4a031e31972da6f83ce1b94a71ff9321616beeed8e48a8e608a79d83520.jpg)  
Figure 4 Ablation on multi-stage reranking for voice dubbing.

Table 4 Human evaluation results for full-duplex dialogue synthesis compared to the latest internal model and groundtruth (GT). MOS scores range from 1 to 5.
<table><tr><td rowspan="2"></td><td colspan="2">Short-Form</td><td colspan="2">Long-Form</td></tr><tr><td>Text-AB</td><td>GT</td><td>Text-AB</td><td>Internal model</td></tr><tr><td>Intonation ↑</td><td>4.61</td><td>4.67</td><td>3.98</td><td>3.28</td></tr><tr><td>Pacing ↑</td><td>4.52</td><td>4.62</td><td>3.73</td><td>3.39</td></tr><tr><td>Expressive Intensity ↑</td><td>4.50</td><td>4.54</td><td>3.93</td><td>3.32</td></tr><tr><td>Expressive Correctness ↑</td><td>4.62</td><td>4.70</td><td>3.96</td><td>3.62</td></tr><tr><td>NSVs and Fillers ↑</td><td>4.76</td><td>4.81</td><td>4.12</td><td>4.57</td></tr><tr><td>Human Likeness ↑</td><td>4.53</td><td>4.62</td><td>3.86</td><td>3.00</td></tr></table>

Text-AB produces dialogues that closely approach real data, with an average gap of only −0.09 on overall human-likeness, and similarly strong scores across all fine-grained aspects. On the long-form evaluation set, Text-AB significantly outperforms the internal model in terms of overall human-likeness. For the more fine-grained dimensions, Text-AB also substantially outperforms the internal model on all aspects except NSVs and fillers, likely due to a mismatch between the content domain of the training conversations and that of the evaluation scripts.

Ablation Study. Figure 5 presents an ablation study of Text-AB across diferent model sizes, numbers of training steps, and initialization schemes. We observe that the 3B model significantly outperforms the 300M model on all metrics. Initializing from the pretrained Text-AB-Mono model yields much better performance than training from scratch, across all objective metrics. Moreover, continuing training for more updates consistently improves the results. As a result, a 3B model initialized from the pretrained checkpoint and trained for 500k updates already achieves performance comparable to ground-truth in terms of content accuracy, speaker similarity, and audio quality.

Figure 6 shows an ablation of multi-difusion configurations, varying chunk size, overlap size, number of reranking candidates, and ODE steps. Increasing the number of ODE steps and reranking candidates generally improves WER and Aes, but at the cost of a higher real-time factor (RTF). Speaker similarity is comparatively insensitive to these two hyperparameters. Overall, a 30 s chunk size with 20 s overlap ofers the best trade-of across all four metrics.

![](images/bb2ac72923a4a554b6cec7105e89bc94a386bab27bc81d8a6f108c7d3908cf40.jpg)

Figure 5 Ablation on model size, training steps, and initialization for full-duplex dialogue synthesis.  
![](images/ffa4eee0533a8b2fa8cd690a18c3bd770d9053079a8f7635e5845b6398e850c6.jpg)  
Figure 6 Ablation on multi-difusion configurations (chunk size, overlap size, number of candidates, and ODE steps) for full-duplex dialogue synthesis.

## 5.4 Emotional Full-Duplex Dialogue Synthesis Evaluation

Main Results. We compare four models: dialogue SFT, emotional dialogue SFT, MoonCast (Ju et al., 2025), and CosyVoice2 (Du et al., 2024b)-based utterance-level concatenation system. Dialogue SFT and emotional dialogue SFT are our proposed models, where the latter incorporates explicit turn-level emotion style embeddings to control emotional expression. MoonCast addresses a related task—speech dialogue generation from turn-level transcriptions—but difers in two key aspects: it produces single-channel speech containing two speakers and conditions on utterance-level speaker prompts without overlap. The CosyVoice2-based baseline independently generates each utterance by prompting CosyVoice2, a zero-shot speech generation model, with emotional reference speech to imitate the target emotion style, then concatenates the resulting utterances to form two-channel dialogues.

We evaluate along three dimensions. For emotion expressiveness, we measure the alignment between the conditioned emotion and the predicted emotion. As shown in Table 5, emotional dialogue SFT consistently outperforms dialogue SFT in both objective and subjective evaluations, demonstrating stronger alignment between intended and generated emotions.

For dialogue naturalness, Table 6 presents scores obtained from both LLM-based and human evaluations. Using ground-truth text dialogues, emotional dialogue SFT achieves the highest LLM-based naturalness score and the second-highest human evaluation score, indicating that explicit emotion conditioning improves perceived naturalness while maintaining strong human preference. Using synthetic emotional text dialogues, our models achieve the top two scores across both evaluation protocols, and emotional dialogue SFT attains the highest human evaluation score. These results further confirm that incorporating explicit emotion conditioning enhances dialogue naturalness under emotional settings.

For emotional interaction smoothness, we assess the quality of user-agent interaction conveyed through emotional acoustic cues, where human annotators evaluate whether emotional expressions across dialogue turns evolve naturally and coherently. As shown in Table 7, emotional dialogue SFT achieves the highest score, demonstrating its superior ability to generate natural and emotional interactions.

Table 5 Results of emotion expressiveness evaluation. Bold and underline indicate the best and second-best values.
<table><tr><td rowspan="2"></td><td colspan="3">SER: Qwen2-Audio (↑)</td><td colspan="3">Emotion Alignment MOS (↑)</td></tr><tr><td>angry</td><td>happy</td><td>sad</td><td>angry</td><td>happy</td><td>sad</td></tr><tr><td>Emotional dialogue SFT</td><td>0.814</td><td>0.428</td><td>0.479</td><td> $\mathbf { 3 . 5 7 6 _ { \pm } } _ { 0 . 2 5 2 }$ </td><td> $\mathbf { 3 . 3 3 3 } \pm \ : 0 . 3 8 0$ </td><td> $\pmb { 3 . 4 2 1 \pm \mathrm { ~ 0 . 2 8 8 } }$ </td></tr><tr><td>Dialogue SFT</td><td>0.650</td><td>0.396</td><td>0.455</td><td> $\underline { { 2 . 3 9 7 } } \pm \ : 0 . 2 3 6$ </td><td> $1 . 9 6 6 \pm \ : 0 . 2 9 5$ </td><td> $2 . 9 4 3 \pm \ : 0 . 2 3 8$ </td></tr></table>

Table 6 Results of speech dialog naturalness evaluation. Bold and underline indicate the best and second-best values.
<table><tr><td rowspan="2"></td><td colspan="2">Ground-Truth Text</td><td colspan="2">Emotional Text</td></tr><tr><td>LLM (↑)</td><td>MOS (↑)</td><td> $\mathrm { L L M } \left( \uparrow \right)$ </td><td>MOS (↑)</td></tr><tr><td>Ground-Truth</td><td> $4 . 8 4 4 { \scriptstyle \pm \mathrm { ~ 0 . 0 3 3 } }$ </td><td> $3 . 8 5 1 \pm \ : 0 . 1 8 8$ </td><td></td><td></td></tr><tr><td>Emotional dialogue SFT</td><td> $\pm . 6 4 3 _ { \pm } \mathrm { ~ } _ { 0 . 0 4 6 }$ </td><td> $3 . 9 0 6 \pm \ : 0 . 1 6 3$ </td><td> $\underline { { 3 . 6 5 3 } } \pm \ : 0 . 0 7 3$ </td><td> $\mathbf { 3 . 8 6 6 } _ { \pm } _ { \ 0 . 0 8 8 }$ </td></tr><tr><td>Dialogue SFT</td><td> $\underline { { 4 . 4 6 8 } } \pm \ : 0 . 0 5 3$ </td><td> $3 . 7 1 4 \pm \ : 0 . 1 8 9$ </td><td> $\mathbf { 3 . 6 8 0 } \pm \mathbf { \sigma } _ { 0 . 0 6 4 }$ </td><td> $3 . 0 9 0 \pm \ : 0 . 1 1 2$ </td></tr><tr><td>CosyVoice2 (Du et al., 2024b)</td><td> $4 . 2 8 5 \pm \ : 0 . 0 5 9$ </td><td> $3 . 8 6 7 _ { \pm \mathrm { ~ 0 . 2 0 3 ~ } }$ </td><td> $2 . 6 9 8 \pm \ : 0 . 0 8 1$ </td><td> $3 . 0 0 9 \pm \ : 0 . 1 8 1$ </td></tr><tr><td>MoonCast (Ju et al., 2025)</td><td> $4 . 4 3 6 \pm \ : 0 . 0 5 6$ </td><td> $\mathbf { 3 . 9 9 0 } \pm \mathbf { \sigma } _ { 0 . 1 7 8 }$ </td><td> $2 . 9 6 0 \pm \ : \mathrm { o . o 8 8 }$ </td><td> $2 . 9 3 0 \pm \ : 0 . 1 2 9$ </td></tr></table>

Table 7 Results of emotional interaction evaluation. Bold and underline indicate the best and second-best values.
<table><tr><td colspan="2">Emotional Interaction MOS (↑)</td></tr><tr><td>Emotional dialogue SFT</td><td> $\mathbf { 3 . 6 7 5 \pm { \ : 0 . 1 7 4 } }$ </td></tr><tr><td>Dialogue SFT</td><td> $\underline { { 3 . 3 1 7 } } \pm \ : 0 . 2 0 1$ </td></tr></table>

## 6 Related Work

Zero-Shot TTS. Recent years have witnessed major breakthroughs in zero-shot TTS driven by large-scale model training. VALL-E (Chen et al., 2025a) and a series of follow-up models (Zhang et al., 2023b; Kharitonov et al., 2023; Huang et al., 2023; Yang et al., 2023; Song et al., 2024; Xin et al., 2024; Łajszczak et al., 2024; Chen et al., 2024) represent speech as discrete codec tokens and formulate TTS as conditional codec language modeling, enabling training on large-scale speech data and performing zero-shot TTS via prompting with strong speaker generalization. Instead of using quantized discrete tokens, more recent work retains this languagemodeling formulation but operates on continuous latent features, achieving higher audio quality (Meng et al., 2025; Wang et al., 2025).

In parallel, many approaches leverage non-autoregressive generative models to improve inference speed and stability, including discrete-token prediction methods (Chang et al., 2022; Borsos et al., 2023) and difusionbased continuous-latent prediction approaches (Li et al., 2024; Du et al., 2024a; Shen et al., 2023; Ju et al., 2024). To further improve modeling capacity and audio quality, Audiobox-style models (Le et al., 2024; Vyas et al., 2023; Eskimez et al., 2024; Anastassiou et al., 2024; Chen et al., 2025b) adopt flow matching (Lipman et al., 2022) for training, achieving strong performance on zero-shot TTS and speech infilling tasks. Our work follows this Audiobox line of research but extends it in several ways: we move to a high-fidelity latent difusion space with DAC-VAE features (Polyak et al., 2024), remove the need for forced alignment via an alignment-free text interface, and scale model size, data, and training to obtain a unified framework that supports both high-quality voice dubbing and full-duplex dialogue synthesis.

Voice Dubbing. Voice dubbing aims to translate speech into a diferent language while preserving the speaker’s voice, emotion, and expressiveness from a source audio prompt. Practical dubbing systems typically adopt a cascaded pipeline (Yang et al., 2020; Federico et al., 2020; Artioli et al., 2025): an ASR model transcribes the source speech, a machine translation system produces the target-language transcript, and a TTS model generates the target speech conditioned on speaker characteristics inferred from the source audio. With recent advances in ASR (Li et al., 2022; Zhang et al., 2023a; Omnilingual et al., 2025) and machine translation (Wang et al., 2022; Barrault et al., 2023a; Costa-jussà et al., 2024), the TTS component has become the main remaining bottleneck. Conventional approaches either train a TTS model specifically for each speaker (Yang et al., 2020; Federico et al., 2020) or explicitly extract speaker attributes (e.g., timbre, style, emotion) or other pre-trained features from the source speech and then perform TTS conditioned on these representations (Cong et al., 2023, 2024; Artioli et al., 2025). Inspired by recent breakthroughs in zero-shot TTS, more recent work instead preserves the source speaker’s characteristics implicitly by conditioning directly on the raw source speech (Sung-Bin et al., 2025). Our work follows this latter line and treats dubbing as a cross-lingual zero-shot TTS problem: we train a large-scale model that conditions on source audio and text, and show that it achieves state-of-the-art dubbing performance.

Full-Duplex Dialogue Synthesis. A common approach to dialogue synthesis is to generate each turn as an independent monologue and then concatenate the turns into a conversation (Guo et al., 2021; Xue et al., 2023; Lee et al., 2023; Hu et al., 2024; Liu et al., 2024; Xie et al., 2025). While this strategy can produce high-quality, natural speech within each turn, it often yields unnatural interactions, weak turn coordination, and limited control over conversational dynamics. Recent work has therefore shifted toward models that generate entire dialogues in a more integrated manner, explicitly modeling multi-speaker interactions. This includes autoregressive token-prediction approaches (Zhang et al., 2024; Schalkwyk et al.; Borsos et al.; Darefsky et al., 2024; Ju et al., 2025; Peng et al., 2025) and fully non-autoregressive flow-matching-based methods (Zhang et al., 2025).

Despite this progress, most of these systems operate in a single-channel setting and do not produce stereo fullduplex dialogue, which is crucial for building high-quality training data for full-duplex speech LLMs (Défossez et al., 2024; Cui et al., 2025). dGSLM-style models (Nguyen et al., 2023; Mitsui et al., 2023; Lu et al., 2025) address this by using dual-tower Transformer architectures to capture interleaved speaker information and generate two-channel spoken dialogue autoregressively. ZipVoice-Dialog (Zhu et al., 2025) further proposes strategies for stereo full-duplex dialogue generation with a non-autoregressive flow-matching-based model. However, existing stereo full-duplex dialogue systems either lack zero-shot voice prompting capabilities or are trained on relatively limited data with modest model scales. In contrast, our work supports zero-shot audio prompting and scales to a 3B-parameter model trained on tens of thousands of hours of two-channel full-duplex dialogue data, targeting high-quality, controllable full-duplex dialogue synthesis.

Emotional Speech Synthesis. Prior work on emotional speech generation primarily conditions emotion using categorical labels (Wu et al., 2019) or style embeddings extracted from reference speech (He et al., 2022). Subsequent studies introduce continuous scalar variables to control emotion intensity (Zhu et al., 2019; Zhou et al., 2023), further enabling mixed or compound emotion synthesis through relative attributes (Zhou et al., 2022; Inoue et al., 2024, 2025) or difusion-based interpolation (Tang et al., 2023). To improve interpretability and fine-grained control of generated emotion, recent approaches explore structured emotion representations, including explicit prosody modeling (Luo et al., 2021; Oh et al., 2023; Ren et al., 2021) and the VAD (Sivaprasad et al., 2021; Zhou et al., 2025; Habib et al., 2019; Cho et al., 2024, 2025), which represents emotion along three continuous dimensions: valence, arousal, and dominance (Russell and Mehrabian, 1977). However, all of these methods focus on single-utterance or monologue synthesis and do not address multi-turn dialogue generation. Our emotional dialogue SFT extends VAD-based emotion conditioning to two-channel full-duplex dialogue synthesis, enabling turn-level emotion control within natural conversations.

## 7 Conclusion

We introduced Text-AB, an alignment-free TTS framework for voice dubbing, full-duplex dialogue synthesis, and emotional full-duplex dialogue synthesis that unifies single- and two-channel speech generation within a single flow-matching-based DiT architecture. Text-AB successfully simplified the modeling stack while improving expressivity and scalability by operating in a high-fidelity latent difusion space with DAC-VAE features, directly conditioning on raw text via cross-attention, and substantially scaling model and data size. We first pretrained Text-AB on a large-scale multilingual monologue dataset, then fine-tuned on three downstream tasks—dubbing SFT, dialogue SFT, and emotional dialogue SFT. This enabled a 3B-parameter model to generalize across monolingual, cross-lingual, and two-channel dialogue settings. At inference time, multi-stage reranking strategy enabled high-quality generation, and a multi-difusion scheme extended one-shot generation to arbitrarily long-form audio with seamless transitions.

Empirically, Text-AB sets a new bar for both voice dubbing and full-duplex dialogue synthesis in our production setting. It substantially outperformed our internal dubbing model in human evaluations, and produced dialogues that closely match real recordings on short-form benchmarks while markedly improving human-likeness and expressivity over the prior internal dialogue synthesis system on long-form tasks. Beyond these results, we believe the alignment-free latent-difusion design and the multi-difusion inference strategy provide a general recipe for scalable, controllable speech generation. Future directions include extending Text-AB to more languages, modalities, and tasks, further scaling model and data size, and exploring finer-grained controls over style, emotion, and conversational structure.

## References

Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024.

Emanuele Artioli, Daniele Lorenzi, Farzad Tashtarian, and Christian Timmerer. Generative ai for realistic voice dubbing across languages. In Proceedings of the 4th Mile-High Video Conference, pages 75–76, 2025.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. NeurIPS, 33:12449–12460, 2020.

Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Hefernan, John Hofman, et al. Seamlessm4t: massively multilingual & multimodal machine translation. arXiv preprint arXiv:2308.11596, 2023a.

Loïc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, et al. Seamless: Multilingual expressive and streaming speech translation. arXiv preprint arXiv:2312.05187, 2023b.

Zalán Borsos, Matt Sharifi, Damien Vincent, Eugene Kharitonov, Neil Zeghidour, and Marco Tagliasacchi. Soundstorm: Eficient parallel audio generation. arXiv preprint arXiv:2305.09636, 2023.

Zalán Borsos, Matt Sharifi, and Marco Tagliasacchi. Pushing the frontiers of audio generation — deepmind.google. https://deepmind.google/blog/pushing-the-frontiers-of-audio-generation. Accessed 30-10-2024.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6):1505–1518, 2022.

Sanyuan Chen, Shujie Liu, Long Zhou, Yanqing Liu, Xu Tan, Jinyu Li, Sheng Zhao, Yao Qian, and Furu Wei. Vall-e 2: Neural codec language models are human parity zero-shot text to speech synthesizers. arXiv preprint arXiv:2406.05370, 2024.

Sanyuan Chen, Chengyi Wang, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Neural codec language models are zero-shot text to speech synthesizers. IEEE Transactions on Audio, Speech and Language Processing, 2025a.

Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, JianZhao JianZhao, Kai Yu, and Xie Chen. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6255–6271, 2025b.

Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Sang-Hoon Lee, and Seong-Whan Lee. Emosphere-tts: Emotional style and intensity modeling via spherical emotion vector for controllable emotional text-to-speech. pages 1810–1814, 09 2024. doi: 10.21437/Interspeech.2024-398.

Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, and Seong-Whan Lee. Emosphere++: Emotion-controllable zero-shot text-to-speech via emotion-adaptive spherical vector. IEEE Transactions on Afective Computing, 16(3): 2365–2380, 2025. doi: 10.1109/TAFFC.2025.3561267.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen2-audio technical report, 2024. https://arxiv.org/abs/2407.10759.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gafney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. https: //arxiv.org/abs/2507.06261.

Gaoxiang Cong, Liang Li, Yuankai Qi, Zheng-Jun Zha, Qi Wu, Wenyu Wang, Bin Jiang, Ming-Hsuan Yang, and Qingming Huang. Learning to dub movies via hierarchical prosody models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14687–14697, 2023.

Gaoxiang Cong, Yuankai Qi, Liang Li, Amin Beheshti, Zhedong Zhang, Anton Hengel, Ming-Hsuan Yang, Chenggang Yan, and Qingming Huang. Styledubber: Towards multi-scale style learning for movie dubbing. In Findings of the Association for Computational Linguistics: ACL 2024, pages 6767–6779, 2024.

Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Hefernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hofman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jef Wang. Scaling neural machine translation to 200 languages. Nature, 630:41–47, 2024. doi: 10.1038/s41586-024-07335-x. https://www.nature.com/articles/ s41586-024-07335-x.

Wenqian Cui, Dianzhi Yu, Xiaoqi Jiao, Ziqiao Meng, Guangyan Zhang, Qichao Wang, Steven Y Guo, and Irwin King. Recent advances in speech language models: A survey. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13943–13970, 2025.

Jordan Darefsky, Ge Zhu, and Zhiyao Duan. Parakeet, 2024. https://jordandarefsky.com/blog/2024/parakeet/.

Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024.

Chenpeng Du, Yiwei Guo, Feiyu Shen, Zhijun Liu, Zheng Liang, Xie Chen, Shuai Wang, Hui Zhang, and Kai Yu. Unicats: A unified context-aware text-to-speech framework with contextual vq-difusion and vocoding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 17924–17932, 2024a.

Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, Fan Yu, Huadai Liu, Zhengyan Sheng, Yue Gu, Chong Deng, Wen Wang, Shiliang Zhang, Zhijie Yan, and Jingren Zhou. Cosyvoice 2: Scalable streaming speech synthesis with large language models, 2024b. https://arxiv.org/abs/2412.10117.

Sefik Emre Eskimez, Xiaofei Wang, Manthan Thakker, Canrun Li, Chung-Hsien Tsai, Zhen Xiao, Hemin Yang, Zirun Zhu, Min Tang, Xu Tan, et al. E2 tts: Embarrassingly easy fully non-autoregressive zero-shot tts. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 682–689. IEEE, 2024.

Marcello Federico, Robert Enyedi, Roberto Barra-Chicote, Ritwik Giri, Umut Isik, Arvindh Krishnaswamy, and Hassan Sawaf. From speech-to-speech translation to automatic dubbing. arXiv preprint arXiv:2001.06785, 2020.

Haohan Guo, Shaofei Zhang, Frank K Soong, Lei He, and Lei Xie. Conversational end-to-end tts for voice agents. In 2021 IEEE Spoken Language Technology Workshop (SLT), pages 403–409. IEEE, 2021.

Raza Habib, Soroosh Mariooryad, Matt Shannon, Eric Battenberg, R. J. Skerry-Ryan, Daisy Stanton, David Kao, and Tom Bagby. Semi-supervised generative modeling for controllable speech synthesis. ArXiv, abs/1910.01709, 2019. https://api.semanticscholar.org/CorpusID:203736888.

Jiaxu He, Cheng Gong, Longbiao Wang, Di Jin, Xiaobao Wang, Junhai Xu, and Jianwu Dang. Improve emotional speech synthesis quality by learning explicit and implicit representations with semi-supervised training. In Interspeech, 2022. https://api.semanticscholar.org/CorpusID:252353149.

Yifan Hu, Rui Liu, Guanglai Gao, and Haizhou Li. Fctalker: Fine and coarse grained context modeling for expressive conversational speech synthesis. In 2024 IEEE 14th International Symposium on Chinese Spoken Language Processing (ISCSLP), pages 299–303. IEEE, 2024.

Rongjie Huang, Chunlei Zhang, Yongqi Wang, Dongchao Yang, Luping Liu, Zhenhui Ye, Ziyue Jiang, Chao Weng, Zhou Zhao, and Dong Yu. Make-a-voice: Unified voice synthesis with discrete representation. arXiv preprint arXiv:2305.19269, 2023.

Sho Inoue, Kun Zhou, Shuai Wang, and Haizhou Li. Hierarchical emotion prediction and control in text-to-speech synthesis. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 10601–10605. IEEE, 2024.

Sho Inoue, Kun Zhou, Shuai Wang, and Haizhou Li. Hierarchical control of emotion rendering in speech synthesis. IEEE Transactions on Afective Computing, 16(4):3316–3328, 2025. doi: 10.1109/TAFFC.2025.3582715.

Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Yanqing Liu, Yichong Leng, Kaitao Song, Siliang Tang, et al. Naturalspeech 3: Zero-shot speech synthesis with factorized codec and difusion models. arXiv preprint arXiv:2403.03100, 2024.

Zeqian Ju, Dongchao Yang, Jianwei Yu, Kai Shen, Yichong Leng, Zhengtao Wang, Xu Tan, Xinyu Zhou, Tao Qin, and Xiangyang Li. Mooncast: High-quality zero-shot podcast generation. arXiv preprint arXiv:2503.14345, 2025.

Eugene Kharitonov, Damien Vincent, Zalán Borsos, Raphaël Marinier, Sertan Girgin, Olivier Pietquin, Matt Sharifi, Marco Tagliasacchi, and Neil Zeghidour. Speak, read and prompt: High-fidelity text-to-speech with minimal supervision. Transactions of the Association for Computational Linguistics, 11:1703–1718, 2023.

Mateusz Łajszczak, Guillermo Cámbara, Yang Li, Fatih Beyhan, Arent van Korlaar, Fan Yang, Arnaud Joly, Álvaro Martín-Cortinas, Ammar Abbas, Adam Michalski, et al. Base tts: Lessons from building a billion-parameter text-to-speech model on 100k hours of data. arXiv preprint arXiv:2402.08093, 2024.

Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, et al. Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36, 2024.

Keon Lee, Kyumin Park, and Daeyoung Kim. Dailytalk: Spoken dialogue dataset for conversational text-to-speech. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Jinyu Li et al. Recent advances in end-to-end automatic speech recognition. APSIPA Transactions on Signal and Information Processing, 11(1), 2022.

Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, and Ming Liu. Neural speech synthesis with transformer network. In AAAI, pages 6706–6713. AAAI, 2019.

Yinghao Aaron Li, Cong Han, Vinay Raghavan, Gavin Mischler, and Nima Mesgarani. Styletts 2: Towards human-level text-to-speech through style difusion and adversarial training with large speech language models. Advances in Neural Information Processing Systems, 36, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Rui Liu, Yifan Hu, Yi Ren, Xiang Yin, and Haizhou Li. Generative expressive conversational speech synthesis. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 4187–4196, 2024.

Haitian Lu, Gaofeng Cheng, Liuping Luo, Leying Zhang, Yanmin Qian, and Pengyuan Zhang. Slide: Integrating speech language model with llm for spontaneous spoken dialogue generation. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

Xuan Luo, Shinnosuke Takamichi, Tomoki Koriyama, Yuki Saito, and Hiroshi Saruwatari. Emotion-controllable speech synthesis using emotion soft labels and fine-grained prosody factors. In 2021 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 794–799, 2021.

Lingwei Meng, Long Zhou, Shujie Liu, Sanyuan Chen, Bing Han, Shujie Hu, Yanqing Liu, Jinyu Li, Sheng Zhao, Xixin Wu, et al. Autoregressive speech synthesis without vector quantization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1287–1300, 2025.

Kentaro Mitsui, Yukiya Hono, and Kei Sawada. Towards human-like spoken dialogue generation between ai agents from written dialogue. arXiv preprint arXiv:2310.01088, 2023.

Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoit Sagot, Abdelrahman Mohamed, et al. Generative spoken dialogue language modeling. Transactions of the Association for Computational Linguistics, 11:250–266, 2023.

Yoori Oh, Juheon Lee, Yoseob Han, and Kyogu Lee. Semi-supervised learning for continuous emotional intensity controllable speech synthesis with disentangled representations. pages 4818–4822, 08 2023. doi: 10.21437/Interspeech. 2023-1405.

ASR Omnilingual, Gil Keren, Artyom Kozhevnikov, Yen Meng, Christophe Ropers, Matthew Setzler, Skyler Wang, Ife Adebara, Michael Auli, Can Balioglu, et al. Omnilingual asr: Open-source multilingual speech recognition for 1600+ languages. arXiv preprint arXiv:2511.09690, 2025.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, et al. Vibevoice technical report. arXiv preprint arXiv:2508.19205, 2025.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.

Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. Fastspeech: Fast, robust and controllable text to speech. In NeurIPS, pages 3165–3174, 2019.

Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. Fastspeech 2: Fast and high-quality end-to-end text to speech. In International Conference on Learning Representations, 2021. https: //openreview.net/forum?id=piLPYqxtWuA.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

James Russell and Albert Mehrabian. Evidence for a three-factor theory of emotions. Journal of Research in Personality, 11:273–294, 09 1977. doi: 10.1016/0092-6566(77)90037-X.

Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, and Raven Jiang. Crossing the uncanny valley of conversational voice — sesame.com. https://www.sesame.com research/crossing\_the\_uncanny\_valley\_of\_voice. Accessed 27-02-2025.

Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ-Skerrv Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, and Yonghui Wu. Natural TTS synthesis by conditioning wavenet on MEL spectrogram predictions. In ICASSP, pages 4779–4783. IEEE, 2018.

Kai Shen, Zeqian Ju, Xu Tan, Eric Liu, Yichong Leng, Lei He, Tao Qin, Jiang Bian, et al. Naturalspeech 2: Latent difusion models are natural and zero-shot speech and singing synthesizers. In The Twelfth International Conference on Learning Representations, 2023.

Sarath Sivaprasad, Saiteja Kosgi, and Vineet Gandhi. Emotional prosody control for speech generation. In Interspeech, 2021. https://api.semanticscholar.org/CorpusID:239714949.

Yakun Song, Zhuo Chen, Xiaofei Wang, Ziyang Ma, and Xie Chen. Ella-v: Stable neural codec language modeling with alignment-guided sequence reordering. arXiv preprint arXiv:2401.07333, 2024.

Kim Sung-Bin, Jeongsoo Choi, Puyuan Peng, Joon Son Chung, Tae-Hyun Oh, and David Harwath. Voicecraft-dub: Automated video dubbing with neural codec language models. arXiv preprint arXiv:2504.02386, 2025.

Haobin Tang, Xulong Zhang, Jianzong Wang, Ning Cheng, and Jing Xiao. Emomix: Emotion mixing via difusion models for emotional speech synthesis, 2023.

Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hofman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 30, 2017.

Apoorv Vyas, Bowen Shi, Matthew Le, Andros Tjandra, Yi-Chiao Wu, Baishan Guo, Jiemin Zhang, Xinyue Zhang, Robert Adkins, William Ngan, et al. Audiobox: Unified audio generation with natural language prompts. arXiv preprint arXiv:2312.15821, 2023.

Haifeng Wang, Hua Wu, Zhongjun He, Liang Huang, and Kenneth Ward Church. Progress in machine translation. Engineering, 18:143–153, 2022.

Hui Wang, Shujie Liu, Lingwei Meng, Jinyu Li, Yifan Yang, Shiwan Zhao, Haiyang Sun, Yanqing Liu, Haoqin Sun, Jiaming Zhou, et al. Felle: Autoregressive speech synthesis with token-wise coarse-to-fine flow matching. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10229–10238, 2025.

Peng-fei Wu, Zhen-hua Ling, Li-juan Liu, Yuan Jiang, Hong-chuan Wu, and Li-rong Dai. End-to-end emotional speech synthesis using style tokens and semi-supervised training, 2019.

Kun Xie, Feiyu Shen, Junjie Li, Fenglong Xie, Xu Tang, and Yao Hu. Fireredtts-2: Towards long conversational speech generation for podcast and chatbot. arXiv preprint arXiv:2509.02020, 2025.

Detai Xin, Xu Tan, Kai Shen, Zeqian Ju, Dongchao Yang, Yuancheng Wang, Shinnosuke Takamichi, Hiroshi Saruwatari, Shujie Liu, Jinyu Li, et al. Rall-e: Robust codec language modeling with chain-of-thought prompting for text-to-speech synthesis. arXiv preprint arXiv:2404.03204, 2024.

Jinlong Xue, Yayue Deng, Fengping Wang, Ya Li, Yingming Gao, Jianhua Tao, Jianqing Sun, and Jiaen Liang. M 2-ctts: End-to-end multi-scale multi-modal conversational text-to-speech synthesis. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Rafel. mt5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 conference of the North American chapter of the association for computational linguistics: Human language technologies, pages 483–498, 2021.

Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Xixin Wu, et al. Uniaudio: An audio foundation model toward universal audio generation. arXiv preprint arXiv:2310.00704, 2023.

Yi Yang, Brendan Shillingford, Yannis Assael, Miaosen Wang, Wendi Liu, Yutian Chen, Yu Zhang, Eren Sezener, Luis C Cobo, Misha Denil, et al. Large-scale multilingual audio visual dubbing. arXiv preprint arXiv:2011.03530, 2020.

Leying Zhang, Yao Qian, Long Zhou, Shujie Liu, Dongmei Wang, Xiaofei Wang, Midia Yousefi, Yanmin Qian, Jinyu Li, Lei He, et al. Covomix: Advancing zero-shot speech generation for human-like multi-talker conversations. Advances in Neural Information Processing Systems, 37:100291–100317, 2024.

Leying Zhang, Yao Qian, Xiaofei Wang, Manthan Thakker, Dongmei Wang, Jianwei Yu, Haibin Wu, Yuxuan Hu, Jinyu Li, Yanmin Qian, et al. Covomix2: Advancing zero-shot dialogue generation with fully non-autoregressive flow matching. arXiv preprint arXiv:2506.00885, 2025.

Yu Zhang, Wei Han, James Qin, Yongqiang Wang, Ankur Bapna, Zhehuai Chen, Nanxin Chen, Bo Li, Vera Axelrod, Gary Wang, et al. Google usm: Scaling automatic speech recognition beyond 100 languages. arXiv preprint arXiv:2303.01037, 2023a.

Ziqiang Zhang, Long Zhou, Chengyi Wang, Sanyuan Chen, Yu Wu, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Speak foreign languages with your own voice: Cross-lingual neural codec language modeling. arXiv preprint arXiv:2303.03926, 2023b.

Kun Zhou, Berrak Sisman, Rajib Rana, B. W. Schuller, and Haizhou Li. Speech synthesis with mixed emotions, 2022.

Kun Zhou, Berrak Sisman, Rajib Rana, Björn W. Schuller, and Haizhou Li. Emotion intensity and its control for emotional voice conversion. IEEE Trans. Afect. Comput., 14(1):31–48, January 2023. ISSN 1949-3045. doi: 10.1109/TAFFC.2022.3175578. https://doi.org/10.1109/TAFFC.2022.3175578.

Kun Zhou, You Zhang, Shengkui Zhao, Hao Wang, Zexu Pan, Dianwen Ng, Chong Zhang, Chongjia Ni, Yukun Ma, Trung Hieu Nguyen, Jia Qi Yip, and Bin Ma. Emotional dimension control in language model-based text-to-speech: Spanning a broad spectrum of human emotions, 2025. https://arxiv.org/abs/2409.16681.

Han Zhu, Wei Kang, Liyong Guo, Zengwei Yao, Fangjun Kuang, Weiji Zhuang, Zhaoqing Li, Zhifeng Han, Dong Zhang, Xin Zhang, et al. Zipvoice-dialog: Non-autoregressive spoken dialogue generation with flow matching. arXiv preprint arXiv:2507.09318, 2025.

Xiaolian Zhu, Shan Yang, Geng Yang, and Lei Xie. Controlling emotion strength with relative attribute for end-to-end speech synthesis. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 192–199, 2019. doi: 10.1109/ASRU46091.2019.9003829.