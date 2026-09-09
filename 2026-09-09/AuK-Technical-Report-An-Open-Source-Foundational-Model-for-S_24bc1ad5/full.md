# AuK Technical Report: An Open-Source Foundational Model for Speech Generation and Editing

 Project Page GitHub AuK AuK-Flash

## Abstract

We introduce AuK, an open-source foundational model that unifies speech generation and editing through a common interface of natural-language instructions and audio context. To support this broad capability set, we construct approximately 3.03 billion instruction–audio instances and 1.95 million hours of effective supervision across five task families: speech generation, content editing, enhancement and separation, paralinguistic editing, and acoustic editing. AuK combines a multimodal large language model for semantic conditioning, an VAE jointly trained on speech, general audio, and music for acoustic conditioning, and a hybrid rectifiedflow Transformer that performs dual-stream MMDiT blocks followed by unified single-stream DiT blocks for generation. Training begins with generation-only warm-up and proceeds to joint generation–editing pre-training. We then apply complementary post-training strategies: human-feedback preference optimization for open-ended editing and reward-based reinforcement learning for speech generation. To reduce inference cost, we further distill the model with consistency initialization and task-routed Decoupled DMD. The resulting AuK-Flash performs 4-step inference without classifier-free guidance and achieves a 4.5× wall-clock speedup over the full model under matched conditions. Experiments demonstrate leading performance on zero-shot and instruction-controlled speech generation and general instruction-guided editing, while remaining competitive on signal-level restoration tasks. We release both the source code and model weights to support reproducibility and further research.

![](images/4dc3b828fda28522ae12aebdf2fb3ca9d896b14febb5e16ec688e0a16bc47782.jpg)  
Figure 1: Performance comparison with SOTA models across speech generation, editing, enhancement and separation. (a) Seed-TTS-Eval WER and SIM, averaged over test-en/test-zh/testzh-hard, measure intelligibility and speaker similarity; InstructTTSEval DSD (mean of DSD-ZH/EN) measures how faithfully a model realises a free-form timbre instruction. (b) MMAE-Speech EMR measures exact-match success across all editing rubrics; SpeechEditBench averages five edit types, Ming-Freeform-Audio-Edit four semantic-editing splits. (c) Per model, DNSMOS-OVRL (darker, left) and UTMOS (lighter, right) rate perceptual quality; DNS Challenge and CHiME-4 are enhancement, Libri2Mix separation. Higher is better everywhere except WER, where shorter bars indicate fewer errors.

## Contents

1 Introduction 3   
2 Data Construction 4   
2.1 Speech Generation 4   
2.2 Acoustic Editing 5   
2.3 Paralinguistic Editing 6   
2.4 Content Editing 6   
2.5 Enhancement and Separation 7   
3 Model Design 8   
3.1 Overall Architecture 8   
3.2 MLLM Semantic Condition 9   
3.3 VAE Acoustic Condition and Reconstruction . 9   
3.4 Transformer Backbone 9   
4 Model Training 10   
4.1 VAE Training 10   
4.2 Unified Pre-Training 11   
4.3 Post-Training 12   
4.3.1 Editing Preference Optimization . 12   
4.3.2 Generation Reinforcement Learning 13   
5 Model Acceleration 14   
6 Model Inference 15   
6.1 Prompt Enhancer 15   
6.2 Inference Strategy and Configuration . 16   
7 Performance 16   
7.1 VAE Reconstruction Results 18   
7.2 Generation Ability 18   
7.3 Editing Ability 19   
7.3.1 General Speech Editing . 19   
7.3.2 Speech Enhancement and Separation 19   
7.4 Discovery 20   
8 Conclusion 21   
Contribution 22   
A Detailed Evaluation Results 29   
A.1 Generation Benchmarks . 29   
A.2 General Speech Editing Benchmarks . 30   
A.3 Speech Enhancement and Separation Benchmarks . 31

![](images/d6fced5d899922355bf0a769bc1cc0c2da96ebf99e76cd5dba6a7b837919c73b.jpg)  
Figure 2: Versatile speech generation and editing capabilities of AuK. The model supports five task families: (1) speech generation, including instruction-based and zero-shot TTS; (2) acoustic editing of speaking rate, loudness, and pitch; (3) paralinguistic editing of emotion, accent, nonverbal vocalizations, timbre, and whisper style; (4) content editing of spoken words and song lyrics; and (5) enhancement and separation of speech and music. The examples illustrate how natural-language instructions and speech input are mapped to output speech.

## 1 Introduction

Recent speech generation systems have advanced from conventional text-to-speech toward zero-shot voice cloning [10, 12, 11, 6, 68, 86, 29, 1, 28, 51, 88], instruction-controlled synthesis [22, 78, 23, 87, 19, 38, 24], and increasingly flexible speech editing [72, 73, 63, 65, 4, 3, 33]. In practical use, however, these capabilities rarely appear in isolation. A user may ask a system to synthesize speech in a described style, replace part of an utterance, alter its emotion or accent, insert a nonverbal vocalization, isolate a speaker, or restore degraded audio. Supporting such requests with separate task-specific models fragments the user experience and duplicates modeling effort. This motivates a unified model that interprets free-form instructions and generates the requested speech.

Unifying these capabilities is challenging for three reasons. First, their output constraints differ fundamentally: generation creates new speech, content editing changes only selected regions, paralinguistic and acoustic editing must preserve linguistic content, and enhancement or separation must retain only the scene components specified by the instruction. Second, the conditioning interface varies across tasks. Some tasks rely on text alone, whereas others require joint reasoning over an instruction and source or reference audio. Third, supervision and evaluation are heterogeneous. Recognition accuracy and speaker similarity provide scalable signals for generation, but open-ended editing also depends on subjective judgments of naturalness, edit strength, contextual appropriateness, and preservation of unspecified attributes. Existing unified audio systems and editing benchmarks have begun to expose this broader problem [72, 73, 44, 79], yet a single model with broad task coverage, scalable training, and efficient inference remains difficult to realize.

We introduce AuK, a unified foundational model for speech generation and editing. As illustrated in Fig. 2, AuK spans speech generation, low-level acoustic control, paralinguistic transformation, content editing, and signal enhancement or separation. We formulate these capabilities through a common interface: a natural-language instruction and optional audio context are mapped to a target waveform. AuK combines an MLLM semantic encoder, an audio VAE, and a hybrid flow Transformer. The MLLM [69] encodes either text alone or text jointly with reference audio, and aggregates hierarchical hidden states into a semantic condition. The AuK-VAE, jointly trained on speech, general audio, and music, provides a shared acoustic latent space for reference conditioning and waveform reconstruction. Dual-stream MMDiT [32] blocks first exchange information between semantic and acoustic streams while preserving their distinct residual pathways; subsequent singlestream DiT blocks jointly refine the fused sequence and predict the rectified-flow velocity of the target latent. This design supports text-only generation and reference-conditioned editing within the same backbone.

Training is organized to address the different requirements of generation and editing. Unified pretraining begins with a generation-only warm-up and then jointly optimizes generation and editing tasks with a shared flow-matching objective. Post-training contains two complementary stages. For editing, where task completion is open-ended and no sufficiently broad preference dataset or reward model exists, we collect human feedback on free-form editing requests and perform flow-based preference optimization. For generation, we apply Flow-GRPO with automatic rewards for content correctness, speaker similarity, and instruction–style consistency. Finally, we distill the post-trained model through consistency initialization and task-routed Decoupled DMD. The resulting AuK-Flash performs 4-step, CFG-free inference and achieves a 4.5× wall-clock speedup over the 32-NFE teacher under matched conditions.

Experiments cover AuK-VAE reconstruction, zero-shot and instruction speech generation, general instruction-guided speech editing, speech enhancement, separation, and super-resolution. Across these evaluations, AuK achieves leading performance on speech generation and general instructionguided editing benchmarks while remaining competitive on signal-level restoration tasks. AuK-Flash retains broad generation and editing capability under substantially reduced inference cost. We release both the source code and model weights to facilitate open-source community and further research.

## 2 Data Construction

As summarized in Fig. 3, we organize the pre-training corpus into five task families: speech generation, acoustic editing, paralinguistic editing, content editing, and enhancement and separation. Despite their different objectives, all tasks share a unified interface consisting of a natural-language in struction, optional input audio, and a target waveform. This formulation allows a single model to learn generation, restoration, separation, and editing from approximately 3.03 billion instruction–audio instances, with a total of 1.95 million hours of effective audio supervision.

## 2.1 Speech Generation

Speech generation teaches the model to synthesize natural, intelligible, and controllable speech from text. We construct two complementary forms of supervision: transcript-free zero-shot TTS conditioned on reference speech, and instruct TTS controlled by free-form descriptions.

Zero-Shot TTS. We build a large-scale bilingual corpus through a multi-stage curation pipeline. Source separation and speech enhancement are first applied to improve signal quality, followed by MOS-based quality filtering, speaker-identity verification, and cross-validation with multiple ASR systems to remove noisy or inconsistent utterances.

![](images/5224e1eaf3a7b5913fff0d23d9fc83d0340aaf536ceccef6a28417496a8b0c36.jpg)  
Figure 3: Overview of the pre-training corpus. The inner ring groups tasks into five capability families, while the outer rings summarize their instruction-level operations. The surrounding panels illustrate representative tasks and their intended functions. Sector widths are adjusted for readability and do not indicate data volume.

Conventional zero-shot TTS often requires both a reference utterance and its transcript, making deployment dependent on an additional ASR system. We instead formulate zero-shot TTS as transcript-free in-context learning. For a speaker with n distinct utterances, we enumerate all C<sup>2</sup><sub>n</sub> unordered pairs and assign each utterance in a pair once as the acoustic prompt and once as the synthesis target, yielding n × (n − 1) bidirectional training instances. Each instance contains only the prompt speech and target text; the prompt transcript is never provided. At inference time, the model can therefore clone a speaker from either a complete reference utterance or a randomly cropped segment without requiring a transcription.

Instruct TTS. We derive the Instruct TTS corpus from a large-scale bilingual speech collection that has undergone standardized preprocessing and quality control. Qwen3-Omni [70] annotates each retained utterance with a free-form natural-language caption and structured attributes covering gender, age, speaking rate, clarity, fluency, vocal state, intonation, loudness, timbre, pitch, accent, emotion, and personality. We combine this description with the target text to form the input instruction and use the corresponding waveform as the synthesis target. Because no reference audio is provided, the resulting pairs teach the model to design the voice directly from the expressive descriptions.

## 2.2 Acoustic Editing

Acoustic editing teaches the model to control low-level speaking attributes, including speaking rate, pitch, and loudness, while preserving linguistic content, speaker identity, and all non-target characteristics. We generate paired supervision using deterministic signal-processing transformations.

For each source utterance, we create targets at five speaking-rate multipliers (0.5×, 0.75×, 1.25×, 1.5×, and 2.0×), six loudness offsets (±5, ±10, and ±15 dB), and six pitch shifts (±1, ±2, and ±3 semitones). Speaking rate is modified using pitch-preserving time stretching, pitch using duration-

preserving pitch shifting, and loudness using waveform gain. Each transformed waveform is paired with its source and a natural-language instruction specifying the attribute and requested magnitude. We validate every transformation and apply peak protection when necessary to prevent clipping.

## 2.3 Paralinguistic Editing

Paralinguistic editing teaches the model to modify how an utterance is delivered while preserving what is said. We construct paired supervision for emotion, timbre, accent, nonverbal vocalization, and whisper-style editing.

Emotion Editing. We construct emotion editing samples from the bilingual speech pool used for Instruct TTS to ensure that the speech is expressive. Target emotions are sampled from eight categories: angry, happy, sad, fearful, surprised, disgusted, calm, and excited. Given a source transcript and target-emotion instruction, Qwen3-TTS-CustomVoice [22] first synthesizes an expressive reference utterance. IndexTTS2 [86] is then conditioned on the original utterance for speaker characteristics and on the synthesized reference for emotion, while the transcript remains fixed. The resulting waveform is paired with the source audio and a natural-language editing instruction.

Timbre Editing. We adopt the X-VC [85] training corpus, which is constructed using SeedVC-Small [41]. Each group contains four aligned source–target pairs that preserve linguistic content while changing speaker timbre. All waveforms are enhanced with speech super-resolution and standardized to a sampling rate of 24 kHz. Qwen3-Omni [70] generates a natural-language timbre description for each target waveform, which is incorporated into the corresponding editing instruction.

De-accent. We construct de-accenting pairs from an in-house corpus spanning 13 Chinese dialect and regional-accent categories. For each accented source utterance, CosyVoice2 [12] first synthesizes a same-speaker standard Mandarin reference from independently sampled text, using the source utterance as the speaker prompt. We then partially mask the source and use OmniVoice [89] to reconstruct it conditioned on the source transcript and synthesized standard Mandarin reference. The target follows standard Mandarin pronunciation while preserving the source speaker’s timbre and prosodic characteristics.

Nonverbal Editing. We build the nonverbal-editing corpus from both public datasets and in-house datasets with heterogeneous human- and model-derived annotations. We normalize these annotations into 39 event types spanning physiological sounds, affective expressions, and discourse vocalizations. For each event, Qwen3-ForcedAligner [59] locates its temporal span. We mask the event and its immediate context, then use F5-TTS [6] to reconstruct an event-free waveform while preserving the surrounding speech, speaker identity, and prosody. Each original–reconstructed pair supports both event removal and insertion, with a natural-language instruction specifying the event type and location.

Whisper-Style Conversion. We construct normal-to-whisper pairs from public Mandarin corpus containing parallel normal and whispered speech. We retain only pairs whose normal and whispered transcripts satisfy WER = 0. All recordings are resampled to 24 kHz, and the normal-speech inputs are normalized per utterance to an RMS target of −24 dBFS, matching the average level of the broader speech training corpus.

## 2.4 Content Editing

Content editing teaches the model to insert, delete, or replace spoken and sung content while preserving speaker identity, prosody, melody, and the acoustic context outside the edited region. We construct paired supervision for both speech-content and lyric-content editing.

Speech Content Editing. Starting from high-quality transcribed speech, we use a large language model (LLM) to generate operator-specific annotations and target transcripts for insertion, deletion, and substitution. Each operation is applied independently, yielding examples with an explicit edit type and a well-defined target transcript.

We synthesize the target waveform through localized masked infilling. Qwen3-ForcedAligner [59] first provides word-level alignments between the source waveform and transcript, allowing each edited span to be mapped to its temporal interval. We mask only these intervals and condition F5-TTS [6] on the masked source waveform and target transcript to generate the requested content. This construction modifies only the designated region while preserving speaker identity, prosody, and acoustic context elsewhere. We transcribe each synthesized waveform and compare it with the target transcript for quality control, and retain only samples with low word error rate.

Lyric Editing. We construct lyric-editing data from high-quality dry-vocal recordings. Each recording is transcribed and aligned to its lyrics at the word level, after which an LLM generates source–target lyric pairs for localized edits. Chinese replacements preserve the number of characters and are checked at the pinyin level, whereas English replacements respect complete word boundaries and preserve the number of words.

YingMusic-Singer-Plus [20] then synthesizes the target vocal. We mask only the latent interval associated with the edited lyrics and condition generation on the complete target lyrics and original melody, preserving the singer’s timbre, rhythm, expression, and surrounding acoustic details. As in speech-content editing, we transcribe each synthesized vocal and retain only samples with low word error rate.

## 2.5 Enhancement and Separation

Enhancement and separation train the model to transform a complex acoustic scene according to a natural-language request. Given the same mixture, the instruction determines which components should be preserved, removed, isolated, or restored, providing unified supervision for speech enhancement, source extraction, source removal, and selective editing.

Speech Enhancement. We construct speech-enhancement examples by applying independently sampled degradations to quality-filtered speech. Candidate non-speech recordings are first transcribed with an ASR system, and clips containing intelligible words are discarded before the remaining audio is mixed as sustained background noise or localized acoustic events. Reverberation is introduced using both measured room impulse responses and simulated rooms with randomized geometry, reverberation time, source locations, and microphone locations. We additionally apply channel degradations, including bandwidth limitation, clipping, signal dropout, telephone and megaphone coloration, underwater-like filtering, and DC offset. Randomizing the type, severity, and combination of these degradations prevents the model from associating an instruction with a single acoustic signature.

Training targets are not limited to fully clean speech. In addition to recovering the original signal, we create selective targets that remove only the corruption named by the instruction. For example, a denoising target may retain reverberation, a dereverberation target may retain environmental sound, and a channel-restoration target may preserve all unrelated scene attributes. The model therefore learns that enhancement is instruction-dependent: a component removed for one request may be intentionally retained for another.

Multi-Speaker Separation. We construct conversational mixtures by arranging multiple speakers on a shared timeline with turn-taking, pauses, interruptions, and partial or complete overlap. Speaker gain, temporal placement, room acoustics, and optional background interference are varied independently, producing scenes that better resemble natural conversations than simple waveform addition.

Natural-language instructions identify the desired speakers through complementary cues, including spoken content, speaking order, relative loudness, or an exclusive timestamp. An instruction may retain or remove one speaker or a subset of speakers. We also construct targets that remove a speaker while preserving the scene’s noise, reverberation, and channel effects. Solving these examples requires the model to interpret the request, locate the relevant source, and preserve all components that are not explicitly targeted.

Music Enhancement and Separation. For music data, we first obtain aligned vocal and accompaniment stems using a source-separation model. The accompaniment is transcribed and compared with the vocal transcript or lyrics; songs with substantial textual overlap are rejected to reduce residual singing in the accompaniment stem. We then construct two complementary forms of supervision. Native-song examples use the original song as input and time-aligned stem crops as targets, avoiding artifacts that would arise from reconstructing the input from separated tracks. Scene-based examples combine speech, singing voices, and background music with controlled timing and gain, creating mixtures such as speech over music, speech mixed with singing, and multiple overlapping singers.

![](images/7af00260d1cd0da8dfec696978a6354eb0c263d92506fd54582365c573699a84.jpg)  
Figure 4: Architecture of AuK. (a) The framework maps a user instruction and optional input audio to complementary semantic and acoustic conditioning streams. A multimodal language model encodes the instruction and audio context, and a learnable weighted sum of its layer-wise hidden states forms the semantic condition. In parallel, the VAE maps input audio, when present, to reference latents, which are combined with noisy target latents to form the acoustic condition. The two streams exchange information through M dual-stream MMDiT blocks before being concatenated and refined by N single-stream DiT blocks. The predicted latent is decoded by the VAE to produce the output audio. (b) Each MMDiT block performs joint attention over semantic and acoustic tokens while preserving stream-specific residual pathways. (c) Each DiT block applies self-attention to the fused token sequence. The symbols + and C denote addition and concatenation, respectively; snowflakes denote frozen modules.

The resulting instructions request spoken speech, a singing voice, a group of vocal sources, or a singer identified by order or timestamp. This design casts music processing as the same instructionconditioned source-selection problem used for conversational separation while retaining the acoustic complexity of real songs.

## 3 Model Design

## 3.1 Overall Architecture

Figure 4 shows the complete architecture of AuK. It consists of three components with complementary roles: the MLLM jointly processes textual instruction and audio context to produce the multimodal semantic condition; the pre-trained Variational Autoencoder (VAE) encodes audio into a latent space that preserves fine-grained acoustic information; and a FLUX-style[32] transformer backbone promotes interaction and fusion between the semantic and acoustic conditions while predicting the target acoustic latent.

During training, we group tasks into two categories based on input audio availability. (1) Tasks with reference audio (e.g., zero-shot TTS, content editing, speech enhancement and separation). The input audio is sent to both the audio encoder of an MLLM and the VAE encoder. The audio encoder supplies audio representations to the LLM, while the textual user instruction is provided directly to the LLM. In parallel, the VAE encoder converts the same input audio into reference acoustic latents. The semantic condition, reference acoustic latents and noisy target latents are concatenated along the sequence dimension. (2) Tasks without reference audio (e.g., Instruct TTS). The user instruction is processed directly by the LLM; no audio is sent to the audio encoder or the reference branch of the VAE encoder. In this case, the acoustic stream contains only the noisy target latents. After M dual-stream blocks, the semantic and acoustic streams are concatenated along the sequence dimension and refined by N single-stream blocks. In both configurations, the hybrid transformer predicts the denoised latent, which is converted to the audio by the VAE decoder.

## 3.2 MLLM Semantic Condition

A fixed output layer of a multimodal language model may not provide optimal conditioning across diverse audio generation and editing tasks, since representations at different depths capture complementary linguistic, acoustic, and cross-modal cues. To retain this information, AuK uses Qwen2.5- Omni [69] as its semantic encoder and aggregates its layer-wise hidden states.

Let t denote the tokenized user instruction and $\mathcal { E } _ { \mathrm { a u d } } ( \mathbf { x } _ { \mathrm { r e f } } )$ denote the audio encoder extracted from an optional reference waveform. The hidden state at MLLM layer ℓ is

$$
\begin{array} { r } { \mathbf { h } ^ { ( \ell ) } = \left\{ \begin{array} { l l } { \mathrm { M L L M } ^ { ( \ell ) } ( \mathbf { t } , { \mathcal E } _ { \mathrm { a u d } } ( { \mathbf x } _ { \mathrm { r e f } } ) ) , } & { \mathrm { w i t h ~ r e f e r e n c e ~ a u d i o , } } \\ { \mathrm { M L L M } ^ { ( \ell ) } ( \mathbf { t } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{1}
$$

Thus, audio-conditioned tasks encode the instruction and reference audio jointly, whereas text-only tasks derive their semantic representation solely from the instruction.

The layer-wise representations are aggregated into the semantic condition as

$$
\mathbf { c } _ { \mathrm { s e m } } = \sum _ { \ell = 1 } ^ { L } w _ { \ell } \cdot \mathrm { L a y e r N o r m } \Big ( \mathbf { h } ^ { ( \ell ) } \Big ) ,\tag{2}
$$

where L is the number of MLLM layers and $w _ { \ell }$ is an unconstrained learnable scalar for layer ℓ. Layer normalization [2] balances the scale of representations across layers before aggregation. The resulting $\mathbf { c } _ { \mathrm { s e m } }$ combines information from multiple levels of abstraction and is used as the semantic condition of the generative backbone.

## 3.3 VAE Acoustic Condition and Reconstruction

We employ a flow-augmented audio VAE [77] to provide a shared latent space for reference-audio conditioning and waveform reconstruction. Its non-causal encoder $\mathcal { E } _ { \mathrm { e n c } }$ maps a 24 kHz waveform to a sequence of 64-dimensional latents at 50 Hz, while its causal decoder $\mathcal { E } _ { \mathrm { d e c } }$ maps a latent sequence back to audio. A normalizing flow regularizes the latent distribution during VAE training.

For a reference waveform $\mathbf { x } _ { \mathrm { r e f } }$ , the encoder produces posterior parameters and a latent sample

$$
\begin{array} { r } { ( \mu _ { \mathrm { r e f } } , \log \sigma _ { \mathrm { r e f } } ) = \mathcal { E } _ { \mathrm { e n c } } \big ( \mathbf { x } _ { \mathrm { r e f } } \big ) , \qquad \mathbf { z } _ { \mathrm { r e f } } = \mu _ { \mathrm { r e f } } + \epsilon \odot \sigma _ { \mathrm { r e f } } , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}\tag{3}
$$

The acoustic condition is the VAE-encoder latent itself:

$$
\mathbf { c } _ { \mathrm { a c } } = { \left\{ \begin{array} { l l } { \mathbf { z } _ { \mathrm { r e f } } , } & { { \mathrm { i f ~ r e f e r e n c e ~ a u d i o ~ i s ~ a v a i l a b l e } } , } \\ { \varnothing , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{4}
$$

Here, $\mathbf { c } _ { \mathrm { a c } }$ is the unprojected reference latent in the VAE space and is projected into the Transformer hidden dimension only when passed to the backbone, as described in Eq. (5). During VAE training, the encoder–decoder pair is optimized to reconstruct the input waveform. During generation, the causal decoder $\mathcal { E } _ { \mathrm { d e c } }$ converts the resulting target latent into the output waveform. Detailed descriptions of the VAE architecture and training objective are provided in Section 4.1.

## 3.4 Transformer Backbone

The Transformer backbone of AuK follows a FLUX-style [32] hybrid design. Its first M layers are dual-stream MMDiT blocks [14], and the following N layers are single-stream DiT blocks [50]. A sinusoidal embedding of the flow time $t \sim \mathcal { U } [ 0 , 1 ]$ is projected by an MLP and used to modulate every block.

Condition Mapping. The semantic condition, optional acoustic condition, and noisy target latent are mapped to the two input streams as

$$
\begin{array} { r } { \mathbf { s } ^ { ( 0 ) } = \mathcal { P } _ { \mathrm { s e m } } ( \mathbf { c } _ { \mathrm { s e m } } ) , \qquad \mathbf { a } ^ { ( 0 ) } = \left[ \mathcal { P } _ { \mathrm { r e f } } ( \mathbf { c } _ { \mathrm { a c } } ) ; \mathcal { P } _ { \mathrm { t g t } } ( \mathbf { z } _ { t } ) \right] . } \end{array}\tag{5}
$$

Here, $\mathbf { c } _ { \mathrm { s e m } }$ is the MLLM semantic condition, $\mathbf { c } _ { \mathrm { a c } }$ is the optional VAE reference latent, and $\mathbf { z } _ { t }$ is the noisy target latent at flow time t. The operator $\mathcal { P } _ { \mathrm { s e m } }$ consists of a linear projection and RMSNorm, whereas ${ \mathcal { P } } _ { \mathrm { r e f } }$ and $\mathcal { P } _ { \mathrm { t g t } }$ each consist of the convolutional positional embedding and linear projection shown in Figure 4.

Dual-Stream MMDiT Block. The first M blocks jointly update the semantic and acoustic streams:

$$
\left( { \bf s } ^ { ( { \cal M } ) } , { \bf a } ^ { ( { \cal M } ) } \right) = \mathrm { M M D i T } ^ { { \cal M } } \left( { \bf s } ^ { ( 0 ) } , { \bf a } ^ { ( 0 ) } ; { \bf e } _ { t } \right) .\tag{6}
$$

Here, $\mathbf { e } _ { t }$ is the projected flow-time embedding, $\mathrm { M M D i T } ^ { M }$ denotes the stack of M MMDiT blocks, and $\mathbf { s } ^ { ( M ) }$ and $\mathbf { a } ^ { ( M ) }$ are its semantic and acoustic outputs. Each block uses stream-specific query, key, value, and residual projections. RoPE [61] is applied according to the positions of each stream before their attention tensors are concatenated for joint attention, enabling bidirectional semantic–acoustic interaction while preserving separate residual pathways.

Single-Stream DiT Block. The updated streams are concatenated and processed by the subsequent N DiT blocks to predict the flow velocity:

$$
\widehat { \mathbf { v } } _ { t } = \mathcal { P } _ { \mathrm { o u t } } \left( \mathrm { D i T } ^ { N } \left( [ \mathbf { s } ^ { ( M ) } ; \mathbf { a } ^ { ( M ) } ] ; \mathbf { e } _ { t } \right) _ { \mathrm { t g t } } \right) .\tag{7}
$$

Here, $\mathrm { D i T } ^ { N }$ denotes the stack of N DiT blocks, the subscript tgt selects the target-latent positions from its output, $\mathcal { P } _ { \mathrm { o u t } }$ is the final output projection, and $\widehat { \mathbf { v } } _ { t }$ is the predicted flow velocity. Both MMDiT and DiT blocks apply RMSNorm-based [76] QK-Norm before RoPE and use zero-initialized, timeconditioned AdaLN [50] to modulate their attention and SwiGLU feed-forward networks [57].

## 4 Model Training

## 4.1 VAE Training

The AuK-VAE is trained to provide a compact latent space that supports both reference-audio conditioning and high-fidelity waveform reconstruction. We describe its architecture, training data, and optimization objective below.

Architecture Configuration. The AuK-VAE operates on 24 kHz mono waveforms. The encoder first projects the waveform to 12 channels with a convolution of kernel size 3, followed by six downsampling blocks with strides (2, 2, 2, 3, 4, 5) and channel widths $1 2  2 4  4 8  \bar { 9 6 } $ $1 9 2  3 8 4  7 6 8$ . Each block contains a strided convolution whose kernel size is twice its stride, six residual units with kernel size 3 and dilation rates (1, 2, 4, 8, 16, 32), and a final LeakyReLU activation. A final convolution with kernel size 3 produces 128 channels, which parameterize a 64-dimensional posterior mean and log standard deviation. The overall downsampling factor is 480, yielding a latent frame rate of 50 Hz. All encoder convolutions use weight normalization.

A normalizing flow ${ \mathcal F } ,$ , composed of four residual coupling layers interleaved with channel flips, maps the posterior sample z to ${ \bf z } _ { p } = \mathcal { F } ( { \bf z } )$ for latent regularization. The flow is used only during VAE training; acoustic conditioning and waveform reconstruction operate in the original latent space z.

The decoder follows BigVGAN [34]. A 3-frame look-ahead convolution with kernel size 7 projects the latent representation to 1536 channels, after which all convolutions are causal. Six transposedconvolution blocks use strides (5, 4, 3, 2, 2, 2) and kernels twice their strides while reducing the channel width as $1 5 3 6 \to 7 6 8 \to 3 8 4 \to 1 9 2 \to 9 6 \to 4 8 \to 2 4$ . Each block is followed by an anti-aliased multi-periodicity composition module with residual branches of kernel sizes $\{ 3 , 7 , 1 1 \}$ and dilation rates {1, 3, 5}. The decoder uses channel-wise SnakeBeta activations and a final causal convolution with kernel size 7 to reconstruct the waveform.

Training Data and Setup. We train the AuK-VAE for 1.24 million updates on approximately 3 million hours of speech, music, and general audio. The three domains are sampled at the instance level with a ratio of 6:3:1. All waveforms are resampled to 24 kHz and divided into 1.28-second segments.

Training Objectives. Following DAC [31], we use a multi-scale log-mel reconstruction loss $\mathcal { L } _ { \mathrm { m e l } }$ defined as the sum of $\ell _ { 1 }$ distances between the mel spectrograms of x and xb at multiple resolutions. We use window lengths {32, 64, 128, 256, 512, 1024, 2048}, hop sizes {8, 16, 32, 64, 128, 256, 512}, and mel-bin counts {5, 10, 20, 40, 80, 160, 320} to capture both short-time acoustic detail and longrange spectral structure.

To improve perceptual quality, we apply adversarial training with a multi-period discriminator [30] using periods {2, 3, 5, 7, 11} and a multi-scale CQT discriminator [18]. Both discriminator families use the least-squares GAN objective [45]; an additional $\ell _ { 1 }$ feature-matching loss ${ \mathcal { L } } _ { \mathrm { f e a t } }$ is computed from their intermediate representations. We regularize the transformed posterior $q _ { \phi } ( \mathbf { z } _ { p } \mid \mathbf { x } )$ toward a standard Gaussian:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } } = D _ { \mathrm { K L } } \big ( q _ { \phi } ( \mathbf { z } _ { p } \mid \mathbf { x } ) \big \| \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \big ) . } \end{array}\tag{8}
$$

The complete generator objective is

$$
{ \mathcal { L } } _ { \mathrm { V A E } } = \lambda _ { \mathrm { m e l } } { \mathcal { L } } _ { \mathrm { m e l } } + \lambda _ { \mathrm { a d v } } { \mathcal { L } } _ { \mathrm { a d v } } + \lambda _ { \mathrm { f e a t } } { \mathcal { L } } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { K L } } { \mathcal { L } } _ { \mathrm { K L } } ,\tag{9}
$$

where $\mathcal { L } _ { \mathrm { a d v } }$ is the generator-side least-squares GAN loss. We set $\lambda _ { \mathrm { m e l } } = 1 5 , \lambda _ { \mathrm { a d v } } = 1 , \lambda _ { \mathrm { f e a t } } = 2$ and $\lambda _ { \mathrm { K L } } = 5$ . The discriminators are optimized separately with their corresponding discriminator objective.

## 4.2 Unified Pre-Training

Unified pre-training follows a two-stage curriculum. The first stage establishes basic speechgeneration capability using only generation tasks, while the second stage jointly optimizes generation and editing with a shared rectified-flow objective, condition-dropout strategy, and task mixture. Throughout both stages, the MLLM semantic encoder and audio VAE remain frozen; only the Transformer backbone and layer-fusion parameters are updated.

Model Configuration. The backbone contains 30 Transformer layers: 10 dual-stream MMDiT blocks followed by 20 single-stream DiT blocks. Every block has a hidden dimension of 1536, 24 attention heads with a head dimension of 64, and a SwiGLU feed-forward network with an intermediate dimension of 3072. The backbone contains approximately 1.5 billion parameters and adopts the RoPE and convolutional positional embedding design used in F5-TTS [6].

Training Curriculum. In the first stage, we train exclusively on speech-generation tasks for 50k updates. This generation-only warm-start establishes stable text-to-speech alignment and basic synthesis quality before the model is exposed to heterogeneous editing objectives. The second stage initializes from this checkpoint and jointly trains on generation and editing tasks for a further 600k updates. The five-family sampling mixture reported below applies to this second stage.

Flow-Matching Objective. Let $\mathbf { z } _ { 1 }$ denote the clean target latent produced by the frozen VAE encoder and let $\mathbf { z } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ denote Gaussian noise. For a sampled flow time $t \in [ 0 , 1 ]$ , we construct the linear interpolation and target velocity as

$$
{ \bf z } _ { t } = ( 1 - t ) { \bf z } _ { 0 } + t { \bf z } _ { 1 } , \qquad { \bf v } _ { t } = { \bf z } _ { 1 } - { \bf z } _ { 0 } .\tag{10}
$$

The model predicts $\widehat { \mathbf { v } } _ { t }$ from $\mathbf { z } _ { t } .$ , the flow-time embedding, and the available semantic and acoustic conditions. Training minimizes a masked mean-squared error over valid, non-padding latent frames:

$$
\mathcal { L } _ { \mathrm { F M } } = \frac { \left. \mathbf { m } \odot ( \widehat { \mathbf { v } } _ { t } - \mathbf { v } _ { t } ) \right. _ { 2 } ^ { 2 } } { \sum _ { i } m _ { i } } ,\tag{11}
$$

where m is the validity mask. We sample $t = \sigma ( u )$ with $u \sim \mathcal { N } ( - 0 . 8 , 0 . 8 ^ { 2 } )$ , biasing training toward smaller t, corresponding to noisier states closer to the Gaussian-noise endpoint, while retaining support over the full trajectory. At inference, the learned velocity field is integrated with a deterministic ODE solver.

Table 1: Fixed per-batch sampling probabilities used in the second pre-training stage for the five task families introduced in Sec. 2.
<table><tr><td>Task family</td><td>Sampling probability</td></tr><tr><td>Speech Generation</td><td>28.10%</td></tr><tr><td>Content Editing</td><td>23.02%</td></tr><tr><td>Enhancement and Separation</td><td>23.17%</td></tr><tr><td>Paralinguistic Editing</td><td>21.75%</td></tr><tr><td>Acoustic Editing</td><td>3.96%</td></tr></table>

Condition Dropout and Reference Augmentation. To support classifier-free guidance for both text-only generation and reference-conditioned tasks, we apply hierarchical condition dropout. We first sample an acoustic-condition dropout mask with probability 0.3. We then independently sample an unconditional mask with probability 0.2; when active, this second mask overrides the first decision and drops both the acoustic and semantic conditions. The resulting training examples cover fully conditioned, text-only, and unconditional configurations under the same objective.

Reference audio at inference can vary substantially in duration for tasks like zero-shot TTS. To reduce sensitivity to the reference lengths observed during training, with probability 0.5 we crop zero-shot TTS references to a uniformly sampled duration between 3 seconds and its original length.

Task Mixture and Dynamic Batching. During the second stage, we realize the five task families in Sec. 2 as fixed per-batch sampling probabilities, summarized in Tab. 1. Utterances range from 1 to 35 seconds and are grouped with dynamic length-bucketed batching. Each accelerator processes at most 10,000 latent frames at 50 Hz or 24 utterances per micro-batch, whichever limit is reached first. The resulting global batch contains at most 6,144 utterances, corresponding to approximately 14 hours of audio per optimizer step across GPUs.

Optimization. We train on 256 GPUs with DeepSpeed ZeRO-2 and bf16 precision, without CPU offload. Gradients are clipped to a global norm of 1.0. We use fused AdamW with $\beta = ( 0 . 9 , 0 . 9 5 )$ and a peak learning rate of $\mathrm { \bar { 1 } \times 1 0 ^ { - \bar { 4 } } }$ , linearly warmed up over the first 2,000 updates and then held constant. An exponential moving average of the model weights with decay 0.9999 is enabled after 100 updates and refreshed every 10 updates; the EMA checkpoint is used for evaluation and release.

## 4.3 Post-Training

Post-training consists of two complementary stages. First, editing preference optimization uses human feedback to align diverse editing behaviors with subjective judgments of task completion and perceptual quality. Second, generation reinforcement learning uses automatic reward models to improve zero-shot and instruction-following speech generation. Both stages update only the Transformer backbone; the Qwen2.5-Omni encoder and audio VAE remain frozen.

## 4.3.1 Editing Preference Optimization

Human Preference Data. Editing tasks are judged by more subjective criteria than generation tasks. Whether a result is acceptable often depends on hard-to-quantify judgments of naturalness, degree of transformation, and contextual appropriateness, unlike the content correctness and speaker similarity in speech generation, which can be reliably measured by a single automatic metric. Moreover, for the diverse set of editing tasks we cover, neither a sufficiently broad preference dataset nor a ready-made reward model exists. We therefore collect human feedback directly and optimize editing behavior with preference learning.

For each editing instruction, we sample 10 or 20 candidate outputs from the pre-trained model. Annotators rate each candidate on a three-level ordinal scale $s _ { i } \in \{ 0 , 1 , 2 \}$ , corresponding to failure, partial completion, and successful completion. Groups in which all 20 candidates receive the same rating are discarded because they provide no preference signal. After filtering the dataset contains 818 informative groups and 9,080 rated candidates.

Flow-Based DPO Score. Following Diffusion-DPO [66], we define an implicit preference score from the improvement of the policy over a frozen reference model in flow-matching error. Although candidate waveforms are generated from independent noise trajectories, the DPO loss reevaluates candidates within a group using a shared noise sample $\mathbf { z } _ { 0 }$ and flow time $t ,$ isolating differences attributable to the candidate target latents. The frozen-reference prediction is cached within each micro-batch.

Ordinal Listwise Objective. We optimize the ordinal feedback with the listwise formulation of LiPO [42], where $R _ { i } ^ { \mathrm { p r e f } }$ is the implicit preference score defined above. Let $\mathcal { T } _ { q } ^ { + }$ and $\mathcal { T } _ { q } ^ { - }$ denote the higher- and lower-rated candidate sets for relation $q \in \{ 1 \succ 0 , 2 \succ 0 , 2 \succ \overset { \cdot } { 1 } \}$ , and let $\Delta R _ { i j } =$ $R _ { i } ^ { \mathrm { p r e f } } - R _ { i } ^ { \mathrm { p r e f } }$ for a pair $( i , j )$ . Each preference pair is penalized by a label-smoothed logistic loss $( \epsilon = 0 . 0 5 )$ , reduced in two levels: first averaged within each relation, then combined across relations with confidence weights:

$$
\mathcal { L } _ { q } = \frac { 1 } { | \mathcal { I } _ { q } ^ { + } | | \mathcal { Z } _ { q } ^ { - } | } \sum _ { i \in \mathcal { I } _ { q } ^ { + } } \sum _ { j \in \mathcal { I } _ { q } ^ { - } } \ell ( \Delta R _ { i j } ) , \qquad \mathcal { L } _ { g } = \frac { \sum _ { q \in A _ { g } } \omega _ { q } \mathcal { L } _ { q } } { \sum _ { q \in A _ { g } } \omega _ { q } } ,\tag{12}
$$

where $\mathcal { A } _ { g }$ is the set of rating relations present in group $g , \omega _ { 2 \succ 0 } = 1$ , and the other relation weights are 0.5. This two-level reduction makes the contribution of a group depend only on $\omega _ { q } .$ , invariant to the number of preference pairs and to the group size. Both the policy and frozen reference are initialized from the unified pre-training checkpoint. We train this stage for 104 optimizer updates and take this as the end point of the stage.

## 4.3.2 Generation Reinforcement Learning

Generation reinforcement learning improves two task families with different reward structures. For zero-shot TTS, we optimize content correctness and speaker similarity using off-the-shelf ASR and speaker-verification models. For instruction TTS, we optimize agreement with the requested speaking style using a dedicated style-consistency model. This second stage also mitigates the generation-quality trade-off introduced by editing-oriented preference optimization.

Prompt Construction. We construct a frozen pool of 10,000 zero-shot prompts from the training set, with 5,000 Chinese and 5,000 English examples. References are stratified into duration ranges of 1–4 s, 4–8 s, and 8–15 s. Following FlowTTS-GRPO [67], we augment the pool with hard texts containing local word repetition, sparse multi-word repetition, or whole-sentence repetition, each paired with the original reference waveform. For instruction-following TTS, we categorize the instructions by their controllable dimensions and assess the base model’s per-dimension style consistency. Guided by this diagnosis, we deliberately over-sample the dimensions on which the model is weakest, and curate an additional pool of 5,000 instruction-following prompts so that optimization pressure concentrates where the model is least reliable.

Flow-GRPO Optimization. Following Flow-GRPO [40], we reformulate sampling as an SDE with the same marginal distributions, yielding Gaussian transition kernels along the denoising trajectory. Candidates in a GRPO group are generated from different noise trajectories, providing the exploration required for relative policy optimization. Following MixGRPO [36], stochastic sampling and gradient computation are restricted to a contiguous six-step window in the low-SNR portion of the first half of the trajectory; the remaining steps use deterministic ODE updates to limit variance and computational cost. The policy objective follows the clipped form of GRPO [56], augmented with a KL penalty of weight $\beta _ { \mathrm { K L } }$ toward the frozen reference policy.

Reward Design. For zero-shot TTS, content correctness is judged by off-the-shelf speech recognition models. Errors caused by Chinese homophones cannot be removed by reinforcement learning, so we distinguish a tolerant error rate $E$ from a strict error rate $E ^ { \mathrm { s } }$ . A homophone substitution does not indicate a pronunciation failure, so the tolerant rate decides whether a candidate counts as fully correct, while the strict rate provides a finer-grained ranking signal among candidates that already pass this check, and is fused with the negative log-likelihood of the target text into a content score $\bar { R _ { i } ^ { \mathrm { c o n t e n t } } }$ . Let $\gamma \in [ 0 , 1 ]$ denote the interpolation weight between the two terms and $\tau > 0$ a temperature controlling the sensitivity of the error term:

$$
R _ { i } ^ { \mathrm { c o n t e n t } } = ( 1 - \gamma ) \Big ( 1 + 4 e ^ { - E _ { i } ^ { \mathrm { s } } / \tau } \Big ) + \gamma \Big ( 1 + 4 e ^ { - \mathrm { N L L } _ { i } } \Big ) .\tag{13}
$$

Speaker similarity is measured as the cosine similarity between the speaker embeddings of the candidate and reference audio, where $\psi ( \cdot )$ denotes the speaker encoder:

$$
R _ { i } ^ { \mathrm { s p k } } = \cos { \left( \psi ( y _ { i } ) , \psi ( y _ { \mathrm { r e f } } \right) } ) .\tag{14}
$$

To prevent high speaker similarity from compensating for incorrect content, $R _ { i } ^ { \mathrm { { c o n t e n t } } }$ is standardized over the entire candidate group, whereas $R _ { i } ^ { \mathrm { { s p k } } }$ is standardized only among candidates satisfying $E _ { i } = 0$ . This constrained construction closes a common reward-hacking path in weighted reward fusion [67, 81].

For instruction TTS, a dedicated style-consistency model judges whether a candidate matches the requested attributes. We collect 30,000 balanced 1:1 positive/negative samples as supervision, and train a multimodal reward model built based on Qwen2.5-Omni-7B [69] to reproduce styleconsistency judgement. In practice, we query the judge V times and use majority voting:

$$
R _ { i } ^ { \mathrm { s t y l e } } = \nVdash \left[ \sum _ { v = 1 } ^ { V } \nVdash [ \mathrm { j u d g e } _ { v } ( \mathbf y _ { i } ) = \mathrm { c o n s i s t e n t } ] > V / 2 \right] .\tag{15}
$$

The style-task advantage is the standard group-wise z-score used in GRPO [56].

Optimization. We initialize the policy from the editing-preference checkpoint, sample $G = 1 6$ candidates per prompt, and train for 500 optimizer updates. Rollouts use 500 sampling steps, classifier-free guidance of 2.0, a sway coefficient of −1.0, and $\lambda = 0 . 7$ . We optimize with AdamW $( \beta = ( 0 . 9 , 0 . 9 5 )$ , weight decay $1 0 ^ { - 4 } )$ , a learning rate of $5 \times 1 0 ^ { - 5 }$ , and gradient clipping at 1.0. The KL coefficient is initialized to $\beta _ { \mathrm { K L } } = 0 . 1 2$ and adjusted proportionally toward a target KL of $\mathcal { D } ^ { \star } = 1 0 ^ { - 3 }$ within the interval $[ 0 . 0 8 , 0 . 5 ]$

## 5 Model Acceleration

The full post-trained model requires iterative flow sampling and classifier-free guidance (CFG)[21], making inference expensive across its broad generation and editing capabilities. We therefore distill it into a four-step, CFG-free student through two stages: consistency initialization provides a stable few-step starting point, and task-routed Decoupled DMD improves distribution matching while preserving separation ability. Both stages use data sampled from the same distribution as unified pre-training. The full post-trained model serves as the teacher throughout acceleration.

Consistency Initialization. Following the common practice in few-step video diffusion distillation of initializing the student before distribution matching stage [82, 84, 35, 74], we initialize the student from the teacher and perform trajectory-level consistency distillation[60] under teacher guidance. The student is trained to map any noisy state $\mathbf { z } _ { t }$ directly to the endpoint $\mathbf { z } _ { 1 }$ of its sampling trajectory: for neighboring timesteps $\bar { t } < t ^ { \prime } .$ , the frozen teacher first advances $\mathbf { z } _ { t }$ to $\mathbf { z } _ { t ^ { \prime } }$ with a single CFGguided velocity step, and the student’s predictions of $\mathbf { z } _ { 1 }$ from $\mathbf { z } _ { t }$ and from $\mathbf { z } _ { t ^ { \prime } }$ are matched, with a stop-gradient on the latter so that gradients flow only through the prediction at t. This stage adapts the student to few-step flow integration, yielding a stable initialization with preliminary four-step generation and editing capability. We also compared other trajectory-level distillation for initialization, including the ODE regression used in CausVid [74] and MeanFlow[17]. At matched update counts, consistency initialization consistently performs best. We therefore keep consistency distillation.

Decoupled DMD with APG. The second stage adopts Decoupled DMD [37], which has been validated at scale in Z-Image [75]. Decoupled DMD separates the student update into two complementary components. CFG Augmentation (CA) transfers the teacher’s conditional guidance to the student, whereas Distribution Matching (DM) aligns the student distribution with the teacher distribution. The two components are evaluated on independently re-noised student predictions so that guidance transfer and distribution matching are not tied to the same noise level.

The consistency checkpoint initializes both the student and the fake-score model, while the frozen post-trained teacher acts as the real-score model. We observe that directly using teacher CFG targets in the CA branch can expose the few-step student to over-saturated predictions, causing overshoot and audible clipping. We therefore replace CFG in this branch with adaptive projected guidance (APG) [55], which suppresses excessive guidance components while retaining instruction adherence.

Task-Routed Decoupled DMD. We empirically observe that applying Decoupled DMD uniformly across tasks degrades multi-speaker and vocal separation, with some student outputs regressing toward the unprocessed mixture. We attribute this behavior to a distribution mismatch: re-noised erroneous separation outputs can fall outside the teacher’s training distribution, causing the teacher field to favor globally plausible but insufficiently separated audio. To preserve separation ability, we route separation examples to a supervised clean-prediction objective and exclude them from both sides of the DMD update.

For sample i with task label $\tau _ { i } ,$ let $\widehat { \mathbf { z } } _ { 1 , i }$ denote the student’s clean-endpoint prediction and $\mathbf { z } _ { 1 , i }$ the ground-truth clean latent. The routed objectives are

$$
\begin{array} { r } { \left( \mathcal { L } _ { \theta } ^ { ( i ) } , \mathcal { L } _ { \phi } ^ { ( i ) } \right) = \left\{ \begin{array} { l l } { \left( \left| \left| \widehat { \mathbf { z } } _ { 1 , i } - \mathbf { z } _ { 1 , i } \right| \right| _ { 2 } ^ { 2 } , 0 \right) , } & { \tau _ { i } \in \mathcal { S } , } \\ { \left( \mathcal { L } _ { \mathrm { D M D } } ^ { ( i ) } , \mathcal { L } _ { \mathrm { f a k e } } ^ { ( i ) } \right) , } & { \tau _ { i } \notin \mathcal { S } , } \end{array} \right. } \end{array}\tag{16}
$$

where $\mathcal { L } _ { \theta } ^ { ( i ) }$ updates the student and $\mathcal { L } _ { \phi } ^ { ( i ) }$ updates the fake-score model. For routed separation examples, clean-latent regression updates only the student, while the zero in the paired objective excludes the sample from fake-score training. All remaining examples use the Decoupled DMD student loss $\mathcal { L } _ { \mathrm { D M D } } ^ { ( i ) }$ and the online flow-matching loss $\mathcal { L } _ { \mathrm { f a k e } } ^ { ( i ) }$ , which keeps the fake-score model aligned with the evolving student distribution.

Training Configuration. Both stages use the same data distribution and batch size as unified pre-training. We train on 256 GPUs with fused AdamW, a constant learning rate of $1 0 ^ { - 5 }$ , and global gradient clipping at 1.0. Consistency initialization runs for 500 updates; the student is initialized from the teacher, and teacher targets use CFG with guidance scale 2.0. The second stage initializes the student and fake-score model from the consistency checkpoint and freezes the teacher as the real-score model. We perform 2,500 student updates and 5 fake-score updates for every student update. The CA branch uses APG with guidance scale 4.0 and $\eta = 0$ . No additional adversarial discriminator or adversarial loss is introduced.

## 6 Model Inference

The inference pipeline contains two stages. For free-form user requests, a Prompt Enhancer (PE) identifies the task, rewrites the request into a model-oriented instruction, prepares optional input audio, and estimates the output duration. The resulting instruction and optional audio context are then passed to AuK for conditional flow sampling. Canonical instructions that already follow the supported format can bypass PE and be provided directly to the model.

## 6.1 Prompt Enhancer

Free-form requests can vary substantially in wording, omit required parameters, or refer implicitly to the input audio. PE provides a task-aware interface that converts these requests into explicit instructions closer to the training distribution while preserving all user-specified content.

Task Routing and Instruction Rewriting. Given a request q and optional input audio a, PE obtains an ASR transcript and language prediction when audio is available. The request, audio, and ASR context are then processed by a capable language model [64, 9, 48] to identify the target task and extract its arguments and control parameters. Audio-derived context helps resolve implicit references, distinguish operations with similar wording, and support duration estimation.

The extracted parameters are validated against the ranges represented during training. Colloquial or continuous descriptions of speaking rate, loudness, and pitch are mapped to supported discrete values, while invalid or unsupported requests are rejected before acoustic inference. PE then renders the validated request using a task-specific template or a curated instruction formulation. User-provided synthesis text and replacement content are preserved, while task descriptions may be lightly rewritten or expanded. Text normalization is applied when required for speech generation.

Audio Preparation and Duration Estimation. For audio-conditioned tasks, PE applies lightweight, task-dependent preprocessing, including leading- and trailing-silence handling, resampling, and simple level normalization when the input distribution differs from that used during training. These operations standardize the model input without altering content that should be preserved. PE also estimates the target duration before flow sampling. Let $T _ { \mathrm { i n } }$ be the input-audio duration, let B(x) denote the UTF-8 byte length of text x, following the duration heuristic used in F5-TTS [6]. The task-dependent estimate is

$$
\begin{array} { r } { T _ { \mathrm { o u t } } = \{ \begin{array} { l l } { T _ { \mathrm { i n } } \frac { B ( x _ { \mathrm { t a r g e t } } ) } { B ( x _ { \mathrm { s o u r c e } } ) } , } & { z \mathrm { e r o - s h o t } \mathrm { T T S ~ a n d ~ c o n t e n t ~ e d i t i n } } \\ { \frac { T _ { \mathrm { i n } } } { s } , } & { \mathrm { s p e e d ~ e d i t i n g ~ w i t h ~ m u l t i p l i e r ~ } s , } \\ { \kappa _ { e } T _ { \mathrm { i n } } , } & { \mathrm { e m o t i o n ~ e d i t i n g , } } \\ { T _ { \mathrm { i n } } + \Delta _ { \mathrm { n v } } , } & { \mathrm { n o n v e r b a l ~ e d i t i n g , } } \\ { ( T _ { \mathrm { L I M } } ( \widehat { T } _ { \mathrm { b a s e } } , d _ { \mathrm { s t y l e } } , x _ { \mathrm { t a r g e t } } ) , } & { \mathrm { i n s t r u c t ~ T T S , } } \\ { T _ { \mathrm { i n , } } } & { \mathrm { d u r a t i o n - p r e s e r v i n g ~ o p e r a t i o n s } . } \end{array}  } \end{array}\tag{17}
$$

The emotion factor $\kappa _ { e }$ and nonverbal offset $\Delta _ { \mathrm { n v } }$ are estimated from the corresponding training data. For instruct TTS, the base estimate $\widehat { T } _ { \mathrm { b a s e } } = \rho _ { \ell } B ( x _ { \mathrm { t a r g e t } } )$ uses a language-dependent byte-rate coefficient $\rho _ { \ell } ;$ the language model then adjusts this estimate using the requested style and output text. For zero-shot $\mathrm { T T S } , x _ { \mathrm { s o u r c e } }$ is obtained internally by transcribing the reference audio, so the user is not required to provide a reference transcript. For speech and lyric content editing, the duration is scaled using the UTF-8 byte-length ratio between the target and source content.

## 6.2 Inference Strategy and Configuration

When input audio is provided, it is converted to mono, resampled to 24 kHz, and encoded by the VAE as an acoustic reference. In parallel, the MLLM jointly processes the enhanced instruction and input audio to construct the semantic condition. For text-only generation, the semantic condition is derived from the instruction alone and no acoustic reference is used. The estimated duration determines the target latent length at the VAE rate of 50 Hz. Sampling starts from Gaussian noise of this length, and the generated latent is decoded by the VAE into the output waveform.

All evaluation-time settings for AuK and AuK-Flash are consolidated in Tab. 2. Both variants use the same prompt enhancement, audio preparation, duration estimation, latent representation, and VAE decoder. The full model uses its EMA checkpoint with classifier-free guidance, whereas the Flash variant uses the task-routed distilled checkpoint and requires no guidance.

Table 2: Inference configurations for AuK and AuK-Flash.
<table><tr><td>Configuration</td><td>AuK</td><td>AuK-Flash</td></tr><tr><td>Checkpoint</td><td>EMA post-trained </td><td>Task-routed distilled</td></tr><tr><td>Numerical precision</td><td></td><td>bfloat16</td></tr><tr><td>ODE solver</td><td>Euler</td><td></td></tr><tr><td>Function evaluations</td><td>32</td><td>4</td></tr><tr><td>CFG scale</td><td>2.0</td><td>None</td></tr><tr><td>Sway coefficient</td><td></td><td>-1.0</td></tr><tr><td>Output sample rate</td><td></td><td>24 kHz</td></tr><tr><td>VAE latent space</td><td></td><td>64 dim, 50 Hz</td></tr><tr><td>Wall-clock speedup</td><td>1.0×</td><td>4.5×</td></tr></table>

The reported 4.5× speedup compares the four-step, CFG-free AuK-Flash sampler against the 32-NFE, CFG-2.0 AuK sampler under the same hardware, output duration, and batch size. Unless otherwise stated, these configurations are used for the corresponding benchmark results.

## 7 Performance

We evaluate AuK and AuK-Flash from three complementary perspectives: reconstruction fidelity, speech generation, and speech editing. Tab. 3 provides a compact comparison with representative prior systems, while Sec. A reports additional metrics, operation-level results, and broader baseline sets where available. Across these evaluations, both variants achieve consistently strong performance in zero-shot and instruction-controlled generation, general speech editing, and signal-level speech processing. The full model generally provides stronger linguistic accuracy and edit fidelity, whereas the Flash variant retains competitive instruction following and often offers better perceptual quality under accelerated inference. We analyze VAE reconstruction, generation, and editing in Secs. 7.1 to 7.3, respectively.

Table 3: Summary of selected speech generation, editing, enhancement, and separation benchmarks. Each block retains its native metrics. The best result among the displayed systems in each column is shown in bold.
<table><tr><td>Task</td><td>Dataset</td><td>Prior SOTA</td><td>Metrics</td><td colspan="4">Results</td></tr><tr><td rowspan="2">Generation Ability</td><td rowspan="2">Seed-TTS-Eval en|zh|zh-hard|avg</td><td>Qwen3-TTS Seed-TTS VoxCPM2 AuK-Flash AuK</td><td>WER↓</td><td>1.231 2.251 1.841 1.031 1.02|</td><td>1.221 6.761 1.121 7.591 0.971 8.131 1.101 6.431 1.021 5.911</td><td>3.07</td><td>3.65 3.65 2.85 2.65</td></tr><tr><td>Qwen3-TTS Seed-TTS VoxCPM2 AuK-Flash</td><td>SIM↑</td><td colspan="4">0.717| 0.7701 0.7481 0.745 0.762|0.796|0.776|0.778 0.753|0.795|0.753|0.767 0.772| 0.814| 0.784| 0.790 0.788| 0.814| 0.782| 0.795</td></tr><tr><td></td><td>InstructTTSEval DSD-ZH\DSD-EN</td><td>Mimo-Audio MOSS-VoiceGenerator AuK-Flash AuK Step-Audio-EditX</td><td>ACC↑</td><td colspan="4">74.30177.60 80.00| 82.00 78.80| 82.40 83.37| 81.60 43.52|77.27| 4.69</td></tr><tr><td rowspan="6">General Speech Editing</td><td>MMAE(Speech)</td><td>Ming-UniAudio AuK-Flash AuK</td><td>IFR↑ICR↑IEMR↑</td><td colspan="3">34.13| 76.01| 7.04 46.62| 86.41|13.85 48.23| 88.11| 12.44 76.461 3.43|26.50|11.25|25.85</td><td rowspan="2"></td></tr><tr><td>SpeechEditBench Content | Emotion| Prosody | Paralinguistic|Acoustic</td><td>Ming-UniAudio Step-Audio-EditX AuK-Flash AuK</td><td>ACC↑</td><td colspan="3">16.501 7.71|20.13|31.25|22.89 87.501 6.29| 70.00| 39.25| 30.26 91.83| 9.94| 71.33| 38.50| 37.07</td></tr><tr><td rowspan="4">Semantic Editing Basic-ZH| Full-ZH Basic-EN|Full-EN</td><td></td><td>Ming-UniAudio AuK-Flash</td><td colspan="4">6.61| 10.46| 10.16| 14.28 WER↓</td></tr><tr><td>AuK Ming-UniAudio Ming-Freeform-Audio-Edit AuK-Flash</td><td rowspan="2">ACC↑</td><td colspan="4">86.21|79.62|71.14|70.98 92.01| 91.21| 84.00| 82.98 91.95| 91.47| 85.47| 85.25</td></tr><tr><td>AuK Ming-UniAudio AuK-Flash</td><td colspan="4">0.811 0.88| 0.88|</td></tr><tr><td>AuK Ming-UniAudio AuK-Flash</td><td colspan="4">no-edit WER↓</td></tr><tr><td rowspan="2">Ming-Freeform-Audio-Edit Acoustic Editing ZH\EN</td><td>AuK Ming-UniAudio AuK-Flash AuK</td><td rowspan="2">WER↓</td><td rowspan="2">0.631</td><td rowspan="2">5.01 | 10.75 2.40| 4.45 2.02| 3.48</td><td rowspan="2">3.11| 15.84| 15.99 0.54</td><td rowspan="2"></td></tr><tr><td>Ming-UniAudio AuK-Flash</td></tr><tr><td rowspan="5">CHiME-4 Speech Enhancement and Separation</td><td>DNS Challenge</td><td>AuK SAM-Audio-Large AnyEnhance RE-USE</td><td>SIM↑</td><td>3.291 3.781 3.411 3.761 3.381</td><td>0.781 0.74 9.671 7.711 3.691 3.311</td><td colspan="2">0.75 0.96 0.98 0.98 2.931 0.99</td></tr><tr><td></td><td>AuK-Flash AuK SAM-Audio-Large AnyEnhance RE-USE</td><td>DNSMOS-OVRL↑IUTMOS↑IWER↓</td><td>3.381 4.051 3.351 3.141 3.191 3.331</td><td>3.861 3.38117.30 3.24| 27.41</td><td>2.66| 3.441 10.71</td><td>0.99</td></tr><tr><td></td><td>AuK-Flash AuK MossFormer2-SS SAM-Audio-Large</td><td>DNSMOS-OVRL↑IUTMOS↑IWER↓ISIM↑</td><td>3.351 3.241 2.871</td><td>3.91| 7.84 3.28| 3.72| 7.98 3.661 9.341</td><td>2.84157.221</td><td>0.96 0.86 2.61|51.791 0.87</td></tr><tr><td></td><td>Sidon (Dialogue) AuK-Flash AuK AudioSR</td><td></td><td>3.32| 3.281 3.081</td><td>4.03| 10.071 3.871 3.131</td><td>9.12| 4.721</td><td>0.96 0.96 0.92</td></tr><tr><td>VCTKSR</td><td>Resemble-Enhance AuK-Flash AuK</td><td>DNSMOS-OVRL↑IUTMOS↑IWER↓ISIM↑</td><td>3.181 3.25| 3.211</td><td>3.59|15.941 4.051 3.931</td><td>2.92| 3.061</td><td>0.95 0.96 0.97</td></tr></table>

Table 4: Comparison of VAE reconstruction quality across speech, general audio, and music. The best and second-best results within each domain are shown in bold and underlined, respectively.
<table><tr><td>Domain</td><td>Test Set</td><td>Method</td><td>PESQ↑</td><td>STOI↑</td><td>Mel Dist↓</td><td>STFT↓</td></tr><tr><td rowspan="5">Speech</td><td rowspan="5">Seed-TTS-Eval</td><td>MiniMax-H3-AudioVAE</td><td>3.633</td><td>0.968</td><td>0.695</td><td>1.615</td></tr><tr><td>Ming-omni-tts</td><td>2.583</td><td>0.926</td><td>1.014</td><td>1.923</td></tr><tr><td>Stable-Audio-3-SAME-L</td><td>2.969</td><td>0.951</td><td>0.954</td><td>1.819</td></tr><tr><td>MMAudio-VAE</td><td>2.696</td><td>0.937</td><td>0.619</td><td>1.627</td></tr><tr><td>AuK-VAE</td><td>4.143</td><td>0.982</td><td>0.574</td><td>1.457</td></tr><tr><td rowspan="5">Audio</td><td rowspan="5">AudioSet</td><td>MiniMax-H3-AudioVAE</td><td>2.835</td><td>0.792</td><td>0.766</td><td>2.119</td></tr><tr><td>Ming-omni-tts</td><td>1.817</td><td>0.650</td><td>1.121</td><td>3.389</td></tr><tr><td>Stable-Audio-3-SAME-L</td><td>2.901</td><td>0.779</td><td>1.052</td><td>3.264</td></tr><tr><td>MMAudio-VAE</td><td>2.455</td><td>0.770</td><td>0.635</td><td>2.216</td></tr><tr><td>AuK-VAE</td><td>3.833</td><td>0.900</td><td>0.571</td><td>1.842</td></tr><tr><td rowspan="5">Music</td><td rowspan="5">MUSDB18-HQ</td><td>MiniMax-H3-AudioVAE</td><td>2.801</td><td>0.847</td><td>0.697</td><td>1.848</td></tr><tr><td>Ming-omni-tts Stable-Audio-3-SAME-L</td><td>1.634</td><td>0.708</td><td>0.941</td><td>2.315</td></tr><tr><td>MMAudio-VAE</td><td>3.051</td><td>0.876</td><td>0.701</td><td>1.795</td></tr><tr><td></td><td>1.753</td><td>0.775</td><td>0.661</td><td>1.826</td></tr><tr><td>AuK-VAE</td><td>3.884</td><td>0.927</td><td>0.525</td><td>1.721</td></tr></table>

## 7.1 VAE Reconstruction Results

We evaluate the reconstruction quality of AuK-VAE across three representative audio domains: speech, general audio, and music. Specifically, the evaluation uses Seed-TTS-Eval[1] for speech, 2,000 randomly sampled clips from AudioSet[16] for general audio, and the MUSDB18-HQ[52] test set for music. The compared models include MiniMax-H3-AudioVAE[46], Ming-Omni-TTS[27], Stable-Audio-3-SAME-L[49], and MMAudio-VAE[7]. Reconstruction fidelity is measured using PESQ, STOI, mel-spectrogram distance (Mel Dist), and multi-resolution STFT distance (STFT).

As shown in Tab. 4, AuK-VAE achieves the best result on all four metrics across all three domains. These results demonstrate that AuK-VAE consistently preserves perceptual quality, intelligibility, and spectral detail across speech, general audio, and music.

## 7.2 Generation Ability

We evaluate two complementary generation capabilities: zero-shot voice cloning on Seed-TTS-Eval [1] and instruction-controlled synthesis on InstructTTSEval [25]. Seed-TTS-Eval measures linguistic accuracy and speaker preservation across English, Chinese, and Chinese hard-text subsets, while InstructTTSEval evaluates control over acoustic parameters, descriptive styles, and role-playing instructions in both languages. For both benchmarks, inference uses only benchmark-specific instruction templates and output-duration estimation, without additional prompt enhancement. We run each evaluation three times and report the mean score.

Zero-Shot TTS. AuK achieves the lowest average recognition error and the highest average speaker similarity among the compared systems, with an average error of 2.65% and a SIM of 0.795. The comparison in Tab. 3 includes Qwen3-TTS [22], Seed-TTS [1], and VoxCPM2 [88], while Tab. 5 provides a broader baseline set. Compared with Qwen3-TTS, the strongest baseline in average recognition error, AuK reduces the error from 3.07% to 2.65%. It also improves the average SIM over Seed-TTS, the strongest baseline on this metric, from 0.778 to 0.795. AuK-Flash remains competitive, achieving an average recognition error of 2.85% and a SIM of 0.790. On test-en, AuK achieves the lowest WER of 1.02%; on test-zh-hard, it obtains the lowest recognition error of 5.91%, while AuK-Flash achieves the highest SIM of 0.784.

Instruct TTS. On the DSD split, AuK achieves the best Chinese accuracy of 83.37%, outperforming Qwen3-TTS-VD [22] by 2.27 percentage points. AuK-Flash achieves 82.40% on English DSD, tying Qwen3-TTS-VD for the best result among the systems summarized in Tab. 3. The complete results in Tab. 6, which also include Mimo-Audio [78] and MOSS-VoiceGenerator [24], show that the full model outperforms the Flash variant on all three Chinese metrics, whereas the Flash variant’s main advantage is its stronger English DSD result.

## 7.3 Editing Ability

We evaluate editing-related capabilities in two complementary regimes. The first tests whether a model can execute natural-language editing instructions while preserving attributes outside the requested change. The second regime evaluates signal-level processing using DNS Challenge 2020 [53] and CHiME-4 [5] for enhancement, Libri2Mix [8] for two-speaker separation, and VCTK-SR [71] for speech super-resolution.

## 7.3.1 General Speech Editing

We evaluate general speech editing on three complementary benchmarks. For MMAE [44], we use its speech subset to assess rubric-based instruction following and preservation. SpeechEditBench [79] evaluates joint success across five editing categories, while Ming-Freeform-Audio-Edit [72] provides operation-level evaluations of semantic and acoustic editing.

MMAE-Speech. IFR measures the average success rate on instruction-following rubrics, CR measures consistency on attributes unrelated to the requested edit, and EMR is the percentage of samples that satisfy all instruction-following and consistency rubrics. AuK achieves the highest IFR and CR scores, reaching 48.23% and 88.11%, respectively, while AuK-Flash achieves the highest EMR of 13.85%. Relative to Step-Audio-EditX [73], the strongest prior baseline on IFR and CR, AuK improves the two metrics by 4.71 and 10.84 percentage points, respectively. Relative to Ming-UniAudio [72], AuK-Flash improves EMR from 7.04% to 13.85%. These results show that the full model performs better on average instruction following and preservation, while the Flash variant more frequently satisfies all evaluation rubrics simultaneously.

SpeechEditBench. Among the dedicated editing models included in Tab. 3, AuK achieves the best results in content, emotion, prosody, and acoustic editing, while AuK-Flash performs best in paralinguistic editing. Relative to Ming-UniAudio, AuK improves the joint success rate from 76.46% to 91.83% for content editing and from 26.50% to 71.33% for prosody editing. AuK-Flash achieves a paralinguistic editing score of 39.25%, compared with 31.25% for Step-Audio-EditX.

Ming-Freeform-Audio-Edit. The benchmark contains complementary semantic and acoustic editing tracks. For semantic editing, the summary results average deletion, insertion, and substitution for each Basic/Full and Chinese/English setting. AuK achieves the lowest average WER and no-edit WER across all four settings. Under the more challenging Full setting, compared with Ming-UniAudio, AuK reduces the average WER from 10.46% to 3.09% in Chinese and from 14.28% to 3.96% in English. It also improves editing accuracy from 79.62% to 91.47% in Chinese and from 70.98% to 85.25% in English. The operation-level results in Tabs. 8 and 9 show that these gains extend across deletion, insertion, and substitution.

For acoustic editing, the summary results average speed, pitch, and volume alteration. Compared with Ming-UniAudio, AuK reduces the average WER from 5.01% to 2.02% in Chinese and from 10.75% to 3.48% in English. In contrast, AuK-Flash achieves the highest average SIM scores of 0.79 and 0.75 in Chinese and English, respectively. The operation-level results in Tab. 10 show that AuK achieves the lowest WER for Chinese pitch and volume alteration and for English speed and pitch alteration. AuK-Flash achieves the highest or tied-highest SIM in all six language–operation settings. Overall, the full model provides lower average recognition error, whereas the Flash variant better preserves speaker identity.

## 7.3.2 Speech Enhancement and Separation

The evaluation covers representative task-specific systems, including Resemble-Enhance [54], DaSheng [62], MossFormer2-SS [83], and AudioSR [39], as well as unified or general-purpose systems such as Ming-UniAudio [72], SAM-Audio-Large [58], AnyEnhance [80], RE-USE [15], and Sidon [47]. The main results are summarized in Tab. 3, with complete metrics reported in Tabs. 11 to 14. Across these tasks, DNSMOS and UTMOS assess perceptual quality, dWER, WER, and PER measure linguistic preservation, and SIM measures speaker-identity preservation.

Speech Enhancement. On DNS Challenge, AuK achieves the lowest dWER of 2.66%, improving over RE-USE by 0.65 percentage points, while both variants achieve the highest SIM of 0.99. AuK-

Flash obtains the highest UTMOS score of 4.05 while matching the full model in speaker similarity. On CHiME-4, AuK-Flash achieves the lowest WER of 7.84% and the highest UTMOS score of 3.91; its OVRL score of 3.35 also ties the best result in the complete comparison. AuK remains close in recognition accuracy, with a WER of 7.98%. Across the two enhancement benchmarks, the Flash variant consistently achieves higher UTMOS, while the best recognition result depends on the dataset.

Speech Separation. On Libri2Mix, both variants achieve a SIM of 0.96, tying MossFormer2-SS for the best speaker-similarity result. AuK achieves the lowest WER and PER, at 9.12% and 6.63%, respectively. In contrast, AuK-Flash achieves the highest OVRL and UTMOS scores, at 3.32 and 4.03. These results reveal a clear trade-off: the full model better preserves linguistic content, whereas the Flash variant achieves higher predicted perceptual quality.

Speech Super-Resolution. For speech super-resolution, we apply eight degradation settings to 500 utterances from five held-out VCTK speakers, producing 4,000 test samples. Five subsets evaluate bandwidth extension at effective cutoffs from 2 to 11 kHz, where the 2-kHz condition lies outside the training range and serves as an extrapolation test. The remaining subsets simulate telephone, megaphone, and underwater channels. All inputs and references remain 24-kHz mono waveforms, and we report results averaged uniformly over the eight subsets.

On VCTK-SR, AuK-Flash achieves the best results on six of the seven reported metrics: OVRL, SIG, BAK, UTMOS, WER, and PER. It obtains OVRL and UTMOS scores of 3.25 and 4.05, respectively, together with a WER of 2.92% and a PER of 4.19%. Meanwhile, AuK achieves the highest SIM of 0.97. These results show that the Flash variant provides the strongest perceptual-quality and recognition results, while the full model provides the strongest speaker preservation.

## 7.4 Discovery

Beyond the benchmark results, the development of AuK reveals several observations about data construction, capability transfer, and native instruction following.

Cross-Utterance In-Context Learning. Our zero-shot TTS data construction differs from the conventional within-utterance setting, where a single recording is divided into prompt and target segments. Instead, we identify distinct utterances from the same speaker and use one utterance as the acoustic prompt and another as the synthesis target. Because the two utterances contain different linguistic content and may exhibit natural variation in prosody and recording conditions, this construction encourages the model to separate speaker-invariant characteristics from utterancespecific factors. In our experiments, cross-utterance training improves both expressiveness and speaker similarity. It also enables a transcript-free interface: the model is conditioned only on the prompt waveform and target text, without requiring the transcript corresponding to the prompt audio.

Emergent Cross-Task and Cross-Lingual Transfer. We qualitatively observe capabilities that are not explicitly represented by matched training tasks. First, our whisper data contains only normal-to whisper or whisper-to-normal editing pairs. Nevertheless, the jointly trained model can also synthesize whispered speech directly from text and a style instruction, suggesting that a transformation learned through editing can transfer to generation. Second, our de-accenting supervision covers Chinese dialects and regional accents only, yet the model can reduce accents in English speech, including English spoken with Indian or Japanese accents, while largely preserving speaker identity. These observations are not substitutes for comprehensive benchmark evaluation, but they suggest that unified training can factorize certain acoustic transformations from the language and task through which they are supervised.

Limits of Native Free-Form Instruction Following. We also explored an agent-based data pipeline modified from Audio-Oscar [13] for free-form audio-editing SFT pairs. Although this pipeline increased the linguistic diversity of editing instructions, it did not produce sufficiently robust general ization to arbitrary user requests. In practice, reaching the model’s capability ceiling still requires the Prompt Enhancer to identify the intended task, normalize control parameters, and rewrite the request toward the training distribution. A similar dependence on prompt rewriting is commonly observed in image, video, and audio-visual generation systems, indicating a broader gap between possessing a capability and invoking it reliably through unconstrained language. Improving native instruction grounding and compositional generalization, while reducing reliance on explicit task routing and prompt enhancement, remains an important direction for future work.

## 8 Conclusion

We presented AuK, an open-source foundational model that unifies speech generation and editing through a common instruction-conditioned waveform generation interface. Its training corpus spans five task families and contains approximately 3.03 billion instruction–audio instances and 1.95 million hours of effective supervision. A multimodal language model provides semantic conditioning, an audio VAE jointly trained on speech, general audio, and music provides a shared acoustic latent space, and a flex-style Transformer for latent diffusion. Generation-only warm-up, joint generation–editing pre-training, human-feedback preference optimization for open-ended editing, and reward-based reinforcement learning for generation together align the model across this heterogeneous capability set. Task-routed distillation further produces AuK-Flash, which retains broad capability with four-step inference, no classifier-free guidance, and a 4.5× wall-clock speedup.

Experiments show leading performance on zero-shot and vocie-design speech generation and general instruction-guided editing, together with competitive results on speech restoration tasks. Our qualitative observations also suggest that unified training enables useful transfer across utterances, tasks, and languages. At the same time, robust native understanding of unconstrained editing requests remains incomplete, and the system still benefits from explicit task routing and prompt enhancement. Future work should improve native instruction grounding, compositional generalization, and scalable alignment for open-ended audio transformations while further reducing inference cost. We release the source code and model weights to facilitate future development and research.

## Contribution

## ⋆ Core Contributors

Ziyang Ma, Zhikang Niu, Wenming Tu, Tianrui Wang, Ruiqi Yan, Junxi Liu, Yanru Huo

## ² Contributors

## Ô Engineering & Training1

Nickk Huang, Yang Liu, Qicong Xie, Zeyu Xie, Hui Wang, Haitao Li, Zixuan Jiang

## f Data & Infra2

Yalin Li, Jie Fang, Yifan Duan, Zeyue Tian, Guangzheng Li, Haina Zhu, Shuyi Wang, Jinwen Wang, Mingyu Cui, Tian Tan, Auden, Sen Liang

## ± Project Sponsors & Advisors

Steve Yves, Shan Yang, Liefeng Bo, Zilong Zheng, Kai Yu, Eng-Siong Chng, Xie Chen

## ♥ Acknowledgements

Yushen Chen, Wenxi Chen, Feiteng Li, Qixi Zheng, Yiwei Guo, Guanrou Yang, Yipeng Kang, Shengpeng Ji, Yuzhe Liang, Peifan Chen, Qixiang Xu, Jiayi Liang, Jubin Zhang, Jiaxin Zhi, Shanyi Zhu, Yiru Fan, Pan Luo

## References

[1] Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024.

[2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016.

[3] Junyang Chen, Yuhang Jia, Hui Wang, Jiaming Zhou, Yongchang Gan, and Yong Qin. Cosyedit2: Speech-editing-oriented reinforcement learning unlocks better zero-shot tts. arXiv preprint arXiv:2605.25930, 2026.

[4] Junyang Chen, Yuhang Jia, Hui Wang, Jiaming Zhou, and Yong Qin. Cosyedit: Unlocking end-to-end speech editing capability from zero-shot text-to-speech models. arXiv preprint arXiv:2601.05329, 2026.

[5] Szu-Jui Chen, Aswin Shanmugam Subramanian, Hainan Xu, and Shinji Watanabe. Building state-of-the-art distant speech recognition using the chime-4 challenge with a setup of speech enhancement baseline. arXiv preprint arXiv:1803.10109, 2018.

[6] Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, and Xie Chen. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. arXiv preprint arXiv:2410.06885, 2024.

[7] Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. Mmaudio: Taming multimodal joint training for high-quality video-to-audio synthesis. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28901–28911. IEEE, 2025.

[8] Joris Cosentino, Manuel Pariente, Samuele Cornell, Antoine Deleforge, and Emmanuel Vincent. Librimix: An open-source dataset for generalizable speech separation. arXiv preprint arXiv:2005.11262, 2020.

[9] DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.

[10] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024.

[11] Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, et al. Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training. arXiv preprint arXiv:2505.17589, 2025.

[12] Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117, 2024.

[13] Yifan Duan, Qixiang Xu, Hengtao Wu, Zhanxun Liu, Wenhao Guan, Junxi Liu, Ziyang Ma, Kelu Xu, and Xie Chen. Audio-oscar: A multi-agent system for complex audio scene generation, orchestration, and refinement. arXiv preprint arXiv:2606.07397, 2026.

[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transform ers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

[15] Szu-Wei Fu, Rong Chao, Xuesong Yang, Sung-Feng Huang, Ryandhimas E Zezario, Rauf Nasretdinov, Ante Jukic, Yu Tsao, and Yu-Chiang Frank Wang. Rethinking training targets, ar-´ chitectures and data quality for universal speech enhancement. arXiv preprint arXiv:2603.02641, 2026.

[16] Jort F Gemmeke, Daniel P W Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780. IEEE, 2017.

[17] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

[18] Yicheng Gu, Xueyao Zhang, Liumeng Xue, and Zhizheng Wu. Multi-scale sub-band constant-q transform discriminator for high-fidelity vocoder, 2023.

[19] Zhifang Guo, Yichong Leng, Yihan Wu, Sheng Zhao, and Xu Tan. Prompttts: Controllable text-to-speech with text descriptions. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

[20] Chunbo Hao, Junjie Zheng, Guobin Ma, Yuepeng Jiang, Huakang Chen, Wenjie Tian, Gongyu Chen, Zihao Chen, and Lei Xie. Yingmusic-singer-plus: Controllable singing voice synthesis with flexible lyric manipulation and annotation-free melody guidance. arXiv preprint arXiv:2603.24589, 2026.

[21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[22] Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, et al. Qwen3-tts technical report. arXiv preprint arXiv:2601.15621, 2026.

[23] Jingbin Hu, Huakang Chen, Linhan Ma, Dake Guo, Qirui Zhan, Wenhao Li, Haoyu Zhang, Kangxiang Xia, Ziyu Zhang, Wenjie Tian, et al. Voicesculptor: Your voice, designed by you. arXiv preprint arXiv:2601.10629, 2026.

[24] Kexin Huang, Liwei Fan, Botian Jiang, Yaozhou Jiang, Qian Tu, Jie Zhu, Yuqian Zhang, Yiwei Zhao, Chenchen Yang, Zhaoye Fei, et al. Moss-voicegenerator: Create realistic voices with natural language descriptions. arXiv preprint arXiv:2603.28086, 2026.

[25] Kexin Huang, Qian Tu, Liwei Fan, Chenchen Yang, Dong Zhang, Shimin Li, Zhaoye Fei, Qinyuan Cheng, and Xipeng Qiu. Instructttseval: Benchmarking complex natural-language instruction following in text-to-speech systems. arXiv preprint arXiv:2506.16381, 2025.

[26] Hume AI. Hume ai: Human feedback for voice, speech, and conversational ai. https: //www.hume.ai/, 2026. Accessed: 2026-09-08.

[27] InclusionAI. Ming-omni-tts: Simple and efficient unified generation of speech, music, and sound with precise control. https://github.com/inclusionAI/Ming-omni-tts, 2026. GitHub repository.

[28] Dongya Jia, Zhuo Chen, Jiawei Chen, Chenpeng Du, Jian Wu, Jian Cong, Xiaobin Zhuang, Chumin Li, Zhen Wei, Yuping Wang, et al. Ditar: Diffusion transformer autoregressive modeling for speech generation. arXiv preprint arXiv:2502.03930, 2025.

[29] Ziyue Jiang, Yi Ren, Ruiqi Li, Shengpeng Ji, Boyang Zhang, Zhenhui Ye, Chen Zhang, Bai Jionghao, Xiaoda Yang, Jialong Zuo, et al. Megatts 3: Sparse alignment enhanced latent diffusion transformer for zero-shot speech synthesis. arXiv preprint arXiv:2502.18924, 2025.

[30] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. In Advances in Neural Information Processing Systems, volume 33, 2020.

[31] Rithesh Kumar, Prem Seetharaman, et al. High-fidelity audio compression with improved rvqgan. In NeurIPS, 2023.

[32] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

[33] Zitong Lan, Yiduo Hao, and Mingmin Zhao. Guiding audio editing with audio language model. arXiv preprint arXiv:2509.21625, 2025.

[34] Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, and Sungroh Yoon. Bigvgan: A universal neural vocoder with large-scale training. In Proc. ICLR, 2023.

[35] Jiaxing Li, Kai Zou, Cindy Zhou, Kaichen Huang, Junyao Gao, Zile Wang, Yang Liu, Bin Liu, Bo An, and Yangguang Li. Distillalign: Coordinating mode covering and mode seeking in autoregressive video distillation. arXiv preprint arXiv:2607.26811, 2026.

[36] Junzhe Li, Yutao Cui, Tao Huang, Weijie Kong, Chuxuan Zeng, Yiming Cheng, Yinping Ma, Chun Fan, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.

[37] Dongyang Liu, Peng Gao, David Liu, Ruoyi Du, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Hongsheng Li, and Steven Hoi. Decoupled DMD: CFG augmentation as the spear, distribution matching as the shield. In International Conference on Learning Representa tions, 2026.

[38] Guanghou Liu, Yongmao Zhang, Yi Lei, Yunlin Chen, Rui Wang, Zhifei Li, and Lei Xie. Promptstyle: Controllable style transfer for text-to-speech with natural language descriptions. arXiv preprint arXiv:2305.19522, 2023.

[39] Haohe Liu, Ke Chen, Qiao Tian, Wenwu Wang, and Mark D Plumbley. Audiosr: Versatile audio super-resolution at scale. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1076–1080. IEEE, 2024.

[40] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

[41] Songting Liu. Zero-shot voice conversion with diffusion transformers. arXiv preprint arXiv:2411.09943, 2024.

[42] Tianqi Liu, Zhen Qin, Junru Wu, Jiaming Shen, Misha Khalman, Rishabh Joshi, Yao Zhao, Mohammad Saleh, Simon Baumgartner, Jialu Liu, Peter J. Liu, and Xuanhui Wang. Lipo: Listwise preference optimization through learning-to-rank. arXiv preprint arXiv:2402.01878, 2024.

[43] Dan Lyth and Simon King. Natural language guidance of high-fidelity text-to-speech with synthetic annotations. arXiv preprint arXiv:2402.01912, 2024.

[44] Ziyang Ma, Ruiqi Yan, Ruiyang Xu, Jie Fang, Zhikang Niu, Yi-Wen Chao, Wenming Tu, Tianrui Wang, Qi Chen, Wenxi Chen, et al. Mmae: A massive multitask audio editing benchmark. arXiv preprint arXiv:2606.07229, 2026.

[45] Xudong Mao, Qing Li, Haoran Xie, Raymond Y. K. Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision, pages 2794–2802, 2017.

[46] MiniMax-AI. MiniMax H3. https://github.com/MiniMax-AI/MiniMax-H3, 2026. GitHub repository, H3-AudioVAE component.

[47] Wataru Nakata, Yuki Saito, Kazuki Yamauchi, Emiru Tsunoo, and Hiroshi Saruwatari. Dialoguesidon: Recovering full-duplex dialogue tracks from in-the-wild dialogue audio. In Proceedings of the 27th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 1–12, 2026.

[48] OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai. com/index/gpt-5-6/, 2026.

[49] Julian D. Parker, Zach Evans, CJ Carr, Zack Zukowski, Josiah Taylor, Matthew Rice, and Jordi Pons. Same: A semantically-aligned music autoencoder, 2026.

[50] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

[51] Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, Shaohan Huang, Yan Xia, and Furu Wei. Vibevoice technical report. arXiv preprint arXiv:2508.19205, 2025.

[52] Zafar Rafii, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner. MUSDB18-HQ - an uncompressed version of musdb18, December 2019.

[53] Chandan KA Reddy, Vishak Gopal, Ross Cutler, Ebrahim Beyrami, Roger Cheng, Harishchandra Dubey, Sergiy Matusevych, Robert Aichner, Ashkan Aazami, Sebastian Braun, et al. The interspeech 2020 deep noise suppression challenge: Datasets, subjective testing framework, and challenge results. arXiv preprint arXiv:2005.13981, 2020.

[54] Resemble AI. Resemble enhance: Ai-powered speech denoising and enhancement. https: //github.com/resemble-ai/resemble-enhance, 2023. Software repository.

[55] Seyedmorteza Sadat, Otmar Hilliges, and Romann M. Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In International Conference on Learning Representations, 2025.

[56] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[57] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

[58] Bowen Shi, Andros Tjandra, John Hoffman, Helin Wang, Yi-Chiao Wu, Luya Gao, Julius Richter, Matt Le, Apoorv Vyas, Sanyuan Chen, et al. Sam audio: Segment anything in audio. arXiv preprint arXiv:2512.18099, 2025.

[59] Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, et al. Qwen3-asr technical report. arXiv preprint arXiv:2601.21337, 2026.

[60] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

[61] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[62] Xingwei Sun, Heinrich Dinkel, Yadong Niu, Linzhang Wang, Junbo Zhang, and Jian Luan. Efficient speech enhancement via embeddings from pre-trained generative audioencoders. arXiv preprint arXiv:2506.11514, 2025.

[63] Ye Tao, Wen Wu, Chao Zhang, Mengyue Wu, Shuai Wang, and Xuenan Xu. Mmedit: A unified framework for multi-type audio editing via audio language model. arXiv preprint arXiv:2512.20339, 2025.

[64] Tencent Hunyuan Team. Hunyuan3. https://github.com/Tencent-Hunyuan/Hy3, 2026.

[65] Zeyue Tian, Binxin Yang, Zhaoyang Liu, Jiexuan Zhang, Ruibin Yuan, Hubery Yin, Qifeng Chen, Chen Li, Jing Lyu, Wei Xue, et al. Audio-omni: Extending multi-modal understanding to versatile audio generation and editing. arXiv preprint arXiv:2604.10708, 2026.

[66] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238. IEEE, 2024.

[67] Haoxu Wang, Biao Tian, Weiqin Li, Xiang Lv, Han Zhao, and Xiangang Li. Flowtts-grpo: Online reinforcement learning with multi-objective reward optimization for flow-matching based text-to-speech. In Interspeech, 2026.

[68] Kun Xie, Feiyu Shen, Junjie Li, Fenglong Xie, Xu Tang, and Yao Hu. Fireredtts-2: Towards long conversational speech generation for podcast and chatbot. arXiv preprint arXiv:2509.02020, 2025.

[69] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

[70] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

[71] Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald. CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92), 2019.

[72] Canxiang Yan, Chunxiang Jin, Dawei Huang, Haibing Yu, Han Peng, Hui Zhan, Jie Gao, Jing Peng, Jingdong Chen, Jun Zhou, et al. Ming-uniaudio: Speech llm for joint understanding, generation and editing with unified representation. arXiv preprint arXiv:2511.05516, 2025.

[73] Chao Yan, Boyong Wu, Peng Yang, Pengfei Tan, Guoqiang Hu, Li Xie, Yuxin Zhang, Fei Tian, Xuerui Yang, Xiangyu Zhang, et al. Step-audio-editx technical report. arXiv preprint arXiv:2511.03601, 2025.

[74] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Frédo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025.

[75] Z-Image Team, Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Aiming Hao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Yuming Jiang, Xin Jin, Liangchen Li, Zhen Li, Zhong-Yu Li, David Liu, Dongyang Liu, Qilong Wu, Feng Yu, Zechao Zhan, Chi Zhang, Shifeng Zhang, Ruikai Zhou, and Shilin Zhou. Z-Image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

[76] Biao Zhang and Rico Sennrich. Root mean square layer normalization, 2019.

[77] Bowen Zhang, Congchao Guo, Geng Yang, Hang Yu, Haozhe Zhang, Heidi Lei, Jialong Mai, Junjie Yan, Kaiyue Yang, Mingqi Yang, Peikai Huang, Ruiyang Jin, Sitan Jiang, Weihua Cheng, Yawei Li, Yichen Xiao, Yiying Zhou, Yongmao Zhang, Yuan Lu, and Yucen He. Minimaxspeech: Intrinsic zero-shot text-to-speech with a learnable speaker encoder. arXiv preprint arXiv:2505.07916, 2025.

[78] Dong Zhang, Gang Wang, Jinlong Xue, Kai Fang, Liang Zhao, Rui Ma, Shuhuai Ren, Shuo Liu, Tao Guo, Weiji Zhuang, et al. Mimo-audio: Audio language models are few-shot learners. arXiv preprint arXiv:2512.23808, 2025.

[79] Hanlin Zhang, Daxin Tan, Dehua Tao, Xiao Chen, Haochen Tan, and Linqi Song. Speecheditbench: A bilingual multi-attribute benchmark for instruction-guided speech editing. arXiv preprint arXiv:2606.01804, 2026.

[80] Junan Zhang, Jing Yang, Zihao Fang, Yuancheng Wang, Zehua Zhang, Zhuo Wang, Fan Fan, and Zhizheng Wu. Anyenhance: A unified generative model with prompt-guidance and self-critic for voice enhancement. IEEE Transactions on Audio, Speech and Language Processing, 2025.

[81] Yu Zhang, Xiang Yin, Cheng Yang, Ruiqi Li, Changhao Pan, and Ke Lei. Swantale: Unified multi-speaker speech and audio generation for instruct and zero-shot tasks. Technical report, ByteDance, 2026. arXiv preprint arXiv:2608.02023.

[82] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026.

[83] Shengkui Zhao, Yukun Ma, Chongjia Ni, Chong Zhang, Hao Wang, Trung Hieu Nguyen, Kun Zhou, Jia Qi Yip, Dianwen Ng, and Bin Ma. Mossformer2: Combining transformer and rnn-free recurrent network for enhanced time-domain monaural speech separation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 10356–10360. IEEE, 2024.

[84] Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rcm: A unified teacher-forcing and selfforcing open recipe for autoregressive diffusion distillation in streaming video generation and interactive world models. arXiv preprint arXiv:2606.25473, 2026.

[85] Qixi Zheng, Yuxiang Zhao, Tianrui Wang, Wenxi Chen, Kele Xu, Yikang Li, Qinyuan Chen, Xipeng Qiu, Kai Yu, and Xie Chen. X-vc: Zero-shot streaming voice conversion in codec space. Proc. ACM Multimedia, 2026.

[86] Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, and Jingchen Shu. Indextts2: A breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 35139–35148, 2026.

[87] Yixuan Zhou, Xiaoyu Qin, Zeyu Jin, Shuoyi Zhou, Shun Lei, Songtao Zhou, Zhiyong Wu, and Jia Jia. Voxinstruct: Expressive human instruction-to-speech generation with unified multilingual codec language modelling. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 554–563, 2024.

[88] Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Jiancheng Gui, Jiaheng Wu, Ziyang Wang, Xudong Shen, Runchuan Ye, et al. Voxcpm2 technical report. arXiv preprint arXiv:2606.06928, 2026.

[89] Han Zhu, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhifeng Han, Weiji Zhuang, Long Lin, and Daniel Povey. Omnivoice: Towards omnilingual zero-shot text-to-speech with diffusion language models. arXiv preprint arXiv:2604.00688, 2026.

## A Detailed Evaluation Results

This appendix complements the benchmark summary in Tab. 3 with detailed comparisons for AuK and AuK-Flash. The evaluation is organized into speech generation, general instruction-guided editing, and signal-level enhancement and separation. We retain each benchmark’s native metrics and include broader baseline sets to compare linguistic accuracy, speaker preservation, instruction adherence, and perceptual quality.

## A.1 Generation Benchmarks

We evaluate two complementary generation settings. Seed-TTS-Eval [1] measures zero-shot voice cloning across English, Chinese, and challenging Chinese text using recognition error and speaker similarity, whereas InstructTTSEval [25] measures control over acoustic parameters, descriptive styles, and role-playing instructions in Chinese and English. For both benchmarks, we report the mean over three runs and use only benchmark-specific instruction templates and duration estimation, without prompt enhancement.

Table 5: Zero-shot TTS performance on Seed-TTS-Eval.
<table><tr><td rowspan="2">Model</td><td colspan="2">test-en</td><td colspan="2">test-zh</td><td colspan="2">test-zh-hard</td><td colspan="2">Average</td></tr><tr><td>WER(%)↓</td><td>SIM↑</td><td>CER(%)↓</td><td>SIM↑</td><td>CER(%)↓</td><td>SIM↑</td><td>CER/WER(%)↓</td><td>SIM↑</td></tr><tr><td>F5-TTS [6]</td><td>2.00</td><td>0.670</td><td>1.53</td><td>0.760</td><td>8.67</td><td>0.713</td><td>4.10</td><td>0.714</td></tr><tr><td>FireRedTTS 2 [68]</td><td>1.95</td><td>0.665</td><td>1.14</td><td>0.736</td><td>8.98</td><td>0.703</td><td>4.02</td><td>0.701</td></tr><tr><td>IndexTTS 2 [86]</td><td>2.23</td><td>0.706</td><td>1.03</td><td>0.765</td><td>7.12</td><td>0.755</td><td>3.46</td><td>0.742</td></tr><tr><td>MegaTTS 3 [29]</td><td>2.79</td><td>0.771</td><td>1.52</td><td>0.790</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-TTS [22]</td><td>1.23</td><td>0.717</td><td>1.22</td><td>0.770</td><td>6.76</td><td>0.748</td><td>3.07</td><td>0.745</td></tr><tr><td>Seed-TTS [1]</td><td>2.25</td><td>0.762</td><td>1.12</td><td>0.796</td><td>7.59</td><td>0.776</td><td>3.65</td><td>0.778</td></tr><tr><td>DiTAR [28]</td><td>1.69</td><td>0.735</td><td>1.02</td><td>0.753</td><td></td><td></td><td></td><td></td></tr><tr><td>VibeVoice [51]</td><td>3.04</td><td>0.689</td><td>1.16</td><td>0.744</td><td></td><td></td><td></td><td></td></tr><tr><td>VoxCPM 2 [88]</td><td>1.84</td><td>0.753</td><td>0.97</td><td>0.795</td><td>8.13</td><td>0.753</td><td>3.65</td><td>0.767</td></tr><tr><td>AuK-Flash</td><td>1.03</td><td>0.772</td><td>1.10</td><td>0.814</td><td>6.43</td><td>0.784</td><td>2.85</td><td>0.790</td></tr><tr><td>AuK</td><td>1.02</td><td>0.788</td><td>1.02</td><td>0.814</td><td>5.91</td><td>0.782</td><td>2.65</td><td>0.795</td></tr></table>

Table 6: Instruction-following TTS performance on InstructTTSEval.
<table><tr><td rowspan="2">Model</td><td colspan="3">Chinese</td><td colspan="3">English</td></tr><tr><td>APS↑</td><td>DSD↑</td><td>RP↑</td><td>APS↑</td><td>DSD↑</td><td>RP↑</td></tr><tr><td>Qwen3-TTS-VD [22]</td><td>85.20</td><td>81.10</td><td>65.10</td><td>82.90</td><td>82.40</td><td>68.40</td></tr><tr><td>Mimo-Audio [78]</td><td>75.70</td><td>74.30</td><td>61.50</td><td>80.60</td><td>77.60</td><td>59.50</td></tr><tr><td>VoiceSculptor [23]</td><td>75.70</td><td>64.70</td><td>61.50</td><td></td><td></td><td></td></tr><tr><td>Hume AI [26]</td><td></td><td></td><td></td><td>83.00</td><td>75.30</td><td>54.30</td></tr><tr><td>VoxInstruct [87]</td><td>47.50</td><td>52.30</td><td>42.60</td><td>54.90</td><td>57.00</td><td>39.30</td></tr><tr><td>Parler-TTS-Mini [43]</td><td></td><td></td><td></td><td>63.40</td><td>48.70</td><td>28.60</td></tr><tr><td>Parler-TTS-Large [43]</td><td></td><td></td><td></td><td>60.00</td><td>45.90</td><td>31.20</td></tr><tr><td>PromptTTS [19]</td><td>1</td><td></td><td></td><td>64.30</td><td>47.20</td><td>31.40</td></tr><tr><td>PromptStyle [38]</td><td></td><td></td><td></td><td>57.40</td><td>46.40</td><td>30.90</td></tr><tr><td>MOSS-VoiceGenerator [24]</td><td>78.00</td><td>80.00</td><td>74.00</td><td>68.20</td><td>82.00</td><td>68.70</td></tr><tr><td>AuK-Flash</td><td>81.78</td><td>78.80</td><td>63.13</td><td>72.33</td><td>82.40</td><td>63.87</td></tr><tr><td>AuK</td><td>83.28</td><td>83.37</td><td>68.23</td><td>75.57</td><td>81.60</td><td>65.93</td></tr></table>

## A.2 General Speech Editing Benchmarks

General speech editing evaluates whether a model follows free-form editing instructions while preserving content and speaker attributes outside the requested change. MMAE-Speech [44] measures instruction following, content retention, and overall edit success with the Prompt Enhancer enabled. Ming-Freeform-Audio-Edit [72] further decomposes semantic editing into deletion, insertion, and substitution under bilingual Basic and Full settings, and evaluates acoustic control over speaking rate, pitch, and volume. The detailed tables report operation-level results rather than only the averages presented in Tab. 3.

Table 7: Instruction-guided speech editing performance on MMAE-Speech with the Prompt Enhancer enabled. Models marked with <sup>∗</sup> are evaluated only on MMAE samples with input durations of at most 10 seconds, following the original benchmark protocol.
<table><tr><td>Model</td><td>IFR↑</td><td>CR↑</td><td>EMR↑</td></tr><tr><td>Step-Audio-EditX [73]</td><td>43.52</td><td>77.27</td><td>4.69</td></tr><tr><td>Ming-UniAudio [72] MMEdit* [63]</td><td>34.13 30.52</td><td>76.01 35.40</td><td>7.04 0.99</td></tr><tr><td>Audio-Omni* [65]</td><td>43.14</td><td>68.29</td><td>1.98</td></tr><tr><td>SmartDJ* w/o planner [33] SmartDJ* w/ planner [33]</td><td>28.03</td><td>56.22</td><td>2.97</td></tr><tr><td></td><td>32.17</td><td>43.00</td><td>0.99</td></tr><tr><td>AuK-Flash AuK</td><td>46.62 48.23 88.11</td><td>86.41</td><td>13.85</td></tr></table>

Table 8: Semantic speech editing performance on the Basic split of Ming-Freeform-Audio-Edit.
<table><tr><td rowspan="2"></td><td rowspan="2">Lang. Model</td><td colspan="4">Deletion</td><td colspan="4">Insertion</td><td colspan="4">Substitution</td></tr><tr><td>WER(%)↓</td><td>ACC↑</td><td>SIM↑</td><td>no-edit WER(%)↓</td><td>WER(%)↓</td><td>ACC↑</td><td>SIM↑</td><td>no-edit WER(%)↓</td><td>WER(%)↓</td><td>ACC↑ SIM↑</td><td></td><td>no-edit WER(%)↓</td></tr><tr><td rowspan="3">ZH</td><td>Ming-UniAudio [72]</td><td>11.89</td><td>100.00</td><td>0.78</td><td>11.49</td><td>3.42</td><td>80.00</td><td>0.83</td><td>3.52</td><td>4.52</td><td>78.62</td><td>0.82</td><td>4.63</td></tr><tr><td>AuK-Flash</td><td>9.85</td><td>100.00</td><td>0.80</td><td>9.68</td><td>3.60</td><td>82.94</td><td>0.93</td><td>3.96</td><td>1.15</td><td>93.08</td><td>0.91</td><td>1.38</td></tr><tr><td>AuK</td><td>9.61</td><td>100.00</td><td>0.80</td><td>9.46</td><td>3.03</td><td>85.29</td><td>0.92</td><td>3.38</td><td>1.21</td><td>90.57</td><td>0.91</td><td>1.46</td></tr><tr><td rowspan="3">EN</td><td>Ming-UniAudio [72]</td><td>14.85</td><td>82.22</td><td>0.76</td><td>24.26</td><td>6.63</td><td>71.43</td><td>0.79</td><td>17.70</td><td>8.99</td><td>59.78</td><td>0.78</td><td>19.28</td></tr><tr><td>AuK-Flash</td><td>7.38</td><td>100.00</td><td>0.84</td><td>18.98</td><td>4.26</td><td>78.26</td><td>0.93</td><td>15.61</td><td>2.48</td><td>73.74</td><td>0.89</td><td>15.06</td></tr><tr><td>AuK</td><td>5.91</td><td>100.00</td><td>0.83</td><td>17.76</td><td>3.47</td><td>83.23</td><td>0.92</td><td>15.01</td><td>2.18</td><td>73.18</td><td>0.89</td><td>14.74</td></tr></table>

Table 9: Semantic speech editing performance on the Full split of Ming-Freeform-Audio-Edit.
<table><tr><td rowspan="2"></td><td rowspan="2">Lang. Model</td><td colspan="4">Deletion</td><td colspan="4">Insertion</td><td colspan="4">Substitution</td></tr><tr><td>WER(%)↓</td><td>ACC↑</td><td>SIM↑</td><td>no-edit WER(%)↓</td><td>WER(%)↓</td><td>ACC↑</td><td>SIM↑</td><td>no-edit WER(%)↓</td><td>WER(%)↓</td><td>ACC↑ SIM↑</td><td></td><td>no-edit WER(%)↓</td></tr><tr><td rowspan="3">ZH</td><td>Ming-UniAudio [72]</td><td>22.92</td><td>82.92</td><td>0.81</td><td>17.50</td><td>3.89</td><td>79.31</td><td>0.83</td><td>4.10</td><td>4.56</td><td>76.62</td><td>0.83</td><td>4.75</td></tr><tr><td>AuK-Flash</td><td>5.17</td><td>98.22</td><td>0.83</td><td>4.50</td><td>3.04</td><td>85.86</td><td>0.93</td><td>3.40</td><td>1.82</td><td>89.54</td><td>0.92</td><td>2.07</td></tr><tr><td>AuK</td><td>4.75</td><td>98.58</td><td>0.82</td><td>4.20</td><td>2.77</td><td>86.90</td><td>0.92</td><td>3.14</td><td>1.75</td><td>88.92</td><td>0.91</td><td>1.99</td></tr><tr><td rowspan="3">EN</td><td>Ming-UniAudio [72]</td><td>27.60</td><td>85.00</td><td>0.74</td><td>35.21</td><td>7.59</td><td>62.31</td><td>0.79</td><td>18.84</td><td>7.64</td><td>65.62</td><td>0.77</td><td>18.39</td></tr><tr><td>AuK-Flash</td><td>7.55</td><td>100.00</td><td>0.84</td><td>18.90</td><td>4.29</td><td>75.88</td><td>0.93</td><td>15.95</td><td>2.68</td><td>73.05</td><td>0.89</td><td>15.44</td></tr><tr><td>AuK</td><td>6.17</td><td>100.00</td><td>0.83</td><td>17.81</td><td>3.38</td><td>81.91</td><td>0.92</td><td>15.09</td><td>2.34</td><td>73.83</td><td>0.89</td><td>15.06</td></tr></table>

Table 10: Acoustic speech editing performance on Ming-Freeform-Audio-Edit. RDE and RAE are reported for speed and volume alteration, respectively; bold and underlined values denote the best and second-best result for each language and task.
<table><tr><td rowspan="3">Lang.</td><td rowspan="3">Model</td><td colspan="3">Speed Alteration</td><td colspan="2">Pitch Alteration</td><td colspan="3">Volume Alteration</td></tr><tr><td>WER(%)↓</td><td>SIM↑</td><td>RDE(%)↓</td><td>WER(%)↓</td><td>SIM↑</td><td>WER(%)↓</td><td>SIM↑</td><td>RAE(%)↓</td></tr><tr><td rowspan="3">ZH</td><td>Ming-UniAudio [72]</td><td>5.88</td><td>0.66</td><td>6.36</td><td>7.45</td><td>0.36</td><td>1.71</td><td>0.86</td><td>14.90</td></tr><tr><td>AuK-Flash</td><td>2.60</td><td>0.82</td><td>10.57</td><td>2.67</td><td>0.58</td><td>1.93</td><td>0.97</td><td>24.69</td></tr><tr><td>AuK</td><td>2.72</td><td>0.79</td><td>10.57</td><td>1.82</td><td>0.58</td><td>1.52</td><td>0.96</td><td>32.89</td></tr><tr><td rowspan="3">EN</td><td>Ming-UniAudio [72]</td><td>17.53</td><td>0.57</td><td>5.92</td><td>13.37</td><td>0.24</td><td>1.35</td><td>0.80</td><td>11.70</td></tr><tr><td>AuK-Flash</td><td>5.66</td><td>0.76</td><td>11.96</td><td>4.94</td><td>0.53</td><td>2.76</td><td>0.96</td><td>18.47</td></tr><tr><td>AuK</td><td>4.80</td><td>0.75</td><td>11.96</td><td>3.92</td><td>0.53</td><td>1.73</td><td>0.95</td><td>25.40</td></tr></table>

## A.3 Speech Enhancement and Separation Benchmarks

These benchmarks assess signal-level processing across denoising, separation, and super-resolution. DNS Challenge 2020 [53] and CHiME-4 [5] evaluate speech enhancement, Libri2Mix [8] evaluates two-speaker separation, and VCTK-SR [71] evaluates recovery from bandwidth and channel degradations. DNSMOS and UTMOS estimate perceptual quality; WER, dWER, and PER measure linguistic preservation; and SIM measures speaker-identity preservation. Reporting these metrics together distinguishes perceptual improvement from content loss or speaker drift.

Table 11: Speech enhancement performance on DNS Challenge 2020.
<table><tr><td rowspan="2">Model</td><td colspan="3">DNSMOS</td><td rowspan="2"></td><td rowspan="2">UTMOS↑ dWER(%)↓ SIM↑</td><td rowspan="2"></td></tr><tr><td>OVRL↑</td><td>SIG↑</td><td>BAK↑</td></tr><tr><td>Ming-UniAudio [72]</td><td>3.26</td><td>3.57</td><td>3.99</td><td>3.73</td><td>6.91</td><td>0.97</td></tr><tr><td>Resemble-Enhance [54]</td><td>3.37</td><td>3.62</td><td>4.12</td><td>3.49</td><td>6.89</td><td>0.97</td></tr><tr><td>DaSheng [62]</td><td>3.37</td><td>3.58</td><td>4.17</td><td>3.46</td><td>3.48</td><td>0.98</td></tr><tr><td>SAM-Audio-Large [58]</td><td>3.29</td><td>3.56</td><td>4.06</td><td>3.78</td><td>9.67</td><td>0.96</td></tr><tr><td>AnyEnhance [80]</td><td>3.41</td><td>3.63</td><td>4.19</td><td>3.76</td><td>7.71</td><td>0.98</td></tr><tr><td>RE-USE [15]</td><td>3.38</td><td>3.60</td><td>4.19</td><td>3.69</td><td>3.31</td><td>0.98</td></tr><tr><td>AuK-Flash</td><td>3.38</td><td>3.62</td><td>4.13</td><td>4.05</td><td>2.93</td><td>0.99</td></tr><tr><td>AuK</td><td>3.35</td><td>3.59</td><td>4.13</td><td>3.86</td><td>2.66</td><td>0.99</td></tr></table>

Table 12: Speech enhancement performance on CHiME-4.
<table><tr><td rowspan="2">Model</td><td colspan="3">DNSMOS</td><td rowspan="2">UTMOS↑</td><td rowspan="2">WER(%)↓</td></tr><tr><td>OVRL↑</td><td>SIG↑</td><td>BAK↑</td></tr><tr><td>Ming-UniAudio [72]</td><td>3.19</td><td>3.47</td><td>4.04</td><td>3.56</td><td>21.66</td></tr><tr><td>Resemble-Enhance [54]</td><td>3.35</td><td>3.60</td><td>4.12</td><td>3.36</td><td>31.90</td></tr><tr><td>DaSheng [62]</td><td>3.27</td><td>3.49</td><td>4.17</td><td>3.34</td><td>10.89</td></tr><tr><td>SAM-Audio-Large [58]</td><td>3.14</td><td>3.49</td><td>3.90</td><td>3.38</td><td>17.30</td></tr><tr><td>AnyEnhance [80]</td><td>3.19</td><td>3.41</td><td>4.13</td><td>3.24</td><td>27.41</td></tr><tr><td>RE-USE [15]</td><td>3.33</td><td>3.55</td><td>4.18</td><td>3.44</td><td>10.71</td></tr><tr><td>AuK-Flash</td><td>3.35</td><td>3.59</td><td>4.14</td><td>3.91</td><td>7.84</td></tr><tr><td>AuK</td><td>3.28</td><td>3.51</td><td>4.14</td><td>3.72</td><td>7.98</td></tr></table>

Table 13: Two-speaker speech separation performance on Libri2Mix.
<table><tr><td rowspan="2">Model</td><td colspan="3">DNSMOS</td><td rowspan="2">UTMOS↑</td><td rowspan="2">WER(%)↓</td><td rowspan="2">PER(%)↓</td><td rowspan="2">SIM↑</td></tr><tr><td>OVRL↑</td><td>SIG↑</td><td>BAK↑</td></tr><tr><td>MossFormer2-SS [83]</td><td>3.24</td><td>3.48</td><td>4.11</td><td>3.66</td><td>9.34</td><td>7.23</td><td>0.96</td></tr><tr><td>SAM-Audio-Large [58]</td><td>2.87</td><td>3.27</td><td>3.55</td><td>2.84</td><td>57.22</td><td>47.00</td><td>0.86</td></tr><tr><td>Sidon (Dialogue) [47]</td><td>3.08</td><td>3.48</td><td>3.77</td><td>2.61</td><td>51.79</td><td>39.57</td><td>0.87</td></tr><tr><td>AuK-Flash</td><td>3.32</td><td>3.57</td><td>4.15</td><td>4.03</td><td>10.07</td><td>7.62</td><td>0.96</td></tr><tr><td>AuK</td><td>3.28</td><td>3.52</td><td>4.15</td><td>3.87</td><td>9.12</td><td>6.63</td><td>0.96</td></tr></table>

Table 14: Speech super-resolution performance on VCTK-SR.
<table><tr><td rowspan="2">Model</td><td colspan="3">DNSMOS</td><td rowspan="2">UTMOS↑</td><td rowspan="2">WER(%)↓</td><td rowspan="2">PER(%)↓</td><td rowspan="2">SIM↑</td></tr><tr><td>OVRL↑</td><td>SIG↑</td><td>BAK↑</td></tr><tr><td>AudioSR [39]</td><td>3.08</td><td>3.39</td><td>4.00</td><td>3.13</td><td>4.72</td><td>20.51</td><td>0.92</td></tr><tr><td>Resemble-Enhance [54]</td><td>3.18</td><td>3.45</td><td>4.09</td><td>3.59</td><td>15.94</td><td>19.61</td><td>0.95</td></tr><tr><td>AuK-Flash</td><td>3.25</td><td>3.51</td><td>4.12</td><td>4.05</td><td>2.92</td><td>4.19</td><td>0.96</td></tr><tr><td>AuK</td><td>3.21</td><td>3.47</td><td>4.08</td><td>3.93</td><td>3.06</td><td>4.72</td><td>0.97</td></tr></table>