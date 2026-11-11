# Walrus Technical Report

Alkis Koudounas Osamu Take Hayato Futami Quentin Jodelet Emiru Tsunoo

Speech & Language AI Lab Sony Group Corporation Tokyo, Japan

## Abstract

We present Walrus, a unified 8B speech-and-audio foundation model handling speech recognition (including code-switching and multi-talker speech), speech-totext and speech-to-speech translation, text-to-speech, emotion recognition, audio captioning, and general audio understanding. Our main contribution is not a new architecture but a data-centric training recipe; we use a standard decoderonly backbone initialized from Gemma 4, so that gains can be traced to the data. For speech generation, Walrus predicts low-bitrate discrete speech tokens, and a lightweight token-to-waveform decoder converts them into high-fidelity audio. The recipe is a three-stage data curriculum that moves from broad multilingual data to multi-talker, code-switching, and domain-specific data. At each stage, task-aware and dataset-aware weighting balances tasks with very different token budgets and protects core skills from forgetting. Trained on over 600M samples covering more than 20 tasks and 100 languages, Walrus leads on most multilingual, multi-talker, and code-switching ASR, speech translation, and audio understanding benchmarks, outperforming both open speech language models of comparable size and dedicated task-specific models. On most benchmarks, it even matches or surpasses the 4× larger Qwen3-Omni’s performance. Results suggest that data engineering is a primary driver of quality in multi-task speech and audio modeling.

Project Page: github.com/sony-speech/walrus

## 1 Introduction

Speech is a natural interface between people and machines, and real applications rarely need only one skill. A meeting assistant must transcribe overlapping speakers and keep track of who said what; a dubbing system must translate speech into another language and speak it back in a natural voice; a media platform must transcribe noisy entertainment audio and describe its music and sound effects. Across these settings, systems also need to recognize emotion, answer questions about audio, and handle speakers who switch languages mid-sentence [24, 37, 45, 63]. Training and serving a separate model for each task is costly, and the resulting pipelines pass errors from one stage to the next; a single model that covers all of these tasks is cheaper to deploy and simpler to maintain.

Speech Large Language Models (SpeechLLMs) are a promising path to such a model. Recent work improves them by scaling the model or redesigning the architecture [1, 11, 18, 21, 67]. Far less is reported about the data: how it is cleaned, mixed, and scheduled across tasks. This gap matters because the hardest cases are data problems: code-switched speech, overlapping speakers, entertainment audio, and reasoning-focused audio questions are rare in public corpora. Tasks also differ greatly in token budget, so a naive mixture lets long generative outputs dominate the loss and weakens core recognition. We take a different position: we argue that careful data engineering is a primary driver of quality, and that it can deliver state-of-the-art results without any new architecture. We demonstrate this by holding the architecture fixed and putting all design effort into the data.

![](images/0a7baa8fe653106b4d2ce2a17c395f569de9784873e0873c6519fcec5603fa7d.jpg)  
Figure 1: Overview of Walrus, our most capable SpeechLLM yet. Input speech is encoded by the USM encoder into continuous representations, which are passed to the decoder-only Gemma 4 backbone together with a natural-language task prompt. For recognition, translation, and understanding tasks, the backbone generates text. For TTS and S2ST, it generates discrete speech tokens at 25 Hz (325 bits per second), which a VQ decoder and a Vocos-based vocoder convert into a waveform. The vocoder is also conditioned on the source speech to transfer speaker identity and prosody.

We present Walrus, a unified speech-and-audio foundation model with about 8B parameters. Walrus is initialized from the Gemma 4 E4B model<sup>1</sup> [57] and keeps the backbone architecture unchanged. It supports automatic speech recognition (ASR), code-switching ASR (CS-ASR), multi-talker ASR (MT-ASR) with speaker diarization, machine translation (MT), automatic speech translation (AST), speech-to-speech translation (S2ST), text-to-speech (TTS), speech emotion recognition (SER), audio captioning, closed captioning, and general audio understanding. For speech generation, Walrus predicts low-bitrate discrete speech tokens at only 325 bits per second; a lightweight token-towaveform decoder converts these tokens into high-fidelity audio.

The core of our work is a three-stage data curriculum, where each stage adds new data and new tasks on top of a stable foundation. We pair the curriculum with task-aware and dataset-aware weighting to balance tasks with very different token budgets and keep core skills strong as the model learns harder tasks. From this curriculum, we build two variants: a Commercial-Ready (CR) model trained only on commercially usable data, and a Research (R) model trained on the full curriculum for the best overall quality. The recipe leads to large and consistent gains over strong open SpeechLLMs of comparable size. Figure 2 summarizes performance across recognition, translation, and understanding. Both Walrus variants push the frontier beyond the best previous system on most axes.

The main results are as follows.

Multilingual ASR. On VoxPopuli [64],<sup>2</sup> Walrus reaches 92.2% word accuracy (7.8 WER), against 88.2% for the strongest comparable system, Gemma 4 (11.8 WER). On FLEURS [7],<sup>3</sup> it reaches 94.7% word accuracy, ahead of 93.8% for the 4× larger Qwen3-Omni-30B [68] and 92.0% for Whisper Large-v3 [50].

Code-switching ASR. On CS-FLEURS [70] read speech, Walrus reaches 92.8% character accuracy, far ahead of 67.7% for Gemma 4 (25 points, 32.30 vs. 7.23 CER) and 61.3% for Qwen3-Omni.

![](images/4ffd9b377e8cb996cff982dc9f2f8cca0285817340a20c1ebc6f347def27a0a2.jpg)  
Figure 2: Walrus against the best previous system (among AF3, Gemma4, Qwen2.5-Omni-7B, Phi4-MM) across recognition, translation, and understanding benchmarks. Scores are normalized so that higher is better (ASR and CS-ASR shown as word/character accuracy, translation as BLEU or ASR-BLEU, understanding as accuracy, captioning as CLAP-Sim×100). Both Walrus variants advance the frontier on most axes, and the Research variant extends it further, most notably on code-switching, internal entertainment ASR, and audio understanding.

Multi-talker ASR. Comparable open SpeechLLMs cannot transcribe overlapping speakers and produce no valid output. On LibriMix clean 2-mix [8], Walrus reaches 19.32 cpWER, half of the 37.04 for Qwen3-Omni-30B and far below the 59.34 of the recent dedicated VibeVoice-ASR [46]. On Japanese Daily Talk [28], it reaches 56.29 cpCER, against 71.99 for VibeVoice-ASR.

Speech translation. On CVSS-C X→En [25], Walrus surpasses the much larger 30B Qwen3-Omni in S2ST (43.63 vs. 41.27 ASR-BLEU), while achieving very close performance on FLEURS X→En and En→X [7] (32.28/22.94 vs. 33.60/25.67, respectively). It achieves, however, better speaker similarity, prosody and emotion preservation on FLEURS X↔En compared to both larger Qwen3-Omni-30B and the dedicated SeamlessM4T v2 [3] (Table 19).

Audio understanding. On MMSU [66], Walrus reaches 64.1% accuracy, above 61.4% for Audio-Flamingo-3 [18] and 60.6% for Qwen2.5-Omni [67]. On MMAU-Pro [35], it reaches 71.92%, above closed-sourced models like Gemini-Flash-3.7 [58] (70.40%) and GPT-4o-Audio [20] (52.50%).

Moving from the Commercial-Ready to the Research variant improves every benchmark in Table 4. For example, it lowers the code-switching character error rate by 19 points and the internal entertainment WER by 9 points.

Our main contributions are as follows. First, we present a data-centric recipe for training a multitask speech and audio model on a fixed backbone. Second, we design a three-stage curriculum with task-aware and dataset-aware weighting, which grows from broad multilingual coverage to complex and domain-specific data. Third, we describe the data pipelines behind the hardest tasks:

synthetic code-switching, multi-talker mixtures, entertainment speech, conversational context, closed captioning, and audio reasoning. Finally, we show through extensive evaluation that this recipe reaches state-of-the-art results on most benchmarks, and we provide a commercial-ready variant trained only on commercially usable data.

## 2 Model

The Walrus architecture is deliberately simple; our contribution is the data strategy, not the network design. This section describes the backbone (Sec. §2.1, including the audio representation used for input and output and the token-to-waveform decoder, and the supported tasks (Sec. §2.2). Figure 1 shows how the components connect.

## 2.1 Architecture

Backbone. Walrus is initialized from Gemma 4 E4B,<sup>1</sup> a decoder-only Transformer with about 4B effective parameters (≈ 8B in total, including embeddings). We keep the architecture unchanged and fine-tune all parameters on our data curriculum, using scaled dot-product attention (SDPA) for efficient training and inference. A single set of weights serves every task, and tasks are selected through natural-language prompts.

Speech input. Walrus keeps the speech input design of Gemma 4 E4B; the Universal Speech Model (USM) encoder [73] converts the input speech into continuous representations, which are then fed to the decoder together with the text prompt. We unfreeze the encoder during fine-tuning, so that it adapts to our audio domains.

Speech output. For TTS and S2ST, Walrus generates discrete speech tokens autoregressively; we extend the backbone vocabulary with 8,192 dedicated speech tokens. Such tokens are then converted back into a waveform via a separate decoder, which we adopt from [16]. It consists of a single-layer vector quantization (VQ) tokenizer and a GAN-based vocoder. Decoupling token prediction from waveform synthesis keeps the backbone lightweight and lets the decoder specialize in high-fidelity generation and speaker transfer.

Speech VQ tokenizer. The tokenizer is a single-layer VQ module trained to reconstruct self-supervised speech features. The target features come from the 20th layer of a pre-trained W2v-BERT 2.0 encoder [3], which empirically best captures semantic content. A strided convolution reduces the feature rate from 50 Hz to 25 Hz; a three-layer Conformer [19] encodes the features, and a single codebook with 8,192 entries in a 512-dimensional space quantizes them. A symmetric Conformer decoder reconstructs the features. We train the module end-to-end with a VQ-VAE objective [61], which combines a cosine reconstruction loss and a commitment loss. The resulting stream has 25 tokens per second at 13 bits each (log 8192 = 13), which gives a bit-rate of 325 bits per second.

Vocoder. The GAN-based vocoder converts the VQ speech tokens into a waveform in a single non-autoregressive step. It builds on UniCATS (CTX-vec2wav) [13] and replaces the generator with Vocos [56] for higher synthesis quality. The VQ tokens are first mapped back to continuous features by the VQ decoder, and then converted onto waveform through the Vocos decoder. The vocoder is additionally conditioned on the source speech, using features from the 8th layer of W2v-BERT 2.0. Lower layers of the encoder encode speaker identity more strongly, so this conditioning transfers the source speaker identity and prosody to the output. It also relaxes the data requirement for expressive generation, because the model no longer needs source and target recordings with matched speaker and prosody. We train the vocoder as in [16], with multi-period and multi-band discriminators, a mel-spectrogram loss, a feature-matching loss [29, 34], and a multi-resolution spectral loss [69].

## 2.2 Supported tasks

Walrus handles a broad range of speech and audio tasks; Table 1 groups them into four families. Each task is specified through a natural-language prompt at both training and inference time. During training, we draw the prompt for each sample from a pool of task-specific templates, which improves robustness to prompt wording; Appendix §A lists the most frequently used prompt for each task.

The task families span the full pipeline from recognition to generation.

Table 1: Task families supported by Walrus, with language coverage. ASR and its variants support more than 100 languages. CS-ASR supports English-Japanese, and English or Japanese mixed with Chinese, French, German, Hindi, Italian, or Spanish (both as major and minor script). MT-ASR supports 2-, 3-, and 4-speaker mixtures in Chinese, English, French, German, Italian, Japanese, and Spanish. AST and S2ST support translation from 10 languages into English or Japanese, and from English or Japanese into the same 10 languages. Closed captioning supports Chinese, English, French, German, Italian, Japanese, and Spanish.
<table><tr><td>Task family</td><td>Supported tasks</td></tr><tr><td>Recognition</td><td>ASR, Context-ASR, Keyword-ASR, Timestamp-ASR,</td></tr><tr><td>Translation</td><td>CS-ASR, MT-ASR, Target-Speaker ASR, Diarization MT, AST (direct, cascaded), S2ST (direct, cascaded)</td></tr><tr><td>Generation</td><td>TTS</td></tr><tr><td>Understanding</td><td>Audio Captioning, Closed Captioning, Audio QA, Timestamp-QA, Audio Reasoning, Instruction Following, SER</td></tr></table>

Recognition. This family covers standard ASR together with all its conditioned variants. Contextconditioned ASR receives preceding text as discourse context, while keyword-conditioned ASR receives a list of candidate keywords to bias transcription toward salient entities, including names and domain terms. Timestamp-aware ASR predicts word or segment times, and a timestamp QA variant answers what is said within a given time span. The family also includes code-switching ASR (CS-ASR), which handles utterances that mix two languages within a single sentence and transcribes each word in the phonetics of its own language. Finally, it covers multi-talker ASR (MT-ASR) for audio with up to 4 speakers: overlapping-speech ASR that produces a speaker-attributed transcription, target-speaker ASR that transcribes only a chosen speaker, and speaker diarization. We also support chain-of-thought variants, which first estimate the number of speakers and then transcribe each one, together with timestamped multi-talker transcription.

Translation. This family covers text-to-text machine translation (MT), speech-to-text translation (AST), and speech-to-speech translation (S2ST). Both AST and S2ST are supported in direct and cascaded forms: the direct form maps source speech to the target in one step, while the cascaded form first transcribes and then translates (and eventually synthesizes). Context- and keyword-conditioned AST and S2ST variants carry the same discourse and entity hints as their recognition counterparts.

Generation. This family covers text-to-speech (TTS). Together with S2ST, generative tasks emit the low-bitrate discrete speech tokens described in Sec. §2.1, which the decoder then renders into audio.

Understanding. This family covers audio captioning, closed captioning (CC), audio question answering, audio reasoning, and speech emotion recognition (SER). Closed captioning spans several granularities, from a simple list of speech/non-speech events up to a full structured scene summary with speakers, sound effects, music, and timing (similar to [36]). Audio reasoning covers five skill families, spanning temporal, speech-in-sound, speaker, spatial, and instruction-following questions (Sec. §3.8.5).

## 3 Data curriculum

The data curriculum is the core of Walrus; we hold the architecture fixed, so that gains can be traced to the data. This section describes how we compose, weight, and stage the training data across all four task families. We first state our design principles and the resulting task distribution (Sec. §3.1-§3.2). We then describe the shared training setup (Sec. §3.3), the three stages (Sec. §3.4 to Sec. §3.6), and the two trained variants (Sec. §3.7). Finally, we describe the data-preparation pipelines (Sec. §3.8).

## 3.1 Design principles

Our data strategy follows three principles that apply to every task family.

![](images/60f0427b27719eb28878def2e88e7752ffd9a12991093a26ab2b49c0fe8137df.jpg)

(a) Overall task distribution across the four task families.  
![](images/e010787bbb195a30b2d171b6f17e94bc33f35f8a711c5184da1d9ec0077b598c.jpg)

![](images/a6127854b02dd93b241a628932974e1703bac3a390459651913774bd6493d1c1.jpg)

![](images/8e6ee96cc74bf47d657dd6c29b4387a631e3810e541f44b16daa2eb3b01c2fb5.jpg)  
(b) Detailed breakdown within the recognition, translation, and understanding families.

Figure 3: Training data composition of Walrus. Recognition dominates the mix (45%), followed by translation (27%) and understanding (21%), while generation is deliberately kept small (7%). The detailed breakdown shows that standard ASR and instruction-following data anchor the recognition and understanding families. This distribution reflects our data-centric design: core recognition is heavily represented to protect it from forgetting, while token-heavy generative tasks are downweighted.

Grow from broad and clean to complex and specialized. Early training uses well-known, highquality corpora that teach stable core skills while retaining Gemma 4’s foundation capabilities. Later training adds harder data, such as overlapping speech, code-switching, and in-domain entertainment synthetic audio. This order lets the model learn easy skills before it faces difficult ones.

Balance the mix with explicit weights. Tasks differ greatly in the number of tokens they produce: generative tasks such as TTS and S2ST emit long token sequences, so they can dominate the training loss; recognition tasks emit far fewer tokens, yet they form the backbone of the model. We therefore give generative tasks low weights in the early stages, and raise them only after the recognition core is stable.

Protect core skills at every stage. We keep foundation data in all three stages, and ASR keeps the highest task weight in every stage. As a result, adding new tasks in later stages does not erode the recognition skills built in the first stage.

## 3.2 Task distribution

Figure 3a shows the overall task distribution of the training data across the four families. Recognition is the largest family with 45%, followed by translation with 27%, understanding with 21%, and generation with 7%. This mix reflects our priorities: a dominant recognition core, strong translation and understanding coverage, and a smaller but sufficient share of generation.

The families are themselves diverse (Figure 3b); all percentages below are shares of the respective family. In recognition, standard ASR makes up 57.5% of the total amount, with the remainder spli across multi-talker ASR (16.4%), context ASR (14.3%), keyword ASR (10.7%), and code-switching ASR (1.1%). In translation, MT and S2ST each contribute 35.3%, AST accounts for 23.5%, and context- and keyword-conditioned variants make up the rest. In understanding, instruction following, audio QA, and reasoning represent 64.6%, followed by closed captioning (18.5%), audio captioning (15.4%), and emotion recognition (1.5%).

Table 2: Training schedule per stage. Each stage is initialized from the previous checkpoint. Warmup and decay are given as a fraction of total steps.
<table><tr><td>Stage</td><td>Steps</td><td>Learning rate</td><td>Warmup</td><td>Decay</td></tr><tr><td>1: Foundation</td><td>87,500</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>1%</td><td>5%</td></tr><tr><td>2: Expansion</td><td>135,000</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>1%</td><td>5%</td></tr><tr><td>3: Specialization</td><td>150,000</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>1%</td><td>5%</td></tr></table>

These breakdowns show that the mix is broad within each family, not concentrated in a single task.

## 3.3 Common setup

All three stages share the same optimization setup. We use the AdamW optimizer with $\beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 8 }$ , and a weight decay of 0.1. We clip the gradient norm to 1.0 and use scaled dotproduct attention (SDPA) for efficient training and inference. We stream the data from WebDataset shards. We use a per-device batch size of 8 with 2 gradient accumulation steps, and a maximum sequence length of 1,024 tokens; on 128 GPUs, this gives an effective batch size of 2,048 samples. Table 2 lists the learning rate and schedule of each stage.

We shape the data mix with two independent controls: task weights scale the contribution of each task to the loss, whereas dataset weights scale how often each dataset is sampled, so a weight of 0.5 halves the effective size of a corpus in the mixture. Together, the two controls let us set the effective data distribution independently of the raw corpus sizes. Table 3 summarizes the task weights of the three stages.

## 3.4 Stage 1: Foundation

The first stage builds a broad multilingual foundation across all four task families. It covers ASR, MT, AST, S2ST, TTS, SER, audio captioning, and audio QA, in many languages. It does not yet include timestamp ASR, code-switching ASR, multi-talker ASR, diarization, or closed captioning.

We draw on well-established public corpora and internal data. For speech recognition and translation, these comprehend CommonVoice (CV14) [2], VoxPopuli [64], LibriSpeech [43] and Multilingual LibriSpeech [49], FLEURS [7], CoVoST-2 [65], AISHELL-1 [4], CSJ [39], YODAS-Granary [51], and IndicVoices-ST [23]. For TTS, we use LibriTTS [72], VCTK [62], and AISHELL-3 [55]. We further leverage internal datasets for audio captioning, and a small subset of SIFT-50M [44] for instruction following. We clean the transcripts as described in Sec. §3.8.

The task weights make recognition the priority (Table 3). ASR has the highest weight (5.0), followed by context and keyword ASR (3.0). AST has a weight of 2.0, and MT and S2ST have a weight of 0.5 each. TTS has a weight of only 0.25, so that its long token sequences do not dominate the loss. Audio QA, audio captioning, and SER have a weight of 1.0 each. All datasets are sampled at their natural size, and we do not yet apply dataset weights. This stage teaches stable core skills before we introduce harder tasks.

## 3.5 Stage 2: Expansion

The second stage starts from the Stage-1 checkpoint and adds new tasks to every family. The foundation data stays in the mix with reduced dataset weights, so that the new data has room in the mixture.

In recognition, we add multi-talker and target-speaker ASR. The data comes from real meetings (AMI [33], AliMeeting [71], and AISHELL-4 [15]) and from simulated mixtures (LibriMix [8]). We also add timestamp prediction, diarization, and synthetic code-switching ASR. For understanding, we add synthetic closed captioning and richer audio QA, including additional SIFT-50M samples [44] and MECAT [42]. We also add the UltraVoice [60] and MSP-Podcast [6] corpora, which provide instruction-style speech and emotion data. Across recognition and translation, we add dialogue-level context tasks built from synthetic dialogues (Sec. §3.8.3). This stage teaches the model to handle overlapping speakers, mixed languages, timing, and conversational context.

Table 3: Task weights across the three stages. Weights scale each task’s contribution to the loss. A dash means the task is not yet present. Recognition is upweighted throughout to protect the core, while generation and understanding are raised in Stage 3.
<table><tr><td>Task Stage 1</td><td>Stage 2</td><td>Stage 3</td></tr><tr><td>Recognition ASR 5.0</td><td>5.0</td><td>4.0</td></tr><tr><td>Context / Keyword ASR 3.0 Timestamp ÁSR CS-ASR 1 MT-ASR 一</td><td>3.0 1.0 1.0 1.0</td><td>3.0 1.0 1.0 1.0</td></tr><tr><td>Diarization</td><td>1.0</td><td>1.0</td></tr><tr><td>Translation AST (and variants) 2.0 MT</td><td>2.0</td><td>2.0</td></tr><tr><td>0.5 S2ST (and variants) 0.5</td><td>0.5 0.5</td><td>0.5 2.0</td></tr><tr><td>Generation</td><td></td><td></td></tr><tr><td>TTS 0.25</td><td>0.25</td><td>2.0</td></tr><tr><td>Understanding</td><td></td><td></td></tr><tr><td>Audio QA / Reasoning</td><td>1.0</td><td>1.0</td></tr><tr><td>Audio captioning Closed captioning 1.0</td><td>1.0 1.0 1.0 1.0</td><td>2.0 2.0</td></tr></table>

We also introduce dataset weights in this phase. They let us lower the influence of large or redundant corpora, so that these corpora do not crowd out smaller, high-value data. For example, we halve the weight of very large ASR corpora such as Common Voice and YODAS-Granary, while giving the newly introduced tasks full weight. The task weights of the shared tasks stay the same as in Stage 1, and the new tasks get a task weight of 1.0; ASR therefore keeps the highest task weight.

## 3.6 Stage 3: Specialization

The third stage starts from the Stage-2 checkpoint; it keeps the data of the first two stages and adds new data that closes specific gaps and covers target domains. This also includes corpora that are not commercially usable.

In recognition, we add synthetic entertainment speech, low-resource languages, and more codeswitching data. We also add a large number of synthetic multi-talker mixtures with two to four speakers, covering seven languages: Chinese, English, French, German, Italian, Japanese, and Spanish. In translation, we add more S2ST data: SpeechMatrix [14], Europarl-ST [22], and internal dubbing-aligned data built with SpeechVecAlign [40]. For closed captioning, we add the remaining segment libraries (Sec. §3.8.5).

We halve the learning rate to $1 \times 1 0 ^ { - 6 }$ , because a model with strong core skills needs refinement rather than large updates. We also rebalance the task weights to strengthen generation and understanding: we lower the ASR weight from 5.0 to 4.0, while raising the weights of TTS (0.25 to 2.0), S2ST (0.5 to 2.0), audio captioning, closed captioning, and SER (each 1.0 to 2.0). The ASR weight stays the highest, so recognition remains the core of the model. Sec. §3.8 describes the data-preparation and filtering pipelines for these corpora.

## 3.7 Training scale and variants

Walrus is trained on over 600M samples that cover more than 20 task variants and 100 languages. The run has three consecutive stages; each one is initialized from the previous checkpoint and uses a different random seed, which changes the data shuffling. Training takes about 30 days on 128 NVIDIA H100 GPUs, which is about 92,000 GPU hours.

We derive two variants from the curriculum. The Commercial-Ready (CR) variant is the Stage-2 checkpoint, trained only on commercially usable data, making it suitable for deployment. The Research (R) variant is the Stage-3 final checkpoint; it uses all data, including non-commercial corpora, and delivers the best overall quality, achieving state-of-the-art performance.

The two variants share the same backbone, but differ in training stage, data license, and task weights. The research variant improves on the commercial-ready variant on every benchmark in Table 4. The largest gains are in CS-ASR (about 19 points lower CER on CS-FLEURS read speech), internal entertainment ASR (about 9 points lower average WER across 6 languages), and MMAU-Pro (more than 20 points higher accuracy). These gains come from Stage 3, which adds high-quality data while rebalancing the task weights.

## 3.8 Data preparation

Raw corpora are rarely usable as they are, and public data is missing for several of our tasks. We therefore clean the source data and build task-specific supervision for every task family; where suitable data does not exist at scale, we generate it with dedicated pipelines. This applies to codeswitching, multi-talker conversation, context- and keyword-conditioned recognition and translation, closed captioning, and reasoning-focused audio QA.

The following subsections describe these pipelines. Sec. §3.8.1 describes transcript cleaning, which applies to all speech corpora. Sec. §3.8.2 covers recognition, Sec. §3.8.3 summarizes synthetic dialogues, Sec. §3.8.4 covers translation, and Sec. §3.8.5 highlights understanding. Appendix §A lists example task prompts, and Appendix §B summarizes all pipelines in one table.

## 3.8.1 Transcript cleaning

We apply two steps to the transcripts of our speech corpora: restoration of casing and punctuation, and text normalization. Clean and readable text benefits every task that uses these transcripts.

Casing and punctuation restoration. Many ASR corpora provide only lowercase transcripts without punctuation. A model trained on such text learns to drop casing and punctuation, which hurts readability and downstream tasks. We restore both with a large language model, Qwen3.5-9B,<sup>4</sup> used as a text normalizer. For each transcript, we build a few-shot prompt with language-specific examples. The prompt has one strict rule: add casing and punctuation only, and never add, remove, or change a word. We run the restoration separately for each language, and we skip transcripts that already contain mixed case.

A language model can silently rewrite words, so we verify every output: we remove casing and punctuation from the original and the restored transcript, and we compute the character error rate between them. We keep a restored transcript only if this CER is at most 0.05: this limits the edits to the surface form and filters out outputs where the model changed the content. We also discard outputs whose length differs too much from the original, as a second guard against hallucination. The filter retains the large majority of samples (90-95% for most datasets) while removing the few cases where the normalizer altered content.

Text normalization. After restoration, we apply a normalization pipeline with separate rules for three language groups: European languages, CJK languages (Chinese, Japanese, and Korean), and others. These scripts differ in spacing, punctuation, and digit conventions, so each group needs its own rules; we tuned the rules by manually inspecting samples from each group. The pipeline gives consistent and readable references across the multilingual mix.

## 3.8.2 Recognition supervision

This subsection describes the data for the recognition tasks that need dedicated pipelines. Context ASR and keyword ASR are built from cleaned ASR corpora. Code-switching ASR is fully synthetic, and multi-talker ASR combines real and simulated recordings. Entertainment speech combines curated available data with synthetic augmentation. Sec. §3.8.3 adds more context- and keywordconditioned data from synthetic dialogues.

Context ASR. Standard ASR treats each utterance on its own, but real speech depends on what was said before. We build context ASR data from the document structure of read-speech corpora. We group utterances by chapter (for MLS [49]) or by domain (for internal datasets), and sort them in reading order. For each target utterance, we randomly attach between one and five preceding utterances as text context. The model is then asked to transcribe the target utterance given this context. This teaches the model to use prior discourse, which helps with proper nouns, topic-specific vocabulary, and consistency across a sequence of utterances.

Keyword ASR. In keyword-biased ASR, the model receives a list of candidate keywords together with the audio. We extract candidate keywords from transcripts using named-entity recognition, keeping people, places, organizations, and other proper nouns. We normalize and de-duplicate the entities, and keep only those that appear in at least two utterances, which removes rare noise. For each utterance, the keyword list contains the entities that the utterance actually includes (positives) and randomly sampled entities that it does not contain (distractors). We draw the list size from {0, 10, 20, 50, 100}, so that the model sees short and long keyword lists, as well as no list at all. We further shuffle each list, so that the position of a keyword carries no information. This teaches the model to use helpful keywords and to ignore irrelevant ones, which matters for domain-specific speech.

Code-switching ASR. Code-switched speech, where a speaker mixes two languages in one utterance, is rare in public corpora. We therefore generate both the text and the audio in three steps: code-switched text generation, segment-based speech synthesis, and quality filtering. We cover 13 language pairs, with a focus on English and Japanese mixed with European and Asian languages.

Text generation. We generate the sentences with a large language model, Gemma 4 26B.<sup>5</sup> For each language pair, we prompt the model to write one natural sentence that is mostly in a matrix language with short embedded phrases from the other language. The prompt enforces realistic switching: it allows switches only where a bilingual speaker would use them, such as technical terms, loanwords, or emotional expressions, and limits each switch to a few words. We guide the model with hand-written few-shot examples for each language pair and each switching direction. To increase diversity, we vary the topic across 25 everyday domains, such as work, travel, food, and technology. After generation, we remove duplicate sentences so that each text is unique.

Speech synthesis. A single-language TTS model cannot pronounce a mixed sentence reliably, because each word must follow the phonetics of its own language; we therefore synthesize each sentence in monolingual segments.<sup>6</sup> First, we detect which words belong to the embedded language, using a per-language word list built from the generated data. Next, we group consecutive words of the same language into segments, and we merge very short segments into their neighbors to avoid irregular, discontinuous audio. Then, we synthesize each segment with the correct language tag, and we join the segments with a random short silence gap. Every word in the resulting audio carries the phonetics of its own language.

Speaker control. For each utterance, we sample one reference speaker from a pool of high-quality FLEURS speakers of the matrix language. We select the pool with the NISQA quality score [41], and we keep only clean, high-fidelity recordings. We clone the reference voice for all segments of the utterance, so that the whole utterance has one consistent speaker. Sampling a new speaker for each utterance gives broad speaker variety across the dataset.

Quality filtering. We apply two filters to the synthesized speech. First, we score the naturalness of each utterance with NISQA, and we discard utterances below a 3.50 threshold. Second, we check the intelligibility; we transcribe each utterance with Whisper Large-v3 Turbo<sup>7</sup> [50] and compute the WER against the intended text (or CER, depending on the language). We discard utterances whose error rate is above a 5% threshold. The remaining audio and its known code-switched text form the training pairs.

Multi-talker ASR. Real-world conversations and media content involve multiple speakers with overlapping speech. We build data for four related tasks: overlapping-speech ASR with speakerattributed output, target-speaker ASR, speaker diarization, and timestamped variants of these tasks. With this data, the model learns to transcribe the content, to separate the speakers, and to predict when each speaker talks. All clips are at most 30 seconds long, which is the input limit of Walrus.

We combine real and simulated conversations. The real recordings come from AMI [33], AliMeeting [71], and AISHELL-4 [15]. We split the full recordings into 30-second clips with the DiCoW recipe<sup>8</sup> [47]. For simulated conversations with overlapped speech, we rely on public recipes and further scale the data with FastMSS [48]. We first adopt the public LibriMix [8] and LibriSpeech Mix [26] corpora, then use FastMSS to cut single-speaker utterances at the word level and to rearrange them into multilingual conversations with two, three, or four speakers, including overlap and turntaking. The single-speaker sources are LibriSpeech for English, an internal corpus for Japanese, and MLS for French, Spanish, German, and Italian. Because MLS and the internal Japanese corpus lack the word- and character-level alignments that FastMSS requires, we generate them with Qwen3-ForcedAligner-0.6B<sup>9</sup> [54].

Entertainment speech. Entertainment audio, such as anime, games, movies, and TV, is a key target domain, yet clean labeled data for it is scarce. We build this data in two ways. First, we curate in-domain speech from large, permissively licensed collections such as YODAS-Granary [51], and we pass it through the cleaning and filtering pipeline of Sec. §3.8.1. Second, we synthesize entertainment-style speech by augmenting existing corpora with domain-characteristic effects, such as high-pitched or stylized voices, expressive prosody, and non-speech vocal and background sounds. This exposes the model to the acoustic conditions of entertainment content that read-speech corpora rarely contain. We use the resulting data for ASR and, because the synthetic effects and background events are known by construction, also for the closed-captioning pipeline of Sec. §3.8.5.

## 3.8.3 Conversational supervision via synthetic dialogues

Real conversational data with reliable context, translations, and keyword annotations is scarce. We therefore build synthetic multi-turn dialogues, following the framework of [31]. The dialogues provide context- and keyword-conditioned supervision for ASR, AST, and S2ST. The pipeline has five steps: dialogue generation, dialogue-level filtering, speech synthesis, quality gating, and task construction.

Dialogue generation. We generate dialogues between two instruction-tuned LLMs across 40 everyday domains, such as travel, food, technology, and health. An emotion-chain manager drives each dialogue, so that emotions evolve naturally from turn to turn and stay appropriate to the domain. We use three model pairings to increase diversity: Gemma 4 26B<sup>5</sup>–Gemma 4 26B, Qwen3. $5 ~ 9 B ^ { 4 } { - } G$ emma 4 26B, and Qwen3.5 9B–Qwen3.5 9B. Each dialogue has two to five turns, because smaller models tend to hallucinate in longer dialogues [31, 38]. Each textual dialogue is then translated from English to 11 different languages using Qwen3.5 27B.<sup>10</sup>

Dialogue-level filtering. We filter the generated dialogues with an ensemble of LLM judges, as in [31]. Each judge gives a binary coherence decision and a five-point quality rating. We keep a dialogue only if all judges find it coherent and give it the highest rating.

Speech synthesis. We synthesize each turn with the in-house Chatterbox variant described in Sec. §3.8.2. Speaker voices are cloned from Common Voice (CV14) reference clips [2]. Each dialogue has two distinct speakers, balanced by age and gender; we use fallback strategies when the pool lacks a matching speaker, which keeps the speaker pool diverse.

Quality gating. We apply strict quality gates to every synthesized turn. We transcribe the turn with Whisper Large-v3 Turbo<sup>7</sup> and compute the WER or CER, depending on the language. For translation tasks, we also score the target text with a reference-free quality estimator, which differs depending on the training stage. In Stage 2, we use an LLM judge (Gemma 4 26B),<sup>5</sup> which we prompt to output a score on the same 0-1 scale as COMET-Kiwi [53]. In Stage 3, we use COMET-Kiwi directly.<sup>11</sup> We keep a turn only if its error rate is at most 0.1 and its translation quality score (COMET-Kiwi or its LLM equivalent) is at least 0.80. These gates keep only clean and faithful turns.

Task construction. We build context- and keyword-conditioned tasks from the filtered turns. For keyword tasks, we extract salient entities (people, locations, and organizations) with a multilingual named-entity recognizer, then present them together with distractor keywords in a list of ten to fifty items. For context tasks, we instead prepend the previous turns of the dialogue. This gives data for six tasks: context ASR/AST/S2ST, keyword ASR/AST/S2ST. Translation tasks are generated in both direct and cascaded forms. This gives the model rich supervision for handling conversational context and salient keywords, which standard corpora rarely provide.

## 3.8.4 Translation supervision

Beyond the synthetic dialogues, we build a large pool of parallel data for MT, AST, and S2ST. We first mine and filter parallel text, then synthesize speech for the retained text pairs, which gives the supervision for AST and S2ST.

Parallel text mining. We mine parallel text from the OPUS collection [59]. Because our target includes a commercial-ready model, we first restrict OPUS to datasets with clear, permissive licenses.<sup>12</sup> We extract the source-target sentence pairs and shuffle them, which removes any corpusor document-level ordering. Mined parallel data is noisy, so we score every pair for cross-lingual semantic similarity. We embed both sides with a multilingual e5 sentence encoder<sup>13</sup> and compute the cosine similarity of the two embeddings. For each language pair, we sort the pairs by this score and keep the top decile: this removes most misaligned, partial, and low-quality translations. We also drop pairs whose source sentence starts with a non-letter character, because these are usually markup, list bullets, or formatting artifacts.

Voice selection. We synthesize the speech with the in-house Chatterbox variant described above. For each side of a pair, we pick a reference voice in the correct language from the FLEURS speakers. We pair a source-side voice with the most similar target-side voice, so that both languages are spoken by acoustically similar speakers. We measure the similarity with ECAPA-TDNN speaker embeddings [10, 52],<sup>14</sup> and keep only voice pairs whose similarity is above a per-language threshold.

Speech synthesis and verification. We synthesize the source and target sentences with the selected voices. Then, we transcribe every synthesized utterance with Whisper Large-v3 Turbo<sup>7</sup> and align the transcript with the intended text. This check detects trailing hallucinations and audio that continues after the intended text ends. We discard the utterances where the recognizer output differs from the reference; the remaining triples of source audio, target text, and target audio supervise MT, AST, and S2ST.

## 3.8.5 Understanding supervision

We construct two kinds of data for the understanding family. Closed captioning data is synthesized by mixing labeled audio segments. AudioReason-10M is a large corpus of reasoning-focused audio question answering.

Closed captioning: scene composition. Rich closed captions describe not only speech but also music, sound effects, speaker turns, and timing, yet no large corpus provides all of these annotations jointly. We therefore build each audio scene from labeled segments and derive its caption from the known composition: every event, timestamp, and speaker label in the caption is exact with respect to the labels of the segments.

We draw segments from three libraries: speech, sound effects (SFX), and music. Each segment has a label, a source (speaker) identifier, and a duration. We sample segments with a simple transition model that alternates between speech and sound events; we place them one after another on a timeline, with randomized blank gaps and optional overlap. Each scene lasts between 10 and 30 seconds, and we balance the corpus across four duration bands.

Closed captioning: speakers and acoustics. We control speaker composition and acoustic realism of each scene explicitly. We can enforce a target number of distinct speakers per scene, drawing speech segments from speakers with enough utterances and encouraging natural speaker turns. Segments are mixed with a separate intensity range for each type: speech is prominent, SFX are attenuated, and music is quiet. We apply linear fade-in and fade-out and loudness normalization, and resize SFX segments at random to vary their duration. Optionally, we add a quiet background music track over the whole scene. The result is varied multi-speaker scenes with music and sound effects.

Closed captioning: caption formats. We generate several caption formats from the composition of each scene, one for each annotation task; a single scene therefore supervises several levels of detail. The formats range from a transcription with bracketed non-speech events, to timestamped events, to timestamped and speaker-attributed captions. The most detailed format is a structured summary, counting the music, speech, and SFX segments, listing the unique speakers, and describing each event with its label and time span, similar to [36]. Prompts for each task are shown in Table 20.

Closed captioning: languages and stages. We cover Chinese, English, French, German, Italian, Japanese, and Spanish. We use a separate segment library for each language, built from opensource [2, 9, 17, 32] and internal datasets. We split this data across two stages: in Stage 2, we use only commercially licensed segment libraries; in Stage 3, we augment the mix with the rest.

AudioReason-10M: reasoning-focused audio QA. Audio QA built from recognition-oriented benchmarks, such as ASR or sound event classification, does not train expert-level auditory reasoning [18]. Following the approach of Audio Flamingo 3 [18], we build a large reasoning-focused corpus, which we call AudioReason-10M. It contains 10M audio QA pairs, most of them multiplechoice, on clips of at most 30 seconds. We build it only from commercially licensed or internally owned audio, and we exclude research-only and license-restricted corpora.

We organize AudioReason-10M around five reasoning skill families, each requiring capabilities that go beyond simple transcription or single-label classification.

• Temporal reasoning. Questions require reasoning over the order, duration, and co-occurrence of events, for example which sound comes first, how long an event lasts, or whether two events overlap.

• Speech-in-sound. The model must combine the spoken content with the surrounding acoustic scene, so that speech and non-speech evidence lead to one answer.

• Speaker reasoning. Questions probe speaker attributes such as gender and approximate age, as well as the number of distinct speakers and their turns, requiring paralinguistic rather than lexical understanding.

• Spatial audio perception. Questions ask about the direction, relative position, and movement of sound sources, which requires spatial cues in the audio.

• Instruction following with verifiable constraints. Each question has an explicit output constraint, such as a format, a length, or an answer set, whose satisfaction can be checked automatically; this teaches the model to follow instructions precisely while reasoning about the audio.

For each skill, we design dedicated prompts and QA templates. We generate distractors for the multiple-choice questions, with the aim of reducing shortcuts that do not need the audio. All skills share one QA schema, so that we can mix them uniformly during training.

Table 4: Performance comparison across all evaluated benchmarks. For each dataset, we report the best prior system among comparable 4-10B models, including Gemma4, Phi4-MM, AF3, Q2.5O (Prior SOTA), the Commercial-Ready Walrus (Walrus-CR), and the Research Walrus (Walrus-R). Metrics: WER/CER (↓) for ASR tasks, cpWER/cpCER (↓) for MT-ASR, BLEU/ASR-BLEU (↑) for AST/S2ST, accuracy (↑) for IF and SER, CIDEr/CLAP-Sim (↑) for captioning, R@1/5/10 (↑) for retrieval. Averages over languages; per-language breakdowns are in Appendix §C. Phi4-MM results on MLS and FLEURS are reported as averages across supported languages, i.e., with Ar/Hi/Ko/Nl/Pl excluded. Best results per dataset in bold.
<table><tr><td>Task</td><td>Dataset</td><td>Metric</td><td>Prior SOTA</td><td>Walrus-CR</td><td>Walrus-R</td></tr><tr><td colspan="6">Recognition</td></tr><tr><td>ASR</td><td>LS Clean/Other [43]</td><td>WER↓</td><td>1.57/3.13 (AF3)</td><td>1.88/4.99</td><td>1.65/3.85</td></tr><tr><td>ASR</td><td>MLS [49]</td><td>WER↓</td><td>8.20 (Phi4-MM)</td><td>9.14</td><td>6.89</td></tr><tr><td>ASR</td><td>FLEURS [7]</td><td>WER↓</td><td>4.00 (Phi4-MM)</td><td>7.54</td><td>5.33</td></tr><tr><td>ASR</td><td>VoxPopuli [64]</td><td>WER↓</td><td>11.78 (Gemma4)</td><td>8.38</td><td>7.84</td></tr><tr><td>ASR</td><td>Internal Entertainment (avg)</td><td>WER↓</td><td>24.35 (Gemma4)</td><td>26.71</td><td>17.45</td></tr><tr><td>CS-ASR</td><td>CS-FLEURS Read/XTTS [70]</td><td>CER↓</td><td>32.30/44.10 (Gemma4)</td><td>26.69/35.36</td><td>7.23/15.01</td></tr><tr><td>MT-ASR MT-ASR</td><td>LibriMix 2Mix [8]</td><td>cpWER↓</td><td>N/A</td><td>26.84</td><td>19.32</td></tr><tr><td></td><td>JA Daily Talk [28]</td><td>cpCER↓</td><td>N/A</td><td>66.60</td><td>56.29</td></tr><tr><td colspan="6">Translation</td></tr><tr><td>AST X→En</td><td>CoVoST-2 [65]/FLEURS [7]</td><td>BLEU↑</td><td>39.33/29.86 (Phi4-MM)</td><td>40.28/30.21</td><td>42.63/33.05</td></tr><tr><td>AST En→X</td><td>FLEURS [7]</td><td>BLEU↑</td><td>32.15 (Phi4-MM)</td><td>32.42</td><td>34.33</td></tr><tr><td>S2ST X→En</td><td>CVSS-C [25]/FLEURS [7]</td><td>ASR-BLEU ↑</td><td>36.02/25.15 (Q250)</td><td>43.27/28.59</td><td>43.63/32.28</td></tr><tr><td>S2ST En→X</td><td>FLEURS [7]</td><td>ASR-BLEU ↑</td><td>N/A</td><td>17.19</td><td>22.94</td></tr><tr><td colspan="6">Understanding</td></tr><tr><td>IF</td><td>MMAU-Pro [35]</td><td>ACC↑</td><td>52.20 (Q250)</td><td>50.18</td><td>71.92</td></tr><tr><td>IF</td><td>MMSU [66]</td><td>ACC↑</td><td>61.40 (AF3)</td><td>55.18</td><td>64.14</td></tr><tr><td>SER</td><td>IEMOCAP [5]</td><td>ACC↑</td><td>63.80 (AF3)</td><td>41.44</td><td>49.55</td></tr><tr><td>Captioning</td><td>Clotho-v2 [12]</td><td>CIDEr/CLAP-Sim ↑</td><td>0.50/0.41 (AF3)</td><td>0.10/0.19</td><td>0.39/0.35</td></tr><tr><td>Retrieval A2T</td><td>AudioCaps [27]</td><td>R@1/5/10↑</td><td>0.20/0.60/0.76 (AF3)</td><td>0.16/0.52/0.67</td><td>0.18/0.63/0.78</td></tr><tr><td>Retrieval T2A</td><td>AudioCaps [27]</td><td>R@1/5/10↑</td><td>0.21/0.59/0.76 (AF3)</td><td>0.16/0.48/0.65</td><td>0.22/0.60/0.78</td></tr></table>

## 4 Results

We evaluate Walrus on recognition, translation, and understanding benchmarks. Table 4 compares Walrus with the best prior system among comparable 4-10B models: Gemma 4 E4B (Gemma4)<sup>1</sup> [57], Phi-4-Multimodal (Phi4-MM)<sup>15</sup> [1], Audio-Flamingo-3 (AF3)<sup>16</sup> [18], and Qwen2.5-Omni (Q2.5O)<sup>17</sup> [67]. We report both variants, the Commercial-Ready model (Walrus-CR, Stage 2) and the Research model (Walrus-R, Stage 3). All scores are averaged over languages; per-language breakdowns are in Appendix §C, where we also compare our models with larger (Qwen3-Omni-30B (Q3O) [68]) or dedicated systems, such as Whisper Large-v3 (WLv3), SeamlessM4T v2 (Seamless) [3], and VibeVoice-ASR [46].

Multilingual and English ASR. On English LibriSpeech, Walrus-R reaches 1.65/3.85 WER on clean/other, slightly behind AF3 (1.57/3.13). On multilingual benchmarks, it has the lowest WER on MLS (6.89 against 8.20 for Phi4-MM, Table 8) and on VoxPopuli (7.84 against 11.78 for Gemma4, Table 10). On FLEURS (Table 9), it reaches 5.33 WER (94.7% word accuracy, Figure 2); this is ahead of the 4×-bigger Qwen3-Omni (93.8%) and Whisper Large-v3 (92.0%). Phi4-MM reaches 4.00 WER, but it does not support five of our evaluation languages (Arabic, Dutch, Hindi, Korean, and Polish; see Table 9). On internal entertainment audio (Table 12), Walrus-R reaches 17.45 WER, against 24.35 for Gemma4.

Code-switching ASR. On CS-FLEURS read speech, Walrus-R reaches 7.23 CER, against 32.30 for the best prior system (Gemma4). On the XTTS subset, it reaches 15.01 CER, against 44.10. Walrus-CR reaches 26.69 and 35.36, which is already better than every prior system. Both systems outperform even the 30B Q3O model, which achieves only 38.70 CER on read speech (Table 13).

Multi-talker ASR. Table 14 shows the results on overlapping speech. Gemma4, Phi4-MM, AF3, and Q2.5O cannot transcribe overlapping speakers, and they produce no valid output. On LibriMix clean

2-mix, Walrus-R reaches 19.32 cpWER; this is about half of the 37.04 of Qwen3-Omni, and far below the 59.34 of the dedicated VibeVoice-ASR. On Japanese Daily Talk, it reaches 56.29 cpCER, against 71.99 for VibeVoice-ASR. Walrus-CR also beats VibeVoice-ASR on the cp metrics. The time-constrained metrics (tcpWER and tcpCER) give the same ranking for Walrus-R. Two points limit this comparison: (i) we evaluate on 30-second segments, which match the input limit of Walrus but not the long-form design of VibeVoice-ASR; and (ii) LibriMix is also part of our training data, so it is in-domain for Walrus. The absolute error on Japanese Daily Talk is still high, thus showing how natural multi-speaker conversation still remains a hard task.

Speech translation. On AST, Walrus-R reaches 42.63 BLEU on CoVoST-2 (Table 15) and 33.05 on FLEURS X→En (Table 16), against 39.33 and 29.86 for Phi4-MM. On FLEURS En→X, it reaches 34.33 BLEU, against 32.15. On S2ST, Walrus is the strongest system: it achieves 43.63 ASR-BLEU on CVSS-C X→En and 32.28 on FLEURS X→En, against 36.02 and 25.15 for Qwen2.5-Omni. On CVSS-C (Table 17), it also surpasses the 30B Qwen3-Omni, while being close on FLEURS (Table 18). It also supports En→X S2ST (22.94 ASR-BLEU on FLEURS), where the other systems report no result. Walrus-CR is already ahead of the prior systems on AST and S2ST X→En, but its gains are smaller on both tasks.

Audio understanding. Walrus-R is the strongest system on audio understanding. On MMAU-Pro, it reaches 71.92% accuracy, far above the best baseline (52.20% for Qwen2.5-Omni), and also closedsource systems such as for Gemini-Flash-3.7 (70.40%), and GPT-4o-Audio (52.50%). On MMSU, it reaches 64.14%, against 61.40% for AF3. On captioning and retrieval, Walrus is competitive with the best specialized baseline (AF3), leading on several AudioCaps retrieval metrics and closing most of the gap on Clotho-v2 captioning. It trails AF3 on IEMOCAP emotion recognition (49.55% against 63.80%, OOD for our model), which reflects the small share of emotion data in our mix.

Commercial-Ready and Research variants. Walrus-R improves on Walrus-CR on every benchmark in Table 4. The largest gains are on CS-FLEURS read speech (26.69 to 7.23 CER), internal entertainment ASR (26.71 to 17.45 WER), and MMAU-Pro (50.18% to 71.92%). The two variants share the backbone, but they differ in the training stage, data license, and task weights (Sec. §3.7). The gains therefore reflect Stage 3 as a whole, and we do not separate the effect of each change.

## 5 Discussion

Walrus uses a standard backbone, so we discuss its results in terms of the data. We highlight three findings that the benchmark tables alone do not make explicit.

Targeted data goes with targeted gains. The largest gains appear where we built dedicated data. CS-ASR is only 1.1% of the recognition data, which is about 0.5% of all data. Still, the CER on CS-FLEURS read speech falls from 32.30 for the best prior system to 26.69 for Walrus-CR and 7.23 for Walrus-R. Entertainment-domain ASR follows the same pattern: the internal entertainment WER falls from 26.71 to 17.45 between the two variants, and the Japanese subset falls from 23.69 to 12.90 (Appendix §C, Table 12). MMAU-Pro accuracy rises from 50.18% to 71.92% between the two variants.

Weighting and recognition. Stage 3 raises the weights of generation and understanding and lowers the ASR weight from 5.0 to 4.0. Even so, Walrus-R improves on Walrus-CR on every ASR benchmark. This is consistent with the idea that a high recognition weight protects the core while the mix shifts. However, as we did not run Stage 3 data with the Stage 2 weights, we cannot fully separate the effect of the weights from the effect of the new data.

Remaining gaps. Walrus trails the best baseline on emotion recognition and audio captioning. On the former, it reaches 49.55% on IEMOCAP (OOD) against 63.80% for AF3. This partly reflect the distribution of SER data in our mix (about 0.3%). On audio captioning, it achieves 0.39 CIDEr on Clotho-v2 (OOD) against 0.50 (AF3), while captioning represents 15.4% of the understanding data. Furthermore. on Japanese Daily Talk, Walrus-R reaches 56.29 cpCER, which beats VibeVoice-ASR but is still high in absolute terms. Our multi-talker training data is mostly simulated, and real overlapping speech is harder.

## 6 Conclusion

We presented Walrus, a unified 8B speech-and-audio model that covers recognition, translation, generation, and understanding. Its core is a three-stage curriculum, from broad multilingual data to complex and domain-specific data. Task-aware and dataset-aware weighting balances the very different token budgets of the tasks. Trained on over 600M samples across more than 20 tasks and 100 languages, Walrus-R matches or surpasses comparable open SpeechLLMs on most benchmarks, and leads clearly on code-switching ASR, multilingual and entertainment-domain recognition, speechto-speech translation, and audio understanding. On several benchmarks, it also outperforms larger or dedicated systems such as Qwen3-Omni-30B, Seamless v2, and VibeVoice-ASR. These results suggest that data engineering is a primary driver of quality in multi-task speech and audio modeling.

## References

[1] A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary, C. Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743, 2025.

[2] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the twelfth language resources and evaluation conference, pages 4218–4222, 2020.

[3] L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, M. Duppenthaler, P.-A. Duquenne, B. Ellis, H. Elsahar, J. Haaheim, et al. Seamless: Multilingual expressive and streaming speech translation. arXiv preprint arXiv:2312.05187, 2023.

[4] H. Bu, J. Du, X. Na, B. Wu, and H. Zheng. Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In 2017 20th conference of the oriental chapter of the international coordinating committee on speech databases and speech I/O systems and assessment (O-COCOSDA), pages 1–5. IEEE, 2017.

[5] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan. Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42(4):335–359, 2008.

[6] C. Busso, R. Lotfian, K. Sridhar, A. N. Salman, W.-C. Lin, L. Goncalves, S. Parthasarathy, A. R. Naini, S.-G. Leem, L. Martinez-Lucas, et al. The msp-podcast corpus. IEEE Transactions on Affective Computing, 2026.

[7] A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, and A. Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805. IEEE, 2023.

[8] J. Cosentino, M. Pariente, S. Cornell, A. Deleforge, and E. Vincent. Librimix: An open-source dataset for generalizable speech separation. arXiv preprint arXiv:2005.11262, 2020.

[9] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840, 2016.

[10] B. Desplanques, J. Thienpondt, and K. Demuynck. ECAPA-TDNN: emphasized channel attention, propagation and aggregation in TDNN based speaker verification. In H. Meng, B. Xu, and T. F. Zheng, editors, Interspeech 2020, pages 3830–3834. ISCA, 2020.

[11] D. Ding, Z. Ju, Y. Leng, S. Liu, T. Liu, Z. Shang, K. Shen, W. Song, X. Tan, H. Tang, et al. Kimi-audio technical report. arXiv preprint arXiv:2504.18425, 2025.

[12] K. Drossos, S. Lipping, and T. Virtanen. Clotho: an audio captioning dataset. In Proc. IEEE Int. Conf. Acoustic., Speech and Signal Process. (ICASSP), pages 736–740, 2020.

[13] C. Du, Y. Guo, F. Shen, Z. Liu, Z. Liang, X. Chen, S. Wang, H. Zhang, and K. Yu. Unicats: A unified context-aware text-to-speech framework with contextual vq-diffusion and vocoding. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 17924–17932, 2024.

[14] P.-A. Duquenne, H. Gong, N. Dong, J. Du, A. Lee, V. Goswami, C. Wang, J. Pino, B. Sagot, and H. Schwenk. Speechmatrix: A large-scale mined corpus of multilingual speech-to-speech translations. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16251–16269, 2023.

[15] Y. Fu, L. Cheng, S. Lv, Y. Jv, Y. Kong, Z. Chen, Y. Hu, L. Xie, J. Wu, H. Bu, et al. Aishell-4: An open source dataset for speech enhancement, separation, recognition and speaker diarization in conference scenario. arXiv preprint arXiv:2104.03603, 2021.

[16] H. Futami, H. Shahmohammadi, T. Dhyani, A. Koudounas, R. Lafargue, Y. Kashiwagi, Q. Jodelet, and E. Tsunoo. Kraken: Llm-based speech-to-speech translation via low-bitrate vq and dual-path source conditioning. arXiv preprint arXiv:2609.13045, 2026.

[17] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780, 2017.

[18] S. Ghosh, A. Goel, J. Kim, S. Kumar, Z. Kong, S.-g. Lee, C.-H. Yang, R. Duraiswami, D. Manocha, R. Valle, et al. Audio flamingo 3: Advancing audio intelligence with fully open large audio language models. Advances in Neural Information Processing Systems, 38:41819–41886, 2026.

[19] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, et al. Conformer: Convolution-augmented transformer for speech recognition. arXiv preprint arXiv:2005.08100, 2020.

[20] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[21] A. A. G. Intelligence. Amazon Nova 2: Multimodal reasoning and generation models. Technical report, Technical Report. Amazon. https://www. amazon. science/publications/amazon . . . , 2025.

[22] J. Iranzo-Sánchez, J. A. Silvestre-Cerda, J. Jorge, N. Roselló, A. Giménez, A. Sanchis, J. Civera, and A. Juan. Europarl-st: A multilingual corpus for speech translation of parliamentary debates. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8229–8233. IEEE, 2020.

[23] T. Javed, J. Nawale, E. George, S. Joshi, K. Bhogale, D. Mehendale, I. Sethi, A. Anantha narayanan, H. Faquih, P. Palit, et al. Indicvoices: Towards building an inclusive multilingual speech dataset for indian languages. In Findings of the Association for Computational Linguistics: ACL 2024, pages 10740–10782, 2024.

[24] S. Ji, Y. Chen, M. Fang, J. Zuo, J. Lu, H. Wang, Z. Jiang, L. Zhou, S. Liu, X. Cheng, et al. Wavchat: A survey of spoken dialogue models. arXiv preprint arXiv:2411.13577, 2024.

[25] Y. Jia, M. T. Ramanovich, Q. Wang, and H. Zen. Cvss corpus and massively multilingual speech-to-speech translation. In Proceedings ofthe thirteenth language resources and evaluation conference, pages 6691–6703, 2022.

[26] N. Kanda, Y. Gaur, X. Wang, Z. Meng, and T. Yoshioka. Serialized output training for end-to-end overlapped speech recognition. In Proc. Interspeech, pages 2797–2801, 2020.

[27] C. D. Kim, B. Kim, H. Lee, and G. Kim. AudioCaps: Generating captions for audios in the wild. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.

[28] H. Koiso, H. Amatani, Y. Den, Y. Iseki, Y. Ishimoto, W. Kashino, Y. Kawabata, K. Nishikawa, Y. Tanaka, Y. Usuda, and Y. Watanabe. Design and evaluation of the corpus of everyday Japanese conversation. In N. Calzolari, F. Béchet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi,

H. Isahara, B. Maegaard, J. Mariani, H. Mazo, J. Odijk, and S. Piperidis, editors, Proceedings ofthe Thirteenth Language Resources and Evaluation Conference, pages 5587–5594, Marseille, France, June 2022. European Language Resources Association.

[29] J. Kong, J. Kim, and J. Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in neural information processing systems, 33:17022–17033, 2020.

[30] A. Koudounas, H. Futami, Q. Jodelet, O. Take, S. Watanabe, and E. Tsunoo. Benchmarking speech-to-speech translation models. arXiv preprint arXiv:2606.03241, 2026.

[31] A. Koudounas, M. La Quatra, and E. Baralis. Deepdialogue: A multi-turn emotionally-rich spoken dialogue dataset. arXiv preprint arXiv:2505.19978, 2025.

[32] A. Koudounas, M. La Quatra, L. Vaiani, L. Colomba, G. Attanasio, E. Pastor, L. Cagliero, and E. Baralis. ITALIC: An Italian Intent Classification Dataset. In Interspeech 2023, pages 2153–2157, 2023.

[33] W. Kraaij, T. Hain, M. Lincoln, and W. Post. The ami meeting corpus. In Proc. International Conference on Methods and Techniques in Behavioral Research, pages 1–4, 2005.

[34] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar. High-fidelity audio compression with improved rvqgan. Advances in Neural Information Processing Systems, 36:27980–27993, 2023.

[35] S. Kumar, Š. Sedlácek, V. Lokegaonkar, F. López, W. Yu, N. Anand, H. Ryu, L. Chen, M. Pliˇ cka,ˇ M. Hlavácek, et al. Mmau-pro: A challenging and comprehensive benchmark for holisticˇ evaluation of audio general intelligence. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 22688–22697, 2026.

[36] S. Kumar, P. Seetharaman, K. Chen, O. Nieto, J. Su, Z. Wang, R. Kumar, D. Manocha, N. J. Bryan, Z. Jin, et al. Tac: Timestamped audio captioning. arXiv preprint arXiv:2602.15766, 2026.

[37] M. La Quatra, A. Koudounas, L. Vaiani, E. Baralis, L. Cagliero, P. Garza, and S. M. Siniscalchi. Benchmarking representations for speech, music, and acoustic events. In 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW), pages 505–509, 2024.

[38] P. Laban, H. Hayashi, Y. Zhou, and J. Neville. Llms get lost in multi-turn conversation. In International Conference on Learning Representations, volume 2026, pages 54738–54778, 2026.

[39] K. Maekawa. Corpus of spontaneous Japanese: its design and evaluation. In ISCA/IEEE Workshop on Spontaneous Speech Processing and Recognition, page paper MMO2, 2003.

[40] C. Meng and P. Koehn. Speech vecalign: an embedding-based method for aligning parallel speech documents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 16489–16505, 2025.

[41] G. Mittag, B. Naderi, A. Chehadi, and S. Möller. NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets. In Interspeech 2021, pages 2127–2131, 2021.

[42] Y. Niu, T. Wang, H. Dinkel, X. Sun, J. Zhou, G. Li, J. Liu, X. Liu, J. Zhang, and J. Luan. Mecat: A multi-experts constructed benchmark for fine-grained audio understanding tasks. arXiv preprint arXiv:2507.23511, 2025.

[43] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. Librispeech: An asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210, 2015.

[44] P. Pandey, R. V. Swaminathan, K. V. Girish, A. Sen, J. Xie, G. Strimel, and A. Schwarz. Sift-50m: A large-scale multilingual dataset for speech instruction fine-tuning. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13921–13942, 2025.

[45] J. Peng, Y. Wang, B. Li, Y. Guo, H. Wang, Y. Fang, Y. Xi, H. Li, X. Li, K. Zhang, S. Wang, and K. Yu. A survey on Speech Large Language Models for Understanding. IEEE Journal of Selected Topics in Signal Processing, 20(1):2–31, 2026.

[46] Z. Peng, J. Yu, Y. Chang, Z. Wang, L. Dong, Y. Hao, Y. Tu, C. Yang, W. Wang, S. Xu, et al. Vibevoice-asr technical report. arXiv preprint arXiv:2601.18184, 2026.

[47] A. Polok, D. Klement, M. Kocour, J. Han, F. Landini, B. Yusuf, M. Wiesner, S. Khudanpur, J. Cernocký, and L. Burget. DiCoW: Diarization-conditioned Whisper for target speaker<sup>ˇ</sup> automatic speech recognition. Computer Speech & Language, 95:101841, 2026.

[48] A. Polok, I. Medennikov, H. Cernocký, S. Watanabe, L. Burget, and S. Cornell. Mind the<sup>ˇ</sup> gap: Impact of synthetic conversational data on multi-talker ASR and speaker diarization. In Interspeech 2026, pages 294–299, 2026.

[49] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411, 2020.

[50] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.

[51] N. Rao Koluguri, M. Sekoyan, G. Zelenfroynd, S. Meister, S. Ding, S. Kostandian, H. Huang, N. Karpov, J. Balam, V. Lavrukhin, Y. Peng, S. Papi, M. Gaido, A. Brutti, and B. Ginsburg. Granary: Speech Recognition and Translation Dataset in 25 European Languages. In Interspeech 2025, pages 3923–3927, 2025.

[52] M. Ravanelli, T. Parcollet, P. Plantinga, A. Rouhe, S. Cornell, L. Lugosch, C. Subakan, N. Dawalatabad, A. Heba, J. Zhong, J.-C. Chou, S.-L. Yeh, S.-W. Fu, C.-F. Liao, E. Rastorgueva, F. Grondin, W. Aris, H. Na, Y. Gao, R. D. Mori, and Y. Bengio. SpeechBrain: A general-purpose speech toolkit, 2021. arXiv:2106.04624.

[53] R. Rei, M. Treviso, N. M. Guerreiro, C. Zerva, A. C. Farinha, C. Maroti, J. G. De Souza, T. Glushkova, D. Alves, L. Coheur, et al. Cometkiwi: Ist-unbabel 2022 submission for the quality estimation shared task. In Proceedings ofthe Seventh Conference on Machine Translation (WMT), pages 634–645, 2022.

[54] X. Shi, X. Wang, Z. Guo, Y. Wang, P. Zhang, X. Z. Z. Guo, H. Hao, Y. Xi, B. Yang, J. Xu, J. Zhou, and J. Lin. Qwen3-ASR technical report. arXiv preprint arXiv:2601.21337, 2026.

[55] Y. Shi, H. Bu, X. Xu, S. Zhang, and M. Li. Aishell-3: A multi-speaker mandarin tts corpus and the baselines. arXiv preprint arXiv:2010.11567, 2020.

[56] H. Siuzdak. Vocos: Closing the gap between time-domain and fourier-based neural vocoders for high-quality audio synthesis. arXiv preprint arXiv:2306.00814, 2023.

[57] G. Team, S. E. Abd, V. Aggarwal, R. Algayres, A. Andreev, O. Bachem, I. Ballantyne, C. Brick, V. Carbune, M. Casbon, et al. Gemma 4 technical report. ˘ arXiv preprint arXiv:2607.02770, 2026.

[58] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[59] J. Tiedemann and L. Nygaard. The opus corpus-parallel and free: http://logos. uio. no/opus. In Proceedings of the Fourth International Conference on Language Resources and Evaluation (LREC’04), 2004.

[60] W. Tu, G. Yang, R. Yan, W. Chen, Z. Ma, Y. Kang, K. Yu, X. Chen, and Z. Zheng. Ultravoice: Scaling fine-grained style-controlled speech conversations for spoken dialogue models. arXiv preprint arXiv:2510.22588, 2025.

[61] A. Van Den Oord, O. Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

[62] C. Veaux, J. Yamagishi, and S. King. The voice bank corpus: Design, collection and data analysis of a large regional accent speech database. In 2013 International Conference Oriental COCOSDA held jointly with 2013 Conference on Asian Spoken Language Research and Evaluation (O-COCOSDA/CASLRE), pages 1–4, 2013.

[63] B. Wang, X. Zou, G. Lin, S. Sun, Z. Liu, W. Zhang, Z. Liu, A. Aw, and N. F. Chen. AudioBench: A universal benchmark for audio large language models. In L. Chiruzzo, A. Ritter, and L. Wang, editors, Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4297–4316, Albuquerque, New Mexico, Apr. 2025. Association for Computational Linguistics.

[64] C. Wang, M. Riviere, A. Lee, A. Wu, C. Talnikar, D. Haziza, M. Williamson, J. Pino, and E. Dupoux. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 993–1003, 2021.

[65] C. Wang, A. Wu, and J. Pino. Covost 2 and massively multilingual speech-to-text translation. arXiv preprint arXiv:2007.10310, 2020.

[66] D. Wang, J. Li, J. Wu, D. Yang, X. Chen, T. Zhang, and H. Meng. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. In International Conference on Learning Representations, volume 2026, pages 31374–31410, 2026.

[67] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

[68] J. Xu, Z. Guo, H. Hu, Y. Chu, X. Wang, J. He, Y. Wang, X. Shi, T. He, X. Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

[69] R. Yamamoto, E. Song, and J.-M. Kim. Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In ICASSP 2020- 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6199–6203. IEEE, 2020.

[70] B. Yan, I. Hamed, S. Shimizu, V. S. Lodagala, W. Chen, O. Iakovenko, B. Talafha, A. Hussein, A. Polok, K. Chang, D. Klement, S. Althubaiti, P. Peng, M. Wiesner, T. Solorio, A. Ali, S. Khudanpur, and S. Watanabe. CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset. In Interspeech 2025, pages 743–747, 2025.

[71] F. Yu, S. Zhang, Y. Fu, L. Xie, S. Zheng, Z. Du, W. Huang, P. Guo, Z. Yan, B. Ma, X. Xu, and H. Bu. M2met: The icassp 2022 multi-channel multi-party meeting transcription challenge. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6167–6171, 2022.

[72] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu. LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech. In Interspeech 2019, pages 1526–1530, 2019.

[73] Y. Zhang, W. Han, J. Qin, Y. Wang, A. Bapna, Z. Chen, N. Chen, B. Li, V. Axelrod, G. Wang, et al. Google usm: Scaling automatic speech recognition beyond 100 languages. arXiv preprint arXiv:2303.01037, 2023.

## A Task prompts

Walrus is steered to each of its tasks only through a natural-language prompt. This appendix describes how we design, sample, and use these prompts. Table 20 lists the canonical prompt of each task.

## A.1 Prompt pools

We do not use a fixed instruction for each task, because the model could overfit to a single phrasing. Instead, we keep a pool of 10-20 paraphrased prompts for every task, and we sample one prompt from the pool for each training example. This encourages robust instruction following, since the model must read the instruction instead of recognizing a fixed string.

The prompts in a pool differ in register and in format.

• Imperative prompts give a direct command, for example “Transcribe the {lang1} speech:”.

• Interrogative prompts ask a question, for example “What is being said in the speech?”.

• Tag-style prompts use a compact marker, for example “[ASR {lang1}]”.

• Structured prompts use explicit delimiters, for example <context>...</context>.

All prompts of a task ask for the same output; only the wording and the format change, so the paraphrases never change the task itself.

## A.2 Slot variables

Prompts are templates with slot variables (Table 5); we fill such slots with the values of each sample when building the example. One template thus serves many languages, contexts, and label sets.

Table 5: Slot variables used in the prompt templates.
<table><tr><td>Slot</td><td>Filled with</td></tr><tr><td>{1ang1}, {1ang2}</td><td>Source and target language</td></tr><tr><td>{context}</td><td>Preceding text, as discourse context</td></tr><tr><td>{keywords}</td><td>List of candidate keywords</td></tr><tr><td>{labels}</td><td>Closed set of allowed labels, for example emotions</td></tr><tr><td>{speaker}</td><td>Target speaker for target-speaker queries</td></tr><tr><td>{start}, {end}</td><td>Time span for time-conditioned queries</td></tr><tr><td>{agent}, {user}</td><td>Roles in dialogue-level context</td></tr></table>

## A.3 Sampling and canonical prompts

We sample uniformly from each pool, except for one prompt that we mark as canonical, for which we give a higher sampling probability. Table 20 lists these canonical prompts of each task.

Each pool has a fixed flag, which selects the canonical prompt deterministically. We use this flag at evaluation time, so that results do not depend on random prompt choice and can be reproduced.

## A.4 Prompts for conditioned tasks

Many tasks need more than a language name, so their prompts carry extra inputs. The structure of these inputs is the same across the variants of a task.

• Context ASR and context AST/S2ST. The prompt contains the preceding text in a marked block, such as <context>...</context>. For dialogues, the source and target sides of earlier turns are marked, and the roles come from {agent} and {user}.

• Keyword tasks. The prompt contains a shuffled list of candidate keywords. The list may contain distractors and may be empty, as described in Sec. §3.8.2.

• Timestamp and multi-talker tasks. The prompt states the requested output, such as speaker labels or timestamps. For target-speaker ASR and timestamp QA, it also contains the target speaker or the time span.

Table 6: Overview of the data pipelines. “Tools” lists the models and recipes used, “Quality control” the filters applied to the output. A dash means that we apply no filter beyond the construction itself.
<table><tr><td>Pipeline</td><td>What we build</td><td>Tools</td><td>Quality control</td></tr><tr><td colspan="3">Transcript cleaning</td><td colspan="2">CER ≤ 0.05 against the original; length check</td></tr><tr><td>Casing and punctuation Text normalization</td><td>Restored transcripts of real Qwen3.5-9B with few-shot ASR corpora Consistent references across languages</td><td>prompts Rules for European, CJK, and Hindi/Arabic scripts</td><td>Manual inspection of samples</td></tr><tr><td colspan="4">Recognition</td></tr><tr><td>Context ASR</td><td>Target utterance with 1-5 preceding utterances</td><td>MLS chapters, internal domains</td><td></td></tr><tr><td>Keyword ASR</td><td>Keyword lists of 0-100 items with distractors</td><td>Named-entity recognition</td><td>Entities in at least two utterances</td></tr><tr><td>Code-switching ASR</td><td>Synthetic text and audio, 13 language pairs</td><td>Gemma 4 26B, Qwen3-TTS, Chatterbox (internal), FLEURS reference voices</td><td>De-duplication; NISQA threshold; Whisper WER/CER threshold</td></tr><tr><td>Multi-talker ASR Real meeting clips and</td><td>simulated 2- to 4-speaker mixtures</td><td>DiCoW recipe, FastMSS, Qwen3-ForcedAligner</td><td>Clips of at most 30 seconds</td></tr><tr><td>Entertainment speech</td><td>Curated web speech and augmented corpora</td><td>YODAS-Granary, audio effects</td><td>Same cleaning pipeline as other ASR data</td></tr><tr><td colspan="4">Translation and dialogues</td></tr><tr><td>Synthetic dialogues</td><td>Multi-turn dialogues with context and keyword tasks</td><td>Gemma 4 26B, Qwen3.5-9B, Chatterbox variant, Whisper, COMET-Kiwi or LLM judge</td><td>Judge ensemble; error rate ≤ 0.1; quality score ≥ 0.80</td></tr><tr><td>Parallel text and speech</td><td>MT, AST, and S2ST triples multilingual e5, Chatterbox from OPUS</td><td>(internal), ECAPA-TDNN, Whisper</td><td>Top decile by similarity; voice-similarity threshold; Whisper check</td></tr><tr><td colspan="4">Understanding</td></tr><tr><td>Closed captioning</td><td>Mixed scenes of 10-30 seconds with exact captions libraries</td><td>Speech, SFX, and music segment Several constraints enforced</td><td>during creation, manual inspection of early samples</td></tr><tr><td>AudioReason- 10M</td><td>10M reasoning QA pairs, five skills</td><td>Dedicated prompts and QA templates</td><td>Commercially licensed or internal audio only; constraints enforced in QA synthesis</td></tr></table>

• Emotion recognition. The prompt lists the allowed labels with {labels}, so that the answer is one of a closed set.

• Closed captioning. Each level of detail has its own prompt, from a plain transcription with non-speech events to the full structured summary (Sec. §3.8.5).

• Translation and generation. The prompt names the source and target language. It also states whether the output is text or speech, and whether the translation is direct or cascaded.

## A.5 Prompts at inference

At inference time, a user can write a prompt in any form, not only the canonical one. We report all results with the canonical prompts of Table 20. Since each task is trained with many paraphrases, we expect the model to follow new wordings; however, we do not measure this robustness in this report.

## B Summary of data pipelines

Sec. §3.8 describes the data pipelines of Walrus in five subsections; this appendix gives a short overview of all of them in one place. Table 6 lists, for each pipeline, what we build, which tools we use, and which quality controls we apply. The rows follow the order of Sec. §3.8.

How to read the table. The column “What we build” names the supervision that the pipeline produces; the column “Tools” names the corpora, models, and recipes that the pipeline uses; and the column “Quality control” names the filters that we apply to the output. A dash means that we apply no filter beyond the construction itself.

Table 7: ASR Results: LibriSpeech (WER ↓). WLv3 indicates Whisper-Large-v3, AF3 Audio-Flamingo3, Q250/Q30 denote Qwen2.5Omni and Qwen3Omni, respectively. Best results in bold, second-best underlined.
<table><tr><td>Model</td><td>Clean</td><td>Other</td></tr><tr><td>WLv3</td><td>2.31</td><td>4.29</td></tr><tr><td>Gemma4</td><td>3.22</td><td>7.74</td></tr><tr><td>Phi4-MM</td><td>1.68</td><td>3.83</td></tr><tr><td>AF3</td><td>1.57</td><td>3.13</td></tr><tr><td>Q250</td><td>1.83</td><td>3.44</td></tr><tr><td>Q30</td><td>1.22</td><td>2.48</td></tr><tr><td>Walrus-CR</td><td>1.88</td><td>4.99</td></tr><tr><td>Walrus-R</td><td>1.65</td><td>3.85</td></tr></table>

Table 8: ASR Results: MLS (WER ↓). \*This average score reflects only the supported languages.
<table><tr><td>Model</td><td>DE</td><td>EN</td><td>ES</td><td>FR</td><td>IT</td><td>NL</td><td>PL</td><td>PT</td><td>AVG</td></tr><tr><td>WLv3</td><td>5.54</td><td>5.30</td><td>4.01</td><td>4.60</td><td>9.09</td><td>10.36</td><td>4.89</td><td>7.20</td><td>6.37</td></tr><tr><td>Gemma4</td><td>8.71</td><td>9.15</td><td>5.43</td><td>7.86</td><td>14.63</td><td>12.06</td><td>10.44</td><td>9.10</td><td>9.67</td></tr><tr><td>Phi4-MM</td><td>7.59</td><td>8.04</td><td>10.49</td><td>5.15</td><td>11.63</td><td>N/A</td><td>N/A</td><td>6.29</td><td>8.20*</td></tr><tr><td>AF3</td><td>27.45</td><td>5.02</td><td>12.57</td><td>20.07</td><td>19.96</td><td>28.46</td><td>53.50</td><td>17.26</td><td>23.04</td></tr><tr><td>Q250</td><td>6.54</td><td>6.14</td><td>8.21</td><td>7.92</td><td>20.27</td><td>17.56</td><td>19.21</td><td>11.96</td><td>12.23</td></tr><tr><td>Q30</td><td>3.77</td><td>3.77</td><td>3.41</td><td>3.00</td><td>9.72</td><td>8.94</td><td>12.06</td><td>5.22</td><td>6.24</td></tr><tr><td>Walrus-CR</td><td>7.88</td><td>6.43</td><td>4.44</td><td>5.67</td><td>14.52</td><td>11.53</td><td>13.63</td><td>9.02</td><td>9.14</td></tr><tr><td>Walrus-R</td><td>5.88</td><td>5.91</td><td>3.47</td><td>4.46</td><td>9.86</td><td>10.22</td><td>7.71</td><td>7.67</td><td>6.89</td></tr></table>

Real and synthetic data. Our pipelines use real and synthetic data in different ways. Some pipelines only restructure real data: context ASR and keyword ASR add context or keyword inputs to real ASR corpora, and the casing restoration changes only the text surface. Some pipelines recombine real audio: simulated multi-talker mixtures rearrange real single-speaker utterances, and closed-captioning scenes mix real speech, sound-effect, and music segments. Other pipelines generate new content: code-switching ASR and the synthetic dialogues generate both text and audio, and the translation pipeline synthesizes speech for mined text pairs, while AudioReason-10M builds new questions and answers over audio.

This difference matters for the reliability of the labels. In recombined data, the labels follow from the construction, but they are only as accurate as the labels of the source segments. In generated data, the text is the label, but the audio can differ from it if the TTS system makes an error. We therefore check the audio of generated data, as described next.

Table 9: ASR Results: FLEURS (WER/CER ↓). \*This average score reflects only the supported languages.
<table><tr><td>Model</td><td>AR</td><td>DE</td><td>EN</td><td>ES</td><td>FR</td><td>HI</td><td>IT</td><td>JA</td><td>KO</td><td>PT</td><td>ZH</td><td>AVG</td></tr><tr><td>WLv3</td><td>16.40</td><td>5.96</td><td>6.24</td><td>4.26</td><td>6.96</td><td>18.61</td><td>4.07</td><td>6.55</td><td>4.34</td><td>5.55</td><td>9.03</td><td>7.99</td></tr><tr><td>Gemma4</td><td>16.20</td><td>6.10</td><td>6.50</td><td>3.50</td><td>8.00</td><td>8.60</td><td>3.20</td><td>7.80</td><td>5.30</td><td>4.60</td><td>13.60</td><td>7.58</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>3.96</td><td>3.38</td><td>3.02</td><td>4.35</td><td>N/A</td><td>1.98</td><td>4.50</td><td>N/A</td><td>3.98</td><td>6.83</td><td>4.00*</td></tr><tr><td>AF3</td><td>N/A</td><td>22.53</td><td>5.06</td><td>25.83</td><td>39.82</td><td>74.45</td><td>19.74</td><td>N/A</td><td>54.65</td><td>49.65</td><td>N/A</td><td>36.47</td></tr><tr><td>Q250</td><td>18.75</td><td>7.67</td><td>6.46</td><td>5.70</td><td>5.08</td><td>14.54</td><td>7.63</td><td>8.12</td><td>7.03</td><td>6.50</td><td>7.29</td><td>8.62</td></tr><tr><td>Q30</td><td>14.84</td><td>5.02</td><td>7.36</td><td>4.27</td><td>5.10</td><td>7.59</td><td>3.94</td><td>4.85</td><td>3.31</td><td>5.25</td><td>6.78</td><td>6.21</td></tr><tr><td>Walrus-CR</td><td>13.48</td><td>6.03</td><td>5.49</td><td>4.16</td><td>8.51</td><td>10.51</td><td>4.06</td><td>7.99</td><td>7.37</td><td>5.03</td><td>10.35</td><td>7.54</td></tr><tr><td>Walrus-R</td><td>10.56</td><td>4.34</td><td>4.50</td><td>2.94</td><td>4.84</td><td>4.94</td><td>2.55</td><td>7.64</td><td>4.84</td><td>3.87</td><td>7.70</td><td>5.33</td></tr></table>

Table 10: ASR Results: VoxPopuli (WER ↓). 5 major EU languages are considered.
<table><tr><td>Model</td><td>DE</td><td>EN</td><td>ES</td><td>FR</td><td>IT</td><td>AVG</td></tr><tr><td>WLv3</td><td>18.26</td><td>9.89</td><td>10.19</td><td>9.98</td><td>27.21</td><td>15.11</td></tr><tr><td>Gemma4</td><td>13.76</td><td>8.67</td><td>8.79</td><td>10.89</td><td>16.80</td><td>11.78</td></tr><tr><td>Phi4-MM</td><td>12.13</td><td>5.91</td><td>10.18</td><td>13.18</td><td>22.67</td><td>12.81</td></tr><tr><td>AF3</td><td>108.73</td><td>5.55</td><td>95.36</td><td>91.02</td><td>142.65</td><td>86.66</td></tr><tr><td>Q250</td><td>15.84</td><td>6.64</td><td>13.24</td><td>11.39</td><td>23.67</td><td>14.16</td></tr><tr><td>Q30</td><td>11.60</td><td>7.66</td><td>8.35</td><td>8.06</td><td>15.73</td><td>10.28</td></tr><tr><td>Walrus-CR Walrus-R</td><td>7.54 6.77</td><td>4.37 4.82</td><td>5.72 4.95</td><td>9.15 8.42</td><td>15.14 14.23</td><td>8.38 7.84</td></tr></table>

Table 11: ASR Results: CoVoST-2 (WER/CER ↓). \*This average score reflects only the supported languages.
<table><tr><td>Model</td><td>DE</td><td>ES</td><td>FR</td><td>IT</td><td>JA</td><td>PT</td><td>RU</td><td>ZH</td><td>AVG</td></tr><tr><td>WLv3</td><td>6.78</td><td>4.19</td><td>12.28</td><td>5.10</td><td>10.46</td><td>4.52</td><td>4.21</td><td>16.37</td><td>7.99</td></tr><tr><td>Gemma4</td><td>9.33</td><td>5.55</td><td>14.58</td><td>7.10</td><td>16.46</td><td>8.16</td><td>9.16</td><td>47.37</td><td>14.71</td></tr><tr><td>Phi4-MM</td><td>10.82</td><td>3.57</td><td>6.05</td><td>3.53</td><td>36.91</td><td>10.26</td><td>N/A</td><td>48.29</td><td>17.49*</td></tr><tr><td>AF3</td><td>28.22</td><td>14.52</td><td>22.27</td><td>18.23</td><td>139.96</td><td>18.72</td><td>95.60</td><td>143.12</td><td>60.08</td></tr><tr><td>Q250</td><td>6.67</td><td>7.51</td><td>8.44</td><td>8.62</td><td>19.85</td><td>12.88</td><td>7.95</td><td>10.69</td><td>10.33</td></tr><tr><td>Q3O</td><td>4.71</td><td>4.26</td><td>6.91</td><td>4.37</td><td>8.87</td><td>6.83</td><td>3.51</td><td>4.70</td><td>5.52</td></tr><tr><td>Walrus-CR</td><td>7.01</td><td>4.12</td><td>8.43</td><td>5.21</td><td>11.61</td><td>5.70</td><td>5.28</td><td>18.49</td><td>8.23</td></tr><tr><td>Walrus-R</td><td>5.81</td><td>3.62</td><td>7.35</td><td>4.09</td><td>10.30</td><td>4.74</td><td>3.78</td><td>11.34</td><td>6.38</td></tr></table>

Quality control. The quality controls in Table 6 fall into three groups.

• Textfilters. These filters check the text before any audio is made. Examples are the CER check after casing restoration (Sec. §3.8.1), the de-duplication of code-switched sentences, the LLM-judge ensemble for dialogues (Sec. §3.8.3), and the top-decile similarity filter for parallel text (Sec. §3.8.4).

• Audio filters. These filters check the synthesized audio itself. Examples are the NISQA threshold for code-switching speech and the speaker-similarity threshold for voice pairs.

• Text-audio checks. These filters check that the audio says the intended text. We transcribe the synthesized speech with Whisper Lv3-Turbo and compare the transcript with the intended text; we use this check for code-switching speech, dialogues, and translation speech.

Translation quality of the dialogues is checked in a fourth way. We score the target text with a reference-free estimator: an LLM judge in Stage 2 and COMET-Kiwi in Stage 3 (Sec. §3.8.3).

## Models used to build the data. The tools in Table 6 play four roles.

• Text generation and judging. Gemma 4 26B and Qwen3.5-9B write code-switched sentences and dialogues, restore casing, and act as judges; Qwen3.5-27B translates dialogues from English onto different languages.

• Speech synthesis. Qwen3-TTS and a fast in-house variant of Chatterbox create the audio for code-switching, dialogues, and translation.

• Recognition and alignment. Whisper Lv3-Turbo checks synthesized speech, and Qwen3-ForcedAligner provides word-level alignments for multi-talker mixtures.

• Scoring. NISQA scores speech quality, ECAPA-TDNN scores speaker similarity, multilingual-e5 scores cross-lingual text similarity, and COMET-Kiwi scores translation quality.

Pipelines and training stages. Not every pipeline is present in every stage. Stage 1 uses cleaned real corpora and does not use the synthetic pipelines for code-switching, multi-talker ASR, or closed

Table 12: ASR Results: Internal entertainment data (WER/CER ↓). Per-language averages over different test sets.
<table><tr><td>Model</td><td>DE</td><td>EN-UK</td><td>EN-US</td><td>ES</td><td>FR</td><td>IT</td><td>JA</td></tr><tr><td>WLv3</td><td>22.33</td><td>8.31</td><td>11.10</td><td>11.67</td><td>22.53</td><td>13.56</td><td>24.77</td></tr><tr><td>Gemma4 Phi4-MM</td><td>39.20 69.58</td><td>16.86 17.44</td><td>17.99 20.25</td><td>17.72 26.91</td><td>34.96 66.01</td><td>18.35 22.40</td><td>25.37 202.35</td></tr><tr><td>AF3</td><td>63.90</td><td>16.05</td><td>17.74</td><td>47.20</td><td>88.49</td><td>41.51</td><td>169.11</td></tr><tr><td>Q250</td><td>34.54</td><td>18.55</td><td>22.72</td><td>30.55</td><td>37.68</td><td>28.55</td><td>43.18</td></tr><tr><td>Q30</td><td>27.11</td><td>11.85</td><td>11.14</td><td>15.50</td><td>21.95</td><td>13.00</td><td>25.88</td></tr><tr><td>Walrus-CR</td><td>39.71</td><td>15.94</td><td>17.42</td><td>30.62</td><td>59.16</td><td>24.20</td><td>23.69</td></tr><tr><td>Walrus-R</td><td>27.68</td><td>14.53</td><td>14.26</td><td>15.15</td><td>24.55</td><td>14.01</td><td>11.97</td></tr></table>

Table 13: CS-ASR Results: CS-FLEURS Read (top) and XTTS (bottom) test sets, CER (↓). \*This average score reflects only the supported languages.
<table><tr><td>Model</td><td>EN-HI</td><td>EN-JA</td><td>EN-DE</td><td>EN-FR</td><td>EN-ZH</td><td>AVG</td></tr><tr><td colspan="7">Read Test Set</td></tr><tr><td>WLv3</td><td>42.75</td><td>35.41</td><td>6.10</td><td>12.21</td><td>17.58</td><td>22.81</td></tr><tr><td>Gemma4</td><td>47.50</td><td>48.07</td><td>19.18</td><td>18.17</td><td>28.56</td><td>32.30</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>69.29</td><td>29.15</td><td>32.78</td><td>81.71</td><td>53.23*</td></tr><tr><td>AF3</td><td>97.95</td><td>85.09</td><td>64.50</td><td>73.62</td><td>86.69</td><td>81.57</td></tr><tr><td>Q250</td><td>56.96</td><td>63.68</td><td>41.66</td><td>37.21</td><td>58.42</td><td>51.58</td></tr><tr><td>Q30</td><td>35.74</td><td>66.47</td><td>38.64</td><td>19.41</td><td>33.02</td><td>38.66</td></tr><tr><td>Walrus-CR Walrus-R</td><td>28.80 7.88</td><td>44.84 12.91</td><td>15.52 3.63</td><td>17.80 3.68</td><td>26.48 8.05</td><td>26.69 7.23</td></tr><tr><td colspan="7">XTTS Test Set</td></tr><tr><td>WLv3</td><td>48.63</td><td>49.80</td><td>9.44</td><td>11.07</td><td>36.68</td><td>31.12</td></tr><tr><td>Gemma4</td><td>38.80</td><td>72.49</td><td>21.28</td><td>24.83</td><td>63.08</td><td>44.10</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>135.82</td><td>38.58</td><td>36.42</td><td>247.56</td><td>114.60*</td></tr><tr><td>AF3</td><td>104.44</td><td>133.84</td><td>78.38</td><td>83.23</td><td>152.97</td><td>110.57</td></tr><tr><td>Q250</td><td>74.88</td><td>184.39</td><td>67.77</td><td>64.63</td><td>132.42</td><td>104.82</td></tr><tr><td>Q30</td><td>48.42</td><td>156.21</td><td>83.32</td><td>45.47</td><td>51.29</td><td>76.94</td></tr><tr><td>Walrus-CR Walrus-R</td><td>32.68 16.80</td><td>40.16 22.95</td><td>21.10 4.68</td><td>24.32 4.67</td><td>58.53 25.97</td><td>35.36 15.01</td></tr></table>

captioning. Stage 2 adds the first versions of these pipelines, and it uses only commercially licensed segment libraries for closed captioning. Stage 3 adds more synthetic multi-talker mixtures, more code-switching and entertainment data, and the remaining closed-captioning libraries. The dialogue pipeline also changes between stages, because the translation estimator differs. Sec. §3 gives the details of each stage.

## C Breakdown by task and language

We report here the full per-language and per-dataset breakdowns for the results discussed in Sec. §4. Numbers for prior systems are taken from their official technical reports where available, and otherwise computed with our internal evaluation system. We include Whisper Large-v3 (WLv3, highlighted in gray) and Qwen3-Omni-30B (Q3O, highlighted in red) as additional reference points. Neither is included in the bold/underline calculations, since WLv3 is not a SpeechLLM and Q3O is four times larger than Walrus and any other SpeechLLM included in the comparison.

## C.1 Automatic Speech Recognition

Tables 7-12 report ASR results across all public and internal benchmarks.

Table 14: MT-ASR Results: LibriMix Clean 2Mix (left) and Japanese Daily Talk (right). Metrics: cpWER/cpCER and tcpWER/tcpCER (↓). tcpWER/tcpCER were calculated with equidistant\_interval option and five-second collar tolerance. Japanese Daily Talk is evalu ated on CEJC (Corpus of Everyday Japanese Conversation) on 30-second segments.
<table><tr><td rowspan="2">Model</td><td colspan="2">LibriMix (Clean 2Mix)</td><td colspan="2">Japanese Daily Talk</td></tr><tr><td>cpWER</td><td>tcpWER</td><td>cpCER</td><td>tcpCER</td></tr><tr><td>VibeVoice-ASR [46]</td><td>59.34</td><td>59.36</td><td>71.99</td><td>72.17</td></tr><tr><td>Gemma4</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>AF3</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q250</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q30</td><td>37.04</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Walrus-CR</td><td>26.84</td><td>27.12</td><td>66.60</td><td>77.22</td></tr><tr><td>Walrus-R</td><td>19.32</td><td>19.43</td><td>56.29</td><td>57.03</td></tr></table>

Table 15: AST Results: CoVoST-2, X→En (BLEU ↑). \*This average score reflects only the supported languages.
<table><tr><td>Model</td><td>DE</td><td>ES</td><td>FR</td><td>IT</td><td>JA</td><td>PT</td><td>RU</td><td>ZH</td><td>AVG</td></tr><tr><td>WLv3</td><td>36.14</td><td>37.06</td><td>33.87</td><td>35.96</td><td>23.53</td><td>53.05</td><td>41.61</td><td>11.50</td><td>34.09</td></tr><tr><td>Gemma4 Phi4-MM</td><td>42.01</td><td>37.15</td><td>41.04</td><td>43.02</td><td>25.50</td><td>51.12</td><td>46.44</td><td>17.90</td><td>38.02</td></tr><tr><td>AF3</td><td>39.81</td><td>43.60</td><td>42.24 23.34</td><td>41.42</td><td>30.54</td><td>55.28</td><td>N/A</td><td>22.39</td><td>39.33*</td></tr><tr><td>Q250</td><td>32.80 42.42</td><td>24.19 40.51</td><td>37.46</td><td>26.68</td><td>13.08 14.57</td><td>36.46</td><td>33.49</td><td>13.24</td><td>25.41</td></tr><tr><td>Q3O</td><td>46.06</td><td>43.01</td><td>41.95</td><td>34.95 42.61</td><td>30.54</td><td>56.65 59.65</td><td>50.44 54.02</td><td>22.58 28.09</td><td>37.44 43.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Walrus-CR Walrus-R</td><td>43.22 44.49</td><td>41.12 41.54</td><td>40.73 42.09</td><td>41.89 43.19</td><td>28.20 31.66</td><td>54.32 59.54</td><td>52.20 54.15</td><td>20.56 24.40</td><td>40.28 42.63</td></tr></table>

LibriSpeech (Table 7). Walrus-R reaches 1.65/3.85 WER on clean/other, close to the best comparable system AF3 (1.57/3.13). The gap is small and consistent with the fact that LibriSpeech is a clean, read-speech benchmark that does not exercise the harder skills Walrus is built for.

MLS (Table 8). Walrus-R achieves the best average WER of 6.89 across all eight languages, ahead of Phi4-MM (8.20, computed over supported languages only) and Gemma4 (9.67). Gains are largest on French, German, Italian, Polish, and Spanish, reflecting the broad multilingual coverage of our Stage-1 and Stage-2 data.

FLEURS (Table 9). Walrus-R reaches an average WER/CER of 5.33, second only to Phi4-MM (4.00, computed over supported languages only). Notable gains appear on Arabic, Hindi, and Korean, where Stage-3 low-resource data has the most impact.

VoxPopuli (Table 10). Walrus leads clearly on this benchmark, with Walrus-R reaching 7.84 WER and Walrus-CR already at 8.38, both well ahead of the best baseline Gemma4 (11.78). The improvement is consistent across all five languages.

CoVoST-2 ASR (Table 11). Walrus-R reaches an average of 6.38 WER/CER, the best among comparable systems. Gains are particularly strong on Japanese and Chinese, where our multilingua curriculum provides substantial coverage.

Internal entertainment ASR (Table 12). This is the benchmark where Stage-3 specialization has the clearest impact. Walrus-R achieves the best result in every language, with the largest gains in Japanese (11.97 vs. 23.69 for Walrus-CR), French (24.55 vs. 59.16), and German (27.68 vs. 39.71).

## C.2 Code-Switching ASR

Table 13 reports CS-ASR results on the CS-FLEURS Read and XTTS test sets, which are OOD for Walrus. Walrus-R is the strongest system by a large margin on both sets, reaching 7.23 CER on Read and 15.01 on XTTS, against 32.30/44.10 for the best baseline (Gemma4) and 26.69/35.36 for Walrus-CR. The gain from Walrus-CR to Walrus-R is the largest single improvement across all benchmarks, and it comes entirely from the synthetic CS-ASR pipeline introduced in Stage 3. Performance is consistent across all five language pairs, with the largest absolute gains on English-Japanese and English-Hindi.

Table 16: AST Results: FLEURS, X→En (top) and En→X (bottom), BLEU (↑). \*This average score reflects only the supported languages.
<table><tr><td>Model</td><td>AR</td><td>DE</td><td>ES</td><td>FR</td><td>HI</td><td>IT</td><td>JA</td><td>KO</td><td>PT</td><td>ZH</td><td>AVG</td></tr><tr><td colspan="10">X→En</td></tr><tr><td>WLv3</td><td>20.63</td><td>33.27</td><td>22.27</td><td>31.67</td><td>19.40</td><td>23.57</td><td>16.51</td><td>19.05</td><td>36.68</td><td>15.80</td><td>23.89</td></tr><tr><td>Gemma4</td><td>29.56</td><td>33.55</td><td>23.11</td><td>31.22</td><td>27.46</td><td>23.74</td><td>22.47</td><td>23.44</td><td>36.79</td><td>21.63</td><td>27.29</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>37.71</td><td>25.33</td><td>35.10</td><td>N/A</td><td>26.06</td><td>21.62</td><td>N/A</td><td>40.80</td><td>22.37</td><td>29.86*</td></tr><tr><td>AF3</td><td>14.05</td><td>33.27</td><td>23.43</td><td>28.83</td><td>8.62</td><td>23.08</td><td>11.90</td><td>14.32</td><td>35.39</td><td>16.43</td><td>20.93</td></tr><tr><td>Q250</td><td>18.79</td><td>35.30</td><td>24.81</td><td>31.68</td><td>16.09</td><td>23.94</td><td>18.36</td><td>19.01</td><td>41.48</td><td>19.43</td><td>24.89</td></tr><tr><td>Q30</td><td>33.31</td><td>44.42</td><td>30.40</td><td>42.10</td><td>32.24</td><td>31.30</td><td>26.41</td><td>29.80</td><td>47.32</td><td>29.20</td><td>34.61</td></tr><tr><td>Walrus-CR</td><td>35.89</td><td>37.71</td><td>26.32</td><td>36.31</td><td>31.77</td><td>27.40</td><td>25.36</td><td>20.07</td><td>40.95</td><td>20.33</td><td>30.21</td></tr><tr><td>Walrus-R</td><td>38.16</td><td>38.91</td><td>27.53</td><td>38.13</td><td>37.36</td><td>29.28</td><td>27.01</td><td>24.23</td><td>42.71</td><td>27.22</td><td>33.05</td></tr><tr><td colspan="10">En→X</td></tr><tr><td>WLv3</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Gemma4</td><td>16.55</td><td>24.49</td><td>22.66</td><td>37.20</td><td>22.26</td><td>23.90</td><td>39.08</td><td>36.91</td><td>37.30</td><td>34.25</td><td>29.46</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>34.44</td><td>23.66</td><td>37.62</td><td>N/A</td><td>23.44</td><td>30.67</td><td>N/A</td><td>37.79</td><td>37.10</td><td>32.15*</td></tr><tr><td>AF3</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q250</td><td>12.65</td><td>25.74</td><td>20.73</td><td>30.99</td><td>18.84</td><td>14.54</td><td>10.05</td><td>11.78</td><td>29.77</td><td>15.15</td><td>19.02</td></tr><tr><td>Q30</td><td>28.18</td><td>35.77</td><td>26.00</td><td>48.97</td><td>46.86</td><td>29.03</td><td>14.50</td><td>13.09</td><td>45.09</td><td>17.79</td><td>29.75</td></tr><tr><td>Walrus-CR</td><td>26.33</td><td>34.06</td><td>25.04</td><td>44.22</td><td>25.32</td><td>28.09</td><td>40.02</td><td>32.18</td><td>38.58</td><td>30.41</td><td>32.42</td></tr><tr><td>Walrus-R</td><td>27.69</td><td>34.85</td><td>25.87</td><td>49.77</td><td>29.44</td><td>29.82</td><td>40.04</td><td>34.23</td><td>39.57</td><td>32.01</td><td>34.33</td></tr></table>

## C.3 Multi-Talker ASR

Table 14 reports MT-ASR results on LibriMix Clean 2-Mix and the Japanese Daily Talk corpus (CEJC). All comparable baselines produce no valid output on these benchmarks, as they are not designed for overlapping-speech scenarios. Walrus-R reaches 19.32 cpWER on LibriMix and 56.29 cpCER on Japanese Daily Talk, with Walrus-CR already competitive at 26.84 and 66.60, respectively.

## C.4 Speech Translation

AST CoVoST-2 X→En (Table 15). Walrus-R achieves the best average BLEU of 42.63, ahead of Phi4-MM (39.33) and Gemma4 (38.02). Gains are consistent across languages, with the strongest improvements on Japanese, Russian, and Chinese.

AST FLEURS X→En and En→X (Table 16). On the OOD FLEURS X→En, Walrus-R reaches 33.05 BLEU, again ahead of Phi4-MM (29.86) and Gemma4 (27.29). On En→X, Walrus-R achieves 34.33 BLEU, the best among all comparable systems, with strong performance on Fr, De, and Ja.

## C.5 Speech-to-Speech Translation

S2ST CVSS-C X→En (Table 17). Walrus-R reaches 43.63 ASR-BLEU, the best result among all systems, including the 4× larger Qwen3-Omni-30B (41.27). Walrus-CR is also competitive at 43.27, showing that S2ST capability is already strong after Stage 2.

S2ST FLEURS X→En and En→X (Table 18). On X→En, Walrus-R achieves 32.28 ASR-BLEU, ahead of Q25O (25.15) and close to the bigger Q3O (33.60). On En→X, Walrus is the only system among comparable models that supports this direction, reaching 22.94 ASR-BLEU (research variant).

S2ST quality beyond ASR-BLEU (Table 19). ASR-BLEU measures translation accuracy but does not capture the full quality of spoken output. A complete evaluation of S2ST must also consider speaker consistency, emotion preservation, prosody preservation, and naturalness, as detailed in [30]. We thus compute several metrics, using the COMPASS toolkit [30]: UTMOS-v2 (UTMOS)<sup>18</sup> predicts the naturalness of the output speech; speaker similarity (SSim) is the similarity between the speaker of the source speech and the speaker of the output, measured via WavLM-SV;<sup>19</sup> emotion similarity (ESim) is the similarity between the emotion of the source and of the output measured using Emotion2Vec;<sup>20</sup> AutoPCP (APCP) predicts how similar the prosody of the output is to the prosody of the source [3]. We compare Walrus with four systems: Qwen2.5/3-Omni, Kraken [16], the system whose speech decoder Walrus adopts, and Seamless [3], a dedicated multilingual S2ST system. Q2.5O and Kraken support only X→En translation.

Table 17: S2ST Results: CVSS-C, X→En (ASR-BLEU ↑).
<table><tr><td>Model</td><td>DE</td><td>ES</td><td>FR</td><td>IT</td><td>JA</td><td>PT</td><td>RU</td><td>ZH</td><td>AVG</td></tr><tr><td>Gemma4</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>AF3</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q250</td><td>40.39</td><td>38.32</td><td>35.54</td><td>33.67</td><td>15.61</td><td>55.10</td><td>48.15</td><td>21.39</td><td>36.02</td></tr><tr><td>Q30</td><td>44.10</td><td>41.32</td><td>38.30</td><td>40.21</td><td>30.04</td><td>58.12</td><td>52.19</td><td>25.90</td><td>41.27</td></tr><tr><td>Walrus-CR</td><td>49.21</td><td>47.12</td><td>40.73</td><td>41.89</td><td>32.14</td><td>57.33</td><td>52.58</td><td>25.16</td><td>43.27</td></tr><tr><td>Walrus-R</td><td>49.01</td><td>48.97</td><td>37.96</td><td>40.27</td><td>33.85</td><td>59.99</td><td>52.51</td><td>26.44</td><td>43.63</td></tr></table>

Table 18: S2ST Results: FLEURS, X→En (top) and En→X (bottom), ASR-BLEU (↑).
<table><tr><td>Model</td><td>AR</td><td>DE</td><td>ES</td><td>FR</td><td>HI</td><td>IT</td><td>JA</td><td>KO</td><td>PT</td><td>ZH</td><td>AVG</td></tr><tr><td colspan="10">X→En</td></tr><tr><td>Gemma4</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>AF3</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q250</td><td>20.63</td><td>36.40</td><td>23.71</td><td>31.54</td><td>15.78</td><td>25.04</td><td>17.77</td><td>20.39</td><td>38.91</td><td>21.33</td><td>25.15</td></tr><tr><td>Q30</td><td>32.80</td><td>43.10</td><td>29.30</td><td>41.00</td><td>31.52</td><td>30.34</td><td>25.61</td><td>29.30</td><td>45.41</td><td>28.30</td><td>33.60</td></tr><tr><td>Walrus-CR</td><td>34.23</td><td>35.00</td><td>25.86</td><td>33.79</td><td>27.94</td><td>25.47</td><td>22.92</td><td>22.79</td><td>36.17</td><td>21.71</td><td>28.59</td></tr><tr><td>Walrus-R</td><td>38.23</td><td>38.87</td><td>27.90</td><td>38.12</td><td>32.75</td><td>28.62</td><td>26.51</td><td>25.38</td><td>40.43</td><td>25.98</td><td>32.28</td></tr><tr><td colspan="10">En→X</td></tr><tr><td>Gemma4</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td></td><td>N/A</td></tr><tr><td>Phi4-MM</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A N/A</td><td>N/A</td></tr><tr><td>AF3</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q250</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q30</td><td>17.83</td><td>32.10</td><td>25.11</td><td>43.89</td><td>23.59</td><td>25.40</td><td>13.04</td><td>11.13</td><td>40.29</td><td>24.29</td><td>25.67</td></tr><tr><td>Walrus-CR</td><td>12.41</td><td>24.14</td><td>18.20</td><td></td><td>11.03</td><td>12.98</td><td>10.41</td><td></td><td>32.75</td><td>12.04</td><td>17.19</td></tr><tr><td>Walrus-R</td><td>15.01</td><td>39.03</td><td>22.55</td><td>30.44 36.07</td><td>18.40</td><td>20.80</td><td>11.75</td><td>7.55 11.14</td><td>38.50</td><td>16.10</td><td>22.94</td></tr></table>

On X→En, Walrus-R reaches 32.28 ASR-BLEU. This is on par with Kraken (32.30) and Seamless (32.10), 7.18 points above Q2.5O (25.10), but 1.32 points lower than Q3O (33.60). On En→X, Walrus-R achieves 22.94 ASR-BLEU, slightly below both Seamless and Q3O.

The strongest result of Walrus is the preservation of the speaker, emotion, and prosody. On X→En, Walrus-R has the best SSim (0.77), ESim (0.81), and AutoPCP (2.57) of all systems, including Q3O (0.61, 0.75, and 2.19). Compared with Seamless, it improves SSim by 0.11, ESim by 0.05, and AutoPCP by 0.52. Compared with Kraken, which shares our speech decoder, the gains are smaller (0.02, 0.04, and 0.09). On En→X, the gaps are larger: Walrus-R reaches 0.70 SSim, 0.73 ESim, and 2.45 AutoPCP, compared to 0.58/0.56/2.25 for Seamless and 0.49/0.44/2.12 for Q3O. Walrus conditions its vocoder on the source speech (Sec. §2), which is consistent with this result.

The naturalness scores are more mixed. Walrus-R has a higher UTMOS than Seamless in both directions (3.19 vs 3.06 and 3.05 vs 2.82). The Qwen Omni models score higher on X→En (3.48 for Q2.5O and 3.59 for Q3O), and Q3O also scores higher on En→X (3.27). Both these models generate speech with fixed reference voices: this can make the speech cleaner and more natural, but it does not preserve the prosody and emotion of the source, which fits their low ESim and AutoPCP scores. Walrus, in contrast, conditions its vocoder on the source speech; FLEURS contains recordings of uneven quality, so this conditioning can carry the noise of the source into the output and eventually lower the UTMOS.

Table 19: Speech-to-speech translation on FLEURS, X→En (left) and En→X (right). Metrics are ASR-BLEU, UTMOS (predicted naturalness), speaker similarity (SSim), emotion similarity (ESim), and AutoPCP (APCP, predicted prosody similarity). Higher is better for all metrics, and scores are averages over languages. N/A means that the system does not support the direction. Qwen3-Omni (Q3O, shaded) is much larger and is not ranked. Among the other systems, the best score is in bold and the second best is underlined.
<table><tr><td rowspan="2">Model</td><td colspan="5">X→En</td><td colspan="5">En→X</td></tr><tr><td>ASR-BLEU</td><td>UTMOS</td><td>SSim</td><td>ESim</td><td>APCP</td><td>ASR-BLEU</td><td>UTMOS</td><td>SSim</td><td>ESim</td><td>APCP</td></tr><tr><td>Kraken [16]</td><td>32.30</td><td>3.13</td><td>0.75</td><td>0.77</td><td>2.48</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Seamless [3]</td><td>32.10</td><td>3.06</td><td>0.66</td><td>0.76</td><td>2.05</td><td>27.33</td><td>2.82</td><td>0.58</td><td>0.56</td><td>2.25</td></tr><tr><td>Q2.50</td><td>25.10</td><td>3.48</td><td>0.49</td><td>0.53</td><td>2.40</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Q30</td><td>33.60</td><td>3.59</td><td>0.61</td><td>0.75</td><td>2.19</td><td>25.67</td><td>3.27</td><td>0.49</td><td>0.44</td><td>2.12</td></tr><tr><td>Walrus-CR</td><td>28.59</td><td>3.17</td><td>0.75</td><td>0.76</td><td>2.54</td><td>17.19</td><td>3.01</td><td>0.67</td><td>0.69</td><td>2.41</td></tr><tr><td>Walrus-R</td><td>32.28</td><td>3.19</td><td>0.77</td><td>0.81</td><td>2.57</td><td>22.94</td><td>3.05</td><td>0.70</td><td>0.73</td><td>2.45</td></tr></table>

Walrus-R improves on Walrus-CR on all columns of Table 19. The largest gains are in ASR-BLEU, with +3.69 on X→En (28.59 to 32.28) and +5.75 on En→X (17.19 to 22.94), while the expressivity metrics improve by smaller amounts: SSim improves by 0.02 and 0.03, ESim by 0.05 and 0.04, and AutoPCP by 0.03 and 0.04. Stage 3 adds S2ST data and raises the S2ST task weight from 0.5 to 2.0 (Sec. §3.6): translation quality benefits most, while the voice preservation of Walrus-CR is alread close to that of Walrus-R.

## C.6 Audio Understanding Benchmarks

Walrus-R achieves 71.92% accuracy on MMAU-Pro, far ahead of the best comparable baseline Q25O (52.20%) and above commercial systems such as Gemini-Flash-3.7 (70.40%) and GPT4o-Audio (52.50%).<sup>21</sup> On MMSU, Walrus-R reaches 64.14%, ahead of AF3 (61.40%) and Q25O (60.57%). The gap between Walrus-CR (50.18% / 55.18%) and Walrus-R on both benchmarks is large and explained by the AudioReason-10M data introduced in Stage 3.

Table 20: Primary (most frequently sampled) prompt for each task. During training, each prompt is drawn uniformly from a pool of 10-20 paraphrases with the listed prompt oversampled; slot variables ({lang1}, {lang2}, {context}, {keywords}, {labels}, {speaker}, {start}, {end}, {agent}, {user}) are filled at data-construction time.
<table><tr><td>Task</td><td>Primary prompt</td></tr><tr><td colspan="2">Recognition</td></tr><tr><td>ASR</td><td>Transcribe the {lang1 } speech:</td></tr><tr><td>Context-ASR</td><td>Transcribe the {lang1 } speech from the context {context}:</td></tr><tr><td>Keyword-biased ASR</td><td>Transcribe the {lang1 } speech, biasing toward these keywords: {keywords}:</td></tr><tr><td>Code-switched ASR</td><td>Transcribe the code-switched speech in {lang1 } and {lang2}:</td></tr><tr><td>Dialogue-contextual ASR</td><td>Given the previous conversation between {agent} and {user}: {con- text} Transcribe the {lang1} speech:</td></tr><tr><td>Timestamped ASR</td><td>Transcribe the {language} utterance with timestamps:</td></tr><tr><td>Multi-talker ASR</td><td>Provide a speaker-attributed transcription of this multi-talker {lan-</td></tr><tr><td>Timestamped multi-talker ASR</td><td>guage} audio: Provide a timestamped, speaker-attributed transcription of this {lan-</td></tr><tr><td>Multi-talker ASR (CoT)</td><td>guage} audio: First, count the speakers in this mixture. Then, transcribe each</td></tr><tr><td>Timestamped multi-talker ASR (CoT)</td><td>speaker: First, estimate how many speakers are in this {language} mixture.</td></tr><tr><td>Timestamp QA</td><td>Then, provide a timestamped transcription of each speaker: What is the speaker saying between seconds {start} and {end}?</td></tr><tr><td>Target-speaker ASR</td><td>Please provide the transcription for {speaker} only:</td></tr><tr><td colspan="2">Translation</td></tr><tr><td>MT</td><td>Translate {lang1} text into {lang2} text:</td></tr><tr><td>AST (direct)</td><td>Translate the {lang1 } speech into {lang2} text:</td></tr><tr><td>AST (cascaded)</td><td>Transcribe the {lang1 } speech and then translate into {lang2} text:</td></tr><tr><td>Context-AST (direct)</td><td>Translate the {lang1 } speech into {lang2} text, with context &#x27;{con- text}&#x27;:</td></tr><tr><td>Context-AST (cascaded)</td><td>Transcribe the {lang1 } speech and then translate it into {lang2} text,</td></tr><tr><td>S2ST (direct)</td><td>with context {context}: Translate the {lang1 } speech into {lang2} speech:</td></tr><tr><td>S2ST (cascaded)</td><td>Transcribe {lang1 } speech and then translate into {lang2} text and then into speech:</td></tr><tr><td>Context-S2ST (direct)</td><td>Translate the {lang1 } speech into {lang2} speech, with context</td></tr><tr><td>Context-S2ST (cascaded)</td><td>{context}: Transcribe the {lang1 } speech, translate it into {lang2} text, and then into speech, with context &#x27;{context}&#x27;:</td></tr><tr><td colspan="2">Generation</td></tr><tr><td></td><td>Generate {lang1 } speech of the text:</td></tr><tr><td colspan="2">Understanding</td></tr><tr><td>SER</td><td></td></tr><tr><td>Audio captioning</td><td>Recognize the emotion in the speech; choose one from [{labels }]: Generate a caption for this speech:</td></tr><tr><td>CC [audio-event ASR]</td><td>Transcribe the {lang1 } speech and include any background audio events or sounds:</td></tr><tr><td>CC [event sequence]</td><td>Identify the structural timeline of this audio (e.g., [Speech], [Sigh], [Cat meowing]):</td></tr><tr><td>CC [timestamped event ASR]</td><td>Provide a timestamped transcription of the {lang1 } speech along with time-aligned sound events:</td></tr><tr><td>CC [diarized-event ASR]</td><td>Provide a timestamped diarized transcription in {lang1 }, including background audio events:</td></tr><tr><td>CC [full scene description]</td><td>Summarize the entire audio scene (total events, speakers, SFX, music, etc.) and provide a detailed timeline breakdown:</td></tr></table>