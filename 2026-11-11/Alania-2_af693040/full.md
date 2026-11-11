# Alania-2

Turkish text-to-speech

with instant and fine-tuned voice cloning

Sezgin Saygili Emre Kaplaner Oncel Ozgul Fikri San Koktas

PatientDesk AI sezgin@patientdesk.ai speech.patientdesk.ai

## Abstract

We present Alania-2, a Turkish text-to-speech (TTS) system with two built-in voices, voice design from text descriptions, and two kinds of voice cloning. Alania-2 is a 488M-parameter model with the VoxCPM architecture, which generates continuous audio latents with a difusion head on top of a language-model backbone. We train it from scratch on 18,195 hours of Turkish speech, 90% of which is synthetic speech produced by a larger teacher model and filtered automatically. A text normalizer rewrites numbers, abbreviations and foreign names into the words a Turkish speaker would say; without it, none of 96 test sentences with digits kept all its critical parts, such as an amount or a date, through synthesis and speech recognition, against 86 with it. On Freya-TR-Eval, a public benchmark of 495 Turkish sentences scored with automatic speech recognition, Alania-2 reaches a word error rate of 1.44%, compared with 5.99% for our previous system. On its built-in voices, it comes within half a point of character error rate of a 2.3B-parameter model at under a quarter of the size, and a 7.6M-parameter single-voice model trained with the same recipe matches the most intelligible systems we measured, which suggests that training data matters more for intelligibility than model size. We also report practical findings on voice cloning: conditioning on the reference audio alone, rather than continuing it, cuts skipped words from 21.9% to 1.7%; 20–30 seconds of reference audio works best; and a low-rank adapter trained in about two minutes raises speaker similarity well above instant cloning. Finally, we describe inference-engine changes that bring first audio down to 32 ms for a single stream. We release our evaluation harness and three Turkish speech and text datasets.

Keywords: text-to-speech; Turkish; voice cloning; synthetic training data; text normalization; low-latency inference

## 1 Introduction

Turkish is spoken by more than eighty million people, but it remains a mid-resource language for speech synthesis. Recent zero-shot TTS models (Chen et al., 2025a,b; Du et al., 2024; Zhou et al., 2025) were trained mainly on English and Chinese speech. Multilingual successors such as VoxCPM2 (Zhou et al., 2026) support Turkish as one of 30 languages, and open Turkish systems difer by an order of magnitude in intelligibility (Section 6).

Turkish text is also hard to read aloud correctly. Numbers, dates and foreign names take case sufixes that must agree with how they are pronounced, and the dotted and dotless i distinguish words but are easily broken by case conversion designed for English.

We built Alania-2 for telephone voice assistants and long-form narration. These uses need reliable reading of numbers, doses, dates and names; low latency when streaming; consistent built-in voices; and voice cloning that customers can use on their own. Our previous system, Alania-1, is a 3.3B-parameter language model over 24 kHz SNAC codec tokens (Siuzdak et al., 2024). It misreads about one word in seventeen on the Freya-TR-Eval benchmark (Pamuk et al., 2026).

![](images/c195a2230ec5754677fae3b61011301776e5da7ee65c7ac8b302bff5e6ab424d.jpg)  
Figure 1 Overview of Alania-2. Text is first rewritten into the words a speaker would say (Section 4); the model then generates 48 kHz audio in the requested voice (Sections 5 and 8).

Figure 1 gives an overview of Alania-2. Our contributions are:

1 A data recipe for a mid-resource language. We train a 488M-parameter model from scratch on 90% synthetic speech and show that, on its built-in voices, it comes within half a point of character error rate of a fine-tuned model more than four times its size (Sections 3 and 5).

2 A text normalizer with 134 readings and a lexicon reviewed by a native-speaker author, and respelling of foreign brand names (Section 4).

3 A comparison with 14 other systems under one protocol (Section 6).

4 Evidence that model size matters little for intelligibility. A 2.3B-parameter model trained on our data is barely more intelligible than Alania-2, and a 7.6M-parameter single-voice model trained with the same recipe matches the best systems we measured (Section 7).

5 Practical findings on voice cloning: why continuing a reference skips words, how much reference audio helps, and fast fine-tuning with low-rank adapters (Section 8).

6 Inference-engine changes for low latency and for serving many fine-tuned voices from one model (Section 10).

We release the Freya-TR-Eval harness and results, a 2,398-hour Turkish caption set, a 3,411-hour synthetic speech set and a 1.44M-sentence domain text corpus. We do not release the model weights.

## 2 Related Work

Zero-shot TTS and voice cloning. Codec language models clone a voice by continuing a short audio prompt (Chen et al., 2025a). CosyVoice (Du et al., 2024) combines supervised semantic tokens with flow matching, and F5-TTS (Chen et al., 2025b) removes the need for explicit text–speech alignment. VoxCPM (Zhou et al., 2025) predicts continuous latents of an audio variational autoencoder (VAE) with a difusion decoder on a language-model backbone, without a discrete speech tokenizer. VoxCPM2 (Zhou et al., 2026) extends it to 30 languages, voice design from text and a VAE that encodes at 16 kHz and decodes at 48 kHz. We use the VoxCPM2 architecture, tokenizer and VAE but train our own weights.

Our previous system, Alania-1, followed the codec route with SNAC tokens (Siuzdak et al., 2024).

Turkish TTS. Yeshpanov et al. (2023) train a Tacotron 2 model on Kazakh alone and synthesize nine other Turkic languages, including Turkish, by mapping their letters through the International Phonetic Alphabet to Kazakh letters. FreyaTTS (Pamuk et al., 2026) is a 183M-parameter Turkish-first flow-matching model released together with the Freya-TR-Eval benchmark, and Ozkose et al. (2026) compare four recent systems on Turkish, fine-tuned and zero-shot, with eighteen objective metrics.

Data and evaluation tools. We build our real-speech set from YODAS (Li et al., 2023), a multilingual corpus of YouTube audio, much of it with manual or automatic subtitles. We filter it with a language classifier trained on VoxLingua107 (Valk and Alumäe, 2021), the AST audio tagger (Gong et al., 2021), DNSMOS quality prediction (Reddy et al., 2021) and Whisper speech recognition (Radford et al., 2023). Our synthetic text comes from FineWeb2 (Penedo et al., 2025), Wikipedia and sentences written by Qwen3 (Yang et al., 2025). For evaluation, we measure intelligibility with Whisper, naturalness with UTMOS (Saeki et al., 2022), and speaker similarity with ECAPA-TDNN embeddings (Desplanques et al., 2020; Ravanelli et al., 2021); FLEURS (Conneau et al., 2023) supplies some of the cloning references.

Eficient serving. PagedAttention (Kwon et al., 2023) manages the key/value cache of language models in fixed-size blocks, and S-LoRA (Sheng et al., 2024) and Punica (Chen et al., 2024) serve many low-rank adapters (LoRA; Hu et al., 2022) on one base model. Our engine builds on Nano-vLLM (a710128, 2025; GeeeekExplorer, 2025), which uses the same block-based cache.

## 3 Data

Figure 2 summarizes the training data. The base mix combines 1,772 hours of filtered real speech with 16,423 hours of synthetic speech; a second, shorter stage adds voice design and style instructions.

## 3.1 Real speech

We start from the Turkish part of YODAS (Li et al., 2023): 38,494 recordings totalling 9,216 hours. Most of this audio is weakly labelled. Of the recordings, 97% carry automatic YouTube captions, and only 2,448 hours are labelled as Turkish at all. We therefore treat every file as untrusted. We split it into segments, check the language with a classifier trained on VoxLingua107 (Valk and Alumäe, 2021) and with Whisper (Radford et al., 2023), remove music and singing with the AST audio tagger (Gong et al., 2021), reject clipped or band-limited audio, and keep segments with a DNSMOS score of at least 2.4 (Reddy et al., 2021). Finally, we transcribe every segment again with Duyu, our Turkish speech recognizer, a fine-tune of Whisper large-v3 turbo.

This leaves 2,398 hours (835,267 segments, median length 9.9 s). Qwen3-32B (Yang et al., 2025) writes a short style caption for each segment, describing the speaker and the delivery. For training, we apply stricter filters and remove segments with a low DNSMOS score (189,631), a Duyu transcript that disagrees with the YouTube caption (42,186), or an implausible speaking rate (152). The training set holds 598,213 segments (1,772 hours) from 19,969 speaker clusters; we hold out about 1% of the clusters for validation.

Table 1 Checks applied to every synthetic clip. A clip is kept only if it passes all of them.
<table><tr><td>Check</td><td>Measured with</td><td>Threshold</td></tr><tr><td>Intelligibility</td><td>CER, faster-whisper large-v3</td><td>≤ 0.05</td></tr><tr><td>Speaking rate</td><td>characters per second</td><td>7-25</td></tr><tr><td>Voice match</td><td>ECAPA similarity to the reference</td><td>≥ 0.45 (≥ 0.35 with a style instruction)</td></tr><tr><td>Naturalness</td><td>UTMOS22</td><td>≥2.5</td></tr><tr><td>Pitch</td><td>median F0</td><td>consistent with the voice&#x27;s gender</td></tr></table>

![](images/7dcc9f5e4d6df20de68a4cb779263c6317f2e782d6cea93089928c81e91c987f.jpg)  
Figure 2 Training data. The base mix is 90% synthetic speech; a short second stage adds voice design and style instructions.

## 3.2 Synthetic speech

Turkish speech with the quality, speaker balance and text coverage we needed is not available at this scale, so we generate it with VoxCPM2 (Zhou et al., 2026), using 10 difusion steps and a guidance scale of 2.0. One H100 GPU produces 83 seconds of audio per second.

Text. The text comes from FineWeb2 (Penedo et al., 2025) (40%), domain sentences written by Qwen3-30B-A3B (Yang et al., 2025) (25%), Wikipedia (22.5%) and YODAS transcripts (12.5%). After removing duplicates, 5.65M of 6.18M text units remain. Each unit is normalized (Section 4) with the normalizer version current at the time.

Voices. The speakers are our two built-in voices and a bank of 2,752 designed voices (1,283 female, 1,469 male). We create the bank from 5,000 text descriptions and drop nearduplicates whose speaker embeddings have a cosine similarity of 0.85 or more. Each clip is generated in one of three modes: continuing a reference recording (77%), following a reference together with an English style instruction (20%), or designing a voice from its description alone (3%). About 22.5% of the training clips are also passed through a simulated room, microphone, noise and codec channel.

Quality control. We keep a clip only if it passes every check in Table 1. In a manual audit of 681 generated clips, 79.2% passed.

Table 2 Examples of written input and the text Alania-2 receives.
<table><tr><td>Case</td><td>Written</td><td>Spoken form given to the model</td></tr><tr><td>Money with a suffix</td><td>750 TL&#x27;dir</td><td>yedi yüz elli liradır</td></tr><tr><td>Decimal amount</td><td>12,50 TL</td><td>on iki lira elli kuruş</td></tr><tr><td>Percentage with a suffix</td><td>%2,5&#x27;lik</td><td>yüzde iki buçukluk</td></tr><tr><td>Turkish initialism</td><td>SGK</td><td>segeka</td></tr><tr><td>Borrowed initialism</td><td>CEO</td><td>siio</td></tr><tr><td>Brand name</td><td>PatientDesk</td><td>Peyşınt Desk</td></tr><tr><td>Brand name</td><td>ChatGPT</td><td>Çet Ci Pi Ti</td></tr></table>

## 3.3 Second stage and released data

The second-stage mix has 1.4M rows: voice descriptions, short paraphrased descriptions, style-instruction rows, and 400,000 plain rows that keep the model’s base behaviour. We release three datasets on Hugging Face:

• Captions<sup>1</sup>: transcripts and style captions for the 835,267 real segments, without audio.

• Synthetic speech<sup>2</sup>: $^ { 3 , 4 1 1 }$ hours at 48 k $\mathbf { I z } ,$ designed voices only.

• Domain sentences<sup>3</sup>: 1.44M sentences.

## 4 Text Normalization

Alania-2 reads Turkish spelling and has no phoneme input, so numbers, symbols, abbreviations and foreign words must first be rewritten into the words a speaker would say. Table 2 shows typical cases. Our normalizer is a sequence of rules for numbers, money, dates, times, ranges, phone numbers, doses, units, titles, addresses and symbols. After expanding a number, an amount or a unit, it re-attaches the Turkish sufix that followed it and adjusts the sufix to the new word, mainly its vowels (vowel harmony). A companion paper describes the normalizer in detail and evaluates its version of 7 October 2026, which has 131 test cases; here we summarize the parts that matter most for the model.

Initialisms and native review. Turkish initialisms are usually read as Turkish letter names joined into one word. Some are read as words, and a few borrowed ones use English letter names. A native-speaker author reviewed 134 written forms in 13 categories, initialisms among them, on a page that showed each planned reading as text. The author accepted 118 as shown and changed 16: 10 marked wrong and 6 marked right with a better reading typed in.

Upper case. Words written entirely in capitals are lowercased with Turkish rules, and every remaining I then becomes ı and every İ becomes i. Other capitals stay. A model that keeps capital letters may otherwise read a capital I the English way: in an earlier test of ours, VoxCPM2 read the city name Iğdır as iğdir.

Foreign words and brands. Read with Turkish spelling rules, some brand names and English words come out garbled. A word seen at least 1,000 times in the text behind our 225,764-word Turkish word list stays as written. Otherwise we respell it in Turkish letters if it has an inner capital, or if it is an English word, meaning ASCII letters only, present in the CMU Pronouncing Dictionary<sup>4</sup> and absent from the word list. The respelling comes from the word’s pronunciation in that dictionary, and these automatic outputs have not had a native review.

Table 3 Recognition of 100 brand and English terms by Duyu, in one carrier sentence with one voice and seed. Similarity is between the part of the transcript that fills the term’s slot and either the term or its respelling, whicheve is higher (0–1) ↑.
<table><tr><td></td><td>As written Respelled</td></tr><tr><td>Mean similarity, all 100 terms</td><td>0.911 0.961</td></tr><tr><td>Transcript equals the term or its respelling</td><td>72 85</td></tr><tr><td>Mean similarity, the 17 respelled terms</td><td>0.627 0.989</td></tr></table>

Lexicon. Automatic respelling is not always better than the written form, so we checked 100 candidate brand and English terms. We generated each term both as written and respelled, in one carrier sentence, and transcribed both with Duyu. A native-speaker author listened to both versions of every term and decided 94 of them, giving a lexicon of 17 respelled and 77 unchanged terms; the other 6 were left to the automatic rule. Users can add their own entries. Table 3 gives the scores: for the 17 respelled terms, recognition similarity rises from 0.63 to 0.99. We used this study to build the lexicon, not to test it. Each form of a term has one render, and a rule using these same scores proposed 16 of the 17 respellings, so the rise is partly built in.

Efect of normalizing at all. Normalization is not optional for Alania-2. The companion paper tests this on 150 constructed sentences in the style of clinic and customer-service messages, covering numbers, dates, units, phone numbers, initialisms, brands and addresses. Most of their expected spoken forms have not had a native review. We ran the model directly, with one voice and one render per sentence, and transcribed the audio with Duyu. Given raw written text, the model had a word error rate of 66.7%, against 6.4% with normalized text. A critical span is a part of a sentence that must survive, such as an amount or a name. Every critical span was recovered from the transcript in 7% of sentences with raw text and in 81% with normalized text. Among the 96 sentences containing digits, this held for none with raw text and for 86 with normalized text. In the transcripts, numbers come out as other words. Nobody listened to these renders, so this description rests on transcripts and audio length.

This describes Alania-2, which was trained on normalized text, not neural TTS in general. Zero-shot VoxCPM2, which generated most of Alania-2’s training audio, did not break down on raw text in a separate evaluation. Over 24 prompts and two seeds, its character error rate (CER) was 10.9% on raw text against 5.6% on normalized text on prompts built to stress normalization, and 7.7% against 5.5% on numeric prompts. That evaluation used other prompts, another recognizer and another scorer, so these figures cannot be set against the ones above number by number, but they suggest that the failure on digits is not inherent to the architecture.

Efect of respelling. Respelling adds a smaller gain whose size is uncertain, on terms already used to build the lexicon. Among the sentences whose text it changes, every critical span was recovered in 12 of 15 with respelling and 3 of 15 without it through our speech API, and in 11 and 6 of 16 when we ran the model directly.

![](images/04ad2bed5853ad04224d919a26013399ecde6b1bcabf9eee6d96776c277bc791.jpg)  
CER (mean, log scale), lower is better  
Figure 3 Choosing an architecture: intelligibility against naturalness on 120 Turkish prompts, 240 renders per system. The best systems sit at the top left.

Table 4 Alania-2 architecture, excluding the frozen audio VAE. Attention heads stay 128-dimensional to match VoxCPM’s position encoding.
<table><tr><td>Component</td><td>Layers × width</td><td>Heads</td><td>Params</td></tr><tr><td>Backbone LM</td><td>16 × 1024</td><td>8 (2 KV)</td><td>268M</td></tr><tr><td>Residual LM</td><td>6×1024</td><td>8 (2 KV)</td><td>72M</td></tr><tr><td>Feature encoder</td><td>8×768</td><td>6</td><td>69M</td></tr><tr><td>Diffusion decoder</td><td>8×768</td><td>6</td><td>72M</td></tr><tr><td>Total</td><td></td><td></td><td>488M</td></tr></table>

## 5 Model

## 5.1 Choosing an architecture

We compared candidate architectures on 120 Turkish prompts with two seeds each (Figure 3). Zero-shot VoxCPM2 was the most intelligible (CER 0.016), the most natural by UTMOS (3.95) and the most consistent between renders of the same voice (speaker similarity 0.83). Unlike Alania-1, which ran on past the end of the text in 7.5% of renders, it never did. Its VAE also preserved speaker identity through encoding and decoding far better than the SNAC codec (similarity 0.93 against 0.71). We therefore adopted the VoxCPM architecture.

## 5.2 Architecture and training

Alania-2 keeps VoxCPM2’s text tokenizer and its frozen audio VAE (Zhou et al., 2026), which encodes 16 kHz audio into 64-dimensional latents at 25 frames per second and decodes them to 48 kHz. A language-model backbone reads the text and the latents generated so far. A difusion transformer then predicts the next patch of four latent frames (0.16 s of audio) in 10 flow-matching steps with classifier-free guidance. Our model is smaller than VoxCPM2 (Table 4), and we train all its other weights from random initialization with the VoxCPM training code.

We pursued two tracks in parallel. Track 1 continued training the full VoxCPM2 on our data (2.3B parameters in the released checkpoint, excluding the VAE). Track 2 trained our own smaller model from scratch. A model trained from scratch needs many steps before it learns to align text with audio: a 1.24B-parameter variant never did within 54k steps (CER 0.84–0.88), while the 488M model did. The released model was trained in three stages:

![](images/d36b19c4fc20b99c2d29973fabf2ac14c4b3826eb3747f0b2306853687a25d35.jpg)  
Figure 4 CER of the candidate models on our 292-clip suite (one seed; Whisper large-v3), lower is better. On its built-in voices, Alania-2 is close to the 2.3B models at under a quarter of their size, and the second training stage fixed most of the base model’s errors with style instructions. Full results are in Appendix A.

1 about 64k steps on an earlier 8,904-hour mix;

2 70k steps on the 18,195-hour base mix, with a batch of 32 per GPU, two gradientaccumulation steps, a learning rate of 10<sup>−4</sup> and weight decay of 0.05;

3 the second stage on design and style data at a learning rate of $5 \times 1 0 ^ { - 5 }$ , stopped at step 4,000.

Training took on the order of 210 H200 GPU-hours, not counting data generation.

## 5.3 Comparison and choice

Figure 4 compares the candidates. On the two built-in voices, Alania-2’s CER is within half a point of VoxCPM2 and of both Track 1 models. With style instructions its CER is higher (3.4% against 2.0–2.2% for Track 1). It produced the requested gender for 30 of 32 voice designs, as often as any candidate. Its UTMOS is 0.2–0.3 lower than VoxCPM2’s, and it is weaker at cloning a voice from an English recording (Section 8). We chose Alania-2 because it is far cheaper to serve at nearly the same intelligibility: on one H200, it keeps up with real time at 12 requests per second, against about 8 for Track 1.

## 5.4 Voices, styles and design

The two built-in voices are designed voices, not recordings of real people. We chose them in a blind comparison of 12 candidates. For each voice we store a reference clip, an average speaker embedding and a pitch range (199.6–238.6 Hz for the female voice, 109.1–131.6 Hz for the male voice). At serving time, we use the pitch range to reject renders that drift away from the voice.

Style control comes from the captioned training rows. At first, the model sometimes read the instruction aloud: Turkish instructions on every take, and long English ones on 4 of 6. We now rewrite each instruction into a short English tag with a small LLM, after which all 54 test takes were clean.

Table 5 Alania-1 and Alania-2 on Freya-TR-Eval, both through our API. RTF is the real-time factor on one L4 GPU (Alania-1 measured on 100 sentences).
<table><tr><td>Alania-1 Alania-2</td></tr><tr><td>Architecture LLM over SNAC tokens VoxCPM</td></tr><tr><td>Parameters 3.3B 488M</td></tr><tr><td>Sample rate 24 kHz 48 kHz</td></tr><tr><td>WER (%) ↓ 5.99 1.44</td></tr><tr><td>CER (%)↓ 1.81 0.28</td></tr><tr><td>UTMOS22↑ 3.19 3.67</td></tr><tr><td>RTF↓ 2.49 0.151</td></tr></table>

Ready-made styles (warm, calm, energetic, smiling) keep CER between 0.33% and 1.82% and widen the pitch range by 0.5–1.4 semitones. Whispering and slow speech are beyond the model. For voice design we use a guidance scale of 3.0, chosen from a sweep between 1.5 and 4.0 in which higher guidance raised UTMOS (from 1.79 to 2.93) but lowered gender accuracy (from 30 to 27 of 32).

## 6 Evaluation

## 6.1 Freya-TR-Eval

Freya-TR-Eval (Pamuk et al., 2026) is a public Turkish benchmark of495 sentences of3–13 words, given as raw written text. It measures intelligibility: each system reads every sentence, a speech recognizer transcribes the audio, and the benchmark reports the word error rate (WER) against the original text. Its released file contains no digits, symbols, initialisms or mixed-case brand names, so it does not exercise text normalization. We re-ran it for 15 systems under one protocol, plus the bare Alania-2 model without our serving checks, and released the harness and results. Appendix B lists the exact checkpoint of every system.<sup>5</sup> Each system reads every sentence three times. We band-limit the audio to 8 kHz, resample it to 16 kHz, transcribe it with Whisper large-v3 (Radford et al., 2023) through faster-whisper<sup>6</sup> (beam size 5) and score it with the benchmark’s own text normalization. We also report UTMOS22 on the full-band audio and the real-time factor on one L4 GPU.

Figure 5 shows the results. Through our API, Alania-2 reaches 1.44% WER, seventh of the 15 systems. The two best are compact models of under 9M parameters: EMA Lightning at 1.04% and our Antalia-2 Mini at 1.05%. The benchmark authors report 1.43% for ElevenLabs v4, which we did not run. Diferences of a tenth of a point are within the spread between the three readings. Our serving checks and retries (Section 9) lower the bare model’s 1.51% to 1.44%. Compared with Alania-1, Alania-2 cuts WER four-fold and character error rate (CER) six-fold (Table 5).

System version. This report describes release alania-v2 of October 2026, with two builtin voices and 10 difusion steps. The service has changed since then. Sampling moved to 6 difusion steps, and on 11 October release alania-v2.1, a fine-tune of alania-v2 for control of speaking style with the same architecture, replaced the model. On 12 October nine voices were added. The Freya-TR-Eval results here use the female voice (alania2-f1)

![](images/c295a1c224faa4cb0d0f2176f285ab0ec7ca4ffc16cd523aa98892040b6d5492.jpg)  
Figure 5 Freya-TR-Eval word error rate (495 sentences, three readings each, faster-whisper large-v3). Our models are in blue. Alania-2 through our API is seventh of the 15 systems; the bare model without serving checks is shown for reference. Antalia-2 Mini is our 7.6M-parameter single-voice model (Section 7).

for Alania-2 and version 1.0.0 of Antalia-2 Mini; our released results also hold later runs of both.

## 6.2 What we did not measure

This report does not yet include a listening test with many listeners. Decisions that needed listening, such as choosing the architecture and the built-in voices, were made in blind comparisons by one native speaker, and our only naturalness measure is UTMOS22, which was trained on English. Intelligibility results depend on the recognizer and on how text is normalized before scoring, so we name the recognizer with every result.

## 7 How Large Does the Model Need to Be?

Alania-2 has 488M parameters, while open speech models of several billion parameters are available, so it is fair to ask why we did not build on a larger one. We did try: in our experiments, the larger models were barely more intelligible, and a far smaller model trained with the same recipe matched the best systems we measured.

## 7.1 Larger models trained on our data

Table 6 compares Alania-2 with the larger models from Section 5. Continuing to train the full VoxCPM2, at 2.3B parameters, improved CER on the built-in voices by only 0.3–0.5 points over Alania-2. The larger model did keep clear advantages elsewhere. It followed style instructions more accurately (CER 2.0% against 3.4%), it sounded somewhat more natural (UTMOS 0.16 higher on the female voice), and it cloned voices from English recordings far better (similarity 0.64 against 0.41, measured before Alania-2’s second training stage). Training a larger model from scratch did not help either: our 1.24B-parameter variant had still not learned to align text with audio after 54k steps, when the 488M model already had. At serving time, the smaller model handles 1.5 times as many requests per GPU.

Table 6 Model size against quality on our 292-clip suite (one seed; CER from Whisper large-v3, lower is better) and serving capacity. Training: “continued” means further training of VoxCPM2 on our data, “scratch” random initialization. Requests/s is the request rate at which about 99% of streams keep ahead of real time on one H200, higher is better.
<table><tr><td rowspan="2">Model</td><td colspan="7">CER (%)↓</td></tr><tr><td>Params</td><td>Training</td><td>Female</td><td>Male</td><td>Style</td><td>UTMOS↑</td><td>Requests/s↑</td></tr><tr><td>VoxCPM2, zero-shot</td><td>2.3B</td><td>released</td><td>2.7</td><td>2.9</td><td>2.2</td><td>3.77</td><td>一</td></tr><tr><td>Track 1, round 2</td><td>2.3B</td><td>continued</td><td>2.4</td><td>2.3</td><td>2.0</td><td>3.73</td><td>8</td></tr><tr><td>Track 2-L</td><td>1.24B</td><td>scratch</td><td></td><td></td><td></td><td>did not align within 54k steps (CER 84–88%)</td><td></td></tr><tr><td>Alania-2</td><td>488M</td><td>scratch</td><td>2.7</td><td>2.8</td><td>3.4</td><td>3.57</td><td>12</td></tr></table>

![](images/b7815735e0cbabdca8bf52be48b3a486ea1ed03c990f48231b741bcbcfcecbca.jpg)  
Figure 6 Freya-TR-Eval word error rate against model size for the 13 systems with published sizes (both axes logarithmic). Our models are in blue. Size does not predict intelligibility.

## 7.2 Size and intelligibility across systems

The same holds across the systems in our Freya-TR-Eval re-run (Figure 6). Model size does not predict word error rate: the three most intelligible systems with published sizes span 7.6M to 2.4B parameters, and our 3.3B-parameter Alania-1 misreads more than four times as many words as the 488M Alania-2. Training data and other design choices separate these systems far more than size does. This benchmark cannot show the efect of text handling, because its sentences contain no digits, initialisms or mixed-case brand names.

## 7.3 Going much smaller

To test how far the recipe scales down, we trained Antalia-2 Mini, an openly released model with a single voice. It has 7.62M parameters: a 3.69M-parameter acoustic model that generates mel spectrograms with few-step flow matching, and a 3.93M-parameter vocoder that produces 48 kHz audio. Like Alania-2, it learns from synthetic speech, here about 1,790 hours of one voice generated by VoxCPM2, and it reads text through the same normalizer (Section 4). On Freya-TR-Eval it reaches 1.05% WER and 0.20% CER, the lowest CER of the 15 systems we ran. Its teacher, VoxCPM2 speaking the same voice, scored 0.92% WER and 0.20% CER in a single run on our development harness. A model about 300 times smaller thus keeps nearly all of its teacher’s intelligibility.

Antalia-2 Mini also shows what the extra parameters pay for. It has one voice and no cloning, voice design or style control, and it sounds less natural than Alania-2 (UTMOS 3.25 against 3.67). For reading plain Turkish sentences in a fixed voice, a few million parameters sufice when the data are right. The larger capacity of Alania-2 goes into flexibility: many voices, cloning from a short recording, voice design and style instructions. We chose 488M as the smallest size we tried that kept these abilities while serving the most requests per GPU.

## 8 Voice Cloning

Alania-2 ofers two kinds of cloned voice. An instant clone is conditioned on a short reference recording at request time. A professional clone is a LoRA adapter trained for one voice on one to five minutes of speech. The reference recordings in this section come from consenting contributors to a Turkish speech corpus and from FLEURS (Conneau et al., 2023). We measure CER and skipped words with Whisper large-v3 and word alignment, and speaker similarity as the cosine similarity of ECAPA embeddings between the generated audio and a held-out recording of the same speaker.

## 8.1 Continuing a reference skips words

VoxCPM can use a reference recording in two ways. In continuation mode, the reference audio and its transcript come before the new text, and the model continues the recording. In reference-only mode, the reference audio sets the voice and no transcript is given.

We first served clones in continuation mode, and users reported skipped words, silent gaps and loops. With references of 20 s or more, continuation skipped about half of every text, often whole chunks, and a wrong reference transcript had the same efect at any length. Switching to reference-only mode, together with per-chunk checks (Section 9), fixed this without losing speaker identity (Figure 7). Over 200 streams with the same seeds and texts, skipped words fell from 21.9% to 1.7%, and to 0.18% on clean references. Speaker similarity on clean references was unchanged (0.569 against 0.568). The nearsilent chunks that users heard as gaps disappeared (from 45 to 0), and the longest pause where two chunks join mid-sentence fell from 1.04 s to 0.37 s.

## 8.2 How much reference audio helps

Figure 8 compares the two modes as the reference grows from 5 to 120 s (four speakers, three texts, three seeds). In reference-only mode, similarity rises from 0.56 at 5 s to 0.64– 0.67 at 20–30 s. Beyond that, the model breaks down, with a CER of 0.47 at 60 s and 0.90 at 120 s. Continuation mode is already unusable at 20 s. Choosing the cleanest 20-second window of a recording is better than taking its first 20 seconds (similarity 0.672 against 0.637, CER 0.024 against 0.032). We therefore accept references of 3–30 s, recommend 20–30 s, and automatically pick the cleanest 20-second window from longer uploads. Encoding the reference takes 31 ms at 5 s and 404 ms at 120 s, while first audio stays between 83 and 114 ms.

Continuation mode Reference-only mode

![](images/968a0db7a1999266871ee17c8ec0db456f194504c027338a01dbe9b5dd13cadf.jpg)  
Figure 7 Streamed instant clones before and after the fix; lower is better. 200 streams each, with the same seeds and texts of 121–249 characters. “Hard” references contain reverberation, music, a laptop microphone, noise or MP3 compression.

![](images/9dd1c1f639fb9668ed6c8600a9664601d44314bfdb35670ee7e7ed2d12eee35d.jpg)

![](images/0ffef42fc79a5a6a828ac04da0621b006000297cd8e2dad9af0925a1947bedc3.jpg)  
Figure 8 Instant clones as the reference grows, using the first N seconds of each recording. Reference-only mode is stable up to 30 s.

Chaining chunks does not help. We also tried conditioning each chunk of a long text on the end of the previous chunk. Over 80 paired renders, this lowered CER by 0.035–0.069 but also lowered similarity by 0.058–0.074, and the voice drifted from chunk to chunk. Every chunk therefore stays conditioned on the original reference.

Reference quality. We convert saved references to mono 48 kHz at −20 dBFS and check them for background noise, echo, clipping and implausible transcripts. Background noise only 12 dB below the speech made the model skip 29–49% of words.

## 8.3 Professional clones

A professional clone is a rank-32 LoRA adapter (Hu et al., 2022) on the backbone and the difusion decoder. The training pipeline splits the upload into 3–15 s utterances with a signal-to-noise ratio above 15 dB, transcribes them, holds out 10% for evaluation, and trains for 200 steps on 80 s of speech, rising linearly to 400 steps for five minutes. The adapter is used only if it beats the instant clone made from the best 20-second reference on the held-out speech: similarity must rise by at least 0.02, and CER must stay below max(0.05, 1.5c), where c is the instant clone’s CER.

![](images/9b0cadb8978884194a77381d8889074ce4334a50c6615722258a73ccec3b7cb0.jpg)  
Figure 9 Speaker similarity of professional clones (LoRA adapters) against instant clones of the same speakers, on held-out speech. Each row is one run; the last is the mean over four speakers through the serving engine.

![](images/3f8ed055984b8e2d15fe3db51b1bc1a3107f4b322d680328e93e453c152a87c5.jpg)  
One GPU machine, shared through CUDA MPS  
Figure 10 Serving layout on each GPU machine.

Training takes 84–126 s on one A100, and a whole job finishes in 3–4.5 minutes. Figure 9 shows large similarity gains in every run. One speaker’s adapter trained on 580 s of speech (similarity 0.68) scored below the same speaker’s 78-second adapter (0.72), which suggests that matching the recording session matters more than the amount of audio at this scale. Cloning a voice from an English recording into Turkish speech remains weak: similarity 0.41, against 0.64 for Track 1, measured before the second training stage.

## 8.4 Verification

Because a professional clone reproduces a voice closely, we require the voice’s owner to be present. The speaker reads a random Turkish sentence aloud in the browser. The sentence is drawn from about three million combinations and expires after 15 minutes. Duyu must recognize the sentence, and the speaker embedding of the reading must match the uploaded recordings. On 13 consenting speakers (84 same-speaker pairs across recordings and 2,460 diferent-speaker pairs), a threshold of 0.45 rejects 0.5% of genuine readings and accepts 0.45% of impostor pairs. In a live test, a diferent speaker’s reading scored 0.20 and was rejected within 3 s. Our API also refuses to synthesize any pending challenge sentence, so the check cannot be passed by generating the reading with an instant clone through our own service.

## 9 Serving

Each GPU machine runs the Alania-2 worker and an API front that also serves Alania-1 (Figure 10). CUDA’s Multi-Process Service (MPS) lets the two models share the GPU fairly:

without it, four Alania-1 streams next to an Alania-2 load fell from 100% to 3% keeping up with real time; with it, they held 100% next to 24 Alania-2 streams.

From text to audio. The front normalizes the text (Section 4) and splits it into chunks of at most 140 characters, preferring sentence and then clause boundaries. It then resolves the voice (built-in, designed, instant clone or professional clone) and streams the first chunk as it is generated, in steps of 0.16 s.

Checks and retries. Later chunks are generated up to two chunks ahead of playback and checked before they are sent. A chunk is rejected if it is empty; if it runs on too long (more than 2.0 s plus 0.11 s per character); if its length does not fit its text; if it contains a pause of more than 1 s; if it has no voiced speech; or, for the built-in voices, if its median pitch is outside the voice’s range. A rejected chunk is generated again with a new seed, up to twice, and the best attempt is kept. If waiting would make playback stall, the chunk is sent unchecked. When joining chunks, we trim their edge silence and insert 350 ms of silence after a sentence and 120 ms after a clause. Every response is labelled as AI-generated in its headers.

Deployment. We run on spot RTX PRO 6000 machines in one EU region, with on-demand A100 machines as a floor that is never preempted. A new machine serves Alania-2 81 s after boot. Professional clones are trained on a separate, always-on GPU machine in the EU.

## 10 Inference Engine

The worker runs Alania-2 on Nano-vLLM-VoxCPM (a710128, 2025), an inference engine for VoxCPM based on Nano-vLLM (GeeeekExplorer, 2025). It batches concurrent streams and keeps the backbone’s key/value cache in fixed-size blocks (Kwon et al., 2023). We changed the engine in three places to cut first-audio latency and raise throughput, and in several more to serve many fine-tuned voices from one model. We checked each change against the unmodified engine on the same seeds and texts.

## 10.1 Latency and throughput

Table 7 lists the three changes:

• Batched VAE decoding. The audio decoder ran once per stream. Running it once per batch cut decoding at 96 streams from 110 to 27 ms per step. At 48 streams, throughput rose from 41 to 74 seconds of audio per second, and the share of streams that kept ahead of real time rose from none to all.

• CUDA-graph prefill. We capture the prompt step as CUDA graphs, as the engine already did for later steps. Single-stream first audio fell from 74 to 32 ms, and the output stayed bit-identical (23 of 23 renders).

• Kernel warm-up. A batch size the engine had not seen before stalled every stream for about 1.4 s. We now run the common batch sizes once at start-up.

Figure 11 shows the efect of the prefill change as load grows, with batched decoding already in place. With all three changes, median first audio is 32 ms for one stream and 94 ms for 96 concurrent streams, which together produce 136 seconds of audio per second.

![](images/4182a78104c6e37308a0a9605e5f1d51618c3a4582fa8e701421106b8d5e5208.jpg)

Table 7 Engine changes, measured on one A100 (40 GB) at the worker, excluding the network.
<table><tr><td>Change</td><td>Metric</td><td>Before</td><td>After</td></tr><tr><td>Batched VAE decoding</td><td>VAE decode per step, 96 streams (ms) ↓</td><td>110</td><td>27</td></tr><tr><td></td><td>Audio seconds per second, 48 streams ↑</td><td>41</td><td>74</td></tr><tr><td>CUDA-graph prefill</td><td>First audio, 1 stream (ms) ↓</td><td>74</td><td>32</td></tr><tr><td></td><td>First audio, 96 streams (ms) ↓</td><td>143</td><td>94</td></tr><tr><td>Kernel warm-up</td><td>Stall on a new batch size (s) ↓</td><td>~1.4</td><td>none</td></tr></table>

Figure 11 Latency and throughput on one A100 as concurrent streams grow, before and after CUDA-graph prefill. Batched VAE decoding is on in both.

Through the API on the same machine, a single stream starts in 97 ms for a fixed voice, 109 ms for a clone and 78 ms for a designed voice. On an H200 with Poisson arrivals of ∼5- s requests, median first audio is 51 ms at 2 requests per second and 75 ms at 12 requests per second, where 98.6% of streams still keep ahead of real time.

## 10.2 Serving many fine-tuned voices

Professional clones (Section 8) are LoRA adapters on one shared base model, as in S-LoRA (Sheng et al., 2024) and Punica (Chen et al., 2024). Each request names its adapter. Up to eight adapters stay on the GPU, and requests with diferent adapters, or none, share a batch. Adapters load from storage on first use and are cached in host memory and on disk. A broken adapter falls back to the instant clone of the same voice.

Prefix-cache correctness. The engine reuses cached key/value blocks when a new request starts with the same tokens as an earlier one. Its cache key covered only the tokens, so a request for a fine-tuned voice could reuse blocks that a plain request had computed from the same reference and text. We saw 8 of 9 prompt blocks reused this way, and 3 of 3 fine-tuned renders difered from renders with an empty cache. We now seed the cache key with the adapter’s name and a digest of its weights. Fine-tuned renders then match empty-cache renders bit for bit, and plain requests keep their existing keys.

Two smaller fixes. We replaced a LoRA kernel that would not compile on our Triton version; outputs agree to within $2 \times 1 0 ^ { - 7 }$ . Batches with a single adapter now run as plain matrix products, which cut the adapter’s first-audio overhead from 33 to 13 ms.

With these fixes, quality through the engine matches the reference implementation (Table 8). In mixed batches, every render was closest to its own speaker (60 of 60), and

Table 8 Professional clones through the engine (4 speakers × 5 texts × 3 seeds). Speaker similarity ↑ / CER ↓.
<table><tr><td>Path</td><td>Instant clone</td><td>Professional clone</td></tr><tr><td>Reference implementation</td><td>0.651/0.019</td><td>0.770/0.026</td></tr><tr><td>Our engine</td><td>0.659 /0.019</td><td>0.772/0.028</td></tr><tr><td>Our engine, mixed batches</td><td>0.676/0.023</td><td>0.780/0.026</td></tr></table>

throughput with every stream on an adapter is 4–15% below plain clones.

## 11 Limitations

Our evaluation has several gaps. First, naturalness rests on UTMOS22, a predictor trained on English speech, and on blind comparisons by one native listener. Second, intelligibility depends on the speech recognizer and on the text normalization used for scoring, and the same recognizers also helped build our training data. Third, ECAPA speaker similarity separates speakers on our data, but diferences below about 0.05 should not be over-read: a style instruction alone moves similarity to the plain voice from about 0.91 to between 0.64 and 0.80. Fourth, most results come from a single seed or a single run.

The model itself has limits. Ninety percent of its training audio comes from VoxCPM2, so Alania-2 inherits its teacher’s prosody and artefacts, including narrow high-frequency tones in some voices that we filter out of the synthetic data but do not yet suppress at serving time. The base training text was also normalized before the native-speaker review of Section 4, so the model now receives some readings, including all brand respellings, that it never saw in training. Finally, exact GPU-hours and some early training configurations were not recorded.

## 12 Ethics and Safety

Training data. The real speech comes from public YouTube videos distributed in YODAS under Creative Commons terms. Our synthetic text includes Wikipedia (CC BY-SA) and FineWeb2 (ODC-By). The built-in voices are designed voices that do not belong to any real person, and our released synthetic speech contains designed voices only. We do not release the model weights.

Cloning. Cloning someone’s voice without their agreement can enable fraud, harassment and disinformation. Every clone requires the user to state that they have the speaker’s consent, and we record this statement with the voice. Professional clones also require the live reading and speaker match of Section 8, with separate consent to the biometric comparison. Speaker embeddings are computed in memory and never stored, uploaded recordings are deleted seven days after training, customer voice data stays in the EU, and every response is labelled as AI-generated.

These controls have gaps. Instant clones have no live verification, so only the user’s statement of consent stands in the way of misuse. We do not yet watermark generated audio (San Roman et al., 2024), and we have no specific policy on the voices of public figures beyond requiring consent. Every voice in this report’s cloning experiments belongs to a consenting contributor or comes from FLEURS.

## 13 Conclusion

We presented Alania-2, a 488M-parameter Turkish TTS model trained mostly on synthetic speech. On its built-in voices, it comes within half a point of CER of a fine-tuned model more than four times its size, generates 48 kHz audio, and starts streaming in tens of milliseconds. Larger models bought little intelligibility, and a 7.6M-parameter model trained with the same recipe matched the best systems we measured; extra parameters pay for flexibility rather than accuracy. Several of the most useful lessons concerned the system around the model. Alania-2, trained on normalized text, needs a text normalizer to read digits and abbreviations. Cloning by continuing a reference fails with long references, while conditioning on the reference audio alone does not, and 20–30 seconds of reference audio works best. A low-rank adapter trained in about two minutes gives a close voice match, and serving many such adapters needs care with caching. Next, we plan a listening test with many native listeners, audio watermarking, live verification for instant clones, and retraining on text from the current normalizer.

## Data and Code Availability

The Freya-TR-Eval harness and per-system results are public at https://huggingface.co/ datasets/PatientdeskAI/freya-tr-eval-rerun. The caption, synthetic speech and domain text datasets are linked in Section 3. Antalia-2 Mini is public at https://huggingface.co/ cloud0day3/antalia-mini. The weights of Alania-2 and Duyu and our serving code are not public; both models can be used through our speech API at https://speech.patientdesk.ai.

## Competing Interests

The authors are afiliated with PatientDesk AI, which ofers Alania-2 and Duyu through a commercial speech API and develops Antalia-2 Mini. Duyu, used for filtering and in some evaluations in this report, is the authors’ own model.

## References

a710128. Nano-vLLM-VoxCPM: An inference engine for VoxCPM based on Nano-vLLM. https://github.com/a710128/nanovllm-voxcpm, 2025.

Edresson Casanova, Kelly Davis, Eren Gölge, Görkem Göknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, and Julian Weber. XTTS: a massively multilingual zero-shot text-to-speech model. In Interspeech 2024, pages 4978–4982, 2024. doi: 10.21437/Interspeech.2024-2016.

Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo, Luis Ceze, and Arvind Krishnamurthy. Punica: Multi-tenant LoRA serving. In Proceedings ofMachine Learning and Systems (MLSys), volume 6, pages 1–13, 2024.

Sanyuan Chen, Chengyi Wang, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, and Furu Wei. Neural codec language models are zero-shot text to speech synthesizers. IEEE Transactions on Audio, Speech and Language Processing, 33:705–718, 2025a. doi: 10.1109/TASLPRO.2025.3530270.

Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, and Xie Chen. F5-TTS: A fairytaler that fakes fluent and faithful speech with flow matching. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 6255–6271, 2025b. doi: 10.18653/v1/2025.acl-long.313.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. FLEURS: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805, 2023. doi: 10.1109/SLT54892.2023.10023141.

Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification. In Interspeech 2020, pages 3830–3834, 2020. doi: 10.21437/Interspeech.2020-2650.

Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, Zhifu Gao, and Zhijie Yan. CosyVoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv:2407.05407, 2024.

GeeeekExplorer. Nano-vLLM: A lightweight vLLM implementation built from scratch. https://github.com/GeeeekExplorer/nano-vllm, 2025.

Yuan Gong, Yu-An Chung, and James Glass. AST: Audio spectrogram transformer. In Interspeech 2021, pages 571–575, 2021. doi: 10.21437/Interspeech.2021-698.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon. Glow-TTS: A generative flow for text-to-speech via monotonic alignment search. In Advances in Neural Information Processing Systems, volume 33, pages 8067–8077, 2020.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), pages 611–626, 2023. doi: 10.1145/3600006.3613165.

Xinjian Li, Shinnosuke Takamichi, Takaaki Saeki, William Chen, Sayaka Shiota, and Shinji Watanabe. YODAS: Youtube-oriented dataset for audio and speech. In 2023 IEEEAutomatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8, 2023. doi: 10.1109/ASRU57964.2023.10389689.

Yunus Emre Ozkose, Alperen Kahraman, and Ali Haznedaroglu. A comprehensive objective evaluation of modern text-to-speech for Turkish using speech quality assessment models. In Speech and Computer (SPECOM 2026), volume 16934 of Lecture Notes in Computer Science, pages 456–472. Springer, 2026. doi: 10.1007/978-3-032-37867-5\_31.

Ahmet Erdem Pamuk, Ömer Yentür, Ahmet Tunga Bayrak, Yavuz Alp Sencer Öztürk, and Mustafa Yavuz. FreyaTTS: A compact tokenizer-free flow-matching transformer for Turkish-first speech synthesis. arXiv:2607.09530, 2026.

Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Rafel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. FineWeb2: One pipeline to scale them all – adapting pre-training data processing to every language. arXiv:2506.20920, 2025.

Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, and Michael Auli. Scaling speech technology to 1,000+ languages. arXiv:2305.13516, 2023.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings ofthe 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings ofMachine Learning Research, pages 28492–28518, 2023.

Mirco Ravanelli, Titouan Parcollet, Peter Plantinga, Aku Rouhe, Samuele Cornell, Loren Lugosch, Cem Subakan, Nauman Dawalatabad, Abdelwahab Heba, Jianyuan Zhong, Ju-Chieh Chou, Sung-Lin Yeh, Szu-Wei Fu, Chien-Feng Liao, Elena Rastorgueva, François Grondin, William Aris, Hwidong Na, Yan Gao, Renato De Mori, and Yoshua Bengio. SpeechBrain: A general-purpose speech toolkit. arXiv:2106.04624, 2021.

Chandan K. A. Reddy, Vishak Gopal, and Ross Cutler. DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In ICASSP 2021 – 2021 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 6493–6497, 2021. doi: 10.1109/ICASSP39728.2021.9414878.

Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari. UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022. In Interspeech 2022, pages 4521–4525, 2022. doi: 10.21437/Interspeech.2022-439.

Robin San Roman, Pierre Fernandez, Alexandre Défossez, Teddy Furon, Tuan Tran, and Hady Elsahar. Proactive detection of voice cloning with localized watermarking. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings ofMachine Learning Research, pages 43180–43196, 2024.

Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, and Ion Stoica. SLoRA: Scalable serving of thousands of LoRA adapters. In Proceedings ofMachine Learning and Systems (MLSys), volume 6, pages 296–311, 2024.

Hubert Siuzdak, Florian Grötschla, and Luca A. Lanzendörfer. SNAC: Multi-scale neural audio codec. arXiv:2410.14411, 2024.

Jörgen Valk and Tanel Alumäe. VoxLingua107: A dataset for spoken language recognition. In 2021 IEEE Spoken Language Technology Workshop (SLT), pages 652–658, 2021. doi: 10.1109/SLT48900.2021.9383459.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv:2505.09388, 2025.

Rustem Yeshpanov, Saida Mussakhojayeva, and Yerbolat Khassanov. Multilingual text-to-speech synthesis for Turkic languages using transliteration. In Interspeech 2023, pages 5521–5525, 2023. doi: 10.21437/Interspeech.2023-249.

Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Ziyang Wang, Runchuan Ye, Weiyue Sun, Jiancheng Gui, Kehan Li, Zhiyong Wu, and Zhiyuan Liu. VoxCPM: Tokenizer-free TTS for context-aware speech generation and true-to-life voice cloning. arXiv:2509.24650, 2025. Published as “Hierarchical Semantic-Acoustic Modeling via Semi-Discrete Residual Representations for Expressive End-to-End Speech Synthesis” at ICLR 2026.

Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Jiancheng Gui, Jiaheng Wu, Ziyang Wang, Xudong Shen, Runchuan Ye, Zhisheng Zhang, Jiuyang Zhou, Bingsong Bai, Weiyue Sun, Mengyuan Deng, Qundong Shi, Zhiyong Wu, and Zhiyuan Liu. VoxCPM2 technical report. arXiv:2606.06928, 2026.

## A Full Candidate Results

Table 9 Candidates on the 292-clip suite (one seed). CER from Whisper large-v3, UTMOS22, ECAPA similarity to the voice reference; “gender” counts voice designs that matched the requested gender.
<table><tr><td rowspan="2">Model</td><td colspan="3">Female voice</td><td colspan="3">Male voice</td><td colspan="3">Style</td><td>Design</td></tr><tr><td>CER</td><td>UTMOS</td><td>Sim</td><td>CER</td><td>UTMOS</td><td>Sim</td><td>CER</td><td>UTMOS</td><td>Sim</td><td>Gender</td></tr><tr><td>VoxCPM2, zero-shot (2.3B)</td><td>.027</td><td>3.77</td><td>.748</td><td>.029</td><td>3.53</td><td>.812</td><td>.022</td><td>2.96</td><td>.542</td><td>23/32</td></tr><tr><td>Track 1, round 1 (2.3B)</td><td>.028</td><td>3.77</td><td>.690</td><td>.027</td><td>3.54</td><td>.806</td><td>.022</td><td>3.66</td><td>.667</td><td>25/32</td></tr><tr><td>Track 1, round 1 LoRA</td><td>.024</td><td>3.78</td><td>.707</td><td>.028</td><td>3.60</td><td>.810</td><td>.037</td><td>3.71</td><td>.714</td><td>23/32</td></tr><tr><td>Track 1, round 2 (2.3B)</td><td>.024</td><td>3.73</td><td>.761</td><td>.023</td><td>3.50</td><td>.737</td><td>.020</td><td>3.13</td><td>.534</td><td>27/32</td></tr><tr><td>Track 2, first run (488M)</td><td>.061</td><td>3.48</td><td>.768</td><td>.055</td><td>3.12</td><td>.723</td><td>.142</td><td>3.21</td><td>.663</td><td>30/32</td></tr><tr><td>Track 2, base 70k (488M)</td><td>.027</td><td>3.64</td><td>.773</td><td>.028</td><td>3.28</td><td>.739</td><td>.090</td><td>3.29</td><td>.666</td><td>24/32</td></tr><tr><td>Alania-2 (fine-tune 4k)</td><td>.027</td><td>3.57</td><td>.766</td><td>.028</td><td>3.24</td><td>.741</td><td>.034</td><td>3.23</td><td>.633</td><td>30/32</td></tr><tr><td>Fine-tune 12k</td><td>.027</td><td>3.62</td><td>.769</td><td>.028</td><td>3.26</td><td>.736</td><td>.044</td><td>3.22</td><td>.612</td><td>30/32</td></tr></table>

## B Systems in the Freya-TR-Eval Re-run

Table 10 Systems in our Freya-TR-Eval re-run (Figure 5), with the exact checkpoint or service we ran. Model identifiers are Hugging Face repositories unless noted. Parameter counts are those in our released results; “–” means not published.
<table><tr><td>System</td><td>Params</td><td>Checkpoint or service</td><td>Paper</td></tr><tr><td>EMA Lightning</td><td>8.6M</td><td>ema-lightning 1.0.1 (PyPI), canberkkkkkk/ema-lightning</td><td>一</td></tr><tr><td>Antalia-2 Mini</td><td>7.62M</td><td>1.0.0,cloud0day3/antalia-mini</td><td>Section 7</td></tr><tr><td>Trendyol-TTS</td><td>2.38B</td><td>Trendyol/Trendyol-TTS</td><td>一</td></tr><tr><td>Gemini 3.8 Flash TTS</td><td></td><td>Vertex Al, gemini-3.8-flash-tts, voice Kore</td><td></td></tr><tr><td>Gemini 3.8 Flash-Lite TTS</td><td></td><td>VertexAl, gemini-3.8-flash-lite-tts,voice</td><td>一</td></tr><tr><td>Anka TTS</td><td>336M</td><td>Kore anka-tts v0.1 (PyPI)</td><td></td></tr><tr><td>Alania-2</td><td>488M</td><td>our API, and the bare model</td><td>this report</td></tr><tr><td>Chatterbox Multilingual</td><td>500M</td><td>resemble-ai/chatterbox (GitHub)</td><td>一</td></tr><tr><td>Piper</td><td>16M</td><td>tr_TR-dfki-mediumfrom rhasspy/piper-voices</td><td></td></tr><tr><td>XTTS-v2</td><td>470M</td><td>coqui/XTTS-v2</td><td>Casanova et al. (2024)</td></tr><tr><td>Alania-1</td><td>3.3B</td><td>our API</td><td></td></tr><tr><td>Antalia-1</td><td>305M</td><td>cloud0day3/antalia-1</td><td></td></tr><tr><td>MMS-TTS</td><td>36M</td><td>facebook/mms-tts-tur</td><td>Pratap et al. (2023)</td></tr><tr><td>Coqui GlowTTS</td><td>28M</td><td>tts_models/tr/common-voice/glow-tts (Coqui</td><td>Kim et al. (2020)</td></tr><tr><td>FreyaTTS-small</td><td>183M</td><td>TTS) freyavoice/Freya-TTS</td><td>Pamuk et al. (2026)</td></tr></table>