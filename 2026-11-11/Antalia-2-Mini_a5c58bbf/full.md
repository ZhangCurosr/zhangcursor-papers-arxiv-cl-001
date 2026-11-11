# Antalia-2 Mini

A 7.6-million-parameter Turkish text-to-speech model trained entirely on synthetic speech

Sezgin Saygili Emre Kaplaner Oncel Ozgul Fikri San Koktas

PatientDesk AI sezgin@patientdesk.ai

7.62M 15.3 MB 48 kHz 0.22% 41×

parameters optional fp16 weights audio output character error rate, public harness real time, M5 CPU, 2 threads

Model and weights Browser demo pip install antalia-mini Benchmark Apache-2.0

Abstract Antalia-2 Mini is a single-voice Turkish text-to-speech model with 7,620,931 parameters. It reads characters and produces 48 kHz audio at 41 times real time on an Apple M5 laptop CPU (2 threads). Its training audio came only from VoxCPM2 (the teacher), about 300 times larger, cloning one synthetic clip; automatic checks kept about 1,790 h. Fine-tuning the vocoder on the acoustic model’s own mel spectrograms cut the word error rate (WER) of Whisper’s transcripts from a mean of 3.89% to 1.75% at adjacent pre-release checkpoints. The teacher corrupts the start of a clip when the first word begins with ${ \ddot { o } } ,$ the student learned this, and a spoken prefix we cut from the output hides it. On the 495-sentence Freya-TR-Eval benchmark, where we tuned settings, version 1.0.1 has WER 1.09% and character error rate 0.22% on our public harness. A paired bootstrap cannot distinguish these from EMA Lightning, an 8.6M Turkish model, which scores higher on full-band UTMOS, an English-trained quality proxy. No human listening test exists. Weights, inference code and the benchmark harness are public under Apache-2.0; training code and data are not.

Input Randevunuz 14 Ekim Salı günü saat 15:30'da.

Model reads randevunuz on dört ekim salı günü saat on beş otuzda.

![](images/83e20890cdce9efa8664bc1a7c65ef940992ff859227f0328627d4f8e4a9a0bc.jpg)  
Figure 1 The whole model on one real sentence (version 1.0.1, seed 0). The normalizer writes the numbers out, the encoder and duration predictor plan the timing, the generator draws a mel spectrogram from noise in 8 steps, and the vocoder turns it into audio.

## 1 Introduction

Antalia-2 Mini has 7.62M parameters, reads characters and learned only from the audio of a model about 300 times larger. On one benchmark a paired bootstrap cannot distinguish its word and character error rates from that teacher’s or EMA Lightning’s, and no human listening test of it exists yet.

A text-to-speech (TTS) model small enough for a CPU can run on the machine that needs the speech, with no GPU server and no text leaving the device. Among the Turkish systems we ran, two small open-weights models, Piper (voice dfki) at 16M parameters and MMS-TTS at 36M, have word error rates (WER) of 3.41% and 6.38%, against 1.16% for Trendyol-TTS at about 2.38B. We get a WER by having the Whisper speech recognizer (Radford et al., 2023) transcribe a model’s audio and comparing the transcript with the input text, here on the public harness of the Freya-TR-Eval benchmark. Antalia-2 Mini (Mini below) aims for the low error rates of large systems at 7.62M, in one voice.

Its default weights are 30.5 MB at fp32, with an optional 15.3 MB fp16 download (Table 7). A browser page and a signed Android engine also run it on the device, but no real phone has been measured.

Mini reads characters, 47 symbols in all, with no phonemizer to turn spelling into sound symbols, and learns how letters sound from its training audio. Turkish spelling suits this choice. Öney and Durgunoğlu (1997) describe it as phonologically transparent, with regular letter-sound correspondences, and Babayiğit (2022) names Ğ as the exception. A text normalizer first turns numbers, dates, times, amounts, initialisms, web addresses and English or brand words into spoken form (Table 2), and its known defects carry into the speech.

All of Mini’s training audio came from one model, VoxCPM2 (Zhou et al., 2026), roughly 300 to 313 times its size. We had this teacher read text of our choosing in one synthetic voice, so every clip came with the text it was asked to read. Mini, the student, trains on those clips with no access to the teacher’s internal features. Section 9 traces this idea back at least to TTS-by-TTS (Hwang et al., 2021). Teacher errors become training targets, as Pipatanakul et al. (2026) state, and Section 6 documents one that Mini learned.

Among the other systems we ran, EMA Lightning (Aslan, 2026) is nearest ours in size and recipe. Its card says it was also trained only on synthetic speech from VoxCPM2, about 1,000 h, so the two models share a teacher model, with diferent voices. Our scoring protocol follows that card. The vocoder is the network that turns a mel spectrogram into audio, and the acoustic model before it turns text into that mel. Mini has 10.93% fewer parameters, 7,620,931 against 8,556,172, with the vocoder in both counts.

We report six results, and Figure 2 traces the recipe behind them. The terms box defines the words they use.

• On the 495 sentences of Freya-TR-Eval (Pamuk et al., 2026), Mini 1.0.1 has WER 1.09% and CER 0.220% on the public harness. A paired 95% bootstrap distinguishes it from neither EMA Lightning 1.0.1 nor, on our own harness, the teacher (1 seed), and both score higher than Mini on full-band UTMOS, an Englishtrained proxy. We tuned Mini’s settings on these sentences, so final-settings numbers are optimistic by selection (Section 7).

<table><tr><td colspan="2">Terms used in this report</td></tr><tr><td>Mel</td><td>A mel spectrogram, audio as energy per frequency band over time, with bands spaced to follow human pitch perception. Ours has 128 bands and 93.75 frames per second. Real mels come from the teacher&#x27;s audio, generated mels from the acoustic model.</td></tr><tr><td>Copy synthesis</td><td>Computing a mel from existing audio, here the teacher&#x27;s, and passing it through the vocoder alone. It shows what the vocoder loses by itself.</td></tr><tr><td>Teacher, student</td><td>The teacher, unmodified VoxCPM2, generated all training audio. The student, Antalia-2 Mini, trained on its clips.</td></tr><tr><td>Long set, short set</td><td>Training clips of sentences up to about 25 s, and of short utterances, mostly fragments of 2 to 6 words.</td></tr><tr><td>Runs</td><td>Acoustic model: the main run on the long set, then the short-clip fine-tune on both sets; a sibling run on filtered clips was stopped. Vocoder: the real-mel vocoder, then the first and final vocoder fine-tunes on generated mels (Table 9).</td></tr><tr><td>Versions 1.0.0 and 1.0.1</td><td>Two public releases with identical weights. Version 1.0.1 changes only the default output chain, the fixed signal processing after the vocoder.</td></tr><tr><td>Final settings</td><td>8 sampling steps, guidance 2.0, temperature 0.6, speed 0.95, hidden prefix and suffix, doubling rule 1.3 (Sections 2 and 6).</td></tr><tr><td>Alania, Antalia</td><td>Our model families: Alania models are closed and sold through an API, and Antalia models have open weights.</td></tr><tr><td>Take</td><td>One render of all 495 benchmark sentences.</td></tr><tr><td>Public harness</td><td>Our test setup for Freya-TR-Eval with public code: 3 takes, seed 100000 s + i for sentence i of take s, one NVIDIA L4 each for Mini and EMA Lightning (different VMs).</td></tr><tr><td>Own harness</td><td>Our second setup, with the same scoring and private code: 3 takes, each with one seed for all its sentences (seeds 0, 1, 2). Mini 1.0.1 ran them on an L40S, an A10G and</td></tr><tr><td>Band matching</td><td>an A100, EMA Lightning on an H200, and the teacher 1 seed on an A100. Audio is resampled down to 8 kHz and back up to 16 kHz before Whisper hears it, as the benchmark&#x27;s card recommends.</td></tr><tr><td>WER, CER</td><td>Word and character error rate: word or character edit distance between Whisper&#x27;s transcript and the reference, over reference length, in percent. On the benchmark,</td></tr><tr><td>Points</td><td>Whisper (large-v3, beam 5) hears band-matched audio and each take gets one score. Percentage points, our unit for differences: 1.09% against 1.04% is 0.05 points.</td></tr><tr><td>Not distinguishable</td><td>The paired 95% interval of a difference includes 0. Figures also say not separable or within noise. “Level with” is a stricter test (Section 7).</td></tr><tr><td>One seed&#x27;s span</td><td>About ±0.13 WER points, the standard deviation across Mini 1.0.0&#x27;s 3 own-harness seeds. The ±0.27 half-width of Section 7 comes from resampling sentences instead</td></tr><tr><td>Clean rate UTMOS</td><td>The share of sentences with no word error in a take, averaged over takes. UTMOS22-strong, an automatic estimate of the mean opinion score (MOS), the average rating listeners give, trained on English speech; higher is better. For scale, the teacher scores 3.42 on the benchmark (full band, own harness, 1 seed). It runs on</td></tr><tr><td>RTF, times real time</td><td>16 kHz audio, so neither UTMOS nor WER hears anything above 8 kHz. Full-band UTMOS skips band matching and 8 kHz UTMOS includes it. Real-time factor is wall-clock synthesis time over audio duration. Times real time is its inverse, so RTF 0.024 is 41×.</td></tr></table>

• Fine-tuning the vocoder on generated mels cut WER from a mean of 3.89% to 1.75% at adjacent pre-release checkpoints (7 pairs, same seed, our own harness), while a frozen vocoder stayed at 3.63–4.14% (Section 5).

• By Whisper, the teacher gets first words right on 26% of 12,554 kept ö-initial long-set lines, against 94% for other capital first letters. The student learned this defect, and a spoken prefix cut from the output hides it without removing it (Section 6).

![](images/081a10b14d3c59a6b44466de5fa26187dcc50e504ea2f4a34b8963dc5a6064b6.jpg)  
Figure 2 All training audio came from unmodified VoxCPM2 cloning one synthetic clip, and the vocoder was fine-tuned on the acoustic model’s own mels. Bars are proportional within each node; generated and kept hours share one scale. F0 is pitch, ECAPA identity is a speaker embedding’s similarity to the voice, and cfg is classifier-free guidance, a sampler setting. Coral shows a Whisper rate on kept long-set lines.

• A mixed-precision weight cache (bf16 autocast) silently gave the generator, the part that draws the mel, no gradients for the first 6,000 steps of the main run (Section 4).

• One machine with 8 GPUs generated all 1,361,734 clips. Automatic checks kept about 1,790 h, but Whisper checked only 7,500 of the 444,437 short clips (Section 3).

• Mini 1.0.1 runs at 35× real time on one NVIDIA L4 with 8 vCPUs (all 495 sentences), against 58× for EMA Lightning 1.0.1 in plain PyTorch and 331× compiled, with 16 vCPUs. Mini 1.0.0 was 1.36 times faster than plain EMA Lightning 1.0.1 on one 8-vCPU AMD EPYC (first 100 sentences), and 1.0.1 is untimed there (Section 8).

## 2 The model

An acoustic model reads characters, its generator draws a mel spectrogram in 8 steps, and a vocoder turns the mel into 48 kHz audio in one pass. At the default settings the generator, trained with shortcut self-consistency, does 84.0% of the multiply-accumulates, and the model also renders hidden text that is cut from the output.

Table 1 lists the model’s 7,620,931 parameters at inference.

Text input. The acoustic model reads 47 symbols: padding, the BOS and EOS markers for the start and end of a sequence, the space, 8 punctuation marks, 32 letters (29 Turkish plus q, w and x) and 3 circumflex vowels. It has no phonemes and no digits, so a normalizer inside the package first writes everything else in these symbols (Table 2). Our checklist records that no native speaker has reviewed address readings such as dabılyu and bölü.

Table 1 The vocoder holds 51.62% of the parameters and does 14.4% of the operations; the generator holds 35.24% and does 84.0%. Shipped parts are counted from the headers of the released weight files. Operations are multiply-accumulates at the defaults on a 5.11 s sentence, from forward hooks on linear and convolution layers and a formula for attention, without LayerNorm, activations, the inverse Fourier transform or the output chain. The Generator row’s 84.0% leaves out its condition ConvNeXt, which does a further 1.1%. Counts of the pieces are in parentheses. No part has dropout layers. dBFS is decibels relative to digital full scale, and Q sets a filter’s bandwidth.
<table><tr><td>Part</td><td>Built from</td><td>Parameters</td><td>Share</td><td>Operations</td></tr><tr><td>Text encoder</td><td>Embedding 47 × 192 (9,024); 3 ConvNeXt blocks, hidden 384, kernel 5 (449,280); attention block, 4 heads (369,408); LayerNorm (384)</td><td>828,096</td><td>10.87%</td><td>0.5%</td></tr><tr><td>Duration predictor</td><td>Convolution from 192 to 128; 3 convolutions of width 128, kernel 3; final convolution</td><td>173,441</td><td>2.28%</td><td>0.1%</td></tr><tr><td>Generator</td><td>5 DiT blocks, width 192, 4 heads, SwiGLU hidden 512 (2,217,600); condition ConvNeXt, hidden 384, kernel 7 (150,144); adaLN-single projection (148,608); input projection (86,208); output projection (49,408); time MLP (32,896); final table</td><td>2,685,440</td><td>35.24%</td><td>84.0%</td></tr><tr><td>Acoustic model</td><td>(384); null condition (192) The three parts above</td><td>3,686,977</td><td>48.38%</td><td></td></tr><tr><td>Vocoder</td><td>Input convolution from 128 to 256, kernel 7 (229,632); 8 ConvNeXt blocks, width 256, hidden 768, kernel 7 (3,176,448); head to 2,050 values (526,850); 2 LayerNorms (512 each). Receptive field 55 frames, 587 ms</td><td>3,933,954</td><td>51.62%</td><td>14.4%</td></tr><tr><td colspan="5">Total at inference</td></tr><tr><td>Training only, not shipped MAS prior</td><td></td><td></td><td></td><td>100%</td></tr><tr><td>Discriminators</td><td>Linear layer from the encoder&#x27;s output to the mel bands</td><td>24,704</td><td></td><td></td></tr><tr><td></td><td>5 period, at periods 2, 3, 5, 7, 11 (41,105,770); 4 spectrogram, at FFT sizes 512, 1,024, 2,048, 4,096 (203,528)</td><td>41,309,298</td><td></td><td></td></tr><tr><td colspan="5">Fixed signal processing in version 1.0.1, no learned parameters</td></tr><tr><td>Equalizer</td><td>High-pass at 80 Hz, Q 0.707; low shelf of –4 dB at 200 Hz; peaks of –2.5 dB at 350 Hz, Q 1.2, and +3 dB at 3.2 kHz, Q 0.9; high shelf of +2 dB at 8 kHz. Runs as one 1,744-tap convolution at 48 kHz</td><td></td><td></td><td></td></tr><tr><td>De-esser</td><td>Acts on the band above 5.5 kHz; threshold –34 dBFS, ratio 3, attack 1.5 ms, release 60 ms; off at 8 kHz output</td><td></td><td></td><td></td></tr><tr><td>Gain, limiter</td><td>Fixed +1.6 dB; tanh knee above –1 dBFS, ceiling –0.1 dBFS. No per-utterance</td><td></td><td></td><td></td></tr><tr><td>Output rates</td><td>loudness normalization and, by default, no silence trimming 48 kHz native; 24, 16 and 8 kHz by Kaiser-windowed sinc resampling</td><td></td><td></td><td></td></tr></table>

Audio features. A mel frame covers 2,048 samples of 48 kHz audio. Frames start 512 samples apart (10.667 ms), and the bands follow the Slaney mel scale from 0 to 24 kHz. We normalize as � = (ln max(mel, 10<sup>−5</sup>) + 5.9)/2.6. The generator stacks frames in pairs, so a token lasts 21.333 ms and holds 256 values.

Encoder, alignment, durations. The text encoder is an embedding followed by 3 ConvNeXt blocks (Liu et al., 2022), a kind of convolutional residual block, and one self-attention block with rotary position embeddings, or RoPE (Su et al., 2024).

Nothing in a training clip says which frames belong to which character. Monotonic alignment search, MAS (Kim et al., 2020), finds this. A training-only layer, the prior, maps each encoder output to an expected mel frame, and a dynamic program picks the most likely left-to-right assignment, with at least one frame per symbol. A small convolutional duration predictor learns the resulting lengths, with gradients stopped at the encoder output, and replaces MAS at synthesis.

Table 2 The normalizer writes digits, amounts, initialisms and addresses as words, and it has known defects. Real calls of Antalia.normalize, package 1.0.1. The right column is the exact string returned, before the hidden prefix and sufix added at synthesis. A word counts as English when it is in the 126,052-entry CMU pronouncing dictionary (Carnegie Mellon University, 2014), a runtime dependency, and not in a list of 225,764 words seen in Turkish text.
<table><tr><td>Case</td><td>Input</td><td>Model text</td></tr><tr><td>Date, time, title</td><td>Randevunuz 14 Ekim Salı günü saat 15:30&#x27;da, Dr. Ayşe Yılmaz ile.</td><td>randevunuz on dört ekim salı günü saat on beş otuzda, doktor ayşe yılmaz ile.</td></tr><tr><td>Area, plus sign, amount</td><td>Oda 45 m² ve 3+1, fiyatı 2,5 milyon TL.</td><td>oda kırk beş metrekare ve üç artı bir, fiyatı iki buçuk milyon lira.</td></tr><tr><td>Dose, blood pressure</td><td>Günde 2x1 tablet, 500 mg; tansiyonunuz 120/80 mmHg.</td><td>günde iki defa bir tablet, beş yüz miligram; tansiyonunuz yüz yirmiye seksen milimetre civa.</td></tr><tr><td>Initialisms with suffixes</td><td>SGK ve KDV kesintileri için THY’nin CEO&#x27;su TBMM&#x27;de konuştu.</td><td>segeka ve kadeve kesintileri için teheye’nin siio&#x27;su tebememe&#x27;de konuştu.</td></tr><tr><td>Brand names</td><td>PatientDesk&#x27;e WhatsApp ya da ChatGPT üzerinden yazabilirsiniz.</td><td>peyşınt desk&#x27;e whatsapp ya da çet ci pi ti üzerinden yazabilirsiniz.</td></tr><tr><td>Web address</td><td>Ayrıntılar için www.ornek.com.tr/kampanya adresini ziyaret edin.</td><td>ayrıntılar için dabılyu dabılyu dabılyu nokta ornek nokta kom nokta te re bölü kampanya adresini ziyaret edin.</td></tr><tr><td colspan="3">Known defects seen in real calls</td></tr><tr><td>Suffix -ki takes the wrong vowel (should be otuzdaki)</td><td>Saat 10:30&#x27;daki toplantı iptal.</td><td>saat on otuzdakı toplantı iptal.</td></tr><tr><td>Range with currency keeps the hyphen</td><td>Fiyat 500-750 TL arası.</td><td>fiyat beş yüz-yedi yüz elli lira arası.</td></tr><tr><td>Unit mg/dL not handled</td><td>Kan şekeri 110 mg/dL çıktı.</td><td>kan şekeri yüz on miligram dl çıktı.</td></tr><tr><td>Words seen in Turkish text are never respelled</td><td>Outlook, Notion ve podcast uygulamaları.</td><td>outlook, notion ve podcast uygulamaları.</td></tr></table>

Flow matching. The generator follows a learned velocity from noise at $t = 0$ to a mel at $t = 1$ . Flow matching (Lipman et al., 2023) trains that velocity by regression. We mix a target mel $x _ { 1 }$ with Gaussian noise $x _ { 0 } ,$ and the network, given the text condition � and a step size of 0 (an input the next paragraph explains), predicts the velocity of the straight line between them:

$$
x _ { t } = \left( 1 - t \right) x _ { 0 } + t x _ { 1 } , \qquad \mathscr { L } _ { \mathrm { F M } } = \bigl \| \nu _ { \theta } ( x _ { t } , t , 0 \mid c ) - ( x _ { 1 } - x _ { 0 } ) \bigr \| ^ { 2 } .\tag{1}
$$

Shortcut self-consistency. The learned velocity averages over many such lines, so its paths curve. Small steps follow them but cost many network passes, and a large step leaves them. Shortcut self-consistency (Frans et al., 2025) gives the network the step size � and trains one step of size � to land where two steps of size $d / 2$ land (Figure 3a). The half steps run on the current weights with no gradients, and one call at the full size regresses onto the mean of their velocities:

$$
\begin{array} { r } { \begin{array} { r l } & { \quad s _ { 1 } = \nu _ { \theta } \left( x _ { t } , t , \frac { d } { 2 } \mid c \right) , \qquad s _ { 2 } = \nu _ { \theta } \left( x _ { t } + \frac { d } { 2 } s _ { 1 } , t + \frac { d } { 2 } , \frac { d } { 2 } \mid c \right) , } \\ & { \quad \mathcal { L } _ { \mathrm { S C } } = { \left\| \nu _ { \theta } ( x _ { t } , t , d \mid c ) - \frac { 1 } { 2 } \left( s _ { 1 } + s _ { 2 } \right) \right\| } ^ { 2 } . } \end{array} } \end{array}\tag{2}
$$

Each utterance trained this way draws one of 7 levels uniformly, from $d = 1$ down to � = 1/64 (Figure 3b). A level takes its half steps from the level below, and the last from plain flow matching at $d = 0 .$ . One model can then sample at 1, $^ { 2 , }$ 4 or 8 steps, and the other powers of 2 up to 64 also lie on the trained grid. On the benchmark we scored $^ { 4 , }$ 8 and 16 steps only.

Training sums these two losses and one term each for the prior and the duration predictor, with unit weights. A quarter of each batch trains $\mathcal { L } _ { \mathrm { S C } }$ and the rest $\mathcal { L } _ { \mathrm { F M } }$ Each is divided by its own token count, so both carry equal total weight.

b The ladder of trained step sizes d  
![](images/4103f1cd257fc6c1b752808701af97062fb2e0993ddbaf7829e76b0a91a6c738.jpg)

![](images/579eaf6f2c2ce6c0a504006dd93b7e199a40dd0cff4ff57d375c8e7b15f008c0.jpg)  
Figure 3 Shortcut self-consistency training lets one model sample at $1 , 2 ,$ 4 or 8 steps; the release uses $^ { 8 , }$ and the benchmark scored 4, 8 and 16 only. (a) Schematic, after Frans et al. (2025). (b) Rows are to scale on t. At the final settings on our own harness, 4 steps gave WER 1.17 against 0.97 at 8 (3 seeds; Section 7).

Generator network. The generator is a difusion transformer, DiT (Peebles and Xie, 2023), of 5 blocks with RoPE and gated SwiGLU feed-forward layers (Shazeer, 2020). The encoder output, repeated per frame, passed through one ConvNeXt block and averaged per token, joins $x _ { t }$ at the input. Time and step size enter through adaptive layer normalization (adaLN), where a small network sets each normalization layer’s scale and shift. In adaLN-single (Chen et al., 2024) all blocks share that network, and each block adds a small learned table.

Sampling. Classifier-free guidance (Ho and Salimans, 2022) runs the generator twice per step, with the text condition and with a learned null vector that replaces the condition on 10% of training utterances. The sampler extrapolates past the conditional velocity $\nu _ { c } ,$ away from the null velocity $\nu _ { \emptyset } ,$ , with weight �, and takes � Euler steps from noise scaled by a temperature of 0.6:

$$
\begin{array} { r } { \hat { \nu } = \nu _ { c } + ( w - 1 ) ( \nu _ { c } - \nu _ { \emptyset } ) , \qquad x ^ { i + 1 } = x ^ { i } + \frac { 1 } { K } \hat { \nu } \big ( x ^ { i } , \frac { i } { K } , \frac { 1 } { K } \mid c \big ) . } \end{array}\tag{3}
$$

The release default, � = 8 and � = 2.0, is 16 generator passes per chunk of text.

Vocoder. The vocoder follows the design of Vocos (Siuzdak, 2024) and has no upsampling layers. Its 8 ConvNeXt blocks and a linear head predict a spectrum per mel frame, 1,025 log-magnitudes and 1,025 phases, and an inverse short-time Fourier transform (iSTFT) rebuilds the waveform from these overlapping spectra.

Training adds GAN losses, in which discriminator networks (Jang et al., 2021; Kong et al., 2020) learn to tell real audio from generated and the vocoder learns to fool them. The loss is 45 $\mathcal { L } _ { \mathrm { m e l } } + \mathcal { L } _ { \mathrm { a d v } } + 2 \mathcal { L } _ { \mathrm { f e a t } }$ , which combines L1 on log-mels at 4 resolutions, a least-squares adversarial term (Mao et al., 2017) and a match on the discriminators’ internal activations.

The inference path. Synthesis normalizes the text and splits it into chunks ofat most 300 characters, about 19 s at the release defaults. Every sentence of Freya-TR-Eval (Pamuk et al., 2026), the benchmark we score on, is one chunk. Predicted durations are divided by a speed of 0.95, rounded and clamped to 1 to 60 frames (640 ms). A doubling rule lengthens repeated sounds first (Section 6).

Every chunk gets a hidden " ee." after it, meant to put the real last word midutterance. We have no measurement isolating a benefit at the final settings, and its WER efect went both ways in earlier seed-0 runs with the first vocoder fine-tune. A chunk that starts with ö or ı also gets a hidden "ee, " before it, for the defect of

![](images/2b84bf55e43c78fecb0f566fff8f7f92cd5511f426d3c3ce4e324c603e280ccd.jpg)  
Figure 4 The model renders more text than the listener hears. Version 1.0.1, seed 0. Some symbol blocks are too narrow for their letter; the “Model reads” row prints the full string. This Freya-TR-Eval sentence (Freya, 2026; Pamuk et al., 2026) uses the hidden prefix, the hidden sufix and the doubling rule. Across the benchmark’s 495 sentences, 4 with the prefix, predicted durations discard 16.8% of rendered frames. Longer chunks lose a smaller share, and this sentence loses more because it has the prefix.

Section 6. All networks run on the whole string, and the audio is cut at the predicted spans with 5 ms fades (Figure 4).

Between chunks the package adds 0.12 s of silence after a sentence end and 0.06 s otherwise. The same text, seed and settings give the same audio on one software stack.

Output chain. Version 1.0.1 ends with fixed signal processing (Table 1), the only default that difers from version 1.0.0. An equalizer (EQ), a gain that varies with frequency, uses 5 second-order filters (Bristow-Johnson, 2021). A de-esser turns down loud energy above 5.5 kHz, the band of hissing sounds such as s. A fixed +1.6 dB gain ofsets the mean loudness drop from those two, 1.596 dB, measured with the BS.1770 standard (ITU-R, 2023) on 24 clips at 48 kHz. A soft limiter guards the peaks, and 96 test clips over 4 output rates did not reach its threshold.

Version 1.0.0’s EQ, the “1.0.0 tone” of our figures, raised the bass and cut the treble (Figure 19a). We chose the 1.0.1 EQ by ear after one outside comment on that tone, and automatic scores moved slightly against it (Section 7).

Where the compute goes. At the defaults a 5.11 s sentence costs 14.49 GMAC (10<sup>9</sup> multiply-accumulates), or 2.84 GMAC per second of output. The generator dominates because it runs 16 times per chunk and the vocoder once. Four steps with guidance cost 1.65 GMAC per second, 8 steps without guidance 1.63 and 1 step without guidance 0.59. These counts do not measure time.

Table 3 The chosen reference put 83.3% of renders in the pitch band; no other candidate passed 50.0%. Four of the 9 candidates, each with 2 renders of each of 60 prompts in 10 categories. CER comes from Whisper large-v3 at beam 5, and adversarial and foreign prompts inflate it for all of them. F0 is the mean of per-render medians. In band means 125–138 Hz.
<table><tr><td>Teacher and reference</td><td>CER,%</td><td>UTMOS</td><td>Identity</td><td>F0,Hz</td><td>In band, %</td></tr><tr><td>VoxCPM2, chosen reference</td><td>2.95</td><td>3.387</td><td>0.697</td><td>129.42</td><td>83.3</td></tr><tr><td>VoxCPM2, earlier model&#x27;s render</td><td>3.03</td><td>3.295</td><td>0.748</td><td>139.33</td><td>45.0</td></tr><tr><td>VoxCPM2, second refinement</td><td>2.82</td><td>3.365</td><td>0.631</td><td>125.13</td><td>50.0</td></tr><tr><td>Alania-2 API, earlier model’s render</td><td>3.12</td><td>2.648</td><td>0.623</td><td>146.71</td><td>23.3</td></tr></table>

## 3 Data from one teacher

Unmodified VoxCPM2, cloning one synthetic clip, generated all our training audio on one 8-GPU machine. Automatic checks kept 1,325,846 of 1,361,734 clips, about 1,790 h, but Whisper checked only 7,500 of the 444,437 short clips, so no rule tested whether the rest read their text correctly.

The teacher. All training audio was generated by unmodified VoxCPM2 (Zhou et al., 2026) (openbmb/VoxCPM2, revision 32279ef, Apache-2.0), cloning the voice ofone 9.88 s reference clip that is itself a synthetic render. On all 1,361,734 clips it ran with classifier-free guidance 2.0 and 10 steps in ultimate mode, which gives the model the clip’s transcript and two encodings of its audio. VoxCPM2’s main checkpoint has about 2.29B parameters, about 2.38B with its AudioVAE (the part that decodes to 48 kHz audio), roughly 300 to 313 times the student. Its authors report over 2 million hours of training speech in 30 languages, with no separate figure for Turkish.

The voice. The training set contains no recordings of people: every clip was synthesized by VoxCPM2 from one synthetic reference clip. The voice is synthetic, chosen by listening from the renders of an earlier in-house model. That model and VoxCPM2 were themselves trained on recordings of real speakers, and the earlier model’s voice-locking started from a reference recording of a consenting voice actor. We have not audited the voice’s similarity to any real speaker. The model has no speaker input and cannot clone voices.

Voice-locking fixes one voice across renders. We clustered renders of that earlier model and, by listening, fixed the voice as 38 of them, with a median F0 (fundamental frequency, heard as pitch) of 130.4 Hz and a 5th to 95th percentile band of 125.1–137.5 Hz. VoxCPM2 then cloned the 24 kHz render closest to the cluster center, and one 48 kHz output became the reference clip for all training data, the chosen reference below.

The bake-of. We compared 9 candidates, each a teacher with a reference clip, on 120 renders each, and pitch set the chosen reference apart (Table 3). A second refinement, which repeats the cloning step on the chosen reference, has a lower CER, and the earlier model’s render a higher identity. Identity is the cosine similarity of a render’s ECAPA speaker embedding (Desplanques et al., 2020) to the centroid of the 38 renders.

Alania-2, our closed production model, was only a candidate and produced no training audio. The table omits other VoxCPM2 settings and refined references, some ofwhich scored higher on UTMOS or lower on CER than the chosen reference, with fewer renders in band. On Freya-TR-Eval (our own harness, 1 seed) the chosen reference gives WER 0.92% and CER 0.200%, and the second refinement 1.15% and 0.289%. That gap, on one seed, counted against the second refinement, so the benchmark had a part in this choice (Section 10).

![](images/a1875cf3bf49e8e90b953f0372e2e42c56ced8d5898edafad5e984bafb60d23d.jpg)

![](images/98c99aafad5dfe16f294d0cb8d627aee9dd6cbf965691ad299282a63d13ed207.jpg)  
Figure 5 The short set lifts kept audio under 3 s from 42.61 h to 226.31 h. Neither validation nor the benchmark shows what the short-clip fine-tune did for short input (Section 4). Hours count unique finished audio, after edge-silence trim and before the ×6 weighting of the short-clip fine-tune.

The long set. We generated 917,297 lines, sentences up to about 25 s, and kept 898,046 clips, 1,581.72 h (Figure 5a), with a median of 5.22 s. The text comes from FineWeb-2 (Penedo et al., 2025), Wikipedia, service-domain sentences written by Qwen3 (Yang et al., 2025) and our own number templates.

The short set. We built it because the main run’s word errors concentrated in sentences of3 to 4 words and little ofthe long set was under 3 s (Figure 5b). That length efect is confounded with register, the style of the sentence (Section 7). We kept 427,800 of 444,437 clips, 207.29 h, with a median of 1.64 s. Fragments of 2 to 6 words cut from the long-set text pools make up 159.81 h. On the shipped path only the short-clip fine-tune used this set, weighted ×6, and the vocoder never saw it.

A bug reached this set. Turkish pairs i with İ and ı with I, so when Python’s str.upper turned a leading i into ASCII I, the normalizer lowered it to ı. As a result 4,580 of the 5,589 kept ı-initial lines are misspelled (for example ıyi, ıki). They stayed in the short set, and we have no measurement isolating their efect on the model.

Keeping the benchmark out. We excluded benchmark sentences from the training text, which dropped 11 long-set lines and 5,562 short-set lines. A later recheck ofthe final text with the same guard found no exact, contained or near match (rapidfuzz string similarity of 85 or more), though it did not re-run the short set’s rule for near matches to part of a sentence.

Quality control. A quality-control (QC) worker rejected a clip at the first rule it failed (Table 4, Figure 6).

Table 4 CER rejected the most long-set clips, and F0 and identity the most short-set clips. QC applied the rules in this order.
<table><tr><td>Rule</td><td>Rejects when</td><td>Relaxed in the short set</td><td>Long set</td><td>Short set</td></tr><tr><td>Length limits</td><td>Clip too long or too short</td><td></td><td>8</td><td>13</td></tr><tr><td>CER</td><td>Above 10%</td><td>Up to 4 words: passes if the clip&#x27;s text appears in the transcript. Interjections: off</td><td>14,227</td><td>289</td></tr><tr><td>Speaking rate</td><td>Outside 8–24 characters per second</td><td>Up to 2 words: no minimum</td><td>5</td><td>1,051</td></tr><tr><td>ECAPA identity</td><td>Below 0.45</td><td>Under 1.2 s: off</td><td>33</td><td>7,092</td></tr><tr><td>Median F0</td><td>Outside 110–155 Hz</td><td>Interjections: off</td><td>146</td><td>7,271</td></tr><tr><td>UTMOS</td><td>Below 2.8</td><td>Up to 4 words: off</td><td>4,832</td><td>921</td></tr><tr><td>All rejections</td><td></td><td></td><td>19,251</td><td>16,637</td></tr></table>

![](images/62ca0a7a9d19a0c952892348a82b1c48635f0711d9c57b50716a74370b1e643e.jpg)  
Figure 6 QC kept 97.4% of 1,361,734 generated clips, and the kept short set sits further from the voice than the kept long set. Mean identity is 0.582 against 0.694, and 57.0% of short clips fall in 125–138 Hz against 82.4% of long-set clips. Short-set clips past a threshold passed under relaxed rules. Axes clip outliers.

What Whisper checked. The Whisper speech recognizer (Radford et al., 2023), with the large-v3 checkpoint in faster-whisper (Klein and SYSTRAN, 2023), checked all 917,297 long-set clips. We turned it of for the short set after 15 shards, 7,500 of its 444,437 clips, because it pads each clip to 30 s, which made QC about 10 times slower than generation. The other 436,937 clips have empty transcripts and a placeholder CER of 0. Speaking rate, ECAPA identity, F0 band and UTMOS rules still applied to them, relaxed for very short ones.

In the long set the 10% CER gate reads the whole line, and 82% of ö-initial lines with a wrong first word passed it (Section 6). Clips Whisper could not transcribe never reached long-set training, and the same checkpoint scores the benchmark, which may flatter WER (Section 10).

Generation. We used one AWS p5en.48xlarge with 8 NVIDIA H200 GPUs, rented for 23 h 04 min (the block of Figure 7a). Soon after launch, segfaults in Numba’s shared cache crashed the post-processing pools of 6 of 8 generators (one VoxCPM2 process per GPU). A private cache per worker and a self-rebuilding pool fixed this.

A Whisper-based QC worker on the same GPU as the generator cost 27% of throughput (Figure 7b). From 12:47:30 to 13:05:30, with the QC workers on GPUs 5–7 down until 13:04:50, those GPUs finished 79.8 seconds of audio per second (audio-s/s) per

![](images/4077dc59eec34682c6442d4940131c0bb89beb5fe0fd087a64caabfd7b2b8a68.jpg)  
Figure 7 Generation itself took 3.63 h (long set) and 1.15 h (short set) from launch to last shard; 23 h 04 min is the whole block. Cobalt marks the shipped path, teacher runs included; the final vocoder fine-tune ran after the block.

H200, and GPUs 0–4, each with a QC worker, finished 58.2. This is one unplanned comparison, in an 18-minute window on diferent GPUs of one machine. Sustained with QC on every GPU, the rate was 58.16 ± 0.18.

Lesson. Count QC in the generation plan. Decide before the rental which clips must be transcribed, and leave GPU time for them.

## 4 Training the acoustic model

The bf16 autocast weight cache kept gradients from the generator for the first 6,000 of the main run’s 420,000 steps. The short-clip fine-tune in the shipped weights is not distinguishable from the main run on the benchmark, and the shipped lineage took 114.5 GPU-hours.

The shipped acoustic weights are an exponential moving average (decay up to 0.9995), taken after 420,000 steps of the main run on the long set and 51,880 steps of the short-clip fine-tune on both sets. Both use AdamW (betas 0.9 and 0.98, weight decay 0.01), gradient clipping at 1.0 and bf16 autocast (mixed precision).

The main run used a base learning rate of $3 \times 1 0 ^ { - 4 }$ and 3 H200 GPUs, then 2 from step 55,000, with 96k mel frames per GPU from step 2,000. In parallel we trained a sibling run on long-set clips with a QC UTMOS of at least 3.1.

A silent bug. We computed shortcut targets under no\_grad inside the same autocast region as the training pass. Autocast caches the weights it casts, so the training pass received detached copies and the generator got no gradients. Setting cache\_enabled=False fixed it.

The signature is a flow-matching loss flat near 2.03 in training and exactly constant in validation on the moving-average weights, with a shortcut loss of 0 (Figure 8). The encoder, prior and duration predictor trained normally (prior loss 0.529 to 0.039 by step 6,000), so total loss still fell.

![](images/321ec759b08b393b59e37c6cd7a8bb2395cec501b815f5810c908d1f52fdad8e.jpg)  
Figure 8 The bf16 autocast weight cache gave the generator no gradients for the first 6,000 steps of the main run and 15,000 of the sibling run. Validation loss on the moving-average weights stayed at exactly 2.0359 in the main run and 2.0342 in the sibling run, and at step 13,000 of the sibling run the generator’s output weights still had zero norm.

Lesson. After the first step, assert that every trainable module has a gradient. Our first launch crashed on an unused-parameter error, which was this bug’s symptom. We silenced it with find\_unused\_parameters=True before we knew the cause.

The short-clip fine-tune. It starts from step 420,000 of the main run with a fresh AdamW and a learning rate of $1 0 ^ { - 4 }$ , and spends its last 23,880 steps at a floor of $5 \times 1 0 ^ { - 6 }$ . With the short set repeated six times, the two sets total 2,802.62 h, of which short clips are 44.07%. Validation runs on the first 256 clips of the validation split, all from the long set, and its flow-matching loss ended at 0.3969, where it stood at step 2,000. A flat validation loss was also the bug’s signature, so we compared checkpoints. The generator’s moving-average weights moved by 6.8% in relative L2 norm between the main run’s last step and the shipped file.

On the benchmark the short-clip fine-tune is not distinguishable from the main run. Its WER minus the main run’s is −0.08 [−0.32, +0.15] at 4 sampling steps and −0.05 [−0.25, +0.15] at 8 (95% intervals from a paired bootstrap over sentences, 3 seeds, our own harness, no output chain, same vocoder). Validation holds no short clips and these whole-benchmark intervals include $0 ,$ so we cannot say what the fine-tune did for short input.

The sibling run’s filter has no fair test either. In the second evaluation round during training, with the same vocoder run, it had WER 6.28% and UTMOS 1.78 against 5.21% and 2.26 for the main run (seed 0, our own harness), at diferent learning rates and step counts.

Compute. MAS took 61.2% of main-thread profiler samples in an acoustic step. Sending only its log-likelihoods to the CPU and giving it more threads raised median throughput on 3 H200s from 15,954 to 20,628 audio-s/s (+29%).

The shipped lineage took 114.5 GPU-hours, 36.4 for the acoustic model on H200s and 78.0 for the vocoder (56.0 on H200s, 22.0 on A100s and an L40S). All training runs took 159.0, of which 130.6 ran on the machine of Section 3, where setup, generation and QC took another 33.7. Evaluation is not counted.

## 5 The vocoder kept WER high

On our own harness, at adjacent pre-release acoustic checkpoints with the same seed, fine-tuning the vocoder on the acoustic model’s own mels cut WER from a mean of 3.89% to 1.75%, while a vocoder frozen over the same stretch stayed at 3.63–4.14%.

Our 2.16M vocoder got worse as it trained on real mels. Its copy-synthesis UTMOS on 40 corpus clips, meaning clips from the training set, fell from 2.54 at step 20,000 to 2.17 at step 148,000, against 3.29 for the teacher (Figure 9a). Mel loss did not track the fall.

Its output also grew a ripple at 93.75 Hz, the vocoder’s frame rate. The envelope peak at that rate rose from 4.2 dB at step 20,000 to 5.3 dB at step 148,000, against 2.79 dB for the teacher (Figure 9b). The larger 3.93M vocoder of the shipped lineage reached UTMOS 2.69 with a peak of 2.82 dB at step 85,000.

Fine-tuning on generated mels. This real-mel vocoder still left WER above 3.6%. It had trained only on real mels, and at inference it receives the generator’s mels, so we fine-tuned a copy of its step 157,000 on those, the first vocoder fine-tune.

The evaluation loop during training gave 7 pairs. Each scores the real-mel vocoder on one acoustic checkpoint and the first vocoder fine-tune on an adjacent one, at seed 0 and 4 sampling steps. Over these pairs the fine-tune cut WER from a mean of 3.89% to 1.75% (Figure 10a). The drop per pair runs from 1.66 to 2.63 points, and every paired sentence-bootstrap 95% interval excludes 0. CER fell from 0.984% to 0.385% and UTMOS rose from 2.45 to 2.93 (Figure 10b).

A pair’s two checkpoints sit 1,000 to 4,000 steps apart. Further acoustic training does not explain the drop, because the real-mel vocoder, frozen at vocoder step 218,000, stayed at WER 3.63–4.14% and UTMOS 2.44–2.47 while the main run moved from acoustic step 263,000 to 330,000, the last 5 of the 7 pairs (Figure 10c). It was still training for the first two pairs.

For this fine-tune the acoustic model samples one fixed mel per clip, at 4 steps and classifier-free guidance 2.0 on the clip’s own MAS durations, so frame counts match the teacher’s audio, which stays the target. Each training crop takes the generated mel with probability 0.8 and the real mel otherwise, which is the “80%” of Figure 10c. The shipped vocoder’s stages use AdamW (betas 0.8 and 0.99, weight decay 0.01, clipping at 10.0) on 1.003 s crops, 32 per GPU, with no moving average.

These are pre-release checkpoints at temperature 1.0 with no output chain, on our own harness. We never scored a real-mel vocoder at the final acoustic checkpoint. WER here is Whisper’s, and no human listening test of this model exists. With more training the main run and the first vocoder fine-tune reached 1.38% on 3 seeds (acoustic step 420,000, fine-tune step 301,490, 4 sampling steps), a later result from outside the 7 pairs.

![](images/40443852c5d0b180235baae26b286d5428eaf80b0dd79f4d510e53f27d89cd74.jpg)

![](images/9b386cccd2b8599798befba19a31a002f5ec324316b6b6a0928e3b7ad91b482f.jpg)

![](images/0ed131e29d0446d2b49da6479c566378c32630b62e98666c8c7a21428b33040a.jpg)

![](images/1ed89185840b6bdf4314e8c00ca135683de968f32bd42bda2dc18d00500ace68.jpg)

![](images/7ca5d34a98f60bb9064c3fdc3621d4b3e33a8fd9567d9ac91d9bbf8d87c9b090.jpg)  
Figure 9 A 2.16M vocoder got worse with training on real mels; after fine-tuning, the 3.93M vocoder alone costs about 0.24 UTMOS. Pre-release checkpoints. (a) Checkpoints difer by about ±0.1 UTMOS. The 2.16M vocoder’s mel loss read 0.3927, 0.4225, 0.3544 and 0.3809 at steps 20k, 60k, 100k and 148k. (b) The peak of the amplitude envelope’s spectrum at the 93.75 Hz frame rate, over its median from 60 to 130 Hz. (c) Benchmark renders on our own harness. “Real mels” and “Teacher through vocoder” are the same measurement on diferent clips. Aligned generated mels use each clip’s MAS durations.  
Fine-tuned Fine-tuned variant Frozen, real mels

![](images/6c29a75750e8a16eaa69bd9f6d20922eb236212978c77b2048f239531949aaa1.jpg)

Figure 10 At adjacent acoustic checkpoints, seed 0 and 4 sampling steps, fine-tuning the vocoder on generated mels cut WER from a mean of 3.89% to 1.75%. (a, b) Every paired 95% interval of the WER drop excludes 0. UTMOS after fine-tuning rises with fine-tune steps, from 2.595 at 17,000 to 3.140 at 135,000. (c) The 2 s crop variant, a real-mel vocoder trained on 2 s crops and frozen at step 112,000, held WER at 3.88–4.44% and UTMOS at 2.40–2.42. The 100% variant, a fine-tune with generated mels on every crop, ran at unequal steps, so we draw no conclusion from it. Benchmark, our own harness, no output chain, pre-release checkpoints.

Lesson. Score the whole chain during training, and fine-tune the vocoder on generated mels early. Our first pair came after 17,000 fine-tune steps, about 34 minutes on one H200, and moved WER from 3.78% to 2.12%.

Precedent. Tacotron 2 and HiFi-GAN (Kong et al., 2020; Shen et al., 2018) also report gains from vocoders trained on predicted features. Those are human ratings on teacher-forced features, where the acoustic model sees the real previous frames, so the efect sizes do not compare with ours (Section 9).

What remains. After fine-tuning, the vocoder alone costs about 0.24 UTMOS (Figure 9c). Copy synthesis through the first vocoder fine-tune at step 301,490, the parent of the shipped vocoder, scores 3.069 against 3.306 for the teacher on 200 corpus clips, and aligned generated mels at temperature 0.6 score 3.045. On the teacher’s 495 benchmark renders it scores 3.173 against 3.419, with WER 1.251% against 0.919% (1 seed, our own harness, unseen text).

The last 76,860 of the shipped vocoder’s updates, the final vocoder fine-tune on mels from the short-clip fine-tune, gave no demonstrated gain. On 3 seeds through the release API on our own harness, before the fixes of Section 6, WER moved from 1.115% to 1.089% (131 to 128 word errors) and UTMOS from 3.268 to 3.256. The final vocoder fine-tune saw mels at 4 steps, temperature 0.8 and MAS durations, while the release samples at 8 steps, temperature 0.6 and predicted durations. The vocoder never saw short-set audio.

## 6 The teacher’s ö defect and two reading fixes

By Whisper, the teacher gets first words right on only 26% of ö-initial kept long-set lines, and the student learned this. A hidden spoken prefix hides the defect without removing it, with 12 of 12 first words right on 4 benchmark sentences. The paired interval for a duration rule on doubled sounds includes 0.

The defect in the teacher. Only 26% ofthe 12,554 kept long-set lines whose first word begins with the vowel ö, capital or not, have that word right in their QC Whisper transcripts (95% CI 25.4–26.9; Figure 11a). Lines with other capital first letters reach 94%, and capital Ö alone, the like-for-like figure, scores 23.66%. No listening test of teacher clips exists.

The same ö words score about 98% later in the clip (Figure 11b), which ties the defect to the clip start. The clip-start rate depends on the word, with 76.8% for özellikle against 0.8% for öyle. Wrong transcripts often open with a phantom syllable, as in latte for önce.

We did not establish the cause inside the teacher. With 4 renders per configuration on ö prompts, Whisper read 4 of 4 first words right in reference-only mode (an encoding of the reference clip’s audio, no transcript), and in ultimate mode 0 or 1 of 4 with most references and 4 of 4 with one. Samples this small are only suggestive.

How we know it is audio. All of this rests on Whisper, which could be misreading good audio. Two checks point to the audio instead. At equal character count, longset ö lines that Whisper reads with a wrong first word run a median 0.315 s longer than those with a right one, which suggests extra sound, and right ö lines match lines starting with ü, the highest-scoring letter, within noise (Figure 12a). The shortset rows use no recognizer, and there all ö lines run 0.169 s longer than ü lines while ı lines do not stand out.

Whisper also does not invent the syllable. On the 4 ö-initial benchmark sentences, a small control, it reads 11 or 12 of 12 first words (3 takes each) from most other systems on the public harness, including EMA Lightning, whose card reports the same teacher model with another voice. On our own harness it reads 0 of 4 from the teacher with the chosen reference, in 1 take (Figure 12c).

a First word right, by first letter of the line  
![](images/475b1c9d1bbe672839df19fb532e4f685019a3deaa6bdfb8fb1778ece7bb6e2a.jpg)

![](images/6e47cba1cc214ed8e911c652867e404739d6758ac69a41704531bd3385f1dc5e.jpg)  
Figure 11 By Whisper, only 26% of ö-initial kept long-set lines have a right first word, against 94% for other capital first letters. (a) ı errors have other causes: the word ı from Roman numerals, Irak and its inflections, heard as drak, and spelling. (b) Later positions are pooled and use aligned transcripts.  
a Extra audio, from duration alone Median diference at equal character count, 95% C

![](images/8b3232bd80dc90e661db89ea334fced6d732f94ffae49312cd95ae5de75b3f7c.jpg)  
c Control: the 4 ö-initial sentences  
b The 10% CER gate passed 82% Wrong-first ö lines per 0.5-point bin of whole-line CER

![](images/16afb370b4c63d8c998e6b627b7c15dfaa3c149ce641ccda8c299ccc3db91c3e.jpg)  
Figure 12 The ö defect adds audio measurable without speech recognition (ASR), 82% of ö lines with a wrong first word passed the 10% CER gate, and most other systems’ first words transcribe correctly on the same sentences. (b) Rejected long-set lines are counted, not binned. (c) Teacher with the chosen reference. Small n: 4 sentences.

How the student inherited it. Our 10% whole-line CER gate missed most ofthe defect. Of the 11,270 generated long-set lines with a wrong ö first word, the gate rejected 1,879, other rules rejected 120, and 9,271 (82%) passed (Figure 12b). The lines that passed stayed in the training data, and the student learned the defect. Without the prefix it gets 5 of 12 first words right on the 4 benchmark sentences (our own harness, 3 seeds, before the final vocoder fine-tune). Informally, one of us heard student renders of three ö-initial sentences as broken.

Lesson. Check clip starts separately from whole-line CER. A wrong first word costs a median of only 5 characters on a median 91-character line, so 82% of ö lines with a wrong first word passed our 10% gate.

The hidden prefix. The defect sits at the clip start, so we make the model say something else first and cut that audio away. A chunk that starts with ö or ı gets a hidden "ee, " in front of it, cut at its predicted span (Section 2). With the prefix, Whisper

## Benchmark sentence 0397 Öğle arasında dışarı çıkıp bir kahve molası verelim mi?

Illustrations: mel spectrograms of CPU renders from the released 1.0.1 weights. Words above each cell are Whisper on benchmark audio, not on these renders.

No prefix 0 s is where the text starts; audio before it is kept Seed 0 Benchmark no-prefix run, Whisper: Löri wrong  
![](images/819bbe05ac389ba6f32dfca094137603b0753f38b101f8fe9fbf8c4cdd34fa7e.jpg)  
With the hidden prefix ee, dimmed, and cut away at 0 s  
Seed 1 Benchmark no-prefix run, Whisper: Löri wrong  
Seed 0 Benchmark prefix run, Whisper: Öğle right

![](images/d64155e1d56b0ed52d0c0d3b04fbd3470c89bb7018088154c4a7b9c6d4d85d76.jpg)

![](images/9f016fb4fc6a4d262c839fbd9273304eb3af1f7a3e17bb725480fa9144d2ccee.jpg)

![](images/a5c2bc7980a894be6ec55d948de39f95984164b1841032d3a4257bf811518f47.jpg)

Seed 1 Benchmark prefix run, Whisper: Öğle right  
![](images/56dcde1945a6c569b8806cc08d1a42b81dac2ea3f27ab981e9e252f59e12299a.jpg)

![](images/cbc3b1e293df631213ee5dd50ffa5e6e200e9384c31e5db1e29977ede2692acd.jpg)  
Figure 13 On benchmark sentence 0397, Whisper wrote Löri in the no-prefix run and Öğle in the prefix run, at seeds 0 and 1. Read the words above the cells; the spectrograms only show where the prefix sits and where we cut it. The transcripts come from two own-harness runs before 1.0.0, and the no-prefix run predates the final vocoder fine-tune, so the figure does not isolate the prefix. Sentence: Freya-TR-Eval (Freya, 2026; Pamuk et al., 2026).

reads 12 of 12 first words on the 4 sentences on both harnesses. Figure 13 shows one of them. With the final vocoder fine-tune, per-clip no-prefix data exists only for seed 0, where Whisper again writes Löri for Öğle.

In own-harness runs before 1.0.0, with the final vocoder fine-tune, the prefix lowers 3-seed WER from 1.089% to 1.013% and CER from about 0.25% to 0.204%. On seed 0 the gain comes from 2 of the 4 sentences, where we had also seen the symptom. The missing per-clip data leaves this step with no 3-seed paired interval.

Applied to every chunk in a probe of 84 new sentences, the prefix broke other words, such as Jöle, heard as göle or Göle on 2 of 3 seeds. So we restrict it to ö and ı. The ı part rests on that probe alone, because the benchmark has no ı-initial sentence. In the probe, first words starting with capital I went from 6 to 8 of 9 right, and Ilgaz was heard as Bulgaz, Dılgaz or Rılgaz without the prefix. The teacher’s ı errors have other causes (Figure 11a).

Doubled sounds. The doubling rule multiplies by 1.3 the predicted duration of the second of two identical symbols, or of the word gap between them (the predicted span of the space symbol). Such pairs occur in 167 of the benchmark’s 495 sentences (112 pairs within words, 111 across word gaps).

The rule targets doubled sounds heard as one, as in o olumsuz read as olumsuz, a pattern we found in benchmark errors. Before the rule, 21 sentences failed on all 3 seeds of our own harness. EMA Lightning also fails 13 of them on all 3 of its seeds and gets 6 right on all 3, so Whisper can read those 6 from another model’s audio. Of those 6, 3 show the pattern (Figure 14a).

A result on Turkish consonants motivates the rule. Lahiri and Hankamer (1988) find that doubled Turkish [t] and [k] between vowels, from 3 male speakers, are held almost three times as long as single ones, about 120 ms longer. Vowels, other consonants and word-gap pairs, which the rule also lengthens, fall outside their data. On the benchmark we tried stretch factors of 1.3, 1.5 and, on seed 0 only, 2.0, and we shipped 1.3 (Figure 14b).

![](images/17e75a8deedbfbdefa31523eccbfac230114cf95b92cb70e4a44ca479614923d.jpg)  
Figure 14 The doubling rule moves 3-seed WER from 1.013% to 0.970%, with a paired interval that includes 0; it fixes o olumsuz on 3 of 3 seeds on both harnesses, and the doubled rr in berraktı stays wrong. Own-harness runs use 1.0.0 settings and public takes 1.0.1. (b) The interval is measured from the mean at stretch factor 1.0. (c) We computed no intervals, and nearly all the gain comes from seed 2. WER is near 6% because of digits and foreign names the scorer does not match. Sentence: Freya-TR-Eval (Freya, 2026; Pamuk et al., 2026).

Over 3 seeds on our own harness, the rule moves WER from 1.013% to 0.970% and CER from 0.204% to 0.193%. Both paired intervals include 0 (WER −0.043 [−0.135, +0.035], CER −0.011 [−0.035, +0.010] points). Word errors fall from 53 to 46 in sentences with such a pair and go from 66 to 68 in the rest. The rule came from these sentences, so these numbers are optimistic by selection.

Of the benchmark the direction agrees, and the change is not separable from seed noise. On 500 other sentences, 464 of them not in the training text by exact match (near-duplicates unchecked), 3-seed WER moves from 6.12% to 6.02% and CER from 1.53% to 1.48% (Figure 14c). Nearly all the WER gain is one seed. For these runs and the stretch factors 1.5 and 2.0, we kept only the lines the scorer printed.

What did not help. On our own harness, stretching every word gap by a factor of 1.25 gave 3-seed WER 1.055% against 1.013% without it.

Filtered training gave no demonstrated gain on the same harness. We fine-tuned the shipped acoustic model twice on filtered data and scored both runs with the shipped vocoder, not fine-tuned again. One run dropped lines with a bad start, gave good ö and ı starts more weight, and scored WER 1.174% and CER 0.256% on seed 0 without the prefix, against 1.047% and 0.241%. The other also dropped every longset line with any character error and scored 1.106% and 0.214% over 3 seeds with the prefix, against 1.013% and 0.204%. We made no fix at the source, and regenerating the afected clips remains open (Section 11).

## 7 Evaluation

On the 495 sentences of Freya-TR-Eval, on which we tuned settings, Mini 1.0.1 has WER 1.09% and CER 0.220% on the public harness. A paired bootstrap cannot distinguish these from EMA Lightning 1.0.1 or, on our own harness, from the teacher (1 seed), and no human listening test of Mini exists. EMA Lightning is ahead on full-band UTMOS and, on our own harness, Mini in the 8 kHz band the benchmark’s card asks for, while 16 sentences fail repeatedly for Mini and not for EMA Lightning.

The benchmark. Freya-TR-Eval (Pamuk et al., 2026) holds 495 Turkish sentences of 3 to 13 words in two registers: 406 everyday sentences written by an LLM and 89 short-native sentences from the public corpora Common Voice and CoVoST2. No reference contains a digit, so these scores do not test number readings. Three takes add up to 11,751 scored words and 80,922 characters. We report the benchmark card’s secondary metric, band-matched WER and CER. We did not compute its primary metric, semantic WER, in which an LLM judge counts only errors that change the meaning.

Two harnesses. The public harness (PatientDesk AI, 2026b) carries the headline numbers and all 15 systems, and a reader can re-run its open-weights rows (Appendix B). Development runs, the teacher and 8 kHz UTMOS are on our own harness, whose code is private. Both pass raw text to each system and score alike: 8 kHz band matching, the faster-whisper large-v3 recognizer (Klein and SYSTRAN, 2023) at beam 5, text normalization for scoring with identical output on all 3,772 strings we compared, corpus-level WER and CER per take from jiwer (Vaessen, 2018), and UTMOS22-strong (Saeki et al., 2022) from SpeechMOS v1.2.0 at 16 kHz.

Results. Mini 1.0.1’s WER of 1.09% on the public harness (Table 5) is second of the 15 systems we ran, behind EMA Lightning by 0.05 points, a gap the paired intervals below cannot distinguish from zero. Its CER of 0.220% is the lowest of the 11 open-weights systems (0.225% for EMA Lightning, 0.229% for Trendyol-TTS) and the second lowest of all 15, behind our own closed Alania-2 voice Öncel (0.217%). These gaps are 2 to 7 character errors in 80,922, and none is distinguishable from zero. EMA Lightning is lower on our own harness (0.198% against 0.205%). Mini ranks 11th of 15 at 3.18 on full-band UTMOS, below most large systems (Trendyol-TTS 3.82, Alania-2 F1 of 7 Oct 3.67). Ranks count Mini 1.0.1 only and Alania-2 at its top voice in each column. We tuned Mini’s settings on these sentences, so finalsettings numbers are optimistic by selection, and the scoring Whisper checkpoint also filtered the training data (Section 10).

Among the 15 systems Mini is the smallest of the 13 with a known size (Figure 15);   
some of those counts exclude the vocoder, and Gemini’s sizes are not published.   
Smaller complete TTS systems exist outside this table.

Paired comparison. Paired bootstrap intervals are the central 95% of corpus-level diferences over at least 10,000 resamples ofthe 495 sentences, with each sentence’s two scores kept together. We call Mini “level with” another system when this interval includes 0 on both WER and CER, on the public harness, on our own harness and pooled. The test then cannot tell the two apart; they need not be equal. With 495 sentences and 3 takes the half-width is about 0.27 WER points and 0.07 CER points, so smaller diferences cannot be resolved. The term never extends to UTMOS.

Table 5 On the public harness no WER or CER diference between Mini 1.0.1 and the rows from EMA Lightning down to Gemini 3.8 Flash-Lite is distinguishable. No reference contains a digit. Rows are sorted by WER within each block, with Mini as two releases of the same weights, which difer only in the output chain (Figure 19), and Alania-2 as three voices with their run dates. Alania-1, Alania-2 and Antalia-1 are also ours.
<table><tr><td>System</td><td>Weights</td><td>WER, %</td><td>CER, %</td><td>Clean, %</td><td>UTMOS</td><td>Parameters</td></tr><tr><td colspan="7">Public harness, 495 sentences, 3 takes</td></tr><tr><td>EMA Lightning 1.0.1</td><td>open</td><td>1.04</td><td>0.225</td><td>93.7</td><td>3.30</td><td>8.6M</td></tr><tr><td>Antalia-2 Mini 1.0.0, previous release</td><td>open</td><td>1.05</td><td>0.200</td><td>93.1</td><td>3.25</td><td>7.62M</td></tr><tr><td>Antalia-2 Mini 1.0.1</td><td>open</td><td>1.09</td><td>0.220</td><td>92.9</td><td>3.18</td><td>7.62M</td></tr><tr><td>Alania-2 API, voice Öncel (12 Oct)</td><td>closed</td><td>1.15</td><td>0.217</td><td>93.0</td><td>3.52</td><td>488Ma</td></tr><tr><td>Trendyol-TTS</td><td>open</td><td>1.16</td><td>0.229</td><td>93.1</td><td>3.82</td><td>2.38Bd</td></tr><tr><td>Gemini 3.8 Flash TTS</td><td>closed</td><td>1.23</td><td>0.289</td><td>92.9</td><td>3.53</td><td></td></tr><tr><td>Alania-2 API, voice Lina (12 Oct)</td><td>closed</td><td>1.25</td><td>0.258</td><td>93.1</td><td>3.66</td><td>488Ma</td></tr><tr><td>Gemini 3.8 Flash-Lite TTS</td><td>closed</td><td>1.34</td><td>0.273</td><td>91.9</td><td>3.63</td><td></td></tr><tr><td>Anka TTS</td><td>open</td><td>1.42</td><td>0.302</td><td>91.6</td><td>3.52</td><td>336Ma</td></tr><tr><td>Alania-2 API, voice F1 (7 Oct)</td><td>closed</td><td>1.44</td><td>0.281</td><td>91.7</td><td>3.67</td><td>488Ma</td></tr><tr><td>Chatterbox Multilingual</td><td>open</td><td>1.91</td><td>0.429</td><td>88.8</td><td>3.51</td><td>801Mc</td></tr><tr><td>Piper, voice dfki</td><td>open</td><td>3.41</td><td>0.733</td><td>79.4</td><td>3.71</td><td>16M</td></tr><tr><td>XTTS-v2</td><td>open</td><td>3.67</td><td>1.498</td><td>77.1</td><td>2.97</td><td>470M</td></tr><tr><td>Alania-1 API</td><td>closed</td><td>5.99</td><td>1.809</td><td>67.3</td><td>3.19</td><td>3.3Bb</td></tr><tr><td>Antalia-1</td><td>open</td><td>6.28</td><td>2.489</td><td>66.7</td><td>2.72</td><td>305Ma</td></tr><tr><td>MMS-TTS</td><td>open</td><td>6.38</td><td>1.480</td><td>65.7</td><td>3.79</td><td>36M</td></tr><tr><td>Coqui GlowTTS, lowercased input</td><td>open</td><td>10.93</td><td>2.958</td><td>46.1</td><td>3.09</td><td>28Ma</td></tr><tr><td>FreyaTTS-small</td><td>open</td><td>12.02</td><td>4.946</td><td>47.1</td><td>2.81</td><td>183Ma</td></tr><tr><td colspan="7">Own harness, 495 sentences, 3 seeds (teacher 1 seed)</td></tr><tr><td>VoxCPM2, the teacher</td><td>open</td><td>0.92</td><td>0.200</td><td>94.5</td><td>3.42</td><td>2.38Bd</td></tr><tr><td>Antalia-2 Mini 1.0.0, previous release</td><td>open</td><td>0.97</td><td>0.193</td><td>93.7</td><td>3.25</td><td>7.62M</td></tr><tr><td>EMA Lightning 1.0.1</td><td>open</td><td>1.00</td><td>0.198</td><td>93.8</td><td>3.30</td><td>8.6M</td></tr><tr><td>Antalia-2 Mini 1.0.1</td><td>open</td><td>1.02</td><td>0.205</td><td>93.3</td><td>3.18</td><td>7.62M</td></tr><tr><td colspan="7">Quoted from EMA Lightning&#x27;s card, not run by us</td></tr><tr><td>ElevenLabs v4</td><td>closed</td><td>1.43</td><td>0.27</td><td>一</td><td>3.30</td><td>一</td></tr><tr><td>ElevenLabs v4 Turbo</td><td>closed</td><td>1.54</td><td>0.29</td><td>一</td><td>一</td><td>一</td></tr></table>

UTMOS is full band in all rows we ran. Open means the weights can be downloaded, whatever the license. A dash means not published. No clean-rate diference between Mini 1.0.1 and the rows above Chatterbox is distinguishable. Mini 1.0.1’s own-harness row used the final weights with the 1.0.1 EQ passed as options. Appendix B says how these rows reproduce.  
Parameters. <sup>a</sup> The count leaves out the part that makes audio. <sup>b</sup> The count does not say whether it includes the vocoder, so Figure 15 half-fills its marker. <sup>c</sup> All parts from the file headers; the published 500M matches the T3 transformer alone. <sup>d</sup> About 2.29B in the main checkpoint, about 2.38B with the AudioVAE. Alania-2’s 488M and Alania-1’s 3.3B are our internal counts. EMA Lightning’s card splits its 8.6M into a 5.6M acoustic model and a 3.0M HiFi-GAN-style decoder. Sources disagree on XTTS-v2 and Anka TTS (footnote in Section 9).  
Quoted rows. We did not run ElevenLabs and have no per-sentence data, so no interval exists and we make no claim of a diference. For scale, the gap between v4 and our public-harness values, 0.34 WER points and 0.05 CER points, is about the half-width of a paired interval at this test size. The card states v4’s UTMOS only in its text.  
Sources. Trendyol-TTS (Trendyol, 2026), Anka TTS (KrmKayabasi, 2026), Chatterbox (Resemble AI, 2025), Piper (Hansen and The Rhasspy Project, 2023), XTTS-v2 (Casanova et al., 2024), Antalia-1 (Saygili et al., 2026), MMS-TTS (Pratap et al., 2024).

Mini 1.0.1 is level with EMA Lightning 1.0.1 (Table 6). Pooled, Mini has 248 word errors to EMA Lightning’s 240 of 23,502, and 344 character errors to 342 of 161,844. The pooled result is only a sensitivity check, because own-harness seeds are not paired across harnesses.

No public-harness WER or CER diference from Alania-2 Öncel or Lina, Trendyol-TTS, or Gemini 3.8 Flash or Flash-Lite is distinguishable (Figure 16). Against Alania-2

![](images/02010cf66327e0a2afb7489868e13e327582a73561e418e0c735608846178362.jpg)  
Figure 15 Mini at 7.62M and Trendyol-TTS at 2.38B are not distinguishable on WER (Figure 16); Mini is lower on full-band UTMOS. UTMOS is an automatic, English-trained proxy.

Table 6 Every paired interval against EMA Lightning 1.0.1 and the teacher includes 0 on WER and CER. Diferences in points, with 95% intervals. The teacher has one seed, so its intervals resample sentences only.
<table><tr><td>Mini 1.0.1 minus</td><td>WER</td><td>CER</td></tr><tr><td>EMA Lightning 1.0.1, public harness</td><td>+0.05 [−0.22, +0.32]</td><td>-0.005 [-0.075, +0.065]</td></tr><tr><td>EMA Lightning 1.0.1, own harness</td><td>+0.02 [-0.25, +0.27]</td><td>+0.007 [-0.052, +0.071]</td></tr><tr><td>EMA Lightning 1.0.1, pooled over 6 takes each</td><td>+0.03 [-0.21, +0.28]</td><td>+0.001 [-0.057, +0.062]</td></tr><tr><td>VoxCPM2 teacher, own harness, 1 seed</td><td>+0.10 [-0.25, +0.45]</td><td>+0.005 [-0.103, +0.105]</td></tr></table>

F1 (7 Oct) and Anka TTS the intervals exclude 0 on one ofWER and CER only, and on neither once takes are resampled as well as sentences, so we call them borderline. Mini has distinguishably fewer word and character errors than the eight systems from Chatterbox down in Table 5. In Figure 16 the UTMOS intervals all exclude 0.

EMA Lightning as published. The card (Aslan, 2026) reports WER 0.92 and CER 0.18, on a release candidate with its own normalizer according to its repository. The pip package 1.0.1 scored 1.00 and 0.198 on our own harness and 1.04 and 0.225 on the public one. PyPI has since moved to 1.1.2, which we did not re-run.

Against the teacher. On our own harness the student is not distinguishable from VoxCPM2 (1 seed) on WER or CER (Table 6), at about 1/300 of the size. It is lower on UTMOS (−0.24) and on 8 kHz UTMOS (−0.20).

Where the errors sit. A sentence that fails in every take points to a cause in the model, the recognizer or the scorer. On the public harness, Mini 1.0.1’s errors repeat across takes more than those of EMA Lightning, Trendyol-TTS and Alania-2 F1 (7 Oct) (Figure 17a). A sentence fails in a take if it has at least one word error, and 70.3% of Mini’s word errors sit in sentences that fail in all 3 takes, against 54.9% for EMA Lightning. Of Mini’s 24 such sentences, 17 also fail on all 3 own-harness seeds.

Many shared errors are recognizer or scorer efects. Mini and EMA Lightning both fail 15 sentences in at least 2 of 3 takes, and 12 of these also fail for the teacher. Ten sentences fail that often for Mini and for all eight rows from EMA Lightning to

![](images/47c36b961f9853446da52b1d9dc88aad77becc2973dd017857d890fac8c099e2.jpg)

![](images/1160d3f25e734ddc5c9d17a71f582258967aa4db9a9bea96bcba5afb802ade39.jpg)  
Figure 16 Mini 1.0.1 is level with EMA Lightning 1.0.1 on WER and CER and lower on full-band UTMOS. “Level with,” never used for UTMOS, means the WER and CER intervals include 0 on both harnesses and pooled.

![](images/201d2618653dc38315c5424fe9a254580d30ed319876746d1023ace5d48564bf.jpg)  
Figure 17 Mini’s errors repeat across takes more than those of the other three systems, and length is confounded with register. Public harness. (a) Many errors shared with EMA Lightning are recognizer or scorer efects.

Alania-2 F1 (7 Oct) in Figure 16, costing Mini 0.357 WER points (42 of its 128 word errors) and EMA Lightning 0.349.

Other repeats look like model defects. Sixteen sentences fail that often for Mini but not for EMA Lightning. None of them fails for the teacher, and from transcripts alone, with no listening, 12 look like real sound errors. In one, Whisper transcribes Çorbanın as Şorbanın in 3 of 3 takes, while 0 of 24 takes of other systems fail.

Short sentences score worse (3 to 4 words: Mini 3.68%, EMA Lightning 3.27%), but length is confounded with register, since all of those and 44 of the 59 sentences of 5 to 6 words are short-native. The hollow markers in Figure 17b are the other 15 sentences of 5 to 6 words, all everyday. Mini scores 0.77% on them (2 word errors), so within that register its short sentences are not worse.

How the results moved. From the end of the 23 h rental, the training block of Figure 7, to 1.0.0, settings, the two fixes of Section 6 and the final vocoder fine-tune together moved Mini from WER 1.30 to 0.97 (−0.33 [−0.56, −0.12]) and from CER 0.299 to 0.193 (−0.106 [−0.181, −0.041]) on our own harness, 3 seeds (Figure 18). No single step is distinguishable on both metrics, and the UTMOS panel has no paired interval. We tuned these settings on the benchmark, and the first row used another, batched code path.

![](images/a038403529779b766de98bbc977e34ba53966b52797ee14c1027587add858875.jpg)  
Figure 18 Settings, two fixes and the final vocoder fine-tune together moved WER from 1.30 to 0.97 and CER from 0.299 to 0.193; no single step is distinguishable on both. Each row adds to the one above. The row bundling five settings, with the code path, moved WER most, 1.30 to 1.115. Top row: no output EQ, batch 16. Third-row CER is about 0.25 because only a range, 0.2447 to 0.2496, survives for that run.

At the final settings on our own harness, 8 steps gave WER 0.97 and CER 0.193 over 3 seeds, against 1.17 and 0.230 at 4 steps. With the prefix but no doubling rule, 16 steps gave no gain and temperature 0.5 tied 0.6 (Appendix A).

UTMOS and the output EQ. EMA Lightning is ahead on full-band UTMOS on the public harness, 3.30 against 3.18, with Mini minus EMA Lightning at −0.115 [−0.136, −0.093]. The benchmark’s card asks for UTMOS in the 8 kHz band, which only our own harness measures, and there Mini is ahead, 3.05 against 2.88, +0.165 [+0.144, +0.186] (Figure 19c). We have no measurement of why the sign changes. UTMOS is an English-trained proxy and the two models have diferent voices, so we read both gaps with care (Section 9).

We have no record of a listener count or a blind test for the 1.0.1 EQ (Section 2). Against 1.0.0, public-harness UTMOS moved by −0.065 [−0.071, −0.060], with WER and CER within noise (last row of Figure 16).

Lesson. Freeze the output EQ before using UTMOS to compare checkpoints. On the same weights, four EQ settings span UTMOS 3.16 to 3.25 on our own harness (Figure 19b). With the first vocoder fine-tune, UTMOS rose with more bass and less treble while WER stayed flat (24 runs, seed 0).

b Same weights, only the EQ difers  
![](images/c375b253757f604805a2e90e78c21b059d0f47d9bd4b30a61cc0da425d5900bf.jpg)

![](images/8f3ca0575c2eb75ee51942a822b7fae27cc6c533220e90dd70a6f2f3a32be1d0.jpg)

![](images/ad8b83f71d641121a5d57630495e7317d745886789515f131fcf5954c218008b.jpg)  
Figure 19 Automatic scores moved slightly against the 1.0.1 EQ, which we chose by ear. (b) The 1.0.1 row includes its de-esser and fixed make-up gain. Paired 95% intervals on our own harness, 1.0.1 minus 1.0.0: UTMOS −0.068 [−0.074, −0.062], WER +0.05 [−0.05, +0.16]. (c) Mini is ahead of EMA Lightning in the 8 kHz band and behind at full band. UTMOS is an automatic, English-trained proxy, and the two models have diferent voices.  
Mini 1.0.1 Mini 1.0.0 EMA Lightning 1.0.1  
Batch 1 in every row. Mini 8 steps, EMA 4 per its card  
b Next to EMA Lightning

![](images/f5cae8aed68cfa697f9602d9b684b9b5d17528e3b2481fc2a19e58f572d3b95b.jpg)  
Figure 20 EMA Lightning was faster on both GPUs compared, and Mini 1.0.0 on the one CPU compared; no real phone was measured. h ll d h h fi h d k h machines. The 1.0.0 L40S and EPYC runs rebuilt the EQ filter per sentence; 1.0.1 was not timed there. The M5 CPU test had other jobs running.

## 8 Speed and where it runs

On one NVIDIA L4 (all 495 sentences, diferent VMs) Mini 1.0.1 runs at 35× real time, 1.65 times slower than plain EMA Lightning 1.0.1 by RTF and 9.5 times slower than compiled. On two threads of an Apple M5 laptop CPU (first 100 sentences, other jobs running) it runs at 41×. No real phone has been measured.

GPU. On one NVIDIA L4, Mini 1.0.1 has an RTF of 0.029 (35× real time). EMA Lightning 1.0.1 has 0.017 (58×) in plain PyTorch and 0.0030 (331×) compiled, with the same WER and CER (Figure 20b). By wall time the gap to plain EMA Lightning is 1.87 times, because Mini speaks 13% longer on the same text.

The public harness reports the middle of 3 passes over all 495 sentences at batch 1, with the normalizer and the output chain inside the timed region. Mini samples 8 steps and EMA Lightning 4, per its card (Aslan, 2026). The VMs difer, with 8 vCPUs for Mini and 16 for EMA Lightning. The timed render uses the published wheel’s cached EQ filter; 1.0.0, without that cache, has an RTF of 0.043. The public-harness scores in Section 7 come from an earlier pass with the same seeds and audio length, and we did not compare the audio.

Table 7 The package fetches 30.5 MB of weights on first use, the browser page 22.8 MB of graphs and a 14.2 MB runtime, and the arm64 Android package (APK) is 35.5 MB. The weight files and browser graphs are identical in 1.0.0 and 1.0.1. The fp16 file is converted back to fp32 at load, and its rounded durations can change on some sentences. The four ONNX graphs hold the networks, and the page’s JavaScript does the iSTFT, duration rounding, noise, chunking, cuts, EQ and limiter.
<table><tr><td>File</td><td>Bytes Holds</td><td></td></tr><tr><td>Weights, fp32, default download</td><td>30,499,764</td><td>Acoustic 14,757,148; vocoder 15,742,616</td></tr><tr><td>Weights, fp16, variant=&quot;fp16&quot;</td><td>15,257,774</td><td>Acoustic 7,383,130; vocoder 7,874,644</td></tr><tr><td>Python wheel, 1.0.1</td><td>1,397,120</td><td>Inference code and normalizer data, no weights</td></tr><tr><td>Browser graphs, mixed, page default</td><td>22,848,212</td><td>Text, condition, flow in fp32; vocoder in fp16 storage</td></tr><tr><td>WebAssembly (WASM) binary</td><td>14,239,897</td><td>ONNX Runtime Web 1.30.0</td></tr><tr><td>Android APK, arm64-v8a</td><td>35,549,175</td><td>Engine v0.1.1, model 1.0.1</td></tr><tr><td>Android APK, armeabi-v7a</td><td>34,574,333</td><td>Same engine, 32-bit ARM</td></tr></table>

Laptop CPU. On an Apple M5 CPU with 2 threads, the package default, and 8 steps, version 1.0.1 has an RTF of 0.024 (41×) on the first 100 benchmark sentences (Figure 20a). This is a median of 4 passes with other jobs running, at load 8 to 12 on 10 cores, and these runs of 5 November 2026 are the only ones with recorded machine load. On the model card’s sample text, the first chunk from stream(), the package’s streaming call, held 2.8 s of audio and took 72 ms. The L4 run covers all 495 sentences, so the two runs do not rank the machines. We have no EMA Lightning number on this machine.

Same machine as EMA Lightning. On one AWS g6e.2xlarge, Mini 1.0.0 was 2.07 times slower than plain EMA Lightning 1.0.1 on the L40S (RTF 0.026 against 0.0126) and 1.36 times faster on the AMD EPYC 7R13 with 8 vCPUs (0.052 against 0.071; Figure 20b). Our own harness ran the first 100 sentences, one pass, seed 0. Version 1.0.0 rebuilt its EQ filter per sentence, which alone took the L40S from 0.012 to 0.029 over 495 sentences. Version 1.0.1 caches the filter and has not been re-timed on this machine, so its ratios here are unknown. We did not run EMA Lightning’s compiled path here.

Browser. The browser page, a Hugging Face Space, runs the model client-side with ONNX Runtime Web (ONNX Runtime developers, 2018). It has run version 1.0.1 since 6 November 2026, and Table 7 gives its files. It matches the Python package to 0.036 dB mean log-spectral distance, the frame-averaged gap between two spectra, on 50 sentences. Lengths are identical, and so are Whisper transcripts on 50 of 50. As a control, a diferent seed gives 48 of 50.

We timed the page at 1.0.0, with the same weights and graphs and 1.0.0’s EQ. In headless Chrome on the same M5, with WebAssembly, it took about 0.38 s per demo sentence of 4.2 to 7.3 s on 4 threads (14×) and ran at 7.2× on 1 thread. We have not re-timed it at 1.0.1, whose output chain costs 4.8 ms per second of audio in Node on the M5, against 0.42 ms for 1.0.0’s. At released settings we have not measured Safari,

Table 8 Weights, inference code and the benchmark harness are public; training code and training data are not. Status on 6 November 2026. The first four public items are under Apache-2.0. The wheel’s model.py holds the training forward pass but imports a module the wheel does not ship, so the wheel does not train.
<table><tr><td>Public</td><td>Not public</td></tr><tr><td>Weights, fp32 and fp16 (Hugging Face)</td><td>Training code</td></tr><tr><td>Inference code, Python normalizer (PyPI wheel, sdist)</td><td>Data generation scripts</td></tr><tr><td>JavaScript normalizer, browser page, ONNX graphs (Space)</td><td>Training data, the chosen reference clip</td></tr><tr><td>Benchmark harness, adapters, per-clip scores (Hugging Face)</td><td>Own-harness code</td></tr><tr><td>Signed Android APK, binary only</td><td>ONNX export and verify tools, parity tests</td></tr><tr><td>Hosted API, needs a key</td><td>Android source, Kotlin normalizer source API server code</td></tr></table>

Firefox, phone browsers, WebGPU or a first download over a real network.

Android. A signed Android system TTS engine (v0.1.1, sideload, Android 7.0+, arm64 and 32-bit ARM) is publicly downloadable. It runs model 1.0.1, with an app shell adapted from sherpa-onnx (k2-fsa, 2022). On an Android API 35 arm64 emulator, a debug build matches the package at 1.0.1 to 0.036 dB on the same 50 sentences, with identical Whisper transcripts on 50 of 50. As a control, a diferent seed, checked on 10 sentences, difers by 3.74 dB. The signed arm64 APK itself loaded 1.0.1 and spoke in a smoke test there, and it has no network permission. The 32-bit APK has not been run.

With 4 vCPUs on the M5, that emulator ran a non-debuggable build of the earlier engine v0.1.0, with 1.0.0’s output chain, at 3.7× on 1 thread and 5.9× on 4 threads at 8 steps. At 1.0.1 the chain took 10.5 ms per second of audio on the debug build at 2 threads, 6.9% of that run’s time, and we have not re-timed the emulator’s speed. Our test notes put run-to-run noise at up to about 30%.

Package and API. The Python package installs from PyPI (PatientDesk AI, 2026a).

```python
pip install antalia-mini
from antalia_mini import Antalia
tts = Antalia() # CUDA if available, else CPU; fetches the fp32 weights once
text = "Randevunuz 14 Ekim Salı günü saat 15:30'da."
speech = tts.say(text, path="randevu.wav", seed=0) # 48 kHz by default
for chunk in tts.stream(text, seed=0): # first sentence first
```

The model is also free on our hosted API, with an API key and limits of 60 requests a minute and 5,000 a day. It is served on CPU at 1.0.1, and its latency is unmeasured. Table 8 lists what is public, the Availability section gives the links, and Appendix B says what a reader can re-run.

## 9 Related work

Training one model on another model’s audio has precedents back to at least 2021. We found no paper that uses shortcut self-consistency in a TTS acoustic model, and the one Turkish human MOS study of open systems we found orders them diferently from our UTMOS, on diferent renders.

Synthetic training data. The earliest precedent we found is TTS-by-TTS (Hwang et al., 2021), which added 179 h of speech from a Tacotron 2-based teacher to 5 h of recordings. It reports a MOS of 3.74, 40% above the conventional method. Zhou et al. (2025) train a 20.9M Matcha-TTS student on 100 h of purely synthetic speech.

![](images/6548842966eaaa3d632555931aab44add736f63ed1c49d41e97881c06f75c5cb.jpg)  
Figure 21 Nine complete TTS models in this list are smaller than Mini; among the works listed, Mini used the most synthetic speech, about 1,790 h. (a) Parameter counts as reported, not like for like; saanoTTS counts twice; Chatterbox at its full 801M. (b) VoxCPM2 is 2.29B, or 2.38B with its AudioVAE. Matcha-TTS’s bar marks CosyVoice2, though that work also used XTTS-v2 and ChatTTS. EMA Lightning’s row follows its card (Aslan, 2026); others, their sources (Hwang et al., 2021; Mahendrakar, 2026; Pipatanakul et al., 2026; Trikutam, 2026; Zhou et al., 2025).

Three recent works come closest to ours. By its card, EMA Lightning shares our teacher model (Section 1). Wayu-Paxa-TTS-Edge (Pipatanakul et al., 2026) is a fixedvoice Thai student trained only on synthetic speech from a 15 s reference. Its authors state that teacher errors become training targets, which matches the startof-clip ö defect our student learned (Section 6). Paradee (Mahendrakar, 2026) is an 8.07M single-voice English student of Kokoro-82M (hexgrad, 2025). It also uses the teacher’s internal features, so it learns no alignment, while we train on audio alone.

Trikutam (2026) reports that a small Hindi model trained from scratch on about 17.6 h fails, and warm-starts the 190M student in the figure from its teacher. Geng et al. (2026) find that synthetic data improves phonetic accuracy while suppressing prosodic variability, the range of rhythm and pitch. We did not check our model for this.

Small models. Figure 21a, a sample of systems, lists nine complete models smaller than Mini. They are two saanoTTS models (Thapa, 2026), two Pocket TTS models (Orsini, 2026), LE2E (Vecino et al., 2023), Nix-TTS (Chevi et al., 2023), Mini-MBiSTFT-VITS (Kawamura et al., 2023) and, by their own README and card, TinyTTS (tronghieuit, 2026) and Kokoro-7M-Distill (oddadmix, 2026).<sup>1</sup> The figure leaves out speed, because hardware difers between sources, and quality, because UTMOS across languages and voices is not comparable.

Few-step sampling. We use shortcut self-consistency (Frans et al., 2025). Its original paper evaluated it on images, and Flow2GAN tested it for mel-conditioned vocoding (Yao et al., 2026). We found no paper that uses it in a TTS acoustic model, which is absence of evidence only. EMA Lightning’s card describes another route, a fourstep student distilled with DMD2 (Yin et al., 2024) from a same-size flow teacher that needs 16 midpoint steps. Nguyen et al. (2025) report image artifacts when classifier-free guidance meets shortcut models. Our generator combines the two, and nothing in our evaluation isolates this efect.

Vocoder fine-tuning. Tacotron 2 (Shen et al., 2018) trains its WaveNet vocoder on ground-truth-aligned predictions and reports a MOS of 4.526 when synthesizing from predicted features, against 4.362 for a vocoder trained on ground truth. HiFi-GAN (Kong et al., 2020) fine-tunes on predicted mels and reports MOS 3.91 to 4.18 for its V1, 3.88 to 4.12 for V2 and 3.89 to 4.02 for V3, with mel error rising as perceptual quality rose. Those vocoders train on teacher-forced features. Our mels are sampled, with durations from alignment to the target clip and no teacher forcing, and we score WER, so the efect sizes do not compare. JETS avoids the step by training both stages jointly (Lim et al., 2022), and Guo et al. (2022) report that a discriminator on the acoustic model is more efective than fine-tuning.

Turkish TTS. The FreyaTTS report (Pamuk et al., 2026) has the only Turkish human MOS of open systems we found, from 24 raters and 2,566 ratings. Among its means are XTTS-v2 3.82, FreyaTTS 3.68, F5-TTS 3.63, MMS-TTS 3.58, Piper 3.47 and GlowTTS 2.53. Mini and EMA Lightning are not in it. Its order of systems disagrees with our UTMOS order on diferent renders, so we report no correlation. FreyaTTS and, by its card, EMA Lightning also read characters.

Speech quality metrics. The UTMOS strong learner we use comes from the English main track of the VoiceMOS 2022 challenge (Saeki et al., 2022), and we found no validation against Turkish listeners. Asonitis et al. (2026) find that automatic predictors agree with listeners on audible defects but fall near chance on clean TTS. Takagi et al. (2026), on Japanese speech, find them insensitive to prosodic errors, with most scoring lower-F0 speakers higher, and Mini and EMA Lightning have diferent voices. Paradee’s author reports a 0.5 UTMOS gap between two decoders that sounded alike by ear (Mahendrakar, 2026), and saanoTTS’s author reports that aggregate predictors missed a failure on sibilants (s-like sounds) evident in listening (Thapa, 2026). Ozkose et al. (2026) find that rankings of Turkish systems over eighteen objective metrics depend largely on metric family and utterance length.

## 10 Limitations and responsible use

No human listening test of this model exists, and we tuned its settings on the benchmark that scores it. The Whisper checkpoint that scores the benchmark also filtered the training data. The teacher’s start-of-clip defect remains in the model, hidden by the prefix, and we have not audited the voice’s likeness to any real speaker.

Evidence. All our quality numbers are automatic metrics on one benchmark. WER and CER are its band-matched secondary metric, and UTMOS is an English-trained proxy that we compare across diferent voices. Own-harness seeds ran on diferent GPUs and EMA Lightning’s on an H200. One seed spans about ±0.13 WER points, and the teacher has one seed.

Selection on the benchmark. Final-settings numbers are optimistic by selection. The doubling rule came from the benchmark’s errors, and we screened temperature, steps, speed, sufix and EQ on its 495 sentences. The public harness uses other seeds on the same sentences. The prefix rule’s scope came from training-data statistics, but its symptom also appeared on benchmark sentences. A worse benchmark score counted against one teacher reference (Section 3).

Whisper ran at beam 1 in QC and at beam 5 in scoring. Clips it could not transcribe left the long set, and our vocoder and settings choices used its WER. WER may therefore flatter the model relative to human intelligibility, by an amount we have not measured.

Errors that remain. The long set kept 9,271 lines whose first ö word is wrong by Whisper, filtered fine-tunes gave no demonstrated gain, and we have not regenerated the data at the source (Section 6). On the public harness, Mini 1.0.1’s errors repeat across takes more than EMA Lightning 1.0.1’s, and some of the repeats look like model defects (Section 7). Nobody has listened to these takes.

Scope. The model has one voice and one style and speaks Turkish only, with no style or emotion control beyond speed. English words are respelled, and whole English sentences are not supported. The model card names short interjections on their own as the weakest area, flat or rushed (PatientDesk AI, 2026a), and we have no metric for them alone. The vocoder never saw short-clip audio, and Whisper checked only 1.7% of the short set. Long text is split at sentence ends into independent chunks, so prosody does not carry across them. Readings follow the normalizer, so Roman numerals, ordinals before names, unusual formats and some foreign names can be read wrongly (Section 2).

Speed and builds. We have measured no real phone, only an Android emulator, and Section 11 lists the other timing runs we have not made. Other jobs were running during the M5 measurements. Version 1.0.1’s default EQ, which we chose by ear, lowers UTMOS by 0.065 against 1.0.0 on the public harness and raises WER and CER within noise.

Retraining, voice and use. Third parties cannot retrain the model (Appendix B). Section 3 gives the voice’s lineage. We ask users to tell listeners when they are hearing a synthetic voice, and not to use the model to impersonate real people or for fraud, harassment or deceptive calls.

## 11 What stays open

Four items stay open: ö clips regenerated by the teacher, a human listening test, real phones, and new timing runs of version 1.0.1.

Regenerate at the source. Our 4 renders per teacher configuration (Section 6) cannot pick a setting that reads first words right, so regeneration starts with more renders, checked by ear as well as by Whisper. Then the acoustic model retrains on the new clips and the vocoder fine-tunes on its new mels, a step neither filtered run completed. The defect is gone if the student, with the prefix of, gets ö first words right as often as first words with other capital letters, on a set larger than the benchmark’s 4 ö-initial sentences.

Listening test. A blind test with Turkish listeners on Mini 1.0.0 and 1.0.1, EMA Lightning 1.0.1 and the teacher would settle three questions that automatic metrics leave open. Listener transcripts would show whether WER flatters Mini. Ratings would show whether the UTMOS gaps to EMA Lightning, which run in opposite directions at full band and at 8 kHz, match what people hear across two voices, and how the 1.0.1 EQ rates against 1.0.0’s.

Phones. Our only Android speeds come from an emulator with 1.0.0’s output chain (Section 8). The engine’s app has a speed test, so running the signed packages on real phones would give the missing numbers, including for the 32-bit package, which has not been run at all.

Re-timing. Timing the browser page and the emulator at 1.0.1 would replace the numbers we measured with 1.0.0’s output chain. Our same-machine comparison with EMA Lightning also used Mini 1.0.0, so 1.0.1’s ratios on that machine’s L40S and 8-vCPU AMD EPYC are unknown (Section 8). Timing Mini 1.0.1 and EMA Lightning 1.0.1 on one machine would settle them, and we have no such run planned.

## Availability

Weights, inference code, the Python normalizer, the ONNX graphs and page code, and the benchmark harness are public under Apache-2.0. Training code, data generation scripts, training data, ONNX export tools and Android source are not (Table 8). The links below were live on 6 November 2026.

<table><tr><td>What</td><td>Where</td></tr><tr><td>Model card and weights</td><td>huggingface.co/cloudoday3/antalia-mini</td></tr><tr><td>Python package</td><td>pypi.org/project/antalia-mini</td></tr><tr><td>Browser page</td><td>huggingface.co/spaces/cloud0day3/antalia-mini</td></tr><tr><td>Signed Android engine, binary only</td><td>speech.patientdesk.ai/en/android</td></tr><tr><td>Hosted API, free with an API key</td><td>speech.patientdesk.ai/en/docs/quickstart</td></tr><tr><td>Benchmark harness, adapters, results.json, per-clip transcripts,</td><td>huggingface.co/datasets/PatientdeskAl/freya-tr-eval-rerun</td></tr><tr><td>scores and timing</td><td></td></tr><tr><td>Benchmark results page</td><td>speech.patientdesk.ai/en/benchmarks</td></tr></table>

## Credits

The benchmark is Freya-TR-Eval by Freya, under CC-BY-4.0 (Freya, 2026; Pamuk et al., 2026). Our scoring protocol and transcription recipe follow EMA Lightning’s card (Aslan, 2026) and the eval.py of its earlier repository, canberkkkkkk/ema-tts, under Apache-2.0, and score\_norm, the public harness’s text normalization for scoring, is copied from that file. The baseline settings for the cloning systems, including the voice reference clip for Chatterbox and XTTS-v2, come from the Anka TTS report (KrmKayabasi, 2026).

## Competing interests

The authors work at PatientDesk AI, which sells Alania-2 and Alania-1 and ran the benchmark on both harnesses. Both products appear in Table 5, along with Antalia-1, our open-weights release. We tuned Mini’s settings on the benchmark’s own sentences (Section 10).

## References

Canberk Aslan. EMA Lightning: Tiny, fast and accurate Turkish text to speech. https://huggingface.co/canbe rkkkkkk/ema-lightning, 2026. Hugging Face model card, revision e7c96f2.

Antonis Asonitis, Juan Pablo Zuluaga Gomez, Francesco Verdini, Aref Farhadipour, Marzieh Razavi, Pierre-Edouard Honnet, and Vijeta Avijeet. The limits of reference-free speech quality metrics as evaluators and rewards on modern text-to-speech. arXiv:2609.13150, 2026.

Selma Babayiğit. Does a truly symmetrically transparent orthography exist? Spelling is more dificult than reading even in an orthography considered highly transparent for both reading and spelling. Reading and Writing, 35 (10):2453–2472, 2022. doi: 10.1007/s11145-022-10259-5.

Robert Bristow-Johnson. Audio EQ cookbook. https://www.w3.org/TR/2021/NOTE-audio-eq-cookbook-202 10608/, 2021. W3C Working Group Note, 8 June 2021, edited by Raymond Toy.

Carnegie Mellon University. The CMU pronouncing dictionary. https://github.com/cmusphinx/cmudict, 2014.

Edresson Casanova, Kelly Davis, Eren Gölge, Görkem Göknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, and Julian Weber. XTTS: a massively multilingual zero-shot text-to-speech model. In Interspeech 2024, pages 4978–4982, 2024. doi: 10.21437/Interspeech.2024-2016.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-�: Fast training of difusion transformer for photorealistic text-to-image synthesis. In International Conference on Learning Representations (ICLR), pages 57611–57640, 2024.

Rendi Chevi, Radityo Eko Prasojo, Alham Fikri Aji, Andros Tjandra, and Sakriani Sakti. Nix-TTS: Lightweight and end-to-end text-to-speech via module-wise distillation. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 970–976, 2023. doi: 10.1109/SLT54892.2023.10023322.

Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification. In Interspeech 2020, pages 3830–3834, 2020. doi: 10.21437/Interspeech.2020-2650.

Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step difusion via shortcut models. In International Conference on Learning Representations (ICLR), pages 34668–34684, 2025.

Freya. Freya-TR-Eval: A general-purpose Turkish TTS evaluation benchmark. https://huggingface.co/dataset s/freyavoice/freya-tr-eval, 2026. Dataset, CC BY 4.0.

Yizhong Geng, Yanliang Li, Jinghan Yang, Tianhan Jiang, Boxun An, Ya Li, and Xiaoyu Shen. Bridging the stabilityexpressivity gap: Synthetic data scaling and preference alignment for low-resource spoken language models. arXiv:2605.27383, 2026.

Haohan Guo, Hui Lu, Xixin Wu, and Helen Meng. A multi-scale time-frequency spectrogram discriminator for GANbased non-autoregressive TTS. In Interspeech 2022, pages 1566–1570, 2022. doi: 10.21437/Interspeech.2022-52.

Michael Hansen and The Rhasspy Project. Piper: A fast, local neural text to speech system. https://github.com /rhasspy/piper, 2023. Archived; development moved to https://github.com/OHF-Voice/piper1-gpl.

hexgrad. Kokoro-82M. https://huggingface.co/hexgrad/Kokoro-82M, 2025. Hugging Face model card.

Jonathan Ho and Tim Salimans. Classifier-free difusion guidance. arXiv:2207.12598, 2022. A short version appeared in the NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

Min-Jae Hwang, Ryuichi Yamamoto, Eunwoo Song, and Jae-Min Kim. TTS-by-TTS: TTS-driven data augmentation for fast and high-quality speech synthesis. In ICASSP 2021 – 2021 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 6598–6602, 2021. doi: 10.1109/ICASSP39728.2021.9414408.

ITU-R. Algorithms to measure audio programme loudness and true-peak audio level. Recommendation ITU-R BS.1770-5, International Telecommunication Union, November 2023. URL https://www.itu.int/rec/R-REC -BS.1770.

Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kim, and Juntae Kim. UnivNet: A neural vocoder with multi-resolution spectrogram discriminators for high-fidelity waveform generation. In Interspeech 2021, pages 2207–2211, 2021. doi: 10.21437/Interspeech.2021-1016.

k2-fsa. sherpa-onnx: Speech-to-text, text-to-speech, speaker diarization, speech enhancement, source separation, and VAD using next-gen Kaldi with ONNX Runtime. https://github.com/k2-fsa/sherpa-onnx, 2022.

Masaya Kawamura, Yuma Shirahata, Ryuichi Yamamoto, and Kentaro Tachibana. Lightweight and high-fidelity end-to-end text-to-speech with multi-band generation and inverse short-time Fourier transform. In ICASSP 2023 – 2023 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10095296.

Hyeongju Kim, Jinhyeok Yang, Yechan Yu, Seunghun Ji, Jacob Morton, Frederik Bous, Joon Byun, and Juheon Lee. SupertonicTTS: Towards highly eficient and streamlined text-to-speech system. arXiv:2503.23108, 2025.

Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon. Glow-TTS: A generative flow for text-to-speech via monotonic alignment search. InAdvances in Neural Information Processing Systems, volume 33, pages 8067–8077, 2020.

Guillaume Klein and SYSTRAN. faster-whisper: Faster Whisper transcription with CTranslate2. https://github.c om/SYSTRAN/faster-whisper, 2023.

Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. HiFi-GAN: Generative adversarial networks for eficient and high fidelity speech synthesis. In Advances in Neural Information Processing Systems, volume 33, pages 17022–17033,

2020.

KrmKayabasi. Anka TTS v0.1. https://huggingface.co/krmkayabasi/Anka-TTS, 2026. Hugging Face model card and benchmark report.

Aditi Lahiri and Jorge Hankamer. The timing of geminate consonants. Journal ofPhonetics, 16(3):327–338, 1988. doi: 10.1016/S0095-4470(19)30506-6.

Dan Lim, Sunghee Jung, and Eesung Kim. JETS: Jointly training FastSpeech2 and HiFi-GAN for end to end text to speech. In Interspeech 2022, pages 21–25, 2022. doi: 10.21437/Interspeech.2022-10294.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11966–11976, 2022. doi: 10.1109/CVPR52688.2022.01167.

Sahil Mahendrakar. Paradee: Distilling Kokoro-82M into an 8M-parameter single-voice text-to-speech model. arXiv:2610.06817.2026

Xudong Mao, Qing Li, Haoran Xie, Raymond Y. K. Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2813–2821, 2017. doi: 10.1109/ICCV.2017.304.

Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, and Gustav Eje Henter. Matcha-TTS: A fast TTS architecture with conditional flow matching. In ICASSP 2024 – 2024 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 11341–11345, 2024. doi: 10.1109/ICASSP48485.2024.10448291.

Anh Nguyen, Viet Nguyen, Duc Vu, Trung Dao, Chi Tran, Toan Tran, and Anh Tran. Improved training technique for shortcut models. arXiv:2510.21250, 2025. Accepted at NeurIPS 2025.

oddadmix. Kokoro-7M-Distill. https://huggingface.co/oddadmix/Kokoro-7M-Distill, 2026. Hugging Face model card.

Banu Öney and Aydin Yücesan Durgunoğlu. Beginning to read in Turkish: A phonologically transparent orthography. Applied Psycholinguistics, 18(1):1–15, 1997. doi: 10.1017/S014271640000984X.

ONNX Runtime developers. ONNX Runtime. https://onnxruntime.ai, 2018.

Manu Orsini. The tiny TTS was actually 10x too big. https://kyutai.org/blog/2026-10-20-pocket-tts-sma ll/, October 2026. Blog post.

Yunus Emre Ozkose, Alperen Kahraman, and Ali Haznedaroglu. A comprehensive objective evaluation of modern text-to-speech for Turkish using speech quality assessment models. In Speech and Computer (SPECOM 2026), Lecture Notes in Computer Science, pages 456–472. Springer, 2026. doi: 10.1007/978-3-032-37867-5\_31.

Ahmet Erdem Pamuk, Ömer Yentür, Ahmet Tunga Bayrak, Yavuz Alp Sencer Öztürk, and Mustafa Yavuz. FreyaTTS: A compact tokenizer-free flow-matching transformer for Turkish-first speech synthesis. arXiv:2607.09530, 2026.

PatientDesk AI. Antalia-2 Mini: A small Turkish text-to-speech model. https://huggingface.co/cloud0day3/an talia-mini, 2026a. Version 1.0.1.

PatientDesk AI. Freya-TR-Eval, re-run: Scripts and per-sentence results. https://huggingface.co/datasets/Pa tientdeskAI/freya-tr-eval-rerun, 2026b. Results page: https://speech.patientdesk.ai/en/bench marks.

William Peebles and Saining Xie. Scalable difusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2023. doi: 10.1109/ICCV51070.2023.00387.

Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Rafel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. FineWeb2: One pipeline to scale them all – adapting pre-training data processing to every language. arXiv:2506.20920, 2025.

Kunat Pipatanakul, Potsawee Manakul, Warit Sirichotedumrong, Sittipong Sripaisarnmongkol, Pakorn Nathong, and Phatrasek Jirabovonvisut. Building and evaluating fixed-voice Thai TTS from synthetic speech. arXiv:2609.03502, 2026.

Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, and Michael Auli. Scaling speech technology to 1,000+ languages. Journal of Machine Learning Research, 25(97):1–52, 2024.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings ofthe 40th International Conference on Machine

Learning (ICML), volume 202 of Proceedings ofMachine Learning Research, pages 28492–28518, 2023.

Resemble AI. Chatterbox-TTS. https://github.com/resemble-ai/chatterbox, 2025. GitHub repository.

Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari. UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022. In Interspeech 2022, pages 4521–4525, 2022. doi: 10.214 37/Interspeech.2022-439.

Sezgin Saygili, Emre Kaplaner, Oncel Ozgul, and Fikri San Koktas. Antalia 1: An open Turkish text-to-speech model from a rights-clean pipeline. https://huggingface.co/cloud0day3/antalia-1, 2026. Technical report.

Noam Shazeer. GLU variants improve transformer. arXiv:2002.05202, 2020.

Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, R. J. Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, and Yonghui Wu. Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4779–4783, 2018. doi: 10.1109/ICASSP.2018.8461368.

Hubert Siuzdak. Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis. In International Conference on Learning Representations (ICLR), pages 25719–25733, 2024.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063.

Supertone. Supertonic: Lightning fast, on-device TTS. https://huggingface.co/Supertone/supertonic, 2025. Hugging Face model card.

Masato Takagi, Masaya Kawamura, Reo Shimizu, and Yuma Shirahata. Investigating human-model discrepancies in speech quality assessment via acoustic and prosodic perturbations. In Interspeech 2026, pages 3528–3532, 2026. doi: 10.21437/Interspeech.2026-1478.

Ashish Thapa. saanoTTS: The smallest real-time neural TTS on a general-purpose microcontroller. arXiv:2608.21378, 2026.

Trendyol. Trendyol-TTS. https://huggingface.co/Trendyol/Trendyol-TTS, 2026. Hugging Face model card.

Sivateja Trikutam. Staged depth-pruning distillation of a flow-matching text-to-speech teacher: A compact Hindi speech synthesizer. arXiv:2607.18662, 2026.

tronghieuit. TinyTTS: Ultra-lightweight English text-to-speech. https://github.com/tronghieuit/tiny-tts, 2026. GitHub repository.

Nik Vaessen. JiWER: Evaluate your speech-to-text system with similarity measures such as word error rate. https: //github.com/jitsi/jiwer, 2018.

Biel Tura Vecino, Adam Gabryś, Daniel Mątwicki, Andrzej Pomirski, Tom Iddon, Marius Cotescu, and Jaime Lorenzo-Trueba. Lightweight end-to-end text-to-speech synthesis for low resource on-device applications. In 12th ISCA Speech Synthesis Workshop (SSW 2023), pages 225–229, 2023. doi: 10.21437/SSW.2023-35.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv:2505.09388, 2025.

Zengwei Yao, Wei Kang, Han Zhu, Liyong Guo, Lingxuan Ye, Fangjun Kuang, Weiji Zhuang, Zhaoqing Li, Zhifeng Han, Long Lin, and Daniel Povey. Flow2GAN: Hybrid flow matching and GAN with multi-resolution network for few-step high-fidelity audio generation. In International Conference on Learning Representations (ICLR), pages 747–761, 2026.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. InAdvances in Neural Information Processing Systems, volume 37, pages 47455–47487, 2024.

Tingxiao Zhou, Leying Zhang, Zhengyang Chen, and Yanmin Qian. Training text-to-speech model with purely synthetic data: Feasibility, sensitivity, and generalization capability. arXiv:2512.17356, 2025.

Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Jiancheng Gui, Jiaheng Wu, Ziyang Wang, Xudong Shen, Runchuan Ye, Zhisheng Zhang, Jiuyang Zhou, Bingsong Bai, Weiyue Sun, Mengyuan Deng, Qundong Shi, Zhiyong Wu, and Zhiyuan Liu. VoxCPM2 technical report. arXiv:2606.06928, 2026.

## A A selection of the settings we tried

Seventeen of the 31 comparisons here rest on one seed. Of the 14 with 3 seeds, only 8 steps against 4 moved WER by more than one seed’s span.

Figure 22 shows about one comparison per setting, picked from 123 evaluation jobs. Each row changes the named setting, on our own harness. In the first row the small dots are the 3 seeds at 8 sampler steps, the large dot is their mean and the dark tick is the same run without the change, at 4 steps. These scores are optimistic by selection, because we tuned on the same 495 benchmark sentences (Section 10). Kept means the release ships the setting, and several kept rows show no gain on WER. Mel sharpening scales each mel bin’s variation around its mean, and edge trim cuts silence at chunk edges down to the named milliseconds.

![](images/847dbfd5cdcbb3a6e8d28a566b5ee818efe4cc9d114b71072bc085ae32533b70.jpg)  
Figure 22 Of 14 rows with 3 seeds, only 8 steps against 4 at the final settings moved WER past the ±0.13 points one seed spans. Final: released weights, 1.0.0’s EQ unless named. Old: the same acoustic model with the first vocoder fine-tune. Batched: batch 16, temperature 1.0, no sufix or EQ.

The figure leaves out the old model’s temperatures 0.8, 0.7, 0.5 and 0.4, the other sufix variants at temperature 1.0, the other every-chunk prefixes, the 24-run tone sweep and the other EQ curves, and intermediate checkpoints.

The tinted band is one seed’s span, applied here to 3-seed means as well as single seeds. It gives a scale only, and a row inside it may still have an efect. Some values come from printed scorer lines instead of per-clip files. On the final vocoder finetune row, we kept seeds 1 and 2 only as one printed mean, drawn as one hollow mark.

Table 9 The shipped weights come from 471,880 acoustic updates and 535,350 vocoder updates. All stages ran on H200 GPUs except the final vocoder fine-tune, on one L40S and then 4 A100 80GB.
<table><tr><td>Stage</td><td>Updates</td><td>Starts from</td><td>Trains on</td></tr><tr><td colspan="4">Acoustic model, 3,686,977 parameters</td></tr><tr><td>Main run</td><td>420,000</td><td>No parent</td><td>Long set</td></tr><tr><td>Short-clip fine-tune</td><td>51,880</td><td>Main run, step 420,000</td><td>Both sets, short clips ×6</td></tr><tr><td>Shipped acoustic model</td><td>471,880</td><td></td><td></td></tr><tr><td colspan="4">Vocoder, 3,933,954 parameters</td></tr><tr><td>Real-mel vocoder</td><td>157,000</td><td>No parent</td><td>Long set, real mels</td></tr><tr><td>First vocoder fine-tune</td><td>301,490</td><td>Real-mel vocoder, step 157,000</td><td>Long set, mels from the main run</td></tr><tr><td>Final vocoder fine-tune</td><td>76,860</td><td>First vocoder fine-tune, step 301,490</td><td>Long set, mels from the short-clip fine-tune</td></tr><tr><td>Shipped vocoder</td><td>535,350</td><td></td><td></td></tr></table>

## B Reproducing the numbers

The public-harness rows of the open-weights systems can be re-run from public files. Retraining the model or re-running our own harness needs files we have not published.

What a reader can re-run. The benchmark sentences are Freya-TR-Eval (Freya, 2026; Pamuk et al., 2026), and our dataset holds the harness, one adapter per open system, results.json, and per-clip transcripts, scores and timing (PatientDesk AI, 2026b). Every one of the 20 rows of results.json, closed systems included, can be recomputed with zero diference from the per-clip files, without rendering anything. Table 5 leaves out EMA Lightning’s compiled build, Alania-2 F1 (7 Oct) as a bare model, and a 100-sentence Alania-1 speed row. Mini 1.0.0 ran on the same harness and seeds but is not in that file, which now holds 1.0.1. To render again, install package version 1.0.1, run run\_antalia\_mini.py on a CUDA GPU, then score.py and aggregate.py, and compare with results.json. For EMA Lightning, install 1.0.1.

Table 9 traces the shipped weights. On the model repository, commit 8a79528 is 1.0.1 and commit e0d3182b is 1.0.0, and Antalia(revision=...) selects one. Re-timing 1.0.0 also needs its own wheel from PyPI, because the 1.0.1 wheel caches the EQ filter. Settings are options, so Antalia(prefix="", gem\_stretch=1.0) turns of the hidden prefix and the doubling rule.

Seeds fix the audio only on one software stack (Section 2). Re-scoring the 7 Oct Alania-2 F1 audio on an A100, the GPU that scored the 12 Oct Alania-2 rows, changed 3 of 1,485 transcripts and WER by 0.0085 points. RTF depends on the machine, and ours were L4s on GCP g2-standard-8 for Mini and g2-standard-16 for EMA Lightning.

What a reader cannot re-run. Training code, data generation scripts, training data and the chosen reference clip are not public (Table 8), and the shipped runs used flags that exist only in our private repository, so third parties cannot retrain the model. The data recipe exists only as prose, on the model card and in Section 3. The analyses in Sections 3 to 6 rest on that private material. The own-harness rows, including every 8 kHz UTMOS value, use code that is not public, and so do the parity tests of Section 8 and the normalizer checks below. The dataset has the per-clip transcripts behind the public-harness intervals of Section 7 but no bootstrap script,

so a reader has to write one.

A normalizer replay of 31,584 recorded text cases on the 1.0.1 code gave 0 mismatches. The browser page’s JavaScript port matches the Python code on 1,171 of 1,171 sentences, and the Android engine’s Kotlin port on 31,584 of 31,584 cases on the JVM. Replays catch code changes and cannot catch a wrong reading.