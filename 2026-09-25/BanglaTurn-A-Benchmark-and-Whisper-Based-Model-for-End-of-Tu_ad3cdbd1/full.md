# BanglaTurn: A Benchmark and Whisper-Based Model for End-of-Turn Detection in Bangla Speech

Mizbaul Haque Maruf <sup>ID</sup> <sup>1</sup>

<sup>1</sup> Vivasoft Limited, Dhaka, Bangladesh

mizbaul.haque@vivasoftltd.com

## Abstract

This paper presents BanglaTurn, a corpus for end-of-turn detection in Bangla conversational speech, and a model trained on it. The corpus holds 35,374 samples of 3 to 15 s of podcast speech, labelled for turn state by combining speaker diarization with an LLM pass, with every label then checked by a human annotator. The model pairs a Whisper encoder with task-specific classification heads. On a class-balanced test set drawn from a heldout podcast, it reaches 84.33% accuracy (95% CI 80.3 to 88.1) against 69.28% for the Smart-Turn v3 baseline, and lowers the false negative rate from 51.57% to 7.55% at the cost of a higher false positive rate. We report what encoder layer fine-tuning, multi-scale pooling and INT8 quantization each contribute, and latency stays within 165 to 191 ms end to end on CPU.

Index Terms: turn-taking, end-of-turn detection, low-resource languages, Bangla, Whisper

## 1. Introduction

Turn-taking governs how speakers coordinate the exchange of speaking turns [1]. Detecting when a speaker has finished, the task of end-of-turn detection, matters for conversational AI systems [2]: get it wrong and the system either cuts the user off or leaves an awkward gap, and users notice both in voice assistants and spoken dialogue applications. We use turn detection as shorthand for this task throughout.

Early systems relied on silence thresholds [3], which cannot separate a genuine turn ending from a pause in the middle of an utterance. Later work modelled acoustic and prosodic cues with neural networks [4, 5], and self-supervised representations from Wav2Vec 2.0 [6], HuBERT [7] and Whisper [8] made transfer learning practical for the task. Low-resource languages have seen little of this progress. Bangla has more than 230 million speakers but no dedicated turn detection dataset, and multilingual models such as Smart-Turn v3 do poorly on it because Bangla conversational speech is thinly represented in their training data.

This paper makes two contributions. The first is BanglaTurn, a Bangla turn detection corpus of 35,374 annotated samples. The second is a Whisper encoder-based model that reaches 84.33% accuracy, 15 points above the baseline, at speeds usable in a live system.

## 2. Related Work

## 2.1. Turn detection systems

Early turn detection systems combined silence thresholds with prosodic features such as pitch contours and energy patterns [3, 9]. Statistical models (HMMs, CRFs) then captured temporal dependencies across acoustic and linguistic features. Neural approaches followed, first with continuous LSTM-based prediction [4], then with models that fused acoustic, prosodic and lexical cues [5]. More recently, TurnGPT [10] predicts turntaking events from text using language-model pretraining, while Voice Activity Projection [11] learns them from self-supervised speech representations. Other work targets latency-aware prediction for real-time systems [12] and multi-party conversation [13]. Prosody remains central throughout: Ward and Vega [14] show that phrase-final words signal completion through falling pitch, lengthening and reduced intensity. Work presented at O-COCOSDA has studied the same cues from the corpus side. Furukawa et al. [15] built a multimodal corpus for modelling turn management in multi-party conversation, and Ishimoto and Enomoto [16] showed that final pitch lowering shapes how listeners perceive the end of an utterance in spontaneous Japanese.

![](images/3c67575c33109cacdf4bab78cda6bca9c3e09e587440d78cf7bd12e0e67354e8.jpg)  
Figure 1: Our proposed model architecture with Whisper-tiny encoder finetuned in Bangla, multi-scale pooling and a classifier.

## 2.2. Speech representations

Self-supervised models learn transferable representations from unlabelled audio. Wav2Vec 2.0 [6] uses contrastive learning over quantized representations, HuBERT [7] predicts masked discrete hidden units, and XLSR [17] extends the approach to low-resource languages through cross-lingual pretraining.

Whisper [8] is trained on 680,000 hours of weakly supervised data covering 96 languages, Bangla among them, which makes its encoder a reasonable starting point for transfer [18]. Fine-tuning the top encoder layers while freezing the lower ones trades adaptation against generalization [19]. We are not aware of work that adapts these models to turn detection in a lowresource language.

![](images/a7253e5ea41596d17ada95faab316f5b16881d605d9a86b9455bc9a5710f711d.jpg)  
Figure 2: Dataset creation workflow. Each box names a stage, with the tool or setting it uses beneath it: public podcasts are normalised to 16 kHz mono, split into speaker turns by PyAnnote, cut into 3–15 s chunks by Silero VAD, labelled for turn state by combining the diarization with Gemini 2.0 Flash Lite, with every label checked by a human annotator, and collected into the 35,374-sample BanglaTurn corpus.

## 2.3. Bangla speech processing

Bangla is underrepresented relative to high-resource languages. Mozilla Common Voice [20] and Kjartansson et al. [21] provide crowdsourced Bangladeshi Bengali speech, and BanglaBERT [22] set benchmarks for Bangla language understanding. Bengali speech resources also have a history at O-COCOSDA, from a read-speech corpus for continuous speech recognition [23] to a prosodically annotated Bengali and Assamese audiobook corpus for sentence boundary detection [24]. Sentence boundaries in read speech are related to turn ends in conversation but are not the same thing, since a conversational speaker can pause at a sentence boundary and still hold the floor. Turn detection has nothing comparable. There is no public Bangla dataset or benchmark for it, and multilingual models degrade on Bangla because of its prosody and its small share of training data. We built BanglaTurn to fill that gap.

## 3. BanglaTurn Detection Dataset

Conversational Bangla audio is scarce, so we built a pipeline that collects it, separates the speakers, cuts the result into usable segments, and annotates them with model assistance.

## 3.1. Dataset creation

Figure 2 shows the pipeline. The audio comes from Banglalanguage conversational podcasts that are publicly viewable on YouTube. We chose podcasts because they are conversational, reasonably clean, and cover a range of topics. Section 8.1 describes the gated terms under which we release this audio. Audio was normalized to 16 kHz, 16-bit mono PCM. Speaker boundaries come from PyAnnote Speaker Diarization 3.1 [25], which combines voice activity detection, ECAPA-TDNN embeddings [26] and agglomerative clustering. We merge adjacent segments from the same speaker and discard anything under 2.0 s. Where the speakers are known in advance, we identify them automatically by verification against reference samples.

Diarization returns segments of arbitrary length, but the model takes a fixed-duration input, so we chunk with Silero VAD under two constraints: each chunk runs 3.0 s to 15.0 s, with at most 2.0 s of trailing silence.

## 3.2. Label scheme

Each sample carries one target label and two auxiliary flags. The target is binary. An endpoint is a clip that ends with a genuine turn completion, where the speaker yields the floor (e.g. “ami kal bajare giyechilam.”, “I went to the market yesterday.”). A non-endpoint is a clip after which the same speaker carries on. The model is trained and evaluated on this label alone.

The two flags mark hesitation sounds and discourse fillers, such as “um” or “mane” (“I mean”). They are not classes. Mid-filler marks a filler inside the clip (e.g. “ami... accha...

bhabchilam je amra eta korte pari.”, “I... well... was thinking that we can do this.”). End-filler marks a filler in the last one to three words (e.g. “ami kal bajare giyechilam... uh...”, “I went to the market yesterday... uh...”). The flags are independent of each other and of the target. A speaker can trail off with a filler and then yield the floor, or finish a filler-free sentence and keep going. In the test set, 248 of the 319 clips carry a mid-filler and 104 an end-filler, and both occur with either label. We keep the flags because fillers are where a pause is most easily mistaken for a turn end, and a system that makes that mistake either interrupts the speaker or stalls.

## 3.3. Annotation protocol

Labels come from two automatic sources, which a human annotator then reconciled. The first is diarization. The last chunk of a speaker’s turn before a different speaker begins is proposed as an endpoint, and every other chunk as a non-endpoint. This rule is noisy, because diarization errors, overlapping speech and backchannels all create spurious speaker changes. The second is an LLM pass with Google’s Gemini 2.0 Flash Lite, following the LLM-assisted annotation approach of Gilardi et al. [27]. For each clip, a fixed prompt asks for a Bangla transcription, the two filler flags, and one of four turn states, each defined with examples: Complete (intent fully expressed), Incomplete (a pause inside an unfinished thought), Backchannel (a brief listener response) and Wait (a request to pause or stop). Complete and Wait map to endpoint, Incomplete and Backchannel to nonendpoint, and only the binary label is kept.

A single annotator, the author, a native Bangla speaker, then listened to every clip alongside both proposals and the transcription. Where the proposals agreed, the annotator confirmed or corrected the label. Where they disagreed, the annotator decided by listening to the clip alone, without the surrounding audio. Filler flags and transcriptions were checked in the same pass.

This protocol has two gaps. Because one person checked every label, we cannot report inter-annotator agreement. Because corrections were made in place without a log, we cannot report how often the human check changed the automatic labels. Section 7 considers what this means for the results.

## 3.4. Dataset details

BanglaTurn holds 35,374 samples split into training (31,549), validation (3,506) and test (319) sets, as detailed in Table 1. Training and validation keep the natural class distribution of roughly 80% endpoints, while the test set is balanced at 49.84% endpoints. The test set is small, but it is not a random in-domain split. It consists of handpicked, class-balanced and deliberately difficult examples taken from a separate podcast that was held out entirely from training and validation. The same annotator who checked the labels selected these examples. The 319 clips run 3.0 to 15.0 s and total 51.6 minutes. Because the test set comes from a different speaker and a different distribution, it tests generalization rather than in-domain memorization, which we consider a harder and more informative check than a larger split drawn from the same recordings. Its small size does make every figure uncertain, so we report confidence intervals (Section 5).

Table 1: BanglaTurn dataset statistics. Endpoint and nonendpoint partition each split.
<table><tr><td>Metric</td><td>Training</td><td>Validation</td><td>Test</td></tr><tr><td>Total samples</td><td>31,549</td><td>3,506</td><td>319</td></tr><tr><td>Endpoint</td><td>25,276</td><td>2,829</td><td>159</td></tr><tr><td>Non-endpoint</td><td>6,273</td><td>677</td><td>160</td></tr></table>

## 4. Model Architecture

Our model reuses Whisper’s encoder for binary turn detection, as Figure 1 shows. The task does not call for transcription, only for the acoustic, prosodic and temporal structure of the audio, and the encoder already represents that, so we drop the decoder.

## 4.1. Encoder

We use a Whisper Tiny encoder fine-tuned on the Mozilla Common Voice 11 Bangla dataset [20]. Its input is an 80-channel log-Mel spectrogram covering 8 seconds of audio.

## 4.2. Multi-scale pooling

Turn detection depends on local acoustic events, such as the final phoneme and the intonation contour, and on the shape of the utterance as a whole. To capture both, we concatenate four pooled views of the encoder output: max pooling for prominent features, mean pooling for the overall representation, attention pooling with a learnable query that can settle on turn-relevant regions, and last-frame pooling for the final temporal state.

## 4.3. Classification head

The head is a stack of linear layers with layer normalization, GELU activations and dropout, ending in a single logit.

## 5. Experimental Setup

We compare against Smart-Turn v3 on the same BanglaTurn test split and ablate three variants: the stock Whisper-tiny encoder, trained on 99 languages including Bangla; the same encoder fine-tuned on Bangla; and that Bangla fine-tuned encoder with multi-scale pooling.

We also compare against two classical baselines that use no learned speech representation. The first is a silence threshold, which predicts an endpoint when the clip’s trailing silence is at least τ seconds. Trailing silence is the time after the last frame within 35 dB of the clip’s peak energy. The second is a logistic regression over trailing silence and four prosodic features of the final 500 ms of speech: F0 slope and F0 level relative to the clip median, both from pYIN [28], energy slope, and the duration of the final speech segment, plus an indicator for clips with no final pitch estimate. We fit both on the test podcast itself by stratified 10-fold cross-validation, choosing τ or the regression weights on nine folds and predicting the tenth. The baselines thereby see the test speaker and recording conditions, which the neural models never do, so their scores are optimistic.

Training uses Binary Cross-Entropy (logits) loss with dynamic class weighting, which keeps the model from collapsing onto the majority class. We use AdamW [29] at learning rate 5e-5, weight decay 0.01 and gradient clipping at 1.0, for 4 epochs at batch size 16, with a 0.2 linear warmup ratio and cosine decay, on a single NVIDIA A40 (48 GB).

Two error rates matter most here. A false positive, predicting an endpoint while the speaker is still mid-turn, makes the system interrupt. A false negative makes it wait when it should respond. With endpoint as the positive class and a decision threshold of 0.5, we report the false positive rate $\mathrm { F P R } \ = \ \mathrm { F P } / ( \mathrm { F P } \ + \ \mathrm { T N } )$ and the false negative rate FNR $= \mathrm { F N } / ( \mathrm { F N } + \mathrm { T P } )$ alongside accuracy, precision, recall and F1, together with inference latency and model size. Because the test set is small, we attach 95% percentile bootstrap confidence intervals to accuracy and F1, computed from 10,000 resamples of the 319 test clips.

## 6. Results and Analysis

## 6.1. Overall performance comparison

Our model reaches 84.33% accuracy where Smart-Turn v3 reaches 69.28% (Table 2). The confidence intervals do not overlap. An unpaired two-proportion test on accuracy, which is conservative for two models scored on the same clips, gives $z = 4 . 5 0 , p < 1 0 ^ { - 5 }$ . The baseline is precise, at 82.80%, but it misses a great deal, at 48.43% recall. Ours is more balanced, at 79.46% precision and 92.45% recall. Table 4 gives the underlying counts. The clearest difference is in missed endpoints, which fall from 82 to 12, a false negative rate of 51.57% against 7.55% and close to a sevenfold reduction. We pay for that with more false positives, 38 against 16, an FPR of 23.75% against 10.00%. F1 rises from 61.11% to 85.47%.

Neither classical baseline comes close, even though both were fitted on the test podcast. The silence threshold scores 49.22%, no better than chance. The prosody and silence regression reaches 62.07%, and nearly all of that comes from trailing silence. Fitted on silence alone, the same regression reaches 63.32%, and on the prosodic features alone, 50.78%. Section 7 explains why silence behaves this way on BanglaTurn.

## 6.2. Ablation study

Table 3 separates three factors: the encoder, generic or Bangla fine-tuned; the addition of Multi-Scale Pooling (MSP); and the number of unfrozen encoder layers.

With the encoder frozen, the Bangla fine-tuned version edges out the generic one, 84.68 against 83.83 F1. MSP helps once encoder layers are unfrozen, most at 2 unfrozen layers (85.47 against 82.66 F1) and less at 4 (84.75 against 83.29). With a frozen encoder it is slightly worse (84.21 against 84.68). Unfreezing 4 layers with MSP gives the lowest FNR in the table, 5.66%, but costs precision, 76.92% against 79.46% at 2 layers, which suggests the extra capacity starts fitting the training podcasts rather than the task. We settled on the Bangla fine-tuned encoder with MSP and 2 unfrozen layers, at 84.33% accuracy and 85.47 F1.

These differences should be read with care. The accuracy intervals span about ±4 points, and no two rows differ by more than 3.14 points, so the ablation suggests an ordering of configurations but does not establish that any one is better than another.

Table 2: Performance comparison on the BanglaTurn test set (all metrics in %, endpoint as the positive class). Brackets give 95% bootstrap confidence intervals. FPR is computed over the 160 non-endpoint clips and FNR over the 159 endpoint clips. <sup>†</sup>Fitted on the test podcast itselfby 10-fold cross-validation, and therefore optimistic.
<table><tr><td>Model</td><td>Acc</td><td>FPR</td><td>FNR</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td>Silence threshold†</td><td>49.22 [43.9, 54.9]</td><td>11.88</td><td>89.94</td><td>45.71</td><td>10.06</td><td>16.49 [9.6, 23.5]</td></tr><tr><td>Prosody + silence LR†</td><td>62.07 [56.7, 67.4]</td><td>50.00</td><td>25.79</td><td>59.60</td><td>74.21</td><td>66.11 [60.2, 71.4]</td></tr><tr><td>Smart-Turn v3</td><td>69.28 [63.9, 74.3]</td><td>10.00</td><td>51.57</td><td>82.80</td><td>48.43</td><td>61.11 [53.7, 67.7]</td></tr><tr><td>Ours</td><td>84.33 [80.3, 88.1]</td><td>23.75</td><td>7.55</td><td>79.46</td><td>92.45</td><td>85.47 [81.2, 89.3]</td></tr></table>

Table 3: Ablation study comparing different model variants on the BanglaTurn test set. MSP denotes Multi-Scale Pooling. FPR and FNR are computed over the non-endpoint and endpoint clips respectively. With 319 test clips, the 95% confidence interval on each accuracy spans about ±4 points, wider than most gaps between rows.
<table><tr><td>Encoder + Components</td><td>Unfrozen layers</td><td>Acc (%)</td><td>FPR (%)</td><td>FNR (%)</td><td>Prec (%)</td><td>Rec (%)</td><td>F1 (%)</td></tr><tr><td>Whisper-tiny</td><td>0</td><td>83.07</td><td>21.88</td><td>11.95</td><td>80.00</td><td>88.05</td><td>83.83</td></tr><tr><td>Whisper-tiny</td><td>2</td><td>81.19</td><td>29.38</td><td>8.18</td><td>75.65</td><td>91.82</td><td>82.95</td></tr><tr><td>Whisper-tiny</td><td>4</td><td>83.07</td><td>27.50</td><td>6.29</td><td>77.20</td><td>93.71</td><td>84.66</td></tr><tr><td>Whisper-tiny (Bangla)</td><td>0</td><td>84.01</td><td>20.62</td><td>11.32</td><td>81.03</td><td>88.68</td><td>84.68</td></tr><tr><td>Whisper-tiny (Bangla)</td><td>2</td><td>81.19</td><td>27.50</td><td>10.06</td><td>76.47</td><td>89.94</td><td>82.66</td></tr><tr><td>Whisper-tiny (Bangla)</td><td>4</td><td>81.50</td><td>29.38</td><td>7.55</td><td>75.77</td><td>92.45</td><td>83.29</td></tr><tr><td>Whisper-tiny (Bangla) + MSP</td><td>0</td><td>83.07</td><td>24.38</td><td>9.43</td><td>78.69</td><td>90.57</td><td>84.21</td></tr><tr><td>Whisper-tiny (Bangla) + MSP</td><td>2</td><td>84.33</td><td>23.75</td><td>7.55</td><td>79.46</td><td>92.45</td><td>85.47</td></tr><tr><td>Whisper-tiny (Bangla) + MSP</td><td>4</td><td>83.07</td><td>28.12</td><td>5.66</td><td>76.92</td><td>94.34</td><td>84.75</td></tr></table>

Table 4: Confusion matrices on the BanglaTurn test set (159 endpoint and 160 non-endpoint clips). Rows give the reference label, columns the prediction. <sup>†</sup>Fitted on the test podcast by 10-fold cross-validation.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Reference</td><td colspan="2">Predicted</td></tr><tr><td></td><td>Endpoint Non-endpoint</td></tr><tr><td>Silence</td><td>Endpoint</td><td>16</td><td>143</td></tr><tr><td>threshold†</td><td>Non-endpoint</td><td>19</td><td>141</td></tr><tr><td>Prosody +</td><td>Endpoint</td><td>118</td><td>41</td></tr><tr><td>silence LR†</td><td>Non-endpoint</td><td>80</td><td>80</td></tr><tr><td rowspan="2">Smart-Turn v3</td><td>Endpoint</td><td>77</td><td>82</td></tr><tr><td>Non-endpoint</td><td>16</td><td>144</td></tr><tr><td rowspan="2">Ours</td><td>Endpoint</td><td>147</td><td>12</td></tr><tr><td>Non-endpoint</td><td>38</td><td>122</td></tr></table>

## 6.3. Quantization analysis

INT8 dynamic quantization shrinks the model by roughly 73%, from 148–149 MB to 39–40 MB [30, 31], and cuts end-to-end latency by 6–13%, from 180–191 ms to 165–170 ms. Table 5 gives the figures per configuration. Accuracy mostly survives quantization. The final model loses 1.26 points and the generic encoder 0.31, but the Bangla fine-tuned encoder without MSP loses 4.70 points, from 84.01% to 79.31%.

Table 5: Impact of INT8 quantization on model size, inference latency, and accuracy.
<table><tr><td>Encoder + Components</td><td>Quant.</td><td>Acc (%)</td><td>Size (MB)</td><td>Lat. (ms)</td></tr><tr><td>Whisper-tiny Whisper-tiny</td><td>FP32 INT8</td><td>83.07 82.76</td><td>148.2 39.4</td><td>182.92 164.98</td></tr><tr><td>Whisper-tiny (Bangla) Whisper-tiny (Bangla)</td><td>FP32 INT8</td><td>84.01 79.31</td><td>148.2 39.4</td><td>180.04 169.96</td></tr><tr><td>Whisper-tiny (BN) + MSP</td><td>FP32</td><td>84.33</td><td>149.1</td><td>191.00</td></tr><tr><td>Whisper-tiny (BN) + MSP</td><td>INT8</td><td>83.07</td><td>39.8</td><td>165.95</td></tr></table>

## 7. Discussion

## 7.1. Error trade-off and operating point

The two models fail in opposite directions. Smart-Turn v3 predicts an endpoint for only 93 of the 319 clips and misses 82 of the 159 real ones, so a voice agent built on it would often leave the user waiting in silence. Our model makes the opposite error. At the 0.5 threshold it declares an endpoint on 38 of the 160 clips where the speaker would have carried on, so it would cut in on roughly one such pause in four. Neither balance suits every application. A fast-responding assistant can tolerate an occasional interruption, while cutting the speaker off is costly in dictation or counselling. Because our model outputs a probability, the threshold can be raised to trade missed endpoints for fewer interruptions, or the prediction can be combined with a short silence timeout before the system takes the turn. We report only the 0.5 operating point here, and choosing a threshold on held-out data for a given application is left to future work.

Deployment also calls for checking accuracy after quantization model by model. INT8 costs the final model 1.26 points but costs the Bangla fine-tuned encoder without MSP 4.70 points (Table 5). We have not isolated the cause of that difference.

## 7.2. What silence does and does not tell us

On BanglaTurn, trailing silence points the wrong way. Endpoint clips end with a median of 0.001 s of silence against 0.051 s for non-endpoint clips. That is why a conventional silence threshold scores at chance, while a regression free to learn the reversed direction reaches 63.32%. The cause is how clips are cut. An endpoint clip ends where diarization detects the next speaker, leaving little silence, while a non-endpoint clip ends at a pause that voice activity detection found inside a turn. This is an artefact of corpus construction, not a property of Bangla turn-taking, and it has two consequences. First, silence duration, the cue deployed systems rely on most, cannot solve the task here, so the benchmark tests other cues. Second, a learned model could use the reversed cue as a shortcut. Our model’s 84.33% is far above the 63.32% that silence alone reaches, so the shortcut cannot account for its performance, but it may account for part of it. Balancing trailing silence across the two classes in a future release would remove that doubt.

The prosodic features were even less useful on their own, at 50.78%. The final pitch slope does fall more steeply before endpoints, consistent with the final lowering reported by Ishimoto and Enomoto [16] and by Ward and Vega [14], but on these short, filler-heavy clips the effect is too weak for a linear model to use. Whatever our model has learned, it is not captured by a handful of utterance-final prosodic measurements.

## 7.3. Uncertainty and label quality

With 319 test clips, every figure carries real uncertainty. The gap to Smart-Turn v3 is large enough to survive it, but most ablation differences are not, and ranking the configurations with confidence would need a larger held-out set covering more podcasts. Label quality adds uncertainty that the intervals do not capture. A single annotator checked every label and also selected the test clips, so there is no agreement figure to bound the label error rate and no independent check on the selection. Where a clip’s turn state is genuinely ambiguous, the reference label reflects one person’s judgement, and the measured accuracy is partly agreement with that judgement.

## 7.4. Beyond podcasts

All of BanglaTurn is podcast speech, and the 84.33% figure should not be expected to carry over unchanged to the settings where turn detection matters most commercially. Phone calls are narrowband, typically sampled at 8 kHz and passed through lossy codecs, which strips much of the spectral detail that a Whisper encoder trained on wideband audio relies on. Voice-assistant interactions are dominated by short commands and questions, whose prosody and pausing differ from the long, planned turns of a podcast conversation. Both settings also bring background noise and overlapping speech that edited podcasts largely avoid. We expect accuracy to drop in each of them, and Bangla test sets built from telephone and assistant speech are the most direct next step.

## 8. Conclusions

We have described the first Bangla turn detection dataset, 35,374 samples built with a pipeline that transfers to other languages, and a Whisper encoder-based model that reaches

84.33% accuracy, 15.05 points above the baseline in absolute terms and 21.7% in relative terms. INT8 quantization compresses that model by 73% and cuts its latency by 13% at a cost of 1.26 accuracy points. We hope the corpus and the pipeline together give other low-resource languages a place to start.

## 8.1. Ethical considerations

We release BanglaTurn through gated access by choice. The audio comes from podcasts that are publicly viewable on YouTube, and although that makes the recordings easy to obtain, we treat them as the work of the people who made them rather than as free material, and we did not seek redistribution permission from individual rights holders. Access is therefore granted on request, for research use only. We release clips of 3 to 15 seconds rather than full episodes, without speaker names or links from clips to their source episodes. Voices are nonetheless identifying and the source recordings remain public, so we do not claim the data is anonymous. Any rights holder or speaker can have their material removed from the dataset by contacting the author at the address on the first page.

## 8.2. Limitations

The dataset covers one domain and one register. Podcast conversation is not phone speech, a voice-assistant exchange, a multi-party meeting or a human-robot exchange, so we cannot claim the model carries over to those settings. The labels rest on a single annotator’s check of automatic proposals, with no inter-annotator agreement and no record of how many labels the check changed, so the label error rate is unknown. The same annotator selected the 319 test clips, so the test set is small and its difficulty reflects one person’s judgement. Trailing silence differs systematically between endpoint and non-endpoint clips because of how the clips are cut, and a model may exploit it. All error rates are reported at a single decision threshold. The model itself sees a fixed 8-second window, which rules out longer-range discourse cues, works from audio alone, and inherits whatever prosodic information Whisper’s features happen to encode.

## 9. Generative AI Use Disclosure

Generative AI was used in two ways. As part of the method, Google’s Gemini 2.0 Flash Lite produced first-pass transcriptions and turn-state annotations for the BanglaTurn corpus, as described in Section 3.3; the author checked every resulting label by hand. Separately, generative AI assistance was used to edit and polish the wording of this manuscript. It did not produce a significant part of the manuscript, and the author takes full responsibility for the content of the paper.

## 10. References

[1] H. Sacks, E. A. Schegloff, and G. Jefferson, “A simplest systematics for the organization of turn-taking for conversation,” Language, vol. 50, no. 4, pp. 696–735, 1974.

[2] A. Raux and M. Eskenazi, “Optimizing endpointing thresholds using dialogue features in a spoken dialogue system,” in Proceedings of the SIGDIAL 2009 Conference, 2009, pp. 1–10. [Online]. Available: https://aclanthology.org/W09-3901/

[3] L. Ferrer, E. Shriberg, and A. Stolcke, “Is the speaker done yet? faster and more accurate end-of-utterance detection using prosody,” in Seventh International Conference on Spoken Language Processing (ICSLP), 2002. [Online]. Available: https: //www.isca-speech.org/archive/icslp 2002/ferrer02 icslp.html

[4] G. Skantze, “Towards a general, continuous model of turn-taking in spoken dialogue using lstm recurrent neural networks,” in Proceedings of the 18th Annual SIGdial Meeting on Discourse and Dialogue, 2017, pp. 220–230. [Online]. Available: https://aclanthology.org/W17-5527/

[5] M. Roddy, G. Skantze, and N. Harte, “Investigating speech features for continuous turn-taking prediction using lstms,” in Proceedings of the 19th Annual Conference of the International Speech Communication Association (Interspeech), 2018, pp. 1264–1268. [Online]. Available: https://www.isca-archive.org/ interspeech 2018/roddy18 interspeech.html

[6] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 12 449–12 460. [Online]. Available: https://proceedings.neurips.cc/paper/2020/hash/ 92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html

[7] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021.

[8] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” arXiv preprint arXiv:2212.04356, 2022. [Online]. Available: https://arxiv.org/abs/2212.04356

[9] A. Gravano and J. Hirschberg, “Turn-taking cues in task-oriented dialogue,” Computer Speech and Language, vol. 25, no. 3, pp. 601–634, 2011.

[10] E. Ekstedt and G. Skantze, “Turngpt: A transformer-based language model for turn-taking prediction,” in Findings of the Association for Computational Linguistics: EMNLP 2020, 2020, pp. 2981–2990.

[11] ——, “Voice activity projection: Self-supervised learning of turntaking events,” in Proceedings of Interspeech 2022, 2022, pp. 5190–5194.

[12] K. Inoue, D. Lala, and T. Kawahara, “Latency-aware turn-taking prediction for spoken dialogue systems,” in Proceedings of Interspeech 2022, 2022, pp. 4871–4875.

[13] R. Masumura, T. Tanaka, A. Ando, A. Takashima, K. Yoneyama, and Y. Aono, “Neural end-to-end detection of speech for multiparty conversations,” in Proceedings of Interspeech 2018, 2018, pp. 2439–2443.

[14] N. G. Ward and A. Vega, “Prosodic and timing features of phrasefinal words in spontaneous american english,” Speech Communication, vol. 107, pp. 11–27, 2019.

[15] H. Furukawa, M. Nishida, K. Jokinen, and S. Yamamoto, “A multimodal corpus for modeling turn management in multi-party conversations,” in Proceedings ofOriental COCOSDA, 2011, pp. 142–146.

[16] Y. Ishimoto and M. Enomoto, “Experimental investigation of end-of-utterance perception by final lowering in spontaneous Japanese,” in Proceedings ofOriental COCOSDA, 2016, pp. 205– 209.

[17] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli, “Unsupervised cross-lingual representation learning for speech recognition,” in Proceedings ofInterspeech 2020, 2020, pp. 2426– 2430.

[18] J. Howard and S. Ruder, “Universal language model fine-tuning for text classification,” in Proceedings ofthe 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2018, pp. 328–339. [Online]. Available: https://aclanthology.org/P18-1031/

[19] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are features in deep neural networks?” in Advances in neural information processing systems, vol. 27, 2014, pp. 3320–3328. [Online]. Available: https://arxiv.org/abs/1411.1792

[20] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, N. Calzolari, F. Bechet, P. Blache, K. Choukri,´ C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, and S. Piperidis, Eds. Marseille, France: European Language Resources Association, May 2020, pp. 4218–4222. [Online]. Available: https://aclanthology.org/2020.lrec-1.520/

[21] O. Kjartansson et al., “Crowd-sourced speech corpora for javanese, sundanese, sinhala, nepali, and bangladeshi bengali,” in Proceedings of the 6th International Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU), 2018, pp. 52–55.

[22] T. Islam, M. S. Arafat et al., “Banglabert: Language model pretraining and benchmarks for low-resource language understanding evaluation in bangla,” in Findings ofthe Associationfor Computational Linguistics: NAACL 2022, 2022, pp. 1318–1327.

[23] B. Das, S. Mandal, and P. Mitra, “Bengali speech corpus for continuous automatic speech recognition system,” in Proceedings of Oriental COCOSDA, 2011, pp. 51–55.

[24] P. Chowdhury, S. Nath, and U. Sharma, “A prosodically annotated Bengali and Assamese audiobook corpus for sentence boundary detection,” in Proceedings ofOriental COCOSDA, 2025, pp. 1–6.

[25] H. Bredin, R. Yin, J. M. Coria, G. Gelly, P. Korshunov, M. Lavechin, D. Fustes, H. Titeux, W. Bouaziz, and M.-P. Gill, “pyannote.audio: neural building blocks for speaker diarization,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 7124–7128.

[26] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, “X-vectors: Robust dnn embeddings for speaker recognition,” in 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2018, pp. 5329–5333.

[27] F. Gilardi, M. Alizadeh, and M. Kubli, “Chatgpt outperforms crowd-workers for text-annotation tasks,” arXiv preprint arXiv:2303.15056, 2023. [Online]. Available: https://arxiv.org/abs/2303.15056

[28] M. Mauch and S. Dixon, “pYIN: A fundamental frequency estimator using probabilistic threshold distributions,” in Proceedings of IEEE ICASSP, 2014, pp. 659–663.

[29] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019. [Online]. Available: https: //openreview.net/forum?id=Bkg6RiCqY7

[30] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2704–2713.

[31] R. Krishnamoorthi, “Quantizing deep convolutional networks for efficient inference: A whitepaper,” arXiv preprint arXiv:1806.08342, 2018. [Online]. Available: https: //arxiv.org/abs/1806.08342