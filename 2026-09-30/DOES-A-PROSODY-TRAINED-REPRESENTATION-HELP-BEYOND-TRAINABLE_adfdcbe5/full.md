# DOES A PROSODY-TRAINED REPRESENTATION HELP BEYOND TRAINABLE FUSION? A PARAMETER-MATCHED STUDY WITH FROZEN HUBERT

Ki Woong Moon<sup>1⋆</sup> Daniel Brenner<sup>2</sup>

<sup>1</sup>Department of Linguistics, University of Arizona, Tucson, AZ 85721 <sup>2</sup>Speak, 360 Spear St., 4F, San Francisco, CA 94105

## ABSTRACT

Explicit prosodic cues may help automatic speech recognition (ASR) of spontaneous speech, but auxiliary representations typically require additional trainable components, making it unclear whether gains come from the auxiliary information or the fusion mechanism. We address this using a frozen HuBERT backbone and a 64-dimensional representation trained to predict log F<sub>0</sub>, voicing, ∆ log F<sub>0</sub>, log energy, and spectral tilt. We compare a frozen-backbone recognizer (Baseline), trainable fusion with zero auxiliary input (Null), and the same fusion supplied with the learned representation (Learned). Across Buckeye, Switchboard, and AMI IHM, Null reduces WER by 0.71–1.45 points over Baseline, whereas Learned differs from Null by +0.07, −0.09, and +0.00 points, with no significant differences. However, removing or mismatching the representation at inference increases Learned WER. Thus, Learned depends on the representation yet shows no measurable incremental WER benefit over the parameter-matched control.

Index Terms— spontaneous speech, prosody, speech recognition, self-supervised learning

## 1. INTRODUCTION

Spontaneous speech is acoustically variable and ambiguous, as segments may be shortened, weakened, deleted, or heavily coarticulated. For example, “I don’t know” can be realized as [˜a ˜on˜o], with substantial deletion and coarticulation. Johnson [1] reported that such reduction in spontaneous speech is systematic, with roughly a quarter of words in conversational American English containing at least one segment deletion. The degree of reduction also varies with linguistic predictability and prosodic prominence, with more predictable materials tending to be shorter and less phonetically prominent [2, 3]. Acoustic measures associated with prominence include F0, loudness, duration, and spectral balance [3–5]. Such cues may provide useful information for recognizing acoustically ambiguous spontaneous speech, raising the question of how effectively modern speech representations capture and exploit them.

Self-supervised learning (SSL) models such as HuBERT [6] learn useful representations directly from the audio signal. SSL representations have also been shown to encode prosodically relevant information [7, 8]. However, recoverability does not establish that a downstream ASR system effectively exploits this information during recognition.

Yet evaluating the contribution of an auxiliary representation introduces an attribution problem. Incorporating auxiliary information typically requires additional trainable components, such as projections, adapters, or fusion modules. A comparison between a frozen

SSL baseline and an auxiliary-conditioned model therefore changes two factors simultaneously: the information supplied to the recognizer and the trainable mechanism used to incorporate it. Any recognition improvement in such a comparison cannot be attributed uniquely to the auxiliary information. This confound is particularly relevant when the backbone is frozen, because the added trainable mechanism provides an additional pathway for transforming otherwise fixed SSL representations.

To separate the effects of auxiliary information from those of the trainable fusion mechanism used to incorporate it, we use a twophase system. We first train a compact prosody encoder to predict log F<sub>0</sub>, voicing, ∆ log F<sub>0</sub>, log energy, and spectral tilt, yielding a 64-dimensional prosody-trained representation. In the ASR system, we keep this representation frozen and use it to condition trainable fusion modules applied to the saved transformer-layer states of a frozen HuBERT. Importantly, we compare the Learned condition that receives the learned auxiliary representation with the Null condition that has an identical fusion architecture but receives a zero-valued auxiliary input. Because the two conditions have the same architecture and number of trainable parameters, their comparison tests the incremental contribution of the auxiliary representation while holding the fusion architecture and trainable parameter count fixed. The effect of adding the trainable fusion pathway is evaluated by comparing the Null with Baseline, which omits the fusion modules.

With this design, we ask whether adding the trainable fusion pathway improves recognition, whether the learned representation provides additional benefit beyond a parameter-matched zero-input pathway, and whether the trained Learned model uses the representation at inference. Across three corpora, the trainable pathway consistently improves recognition, whereas the learned representation provides little additional WER improvement. We use inference-time interventions to test the last question directly.

## 2. RELATED WORK

Phonetic realization in spontaneous speech varies systematically with lexical predictability and prosodic prominence [2, 3, 9, 10]. This motivates interest in whether pretrained speech representations encode information relevant to such variation. Layerwise analyses show that acoustic and linguistic information is distributed non-uniformly across SSL encoder layers [11]. Prosodically relevant information is also recoverable from these representations: SSL models support prosody reconstruction and future-prosody prediction [7] and distinguish strong from weak prosodic boundaries [8]. However, recoverability does not imply that a downstream ASR system effectively exploits the same information during recognition.

![](images/e1e65b32cfd658d2919f75353cd7d6e9a8fc2a67e838dd0ddefde4003445f589.jpg)  
Fig. 1. Two-phase system. Phase 1 trains a 64-D representation using five acoustic-prosodic targets. In Phase 2, frozen HuBERT and the frozen Phase 1 encoder produce representations independently. Twelve post-hoc fusion modules modify h<sub>1</sub>, . . . , h<sub>12</sub>; h<sub>0</sub> bypasses fusion, and modified states are not fed back into HuBERT. Baseline omits fusion, Null uses the same modules with $p = 0$ , and Learned uses the prosody-trained p.

Recent work has explored explicit prosodic supervision in pretrained speech recognizers. Sasu and Schluter [12], for example, jointly train pitch-accent detection and ASR with wav2vec 2.0 [13] and report improved recognition performance. Their comparison contrasts ASR-only training with a joint auxiliary objective, leaving open whether the improvement is specific to the prosodic supervision or reflects the broader effect of introducing the auxiliary training objective. Our study addresses a related but distinct attribution question by holding the trainable fusion mechanism fixed while varying whether it receives an informative auxiliary representation.

The difficulty of attributing downstream performance to a frozen representation rather than to its trainable head is well established in probing. Hewitt and Liang [14] introduced control tasks to contextualize probe accuracy, and Zaiem et al. [15] showed that changing the probing head can alter speech SSL model rankings. Related ASR systems also inject auxiliary cues through trainable pathways, including FiLM conditioning on enhanced speech [16]. Such systems couple the auxiliary information with a trainable mechanism for incorporating it. Our Learned–Null comparison instead holds the fusion pathway fixed while varying its auxiliary input; the interventions separately test whether the trained system uses the representation.

## 3. METHODS

## 3.1. Overview

The system tests the contributions of the prosody-trained representation and the trainable fusion mechanism in two phases. Phase 1 trains a compact encoder to produce a frame-level representation supervised by five acoustic-prosodic targets. Phase 2 uses this representation to condition trainable fusion modules applied independently to the hidden-state output of each HuBERT transformer layer. Because each modified state is used only for downstream layer aggregation and is not passed to the next HuBERT layer, we call this operation post-hoc layerwise fusion. Figure 1 summarizes the system.

## 3.2. Data

Phase 1 uses CASPER [17]. We resample recordings to 16 kHz and divide them into non-overlapping segments of at most 15 s. We use a fixed 80/20 segment-level training/validation split. CASPER is used only to train and select the Phase 1 encoder and is disjoint from the Phase 2 ASR corpora.

Phase 2 uses three English corpora spanning conversational and meeting settings with different recording conditions. Buckeye [18] contains close-microphone spontaneous speech, Switchboard [19] contains two-party telephone conversations, and AMI IHM [20] contains multi-party meeting speech recorded with individual headset microphones. Buckeye uses 30/5/5 speaker-disjoint train/validation/test speakers, corresponding to 4,915/797/925 segments and 26.86/4.35/5.06 h. For Switchboard, we follow the preprocessing and train/validation/test splits of Ho et al. [21] (185,402/20,601/51,501 utterances) and remove annotations enclosed in angle or square brackets. For AMI IHM, the train/validation/test partitions contain 75,174/9,428/8,514 usable utterances.

## 3.3. Phase 1: Prosody-trained encoder

The Phase 1 encoder maps 16 kHz waveforms to a 64-dimensional representation at a 20 ms frame interval. Its frontend computes an 80-bin log-mel spectrogram using a 50 ms window, 20 ms hop, and non-centered framing. The log-mel features are projected to 128 dimensions, then passed through four depthwise-separable convolutional blocks [22], a bidirectional GRU [23], and finally projected to 64 dimensions.

Five prediction heads predict log F<sub>0</sub>, voicing, ∆ log F<sub>0</sub>, log energy, and spectral tilt. $F _ { 0 }$ and periodicity are estimated using CREPEtiny [24] and resampled by timestamp to the encoder frame grid, with $F _ { 0 }$ restricted to 50–500 Hz. A frame is treated as voiced when its energy exceeds the utterance-specific 20th percentile, periodicity is at least .01, and $F _ { 0 }$ lies within the valid range. In the Phase 1 training set, the periodicity criterion excluded 22.1% of frames that passed the energy criterion. $\Delta$ log $F _ { 0 }$ is defined only across consecutive voiced frames, and continuous targets are standardized using statistics estimated from eligible frames in the Phase 1 training partition. Targets undefined at a given frame are masked from the corresponding loss rather than imputed.

Mean-squared error is used for the continuous targets and binary cross-entropy for voicing. The training objective is

$$
L = L _ { \log F _ { 0 } } + L _ { \mathrm { v o i } } + 0 . 5 L _ { \Delta \log F _ { 0 } } + 0 . 5 L _ { \mathrm { e n g } } + 0 . 5 L _ { \mathrm { t i l t } } ,\tag{1}
$$

where each term is averaged over eligible, non-padded frames. Phase 1 uses a batch size of 32, learning rate of $1 \dot { 0 } ^ { - 3 }$ , and early stopping (patience 10) for up to 50 epochs, selecting the checkpoint with the lowest masked validation loss. Only the resulting 64-D hidden representation is retained for Phase 2.

## 3.4. Phase 2: Post-hoc HuBERT fusion

We use HuBERT-base [6] with the SSL backbone frozen throughout Phase 2. HuBERT produces a pre-transformer hidden state $h _ { 0 }$ followed by twelve transformer-layer outputs $h _ { 1 } , \ldots , h _ { 1 2 }$ , each with 768 dimensions. For the fusion conditions, the frozen Phase 1 encoder is evaluated independently, and the non-padded representation for each utterance is linearly resampled to that utterance’s HuBERT frame length before fusion.

For each transformer layer $\ell \geq 1$ , let $h _ { \ell }$ denote the frozen Hu-BERT state and $p \in \mathbb { R } ^ { B \times T \times 6 4 }$ the aligned auxiliary representation. The fusion module applies FiLM-style conditioning [25] followed by a gated residual:

$$
\bar { p } = \mathrm { L a y e r N o r m } ( p ) ,\tag{2}
$$

$$
( \gamma , \beta ) = \operatorname { L i n e a r } ( \bar { p } ) ,\tag{3}
$$

$$
c = \mathrm { L a y e r N o r m } ( h _ { \ell } + \gamma \odot h _ { \ell } + \beta ) ,\tag{4}
$$

$$
g = \sigma ( \mathrm { L i n e a r } ( [ h _ { \ell } ; p ] ) ) ,\tag{5}
$$

$$
h _ { \mathrm { o u t } , \ell } = h _ { \ell } + \operatorname { t a n h } ( s _ { \ell } ) g \odot ( c - h _ { \ell } ) ,\tag{6}
$$

where $\gamma , \beta \in \mathbb { R } ^ { B \times T \times 7 6 8 } , g \in \mathbb { R } ^ { B \times T \times 1 }$ is broadcast along the feature dimension, ⊙ denotes element-wise multiplication, and $[ h _ { \ell } ; p ]$ denotes concatenation along the feature dimension. Each fusion module’s learnable residual scale $s \ell$ is initialized to zero, making the module an exact identity mapping at initialization. State $h _ { 0 }$ bypasses the fusion modules. A learned softmax-weighted sum over all thirteen states is then projected from 768 to 512 dimensions, passed through a two-layer bidirectional LSTM [26], and fed into a CTC output layer [27].

## 3.5. Experimental conditions and training

We compare three conditions. Baseline uses frozen HuBERT without the post-hoc fusion modules. Null includes all twelve fusion modules, with the auxiliary input set to zero throughout training and inference. Learned uses the identical fusion architecture with the learned auxiliary representation as input. Null and Learned therefore have identical architectures and trainable parameter counts (12.16M), whereas Baseline has 10.93M trainable parameters. Although matched in trainable parameter count, Null cannot exploit utterance-varying auxiliary conditioning because its auxiliary input is fixed to zero.

Only the layer-aggregation weights, post-hoc fusion modules when present, 768-to-512 projection, BiLSTM, and CTC head are optimized. HuBERT remains frozen and in evaluation mode through out training; the Phase 1 encoder is also frozen and kept in evaluation mode when present. Training uses AdamW [28] with a learning rate of $1 0 ^ { - 4 }$ , weight decay of 0.01, and $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 8 )$ . We use a batch size of 8 with gradient accumulation over 4 steps (effective batch size 32), mixed-precision training, and gradient clipping at 1.0. Early stopping is based on validation WER with a patience of 10 epochs. We use greedy CTC decoding for evaluation. We use three seeds per corpus and condition.

## 3.6. Statistical analysis

Our primary comparison is Learned–Null, which tests whether supplying the learned auxiliary representation improves recognition relative to the same trainable fusion mechanism with its auxiliary input fixed to zero. Null–Baseline is a secondary comparison assessing the effect of adding the trainable fusion pathway.

Statistical inference is based on paired test-set predictions and a hierarchical Poisson bootstrap with 100,000 draws. In each bootstrap draw, training seeds are resampled while preserving the pairing between conditions. For Buckeye and AMI IHM, speakers are also resampled before applying Poisson weights to utterances within each speaker. Switchboard lacks usable speaker labels in the stored predictions and therefore uses seed-utterance resampling. We report absolute WER percentage-point differences, 95% confidence intervals, and two-sided p-values. Holm correction is applied separately across the three corpus-level Learned–Null tests and the three Null– Baseline tests.

## 3.7. Inference-time diagnostics

To determine whether the trained Learned model makes functional use of its auxiliary representation, each seed-specific Learned checkpoint is evaluated without retraining under four input conditions: the original representation (TRUE), an all-zero representation (ZERO), a within-utterance time-shuffled representation (TIME), and a representation obtained from a different utterance (UTT). For UTT, donor utterances are assigned deterministically by a circular shift of half the test set, without constraining speaker identity, and the donor representation is resampled to the recipient utterance’s HuBERT frame length.

These interventions test the dependence of an already-trained Learned model on its auxiliary representation and must be distinguished from the Null training condition. UTT substitutes a representation from another utterance, TIME disrupts its temporal alignment, and ZERO removes the learned input entirely. Because Learned is never trained with an all-zero auxiliary sequence, ZERO is an outof-distribution sanity check rather than a matched control. Null, by contrast, is optimized from initialization with zero auxiliary input. Exploratory intervention contrasts use the same paired hierarchical Poisson bootstrap with 100,000 draws, with Holm correction applied jointly across the nine corpus-by-intervention contrasts.

Finally, we quantify each fusion module’s residual magnitude $r _ { \ell , t } = \operatorname { t a n h } ( s _ { \ell } ) g _ { \ell , t } \odot ( c _ { \ell , t } - h _ { \ell , t } )$ relative to the corresponding frozen HuBERT state:

$$
\rho _ { \ell } = \mathbb { E } _ { t \in \mathrm { v a l i d } } \left[ { \frac { \| r _ { \ell , t } \| _ { 2 } } { \| h _ { \ell , t } \| _ { 2 } } } \right] .\tag{7}
$$

We report these layerwise quantities descriptively rather than performing post-hoc significance tests across individual layers.

## 4. RESULTS

## 4.1. Recognition performance

To assess cross-corpus agreement of the frozen Phase 1 encoder, we evaluated its target predictions on the CASPER validation set and the three downstream validation sets. On CASPER, correlations for log $F _ { 0 } ,$ ∆ log $F _ { 0 }$ , log energy, and spectral tilt were .889, .703, .998, and .998, respectively, and voicing F1 was .890. Across Buckeye, Switchboard, and AMI IHM, the corresponding correlations were .804/.793/.944 for log $F _ { 0 } ,$ .694/.786/.830 for ∆ log $F _ { 0 } ,$ .996/.993/.988 for log energy, and .997/.995/.996 for spectral tilt; voicing F1 was .882/.864/.880. These diagnostics measure agreement with automatically extracted supervision targets rather than independent ground-truth prosody.

Table 1. Test WER $( \% ,$ mean ± SD over three training seeds) with 95% hierarchical bootstrap CIs for absolute WER differences. Negative values favor the first condition; bold indicates the lowest unrounded mean WER.
<table><tr><td>Corpus</td><td>Baseline</td><td>Null</td><td>Learned</td><td>Learned—Null</td><td>Null-Baseline</td></tr><tr><td>Buckeye</td><td> $3 7 . 2 7 \pm 0 . 1 0$ </td><td> $3 5 . 8 2 \pm 0 . 1 3$ </td><td> $3 5 . 8 9 \pm 0 . 1 1$ </td><td> $+ 0 . 0 7 \ [ - 0 . 2 2 , + 0 . 3 4 ]$ </td><td> $- 1 . 4 5 \ [ - 1 . 9 3 , - 0 . 9 4 ]$ </td></tr><tr><td>Switchboard</td><td> $3 1 . 9 5 \pm 0 . 0 9$ </td><td> $3 1 . 2 4 \pm 0 . 0 4$ </td><td> $3 1 . 1 5 \pm 0 . 0 7$ </td><td> $- 0 . 0 9 \left[ - 0 . 2 1 , + 0 . 0 3 \right]$ </td><td> $- 0 . 7 1 \ [ - 0 . 8 5 , - 0 . 5 8 ]$ </td></tr><tr><td>AMI IHM</td><td> $3 9 . 1 1 \pm 0 . 2 2$ </td><td> $3 8 . 2 3 \pm 0 . 2 2$ </td><td> $3 8 . 2 3 \pm 0 . 2 0$ </td><td> $+ 0 . 0 0 [ - 0 . 2 7 , + 0 . 2 8 ]$ </td><td> $- 0 . 8 9 \ [ - 1 . 2 4 , - 0 . 5 4 ]$ </td></tr></table>

Table 2. WER increase (percentage points) after intervening on the auxiliary representation of the trained Learned model. Positive values indicate degradation relative to the unmodified representation.
<table><tr><td>Corpus</td><td>Time shuffle</td><td>Utt. subst.</td><td>Zero</td></tr><tr><td>Buckeye</td><td>+0.17</td><td>+0.42</td><td> $+ 4 . 8 5$ </td></tr><tr><td>Switchboard</td><td>+0.30</td><td>+1.06</td><td> $+ 1 6 . 9 3$ </td></tr><tr><td>AMI IHM</td><td>+0.35</td><td>+1.01</td><td> $+ 1 2 . 7 7$ </td></tr></table>

Table 1 reports test WER averaged over three training seeds. Learned differs from Null by +0.069 WER percentage points on Buckeye, −0.090 on Switchboard, and +0.004 on AMI IHM. All three confidence intervals include zero, and none is significant after Holm correction (all $p _ { \mathrm { H o l m } } ~ \geq ~ . 4 0 0 \}$ . Moreover, the 95% confidence intervals are compatible with at most a few tenths of a percentage point of improvement from the learned representation under the present architecture.

In contrast, Null consistently outperforms Baseline by 0.71–1.45 WER points. All confidence intervals exclude zero $( p _ { \mathrm { H o l m } } < . 0 0 1 )$ with the same direction for every seed. The trainable fusion pathway therefore yields a robust improvement, whereas the learned representation adds little further benefit.

## 4.2. Functional use of the auxiliary representation

The near-zero Learned–Null difference does not imply that Learned ignores its auxiliary input. We therefore intervene on the input at inference without retraining; Table 2 reports the WER increase relative to the unmodified model.

Utterance substitution increased WER by 0.42–1.06 points and was significant on all three corpora (all $p _ { \mathrm { H o l m } } \leq . 0 0 6 1 )$ , indicating use of utterance-specific auxiliary information. Temporal shuffling produced smaller increases of 0.17–0.35 points, significant for Switchboard and AMI IHM but not Buckeye, suggesting weaker sensitivity to temporal ordering. Replacing the representation with zeros produced larger degradations of 4.85–16.93 points; because this input is out of distribution for Learned, we treat it as a sanity check rather than direct evidence of incremental utility.

## 4.3. Fusion behavior

Both Null and Learned modify frozen HuBERT states, especially in upper layers (Figure 2). Mean relative residual magnitude is larger for

![](images/3463caf0b9899bc270845aa278a7827a05ac201f42a1aa46a905610412f57f7c.jpg)  
Fig. 2. Layerwise relative residual magnitude $\rho _ { \ell } ,$ averaged over three seeds. Larger values indicate greater modification of frozen HuBERT states.

Null than Learned on every corpus: .351 versus .249 on Buckeye, .825 versus .526 on Switchboard, and 1.091 versus .672 on AMI IHM. The nonzero residuals show that Null is an active control, while Learned shows its strongest modifications around layers 9–11.

## 5. DISCUSSION

The trainable fusion mechanism improves the frozen HuBERT recognizer, whereas the prosody-trained representation yields little additional WER reduction. Under this architecture, most of the observed gain is associated with the trainable fusion pathway rather than the auxiliary representation.

Learned nevertheless depends on its auxiliary input: utterance substitution degrades recognition even though Null achieves comparable WER without an utterance-varying signal. One possible explanation is redundancy: frozen HuBERT states may already encode information correlated with the supervised acoustic-prosodic cues, leaving little incremental information for the auxiliary representation.

These conclusions are limited to one SSL backbone, one supervision set that omits duration, a CTC recognizer, and a frozen post-hoc design. Full fine-tuning, propagating fused states through HuBERT, or localized error analysis may yield different effects. Parametermatched controls remain important for separating gains due to auxiliary information from those due to the trainable mechanism used to incorporate it.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This research study was conducted retrospectively using previously collected human speech data from the CASPER [17], Buckeye [18], Switchboard [19], and AMI [20] corpora. No funding was received for conducting this study. The authors have no relevant financial or nonfinancial interests to disclose.

## 7. REFERENCES

[1] Keith Johnson, “Massive reduction in conversational American English,” in Spontaneous Speech: Data and Analysis, Kiyoko Yoneyama and Kikuo Maekawa, Eds., pp. 29–54. National Institute for Japanese Language, Tokyo, Japan, 2004.

[2] Matthew Aylett and Alice Turk, “Language redundancy predicts syllabic duration and the spectral characteristics of vocalic syllable nuclei,” JASA, vol. 119, no. 5, pp. 3048–3058, 2006.

[3] Rory Turnbull, “The role of predictability in intonational variability,” Lang. Speech, vol. 60, no. 1, pp. 123–153, 2017.

[4] Agaath M. C. Sluijter and Vincent J. van Heuven, “Spectral balance as an acoustic correlate of linguistic stress,” JASA, vol. 100, no. 4, pp. 2471–2485, 1996.

[5] Greg Kochanski, Esther Grabe, John Coleman, and Burton Rosner, “Loudness predicts prominence: Fundamental frequency lends little,” JASA, vol. 118, no. 2, pp. 1038–1054, 2005.

[6] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 29, pp. 3451–3460, 2021.

[7] Guan-Ting Lin, Chi-Luen Feng, Wei-Ping Huang, et al., “On the utility of self-supervised models for prosody-related tasks,” in SLT, 2022.

[8] Maureen de Seyssel, Marvin Lavechin, Hadrien Titeux, et al., “ProsAudit: A prosodic benchmark for self-supervised speech models,” in Interspeech, 2023, pp. 2963–2967.

[9] Daniel Jurafsky, Alan Bell, Michelle Gregory, and William D. Raymond, “The effect of language model probability on pronunciation reduction,” in ICASSP, 2001.

[10] Alan Bell, Jason M. Brenier, Michelle Gregory, Cynthia Girand, and Dan Jurafsky, “Predictability effects on durations of content and function words in conversational English,” J. Mem. Lang., vol. 60, no. 1, pp. 92–111, 2009.

[11] Ankita Pasad, Bowen Shi, and Karen Livescu, “Comparative layer-wise analysis of self-supervised speech models,” in ICASSP, 2023.

[12] David Sasu and Natalie Schluter, “Pitch accent detection improves pretrained automatic speech recognition,” in Interspeech, 2025.

[13] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in NeurIPS, 2020.

[14] John Hewitt and Percy Liang, “Designing and interpreting probes with control tasks,” in EMNLP-IJCNLP, 2019, pp. 2733– 2743.

[15] Salah Zaiem, Youcef Kemiche, Titouan Parcollet, Slim Essid, and Mirco Ravanelli, “Speech self-supervised representations benchmarking: A case for larger probing heads,” Comput. Speech Lang., vol. 89, pp. 101695, 2025.

[16] Da-Hee Yang and Joon-Hyuk Chang, “FiLM conditioning with enhanced feature to the transformer-based end-to-end noisy speech recognition,” in Interspeech, 2022, pp. 4098–4102.

[17] Cihan Xiao, Ruixing Liang, Xiangyu Zhang, Mehmet Emre Tiryaki, Veronica Bae, Lavanya Shankar, Rong Yang, Ethan Poon, Emmanuel Dupoux, Sanjeev Khudanpur, et al., “CASPER: A large scale spontaneous speech dataset,” in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2025, pp. 1–7.

[18] Mark A. Pitt, Keith Johnson, Elizabeth Hume, Scott F. Kiesling, and William D. Raymond, “The Buckeye corpus of conversational speech: labeling conventions and a test of transcriber reliability,” Speech Commun., vol. 45, pp. 89–95, 2005.

[19] John J. Godfrey, Edward C. Holliman, and Jane McDaniel, “SWITCHBOARD: Telephone speech corpus for research and development,” in ICASSP, 1992.

[20] I. McCowan, J. Carletta, W. Kraaij, S. Ashby, S. Bourban, M. Flynn, M. Guillemot, T. Hain, J. Kadlec, V. Karaiskos, M. Kronenthal, G. Lathoud, M. Lincoln, A. Lisowska, W. Post, Dennis Reidsma, and P. Wellner, “The AMI meeting corpus,” in Measuring Behavior, 2005.

[21] Phuoc Hoang Ho, Dragos Alexandru Balan, Dirk K. J. Heylen,˘ and Khiet P. Truong, “Enhancing Transcripts of Open-Source Automatic Speech Recognition Models Through Fine-Tuning with Laughter and Speech-Laugh,” in Interspeech, 2025, pp. 4513–4517.

[22] Andrew G. Howard, Menglong Zhu, Bo Chen, et al., “MobileNets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.

[23] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, et al.,¨ “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in EMNLP, 2014.

[24] Jong Wook Kim, Justin Salamon, Peter Li, and Juan Pablo Bello, “CREPE: A convolutional representation for pitch estimation,” in ICASSP, 2018.

[25] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville, “FiLM: Visual reasoning with a general conditioning layer,” in AAAI, 2018.

[26] Alex Graves, Santiago Fernandez, and J´ urgen Schmidhuber,¨ “Bidirectional LSTM networks for improved phoneme classification and recognition,” in ICANN, 2005.

[27] Alex Graves, Santiago Fernandez, Faustino Gomez, and J´ urgen¨ Schmidhuber, “Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,” in ICML, 2006.

[28] Ilya Loshchilov and Frank Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.