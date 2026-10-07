# LOUD AND CLEAR: DYNAMIC ACTIVATION STEERING FOR IMPROVING SPEECH INTELLIGIBILITY IN NOISY ENVIRONMENTS

Seymanur Akti<sup>1,2</sup>, Alexander Waibel<sup>1,3</sup>

<sup>1</sup>Karlsruhe Institute of Technology (KIT), Karlsruhe, Germany <sup>2</sup>KIT Campus Transfer (KCT), Karlsruhe, Germany <sup>3</sup>Carnegie Mellon University (CMU), Pittsburgh, USA

## ABSTRACT

Speech becomes less intelligible in noisy environments, and humans naturally adapt their voice to compensate. Inspired by this behavior, we investigate whether a text-to-speech (TTS) model can be guided to produce more intelligible speech using activation steering, without retraining. We focus on two characteristics of the Lombard effect: increased vocal effort and hyper-articulation. We introduce a prompt-relative steering mechanism that prevents steering effects from accumulating during generation while allowing their strength to be adjusted dynamically. Across seen and unseen speakers and multiple languages, our method produces systematic changes in Lombard-related acoustic features, preserves speaker similarity (89–95%), and reduces WER under background noise by 7–22% at 1 dB SNR. These results show that pretrained TTS models can be dynamically controlled to generate more intelligible speech without retraining.

Index Terms— text-to-speech, activation steering, speech intelligibility, Lombard effect

## 1. INTRODUCTION

Humans naturally adapt their speaking style to challenging acoustic conditions such as noise, a phenomenon known as the Lombard effect. These adaptations improve speech intelligibility through increased vocal effort, higher pitch, slower speaking rate, flattened spectral tilt, and clearer articulation [1, 2]. Such articulatory changes have also been shown to be relevant to robust speech recognition [3]. Reproducing such adaptations in synthesized speech could benefit speech synthesis in noisy environments and applications requiring enhanced intelligibility, such as second-language (L2) learning and assistive communication.

Early near-end listening enhancement (NELE) and Lombard synthesis relied on signal-processing techniques, including spectral modification [4, 5, 6, 7], dynamic-range compression [8, 9], and prosodic manipulation [10], often introducing acoustic distortions. Recent neural TTS approaches instead use transfer learning [11, 12], fine-tuning [13], spectraltilt augmentation [14], explicit pitch/energy modeling [15],

ASR-guided loudness control [16], and latent-space interpolation [17]. These methods primarily target global acoustic properties such as vocal effort, pitch, and spectral characteristics, while hyper-articulation remains less explored. Articulation has been modeled using HMMs [18] and Global Style Tokens (GSTs) [19], but vocal effort and hyper-articulation have generally been treated as separate control dimensions. A recent study proposed joint continuous control of both attributes at the utterance and word levels [20].

In real-world settings, the required degree of intelligibility enhancement may vary with acoustic conditions or listener requirements. Human speakers continuously adapt their speak ing style to environmental feedback, motivating TTS systems that can dynamically adjust intelligibility-related attributes, potentially within an utterance. Prior work has explored incremental speech transformation [21] and ASR-based feedback for dynamic loudness control [16], but does not provide joint, continuous, within-utterance control of vocal effort and hyper-articulation. Moreover, existing Lombard TTS methods generally rely on training, fine-tuning, or fixed signallevel transformations, limiting dynamic inference-time control of multiple attributes across speakers and languages.

We address these limitations with inference-time activation steering, which modifies hidden representations along attribute-specific directions without changing model parameters [22]. Recent work has applied activation steering to TTS for emotion control [23, 24, 25] and accent neutralization [26], but repeatedly applying a fixed intervention during autoregressive generation can accumulate across tokens and distort the output [27]. To overcome this limitation, we introduce Prompt-Relative Activation Steering, which compares each generated token’s activation with its baseline activation from the reference prompt and applies only the residual steering needed to reach the target. This prevents steering accumulation while enabling stable, continuous control. Using paired speech data, we extract activation directions for vocal effort and hyper-articulation from Qwen3-TTS [28] and evaluate the approach across multiple speakers and languages. Audio samples are shared at demo page. <sup>1</sup>

Our contributions are threefold: (1) joint and continuous control of vocal effort and hyper-articulation through training-free activation steering, demonstrated across speakers and languages, (2) a prompt-relative mechanism that prevents steering accumulation during autoregressive generation, and (3) dynamic within-utterance control, allowing steering strength to be modified during synthesis.

## 2. METHOD

Following prior work on activation steering [25, 24, 26], we extract attribute-specific steering directions from paired speech conditions and apply them to hidden representations during inference. We use Qwen3-TTS and intervene on the residual outputs of its LLM-based autoregressive backbone. We consider two intelligibility-related attributes: vocal effort and hyper-articulation. Their steering directions are extracted independently and combined for joint control. The control is applied only to generated tokens to enable dynamic control in streaming mode.

## 2.1. Steering Vector Extraction

We first synthesize paired speech samples using the voicecloning generation method of Qwen3-TTS. Each pair is produced by the same speaker with identical linguistic content but different speaking styles, isolating the target attribute.

Let $\mathbf { h } _ { l , t } ^ { ( x , i ) }$ denote the hidden representation at layer l and generated-token position t for utterance i with speaking style x. We obtain an utterance-level representation by averaging over generated tokens:

$$
\mathbf { h } _ { l , i } ^ { ( x ) } = \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } \mathbf { h } _ { l , t } ^ { ( x ) }\tag{1}
$$

where $T _ { i }$ is the number of generated tokens for utterance i. The steering direction for attribute k at layer l is computed as the average difference between the target and reference speaking styles across $N _ { k }$ paired samples:

$$
\mathbf { v } _ { l } ^ { k } = \frac { 1 } { N _ { k } } \sum _ { i = 1 } ^ { N _ { k } } ( \mathbf { h } _ { l , i } ^ { ( t a r g e t ) } - \mathbf { h } _ { l , i } ^ { ( r e f e r e n c e ) } )\tag{2}
$$

where $k \in \{ E , A \}$ denotes vocal effort and hyper-articulation, respectively. For vocal effort, the target and reference conditions correspond to loud and regular speech from EARS dataset [29], while for hyper-articulation they correspond to enunciated and default speech from Expresso dataset [30]. The directions are extracted only from generated-token positions rather than input prompt tokens, so that they characterize the representations used during speech generation and can be applied directly to the autoregressive generation process.

## 2.2. Joint Control of Vocal Effort and Articulation

The two attributes are controlled jointly by combining their layer-wise steering directions. Let $\mathbf { v } _ { l } ^ { \dot { E } }$ and $\mathbf { v } _ { l } ^ { A }$ denote the

vocal-effort and articulation directions, respectively and $\lambda _ { E }$ and $\lambda _ { A }$ are the user-defined control coefficients:

$$
\mathbf { v } _ { l } ^ { \mathrm { j o i n t } } = \lambda _ { E } \mathbf { v } _ { l } ^ { E } + \lambda _ { A } \mathbf { v } _ { l } ^ { A }\tag{3}
$$

This formulation provides continuous control over the relative contribution of vocal effort and articulation. In particular, setting one coefficient to zero produces single-attribute steering, while non-zero values for both coefficients provide joint continuous control.

## 2.3. Prompt-Relative Activation Steering

Adding a fixed steering vector to every generated token can lead to cumulative changes in the hidden representations, as each token already inherits part of the steering effect through causal attention. Consequently, the generated audio gets distorted as the sequence becomes longer. To mitigate this effect, we perform prompt-relative steering.

During prompt processing, we compute the mean projection of the hidden representations corresponding to the reference audio segment onto the normalized joint steering direction. This projection $p _ { l } ^ { b a s e l i n e }$ serves as the prompt-specific baseline for subsequent generation. The target projection $p _ { l } ^ { * }$ is then defined as the baseline projection plus the magnitude of the weighted joint steering vector at each layer.

$$
p _ { l } ^ { b a s e l i n e } = \frac { 1 } { T _ { p } } \sum _ { t = 1 } ^ { T _ { p } } \mathbf { h } _ { l , t } ^ { \top } \frac { \mathbf { v } _ { l } ^ { \mathrm { j o i n t } } } { \vert \vert \mathbf { v } _ { l } ^ { \mathrm { j o i n t } } \vert \vert _ { 2 } }\tag{4}
$$

$$
p _ { l } ^ { * } = p _ { l } ^ { b a s e l i n e } + | | \mathbf { v } _ { l } ^ { \mathrm { j o i n t } } | | _ { 2 }\tag{5}
$$

During generation, for each hidden representation $\mathbf { h } _ { l , t } ,$ , we compute its projection onto the unit joint steering vector, representing its current alignment with the steering direction.

$$
p _ { l , t } = \mathbf { h } _ { l , t } ^ { \top } \frac { \mathbf { v } _ { l } ^ { \mathrm { j o i n t } } } { \vert \vert \mathbf { v } _ { l } ^ { \mathrm { j o i n t } } \vert \vert _ { 2 } }\tag{6}
$$

The required steering magnitude is calculated as the difference between the target and current projections, and the residual is applied along the unit joint steering direction:

$$
\tilde { \mathbf { h } } _ { l , t } = \mathbf { h } _ { l , t } + ( p _ { l } ^ { * } - p _ { l , t } ) \frac { \mathbf { v } _ { l } ^ { \mathrm { j o i n t } } } { \lvert \lvert \mathbf { v } _ { l } ^ { \mathrm { j o i n t } } \rvert \rvert _ { 2 } }\tag{7}
$$

The residual steering is signed and can therefore be either steering the token in positive or negative direction. This is important for online control: when the steering strengths are reduced during generation, the target projection is reduced accordingly, and a negative residual removes part of the previously applied steering, allowing the hidden representation to move back toward the prompt baseline.

Finally, the modified representation is normalized to preserve the original hidden-state magnitude. The steering procedure is applied to selected Transformer layers (19–20) during autoregressive inference.

## 3. EXPERIMENTS AND RESULTS

The activation steering vectors were extracted from the regular and loud subsets of EARS [29] (107 speakers) and the default and enunciated subsets of Expresso [30] (4 speakers). We follow the same evaluation protocol with SLE [20], synthesizing first 5 lists of Harvard sentences [31] for four Expresso speakers. We additionally evaluate generalization to unseen speakers using the first 10 speakers of VCTK and to unseen languages using the German, Spanish, and Japanese test sets from TTS Multilingual <sup>2</sup>.

We use acoustic and intelligibility measures to assess the effects of steering. Vowel Space Area (VSA) is the area spanned by the corner vowels /a/, /i/, and /u/ in the F1–F2 space, correlated with increased articulatory distinctiveness. Phoneme alignments were obtained by the Montreal Forced Aligner <sup>3</sup>, with formants extracted using Parselmouth <sup>4</sup>. Spectral Tilt (ST) is computed as the ratio of spectral energy in the 1–5 kHz band to that below 1 kHz and reflects changes associated with vocal effort. Phoneme Rate (PR) is the number of phonemes produced per second. Speaker Similarity (SSIM) is measured as the cosine similarity between WavLM speaker embeddings <sup>5</sup> of synthesized and reference speech.Word Error Rate (WER) is computed using Whisper-medium <sup>6</sup> as an intelligibility measure.

Table 1. Performance on the Expresso dataset. Highlighted rows indicate proposed configurations.
<table><tr><td>Condition</td><td>WER↓</td><td>ST↑</td><td>VSA↑</td><td>PR↓</td><td>SSIM↑</td></tr><tr><td colspan="6">SLE [20]</td></tr><tr><td>Baseline</td><td>4.86</td><td>-19.89</td><td>5.51</td><td>15.97</td><td>0.86</td></tr><tr><td>Half scaling Full scaling</td><td>0.70 1.09</td><td>-17.81 -14.70</td><td>7.71 7.30</td><td>12.36 10.63</td><td>0.86 0.84</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td>Baseline</td><td>1.89</td><td>-18.87</td><td>4.18</td><td>15.14</td><td>0.90</td></tr><tr><td>Low  $( \lambda _ { E , A } = 0 . 5 )$ </td><td>1.77</td><td>-16.61</td><td>5.35</td><td>14.13</td><td>0.90</td></tr><tr><td>Mid  $( \lambda _ { E , A } = 1 . 0 )$ </td><td>1.41</td><td>-15.89</td><td>5.77</td><td>13.25</td><td>0.90</td></tr><tr><td>High  $( \lambda _ { E , A } = 2 . 0 )$ </td><td>1.22</td><td>-15.03</td><td>5.84</td><td>12.30</td><td>0.89</td></tr><tr><td>Vocal effort only</td><td>1.81</td><td>-14.76</td><td>5.52</td><td>15.06</td><td>0.90</td></tr><tr><td>Articulation only</td><td>1.71</td><td>-18.83</td><td>5.28</td><td>12.36</td><td>0.89</td></tr></table>

## 3.1. Lombard-Related Acoustic Changes

We evaluate multiple steering strengths, with $\lambda _ { E } = \lambda _ { A } = 2 . 0$ being the maximum safe strength observed experimentally, thereby also testing extrapolation beyond the reference data. The baseline is unsteered Qwen3-TTS speech.

Table 1 shows systematic changes with increasing steering strength: spectral tilt becomes less negative, speaking rate decreases, and vowel-space dispersion increases. These trends are consistent with increased vocal effort and hyperarticulation. The individual directions produce distinct effects, with vocal-effort steering primarily affecting spectral tilt and articulation steering producing a stronger reduction in speaking rate. Both directions also increase VSA, while joint steering combines these effects. Compared to SLE [20], Qwen3-TTS baseline achieves lower WER $( \Delta = - 2 . 9 7 )$ for regular speech and maintains slightly higher speaker similarity across steering conditions. The proposed method produces comparable changes in spectral tilt, speaking rate, and VSA relative to their corresponding baselines. While SLE was trained with the same speakers, our model achieves comparable results without additional training, suggesting potential generalization to other speakers.

Table 2. Multilingual and multispeaker evaluation across VCTK, German, Spanish, and Japanese test sets.
<table><tr><td>Test Set</td><td>Condition</td><td>WER↓</td><td>ST↑</td><td>VSA ↑</td><td>PR↓</td><td>SSIM ↑</td></tr><tr><td rowspan="2">VCTK</td><td>Baseline</td><td>0.51</td><td>-21.19</td><td>1.31</td><td>16.88</td><td>0.96</td></tr><tr><td>Steered</td><td>0.29</td><td>-17.41</td><td>1.93</td><td>14.93</td><td>0.95</td></tr><tr><td rowspan="2">German</td><td>Baseline</td><td>0.77</td><td>-18.18</td><td>9.05</td><td>14.22</td><td>0.96</td></tr><tr><td>Steered</td><td>1.26</td><td>-13.04</td><td>4.00</td><td>11.59</td><td>0.95</td></tr><tr><td rowspan="2">Spanish</td><td>Baseline</td><td>1.76</td><td>-22.91</td><td>5.09</td><td>16.88</td><td>0.98</td></tr><tr><td>Steered</td><td>1.80</td><td>-18.29</td><td>6.76</td><td>10.93</td><td>0.95</td></tr><tr><td rowspan="2">Japanese</td><td>Baseline</td><td>4.68</td><td>-19.61</td><td>2.13</td><td>12.35</td><td>0.96</td></tr><tr><td>Steered</td><td>5.64</td><td>-15.88</td><td>2.32</td><td>11.95</td><td>0.94</td></tr></table>

We further evaluate generalization to unseen VCTK speakers and to German, Spanish, and Japanese, as results shown in Table 2. The steering directions consistently modify spectral tilt and speaking rate across speakers and languages, although VSA changes are more speaker- and languagedependent. This is expected since the acoustic realization of hyper-articulation depends on phonological structure. The multilingual results nevertheless provide evidence of crosslingual transfer, while suggesting that precise articulatory control may benefit from language-specific calibration.

Speaker similarity remains largely preserved across steering conditions, indicating that the method can modify Lombardrelated attributes without compromising speaker identity.

## 3.2. Speech Intelligibility In Noise

We evaluate speech intelligibility under restaurant babble noise at SNRs of 1, 5, and 10 dB using WER. For each condition, the noise is scaled to the speech power to maintain a fixed SNR across conditions and avoid attributing intelligibility differences to overall energy gain which could be simply achieved by turning on the device volume. The results in Figure 1 show that WER consistently decreases when steering is applied. The effect is more prominent in higher noise levels, especially for German, Spanish and Japanese.

![](images/81411a908b658206877c8032102344133301fae060054748fa606474f7529960.jpg)

![](images/7ad60cf051e9d31672b59ad4e805c4d3976dd5295435691ab3402248e760c098.jpg)

![](images/0ad33601ec0be76b9b53c4fb20b2dfce05c4a70fdd3e869f34c7568fa2859921.jpg)

![](images/318dc32d01b6dc4f374bac45fec6f1de64d3791c1933e537b48aac638534ca89.jpg)

![](images/178c0b7abc9a94bb007b3ec1a77aae439a17e265aaf9be795df3bb7b64e18ec6.jpg)  
Fig. 1. WER results in background noise with different SNR levels.

## 3.3. Human Evaluation

We conduct a user study to evaluate the perceived intelligibility of the proposed method. Participants evaluate 15 pairs of baseline and steered utterances presented with background noise at an SNR of 5 dB. For each pair, participants rate their preference on a scale from −3 to +3, where +3 indicates a strong preference for the steered audio and −3 indicates a strong preference for the baseline. Participants are not informed which sample corresponds to which system, and the order is randomized. Seven participants completed the evaluation, resulting in a Comparison Mean Opinion Score (CMOS) of $\mathbf { + 0 . 9 5 8 \ : \pm { \ : 0 . 4 8 } }$ with a 95% confidence interval. The positive CMOS indicates a preference for the proposed method and provides subjective evidence consistent with the objective intelligibility results.

## 3.4. Dynamic Steering Evaluation

We adapt the steering method to Qwen3-TTS streaming mode to simulate dynamic steering-strength adaptation. We synthesize 10 long sentences from LibriTTS [32] (12–18 seconds) using a single Expresso reference speaker. Steering is applied only during an interval of the utterance, after which the steering strength returns to zero. We measure spectral tilt changes over time and the additional latency introduced by token-level steering during streaming generation.

Table 3. Streaming performance across evaluation modes.
<table><tr><td>Steering Mode</td><td>Mean TTFA (s)</td><td>Mean RTF</td></tr><tr><td>No Steering</td><td>0.469</td><td>1.078</td></tr><tr><td>Dynamic Steering</td><td>0.474</td><td>1.088</td></tr><tr><td>Constant Steering</td><td>0.475</td><td>1.086</td></tr></table>

As shown in Table 3, our streaming activation steering mechanism maintains real-time efficiency, confirming that the streaming hooks add negligible computational overhead. Meanwhile, Table 4 demonstrates temporal precision: dynamic steering successfully follows the no steering baseline prior to intervention, shifts instantaneously during the active steering window and matches constant steering behavior, and cleanly reverts post-intervention.

Table 4. Segmented spectral tilt (dB) across steering modes.
<table><tr><td>Steering Mode</td><td>Before</td><td>During</td><td>After</td></tr><tr><td>No Steering</td><td>-20.84</td><td>-21.06</td><td>-20.96</td></tr><tr><td>Dynamic Steering</td><td>-20.51</td><td>-16.94</td><td>-20.68</td></tr><tr><td>Constant Steering</td><td>-17.75</td><td>-17.26</td><td>-17.65</td></tr></table>

Figure 2 shows an example of dynamically steered speech. The increase in amplitude and shift in energy toward higher frequencies during steering correspond to the simulated Lombard effect. After the steering interval ends, speech returns to its baseline amplitude and frequency distribution, demonstrating dynamic control during streaming.

![](images/cb6c3e75d94d01c8a8d58f585d27afec34f7bee5e9b3c03006a190b3bbfb8705.jpg)  
Fig. 2. Features of the dynamically steered speech. Highlighted part corresponds to active steering window.

## 4. CONCLUSION

We presented a training-free activation steering approach for controllable Lombard speech synthesis with Qwen3-TTS. Using paired speech data, our method jointly controls vocal effort and hyper-articulation without model fine-tuning, with experiments across seen and unseen speakers and multiple languages. To prevent cumulative effects in autoregressive generation, we introduced Prompt-Relative Activation Steering, which regulates each activation relative to the reference voice and enables stable, dynamic control within an utterance. Experiments demonstrate systematic Lombard-related acoustic changes, preserved speaker similarity, and improved ASR-based intelligibility under noise. Future work will address more diverse acoustic environments and listener types and improve hyper-articulation control across languages.

## 5. REFERENCES

[1] J.-C. Junqua, “The lombard reflex and its role on human listeners and automatic speech recognizers,” The Journal of the Acoustical Society of America, vol. 93, no. 1, pp. 510–524, 1993.

[2] H. Soltau and A. Waibel, “Specialized acoustic models for hyperarticulated speech,” in ICASSP, 2000.

[3] F. Metze and A. Waibel, “A flexible stream architecture for asr using articulatory features,” in Proc. ICSLP 2002, 2002, pp. 2133–2136.

[4] H. Brouckxon, W. Verhelst, and B. D. Schuymer, “Time and frequency dependent amplification for speech intelligibility enhancement in noisy environments,” in Interspeech, 2008.

[5] B. Sauert and P. Vary, “Near-end listening enhancement in the presence of bandpass noises,” in Speech Communication; 10. ITG Symposium, 2012.

[6] C. H. Taal and J. Jensen, “Sii-based speech preprocessing for intelligibility improvement in noise.” in Interspeech, 2013.

[7] C. H. Taal, R. C. Hendriks, and R. Heusdens, “Speech energy redistribution for intelligibility improvement in noise based on a perceptual distortion measure,” Computer Speech & Language, vol. 28, no. 4, pp. 858–872, 2014.

[8] T.-C. Zorila and Y. Stylianou, “On spectral and time domain energy reallocation for speech-in-noise intelligibility enhancement.” in Interspeech, 2014.

[9] V. Kandia, “Speech-in-noise intelligibility improvement based on spectral shaping and dynamic range compression,” in Interspeech, 2012.

[10] T. Raitio, A. Suni, M. Vainio, and P. Alku, “Analysis of hmmbased lombard speech synthesis.” in Interspeech, 2011.

[11] B. Bollepalli, L. Juvela, and P. Alku, “Lombard speech synthesis using transfer learning in a tacotron text-to-speech system,” in Interspeech, 2019.

[12] D. Paul, M. P. Shifas, Y. Pantazis, and Y. Stylianou, “Enhancing speech intelligibility in text-to-speech synthesis using speaking style conversion,” in Interspeech, 2020.

[13] Q. Hu, T. Bleisch, P. Petkov, T. Raitio, E. Marchi, and V. Lakshminarasimhan, “Whispered and lombard neural speech synthesis,” in SLT, 2021.

[14] T. Raitio, P. Petkov, J. Li, M. Shifas, A. Davis, and Y. Stylianou, “Vocal effort modeling in neural tts for improving the intelligibility of synthetic speech in noise,” in Interspeech, 2022.

[15] D. Woszczyk, M. S. Ribeiro, T. Merritt, and D. Korzekwa, “Voice conversion for lombard speaking style with implicit and explicit acoustic feature conditioning,” arXiv preprint arXiv:2507.09310, 2025.

[16] S. Novitasari, S. Sakti, and S. Nakamura, “Dynamically adaptive machine speech chain inference for tts in noisy environment: Listen and speak louder.” in Interspeech, 2021.

[17] T. H. G. Lobato and M. Schafer, “Gradual modeling of the¨ lombard effect by modifying speaker embeddings from a text-

to-speech model,” in Interspeech, 2025.

[18] B. Picart, T. Drugman, and T. Dutoit, “Analysis and synthesis of hypo and hyperarticulated speech,” arXiv preprint arXiv:2006.04136, 2020.

[19] M. Nishihara, D. Wells, K. Richmond, and A. Pine, “Lowdimensional style token control for hyperarticulated speech synthesis,” in Interspeech, 2024.

[20] S. Akti and A. Waibel, “Synthesizing the lombard effect: Multi-level control of speech clarity and vocal effort in tts,” arXiv preprint arXiv:2606.23176, 2026.

[21] S. Rottschafer, H. Buschmeier, H. van Welbergen, and¨ S. Kopp, “Online lombard-adaptation in incremental speech synthesis,” in Interspeech, 2015.

[22] A. M. Turner, L. Thiergart, G. Leech, D. Udell, J. J. Vazquez, U. Mini, and M. MacDiarmid, “Steering language models with activation engineering,” arXiv preprint arXiv:2308.10248, 2023.

[23] L. Zhou, H. Jiang, J. Li, T. Wang, and H. Li, “Emoshift: Lightweight activation steering for enhanced emotion-aware speech synthesis,” in ICASSP, 2026.

[24] S. Wang, S. Tan, S. Liu, H. Jia, G. Huang, J. Bailey, and T. Dang, “Cocoemo: Composable and controllable humanlike emotional tts via activation steering,” arXiv preprint arXiv:2602.03420, 2026.

[25] T. Xie, S. Yang, C. Li, D. Yu, and L. Liu, “Emosteer-tts: Finegrained and training-free emotion-controllable text-to-speech via activation steering,” arXiv preprint arXiv:2508.03543, 2025.

[26] M. Yang and J. H. Hansen, “Activation steering for accent-neutralized zero-shot text-to-speech,” arXiv preprint arXiv:2603.05977, 2026.

[27] D. Kang, Z. Liu, N. Ma, Y. Huang, Z. Tan, and M. Jiang, “Prompt-activation duality: Improving activation steering via attention-level interventions,” arXiv preprint arXiv:2605.10664, 2026.

[28] H. Hu, X. Zhu, T. He, D. Guo, B. Zhang, X. Wang, Z. Guo, Z. Jiang, H. Hao, Z. Guo et al., “Qwen3-tts technical report,” arXiv preprint arXiv:2601.15621, 2026.

[29] J. Richter, Y.-C. Wu, S. Krenn, S. Welker, B. Lay, S. Watanabe, A. Richard, and T. Gerkmann, “Ears: An anechoic fullband speech dataset benchmarked for speech enhancement and dereverberation,” arXiv preprint arXiv:2406.06185, 2024.

[30] T. A. Nguyen, W.-N. Hsu, A. d’Avirro, B. Shi, I. Gat, M. Fazel-Zarani, T. Remez, J. Copet, G. Synnaeve, M. Hassid et al., “Expresso: A benchmark and analysis of discrete expressive speech resynthesis,” in Interspeech, 2023.

[31] E. H. Rothauser, “Ieee recommended practice for speech quality measurements,” IEEE Transactions on Audio and Electroacoustics, vol. 17, no. 3, pp. 225–246, 1969.

[32] Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani, Y. Zhang, W. Han, and A. Bapna, “Libritts-r: A restored multi-speaker text-to-speech corpus,” in Interspeech, 2023.