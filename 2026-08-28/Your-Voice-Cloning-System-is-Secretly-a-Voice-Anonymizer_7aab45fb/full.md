# Your Voice Cloning System is Secretly a Voice Anonymizer

Romolo Muletta<sup>1</sup>, Felix Saaro <sup>ID</sup> <sup>1</sup>, Mark Cieliebak <sup>ID</sup> <sup>1</sup>, Jan Deriu <sup>ID</sup> <sup>1,∗∗</sup>

<sup>1</sup> Centre for Artificial Intelligence ZHAW School of Engineering

muletrom@zhaw.ch, saaf@zhaw.ch, ciel@zhaw.ch, deri@zhaw.ch

## Abstract

Speaker anonymization suppresses speaker-identifying attributes from speech while preserving linguistic content and quality. We propose repurposing XTTSv2, a multilingual voice cloning model trained on 27k hours of speech, for speaker anonymization without retraining. Our key insight is that XTTSv2’s voice cloning capabilities preserve prosodic structure independently of speaker identity, enabling voice conversion by conditioning on a pseudo-speaker. We introduce an iterative refinement strategy that balances privacy and utility by maximizing a harmonic mean of speaker dissimilarity and intelligibility. Evaluated on seven European languages across CommonVoice and Multilingual LibriSpeech, our system achieves near-optimal privacy (EER ≈ 0.49), competitive intelligibility, and substantially better speech quality than dedicated anonymization baselines, while requiring no language-specific training. We release the code here https://github.com/ rm00cr/coqui-tts.

Index Terms: voice anonymization

## 1. Introduction

Speech signals contain biometric information that can be used to identify speakers, which can be misused for unauthorized profiling. The task of speaker anonymization is to automatically suppress speaker identifying attributes from the speech, making identification impossible [1]. That is, to modify speech so that the speaker’s identity remains hidden while preserving linguistic content, prosody, and speech quality. The original speaker’s voice is converted to an artificial voice, so that the output cannot be linked to the original speaker by an automatic speaker verification (ASV) system. The VoicePrivacy Challenge [1, 2, 3] has emerged as the primary benchmark for evaluating speaker anonymization systems. The challenge defines anonymization as a privacy-utility trade-off: systems must maximize the equal error rate (EER) of automatic speaker verification (ASV) attacks while minimizing degradation to word error rate (WER) and preserving emotional content.

Speaker anonymization approaches can be broadly categorized into signal processing methods and neural networkbased methods. Signal processing techniques, such as the McAdams coefficient [4] or phase vocoder-based time-scale modification [5], offer parameter-free anonymization but provide limited privacy protection against sophisticated attackers. Recent work is based on neural network anonymization strategies. Meyer et al. [6] demonstrated that Generative Adversarial Networks (GANs) can generate diverse pseudo-speaker embeddings with controllable dissimilarity to source speakers. Meyer et al. [7] further showed that prosody can be preserved independently of speaker identity by cloning pitch, energy, and duration patterns while anonymizing the speaker embedding. Lv et al. [8] proposed SALT, which leverages self-supervised learning (WavLM) to extract latent features and creates pseudo-speakers using multiple speaker representations in latent space, achieving state-of-the-art speaker distinctiveness while maintaining privacy. Neural audio codecs (NAC) language models [9] exploit the natural disentanglement between semantic tokens (encoding linguistic content) and acoustic tokens (encoding speaker characteristics) to achieve strong anonymization while maintaining speech quality.

Despite significant progress, speaker anonymization research has been predominantly limited to English. This linguistic bias severely constrains the applicability of anonymization technologies to the vast majority of the world’s languages and speakers. Only recently have researchers begun addressing multilingual scenarios. Miao et al. [10, 11] proposed a language-independent approach using self-supervised learning (SSL) models such as HuBERT to extract content representations without explicit language-specific components. Their system, evaluated on English and Mandarin, achieved promising results but still required careful domain adaptation for optimal performance. Meyer et al. [12] extended their GAN-based anonymization system to nine languages by replacing monolingual ASR and text-to-speech (TTS) components with multilingual counterparts. Their results demonstrated that speaker embeddings trained on English generalize well across languages, and that anonymization performance is primarily determined by the quality of the language-specific synthesis component. This finding suggests that high-quality multilingual TTS systems could enable effective anonymization across diverse languages.

State-of-the-art systems typically require training complex multi-component pipelines from scratch, including speaker encoders, linguistic feature extractors, and neural vocoders. This complexity creates barriers to entry for low-resource languages where training data for each component may be scarce.

In parallel, the text-to-speech (TTS) community has developed powerful multilingual voice cloning models. XTTSv2 [13] achieves state-of-the-art zero-shot voice cloning across 16 languages with only a few seconds of reference audio. XTTS was trained on 27,000 hours of multilingual speech and enables cross-lingual voice cloning while preserving prosody and naturalness.

In this work, we observe that voice cloning and speaker anonymization share a common core operation: voice conversion conditioned on a target speaker identity. Rather than training a dedicated anonymization system from scratch, we propose repurposing XTTSv2 for speaker anonymization by leveraging its inference-time flexibility. Our key insight is that XTTSv2 naturally preserves prosodic structure independently of speaker identity by leveraging its voice cloning capabilities. By combining the codebook representation used in XTTS with a synthesized pseudo-speaker identity, we can transform the original speaker’s voice while maintaining prosody, linguistic content, and speech quality. Our contributions are twofold:

![](images/af421151aa0c837ac43b30ab94e020c7211f11a646a9ca7b42ee7aa90cb00519.jpg)  
Figure 1: Speaker anonymization pipeline using XTTSv2. The original speech is encoded into a codebook representation via VQ-VAE, preserving prosody. The GPT-2 backbone transforms this representation to match the pseudo-speaker identity provided by the Perceiver conditioner.

• We introduce a simple method for repurposing XTTSv2, a multilingual voice-cloning model, for speaker anonymization without retraining.

• We propose an iterative refinement strategy that balances privacy and utility by maximizing a harmonic mean of speaker dissimilarity and intelligibility preservation.

Unlike previous work that builds complex systems from scratch, our approach leverages the capabilites state-of-the-art TTS model. This makes speaker anonymization more accessible, particularly for languages already supported by XTTSv2, while achieving competitive privacy-utility trade-offs. We will release the code and the generated, anonymized audio files. For the version under review, we add the fully functional annotation tool and anonymized audio files as supplemental material.

## 2. Model

We repurpose XTTSv2 [13], a multilingual voice cloning model trained on 27k hours of speech, for speaker anonymization. Rather than training a dedicated system, we exploit the model’s voice conversion capabilities by conditioning generation on both a target speaker identity and the prosodic structure of the original speech.

## 2.1. XTTSv2 Components

We briefly describe the relevant components of XTTSv2; see Figure 1.

VQ-VAE: A Vector Quantized Variational Autoencoder [14] that encodes mel-spectrograms into discrete codebook indices. In standard voice cloning, this component is used only during training. For anonymization, we use it at inference time to extract a codebook representation of the original speech that preserves prosodic structure (pitch contour, rhythm, energy) independently of speaker identity.

Perceiver Conditioner: Six attention layers followed by a Perceiver Resampler [15], which compresses variable-length reference audio into a fixed set of 32 embeddings representing the target speaker’s voice characteristics.

GPT-2 Backbone: A decoder-only transformer [16] that autoregressively predicts codebook tokens. In standard cloning, it receives only the Perceiver output and text tokens. For anonymization, we additionally provide the original speech’s codebook representation, enabling the model to preserve prosody while transforming speaker identity.

Text Encoder: A BPE tokenizer [17] with 6681 tokens, providing the linguistic content.

HiFi-GAN Vocoder: Synthesizes the final waveform from the transformer’s latent output, additionally conditioned on a speaker embedding from the H/ASP model [18].

## 2.2. Anonymization Pipeline

The full pipeline is depicted in Figure 1. It comprises an ASR model (in our case, Whisper-Large-V3 [19]), a gender classifier, a pool of 10 reference speakers per language and gender, each with a set of utterances used to create a novel, anonymized voice, and the XTTSv2 model. Given speech to be anonymized:

1. Feature extraction: Transcribe the speech using Whisper-Large-V3 [19], extract the codebook representation via VQ-VAE, and compute an ECAPA2 [20] speaker embedding.

2. Pseudo-speaker construction: Classify the speaker’s gender and select the corresponding reference pool. For each of the 10 pool speakers, select the utterance whose ECAPA2 embedding has the lowest cosine similarity to the original speaker’s embedding. Concatenate these 10 utterances and pass them through the Perceiver conditioner to obtain a composite pseudo-speaker representation. At the same time, they are passed through the speaker encoder to create a speaker embedding.

3. Voice conversion: Concatenate the embeddings from the perceiver conditioner, text tokens, and original codebook representation. The GPT-2 backbone transforms this into a new codebook sequence aligned with the pseudo-speaker identity.

4. Synthesis: The HiFi-GAN vocoder generates the anonymized waveform from the transformed codebook and the pseudo-speaker embedding.

## 2.3. Iterative Refinement

We observed that the transformer occasionally reverts to characteristics of the original speaker mid-utterance, likely due to attention over the original codebook tokens. To address this, we iteratively reapply the pipeline to its own output until convergence. At each iteration, we compute a selection criterion balancing privacy and utility:

$$
H = \frac { 2 \cdot ( 1 - \mathrm { { W E R } ) \cdot ( 1 - \mathrm { { S i m } _ { \mathrm { { o r i g } } } ) } } } { ( 1 - \mathrm { { W E R } ) + ( 1 - \mathrm { { S i m } _ { \mathrm { { o r i g } } } ) } } }\tag{1}
$$

where $\mathrm { S i m _ { o r i g } }$ is the cosine similarity between the anonymized and original speaker embeddings. We select the iteration that maximizes H.

<table><tr><td rowspan="2">Language</td><td colspan="2">Common Voice</td><td colspan="2">MLS</td></tr><tr><td>Spkrs</td><td>Samples</td><td>Spkrs</td><td>Samples</td></tr><tr><td>German (de)</td><td>819</td><td>3,003</td><td>60</td><td>6,863</td></tr><tr><td>English (en)</td><td>2,538</td><td>4,481</td><td>84</td><td>7,576</td></tr><tr><td>Spanish (es)</td><td>2,068</td><td>6,429</td><td>40</td><td>4,792</td></tr><tr><td>French (fr)</td><td>934</td><td>3,506</td><td>36</td><td>4,838</td></tr><tr><td>Italian (it)</td><td>1,083</td><td>6,836</td><td>20</td><td>2,510</td></tr><tr><td>Dutch (nl)</td><td>631</td><td>12,400</td><td>12</td><td>6,166</td></tr><tr><td>Portuguese (pt)</td><td>711</td><td>7,224</td><td>20</td><td>1,697</td></tr><tr><td>Total</td><td>8,784</td><td>43,879</td><td>272</td><td>34,442</td></tr></table>

Table 1: Evaluation data: speaker and sample counts across seven languages (dev + test splits combined).

## 2.4. Pseudo-Speaker Pool Construction

The quality of anonymized speech depends heavily on the reference audio used to define the pseudo-speaker. We construct language- and gender-specific pools of 10 high-quality reference speakers from Multilingual LibriSpeech (MLS) [21], selected as follows:

For each candidate speaker, we anonymize a held-out set of utterances using that speaker as the target voice. We then evaluate each candidate on two criteria: (1) speaker dissimilarity, measured as cosine distance between ECAPA2 [20] embeddings of the original and anonymized speech, and (2) intelligibility preservation, measured as WER of the anonymized speech. We retain the 10 speakers per language and gender who achieve the best trade-off between these criteria.

## 3. Experimental Setup

## 3.1. Datasets and Languages

We evaluate speaker anonymization on two multilingual datasets: Multilingual LibriSpeech (MLS) [21] and Common-Voice 23.0 (CV) [22]. We focus on seven languages: English (en), German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt), and Dutch (nl). Table 1 shows the test-set overview.

For MLS, we use the full dev and test splits provided by the dataset. The MLS dev and test splits contain 272 speakers with high-quality audiobook recordings. For CV, we use the dev and test splits containing 8,784 speakers. Generally, MLS samples exhibit higher audio quality than CV samples, which often contain background noise and are recorded using consumer-grade laptop or smartphone microphones. This quality difference allows us to evaluate robustness across varying acoustic conditions.

The pseudo-speaker pools are constructed from MLS training splits, which provide high-quality, clean speech necessary for effective voice synthesis. We use 10 speakers per languagegender combination, selected based on their anonymization effectiveness, as described in Section 2.4.

## 3.2. Baseline Systems

We compare our approach against two competitive anonymization systems:

SALT [8] uses self-supervised learning features from WavLM to extract latent representations. Pseudo-speakers are created through interpolation and extrapolation of multiple speaker representations in latent space. SALT achieved first place in the VoicePrivacy Challenge 2022, demonstrating state-of-the-art speaker distinctiveness while maintaining privacy. We use the official implementation<sup>1</sup> with default parameters.

MultiLingual [12] extends GAN-based speaker anonymization to nine languages using multilingual ASR (Whisper-large-v3) and TTS (IMS Toucan) components. The system extracts linguistic content, prosody, and speaker embeddings, then replaces the speaker embedding with a GAN-generated pseudo-speaker while preserving prosodic features. We use the official implementation<sup>2</sup> configured for the seven evaluation languages.

## 3.3. Our Anonymization System

Our system repurposes XTTSv2 [13] for speaker anonymization without retraining. The pipeline consists of the following components:

## Feature Extraction:

• Transcription: Whisper-Large-V3 [19] for multilingual automatic speech recognition

• Prosody: VQ-VAE codebook representation extracted from XTTSv2’s encoder

• Speaker Identity: ECAPA2 [20] embeddings for both original and pseudo-speaker characterization

• Gender: Binary gender classifier for pool selection

Pseudo-Speaker Construction: For each utterance, we:

1. Classify the speaker’s gender

2. Select the corresponding reference pool (10 speakers per language/gender)

3. For each pool speaker, choose the utterance with the lowest ECAPA2 cosine similarity to the original speaker

4. Concatenate these 10 utterances and process through XTTSv2’s Perceiver conditioner to create a composite pseudo-speaker representation

Voice Conversion: The GPT-2 backbone in XTTSv2 receives:

• Pseudo-speaker embeddings from the Perceiver conditioner

• Text tokens from the Whisper transcription

• Original codebook representation preserving prosodic structure

The HiFi-GAN vocoder [23] synthesizes the final anonymized waveform conditioned on the transformed codebook and pseudo-speaker embedding.

Iterative Refinement: We apply the anonymization pipeline iteratively to its own output, computing, at each iteration, a harmonic mean that balances privacy and utility, as described in Equation 1.

## 3.4. Evaluation Protocol

Privacy Evaluation: Privacy is measured using Equal Error Rate (EER) from an automatic speaker verification (ASV) system. We use an ECAPA2 as our ASV system. Higher EER indicates better anonymization; 50% corresponds to random-guess performance (perfect anonymization).

Intelligibility: Computed using Whisper-Large-V3 on the anonymized speech, and then computed the WER between the original transcript and the generated transcript of the anonymized speech. Lower WER indicates better intelligibility preservation.

<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="8">EER↑</td><td colspan="8">WER↓</td><td>∆UTM.↑</td></tr><tr><td>de</td><td>en</td><td>es</td><td>fr</td><td>it</td><td>nl</td><td>pt</td><td>Avg</td><td>de</td><td>en</td><td>es</td><td>fr</td><td>it</td><td>nl</td><td>pt</td><td>Avg</td><td>Avg</td></tr><tr><td rowspan="3">CV</td><td>SALT</td><td>.37</td><td>.35</td><td>.38</td><td>.38</td><td>.37</td><td>.34</td><td>.38</td><td>.37</td><td>.21</td><td>.37</td><td>.18</td><td>.32</td><td>.23</td><td>.17</td><td>.35</td><td>.26</td><td>-0.74</td></tr><tr><td>MultiLingual</td><td>.46</td><td>.46</td><td>.46</td><td>.46</td><td>.46</td><td>.46</td><td>.47</td><td>.46</td><td>.20</td><td>.31</td><td>.19</td><td>.28</td><td>.25</td><td>.23</td><td>.39</td><td>.27</td><td>-0.62</td></tr><tr><td>XTTSv2 (Ours)</td><td>.49</td><td>.49</td><td>.49</td><td>.50</td><td>.49</td><td>.50</td><td>.48</td><td>.49</td><td>.16</td><td>.28</td><td>.10</td><td>.20</td><td>.12</td><td>.07</td><td>.22</td><td>.16</td><td>+0.17</td></tr><tr><td rowspan="3">MIS</td><td>SALT</td><td>.35</td><td>.29</td><td>.32</td><td>.35</td><td>.27</td><td>.16</td><td>.36</td><td>.30</td><td>.09</td><td>.12</td><td>.06</td><td>.12</td><td>.17</td><td>.23</td><td>.13</td><td>.13</td><td>-1.29</td></tr><tr><td>MultiLingual</td><td>.46</td><td>.45</td><td>.45</td><td>.45</td><td>.44</td><td>.36</td><td>.45</td><td>.44</td><td>.14</td><td>.17</td><td>.08</td><td>.17</td><td>.20</td><td>.24</td><td>.21</td><td>.17</td><td>-1.23</td></tr><tr><td>XTTSv2 (Ours)</td><td>.49</td><td>.48</td><td>.48</td><td>.48</td><td>.42</td><td>.44</td><td>.44</td><td>.46</td><td>.15</td><td>.13</td><td>.08</td><td>.12</td><td>.17</td><td>.24</td><td>.20</td><td>.16</td><td>-0.35</td></tr></table>

Table 2: Speaker anonymization results across seven languages (dev+test combined). EER: higher is better (0.5 = optimal privacy). WER: lower is better. ∆UTMOS: naturalness changefrom original. Best results in bold.

Speech Quality: We measure speech quality using the UTokyo-SaruLab MOS Prediction System (UTMOS) [24]. We report ∆UTMOS = UTMOS<sub>anonymized</sub> - UTMOS<sub>original</sub>, where values near 0 indicate preserved quality, and higher values indicate better relative quality (range: -4 to +4).

Comparative Mean Opinion Score (CMOS): We conduct human listening tests on English to evaluate perceived speech quality. We randomly sample 30 utterances from the MLS English test set and anonymize them using all three systems. Nine fluent English speakers rate each original-anonymized pair on a 7-point scale from -3 (much worse) to +3 (much better), with 0 indicating equal quality. Each system receives 270 ratings (9 raters × 30 samples).

## 3.5. Implementation Details

All experiments use the XTTSv2 version from the Coqui TTS library <sup>3</sup>. For fair comparison, all systems use the same evaluation models (Whisper-Large-V3 for WER, ECAPA2 for EER).

## 4. Results

Table 2 presents the main results comparing our XTTSv2-based system against SALT and MultiLingual baselines across seven languages on both CommonVoice (CV) and Multilingual LibriSpeech (MLS) datasets.

Privacy Protection (EER): Our system achieves an average EER of 0.49 on CV and 0.46 on MLS, approaching the theoretical maximum of 0.50 (indicating random guessing by the ASV system). This represents improved anonymization compared to SALT (0.37 CV, 0.30 MLS) and performs comparably or better than MultiLingual (0.46 CV, 0.44 MLS). On CV, EER values are consistent across languages (0.48–0.50), while on MLS, they range from 0.42 to 0.49. The lower performance on some MLS languages may reflect differences in recording quality or speaker diversity within the dataset.

Intelligibility (WER): On CV, our approach achieves an average WER of 0.16, compared to 0.26 for SALT and 0.27 for MultiLingual. This corresponds to a 38% relative reduction compared to the baselines. Individual language WER ranges from 0.07 (Dutch) to 0.28 (English). On MLS, SALT achieves the lowest average WER at 0.13, while our system obtains 0.16, slightly outperforming MultiLingual (0.17).

Speech Quality (∆UTMOS): On CV, our system achieves ∆UTMOS of +0.17, indicating improved quality relative to the original recordings. SALT and MultiLingual show degradation of -0.74 and -0.62, respectively. The positive value for our system likely reflects the cleaner synthesis, which compensates for noise in the original CV recordings. On MLS, all systems show quality degradation because the audiobook recordings have a high baseline quality. Our system achieves ∆UTMOS of -0.35, compared to -1.29 for SALT and -1.23 for MultiLingual.

<table><tr><td>System</td><td>CMOS</td><td>Std Dev</td></tr><tr><td>SALT</td><td>-1.00</td><td>±1.12</td></tr><tr><td>MultiLingual</td><td>-1.85</td><td>±0.88</td></tr><tr><td>XTTSv2 (Ours)</td><td>-0.90</td><td>±1.39</td></tr></table>

Table 3: Comparative Mean Opinion Score (CMOS) on English MLS test samples (30 samples, 9 raters, 270 total comparisons per system). Anonymized speech compared against original recordings. Scale: -3 (much worse) to +3 (much better), 0 (equal quality). Higher values indicate better quality preservation.

Naturalness (CMOS): Table 3 shows human evaluation results for English on MLS. All systems exhibit quality degradation relative to the high-quality audiobook originals, consistent with the negative ∆UTMOS values observed on MLS. Our system achieves the least degradation (CMOS = -0.90), outperforming SALT (-1.00) and MultiLingual (-1.85). The results confirm that XTTSv2’s vocoder better preserves naturalness despite anonymization.

Iterative Refinement. The iterative refinement strategy selects different iterations across utterances: iteration 0 is chosen in 19.6% of cases, iterations 1-3 are each selected approximately 18% of the time, and iteration 4 is chosen in 25.0% of cases. This distribution indicates that different utterances require different levels of refinement. None of the manually tested audios exhibit the mid-utterance degradation that motivated the iterative refinement.

## 5. Discussion & Conclusion

We presented a speaker anonymization approach that leverages XTTSv2 for privacy protection without retraining, achieving near-optimal EER across seven languages while substantially improving speech quality relative to dedicated baselines. Rather than building anonymization systems from scratch, our work demonstrates that the community can leverage existing multilingual TTS models. Future work should address adversarial robustness against informed attackers, extend to non-European languages, and explore generalization to other voice cloning architectures.

## 6. AI Declaration

We used Claude Opus 4.5 to edit and rewrite passages of text for better readability.

## 7. References

[1] N. Tomashenko, B. M. L. Srivastava, X. Wang, E. Vincent, A. Nautsch, J. Yamagishi, N. Evans, J. Patino, J.-F. Bonastre, P.- G. Noe, and M. Todisco, “Introducing the voiceprivacy initiative,”´ in Interspeech 2020, 2020, pp. 1693–1697.

[2] M. Panariello, N. Tomashenko, X. Wang, X. Miao, P. Champion, H. Nourtel, M. Todisco, N. Evans, E. Vincent, and J. Yamagishi, “The voiceprivacy 2022 challenge: Progress and perspectives in voice anonymisation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.

[3] N. Tomashenko, X. Miao, P. Champion, S. Meyer, X. Wang, E. Vincent, M. Panariello, N. Evans, J. Yamagishi, and M. Todisco, “The voiceprivacy 2024 challenge evaluation plan,” arXiv preprint arXiv:2404.02677, 2024.

[4] J. Patino, N. Tomashenko, M. Todisco, A. Nautsch, and N. Evans, “Speaker anonymisation using the mcadams coefficient,” in Interspeech 2021, 2021, pp. 1099–1103.

[5] C. O. Mawalim, S. Okada, and M. Unoki, “Speaker anonymization by pitch shifting based on time-scale modification,” in 2nd Symposium on Security and Privacy in Speech Communication, 2022, pp. 35–42.

[6] S. Meyer, P. Tilli, P. Denisov, F. Lux, J. Koch, and N. T. Vu, “Anonymizing speech with generative adversarial networks to preserve speaker privacy,” in 2022 IEEE Spoken Language Technology Workshop (SLT), 2023, pp. 912–919.

[7] S. Meyer, F. Lux, J. Koch, P. Denisov, P. Tilli, and N. T. Vu, “Prosody is not identity: A speaker anonymization approach using prosody cloning,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[8] Y. Lv, J. Yao, P. Chen, H. Zhou, H. Lu, and L. Xie, “Salt: Distinguishable speaker anonymization through latent space transformation,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–8.

[9] M. Panariello, F. Nespoli, M. Todisco, and N. Evans, “Speaker anonymization using neural audio codec language models,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 4725– 4729.

[10] X. Miao, X. Wang, E. Cooper, J. Yamagishi, and N. Tomashenko, “Language-independent speaker anonymization approach using self-supervised pre-trained models,” in The Speaker and Language Recognition Workshop (Odyssey 2022), 2022, pp. 279–286.

[11] Xiaoxiao Miao and Xin Wang and Erica Cooper and Junichi Yamagishi and Natalia Tomashenko, “Analyzing Language-Independent Speaker Anonymization Framework under Unseen Conditions,” in Interspeech 2022, 2022, pp. 4426–4430.

[12] S. Meyer, F. Lux, and N. T. Vu, “Probing the Feasibility of Multilingual Speaker Anonymization,” in Interspeech 2024, 2024, pp. 4448–4452.

[13] E. Casanova, K. Davis, E. Golge, G. G¨ oknar, I. Gulea, L. Hart,¨ A. Aljafari, J. Meyer, R. Morais, S. Olayemi, and J. Weber, “XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model,” in Interspeech 2024, 2024, pp. 4978–4982.

[14] J. Betker, “Better speech synthesis through scaling,” arXiv preprint arXiv:2305.07243, 2023.

[15] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millicah, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan, “Flamingo: a visual language model for fewshot learning,” ser. NIPS ’22. Red Hook, NY, USA: Curran As sociates Inc., 2022.

[16] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[17] P. Gage, “A new algorithm for data compression,” C Users J., vol. 12, no. 2, p. 23–38, Feb. 1994.

[18] H. S. Heo, B.-J. Lee, J. Huh, and J. S. Chung, “Clova baseline system for the voxceleb speaker recognition challenge 2020,” arXiv preprint arXiv:2009.14153, 2020.

[19] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.

[20] J. Thienpondt and K. Demuynck, “Ecapa2: A hybrid neural network architecture and training strategy for robust speaker embeddings,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2023, pp. 1–8.

[21] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “Mls: A large-scale multilingual dataset for speech research,” ArXiv, vol. abs/2012.03411, 2020.

[22] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, N. Calzolari, F. Bechet, P. Blache, K. Choukri,´ C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, and S. Piperidis, Eds. Marseille, France: European Language Resources Association, May 2020, pp. 4218–4222. [Online]. Available: https://aclanthology.org/2020.lrec-1.520/

[23] J. Kong, J. Kim, and J. Bae, “Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 17 022–17 033. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2020/ file/c5d736809766d46260d816d8dbc9eb44-Paper.pdf

[24] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, “Utmos: Utokyo-sarulab system for voicemos challenge 2022,” Interspeech 2022, 2022.