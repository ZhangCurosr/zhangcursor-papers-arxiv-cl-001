# Ruby-OPD: Enhancing Pronunciation Controllability and Accuracy for Flow-Matching Text-to-Speech via On-Policy Distillation

Rui Hu <sup>ID</sup> <sup>1</sup>, Xiaolong Lin <sup>ID</sup> <sup>1</sup>, Zhenpeng Zhan <sup>ID</sup> <sup>1</sup>

<sup>1</sup> Baidu Inc., Shenzhen, China

hurui05@baidu.com, linxiaolong01@baidu.com

## Abstract

Achieving both high pronunciation accuracy and fine-grained controllability remains a central challenge in Japanese text-tospeech (TTS), as Japanese is an unsegmented language mixing logographic Kanji and phonographic Kana with extensive polyphony. Traditional cascaded systems convert text into phoneme sequences via a front-end, ensuring pronunciation control but discarding lexical semantics and prosodic cues in surface orthography. Conversely, recent zero-shot TTS models directly consume raw orthographic text to improve naturalness, yet suffer from polyphone errors and lack explicit pronunciation control. To bridge this gap, we propose a two-stage framework for flow-matching TTS. First, we investigate Ruby text representations with orthography perturbation to jointly preserve surface forms and target pronunciations for reliable control. Second, we introduce On-Policy Distillation (OPD) on large-scale text-only corpora to enhance pronunciation accuracy on raw orthographic text without paired speech. On the Joyo-Kanji-Yomi benchmark, our model achieves 97.13% Accuracy, 2.02% Target PER, 0.70% Sentence PER, and 4.21% Sentence CER under raw text while retaining strong controllability.

Index Terms: text-to-speech, on-policy distillation, pronunciation accuracy, controllability, Japanese

## 1. Introduction

Traditional neural text-to-speech (TTS) systems [1, 2, 3] follow a cascaded pipeline comprising a text front-end, an acoustic model, and a vocoder, where the front-end extracts phonemes and prosodic features from raw text. Recently, scaling model capacity and data has driven the rise of zero-shot TTS across both autoregressive (AR) [4, 5, 6, 7, 8] and non-autoregressive (NAR) [9, 10, 11, 12] paradigms. These models bypass complex front-ends and directly consume raw orthographic text, leveraging surface lexical semantics to synthesize natural speech.

In Japanese TTS, however, directly consuming orthographic text poses severe challenges for pronunciation accuracy and controllability. As an unsegmented language, Japanese combines logograms (Kanji) and phonograms (Kana). While Kana maps almost deterministically to pronunciation, Kanji pronunciations vary drastically with morphology and context. For instance, <sup>秋</sup> (autumn) takes the native pronunciation (Kun’yomi) aki standalone, the Sino-Japanese pronunciation (On’yomi) shu¯ in compounds such as <sup>晩秋</sup> (late autumn), and the metaphorical pronunciation (Gikun) toki (<sup>時</sup>, moment) <sub>in classical expressions such as</sub> 此れ誠に危急存亡の秋なり (this is truly a critical moment of survival).

Early Japanese TTS relied on morphological analyzers with dictionaries [13, 14, 15] and pretrained language models for polyphone disambiguation and accent prediction [16, 17, 18, 19]. Although phonemic inputs offer direct pronunciation control, stripping away Kanji orthography discards semantic and prosodic cues, limiting naturalness [20]. Conversely, recent end-to-end Japanese TTS models train directly on orthographic text. Irodori-TTS [21] models continuous acoustic latents [22] via conditional flow matching with a Diffusion Transformer (DiT) [23], while Sarashina2.2-TTS [24] uses an AR architecture and mitigates Kanji mispronunciations via local Kana substitution and offline speech synthesis over polyphones. Yet, this still sacrifices orthographic semantics and incurs costly offpolicy synthesis.

In this paper, we propose a two-stage framework to enhance pronunciation controllability and accuracy in flow-matching Japanese TTS. First, we introduce Ruby text representations coupled with orthography perturbation to jointly preserve logographic surface forms and target pronunciations for precise pronunciation control. Second, we propose cross-condition On-Policy Distillation (OPD) on large-scale text-only corpora to improve raw-text pronunciation accuracy without paired speech. Experiments on the Joyo-Kanji-Yomi benchmark show that our method substantially improves both pronunciation controllability and accuracy, achieving 97.13% Accuracy, 2.02% Target PER, 0.70% Sentence PER, and 4.21% Sentence CER.

## 2. Method

## 2.1. Ruby text representation

Ruby is widely used in Japanese books, webpages, and subtitles to annotate the pronunciations of proper nouns, polyphones, rare words, or stylistically specialized terms. Visually, Ruby attaches small Kana characters above or beside target Kanji, simultaneously conveying both orthographic surface form and phonemic pronunciation within a single visual unit.

When linearizing two-dimensional Ruby typography into a sequential text input for a TTS text encoder, we explore two complementary representations:

• Implicit Ruby: Target pronunciations are appended directly after annotated Kanji words using parentheses. This aligns naturally with human users’ everyday writing and interactive editing habits when manually annotating occasional rare words or proper nouns on the fly;

• Explicit Ruby: One limitation of implicit Ruby is the lack of explicit left boundaries for the target span, which can sometimes result in ambiguous annotations in unsegmented text. Explicit Ruby, however, employs explicit boundary markers for both the target span and its pronunciation, making it well suited for unambiguous and precise pronunciation annotation.

An example is illustrated in Table 1. The compound word <sup>超電磁砲</sup> is canonically pronounced as $5 \times 5 \% = 1 \times 1 \times 5$ (chodenjih¯ o¯). However, in the title $\ge \dot { \mathcal { D } } \gtrsim$ 科学の超電磁砲 <sub>(</sub>A Certain Scientific Railgun), it takes the stylized pronunciation $V - \pi \pi \cdot r$ (rerugan¯ ). Under implicit Ruby, because this nonstandard pronunciation has no morphological correspondence to individual Kanji characters, the left boundary of the annotated span becomes ambiguous, where it could be interpreted as annotating either <sup>電磁砲</sup> or the entire <sup>超電磁砲</sup>. In contrast, explicit Ruby eliminates such scope ambiguity by introducing explicit boundary markers.

Table 1: An example of orthographic text, implicit Ruby, and explicit Ruby. ${ | \ , \quad \langle } ,$ and 》 denote the start of the target word, and the start and end ofthe target pronunciation in explicit Ruby, respectively.  
Orthographic text  
とある科学の超電磁砲  
Implicit Ruby  
とある科学 $( x > - 1 ) ^ { 3 } < )$ の超電磁砲 $( V - \pi \pi y ^ { \circ } )$   
Explicit Ruby  
とある｜科学《かがく》の｜超電磁砲 $\{ V - \pi \neq j \} \cup$

## 2.2. Training data construction

Constructing paired Ruby training data for supervised finetuning (SFT) solely via Grapheme-to-Phoneme (G2P) tools is prone to polyphone disambiguation errors that may mismatch the actual pronunciations uttered in the paired speech. To construct precisely aligned Ruby training data, we employ a crossmodal verification approach that leverages information from both text and audio modalities. Specifically, we first use a phoneme recognition model to transcribe the input speech into a pure Kana sequence, capturing the actual pronunciation of the utterance. Then, for the corresponding orthographic text, we use UniDic [14], a widely used morphological dictionary, to construct a word lattice encompassing all valid morphological segmentations and candidate pronunciations. Finally, we apply dynamic programming to retrieve the alignment path through the lattice that exactly matches the recognized Kana transcription, precisely pinpointing the character span and true contextual pronunciation of every Kanji word unit in the utterance.

In practical applications, users exhibit diverse annotation habits across different granularities, ranging from short single words to longer multi-word phrases, and may feed either completely unannotated orthographic text or partially annotated Ruby text where only selected words are annotated. To simulate these scenarios, rather than enumerating fixed full-annotation paths after the alignment step, we store the aligned Ruby units and adopt a dynamic multi-granularity sampling strategy during training. First, at the utterance level, we sample from a preset probability distribution to determine whether an utterance is formatted as unannotated, partially annotated, or fully annotated. Specifically, under the partially annotated mode, each Ruby unit is converted into Ruby format with a certain probability and otherwise reverts to raw orthographic text. Furthermore, adjacent Ruby units that are simultaneously selected for annotation are merged with a certain probability into a single contiguous Ruby unit. These data augmentation strategies enable the model to seamlessly handle unannotated raw text, sparse local annotations, and annotations of varying granularities.

## 2.3. Orthography perturbation

In our preliminary experiments, we found that directly training the model with Ruby text failed to make it properly interpret the Ruby format or follow the pronunciation annotations at inference time. Instead, the model often read aloud both the surface form and the annotated pronunciation within Ruby units. We conjecture that because the base model was pretrained on raw orthographic text, it can already infer canonical pronunciations for most Kanji solely from their surface glyphs, leaving little incentive to attend to the pronunciation cues provided in the Ruby annotations.

To address this issue, we keep the target pronunciation annotation intact while stochastically perturbing the surface form of each Ruby unit during training via three complementary operations:

• Kanji masking: replaces partial or full Kanji spans in the surface form with special mask tokens. Partial masking encourages the model to align unmasked characters with the target pronunciation to infer span boundaries in implicit Ruby, whereas full masking forces the model to rely on the pronunciation annotation when surface glyph cues are absent;

• Random substitution: replaces the surface form in explicit Ruby with random characters or unrelated words of arbitrary length, breaking character-to-mora correspondence and compelling the model to follow the annotated pronunciation regardless of surface glyphs;

• Semantic substitution: replaces the surface form in explicit Ruby with semantically similar Kanji words that have distinctly different canonical pronunciations, retrieved via text embeddings over UniDic. This design specifically simulates the aforementioned Gikun phenomenon, where the surface form is semantically aligned with both the annotated pronunciation and surrounding context yet phonemically unrelated to its canonical pronunciation. For example, the Kana word $\varkappa - \varkappa \Vdash$ (world), pronounced as warudo¯ , may be assigned a semantically similar yet orthographically and phonemically distinct surface form <sup>世</sup>界, canonically pronounced as sekai. This strategy ensures that the model strictly adheres to the annotated pronunciation while leveraging surface orthography for semantic and prosodic cues.

## 2.4. On-policy distillation

After SFT with Ruby data, the model achieves high pronunciation accuracy when provided with full Ruby annotations. However, as noted earlier, real-world inputs are often unannotated orthographic text or only partially annotated, leaving unannotated spans vulnerable to polyphone disambiguation errors. Attaching an external G2P module at inference time or synthesizing paired text-speech data over polyphones in an off-policy manner [24] are viable workarounds, but they introduce extra pipeline modules or costly offline synthesis steps. Instead, we propose an On-Policy Distillation (OPD) approach for flow-matching TTS that relies solely on paired (orthographic text, Ruby text) data without requiring parallel audio, improving pronunciation accuracy when Ruby annotations are absent.

In conditional flow matching [25], a neural network $v _ { \theta } ( x _ { t } , t , c )$ learns a time-dependent velocity field that transports standard Gaussian noise $x _ { 1 } \sim \mathcal { N } ( 0 , I )$ at t = 1 to clean acoustic latents $x _ { 0 }$ at $t = 0$ conditioned on text c. With paired speech $x _ { 0 } ,$ , noisy states are constructed via linear interpolation $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ , and the model is trained by minimizing:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \Big [ \big | | v _ { \theta } ( x _ { t } , t , c ) - ( x _ { 1 } - x _ { 0 } ) \big | \big | _ { 2 } ^ { 2 } \Big ] . } \end{array}
$$

In our OPD framework, the SFT model serves as a frozen teacher $v _ { \phi }$ conditioned on full Ruby text $c _ { \mathrm { r u b y } }$ , while the student $v _ { \theta } .$ , initialized from $v _ { \phi }$ , is conditioned on unannotated orthographic text $c _ { \mathrm { r a w } }$ (other conditioning is omitted for brevity). Because text-only samples lack ground-truth speech latents x<sub>0</sub>, we generate the intermediate state $x _ { t }$ online via student rollout [26, 27]. Starting from pure Gaussian noise $x _ { 1 } \sim \mathcal { N } ( 0 , I )$ with target sequence length predicted by the teacher’s duration predictor, we integrate the student’s ordinary differential equation (ODE) trajectory up to a randomly sampled timestep t without gradients to obtain the on-policy state $x _ { t }$ . At $( x _ { t } , t )$ , the student is trained to match the teacher’s velocity field:

$$
\mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { t , x _ { t } } \Bigl [ \big | | v _ { \theta } ( x _ { t } , t , c _ { \mathrm { r a w } } ) - \mathrm { s g } \bigl [ v _ { \phi } ( x _ { t } , t , c _ { \mathrm { r u b y } } ) \bigr ] \bigr | \bigr ] _ { 2 } ^ { 2 } \Bigr ] ,
$$

where sg[·] denotes the stop-gradient operator, alongside a Huber loss aligning the student’s duration predictor with the teacher’s. Since the rollout starts from pure noise, $x _ { t }$ contains zero phonemic leakage from ground-truth audio, forcing the student to disambiguate polyphones from context to match the teacher’s trajectory. During training, we also interleave the paired speech data used in the SFT stage to anchor acoustic naturalness and retain Ruby controllability.

## 3. Experimental setup

## 3.1. Data

For the SFT stage, we use the Japanese subset of the opensource multilingual speech dataset Emilia [28]. To ensure strict consistency between pronunciation annotations and actual speech content, we transcribe all utterances using kanawhisper<sup>1</sup> and filter for utterances whose Kana transcriptions exactly match the word lattice constructed via UniDic [14]. Combined with duration filtering (0.5 ∼ 30 s), we obtain approximately 774 hours of high-quality aligned data.

For subsequent text-only OPD, we follow [19] by using entries from UniDic [14] as seed words and leveraging Gemini 3.1 Pro to synthesize text-pronunciation pairs. For each monophonic and polyphonic entry, we synthesize 5 and 10 contextual sentences, respectively. Specifically, rather than asking the LLM to directly output Ruby annotations, we prompt it to generate sentence-level Kana pronunciation sequences and apply the same lattice alignment method used for Emilia. This yields approximately 680k monophonic sentences and 100k polyphonic sentences.

## 3.2. Implementation details

We build upon the open-source Japanese flow-matching zeroshot TTS model Irodori-TTS-v4.1-Small $[ 2 1 ] ^ { 2 }$ as our base model. It initializes its text encoder with a pretrained bidirectional masked language model [29] and predicts the conditional velocity field of 25 Hz, 32-dimensional continuous DACVAE latents [22] via a DiT backbone.

In the SFT stage, utterance-level annotation modes are sampled with probabilities of 50% full Ruby, 30% partial Ruby, in which each Ruby unit is annotated with a probability of 0.5, and 20% unannotated raw text. Each annotated Ruby unit is formatted as implicit or explicit Ruby with an equal probability of 0.5, and adjacent annotated Ruby units are merged with a probability of 0.5, up to 3 merges and 8 characters. For orthography perturbation, we use Qwen3-Embedding-0.6B<sup>3</sup> to retrieve semantic substitution candidates over UniDic [14], and each annotated surface span undergoes full-span Kanji masking with a probability of 0.2, partial character-level masking with a probability of 0.25, semantic substitution with a probability of 0.2, and random substitution with a probability of 0.15. The model is trained for 50k steps with a global batch size of 256 using AdamW, with a learning rate of $1 \times 1 0 ^ { - 4 }$ for the main model and $1 \times 1 0 ^ { - 5 }$ for the pretrained text encoder.

In the OPD stage, we freeze the SFT model as the teacher, which receives 100% full Ruby text, and train the student for 50k steps with a global batch size of 256, a learning rate of $5 \times 1 0 ^ { - 5 }$ , a text encoder learning rate of $2 . 5 \times 1 0 ^ { - 5 }$ , and 8 rollout steps. Each training step samples text-only data with an 80% probability, equally weighted between monophonic and polyphonic subsets with 100% raw text input for the student, and paired speech data with a 20% probability, where student inputs are sampled using 80% raw text, 10% partial Ruby, and 10% full Ruby with the same orthography perturbation settings as in the SFT stage.

## 3.3. Evaluation metrics

To evaluate pronunciation accuracy and controllability, we conduct evaluations on a revised version of the Joyo-Kanji-Yomi benchmark (JKYB) [24]<sup>4</sup>. It comprises 13, 536 sentences covering 2,136 Joyo Kanji (regular-use Kanji) and their 4,512 official pronunciations specified by the Agency for Cultural Affairs of Japan. JKYB provides the following metrics:

• Accuracy: Percentage of evaluated Kanji pronunciations that exactly match the ground truth;

• Target PER: Kana character error rate (Kana-CER) against the ground truth of the evaluated Kanji. Values are capped to 100%;

• Sentence PER: Kana-CER for the full-sentence pronunciation. Values are capped to 100%;

• Sentence CER: Standard CER between the ASR transcription of the utterance and the source text. Values are capped to 100%.

We also evaluate speech quality using UTMOSv2 [30] and DNSMOS [31], and voice cloning speaker similarity using CAM++ [32] and ERes2Net [33]. During inference, reference audio prompts are randomly sampled from the JVS corpus [34], which consists of 100 speakers with 100 normal-style utterances per speaker.

## 4. Experimental results

## 4.1. Pronunciation controllability

We first investigate the effectiveness of Ruby representations for pronunciation control. Using the reference text and groundtruth pronunciations from JKYB, we construct oracle Ruby inputs at both the target-word and full-sentence levels.

Table 2: Comparison ofpronunciation accuracy, speech quality, and voice cloning performance on the Joyo-Kanji-Yomi benchmark (JKYB) using JVS reference audio prompts. All models directly consume unannotated raw orthographic text at inference time.
<table><tr><td rowspan="2">Model</td><td colspan="4">Pronunciation Accuracy</td><td colspan="2">Speech Quality</td><td colspan="2">Voice Cloning</td></tr><tr><td>Accuracy</td><td>Target PER</td><td>Sentence PER</td><td>Sentence CER</td><td>UTMOSv2</td><td>DNSMOS</td><td>CAM++</td><td>ERes2Net</td></tr><tr><td>Qwen3-TTS 1.7B Base [6]</td><td>72.10%</td><td>23.68%</td><td>4.38%</td><td>6.52%</td><td>3.50</td><td>3.23</td><td>0.757</td><td>0.779</td></tr><tr><td>FireRedTTS3 [8]</td><td>73.29%</td><td>22.35%</td><td>4.04%</td><td>6.12%</td><td>3.36</td><td>3.26</td><td>0.833</td><td>0.832</td></tr><tr><td>OmniVoice [12]</td><td>83.66%</td><td>13.24%</td><td>2.91%</td><td>5.18%</td><td>3.19</td><td>3.21</td><td>0.797</td><td>0.786</td></tr><tr><td>FishAudio S2 Pro [7]</td><td>89.90%</td><td>8.27%</td><td>1.52%</td><td>4.77%</td><td>3.30</td><td>3.31</td><td>0.735</td><td>0.726</td></tr><tr><td>Sarashina2.2-TTS [24]</td><td>90.42%</td><td>7.69%</td><td>3.15%</td><td>6.26%</td><td>3.42</td><td>3.33</td><td>0.819</td><td>0.825</td></tr><tr><td>Irodori-Base [21]</td><td>94.11%</td><td>4.68%</td><td>0.97%</td><td>4.04%</td><td>3.21</td><td>3.23</td><td>0.728</td><td>0.697</td></tr><tr><td>Irodori-Ruby-SFT</td><td>92.32%</td><td>6.05%</td><td>1.32%</td><td>4.40%</td><td>3.18</td><td>3.22</td><td>0.703</td><td>0.663</td></tr><tr><td>Irodori-Ruby-OPD</td><td>97.13%</td><td>2.02%</td><td>0.70%</td><td>4.21%</td><td>3.19</td><td>3.23</td><td>0.703</td><td>0.663</td></tr></table>

As shown in Table 3, for Irodori-Base, directly replacing the target Kanji word with Kana (Target Kana) effectively improves target pronunciation accuracy, whereas replacing the entire sentence with Kana (Full Kana) causes a notable performance drop at both the target-word and sentence levels, yielding an even higher Sentence PER than unannotated raw text. This confirms that stripping away Kanji orthography across an unsegmented sentence eliminates crucial lexical boundary and semantic cues. In contrast, by jointly preserving surface Kanji and phonemic annotations, both Irodori-Ruby-SFT and Irodori-Ruby-OPD avoid this degradation: expanding annotations from the target word (Target) to the full sentence (Full) maintains high target accuracy while further reducing Sentence PER.

Moreover, explicit Ruby (Expl.) consistently outperforms implicit Ruby (Impl.) across both SFT and OPD models under both target-word and full-sentence settings. While implicit Ruby already provides strong controllability for convenient user interaction, the explicit boundary markers in explicit Ruby effectively eliminate left-boundary scope ambiguity, particularly for multi-character words with non-compositional pronunciations.

Table 3: Results of pronunciation controllability for Irodori-Base (Base), Irodori-Ruby-SFT (SFT), and Irodori-Ruby-OPD (OPD). Target and Full indicate target-word orfull-sentence Kana replacement or Ruby annotation. Impl. and Expl. denote implicit and explicit Ruby.
<table><tr><td>Model</td><td>Input</td><td>Accuracy</td><td>Target PER</td><td>Sentence PER</td></tr><tr><td>Base</td><td>Raw text</td><td>94.11%</td><td>4.68%</td><td>0.97%</td></tr><tr><td>Base</td><td>Target Kana</td><td>97.89%</td><td>1.37%</td><td>0.56%</td></tr><tr><td>Base</td><td>Full Kana</td><td>95.40%</td><td>2.89%</td><td>1.06%</td></tr><tr><td>SFT</td><td>Target Impl.</td><td>97.56%</td><td>1.71%</td><td>0.75%</td></tr><tr><td>SFT</td><td>Target Expl.</td><td>98.12%</td><td>1.26%</td><td>0.72%</td></tr><tr><td>SFT</td><td>Full Impl.</td><td>98.05%</td><td>1.33%</td><td>0.53%</td></tr><tr><td>SFT</td><td>Full Expl.</td><td>98.20%</td><td>1.19%</td><td>0.49%</td></tr><tr><td>OPD</td><td>Target Impl.</td><td>98.09%</td><td>1.30%</td><td>0.57%</td></tr><tr><td>OPD</td><td>Target Expl.</td><td>98.33%</td><td>1.11%</td><td>0.56%</td></tr><tr><td>OPD</td><td>Full Impl.</td><td>98.14%</td><td>1.23%</td><td>0.45%</td></tr><tr><td>OPD</td><td>Full Expl.</td><td>98.24%</td><td>1.15%</td><td>0.42%</td></tr></table>

## 4.2. Main results

Table 2 compares our models (Irodori-Ruby-SFT and Irodori-Ruby-OPD) against the base model (Irodori-Base) and public state-of-the-art (SOTA) baselines across pronunciation accuracy, speech quality, and voice cloning. At inference time, all models directly consume raw orthographic text without any Ruby annotations. First, while general multilingual TTS models, such as Qwen3-TTS, FireRedTTS3, OmniVoice, and FishAudio S2 Pro, achieve leading performance in speech quality and voice cloning, they still struggle with Japanese polyphonic Kanji pronunciation, exhibiting relatively low Accuracy (72.10%–89.90%) and high Target PER (8.27%–23.68%). In contrast, Japanese-specialized models such as Sarashina2.2- TTS and Irodori-Base benefit from large-scale Japanese pretraining and demonstrate a clear advantage in pronunciation accuracy, with Irodori-Base achieving 94.11% Accuracy and the lowest Sentence CER (4.04%).

Compared with Irodori-Base, Irodori-Ruby-SFT, fine-tuned solely on Emilia Ruby data, exhibits a slight drop on raw text (92.32% Accuracy and 6.05% Target PER) and minor shifts in acoustic and speaker metrics, reflecting the limited scale and modest recording quality of the SFT corpus as well as capacity allocated to learning Ruby conditioning. After large-scale text-only OPD, however, Irodori-Ruby-OPD achieves substantial gains on raw text without additional paired speech, boosting Accuracy to 97.13% (+4.81% over its teacher and +3.02% over Irodori-Base), cutting Target PER by more than half vs. Irodori-Base (4.68% → 2.02%), and achieving the lowest Sentence PER (0.70%). Furthermore, Irodori-Ruby-OPD consistently surpasses Irodori-Ruby-SFT across all four Ruby control modes (Table 3) while maintaining speech quality and voice cloning on par with its teacher. In particular, under targetonly annotation (Target Impl. and Target Expl.), Irodori-Ruby-OPD notably reduces Sentence PER compared with Irodori-Ruby-SFT (0.56%–0.57% vs. 0.72%–0.75%), as the remaining unannotated Kanji words in the utterance simultaneously benefit from the enhanced raw-text pronunciation capability distilled during OPD.

## 5. Conclusion

In this paper, we proposed a two-stage framework for flowmatching Japanese TTS that first introduces implicit and explicit Ruby representations coupled with orthography perturbation during SFT to enable fine-grained pronunciation control while preserving surface Kanji semantics. Building upon the Ruby-capable SFT teacher, we further introduced cross-condition On-Policy Distillation (OPD) on largescale LLM-synthesized text-pronunciation pairs, transferring context-dependent Kanji pronunciation knowledge into rawtext inference without requiring paired speech. Experimental results on the Joyo-Kanji-Yomi benchmark demonstrate that Irodori-Ruby-OPD achieves state-of-the-art pronunciation accuracy on unannotated orthographic text while providing flexible and reliable Ruby-based pronunciation control.

## 6. References

[1] Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio, Q. Le, Y. Agiomyrgiannakis, R. Clark, and R. A. Saurous, “Tacotron: Towards End-to-End Speech Synthesis,” in Proc. Interspeech, 2017, pp. 4006–4010.

[2] Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, “Fastspeech 2: Fast and high-quality end-to-end text to speech,” in International Conference on Learning Representations (ICLR), 2021.

[3] J. Kim, J. Kong, and J. Son, “Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech,” in Proceedings ofthe 38th International Conference on Machine Learning (ICML), 2021, pp. 5530–5540.

[4] Z. Du, Q. Chen, S. Zhang, K. Hu, H. Lu, Y. Yang, H. Hu, S. Zheng, Y. Gu, Z. Ma et al., “Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens,” arXiv preprint arXiv:2407.05407, 2024.

[5] Y. Zhou, G. Zeng, X. Liu, X. Li, R. Yu, Z. Wang, R. Ye, W. Sun, J. Gui, K. Li et al., “Voxcpm: Tokenizer-free tts for context-aware speech generation and true-to-life voice cloning,” arXiv preprint arXiv:2509.24650, 2025.

[6] H. Hu, X. Zhu, T. He, D. Guo, B. Zhang, X. Wang, Z. Guo, Z. Jiang, H. Hao, Z. Guo et al., “Qwen3-tts technical report,” arXiv preprint arXiv:2601.15621, 2026.

[7] S. Liao, Y. Wang, S. Liu, Y. Cheng, R. Zhang, T. Li, S. Li, Y. Zheng, X. Liu, Q. Wang et al., “Fish audio s2 technical report,” arXiv preprint arXiv:2603.08823, 2026.

[8] F. Shen, K. Xie, Y. Wu, Z. Dai, Y. Han, J. Li, X. Geng, F. Xie, L. Xie, X. Tang et al., “Fireredtts3: Unified speech generation and editing with semantically enriched speech representations,” arXiv preprint arXiv:2608.17492, 2026.

[9] M. Le, A. Vyas, B. Shi, B. Karrer, L. Sari, R. Moritz, M. Williamson, V. Manohar, Y. Adi, J. Mahadeokar, and W.-N. Hsu, “Voicebox: Text-guided multilingual universal speech generation at scale,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[10] S. E. Eskimez, X. Wang, M. Thakker, C. Li, C.-H. Tsai, Z. Xiao, H. Yang, Z. Zhu, M. Tang, X. Tan et al., “E2 TTS: Embarrassingly easy fully non-autoregressive zero-shot TTS,” in IEEE Spo ken Language Technology Workshop (SLT), 2024, pp. 682–689.

[11] Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. Zhao, K. Yu, and X. Chen, “F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025, pp. 6255–6271.

[12] H. Zhu, L. Ye, W. Kang, Z. Yao, L. Guo, F. Kuang, Z. Han, W. Zhuang, L. Lin, and D. Povey, “Omnivoice: Towards omnilingual zero-shot text-to-speech with diffusion language models,” arXiv preprint arXiv:2604.00688, 2026.

[13] T. Kudo, “Mecab: Yet another part-of-speech and morphological analyzer,” http://mecab. sourceforge. net/, 2005.

[14] Y. Den, T. Ogiso, H. Ogura, A. Yamada, N. Minematsu, K. Uchimoto, and H. Koiso, “The development of an electronic dictionary for morphological analysis and its application to japanese corpus linguistics,” in Japanese Linguistics, vol. 22, 2007, pp. 101–123.

[15] H. Morita, D. Kawahara, and S. Kurohashi, “Morphological analysis for unsegmented languages using recurrent neural network language model,” in Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, 2015, pp. 2292– 2297.

[16] R. Hida, M. Hamada, C. Kamada, E. Tsunoo, T. Sekiya, and T. Kumakura, “Polyphone disambiguation and accent prediction using pre-trained language models in japanese TTS front-end,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 7132–7136.

[17] B. Park, R. Yamamoto, and K. Tachibana, “A Unified Accent Estimation Method Based on Multi-Task Learning for Japanese Textto-Speech,” in Proc. Interspeech, 2022, pp. 1931–1935.

[18] K. Kurihara and M. Sano, “Enhancing japanese text-to-speech accuracy with a novel combination transformer-BERT-based G2P: Integrating pronunciation dictionaries and accent sandhi,” in Proc. Interspeech, 2024.

[19] R. Hu, Z. Zhan, and X. Lin, “Dictionary-constrained graphemeto-phoneme for unsegmented languages from LLM-annotated data,” arXiv preprint arXiv:2609.19805, 2026.

[20] Y. Yasuda, X. Wang, S. Takaki, and J. Yamagishi, “Investigation of learning abilities on linguistic features in sequence-to-sequence text-to-speech synthesis,” Computer Speech & Language, vol. 58, p. 101008, 2019.

[21] C. Arata, “Irodori-TTS: A flow matching-based text-to-speech model with emoji-driven style control,” https://github.com/ Aratako/Irodori-TTS, 2026.

[22] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “High-fidelity audio compression with improved RVQGAN,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[23] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4195–4205.

[24] L. Liu, S. Zhu, K. Washizaki, R. Yoneyama, H. Jeon, M. Zhao, Y. Fujita, H. Shi, N. Yoshida, Y. Gao et al., “Sarashina2.2- TTS: Tackling kanji polyphony in japanese speech generation via data scaling and targeted data synthesis,” arXiv preprint arXiv:2606.25369, 2026.

[25] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in International Conference on Learning Representations (ICLR), 2023.

[26] R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem, “On-policy distillation of language models: Learning from self-generated mistakes,” in International Conference on Learning Representations (ICLR), 2024.

[27] N. Shaul, C. Liu, A. Vahdat, and J. Berner, “Parallel decoding distillation for fast image and video generation,” arXiv preprint arXiv:2607.26004, 2026.

[28] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi et al., “Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation,” in IEEE Spoken Language Technology Workshop (SLT), 2024.

[29] B. Warner, A. Chaffin, B. Clavie, O. Weller, O. Hallstr´ om,¨ S. Taghadouini, A. Gallagher, R. Biswas, F. Ladhak, T. Aarsen et al., “Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference,” arXiv preprint arXiv:2412.13663, 2024.

[30] K. Baba, W. Nakata, Y. Saito, and H. Saruwatari, “The T05 system for the VoiceMOS challenge 2024: Transfer learning from deep image classifier to naturalness MOS prediction of high-quality synthetic speech,” in IEEE Spoken Language Technology Workshop (SLT), 2024.

[31] C. K. A. Reddy, V. Gopal, and R. Cutler, “DNSMOS: A nonintrusive perceptual objective speech quality metric to evaluate noise suppressors,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 6493– 6497.

[32] H. Wang, S. Zheng, Y. Chen, L. Cheng, and Q. Chen, “CAM++: A fast and efficient network for speaker verification using contextaware masking,” in Proc. Interspeech, 2023, pp. 5331–5335.

[33] Y. Chen, S. Zheng, H. Wang, L. Cheng, Q. Chen, and J. Qi, “An enhanced Res2Net with local and global feature fusion for speaker verification,” in Proc. Interspeech, 2023, pp. 2238–2242.

[34] S. Takamichi, K. Mitsui, Y. Saito, T. Koriyama, N. Tanji, and H. Saruwatari, “JVS corpus: free japanese multi-speaker voice corpus,” arXiv preprint arXiv:1908.06248, 2019.