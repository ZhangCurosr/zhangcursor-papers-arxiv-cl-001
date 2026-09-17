# T-SANDHI: Tone Sandhi-aware Adaptive Network with Decoupled Hybrid Injection for Low-resource Taiwanese Hokkien Speech Recognition

Hung-Yang Sung<sup>∗</sup>, Chien-Chun Wang<sup>†</sup>, Tien-Hong Lo<sup>∗</sup>, Yu-Sheng Tsao<sup>‡</sup>, Yung-Chang Hsu<sup>‡</sup>, Berlin Chen<sup>∗</sup>

<sup>∗</sup>Department of Computer Science and Information Engineering, National Taiwan Normal University, Taiwan

<sup>†</sup>E.SUN Financial Holding Co., Ltd., Taiwan

<sup>‡</sup>EZAI, Taiwan

Abstract—In Taiwanese Hokkien automatic speech recognition (ASR), prior studies often treat tone sandhi as a major challenge under the assumption that models fail to process implicit phonological variations. However, our experiments on Taiwanese Hokkien reveal that speech foundation models actually handle tone sandhi variations effectively, and the real performance bottleneck stems from a localized confusion between these variations and retained citation tones. To address this, we propose T-SANDHI to explicitly decouple surface acoustics from underlying lexical intent on top of a frozen Whisper backbone. Using a lexicon-guided multi-task learning structure driven by textderived pseudo labels, our lightweight hybrid injection module integrates independent citation and sandhi phonetic streams via dynamic gating. Extensive evaluation on the TAT-MOE corpus and two blind test sets demonstrates that this explicit disentanglement effectively resolves tonal mapping confusion, outperforming baselines with strict parameter efficiency.

Index Terms—automatic speech recognition, low resource, Taiwanese Hokkien, tone sandhi, lexicon-guided

## I. INTRODUCTION

A fundamental assumption in most automatic speech recognition (ASR) systems is a consistent, reliable mapping between surface acoustics and underlying lexical units [1]–[7]. However, this mapping relationship becomes less straightforward in tonal languages with complex phonological variations. A prime example is Taiwanese Hokkien (Taiwanese) [8]–[14]. Unlike Mandarin [15]–[17], which maintains relatively static tonal mappings, Taiwanese Hokkien features a complex web of tone sandhi that applies to syllables based on their position within syntactic units [13], [18], [19]. In continuous speech, tone sandhi applies to every non-final syllable, meaning that the actual surface tone frequently shifts based on the grammatical context [13], [20]. Consequently, a syllable’s underlying dictionary pronunciation (the citation tone) only occurs at specific morphosyntactically defined boundaries, while all other syllables are realized with altered tones (the sandhi tone) [20]–[22]. As illustrated in Fig. 1, when a speaker utters the pronoun “You” (Taiwanese Hanzi<sup>1</sup>: <sup>你</sup>), the citation syllable is “l´ı” (Tone 2) and the sandhi tone is Tone 1.

Current ASR systems typically rely on models to implicitly internalize these complex phonological mappings through the final transcription loss [14], [21], [23]. While scaled parameters in data-abundant scenarios can partially absorb such variations, this implicit learning paradigm often faces substantial challenges under low-resource constraints [24]–[27]. To address these constraints, adopting parameter-efficient finetuning (PEFT) techniques, such as AdaLoRA, has emerged as a standard and highly effective approach to adapt speech foundation models to various low-resource languages [28]– [30]. Following this established paradigm, we build our study upon a parameter-efficient framework to explore foundation model behavior. Prior study focused on utilizing syllables with citation tones as target labels, evaluating various selfsupervised learning (SSL) models with connectionist temporal classification (CTC) loss [13]. Their findings indicate that tone sandhi represent a major source of character substitution errors, frequently confusing phonetically similar phones in the predictions, thereby suggesting that an independent tonal handling mechanism would be beneficial for future efforts [13]. However, it remains unclear whether modern speech foundation models such as Whisper are similarly affected by tone sandhi variations.

![](images/f38a8ec0514d2c235b7236353bd4090faa52f19e3f95378cb2cd02b3e56e8ccd.jpg)  
Fig. 1. Illustration of the acoustic-to-label discrepancy in Taiwanese Hokkien continuous speech. Due to the tone sandhi system, the actual surface tone (sandhi tone, highlighted in orange) frequently shifts away from its dictionary pronunciation (citation tone). For example, in the phrase “Today where do you want to go”, the pronoun “l´ı” shifts from Tone 2 to Tone 1, whereas only the sentence-final syllable retains its original citation tone (highlighted in blue).

To address this gap, we conduct an empirical analysis to evaluate baseline predictions across various model scales. Our findings reveal a counter-intuitive phenomenon: while modern foundation models process tone sandhi variations with reasonable proficiency, their performance bottlenecks significantly on syllables that genuinely retain their citation tones. As we will detail in Section II, this localized performance gap is primarily driven by an extreme volume imbalance in continuous speech, where distinct underlying citation categories frequently conflate into identical surface realizations. These empirical findings indicate that explicitly incorporating both citation and sandhi syllable information can serve as a viable path to reduce this localized performance gap and enhance overall speech recognition accuracy.

![](images/456ced62892eb4c9ae09fe15102b4941316c1d650a728f19449159bc1756e6e3.jpg)  
(a) Overall Tone Distribution

![](images/567a72fc75aa1f92551d2c1841558e7448baf0afbc41a617c587ab66c4075f1c.jpg)  
(b) Tone Transition Matrix  
Fig. 2. Quantitative analysis of tonal distribution and phonetic variations across TAT-MOE corpus. (a) Overall distribution of syllables categorized by their citation forms (blue bars) and sandhi forms (orange bars). (b) Tone transition matrix detailing the absolute counts of syllables mapping from their citation tone to their realized tone.

TABLE I  
STATISTICS OF THE TAT-MOE DATASET ACROSS TRAINING, DEVELOPMENT AND TEST SPLITS, INCLUDING THE NUMBER OF SPEAKERS, UTTERANCES AND TOTAL DURATION IN HOURS.
<table><tr><td>Split</td><td># Speakers</td><td>s # Utterances</td><td>Duration (Hours)</td></tr><tr><td>Training</td><td>328</td><td>86,072</td><td>153.33</td></tr><tr><td>Development</td><td>58</td><td>16,357</td><td>28.60</td></tr><tr><td>Test</td><td>54</td><td>15,962</td><td>26.28</td></tr><tr><td>Total</td><td>440</td><td>118,391</td><td>208.21</td></tr></table>

To bridge this critical gap, we propose T-SANDHI, a Tone Sandhi-aware Adaptive Network designed specifically to resolve tonal mapping ambiguity. Instead of treating the acoustic-to-label mapping as a monolithic black box, we introduce a parameter-efficient Decoupled Hybrid Injection mechanism on top of a frozen Whisper backbone [1], [29], [31]. To overcome data scarcity without reliance on expensive phonetic annotations, we adopt a rule-derived multi-task learning framework driven by automatically generated text-based pseudo labels. Specifically, by leveraging CTC [32]–[36], we explicitly construct two auxiliary streams, where one predicts the citation syllables and the other tracks the sandhi tones. Since sandhi mutations are highly context-dependent, a dynamic gating mechanism is employed to seamlessly integrate these dual phonetic streams. This explicit decoupling allows the decoder to ground its predictions simultaneously on surface

TABLE II  
BASELINE PERFORMANCE ANALYSIS ON THE TAT-MOE CORPUS, EVALUATED ACROSS SANDHI AND CITATION CONTEXTS.
<table><tr><td>Size</td><td colspan="4">Type (Dist.%) CER (%) Precision (%) Recall (%)F1-Score (%)</td></tr><tr><td rowspan="2">Small</td><td>Sandhi (87.36) 15.27</td><td>85.61</td><td>84.73</td><td>85.17</td></tr><tr><td>Citation (12.64) 17.21</td><td>84.25</td><td>82.79</td><td>83.51</td></tr><tr><td rowspan="2">Medium</td><td>Sandhi (87.36) 13.39</td><td>87.32</td><td>86.61</td><td>86.96</td></tr><tr><td>Citation (12.64) 15.21</td><td>85.94</td><td>84.79</td><td>85.36</td></tr><tr><td rowspan="2">Large</td><td>Sandhi (87.36) 12.75</td><td>87.83</td><td>87.25</td><td>87.54</td></tr><tr><td>Citation (12.64) 14.19</td><td>86.85</td><td>85.81</td><td>86.33</td></tr></table>

acoustic variations and underlying lexical intent.  
The main contributions of this study are as follows:

1) Novel Insights on Tone Sandhi: We reveal that while foundation models effectively process tone sandhi, they struggle to map these acoustics back to citation forms due to severe tonal confusion.

2) Explicit Decoupling Architecture: We propose the first ASR framework to disentangle surface acoustics from underlying lexical intent via a dynamically gated, dualstream injection module.

3) High Efficacy with Minimal Overhead: Our approach effectively mitigates mapping ambiguity on TAT-MOE [37], achieving robust performance gains while adding merely 3% to the parameter count of the frozen backbone.

## II. CORPORA AND PHONOLOGICAL ANALYSIS

## A. Corpora

To evaluate under authentic low-resource conditions, we utilized the TAT-MOE corpus [37], alongside two external blind test sets: the FSRC 2020 corpus [38] and the yttd taigi trs corpus [10]. All three datasets feature diverse accents and spontaneous speech. The detailed statistical partitions of the TAT-MOE corpus across the training, development, and test splits are summarized in Table I. Acoustic signals were uniformly resampled to 16 kHz for backbone alignment. As a tonal language, Taiwanese Hokkien exhibits a rich and complex tone sandhi system [20], [21]. In continuous spoken streams, these sandhi mutations apply systematically to virtually every nonfinal syllable within a morphosyntactically defined unit [20], resulting in a prominent discrepancy between citation dictionary forms and contextual surface realizations. A statistical phonological analysis regarding this tonal distribution and its impact on modern ASR is detailed below.

## B. Quantitative Analysis of Tone Sandhi Impact

To investigate the impact of this phonological discrepancy on modern speech foundation models, we evaluate Whisper baselines across various scales based on phonological boundaries. The contextual tone sandhi variants within the training split are deterministically derived from standard tone sandhi rules [20]. As detailed in Table II, scaling up the architecture improves overall performance, yet a localized gap persists. The models consistently exhibit lower character error rates (CERs) when decoding sandhi-form syllables compared to citationform syllables. Regardless of model size, the corresponding drops in precision and recall confirm a tonal mapping confusion, where the networks struggle to distinguish citation features from tone sandhi variations.

To analyze this localized performance gap, we examine the data distribution from a phonological perspective. Fig. 2(a) highlights a severe volume imbalance in continuous Taiwanese Hokkien speech, where approximately 87% of syllables undergo tone sandhi and only about 12% retain their citation tones. This extreme imbalance biases foundation models toward sandhi acoustic characteristics during pre-training. Consequently, implicit end-to-end learning fails to construct sufficiently robust representations for citation-retained syllables under low-resource fine-tuning. Furthermore, the tone transition matrix in Fig. 2(b) illustrates that distinct underlying citation categories frequently conflate into identical surface realizations. For instance, underlying Tone 1 (T1) and Tone<sup>❄️</sup> 5 (T5) often converge into surface Tone 7 (T7) after sandhi mutations. This many-to-one mapping ambiguity creates localized confusion between sandhi and citation forms, limiting overall performance and motivating the explicit phonetic disentanglement in our proposed architecture.

## III. PROPOSED METHOD

## A. Architecture Overview

As established, forcing an ASR model to implicitly memorize the complex, non-linear mapping between speech undergo tone sandhi and citation labels leads to severe surfaceto-underlying mapping discrepancy. To explicitly break this entanglement, we propose T-SANDHI. As illustrated in Fig. 3, the architecture builds upon the robust acoustic priors of a frozen Whisper encoder-decoder backbone. To maintain parameter efficiency while adapting to the intricate Taiwanese phonology, we avoid catastrophic forgetting by applying AdaLoRA [28] only to the attention and feed-forward mod-<sup>❄️❄️</sup> ules. Crucially, rather than relying on the decoder to implicitly⚙️⚙️ resolve tonal ambiguities, we introduce a Decoupled Hybrid

![](images/49309cd3f766bfc1b605bbea6035ff3d9406ae735984b114549188c83c3b5659.jpg)  
Fig. 3. Architecture of T-SANDHI. To explicitly decouple lexical intent from surface acoustics, a decoupled hybrid injection module augments the frozen Whisper encoder. It projects features into independent citation and sandhi streams, dynamically fusing them into ${ \mathbf { H } } _ { f u s e d }$ via a gate generator. Lightweight linear heads provide CTC supervision during training and supply decoupled representations during inference with negligible overhead.

Injection module directly on top of the encoder. This module explicitly constructs two independent auxiliary streams: one modeling the underlying lexical intent (citation syllables), and the other tracking the actual surface tone (sandhi tones).

## B. Decoupled Hybrid Injection

The core design philosophy behind our injection module is to enforce representation disentanglement without introducing heavy computational overhead. Let $\mathbf { H } _ { e n c } \in \mathbb { R } ^ { T \times d }$ denote the acoustic hidden states from the Whisper encoder, where $T$ is the sequence length and d is the hidden dimension. We project these states through two auxiliary CTC heads to obtain the citation logits $\mathbf { Z } _ { c i t } \in \mathbb { R } ^ { T \times | V _ { c i t } | }$ and the sandhi logits ${ \bf Z } _ { s a n }$ ∈ $\mathbb { R } ^ { T \times | V _ { s a n } | }$

$$
{ \bf Z } _ { c i t } = { \bf H } _ { e n c } { \bf W } _ { c i t } + { \bf b } _ { c i t } ,
$$

$$
{ \bf Z } _ { s a n } = { \bf H } _ { e n c } { \bf W } _ { s a n } + { \bf b } _ { s a n } ,\tag{1}
$$

(2)

where W and b denote the learnable weight matrices and bias vectors, while $| V _ { c i t } |$ and $| V _ { s a n } |$ represent the vocabulary sizes of the citation syllables and sandhi tones, respectively. Crucially, we deliberately restrict these heads to simple linear projections. This architectural bottleneck ensures that the parameter overhead remains minimal, actively forcing the core Whisper encoder to learn highly disentangled and robust acoustic representations rather than outsourcing the task to deep, heavy sub-networks.

To prepare these decoupled streams for integration, we first convert the raw logits into probability distributions via a softmax function, and then project them back to the hidden dimension d to form phonetic embeddings:

$$
\mathbf { E } _ { c i t } = \mathrm { P r o j } _ { c i t } ( \mathrm { S o f t m a x } ( \mathbf { Z } _ { c i t } ) ) ,\tag{3}
$$

$$
\mathbf { E } _ { s a n } = \operatorname { P r o j } _ { s a n } ( \operatorname { S o f t m a x } ( \mathbf { Z } _ { s a n } ) ) .\tag{4}
$$

Applying softmax grounds the projections in discrete phonetic probabilities, preventing a linear collapse of the acoustic states.

Subsequently, we employ a dynamic gating mechanism to effectively fuse these features back into the main network. Rather than simply adding the features, we concatenate the original acoustic states with the projected phonetic embeddings to form a joint representation ${ \bf C } = [ { \bf H } _ { e n c } ; { \bf E } _ { c i t } ; { \bf E } _ { s a n } ] \in$ $\mathbb { R } ^ { T \times 3 d }$ . A gate generator then processes this joint representation to output a frame-level weight matrix $\mathbf { G } \in \mathbb { R } ^ { T \times 2 }$

$$
\mathbf { G } = \sigma ( \mathbf { C W } _ { g a t e } + \mathbf { b } _ { g a t e } ) ,\tag{5}
$$

where $\mathbf { W } _ { g a t e } \in \mathbb { R } ^ { 3 d \times 2 }$ and $\mathbf { G } = [ \mathbf { g } _ { c i t } , \mathbf { g } _ { s a n } ]$ . The physical significance of this dynamic gate directly mirrors the contextdependent nature of Taiwanese tone sandhi. Since tonal shifts occur strictly based on grammatical positions, G acts as an adaptive soft switch. By observing both the acoustic context and the explicit phonetic hypotheses, the gate dynamically allocates attention between the underlying intent $( \mathbf { g } _ { c i t } )$ and the surface realization $( { \bf g } _ { s a n } )$ frame by frame.

To prevent the newly initialized gate from catastrophically interfering with the frozen Whisper backbone during the crucial early stages of fine-tuning, we initialize the bias $\mathbf { b } _ { g a t e }$ to a strong negative scalar (−3). This insight-driven initialization forces the initial gate values toward zero, compelling the model to rely on the robust pre-trained features first and gradually learn to blend in the decoupled phonetic patches. The final fused representation ${ \mathbf { H } } _ { f u s e d } .$ which is subsequently fed to the decoder, is computed as:

$$
{ \bf H } _ { f u s e d } = { \bf H } _ { e n c } + { \bf g } _ { c i t } \odot { \bf E } _ { c i t } + { \bf g } _ { s a n } \odot { \bf E } _ { s a n } ,\tag{6}
$$

where ⊙ represents element-wise multiplication, with $\mathbf { g } _ { c i t }$ and ${ \bf g } _ { s a n }$ implicitly broadcasted across the hidden dimension d.

## C. Multi-Task Learning Objective

To actualize this decoupled architecture, the training objective must explicitly penalize entangled representations. Our primary objective remains the standard sequence-to-sequence cross-entropy loss $\mathcal { L } _ { A S R }$ generated by the Whisper decoder for the final Taiwanese Hanzi transcriptions. However, to provide the necessary phonetic grounding for our injection module, we must introduce auxiliary losses. Since framelevel forced alignment data is prohibitively expensive and largely unavailable for low-resource languages, we formulate these as CTC objectives. CTC naturally marginalizes over all possible unsegmented acoustic alignments. Thus, $\mathcal { L } _ { C T C } ^ { ( c i t ) }$ explicitly guides the extraction of citation syllables, while $\mathcal { L } _ { C T C } ^ { ( \dot { s } a n ) }$ supervises the sandhi tone tracking:

$$
\mathcal { L } _ { T o t a l } = \mathcal { L } _ { A S R } + \lambda _ { c i t } \mathcal { L } _ { C T C } ^ { ( c i t ) } + \lambda _ { s a n } \mathcal { L } _ { C T C } ^ { ( s a n ) } ,\tag{7}
$$

where $\lambda _ { c i t }$ and $\lambda _ { s a n }$ are scalar hyperparameters. This joint optimization ensures that the encoder effectively disentangles “what is heard” from “what is meant” before passing the representation to the decoder.

## IV. EXPERIMENTAL SETUP

## A. Dual-Track Supervision Strategy

Our data preparation goes beyond formatting to construct explicit supervision signals for the decoupled architecture. Using the official MOE Taiwanese Dictionary<sup>2</sup> and Taibun toolkit<sup>3</sup>, we extracted two strictly parallel phonetic tracks from the Hanzi transcriptions: the citation dictionary forms and the contextual sandhi tones. This dual-track extraction yields the text-derived pseudo labels to independently guide our auxiliary CTC streams under weak supervision. Specifically, the citation vocabulary $| V _ { c i t } |$ comprises 1309 unique syllables with tone marks, while the sandhi vocabulary $| V _ { s a n } |$ tracks 10 classes (8 tones, a neutral tone, and a blank token).

## B. Implementation Details

We employed Whisper-small [1] as our acoustic foundation. To preserve its pre-trained speech representations while adapting to Taiwanese Hokkien, we did not perform full fine-tuning. Instead, we applied AdaLoRA [28] to the attention and feedforward blocks with an initial rank of 12 (pruned down to 4) and a dropout rate of 0.1 to ensure parameter efficiency. To guarantee stable training for our decoupled hybrid injection module, the bias vector of the gate mechanism, $\mathbf { b } _ { \mathrm { g a t e } }$ , was initialized to −3. This configuration ensures that the model heavily relies on the robust frozen backbone during the initial training phase, gradually incorporating the auxiliary phonetic information as training converges. For the multi-task objective function (Eq. 7), the CTC loss weights $\lambda _ { \mathrm { c i t } }$ and $\lambda _ { \mathrm { s a n } }$ were empirically set to 0.9 and 0.1, respectively. This weighting ensures adequate phonetic supervision without overshadowing the primary sequence-to-sequence objective.

## C. Evaluation and Diagnostic Metrics

Our primary performance metric is the CER on Taiwanese Hanzi, which directly reflects practical utility. However, to evaluate whether our architecture effectively addresses the localized mapping confusion between surface acoustics and underlying lexical intent, we must look beyond final transcription errors. Therefore, we introduce two diagnostic metrics: syllable error rate (SER) to evaluate underlying lexical intent tracking, and tone error rate (TER) to measure surface tone resolution. Tracking these metrics allows us to empirically verify that the performance gains stem directly from our explicit phonetic disentanglement.

TABLE III  
CER (%) COMPARISON ON THE TAT-MOE, FSRC 2020, AND YTTD TAIGI TRS BLIND TEST SETS.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Parameters (M)</td><td colspan="2">TAT-MOE</td><td>FSRC 2020</td><td>yttd_taigi_trs</td></tr><tr><td>Development</td><td>Test</td><td>Blind Test</td><td>Blind Test</td></tr><tr><td>Zipformer [6]</td><td>65</td><td>48.57</td><td>45.82</td><td>15.69</td><td>58.15</td></tr><tr><td>HuBERT-base [2]</td><td>96</td><td>26.16</td><td>24.49</td><td>12.97</td><td>61.68</td></tr><tr><td>CLiFT-ASR [39]</td><td>96</td><td>22.37</td><td>20.94</td><td>8.06</td><td>57.69</td></tr><tr><td>Whisper-small (Encoder-CTC)</td><td>104</td><td>19.90</td><td>17.98</td><td>9.43</td><td>42.73</td></tr><tr><td>Whisper-small (Fully Fine-tuned)</td><td>244</td><td>22.47</td><td>18.68</td><td>7.66</td><td>44.43</td></tr><tr><td>Whisper-small (AdaLoRA)</td><td>244</td><td>18.54</td><td>16.30</td><td>8.75</td><td>39.11</td></tr><tr><td>T-SANDHI (Ours)</td><td>249</td><td>16.49</td><td>14.30</td><td>7.36</td><td>36.57</td></tr><tr><td>Qwen3-ASR</td><td>600</td><td>16.74</td><td>14.03</td><td>7.38</td><td>44.91</td></tr><tr><td>Whisper-large (AdaLoRA)</td><td>1,550</td><td>16.03</td><td>13.89</td><td>7.20</td><td>29.31</td></tr></table>

TABLE IV

CER (%) PERFORMANCE AND RELATIVE REDUCTION (REL., %) ACROSS WHISPER SCALES ON THE TAT-MOE CORPUS.
<table><tr><td>Model Scale</td><td>Baseline</td><td>T-SANDHI</td><td>Rel.</td></tr><tr><td>Small</td><td>16.30</td><td>14.30</td><td>12.27</td></tr><tr><td>Medium</td><td>14.58</td><td>13.65</td><td>6.38</td></tr><tr><td>Large</td><td>13.89</td><td>12.93</td><td>6.91</td></tr></table>

## V. RESULTS AND DISCUSSION

## A. Overall ASR Performance

Table III demonstrates that explicitly decoupling surface contextual features from underlying canonical features effectively mitigates the tonal mapping confusion caused by tone sandhi. To ensure a rigorous evaluation, our baselines span three architectural paradigms: RNN-Transducer (RNN-T) models, pure CTC frameworks, and attention-based encoderdecoder (AED) foundation models. Traditional end-to-end RNN-T models struggle to implicitly memorize dynamic tonal mappings, as demonstrated by Zipformer [6], HuBERT-base [2], and CLiFT-ASR [39] yielding Test CERs of 45.82%, 24.49%, and 20.94%, respectively. Augmenting the foundation encoder with a standard CTC head (Whisper-small (Encoder-CTC)) improves performance but yields a suboptimal 17.98% CER, constrained by a monolithic sequence loss. Naive adaptations of the Whisper backbone similarly fall short: full fine-tuning suffers from representation distortion (18.68%), and standard AdaLoRA (16.30%) remains bottlenecked by entangled phonetic features.

Crucially, T-SANDHI overcomes these limitations. With only a 5M parameter overhead for the decoupled hybrid injection module, our framework achieves a 14.30% Test CER, a 12.27% relative error reduction over AdaLoRA. Coupled with robust out-of-domain generalization on the FSRC 2020 (7.36%) and yttd taigi trs (36.57%) blind test sets, these results confirm that explicit phonetic disentanglement, rather than mere parameter scaling, drives the performance gains.

## B. Scalability Across Backbone Capacities

We further investigate T-SANDHI’s scalability across larger foundation models. As detailed in Table IV, the method yields consistent gains across all evaluated Whisper backbones. While increasing model capacity naturally lowers the baseline error rate, our dual-stream module extracts additional improvements, achieving relative error reductions of 12.27%, 6.38%, and 6.91% on the small, medium, and large backbones, respectively. This confirms that phonetic disentanglement provides orthogonal benefits to naive parameter scaling.

TABLE V  
ABLATION STUDIES OF T-SANDHI ON THE TAT-MOE CORPUS.
<table><tr><td>Model Configuration</td><td>SER %</td><td>TER %</td><td>CER %</td></tr><tr><td>T-SANDHI (Full Model)</td><td>16.27</td><td>13.99</td><td>14.30</td></tr><tr><td>w/o Decoupled Streams</td><td>-</td><td></td><td>16.30</td></tr><tr><td>w/o Citation CTC Head</td><td></td><td>13.52</td><td>15.44</td></tr><tr><td>w/o Sandhi CTC Head</td><td>16.68</td><td>一</td><td>15.19</td></tr><tr><td>w/o Gate Generator</td><td>17.45</td><td>14.30</td><td>15.55</td></tr><tr><td>w/o Dynamic Gating</td><td>17.36</td><td>14.19</td><td>15.67</td></tr></table>

## C. Ablation Studies

Table V validates that our performance gains stem from explicit phonetic disentanglement. Removing the dual-stream architecture entirely regresses the CER to 16.30%, indicating that the frozen backbone alone cannot fully resolve acousticto-lexical confusion. Ablating individual CTC heads highlights the mechanics of tone sandhi resolution. Omitting the citation head improves the surface TER (13.52%) but degrades the final CER (15.44%), as the model overfits surface acoustics and loses underlying Hanzi identity. Conversely, removing the sandhi head worsens the SER (16.68%), demonstrating that surface tracking provides essential acoustic grounding.

Moreover, optimal integration requires dynamic fusion. An unweighted summation degrades CER to 15.55% by reentangling features, and using static global weights (heuristically tuned on the development set) yields a suboptimal 15.67%. Static weights fail to capture position-dependent sandhi rules (e.g., word-medial mutation vs. sentence-final retention), underscoring the necessity of a frame-by-frame neural soft-switch.

## D. Impact of Information Granularity in Dual Streams

To determine the optimal feature resolution, we ablated the linguistic granularity injected into the dual streams. Table VI contrasts configuring either stream with phone-level tone labels versus full syllable-level targets. Pairing citation syllables with surface tones yields the lowest CER (14.30%), outperforming all symmetric configurations.

TABLE VI  
ABLATION STUDY ON THE GRANULARITY OF INJECTED LINGUISTIC INFORMATION FOR THE DUAL STREAMS.
<table><tr><td>Citation Info</td><td>Sandhi Info</td><td>Test CER (%)</td></tr><tr><td>Tone</td><td>Tone</td><td>15.60</td></tr><tr><td>Tone</td><td>Syllable</td><td>15.84</td></tr><tr><td>Syllable</td><td>Tone</td><td>14.30</td></tr><tr><td>Syllable</td><td>Syllable</td><td>14.80</td></tr></table>

![](images/d61d8696949de30f3cde0aae0d501ec226239704b122e55fe7813fd103c6d399.jpg)  
<sub>Fig.</sub> <sub>4.</sub> <sub>Correlation</sub> <sub>between</sub> <sub>phonetic</sub> <sub>errors</sub> <sub>and</sub> <sub>transcription</sub> <sub>accuracy.</sub>日 子

Phonetically, the citation stream relies on full syllables to anchor lexical identity. Conversely, the surface stream, tracking actual acoustic realizations, functions best when constrained to tone-level variations. This structural bottleneck prevents the sandhi head from overfitting to specific lexical items, effectively mitigating localized mapping confusion in lowresource settings.

## E. Correlation between Phonetic and Character Errors

Fig. 4 plots final CER against auxiliary phonetic error rates, verifying that the decoder actively utilizes decoupled<sup>i̍</sup> representations. The strong positive correlation confirms that accurate phonetic grounding is a prerequisite for transcription. Crucially, sandhi TER and citation SER trajectories diverge as errors increase. The SER’s steeper slope indicates the citation syllable serves as the primary structural anchor; misidentifying it severely degrades character prediction. The TER’s gentler slope suggests the decoder leverages contextual language modeling to tolerate minor tone errors. Nevertheless, minimizing TER remains essential to resolve localized mapping confusion and push CER below 20%. This hierarchy, where syllables provide structure and tones provide disambiguation, empirically justifies the dual-stream strategy.

## F. Emergent Phonological Awareness in Dynamic Gating

Fig. 5 visualizes dynamic gate behavior across a continuous utterance to illustrate mapping resolution. During phraseinternal mutations (e.g., Tsok4 realized as Gik8), the network autonomously suppresses the surface stream日 子 $( \mathbf { g } _ { \mathrm { s a n } } \ \approx \ 0 )$ to filter deceptive acoustics, anchoring instead on the citation intent $( \mathbf { g } _ { \mathrm { c i t } } )$ . At the phrase-final position (Tsu1), where surface acoustics and citation tones align, $\mathbf { g } _ { \mathrm { s a n } }$ activates sharply $( \mathbf { g } _ { \mathrm { s a n } } \to 1 )$ . This demonstrates the network learns to rely on surface acoustics only when phonologically reliable. These findings indicate an emergent, data-driven awareness of rightprominent sandhi rules within the gating mechanism.

![](images/4707eb4a8c990c974332ba996f4f85d399e25f4a5180c553a4aabf8b2e15179f.jpg)  
Fig. 5. Frame-level visualization of the dynamic gate.

![](images/ab671498669164e6f0c7f92adda1e8cfa2eebd0efc64d9ac032ca964ba74ab1c.jpg)  
Fig. 6. Qualitative comparison of error resolution.

## G. Targeted Error Analysis

Fig. 6 illustrates T-SANDHI’s resolution of character map-<sub>ping confusion. In the phrase “</sub>顛倒<sub>” (</sub>tian-to<sub>\`, unexpectedly),</sub> underlying citation tones (T1-T3) mutate into surface tones (T7-T2). Misled by these surface acoustics, the Whispersmall (AdaLoRA) baseline incorrectly predicts the phonetically identical but semantically unrelated $ \overrightarrow { E E } \overrightarrow { \equiv } \overrightarrow { \sf { A } } > \overrightarrow { \sf { A } }$ (tian th¯ o´, citation T7-T2). This substitution highlights how conventional models, lacking explicit guidance, overfit surface tone variations and fail to recover lexical intent. In contrast, T-SANDHI successfully outputs the correct Hanzi. By tracking surface realizations (T7-T2) while anchoring on citation identity (T1- T3), the decoupled framework bridges the phonological gap. This frame-by-frame dynamic gating effectively resolves the confusion of entangled architectures.

## VI. CONCLUSION AND FUTURE WORK

In this paper, we proposed T-SANDHI<sup>4</sup>, a parameterefficient framework that explicitly disentangles surface acoustics from underlying lexical intent to mitigate tone sandhi mapping confusion. By employing a dynamic gating mechanism, our approach achieves robust performance gains across foundation models without substantial parameter scaling. While currently supervised by rule-derived pseudo-labels, future work will explore unsupervised disentanglement for undocumented dialects, as well as extending this decoupled paradigm to Mandarin-Taiwanese code-switching.

<sup>4</sup>Our source code: https://anonymous.4open.science/r/T-SANDHI-0D38

## REFERENCES

[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. PMLR, 2023.

[2] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021.

[3] A. Graves, “Sequence transduction with recurrent neural networks,” in Proc. ICML, 2012.

[4] X. Wang, Z. Yao, X. Shi, and L. Xie, “Cascade RNN-transducer: Syllable based streaming on-device Mandarin speech recognition with a syllable-to-character converter,” in Proc. IEEE SLT, 2021.

[5] M. Ghodsi, X. Liu, J. Apfel, R. Cabrera, and E. Weinstein, “RNNtransducer with stateless prediction network,” in Proc. ICASSP, 2020.

[6] Z. Yao, L. Guo, X. Yang, W. Kang, F. Kuang, Y. Yang, Z. Jin, L. Lin, and D. Povey, “Zipformer: A faster and better encoder for automatic speech recognition,” in Proc. ICLR, 2024.

[7] S. Watanabe, T. Hori, S. Kim, J. R. Hershey, and T. Hayashi, “Hybrid CTC/Attention architecture for end-to-end speech recognition,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 8, pp. 1240–1253, 2017.

[8] H.-l. Khoo, “The dynamics of Southern Min in Taiwan: From Southern Min dialects to “Taigi”,” in The Routledge Handbook of Chinese Discourse Analysis, C. Shei, Ed. Routledge, 2019, pp. 596–610.

[9] C.-W. Chen, Y.-F. Yeh, C.-L. Lin, S.-P. Tseng, and J.-F. Wang, “Hybrid deep neural network acoustic model for Taiwanese speech recognition,” in Proc. IEEE ICOT, 2020.

[10] P.-Y. Chen, C.-H. Wu, H.-S. Lee, S.-K. Tsao, M.-T. Ko, and H.-M. Wang, “Using Taigi dramas with Mandarin Chinese subtitles to improve Taigi speech recognition,” in Proc. O-COCOSDA, 2020.

[11] M.-L. Hsieh, “Taiwanese Hokkien/Southern Min,” in The Handbook of Chinese Linguistics, 2014, pp. 629–656.

[12] Y.-F. Liao, W.-H. Hsu, C.-M. Pan, W.-J. Wang, M. Pleva, and D. Hladek, “Personalized Taiwanese speech synthesis using cascaded ASR and TTS framework,” in Proc. RADIOELEKTRONIKA, 2022.

[13] Y.-H. Chou, K. Chang, M.-J. Wu, W. Ou, A. W.-H. Bi, C. Yang, B. Y. Chen, R.-W. Pai, P.-Y. Yeh, J.-P. Chiang, I.-T. Phoann, W. Chang, C. Cui, N. Chen, and J. Shi, “Evaluating self-supervised speech models on a Taiwanese Hokkien corpus,” in Proc. IEEE ASRU, 2023.

[14] J. Lin, S. Lu, H. Huang, W. Guan, B. Xu, H. Bu, Q. Hong, and L. Li, “MinSpeech: A corpus of Southern Min dialect for automatic speech recognition,” in Proc. Interspeech, 2024.

[15] B. Zhang, H. Lv, P. Guo, Q. Shao, C. Yang, L. Xie, X. Xu, H. Bu, X. Chen, C. Zeng, D. Wu, and Z. Peng, “WenetSpeech: A 10000+ hours multi-domain Mandarin corpus for speech recognition,” in Proc. ICASSP, 2022.

[16] H. Bu, J. Du, X. Na, B. Wu, and H. Zheng, “AISHELL-1: An opensource Mandarin speech corpus and a speech recognition baseline,” in Proc. O-COCOSDA, 2017.

[17] Y. Fu, L. Cheng, S. Lv, Y. Jv, Y. Kong, Z. Chen, Y. Hu, L. Xie, J. Wu, H. Bu, X. Xu, J. Du, and J. Chen, “AISHELL-4: An open source dataset for speech enhancement, separation, recognition and speaker diarization in conference scenario,” in Proc. Interspeech, 2021.

[18] R. L. Cheng, “Tone sandhi in Taiwanese,” Linguistics, vol. 6, no. 41, pp. 19–42, 1968.

[19] Y.-F. Chien and A. Jongman, “Tonal neutralization of Taiwanese checked and smooth syllables: An acoustic study,” Language and Speech, vol. 62, no. 3, pp. 452–474, 2019.

[20] Y.-Y. Chuang and S.-F. Wang, “Tonal variation and word meaning in Taiwanese,” in Proc. Interspeech, 2025.

[21] P.-J. Chen, K. Tran, Y. Yang, J. Du, J. Kao, Y.-A. Chung, P. Tomasello, P.-A. Duquenne, H. Schwenk, H. Gong, H. Inaguma, S. Popuri, C. Wang, J. Pino, W.-N. Hsu, and A. Lee, “Speech-to-speech translation for a realworld unwritten language,” in Findings of ACL, 2023.

[22] J. Myers and J. Tsay, “Neutralization in Taiwan Southern Min Tone Sandhi,” Interfaces in Chinese Phonology, 2008.

[23] G. Shen, M. Watkins, A. Alishahi, A. Bisazza, and G. Chrupała, “Encoding of lexical tone in self-supervised models of spoken language,” in Proc. NAACL, 2024.

[24] S. Bandarupalli, B. Akkiraju, C. Devarakonda, V. Narsinga, and A. K. Vuppala, “Efficient ASR for low-resource languages: Leveraging crosslingual unlabeled data,” in Findings of IJCNLP, 2025.

[25] O. Klejch, W. Lamb, and P. Bell, “A practitioner’s guide to building ASR models for low-resource languages: a case study on Scottish Gaelic,” in Proc. Interspeech, 2025.

[26] M. Bartelds, N. San, B. McDonnell, D. Jurafsky, and M. Wieling, “Making more of little data: Improving low-resource automatic speech recognition using data augmentation,” in Proc. ACL, 2023.

[27] C. Jacobs, A. Smith, D. Klop, O. Klejch, F. de Wet, and H. Kamper, “Speech recognition for automatically assessing Afrikaans and IsiXhosa preschool oral narratives,” in Proc. ICASSP, 2025.

[28] Q. Zhang, M. Chen, A. Bukharin, N. Karampatziakis, P. He, Y. Cheng, W. Chen, and T. Zhao, “AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning,” in Proc. ICLR, 2023.

[29] Z. Song, J. Zhuo, Y. Yang, Z. Ma, S. Zhang, and X. Chen, “LoRA-Whisper: Parameter-Efficient and Extensible Multilingual ASR,” in Proc. Interspeech, 2024.

[30] T. Tan, X. Chen, X. Le, W. Fan, X. Xia, C. Huang, and J. Lu, “CBA-Whisper: curriculum learning-based AdaLoRA fine-tuning on whisper for low-resource dysarthric speech recognition,” in Proc. Interspeech, 2025.

[31] W. Liu, Y. Qin, Z. Peng, and T. Lee, “Sparsely Shared Lora on Whisper for Child Speech Recognition,” in Proc. ICASSP, 2024.

[32] A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber, “Connection-´ ist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,” in Proc. ICML, 2006.

[33] S. Kim, T. Hori, and S. Watanabe, “Joint CTC-attention based endto-end speech recognition using multi-task learning,” in Proc. ICASSP, 2017.

[34] K. Hojo, Y. Wakabayashi, K. Ohta, A. Ogawa, and N. Kitaoka, “Boosting CTC-based ASR using inter-layer attention-based CTC loss,” in Proc. Interspeech, 2024.

[35] S. Han, M. Xu, Z. Lei, Z. Huang, and X. Na, “Enhancing CTC-based speech recognition with diverse modeling units,” in Proc. Interspeech, 2024.

[36] N. Kusunoki, Y. Higuchi, T. Ogawa, and T. Kobayashi, “Hierarchical multi-task learning with CTC and recursive operation,” in Proc. Interspeech, 2024.

[37] Y.-F. Liao, J. S. Tsay, P. Kang, H.-L. Khoo, L.-K. Tan, L.-C. Chang, U.- G. Iunn, H.-L. Su, T.-G. Thiann, H.-K. Tiun, and S.-L. Liao, “Taiwanese across Taiwan corpus and its applications,” in Proc. O-COCOSDA, 2022.

[38] Y.-F. Liao, C.-Y. Chang, H.-K. Tiun, H.-L. Su, H.-L. Khoo, J. S. Tsay, L.-K. Tan, P. Kang, T.-g. Thiann, U.-G. Iunn, J.-H. Yang, and C.- N. Liang, “Formosa speech recognition challenge 2020 and Taiwanese across Taiwan corpus,” in Proc. O-COCOSDA, 2020.

[39] H.-Y. Sung, C.-C. Wang, K.-T. Huang, T.-H. Lo, Y.-S. Tsao, Y.-C. Hsu, and B. Chen, “CLiFT-ASR: A cross-lingual fine-tuning framework for low-resource Taiwanese Hokkien speech recognition,” in Proc. ROCLING, 2025.