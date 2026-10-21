# AudiPsy: Evidence-Grounded Multimodal Reasoning over Spoken Counseling Conversations for Depression Assessment

Chuanbo Zhu<sup>1</sup>, Run Luo<sup>3</sup>, Weibo Zhao<sup>4</sup>, Yue Li<sup>2</sup>, Kun Qian<sup>5</sup>, Qiang Qu<sup>2</sup>, Yike Guo<sup>1</sup>, Minghuan Tan<sup>2</sup>\*, Wei Xue<sup>1</sup>\*

<sup>1</sup> The Hong Kong University of Science and Technology,

<sup>2</sup> Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, <sup>3</sup> National University of Singapore, <sup>4</sup> University of the Chinese Academy of Sciences, <sup>5</sup> Beijing Institute of Technology

Correspondence: czhuat@connect.ust.hk, mh.tan@siat.ac.cn, weixue@ust.hk

## Abstract

Depression assessment from counseling speech requires integrating textual and acoustic psychological evidence across turns, yet existing end-to-end classification approaches lack explicit multimodal evidence interpretation and context-aware reasoning. We propose AudiPsy, an evidence-grounded omni-LLM for depression assessment over spoken counseling conversations. AudiPsy employs an Observation– Assessment–Summary (OAS) paradigm to integrate dialogue transcripts, textual and acoustic emotions, and textual–acoustic emotion incongruence into structured reasoning. OAS incrementally organizes multimodal observations, performs contextualized assessment, and maintains an evolving dialogue summary for subsequent turns. To support this framework, we construct AudiPsy-Counsel, a synthetic spoken counseling conversation corpus with multimodal context augmentation, and develop a unified three-stage curriculum for depressionoriented reasoning. Experiments on DAIC-WOZ demonstrate superior depression assessment performance over specialized depression detection methods and general-purpose speech/omni-LLMs, along with improved reasoning quality, while supplementary evaluation on EATD-Corpus supports AudiPsy’s applicability to naturalistic conversations.

## 1 Introduction

Depression affects more than 322 million people worldwide and remains one of the leading causes of disability (World Health Organization, 2026). Depression assessment is typically conducted through counseling conversations, where human counselors observe seekers’ behaviors, interpret cues, accumulate evidence and reach a judgment (Mao et al., 2023). Recent advances in AI for medical diagnosis have shifted from prediction toward evidencebased reasoning (Wang et al., 2025; Zhu et al.,

2026). In the domain of mental health, automated depression assessment similarly calls for evidencegrounded multimodal reasoning across evolving spoken conversations.

Existing depression detection systems predominantly formulate the assessment as a multimodal classification problem, directly mapping linguistic and acoustic representations to diagnostic labels (Wu et al., 2023; Zhang et al., 2024b, 2025). Although effective on benchmark datasets, this paradigm suffers from two limitations. First, at the representation level, most multimodal methods rely on low-level acoustic features or latent representations whose psychological meaning remains largely implicit. Second, at the process level, existing approaches typically aggregate multimodal representations into a holistic prediction without interpreting them under the conversational context. These two limitations are coupled: the model cannot state what it observes if it relies on implicit representations; the model cannot explain how the observed evidence supports its assessment without an interpretive process. We therefore address them jointly.

For what the model reasons over, we ground the assessment in affective evidence: textual emotion, acoustic emotion, and the incongruence between the two, expressed as affective concepts in natural language so that an LLM can reason over them. Psychological studies have shown that individuals with depression often exhibit atypical emotional patterns (Clark and Watson, 1991), and inconsistency between semantic and acoustic emotion, as described by the emotion context insensitivity (ECI) theory (Bylsma et al., 2008; Wu et al., 2024). Recent studies have observed that modeling this emotional incongruence improves depression detection performance (Su et al., 2026). This evidence is psychologically interpretable and naturally amenable to language-model reasoning.

For how the evidence is interpreted, following the structure of clinical progress notes (Weed, 1968), we introduce the Observation–Assessment–Summary (OAS) paradigm for contextual reasoning over multi-turn spoken counseling conversations. At each conversational turn, OAS extracts structured semantic and acoustic observations (O), interprets them in light of accumulated dialogue context and textual–acoustic emotion incongruence (A), and updates a compact dialogue summary (S) to guide subsequent assessment. This incremental process preserves the contextual meaning of psychological evidence while reducing the burden of long-context reasoning for current omni-LLMs (Chen et al., 2026; Wang et al., 2026).

![](images/7a6d3ec53f18d9190b66e5310bda7c1234c32d68ad3dd57f53818f145d8bcb76.jpg)  
Figure 1: Overview of the AudiPsy, from structured multimodal reasoning to data augmentation and model training.

To this end, we propose AudiPsy, an omni-LLM for depression assessment, which explicitly reasons over spoken counseling conversations in terms of multimodal affective evidence based on the proposed Observation–Assessment– Summary (OAS) reasoning paradigm. To overcome the scarcity of real-world counseling speech corpus, we further develop a counseling context augmentation pipeline and a unified three-stage curriculum for depression-oriented reasoning. Experiments on DAIC-WOZ (Gratch et al., 2014), featuring structured spoken counseling conversations, demonstrate superior depression assessment and reasoning performance. Additional evaluation on EATD-Corpus (Shen et al., 2022a), comprising naturalistic conversations, further examines AudiPsy’s applicability across conversational settings. Our contributions are summarized as follows:

• We propose AudiPsy, an evidence-grounded omni-LLM for depression assessment through structured OAS reasoning.

• We construct AudiPsy-Counsel, a bilingual synthetic counseling speech corpus with OAS annotations and controlled emotionincongruence cases.

• We develop a unified three-stage curriculum for Depression-oriented Reasoning, integrating affective perception, structured reasoning, and preference alignment.

• Experiments demonstrate improved depression assessment and reasoning quality, while analyses reveal how AudiPsy perceive, utilize, and interpret affective evidence.

## 2 Related Work

Automated depression assessment over spoken counseling conversations leverages linguistic, acoustic, and multimodal signals to identify mental health markers. Existing methods can be broadly categorized into three areas: (1) Text-based approaches utilize transcripts to capture depressive states. Early methods rely on affective lexicons or neural representations (Mallol-Ragolta et al., 2019; Xezonaki et al., 2020), while recent frameworks employ large language models (LLMs) through prompting and instruction tuning (Xu et al., 2024; Liu et al., 2025). (2) Speechbased approaches identify depression-related cues from acoustic characteristics such as prosody and spectral features (Sardari et al., 2022; Wu et al., 2023), with recent methods incorporating data augmentation and retrieval-based reasoning to improve robustness (Liang et al., 2025; Zhang et al., 2025). (3) Multimodal approaches integrate linguistic and acoustic signals through feature fusion (Shen et al., 2022a; Chen et al., 2024) or LLM-based modeling of textual and speech-derived representations (Zhang et al., 2024b, 2025). Despite this progress, existing methods predominantly formulate depression assessment as end-to-end classification, leaving the psychological meaning of multimodal representations largely implicit and lacking contextualized interpretation of evidence as the conversation unfolds.

## 3 Methodology

As illustrated in Figure 1, AudiPsy combines the OAS reasoning paradigm with counseling context augmentation and a three-stage training curriculum for depression-oriented reasoning. Together, these components enable interpretable, context-aware multimodal reasoning for depression assessment.

## 3.1 OAS Structured Reasoning Paradigm

We propose an Observation–Assessment– Summary (OAS) reasoning paradigm that decomposes multimodal depression reasoning into three structured fields, as illustrated in Figure 1a. Given a counseling speech with T dialogue turns, we denote it as $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { T } \}$ , where each turn is represented by $\boldsymbol { x } _ { t } = \left( c _ { t } , s _ { t } \right)$ , with $c _ { t }$ and $s _ { t }$ denoting the counselor’s and seeker’s spoken utterances, respectively. The ordering is used only for notation and does not imply which speaker initiates the dialogue. At each turn, AudiPsy maps the current dialogue turn and the previous summary state to an observation, an assessment, and an updated summary:

$$
( O _ { t } , A _ { t } , S _ { t } ) = \mathcal { F } _ { \theta } ( x _ { t } , S _ { t - 1 } ) ,\tag{1}
$$

where $\mathcal { F } _ { \theta }$ denotes AudiPsy parameterized by θ, and $S _ { t - 1 }$ summarizes the accumulated dialogue context from previous turns. The three fields are defined as follows.

The Observation $O _ { t } ~ = ~ ( O _ { t } ^ { \mathrm { d t } } , O _ { t } ^ { \mathrm { t e } } , O _ { t } ^ { \mathrm { a e } } )$ consists of the dialogue transcript $O _ { t } ^ { \mathrm { d t } }$ , the seeker’s textual emotion $O _ { t } ^ { \mathrm { t e } }$ and acoustic emotion $O _ { t } ^ { \mathrm { a e } }$ . Together, they provide explicit and interpretable multimodal evidence for subsequent reasoning.

The Assessment $A _ { t } = ( A _ { t } ^ { \mathrm { e i } } , A _ { t } ^ { \mathrm { d e p } } )$ contains the textual–acoustic emotion incongruence (EI) assessment $A _ { t } ^ { \mathrm { { e i } } }$ and the depression assessment $A _ { t } ^ { \mathrm { { d e p } } }$ , each consisting of a binary prediction answer together with its supporting rationale. Therefore, the depression assessment can be formulated as a conditional dependency $p _ { \theta } ( A _ { t } ^ { \mathrm { d e p } } \ | \ A _ { t } ^ { \mathrm { e i } } , O _ { t } , S _ { t - 1 } )$ , indicating that the reasoning $A _ { t } ^ { \mathrm { { d e p } } }$ is grounded by evidence $A _ { t } ^ { \mathrm { e i } } , O _ { t }$ and $S _ { t - 1 }$

The Summary $S _ { t }$ recursively maintains a compact dialogue state, including a concise summary of the counseling content together with accumulated emotional statistics and affective patterns.

Finally, the dialogue-level depression prediction is obtained by aggregating the turn-level depression assessments:

$$
\begin{array} { r } { \hat { y } = \mathrm { A g g } \left( A _ { 1 } ^ { \mathrm { d e p } } , A _ { 2 } ^ { \mathrm { d e p } } , \ldots , A _ { T } ^ { \mathrm { d e p } } \right) , } \end{array}\tag{2}
$$

where $\operatorname { A g g } ( { \mathord { \cdot } } )$ denotes a consistency-based aggregation function. In our implementation, $\mathrm { A g g ( \cdot ) }$ uses majority voting over turn-level binary predictions for robust conversation-level aggregation. Detailed output templates and field definitions are provided in Appendix A.

## 3.2 Counseling Context Augmentation

Real-world speech-based counseling data is extremely scarce due to privacy concerns and acquisition costs, while text-only counseling resources are considerably abundant, but lack the affective context and its acoustic realization required for multimodal depression reasoning. To bridge this gap, we formulate data construction as a context augmentation problem. Starting from publicly available text-only counseling records, we progressively enrich them along three orthogonal dimensions: situational, affective, and acoustic, so that each resulting sample supplies the contextual evidence consumed by the OAS paradigm, as shown in Figure 1b. Although synthetic, the resulting dataset, AudiPsy-Counsel, provides structured supervision for learning how to perceive and interpret multimodal evidence to support depression assessment across dialogue turns.

Situational augmentation. The seed corpus is an union of two public text counseling corpora, CounselChat (Bertagnolli, 2020) and CPsy-CounR (Zhang et al., 2024a). We employ GPT-5 (OpenAI, 2025) as a situational re-writer which maps each counseling record into a bilingual dialogue transcript $O ^ { \mathrm { d t } }$ between a counselor and a seeker. The rewriting preserves the original clinical issue while diversifying emotional conditions and interaction trajectories, resulting in richer situational variations while preserving the natural distribution of counseling topics, including depression, relationships, self-development, and more (Figure 2).

Affective annotation. For each utterance, GPT-5 also acts as an affective annotator to produce textual and acoustic emotion annotations $O _ { t } ^ { \mathrm { t e } }$ and $O _ { t } ^ { \mathrm { a e } }$ , together with the textual–acoustic emotion incongruence annotation $A _ { t } ^ { \mathrm { { e i } } }$ , while the seeker gender $g$ is annotated at the dialogue level. Together, these annotations provide explicit ground-truth multimodal evidence including textual–acoustic emotion incongruence, which is rarely available and difficult to verify in organic data.

Conditioned on the annotated $O _ { t } , A _ { t } ^ { e i }$ , and $S _ { t - 1 }$ together with the original counseling record, the LLM generates the depression assessment $A _ { t } ^ { d e p }$ following a reasoning skill informed by the ECI theory and PHQ-8 diagnostic algorithm (Kroenke et al., 2009). Since each assessment is conditioned on explicit observations and guided by these principles, the resulting reasoning is grounded in multimodal evidence. Consequently, each dialogue turn is annotated under the complete OAS paradigm.

Acoustic realization. The annotated scripts are rendered into emotional speech with the promptcontrolled text-to-speech synthesis model Qwen3- TTS (Hu et al., 2026), where dialogue transcripts $O _ { t } ^ { d t }$ serve as synthesis inputs, the acoustic emotion descriptions $O _ { t } ^ { a e }$ serve as speech-style prompts, and the speaker gender $g$ serves as the speaker selection reference.

Consequently, the pipeline yields AudiPsy-Counsel, denoted as $\mathcal { D } _ { \mathrm { a p c } }$ , a synthetic bilingual speech-counseling dataset with 2,400 conversations for downstream model training. All the details of the dataset construction, including prompt templates, quality-control procedures, and example instances are provided in Appendix B.

![](images/262ea823ca3d12f96cc20a2a2404a1243360a5c1cd6ad856938cf556d62e415e.jpg)  
Figure 2: Topic distribution of AudiPsy-Counsel.

## 3.3 Depression-oriented Reasoning

To equip $\mathcal { F } _ { \theta }$ with the capability of evidencegrounded multimodal reasoning for depression assessment, we optimize it through a three-stage training curriculum, as illustrated in Figure 1c. The curriculum progressively develops affective perception, structured counseling reasoning, and corrective preference alignment.

Stage-1: Affective perception. We first train $\mathcal { F } _ { \theta }$ to learn the observation field $O _ { t }$ from emotional speech to enable robust multimodal affective perception under both emotion-congruent and emotion-incongruent conditions. Specifically, we use IEMOCAP (Busso et al., 2008) as the spontaneous corpus, where we annotate the textual polarity of each utterance and construct a balanced training set by randomly downsampling each textual– acoustic emotion label combination.

We further construct a Synthetic Emotion-Incongruence (SynEI) dataset, a depression-related emotional speech corpus derived from the training split of DAIC-WOZ and EATD-Corpus, by altering the speech emotion through speech resynthesis (Zhou et al., 2026) while preserving the original text and speaker characteristics. Construction and dataset details are provided in Appendices C and D. Accordingly, the Stage-1 objective is defined as:

$$
\mathscr { L } _ { \mathrm { S 1 } } = - \mathbb { E } _ { \mathcal { D } _ { \mathrm { e m o } } } \log p _ { \theta } ( O _ { t } \mid x _ { t } ) .\tag{3}
$$

Stage-2: Structured counseling reasoning. We then continue supervised fine-tuning on AudiPsy-Counsel $D _ { a p c }$ to adapt the model to depressionoriented counseling reasoning. $\mathcal { F } _ { \theta }$ learns the complete OAS reasoning process with the Stage-2 objective defined as:

$$
\mathcal { L } _ { \mathrm { S 2 } } = - \mathbb { E } _ { \mathcal { D } _ { \mathrm { a p c } } } \log p _ { \theta } \left( \boldsymbol { O } _ { t } , \boldsymbol { A } _ { t } , \boldsymbol { S } _ { t } \mid \boldsymbol { x } _ { t } , \boldsymbol { S } _ { t - 1 } \right) .\tag{4}
$$

Stage-3: Corrective preference alignment. Finally, we align $\mathcal { F } _ { \theta }$ on real depression-assessment data (DAIC-WOZ; EATD-Corpus), denoted as $\mathcal { D } _ { \mathrm { r e a l } }$ . Considering its limited scale, we adopt the lightweight ORPO objective (Hong et al., 2024), which jointly performs supervised learning and preference optimization in a single stage. For EATD-Corpus, the counselor speech signals $c _ { t }$ are reconstructed using Qwen-TTS, as they are unavailable in the original dataset. During ORPO training, the Stage-2 output is treated as the rejected response, while a preferred response is regenerated using a pretrained speech emotion recognition (SER) model (Sailor et al., 2025) together with the AudiPsy-Counsel affective annotation pipeline conditioned on the ground-truth depression label. The model is optimized using the standard ORPO objective:

$$
\mathcal { L } _ { \mathrm { O R P O } } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { r e a l } } } \left[ \mathcal { L } _ { \mathrm { S F T } } + \lambda \mathcal { L } _ { \mathrm { O R } } \right] .\tag{5}
$$

## 4 Experiments

We evaluate AudiPsy from three perspectives: (1) depression assessment and human reasoning quality evaluation on DAIC-WOZ; (2) supplementary depression assessment on EATD-Corpus; and (3) multimodal evidence perception, utilization, and interpretation through speech emotion recognition, evidence utilization analysis, and reasoning case studies.

Training AudiPsy is initialized from Qwen2.5- Omni-7B (Xu et al., 2025a) and trained with the proposed three-stage curriculum using the ms-swift framework (Zhao et al., 2024) on 4 NVIDIA H800 GPUs. Stage-1 and -2 perform LoRA-based supervised fine-tuning on the affective perception corpus $\mathcal { D } _ { \mathrm { e m o } }$ (2,109 samples) and the AudiPsy-Counsel corpus $\mathcal { D } _ { \mathrm { a p c } }$ (2,400 samples), respectively. Stage 3 further aligns the model using ORPO on the depression-assessment corpus $\mathcal { D } _ { \mathrm { r e a l } }$ (280 conversations after conversation-level resampling to mitigate class imbalance).

Inference During inference, AudiPsy performs turn-by-turn reasoning, conditioning each turn on the previous summary. Dialogue-level predictions are obtained by majority voting, with ties resolved as yes. Greedy decoding (temperature = 0) ensures deterministic evaluation. Following prior work (Wu et al., 2023; Zhang et al., 2024b, 2025), we evaluate on the DAIC-WOZ development set (33 conversations: 12 depressed, 21 non-depressed)

for direct comparability. Separately, we evaluate on the EATD-Corpus validation set (79 conversations: 11 depressed, 68 non-depressed) to assess performance across conversational settings. Further implementation details and prompts are provided in Appendix E.

## 4.1 Depression Assessment Performance

Baselines We compare AudiPsy against representative speech-based binary depression classification methods, including BiLSTM+GRU (Shen et al., 2022b) and WavLM Layer 10 (Wu et al., 2023), as well as multi-stage LLM pipelines, including Speech Landmarks+Llama2 (Zhang et al., 2024b) and SpeechT-RAG (Zhang et al., 2025). We reproduce BiLSTM+GRU using its official implementation and test set, while reporting published single-model F1 results for the other methods.

For speech LLMs, we select models of comparable scale to AudiPsy’s Qwen2.5-Omni-7B backbone (Ding et al., 2025; Yang et al., 2026). For omni-LLMs, we include the Gemini 2.5 series (Comanici et al., 2025) as representative proprietary models, together with Qwen2.5-Omni-7B (Xu et al., 2025a) and the larger Qwen3-Omni-30B series (Xu et al., 2025b) from the same Omni family. A subset of these models is additionally evaluated under the OAS paradigm without task-specific finetuning. All evaluated models are listed in Table 1.

Metrics We evaluate AudiPsy from two perspectives: depression assessment performance and reasoning quality. For depression assessment, we report Matthews Correlation Coefficient (MCC), binary F1, recall, precision, and specificity. In addition to standard binary classification metrics, we include MCC and specificity to provide a balanced measure under class imbalance, accounting for performance on both depressed and non-depressed individuals (Chicco and Jurman, 2020).

For reasoning quality, four practicing psychological counselors evaluate 5 methods on 10 classbalanced conversations from DAIC-WOZ, examining 250 LLM outputs per evaluator. Reasoning quality is rated along four dimensions: Evidence Grounding (EG), Clinical Plausibility (CP), Multi-Modal Integration (MMI), and Contextual Evidence Integration (CEI). Evaluators assigned integer scores on a 1–5 scale. To accommodate dimension-specific rating baselines, we report mean-centered scores and win–tie–loss results across methods for each dimension. Detailed evaluation protocols are provided in Appendix F.

<table><tr><td>Method</td><td>MCC</td><td>F1</td><td>Recall</td><td>Precision</td><td>Specificity</td></tr><tr><td>BiLSTM + GRU*</td><td>0.11</td><td>0.53</td><td>0.56</td><td>0.56</td><td></td></tr><tr><td>WavLM Layer 10 (Wu et al., 2023)</td><td></td><td>0.72</td><td></td><td></td><td></td></tr><tr><td>Llama2 + Acoustic Landmarks (Zhang et al., 2024b)</td><td></td><td>0.72</td><td></td><td></td><td></td></tr><tr><td>SpeechT-RAG (Zhang et al., 2025)</td><td>-</td><td>0.69</td><td>-</td><td>一</td><td></td></tr><tr><td>Kimi-Audio-7B-Instruct (Ding et al., 2025)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>MOSS-Audio-8B-Instruct (Yang et al., 2026)</td><td>0.36</td><td>0.62</td><td>0.75</td><td>0.53</td><td>0.62</td></tr><tr><td>MOSS-Audio-8B-Thinking (Yang et al., 2026)</td><td>0.52</td><td>0.71</td><td>0.92</td><td>0.58</td><td>0.62</td></tr><tr><td>Gemini 2.5 Flash (Comanici et al., 2025)</td><td>0.61</td><td>0.76</td><td>0.92</td><td>0.65</td><td>0.71</td></tr><tr><td>Gemini 2.5 Pro (Comanici et al., 2025)</td><td>0.65</td><td>0.79</td><td>0.92</td><td>0.69</td><td>0.76</td></tr><tr><td>Qwen2.5-Omni-7B (Xu et al., 2025a)</td><td>0.24</td><td>0.42</td><td>0.33</td><td>0.57</td><td>0.82</td></tr><tr><td>Qwen3-Omni-30B-A3B-Instruct (Xu et al., 2025b)</td><td>0.19</td><td>0.56</td><td>1.00</td><td>0.39</td><td>0.10</td></tr><tr><td>Qwen3-Omni-30B-A3B-Thinking (Xu et al., 2025b)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td></tr><tr><td>Gemini 2.5 Pro (OAS)</td><td>0.00</td><td>0.53</td><td>1.00</td><td>0.36</td><td>0.00</td></tr><tr><td>Qwen2.5-Omni-7B (OAS)</td><td>0.35</td><td>0.62</td><td>0.83</td><td>0.50</td><td>0.52</td></tr><tr><td>Qwen3-Omni-30B-A3B-Instruct (OAS)</td><td>0.24</td><td>0.57</td><td>1.00</td><td>0.40</td><td>0.14</td></tr><tr><td>AudiPsy (Ours)</td><td>0.70</td><td>0.81</td><td>0.92</td><td>0.73</td><td>0.81</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { S 1 } }$ </td><td>0.58</td><td>0.74</td><td>0.83</td><td>0.67</td><td>0.76</td></tr><tr><td>w/o  $\scriptstyle { \mathcal { L } } _ { \mathrm { O R P O } }$ </td><td>0.65</td><td>0.79</td><td>0.92</td><td>0.73</td><td>0.76</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { S 1 } } + \mathcal { L } _ { \mathrm { O R P O } }$ </td><td>0.61</td><td>0.75</td><td>1.00</td><td>0.60</td><td>0.62</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { S 2 } } + \mathcal { L } _ { \mathrm { O R P O } }$ </td><td>0.53</td><td>0.71</td><td>0.83</td><td>0.63</td><td>0.71</td></tr><tr><td>w/o Emotion Incongruence</td><td>0.58</td><td>0.73</td><td>0.92</td><td>0.81</td><td>0.76</td></tr><tr><td>w/ Unstructured Rationale</td><td>0.65</td><td>0.69</td><td>0.92</td><td>0.79</td><td>0.76</td></tr></table>

Table 1: Depression assessment performance on the DAIC-WOZ dataset in terms of MCC, F1, recall, precision, and specificity.

Main results As shown in Table 1, AudiPsy attains the highest MCC, F1, and precision, outperforming specialized speech-based methods and general-purpose speech and omni-LLMs on DAIC-WOZ. Notably, Kimi-Audio fails to generate outputs for long audio inputs, while Qwen3-Omni-30B-A3B-Thinking frequently declines to make depression predictions.

Simply applying OAS to off-the-shelf models does not consistently improve performance. Although OAS explicitly structures multimodal evidence, Qwen2.5-Omni-7B often produces inaccurate observations and assessments, while Gemini 2.5 Pro and Qwen3-Omni-Instruct exhibit high recall but limited MCC, suggesting a tendency toward positive depression predictions. These results reinforce the coupling between evidence representation and contextual interpretation: structured reasoning alone is insufficient without reliable affective perception and task-specific training.

Ablation study We conduct six ablation experiments to examine the contributions of the training curriculum, emotion incongruence, and the structured OAS representation. Removing individual training objectives or their combinations generally degrades performance, showing the necessity of the full three-stage curriculum. In particular, removing ${ \mathcal L } _ { \mathrm { S 1 } }$ leads to a clear performance drop, highlighting the importance of affective perception for subsequent depression reasoning. Removing emotion incongruence also substantially degrades performance, demonstrating its importance as a source of multimodal evidence.

We further analyze their effects on affective perception and evidence utilization, respectively, in Section 4.4. We also rewrite the structured OAS fields into an unstructured natural-language rationale while preserving the same evidence content and passing it across dialogue turns during training. The resulting performance degradation shows that the benefit of OAS goes beyond rationale content itself, with its explicit structure facilitating evidence organization and information maintenance across dialogue turns.

## 4.2 Human Evaluation of Reasoning Quality

As shown in Figure 3, AudiPsy achieves the highest reasoning quality across all evaluation dimensions, complementing its strong performance on the binary depression assessment. For both Gemini 2.5 Pro and Qwen2.5-Omni-7B, applying OAS generally improves reasoning quality, particularly in MMI and CEI, demonstrating the benefit of explicitly organizing multimodal evidence and conversational context. However, Gemini 2.5 Pro (OAS) shows a slight decrease in CP, suggesting that structured reasoning alone does not necessarily improve the interpretation of multimodal evidence for depression assessment. Moreover, the win–tie–loss results show that AudiPsy is consistently preferred over the baselines, indicating that its higher relative scores reflect systematic improvements across evaluated cases rather than gains driven by a small number of examples.

![](images/c6ab361abcb0bbd66b0904a2237494ab9f3e18a3b7a1fbe4a9ab989864c37ece.jpg)  
(a) Relative scores.

![](images/fd58a529e7fbfa08209e0b399efda8e94a805b4e5e003e9552acbf83adc39a38.jpg)  
(b) Win–Tie–Loss comparison.  
Figure 3: Human evaluation results on the DAIC-WOZ dataset. (a) Relative scores of each method across four evaluation dimensions (EG, CP, MMI, and CEI), calculated as deviations from the mean score across methods. (b) Win–Tie–Loss percentages of AudiPsy against four baseline methods across the same evaluation dimensions.

## 4.3 Depression Assessment in Everyday Conversations

To complement the primary DAIC-WOZ evaluation, we assess AudiPsy on EATD-Corpus, which contains naturalistic conversations with less structured diagnostic context, posing additional challenges for depression assessment, as corroborated by practicing psychological counselors. As shown in Table 2, AudiPsy achieves the highest MCC and competitive F1, supporting its applicability across conversational settings. Consistent with

<table><tr><td>Method</td><td>MCC</td><td>F1</td></tr><tr><td>BiLSTM + GRU*</td><td>0.19</td><td>0.32</td></tr><tr><td>MOSS-Audio-8B-Thinking</td><td>0.27</td><td>0.37</td></tr><tr><td>Gemini 2.5 Flash Gemini 2.5 Pro</td><td>0.13 0.23</td><td>0.29 0.33</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>0.11</td><td>0.21</td></tr><tr><td>Qwen3-Omni-30B-A3B-Instruct Qwen3-Omni-30B-A3B-Thinking</td><td>0.20 0.32</td><td>0.33 0.40</td></tr><tr><td>AudiPsy (Ours)</td><td>0.41</td><td>0.40</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { S 1 } }$ </td><td>0.28</td><td>0.17</td></tr><tr><td>w/o Emotion Incongruence</td><td>-0.05</td><td>0.00</td></tr></table>

Table 2: Depression assessment performance on EATD-Corpus in terms of MCC and F1.

DAIC-WOZ, ablations highlight the importance of affective perception and emotion incongruence modeling, suggesting that modeling subtle affective cues and cross-modal relationships contributes to depression assessment in this setting.

## 4.4 Probing the Assessment Behavior

Perceptual foundation: cross-modal speech emotion recognition We first probe whether AudiPsy perceives speech emotion independently of textual sentiment, which is the perceptual foundation of incongruence analysis and affective evidence integration. Table 3 reports per-class F1 scores for four speech emotion categories on the IEMOCAP test set, stratified by transcript polarity. The Qwen2.5- Omni-7B backbone relies heavily on textual affective cues, struggling when speech emotion conflicts with transcript polarity. This tendency persists without ${ \mathcal { L } } _ { \mathrm { S 1 } }$ , whereas full AudiPsy improves emotion recognition, particularly for Neutral and Negative speech, consistent with the downstream ablation results in Table 1. Meanwhile, Gemini 2.5 Pro achieves comparable emotion recognition but substantially weaker depression assessment following OAS paradigm, suggesting that accurate affective perception alone is insufficient without appropriate evidence interpretation.

Evidence utilization: impact of emotion incongruence We next probe how explicit emotion incongruence modeling shapes the evidence used in assessment reasoning. Figure 4 summarizes the percentage of evidence types used in depression assessment reasoning across the DAIC-WOZ and EATD-Corpus test sets. We use GPT-5 to extract and categorize evidence types utilized in the rationale. The full AudiPsy exploits all five types of evidence in a balanced manner, whereas removing explicit emotion incongruence analysis causes the model to rely more heavily on textual cues while using substantially fewer acoustic emotion cues and accumulated multimodal patterns. This effect is particularly evident of the “Yes” prediction on EATD-Corpus, where less informative textual content leads the full model to naturally shift toward acoustic evidence, while the ablated model fails to exploit such complementary multimodal information. Consistent with this shift in evidence utilization, removing emotion incongruence modeling also degrades the overall depression assessment performance in Table 1. Together, these results suggest that explicit emotion incongruence analysis promotes more effective integration of multimodal evidence, thereby contributing to improved depression assessment.

<table><tr><td rowspan="2">Model</td><td colspan="4">Positive</td><td colspan="4">Neutral</td><td colspan="4">Negative</td></tr><tr><td>Happy</td><td>Angry</td><td>Neutral</td><td>Sad</td><td>Happy</td><td>Angry</td><td>Neutral</td><td>Sad</td><td>Happy</td><td>Angry</td><td>Neutral</td><td>Sad</td></tr><tr><td>Gemini 2.5 Pro Qwen2.5-Omni-7B</td><td>0.58 0.45</td><td>0.40 0.18</td><td>0.64 0.64</td><td>0.62 0.33</td><td>0.47 0.00</td><td>0.75 0.33</td><td>0.61 0.45</td><td>0.43 0.17</td><td>0.57 0.00</td><td>0.83 0.80</td><td>0.47 0.36</td><td>0.80 0.56</td></tr><tr><td>AudiPsy (Ours)</td><td>0.39</td><td>0.18</td><td>0.44</td><td>0.67</td><td>0.64</td><td>0.33</td><td>0.67</td><td>0.64</td><td>0.63</td><td>0.89</td><td>0.67 +.31</td><td>0.84</td></tr><tr><td>w/o Ls1</td><td>-.06 0.44 -.01</td><td>0.00 -.18</td><td>-.20 0.53 -.11</td><td>+.34 0.00 -.33</td><td>+.64 0.56 +.56</td><td>0.33</td><td>+.22 0.56 +.11</td><td>+.47 0.29 +.12</td><td>+.63 0.18 +.18</td><td>+.09 0.67 -.13</td><td>0.32 -.04</td><td>+.28 0.62 +.06</td></tr></table>

Table 3: Per-class F1 scores across transcript polarity groups. Smaller values below AudiPsy and its ablation indicate absolute F1 changes relative to the Qwen2.5-Omni-7B backbone; green and red denote improvements and degradations, respectively.

![](images/b9fc1084f89fe2b81fe37c83344912eb0d40175b1d035af51e4c483b5248e604.jpg)  
Figure 4: Occurrence rates of evidence types employed by AudiPsy and its ablation without Emotion Incongruence predicting “Yes (Depressed)” or “No (Nondepressed)”.

Evidence interpretation: contextualizing affective evidence for depression assessment Finally, we present two illustrative reasoning chains from selected dialogue turns in EATD-Corpus and DIAC-WOZ, respectively, in Table 4, to examine how

<table><tr><td>Case 1 (EATD-Corpus)</td><td>Case 2 (DAIC-WOZ)</td></tr><tr><td>Regretful reflection + persistent low VAD → Prior EI suggests masked distress → Accumulated depressive</td><td>Family concerns + sad speech → Prior EI, but context-consistent sadness → Limited cumulative</td></tr><tr><td>evidence ⇒ Depression assessment: Yes</td><td>depressive evidence ⇒ Depression assessment: No</td></tr></table>

Table 4: Illustrative reasoning chains showing contextualized interpretation of affective evidence.

AudiPsy interprets affective evidence within conversational context. In Case 1, recurring negative acoustic cues and prior EI, together with regretful reflection, form a consistent pattern of depressive evidence, supporting a positive assessment. In contrast, Case 2 also exhibits negative acoustic affect and prior EI, but the sadness is consistent with the seeker’s interpersonal concerns, with limited cumulative evidence supporting depression. The contrast highlights that similar affective cues can carry different implications for depression assessment depending on conversational context and accumulated evidence. Together, these cases illustrate AudiPsy’s ability to interpret multimodal affective evidence in context, rather than directly mapping emotional cues to depression predictions.

## 5 Conclusion

We presented AudiPsy, an omni-LLM for depression assessment based on the Observation– Assessment–Summary (OAS) reasoning paradigm. By progressively accumulating multimodal evidence and modeling textual–acoustic emotion incongruence, AudiPsy demonstrates the potential of evidence-grounded multimodal reasoning for interpretable depression assessment.

## Limitations

Evaluation Data and Protocols. Our evaluation is confined to DAIC-WOZ and EATD-Corpus, which are currently the only publicly available speech datasets for depression assessment. Largescale speech counseling datasets with depression labels remain scarce due to privacy concerns and high acquisition costs, limiting our ability to further evaluate the generalizability of our findings. Nevertheless, evaluating on these widely used benchmarks aligns with established practices in the field.

We follow the official train/test splits of both datasets, resulting in unavoidable class imbalance in testing. Besides, some prior studies on EATD-Corpus adopt random three-fold evaluation (Shen et al., 2022a), this setting is not readily applicable to our multi-stage training and alignment pipeline. Consequently, metrics reported under these different evaluation protocols could not be directly compared numerically.

Synthetic Counseling Data. AudiPsy-Counsel is constructed by augmenting real-world textual counseling records with affective annotations and synthesized speech. Its acoustic realizations may therefore differ from naturally occurring counseling speech in acoustic variability and emotional expression. AudiPsy-Counsel is primarily intended to provide controlled supervision for learning structured reasoning and interpreting multimodal affective evidence, rather than to approximate the distribution of real clinical speech.

## Ethical Considerations

This work is intended solely for research and experimental evaluation and should not be used for real-world clinical diagnosis or treatment decisions. Although AudiPsy is designed to reason over multimodal evidence for depression assessment, its predictions and generated rationales may contain clinically inappropriate interpretations. The reported performance on benchmark datasets should therefore not be interpreted as evidence of clinical validity or readiness for deployment. Depression assessment is a sensitive clinical task that requires qualified professional judgment, and outputs from our system should not replace evaluation by mental health professionals. Our experiments use publicly available research and synthetic datasets, which do not involve the collection of new patient data.

## References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. 2025. gpt-oss-120b and gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Nicolas Bertagnolli. 2020. Counsel-chat: Bootstrapping high-quality therapy data. https://huggingface. co/datasets/nbertagnolli/counsel-chat.

Carlos Busso, Murtaza Bulut, ChiChun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. 2008. IEMOCAP: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42(4):335–359.

Lauren M Bylsma, Bethany H Morris, and Jonathan Rottenberg. 2008. A meta-analysis of emotional reactivity in major depressive disorder. Clinical psychology review, 28(4):676–691.

Jingyi Chen, Zhimeng Guo, Jiyun Chun, Pichao Wang, Andrew Perrault, and Micha Elsner. 2026. Do audio llms really listen, or just transcribe? measuring lexical vs. acoustic emotion cues reliance. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5848–5877.

Zhuang Chen, Jiawen Deng, Jinfeng Zhou, Jincenzi Wu, Tieyun Qian, and Minlie Huang. 2024. Depression detection in clinical interviews with llm-empowered structural element graph. In Proceedings ofthe 2024 conference ofthe north american chapter ofthe associationfor computational linguistics: Human language technologies (volume 1: Long papers), pages 8181–8194.

Davide Chicco and Giuseppe Jurman. 2020. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC genomics, 21(1):6.

Lee Anna Clark and David Watson. 1991. Tripartite model of anxiety and depression: psychometric evidence and taxonomic implications. Journal ofabnormal psychology, 100(3):316.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, et al. 2025. Kimi-audio technical report. arXiv preprint arXiv:2504.18425.

Jonathan Gratch, Ron Artstein, Gale M Lucas, Giota Stratou, Stefan Scherer, Angela Nazarian, Rachel

Wood, Jill Boberg, David DeVault, Stacy Marsella, et al. 2014. The distress analysis interview corpus of human and computer interviews. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), volume 14, pages 3123–3128.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. Orpo: Monolithic preference optimization without reference model. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 11170–11189.

Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, Xinyu Zhang, Pei Zhang, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. 2026. Qwen3-TTS Technical Report. arXiv preprint arXiv:2601.15621.

Kurt Kroenke, Tara W Strine, Robert L Spitzer, Janet BW Williams, Joyce T Berry, and Ali H Mokdad. 2009. The phq-8 as a measure of current depression in the general population. Journal of affective disorders, 114(1-3):163–173.

Wenrui Liang, Rong Zhang, Xuezhen Zhang, Ying Ma, and Wei-Qiang Zhang. 2025. Depressgen: Synthetic data generation framework for depression detection. In Proceedings ofInterspeech, pages 464–468.

June M Liu, Mengxia Gao, Sahand Sabour, Zhuang Chen, Minlie Huang, and Tatia MC Lee. 2025. Enhanced large language models for effective screening of depression and anxiety. Communications Medicine, 5(1):457.

Adria Mallol-Ragolta, Ziping Zhao, Lukas Stappen, Nicholas Cummins, and Björn W Schuller. 2019. A hierarchical attention network-based approach for depression detection from transcribed clinical interviews. In Proceedings of Interspeech, pages 221– 225.

Kaining Mao, Yuqi Wu, and Jie Chen. 2023. A systematic review on automated clinical depression diagnosis. Npj mental health research, 2(1):20.

OpenAI. 2025. GPT-5. https://developers. openai.com/api/docs/models/gpt-5.

Hardik B Sailor, Aw Ai Ti, Chen Fang Yih Nancy, Chiu Ying Lay, Ding Yang, He Yingxu, Jiang Ridong, Li Jingtao, Liao Jingyi, Liu Zhuohan, et al. 2025. Meralion-ser: Robust speech emotion recognition model for english and sea languages. arXiv preprint arXiv:2511.04914.

Sara Sardari, Bahareh Nakisa, Mohammed Naim Rastgoo, and Peter Eklund. 2022. Audio based depression detection using convolutional autoencoder. Expert Systems with Applications, 189:116076.

Ying Shen, Huiyu Yang, and Lin Lin. 2022a. Automatic depression detection: An emotional audio-textual corpus and a gru/bilstm-based model. In Proceedings of

IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6247–6251.

Ying Shen, Huiyu Yang, and Lin Lin. 2022b. Automatic depression detection: An emotional audio-textual corpus and a gru/bilstm-based model. In Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6247– 6251.

Rongfeng Su, Changqing Xu, Hangbin Yu, Xinyi Wu, Feng Xu, Xie Chen, Lan Wang, and Nan Yan. 2026. Investigating acoustic-textual emotional inconsistency information for automatic depression detection. IEEE Transactions on Affective Computing.

Shuiyuan Wang, Zhixian Zhao, Hongfei Xue, Chengyou Wang, Shuai Wang, Hui Bu, Xin Xu, and Lei Xie. 2026. Humdial-eibench: A human-recorded multiturn emotional intelligence benchmark for audio language models. arXiv preprint arXiv:2604.11594.

Wenxuan Wang, Zizhan Ma, Meidan Ding, Shiyi Zheng, Shengyuan Liu, Jie Liu, Jiaming Ji, Wenting Chen, Xiang Li, Linlin Shen, et al. 2025. Medical reasoning in the era of llms: a systematic review of enhancement techniques and applications. arXiv preprint arXiv:2508.00669.

Lawrence L. Weed. 1968. Medical records that guide and teach. New England Journal of Medicine, 278(12):652–657.

World Health Organization. 2026. Depressive disorder (depression). https://www.who.int/news-room/ fact-sheets/detail/depression.

Wen Wu, Chao Zhang, and Philip C Woodland. 2023. Self-supervised representations in speech-based depression detection. In Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5.

Xinyi Wu, Changqing Xu, Nan Li, Rongfeng Su, Lan Wang, and Nan Yan. 2024. Depression enhances internal inconsistency between spoken and semantic emotion: Evidence from the analysis of emotion expression in conversation. In Proceedings ofInterspeech, pages 4219–4223.

Danai Xezonaki, Georgios Paraskevopoulos, Alexandros Potamianos, and Shrikanth Narayanan. 2020. Affective conditioning on hierarchical networks applied to depression detection from transcribed clinical interviews. In Proceedings ofInterspeech, pages 4556–4560.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025a. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. 2025b. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765.

Xuhai Xu, Bingsheng Yao, Yuanzhe Dong, Saadia Gabriel, Hong Yu, James Hendler, Marzyeh Ghassemi, Anind K Dey, and Dakuo Wang. 2024. Mentalllm: Leveraging large language models for mental health prediction via online text data. Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies, 8(1):1–32.

Chen Yang, Chufan Yu, Hanfu Chen, Jie Zhu, Jingqi Chen, Ke Chen, Wenxuan Wang, Yang Wang, Yaozhou Jiang, Yi Jiang, et al. 2026. Moss-audio technical report. arXiv preprint arXiv:2606.01802.

Chenhao Zhang, Renhao Li, Minghuan Tan, Min Yang, Jingwei Zhu, Di Yang, Jiahao Zhao, Guancheng Ye, Chengming Li, and Xiping Hu. 2024a. CPsyCoun: A report-based multi-turn dialogue reconstruction and evaluation framework for chinese psychological counseling. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13947– 13966.

Xiangyu Zhang, Hexin Liu, Kaishuai Xu, Qiquan Zhang, Daijiao Liu, Beena Ahmed, and Julien Epps. 2024b. When llms meets acoustic landmarks: An efficient approach to integrate speech into large language models for depression detection. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 146–158.

Xiangyu Zhang, Hexin Liu, Qiquan Zhang, Beena Ahmed, and Julien Epps. 2025. SpeechT-RAG: Reliable depression detection in llms with retrievalaugmented generation using speech timing information. In Findings of the Association for Computational Linguistics: ACL 2025, pages 10019–10030.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. 2024. Swift:a scalable lightweight infrastructure for fine-tuning. arXiv preprint arXiv:2408.05517.

Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, and Jingchen Shu. 2026. Indextts2: A breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 35139–35148.

Yakun Zhu, Zhongzhen Huang, Linjie Mu, Yutong Huang, Wei Nie, Jiaji Liu, Shaoting Zhang, Pengfei Liu, and Xiaofan Zhang. 2026. DiagnosisArena: benchmarking diagnostic reasoning for large language models. In Findings of the Association for Computational Linguistics: ACL 2026, pages 3074– 3098.

## A Details of OAS Reasoning Paradigm

## A.1 OAS Template

We introduce the Observation–Assessment– Summary (OAS) structured reasoning paradigm.

The paradigm organizes the model output into three structured fields: Observation, Assessment, and Summary. This design encourages the model to explicitly separate evidence extraction, cross-modal emotion incongruence analysis and depression assessment, along with updated conversation summary during multimodal depression reasoning.

The Observation records turn-level multimodal evidence, including the dialogue transcript, textual emotion and polarity, speech acoustic emotion and category label, and discretized valence–arousal– dominance (VAD) attributes for acoustic emotion.

The Assessment performs higher-level reasoning based on the extracted evidence. Specifically, the model evaluates cross-channel emotional incongruence between linguistic and vocal expressions, and assesses depressive tendency by considering both current-turn observations and accumulated dialogue context.

The Summary block maintains a compact conversation-level representation consisting of a natural-language conversation summary and cumulative affective statistics (e.g., counts of negativepolarity turns, low-valence speech turns, and emotionally incongruent turns). This summary is recursively updated after each dialogue turn and serves as the conversation context for subsequent reasoning steps.

OAS’s fields and their value are summarized in Table 5. The complete OAS template is shown below.

OAS template   
[OBSERVATION]   
\*Turn Context\*   
Transcript:   
(Counselor) {question}   
(Seeker) {answer}   
\*Seeker Analysis\*   
Text Emotion={description}   
Text Polarity={polarity\_label}   
Speech Emotion={description}   
Speech Emotion Category={category\_label}   
VAD=({V},{A},{D})   
[/OBSERVATION]   
[ASSESSMENT]   
Emotion Incongruence={Yes|No}{rationale}   
Depression={Yes|No}{rationale}   
[/ASSESSMENT]   
[SUMMARY\_UPDATE]   
Turns={n}   
Conversation Summary={summary}   
Stats:   
Negative transcript={x1}/{n}

Low valence={x2}/{n}   
Low arousal={x3}/{n}   
Low dominance={x4}/{n}   
Emotion incongruence={x5}/{n}   
[/SUMMARY\_UPDATE]

## A.2 Observation Field Annotation

The Observation field contains six attributes: Dialgoue Transcript, Textual Emotion, Text Polarity, Acoustic Emotion, Acoustic Emotion Category, and VAD. Their annotation procedures are described below.

Dialogue Transcript. Each dialogue is represented as an ordered sequence of alternating utterances between the Counselor and the Seeker. The original speaker order is preserved. The ground truth is from the original dataset.

Textual Emotion. The Text Emotion attribute is generated using GPT-OSS-120B (Agarwal et al., 2025) based solely on the dialogue transcript using the prompt presented in Appendix G.3 to produce a natural-language description of the textual emotional state.

Text Polarity. The Text Polarity attribute classifies the transcript into one of three sentiment polarity categories: Positive, Neutral, or Negative. The annotation is performed using GPT-OSS-120B, using the prompt presented in Appendix G.4.

Acoustic Emotion. The annotation procedure depends on the source dataset. For AudiPsy-Counsel, the acoustic emotion description is directly obtained from the synthesis prompt used to generate the speech. For DAIC-WOZ and EATD-Corpus, acoustic emotion is predicted using a speech emotion recognition (SER) model (Sailor et al., 2025) and hold the same value as Speech Emotion Category.

Acoustic Emotion Category. The categorical acoustic emotion is predicted by the SER model (Sailor et al., 2025), which outputs one of seven categories: Anger, Happy, Neutral, Sad, Surprise, Disgust, and Fear.

VAD. The valence, arousal, and dominance (VAD) scores are predicted as continuous values in the range [0, 1] using the SER model (Sailor et al., 2025). Each dimension is then discretized into five equal-width intervals: [0, 0.2) (Very Low), [0.2, 0.4) (Low), [0.4, 0.6) (Medium), [0.6, 0.8) (High), and [0.8, 1] (Very High).

## A.3 Assessment Field Annotation

The Assessment field contains two attributes: Emotion Incongruence and Depression. Their annotation procedures are described below.

Emotion Incongruence. The Emotion Incongruence attribute indicates whether an inconsistency exists between the textual and acoustic emotional expressions of the speaker. The annotation procedure depends on the source dataset. For AudiPsy-Counsel, the ground-truth incongruence labels are obtained during dataset construction, and GPT-5 is employed to generate a natural-language rationale explaining the label using the system and user prompts in Appendix G.5.

For DAIC-WOZ and EATD-Corpus, the dataset does not provide emotion incongruence labels. Therefore, GPT-5 is first required to determine whether emotion incongruence is present based on the transcript and acoustic emotion annotations, and then generate a brief rationale for its decision. We employ the system prompt described in Appendix G.6, together with the same user prompt used for AudiPsy-Counsel in Appendix G.5.

Depression. The Depression attribute indicates whether the speaker exhibits depressive symptoms and provides a corresponding explanation. For AudiPsy-Counsel, the dialogue topic serves as the depression label. For DAIC-WOZ and EATD-Corpus, the depression label is derived from the dataset questionnaire results. A skill informed by PHQ-8/SDS and the ECI theory is combined with GPT-5 to produce a natural-language explanation, with the prompts presented in Appendix G.7.

## A.4 Summary Field Annotation

The summary field provides a compact overview of each dialogue and its overall emotional characteristics. The conversation summary is generated based on the complete dialogue till the current turn, using GPT-OSS-120B and the prompt presented in Appendix G.8. The statistical attributes are computed by aggregating annotations across the conversation. For each emotional attribute, the corresponding statistic is reported as the ratio of utterances satisfying the criterion (e.g., negative sentiment or low valence) to the total number of utterances, yielding a set of conversation-level summaries that can be used for efficient filtering and retrieval.

<table><tr><td>Field</td><td>Attribute</td><td>Value</td></tr><tr><td rowspan="6">Observation</td><td>Dialogue Transcript</td><td>Text dialogue content by counselor and seeker</td></tr><tr><td>Textual Emotion</td><td>Natural-language description</td></tr><tr><td>Text Polarity</td><td>Positive / Neutral / Negative</td></tr><tr><td>Acoustic Emotion</td><td>Natural-language description</td></tr><tr><td>Acoustic Emotion Category</td><td>Anger / Happy / Neutral / Sad / Surprise / Disgust / Fear</td></tr><tr><td>VAD</td><td>Very Low / Low / Medium / High / Very High</td></tr><tr><td rowspan="2">Assessment</td><td>Emotion Incongruence</td><td>Yes / No, with rationale</td></tr><tr><td>Depression</td><td>Yes / No, with rationale</td></tr><tr><td rowspan="7">Summary</td><td>Turns</td><td>Integer</td></tr><tr><td>Conversation Summary</td><td>Natural-language summary</td></tr><tr><td>Negative Transcript</td><td>x/n</td></tr><tr><td>Low Valence</td><td>x/n</td></tr><tr><td>Low Arousal</td><td>x/n</td></tr><tr><td>Low Dominance</td><td>x/n</td></tr><tr><td>Emotion Incongruence</td><td>x/n</td></tr></table>

Table 5: Attributes and corresponding value specifications for each field in the OAS reasoning paradigm.

## A.5 Example OAS data

The following example illustrates the data under OAS paradigm, constructed from the third dialogue turn of a conversation in the EATD-Corpus. For readability, the original Chinese dialogue is translated into English, while the model has the capability to transcribe bilingual data. The other annotations are in English. The example illustrates how observations from both text and speech are integrated into the Observation field, followed by LLM-based Assessment and dialogue-level Summary updates.

```ini
Example OAS data
[CURRENT_TURN_OBSERVATION]
*Turn Context*
Transcript:
(Counselor) What kind of person do you like?
(Seeker) Someone cheerful and warm-hearted.
It's not that there is
anything specific that attracts me; I
simply feel at ease around them.
*Seeker Analysis*
Transcript Emotion:
warm, cheerful, reassuring
Transcript Polarity:
positive
Speech Emotion:
Sad
Speech Emotion Category:
Sad
Speech VAD:
Valence = low
Arousal = very low
Dominance = low
```

[/CURRENT\_TURN\_OBSERVATION]

[ASSESSMENT]

Cross-channel Emotion Incongruence (Current Turn):

[Answer] Yes

[Reason] The text conveys a positive, warm and reassuring sentiment, while the speech is sad with very low arousal, low dominance and low valence, so the low-energy negative vocal tone clearly contradicts the positive meaning.

Depressive Tendency Assessment (Cumulative Evidence):

[Reason] Across multiple turns, the user's voice has remained sad with very low arousal and low dominance, even while describing a warm, reassuring person. This persistent cross-channel incongruence and flattened affect point to reduced positive reactivity/emotion context insensitivity and suggest low energy/psychomotor slowing rather than a purely situational state. They also acknowledge feeling upset online-though minimizing it as fleeting-which hints at ongoing negative affect.

[/ASSESSMENT]

[SUMMARY\_UPDATE]

Processed Turns: 3

Conversation Content Summary: The seeker feels uncertain and emotionally impulsive when engaging with the internet, and seeks reassurance and comfort through

relationships with youthful, energetic,   
cheerful, and warm-hearted people.   
Cumulative Trends:   
Negative-polarity transcript turns: 0 / 3   
Low-valence speech turns: 3 / 3   
Low-arousal speech turns: 3 / 3   
Low-dominance speech turns: 3 / 3   
Emotionally incongruent turns: 2 / 3   
[/SUMMARY\_UPDATE]

## B AudiPsy-Counsel Dataset Counstruction

## B.1 Situational Augmentation and Affective Annotation

To construct the textual component of AudiPsy-Counsel, we collect source materials from two psychological counseling datasets: CounselChat (Bertagnolli, 2020) and CPsy-CounR (Zhang et al., 2024a). These datasets contain authentic mental health discussions and counseling reports, providing realistic scenarios for depression-oriented dialogue generation. Given each source instance, we prompt an LLM to rewrite it into a counseling dialogue between a seeker and a counselor in both English and Chinese. This process transforms static counseling records into interactive conversations suitable for subsequent speech synthesis and multimodal training.

In addition to the dialogue content, the LLM generates several affective annotations required by AudiPsy. For each seeker utterance, it produces a acoustic emotion description and an emotion incongruence label indicating whether the vocal emotion is consistent with the semantic content. To ensure sufficient supervision for emotion incongruence reasoning, each generated dialogue contains at least one incongruent seeker utterance. The LLM also infers the seeker’s gender (male,female, or unknown), which is later used for speaker assignment during speech synthesis. The resulting dataset therefore provides counseling conversations with rich affective annotations that support emotion perception, incongruence detection, and depression-oriented reasoning.

We use a unified prompt (Appendices G.1 and G.2) to convert each counseling record into a threeturn dialogue with basic affective annotation. For other affective attribute annotation, please refer to Appendix A.

## B.2 Acoustic Realization

We convert the generated counseling conversations into speech using the prompt-controlled text-tospeech system Qwen3-TTS (Hu et al., 2026). For each utterance, the dialogue text is used as the synthesis input, while the corresponding emotion description generated during dialogue construction is provided as an instruction to control the emotional speaking style.

Since Qwen3-TTS provides a fixed set of predefined voice identities, we assign speakers before speech synthesis. For seeker utterances, a voice is randomly sampled from the predefined speaker pool according to the predicted gender. Specifically, voices are sampled from the corresponding male or female speaker pool when the predicted gender is Male or Female, respectively, while utterances labeled as Unknown are assigned a random voice from the entire speaker pool. Counselor voices are sampled independently from the predefined speaker set. Once assigned, the same voice identity is used consistently for all utterances of the corresponding speaker throughout the dialogue.

Through this process, each bilingual counseling dialogue is transformed into a multi-speaker emotional speech interaction. Combined with the dialogue content, emotion descriptions, emotion incongruence labels, and speaker profiles, the resulting AudiPsy-Counsel dataset provides rich multimodal supervision for emotion perception, incongruence detection, and depression-oriented counseling analysis.

## B.3 Quality Check.

Since Qwen3-TTS generally produces reliable speech content and speaker identities, our qualitycontrol procedure primarily focuses on verifying whether the synthesized speech conveys the intended emotion and preventing potential emotion hallucinations. Specifically, we first use an LLM to map each free-form emotion description into one of seven canonical emotion categories: Neutral, Happy, Sad, Angry, Surprised, Disgust, and Fear. We then apply a speech emotion recognition (SER) model (Sailor et al., 2025) to the synthesized audio. If the emotion predicted by the SER model is inconsistent with the category derived from the emotion description, the audio is discarded and regenerated. This verification-and-regeneration process is repeated until the two emotion labels are consistent, with a maximum of 50 generation at-

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total Dialogues</td><td>11,818</td></tr><tr><td>Total Utterances</td><td>70,908</td></tr><tr><td>Hours of Speech</td><td>163.75</td></tr><tr><td>CounselChat / CpsyCounR (Dia- logue)</td><td>5,550 / 6,268</td></tr><tr><td>English / Chinese (Dialogue)</td><td>5,909 / 5,909</td></tr><tr><td>Male / Female (Dialogue)</td><td>39,018 / 31,890</td></tr><tr><td>Depression (Dialogue) Positive / Negative</td><td>1,330 / 10,488</td></tr><tr><td></td><td></td></tr><tr><td>Emotion Incongruence (Utterance) Positive / Negative</td><td>11,818 / 23,572</td></tr><tr><td>Text Emotion (Utterance)</td><td></td></tr><tr><td>Positive / Negative / Neutral</td><td>4,816 / 25,261 / 5,377</td></tr><tr><td>Speech Emotion (Utterance)</td><td></td></tr><tr><td>Neu / Hap / Sad</td><td>13,475 / 7,764 / 8,448</td></tr><tr><td>Ang / Sur / Dis / Fea</td><td>1,187 / 753 / 50 / 3,777</td></tr></table>

Table 6: Statistics of AudiPsy-Counsel.

tempts per utterance.

## B.4 Dialogue Augmentation

The generated counseling conversations are initially constructed as three-turn conversations for efficient and consistent data generation. However, real-world counseling sessions, such as those in DAIC-WOZ, typically involve substantially longer interactions. To reduce this discrepancy, we further extend the conversations using two simple augmentation strategies.

First, we prepend several short rapport-building exchanges (e.g., greetings and opening questions) before the original dialogue to simulate the natural opening phase of counseling sessions. Second, we concatenate multiple generated conversations sharing the same counseling topic into a longer conversation. The transitions between adjacent dialogue segments are rewritten by the LLM to ensure supportive coherence and conversational continuity.

These dialogue augmentation strategies increase the diversity of dialogue lengths while preserving the original counseling content and affective annotations, enabling the synthesized corpus to better resemble real-world multi-turn counseling conversations.

## B.5 Detailed Statistics

Table 6 summarizes the statistics of AudiPsy-Counsel. The dataset contains 11,818 synthesized counseling conversations with approximately 100 hours of speech. Dialogues are constructed from both CounselChat and CPsyCounR and are evenly distributed between English and Chinese. The corpus covers a diverse set of psychological counseling scenarios, including depression, relationships, self-development, and other common mental health concerns. We randomly sample 2,400 conversations in Stage-2 training to match the size of data used in Stage-1 and Stage-3.

Beyond its scale and topical diversity, AudiPsy-Counsel provides rich affective supervision for depression-oriented counseling analysis. The dataset includes balanced speaker demographics, emotion incongruence labels, and depressionrelated assessments. In addition, acoustic emotion and textual emotion distributions exhibit substantial diversity across multiple affective categories, supporting the study of cross-modal emotional perception and emotion incongruence reasoning.

## B.6 Example Data

Table 7 presents an example instance from AudiPsy-Counsel. Each sample consists of a three-turn counseling dialogue between a seeker and a Counselor. In addition to the dialogue content, the dataset provides acoustic emotion descriptions and emotion incongruence labels for each seeker utterance, together with speaker metadata used for emotional speech synthesis.

## C IEMOCAP Annotation and Filtering

IEMOCAP (Busso et al., 2008) is a multimodal conversational emotion dataset containing natural spoken interactions with manually annotated acoustic emotions, making it well suited for Stage-1 affective perception. Our Stage-1 training operates on individual utterances rather than question–answer conversations, so each utterance is treated as an independent training sample. Following the official split, Sessions 1–4 are used for training and Session 5 is reserved for evaluation.

We annotate the textual emotional polarity of every utterance using GPT-OSS-120B with three categories: positive, neutral, and negative. The annotation is performed solely based on the textual transcript, independent of the original acoustic emotion labels.

To obtain a balanced distribution over acoustic emotion and text polarity combinations, we perform class-wise downsampling within each text polarity category. Specifically, for each polarity, utterances belonging to the same acoustic emotion are randomly sampled to produce a more balanced distribution while preserving as much diversity as possible. The resulting training set contains 1,031 utterances. Detailed statistics are shown in Table 8.

<table><tr><td>Turn</td><td>Speaker</td><td>Utterance</td><td>Acoustic Emotion</td><td>EI</td></tr><tr><td>1</td><td>Seeker</td><td>People always think success keeps you safe, but lately I&#x27;ve felt exhausted and emotionally numb despite all my achievements.</td><td>indifferent, slightly amused</td><td>√</td></tr><tr><td>1</td><td>Counselor</td><td>It&#x27;s more common than you think. Even those who seem successful on the outside may struggle with depression, especially when emotions have to be hidden under a strong exterior.</td><td>understanding, gentle</td><td></td></tr><tr><td>2</td><td>Seeker</td><td>I really don&#x27;t find joy in my daily life anymore, and it&#x27;s hard to share that with people who see me as someone who “has it all.&quot;</td><td>downcast, hesi- tant</td><td>×</td></tr><tr><td>2</td><td>Counselor</td><td>That pressure to always maintain your image can be exhausting. It can actually make it more difficult to express negative feelings or ask for help.</td><td>warm, emphasiz-</td><td></td></tr><tr><td>3</td><td>Seeker</td><td>So maybe I should talk to close friends or professionals instead of bottling it up?</td><td>ing tentative, seeking reassurance</td><td>×</td></tr><tr><td>3</td><td>Counselor</td><td>Absolutely. Reaching out is an important step. Taking time for personal relaxation and lowering your guard around trusted people can really help portive</td><td>encouraging, sup-</td><td></td></tr></table>

Table 7: An example instance from AudiPsy-Counsel (Topic: Depression & Self-harm). The first seeker utterance is intentionally annotated as emotional incongruence (EI), where the expressed vocal emotion is inconsistent with the semantic content. Metadata: Source = CPsyCounR, Language = English, Topic = Depression & Self-harm, Gender = Male.

<table><tr><td>Speech Emotion</td><td>Neg.</td><td>Neu.</td><td>Pos.</td><td>Total</td></tr><tr><td>Angry</td><td>45</td><td>45</td><td>45</td><td>135</td></tr><tr><td>Disgust</td><td>2</td><td>0</td><td>0</td><td>2</td></tr><tr><td>Fear</td><td>2</td><td>26</td><td>2</td><td>30</td></tr><tr><td>Happy</td><td>90</td><td>90</td><td>90</td><td>270</td></tr><tr><td>Neutral</td><td>90</td><td>90</td><td>89</td><td>269</td></tr><tr><td>Sad</td><td>90</td><td>90</td><td>90</td><td>270</td></tr><tr><td>Surprised</td><td>24</td><td>59</td><td>6</td><td>89</td></tr><tr><td>Total</td><td>343</td><td>400</td><td>322</td><td>1065</td></tr></table>

Table 8: Distribution of the filtered IEMOCAP training set across acoustic emotion and text polarity categories.

Since both the training set and the original Session-5 contains very limited samples for fear, disgust, and surprise, we evaluate only the four major acoustic emotions (angry, happy, neutral, and sad). For each acoustic emotion, we uniformly sample 10 utterances from each text polarity category, resulting in a balanced test set of 120 utterances.

## D Synthetic Emotion-Incongruence (SynEI) Dataset Construction

To expose the model to textual–acoustic emotion incongruence during Stage-1 training, we construct a Synthetic Emotion-Incongruence (SynEI) dataset using the training splits of DAIC-WOZ and EATD-Corpus.

We first group all utterances according to their annotated text polarity (positive, neutral, and negative). Utterances are randomly sampled from each polarity category to obtain a balanced text polarity distribution. Each selected utterance is then synthesized with seven target acoustic emotions (neutral, happy, sad, angry, fearful, disgusted, and surprised) using IndexTTS2 (Zhou et al., 2026). During synthesis, the speaker prompt is extracted from the original utterance so that the synthesized speech preserves the original speaker identity while modifying only the emotional expression.

<table><tr><td>Speech Emotion</td><td>Neg.</td><td>Neu.</td><td>Pos.</td><td>Total</td></tr><tr><td>Neutral</td><td>56</td><td>56</td><td>56</td><td>168</td></tr><tr><td>Happy</td><td>52</td><td>52</td><td>52</td><td>156</td></tr><tr><td>Sad</td><td>147</td><td>147</td><td>147</td><td>441</td></tr><tr><td>Angry</td><td>58</td><td>58</td><td>58</td><td>174</td></tr><tr><td>Fearful</td><td>23</td><td>27</td><td>11</td><td>61</td></tr><tr><td>Disgusted</td><td>21</td><td>7</td><td>1</td><td>29</td></tr><tr><td>Surprised</td><td>14</td><td>27</td><td>8</td><td>49</td></tr><tr><td>Total</td><td>371</td><td>377</td><td>333</td><td>1078</td></tr></table>

Table 9: Emotion Distribution of the SynEI dataset

To ensure the quality of the synthesized samples, we perform an automatic filtering step using a pretrained SER model (Sailor et al., 2025). Only samples whose predicted acoustic emotion matches the intended target emotion are retained. We further downsample the retained samples within each acoustic emotion category to obtain a balanced distribution across the three text polarity categories. Tables 9 summarize the final Synthetic EI Dataset after SER-based filtering and downsampling.

<table><tr><td>Hyperparameter</td><td>Stage-1</td><td>Stage-2</td><td>Stage-3</td></tr><tr><td>Objective</td><td>SFT</td><td>SFT</td><td>ORPO</td></tr><tr><td>Epochs</td><td>3</td><td>3</td><td>1</td></tr><tr><td>Learning rate</td><td>2×10−⁵</td><td>5×10 -5</td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Per-device batch size</td><td>8</td><td>4</td><td>4</td></tr><tr><td>Gradient accumulation</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Max sequence length</td><td>2048</td><td>3072</td><td>3072</td></tr><tr><td>LoRA rank</td><td>16</td><td>16</td><td>8</td></tr><tr><td>LoRA alpha</td><td>32</td><td>32</td><td>16</td></tr><tr><td>Warmup ratio</td><td>0.05</td><td>0.05</td><td>0.1</td></tr></table>

Table 10: Stage-specific training hyperparameters.

## E Training and Inference Details

Training parameters AudiPsy is trained with a three-stage curriculum using the ms-swift framework (Zhao et al., 2024). The first two stages adopt LoRA-based supervised fine-tuning (SFT), while the final stage performs preference optimization with ORPO. All experiments are conducted on 4 NVIDIA H800 GPUs. Table 10 summarizes the training hyperparameters used in each stage.

Training and Inference prompts The system prompt and user prompt are presented in Appendix G.9. The user prompt has two variants depending on the dialogue context: one is used when a summary of previous turns is available, and the other is used when no previous-turn summary is provided.

## F Human Evaluation of Reasoning

## F.1 Evaluation Protocol

We recruited four practicing psychological counselors from established mental health counseling institutions as human experts to evaluate the reasoning generated by five methods. The experts were recruited through personal referrals and were compensated based on their standard hourly consultation rates. The compensation was aligned with their usual professional fees to ensure fair payment for their time and expertise. All experts were informed of the evaluation task and compensation arrangements in advance and voluntarily agreed to participate. We collected only their evaluation scores and did not collect any personal or sensitive information. The evaluation did not involve clinical interventions or the collection of patient data.

The evaluation set consists of 10 conversations randomly sampled from DAIC-WOZ, balanced between depressed and non-depressed cases. Among the five methods, three follow the OAS paradigm and produce turn-level reasoning, while the other two generate a single dialogue-level reasoning output. We randomly sample five turn-level outputs from each OAS-based method, totaling 250 model outputs. Evaluators consider the sampled turn-level outputs jointly and assign one dialogue-level score to each method under each criterion. Thus, each evaluator provides scores for 50 method–dialogue pairs while examining 140 individual model outputs.

For each criterion, evaluators assessed dialoguelevel reasoning using integer scores on a 1–5 Likert scale. They were instructed to make comparative judgments among the methods, assigning higher scores to stronger methods and lower scores to weaker ones. As a reference, a method judged slightly above the average quality of the compared methods received a 3, while one judged slightly below average received a 2. We use numerical ratings rather than rankings to capture both the relative ordering of methods and the perceived magnitude of their differences. Accordingly, we report mean-centered scores that reflect each method’s performance relative to the average of the compared methods.

For each method m and criterion c, we first aggregate its ratings across evaluators and dialogue samples:

$$
S _ { m , c } = { \frac { 1 } { | { \mathcal { E } } | | { \mathcal { D } } | } } \sum _ { e \in { \mathcal { E } } } \sum _ { d \in { \mathcal { D } } } s _ { e , d , m , c } ,
$$

where $s _ { e , d , m , c }$ denotes the rating assigned by evaluator e to method m on dialogue d under criterion c. We then report the mean-centered relative score

$$
R _ { m , c } = S _ { m , c } - \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { m ^ { \prime } \in \mathcal { M } } S _ { m ^ { \prime } , c } ,
$$

where M denotes the set of evaluated methods. A positive (negative) $R _ { m , c }$ indicates that a method is rated above (below) the average across compared methods for criterion c.

All four human evaluators were Chinese nationals and native Chinese speakers. Since the model outputs and DAIC-WOZ conversations were in English, we additionally provided GPT-5-generated Chinese translations for reference. The original English text was presented alongside the translation to facilitate understanding while preserving the original model outputs.

## F.2 Evaluation Criteria

We evaluate reasoning quality along four complementary dimensions. Evaluators assign higher scores when the generated reasoning better satisfies the corresponding criterion and lower scores otherwise.

Evidence Grounding (EG) This criterion evaluates whether the reasoning identifies depressionrelevant evidence from the input speech, such as linguistic content, emotional expression, and vocal affect, and grounds its assessment in observable evidence rather than unsupported speculation. Stronger reasoning identifies relevant evidence and clearly connects it to the assessment.

Clinical Plausibility (CP) This criterion evaluates whether the reasoning is consistent with established psychological and clinical knowledge of depression. Stronger reasoning provides clinically plausible interpretations of observed symptoms, behaviors, and risk factors while avoiding unsupported or unreasonable clinical inferences.

Multi-Modal Integration (MMI) This criterion evaluates whether the reasoning integrates information across modalities, particularly linguistic content and vocal characteristics, rather than relying predominantly on a single modality. Stronger reasoning considers the consistency, complementarity, or discrepancy between modalities and incorporates these relationships into the assessment.

Contextual Evidence Integration (CEI) This criterion evaluates whether the reasoning integrates evidence across the dialogue context rather than drawing conclusions from isolated observations. Stronger reasoning connects information across dialogue turns and identifies cumulative evidence, recurring patterns, or changes over the course of the interaction.

## F.3 Instructions for Human Evaluators

The human evaluators were provided with detailed task instructions in Chinese, covering the evaluation procedure, dialogue materials, model output formats, and scoring process. The original instruction slides and criteria are included in the data supplementary materials to ensure transparency and reproducibility.

## G LLM Prompts

## G.1 AudiPsy-Counsel Situational Augmentation and Affective Annotation Prompt

System Prompt   
You are given:   
A mental health counseling title and the   
question (from a seeker).   
A counseling response (from a counselor).   
Your Task   
Rewrite the content into an {English |   
Chinese} 3-turn spoken conversation between   
a Seeker and a mental Counselor.   
You need to write the "Text" and the   
"Emotion" for each utterance.   
The "Text" is the exactly the text content   
of each utterance. What the speaker says.   
The "Emotion" is the description of the   
acoustic emotion when the speaker speaks   
the "Text".   
Rules of the "Text":   
The "Text" should be less than 3 sentences   
to make the spoken conversation natural   
Do not include any emotion description here.   
Rules of the "Emotion":   
For the acoustic emotion, use a set of   
adjective or phrases to describe it.   
You should only describe the emotion that   
the speaker sounds like. Do not describe   
the hidden emotion.   
Rules of the "Emotion Incongruence":   
For the seeker, there must be at least one   
utterance that demonstrates text-speech   
Emotion Incongruence. It means that the   
emotion of the text content is different   
from the acoustic emotion when he talks   
about it.   
Example when emotion incongruence is true:   
"Text": "I feel hopeless."   
"Emotion": "positive, happy"   
"Emotion Incongruence": true   
Set "Emotion Incongruence" to true in this   
situation.   
Output Format   
Return strictly valid JSON in the following   
structure:   
{   
"1": {   
"Seeker": {   
"Text": ".   
"Emotion":   
"Emotion Incongruence": true   
},   
"Counselor": {   
"Text": ".   
"Emotion": ".   
}   
},   
"2": {   
"Seeker": {

"Text": "...",   
"Emotion": "..."   
"Emotion Incongruence": false   
},   
"Counselor": {   
"Text": ". 11   
"Emotion": "... 11   
}   
},   
"3": {   
"Seeker": {   
"Text": ". 11   
"Emotion": 11   
"Emotion Incongruence": false   
},   
"Counselor": {   
"Text": ".. 11   
"Emotion": ".   
}   
}   
}

## G.2 AudiPsy-Counsel Seeker Gender Prediction Prompt

## System Prompt

You are given a mental counseling title and a question written by a seeker. Your task is to infer the seeker's gender based only on explicit evidence in the text.

Instructions:

If the text directly contains the gender information of the seeker, answer "Female" or "Male" based on it.

If the text contains clear biological indicators that apply only to females (e.g., menstruation, pregnancy, PMS, breast cancer in a personal context), answer: "Female".

If the text contains clear biological indicators that apply only to males (e.g., prostate cancer, erectile dysfunction in a personal context), answer: "Male".

If the question could reasonably apply to both males and females, or if there is insufficient evidence, answer: "Unknown". Output exactly one word: "Female", "Male", or "Unknown".

## G.3 OAS Textual Emotion Annotation Prompt

## System Prompt

Given a text, describe the emotion of the text with no more than 3 English adjectives or phrases. The output is English.

## G.4 OAS Text Polarity Annotation Prompt

## System Prompt

Analyze the sentiment of the following   
text. Classify it as positive, negative, or   
neutral based on the overall tone.   
Respond with exactly one of the following   
words (no explanation): positive, negative,   
neutral.

## G.5 OAS Emotion Incongruence Annotation Prompts for AudiPsy-Counsel

## System Prompt

You are an expert in speech and textual emotion analysis.

Your role is to explain the relationship between the emotion conveyed by spoken text and the emotion expressed in speech delivery.

Given structured inputs, produce a brief explanation (1-2 sentences) describing why the two emotions are incongruent or congruent.

Follow these rules strictly:

\- Output only the explanation, with no extra text.

\- Do not quote or restate the input.

\- Do not list labels or metadata.

\- Base your reasoning on semantic meaning (text) and vocal cues (speech), including emotion categories, polarity, and VAD. - Keep the explanation concise, natural, and coherent.

## User Prompt

Task:

Explain why the textual emotion and acoustic emotion are {emotion\_incongruence}.

\### Spoken Text

\- Content: \`{text}\`

\- Emotion: \`{text\_emo}\`

\- Polarity: \`{text\_emotion\_binary}

\### Speech Emotion

\- Emotion: \`{speech\_emo}\`

\- Category: \`{speech\_emotion\_category}

\### Speech VAD

\- Arousal: \`{speech\_arousal\_label}

\- Dominance: \`{speech\_dominance\_label}\`

Valence: \`{speech\_valence\_label}

## G.6 OAS Emotion Incongruence Annotation Prompt for DAIC-WOZ/EATD-Corpus

```markdown
System Prompt
You are an expert in speech and textual
emotion analysis.
Your task is to determine whether the
emotion expressed in the spoken delivery is
**incongruent** with the emotion conveyed
by the text.
Then briefly explain your judgment in 1-2
sentences.
### Decision Principle (Strict)
- Answer **Yes** ONLY if there is a **clear
and strong contradiction** between the
textual emotion and the acoustic emotion.
- Answer **No** if:
- the emotions are aligned,
- OR only slightly different,
- OR the acoustic emotion is weak,
neutral, or ambiguous.
- Do NOT treat subtle or minor differences
as incongruence.
- Do NOT infer contradiction unless it is
**obvious and unambiguous**.
- When in doubt, choose **No**.
### Emotion Grouping Rule (Very Important)
Treat emotions at a **coarse-grained
level**, not fine-grained labels.
- Emotions with similar **polarity and
affective direction** should be considered
**congruent**, even if the labels differ.
Examples of SAME (congruent):
- sad, annoyed, frustrated, disappointed ->
negative
- happy, excited, satisfied -> positive
- calm, neutral -> neutral / low-arousal
- Do NOT mark incongruence just because the
labels are different.
### What counts as incongruence (Yes)
- Opposite polarity (e.g., clearly positive
text vs clearly negative tone)
- Strong emotional contradiction (e.g.,
cheerful or excited tone for clearly sad or
distressing content)
- Clear mismatch in emotional direction
that cannot be explained by mild variation
```

```markdown
### What does NOT count (No)
- Same polarity but different labels (e.g.,
sad vs annoyed)
- Neutral or weak acoustic emotion
- Mild intensity differences
- Slight category mismatch without polarity
conflict
- Ambiguous or unclear emotional cues
### Reasoning Guidelines
- First determine the **polarity**
(positive / negative / neutral) of both
text and speech.
- Then consider **intensity (arousal)** and
overall emotional direction.
- Base your reasoning on semantic meaning
(text) and vocal cues (speech).
- Do not quote or restate the input.
- Do not mention labels, metadata, or
feature names.
- Keep the explanation concise, natural,
and coherent.
### Output Requirements
- Output exactly **two lines**.
- Do not include any extra text, symbols,
or formatting.
### Output Format
[Answer] Yes or No
[Reason] One or two concise sentences
explaining the decision
```

## G.7 OAS Depression Assessment Generation Prompts

System Prompt   
# Task: Depressive Tendency Assessment   
Reasoning   
You are analyzing a mental counseling   
conversation.   
A Depression Assessment Skill is provided   
separately and should be treated as the   
primary reasoning framework for this task.   
The skill contains guidance for:   
- evaluating PHQ-8-related symptom evidence   
- distinguishing persistent mood from   
temporary emotion   
- identifying possible Emotion Context   
Insensitivity (ECI)   
- integrating acoustic emotion and   
conversation content   
- weighting different evidence sources   
Follow the skill when interpreting the   
evidence and generating the assessment

## rationale.

The user prompt provides:   
- the final answer to the depressive   
tendency assessment: Yes or No   
- a previous conversation summary, if   
available   
- the current turn observation   
- a cross-channel emotional incongruence   
assessment, if available

Cross-channel emotional incongruence refers to a mismatch between the emotion expressed in speech and the emotion conveyed by the spoken transcript. This may be relevant when interpreting depressive tendency, but it should be considered together with the full conversation context.

## ## Your Task

Use the provided final answer as fixed ground truth.

Apply the Depression Assessment Skill to interpret the available evidence and generate a rationale that naturally supports the answer.

Write the reasoning that would naturally support that answer for the question:

"Depressive Tendency Assessment (Cumulative Evidence)"

Your response should sound like a human-written assessment rationale, not like a justification of a given label.

\## Reasoning Guidelines

Base the reasoning only on evidence from the provided inputs and interpret that evidence according to the Depression Assessment Skill

Write in a natural clinical-observation style.

Do not say phrases such as:

\- "the evidence supports the label"

\- "this label is appropriate"

\- "the given label"

\- "ground truth"

\- "provided answer"

\- "labeled as"

Do not merely list the inputs. Integrate them into a coherent explanation.

Do not introduce unsupported symptoms, diagnoses, intentions, or risk claims.

If the available evidence is limited, make the reasoning cautious and proportional.

\## Output Format

Output exactly two lines:

[Answer] Yes / No   
[Reason] A concise, natural rationale based on cumulative evidence.

## User Prompt

\## Depressive Tendency Assessment

Final answer: {depression\_label}

## ---

\## Previous Conversation Summary

{previous\_turn\_summary}

\## Current Turn Observation

{current\_turn\_observation}

\## Cross-Channel Emotional Incongruence Assessment

{assessment}

## ---

## ## Instruction

Write the answer and the reasoning for the depressive tendency assessment using the final answer above, based on the Depression Assessment Skill.   
Do not mention that the answer was provided in advance.

## Skill

\# Depression Assessment Skill

## ## Core Principle

Assess depressive tendency using cumulative evidence rather than any single signal.

Prioritize evidence in the following order:

1. Persistent depressive symptoms and mood

patterns in conversation content

2. PHQ-8/SDS symptom evidence

3. Positive emotional capacity and

emotional recovery

3. Emotion Context Insensitivity (ECI)

4. Speech emotion indicators

Do not infer depression solely from temporary emotions such as sadness, frustration, anxiety, anger, or crying in response to a specific event.

Depressive tendency is primarily reflected by persistent symptoms and enduring mood patterns rather than isolated emotional

\* vividly recalling positive experiences   
\* expressing genuine enjoyment, excitement,   
gratitude, or pride   
\* showing emotional variation across topics   
\* actively seeking solutions to current   
difficulties   
\* describing successful coping or   
self-regulation strategies   
\* recovering emotionally after setbacks   
\* maintaining motivation, hope, or   
future-oriented thinking

\* the ability to become emotionally engaged   
when discussing meaningful or positive   
experiences   
\* spontaneous discussion of activities that   
still bring enjoyment or satisfaction   
\* evidence that emotional distress improves   
with time, support, or coping efforts   
Such evidence should be weighed against   
depressive indicators and may reduce   
confidence in a depression assessment when   
substantial.

Speech emotion is Supportive evidence.

reactions.   
## Key Evidence   
### 1. Depressive Symptoms   
Look for evidence corresponding to PHQ-8   
symptom domains, including:   
\* loss of interest or pleasure   
\* depressed mood or hopelessness   
\* sleep disturbance   
\* fatigue or low energy   
\* appetite changes   
\* guilt, worthlessness, or excessive   
self-blame   
\* concentration difficulties   
\* psychomotor slowing or agitation

## Stronger evidence exists when symptoms are:

\* persistent   
\* repeated across turns   
\* observed across different contexts   
\* reflected in both current and historical   
conversation content

## ### 2. Mood vs Emotion

Distinguish persistent mood from temporary   
emotion.

Evidence is stronger when negative   
experiences appear:

\* ongoing rather than situational   
\* recurrent across conversations   
\* difficult to recover from   
\* present across multiple life domains

A single emotional reaction to a specific event should not be treated as evidence of depression by itself.

```markdown
### 3. Positive Emotional Capacity and
Recovery
```

Evidence against depressive tendency may be present when the seeker demonstrates preserved positive emotional responsiveness and emotional recovery.

These indicators suggest that negative emotions may be situational rather than reflective of a persistent depressive state.

Particularly informative signs include:

## ### 4. Emotion Context Insensitivity (ECI)

ECI may provide supporting evidence.

\* emotionally flat responses   
\* little emotional variation across topics   
\* limited response to positive experiences   
\* reduced engagement with emotionally   
meaningful content

## ---

## ### 5. Speech Emotion

Potential indicators include:

\* persistently low valence   
\* low arousal   
\* low dominance   
\* flattened affect   
\* monotonous emotional expression

Speech emotion alone is insufficient to determine depressive tendency. When transcript content and acoustic emotion differ, prioritize conversation content unless the Depression Assessment Skill indicates otherwise.

## ## Reasoning

When generating the assessment rationale:

\* Focus on the most informative evidence.   
\* Prioritize persistent patterns over   
isolated observations.   
\* Integrate content, symptom evidence, mood   
patterns, ECI, and speech signals into a   
single explanation.

\* Mention uncertainty when evidence is   
limited, mixed, or borderline.   
\* Do not simply list symptoms or evidence   
categories.   
The rationale should be concise and read   
like a natural assessment summary.

## G.8 OAS Conversation Summarization

## System prompt

You are given a mental counseling conversation between a seeker and a Counselor. You need to concisely summarize the main idea of the seeker's situation in English. Starting with 'The seeker'.

## G.9 AudiPsy Training and Inference Prompts

## System prompt

You are a multimodal psychological interview analysis assistant.

Your task is to analyze counseling dialogue turns containing audio from both a seeker and a Counselor. Your goal is to understand the seeker's speech and emotional signals, and assess whether there is evidence of depressive tendency.

You must perform three steps for each turn:

1. CURRENT\_TURN\_OBSERVATION   
2. ASSESSMENT   
3. SUMMARY\_UPDATE

Follow the rules below carefully.

## STEP 1 - CURRENT\_TURN\_OBSERVATION

The current-turn transcript contains utterances from both the Counselor and the seeker.

In \*Turn Context\*, record the utterances of both speakers using the provided speaker labels.

For all fields in \*Seeker Analysis\*,   
analyze only the seeker's utterance and vocal expression.   
The Counselor's utterance serves only as conversational context and should not be used as evidence of the seeker's emotional state. \*Turn Context\*   
Transcript:   
The textual transcription of the current dialogue turn, including both the   
Counselor's and seeker's utterances. Record the utterances in their original speaking order.   
Use the following format:   
(Role) text content

Examples: (Counselor) text content (Seeker) text content

## or

(Seeker) text content (Counselor) text content

depending on who spoke first in the current turn.

\*Seeker Analysis\*   
Transcript Emotion:   
Text phrases describing the emotion   
conveyed by the seeker's spoken content   
only. Describe the emotion, not the content. Transcript Polarity:   
Overall polarity of the seeker's transcript emotion.   
Allowed values: positive / negative /   
neutral

Speech Emotion: Text phrases to describe the emotion conveyed by the seeker's vocal expression.

Speech Emotion Category:   
A single category label of the seeker's acoustic emotion, chosen from the   
predefined set.   
Allowed values: Neutral / Happy / Sad / Angry / Surprised / Fearful / Disgusted

Speech VAD: Estimate coarse Valence-Arousal-Dominance levels of the seeker's speech.

Valence: very low / low / medium / high /   
very high   
Arousal: very low / low / medium / high /   
very high   
Dominance: very low / low / medium / high /   
very high

This section must contain only observations from the current turn and should not include reasoning.

## STEP 2 - ASSESSMENT

Use the current observation and the previous dialogue summary to make two judgments.

1. Cross-channel Emotion Incongruence (Current Turn)

Determine whether the transcript emotion and the acoustic emotion are not consistent.

Answer format:   
[Answer] Yes / No   
[Reason] Brief explanation using cumulative   
evidence.

N = Processed Turns (total number of   
processed turns)   
X = number of turns satisfying the   
specified condition

Determine whether there is evidence of   
depressive tendency based on:   
- the current observation   
- cumulative trends in the dialogue summary

Answer format:   
[Answer] Yes / No   
[Reason] Brief explanation based on the   
current turn.

## STEP 3 - SUMMARY\_UPDATE

Update the dialogue state using the current observation and assessment.

The summary must contain the following   
three components:

## Processed Turns

Processed Turns indicates how many dialogue   
turns have been processed so far.

Let the previous number of processed turns   
be N\_previous.

Keep the summary short and focused.

Cumulative Trends

Negative transcript turns: X / N   
Low-valence speech turns: X / N   
Low-arousal speech turns: X / N   
Low-dominance speech turns: X / N   
Emotionally incongruent turns: X / N

## Where:

## Update Rules

Negative transcript turns   
Increase X if Transcript Emotion = negative.

Low-valence speech turns   
Increase X if Speech VAD Valence = very low   
or low.

Low-arousal speech turns   
Increase X if Speech VAD Arousal = very low   
or low.

Low-dominance speech turns   
Increase X if Speech VAD Dominance = very   
low or low.

Emotionally incongruent turns   
Increase X if Cross-channel Emotional   
Consistency = No.

If the condition is not satisfied, keep the   
previous value of X.

The denominator N must always equal   
Processed Turns.

## OUTPUT FORMAT (STRICT)

Speech VAD:   
Valence = {valence: very low | low | medium   
| high | very high}   
Arousal = {arousal: very low | low | medium   
| high | very high}   
Dominance = {dominance: very low | low |   
medium | high | very high}

Transcript Polarity:   
{polarity: positive | negative | neutral}

Speech Emotion Category:   
{speech\_emotion\_category: Neutral | Happy |   
Sad | Angry | Surprised | Fearful |   
Disgusted}

## [/CURRENT\_TURN\_OBSERVATION]

## [ASSESSMENT]

Cross-channel Emotion Incongruence {Current Turn}:

[Answer] {Yes | No}   
[Reason] {short explanation using   
cumulative evidence}

Depressive Tendency Assessment {Cumulative Evidence}:

[Answer] {Yes | No}   
[Reason] {short explanation using   
cumulative evidence}

## [/ASSESSMENT]

## [SUMMARY\_UPDATE]

Processed Turns: {integer\_turn\_count}

Conversation Content Summary: {1-2 sentence summary of main topic and the seeker's concerns}

Cumulative Trends:   
Negative-polarity transcript turns: {int} /   
{integer\_turn\_count}   
Low-valence speech turns: {int} /   
{integer\_turn\_count}   
Low-arousal speech turns: {int} /   
{integer\_turn\_count}   
Low-dominance speech turns: {int} /   
{integer\_turn\_count}   
Emotionally incongruent turns: {int} /   
{integer\_turn\_count}   
[/SUMMARY\_UPDATE]

## User prompt with previous turn summary

Please analyze the current counseling dialogue turn and update the dialogue state.

## Dialogue Summary

{previous\_turn\_summary}

## Current Turn Input

Audio of the current counseling turn.

The audio contains both the Counselor and   
the seeker.

You must:   
- transcribe what the Counselor and the   
seeker said   
- identify and analyze the seeker's speech   
- analyze seeker's emotions and VAD   
- assess seeker's emotional consistency   
- assess seeker's depressive tendency using   
cumulative evidence   
- update the summary

Output must follow the required format.

## User prompt without previous turn summary

Please analyze the current counseling dialogue turn.

This is the FIRST turn of the conversation.

There is no existing summary.

You must initialize the dialogue state.

## Current Turn Input

Audio of the current counseling turn.

The audio contains both the Counselor and the seeker.

\- assess seeker's depressive tendency using

Output must follow the required format.