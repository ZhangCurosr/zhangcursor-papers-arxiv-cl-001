# EVA-Cap: Optimizing Audiovisual Video Captioning via Event-Centric Alignment

Jiapeng Shi<sup>1,\*</sup>, Qiuxia Hou<sup>2,\*</sup>, Jiaqi Leng<sup>1</sup>, Junke Wang<sup>1</sup>, Yanhao Zhang<sup>2</sup>, Haonan Lu<sup>2</sup>, Ziyi Ye<sup>1,†</sup>, Zuxuan Wu<sup>1,†</sup>

<sup>1</sup>Fudan University, <sup>2</sup>OPPO AI Center, OPPO Inc.

## Abstract

Recent advances in Omni Language Models (OLMs) provide a unified solution for audiovisual video captioning. Nevertheless, existing approaches predominantly treat captioning as an unstructured text generation process, neglecting the inherent semantic structure of videos and leading to suboptimal fine-grained audiovisual alignment. To bridge this gap, we propose EVA-Cap, a novel framework that decomposes captions into atomic audiovisual events and transforms unstructured text supervision into event-centric alignment. We train EVA-Cap via Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) on a self-curated dataset, EVA-Data, which is built with event-centric quality control. During the GRPO phase, the model is explicitly optimized against three event-centric alignment targets: global event completeness, intra-event attribution, and inter-event synchronization. Extensive experiments demonstrate that EVA-Cap significantly outperforms existing methods in event-centric evaluation (e.g., +9.0% average gain on ChronusAV), while achieving competitive performance on general audiovisual captioning benchmarks (e.g., +3.3% on WorldSense).

Correspondence: zyye@fudan.edu.cn, zxwu@fudan.edu.cn

Code: https://github.com/OPPO-Mente-Lab/EVA-Cap

## 1 Introduction

Video captioning [3, 13, 39, 48, 61] aims to extract rich semantic content from videos and generate coherent natural language descriptions. As a fundamental task in video understanding, it underpins various downstream applications, including video retrieval [34, 38, 47, 57], video reasoning [10, 11, 28, 37, 56], and video generation [2, 35, 36, 40, 54]. Recently, advances in Transformer-based end-to-end models have revolutionized the conventional modularized video caption pipelines. Especially, Omni Language Models (OLMs) [7, 20, 29, 46, 52, 53] have driven a paradigm shift from visual-only modeling to an omni-modal paradigm that integrates audio and visual signals, enabling more efective cross-modal reasoning and stronger generalization in real-world audiovisual scenarios.

Despite its promise, achieving factually accurate and comprehensive audiovisual captioning remains dificult due to the intrinsic complexity and dense interdependence between visual and auditory content. Existing methods treat video captioning as a text generation problem for OLMs based on caption-level supervision [4, 43]. This overlooks the structured semantics inherent in videos, meaning a generated caption may be linguistically correct but misaligned with the logical structure and event-level correspondences. While some recent research has incorporated additional supervision over specific aspects (e.g., dialogue [5, 30] or object attributes [6, 19]), such narrowly scoped supervision still struggles to achieve fine-grained audiovisual alignment. Consequently, a critical bottleneck stems from the lack of an efective caption representation that unifies video-grounded structural semantics with general language modeling.

![](images/4644c802b047ba68750939608708b990b9ec5efd4e676a5743fb8dd99a404487.jpg)  
Figure 1 Illustration of the event-centric representation of audiovisual captions, where aligned audiovisual events are composed into the predicted caption.

To address this issue, we propose framing video captioning as assembling discrete, atomic audiovisual event sequences through joint optimization of event content and their correspondences. As illustrated in Fig. 1, a video consists of multiple audio and visual events, where each event is represented as a tuple of type, character, and content. These events can be aligned temporally to form a coherent caption clip, linked via shared semantic content and explicit synchronous connectors (e.g., “as”, “at the same time”). Crucially, such an event-centric perspective bridges audiovisual events and captions by modeling both intra-event attribution (i.e., whether the attributions of a single event are correct?) and inter-event synchronization (i.e., what are the temporal order and synchronization of diferent events?). While such an event-centric perspective provides a structured semantic understanding of videos beyond mere language modeling, efectively integrating such insights into OLMs post-training remains an open question. This gap is largely due to (1) the scarcity of high-quality audiovisual caption data with event-level fidelity, and (2) the absence of dedicated event-centric frameworks for leveraging supervision with event-level heuristics.

To this end, we propose EVA-Cap, which builds upon Qwen2.5-Omni [45] and unifies event-centric alignment within the language modeling post-training phase. Specifically, we first construct EVA-Data, a 63K audiovisual video–caption dataset, via an automatic initialization–refinement pipeline followed by event-centric quality control. In this process, timestamped audiovisual events are extracted from raw videos and subjected to per-event verification to ensure high fidelity. Furthermore, we introduce a two-stage post-training strategy to facilitate event-centric alignment, which consists of Supervised Fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) [27] on the EVA-Data. During the GRPO stage, three event-centric rewards are designed to enforce global completeness (via event recall), intra-event attribution (via character matching), and inter-event synchronization (via cross-modal retrieval). Collectively, these post-training objectives enhance the fine-grained alignment of audiovisual events, enabling factually accurate, event-grounded audiovisual caption generation.

Extensive experiments demonstrate that EVA-Cap outperforms existing open-source models across 6 audiovisual benchmarks. For example, on general captioning benchmarks, EVA-Cap achieves a total error rate of 34.6% on video-SALMONN-2 testset [29] and surpasses existing methods by a clear margin of 3.3% on WorldSense [15]. Most importantly, on event-centric evaluation with ChronusAV [7] and AV-SpeakerBench [23], EVA-Cap exhibits substantial advantages with an accuracy of 55.4%, 56.2%, and 40.1% for Visual-to-Audio (V2A) inference, Audio-to-Visual (A2V) grounding, and speaker attribution, respectively. These results underscore the capability of EVA-Cap in generating highly faithful captions with precise temporal alignment and accurate event attribution.

## 2 Related Work

## 2.1 Audiovisual Video Captioning.

Recent advancements in Omni Language Models (OLMs) [20, 29, 46, 53] have significantly improved multimodal perception in complex real-world scenarios, thereby promoting the development of audiovisual captioning [12, 18, 21]. However, existing approaches remain constrained by the scarcity of high-quality audiovisual caption data, and largely rely on either coarse-grained supervision or focus on limited aspects of the task. For instance, AVoCaDO [4] constructs large-scale audiovisual video-caption pairs through temporal orchestration, but sufers from insuficient factual verification and coarse-grained temporal alignment. UGC-Captioner [43] relies on holistic caption-level supervision, whereas DiaDem [5] and D-ORCA [30] focus narrowly on specific aspects of captions, failing to capture fine-grained event-level correspondences. In contrast, we introduce an event-centric perspective for audiovisual captioning. We construct an event-enhanced dataset with event-centric quality control, and propose an event-centric training strategy that explicitly models event-level alignment, enabling accurate and event-grounded audiovisual caption generation.

## 2.2 Reinforcement Learning for Visual Content Captioning.

Reinforcement learning [27, 55, 60] plays an increasingly important role in caption optimization, spanning from visual-only to omni-modal scenarios. CapRL [44] and RubiCap [16] design verifiable rubrics for dense image captioning. OwlCap [61] and VideoCap-R1 [22] improve video captioning by matching key information extracted from visual content. In the audiovisual domain, D-ORCA [30] introduces dialogue-oriented rewards focusing on speaker and ASR consistency, while TimeChat-Captioner [51] optimizes timestamp alignment under structured caption formats. However, these approaches typically focus on limited aspects and struggle to achieve fine-grained audiovisual alignment. In this paper, we develop three event-centric rewards that jointly enforce fine-grained supervision within and across modalities. This approach provides more precise event alignment than prior methods, which either focus on modality-specific objectives [5, 30] or rely on coarse audiovisual synchronization [4, 51].

## 3 Method

This section introduces the construction of EVA-Cap, which is built upon Qwen-2.5-Omni [45] and optimized for audiovisual video captioning via event-centric alignment. First, we address the scarcity of high-quality data by curating EVA-Data, a 63K-scale audiovisual captioning dataset. The dataset is generated using an automated pipeline with rigorous event-centric quality control (Sec. 3.1). Next, we introduce a two-stage training strategy comprising SFT and GRPO to improve EVA-Cap with event-centric alignment. The training process integrates a suite of event-centric rewards to efectively exploit event-level supervision (Sec. 3.2).

## 3.1 EVA-Data Construction

## 3.1.1 Annotation Pipeline

We develop an automatic annotation pipeline to construct audiovisual video–caption pairs with event-level fidelity. As shown in Fig. 2, the pipeline comprises three main stages:

Stage 1: Initialization. We first utilize the open-source audiovisual captioner AVoCaDO [4], which excels in holistic perception, to generate initial audiovisual descriptions for raw videos. This provides a coherent narrative of each video and captures key events and details, covering scene backgrounds, interactive actions, and audio transcriptions. We then filter out samples with repetitive descriptions or abnormal text lengths.

Stage 2: Refinement. We then feed both raw videos and initial descriptions to the advanced omni-model Gemini-2.5-Pro [8] for content refinement and temporal alignment. Guided by raw audiovisual signals, we instruct the model to strengthen the temporal synchronization of audiovisual events and revise event character attribution, while preserving the original natural language style. This yields refined descriptions with precise audiovisual event alignment, achieving fine-grained annotation.

![](images/0b7c761fcf4d7af39dd0359ca09e0e6bfe9bf3763c69fe83d5e99041c21b2a1b.jpg)  
Figure 2 Illustration of the designed data annotation pipeline, comprising three stages: initialization, refinement and event-centric quality control. Raw videos are transformed into audiovisual captions with high-fidelity audiovisual synchronization and event attributions.

![](images/1b97667f23268c9feb2994bd1316dc9c34075b5316a8d774bcd7551c4eea8962.jpg)

![](images/fe64d26ae033bc483924f0bc763df757c56ee95ef1df94fa3dda91ff3f2a6673.jpg)  
(b) Video Duration (Seconds)

![](images/52f99fd8b2bd423a401bc35ef291126d3651148913cc7b6d6ee637949a71e14b.jpg)  
(c) Event Count (Audio/Visual)  
Figure 3 Statistics of EVA-Data. (a) Video category distribution and (b) Video duration distribution demonstrate the diversity of this dataset, while (c) Event count distribution shows comparable counts of audio and visual events, indicating no modality bias in annotated captions.

Stage 3: Event-Centric Quality Control. To ensure the quality of the curated data, we introduce an aligned event hit mechanism for the automatic validation of refined descriptions. Specifically, we employ Gemini-2.5- Pro to extract timestamp-grounded audiovisual event tuples directly from raw videos. We then input all event tuples together with refined descriptions to GPT-4.1 [1] to verify whether each event tuple is hit in the caption. An event hit requires both audio and visual events to be entailed with consistent temporal synchronization. This process jointly assesses the accuracy of audiovisual synchronization and event attribution in captions. We discard samples with an event hit ratio below a threshold of 0.8. In this way, we build a large-scale audiovisual caption dataset, EVA-Data, which provides high-quality event-level supervision for event-centric post-training.

## 3.1.2 Data Statistics

To guarantee both diversity and quality, we collect 100K audiovisual videos (each under 200 seconds in duration) from LLaVA-Video [58], FineVideo [9], and Shot2Story [14]. Through the annotation pipeline, EVA-Data ultimately yields 63K audiovisual video-caption pairs with event-level fidelity. Fig. 3 (b) illustrates the duration distribution of these videos. On average, each video lasts 73.7 seconds, while the corresponding captions average 504.3 words. We further randomly sample 20K videos from EVA-Data and analyze the distribution of audio and visual events in the corresponding captions, as shown in Fig. 3 (c). The results demonstrate that the annotated videos contain rich audiovisual information and the captions are free of modality bias, thus providing efective supervision for event-centric alignment.

![](images/35a418512e0e889e46366b93dca1436a0a72351bc32ea7e32b4acf191e4d363f.jpg)  
Figure 4 Illustration of the event-centric alignment with GRPO for EVA-Cap. We introduce event-centric rewards during GRPO training, focusing on three key dimensions: Global Completeness (event recall), Intra-Event Attribution (character matching), and Inter-Event Synchronization (cross-modal retrieval between audio and visual events)

## 3.2 Post-Training of EVA-Cap

Leveraging this dataset, we propose EVA-Cap, an audiovisual captioner optimized via event-centric alignment. We first present our two-stage training pipeline in Sec. 3.2.1, and then introduce the event-centric reward functions dedicated to fine-grained alignment in Sec. 3.2.2.

## 3.2.1 Training Pipeline

We build EVA-Cap upon the Qwen2.5-Omni Thinker module, which jointly encodes audio and video into a unified, temporally interleaved token sequence to facilitate synchronous audiovisual understanding. We then adopt a two-stage post-training pipeline consisting of supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO).

In the first phase, we perform SFT on our constructed 63K EVA-Data to empower the model to generate detailed captions that integrate both audio and visual modalities. In the second phase, we further optimize the SFT model with GRPO on 3.2K rollout videos from EVA-Data (details of the rollout process are provided in Appendix A.2.2). To enhance fine-grained alignment in audiovisual captions, we design event-centric rewards covering three dimensions: event completeness, character attribution, and audiovisual synchronization. The detailed SFT and GRPO objectives are provided in Appendix A.1, and the reward design is elaborated in Sec. 3.2.2.

## 3.2.2 Event-Centric Reward Modeling

Typically, given an audiovisual video $v = ( v ^ { \mathrm { v i s } } , v ^ { \mathrm { a u d } } )$ and a caption y, conventional preference optimization assigns a holistic reward $\mathcal { R } ( y , v )$ . However, this paradigm treats the caption as an atomic semantic unit and

overlooks its underlying event structure. To address this issue, we represent a caption as a sequence of modality-grounded events:

$$
E = ( e _ { 1 } , e _ { 2 } , . . . , e _ { K } ) , \quad e _ { k } = ( m _ { k } , \tau _ { k } , c _ { k } , d _ { k } ) , \quad m _ { k } \in \{ \mathrm { v i s } , \mathrm { a u d } \} ,\tag{1}
$$

where $m _ { k }$ denotes the modality, $\tau _ { k }$ denotes the event $\mathrm { t y p e } , c _ { k }$ denotes the character, and $d _ { k }$ denotes the event content.

Based on this event-centric representation, we further define the reference structures to support the eventcentric reward calculation. For a reference caption $y ^ { \star } .$ , we denote its reference event set as $\begin{array} { r } { \bar { \mathcal { E } } ^ { \star } = \mathcal { E } _ { \mathrm { a u d } } ^ { \star } \cup \mathcal { E } _ { \mathrm { v i s } } ^ { \star } , } \end{array}$ where $\mathcal { E } _ { m } ^ { \star } = \{ e _ { i } ^ { m } \} _ { i = 1 } ^ { N _ { m } }$ is the set of modality-specific reference events for $m \in \{ \mathrm { a u d } , \mathrm { v i s } \}$ . This event set is used to evaluate event coverage and character attribution. In addition, we denote by $\mathcal { Q } ^ { \star } = \{ q _ { i } ^ { \star } \} _ { i = 1 } ^ { M }$ the set of audio-to-visual synchronization queries, where each query consists of an audio event, a set of candidate visual events, and the reference synchronized visual indices. This query set is used to evaluate cross-modal temporal synchronization.

Based on the above event formulation and definitions, we devise three event-centric reward components to achieve fine-grained alignment (as shown in Fig. 4):

Global Event Completeness. Global event completeness measures whether the generated caption covers the reference audio and visual events that should be described in the video. Given a reference event set $\mathcal { E } ^ { \star } = \mathcal { E } _ { \mathrm { a u d } } ^ { \star } \cup \mathcal { E } _ { \mathrm { v i s } } ^ { \star } ,$ where $\mathcal { E } _ { m } ^ { \star } = \{ e _ { i } ^ { m } \} _ { i = 1 } ^ { N _ { m } }$ denotes the modality-specific events for $m \in \{ \mathrm { a u d } , \mathrm { v i s } \}$ , we define an LLM-evaluated coverage indicator:

$$
\begin{array} { r } { \mathbb { I } _ { \mathrm { c o v } } ^ { m } ( e _ { i } ^ { m } , y ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } \mathrm { C o v e r s } ( y , e _ { i } ^ { m } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. m \in \{ \mathrm { a u d } , \mathrm { v i s } \} . } \end{array}\tag{2}
$$

Here, $\mathrm { C o v e r s } ( y , e _ { i } ^ { m } )$ indicates that the event content of $e _ { i } ^ { m }$ is semantically supported by the generated caption $y .$ The global event completeness reward is defined as event recall over all reference audio and visual events:

$$
\mathcal { R } _ { \mathrm { c o m p } } ( y , \mathcal { E } ^ { \star } ) = \frac { \sum _ { m \in \{ \mathrm { a u d } , \mathrm { v i s } \} } \sum _ { i = 1 } ^ { N _ { m } } \mathbb { I } _ { \mathrm { c o v } } ^ { m } ( e _ { i } ^ { m } , y ) } { \sum _ { m \in \{ \mathrm { a u d } , \mathrm { v i s } \} } N _ { m } } .\tag{3}
$$

Intra-Event Attribution. Event completeness alone does not guarantee that a caption assigns events to the correct entities. For example, a caption may mention the correct speech content but attribute it to the wrong speaker. To address this issue, we introduce an intra-event attribution reward that evaluates whether each character-bearing reference event is both covered and associated with the correct character. Given the reference event set $\bar { \mathcal { E } } ^ { \star } = \mathcal { E } _ { \mathrm { a u d } } ^ { \star } \cup \mathcal { E } _ { \mathrm { v i s } } ^ { \star } ,$ where $\mathcal { E } _ { m } ^ { \star } = \{ e _ { i } ^ { m } \} _ { i = 1 } ^ { N _ { m } }$ denotes the modality-specific events for $m \in \{ \mathrm { a u d } , \mathrm { v i s } \}$ , we define an LLM-evaluated attribution indicator for each valid event with an explicit character:

$$
\mathbb { I } _ { \mathrm { a t t r } } ^ { m } ( e _ { i } ^ { m } , y , y ^ { \star } ) = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \mathrm { M a t c h e s } ( y , y ^ { \star } , e _ { i } ^ { m } ) , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right. m \in \{ { \mathrm { a u d } } , { \mathrm { v i s } } \} .\tag{4}
$$

Here, Matches $( y , y ^ { \star } , e _ { i } ^ { m } )$ indicates that the generated caption $y$ and the reference caption $y ^ { \star }$ attribute the event content $d _ { i } ^ { m }$ to the same underlying character. The attribution reward is computed as

$$
{ \mathcal R } _ { \mathrm { a t t r } } ( y , y ^ { \star } , \mathcal { E } ^ { \star } ) = \frac { \sum _ { m \in \{ \mathrm { a u d } , \mathrm { v i s } \} } \sum _ { i = 1 } ^ { N _ { m } } \delta _ { i , \mathrm { c h a r } } ^ { m } \mathbb { I } _ { \mathrm { c o v } } ^ { m } ( e _ { i } ^ { m } , y ) \mathbb { I } _ { \mathrm { a t t r } } ^ { m } ( e _ { i } ^ { m } , y , y ^ { \star } ) } { \sum _ { m \in \{ \mathrm { a u d } , \mathrm { v i s } \} } \sum _ { i = 1 } ^ { N _ { m } } \delta _ { i , \mathrm { c h a r } } ^ { m } } ,\tag{5}
$$

where $\delta _ { i , \mathrm { c h a r } } ^ { m } = \mathbb { I } [ c _ { i } ^ { m } \neq \emptyset ]$ indicates whether the event has an explicit character. Character-free events are not counted in this reward.

Inter-Event Synchronization. To ensure accurate temporal correspondence between audio and visual events, we model cross-modal alignment explicitly by introducing an inter-event synchronization reward, which is formulated as audio-to-visual synchronization retrieval. Given a set of audio synchronization queries

$\mathcal { Q } ^ { \star } = \{ q _ { i } ^ { \star } \} _ { i = 1 } ^ { M }$ , each query consists of an audio event $e _ { i } ^ { \mathrm { a u d } }$ , a candidate visual event set $\mathcal { C } _ { i } ^ { \mathrm { v i s } } = \{ v _ { i , j } \} _ { j = 1 } ^ { L _ { i } }$ , and the reference synchronized visual index set $\mathcal { G } _ { i } \subseteq \{ 1 , \ldots , L _ { i } \}$ . For each query, an LLM evaluator selects the visual candidates that are temporally synchronized with the audio event according to the generated caption y:

$$
\hat { \mathcal { G } } _ { i } = \mathrm { R e t r i e v e S y n c } \left( y , e _ { i } ^ { \mathrm { a u d } } , \mathcal { C } _ { i } ^ { \mathrm { v i s } } \right) ,\tag{6}
$$

where $\hat { \mathcal { G } } _ { i } \subseteq \{ 1 , \ldots , L _ { i } \}$ denotes the predicted synchronized visual index set. If no synchronized visual event is found, RetrieveSync returns an empty set. We then compute the set-level F1 score:

$$
\mathrm { P r e c } _ { i } = \frac { \vert \hat { \mathcal { G } } _ { i } \cap \mathcal { G } _ { i } \vert } { \vert \hat { \mathcal { G } } _ { i } \vert } , \quad \mathrm { R e c } _ { i } = \frac { \vert \hat { \mathcal { G } } _ { i } \cap \mathcal { G } _ { i } \vert } { \vert \mathcal { G } _ { i } \vert } , \quad F _ { i } = \frac { 2 \mathrm { P r e c } _ { i } \mathrm { R e c } _ { i } } { \mathrm { P r e c } _ { i } + \mathrm { R e c } _ { i } } .\tag{7}
$$

When both $\hat { \mathcal { G } } _ { i }$ and $\mathcal { G } _ { i }$ are empty, we set $F _ { i } = 1$ , when only one of them is empty, or their intersection is empty, we set $F _ { i } = 0$ . The synchronization reward is then defined as

$$
\mathcal { R } _ { \mathrm { s y n c } } ( y , \mathcal { Q } ^ { \star } ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } F _ { i } .\tag{8}
$$

We therefore formulate our overall reward $\mathcal { R } _ { \mathrm { E V A } }$ with three complementary event-centric terms:

$$
\mathcal { R } _ { \mathrm { E V A } } ( y , y ^ { \star } ) = \lambda _ { \mathrm { c o m p } } \mathcal { R } _ { \mathrm { c o m p } } ( y , \mathcal { E } ^ { \star } ) + \lambda _ { \mathrm { a t t r } } \mathcal { R } _ { \mathrm { a t t r } } ( y , y ^ { \star } , \mathcal { E } ^ { \star } ) + \lambda _ { \mathrm { s y n c } } \mathcal { R } _ { \mathrm { s y n c } } ( y , \mathcal { Q } ^ { \star } ) .\tag{9}
$$

Here, $\lambda _ { \mathrm { c o m p } } , \lambda _ { \mathrm { a t t r } } ,$ and $\lambda _ { \mathrm { s y n c } }$ are balancing coeficients. For all reward terms, if the corresponding valid evaluation set is empty, i.e., $\begin{array} { r } { \sum _ { m } N _ { m } = 0 , \sum _ { m } \sum _ { i } \delta _ { i , \mathrm { c h a r } } ^ { m } = 0 , } \end{array}$ , or $M = 0$ , we set the corresponding reward value to 0.

During training, we adopt this reward with two rounds of GRPO to achieve progressive alignment. Specifically, in the first round, we set $\lambda _ { \mathrm { c o m p } }$ and $\lambda _ { \mathrm { s y n c } }$ to 1 while setting $\lambda _ { \mathrm { a t t r } }$ to 0. In the second round, we set all three coeficients to 1. The process transitions from global coarse-grained event alignment and audiovisual synchronization to fine-grained character-level attribution.

## 4 Experiments

## 4.1 Experimental Setup

Implementation Details: We adopt Qwen2.5-Omni-7B [45] as our backbone model. During training, the vision and audio encoders are kept frozen, while only the adapters and LLM parameters are updated. Videos are sampled at 2 FPS, with a maximum per-frame pixels of $5 1 2 \times 2 8 \times 2 8$ and total pixels of $3 0 \times 5 1 2 \times 2 8 \times 2 8$ And the maximum generation length is set to 2048. For SFT, we train the model for two epochs with a batch size of 128 and a learning rate of $\mathrm { \bar { 1 } } \times \mathrm { 1 0 ^ { - 5 } }$ . For GRPO, we conduct training for two epochs with a batch size of 16, a learning rate of $2 \times 1 0 ^ { - 6 }$ , a KL penalty of 0.04, and a group size of $^ { 8 , }$ following a progressive reward weighting strategy. We employ Qwen3.5-27B [33] as the reward judge. All training is conducted on 32 NVIDIA H20 GPUs.

Benchmarks: For general audiovisual captioning, we evaluate EVA-Cap under two settings: direct quality assessment (video-SALMONN-2 testset [29] and UGC-VideoCap [43]) and QA-cascaded evaluation (Daily-Omni [62] and WorldSense [15]). The former directly evaluates the overall quality of generated captions through dedicated scoring mechanisms, while for the latter, we first generate descriptive captions with our model and then adopt Gemini-2.5-Pro [8] to answer QA questions solely based on the produced captions.

To assess the capabilities in event-centric alignment, we further conduct evaluations on subsets of ChronusAV [7] and AV-SpeakerBench [23]. For ChronusAV, we select 1,066 videos under 200 seconds to assess audiovisual synchronization via video-to-audio (V2A) and audio-to-video (A2V) grounding. Evaluation follows a twostage protocol: Gemini-2.5-Pro first generates answers for the V2A and A2V grounding queries conditioned

Table 1 Performance comparison on general audiovisual video captioning benchmarks. Following previous works [4, 5], we adopt GPT-4.1 [1] as the judge model for the video-SALMONN-2 testset, and GPT-4o [17] for UGC-VideoCap evaluation.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="3">video-SALMONN-2 testset</td><td colspan="4">UGC-VideoCap</td></tr><tr><td>Miss.↓</td><td>Hall.↓</td><td>Total↓</td><td>Audio↑</td><td>Visual↑</td><td>Detail↑</td><td>Avg.↑</td></tr><tr><td>Gemini-2.5-Pro [8]</td><td></td><td>18.1</td><td>13.3</td><td>31.3</td><td>69.5</td><td>74.7</td><td>73.7</td><td>72.6</td></tr><tr><td>Gemini-2.5-Flash [8]</td><td>-</td><td>19.3</td><td>13.9</td><td>33.3</td><td>69.1</td><td>75.8</td><td>74.0</td><td>73.0</td></tr><tr><td>HumanOmniV2 [49]</td><td>7B</td><td>49.2</td><td>12.3</td><td>61.6</td><td>45.6</td><td>66.3</td><td>59.5</td><td>57.1</td></tr><tr><td>ARC-Hunyuan-Video [12]</td><td>7B</td><td>45.7</td><td>12.5</td><td>58.2</td><td>52.7</td><td>56.0</td><td>55.8</td><td>54.8</td></tr><tr><td>Qwen2.5-Omni [45]</td><td>7B</td><td>41.7</td><td>15.4</td><td>57.1</td><td>46.9</td><td>66.1</td><td>60.0</td><td>57.7</td></tr><tr><td>MiniCPM-o-2.6 [52]</td><td>8B</td><td>42.2</td><td>14.3</td><td>56.5</td><td>38.6</td><td>68.5</td><td>57.7</td><td>54.9</td></tr><tr><td>UGC-VideoCaptioner [43]</td><td>3B</td><td>31.6</td><td>17.0</td><td>48.6</td><td>61.4</td><td>58.4</td><td>57.5</td><td>59.1</td></tr><tr><td>Qwen3-Omni-Instruct [46]</td><td>30B-A3B</td><td>32.0</td><td>13.6</td><td>45.6</td><td>67.5</td><td>74.8</td><td>72.3</td><td>71.5</td></tr><tr><td>Qwen3-Omni-Captioner [46]</td><td>30B-A3B</td><td>31.0</td><td>16.6</td><td>47.6</td><td>69.0</td><td>75.5</td><td>72.3</td><td>72.3</td></tr><tr><td>video-SALMONN-2 [29]</td><td>7B</td><td>21.2</td><td>17.6</td><td>38.8</td><td>61.8</td><td>71.4</td><td>68.5</td><td>67.2</td></tr><tr><td>AVoCaDO [4]</td><td>7B</td><td>21.1</td><td>16.2</td><td>37.3</td><td>73.0</td><td>74.6</td><td>71.8</td><td>73.2</td></tr><tr><td>DiaDem [5]</td><td>7B</td><td>21.1</td><td>15.5</td><td>36.5</td><td>75.4</td><td>76.8</td><td>74.6</td><td>75.6</td></tr><tr><td>EVA-Cap</td><td>7B</td><td>17.6</td><td>17.0</td><td>34.6</td><td>76.9</td><td>79.0</td><td>74.7</td><td>76.9</td></tr></table>

on the model-generated captions, and Seed-1.6 [31] subsequently scores the outputs to compute VDCscore [3] and accuracy. For AV-SpeakerBench, we adopt 422 QA samples from its Speaker Recognition subset to assess audio event attribution, using the same QA-cascaded evaluation paradigm with Gemini-2.5-Pro.

## 4.2 Main Results

Performance on direct quality assessment. We evaluate EVA-Cap on two representative general audiovisual captioning benchmarks. Those benchmarks assess the overall captioning performance from multiple dimensions, including video content omission, hallucination, and multiaspect scoring. The comparison with existing audiovisual models is presented in Tab. 1. EVA-Cap achieves state-of-the-art or competitive results across various metrics, e.g., a total error rate of 34.6% on the video-SALMONN-2 testset [29] and an average score of 76.9 on UGC-VideoCap [43], surpassing both unified OLMs, e.g., Qwen3- Omni-Instruct [46] and task-specific captioners, e.g., DiaDem [5]. On the video-SALMONN-2 test set, EVA-Cap demonstrates the lowest miss rate of 17.6%. While some baselines achieve lower hallucination rates, they often do so at the cost of substantial event omissions, e.g., HumanOmniV2 maintains a 12.3% hallucination rate at the cost of a 49.2% miss rate. This indi-

Table 2 Performance comparison on QA-cascaded benchmarks, with Gemini-2.5-Pro serving as the judge.
<table><tr><td>Model</td><td>Size</td><td>Daily- Omni</td><td>World- Sense</td></tr><tr><td>Gemini-2.5-Pro [8]</td><td>-</td><td>60.2</td><td>33.8</td></tr><tr><td>Gemini-2.5-Flash [8]</td><td>-</td><td>55.3</td><td>31.0</td></tr><tr><td>HumanOmniV2 [49]</td><td>7B</td><td>8.2</td><td>6.6</td></tr><tr><td>ARC-Hunyuan-Video [12]</td><td>7B</td><td>8.6</td><td>8.7</td></tr><tr><td>MiniCPM-o-2.6 [52]</td><td>8B</td><td>9.8</td><td>7.2</td></tr><tr><td>Qwen2.5-Omni [45]</td><td>7B</td><td>12.4</td><td>8.6</td></tr><tr><td>UGC-VideoCaptioner [43]</td><td>3B</td><td>17.0</td><td>11.2</td></tr><tr><td>Qwen3-Omni-Instruct [46]</td><td>30B-A3B</td><td>17.5</td><td>12.7</td></tr><tr><td>Qwen3-Omni-Captioner [46] 30B-A3B</td><td></td><td>27.2</td><td>14.1</td></tr><tr><td>video-SALMONN-2 [29]</td><td>7B</td><td>29.9</td><td>18.2</td></tr><tr><td>AVoCaDO [4]</td><td>7B</td><td>50.1</td><td>25.7</td></tr><tr><td>TimeChat-Captioner [51]</td><td>7B</td><td>52.8</td><td>22.6</td></tr><tr><td>EVA-Cap</td><td>7B</td><td>56.1</td><td>29.0</td></tr></table>

cates OLMs may omit many valid yet less salient events without event-centric training. This demonstrates the strong general audiovisual captioning capability of our method.

Performance on QA-cascaded evaluation. To further evaluate the information completeness and accuracy of the generated captions, we conduct reasoning-based QA evaluation under the cascaded paradigm described above. Specifically, we prompt Gemini-2.5-Pro [8] to answer questions based on the model-generated captions, and the resulting QA accuracy reflects the overall quality of audiovisual captions. As shown in Tab. 2, EVA-Cap outperforms existing open-source models by a clear margin, with gains of +3.3% on Daily-Omni [62] and +3.3% on WorldSense [15].

Performance on event-centric alignment. Finally, we evaluate eventcentric alignment on ChronusAV [7] and AV-SpeakerBench [23]. For ChronusAV, we adopt METEOR to measure textual similarity, and further compute accuracy and VD-Cscore following the protocol of the VDC benchmark [3]. For AV-SpeakerBench, we take the accuracy of multiple-choice questions as the evaluation metric. For comparison, we include three advanced open-

Table 3 Performance comparison on event-centric evaluation.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td>V2A</td><td>A2V</td><td rowspan="2">AV-Speaker</td></tr><tr><td>M Acc Score</td><td>M Acc Score</td></tr><tr><td>Gemini-2.5-Pro [8]</td><td>–</td><td>2.461.2 2.4</td><td>1.2 61.0 2.4</td><td>Acc 49.1</td></tr><tr><td>TimeChat [51]</td><td>7B</td><td>1.2 231.6 1.2</td><td>0.7 30.1 1.2</td><td>23.7</td></tr><tr><td>AVoCaDO [4]</td><td>7B</td><td>1.8 40.9 1.6</td><td>1.1 44.9 1.8</td><td>26.8</td></tr><tr><td>DiaDem [5]</td><td>7B</td><td>1.8 45.9 1.7</td><td>1.1 47.7 1.9</td><td>32.7</td></tr><tr><td>EVA-Cap</td><td>7B</td><td>2.4 55.4 2.2</td><td>1.2 56.2 2.2</td><td>40.1</td></tr></table>

source models that achieve competitive results on general audiovisual captioning benchmarks, i.e., TimeChat-Captioner [51], AVoCaDO [4], and DiaDem [5].

As shown in Tab. 3, EVA-Cap significantly outperforms the aforementioned models across all metrics. This not only validates the capability of our model to attribute events to the corresponding characters, but also highlights its strength in achieving fine-grained, precise audiovisual event synchronization.

## 4.3 Ablation Studies

In Tab. 4, we conduct ablation experiments to study the efects of our post-training pipeline and reward suite. First, the baseline Qwen2.5-Omni sufers from inherent modality bias [26], where audio content is frequently omitted in its generated captions, resulting in poor performance across all tasks. After SFT on our constructed EVA-Data, the model achieves considerable improvements in both general audiovisual captioning and event alignment, e.g., the V2A and A2V accuracy increases from 8.4% to 44.8% and from 11.9% to 46.1%, respectively.

Table 4 Ablation study on the post-training stages and event-centric reward mechanisms of EVA-Cap. The two-stage GRPO training setting we adopted is marked in light gray . All GRPO stage1 settings are derived from the SFT model, whereas GRPO stage 2 settings build upon the marked stage1.
<table><tr><td rowspan="2">Setting</td><td colspan="2">Reward</td><td rowspan="2"></td><td colspan="2">V2A</td><td colspan="4">A2V</td><td rowspan="2">AV-Speaker</td><td rowspan="2">Daily-Omni Acc</td></tr><tr><td>Rcomp</td><td> $\mathcal { R } _ { \mathrm { s y n c } }$ </td><td> ${ \mathcal { R } } _ { \mathrm { a t t r } }$ </td><td>M</td><td>Acc Score</td><td>M</td><td>Acc Score|</td><td>Acc</td></tr><tr><td>Baseline</td><td>-</td><td></td><td>–</td><td>0.1</td><td>8.4</td><td>0.2</td><td>0.3</td><td>11.9</td><td>0.5</td><td>3.1</td><td>12.4</td></tr><tr><td>SFT</td><td>一</td><td>I</td><td>1</td><td>2.0</td><td>44.8</td><td>1.7</td><td>1.0</td><td>46.1</td><td>1.8</td><td>31.8</td><td>50.1</td></tr><tr><td>SFT → GRPO stage1</td><td>√</td><td>-</td><td>-</td><td>2.2</td><td>53.3</td><td>2.0</td><td>1.2</td><td>52.9</td><td>2.0</td><td>33.9</td><td>53.8</td></tr><tr><td>SFT → GRPO stage1</td><td>√</td><td>√</td><td>-</td><td>2.3</td><td>53.8</td><td>2.1</td><td>1.2</td><td>55.0</td><td>2.1</td><td>37.0</td><td>55.0</td></tr><tr><td>SFT → GRPO stage1</td><td>√</td><td>√</td><td>√</td><td>2.1</td><td>47.9</td><td>1.8</td><td>1.1</td><td>50.8</td><td>1.9</td><td>35.6</td><td>52.9</td></tr><tr><td>SFT → GRPO stage2</td><td>√</td><td>√</td><td>-</td><td>2.2</td><td>56.5</td><td>2.2</td><td>1.3</td><td>55.5</td><td>2.2</td><td>37.0</td><td>55.1</td></tr><tr><td>SFT → GRPO stage2</td><td>√</td><td>√</td><td>√</td><td>2.4</td><td>55.4</td><td>2.2</td><td>1.2</td><td>56.2</td><td>2.2</td><td>40.1</td><td>56.1</td></tr></table>

We further investigate the efects of three event-centric rewards during the GRPO phase. The global event completeness reward $\mathcal { R } _ { \mathrm { c o m p } }$ penalizes event omission, thereby boosting model performance across all dimensions (+8.5%, +2.1%, and +3.7% accuracy on V2A, AV-Speaker, and Daily-Omni). Furthermore, integrating the inter-event synchronization reward $\mathcal { R } _ { \mathrm { s y n c } }$ and intra-event attribution reward ${ \mathcal { R } } _ { \mathrm { a t t r } }$ yields consistent gains on corresponding tasks (+2.1% on A2V and +3.1% on AV-Speaker in terms of accuracy), while also improving the overall captioning performance steadily. Collectively, these three rewards enable precise alignment in both event content and event correspondence, facilitating factually accurate, event-grounded audiovisual video captioning.

![](images/3346e9b05591eb232aff4f7c2a8e2e9da91b90a18d15cc2df9a03ffea5771ade.jpg)  
Figure 5 Case analysis of EVA-Cap and AVoCaDO, with features relevant to audiovisual synchronization in bold, and accurate event attribution underlined. Incorrect event-centric alignments are marked in red, and correct ones in green.

To further dissect our two-stage GRPO design, we conduct additional ablations. Introducing attribution reward ${ \mathcal { R } } _ { \mathrm { a t t r } }$ as early as the first round not only compromises general performance but also degrades eventcentric alignment. We hypothesize that injecting both $\mathcal { R } _ { \mathrm { s y n c } }$ and ${ \mathcal { R } } _ { \mathrm { a t t r } }$ prematurely leads to optimization conflicts between inter-event relations and intra-event accuracy. Conversely, a progressive strategy facilitates a gradual alignment process, from global coarse-grained structures to fine-grained character-level details. Moreover, comparisons against a baseline that sustains only $\mathcal { R } _ { \mathrm { c o m p } }$ and $\mathcal { R } _ { \mathrm { s y n c } }$ in the second round demonstrate that the sustained improvements stem from efective reward design rather than mere data scaling.

## 4.4 Visualizations

We present a caption generated by EVA-Cap in Fig. 5 for comparison with AVoCaDO [4], highlighting the superior event alignment capabilities (i.e., audiovisual synchronization and event attribution) of our model. For instance, the audio ASR event "Puppy: I know. You saw me and took pity on me" occurs when characters converse on the sidewalk. AVoCaDO incorrectly synchronizes it with the adjacent scene in the middle of a crosswalk and misattributes the event to the boy. In contrast, our model precisely aligns these audiovisual events and accurately identifies the corresponding speaker. This example demonstrates that our model can produce high-fidelity captions in complex audiovisual scenarios involving multi-view transitions and multi-object interactions, and further underscores the eficacy of the event-centric alignment mechanism for long-form audiovisual captioning.

## 5 Conclusion

This paper presents EVA-Cap, an audiovisual video captioning model with a training framework that transforms holistic caption optimization into event-centric alignment. To support event-level supervision, we first constructed EVA-Data with event-centric quality control and adopted a two-stage post-training strategy with SFT and GRPO. Then, by decomposing captions into modality-grounded events, EVA-Cap elaborates fine-grained GRPO rewards tailored to event completeness, character attribution, and audiovisual synchronization. Extensive experiments demonstrate that SFT on EVA-Data significantly improves OLMs to internalize event-centric knowledge. Furthermore, the subsequent GRPO optimization improves event-centric alignment while maintaining strong general captioning performance. These results suggest that event-centric alignment ofers an efective path toward more faithful and temporally coherent audiovisual video captioning.

## References

[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[2] O. Bar-Tal, H. Chefer, O. Tov, C. Herrmann, R. Paiss, S. Zada, A. Ephrat, J. Hur, G. Liu, A. Raj, et al. Lumiere: A space-time difusion model for video generation. In SIGGRAPH Asia, 2024.

[3] W. Chai, E. Song, Y. Du, C. Meng, V. Madhavan, O. Bar-Tal, J.-N. Hwang, S. Xie, and C. D. Manning. Auroracap: Eficient, performant video detailed captioning and a new benchmark. In ICLR, 2025.

[4] X. Chen, Y. Ding, W. Lin, J. Hua, L. Yao, Y. Shi, B. Li, Y. Zhang, Q. Liu, P. Wan, et al. Avocado: An audiovisual video captioner driven by temporal orchestration. In ICLR, 2026.

[5] X. Chen, W. Lin, J. Hua, L. Yao, Y. Ding, B. Li, B. Zeng, Y. Shi, Q. Liu, Y. Zhang, et al. Diadem: Advancing dia logue de scriptions in audiovisual video captioning for multi m odal large language models. In ECCV, 2026.

[6] X. Chen, J. Tang, Y. Ding, Y. Jia, B. Li, B. Zeng, Y. Shi, S. Li, Y. Ji, Q. Liu, et al. Capricorn-1k: A comprehensive benchmark for video captioning and subject referential consistency across temporal scales. In EMNLP, 2026.

[7] Y. Chen, Y. Wu, K. Guan, Y. Ren, Y. Wang, R. Song, and L. Ru. Chronusomni: Improving time awareness of omni large language models. In ECCV, 2026.

[8] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[9] M. Farré, A. Marafioti, L. Tunstall, L. Von Werra, and T. Wolf. Finevideo, 2024.

[10] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, J. Wu, X. Zhang, B. Wang, and X. Yue. Video-r1: Reinforcing video reasoning in mllms. In NeurIPS, 2025.

[11] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, 2025.

[12] Y. Ge, Y. Ge, C. Li, T. Wang, J. Pu, Y. Li, L. Qiu, J. Ma, L. Duan, X. Zuo, et al. Arc-hunyuan-video-7b: Structured video comprehension of real-world shorts. arXiv preprint arXiv:2507.20939, 2025.

[13] Y. Guo, J. Liu, M. Li, Q. Liu, X. Chen, and X. Tang. Trace: Temporal grounding video llm via causal event modeling. In ICLR, 2025.

[14] M. Han, L. Yang, X. Chang, L. Yao, and H. Wang. Shot2story: A new benchmark for comprehensive understanding of multi-shot videos. In ICLR, 2025.

[15] J. Hong, S. Yan, J. Cai, X. Jiang, Y. Hu, and W. Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. In ICLR, 2026.

[16] T.-H. Huang, S. Salekin, J. Movellan, F. Sala, and M. Bilkhu. Rubicap: Rubric-guided reinforcement learning for dense image captioning. arXiv preprint arXiv:2603.09160, 2026.

[17] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[18] S. Kumar, P. Seetharaman, K. Chen, O. Nieto, J. Su, Z. Wang, R. Kumar, D. Manocha, N. J. Bryan, Z. Jin, et al. Tac: Timestamped audio captioning. arXiv preprint arXiv:2602.15766, 2026.

[19] Y. Li, H. Zhang, M.-H. Guo, W. Gao, S. Jia, S. Jiao, Q. Hou, and M.-M. Cheng. Towards universal video mllms with attribute-structured and quality-verified instructions. arXiv preprint arXiv:2602.13013, 2026.

[20] Z. Liu, Y. Dong, J. Wang, Z. Liu, W. Hu, J. Lu, and Y. Rao. Ola: Pushing the frontiers of omni-modal language model. arXiv preprint arXiv:2502.04328, 2025.

[21] Z. Ma, R. Xu, Z. Xing, Y. Chu, Y. Wang, J. He, J. Xu, P.-A. Heng, K. Yu, J. Lin, et al. Omni-captioner: Data pipeline, models, and benchmark for omni detailed perception. In ICLR, 2026.

[22] D. Meng, R. Huang, Z. Dai, X. Li, Y. Xu, J. Zhang, Z. Huang, M. Zhang, L. Zhang, Y. Liu, et al. Videocap-r1: Enhancing mllms for video captioning via structured thinking. arXiv preprint arXiv:2506.01725, 2025.

[23] L. T. P. Nguyen, Z. Yu, S. L. Y. Hang, S. An, J. Lee, Y. Ban, S. Chung, T.-H. Nguyen, J. Maeng, S. Lee, et al. See, hear, and understand: Benchmarking audiovisual human speech understanding in multimodal large language models. In CVPR Findings, 2026.

[24] J. Pu, Y. Chen, T. Wang, and Y. Shan. Omniscript: Towards audio-visual script generation for long-form cinematic video. In ECCV, 2026.

[25] K. Qiu, M. Han, K. Liu, Y. Liu, J. Han, Y. Jiang, X. Wu, S. Wang, L. Zhang, and D. Yang. Omnifysics-captioner technical report: Grounding omni-modal understanding in the physical world for better captioning. arXiv preprint arXiv:2609.31714, 2026.

[26] R. Selvakumar, K. Jayakumar, S. Sakshi, S. Ghosh, R. Gao, and D. Manocha. Do audio-visual large language models really see and hear? In CVPR Findings, 2026.

[27] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[28] J. Shi, J. Wang, Z. You, B. He, and Z. Wu. Videoloom: A video large language model for joint spatial-temporal understanding. In ICML, 2026.

[29] C. Tang, Y. Li, Y. Yang, J. Zhuang, G. Sun, W. Li, Z. Ma, and C. Zhang. video-salmonn 2: Caption-enhanced audio-visual large language models. arXiv preprint arXiv:2506.15220, 2025.

[30] C. Tang, T. Wang, F. Rao, J. Lyu, and C. Zhang. D-orca: Dialogue-centric optimization for robust audio-visual captioning. arXiv preprint arXiv:2602.07960, 2026.

[31] B. S. Team. Introduction to techniques used in seed1. 6, 2025.

[32] G. Team, S. E. Abd, V. Aggarwal, R. Algayres, A. Andreev, O. Bachem, I. Ballantyne, C. Brick, V. Cărbune, M. Casbon, et al. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

[33] Q. Team. Qwen3.5: Accelerating productivity with native multimodal agents, 2026.

[34] R. Tian, Q. Dai, H. Hu, and Z. Wu. Tinymem: Condensing multimodal memory for long-form video action detection. TPAMI, 2026.

[35] S. Tu, Q. Tian, Y. Huang, Y. Wu, X. Han, K. Pan, W. Kong, J. Xiong, J.-W. Zhang, Z. Wu, et al. Prism: Dynamic sparse attention for native 2k joint video-audio generation model training. arXiv preprint arXiv:2610.05416, 2026.

[36] S. Tu, Q. Tian, Z. Yang, Y. Wu, X. Han, W. Kong, J. Xiong, J.-W. Zhang, Z. Zhong, L. Bo, et al. Baton: Explicit semantic blueprints for joint video-audio generation. In NeurIPS, 2026.

[37] J. Wang, D. Chen, C. Luo, B. He, L. Yuan, Z. Wu, and Y.-G. Jiang. Omnivid: A generative framework for universal video understanding. In CVPR, 2024.

[38] J. Wang, D. Chen, Z. Wu, C. Luo, L. Zhou, Y. Zhao, Y. Xie, C. Liu, Y.-G. Jiang, and L. Yuan. Omnivl: One foundation model for image-language and video-language tasks. In NeurIPS, 2022.

[39] J. Wang, L. Yuan, Y. Zhang, and H. Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024.

[40] W. Wang and Y. Yang. Videoufo: A million-scale user-focused dataset for text-to-video generation. In NeurIPS, 2025.

[41] Y. Wang, J. Wang, J. Tang, Y. Zhang, Z. Cao, H. Bian, Z. Zhang, W. Luo, Z. Pan, Z. Dong, J. Liu, and Z. Zhang. Avscap: Orchestrating audio-visual synergy for omni-modal video captioning. In EMNLP, 2026.

[42] M. Wu, K. Feng, B. Li, K. Gong, Z. Yin, and X. Yue. Avcap: Reinforcing audio-video joint caption with detail-aware reward. arXiv preprint arXiv:2608.06930, 2026.

[43] P. Wu, Y. Liu, Z. Zhu, E. Zhou, and J. Shen. Ugc-videocaptioner: An omni ugc video detail caption model and new benchmarks. arXiv preprint arXiv:2507.11336, 2025.

[44] L. Xing, X. Dong, Y. Zang, Y. Cao, J. Liang, Q. Huang, J. Wang, F. Wu, and D. Lin. Caprl: Stimulating dense image caption capabilities via reinforcement learning. In ICLR, 2026.

[45] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

[46] J. Xu, Z. Guo, H. Hu, Y. Chu, X. Wang, J. He, Y. Wang, X. Shi, T. He, X. Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

[47] Y. Xu, X. Li, Y. Yang, D. Meng, R. Huang, and L. Wang. Carebench: A fine-grained benchmark for video captioning and retrieval. In ICLR, 2026.

[48] Y. Xu, Z. Wang, Z. Wang, J. Zhang, Y. Yang, D. Meng, Y. Zhang, P. Wan, and L. Wang. Percepcap: Video captioner with structured spatio-temporal perception. In NeurIPS, 2026.

[49] Q. Yang, S. Yao, W. Chen, S. Fu, D. Bai, J. Zhao, B. Sun, B. Yin, X. Wei, and J. Zhou. Humanomniv2: From understanding to omni-modal reasoning with context. arXiv preprint arXiv:2506.21277, 2025.

[50] Z. Yang, J. Tao, R. Chen, Z. Yang, Y. Yuan, X. Zhao, Auden, K. Wang, S. Shao, B. Wang, S. Yves, and Q. Lu. Omnicapbench: A deep-structured evaluation framework for fine-grained audio-visual captioning. In NeurIPS, 2026.

[51] L. Yao, Y. Wei, Y. Zhang, L. Li, X. Chen, F. Song, Z. Wang, K. Ouyang, Y. Liu, L. Kong, et al. Timechat-captioner: Scripting multi-scene videos with time-aware and structural audio-visual captions. In ICML, 2026.

[52] Y. Yao, T. Yu, C. Wang, J. Cui, B. Xu, H. Zhu, T. Cai, F. Huang, T. Wang, W. Ma, et al. Minicpm-o 2.6: A gpt-4o level mllm for vision, speech, and multimodal live streaming on your phone, 2025.

[53] H. Ye, C.-H. H. Yang, A. Goel, W. Huang, L. Zhu, Y. Su, S. Lin, A.-C. Cheng, Z. Wan, J. Tian, et al. Omnivinci: Enhancing architecture and data for omni-modal understanding llm. In ICLR, 2026.

[54] H. Yu, B. Gong, H. Yuan, D. Zheng, W. Chai, J. Chen, K. Zheng, and F. Zhao. Videomar: Autoregressive video generation with continuous tokens. In NeurIPS, 2025.

[55] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. In NeurIPS, 2025.

[56] Y. Yuan, H. Zhang, W. Li, Z. Cheng, B. Zhang, L. Li, X. Li, D. Zhao, W. Zhang, Y. Zhuang, et al. Videorefer suite: Advancing spatial-temporal object understanding with video llm. In CVPR, 2025.

[57] B. Zhang, P. Zhang, X. Dong, Y. Zang, and J. Wang. Long-clip: Unlocking the long-text capability of clip. In ECCV, 2024.

[58] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li. Llava-video: Video instruction tuning with synthetic data. TMLR, 2025.

[59] C. Zhao, J. Ma, Q. Huang, T. Fan, H. Li, Z. Kang, X. Wei, J. Yang, and Y. Tai. Temporal and cross-modal alignment for enhanced audiovisual video captioning. In ECCV, 2026.

[60] C. Zheng, S. Liu, M. Li, X.-H. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

[61] C. Zhong, Q. Hou, Z. Zhou, Y. Zhang, S. Hao, H. Lu, H. Tang, and X. Bai. Owlcap: Harmonizing motion-detail for video captioning via hmd-270k and caption set equivalence reward. In AAAI, 2026.

[62] Z. Zhou, R. Wang, Z. Wu, and Y.-G. Jiang. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2025.

## A Technical appendices and supplementary material

## A.1 Details about the Training Objectives

## A.1.1 Formulation of Supervised Fine-Tuning

In the first stage, we perform supervised fine-tuning (SFT) on EVA-Data to enable the model to generate long-form audiovisual captions that integrate both audio and visual information. Given a video-instruction pair $( v , q )$ and its ground-truth caption $y = ( y _ { 1 } , \dots , y _ { T } )$ , the model is optimized with the standard next-token prediction objective:

$$
{ \mathcal { L } } _ { \mathrm { S F T } } ( \theta ) = - \mathbb { E } _ { ( v , q , y ) \sim { \mathcal { D } } _ { \mathrm { S F T } } } \left[ \log \pi _ { \theta } ( y \mid v , q ) \right] ,\tag{10}
$$

where $\mathcal { D } _ { \mathrm { S F T } }$ denotes the supervised training set, and θ represents the model parameters. This stage provides the initialization for the subsequent GRPO optimization.

## A.1.2 Formulation of Group Relative Policy Optimization

In the second stage, we optimize the SFT model with GRPO. For each audiovisual instruction input q, where q contains the video v and the textual instruction, the old policy $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ samples a group of outputs $\left\{ o _ { 1 } , o _ { 2 } , \ldots , o _ { G } \right\}$ Each output is evaluated by the event-centric reward function to obtain $R _ { i } = R _ { \mathrm { E V A } } ( o _ { i } , y ^ { \star } )$ $i = 1 , \ldots , G$ where $y ^ { \star }$ denotes the reference caption and its associated reference event structures. The advantage of the i-th response is computed by normalizing its reward within the sampled group: $\begin{array} { r } { \hat { A } _ { i } = \frac { R _ { i } - \operatorname* { m e a n } ( \{ R _ { 1 } , \ldots , R _ { G } \} ) } { \mathrm { s t d } ( \{ R _ { 1 } , \ldots , R _ { G } \} ) } } \end{array}$ The policy is optimized with the following clipped surrogate objective:

$$
\begin{array} { r l } & { \mathcal { I } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { P } ( q ) , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | q ) } \Bigg [ \frac { 1 } { G } \underset { i = 1 } { \overset { G } { \sum } } \frac { 1 } { | o _ { i } | } \underset { t = 1 } { \overset { | o _ { i } | } { \sum } } } \\ & { \quad \quad \quad \operatorname* { m i n } \left( r _ { i , t } ( \theta ) \hat { A } _ { i } , \mathrm { c l i p } ( r _ { i , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } _ { i } \right) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \Bigg ] , } \end{array}\tag{11}
$$

where

$$
r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { i , t } \mid q , o _ { i , < t } ) } .\tag{12}
$$

Here, $D _ { \mathrm { K L } } ( \cdot )$ denotes the KL divergence to the reference policy $\pi _ { \mathrm { r e f } } ,$ ε is the clipping range, and $\beta$ controls the KL penalty.

## A.2 More Implementation Details

## A.2.1 Details about the Construction of Reference Structures

We filter 6.4K videos from EVA-Data, all of which achieve a 100% hit ratio in Stage 3 of the annotation pipeline, ensuring highly accurate event-level supervision signals. To facilitate eficient reward computation for both GRPO training and rollout, we construct reference structures for these 6.4K selected videos. Specifically, this involves extracting audiovisual event units ${ \mathcal { E } } ^ { \star }$ from the ground-truth captions $y ^ { \star }$ and formulating audio-to-visual synchronization queries $\mathcal { Q } ^ { \star }$

We employ the reward judge LLM, Qwen3.5-27B [33], prompting it to sequentially extract audio events and visual events from the given caption to form a unified event list ${ \mathcal { E } } ^ { \star }$ . Each extracted audio event serves as a separate query. We input the complete ground-truth caption along with all the extracted visual events into the LLM, prompting it to select the temporally synchronized visual events corresponding to the current audio event $( \mathrm { i . e . , }$ the current query) from all the input visual events. Each visual event is assigned a unique index based on its chronological extraction order (i.e., the first extracted visual event has an index of 1). Since all visual events are extracted sequentially, we discard audio events whose matched visual event index span exceeds 5. This filtering step excludes audio events that cover an excessively long temporal range (e.g., background music that plays throughout the entire video), as such coarse-grained audio events provide limited support for fine-grained audiovisual alignment. Next, we construct a candidate visual event set containing 5 visual events by selecting those adjacent to the indices of the selected synchronized visual events. This strategy naturally incorporates neighboring visual events, which are prone to misalignment with the audio events—thereby creating challenging hard negative samples for subsequent reward assessment. Collectively, the filtered audio events paired with their corresponding constructed visual candidate sets form the final synchronization queries Q<sup>⋆</sup>.

As a result, we obtain an audiovisual event unit set and an audio-to-visual synchronization query set for each video. These well-defined reference structures enable reliable event-level reward calculation and support the subsequent GRPO training pipeline.

## A.2.2 Details about the Rollout Process

SFT initializes the audiovisual captioning format of the model and enables preliminary event alignment capabilities. For GRPO, we first conduct the rollout process to identify qualified videos and yield high-quality event-level supervision signals. We perform rollout on the videos selected in Sec. A.2.1, all equipped with the well-constructed reference structures (i.e., sets of audiovisual event units and audio-to-visual synchronization queries). Each selected video is then inferred 8 times (i.e., group size) using the SFT model.

Leveraging the constructed reference structures, we compute two types of event-level rewards for each generated caption during the rollout process: the intra-event attribution reward and the inter-event synchronization reward. Specifically, the intra-event attribution reward focuses on evaluating the proportion of events that are correctly attributed to the appropriate characters, while the inter-event synchronization reward quantifies the alignment between audio events and the constructed candidate visual events by calculating the F1-score.

To ensure the quality of supervision signals for subsequent preference training, we implement a twostage filtering mechanism at the query level and the video level. At the query level, invalid audioto-visual synchronization queries are filtered by the inter-event synchronization reward. Specifically, we aggregate reward values for each individual query across the 8 generated captions, and retain only queries whose average synchronization reward falls within the interval (0,0.75), following the filtering protocol of HumanOmniV2 [49]. At the video level, filtering is performed based on the sum of intra-event attribution reward and inter-event synchronization reward. We first eliminate videos with fewer than 5 valid synchronization queries, as such cases lack suficient synchronized semantic information to support reliable training. We then compute the standard deviation of the combined reward scores across the 8 generated captions per video, and finally retain the top 50% of qualified videos, yielding 3.2K valid video samples for downstream GRPO training.

We conduct two rounds of GRPO training, with one training epoch set for each round. In the first round, we only leverage Global Event Completeness and Inter-Event Synchronization for model optimization. Accordingly, the rollout video selection relies solely on the standard deviation of the inter-event synchronization reward. In the second round, we incorporate all three designed reward components into the joint training objective. Correspondingly, the rollout selection strategy is updated to adopt the standard deviation of the sum of inter-event synchronization reward and intra-event attribution reward as the screening criterion.

## A.3 Additional Experiment Results

To ensure testing eficiency, we use Qwen3.5-27B [33] instead of Gemini-2.5-Pro [8] and Seed-1.6 [31] as the judge LLM for event-centric evaluation in the experiments in this section. Additionally, we halve the GRPO training process, retaining only 1.6K video samples after rollout.

## A.3.1 Ablation on the Independent Efect of Data and Reward

In Sec. 4.3, we conduct thorough ablation experiments on our post-training pipeline. To clearly disentangle gains from constructed data versus reward guidance, we decouple the data and reward suites to evaluate each contribution to performance.

Data Efect. To separate the contribution of EVA-Data from the GRPO process, we compare SFT on EVA-Data with SFT on a same-size subset (63K) sampled from ASID-1M [19], which lacks event-centric quality control.

As shown in Tab. 5, replacing the SFT data with event-centric refined data improves both event-oriented and general performance, e.g., by +4.9% on A2V grounding and +3.6% on Daily-Omni. This demonstrates the efectiveness of our data construction pipeline and the quality of EVA-Data.

Table 5 Ablation experiments on diferent SFT dataset.
<table><tr><td rowspan="2">Dataset</td><td colspan="3">V2A</td><td colspan="3">A2V</td><td>AV-Speaker</td><td>Daily-Omni</td></tr><tr><td>M</td><td>Acc</td><td>Score</td><td>M</td><td>Acc</td><td>Score</td><td>Acc</td><td>Acc</td></tr><tr><td>ASID-1M (63K)</td><td>0.9</td><td>15.2</td><td>0.8</td><td>0.9</td><td>30.7</td><td>1.8</td><td>34.1</td><td>46.5</td></tr><tr><td>EVA-Data</td><td>1.2</td><td>20.5</td><td>1.2</td><td>1.1</td><td>35.6</td><td>2.1</td><td>39.6</td><td>50.1</td></tr></table>

Reward Efect. To test whether our event-centric reward framework generalizes beyond EVA-Data, we apply the same training recipe to a diferent publicly available model, ASID-Captioner [19]. Specifically, we use its released SFT model as the baseline and then perform two stages of GRPO training on the publicly available ASID-1M dataset, using the same reward suite we proposed.

Table 6 Ablation experiments on event-centric reward framework with public SFT model and data.
<table><tr><td rowspan="2">Setting</td><td colspan="3">V2A</td><td colspan="3">A2V</td><td>AV-Speaker</td><td>Daily-Omni</td></tr><tr><td>M</td><td>Acc</td><td>Score</td><td>M</td><td>Acc</td><td>Score</td><td>Acc</td><td>Acc</td></tr><tr><td>ASID baseline</td><td>1.1</td><td>16.8</td><td>1.0</td><td>0.9</td><td>30.4</td><td>1.8</td><td>38.4</td><td>48.5</td></tr><tr><td>ASID GRPO stage1</td><td>1.3</td><td>21.0</td><td>1.2</td><td>1.1</td><td>36.4</td><td>2.1</td><td>41.0</td><td>52.2</td></tr><tr><td>ASID GRPO stage2</td><td>1.3</td><td>22.2</td><td>1.3</td><td>1.1</td><td>39.0</td><td>2.1</td><td>42.2</td><td>53.3</td></tr></table>

These results in Tab. 6 show that, independent of EVA-Data, our event-centric framework can be efectively applied to another publicly available SFT model and yields consistent improvements across both event-centric and general benchmarks. This supports the generalizability of the reward framework and, together with the ablation in Tab. 5, further demonstrates that the final gains come from both the event-centric data and the framework.

## A.3.2 Robustness of the Reward Judge

We assess the robustness of the reward judge in two ways. First, before selecting Qwen3.5-27B as the reward LLM, we sampled 200 reward-evaluation items covering the three reward dimensions and compared its judgments against Gemini-2.5-Pro. The agreement rate is around 85%, suggesting that Qwen3.5-27B provides reliable event-level judgments for reward supervision without obvious systematic bias.

Second, we also replace Qwen3.5-27B with the open-source Gemma-4-31B-it [32] as the reward judge to further test training stability under diferent judges. We perform GRPO training (stage 1) using the same reward design (Global Event Completeness and Inter-Event Synchronization).

Table 7 Ablation experiments on diferent reward LLMs.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Reward LLM</td><td colspan="3">V2A</td><td colspan="3">A2V</td><td>Daily-Omni</td></tr><tr><td>M</td><td>Acc</td><td>Score</td><td>M</td><td>Acc</td><td>Score</td><td>Acc</td></tr><tr><td>SFT</td><td></td><td>1.2</td><td>20.5</td><td>1.2</td><td>1.1</td><td>35.6</td><td>2.1</td><td>50.1</td></tr><tr><td>GRPO</td><td>Qwen3.5-27B</td><td>1.4</td><td>23.0</td><td>1.3</td><td>1.2</td><td>40.1</td><td>2.3</td><td>53.0</td></tr><tr><td>GRPO</td><td>Gemma-4-31B-it</td><td>1.4</td><td>24.2</td><td>1.4</td><td>1.2</td><td>39.9</td><td>2.3</td><td>52.2</td></tr></table>

As shown in Tab. 7, under supervision from the Global Event Completeness and Inter-Event Synchronization rewards, both reward judges lead to similar and consistent improvements over SFT across event-centric and general benchmarks. This suggests that our reward design and training framework are robust to the choice of reward LLM, and that EVA-Cap learns fundamental event-centric alignment rather than merely exploiting the preferences of a specific judge.

## A.3.3 Length-Controlled Hallucination Evaluation

All models in our evaluation use the same maximum generation length of 2048. For length-controlled evaluation, we split video-SALMONN2 testset [29] predictions by token length into [0, 700), [700, 1000), and [1000, 2048], and recompute the hallucination rate within each partition.

Table 8 Hallucination comparison under diferent output length.
<table><tr><td>Model</td><td>[0, 700)</td><td>[700, 1000)</td><td>[1000, 2048)</td></tr><tr><td>DiaDem [5]</td><td>18.6</td><td>17.7</td><td>14.9</td></tr><tr><td>EVA-Cap</td><td>17.4</td><td>16.8</td><td>17.0</td></tr></table>

Under this length-matched comparison, EVA-Cap shows lower hallucination than DiaDem in the short-length and mid-length splits, while DiaDem is lower in the longest split. This suggests that EVA-Cap is more reliable when producing compact, event-centric audiovisual captions, but still lags behind strong baselines on long-form outputs. We will further study the trade-of between detail and hallucination in future work.

## A.3.4 Experiments with Event Precision Reward

The Global Event Completeness reward $\mathcal { R } _ { \mathrm { c o m p } }$ encourages more detailed captions. After GRPO training with ${ \mathcal { R } } _ { \mathrm { c o m p } } ,$ the average caption length on video-SALMONN2 testset increases from 633.0 tokens (SFT) to 748.2 tokens. However, the per-token hallucination rate does not increase (8.2 → 8.1 Hall./1k tokens), while event coverage is substantially improved. This suggests that the higher absolute hallucination rate is largely coupled with longer, more comprehensive descriptions, rather than a pure collapse into unfaithful verbosity.

Furthermore, we add an event-level precision term to the recall-based GRPO baseline, which is computed by decomposing the predicted caption into a sequence of audiovisual events and then determining the proportion of these events that are entailed in the groundtruth caption.

Table 9 Performance comparison with event precision reward.
<table><tr><td>Reward</td><td>Avg tokens</td><td>Hall.↓</td><td>Hall./1k token↓</td><td>Miss.↓</td><td>Total↓</td><td>Daily-Omni↑</td></tr><tr><td>Recall</td><td>748.2</td><td>17.7</td><td>8.1</td><td>19.0</td><td>36.7</td><td>53.7</td></tr><tr><td>Recall + Precision</td><td>624.5</td><td>16.2</td><td>8.9</td><td>20.9</td><td>37.1</td><td>51.6</td></tr></table>

Adding precision shortens captions and reduces absolute hallucination, but increases the miss rate, slightly hurts general performance, and does not improve length-normalized faithfulness (Hall./1k: 8.1 → 8.9), while raising training cost due to extra judge calls. This indicates that precision can be viewed as encouraging more conservative outputs. In contrast, our Intra-Event Attribution reward ${ \mathcal { R } } _ { \mathrm { a t t r } }$ already penalizes hallucinated or incorrectly grounded event content, providing a more targeted improvement in faithfulness. We therefore do not include precision/F1 in our reward suite.

## A.4 Discussion with Recent Work

Recent studies [6, 24, 25, 42] shift caption supervision from a holistic level to fine-grained instance and event levels. Here, we focus on three concurrent works highly relevant to our own: OmniCapBench [50], TCA-Captioner [59], and AVSCap [41]. Targeting fine-grained audiovisual captioning, OmniCapBench formalizes video content into a structured representation and employs three interdependent tracks (i.e.,

![](images/f46ff05024fabd721781c659a363b7f338abfc6e4710736aec5d6422da9fa8a1.jpg)  
off-screen: … the first man in space. Just weeks later, the United States launched its first astronaut, Alan Shepard

Predicted Caption: The audio begins with a deep, resonant, and slightly mysterious electronic score, … cosmonaut Yuri Gagarin became the first man in space." As he speaks, the view transitions to a blurry, … and a black-and-white photograph of a man in a military uniform, Yuri Gagarin. The narrator continues, "Just weeks later, the United States launched its first astronaut, Alan Shepard," while a graphic appears showing a line drawing of a "MERCURY-REDSTONE 3" rocket with the date "MAY 5, 1961" next to a photograph of the rocket on its launchpad. This is followed by a black-and-white archival clip of a man in a spacesuit …

![](images/1e356bc45015927bd69a2b5e0dd42e38e0eb28bc1efd853b33571c6ddeb30082.jpg)  
the man on the left: And today, we're going to be answering some questions from this … cookie jar?  
Predicted Caption: A medium shot shows three young people … zooms in slightly as Noah, the man on the left, looks toward Ashby and continues with a friendly, engaging tone, "And today, we're going to be answering some questions from this..." As he speaks, the camera cuts to a close-up of the cookie jar, which has the word "cookies" written in green on its side, while Ashby, the man on the right, reaches for it and finishes his sentence with a playful, slightly dramatic flourish, "...cookie jar?" The shot returns to …  
Figure 6 Failure cases of EVA-Cap, e.g., videos involving rapid scene transitions and multi-character scenes with of-screen speech. Incorrect descriptions are marked in red.

references, events, and shots) to evaluate whether a predicted event occurs at the correct time and links to the appropriate visual entities. TCA-Captioner emphasizes temporal and cross-modal alignment, binding auditory events to visual entities and capturing complex causal dynamics through data construction and SFT training. AVSCap achieves explicit cross-modal event binding through a designed reward, focusing on whether the caption captures temporally synchronized audiovisual events.

While our EVA-Cap shares the same objectives as these studies, we emphasize comprehensive, fine-grained, event-centric caption optimization and ofer a distinct solution. Although the data curation pipelines difer in detail, they all achieve high-fidelity caption annotation, providing diverse options for data scaling. Methodologically, the reward design in AVSCap is closely related to our approach, paralleling the mechanism employed in our Stage 3 annotation pipeline. A key distinction in reward design is that we decouple audio and visual events for calculation during the GRPO process, whereas AVSCap treats synchronized audiovisual events as a unified whole. We leave the experimental comparison of these design choices, as well as the exploration of data mixing and methodological integration, for future work.

## A.5 Failure Cases and Limitations

As shown in Fig. 6, we summarize two common failure modes of EVA-Cap: 1) In videos with rapid scene transitions, omissions in visual descriptions and misalignment between audio and visual descriptions still tend to occur. 2) In multi-character scenes with of-screen speech, the speaker identity is also prone to being incorrectly bound. This issue likely arises from limitations in understanding instances and capturing key information in complex, dynamic scenes. We intend to mitigate this limitation in future work by introducing grounded audiovisual captioning, to further enhance the fidelity and trustworthiness of audiovisual captioning by binding audio and visual cues to specific instances in space and time.

Although EVA-Data is constructed via a multi-stage pipeline (initialization–revision–filtering) that enables scalable annotation, it still relies on proprietary MLLMs for caption generation and quality control. This dependence may introduce biases from proprietary teachers and leave residual annotation errors after automatic verification. In the future, we plan to improve this pipeline by integrating agent-based, multi-level cross-verification: multiple models and rule-based checkers cross-validate captions and events at fine granularity, recursively refine inconsistent annotations, and reject insuficiently verified samples, thereby improving annotation fidelity while reducing dependence on single models.

## A.6 Prompt Design

## A.6.1 Prompt for Annotation Pipeline

In Figs. 7, 8, and 9, we present the prompts adopted in the audiovisual captioning data annotation pipeline introduced in Sec. 3.1.1. In this process, we first prompt Gemini-2.5-Pro to refine the initial audiovisual captions in terms of cross-modal synchronization and event attribution, and to extract timestamped audiovisual event tuples in parallel. We then employ GPT-4.1 to conduct event hit verification for data quality control.

## A.6.2 Prompt for Reward Implementation

Fig. 10 illustrates the prompt used to extract audiovisual events from audiovisual captions, facilitating the construction of reference structures. Furthermore, Figs. 11, 12 and 13 list the prompts used, respectively, for the judgement of global event completeness reward, intra-event attribution reward and inter-event synchronization reward. All these prompts are carefully designed to standardize LLM-based reward judgement and ensure consistent evaluation across all samples.

## A.7 Broader Impact

EVA-Cap aims to advance audiovisual video captioning by generating more faithful, comprehensive, and temporally coherent descriptions of videos. This can benefit various downstream applications that rely on detailed and accurate video understanding. Better alignment between audio and visual events may help reduce misleading descriptions in scenarios where both modalities are necessary for understanding the video content.

At the same time, improved audiovisual captioning systems may introduce potential risks if deployed without appropriate safeguards. First, generated captions may still contain omissions, hallucinations, incorrect speaker attributions, or inaccurate audio-visual temporal relations. Such errors could mislead users if the captions are treated as factual summaries. Second, automatic captioning of user-generated videos may raise privacy concerns when videos contain identifiable people, sensitive conversations, or private environments. Third, high-quality video descriptions could be misused for large-scale surveillance, profiling, or automated indexing of sensitive content.

![](images/2ef9b5abeec6097a4430c39d6912580fce3e01fe5022860c638009bad1f598a1.jpg)  
Figure 7 Prompt for caption refinement during annotation stage 2 with Gemini-2.5-Pro.

![](images/c4a82788e7707253df941552f1b8e383c1a90452cbf0d594c9ec1d921d1053e5.jpg)  
Figure 8 Prompt for timestamped event tuple extraction during annotation stage 3 with Gemini-2.5-Pro.

![](images/db21adcb158fb8eb6dc3f23fb4f4581499125500f727b05612f5ad18352da3bc.jpg)  
Figure 9 Prompt for event-centric data verification during annotation stage 3 with GPT-4.1.

![](images/0b443515a5dab3a99b06c7f4ee4c81218e6716d76fb4c309eda2739d473a9772.jpg)  
Figure 10 Prompt for audiovisual event extraction with Qwen3.5.

![](images/1c063a599c4acac4c8e636835cf2ae3c478edd2ac703e95d861243555ff48a9b.jpg)  
Figure 11 Prompt for global event completeness reward judgement via Qwen3.5.

![](images/aeea585b9e6c6dfc84dc9d77b913958dda6b4b6b7908ac0f4d6fd2498538f6d1.jpg)  
Figure 12 Prompt for intra-event attribution reward judgement via Qwen3.5.

![](images/d2282b6cfef64fd72ec415ac99510997d23371425281f33b2e0f35a31678bf21.jpg)  
Figure 13 Prompt for inter-event synchronization reward judgement via Qwen3.5.