# Encoder Awakening via Adapters: Effective Domain-Adaptive Fine-tuning of Speech-LLMs

Mohan Shi, Zilai Wang, Natarajan Balaji Shankar, Kaiyuan Zhang, Eray Eren, Abeer Alwan

Department of Electrical and Computer Engineering

University of California Los Angeles

Los Angeles, USA

{shimohan, zilaiwang2001, balaji1312, kaiyuanzhang, erayeren}@ucla.edu, alwan@ee.ucla.edu

Abstract—Speech Large Language Models (Speech-LLMs), typically built from a pre-trained speech encoder, a modality projector, and an LLM fine-tuned with Low-Rank Adapters (LoRA), have shown strong Automatic Speech Recognition (ASR) performance on general-domain speech. However, adapting them to domain-shifted speech, such as child or dialectal speech, remains challenging under limited target-domain data. Given the dominant role of the LLM in Speech-LLMs, with cross-entropy loss applied only at the LLM output, the speech encoder may receive insufficient adaptation to new acoustic conditions.

In this paper, we propose Encoder Awakening via Adapters (EAVA), a simple yet effective domain-adaptive fine-tuning method for Speech-LLM-based ASR. First, lightweight adapters are inserted into each encoder layer and trained exclusively, enabling target-domain acoustic knowledge to be incorporated into the encoder while preserving its pre-trained knowledge. Second, the full model is jointly fine-tuned on the target domain with LoRA applied to the LLM. Experiments on three domainshifted ASR datasets, covering child and dialectal speech, show that EAVA consistently outperforms vanilla fine-tuning and other baselines, achieving new state-of-the-art performance.<sup>1</sup>

Index Terms—Speech-LLMs, Automatic Speech Recognition, Domain Adaptation, Children’s Speech, Dialectal Speech.

## I. INTRODUCTION

Speech Large Language Models (Speech-LLMs), which benefit from a pre-trained speech encoder [1]–[3] or tokenizer [4], [5] and the language generation and reasoning capabilities of powerful Large Language Models (LLMs) [6]–[8], have shown strong performance on a wide range of speechrelated tasks [9]–[15]. A standard Speech-LLM for Automatic Speech Recognition (ASR) leverages a pre-trained speech encoder, a modality projector, and a powerful LLM, finetuned together on large-scale paired speech-transcription data with Low-Rank Adaptation (LoRA) [16] applied to the LLM. This framework has shown impressive ASR performance on general-domain speech [17], [18].

However, for speech domains with substantial domain shift relative to the general domain, such as child speech [19], [20] and dialectal speech [21], pre-trained Speech-LLMs often perform poorly due to the acoustic mismatch. Simply finetuning the encoder, projector and LoRA in LLM using crossentropy (CE) loss is the standard way to adapt a well-trained Speech-LLM to a target domain for ASR task, but it is not effective enough for handling the acoustic domain shift. For example, no work has shown Speech-LLMs can improve performance on children ASR; the current state-of-the-art results are still from fine-tuning other speech foundation models (not Speech-LLMs) [22]–[24]. One challenge in domain-adaptive fine-tuning of Speech-LLMs is that the speech encoder may remain insufficiently adapted to the target acoustic domain, as the large LLM component tends to dominate the model and the cross-entropy loss is applied only at the LLM output. Moreover, such domain-shift scenarios typically involve limited data for fine-tuning, further complicating adaptation [25]. On the other hand, if the encoder is trained too aggressively, it may fail to preserve its pre-trained knowledge, leading to degraded performance. So how to effectively adapt a pretrained Speech-LLM to a target acoustic domain is a challenging and underexplored problem. Other works adopt carefully designed multi-stage alignment training procedures to better align the speech and LLM representation spaces. [11], [26]. Although these approaches achieve competitive performance in their settings, their objective is to integrate a raw speech encoder with an LLM from scratch to build a Speech-LLM, which is fundamentally different from adapting a well-trained model to a new acoustic domain. Moreover, their procedures are complex and may not generalize to domain-adaptive finetuning task.

To address this gap, we propose Encoder Awakening via Adapters (EAVA), a simple yet effective method for domainadaptive fine-tuning of Speech-LLMs for ASR. EAVA consists of two stages. In the first stage, termed Encoder Awakening, lightweight adapters are inserted into each layer of the pretrained speech encoder, and only these adapters are trained while all other components remain frozen. This design injects target-domain acoustic knowledge into the encoder while preserving its pre-trained capabilities. In the second stage, termed Continual Fine-tuning, the encoder adapters, the projector, and the LoRA modules in the LLM are jointly fine-tuned to further adapt the Speech-LLM to the target domain, leveraging the domain-aware initialization obtained from the first stage. Experiments on several representative domain-shifted datasets, including child and dialectal speech, show that EAVA significantly outperforms all baselines and achieves new stateof-the-art ASR results on all evaluated datasets. The method also demonstrates consistent improvements across different Speech-LLM backbones.

## II. BACKGROUND AND RELATED WORKS

## A. Speech Large Language Models for ASR

Speech-LLMs combine a pre-trained speech encoder with a powerful LLM via a modality projector to enable end-toend Automatic Speech Recognition (ASR) [9], as illustrated in Figure 1. The speech encoder, typically based on the Conformer [27] or similar architectures, extracts frame-level acoustic representations from the input speech signal. The projector maps these representations into the token embedding space of the LLM, which then generates text output autoregressively. These components are jointly fine-tuned, with Low-Rank Adaptation (LoRA) commonly applied to the LLM. Formally, given a speech signal x and a text prompt p, the forward process is:

$$
\mathbf { H } = \mathrm { E n c o d e r } ( \mathbf { x } ) \in \mathbb { R } ^ { T \times d _ { e } } ,\tag{1}
$$

$$
\mathbf { Z } = \mathrm { { P r o j e c t o r } } ( \mathbf { H } ) \in \mathbb { R } ^ { T \times d _ { l } } ,\tag{2}
$$

where $T$ is the sequence length, and $d _ { e } , \ d _ { l }$ are the encoder and LLM hidden sizes, respectively. The LLM then autoregressively generates the transcription $\begin{array} { r l } { \hat { \mathbf { y } } } & { { } = } \end{array}$ $\left( \hat { y } _ { 1 } , \dotsc , \hat { y } _ { N } \right)$ conditioned on Z and the embedded prompt: Embedding(Tokenizer(p)).

Representative open-source Speech-LLMs include Canary-Qwen [17], which pairs a pre-trained FastConformer encoder [28] with a decoder-only LLM Qwen3-1.7B [8], and Phi-4-Multimodal [18], which integrates speech understanding into the Phi-4-Mini [29] language model via a dedicated speech encoder. These models are pre-trained on large-scale paired speech-transcription data and achieve strong ASR performance on general-domain speech.

## B. Problem Formulation

Consider a Speech-LLM pre-trained on a large generaldomain dataset, which generates ASR transcriptions from speech inputs. Given a target-domain training set $\mathcal { D } _ { \mathrm { t g t } } ^ { \mathrm { t r a i n } }$ exhibiting acoustic domain shift (e.g., child or dialectal speech), domain-adaptive fine-tuning aims to adapt the model for improved ASR performance on the target domain.

## C. Previous Fine-tuning Strategies

1) Vanilla Fine-tuning: The most straightforward approach is vanilla fine-tuning, which involves jointly fine-tuning the encoder, projector, and LLM LoRA on the target-domain training set $\mathcal { D } _ { \mathrm { t g t } } ^ { \mathrm { t r a i n } }$ using the cross-entropy (CE) loss computed between the predicted and ground-truth transcriptions. However, as discussed in the Introduction, this approach struggles to balance effective encoder adaptation and preservation of pre-trained knowledge under limited target-domain data.

2) Multi-stage Alignment: Previous works [11], [26] have adopted multi-stage alignment for Speech-LLMs to better align the speech and LLM representation spaces. Specifically, [26] proposed a training strategy that first fine-tuned the encoder, then trained only the projector, and finally trained the projector and LLM LoRA jointly. However, their objective is to integrate a raw speech encoder with an LLM from scratch, rather than to adapt a well-trained model to a new domain.

![](images/b30f5c8b04b4e948b580660b80966ada8e2b38807ed999abdd00bf2c0577f2bc.jpg)  
Fig. 1. Illustration of a standard well-trained Speech-LLM for ASR. LLM autoregressively generates the transcription conditioned on embedded speech representation and the embedded prompt.

## III. METHOD

As discussed in the previous sections, effective domainadaptive fine-tuning strategies for Speech-LLMs remain largely unexplored. To address this gap, we propose Encoder Awakening via Adapters (EAVA), a simple yet effective method for this task. Figure 2 illustrates an overview of EAVA, which consists of two stages.

## A. Stage 1: Encoder Awakening

The first stage is the encoder awakening stage, which aims to awaken the encoder toward the target domain. Specifically, starting from a well-trained Speech-LLM, randomly initialized lightweight adapters are inserted into each encoder layer, and only these adapters are trained on the target-domain training set $\mathcal { D } _ { \mathrm { t g t } } ^ { \mathrm { t r a i n } }$ using the standard LLM CE loss, while all pre-trained parameters remain frozen, as shown in Figure 2 (Stage 1). This stage aims to effectively inject target-domain knowledge into the encoder via the adapters, while preserving the pretrained knowledge of the encoder itself. In this work, we employ Residual Adapters (RA) [30] as the default adapter choice, each consisting of two feed-forward layers with a residual connection, along with layer normalization and a Swish activation function [31]. Formally, let $\mathbf { h } \in \mathbb { R } ^ { d _ { e } }$ be the output hidden state of an encoder layer. The RA computes:

$$
\mathrm { R A } ( \mathbf { h } ) = \mathbf { h } + W _ { \mathrm { u p } } \cdot \mathrm { S w i s h } ( W _ { \mathrm { d o w n } } \cdot \mathrm { L a y e r N o r m } ( \mathbf { h } ) )\tag{3}
$$

where $W _ { \mathrm { d o w n } } ~ \in ~ \mathbb { R } ^ { r \times d _ { e } }$ and $W _ { \mathsf { u p } } \in \mathbb { R } ^ { d _ { e } \times r }$ are the adapter weights, and r is the bottleneck dimension (referred to as the adapter dimension in experiments). During Stage 1, only $\{ W _ { \mathrm { d o w n } } , W _ { \mathrm { u p } } \}$ in each encoder layer are updated; all other parameters remain frozen.

## B. Stage 2: Continual Fine-tuning

After the Encoder Awakening stage, the encoder adapters, the pre-trained encoder parameters, the projector, and the

![](images/0a452257e6f83eb55adc5efe061fd844fb0717a695b9c15b0c16e800c3c38105.jpg)  
Fig. 2. Overview of the proposed EAVA method for domain-adaptive fine-tuning of a well-trained Speech-LLM. A Speech-LLM trained on large-scale general-domain speech is used as the parameter initialization and adapted to target-domain speech, such as child or dialectal speech. In Stage 1 (Encoder Awakening), lightweight adapters are inserted into each encoder layer and only the adapter parameters are trained, while all other components remain frozen. In Stage 2 (Continual Fine-tuning), the encoder adapters, the pre-trained encoder parameters, the projector, and the LLM LoRA modules are jointly fine-tuned.

LLM LoRA modules are jointly fine-tuned on $\mathcal { D } _ { \mathrm { t g t } } ^ { \mathrm { t r a i n } }$ using the same CE loss, as shown in Figure 2 (Stage 2). The encoder adapters, having been initialized with target-domain acoustic knowledge in Stage 1, provide a domain-aware starting point for the encoder. Joint fine-tuning then further adapts the entire Speech-LLM to the target domain, allowing the encoder, projector, and LLM LoRA modules to be updated coherently under the same ASR objective. This two-stage design first effectively awakens the encoder toward the target acoustic domain while preserving its pre-trained knowledge, and then refines the entire Speech-LLM for target-domain ASR, resulting in more targeted adaptation than directly finetuning all trainable components jointly.

## IV. EXPERIMENTAL SETTINGS

## A. Datasets

To evaluate our proposed method, we conduct experiments on three datasets that exhibit acoustic domain shift from the general adult English speech domain, covering child speech and dialectal speech variations in english.

• OGI [19]: The OGI Kids corpus contains child speech collected in classroom settings. We select its spontaneous portion, which consists of responses to open-ended questions from children aged 4–15, with transcriptions that preserve disfluencies. We refer to this subset as OGI throughout the paper. Following [23], we split OGI into 22/2/7 hours for train/dev/test.

• MyST [20]: The MyST corpus comprises dialogues between elementary school students aged 8–10 and virtual tutors. We split the transcribed portion following [22], [33], resulting in 133/21/25 hours for train/dev/test.

• CORAAL [21]: The CORAAL corpus contains sociolinguistic interviews in African American Language. <sup>❄️</sup>Following [24], [32], we use six subsets (ATL, LES,

DCA, DCB, DTA, PRV; 137h) for training, and hold out ROC (13h) and VLD (12h) for development and testing, ensuring speaker and regional disjointness. Utterances are trimmed to retain only interviewee speech and are capped at 30 seconds.

## B. Model Settings

For the main experiments, we adopt Canary-Qwen [17] as the primary backbone. Canary-Qwen is an open-source Speech-LLM developed by NVIDIA that pairs a pre-trained FastConformer encoder [28] (810M parameters) with a Qwen3-1.7B LLM [8]. It is pre-trained on large-scale multilingual speech-transcription data and achieves strong performance on general-domain ASR benchmarks. For the adapter dimension, we use 64 by default and also evaluate other sizes: 32, 128, 256, and 512. For model training, in the encoder awakening stage, the adapters are trained with a peak learning rate of 1e-3 using a linear warmup followed by cosine annealing decay. For the Continual fine-tuning stage, a peak learning rate of 1e-4 is used with the same schedule. Each stage is trained for 5 epochs, and the best checkpoint is selected based on development set performance. We also evaluate the proposed method on the speech branch of Phi-4-Multimodal [18], a multimodal LLM from Microsoft that integrates speech and vision encoders with the Phi-4- Mini language model [29], to verify its generalizability across different Speech-LLM backbones. All models are trained on a single NVIDIA A6000 GPU with an effective batch size of 16 (batch size × gradient accumulation steps). We use the Word Error Rate (WER) as the evaluation metric.

## C. Baselines

We compare against the following baselines:

• Previous SOTA: The best previously published WER results on each dataset using standard training data and data processing [22], [23], [32]. For reference, we also report stronger results obtained using additional unlabeled data for training [24].

TABLE I  
WER (%) ON THREE TARGET-DOMAIN DATASETS USING CANARY-QWEN AS THE BACKBONE. ADAPTER PARAMETERS ARE REPORTED AS PERCENTAGES OF THE ENCODER SIZE (810M). BOLD DENOTES THE BEST RESULT IN EACH COLUMN. ALL EAVA VARIANTS ACHIEVE STATISTICALLY SIGNIFICANT IMPROVEMENTS OVER ALL THREE BASELINES IMPLEMENTED IN OUR EXPERIMENTS (ZERO-SHOT, VANILLA FINE-TUNING, AND MULTI-STAGE ALIGNMENT), AS MEASURED BY MAPSSWE (p < 0.05).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Adapter Dim</td><td rowspan="2">Adapter Params</td><td colspan="2">OGI</td><td colspan="2"> $\mathbf { M y S T }$ </td><td colspan="2">CORAAL</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>Previous SOTA [22], [23], [32]</td><td></td><td></td><td>10.40</td><td>11.60</td><td>7.90</td><td>8.50</td><td>1</td><td>9.70</td></tr><tr><td>+ Additional Unlabeled Training Data [24]</td><td></td><td></td><td></td><td>11.06</td><td>7.74</td><td>8.21</td><td>6.10</td><td>9.25</td></tr><tr><td>Zero-shot Canary-Qwen</td><td></td><td></td><td>14.28</td><td>16.30</td><td>8.24</td><td>8.96</td><td>8.93</td><td>12.37</td></tr><tr><td>Vanilla Fine-tuning</td><td>一</td><td></td><td>9.39</td><td>10.95</td><td>7.63</td><td>8.37</td><td>6.11</td><td>8.87</td></tr><tr><td>Multi-stage Alignment [26]</td><td>1</td><td></td><td>9.29</td><td>10.79</td><td>7.86</td><td>8.60</td><td>6.50</td><td>9.54</td></tr><tr><td rowspan="5">EAVA (Ours)</td><td>32</td><td>2.2M (0.27%)</td><td>7.85</td><td>9.32</td><td>7.35</td><td>8.07</td><td>5.84</td><td>8.53</td></tr><tr><td>64</td><td>4.3M (0.53%)</td><td>8.15</td><td>9.05</td><td>7.33</td><td>8.04</td><td>5.84</td><td>8.54</td></tr><tr><td>128</td><td>8.5M (1.05%)</td><td>8.13</td><td>9.59</td><td>7.35</td><td>8.04</td><td>5.86</td><td>8.55</td></tr><tr><td>256</td><td>16.8M (2.07%)</td><td>8.19</td><td>9.34</td><td>7.44</td><td>8.09</td><td>5.87</td><td>8.67</td></tr><tr><td>512</td><td>33.6M (4.15%)</td><td>8.50</td><td>9.81</td><td>7.40</td><td>8.11</td><td>5.89</td><td>8.67</td></tr></table>

• Zero-shot: The backbone Speech-LLM evaluated directly on the target domain without fine-tuning, serving as a measure of its zero-shot generalization capability.

• Vanilla Fine-tuning: The encoder, projector, and LLM LoRA are jointly fine-tuned on the target-domain training set using Cross-Entropy (CE) loss, as described in Section II-C1. This represents the most straightforward adaptation strategy for Speech-LLMs.

• Multi-stage Alignment: The multi-stage alignment procedure from [26] as described in Section II-C2, which sequentially fine-tunes different model components to align speech and language representations.

We follow the same data splits and processing procedures as previous SOTA works to ensure fair comparisons. To ensure a fair comparison in terms of training budget, both Vanilla and Multi-stage Alignment are trained for the same number of total epochs as the proposed method, and results are reported using the checkpoint with the best development set performance. In practice, the best checkpoint is typically obtained within the first few epochs, while additional training often leads to overfitting. Although many fine-tuning strategies have been proposed, we select Vanilla Fine-tuning and Multistage Alignment [26] as two representative and competitive adaptation baselines from prior work.

## V. EXPERIMENTAL RESULTS

## A. Main Results and Analysis

Table I presents the main results using Canary-Qwen as the backbone across all three target-domain datasets. Vanilla fine-tuning improves over zero-shot performance on all test sets, but the gains remain limited. Multi-stage alignment yields marginal improvements on OGI but degrades performance on the other two datasets, suggesting it is ill-suited for domainadaptive fine-tuning of a well-trained Speech-LLM.

EAVA consistently outperforms all baselines across all evaluated adapter dimensions and all three test datasets. Notably, it achieves new state-of-the-art results even compared to prior works that leverage additional unlabeled training data, with the largest gains on OGI, likely because spontaneous speech from younger children represents a particularly challenging domain shift. Furthermore, larger adapters do not yield consistent improvements, possibly because the target-domain adaptation can be achieved with a relatively small number of trainable parameters, making additional adapter capacity less beneficial. Overall, the 64-dim adapter yields the most stable performance across all three datasets (test WER: OGI: 9.05%, MyST: 8.04%, CORAAL: 8.54%) and is adopted as the default configuration for the remaining experiments.

![](images/e28cfdd1bde8e9035443db96ae87c73e04b9b775a81359be6d2015192ad12c3c.jpg)  
Fig. 3. Layer-wise CKA similarity between the zero-shot and fine-tuned encoder representations on the OGI test set. Lower CKA indicates greater representational change relative to the zero-shot encoder.

To understand how each fine-tuning strategy reshapes the encoder, we compute layer-wise Centered Kernel Alignment (CKA) [34] between each fine-tuned encoder and the zeroshot encoder on the OGI test set, as it is the most challenging dataset. For each encoder layer, frame-level activations are mean-pooled over time to obtain per-utterance representations; CKA is measured at the same layer depth. Lower CKA indicates greater representational shift from the zero-shot encoder.

Figure 3 shows the layer-wise CKA diagonal. Vanilla finetuning produces a nearly flat curve (CKA: 0.92–0.99), indicating mild and uniformly distributed drift across all layers. Our proposed method EAVA exhibits a distinctly different profile: lower encoder layers (0–10) are almost perfectly preserved $( \mathrm { C K A } > 0 . 9 7 )$ , while upper layers (23–31) diverge substantially (CKA as low as 0.82). This pattern aligns with the design rationale of our method: upper encoder layers, whose representations are passed through the projector to the LLM, benefit most from domain adaptation, while lower layers retain general acoustic features. The targeted adaptation of upper layers while preserving lower-level representations leads to more effective domain transfer.

TABLE II  
ABLATION STUDY ON THE TRAINING PROCEDURE USING CANARY-QWEN AS THE BACKBONE, EVALUATED BY WER (%) ON THE OGI TEST SET. THE TABLE SHOWS THE FINE-TUNED MODULES AND THE NUMBER OF TRAINABLE PARAMETERS AT EACH STAGE. FOR SETTINGS WITH ADAPTERS, THE ADAPTER DIMENSION IS SET TO 64. † DENOTES THE DEFAULT TRAINING PROCEDURE OF THE PROPOSED METHOD.
<table><tr><td>Stage 1</td><td>Params</td><td>Stage 2</td><td>Params</td><td>WER</td></tr><tr><td>Adapter</td><td>4.3M</td><td></td><td></td><td>9.16</td></tr><tr><td>Adapter†</td><td>4.3M</td><td>Adapter+Encoder+Projector+LoRA†</td><td>842M</td><td>9.05</td></tr><tr><td>Adapter</td><td>4.3M</td><td>Encoder+Projector+LoRA</td><td>838M</td><td>9.10</td></tr><tr><td>Adapter</td><td>4.3M</td><td>Projector+LoRA</td><td>27.8M</td><td>9.27</td></tr><tr><td>Adapter+Projector+LoRA</td><td>32M</td><td></td><td></td><td>9.89</td></tr><tr><td>Adapter+Projector+LoRA</td><td>32M</td><td>Adapter+Encoder+Projector+LoRA</td><td>842M</td><td>9.84</td></tr><tr><td>Adapter+Encoder+Projector+LoRA</td><td>842M</td><td></td><td></td><td>10.23</td></tr><tr><td>Full Encoder (No Adapters)</td><td>810M</td><td></td><td></td><td>13.97</td></tr><tr><td>Full Encoder (No Adapters)</td><td>810M</td><td>Encoder+Projector+LoRA (No Adapters)</td><td>838M</td><td>11.82</td></tr></table>

## B. Ablation Study of Training Procedure

Our proposed method consists of two stages: in Stage 1 (Encoder Awakening), lightweight adapters are inserted into each encoder layer and only the encoder adapters are trained; in Stage 2 (Continual Fine-tuning), the adapter, encoder, projector, and LLM LoRA are jointly fine-tuned. To validate this design, we conduct an ablation study on the OGI test set using Canary-Qwen as the backbone, examining the effect of varying the trainable modules in each stage (Table II).

The first four rows share the same Stage 1 configuration (adapter-only) and vary the trainable components in Stage 2. Stage 1 performs the main encoder adaptation and already provides a strong starting point, while Stage 2 further refines the model by jointly optimizing the encoder, adapters, projector, and LLM LoRA. With the Canary-Qwen backbone, skipping Stage 2 results in a WER of 9.16%, while the proposed procedure achieves the best performance. Fine-tuning only a subset of components in Stage 2 also leads to worse results, indicating that joint optimization provides complementary refinement beyond Stage 1.

The next two rows examine alternative Stage 1 configurations. Training the projector and LoRA alongside the adapter in Stage 1 degrades performance compared to adapter-only training, as optimizing multiple modules simultaneously divides the training focus, making it less effective at injecting target-domain knowledge into the encoder adapters in a targeted manner. In addition, inserting adapters and training all modules jointly in a single step, without a dedicated Encoder Awakening stage, is also worse than the proposed two-stage design, demonstrating the importance of first awakening the encoder adapters before joint fine-tuning. Finally, replacing the adapters with full encoder fine-tuning in Stage 1 leads to a substantial performance drop, as it fails to preserve the encoder’s pre-trained knowledge.

TABLE III  
COMPARISON OF ADAPTER ARCHITECTURES IN THE EAVA FRAMEWORK USING CANARY-QWEN AS THE BACKBONE. ALL ADAPTER VARIANTS HAVE APPROXIMATELY EQUAL PARAMETER COUNTS (∼4.2–4.3M). † DENOTES THE DEFAULT ADAPTER CHOICE OF OUR PROPOSED METHOD. ALL ADAPTER VARIANTS SHOW STATISTICALLY SIGNIFICANT IMPROVEMENTS OVER VANILLA FINE-TUNING (p < 0.05).
<table><tr><td rowspan="2">Adapter Choice</td><td rowspan="2">Adapter Params</td><td colspan="2">OGI</td><td colspan="2">MyST</td><td colspan="2">CORAAL</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>Vanilla Fine-tuning</td><td>1</td><td>9.39</td><td>10.95</td><td>7.63</td><td>8.37</td><td>6.11</td><td>8.87</td></tr><tr><td>Residual Adapter [30]†</td><td>4.3M</td><td>8.15</td><td>9.05</td><td>7.33</td><td>8.04</td><td>5.84</td><td>8.54</td></tr><tr><td>Houlsby Adapter [35]</td><td>4.3M</td><td>8.10</td><td>9.41</td><td>7.32</td><td>8.00</td><td>5.70</td><td>8.49</td></tr><tr><td>LoRA [16]</td><td>4.2M</td><td>8.07</td><td>9.42</td><td>7.28</td><td>8.08</td><td>5.89</td><td>8.60</td></tr></table>

## C. Effect of Adapter Choice

To investigate whether the choice of adapter architecture affects the effectiveness of EAVA, we compare three adapter variants using Canary-Qwen as the backbone, all with approximately equal parameter counts (∼4.2–4.3M): the Residual Adapter (dim=64) in the proposed method, Houlsby Adapter [35] (dim=32, inserted after both the attention and FFN sublayers within each encoder layer), and LoRA (rank=16, alpha=32, applied to Q, K, V, and O projections). As shown in Table III, all three variants consistently outperform vanilla fine-tuning across all three datasets, demonstrating that the general design of EAVA is the primary driver of improvement, rather than the specific adapter architecture. The performance differences among the three adapter types are small, with the proposed Residual Adapter achieving the strongest result on the most challenging OGI dataset. These results confirm that EAVA is a flexible framework that is not tied to a particular adapter design.

## D. Cross-Dataset Transfer Analysis

To evaluate whether the first training stage (Encoder Awakening) learns transferable domain-adaptive encoder representations, we conduct a cross-dataset transfer analysis using Canary-Qwen as the backbone. In this setting, Stage 1 is trained on one dataset, and Stage 2 is then performed separately on each target-domain dataset. As shown in Table IV, even using mismatched data in Stage 1 outperforms vanilla fine-tuning in most cases, while using the matched targetdomain data for Stage 1 generally yields the strongest results. These findings support the effectiveness of Encoder Awakening and suggest that the encoder can often still benefit from the awakening stage even when mismatched data are used.

TABLE IV  
CROSS-DATASET TRANSFER RESULTS FOR STAGE 1 (ENCODER AWAKENING) USING CANARY-QWEN AS THE BACKBONE, EVALUATED BY TEST WER (%). STAGE 2 IS PERFORMED ON EACH TARGET-DOMAIN DATASET WITH 64-DIM ADAPTERS.
<table><tr><td>Method</td><td>Training Data for Stage 1</td><td>OGI</td><td>MyST</td><td>CORAAL</td></tr><tr><td>Vanilla Fine-tuning</td><td>一</td><td>10.95</td><td>8.37</td><td>8.87</td></tr><tr><td rowspan="3">EAVA (Ours)</td><td>OGI</td><td>9.05</td><td>8.12</td><td>8.84</td></tr><tr><td>MyST</td><td>10.06</td><td>8.04</td><td>8.98</td></tr><tr><td>CORAAL</td><td>9.45</td><td>8.10</td><td>8.54</td></tr></table>

TABLE V

COMPARISON OF LOSS FUNCTIONS USED IN THE FIRST STAGE (ENCODER AWAKENING), EVALUATED BY TEST WER (%) ON ALL THREE DATASETS. THE BACKBONE IS CANARY-QWEN, AND 64-DIM ADAPTERS ARE USED. † DENOTES THE DEFAULT SETTING OF OUR PROPOSED METHOD.
<table><tr><td>Method</td><td>Loss for Stage 1</td><td>OGI</td><td>MyST</td><td>CORAAL</td></tr><tr><td>Vanilla Fine-tuning</td><td>一</td><td>10.95</td><td>8.37</td><td>8.87</td></tr><tr><td rowspan="2">EAVA (Ours)</td><td>CTC on Encoder</td><td>9.60</td><td>8.12</td><td>9.05</td></tr><tr><td>CE on LLM†</td><td>9.05</td><td>8.04</td><td>8.54</td></tr></table>

## E. Effect of Loss Function in the Encoder Awakening Stage

In the Encoder Awakening stage, the default supervision signal is the LLM Cross-Entropy (CE) loss for next-token prediction. We investigate whether alternative loss functions can also awaken the encoder adapters. Specifically, we attach a Connectionist Temporal Classification (CTC) [36] head directly to the encoder output and train the adapters using CTC ASR loss, bypassing the projector and LLM entirely during Stage 1. As shown in Table V, using CTC loss still outperforms vanilla fine-tuning on the OGI and MyST test sets. This observation suggests that the primary benefit comes from awakening the encoder adapters, while the choice of loss function mainly influences how this adaptation is achieved. Nevertheless, the default LLM CE loss achieves better overall performance, as it is consistent with the fine-tuning objective in Stage 2 (Continual Fine-tuning), leading to better alignment between the two stages.

## F. Generalizability to Other Speech-LLMs

To verify the generalizability of the proposed method, we conduct experiments on a second Speech-LLM, Phi-4- Multimodal [18], as shown in Table VI. The proposed method consistently outperforms all baselines across all three datasets, demonstrating that its effectiveness is not limited to a particular backbone architecture.

Compared with the Canary-Qwen results, Phi-4-Multimodal exhibits stronger baseline adaptation on OGI under vanilla fine-tuning, resulting in a smaller margin of improvement from EAVA. Nevertheless, similar performance trends are observed across both backbones, with EAVA consistently outperforming the corresponding baselines. These results suggest that the effectiveness of EAVA extends beyond a specific Speech-LLM architecture, although the magnitude of the gains varies across backbones and datasets.

TABLE VI  
WER (%) ON THREE TARGET-DOMAIN DATASETS USING PHI-4-MULTIMODAL AS THE BACKBONE WITH 64-DIM ADAPTERS. ALL EAVA RESULTS ARE STATISTICALLY SIGNIFICANT COMPARED TO BASELINES (p < 0.05).
<table><tr><td rowspan="2">Method</td><td colspan="2">OGI</td><td colspan="2">MyST</td><td colspan="2">CORAAL</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>Zero-shot</td><td>18.16</td><td>19.23</td><td>9.59</td><td>10.02</td><td>9.71</td><td>13.47</td></tr><tr><td>Vanilla Fine-tuning</td><td>9.29</td><td>9.92</td><td>7.46</td><td>8.31</td><td>6.24</td><td>9.22</td></tr><tr><td>Multi-stage Alignment</td><td>9.15</td><td>9.84</td><td>8.02</td><td>8.90</td><td>5.99</td><td>9.31</td></tr><tr><td>EAVA (Ours)</td><td>8.67</td><td>9.66</td><td>7.33</td><td>8.07</td><td>5.82</td><td>8.85</td></tr></table>

## VI. CONCLUSIONS

In this work, we proposed Encoder Awakening via Adapters (EAVA), a simple yet effective method for domain-adaptive fine-tuning of Speech-LLMs. Through Encoder Awakening, target-domain knowledge is effectively injected into the encoder via lightweight adapters while preserving the encoder’s pre-trained knowledge. The subsequent Continual Fine-tuning stage further adapts the entire model to the target domain. Experiments across multiple datasets and Speech-LLM backbones demonstrate that the proposed method consistently outperforms all baselines. Moreover, different adapter architectures prove effective within the EAVA framework, indicating that the general idea is the key factor rather than the specific adapter choice. We further find that larger adapters do not yield consistent improvements, likely because target-domain adaptation can be achieved with a relatively small number of trainable parameters. In addition, using mismatched data in the first stage can still awaken the encoder in most cases, indicating broader applicability of the proposed method. Furthermore, alternative supervision signals such as CTC loss can awaken the encoder adapters in the first stage in most cases, though the default LLM CE loss achieves better and more consistent overall performance. Overall, the proposed method achieves new state-of-the-art performance on all three domain-shifted datasets, surpassing even prior works that rely on additional unlabeled training data. In future work, we plan to explore more effective and efficient adaptation strategies for Speech-LLMs and other speech foundation models.

## VII. ACKNOWLEDGEMENTS

This research is supported in part by the National Science Foundation (NSF) and the Institute of Education Sciences (IES), U.S. Department of Education (DoE), through Grant R305C240046 to the U. at Buffalo. The opinions expressed are those of the authors and do not represent views of the IES, DoE, or the NSF.

## VIII. GENERATIVE AI USE DISCLOSURE

During the preparation of this work, the authors used Chat-GPT (GPT-5.5) for language editing, including proofreading and improving clarity and readability of the manuscript. All technical content, experimental design, results, and conclusions were produced and verified by the authors. After the use of Generative AI, the authors reviewed and edited the manuscript and take full responsibility for the content of the publication. Generative AI tools were not used to produce a significant portion of the manuscript and are not listed as authors.

## REFERENCES

[1] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12 449– 12 460, 2020.

[2] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM transactions on audio, speech, and language processing, vol. 29, pp. 3451–3460, 2021.

[3] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao et al., “Wavlm: Large-scale self-supervised pretraining for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[4] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural´ audio compression,” Trans. Mach. Learn. Res., vol. 2023, 2023.

[5] X. Zhang, D. Zhang, S. Li, Y. Zhou et al., “Speechtokenizer: Unified speech tokenizer for speech language models,” in ICLR. OpenReview.net, 2024.

[6] A. Grattafiori et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[7] M. Abdin et al., “Phi-4 technical report,” arXiv preprint arXiv:2412.08905, 2024.

[8] A. Yang et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[9] Y. Fathullah, C. Wu, E. Lakomkin, J. Jia, Y. Shangguan, K. Li, J. Guo, W. Xiong, J. Mahadeokar, O. Kalinli, C. Fuegen, and M. Seltzer, “Prompting large language models with speech recognition abilities,” in ICASSP. IEEE, 2024, pp. 13 351–13 355.

[10] Z. Ma, G. Yang, Y. Yang, Z. Gao, J. Wang, Z. Du, F. Yu, Q. Chen, S. Zheng, S. Zhang, and X. Chen, “An embarrassingly simple approach for llm with strong asr capacity,” arXiv preprint arXiv:2402.08846, 2024.

[11] M. Shi, Z. Jin, Y. Xu, Y. Xu, S. Zhang, K. Wei, Y. Shao, C. Zhang, and D. Yu, “Advancing multi-talker ASR performance with large language models,” in SLT. IEEE, 2024, pp. 14–21.

[12] S. Chen, C. Wang, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li et al., “Neural codec language models are zero-shot text to speech synthesizers,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 705–718, 2025.

[13] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, Z. Ma, and C. Zhang, “SALMONN: towards generic hearing abilities for large language models,” in ICLR. OpenReview.net, 2024.

[14] D. WANG, S. LIU, T. Zhang, Y. Chen, J. Li, and H. M. Meng, “Emotionthinker: Prosody-aware reinforcement learning for explainable speech emotion reasoning,” in ICLR. OpenReview.net, 2026.

[15] M. Shi, X. Xiao, R. Fan, S. Ling, and J. Li, “Train short, infer long: Speech-llm enables zero-shot streamable joint asr and diarization on long audio,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 17 442–17 446.

[16] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in ICLR. OpenReview.net, 2022.

[17] NVIDIA, “Canary-Qwen-2.5B,” https://huggingface.co/nvidia/ canary-qwen-2.5b, 2025, hugging Face model.

[18] Microsoft, “Phi-4-Multimodal-Instruct,” https://huggingface.co/ microsoft/Phi-4-multimodal-instruct, 2025, hugging Face model.

[19] K. Shobaki, J. Hosom, and R. A. Cole, “The OGI kids<sup>2</sup> speech corpus and recognizers,” in INTERSPEECH. ISCA, 2000, pp. 258–261.

[20] S. Pradhan, R. A. Cole, and W. H. Ward, “My science tutor (myst)-a large corpus of children’s conversational speech,” in LREC/COLING. ELRA and ICCL, 2024, pp. 12 040–12 045.

[21] T. Kendall and C. Farrington, “The Corpus of Regional African American Language,” Eugene, OR, 2023. [Online]. Available: https: //doi.org/10.7264/1ad5-6t35

[22] R. Fan, N. B. Shankar, and A. Alwan, “Benchmarking children’s ASR with supervised and self-supervised speech foundation models,” in INTERSPEECH. ISCA, 2024.

[23] A. Ying, N. B. Shankar, C.-J. Lin, M. Shi, P. Wang, H. jin Shim, S. Arora, H. V. hamme, A. Alwan, and S. Watanabe, “Benchmarking Training Paradigms, Dataset Composition, and Model Scaling for Child ASR in ESPnet,” in Workshop on Child Computer Interaction - WOCCI 2025, 2025, pp. 6–10.

[24] Z. Wang, N. B. Shankar, M. Shi, K. Zhang, and A. Alwan, “Gumbelbeard: Automatic layer selection for self-supervised adaptation of whisper in low-resource domains,” arXiv preprint arXiv:2606.11429, 2026.

[25] N. B. Shankar, R. Fan, and A. Alwan, “SOA: reducing domain mismatch in SSL pipeline by speech only adaptation for low resource ASR,” in ICASSP Workshops, 2024.

[26] B. Mu, Y. Shao, K. Wei, D. Yu, and L. Xie, “Efficient scaling for llmbased ASR,” in ASRU. IEEE, 2025, pp. 1–7.

[27] A. Gulati, J. Qin, C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution-augmented transformer for speech recognition,” in INTERSPEECH. ISCA, 2020, pp. 5036–5040.

[28] D. Rekesh, N. R. Koluguri, S. Kriman, S. Majumdar, V. Noroozi, H. Huang, O. Hrinchuk, K. C. Puvvada, A. Kumar, J. Balam, and B. Ginsburg, “Fast conformer with linearly scalable attention for efficient speech recognition,” in ASRU. IEEE, 2023, pp. 1–8.

[29] A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary, C. Chen et al., “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixtureof-loras,” arXiv preprint arXiv:2503.01743, 2025.

[30] A. Bapna and O. Firat, “Simple, scalable adaptation for neural machine translation,” in EMNLP/IJCNLP (1). Association for Computational Linguistics, 2019, pp. 1538–1548.

[31] P. Ramachandran, B. Zoph, and Q. V. Le, “Swish: a self-gated activation function,” arXiv preprint arXiv:1710.05941, vol. 7, no. 1, p. 5, 2017.

[32] N. B. Shankar, Z. Wang, K. Zhang, M. Shi, and A. Alwan, “Gc-lora: Gated convolutional lora for parameter-efficient acoustic adaptation,” arXiv preprint arXiv:2606.10464, 2026.

[33] A. A. Attia, J. Liu, W. Ai, D. Demszky, and C. Y. Espy-Wilson, “Kidwhisper: Towards bridging the performance gap in automatic speech recognition for children VS. adults,” in AIES (1). AAAI Press, 2024, pp. 74–80.

[34] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in International Conference on Machine Learning (ICML), 2019, pp. 3519–3529.

[35] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for NLP,” in International Conference on Machine Learning (ICML), 2019, pp. 2790–2799.

[36] A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber, “Connection-´ ist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,” in International Conference on Machine Learning (ICML), 2006, pp. 369–376.