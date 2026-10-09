# From Sparse Representations to Behavioral Insights for Multimodal Depression Assessment

Guimin Hu<sup>1</sup>, Zihao Song<sup>1</sup>, Jiachen Luo<sup>2</sup>, Jiayuan Xie<sup>3</sup>, Ruichu Cai<sup>1</sup>

<sup>1</sup>Guangdong University of Technology

<sup>2</sup>Queen Mary University of London

<sup>3</sup>The Hong Kong Polytechnic University rice.hu.x@gmail.com

## Abstract

Multimodal depression assessment offers a promising approach to analyzing behavioral patterns associated with depression. However, existing methods often rely on dense and opaque multimodal representations, making it difficult to interpret the behavioral patterns underlying their predictions. In this work, we introduce BehavDep, a sparse factor-based framework that decomposes multimodal behavioral representations into sparse latent factors and associates them with behaviorally meaningful concepts through a semantic bridge. To address the mismatch between user-level annotations and heterogeneous video-level behaviors, BehavDep further learns video-level depression tendency scores under weak supervision and aggregates information across multiple observations for user-level assessment. Extensive experiments demonstrate that BehavDep achieves the best overall assessment performance while revealing complementary modality contributions, heterogeneous behavioral patterns across observations, and prediction responses to concept-level editing. These results show that BehavDep provides a structured and interpretable approach to analyzing multimodal behavioral representations for depression assessment.

## 1 Introduction

Depression is a prevalent mental health condition characterized by persistent changes in mood, cognition, and behavior. Its assessment typically relies on clinical interviews and standardized questionnaires that evaluate symptoms reported by individuals or observed by clinicians. However, depressive symptoms can also manifest through observable behavioral patterns, such as reduced facial expressiveness, changes in speech dynamics, and altered verbal communication. These behavioral manifestations provide a potential basis for computational approaches to depression assessment from naturally occurring behavioral data.

Existing multimodal depression assessment methods primarily focus on predicting depressionrelated outcomes from multimodal behavioral signals. While these approaches have demonstrated promising predictive performance, they provide limited insight into how the underlying behavioral patterns are organized within the learned representations. A key challenge lies in the dense nature of multimodal representations. Different modalities and behavioral patterns are often encoded together in a shared latent space, making it difficult to disentangle and examine individual behavioral patterns and their associations with depression assessment. This motivates us to look beyond predictive performance and investigate the internal structure of multimodal behavioral representations.

Furthermore, depression-related behavioral patterns may vary across observations from the same individual. In real-world mental health monitoring, an individual’s behavioral data are often collected longitudinally, yet supervision is typically provided only at the user level. This granularity mismatch makes it difficult to characterize how behavioral patterns vary across individual observations and how such variation relates to depression assessment. To address these challenges, we propose BehavDep, a sparse factor-based framework that transforms multimodal behavioral representations into interpretable behavioral insights for depression assessment. BehavDep first decomposes dense multimodal behavioral representations into sparse latent factors without imposing predefined semantic meanings. These factors are subsequently associated with depression-related behavioral concepts, providing a semantic basis for characterizing the behavioral patterns captured by the learned representation. To account for the heterogeneity of behavioral patterns across observations, BehavDep further estimates video-level depression tendency scores under user-level weak supervision, enabling the analysis of behavioral variation across individual observations. Together, these components provide a structured approach to analyzing multimodal behavioral representations and the behavioral patterns associated with depression assessment

The main contributions of this work are summarized as follows:

• We introduce a sparse factor-based representation learning framework that decomposes multimodal behavioral representations into sparse latent factors without imposing predefined semantic meanings, enabling more explicit analysis of the underlying behavioral patterns.

• We ground the learned factors in clinically meaningful, depression-related behavioral concepts to provide a semantic interpretation of the factorized representation, while estimating video-level depression tendency scores to characterize heterogeneous behavioral patterns across individual observations.

• Extensive experiments demonstrate that BehavDep achieves state-of-the-art performance while enabling interpretable analysis of multimodal behavioral patterns and their variations across individual observations.

## 2 Related Work

## 2.1 Multimodal Depression Understanding

Depression assessment has been increasingly explored using multimodal behavioral signals from online platforms and video-based datasets, where facial, vocal, and linguistic cues provide complementary information for identifying depressive states (Pirina and Çöltekin, 2018; Yates et al., 2017; Tao et al., 2024; Zhou et al., 2022). Recent approaches leverage transformer-based fusion and cross-modal representation learning to capture complex behavioral patterns and achieve promising performance. However, these models typically rely on dense latent representations, making it difficult to disentangle and examine the behavioral patterns captured by different components of the representation. Furthermore, depression-related behavioral patterns may vary across observations from the same individual, posing additional challenges for analyzing heterogeneous patterns under user-level weak supervision.

## 2.2 AI for Mental Health Assessment

Artificial intelligence has been increasingly applied to mental health analysis, leveraging advances in representation learning and large-scale pretrained models (Holzinger et al., 2017; Sun et al., 2021; Moor et al., 2023). Recent studies explore knowledge-enhanced and instructiontuned language models for mental health reasoning. MentaLLaMA (Yang et al., 2024) introduced a large-scale mental health instruction dataset and adapted LLaMA2 (Touvron et al., 2023) for interpretable mental health tasks, while MentalLLM (Xu et al., 2024b) investigated prompting and finetuning strategies for mental health prediction. Despite these advances, existing AI-based mental health systems often rely on implicit multimodal representations, making it difficult to disentangle and interpret the behavioral patterns captured by the models. Recent approaches have explored the use of clinically meaningful concepts to improve the interpretability of learned representations, but the relationship between these concepts and the underlying multimodal behavioral representations remains relatively underexplored.

## 3 Method

## 3.1 Problem Formalization

Formally, let $u _ { i } = \{ v _ { 1 } , \ldots , v _ { N } \}$ denote the i-th user with N videos, where the j-th video is represented as $v _ { j } = \{ f _ { 1 } ^ { j } , f _ { 2 } ^ { j } , . . . , f _ { M } ^ { j } \}$ , consisting of M frames. The associated label $y _ { i }$ represents the mental state of user $u _ { i }$ . Since raw video and audio data are not directly processed by our model, we use preextracted visual and acoustic feature embeddings obtained from off-the-shelf feature extractors as the model inputs throughout training and inference.

## 3.2 BehavDep

Figure 1 presents an overview of BehavDep, a clinically informed framework for interpretable multimodal depression assessment. Given an input video, BehavDep extracts multimodal behavioral representations and factorizes them into sparse latent factors without predefined semantics. The factors are then associated with a depression-related behavioral concept set curated from clinical literature and descriptions of depression-related behavioral manifestations, establishing a semantic bridge between latent factors and observable behaviors (e.g., low eye contact, long pauses, and slow speech). The complete concept set and supporting references are provided in Appendix A. Under user-level weak supervision, BehavDep further estimates video-level depression tendency scores, enabling fine-grained analysis of heterogeneous behavioral patterns and user-level depression assessment.

![](images/5aef289fb0de702b1bd2295113b15ba35e56414430a682025824b4bfdc5f0535.jpg)  
Figure 1: Overview of BehavDep.

## 3.2.1 Sparse Behavioral Representation

Multimodal Behavioral Encoding. Given a user’s video collection, each video is represented as a sequence of pre-extracted multimodal framelevel features. To capture temporal dependencies in multimodal behavioral signals, we employ a temporal encoder that projects frame-level features into a shared latent space and processes them with a Transformer encoder. Masked pooling is then applied to obtain a compact video-level representation. The resulting representations capture temporal behavioral patterns and serve as inputs for subsequent sparse factor learning.

Sparse Latent Factor Discovery. Although multimodal behavioral encoders produce rich representations, these representations are typically dense and difficult to interpret at the component level. To obtain sparse latent factors, we pretrain a TopK Sparse Autoencoder (SAE) on the video-level representations. This factorization exposes structured latent patterns that facilitate subsequent analysis of behavioral characteristics relevant to depression assessment. Given the encoded video representation x, the SAE encodes sparse latent factors and reconstructs the input as:

$$
{ \bf z } = f _ { \mathrm { e n c } } ( { \bf x } ) , \quad \hat { \bf x } = f _ { \mathrm { d e c } } ( { \bf z } ) ,\tag{1}
$$

where TopK activation retains the top-ranked latent dimensions to encourage sparse representations. The SAE is optimized with reconstruction and sparsity objectives, and the pretrained encoder is subsequently used for behavioral concept association.

Behavioral Concept Association. Although the pretrained SAE discovers sparse latent factors, these factors remain semantically ambiguous, as individual dimensions do not have explicit semantic interpretations. To facilitate their interpretation, we construct a Depression-Related Behavioral Concept Set $\mathcal { C } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { K } \}$ , where each concept describes a behavioral pattern associated with depression, such as reduced facial expressiveness, monotonous speech, and social withdrawal.

Each behavioral concept is encoded into a semantic embedding using a pretrained text encoder:

$$
\mathbf { t } _ { k } = f _ { \mathrm { t e x t } } ( c _ { k } ) ,\tag{2}
$$

where $\mathbf { t } _ { k }$ denotes the semantic representation of the k-th behavioral concept. Given the sparse latent representation z obtained from the pretrained SAE, we project it into the same semantic space and measure its alignment with each concept embedding:

$$
s _ { k } = \cos ( g ( \mathbf { z } ) , \mathbf { t } _ { k } ) ,\tag{3}
$$

where $g ( \cdot )$ denotes a learnable projection function. The resulting concept activation vector s = $[ s _ { 1 } , \ldots , s _ { K } ]$ characterizes the association between the learned multimodal representation and the corresponding behavioral concepts for each video.

By associating the sparse latent factors of each video with behavioral concepts, BehavDep provides a semantic characterization of the learned factors and facilitates the analysis of behavioral patterns associated with depression.

## 3.2.2 Concept-Aware Depression Assessment

Concept Activation. After behavioral concept association, each video is represented by a concept activation vector s, where each dimension corresponds to a depression-related behavioral concept. Unlike implicit latent representations, the concept activations provide a semantic space for characterizing the behavioral patterns captured by each observation. Specifically, a higher activation score indicates a stronger association between the observation and the corresponding behavioral concept. The resulting concept representation provides an interpretable intermediate space for characterizing multimodal behavioral patterns and supporting subsequent depression assessment.

Since depression-related behavioral patterns may vary across observations, directly averaging videolevel representations may obscure differences in their contributions. To construct a user-level behavioral profile, we introduce a concept-aware aggregation strategy that adaptively weights videos according to their concept activation patterns.

Given the concept activation score of the j-th video s<sub>j</sub>, a learnable attention function estimates its relative contribution and then form concept aggregation across videos for user u:

$$
\gamma _ { j } = \frac { \exp ( h ( \mathbf { s } _ { j } ) ) } { \sum _ { i = 1 } ^ { N _ { u } } \exp ( h ( \mathbf { s } _ { i } ) ) } ,\tag{4}
$$

where $h ( \cdot )$ denotes a learnable scoring function. $N _ { u }$ denotes the video number of user u. The userlevel concept profile is then obtained by weighted aggregation:

$$
{ \bf p } _ { u } = \sum _ { j = 1 } ^ { N _ { u } } \gamma _ { j } { \bf s } _ { j } .\tag{5}
$$

The learned weights assign different contributions to individual observations based on their concept activation patterns, allowing the aggregation to emphasize more informative behavioral patterns while reducing the influence of less informative observations.

Concept-Guided Label Refinement. Although depression is assessed at the user level, depressionrelated behavioral patterns may vary across individual observations. Therefore, user-level annotations do not directly provide supervision for individual videos. To bridge this granularity gap, we derive video-level depression tendency scores from behavioral concept activations and learn them under user-level supervision.

For the j-th video, the concept activation vector ${ \bf s } _ { j }$ characterizes its associated behavioral patterns. $\mathbf { A }$ learnable mapping function generates a videolevel depression tendency score, hereafter referred to as the depression soft label:

$$
d _ { j } = g ( \mathbf { s } _ { j } ) ,\tag{6}
$$

where $d _ { j }$ denotes the estimated depression tendency of the j-th video. The video-level tendency scores are then aggregated according to their contribution weights to obtain a user-level tendency:

$$
d _ { u } ^ { * } = \sum _ { j = 1 } ^ { N _ { u } } \gamma _ { j } d _ { j } ,\tag{7}
$$

where $\gamma _ { j }$ denotes the contribution weight of the j-th video. The weights are estimated from the concept-aware representations and assign different contributions to individual observations, allowing more informative behavioral patterns to have greater influence on the user-level tendency.

The resulting user-level tendency is combined with the aggregated concept profile $\mathbf { p } _ { u }$ for depression assessment:

$$
{ \hat { y } } _ { u } = g ( [ d _ { u } ^ { * } ; \mathbf { p } _ { u } ] ) , \qquad { \hat { y } } _ { u } \in \{ 0 , 1 \} ,\tag{8}
$$

where [; ] denotes concatenation operation and 1 denotes the depressive class. The user-level prediction is supervised by the ground-truth user label $y _ { u } ,$ and the resulting loss provides weak supervision for learning the video-level tendency scores.

To obtain more stable video-level supervision, we further maintain a soft label bank that progressively refines the estimated tendency scores. Specifically, the newly estimated tendency scores are integrated with their historical values:

$$
\mathbf { y } ^ { t + 1 } = \alpha \mathbf { y } ^ { t } + \beta \mathbf { d } ^ { t } ,\tag{9}
$$

where $\mathbf { y } ^ { t }$ denotes the stored soft labels at iteration t, $\mathbf { d } ^ { t } = [ d _ { 1 } ^ { t } , \dots , d _ { N _ { u } } ^ { t } ]$ denotes the newly estimated video-level tendency scores, and $\alpha , \beta$ control the contributions of historical and current estimates. The label bank is initialized with a temporal prior and iteratively updated using the learned tendency scores, providing more stable video-level supervision under user-level weak labels.

Longitudinal Behavioral Pattern Analysis. The video-level tendency score $d _ { i }$ also provides a useful signal for examining behavioral patterns across a user $\mathrm { \mathit { \Omega } } ^ { \prime } \mathrm { \bar { s } }$ longitudinal video history. Videos with higher $d _ { i }$ indicate stronger associations with the depressive class. By examining these videos together with their behavioral concept activations ${ \bf s } _ { i } ,$ we further analyze the behavioral patterns associated with higher depression tendency and their variation across observations.

## 3.3 Experiments

## 3.3.1 Experimental Setting

Datasets. MUD3 (Wang et al., 2025) is a multimodal user-level depression detection dataset with longitudinal social media videos, split into training, validation, and test sets with an 8:1:1 ratio. DAIC-WOZ (Gratch et al., 2014) is a clinical interviewbased benchmark dataset providing synchronized audio, visual, and textual modalities with PHQ-8 scores. Detailed information is provided in the Appendix.

Implementation Details. All experiments are implemented in PyTorch and conducted on a single

<table><tr><td>Method</td><td>Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td></tr><tr><td> $\mathrm { D e p D e t e c t o r } _ { \mathrm { c a t } }$ </td><td>54.4</td><td>45.0</td><td>44.8</td><td>34.1</td></tr><tr><td> $\mathrm { D e p D e t e c t o r } _ { \mathrm { h a n } }$ </td><td>74.7</td><td>74.5</td><td>69.1</td><td>81.1</td></tr><tr><td> $\mathrm { T A M F N _ { c a t } }$ </td><td>64.4</td><td>57.2</td><td>59.8</td><td>54.7</td></tr><tr><td> $\mathrm { T A M F N _ { h a n } }$ </td><td>71.2</td><td>70.7</td><td>65.7</td><td>76.7</td></tr><tr><td> $\mathbf { B i L S T M _ { \mathrm { { c a t } } } }$ </td><td>73.6</td><td>71.1</td><td>71.5</td><td>72.2</td></tr><tr><td> $\mathbf { B i L S T M _ { \mathrm { h a n } } }$ </td><td>76.4</td><td>75.4</td><td>72.6</td><td>79.7</td></tr><tr><td> $\mathrm { T r a n s f o r m e r } _ { \mathrm { c a t } }$ </td><td>73.7</td><td>70.3</td><td>73.3</td><td>68.7</td></tr><tr><td> $\mathrm { T r a n s f o r m e r } _ { \mathrm { h a n } }$ </td><td>78.3</td><td>77.3</td><td>73.9</td><td>81.1</td></tr><tr><td>BehavDep</td><td> ${ \bf 8 3 . 3 \pm 0 . 3 }$ </td><td> ${ \bf 8 1 . 9 { \scriptstyle \pm 0 . 4 } }$ </td><td> $\mathbf { 8 0 . 6 \pm 0 . 2 }$ </td><td> ${ \bf 8 3 . 3 { \scriptstyle \pm 0 . 4 } }$ </td></tr></table>

Table 1: Performance comparison on the MUD3 dataset. Baseline results are adopted from (Wang et al., 2025). Results are reported as mean ± standard deviation over five independent runs.

<table><tr><td rowspan="2">Method</td><td colspan="2">Normal</td><td colspan="2">Depressed</td></tr><tr><td>Pre.</td><td>F1</td><td>Pre.</td><td>F1</td></tr><tr><td>ConvLSTM</td><td>83.1</td><td>35.1</td><td>30.1</td><td>45.2</td></tr><tr><td>PerceiverIO</td><td>74.3</td><td>81.5</td><td>50.0</td><td>36.3</td></tr><tr><td>AFABNet</td><td>77.4</td><td>83.6</td><td>62.2</td><td>45.8</td></tr><tr><td>Qwen2-Audio-Instruct</td><td>79.1</td><td>79.1</td><td>50.0</td><td>51.3</td></tr><tr><td>Mental-Perceiver</td><td>78.4</td><td>86.8</td><td>83.6</td><td>50.4</td></tr><tr><td colspan="5">BehavDep  $\mathbf { 8 4 . 2 \pm 0 . 2 8 7 . 5 \pm 0 . 2 8 5 . 6 \pm 0 . 2 7 4 . 6 \pm 0 . 3 }$ </td></tr></table>

Table 2: Performance comparison on the DAIC-WOZ dataset. Baseline results are reproduced from (Qin et al., 2025).

NVIDIA GeForce RTX 4080 GPU. Training is performed in two stages. The sparse autoencoder is first trained for 20 epochs using a batch size of 256 and a learning rate of $1 \times 1 0 ^ { - 4 }$ , followed by 30 epochs of end-to-end optimization with a batch size of 64 and the same learning rate. Early stopping based on validation performance is employed to avoid overfitting. Please see Appendix G for more implementation details.

## 3.4 Main Results

Result Analysis on MUD3 As shown in Table 1, BehavDep achieves the best performance across all evaluation metrics, improving accuracy and F1-score over the strongest baseline, Transformer<sub>han</sub>, by 5.05 and 4.62 percentage points, respectively. We attribute these gains to the factorized and concept-guided representation learned by BehavDep. Rather than directly modeling dense multimodal representations through sequential modeling or feature-level fusion, BehavDep decomposes them into sparse latent factors and characterizes these factors through behavioral concepts, allowing the model to focus on more discriminative behavioral patterns. Moreover, the learned videolevel depression tendencies provide finer-grained supervision for heterogeneous observations within the same user, complementing the user-level representation. The highest recall of 83.33% further suggests that these complementary signals help the model identify depressive cases more effectively.

![](images/520cd8b52e12bf32e51249326417f1b2f521f6e0ab75cc97e205e9c90d98be99.jpg)  
Figure 2: Temporal activation analysis of clinically grounded concepts for a depressed user. The heatmap illustrates the dynamic activation of depression-related behavioral concepts across video segments.

Result Analysis on DAIC-WOZ Table 2 compares BehavDep with state-of-the-art methods on the DAIC-WOZ dataset. BehavDep achieves the best performance for both Normal and Depressed samples, with F1-scores of 0.87 and 0.74, respectively. Compared with the strongest baseline, Mental-Perceiver, BehavDep improves the F1-score by 1.2% for Normal samples and 24.0% for Depressed samples. These results demonstrate that the sparse factor-based representation can effectively capture discriminative behavioral patterns for depression assessment while providing a structured basis for their subsequent analysis.

## 3.5 Behavioral Concept Analysis

As shown in Figure 2, BehavDep identifies several highly activated behavioral concepts, including low pitch variability, low eye contact, soft speaking, and reduced expressiveness. These concepts are consistent with behavioral manifestations commonly associated with depression, providing a semantic interpretation of the learned sparse factors. The observed factor–concept associations indicate that the sparse representation captures structured behavioral patterns rather than remaining as uninterpreted latent dimensions.

The temporal activation patterns further reveal heterogeneous contribution dynamics across video segments. Concepts such as low pitch variability and low eye contact maintain relatively high activation across observations, whereas speech latency, reduced blinking, and gaze instability exhibit greater temporal variation. This suggests that different behavioral manifestations may emerge with different temporal patterns, with some expressed more consistently and others appearing intermittently across observations. Moreover, the complementary activation of multiple concepts indicates that the model captures diverse acoustic and visual behavioral patterns rather than relying on a single behavioral cue. Overall, these findings show that the learned sparse factors can be associated with semantically meaningful behavioral concepts while preserving the heterogeneous and temporally varying nature of multimodal behaviors.

![](images/3e5528deff93569a2a352d061a6c3ff5f3abc5987cb864ee6d777aee9dc2befd.jpg)

![](images/e3b85b4289ea8dff18775f69bce90e5c9836594bee99ef929b568f4226bfea1f.jpg)  
(b) Depressed User

(a) Depressed User  
![](images/3425cbb7713bb864c5c20315a5c547891accacfff63839cfcec604069ecdab50.jpg)  
(c) Non-Depressed User

![](images/fa501817ec0bfbcbe60a2bc3277d386812904ca39c985c253a65888be8bf9426.jpg)  
(d) Non-Depressed User

Figure 3: Video-level prediction trajectory. Dashed line denotes the threshold for depressive symptoms.  
![](images/1af8fcdd5dd57267d90307f489c52cf48aad2ce9cf822a2b7b194b3815085d8b.jpg)  
Figure 4: Representative clinical prediction reports generated by BehavDep.

<table><tr><td>Method</td><td>Acc.</td><td>F1</td><td>Pre.</td><td>Rec.</td></tr><tr><td>BehavDep</td><td>83.3</td><td>81.9</td><td>80.6</td><td>83.3</td></tr><tr><td>w/o SAE</td><td>77.2</td><td>76.5</td><td>71.4</td><td>78.1</td></tr><tr><td>w/o Audio</td><td>80.3</td><td>79.4</td><td>73.1</td><td>80.7</td></tr><tr><td>w/o Video</td><td>78.6</td><td>78.1</td><td>72.8</td><td>78.5</td></tr><tr><td>w/o Concept</td><td>77.3</td><td>76.4</td><td>71.9</td><td>78.5</td></tr><tr><td>w/o Audio Concept</td><td>78.7</td><td>77.6</td><td>72.1</td><td>78.9</td></tr><tr><td>w/o Video Concept</td><td>78.3</td><td>76.6</td><td>71.6</td><td>78.4</td></tr></table>

Table 3: Ablation study on different components of BehavDep.

## 3.6 Behavioral Pattern Analysis

Beyond user-level depression assessment, the video-level tendency score $d _ { i }$ provides an additional signal for examining how predictions vary across a user’s longitudinal observations. Figure 3 presents representative trajectories for two users with depression and two non-depressed users. The calculation of threshold for depressive symptoms is provided in Appendix G. The variation in scores across observations indicates that the model does not assign a uniform depression tendency within a user, but instead captures heterogeneous prediction patterns across their video history.

To further characterize these observation-level differences, Figure 4 presents a representative prediction case together with its associated behavioral concepts with the score. Both acoustic and visual concepts are activated, including speech latency, long pauses, and reduced blinking, illustrating how the factorized representation can be semantically characterized in terms of multimodal behavioral patterns. Overall, these results highlight two complementary capabilities of BehavDep: the video-level tendency score captures variation across observations, while factor-to-concept associations characterize the behavioral patterns underlying this variation. As the video-level scores are indirectly learned from user-level supervision, we regard this analysis as an exploratory characterization of behavioral heterogeneity rather than a definitive temporal characterization of depressive states.

## 3.7 Ablation Study

Table 3 reports the impact of different components in BehavDep. Removing the SAE substantially reduces F1-score from 81.97% to 76.52%, demonstrating the importance of sparse factorization for capturing structured behavioral patterns. Removing concept guidance yields a similar drop to 76.35%, highlighting the contribution of semantic guidance for interpreting depression-related behaviors. Modality ablations further demonstrate the complementary roles of acoustic and visual cues, as removing either modality consistently degrades performance. Removing modality-specific concepts also causes additional performance drops, suggesting that concept guidance captures complementary behavioral indicators across modalities. These results support the contributions of sparse factorization, concept guidance, and multimodal behavioral cues to BehavDep.

<table><tr><td>Edited Concepts</td><td>Modality</td><td>Total Impact ∆F1 (%)</td><td></td><td>Label Flip</td></tr><tr><td>Slow speech (c1)</td><td>A</td><td>0.65</td><td>-0.11</td><td>Depressive → Depressive</td></tr><tr><td>Slow speech  $\dot { ( c _ { 1 } ) }$ </td><td>A</td><td>1.14</td><td>-0.25</td><td>Depressive → Depressive</td></tr><tr><td>Long pauses (c2) Flat facial affect (c6)</td><td>V</td><td>0.72</td><td>-0.27</td><td>Depressive → Depressive</td></tr><tr><td>Flat facial affect  $( c _ { 6 } )$  Reduced facial movement (c7)</td><td>V</td><td>1.36</td><td>-1.84</td><td>Depressive → Normal</td></tr><tr><td>Slow speech  $( c _ { 1 } ) ,$  Long pauses  $\left( c _ { 2 } \right)$  Flat facial affect (c6), Reduced facial movement (c7)</td><td>A,V</td><td>3.53</td><td>-3.35</td><td>Depressive → Normal</td></tr><tr><td>Low eye contact  $\left( c _ { 8 } \right)$  Long pauses  $( { \dot { c } } _ { 2 } ) ,$  Flat facial affect  $\ddot { ( c _ { 6 } ) }$ </td><td> $\mathsf { A } , \mathsf { V } , \mathsf { S }$ </td><td>3.81</td><td>-4.58</td><td>Depressive → Normal</td></tr></table>

Table 4: Counterfactual editing effects of clinically meaningful behavioral concepts. A, V, and S denote the audio, visual, and cross-modal concept categories, respectively.

![](images/7edcfc90468ae27ade868c0138f4cadcf1b402c1a14a408523ea5a9e45434853.jpg)  
(a)

![](images/1ca330de1450c531ddb8048e076415b50005d6b870ad95e39866b65947c0f526.jpg)  
(b)  
Figure 5: (a) Contribution analysis of different modalities and (b) acoustic feature importance

## 4 Behavioral Concept Editing Analysis

Counterfactual concept editing examines how changes in individual behavioral concepts affect model predictions. As shown in Table 4, editing a single concept generally produces only a modest reduction in the predicted depression tendency and rarely changes the final outcome. This suggests that no single behavioral cue dominates the assessment, and that the model can maintain its prediction when only one aspect of the behavioral profile is altered. In contrast, jointly editing multiple concepts across different behavioral dimensions produces substantially larger changes and consistently shifts the prediction from Depressive to Normal. The progressively larger prediction changes with additional concept edits indicate that the model responds to the accumulation of behavioral evidence rather than to an isolated cue. Notably, the edited concepts span different behavioral dimensions, such as acoustic and visual patterns, suggesting that their effects can accumulate across modalities and provide complementary evidence for the final prediction.

This contrast between isolated and joint editing provides insight into how the concept-aware representation supports depression assessment. Individual behavioral manifestations may provide only partial evidence, whereas coordinated changes across multiple concepts can substantially alter the overall behavioral profile and, consequently, the model’s prediction. Thus, the counterfactual results suggest that BehavDep captures depression-related behavioral patterns as a combination of complementary concepts, rather than encoding them as a response to a single dominant behavioral indicator.

## 4.1 Modal Contribution Analysis

To investigate the underlying decision mechanism of the proposed model, we analyze both modalitylevel contribution and feature-level importance. As shown in Figure 5, the model primarily relies on acoustic information, with audio contributing the largest proportion among all modalities. This dominance is further supported by the feature importance analysis, where speech-related patterns such as long pauses and reduced speech dynamics are identified as the most influential cues. These results indicate that the model effectively captures behavioral characteristics reflected in temporal speaking patterns and vocal expressiveness, rather than relying on superficial correlations. Meanwhile, the visual modality provides complementary information, suggesting that multimodal integration helps enrich the representation of the target behavior. Overall, the consistency between modality-level contribution and feature-level importance demonstrates that the model learns meaningful and interpretable cues aligned with domain knowledge.

![](images/e61313b7ad383f852fb853f4ba159bd60967d96e5f0115639cececec34a8a1be.jpg)  
Figure 6: User-level case study involving three participants: User A is non-depressed, while Users B and C are diagnosed with depression.

## 4.2 User Study

To examine the interpretability of the learned concept representations, we conduct a user-level case study involving three participants: User A is nondepressed, while Users B and C are diagnosed with depression. Figure 6 visualizes concept importance (left) and concept activations across multiple videos (right). Concept importance jointly considers model attention and the semantic alignment between each concept embedding and the learned sparse representation (see Appendix for details).

User A shows consistently low concept activations across videos, with no prominent depressionrelated concepts. In contrast, Users B and C exhibit relatively consistent activations across multiple videos, while their concept profiles differ substantially despite sharing the same depression label. User B shows concentrated importance on several concepts, whereas User C exhibits a more distributed pattern. These observations illustrate both cross-video consistency and behavioral heterogeneity across users. Overall, the case study shows that concept representations expose fine-grained behavioral patterns underlying the model’s depression assessment.

## 5 Conclusion

In this work, we presented BehavDep, a sparse factor-based framework for interpretable multimodal depression assessment. BehavDep decomposes dense multimodal behavioral representations into sparse latent factors and connects them with behaviorally meaningful concepts, enabling finegrained characterization under coarse user-level supervision. Extensive experiments demonstrate improved assessment performance and reveal complementary modality contributions, heterogeneous and temporally varying behavioral patterns, and prediction responses to concept-level editing. Overall, BehavDep provides a structured approach to transforming opaque multimodal representations into interpretable behavioral evidence for depression assessment.

## Limitations

The longitudinal behavioral analysis should be interpreted as an exploratory analysis rather than a clinically validated characterization of depressionrelated behavior. The video-level tendency scores $d _ { i }$ are indirectly learned from user-level supervision and therefore may be affected by the distribution of observations within each user’s video history. Similarly, the associated behavioral concepts provide a semantic characterization of the learned representations but do not constitute clinically verified behavioral indicators. Future work could incorporate finer-grained annotations and expert evaluation to further validate the relationship between the identified behavioral patterns and depressionrelated assessments. BehavDep uses sparse factors and behaviorally meaningful concepts to provide interpretable descriptions of the behavioral patterns underlying its predictions. These model-generated characterizations are intended to provide a reference for fine-grained behavioral annotation, potentially alleviating the cost and difficulty of manually annotating behavioral patterns at the video level, rather than serving as clinical conclusions.

## Ethical Considerations

This study involves multimodal behavioral data for depression assessment, raising ethical considerations regarding privacy and potential misuse. All data are collected and processed in accordance with applicable ethical guidelines, with personally identifiable information removed or anonymized before analysis.

BehavDep is designed as an assistive research framework for analyzing multimodal behavioral patterns, rather than a clinical diagnostic system. The video-level depression tendency learned under weak supervision should not be interpreted as a clinical diagnosis or definitive measure of an individual’s mental health status, and model outputs should not replace professional assessment.

We acknowledge potential risks related to algorithmic bias, model uncertainty, and misinterpretation. Any practical deployment should incorporate appropriate human oversight and comply with relevant ethical standards, with particular attention to privacy, fairness, informed consent, and the consequences of erroneous predictions.

## References

Shaykhah A Almaghrabi, Scott R Clark, and Mathias Baumert. 2023. Bio-acoustic features of depression: A review. Biomedical Signal Processing and Control, 85:105020.

Djamila Bennabi, Pierre Vandel, Charalambos Papaxanthis, Thierry Pozzo, and Emmanuel Haffen. 2013. Psychomotor retardation in depression: a systematic review of diagnostic, pathophysiologic, and therapeutic implications. BioMed research international, 2013(1):158746.

Jeylan S Buyukdura, Shawn M McClintock, and Paul E Croarkin. 2011. Psychomotor retardation in depression: biological underpinnings, measurement, and treatment. Progress in Neuro-Psychopharmacology and Biological Psychiatry, 35(2):395–409.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. 2024. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759.

Fifth Edition et al. 2013. Diagnostic and statistical manual of mental disorders. Am Psychiatric Assoc, 21(21):591–643.

Jonathan Gratch, Ron Artstein, Gale M Lucas, Giota Stratou, Stefan Scherer, Angela Nazarian, Rachel Wood, Jill Boberg, David DeVault, Stacy Marsella, et al. 2014. The distress analysis interview corpus of human and computer interviews. In Lrec, volume 14, pages 3123–3128. Reykjavik.

Andreas Holzinger, Chris Biemann, Constantinos S Pattichis, and Douglas B Kell. 2017. What do we need to build explainable ai systems for the medical domain? arXiv preprint arXiv:1712.09923.

Zhiheng Huang, Wei Xu, and Kai Yu. 2015. Bidirectional lstm-crf models for sequence tagging. arXiv preprint arXiv:1508.01991.

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. 2021. Perceiver io: A general architecture for structured inputs & outputs. arXiv preprint arXiv:2107.14795.

Kurt Kroenke, Robert L Spitzer, and Janet BW Williams. 2001. The phq-9: validity of a brief depression severity measure. Journal of general internal medicine, 16(9):606–613.

Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. 2023. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265.

Aude Paquet, Aurélie Lacroix, Benjamin Calvet, and Murielle Girard. 2022. Psychomotor semiology in depression: a standardized clinical psychomotor approach. BMC psychiatry, 22(1):474.

Inna Pirina and Çagri Çöltekin. 2018. Identifying depression on reddit: The effect of training data. In Proceedings ofthe 2018 EMNLP Workshop SMM4H: The 3rd Social Media Miningfor Health Applications Workshop & Shared Task, SMM4H@EMNLP 2018, Brussels, Belgium, October 31, 2018, pages 9–12. Association for Computational Linguistics.

Jinghui Qin, Changsong Liu, Tianchi Tang, Dahuang Liu, Minghao Wang, Qianying Huang, and Rumin Zhang. 2025. Mental-perceiver: audio-textual multimodal learning for estimating mental disorders. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 25029–25037.

Hao Sun, Zhenru Lin, Chujie Zheng, Siyang Liu, and Minlie Huang. 2021. Psyqa: A chinese dataset for generating long counseling text for mental health support. In Findings ofthe associationfor computational linguistics: ACL-IJCNLP 2021, pages 1489– 1503.

Yongfeng Tao, Minqiang Yang, Huiru Li, Yushan Wu, and Bin Hu. 2024. Depmstat: Multimodal spatiotemporal attentional transformer for depression detection. IEEE Trans. Knowl. Data Eng., 36(7):2956– 2966.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Bichen Wang, Yixin Sun, Yanyan Zhao, and Bing Qin. 2025. Beyond snapshots: A multimodal user-level dataset for depression detection in dynamic social media streams. In Proceedings of the 33rd ACM International Conference on Multimedia, MM 2025, Dublin, Ireland, October 27-31, 2025, pages 12912– 12918. ACM.

Ping-Cheng Wei, Kunyu Peng, Alina Roitberg, Kailun Yang, Jiaming Zhang, and Rainer Stiefelhagen. 2022. Multi-modal depression estimation based on subattentional fusion. In European Conference on Computer Vision, pages 623–639. Springer.

Xiao Xu, Yang Wang, Xinru Wei, Fei Wang, and Xizhe Zhang. 2024a. Attention-based acoustic feature fusion network for depression detection. Neurocomputing, 601:128209.

Xuhai Xu, Bingsheng Yao, Yuanzhe Dong, Saadia Gabriel, Hong Yu, James Hendler, Marzyeh Ghassemi, Anind K Dey, and Dakuo Wang. 2024b. Mental-llm: Leveraging large language models for mental health prediction via online text data. Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies, 8(1):1–32.

Kailai Yang, Tianlin Zhang, Ziyan Kuang, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. 2024. Mentallama: interpretable mental health analysis on social media with large language models. In Proceedings ofthe ACM Web Conference 2024, pages 4489–4500.

Andrew Yates, Arman Cohan, and Nazli Goharian. 2017. Depression and self-harm risk assessment in online forums. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017, pages 2968–2978. Association for Computational Linguistics.

Jeewoo Yoon, Chaewon Kang, Seungbae Kim, and Jinyoung Han. 2022. D-vlog: Multimodal vlog dataset for depression detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 12226–12234.

Li Zhou, Zhenyu Liu, Zixuan Shangguan, Xiaoyan Yuan, Yutong Li, and Bin Hu. 2022. Tamfn: Timeaware attention multimodal fusion network for depression detection. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31:669–679.

## A Depression Concept Set

To establish a clinically grounded representation for depression assessment, the proposed Depression Concept Set is constructed based on clinically recognized depressive symptoms described in the DSM-5 (Edition et al., 2013; Buyukdura et al., 2011). We summarize representative behavioral concepts across modalities by reviewing studies on depressive symptoms, affective behaviors, and observable depression markers.

The concept bank is organized into three categories: acoustic concepts, visual concepts, and cross-modal concepts. The acoustic concepts are derived from psychological evidence regarding speech-related changes in depression, capturing characteristics such as slowed speech, prolonged pauses, reduced vocal energy, monotonic prosody, decreased speech fluency, and diminished verbal engagement. These concepts reflect alterations in speech production, emotional expressiveness, and communicative motivation frequently reported in individuals with depressive symptoms. The visual concepts are constructed based on documented nonverbal behavioral indicators associated with depression, including reduced facial expressiveness, limited eye contact, downward gaze, decreased facial movement, slumped posture, and psychomotor slowing. These concepts represent changes in emotional display, social interaction, and behavioral activation that can be observed through facial and body cues. In addition, we introduce cross-modal

<table><tr><td>Concept Bank for Depression Understand- ing</td><td>B Baselines</td></tr><tr><td rowspan="5">Acoustic Concepts: slow speech, long pauses, reduced speech, low vocal energy, monotone speech, low pitch vari- ability, hesitant speaking, breathy voice, weak articulation, speech irregularity, soft speak- ing, reduced verbal output, speech latency, slowed response timing, voice instability, re- duced speaking engagement, flat vocal affect, low speech dynamics, fatigued voice Visual Concepts: flat facial affect, reduced facial movement, low eye contact, reduced blinking, downward gaze, head lowering, reduced movement, fa- cial tension, emotion suppression, reduced ex- pressiveness, gaze instability, slumped pos- ture, psychomotor slowing, low behavioral ac- tivation, facial asymmetry, micro-expression reduction, reduced smiling, avoidant gaze, vi- sual disengagement, fatigue appearance Cross-modal Concepts:</td><td>To comprehensively evaluate BehavDep, we com- pare it with representative baselines, covering tra- ditional machine learning approaches, multimodal neural architectures, and recent state-of-the-art de-</td></tr><tr><td>pression assessment models: • Dep (Yoon et al., 2022) a Transformer-based model that extracts unimodal temporal fea- tures and aligns modalities through cross-</td></tr><tr><td>attention and concatenation. • TAMFN (Zhou et al., 2022) a multimodal model that performs early feature fusion by</td></tr><tr><td>integrating intermediate unimodal representa- tions through temporal attention-based cross- modal fusion. • BiLSTM (Huang et al., 2015) a bidirectional</td></tr><tr><td>LSTM-based model that jointly processes two modalities and aggregates temporal informa- tion through an attention mechanism. • Trans (Vaswani et al., 2017) a Transformer-</td></tr><tr><td>psychomotor retardation, emotional flattening, behavioral withdrawal, social disengagement, cognitive fatigue, low interpersonal engage- ment Figure 7: Structured concept bank.</td><td>based model that captures temporal depen- dencies by aggregating information across the temporal dimension • ConvLSTM (Wei et al., 2022) a convolutional</td></tr><tr><td>concepts to capture high-level depressive behav- iors that typically emerge from the integration of multiple behavioral channels, such as psychomo- tor retardation, emotional flattening, behavioral withdrawal, and social disengagement. These con- cepts provide a more comprehensive description</td><td>bidirectional LSTM with a sub-attention mechanism for modeling heterogeneous in- formation.</td></tr><tr><td rowspan="2">of depression-related patterns beyond individual modality-specific observations. By grounding the concept bank in established psychological findings, we aim to bridge low-level multimodal signals and high-level clinical interpretations. This literature- guided design enables the model to perform de- pression understanding through semantically mean- ingful concepts, improving interpretability while maintaining alignment with psychological knowl- edge. Table 5 presents representative clinical con- cepts along with their corresponding supporting</td><td>• PerceiverIO (Jaegle et al., 2021) a general- purpose architecture for multimodal learning with a fully attention-based design and a flexi- ble querying mechanism.</td></tr><tr><td>• AFABNet (Xu et al., 2024a) an attention- based acoustic feature fusion network that combines four complementary acoustic fea- tures for depression detection. • Qwen2-Audio-Instruct (Chu et al., 2024) a 7B audio-language model based on Qwen</td></tr></table>

<table><tr><td>Clinical Concept</td><td>Clinical Basis</td><td>Observable Manifestation</td></tr><tr><td>Low eye contact</td><td>Reduced social engagement Reduced gaze toward the interlocutor (Buyukdura et al., 2011)</td><td></td></tr><tr><td>Flat facial affect</td><td>Diminished emotional expression Limited facial and emotional expression (Buyukdura et al., 2011)</td><td></td></tr><tr><td>Long pauses</td><td>Psychomotor slowing (Bennabi Prolonged pauses or delayed responses et al., 2013)</td><td></td></tr><tr><td>Monotone speech</td><td>Diminished emotional expression Reduced pitch and prosodic variation (Bennabi et al., 2013)</td><td></td></tr><tr><td>Soft speaking</td><td>Reduced energy (Almaghrabi et al., Low vocal intensity and speaking volume 2023)</td><td></td></tr><tr><td>Slow speech</td><td>Psychomotor slowing (Buyukdura Reduced speech rate et al., 2011)</td><td></td></tr><tr><td></td><td>et al., 2022)</td><td>Reduced movement Psychomotor retardation (Paquet Limited head, facial, or upper-body movement</td></tr><tr><td>Reduced speech</td><td>maghrabi et al., 2023)</td><td>Reduced verbal activity (Al- Reduced amount of verbal output or shorter utterances</td></tr></table>

Table 5: Behavioral concepts, their literature bases, and corresponding observable manifestations used in BehavDep.

![](images/972436c3c50a252283f5d4d330cf2d508a6de1a3138110a303ac0e7e1dcd3251.jpg)  
Figure 8: Distribution of Depression Evidence.

## C Observation-level Behavioral Heterogeneity

Figure 8 shows the distribution of the learned videolevel depression tendency scores across observations from users with depression. The scores exhibit a highly uneven distribution: 58.5% of the 3,192 video segments receive scores below 0.1, while only 14.1% receive scores above 0.5. This indicates that depression-related tendencies are not uniformly distributed across a user’s observations, with stronger tendencies concentrated in a relatively small subset of segments.

This observation supports the need for videolevel tendency estimation under user-level supervision. Rather than assuming that all observations from a depressed user contribute equally to the assessment, BehavDep allows individual observations to exhibit different levels of depression tendency. Combined with the associated behavioral concepts, these scores further provide a basis for characterizing the behavioral patterns represented in observations with different depression tendencies.

## D Intra-User Variability of Depression Tendency

To characterize the variability of depression-related signals across observations, we visualize the videolevel depression soft-label trajectories of representative non-depressed and depressed users. Each point denotes the predicted depression tendency of an individual video. As shown in Figure ??, the predicted scores vary substantially across videos within the same user, rather than remaining at a consistent level. This variation indicates that depression-related behavioral signals can differ considerably across observations, suggesting that a single video may not fully characterize a user’s overall mental state.

These observations motivate the use of user-level aggregation to integrate multiple heterogeneous video instances rather than relying on individual video predictions. They also highlight the importance of modeling intra-user variability when learning video-level depression tendencies, as different observations from the same user may exhibit substantially different levels of depression-related behavioral evidence.

## E Computation of Concept Importance

To quantify the contribution of each clinically grounded concept, we define concept importance by integrating two complementary signals: (1) the model attention indicating how much the prediction relies on each concept, and (2) the semantic alignment score measuring how well the learned sparse representation encodes the corresponding clinical concept.

![](images/3ad682ad6e841ee23bd3d5ed6cb10fa1f40c87a667ea89a14ba624fc095df069.jpg)

![](images/138a3170d1dcfb1f0f2fdd230c9774ae796dc7bdd17714df28ab9fe7879a2f3e.jpg)  
(a) Non-Depressed User

![](images/a881615cd5eb39a2e2b9897d82546d1ca79cf3348b4ef2af856001ca334b4de4.jpg)

![](images/4de6dc47f50e1b66a114c1ab566258e832180b79403333f2b2ba0c8eb6909f33.jpg)  
(c) Depressed User

(b) Depressed User  
![](images/854e225c41b556f56e79453a70a87c7ac906173c6269de039487e20ce52a27ec.jpg)  
(d) Non-Depressed User  
Figure 9: Temporal prediction trajectory of BehavDep.

Given the sparse representation z and concept embedding $\mathbf { c } _ { k } .$ , the semantic alignment score for the k-th concept is computed as:

$$
s _ { k } = \frac { \mathbf { z } ^ { T } \mathbf { c } _ { k } } { \Vert \mathbf { z } \Vert _ { 2 } \Vert \mathbf { c } _ { k } \Vert _ { 2 } } ,\tag{10}
$$

where $s _ { k }$ measures the correspondence between the learned sparse feature and the clinical concept. Meanwhile, the attention weight $\alpha _ { k }$ represents the model’s reliance on the k-th concept during depression assessment. The final concept importance is obtained by combining these two factors:

$$
I _ { k } = { \boldsymbol { \alpha } } _ { k } \cdot { \boldsymbol { s } } _ { k } ,\tag{11}
$$

where $I _ { k }$ denotes the importance of the k-th concept, reflecting its overall contribution to the depression prediction.

## F Patient Health Questionnaire-9 (PHQ-9)

The Patient Health Questionnaire-9 (PHQ-9) is one of the most widely adopted clinical instruments for depression assessment and is frequently used as the primary supervision signal in multimodal depression recognition studies. Introduced as a self-administered screening tool aligned with DSM diagnostic criteria, PHQ-9 evaluates depressive symptoms through nine items, each scored from 0 to 3, producing a total score ranging from 0 to 27<sup>1</sup>. While PHQ-9 provides a validated measure of depression severity, its aggregated score alone does not reveal the behavioral evidence underlying

Figure 10: Representative behavioral insight report generated by BehavDep, illustrating how sparse multimodal behavioral factors are translated into interpretable evidence for depression assessment.

the assessment. In clinical practice, mental health professionals interpret PHQ-9 scores together with observable psychological indicators, including affective expression, speech characteristics, cognitive responses, and social behaviors. This limitation motivates computational models that not only achieve accurate PHQ-9-based depression assessment but also provide interpretable behavioral evidence to support clinical decision-making.

## G Implementation Details

Following the standard Patient Health Questionnaire-9 (PHQ-9) protocol (Kroenke et al., 2001), subjects are classified as depressed or non-depressed using the clinical diagnostic threshold. Details of the preprocessing pipeline and label assignment are provided in the Appendix. Unless otherwise specified, identical training protocols and hyperparameter settings are adopted across all experiments.

PHQ-9-Based Clinical Threshold The Patient Health Questionnaire-9 (PHQ-9) score ranges from 0 to 27, where higher scores indicate more severe depressive symptoms. To transform the original PHQ-9 score into a normalized value between 0 and 1, a min-max normalization method is applied:

$$
X _ { n o r m } = { \frac { X _ { P H Q 9 } } { 2 7 } }\tag{12}
$$

where $X _ { P H Q 9 }$ represents the original PHQ-9 total score and $( X _ { n o r m } )$ represents the normalized depression severity score. After normalization, a score of 0 corresponds to the minimum symptom level, while a score of 1 corresponds to the maximum possible PHQ-9 severity.

For depression screening, a commonly used clinical cutoff is a PHQ-9 score of 10. Therefore, the normalized threshold can be calculated as:

$$
T h r e s h o l d = \frac { 1 0 } { 2 7 } = 0 . 3 7 0\tag{13}
$$

Accordingly, participants are classified into two groups based on the normalized score:

$$
Y = \left\{ \begin{array} { l l } { 1 , } & { X _ { n o r m } \ge 0 . 3 7 0 } \\ { 0 , } & { X _ { n o r m } < 0 . 3 7 0 } \end{array} \right.\tag{14}
$$

where (Y=1) indicates a positive depression screening result (potential depressive symptoms), and (Y=0) indicates a negative screening result. We set the threshold at 0.370 as the clinical diagnostic threshold for depression soft labels.

For machine learning applications, the normalized PHQ-9 score can be used as a continuous severity indicator, while the binary label generated using the cutoff value can be used as the classification target. It should be noted that the normalized score represents symptom severity rather than a direct probability of having depression.

## H Case Study

To further examine the practical interpretability of BehavDep, Figure 10 presents an example prediction report generated by the proposed framework. The report summarizes the model prediction together with the most influential behavioral concepts and their corresponding behavioral evidence. Two mental health professionals independently reviewed the extracted evidence and found that it was well aligned with the observed user behaviors.

In this case, BehavDep predicts a depressive state with a probability of 60.5%, primarily supported by acoustic concepts. In particular, speech latency, long pauses, and reduced speech fluency are identified as salient evidence, which are summarized as behavioral patterns related to reduced speech energy and psychomotor slowing. Rather than providing only an overall prediction, BehavDep traces the prediction to specific and observable behavioral concepts, making the model output easier to interpret and inspect.

Overall, the case study shows how BehavDep can translate multimodal representations into human-readable behavioral evidence, providing a more transparent view of the patterns underlying individual predictions while leaving the final assessment to human professionals.