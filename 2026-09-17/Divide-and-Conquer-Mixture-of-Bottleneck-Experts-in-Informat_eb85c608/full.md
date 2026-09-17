# Divide and Conquer: Mixture-of-Bottleneck Experts in Informative Ordinal Space for Video-based Multimodal Sentiment Analysis

Ronghao Lin <sup>˙ID</sup> , Qiaolin He, Zefeng Lu <sup>˙ID</sup> , Yichu Liu, Li Huang, Sijie Mai <sup>˙ID</sup> , Haifeng Hu <sup>˙ID</sup> , Member, IEEE ,

and Yap-peng Tan <sup>˙ID</sup> , Fellow, IEEE

Abstract—Video-based Multimodal sentiment analysis (MSA) must handle information from text, audio, and image sequence in human speaking videos, yet current methods often fail to integrate modalities with task awareness. Most models treat video sentiment prediction as a single task, overlooking its ordinal nature, and their fusion strategies struggle to capture diverse unique and synergic cues across modalities. To address these limitations, we adopt a divide-and-conquer perspective by reformulating MSA as an ordinal regression problem and decoupling it into polarity recognition and intensity prediction. Driven by information theory, we introduce a Mixture-of-Bottleneck (MoB) framework that assigns different latents to polarityand intensity-specific experts for different modalities. With the learning of information bottleneck, each expert learns compact and task-relevant representations while filtering out redundancy and noise. A multimodal bottleneck routing fusion module then fuses these expert latents with hard mining strategy, guiding the prediction in the ordinal sentiment space. Extensive experiments on 4 MSA datasets and 4 language models show that MoB effectively leverages informative latents from diverse modalities and captures general sentiment structure. Beyond stronger performance, MoB comprehensively captures fine-grained intra- and inter-modal dynamics, enabling more trustworthy localization of nuanced video sentiment signals.

Index Terms—Mixture-of-Experts, Information Bottleneck, Ordinal Learning, Video-based Multimodal Sentiment Analysis

## I. INTRODUCTION

ULTIMODAL machine learning [1] aims to integrate text, audio, and image. As a sub-field of multimodal learning, video-based Multimodal Sentiment Analysis (MSA) aims at jointly conveying rich affective cues from the human speaking videos, where spoken words provide explicit semantics, tone reflects emotional strength, and facial or bodily movements reveal implicit attitude [2]. However, effectively combining these heterogeneous sources remains challenging due to modality heterogeneity, task diversity, and the ordinal nature of sentiment intensity [3], [4].

![](images/0937323d667bd2d4955eae1871fa362161e7aae09ed0121d0ba18ab11fea447d.jpg)  
Fig. 1. Illustration of task-related information extraction for each modality in the ordinal sentiment space for video-based multimodal sentiment analysis.

Current MSA approaches mainly focus on building strong unimodal encoders and designing exquisite multimodal fusion strategies [5]. Despite recent advances, most fusion mechanisms remain task-agnostic and combine features from different modalities in a static manner [6], [7]. Besides, they typically treat sentiment prediction as a single task, ignoring its intrinsic hierarchical and ordinal structure [8], [9]. These limitations prevent previous methods from considering the distinctive roles that different modalities play in recognizing diverse aspects of sentiment. Consequently, they are easily affected by unimodal noise or cross-modal redundancy, leading to reduced interpretability and sub-optimal performance [10]. As illustrated in the example of Figure 1, visual gestures may better convey polarity (positive vs. negative), while acoustic tone more effectively captures intensity (emotional strength), which emphasizes the need for dynamic and task-aware multimodal learning in video-based MSA.

To address these challenges, we rethink MSA from an information-theoretic and divide-and-conquer perspective. The Information Bottleneck (IB) principle provides a principled way to extract task-relevant yet compact latent representations by filtering out redundant or noisy signals [11]. However, conventional IB-based models treat all features equally and fail to capture the varying difficulty and complementary nature of modal-specific and -shared feature [8], [12]–[14]. Meanwhile, Mixture-of-Experts (MoE) networks [15] specialize experts to divide and process modality-specific information in multimodal learning [16], [17], yet they are typically limited to feature-level routing and lack task-level disentanglement.

Motivated by these observations, we propose the Mixtureof-Bottleneck (MoB) framework, as illustrated in Figure 1, which unifies the Information Bottleneck principle and Mixture-of-Experts routing as a dynamic multimodal learning scheme. We first reformulate MSA task in an ordinal sentiment space and explicitly decouple it into polarity recognition and intensity estimation. This division aligns with the intrinsic structure of sentiment and enables each subtask to exploit the relevant cues from each modality. MoB then assigns polarity and intensity bottleneck experts to process the unimodal informative latents extracted by IB, capturing minimal yet sufficient information for each subtask. Moreover, we present a multimodal bottleneck router to adaptively fuse the unimodal latents and produce the final sentiment prediction with an effective hard mining strategy in the ordinal space. Following a divide-and-conquer paradigm, MoB partitions the feature space into multiple informative bottleneck experts and the label space into ordered polarity and intensity levels, achieving both interpretability and adaptability in multimodal fusion for video-based MSA. Our contribution can be summarized as:

• Ordinal Sentiment Modeling: We rethink video-based MSA as an ordinal regression task, decoupling polarity recognition and intensity estimation to better reflect the natural hierarchical structure of sentiment.

• Mixture-of-Bottleneck Experts: The MoB framework integrates the Information Bottleneck principle with expert specialization, assigning polarity- and intensityspecific bottleneck experts to adaptively extract compact and task-relevant latents from each modality.

• Dynamic Bottleneck Routing Fusion: A multimodal bottleneck router dynamically fuses task-aware unimodal latents across modalities, enabling fine-grained intermodal interaction and capturing diverse cross-modal synergy with various combination of information.

• Extensive and Comprehensive Experiment: Comprehensive experiments on 4 datasets and 4 pre-trained language models demonstrate that MoB consistently outperforms state-of-the-art methods, while offering enhanced interpretability for video-based MSA.

## II. RELATED WORK

## A. Multimodal Sentiment Analysis

Video-based Multimodal Sentiment Analysis (MSA) has emerged as a critical research area in cognition computation, aiming to identify speakers’ sentiments by integrating information from multiple modalities in the talking videos, such as text, audio, and image modality signals [2], [3]. Previous approaches primarily focused on capturing unimodal semantic representations and designing effective multimodal fusion strategies [5], which can be broadly categorized into early [18], [19], late [20]–[23], and hybrid fusion methods [24]–[28]. Since the fusion process is typically task-agnostic, these methods fail to provide interpretability for sentiment prediction, resulting in static multimodal fusion [6], [7]. Besides, considering the semantic dependency nature of the MSA task [10], performance improvement increasingly rely on the pre-trained language models [22], [26], [29]. However, due to the heterogeneity issues [28], integrating textual features [30]– [32]with acoustic and visual cues from speaker videos remains a major challenge. Thus, developing unified multimodal frameworks that can effectively combine video-based signals with diverse pre-trained language models are imperative in the field of MSA [10].

## B. Information-Theoretic Deep Learning

Information theory has been widely utilized in the development of deep learning models, offering a interpretable framework to conduct representation learning for understanding various modalities [12], [14]. Information Bottleneck (IB) [11], [33] has been extended and adapted in various multimodal contexts, beneficial in filtering task-irrelevant noise and capturing intra-modal dynamics in MSA domain [8], [13], [34], [35]. Regardless of the success in learning task-relevant features, IB remains limited in disentangling modality-specific and -shared representations, which is crucial for efficient multimodal representation learning [14]. Besides, existing methods typically apply VIB to individual tasks separately, without considering efficient fusion or coordination across various bottlenecks, especially in scenarios where broader multimodal learning goal accompanied by diverse unimodal learning biases [36]. Such gap highlights the motivation of our method in discovering more dynamic approaches for IBbased multimodal representation learning.

## C. Mixture-of-Expert Network

Mixture-of-Experts (MoE) was originally introduced as a conditional computation strategy in which a sparse gating network dynamically routes each input to a subset of specialized experts [15], [37]. MoE has been extended to multimodal learning where routing networks choose experts specialized for different modalities or modality combinations recently [16], [38]. Due to the productive effect in capturing relationships across diverse tasks, We consider MoE as a compelling mechanism in enhancing the performance and flexibility of information bottleneck learning. By embedding unimodal bottleneck with multiple experts, we can dynamically compress and store information based on the difficulty and relevance of specific subtasks.

## D. Ordinal Nature of Sentiment

Modeling sentiment has been a long-standing and challenging research due to the inherent subjective and ambiguous nature of sentiment cues [4], [39]. Recent advances in ordinal sentiment learning [34], [40] have demonstrated the potency of treating affective labels as inherently relative rather than absolute. Inspired by the separation thought of sentiment aspects [41], we decompose the sentiment space into distinctive polarity and intensity dimensions, thereby decoupling the multimodal learning process of sentiment modeling. By doing so, the model can adaptively assign the related polarity and intensity cues of each modality and exploit ordinal information at both label and feature levels.

![](images/a7d10d768949b9ac33cc4ecff12a810934183dfe7f77b4cfec7cea9cb3f65ac8.jpg)  
Fig. 2. The framework of Mixture-of-Bottleneck (MoB) to dynamically process modalities in the informative ordinal space.

## III. METHODOLOGY

Considering MSA task in a divide-and-conquer paradigm, we firstly construct the ordinal sentiment space in Sec. III-B, which divide the original sentiment space into two decoupled subspace to explicitly enable task-aware feature extraction. Then, we present PIBE in Sec. III-C, aiming to combine the strength of MoE and IB in dynamically integrating task-related information and filtering task-unrelated noise according to the supervision from each subspace. Next, we present MBRF in Sec. III-D, focusing on adaptively capturing the cross-modal dynamics and exploring the optimal way in conducting multimodal information fusion. Lastly, we design a hard mining strategy based on the ordinal sentiment space to enhance convergence on difficult samples and present the total multitask optimization objective in Sec. III-D.

## A. Problem Definition

Given human-centric videos with three modalities, including language modality $X _ { l }$ (utterance), acoustic modality $X _ { a }$ (speaking audios), and visual modality $X _ { v }$ (face frames), the task of multimodal sentiment analysis aims at extracting features $F _ { u } ~ \in ~ \mathbb { R } ^ { N _ { u } \times d _ { u } }$ for unimodal signals $u \in \{ l , a , v \}$ to conduct multimodal fusion and leveraging the multimodal representation $F _ { M } \in \mathbb { R } ^ { d _ { M } }$ analyze the sentiment state $\hat { y } \in \mathbb { R } ^ { \mathcal { k } }$ of the speaker in the video. Note that $N _ { u }$ denotes the sequence length for temporal unimodal data in the video, while $d _ { u }$ and $d _ { M }$ denotes the feature dimension. With the ground truth sentiment labels $y ,$ the optimization objective for the final sentiment analysis task can be formulated as:

$$
\mathcal { L } _ { t a s k } = M S E ( y , \hat { y } ) = \parallel y - \hat { y } \parallel ^ { 2 }\tag{1}
$$

## B. Ordinal Sentiment Space

Since sentiment analysis task demands of abundant semantics from diverse modalities, directly predicting the final sentiment states may raise issues of coarse affective understanding and subjective unimodal bias [3]. Inspired by previous ordinal learning methods [34], [40], we organize the ground truth sentiment space in ordinal regression manner by dividing the sentiment state into two fine-grained aspects, polarity and intensity, referring to positive/neutral/negative tendency and the strength degree of sentiment [41], defined as:

$$
\begin{array} { r l r } & { } & { y _ { p o l a r } = \left\{ \begin{array} { l l } { + 1 , ~ y > 0 } \\ { 0 , ~ y = 0 } \\ { - 1 , ~ y < 0 } \end{array} \right. } \\ & { } & { y _ { i n t e n } = \left| ~ y ~ | < H , ~ \forall y \right. } \end{array}\tag{2}
$$

(3)

where $y _ { p o l a r }$ and $y _ { i n t e n }$ denote the polarity and intensity ground truth labels, H denote the widest annotation range for diverse datasets.

## C. Polarity and Intensity Bottleneck Experts

Considering modality input $X _ { u } ,$ where $u \in \{ l , a , v \}$ denote language, acoustic, and visual modality. After unimodal extraction by pre-trained large multimodal model, the feature of each modality can be denotes as $F _ { u } ~ \in \ \mathbb { R } ^ { n _ { u } \times d _ { u } }$ where $n _ { u }$ denotes the sequence length for temporal unimodal data in the video, while $d _ { u }$ denotes the feature dimension. To sufficiently capture task-related information, we leverage Information Bottleneck (IB) to extract unimodal features by variation approximation [11], [33], [42] with the supervision of diverse targets.

Given the input S and target $T$ to construct the Markov chain as $S  B  T$ , the optimization of mutual information learning can be formulated as:

$$
\operatorname* { m i n } \mathcal { L } _ { I B } ( S , T ) = \operatorname* { m i n } _ { p ( B | S ) } I ( S ; B ) - \beta ~ I ( B ; T )\tag{4}
$$

where B denotes the bottleneck containing necessary information of input S to conduct the task prediction for $T$ and $\beta$ serves as trade-off for two mutual information terms.

To inherit the optimization ability for backward gradient propagation of neural networks, previous methods [42], [43] adopt variational approximation $q ( T | B )$ for the conditional probability in Equ. 4 and turn the optimization of IB into:

$$
\operatorname* { m i n } \mathbb { E } _ { S } \Big [ K L ( p ( B | S ) \parallel q ( B ) ) \Big ] - \beta \mathbb { E } _ { B , S } \Big [ \log q ( T | B ) \Big ]\tag{5}
$$

Then a reparameterization trick utilizes multi-layer perception $( M L P )$ as the IB encoder to output the mean and variance for the bottleneck of the informative latents, which can be formulated as:

$$
B \sim { \mathcal { N } } ( \mu , \varepsilon ^ { 2 } ) , { \mathrm { w h e r e ~ } } \{ \mu , \varepsilon \} = M L P ( S )\tag{6}
$$

where $\boldsymbol { B } = \boldsymbol { \mu } + \boldsymbol { \varepsilon } \cdot \mathbf { z }$ and $\mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } )$ denotes to sample from a standard Gaussian distribution. Then for Equ. 5, the first term can be derived into $\mathcal { L } _ { i b } = K L ( \mathcal { N } ( \mu , \varepsilon ^ { 2 } ) \parallel \mathcal { N } ( 0 , \mathbf { I } ) )$ while the second term denotes the task prediction loss.

Aiming at adaptively extracting both unique and shared information from unimodal features and assigning various task-related bottlenecks to the downstream subtasks, we then utilize Mixture-of-Expert (MoE) networks [15], [44] to replace the original $M L P$ as the IB encoders. For $F _ { u } ,$ , the bottleneck expert consists of a group of N experts network $\left\{ \varphi _ { 1 } , \varphi _ { 2 } , . . . , \varphi _ { N - 1 } , \varphi _ { N } \right\}$ along with a router $r ( \cdot )$ which dynamically determine the flowing of each information bottleneck. Then, the output $\widetilde { y }$ of the Mixture-of-Bottleneck Experts network is derived from the summation of Top-k related expert, denoted as:

$$
\begin{array} { l } { \displaystyle \widetilde { \boldsymbol { y } } = \sum _ { i = 1 } ^ { N } \boldsymbol { r } ( \boldsymbol { B } ) _ { i } \cdot \boldsymbol { \varphi } _ { i } ( \boldsymbol { B } ) } \\ { \displaystyle = \sum _ { i = 1 } ^ { N } \boldsymbol { r } ( \boldsymbol { B } ) _ { i } \cdot \mathcal { N } ( \mu ; \mu _ { i } , \varepsilon _ { i } ^ { 2 } ) } \end{array}\tag{7}
$$

where $r ( B ) _ { i } = 0$ when i is not in the top k elements and the output can be regarded as a special type of Gaussian Mixture Models [45] with adaptive weights rely on the input.

The input latent B from Equ. 7 consists of task-related features from each modality while does not contain task-aware information. Inspired by [46], [47], we employ two settings to feed the subtask guidance into the router:

Multi-gate Setting. The router utilizes $M L P$ as non-linear projection with GELU activation layer [48] for each subtask to specify the corresponding experts, denoted as:

$$
r _ { t a s k } ( B ) = G a t e ( M L P ( B ; \theta _ { t a s k } ) ) \in \mathbb { R } ^ { b \times N }\tag{8}
$$

Multi-embed Setting. The information of each subtask is initialized with a specific embedding $E m b _ { t a s k }$ and then added into the original input for the router, denoted as:

$$
\begin{array} { r } { r _ { t a s k } ( B ) = G a t e ( B + E m b _ { t a s k } ) \in \mathbb { R } ^ { b \times N } } \end{array}\tag{9}
$$

where b denotes the batch size and N denotes the number of bottleneck experts.

By these settings, the router can adaptively decide the experts with specific or sharing weights for different subtasks based on the Gate function. Following [15], both tunable Gaussian noise and additional soft constraint are included to enhance the load balance when routing bottleneck experts. The balancing constraint can be denoted as:

$$
\mathcal { L } _ { m o e } = C V ^ { 2 } \Big [ \sum _ { B } r ( B ) \Big ] + C V ^ { 2 } \Big [ \sum _ { B } P ( B , i ) \Big ]\tag{10}
$$

$$
\mathrm { w h e r e } \ C V ^ { 2 } ( x ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } | x | = 1 , } \\ { \displaystyle \frac { \mathrm { V a r } ( x ) } { \left( \mathrm { E } [ x ] \right) ^ { 2 } + 1 e ^ { - 1 0 } } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{11}
$$

where $P ( B , i )$ is defined as the probability that $r ( B ) _ { i }$ is nonzero when we resample the noise on the i-th expert while keeping all previously sampled noise on the other elements fixed. With such constraint, the router tend to encourage similar numbers of bottlenecks and strengthen the convergence efficiency of the model.

The IB decoder is lastly adopted to map the output of Mixture-of-Bottleneck Experts into the sentiment label space. Considering ordinal learning by polarity and intensity prediction of sentiment, the subtasks prediction loss for each router is defined as follows:

Polarity-guided Router. With the supervision label $y _ { p o l a r }$ from Equ. 2, the objective for polarity-guided router is computed in a classification manner, denoted as:

$$
\mathcal { L } _ { p o l a r } = - \mathbb { E } _ { x } \ \left[ y _ { p o l a r } \log \widetilde { y } _ { p o l a r } \right]\tag{12}
$$

Intensity-guided Router. With the supervision signal y<sub>inten</sub> from Equ. 3, the objective of intensity-guided router is measured by regression learning, denoted as:

$$
\mathcal { L } _ { i n t e n } = \mathbb { E } _ { x } \left[ \left| y _ { i n t e n } - \widetilde { y } _ { i n t e n } \right| \right]\tag{13}
$$

For unimodal features $u \in \{ l , a , v \}$ , deriving from Equ. 5, 10,12 and 13, the optimization objective of polarity and intensity bottleneck experts is denoted as:

$$
\mathcal { L } _ { b t n k } = \sum _ { u } ( w _ { i b } \mathcal { L } _ { i b } ^ { u } + w _ { m o e } \mathcal { L } _ { m o e } ^ { u } ) + w _ { o r d } ( \mathcal { L } _ { p o l a r } + \mathcal { L } _ { i n t e n } )\tag{14}
$$

where $w _ { i b } , \ w _ { m o e }$ and $w _ { o r d }$ are hyper-parameters adjusting the contribution of IB compression, MoE load balancing and subtask prediction.

## D. Multimodal Bottleneck Routing Fusion

To capture unique and shared information inside and across modalities, we propose Multimodal Bottleneck Routing Fusion (MBRF) to explore various cross-modal synergy with diverse combinations of unimodal bottlenecks.

Given the unimodal informative latents $\{ B _ { j } ^ { u } | _ { j = 1 } ^ { b } \} \in \mathbb { R } ^ { n _ { b } \times d }$ extracted by previous bottleneck experts and a series of fusion experts $\{ \phi _ { 1 } , . . . , \phi _ { N } \}$ , we can derive a latent-to-expert affinity score matrix $S _ { j }$ for each instance:

$$
S ( B ) _ { j } = G a t e ( B _ { j } ^ { u } ) \in \mathbb { R } ^ { 6 \times N }\tag{15}
$$

where N denotes the number of experts and d is the common hidden dimension and $n _ { b } = b \times 6$ denotes the total number of latents for each batch with batch size b, since each modality contains two kinds of informative latents output by polarity and intensity router, respectively.

Then we can compute the routing weight matrix R(B) of expert for the selected latents and the index matrix I with selected latents, denoted as:

$$
\mathcal { R } ( B ) , \mathcal { T } = T o p K ( S ^ { \top } , K )\tag{16}
$$

where $\boldsymbol { \mathcal { T } } = ( I [ i , j ] ) _ { N \times K }$ specifies $j \mathrm { - t h }$ selected latent of the i-th expert. The aggregated output of all fusion experts $\phi _ { i }$ can be denoted as:

$$
\Delta _ { j } = \sum _ { i , j \in \mathbb { Z } _ { i } } \mathcal { R } ( B ) _ { i } \cdot \phi _ { i } ( B )\tag{17}
$$

Considering enhancing such cross-modal dynamics for each unimodal latents $F _ { u }$ [8], we adopt residual connection [49] to obtain the final multimodal representation as:

$$
F _ { M } = C o n c a t ( F _ { l } , F _ { a } , F _ { v } ) + \Delta \in \mathbb { R } ^ { b \times 6 d }\tag{18}
$$

Similar with [37], we regularize the multimodal routing fusion process by limiting the maximization number of experts for each latent. We firstly apply Softmax over the transposed latent-to-expert score matrix S to produce the probabilistic routing distribution where each expert softly selects latents according to their relative scores, denoted as:

$$
A _ { i j } = \frac { \exp ( S _ { i j } ^ { \top } ) } { \sum _ { m = 1 } ^ { n } \exp ( S _ { i m } ^ { \top } ) }\tag{19}
$$

Besides, we add an entropy regularization term for matrix A to discourages overly peaked assignments by penalizing lowentropy distributions, denoted as:

$$
\mathcal { L } _ { e t } = - \sum _ { i = 1 } ^ { N } \mathbb { E } _ { j } \Big [ A _ { i j } \log A _ { i j } \Big ]\tag{20}
$$

Combining both the constraint penalties and the entropy term, the fusion routing objective can be denoted as a entropyregularized linear programming problem:

$$
\mathcal { L } _ { f u s i o n } = - \sum _ { i = 1 } ^ { N } \mathbb { E } _ { j } \left[ S _ { i j } ^ { \top } A _ { i j } \right] + \lambda \cdot \mathcal { L } _ { e t } \mathrm { ~ , ~ } \forall j : \sum _ { i = 1 } ^ { N } A _ { i j } \leq h\tag{21}
$$

where each latent is routed to at most h experts.

## E. Multi-task Objectives

To further explore the ordinal sentiment space, we adopt a hard-sample mining strategy that dynamically adjusts the task loss based on the polarity and intensity prediction errors δ. We define three types of hard samples $C _ { h }$ selected with the prediction errors δ:

$$
\begin{array} { r l } & { C _ { 1 } : \ \delta _ { p } = 1 , \ \delta _ { i } \leq \tau } \\ & { C _ { 2 } : \ \delta _ { p } = 0 , \ \delta _ { i } \geq 2 \tau \ , \ \mathrm { w i t h } \left\{ \delta _ { i } = \mathbb { I } [ \mathrm { a r g } \operatorname* { m a x } ( \hat { y } _ { \mathrm { p o l a r } } ) \neq y _ { \mathrm { p o l a r } } ] \right. } \\ & { C _ { 3 } : \ \delta _ { p } = 1 , \ \delta _ { i } \geq 2 \tau } \end{array}\tag{22}
$$

where τ denotes the difficulty threshold of the sentiment score range H. Thus, the hard mining loss is defined as:

$$
\mathcal { L } _ { h a r d } = \sum _ { h = 1 } ^ { 3 } w _ { h } \big | M L P ( F _ { M } ) - y _ { g t } \big |\tag{23}
$$

where $w _ { h }$ denotes the weight assigned for hard sample $C _ { h }$ which encourages the model to learn more effectively from the difficult cases.

Overall, the joint training process for above modules including bottleneck extraction, experts balancing, and hard mining, the multi-task objective can be summarized as:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { t a s k } + \mathcal { L } _ { b t n k } + w _ { f u s } \cdot \mathcal { L } _ { f u s i o n } + w _ { h a r d } \cdot \mathcal { L } _ { h a r d }\tag{24}
$$

## IV. EXPERIMENT

## A. Datasets

We evaluate our method on multilingual MSA task, including 2 English and 2 Chinese corpora. CMU-MOSI (MOSI) [50] contains 2,199 monologue utterances from 93 opinion-based YouTube videos featuring 89 movie reviewers. Each segment is annotated for sentiment on a continuous scale ranging from -3 (strongly negative) to +3 (strongly positive). CMU-MOSEI (MOSEI) [51] extends MOSI by providing roughly 66 hours of video clips across 250 topics from 1,000 speakers with the same sentiment range. CH-SIMS (SIMS) [52] comprises 2,281 segments from 60 Chinese videos in different films, TV dramas, and variety shows, covering natural expressions with varied poses, occlusions, and illumination from 474 speakers, using the sentiment scale from -1 (negative) to +1 (postive). CH-SIMS v2 (SIMSv2) [53] approximately doubles the scale by adding more supervised and unsupervised samples under the same annotation scheme where only supervised samples are used in our papers for fair comparison.

## B. Implementation Details

For language, on MOSI and MOSEI datasets, we employ three commonly-used language models trained on English scopes including BERT [30], RoBERTa [31] and DeBERTaV3 [32], [58], while on SIMS and SIMSv2 datasets, we adopt three advanced language models tuned on Chinese scopes accordingly, including BERT-chinese [30], chinese-RoBERTawwm-ext [59], chinese-MacBERT [60]. For acoustic and visual modalities, we adopt ImageBind [61] to extract the unimodal features for its excellent cross-modal alignment performance [10].

## C. Quantitative Results

Comparing with current models, MoB achieves superior performance, outperforming previous methods on English (CMU-MOSI, CMU-MOSEI) and Chinese (CH-SIMS, CH-SIMSv2) benchmark datasets. Variants like MoB-DeBERTaV3 and MoB-RoBERTa achieve the best scores across a majority of key metrics. Besides, MoB consistently enhances the results of various pre-trained language models, revealing its excellent generalizability.

We attribute the effectiveness of MoB on diverse metrics to the decoupling of sentiment space with polarity (classification) and intensity (regression). This allows it to achieve better balance across two subtasks, simultaneously excelling at both where other models often compromise. The results also show a clear performance gain when using larger base models (DeBERTaV3) on English datasets, while on more challenging Chinese datasets, the improvements from model scaling (MacBERT) are more subtle, indicating the necessity of exploring cross-modal synergy.

TABLE I  
PERFORMANCE COMPARISON BETWEEN THE PROPOSED MOB AND BASELINES ON CMU-MOSI AND CMU-MOSEI DATASETS. THE BASELINE MODELS ARE REPRODUCED EXCEPT FOR <sup>†</sup> MEANING THAT THE CORRESPONDING RESULTS ARE COPIED FROM THE ORIGINAL PAPER. ’-/-’ FOR ACC2 AND F1 DENOTE THE TWO EVALUATION SETTINGS OF NON-NEGATIVE/NEGATIVE AND POSITIVE/NEGATIVE RESPECTIVELY.
<table><tr><td rowspan="2">Models</td><td colspan="5">CMU-MOSI</td><td colspan="5">CMU-MOSEI</td></tr><tr><td>Acc7↑</td><td>Acc2↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td><td>Acc7↑</td><td>Acc2↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td></tr><tr><td>Self-MM [20]</td><td>45.8</td><td>82.7  / 84.9</td><td>82.6 / 84.8</td><td>0.731</td><td>0.785</td><td>53.0</td><td>82.6 / 85.2</td><td>82.8 / 85.2</td><td>0.540</td><td>0.763</td></tr><tr><td>MMIM [21]]</td><td>45.0</td><td>83.0 / 85.1</td><td>82.9 / 85.0</td><td>0.738</td><td>0.781</td><td>53.1</td><td>81.9 / 85.1</td><td>82.3 / 85.0</td><td>0.547</td><td>0.752</td></tr><tr><td>MMCL [24]</td><td>46.5</td><td>84.0 / 86.3</td><td>83.8 / 86.2</td><td>0.705</td><td>0.797</td><td>53.6</td><td>84.8 / 85.9</td><td>84.8 / 85.7</td><td>0.537</td><td>0.765</td></tr><tr><td>ConFEDE [19]</td><td>42.3</td><td>84.2 / 85.5</td><td>84.1 / 85.5</td><td>0.742</td><td>0.784</td><td>54.9</td><td>81.7  / 85.8</td><td>82.2 / 85.8</td><td>0.522</td><td>0.780</td></tr><tr><td>MIB† [13]</td><td>48.2</td><td>- / 85.2</td><td>- / 85.2</td><td>0.728</td><td>0.793</td><td>53.0</td><td>- / 86.2</td><td>- / 86.2</td><td>0.584</td><td>0.789</td></tr><tr><td>MTMD [25]</td><td>47.5</td><td>84.0 / 86.0</td><td>83.9 / 86.0</td><td>0.705</td><td>0.799</td><td>53.7</td><td>84.8 /  86.1</td><td>84.9 / 85.9</td><td>0.531</td><td>0.767</td></tr><tr><td>ALMT [26]</td><td>48.2</td><td>84.2 / 85.9</td><td>84.1 / 86.0</td><td>0.698</td><td>0.811</td><td>54.1</td><td>83.9 / 85.9</td><td>84.0 / 86.0</td><td>0.536</td><td>0.762</td></tr><tr><td>EMT† [54]</td><td>47.4</td><td>83.3 / 85.0</td><td>83.2 / 85.0</td><td>0.705</td><td>0.798</td><td>54.5</td><td>83.4 / 86.0</td><td>83.7  / 86.0</td><td>0.527</td><td>0.774</td></tr><tr><td>TMSON† [34]</td><td>47.4</td><td>85.4 / 87.2</td><td>85.4 / 87.2</td><td>0.687</td><td>0.809</td><td>55.6</td><td>85.2 / 86.4</td><td>85.3 / 86.2</td><td>0.526</td><td>0.766</td></tr><tr><td>EAU† [35]</td><td>48.8</td><td>-1-</td><td>- / 86.2</td><td></td><td>0.809</td><td>54.8</td><td>-1-</td><td>- / 86.9</td><td></td><td>0.816</td></tr><tr><td>KuDA† [55]</td><td>47.1</td><td>84.4 / 86.4</td><td>84.5 / 86.5</td><td>0.705</td><td>0.795</td><td>52.9</td><td>83.3 / 86.5</td><td>83.0 / 86.6</td><td>0.529</td><td>0.776</td></tr><tr><td>MFON† [28]</td><td>44.9</td><td>84.8 / 86.9</td><td>84.8 / 86.9</td><td>0.725</td><td>0.797</td><td>53.7</td><td>82.7 / 86.3</td><td>83.1 / 86.3</td><td>0.528</td><td>0.780</td></tr><tr><td>DLF† [9]</td><td>47.1</td><td>- / 85.1</td><td>- / 85.0</td><td>0.731</td><td>0.781</td><td>53.9</td><td>- / 85.4</td><td>- / 85.3</td><td>0.536</td><td>0.764</td></tr><tr><td>MCL-MCF† [56]</td><td></td><td>84.9 / 87.3</td><td>84.7 / 87.2</td><td>0.692</td><td>0.799</td><td></td><td>84.2 / 86.4</td><td>84.4 / 86.3</td><td>0.536</td><td>0.767</td></tr><tr><td>MLCL† [57]</td><td>46.6</td><td>84.1 / 86.4</td><td>83.9 / 86.3</td><td>0.701</td><td>0.798</td><td>53.2</td><td>84.1  / 86.3</td><td>84.3 / 86.2</td><td>0.551</td><td>0.756</td></tr><tr><td>MAG-XLNet []</td><td>44.4</td><td>- / 85.1</td><td>- / 85.2</td><td>0.740</td><td>0.804</td><td>51.3</td><td>- / 85.8</td><td>- / 85.9</td><td>0.593</td><td>0.790</td></tr><tr><td>MSG-MBA [29]</td><td>46.1</td><td>- / 87.0</td><td>- / 87.0</td><td>0.696</td><td>0.816</td><td>52.6</td><td>- / 86.3</td><td>- / 86.3</td><td>0.581</td><td>0.795</td></tr><tr><td>UniMSE† [22]</td><td>48.7</td><td>85.9 / 86.9</td><td>85.8 / 86.4</td><td>0.691</td><td>0.809</td><td>54.4</td><td>85.9 /  87.5</td><td>85.8 / 87.5</td><td>0.523</td><td>0.773</td></tr><tr><td>MMML† [23]</td><td>48.3</td><td>85.9 / 88.2</td><td>85.9 / 88.2</td><td>0.643</td><td>0.838</td><td>55.0</td><td>86.3 / 86.7</td><td>86.2 / 86.5</td><td>0.517</td><td>0.791</td></tr><tr><td>ITHP† [8]</td><td></td><td>- / 88.7</td><td>- / 88.6</td><td>0.643</td><td>0.852</td><td></td><td>- / 87.3</td><td>- / 87.4</td><td>0.564</td><td>0.813</td></tr><tr><td>MoB-BERT</td><td>47.5</td><td>84.7 / 86.6</td><td>84.6 / 86.5</td><td>0.681</td><td>0.824</td><td>54.4</td><td>84.1 / 87.1</td><td>84.3 / 87.0</td><td>0.513</td><td>0.797</td></tr><tr><td>MoB-RoBERTa</td><td>49.9</td><td>86.6 / 88.9</td><td>86.5 / 88.8</td><td>0.612</td><td>0.854</td><td>55.9</td><td>84.0 / 88.3</td><td>84.5 / 88.3</td><td>0.499</td><td>0.819</td></tr><tr><td>MoB-DeBERTaV3</td><td>51.1</td><td>87.4 / 89.5</td><td>87.3 / 89.5</td><td>0.598</td><td>0.873</td><td>55.1</td><td>86.7 / 88.5</td><td>86.8 / 88.4</td><td>0.496</td><td>0.821</td></tr></table>

TABLE II

PERFORMANCE COMPARISON BETWEEN THE PROPOSED MOB AND BASELINES ON CH-SIMS AND CH-SIMSV2 DATASETS. THE BASELINE MODELS ARE REPRODUCED EXCEPT FOR † MEANING THAT THE CORRESPONDING RESULTS ARE COPIED FROM THE ORIGINAL PAPER.
<table><tr><td rowspan="2">Models</td><td colspan="6">CH-SIMS</td><td colspan="6">CH-SIMS v2</td></tr><tr><td>Acc5↑</td><td>Acc3↑</td><td>Acc2↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td><td>Acc5↑</td><td>Acc3↑</td><td>Acc2↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td></tr><tr><td>Self-MM [20]</td><td>43.8</td><td>66.1</td><td>79.3</td><td>79.4</td><td>0.416</td><td>0.600</td><td>53.5</td><td>72.7</td><td>78.7</td><td>78.6</td><td>0.315</td><td>0.691</td></tr><tr><td>MMIM [21]</td><td>43.3</td><td>66.8</td><td>78.4</td><td>78.1</td><td>0.431</td><td>0.587</td><td>50.5</td><td>70.4</td><td>77.8</td><td>77.8</td><td>0.339</td><td>0.641</td></tr><tr><td>AV-MC [53]</td><td>45.5</td><td>68.5</td><td>79.7</td><td>80.2</td><td>0.372</td><td>0.685</td><td>52.1</td><td>73.2</td><td>80.6</td><td>80.7</td><td>0.301</td><td>0.721</td></tr><tr><td>ALMT [26]</td><td>44.9</td><td>68.8</td><td>80.8</td><td>80.4</td><td>0.409</td><td>0.625</td><td>49.8</td><td>72.1</td><td>80.5</td><td>80.6</td><td>0.342</td><td>0.652</td></tr><tr><td>ConFEDE [19]</td><td>42.5</td><td>69.4</td><td>81.6</td><td>81.4</td><td>0.391</td><td>0.669</td><td>50.2</td><td>71.6</td><td>80.4</td><td>80.2</td><td>0.355</td><td>0.702</td></tr><tr><td>EMT† [54]</td><td>43.5</td><td>67.4</td><td>80.1</td><td>80.1</td><td>0.396</td><td>0.623</td><td>一</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TMSON† [34]</td><td>45.3</td><td>66.4</td><td>81.4</td><td>81.2</td><td>0.417</td><td>0.658</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KuDA† [55]</td><td>43.5</td><td>66.5</td><td>80.7</td><td>80.7</td><td>0.408</td><td>0.613</td><td>53.1</td><td>74.3</td><td>80.2</td><td>80.1</td><td>0.289</td><td>0.741</td></tr><tr><td>MFON† [28]</td><td></td><td>一</td><td>78.6</td><td>78.5</td><td>0.420</td><td>0.594</td><td>一</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MCL-MCF† [56]</td><td></td><td>1</td><td>81.8</td><td>81.8</td><td>0.410</td><td>0.588</td><td>-</td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td>MLCL† [57]</td><td>45.5</td><td>-</td><td>81.4</td><td>81.3</td><td>0.412</td><td>0.596</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MoB-BERT</td><td>45.1</td><td>69.2</td><td>81.2</td><td>81.2</td><td>0.381</td><td>0.696</td><td>51.2</td><td>72.7</td><td>81.3</td><td>81.2</td><td>0.336</td><td>0.727</td></tr><tr><td>MoB-RoBERTa</td><td>45.7</td><td>70.9</td><td>81.9</td><td>82.1</td><td>0.371</td><td>0.684</td><td>50.3</td><td>72.8</td><td>81.1</td><td>81.0</td><td>0.332</td><td>0.728</td></tr><tr><td>MoB-MacBERT</td><td>44.2</td><td>69.4</td><td>81.4</td><td>81.6</td><td>0.378</td><td>0.697</td><td>51.3</td><td>73.6</td><td>82.1</td><td>82.1</td><td>0.321</td><td>0.743</td></tr></table>

## D. Ablation Study

The ablation study demonstrates that MoB achieves its superior performance by effectively integrating three key innovations: the Information Bottleneck (IB) principle, Mixtureof-Experts (MoE) routing, and ordinal sentiment decoupling. Comparing the full MoB model against baselines using only IB or only MoE confirms that the synergistic combination of IB (for minimal yet sufficient information compression) and MoE (for dynamic specialization) is necessary for optimal results. Crucially, removing the specialized Polarity and Intensity Bottleneck Experts (PIBE) or Multimodal Bottleneck Routing Fusion (MBRF) results in the most significant performance degradations, further highlighting the vital role of both IB and MoE in learning compressed and task-relevant unimodal and multimodal features.

TABLE III  
ABLATION STUDY ON THE MODULE OF MOB ON CMU-MOSI AND CH-SIMS DATASETS. F1 IS REPORTED IN POSITIVE/NEGATIVE SETTING.
<table><tr><td rowspan="2">Module</td><td colspan="4">CMU-MOSI</td><td colspan="4">CH-SIMS</td></tr><tr><td>Acc7↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td><td>Acc5↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td></tr><tr><td>MoB</td><td>51.1</td><td>89.5</td><td>0.598</td><td>0.873</td><td>44.2</td><td>81.6</td><td>0.378</td><td>0.697</td></tr><tr><td>only IB</td><td>47.1</td><td>87.1</td><td>0.652</td><td>0.849</td><td>40.6</td><td>77.2</td><td>0.436</td><td>0.594</td></tr><tr><td>only MoE</td><td>46.5</td><td>86.5</td><td>0.671</td><td>0.835</td><td>38.7</td><td>76.8</td><td>0.425</td><td>0.586</td></tr><tr><td>w/o PIBE</td><td>49.4</td><td>87.7</td><td>0.641</td><td>0.859</td><td>41.2</td><td>79.0</td><td>0.418</td><td>0.631</td></tr><tr><td>w/o Polarity</td><td>50.4</td><td>88.4</td><td>0.601</td><td>0.871</td><td>42.0</td><td>79.9</td><td>0.387</td><td>0.667</td></tr><tr><td>w/o Intensity</td><td>47.4</td><td>89.1</td><td>0.631</td><td>0.863</td><td>42.4</td><td>80.6</td><td>0.389</td><td>0.669</td></tr><tr><td>w/o  $\mathcal { L } _ { i b }$ </td><td>50.2</td><td>88.6</td><td>0.612</td><td>0.861</td><td>41.4</td><td>78.7</td><td>0.407</td><td>0.660</td></tr><tr><td>w/o  $\mathcal { L } _ { m o e }$ </td><td>49.2</td><td>89.4</td><td>0.602</td><td>0.871</td><td>43.9</td><td>81.4</td><td>0.379</td><td>0.683</td></tr><tr><td>w/o  $\mathcal { L } _ { b t n k }$ </td><td>47.7</td><td>88.7</td><td>0.623</td><td>0.858</td><td>43.0</td><td>79.4</td><td>0.398</td><td>0.659</td></tr><tr><td>w/o MBRF</td><td>48.3</td><td>87.4</td><td>0.626</td><td>0.862</td><td>41.3</td><td>78.5</td><td>0.406</td><td>0.645</td></tr><tr><td>w/o residual</td><td>50.8</td><td>88.2</td><td>0.602</td><td>0.871</td><td>41.9</td><td>80.5</td><td>0.398</td><td>0.657</td></tr><tr><td>w/o  $\mathcal { L } _ { e t }$ </td><td>49.3</td><td>89.2</td><td>0.599</td><td>0.870</td><td>43.1</td><td>79.9</td><td>0.382</td><td>0.679</td></tr><tr><td>wlo Lfusion</td><td>49.1</td><td>87.9</td><td>0.615</td><td>0.861</td><td>42.3</td><td>79.1</td><td>0.394</td><td>0.649</td></tr><tr><td> $\mathrm { w } / \mathrm { o } ~ \dot { \mathcal { L } } _ { h a r d }$ </td><td>48.6</td><td>88.9</td><td>0.609</td><td>0.854</td><td>40.9</td><td>79.0</td><td>0.412</td><td>0.638</td></tr></table>

Additionally, the ordinal sentiment decoupling is proved highly effective, particularly the Intensity Experts component. Removing the intensity estimation module (w/o Intensity) cause substantial drops on most metrics, showing the importance in providing the fine-grained information needed for accurate sentiment prediction, especially for regression metrics like MAE and Corr. While the polarity recognition module focuses on the triple classes of sentiment, removing which (w/o Polarity) directly impacts the classification metrics including Acc7 and F1. The results on each module consistently validate that MoB is a tightly integrated system with PIBE and MBRF to capture intra- and inter-modal dynamics, governed by the bottleneck compressing, expert routing and hard mining losses.

## E. Flexible Modality input

As shown in Table IV, the ablation study on diverse modality input demonstrate the MoB framework’s exceptional flexibility and robustness in handling dynamic modal inputs. The model achieves optimal performance when all modalities $( \{ l , a , v \} )$ are available, effectively utilizing the dynamic bottleneck routing fusion to synthesize diverse cross-modal synergies. While when modalities are missing, MoB also exhibits strong adaptability. For instance, the performance drop is minimal when shifting from full-modal input to subsets containing language modality (e.g., {l, a}, {l, v} or only {l}). This indicates that the routing mechanism can dynamically adjust its fusion strategy to effectively use available information, rather than relying on a static input structure.

TABLE IV  
PERFORMANCE WITH FLEXIBLE MODALITY INPUT FOR MOB ON MOSIAND SIMS DATASETS, WHERE $\{ l , a , v \}$ DENOTE LANGUAGE, AUDIO ANDVISION MODALITY, REPECTIVELY.
<table><tr><td rowspan="3">Modality</td><td colspan="4">CMU-MOSI</td><td colspan="4">CH-SIMS</td></tr><tr><td>Acc7↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td><td>Acc5↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td></tr><tr><td> $l , a , v$ </td><td>51.1</td><td>89.5</td><td>0.598</td><td>0.873</td><td>44.2</td><td>81.6</td><td>0.378</td><td>0.697</td></tr><tr><td> $l , a$ </td><td>51.0</td><td>89.3</td><td>0.599</td><td>0.870</td><td>43.7</td><td>80.1</td><td>0.401</td><td>0.643</td></tr><tr><td> $l , v$ </td><td>49.9</td><td>89.4</td><td>0.602</td><td>0.867</td><td>43.2</td><td>80.5</td><td>0.412</td><td>0.639</td></tr><tr><td> $a , v$ </td><td>20.6</td><td>55.7</td><td>1.446</td><td>0.207</td><td>36.5</td><td>70.1</td><td>0.475</td><td>0.516</td></tr><tr><td>only l</td><td>50.1</td><td>87.0</td><td>0.619</td><td>0.845</td><td>43.1</td><td>79.3</td><td>0.418</td><td>0.606</td></tr><tr><td>only a</td><td>19.7</td><td>54.9</td><td>1.516</td><td>0.113</td><td>26.4</td><td>63.1</td><td>0.539</td><td>0.358</td></tr><tr><td>only v</td><td>19.5</td><td>53.8</td><td>1.505</td><td>0.101</td><td>21.6</td><td>68.5</td><td>0.572</td><td>0.240</td></tr></table>

Furthermore, the result highlights the framework’s capability to discern the contribution from each modality and $\mathrm { { d y } \mathrm { { - } } }$ namically assign appropriate bottleneck latents for sentiment prediction. The significant performance gap between ”only $l ^ { \ast }$ and ”only $a / v ^ { , , }$ confirms that language modality plays the primary role for polarity recognition, while the inclusion of acoustic and visual modalities enhances the regression metric, suggesting their vital role in fine-grained intensity estimation. By applying the Information Bottleneck principle, MoB effectively filters out noise from inferior modalities (audio/vision) while extracting minimal yet sufficient latents to refine the final prediction.

## F. Variants of Router Settings

We evaluate the proposed two settings of bottleneck experts router in Table V. Following [17], we present three kinds of Gate function (Softmax, Leplace, Gaussian) for the Multiembed setting. Results demonstrate that on CMU-MOSI, the Multi-gate (MLP) setting achieves the highest classification metrics (Acc7: 51.1, F1: 89.5), while Multi-embed (Laplace) performs best in regression metrics (MAE: 0.581, Corr: 0.877). While on CH-SIMS, Multi-gate (MLP) excels on both classification and regression metrics (Acc5: 44.2, MAE: 0.378). Overall, Multi-gate (MLP) reaches its consistent excellence in the design of polarity and intensity router, which is attributed to its effectiveness in using learnable weights for dynamically assigning diverse latents based on the specific subtask.

TABLE V  
VARIANTS ON ROUTER SETTINGS OF MOB ON CMU-MOSI AND CH-SIMS DATASETS. F1 IS REPORTED IN POSITIVE/NEGATIVE SETTING.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Gate</td><td colspan="4">CMU-MOSI</td><td colspan="4">CH-SIMS</td></tr><tr><td>Acc7↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td><td>Acc5↑</td><td>F1↑</td><td>MAE↓</td><td>Corr↑</td></tr><tr><td>Multi-gate</td><td>MLP</td><td>51.1</td><td>89.5</td><td>0.598</td><td>0.873</td><td>44.2</td><td>81.6</td><td>0.378</td><td>0.697</td></tr><tr><td rowspan="3">Multi-embed</td><td>Softmax</td><td>50.3</td><td>89.2</td><td>0.625</td><td>0.853</td><td>42.6</td><td>80.1</td><td>0.411</td><td>0.663</td></tr><tr><td>Laplace</td><td>51.0</td><td>89.4</td><td>0.581</td><td>0.877</td><td>43.5</td><td>80.9</td><td>0.412</td><td>0.669</td></tr><tr><td>Gaussian</td><td>49.9</td><td>88.9</td><td>0.597</td><td>0.871</td><td>43.9</td><td>81.2</td><td>0.382</td><td>0.683</td></tr></table>

## G. Polarity and Intensity Routing During Training

As shown in Figure 3, the visualization of expert usage during training confirms the highly effective and dynamic routing capabilities of the MoB framework across language, audio, and vision modalities for either Polarity or Intensity tasks. A key observation is the consistent and near-uniform utilization of all six bottleneck experts on both subtask for each modality after efficient training. The contribution of experts shift from imbalance to balance during training, which demonstrates the effectiveness of MoB router and ensures every expert to actively involve in capturing the necessary finegrained latents required for accurate prediction.

Moreover, the consistent and balanced expert utilization across all three modalities strongly suggests MoB possesses excellent generalization ability in dynamically handling diverse input data. The framework doesn’t rely on one modality for each subtask. Such uniform distribution indicates that the router can robustly identify and route task-relevant information, enabling the model to learn better modality-specific representations. The balanced engagement across the multimodal input space is a critical factor in preventing performance collapse when one modality is noisy or missing, thereby promoting overall model robustness and generalization.

## H. Task-aware Routing During Inference.

The visualization in Table 4 provides the shared and specific expert usage ratio during inference, explicitly demonstrating the effectiveness of PIBE routers in decoupling and assigning task-aware latents. By dynamically categorizing the bottleneck experts into ”Shared”, ”Polarity-only” and ”Intensity-only”, the result proves MoB does not blindly extract information but actively discerns each latent according to specific subtask, revealing trustworthy unimodal feature extraction.

![](images/65c00efbda6d59c595b721194a592f0d4f94e38839cac11d8b4f9c040091b611.jpg)  
(a) Language Modality

![](images/17155f866ad76b7ee3d1f362772039c3553f6d6041baea9bc0596875170df52e.jpg)  
(b) Audio Modality

![](images/c11ecb1816dbee9040bfa551e7cb83c900eb7c3450eaccdc2838026211619373.jpg)  
(c) Vision Modality

Fig. 3. Tracing of the polarity and intensity routing during training for (a) language, (b) audio and (c) vision modality.  
![](images/6b67fc281cf7718de14418bbdcbbdc261421a2cd08c3a2e62eadcf4d9dad784d.jpg)  
Fig. 4. Illustration of the shared and specific expert usage ratio between the polarity and intensity router on MOSI.

We observe that the dominance of the ”Shared” category across all three modalities (ranging from roughly 51.8% to 62.8%) illustrates that the intrinsic correlation between polarity recognition and intensity regression subtasks highly rely on shared information. This aligns with the natural structure of sentiment, where the emotional sign (polarity) and its magnitude (intensity) often stem from the same core expressions, such as a smile or a distinct tonal shift. Besides, the existence of distinct ”Polarity-only” and ”Intensity-only” parts confirms that MoB effectively avoids feature redundancy in each subtask, isolating unique cues that would be otherwise confused in a purely joint latent space.

Compared to vision (62.8% shared) and audio (58.8% shared), language has the lowest shared ratio (51.8%) and the highest proportion of task-specific experts (24.1% for each). This indicates that the router assigns more fine-grained and distinctive features for language modality, leveraging its rich semantic capacity in explicitly decouple sentiment polarity from strength. In contrast, audio and vision modalities rely more on holistic and shared representations, suggesting that non-verbal cues tend to contribute to polarity and intensity simultaneously rather than independently.

## I. Dynamics in Bottleneck Routing Fusion

We visualize the gating weights of the experts in MBRF in Figure 5, indicating MoB’s sophisticated capacity in managing cross-modal interactions by dynamically assigning diverse experts to model different types of information flows, ranging from self-modal preservation to complex multimodal fusion. Instead of integrating features from different modalities jointly, which often leads to noise and interference due to modality gap, the router specializes experts to handle specific interaction pathways. This decoupling pattern confirms that the framework effectively minimizes mutual interference among diverse cross-modal synergy, which ensures that the unique characteristics of each modality are preserved while simultaneously enabling effective fusion process.

![](images/1488cad0e78bc39e85da72a141ac659506a38f0b56d919aa371af48c5d7fc967.jpg)  
Fig. 5. Visualization on the gating weight of informative unimodal latents routing during training for multimodal bottleneck routing fusion module..

From Figure 5, we can observe that some experts are dedicated to refining unimodal signals, while others are responsible for capturing the synergy between modalities, illustrating the fine-grained specialization in distinguishing unimodal and cross-modal dynamics. For instance, Expert 8 acts as a clear language specialist, assigning dominant weights to language modality $( B _ { p } ^ { l } \ 0 . 3 3 , B _ { i } ^ { l } \ 0 . 2 7 )$ while suppressing audio and visual inputs, thereby safeguarding the semantic core for textual information. Similarly, Expert 9 focuses heavily on acoustic cues $( B _ { p } ^ { a } \ 0 . 3 4 , \ B _ { i } ^ { a } \ 0 . 3 2 )$ , isolating the acoustic information. In contrast, the router assigns Expert 11 to handle complex relationship between facial expressions and voice tone, as indicated by its simultaneous high weights on both visual $( B _ { p } ^ { v }$ 0.14, B<sup>v</sup> 0.14) and audio $( B _ { p } ^ { a } \ 0 . 1 7 , \ B _ { i } ^ { a } \ 0 . 1 6 )$ latents. This proves that the MBRF automatically learns to allocate specific experts to capture the interaction across verbal and non-verbal signals, distinguishing cross-modal synergy from independent unimodal features.

Ultimately, the observed patterns of highly specialized yet balanced expert engagement across modalities confirm that the Information Bottleneck principle, when integrated with Mixture-of-Experts routing, successfully achieving the desired dynamic multimodal learning scheme for interpretability and adaptability of video-based MSA.

## V. CONCLUSION

In this paper, we present Mixture-of-Bottleneck (MoB) framework, a novel approach for dynamic and comprehensive video-based multimodal sentiment analysis. By reformulating sentiment into an ordinal space, MoB explicitly decouples polarity recognition and intensity estimation subtasks. With integration of the IB and MoE routing, MoB enables each expert to specialize in extracting compact and task-relevant information from different modalities. The designed multimodal bottleneck routing fusion further captures fine-grained crossmodal synergy adaptively. Extensive experiments validate its superior performance and better interpretability.

## REFERENCES

[1] P. P. Liang, A. Zadeh, and L.-P. Morency, “Foundations & trends in multimodal machine learning: Principles, challenges, and open questions,” ACM Comput. Surv., apr 2024, just Accepted.

[2] A. Pandey and D. K. Vishwakarma, “Progress, achievements, and challenges in multimodal sentiment analysis using deep learning: A survey,” Applied Soft Computing, vol. 152, p. 111206, 2024.

[3] S. Poria, D. Hazarika, N. Majumder, and R. Mihalcea, “Beneath the tip of the iceberg: Current challenges and new directions in sentiment analysis research,” IEEE Transactions on Affective Computing, 2020.

[4] N. Stoehr, R. Cotterell, and A. Schein, “Sentiment as an ordinal latent variable,” in Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, A. Vlachos and I. Augenstein, Eds. Dubrovnik, Croatia: Association for Computational Linguistics, May 2023, pp. 103–115.

[5] D. Gkoumas, Q. Li, C. Lioma, Y. Yu, and D. Song, “What makes the difference? an empirical comparison of fusion strategies for multimodal language analysis,” Information Fusion, vol. 66, pp. 184–197, 2021.

[6] Z. Han, F. Yang, J. Huang, C. Zhang, and J. Yao, “Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 20 675–20 685.

[7] Z. Xue and R. Marculescu, “Dynamic multimodal fusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2023, pp. 2575–2584.

[8] X. Xiao, G. Liu, G. Gupta, D. Cao, S. Li, Y. Li, T. Fang, M. Cheng, and P. Bogdan, “Neuro-inspired information-theoretic hierarchical perception for multimodal learning,” in The Twelfth International Conference on Learning Representations, 2024.

[9] P. Wang, Q. Zhou, Y. Wu, T. Chen, and J. Hu, “Dlf: Disentangledlanguage-focused multimodal sentiment analysis,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 20, pp. 21 180– 21 188, Apr. 2025.

[10] R. Lin, Y. Zeng, S. Mai, and H. Hu, “End-to-end semanticcentric video-based multimodal affective computing,” arXiv preprint arXiv:2408.07694, 2024.

[11] N. Tishby and N. Zaslavsky, “Deep learning and the information bottleneck principle,” in 2015 IEEE Information Theory Workshop (ITW), 2015, pp. 1–5.

[12] P. P. Liang, Z. Deng, M. Q. Ma, J. Y. Zou, L.-P. Morency, and R. Salakhutdinov, “Factorized contrastive learning: Going beyond multiview redundancy,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 32 971– 32 998.

[13] S. Mai, Y. Zeng, and H. Hu, “Multimodal information bottleneck: Learning minimal sufficient unimodal and multimodal representations,” IEEE Transactions on Multimedia, vol. 25, pp. 4121–4134, 2023.

[14] P. P. Liang, Y. Cheng, X. Fan, C. K. Ling, S. Nie, R. Chen, Z. Deng, N. Allen, R. Auerbach, F. Mahmood, R. R. Salakhutdinov, and L.- P. Morency, “Quantifying &amp; modeling multimodal interactions: An information decomposition framework,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 27 351–27 393.

[15] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” 2017.

[16] H. Yu, Z. Qi, L. K. Jang, R. Salakhutdinov, L.-P. Morency, and P. P. Liang, “MMoE: Enhancing multimodal models with mixtures of multimodal interaction experts,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 10 006–10 030.

[17] X. Han, H. Nguyen, C. Harris, N. Ho, and S. Saria, “Fusemoe: mixture-of-experts transformers for fleximodal fusion,” in Proceedings of the 38th International Conference on Neural Information Processing Systems, ser. NIPS ’24. Red Hook, NY, USA: Curran Associates Inc., 2024.

[18] A. Zadeh, P. P. Liang, S. Poria, P. Vij, E. Cambria, and L.-P. Morency, “Multi-attention recurrent network for human communication comprehension,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, Apr. 2018.

[19] J. Yang, Y. Yu, D. Niu, W. Guo, and Y. Xu, “ConFEDE: Contrastive feature decomposition for multimodal sentiment analysis,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 7617–7630.

[20] W. Yu, H. Xu, Z. Yuan, and J. Wu, “Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, 2021, pp. 10 790–10 797.

[21] W. Han, H. Chen, and S. Poria, “Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp. 9180–9192.

[22] G. Hu, T.-E. Lin, Y. Zhao, G. Lu, Y. Wu, and Y. Li, “UniMSE: Towards unified multimodal sentiment analysis and emotion recognition,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Y. Goldberg, Z. ozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 7837–7851.

[23] Z. Wu, Z. Gong, J. Koo, and J. Hirschberg, “Multimodal multi-loss fusion network for sentiment analysis,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 3588–3602.

[24] R. Lin and H. Hu, “Multimodal contrastive learning via uni-modal coding and cross-modal prediction for multimodal sentiment analysis,” in Findings of the Association for Computational Linguistics: EMNLP 2022, Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 511–523.

[25] ——, “Multi-task momentum distillation for multimodal sentiment analysis,” IEEE Transactions on Affective Computing, pp. 1–18, 2023.

[26] H. Zhang, Y. Wang, G. Yin, K. Liu, Y. Liu, and T. Yu, “Learning language-guided adaptive hyper-modality representation for multimodal sentiment analysis,” in Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 756–767.

[27] T. Sun, Y. Wei, J. Ni, Z. Liu, X. Song, Y. Wang, and L. Nie, “Mutimodal emotion recognition via hierarchical knowledge distillation,” IEEE Transactions on Multimedia, vol. 26, pp. 9036–9046, 2024.

[28] X. Zhang, W. Wei, and S. Zou, “Modal feature optimization network with prompt for multimodal sentiment analysis,” in Proceedings of the 31st International Conference on Computational Linguistics, O. Rambow, L. Wanner, M. Apidianaki, H. Al-Khalifa, B. D. Eugenio, and S. Schockaert, Eds. Abu Dhabi, UAE: Association for Computational Linguistics, Jan. 2025, pp. 4611–4621.

[29] R. Lin and H. Hu, “Dynamically shifting multimodal representations via hybrid-modal attention for multimodal sentiment analysis,” IEEE Transactions on Multimedia, vol. 26, pp. 2740–2755, 2023.

[30] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186.

[31] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Ro{bert}a: A robustly optimized {bert} pretraining approach,” 2020.

[32] P. He, J. Gao, and W. Chen, “DeBERTav3: Improving deBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing,” in The Eleventh International Conference on Learning Representations, 2023.

[33] N. Tishby, F. C. Pereira, and W. Bialek, “The information bottleneck method,” in Proc. of the 37-th Annual Allerton Conference on Communication, Control and Computing, 1999, pp. 368–377.

[34] Z. Xie, Y. Yang, J. Wang, X. Liu, and X. Li, “Trustworthy multimodal fusion for sentiment analysis in ordinal sentiment space,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 8, pp. 7657–7670, 2024.

[35] Z. Gao, X. Jiang, X. Xu, F. Shen, Y. Li, and H. T. Shen, “Embracing unimodal aleatoric uncertainty for robust multimodal fusion,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 26 866–26 875.

[36] Y. Zhang, P. E. Latham, and A. Saxe, “Understanding unimodal bias in multimodal deep linear networks,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[37] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Y. Zhao, A. Dai, Z. Chen, Q. Le, and J. Laudon, “Mixture-of-experts with expert choice routing,” in Proceedings ofthe 36th International Conference on Neural Information Processing Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

[38] B. Lin, Z. Tang, Y. Ye, J. Huang, J. Zhang, Y. Pang, P. Jin, M. Ning, J. Luo, and L. Yuan, “Moe-llava: Mixture of experts for large visionlanguage models,” IEEE Transactions on Multimedia, pp. 1–14, 2026.

[39] G. N. Yannakakis, R. Cowie, and C. Busso, “The ordinal nature of emotions: An emerging approach,” IEEE Transactions on Affective Computing, vol. 12, no. 1, pp. 16–35, 2021.

[40] M. K. Tellamekala, S. Amiriparian, B. W. Schuller, E. Andre, T. Gies-´ brecht, and M. Valstar, “Cold fusion: Calibrated and ordinal latent distribution fusion for uncertainty-aware multimodal emotion recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 2, pp. 805–822, 2024.

[41] L. Tian, C. Lai, and J. Moore, “Polarity and intensity: the two aspects of sentiment analysis,” in Proceedings of Grand Challenge and Workshop on Human Multimodal Language (Challenge-HML), A. Zadeh, P. P. Liang, L.-P. Morency, S. Poria, E. Cambria, and S. Scherer, Eds. Melbourne, Australia: Association for Computational Linguistics, Jul. 2018, pp. 40–47.

[42] A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy, “Deep variational information bottleneck,” in International Conference on Learning Representations, 2017.

[43] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” in 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

[44] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, “Scaling vision with

sparse mixture of experts,” in Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., vol. 34. Curran Associates, Inc., 2021, pp. 8583– 8595.

[45] H. Chen, K. Zhang, H. Tan, Z. Xu, F. Luan, L. Guibas, G. Wetzstein, and S. Bi, “Gaussian mixture flow matching models,” in Forty-second International Conference on Machine Learning, 2025.

[46] H. Liang, Z. Fan, R. Sarkar, Z. Jiang, T. Chen, K. Zou, Y. Cheng, C. Hao, and Z. Wang, “M3vit: mixture-of-experts vision transformer for efficient multi-task learning with model-accelerator co-design,” in Proceedings of the 36th International Conference on Neural Information Processing Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

[47] T. Chen, X. Chen, X. Du, A. Rashwan, F. Yang, H. Chen, Z. Wang, and Y. Li, “Adamv-moe: Adaptive multi-task vision mixture-of-experts,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 17 300–17 311.

[48] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016.

[49] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[50] A. Zadeh, R. Zellers, E. Pincus, and L.-P. Morency, “Mosi: multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos,” arXiv preprint arXiv:1606.06259, 2016.

[51] A. Zadeh, P. P. Liang, S. Poria, E. Cambria, and L.-P. Morency, “Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Melbourne, Australia: Association for Computational Linguistics, Jul. 2018, pp. 2236–2246.

[52] W. Yu, H. Xu, F. Meng, Y. Zhu, Y. Ma, J. Wu, J. Zou, and K. Yang, “CH-SIMS: A Chinese multimodal sentiment analysis dataset with finegrained annotation of modality,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, Jul. 2020, pp. 3718–3727.

[53] Y. Liu, Z. Yuan, H. Mao, Z. Liang, W. Yang, Y. Qiu, T. Cheng, X. Li, H. Xu, and K. Gao, “Make acoustic and visual cues matter: Ch-sims v2.0 dataset and av-mixup consistent module,” in Proceedings of the 2022 International Conference on Multimodal Interaction, ser. ICMI ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 247–258.

[54] L. Sun, Z. Lian, B. Liu, and J. Tao, “Efficient multimodal transformer with dual-level feature restoration for robust multimodal sentiment analysis,” IEEE Transactions on Affective Computing, vol. 15, no. 1, pp. 309–325, 2024.

[55] X. Feng, Y. Lin, L. He, Y. Li, L. Chang, and Y. Zhou, “Knowledgeguided dynamic modality attention fusion framework for multimodal sentiment analysis,” in Findings of the Association for Computational Linguistics: EMNLP 2024, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 14 755–14 766.

[56] C. Fan, K. Zhu, J. Tao, G. Yi, J. Xue, and Z. Lv, “Multi-level contrastive learning: Hierarchical alleviation of heterogeneity in multimodal sentiment analysis,” IEEE Transactions on Affective Computing, vol. 16, no. 1, pp. 207–222, 2025.

[57] Y. Zhuang, W. Bai, Y. Zhang, J. Deng, Z. Hu, X. Zhang, and F. Ren, “Multi-level contrastive learning for multimodal sentiment analysis,” IEEE Transactions on Multimedia, vol. 27, pp. 9044–9058, 2025.

[58] P. He, X. Liu, J. Gao, and W. Chen, “{DEBERTA}: {DECODING}- {enhanced} {bert} {with} {disentangled} {attention},” in International Conference on Learning Representations, 2021.

[59] Y. Cui, W. Che, T. Liu, B. Qin, and Z. Yang, “Pre-training with whole word masking for chinese bert,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3504–3514, 2021.

[60] Y. Cui, W. Che, T. Liu, B. Qin, S. Wang, and G. Hu, “Revisiting pretrained models for Chinese natural language processing,” in Findings of the Association for Computational Linguistics: EMNLP 2020, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 657–668.

[61] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “Imagebind: One embedding space to bind them all,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 15 180–15 190.