# Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction

Han-Jun Choi\* and Byunggill Joe\* and Saim Shin and Jin Yea Jang<sup>†</sup>

Korea Electronics Technology Institute (KETI)

Seongnam, South Korea

{hanjun\_c, byunggill, sishin, jinyea.jang}@keti.re.kr

## Abstract

Recent multimodal sentiment analysis studies increasingly adopt text-centric fusion approaches to exploit the rich sentiment information inherent in the textual modality. However, these approaches often suffer from performance degradation during inference due to partially missing or noisy data in real-world scenarios, especially when sentiment-related cues are missing. To address this issue, we introduce a new completeness estimation approach that quantifies the degree of sentiment-relevant information preserved in incomplete data to guide the reconstruction of missing semantics. Furthermore, we propose a training strategy that stabilizes multi-task learning while jointly optimizing sentiment prediction and completeness estimation. Extensive experiments and in-depth analyses on three benchmark datasets demonstrate that the proposed approach enables more accurate semantic reconstruction, leading to more precise sentiment prediction.

## 1 Introduction

Building on the rich sentiment-relevant information in text (Hazarika et al., 2022; Wei et al., 2023), recent Multimodal Sentiment Analysis (MSA) studies have increasingly explored text-centric approaches (Han et al., 2021; Wang et al., 2023a,b; Zhang et al., 2023) that treat text as the dominant modality while considering audio and vision modalities as auxiliary sources. Although these approaches have shown strong performance with complete data, they suffer substantial performance degradation when modalities are partially missing or noisy, particularly when the textual modality is incomplete.

To address this issue, several reconstructionbased approaches have been proposed (Yuan et al., 2021; Zhang et al., 2024a; Zhu et al., 2025). In particular, Zhang et al. (2024a) proposed a missing rate-based completeness estimation method to utilize the missing rate as a reconstruction weight, assuming that a higher missing rate indicates lower completeness, and vice versa. However, we argue that relying solely on the missing rate may fail to accurately capture semantic informativeness.

![](images/558e1ac0231fa65bf0e851be821e3d764dcf8d27067dc2486aa876bda681351b.jpg)  
I real y lo ve this movieFigure 1: Comparison between missing rate–based and semantic–based completeness. The former is computed as 1 − r, where r ∈ [0, 1) denotes the missing rate, while the latter is estimated by our proposed method.

Figure 1 highlights this limitation through two contrasting cases. In Case 1, a missing key sentiment cue [‘love’] leads to substantial semantic loss despite a low missing rate, causing an overestimation of missing rate-based completeness. On the other hand, in Case 2, it is underestimated even though the sentiment meaning remains intact. Such inaccurate completeness may lead to unreliable reconstruction, which highlights the necessity of a semantic-aware completeness approach that determines completeness based on whether key sentiment cues are missing.

Motivated by the above observations, we propose a semantic-aware completeness estimation approach for more reliable reconstruction of missing textual semantics. Specifically, we first estimate the completeness of incomplete text through a semantic completeness estimator. To supervise the estimator, we introduce Target Probability-based Semantic Completeness (TPSC), which is a pseudolabeling strategy that generates completeness labels. Meanwhile, the Importance-aware Proxy Feature Generator (IPFG) generates proxy features from the auxiliary modalities by adaptively weighting their contributions for text reconstruction. The estimated completeness is then used as a weighting factor to adaptively combine proxy features and incomplete text, producing reconstructed textual representations. Furthermore, we propose an Alternating Optimization Strategy (AOS) to mitigate gradient conflicts that stem from jointly optimizing the sentiment prediction and completeness estimation tasks on a shared encoder. The contributions of this paper are summarized as follows:

• To the best of our knowledge, we are the first to introduce a semantic-aware completeness estimation approach that quantifies the degree of semantic preservation in incomplete data.

• We propose an optimization strategy that mitigates gradient conflicts in hierarchical multitask learning, thereby stabilizing training between the completeness estimation and sentiment prediction tasks.

• Extensive experiments on three benchmark datasets with varying missing rates show that our proposed method consistently outperforms 12 competitive baselines, demonstrating the effectiveness of the proposed completeness label in real-world data scenarios.

## 2 Related Work

Recent studies in MSA have explored various multimodal fusion strategies to integrate textual, acoustic, and visual information (Zadeh et al., 2017; Liu et al., 2018; Tsai et al., 2019; Hazarika et al., 2020; Yu et al., 2021; Sun et al., 2022). These approaches aim to capture cross-modal interactions and leverage complementary information across modalities to improve sentiment prediction. Another line of work focuses on text-centric approaches (Han et al., 2021; Wang et al., 2023a,b; Zhang et al., 2023), which treat text as the dominant modality and utilize audio and visual signals as auxiliary sources to enhance textual representations. However, most existing methods assume that all modalities are fully available during inference, which limits their robustness in real-world scenarios. This issue becomes particularly critical for text-centric approaches, as the loss of important semantic information in text directly undermines the effectiveness of text-centric fusion.

To mitigate the above issue, recent studies have explored reconstruction-based approaches that aim to restore semantic information from missing modalities (Yuan et al., 2021; Lin and Hu, 2023; Zhang et al., 2024a; Zhu et al., 2025; Yang et al., 2025; Shi et al., 2025; Li et al., 2025; Lin et al., 2025). In particular, LNLN (Zhang et al., 2024a) introduces a reconstruction method that restores missing textual semantics using proxy features derived from audio and vision inputs, where the reconstruction weight is estimated from the missing rate of the incomplete text. P-RMF (Zhu et al., 2025) introduces a proxy-driven strategy that dynamically reconstructs incomplete multimodal inputs through cross-modal feature generation and adaptive fusion, further improving robustness under uncertain missing conditions. Additionally, TF-Mamba (Li et al., 2025) proposes a text-enhanced Mamba-based framework that reconstructs missing textual semantics by aligning and enhancing auxiliary modalities through a text-aware modality enhancement module while modeling multimodal dependencies via efficient Mamba-based fusion. While these methods improve robustness to missing modalities, they do not explicitly account for the semantic information loss in incomplete text.

## 3 Proposed Method

## 3.1 Problem Definition

Given multimodal inputs from text (t), audio (a), and vision (v) derived from the same utterance, the MSA task aims to predict a continuous sentiment score $y \in \mathbb { R }$

## 3.2 Framework Overview

In this work, we introduce a Text Completenessbased Missing Reconstruction (TCMR) framework that estimates textual completeness from incomplete text and reconstructs the missing semantic information by adaptively weighting proxy and incomplete text features based on the estimated completeness.

The overall workflow of TCMR is illustrated in Figure 2. First, each modality input is randomly masked and then embedded into feature representations via modality-specific encoders ( 1 ). Next, a completeness estimator predicts how much semantic information remains in incomplete text and reconstructs missing semantics using proxy features derived from auxiliary modalities, weighted by the predicted completeness ( 2 – 4 ). Lastly, a multimodal fusion module performs text-centric fusion to produce the final sentiment score ( 5 ). Note that the multimodal fusion module is adopted from LNLN, as designing a new fusion module is not the main objective of this work (see Appendix A.4).

![](images/f11a286bbf7af311cfe1aa5fff43e48a919b6c114fd2f368e4541da63e9918ff.jpg)  
Figure 2: Overview of the proposed framework. Models in the light purple boxes are used only during pretraining or pseudo-labeling and are not part of the TCMR training pipeline.

## 3.3 Input Processing

Raw multimodal inputs are processed into highlevel embeddings (Figure 2 1 ). First, each complete multimodal input $I _ { m } ^ { c }$ for modality $m \in$ $\{ v , a , t \}$ , is processed by modality-specific feature extractors: BERT for text, Librosa for audio, and OpenFace for vision (Devlin et al., 2019; McFee et al., 2015; Baltrušaitis et al., 2016).

Following prior works (Zhang et al., 2024a; Zhu et al., 2025), we adopt a random missing scenario to generate incomplete features $X _ { m } \in \mathbf { \mathbb { R } } ^ { T _ { m } \times d _ { m } }$ where $T _ { m }$ and $d _ { m }$ denote the sequence length and feature dimension. For text, a proportion of tokens in $I _ { t } ^ { c }$ are randomly replaced with [UNK] according to a missing rate sampled from [0, 1). For audio and vision features, a proportion of the sequence is replaced with zero vectors. Subsequently, the incomplete features are processed by modality-specific Transformer encoders (Vaswani et al., 2017) to obtain the high-level embeddings $H _ { m } \in \mathbb { R } ^ { T \times d }$

$$
H _ { m } = \mathrm { T r a n s f o r m e r } ( X _ { m } ; \theta _ { m } ^ { \mathrm { t r a n s } } )\tag{1}
$$

## 3.4 Missing Semantic Reconstruction

In this section, we elaborate on the completeness–guided semantic reconstruction pipeline of

TCMR (Figure $2 \textcircled{2} - \textcircled{4} )$ ), which is the core contribution of our work.

Text Completeness Estimation To estimate the completeness of incomplete text, we design a semantic completeness estimator CompNet, which consists of fully connected layers followed by a sigmoid activation (Figure $2 \textcircled{2} - 1 )$ .

Given the BERT [CLS] embedding $S _ { t } \in \mathbb { R } ^ { d }$ CompNet estimates a completeness weight $\hat { w } \in$ [0, 1]:

$$
\hat { w } = \mathrm { C o m p N e t } ( S _ { t } ; \theta ^ { \mathrm { c o m p } } ) ,\tag{2}
$$

The model is trained to minimize mean squared error between the predicted score wˆ and the completeness label w:

$$
\mathcal { L } _ { \mathrm { c o m p } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left\| \boldsymbol { \hat { w } } ^ { ( i ) } - \boldsymbol { w } ^ { ( i ) } \right\| ^ { 2 } ,\tag{3}
$$

where i denotes the data sample index.

Pseudo-Label Generation In our study, we define textual completeness as the extent to which sentiment-related semantic information is preserved in incomplete text. Following this definition, we introduce a pseudo-labeling strategy to generate completeness labels w for training CompNet (Figure 2 2 –2).

To quantify the completeness, we extend the concept of True Class Probability (TCP) (Corbière et al., 2019), defined as the probability assigned to the target class. Our hypothesis is that if a classifier trained on complete text assigns a high TCP to a given complete text, the TCP for the corresponding incomplete text will remain high as long as sentiment-relevant tokens are preserved. Conversely, the TCP may substantially decrease if key sentiment tokens are missing.

Based on this insight, we design a sentiment polarity classifier that takes only the text modality as input. First, the classifier is trained on the complete text $I _ { t } ^ { c }$ :

$$
\hat { y } _ { \mathrm { s p } } = \mathrm { C l a s s i f i e r } ( \mathrm { B E R T ^ { c l s } } ( I _ { t } ^ { c } ) ; \theta ^ { \mathrm { c l s } } )\tag{4}
$$

$$
\mathcal { L } _ { \mathrm { c e } } = - \sum _ { c = 1 } ^ { C } y _ { s p } ^ { ( c ) } \log \hat { y } _ { s p } ^ { ( c ) }\tag{5}
$$

where $y _ { s p }$ denotes the polarity class obtained by discretizing the continuous sentiment score according to predefined thresholds. After training, we perform inference on incomplete text using the pretrained classifier, and the probability assigned to the ground-truth polarity class $y _ { s p }$ is used as the pseudo completeness label $w \in [ 0 , 1 ]$ :

$$
\begin{array} { r } { \boldsymbol { w } ^ { ( i ) } \triangleq \boldsymbol { p } \Big ( \boldsymbol { y } _ { \mathrm { s p } } ^ { ( i ) } \mid \mathrm { B E R T } ^ { \mathrm { c l s } } ( I _ { t } ) ; \boldsymbol { \theta } _ { \mathrm { p r e } } ^ { \mathrm { c l s } } \Big ) , } \end{array}\tag{6}
$$

where $\theta _ { \mathrm { p r e } } ^ { \mathrm { c l s } }$ are the pretrained classifier parameters. The definition of sentiment polarity classes and the choice of the optimal number of classes are provided in Appendix B.1 and Appendix B.2, respectively.

Proxy Feature Generation Prior work (Zhang et al., 2024a) generates proxy features without considering the relative importance of auxiliary modalities for reconstruction. However, the contributions of each modality to the reconstruction can vary under random missing conditions.

Therefore, we design an Importance-aware Proxy Feature Generator (IPFG), which consists of Transformer-based modality-specific generators and an MLP-based gating network with a sigmoid:

$$
H _ { m ^ { \prime } } ^ { p } = \mathrm { G e n } ( H _ { m ^ { \prime } } ; \theta _ { m ^ { \prime } } ^ { \mathrm { g e n } } ) ,\tag{7}
$$

$$
\rho = \mathrm { G a t e } ( [ H _ { a } ^ { p } , H _ { v } ^ { p } ] ; \theta ^ { \mathrm { g a t e } } ) ,\tag{8}
$$

$$
\bar { H } _ { t } = \rho \odot H _ { a } ^ { p } + ( 1 - \rho ) \odot H _ { v } ^ { p } .\tag{9}
$$

where $m ^ { \prime } \in \{ a , v \}$ denotes the auxiliary modalities, ⊙ denotes element-wise multiplication, and $\rho \in [ 0 , 1 ]$ is a gating weight that adaptively balances the contributions of the auxiliary modalities.

Semantic Reconstruction The reconstructed text representation $\hat { H } _ { t } ~ \in ~ \mathbb { R } ^ { T \times d }$ is generated by integrating the incomplete text representation $H _ { t }$ with the generated proxy feature ${ \bar { H } } _ { t }$ , guided by the predicted completeness weight wˆ from CompNet:

$$
\begin{array} { r } { \hat { H } _ { t } = \hat { w } H _ { t } + ( 1 - \hat { w } ) \bar { H } _ { t } . } \end{array}\tag{10}
$$

To encourage the reconstructed features to be similar to the corresponding complete features, we minimize the mean squared error loss $\mathcal { L } _ { r e c } ^ { t } .$ . For the auxiliary modalities, following prior work (Zhang et al., 2024a), we adopt a self-reconstruction method that takes $H _ { m ^ { \prime } }$ as input and predicts the corresponding complete representation $\hat { H } _ { m ^ { \prime } }$ . The reconstruction losses are defined as:

$$
\mathcal { L } _ { \mathrm { r e c } } ^ { m } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left\| \hat { H } _ { m } ^ { ( i ) } - H _ { m } ^ { c , ( i ) } \right\| ^ { 2 } ,\tag{11}
$$

where $H _ { m } ^ { c , ( i ) }$ denotes the complete feature of modality m obtained from a model trained with the corresponding complete input.

## 3.5 Training Objective

TCMR predicts the final sentiment score $\hat { y }$ through the text-centric multimodal fusion module. The model is trained using the following loss:

$$
\mathcal { L } _ { \mathrm { t a s k } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left\| \boldsymbol { \hat { y } } ^ { ( i ) } - \boldsymbol { y } ^ { ( i ) } \right\| ^ { 2 } .\tag{12}
$$

The overall training objective is defined as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \alpha \mathcal { L } _ { \mathrm { c o m p } } + \beta \mathcal { L } _ { \mathrm { r e c } } ^ { t } + \gamma \mathcal { L } _ { \mathrm { r e c } } ^ { m ^ { \prime } } + \sigma \mathcal { L } _ { \mathrm { t a s k } } .\tag{13}
$$

where $\alpha , \beta , \gamma$ , and $\sigma$ are hyperparameters.

## 3.6 Alternating Optimization Strategy

We found that naively minimizing ${ \mathcal { L } } _ { \mathrm { t o t a l } }$ in an endto-end manner often leads to unstable training, where $\mathcal { L } _ { \mathrm { c o m p } }$ is effectively ignored and CompNet remains stuck at a poor local minimum. We attribute this behavior to gradient conflict (Yu et al., 2020a). This phenomenon commonly arises in multi-task learning, where the gradients of a dominant task suppress the optimization of relatively weaker tasks. In our case, $\mathcal { L } _ { \mathrm { t a s k } }$ can also be minimized through shortcut solutions that bypass learning completeness estimation, causing the optimization of $\mathcal { L } _ { \mathrm { c o m p } }$ to be dominated during training.

To address this issue, we introduce a simple yet effective training strategy called Alternating Optimization Strategy (AOS). We first partition the total objective $\mathcal { L } _ { \mathrm { t o t a l } }$ into two groups according to their respective roles:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { c o m p } } , \quad \mathcal { L } _ { \mathrm { o t h e r } } = \beta \mathcal { L } _ { \mathrm { r e c } } ^ { t } + \gamma \mathcal { L } _ { \mathrm { r e c } } ^ { m ^ { \prime } } + \sigma \mathcal { L } _ { \mathrm { t a s k } } . } \end{array}\tag{14}
$$

Algorithm 1 Alternating Optimization Strategy   
Input: dataset D, epochs E, batch size B, weights   
$\alpha , \beta , \gamma , \sigma .$   
Params: $\Theta ^ { \mathrm { c o m p } } = \theta ^ { \mathrm { c o m p } } \cup \theta ^ { \mathrm { b e r t } } .$ , Θ<sup>other</sup> for the re  
maining modules including $\theta ^ { \mathrm { { b e r t } } }$   
Opt: $\mathrm { O p t } _ { \mathrm { c o m p } } , \mathrm { O p t } _ { \mathrm { o t h e r } } .$   
1: for $e = 1$ to E do   
Phase 1: optimize $\Theta ^ { \mathrm { c o m p } }$ to minimize ${ \mathcal { L } } _ { \mathrm { c o m p } }$   
2: for mini-batch $B \subset D , | B | = B$ do   
3: $\mathcal { L } _ { \mathrm { c o m p } } \gets \mathrm { L o s s C o m p } ( \boldsymbol { B } ; \Theta ^ { \mathrm { c o m p } } )$   
4: $\mathcal { L } _ { \mathrm { c o m p } }  \alpha \mathcal { L } _ { \mathrm { c o m p } }$   
5: $\mathrm { O p t } _ { \mathrm { c o m p } } . \mathrm { s t e p } \big ( \mathrm { \nabla } \varphi \mathrm { c o m p } { \mathcal L } _ { \mathrm { c o m p } } \big )$   
6: end for   
Phase 2: optimize $\Theta ^ { \mathrm { o t h e r } }$ to minimize $\mathcal { L } _ { \mathrm { o t h e r } }$   
7: for mini-batch $B \subset D , | B | = B$ do   
8: $\mathcal { L } _ { \mathrm { t a s k } }  \mathrm { L o s s T a s k } ( \mathcal { B } ; \Theta ^ { \mathrm { o t h e r } } )$   
9: $\mathcal { L } _ { \mathrm { r e c } } ^ { t }  \mathrm { L o s s R e c T e x t } ( \mathcal { B } ; \Theta ^ { \mathrm { o t h e r } } )$   
10: $\mathcal { L } _ { \mathrm { r e c } } ^ { m ^ { \prime } }  \mathrm { L o s s R e c A u x } ( \mathcal { B } ; \Theta ^ { \mathrm { o t h e r } } )$   
11: $\mathcal { L } _ { \mathrm { o t h e r } }  \beta \mathcal { L } _ { \mathrm { r e c } } ^ { t } + \gamma \mathcal { L } _ { \mathrm { r e c } } ^ { m ^ { \prime } } + \sigma \mathcal { L } _ { \mathrm { t a s k } }$   
12: $\mathrm { O p t } _ { \mathrm { o t h e r } } . \mathrm { s t e p } ( \nabla _ { \Theta ^ { \mathrm { o t h e r } } } \mathcal { L } _ { \mathrm { o t h e r } } )$   
13: end for   
14: end for

As shown in Algorithm 1, AOS performs a twophase optimization within each training epoch. In Phase 1, we focus on completeness learning by minimizing $\mathcal { L } _ { \mathrm { { c o m p } } }$ , while fixing the remaining parameters and optimizing only $\Theta ^ { \mathrm { c o m p } }$ . In Phase 2, we fix the learned $\theta ^ { \mathrm { { c o m p } } }$ and optimize $\Theta ^ { \mathrm { o t h e r } }$ by minimizing $\mathcal { L } _ { \mathrm { o t h e r } } .$ . At this stage, the model focuses on reconstruction and sentiment prediction based on the completeness estimation learned in Phase 1. By alternately optimizing the two objectives, AOS stabilizes the training of CompNet and enables the remaining modules to be effectively trained according to their respective objectives. Figure 3 illustrates this alternating optimization process, and further experimental analysis of gradient conflicts is provided in Appendix C.

## 4 Experiments

## 4.1 Experimental Setup

Dataset and Evaluation Metrics We conduct experiments on three MSA benchmark datasets: MOSI (Zadeh et al., 2016), MOSEI (Bagher Zadeh et al., 2018), and SIMS (Yu et al., 2020b). In MOSI and MOSEI, sentiment scores are annotated with continuous values ranging from −3 (strongly negative) to +3 (strongly positive), while SIMS provides sentiment labels on a scale from −1 (negative) to +1 (positive). Model performance is evaluated using Acc-2, Acc-5, Acc-7, F1, MAE, and Corr on MOSI and MOSEI, and Acc-2, Acc-3, Acc-5, F1, MAE, and Corr on SIMS. Dataset statistics and metric definitions are provided in Appendix A.1 and Appendix A.3, respectively.

![](images/947352ad81af4e7312c1207fdf38d8c7a8debf81cd96f1bd10e8f47cd4ca182c.jpg)  
Figure 3: Illustration of the AOS optimization process.

Training and Evaluation Settings Following previous studies (Zhang et al., 2024a; Zhu et al., 2025), we adopt a partial random missing scenario. During training, each modality is randomly erased according to a missing rate r sampled from a uniform distribution over the range [0, 1.0). For each of the three seeds, the best model is selected based on its performance at a missing rate of $r = 0 . 5$ on the validation set. For testing, we conduct experiments across missing rates ranging from 0 to 0.9 with an increment of 0.1, resulting in ten evaluation settings for each seed.

## 4.2 Main Experimental Results

Baseline Models We select baseline models with publicly available implementations to ensure reproducibility. For a fair comparison, we reproduce all baseline models using publicly available implementations. For MISA, Self-MM, MMIM, CENet, TETFN, TFR-Net, and ALMT, we use the MMSA implementations (Mao et al., 2022). For LNLN<sup>1</sup>, $\mathrm { P - R M F } ^ { 2 }$ , and TF-Mamba<sup>3</sup>, we use their official repositories. In addition, we implement three variants of TCMR with different completeness labeling strategies to analyze their effectiveness:

• TCMR-TPSC: estimates completeness using Target Probability-based Semantic Completeness (TPSC), as defined in Section 3.4. Note that unless otherwise specified, TCMR refers to TCMR-TPSC for simplicity.

Table 1: Comparison of the overall performance on MOSI and MOSEI datasets.
<table><tr><td rowspan="2">Method</td><td colspan="5">MOSI</td><td colspan="5">MOSEI</td></tr><tr><td>Acc-7 Acc-5</td><td>Non0 Acc / F1</td><td></td><td>Has0 Acc / F1</td><td>MAE Corr</td><td>Acc-7</td><td>Acc-5 Non0 Acc / F1</td><td></td><td>Has0 Acc / F1</td><td>MAE Corr</td></tr><tr><td>MISA</td><td>29.03</td><td>31.61</td><td>68.77 / 68.67</td><td>67.94 / 67.72 1.1637</td><td>47.57</td><td>43.89</td><td>44.43 72.22 / 68.21</td><td></td><td>74.16 / 71.24 0.7346</td><td>43.57</td></tr><tr><td>Self-MM</td><td>30.38 33.69</td><td>68.83 / 68.63</td><td>68.65 / 69.34</td><td>1.1843</td><td>47.25</td><td>46.45 47.36</td><td>73.02 / 71.43</td><td>72.66 / 71.82</td><td>0.6818</td><td>53.68</td></tr><tr><td>MMIM</td><td>30.67</td><td>34.31 70.19 / 69.87</td><td>69.59 / 69.15</td><td>1.1649</td><td>49.01</td><td>44.89</td><td>45.42 74.29 / 73.19</td><td>73.46 / 73.19</td><td>0.7032</td><td>52.75</td></tr><tr><td>CENet</td><td>29.49</td><td>32.90</td><td>69.88 / 69.94</td><td>69.43 / 69.39 1.1801</td><td>48.24</td><td>47.36</td><td>48.24 77.01 / 77.26</td><td>75.96 / 76.23</td><td>0.6622</td><td>58.24</td></tr><tr><td>TETFN</td><td>29.86 32.56</td><td>70.62 / 70.67</td><td>69.85 / 69.79</td><td>1.1327</td><td>49.16</td><td>46.78</td><td>47.83 77.87 / 77.45</td><td>76.11 /76.37</td><td>0.6741</td><td>58.31</td></tr><tr><td>TFR-Net</td><td>28.22 30.31</td><td>70.88 / 70.73</td><td>70.18 / 69.93</td><td>1.1454</td><td>50.05</td><td>46.08</td><td>46.47 75.46 / 74.10</td><td>74.18 / 73.61</td><td>0.6784</td><td>56.24</td></tr><tr><td>ALMT</td><td>29.57 32.05</td><td>71.51 /71.48</td><td>70.52 / 70.39</td><td>1.1671</td><td>47.57</td><td>46.66</td><td>47.37 76.83 / 76.41</td><td>74.47 / 74.84</td><td>0.6749</td><td>56.45</td></tr><tr><td>LNLN</td><td>31.36</td><td>34.43</td><td>70.00 / 69.99 69.51 / 69.41</td><td>1.1450</td><td>48.08</td><td>46.36</td><td>47.14 77.79 /77.27</td><td>76.43 / 76.58</td><td>0.6698</td><td>58.17</td></tr><tr><td>P-RMF</td><td>28.91</td><td>31.24 69.91 / 69.57</td><td>69.28 / 68.84</td><td>1.1302</td><td>50.07</td><td>47.01</td><td>47.82 78.07 / 77.76</td><td>77.18 / 76.91</td><td>0.6667</td><td>58.45</td></tr><tr><td>TF-Mamba</td><td>28.85 31.46</td><td>71.05 / 71.11</td><td>70.73 / 70.69</td><td>1.1228</td><td>51.97</td><td>45.11</td><td>46.09 76.11 / 76.17</td><td></td><td>72.54 / 73.49 0.6891</td><td>57.51</td></tr><tr><td>TCMR-MRSC</td><td>30.39</td><td>33.36 70.84 / 70.95</td><td>69.81 / 69.82</td><td>1.1061</td><td>50.86</td><td>46.61</td><td>47.46 77.65 / 76.74</td><td></td><td>76.91 / 76.64 0.6673</td><td>58.46</td></tr><tr><td>TCMR-LDSC</td><td>31.63</td><td>35.38 70.61 / 70.55</td><td>69.83 / 69.58</td><td>1.1322</td><td>49.07</td><td>47.13</td><td>47.87 77.52 / 76.97</td><td></td><td>76.16 / 76.29</td><td>0.6641 58.51</td></tr><tr><td>TCMR-TPSC</td><td>32.80 36.73</td><td>72.01 / 71.67</td><td>70.93 / 70.49</td><td>1.0721</td><td>52.28</td><td>47.27</td><td>48.16 77.99 / 76.99</td><td>77.59 / 77.21</td><td></td><td>0.6614 58.63</td></tr></table>

<table><tr><td>Method</td><td>Acc-5</td><td>Acc-3</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>MISA</td><td>32.97</td><td>56.05</td><td>71.25</td><td>66.95</td><td>0.5731</td><td>0.319</td></tr><tr><td>Self-MM</td><td>33.57</td><td>55.82</td><td>69.02</td><td>68.38</td><td>0.5243</td><td>0.363</td></tr><tr><td>MMIM</td><td>30.33</td><td>55.04</td><td>69.46</td><td>66.74</td><td>0.5629</td><td>0.301</td></tr><tr><td>CENet</td><td>21.61</td><td>53.48</td><td>69.23</td><td>56.97</td><td>0.6304</td><td>0.014</td></tr><tr><td>TETFN</td><td>34.91</td><td>56.01</td><td>70.85</td><td>69.17</td><td>0.5439</td><td>0.342</td></tr><tr><td>TFR-Net</td><td>27.21</td><td>54.26</td><td>69.37</td><td>56.82</td><td>0.6258</td><td>0.091</td></tr><tr><td>ALMT</td><td>30.99</td><td>53.23</td><td>70.65</td><td>67.34</td><td>0.5445</td><td>0.311</td></tr><tr><td>LNLN</td><td>32.17</td><td>56.77</td><td>71.51</td><td>69.21</td><td>0.5423</td><td>0.345</td></tr><tr><td>P-RMF</td><td>32.72</td><td>57.30</td><td>70.58</td><td>70.36</td><td>0.5432</td><td>0.361</td></tr><tr><td>TF-Mamba</td><td>33.05</td><td>55.18</td><td>69.88</td><td>69.24</td><td>0.5235</td><td>0.359</td></tr><tr><td>TCMR-MRSC</td><td>33.27</td><td>55.46</td><td>70.03</td><td>68.76</td><td>0.5446</td><td>0.341</td></tr><tr><td>TCMR-TPSC</td><td>31.15</td><td>57.45</td><td>72.81</td><td>68.88</td><td>0.5212</td><td>0.379</td></tr></table>

Table 2: Comparison of the overall performance on the SIMS dataset. Note: TCMR-LDSC is excluded since it is based on an English sentiment lexicon.

• TCMR-MRSC: estimates completeness using Missing Rate-based Semantic Completeness (MRSC), which measures completeness according to the missing rate of text.

• TCMR-LDSC: To better analyze the effectiveness of TPSC, we design Lexicon-Derived Semantic Completeness (LDSC). LDSC generates completeness labels based on sentiment scores of individual tokens obtained from SentiWordNet (Baccianella et al., 2010). Specifically, we first compute the sentiment strength of the complete text by accumulating the sentiment scores of all tokens. Then, the same procedure is applied to the incomplete text. As a result, the completeness label is defined as the ratio between the two strengths. Further implementation details are provided in Appendix D.1.

Overall Performance Tables 1–2 present the evaluation results of various models on three MSA benchmark datasets. On the MOSI dataset, TCMR-TPSC achieves state-of-the-art performance across all evaluation metrics. In particular, TCMR-TPSC improves Acc-5 by 6.68% and reduces MAE by 6.37% compared to LNLN. Among the TCMR- variants, TCMR-TPSC consistently achieves the best performance across all datasets. This is because TCMR-MRSC struggles to capture the actual semantic loss from the text, as it relies solely on the missing rate. TCMR-LDSC, on the other hand, estimates completeness based on sentiment scores assigned to individual tokens, which does not consider contextual semantics and can lead to inaccurate completeness labels. These results further highlight the importance of semantic-aware completeness estimation. Further analyses with competitive baselines are provided in Appendix E.

![](images/86951ba239dbda3b6b7e6d0c25300cebcb746ce6db9173bcb0d1136e44c0c92a.jpg)  
Figure 4: Performance curves under various missing rates. (a)–(c): Accuracy on MOSI (Acc-7), MOSEI (Has0 Acc), and SIMS (Acc-2). (d)–(f): MAE on MOSI, MOSEI, and SIMS.

On the MOSEI and SIMS datasets, TCMR-TPSC consistently shows strong performance across most evaluation metrics. In particular, on MOSEI, it achieves the best results on Has0 Acc / F1, MAE, and Corr, while remaining competitive on Acc-7, Acc-5, and Non0 Acc / F1. On SIMS, it achieves the best results on Acc-3, Acc-2, MAE, and Corr. While TCMR-TPSC shows slightly lower performance on a few classification metrics compared to CENet and P-RMF, this can be attributed to the discretization involved in computing such metrics. For instance, Acc-7 converts continuous sentiment predictions into discrete classes via rounding. When a prediction lies close to a rounding boundary, it may fall into a different class than the ground truth, leading to misclassification despite a lower regression error. For this reason, regression metrics more accurately reflect the precision of sentiment prediction. A more detailed analysis is provided in Appendix E.1.

Table 3: A comprehensive ablation study of the proposed TCMR framework on the MOSI and MOSEI datasets.
<table><tr><td rowspan="2">Method</td><td colspan="5">MOSI</td><td colspan="5">MOSEI</td></tr><tr><td></td><td></td><td>Acc-7 Acc-5 Non0 Acc / F1 Has0 Acc / F1</td><td>MAE</td><td>Corr</td><td></td><td></td><td>Acc-7 Acc-5 Non0 Acc / F1 Has0 Acc / F1</td><td></td><td>MAE Corr</td></tr><tr><td>w/o AOS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CompNet-first</td><td>27.67</td><td>30.65</td><td>66.88 / 66.84</td><td>66.45 / 66.29 1.1671</td><td>43.06</td><td>43.33</td><td>43.48</td><td>65.37 / 64.09</td><td>69.14 / 65.16</td><td>0.7827 32.61</td></tr><tr><td>End2End</td><td>29.68</td><td>32.07</td><td>68.15 / 68.04 67.58 / 67.36</td><td>1.1849</td><td>49.15</td><td>46.82 47.75</td><td>77.97 / 77.43</td><td>76.27 / 76.44</td><td>0.6619</td><td>58.71</td></tr><tr><td>CompNet-later</td><td>32.24</td><td>36.17</td><td>71.16 /71.22</td><td>70.41 / 70.37 1.0944</td><td>50.91</td><td>43.11</td><td>43.86</td><td>73.75 / 71.83</td><td>74.58 / 73.47 0.7547</td><td>46.85</td></tr><tr><td>w/o IPFG</td><td>31.31</td><td>34.81</td><td>69.02 / 68.91</td><td>68.65 / 68.42</td><td>1.1674 45.89</td><td>46.05</td><td>47.06</td><td>75.92 /75.80</td><td>72.71 /73.45</td><td>0.6816 57.22</td></tr><tr><td>TCMR (Full)</td><td></td><td>32.8036.73</td><td>72.01 / 71.67</td><td>70.93 / 70.49</td><td>1.0721 52.28</td><td>47.27</td><td>48.16</td><td>77.99 / 76.99</td><td>77.59 /77.21</td><td>0.6614 58.63</td></tr></table>

Table 4: A comprehensive ablation study of the proposed TCMR framework on the SIMS dataset.
<table><tr><td>Method</td><td>Acc-5 Acc-3</td><td></td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>w/o AOS</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CompNet-first</td><td>22.92</td><td>53.79</td><td>69.82</td><td>60.81</td><td></td><td>0.57640.171</td></tr><tr><td>End2End</td><td>29.26</td><td>53.92</td><td>68.57</td><td>67.71</td><td>0.5374</td><td>0.321</td></tr><tr><td>CompNet-later</td><td>29.71</td><td>55.49</td><td>70.04</td><td>68.48</td><td>0.5416</td><td>0.339</td></tr><tr><td>w/o IPFG</td><td>30.15</td><td>55.57</td><td>70.09</td><td>66.93</td><td>0.5451</td><td>0.336</td></tr><tr><td>TCMR (Full)</td><td>31.15</td><td>57.45</td><td>72.81</td><td>68.88</td><td>0.5212</td><td>0.379</td></tr></table>

To provide a comprehensive comparison with reconstruction-based approaches, we plot performance changes across missing rates in Figure 4. As shown in the figure, TCMR-TPSC consistently outperforms baseline methods, even as the missing rate increases. These results support our motivation that accurately estimating semantic information loss in incomplete text enables effective reconstruction of missing semantics, thereby leading to improved overall performance.

## 5 Ablation Study

As shown in Table 3 and Table 4, we conduct an ablation study on the MOSI, MOSEI, and SIMS datasets to analyze the contribution of each component. In the w/o AOS setting, we design several alternative multi-task optimization strategies. In CompNet-first, we first optimize Θ<sup>comp</sup>, and then optimize the remaining modules while freezing both the shared BERT encoder and the trained CompNet. In contrast, CompNet-later first optimizes Θ<sup>other</sup> using the completeness label w, and then optimizes θ<sup>comp</sup> while freezing the shared BERT encoder. End2End jointly optimizes all model parameters without any staged optimization. In the w/o IPFG setting, we replace IPFG with a simpler proxy feature generator following prior work (Zhang et al., 2024a). Specifically, instead of generating modality-specific proxy features and combining them through a gating mechanism, the audio and vision features are directly used as inputs to a single network that produces the proxy feature.

According to Table 3, CompNet-first consistently achieves the lowest performance across all datasets. One possible reason is that freezing the shared BERT encoder after completeness learning prevents it from being further adapted for sentiment prediction, leading to misalignment between the two objectives. This result highlights the necessity of continuously updating the shared BERT encoder across both tasks, while preserving the hierarchical structure from completeness learning to sentiment learning. In contrast, CompNet-later achieves relatively competitive performance with TCMR (Full) on MOSI, while it performs relatively worse on MOSEI. We attribute this discrepancy to the differences in scale and characteristics between the two datasets. Compared to MOSI, MOSEI contains substantially more samples and covers a wider range of topics, indicating that a broader range of sentiment expression patterns must be learned. As a result, learning completeness estimation by optimizing a

![](images/3a65c4f174a871b26fa9dc41facf4a57b4128e25984765e067eec3f848763061.jpg)

(a) MOSI  
![](images/06026ab0a7d715270303d548a08dbbfcb852a587f60480e428bb71f94bee2b6a.jpg)  
(b) MOSEI  
Figure 5: Comparison of error-triggering sentiment clue tokens across two benchmark datasets. Each histogram shows the frequency of missing tokens that cause misclassification in the sentiment classifier.

CompNet with relatively few parameters after the shared BERT encoder has already been optimized for ground-truth sentiment labels may be insufficient to capture the diverse patterns present in MO-SEI. In this regard, End2End shows relatively more competitive performance on MOSEI than on MOSI, which we attribute to MOSEI’s greater diversity reducing the influence of the dominant task, thereby naturally alleviating gradient conflict. Nevertheless, its inherent end-to-end nature still results in suboptimal performance compared to TCMR (Full). Removing IPFG consistently degrades performance across all datasets. This suggests that it adaptively regulates the contribution of each auxiliary modality for incomplete text reconstruction, effectively filtering redundant information while enhancing complementary semantic cues.

## 6 In-depth Analysis

Sentiment Token Sensitivity To further examine the effectiveness of the TPSC label, we conduct a sensitivity analysis on sentiment-related tokens. First, we collect samples that are correctly classified by a pretrained sentiment polarity classifier on complete textual inputs. For each sample, we construct a perturbation token set by masking each word token in turn. Then, we count the frequency of tokens whose masking causes a previously correct prediction to become incorrect (detailed procedures are provided in Appendix D.2).

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Key Semantic Cue “PRESENT&quot;</td><td rowspan=1 colspan=1>Key Semantic Cue “MISSING”</td></tr><tr><td rowspan=1 colspan=1>MissingRate“LOW&quot;</td><td rowspan=1 colspan=1>“and he was still boring&quot;LNLN: Negative✓TCMR: Negative✓</td><td rowspan=1 colspan=1>“uh huh, how about theacting andtheaction wasjust so dull&quot;LNLN: Positive TCMR: Negative✓</td></tr><tr><td rowspan=1 colspan=1>MissingRate“HIGH”</td><td rowspan=1 colspan=1>“and it&#x27;s truly heartbreakingtosee that contrasted with thestate of things&quot;LNLN: Positive  TCMR: NegativeV</td><td rowspan=1 colspan=1>&quot;There&#x27;re also two lord of the ringsgradswhich I absolutely love&quot;LNLN: Negative×TCMR: Positiveè</td></tr></table>

Figure 6: Qualitative comparison between the LNLN and TCMR models under varying missing-text conditions. Each cell presents an example utterance where key semantic cues are either present or missing.

As shown in Figure 5, misclassifications are concentrated on affective tokens such as [’boring’], [’not’], and [’fun’]. This observation supports our hypothesis that TCP indeed reflects the preservation of sentiment-relevant information, as discussed in Section 3.4. Consequently, CompNet provides a more accurate estimate of semantic completeness, enabling more reliable reconstruction by maximizing complementary semantic information from auxiliary modalities.

Case Study Figure 6 illustrates how LNLN and TCMR behave under different missing-text conditions. Overall, LNLN fails to accurately reconstruct missing semantics. This stems from overestimating or underestimating the semantic information remaining in incomplete text, leading to incorrect sentiment prediction. In contrast, TCMR produces correct predictions through more accurate completeness estimation. This allows TCMR to adaptively balance its reliance on auxiliary modalities based on the extent of semantic loss, resulting in a more faithful reconstruction. Notably, the bottomright cell provides an interesting insight. Although LNLN can estimate completeness accurately in this case, due to the high missing rate and the loss of a key sentiment token, it still fails to produce the correct sentiment prediction. This suggests that inaccurate completeness estimation during training may limit LNLN’s ability to reconstruct missing semantics. Therefore, accurately estimating the semantic information retained in incomplete text is crucial not only for effective reconstruction but also for stable sentiment prediction.

Comparison of Completeness Labels Table 5 provides a qualitative comparison of different completeness labeling strategies under random missing conditions. Cases 3 and 4 highlight the limitations of MRSC. Since MRSC estimates completeness solely based on the proportion of missing tokens, it cannot capture the actual semantic information loss in the incomplete text. In contrast, LDSC and TPSC more appropriately reflect completeness by considering the semantic importance of sentimentrelated tokens. Cases 5–7 reveal the limitations of LDSC. For example, in Cases 5 and 7, words such as [’fan’] and [’dig’] are used as positive sentiment expressions. However, as SentiWordNet assigns sentiment scores based on the most frequent sense of a word, the assigned score may differ from the sense used in the actual context. Moreover, in Case 6, [’better’] is assigned a positive sentiment score in SentiWordNet, leading LDSC to yield a high completeness value even though the overall sentence conveys negative sentiment. In contrast, TPSC produces relatively correct completeness in these challenging cases due to its ability to capture sentiment based on the context of the full sentence rather than individual words.

Table 5: Qualitative comparison of completeness labels. Each case shows how MRSC, LDSC, and TPSC assign completeness labels depending on whether sentiment-related tokens are missing. In the TPSC column, the value in parentheses denotes $P ( y _ { s p }$ | complete), while the value above it denotes $P ( y _ { s p }$ | incomplete). ✓ and ✗ indicate correct and incorrect completeness estimation, respectively.
<table><tr><td rowspan=1 colspan=1>Case</td><td rowspan=1 colspan=1>Complete Text</td><td rowspan=1 colspan=1>Incomplete Text</td><td rowspan=1 colspan=1>MRSC</td><td rowspan=1 colspan=1>LDSC</td><td rowspan=1 colspan=1>TPSC</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>I was happy to see it</td><td rowspan=1 colspan=1>I [UNK] happy to see it</td><td rowspan=1 colspan=1>0.8333√</td><td rowspan=1 colspan=1>0.9888√</td><td rowspan=1 colspan=1>0.9526(0.9580)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>The soundtrack is good its really goodsome really great songs</td><td rowspan=1 colspan=1>[UNK] soundtrack [UNK] [UNK][UNK] really [UNK] [UNK] [UNK][UNK] [UNK]</td><td rowspan=1 colspan=1>0.1818√</td><td rowspan=1 colspan=1>0.0044√</td><td rowspan=1 colspan=1>0.3077(0.9665)</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>But it was really really awesome</td><td rowspan=1 colspan=1>but it was [UNK] [UNK] [UNK]</td><td rowspan=1 colspan=1>0.5000X</td><td rowspan=1 colspan=1>0.0112√</td><td rowspan=1 colspan=1>0.3179(0.9680)</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Kids are gonna love the film</td><td rowspan=1 colspan=1>[UNK] [UNK] [UNK] love the [UNK]</td><td rowspan=1 colspan=1>0.3333X</td><td rowspan=1 colspan=1>0.9808√</td><td rowspan=1 colspan=1>0.8638(0.9571)</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>I will admit I&#x27;m a big Johnny Deppfan</td><td rowspan=1 colspan=1>I will [UNK] I&#x27;m [UNK] big JohnnyDepp fan</td><td rowspan=1 colspan=1>0.7931√</td><td rowspan=1 colspan=1>0.0690X</td><td rowspan=1 colspan=1>0.9160(0.9491)</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>and you’re better off saving yourmoney uh and maybe renting it whenit comes out on uh video so</td><td rowspan=1 colspan=1>and [UNK] better [UNK] saving [UNK]money [UNK] [UNK] maybe renting itwhen it [UNK] out [UNK] uh [UNK][UNK]</td><td rowspan=1 colspan=1>0.5921√</td><td rowspan=1 colspan=1>0.9928 X</td><td rowspan=1 colspan=1>0.4608(0.6153)</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>I really dig this movie</td><td rowspan=1 colspan=1>I [UNK] [UNK] this [UNK]</td><td rowspan=1 colspan=1>0.4000X</td><td rowspan=1 colspan=1>0.5000X</td><td rowspan=1 colspan=1>0.1006(0.9477)</td></tr></table>

## 7 Conclusion

In this work, we reinterpret semantic information loss in incomplete data based on the change in target-class probability between complete and incomplete data to address the performance degradation of text-centric fusion models in MSA caused by noisy or missing data. First, we generate pseudolabels that provide supervision for training a completeness estimator. We further propose an optimization strategy that mitigates gradient conflicts between the completeness estimation and sentiment prediction tasks, thereby stabilizing hierarchical multi-task learning. We believe the proposed completeness estimation approach can be broadly applied beyond MSA, such as confidence-aware decision making and low-quality data detection or restoration.

## Limitations

Although TCMR consistently outperforms existing reconstruction-based MSA models, several limitations remain to be addressed in future work. First, this study only focuses on reconstructing semantic information loss in the textual modality. Extending the TPSC-based completeness estimation to other modalities, such as audio and vision, could improve general multimodal fusion approaches beyond text-centric fusion. Second, TCMR relies on pseudo-labeling, which inherently introduces potential noise in the supervision signal. Future work could mitigate this issue by adopting confidenceaware filtering, using only high-confidence TPSC predictions as pseudo-labels while relying on complementary strategies such as LDSC for less confident samples. For samples where both TPSC and LDSC yield low confidence, human verification could be incorporated to further reduce label noise.

## Acknowledgments

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2022-II220608, RS-2022-II220320, and RS-2024-00398115); and the Korea Electronics Technology Institute (KETI) through the project “Development of Modular Humanoid Physical AI Technology for Performing Complex Tasks in Industrial Environments.”

## References

Stefano Baccianella, Andrea Esuli, and Fabrizio Sebastiani. 2010. SentiWordNet 3.0: An enhanced lexical resource for sentiment analysis and opinion mining. In Proceedings of the Seventh International Conference on Language Resources and Evaluation (LREC’10), Valletta, Malta. European Language Resources Association (ELRA).

AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018. Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph. In Proceedings ofthe 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2236–2246, Melbourne, Australia. Association for Computational Linguistics.

Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 2016. OpenFace: An open source facial behavior analysis toolkit. In 2016 IEEE Winter Conference on Applications ofComputer Vision (WACV), pages 1–10.

Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 2018. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 794–803. PMLR.

Charles Corbière, Nicolas THOME, Avner Bar-Hen, Matthieu Cord, and Patrick Pérez. 2019. Addressing failure prediction by learning model confidence. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Wei Han, Hui Chen, Alexander Gelbukh, Amir Zadeh, Louis-philippe Morency, and Soujanya Poria.

2021. Bi-bimodal modality fusion for correlationcontrolled multimodal sentiment analysis. In Proceedings of the 2021 International Conference on Multimodal Interaction, ICMI ’21, pages 6–15, New York, NY, USA. Association for Computing Machinery.

Devamanyu Hazarika, Yingting Li, Bo Cheng, Shuai Zhao, Roger Zimmermann, and Soujanya Poria. 2022. Analyzing modality robustness in multimodal sentiment analysis. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 685–696, Seattle, United States. Association for Computational Linguistics.

Devamanyu Hazarika, Roger Zimmermann, and Soujanya Poria. 2020. MISA: Modality-invariant and -specific representations for multimodal sentiment analysis. In Proceedings of the 28th ACM International Conference on Multimedia, MM ’20, pages 1122–1131, New York, NY, USA. Association for Computing Machinery.

Xiang Li, Xianfu Cheng, Dezhuang Miao, Xiaoming Zhang, and Zhoujun Li. 2025. TF-Mamba: Textenhanced fusion mamba with missing modalities for robust multimodal sentiment analysis. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 11252–11267, Suzhou, China. Association for Computational Linguistics.

Ronghao Lin, Qiaolin He, Sijie Mai, Ying Zeng, Aolin Xiong, Li Huang, Yap-Peng Tan, and Haifeng Hu. 2025. CyIN: Cyclic informative latent space for bridging complete and incomplete multimodal learning. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 17605– 17642. Curran Associates, Inc.

Ronghao Lin and Haifeng Hu. 2023. MissModal: Increasing robustness to missing modality in multimodal sentiment analysis. Transactions ofthe Associationfor Computational Linguistics, 11:1686–1702.

Zhun Liu, Ying Shen, Varun Bharadhwaj Lakshminarasimhan, Paul Pu Liang, AmirAli Bagher Zadeh, and Louis-Philippe Morency. 2018. Efficient lowrank multimodal fusion with modality-specific factors. In Proceedings ofthe 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2247–2256, Melbourne, Australia. Association for Computational Linguistics.

Huisheng Mao, Ziqi Yuan, Hua Xu, Wenmeng Yu, Yihe Liu, and Kai Gao. 2022. M-SENA: An integrated platform for multimodal sentiment analysis. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics: System Demonstrations, pages 204–213, Dublin, Ireland. Association for Computational Linguistics.

Brian McFee, Colin Raffel, Dawen Liang, Daniel P. W. Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. 2015. librosa: Audio and music signal analysis in

python. In Proceedings of the 14th Python in Science Conference, SciPy 2015, Austin, Texas, USA, July 6-12, 2015, pages 18–24. scipy.org.

Piao Shi, Min Hu, Satoshi Nakagawa, Xiangming Zheng, Xuefeng Shi, and Fuji Ren. 2025. Textguided reconstruction network for sentiment analysis with uncertain missing modalities. IEEE Transac tions on Affective Computing, 16(3):1825–1838.

Hao Sun, Hongyi Wang, Jiaqing Liu, Yen-Wei Chen, and Lanfen Lin. 2022. CubeMLP: An mlp-based model for multimodal sentiment analysis and depression estimation. In Proceedings of the 30th ACM International Conference on Multimedia, MM ’22, pages 3722–3729, New York, NY, USA. Association for Computing Machinery.

Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019. Multimodal transformer for unaligned multimodal language sequences. In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics, pages 6558– 6569, Florence, Italy. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Di Wang, Xutong Guo, Yumin Tian, Jinhui Liu, Li-Huo He, and Xuemei Luo. 2023a. TETFN: A text enhanced transformer fusion network for multimodal sentiment analysis. Pattern Recognition, 136:109259.

Di Wang, Shuai Liu, Quan Wang, Yumin Tian, Lihuo He, and Xinbo Gao. 2023b. Cross-modal enhancement network for multimodal sentiment analysis. IEEE Transactions on Multimedia, 25:4909–4921.

Yiwei Wei, Shaozu Yuan, Ruosong Yang, Lei Shen, Zhangmeizhi Li, Longbiao Wang, and Meng Chen. 2023. Tackling modality heterogeneity with multiview calibration network for multimodal sentiment detection. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5240–5252, Toronto, Canada. Association for Computational Linguistics.

Mingzheng Yang, Kai Zhang, Yuyang Ye, Yanghai Zhang, Runlong Yu, and Min Hou. 2025. Decoupling and reconstructing: A multimodal sentiment analysis framework towards robustness. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, pages 6803–6811. International Joint Conferences on Artificial Intelligence Organization. Main Track.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020a. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, pages 5824–5836. Curran Associates, Inc.

Wenmeng Yu, Hua Xu, Fanyang Meng, Yilin Zhu, Yixiao Ma, Jiele Wu, Jiyun Zou, and Kaicheng Yang. 2020b. CH-SIMS: A Chinese multimodal sentiment analysis dataset with fine-grained annotation of modality. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3718–3727, Online. Association for Computational Linguistics.

Wenmeng Yu, Hua Xu, Ziqi Yuan, and Jiele Wu. 2021. Learning modality-specific representations with selfsupervised multi-task learning for multimodal sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 10790–10797.

Ziqi Yuan, Wei Li, Hua Xu, and Wenmeng Yu. 2021. Transformer-based feature reconstruction network for robust multimodal sentiment analysis. In Proceedings of the 29th ACM International Conference on Multimedia, MM ’21, pages 4400–4407, New York, NY, USA. Association for Computing Machinery.

Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2017. Tensor fusion network for multimodal sentiment analysis. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1103–1114, Copenhagen, Denmark. Association for Computational Linguistics.

Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. 2016. Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages. IEEE Intelligent Systems, 31(6):82–88.

Haoyu Zhang, Wenbin Wang, and Tianshu Yu. 2024a. Towards robust multimodal sentiment analysis with incomplete data. In Advances in Neural Information Processing Systems, volume 37, pages 55943–55974. Curran Associates, Inc.

Haoyu Zhang, Yu Wang, Guanghao Yin, Kejun Liu, Yuanyuan Liu, and Tianshu Yu. 2023. Learning language-guided adaptive hyper-modality representation for multimodal sentiment analysis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 756–767, Singapore. Association for Computational Linguistics.

Zhi Zhang, Jiayi Shen, Congfeng Cao, Gaole Dai, Shiji Zhou, Qizhe Zhang, Shanghang Zhang, and Ekaterina Shutova. 2024b. Proactive gradient conflict mitigation in multi-task learning: A sparse training perspective. arXiv preprint arXiv:2411.18615.

Aoqiang Zhu, Min Hu, Xiaohua Wang, Jiaoyun Yang, Yiming Tang, and Ning An. 2025. Proxy-driven robust multimodal sentiment analysis with incomplete data. In Proceedings ofthe 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 22123–22138, Vienna, Austria. Association for Computational Linguistics.

## A Implementation Details

## A.1 Dataset Statistics

Table 6: Statistics of the MOSI, MOSEI, and SIMS datasets used in our experiments.
<table><tr><td></td><td>MOSI</td><td>MOSEI</td><td>SIMS</td></tr><tr><td>#Samples</td><td>2,199</td><td>22,856</td><td>2,281</td></tr><tr><td>Train / Val / Test 1,284 / 229 / 686</td><td></td><td>16,326 / 1,871 / 4,659</td><td>1,368 / 456 / 457</td></tr><tr><td>#Speakers</td><td>93</td><td>1,000+</td><td></td></tr><tr><td>#Topics</td><td>89</td><td>250+</td><td></td></tr><tr><td>Source</td><td>YouTube</td><td>YouTube</td><td>Movies / TV</td></tr><tr><td>Annotation Type</td><td>Utterance-level</td><td>Utterance-level</td><td>Utterance-level</td></tr><tr><td>Sentiment Score</td><td>-3 to +3</td><td>-3 to +3</td><td>-1 to +1</td></tr></table>

Table 6 provides key statistics of the three benchmark datasets used in our experiments. Among them, MOSEI stands out as substantially larger and more diverse in speakers and topics than MOSI and SIMS, resulting in greater variability across acoustic, visual, and textual modalities. Due to this diversity, MOSEI is considered a more challenging benchmark than MOSI and SIMS.

## A.2 Model Configuration

The proposed TCMR model was implemented in PyTorch (v2.2.1) with Python 3.11.7 and trained on a workstation with an NVIDIA GeForce RTX 4090 GPU and an AMD Ryzen Threadripper PRO 5975WX CPU. To offer a thorough specification of our architecture design, Table 7 provides the detailed configurations of all modules within TCMR.

## A.3 Evaluation Metrics

Following prior work (Zhang et al., 2024a), we adopt the same evaluation metrics. For MOSI and MOSEI, we report both classification metrics (Acc-7, Acc-5, Non0 Acc / F1, Has0 Acc / F1) and regression metrics (MAE, Corr). The classification metrics are computed by discretizing the continuous sentiment scores into predefined sentiment intervals. For SIMS, we report Acc-2, Acc-3, Acc-5, F1, MAE, and Corr. Non0 Acc denotes negative/positive classification (excluding zero-valued samples), whereas Has0 Acc denotes negative/nonnegative classification, where zero-valued samples are treated as non-negative.

## A.4 Multimodal Fusion Module

The multimodal fusion module follows the design of prior work (Zhang et al., 2024a), and its key components are briefly summarized here. Starting from the reconstructed text representation ${ \hat { H } } _ { t } ,$ a refinement Transformer encoder applies self-attention, producing layerwise refined features $H _ { t } ^ { + ( i ) }$ with $i \in \{ 1 , \ldots , l ^ { \mathrm { r e f } } \}$ . Based on $H _ { t } ^ { + ( i ) }$ , cross-modal attention integrates complementary cues from vision and audio across multiple layers. At each layer, the current refined text feature $H _ { t } ^ { + ( i ) }$ serves as the query and attends to $H _ { v }$ and $H _ { a } ,$ while the fused representation updates $H _ { f } ^ { ( i ) }$ through residual accumulation. In compact form, $H _ { f } ^ { ( i ) } = H _ { f } ^ { ( i - 1 ) } +$ $\mathrm { M H A } ( H _ { t } ^ { + ( i ) } , H _ { v } ) + \mathrm { M H A } ( H _ { t } ^ { + ( i ) } , H _ { a } )$ where $H _ { f } ^ { 0 }$ is a learnable embedding and MHA(Q, K) denotes multi-head attention with query Q and key and value from K. A CrossTransformer then models interactions between $[ H _ { f } ^ { ( l ^ { \mathrm { r e f } } ) } , H _ { t } ^ { + ( l ^ { \mathrm { r e f } } ) } ]$ , and a regression head outputs the final sentiment prediction yˆ.

## B Sentiment Polarity Classes

## B.1 Definition of Sentiment Polarity Classes

To utilize TCP, we convert continuous sentiment scores into discrete polarity classes. As the score ranges differ across datasets, the class boundaries are defined separately for SIMS and MOSI/MOSEI.

For the SIMS dataset, polarity classes are defined using fixed interval thresholds. Specifically, the 2-class setting divides the scores into $[ - 1 . 0 , 0 . 0 ]$ and (0.0, 1.0]. The 3-class setting introduces a neutral region, resulting in [−1.0, −0.1], (−0.1, 0.1], and (0.1, 1.0]. The 5-class setting adopts a finer partition given by $[ - 1 . 0 , - 0 . 7 ] , \ ( - 0 . 7 , - 0 . 1 ]$ ， (−0.1, 0.1], (0.1, 0.7], and (0.7, 1.0].

For the MOSI and MOSEI datasets, polarity classes are defined by mapping sentiment scores into discrete levels. The 2-class setting separates negative (< 0) and non-negative $( \geq 0 )$ samples. The 3-class, 5-class, and 7-class settings correspond to $\{ - 1 , 0 , 1 \} , \ \{ - 2 , - 1 , 0 , 1 , 2 \}$ , and $\{ - 3 , - 2 , - 1 , 0 , 1 , 2 , 3 \}$ , respectively.

## B.2 Optimal Choice of Class Granularity

We conduct comparative experiments with $k \in$ {2, 3, 5, 7} (MOSI/MOSEI) and $k ~ \in ~ \{ 2 , 3 , 5 \}$ (SIMS) and select k = 3 (MOSI/MOSEI) and k = 2 (SIMS) based on validation performance. As shown in Table 8, this configuration also yields the best test-set performance.

## C Gradient Conflict Analysis

In multi-task learning, jointly optimizing multiple losses often leads to gradient conflicts, resulting in biased optimization where certain tasks dominate while others are under-optimized. To address this problem, prior work (Chen et al., 2018; Yu et al., 2020a) has proposed methods that mitigate gradient conflicts by adjusting gradient magnitudes or directions during training. For example, Chen et al. (2018) propose GradNorm, a method that automatically balances multiple tasks during training by adjusting the magnitudes of gradients derived from each task-specific loss, ensuring that all tasks learn at a similar rate. Yu et al. (2020a) propose PCGrad, which reduces gradient interference by projecting conflicting gradients onto the normal plane of other task gradients.

Table 7: Network configurations of TCMR. Transformer-based modules list the number of layers, sequence lengths, token counts, input dimensions, attention heads, and hidden dimensions for both MOSI and MOSEI (shown as MOSI / MOSEI). MLP-based modules display the input dimensions and hidden-layer widths used in each component.
<table><tr><td>Module (MOSI / MOSEI)</td><td>Type</td><td>Notation</td><td>#Layers</td><td>Seq. len</td><td>Token len</td><td>Input dim</td><td>#Heads</td><td>Hidden dim</td></tr><tr><td colspan="9">Transformer-based modules</td></tr><tr><td>BERT Encoder</td><td>Transformer</td><td> $\theta ^ { \mathrm { { b e r t } } }$ </td><td>12</td><td>50/ 50</td><td>50</td><td>768 /768</td><td>12</td><td>768</td></tr><tr><td>Text Encoder</td><td>Transformer</td><td> $\theta _ { t } ^ { \mathrm { t r a n s } }$ </td><td></td><td>49/ 49</td><td>8</td><td>768/768</td><td>8</td><td>128</td></tr><tr><td>Audio Encoder</td><td>Transformer</td><td> $\theta _ { \alpha } ^ { \mathrm { { t r a n s } } }$ </td><td></td><td>375 / 500</td><td>8</td><td>5 /74</td><td>8</td><td>128</td></tr><tr><td>Vision Encoder</td><td>Transformer</td><td> $\theta _ { v } ^ { \mathrm { t r a n s } }$ </td><td></td><td>500 / 500</td><td>8</td><td>20/ 35</td><td>8</td><td>128</td></tr><tr><td>A2T Generator (audio → text)</td><td>Transformer</td><td> $\theta _ { \alpha } ^ { \mathrm { g e n } }$ </td><td></td><td>16/16</td><td></td><td>128 / 128</td><td>8</td><td>128</td></tr><tr><td>V2T Generator (vision → text)</td><td>Transformer</td><td> $\theta _ { v } ^ { \mathrm { g e n } }$ </td><td></td><td>16 /16</td><td></td><td>128 / 128</td><td>8</td><td>128</td></tr><tr><td>Text Refinement Transformer</td><td>Transformer</td><td> $\theta _ { t } ^ { \mathrm { r e f } }$ </td><td></td><td>8/8</td><td></td><td>128 / 128</td><td>8</td><td>128</td></tr><tr><td>CrossTransformer (fusion head)</td><td>Transformer</td><td> $\theta ^ { \mathrm { c r o s s } }$ </td><td></td><td>8/8</td><td></td><td>128 / 128</td><td>8</td><td>128</td></tr><tr><td>Reconstructor (Audio)</td><td>Transformer</td><td> $\theta _ { a } ^ { \mathrm { r e c o n } }$ </td><td>２２２２２２２２２</td><td>8/8</td><td></td><td>128 / 128</td><td>8</td><td>128</td></tr><tr><td>Reconstructor (Vision)</td><td>Transformer</td><td> $\theta _ { v } ^ { \mathrm { { r e c o n } } }$ </td><td></td><td>8/8</td><td></td><td>128 / 128</td><td>8</td><td>128</td></tr><tr><td colspan="9">MLP-based modules</td></tr><tr><td>Completeness Estimator (CompNet)</td><td>MLP</td><td> $\theta ^ { \mathrm { c o m p } }$ </td><td>6</td><td></td><td></td><td>768 / 768</td><td></td><td>[768, 768, 1536, 768, 384, 1]</td></tr><tr><td>Polarity Classifier (Classifier)</td><td>MLP</td><td> $\theta _ { \mathrm { n r e a } } ^ { \mathrm { c l a s s i f i e r } }$ </td><td>2</td><td></td><td></td><td>768 /768</td><td></td><td>[384, 3]</td></tr><tr><td>Gating Network (GateNet)</td><td>MLP</td><td> $\mathbf { \partial } \cdot \mathbf { \partial } \theta ^ { \mathrm { g a t e } }$ </td><td>2</td><td></td><td></td><td>256 / 256</td><td></td><td>[128, 1]</td></tr><tr><td>Text Regressor (Regressor)</td><td>MLP</td><td> $\theta ^ { \mathrm { r e g r e s s o r } }$ </td><td>1</td><td></td><td></td><td>128 / 128</td><td></td><td>[1]</td></tr></table>

Table 8: Performance sensitivity to sentiment class granularity (k) in TPSC label generation.

(a) MOSI dataset
<table><tr><td>Model</td><td>Acc-7</td><td>Acc-5</td><td>Non0 Acc/F1</td><td>Has0 Acc/F1</td><td>MAE</td><td>Corr</td></tr><tr><td>TPSC-2cls</td><td>31.45</td><td>36.01</td><td>70.01 / 69.91</td><td>69.46 / 69.25</td><td>1.1073</td><td>50.66</td></tr><tr><td>TPSC-3cls</td><td>32.80</td><td>36.73</td><td>72.01 / 71.67</td><td>70.93 / 70.49</td><td>1.0721</td><td>52.28</td></tr><tr><td>TPSC-5cls</td><td>30.96</td><td>35.12</td><td>69.51 / 69.09</td><td>68.93 / 68.42</td><td>1.1449</td><td>49.62</td></tr><tr><td>TPSC-7cls</td><td>32.16</td><td>35.35</td><td>70.85 / 70.76</td><td>69.74 / 69.54</td><td>1.1040</td><td>49.28</td></tr></table>

(b) MOSEI dataset
<table><tr><td>Model</td><td>Acc-7</td><td>Acc-5</td><td>Non0 Acc/F1</td><td>Has0 Acc/F1</td><td>MAE</td><td>Corr</td></tr><tr><td>TPSC-2cls</td><td>46.41</td><td>47.25</td><td>77.76/77.32</td><td>76.31 / 76.53</td><td>0.6677</td><td>57.95</td></tr><tr><td>TPSC-3cls</td><td>47.27</td><td>48.16</td><td>77.99 / 76.99</td><td>77.59/77.21</td><td>0.6614</td><td>58.63</td></tr><tr><td>TPSC-5cls</td><td>46.94</td><td>47.96</td><td>77.79 / 77.56</td><td>75.46 / 75.95</td><td>0.6638</td><td>59.28</td></tr><tr><td>TPSC-7cls</td><td>46.75</td><td>47.63</td><td>77.74 / 76.74</td><td> $7 7 . 1 5 / 7 6 . 7 9$ </td><td>0.6667</td><td>57.81</td></tr></table>

(c) SIMS dataset
<table><tr><td>Model</td><td>Acc-5</td><td>Acc-3</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>TPSC-2cls</td><td>31.15</td><td>57.45</td><td>72.81</td><td>68.88</td><td>0.5212</td><td>0.379</td></tr><tr><td>TPSC-3cls</td><td>32.43</td><td>56.23</td><td>71.72</td><td>68.96</td><td>0.5478</td><td>0.321</td></tr><tr><td>TPSC-5cls</td><td>33.14</td><td>55.45</td><td>69.13</td><td>68.06</td><td>0.5465</td><td>0.329</td></tr></table>

To validate the effectiveness of AOS, we conduct comparative experiments using existing methods as baselines. Table 9 presents the gradient conflict rates during training and the corresponding convergence behavior of the completeness estimation loss, following the protocol of prior work (Zhang et al., 2024b). In joint training, $\mathcal { L } _ { \mathrm { c o m p } }$ tends to converge to a local optimum, which leads to an increase in GCR (see E2E in Table 9). While GradNorm and PCGrad mitigate gradient conflict, they are still insufficient to escape local optima due to the fundamental limitation of joint training. In contrast, AOS removes interference from other losses during the CompNet training phase, thereby effectively mitigating suboptimal convergence of L<sub>comp</sub>.

Table 9: Gradient conflict rate (GCR) analysis measured during training. F50/L50 denote the first and last 50% of training epochs, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">GCR (%) ↓</td><td rowspan="2"> $\mathcal { L } _ { c o m p } \downarrow$ </td></tr><tr><td>All F50</td><td>L50</td></tr><tr><td>E2E</td><td>51.27 51.16</td><td>54.36</td><td>0.1634</td></tr><tr><td>GradNorm</td><td>54.84</td><td>56.55 53.16</td><td>0.1294</td></tr><tr><td>PCGrad</td><td>52.33</td><td>57.43 48.70</td><td>0.0340</td></tr><tr><td>AOS</td><td>48.47</td><td>53.67 43.33</td><td>0.0055</td></tr></table>

## D Analysis of Completeness Labels

## D.1 Lexicon-derived Completeness Label

To further validate the effectiveness of the proposed TPSC, we introduce Lexicon-Derived Semantic Completeness (LDSC) as a comparison baseline. LDSC measures completeness by explicitly quantifying how much sentiment-related lexical information remains in the incomplete text. Specifically, we adopt SentiWordNet (Baccianella et al., 2010), a widely used lexical resource for sentiment analysis. SentiWordNet provides positive and negative sentiment scores for each word.

![](images/12f9be729a0b38afe291d5e1c2cf2e600fc9f40e918ec63b907bf4ef931b1a42.jpg)  
Figure 7: Illustration of LDSC computation

Figure 7 illustrates the overall process of generating the LDSC label. First, we define the sentiment intensity score of a token t as:

$$
s ( t ) = \operatorname* { m a x } \big ( \operatorname { p o s } ( t ) , \operatorname { n e g } ( t ) \big ) ,\tag{15}
$$

where pos(t) and neg(t) denote the positive and negative sentiment scores of the token. Using this token-level score, we compute the total sentimentrelated content of the complete and incomplete text:

$$
S _ { \mathrm { c o m p } } = \sum _ { t \in T _ { \mathrm { c o m p } } } s ( t ) ,\tag{16}
$$

$$
S _ { \mathrm { i n c o m p } } = \sum _ { t \in T _ { \mathrm { i n c o m p } } } s ( t ) ,\tag{17}
$$

where $T _ { \mathrm { c o m p } }$ and $T _ { \mathrm { i n c o m p } }$ denote the token sets of the complete and incomplete text, respectively. Finally, LDSC-based completeness is computed as:

$$
w _ { l d s c } = \frac { S _ { \mathrm { i n c o m p } } + \varepsilon } { S _ { \mathrm { c o m p } } + 2 \varepsilon } ,\tag{18}
$$

where ε is a small smoothing constant introduced to avoid numerical instability and degenerate cases in which both $S _ { \mathrm { c o m p } }$ and $S _ { \mathrm { i n c o m p } }$ become zero. Intuitively, when $w _ { l d s c } \approx 0 .$ , most key sentiment tokens are missing, whereas $w _ { l d s c } \approx 1$ indicates that the sentiment-related information is largely preserved.

## D.2 Sentiment Token Sensitivity of TPSC

This subsection provides a more detailed description of the sentiment token sensitivity analysis introduced in Section 6.

![](images/f7d1a7db98b44c6d98a0beb3ac6eac285e9b8b02bbe5339b14af76e39a7f593f.jpg)  
(a) c ≥ 0.7

![](images/2fbc7d38acea1c34e7b4770191043ca68ffb65a05b617a6c2d2ee97e68cc4d43.jpg)  
(b) c ≥ 0.8

![](images/ff9507c75f185c0fc6c8c34c16ef8aae8ae4b7ee3c91f5d072498f81ce018ac6.jpg)

![](images/8711064bf37b2200b0af86eab09c111df53db76b11df2baffd1e43bc5c5d3755.jpg)  
(d) c ≥ 0.7

(c) c ≥ 0.9  
![](images/98aebf7f85e68ea730a4c5c2c7d8f9742293f0d5d0cbdc576600e626609e4446.jpg)  
(e) c ≥ 0.8

![](images/941339b970ac773d4e37d2917b58e5a9f57340120b7a39c8665f8586d9df2e94.jpg)  
(f) c ≥ 0.9  
Figure 8: Token-level sensitivity analysis across different confidence thresholds. Subfigures (a)–(c) present results on the MOSI dataset, while (d)–(f) show the corresponding results on the MOSEI dataset.

Analysis Procedure. To evaluate the sensitivity of pseudo completeness labels to key sentiment tokens, we conduct an analysis following a fourstep procedure:

• Step 1: Correct-sample filtering. A pretrained sentiment polarity classifier is applied to the complete textual inputs, and only the correctly classified samples are selected for subsequent steps. Note that a sample is included only when the true-class probability $c \in [ 0 , 1 ]$ exceeds a predefined threshold.

• Step 2: Token-wise Augmentation via Masking. For each correctly classified sample, we generate augmented samples by sliding a window over the sequence and removing exactly one token at each position, replacing it with [UNK]. For example, the sentence “I love this movie” produces four augmented samples:

(1) [UNK] love this movie.

(2) I [UNK] this movie.

(3) I love [UNK] movie.

(4) I love this [UNK].

• Step 3: Re-evaluation of augmented samples. Then, each augmented sample is passed through the same classifier used in Step 1, and we collect the samples in which masking a specific token causes the predicted sentiment to flip to the opposite polarity (e.g., positive → negative or negative → positive).

• Step 4: Identification of error-triggering tokens. Using the polarity-flip samples collected in Step 3, we examine which token in each flipped sample triggers misclassification and compute how frequently each token leads to a misprediction.

Analysis Results Figure 8 illustrates which tokens trigger polarity flips when masked under different thresholds. In both datasets, we consistently observed that as the threshold increases key sentiment tokens remain at the top while semantically ambiguous tokens naturally disappear. In MOSI, [‘I’] is a sentiment-neutral token. One possible explanation is that masking it creates an ambiguous context before the verb, which may resemble negation patterns (e.g., “don’t”) learned during training and occasionally lead to prediction flips. In MO-SEI, the token [’two’] is also observed among the flip-triggering words. This is because in the original sentence “I give the movie two out offive stars” the token [’two’] serves as an explicit negative cue. Overall, the consistent emergence of key sentiment tokens at higher confidence thresholds suggests that the text classifier is well calibrated with respect to semantic cues.

## D.3 Upper-Bound Analysis

Table 10: Comparison between TCMR and its upperbound variant TCMR-ub.
<table><tr><td>Method</td><td>Acc-5</td><td>Acc-2</td><td>MAE</td><td>Corr</td></tr><tr><td colspan="5">MOSI Dataset</td></tr><tr><td>TCMR</td><td>36.73</td><td>70.93</td><td>1.0721</td><td>52.28</td></tr><tr><td>TCMR-ub</td><td>41.24</td><td>77.57</td><td>0.9305</td><td>65.03</td></tr><tr><td colspan="5">MOSEI Dataset</td></tr><tr><td>TCMR</td><td>48.16</td><td>77.59</td><td>0.6614</td><td>58.63</td></tr><tr><td>TCMR-ub</td><td>58.72</td><td>79.83</td><td>0.5697</td><td>71.78</td></tr><tr><td colspan="5">SIMS Dataset</td></tr><tr><td>TCMR</td><td>31.15</td><td>72.81</td><td>0.5212</td><td>37.91</td></tr><tr><td>TCMR-ub</td><td>40.39</td><td>73.69</td><td>0.3991</td><td>66.47</td></tr></table>

To validate the effectiveness of the pseudo-labels, we compare the performance of TCMR-ub, trained with target TPSC labels. Specifically, TCMR-ub is trained by directly using the pseudo-labels instead of training the confidence estimator, enabling us to assess the upper-bound performance of our framework. As shown in Table 10, TCMR-ub consistently outperforms TCMR across all metrics. This indicates that the pseudo-labels provide a meaningful and effective signal for guiding the reconstruction process.

## D.4 Failure Cases of TPSC

<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Error Rate (%)</td></tr><tr><td>MOSI</td><td>74.92</td><td>25.08</td></tr><tr><td>MOSEI</td><td>66.98</td><td>33.02</td></tr><tr><td>SIMS</td><td>79.07</td><td>20.93</td></tr></table>

Table 11: Accuracy of the pretrained sentiment polarity classifier used for TPSC pseudo-labeling on each dataset’s complete text. The higher error rate indicates a greater likelihood of noisy pseudo-labels.

Since TPSC is based on pseudo-labeling, incorrect completeness labels may be generated when the classifier misclassifies samples even on complete data. This is a fundamental limitation of pseudolabel-based approaches that do not rely on human annotations. To assess the prevalence of such failure cases, we evaluated the performance of the pretrained classifier on each dataset. Table 11 shows that MOSEI exhibits relatively lower accuracy compared to other datasets, likely due to its higher complexity. This suggests that the noisier pseudolabels in MOSEI can partially explain its relatively weaker generalization performance.

We further analyzed failure cases and observed that misclassifications mainly occur in ambiguous and sarcastic expressions, as shown in Table 12. These cases are inherently challenging to predict using only the text modality. Despite these limitations, the impact of such mispredictions on completeness estimation is limited. Due to their low TCP, TPSC assigns low completeness scores to misclassified samples, preventing overestimation of completeness and encouraging reliance on proxy features.

## E Analyses with competitive baselines

## E.1 Metric-Level Comparison

In MSA tasks, both classification and regression performances are evaluated to provide a comprehensive assessment. However, classification metrics may not fully represent the quality of sentiment prediction in certain cases due to discretization in their computation.

Specifically, this discretization process discards fine-grained differences between predictions within the same rounding interval. For example, when computing the classification metrics (e.g., Acc-7, Acc-5), continuous sentiment predictions are first converted into discrete classes via rounding, which may lead to inconsistencies near class boundaries, as follows:

<table><tr><td>Pattern</td><td>Utterance</td><td>GT</td><td>Pred</td><td>TCP</td></tr><tr><td>Ambiguous</td><td>There are some funny moments Um I did enjoy it</td><td>Neutral Neutral</td><td>Positive Positive</td><td>0.038 0.027</td></tr><tr><td>Sarcasm</td><td>Hi I&#x27;m pretty I have a giant smile I&#x27;m supposed to know things um walk of screen He um had all the charm of a narcissist xxx boy the whole film</td><td>Negative Negative</td><td>Positive Positive</td><td>0.131 0.022</td></tr></table>

Table 12: Representative failure cases of the pretrained classifier in TPSC pseudo-labeling.
<table><tr><td>Dataset</td><td>TCMR vs</td><td>Acc-7</td><td>Acc-5</td><td colspan="2">Non0 A/F</td><td colspan="2">Has0 A/F</td><td>MAE</td><td>Corr</td><td>W-L</td></tr><tr><td>MOSI</td><td>CENet</td><td>+3.31</td><td>+3.83</td><td></td><td> $+ 2 . 1 3 / + 1 . 7 3$ </td><td> $+ 1 . 5 0 / + 1 . 1 0$ </td><td></td><td>+0.108</td><td>+4.04</td><td>8-0</td></tr><tr><td></td><td>P-RMF</td><td>+3.89</td><td>+5.49</td><td></td><td> $+ 2 . 1 0 \mathrm { ~ / ~ } { + 2 . 1 0 }$ </td><td></td><td> $+ 1 . 6 5 \ : / + 1 . 6 5$ </td><td>+0.058</td><td>+2.21</td><td>8-0</td></tr><tr><td>MOSEI</td><td>CENet</td><td>-0.09</td><td>-0.08</td><td></td><td>+0.98 / -0.27</td><td> $+ 1 . 6 3 \ : / \left. + 0 . 9 8 \right.$ </td><td></td><td>+0.001</td><td>+0.39</td><td>5-3</td></tr><tr><td></td><td>P-RMF</td><td>+0.26</td><td>+0.34</td><td></td><td>-0.08 /-0.77</td><td>+0.41 / +0.30</td><td></td><td>+0.005</td><td>+0.18</td><td>6-2</td></tr><tr><td></td><td>Dataset</td><td>TCMR vs</td><td>Acc-5</td><td></td><td>Acc-3 Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td><td>W-L</td><td></td></tr><tr><td></td><td></td><td>CENet</td><td>+9.54</td><td>+3.97</td><td>+3.58</td><td>+11.91</td><td>+0.109</td><td>+0.365</td><td>6-0</td><td></td></tr><tr><td></td><td>SIMS</td><td>P-RMF</td><td>-1.57</td><td>+0.15</td><td>+2.23</td><td>-1.48</td><td>+0.022</td><td>+0.018</td><td>4-2</td><td></td></tr></table>

Table 13: Detailed comparison between TCMR-TPSC and CENet/P-RMF on MOSI, MOSEI, and SIMS.

• GT: 1.4 → round(1.4) → class 1

• Model A: 1.55 (MAE = 0.15) → class 2

• Model B: 1.10 (MAE = 0.30) → class 1

As shown in Table 13, except for a few classification metrics, TCMR-TPSC consistently outperforms CENet and P-RMF across all benchmark datasets. This indicates that TCMR provides more reliable sentiment prediction overall.

## E.2 Training Efficiency Analysis

Table 14: Comparison of model complexity and training time per epoch.
<table><tr><td>Model</td><td># Params (M)</td><td>Time / Epoch (s)</td></tr><tr><td>LNLN</td><td>115.97</td><td>15.40</td></tr><tr><td>P-RMF</td><td>117.31</td><td>18.00</td></tr><tr><td>TF-Mamba</td><td>111.30</td><td>11.12</td></tr><tr><td>TCMR</td><td>119.23</td><td>12.48</td></tr></table>

As shown in Table 14, TCMR achieves the second-fastest training speed, following TF-Mamba. This is because the reconstruction modules used in LNLN and P-RMF involve more computationally intensive operations, whereas TCMR employs a lightweight MLP-based CompNet. As a result, the additional optimization steps introduced by AOS do not significantly increase the overall training cost.

![](images/67f981e59d19c65008a626fb2395ba719e9a5670a75d920fbaec574528aa7080.jpg)

(a) TCMR vs LNLN  
![](images/621e1c7ec87eeba11d4a3364dbf46bc522c7c47471ae7c10838eba5932e6181c.jpg)  
(b) TCMR vs P-RMF  
Figure 9: Error bars indicate standard deviations. Asterisks denote statistical significance $( ^ { \ast } \mathrm { \sf ~ p } < 0 . 1 , ^ { \ast \ast } \mathrm { \sf ~ p } <$ $0 . 0 5 , ^ { * * * } \mathfrak { p } < 0 . 0 1 )$ .

## E.3 Statistical Analysis

To examine the statistical significance of performance differences among models, we conduct comparative experiments on the MOSI dataset with LNLN, P-RMF, and TCMR. As shown in Figure 9, TCMR achieves statistically significant performance improvements across most evaluation metrics.