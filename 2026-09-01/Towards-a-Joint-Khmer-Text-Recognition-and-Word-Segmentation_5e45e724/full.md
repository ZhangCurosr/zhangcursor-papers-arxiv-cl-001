# Towards a Joint Khmer Text Recognition and Word Segmentation

Marry Kong<sup>1</sup>, Rina Buoy<sup>1,2[0000−0002−6960−4262]</sup>, Sovisal Chenda<sup>1</sup>, Nguonly Taing<sup>1</sup>, Masakazu Iwamura<sup>2</sup>, and Koichi Kise<sup>2</sup>

<sup>1</sup> Techo Startup Center, Ministry of Economy and Finance, Phnom Penh, Cambodia <sup>2</sup> Osaka Metropolitan University, Osaka, Japan

Abstract. Text recognition, or extracting electronic text from document images, has been indispensable for knowledge retrieval tasks, such as retrieval-augmented generation (RAG). For Khmer, extracted text is subject to an extra word segmentation step, as Khmer does not use any visible word delimiters to denote word boundaries. Thus, a recognitionthen-segmentation pipeline for Khmer requires two separate sequential models; this is not only error-prone but also adds significant latency for large-scale document processing. This paper proposes a novel joint Khmer text recognition and word segmentation framework in a unified model. The proposed model, using a connectionist-temporal-classification (CTC) decoder for fast, parallel decoding, can be instructed to recognize Khmer text with (b = 1) and without (b = 0) word segmentation. Experimental results on diferent benchmark datasets of diferent document modalities (document, scene, and handwritten images) show that the proposed model can not only recognize characters in document images but also locate word boundaries, removing the need for an extra word segmentation step in a conventional sequential pipeline.

Keywords: Text recognition · word segmentation · low-resource language

## 1 Introduction

Central to knowledge retrieval tasks, such as retrieval-augmented generation (RAG) [15], is text recognition, which is the process of extracting electronic text from document images. For Khmer, an extra word segmentation step is often needed to segment extracted text into words in order to facilitate a hybrid lexical-vector search or apply any post-recognition spell-checking, as Khmer does not use any visible word delimiters in its writing system [7].

A common pipeline for Khmer text recognition and word segmentation requires two separate models applied in sequence. Separately, Khmer text recognition and word segmentation are relatively well-explored. Many text recognition methods [3, 4, 13, 17, 18, 23] have been proposed for the accurate recognition of Khmer characters. Similarly, the task of Khmer word segmentation has also been addressed in multiple previous works [1, 7, 11]. However, chaining both models together in a sequential pipeline is not only error-prone but also introduces significant latency when processing a large batch of document images in industrial settings. Furthermore, no previous research has focused on addressing joint text recognition and word segmentation, at least for Khmer text.

![](images/586da3badad39090ed20932ef97b1c6b4d139234f507ebaca9703d44ac5c11d3.jpg)  
Fig. 1: The multi-layered layout of Khmer text, highlighting complex character stacking and ligatures. Green: base consonant. Blue: consonant subscript. Orange: dependent vowel. Purple: diacritic. Best viewed in color.

In this paper, we propose the first joint Khmer text recognition and word segmentation (KTRWS) approach that performs both recognition and segmentation of Khmer text in a single model. The proposed approach is fast, as it is based on a connectionist-temporal-classification [10] (CTC) decoder for parallel character recognition. In addition, the proposed model can be instructed to recognize Khmer text with (b = 1) and without (b = 0) word segmentation. Thus, there is no need for an extra word segmentation step, making the proposed approach particularly suitable for real-world, low-latency, large-scale applications. Since existing Khmer text recognition datasets do not have word boundary annotations, we synthetically generate a new open dataset<sup>3</sup> of Khmer text line images with word boundary annotations for model training. Our contributions can be summarized as follows:

1. We propose the KTRWS method, which performs joint Khmer text recognition and word segmentation in a single model, along with the first Khmer textline recognition dataset with word boundary annotations.

2. By performing joint text recognition and word segmentation with a parallel CTC decoder, the proposed method achieves significantly lower latency for this combined task.

3. Experimental results show that the proposed method can not only recognize Khmer characters accurately but can also eliminate the need for an extra word segmentation step.

## 2 Related Work

Khmer Text Recognition Khmer script, as shown in Figure 1, is one of the most complex writing systems, characterized by multi-layered character stacking, absence of visible word delimiters, and a large inventory of characters, including 33 consonants, 37 dependent and independent vowels, and eight diacritics [3, 23]. Despite some recent research progress, Khmer text recognition is a relatively under-explored yet maturing field because of limited public training and benchmark datasets. Early Khmer text recognition methods [8,14,21] relied on classical feature extractors, such as wavelet descriptors, and classical machine learning methods, such as support vector machines (SVM). Such methods often require explicit character segmentation and are unable to generalize robustly. Deep-learning-based approaches have been proven to achieve superior performance across document modalities (i.e., scene vs. document). Keo et al. [12] adopted two prominent Latin-based methods for the task of Khmer scene text recognition on their published dataset using a convolutional feature extractor and two diferent decoding mechanisms, including a CTC-based decoder and an attention-based decoder. Similarly, Buoy et al. [5] addressed the task of Khmer document text recognition using a sequence-to-sequence attentional network. On the other hand, Valy et al. [23, 24] proposed multiple deep-learning-based approaches for recognizing Khmer characters and words on historical palm leaf manuscript documents.

Various recent state-of-the-art (SoTA) methods [2,3,13] for Khmer text recognition have incorporated Transformers [25] for both feature extraction and character decoding, and proposed a Khmer-native tokenization scheme called Khmer character clusters (KCC) instead of conventional character-level tokenization. As a result, character error rates (CERs) across public benchmark datasets, including scene, document, and handwritten images, have been significantly reduced.

Khmer Word Segmentation Since Khmer does not use any visible delimiters to denote word boundaries, word segmentation is a necessary prior step for any downstream natural language processing (NLP) tasks, such as part-of-speech (POS) tagging, named entity recognition (NER), spell checking, lexical search, and so on. Nonetheless, there is no standard definition of a word in Khmer, nor a standard corpus for this task [11]. As a result, researchers often construct their own corpora and train their own models.

The early work [1] on Khmer word segmentation was based on a dictionarybased bidirectional maximal matching (BiMM) approach. Despite being simple (requiring no training) and lightweight, the BiMM approach is unable to handle out-of-vocabulary (OOV) words. To address this limitation, Chea et al. [7] proposed a statistical machine learning approach using a conditional random field (CRF) algorithm. The authors manually constructed training and evaluation datasets using word-level and compound-level (i.e., prefixes and sufixes) annotations and trained the CRF-based model. The CRF-based model achieved an accuracy of 98.5% and has become a de facto baseline model.

In addition, Buoy et al. [6] proposed a neural Khmer word segmentation network using a bidirectional long short-term memory (BiLSTM) architecture to improve contextual information modeling. The authors proposed two variants using character-level and character-cluster-level tokenization schemes. Recognizing that word segmentation and POS tagging can be jointly trained, Kaing et al. [11] proposed a joint word segmentation and POS tagging method.

![](images/353de62d6da001ad744dd9f5243d947bb8a738cef58c3cc2a64abd424f412cb7.jpg)  
Fig. 2: The proposed KTRWS framework. Best viewed in color.

In summary, while both word segmentation and text recognition for Khmer have significantly evolved along their respective, seemingly diferent trajectories, both tasks meet and complement each other in downstream applications, such as knowledge retrieval, where electronic texts are recognized from document images and words are segmented for ingestion and retrieval. Thus, unifying both tasks within a single model can lead to significant latency reduction, which is of great importance for practical applications. To this end, this paper proposes a novel KTRWS approach.

## 3 The Proposed Method

As shown in Figure 2, our proposed KTRWS framework comprises four key components: a word boundary projector, a visual encoder, a modality-aware feature selector (MAFS), and a text decoder. The boundary projector embeds and projects a binary flag (b) indicating whether to output word boundaries. The visual encoder extracts both visual and sequential features from an input image. The MAFS fuses word boundary projections and visual-temporal features, and adapts according to a specific modality (i.e., scene, handwritten, and document text). Finally, the text decoder uses a CTC decoder to output Khmer tokens with (b = 1) or without word boundary tokens (b = 0).

Word Boundary Projector Module This module embeds and projects a binary flag indicating whether to output word boundaries, and returns two projection vectors, $\beta \in \mathbb { R } ^ { d }$ and $\gamma \in \mathbb { R } ^ { d }$ . These projection vectors are fused with the adapted features from the encoder before Khmer character decoding. With this design, the model can be instructed to output word boundaries or not during inference. Mathematically, the word boundary projector module can be expressed as

$$
E _ { b } = \mathrm { L I N E A R L A Y E R } ( b )\tag{1}
$$

$$
\beta = \mathrm { L I N E A R L A Y E R } ( E _ { b } )\tag{2}
$$

$$
\gamma = \mathrm { L I N E A R L A Y E R } ( E _ { b } ) ,\tag{3}
$$

where $b \in \{ 0 , 1 \}$ is a binary variable indicating whether to output word boundaries, LINEARLAYER is a dense neural network layer, $\pmb { { \cal E } } _ { b } \in \mathbb { R } ^ { d }$ is an embedding vector of $b ,$ and $\beta$ and $\gamma$ are the resulting projection vectors.

Visual Encoder The visual encoder is designed to extract the visual-temporal features required for accurate character recognition. For a given RGB image (I), our visual encoder consists of a base convolutional network for two-dimensional (2D) visual features $( \boldsymbol { F } \in \mathbb { R } ^ { w ^ { \prime } \times h ^ { \prime } \times d } )$ and a Transformer-based encoder network for visual-temporal features $( G \in \mathring { \mathbb { R } } ^ { w ^ { \prime } \times h ^ { \prime } \times d } )$ . Since the CTC decoder expects a sequence of one-dimensional (1D) features $( G _ { \mathrm { 1 D } } \in \mathbb { R } ^ { w ^ { \prime } \times d } )$ along the width direction, the extracted visual-temporal features are averaged over the height direction. Mathematically, our visual encoder can be expressed as

$$
F = \mathrm { C N N } ( I )\tag{4}
$$

$$
G = \mathrm { T R } _ { \mathrm { E N C } } ( F )\tag{5}
$$

$$
G _ { 1 D } = \mathrm { H P O O L I N G } ( G ) ,\tag{6}
$$

where CNN, TR<sub>ENC</sub>, and HPOOL are the base convolutional network, the Transformer encoder network, and the height-averaging operator, respectively; I is an input RGB image; and d (= 512) is the feature dimension.

Modality-Aware Feature Selector To enable the proposed framework to robustly recognize Khmer text across diferent modalities (e.g., scene, handwritten, and document), we modify the modality-aware feature selector [13] (MAFS) to incorporate the word boundary projections $( \mathrm { i . e . , } \beta$ and $\gamma )$ . Specifically, this module computes an average vector $z \in \mathbb { R } ^ { d }$ using the GPOOL operator, given $G _ { \mathrm { 1 D } }$ . The module then uses ROUTER to compute a distribution over n modality sources $( \pmb { r } \in \mathbb { R } ^ { n } )$ , from which modality-adapted features $( H _ { \mathrm { 1 D } } \in \mathbb { R } ^ { d \times n } )$ from ADAPTER are marginalized to yield the final features $( U _ { \mathrm { 1 D } } \in \mathbb { R } ^ { w ^ { \prime } \times d } )$ , which are then subject to feature-wise linear modulation [20] (FiLM), scaled and shifted by $\gamma$ and $\beta ,$ , for output token generation. Mathematically, the modified MAFS can be expressed as

$$
z = \mathrm { G P O O L I N G } ( G _ { 1 D } )\tag{7}
$$

$$
r = \mathrm { R O U T E R } ( z )\tag{8}
$$

$$
H _ { 1 D , k } = \mathrm { A D A P T E R } _ { \mathrm { k } } ( G _ { 1 D } )\tag{9}
$$

$$
U _ { 1 D } = \gamma \odot ( H _ { 1 D } \cdot r ) + \beta ,\tag{10}
$$

where $U _ { \mathrm { 1 D } }$ is the input feature map for the CTC decoder to generate output Khmer tokens. We adopt the same tokenization scheme as [2,13], which operates at the character-cluster level rather than the character level.

Text Decoder As shown in Figure 2, we utilize a CTC decoder for parallel Khmer token generation. Given the input feature map $U _ { \mathrm { 1 D } } \in \mathbb { R } ^ { w ^ { \prime } \times d }$ from the MAFS module, the decoder projects each of the $w ^ { \prime }$ frames onto the augmented vocabulary $\mathcal { V } ^ { \prime } = \mathcal { V } \cup \{ \emptyset \} \cup \{ \mathsf { u } 2 0 0 \mathsf { b } \}$ , where V is the set of character-cluster tokens, ∅ is the blank symbol, and u200b is a word boundary marker token (invisible when rendered). A softmax classifier produces a per-frame distribution as

$$
p _ { i } = \mathrm { S O F T M A X } ( U _ { 1 D , i } ) ,\tag{11}
$$

where $p _ { i } ( c )$ denotes the probability of emitting token c at frame $i .$

For a given target sequence y, a L<sub>CTC</sub> loss is given by

$$
\mathcal { L } _ { \mathrm { C T C } } = - \sum _ { ( I , y ) \in \mathcal { D } } \log p ( y \mid U _ { 1 D } ) ,\tag{14}
$$

where log p $\textbf { \ ( \textit { y } | } U _ { 1 } D )$ is obtained by marginalizing over all alignments that collapse to y. At test time, we use greedy (best-path) decoding, which approximates the most probable sequence by selecting the most likely token independently at each frame and collapsing repeated tokens and $\mathcal { D }$

$$
\hat { \pi } _ { i } = \arg \operatorname* { m a x } _ { c \in \mathcal { V } ^ { \prime } } p _ { i } ( c ) , \qquad i = 1 , \ldots , w ^ { \prime }\tag{12}
$$

Alternative Design for Word Boundary Projector Module Alternatively, the boundary embedding vector can be fused via gating. Mathematically, the word boundary projector with gating can be expressed as

$$
\pmb { \alpha } = \sigma ( \mathrm { L I N E A R L A Y E R } ( E _ { b } ) )\tag{13}
$$

$$
\pmb { p } = \mathrm { L I N E A R L A Y E R } ( \pmb { E _ { b } } )\tag{14}
$$

$$
U _ { 1 D } = \alpha \odot ( H _ { 1 D } \cdot r ) + ( 1 - \alpha ) \odot p ,\tag{15}
$$

where $b \in \{ 0 , 1 \}$ is a binary variable indicating whether to output word boundaries, LINEARLAYER is a dense neural network layer, $\pmb { { \cal E } } _ { b } \in \mathbb { R } ^ { d }$ is an embedding vector of b, and $\sigma$ is a sigmoid gating function applied element-wise to the embedding vector.

![](images/1f821d95f5a959de1f63279b20878debf25fd64288f96201972d4a9fb550e473.jpg)  
Fig. 3: Sample preview images from the existing datasets.

## 4 Datasets & Experimental Setup

Datasets Prior Khmer text recognition methods have relied largely on synthetic training data, which is convenient to generate at scale, whereas real data for the scene and handwritten modalities remains scarce. We adopt the publicly available datasets listed below, comprising real and synthetic Khmer document, scene, and handwritten text images. A summary of all datasets and some preview images are given in Table 1 and Figure 3, respectively.

Buoy et al.: [3] This dataset comprises about 2.8M synthetic textlines (1.5M document, 1.3M scene) rendered with 11 Khmer fonts. This dataset features document and scene modalities.

KHOB [9]: This dataset comprises 1,318 textlines manually cropped from scanned low-resolution Khmer PDF documents. This dataset is used for documentmodality evaluation.

SynthText [27]: This dataset comprises 70,000 textlines extracted from 10,000 synthetic Khmer identity card images. This dataset features document modality.

HierText [16]: This dataset comprises a large-scale Latin printed/handwritten dataset yielding about 518,726 textline images after cropping and removing vertical texts. This dataset is used to learn Latin visual representations (scene and handwritten modalities).

KhmerST [17]: This dataset comprises 3,022 real Khmer scene text images (mostly logos and billboards) with diverse artistic fonts and challenging imaging conditions; This dataset is used for scene-modality evaluation.

WildKhmerST [18]: This dataset comprises 29,601 textlines from 10,000 real images across Cambodia, including artistic, blurred, low-light, curved, and occluded text. This dataset is used for scene modality.

KH [26]: This dataset comprises about 4.2k Khmer and Latin textlines (3,991 train, 211 eval), both handwritten and printed. This dataset features document and handwritten modalities.

GKST [13]: This dataset comprises 4,221 smartphone-captured Khmer scene text images (4,009 train, 212 eval), annotated from general scenes rather than focused close-ups. This dataset features scene modality.

Table 1: Summary of datasets. Size: train / eval (– if none). Purpose: Tr = training (D), Ad = adapting (S&H), Ev = evaluation. Modality: D = document, S = scene, H = handwritten.
<table><tr><td>Dataset</td><td>Size</td><td>Purpose</td><td>Modality</td></tr><tr><td>Buoy et al.</td><td>2.8M/-</td><td>Tr</td><td>D, S</td></tr><tr><td>KHOB</td><td>-/1,318</td><td>Ad, Ev</td><td>D</td></tr><tr><td>SynthText</td><td>70k/-</td><td>Tr</td><td>D</td></tr><tr><td>HierText</td><td>518,726/–</td><td>Tr</td><td>S, H</td></tr><tr><td>KhmerST</td><td>-/3,022</td><td>Ad, Ev</td><td>S</td></tr><tr><td>WildKhmerST</td><td>29,601/-</td><td>Ad</td><td>S</td></tr><tr><td>KH</td><td>3,991 /211</td><td>Ad, Ev</td><td>D, H</td></tr><tr><td>GKST</td><td>4,009/212</td><td>Ad, Ev</td><td>S</td></tr><tr><td>KHT</td><td>13,457/711</td><td>Ad, Ev</td><td>H</td></tr></table>

KHT [13]: This dataset comprises 14,168 Khmer handwritten text images (13,457 train, 711 eval) from sources such as birth certificates, exam papers, and notes. This dataset features handwritten modality.

Since existing datasets do not provide word boundary annotations, we synthetically generate 60,000 text line images with word boundary annotations. To achieve this, we begin by training a Khmer Transformer-based model for the word segmentation task using the manually labelled dataset provided by Chea et al. [7]. The textlines, sourced mainly from news articles and segmented by an invisible word boundary marker (i.e., u200b), are rendered synthetically as images for model training. Consequently, the rendered text images are not aesthetically afected by the extra word boundary information. Some preview images from this new dataset with word boundary annotations are provided in Figures 4.

![](images/d9be9e5642b01872f323c43f52756e6b38d0710ce70f75b719b17bc567fcca2e.jpg)  
(c) English translation: The spread of conflict.  
Fig. 4: Sample generated images with word boundary annotations (red bar) from our new dataset.

Table 2: The visual encoder’s complexity and model size. Params: model parameters. Flops: floating operations on an input of 32 × 116 pixels.
<table><tr><td>Name</td><td>Params. Flops</td></tr><tr><td>ResNet backbone (visual features)</td><td>13.0M 3.2G</td></tr><tr><td>Transformers encoder (sequential features)</td><td>9.5M 2.2G</td></tr><tr><td>MAFS</td><td>0.80M 0.15G</td></tr></table>

Experimental Setup & Training Strategy The base CNN feature extractor is composed of six sequential ResNet blocks with channel dimensions progressively increasing from 32 to 512, where each block comprises 1×1 and/or 3×3 convolutions repeated multiple times. Spatial downsampling by a factor of (2, 2) is performed at ResNet Blocks 1 and 4, while the remaining blocks maintain the spatial resolution. The TR<sub>ENC</sub> network is configured with an embedding dimension (d) of 512, three stacked encoder layers, eight attention heads, a dropout rate of 0.1, and a feed-forward network dimension of 2048 $( 4 \times d )$ . The model size and computational complexity of the visual encoder are reported in Table 2. Although there are three discrete modalities, document, scene, and handwritten, real images rarely belong to only one modality. Instead, they can be represented as points within a simplex spanning these three modalities. For example, a text image may be simultaneously both handwritten and scene text. By default, we set this hyperparameter (n) to five, providing additional capacity to model hybrid or transitional cases [13].

Following [13], we group these datasets by primary modality: Document (D) for large-scale printed data (Buoy et al., SynthText, HierText); Scene & Handwritten (S&H) for real data (WildKhmerST, the KH training set, GKST, and KHT); and an Evaluation group comprising KhmerST (S), KHOB (D), the KH evaluation set (D&H), GKST (S) and KHT (H).

The KTRWS models were trained in two phases: a general training phase followed by a modality-adapting phase. In the general training phase, the models were first trained on the large-scale document (D group) datasets to learn robust visual representations of Khmer and Latin characters. A cyclic learning rate schedule was employed, with a minimum value of $1 0 ^ { - 5 }$ and a maximum of $1 0 ^ { - 4 }$ . This phase lasted five full epochs with a batch size of 32 images. In the modality-adapting phase, the trained models then underwent adaptation on the real scene and handwritten (S&H group) datasets together with our newly annotated word-boundary dataset for 50 epochs, using a lower cyclic learning rate schedule with a minimum of $1 0 ^ { - 6 }$ and a maximum of $1 0 ^ { - 5 }$ . In both phases, the Adam optimizer was used with a gradient clipping value of 50. To prevent the models from underperforming on printed document text during this phase, we sampled an equal number of document images and mixed them with the S&H images. As a result, the same model can both recognize Khmer text across the document, scene, and handwritten modalities and, when conditioned on $b = 1$ additionally output word boundaries.

Table 3: Character error rates (CER) on the evaluation datasets of diferent modalities (D,S,&H). \* : model-specific tokenizer. Char.: character. KCC: Khmer character cluster. Seg.: joint character recognition and word segmentation. Bold: best. Italic: second best.
<table><tr><td colspan="5">Dec. Tok. Seg. KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)</td></tr><tr><td>Surya-OCR [19] Tr. *</td><td>No 17.69</td><td>43.21</td><td></td></tr><tr><td>Buoy et al. [3] Tr.</td><td>Char. No 3.03 *</td><td></td><td></td></tr><tr><td>Nom et al. [17] Tr.</td><td>No</td><td>17.00</td><td></td></tr><tr><td>Soy et al. [26] Tr.</td><td>No</td><td>17.00</td><td></td></tr><tr><td>Buoy et al. [2] Tr.</td><td>No 2.13 7.01</td><td></td><td></td></tr><tr><td>Tesseract-OCR [22] CTC Char. No</td><td>9.19</td><td>40.96</td><td></td></tr><tr><td>Buoy et al. [4] CTC KCC</td><td>No 2.33</td><td></td><td></td></tr><tr><td>UKTR [13] CTC KCC</td><td>No 2.46</td><td>3.02 5.89</td><td>4.41 9.52</td></tr><tr><td>KTRWS (b = 0) CTC KCC</td><td>No 1.75</td><td>2.77 5.78</td><td>4.51 8.91</td></tr><tr><td>KTRWS (b = 1)</td><td>CTC KCC Yes 1.79</td><td>3.04 6.06</td><td>5.01 9.64</td></tr></table>

## 5 Results & Discussion

Recognition Performance of the KTRWS Model We begin by comparing the recognition performance of the proposed method in terms of character error rate (CER) across the evaluation datasets of diferent modalities. Since the evaluation datasets do not have word boundary annotations, CERs are computed by excluding the u200b word boundary marker token.

Table 3 provides the recognition accuracy comparisons of the proposed models, with and without word boundary outputs, against the SoTA methods. Compared with existing methods, the proposed models either outperform on the KHOB dataset or are competitive on the other datasets (KhmerST, KH, GKST, and KHT). Compared with the recent SoTA UKTR model using the same CTC decoder, our KTRWS (b = 0) model without outputting word boundaries achieves improved CERs on all datasets except GKST. Specifically, our KTRWS (b = 0) model (i.e., no word boundary outputs) obtained CERs of 1.75%, 2.77%, 5.78%, 4.51%, and 8.91% on KHOB, KhmerST, KH, GKST, and KHT, respectively, versus 2.46%, 3.02%, 5.89%, 4.41%, and 9.52% by the CTCbased UKTR model. When outputting word boundaries, our KTRWS (b = 1) model remains competitive. Specifically, our KTRWS (b = 1) model obtained CERs of 1.79%, 3.04%, 6.06%, 5.01%, and 9.64%. The marginal increases in CER with b = 1 are expected, as the model needs to generate both normal Khmer tokens and word boundary tokens, which increases the probability of making errors. On the other hand, the significant CER improvements on the KHOB dataset can be attributed to the fact that the new dataset introduced in this study is more aligned with the nature of the KHOB dataset (i.e., long document textlines).

As shown in Figure 5, our KTRWS (b = 1) model achieves significantly lower combined recognition and segmentation latency on the KHOB evaluation set than the sequential recognition-then-segmentation approaches (i.e., UKTR or KTRWS $\left( b = 0 \right) )$ . This is attributed to its parallel, joint decoding nature, which eliminates the need to perform word segmentation as a separate step using an external model such as the Transformer-based Khmer word segmentation model described in Section 4. It should be noted that the degree of latency reduction is directly proportional to the latency of the word segmentation model used in the sequential pipeline.

![](images/27c464151433ef76702290d2cc21b95bd5d31aabe34115515ac669bc7088baf3.jpg)  
Fig. 5: Latency (seconds) comparison on for the join recognition and segmentation task on the KHOB evaluation set.

Table 4: F1 scores on the evaluation datasets of diferent modalities $( \mathrm { D } , \mathrm { S } , \& \mathrm { H } ) . \ ^ { * }$ model-specific tokenizer. Char.: character. KCC: Khmer character cluster. Bold: best. Italic: second best.
<table><tr><td colspan="6">KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)</td></tr><tr><td>KTRWS (b = 0)</td><td>98.49</td><td>97.90</td><td>95.84</td><td>96.41</td><td>93.40</td></tr><tr><td>KTRWS (b = 1)</td><td>97.68</td><td>95.08</td><td>91.09</td><td>93.35</td><td>89.24</td></tr></table>

Segmentation Accuracy of the KTRWS Models We evaluate word segmentation using F1 on the evaluation datasets. For each textline image, let $\mathcal { R }$ and H denote the reference and hypothesis word sequences, respectively. There are two possible cases:

– Without OCR errors: when the concatenated character sequences agree $( \cup \mathcal R =$ $\cup \mathcal { H } )$ , we compare the sets of cumulative character-ofset boundaries $B _ { \mathcal { R } }$ and $B _ { \mathcal { H } }$ directly and compute true positive (TP), false positive (FP), and false negative (FN).

– With OCR errors: When OCR errors cause a character mismatch, boundary ofsets are no longer comparable. We fall back to word-level multiset overlap $\begin{array} { r } { \mathrm { T P } = \sum _ { w } \operatorname* { m i n } ( c _ { \mathcal { R } } ( w ) , c _ { \mathcal { H } } ( w ) ) } \end{array}$ , where $c ( \cdot )$ denotes word counts, with FP and FN defined analogously.

TP, FP, and FN are summed across all textline images and used to compute micro-averaged F1 score.

![](images/47fded3b5653eee95ebf5d44296d6b1eb25fb69bb536e90ed1f051c88826ccc5.jpg)  
Fig. 6: Qualitative assessment of recognition and segmentation accuracy on some selected cases. Red bar: missing word boundary. Strided red bar: extra word boundary. Red: recognition mistake.

Since none of the evaluation datasets originally have word boundary annotations, we used our trained word segmentor discussed in Section 4 to segment the ground-truth texts. The same applies to the model predictions of our KTRWS (b = 0) model (i.e., without word boundaries). In contrast, the model predictions of our KTRWS (b = 1) model already include word boundaries. Table 4 shows that the segmentation accuracy of our KTRWS (b = 1) model degrades as the document modality transitions from document to scene and handwritten text. This is because the document dataset (KHOB) mainly comprises long text-line images with complete sentences, which are not hard to recognize and provide sufficient contextual cues for word segmentation. On the other hand, the scene and handwritten cases (KhmerST, KH, GKST, and KHT) mainly comprise single words, short phrases, or non-textual content (e.g., numbers), which are out-ofcontext and challenging for both recognition and word boundary identification.

Qualitative Assessment Figure 6 provides a qualitative assessment of the recognition and segmentation outputs from our KTRWS (b = 1) model on selected cases. Consistent with the above observations, while recognition errors are infrequent, except in Figure 6f (i.e., a historically degraded image), segmentation errors (mainly missing word boundaries) occur more frequently on scene and handwritten text images, as in the cases of Figures 6b, 6c, 6e, and 6f.

Visual Grounding & Alternative Design for Word Boundary Projector Module Another benefit of the proposed joint recognition and word segmentation method is the visual grounding of recognized words. Thanks to the CTC decoder predicting p<sub>i</sub>(c) for each feature frame along the width direction, it is possible to locate the word boundary tokens and map them to their actual positions on the input image. As a result, each predicted word can be accurately visually grounded.

Figure 7 shows the projections of the word boundary locations on the input images. Except for highly curved text (e.g., Figure 7c), the projected word boundary locations are reasonably accurate. It should be noted that the visual encoder downsamples the features by a factor of four in the width direction, which makes it inevitable that the predicted locations may be slightly of.

![](images/2babaec44da550849cd34e3f908bdded4db106abfe96916ea176c0d9908bf9e7.jpg)  
Fig. 7: The projected word boundary locations on the input images. Red bar: boundary location.

In terms of the alternative design for the word boundary projector module, Tables 5 and 6 show that the default design of the word boundary projector module achieves better character recognition and word segmentation accuracies across the evaluation datasets. Thus, fusing the word boundary instruction via feature-wise linear modulation is more efective than simple gated feature fusion.

## 6 Limitations & Future Work

We identify the following limitations and future directions associated with this study:

1. The training relies on the new synthetic dataset with word boundary annotations. Although both recognition and segmentation performance are high for document images, there is room for improvement for other modalities (i.e., scene and handwritten). Thus, future work will focus on constructing a diverse real dataset with word boundary annotations.

Table 5: Character error rates (CER) on the evaluation datasets of diferent modalities $( \mathrm { D } , \mathrm { S } , \& \mathrm { H } ) .$ \* : model-specific tokenizer. Char.: character. KCC: Khmer character cluster. Bold: best. Italic: second best.
<table><tr><td colspan="6">KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)</td></tr><tr><td>KTRWS  $( \mathrm { G a t i n g } ; b = 0 )$ </td><td>1.75</td><td>2.98</td><td>6.23</td><td>4.92</td><td>9.84</td></tr><tr><td>KTRWS  $( \mathrm { G a t i n g } ; b = 1 )$ </td><td>1.82</td><td>3.27</td><td>7.62</td><td>5.17</td><td>10.46</td></tr><tr><td></td><td></td><td></td><td>5.78</td><td></td><td></td></tr><tr><td>KTRWS  $( { \mathrm { F i L M } } ; b = 0 )$  KTRWS  $( { \mathrm { F i L M } } ; b = 1 )$ </td><td>1.75 1.79</td><td>2.77 3.04</td><td>6.06</td><td>4.51 5.01</td><td>8.91 9.64</td></tr></table>

Table 6: F1 scores on the evaluation datasets of diferent modalities $( \mathrm { D } , \mathrm { S } , \& \mathrm { H } ) . \ ^ { * }$ model-specific tokenizer. Char.: character. KCC: Khmer character cluster. Bold: best. Italic: second best.
<table><tr><td colspan="6">KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)</td></tr><tr><td>KTRWS (Gating; b = 0)</td><td></td><td>98.48 97.73</td><td>94.06</td><td>96.23</td><td>93.01</td></tr><tr><td>KTRWS</td><td>(Gating; b = 1) 97.50</td><td>94.69</td><td>87.97</td><td>93.01</td><td>87.87</td></tr><tr><td>KTRWS</td><td>98.49</td><td>97.90</td><td>95.84</td><td>96.41</td><td>93.40</td></tr><tr><td>KTRWS</td><td> $( { \mathrm { F i L M } } ; b = 0 )$   $( { \mathrm { F i L M } } \ ; b = 1 )$  97.68</td><td>95.08</td><td>91.09</td><td>93.35</td><td>89.24</td></tr></table>

2. The word boundary is implicitly handled by minimizing the $\mathcal { L } _ { \mathrm { C T C } }$ loss. Thus, future work will incorporate explicit word boundary supervision.

## 7 Conclusion

We propose a novel joint Khmer text recognition and word segmentation (KTRWS) framework within a single model. With a CTC decoder for fast, parallel decoding, the KTRWS model can be instructed to recognize Khmer text with $( b = 1 )$ and without $( b = 0 )$ word segmentation. Experimental results on benchmark datasets of diferent document modalities (document, scene, and handwritten images) show that the proposed model not only recognizes characters in document images but also locates word boundaries, removing the need for an extra word segmentation step in a conventional sequential pipeline.

## References

1. Bi, N., Taing, N.: Khmer word segmentation based on bi-directional maximal matching for plaintext and microsoft word document. In: Signal and Information Processing Association Annual Summit and Conference (APSIPA), 2014 Asia-Pacific. pp. 1–9. IEEE (2014)

2. Buoy, R., Chenda, S., Taing, N., Kong, M., Iwamura, M., Kise, K.: Addressing the attention drift problem for khmer long textline recognition: R. buoy et al. International Journal on Document Analysis and Recognition (IJDAR) pp. 1–26 (2025)

3. Buoy, R., Iwamura, M., Srun, S., Kise, K.: Toward a low-resource non-latincomplete baseline: an exploration of khmer optical character recognition. IEEE Access 11, 128044–128060 (2023)

4. Buoy, R., Iwamura, M., Srun, S., Kise, K.: Language-aware non-autoregressive khmer textline recognition. In: International Conference on Pattern Recognition and Artificial Intelligence. pp. 339–353. Springer (2024)

5. Buoy, R., Taing, N., Chenda, S., Kor, S.: Khmer printed character recognition using attention-based seq2seq network. Ho Chi Minh City Open University Journal Of Science-Engineering And Technology pp. 3–16 (2022)

6. Buoy, R., Taing, N., Kor, S.: Khmer word segmentation using bilstm networks. In: 4th Regional Conference on OCR and NLP for ASEAN Languages (ONA 2020), Phnom Penh, Cambodia (2020)

7. Chea, V., Thu, Y.K., Ding, C., Utiyama, M., Finch, A., Sumita, E.: Khmer word segmentation using conditional random fields. Khmer Natural Language Processing pp. 62–69 (2015)

8. Chey, C., Kumhom, P., Chamnongthai, K.: Khmer printed character recognition by using wavelet descriptors. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems 14(03), 337–350 (2006)

9. EKYC Solutions: Khmer ocr benchmark dataset. https : / / github . com / EKYCSolutions/khmer-ocr-benchmark-dataset (2022), a standardized benchmark dataset for Khmer Optical Character Recognition (OCR), developed in collaboration with EKYC Solutions, Prudential Life Assurance PLC, and Paragon International University

10. Graves, A., Fernández, S., Gomez, F., Schmidhuber, J.: Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In: Proceedings of the 23rd international conference on Machine learning. pp. 369– 376 (2006)

11. Kaing, H.: Towards morphological and syntactic analyses for the khmer language (2022)

12. Keo, S., Coustaty, M., Bakkali, S., Rossinyol, M.: State-of-the-art khmer text recognition using deep learning models. In: ASEAN Conference on Emerging Technologies 2024 (2024)

13. Kong, M., Buoy, R., Chenda, S., Taing, N., Iwamura, M., Kise, K.: Towards universal khmer text recognition. arXiv preprint arXiv:2603.00702 (2026)

14. Kruy, V., Kameyama, W.: Preliminary experiment on khmer ocr. FIT (2010)

15. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33, 9459–9474 (2020)

16. Long, S., Qin, S., Panteleev, D., Bissacco, A., Fujii, Y., Raptis, M.: Icdar 2023 competition on hierarchical text detection and recognition. In: International Conference on Document Analysis and Recognition. pp. 483–497. Springer (2023)

17. Nom, V., Bakkali, S., Luqman, M.M., Coustaty, M., Ogier, J.M.: Khmerst: a lowresource khmer scene text detection and recognition benchmark. In: Proceedings of the Asian Conference on Computer Vision. pp. 1777–1792 (2024)

18. Nom, V., Keo, S., Bakkali, S., Luqman, M.M., Coustaty, M., Rossinyol, M., Ogier, J.M.: Wildkhmerst: a comprehensive dataset and benchmark for khmer scene text detection and recognition in the wild. In: International Conference on Document Analysis and Recognition. pp. 351–368. Springer (2025)

19. Paruchuri, V., Team, D.: Surya: A lightweight document ocr and analysis toolkit. https://github.com/datalab-to/surya (2025), gitHub repository

20. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

21. Sok, P., Taing, N.: Support vector machine (svm) based classifier for khmer printed character-set recognition. In: Signal and information processing association annual summit and conference (APSIPA), 2014 Asia-Pacific. pp. 1–9. IEEE (2014)

22. Tesseract OCR: Tesseract open source ocr engine. https : / / github . com / tesseract-ocr/tesseract (2024)

23. Valy, D., Verleysen, M., Chhun, S.: Data augmentation and text recognition on khmer historical manuscripts. In: 2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR). pp. 73–78. IEEE (2020)

24. Valy, D., Verleysen, M., Chhun, S., Burie, J.C.: A new khmer palm leaf manuscript dataset for document analysis and recognition: Sleukrith set. In: Proceedings of the 4th International Workshop on Historical Document Imaging and Processing. pp. 1–6 (2017)

25. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

26. Vitou, S., Y., K., Lany, M., Botum, C., Kor, S.: Khmer handwritten ocr using pretrained trocr architecture (10 2024). https://doi.org/10.13140/RG.2.2.22531. 92967

27. Yath, S.: Synthkhmer-10k. https : / / huggingface . co / datasets / seanghay / SynthKhmer-10k (2024), synthetic Khmer text/OCR dataset