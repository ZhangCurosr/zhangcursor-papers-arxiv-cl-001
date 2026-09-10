# The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding

Gilad D. Landau PNPL University of Oxford gilad.landau@jesus.ox.ac.uk

Dulhan Jayalath PNPL University of Oxford

Oiwi Parker Jones PNPL University of Oxford

## Abstract

Non-invasive speech decoding remains constrained by the low signal-to-noise ratio of neural recordings, which makes fine-grained reconstruction of phonemes or individual words difficult. Motivated by neuroscientific evidence that high-level semantic representations are distributed across cortical regions and evolve over slower temporal scales, we hypothesize that semantic content may provide a more suitable target for non-invasive decoding than low-level acoustic or lexical features. We introduce Brain2Semantics2Text, a method that reconstructs text through an intermediate semantic embedding space. Our model maps sentence-level MEG responses into a semantic manifold and then inverts the predicted embeddings into natural language. This semantic bottleneck enables recovery of high-level meaning without word-level alignment. We describe the core principles of the approach, its implementation, and the strategies used to mitigate the challenges of learning a reliable neural-to-semantic mapping. Finally, we compare against prior non-invasive Brain2Text methods and show improved sentence-level results.

## 1 Introduction

Speech decoding BCIs have long been a sought-after goal in both neuroscience and healthcare. Recent progress in invasive Brain2Text systems has demonstrated the feasibility of translating neural activity into language [Anumanchipalli et al., 2019, Moses et al., 2021, Willett et al., 2023, Card et al., 2024]. However, these invasive approaches require surgical implantation of intracranial electrodes, creating a strong incentive to develop non-invasive speech decoding solutions that are safer, more accessible, and easier to deploy.

Extending this paradigm to non-invasive systems remains a major challenge, largely due to their inherently lower signal-to-noise ratios. Lower signal fidelity makes it difficult to reliably decode fine grained linguistic units such as phonemes or individual words. In contrast, higher-order contextual semantic representations are spatially distributed across the cortex [Huth et al., 2016], exhibit substantial redundancy, and evolve over slower temporal scales [Gwilliams et al., 2025b]. These properties may make semantic representations particularly amenable to decoding from non-invasive neural recordings, whose spatial and temporal characteristics are better matched to distributed, slowly evolving signals.

Recently, a growing body of work in both the speech decoding literature and the neuroscience of language has converged on the view that speech comprehension and production rely on a hierarchical organization of neural representations [Gwilliams et al., 2025b]. Lower levels of this hierarchy are dominated by auditory and articulatory representations closely tied to the acoustic structure of speech, while progressively higher levels abstract away from these surface properties, giving rise to representations that are less dependent on specific phonetic or lexical features and more closely associated with the meaning of speech (Gwilliams et al. [2025a], Goldstein et al. [2025]).

![](images/9c46d2e8c585672b50f6faba0b64c4efc11eb2f5ecd349bde39dc5151bc931a6.jpg)  
Figure 1: (1) MEG responses are collected while the participant listens to spoken sentence stimuli. (2) The Brain Module extracts time-resolved neural features using spatial attention and dilated temporal convolutions. (3) A backbone with temporal masked attention pools the neural sequence into a fixed dimensional vector. (4) The vector is trained to be aligned with a pre-trained sentence-embedding space representing semantic content. (5) The predicted semantic embedding is inverted back to text.

Within this framework, the success of invasive approaches can be largely attributed to their ability to directly access high-fidelity neural signals from the auditory and articulatory components of speech processing, which occupy the lower levels of the cortical hierarchy (often supplemented by post-hoc language models that guide generation; [Willett et al., 2024]). The same hierarchical view also motivates directly targeting higher-level semantic representations as a distinct and parallel neural signal. If such representations can be reliably decoded, their defining properties—slow temporal dynamics, distributed cortical organization, and representational redundancy—are better matched to the spatial and temporal characteristics of non-invasive modalities such as fMRI, MEG, and EEG.

This work targets semantic representations in brain activity, focusing on higher-level stages of the speech-processing hierarchy. Brain2Semantics2Text maps sentence-length MEG responses during heard speech into a pretrained semantic embedding space, from which text is subsequently reconstructed. By constraining neural decoding to pass through this semantic bottleneck, the method aims to shift the objective toward higher-level speech representations that carry information about sentence-level meaning.

## Several aspects distinguish our method from previous work on Brain2Text:

• Semantic embedding inversion. We build on recent advances in semantic embedding inversion [Morris et al., 2023], which enables the reconstruction of text from semantic embeddings either as an intrinsic property of newer embedding models [Duquenne et al., 2023] or via general inversion techniques applicable to arbitrary pre-trained semantic spaces [Jha et al., 2025]. This allows us to frame speech decoding as semantic reconstruction rather than word or phoneme prediction.

• Sentence-level semantic decoding. Our method operates directly at the sentence level, targeting compositional semantic representations near the top of the speech-processing hierarchy. This formulation shifts the decoding problem away from exact word or phoneme recovery and toward reconstruction of the intended semantic content. As a result, it avoids dependence on precise word-level alignment and closed-vocabulary supervision, both of which are difficult to assume in realistic settings. Although full-sentence decoding from

MEG is ambitious, sentence-level semantic decoding is well matched to the distributed and temporally extended nature of high-level language representations, making it a promising direction for future non-invasive communication systems.

• Semantic decoding from MEG. Unlike most prior semantic decoding work, which relies on the high spatial resolution of fMRI [Tang et al., 2023], we leverage the largest singlesubject heard-speech MEG dataset of its kind to date. Decoding semantic-level information from MEG enables the joint exploitation of slow, distributed semantic signals and local, high-frequency neural activity within the same recordings, opening new possibilities for improved speech decoding.

## 2 Related Work

Semantic decoding. Semantic representations have previously been used as an intermediate target for non-invasive language decoding. Pereira et al. [2018] demonstrated that fMRI responses to sentences could be mapped into semantic embedding spaces, providing early evidence that distributed neural activity can be aligned with sentence-level meaning. More recently, Tang et al. [2023] reconstructed continuous perceived and imagined language from fMRI by mapping distributed cortical responses into semantic representations and using these representations to constrain language generation. These approaches exploit the high spatial resolution of fMRI to recover distributed semantic information, but sacrifice the temporal resolution available in electrophysiological recordings.

Wang et al. [2023] extended semantic reconstruction to MEG, demonstrating that semantic information can also be recovered from temporally resolved non-invasive recordings. Their approach, however, operates at the word level, reconstructing a temporally aligned sequence of contextual word embeddings that is subsequently used to generate continuous text. In contrast, our method treats the entire sentence as the unit of decoding, mapping sentence-length MEG responses directly into a single pre-trained sentence-level semantic embedding. This formulation makes the semantic representation itself the decoding bottleneck and removes the need for word-level alignment.

Auditory and speech-based MEG decoding. A complementary line of work maps MEG activity to representations closely tied to the acoustic structure of speech. Défossez et al. [2023] aligned MEG responses with Wav2Vec representations [Baevski et al., 2020], establishing a contrastive-learning framework and neural architecture that have influenced subsequent MEG decoding systems. More recent approaches align MEG with auditory representations [Yang et al., 2024c], adapt Whisper for neural speech decoding [Yang et al., 2024b], or incorporate neural signals into multimodal foundation model architectures [Yang et al., 2024a]. These approaches exploit neural information associated with the acoustic realization of speech, whereas our method deliberately targets higher-level semantic content.

Within this line of work, BrainECHO [Li et al., 2024] provides the closest comparison to our method. Like Brain2Semantics2Text, BrainECHO operates at the sentence level and does not require wordlevel alignment, but the two methods differ in the representation through which decoding proceeds. BrainECHO maps neural activity into a vector-quantized audio-spectrogram latent space before generating text with Whisper, whereas our method maps MEG directly into a sentence-level semantic embedding. BrainECHO therefore provides a particularly informative baseline for evaluating the use of semantic, rather than acoustic, representations as a bottleneck for sentence-level decoding.

Word-level classification. Semantic representations have also been used for word-level MEG decoding. d’Ascoli et al. [2024] achieve strong closed-vocabulary word classification by aligning MEG responses with lexical semantic embeddings augmented by sentence context. Their approach demonstrates the utility of semantic representations for MEG decoding, but relies on exact word-level timing and formulates decoding as classification among candidate words. Our approach instead targets a single compositional representation of the complete sentence, removing the requirement for word-level alignment at the cost of a substantially less constrained reconstruction problem.

## 3 Method

The Brain2Semantics2Text method operates in two stages. In the first stage, MEG neural responses corresponding to continuously presented spoken sentences are mapped to vector representations in the pre-trained semantic embedding space. Training is guided by objectives that encourage alignment with the target embeddings while preserving their global statistical structure. In the second stage, the predicted semantic embedding is inverted into natural language using a pre-trained inversion model [Morris et al., 2023] that reconstructs text from semantic vectors.

## 3.1 Backbone

MEG-to-semantic mapping. The MEG input signal $\mathbf { x } \in \mathbb { R } ^ { C \times T }$ , where C denotes the number of sensors and T the number of temporal samples, is first processed by a spatial attention module, followed by an initial $1 \times 1$ convolution that projects the sensor dimension into a latent feature space. The resulting representation is then passed through a stack of dilated temporal convolutional blocks. Each block consists of dilated convolutions equipped with residual connections and gated linear units (GLUs), enabling the model to capture long-range temporal dependencies.

A subject-specific layer can optionally be inserted after the initial projection to model inter-subject variability. In the experiments reported here all data originate from a single subject.

Temporal aggregation. To obtain a fixed-dimensional semantic representation from the timeresolved features, we use a 4-head self-attention Transformer over the temporal axis. We then pool over time with a masked mean

$$
\hat { \mathbf { y } } = \frac { \sum _ { t = 1 } ^ { T } m _ { t } H _ { t } } { \sum _ { t = 1 } ^ { T } m _ { t } } \in \mathbb { R } ^ { d } ,
$$

where $m _ { t }$ is a temporal mask based on the true segment lengths, preventing length-related surface confounders from influencing the pooled embedding.

## 3.2 Semantic Embeddings

Many semantic embedding models are available, with different architectures, training objectives, and benchmark performance [Muennighoff et al., 2022]. For Brain2Semantics2Text, however, standard evaluations on semantic similarity or retrieval tasks provide only partial guidance. Our method requires the embedding space to function as an invertible bottleneck between MEG responses and text, which introduces additional constraints beyond general semantic performance. In particular, the embedding must have an available inversion mechanism and must support reliable reconstruction from both exact text embeddings and imperfect neural predictions. We therefore evaluate candidate embedding spaces according to four criteria.

Expressivity. The embedding space should preserve enough information about the original sentence to support reconstruction. We measure this using a round-trip reconstruction test, in which each sentence is embedded and then inverted back into text. Higher reconstruction quality indicates that more sentence-level information is retained by the embedding and its inversion procedure.

Soft reversibility. At test time, the vectors being inverted are not exact text embeddings, but embeddings predicted from noisy MEG responses. The embedding space should therefore be robust to prediction error: vectors near the target should still invert to text with similar meaning. We assess this by perturbing target embeddings and calculating the relation between the introduced noise and reconstruction fidelity.

Length bias. The embedding should encode sentence meaning without being dominated by surfacelevel properties such as sentence length. This is particularly important because stimulus duration and sentence length may be available to the neural decoder and could provide a shortcut that competes with semantic learning. We estimate length bias by computing the Spearman rank correlation (ρ) between sentence length and the leading principal components of each embedding space. Although this measure does not capture all forms of embedding sensitivity to sentence length, we find that it provides an effective empirical proxy for the extent to which sentence length is reflected in the global geometry of the embedding space. A visual illustration of this analysis is provided in Appendix F.

Intrinsic dimensionality. The target space should be learnable from limited MEG data. We therefore prefer embedding spaces with lower intrinsic dimensionality, measured by effective rank, provided that they remain sufficiently expressive and reversible.

Table 1: Comparison of candidate embedding spaces according to expressivity, soft reversibility, length bias, and intrinsic dimensionality. ADA was chosen for its combination of high soft reversibility, low length bias, and good expressivity.
<table><tr><td>Embedding</td><td>Expressivity ↑</td><td>Soft reversibility ↑</td><td>Length bias ↓</td><td>Intrinsic dim. ↓</td></tr><tr><td>SONAR</td><td> $\mathbf { 0 . 9 8 1 \pm 0 . 0 1 9 }$ </td><td>0.902</td><td>0.819</td><td>1024/197</td></tr><tr><td>T5</td><td> $0 . 8 1 3 \pm 0 . 0 1 8$ </td><td>0.333</td><td>0.569</td><td>768/183</td></tr><tr><td>ADA</td><td> $0 . 8 8 8 \pm 0 . 0 3 1$ </td><td>0.925</td><td>0.432</td><td>1536/195</td></tr></table>

A further consideration is training-data transparency. In principle, a fully open embedding and inversion pipeline would be preferable, since it would allow us to verify that evaluation sentences were not present in the training data of either the embedding model or the inversion model. Among the embedding–inversion pairs we considered, however, we did not find a fully data-transparent option that also satisfied the practical requirements of expressivity and soft reversibility. We therefore control for possible corpus-leakage by comparing neural-based predictions against noise-control.

## 3.3 Objectives for Manifold Learning

At the core of our training setup is a SigLIP-style contrastive loss [Zhai et al., 2023, d’Ascoli et al., 2024], which has been shown to be effective for aligning representations across modalities with different dimensionalities and statistical characteristics [Radford et al., 2021]. However, in the low-data regime typical of non-invasive speech decoding, contrastive objectives alone are insufficient to learn the target manifold of semantic embeddings.

Previous brain-to-text and word-decoding approaches have largely relied on contrastive objectives to align neural signals with semantic embeddings [Défossez et al., 2023, d’Ascoli et al., 2024]. However, we observe that in the low-data regimes typical of non-invasive speech decoding, contrastive loss primarily optimizes a retrieval objective. In this setting, the model learns a mapping that enables nearest-neighbor matching under cosine similarity [Minnema and Herbelot, 2019], but does not necessarily preserve the inter-vector distances or the scale of embedding magnitudes.

This limitation is problematic for our setting, where the goal is not merely to retrieve a correct target embedding, but to learn a mapping that faithfully captures the global geometry of the target semantic manifold. To address this, we draw inspiration from the manifold learning literature [Meila and˘ Zhang, 2023] and introduce several auxiliary losses in addition to the SigLIP that force the model to learn the global properties of the target manifold and prevent collapse. The resulting training objective consists of the following components (invariance, covariance and variance losses are adopted from VICReg [Bardes et al., 2021]):

SigLIP Loss: A contrastive alignment term that formulates predicted–target matching as independent pairwise classification rather than a batch-wise softmax objective. This is useful for sentence-level semantic decoding, where different non-matching sentences may still be semantically related and should not necessarily be treated as mutually exclusive classes. We also found this objective more stable in the low-data MEG setting, particularly with small batches.

$$
\mathcal { L } _ { \mathrm { c t r } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \log \Bigl ( 1 + \exp \Bigl ( - t _ { i j } \bigl ( \tau \hat { \mathbf { y } } _ { i } ^ { \top } \mathbf { y } _ { j } + b \bigr ) \Bigr ) \Bigr ) ,\tag{1}
$$

Invariance Loss : mean squared distance between predicted and target embeddings.

$$
\mathcal { L } _ { \mathrm { i n v } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left. \mathbf { x } _ { i } - \mathbf { y } _ { i } \right. _ { 2 } ^ { 2 } .\tag{2}
$$

Covariance Loss: a decorrelation term that penalizes off-diagonal covariances between embedding dimensions, reducing redundancy and preventing informational collapse.

$$
\mathcal { L } _ { \mathrm { c o v } } = \frac { 1 } { d } \sum _ { i \neq j } C ( \mathbf { x } ) _ { i , j } ^ { 2 } .\tag{3}
$$

Table 2: LibriBrain’s Sherlock Holmes data splits used in our experiments. Validation and test sets are held-out recording sessions.
<table><tr><td>Stimuli</td><td>Words</td><td>Unique</td><td>Sentences</td><td>Hours</td></tr><tr><td>Sherlock Holmes Books (Train)</td><td>600,107</td><td>20,837</td><td>40,659</td><td>61.80</td></tr><tr><td>Sherlock Holmes Books (Validation)</td><td>3,427</td><td>1,155</td><td>198</td><td>0.36</td></tr><tr><td>Sherlock Holmes Books (Test)</td><td>3,577</td><td>1,210</td><td>172</td><td>0.38</td></tr><tr><td>Sherlock Holmes Books (Total)</td><td>607,111</td><td>20,971</td><td>41,029</td><td>62.54</td></tr></table>

$$
C ( \mathbf { x } ) = { \frac { 1 } { n - 1 } } \sum _ { k = 1 } ^ { n } ( \mathbf { x } _ { k } - { \bar { \mathbf { x } } } ) ( \mathbf { x } _ { k } - { \bar { \mathbf { x } } } ) ^ { \top } ,\tag{4}
$$

Variance Loss: a hinge loss that enforces a minimum standard deviation across the batch for each embedding dimension, preventing collapse:

$$
\mathcal { L } _ { \mathrm { v a r } } = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \operatorname* { m a x } ( 0 , \gamma - \sigma _ { j } ( \mathbf { x } ) ) ,\tag{5}
$$

$$
\sigma _ { j } ( \mathbf x ) = \sqrt { \mathrm { V a r } ( x ^ { j } ) + \epsilon } .\tag{6}
$$

Global Cosine Alignment Loss: maximizes the average cosine similarity between predicted and target embedding vectors.

$$
\mathcal { L } _ { \mathrm { g c s } } = 1 - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { \hat { \mathbf { y } } _ { i } ^ { \top } \mathbf { y } _ { i } } { \left\| \hat { \mathbf { y } } _ { i } \right\| _ { 2 } \left\| \mathbf { y } _ { i } \right\| _ { 2 } } .\tag{7}
$$

The final loss is:

$$
\mathcal { L } = \alpha \mathcal { L } _ { \mathrm { c t r } } + \beta \mathcal { L } _ { \mathrm { g c s } } + \gamma \mathcal { L } _ { \mathrm { i n v } } + \delta \mathcal { L } _ { \mathrm { v a r } } + \eta \mathcal { L } _ { \mathrm { c o v } } .\tag{8}
$$

To test whether each component contributes to the final decoding performance, we ablate individual terms from the training objective while keeping the rest of the pipeline fixed, as shown in Table 4.

## 3.4 Inverting Semantic Embeddings Back to Text

Embedding inversion [Morris et al., 2023] is formulated as an iterative conditional generation problem, where the objective is to recover a text sequence $x ^ { * }$ given only its embedding $e ^ { * } = f ( \bar { x ^ { * } } )$ . The procedure initializes by sampling an initial hypothesis from a base generator,

$$
x _ { 0 } \sim p _ { \theta } ( x \mid e ^ { * } ) .
$$

At each iteration $t ,$ the current hypothesis $x _ { t }$ is re-embedded to obtain $e _ { t } = f ( x _ { t } )$ , and a learned correction model generates an improved hypothesis conditioned on the current text and the embedding discrepancy:

$$
\boldsymbol { x } _ { t + 1 } \sim p _ { \phi } ( \boldsymbol { x } \mid \boldsymbol { x } _ { t } , \boldsymbol { e } _ { t } , e ^ { * } ) .
$$

This iterative refinement progressively reduces the embedding distance $\| e _ { t } - e ^ { * } \|$ , yielding increasingly faithful reconstructions without direct optimization in discrete token space.

## 3.5 Data

For training and evaluation, we use LibriBrain [Özdogan et al., 2025], the largest single-subject speech-decoding MEG dataset available at the time of writing. Specifically, we use the Sherlock Holmes subset, which provides over 62 hours of MEG recordings from a single participant listening to continuous spoken narrative. The validation and test sets are held-out recording sessions, allowing us to evaluate generalization across sessions rather than across randomly sampled sentences.

Text and audio were manually corrected, normalized, and force-aligned, with sentence boundaries defined by corpus punctuation. In this work, we prioritize dataset scale and semantic variability as key factors for semantic decoding, while deferring subject variability and cross-subject generalization to future studies. We chose MEG as our recording modality because it occupies a middle ground between fMRI and EEG: it offers high temporal resolution while providing substantially better spatial specificity than EEG. If MEG spatial resolution proves sufficient for capturing distributed semantic representations, this would open the possibility of jointly exploiting slow, distributed semantic signals and high-frequency auditory features within a single non-invasive modality.

Table 3: Comparison of decoding methods (mean ± s.d.). Brain2Semantics2Text and BrainECHO operate at the sentence level and do not rely on word-level alignment, whereas d’Ascoli et al. [2024] use word-aligned supervision. Lexical-overlap metrics such as WER, BLEU-1, and ROUGE-1 therefore favor methods with access to word-level timing or lexical supervision, while BERTScore better reflects the sentence-level semantic reconstruction objective of our method.
<table><tr><td>Method</td><td>WER↓</td><td>BLEU-1 ↑</td><td>ROUGE-1 ↑</td><td>BERTScore ↑</td></tr><tr><td colspan="5">Sentence-level decoding, no word-level alignment</td></tr><tr><td>Ours</td><td> $1 . 9 2 5 \pm 0 . 1 0 0$ </td><td> ${ \bf 0 . 1 0 0 \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 1 3 2 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 8 3 0 \pm 0 . 0 0 1 }$  </td></tr><tr><td>Ours (noise control)</td><td> $2 . 6 4 8 \pm 0 . 4 1 5$ </td><td> $0 . 0 8 6 \pm 0 . 0 0 9$ </td><td> $0 . 1 1 3 \pm 0 . 0 0 4$ </td><td> $0 . 8 1 9 \pm 0 . 0 0 6$ </td></tr><tr><td>BrainECHO</td><td> $1 . 0 1 8 \pm 0 . 0 3 0$ </td><td> $0 . 0 6 1 \pm 0 . 0 0 8$ </td><td> $0 . 0 9 1 \pm 0 . 0 0 7$ </td><td> $0 . 8 2 8 \pm 0 . 0 0 2$ </td></tr><tr><td>BrainECHO (noise control)</td><td> $1 . 0 1 2 \pm 0 . 0 0 5$ </td><td> $0 . 0 5 6 \pm 0 . 0 0 6$ </td><td> $0 . 0 8 5 \pm 0 . 0 0 8$ </td><td> $0 . 8 2 5 \pm 0 . 0 0 2$ </td></tr><tr><td colspan="5">Word-level decoding with word-aligned supervision</td></tr><tr><td>d’Ascoli et al.</td><td> $\mathbf { 0 . 8 7 1 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 1 9 0 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 1 7 2 \pm 0 . 0 0 4 }$ </td><td> $0 . 8 2 0 \pm 0 . 0 0 1$ </td></tr><tr><td>d’Ascoli et al. (noise control)</td><td> $0 . 9 9 4 \pm 0 . 0 0 3$ </td><td> $0 . 0 7 2 \pm 0 . 0 2 4$ </td><td> $0 . 0 5 7 \pm 0 . 0 2 1$ </td><td> $0 . 7 9 4 \pm 0 . 0 0 4$ </td></tr></table>

## 3.5.1 Preprocessing

The recordings were originally sampled at 1 kHz and downsampled to 250 Hz to preserve oscillations into the high-gamma range (70–125 Hz).

## 4 Experiments

We compare our results against prior Brain2Text approaches using standard text-generation metrics: WER, BLEU, ROUGE, and BERTScore. While WER, BLEU, and ROUGE primarily measure lexical overlap and word-level reconstruction accuracy, BERTScore provides a complementary estimate of sentence-level semantic similarity. This is particularly important for our setting, where successful decoding may preserve the meaning of a sentence even when its exact wording is not recovered. At the same time, text-generation metrics alone cannot determine whether a decoded sentence was driven by neural information or by linguistic and dataset-level priors in the generation model. Following Jo et al. [2024], we therefore include a noise-control analysis to estimate how much of the decoded output is attributable to the neural input rather than to textual priors alone.

Reversing semantic embeddings reliably requires learning a high-fidelity representation of the target semantic space; otherwise, inversion becomes infeasible. It is therefore critical to identify the minimum training-set size required for the method to become reliable. To this end, we report empirical scaling laws demonstrating that the fidelity of semantic-embedding mapping improves systematically with data scale, and we identify a minimum data regime beyond which the method becomes feasible.

## 4.1 Results

Comparison of method performance. Table 3 compares the proposed Brain2Semantics2Text approach with prior Brain2Text decoding methods. The word-level metrics reveal a clear performance gap between word-level decoding, represented by d’Ascoli [d’Ascoli et al., 2024], and sentence-level decoding. This gap is expected, since word-level decoding operates in a much more constrained setting, with a closed vocabulary and access to exact word-level alignment. By contrast, when compared with acoustic-based sentence-level decoding, represented by BrainECHO [Li et al., 2024], our method performs favorably, achieving better results on BLEU-1, ROUGE-1, and BERTScore. These improvements hold both in absolute terms and when measured as neural-signal uplift relative to the noise control (Figure 3). Our method tends to generate more words than appear in the ground-truth sentence, making WER less informative; recall-based metrics such as ROUGE-1, however, show a distinct signal-based improvement.

![](images/48825fc79e9e43a1de98b351f90b2151cb089c4be05e9379b185780c6aa8a544.jpg)  
Figure 2: Semantic embedding inversion. We evaluated semantic embedding inversion by adding controlled Gaussian noise with increasing standard deviation to the sentence embedding vectors and measuring the BERTScore F1 of the reconstructed text. Expressivity is defined as the reconstruction score obtained from the unperturbed embedding, corresponding to the ideal case of a perfectly predicted vector. Soft reversibility is measured by how smoothly reconstruction quality degrades as embedding noise increases, summarized by the $R ^ { 2 }$ of the fitted relationship between noise level and BERTScore. Actual neural predictions are overlaid to show where model outputs fall along the resulting noise–performance curve (similar analyses for all candidate embeddings are provided in Appendix F).

Table 4: Loss ablation results measured as mean decoded-sentence BERTScore F1 on the test set. Values are mean ± standard deviation across runs. Higher values indicate better decoded-sentence semantic similarity.
<table><tr><td>Method</td><td>BERTScore</td></tr><tr><td>Full objective (SigLIP + VICReg + global cosine)</td><td> $\mathbf { 0 . 8 2 9 7 \mathop { \pm } 0 . 0 0 0 8 }$ </td></tr><tr><td>w/o global cosine loss</td><td> $0 . 8 1 0 3 \pm 0 . 0 0 7 7$ </td></tr><tr><td>w/o VICReg invariance term</td><td> $0 . 8 2 6 3 \pm 0 . 0 0 2 5$ </td></tr><tr><td>w/o VICReg variance term</td><td> $0 . 8 2 6 7 \pm 0 . 0 0 1 5$ </td></tr><tr><td>w/o VICReg covariance term</td><td> $0 . 8 2 7 0 \pm 0 . 0 0 1 2$ </td></tr></table>

Signal uplift. Figure 3 evaluates the contribution of the neural signal across methods, measured using both BERTScore and ADA cosine similarity. BERTScore serves as the standard metric for comparing sentence-level reconstructions, while ADA cosine similarity provides a complementary measure of semantic similarity that more directly reflects the explicit training target of Brain2Semantics2Text. Because all methods may exploit text priors and corpus-level regularities that are not driven by the brain signal, we measure performance relative to a noise-control baseline. This signal-dependent uplift estimates the contribution of the neural signal itself, rather than reconstruction quality attributable to the inversion model, language prior, or corpus-level bias.

![](images/6c99055eb85639a4001c3f4952b0cf257ba2ede0ced1a412faabf648275b7c31.jpg)

![](images/79d6682f470cc1d2fad7831ac7dcecbb79bc2929b54149f97b4672e2bfd02f03.jpg)  
Figure 3: Signal uplift. Comparison of signal-dependent improvement over noise-control baselines across decoding methods.

![](images/93c13f256ccae6a42be7f16000653c8bd5d9f98d7f3219712095f9f50db169c3.jpg)  
Figure 4: Scaling Laws

On BERTScore, our method yields a 1.2-point uplift over its noise-control baseline, exceeding the uplift of the previous sentence-level method, BrainECHO, but remaining below the 3.2-point uplift of d’Ascoli et al. However, d’Ascoli et al. operate at the word level and require exact word-aligned supervision, whereas our method performs sentence-level decoding without word-level alignment. On ADA cosine similarity, our method shows the largest signal-dependent uplift, improving by 6.0 points and exceeding the corresponding uplift of the word-level method. This discrepancy between semantic metrics suggests that current evaluation measures capture different aspects of sentence-level decoding performance. More broadly, it highlights the need for better standardized metrics for evaluating semantic reconstruction in brain-decoding models.

Scaling behavior of the Brain2Semantics2Text method. The method shows promising scaling behavior with increasing amounts of training data. Performance generally improved as training data increased, with gains appearing to saturate around 55.6 hours. These results indicate that the method makes effective use of additional MEG recordings of spoken language, supporting the potential value of larger-scale data collection while suggesting that future improvements may also benefit from increased data diversity.

We report performance using a retrieval-based metric: “Discounted Cumulative Gain” [Järvelin and Kekäläinen, 2002], rather than the text-based metrics, as below a certain performance threshold the full semantic inversion pipeline does not operate reliably and sentences reconstructions are not available. The retrieval metric evaluates the model’s ability to identify the closest matching sentence in the embedding space and, as such, does not capture global structural properties of the semantic manifold. Nevertheless, when all other factors remain constant, the retrieval performance provides a meaningful proxy for the overall effectiveness of the proposed method.

## 5 Limitations and Future Work

The current work represents an initial attempt at speech decoding through a direct mapping of MEG signals into semantic representations. Learning this semantic manifold proved to be challenging. Beyond the usual constraints of non-invasive neural recordings, such as low SNR and limited training data, reliable semantic encoding is likely to require greater semantic variability in the training corpus. We observed that the model learned semantic structure that was strongly biased toward the specific corpus used in this study. Therefore, future work should involve data collection protocols that emphasize variability in topics and concepts, perhaps guided by the properties of the target semantic manifold [Pereira et al., 2018].

The inversion methods used in this work were treated as black-box components and may not be optimally suited for semantic decoding. Further work is needed to better understand the “soft reversibility” of predicted vectors, introduced in the soft-reversibility analysis (Section 3.2), and to optimize the preservation of semantic content, potentially by training inversion mechanisms specifically for this purpose.

The current work also does not address subject variability. Although this issue is beyond the present study, semantic representations may offer new routes for cross-subject generalization by modeling both shared semantic structure and participant-specific profiles in semantic representation space.

## 6 Conclusion

While fully non-invasive speech decoding remains a long-term goal, decoding contextual semantic content from brain activity represents an important step toward its realization. Neuroscientific evidence suggests that speech is processed across multiple levels of representation, from fast acoustic and lexical features to slower, higher-level semantic information. These levels may provide complementary targets for neural decoding. Recovering contextual semantics from MEG alongside lower-level speech information could therefore provide multiple sources of information for reconstruction, helping compensate for the limited signal quality of non-invasive neural recordings.

## References

Gopala K. Anumanchipalli, Josh Chartier, and Edward F. Chang. Speech synthesis from neural decoding of spoken sentences. Nature, 568:493–498, 2019. doi: 10.1038/s41586-019-1119-1.

Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Advances in Neural Information Processing Systems (NeurIPS 2020), 2020. URL https://arxiv.org/abs/2006.11477. arXiv:2006.11477.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In Advances in Neural Information Processing Systems (NeurIPS), 2021. doi: 10.48550/arXiv.2105.04906. URL https://arxiv.org/abs/2105.04906.

Nicholas S. Card, Maitreyee Wairagkar, Carrina Iacobacci, Xianda Hou, Tyler Singer-Clark, Francis R. Willett, Erin M. Kunz, Chaofei Fan, Maryam Vahdati Nia, Darrel R. Deo, Aparna Srinivasan, Eun Young Choi, Matthew F. Glasser, Leigh R. Hochberg, Kiarash Shahlaie, Sergey D. Stavisky, and David M. Brandman. An accurate and rapidly calibrating speech neuroprosthesis. New England Journal of Medicine, 391(7):609–618, 2024. doi: 10.1056/NEJMoa2314132. URL https://www.nejm.org/doi/full/10.1056/NEJMoa2314132.

Stéphane d’Ascoli, Corentin Bel, Jérémy Rapin, Hubert Banville, Yohann Benchetrit, Christophe Pallier, and Jean-Rémi King. Decoding individual words from non-invasive brain recordings across

723 participants. arXiv preprint arXiv:2412.17829, 2024. URL https://arxiv.org/abs/2412. 17829.

Paul-Ambroise Duquenne, Holger Schwenk, and Benoît Sagot. Sonar: Sentence-level multimodal and language-agnostic representations, 2023. URL https://arxiv.org/abs/2308.11466.

Alexandre Défossez, Charlotte Caucheteux, Jérémy Rapin, Ori Kabeli, and Jean-Rémi King. Decoding speech perception from non-invasive brain recordings. Nature Machine Intelligence, 5: 1097–1107, 2023. doi: 10.1038/s42256-023-00714-5.

Ariel Goldstein, Eric Ham, Mariano Schain, Samuel A. Nastase, Bobbi Aubrey, Zaid Zada, Avigail Grinstein-Dabush, Harshvardhan Gazula, Amir Feder, Werner Doyle, Sara Devore, Philip Dugan, Daniel Friedman, Michael Brenner, Amir Hassidim, Yair Matias, Orrin Devinsky, Nathan Siegelman, Alan Flinker, Oren Levy, Roi Reichart, and Uri Hasson. Temporal structure of natural language processing in the human brain corresponds to layered hierarchy of large language models. Nature Communications, 16(1):10529, 2025. doi: 10.1038/s41467-025-65518-0. URL https://www.nature.com/articles/s41467-025-65518-0.

Laura Gwilliams, Ilina Bhaya-Grossman, Yizhen Zhang, Terri Scott, Sarah Harper, and Deborah Levy. Computational architecture of speech comprehension in the human brain. Annual Review of Linguistics, 11:209–226, 2025a. doi: 10.1146/annurev-linguistics-031120-111245. URL https: //www.annualreviews.org/doi/10.1146/annurev-linguistics-031120-111245.

Lawrence Gwilliams et al. Hierarchical dynamic coding coordinates speech comprehension in the human brain. bioRxiv Preprint, 2025b. doi: 10.1101/2024.04.19.590280v2. URL https:// pmc.ncbi.nlm.nih.gov/articles/PMC11042271/. Preprint. PMCID: PMC11042271; PMID: 38659750.

Alexander G. Huth, Wendy A. de Heer, Thomas L. Griffiths, Frédéric E. Theunissen, and Jack L. Gallant. Natural speech reveals the semantic maps that tile human cerebral cortex. Nature, 532(7600):453–458, 2016. doi: 10.1038/nature17637. URL https://doi.org/10.1038/ nature17637.

Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems, 20(4):422–446, 2002.

Rishi Jha, Collin Zhang, Vitaly Shmatikov, and John X. Morris. Harnessing the universal geometry of embeddings, 2025. URL https://arxiv.org/abs/2505.12540.

Hyejeong Jo, Yiqian Yang, Juhyeok Han, Yiqun Duan, Hui Xiong, and Won Hee Lee. Are eeg-to-text models working? arXiv preprint arXiv:2405.06459, 2024. doi: 10.48550/arXiv.2405.06459. URL https://arxiv.org/abs/2405.06459.

Jilong Li, Zhenxi Song, Jiaqi Wang, Min Zhang, and Zhiguo Zhang. Brainecho: Semantic brain signal decoding through vector-quantized spectrogram reconstruction for whisper-enhanced text generation. arXiv preprint arXiv:2410.14971, 2024. doi: 10.48550/arXiv.2410.14971. URL https://arxiv.org/abs/2410.14971.

Marina Meila and Hanyu Zhang. Manifold learning: what, how, and why. ˘ arXiv preprint arXiv:2311.03757, 2023. doi: 10.48550/arXiv.2311.03757. URL https://arxiv.org/abs/ 2311.03757.

Gosse Minnema and Aurélie Herbelot. From brain space to distributional space: The perilous journeys of fmri decoding. In Fernando Alva-Manchego, Eunsol Choi, and Daniel Khashabi, editors, Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics: Student Research Workshop, pages 155–161, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-2021. URL https://aclanthology.org/P19-2021/.

John X. Morris, Wenting Zhao, Justin T. Chiu, Vitaly Shmatikov, and Alexander M. Rush. Language model inversion. arXiv, abs/2311.13647, 2023. URL https://arxiv.org/abs/2311.13647. Preprint.

David A. Moses, Sean L. Metzger, Jessie R. Liu, Gopala K. Anumanchipalli, Joseph G. Makin, Pengfei F. Sun, Josh Chartier, Maximilian E. Dougherty, Patricia M. Liu, Gary M. Abrams, Adelyn Tu-Chan, Karunesh Ganguly, and Edward F. Chang. Neuroprosthesis for decoding speech in a paralyzed person with anarthria. New England Journal ofMedicine, 385(3):217–227, 2021. doi: 10.1056/NEJMoa2027540.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316, 2022. doi: 10.48550/arXiv.2210.07316. URL https://arxiv.org/abs/2210.07316.

Francisco Pereira, Bin Lou, Bennett Pritchett, Samuel Ritter, Samuel J. Gershman, Nancy Kanwisher, Matthew Botvinick, and Evelina Fedorenko. Toward a universal decoder of linguistic meaning from brain activation. Nature Communications, 9(1):963, 2018. doi: 10.1038/s41467-018-03068-4.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. URL https://arxiv.org/abs/2103.00020.

Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth. Semantic reconstruction of continuous language from non-invasive brain recordings. Nature Neuroscience, 26(5):858–866, 2023. doi: 10.1038/s41593-023-01304-9.

Bo Wang, Xiran Xu, Longxiang Zhang, Boda Xiao, Xihong Wu, and Jing Chen. Semantic reconstruction of continuous language from meg signals. arXiv, abs/2309.07701, 2023. URL https://arxiv.org/abs/2309.07701.

Francis R. Willett, Erin M. Kunz, Chaofei Fan, Donald T. Avansino, Guy H. Wilson, Eun Young Choi, Foram Kamdar, Matthew F. Glasser, Leigh R. Hochberg, Shaul Druckmann, Krishna V. Shenoy, and Jaimie M. Henderson. A high-performance speech neuroprosthesis. Nature, 620 (7976):1031–1036, 2023. doi: 10.1038/s41586-023-06377-x. URL https://www.nature.com/ articles/s41586-023-06377-x.

Francis R. Willett, Jingyuan Li, Trung Le, Chaofei Fan, Mingfei Chen, Eli Shlizerman, Yue Chen, Xin Zheng, Tatsuo S. Okubo, Tyler Benster, Hyun Dong Lee, Maxwell Kounga, E. Kelly Buchanan, David Zoltowski, Scott W. Linderman, and Jaimie M. Henderson. Brain-to-text benchmark ’24: Lessons learned. arXiv preprint, arXiv:2412.17227, 2024. doi: 10.48550/arXiv.2412.17227. URL https://arxiv.org/abs/2412.17227.

Yiqian Yang, Yiqun Duan, Hyejeong Jo, Qiang Zhang, Renjing Xu, Oiwi Parker Jones, Xuming Hu, Chin-teng Lin, and Hui Xiong. Neugpt: Unified multi-modal neural gpt. arXiv preprint arXiv:2410.20916, 2024a. URL https://arxiv.org/abs/2410.20916.

Yiqian Yang, Yiqun Duan, Qiang Zhang, Hyejeong Jo, Jinni Zhou, Won Hee Lee, Renjing Xu, and Hui Xiong. Neuspeech: Decode neural signal as speech. arXiv preprint arXiv:2403.01748, 2024b. URL https://arxiv.org/abs/2403.01748.

Yiqian Yang, Hyejeong Jo, Yiqun Duan, Qiang Zhang, Jinni Zhou, Won Hee Lee, Renjing Xu, and Hui Xiong. Mad: Multi-alignment meg-to-text decoding. arXiv, abs/2406.01512, 2024c. URL https://arxiv.org/abs/2406.01512.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2023. URL https://arxiv.org/abs/2303.15343.

Miran Özdogan, Gilad Landau, Gereon Elvers, Dulhan Jayalath, Pratik Somaiya, Francesco Mantegna, Mark Woolrich, and Oiwi Parker Jones. LibriBrain: Over 50 hours of within-subject MEG to improve speech decoding methods at scale. arXiv preprint arXiv:2506.02098, 2025. doi: 10.48550/arXiv.2506.02098. URL https://arxiv.org/abs/2506.02098.

## A Impact Statement

This paper presents work toward non-invasive speech decoding, with potential applications in braincomputer interfaces for individuals who have lost the ability to speak. Clinical deployment remains distant, as current performance is still below what communication aids require, and substantial further work is needed. We also note that neural decoding technologies raise clear privacy concerns, since they involve inferring mental content from brain activity. Our work uses only publicly available research datasets with their own ethics approvals and decodes perceived speech rather than covert thought. As decoding capabilities improve, the field will need norms around consent, data ownership, and the boundary between assistive and surveillant applications. These are questions we do not resolve here, but consider essential.

## B Hyperparameters

Table 5: Hyperparameters used for MEG-to-semantic embedding training.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Dropout</td><td>0.6</td></tr><tr><td>Learning rate Optimizer</td><td> $3 \times 1 0 ^ { - 5 }$  AdamW</td></tr><tr><td>Loss type</td><td>SigLIP</td></tr><tr><td>VICReg weight</td><td>5.0</td></tr><tr><td>Cosine loss weight</td><td>6.0</td></tr><tr><td>Contrastive loss weight</td><td></td></tr><tr><td></td><td>2.0</td></tr><tr><td>Aggregation</td><td>Attention, 4 heads</td></tr></table>

## C Compute Resources

All MEG-to-semantic embedding models were trained on a single NVIDIA GPU using 4 CPU cores and 64 GiB of system memory. A typical full-data run took approximately 16–18 GPUhours, corresponding to about 6 minutes per epoch for 150–170 epochs. Final training across seven random seeds used approximately 110–130 GPU-hours, excluding exploratory runs, failed jobs, and downstream evaluation or decoding.

## D Decoding Examples

![](images/94d91676fde966bbf448042cbcdfdf0669113e2e56773673ce8ad51fd852428e.jpg)  
Figure 5: Semantic geometry of reconstructed sentence embeddings. PCA projections of sentencelevel embeddings for ground-truth sentences and reconstructed predictions. Highlighted examples illustrate that relative positions and global structure are largely preserved under reconstruction. Examples are shown for geometric comparison only, and textual reconstructions may differ from the ground truth.

![](images/c851e7c5d82f7f81a72a8cf23cfa9e8fe55a64fab3fdcee8941c603b980cfca2.jpg)  
Figure 6: Qualitative decoding examples. Example decoded sentences shown for qualitative illustration of semantic similarity between the target sentence and the reconstructed output.

## E Neural Signal Ablations

![](images/8699818040d7200241d2cfe58e4c96b12bfd231d010082deb0786282c697cfdc.jpg)  
Figure 7: Neural signal properties supporting semantic decoding. We examine how different properties of the MEG signal affect semantic mapping performance, providing an interpretable view of which aspects of the neural response contribute most to decoding.

## F Semantic Embedding Analysis

![](images/6b7341eb8be6963729b82d54c68b0725cce72c13c408f8460dff8641cec8863e.jpg)

![](images/46fb3387a72a79bcf8881281af431098ff08bf9cb7971e9ef2d1ad96e1915131.jpg)

![](images/b3e2a53122b39c76b4893cc42b2f6379ff59d75635d84def441b036150384d5c.jpg)  
Figure 8: PCA comparison of ADA and SONAR semantic embeddings, colored by sentence length. While SONAR embeddings segregate sentences by length, forming length-dependent regions in the embedding space, ADA embeddings remain more invariant to sentence length. This suggests that ADA is less vulnerable to the sentence-length confound.

![](images/2a1412e2180b7f8d8b17c5c13558e40165ecb84631d3a38b376d012c6b6c2856.jpg)  
Figure 9: Comparison of candidate embedding spaces across embedding-space diagnostics. Candidate semantic embedding spaces are compared according to expressivity, soft reversibility, length bias, and intrinsic dimensionality. A full explanation of how these measures are computed is provided in the appendix.