# Reading Time Modeling with Pixel-Based Language Models

Junlin Li<sup>1</sup> Sara Alhajeri<sup>1</sup> Tatsuki Kuribayashi<sup>1,2</sup>

<sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence <sup>2</sup>Tohoku University {junlin.li, sara.alhajeri, tatsuki.kuribayashi}@mbzuai.ac.ae

## Abstract

Tokenization-free, pixel-based language models (PLMs) have recently been proposed as an alternative to conventional token-based language models (TLMs), typically motivated by practical considerations, e.g., the flexibility to support new languages or vocabularies. Nevertheless, PLMs also have the potential to serve as infrastructure for computational (psycho)linguistics, where tokenization has been a source of bias in estimating informationtheoretic values, such as surprisal, and has pushed the bottom-up visual orthographic features outside the scope. In this study, we first integrate PLMs into the modeling of naturalistic human reading and, in particular, examine their use to estimate next-word predictability, which can also integrate orthographic and parafoveal information.

Our results demonstrate that next-word prediction difficulty, computed in pixel space, has strong predictive power for gaze duration and word skipping behavior across various languages. In particular, their advantage is complementary to existing token-level surprisal and entropy estimates. Cross-lingual analyses with script variations also clarify their distinctive visual processing biases, facilitating research into how bottom-up visual information is incrementally integrated into expectation-based reading.

§ reading-time-modeling-with-pixel-lm

## 1 Introduction

Reading coordinates a stratification of low- and high-level cognitive processes (Engbert et al., 2005) across multiple modalities (Hoover and Gough, 1990). Naturalistic reading data, for example, indicate that a considerable proportion of words is processed without a fixation; they are possibly recognized in the parafovea (Schotter et al., 2012; Rayner, 1998) with only partial visual information (Reichle et al., 2003). Such a rapid pre-attentive visual processing suggests the importance of a bottom-up sensory prediction (Kok and de Lange, 2015; Tarkiainen et al., 1999), especially how graded visual presentation affects reading time and skipping (Gordon et al., 2013) via top-down prediction (Ding et al., 2025).

![](images/6c5d8621d3d304439e8264436c6dfdda3f2c0a7ab863d19363b37095deed090f.jpg)  
Figure 1: Simulating visual sentence processing with orthographic and parafoveal information using pixelbased LMs

In computational psycholinguistics, neural language models (LMs) have been ubiquitous in estimating information-theoretic values, such as surprisal (Hale, 2001; Levy, 2008), to instantiate linguistic theories. Despite their successful interdisciplinary bridge, recent studies have also identified drawbacks of ordinal LMs defined over a fixed set of token IDs (token-based language models; TLMs) as tools for cognitive modeling. In particular, their tokenizers and vocabularies, predefined regardless of their cognitive plausibility (Haslett, 2025), potentially limit feasible experimental designs, e.g., estimating highly incremental probabilities across more fine-grained units, or implicitly bias experimental results (Nair and Resnik, 2023; Oh and Schuler, 2024; Giulianelli et al., 2024; Oh and Schuler, 2025; Kiegeland et al., 2026). For example, Nair and Resnik (2023) reported that a tokenizer or (sub)word definition easily biases the surprisal estimate − log p(word|context). Another example is that a naive accumulation of subword probabilities results in their misallocation to the following word, e.g., “He give me”, where the load of “give” can be dislocated to “me,” if the alternative “gives” is also split into the common prefix “give” and “s” (Oh and Schuler, 2024; Giulianelli et al., 2024). Although some problems have quick solutions, tokenizer selection is inherently the source of biases in computational psycholinguistic experiments with TLMs. More fundamentally, TLM also limits the possibility of experimental design; for example, the word-to-ID conversion makes it impossible to study the visual and orthographic effects in reading.

In this paper, to handle such tokenization bottlenecks spread in modern TLM-based cognitive modeling, we examine the use of tokenization-free, pixel-based language models (PLMs) (Borenstein et al., 2023; Rust et al., 2023; Lotz et al., 2023) in computational psycholinguistics. PLMs encode text as pixel-based images, potentially without loss of visual fidelity to what humans can indeed see, enabling flexible research designs, for example, to explore how text layout, font, or orthographic complexity drives reading (Borenstein et al., 2023). This work, as a case study, specifically explores the psychometric predictive power of PLMs and parafoveal effects in reading, positing that earlystage visual processing may already begin before fixating on the exact word, and from an expectationbased reading perspective, visual word identifiability will be a strong predictor of human gaze duration and scan path. We compute word predictability and entropy with PLMs that rely on orthographic information and a blurred future context, intended to reflect parafoveal sight. These estimates are used to model reading time and word skipping behavior during naturalistic reading.

Our experiments address the following questions:

• Q1: Do PLMs without a tokenizer predict reading time and word skipping better than or complementary to TLM surprisal, which relies on a tokenizer?

• Q2 : Does PLM-estimated visual wordprediction difficulty offer a general predictor for cognitive modeling across different languages and writing systems?

Our experimental results show that the visual word prediction difficulty estimated by PLMs offers complementary predictive power to conventional token-based surprisal. This advantage is observed across languages, particularly in those with scripts other than the Latin alphabet, demonstrating that PLM is a useful infrastructure for estimating word predictability and modeling human reading across diverse languages.

## 2 Related Work

## 2.1 Reading Time Modeling with Token-based Language Models

Language models (LMs) have been widely used as tools to implement and quantitatively test (neuro and psycho)linguistic theories (Caucheteux et al., 2023; Ohams et al., 2026). Given that LMs are basically next-word prediction machines, their use is typically relevant to testing an expectation-based account of human sentence processing through estimating information-theoretic measures such as surprisal (Hale, 2001; Levy, 2008; Hale, 2016). Curiously, a misalignment between human reading measures and LM-estimated predictive measures has been indicated by recent studies (Kuribayashi et al., 2021; Shain et al., 2024; Oh and Schuler, 2023), raising several possible explanations. A conceptually intuitive implication is that prediction is not a complete representation of human sentence processing (Slaats and Martin, 2025), which features structural modularistic underpinnings (Coltheart et al., 2001) and thus is not purely probabilistic. But practically, another possibility rests on the biases of the proxies, i.e., LMs, in estimating predictive measures (Nguyen and Arehalli, 2026). Existing studies (Nair and Resnik, 2023; Giulianelli et al., 2024; Oh and Schuler, 2024, 2025), for example, demonstrate that the tokenizer substantially affects LM-based cognitive modeling. Besides, language prediction based on token identity prevents any bottom-up visual and orthographic decoding processes. This study explores a more neutral, realistic design for text encoding in LM-based cognitive modeling, i.e., simply inputting the text image as we read texts with our eyes.

## 2.2 Pixel-Based Language Models

Fortunately, there are practical techniques for processing language using text images as input. Pixelbased language model (PLM) (Rust et al., 2023), using a visual Transformer (ViT) as its backbone, is a representative model that seamlessly models language in a tokenizer-free paradigm, regardless of vocabulary and character-inventory constraints (Lotz et al., 2023). Such PLMs have been adopted, for example, to enable a flexible addition of a new language or to process low-resource languages, such as historical languages without transcriptions (Borenstein et al., 2023). Beneath their powerful performance in the visual domain, probing analyses indicate that PLMS also learn linguistic knowledge, such as semantics and syntax (Tatariya et al., 2024) and exhibit cross-script generalizability through multilingual pre-training (Kesen et al., 2025). In contrast to these studies, we examine the use of PLMs for cognitive modeling.

![](images/c1fea040155ed7b4b4f4e8b4858011be62a27984c5232f983c20ab7c2276cf8a.jpg)  
Figure 2: We approximate next-word predictability by patch-by-patch masking and blurring of the future context, simulating parafoveal sight, using PLM.

## 3 Methods

This section briefly introduces PLMs, which model text sequences in the visual modality without relying on a tokenizer or a discrete vocabulary. We then explain how to compute next-word predictability with PLMs.

## 3.1 Pixel-Based Language Models

Given a token sequence $\pmb { w } = [ w _ { 1 } , w _ { 2 } , \dots , w _ { n } ]$ a rendering function converts the sequence into an image $\textbf { \textit { x } } \in \ \mathbb { R } ^ { H \times W \times C }$ , where C, H, and W denote the number of channels (e.g., RGB), height, and width, respectively. The rendered image is divided into $\begin{array} { r } { \dot { N } \ = \ \frac { H \mathbf { \bar { W } } } { P ^ { 2 } } } \end{array}$ non-overlapping patches $\mathbf { \Delta X } = [ \pmb { x } _ { 1 } , \cdots , \pmb { x } _ { N } ] \mathrm { ~ \hat { \ t } { \Psi } ~ } \mathbb { R } ^ { N \times P ^ { 2 } \times C }$ , where P is the height/width of each patch. Each patch $\pmb { x } _ { i } \in \mathbb { R } ^ { P ^ { 2 } \times \breve { C } }$ is flattened and projected into a ddimensional embedding $\mathbf { \boldsymbol { h } } _ { i } \in \mathbb { R } ^ { d }$ with an embedding layer of a PLM:

$$
\pmb { h } _ { i } = \mathrm { F l a t t e n } ( \pmb { x } _ { i } ) \pmb { W } _ { E } + \pmb { b } _ { E } + \pmb { e } _ { i } ^ { \mathrm { p o s } } ,\tag{1}
$$

where $W _ { E } \in \mathbb { R } ^ { P ^ { 2 } C \times d }$ and $\pmb { b } _ { E } \in \mathbb { R } ^ { d }$ are learnable parameters, and $e _ { i } ^ { \mathrm { p o s } } \in \mathbb { R } ^ { d }$ denotes the positional

![](images/6332b479fb03ca4116a3529d132cd94d99d976c1eefeb85747c7d0dac57a3deb.jpg)  
Figure 3: Examples of visual prediction from PLMs across multiple languages. The grayed areas are the PLM predictions for the regions originally masked.

embedding of the i-th patch. $\pmb { H } = [ \pmb { h } _ { 1 } , \dotsc , \pmb { h } _ { N } ]$   
denotes the sequence of patch embeddings.

During training, a subset of patches in H is randomly masked. Let indices for masked patches be $\mathcal { T } \subset [ 1 , \cdots , N ]$ . The masked input $H _ { \mathrm { m a s k e d } }$ is processed by a PLM F to reconstruct the original, unmasked version of the patches:<sup>1</sup>

$$
\hat { X } = \mathcal { F } ( H _ { \mathrm { m a s k e d } } ) \mathrm { ~ , ~ }\tag{2}
$$

where $\hat { \pmb X } = \{ \hat { \pmb x } _ { 1 } , \dots , \hat { \pmb x } _ { N } \} \in \mathbb { R } ^ { N \times P ^ { 2 } \times C }$ . Then the reconstruction loss L is calculated over the indices I where the input was masked, and basically, the following objective is minimized:

$$
\mathcal { L } = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { I } } \left. \hat { \pmb x } _ { i } - { \pmb x } _ { i } \right. _ { 2 } ^ { 2 } \ \mathrm { ~ . ~ }\tag{3}
$$

## 3.2 Visual Word Prediction Difficulty (VWPD) in Context

PLMs can quantify the prediction difficulty (VWPD) of the next word in the pixel space. Given a sentence $\pmb { w } = [ w _ { 1 } , \cdots , w _ { n } ]$ and its image patches $\pmb { X } \ = \ [ \pmb { x } _ { 1 } , \cdots , \pmb { x } _ { N } ]$ , suppose a target token w<sub>t</sub> spans from s<sub>t</sub>-th to e<sub>t</sub>-th patches $[ { \pmb x } _ { s _ { t } } , \cdots , { \pmb x } _ { e _ { t } } ] \subset { \pmb X }$ in the rendered image $( 1 ~ \leq$ $s _ { t } \leq e _ { t } \leq N )$ . A token may span over several patches, and a straightforward way to compute the next-word prediction difficulty is:

$$
\vec { I } ( w _ { t } | { \bf w } _ { < t } ) : = \sum _ { k = s _ { t } } ^ { e _ { t } } \vec { s } ( { \bf x } _ { k } | X _ { < k } ) ,\tag{4}
$$

where the target patches $[ \pmb { x } _ { s _ { t } } , \cdots , \pmb { x } _ { e _ { t } } ]$ are masked one-by-one to enforce the prediction (black masking parts in Figure 2). A function $s ( \cdot )$ quantifies the reconstruction error of the actual visual token (also see Figure 3 for examples of predictions). For example, one can compute the instance-wise MSE in the pixel space:

$$
\vec { s } ( \mathbf { x } _ { k } | \mathbf { X } _ { < k } ) = \left. \vec { \tilde { x } } _ { k } - \mathbf { x } _ { k } \right. _ { 2 } ^ { 2 } \ ,\tag{5}
$$

$$
\vec { \tilde { \bf x } } _ { k } = \mathcal { F } _ { k } ( X _ { < k } \circ m ( { \pmb x } _ { k } ) ) ~ ,\tag{6}
$$

where $\mathcal { F } _ { k }$ returns the k-th patch of the output, ◦ denotes the concatenation of patches, and $m ( \cdot )$ is a masking function. A high prediction error of patches $\vec { s } ( \pmb { x } _ { k } | \pmb { X } _ { < k } )$ is aggregated into the prediction difficulty of the word in Eq. 4; a high value of $\vec { I } ( w _ { t } | \mathbf { \boldsymbol { w } } _ { < t } )$ is expected to be linked with longer reading time. Another reconstruction error is also examined in Section 4.

Off-the-shelf PLMs, however, are typically implemented with a bidirectional encoder or encoderdecoder architecture, and fully omitting future context $X _ { > t }$ from the input may incur a severe traininginference gap. Reflecting our vision, the future context is not fully invisible, but parafoveal sight still catches some superficial cues, e.g., approximate word length or sentence length. Thus, we propose the simulation of parafoveal sight by feeding blurred future context information $X _ { > t }$ with a blurring function $g ( \cdot )$ (see Figure 2).<sup>2</sup> This blurring design is ablated in the main experiments. Now, the prediction difficulty of a word $w _ { t }$ is redefined as follows:

$$
\overleftrightarrow { \boldsymbol { I } } ^ { \prime } ( w _ { t } | \pmb { w } _ { \backslash t } ) : = \sum _ { k = s _ { t } } ^ { e _ { t } } \overleftrightarrow { \boldsymbol { s } } ^ { \prime } ( \pmb { x } _ { k } | \pmb { X } _ { < k } , g ( \pmb { X } _ { > k } ) ) ,\tag{7}
$$

$$
\overleftrightarrow { s } ^ { \prime } ( { \pmb x } _ { k } | { \pmb X } _ { < k } , g ( { \pmb X } _ { > k } ) ) = \left\| \overleftrightarrow { \tilde { \pmb x } } _ { k } - { \pmb x } _ { k } \right\| _ { 2 } ^ { 2 } \ ,\tag{8}
$$

$$
\overleftrightarrow { \pmb { \tilde { x } } } _ { k } = \mathcal { F } _ { k } ( \pmb { X } _ { < k } \circ m ( \pmb { x } _ { k } ) \circ g ( \pmb { X } _ { > k } ) ) .\tag{9}
$$

where the prediction difficulty $ _ { T } { } _ { ( w _ { t } | w _ { \backslash t } ) }$ is conditioned by bi-directional context (with partial blurring to the future context). Through our experiments, we use this version (Eq. 7) of visual word predictability.

## 3.3 SSIM Reconstruction Measure

For the reconstruction error term, $\vec { s } ( \cdot )$ in Eq. 5 or $ _ { s } ( . )$ in Eq. 8, another formulation beyond instance-wise MSE can also be examined. For example, structural similarity (SSIM) (Wang et al., 2004) is widely used in the computer vision field. Given that the SSIM is a similarity measure, one minus the SSIM can be adopted for estimating VWPD:

$$
\begin{array} { r l } & { 1 - \mathrm { S S I M } ( \tilde { x } _ { k } , x _ { k } ) } \\ & { = 1 - \displaystyle \frac { ( 2 \mu _ { \tilde { x } } \mu _ { x } + C _ { 1 } ) ( 2 \sigma _ { \tilde { x } x } + C _ { 2 } ) } { ( \mu _ { \tilde { x } } ^ { 2 } + \mu _ { x } ^ { 2 } + C _ { 1 } ) ( \sigma _ { \tilde { x } } ^ { 2 } + \sigma _ { x } ^ { 2 } + C _ { 2 } ) } \ : \ : , } \end{array}\tag{10}
$$

where $\mu _ { \tilde { x } }$ and $\mu _ { x }$ denote the mean pixel intensities of the reconstructed patch $\tilde { \mathbfit { x } } _ { k }$ and ground-truth patch $\mathbf { \nabla } _ { \mathbf { x } _ { k } } .$ , respectively. $\sigma _ { \tilde { x } } ^ { 2 }$ and $\sigma _ { x } ^ { 2 }$ denote their variances, $\sigma _ { \tilde { x } x }$ denotes the covariance between them, and $C _ { 1 }$ and $C _ { 2 }$ are small constants for numerical stability. One can apply this formula in Eq. 5 or Eq. 8 instead of instance-wise MSE, and we preliminarily examine this variant in Section 5.1.

## 3.4 Visual Prediction Uncertainty in Context

We also examined the entropy of next-word prediction in the pixel space, which quantifies the uncertainty in predicting black (stroke) or white (no stroke) rather than noisy gray predictions. Each dimension in a reconstructed pixel $\hat { p } _ { k }$ at position k is first normalized with a sigmoid function (0 for white pixel prediction and 1 for black pixel prediction):

$$
\hat { p } _ { k } = \sigma ( \stackrel { \longleftrightarrow } { \hat { x } } _ { k } ) \in [ 0 , 1 ] ^ { P ^ { 2 } \times C } \mathrm { ~ . ~ }\tag{11}
$$

The entropy measure is computed as follows:

$$
H ( w _ { t } | \pmb { w } _ { \backslash t } ) : = \sum _ { k = s _ { t } } ^ { e _ { t } } \sum _ { l = 1 } ^ { P } \sum _ { m = 1 } ^ { P } \sum _ { c = 1 } ^ { C } h ( \hat { p } _ { k , l , m , c } ) ,\tag{12}
$$

$$
h ( p ) = - p \log p - ( 1 - p ) \log ( 1 - p ) ~ ;\tag{13}
$$

where $\hat { p } _ { k , l , m , c } \in [ 0 , 1 ]$ denotes (l, m)-th element in c-th channel of k-th patch prediction. A high uncertainty value may parallel with the general difficulty of anticipating the next word, expected to be linked with longer reading time. We also examine the entropy measures from PLMs and TLMs for word skipping modeling.

## 4 Experimental Settings

## 4.1 Language Models

We examine PLMs as well as TLMs with a comparable model size. All model details are in Table 2 in Appendix A.

PLM Settings. For English experiments, we evaluate pixel-m4. For Chinese reading, we also use the off-the-shelf pixel-m4 model (Kesen et al., 2025), which has been pretrained on Chinese script. For other languages, we finetune pixel-m4 on 100,000 examples from the allenai/c4 dataset (Raffel et al., 2020) for 150,000 steps with a batch size of 32. The finetuned models reach a loss below 0.020, comparable to the official checkpoint on the pre-training data. For the blurring function $g ( \cdot )$ in Eq. 8, we use Gaussian Blurring with a kernel size of 5 and a sigma of 3 (see Appendix B.2). We also examine a no-blurring condition, in which the future context is fully visible, similarly to BERT (Devlin et al., 2019).

TLM Settings. As baselines, we use monolingual GPT2 (Radford et al., 2019), monolingual BERT (Devlin et al., 2019), and multilingual mGPT (Shliazhko et al., 2024). Comparisons between TLMs and PLMs with parafoveal information clarify the general advantage of orthographically grounded prediction measures compared to discrete token prediction without any access to future word information. BERT is also compared, given the bi-directional nature of the PLM encoder. We use the surprisal (Hale, 2001; Levy, 2008) and entropy (Hale, 2016) computed by TLMs as predictors. Specifically, for the GPT and mGPT models, the surprisal of a word is calculated as:

$$
\vec { I } _ { \mathrm { t o k } } ( w _ { t } | \boldsymbol { w } _ { < t } ) = - \log _ { 2 } p ( w _ { t } \mid w _ { 1 } , . . . , w _ { t - 1 } ) .\tag{14}
$$

For the BERT models, the surprisal is computed with a bidirectional context:

<table><tr><td>Lang.</td><td>Dataset Code</td><td>Data</td><td>Skip Rate</td><td>Avg. TRT</td><td></td><td>#Subj. Script</td></tr><tr><td>German</td><td>Deu (meco po) Deu (meco zu)</td><td>MECO-2 (ge_po) MECO-2 (ge_zu)</td><td>0.19 0.21</td><td>380.8 390.6</td><td>40 45</td><td rowspan="3">Latin</td></tr><tr><td>English</td><td>Eng (meco en) Eng (meco uk)</td><td>MECO-1 (en) MECO-2 (en_uk)</td><td>0.30 0.28</td><td>346.1 348.2</td><td>46 50</td></tr><tr><td>Spanish</td><td>Esp (meco es) Esp (meco ch)</td><td>MECO-1 (es) MECO-1 (es_ch)</td><td>0.33 0.29</td><td>406.7 367.7</td><td>48 45</td></tr><tr><td>Persian</td><td>Far</td><td>PSR</td><td>0.13</td><td>450.4</td><td>60</td><td>Abjad</td></tr><tr><td>Korean</td><td>Kor</td><td>MECO-1 (ko)</td><td>0.31</td><td>348.9</td><td>32</td><td>Hangul</td></tr><tr><td rowspan="2">Chinese</td><td>Simp (meco)</td><td>MECO-2 (ch-s)</td><td>0.43</td><td>348.7</td><td>39</td><td rowspan="2">Hanzi</td></tr><tr><td>Trad (meco)</td><td>MECO-2 (ch-t)</td><td>0.44</td><td>435.3</td><td>38</td></tr><tr><td>Turkish Tur</td><td></td><td>MECO-2 (tur)</td><td>0.19</td><td>363.7</td><td>16</td><td>Latin</td></tr></table>

Table 1: We use multilingual and multi-script eyemovement-while-reading data across 459 readers.

$$
\begin{array} { r } { \overleftrightarrow { \boldsymbol { T } } _ { \mathrm { t o k } } ( w _ { t } | \boldsymbol { w } _ { \backslash t } ) = - \log _ { 2 } p ( w _ { t } \mid w _ { 1 } , \ldots , w _ { t - 1 } , } \\ { \lbrack \boldsymbol { \mathsf { M A S K } } ] , w _ { t + 1 } , \ldots , w _ { n } ) \ . \ ( 1 5 ) } \end{array}
$$

Entropy is calculated as the expected value of surprisal over the model’s tokenizer vocabulary (Pimentel et al., 2023).

## 4.2 Human Reading Data

We model reading time and word skipping behavior of a total of 459 subjects from different eyemovement-while-reading corpora, covering German (Deu), English (Eng), Spanish (Esp), Farsi (Far), Korean (Kor), Simplified Chinese (Simp), Traditional Chinese (Trad), and Turkish (Tur). Data are from MECO Wave 1 (Siegelman et al., 2022) or Wave 2 (Siegelman et al., 2025), except for Persian reading corpus (Tekbudak et al., 2024). Data statistics and details are presented in Table 1. We take the following reading measures into consideration: First Fixation Duration (FFD), First-pass Reading Time (FPT), Total Reading Time (TRT), First-pass Skipping (FSKP), and Overall Skipping (SKP). For durative measures, any zero fixation duration and zero reading time are excluded. We adopt two word skipping measures: (i) first-pass skipping — whether an interest area is skipped before a later interest area is fixated (1=yes, 0=no); and (ii) Overall skipping — whether an interest area is not fixated throughout the current trial (1=yes, 0=no).

## 4.3 Regression Models

Predictors of Interest. We extract visual prediction difficulty (Eq. 7) and visual entropy (Eq. 12) from PLMs, as well as the widely used token-level

surprisal and entropy from TLMs. For each regression model, we only use one predictor of interest, denoted as x. The regression formula is as follows:

$$
\begin{array} { r l r } & { } & { \mathrm { \lbrace y _ { n } \rbrace \sim \lbrace x _ { n } \rbrace + l _ { n } + f _ { \mathrm { \ell } } + l _ { n - 1 } + f _ { n - 1 } + i d x _ { n } } } \\ & { } & { ~ + \left( 1 + \lbrace \mathbf { x _ { n } \rbrace + l _ { n - 1 } + f _ { n - 1 } \vert \mathrm { s u b j e c t \_ i d } } \right) } \\ & { } & { ~ + \left( 1 \vert \mathrm { s t i m u l i \_ i d } \right) ~ , ~ \ ~ ( 1 6 ) } \end{array}
$$

where l is word length, and f is zipf word frequency, $\mathrm { i d x } _ { \mathrm { n } }$ is the word index in sentence. The regression formula for the baseline is as follows:

$$
\begin{array} { r l } & { \{ \mathrm { y _ { n } } \} \sim \mathrm { l _ { n } } + \mathrm { f } _ { n } + \mathrm { l _ { n - 1 } } + \mathrm { f _ { n - 1 } } + \mathrm { i d x _ { n } } } \\ & { \qquad + \left( 1 + \mathrm { l _ { n - 1 } } + \mathrm { f _ { n - 1 } } | \mathrm { s u b j e c t \_ i d } \right) } \\ & { \qquad + \left( 1 | \mathrm { s t i m u l i \_ i d } \right) \ . } \end{array}\tag{17}
$$

Here, y is the reading measures, i.e., FFD, FPT, TRT, FSKP, or SKP. For reading measures (FFD, FPT, and TRT), we applied log transformation, and skipping measures (FSKP and SKP) are converted with binomial family. We conduct our main experiments with generalized additive mixed models (GAMM) (Sóskuthy, 2017; Wood et al., 2015), using mgcv packages (Wood, 2017) and setting all effects as linear. For main experiments, we use the following designs of x:

• pixel: VWPD from PLM, without blurring the future context.

• pixel\_uni: VWPD from PLM, with blurring the future context.

• gpt: surprisal from the respective monolingual GPT2.

• mgpt: surprisal from multilingual GPT.

• bert: bi-directional surprisal from the respective monolingual BERT.

• pixel\_uni + gpt: pixel\_uni and gpt are both included, without interaction.

• pixel\_uni ∗ gpt: Interaction between pixel\_uni and gpt with their main effects.

## 4.4 Evaluation Metrics

Following existing studies (Goodkind and Bicknell, 2018), we use the difference in loglikelihood between two nested models with and without the predictor of interest ${ \tt x } _ { \tt n }$ from TLMs or PLMs. In our case, the joint contribution of the predictor and its subject-specific random slope is measured. We refer to this metric as psychometric predictive power (PPP). To test robustness, we rerun each formula 100 times using a crossed random-effects bootstrap with replacement to obtain a distribution of the loglikelihood difference. Similarly to Goodkind and Bicknell (2018), we fit regression models to explain the data of interest, but in Appendix D, we also adopt 10-fold cross-validation, as in (Wilcox et al., 2023), yielding consistent results. When reporting a single PPP value for each language with several datasets (Table 1), PPPs for different datasets in the target language are averaged.

![](images/c2fc37eec78fcccf27910c3f35477170fd7845916bbf67089751e71c13a38687.jpg)  
Figure 4: MSE-based and SSIM-based VWPD estimates show slight, but different PPP patterns across languages. The error bars in the figures indicate 95% percentile bootstrap confidence intervals of PPP across the settings.

## 5 Experimental Results

## 5.1 Preliminary Experiments: Comparing Two VWPD Measures

In Section 3.2, we introduced MSE and SSIM as an image reconstruction error in Eq. 8. As preliminary experiments, we compare MSE-based and SSIM-based image reconstruction errors as general indicators of next-word prediction difficulty. Figure 4 shows that different PPPs are obtained with different error formulations, and there was a clear empirical trend that SSIM is slightly better or comparable to MSE, except for the Chinese (Simplified and Traditional) language. This might reflect a special visual processing of logograms or the complexity of Chinese characters. In our main experiments, we show SSIM results across all languages, except for MSE for the Chinese (Simplified and Traditional) data. Full SSIM and MSE comparison is shown in Appendix E.

![](images/0f106966dde4062fc7700bcab965e6e860c75b6882312e49c14f1dde90d42727.jpg)

Figure 5: PPP (larger better) of VWPD and token surprisal for fixation measures (FFD = First Fixation Duration, FPT = First-pass Reading time, TRT = Total Reading Time). All zero-duration or zero-reading-time is filtered to disentangle from skipping measures. The Y-axis is PPP. The error bars in the figures indicate standard deviation.  
![](images/096662ca9fc0b907760745022dec4a9337ff410e73b75cac9e83308b0132c07f.jpg)  
Figure 6: PPP (larger better) of VWPD and token surprisal for skipping measures (FSKP = First-pass skipping, SKP = Overall skipping). The Y-axis is PPP. The error bars in the figures indicate standard deviation.

## 5.2 Comparing PLMs and TLMs

We evaluate two versions of PLMs with and without blurring for the future context, and three TLMs (monolingual GPT2, BERT, and mGPT).

Fixation and Reading Time Figure 5 shows the results for FFD, FPT, and TRT modeling. Visual word predictability (VWPD) is used for PLMs, and surprisal is used for TLMs. PPPs for PLMs are somewhat worse than TLMs, but when these two predictors are jointly used, the PPP substantially increases, indicating their complementary advantage in reading time modeling.

Word Skipping Figure 6 shows the results for FSKP and SKP; whether the word is skipped or not during reading is predicted. In contrast to gaze duration measures, PLMs yield comparable or better predictive power, especially for the Farsi (Far), Korean (Kor), and Chinese (Simp and Trad) languages; interestingly, these languages do not adopt the Latin alphabet. In these languages, in other words, if a word can easily be predicted visually from the surrounding context, it is likely to be skipped during reading. This idea is also compatible with entropy measures (how the next word is uncertain), and Figure 7 shows the comparison of entropy computed by PLMs and TLMs. The results generally reproduce our observations with surprisal, i.e., token-level and visual entropy are complementary. We report the cross-validation analysis in appendix D, which substantiates the tendencies we have pointed out above.

![](images/7aafc7ab371b3578af289650657f2d3a71d9be0dca8c3c0de503e2a354154058.jpg)  
Figure 7: PPP (larger better) of visual and token entropy for skipping measures (FSKP = First-pass skipping, SKP = Overall skipping). The Y-axis is PPP. The error bars in the figures indicate standard deviation.

## 5.3 Comparing Addition and Interaction

Comparing the PPP of the additive model (pixel\_uni + gpt) and the interactive model (pixel\_uni ∗ gpt), we sometimes observe an advantage of the interaction term, especially in the first-pass measures (FFD and FSKP) in German (Deu), Spanish (Esp), and Turkish (Tur). These have somewhat longer word length and richer morphology than those of other languages (Siegelman et al., 2022). Thus, a strong interaction may imply a more general hierarchical engagement of prediction in reading, spanning from shallow visual perception to morphological or richer linguistic processing. In the following section, we look into how this interaction shapes first-pass reading behavior. The detailed breakdown of our main results into individual dataset is in Figures 13 and 14.

6 Analysis: Characteristics of VWPD

## 6.1 Interaction with TLM Surprisal

Given the complementary advantage of predictors from PLM and TLM, we exploratorily extend the analysis, specifically on how PLM and TLM predictors interact. We focus on early-stage reading measures, such as first fixation duration (FFD), which typically aligns with early lexical access (Kurnik, 2016), first-pass reading time (FPT), as well as firstpass refixation duration (FPT minus FFD; FPREF), which indicates the “failure to achieve lexical access” (Lowell and Morris, 2014). For simplicity, we only consider the fixed effect of PLM surprisal (namely VWPD) and TLM surprisal (we use GPT2 surprisal in each language). We split the data into those with relatively high surprisal (+1 SD) and low surprisal (−1 SD) and analyze how the VWPD effect differs between the two subsets.

Figure 8 shows the interaction between VWPD and GPT2 surprisal. Interestingly, for FFD, we found that high surprisal (+1 SD in the figure) can reverse (e.g., in Deu and Tur) the VWPD–FFD relationship to be negative. In contrast to FFD, FPREF results indicate a positive mediation of VWPD, as high surprisal (+1SD in the figure) leads to a positive VWPD-FPREF relationship. That is, words with a high surprisal (lexical difficulty) tend to weaken the VWPD effect for first fixation but strengthen it for refixation. One possible interpretation of the neutralized or negative interaction between VWPD and surprisal in FFD prediction is that lexical prediction (token-level surprisal) may involve pre-activation of visual representations (Buggy et al., 2026); if more effort is devoted to predicting the next word, early-stage visual processing can be skipped. The positive interaction in FPREF aligns with the finding that orthographic confusion is significant only for unpredictable words (Yao et al., 2022), as the positive interaction was only observed when people need multiple fixations (FPREF), potentially when lexical access initially failed.

![](images/ce33b4f7f4aed9f331df2783007b62367f404511349e7e85e05826c3e4024098.jpg)  
Figure 8: Interaction between VWPD and GPT2 surprisal (+1 $\mathrm { S D } = \mathrm { h i g h } , - 1 \mathrm { S D } = \mathrm { l o w } )$ to FFD, FPT, and FPREF.

![](images/3a26750859d62447c633b40e8c4fc8400f36aa22dc639db48f8deeda8dd54f74.jpg)  
Figure 9: Effect of Unidirectional Gaussian Blurring. The points are obtained through bootstrapping, and the blue (left) ones are from the pixel setting without blurring, while the red (right) ones are from the pixel\_uni one with blurring. Y-axis is PPP.

## 6.2 Gaussian Blurring Effects

We also analyze how Gaussian blurring of the future context affects PPP. The improvement in predictive power (See Figure 9) with the blurring treatment is not substantial. Still, the advantage is slightly pronounced for SKP in Farsi (Far), Korean (Kor), and Traditional Chinese (Trad), where non-Latin alphabets are used. The advantage of blurring in Trad has also been suggested by the finding that reading Traditional Chinese characters narrows the perceptual span (Pan and Yan, 2024), supporting an information-processing account of visual perception during reading, as assumed in the OB1-Reader model (Snell et al., 2018). Nevertheless, our Gaussian Blurring is our tentative implementation, and more cognitively motivated designs should be explored in future studies.

## 7 Conclusion

We demonstrate that pixel-based LMs can estimate expectation-based visual word-processing measures that complement those estimated by tokenbased LMs, which are especially relevant to prefixation processing on partial input. This study is the first to introduce a pixel-based language model into computational cognitive modeling. The results demonstrate, at least partial cognitive plausibility of PLM, especially its alignment with early-stage language processing, such as first-pass skipping.

## Limitations

First, a pixel-based language model has various rendering strategies (Lotz et al., 2023). We only use the most cross-lingually generalizable setting, namely bigram rendering, which is the foundation of the only multilingual pixel-based model so far. We agree that calculating visual word prediction metrics is highly sensitive to the text rendering strategy. We encourage future work to explore different rendering strategies and analyze their effect on the cognitive plausibility of pixel-based language modeling. Second, for pixel blurring, we have not conducted an intensive hyperparameter analysis regarding the effects of kernel size and sigma on model performance. We also only tried Gaussian Blurring, without exploring other image corruption methods such as Gaussian Noise. Future work could also compare more cognitively motivated parafoveal blurring and noise. Besides, the human parafovea exhibits graded visual acuity, which is not accounted for in this study. We hope that the current study initiates the primary work and establishes the foundational settings, and that future work will continue to explore parafoveal vision modeling by manipulating the gradedness of visual acuity.

## Ethical Considerations

This study uses open-source eye-movement-whilereading data collected from IRB-approved studies. No personally identifiable information is used in our experiment. Our study aims to gain knowledge about human cognition. We do not curate any system to replace humans in terms of reading.

## AI Writing/Coding Assistance Policy

AI writing assistance systems like Grammarly and ChatGPT are occasionally used to correct grammatical and spelling errors and to adjust phrasing. Our use of writing assistants complies with the AI writing assistance policy described in ARR guidelines under category (a), assistance purely with the language of the paper.

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Numbers 23K16938 and 26H02511.

## References

Nadav Borenstein, Phillip Rust, Desmond Elliott, and Isabelle Augenstein. 2023. PHD: Pixel-based language modeling of historical documents. In Proceedings of EMNLP 2023, pages 87–107.

Ryan Buggy, Stephanie Cho, Cory Shain, and Adrian Staub. 2026. Predictability Effects in Natural

Reading are Logarithmic: Evidence from an Eye-Movement Replication of Brothers and Kuperberg (2021). Journal ofExperimental Psychology: Learning, Memory, and Cognition.

Charlotte Caucheteux, Alexandre Gramfort, and Jean-Rémi King. 2023. Evidence of a predictive coding hierarchy in the human brain listening to speech. Nature Human Behaviour, 7(3):430–441.

Max Coltheart, Kathleen Rastle, Conrad Perry, Robyn Langdon, and Johannes Ziegler. 2001. DRC: a dual route cascaded model of visual word recognition and reading aloud. Psychological review, 108(1):204– 256.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings NAACL 2019, pages 4171– 4186.

Cui Ding, Yanning Yin, Lena Ann Jäger, and Ethan Wilcox. 2025. Modeling bottom-up information quality during language processing. In Proceedings of EMNLP 2025, pages 11709–11721.

Ralf Engbert, Antje Nuthmann, Eike M Richter, and Reinhold Kliegl. 2005. SWIFT: a dynamical model of saccade generation during reading. Psychological review, 112(4):777.

Mario Giulianelli, Luca Malagutti, Juan Luis Gastaldi, Brian DuSell, Tim Vieira, and Ryan Cotterell. 2024. On the proper treatment of tokenization in psycholinguistics. In Proceedings of EMNLP 2024, pages 18556–18572.

Adam Goodkind and Klinton Bicknell. 2018. Predictive Power of Word Surprisal for Reading Times is a Linear Function of Language Model Quality. In Proceedings ofCMCL 2018, pages 10–18.

Peter C Gordon, Patrick Plummer, and Wonil Choi. 2013. See before you jump: full recognition of parafoveal words precedes skips during reading. Journal ofExperimental Psychology: Learning, Memory, and Cognition, 39(2):633–641.

John Hale. 2001. A probabilistic Earley parser as a psycholinguistic model. In Proceedings ofNAACL 2001.

John Hale. 2016. Information-theoretical complexity metrics. Language and Linguistics Compass, 10(9):397–412.

David A. Haslett. 2025. Tokenization Changes Meaning in Large Language Models: Evidence from Chinese. Computational Linguistics, 51(3):785–814.

Wesley A Hoover and Philip B Gough. 1990. The simple view of reading. Reading and Writing, 2(2):127– 160.

Ilker Kesen, Jonas F. Lotz, Ingo Ziegler, Phillip Rust, and Desmond Elliott. 2025. Multilingual pretraining for pixel language models. In Proceedings of EMNLP 2025, pages 29594–29611.

Samuel Kiegeland, Vésteinn Snæbjarnarson, Tim Vieira, and Ryan Cotterell. 2026. On the Proper Treatment of Units in Surprisal Theory. In Proceedings of ACL 2026, pages 32202–32224.

Peter Kok and Floris P. de Lange. 2015. Predictive coding in sensory cortex. In Birte U. Forstmann and Eric-Jan Wagenmakers, editors, An Introduction to Model-Based Cognitive Neuroscience, pages 221– 244.

Tatsuki Kuribayashi, Yohei Oseki, Takumi Ito, Ryo Yoshida, Masayuki Asahara, and Kentaro Inui. 2021. Lower perplexity is not always human-like. In Proceedings ofACL-IJCNLP 2021, pages 5203–5217.

Mattias Kurnik. 2016. Bilingual lexical access in reading : Analyzing the effect of semantic context on non-selective access in bilingual memory. Master’s thesis, Stockholm University, Centre for Research on Bilingualism.

Roger Levy. 2008. Expectation-based Syntactic Comprehension. Cognition, 106(3):1126–1177.

Jonas Lotz, Elizabeth Salesky, Phillip Rust, and Desmond Elliott. 2023. Text Rendering Strategies for Pixel Language Models. In Proceedings of EMNLP 2023, pages 10155–10172.

Randy Lowell and Robin K Morris. 2014. Word length effects on novel words: Evidence from eye movements. Attention, Perception, & Psychophysics, 76(1):179–189.

Sathvik Nair and Philip Resnik. 2023. Words, subwords, and morphemes: What really matters in the surprisalreading time relationship? In Findings of EMNLP 2023, pages 11251–11260.

Kien Nguyen and Suhas Arehalli. 2026. Word predictability estimates from language models are not robust to tokenizer vocabulary. In Proceedings of CoNLL 2026, pages 34–44.

Byung-Doh Oh and William Schuler. 2023. Why Does Surprisal From Larger Transformer-Based Language Models Provide a Poorer Fit to Human Reading Times? Transactions of the Association for Computational Linguistics, 11:336–350.

Byung-Doh Oh and William Schuler. 2024. Leading Whitespaces of Language Models’ Subword Vocabulary Pose a Confound for Calculating Word Probabilities. In Proceedings of EMNLP 2024, pages 3464–3472.

Byung-Doh Oh and William Schuler. 2025. The impact of token granularity on the predictive power of language model surprisal. In Proceedings of ACL 2025, pages 4150–4162.

Chiebuka Ohams, Sathvik Nair, Shohini Bhattasali, and Philip Resnik. 2026. A predictive coding model for online sentence processing. Journal ofMemory and Language, 146:104705.

Jinger Pan and Ming Yan. 2024. The perceptual span in traditional chinese. Language and Cognition, 16(1):134–147.

Tiago Pimentel, Clara Meister, Ethan G Wilcox, Roger P Levy, and Ryan Cotterell. 2023. On the effect of anticipation on reading times. Transactions of the Associationfor Computational Linguistics, 11:1624– 1642.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67.

Keith Rayner. 1998. Eye Movements in Reading and Information Processing: 20 Years of Research. Psychological Bulletin, 124(3):372–422.

Erik D. Reichle, Keith Rayner, and Alexander Pollatsek. 2003. The E-Z Reader model of eye-movement control in reading: Comparisons to other models. Behavioral and Brain Sciences, 26(4):445–476.

Phillip Rust, Jonas F. Lotz, Emanuele Bugliarello, Elizabeth Salesky, Miryam de Lhoneux, and Desmond Elliott. 2023. Language Modelling with Pixels. In Proceedings ofICLR 2023.

Elizabeth R Schotter, Bernhard Angele, and Keith Rayner. 2012. Parafoveal processing in reading. Attention, Perception, & Psychophysics, 74(1):5–35.

Cory Shain, Clara Meister, Tiago Pimentel, Ryan Cotterell, and Roger Levy. 2024. Large-Scale Evidence for Logarithmic Effects of Word Predictability on Reading Time. Proceedings ofthe National Academy ofSciences, 121(10).

Oleh Shliazhko, Alena Fenogenova, Maria Tikhonova, Anastasia Kozlova, Vladislav Mikhailov, and Tatiana Shavrina. 2024. mGPT: Few-shot learners go multilingual. Transactions ofthe Associationfor Computational Linguistics, 12:58–79.

Noam Siegelman, Sascha Schroeder, Cengiz Acartürk, Hee-Don Ahn, Svetlana Alexeeva, Simona Amenta, Raymond Bertram, Rolando Bonandrini, Marc Brysbaert, Daria Chernova, Sara Maria Da Fonseca, Nicolas Dirix, Wouter Duyck, Argyro Fella, Ram Frost, Carolina A. Gattei, Areti Kalaitzi, Nayoung Kwon, Kaidi Lõo, and 12 others. 2022. Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). Behavior Research Methods, 54(6):2843–2863.

Noam Siegelman, Sascha Schroeder, Yaqian Borogjoon Bao, Cengiz Acartürk, Niket Agrawal, Lena S. Bolliger, Jan Brasser, César Campos-Rojas, Denis Drieghe, Dušica Filipovic Ður ´ devi ¯ c, Sofya Goldina,´ Romualdo Ibáñez Orellana, Lena A. Jäger, Ómar I. Jóhannesson, Anurag Khare, Nik Kharlamov, Hanne B. S. Knudsen, Árni Kristjánsson, Charlotte E. Lee, and 30 others. 2025. Wave 2 of the Multilingual Eye-Movement Corpus (MECO): New text reading data across languages. Scientific Data, 12(1):1183.

Sophie Slaats and Andrea E Martin. 2025. What’s surprising about surprisal. Computational Brain & Behavior, 8(2):233–248.

Joshua Snell, Sam van Leipsig, Jonathan Grainger, and Martijn Meeter. 2018. Ob1-reader: A model of word recognition and eye movements in text reading. Psychological review, 125(6):969–984.

Márton Sóskuthy. 2017. Generalised additive mixed models for dynamic analysis in linguistics: A practical introduction. arXiv preprint arXiv:1703.05339.

Antti Tarkiainen, Päivi Helenius, Peter C Hansen, Piers L Cornelissen, and Riitta Salmelin. 1999. Dynamics of letter string perception in the human occipitotemporal cortex. Brain, 122(11):2119–2132.

Kushal Tatariya, Vladimir Araujo, Thomas Bauwens, and Miryam De Lhoneux. 2024. Pixology: Probing the linguistic and visual capabilities of pixel-based language models. In Proceedings of EMNLP 2024, pages 3307–3320.

Zohre Soleymani Tekbudak, Mehdi Purmohammad, Ay¸segül Özkan, and Cengiz Acartürk. 2024. The PSR corpus: A Persian sentence reading corpus of eye movements. Behavior Research Methods, 57(1):14.

Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612.

Ethan G Wilcox, Tiago Pimentel, Clara Meister, Ryan Cotterell, and Roger P Levy. 2023. Testing the predictions of surprisal theory in 11 languages. Transactions ofthe Associationfor Computational Linguistics, 11:1451–1470.

Simon N. Wood. 2017. Generalized Additive Models: An Introduction with R, 2 edition. Chapman and Hall/CRC.

Simon N. Wood, Yannig Goude, and Simon Shaw. 2015. Generalized additive models for large data sets. Journal ofthe Royal Statistical Society Series C: Applied Statistics, 64(1):139–155.

Panpan Yao, Adrian Staub, and Xingshan Li. 2022. Predictability eliminates neighborhood effects during Chinese sentence reading. Psychonomic Bulletin & Review, 29(1):243–252.

<table><tr><td>Lang.</td><td>Unidirectional LMs</td><td>Model size (#parameters)</td><td>Bidirectional LMs</td><td>Model size (#parameters)</td></tr><tr><td>Deu</td><td>dbmdz/german-gpt2</td><td></td><td>510MB (～137M) google-bert/bert-base-german-cased</td><td>439MB (~109M)</td></tr><tr><td>Eng</td><td>openai-community/gpt2</td><td></td><td>548MB (~137M) google-bert/bert-base-cased</td><td>436MB (~108M)</td></tr><tr><td>Esp</td><td>DeepESP/gpt2-spanish</td><td></td><td>548MB (～124M) dccuchile/bert-base-spanish-wwm-cased</td><td>440MB (~110M)</td></tr><tr><td>Far</td><td>gpt2-persian</td><td></td><td>1.31GB (~328M) HooshvareLab/bert-base-parsbert-uncased 654MB (～163M)</td><td></td></tr><tr><td>Kor</td><td>skt/kogpt2-base-v2</td><td>513MB (~125M) kykim/bert-kor-base</td><td></td><td>476MB (~118M)</td></tr><tr><td></td><td></td><td></td><td>Simp (zh) uer/gpt2-chinese-cluecorpussmall 421MB (～102M) google-bert/bert-base-chinese</td><td>412MB (~102M)</td></tr><tr><td></td><td>Trad (zh) ckiplab/gpt2-base-chinese</td><td></td><td>421MB (～102M) ckiplab/bert-base-chinese</td><td>409MB (~102M)</td></tr><tr><td>Tur</td><td>ytu-ce-cosmos/turkish-gpt2</td><td></td><td>498MB (~124M) dbmdz/bert-base-turkish-cased</td><td>445MB (~111M)</td></tr><tr><td>All</td><td>mGPT</td><td></td><td>3.45GB (～1.4B) Pixel-m4 (including finetuned ones)</td><td>449MB (~112M)</td></tr></table>

Table 2: Models with their sizes in our experiment. All checkpoints of GPT and BERT are from https:// huggingface.co/.

![](images/01696e7f0ec7f023bdd8337a0bde3a91d617d1682abaf2bdf5036f5e8bea5b14.jpg)  
Figure 10: We approximate next-word predictability with patch-by-patch masking and add noise for the future context, simulating parafoveal sight, using PLM.

## A Models

In Table 2, we present the details of models used in this study.

## B PLM details

## B.1 Conditions

Figure 10 shows all four conditions explored in this study. The first option is whether to mask all the patches related to the target word (whole masking) or only the target patch (partial masking) when computing patch predictability. The second option is whether to apply the noisy blurring to the future context, potentially simulating the parafoveal information, or not.

## B.2 Gaussian Blurring

We apply the following noise function to the original image X:

$$
\pmb { X } ^ { \prime } ( i , j ) = \sum _ { k = - m } ^ { m } \sum _ { l = - m } ^ { m } \pmb { X } ( i - k , ~ j - l ) \cdot \pmb { G } ( k , l )\tag{18}
$$

$$
m = \left\lfloor { \frac { \mathsf { k e r n e l } _ { - } \mathsf { s i z e } - 1 } { 2 } } \right\rfloor\tag{19}
$$

$$
G ( k , l ) = \frac { 1 } { 2 \pi \sigma ^ { 2 } } e ^ { - \frac { k ^ { 2 } + l ^ { 2 } } { 2 \sigma ^ { 2 } } }\tag{20}
$$

(21)

As a cognitively implausible baseline, we also examine a no-blurring condition in which the rightside context is visible.

## B.3 Others

To adapt Persian to the bigram renderer, we leave the space between two characters, even if they are connected visually. However, the space does not change the pre-space word into an unconnected form if it is originally in a connected form. To gather VWPD from Persian, we did not include the white-area. For the results in the current paper, we used the checkpoint after the 33,500th step results of the PSR dataset.

## C Detailed Results

## C.1 Interactive Effect of PLM prediction difficulty to TLM surprisal

Figure 11 shows a more detailed comparison between the regression model with GPT surprisal and that with GPT surprisal and the interaction between GPT surprisal and PLM VWPD.

![](images/49ea29e935902c226cd2163a4c6d669afd69edd89ca12ce538b99c9a9adf5163.jpg)  
Figure 11: Gain of PPP after introducing PLM predictability to TLM $( \frac { \mathrm { P P P } _ { \mathrm { p i x e l \_ u n i * g p t } } - \mathrm { P P P } _ { \mathrm { g p t } } } { \mathrm { P P P } _ { \mathrm { g p t } } } )$ , which suggests multimodal prediction encodes rich metacognitive knowledge about skipping.

## C.2 Effects of Whole Word Masking

![](images/3c513faa9e36135c55ffb7b34a4d4e3322e4377527d2dcc784eaa949ca68ec28.jpg)  
Figure 12: Effect of Whole Masking on PPP (Paired-t test of PPP is implemented, with matched regression model formula)

Compared with blurring, the effect of masking is more nuanced across languages. We notice an advantage of whole-word masking in word skipping in Traditional Chinese, but a disadvantage in Korean. This potentially indicates a languagedependent sub-word processing strategy during naturalistic reading.

## C.3 Main Results by Dataset

Below are the detailed main results for each dataset:

![](images/794bcade5d1fe2ffcd84402da666c4ec2b37138f83a3257c369dff8af44a2300.jpg)  
Figure 13: Main results by dataset (Reading Time)

![](images/47af773d5611e4af60197668deb4ee4772fe1b7934ab0bd7f6d436794a52438b.jpg)  
Figure 14: Main results by dataset (Skipping)

## D Cross-validation

In Figure 15, we reran the inter-measure comparison results in Figure 4 through 10-fold crossvalidation. In Figure 16, Figure 17 and Figure 18, we reproduce the results in Figure 5, Figures 6 and 7, and Table 4.

The cross-validation results do not contradict the observation that the MECO Chinese data show a stronger advantage of MSE-based VWPD. A revisit of the 10-fold cross-validation result supports the identical tendencies we have captured in Section 5. In the first place, an integration, especially an interaction, between token and visual prediction improves reading-time and skipping modeling. Besides, visual prediction accounts for skipping better than token prediction in certain scripts, notably Persian (Far), Korean, and Chinese scripts.

## E Comparing MSE and SSIM across datasets

We report the performance of MSE and SSIM in terms of their advantage against the GPT2 model in Table 3.

![](images/49bf0d942158dbb03d91d53fb8b6657a64c98db7b659460e2f352643e7768cbf.jpg)  
Figure 15: Comparison of MSE and SSIM measures in 10-fold cross-validation settings.

![](images/2f9ec4820af9a27604932136761eae8394b28babf9f24d66f83c25575637fc30.jpg)  
Figure 16: Main results by dataset (Reading Time) using 10-fold cross-validation based on visual and token surprisal. (Log-likelihood change is case-wise averaged)

![](images/d2f912c94000248a474f9c5d539ec24bb82fd870ecebb12b9f21b4a3cdc7fb04.jpg)  
Figure 17: Main results by dataset (Skipping) using 10-fold cross-validation based on visual and token surprisal. (Log-likelihood change is case-wise averaged)

![](images/1834177d0a823b110f1a90abc1c5bcdbfbbf16e6bba92f76c354459c1b010d90.jpg)  
Figure 18: Main results by dataset (Skipping) using 10-fold cross-validation, based on visual and token entropy. (Log-likelihood change is case-wise averaged)

<table><tr><td>VWPDMeasure Esp (es) Esp (ch) Deu (po) Deu (zu) Eng (uk) Eng (en)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Kor</td><td>Far (PSR)</td><td>Simp</td><td>Trad</td><td>Tur</td></tr><tr><td colspan="10">pixel_uni vs. gpt</td></tr><tr><td>SSIM FFD</td><td></td><td>-23.0*** -28.3*** -32.7***</td><td></td><td></td><td>-48.4***</td><td>-68.9***</td><td>-30.6***</td><td>-5.6*</td><td></td><td>-126.6***-27.5***</td><td> $- 2 4 . 3 ^ { \ast \ast \ast }$ </td><td> $- 4 . 1 ^ { * }$ </td></tr><tr><td>FPT</td><td></td><td> $- 3 7 . 3 ^ { \ast \ast \ast }$ </td><td>-74.3***-60.1***</td><td></td><td>-84.1***</td><td>-100.4***</td><td>* -49.9***</td><td>-44.8***</td><td></td><td>-321.2*** -63.8***</td><td> $- 2 4 . 0 ^ { * * * }$ </td><td> $- 1 3 . 2 ^ { * * * }$ </td></tr><tr><td>TRT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $- 2 0 . 8 ^ { * * * } - 6 6 . 7 ^ { * * * } - 2 1 4 . 0 ^ { * * * } - 2 3 5 . 6 ^ { * * * } - 1 9 4 . 3 ^ { * * * } - 9 4 . 6 ^ { * * * } - 6 8 . 1 ^ { * * * } - 1 2 9 . 1 ^ { * * * } - 9 0 . 6 ^ { * * * } - 3 0 . 9 ^ { * * * }$ </td><td></td><td>-15.5***</td></tr><tr><td></td><td>FSKP</td><td> $+ 2 6 . 4 ^ { * * * }$ </td><td> ${ _ { - 6 . 9 } } ^ { * * }$ </td><td> $+ 1 6 . 4 ^ { * * * }$ </td><td> $+ 2 8 . 9 ^ { * * * }$ </td><td>+41.8***</td><td>-6.1*</td><td></td><td>+49.2*** +177.0***</td><td> $- 2 3 . 1 ^ { * * * }$ </td><td> $- 1 6 . 1 ^ { * * * }$ </td><td> $- 6 . 3 ^ { \ast \ast \ast }$ </td></tr><tr><td></td><td>SKP</td><td>-4.3</td><td> $- 2 2 . 1 ^ { * * * }$ </td><td> $- 8 . 7 ^ { * * * }$ </td><td> $- 5 5 . 4 ^ { * * * }$ </td><td>-6.2</td><td></td><td></td><td> $- 2 8 . 0 ^ { * * * } + 1 0 7 . 6 ^ { * * * } + 2 1 4 . 9 ^ { * * * }$ </td><td>-45.6***</td><td> $- 2 1 . 1 ^ { * * * }$ </td><td> $+ 7 . 8 ^ { * * * }$ </td></tr><tr><td>MSE</td><td>FFD</td><td> $- 2 0 . 3 ^ { * * * }$ </td><td> $- 2 1 . 0 ^ { * * * }$ </td><td> $- 2 5 . 9 ^ { * * * }$ </td><td> $- 3 0 . 1 ^ { * * * }$ </td><td></td><td>-75.6*** -34.3***</td><td> $- 3 1 . 4 ^ { * * * }$ </td><td> $- 1 4 0 . 0 ^ { * * * }$ </td><td> $- 1 3 . 9 ^ { * * * }$ </td><td>-0.9</td><td> $- 1 1 . 2 ^ { * * * }$ </td></tr><tr><td></td><td>FPT</td><td> $- 3 3 . 7 ^ { * * * }$ </td><td> $- 6 3 . 9 ^ { * * * }$ </td><td> $- 7 3 . 6 ^ { * * * }$ </td><td> $- 9 4 . 2 ^ { * * * }$ </td><td></td><td>-126.2***-72.4***</td><td> $- 7 3 . 0 ^ { * * * }$ </td><td> $- 4 0 8 . 7 ^ { * * * }$ </td><td> ${ } _ { - 4 9 . 9 ^ { * * * } }$ </td><td>-1.5</td><td> $- 1 2 . 4 ^ { * * * }$ </td></tr><tr><td>TRT</td><td></td><td> $- 2 3 . 8 ^ { * * * }$ </td><td> $- 7 0 . 3 ^ { * * * }$ </td><td></td><td></td><td></td><td>-250.2*** -257.0*** -268.6*** -128.4*** -123.7***</td><td></td><td> $- 2 4 9 . 6 ^ { * * * }$ </td><td> $- 5 2 . 9 ^ { * * * }$ </td><td></td><td> $+ 3 9 . 2 ^ { * * * } - 2 2 . 6 ^ { * * * }$ </td></tr><tr><td>FSKP</td><td></td><td> $+ 2 2 . 6 ^ { * * * }$ </td><td> ${ } _ { - 4 . 6 } \cdot$ </td><td> $+ 1 0 . 3 ^ { * * }$ </td><td> $+ 7 . 3 ^ { * }$ </td><td>+16.1***</td><td>-0.3</td><td></td><td>-14.2*** -109.4***</td><td> $+ 1 4 . 2 ^ { * * * }$ </td><td></td><td> $+ 3 7 . 9 ^ { * * * } - 1 4 . 2 ^ { * * * }$ </td></tr><tr><td>SKP</td><td></td><td> $- 9 . 1 ^ { \ast \ast \ast }$ </td><td> $- 2 1 . 7 ^ { * * * }$ </td><td> $- 1 4 . 5 ^ { * * * }$ </td><td> $- 6 3 . 9 ^ { * * * }$ </td><td> $- 6 8 . 1 ^ { * * * }$ </td><td> $- 3 5 . 8 ^ { * * * }$ </td><td> $- 4 0 . 5 ^ { * * * }$ </td><td> $- 7 0 . 7 ^ { * * * }$ </td><td> $+ 3 0 . 0 ^ { * * * }$ </td><td></td><td> $+ 5 3 . 0 ^ { * * * } ~ { - 8 . 1 } ^ { * * * }$ </td></tr><tr><td colspan="10">pixel_uni × gpt vs. gpt</td><td></td><td></td><td></td></tr><tr><td>SSIM FFD</td><td></td><td>+26.6***</td><td>+13.7***+31.7*** +37.0***</td><td></td><td></td><td>+18.1***</td><td>+12.1***</td><td></td><td></td><td>+21.7***+103.8***+11.1***</td><td> $+ 4 . 0 ^ { * * * }$ </td><td> $- 6 . 0 ^ { * }$ </td></tr><tr><td></td><td>FPT</td><td>+20.2***</td><td>+28.7*** +55.9***</td><td></td><td>+78.9***</td><td>+76.5***</td><td></td><td></td><td></td><td>+53.0*** +36.2*** +181.7*** +37.0*** +34.7*** -17.1***</td><td></td><td></td></tr><tr><td></td><td>TRT</td><td></td><td>+43.1*** +26.6*** +74.4*** +65.9*** +100.3*** +54.2*** +59.8*** +228.4*** +19.3***</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $+ 7 . 3 ^ { * * * }$ </td><td> $- 4 . 4 ^ { * }$ </td></tr><tr><td></td><td>FSKP</td><td></td><td> $+ 1 3 4 . 7 ^ { * * * } + 4 9 . 0 ^ { * * * } + 7 6 . 1 ^ { * * * } + 1 4 3 . 9 ^ { * * * } + 6 5 . 1 ^ { * * * } + 5 6 . 2 ^ { * * * } + 1 1 4 . 5 ^ { * * * } + 3 2 8 . 7 ^ { * * * } + 1 6 . 2 ^ { * * * }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $+ 2 3 . 7 ^ { * * * } + 1 6 . 1 ^ { * * * }$ </td></tr><tr><td></td><td>SKP</td><td> $+ 4 2 . 5 ^ { * * * }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>+16.9*** +43.0*** +19.1*** +85.4*** +42.0*** +167.4*** +337.6*** +19.2*** +23.7*** +15.7***</td></tr><tr><td>MSE</td><td>FFD</td><td></td><td> $+ 3 0 . 6 ^ { * * * } + 2 1 . 8 ^ { * * * } + 3 4 . 8 ^ { * * * } + 4 5 . 0 ^ { * * * }$ </td><td></td><td></td><td>+16.1*** +8.7***</td><td></td><td>-0.7</td><td> $+ 5 6 . 2 ^ { * * * }$ </td><td> $+ 3 3 . 4 ^ { * * * }$ </td><td></td><td> $+ 2 8 . 9 ^ { * * * } - 1 1 . 1 ^ { * * * }$ </td></tr><tr><td></td><td>FPT</td><td></td><td> $+ 2 3 . 1 ^ { * * * } + 3 7 . 1 ^ { * * * } + 5 7 . 2 ^ { * * * }$ </td><td></td><td>+84.9***</td><td>+64.0***</td><td>+40.0***+18.4***</td><td></td><td> $+ 5 7 . 7 ^ { * * * }$ </td><td> $+ 6 6 . 3 ^ { * * * }$ </td><td></td><td>+59.9*** -14.8***</td></tr><tr><td></td><td>TRT</td><td></td><td> $+ 4 1 . 2 ^ { * * * } + 2 1 . 4 ^ { * * * } + 6 4 . 6 ^ { * * * } + 7 3 . 6 ^ { * * * }$ </td><td></td><td></td><td>+51.8***</td><td>+31.1***</td><td>+23.3***</td><td> $+ 7 9 . 2 ^ { * * * }$ </td><td></td><td></td><td>+78.4***+82.7***-10.8***</td></tr><tr><td></td><td>FSKP</td><td></td><td> $+ 1 3 2 . 7 ^ { * * * } + 5 4 . 0 ^ { * * * } + 6 6 . 1 ^ { * * * } + 1 1 2 . 0 ^ { * * * }$ </td><td></td><td></td><td>+36.8***</td><td>+56.1***</td><td>+44.2***</td><td> $+ 5 6 . 2 ^ { * * * }$ </td><td></td><td> $+ 6 1 . 0 ^ { * * * } + 7 2 . 2 ^ { * * * } + 5 . 6 ^ { * }$ </td><td></td></tr><tr><td></td><td>SKP</td><td></td><td> $+ 3 7 . 1 ^ { * * * } + 1 9 . 5 ^ { * * * } + 4 1 . 9 ^ { * * * } + 2 7 . 5 ^ { * * * }$ </td><td></td><td></td><td> $+ 3 5 . 4 ^ { * * * }$ </td><td>+46.6***</td><td> $+ 3 1 . 1 ^ { * * * }$ </td><td> $+ 6 3 . 0 ^ { * * * }$ </td><td>+119.1*** +97.4*** +2.9</td><td></td><td></td></tr></table>

Table 3: Mean paired log-likelihood difference relative to GPT. Positive values favour the reported model. The estimation is based on boostrapping. The significance is based on paired t-test. $^ { * } p < . 0 5 , ^ { * * } p < . 0 1 , ^ { * * * } p < . 0 0 1$ $\cdot \ p < . 1$

<table><tr><td>VWPDMeasure</td><td></td><td>Esp (es)</td><td>Esp (ch)</td><td>Deu (po)</td><td>Deu (zu)</td><td>Eng (uk)</td><td>Eng (en)</td></tr><tr><td colspan="10">pixel_uni vs. gpt</td></tr><tr><td>SSIM FFD</td><td></td><td>-0.000033</td><td>-0.000153</td><td>+0.000029</td><td>+0.000045</td><td>-0.000421</td><td>-0.000188</td></tr><tr><td></td><td>FPT</td><td>-0.000374</td><td>-0.001012</td><td>-0.000885</td><td>-0.000158</td><td>-0.001189</td><td> $- 0 . 0 0 0 7 6 8 ^ { \ast \ast }$ </td></tr><tr><td></td><td>TRT</td><td>-0.000125</td><td>-0.000871 ***</td><td>-0.004425 ***</td><td>-0.002344***</td><td>-0.002263* ***</td><td> $- 0 . 0 0 1 3 2 3 ^ { \ast \ast \ast }$ </td></tr><tr><td></td><td>FSKP</td><td>+0.000098</td><td>+0.000007</td><td>-0.000392* ***</td><td>-0.000599 ***</td><td> $+ 0 . 0 0 0 3 2 3 ^ { \ast }$ </td><td> $+ 0 . 0 0 0 0 0 3$ </td></tr><tr><td></td><td>SKP</td><td>-0.000009</td><td>-0.000087</td><td>-0.000167</td><td> $- 0 . 0 0 0 5 6 2 ^ { \ast \ast \ast }$ </td><td>-0.000131</td><td> ${ - 0 . 0 0 0 3 1 3 ^ { \ast \ast } }$ </td></tr><tr><td>MSE</td><td>FFD</td><td>-0.000036</td><td>-0.000146</td><td>+0.000089</td><td>+0.000148</td><td>-0.000500*</td><td> $- 0 . 0 0 0 2 1 5 ^ { \ast }$ </td></tr><tr><td></td><td>FPT</td><td>-0.000376</td><td>-0.001025 **</td><td>-0.001090*</td><td>-0.001178*</td><td> $- 0 . 0 0 1 9 8 4 ^ { * * }$ </td><td> $- 0 . 0 0 1 0 0 1 ^ { \ast \ast }$ </td></tr><tr><td></td><td>TRT</td><td>-0.000134</td><td>-0.000860*</td><td>-0.005057* ***</td><td> $- 0 . 0 0 2 8 8 2 ^ { * * * }$ </td><td> $- 0 . 0 0 3 2 3 8 ^ { \ast \ast \ast }$ </td><td> $- 0 . 0 0 1 7 5 9 ^ { \ast \ast \ast }$ </td></tr><tr><td></td><td>FSKP</td><td>+0.000123</td><td>+0.000002</td><td> ${ - 0 . 0 0 0 3 4 2 ^ { \ast \ast } }$ </td><td> $- 0 . 0 0 1 4 4 4 ^ { \ast \ast \ast }$ </td><td> $+ 0 . 0 0 0 2 3 2 ^ { \cdot }$ </td><td> $+ 0 . 0 0 0 0 2 7$ </td></tr><tr><td></td><td>SKP</td><td>-0.000004</td><td>-0.000005</td><td>-0.000246*</td><td> $- 0 . 0 0 0 7 5 6 ^ { \ast \ast }$ </td><td> $- 0 . 0 0 0 8 5 7 ^ { \ast \ast \ast }$ </td><td> $- 0 . 0 0 0 3 8 2 ^ { \ast \ast }$ </td></tr></table>

<table><tr><td colspan="8">pixel_uni × gpt vs. gpt</td></tr><tr><td>SSIM</td><td>FFD</td><td>-0.000008</td><td>+0.000003</td><td>+0.000429*</td><td>+0.000408</td><td>-0.000064</td><td>-0.000207</td></tr><tr><td></td><td>FPT</td><td>+0.000040</td><td>+0.000230*</td><td>+0.000511</td><td>+0.001832***</td><td>+0.001246 ***</td><td> $+ 0 . 0 0 0 3 6 5 ^ { * * * }$ </td></tr><tr><td></td><td>TRT</td><td>-0.000155</td><td> $+ 0 . 0 0 0 2 3 2 ^ { \ast \ast }$ </td><td>+0.000863*</td><td> $+ 0 . 0 0 0 7 4 3 ^ { \ast }$ </td><td> $+ 0 . 0 0 0 5 0 5$ </td><td> $+ 0 . 0 0 0 3 5 2 ^ { \ast }$ </td></tr><tr><td></td><td>FSKP</td><td> $+ 0 . 0 0 1 0 9 4 ^ { * * * }$ </td><td> $+ 0 . 0 0 0 1 8 4$ </td><td>+0.000259*</td><td> $+ 0 . 0 0 1 0 3 5 ^ { * * * }$ </td><td> $+ 0 . 0 0 0 1 3 1$ </td><td> $+ 0 . 0 0 0 3 1 8 ^ { \ast \ast }$ </td></tr><tr><td></td><td>SKP</td><td> $+ 0 . 0 0 0 1 8 6 ^ { * * }$ </td><td> $- 0 . 0 0 0 1 6 4 ^ { * }$ </td><td> $- 0 . 0 0 0 4 5 4 ^ { * }$ </td><td> $- 0 . 0 0 0 3 3 8 ^ { \ast \ast \ast }$ </td><td> $+ 0 . 0 0 0 5 7 8 ^ { \ast \ast }$ </td><td> $+ 0 . 0 0 0 1 7 5 ^ { \ast }$ </td></tr><tr><td>MSE</td><td>FFD</td><td>-0.000006</td><td> $+ 0 . 0 0 0 0 2 7$ </td><td> $+ 0 . 0 0 0 3 9 7 ^ { \ast \ast }$ </td><td> $+ 0 . 0 0 0 2 8 3$ </td><td> $- 0 . 0 0 0 0 7 2$ </td><td> $- 0 . 0 0 0 2 1 9 \cdot$ </td></tr><tr><td></td><td>FPT</td><td> $- 0 . 0 0 0 0 0 5$ </td><td> $+ 0 . 0 0 0 1 0 8$ </td><td> $+ 0 . 0 0 0 6 7 7$ </td><td> $+ 0 . 0 0 1 2 9 1 ^ { \ast \ast }$ </td><td> $+ 0 . 0 0 0 8 3 9 ^ { \ast \ast \ast }$ </td><td> $+ 0 . 0 0 0 2 2 2 ^ { \ast \ast }$ </td></tr><tr><td></td><td>TRT</td><td>-0.000179</td><td> $+ 0 . 0 0 0 1 7 1 \cdot$ </td><td> $+ 0 . 0 0 0 8 6 1 ^ { \ast }$ </td><td> $+ 0 . 0 0 0 6 4 2 ^ { \ast }$ </td><td> $+ 0 . 0 0 0 0 2 1$ </td><td> $+ 0 . 0 0 0 0 4 2$ </td></tr><tr><td></td><td>FSKP</td><td> $+ 0 . 0 0 1 1 8 0 ^ { * * * }$ </td><td>+0.000225</td><td>+0.000162</td><td> $+ 0 . 0 0 0 2 1 0$ </td><td>+0.000030</td><td> $+ 0 . 0 0 0 3 3 1 ^ { \ast \ast }$ </td></tr><tr><td></td><td>SKP</td><td> $+ 0 . 0 0 0 1 6 9 ^ { \ast \ast }$ </td><td>-0.000063</td><td>-0.000309</td><td> $- 0 . 0 0 0 2 9 9 ^ { \ast \ast }$ </td><td>+0.000060</td><td> $+ 0 . 0 0 0 2 5 8 ^ { \ast \ast }$ </td></tr></table>

<table><tr><td colspan="2">VWPDMeasure</td><td>Kor</td><td>Far (PSR)</td><td>Simp</td><td>Trad</td><td>Tur</td></tr><tr><td colspan="9">pixel_uni vs. gpt</td></tr><tr><td>SSIM</td><td>FFD</td><td>+0.000710</td><td>+0.000173</td><td>-0.000287</td><td>+0.000074</td><td></td><td>+0.000127</td></tr><tr><td></td><td>FPT</td><td></td><td>-0.001143</td><td>-0.000866</td><td>-0.000944**</td><td>+0.000049</td><td>+0.000777</td></tr><tr><td></td><td>TRT</td><td></td><td>-0.002521 **</td><td>+0.003256 ***</td><td>-0.000795</td><td>+0.000122</td><td>-0.000284</td></tr><tr><td></td><td>FSKP</td><td>+0.001036</td><td>• +0.012088</td><td>**</td><td>-0.000023</td><td>-0.000052</td><td> $+ 0 . 0 0 0 8 5 4 ^ { \ast }$ </td></tr><tr><td></td><td>SKP</td><td></td><td> $+ 0 . 0 0 3 7 9 6 ^ { * * * } + 0 . 0 1 3 2 0 1 ^ { * * * }$ </td><td></td><td>-0.000286*</td><td>-0.000024</td><td> $+ 0 . 0 0 1 4 5 3 ^ { \ast }$ </td></tr><tr><td>MSE</td><td>FFD</td><td>-0.000061</td><td>+0.000217</td><td>+0.000329</td><td></td><td> $+ 0 . 0 0 0 9 7 8 ^ { \ast }$ </td><td>+0.000007</td></tr><tr><td></td><td>FPT</td><td>-0.003004</td><td>**  $- 0 . 0 0 3 0 1 3 ^ { \ast \ast * }$ </td><td></td><td>-0.000419</td><td> $+ 0 . 0 0 0 9 1 7 \cdot$ </td><td>-0.000087</td></tr><tr><td></td><td>TRT</td><td> $- 0 . 0 0 5 0 1 0 ^ { * * * }$ </td><td></td><td>+0.000656*</td><td>+0.000167</td><td> $+ 0 . 0 0 1 5 7 9 ^ { \ast \ast }$ </td><td> $- 0 . 0 0 0 6 4 5 ^ { \ast }$ </td></tr><tr><td></td><td>FSKP</td><td>+0.000079</td><td></td><td> $+ 0 . 0 0 2 9 4 4 ^ { * * * }$ </td><td>+0.000431*</td><td> $+ 0 . 0 0 0 8 7 7 ^ { \ast \ast \ast }$ </td><td> $+ 0 . 0 0 0 4 6 4 ^ { \ast }$ </td></tr><tr><td></td><td>SKP</td><td>+0.000351</td><td>+0.003176*</td><td>** +0.001179</td><td>***</td><td> $+ 0 . 0 0 1 2 7 4 ^ { * * }$ </td><td> $+ 0 . 0 0 0 6 4 3 \cdot$ </td></tr><tr><td colspan="8">pixel_uni × gpt vs. gpt</td></tr><tr><td>SSIM</td><td>FFD</td><td>+0.000555</td><td>+0.001022</td><td>*** -0.000181 ***</td><td></td><td>-0.000408</td><td>+0.000556</td></tr><tr><td></td><td>FPT</td><td></td><td>+0.001384 +0.003938</td><td>+0.000578</td><td></td><td>+0.000228</td><td> $+ 0 . 0 0 0 7 7 7 ^ { \ast }$ </td></tr><tr><td></td><td>TRT</td><td>+0.001847**</td><td>+0.004196</td><td>**</td><td>+0.000071</td><td>+0.000149</td><td>+0.000076</td></tr><tr><td></td><td>FSKP</td><td>+0.001260*</td><td>+0.009560</td><td>***</td><td>-0.000053</td><td>+0.000180</td><td> $+ 0 . 0 0 3 4 2 2 ^ { \ast \ast }$ </td></tr><tr><td></td><td>SKP</td><td></td><td> $+ 0 . 0 0 3 7 5 4 ^ { * * * } + 0 . 0 1 0 9 0 7 ^ { * * * }$ </td><td></td><td>-0.000003</td><td>+0.000073</td><td> $+ 0 . 0 0 1 8 7 2 ^ { \ast }$ </td></tr><tr><td>MSE</td><td>FFD</td><td>-0.000212</td><td>+0.000435*</td><td></td><td>+0.000687*</td><td>+0.000520</td><td>+0.000395</td></tr><tr><td></td><td>FPT</td><td>+0.000341</td><td>+0.000923</td><td>**  $+ 0 . 0 0 1 4 2 4 ^ { * * }$ </td><td></td><td>+0.000952</td><td>-0.000039</td></tr><tr><td></td><td>TRT</td><td>+0.000694</td><td>** +0.000879</td><td>+0.001578*</td><td>**</td><td>+0.001596</td><td>-0.000039</td></tr><tr><td></td><td>FSKP</td><td>+0.000048</td><td>+0.001996</td><td>**</td><td>+0.000391 +0.000988</td><td>**</td><td>+0.003323*</td></tr><tr><td></td><td>SKP</td><td></td><td> $+ 0 . 0 0 0 7 0 2 ^ { * * * } + 0 . 0 0 2 0 4 7 ^ { * * * } + 0 . 0 0 1 8 2 2 ^ { * * * } + 0 . 0 0 1 3 5 5 ^ { * * * } + 0 . 0 0 1 3 6 7 ^ { * * * }$ </td><td></td><td></td><td></td><td></td></tr></table>

Table 4: Mean paired log-likelihood difference relative to GPT. Positive values favour the reported model. $^ { * } p < . 0 5 ,$ ${ } ^ { * * } p < . 0 1 , { } ^ { * * * } p < . 0 0 1 , \cdot { } p < . 1$ . The estimations are based on cross-validation. The significance is based on paired t-test.