# A Novel Sentence Stress Detection Framework Leveraging Auxiliary Word-Stress Modeling and Loss Optimization

Tien-Hong Lo <sup>ID</sup> , Fong-Chun Tsai <sup>ID</sup> , Ting-An Hung, Yu-Hsuan Hsieh, Yao-Ting Sung <sup>ID</sup> , Berlin Chen <sup>ID</sup>

National Taiwan Normal University, Taiwan {teinhonglo,berlin}@ntnu.edu.tw

## Abstract

Prosodic stress is a crucial aspect of automatic pronunciation assessment (APA), encompassing both sentence stress detection (SSD) and word stress detection (WSD). SSD highlights semantically salient words that shape discourse meaning, while WSD identifies the primary stressed syllable within each word to ensure lexical clarity. However, most prior work treats SSD and WSD as independent tasks, overlooking their shared reliance on prosodic cues such as pitch, duration, and intensity. To address this gap, we propose an effective SSD approach combining SSD with auxiliary WSD via a novel modeling paradigm. In addition, we introduce a word-span stress regularizer (WSR) that concentrates token-level SSD probabilities within each stressed word span. Experiments on the TinyStress-15K benchmark show that the proposed method outperforms strong baselines, with the complete configuration achieving the best SSD result. Index Terms: Automatic pronunciation assessment, sentence stress detection, word stress detection.

## 1. Introduction

With the rapid advances in computing technology and the growing global population of second-language (L2) learners, automatic pronunciation assessment (APA) has attracted considerable attention and plays an increasingly prominent role in computer-assisted language learning (CALL) [1]. APA systems are designed to provide timely and objective feedback on learners’ speaking performance, thereby facilitating self-directed improvement and reducing the instructional burden on teachers. Research on APA focuses specifically on evaluating multiple aspects of L2 learners’ speech, such as fluency and prosody [2].

Prosodic stress constitutes a critical component of APA, encompassing both sentence stress detection (SSD) and word stress detection (WSD). As displayed in Fig. 1, sentence stress detection (SSD) identifies semantically salient words that shape discourse meaning, while word stress detection (WSD) locates the primary stressed syllable within each word to ensure lexical clarity. As for SSD, theoretical studies have framed stress as either a phonological default or a semantic device for emphasis [3, 4, 5], while conceptual accounts have further characterized its acoustic correlates such as duration, intensity, and pitch [6]. Building on these acoustic and linguistic foundations, early SSD methodologies primarily employed shallow classifiers alongside handcrafted features [7, 8], later advancing to neural architectures [9, 10]. Additionally, some research has demonstrated that exploiting syntactic boundaries and intonation units can further improve prediction [11, 12, 13]. More recently, large-scale end-to-end frameworks with stress detection objectives, trained on the synthetic dataset, have achieved competitive benchmark performance [14, 15]. On the separate (a) Sentence Stress Detection She bought a new car. Emphasizes ‘she’ She bought a new car. Emphasizes ‘bought’ She bought a new car. Emphasizes ‘a’ She bought a new car. Emphasizes ‘new’ She bought a new car. Emphasizes ‘car’

(b) Word Stress Detection Record /R EH1 K ER0 D/: (noun) Record /R IH0 K AO1 R D/: (verb)

Figure 1: Illustration of stress detection at two linguistic levels. (a) SSD highlights the emphasized word within a sentence by shifting prominence across different lexical items. (b) WSD identifies the primary stressed syllable within a lexical item, where primary stress placement is indicated by the digit “1”.

front, WSD research has long relied on acoustic cues such as intensity, duration, and pitch [16, 17, 18], later augmented with contextual sonority-based features [19] and nucleus-level clustering [20]. Early classifiers included decision trees and SVMs [21], while deep models such as CNNs and transformers have since advanced word stress classification [22, 23, 24]. Recent work has instead integrated adaptive loss functions that embed linguistic constraints directly into training, reducing the need for post-hoc correction [25]. More broadly, structured training objectives for prosody often take the form of soft constraints or regularizers, such as entropy penalties or sparsity-inducing terms, to encourage well-formed stress patterns. Despite the continued efforts, SSD and WSD continue to be modeled independently, leaving unexplored their shared reliance on prosodic cues and the potential benefits of a unified framework.

To tackle the aforementioned challenge, we propose STRAW, a Sentence sTress detection (SSD) framework built on a frozen Whisper backbone that augments token-level SSD with a word-span stress Regularizer (WSR) and an Auxiliary phone-level Word stress detection (WSD) branch. Unlike previous approaches that treat SSD and WSD in isolation, the proposed framework supports both tasks within a common frozen-Whisper framework. Experiments on the TinyStress [14] benchmark dataset demonstrate improvements over strong baselines. The main contributions of this work are summarized as follows:

1. We present a unified framework for stress detection with separate task-specific branches for Sentence Stress Detection (SSD) and Word Stress Detection (WSD).

![](images/ab14b47b12a72d901689a325d065ddde1a8c86a9c5c3a7f10afc67c220052302.jpg)  
Figure 2: Overall architecture of the proposed framework. A frozen Whisper backbone provides encoder representations to both heads and decoder representations to the SSD head. Sentence stress detection (SSD) operates at the token level with an additional word-span stress regularizer (WSR), while word stress detection (WSD) operates at the phone level. The SSD and WSD heads are task-specific and do not directly exchange hidden states or predictions.

2. We introduce a word-span stress regularizer (WSR), a linguistically motivated constraint that discourages diffuse SSD probabilities across the subword tokens of a stressed word.

## 2. Method

In this section, prosodic stress detection is formulated as a supervised classification task over spoken utterances. As illustrated in Figure 2, each utterance is represented as a log-mel spectrogram s<sub>1:T</sub> with T acoustic frames.

The corresponding transcription is expressed as a word sequence $\mathbf { w } _ { 1 : L } ,$ a token sequence $\mathbf { t } _ { 1 : M } .$ , and a phone sequence $\mathbf { p } _ { 1 : N }$ using an off-the-shelf grapheme-to-phoneme (G2P) converter with CMU-style stress-marked phones. Subsequently, the Whisper backbone is employed to encode s through its encoder and decoder blocks, producing contextualized encoder states $\mathbf { e } _ { 1 : T }$ and decoder states $\mathbf { d } _ { 1 : M }$ . Two task-specific modules, sentence stress detection (SSD) and word stress detection (WSD), are introduced on top of the frozen Whisper backbone. Specifically, SSD is described in Section 2.1, WSD is described in Section 2.2, and the word-span stress regularizer (WSR) is described in Section 2.3.

In our experiments, the Whisper backbone is frozen and we train only the task-specific heads and the phone embedding layer. SSD is trained at the token level, where each token receives a binary stress label $y _ { 1 : M } ^ { s s d } \in \{ 0 , 1 \}$ , while evaluation aggregates predictions $\hat { y } _ { 1 : M } ^ { s s d }$ to the word level by assigning a word as stressed if any of its tokens is predicted as stressed. WSD, in contrast, operates at the phone level, where each phone is labeled $y _ { 1 : N } ^ { w s d } \in \{ 0 , 1 \}$ as stressed or unstressed.

## 2.1. Sentence Stress Detection

As shown in the right branch of Figure 2, the SSD module operates at the token level. For each utterance, the Whisper decoder-state sequence $\mathbf { d } _ { 1 : M }$ serves as the queries, while the encoder-state sequence $\mathbf { e } _ { 1 : T }$ serves as the keys and values. A Transformer decoder block integrates the two sources of information, followed by a fully connected neural network (FCNN)

to predict token-level stress:

$$
{ \bf d } _ { 1 : M } ^ { s s d } = \mathrm { T r a n s f o r m e r } ^ { s s d } ( { \bf d } _ { 1 : M } , { \bf e } _ { 1 : T } ) ,\tag{1}
$$

$$
\hat { \mathbf { y } } _ { 1 : M } ^ { s s d } = \mathrm { S o f t m a x } \left( \mathrm { F C N N } ^ { s s d } ( \mathbf { d } _ { 1 : M } ^ { s s d } ) \right) ,\tag{2}
$$

where $\hat { \bf y } _ { m } ^ { s s d } \mathrm { ~ \bf ~ \in ~ { ~ \bf ~ \mathbb { R } ^ { 2 } } ~ }$ is a two-class posterior over unstressed vs. stressed for the m-th token, and we denote $\hat { y } _ { m } ^ { s s d }$ as the stressed-class probability. The loss function of $\mathcal { L } _ { S S D }$ is computed using cross-entropy over token-level SSD labels.

## 2.2. Word Stress Detection

The left branch of Figure 2 shows the phone-level WSD module. Given an utterance-level phone sequence $p _ { 1 : N } .$ , we obtain trainable phone embeddings $\mathbf { p } _ { 1 : N }$ through an embedding layer. We then compute phone-aware acoustic representations via cross-attention, where $\mathbf { p } _ { 1 : N }$ serves as queries and the Whisper encoder-state sequence $\mathbf { e } _ { 1 : T }$ serves as the keys and values. The resulting phone representations are then passed to an FCNN to yield phone-level stress predictions:

$$
{ \bf d } _ { 1 : N } ^ { w s d } = \mathrm { T r a n s f o r m e r } ^ { w s d } ( { \bf p } _ { 1 : N } , { \bf e } _ { 1 : T } ) ,\tag{3}
$$

$$
\hat { \mathbf { y } } _ { 1 : N } ^ { w s d } = \mathrm { S o f t m a x } \left( \mathrm { F C N N } ^ { w s d } ( \mathbf { d } _ { 1 : N } ^ { w s d } ) \right) ,\tag{4}
$$

where $\hat { \textbf { y } } _ { n } ^ { w s d } \quad \in \quad \mathbb { R } ^ { 2 }$ is a two-class posterior over unstressed vs. stressed for the n-th phone, and we denote $\hat { y } _ { n } ^ { w s d }$ as the stressed-class probability. The WSD loss ${ \mathcal { L } } _ { W S D }$ is optimized using the cross-entropy objective over phone-level WSD labels.

## 2.3. Word-Span Stress Regularizer

Since the SSD branch is formulated at the token level, a lexical word may correspond to a span of subword tokens under Whisper tokenization. When a word is annotated as stressed in SSD, the token-level SSD head may assign elevated stress probabilities to multiple tokens within its subword span, yielding an underconstrained within-word allocation of prominence. We therefore encourage a single dominant token-level stress position within each ground-truth stressed word span. Accordingly, we introduce a word-span stress regularizer over tokenlevel SSD outputs to encourage a unique dominant stressed position for each ground-truth stressed word $w _ { m }$ with token span $[ b _ { m } , e _ { m } ] \colon$

$$
i ^ { \star } = \operatorname * { a r g m a x } _ { i \in \{ b _ { m } , \ldots , e _ { m } \} } \hat { y } _ { i } ^ { s s d } ,\tag{5}
$$

$$
\omega ( w _ { m } ) = \Big | 1 - \sum _ { i = b _ { m } } ^ { e _ { m } } \hat { y } _ { i } ^ { s s d } \Big | ,\tag{6}
$$

where $\omega ( w _ { m } )$ measures the deviation of the total SSD stress probability within $w _ { m }$ from one and weights the subsequent peak-sharpening term. The word-span stress regularizer $\mathcal { L } _ { W S R }$ for $w _ { m }$ is formulated as:

$$
\begin{array} { r l } { \displaystyle \mathcal { L } _ { W S R } ( w _ { m } ) = \omega ( w _ { m } ) \cdot \Big ( - \log \hat { y } _ { i } ^ { s s d } } & { } \\ { \displaystyle - \sum _ { i \in \{ b _ { m } , \dots , e _ { m } \} \setminus \{ i ^ { \star } \} } \log \big ( 1 - \hat { y } _ { i } ^ { s s d } \big ) \Big ) , } & { } \end{array}\tag{7}
$$

The final $\mathcal { L } _ { W S R }$ is averaged over the ground-truth stressed words in the utterance, encouraging the dominant SSD probability $\hat { y } _ { i ^ { \star } } ^ { s s d }$ to approach 1 while suppressing the remaining positions within the same word span. L<sub>WSR</sub> is used as a regularization constraint over the SSD posteriors.

Table 1: Statistics of the TinyStress-15K corpus.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>#Audios</td><td rowspan=1 colspan=1>#Tokens</td><td rowspan=1 colspan=1>#Stress</td><td rowspan=1 colspan=1>#Unstress</td></tr><tr><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>13,500</td><td rowspan=1 colspan=1>171,206</td><td rowspan=1 colspan=1>22,537</td><td rowspan=1 colspan=1>148,669</td></tr><tr><td rowspan=1 colspan=1>Valid</td><td rowspan=1 colspan=1>1,500</td><td rowspan=1 colspan=1>18,676</td><td rowspan=1 colspan=1>2,541</td><td rowspan=1 colspan=1>16,135</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>1,000</td><td rowspan=1 colspan=1>12,548</td><td rowspan=1 colspan=1>1,705</td><td rowspan=1 colspan=1>10,843</td></tr></table>

In this view, $\mathcal { L } _ { S S D }$ provides coarse supervision for identifying stressed words, while $\mathcal { L } _ { W S R }$ serves as a within-word sharpening constraint that resolves the allocation ambiguity induced by subword tokenization.

## 2.4. Optimization

The final loss function is formulated as:

$$
\mathcal { L } = \alpha \cdot \mathcal { L } _ { S S D } + \beta \cdot \mathcal { L } _ { W S D } + \lambda \cdot \mathcal { L } _ { W S R } ,\tag{8}
$$

where $\alpha , \beta , \lambda \ge 0$ are the weighting coefficients. This design enables the framework to capture SSD, WSD, and WSR in a unified manner.

## 3. Experiment Setup

## 3.1. Dataset

Experiments are conducted on the TinyStress-15K corpus [14], which comprises synthetic speech with word-level stress annotations. Each instance includes an input utterance a, tokenlevel sentence stress detection (SSD) labels $y _ { m } ^ { s s d }$ derived via Whisper tokenization, and phone-level word stress detection (WSD) labels $y _ { n } ^ { w s d }$ obtained using a grapheme-to-phoneme (G2P) toolkit<sup>1</sup>. We apply the G2P converter to the complete refwsd erence transcript rather than to isolated words. We derive $y _ { n } ^ { u }$ from the G2P stress digits by labeling only primary-stressed phones (digit ‘1’) as 1, and mapping all other phones, including digit ‘2’, to 0. As summarized in Table 1, the training split contains 13.5k utterances and 171k tokens, including approximately 22.5k stressed and 148k unstressed tokens. The validation split comprises 1.5k utterances with 18.6k tokens, and the test split comprises 1k utterances with 12.5k tokens. Following the split protocol of [14], we adopt the official training and test partitions, and hold out 10% of the training split as a validation set. This design preserves a consistent evaluation protocol while providing a held-out validation set for model selection.

## 3.2. Comparison Methods

Table 2: SSD performance on TinyStress-15K. STRAW denotes our proposedframework with auxiliary WSD and the WSR. Ablations remove the corresponding components.  
Table 3: WSD performance of STRAW on TinyStress-15K.
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>GT alignment [14]</td><td>0.862</td><td>0.853</td><td>0.858</td></tr><tr><td>MFA [14]</td><td>0.776</td><td>0.859</td><td>0.815</td></tr><tr><td>WhiStress [14]</td><td>0.912</td><td>0.906</td><td>0.909</td></tr><tr><td>STRAW</td><td>0.945</td><td>0.924</td><td>0.934</td></tr><tr><td>- WSD</td><td>0.938</td><td>0.906</td><td>0.922</td></tr><tr><td>- WSR</td><td>0.942</td><td>0.917</td><td>0.929</td></tr><tr><td>- WSR &amp; WSD</td><td>0.940</td><td>0.892</td><td>0.915</td></tr></table>

We compare the proposed framework against both alignmentbased and alignment-free baselines. The alignment-based systems use ground-truth word-level timestamps or the Montreal Forced Aligner (MFA) to extract prosodic features such as duration, energy, and pitch, followed by a BLSTM classifier [11]. These models represent traditional pipelines that heavily rely on accurate alignment. The alignment-free baseline is WhiStress [14], which extends Whisper with a stress detection head trained on synthetic resources. WhiStress reports strong performance without requiring timestamps, but it treats sentence stress as an isolated objective and does not impose lexical stress constraints. Our extensions build upon this alignment-free setting by incorporating an auxiliary WSD objective and a WSR, enabling sentence- and word-stress prediction within a unified framework.

<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>STRAW</td><td>0.924</td><td>0.916</td><td>0.920</td></tr><tr><td>- WSR</td><td>0.935</td><td>0.907</td><td>0.921</td></tr></table>

## 3.3. Implementation Details

Model configurations were initialized using pretrained models from the HuggingFace Transformers library [26]. We employed whisper-small<sup>2</sup> as the backbone model and extracted hidden states from the ninth layer of the Whisper encoder/decoder for downstream modeling, following the configuration of WhiStress [14]. We freeze all Whisper parameters and train only the SSD head, the WSD head, and the phone embedding layer. Both prediction heads were optimized in the same training run using the multi-task objective in Section 2. Unless otherwise stated, we set $\alpha = 1 . 0 , \beta = 1 . 0 $ , and $\lambda = 1 . 0$ in Eq. 8 for all experiments. All models were trained on an NVIDIA 3090 GPU using the AdamW optimizer with a batch size of 16 and an initial learning rate of 1e-4. All models were trained for 20 epochs, and the checkpoint achieving the highest binary F1 score for SSD on the validation set was selected for evaluation. At inference time, SSD is performed given the input speech and a Whisper token sequence, while WSD estimates phone-level stress based on a G2P-derived phone sequence. In our read-aloud APA setting, the reference transcript provides Whisper decoder tokens for SSD and a G2P-derived phone sequence for WSD, without requiring an additional ASR pass. The WSD branch is optional at inference and may be disabled when only SSD feedback is required.

## 3.4. Evaluation Metrics

Evaluation adheres to the protocol in [14]. We report precision, recall, and F1 with respect to the stressed class. For SSD, token-level predictions are mapped back to their corresponding word spans in the reference transcript, and a word is marked as stressed if any of its constituent tokens is predicted as stressed. For WSD, predictions are produced at the phone level, and precision, recall, and F1 are computed directly over stressed phone labels.

## 4. Experimental Results

## 4.1. SSD Performance Comparison with Baselines

Table 2 summarizes SSD results on TinyStress-15K. We include both alignment-based and alignment-free baselines for reference; overall, alignment-based pipelines are sensitive to boundary quality, whereas the alignment-free WhiStress baseline provides a strong Whisper-based point of comparison. Building on the same frozen-Whisper setting, our proposed STRAW achieves the best overall performance, reaching an F1 of 0.934 and improving over WhiStress by 2.5 points (0.909 → 0.934) with consistent gains in both precision and recall.

![](images/e24ee79831a8db61862d00acea0ca53489db0d5ae0c962bcb769003ae351a814.jpg)  
(a) WhiStress

![](images/e7895319b63666527707000eb8c0890d776ee576ba994f167ab8ed2d315f4274.jpg)  
(b) STRAW  
Figure 3: Error analysis by part-of-speech (POS) on the test set. Bars show false negative rate (FNR) and false positive rate (FPR). The black line indicates the sample count per POS.

The ablation results show that removing WSD, WSR, or both reduces F1 to 0.922, 0.929, and 0.915, respectively. Because the two heads have separate trainable parameters, the WSD ablation does not establish direct knowledge transfer from WSD to SSD. The WSR ablation is consistent with its intended role of reducing diffuse SSD probabilities within stressed word spans. Overall, the complete configuration achieves the highest score among the evaluated settings.

## 4.2. WSD Performance

Table 3 reports WSD performance on TinyStress-15K. As TinyStress-15K is primarily used for SSD benchmarking, directly comparable external WSD baselines are limited; we report WSD results to contextualize the auxiliary task behavior. Overall, WSD achieves consistently strong results, with F1 around 0.92 across settings, suggesting that phone-level stress prediction on this benchmark is relatively well-conditioned under the provided supervision. In particular, STRAW attains 0.924 precision, 0.916 recall, and 0.920 F1. We further examine whether WSR affects WSD. Removing WSR yields only a marginal change in F1 (0.920 → 0.921). This behavior is expected given our training design, in which SSD is prioritized and WSR is introduced primarily to shape stress distributions for word-level prominence, whereas WSD is optimized with its own supervised objective. A promising direction for future work is to introduce explicit coupling between SSD and WSD, for example through shared trainable modules or cross-task consistency constraints.

## 4.3. Error Analysis

To better understand the behavior of the proposed models, we analyze error distribution across part-of-speech (POS) categories, as illustrated in Figure 3. The baseline system exhibits high false negative rates on functional categories, particularly PART (particle) and DET (determiner). The complete STRAW configuration shows a modest reduction in these errors, with additional improvements observed for AUX (auxiliary) and ADP (adposition). False positives remain consistently low, indicating that precision is not compromised. One exception is SCONJ (subordinating conjunction)<sup>3</sup>, where high false negative rates persist due to the limited number of samples. These observations suggest that the proposed framework primarily benefits frequent categories with sufficient coverage, which aligns with the overall recall gains reported in Table 2.

## 5. Conclusion

This work introduces STRAW, an SSD framework that augments token-level SSD with an auxiliary WSD head and a WSR. By supporting sentence- and word-level stress analysis within a unified framework and explicitly constraining token-level SSD predictions within stressed word spans, the proposed design mitigates within-word ambiguity induced by subword tokenization. Experiments on TinyStress-15K demonstrate improvements over strong baselines, with the complete configuration achieving the highest F1 score among the evaluated configurations. Three limitations remain. First, the current word-stress detection simplifies lexical stress to a single primary position per word and does not explicitly model syllabic structure or secondary stress. Second, the frozen backbone and separate taskspecific heads do not provide direct interaction between SSD and WSD. Third, the present evaluation is confined to synthetic speech due to the scarcity of large-scale, fine-grained stress annotations. Future work will address these issues by extending STRAW to spontaneous and human-recorded speech (e.g., Aix-MARSEC [27], Expresso [28], and EmphAssess [29]), introducing explicit coupling between the SSD and WSD heads, and incorporating richer linguistic knowledge to better capture prosodic structure in SSD.

## 6. Generative AI Use Disclosure

ChatGPT was used for language editing and for rephrasing selected passages to improve clarity and coherence. The authors take full responsibility for the technical content and claims, and they verified all descriptions of methods, experiments, and results.

## 7. References

[1] A. Van Moere and R. Downey, “21. technology and artificial intelligence in language assessment,” Handbook of second language assessment, vol. 12, 2016.

[2] C. Cucchiarini, H. Strik, and L. Boves, “Quantitative assessment of second language learners’ fluency: an automatic approach,” in Proceedings of ICSLP, 1998, pp. 2619–2622.

[3] D. R. Ladd, Intonational Phonology, 2nd ed., ser. Cambridge Studies in Linguistics. Cambridge University Press, 2008.

[4] N. Chomsky and M. Halle, “The sound pattern of english,” 1968. [Online]. Available: https://api.semanticscholar.org/CorpusID: 60457972

[5] D. Bolinger, “Accent is predictable (if you’re a mind-reader),” Language, vol. 48, no. 3, pp. 633–644, 1972. [Online]. Available: http://www.jstor.org/stable/412039

[6] V. Van Heuven, Acoustic Correlates and Perceptual Cues ofWord and Sentence Stress: Theories, Methods and Data, 2018, pp. 15– 59.

[7] S. Kakouros and O. Ras¨ anen, “3pro – an unsupervised method¨ for the automatic detection of sentence prominence in speech,” Speech Communication, vol. 82, pp. 67–84, 2016. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S0167639315300960

[8] T. Mishra, V. R. Sridhar, and A. Conkie, “Word prominence detection using robust yet simple prosodic features,” in Proceeding ofInterspeech, 2012, pp. 1864–1867.

[9] M. Morrison, P. Pawar, N. Pruyne, J. Cole, and B. Pardo, “Crowdsourced and automatic speech prominence estimation,” in Proceeding ofICASSP, 2024, pp. 12 281–12 285.

[10] M. de Seyssel, A. D’Avirro, A. Williams, and E. Dupoux, “EmphAssess : a prosodic benchmark on assessing emphasis transfer in speech-to-speech models,” in Proceedings of EMNLP, 2024, pp. 495–507. [Online]. Available: https://aclanthology.org/ 2024.emnlp-main.30/

[11] B. Lin, L. Wang, X. Feng, and J. Zhang, “Joint detection of sentence stress and phrase boundary for prosody,” in Proceeding of Interspeech, 2020, pp. 4392–4396.

[12] G. G. Lee, H.-Y. Lee, J. Song, B. Kim, S. Kang, J. Lee, and H. Hwang, “Automatic sentence stress feedback for non-native english learners,” Computer Speech & Language, vol. 41, pp. 29– 42, 2017.

[13] A. Suni, D. Aalto, and M. Vainio, “Hierarchical representation of prosody for statistical speech synthesis,” arXiv preprint arXiv:1510.01949, 2015.

[14] I. Yosha, D. Shteyman, and Y. Adi, “WhiStress: Enriching Transcriptions with Sentence Stress Detection,” in Proceeding of Interspeech, 2025, pp. 4718–4722.

[15] T.-A. Hung, Y.-H. Hsieh, T.-H. Lo, Y.-C. Hsu, and B. Chen, “Exploring sentence stress detection using whisper-based speech models,” in in Proceedings ofROCLING, 2025, pp. 314–319.

[16] L. Ferrer, H. Bratt, C. Richey, H. Franco, V. Abrash, and K. Precoda, “Classification of lexical stress using spectral and prosodic features for computer-assisted language learning systems,” Speech Communication, vol. 69, pp. 31–45, 2015.

[17] R. Delmonte, M. Petrea, C. Bacalu et al., “Slim prosodic module for learning activities in a foreign language,” in Proceeding of Eurospeech, vol. 2, 1997, pp. 669–672.

[18] J. Tepperman and S. Narayanan, “Automatic syllable stress detection using prosodic features for pronunciation evaluation of language learners,” in Proceedings of ICASSP, vol. 1, 2005, pp. I– 937.

[19] C. Yarra, O. D. Deshmukh, and P. K. Ghosh, “Automatic detection of syllable stress using sonority based prominence features for pronunciation evaluation,” in Proceedings of ICASSP, 2017, pp. 5845–5849.

[20] O. D. Deshmukh and A. Verma, “Nucleus-level clustering for word-independent syllable stress classification,” Speech Communication, vol. 51, no. 12, pp. 1224–1233, 2009.

[21] D. O. Johnson and O. Kang, “Automatic prominent syllable detection with machine learning classifiers,” International Journal ofSpeech Technology, vol. 18, no. 4, pp. 583–592, 2015.

[22] M. A. Shahin, J. Epps, and B. Ahmed, “Automatic classification of lexical stress in english and arabic languages using deep learning.” in Proceedings ofInterspeech, 2016, pp. 175–179.

[23] Y. Ruan, X. Wang, H. Liu, Z. Ou, Y. Gao, J. Cheng, and Y. Qian, “An end-to-end approach for lexical stress detection based on transformer,” arXiv preprint arXiv:1911.04862, 2019.

[24] J. Mallela, P. S. Boyina, and C. Yarra, “A comparison of learned representations with jointly optimized vae and dnn for syllable stress detection,” in Proceedings ofICSC, 2023, pp. 322–334.

[25] S. H. Aluru, J. Mallela, and C. Yarra, “Post-net2.0: An adaptive weighted loss function driven by linguistic constraint for automatic syllable stress detection,” in Proceedings ofICASSP, 2025, pp. 1–5.

[26] T. W. et al., “Transformers: State-of-the-art natural language processing,” in Proceedings ofEMNLP, 2020, pp. 38–45.

[27] C. Auran, C. Bouzon, and D. J. Hirst, “The aix-marsec project: an evolutive database of spoken british english,” Speech Prosody 2004, 2004. [Online]. Available: https: //api.semanticscholar.org/CorpusID:204957167

[28] T. A. Nguyen, W.-N. Hsu, A. d’Avirro, B. Shi, I. Gat, M. Fazel-Zarani, T. Remez, J. Copet, G. Synnaeve, M. Hassid et al., “Expresso: A benchmark and analysis of discrete expressive speech resynthesis,” arXiv preprint arXiv:2308.05725, 2023.

[29] M. de Seyssel, A. D’Avirro, A. Williams, and E. Dupoux, “Emphassess: a prosodic benchmark on assessing emphasis transfer in speech-to-speech models,” arXiv preprint arXiv:2312.14069, 2023.