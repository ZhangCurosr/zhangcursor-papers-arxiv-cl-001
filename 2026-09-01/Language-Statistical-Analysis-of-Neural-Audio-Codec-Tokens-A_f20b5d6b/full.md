# Language-Statistical Analysis of Neural Audio Codec Tokens Across Architectures, Corpora, and Noise Conditions

Joonyong Park, Student Member, IEEE, Shinnosuke Takamichi, Member, IEEE, David M. Chan, Member, IEEE, Shunsuke Kando, Student Member, IEEE, Yuki Saito, Member, IEEE, and Hiroshi Saruwatari, Member, IEEE

Abstract—Neural audio codecs (NACs) convert speech into discrete token sequences, and prior work has reported that these sequences follow language-like statistical laws. This paper analyzes the token statistics of 13 NACs spanning multicodebook residual vector quantization (RVQ), single-codebook VQ, and non-VQ designs, evaluated on three corpora under clean, white-noise, and real-world DEMAND-noise conditions. Zipf and Heaps parameters, unigram entropy, codebook occupancy, and Jensen–Shannon divergence (JSD) are estimated from matched token samples with explicit fit-validity safeguards and family-conditional n-gram orders. Corpus identity explains little variance in any metric, whereas acoustic condition and quantizer meta-category dominate in a metric-dependent way, and unigram entropy is the metric most strongly associated with meta-category. Clean-to-noise JSD computed at a common unigram order is associated with mel-cepstral distortion most clearly under DE-MAND noise. The collapse and explosion degradation signatures previously reported for RVQ codecs concentrate in RVQ cells under white and DEMAND noise, respectively; explosion also occurs in non-VQ codecs, and single-codebook VQ codecs shift in occupancy and distribution shape without either signature. These results provide architecture-conditioned conventions for applying language-statistical analysis to NAC tokens.

Index Terms—Neural audio codec, Zipf’s law, Heaps’ law, speech tokenization, distributional analysis, noise robustness, speech quality prediction.

## I. INTRODUCTION

N <sup>EURAL</sup> <sup>audio</sup> <sup>codecs</sup> <sup>(NACs)</sup> <sup>have</sup> <sup>emerged</sup> <sup>as</sup> <sup>a</sup> foundational component in modern speech processing pipelines. By converting continuous acoustic waveforms into sequences of discrete tokens through learned codebooks, NACs enable the direct application of powerful sequencemodeling techniques—originally developed for natural language processing—to speech data [1], [2]. This paradigm has been adopted across a wide range of downstream tasks, including text-to-speech (TTS) synthesis [3], [4], automatic speech recognition (ASR) [5], and audio generation [6].

Despite their growing importance, a fundamental question remains underexplored: do NAC token sequences exhibit statistical regularities similar to those found in natural language, and if so, what do these regularities reveal about codec quality and robustness? Natural language text is known to follow well-established statistical laws, notably Zipf’s law [7] governing rank–frequency distributions and Heaps’ law [8] governing vocabulary growth. If NAC tokens share these properties, analytical tools developed for natural language become applicable to speech tokens on a known statistical footing: these laws provide compact descriptions of frequency concentration, vocabulary growth, and per-token information content. If the distributions diverge substantially, conclusions and design heuristics imported from natural language processing may not hold for speech tokens.

Our previous work [9] provided initial evidence that NAC tokens, particularly at the 3-gram level, exhibit Zipfian distributions. The degree of adherence was further shown to correlate with the resynthesis quality of the same codec, measured by ASR word error rate and by the pseudo-MOS predictor UTMOS [10]. However, that analysis was limited to clean speech from a single corpus and a small set of NACs based on multi-codebook residual vector quantization (RVQ). It left open questions about estimation reliability, noise robustness, cross-corpus generalization, and the mechanisms underlying distributional changes.

In this paper, we substantially broaden this line of inquiry along both empirical and methodological dimensions. On the empirical side, we expand from six RVQ-based NACs to 13 NACs spanning three quantizer meta-categories: multicodebook RVQ, single-codebook VQ, and non-VQ alternatives including scalar-quantized, focal-modulation, and semantic designs. The evaluation covers single-speaker, multi-speaker, and multilingual corpora under clean, white-noise, and DEMANDnoise conditions [11]. In place of the single fixed corpus size of our previous work [9], a subdivided chunk design makes finite-sample stability explicit. On the methodological side, we augment the Zipf / Heaps / entropy triad with codebook occupancy and token perplexity. We further introduce Jensen– Shannon divergence (JSD)—computed between the n-gram frequency distributions that the same codec produces on clean and on noise-corrupted versions of the same corpus—as a nonparametric measure of how far noise displaces the token distribution. Finally, we decompose the variance of every metric with a three-way ANOVA over codec quantizer category, corpus, and noise condition.

Building on this broadened design, our contributions fall into three themes. First, we show that which factor shapes the token distribution depends on which statistic is measured, while the choice of corpus matters remarkably little throughout. On clean speech, the quantizer architecture separates the codec families: semantically informed RVQ designs sit near the word-frequency power-law exponent of natural language, whereas single-codebook designs produce far steeper frequency distributions. Once noisy conditions enter, the acoustic condition becomes the main driver of the fitted exponents, and its effect differs across architecture families. Second, we find that clean-to-noisy JSD is a decoder-free distributionlevel diagnostic whose association with perceived quality is clearest under real-world DEMAND noise. Third, we identify collapse and explosion degradation signatures in token sequences through analysis of codebook occupancy, token repetition rates, and transition entropy. Collapse is concentrated in RVQ codecs under white noise. Explosion is most frequent in RVQ codecs under DEMAND noise and otherwise appears almost exclusively for the semantic tokenizer. Single-codebook VQ codecs deform distributionally under noise without either signature. We further show that the 3-gram optimum reported in [9] is family-dependent rather than universal, motivating per-family analysis conventions for non-RVQ codecs.

The remainder of this paper is organized as follows. Section II reviews related work. Section III defines the statistical metrics and analysis framework. Section IV describes the experimental setup. Section V establishes estimation reliability. Sections VI–VII present cross-architecture, cross-corpus, and quality correlation analyses. Section VIII characterizes architecture-dependent degradation signatures. Section IX discusses implications and limitations, and Section X concludes this paper.

## II. RELATED WORK

## A. Discrete Tokens from Speech

Self-supervised learning (SSL) models such as Hu-BERT [12] and wav2vec 2.0 [13] learn frame-level speech representations from unlabeled audio. Although these representations are themselves continuous, they are commonly discretized—typically by k-means clustering—to obtain token sequences for downstream processing. The resulting tokens capture phonetic and speaker-related information [14], making them effective for tasks including spoken language modeling and speech recognition.

NACs offer a complementary approach, producing discrete tokens explicitly optimized for high-fidelity audio reconstruction. Representative models such as SoundStream [1] and EnCodec [2] employ encoder–quantizer–decoder architectures with RVQ, compressing audio into multi-level token sequences at various bitrates. Subsequent work has expanded this design space considerably. SpeechTokenizer [3] disentangles semantic and acoustic information across RVQ layers, HiFi-Codec [15] introduces group-RVQ for improved fidelity, and AudioDec [16] targets streaming scenarios. FunCodec [17] provides a flexible open-source toolkit, and DAC [18] achieves high compression ratios with improved training on the basis of generative adversarial networks (GANs) [19]. More recent designs explore fundamentally different architectures. WavTokenizer [20] and BigCodec [21] adopt large single-codebook designs, XCodec2 [22] uses a massive 65,536-entry vocabulary, and SQCodec [23] replaces vector quantization with scalar quantization. Mimi [24] combines RVQ with semantic objectives for real-time dialogue, FocalCodec [25] applies focal modulation for low-bitrate coding, and S3Tokenizer [26] produces semantic-only tokens without a decoder. A recent survey by Guo et al. [27] provides a comprehensive overview of this rapidly diversifying design space.

The availability of discrete speech tokens has enabled a new generation of language-model-based speech processing systems. AudioLM [6] demonstrated that autoregressive modeling of audio tokens can generate coherent speech continuations, while VALL-E [4] showed that treating TTS as a language modeling problem over codec tokens enables zero-shot speaker adaptation. Codec tokens have also been integrated into ASR pipelines [5], where compressed representations reduce computational costs while maintaining recognition accuracy. This growing reliance on NAC tokens as the interface between speech and language modeling motivates a deeper understanding of their distributional structure.

## B. Statistical Laws of Language

Zipf’s law [7] is among the most robust empirical regularities observed in natural language: the frequency of the r-th ranked word in a corpus follows a power law $f ( r ) \propto r ^ { - s } ,$ with $s \approx 1$ at the word level. Equivalently, the distribution of word frequencies follows a power law with exponent $\alpha = 1 + 1 / s \approx 2 ~ [ 2 8 ]$ . The phenomenon was formalized by Mandelbrot [29] and has been confirmed across typologically diverse languages [30]. Complementarily, Heaps law [8] describes sublinear vocabulary growth as a function of corpus size, reflecting the diminishing rate at which new word types are introduced. Shannon entropy [31] provides a third lens, quantifying the average information content per symbol and enabling analysis of coding efficiency and redundancy. Together, these three measures—frequency distribution shape, vocabulary growth rate, and information density—provide a comprehensive characterization of the statistical structure of any symbolic system.

While originally studied in human language, these statistical laws exhibit remarkable universality. Power-law distributions have been documented in domains ranging from city population sizes and citation counts to internet traffic patterns [28], [32]. More recently, Chan et al. [33] demonstrated that tokens produced by visual codecs (VQ-VAE models for image compression) also exhibit Zipf-like and Heaps-like behavior, and that the degree of adherence to these laws correlates with downstream task performance. Zipf- and Heaps-like behavior can therefore arise in learned visual token systems as well as in text.

## C. Statistical Analysis of Speech Tokens

The statistical analysis of discrete speech tokens has attracted growing attention as these tokens now serve as the direct modeling target of speech language models. Their frequency structure determines how reliably n-gram statistics and neural language models can be estimated from a given amount of speech. Takamichi et al. [34] conducted the first systematic investigation of whether speech tokens follow Zipf’s law, focusing on SSL-based representations from HuBERT. Their results confirmed power-law behavior in token frequency distributions, providing evidence that discrete speech tokens share fundamental statistical properties with natural language text. Sicherman and Adi [35] further examined the interpretability and redundancy of SSL-based speech tokens, revealing strong correlations between learned tokens and phonemic categories. They proposed deduplication and redundancy-reduction techniques that improved downstream performance in spoken language modeling, highlighting the practical implications of understanding token distributional structure. Most recently, Ashihara et al. [36] extended this line of analysis along the domain axis, showing through rank–frequency, perplexity, and TF-IDF analyses that both SSL-based semantic tokens and NAC tokens retain power-law structure across the speech, music, and general-sound domains, while their token-usage patterns remain domain-dependent.

While these studies focused on SSL-based tokens or on cross-domain comparisons, the statistical properties of NAC tokens under controlled within-speech variation have received comparatively less attention. Our previous study [9] presented the first systematic analysis of NAC token distributions across six NACs in 15 model configurations that vary bitrate and sampling rate, demonstrating that 3-gram token sequences exhibit Zipf and Heaps law behavior similar to natural language. The degree of Zipfian adherence (α approaching 2 with lower Kolmogorov–Smirnov (KS) distance [37]) was found to correlate with resynthesis quality metrics such as word error rate (WER) and UTMOS [10].

However, these prior studies share limitations that motivate the present work. The analysis in [9] was confined to a single clean corpus, and the condition-level correlation coefficients were modest $( | r | < 0 . 3 )$ . Estimation reliability, multi-corpus generalizability, and the mechanisms underlying distributional degradation remained unexplored.

## D. Noise Robustness of Neural Audio Codecs

The robustness of NAC models to acoustic degradation is a critical practical concern, as real-world audio frequently contains environmental noise, reverberation, and channel distortions. Most NAC models are trained predominantly on clean or studio-quality speech [27], and their behavior under mismatched conditions—where input audio quality deviates substantially from training data—remains insufficiently characterized.

Existing codec evaluation frameworks such as Codec-SUPERB [38] provide standardized benchmarks for comparing codec quality across multiple downstream tasks, but focus predominantly on clean input conditions. Wu et al. [39] surveyed the broader landscape of audio language modeling, noting the importance of codec robustness without providing systematic degradation analysis. While individual codec papers report reconstruction quality under controlled conditions, no prior work has systematically examined how the distributional structure of codec tokens—as characterized by Zipf’s law, Heaps’ law, and entropy—changes under noise. This gap matters because distributional metrics can be computed directly from token sequences without audio resynthesis, making them candidate decoder-free, token-space indicators of codec behavior.

## III. METHODOLOGY

The present study addresses the gaps identified above through a fully crossed design: 13 codecs spanning three quantizer meta-categories are evaluated on three corpora under a uniform three-condition protocol (clean, white noise, DEMAND noise). Every token stream then passes through a single pipeline of deduplication, family-conditional n-gram mapping, and chunk-based estimation; the sequence-level diagnostics of Section VIII are computed before deduplication. On this grid, we quantify estimation reliability explicitly and decompose the variance of every distributional metric across meta-category, corpus, and condition. We further evaluate JSD as a decoder-free degradation diagnostic alongside resynthesisquality metrics. This section defines the pipeline, the metrics, and the analysis framework that implement this design.

## A. Token Extraction Pipeline

We adopt the Codec-SUPERB framework [38] for standardized token extraction across all NACs. Each NAC processes input audio and produces a token sequence. For RVQ-based NACs the output has one level per residual codebook, and we analyze the first level, which is empirically validated in Section V-C1. For single-codebook, scalar-quantized, or semantic tokenizers, we use the single available token stream.

From the extracted tokens, we construct n-gram sequences using a family-conditional order chosen on the basis of the minimum-KS ablation of Section V-B (column n of Table I). Because only the first stream is analyzed, each utterance yields a one-dimensional token sequence. Following [9], we apply consecutive deduplication to this sequence: runs of identical tokens are collapsed to a single instance, which removes frame-level repetitions that would bias n-gram frequency counts. The effect of this step is quantified in Section V-C2. Deduplicated utterance sequences are then concatenated into corpus-level streams for the chunk analysis of Section III-D.

## B. Statistical Metrics

1) Zipf’s Law: Zipf’s law states that the frequency f(r) of the r-th ranked element in a corpus follows a power law:

$$
f ( r ) = a \cdot r ^ { - s } ,\tag{1}
$$

where $a > 0$ is a scaling constant and $s > 0$ is the rank– frequency exponent; natural language text typically exhibits s ≈ 1 at the word level [7]. We quantify this behavior through the equivalent frequency-distribution form: under Eq. 1, the occurrence counts x of the individual n-gram types follow $p ( x ) \propto x ^ { - \alpha }$ with $\alpha = 1 + 1 / s [ 2 8 ]$ , so the word-level reference value corresponds to $\alpha \approx 2 .$ Every Zipf exponent reported in this paper is this frequency-distribution exponent α.

We estimate α using maximum likelihood estimation (MLE) over the multiset of type frequencies via the powerlaw library [40]:

$$
\hat { \alpha } _ { \mathrm { M L E } } = 1 + \frac { n } { \sum _ { i = 1 } ^ { n } \log ( x _ { i } / x _ { \operatorname* { m i n } } ) } ,\tag{2}
$$

where $x _ { i }$ is the occurrence count of the i-th n-gram type, x<sub>min</sub> is the minimum-frequency threshold selected by the library, and n is the number of types with $x _ { i } \geq x _ { \operatorname* { m i n } }$ . Goodness of fit is assessed using the KS distance between the empirical and fitted cumulative distribution functions of the type frequencies;

lower KS values indicate better fit. Two safeguards apply to every fit in this paper. First, whenever the library’s default search bound truncates the estimate at $\alpha \ = \ 3$ , we refit with the parameter range expanded to $\alpha \in [ 1 . 0 1 , 8 ]$ . Second, we treat a fit as undefined when the type–token ratio $V / N$ exceeds 0.9. In this regime, the observed n-gram inventory is nearly unique relative to the sample size and the frequency profile is degenerate, so the estimator produces arbitrarily steep exponents with spuriously small KS values.

2) Heaps’ Law: Heaps’ law [8] describes the sublinear growth of vocabulary size V as a function of the total number of tokens N:

$$
V ( N ) = K \cdot N ^ { \beta } , \quad 0 < \beta < 1 ,\tag{3}
$$

where $K \ > \ 0$ is a scaling factor reflecting the initial vocabulary introduction rate, and $\beta$ determines how quickly new types continue to appear. For word-level English text, K typically lies in the range 10–100 [8]. Values of $\beta$ close to 1 indicate near-linear vocabulary growth (high diversity), while lower values suggest vocabulary saturation. Word-level natural language text typically exhibits $\beta \approx 0 . 4 \mathrm { - } 0 . 6 ~ [ 8 ]$

We estimate $\beta$ and K via ordinary least squares (OLS) regression on log-transformed data:

$$
\log V ( N ) = \log K + \beta \cdot \log N .\tag{4}
$$

3) Shannon Entropy and Redundancy: The Shannon entropy of a token sequence measures the average information content per token:

$$
H = - \sum _ { i = 1 } ^ { | V | } p _ { i } \log _ { 2 } p _ { i } ,\tag{5}
$$

where $p _ { i }$ is the probability of the i-th token type and |V | is the number of observed token types. The redundancy R quantifies how far the observed entropy falls from the theoretical maximum under a uniform distribution:

$$
R = { \frac { \log _ { 2 } | V | - H } { \log _ { 2 } | V | } } ,\tag{6}
$$

where $\log _ { 2 } | V |$ is the maximum entropy achievable over the observed vocabulary. For printed English, Shannon estimated a redundancy of roughly one half, corresponding to about one bit of information per letter [31]. We also report the token perplexity $2 ^ { H }$ , which re-expresses the entropy as the effective number of frequently used token types under the unigram distribution.

4) JSD: To quantify how noise alters the token distribution relative to clean speech, we compute JSD. JSD serves as a nonparametric measure computed over the empirical frequency distribution of all observed n-gram types, rather than over a small number of fitted summary parameters such as α or β. Given a clean reference distribution P and a degraded distribution Q over the shared n-gram vocabulary, JSD is defined as:

$$
\begin{array} { r } { \mathrm { J S D } ( P \| Q ) = \frac { 1 } { 2 } D _ { \mathrm { K L } } ( P \| M ) + \frac { 1 } { 2 } D _ { \mathrm { K L } } ( Q \| M ) , } \end{array}\tag{7}
$$

where $M = { \textstyle \frac { 1 } { 2 } } ( P + Q )$ is the mixture distribution and $D _ { \mathrm { K L } }$ denotes the Kullback–Leibler divergence. JSD is bounded in [0, 1], symmetric, and well-defined even when P and Q have non-overlapping support.

## C. Quality Metrics

Each codec is driven at the sampling rate it was trained for. Thus, the corpus waveform is resampled to that rate before encoding. Resynthesis uses each codec’s complete standard decoding path: the original, non-deduplicated token sequences at all codebook levels are decoded to a waveform. The decoded waveform is then resampled back to 16 kHz, at which every metric below is computed. We evaluate resynthesized speech using five metrics:

• dCER: Differential character error rate. Whisper-largev3 [41] transcribes both the clean ground-truth waveform and the resynthesized waveform. The character error rate between the two transcripts is computed per utterance after multilingual text normalization and capped at 1.0.

• UER: Unit error rate, the Levenshtein distance between the token sequences obtained from the clean and noisecorrupted versions of the same utterance, normalized by the clean sequence length. UER is defined only under noisy conditions.

• UTMOS: Pseudo-MOS score estimating perceived naturalness [10].

• MCD: Standard mel-cepstral distortion in dB, computed from mel-cepstral features extracted with WORLD [42], with DTW alignment, using the clean ground-truth waveform as reference. The distance includes the zeroth melcepstral coefficient, so the resynthesis is rescaled to the root-mean-square level of its reference.

• F0 error: F0 is extracted with Praat’s autocorrelation method [43] (search range 50–1,100 Hz, voicing threshold 0.6). The two contours are aligned by DTW as for MCD, and the error is taken over frames voiced in both signals. We report the per-utterance median absolute error (F0MedAE) alongside the conventional root-mean-square error.

## D. Analysis Framework

1) Chunk Analysis: We partition each corpus-level token stream into non-overlapping chunks of increasing size (5K to 500K tokens) and compute the Zipf/Heaps parameters once per chunk; deviations from the matched 500K-token reference of Section V-A quantify estimation reliability, i.e., finite-sample stability.

2) Codebook Occupancy: As an additional diagnostic capturing token-usage concentration rather than distribution shape, we report codebook occupancy: the number of unique tokens that appear in a sequence, normalized by an estimate of the codebook size. Because the nominal codebook size is unavailable for some non-VQ codecs (column |V| of Table I), the denominator is the largest observed token index plus one for every codec. This proxy assumes contiguous token indexing; for codecs without a nominal codebook size, we interpret occupancy only through within-codec changes across conditions.

3) Variance Decomposition: We perform a three-way ANOVA with codec quantizer category, corpus, and noise condition as factors. The ANOVA decomposes the variance in the distributional parameters, with effect sizes reported via $\eta ^ { 2 }$ . Each codec contributes multiple cells across corpora and conditions, so the reported $\eta ^ { 2 }$ values are descriptive variance partitions over the cell grid rather than tests on independent samples. A linear mixed model with a codec random intercept serves as the corresponding robustness check.

![](images/47ea018f98c2e59dac891f515537fdfba02f3ccfa9f9f43f55ce5c1bd9979ac1.jpg)  
Fig. 1: Overview of the experimental design.

## IV. EXPERIMENTAL SETUP

Figure 1 summarizes the evaluation design and the analysis flow of Sections V–VIII.

## A. NACs

We evaluate 13 NAC models spanning three quantizer metacategories, as summarized in Table I. All 13 codecs are analyzed on the same three corpora × three conditions grid described in Sections IV-B–IV-C, which yields a total of 117 codec–corpus–condition cells. Each codec is evaluated in the default configuration of its released implementation, without manual bandwidth adjustment (column kbps of Table I).

For analytical purposes, we group the 13 codecs into three meta-categories based on quantizer topology (column M of Table I): (R) multi-codebook RVQ — classic 1,024-entry RVQ designs (SpeechTokenizer, AcademiCodec, AudioDec, EnCodec, FunCodec, DAC-24k) plus the hybrid RVQ codec Mimi [24]; (S) single-codebook VQ — designs using a single but larger codebook (BigCodec [21], WavTokenizer [20], XCodec2 [22]); and (N) non-VQ alternatives — architectures that replace or bypass standard vector quantization (SQCodec [23]: scalar quantization; FocalCodec [25]: focal modulation; S3Tokenizer [26]: semantic-only). This grouping enables family-aware analysis of how codebook size, quantizer type, and architectural choice affect distributional properties.

## B. Speech Corpora

We use three corpora selected to cover three practically relevant sources of variation beyond acoustic condition: speaker diversity, language diversity, and recording style (Table II). All corpora are approximately matched at ∼10 hours of speech. LJSpeech [44] provides a single-speaker monolingual reference, and VoxCeleb [45] adds multi-speaker variability. TrIJEK contributes multilingual coverage (Japanese, Korean,

TABLE I: NAC configurations. Family: architectural family. M: meta-category (R: multi-codebook RVQ; S: singlecodebook VQ; N: non-VQ). |V |: codebook vocabulary size (‘var.’: no nominal size; the observed-token proxy of Section III-D is used). n: n-gram order used in all analyses. SR: sampling rate. kbps: bitrate of the evaluated configuration (‘–’: not applicable or not published). †: semantic-only, no decoder; audio-based quality metrics unavailable.
<table><tr><td>Codec</td><td>Family</td><td>M</td><td>|V|</td><td>n</td><td>SR</td><td>kbps</td></tr><tr><td>SpeechTokenizer</td><td>Classic RVQ</td><td>R</td><td>1,024</td><td>3</td><td>16k</td><td>4</td></tr><tr><td>AcademiCodec</td><td>Classic RVQ</td><td>R</td><td>1,024</td><td>3</td><td>24k</td><td>3</td></tr><tr><td>AudioDec</td><td>Classic RVQ</td><td>R</td><td>1,024</td><td>3</td><td>24k</td><td>6.4</td></tr><tr><td>EnCodec</td><td>Classic RVQ</td><td>R</td><td>1,024</td><td>3</td><td>24k</td><td>6</td></tr><tr><td>FunCodec</td><td>Classic RVQ</td><td>R</td><td>1,024</td><td>3</td><td>16k</td><td>16</td></tr><tr><td>DAC-24k</td><td>Classic RVQ</td><td>R</td><td>1,024</td><td>3</td><td>24k</td><td>24</td></tr><tr><td>Mimi</td><td>Hybrid RVQ</td><td>R</td><td>2,048</td><td>2</td><td>24k</td><td>4.4</td></tr><tr><td>BigCodec</td><td>Large single</td><td>S</td><td>8,192</td><td>2</td><td>16k</td><td>1.04</td></tr><tr><td>WavTokenizer</td><td>Large single</td><td>S</td><td>4,096</td><td>2</td><td>24k</td><td>一</td></tr><tr><td>XCodec2</td><td>Massive vocab</td><td>S</td><td>65,536</td><td>1</td><td>16k</td><td>一</td></tr><tr><td>SQCodec</td><td>Scalar quant.</td><td>N</td><td>var.</td><td>1</td><td>16k</td><td>3</td></tr><tr><td>FocalCodec</td><td>Focal mod.</td><td>N</td><td>8,192</td><td>1</td><td>16k</td><td>0.65</td></tr><tr><td>S3Tokenizer†</td><td>Semantic</td><td>N</td><td>4,096</td><td>3</td><td>16k</td><td>一</td></tr></table>

TABLE II: Speech corpora used in this study.
<table><tr><td>Corpus</td><td>Category</td><td>Language</td><td>Speakers</td><td>Hours</td></tr><tr><td>LJSpeech</td><td>Single-speaker mono</td><td>English</td><td>1</td><td>~10</td></tr><tr><td>VoxCeleb</td><td>Multi-speaker</td><td>English</td><td>Many</td><td>~10</td></tr><tr><td>TrIJEK</td><td>Multilingual mono</td><td>JA/KO/EN</td><td>1</td><td>~10</td></tr></table>

English) recorded by a single speaker, which reduces speaker variation across the three languages. All 13 codecs are evaluated on the full three-corpora set.

## C. Noise Conditions

To assess noise robustness, we evaluate two noisy variants of each corpus alongside the clean condition:

• Clean: the original recordings with no additional noise.

• White Noise: Gaussian white noise added at 0 dB SNR.

• DEMAND Noise: real-world noise from the DEMAND corpus [11] added at 0 dB SNR. For each utterance, one of five DEMAND environments (office meeting, cafeteria, restaurant, bus, metro) is drawn at random, together with a random recording channel and start offset. Each cell therefore pools several realistic environments.

Noisy waveforms are generated once per corpus with a fixed random seed, and the identical noisy waveform is presented to every codec. Cross-codec differences under noise therefore do not reflect noise-realization variability. Clean is treated as the reference condition for computing all clean-to-noisy deviation metrics.

## V. ESTIMATION RELIABILITY AND ANALYSIS CONFIGURATION

## A. Finite-Sample Stability

Before comparing codecs and acoustic conditions, we assess the finite-sample stability of the distributional parameters at the sample scale used in the subsequent experiments, using the chunk analysis defined in Section III-D1. For each codec– corpus–condition cell, we calculate each parameter from nonoverlapping chunks of increasing size and compare it with the estimate obtained from a matched 500K-token sample. For the Heaps intercept K, whose scale varies substantially across codecs, we use the absolute log-ratio rather than the raw absolute difference. A matched reference is available for 114 of the 117 cells; the three Mimi–TrIJEK cells are excluded because their token streams do not reach this size.

![](images/185a2ba56e4354a0e6e93504b113d8a7c8c86ed44ca85e1d2497434d06ab4173.jpg)  
Fig. 2: Finite-sample stability across the 114 codec–corpus–condition cells with a matched 500K-token reference. Lines and shaded regions show the median and interquartile range of the absolute deviation from that reference (absolute log-ratio for K), normalized by the cross-cell standard deviation of the reference values so that the four panels share one scale. Dashed guides mark deviations of 0.1 and 1.0 times that spread.

Figure 2 summarizes the 114 cells with a matched reference. As the four parameters have different units and scales, each deviation is divided by the cross-cell standard deviation of the 500K-token reference values, so that all panels share one scale. The KS distance converges fastest. Pooled across conditions, its median deviation falls from 0.55 of the cross-cell spread at 5K tokens to 0.07 at 200K, making it the only parameter to reach the lower 0.1 guide. β and α remain at 0.17 and K at 0.19 at 200K tokens. At the opposite end, α is the least stable parameter at 5K tokens, where its median deviation 1.06 exceeds the full cross-cell spread. We therefore use matched 500K-token samples as the common reference scale for all downstream comparisons.

## B. Selection and Sensitivity of n-gram Order

Sweeping the n-gram order for all 13 codecs, with the fitting safeguards of Section III-B, shows that the optimal order is architecture-dependent, as illustrated in Fig. 3. Most multi-codebook RVQ codecs favor n = 3 or 4, with DAC-24k as the exception (Table III). Single-codebook and largevocabulary codecs favor lower orders because their higherorder inventories rapidly approach V ≈ N, beyond which the fit is undefined.

The analysis orders used in the main experiments were fixed before this ablation and are retained for consistency with the completed analyses. Table III reports the per-corpus optima: the adopted order matches the corpus-mean KS optimum for eight of the 13 codecs, deviates by one order for four codecs, and by two orders for Mimi. Table IV quantifies the crossfamily α comparison at fixed common orders. At the unigram order, which covers the full grid, single-codebook VQ codecs have a higher mean α than multi-codebook RVQ codecs, but the meta-category effect is not significant. At n = 2 the separation is large and significant, but valid fits remain for only 25 of the 39 codec–corpus cells, and the singlecodebook VQ family retains a single valid cell. We therefore use the adopted orders for within-codec analyses and treat absolute cross-family differences under codec-specific orders as descriptive.

![](images/211f515aad7adb6ac7a65d0dffb877beaefe1ba5a1287715b00501fbf8073680.jpg)  
Fig. 3: Effect of n-gram order on the Zipf exponent α and KS fit quality, averaged over three clean corpora. For codecs whose higher orders fail the V /N validity criterion the sweep terminates early; the last valid order of such series is marked with ×.

## C. Pipeline Validation

Beyond the n-gram order, the extraction pipeline of Section III-A involves two further choices: the analyzed RVQ level and consecutive deduplication. We validate those in Table V.

1) RVQ-Level Selection: For multi-codebook RVQ codecs, we use the first residual-quantizer level. In the five codecs with level-wise results, the first level has a lower KS distance than later levels (Table Va). Thus, we retain this convention for the full codec set.

2) Deduplication: We apply consecutive deduplication before fitting the distributional models, following [9]. On the same cells, deduplication increases the estimated α and β with a medium standardized effect (Cohen’s d ≈ 0.5) but does not significantly change KS or K (Table Vb).

TABLE III: Per-corpus n-gram order selection. M: metacategory, where R, S, and N denote multi-codebook RVQ, single-codebook VQ, and non-VQ codecs. The remaining columns give the KS-optimal order for each clean corpus, the order minimizing the corpus-mean KS, and the order used in the main analyses, computed under the fitting safeguards of Section III-B. <sup>∗</sup>: the adopted order deviates from the corpusmean optimum by one; <sup>†</sup>: by two.
<table><tr><td>Codec</td><td>M</td><td>LJS</td><td>TrIJEK</td><td>Vox</td><td>Argmin KS</td><td>Used</td></tr><tr><td>AcademiCodec*</td><td>R</td><td>4</td><td>3</td><td>4</td><td>4</td><td>3</td></tr><tr><td>AudioDec</td><td>R</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>DAC-24k*</td><td>R</td><td>2</td><td>2</td><td>2</td><td>2</td><td>3</td></tr><tr><td>EnCodec</td><td>R</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>FunCodec</td><td>R</td><td>3</td><td>3</td><td>2</td><td>3</td><td>3</td></tr><tr><td>Mimi†</td><td>R</td><td>4</td><td>3</td><td>3</td><td>4</td><td>2</td></tr><tr><td>SpeechTokenizer*</td><td>R</td><td>4</td><td>4</td><td>4</td><td>4</td><td>3</td></tr><tr><td>BigCodec</td><td>S</td><td>1</td><td>1</td><td>1</td><td>1</td><td>2</td></tr><tr><td>WavTokenizer</td><td>S</td><td>1</td><td>2</td><td>1</td><td>2</td><td>2</td></tr><tr><td>XCodec2</td><td>S</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>FocalCodec</td><td>N</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>S3Tokenizer</td><td>N</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>SQCodec</td><td>N</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></table>

TABLE IV: Fixed-order sensitivity of the cross-family α comparison on clean speech: number of codec–corpus cells with valid fits, per-family mean±std α, and the one-way metacategory effect size at each common order. $\ddagger ;$ a single valid cell remains, so no std is defined.
<table><tr><td>Order</td><td>N</td><td>αRVQ</td><td> $\overline { { \alpha } } _ { \mathrm { S i n g l e V Q } }$ </td><td> $\overline { { \alpha } } _ { \mathrm { N o n V Q } }$ </td><td> $\eta _ { \mathrm { m e t a } } ^ { 2 } \ ( p )$ </td></tr><tr><td>n = 1</td><td>39</td><td>2.71±0.63</td><td> $2 . 9 7 { \pm } 0 . 0 3$ </td><td>2.88±0.11</td><td>0.05 (0.37)</td></tr><tr><td>n = 2</td><td>25</td><td>2.24±0.41</td><td> $4 . 3 9 ^ { \ddagger }$ </td><td>2.99±0.01</td><td>0.63 (&lt;0.001)</td></tr></table>

## VI. CROSS-CORPUS AND CROSS-ARCHITECTUREANALYSIS

## A. Distributional Landscape

Figure 5 shows the clean-speech $\alpha { - } \beta$ landscape of 13 codecs on LJSpeech, VoxCeleb, and TrIJEK at the 500Ktoken reference scale. The estimates use the codec-specific n-gram orders adopted in Section V-B. Corpus-level variation is limited within the multi-codebook RVQ family. The RVQ-family averages are similar for LJSpeech $( \alpha = 2 . 3 2$ $\beta ~ = ~ 0 . 8 4 )$ , VoxCeleb $( \alpha = 2 . 4 4 , \beta = 0 . 9 1 )$ , and TrIJEK $( \alpha = 2 . 4 2 , \beta = 0 . 9 1 )$ . Table VI lists the per-codec parameters underlying this landscape, averaged over the corpora with valid fits. Figure 4 shows the empirical frequency distributions and fitted power laws behind three representative rows, visualizing what small and large KS distances correspond to.

Architecture families occupy distinct regions of the distributional space. Under the adopted analysis orders, multi-codebook RVQ codecs span $\alpha ~ \approx ~ 2 . 0 \ – 2 . 7$ , with the semantically informed Mimi and SpeechTokenizer at the low end (Table VI). Among the non-VQ codecs, SQCodec and FocalCodec sit near $\alpha \approx 2 . 8 – 3 . 0$ under their adopted orders, adjacent to XCodec2 at its unigram order. The semantic tokenizer S3Tokenizer is lower at 2.41. FocalCodec has the lowest Heaps exponent in the codec set, reflecting its nearcomplete use of the 8,192-entry codebook within each 500Ktoken sample. Among valid clean-speech fits, α is lowest for Mimi and highest for WavTokenizer.

TABLE V: Pipeline validation on the five multi-codebook RVQ codecs with level-wise extractions (first-level 3-gram streams; 105 codec–condition cells). (a) KS fit distance by RVQ level (mean±std). (b) Effect of consecutive deduplication on the first-level stream; p from paired t-tests.
<table><tr><td>(a) KS distance by RVQ level Level</td><td>1st</td><td>2nd-4th</td><td>5th-8th</td><td>9th+</td></tr><tr><td>KS</td><td>0.012±0.007</td><td>0.049±0.068</td><td>0.110±0.095</td><td>0.105±0.052</td></tr><tr><td colspan="5">(b) Deduplication effect on the first level</td></tr><tr><td></td><td>α</td><td>KS</td><td>β</td><td>K</td></tr><tr><td>Without dedup</td><td>2.26</td><td>0.012</td><td>0.81</td><td>4.72</td></tr><tr><td>With dedup</td><td>2.33</td><td>0.012</td><td>0.82</td><td>4.63</td></tr><tr><td>Cohen&#x27;s d (p)</td><td>0.46 (&lt;0.001)</td><td>-0.06 (0.54)</td><td>0.49 (&lt;0.001)</td><td>-0.05 (0.58)</td></tr></table>

![](images/701852fa454a0a23ece752dfa837a320c1a116f7a6285cff243318ed28b5059d.jpg)  
Fig. 4: Empirical complementary CDFs of the n-gram type frequencies (solid, with markers) and fitted power laws above $x _ { \mathrm { m i n } }$ (dashed) for one representative codec per meta-category on clean LJSpeech at the 500K-token reference; larger KS corresponds to larger visible departure from the fitted line.

## B. Entropy, Redundancy, and Perplexity

Table VII reports the unigram Shannon entropy, redundancy, and token perplexity of Section III-B, computed on the deduplicated token streams at the 500K-token reference. The common unigram order keeps the values comparable across codecs.

On clean speech, entropy tracks vocabulary scale and redundancy is uniformly low. Entropy is positively associated with the available vocabulary scale in this codec set. Every codec uses its observed inventory close to uniformly, with the single-codebook VQ group and FocalCodec lowest. These values sit far below classical redundancy estimates for naturallanguage text $( R \approx 0 . 5$ for printed English), which include the sequential dependencies that a unigram measure ignores [31]. The token perplexity $2 ^ { H }$ varies by three orders of magnitude across the codec set at a matched data scale.

White noise removes entropy; DEMAND noise barely changes it. About 1–4.4 bits vanish under white noise for every codec except S3Tokenizer and SQCodec, whose entropy changes by less than 0.5 bits. Probability mass concentrates on a smaller effective inventory, raising the grid-mean redundancy from 0.08 on clean speech to 0.27. DEMAND noise changes entropy comparatively little.

## C. Clean-to-Noisy Distributional Shifts

For each codec, corpus, and distributional metric m, the α, KS, β, K, H, and R of Section III-B, together with the sequence diagnostics of Section VIII-A, we calculate the shift from clean speech to 0 dB noise as $\Delta m = m _ { 0 } \mathrm { { d B } } - m _ { \mathrm { { c l e a } } }$ <sub>n</sub>. Figure 6 reports the codec-level shifts averaged over the three corpora. The median absolute clean-to-0 dB change in α is 0.35 under white noise and 0.23 under DEMAND noise, with the most affected cells shifting by up to 1.91.

TABLE VI: Distributional parameters of the 13 codecs at the 500K-token reference: the Zipf exponent α with its KS fit distance, and the Heaps exponent $\beta$ with its intercept K. Values are averaged over the $N _ { \mathrm { { f i t } } }$ corpora with valid fits, use the codec-specific analysis orders under the fitting safeguards of Section III-B, and are reported descriptively.
<table><tr><td>Codec</td><td>Meta-cat.</td><td> $N _ { \mathrm { f i t } }$ </td><td>α</td><td>KS</td><td>β</td><td>K</td></tr><tr><td>Mimi</td><td>RVQ</td><td>2</td><td>1.97</td><td>0.021</td><td>0.75</td><td>7.8</td></tr><tr><td>SpeechTokenizer</td><td>RVQ</td><td>3</td><td>2.01</td><td>0.014</td><td>0.80</td><td>4.0</td></tr><tr><td>AudioDec</td><td>RVQ</td><td>3</td><td>2.36</td><td>0.005</td><td>0.91</td><td>2.0</td></tr><tr><td>EnCodec</td><td>RVQ</td><td>3</td><td>2.50</td><td>0.005</td><td>0.92</td><td>1.9</td></tr><tr><td>FunCodec</td><td>RVQ</td><td>3</td><td>2.39</td><td>0.011</td><td>0.96</td><td>1.3</td></tr><tr><td>AcademiCodec</td><td>RVQ</td><td>3</td><td>2.66</td><td>0.009</td><td>0.84</td><td>3.8</td></tr><tr><td>DAC-24k</td><td>RVQ</td><td>3</td><td>2.72</td><td>0.010</td><td>0.97</td><td>1.3</td></tr><tr><td>XCodec2</td><td>Single VQ</td><td>3</td><td>2.96</td><td>0.050</td><td>0.57</td><td>41.7</td></tr><tr><td>BigCodec</td><td>Single VQ</td><td>2</td><td>2.88</td><td>0.179</td><td>0.98</td><td>1.2</td></tr><tr><td>WavTokenizer</td><td>Single VQ</td><td>1</td><td>4.48</td><td>0.002</td><td>0.97</td><td>1.3</td></tr><tr><td>FocalCodec</td><td>Non-VQ</td><td>3</td><td>2.97</td><td>0.093</td><td>0.27</td><td>286.9</td></tr><tr><td>SQCodec</td><td>Non-VQ</td><td>3</td><td>2.81</td><td>0.083</td><td>0.69</td><td>18.5</td></tr><tr><td>S3Tokenizer</td><td>Non-VQ</td><td>3</td><td>2.41</td><td>0.004</td><td>0.91</td><td>2.0</td></tr></table>

![](images/dcffc266e9db7d32f48d9613059187a56afab60cf3a2471021723473d55243f1.jpg)  
Fig. 5: Clean-speech Zipf exponent α and Heaps exponent β for 13 codecs on LJSpeech, VoxCeleb, and TrIJEK at the 500K-token reference scale. Colors denote the architecture families of Table I. Cells without a defined power-law fit $( V / N > 0 . 9 )$ are omitted.

White noise depresses α most strongly for the singlecodebook VQ codecs. Among the six single-codebook cells with valid clean and white-noise fits, the group mean is $\overline { { \Delta \alpha } } =$ −1.02, compared with −0.36 for the multi-codebook RVQ cells (20 cells) and 0.00 for the non-VQ cells (nine cells). DEMAND-noise shifts are smaller on average and vary more across codecs and corpora than white-noise shifts.

No single noise-robustness ranking of codecs exists: cross-noise agreement is metric-dependent. To test whether the two noise types stress the same codecs in the same order, we compare, for each metric, the codec rankings induced under white and under DEMAND noise. UER-based codec rankings are similar across the two noise types (Spearman $\rho = 0 . 8 1$ $p = 0 . 0 0 1$ across the 13 codecs). Occupancy-shift rankings show moderate agreement $( \rho = 0 . 6 5 , p = 0 . 0 1 5 ,$ , 13 codecs), while α-shift rankings show no agreement $( \rho ~ = ~ - 0 . 0 4$ $p = 0 . 9 0$ , the 12 codecs with valid fits). Therefore, robustness rankings are metric-dependent: token-identity disruption (UER) is consistent across noise types, whereas distributionshape sensitivity (∆α) is noise-type-specific.

TABLE VII: Unigram Shannon entropy H (bits), redundancy $R ,$ and token perplexity $2 ^ { H }$ on clean speech, with the change in H from clean speech to each 0 dB condition. Values are computed on deduplicated streams at the 500K-token reference and averaged over the corpora with a matched reference (the Mimi–TrIJEK cells are excluded).
<table><tr><td>Codec</td><td>H</td><td>R</td><td> $2 ^ { H }$ </td><td> $\Delta H _ { \mathrm { W h i t e } }$ </td><td>∆HDEMAND</td></tr><tr><td>Mimi</td><td>10.2</td><td>0.058</td><td>1,232</td><td>-2.67</td><td>-0.31</td></tr><tr><td>SpeechTokenizer</td><td>9.3</td><td>0.063</td><td>654</td><td>-4.14</td><td>-1.54</td></tr><tr><td>AudioDec</td><td>7.4</td><td>0.237</td><td>172</td><td>-4.35</td><td>-0.53</td></tr><tr><td>EnCodec</td><td>8.0</td><td>0.162</td><td>250</td><td>-2.90</td><td>+0.26</td></tr><tr><td>FunCodec</td><td>9.3</td><td>0.073</td><td>619</td><td>-4.40</td><td>-0.66</td></tr><tr><td>AcademiCodec</td><td>6.3</td><td>0.120</td><td>82</td><td>-0.99</td><td>-0.26</td></tr><tr><td>DAC-24k</td><td>8.9</td><td>0.108</td><td>486</td><td>-2.12</td><td>+0.45</td></tr><tr><td>XCodec2</td><td>15.2</td><td>0.043</td><td>36,830</td><td>-3.33</td><td>-0.87</td></tr><tr><td>BigCodec</td><td>12.7</td><td>0.023</td><td>6,648</td><td>-3.48</td><td>-0.61</td></tr><tr><td>WavTokenizer</td><td>11.5</td><td>0.041</td><td>2,883</td><td>-3.47</td><td>-1.04</td></tr><tr><td>FocalCodec</td><td>12.6</td><td>0.029</td><td>6,309</td><td>-1.09</td><td>-0.19</td></tr><tr><td>SQCodec</td><td>16.3</td><td>0.027</td><td>83,077</td><td>-0.08</td><td>+0.05</td></tr><tr><td>S3Tokenizer</td><td>9.3</td><td>0.049</td><td>634</td><td>-0.47</td><td>+0.15</td></tr></table>

## D. Variance Decomposition

We fit a full-factorial three-way ANOVA using one 100Ktoken aggregate per valid codec–corpus–condition cell of the 117-cell grid. Because validity filtering yields an unbalanced design, effect sizes are computed from Type-II sums of squares. Table VIII reports eta-squared values, where $\eta ^ { 2 }$ is the fraction of total variance associated with each term, together with the number of valid cells per metric.

The dominant factor differs across metrics. Acoustic condition is the largest main effect for $\alpha , \beta ,$ and the redundancy R. Meta-category leads for KS, the Heaps intercept K, and most strongly for the unigram entropy $H ,$ , which is the most architecture-determined metric in the panel. Corpus identity explains little variation in the evaluated three-corpus grid.

The meta×condition interaction accounts for an appreciable variance fraction for $\alpha , \ K S ,$ and $R ,$ indicating that the effect of acoustic condition differs across quantizer metacategories; Section VIII characterizes these family-specific responses as sequence-level degradation signatures. A large residual component remains for all metrics. The reported values therefore describe variation associated with the grouped factors; differences among individual codecs within a metacategory are absorbed by the residual.

The noise effects survive a codec random intercept. Because the cells repeatedly sample the same 13 codecs, we refit each metric with a linear mixed model comprising the meta×condition and corpus fixed effects and a codec random intercept, which absorbs a large variance share (intraclass correlation 0.46–0.96). Under this model, the joint noise effect—condition together with its meta-category interaction— remains significant for every metric (Wald $p \leq 0 . 0 0 1 )$ , as does the meta×condition interaction alone $( p \leq 0 . 0 0 1 )$ . The metacategory contrast evaluated on clean speech remains significant for α, H, and R. The corpus terms reach significance only for $\beta ,$ with a test statistic well below the noise terms.

![](images/3cba12fe0782622eb8e4831baacb548cfac76f1769aaffd6a499289bfaad9e87.jpg)  
Fig. 6: Per-codec clean-to-noise shifts in the Zipf exponent, KS distance, codebook occupancy, and transition entropy (Section VIII-A) under the two 0 dB conditions (solid: white; hatched: DEMAND). Bars show means across the three corpora and error bars the standard deviation; colors denote the quantizer meta-category, and dashed vertical lines separate the meta categories.

TABLE VIII: Type-II eta-squared values from the full-factorial three-way ANOVA. $N _ { \mathrm { c e l l s } }$ is the number of valid cells per metric. MC denotes the Meta-Codec quantizer category, Crp. the corpus, and Cnd. the acoustic condition; the tabulated terms partition the total variance, so each row sums to one.
<table><tr><td>Metric</td><td> $N _ { \mathrm { c e l l s } }$ </td><td>MC</td><td> $\mathbf { C r p . }$ </td><td>Cnd.</td><td>MC ×Cnd.</td><td>MC ×Crp.</td><td> $\mathbf { C r p . }$  ×Cnd.</td><td> $\begin{array} { c } { { \bf M C } \times { \bf C r p } . } \\ { { \bf \nabla \times C n d . } } \end{array}$ </td><td>Resid.</td></tr><tr><td>α</td><td>100</td><td>0.134</td><td>0.003</td><td>0.278</td><td>0.088</td><td>0.020</td><td>0.015</td><td>0.014</td><td>0.449</td></tr><tr><td>KS</td><td>100</td><td>0.158</td><td>0.003</td><td>0.016</td><td>0.096</td><td>0.012</td><td>0.002</td><td>0.009</td><td>0.704</td></tr><tr><td>β</td><td>117</td><td>0.086</td><td>0.005</td><td>0.120</td><td>0.039</td><td>0.002</td><td>0.002</td><td>0.001</td><td>0.745</td></tr><tr><td>K</td><td>117</td><td>0.240</td><td>0.001</td><td>0.000</td><td>0.016</td><td>0.000</td><td>0.001</td><td>0.000</td><td>0.742</td></tr><tr><td>H</td><td>117</td><td>0.551</td><td>0.000</td><td>0.126</td><td>0.023</td><td>0.000</td><td>0.001</td><td>0.000</td><td>0.299</td></tr><tr><td>R</td><td>117</td><td>0.320</td><td>0.002</td><td>0.351</td><td>0.080</td><td>0.001</td><td>0.004</td><td>0.001</td><td>0.241</td></tr></table>

## VII. DISTRIBUTIONAL STRUCTURE AND QUALITY

## A. Clean-Speech Quality Overview

Table IX summarizes the quality metrics of the 12 decodable codecs, averaged over the three clean corpora. S3Tokenizer does not provide reconstructed audio, so its quality metrics are unavailable.

Clean-speech quality sharply separates the codec set. The clean-speech dCER draws the sharpest line: every codec except WavTokenizer preserves intelligibility with dCER of at most 0.06, whereas WavTokenizer reaches 0.21. This degradation appears at the transcription level rather than in the framelevel acoustic distances: its MCD and F0 errors stay within the range of the remaining codecs.

TABLE IX: Clean-speech resynthesis quality for the 12 decodable codecs, averaged over the three clean corpora. The first row gives the UTMOS of the reference recordings themselves; their reference-based metrics are zero by definition. S3Tokenizer is excluded because it provides no decoder; UER is omitted because it is defined only under noisy conditions.
<table><tr><td>Codec</td><td>UTMOS↑</td><td>MCD↓ (dB)</td><td>F0RMSE↓ (Hz)</td><td>F0MedAE↓ (Hz)</td><td>dCER↓</td></tr><tr><td>Reference</td><td>3.09</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>Mimi</td><td>3.05</td><td>3.1</td><td>15.7</td><td>0.8</td><td>0.04</td></tr><tr><td>SpeechTokenizer</td><td>2.96</td><td>3.9</td><td>21.9</td><td>1.3</td><td>0.04</td></tr><tr><td>AudioDec</td><td>2.34</td><td>4.0</td><td>25.2</td><td>2.2</td><td>0.04</td></tr><tr><td>EnCodec</td><td>2.26</td><td>3.2</td><td>19.0</td><td>0.8</td><td>0.02</td></tr><tr><td>FunCodec</td><td>2.93</td><td>3.0</td><td>24.3</td><td>1.1</td><td>0.02</td></tr><tr><td>AcademiCodec</td><td>2.94</td><td>3.7</td><td>24.7</td><td>1.0</td><td>0.04</td></tr><tr><td>DAC-24k</td><td>3.06</td><td>1.3</td><td>7.0</td><td>0.1</td><td>0.01</td></tr><tr><td>XCodec2</td><td>3.32</td><td>3.9</td><td>23.8</td><td>1.2</td><td>0.04</td></tr><tr><td>BigCodec</td><td>3.32</td><td>3.6</td><td>23.2</td><td>1.0</td><td>0.06</td></tr><tr><td>WavTokenizer</td><td>2.74</td><td>5.0</td><td>24.3</td><td>1.5</td><td>0.21</td></tr><tr><td>FocalCodec</td><td>3.33</td><td>5.1</td><td>28.3</td><td>3.6</td><td>0.05</td></tr><tr><td>SQCodec</td><td>3.08</td><td>3.5</td><td>21.0</td><td>0.6</td><td>0.02</td></tr></table>

Faithful decoders are bounded by their reference; regenerative decoders exceed it. The remaining spread in Table IX follows from bitrate and decoder design. For example, DAC-24k’s resynthesis is close to transparent in terms of MCD and F0 error, while its per-corpus UTMOS matches that of the reference recordings to within 0.07. Therefore, its referencefidelity metrics are the best in the table, while its UTMOS is bounded by the naturalness of the references themselves. XCodec2 and BigCodec instead exceed the reference UTMOS on TrIJEK and VoxCeleb by up to 0.5. This shows that their decoders regenerate speech toward the clean characteristics of their training data, thereby raising naturalness while increasing the reference-based spectral distance. Figure 7 places the adopted configurations of the four RVQ codecs with an inference-time bandwidth control on their clean-speech quality–bitrate curves. The first-level token streams are identical across these configurations, so all distributional analyses are unaffected by this choice.

![](images/7e9c7f896f9174c3e14acab6a3e64cde675a6900d861bb5aed289b1c63c913f3.jpg)

![](images/bef65ac3110732e76d74366d100a34612002e4a31177fdd22aa575808c1b6b9d.jpg)  
Fig. 7: Clean-speech UTMOS and MCD versus bitrate for the four RVQ codecs whose bandwidth is an inference-time choice. Stars mark the configurations evaluated in this paper, and the dashed line gives the UTMOS of the reference.

## B. JSD and Quality under Noise

We measure distributional degradation using the JSD between equal-sized clean and noisy token samples of each codec–corpus cell, with the sample size fixed to the largest value available for every evaluated cell (≈423K tokens). JSD is computed in token space and requires a clean reference distribution, but it does not require waveform reconstruction. Because JSD approaches its upper bound when the two samples share little n-gram support, we compute it at the common unigram order (n = 1) for every codec—deliberately departing from the codec-specific analysis orders of Section V-B: under those orders JSD saturates above 0.9 for most higher-order codecs and leaves little cross-codec variation, and the unigram order is the only order valid across the full grid.

Table X reports Pearson and Spearman correlations between JSD and the quality metrics measured under the corresponding noisy condition, using the twelve decodable codecs after averaging each codec–noise cell across the three corpora. Figure 8 shows the corresponding JSD–UTMOS scatter under the two noisy conditions.

JSD tracks perceived quality most clearly under DE-MAND noise. The association between JSD and quality depends on noise type and on the statistic used. Under DEMAND noise, cells with a larger distributional shift have lower UTMOS and a larger median F0 error, whereas the MCD association is weak once the level correction of Section III-C is applied. The UTMOS association survives removing the two non-VQ codecs that occupy the low-JSD end $( r \ = \ - 0 . 6 8 .$ $p = 0 . 0 2 9 )$ , and in the scatter of Figure 8 no single codec moves the coefficient by more than 0.10. Under white noise no association with the audio-based quality metrics survives both statistics: the MCD coefficient holds only in Pearson terms and reverses in sign $( r = - 0 . 2 7 )$ without the two non-VQ codecs, reflecting the gap between those codecs and the rest rather than a graded relation. The dCER associations are weak throughout, and the negative UER association, although present under white noise in both statistics, vanishes under DEMAND noise.

![](images/04b35cd37d5cd26442990f85c763bf636afc0c1c2ee45168f934c72e394dcf50.jpg)

![](images/cee3e6379e01782371c7618ebfd5277696e2859edc1e917c7412d3a8561691fa.jpg)  
Fig. 8: Clean-to-noise JSD at the common unigram order versus UTMOS under the corresponding noisy condition, for the twelve decodable codecs averaged across the three corpora. Dashed lines are least-squares fits; r is the Pearson correlation reported in Table X.

TABLE X: Pearson (r) and Spearman (ρ) correlations between clean-to-noise JSD (common unigram order, equal-sized samples) and quality metrics under the corresponding noisy condition. Cells are the twelve decodable codecs averaged across the three corpora (12 codecs per condition). Bold values indicate a magnitude above 0.5.
<table><tr><td>Condition</td><td>Stat.</td><td>UTMOS</td><td>MCD</td><td>F0MedAE</td><td>UER</td><td>dCER</td></tr><tr><td>White-0 dB</td><td>r</td><td>0.15</td><td>0.62</td><td>0.19</td><td>-0.56</td><td>-0.01</td></tr><tr><td></td><td>ρ</td><td>0.06</td><td>0.06</td><td>-0.17</td><td>-0.73</td><td>-0.50</td></tr><tr><td>DEMAND-0 dB</td><td>r</td><td>-0.76</td><td>0.38</td><td>0.57</td><td>-0.03</td><td>-0.14</td></tr><tr><td></td><td>ρ</td><td>-0.65</td><td>0.49</td><td>0.34</td><td>-0.23</td><td>-0.25</td></tr></table>

Distribution-level shift and token-level integrity capture different aspects of degradation. The UER cell means span 0.66–1.00, so the weak JSD–UER association under DEMAND noise is partly attributable to the uniformly high token-level degradation at 0 dB.

## VIII. ARCHITECTURE-DEPENDENT DEGRADATION SIGNATURES

We analyze whether noise produces different sequence-level degradation patterns across quantizer architectures.

## A. Operational Definitions

We characterize each clean-to-noise shift using two metrics: repetition rate and transition entropy. We define repetition rate $r ,$ which is the fraction of adjacent positions that contain the same token before deduplication. We also define transition entropy $H _ { \mathrm { t r a n s } } = H ( t _ { i } \mid t _ { i - 1 } )$ , which measures the uncertainty of the next token under the empirical bigram distribution.

For each codec, corpus, and noise type, we calculate $\Delta r$ and $\Delta H _ { \mathrm { t r a n s } }$ between clean speech and the corresponding 0 dB condition. We define collapse as $\Delta r > 0 . 0 1$ and $\Delta H _ { \mathrm { t r a n s } } <$ −0.1, and explosion as $\Delta r ~ < ~ - 0 . 0 1$ and $\Delta H _ { \mathrm { t r a n s } } ~ > ~ 0 . 1$ All remaining cells are classified as neutral. These deadband thresholds are operational choices that exclude near-zero shifts. Both diagnostics are computed on the original streams before consecutive deduplication.

## B. Shifts by Quantizer Architecture

Figure 9 complements the view of Figure 6 by placing each codec–noise pair on the signature plane spanned by the repetition-rate and transition-entropy shifts. Table XI summarizes the group-mean shifts.

White noise induces collapse, and DEMAND noise explosion, within the same RVQ family. White noise produces a collapse-like pattern in the multi-codebook RVQ group: repetition rises while transition entropy and codebook occupancy drop sharply, mirroring the unigram entropy loss reported in Section VI-B. The same group responds to DEMAND noise in the opposite direction, with decreasing repetition, increasing transition entropy, and little average change in occupancy. In the signature plane of Fig. 9, the RVQ–white points accordingly cluster in the collapse quadrant, and the RVQ–DEMAND points in the explosion quadrant.

Single-codebook VQ codecs deform without either signature. They show little change in repetition rate under either noise type, remaining near the $\Delta r = 0$ axis of the signature plane, while losing 33–75 percentage points of occupancy per cell under white noise. These changes do not satisfy the collapse or explosion criteria. Their observed shifts appear in occupancy and distribution-shape parameters rather than in repetition rate.

The non-VQ group does not show a collapse signature under either noise type. Transition entropy generally increases while repetition-rate changes remain small, but the magnitude differs sharply within the group. S3Tokenizer crosses the explosion deadband in four of its six cells, whereas FocalCodec and SQCodec change too little to leave the neutral region.

## C. Distribution of Degradation Signatures

Collapse concentrates in RVQ–white and explosion in RVQ–DEMAND, robustly to the deadband choice. Table XI reports the classification of the 78 codec–corpus–noise cells under the deadband criteria of Section VIII-A. Collapse is concentrated in the RVQ–white condition, which accounts for 14 of the 15 collapse instances in the full analysis. Explosion is concentrated in the RVQ–DEMAND condition, consistent with the group-average increase in transition entropy under DEMAND noise. The single RVQ–white explosion cell belongs to DAC-24k on TrIJEK. Halving both thresholds changes the RVQ–white collapse and RVQ–DEMAND explosion counts to 15 and 15 of the 21 cells in each group, while doubling them yields 14 and 11.

All 18 SingleVQ cells are classified as neutral. The non-VQ group contributes a further four explosion cells, all belonging to S3Tokenizer. The semantic tokenizer satisfies the explosion criterion in all three corpora under DEMAND noise and in one under white noise, whereas FocalCodec and SQCodec are neutral throughout.

![](images/0b288b8c285a31ec045f336d18247a8e68e913d06dd48e92729057b3b019643e.jpg)  
Fig. 9: Per-codec shifts from clean speech to the 0 dB conditions in repetition rate and transition entropy, averaged across the three corpora. Colors denote quantizer meta-category and markers the noise type; shaded quadrants indicate the collapse and explosion signatures.

TABLE XI: Degradation-signature classification of the 78 codec–corpus–noise cells under the deadband criteria of Section VIII-A. Col./Exp./Neu.: number of cells classified as collapse, explosion, and neutral; $\overline { { \Delta r } } , \overline { { \Delta H _ { \mathrm { t r a n s } } } }$ , ∆Occ.: groupmean clean-to-noise shifts in repetition rate, transition entropy, and codebook occupancy (percentage points).
<table><tr><td>Meta-cat.</td><td>Noise</td><td>Col.</td><td> $\mathbf { E x p . }$ </td><td>Neu.</td><td> $\overline { { \Delta r } }$ </td><td> $\overline { { \Delta H _ { \mathrm { t r a n s } } } }$ </td><td> ${ \overline { { \Delta 0 \mathrm { c c } } } } .$ </td></tr><tr><td>RVQ</td><td>White</td><td>14</td><td>1</td><td>6</td><td>+0.08</td><td>-1.31</td><td>-40</td></tr><tr><td>RVQ</td><td>DEMAND</td><td>1</td><td>13</td><td>7</td><td>-0.03</td><td>+0.63</td><td>+2</td></tr><tr><td>SingleVQ</td><td>White</td><td>0</td><td>0</td><td>9</td><td>+0.00</td><td>+0.74</td><td>-57</td></tr><tr><td>SingleVQ</td><td>DEMAND</td><td>0</td><td>0</td><td>9</td><td>+0.00</td><td>+0.56</td><td>-9</td></tr><tr><td>NonVQ</td><td>White</td><td>0</td><td>1</td><td>8</td><td>-0.00</td><td>+0.34</td><td>-3</td></tr><tr><td>NonVQ</td><td>DEMAND</td><td>0</td><td>3</td><td>6</td><td>-0.01</td><td>+0.27</td><td>+1</td></tr><tr><td>Total</td><td></td><td>15</td><td>18</td><td>45</td><td></td><td></td><td></td></tr></table>

## IX. DISCUSSION

## A. Architecture and Condition as the Dominant Distributional Factors

A consistent thread across our results is that quantizer architecture exerts a far larger influence on token-level statistics than any of the conventional sources of corpus variation we tested, with corpus identity contributing negligibly across all distributional parameters (Section VI-D). The Zipf exponent α spans from approximately 2.0 to above 4 on clean speech across the 13-codec set (Section VI-A), a range larger than the typical clean→0 dB shift of individual codecs. At the same time, the three-way ANOVA (Section VI-D) shows that architecture does not dominate uniformly. Meta-category is the leading main effect for the fit quality KS, the Heaps intercept K, and most strongly for the unigram entropy H. Acoustic condition dominates $\alpha , \beta ,$ , and the redundancy R at the pooled level. The meta×condition interaction remains significant for every metric once codec identity is modeled as a random effect. This pattern motivates a methodological shift relative to [9], where a single corpus was sufficient to characterize a codec. In our expanded grid, any cross-codec comparison must be conducted at fixed meta-category, n-gram order, and where possible vocabulary scale, because between-family contrasts are confounded with all three.

A practical implication is that the natural-language analogy that motivated the original Zipf analysis of NAC tokens— namely, that an α near the word-level reference value of 2 reflects language-like statistical structure—should be interpreted family-conditionally rather than as a universal target. Throughout this paper, language-like refers to the qualitative statistical structure that codec tokens share with text— a well-fitting power-law frequency distribution and sublinear Heaps-type vocabulary growth that persist across corpora—not to quantitative agreement with word-level parameter values, which the differences in unit granularity and vocabulary size preclude. Under this reading, a codec is language-like to the extent that its fits satisfy the validity and goodness-offit criteria of Section III-B across corpora, and every codec in the panel meets this criterion on clean speech.

## B. JSD as a Complementary Distributional Diagnostic

Among the distributional diagnostics we considered, JSD shows the clearest association with acoustic quality under reallife DEMAND noise, where it tracks UTMOS and the median F0 error in the expected directions (Section VII-B). JSD operates on the full token frequency distribution, so probabilitymass redistribution that preserves the overall power-law shape but alters individual token frequencies—a common consequence of noise—registers in JSD but not in α or KS. Because JSD is computed from token-level counts and requires no waveform reconstruction, it can also be evaluated for nondecodable tokenizers such as S3Tokenizer, for which audiobased quality metrics are unavailable.

## C. Codebook Size and the Distributional Regime

The multi-codebook RVQ codecs, whose per-level codebooks hold 1,024 or 2,048 entries, occupy the low-α end of the spectrum. Every waveform-reconstructing design built on a single codebook of 4,096 entries or more sits above 2.8, with WavTokenizer highest at 4.48. The semantic tokenizer S3Tokenizer, whose codebook is trained for recognition rather than reconstruction, sits lower at 2.41 despite its 4,096 entries. This ordering is consistent with a sampling argument: as |V| grows at a fixed corpus size, the frequency profile is increasingly dominated by a small set of acoustically frequent units, with a long tail of rarely used codewords that steepens the fitted exponent. The degradation-signature analysis (Section VIII) is consistent with the same axis: single-codebook VQ codecs respond to noise predominantly via probabilitymass redistribution within the codebook (large ∆Occupancy, small $\Delta r )$ , while multi-codebook RVQ codecs respond via local sequence dynamics (large $\Delta r ,$ large $\Delta H _ { \mathrm { t r a n s } } )$ .

## D. Limitations

Several limitations should be considered when generalizing our findings. First, cross-codec comparisons are mediated by family-conditional n-gram orders, so absolute parameter values are not directly comparable across families (Section V-B).

Second, the noise design is restricted to two source categories at a single 0 dB SNR, leaving other SNRs, reverberation, and channel distortions unexplored. Third, the quality metrics rely on automatic estimators (Whisper-large-v3, UTMOS) whose biases may correlate with codec architecture. Fourth, S3Tokenizer lacks a decoder, so the JSD–quality analyses of Section VII-B cover 12 of the 13 codecs. Fifth, the ANOVA cells repeatedly sample the same 13 codecs, so the reported $\eta ^ { 2 }$ values are descriptive partitions and the mixed-model metacategory tests rest on 13 codec-level units. Sixth, the JSD– quality correlations are computed over twelve codec-level means, a sample size at which the choice of statistic matters (Section VII-B). Seventh, cross-codec JSD requires a matched low n-gram order because the measure saturates when the compared samples share little n-gram support. Eighth, vocabulary size is confounded with architecture, bitrate, and training objective, so the codebook-size patterns of Section IX-C are associations rather than isolated effects. Ninth, the collapse and explosion counts depend on the operational deadband thresholds, within the two-fold sensitivity range reported in Section VIII-C. Finally, we treat each codec as a fixed blackbox pipeline, leaving the relation between training objective or data composition and the measured distributional regime to future work.

## X. CONCLUSION

We presented a language-statistical analysis of 13 neural audio codecs spanning multi-codebook RVQ, single-codebook VQ, and non-VQ designs, evaluated on three corpora and three acoustic conditions in a fully crossed 117-cell grid.

The analysis supports three principal conclusions. First, corpus identity is a negligible source of distributional variance: acoustic condition and quantizer architecture dominate in a metric-dependent way. The significant meta–condition interaction indicates family-specific noise responses. Second, JSD computed at a common unigram order from equal-sized token samples provides a decoder-free distribution-level diagnostic. It tracks perceived quality across codecs under real-world DEMAND noise, whereas none of its white-noise associations survives a rank-based check. Third, the collapse and explosion degradation signatures previously identified for RVQ codecs do not generalize uniformly. Collapse concentrates in RVQ×white-noise cells, and explosion in RVQ×DEMAND cells and the semantic tokenizer. Single-codebook VQ codecs deform along occupancy and α without either signature.

Beyond these findings, the paper contributes an architectureconditioned protocol for applying language-statistical analysis to codec tokens: family-conditional n-gram orders with explicit fit-validity safeguards, matched-sample chunk-based reliability assessment, and variance decomposition across metacategory, corpus, and condition. This protocol makes the natural-language analogy underlying prior Zipf analyses of codec tokens usable as a family-conditional convention rather than a universal target.

Future work should broaden the noise design beyond the 0 dB clean→noisy contrast, to continuous SNR sweeps and to non-stationary distortions including reverberation and channel effects. The signature analysis should also extend to emerging codec families, in particular large-vocabulary and semantic-prior designs whose distributional behavior under noise remains underexplored. A further direction is to connect the distributional regimes characterized here to the training objective and data composition of each codec. We hope that the metric panel, reliability protocol, and analysis conventions established provide a useful template for such studies.

Acknowledgment: The work was supported by JSPS KAK-ENHI Grant Number 26KJ0771, JST Moonshot Grant Number JPMJMS2011, and based on results from a project outsourced by the New Energy and Industrial Technology Development Organization (NEDO), and the BRIDGE Program (R7-H05), implemented by the Cabinet Office, Government of Japan.

## REFERENCES

[1] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “SoundStream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2022.

[2] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural´ audio compression,” Transactions on Machine Learning Research, 2023.

[3] X. Zhang, D. Zhang, S. Li, Y. Zhou, and X. Qiu, “SpeechTokenizer: Unified speech tokenizer for speech large language models,” in Proc. ICLR, 2024.

[4] C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu et al., “Neural codec language models are zero-shot text to speech synthesizers,” arXiv preprint arXiv:2301.02111, 2023.

[5] K. Dhawan, N. R. Koluguri, A. Jukic, R. Langman, J. Balam, and´ B. Ginsburg, “Codec-ASR: Training performant automatic speech recognition systems with discrete speech representations,” in Proc. INTER-SPEECH, 2024.

[6] Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi et al., “AudioLM: A language modeling approach to audio generation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2523–2533, 2023.

[7] G. K. Zipf, Human Behavior and the Principle of Least Effort. Addison-Wesley, 1949.

[8] H. S. Heaps, Information Retrieval: Computational and Theoretical Aspects. Academic Press, 1978.

[9] J. Park, S. Takamichi, D. M. Chan, S. Kando, Y. Saito, and H. Saruwatari, “Analysing the language of neural audio codecs,” in Proc. ASRU, 2025.

[10] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, “UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022,” in Proc. INTERSPEECH, 2022.

[11] J. Thiemann, N. Ito, and E. Vincent, “The diverse environments multichannel acoustic noise database (DEMAND): A database of multichannel environmental noise recordings,” in Proc. Meetings on Acoustics, 2013.

[12] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021.

[13] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Proc. NeurIPS, 2020.

[14] A. Mohamed, H.-y. Lee, L. Borgholt, J. D. Havtorn, J. Edin, C. Igel, K. Kirchhoff, S.-W. Li, K. Livescu, L. Maaløe, T. N. Sainath, and S. Watanabe, “Self-supervised speech representation learning: A review,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1179–1210, 2022.

[15] D. Yang, S. Liu, R. Huang, J. Tian, C. Weng, and Y. Zou, “HiFi-Codec: Group-residual vector quantization for high fidelity audio codec,” arXiv preprint arXiv:2305.02765, 2023.

[16] Y.-C. Wu, I. D. Gebru, D. Markovic, and A. Richard, “AudioDec:´ An open-source streaming high-fidelity neural audio codec,” in Proc. ICASSP, 2023.

[17] Z. Du, S. Zhang, K. Hu, and S. Zheng, “FunCodec: A fundamental, reproducible and integrable open-source toolkit for neural speech codec,” in Proc. ICASSP, 2024.

[18] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “Highfidelity audio compression with improved RVQGAN,” in Proc. NeurIPS, 2023.

[19] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in Neural Information Processing Systems (NeurIPS), 2014.

[20] S. Ji, Z. Jiang, W. Wang, Y. Chen, M. Fang, J. Zuo et al., “WavTokenizer: An efficient acoustic discrete codec tokenizer for audio language modeling,” in Proc. ICLR, 2025.

[21] D. Xin, X. Tan, S. Takamichi, and H. Saruwatari, “BigCodec: Pushing the limits of low-bitrate neural speech codec,” arXiv preprint arXiv:2409.05377, 2024.

[22] Z. Ye, X. Zhu, C.-M. Chan, X. Wang, X. Tan et al., “Llasa: Scaling traintime and inference-time compute for Llama-based speech synthesis,” arXiv preprint arXiv:2502.04128, 2025.

[23] L. Zhai, H. Ding, C. Zhao, F. Wang, G. Wang, Z. Wang, and W. Xi, “L3AC: Towards a lightweight and lossless audio codec,” arXiv preprint arXiv:2504.04949, 2025.

[24] A. Defossez, L. Mazar ´ e, M. Orsini, A. Royer, P. P ´ erez, H. J ´ egou,´ E. Grave, and N. Zeghidour, “Moshi: A speech-text foundation model for real-time dialogue,” arXiv preprint arXiv:2410.00037, 2024.

[25] L. Della Libera, F. Paissan, C. Subakan, and M. Ravanelli, “Focal-Codec: Low-bitrate speech coding via focal modulation networks,” arXiv preprint arXiv:2502.04465, 2025.

[26] Z. Du, Q. Chen, S. Zhang, K. Hu, H. Lu et al., “CosyVoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens,” arXiv preprint arXiv:2407.05407, 2024.

[27] Y. Guo, Z. Li, H. Wang, B. Li, C. Shao, H. Zhang, C. Du, X. Chen, S. Liu, and K. Yu, “Recent advances in discrete speech tokens: A review,” arXiv preprint arXiv:2502.06490, 2025.

[28] M. E. J. Newman, “Power laws, Pareto distributions and Zipf’s law,” Contemporary Physics, vol. 46, no. 5, pp. 323–351, 2005.

[29] B. Mandelbrot, “Contribution a la th \` eorie math ´ ematique des jeux de´ communication,” Annales de l’ISUP, vol. 2, pp. 3–124, 1953.

[30] A. Gelbukh and G. Sidorov, “Zipf and heaps laws’ coefficients depend on language,” in Proc. CICLing, 2001, pp. 332–335.

[31] C. E. Shannon, “Prediction and entropy of printed English,” Bell System Technical Journal, vol. 30, no. 1, pp. 50–64, 1951.

[32] A. Clauset, C. R. Shalizi, and M. E. J. Newman, “Power-law distributions in empirical data,” SIAM Review, vol. 51, no. 4, pp. 661–703, 2009.

[33] D. M. Chan, R. Corona, J. Park, C. J. Cho, Y. Bai, and T. Darrell, “Analyzing the language of visual tokens,” arXiv preprint arXiv:2411.05001, 2024.

[34] S. Takamichi, H. Maeda, J. Park, D. Saito, and H. Saruwatari, “Do learned speech symbols follow Zipf’s law?” in Proc. ICASSP, 2024.

[35] A. Sicherman and Y. Adi, “Analysing discrete self supervised speech representation for spoken language modeling,” in Proc. ICASSP, 2023.

[36] T. Ashihara, M. Delcroix, T. Ochiai, K. Matsuura, and S. Horiguchi, “Analysis of semantic and acoustic token variability across speech, music, and audio domains,” in Proc. Interspeech, 2025, pp. 226–230.

[37] F. J. Massey, “The Kolmogorov-Smirnov test for goodness of fit,” Journal of the American Statistical Association, vol. 46, no. 253, pp. 68–78, 1951.

[38] H. Wu, H.-L. Chung, Y.-C. Lin, Y.-K. Wu, X. Chen et al., “Codec-SUPERB: An in-depth analysis of sound codec models,” in Findings of ACL, 2024.

[39] H. Wu, X. Chen et al., “Towards audio language modeling – an overview,” arXiv preprint arXiv:2402.13236, 2024.

[40] J. Alstott, E. Bullmore, and D. Plenz, “powerlaw: A Python package for analysis of heavy-tailed distributions,” PLoS ONE, vol. 9, no. 1, p. e85777, 2014.

[41] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023.

[42] M. Morise, F. Yokomori, and K. Ozawa, “WORLD: A vocoder-based high-quality speech synthesis system for real-time applications,” IEICE Transactions on Information and Systems, vol. E99-D, no. 7, pp. 1877– 1884, 2016.

[43] P. Boersma, “Praat, a system for doing phonetics by computer,” Glot International, vol. 5, no. 9/10, pp. 341–345, 2001.

[44] K. Ito and L. Johnson, “The LJ Speech dataset,” 2017, https://keithito. com/LJ-Speech-Dataset/.

[45] A. Nagrani, J. S. Chung, and A. Zisserman, “VoxCeleb: A large-scale speaker identification dataset,” in Proc. INTERSPEECH, 2017.