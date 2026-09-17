# Beyond frequency measures: Can contextual embeddings capture meaning change in scientific texts?

Jianying Liu<sup>1</sup>, Kim Gerdes<sup>2</sup>, Jean-Marc Deltorn<sup>3</sup>

<sup>1</sup>LISN, Université Paris-Saclay; CEIPI, Université de Strasbourg − jianying.liu@universite-paris-saclay.fr

<sup>2</sup>LISN, Université Paris-Saclay − kim.gerdes@universite-paris-saclay.fr <sup>3</sup>CEIPI, Université de Strasbourg − jm.deltorn@ceipi.edu

## Abstract

Identifying technological trends is a core scientometric task, yet traditional frequency-based approaches struggle to capture substantial meaning shifts of domain-specific terms. We hypothesise that contextual embeddings can complement frequency dynamics to effectively track diachronic semantic change. We compare frequency and embedding-based approaches across Astrophysics and NLP corpora spanning from 2010 to 2024. Candidate terms are extracted using KeyBERT (utilizing SciBERT as its underlying language model) and filtered for significant frequency increases using Fisher’s exact test. These terms are then evaluated for genuine semantic shift by domain experts to establish ground-truth labels. To quantify semantic drift, each term’s contextual embedding “clouds” from the two discrete periods are compared using multiple metrics: cosine distance, average pairwise distance, Hotelling-type T<sup>2</sup>, and maximum mean discrepancy. Results indicate that frequency-based methods align slightly better with human judgments of “trend-related terms” than semantic metrics (Precision@50 of 0.62 vs 0.60 in Astrophysics). The two signals show a correlation of around 0.6. Several terms identified exclusively by embedding metrics (e.g., “primordial black holes”) represent critical conceptual developments invisible to pure frequency analysis. These findings indicate that semantic metrics may capture complementary information, highlighting the value of integrating contextual embeddings into scientometric trend analysis.

Keywords: contextual embeddings, Lexical semantic change, Large/pretrained language model, Embedding distance metrics, Diachronic statistical analysis of scientific terms

## 1. Introduction

Identifying technological hot spots and trends from large textual corpora is a core task in scientometrics. Traditional analysis mainly relies on frequency-based indicators and co-word networks (Callon et al., 1991), which provide useful macroscopic views but remain limited in capturing substantial meaning shifts of domain-specific key terms over time.

In this work, we hypothesize that diachronic semantic change of in-domain key terms can occur independently of term frequency dynamics and can be captured by contextual vector representations, thereby providing complementary information. To test this hypothesis, we conduct a comparative study between frequency-based and contextual embedding-based approaches in two scientific domains: Astrophysics and speech and natural language processing (SNLP), using article corpora from 2010 and 2024 for Astrophysics, and from 2010 and 2020 for SNLP.

This work’s contributions are fourfold: (1) we propose a trend-detection method using contextual embeddings to avoid full-corpus LLM processing, and we show how its performance varies across domains; (2) we evaluate keyword quality and the difficulty of semantic-shift annotation via human and LLM annotations; (3) we compare various embedding distance metrics against frequency-based methods and analyze the correlation between these two types of metrics; and (4) we qualitatively analyze categories of semantic shifts in scientific domains, with examples from astrophysics and SNLP.

## 2. Related Work

## 2.1. Diachronic Embedding Models

Detecting semantic change using vector representations has evolved from static to contextual frameworks. Early works predominantly utilized static dense word vectors trained independently and aligned post-hoc (Hamilton et al., 2016), or enforced temporal alignment during representation learning (Bianchi et al., 2020; Di Carlo et al., 2019). Recently, however, contextualized models like BERT and RoBERTa have dominated lexical semantic change (LSC) tasks (Schlechtweg et al., 2020). For scientific literature, domain-adapted language models such as SciBERT (Beltagy et al., 2019) demonstrate superior semantic tracking capabilities.

## 2.2. Semantic Change Assessment Metrics

To quantify semantic drift, LSC methodologies typically operate under two paradigms: prototype tracking (e.g., Cosine Distance) and distribution tracking (Periti and Montanelli, 2024). The latter models the full sense inventory by clustering contextual embeddings across time (e.g., APD). Beyond these, metrics like the Regularized Hotelling statistic $T ^ { 2 }$ stabilized by Ledoit– Wolf shrinkage (Ledoit and Wolf, 2004) and Maximum Mean Discrepancy (MMD) (Gretton et al., 2012; Jayasumana et al., 2024) have gained traction for high-dimensional Mahalanobis and distribution distance estimation (Han et al., 2018; Podolskiy et al., 2021).

Beyond these, some studies have also used the Regularized Hotelling statistic $T ^ { 2 }$ for distribution distance. While relatively niche in NLP (Han et al., 2018), it is a standard approach in gene set analysis for high-dimensional data (Chen and Qin, 2010), especially when stabilized by Ledoit– Wolf shrinkage (Ledoit and Wolf, 2004; Robinson et al., 2022). Fundamentally, $T ^ { 2 }$ represents the Mahalanobis distance between mean vectors, a metric extensively applied in NLP tasks like Out-Of-Distribution (OOD) detection (Podolskiy et al., 2021). Furthermore, Maximum Mean Discrepancy (MMD) (Gretton et al., 2012) measures the distance between two embedding distributions and can also be used to align latent spaces. It has been successfully applied in both computer vision (Baktashmotlagh et al., 2016; Jayasumana et al., 2024) and NLP (Fonseca and van Dijk, 2020).

The choice of distance metrics is critical, as relying solely on cosine distance can be insufficient for robust semantic shift detection. Azarpanah and Farhadloo (2021) demonstrate that the selection of similarity measures and descriptive statistics (e.g., min, max, mean, median) significantly influences the conclusions of word embedding association tests. To address these sensitivities, Liu et al. (2021) incorporate non-parametric permutation tests with contextual embeddings, ensuring that observed semantic shifts are statistically significant and not artifacts of sampling variance.

## 2.3. Correlation Between Frequency and Semantic Shift

The interaction between frequency trajectories and meaning change remains a debated null hypothesis. Hamilton et al. (2016) proposed statistical laws of semantic change (conformity and innovation), which were later challenged as potential artifacts of static embeddings and simple cosine distances (Dubossarsky et al., 2017). Subsequent work (Keidar et al., 2022) using causal DAG modeling observed a decoupling between volatile frequency changes and genuine conceptual drift, justifying the exploration of advanced quantitative techniques to isolate substantive semantic evolution in scientific corpora.

## 3. Methodology

![](images/a5d50d34e377a263559fce534a19a6a464136e865035dc6f0db069785f26859f.jpg)  
Figure 1: Processing pipeline and methodology flowchart including semantic extraction, dual-metric distance comparison, and LLM annotation validation steps.

## 3.1. Data Sources

We test our hypothesis on two distant scientific domains: Astrophysics (Astro) and Speech and Natural Language Processing (SNLP). For both corpora, only article titles and abstracts are utilized. We sample data from two distinct time periods for comparative analysis (2010 vs. 2024 for Astrophysics, and 2010 vs. 2020 for SNLP). <sup>1</sup> The Astrophysics corpus is built from arXiv articles labeled astro-ph, yielding 32,336 articles. The SNLP domain reuses the NLP4NLP corpus from (Mariani et al., 2022), scoped to 8,413 English papers. Table 1 summarizes the corpus statistics.

Table 1: Corpus statistics: Astrophysics (Astro) and SNLP
<table><tr><td>Domain</td><td>Statistic</td><td>2010</td><td>2020 /2024</td><td>Total</td></tr><tr><td rowspan="4">Astrophysics</td><td>Paper number</td><td>13,118</td><td>19,218</td><td>32,336</td></tr><tr><td>Token count</td><td>2,565,045</td><td>4,186,388</td><td>6,751,433</td></tr><tr><td>Keyword candidates (forms)</td><td>64,017</td><td>97,050</td><td>147,728 (union)</td></tr><tr><td>Studied keyterms (occ ≥ 5)</td><td></td><td></td><td>8,013</td></tr><tr><td rowspan="4">SNLP</td><td>Paper number</td><td>3,274</td><td>5,139</td><td>8,413</td></tr><tr><td>Token count</td><td>445,562</td><td>824,357</td><td>1,269,919</td></tr><tr><td>Keyword candidates (forms)</td><td>19,706</td><td>29,537</td><td>46,572 (union)</td></tr><tr><td>Studied key terms (occ ≥ 1)</td><td></td><td></td><td>6,140</td></tr></table>

## 3.2. Keyword Extraction and Embedding

We perform a two-step domain-specific keyword extraction (Figure 1). First, KeyBERT (Grootendorst et al., 2021), configured with SciBERT, extracts the top 10 n-grams per article to act as candidate terms. Instead of restricting our tracking strictly to articles where a term was flagged as important, we consolidate the unique candidates and perform a global parse to extract their occurrence counts (occ) across all texts in the two distinct years. Filtering out extremely rare terms with an occurrence threshold yields an effective vocabulary of 8,013 terms for Astrophysics and $6 { , } 1 4 0$ for SNLP. We then apply SciBERT to compute contextual embeddings for each valid term, using a surrounding context window of 125 words $( w s = 1 2 5 )$ on each side.

Ultimately, between the initial time period $\left( t _ { 1 } \right)$ and the target time period $( t _ { 2 } )$ , each term X is represented by its total occurrences $( o c c _ { 1 }$ and $o c c _ { 2 } )$ , its popularity ranks sorted descendingly by occurrences $( r a n k _ { 1 }$ and $r a n k _ { 2 } )$ , and two temporal sets of contextual embeddings $( \Phi _ { 1 }$ and $\Phi _ { 2 } )$ complete with their respective mean prototypes $( \mu _ { 1 }$ and $\mu _ { 2 } )$ .

## 3.3. Popularity Metrics

We measure term popularity change using two frequency-based methods. The first relies on $\mathrm { Z i p f ^ { \prime } s }$ law to attenuate sensitivities caused by drastic rank scaling compared to most frequent words. To avoid noise in the long tail of the distribution, we fit an inverse power-law coefficient α on the top 500 terms: $\widehat { o c c } = \bar { 1 } 0 ^ { b } \cdot r a n k ^ { - \alpha }$ . A popularity growth ratio, denoted as $\rho ,$ , is then calculated to score the relative usage expansion:

$$
\rho = - \left( { \frac { r a n k _ { 2 } } { r a n k _ { 1 } } } \right) ^ { \alpha }\tag{1}
$$

Secondly, to rigorously account for the exponential baseline growth of the entire scientific corpus, we measure the specificity and statistical significance of an absolute frequency surge in $t _ { 2 }$ using Fisher’s exact test. For a given term X, the probability mass function representing its hypergeometric distribution is formulated as:

$$
p ( o c c _ { 2 } ; N , o c c _ { \mathrm { a l l } } , n ) = \frac { { \binom { o c c _ { \mathrm { a l l } } } { o c c _ { 2 } } } { \binom { N - o c c _ { \mathrm { a l l } } } { n - o c c _ { 2 } } } } { \binom { N } { n } }\tag{2}
$$

where N is the total occurrence count of all selected keywords in the corpus, n is the number of selected keywords, and $o c c _ { \mathrm { a l l } } = o c c _ { 1 } + o c c _ { 2 }$ is the total occurrence count of term X. The Fisher Specificity Score $F _ { s p e c }$ is derived from the survival function $1 - c d f ( o c c _ { 2 } )$ ; higher values indicate statistically significant usage increases over the studied span.

## 3.4. Embedding Difference Metrics

We try both paradigms and different metrics of semantic shift proposed in (Periti and Montanelli, 2024). Under the mean-vector (word-prototype) paradigm, we evaluate both cosine distance and the inverse of cosine similarity (PRT); for the meaning cluster paradigm, we employ average pairwise distance, maximum mean discrepancy, and the Regularized Hotelling statistic $T ^ { \tilde { 2 } }$ . Since PRT performs identically to cosine distance, we focus our reporting on cosine distance results.

Let $e _ { 1 , i } \in \Phi _ { 1 }$ and $e _ { 2 , j } \in \Phi _ { 2 }$ represent individual contextual embeddings of term X from time periods 1 and 2, with sizes $N _ { 1 } = | \Phi _ { 1 } |$ and $N _ { 2 } = | \Phi _ { 2 } |$ respectively.

• Cosine Distance: Directly measured on the mean embeddings $\mu _ { 1 }$ and $\mu _ { 2 }$ .

• Average Pairwise Distance (APD): The average distance between all pairs of embeddings from the two time periods:

$$
\mathrm { A P D } ( \Phi _ { 1 } , \Phi _ { 2 } ) = \frac { 1 } { N _ { 1 } N _ { 2 } } \sum _ { i = 1 } ^ { N _ { 1 } } \sum _ { j = 1 } ^ { N _ { 2 } } d ( e _ { 1 , i } , e _ { 2 , j } )\tag{3}
$$

• Maximum Mean Discrepancy (MMD): We use a biased MMD estimator with an RBF Gaussian kernel $k ( x , y ) = \exp ( - \gamma \| x - y \| ^ { 2 } )$ and fixed bandwidth $\sigma = 1 0 . 0 ^ { 2 }$

• Regularized Hotelling statistic $T ^ { 2 }$ : We estimate a regularized pooled covariance matrix $\Sigma _ { \mathrm { r e g } }$ using Ledoit–Wolf shrinkage and measure the separation between sample mean embeddings via:

$$
T ^ { 2 } = ( \mu _ { 1 } - \mu _ { 2 } ) ^ { T } \Sigma _ { \mathrm { r e g } } ^ { - 1 } ( \mu _ { 1 } - \mu _ { 2 } )\tag{4}
$$

## 3.5. Ground-truth expert annotation

To establish the human reference labels used in the main evaluation, we curated two samples of 500 candidate keywords, one for Astrophysics and one for SNLP. In each domain, candidates were first ranked by Fisher Specificity Score $( F _ { s p e c } )$ , then we selected the highest-ranked 200 terms and stratified the remaining 300 across lower-ranked percentiles.

The annotation protocol at this stage used a lightweight guideline, intended to preserve domainexpert judgment on inherently fuzzy boundary cases while remaining simple enough for downstream reliability checks. For each candidate exhibiting salient frequency changes, the annotators judged whether it should be treated as outside the target domain scope (Class G), as an in-domain but semantically stable term (Class D), or as an in-domain term whose contextual or conceptual meaning had changed over time (Class DC). Broad methodological phrases, popular writing formulas, and noisy extractions were generally treated as Class G even when their usage increased, because they do not constitute the target phenomenon of scientific meaning change.

## 4. Experiments and Results

## 4.1. Shift Discovery compared to expert judgement

Because Class DC exclusively represents our target phenomenon of genuine semantic shift, we first benchmark the retrieval performance of each metric specifically against this class. After ranking all candidates by their score according to each metric, we evaluate both the top-k retrieval precision (Precision@k) and recall (Recall@k), using the human annotations as reference labels.

## Table 2 reveals three main patterns.

Frequency-based methods align more consistently with human labels overall: Statistically, frequency-driven methods (popularity growth ratio (ρ) and Fisher Specificity Score $( F _ { s p e c } ) )$ align more consistently with human annotations across the full annotated set. They remain among the top-performing methods across the global ranking, with at least one of them consistently appearing in the top two.

Reg. $T ^ { 2 }$ achieves the strongest top-tier precision: Among all the tested embedding distance paradigms, the Regularized Hotelling statistic (Reg. $T ^ { 2 } )$ drastically outperforms alternatives like Cosine distance and MMD. When constrained to high-confidence retrieval windows (e.g., $k \leq 5 0 )$ , the uncalibrated Reg. $T ^ { 2 }$ can even surpass purely frequency-based approaches. It achieves strong top-tier detection (e.g., P@20 of 0.750 in Astrophysics and 0.700 in SNLP), aligning closely with human intuition for profound conceptual semantic shift. However, affected by long-tail effects, this strong early precision does not extend to the full ranking. When all annotated candidates are considered and labelled with a binary Class-DC indicator, only modest Spearman correlation $r _ { s }$ is shown between this indicator and the raw Reg. $T ^ { 2 }$ , with $r _ { s } = 0 . 1 3 2$ for Astro and $r _ { s } = 0 . 1 2 9$ for SNLP.

Naive distances remain relatively insensitive to subtle meaning change: The smaller retrieval ranges (P@20 and P@50) expose a massive performance gap between the two $T ^ { 2 } .$ -based metrics and alternative embedding distances, indicating that metric baselines like MMD or raw Cosine mappings are insensitive to the nuanced, high-dimensional meaning shifts recognized by domain experts.

Table 2: Retrieval performance (Precision/Recall@K) for Class DC and Spearman correlation $( r _ { s } )$ against human binary labels. Evaluated across Astrophysics $( N = 4 9 8 )$ and SNLP (N = 390) domains.
<table><tr><td rowspan="2">Metric</td><td colspan="5">Astro (Class DC N = 133)</td><td colspan="5">SNLP (Class DC N = 131)</td></tr><tr><td>P@20</td><td>P@50</td><td>P@100</td><td>R@100</td><td> $r _ { s }$  (Human)</td><td>P@20</td><td>P@50</td><td>P@100</td><td>R@100</td><td> $r _ { s }$  (Human)</td></tr><tr><td>Cosine Distance</td><td>0.100</td><td>0.280</td><td>0.230</td><td>0.173</td><td>-0.019</td><td>0.250</td><td>0.340</td><td>0.320</td><td>0.244</td><td>0.024</td></tr><tr><td>APD (Cosine)</td><td>0.350</td><td>0.200</td><td>0.190</td><td>0.143</td><td>-0.075</td><td>0.300</td><td>0.280</td><td>0.270</td><td>0.206</td><td>-0.016</td></tr><tr><td>MMD</td><td>0.100</td><td>0.120</td><td>0.210</td><td>0.158</td><td>-0.040</td><td>0.250</td><td>0.300</td><td>0.330</td><td>0.252</td><td>0.018</td></tr><tr><td>Reg.  $T ^ { 2 }$ </td><td>0.750</td><td>0.600</td><td>0.400</td><td>0.301</td><td>0.132</td><td>0.700</td><td>0.540</td><td>0.460</td><td>0.351</td><td>0.129</td></tr><tr><td>Reg.  $T ^ { 2 } \left( 1 - p \mathrm { - v a l u e } \right)$ </td><td>0.350</td><td>0.340</td><td>0.320</td><td>0.241</td><td>0.168</td><td>0.400</td><td>0.420</td><td>0.420</td><td>0.321</td><td>0.129</td></tr><tr><td> $\rho$ </td><td>0.750</td><td>0.620</td><td>0.550</td><td>0.414</td><td>0.386</td><td>0.600</td><td>0.480</td><td>0.450</td><td>0.344</td><td>0.176</td></tr><tr><td> $F _ { s p e c }$ </td><td>0.500</td><td>0.520</td><td>0.460</td><td>0.346</td><td>0.344</td><td>0.550</td><td>0.400</td><td>0.460</td><td>0.351</td><td>0.127</td></tr></table>

Note: Precision and Recall of Class DC annotated by human experts. The total number of Class DC target keywords is 133 for Astrophysics and 131 for SNLP. $r _ { s }$ (Human) represents the Spearman correlation of the metric against the binary indicator for Class DC. Best performing scores are highlighted in bold, and second-best are underlined. Same for the following tables.

To test whether a large Reg. $T ^ { 2 }$ value reflects a robust period difference rather than a sampling artifact, we estimated a permutation-based p-value for each keyword by shuffling time period labels 1,000 times, then used $1 - p$ as a calibrated score, so that larger values indicate stronger evidence against statistical noise.

Conversely, the statistically calibrated permutation significance $( 1 - p \mathrm { - v a l u e } )$ metric consistently outperforms the raw geometric distance across the full annotated subset $( r _ { s } = 0 . 1 6 8$ in Astro, $r _ { s } = 0 . 1 2 9$ in SNLP, whereas the corresponding values for the raw geometric distances remain below 0.1 in absolute value). This highlights the “denoising” effect of permutation testing: high-dimensional geometric distances are often deceptive, artificially inflated by sampling variance and outlier contexts in small-sample scenarios. Statistical calibration effectively filters this noise, isolating authentic semantic evolution from linguistic background fluctuations. This observation coheres with Liu et al. (2021)’s assertion that significance testing is important for embedding shift analysis.

## SCIENTIFIC TEXTS?

## 4.2. Correlation Between Metrics

To understand whether the metrics capture redundant or distinct phenomena, we analyzed the Spearman rank correlation between frequency evolution indicators $( \rho , F _ { s p e c } )$ and semantic embedding distances within the Class DC subset, i.e., terms confirmed to have undergone semantic meaning change (Table 3).

Table 3: Cross-metric Spearman Correlation on the Class DC Subset. For the strongest correlation per column (bold), the 95% confidence interval and statistical significance (p-value) are provided in parentheses.
<table><tr><td rowspan="2">Semantic Metric</td><td colspan="2">Astro (Class DC N = 133)</td><td colspan="2">SNLP (Class DC N = 131)</td></tr><tr><td>ρ</td><td> $F _ { s p e c }$ </td><td>ρ</td><td> $F _ { s p e c }$ </td></tr><tr><td>Cosine Distance</td><td>0.161</td><td>-0.290</td><td>0.135</td><td>-0.264</td></tr><tr><td>APD (Cosine)</td><td>0.172</td><td>0.366</td><td>0.154</td><td>0.384</td></tr><tr><td>MMD</td><td>0.090</td><td>-0.445</td><td>0.087</td><td>-0.380</td></tr><tr><td>Reg.  $T ^ { 2 }$ </td><td>0.646</td><td>0.058</td><td>0.690</td><td>0.198</td></tr><tr><td rowspan="2">Reg.  $T ^ { 2 } \left( 1 - p \mathrm { - v a l u e } \right)$ </td><td>([0.496, 0.763], p = 4.947 × 10−17)</td><td></td><td>([0.548, 0.789], p = 8.314 × 10−20)</td><td></td></tr><tr><td>0.155</td><td>0.489  $( [ 0 . 3 4 0 , 0 . 6 0 7 ] , p = 2 . 4 3 \times 1 0 ^ { - 9 } )$ </td><td>0.259</td><td>0.545  $( [ 0 . 4 0 9 , 0 . 6 5 8 ] , p = 1 . 7 2 3 \times 1 0 ^ { - 1 1 } )$ </td></tr></table>

Our cross-metric analysis reveals a clear divergence in how different methods capture semantic shift. As shown in Table 3, the Reg. $T ^ { 2 }$ statistic demonstrates a strong positive correlation (0.65 for Astro and 0.69 for SNLP) with the absolute frequency growth (ρ). In contrast, baseline distance metrics (e.g., Cosine Distance, MMD) show minimal correlation with frequency changes, with APD occasionally being the second highest but still remaining relatively low. This suggests that Reg. $T ^ { 2 }$ acts more as a frequency-sensitive metric: its high values are closely associated with drastic shifts in term frequencies, which geometrically pull the sampling statistics apart.

Conversely, the Reg. $T ^ { 2 } \left( 1 - p \mathrm { - v a l u e } \right)$ exhibits a different behavior. Its correlation with absolute frequency growth drops, and it instead aligns more closely with the specificity indicator $F _ { s p e c }$ (0.49 for Astro and 0.55 for SNLP).

These observations suggest two distinct dimensions for tracking terminology evolution. Empirical measurements of shift magnitude (e.g., Reg. $T ^ { 2 }$ and frequency rank drift $\rho )$ characterize the physical scale of term displacement, which is often tied to surges in popularity. Meanwhile, statistical measurements (e.g., the Reg. $T ^ { 2 } ( 1 - p \mathrm { - v a l u e } )$ and the $F _ { s p e c } )$ act as significance filters. They help control for sampling variance caused by frequency volatility and highlight terms whose newer usage has become more contextually specialized.

## 4.3. Annotation Reliability and LLM-Assisted Scalability

Because expert annotation limits the scalability of semantic-shift evaluation, we assess whether LLM-based annotation can support this process with acceptable reliability.

We completed three consistency analyses: Human–LLM comparison for GPT-4.1-mini, Llama-3.3-70b-versatile, and Qwen3-32b; inter-model comparison among the same three models; and inter-annotator comparison. As Table 4 shows, the two domains exhibit different disagreement profiles. Astrophysics shows stronger model-dependent variation, whereas SNLP yields more uniformly moderate agreement across models. Overall, LLM annotation is informative but not stable enough to replace human ground truth.

To diagnose disagreement more precisely, we also conducted a second-round expert agreement study on the subset of annotated SNLP terms occurring more than five times in both periods.

Table 4: Human–LLM and inter-human annotation consistency.
<table><tr><td rowspan="2">Row</td><td colspan="4">Astro</td><td colspan="4">SNLP</td></tr><tr><td>N</td><td>Agreement</td><td>K</td><td>DC Agr.</td><td>N</td><td>Agreement</td><td>K</td><td>DC Agr.</td></tr><tr><td>GPT-4.1-mini</td><td>500</td><td>0.520</td><td>0.309</td><td>0.842</td><td>390</td><td>0.610</td><td>0.409</td><td>0.634</td></tr><tr><td>Llama-3.3-70b-versatile</td><td>496</td><td>0.643</td><td>0.449</td><td>0.654</td><td>390</td><td>0.600</td><td>0.388</td><td>0.603</td></tr><tr><td>Qwen3-32b</td><td>500</td><td>0.670</td><td>0.481</td><td>0.466</td><td>390</td><td>0.590</td><td>0.381</td><td>0.565</td></tr><tr><td>Expert B</td><td></td><td></td><td></td><td></td><td>215</td><td>0.577</td><td>0.369</td><td>0.300</td></tr><tr><td>Mean inter-LLM</td><td>497</td><td>0.591</td><td>0.403</td><td>0.433</td><td>390</td><td>0.658</td><td>0.481</td><td>0.664</td></tr><tr><td>Mean inter-human</td><td>一</td><td></td><td></td><td></td><td>189</td><td>0.557</td><td>0.311</td><td>0.191</td></tr></table>

Note: The first three rows compare each model against Expert A. The last two rows report the mean pairwise scores across the three listed models and across the three human annotations. Agreement is the proportion of identical labels over the shared keyword set; κ is Cohen’s kappa; DC Agr. is the agreement rate restricted to human-labeled Class DC items.

This follow-up uses a refined guideline that decomposes the original decision into three parallel binary judgments: whether a term is in-domain, trend-related, and meaning-shifting. Annotators may additionally mark certainty and assign standardized notes for ambiguous cases, such as truncation, excessive generality, or popular tasks associated with new methods. The study involves the original annotator (Expert A) plus seven additional in-domain experts; Expert A and B completed the full 215-term list, while the others jointly covered a 189-term subset. As Table 4 shows, inter-human agreement is not higher than human-model agreement, confirming that semantic-shift annotation is itself a difficult task.

## 4.4. Ablation on Context Window Size

Our main setting uses ws = 125, chosen with reference to text length in our corpora: because this study relies on article titles and abstracts only, which together average roughly 200 words, a 125-token window on each side already covers most of the available local context. We also compared results for ws = 5, 50, and 125. The main conclusions remain stable across these settings: frequency-based methods align better with expert labels globally, and Reg. $T ^ { 2 }$ remains the strongest embedding-based retrieval metric. Smaller windows weaken the T<sup>2</sup>-based scores overall and make APD relatively more correlated with Class DC, suggesting that broader context is beneficial for the statistical distance measures used here.

## 5. Discussion: Categories of Semantic Shifts in Scientific Discourse

A qualitative inspection of high-shift terms demonstrates that semantic evolution in scientific writing follows distinct typological patterns, which are largely decoupled from raw frequency metrics. The scatter plots in Figure 2 provide explicit visual evidence that term popularity does not inherently indicate a shift in underlying meaning. For instance, while keywords like “dataset” experienced massive frequency growth over the 10-year observational window, their contextual embedding clusters from the two epochs overlap almost entirely, indicating stable phenomenological semantics. Conversely, the contextual coordinates for “radio bursts” and “neural network” exhibit geometric divergence across the temporal subsets, confirming that their conceptual application has structurally evolved alongside their rise in citation popularity.

Beyond frequency independence, our qualitative review isolates two different modes of semantic change in scientific domains: (1) Interpretive Shifts Under a Stable Label, where the

BEYOND FREQUENCY MEASURES: CAN CONTEXTUAL EMBEDDINGS CAPTURE MEANING CHANGE IN SCIENTIFIC TEXTS?

![](images/81b678202936679fd108be94aa8512fbcc9fe2de9e4151003212e3e10499d592.jpg)  
Figure 2: Visualizing the decoupling offrequency growth from semantic shift via contextual embedding tracking. From left to right: (A) Astro: galaxies (High Freq, Stable Semantics). (B) Astro: radio bursts (High Freq, High Shift). (C) SNLP: dataset (High Freq, Stable Semantics). (D) SNLP: neural network (High Freq, High Shift). Terms with massive occurrence scale-ups do not automatically incur geometric embedding divergence.

prevailing scientific interpretation fundamentally evolves despite a constant lexical form (e.g., the paradigm shift of “dark matter” in Astrophysics, or “language model” evolving from statistical n-grams to generative architectures in SNLP); and (2) Methodological Diffusion, where a computational tool transitions to widespread foundational deployment, broadening its contextual neighborhood (e.g., the integration of “machine learning” in Astrophysics and “neural network” in SNLP).

## 6. Conclusion and Future Work

This paper presents a hybrid pipeline combining keyphrase extraction, frequency analytics, and contextual embeddings to detect semantic shifts in scientific corpora. We demonstrate that the correlation between frequency-based indicators and embedding-based scores is highly influenced by the specific metrics chosen. Although frequency statistics provide a strong initial proxy for identifying emerging trends, contextual embeddings uncover unique semantic patterns that occur independently of frequency dynamics. Furthermore, zero-shot LLM annotations show strong domain dependence, making human-in-the-loop validation still indispensable for this task. Future work will extend this framework to finer-grained temporal datasets in order to trace continuous conceptual evolution, distinguish different types of semantic shift more clearly, and compare how they are reflected in contextual embedding distributions.

## References

Azarpanah H. and Farhadloo M. (2021). Measuring biases of word embeddings: What similarity measures and descriptive statistics to use? In Proceedings of the First Workshop on Trustworthy Natural Language Processing, pp. 8–14.

Baktashmotlagh M., Harandi M., and Salzmann M. (2016). Distribution-matching embedding for visual domain adaptation. Journal ofMachine Learning Research, 17(108):1–30.

Beltagy I., Lo K., and Cohan A. (2019). Scibert: A pretrained language model for scientific text. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 3615–3620.

Bianchi F., Carlo V. D., Nicoli P., and Palmonari M. (2020). Compass-aligned Distributional Embeddings for Studying Semantic Differences across Corpora.

Callon M., Courtial J. P., and Laville F. (1991). Co-word analysis as a tool for describing the network of interactions between basic and technological research: The case of polymer chemistry. Scientometrics, 22(1):155–205.

Chen S. X. and Qin Y.-L. (2010). A two-sample test for high-dimensional data with applications to gene-set testing. The Annals ofStatistics, 38(2):808–835.

Di Carlo V., Bianchi F., and Palmonari M. (2019). Training Temporal Word Embeddings with a Compass. Proceedings ofthe AAAI Conference on Artificial Intelligence, 33(01):6326–6334.

Dubossarsky H., Weinshall D., and Grossman E. (2017). Outta control: Laws of semantic change and inherent biases in word representation models. In Proceedings of the 2017 conference on empirical methods in natural language processing, pp. 1136–1145.

Fonseca A. H. and van Dijk D. (2020). Learning aligned embeddings for semi-supervised word translation using maximum mean discrepancy. arXiv preprint arXiv:2006.11578.

Gretton A., Borgwardt K. M., Rasch M. J., Schölkopf B., and Smola A. (2012). A Kernel Two-Sample Test. Journal ofMachine Learning Research, 13(25):723–773.

Grootendorst M., Fuetterer H.-A., Luca F., Dhadse A., Matsak A., Pechersky I., Govil P., Frampton S., Ogura Y., et al. (2021). Maartengr/keybert: v0. 9.

Hamilton W. L., Leskovec J., and Jurafsky D. (2016). Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change. In Erk K. and Smith N. A., editors, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1489–1501. Association for Computational Linguistics.

Han R., Gill M., Spirling A., and Cho K. (2018). Conditional Word Embedding and Hypothesis Testing via Bayes-by-Backprop. In Riloff E., Chiang D., Hockenmaier J., and Tsujii J., editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 4890–4895. Association for Computational Linguistics.

Jayasumana S., Ramalingam S., Veit A., Glasner D., Chakrabarti A., and Kumar S. (2024). Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9307–9315.

Keidar D., Opedal A., Jin Z., and Sachan M. (2022). Slangvolution: A causal analysis of semantic change and frequency dynamics in slang.

Ledoit O. and Wolf M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal ofMultivariate Analysis, 88(2):365–411.

Liu Y., Medlar A., and Glowacka D. (2021). Statistically significant detection of semantic shifts using contextual word embeddings. In Gao Y., Eger S., Zhao W., Lertvittayakumjorn P., and Fomicheva M., editors, Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems, pp. 104–113, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Mariani J., Francopoulo G., Paroubek P., and Vernier F. (2022). NLP4NLP+5: The Deep (R)evolution in Speech and Language Processing. Frontiers in Research Metrics and Analytics, 7.

Periti F. and Montanelli S. (2024). Lexical Semantic Change through Large Language Models: a Survey. ACM Computing Surveys, 56(11):1–38.

Podolskiy A., Lipin D., Bout A., Artemova E., and Piontkovskaya I. (2021). Revisiting mahalanobis distance for transformer-based out-of-domain detection. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 13675–13682.

Robinson B., Malinas R., Latimer V., Morrison B. B., and Hero A. O. (2022). An improvement on the hotelling t<sup>2</sup> test using the ledoit-wolf nonlinear shrinkage estimator. In 2022 30th European Signal Processing Conference (EUSIPCO), pp. 2106–2110. IEEE.

Schlechtweg D., McGillivray B., Hengchen S., Dubossarsky H., and Tahmasebi N. (2020). Semeval-2020 task 1: Unsupervised lexical semantic change detection.