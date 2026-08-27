# Loss-Based Active Learning for Neural Abstractive Summarization

Michail Ioannou, Tatiana Passali, George Michalopoulos, Grigorios Tsoumakas

School of Informatics

Aristotle University of Thessaloniki

Greece

mioannoug@csd.auth.gr, scpassali@csd.auth.gr, georgemi@microsoft.com, greg@csd.auth.gr

## Abstract

Fine-tuning abstractive summarization models requires high-quality annotated data. However, obtaining such corpora is expensive and timeconsuming, as it requires human annotators to read and comprehend long documents to create accurate summaries. Active learning mitigates this issue by selecting only the most informative instances for annotation, allowing models to achieve competitive results with significantly fewer labels. However, the application of active learning to summarization remains underexplored, and existing studies often suffer from instability and significant computational bottlenecks. To overcome these challenges, we propose LOBSTER (LOss-BaSed acTivE leaRning), a novel active learning framework designed specifically for abstractive summarization. LOBSTER improves performance by prioritizing unlabeled instances semantically similar to the model’s current high-loss training examples, enabling the model to explicitly correct its specific weaknesses. Our empirical evaluation across three benchmark datasets and two summarization backbone models demonstrates that LOBSTER consistently matches or outperforms current state-of-the-art approaches while achieving a query selection speedup of up to 665×.

## 1 Introduction

Abstractive summarization generates concise summaries by producing novel words and phrases rather than directly copying from the source (See et al., 2017; Zhang et al., 2020a). State-of-theart performance has been achieved by pre-trained large language models (LLMs) (Lewis et al., 2020; Zhang et al., 2020a; Zaheer et al., 2020), based on the Transformers architecture (Vaswani et al., 2017). Nevertheless, fine-tuning LLMs for abstractive summarization is costly, since producing highquality summaries requires human annotators to carefully read the full document and identify important information (Tsvigun et al., 2022; Perlitz et al., 2023).

Active learning addresses this challenge by iteratively selecting the most informative examples for annotation. This enables models to achieve strong performance with fewer training instances, thereby reducing the overall cost of obtaining labeled data (Settles, 2012). While recent efforts to apply active learning to abstractive summarization have shown promise, existing methods suffer from critical limitations. Uncertainty-based strategies are computationally demanding and tend to select outliers, as these instances often exhibit high uncertainty (Gidiotis and Tsoumakas, 2024). Conversely, diversity-based approaches can be unstable since they may overemphasize dense regions of the data space (Tsvigun et al., 2022), and more recent methods rely on external LLM systems, which can introduce additional cost and dependency (Li et al., 2024).

This work introduces a novel technique that strikes a balance between summarization performance and computational efficiency. We demonstrate that the cross-entropy loss of current labeled examples provides a useful signal for guiding the model toward the specific data it requires to improve. Building on this insight, we present LOB-STER (LOss-BaSed acTivE leaRning), a method that integrates density-based diversity with a mechanism that projects the hardness of labeled examples to unlabeled ones based on their semantic similarity. Our experiments demonstrate that LOBSTER matches or surpasses state-of-the-art performance while being significantly faster than uncertainty-based strategies and more stable than methods relying solely on diversity. To the best of our knowledge, we are the first to use sequenceto-sequence cross-entropy loss of labeled data to guide the selection of unlabeled data in abstractive summarization.

The main contributions of this work are as fol-

lows:

• We propose a hybrid selection strategy that explicitly targets the weaknesses of the current model while maintaining data diversity.

• We conduct extensive experiments across three benchmark datasets and two model backbones, demonstrating the effectiveness of LOBSTER while highlighting the inherent instability of diversity-based methods and the high latency of uncertainty-based strategies.

• We present a detailed data efficiency analysis revealing the surprising competitiveness of random sampling, under large annotation budgets.

## 2 Related Work

## 2.1 Abstractive Summarization

The introduction of neural sequence-to-sequence models (Sutskever et al., 2014) and the attention mechanism (Bahdanau et al., 2015) established the foundations for modern abstractive summarization. Building on these advances, the Transformer architecture (Vaswani et al., 2017) enabled large-scale pre-trained language models such as BART (Lewis et al., 2020), T5 (Raffel et al., 2020) and PEGA-SUS (Zhang et al., 2020a), which set the state of the art. Recent work introduced model adaptations for long or multi-document inputs (Khanna et al., 2022; Xiao et al., 2022), while instruction-tuned LLMs demonstrated strong few-shot and zero-shot capabilities across various tasks, including abstractive summarization, without task-specific fine-tuning (Achiam et al., 2023; Chung et al., 2024).

## 2.2 Active Natural Language Generation

Within the field of NLP, active learning has been extensively studied and applied to discriminative tasks such as text classification (Ein-Dor et al., 2020; Margatina et al., 2021; Yu et al., 2022) and sequence tagging (Shen et al., 2017; Radmard et al., 2021). However, its application to generative tasks has received comparatively little attention (Perlitz et al., 2023). Existing active learning approaches for natural language generation are mainly categorized into uncertainty-based, diversity-based, and hybrid methods (Zhang et al., 2022).

Uncertainty-based methods. These methods select examples for annotation where the model is least confident in its predictions (Settles, 2012).

Gidiotis and Tsoumakas (2022) leverage Bayesian approximation to model uncertainty, by adapting the Monte Carlo BLEU variance metric (Xiao et al., 2020) to summarization. They subsequently apply this uncertainty estimation framework to active learning via their Bayesian Active Summarization (BAS) approach (Gidiotis and Tsoumakas, 2024). Documents with the highest variance are selected for annotation, and the model is iteratively retrained on the labeled set. Empirical results indicate that this approach achieves performance comparable to full-data training with only a few hundred examples under specific experimental settings. Despite these gains, however, purely uncertainty-based strategies in abstractive summarization can be sensitive to data quality, often remaining susceptible to selecting noisy or uninformative examples (Tsvigun et al., 2022; Li et al., 2024).

Diversity-based methods. The goal of these methods is to select examples that are both diverse with respect to the labeled set while being representative of the data distribution (Kim et al., 2006). Tsvigun et al. (2022) propose In-Domain Diversity Sampling (IDDS), which selects instances that differ from the labeled set while still remaining close to the unlabeled pool, balancing diversity with domain relevance. Building on this idea, Li et al. (2024) introduce LLM-Determined Curriculum Active Learning (LDCAL), which leverages LLMs to rate instance difficulty and guide training. They also propose certainty gain maximization, a complementary strategy that selects instances whose distribution aligns closely with the overall data.

Hybrid methods. Recent work has shown that combining uncertainty and diversity methods, can outperform uncertainty-only and diversity-only methods in multiple settings (Azeemi et al., 2025; Giouroukis et al., 2025). Azeemi et al. (2025) introduce Hybrid Uncertainty and Diversity Sampling (HUDS), a hybrid strategy for neural machine translation that combines normalized negative log-likelihood for uncertainty with cosine distance from cluster centroids for diversity. For abstractive summarization, Diversity and Uncertainty Active Learning (DUAL) (Giouroukis et al., 2025) first selects a compact, diverse yet representative subset via IDDS, and then measures uncertainty within that subset using BAS.

## 3 Methodology

Under cross-entropy training, high-loss examples tend to generate larger gradients and thus dominate parameter updates, a property reflected in hard example mining (Shrivastava et al., 2016) and boosting methods (Freund and Schapire, 1997). We exploit this property in the active learning setting by directing supervision toward unlabeled instances that are semantically aligned with high-loss training samples, concentrating annotation budget in regions where parameter updates have greater impact. Our core hypothesis is that an active learning strategy can improve performance by prioritizing unlabeled instances that are semantically similar to the model’s current high-loss training samples. By explicitly targeting the unlabeled semantic counterparts of these hard examples, the model is driven to correct its specific weaknesses.

To achieve this, we introduce LOBSTER (LOss-BaSed acTivE leaRning), a 3-stage sample acquisition strategy that first identifies high-loss examples in the labeled set, then applies density-based filtering to construct a representative candidate pool, and finally selects samples that are semantically similar to the difficult instances without the need for the expensive autoregressive decoding used by uncertainty methods. The complete workflow of the LOBSTER strategy is illustrated in Figure 1.

## 3.1 Problem Formulation

We define the active learning task for abstractive text summarization as follows. Let U represent the large unlabeled pool of documents, and L represent the initial small labeled set. At each active learning cycle $t ,$ an acquisition function utilizes the current model $\theta _ { t }$ to select a batch of B instances from U to be annotated. The goal is to maximize summarization performance on a held-out test set while minimizing the total number of annotated samples.

## 3.2 Loss-Based Active Learning

LOBSTER is a hybrid active learning method that combines loss-guided sampling with IDDS in order to maximize performance. It consists of three stages which we introduce below.

Stage 1: Loss-Based Hard Example Selection. Unlike classification tasks, where errors correspond to strict binary failures, difficulty in abstractive summarization is better reflected by generation loss. High-loss examples are therefore more likely to correspond to error-prone regions of the data distribution. The first stage of LOBSTER aims to diagnose these regions. Unlike uncertainty sampling, which estimates uncertainty on unlabeled data, we explicitly measure generation difficulty on the labeled data. We first compute the token-level cross-entropy loss for all samples in the current labeled set $\mathcal { L } .$ For a given source document x and target summary $y$ of length $T$ , the loss is defined as the average negative log-likelihood over the sequence:

$$
L ( x , y ) = - { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \log P ( y _ { t } \mid y _ { < t } , x ; \theta )\tag{1}
$$

We then identify the set of hard examples, $\mathcal { H } .$ , defined as the top-k examples with the highest loss:

$$
\mathcal { H } = \{ x _ { i } \in \mathcal { L } \mid L ( x _ { i } , y _ { i } ) \geq \tau _ { k } \}\tag{2}
$$

where $\tau _ { k }$ corresponds to the loss of the k-th example in $\mathcal { L }$ when sorted by loss in descending order. This acts as a dynamic threshold that adapts to the model’s current performance, rather than a fixed predefined threshold. These high-loss instances serve as semantic anchors, representing the specific linguistic patterns or domains where the model is currently underperforming.

Stage 2: Representative Candidate Filtering. We observed that retrieving candidates purely via embedding similarity to the hard example set frequently produced clusters of highly correlated, near-duplicate examples. Therefore, we employ IDDS (Tsvigun et al., 2022) as a diversity-aware pre-filter to ensure the candidate pool covers broad, representative regions of the data distribution. Crucially, this mitigates the risk of selecting noisy outliers (a common failure mode in uncertainty-based active learning (Tsvigun et al., 2022)). Furthermore, while density-based methods are known to oversample dense regions when used as a final selector (Li et al., 2024), we avoid this issue by using IDDS strictly to construct a large candidate pool $( U _ { r e p } \gg B )$ rather than for final budget allocation. Filtering is separated from the final selection in Stage 3, ensuring the candidate pool is representative and distinct from already labeled data, effectively mitigating the risks of both outlier selection and dense-region oversampling.

Stage 3: Semantic Hardness Projection. The core contribution of LOBSTER is the explicit targeting of high-loss regions, i.e., difficult regions of the data. We assume that the optimal sample to annotate is the semantic twin of a known hard example. Under this assumption, documents that are semantically close to high-loss labeled examples are likely to provide informative supervision for improving the model.

![](images/7b3c8f617d3f3c85e211c3e19a75bd624c4d4b9441666161ad93206e93b85e08.jpg)  
Figure 1: The three-stage pipeline of LOBSTER. (1) Loss-based Hard Example Selection identifies “Hard Example” set H (red squares) via high cross-entropy loss to serve as semantic anchors. (2) Representative Candidate Filtering uses IDDS to distill a pool of representative candidates (green circles) from the unlabeled set U. (3) Semantic Hardness Projection computes the Max-Similarity Score $S ( x _ { u } )$ (Eq. 3) using Sentence-BERT embeddings. This helps to identify a “semantic twin” for each hard example, forming the final Top-B batch for annotation.

For every document $x _ { u }$ in the representative candidate pool $\mathcal { U } _ { r e p }$ , we compute a maximum similarity score $S ( x _ { u } )$ against the set of hard examples H using Sentence-BERT (Reimers and Gurevych, 2019) embeddings ϕ(·):

$$
S ( x _ { u } ) = \operatorname* { m a x } _ { x _ { h } \in \mathcal { H } } \left( \frac { \phi ( x _ { u } ) \cdot \phi ( x _ { h } ) } { \| \phi ( x _ { u } ) \| \| \phi ( x _ { h } ) \| } \right)\tag{3}
$$

Crucially, we use the max operator rather than the average. We do not want documents that are broadly related to the hard example set, rather we are searching for instances that are similar to specific hard examples. The final batch is constructed by selecting the top B candidates with the highest $S ( x _ { u } )$

## 4 Experimental Setup

## 4.1 Datasets

Following standard benchmarks in the field, we evaluated our active summarization models using three widely used datasets representing diverse summarization styles. AESLC contains short emails with their subject lines as summaries (Zhang and Tetreault, 2019). XSum contains BBC news articles with their single-sentence summaries, requiring high-level abstraction (Narayan et al., 2018). Finally, the CNN/DailyMail dataset consists of news articles accompanied by multi-sentence, bullet-point summaries, providing a benchmark for longer-form abstractive summarization (Hermann et al., 2015; Nallapati et al., 2016; See et al., 2017). Table 2 in Appendix A reports the size of the training and test sets, alongside the average token lengths for both source documents and reference summaries. All datasets consist entirely of English text. We cleaned all datasets by removing noisy instances, such as empty strings and duplicates. Additionally, we filtered the XSum and CNN/DailyMail datasets by excluding documents with fewer than 10 tokens and summaries with fewer than 3 tokens. Finally, due to the high computational cost of the uncertainty-based baselines across multiple active learning cycles, we evaluated our models on a random subset of 1,000 instances from the test sets.

## 4.2 Active Learning Setup

We follow the standard active learning framework used in prior studies (Siddhant and Lipton, 2018; Shelmanov et al., 2021). We initialize each cycle with a randomly sampled seed set of 10 annotated instances, which enables model-dependent strategies to obtain initial acquisition scores. At each iteration, the top 10 examples from the unlabeled pool U are selected according to the acquisition strategy and added to the labeled pool L with their ground-truth summaries. We then fine-tune the summarization model from scratch on L and evaluate it on the held-out test set. The process is repeated for 15 iterations, resulting in a total annotation budget of 150 instances, and results are averaged over 5 independent runs.

## 4.3 Baselines

We compare our approach against the following baselines and state-of-the-art active summarization methods: random sampling, IDDS (Tsvigun et al., 2022), BAS (Gidiotis and Tsoumakas, 2024), and DUAL (Giouroukis et al., 2025). As discussed in Section 2, these methods represent a diverse spectrum of strategies, including diversity-based, uncertainty-based, and hybrid approaches. To ensure a fair comparison with previous studies, we used the code provided by the respective authors.

## 4.4 Implementation Details

We use BART-base (Lewis et al., 2020) and PEGASUS-large (Zhang et al., 2020a) as the summarization backbones. Hyperparameters were tuned on a small randomly sampled subset of the Gigaword dataset (Graff et al., 2003). All experiments were conducted using the HuggingFace Transformers library, with model-specific hyperparameters reported in Table 3 in Appendix A. All experiments were conducted on a Google Cloud Platform (GCP) virtual machine equipped with an NVIDIA L4 GPU. Our code is publicly available at GitHub.

## 4.5 Evaluation Metrics

We evaluate summary quality using both lexical overlap and semantic similarity metrics. For lexical overlap, we compute ROUGE scores (Lin, 2004), including ROUGE-1, ROUGE-2, and ROUGE-L, which measure unigram overlap, bigram overlap, and longest common subsequence overlap, respectively. For semantic similarity, we additionally use BERTScore (Zhang et al., 2020b), which leverages pre-trained contextual embeddings to measure the alignment in meaning between generated and reference summaries. Although we report all ROUGE variants, our analysis primarily focuses on ROUGE-1 and BERTScore, as they provide complementary measures of lexical content preservation and semantic similarity. We also note that ROUGE-2 and ROUGE-L follow trends consistent with ROUGE-1 across all experiments.

## 5 Results Analysis

## 5.1 Performance Comparison

We evaluate the summarization performance of LOBSTER against four strategies across three benchmark datasets and two backbone architectures. Figures 2 and 3 illustrate the ROUGE-1 and BERTScore learning curves, respectively, on the test set across active learning iterations, where solid lines represent the mean score across 5 random seeds and shaded regions indicate the standard deviation. The full results at the final annotation budget are reported in Appendix B.

Competitive Effectiveness. Our analysis indicates that the efficacy of active learning strategies is often context-dependent, varying significantly across different datasets, metrics, and architectures. For example, regarding ROUGE-1 scores on the CNN/DM dataset, BAS achieves the top performance with the PEGASUS model. Conversely, on the XSum dataset using the BART backbone, DUAL emerges as the strongest performer. These dataset- and model-specific variations are similarly observed when evaluating semantic quality with BERTScore. This variability suggests that standard active learning approaches are not universally optimal and are sensitive to the interaction between model architecture and summarization style. However, LOBSTER exhibits strong results, delivering performance that is consistently competitive with or superior to state-of-the-art baselines across these diverse experimental conditions. Specifically, on highly abstractive benchmarks such as XSum, LOBSTER achieves very strong scores with the PEGASUS backbone, outperforming or matching computationally expensive uncertainty-based approaches like BAS and DUAL. To further assess the reliability of these observations at the final annotation budget, we conducted paired permutation tests between LOBSTER and each baseline method across all datasets and backbone architectures. As reported in Table 5, several comparisons yield statistically significant differences $( p < 0 . 0 5 )$ , indicating that the observed performance variations are statistically meaningful in specific experimental settings.

![](images/1b92d352db6c3fe10487dfc4a936108e013a90f49f40d45fe2c82a0505b9dbba.jpg)  
(A) BART on AESLC

![](images/b2dd903ee802f816ae14e9cad585cd248931fcedd443cedb2665a7d8ebd6193b.jpg)  
(B) BART on XSum

![](images/ccef26adff981eb45a3ea33c49ee9d61062755c6e454cb0b68b6eaec4f2a6b54.jpg)  
(C) BART on CNN/DM

![](images/f40094fda0f7f634305bb53560cd34be51d006ea9892e88276f54f0b497143de.jpg)  
(D) PEGASUS on AESLC

![](images/b73dfe3f6dc960b3e2c3e863ffd21d5d0a2ac0b68a2e385133521fea00e02c8e.jpg)  
(E) PEGASUS on XSum

![](images/7acbad6c9368da0b55a0e49bd3518a57e76d0206d2071cc865308819295026b2.jpg)  
(F) PEGASUS on CNN/DM  
Figure 2: Number of labeled instances vs. ROUGE-1. Comparison of active strategies using BART and PEGASUS backbones on AESLC, XSum, and CNN/DM datasets.

Addressing the Cold Start Problem. A key finding is LOBSTER’s ability to correct the inherent weaknesses of the model. Although LOB-STER builds upon the IDDS framework, it demonstrates superior performance. While IDDS is generally competitive, it exhibits notable instability in our experiments, particularly on the CNN/DM dataset and on AESLC with PEGASUS. As seen in Figures 2C, 2D, and 2F and their corresponding BERTScore plots, IDDS suffers from a severe “cold start” problem, lagging significantly behind all other methods during the initial acquisition cycles. This initial instability is particularly critical because the core premise of active learning is to maximize performance under strict annotation budgets. Furthermore, the BERTScore evaluation confirms that this lag is not just a failure to predict exact n-grams, but a drop in actual semantic understanding. In contrast, LOBSTER demonstrates more stable performance throughout the active learning process, as it avoids the initial cold start problem that IDDS faces.

## 5.2 Computational Efficiency

While LOBSTER demonstrates competitive results in all different scenarios, its primary contribution lies in its computational efficiency. Table 1 compares the average runtime (in seconds) for a single active learning selection iteration across different strategies, datasets and models.

Table 1: Average runtime (in seconds) for instance selection, for one active learning iteration, across different strategies. Comparison across backbones and datasets.
<table><tr><td></td><td></td><td colspan="4">Avg. Selection Time (s)</td></tr><tr><td>Model</td><td>Dataset</td><td>BAS</td><td>DUAL</td><td>IDDS</td><td>LOBSTER</td></tr><tr><td rowspan="3">BART-base</td><td>AESLC</td><td>46.1</td><td>38.7</td><td>0.02</td><td>0.6</td></tr><tr><td>XSum</td><td>66.6</td><td>63.7</td><td>0.4</td><td>0.8</td></tr><tr><td>CNN/DM</td><td>70.2</td><td>69.2</td><td>0.3</td><td>0.8</td></tr><tr><td rowspan="3">PEGASUS-large</td><td>AESLC</td><td>221.1</td><td>113.6</td><td>0.01</td><td>1.0</td></tr><tr><td>XSum</td><td>390.0</td><td>267.6</td><td>0.4</td><td>1.5</td></tr><tr><td>CNN/DM</td><td>1064.2</td><td>699.4</td><td>0.4</td><td>1.6</td></tr></table>

Scalability and Runtime Analysis. Strategies relying on autoregressive decoding (e.g., BAS, DUAL) become prohibitively expensive as model and dataset sizes increase. These approaches estimate uncertainty by generating multiple distinct summaries for each candidate document via multiple stochastic forward passes (Monte Carlo dropout). Because these passes must be repeated at each active learning iteration to reassess the unlabeled pool, the computational overhead grows considerably with larger datasets and model scales. These strategies exhibit poor scalability, becoming increasingly impractical as model architectures scale up or as generation complexity increases.

In contrast, LOBSTER adopts a deterministic approach that shifts the computational burden from the expansive unlabeled pool to the much smaller labeled set. After the training phase, we compute the per-example cross-entropy loss using teacher forcing, which allows parallelized forward passes that are significantly faster than autoregressive token-by-token generation. Furthermore, LOB-STER replaces repeated model inference with efficient vector similarity computations in the embedding space, avoiding the expensive autoregressive bottleneck entirely and ensuring scalability for large-scale document pools and high-parameter models.

As demonstrated in Table 1, LOBSTER consistently and significantly outperforms both BAS and DUAL in terms of computational efficiency (we omit random sampling as its selection overhead is negligible). For example, with BART-Base on AESLC, LOBSTER reduces selection time to just 0.6 seconds, compared to 46.1 seconds for BAS and 38.7 seconds for DUAL. This efficiency gap widens as the model and document length increase; on CNN/DM with PEGASUS-Large, BAS and DUAL require 1064.2 and 699.4 seconds respectively, whereas LOBSTER completes the selection in only 1.6 seconds. This shows that LOBSTER achieves a speedup of 665× over BAS and 437× over DUAL in this configuration, effectively reducing the selection latency from nearly 18 minutes to near real-time (approaching the near-zero overhead of the model-agnostic IDDS) and enabling LOBSTER’s viability for large-scale summarization applications. Crucially, LOBSTER achieves this near-zero latency without sacrificing the high summarization quality of uncertainty-based methods or suffering the cold-start instability of densitybased baselines like IDDS.

## 5.3 Ablation Studies

## 5.3.1 Impact of Loss-Based Hard Example Selection

To investigate the contribution of the loss-based hard example selection mechanism (Stage 1), we replaced the high-loss example selection with a random selection strategy. Specifically, instead of selecting the top-k examples with the highest generation loss from the labeled set ${ \mathcal { L } } ,$ we randomly sampled k examples as semantic anchors. The subsequent stages of LOBSTER, including densitybased pre-filtering and semantic hardness projection, remained unchanged. This ablation evaluates whether selecting high-loss examples provides an advantage over random anchor selection. As illustrated in Figure 5, on CNN/DM, LOBSTER outperforms the random-anchor variant across both BART and PEGASUS, suggesting that high-loss examples provide more effective semantic anchors for this dataset. In contrast, the performance gap on AESLC and XSum is relatively small, although standard LOBSTER maintains a slight advantage. This can be attributed to the fact that CNN/DM contains longer documents and summaries, making loss-guided example selection more beneficial. Since similar trends were observed across both ROUGE and BERTScore metrics, we report the BERTScore curves, which provide a complementary semantic evaluation of the generated summaries.

## 5.3.2 Ablation Study: Impact of IDDS

To isolate the contribution of the density-based prefiltering (Stage 2), we removed the representative candidate pool $( \mathcal { U } _ { r e p } )$ and applied the semantic hardness projection (Stage 3) directly to the entire unlabeled pool (U). We evaluated this ablation on AESLC and XSum to assess robustness across different summarization styles.

Visualization of Semantic Collapse To visualize the behavior of the selection mechanism (Stage 3) without filtering, we applied PCA to Sentence-BERT embeddings. Figure 6 plots selected samples (red) against the unlabeled pool (blue), revealing a critical failure mode: without IDDS, LOBSTER becomes concentrated in specific embedding regions. This reduces exploration of the data distribution and leads to redundant selections, particularly in datasets such as AESLC and XSum that contain many semantically similar examples. In contrast, pre-filtering with IDDS mitigates this issue by constructing a representative candidate pool $( U _ { \mathrm { r e p } } )$ that preserves broader coverage of the unlabeled distribution before semantic hardness projection. This diversity constraint prevents the selection process from repeatedly focusing on local clusters, enabling more informative use of the annotation budget.

## 5.4 Data Efficiency Setup

In preliminary experiments with a 150-sample budget, several configurations failed to reach 90% of full-dataset performance. Since such a small annotation budget is often unrealistic in practical settings, we expanded the budget for these underperforming setups (BART/AESLC, PEGASUS/AESLC, and BART/XSum) to 2,000 samples (B = 100 over 20 iterations). Configurations already achieving the 90% target (e.g., CNN/DM) required no expansion. Figure 4 and Appendix C detail the convergence rates and sample requirements needed to reach the 90% performance threshold, alongside a comprehensive runtime analysis of the selection process.

The Competitiveness of Random Sampling Table 7 in Appendix C reveals that the performance gap between random sampling and active learning methods narrows as the annotation budget increases. In several configurations, particularly on AESLC and XSum, random sampling matches or surpasses sophisticated active learning strategies. For example, in the BART-base/XSum setup, random sampling reaches the 90% threshold using only 1,600 samples, outperforming all other approaches. IDDS fails to reach this threshold in half of the evaluated scenarios, while BAS succeeds on the PEGASUS-large/CNN/DM setup at the cost of prohibitive computational latency. LOBSTER reaches the target in 5 out of 6 configurations, providing a balance between reliability and efficiency. This competitive behavior of random sampling at larger annotation budgets can be attributed to the relatively consistent structure of standard summarization benchmarks. As the annotation budget increases, larger uniformly sampled subsets are increasingly likely to capture the underlying document distribution, naturally reducing the advantage of more complex acquisition strategies. Consequently, active learning frameworks provide the greatest benefit under strict low-budget constraints, such as our primary 150-instance setting, where careful instance selection is critical for maximizing annotation efficiency.

## 5.5 Comparison with Modern LLMs

Given the increasing adoption of instruction-tuned large language models for summarization, we compare models fine-tuned using examples selected by LOBSTER (at a 150-instance annotation budget) against a zero-shot Llama-3-8B-Instruct baseline. As shown in Table 6 in Appendix C, LOBSTER remains highly competitive and, in several cases, outperforms the zero-shot LLM across both lexical (ROUGE) and semantic (BERTScore) metrics. These findings demonstrate that carefully selecting only 150 informative training examples through active learning is sufficient for a compact encoder-decoder model to achieve competitive summarization performance. Beyond accuracy, this approach offers important practical advantages, as such models typically require substantially less memory and lower inference latency than multibillion-parameter autoregressive LLMs and, when deployed locally, avoid the recurring token costs associated with API-based LLM services.

## 6 Conclusions and Future Work

This work demonstrated that the cross-entropy loss of labeled examples can serve as an effective acquisition signal for active learning in abstractive summarization. By identifying the model’s highloss examples from the labeled set, our approach targets the specific areas where the model struggles. Building on this, we select the unlabeled “semantic twins” of those high-loss cases so the model is explicitly guided to correct its specific weaknesses. Furthermore, we showed that using a density-based pre-filter (IDDS) to find a representative candidate pool before final selection prevents semantic collapse. Extensive evaluation across three benchmark datasets and two backbone architectures confirms that LOBSTER matches or outperforms stateof-the-art baselines while remaining significantly faster. Finally, our data efficiency analysis indicates that random sampling remains surprisingly competitive under large annotation budgets. As the budget increases, random selection captures enough of the underlying data distribution to yield strong summarization performance. As future work, we plan to explore the applicability of LOBSTER to other sequence-to-sequence generation tasks beyond abstractive summarization, as well as its adaptation to modern LLMs.

## Limitations

While our study demonstrates promising results, a few limitations remain. Our experiments are limited to English-language datasets (AESLC, XSum, and CNN/DailyMail), and therefore the effectiveness of LOBSTER for other languages remains unexplored. Languages with different morphology, syntax, or summarization styles may influence the behavior of embedding similarity and active learning selection strategies. Second, although we evaluate using both lexical (ROUGE) and semantic (BERTScore) metrics, automatic evaluation metrics may not fully capture aspects such as factual consistency, coherence, and overall summary quality. In addition, LOBSTER assumes that semantic similarity in the Sentence-BERT embedding space provides a useful proxy for transferring hardness from labeled to unlabeled examples. Although our empirical results support this design choice, investigating alternative retrieval spaces or jointly learned task-specific embeddings remains an important direction for future work. Finally, our experiments simulate human annotation by retrieving groundtruth summaries from existing datasets rather than collecting new human annotations, which may not fully reflect real-world annotation scenarios.

## Ethical Considerations

This work focuses on improving the efficiency of active learning for abstractive text summarization. The proposed method does not introduce new datasets, but it relies on pre-trained language models and existing benchmark corpora, which may contain biases present in their original training data. As a result, biases in the underlying models or datasets could influence the selection of instances during the active learning process and may propagate to downstream summarization systems. Future work could explore bias-aware acquisition strategies or incorporate fairness constraints when selecting samples for annotation.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Abdul Hameed Azeemi, Ihsan Ayyub Qazi, and Agha Ali Raza. 2025. To label or not to label: Hybrid active learning for neural machine translation. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 3071–3082, Abu Dhabi, UAE. Association for Computational Linguistics.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In Proceedings ofthe 3rd International Conference on Learning Representations (ICLR).

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, and 16 others. 2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53.

Liat Ein-Dor, Alon Halfon, Ariel Gera, Eyal Shnarch, Lena Dankin, Leshem Choshen, Marina Danilevsky, Ranit Aharonov, Yoav Katz, and Noam Slonim. 2020. Active Learning for BERT: An Empirical Study. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7949–7962, Online. Association for Computational Linguistics.

Yoav Freund and Robert E Schapire. 1997. A decisiontheoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1):119–139.

Alexios Gidiotis and Grigorios Tsoumakas. 2022. Should we trust this summary? Bayesian abstractive summarization to the rescue. In Findings of the Associationfor Computational Linguistics: ACL 2022, pages 4119–4131, Dublin, Ireland. Association for Computational Linguistics.

Alexios Gidiotis and Grigorios Tsoumakas. 2024. Bayesian active summarization. Computer Speech & Language, 83:101553.

Petros Stylianos Giouroukis, Alexios Gidiotis, and Grigorios Tsoumakas. 2025. Dual: Diversity and uncertainty active learning for text summarization. Preprint, arXiv:2503.00867.

David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda. 2003. English gigaword. Linguistic Data Consortium, Philadelphia, 4(1):34.

Karl Moritz Hermann, Tomáš Kociský, Edward Grefen-ˇ stette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 1, NIPS’15, page 1693–1701, Cambridge, MA, USA. MIT Press.

Urvashi Khanna, Samira Ghodratnama, Diego Mollá, and Amin Beheshti. 2022. Transformer-based models for long document summarisation in financial domain. In Proceedings ofthe 4th Financial Narrative Processing Workshop @LREC2022, pages 73– 78, Marseille, France. European Language Resources Association.

Seokhwan Kim, Yu Song, Kyungduk Kim, Jeong-Won Cha, and Gary Geunbae Lee. 2006. MMR-based active machine learning for bio named entity recognition. In Proceedings of the Human Language Technology Conference of the NAACL, Companion Volume: Short Papers, pages 69–72, New York City, USA. Association for Computational Linguistics.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Dongyuan Li, Ying Zhang, Zhen Wang, Shiyin Tan, Satoshi Kosugi, and Manabu Okumura. 2024. Active learning for abstractive text summarization via LLMdetermined curriculum and certainty gain maximization. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 8959–8971, Miami, Florida, USA. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Katerina Margatina, Giorgos Vernikos, Loïc Barrault, and Nikolaos Aletras. 2021. Active learning by acquiring contrastive examples. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 650–663, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, Çaglar Gu ˘ ˙lçehre, and Bing Xiang. 2016. Abstractive text summarization using sequence-to-sequence RNNs and beyond. In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, pages 280–290, Berlin, Germany. Association for Computational Linguistics.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1797–1807, Brussels, Belgium. Association for Computational Linguistics.

Yotam Perlitz, Ariel Gera, Michal Shmueli-Scheuer, Dafna Sheinwald, Noam Slonim, and Liat Ein-Dor. 2023. Active learning for natural language generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9862–9877, Singapore. Association for Computational Linguistics.

Puria Radmard, Yassir Fathullah, and Aldo Lipani. 2021. Subsequence based deep active learning for named entity recognition. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4310–4321, Online. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. Get to the point: Summarization with pointergenerator networks. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1073– 1083, Vancouver, Canada. Association for Computational Linguistics.

Burr Settles. 2012. Active Learning. Synthesis Lectures on Artificial Intelligence and Machine Learning. Springer Cham, Cham. EBook published 31 May 2022.

Artem Shelmanov, Dmitri Puzyrev, Lyubov Kupriyanova, Denis Belyakov, Daniil Larionov, Nikita Khromov, Olga Kozlova, Ekaterina Artemova, Dmitry V. Dylov, and Alexander Panchenko. 2021. Active learning for sequence tagging with deep pre-trained models and Bayesian uncertainty estimates. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 1698–1712, Online. Association for Computational Linguistics.

Yanyao Shen, Hyokun Yun, Zachary Lipton, Yakov Kronrod, and Animashree Anandkumar. 2017. Deep active learning for named entity recognition. In Proceedings of the 2nd Workshop on Representation Learningfor NLP, pages 252–256, Vancouver, Canada. Association for Computational Linguistics.

Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. 2016. Training region-based object detectors with online hard example mining. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 761–769.

Aditya Siddhant and Zachary C. Lipton. 2018. Deep Bayesian active learning for natural language processing: Results of a large-scale empirical study. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2904–2909, Brussels, Belgium. Association for Computational Linguistics.

Ilya Sutskever, Oriol Vinyals, and Quoc V Le. 2014. Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27.

Akim Tsvigun, Ivan Lysenko, Danila Sedashov, Ivan Lazichny, Eldar Damirov, Vladimir Karlov, Artemy Belousov, Leonid Sanochkin, Maxim Panov, Alexander Panchenko, Mikhail Burtsev, and Artem Shelmanov. 2022. Active learning for abstractive text summarization. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5128–5152, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Tim Z. Xiao, Aidan N. Gomez, and Yarin Gal. 2020. Wat zei je? detecting out-of-distribution translations with variational transformers. Preprint, arXiv:2006.08344.

Wen Xiao, Iz Beltagy, Giuseppe Carenini, and Arman Cohan. 2022. PRIMERA: Pyramid-based masked sentence pre-training for multi-document summarization. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5245–5263, Dublin, Ireland. Association for Computational Linguistics.

Yue Yu, Lingkai Kong, Jieyu Zhang, Rongzhi Zhang, and Chao Zhang. 2022. AcTune: Uncertainty-based active self-training for active fine-tuning of pretrained language models. In Proceedings ofthe 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1422–1436, Seattle, United States. Association for Computational Linguistics.

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. 2020. Big bird: Transformers for longer sequences. In Advances in Neural Information Processing Systems, volume 33, pages 17283–17297. Curran Associates, Inc.

Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020a. PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 11328–11339. PMLR.

Rui Zhang and Joel Tetreault. 2019. This email could save your life: Introducing the task of email subject line generation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 446–456, Florence, Italy. Association for Computational Linguistics.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020b. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

Zhisong Zhang, Emma Strubell, and Eduard Hovy. 2022. A survey of active learning for natural language processing. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 6166–6190, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

## Appendix

## A Dataset Statistics and Model Hyperparameters

In this section, we provide additional details regarding the datasets and models used in our experiments. Table 2 reports dataset statistics, including the number of instances and the average lengths of source documents and summaries for both training and test splits. Table 3 provides the hyperparameters used for fine-tuning the backbone summarization models.

Table 2: Statistics for the datasets used in this work. We report the number of instances (# Ins.) and the average length of source documents (Doc. len.) and summaries (Sum. len.) for both training and test splits.
<table><tr><td>Dataset</td><td>Subset</td><td># Ins.</td><td>Doc. len.</td><td>Sum. len.</td></tr><tr><td rowspan="2">Gigaword</td><td>Train</td><td>200</td><td>39.6</td><td>11.2</td></tr><tr><td>Test</td><td>1K</td><td>38.3</td><td>10.4</td></tr><tr><td rowspan="2">AESLC</td><td>Train</td><td>14,436</td><td>115.3</td><td>4</td></tr><tr><td>Test</td><td>1,906</td><td>114.6</td><td>4.1</td></tr><tr><td rowspan="2">XSum</td><td>Train</td><td>204,045</td><td>388.6</td><td>23.1</td></tr><tr><td>Test</td><td>1K</td><td>384.3</td><td>23</td></tr><tr><td rowspan="2">CNN/DM</td><td>Train</td><td>287,113</td><td>692.4</td><td>51.6</td></tr><tr><td>Test</td><td>1K</td><td>676.1</td><td>55.2</td></tr></table>

Table 3: Hyperparameters used for fine-tuning the backbone summarization models.
<table><tr><td>Hyperparameter</td><td>BART-base</td><td>PEGASUS-large</td></tr><tr><td>Learning Rate</td><td> $2 e ^ { - 5 }$ </td><td> $5 e ^ { - 5 }$ </td></tr><tr><td>Batch Size (Train)</td><td>16</td><td>8</td></tr><tr><td>Batch Size (Eval)</td><td>32</td><td>16</td></tr><tr><td>Num. Epochs</td><td>6</td><td>4</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Weight Decay</td><td>0.028</td><td>0.03</td></tr><tr><td>Warmup Ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>Beam Size</td><td>4</td><td>4</td></tr><tr><td>Min. Train Steps</td><td>350</td><td>200</td></tr><tr><td>Grad. Accumulation</td><td>1</td><td>1</td></tr></table>

In this work, we utilize standard public datasets (AESLC, XSum, CNN/DailyMail) and pre-trained models (e.g., BART, PEGASUS). These artifacts are established benchmarks within the NLP community and are used for academic benchmarking and summarization evaluation. All datasets and models are used in accordance with their respective open-source licenses and research usage guidelines. Because our core contribution focuses on an algorithmic active learning strategy rather than novel dataset curation, we relied on the standard versions and original preprocessing of these datasets. Consequently, we did not perform additional data collection. We also did not apply further filtering for personally identifiable information or offensive content beyond the preprocessing steps already applied by the original dataset creators.

All experiments were conducted using publicly available implementations of the selected backbone models through the HuggingFace Transformers library. Training and evaluation procedures follow standard fine-tuning practices for abstractive summarization models. Hyperparameters were selected based on preliminary tuning experiments and remain consistent across all active learning strategies to ensure a fair comparison.

To promote reproducibility, all code created for this study, including the full experimental pipeline and the implementation of the proposed active learning method, is provided anonymously solely for the intended purpose of peer-review.

## B Summarization Performance

Table 4 reports summarization performance of all methods under a 150-instance annotation budget across AESLC, XSum, and CNN/DailyMail using ROUGE and BERTScore. Statistical significance tests between LOBSTER and baselines are reported in Table 5.

## C Detailed Data Efficiency Setup Results

In this section, we provide additional results for the data efficiency analysis discussed in Section 5.4. Table 7 reports the number of annotated samples required for each method to reach 90% of the full dataset performance across different datasets and backbone models. Additionally, Table 8 presents the average runtime required for instance selection during a single active learning iteration, highlighting the computational efficiency of the compared strategies.

These results are consistent with the trends discussed in the main text. In particular, Table 7 highlights the strong competitiveness of random sampling under larger annotation budgets, while LOB-STER also achieves strong and consistent performance across most setups. At the same time, LOB-STER provides a balance between performance reliability and computational efficiency compared to uncertainty-based approaches, as shown in Table 8.

Table 4: Summarization performance (ROUGE and rescaled BERTScore) with a 150-instance annotation budget. Results are reported as mean ± standard deviation. The best performance for each metric and dataset is highlighted in bold.
<table><tr><td>Model</td><td>Method</td><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>BERTScore</td></tr><tr><td colspan="6">Dataset: AESLC</td></tr><tr><td rowspan="5">BART-base</td><td>Random</td><td> $2 6 . 6 { \pm } 0 . 6 5$ </td><td> $1 3 . 2 { \pm } 0 . 6 5 $ </td><td> $2 6 . 0 { \pm } 0 . 6 3$ </td><td> $1 3 . 5 1 { \pm } 0 . 5 7 $ </td></tr><tr><td>BAS</td><td> $2 6 . 5 { \pm } 0 . 1 7$ </td><td> $1 3 . 1 { \pm } 0 . 3 1 $ </td><td> $2 5 . 8 { \pm } 0 . 2 4 $ </td><td> $1 3 . 6 3 { \scriptstyle \pm 0 . 9 9 }$ </td></tr><tr><td>IDDS</td><td> $2 7 . 0 { \pm } 0 . 3 2 $ </td><td> $1 2 . 5 { \pm } 0 . 3 2 $ </td><td> $2 5 . 4 { \pm } 0 . 3 1$ </td><td> $1 3 . 0 3 { \pm } 0 . 1 9$ </td></tr><tr><td>DUAL</td><td> $2 6 . 9 { \pm } 0 . 8 6 $ </td><td> $1 3 . 2 { \pm } 0 . 7 0 $ </td><td> $2 6 . 0 { \pm } 0 . 8 5 $ </td><td> $1 2 . 9 5 { \pm } 0 . 1 4$ </td></tr><tr><td>LOBSTER</td><td> $2 7 . 1 { \pm } 0 . 8 8 $ </td><td> $1 3 . 4 { \pm } 0 . 4 5 $ </td><td> $2 6 . 3 { \pm } 0 . 8 0 $ </td><td> $1 2 . 9 6 { \pm } 0 . 6 5$ </td></tr><tr><td rowspan="5">PEGASUS-large</td><td>Random</td><td> $2 1 . 1 { \pm } 0 . 5 4 $ </td><td> $9 . 9 { \pm } 0 . 4 8 $ </td><td> $2 0 . 0 { \pm } 0 . 5 5 $ </td><td> $6 . 1 4 { \pm } 0 . 6 8$ </td></tr><tr><td>BAS</td><td> $1 9 . 6 { \pm } 1 . 3 4 $ </td><td> $9 . 0 { \pm } 0 . 8 6 $ </td><td> $1 8 . 9 { \pm } 1 . 3 6 $ </td><td> $4 . 8 2 { \pm } 1 . 3 6 $ </td></tr><tr><td>IDDS</td><td> $2 2 . 6 { \pm } 0 . 4 2$ </td><td> ${ \bf 1 1 . 0 { \pm } } 0 . 3 4$ </td><td> $2 1 . 5 { \pm } 0 . 4 7 $ </td><td> $\mathbf { 7 . 7 0 \pm 0 . 5 9 }$ </td></tr><tr><td>DUAL</td><td> $2 1 . 2 { \pm } 1 . 1 5$ </td><td> $1 0 . 1 { \pm } 0 . 4 7$ </td><td> $2 0 . 5 { \pm } 1 . 0 8 $ </td><td> $6 . 2 9 { \pm } 0 . 9 0$ </td></tr><tr><td>LOBSTER</td><td> $2 2 . 1 { \pm } 1 . 0 1 $ </td><td> $1 0 . 9 { \pm } 0 . 6 1 $ </td><td> $2 1 . 5 { \pm } 0 . 9 4 $ </td><td> $7 . 6 6 { \pm } 0 . 5 5$ </td></tr><tr><td colspan="6">Dataset: XSum</td></tr><tr><td rowspan="5">BART-base</td><td>Random</td><td> $3 2 . 6 { \pm } 0 . 3 6 $ </td><td> $1 0 . 7 { \pm } 0 . 3 8 $ </td><td> $2 5 . 3 { \pm } 0 . 3 8 $ </td><td> $3 7 . 2 7 { \pm } 0 . 3 5 $ </td></tr><tr><td>BAS</td><td> $3 2 . 3 { \pm } 0 . 1 5 $ </td><td> $1 0 . 6 { \pm } 0 . 1 4$ </td><td> $2 5 . 1 { \pm } 0 . 2 3 $ </td><td> $3 6 . 9 3 { \pm } 0 . 3 0$ </td></tr><tr><td>IDDS</td><td> $3 2 . 6 { \pm } 0 . 1 4 $ </td><td> $1 0 . 6 { \pm } 0 . 1 0 $ </td><td> $2 5 . 0 { \pm } 0 . 1 7 $ </td><td> $3 6 . 7 1 { \pm } 0 . 1 5$ </td></tr><tr><td>DUAL</td><td> $3 2 . 9 { \pm } 0 . 1 7 $ </td><td> ${ \bf 1 0 . 8 \pm 0 . 1 5 }$ </td><td> $2 5 . 6 { \pm } 0 . 1 4$ </td><td> $3 7 . 4 2 { \scriptstyle \pm 0 . 2 8 }$ </td></tr><tr><td>LOBSTER</td><td> $3 2 . 5 { \pm } 0 . 2 1 $ </td><td> $1 0 . 6 { \pm } 0 . 2 1 $ </td><td> $2 5 . 3 { \pm } 0 . 2 8 $ </td><td> $3 7 . 0 0 { \scriptstyle \pm 0 . 2 2 }$ </td></tr><tr><td rowspan="5">PEGASUS-large</td><td>Random</td><td> $4 3 . 4 { \pm } 0 . 3 7 $ </td><td> $2 0 . 1 { \pm } 0 . 3 1$ </td><td> $3 4 . 9 { \pm } 0 . 5 0 $ </td><td> $\mathbf { 4 8 . 0 0 } \pm 0 . 3 7$ </td></tr><tr><td>BAS</td><td> $4 3 . 3 { \pm } 0 . 3 0 $ </td><td> $2 0 . 2 { \pm } 0 . 3 3$ </td><td> $3 4 . 9 { \pm } 0 . 3 6 $ </td><td> $4 7 . 7 1 { \pm } 0 . 4 0 $ </td></tr><tr><td>IDDS</td><td> $4 2 . 8 { \pm } 0 . 1 3$ </td><td> $1 9 . 6 { \pm } 0 . 1 2$ </td><td> $3 4 . 2 { \pm } 0 . 1 1$ </td><td> $4 7 . 2 8 { \pm } 0 . 2 4 $ </td></tr><tr><td>DUAL</td><td> $4 3 . 3 { \pm } 0 . 2 4 $ </td><td> $2 \mathbf { 0 . 3 \pm } 0 . 3 0 $ </td><td> ${ \bf 3 5 . 0 { \pm } } 0 . 4 2$ </td><td> $4 7 . 7 4 { \pm } 0 . 3 8 $ </td></tr><tr><td>LOBSTER</td><td> ${ \pm } 3 . 5 { \pm } 0 . 4 9 $ </td><td> $2 0 . 3 { \pm } 0 . 4 9$ </td><td> ${ \bf 3 5 . 0 { \pm } } 0 . 4 7 $ </td><td> $4 7 . 7 2 { \pm } 0 . 3 5 $ </td></tr><tr><td colspan="6">Dataset: CNN/DM</td></tr><tr><td rowspan="5">BART-base</td><td>Random</td><td> $\mathbf { 3 9 . 5 } { \pm } 0 . 2 4 $ </td><td> $1 7 . 0 { \pm } 0 . 1 8$ </td><td> $2 6 . 3 { \pm } 0 . 1 8$ </td><td> $2 9 . 3 5 { \pm } 0 . 2 4 $ </td></tr><tr><td>BAS</td><td> $\mathbf { 3 9 . 5 } { \pm } 0 . 2 2$ </td><td> $1 7 . 0 { \pm } 0 . 1 9$ </td><td> $2 6 . 6 { \pm } 0 . 1 6$ </td><td> $2 9 . 4 2 { \pm } 0 . 1 8$ </td></tr><tr><td>IDDS</td><td> $3 9 . 2 { \pm } 0 . 2 9 $ </td><td> $1 6 . 5 { \pm } 0 . 1 8 $ </td><td> $2 6 . 2 { \pm } 0 . 1 4 $ </td><td> $2 9 . 4 0 { \pm } 0 . 1 8$ </td></tr><tr><td>DUAL</td><td> $3 9 . 4 { \pm } 0 . 6 6$ </td><td> $1 7 . 1 { \pm } 0 . 3 7 $ </td><td> $\pm 6 . 6 { \pm } 0 . 3 2 $ </td><td> ${ \pm 9 . 4 4 \pm 0 . 8 5 }$ </td></tr><tr><td>LOBSTER</td><td> $3 9 . 4 { \pm } 0 . 5 8 $ </td><td> $1 7 . 0 { \pm } 0 . 4 8 $ </td><td> $2 6 . 4 { \pm } 0 . 3 9$ </td><td> $2 9 . 3 7 { \pm } 0 . 2 0 $ </td></tr><tr><td rowspan="5">PEGASUS-large</td><td>Random</td><td> $3 9 . 2 { \pm } 0 . 2 4 $ </td><td> $1 7 . 0 { \pm } 0 . 1 5 $ </td><td> $2 6 . 3 { \pm } 0 . 2 0 $ </td><td> $2 6 . 4 0 { \pm } 0 . 1 5$ </td></tr><tr><td>BAS</td><td> $\mathbf { 3 9 . 4 } \pm 0 . 3 2$ </td><td> $1 7 . 1 { \pm } 0 . 2 0 $ </td><td> $2 6 . 2 { \pm } 0 . 1 7$ </td><td> $2 6 . 1 7 { \scriptstyle \pm 0 . 3 2 }$ </td></tr><tr><td>IDDS</td><td> $3 8 . 8 { \pm } 0 . 3 3 $ </td><td> $1 6 . 8 { \pm } 0 . 2 0 $ </td><td> $2 6 . 2 { \pm } 0 . 2 0 $ </td><td> $2 5 . 3 7 { \pm } 0 . 3 0$ </td></tr><tr><td>DUAL</td><td> $3 9 . 2 \pm 0 . 3 7$ </td><td> $1 7 . 1 { \pm } 0 . 1 8$ </td><td> $2 6 . 4 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $\pm 6 . 4 9 { \pm } 0 . 3 3$ </td></tr><tr><td>LOBSTER</td><td> $3 9 . 2 { \pm } 0 . 4 3 $ </td><td> $1 6 . 9 { \pm } 0 . 2 2$ </td><td> $2 6 . 2 { \pm } 0 . 2 0 $ </td><td> $2 5 . 9 6 { \pm } 0 . 4 8$ </td></tr></table>

Table 5: Paired permutation test p-values comparing LOBSTER against baseline methods at the 150-instance annotation budget for ROUGE-1 and BERTScore. An asterisk (\*) indicates a statistically significant difference $( p < 0 . 0 5 )$ . For significant results, arrows indicate whether LOBSTER scored significantly higher (↑) or lower (↓) than the baseline.
<table><tr><td colspan="2"></td><td colspan="2"></td><td colspan="2">LOBSTER vs. Baseline (p-value)</td></tr><tr><td>Dataset</td><td>Model</td><td>DUAL</td><td>Random</td><td>IDDS</td><td>BAS</td></tr><tr><td>ROUGE-1</td><td></td><td>0.1681</td><td>0.4250</td><td>0.0083*↑</td><td> $0 . 0 2 0 3 ^ { \ast } \uparrow$ </td></tr><tr><td>AESLC</td><td>BART-base PEGASUS-large</td><td>&lt;0.001*↑</td><td>&lt;0.001*↑</td><td>0.5113</td><td> ${ < } 0 . 0 0 1 ^ { * } \mathrm  ~ \}$ </td></tr><tr><td>XSum</td><td>BART-base PEGASUS-large</td><td>0.0224*↓ 0.2192</td><td>0.1649 0.8924</td><td>0.6326  ${ < } 0 . 0 0 1 ^ { * } \uparrow$ </td><td>0.2942 0.2532</td></tr><tr><td>CNN/DM</td><td>BART-base PEGASUS-large</td><td>0.1807 &lt;0.001*↓</td><td>0.0722  ${ < } 0 . 0 0 1 ^ { * } \downarrow$ </td><td>0.8836 0.2663</td><td> $0 . 0 1 0 6 ^ { \ast } \downarrow$   ${ < } 0 . 0 0 1 ^ { * } \downarrow$ </td></tr><tr><td>BERTScore</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AESLC</td><td>BART-base PEGASUS-large</td><td>0.3666 &lt;0.001*↑</td><td> $0 . 0 0 7 4 ^ { \ast } \downarrow$   ${ < } 0 . 0 0 1 ^ { * } \mathrm { ~ \dot { \uparrow } ~ }$ </td><td>0.3685 0.9037</td><td> $0 . 0 0 1 3 ^ { \ast } \downarrow$   ${ < } 0 . 0 0 1 ^ { * } \mathrm  ~ \}$ </td></tr><tr><td>XSum</td><td>BART-base</td><td>0.0150*↓</td><td>0.1250</td><td>0.2651</td><td>0.7348</td></tr><tr><td>CNN/DM</td><td>PEGASUS-large BART-base</td><td>0.9361  $\begin{array} { c } { < 0 . 0 0 1 ^ { \ast } \downarrow } \\ { < 0 . 0 0 1 ^ { \ast } \downarrow } \end{array}$ </td><td>0.0946  $\begin{array} { c } { { < 0 . 0 0 1 ^ { * } \uparrow } } \\ { { < 0 . 0 0 1 ^ { * } \downarrow } } \end{array}$ </td><td> $0 . 0 3 3 8 ^ { \ast } \uparrow$   $^ { < 0 . 0 0 1 ^ { * } \downarrow } _ { 0 . 6 1 1 3 }$ </td><td>0.9475  ${ < } 0 . 0 0 1 ^ { * } \downarrow$ </td></tr></table>

Table 6: Performance comparison of models fine-tuned using examples selected by LOBSTER (at a 150-instance annotation budget) against a zero-shot Llama-3-8B-Instruct baseline. Bold values indicate the best performance for each metric within each dataset.
<table><tr><td>Dataset</td><td>Setup</td><td>ROUGE (1 / 2 / L)</td><td>BERTScore</td></tr><tr><td rowspan="3">AESLC</td><td>Llama-3-8B-Instruct (Zero-shot)</td><td>27.16 / 12.06 / 25.22</td><td>11.61</td></tr><tr><td>BART-base + LOBSTER</td><td>27.10 / 13.40 / 26.30</td><td>12.96</td></tr><tr><td>PEGASUS-large + LOBSTER</td><td>22.10 / 10.90 / 21.50</td><td>7.66</td></tr><tr><td rowspan="3">XSum</td><td>Llama-3-8B-Instruct (Zero-shot)</td><td>29.41 / 8.40 / 21.20</td><td>29.13</td></tr><tr><td>BART-base + LOBSTER</td><td>32.50 / 10.60 / 25.30</td><td>37.00</td></tr><tr><td>PEGASUS-large + LOBSTER</td><td>43.50 / 20.30 / 35.00</td><td>47.72</td></tr><tr><td rowspan="3">CNN/DM</td><td>Llama-3-8B-Instruct (Zero-shot)</td><td>39.39 / 15.61 / 24.72</td><td>26.28</td></tr><tr><td>BART-base + LOBSTER</td><td>39.40 / 17.00 / 26.40</td><td>29.37</td></tr><tr><td>PEGASUS-large + LOBSTER</td><td>39.20 / 16.90 / 26.20</td><td>25.96</td></tr></table>

Table 7: Data Efficiency Analysis: Number of annotated samples required to reach 90% of the full-dataset performance. Bold values indicate the winning strategy for each configuration, and a dash (—) indicates that the configuration failed to reach the 90% target.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="5">Active Learning</td></tr><tr><td>LOBSTER</td><td>DUAL</td><td>BAS</td><td>IDDS</td><td>Rand.</td></tr><tr><td rowspan="3">BART-base</td><td>AESLC</td><td>1900</td><td>1900</td><td>1300</td><td></td><td>2000</td></tr><tr><td>XSum</td><td>2000</td><td>1700</td><td>1700</td><td></td><td>1600</td></tr><tr><td>CNN/DM</td><td>50</td><td>60</td><td>110</td><td>100</td><td>100</td></tr><tr><td rowspan="3">PEGASUS-large</td><td>AESLC</td><td>1800</td><td>1800</td><td>2000</td><td>2000</td><td>1800</td></tr><tr><td>XSum</td><td>80</td><td>90</td><td>80</td><td>100</td><td>80</td></tr><tr><td>CNN/DM</td><td></td><td></td><td>150</td><td></td><td></td></tr></table>

Table 8: Data Efficiency Analysis: Average runtime (in seconds) for instance selection, for one active learning iteration, across different strategies. Comparison across backbones and datasets.
<table><tr><td colspan="2"></td><td colspan="4">Avg Selection time (s)</td></tr><tr><td rowspan="2">Model</td><td>Dataset</td><td>BAS</td><td>DUAL</td><td>IDDS</td><td>LOBSTER</td></tr><tr><td>AESLC</td><td>61.8</td><td>366.2</td><td>0.3</td><td>3.9</td></tr><tr><td rowspan="3">BART-base</td><td>XSum</td><td>96.1</td><td>560.5</td><td>2.8</td><td>12.1</td></tr><tr><td>CNN/DM</td><td>70.2</td><td>69.2</td><td>0.3</td><td>0.8</td></tr><tr><td>AESLC</td><td>164.6</td><td>618.1</td><td>0.3</td><td>12.8</td></tr><tr><td rowspan="3">PEGASUS-large</td><td>XSum</td><td>390.0</td><td>267.6</td><td>0.4</td><td>1.5</td></tr><tr><td>CNN/DM</td><td>1064.2</td><td>699.4</td><td>0.4</td><td>1.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/84582e8cce44a48f7a67ff4cc4efde72fdbcfa6bdecae18beb40d3e3d5034533.jpg)  
(A) BART on AESLC

![](images/394ac17c178861c8b020121f5df060ae2f69b35748de42d8dc8d6cc47a8baf2b.jpg)  
(B) BART on XSum

![](images/a97dd6b18ea5209eabbbd5ef1eddd22b03ec40f12a3e1905fe4b5e2b85578af4.jpg)  
(C) BART on CNN/DM

![](images/ab7fb16a058602acc0a890448cdcf082d9436efc0afeb786f83356794fd26c61.jpg)  
(D) PEGASUS on AESLC

![](images/c8020d6e0469ee53d9c23927632a60722cd78fcd49893da733d48e65650bc6f3.jpg)  
(E) PEGASUS on XSum

![](images/a3104651f65b123d7d410d32777e5110ec26531816165175b69ce332bf2812e4.jpg)  
(F) PEGASUS on CNN/DM  
Figure 3: Number of labeled instances vs. rescaled BERTScore. Comparison of active strategies using BART and PEGASUS backbones on AESLC, XSum, and CNN/DM datasets.

![](images/bd22a5ae35f4864ed52048cf4576eb78fb9c1a274626b600101c347522c8b772.jpg)  
(A) BART on AESLC

![](images/a3502da4673549c1fd96853801ee199664a5fdb075200439e93d67f80aba55fe.jpg)  
(B) BART on XSum

![](images/5e7477b12f5ee11d01127bef413b15007ac779f150900bae80d8f9af3fbea627.jpg)  
(C) BART on CNN/DM

![](images/76bb7ae1dd0f44bb01538618c1462722901491adee4aaeab528d59fb8b781530.jpg)  
(D) PEGASUS on AESLC

![](images/5ff26f213abbe03969f0fe15573bc5068f46737ce8262604bdffcb61552e8c46.jpg)  
(E) PEGASUS on XSum

![](images/9a07c40d690ab9f80d6e2a8b86e1bbd6b5fd01e3fd6bda8e8c03a56077b2b0b0.jpg)  
(F) PEGASUS on CNN/DM  
Figure 4: Number of labeled instances vs. ROUGE-1. Comparison of active learning strategies using BART and PEGASUS backbones on AESLC, XSum, and CNN/DM datasets. The dashed horizontal line in each subplot denotes the 90% performance threshold of the full-dataset baseline.

![](images/78cfab9fc41d7eff434c6fbcb66ed14b714a061797d53c402f895d29e7d965a4.jpg)  
(A) BART on AESLC

![](images/776fda74049c0552a5acc9355abbd7741d57ba6b1ba9869f9da38cb29857d22e.jpg)  
(B) BART on XSum

![](images/b3b059c3e60f17d07212a053f5e4b1444798d9a5a2d43728e835b99501439b61.jpg)  
(C) BART on CNN/DM

![](images/2a02ea919ebdffd930b1f55d3906aff53050dcd39f351216f3a9feb6a242360d.jpg)  
(D) PEGASUS on AESLC

![](images/5bca10411c349e1da4b0713d647cc5f088b2b36db03292d8e19444252f1ea872.jpg)  
(E) PEGASUS on XSum

![](images/0eb54433d270cdcb70a94e79bf56cb38b8533c922d87bfcdaa07584cf7f0ecaa.jpg)  
(F) PEGASUS on CNN/DM

Figure 5: Number of labeled instances vs. BERTScore. Comparison of LOBSTER and the LOBSTER (Random Anchors) ablation using BART and PEGASUS backbones on AESLC, XSum, and CNN/DM datasets.

ROUGE-1: 25.3  
![](images/78cb71d96bbdaaa779a0058d9c1120c60c078d5b50acaa83d9aea2c4a53aac74.jpg)  
(a) AESLC, without IDDS

![](images/454bbbc46d7feda16286deeaddfbe51d68176e89f2ae9ca3629c9ba7926c1df4.jpg)  
ROUGE-1: 27.1  
(b) AESLC, with IDDS

ROUGE-1: 30.6  
![](images/ebd93b3551e26050179a5565b56d091c6092e06065a6f4863cb9a789184a6533.jpg)  
(c) XSum, without IDDS

ROUGE-1: 32.7  
![](images/d91e91c3f891145dbcb13a788cae736709d0f900f74fe6de921869846fb48b33.jpg)  
(d) XSum, with IDDS

Figure 6: Impact of IDDS on Selection Diversity (PCA visualization of selected instances). Comparison of the proposed LOBSTER method with IDDS versus the ablation without IDDS. The IDDS module prevents semantic collapse by filtering redundant candidates, leading to higher ROUGE-1 scores.