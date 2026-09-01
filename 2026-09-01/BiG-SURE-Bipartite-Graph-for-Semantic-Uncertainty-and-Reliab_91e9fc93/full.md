# BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs

Debarpan Bhattacharya<sup>\*</sup> Indian Institute of Science Bangalore, India debarpanb@iisc.ac.in

Malay Phadke<sup>\*</sup>   
Indian Institute of Science   
Bangalore, India   
malaymilindp@iisc.ac.in   
Sriram Ganapathy   
Indian Institute of Science   
Bangalore, India   
sriramg@iisc.ac.in

 https://iiscleap.github.io/projects/BiG-SURE | § https://github.com/iiscleap/BiG-SURE

## Abstract

Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BIG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model’s stable lowtemperature belief or not. We evaluate BIG-SURE on text QA, multilingual QA, and multimodal QA tasks across multiple model families. In these experiments, BIG-SURE improves average abstention AUROC over prior black-box uncertainty estimators, while remaining simple, unsupervised, and applicable to black-box model settings.

## 1 Introduction

The development of large language models (LLMs)- ChatGPT (Achiam et al., 2023), Gemini (Team et al., 2023), Llama (Touvron et al., 2023), Mistral (Jiang et al., 2023), Qwen (Qwen et al., 2025) and others - has advanced various natural language tasks. However, multimodal and multilingual domains still underperform the English textual counterparts, affected by hallucinations and mispredictions (Wang et al., 2025; Song et al., 2025). Significant efforts in alignment, grounding, and training on large-scale data are underway to address this gap. However, in safety-critical scenarios, like finance, legal and health, the mis-predictions and hallucinations of the LLMs can result in substantial loss and hence, calibration and uncertainty measures are the key bottlenecks that dictate LLM deployment readiness (Bedi et al., 2025).

![](images/969e3377c571b2f4ab15ba54734f82d49f1941e8baa291109bd1425b475ba417.jpg)

![](images/ba38c4c80e851a9d2da7cd62ad8c466b98da3d38f4ad308d2d2eb6bf939180d6.jpg)  
Figure 1: (Top-panel) Bipartite semantic agreement between cross-temperature responses for an example input. (Bottom-panel) SURE gives a substantial improvement in abstention AUROC over prior works on Trivia-QA. The AUROCs are averaged over 6 LLMs (Table 2).

The early black-box approaches for uncertainty estimation focus on the semantic clustering and entropy (Kuhn et al., 2023; Farquhar et al., 2024). Further improvements are achieved using Von-Neumann entropy on kernelized response similarities (Nikitin et al., 2024), that are devoid of the semantic clustering (Nguyen et al., 2025). The graph-based approaches consider the responses to be different nodes of a graph and determine the effective number of semantic clusters (Lin et al., 2024). However, all the existing approaches attempt to extract the signal of uncertainty from the responses sampled at a fixed temperature.

We propose BIG-SURE, a black-box uncertainty estimation framework based on crosstemperature bipartite semantic agreement. Lowtemperature responses are used as stable anchors, while high-temperature responses act as probes that reveal semantic drift. SURE constructs an anchor–probe bipartite semantic similarity graph and uses its normalized squared spectral energy as a confidence score, with uncertainty defined as its complement. The key contributions of this work are:

• We identify cross-temperature response $\mathbf { d y } .$ namics as a useful black-box signal for uncertainty estimation in LLMs and VLMs.

• We propose a bipartite anchor–probe formulation that compares stable low-temperature generations against diverse high-temperature probes.

• We define SURE as a normalized squared spectral-energy score over the anchor–probe similarity matrix, producing a bounded uncertainty estimate.

• We benchmark SURE across text-only, multilingual, and multimodal QA tasks, showing improved abstention AUROC over prior blackbox uncertainty estimators.

Figure 1 illustrates the intuition behind SURE. We sample low-temperature responses as stable anchors and high-temperature responses as diverse probes, then measure their bipartite semantic agreement. Reliable predictions exhibit strong crosstemperature agreement: the probes remain semantically aligned with the anchors despite surfacelevel variation. Uncertain or hallucinated predictions instead show semantic drift, weakening the anchor–probe agreement. SURE converts this cross-temperature agreement structure into a normalized confidence score, whose complement serves as uncertainty; the bottom panel of Figure 1 shows that this signal improves abstention performance over prior estimators.

## 2 Problem Statement

We consider black-box uncertainty estimation for an LLM parameterized by θ, where internal states, logits, and token probabilities are not accessible. Given an input x, the goal is to estimate an uncertainty score $\mathcal { U } ( x ; \theta )$ for the model’s generated response y.

Let $T > 0$ denote the sampling temperature used to generate multiple responses from the model. We define two response sets: a set of lowtemperature anchor responses $\mathcal { A } = \{ a _ { 1 } , . . . , a _ { M } \}$ and a set of high-temperature probe responses $\mathcal { H } = \{ h _ { 1 } , \ldots , h _ { N } \}$ . The probes may be sampled either from the original input x or from semantically equivalent input augmentations of $x ,$ such as paraphrases. Our objective is to estimate $\mathcal { U } ( x ; \theta )$ from the cross-temperature response sets (A, H).

## 3 Cross-temperature Spectral Energy

## 3.1 Definition

We define a weighted Bipartite Graph (BiG) $G =$ $( \mathcal { A } , \mathcal { H } , W )$ characterized by the Bi-adjacency Matrix $W \in \dot { \mathbb { R } } ^ { M \times N }$ , where $W _ { i j }$ represents the semantic similarity between anchor $a _ { i }$ and probe $h _ { j }$ :

$$
W _ { i j } = s \mathsf { e m } - s \mathsf { i m } ( a _ { i } , h _ { j } ) , \quad W _ { i j } \geq 0\tag{1}
$$

We hypothesize that the cross-temperature dynamics between anchor and probe responses capture the model uncertainty. We define the spectral energy of their agreements using W as,

$$
E _ { \mathsf { S U R E } } ( W ; \theta ) = \sum _ { i = 1 } ^ { m i n \{ M , N \} } \sigma _ { i } ^ { 2 } = \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { N } W _ { i j } ^ { 2 }\tag{2}
$$

where $\sigma _ { i }$ are singular values of $W .$ The equality shows that $E _ { \mathsf { S U R E } } ( W ; \theta )$ is same as the squared Frobenius norm of W. We define normalized confidence and compute the Semantic Uncertainty and Reliability Estimate (SURE) score as,

$$
E _ { \mathsf { n o r m } } ( W ; \theta ) = \frac { E _ { \mathsf { S U R E } } ( W ; \theta ) } { M N }\tag{3a}
$$

$$
\mathcal { U } _ { \sf n o r m } ( W ; \theta ) = 1 - E _ { \sf n o r m } ( W ; \theta )\tag{3b}
$$

Mathematically, the measure satisfies $0 \_ \leq$ $\mathcal { U } _ { \sf n o r m } ( W ; \theta ) \leq 1$ (Appendix A.1). We formally show that SURE also possesses the properties that make it a valid uncertainty measure (Section 3.4).

## 3.2 Illustration

As shown in Figure 2, when an LLM generates an accurate response to a factual question (in greedy decoding mode $( T = 0 ) )$ , the mutual relationship between low and high temperature responses exhibits different properties compared to the setting when the LLM generates an inaccurate response. When it generates an accurate response, the low and high-temperature responses are semantically similar, which indicates a low value of $\mathcal { U } _ { \mathrm { n o r m } }$ . On the other hand, when the LLM generates an incorrect response, the high-temperature responses vary significantly from the low-temperature responses, resulting in a high value of $\mathcal { U } _ { \mathrm { n o r m } }$

![](images/a7f107e36d50e18b4e38d2a2cab19dc43c88447d73288a6a884af79df8393cb8.jpg)  
Figure 2: Cross-temperature semantic agreement reflects prediction reliability. For a correct greedy prediction (left), low- and high-temperature samples are semantically consistent, resulting in strong bipartite connectivity and high normalized spectral energy, and vice-versa.

## 3.3 Implementation

Given an input prompt x, we sample M low-temperature responses at $T _ { L }$ and N hightemperature responses at $T _ { H }$ , with $T _ { H } ~ > ~ T _ { L }$ We construct a bipartite graph between crosstemperature samples, where the edge weights are based on the pairwise semantic similarity scores, and form the matrix $W \in \mathbb { R } ^ { M \times N }$ . We compute the SURE metric based on Eqn. 3. Note that the computation of uncertainty is unsupervised and does not require ground truth labels. The algorithmic implementation is shown in Algorithm 1.

## 3.4 SURE - A valid uncertainty estimator

We show that $\mathcal { U } _ { \sf n o r m } ( W ; \theta )$ is a valid estimator of semantic uncertainty for LLMs:

• Semantic identity: $\mathcal { U } _ { \sf n o r m } = 0$ iff all the sampled responses of an LLM are semantically similar.

• Semantic disjointness: $\mathcal { U } _ { \sf n o r m } = 1$ iff all the high temperature responses are semantically different from low temperature responses.

• Monotonicity: For a given number of sampling responses, $\mathcal { U } _ { \mathrm { n o r m } }$ increases if the number of se-

Algorithm 1 Computation of $\mathcal { U } _ { n o r m }$   
Require: Input prompt x; LLM θ; low tempera  
ture $T _ { L } ;$ high temperature $T _ { H } > T _ { L } ;$   
Require: # of low-temperature responses $M ;$ #   
of high-temperature responses $N ;$ semantic   
similarity operator sem-sim(·, ·).   
Ensure: Normalized uncertainty $\mathcal { U } _ { \sf n o r m } ( W ; \theta ) \in$   
[0, 1].   
1: $\mathcal { A } \gets \mathrm { S A M P L E } ( \theta , x , T _ { L } , M )$   
2: $\mathcal { H }  \mathrm { S A M P L E } ( \theta , x , T _ { H } , N )$   
3: Initialize $W \in \dot { \mathbb { R } } ^ { \dot { M } \times N }$   
4: for i ← 1 to $M , a _ { i } \in { \mathcal { A } }$ do   
5: for $j  1 \mathrm { t o } \ N , h _ { j } \in \mathcal { H }$ do   
6: $W _ { i j } \gets s e m \mathrm { - } s \mathrm { i } \mathsf { m } ( a _ { i } , h _ { j } )$   
7: end for   
8: end for   
9: $\begin{array} { r } { E _ { \mathsf { S U R E } } ( W ; \theta ) \gets \sum _ { i = 1 } ^ { \operatorname* { m i n } \{ M , N \} } \sigma _ { i } ( W ) ^ { 2 } } \end{array}$   
10: $E _ { \mathsf { n o r m } } ( W ; \theta ) \gets \frac { E _ { \mathsf { S U R E } } ( W ; \theta ) } { \boldsymbol { \imath } \boldsymbol { \imath } \boldsymbol { \imath } ^ { T } }$   
11: $\mathcal { U } _ { \sf { n o r m } } ( W ; \theta )  1 - E _ { \sf { n o r m } } ( W ; \theta )$   
12: return $\mathcal { U } _ { \sf n o r m } ( W ; \theta )$

mantically unique sequences increases in the prediction set. These properties are discussed in detail in Appendix A.1.

## 4 Prior Works

Several prior works attempt black-box uncertainty estimation for LLMs, like semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024), evidential semantic entropy (Kunitomo-Jacquin et al., 2026),

SPUQ (Gao et al., 2024), KLE (Nikitin et al., 2024), SNNE (Nguyen et al., 2025), graph-based (Lin et al., 2024) and distillation-based (Phillips et al., 2026) approaches. Only a handful of recent works attempt uncertainty estimation for multimodal LLMs; some of them are FESTA (Bhattacharya et al., 2025b), TREA (Bhattacharya et al., 2025a), Uncertainty-O (Zhang et al., 2025) and UMPIRE (Lau et al., 2025).

Discrete Semantic Entropy (DSE): Directly computing lexical entropy (Kadavath et al., 2022) is not applicable for LLM-generated sequences, as the lexical variability can obscure semantic uncertainty. Discrete Semantic Entropy (DSE) addresses this issue by performing semantic clustering and computing entropy over the clusters (Kuhn et al., 2023; Farquhar et al., 2024). In contrast, our work does not require semantic clustering.

Graph-based approaches: Another family of methods explores high-temperature sampled outputs as nodes in a graph, with edges defined by soft pairwise similarity. Various graph-derived uncertainty scores include NumSet (discretized similarity), EigV (largest eigenvalue of a degree-based matrix), and Deg (negative trace of the degree matrix), as studied in (Lin et al., 2024).

Lexical Similarity (LexSim): LexSim computes a pairwise similarity matrix using a lexical overlap metric such as ROUGE-L (Lin, 2004), and computes confidence as a normalized aggregate of offdiagonal similarities (Fomicheva et al., 2020).

Kernel Language Entropy (KLE): KLE constructs unit-trace semantic kernels capturing pairwise similarity between generations, and defines uncertainty via the von Neumann entropy of the kernel matrix (Nikitin et al., 2024).

Semantic Nearest Neighbour Entropy (SNNE): SNNE (Nguyen et al., 2025) avoids explicit clustering by operating directly on pairwise similarity of sampled responses. SNNE and KLE solely rely on high-temperature samples.

Table 1 highlights the main distinction between prior black-box uncertainty estimators and SURE. Prior approaches largely rely on fixed-temperature sampling responses and then derive uncertainty from them. SURE departs from this paradigm by using agreement dynamics between lowtemperature anchors and high-temperature probes to quantify uncertainty. SURE also incorporates semantically equivalent input augmentation within the same framework. To the best of our knowledge, SURE is the first attempt to leverage crosstemperature sampling dynamics in LLM responses for uncertainty quantification.

Table 1: Conceptual comparison of SURE with prior works on black-box, open-text uncertainty estimation.
<table><tr><td>Method</td><td>Sampling type</td><td>Input Aug.</td><td>Quantification</td></tr><tr><td>DSE</td><td>High-temperature sampling (fixed T)</td><td>No</td><td>Semantic entropy over clustered responses</td></tr><tr><td>Graph- based</td><td>High-temperature sampling (fixed T)</td><td>No</td><td>Graph statistics (e.g., degree, eigenvalue, number of sets)</td></tr><tr><td>LexSim</td><td>High-temperature sampling (fixed T)</td><td>No</td><td>Lexical similarity across sampled responses</td></tr><tr><td>KLE</td><td>High-temperature sampling (fixed T)</td><td>No</td><td>Von Neumann entropy of semantic kernel</td></tr><tr><td>SNNE</td><td>High-temperature sampling (fixed T)</td><td>No</td><td>Nearest-neighbour semantic entropy</td></tr><tr><td>SURE</td><td>Low-T anchors + high-T probes</td><td>Yes</td><td>Cross-temperature agreement</td></tr></table>

## 5 Experimental Setup

## 5.1 Tasks and models

Textual QA: We experiment with text-based factual question answering and math word question answering tasks. We use Trivia-QA (Joshi et al., 2017) and SVAMP (Patel et al., 2021) datasets. We use 6 LLMs for these datasets - Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct (Grattafiori et al., 2024), Qwen-2.5-7B-Instruct and Qwen-2.5- 72B-Instruct (Qwen et al., 2025), Gemma-3-12b-it and Gemma-3-27b-it (Team et al., 2025).

Multilingual QA: We experiment with multilingual SciQ (Xue et al., 2025) in 4 languages as English(en), French(fr), Japanese(ja) and Mandarin(zh). We use two LLMs - Aya-expanse-8b (Dang et al., 2024) and Apertus-8B-Instruct-2509 (Hernández-Cano et al., 2025), because of their explicit multilingual training and superior performance.

Multimodal QA: We use the OK-VQA dataset (Marino et al., 2019) for visual question-answering tasks. We use vision-language models (VLMs) - Pixtral-12B-2409 (Agrawal et al., 2024), Llavav1.6-mistral-7b-hf (Liu et al., 2023) and Qwen3- VL-8B-Instruct (Bai et al., 2025).

## 5.2 Sampling and input transformations

Most of the prior works on sampling-based uncertainty estimation use temperature sampling. Input transformations that do not change input semantics have also been proven to be effective (Gao et al., 2024). SURE uses semantics-preserving input transformations in addition to temperature sampling to obtain probe responses. For the original input x, M anchor responses are obtained by lowtemperature $( T _ { L } )$ sampling of x. N probe samples are generated by high-temperature $( T _ { H } )$ sampling of a mix of equivalent input transformations, $\{ x ^ { e } \}$ In our experiments, $M = 3 , N = 1 0 , T _ { L } = 0 . 1$ and $T _ { H } = 1 . 0$ are used across all models and tasks. The $N = 1 0$ probe samples are obtained from 5 input transformations and high-temperature sampling twice from each. The equivalent samples for the textual inputs correspond to paraphrasing of the input. For multimodal inputs, we use both question paraphrases and mild image perturbations: mean blur with kernel size 5, affine rotation by $\pm 1 0 ^ { \circ }$ and Gaussian pixel noise sampled from $\mathcal { N } ( 0 , 1 0 )$ followed by clipping to [0, 255]. The exact prompts and implementation details of these input transformations are in Appendix A.6.

## 5.3 Semantic similarity backend module

The semantic similarity function sem-sim $( a _ { i } , h _ { j } )$ in Equation 1 is implemented using NLI-based entailment module. We feed the anchor-probe pair $( a _ { i } , h _ { j } )$ to the entailment model and obtain the softmax over the logits. The corresponding entailment probability forms the edge weight:

$$
W _ { i j } = \mathrm { s e m - s i m } ( a _ { i } , h _ { j } ) = p _ { \mathrm { e n t a i l } } ( a _ { i } , h _ { j } ) \in [ 0 , 1 ] ,
$$

where larger values indicate stronger semantic agreement between the anchors and probes. For text-QA and multimodal-QA, we use a DeBERTalarge NLI model (He et al., 2021) <sup>1</sup> as the entailment backend. For multilingual QA, we use a multilingual DeBERTa NLI model (mDeBERTa <sup>2</sup>), which is trained on multilingual NLI data. We do not train or fine-tune either semantic-similarity model. Within each experimental setting, the entailment backend remains fixed across SURE and all similarity-based baselines for fair comparison.

## 5.4 Evaluation and performance metric

We use the area under the receiver operating characteristic curve (AUROC) to evaluate the uncertainty estimators. Each uncertainty score is interpreted as an abstention score. First, every test prediction is labeled as correct or incorrect based on the match between the greedy prediction $( T = 0$ response) and the ground-truth answer. The matching is based on the standard evaluation protocols for textual QA and multimodal QA tasks, and we use Gemini-2.5-Flash (Comanici et al., 2025) as a judge for multilingual QA (details in Appendix A.8). Then, the binary matching labels and abstention scores are used to compute the AUROC.

## 6 Results

## 6.1 Illustrative examples

Figure 3 provides examples of cross-temperature response patterns captured by SURE. Across multilingual and multimodal QA examples, we observe a clear separation between reliable and unreliable predictions in terms of cross-temperature semantic agreement dynamics. The quantitative evaluation of SURE in terms of abstention AUROC and a comparison with 7 existing works on black-box, opentext uncertainty quantification methods across multilingual and multimodal tasks is reported in Table 2. All prior works use $N = 1 0$ in Table 2.

## 6.2 Results on text-only QA tasks

On the text-only QA benchmarks, SURE achieves the best average AUROC on both datasets. On TRIVIA-QA, SURE obtains an average AUROC of $0 . 7 5 3 \pm 0 . 0 1 7 $ , improving over the strongest baseline average, Deg $( 0 . 7 2 2 \pm 0 . 0 0 7 )$ . On SVAMP, SURE obtains $0 . 8 7 7 \pm 0 . 0 2 2$ , outperforming the strongest baseline average, SNNE $( 0 . 8 4 6 \pm 0 . 0 1 8 )$ The gains are also consistent across model families and scales. On TRIVIA-QA, SURE achieves the best performance for 5 out of 6 models, with the only exception being Llama-70B, where Deg performs better. On SVAMP, SURE achieves the best results for 5 out of 6 models, with Qwen-72B being the exception where SumEigv is strongest. These results indicate that crosstemperature anchor–probe agreement provides a robust uncertainty signal across both factual QA and math word-problem settings. Graph-based baselines such as SumEigv and Deg are competitive, especially on TRIVIA-QA, while SNNE is a competitive baseline on SVAMP; however, SURE achieves the best average performance across both. To understand the corresponding abstention behavior, AURC values are also reported in Appendix A.9.

![](images/77ce850ecee61acb5c2dd2b8f1f70aec7449de9c2b979776cf38666cbbc4800e.jpg)  
Figure 3: Illustrative examples of cross-temperature semantic agreement in multilingual and multimodal QA. For each example, we show the greedy prediction together with low-temperature anchor samples and hightemperature probe samples. Responses aligned with the ground truth are marked in green, while responses that do not align are highlighted in red. Reliable predictions exhibit strong semantic agreement across temperatures, while unreliable predictions show semantic drift and disagreement.

## 6.3 Results on multilingual QA tasks

On multilingual SCIQ, SURE performs well across languages and LLMs, and achieves the best average AUROC on English $( 0 . 6 7 7 \pm 0 . 0 1 5 )$ , Japanese $( 0 . 6 8 5 \pm 0 . 0 1 3 )$ , and Mandarin $( 0 . 6 9 1 \pm 0 . 0 1 2 )$ These results improve over the best baseline systems in three language settings. French SCIQ is the only exception, where LexSim obtains the best average AUROC. Overall, SURE achieves the best average performance in 3 out of 4 multilingual language groups, and is especially consistent on Japanese and Mandarin for both Aya and Apertus.

## 6.4 Results on multimodal QA tasks

On OKVQA, SURE achieves the best average AUROC of $0 . 7 5 2 \pm 0 . 0 1 4 .$ outperforming the best baseline system, SNNE $( 0 . 7 3 5 \pm 0 . 0 1 6 )$ SURE is also the best method for all three VLMs. This suggests that cross-temperature semantic agreement remains useful even when the input contains visual modalities.

Summary. Across the seven dataset/language groups in Table 2, SURE achieves the best average AUROC in six groups: TRIVIA-QA, SVAMP, SCIQ en, SCIQ ja, SCIQ zh, and OKVQA. Overall, they support the central hypothesis that crosstemperature semantic agreement provides an effective black-box uncertainty signal across text-only, multilingual, and multimodal QA settings.

## 7 Discussion

## 7.1 Spectral mode analysis

The normalised spectral energy formulation of SURE enables the analysis in terms of singularvalue spectrum. If the singular values of $W \in$ $\mathbb { R } ^ { M \times N }$ are ordered as $\sigma _ { 1 } \ge \cdots \ge \sigma _ { \operatorname* { m i n } ( M , N ) }$ , we define truncated spectral-energy scores to understand the energy distribution,

$$
\begin{array} { l } { \displaystyle E _ { \mathrm { T o p } - k } ( W ) = \sum _ { r = 1 } ^ { k } \sigma _ { r } ^ { 2 } , } \\ { \displaystyle U _ { \mathrm { T o p } - k } ( W ) = 1 - \frac { E _ { \mathrm { T o p } - k } ( W ) } { M N } . } \end{array}
$$

Here, Top-1 uses only the largest singular value, while Top-2 uses the two largest singular values. Since our experimental setting uses M = 3 and N = 10, the rank of W is at most 3.

We find that Top-1 and Top-2 closely match the full SURE score across both SVAMP and Trivia-QA, as detailed in Appendix Table 3. This indicates that most of the useful cross-temperature agreement energy is concentrated in the leading singular mode. When the model is certain, hightemperature probes align with this dominant anchor mode. When the model is uncertain, the hightemperature probes drift away from the anchor semantics, weakening this dominant mode. This analysis clarifies the role of the spectral formulation.

Table 2: Performance comparison of SURE with prior works in terms of abstention AUROC across text-only, multimodal, and multilingual QA datasets. The third column reports the task accuracy of greedy responses. The abstention AUROCs are presented with repeated-run statistics (mean ± std) across 5 seeds. The best result, in terms of mean AUROC, is highlighted in bold, while the second best result is underlined.
<table><tr><td rowspan="2">Dataset [lang]</td><td rowspan="2">Model</td><td rowspan="2">Acc.</td><td colspan="7">Baselines (↑)</td><td rowspan="2">SURE (↑)</td></tr><tr><td>DSE</td><td>SumEigv</td><td>Deg</td><td>NumSet</td><td>LexSim</td><td>KLE</td><td>SNNE</td></tr><tr><td rowspan="7">Trivia-QA [en]</td><td>Qwen (7B)</td><td>0.495</td><td>0.786 ± 0.003</td><td>0.791 ± 0.005</td><td>0.793 ± 0.004</td><td>0.784 ± 0.005</td><td>0.778 ± 0.008</td><td>0.773 ± 0.007</td><td>0.783 ± 0.003</td><td>0.806 ± 0.014</td></tr><tr><td>Llama (8B)</td><td>0.661</td><td>0.731 ± 0.005</td><td>0.730 ± 0.005</td><td>0.740 ± 0.005</td><td>0.724 ± 0.007</td><td>0.736 ± 0.006</td><td>0.722 ± 0.003</td><td>0.736 ± 0.006</td><td>0.772 ± 0.021</td></tr><tr><td>Gemma (12B)</td><td>0.643</td><td>0.698 ± 0.010</td><td>0.697 ± 0.011</td><td>0.695 ± 0.011</td><td>0.700 ± 0.010</td><td>0.727 ± 0.014</td><td>0.689 ± 0.008</td><td>0.713 ± 0.014</td><td>0.789 ± 0.011</td></tr><tr><td>Gemma (27B)</td><td>0.755</td><td>0.629 ± 0.005</td><td>0.636 ± 0.005</td><td>0.634 ± 0.005</td><td>0.630 ± 0.005</td><td>0.657 ± 0.005</td><td>0.629 ± 0.006</td><td>0.617 ± 0.010</td><td>0.699 ± 0.017</td></tr><tr><td>Llama (70B)</td><td>0.767</td><td>0.712 ± 0.008</td><td>0.754 ± 0.011</td><td>0.756 ± 0.011</td><td>0.712 ± 0.008</td><td>0.737 ± 0.005</td><td>0.702 ± 0.010</td><td>0.715 ± 0.007</td><td>0.715 ± 0.009</td></tr><tr><td>Qwen (72B)</td><td>0.690</td><td>0.689 ± 0.012</td><td>0.722 ± 0.005</td><td>0.716 ± 0.007</td><td>0.680 ± 0.013</td><td>0.631 ± 0.009</td><td>0.695 ± 0.005</td><td>0.698 ± 0.012</td><td>0.740 ± 0.030</td></tr><tr><td>Avg.</td><td>0.668</td><td>0.708 ± 0.007</td><td>0.722 ± 0.007</td><td>0.722 ± 0.007</td><td>0.705 ± 0.008</td><td>0.711 ± 0.008</td><td>0.702 ± 0.006</td><td>0.710 ± 0.009</td><td>0.753 ± 0.017</td></tr><tr><td rowspan="7">SVAMP [en]</td><td>Qwen (7B)</td><td>0.688</td><td>0.794 ± 0.027</td><td>0.790 ± 0.019</td><td>0.789 ± 0.019</td><td>0.791 ± 0.027</td><td>0.780 ± 0.020</td><td>0.788 ± 0.018</td><td>0.803 ± 0.029</td><td>0.804 ± 0.017</td></tr><tr><td>Llama (8B)</td><td>0.672</td><td>0.857 ± 0.022</td><td>0.844 ± 0.025</td><td>0.845 ± 0.025</td><td>0.855 ± 0.022</td><td>0.845 ± 0.016</td><td>0.861 ± 0.019</td><td>0.888 ± 0.020</td><td>0.890 ± 0.026</td></tr><tr><td>Gemma (12B)</td><td>0.730</td><td>0.779 ± 0.016</td><td>0.787 ± 0.012</td><td>0.790 ± 0.013</td><td>0.779 ± 0.015</td><td>0.783 ± 0.013</td><td>0.781 ± 0.012</td><td>0.794 ± 0.015</td><td>0.882 ± 0.021</td></tr><tr><td>Gemma (27B)</td><td>0.794</td><td>0.735 ± 0.014</td><td>0.751 ± 0.015</td><td>0.752 ± 0.015</td><td>0.735 ± 0.014</td><td>0.734 ± 0.015</td><td>0.733 ± 0.015</td><td>0.796 ± 0.016</td><td>0.879 ± 0.017</td></tr><tr><td>Llama (70B)</td><td>0.797</td><td>0.884 ± 0.011</td><td>0.882 ± 0.011</td><td>0.884 ± 0.011</td><td>0.882 ± 0.010</td><td>0.868 ± 0.009</td><td>0.881 ± 0.013</td><td>0.896 ± 0.010</td><td>0.905 ± 0.013</td></tr><tr><td>Qwen (72B)</td><td>0.849</td><td>0.893 ± 0.021</td><td>0.915 ± 0.020</td><td>0.914 ± 0.018</td><td>0.880 ± 0.022</td><td>0.800 ± 0.014</td><td>0.814 ± 0.027</td><td>0.897 ± 0.016</td><td>0.902 ± 0.039</td></tr><tr><td>Avg.</td><td>0.755</td><td>0.823 ± 0.018</td><td>0.828 ± 0.017</td><td>0.829 ± 0.017</td><td>0.820 ± 0.018</td><td>0.802 ± 0.015</td><td>0.810 ± 0.017</td><td>0.846 ± 0.018</td><td>0.877 ± 0.022</td></tr><tr><td rowspan="3">SciQ [en]</td><td>Aya</td><td>0.786</td><td>0.591 ± 0.023</td><td>0.603 ± 0.040</td><td>0.613 ± 0.028</td><td>0.562 ± 0.026</td><td>0.591 ± 0.012</td><td>0.580 ± 0.008</td><td>0.537 ± 0.043</td><td>0.633 ± 0.004</td></tr><tr><td>Apertus</td><td>0.763</td><td>0.666 ± 0.047</td><td>0.682 ± 0.022</td><td>0.679 ± 0.014</td><td>0.603 ± 0.031</td><td>0.620 ± 0.020</td><td>0.631 ± 0.009</td><td>0.584 ± 0.035</td><td>0.722 ± 0.026</td></tr><tr><td>Avg.</td><td>0.775</td><td>0.628 ± 0.035</td><td>0.642 ± 0.031</td><td>0.646 ± 0.021</td><td>0.582 ± 0.028</td><td>0.606 ± 0.016</td><td>0.606 ± 0.008</td><td>0.561 ± 0.039</td><td>0.677 ± 0.015</td></tr><tr><td rowspan="3">SciQ [fr]</td><td>Aya</td><td>0.675</td><td>0.647 ± 0.028</td><td>0.631 ± 0.017</td><td>0.639 ± 0.015</td><td>0.634 ± 0.014</td><td>0.647 ± 0.013</td><td>0.611 ± 0.024</td><td>0.635 ± 0.015</td><td>0.639 ± 0.015</td></tr><tr><td>Apertus</td><td>0.668</td><td>0.535 ± 0.022</td><td>0.573 ± 0.013</td><td>0.563 ± 0.017</td><td>0.528 ± 0.032</td><td>0.569 ± 0.018</td><td>0.527 ± 0.032</td><td>0.534 ± 0.039</td><td>0.542 ± 0.022</td></tr><tr><td>Avg.</td><td>0.672</td><td>0.591 ± 0.025</td><td>0.602 ± 0.015</td><td>0.601 ± 0.016</td><td>0.581 ± 0.023</td><td>0.608 ± 0.016</td><td>0.569 ± 0.028</td><td>0.585 ± 0.027</td><td>0.590 ± 0.019</td></tr><tr><td rowspan="3">SciQ [ja]</td><td>Aya</td><td>0.539</td><td>0.586 ± 0.017</td><td>0.564 ± 0.033</td><td>0.573 ± 0.025</td><td>0.546 ± 0.027</td><td>0.508 ± 0.009</td><td>0.558 ± 0.030</td><td>0.541 ± 0.030</td><td>0.662 ± 0.010</td></tr><tr><td>Apertus</td><td>0.441</td><td>0.658 ± 0.007</td><td>0.678 ± 0.025</td><td>0.690 ± 0.015</td><td>0.639 ± 0.002</td><td>0.503 ± 0.014</td><td>0.678 ± 0.015</td><td>0.625 ± 0.016</td><td>0.707 ± 0.016</td></tr><tr><td>Avg.</td><td>0.490</td><td>0.622 ± 0.012</td><td>0.621 ± 0.029</td><td>0.632 ± 0.020</td><td>0.593 ± 0.015</td><td>0.506 ± 0.011</td><td>0.618 ± 0.022</td><td>0.583 ± 0.023</td><td>0.685 ± 0.013</td></tr><tr><td rowspan="3">SciQ [zh]</td><td>Aya</td><td>0.556</td><td>0.602 ± 0.035</td><td>0.592 ± 0.028</td><td>0.593 ± 0.031</td><td>0.553 ± 0.034</td><td>0.500 ± 0.007</td><td>0.568 ± 0.021</td><td>0.552 ± 0.028</td><td>0.665 ± 0.012</td></tr><tr><td>Apertus</td><td>0.458</td><td>0.678 ± 0.017</td><td>0.689 ± 0.012</td><td>0.695 ± 0.016</td><td>0.645 ± 0.011</td><td>0.503 ± 0.009</td><td>0.695 ± 0.018</td><td>0.613 ± 0.016</td><td>0.717 ± 0.012</td></tr><tr><td>Avg.</td><td>0.507</td><td>0.640 ± 0.026</td><td>0.640 ± 0.020</td><td>0.644 ± 0.023</td><td>0.599 ± 0.023</td><td>0.502 ± 0.008</td><td>0.632 ± 0.019</td><td>0.583 ± 0.022</td><td>0.691 ± 0.012</td></tr><tr><td rowspan="4">OKVQA [en]</td><td>Pixtral</td><td>0.630</td><td>0.731 ± 0.018</td><td>0.750 ± 0.013</td><td>0.739 ± 0.017</td><td>0.740 ± 0.017</td><td>0.728 ± 0.022</td><td>0.742 ± 0.017</td><td>0.743 ± 0.018</td><td>0.775 ± 0.008</td></tr><tr><td>LLaVa</td><td>0.643</td><td>0.740 ± 0.012</td><td>0.733 ± 0.007</td><td>0.755 ± 0.008</td><td>0.724 ± 0.013</td><td>0.738 ± 0.016</td><td>0.742 ± 0.011</td><td>0.755 ± 0.013</td><td>0.756 ± 0.022</td></tr><tr><td>Qwen</td><td>0.637</td><td>0.695 ± 0.012</td><td>0.711 ± 0.010</td><td>0.705 ± 0.012</td><td>0.693 ± 0.012</td><td>0.718 ± 0.010</td><td>0.711 ± 0.009</td><td>0.707 ± 0.016</td><td>0.725 ± 0.011</td></tr><tr><td>Avg.</td><td>0.637</td><td>0.722 ± 0.014</td><td>0.731 ± 0.010</td><td>0.733 ± 0.012</td><td>0.719 ± 0.014</td><td>0.728 ± 0.016</td><td>0.732 ± 0.012</td><td>0.735 ± 0.016</td><td>0.752 ± 0.014</td></tr></table>

## 7.2 Ablations

## 7.2.1 Effect of input augmentations

Non-Rephrased uses only the original input, and the probe set consists only of high-temperature samples from the original query.

Same-Input uses semantically equivalent rephrased inputs, but does not use the mix of rephrased and low-T sampled inputs. Instead, it evaluates each rephrase independently: for each rephrased input, we generate 10 high-temperature probes, compute the uncertainty score, and average the resulting AUROCs across rephrases.

Rephrased-only computes uncertainty using 10 rephrased inputs with high temperature responses obtained by sampling each of them once. The rephrase-only setting improves marginally, indicating that input rephrasing is useful, but it adds 2X rephrasing cost. Same ablation on DSE and SNNE are added in Appendix A.3.

SURE (Mixed-Rephrased) uses 5 equivalent input augmentations and 2 high-temperature responses from each to form 10 probe responses. Figure 4 shows the average AUROC across models, with full per-model results provided in Appendix A.3. Input augmentation significantly improves performance over the non-rephrased variant. Same-Input also improves over Non-Rephrased, showing that equivalent-input augmentation itself is beneficial. SURE (Mixed-Rephrased) gives the best average performance on both datasets, suggesting that mixing probe responses exposes the most informative set of semantic variations.

## 7.2.2 Dissecting cross-temperature agreement.

We next isolate the role of cross-temperature comparison by comparing SURE with sametemperature controls.

Low-T computes spectral energy using only lowtemperature samples.

High-T uses only high-temperature samples. For a matched setup, both controls use 10 samples.

As shown in Figure 4, Low-T performs poorly, indicating that low-temperature generations are often too stable to reveal uncertainty. High-T is stronger because it exposes semantic variability, but it still underperforms SURE. This supports the central idea of SURE: low-temperature samples provide stable anchors, while high-temperature probes test whether this stable belief persists under stochastic sampling.

![](images/128de467dcb3fd0d18aedc67324ee94546a682eb54c5d17c9c574e0eb3518b12.jpg)  
Figure 4: Ablations and direct controls for SURE. Average AUROC across the six text-QA LLMs on Trivia-QA and SVAMP for input-augmentation variants, sametemperature controls, and direct aggregation baselines. Equivalent-input augmentation and cross-temperature comparison majorly contribute to the final performance.

## 7.2.3 Temperature sensitivity

We analyze the sensitivity of SURE performance to the varying anchor temperature $( T _ { L } )$ and probe temperature $( T _ { H } )$ . With $T _ { H } = 1 . 0$ , performance remains stable across $( T _ { L } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \} )$ as shown in Figure 5, confirming that low-temperature responses provide stable semantic anchors. With $T _ { L } = 0 . 1$ , performance generally improves as $T _ { H }$ increases toward 0.9-1.0, but declines at higher temperatures. This reveals a trade-off that probes that are too stable fail to expose uncertainty, whereas excessively high temperatures produce noisy generations. We use $T _ { L } = 0 . 1$ and $T _ { H } = 1 . 0$ throughout without tuning specific to the dataset and model. Additional gridwise analysis is provided in Appendix A.5.

## 7.3 Strawman baselines

We compare SURE against direct aggregation controls computed on the same W. These include average similarity (AVGSIM), maximum anchor–probe similarity (MAXSIM), anchor–probe entropy, and average squared similarity (AVGSQSIM). Figure 4 compares SURE with AVGSIM and MAXSIM. Full results are shown in Appendix A.2. AVGSQSIM is numerically identical to BiG-SURE (Eqn. 2) and serves as a sanity check. This ablation study shows that the specific aggregation method used in SURE carries little value, and it is quite robust to aggregation methods like average similarity, maximum similarity and so on. The squared-similarity formulation remains useful for spectral analysis to inspect singular/spectral modes from a diagnostic perspective (Section 7.1). Anchor-probe entropy performs poorly as complete anchor-probe dissimilarity $( W _ { i j } = 0 )$ indicates high uncertainty, while anchor-probe entropy approaches zero in such cases.

![](images/af6e1b5527feafd27e2321c82c3e7e812b24bba6ca2cf35d552f615052ddcc20.jpg)

![](images/a6c303f9206cd474e1a21510f94e4b8abec02549b8abf286992fd61fc1e58f75.jpg)  
Figure 5: Temperature sensitivity of BiG-SURE. Top: Abstention AUROC when varying the anchor temperature $( T _ { L } )$ , with the probe temperature fixed at $( T _ { H } ~ = ~ 1 . 0 )$ . Performance remains stable across $( T _ { L } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \} )$ ). Bottom: Abstention AU-ROC when varying $( T _ { H } )$ , with $( T _ { L } = 0 . 1 )$ . Performance generally improves as $( T _ { H } )$ approaches 0.9-1.0, while excessively high temperatures degrade performance, revealing a trade-off.

## 7.4 Sample sensitivity analysis

We study the sensitivity of SURE to the number of anchors (M) and probes (N) in Figure 6. Across both datasets, the performance is more sensitive to the number of probe samples than to the number of anchor samples. Increasing N from 5 to 10 gives the largest improvement. Further increasing N yields only marginal gains, indicating saturation. In contrast, increasing M has little effect for fixed N, suggesting a small number of stable low-temperature anchors is sufficient.

![](images/a467547a23d76b1586270686bef678375f99c2e68b16aa2a5f52a721758055d0.jpg)  
Figure 6: Sample sensitivity analysis for SURE. Abstention AUROCs averaged over LLMs for different numbers of anchor samples (M) and probe samples (N) for Trivia-QA and SVAMP.

## 7.5 Human subjective evaluation

Multilingual correctness judge: For multilingual SciQ, we use Gemini-2.5-Flash to obtain correctness labels by comparing the question, reference answer, and greedy prediction. To assess the reliability of these labels, we conducted a human evaluation on the French, Japanese, and Chinese subsets of SciQ. For each language, we randomly sampled 20 responses generated by Aya, balanced between 10 labelled correct and 10 labelled incorrect by Gemini-2.5-Flash. 10 human annotators per language independently judged whether each prediction was correct with respect to the question and reference answer. We then obtain a humanconsensus label through majority voting. Gemini agreed with the human consensus on 90% of the French examples, 95% of the Japanese examples, and 90% of the Chinese examples, with Cohen’s κ of 0.80, 0.90, and 0.80, respectively, showing strong agreement in this limited sample study.

Evaluation of rephrasing quality SURE assumes that the LLM-generated rephrased questions preserve the meaning of the original input. To evaluate this, we conducted a human study in English and Japanese. For each language, we constructed a controlled set of 20 original-rephrase pairs. 10 human annotators per language independently rated the semantic equivalence of each pair on a fivepoint Likert scale. Ratings from 1 to 2 were treated as non-equivalent, while ratings from 3 to 5 were treated as equivalent. For every pair, we obtained a binary human-consensus label through majority voting. The consensus labels agreed with the equivalence labels for all 20 examples in both English and Japanese, yielding 100% agreement and Cohen’s $\kappa = 1 . 0 0$ within this limited subjective study.

## 7.6 Computational complexity

A practical consideration for black-box uncertainty estimation is the cost of uncertainty computation. For SURE, the bipartite similarity matrix $W ~ \in ~ \mathbb { R } ^ { M \times N }$ formation leads to complexity of $\mathcal { O } ( M N C _ { \mathrm { s e m } } )$ , where $C _ { \mathrm { s e m } }$ denotes a single semantic similarity evaluation cost. The subsequent computation of spectral energy is comparatively lightweight. Most existing methods construct a full pairwise similarity matrix over N sampled responses, leading to $\mathcal { O } ( N ^ { 2 } C _ { \mathrm { s e m } } )$ . KLE further incurs $\mathcal { O } ( N ^ { 3 } )$ cost. However, as mentioned by (Nikitin et al., 2024), the uncertainty computation time remains lightweight compared to the LLM inference cost. Hence, we also perform a matchedinference budget comparison to reassess the gains.

## 7.7 Matched inference-budget comparison

From the LLM inference cost perspective, SURE uses $M + N = 3 + 1 0$ responses per input, whereas most prior sampling-based baselines use N = 10 responses. To ensure that the gains are not simply due to more LLM calls, we re-evaluate the prior works with equalized budget of N = 13 for text-QA tasks. As shown in Appendix A.4, SURE outperforms the baselines under this setting as well.

## 8 Conclusion

We introduced BiG-SURE, a black-box uncertainty estimation framework based on cross-temperature bipartite semantic agreement. BiG-SURE uses low-temperature responses as stable anchors and high-temperature responses as probes, constructs an anchor–probe semantic similarity graph, and defines uncertainty as the complement of its normalized squared spectral energy. Empirically, SURE achieves strong abstention AUROC across textonly, multilingual, and multimodal QA tasks, improving over prior black-box uncertainty estimators in most evaluated settings. Our ablations further show that the cross-temperature comparison and the equivalent-input augmentation majorly contribute to the final performance. These results suggest that SURE is a simple, interpretable, and practical method for black-box uncertainty estimation, making BiG-SURE a promising candidate for reliability-aware deployment of LLMs and VLMs in safety-critical applications.

## 9 Limitations

While BIG-SURE shows strong empirical performance across text-only, multilingual, and multimodal QA settings, it has the following limitations.

Dependence on the semantic similarity backend. SURE relies on an entailment-based semantic similarity function to populate the anchor–probe bipartite graph. If this scorer fails to reliably capture semantic equivalence, the resulting similarity matrix may not accurately reflect the model’s uncertainty. This dependence is especially important in multilingual and multimodal settings, where semantic matching across languages or visually grounded responses can itself be challenging.

Consistency does not guarantee correctness. SURE can fundamentally underperform if a model mispredicts and produces the same incorrect answer semantically consistently during sampling. In this case, even a perfect semantic-similarity backend causes a consistency-based estimator to be overconfident. This failure mode can be distinguished from errors caused by the NLI backend. Further analysis and examples are discussed in Appendix A.10.

Visual grounding failure. For visual QA, the current semantic-similarity backend operates only on the generated textual responses and does not examine the image. Consequently, SURE measures textual generation consistency without considering image-text grounding. It may, therefore, assign high confidence when a VLM consistently produces the same visually overlooked answer. Extending the agreement function to incorporate nontext modalities is an important future work. Further analysis with examples are in Appendix A.10.

Limited subjective assessment. BiG-SURE depends on the quality of meaning-preserving paraphrases, while Gemini-2.5-Flash is used to assign correctness labels for multilingual QA. Our human evaluations provide encouraging evidence for both components, but their limited scale does not constitute comprehensive validation across languages, domains, or question types. These results should therefore be interpreted as sanity checks rather than definitive validation of either the rephrasing model or the correctness judge. A systematic human evaluation of the NLI-based semantic-similarity backend remains for future work.

Evaluation scope. Our experiments focus on question-answering style benchmarks, including factual QA, math word problems, multilingual science QA, and visual QA. Although these tasks cover multiple languages, modalities, and model families, they do not fully represent broader openended generation settings such as long-form generation, summarization, dialogue, planning, or code synthesis. Extending cross-temperature bipartite agreement to these broader tasks remains an important direction for future work.

## Acknowledgments

We thank Google Research for its support through the Gemma Academic Award, which substantially helped meet the computational requirements of this work. We also gratefully acknowledge Google’s support for our conference travel, provided through the Kotak IISc AI–ML Centre.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, Albert Q. Jiang, Kartik Khandelwal, Timothée Lacroix, Guillaume Lample, Diego Las Casas, Thibaut Lavril, and 23 others. 2024. Pixtral 12b. Preprint, arXiv:2410.07073.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In Proceedings ofthe IEEE international conference on computer vision, pages 2425–2433.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Suhana Bedi, Yutong Liu, Lucy Orr-Ewing, Dev Dash, Sanmi Koyejo, Alison Callahan, Jason A Fries,

Michael Wornow, Akshay Swaminathan, Lisa Soleymani Lehmann, and 1 others. 2025. Testing and evaluation of health care applications of large language models: a systematic review. Jama, 333(4):319–328.

Debarpan Bhattacharya, Apoorva Kulkarni, and Sriram Ganapathy. 2025a. Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning. In Interspeech 2025, pages 2068–2072.

Debarpan Bhattacharya, Apoorva Kulkarni, and Sriram Ganapathy. 2025b. Festa: Functionally equivalent sampling for trust assessment of multimodal llms. In EMNLP (Findings), pages 12277–12295.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

John Dang, Shivalika Singh, Daniel D’souza, Arash Ahmadian, Alejandro Salamanca, Madeline Smith, Aidan Peppin, Sungjin Hong, Manoj Govindassamy, Terrence Zhao, Sandra Kublik, Meor Amer, Viraat Aryabumi, Jon Ander Campos, Yi-Chern Tan, Tom Kocmi, Florian Strub, Nathan Grinsztajn, Yannis Flet-Berliac, and 26 others. 2024. Aya expanse: Combining research breakthroughs for a new multilingual frontier. Preprint, arXiv:2412.04261.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630.

Marina Fomicheva, Shuo Sun, Lisa Yankovskaya, Frédéric Blain, Francisco Guzmán, Mark Fishel, Nikolaos Aletras, Vishrav Chaudhary, and Lucia Specia. 2020. Unsupervised quality estimation for neural machine translation. Transactions ofthe Association for Computational Linguistics, 8:539–555.

Xiang Gao, Jiaxin Zhang, Lalla Mouatadid, and Kamalika Das. 2024. SPUQ: Perturbation-based uncertainty quantification for large language models. In Proceedings ofthe 18th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2336–2346, St. Julian’s, Malta. Association for Computational Linguistics.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. DeBERTa: Decoding-enhanced BERT with Disentangled Attention. In International Conference on Learning Representations.

Alejandro Hernández-Cano, Alexander Hägele, Allen Hao Huang, Angelika Romanou, Antoni-Joan Solergibert, Barna Pasztor, Bettina Messmer, Dhia Garbaya, Eduard Frank Durech, Ido<sup>ˇ</sup> Hakimi, Juan García Giraldo, Mete Ismayilzada, Negar Foroutan, Skander Moalla, Tiancheng Chen, Vinko Sabolcec, Yixuan Xu, Michaelˇ Aerni, Badr AlKhamissi, and 82 others. 2025. Apertus: Democratizing Open and Compliant LLMs for Global Language Environments. https://arxiv.org/abs/2509.14233.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, and 1 others. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations.

Lucie Kunitomo-Jacquin, Edison Marrese-Taylor, Ken Fukuda, and Masahiro Hamasaki. 2026. Evidential semantic entropy for llm uncertainty quantification. In Proceedings ofthe 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7107– 7122.

Gregory Kang Ruey Lau, Hieu Dao, and Bryan Kian Hsiang Low. 2025. Uncertainty quantification for MLLMs. In ICLR Workshop: Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2024. Generating with confidence: Uncertainty quantification for black-box large language models. Transactions on Machine Learning Research.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023. Improved baselines with visual instruction tuning. Preprint, arXiv:2310.03744.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Conference on Computer Vision and Pattern Recognition (CVPR).

Dang Nguyen, Ali Payani, and Baharan Mirzasoleiman. 2025. Beyond semantic entropy: Boosting llm uncertainty quantification with pairwise semantic similarity. arXiv preprint arXiv:2506.00245.

Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka Marttinen. 2024. Kernel language entropy: Finegrained uncertainty quantification for llms from semantic similarities. Advances in Neural Information Processing Systems, 37:8901–8929.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2080–2094, Online. Association for Computational Linguistics.

Edward Phillips, Sean Wu, Fredrik K. Gustafsson, Boyan Gao, and David A. Clifton. 2026. Semantic self-distillation for language model uncertainty. In Proceedings ofthe 42nd Conference on Uncertainty in Artificial Intelligence, volume 337 of Proceedings of Machine Learning Research, pages 5427–5447. PMLR.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Shezheng Song, Xiaopeng Li, Shasha Li, Shan Zhao, Jie Yu, Jun Ma, Xiaoguang Mao, Weimin Zhang, and Meng Wang. 2025. How to bridge the gap between modalities: Survey on multimodal large language model. IEEE Transactions on Knowledge and Data Engineering, 37(9):5311–5329.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth ’ee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Mingyang Wang, Heike Adel, Lukas Lange, Yihong Liu, Ercong Nie, Jannik Strötgen, and Hinrich Schütze. 2025. Lost in multilinguality: Dissecting crosslingual factual inconsistency in transformer language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5075–5094.

Boyang Xue, Hongru Wang, Rui Wang, Sheng Wang, Zezhong Wang, Yiming Du, Bin Liang, Wenxuan Zhang, and Kam-Fai Wong. 2025. MlingConf: A comprehensive study of multilingual confidence estimation on large language models. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 2535–2556, Vienna, Austria. Association for Computational Linguistics.

Ruiyang Zhang, Hu Zhang, Hao Fei, and Zhedong Zheng. 2025. Uncertainty-o: One model-agnostic framework for unveiling uncertainty in large multimodal models. arXiv preprint arXiv:2506.07575.

## A Appendix

A.1 Theoretical properties of SURE as an uncertainty estimator

1. Semantic identity: If all high temperature sampled prediction sequences are semantically identical and aligned with low-temperature samples, then W converges to $\mathbb { 1 } _ { M \times N }$ and $\mathcal { U } _ { \sf n o r m } ( W ; \theta ) = 0 .$

Intuition. When all high-temperature generations are semantically identical, they all lie in the same semantic cluster as the low-temperature anchors. Consequently, every anchor–probe pair exhibits maximal entailment, so $W _ { i j } \approx 1$ for all $i , j$ , and W approaches the matrix of all ones 1 $M \times N$ . The spectral energy can be expressed as square of the Frobenius norm $\| W \| _ { F }$ (proof in Appendix $\mathbf { A . } 1 )$ With W being approximately equal to $\mathbb { 1 } _ { M \times N }$ , the normalized spectral energy approaches a value of 1 and the uncertainty, defined as its complement, approaches a value of 0. In other words, perfect semantic consensus across temperatures leaves no room for ambiguity. The detailed proof is given in Proposition A.2.

2. Semantic disjointness: If all high temperature responses are mutually semantically disjoint, and mis-aligned with low-temperature samples, then W converges to ${ \mathbf { 0 } } _ { M \times N }$ and $\mathcal { U } _ { \sf n o r m } ( W ; \theta ) = 1$

Intuition. If the high-temperature samples are mutually semantically disjoint (i.e., they do not align with the low-temperature meaning and also share no common semantic cluster), then each probe $h _ { j }$ exhibits negligible semantic entailment with every anchor $a _ { i }$ . The similarity matrix satisfies $W _ { i j }$ ≈ 0 for all $( i , j )$ , and W approaches the all-zero matrix ${ \bf 0 } _ { M \times N }$ . In this case, the spectral energy $\| W \| _ { F }$ goes to 0. As a result, $\mathcal { U } _ { \sf n o r m } ( W ; \theta ) = 1$ . Intuitively, complete semantic mismatch across temperatures corresponds to maximal ambiguity which results in maximal value of SURE metric. A formal proof is shown in Proposition A.3.

3. Monotonicity: For fixed (M, N), if W<sup>′</sup> corresponds to an input $x ^ { \prime }$ where all elements of $W ^ { \prime } \preceq W , t h e n , \mathcal { U } _ { \mathsf { n o r m } } ( W ^ { \prime } ; \theta ) \geq \mathcal { U } _ { \mathsf { n o r m } } ( W ; \theta ) .$

Intuition. Holding the number of low and high temperature responses fixed, an increased semantic diversity in the high-temperature set typically reduces the cross-temperature agreement: each high temperature response $h _ { j }$ has reduced alignment with low temperature response $a _ { i } ,$ , and the pairwise similarities shrink. The condition $W ^ { \prime } \preceq W$ elementwise formalizes this weakening. Since the (normalized) spectral energy is proportional to the Frobenius norm, decreasing the entries decreases $\| W \| _ { F }$ yielding $\mathcal { U } _ { \sf n o r m } ( W ^ { \prime } ; \theta ) \geq \mathcal { U } _ { \sf n o r m } ( W ; \theta )$ . Intuitively, when predictions spread across more semantic modes, cross-temperature consistency breaks down and the uncertainty increases. A formal proof is given in Proposition A.6.

Proposition A.1. Let $W \in \mathbb { R } ^ { M \times N }$ with entries satisfying $0 \leq W _ { i j } \leq 1$ for all $i , j .$ . Let

$$
E _ { S U R E } ( W ; \theta ) = \sum _ { i = 1 } ^ { r } \sigma _ { i } ^ { 2 }
$$

$$
r = \operatorname { r a n k } ( W ) \leq \operatorname* { m i n } ( M , N ) ,
$$

where $\sigma _ { i }$ are the non-zero singular values of W. Define

$$
E _ { n o r m } ( W ; \theta ) = \frac { E _ { S U R E } ( W ; \theta ) } { M N } .
$$

Then

$$
0 \leq E _ { n o r m } ( W ; \theta ) \leq 1 .
$$

Consequently, if

$$
U _ { n o r m } ( W ; \theta ) = 1 - E _ { n o r m } ( W ; \theta ) ,
$$

then

$$
0 \leq U _ { n o r m } ( W ; \theta ) \leq 1 .
$$

Proof. Let the singular value decomposition of W be

$$
\boldsymbol { W } = \boldsymbol { U \Sigma V } ^ { \top } ,
$$

where $U \in \mathbb { R } ^ { M \times M }$ and $V \in \mathbb { R } ^ { N \times N }$ are orthogonal matrices, and

$$
\Sigma = \operatorname { d i a g } ( \sigma _ { 1 } , \sigma _ { 2 } , \ldots , \sigma _ { r } , 0 , \ldots , 0 ) .
$$

Using the unitary invariance of the Frobenius norm,

$$
\| W \| _ { F } ^ { 2 } = \| U \Sigma V ^ { \top } \| _ { F } ^ { 2 } = \| \Sigma \| _ { F } ^ { 2 } .
$$

Since the Frobenius norm of a diagonal matrix is the sum of squares of its diagonal entries, we obtain

$$
\| \boldsymbol { W } \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { r } \sigma _ { i } ^ { 2 } .
$$

Therefore,

$$
E _ { S U R E } ( W ; \theta ) = \| W \| _ { F } ^ { 2 } .
$$

Now, by definition of the Frobenius norm,

$$
E _ { S U R E } ( W ; \theta ) = \| W \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { N } W _ { i j } ^ { 2 } .\tag{4}
$$

Since $0 \leq W _ { i j } \leq 1$ , it follows that $0 \leq W _ { i j } ^ { 2 } \leq 1$ for all $i , j$ . Hence,

$$
0 \leq \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { N } W _ { i j } ^ { 2 } \leq M N .
$$

Thus,

$$
0 \leq E _ { S U R E } ( W ; \theta ) \leq M N .
$$

Dividing throughout by MN yields

$$
0 \leq E _ { n o r m } ( W ; \theta ) = \frac { E _ { S U R E } ( W ; \theta ) } { M N } \leq 1 .
$$

Finally, since

$$
U _ { n o r m } ( W ; \theta ) = 1 - E _ { n o r m } ( W ; \theta ) ,
$$

we immediately obtain

$$
0 \leq U _ { n o r m } ( W ; \theta ) \leq 1 .
$$

Proposition A.2. Given M low-temperature and N high-temperature responses, $U _ { n o r m } \to 0$ iff all sampling responses are semantically similar.

Proof. To prove this, we show the following:

$U _ { n o r m } \to 0$ if all sampling responses are semantically similar.

• All sampling responses are semantically similar if $U _ { n o r m }  0$

Case I: As all sampling responses are semantically similar, let $w _ { i j } = s , \forall i , j$ . Then the bi-adjacency matrix is constant:

$$
W = s { \bf 1 } _ { M \times N } , \qquad w _ { i j } = s , \forall i , j ,
$$

where ${ \bf 1 } _ { M \times N }$ is the all-ones matrix. The Frobenius norm satisfies

$$
\| W \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { N } w _ { i j } ^ { 2 } = \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { N } s ^ { 2 } = M N s ^ { 2 } .
$$

Since $\begin{array} { r } { E _ { S U R E } ( W ; \theta ) = \sum _ { i = 1 } ^ { \operatorname* { m i n } \{ M , N \} } \sigma _ { i } ^ { 2 } = \| W \| _ { F } ^ { 2 } } \end{array}$ (Eqn. 4), we obtain

$$
E _ { S U R E } ( W ; \theta ) = M N s ^ { 2 } .
$$

Therefore the normalized confidence is

$$
E _ { \mathrm { n o r m } } ( W ; \theta ) = \frac { E _ { S U R E } ( W ; \theta ) } { M N } = \frac { M N s ^ { 2 } } { M N } = s ^ { 2 } ,
$$

and the normalized uncertainty is

$$
U _ { \mathrm { n o r m } } ( W ; \theta ) = 1 - E _ { \mathrm { n o r m } } ( W ; \theta ) = 1 - s ^ { 2 } .
$$

Hence, as $s \to 1$ (perfect semantic similarity), we have $U _ { \mathrm { n o r m } } ( W ; \theta ) \to 0$

Case II: Suppose $U _ { \mathrm { n o r m } } ( W ; \theta ) = 0$ . Then $E _ { \mathrm { n o r m } } ( W ; \theta ) = 1$ , hence

$$
E _ { S U R E } ( W ; \theta ) = M N .
$$

Therefore

$$
\sum _ { i , j } W _ { i j } ^ { 2 } = M N .
$$

Since $0 \leq W _ { i j } \leq 1$ , we have $W _ { i j } ^ { 2 } \leq 1$ for each entry, and thus

$$
\sum _ { i , j } W _ { i j } ^ { 2 } \leq \sum _ { i , j } 1 = M N ,
$$

with equality if and only if $W _ { i j } ^ { 2 } = 1$ for every $( i , j )$ i.e., $W _ { i j } = 1$ for all $( i , j )$ . Hence $W = \mathbf { 1 } _ { M \times N }$

Finally, if $W _ { i j } = s$ for all $i , j$ , then $W = \mathbf { 1 } _ { M \times N }$ implies $s = 1$ , and conversely $s = 1$ implies $W =$ ${ \bf 1 } _ { M \times N }$ , completing the iff statement. □

Proposition A.3. Given M low-temperature and N high-temperature responses, $U _ { \mathrm { n o r m } } ( W ; \theta ) = 1$ iff all cross-temperature response pairs are semantically disjoint, i.e., sim $( a _ { i } , h _ { j } ) = 0 f o r a l l i , j .$

Proof. We show both directions of implications to prove the iff condition.

Case I: If $U _ { \mathrm { n o r m } } ( W ; \theta ) = 1$ , then $E _ { \mathrm { n o r m } } ( W ; \theta ) =$ 0, hence $E _ { S U R E } ( W ; \theta ) = 0$ and therefore

$$
\sum _ { i , j } W _ { i j } ^ { 2 } = 0 .
$$

Since each term $W _ { i j } ^ { 2 } \geq 0$ , the only way their sum is zero is if $W _ { i j } ^ { 2 } = \stackrel { \cdot } { 0 }$ for every $( i , j )$ , i.e., $W _ { i j } = 0$ for all $( i , j )$ . Thus $W = { \bf 0 } _ { M \times N }$ Case II: Conversely, if $\begin{array} { r c l } { W } & { = } & { \mathbf { 0 } _ { M \times N } . } \end{array}$ , then $\| W \| _ { F } ^ { 2 } = 0 .$ , hence $E _ { \mathrm { n o r m } } ( W ; \theta ) = 0$ and $U _ { \mathrm { n o r m } } ( W ; \theta ) = 1$ □

Theorem A.4. For M low and N high-temperature responses, we show that,

1. The number of feasible clusters $K ( W )$ increases with decreasing $W _ { i j } ,$ , and vice-versa.

2. P $\begin{array} { r } { \left( C _ { N } ( W ) = N \right) = \frac { ( K ( W ) ) _ { N } } { ( K ( W ) ) ^ { N } } } \end{array}$

3. P $\begin{array} { r } { \mathrm { ( } C _ { N } ( W ) < N \mathrm { ) } \le \binom { N } { 2 } \frac { 1 } { K ( W ) } } \end{array}$

$$
\begin{array} { r } { 4 . \ \mathbb { E } [ C _ { N } ( W ) ] = \sum _ { i = 1 } ^ { K ( W ) } 1 - ( 1 - p _ { i } ) ^ { N } \le N } \end{array}
$$

5. Under uniform cluster probability distribution and $0 \leq W _ { i j } < 1 , \mathbb { E } [ C _ { N } ( W ) ]  N .$

6. Also, $\mathbb { E } [ C _ { N } ( W ) ]$ is a monotonic function of K(W).

Proof. For simplicity, we assume that the lowtemperature responses are semantically similar, and the high-temperature responses show semantic variability.

Let N be the number of high-temperature samples. Also, given the similarity matrix W, let the number of possible semantic clusters is $K ( W )$ (for example, for $W = \mathbf { 1 } _ { M \times N } , K ( W ) = 1 )$ . But $K ( W )$ clusters are observed only if an infinite number of samples are drawn. Because of the finite sampling of N, we do observe a subset of them.

Assume the M low-temperature responses $\{ q _ { i } \} _ { i = 1 } ^ { M }$ be semantically consistent and belong to a single semantic cluster, denoted by index ℓ. Let $\{ 1 , \ldots , \bar { K } \}$ be a universal set of semantic clusters for a fixed input. Associate each cluster $j$ with a prototype similarity to the cluster ℓ,

$$
\mu _ { j } \triangleq \operatorname { s i m } ( j , \ell ) \in [ 0 , 1 ] , \qquad \mu _ { \ell } = 1 ,
$$

where larger $\mu _ { j }$ means cluster $j$ is semantically closer to the cluster ℓ.

We assume that, conditional on a high-temperature response belonging to cluster $j ,$ the observed crosstemperature similarity score $s \in [ 0 , 1 ]$ follows a $\beta \mathrm { . }$ -distribution and has a likelihood of

$$
L _ { j } ( s ) \propto s ^ { \nu \mu _ { j } - 1 } ( 1 - s ) ^ { \nu ( 1 - \mu _ { j } ) - 1 } , \qquad \nu > 0 ,
$$

i.e., $s \mid ( Z = j ) \sim \operatorname { B e t a } ( \nu \mu _ { j } , \nu ( 1 - \mu _ { j } ) )$ . Based on a threshold $\alpha ~ \in ~ ( 0 , 1 ]$ and define the set of feasible clusters at similarity level s by a likelihoodratio rule:

$$
{ \mathcal { F } } _ { \alpha } ( s ) \ \triangleq \ \left\{ j \in \{ 1 , \dots , \bar { K } \} : \ \frac { L _ { j } ( s ) } { L _ { \ell } ( s ) } \ \geq \ \alpha \right\} .
$$

Define the number of feasible clusters as $K ( W ) =$ $K ( s ) \triangleq | \mathcal { F } _ { \alpha } ( s ) |$ . Using the Beta likelihood above and $\mu _ { \ell } = 1$ , the log-likelihood ratio simplifies (up to a constant independent of $j )$ to

$$
\begin{array} { c } { { \displaystyle \log \displaystyle \frac { L _ { j } ( s ) } { L _ { \ell } ( s ) } = \nu ( \mu _ { j } - 1 ) \log s } } \\ { { { } } } \\ { { { } + \nu ( 1 - \mu _ { j } ) \log ( 1 - s ) + \mathrm { c o n s t } ( j ) } } \end{array}
$$

For s bounded away from 1 and in the lowsimilarity regime (small s), the dominant term is

$$
\nu ( \mu _ { j } - 1 ) \log s ,
$$

since log $s \in  { \mathcal { S } } _ { } \varepsilon ^ { } + \infty \to  { \mathbb { Z } } _ { } { } \varepsilon ^ { } \to \to \infty $ while log $1 - s ) = { \cal { O } } ( 1 )$ Because $\mu _ { j } ~ < ~ 1$ for $j \neq \ell ,$ we have $( \mu _ { j } - 1 ) < 0$ , hence $( \mu _ { j } - 1 )$ log s increases as s decreases (it becomes large positive). Therefore, for each fixed $j \neq \ell ,$ , the likelihood ratio $L _ { j } ( s ) / L _ { \ell } ( s )$ is nondecreasing as s decreases, implying that once a cluster satisfies the feasibility condition at some $s _ { 2 } .$ , it will also satisfy it for any smaller $s _ { 1 } ~ < ~ s _ { 2 }$ . Thus ${ \mathcal { F } } _ { \alpha } ( s _ { 1 } ) \ \supseteq \ { \mathcal { F } } _ { \alpha } ( s _ { 2 } )$ and $K ( s _ { 1 } ) \geq K ( s _ { 2 } ) \left( K ( W _ { 1 } ) \geq K ( W _ { 2 } ) \right)$

Let, $C _ { N } ( W )$ be the observed number of semantic clusters among $K ( W )$ because we use finite N samples instead of infinite, and $C _ { N } ( W ) \leq K ( W )$

Assuming N is not too large and $K ( W ) \geq N$

$$
\begin{array} { l } { \displaystyle \mathbb { P } \big ( C _ { N } ( W ) = N \big ) = \displaystyle \frac { K ( W ) } { K ( W ) } \cdot \displaystyle \frac { K ( W ) - 1 } { K ( W ) } \cdot \cdot \cdot } \\ { \displaystyle \qquad \cdot \cdot \cdot \frac { K ( W ) - N + 1 } { K ( W ) } } \\ { \displaystyle \qquad = \frac { ( K ( W ) ) _ { N } } { ( K ( W ) ) ^ { N } } } \end{array}\tag{5}
$$

Also, let the event of responses i and $j$ being assigned to the same cluster be $E _ { i j }$ . If $Z _ { i }$ is the semantic cluster id assigned to response i, then

$$
E _ { i j } = \{ Z _ { i } = Z _ { j } \} , \mathbb { P } ( E _ { i j } ) = \frac { 1 } { K ( W ) }\tag{6}
$$

Now,

$$
\begin{array} { r l r } {  { \mathbb { P } ( \exists i < j : Z _ { i } = Z _ { j } ) = \mathbb { P } ( \cup _ { i < j } E _ { i j } ) } } \\ & { } & { \quad \le \displaystyle \sum _ { i < j } \frac { 1 } { K ( W ) } } \\ & { } & { \quad \implies \mathbb { P } \big ( C _ { N } ( W ) < N \big ) \le \binom { N } { 2 } \frac { 1 } { K ( W ) } } \end{array}\tag{7}
$$

Let the cluster probabilities among the $K ( W )$ feasible set of clusters given W are $\{ p _ { i } \} _ { i = 1 } ^ { K ( W ) }$ . We calculate the expected number of observed clusters among $N$ samples.

$$
\begin{array} { c } { { C _ { N } ( W ) = \displaystyle \sum _ { i = 1 } ^ { K ( W ) } I _ { i } } } \\ { { I _ { i } = \{ \mathrm { c l u s t e r } i \mathrm { a p p e a r s \ a t \ : l e a s t \ : o n c e \ : a m o n g \ : } N \} } } \\ { { ( 8 \ell ) } } \end{array}
$$

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ C _ { N } ( W ) ] = \mathbb { E } \bigg [ \sum _ { i = 1 } ^ { K ( W ) } I _ { i } \bigg ] } \\ { = \sum _ { i = 1 } ^ { K ( W ) } \mathbb { E } [ I _ { i } ] } \\ { = \sum _ { i = 1 } ^ { K ( W ) } 1 - ( 1 - p _ { i } ) ^ { N } } \end{array}\tag{9}
$$

Case I: For $W _ { i j } = 1$ , implies $K ( W ) = 1$ and hence, $p _ { i } = 1 . \mathrm { { S o } } .$

$$
\mathbb { E } [ C _ { N } ( W ) ] = 1 - ( 1 - 1 ) ^ { N } = 1\tag{10}
$$

Case II: For $W _ { i j } = 0$ , implies K(W) is large. So, Notice that,

$$
\begin{array} { r l } { \mathbb { E } \langle C _ { N } ( W ) \rangle = } & { \overset { K ( W ) } { \underset { i = 1 } { \overset { N ( W ) } { \sum } } } 1 - ( 1 - p _ { i } ) ^ { N } } \\ & { = \displaystyle \sum _ { k = 1 } ^ { K ( W ) } \left[ 1 - ( 1 - N p _ { i } + \binom { N } { 2 } p _ { i } ^ { 2 } - \dotsc ) \right] } \\ & { \overset { K ( W ) } { = } \displaystyle \sum _ { i = 1 } ^ { K ( W ) } \left[ N p _ { i } - \binom { N } { 2 } p _ { i } ^ { 2 } + \dotsc \right] } \\ & { = \displaystyle \sum _ { i = 1 } ^ { K ( W ) } \left[ N p _ { i } - \binom { N } { 2 } p _ { i } ^ { 2 } + \dotsc \right] } \\ & { \overset { K ( W ) } { = } \displaystyle \sum _ { i = 1 } ^ { K ( W ) } p _ { i } - \binom { N } { 2 } \underset { i = 1 } { \overset { N ( W ) } { \sum } } p _ { i } ^ { 2 } + \dotsc } \\ & { = \displaystyle - \sum _ { i = 1 } ^ { K } \left( \sum _ { j } ^ { N } \sum _ { i = 1 } ^ { K ( W ) } p _ { i } ^ { 2 } + \dotsc \right. } \\ & { \left. \underset { i \leq N } { \overset { N ( W ) } { \sum } } p _ { i } ^ { 2 } + \dotsc \right. } \end{array}\tag{11}
$$

For uniform cluster distribution, $\begin{array} { r } { p _ { i } = \frac { 1 } { K ( W ) } } \end{array}$ , and $\mathbb { E } [ C _ { N } ( W ) ] \to N$ as $K ( W )$ is large and hence $\textstyle { \frac { N ^ { 2 } } { K ( W ) } } < < 1$

For uniform distribution with $K ( W ) = k$ clusters,

$$
p _ { i } = \frac { 1 } { K ( W ) } = \frac { 1 } { k } , k > 0\tag{12}
$$

Now, replacing $p _ { i }$ in the cluster equation, we get,

$$
\mathbb { E } [ C _ { N } ] = k \bigg ( 1 - \bigg ( 1 - \frac { 1 } { k } \bigg ) ^ { N } \bigg )\tag{13}
$$

$C _ { N }$ is shorthand notation of $C _ { N } ( W )$ . Now, we need to show, $\frac { \partial \mathbb { E } [ C _ { N } ] } { \partial k } \geq 0$

Denote,

$$
a ( k ) = 1 - { \frac { 1 } { k } } , a ^ { \prime } ( k ) = { \frac { 1 } { k ^ { 2 } } }\tag{14}
$$

$$
\begin{array} { c } { { f ( k ) = k - k a ( k ) ^ { N } } } \\ { { \Longrightarrow f ^ { \prime } ( k ) = 1 - a ( k ) ^ { N } - k N a ( k ) ^ { N - 1 } a ^ { \prime } ( k ) } } \\ { { = 1 - a ( k ) ^ { N } - \displaystyle \frac { k N } { k ^ { 2 } } a ( k ) ^ { N - 1 } } } \end{array}\tag{15}
$$

Now, let, $\begin{array} { r } { x = \frac { 1 } { k } \in ( 0 , 1 ] } \end{array}$ . So, a = 1 − x. Now,

$$
\begin{array} { c } { { f ^ { \prime } ( k ) = 1 - ( 1 - x ) ^ { N - 1 } ( 1 - x + N x ) } } \\ { { = 1 - ( 1 - x ) ^ { N - 1 } ( 1 + ( N - 1 ) x ) } } \end{array}\tag{16}
$$

Now, we have to show, $( 1 - x ) ^ { N - 1 } ( 1 + ( N - 1 ) x ) \leq$ 1 for $x \in [ 0 , 1 ]$ . Let $m = N - 1 \geq 0 .$ . Define,

$$
h ( x ) : = \ln ( 1 + m x ) + m \ln ( 1 - x )\tag{17}
$$

$$
\begin{array} { r } { h ( x ) \leq 0 } \\ { \implies \exp ( h ( x ) ) \leq 1 } \\ { \implies ( 1 + m x ) ( 1 - x ) ^ { m } \leq 1 } \end{array}\tag{18}
$$

Now,

$$
\begin{array} { l } { \displaystyle h ^ { \prime } ( x ) = \frac { m } { 1 + m x } + m \frac { - 1 } { 1 - x } } \\ { \displaystyle \quad = m \bigg ( \frac { 1 } { 1 + m x } - \frac { 1 } { 1 - x } \bigg ) } \\ { \displaystyle \quad = - \frac { m ( m + 1 ) x } { ( 1 + m x ) ( 1 - x ) } \le 0 } \end{array}\tag{19}
$$

Also, we know $h ( 0 ) = 0$ . Hence, $h ( x ) \leq 0$ , and hence proved. □

Proposition A.5. For the normalized uncertainty,

$$
U _ { n o r m } ( W ; \theta ) = 1 - \frac { E _ { S U R E } ( W ; \theta ) } { M N } .
$$

Then $U _ { n o r m } ( W ; \theta )$ is monotone non-increasing in each entry of W. In particular, iffor some fixed $( p , q )$

$$
\widetilde { W } = W + \delta e _ { p } e _ { q } ^ { \top } , \qquad \delta \geq 0 ,
$$

then

$$
U _ { n o r m } ( \widetilde { W } ; \theta ) \le U _ { n o r m } ( W ; \theta ) .
$$

Proof. We have

$$
U _ { n o r m } ( W ; \theta ) = 1 - \frac { 1 } { M N } \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { N } W _ { i j } ^ { 2 } .
$$

Then

$$
\begin{array} { c } { { U _ { n o r m } ( \widetilde { W } ; \theta ) = 1 - \displaystyle \frac { 1 } { M N } . } } \\ { { \displaystyle \left( \sum _ { ( i , j ) \neq ( p , q ) } W _ { i j } ^ { 2 } + ( W _ { p q } + \delta ) ^ { 2 } \right) } } \end{array}
$$

Therefore,

$$
\begin{array} { c c } { { U _ { n o r m } ( \widetilde { W } ; \theta ) - U _ { n o r m } ( W ; \theta ) } } & { { ( 2 0 ) } } \\ { { \displaystyle = - \frac { 1 } { M N } \Big ( ( W _ { p q } + \delta ) ^ { 2 } - W _ { p q } ^ { 2 } \Big ) } } & { { } } \\ { { \displaystyle ( 2 1 ) } } & { { } } \\ { { \displaystyle = - \frac { 1 } { M N } \Big ( 2 \delta W _ { p q } + \delta ^ { 2 } \Big ) . } } & { { ( 2 2 ) } } \end{array}
$$

Since $W _ { p q } \geq 0$ and $\delta \geq 0$ , it follows that

$$
2 \delta W _ { p q } + \delta ^ { 2 } \geq 0 .
$$

Hence,

$$
U _ { n o r m } ( { \widetilde { W } } ; \theta ) \leq U _ { n o r m } ( W ; \theta ) .
$$

Thus, $U _ { n o r m } ( W ; \theta )$ is monotone non-increasing ineach entry of W. □

Proposition A.6. Assume that the low-temperature responses are semantically similar. Then increasing the number ofsemantically unique hightemperature answers weakens cross-temperature agreement in the sense that for two prediction sets V and V<sup>′</sup> (with the same $M , N )$ , the corresponding similarity matrices satisfy

$$
W ^ { \prime } \preceq W \quad e l e m e n t w i s e , w i t h W ^ { \prime } \neq W .\tag{A1}
$$

Consequently, increasing semantic uniqueness of high-temperature answers monotonically increases $U _ { \mathrm { n o r m } } ( W ; \theta )$ and simultaneously increases the expected number ofobserved semantic clusters.

Proof. We prove the two monotone conclusions. (i) $U _ { \mathrm { n o r m } }$ increases under weakened crosstemperature agreement. By Proposition A.5, the map $W \mapsto U _ { \mathrm { n o r m } } ( W ; \theta )$ is monotone nonincreasing in each entry of W. Hence, if $W ^ { \prime } \preceq W$ elementwise, then

$$
U _ { \mathrm { n o r m } } ( W ^ { \prime } ; \theta ) \ge U _ { \mathrm { n o r m } } ( W ; \theta ) .
$$

Moreover, since $W ^ { \prime } \neq W$ , at least one entry decreases strictly, implying the inequality is strict:

$$
U _ { \mathrm { n o r m } } ( W ^ { \prime } ; \theta ) > U _ { \mathrm { n o r m } } ( W ; \theta ) .
$$

(ii) The expected number of observed semantic clusters increases. Since the low-temperature responses are assumed semantically similar, they define a single semantic cluster. By Theorem $\mathsf { A } . 4 ( 1 )$ , decreasing cross-temperature similarities $( \mathrm { i . e . , } W ^ { \prime } \preceq W )$ cannot reduce the number of feasible semantic clusters, hence

$$
K ( W ^ { \prime } ) \ge K ( W ) .
$$

By Theorem $\mathsf { A } . 4 ( 6 ) , \mathbb { E } [ C _ { N } ( W ) ]$ is a monotone nondecreasing function of $K ( W )$ , and therefore

$$
\mathbb { E } [ C _ { N } ( W ^ { \prime } ) ] ~ \ge ~ \mathbb { E } [ C _ { N } ( W ) ]
$$

In particular, under uniform cluster probabilities (Theorem A.4(5)), increasing K(W) drives $\mathbb { E } [ C _ { N } ( W ) ]$ upward towards N.

Combining (i) and (ii), the elementwise weakening of cross-temperature agreement induced by increasing semantic uniqueness of the high-temperature answers monotonically increases $U _ { \mathrm { n o r m } } ( W ; \theta )$ and simultaneously increases the expected number of observed semantic clusters. This supports the interpretation of $\mathcal { U } _ { \sf n o r m } ( W ; \theta )$ as a cross-temperature semantic instability score: as high-temperature responses spread across more semantic clusters, the uncertainty score increases- obeying the fundamental notion of uncertainty. □

## A.2 Full strawman-baseline analysis

Table 3 reports the full per-model comparison between BiG-SURE and direct aggregation baselines computed from the same cross-temperature anchor– probe similarity matrix W. This analysis is designed to isolate the contribution of the aggregation rule, since all methods use the same generated samples and the same entailment-based similarity scores.

We compare the following variants:

• BiG-SURE: the proposed normalized squared spectral-energy score, computed using all singular values of W.

• AvgSim: the mean anchor–probe similarity, $\begin{array} { r } { \frac { 1 } { M N } \sum _ { i , j } W _ { i j } } \end{array}$

• AvgSqSim: the mean squared anchor–probe similarity, $\begin{array} { r } { \frac { 1 } { M N } \sum _ { i , j } W _ { i j } ^ { 2 } } \end{array}$ . Since $\begin{array} { r } { \sum _ { r } \bar { \sigma } _ { r } ^ { 2 } \ = } \end{array}$ $\begin{array} { r } { \| W \| _ { F } ^ { 2 } = \sum _ { i , j } W _ { i j } ^ { 2 } } \end{array}$ , this is numerically identical to BiG-SURE.

• MaxSim: the maximum anchor–probe similarity score.

• Anchor-probe Ent.: an entropy-style statistic computed from the anchor–probe similarity distribution.

The results show that BiG-SURE and AvgSqSim match exactly, as expected from the Frobeniusenergy identity. The direct controls are weaker on average. AvgSim and MaxSim underperform BiG-SURE on both datasets, while anchor–probe entropy performs substantially worse. These results support the use of squared cross-temperature agreement as a compact reliability signal.

## A.3 Input-augmentation ablations

We provide the full per-model ablation of input variation strategies in Table 4. The low-temperature anchors are sampled from the original input in all variants. The variants differ only in how the hightemperature probe set is constructed.

Table 3: Comparison with direct aggregation baselines and truncated spectral variants using entailment-based similarity. Results are reported as mean ± standard deviation over 5 seeds. $\operatorname { A v g S q S i m }$ is numerically identical to BiG-SURE because normalized spectral energy equals the normalized squared Frobenius norm of the anchor–probe similarity matrix.
<table><tr><td>Model</td><td>BiG-SURE</td><td>Top-1</td><td>Top-2</td><td>AvgSim</td><td>AvgSqSim</td><td>MaxSim</td><td>Anchor-probe Ent.</td></tr><tr><td colspan="8"></td></tr><tr><td>Llama-8B</td><td> $\overline { { { \bf 0 . 8 8 9 7 \pm 0 . 0 2 6 1 } } }$ </td><td> $\overline { { 0 . 8 8 1 2 \pm 0 . 0 2 5 6 } }$ </td><td> $\overline { { 0 . 8 8 9 1 \pm 0 . 0 2 6 1 } }$ </td><td> $\overline { { 0 . 8 8 5 6 \pm 0 . 0 3 2 9 } }$ </td><td> $\overline { { { \bf 0 . 8 8 9 7 \pm 0 . 0 2 6 1 } } }$ </td><td> $\overline { { 0 . 8 8 3 0 \pm 0 . 0 3 2 1 } }$ </td><td> $\overline { { 0 . 4 4 5 5 \pm 0 . 0 1 1 1 } }$ </td></tr><tr><td>Qwen-7B</td><td> $\mathbf { 0 . 8 0 3 9 \pm 0 . 0 1 6 7 }$ </td><td> $0 . 8 0 2 2 \pm 0 . 0 1 3 4$ </td><td> $0 . 8 0 3 1 \pm 0 . 0 1 6 1$ </td><td> $0 . 8 0 3 7 \pm 0 . 0 1 8 5$ </td><td> $\mathbf { 0 . 8 0 3 9 \pm 0 . 0 1 6 7 }$ </td><td> $0 . 7 9 9 6 \pm 0 . 0 2 2 3$ </td><td> $0 . 4 6 9 9 \pm 0 . 0 2 1 4$ </td></tr><tr><td>Llama-70B</td><td> $0 . 9 0 5 1 \pm 0 . 0 1 3 2$ </td><td> $\mathbf { 0 . 9 0 5 3 \pm 0 . 0 1 4 5 }$ </td><td> $0 . 9 0 5 1 \pm 0 . 0 1 3 2$ </td><td> $0 . 9 0 2 6 \pm 0 . 0 1 0 5$ </td><td> $0 . 9 0 5 1 \pm 0 . 0 1 3 2$ </td><td> $0 . 9 0 3 7 \pm 0 . 0 0 8 4$ </td><td> $0 . 4 8 9 9 \pm 0 . 0 1 4 9$ </td></tr><tr><td>Qwen-72B</td><td> $\mathbf { 0 . 9 0 1 8 \pm 0 . 0 3 9 2 }$ </td><td> $0 . 9 0 1 7 \pm 0 . 0 3 2 5$ </td><td> $0 . 9 0 1 8 \pm 0 . 0 3 8 0$ </td><td> $0 . 9 0 1 1 \pm 0 . 0 3 5 6$ </td><td> $\mathbf { 0 . 9 0 1 8 \pm 0 . 0 3 9 2 }$ </td><td> $0 . 9 0 0 2 \pm 0 . 0 3 4 2$ </td><td> $0 . 3 7 1 4 \pm 0 . 0 1 2 4$ </td></tr><tr><td>Gemma-12B</td><td> $0 . 8 8 2 1 \pm 0 . 0 2 0 8$ </td><td> $0 . 8 7 8 6 \pm 0 . 0 2 1 7$ </td><td> $0 . 8 8 2 0 \pm 0 . 0 2 0 8$ </td><td> $\mathbf { 0 . 8 8 3 2 \mathop { \pm } 0 . 0 1 8 0 }$ </td><td> $0 . 8 8 2 1 \pm 0 . 0 2 0 8$ </td><td> $0 . 8 8 1 5 \pm 0 . 0 1 8 7$ </td><td> $0 . 4 4 1 4 \pm 0 . 0 1 3 1$ </td></tr><tr><td>Gemma-27B</td><td> $\underline { { 0 . 8 7 8 6 \pm 0 . 0 1 6 8 } }$ </td><td> $\mathbf { 0 . 8 7 8 8 \pm 0 . 0 1 5 6 }$ </td><td> $0 . 8 7 8 5 \pm 0 . 0 1 6 2$ </td><td> $\mid 0 . 8 6 6 1 \pm 0 . 0 2 3 7$ </td><td> $0 . 8 7 8 6 \pm 0 . 0 1 6 8$ </td><td> $0 . 8 6 6 2 \pm 0 . 0 2 3 6$ </td><td> $0 . 4 7 6 6 \pm 0 . 0 1 5 1$ </td></tr><tr><td>Average</td><td> $\overline { { { \bf 0 . 8 7 6 9 \pm 0 . 0 2 2 1 } } }$ </td><td> $\overline { { 0 . 8 7 4 6 \pm 0 . 0 2 0 5 } }$ </td><td> $\overline { { 0 . 8 7 6 6 \pm 0 . 0 2 1 7 } }$ </td><td> $\overline { { 0 . 8 7 3 7 \pm 0 . 0 2 3 2 } }$ </td><td> $\overline { { { \bf 0 . 8 7 6 9 \pm 0 . 0 2 2 1 } } }$ </td><td> $\overline { { 0 . 8 7 2 4 \pm 0 . 0 2 3 2 } }$ </td><td> $\overline { { 0 . 4 4 9 1 \pm 0 . 0 1 4 7 } }$ </td></tr><tr><td colspan="8">Trivia-QA</td></tr><tr><td>Llama-8B</td><td> $\overline { { { \bf 0 . 7 7 2 0 \pm 0 . 0 2 1 2 } } }$ </td><td> $\overline { { 1 . 7 7 1 6 \pm 0 . 0 1 8 5 } }$ </td><td> $\overline { { 0 . 7 7 2 0 \pm 0 . 0 2 1 2 } }$ </td><td> $\overline { { 1 . 7 6 2 4 \pm 0 . 0 2 3 7 } }$ </td><td> $\overline { { { \bf 0 . 7 7 2 0 \pm 0 . 0 2 1 2 } } }$ </td><td> $\overline { { 0 . 7 5 8 6 \pm 0 . 0 2 2 8 } }$ </td><td> $\overline { { 0 . 4 0 8 5 \pm 0 . 0 1 1 6 } }$ </td></tr><tr><td>Qwen-7B</td><td> $\mathbf { 0 . 8 0 6 0 \pm 0 . 0 1 4 1 }$ </td><td> $0 . 8 0 5 2 \pm 0 . 0 1 6 2$ </td><td> $0 . 8 0 5 9 \pm 0 . 0 1 3 8$ </td><td> $0 . 8 0 5 9 \pm 0 . 0 1 5 3$ </td><td> $\mathbf { 0 . 8 0 6 0 \pm 0 . 0 1 4 1 }$ </td><td> $0 . 8 0 2 6 \pm 0 . 0 1 7 0$ </td><td> $0 . 4 3 9 0 \pm 0 . 0 0 5 5$ </td></tr><tr><td>Llama-70B</td><td> $0 . 7 1 4 5 \pm 0 . 0 0 8 7$ </td><td> $0 . 7 1 3 9 \pm 0 . 0 0 9 5$ </td><td> $0 . 7 1 4 4 \pm 0 . 0 0 6 0$ </td><td> $\mathbf { 0 . 7 1 8 8 \pm 0 . 0 1 7 9 }$ </td><td> $0 . 7 1 4 5 \pm 0 . 0 0 8 7$ </td><td> $0 . 7 1 5 6 \pm 0 . 0 1 9 1$ </td><td> $0 . 4 7 3 1 \pm 0 . 0 1 1 1$ </td></tr><tr><td>Qwen-72B</td><td> $0 . 7 3 9 7 \pm 0 . 0 2 9 5$ </td><td> $\mathbf { 0 . 7 3 9 8 \pm 0 . 0 1 8 8 }$ </td><td> $0 . 7 3 9 7 \pm 0 . 0 2 9 5$ </td><td> $0 . 7 3 2 0 \pm 0 . 0 3 4 4$ </td><td> $0 . 7 3 9 7 \pm 0 . 0 2 9 5$ </td><td> $0 . 7 2 4 8 \pm 0 . 0 3 4 5$ </td><td> $0 . 4 1 2 4 \pm 0 . 0 1 8 3$ </td></tr><tr><td>Gemma-12B</td><td> $\mathbf { 0 . 7 8 8 6 \pm 0 . 0 1 1 0 }$ </td><td> $0 . 7 8 6 5 \pm 0 . 0 2 4 4$ </td><td> $0 . 7 8 8 3 \pm 0 . 0 1 6 2$ </td><td> $0 . 7 8 7 1 \pm 0 . 0 0 7 0$ </td><td> $\mathbf { 0 . 7 8 8 6 \pm 0 . 0 1 1 0 }$ </td><td> $0 . 7 8 6 8 \pm 0 . 0 0 5 1$ </td><td> $0 . 4 9 7 4 \pm 0 . 0 1 1 3$ </td></tr><tr><td>Gemma-27B</td><td> $\mathbf { 0 . 6 9 9 1 \pm 0 . 0 1 6 5 }$ </td><td> $0 . 6 9 8 9 \pm 0 . 0 1 4 9$ </td><td> $0 . 6 9 9 1 \pm 0 . 0 1 6 5$ </td><td> $0 . 6 8 7 2 \pm 0 . 0 2 0 8$ </td><td> $\mathbf { 0 . 6 9 9 1 \pm 0 . 0 1 6 5 }$ </td><td> $0 . 6 7 7 3 \pm 0 . 0 1 8 8$ </td><td> $0 . 5 1 2 1 \pm 0 . 0 3 1 1$ </td></tr><tr><td>Average</td><td> $\mathbf { 0 . 7 5 3 3 \pm 0 . 0 1 6 8 }$ </td><td> $\overline { { 0 . 7 5 2 7 \pm 0 . 0 1 7 0 } }$ </td><td></td><td> $\overline { { 0 . 7 5 3 2 \pm 0 . 0 1 7 2 \ \mathrm { ~ } _ { 1 } \ 0 . 7 4 8 9 \pm 0 . 0 1 9 8 } }$ </td><td> $\mathbf { 0 . 7 5 3 3 \pm 0 . 0 1 6 8 }$ </td><td> $\overline { { 0 . 7 4 4 3 \pm 0 . 0 1 9 6 } }$ </td><td> $\overline { { 0 . 4 5 7 1 \pm 0 . 0 1 4 8 } }$ </td></tr></table>

• Non-Rephrased: probes are generated only from the original input using high-temperature sampling. No equivalent-input augmentation is used.

• Rephrased-only: In this case, 10 input rephrasings are used instead of 5, and for each of them high-temperature sampling is used once.

• Same-Input: each equivalent rephrased input is treated independently. For each rephrase, we generate 10 high-temperature probes, compute the SURE uncertainty score, evaluate AU-ROC, and then average the AUROCs across rephrases.

• SURE (Mixed-Rephrased): this is the default setting. We use 5 equivalent input augmentations and sample 2 high-temperature responses from each, forming one mixed probe pool of 10 responses.

Table 4: Ablation of input variation strategies for SURE. We compare Non-Rephrased (no input augmentation), Same-Input (multiple generations from the same input), and Mixed (Rephrased) (our default setting, using semantically equivalent rephrased inputs). Results are reported as abstention AUROC (mean ± std over 5 seeds). The best result in each row is shown in bold, and the second-best result is underlined.
<table><tr><td rowspan="2">Model</td><td colspan="3">AUROC (↑)</td></tr><tr><td>Non-Rephrased</td><td>Same-Input</td><td>SURE (Mixed-Rephrased)</td></tr><tr><td colspan="4">SVAMP</td></tr><tr><td>Gemma-12B</td><td> $\overline { { 0 . 7 7 7 1 \pm 0 . 0 0 9 2 } }$ </td><td> $\overline { { 0 . 8 5 3 6 \pm 0 . 0 2 2 9 } }$ </td><td> $\overline { { { \bf 0 . 8 8 2 1 } \pm 0 . 0 2 0 8 } }$ </td></tr><tr><td>Gemma-27B</td><td> $0 . 7 7 6 1 \pm 0 . 0 0 8 7$ </td><td> $0 . 8 4 7 0 \pm 0 . 0 2 4 2$ </td><td> $\mathbf { 0 . 8 7 8 6 \pm 0 . 0 1 6 8 }$ </td></tr><tr><td>Llama-8B</td><td></td><td>0.7050 ± 0.0394 0.8989 ± 0.0239</td><td>0.8897 ± 0.0261</td></tr><tr><td>Llama-70B</td><td> $0 . 8 8 9 3 \pm 0 . 0 0 7 4$ </td><td> $\underline { { 0 . 9 0 3 1 \pm 0 . 0 1 1 8 } }$ </td><td> $\overline { { 0 . 9 0 5 1 \pm 0 . 0 1 3 2 } }$ </td></tr><tr><td>Qwen-7B</td><td> $0 . 7 6 1 9 \pm 0 . 0 2 0 0$ </td><td> $\overline { { 0 . 7 8 5 8 \pm 0 . 0 1 0 0 } }$ </td><td> $\mathbf { 0 . 8 0 3 9 \pm 0 . 0 1 6 7 }$ </td></tr><tr><td>Qwen-72B</td><td>0.9011 ± 0.0110</td><td>0.9102 ± 0.0211</td><td> $\underline { { 0 . 9 0 1 8 \pm 0 . 0 3 9 2 } }$ </td></tr><tr><td>Avg.</td><td>0.8018</td><td>0.8664</td><td>0.8769</td></tr><tr><td colspan="4">Trivia-QA</td></tr><tr><td>Gemma-12B</td><td> $\overline { { 0 . 6 9 7 8 \pm 0 . 0 1 1 7 } }$ </td><td> $\overline { { 0 . 7 6 0 6 \pm 0 . 0 1 1 3 } }$ </td><td>0.7886 ± 0.0110</td></tr><tr><td>Gemma-27B</td><td> $0 . 6 0 5 1 \pm 0 . 0 0 8 6$ </td><td> $0 . 6 8 9 6 \pm 0 . 0 1 6 6$ </td><td> $\mathbf { 0 . 6 9 9 1 \pm 0 . 0 1 6 5 }$ </td></tr><tr><td>Llama-8B</td><td> $0 . 7 4 3 5 \pm 0 . 0 0 6 8$ </td><td> $\textcircled { 0 . 7 8 0 4 } \pm \mathbf { 0 . 0 1 3 0 }$ </td><td> $\underline { { 0 . 7 7 2 0 \pm 0 . 0 2 1 2 } }$ </td></tr><tr><td>Llama-70B</td><td> $0 . 7 0 6 5 \pm 0 . 0 0 8 1$ </td><td> $\mathbf { 0 . 7 2 4 4 \pm 0 . 0 1 4 2 }$ </td><td> $0 . 7 1 4 5 \pm 0 . 0 0 8 7$ </td></tr><tr><td>Qwen-7B</td><td> $0 . 7 8 0 7 \pm 0 . 0 0 5 4$ </td><td> $\underline { { 0 . 7 8 8 5 \pm 0 . 0 0 8 7 } }$ </td><td> $\mathbf { 0 . 8 0 6 0 \pm 0 . 0 1 4 1 }$ </td></tr><tr><td>Qwen-72B</td><td> $0 . 7 0 9 6 \pm 0 . 0 1 0 4$ </td><td> $\overline { { 0 . 7 2 8 4 \pm 0 . 0 1 7 4 } }$ </td><td> $\mathbf { 0 . 7 3 9 7 \pm 0 . 0 2 9 5 }$ </td></tr><tr><td>Avg.</td><td>0.7072</td><td>0.7453</td><td>0.7533</td></tr></table>

## A.4 Matched inference-budget analysis

The Figure 7 also shows that using only rephrasings improve the performance across different methods as well. However, it comes with additional rephrasing cost. The results show that input augmentation is the main contributor: both Same-Input and Mixed-Rephrased substantially outperform Non-Rephrased on average. Mixed-Rephrased further improves over Same-Input on both dataset averages, suggesting that combining semantically equivalent perturbations within the same probe pool provides a modest but consistent additional benefit.

Our default SURE configuration generates M = 3 low-temperature anchors and $N ~ = ~ 1 0$ hightemperature probes, resulting in 13 total LLM generations per input. In contrast, several samplingbased baselines are commonly evaluated with 10 generations. To verify that the observed improvements are not simply due to this larger generation budget, we rerun the text-QA baselines using N = 13 samples. As reported in Table 5, SURE retains its performance advantage even when the baselines are given the same number of LLM calls, suggesting that the improvement comes from the cross-temperature anchor–probe construction rather than from additional sampling alone.

Table 5: Comparison of baseline methods using 13 samples. Results are reported as abstention AUROC (mean ± std over 5 seeds). The second column reports the base task accuracy. The best result in each row is highlighted in bold, and the second-best result is underlined.
<table><tr><td>Model</td><td>Acc.</td><td>DSE</td><td>SumEigv</td><td>Deg</td><td>NumSet</td><td>LexSim</td><td>KLE</td><td>SNNE</td><td>SURE</td></tr><tr><td colspan="10">Trivia-QA</td></tr><tr><td>Qwen-7B</td><td>0.495</td><td> $\overline { { 0 . 7 8 9 7 \pm 0 . 0 0 3 4 } }$ </td><td> $\overline { { 0 . 7 9 8 4 \pm 0 . 0 0 5 2 } }$ </td><td> $\overline { { 0 . 7 9 7 9 \pm 0 . 0 0 4 8 } }$ </td><td> $\overline { { 0 . 7 8 8 4 \pm 0 . 0 0 4 5 } }$ </td><td> $\overline { { 0 . 7 8 7 0 \pm 0 . 0 0 8 1 } }$ </td><td> $\overline { { 0 . 7 7 3 3 \pm 0 . 0 0 7 6 } }$ </td><td> $\overline { { 0 . 7 8 8 4 \pm 0 . 0 0 3 3 } }$ </td><td> $\overline { { { \bf 0 . 8 0 6 0 \pm 0 . 0 1 4 1 } } }$ </td></tr><tr><td>Llama-8B</td><td>0.661</td><td> $0 . 7 2 2 1 \pm 0 . 0 0 5 2$ </td><td> $0 . 7 2 0 3 \pm 0 . 0 0 5 7$ </td><td> $0 . 7 3 2 0 \pm 0 . 0 0 5 8$ </td><td> $0 . 7 1 2 3 \pm 0 . 0 0 7 2$ </td><td> $\underline { { 0 . 7 3 6 7 \pm 0 . 0 0 6 9 } }$ </td><td> $0 . 7 1 8 5 \pm 0 . 0 0 3 2$ </td><td> $0 . 7 3 2 5 \pm 0 . 0 0 6 8$ </td><td> $\mathbf { 0 . 7 7 2 0 \pm 0 . 0 2 1 2 }$ </td></tr><tr><td>Gemma-12B</td><td>0.643</td><td> $0 . 7 0 8 4 \pm 0 . 0 1 0 2$ </td><td> $0 . 7 0 7 8 \pm 0 . 0 1 0 7$ </td><td> $0 . 7 0 4 5 \pm 0 . 0 0 9 4$ </td><td> $0 . 7 0 9 4 \pm 0 . 0 1 0 0$ </td><td> $\underline { { 0 . 7 3 2 6 \pm 0 . 0 1 4 4 } }$ </td><td> $0 . 6 9 7 4 \pm 0 . 0 1 2 2$ </td><td> $0 . 7 1 8 2 \pm 0 . 0 1 4 2$ </td><td> $\mathbf { 0 . 7 8 8 6 \pm 0 . 0 1 1 0 }$ </td></tr><tr><td>Gemma-27B</td><td>0.755</td><td> $0 . 6 3 1 7 \pm 0 . 0 0 5 5$ </td><td> $0 . 6 3 6 4 \pm 0 . 0 0 3 3$ </td><td> $0 . 6 3 5 3 \pm 0 . 0 0 3 0$ </td><td> $0 . 6 3 0 5 \pm 0 . 0 0 5 4$ </td><td> $0 . 6 5 8 4 \pm 0 . 0 0 5 7$ </td><td> $0 . 6 2 9 0 \pm 0 . 0 0 6 4$ </td><td> $0 . 6 1 8 9 \pm 0 . 0 1 0 2$ </td><td> $\mathbf { 0 . 6 9 9 1 \pm 0 . 0 1 6 5 }$ </td></tr><tr><td>Llama-70B</td><td>0.767</td><td> $0 . 7 2 8 5 \pm 0 . 0 0 8 6$ </td><td> $0 . 7 5 8 2 \pm 0 . 0 0 9 7$ </td><td> $\mathbf { 0 . 7 6 1 7 \pm 0 . 0 0 9 5 }$ </td><td> $0 . 7 1 5 7 \pm 0 . 0 1 1 7$ </td><td> $0 . 7 4 1 9 \pm 0 . 0 0 7 1$ </td><td> $0 . 7 0 6 4 \pm 0 . 0 1 0 9$ </td><td> $0 . 7 1 8 7 \pm 0 . 0 0 6 3$ </td><td> $0 . 7 1 4 5 \pm 0 . 0 0 8 7$ </td></tr><tr><td>Qwen-72B</td><td>0.690</td><td> $0 . 6 9 8 8 \pm 0 . 0 1 2 5$ </td><td> $0 . 7 2 3 2 \pm 0 . 0 0 5 1$ </td><td> $0 . 7 1 6 7 \pm 0 . 0 0 5 4$ </td><td> $0 . 6 7 7 9 \pm 0 . 0 0 8 7$ </td><td> $0 . 6 3 2 6 \pm 0 . 0 0 3 1$ </td><td> $0 . 6 9 9 9 \pm 0 . 0 0 4 2$ </td><td> $0 . 7 0 0 6 \pm 0 . 0 0 8 7$ </td><td> $\mathbf { 0 . 7 3 9 7 \pm 0 . 0 2 9 5 }$ </td></tr><tr><td>Avg.</td><td>0.669</td><td>0.7132</td><td>0.7240</td><td>0.7247</td><td>0.7057</td><td>0.7149</td><td>0.7041</td><td>0.7129</td><td>0.7533</td></tr><tr><td colspan="10">SVAMP 一</td></tr><tr><td>Qwen-7B</td><td>0.688</td><td> $\overline { { 0 . 7 1 5 2 \pm 0 . 0 2 7 1 } }$ </td><td> $\overline { { 0 . 7 8 3 9 \pm 0 . 0 1 9 9 } }$ </td><td> $\overline { { 0 . 7 7 7 9 \pm 0 . 0 1 8 6 } }$ </td><td> $\overline { { 0 . 7 8 3 8 \pm 0 . 0 2 7 9 } }$ </td><td> $\overline { { 0 . 7 5 7 2 \pm 0 . 0 2 0 2 } }$ </td><td> $\overline { { 0 . 7 8 7 1 \pm 0 . 0 1 8 5 } }$ </td><td> $\overline { { 0 . 7 7 4 5 \pm 0 . 0 2 9 7 } }$  一</td><td> $\mathbf { \overline { { 0 . 8 0 3 9 \pm 0 . 0 1 6 7 } } }$ </td></tr><tr><td>Llama-8B</td><td>0.672</td><td> $0 . 6 2 0 3 \pm 0 . 0 2 2 4$ </td><td> $0 . 8 5 9 8 \pm 0 . 0 2 5 7$ </td><td> $0 . 8 6 1 7 \pm 0 . 0 2 6 7$ </td><td> $0 . 8 7 4 7 \pm 0 . 0 2 1 4$ </td><td> $0 . 8 5 9 5 \pm 0 . 0 1 6 9$ </td><td> $0 . 8 6 2 6 \pm 0 . 0 1 9 1$ </td><td> $0 . 8 8 2 5 \pm 0 . 0 2 0 5$ </td><td> $\mathbf { 0 . 8 8 9 7 \pm 0 . 0 2 6 1 }$ </td></tr><tr><td>Gemma-12B</td><td>0.730</td><td> $0 . 7 7 3 1 \pm 0 . 0 1 6 2$ </td><td> $\underline { { 0 . 7 9 7 9 \pm 0 . 0 1 5 7 } }$ </td><td> $0 . 7 9 9 8 \pm 0 . 0 1 6 2$ </td><td> $0 . 7 9 0 2 \pm 0 . 0 1 7 9$ </td><td> $0 . 7 9 4 0 \pm 0 . 0 1 6 6$ </td><td> $0 . 7 9 4 0 \pm 0 . 0 1 4 9$ </td><td> $0 . 8 0 1 1 \pm 0 . 0 1 5 6$ </td><td> $\mathbf { 0 . 8 8 2 1 \pm 0 . 0 2 0 8 }$ </td></tr><tr><td>Gemma-27B</td><td>0.794</td><td> $0 . 7 4 8 2 \pm 0 . 0 1 4 8$ </td><td> $0 . 7 5 6 4 \pm 0 . 0 1 1 3$ </td><td> $0 . 7 5 7 3 \pm 0 . 0 1 1 1$ </td><td> $0 . 7 4 0 4 \pm 0 . 0 1 4 0$ </td><td> $0 . 7 4 2 6 \pm 0 . 0 1 4 6$ </td><td> $0 . 7 4 2 2 \pm 0 . 0 1 5 0$ </td><td> $\underline { { 0 . 8 0 3 9 \pm 0 . 0 1 2 3 } }$ </td><td> $\mathbf { 0 . 8 7 8 6 \pm 0 . 0 1 6 8 }$ </td></tr><tr><td>Llama-70B</td><td>0.797</td><td> $0 . 8 7 8 8 \pm 0 . 0 1 1 2$ </td><td> $0 . 8 8 4 4 \pm 0 . 0 1 3 4$ </td><td> $0 . 8 8 7 2 \pm 0 . 0 1 3 0$ </td><td> $0 . 8 8 8 5 \pm 0 . 0 0 7 9$ </td><td> $0 . 8 7 5 9 \pm 0 . 0 0 7 3$ </td><td> $0 . 8 8 9 6 \pm 0 . 0 1 1 6$ </td><td> $0 . 8 9 8 2 \pm 0 . 0 1 1 5$ </td><td> $\mathbf { 0 . 9 0 5 1 } \pm \mathbf { 0 . 0 1 3 2 }$ </td></tr><tr><td>Qwen-72B</td><td>0.849</td><td> $0 . 9 0 2 0 \pm 0 . 0 2 2 5$ </td><td> $\mathbf { 0 . 9 2 3 5 \pm 0 . 0 0 8 8 }$ </td><td> $0 . 9 2 3 4 \pm 0 . 0 0 9 4$ </td><td> $0 . 8 8 3 7 \pm 0 . 0 2 2 1$ </td><td> $0 . 8 0 7 4 \pm 0 . 0 1 3 8$ </td><td> $0 . 8 4 2 1 \pm 0 . 0 2 2 5$ </td><td> $\overline { { 0 . 9 0 6 5 \pm 0 . 0 0 9 8 } }$ </td><td> $0 . 9 0 1 8 \pm 0 . 0 3 9 2$ </td></tr><tr><td>Avg.</td><td>0.755</td><td>0.7729</td><td>0.8343</td><td>0.8346</td><td>0.8269</td><td>0.8061</td><td>0.8196</td><td>0.8445</td><td>0.8769</td></tr></table>

![](images/901fab2594ca3b1c94e0ac04c158d90564b9075de803b595b08a96560547e3b8.jpg)  
Figure 7: Effect of Rephrase-only ablation on all methods. Using Rephrase-only inputs for different methods improves them most of the time, however it comes with 2X rephrase cost.

## A.5 Temperature sensitivity analysis

![](images/c3044085975680ea31352cdf4c4c2e844d91542065bb646f61d631e682bfa528.jpg)  
Figure 8: Temperature sensitivity analysis for SURE. Abstention AUROC for different anchor temperatures (T<sub>l</sub>) and probe temperatures $( T _ { h } )$ on Trivia-QA and SVAMP. Performance is relatively stable across lowtemperature anchor choices, while very high probe temperatures, especially $T _ { h } = 2 . 0$ , reduce AUROC.

We evaluate the sensitivity of SURE to the choice of low-temperature anchor sampling temperature T<sub>L</sub> and high-temperature probe sampling temperature $T _ { H }$ Figure 8 reports abstention AUROC on SVAMP and Trivia-QA for $T _ { L } \in$ $\{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \}$ and $T _ { H } \in \{ 1 . 0 , 1 . 2 , 1 . 5 , 2 . 0 \}$ SURE is relatively insensitive to the exact choice of $T _ { L }$ within the low-temperature range: for a fixed $T _ { H }$ , changing $T _ { L }$ produces only small variations in AUROC. This supports the role of low-temperature samples as stable semantic anchors. Second, performance is more sensitive to the probe temperature $T _ { H }$ . Moderate probe temperatures, especially

$T _ { H } ~ = ~ 1 . 0$ and $T _ { H } = 1 . 2 .$ , perform best, while very high temperatures such as $T _ { H } = 2 . 0$ degrade performance substantially. This suggests that the probes should introduce semantic diversity without becoming excessively noisy. We use the $T _ { L } = 0 . 1$ and $T _ { H } = 1 . 0$ in all results in the paper, without tuning.

## A.6 Input augmentation prompts and implementation details

This section provides the implementation details for the equivalent-input augmentations used in SURE. For textual inputs, we use LLM-based rephrasing to create semantically equivalent versions of the original question. The rephrased questions are used only to generate high-temperature probe responses; the low-temperature anchors are generated from the original input. This preserves the interpretation of anchors as stable responses to the original question, while allowing the probes to test semantic stability under both stochastic decoding and meaning-preserving input variation. The prompts used to create semantics-preserving textual augmentations using Llama-3.1-8B-Instruct (for text-QA) and Gemini-2.5-Flash (for multilingual QA) are shown below. For multimodal QA, we apply both textual and visual input augmentations. The textual component rephrases the question while keeping the associated image fixed. The visual component perturbs the image while preserving its semantic content. We use lightweight image transformations including mean blur with kernel size 5, affine rotation of ±10<sup>◦</sup> around the image center, and Gaussian pixel noise sampled from N (0, 10) followed by clipping to the valid pixel range [0, 255]. These transformations are intentionally mild: they are designed to probe robustness to small input changes without changing the answer semantics. Together, these augmentations allow SURE to test whether high-temperature probe responses remain aligned with stable lowtemperature anchors under equivalent input variations.

## Prompt for Gemini: Multilingual Rephrasing

## System instruction:

You are a multilingual question rephrasing assistant. Your task is to rephrase questions in multiple languages using only the information present in the original questions. You must never add external knowledge.

## Strict rules:

1. Use only words and concepts from the original question; do not add any external knowledge.

2. Do not add descriptors that hint at the answer.

3. Keep all specific names, titles, dates, and key terms from the original exactly as written.

4. Each rephrase should ask for the exact same information—no more, no less.

5. Maintain the same language style and formality level for each language.

6. Generate rephrasings for each language.

## Input template:

Original {questions\_text}

questions:

## Output format:

[english] rephrase here [chinese] rephrase here [japanese] rephrase here [french] rephrase here

Instruction: Provide the output in exactly the format above.

## Prompt for Llama: English Rephrasing

## System instruction:

You are a question rephrasing assistant. You rephrase questions using only the information present in the original question. You must never add external knowledge.

## User prompt:

Rephrase the following question.

## Strict rules:

1. Use only words and concepts from the original question; do not add any external knowledge.

2. Do not add descriptors that hint at the answer.

3. Keep all specific names, titles, and key terms from the original exactly as written.

4. Each rephrase should ask for the exact same information—no more, no less.

Input template:

Original question: {question}

Output instruction:

Output exactly the rephrased question.

## A.7 Recruiting human subjects

Participants for the human-evaluation studies were recruited through Prolific, an online participantrecruitment platform. For the multilingual evaluations, we used Prolific’s language-based prescreening filters to recruit participants. Participants were compensated through Prolific according to the platform’s standard compensation rates. Recruitment, communication, and payment were handled through the platform. Prolific-assigned participant identifiers were used throughout the studies, and no directly identifying information, such as participants’ names or contact details, was collected by the authors. Participant identities therefore remained anonymous to the authors.

## A.8 Evaluation of greedy responses

Evaluation of greedy responses for the 3 tasks of textual QA, multilingual QA and multimodal QA is performed as below:

Textual QA: The evaluation of greedy responses is performed using the SQuAD F1 score metric, followed by using 0.5 threshold to make it binary as suggested in the prior works (Farquhar et al., 2024; Nguyen et al., 2025).

Multimodal QA: For this, the correctness is computed based on agreement with the multiple human annotations provided in the dataset, like prior works (Antol et al., 2015; Marino et al., 2019).

Multilingual QA: We use Gemini-2.5-Flash as a judge for evaluating responses in multilingual QA.

The prompt used for the purpose is stated below,

Prompt for Gemini-2.5-Flash: Multilingual   
Correctness Judging   
System instruction:   
You are evaluating science questions across multiple lan  
guages. Determine if each predicted answer matches the   
ground truth. Be flexible with extra explanation, but the   
core answer must be correct.   
Output format:   
For each language below, decide if   
the predicted answer is correct (1)   
or incorrect (0). Return ONLY a   
JSON object mapping each language   
code to 1 or 0.   
Example response: {"en": 1, "zh":   
0, "ja": 1, "fr": 0, "th": 1}   
Input template:   
=== {lang} ===   
Question: {question}   
Ground truth: {ground\_truth}   
Predicted:   
{first\_greedy\_prediction}   
Output instruction:   
Return ONLY the JSON object. No   
extra text.

## A.9 Area Under the Risk–Coverage Curve

In addition to the abstention AUROC, we evaluate selective-prediction performance using the Area Under the Risk–Coverage Curve (AURC). For each uncertainty estimator, predictions are ordered from the most to the least confident. At each coverage level, the risk is computed as the error rate among the retained predictions. A lower AURC value is better. Table 6 reports AURC averaged across the six text-QA models on SVAMP and TriviaQA. BiG-SURE achieves the lowest AURC on both datasets. These results are consistent with the abstention-AUROC findings.

## A.10 Failure analysis of SURE

We analyze representative failure cases in which textual semantic agreement can produce misleading uncertainty estimates. We discuss two distinct failure modes: (i) Case I: The LLM mispredicts and produces different semantically equivalent forms of the same incorrect responses during sampling, and (ii) Case II: The semantic-similarity backend incorrectly identifies non-equivalent responses as equivalent.

Table 6: AURC averaged across the six text-QA models. Values are reported as mean ± standard deviation across models. Lower values indicate better selectiveprediction performance.  
Method SVAMP AURC ↓ TriviaQA AURC ↓   
DSE $0 . 1 0 9 7 \pm 0 . 0 4 4 1$ $0 . 2 2 4 4 \pm 0 . 0 5 7 7$   
SumEigV $0 . 1 1 6 9 \pm 0 . 0 4 8 7$ $0 . 2 2 0 9 \pm 0 . 0 5 6 0$   
Deg $0 . 1 1 6 9 \pm 0 . 0 4 8 7$ $0 . 2 2 0 9 \pm 0 . 0 5 6 2$   
NumSet 0.1104 ± 0.0439 $0 . 2 2 3 0 \pm 0 . 0 5 9 6$   
LexSim 0.1146 ± 0.0394 $\underline { { 0 . 2 1 2 6 \pm 0 . 0 5 7 4 } }$   
KLE 0.1212 ± 0.0398 $0 . 2 1 7 6 \pm 0 . 0 6 1 9$   
SNNE 0.1017 ± 0.0391 $0 . 2 3 2 8 \pm 0 . 0 5 3 5$   
BiG-SURE 0.0839 ± 0.0328 0.1975 ± 0.0520

![](images/12dbff4d799a1e538718fc72af2f380554104f0741a31cc105a856c9aff0c0ed.jpg)  
Figure 9: A representative visual-QA failure involving semantically consistent but incorrect responses. For the displayed image, the VLM produces “Louis Philippe” and “Louis Philippe Suits”, whereas the acceptable reference answers include Brooks Brothers, Giorgio Armani, and Gucci. The high NLI entailment score (0.9863) correctly reflects agreement between the generated responses, but cannot determine whether they are visually grounded.

Case I: Figure 9 shows an example from visual QA. For the question “What brand of suit is the man in the image wearing?”, the acceptable reference answers include Brooks Brothers, Giorgio Armani, and Gucci. However, the VLM repeatedly generates “Louis Philippe” and “Louis Philippe Suits”. The NLI backend assigns these responses an entailment score of 0.9863 and estimates very low uncertainty. This is the case of consistent incorrect predictions. This should not be interpreted as an NLI-backend error: the two generated responses are semantically equivalent, and therefore the high entailment score is reasonable. Instead, it illustrates a fundamental limitation of response-consistencybased uncertainty estimation. This limitation is especially important in visual QA because the current semantic backend compares only the textual responses and does not examine whether they are grounded in the image.

Case II A separate failure mode occurs when the NLI backend assigns high entailment to responses that are not semantically equivalent. For example, for a Chinese SciQ question asking what type of energy is released when iron filings react with air, the responses corresponding to “heat” and “combustion” receive an entailment score of 0.8453. Although the concepts are related, combustion is a process rather than the requested form of energy, and the two answers are not interchangeable. We observe a similar issue for numerical and frequency-based answers. For a Japanese SciQ question asking how often partial lunar eclipses occur, the responses corresponding to “more than once per month” and “multiple times per year” receive an entailment score of 0.9895, despite expressing substantially different frequencies.

The two failure modes have different origins. Semantically consistent but incorrect predictions cannot be resolved by improving textual NLI alone and instead require grounding-aware comparison with the original input, including the image in multimodal settings. Semantic-backend errors may be mitigated through stronger multilingual and numerical entailment models, and calls for improving such models. Although the same semantic backend is used for BiG-SURE and the similaritybased baselines in our experiments, such errors does not guarantee that scorer errors affect all methods equally; as their impact may vary across methods. Developing modality-aware and more reliable semantic-agreement measures therefore remains an important direction for future work.

## A.11 Use of AI assistants

We used an AI assistant- ChatGPT (Achiam et al., 2023) solely as a language-editing aid. Its use was limited to identifying grammatical errors and typographical mistakes and suggesting localized revisions to improve clarity, and fluency. It was not used to generate ideas, formulate hypotheses, or design methods. All technical content, arguments, analyses, and conclusions were developed and verified by the authors, who also reviewed and approved all final edits.