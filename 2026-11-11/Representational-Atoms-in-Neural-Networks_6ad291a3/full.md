# Representational Atoms in Neural Networks

Chenhui Hu<sup>1,2</sup>, Pengfei Cao<sup>1,2</sup>, Kang Liu<sup>1,2</sup>, Jun Zhao<sup>1,2\*</sup>

<sup>1</sup>The Key Laboratory of Cognition and Decision Intelligence for Complex Systems,

Institute of Automation, Chinese Academy of Sciences, Beijing, China.

<sup>2</sup>School of Artificial Intelligence,

University of Chinese Academy of Sciences, Beijing, China.

\*Corresponding author: jzhao@nlpr.ia.ac.cn

## Abstract

Understanding the internal representations of neural networks and characterizing their relationship with human conceptual structure are two critical questions for elucidating model mechanisms. The former reveals how learned knowledge is organized in internal representations, whereas the latter provides a basis for interpreting these representations through human conceptual structure. Here, we formally define fundamental representational units (FRUs) and establish two evaluation criteria: fidelity and stability. We further prove that threshold-activated sparse autoencoders (TSAEs) can identify FRUs with high fidelity and stability under appropriate conditions, and validate this result through large-scale experiments. Further analyses reveal that FRUs with higher stability exhibit greater monosemanticity and can be composed across layers to recover semantic organization corresponding to human conceptual structure. These results provide a unified framework for defining, evaluating and identifying FRUs in neural networks, and establish a foundation for characterizing the relationship between neural network representations and human conceptual structure.

Neural networks progressively form internal representations during forward propagation, encoding inputrelated information that supports subsequent prediction or generation [1, 2]. However, how these representations encode and organize the information learned by the model remains incompletely understood [3, 4]. A fundamental question is whether this representational organization can be understood in terms of basic units that can be rigorously defined and reliably identified. If so, these units would provide a natural basis for characterizing the semantic organization of internal representations, raising a further question of whether and how this organization relates to human conceptual structure. Addressing these questions is essential for understanding neural representations and their relationship with human conceptual structure.

To answer the first question, we develop a theoretical framework for defining, evaluating and identifying fundamental representational units (FRUs) underlying internal representations. Drawing on compressed sensing [5, 6], we formulate internal representations as sparse linear combinations of FRUs: linearity provides a simple and analytically tractable form for characterizing how these FRUs combine, while sparsity permits them to remain distinguishable under appropriate conditions. On this basis, we formally define the FRUs as representational atoms, or atoms for brevity. This definition operates at the level of internal representation spaces and does not depend on particular neural network architectures.

Within this framework, we next evaluate how closely candidate representational units conform to the definition of representational atoms. Specifically, we introduce two evaluation criteria, fidelity and stability. Fidelity measures how well candidate FRUs reconstruct the original representations, whereas stability measures whether distinct candidate FRUs remain distinguishable from one another. We further establish that the distinguishability captured by stability is a prerequisite for monosemanticity, that is, for each FRU to support a unique semantic interpretation. These criteria enable two widely studied representational units, neurons and learned features [7–10], to be quantitatively compared as candidate FRUs within a unified framework. Our experiments show that neurons generally exhibit high fidelity but low stability, whereas learned features exhibit higher stability but lower fidelity. This contrast is consistent with previous observations that learned features tend to be more monosemantic than individual neurons [9–12].

The limitations of existing candidate FRUs raises a critical question: what kind of representation learning mechanism can reliably identify atoms with both high fidelity and stability? To address this question, we prove that threshold-activated sparse autoencoders (TSAEs) can efectively identify representational atoms under appropriate conditions. In practice, we instantiate TSAEs as BiJumpReLU SAEs and evaluate them on internal representations from diverse pretrained neural networks spanning diferent architectures and representation modalities. Across these settings, BiJumpReLU SAEs consistently identify atoms with high fidelity and stability, outperforming representative SAE variants spanning diferent activation and sparsity mechanisms [8, 13–15]. These large-scale experiments support our theoretical results and establish a practical basis for identifying and analyzing FRUs in neural networks. We further examine whether the theoretical connection between stability and monosemanticity is reflected empirically and find a significant positive correlation between the two, with more stable atoms exhibiting clearer and more consistent semantics.

We finally examine whether and how representational atoms relate to human conceptual structure using HowNet [16, 17], an expert-annotated knowledge base that represents concepts in terms of finegrained semantic primitives known as sememes. We find that their relationship is not one-to-one: although a small subset of atoms directly aligns with complete sememes, individual sememes are more commonly represented in a distributed manner by multiple atoms from diferent layers. These findings indicate that representational atoms function as finer-grained semantic components that can be composed into higher-order structures corresponding to human conceptual organization. Notably, their compositions recover not only sememe structures with substantially greater alignment than single atoms but also additional semantic structures that, upon manual verification, are consistent with the corresponding sememes yet not captured by their HowNet annotations. Overall, our work integrates the definition, evaluation and identification of FRUs with the analysis of their semantic and compositional organization, providing a systematic characterization of neural representational structure and its relationship with human conceptual structure.

## Results

In the following, we first formally define FRUs and introduce fidelity and stability as two evaluation criteria. We then quantitatively compare neurons and learned features as candidate FRUs within this unified framework. Next, we prove that TSAEs can efectively identify representational atoms under appropriate conditions and validate their practical efectiveness through large-scale experiments. We further examine the relationship between stability and monosemanticity, and finally use HowNet to characterize the relationship between neural representations and human conceptual structure.

## Representational atoms

Neurons and learned features are two widely studied candidates for the basic units of internal representations [7–10], yet neither provides a principled basis for determining what properties representational components must possess to qualify as FRUs. We therefore begin by formalizing this problem and developing a principled definition of FRUs.

Given a set of high-dimensional internal representations $M = \{ m _ { i } \ | \ m _ { i } \ \in \mathbb { R } ^ { H } \} _ { i = 1 } ^ { | M | }$ , we seek a set of FRUs $D = \{ d _ { j } \ | \ d _ { j } \in \mathbb { R } ^ { H } \} _ { j = 1 } ^ { | D | }$ . A natural starting point is the standard orthonormal basis of the representation space, treating the coordinate directions (i.e. neurons) as candidate FRUs. However, this choice has two limitations. First, individual coordinate directions need not support stable and welldefined semantics, consistent with the widespread polysemanticity of individual neurons [9–11]. Second, orthogonality restricts the number of candidate FRUs to the dimensionality of the representation space, limiting their ability to capture richer representational structure.

The key property of the standard orthonormal basis is that orthogonality makes distinct coordinate directions and their combinations unambiguously distinguishable, which ensures a unique decomposition of each representation. However, a key insight here is that orthogonality is not necessary for distinguishability. In particular, compressed sensing shows that sparsity can preserve the distinguishability among non-orthogonal components under appropriate conditions [5, 6], which we specify below. We therefore model each internal representation as a sparse linear combination

$$
\forall i \in [ | M | ] , \quad m _ { i } = D \delta _ { i } , \qquad \| \delta _ { i } \| _ { 0 } \leq K ,\tag{1}
$$

![](images/b4966569044053b78c9672ce8908cf819170c4ee05e9a0c5204b83aae6a106d9.jpg)  
Fig. 1 Overview of the framework for defining, evaluating and identifying representational atoms. a, Representational atoms are defined as fundamental representational units (FRUs) that faithfully reconstruct model representations while remaining distinguishable. This definition is characterized by three properties: representability, sparsity and separability under the atomic inner product (AIP). b, Candidate FRUs are evaluated along two quantitative dimensions. Fidelity, quantified by the coeficient of determination $R ^ { 2 } ,$ measures the reconstruction of the original representations, whereas stability, quantified by $q ^ { * }$ , measures the extent to which atoms remain distinguishable under the condition $\mu _ { q } < 1 / ( 2 K _ { q } - 1 )$ Higher values indicate better performance for both metrics, with $R ^ { 2 } \in ( - \infty , 1 ]$ and $q ^ { * } \in [ 0 , 1 ]$ . Stability captures the distinguishability of atoms and their compositions and constitutes a prerequisite for monosemanticity. c, Representational atoms are identified using threshold-activated sparse autoencoders (TSAEs). The encoder acts as a detector that activates atoms when its response exceeds the positive or negative threshold, while the corresponding decoder vectors constitute the atom set and reconstruct the original representations.

where $[ \vert M \vert ] = \{ 1 , 2 , \dotsc , \vert M \vert \} , D = [ d _ { 1 } \ d _ { 2 } \ \cdot \cdot \cdot \ d _ { \vert D \vert } ] \in \mathbb { R } ^ { H \times \vert D \vert }$ denotes the dictionary of candidate FRUs, $\pmb { \delta } _ { i } \in \mathbb { R } ^ { | D | }$ is the sparse coeficient vector associated with $m _ { i } , \lVert \cdot \rVert _ { 0 }$ denotes the number of non-zero entries, and $K \in \mathbb { N } ^ { + }$ is an upper bound on the sparsity level.

Candidate FRUs that satisfy the distinguishability condition derived below are termed representational atoms, or simply atoms (Fig. 1a). Characterizing this condition in the non-orthogonal setting requires an inner product that captures the relative geometry among candidate FRUs. The standard Euclidean inner product, $\langle d _ { i } , d _ { j } \rangle = d _ { i } ^ { \top } d _ { j }$ , characterizes geometry in the ambient representation space, but does not fully capture the geometry induced by internal representations (see ‘Representational atoms theoretical framework’ in Methods). We therefore introduce the atomic inner product (AIP), $\langle \pmb { d } _ { i } , \pmb { d } _ { j } \rangle _ { \tilde { S } } = \pmb { d } _ { i } ^ { \top } \tilde { S } \pmb { d } _ { j }$ , where $\tilde { S }$ denotes the representation-induced metric, to characterize their relative geometry (see ‘Atomic inner product’ in Methods). Empirical analyses further show that the AIP better captures the geometry relevant to representational distinguishability (see ‘Empirical validation of the atomic inner product’ section in Supplementary Information).

Using the AIP, we derive a condition under which candidate FRUs remain distinguishable in the non-orthogonal setting. Specifically, extending the classical coherence-based condition [18, 19], we prove that, for a K-sparse representation, distinguishability is preserved when $\mu ~ < ~ \frac { 1 } { 2 K - 1 }$ , where $\mu = \mathrm { m a x } _ { i \neq j } | \langle { \pmb d } _ { i } , { \pmb d } _ { j } \rangle _ { \tilde { S } } |$ denotes the maximum coherence among candidate FRUs (see ‘Evaluation of atoms’ in Methods). Within this regime, distinct units and their compositions remain distinguishable from alternative units or compositions, a property we term monorepresentationality. Monorepresentationality constitutes a necessary structural condition for monosemanticity, as each representational unit can support a unique semantic interpretation only when it remains distinguishable from alternative units or compositions. See ‘Evaluation of atoms’ in Methods for the formal definition and discussion of monorepresentationality.

To quantitatively assess how closely candidate FRUs conform to the properties of representational atoms, we introduce two complementary metrics, fidelity and stability (Fig. 1b). Fidelity measures how well candidate FRUs reconstruct the original internal representations and is quantified by the coeficient of determination

$$
R ^ { 2 } : = 1 - \frac { \sum _ { i } \| \pmb { m } _ { i } - \pmb { \hat { m } } _ { i } \| _ { 2 } ^ { 2 } } { \sum _ { i } \| \pmb { m } _ { i } - \pmb { \hat { m } } \| _ { 2 } ^ { 2 } } ,\tag{2}
$$

where $\hat { m } _ { i }$ denotes the reconstructed representation and m¯ denotes the mean of the internal representations (see ‘Fidelity metrics and computation’ in Methods). Stability measures the extent to which

![](images/95d3cf1fbcd0484fe124973c998a7e76f6b5c5af6c6622d7e7eaa2ecc38263f3.jpg)  
Fig. 2 Existing representational units exhibit distinct limitations in fidelity and stability. a–c, Fidelity $( R ^ { 2 }$ top) and stability $( q ^ { * }$ , bottom) of neurons and learned features across layers of Gemma2-2B (a), Gemma2-9B (b) and Llama3.1-8B (c). Angular coordinates denote layer indices, and radial coordinates denote the corresponding metric values. The dashed red circle marks the ideal-atom reference, for which both fidelity and stability attain their maximum value of 1. Neurons consistently exhibit high fidelity but substantially lower stability, whereas learned features from Gemma Scope and Llama Scope show improved stability to varying degrees but generally sacrifice fidelity, with neither class simultaneously approaching the ideal reference on both criteria. Across model scales and feature widths, these results reveal a systematic gap between existing candidate FRUs and the criteria for ideal representational atoms.

candidate FRUs satisfy the distinguishability condition and is quantified by the maximum quantile

$$
q ^ { * } : = \operatorname* { s u p } \left\{ q \mid \mu _ { q } < { \frac { 1 } { 2 K _ { q } - 1 } } \right\} ,\tag{3}
$$

where $\mu _ { q }$ and $K _ { q }$ denote the coherence and sparsity level at quantile $q _ { \mathrm { { i } } }$ , respectively (see ‘Stability metrics and computation’ in Methods). Ideal representational atoms achieve $R ^ { 2 } = 1$ and $q ^ { * } = 1$ , providing a quantitative reference for evaluating candidate FRUs in the following analyses.

## Limitations of existing representational units

Having defined fidelity and stability, we next use these criteria to evaluate two widely studied classes of candidate FRUs: neurons and learned features [7–10]. Neurons correspond directly to the native coordinates of internal representations, whereas learned features are obtained by decomposing these representations into latent components. For learned features, we use Gemma Scope [20] and Llama Scope [21], both trained on large-scale general-domain corpora (see ‘Evaluation of existing representational units’ in Methods). The results are shown in Fig. 2.

In terms of fidelity, neurons and learned features exhibit a clear contrast. Because internal representations are expressed directly in the original neuronal coordinate system, treating neurons as candidate FRUs preserves the representational information without introducing an additional approximation, making fidelity rarely a primary limitation. Learned features, by contrast, are obtained through learned decompositions of internal representations, and their reconstruction quality depends on model architecture and scale, training data and optimization settings. Pretrained learned-feature models such as Gemma Scope and Llama Scope may therefore not be fully optimized for the target representation distribution, potentially leading to information loss and lower fidelity than neurons, even when yielding more interpretable components [8–10].

Table 1 Evaluation of sparse autoencoder representations across datasets and models.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">Metric</td><td colspan="2">GemmaScope</td><td colspan="5">Trained SAEs</td></tr><tr><td>Small</td><td>Large</td><td>ReLU</td><td>TopK</td><td>AbsTopK</td><td>JumpReLU</td><td>BiJumpReLU</td></tr><tr><td rowspan="6">Wikipedia</td><td rowspan="2">Gemma2-2B</td><td>Fidelity ↑</td><td>72.17(±13.21)</td><td>75.52 (±13.12)</td><td>75.46 (±16.17)</td><td>88.68 (±7.05)</td><td>88.11 (±6.93)</td><td>94.91 (±2.38)</td><td>97.25 (±1.44)</td></tr><tr><td>Stability ↑</td><td>67.87 (±16.69)</td><td>67.43 (±18.43)</td><td>85.58 (±7.40)</td><td>39.34 (±1.01)</td><td>39.07 (±1.02)</td><td>98.41 (±1.57)</td><td>99.78 (±0.13)</td></tr><tr><td rowspan="2">Gemma2-9B</td><td>Fidelity ↑</td><td>65.56 (±17.21)</td><td>70.78 (±15.96)</td><td>74.25 (±22.06)</td><td>87.42 (±7.56)</td><td>87.65 (±7.05)</td><td>93.38 (±3.24)</td><td>96.75 (±1.86)</td></tr><tr><td>Stability ↑</td><td>91.93(±5.17)</td><td>89.00 (±8.03)</td><td>89.46 (±4.57)</td><td>49.25 (±1.81)</td><td>48.12 (±0.70)</td><td>97.92 (±1.43)</td><td>99.81 (±0.08)</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>Fidelity ↑</td><td></td><td></td><td>77.37(±7.32)</td><td>87.13 (±6.34)</td><td>86.43 (±6.33)</td><td>94.47(±4.52)</td><td>95.66 (±2.14)</td></tr><tr><td>Stability ↑</td><td></td><td></td><td>82.84(±6.25)</td><td>51.11 (±1.20)</td><td>50.73 (±1.03)</td><td>82.74(±14.20)</td><td>99.35 (±0.57)</td></tr><tr><td rowspan="6">Wikidata</td><td rowspan="2">Gemma2-2B</td><td>Fidelity ↑</td><td>72.44 (±6.51)</td><td>76.50 (±5.75)</td><td>87.42 (±5.58)</td><td>92.15 (±4.33)</td><td>92.03 (±3.86)</td><td>93.78 (±2.97)</td><td>95.21 (±3.96)</td></tr><tr><td>Stability ↑</td><td>70.15 (±16.62)</td><td>69.21 (±17.35)</td><td>84.77 (±6.95)</td><td>42.06 (±3.97)</td><td>40.95 (±2.38)</td><td>96.97 (±2.21)</td><td>98.32 (±2.00)</td></tr><tr><td rowspan="2">Gemma2-9B</td><td>Fidelity ↑</td><td>67.81 (±14.09)</td><td>73.76 (±12.01)</td><td>86.72 (±9.09)</td><td>90.57 (±6.68)</td><td>91.19 (±6.35)</td><td>92.52 (±4.49)</td><td>96.85 (±1.39)</td></tr><tr><td>Stability ↑</td><td>92.47(±9.33)</td><td>89.53 (±13.16)</td><td>86.18 (±9.03)</td><td>53.69 (±4.78)</td><td>49.90 (±1.41)</td><td>93.70 (±6.48)</td><td>99.11 (±1.02)</td></tr><tr><td rowspan="2">Olmo-Hybrid-7B</td><td>Fidelity ↑</td><td></td><td></td><td>80.04 (±12.38)</td><td>86.18 (±9.35)</td><td>86.60 (±8.52)</td><td>90.29 (±6.87)</td><td>96.61 (±1.70)</td></tr><tr><td>Stability↑</td><td></td><td></td><td>79.04 (±12.59)</td><td>46.80 (±8.05)</td><td>49.31 (±1.24)</td><td>90.04(±7.57)</td><td>98.56 (±1.62)</td></tr><tr><td rowspan="4">LAION-COCO</td><td rowspan="2">Stable Diffusion v1.5</td><td>Fidelity ↑</td><td></td><td></td><td>63.77 (±8.90)</td><td>81.04 (±18.14)</td><td>80.06 (±16.34)</td><td>75.22 (±22.51)</td><td>85.50 (±21.18)</td></tr><tr><td>Stability ↑</td><td></td><td></td><td>72.94 (±14.41)</td><td>29.71 (±0.49)</td><td>29.44(±0.40)</td><td>82.86 (±28.67)</td><td>98.52 (±4.14)</td></tr><tr><td rowspan="2">FLUX.2-klein-9B</td><td>Fidelity ↑</td><td></td><td></td><td>80.61 (±11.97)</td><td>88.33 (±8.43)</td><td>90.05 (±9.11)</td><td>87.43 (±7.18)</td><td>96.03 (±2.37)</td></tr><tr><td>Stability ↑</td><td></td><td></td><td>81.83 (±18.80)</td><td>41.56 (±14.95)</td><td>53.56 (±12.21)</td><td>64.12 (±10.20)</td><td>98.47 (±2.48)</td></tr></table>

Values are reported as mean followed by the half-width of the two-sided 95% t-confidence interval in parentheses, and all values are scaled by 100. Higher values indicate better performance. Bold indicates the highest mean value in each row. For Wikipedia and Wikidata, confidence intervals are computed across six evaluated model layers. For LAION-COCO, confidence intervals are computed across three difusion timesteps for Stable Difusion v1.5 and across six layer-step combinations for FLUX.2-klein-9B. GemmaScope Small denotes width 16k; GemmaScope Large denotes width 65k for Gemma2-2B and width 131k for Gemma2-9B. TopK and AbsTopK use K = 50, and additional TopK results with K = 20 for Gemma2-2B and Gemma2-9B are reported in Extended Data Table 2. Dashes indicate that the corresponding GemmaScope model is unavailable.

In terms of stability, neither neurons nor learned features fully satisfy the distinguishability required of ideal representational atoms. Neurons correspond directly to the coordinate directions of the internal representation space, but these directions need not be suficiently distinguishable from one another. Consequently, neurons generally exhibit low stability, consistent with the widespread polysemanticity of individual neurons [22]. Learned features reduce representational entanglement through learned decomposition and exhibit higher stability than neurons, consistent with their greater monosemanticity [9, 10]. However, their stability remains substantially below the ideal regime.

Overall, neurons and learned features exhibit distinct limitations: neurons preserve high fidelity but exhibit low stability, whereas learned features improve stability but exhibit reduced fidelity and remain below the ideal stability regime. Accordingly, neither fully satisfies the requirements of ideal representational atoms.

## Threshold-activated SAEs identify representational atoms

The limitations of existing candidate FRUs raise a central question: what kind of representation learning mechanism can reliably identify representational atoms with both high fidelity and stability?

Our theoretical analysis shows that, under appropriate conditions, threshold-activated sparse autoencoders (TSAEs) can identify FRUs with both high fidelity and stability (see ‘Identification of atoms’ in Methods; Fig. 1c). We instantiate TSAEs using BiJumpReLU SAEs, which utilize learnable activation thresholds to identify representational atoms (see ‘SAE architectures’ in Methods). We evaluate their efectiveness across multiple model families, network architectures, and representation modalities, spanning diverse data distributions. We compare BiJumpReLU SAEs with several established SAE architectures [8, 13–15] under the same evaluation framework. Full experimental settings are described in ‘Training details’ in Methods. Table 1 reports the main results for fidelity and stability, the two criteria used to evaluate representational atoms, with additional metrics provided in Extended Data Table 1.

Among the evaluated SAE architectures, BiJumpReLU SAEs consistently maintain high fidelity and stability across all settings, achieving the highest mean value for both metrics in all eight model–dataset combinations. TopK and AbsTopK SAEs retain relatively high fidelity but exhibit substantially lower stability, whereas ReLU SAEs achieve higher stability but generally lower fidelity. JumpReLU SAEs provide the strongest competing results in many settings, but their stability varies markedly across models, from 64.12 to 98.41. In contrast, BiJumpReLU SAEs maintain stability above 98 across all settings while further improving fidelity.

Together, these theoretical and empirical results show that TSAEs provide an efective approach for identifying representational atoms. BiJumpReLU SAEs serve as a concrete instantiation of this approach and perform consistently across diverse models, architectures, data distributions and representation modalities, providing a practical basis for identifying atoms in neural representations.

a  
![](images/0ea66a1348e7e45db6bb0bdcf7b1a8c2ac9da6f97bc17babaec7eadcbeac4277.jpg)

b  
![](images/db83c3cc5e6f70b1f9f67061cd1e6c33dda519eb19e2b230163fe94e40333452.jpg)

![](images/c1c92689ac33baacf91dd2eb4b120c0ac998c7b9fa747a025568fad2f42efc44.jpg)

d  
![](images/a0eb23e908b03fd15db1b7bf71dbcd42f622666a942ecb535904bfe6413d319a.jpg)

![](images/baebd8b9da0ca5b507381d1de5a1e34e388eafcc85e8d1b27f08b88e1e863548.jpg)

e  
![](images/8c9a7bbf23a6e5b72d0f29d31d4c9ed7115f56c385a23d82f6f57d2b7eb26e39.jpg)

![](images/e664f392f41d873ae9f535d0a50f6d1631553df54dfd0bd2cf9c8232004a5743.jpg)

![](images/1417328c8ea76637d5dedd9fd336ac8c273a30fa6152f2294e7001f50c678fad.jpg)

![](images/eb40a68ea5a0bec2197e79c1ffc5648ea50fc3c7029bb4f99a8a65c2bb3811e0.jpg)

h  
![](images/4f479a013085a84ffcccfce1231a70f34d5d8dab862e45720e0d058bd5a637f0.jpg)

![](images/223f2b1216326d2d0948d1a96a0ea58854cf148edb1492bef752d1fbb5d3e846.jpg)

![](images/f78e3eb047fadf25633e2f8eba58b6de856ae9f154e0f5bd40b0f64d75156e66.jpg)

l  
![](images/049283dff2d08fc85630b56ddcced4686d5dde8cdc28df6715f2abb80717634f.jpg)  
Fig. 3 Monosemanticity is positively correlated with stability across datasets and models. a, Overall rela tionship between stability and monosemanticity across all datasets, models and SAE architectures. Each point denotes one dataset-model-SAE combination $( N = 4 0 )$ . For language models, stability and monosemanticity are averaged across the six evaluated layers; for Stable Difusion v1.5, across three difusion timesteps; and for FLUX.2-klein-9B, across six layer-step combinations. The positive correlation indicates a consistent association between stability and monosemanticity across heterogeneous representation settings. b-d, Relationships within Wikipedia (b; $N = 1 5 )$ , Wikidata $( \mathbf { c } ; N = 1 5 )$ and $\mathrm { L A I O N  – C O C O } \left( \mathbf { d } ; N = 1 0 \right)$ , using the same model-SAE-level averages as in a. Positive correlations within all three datasets show that the overall association is preserved within individual datasets and modalities. e-l, Relationships within individual models: Gemma2-2B (e), Gemma2-9B (f) and Qwen3.5-9B (g) on Wikipedia; Gemma2-2B (h), Gemma2-9B (i) and Olmo-Hybrid-7B (j) on Wikidata; and Stable Difusion v1.5 (k) and FLUX.2-klein-9B (l) on LAION-COCO. For language models, each point denotes one SAE architecture at one layer; for Stable Difusion v1.5, one SAE architecture at one difu sion timestep; and for FLUX.2-klein-9B, one SAE architecture at one layer-step combination. Monosemanticity is averaged across the manually evaluated features at each corresponding layer, timestep or layer-step combination. These model-level results show that the positive correlation persists at finer representational granularity, although its strength varies across models. Colors denote SAE architectures (BiJumpReLU, JumpReLU, ReLU, TopK with $K = 5 0$ , and AbsTopK with $K = 5 0 )$ , and marker shapes denote datasets. Black lines show linear regression fits with 95% confidence bands. Spearman’s rank correlation coeficients (ρ) and two-sided P values are shown in each panel.

## Stable atoms are more monosemantic

Monosemanticity refers to a representational unit has a consistent semantic meaning, such that its activation reliably corresponds to a specific semantic pattern [8–10]. Such consistency is necessary for a well-defined interpretation of the unit, as its semantic attribution otherwise remains ambiguous. Monosemanticity therefore provides a basis for analyzing the semantic organization of internal representations.

Our theoretical analysis shows that stability characterizes the distinguishability required for monorepresentationality, which constitutes a necessary structural condition for monosemanticity (see ‘Evaluation of atoms’ in Methods). However, monorepresentationality is not suficient, because a structurally welldefined unit may still encode distinctions that do not align with human semantic categories. The relationship between stability and monosemanticity therefore requires empirical validation.

![](images/c0118bd8346570d7492282489ef535051072a73b59ccc565bc50a07d1e509749.jpg)

![](images/548038a1595b6b43633bc8b9e1ae87572be95808be20563799b8bed64665ad08.jpg)

c  
![](images/44ff093bb8bf11dbcff6e78642d32bccb1b25fde2f91daa06d319668983cc1ae.jpg)

d  
![](images/e4a9ed7858657736be96e7a050d593047bc96c31592a5c9e8e820c530d66cc30.jpg)  
Fig. 4 Human sememes are compositionally represented by multiple representational atoms. a, Sememe– atom structural map for Gemma2-2B. Large light-blue circles denote human sememes and smaller blue circles denote representational atoms selected for their compositional recovery. Solid orange circles indicate the best single atoms that are retained in the final composition, whereas dashed orange circles indicate the best single atoms that are not selected. Circle sizes reflect the sizes of the corresponding semantic sets, and the geometric overlap between an atom and its sememe reflects their single-atom precision. Representative examples show that near one-to-one sememe–atom correspondence is rare, whereas compositional recovery by multiple atoms is common. More complete sememe–atom structural maps for Gemma2-2B, Gemma2-9B, and Llama3.1-8B are shown in Extended Data Figs. 3–5, respectively. b, Distribution of the number of atoms selected for each sememe across Gemma2-2B, Gemma2-9B and Llama3.1-8B, grouped into 1, 2–3, 4–6 and > 6 atoms. Most sememes are recovered using more than six atoms. c, Semantic recovery of the best single atom and the corresponding atom combination across the three models. Each point denotes an individual sememe, and F1 measures the agreement between the recovered representational set and the corresponding HowNet sememe set. Atom combinations consistently achieve substantially higher semantic alignment than the best single atoms. d, Relationship between sememe complexity and the number of atoms selected for its compositional recovery across the three models. Sememe complexity is measured by the number of entities associated with each HowNet sememe. Each point denotes an individual sememe with merged $\mathrm { F } 1 \geq 0 . 5$ , and dashed lines show model-specific linear regression fits in log–log space. More complex sememes tend to require more atoms for recovery.

To test this relationship, we evaluate the monosemanticity of the identified FRUs and examine its association with stability (see ‘Monosemanticity evaluation details’ in Methods). Across all 40 dataset– model–SAE combinations, stability is positively correlated with monosemanticity (Spearman $\rho = 0 . 7 1$ $P < 0 . 0 0 1$ ; Fig. 3a). This association is preserved within each dataset, with correlations of $\rho = 0 . 7 4$ for Wikipedia, $\rho = 0 . 7 9$ for Wikidata and $\rho = 0 . 9 5$ for LAION-COCO (Fig. 3b–d). Across individual models and representation settings, all eight correlations remain positive, ranging from $\rho = 0 . 3 7$ to 0.93 $\mathrm { ( F i g . \ 3 e { - } l ) }$ . Architecture-stratified correlations and monosemanticity distributions across SAE architectures are reported in Extended Data Figs. 1 and 2. Together, these results show that greater stability is consistently associated with greater monosemanticity across model families, representation modalities and SAE architectures.

## Atoms compositionally encode sememe structures

Having shown that representational atoms identified by BiJumpReLU SAEs exhibit higher stability and greater monosemanticity, we next examine how these atoms relate to human conceptual structure. Monosemanticity captures the consistency of an FRU’s semantic interpretation, but does not determine whether that FRU alone captures a complete human semantic concept. We therefore investigate whether higher-order semantic structures are encoded by single atoms or by their compositions.

We use HowNet [16, 17], an expert-annotated knowledge base in which concepts are described by fine-grained semantic units called sememes. We extract internal representations of HowNet entities across layers from multiple models and identify corresponding representational atoms using BiJumpReLU SAEs (see ‘Representation extraction for cross-layer semantic alignment’ in Methods). These SAEs maintain high fidelity and stability across the evaluated models and layers (Extended Data Table 4). For each sememe, we identify the single atom with the highest set-matching F1 score across layers, where F1 summarizes the precision and recall of the atom–sememe match, and then search for the cross-layer atom combination that maximizes the same score (see ‘Semantic alignment evaluation details’ in Methods).

![](images/5236360245b81269f318f0ac2bd9d910ae36c67e4e50676b6823e16e1a61fa26.jpg)

![](images/42621de9511b039f91b785e0c610a0eb468f809802cec906e152587b000e4b57.jpg)

c  
![](images/992cacf3ad88ecf49234a47fd8dc79de02c7651eac9c02fc2d198c770503b7f5.jpg)  
Sememe-level merged F1 Linear regression fit (95% confidence band) Mean single-atom F1  
Fig. 5 Sememe-level merged F1 is largely insensitive to cross-layer width. a–c, Relationship between cross-layer width and sememe-level merged F1 for Gemma2-2B (a), Gemma2-9B (b) and Llama3.1-8B (c). Each blue point denotes the merged F1 obtained for an individual sememe at corresponding cross-layer width. Red lines show linear regression fits with 95% confidence bands, and dashed grey lines indicate the mean single-atom F1 baseline. Across all three models, the fitted slopes are negligible and Pearson correlations remain close to zero $( r = 0 . 0 2 5$ , 0.006 and −0.032, respectively). Equivalence tests further show that the absolute correlations fall within the prespecified bound of |r| < 0.10 (P < 0.001 in all cases), indicating that sememe-level performance is not systematically associated with cross-layer width.

As shown in Fig. 4, near one-to-one correspondences between sememes and atoms are rare. Although some sememes can be recovered almost completely by a single atom, with F1 reaching 1 (Fig. 4a), such cases account for only approximately 1% of all sememes (Fig. 4b). Most sememes require multiple atoms, with the majority involving more than six atoms across all three models (Fig. 4b). These atom combinations achieve substantially higher semantic alignment (F1) than the corresponding best single atoms (Fig. 4c), primarily through higher recall without sacrificing precision (Extended Data Table 3). The number of selected atoms also increases systematically with sememe complexity, measured by the number of associated entities (Fig. 4d). These results suggest that representational atoms generally function as finer-grained semantic components whose compositions encode higher-order human semantic structures.

To determine whether this compositional advantage could arise from the cross-layer search procedure, we examine the association between cross-layer width and final F1. The Pearson correlation between them is close to zero, and the equivalence test [23] confirms that it falls within the prespecified bound of $| r | < 0 . 1 0 \ ( P < 0 . 0 0 1$ ; Fig. 5), indicating no practically meaningful association. Final F1 therefore does not increase systematically with cross-layer width, supporting the interpretation that the advantage of atom combinations reflects compositional structure in the internal representations rather than the search procedure.

Case-study analysis further reveals discrepancies between the entity sets recovered by atom combinations and the corresponding HowNet annotations (Extended Data Table 5; see ‘Sememe evaluation details’ in Methods). Manual inspection identifies unannotated entities that are consistent with the target sememes, as well as annotated entities that appear inconsistent with them. These cases suggest that some apparent mismatches may reflect incomplete or inaccurate annotations. Together, these findings show that single atoms capture finer-grained semantic components, while their compositions recover higher-order semantic organization that closely corresponds to human conceptual structure.

## Discussion

This study focuses on a fundamental question concerning the basic units that constitute internal representations of neural networks and how their organization relates to human conceptual structure. We formulate this problem in terms of the definition, evaluation, and identification of fundamental representational units (FRUs). Drawing on compressed sensing [5, 6], we define FRUs as representational atoms and introduce fidelity and stability to characterize information preservation and structural distinguishability. We further show that threshold-activated sparse autoencoders (TSAEs) can identify such atoms under appropriate conditions, instantiate TSAEs as BiJumpReLU SAEs in practice, and validate their efectiveness across diverse models and representation settings. Further analysis indicates that more stable atoms also exhibit greater empirical monosemanticity, and their cross-layer compositions recover higher-order semantic organization corresponding to human conceptual structure. Together, these findings provide a unified theoretical and practical framework for characterizing the basic organization of neural representations and its relationship to human conceptual structure.

This framework provides a principled view of commonly studied representational units. Neurons and learned features can be evaluated within the same theoretical setting through fidelity and stability, rather than being treated solely as empirical units of interpretation. More broadly, the framework connects two influential perspectives on neural representation: the linear representation hypothesis [24], in which semantic information is encoded along specific directions, and the superposition hypothesis [11], which explains how finite-dimensional spaces can support substantially more linear components than their dimensionality. Their combination within a sparse linear framework provides a unified account of how representational components can remain both expressive and distinguishable.

The compositional organization of representational atoms also has implications for interventions on internal representations, including knowledge editing [25–27] and representation steering [28, 29]. Such methods often rely on mapping human-defined target concepts to localizable representational structures. Our results suggest that these correspondences are often mediated by finer-grained representational atoms and their combinations, which jointly form higher-order semantic structures. Intervening on a single direction or localized representation may therefore afect only part of a target concept or simultaneously perturb multiple overlapping semantic components. This perspective may help explain incomplete edits, collateral efects and instability, and suggests that future intervention methods should explicitly consider the compositional and superpositional organization of FRUs.

This representational perspective is also relevant to continual learning [30, 31]. Continual learning modifies internal representations through parameter updates, and superposition implies that changes to representations may perturb multiple overlapping components and thereby afect existing knowledge. This provides a possible structural account of catastrophic forgetting and suggests that robust continual learning may therefore benefit from explicitly considering the superpositional organization of FRUs. More broadly, this framework provides a basis for studying how models integrate new information while preserving previously encoded knowledge.

A natural question is whether this framework generalizes across diverse neural networks and internal representations. Its generality ultimately depends on whether stable and semantically meaningful representational atoms emerge across diferent models and representation spaces. Importantly, fidelity and stability provide quantitative criteria for assessing this applicability without assuming it a priori. Across the models and representation settings examined here, we observe consistent evidence for representational atoms, supporting their use as a testable and extensible basis for studying neural representations and their semantic organization.

## Methods

## Representational atoms theoretical framework

In neural networks, all information is embedded in high-dimensional representations. Our objective is to identify the fundamental representational units (FRUs), which we term representational atoms, or atoms for brevity. Formally, we consider a collection of representations ${ \cal M } = \{ { \pmb { m } } _ { i } \} _ { i = 1 } ^ { | M | }$ , where ${ m } _ { i } \in \mathbb { R } ^ { H }$ . Each representation can be expressed as $\begin{array} { r } { m _ { i } = \sum _ { i } \delta ( i , j ) d _ { j } } \end{array}$ , where $\delta ( i , j )$ denotes the presence and magnitude of the j-th representational unit $\pmb { d } _ { j } \in \mathbb { R } ^ { H }$ in the i-th representation $\mathbf { \nabla } m _ { i }$ . However, in representation space, the family of representational units admitting such a decomposition is, in principle, infinite.

The central question is how to define FRUs. A natural criterion is distinguishability: each atom should be detectable or manipulable without interfering with others. In high-dimensional spaces, this criterion translates into orthogonality: atoms occupy mutually orthogonal directions, making their identities distinguishable via inner products. This motivates introducing an inner product to analyze the geometry of FRUs.

Thus, the choice of inner product is critical. Although the Euclidean inner product is commonly used, it is not necessarily appropriate for characterizing the intrinsic geometry of neural representation spaces $[ 2 4 , \ 3 2 ]$ . To illustrate this point, let $\boldsymbol { h } \in \mathbb { R } ^ { H }$ denote the internal representation at a given layer and $f _ { \mathrm { d o w n } }$ the subsequent network mapping, so that the downstream computation is $\begin{array} { r } { \pmb { y } = f _ { \mathrm { d o w n } } ( \pmb { h } ) } \end{array}$ . Because our analysis concerns the linear structure and its geometry in the representation space, we consider reparameterizations that preserve this linear structure. In a finite-dimensional real vector space, such reparameterizations are exactly invertible linear transformations, so that any bijection that preserves linearity can be uniquely written as $h ^ { \prime } = A h$ , where $\pmb { A } \in \mathbb { R } ^ { H \times H }$ is invertible. Accordingly, for any invertible linear transformation A, we can introduce the following reparameterization:

$$
f _ { \mathrm { d o w n } } ^ { \prime } ( z )  f _ { \mathrm { d o w n } } ( A ^ { - 1 } z ) , \quad h ^ { \prime }  A h .\tag{4}
$$

Then it follows that $f _ { \mathrm { d o w n } } ^ { \prime } ( h ^ { \prime } ) = f _ { \mathrm { d o w n } } ( h )$ , leaving the input–output function unchanged. The results of Park et al. [24] and Hu et al. [32] can be viewed as specific instances of this reparameterization principle under particular model parameterizations.

Since the above reparameterization leaves the network input–output function unchanged, h and $h ^ { \prime } = A h$ are indistinguishable from the perspective of observable network behavior. Thus, even for a fully trained network with a fixed input–output function, internal representations are identifiable only up to the equivalence class induced by invertible linear transformations A, rather than by unique coordinate representations determined by the network itself. By linearity, this non-uniqueness extends to the corresponding representational units $\pmb { d } _ { j }$ . Under transformation $\pmb { d } _ { j } ^ { \prime } = \pmb { A } \pmb { d } _ { j }$ , the transformed units describe an equivalent representational structure associated with the same network function.

This has direct consequences for the geometry of the representation space. In particular, the Euclidean inner product is not invariant under the reparameterization in equation (4): $\langle d _ { i } , d _ { j } \rangle \neq \langle A d _ { i } , A d _ { j } \rangle$ Therefore, geometric relations induced by the Euclidean inner product, such as angles and orthogonality, between $\mathbf { \ b { d } } _ { i }$ and $d _ { j }$ depend on the chosen parameterization. Thus, the Euclidean inner product does not provide a canonical geometry of neural representations that is independent of the chosen parameterization.

Atomic inner product. To better understand neural representations and thereby define atoms within them, we require additional principles to specify an appropriate inner product. We therefore introduce an inner product with the desired properties.

Definition 1 (Atomic inner product; AIP) Let ${ \mathcal { D } } = \operatorname { s p a n } ( D )$ , where $D = \{ d _ { j } \} _ { j = 1 } ^ { | D | }$ denotes the atom set. The atomic inner product $\langle \cdot , \cdot \rangle _ { S }$ is an inner product on D such that $\langle d _ { i } , d _ { j } \rangle _ { S } = 0$ for all $\mathbf { \boldsymbol { d } } _ { i } , \mathbf { \boldsymbol { d } } _ { j } \in D$ with $i \neq j$

Atoms are indexed arbitrarily, and any permutation of indices leaves their geometry invariant. Hence, there is no principled basis for assigning diferent norms to diferent atoms. By this symmetry, we assume a common norm under the chosen inner product, i.e. $\lVert \pmb { d } _ { i } \rVert _ { S } = c > 0 , \forall i \in [ | D | ]$ . The constant c cancels naturally in subsequent analysis.

By abuse of notation, we also use $D \in \mathbb { R } ^ { H \times | D | }$ to denote the matrix with columns $d _ { j }$ . We next characterize the atomic inner product in an explicit form.

Theorem 2 (Explicit form of the atomic inner product) Let $\langle { d } _ { i } , { d } _ { j } \rangle _ { S } = { d } _ { i } ^ { \top } { S } { d } _ { j }$ be an atomic inner product with $\pmb { S } \in \mathbb { R } ^ { H \times H }$ symmetric and positive definite. If the columns of $D = [ d _ { 1 } , d _ { 2 } , \cdot \cdot \cdot , d _ { | D | } ]$ form the atom set such that $\forall i , \ \lVert d _ { i } \rVert _ { S } = c > 0$ , and $\mathcal { D } \simeq \mathbb { R } ^ { H }$ , then $S = c ^ { 2 } ( D D ^ { \top } ) ^ { - 1 }$

All proofs are provided in the ‘Proofs’ section of the Supplementary Information. By analogy to cosine similarity, we introduce the normalized atomic inner product to remove the dependence on c.

Corollary 3 (Normalized atomic inner product; NAIP) Let the atomic inner product be defined by $\langle d _ { i } , d _ { j } \rangle _ { S } =$ $d _ { i } ^ { \top } S d _ { j } ,$ where S is symmetric and positive definite. Suppose the columns of $D = [ d _ { 1 } , \cdots , d _ { | D | } ]$ form the atom set satisfying ∀i, $\lVert d _ { i } \rVert _ { S } = c > 0$ . Then, for any i, j,

$$
\langle \boldsymbol { d } _ { i } , \boldsymbol { d } _ { j } \rangle _ { \tilde { S } } : = \frac { \langle \boldsymbol { d } _ { i } , \boldsymbol { d } _ { j } \rangle _ { \boldsymbol { S } } } { \| \boldsymbol { d } _ { i } \| _ { \boldsymbol { S } } \| \boldsymbol { d } _ { j } \| _ { \boldsymbol { S } } } = \boldsymbol { d } _ { i } ^ { \top } \tilde { \boldsymbol { S } } \boldsymbol { d } _ { j } , \quad \tilde { \boldsymbol { S } } = ( \boldsymbol { D } \boldsymbol { D } ^ { \top } ) ^ { - 1 } .\tag{5}
$$

Consequently, the bilinear form $\langle \pmb { d } _ { i } , \pmb { d } _ { j } \rangle _ { \tilde { S } } = \pmb { d } _ { i } ^ { \top } \tilde { \pmb { S } } \pmb { d } _ { j }$ defines the normalized atomic inner product.

Unlike the causal inner product [24], which is defined in the static and input-independent unembedding space over the output tokens, our AIP (or NAIP) operates in the dynamic and input-dependent representation space, characterizing the geometry of representations and their constituent units. This enables direct analysis of internal representations and thereby allows the definition of atoms as fundamental representational units under the AIP.

Remark Define $\tilde { d } _ { i } = \tilde { S } ^ { \frac { 1 } { 2 } } d _ { i }$ and $\tilde { d } _ { j } = \tilde { S } ^ { \frac { 1 } { 2 } } d _ { j }$ . Under this transformation, $\langle d _ { i } , d _ { j } \rangle _ { \tilde { S } } = \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle$ , where the right-hand side denotes the Euclidean inner product. Hence, properties of the Euclidean inner product are transferred directly to the NAIP. Consequently, $\tilde { d } _ { i }$ and $\tilde { d } _ { j }$ are termed normalized atoms.

Formal definition of atoms. Having established a principled perspective for understanding neural representations, we proceed to formally define atoms. However, the preceding analysis assumes an idealized setting in which atoms are strictly orthogonal under the chosen inner product. Although this assumption yields a clean notion, it constrains the number of atoms to at most the representation dimension $( | D | = H )$ , rendering the formulation impractical.

Elhage et al. [11] observed that sparsity induces the emergence of approximately orthogonal representations in neural networks to cope with limited representational dimensionality, a phenomenon termed superposition. Hu et al. [32] later identified pervasive superposition in neural networks. Motivated by these findings, we introduce sparsity, which naturally leads to approximate orthogonality and enables a formally grounded, practical definition of atoms.

Definition 4 (Sparsity level) Let $M = \{ m _ { i } \} _ { i = 1 } ^ { | M | } \subset \mathbb { R } ^ { H }$ be a collection of representations. Suppose there exist $\pmb { D } = [ \pmb { d } _ { 1 } , \cdot \cdot \cdot , \pmb { d } _ { | D | } ] \in \mathbb { R } ^ { H \times | D | }$ and $\delta _ { i } \in \mathbb { R } _ { \geq 0 } ^ { | D | }$ such that $\pmb { m } _ { i } = \pmb { D } \pmb { \delta } _ { i } , \forall i \in [ | \boldsymbol { M } | ]$ . The sparsity level is $K : = \operatorname* { m a x } _ { i } \| \pmb { \delta } _ { i } \| _ { 0 }$ where $\| \cdot \| _ { 0 }$ denotes the $\ell _ { 0 }$ norm.

Sparsity enables the number of atoms to substantially exceed the ambient dimension, yielding an overcomplete structure with $| D | \gg H$ that captures rich world knowledge, while simultaneously minimizing mutual interference.

Remark In the basic setting described in the ‘Atomic inner product’ subsection, with $| D | = H ,$ , one can verify that $\tilde { \pmb { S } } = ( \pmb { D } \pmb { D } ^ { \top } ) ^ { - 1 }$ satisfies $D ^ { \top } \tilde { S } D = I _ { | D | }$ , so the atoms form an orthonormal basis under $\langle \cdot , \cdot \rangle _ { S }$ . When $| D | \gg H , \ \tilde { S } = ( D D ^ { \top } ) ^ { - 1 }$ is well defined provided rank $( D ) = H$ , but $\boldsymbol { G } : = \boldsymbol { D } ^ { \top } \tilde { \boldsymbol { S } } \boldsymbol { D }$ becomes a rank-H projection rather than ${ \cal I } _ { | D | }$ , thus the atoms cannot all be mutually orthogonal under $\langle \cdot , \cdot \rangle _ { S }$ . Therefore, exact orthogonality is unattainable, motivating the introduction of approximate orthogonality.

This consideration motivates the following definition.

Definition 5 (ϵ-Approximately Orthogonal Atoms) The atom set $\{ d _ { i } \} _ { i = 1 } ^ { \left| D \right| }$ is ϵ-approximately orthogonal if $\left| \langle d _ { i } , d _ { j } \rangle _ { \tilde { S } } \right| \leq \epsilon , \forall i \neq j .$ , where $\langle \pmb { x } , \pmb { y } \rangle _ { \tilde { S } } : = \pmb { x } ^ { \top } \tilde { S } \pmb { y }$ denotes the normalized atomic inner product and $\tilde { \pmb { S } } : = ( \pmb { D } \pmb { D } ^ { \top } ) ^ { - 1 }$

Remark In the ideal setting of exact orthogonality, the inner products $\langle d _ { i } , d _ { j } \rangle _ { \tilde { S } }$ (equivalently, $\langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle )$ for $i \neq j$ follow a Dirac measure concentrated at the origin. Under ϵ-approximate orthogonality, they are instead expected to follow a Gaussian distribution ${ \mathcal { N } } ( 0 , s ^ { 2 } )$ with small variance, which converges to the Dirac measure as $s \to 0$

We now introduce a formal definition of atoms.

Definition 6 (Atoms) Let $\textbf { \textit { M } } = \{ \pmb { m } _ { i } \} _ { i = 1 } ^ { | M | }$ be a collection of representations. Suppose there exists $\textbf { = }$ $[ { \pmb d } _ { 1 } , \cdot \cdot \cdot , { \pmb d } _ { | D | } ] \in \mathbb { R } ^ { H \times | D | }$ and $\pmb { \Delta } = [ \pmb { \delta } _ { 1 } , \cdot \cdot \cdot , \pmb { \delta } _ { | M | } ] \in \mathbb { R } _ { \ge 0 } ^ { | D | \times | M | }$ such that, for a given sparsity level $K \in \mathbb { N } ^ { + }$

$$
\forall i \in [ | M | ] , \quad m _ { i } = D \delta _ { i } \mathrm { w i t h } \| \delta _ { i } \| _ { 0 } \leq K .\tag{6}
$$

Furthermore, $\lvert \langle d _ { i } , d _ { j } \rangle _ { \tilde { S } } \rvert \le \epsilon , \forall i \ne j$ , where $\tilde { \pmb S } : = ( \pmb { D } \pmb { D } ^ { \top } ) ^ { - 1 }$ . Under these conditions, $\{ d _ { i } \} _ { i = 1 } ^ { | D | }$ is called the atom set of M, and each $\mathbf { \ b { d } } _ { i }$ is referred to as an atom.

Intuitively, atoms are characterized by three properties: representability, where each representation can be faithfully expressed by atoms, $\mathrm { i . e . , } m _ { i } = D \delta _ { i } ;$ sparsity, where each representation involves only a few atoms, $\mathrm { i . e . , } \lVert \delta _ { i } \rVert _ { 0 } \leq K$ ; and separability, where atoms are approximately orthogonal, i.e., $| \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | \leq \epsilon$ Representability is a natural requirement, while sparsity and separability jointly enable eficient encoding of separable information under approximate orthogonality. We further quantitatively characterize the relationship between sparsity and separability in the following ‘Evaluation of atoms’ subsection.

Remark K quantifies sparsity without enforcing a specific sparsity regime, ensuring broad applicability of the definition. Moreover, pre-multiplying both sides of equation (6) by $\tilde { \mathbf { S } } ^ { \frac { 1 } { 2 } }$ yields $\tilde { m } _ { i } = \tilde { D } \delta _ { i }$ , with $\tilde { m } _ { i } : = \tilde { S } ^ { \frac { 1 } { 2 } } m _ { i }$ and $\tilde { \pmb { D } } : = \tilde { \pmb { S } } ^ { \frac { 1 } { 2 } } \pmb { D } = [ \tilde { \pmb { d } } _ { 1 } , \cdots , \tilde { \pmb { d } } _ { | D | } ]$ , which simplifies subsequent derivations.

Evaluation of atoms. Having formalized atoms via representability, sparsity, and separability, we now evaluate whether candidate representational units satisfy the criteria of ideal atoms. Representability is measured by the coeficient of determination $R ^ { 2 } : = 1 - \frac { \sum _ { i } \| \pmb { x } _ { i } - \hat { \pmb { x } } _ { i } \| ^ { 2 } } { \sum _ { i } \| \pmb { x } _ { i } - \bar { \pmb { x } } \| ^ { 2 } }$ , which quantifies the proportion of variance in the original representations explained by atoms and thus reflects fidelity. While sparsity and separability can be quantified by K and $\epsilon ,$ they do not directly indicate unit quality. In this section, we introduce a unified metric that integrates sparsity and separability to characterize the stability of representational units.

We note that $\pmb { \delta } _ { i } \in \mathbb { R } ^ { | D | }$ can be viewed as a sparse representation of high-dimensional semantics, which is compressed by $\pmb { D } \in \mathbb { R } ^ { H \times | D | }$ to produce $\boldsymbol { m } _ { i } \in \mathbb { R } ^ { H }$ . This perspective reveals a close connection to compressed sensing [5, 6], whose core idea is that a signal sparse in some basis can be recovered from far fewer linear measurements than its ambient dimension, with the Restricted Isometry Property (RIP) [33, 34] providing the essential guarantee.

Definition 7 (Restricted Isometry Property; RIP) A matrix $\tilde { D } \in \mathbb { R } ^ { H \times | D | }$ is said to satisfy the K-RIP if there exists a constant $\delta _ { K } \in [ 0 , 1 )$ such that, for any K-sparse vector ${ \pmb \delta } \in \mathbb { R } ^ { | D | } ~ ( \mathrm { i . e . , ~ } \| \pmb \delta \| _ { 0 } \le K )$ ，

$$
( 1 - \delta _ { K } ) \| \pmb { \delta } \| _ { 2 } ^ { 2 } \leq \| \tilde { \pmb { D } } \pmb { \delta } \| _ { 2 } ^ { 2 } \leq ( 1 + \delta _ { K } ) \| \pmb { \delta } \| _ { 2 } ^ { 2 } .\tag{7}
$$

Here, $\delta _ { K }$ is called the K-RIP constant of $\tilde { D }$

Intuitively, projecting a sparse vector via D<sup>˜</sup> to lower dimension preserves its geometric structure, ensuring the possibility of recovery. Direct verification of the RIP is NP-hard, but coherence [35, 36] provides a computable upper bound on $\delta _ { k }$

Theorem 8 (Coherence–RIP Upper Bound) Let $\tilde { D } \in \mathbb { R } ^ { H \times | D | }$ and define the coherence $\begin{array} { r } { \mu { : = } \operatorname* { m a x } _ { i \neq j } | \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | \leq \varepsilon . } \end{array}$ For any K-sparse vector $\delta \in \mathbb { R } ^ { | D | }$ with $\| \delta \| _ { 0 } \le K$

$$
\left( 1 - ( K - 1 ) \mu \right) \| \pmb { \delta } \| _ { 2 } ^ { 2 } \leq \| \tilde { \pmb { D } } \pmb { \delta } \| _ { 2 } ^ { 2 } \leq \left( 1 + ( K - 1 ) \mu \right) \| \pmb { \delta } \| _ { 2 } ^ { 2 } .\tag{8}
$$

Hence $\delta _ { K } ( \tilde { D } ) \leq ( K - 1 ) \mu$ . In particular, D<sup>˜</sup> satisfies the $K { \cdot } R I P$ whenever $( K - 1 ) \mu < 1$

In other words, coherence provides a computable criterion for verifying the RIP, ensuring that all $K -$ sparse vectors projected through $\tilde { D }$ preserve geometric structure, an essential prerequisite in compressed sensing. Nevertheless, the RIP alone does not preclude non-uniqueness [37, 38]: even if $( K - 1 ) \mu < 1$ holds, the sparse coeficients associated with representations need not be unique.

Theorem 9 (Uniqueness and Exact $\ell _ { 1 }$ Recoverability) Let $\tilde { D } \ \in \ \mathbb { R } ^ { H \times | D | }$ and define the coherence $\mu : =$ $\begin{array} { r } { \operatorname* { m a x } _ { i \neq j } | \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | \leq \varepsilon . \ I f \mu < \frac { 1 } { 2 K - 1 } } \end{array}$ , then for every $\boldsymbol \delta \in \mathbb { R } ^ { | D | }$ with $\| \delta \| _ { 0 } \le K$ , the K-sparse representation determined by $\tilde { \ b { m } } = \tilde { \ b { D } } \delta$ is unique; that is, no other K-sparse vector yields the same m˜ . Moreover, δ is the unique minimizer of the convex program

$$
\operatorname* { m i n } _ { \substack { \mathbf { x } \in \mathbb { R } ^ { | D | } } } \| \pmb { x } \| _ { 1 } \quad s u b j e c t \ t o \quad \tilde { D } \mathbf { x } = \tilde { m } .\tag{9}
$$

Crucially, this intrinsically characterizes the monorepresentationality of ${ \tilde { D } } \colon$ under the condition $\mu <$ $\frac { 1 } { 2 K - 1 }$ , any representation formed as a K-sparse linear combination of representational units in $\tilde { D }$ has a unique sparse decomposition, with no other combination yielding the same representation.

Corollary 10 (Monorepresentationality / Injectivity) Under the condition $\mu < \frac { 1 } { 2 K - 1 }$ of Theorem $^ { g } ,$ define $\Sigma _ { K } : = \{ \pmb \delta \in \mathbb { R } ^ { | D | } : \| \pmb \delta \| _ { 0 } \leq K \}$ and $\Phi : \Sigma _ { K }  \mathbb { R } ^ { H } \ b y \ \Phi ( \delta ) : = \tilde { D } \delta$ . Then Φ is injective on $\Sigma _ { K }$ . That $i s ,$ for any $\pmb { x } , \pmb { y } \in \Sigma _ { K } , \ i f \ \tilde { D } \pmb { x } = \tilde { D } \pmb { y }$ , it follows that $\mathbf { \boldsymbol { x } } = \mathbf { \boldsymbol { y } } .$

Within this regime, representational units and their combinations are unambiguous in representation space, i.e., monorepresentationality. By contrast, monosemanticity [9, 10] concerns the alignment of a unit with a specific meaning, concept, or function. The former is formally provable, whereas the latter is statistical and interpretive. Monorepresentationality is a prerequisite for monosemanticity, as it provides the structural stability; otherwise, units would be non-unique and interpretation unstable. Without additional semantic anchoring or inductive assumptions, theoretical guarantees primarily concern structural stability induced by monorepresentationality, while monosemanticity must be determined empirically via further semantic alignment experiments.

Sparsity (K) and separability $( \mu )$ are thus unified by the condition $\textstyle \mu < { \frac { 1 } { 2 K - 1 } }$ , under which stability holds. More practical metrics are discussed in the following ‘Stability metrics and computation’ subsection.

Remark For intuition, the above result can be viewed as a generalization of the strictly orthogonal case $( | D | = H )$ In this setting, $\tilde { { \cal D } } \in \mathbb { R } ^ { H \times H }$ has orthogonal columns $( \mu = 0 ) , \mathrm { i . e . , } \tilde { D } ^ { \stackrel { ! } { \cal D } } = I .$ . Any representation m˜ $\in \mathbb { R } ^ { H }$ then admits a unique decomposition $\pmb { \delta } = \tilde { \pmb { D } } ^ { \top } \tilde { \pmb { m } } .$ , satisfying $\tilde { \ b { m } } = \tilde { \ b { D } } \delta$ . Here the sparsity leve $K = H = | D |$ satisfies $\mu \ < \ \frac { 1 } { 2 K - 1 }$ . Thus, under strict orthogonality, representations and atom coeficients are in one-to-one correspondence, yielding a direct and unique identification of atoms.

Identification of atoms. We have defined atoms and established criteria for ideal atoms. A central question remains whether such atoms can be identified in practice. Since sparse autoencoders (SAEs) are a standard approach for learning disentangled representations [8], we next demonstrate that, under appropriate conditions, SAEs can indeed identify these atoms, rendering the theory practically applicable.

Theorem 11 (Identifiability of Threshold-activated SAEs; TSAEs) Let $M = \{ m _ { i } \} _ { i = 1 } ^ { | M | } \subset \mathbb { R } ^ { H }$ with $m _ { i } = D \delta _ { i } $ where $\boldsymbol { D } = [ \boldsymbol { d } _ { 1 } , \therefore , \boldsymbol { d } _ { | D | } ] \in \mathbb { R } ^ { H \times | D | }$ satisfies $| \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | ~ \leq ~ \epsilon , ~ \forall i ~ \neq ~ j$ , and each $\pmb { \delta } _ { i } ~ \in ~ \mathbb { R } ^ { | D | }$ is K-sparse, i.e. $\lVert \pmb { \delta } _ { i } \rVert _ { 0 } \leq K$ . Define the positive and negative supports as $S _ { i } ^ { + } : = \{ j : \delta _ { i j } > 0 \} , \ : \ : S _ { i } ^ { - } : = \{ j : \delta _ { i j } < 0 \}$ , with $S _ { i } = S _ { i } ^ { + } \cup S _ { i } ^ { - } = \mathrm { s u p p } ( \delta _ { i } )$ . Suppose there exist constants $0 < \delta _ { \mathrm { m i n } } ^ { + } \ \le \ \delta _ { \mathrm { m a x } } ^ { + } , \ 0 \ < \ \delta _ { \mathrm { m i n } } ^ { - } \ \le \ \delta _ { \mathrm { m a x } } ^ { - } ,$ such that $\delta _ { \mathrm { m i n } } ^ { + } \leq \delta _ { i j } \leq \delta _ { \mathrm { m a x } } ^ { + } , \forall j \in S _ { i } ^ { + }$ , and $- \delta _ { \mathrm { m a x } } ^ { - } \leq \delta _ { i j } \leq - \delta _ { \mathrm { m i n } } ^ { - } , \forall j \in S _ { i } ^ { - }$ . Consider the bidirectional threshold activation function

$$
\sigma _ { \tau _ { - } , \tau _ { + } } ( x ) = \left\{ \begin{array} { l l } { { 0 , } } & { { \tau _ { - } \leq x \leq \tau _ { + } , } } \\ { { x , } } & { { x < \tau _ { - } \ o r \ x > \tau _ { + } , } } \end{array} \right.\tag{10}
$$

with $\tau _ { - } < 0 < \tau _ { + }$ . Define $B _ { 0 } : = \epsilon \left( | S _ { i } ^ { + } | \delta _ { \operatorname* { m a x } } ^ { + } + | S _ { i } ^ { - } | \delta _ { \operatorname* { m a x } } ^ { - } \right) , B _ { + } : = \epsilon \left( ( | S _ { i } ^ { + } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { + } + | S _ { i } ^ { - } | \delta _ { \operatorname* { m a x } } ^ { - } \right)$ , and $B _ { - } : =$ $\epsilon \left( | S _ { i } ^ { + } | \delta _ { \operatorname* { m a x } } ^ { + } + ( | S _ { i } ^ { - } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { - } \right)$ . If the bidirectional thresholds s $a t i s f y ~ - \delta _ { \mathrm { m i n } } ^ { - } + B _ { - } ~ < ~ \tau _ { - } ~ \leq ~ - B _ { 0 } , ~ B _ { 0 } ~ \leq ~ \tau _ { + } ~ <$ $\delta _ { \mathrm { m i n } } ^ { + } - B _ { + }$ , with these intervals being non-empty whenever $\delta _ { \mathrm { m i n } } ^ { - } > B _ { 0 } + B _ { - } , ~ \delta _ { \mathrm { m i n } } ^ { + } > B _ { 0 } + B _ { + }$ , then setting the TSAE with $W _ { d e c } = D$ and $\boldsymbol { W _ { \mathrm { e n c } } } = \boldsymbol { D } ^ { \top } \boldsymbol { \tilde { S } }$ yields

$$
\forall i , \quad W _ { \mathrm { d e c } } \sigma _ { \tau _ { - } , \tau _ { + } } \left( W _ { \mathrm { e n c } } m _ { i } \right) = m _ { i } .\tag{11}
$$

This parameterization exactly recover the atom set $D .$

Therefore, TSAEs can identify atoms in principle. By contrast, ReLU [8], lacking a threshold term, fails to satisfy the support-separation condition and is thus theoretically insuficient. This responds to O’Neill et al. [39]: the limitation of SAEs arises not from their linear–nonlinear mechanism, but from the absence of threshold activation, which prevents efective atom identification.

The threshold activation considered here denotes a class of activation mechanisms rather than a specific instantiation. AbsTopK [14] is formally related to this class, but relies on a prespecified K, which limits adaptivity and practical applicability. JumpReLU [15] and TopK [13] can also be viewed as special cases of threshold-based activation, but their one-sided activation implicitly fixes a positive orientation for each representational direction. In a general representation space, such an orientation is arbitrary, and one-sided activation may therefore introduce unnecessary dependence on the sign convention of the representational direction, whereas bidirectional threshold activation avoids this limitation.

The motivations underlying these methods are also fundamentally diferent. We introduce threshold activation to enable efective atom identification under approximate orthogonality. By contrast, JumpReLU [15] was originally proposed primarily to alleviate feature shrinkage [40], whereas TopK was designed to control sparsity directly by retaining a fixed number of activations [13]. Thus, although these methods are formally related, their original objectives difer from the atom identifiability considered here.

## Training details

To assess the applicability of the theoretical framework across model architectures and representation types, we conduct experiments on neural networks spanning multiple parameter scales, architectures and generative modalities. In the following section, we describe the extraction of internal representations and the implementation and optimization settings used to train the corresponding SAE variants.

Models and representation extraction. To cover diverse model architectures and generative paradigms, we consider Transformer-based language models (Gemma-2-2B, Gemma-2-9B [41], Llama-3.1-8B [42] and Qwen-3.5-9B [43]), the hybrid-architecture language model Olmo-Hybrid-7B [44], and difusion models (Stable Difusion v1.5 [45] and FLUX.2-klein-9B [46]). We further cover multiple input and representation types using Wikipedia [47] for continuous-text representations, Wikidata [48] for discrete-entity representations, LAION-COCO [49] for representations arising during image generation, and OpenHowNet [17] entity–sememe annotations for analyzing alignment between neural representational structure and human conceptual structure.

Representation extraction for SAE benchmarking. To compare BiJumpReLU SAEs with alternative SAE architectures, we select representative layers or inference stages at approximately uniform intervals and train each SAE independently on the corresponding internal representations. For language models, we follow common settings in prior representation-analysis studies [20, 21] and sample layers uniformly over model depth. Specifically, for Gemma-2-2B (26 layers), we use $l \in \{ 0 , 5 , 1 0 , 1 5 , 2 0 , 2 5 \}$ and for Gemma-2-9B (42 layers), $l ~ \in ~ \{ 0 , 8 , 1 6 , 2 4 , 3 2 , 4 0 \}$ , extracting the output at the location ‘post feedforward layernorm‘. For Qwen-3.5-9B and Olmo-Hybrid-7B (both 32 layers), we use $l \in$ {0, 6, 12, 18, 24, 30} and extract the output at ‘down proj‘. For difusion models, we follow common settings from prior representation-analysis work [50]. For Stable Difusion v1.5, we extract the output of ‘unet.mid block.attentions‘ at normalized difusion times $t \in \{ 0 . 0 , 0 . 5 , 1 . 0 \}$ . For FLUX.2-klein-9B, we extract the raw image-token block output from the global ‘transformer blocks‘ at layers $l \in \{ 5 , 1 5 , 3 0 \}$ and inference steps $s \in \{ 0 , 3 \}$ . All internal representations are extracted directly from the original forward passes of the models without modifying the underlying computation and are used as input for subsequent SAE training.

Following prior empirical guidance on matching SAE capacity to representation scale [51], we choose dataset sizes to maintain this balance while remaining compatible with the largest SAE configurations that can be trained reliably on a single A100 80-GB GPU. For continuous-text representations, we use Wikipedia and extract 300K internal representations at each of six representative layers for Gemma-2-2B, Gemma-2-9B, and Qwen3.5-9B, yielding $6 \times 3 0 0 \mathrm { K } = 1 . 8 \mathrm { M }$ representations per model. For discreteentity representations, we use Wikidata and extract 1.8M representations per model for Gemma-2-2B, Gemma-2-9B and Olmo-Hybrid-7B under the same setting. For image-generation representations, we use LAION-COCO. Stable Difusion v1.5 yields 3 × (8 × 8) × 10K = 1.92M representations in three difusion stages and $8 \times 8$ spatial positions. FLUX.2-klein-9B yields $2 \times 3 \times ( 8 \times 8 ) \times 1 0 \mathrm { K } = 3 . 8 4 \mathrm { M }$ representations across two inference stages, three representative layers, and 8 × 8 spatial positions.

Representation extraction for cross-layer semantic alignment. To characterize the alignment between the neural representational structure and human conceptual structure, we extend representation extraction to all model layers and obtain internal representations for HowNet entities at each layer, enabling subsequent cross-layer atom identification and compositional analysis. We consider Gemma-2- 2B, Gemma-2-9B, and Llama-3.1-8B. For Gemma-2-2B (26 layers) and Gemma-2-9B (42 layers), internal representations are extracted from the output at ‘post feedforward layernorm‘ in every layer $l \in [ 0 , 2 6 )$ and $l \in [ 0 , 4 2 )$ , respectively. For Llama-3.1-8B (32 layers), internal representations are extracted from the output at ‘down proj‘ in every layer $l \in [ 0 , 3 2 )$ . BiJumpReLU SAEs are then trained and applied at each layer to obtain representational atoms for cross-layer composition analysis. HowNet provides approximately 123K unique English entity entries for semantic alignment. Extracting representations across all layers yields approximately 26 × 123K ≈ 3.20M internal representations for Gemma-2-2B, $4 2 \times 1 2 3 \mathrm { K } \approx 5 . 1 7 \mathrm { M }$ for Gemma-2-9B and 32 × 123K ≈ 3.94M for Llama-3.1-8B.

SAE architectures. An SAE can generally be written as

$$
\boldsymbol { f } : \boldsymbol { x } \mapsto \hat { \boldsymbol { x } } = \boldsymbol { W } _ { \mathrm { d e c } } \sigma ( \boldsymbol { W } _ { \mathrm { e n c } } \pmb { x } ) ,\tag{12}
$$

where $\pmb { x } \in \mathbb { R } ^ { H } , \pmb { W } _ { \mathrm { e n c } } ^ { \top } , \pmb { W } _ { \mathrm { d e c } } \in \mathbb { R } ^ { H \times D }$ , D denotes the SAE dictionary width and $\sigma$ is the activation function applied to the latent representation. Diferent SAE architectures are primarily distinguished by

the form of $\sigma .$ Cunningham et al. [40] introduced the ReLU SAEs, for which

$$
\mathrm { R e L U : } \quad \sigma _ { i } ( z ) = \operatorname* { m a x } ( 0 , z _ { i } ) .\tag{13}
$$

Gao et al. [13] proposed TopK SAEs, which retain only the K largest components of z, i.e.

$$
\mathrm { T o p K : } \quad \sigma _ { i } ( z ) = z _ { i } { \bf 1 } [ i \in \mathrm { T o p K } ( z , K ) ] ,\tag{14}
$$

where $\mathrm { T o p K } ( z , K )$ denotes the index set of the K largest components of z. Rajamanoharan et al. [15] proposed the JumpReLU SAE, which introduces learnable thresholds $\theta _ { i }$ and retains only positive activations above the corresponding threshold

$$
\mathrm { J u m p R e L U : } \quad \sigma _ { i } ( z ) = z _ { i } \mathbf { 1 } [ z _ { i } > \theta _ { i } ] .\tag{15}
$$

Zhu et al. [14] proposed the AbsTopK SAE, which selects the K components with the largest activation magnitudes while preserving their original signs, i.e.

$$
\mathrm { A b s T o p K } : \quad \sigma _ { i } ( z ) = z _ { i } \mathbf { 1 } [ i \in \mathrm { T o p K } ( | z | , K ) ] .\tag{16}
$$

For the BiJumpReLU SAE introduced here, we construct a bidirectional threshold activation according to the theoretical conditions for representational atom identification:

$$
\mathrm { B i J u m p R e L U : } \quad \sigma _ { i } ( z ) = z _ { i } { \bf 1 } [ z _ { i } < \tau _ { i } ^ { - } \lor z _ { i } > \tau _ { i } ^ { + } ] .\tag{17}
$$

This form follows from both the sign symmetry of representational directions and the theoretical requirements for atom identifiability. A representational direction and its negation have equivalent geometric status in the internal representation space and therefore should not be assigned an a priori preferred orientation. BiJumpReLU accordingly applies learnable thresholds to both positive and negative activations, allowing latent components to be selected without introducing a sign-dependent bias and aligning the architecture with the theoretical framework for representational atom identification.

Loss function. SAEs are trained by minimizing a joint reconstruction–sparsity objective (see Theorem 9 in ‘Evaluation of atoms’ subsection for the theoretical motivation):

$$
\begin{array} { r } { \mathcal { L } ( \pmb { x } ) = \underbrace { | | \pmb { x } - \hat { \pmb { x } } | | _ { 2 } ^ { 2 } } _ { \mathcal { L } _ { \mathrm { r e c o n s t r u c t } } } + \lambda \underbrace { | | \sigma ( z ) | | _ { 1 } } _ { \mathcal { L } _ { \mathrm { s p a r s i t y } } } , } \end{array}\tag{18}
$$

where $z = W _ { \mathrm { e n c } } x .$ , σ denotes the activation function of the corresponding SAEs, and λ controls the trade-of between reconstruction and sparsity. ReLU SAEs use this objective directly. Because JumpReLU and BiJumpReLU SAEs contain learnable thresholds, we additionally use a straight-through estimator to approximate gradients through the thresholding operation during backpropagation [40]. By contrast, TopK and AbsTopK SAEs explicitly control activation sparsity through a prespecified K and are therefore trained using only the reconstruction term, $\mathcal { L } ( \pmb { x } ) = \| \pmb { x } - \hat { \pmb { x } } \| _ { 2 } ^ { 2 }$

Data preprocessing and hyperparameter selection. Internal representations can difer substantially in numerical scale across models, layers, and extraction sites, making SAE training highly sensitive to hyperparameter choices and complicating the use of a unified training configuration. We first observed on Gemma-2-2B that SAEs trained directly on raw activations were stable in shallow layers but became increasingly unstable in deeper layers, largely due to pronounced variation in activation scale across depth. Scaling each internal representation by the global standard deviation restored stable training across layers, with consistent behavior in Gemma-2-9B. For representations extracted from sites such as ‘down proj‘, which are not normalized by ‘post feedforward layernorm‘, standard-deviation scaling alone did not yield suficiently consistent numerical distributions. We therefore applied RMS normalization without learnable afine parameters, scaling each internal representation by its root mean square. This produced stable training for Llama-3.1-8B, Qwen3.5-9B, Olmo-Hybrid-7B, Stable Difusion v1.5 and FLUX.2-klein-9B. Accordingly, representations extracted from ‘post feedforward layernorm‘ in Gemma-2-2B and Gemma-2-9B use standard-deviation scaling, whereas representations from ‘down proj‘ and other unnormalized sites use RMS normalization. Both operations apply only a scalar rescaling to each representation vector, preserving its direction and therefore the representational geometry considered here, and the original numerical scale can be recovered directly using the corresponding scaling factor.

The unified preprocessing yields comparable numerical scales across models, layers, and modalities, allowing all SAEs to use a common set of core hyperparameters without model- or layer-specific grid searches. BiJumpReLU SAEs use $\lambda \ : = \ : 5 . 0$ in all settings, with the negative and positive thresholds initialized $\mathrm { t o - 0 . 5 }$ and 0.5, respectively. ReLU SAEs likewise use $\lambda = 5 . 0$ , whereas JumpReLU SAEs use $\lambda = 5 . 0$ with the threshold initialized to 0.5. For TopK SAEs, which control sparsity by fixing the number of active components, we follow the standard setting of Gao et al. [13] with $K = 5 0$ and additionally evaluate $K = 2 0$ as a lower-sparsity baseline. AbsTopK SAEs use $K = 5 0$ , following the standard setting of Zhu et al.[14]

All remaining training settings are held fixed to minimize confounding from hyperparameter diferences across SAE architectures. We use the learning rate of $2 \times 1 0 ^ { - 4 }$ , the Adam optimizer with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ , and the batch size of 4096 for all models. Increasing the batch size further did not yield measurable performance gains.

SAE Capacity. For SAE benchmarking, we compare the ability of diferent SAE architectures to identify representational atoms under otherwise matched training conditions, while keeping dictionary sizes comparable across base models. We therefore choose model-specific expansion factors such that dictionary widths remain within approximately $2 . 1 - 2 . 2 \times 1 0 ^ { 5 } \colon 9 6 \times 2 3 0 4 \approx 2 2 1 \mathrm { K }$ for Gemma-2-2B, 60 × 3584 ≈ 215K for Gemma-2-9B, 54 × 4096 ≈ 221K for Qwen3.5-9B, and 58 × 3840 ≈ 223K for Olmo Hybrid-7B. For cross-layer semantic alignment, the objective is instead to obtain representational atoms with high fidelity and high stability, defined as $R ^ { 2 } > 0 . 9 9$ and $q ^ { * } > 0 . 9 9$ . Following the empirical guidance of Hu et al. [51], we select the SAE dictionary as compact as possible that satisfies these criteria and keep its width fixed across all layers of the same base model. Specifically, we use $6 4 \times 2 3 0 4 \approx 1 4 7 \mathrm { K }$ for Gemma-2-2B, 40 × 3584 ≈ 143K for Gemma-2-9B, and 35 × 4096 ≈ 143K for Llama-3.1-8B.

Training dynamics and early stopping. SAE training is commonly viewed as a trade-of between reconstruction quality and sparsity [13], but we observe a consistent training dynamic across multiple experimental settings. For BiJumpReLU SAEs, reconstruction and sparsity losses typically decrease together during early and intermediate training. As training progresses, the reconstruction loss gradually plateaus and may subsequently increase slightly but persistently, while sparsity continues to improve. Both objectives eventually reach an approximately stable regime. This pattern indicates that late-stage optimization primarily trades a limited change in reconstruction quality for increased sparsity, rather than continuing to improve both objectives simultaneously. Based on this dynamic, we adopt a trajectory-based early stopping criterion. For BiJumpReLU and JumpReLU SAEs, we smooth the training-loss curves and stop training near the transition at which reconstruction loss shifts from sustained decrease to a stable increase while sparsity continues to improve. This stage typically already corresponds to high fidelity and stability, allowing us to avoid unnecessary late-stage trade-ofs while preserving atom-identification quality. For ReLU SAEs, where this transition is generally less pronounced, training is stopped when neither reconstruction nor sparsity loss shows further substantial improvement. For $\mathrm { T o p } K$ and AbsTopK SAEs, whose sparsity is explicitly fixed by K, training is stopped when the reconstruction performance no longer improves appreciably.

## Evaluation details

To systematically evaluate the quality and semantic organization of identified representational atoms, we describe the implementation of fidelity and stability metrics, monosemanticity evaluation, and alignment analysis between neural representational structure and human conceptual structure in the following subsection.

Fidelity metrics and computation. Fidelity measures how well candidate FRUs preserve or reconstruct the original internal representations and is quantified by the coeficient of determination

$$
R ^ { 2 } : = 1 - \frac { \sum _ { i } \| \pmb { m } _ { i } - \pmb { \hat { m } } _ { i } \| _ { 2 } ^ { 2 } } { \sum _ { i } \| \pmb { m } _ { i } - \pmb { \hat { m } } \| _ { 2 } ^ { 2 } } ,\tag{19}
$$

where $\hat { m } _ { i }$ denotes the internal representation reconstructed from the candidate FRUs and m¯ is the sample mean. Values of $R ^ { 2 }$ closer to 1 indicate more faithful reconstruction of the original representations.

Stability metrics and computation. To transform the distinguishability condition $\mu < 1 / ( 2 K - 1 )$ from Theorems 9 and Corollary 10 into a computable stability metric, we introduce quantile statistics that characterize the empirical distributions of sparsity and coherence. We first define the quantile sparsity $K _ { q }$ as

$$
K _ { q } = \operatorname* { i n f } \left\{ k \in { \mathbb { N } } ^ { + } \mid { \mathbb { P } } _ { \delta \sim { \mathcal { P } } _ { \Delta } } \left( \| \delta \| _ { 0 } \leq k \right) \geq q \right\} ,\tag{20}
$$

where δ is sampled from the coeficient distribution ${ \mathcal { P } } _ { \Delta }$ . Thus, $K _ { q }$ is the smallest integer such that at least a fraction $q$ of samples have sparsity no greater than $K _ { q } .$

Similarly, we define the quantile coherence $\mu _ { q }$ as

$$
\mu _ { q } = \operatorname* { i n f } \left. \mu \geq 0 \mid \mathbb { P } _ { ( \mathcal { Z } , \mathcal { T } ) } \left( \left. \left. \tilde { d } _ { \mathcal { Z } } , \tilde { d } _ { \mathcal { T } } \right. \right. \leq \mu \right) \geq q \right. ,\tag{21}
$$

where $( \mathcal { T } , \mathcal { T } )$ is sampled uniformly from the unordered pairs of distinct representational-unit indices, and $\left| \left. \tilde { \mathbfit { d } } _ { \mathcal { T } } , \tilde { \mathbfit { d } } _ { \mathcal { T } } \right. \right|$ denotes their coherence. Consequently, $\mu _ { q }$ gives an upper bound on coherence satisfied by at least a fraction $q$ of unit pairs.

Using the distinguishability criterion above, we evaluate $\begin{array} { r } { { \mu _ { q } } \ < \ \frac { 1 } { 2 K _ { q } - 1 } } \end{array}$ at each quantile and define stability as the largest quantile that satisfies the condition

$$
q ^ { * } : = \operatorname* { s u p } \left\{ q \mid \mu _ { q } < { \frac { 1 } { 2 K _ { q } - 1 } } \right\} .\tag{22}
$$

Thus, $q ^ { * }$ denotes the largest quantile at which empirical sparsity and coherence satisfy the theoretical distinguishability condition, with the ideal value $q ^ { * } = 1$

In practice, we compute $q ^ { * }$ using binary search. Specifically, on [0, 0.999999], we iteratively evaluate

$$
\mu _ { \alpha } : = \operatorname { Q u a n t i l e } ( \mu , \alpha ) , \quad K _ { \alpha } : = \operatorname { Q u a n t i l e } ( K , \alpha ) ,\tag{23}
$$

and test whether $\textstyle \mu _ { \alpha } < { \frac { 1 } { 2 K _ { \alpha } - 1 } }$ .If the condition is satisfied, the lower search bound increases; otherwise, the upper bound decreases until convergence. The largest feasible quantile is reported as $q ^ { * }$

Evaluation of existing representational units. To evaluate existing representational units within a unified framework, we treat neurons and learned features by publicly available SAEs as candidate FRUs and assess them using the fidelity and stability metrics defined above. All evaluations use approximately 20K entities appearing in CounterFact [25], which can be regarded as a subset of Wikidata entities. Each entity is converted into a corresponding prompt and fed into the model, and internal representations are extracted at the final token position of the target site. Neurons and learned features are evaluated directly using the original models and publicly released SAE checkpoints, without additional training or parameter tuning.

For neurons, we extract the input to the MLP ‘down proj‘ at each layer. This vector corresponds to the activations of the intermediate MLP neurons, with each component indicating the activation strength of the corresponding neuron. The columns of the ‘down proj‘ weight matrix are therefore treated as the representational directions associated with individual neurons, and the corresponding input activations as their coeficients, allowing the MLP residual update to be expressed as a linear combination of neuron directions. Fidelity and stability are then computed for neurons as candidate FRUs according to the definitions above.

For learned features, we use the pretrained SAEs provided by Gemma Scope [20] and Llama Scope [21], following their oficial settings. For Gemma-2-2B and Gemma-2-9B, we extract representations at the output of ‘post feedforward layernorm‘ from each layer and apply the corresponding Gemma Scope SAE. For Llama-3.1-8B, we extract representations at the output of ‘down proj‘ from each layer and apply the corresponding Llama Scope SAE. The decoder directions of each SAE are treated as candidate FRUs, and the latent activations as their coeficients. The resulting SAE reconstructions are used to compute fidelity, whereas feature coherence and activation sparsity are used to compute stability under the same evaluation procedure.

Monosemanticity evaluation details. To evaluate the monosemanticity of the candidate FRUs, we use a common scoring criterion across language models and difusion models, with modality-specific evaluation procedures. For language models, including Gemma-2-2B, Gemma-2-9B, Qwen3.5-9B and Olmo-Hybrid-7B, we use an LLM-as-a-judge protocol supplemented by manual verification. For difusion models, including Stable Difusion v1.5 and FLUX.2-klein-9B, we use manual evaluation. Both procedures follow the same definition of monosemanticity and a closely matched scoring template.

For each model and representation extraction site, we randomly sample 10 candidate FRUs identified by the corresponding SAE and collect the input instances that activate each unit. If more than 10 instances are activated, we randomly sample 10 for evaluation. The evaluated instances are contextualized tokens for Wikipedia, entities for Wikidata and local image patches for LAION-COCO. For each candidate FRU, we identify the largest subset $S _ { \mathrm { m a x } }$ of evaluated instances that can be coherently explained by a common concept, attribute or semantic pattern, and define its monosemanticity score as

$$
\mathrm { M o n o s e m a n t i c i t y } = \frac { | S _ { \mathrm { m a x } } | } { | S | } ,\tag{24}
$$

where S denotes the complete set of evaluated instances for that unit. Scores closer to 1 indicate that the unit activates more consistently on a coherent semantic pattern. Finally, we average the monosemanticity scores of the sampled units within each SAE setting to obtain its overall monosemanticity score.

Semantic alignment evaluation details. HowNet [17] contains 2,540 sememes, each associated with a set of entities belonging to the corresponding semantic category. To ensure that subsequent set-based comparisons are meaningful, we retain only sememes containing at least two unique entities, yielding 2,135 sememes, which we use as an operational representation of human conceptual semantic structure. On the model side, we extract internal representations corresponding to HowNet entities at every layer and train BiJumpReLU SAEs at each layer to identify atoms. For each unit, we record the set of entities that activate it and likewise retain only units associated with at least two unique entities. Each retained sememe and representational unit can therefore be represented as an entity set, providing a common basis for comparing neural representational structure with human conceptual structure.

To evaluate semantic alignment between single atoms and HowNet sememes, we compute precision, recall and F1 based on the overlap between their corresponding entity sets. For each sememe, we search across all atoms identified at all layers and define the unit with the highest F1 score as its best single-unit match. This metric jointly reflects the semantic purity of a atom and its coverage of the target sememe, thereby quantifying the extent to which an single atoms corresponds to human conceptual semantic structure. Detailed set construction, metric definitions and cross-layer search procedures are provided in ‘Single-atom semantic alignment evaluation’ in the Supplementary Information.

Building on the single-atom analysis, we further assess whether combinations of atoms can recover sememe structure more completely. We aggregate candidate FRUs identified across all layers, prefilter them according to their correspondence with the target sememe entity set, and then perform greedy compositional search by iteratively adding the atom that yields the largest improvement in the current F1 score. Search terminates when no further positive gain is obtained or when a prespecified maximum combination size is reached. The candidate-filtering threshold and maximum combination size are selected by grid search (Supplementary Fig. 1). This analysis compares the coverage achieved by single atoms and atom combinations for the same sememe and tests whether higher-level human conceptual semantic structures can be formed compositionally from representational atoms distributed across layers. Detailed procedures for candidate construction, compositional search and hyperparameter selection are provided in ‘Semantic alignment evaluation’ in the Supplementary Information.

## Use of large language models

During the preparation of this manuscript, we used ChatGPT (OpenAI; GPT-5.6 Sol) as a writingsupport tool. Its use was limited to language refinement, checks of logical consistency and coherence, and comparison of alternative phrasings. All AI-assisted suggestions were reviewed and evaluated by the authors, and any suggestions incorporated into the manuscript were revised as appropriate. The theoretical development, methodological design, experimental implementation and data analysis were conducted by the authors. The authors independently determined the scientific conclusions and take full responsibility for the accuracy and final content of the manuscript.

## Data availability

The Wikipedia data used in this study were obtained from the ‘20231101.en’ configuration at https: //huggingface.co/datasets/wikimedia/wikipedia. The Wikidata data were obtained from the publicly available Wikidata resources [48] at https://www.wikidata.org/wiki/Wikidata:Data access and were further preprocessed to obtain a more balanced distribution across entity categories. The LAION-COCO data were obtained from the public LAION-COCO release [49] at https://huggingface.co/datasets/ guangyil/laion-coco-aesthetic. HowNet data were obtained through OpenHowNet [16, 17] at https: //github.com/thunlp/OpenHowNet.

## Code availability

The source code and configuration files used for representation extraction, sparse autoencoder training, fidelity and stability evaluation, monosemanticity analysis, and semantic alignment are publicly available at https://github.com/ChenhuiHu/representational atoms. The repository also provides documentation and example configurations for reproducing the experimental pipeline described in this study.

## References

[1] LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. nature 521, 436–444 (2015).

[2] Bengio, Y., Courville, A. & Vincent, P. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence 35, 1798–1828 (2013).

[3] Bau, D. et al. Understanding the role of individual units in a deep neural network. Proceedings of the National Academy of Sciences 117, 30071–30078 (2020).

[4] Klindt, D., O’Neill, C., Reizinger, P., Maurer, H. & Miolane, N. A unifying framework from neural superposition to sparse interpretable codes. Nature Machine Intelligence 1–13 (2026).

[5] Donoho, D. L. Compressed sensing. IEEE Transactions on information theory 52, 1289–1306 (2006).

[6] Cand\`es, E. J., Romberg, J. & Tao, T. Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. IEEE Transactions on information theory 52, 489–509 (2006).

[7] Chen, Y., Cao, P., Chen, Y., Liu, K. & Zhao, J. Knowledge localization: Mission not accomplished? enter query localization! International Conference on Learning Representations 2025, 29589–29613 (2025).

[8] Huben, R., Cunningham, H., Smith, L. R., Ewart, A. & Sharkey, L. Sparse autoencoders find highly interpretable features in language models. The Twelfth International Conference on Learning Representations (2024). URL https://openreview.net/forum?id=F76bwRSLeK.

[9] Bricken, T. et al. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread (2023). Https://transformer-circuits.pub/2023/monosemanticfeatures/index.html.

[10] Templeton, A. et al. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread (2024). URL https://transformer-circuits.pub/2024/ scaling-monosemanticity/index.html.

[11] Elhage, N. et al. Toy models of superposition. arXiv preprint arXiv:2209.10652 (2022).

[12] Chen, Y., Cao, P., Liu, K. & Zhao, J. The knowledge microscope: Features as better analytical lenses than neurons. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) 10493–10515 (2025).

[13] Gao, L. et al. Scaling and evaluating sparse autoencoders. International Conference on Learning Representations 2025, 26721–26754 (2025).

[14] Zhu, X., Khalili, M. M. & Zhu, Z. Abstopk: Rethinking sparse autoencoders for bidirectional features. International Conference on Learning Representations 2026, 50325–50352 (2026).

[15] Rajamanoharan, S. et al. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders. arXiv preprint arXiv:2407.14435 (2024).

[16] Dong, Z. & Dong, Q. Hownet - a hybrid language and knowledge resource. International Conference on Natural Language Processing and Knowledge Engineering, 2003. Proceedings. 2003 820–824 (2003). URL https://api.semanticscholar.org/CorpusID:7462030.

[17] Qi, F. et al. Openhownet: An open sememe-based lexical knowledge base. arXiv preprint arXiv:1901.09957 (2019).

[18] Donoho, D. L. & Elad, M. Optimally sparse representation in general (nonorthogonal) dictionaries via l1 minimization. Proceedings of the National Academy of Sciences 100, 2197–2202 (2003).

[19] Tropp, J. A. Greed is good: Algorithmic results for sparse approximation. IEEE Transactions on Information theory 50, 2231–2242 (2004).

[20] Lieberum, T. et al. Belinkov, Y. et al. (eds) Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. (eds Belinkov, Y. et al.) Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, 278–300 (Association for Computational Linguistics, Miami, Florida, US, 2024). URL https://aclanthology.org/2024.blackboxnlp-1.19/.

[21] He, Z. et al. Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders. arXiv preprint arXiv:2410.20526 (2024).

[22] Olah, C. et al. Zoom in: An introduction to circuits. Distill 5, e00024–001 (2020).

[23] Lakens, D. Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. Social psychological and personality science 8, 355–362 (2017).

[24] Park, K., Choe, Y. J. & Veitch, V. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658 (2023).

[25] Meng, K., Bau, D., Andonian, A. & Belinkov, Y. Locating and editing factual associations in gpt. Advances in neural information processing systems 35, 17359–17372 (2022).

[26] Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y. & Bau, D. Mass-editing memory in a transformer. The Eleventh International Conference on Learning Representations (2023). URL https://openreview.net/forum?id=MkbcAHIYgyS.

[27] Hu, C., Cao, P., Chen, Y., Liu, K. & Zhao, J. Wilke: Wise-layer knowledge editor for lifelong knowledge editing. Findings of the Association for Computational Linguistics: ACL 2024 3476–3503 (2024).

[28] Rimsky, N. et al. Steering llama 2 via contrastive activation addition. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) 15504–15522 (2024).

[29] Rodriguez, P. et al. Controlling language and difusion models by transporting activations. International Conference on Learning Representations 2025, 89812–89855 (2025).

[30] Kirkpatrick, J. et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114, 3521–3526 (2017).

[31] Van de Ven, G. M., Tuytelaars, T. & Tolias, A. S. Three types of incremental learning. Nature Machine Intelligence 4, 1185–1197 (2022).

[32] Hu, C., Cao, P., Chen, Y., Liu, K. & Zhao, J. Knowledge in superposition: Unveiling the failures of lifelong knowledge editing for large language models. Proceedings of the AAAI Conference on Artificial Intelligence 39, 24086–24094 (2025).

[33] Goeßmann, A. & Kutyniok, G. The restricted isometry of relu networks: Generalization through norm concentration. arXiv preprint arXiv:2007.00479 (2020).

[34] Chen, Z. & Schaefer, H. Conditioning of random feature matrices: Double descent and generalization error. arXiv preprint arXiv:2110.11477 (2021).

[35] Foucart, S. & Rauhut, H. An Invitation to Compressive Sensing, 1–39 (Springer New York, New York, NY, 2013). URL https://doi.org/10.1007/978-0-8176-4948-7 1.

[36] Murdock, C. & Lucey, S. Dataless model selection with the deep frame potential. CoRR abs/2003.13866 (2020). URL https://doi.org/10.48550/arXiv.2003.13866.

[37] Donoho, D. L. & Huo, X. Uncertainty principles and ideal atomic decomposition. IEEE transactions on information theory 47, 2845–2862 (2001).

[38] Candes, E. J. & Tao, T. Decoding by linear programming. IEEE transactions on information theory 51, 4203–4215 (2005).

[39] O’Neill, C., Gumran, A. & Klindt, D. Compute optimal inference and provable amortisation gap in sparse autoencoders. arXiv preprint arXiv:2411.13117 (2024).

[40] Rajamanoharan, S. et al. Improving dictionary learning with gated sparse autoencoders. arXiv preprint arXiv:2404.16014 (2024).

[41] Team, G. et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118 (2024).

[42] Grattafiori, A. et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).

[43] Team, Q. Qwen3. 5-omni technical report. arXiv preprint arXiv:2604.15804 (2026).

[44] Merrill, W. et al. Olmo hybrid: From theory to practice and back. arXiv preprint arXiv:2604.03444 (2026).

[45] Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. High-resolution image synthesis with latent difusion models. 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR) 10674–10685 (2022).

[46] Labs, B. F. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2 (2025).

[47] V¨olkel, M., Kr¨otzsch, M., Vrandecic, D., Haller, H. & Studer, R. Semantic wikipedia. Proceedings of the 15th international conference on World Wide Web 585–594 (2006).

[48] Vrandeˇci´c, D. & Kr¨otzsch, M. Wikidata: a free collaborative knowledgebase. Communications of the ACM 57, 78–85 (2014).

[49] Christoph, S., Andreas, K., Richard, V., Theo, C. & Romain, B. Laion coco: 600m synthetic captions from laion2b-en. https://laion.ai/blog/laion-coco/ (2022).

[50] Tinaz, B., Fabian, Z. & Soltanolkotabi, M. Emergence and evolution of interpretable concepts in difusion models. Advances in neural information processing systems 38, 166943 (2025).

[51] Hu, C., Cao, P., Chen, Y., Liu, K. & Zhao, J. Towards atoms of large language models. Forty-third International Conference on Machine Learning (2026).

a  
![](images/894867af98e0c9dce42a4f66eea4fde7f35fb2c06dda276e1c4b68e14efdbb6e.jpg)

![](images/ac26e35fca3b540ecd05a00d377a7d99c218acaae48db7fad47ce65b65226165.jpg)

![](images/a4bafabbc10bb7a2efaa60cc02a93f8d9aa23aad5710ea4999d22ff3e2428e45.jpg)

![](images/4de2545c8217835aa0314bbf789e76ba9bf3579a88834eb0e5d0bead160bbd5b.jpg)

![](images/010f35d2f228b4f743db0d76ffe1182cf698c81b3fc10f00acfb536a87aeca44.jpg)

b  
![](images/5131c8571cc34da1e49940780834d009242258b303919214828eceaf9807270b.jpg)

![](images/78eede19f6e24b4a87c7498c41f64f39ab642bc6f380ca32a5ab184e765a3365.jpg)

Gemma2-2B Gemma2-9B Qwen3.5-9B  
![](images/f4e5a0469bb5b189b9e343e2528a870a30fd75a158af8a9132735ab52b94fe40.jpg)

![](images/fd6a464d54818e8d5e4237a9842cfa8f57308c27eb4455ad8ffbf2f13ac3ecac.jpg)

![](images/8d77fb3cb1c61ee5e3fb1c14941966f02bcc3cd4ffb5e8fbc345a0dfd76d908a.jpg)

c  
![](images/1c9b6438ed797bc018993e7c2f982dd29cf2fc8775b3af489ee995f5b0dbf8e4.jpg)

![](images/cf4a71bb984a37ed9d3621a99b3a62d0bc372a8f8b2b905bee80851407289df1.jpg)

Gemma2-2B Gemma2-9B Olmo-Hybrid-7B  
![](images/4c7d1e6396681f2be9e69d439c0ccbb9438fb8ccd362673cf8fbebeef2348cfd.jpg)

![](images/8784a08008dd4c3da54e18ade4b8c9d4bf0e3bd5425bef1c9555f271cb660228.jpg)

![](images/f855e1d275675ba687a68c2b7c1dadd903f4c0df7898368982309cd2e9b3bc97.jpg)

d  
![](images/bc3eaa0b80855fe4efffcc354d7db737cbbfb4659c7736ce46b82b8ec3ffa183.jpg)

![](images/d7069e9203ef4b5e3ad67ee6a31cae422e6ae1d1c323ed2942ad13a7b5b6f8d0.jpg)

Stable Diffusion v1.5 FLUX.2-klein-9B  
![](images/eef38cae53e4702cb02eebab18e3f0943e9eaceead984dd32d368798991438d4.jpg)

![](images/f7da8dbbaf03c4e493c9bba87258720d05c58e8ac4eff85990cb088eeeea9147.jpg)

![](images/30af235058018a530abe7321722067ba718cb661a189c80914cfaaad55e9599b.jpg)

Extended Data Fig. 1 Architecture-stratified associations between stability and monosemanticity. a, Stability scores are plotted against mean monosemanticity scores separately for BiJumpReLU, JumpReLU, ReLU, TopK $( K = 5 0 )$ and AbsTopK $( K = 5 0 )$ SAEs, pooling all evaluated models and datasets. Within each SAE panel, each point represents one evaluated model–condition pair: one layer for language models, one difusion timestep for Stable Difusion v1.5, or one layer–step combination for FLUX.2-klein-9B (N = 45 per SAE architecture). Marker shapes denote datasets. b, The same analysis restricted to Wikipedia, pooling Gemma2-2B, Gemma2-9B and Qwen3.5-9B $( N = 1 8$ per SAE architecture). c, The same analysis restricted to Wikidata, pooling Gemma2-2B, Gemma2-9B and Olmo-Hybrid-7B $( N = 1 8$ per SAE architecture). d, The same analysis restricted to LAION-COCO, pooling Stable Difusion v1.5 and FLUX.2-klein-9B $( N = 9$ per SAE architecture). In b–d, marker shapes denote individual models. Black lines show linear regression fits with 95% confidence bands. Spearman’s rank correlation coeficients (ρ) and two-sided P values are shown in each panel.

a  
![](images/6243579f3dc20c3a91417fe3661c63e415ae0637448ec09b1b1a065c2a44ed51.jpg)

b  
![](images/b1d40dd271a140ea8659d3803e22f1fa0507c45e0d90b87bb798b8cc85f0a56b.jpg)

c  
![](images/a60a4a566e694bad48fc8fae200e608d05aa9f20c0fe58fd3776bcf6d428258b.jpg)  
Extended Data Fig. 2 Distributions of monosemanticity scores across SAE architectures. a–c, Distributions of monosemanticity scores for BiJumpReLU, JumpReLU, ReLU, TopK (K = 50) and AbsTopK (K = 50) SAEs on Wikipedia (a), Wikidata (b) and LAION-COCO (c). For each SAE trained on representations from the corresponding evaluated layer, difusion timestep or layer–step combination, ten FRUs were assessed individually using their activation contexts, yielding one monosemanticity score per FRU. The ten scores were averaged to obtain the monosemanticity score for that SAE at the corresponding representation condition.

Sememe-atom structural map (Gemma2-2B) (One-to-one is rare; composition is common.)

![](images/02aa7aac9eda723e1993c41396089829000e97450a063935f8d7301856771ae5.jpg)

Human sememe Representational atom Best single atom (selected for composition) Best single atom (not selected for composition)

Extended Data Fig. 3 Sememe–atom structural map for Gemma2-2B. Visualization of the alignment between HowNet sememes and representational atoms identified by BiJumpReLU SAEs across layers of Gemma2-2B. Large lightblue circles denote human sememes and smaller blue circles denote representational atoms selected for their compositional recovery. Solid orange circles indicate the best single atoms that are retained in the composition, whereas dashed orange circles indicate the best single atoms that are not retained. Circle sizes reflect the sizes of the corresponding semantic sets, and the geometric overlap between an atom and its sememe reflects their single-atom precision. Geometric overlap between atom circles is not intended to represent overlap between their associated entity sets.

Sememe-atom structural map (Gemma2-9B) (One-to-one is rare; composition is common.)

![](images/af01d799e5133b6030485bfff775fc3dedea0bbd16fe36841cfffcbad244e9a6.jpg)

Human sememe Representational atom Best single atom (selected for composition) Best single atom (not selected for composition)

Extended Data Fig. 4 Sememe–atom structural map for Gemma2-9B. Visualization of the alignment between HowNet sememes and representational atoms identified by BiJumpReLU SAEs across layers of Gemma2-9B. Large lightblue circles denote human sememes and smaller blue circles denote representational atoms selected for their compositional recovery. Solid orange circles indicate the best single atoms that are retained in the composition, whereas dashed orange circles indicate the best single atoms that are not retained. Circle sizes reflect the sizes of the corresponding semantic sets, and the geometric overlap between an atom and its sememe reflects their single-atom precision. Geometric overlap between atom circles is not intended to represent overlap between their associated entity sets.

Sememe-atom structural map (Llama3.1-8B) (One-to-one is rare; composition is common.)

![](images/8ee3d8f1e4e6a98e2be6d779753bb455ea42b2a4dac0f2c9d75499feb998518c.jpg)

Human sememe Representational atom Best single atom (selected for composition) Best single atom (not selected for composition)

Extended Data Fig. 5 Sememe–atom structural map for Llama3.1-8B. Visualization of the alignment between HowNet sememes and representational atoms identified by BiJumpReLU SAEs across layers of Llama3.1-8B. Large lightblue circles denote human sememes and smaller blue circles denote representational atoms selected for their compositional recovery. Solid orange circles indicate the best single atoms that are retained in the composition, whereas dashed orange circles indicate the best single atoms that are not retained. Circle sizes reflect the sizes of the corresponding semantic sets, and the geometric overlap between an atom and its sememe reflects their single-atom precision. Geometric overlap between atom circles is not intended to represent overlap between their associated entity sets.

Extended Data Table 1 Comprehensive evaluation of sparse autoencoder representations across datasets and models. In addition to fidelity and stability, the table reports feature utilization and retention.
<table><tr><td rowspan=1 colspan=16>GemmaScope                             Trained SAEsDataset      Model           MetricSmall      Large     ReLU       $\mathrm { T o p K }$     AbsTopK  JumpReLU BiJumpReLU</td></tr><tr><td rowspan=1 colspan=7>Utilization ↑  95.62 (±5.55) 76.02 (±6.66)</td><td rowspan=1 colspan=7>23.63 (±32.29)15.62 (±5.24) 16.45 (±5.74) 67.63 (±14.58)</td><td rowspan=1 colspan=2>91.47 (±20.80)</td></tr><tr><td rowspan=3 colspan=4>Gemma2-2B</td><td rowspan=2 colspan=2>Retention α1 ↑85.74 (±11.90)Fidelity ↑    72.17(±13.21)</td><td rowspan=1 colspan=1>85.17(±12.55)</td><td rowspan=1 colspan=1>96.41 (±1.75)</td><td rowspan=1 colspan=3>70.02 (±1.31) 69.72 (</td><td rowspan=1 colspan=1>±1.33)</td><td rowspan=1 colspan=2>99.87 (±0.10)</td><td rowspan=1 colspan=2>99.97 (±0.02)</td></tr><tr><td rowspan=1 colspan=1>75.52 (±13.12)</td><td rowspan=1 colspan=1>75.46 (±16.17)</td><td rowspan=1 colspan=3>88.68 (±7.05) 88.11 (</td><td rowspan=1 colspan=1>±6.93)</td><td rowspan=1 colspan=2>94.91 (±2.38)</td><td rowspan=1 colspan=2>97.25 (±1.44)</td></tr><tr><td rowspan=2 colspan=4></td><td rowspan=1 colspan=2>Stability α2 ↑ 67.87(±16.69)</td><td rowspan=1 colspan=1>67.43 (±18.43)</td><td rowspan=1 colspan=1>85.58 (±7.40)</td><td rowspan=1 colspan=3>39.34 (±1.01) 39.07 (</td><td rowspan=1 colspan=1>±1.02)</td><td rowspan=1 colspan=2>98.41 (±1.57)</td><td rowspan=1 colspan=2>99.78 (±0.13)</td></tr><tr><td rowspan=1 colspan=2>Utilization ↑  89.65 (±7.05)</td><td rowspan=1 colspan=1>57.61 (±7.25)</td><td rowspan=1 colspan=1>25.65 (±34.93)</td><td rowspan=1 colspan=4>16.35 (±6.04) 19.28 (±5.45)</td><td rowspan=1 colspan=2>54.65 (±34.54)</td><td rowspan=1 colspan=2>90.13 (±25.28)</td></tr><tr><td rowspan=3 colspan=4>Wikipedia    Gemma2-9BF</td><td rowspan=2 colspan=2>Retention α1 ↑98.67(±1.10)idelity ↑    65.56 (±17.21)</td><td></td><td></td><td rowspan=1 colspan=4>81.58 (±1.64) 80.58 (±0.72)</td><td rowspan=1 colspan=2>99.79 (±0.20)</td><td rowspan=1 colspan=2>99.97 (±0.01)</td></tr><tr><td rowspan=1 colspan=1>70.78 (±15.96)</td><td rowspan=1 colspan=1>74.25 (±22.06)</td><td rowspan=1 colspan=2>87.42 (±7.56)</td><td rowspan=1 colspan=2>87.65 (±7.05)</td><td rowspan=1 colspan=2>93.38 (±3.24)</td><td rowspan=1 colspan=2>96.75 (±1.86)</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Stability α2 ↑</td><td rowspan=1 colspan=1>91.93 (±5.17)</td><td rowspan=1 colspan=1>89.00 (±8.03)</td><td rowspan=1 colspan=1>89.46 (±4.57)</td><td rowspan=1 colspan=2>49.25 (±1.81)</td><td rowspan=1 colspan=1>48.12 (</td><td rowspan=1 colspan=1>±0.70)</td><td rowspan=1 colspan=2>97.92 (±1.43)</td><td rowspan=1 colspan=2>99.81 (±0.08)</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Utilization ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>17.05 (±10.09)</td><td rowspan=1 colspan=2>28.03 (±8.79)</td><td rowspan=1 colspan=1>29.13 (</td><td rowspan=1 colspan=1>±8.52)</td><td rowspan=1 colspan=2>81.69 (±33.50)</td><td rowspan=1 colspan=2>97.48 (±6.48)</td></tr><tr><td rowspan=3 colspan=4>S</td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Retention α1 ↑Fidelity ↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>77.37 (±7.32)</td><td rowspan=1 colspan=2>87.13 (±6.34)</td><td rowspan=1 colspan=1>86.43 (</td><td rowspan=1 colspan=1>±6.33)</td><td rowspan=1 colspan=2>94.47(±4.52)</td><td rowspan=1 colspan=2>95.66 (±2.14)</td></tr><tr><td rowspan=1 colspan=1>tability α2 ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>82.84 (±6.25)</td><td rowspan=1 colspan=2>51.11 (±1.20)</td><td rowspan=1 colspan=1>50.73 (</td><td rowspan=1 colspan=1>±1.03)</td><td rowspan=1 colspan=2>82.74 (±14.20)</td><td rowspan=1 colspan=2>99.35 (±0.57)</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>Utilization ↑  85.54 (±16.78)</td><td rowspan=1 colspan=1>57.47 (±17.53)</td><td rowspan=1 colspan=1>7.65 (±3.29)</td><td rowspan=1 colspan=2>8.12 (±5.45)</td><td rowspan=1 colspan=2>8.49 (±5.41)</td><td rowspan=1 colspan=2>34.68 (±25.10)</td><td rowspan=1 colspan=2>56.91 (±31.55)</td></tr><tr><td rowspan=2 colspan=4>Gemma2-2BF</td><td rowspan=2 colspan=2>Retention α1 ↑87.81 (±11.63)idelity ↑    72.44(±6.51)</td><td rowspan=1 colspan=1>87.12 (±12.58)</td><td rowspan=1 colspan=1>96.40(±2.07)</td><td rowspan=1 colspan=1>73.14</td><td rowspan=1 colspan=1>(±4.43)</td><td rowspan=1 colspan=1>72.05</td><td rowspan=1 colspan=1>(+2.93)</td><td rowspan=1 colspan=2>99.70 (±0.22)</td><td rowspan=1 colspan=1>99.87</td><td rowspan=1 colspan=1>(±0.17)</td></tr><tr><td rowspan=1 colspan=1>76.50(±5.75)</td><td rowspan=1 colspan=1>87.42 (±5.58)</td><td rowspan=1 colspan=1>92.15 (</td><td rowspan=1 colspan=1>±4.33)</td><td rowspan=1 colspan=1>92.03</td><td rowspan=1 colspan=1>(+3.86)</td><td rowspan=1 colspan=1>93.78</td><td rowspan=1 colspan=1>(±2.97)</td><td rowspan=1 colspan=1>95.21</td><td rowspan=1 colspan=1>(+3.96)</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>Stability α2 ↑ 70.15 (±16.62)</td><td rowspan=1 colspan=1>69.21 (±17.35)</td><td rowspan=1 colspan=1>84.77 (±6.95)</td><td rowspan=1 colspan=1>42.06 (</td><td rowspan=1 colspan=1>±3.97)</td><td rowspan=1 colspan=1>40.95 (</td><td rowspan=1 colspan=1>±2.38)</td><td rowspan=1 colspan=1>96.97 (</td><td rowspan=1 colspan=1>±2.21)</td><td rowspan=1 colspan=1>98.32 (</td><td rowspan=1 colspan=1>±2.00)</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>Utilization ↑  70.94 (±17.20)</td><td rowspan=1 colspan=1>33.89 (±13.23)</td><td rowspan=1 colspan=1>8.07 (±3.90)</td><td rowspan=1 colspan=1>8.10 (±</td><td rowspan=1 colspan=1>4.88)</td><td rowspan=1 colspan=1>10.91 (</td><td rowspan=1 colspan=1>±4.07)</td><td rowspan=1 colspan=1>25.68 (±</td><td rowspan=1 colspan=1>12.69)</td><td rowspan=1 colspan=1>68.81 (±</td><td rowspan=1 colspan=1>24.06)</td></tr><tr><td rowspan=1 colspan=4>Wikidata     Gemma2-9B</td><td rowspan=1 colspan=2>Retention α1 ↑98.46 (±2.64)Fidelity ↑    67.81 (±14.09)</td><td rowspan=1 colspan=1>96.94(±5.51)</td><td rowspan=1 colspan=1>96.84(±3.07)</td><td rowspan=1 colspan=1>85.29(</td><td rowspan=1 colspan=1>±3.77)</td><td rowspan=1 colspan=1>82.31 (</td><td rowspan=1 colspan=1>±1.34)</td><td rowspan=1 colspan=1>98.78 (</td><td rowspan=1 colspan=1>±1.53)</td><td rowspan=1 colspan=1>99.94(</td><td rowspan=1 colspan=1>±0.03)</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>Stability α2 ↑ 92.47(±9.33)</td><td rowspan=1 colspan=1>89.53 (±13.16)</td><td rowspan=1 colspan=1>86.18 (±9.03)</td><td rowspan=1 colspan=1>53.69 (</td><td rowspan=1 colspan=1>±4.78)</td><td rowspan=1 colspan=1>49.90 (</td><td rowspan=1 colspan=1>±1.41)</td><td rowspan=1 colspan=2>93.70 (±6.48)</td><td rowspan=1 colspan=1>99.11 (</td><td rowspan=1 colspan=1>±1.02)</td></tr><tr><td rowspan=4 colspan=4>Olmo-Hybrid-7BF</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>Utilization ↑</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>13.72 (±10.60)</td><td rowspan=1 colspan=1>18.00 (</td><td rowspan=1 colspan=1>±15.33)</td><td rowspan=1 colspan=1>22.52 (±</td><td rowspan=1 colspan=1>12.38)</td></tr><tr><td rowspan=2 colspan=2>Retention α1 ↑idelity ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>93.34 (±5.63)</td><td rowspan=1 colspan=1>78.03 (</td><td rowspan=1 colspan=1>±11.11)</td><td rowspan=1 colspan=1>81.86 (</td><td rowspan=1 colspan=1>±1.13)</td><td rowspan=1 colspan=2>98.04 (±1.84)</td><td rowspan=1 colspan=1>99.95 (</td><td rowspan=1 colspan=1>±0.03)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>80.04 (±12.38)</td><td rowspan=1 colspan=1>86.18 (</td><td rowspan=1 colspan=1>±9.35)</td><td rowspan=1 colspan=1>86.60 (</td><td rowspan=1 colspan=1>±8.52)</td><td rowspan=1 colspan=2>90.29 (±6.87)</td><td rowspan=1 colspan=1>96.61 (</td><td rowspan=1 colspan=1>±1.70)</td></tr><tr><td rowspan=1 colspan=2>Stability α2 ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>79.04 (±12.59)</td><td rowspan=1 colspan=1>46.80 (</td><td rowspan=1 colspan=1>±8.05)</td><td rowspan=1 colspan=1>49.31 (</td><td rowspan=1 colspan=1>±1.24)</td><td rowspan=1 colspan=2>90.04 (±7.57)</td><td rowspan=1 colspan=1>98.56 (</td><td rowspan=1 colspan=1>±1.62)</td></tr><tr><td rowspan=4 colspan=4>Stable Diffusion v1.5LAION-COCO</td><td rowspan=2 colspan=2>Utilization ↑Retention α1 ↑Fidelity ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.79 (±8.96)</td><td rowspan=1 colspan=1>9.67 (</td><td rowspan=1 colspan=1>±3.74)</td><td rowspan=1 colspan=1>9.59 (±</td><td rowspan=1 colspan=1>3.84)</td><td rowspan=1 colspan=2>13.96 (±7.99)</td><td rowspan=1 colspan=1>59.46 (±</td><td rowspan=1 colspan=1>46.56)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>63.77 (±8.90)</td><td rowspan=1 colspan=1>81.04</td><td rowspan=1 colspan=1>+18.14)</td><td rowspan=1 colspan=1>80.06 (±</td><td rowspan=1 colspan=1>16.34)</td><td rowspan=1 colspan=1>75.22 (</td><td rowspan=1 colspan=1>±22.51)</td><td rowspan=1 colspan=1>85.50 (±</td><td rowspan=1 colspan=1>21.18)</td></tr><tr><td rowspan=1 colspan=3>Stability α2 ↑</td><td rowspan=1 colspan=1>72.94 (±14.41)</td><td rowspan=1 colspan=1>29.71 (</td><td rowspan=1 colspan=1>±0.49)</td><td rowspan=1 colspan=1>29.44(</td><td rowspan=1 colspan=1>±0.40)</td><td rowspan=1 colspan=1>82.86 (±</td><td rowspan=1 colspan=1>28.67)</td><td rowspan=1 colspan=1>98.52 (</td><td rowspan=1 colspan=1>±4.14)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Utilization ↑</td><td rowspan=1 colspan=1>1.13 (±0.82)</td><td rowspan=1 colspan=1>2.40 (</td><td rowspan=1 colspan=1>±2.91)</td><td rowspan=1 colspan=1>13.73 (±</td><td rowspan=1 colspan=1>15.20)</td><td rowspan=1 colspan=2>3.17 (±4.39)</td><td rowspan=1 colspan=1>65.00 (±</td><td rowspan=1 colspan=1>46.48)</td></tr><tr><td rowspan=3 colspan=4>FLUX.2-klein-9B</td><td rowspan=2 colspan=3>Retention α1 ↑Fidelity ↑</td><td rowspan=1 colspan=1>92.79 (±9.08)</td><td rowspan=1 colspan=1>68.20</td><td rowspan=1 colspan=1>11777)</td><td rowspan=1 colspan=1>82.85</td><td rowspan=1 colspan=1>+11.77)</td><td rowspan=1 colspan=2>84.14(±7.90)</td><td rowspan=1 colspan=1>99.86</td><td rowspan=1 colspan=1>(+0.28)</td></tr><tr><td rowspan=1 colspan=1>80.61 (±11.97)</td><td rowspan=1 colspan=1>88.33 (</td><td rowspan=1 colspan=1>±8.43)</td><td rowspan=1 colspan=1>90.05(</td><td rowspan=1 colspan=1>±9.11)</td><td rowspan=1 colspan=2>87.43 (±7.18)</td><td rowspan=1 colspan=1>96.03 (</td><td rowspan=1 colspan=1>±2.37)</td></tr><tr><td rowspan=1 colspan=3>Stability α2 ↑</td><td rowspan=1 colspan=1>81.83 (±18.80)</td><td rowspan=1 colspan=1>41.56 (</td><td rowspan=1 colspan=1>±14.95)</td><td rowspan=1 colspan=1>53.56 (±</td><td rowspan=1 colspan=1>12.21)</td><td rowspan=1 colspan=2>64.12 (±10.20)</td><td rowspan=1 colspan=1>98.47 (</td><td rowspan=1 colspan=1>±2.48)</td></tr></table>

Values are reported as mean followed by the half-width of the two-sided 95% t-confidence interval in parentheses, and all values are scaled by 100. Higher values indicate better performance. Bold indicates the highest mean value in each row. Utilization denotes the fraction of active SAE features and is reported only for the trained SAEs because GemmaScope and the trained SAEs use diferent SAE widths, which would confound direct comparisons of feature utilization. Retention α<sub>1</sub> is computed from the coherence condition $\mu _ { q } < 1 / ( K _ { q } - 1 )$ , whereas stability α<sub>2</sub> is computed from the distinguishability condition $\mu _ { q } < 1 / ( 2 K _ { q } - 1 )$ . For Wikipedia and Wikidata, confidence intervals are computed across six evaluated model layers. For LAION-COCO, confidence intervals are computed across three difusion timesteps for Stable Difusion v1.5 and across six layer-step combinations for FLUX.2-klein-9B. GemmaScope Small denotes width 16k; GemmaScope Large denotes width 65k for Gemma2-2B and width 131k for Gemma2-9B. TopK and AbsTopK use K = 50. Dashes indicate unavailable or non-comparable evaluations.

Extended Data Table 2 Efect of the sparsity parameter K on the fidelity and stability of TopK SAEs.
<table><tr><td>Model</td><td>K</td><td>Fidelity ↑</td><td> $\mathrm { S t a b i l i t y ~ } \uparrow$ </td></tr><tr><td rowspan="2"> $\mathrm { G e m m a 2 – 2 B }$ </td><td>20</td><td> $7 9 . 6 1 ( \pm 1 1 . 8 6 )$ </td><td> $8 4 . 8 6 \left( \pm 3 . 8 1 \right)$ </td></tr><tr><td>50</td><td> $8 8 . 6 8 ( \pm 7 . 0 5 ) $ </td><td> $3 9 . 3 4 \left( \pm 1 . 0 1 \right)$ </td></tr><tr><td rowspan="2"> $\mathrm { G e m m a 2 { - } 9 B }$ </td><td>20</td><td> $7 8 . 6 6 ( \pm 1 1 . 8 7 )$ </td><td> $9 3 . 3 9 \left( \pm 1 . 9 1 \right)$ </td></tr><tr><td>50</td><td> $8 7 . 4 2 ( \pm 7 . 5 6 )$ </td><td> $4 9 . 2 5 \left( \pm 1 . 8 1 \right)$ </td></tr></table>

All results are obtained using TopK SAEs trained on representations from Wikipedia. K denotes the number of latent feature activations retained for each input by the TopK activation. Values are reported as mean followed by the half-width of the two-sided 95% t-confidence interval in parentheses, and all values are scaled by 100.

<table><tr><td rowspan="2">Model</td><td colspan="3">Best single atom</td><td colspan="3">Atom combination</td><td colspan="3">Difference</td><td rowspan="2">Mean # atoms</td></tr><tr><td> $\operatorname* { P r e c i s i o n }$ </td><td> $\operatorname { R e c a l l }$ </td><td>F1</td><td> $\operatorname* { P r e c i s i o n }$ </td><td>Recall</td><td>F1</td><td> $\Delta { \mathrm { ~ p r e c i s i o n } }$ </td><td> $\Delta \mathrm { \ r e c a l l }$ </td><td>∆ F1</td></tr><tr><td>Gemma2-2B</td><td>0.691 (±0.011)</td><td>0.232 (±0.008)</td><td>0.314 (±0.008)</td><td>0.691 (±0.004)</td><td>0.989 (±0.002)</td><td>0.809 (±0.003)</td><td>+0.000 (±0.011)</td><td> $+ 0 . 7 5 7 \left( \pm 0 . 0 0 8 \right)$ </td><td> $+ 0 . 4 9 5 \left( \pm 0 . 0 0 8 \right)$ </td><td> $6 2 . 0 2 ( \pm 3 . 8 7 )$ </td></tr><tr><td>Gemma2-9B</td><td>0.682 (±0.010)</td><td>0.268 (±0.009)</td><td>0.349 (±0.009)</td><td>0.710 (±0.004)</td><td>0.986 (±0.002)</td><td>0.820 (±0.003)</td><td>+0.028 (±0.010)</td><td>+0.718 (±0.009)</td><td> $+ 0 . 4 7 0 \left( \pm 0 . 0 0 8 \right)$ </td><td> $5 4 . 4 1 \dot { ( \pm 3 . 3 9 ) }$ </td></tr><tr><td></td><td>Llama3.1-8B 0.624 (±0.011)</td><td>0.266 (±0.008)</td><td>0.340 (±0.008)</td><td> $0 . 6 7 2 \left( \pm 0 . 0 0 4 \right)$ </td><td>0.954 (±0.003)</td><td>0.783 (±0.003)</td><td>+0.048 (±0.011)</td><td> $+ 0 . 6 8 8 \left( \pm 0 . 0 0 8 \right) + 0 . 4 4 3 \left( \pm 0 . 0 0 7 \right)$ </td><td></td><td> $5 7 . 5 5 \left( \pm 3 . 5 9 \right)$ </td></tr></table>

Values are reported as mean followed by the half-width of the two-sided 95% t-confidence interval in parentheses, computed across evaluated HowNet sememes. Precision, recall and F1 quantify the set match between each sememe and either its best individual representational atom or the selected atom combination. Diferences (∆) are computed for each sememe as the atom-combination value minus the corresponding best-single-atom value before calculating the mean and confidence interval. Mean # atoms denotes the mean number of atoms selected for compositional recovery.  
Extended Data Table 3 Semantic recovery of single atom and atom combinations across models.

Extended Data Table 4 Evaluation of BiJumpReLU SAE representations on HowNet across models.
<table><tr><td>Model</td><td>Utilization</td><td>Retention</td><td> $\mathbf { F i d e l i t y }$ </td><td> $\mathbf { S t a b i l i t y }$ </td></tr><tr><td> $\mathrm { G e m m a 2 – 2 B }$ </td><td> $8 9 . 6 0 \left( \pm 3 . 9 4 \right)$ </td><td> $9 9 . 9 8 \left( \pm 0 . 0 1 \right)$ </td><td> $\mathbf { 9 9 . 6 2 \ ( \pm 0 . 1 0 ) }$ </td><td> $\mathbf { 9 9 . 8 2 \ ( \pm 0 . 0 4 ) }$ </td></tr><tr><td> $\mathrm { G e m m a 2 – 9 B }$ </td><td> $9 3 . 4 5 \ : ( \pm 2 . 7 7 )$ </td><td> $9 9 . 9 9 \left( \pm 0 . 0 0 \right)$ </td><td> $\mathbf { 9 9 . 6 4 } \left( \pm \mathbf { 0 . 0 8 } \right)$ </td><td> $\mathbf { 9 9 . 9 1 } \left( \pm \mathbf { 0 . 0 2 } \right)$ </td></tr><tr><td> $\mathrm { L l a m a 3 . 1 – 8 B }$ </td><td> $8 6 . 5 9 \left( \pm 1 0 . 4 8 \right)$ </td><td> $9 9 . 9 8 \left( \pm 0 . 0 1 \right)$ </td><td> $\mathbf { 9 9 . 3 0 } \left( \pm \mathbf { 0 . 4 8 } \right)$ </td><td> $\mathbf { 9 9 . 8 9 } \left( \pm \mathbf { 0 . 0 5 } \right)$ </td></tr></table>

Values are reported as the mean followed by the half-width of the two-sided 95% t-confidence interval in parentheses, computed across layers. All values are scaled by 100, and bold values denote the two key evaluation criteria for representational atoms: fidelity and stability. Utilization, retention, fidelity and stability are defined as in the preceding evaluation.

Extended Data Table 5 Case studies of sememe recovery by atom combinations.
<table><tr><td>Sememe</td><td>Model</td><td># atoms</td><td>|S|/|A|</td><td>P/R/F1</td><td>Difference</td><td>Size Marked</td><td></td><td>Entities</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>A\ S</td><td>36</td><td>24</td><td>Afar; BYD; Bhutan; Hongqiao; Huangbeiling; Jiaozhou; Kabukicho; Keshan; Lujiazui; Lukou Street; Male; Mount Songshan; Nankang; O; Panyu; Pingshan; Potala; Sanba Square; Shengzhou; Tongzhou;</td></tr><tr><td>DISTRICT</td><td>Gemma2-2B 49</td><td></td><td></td><td>1076/11100.968/0.998/0.983</td><td></td><td></td><td></td><td>Yalta; analytics; attention; call on a hotline; city of Beijing; coleorhizae; counterspy; couple with big age gap; friend; hardtack; newly added district; protease; regional; schoolhouse; small island; urban district</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>S\A</td><td>2</td><td></td><td>1 &quot;&quot;; district League committee 0</td></tr><tr><td></td><td>DISTRICT Gemma2-9B 28</td><td></td><td></td><td>1076/10501.000/0.976/0.988</td><td>A\S S\A</td><td></td><td></td><td>&quot;&quot;; Baoan; Beitou; Bund; Honghuagang; Mentougou; Nangang; Neihu; Potomac; Qingshan; Tongan; Wanhua; Wujin; Xiangzhou; Xindu; Zengcheng; area; area code; borough; canton; district League</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>26</td><td></td><td>committee; district and county; districts of Beijing; region; troublespot; whole district Aoshan; Baboshan; Bank of Chaoyang; Chinese chequers; Columbia; Fengcheng; Fujin; Funing; Hangjinqi; Hindu; Hongqiao; Jiading; Kabukicho; Liangzhou; Linzhi; Maigaoqiao; Melanogrammus</td></tr><tr><td></td><td>DISTRICT Llama3.1-8B 50</td><td></td><td></td><td>1076/1114 0.963/0.997/0.980</td><td>A\S</td><td>41</td><td>27</td><td>aeglefinus; Nankang; National Westminster; Peking; Quanjude; Red Square; Sanba Square; Tongzhou; Xingtai; Yuwen; and then; be uncertain; city and county; city of Beijing; fluorocarbon; footballer; force-fed duck; frontage house; itinerant doctor; meteorology; object clause; phantasy;</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>S\ A</td><td></td><td></td><td>racecourse; subarea; utter a threat &quot;&quot;; district; region</td></tr><tr><td>MALAWI</td><td>Gemma2-2B</td><td>2</td><td>8/9</td><td>0.889/1.000/0.941</td><td>A\S</td><td></td><td></td><td>Kwacha</td></tr><tr><td>MALAWI</td><td></td><td></td><td></td><td></td><td>S\A A\S</td><td></td><td></td><td>Kwacha</td></tr><tr><td></td><td>Gemma2-9B 2</td><td></td><td>8/9</td><td>0.889/1.000/0.941</td><td>S\A</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>Heshun; bagel; be up to date; call the roll in the morning; cryptanalysis; do not speak; early-wake; evening peak; get out of bed; get up early; just dawning; lunch; matutinal; night job; quick-meal stall;</td></tr><tr><td></td><td>MORNING Gemma2-2B 40</td><td></td><td>113/133</td><td>0.850/1.000/0.919</td><td>A\S</td><td></td><td>20</td><td>rosy cloud; sunset; try to catch a train; whitish colour of a fish&#x27;s belly-grey dawn; work night and day</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>S\A</td><td></td><td>0</td><td>brilliance of the rising sun; elevenses; evening peak; evening running; evening show; get up early;</td></tr><tr><td></td><td>MORNING Gemma2-9B 34</td><td></td><td>113/132</td><td>0.856/1.000/0.922</td><td>A\S</td><td></td><td>19</td><td>5 hypnone; matutinal; on the morning of; p.m.; period of the day from 9 a.m to 11 a.m; post meridiem; quick-lunch; quick-meal stall; signal flare; teatime; time of day; tourist coach; work night and day 0</td></tr><tr><td>CAPITAL</td><td></td><td></td><td></td><td></td><td>S\ A A\S</td><td></td><td>0</td><td>0</td></tr><tr><td></td><td></td><td>Gemma2-9B 126</td><td>619/609</td><td>1.000/0.984/0.992</td><td>S\A</td><td></td><td>0 10</td><td>Ancient Rome; BERIN; Bata; Beijing, Shanghai, Guangzhou and Shenzhen; Chang&#x27;an; Nine-Gates 4 Provincial Military Commander; Victoria; cockneyfy; guard the capital and the imperial palace; move</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>the capital to another place A/H1N1; Bartonella; CTA; Heilbronn; ISO; PAHs; animalcula; announcer; archesporia; basidia; bluetongue; business zone; campaign for president; causal agent; cicadae; clostridia; dicker; distiller&#x27;s yeast; echinococcus; enterohemorrhagic; erysipelas; free port; game theory; hardtack liquor made with</td></tr><tr><td></td><td>BACTERIA Gemma2-2B 109</td><td></td><td>371/419</td><td>0.883/0.997/0.937</td><td>A\S</td><td></td><td>49</td><td>5 a kind of yeast; herpes; it be alleged; liquid; lutetium; medroxalol; microbiology; moneran; natatorium; neonicotinoid; pogrom; protophyte; pullulate; snare; sol; state funeral; sterility; subcommission; subfunction; sulfadiazine; sulfamethazine; toleration; toxin; tubercle; waste gas or steam; zyme carrier</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>S\A</td><td></td><td>1</td><td>1 April fool; Boxing Day; Chinese New Year; Christmas Day; Christmas Eve; Christmas eve; Eid al-Fitr; Hallowe&#x27;en; Halloween; Happy Chinese New Year; Happy Valentine&#x27;s Day; Lord&#x27;s Supper; Lunar New Year; Muharram; September 11th; Vesper; all afternoon; all evening; all morning; another day; bank holiday; be up drinking all night; become more prosperous every day; bicennaries; bimillenary; bimillennium; boxing day; call the roll in the morning; change with each passing day; day show;</td></tr><tr><td>DAY</td><td></td><td></td><td>761/787</td><td>0.879/0.909/0.894</td><td>A\ S</td><td></td><td></td><td>day-by-day; early-wake; elevenses; every Friday; every Monday; every Saturday; every Sunday; every Thursday; every Tuesday; every Wednesday; express good wishes; film festival; film week; first month of the lunar year; for years on end; get up early; glow; good day; group attending a celebration; have a 45 day off; in a couple of days; in a few days; in one day; in one morning or evening; in these days; informal dinner; intercalary month in the lunar calendar; lie awake all night; lodge for the night; market day; matinee; matutinal; new calendar; night flight; night school; nightbird; of the previous night; one morning; overslept; period of the day from 1 a.m. to 3 a.m.; period of the day from 3 a.m. to 5 a.m.;</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>period of the day from 9 a.m to 11 a.m; pigeonhole; rays of morning or evening sunlight; same evening; second month of the lunar year; setting sun; sleep all afternoon; sleep all morning; solar calendar; stay a few days; stay up the whole night; that very evening; that very noon; today and yesterday; travel by night; warming sunlight; wed; weeknight; what day is tomorrow; what&#x27;s the date tomorrow; which day; whitish colour of a fish&#x27;s belly-grey dawn; within the next few days; working lunch Dec, 12th, 1936: Frost&#x27;s descent: Grain Full: Grain Rain: Grain in Ear: Islam Corban: at the end of the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>journal; daily produce; daily salt intake; date-changing phase; dateless; dateline; datemark; dawning; day; day of rest for factory workers; day of retribution; day off for staff as specified by the factory; dies faustus; endorse; eve; evening class; eventide; expected date of childbirth; forenone; get up in the night 7 to urinate; grow with the passing yearss; holiday; journal; logbook; lunar calendar used by the Zang</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>nationality; lunch meeting; luncheon meeting; menology; midday meal; millenary; millennium; misdate; monkey jacket; morning coach; morning train; nightly; nyctalopia; on the wane; pass the night; per</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>diem pay; red-eye; rosy clouds; seasonal division point; shell jacket; sky grows dark; solar flare;</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>statutory holiday; three days from now; tiffin; travel day and night; tux; tuxedo; twenty-four hours;</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>undated; vigil; white night</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FEEL</td><td></td><td></td><td></td><td></td><td>A\S</td><td></td><td>4</td><td>dermal sensation; pictural; somatosense; suddenly feel</td></tr><tr><td></td><td>Gemma2-2B 5</td><td></td><td>9/13</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>0.692/1.000/0.818</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>S\A</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>A\S</td><td></td><td></td></tr><tr><td></td><td></td></table>

S denotes the HowNet entity set annotated with the target sememe, and A denotes the entity set recovered by the selected atom combination. P/R/F1 denote precision, recall and F1 for the set match between A and S. For A \ S, bold entities were manually judged to be clearly consistent with the target sememe despite being absent from HowNet, indicating potential missing annotations. For S \ A, bold entities were manually judged to be clearly inconsistent with the target sememe despite being included in HowNet, indicating potential incorrect annotations. Marked gives the number of such manually identified entities in each diference set. "" denotes an empty entity string. Only clear cases were marked during manual inspection.

## Supplementary Information

Proofs   
Proof of Corollary 3   
Proof of Theorem 8 .   
Proof of Theorem 9 .   
Proof of Corollary 10 .   
Proof of Theorem 11 4   
Empirical validation of the atomic inner product 6   
Monosemanticity evaluation 6   
Semantic alignment evaluation 11   
Single-atom semantic alignment evaluation . 11   
Cross-layer compositional semantic alignment evaluation 13

## Proofs

## Proof of Theorem 2

Theorem 2 (Explicit form of the atomic inner product) Let $\langle { \pmb d } _ { i } , { \pmb d } _ { j } \rangle _ { S } = { \pmb d } _ { i } ^ { \top } { \pmb S } { \pmb d } _ { j }$ be an atomic inner product with $\pmb { S } \in \mathbb { R } ^ { H \times H }$ symmetric and positive definite. If the columns of $D = [ d _ { 1 } , d _ { 2 } , \cdot \cdot \cdot , d _ { | D | } ]$ form the atom set such that $\forall i , \ \lVert d _ { i } \rVert _ { S } = c > 0 ,$ , and $\mathcal { D } \simeq \mathbb { R } ^ { H }$ , then $S = c ^ { 2 } ( D D ^ { \top } ) ^ { - 1 }$

Proof Since $\langle \cdot , \cdot \rangle _ { S }$ is atomic inner product, for any pair of atoms $\mathbf { \ b { d } } _ { i }$ and $d _ { j }$ we have

$$
\pmb { d } _ { i } ^ { \top } S \pmb { d } _ { j } = \langle \pmb { d } _ { i } , \pmb { d } _ { j } \rangle _ { S } = \left\{ 0 \begin{array} { l l } { { \mathrm { \ } } i \neq j , } \\ { { \mathrm { \ } } c ^ { 2 } } & { { \mathrm { \ } i = j . } } \end{array} \right.\tag{1}
$$

Applying this property to the atom set $D = [ d _ { 1 } , d _ { 2 } , \cdot \cdot \cdot , d _ { | D | } ]$ yields

$$
c ^ { 2 } I = D ^ { \top } S D .\tag{2}
$$

Since $\{ d _ { 1 } , d _ { 2 } , \dotsb , d _ { | D | } \}$ spans the atomic space $\mathcal { D } ,$ and $\mathcal { D } \simeq \mathbb { R } ^ { H }$ , it follows that ran $\mathfrak { c } ( D ) = H$ . Let $S ^ { 1 / 2 }$ denote the symmetric positive-definite square root of S. Then

$$
D ^ { \top } S D = ( S ^ { 1 / 2 } D ) ^ { \top } ( S ^ { 1 / 2 } D ) ,\tag{3}
$$

which implies rank $: ( I ) = \operatorname { r a n k } ( D ^ { \top } S D ) = \operatorname { r a n k } ( S ^ { 1 / 2 } D ) = \operatorname { r a n k } ( D )$ . Therefore, $\mathrm { r a n k } ( D ) = | D | \leq H$ . Combining this with the earlier condition gives $| D | = \operatorname { r a n k } ( D ) = H$ , which shows that D is invertible. Consequently,

$$
S = c ^ { 2 } ( D D ^ { \top } ) ^ { - 1 } .\tag{4}
$$

□

## Proof of Corollary 3

Corollary 3 (Normalized atomic inner product; NAIP) Let the atomic inner product be defined by $\langle d _ { i } , d _ { j } \rangle _ { S } =$ $d _ { i } ^ { \top } S d _ { j } .$ , where S is symmetric and positive definite. Suppose the columns of $D = [ d _ { 1 } , \cdots , d _ { | D | } ]$ form the atom set satisfying ∀i, $\lVert d _ { i } \rVert _ { S } = c > 0$ . Then, for any $i , j$

$$
\langle \boldsymbol { d } _ { i } , \boldsymbol { d } _ { j } \rangle _ { \tilde { S } } : = \frac { \langle \boldsymbol { d } _ { i } , \boldsymbol { d } _ { j } \rangle _ { \boldsymbol { S } } } { \| \boldsymbol { d } _ { i } \| _ { \boldsymbol { S } } \| \boldsymbol { d } _ { j } \| _ { \boldsymbol { S } } } = \boldsymbol { d } _ { i } ^ { \top } \tilde { \boldsymbol { S } } \boldsymbol { d } _ { j } , \quad \tilde { \boldsymbol { S } } = ( \boldsymbol { D } \boldsymbol { D } ^ { \top } ) ^ { - 1 } .\tag{5}
$$

Consequently, the bilinear form $\langle \pmb { d } _ { i } , \pmb { d } _ { j } \rangle _ { \tilde { S } } = \pmb { d } _ { i } ^ { \top } \tilde { S } \pmb { d } _ { j }$ defines the normalized atomic inner product.

Proof Since $\langle d _ { i } , d _ { j } \rangle _ { S } = 0$ for $i \neq j$ , and $\| \pmb { d } _ { i } \| _ { S } = \| \pmb { d } _ { j } \| _ { S } = c > 0$ , it follows that $D ^ { \top } S D = c ^ { 2 } I$ . Therefore, $\begin{array} { r } { \tilde { S } = \frac { 1 } { r ^ { 2 } } S } \end{array}$ satisfies $D ^ { \top } \tilde { S } D = I$ , which implies that the atoms are orthonormal. Since D is invertible, we also have $\tilde { S } = \breve { D } ^ { - \top } D ^ { - 1 } = ( D D ^ { \top } ) ^ { - 1 }$ , and thus $\langle \cdot , \cdot \rangle _ { \tilde { S } }$ is a symmetric positive-definite inner product. □

## Proof of Theorem 8

Theorem 8 (Coherence–RIP Upper Bound) Let $\tilde { D } \in \mathbb { R } ^ { H \times | D | }$ and define the coherence $\mu { : = \operatorname* { m a x } _ { i \neq j } | \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | } \leq \varepsilon$ For any K-sparse vector $\delta \in \mathbb { R } ^ { | D | }$ with $\| \delta \| _ { 0 } \le K$

$$
\begin{array} { r } { \left( 1 - ( K - 1 ) \mu \right) \| \pmb { \delta } \| _ { 2 } ^ { 2 } \leq \| \tilde { \pmb { D } } \pmb { \delta } \| _ { 2 } ^ { 2 } \leq \left( 1 + ( K - 1 ) \mu \right) \| \pmb { \delta } \| _ { 2 } ^ { 2 } . } \end{array}\tag{8}
$$

Hence $\delta _ { K } ( \tilde { D } ) \leq ( K - 1 ) \mu$ . In particular, D<sup>˜</sup> satisfies the $K { \cdot } R I P$ whenever $( K - 1 ) \mu < 1$

Proof Let supp $\mathbf { \chi } ^ { \prime } \delta ) = { \mathcal { S } } \subseteq [ | D | ]$ and $| S | \le K$ . Then, we have

$$
\| \tilde { D } \delta \| _ { 2 } ^ { 2 } = \delta ^ { \top } ( \tilde { D } ^ { \top } \tilde { D } ) \delta = \sum _ { i \in \cal { S } } \delta _ { i } ^ { 2 } + 2 \sum _ { i < j \atop i , j \in \cal { S } } \delta _ { i } \delta _ { j } \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle .\tag{5}
$$

By the fact that $| \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | \leq \mu$ and applying the triangle inequality, we obtain:

$$
\| \tilde { D } \delta \| _ { 2 } ^ { 2 } \geq \sum _ { i \in \cal S } \delta _ { i } ^ { 2 } - 2 \mu \sum _ { i < j } | \delta _ { i } \delta _ { j } | ,\tag{6}
$$

$$
\| \tilde { D } \delta \| _ { 2 } ^ { 2 } \leq \sum _ { i \in \cal S } \delta _ { i } ^ { 2 } + 2 \mu \sum _ { i < j } | \delta _ { i } \delta _ { j } | .\tag{7}
$$

Next, we observe that:

$$
\left( \sum _ { i \in \mathcal { S } } \left. \delta _ { i } \right. \right) ^ { 2 } = \sum _ { i \in \mathcal { S } } \delta _ { i } ^ { 2 } + 2 \sum _ { i < j } \left. \delta _ { i } \delta _ { j } \right. \leq \left. \mathcal { S } \right. \sum _ { i \in \mathcal { S } } \delta _ { i } ^ { 2 } \leq K \sum _ { i \in \mathcal { S } } \delta _ { i } ^ { 2 } .\tag{8}
$$

Thus, we have $\begin{array} { r } { 2 \sum _ { i < j } | \delta _ { i } \delta _ { j } | \le ( K - 1 ) \sum _ { i \in { \mathcal { S } } } \delta _ { i } ^ { 2 } } \end{array}$ . Substituting this back, we conclude the proof.

□

## Proof of Theorem 9

Theorem 9 (Uniqueness and Exact $\ell _ { 1 }$ Recoverability) Let $\tilde { D } \in \mathbb { R } ^ { H \times | D | }$ and define the coherence $\mu : =$ ma $\begin{array} { r } { \mathrm { x } _ { i \neq j } | \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | \leq \varepsilon . \ I f \mu < \frac { 1 } { 2 K - 1 } } \end{array}$ , then for every $\boldsymbol \delta \in \mathbb { R } ^ { | D | }$ with $\| \delta \| _ { 0 } \le K$ , the K-sparse representation determined by $\tilde { m } = \tilde { D } \delta$ is unique; that is, no other K-sparse vector yields the same m˜ . Moreover, δ is the unique minimizer of the convex program

$$
\operatorname* { m i n } _ { \substack { \mathbf { x } \in \mathbb { R } ^ { | D | } } } \| \mathbf { x } \| _ { 1 } \quad s u b j e c t \ t o \quad \tilde { D } \mathbf { x } = \tilde { m } .\tag{9}
$$

Proof We first prove that under the condition $\mu < \frac { 1 } { 2 K - 1 }$ , the K-sparse representation is unique.

Suppose there exist two distinct K-sparse coeficient vectors $\delta , \delta ^ { \prime }$ such that $\tilde { D } \delta = \tilde { D } \delta ^ { \prime }$ . Let $\pmb { h } = \pmb { \delta } - \pmb { \delta } ^ { \prime } \neq \mathbf { 0 }$ Then $\tilde { D } h = 0$ and $\| h \| _ { 0 } \leq 2 K$ . By Theorem 8 (applied with K replaced by 2K), we have

$$
\big ( 1 - ( 2 K - 1 ) \mu \big ) \| h \| _ { 2 } ^ { 2 } \ \leq \ \| \tilde { D } h \| _ { 2 } ^ { 2 } \ = \ 0 .\tag{9}
$$

If $\textstyle \mu < { \frac { 1 } { 2 K - 1 } }$ , then the prefactor on the left is strictly positive, which forces $\| h \| _ { 2 } = 0$ . This contradicts $\mathbf { \nabla } _ { h } \neq \mathbf { 0 }$ Hence uniqueness holds.

Next, we prove that under the same condition $\textstyle \mu < { \frac { 1 } { 2 K - 1 } }$ , the sparse vector δ is also the unique solution of the convex optimization problem

$$
\operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { | D | } } \| \pmb { x } \| _ { 1 } \quad \mathrm { s . t . } \quad \tilde { D } \pmb { x } = \tilde { m } .\tag{10}
$$

The overall strategy is as follows: (i) show that the null space property of order $K ~ \left( \mathrm { N S P } _ { K } \right)$ holds under the assumption $\textstyle \mu < { \frac { 1 } { 2 K - 1 } } ; ( \operatorname { i i } )$ recall the equivalence $\mathrm { N S P } _ { K } \iff$ exact and unique recovery of any K-sparse solution via noiseless ℓ -minimization.

Formally, the null space property of order $K \ ( \mathrm { N S P } _ { K } )$ is defined as

$$
\forall h \in \ker ( \tilde { D } ) \setminus \{ \mathbf { 0 } \} , \ \forall \mathcal { S } \subseteq [ n ] , \ | \mathcal { S } | \leq K : \ \| h _ { \mathcal { S } } \| _ { 1 } < \| h _ { \mathcal { S } ^ { c } } \| _ { 1 } ,\tag{11}
$$

where $\ker ( \tilde { D } ) = \{ h : \tilde { D } h = \mathbf { 0 } \} , \mathcal { S } \subseteq [ n ]$ is an index set with $[ n ] = \{ 1 , \dots , n \} , h _ { \cal S }$ denotes the restriction of h to the coordinates in S (with other entries set to zero), and $S ^ { c } = [ n ] \setminus S$ is a complementary set.

## Step (i): Proof of NSP<sub>K</sub>

Let $G = \tilde { D } ^ { \top } \tilde { D }$ denote the Gram matrix. Since each column of $\tilde { D }$ is normalized, we have $G _ { j j } = 1$ and $| G _ { i j } | \le \mu$ for i ̸= j. Take any $\pmb { { h } } \in \ker ( \tilde { D } ) \setminus \{ \mathbf { 0 } \}$ and any index set S with $| S | = K$

Since $\tilde { D } h = 0$ , we have $G h = 0$ . For any j,

$$
0 = ( G h ) _ { j } = \sum _ { i } G _ { j i } h _ { i } = G _ { j j } h _ { j } + \sum _ { i \ne j } G _ { j i } h _ { i } \quad \Rightarrow \quad h _ { j } = - \sum _ { i \ne j } G _ { j i } h _ { i } .\tag{12}
$$

Taking absolute values and using $| G _ { j i } | \leq \mu _ { : }$ , we obtain

$$
| h _ { j } | \leq \mu \sum _ { i \neq j } | h _ { i } | .\tag{13}
$$

Summing over $j \in \mathcal S$ gives

$$
\sum _ { j \in S } | h _ { j } | \ \leq \ \mu \sum _ { j \in S } \sum _ { i \neq j } | h _ { i } | .\tag{14}
$$

The inner summation can be decomposed into contributions from $i \in \mathcal { S } \setminus \{ j \}$ and $i \in S ^ { c . } $

$$
\sum _ { j \in S } \sum _ { i \neq j } | h _ { i } | = \begin{array} { l l l l l l l } & { \displaystyle \sum _ { j \in S } \sum _ { i \in S } | h _ { i } | } & { \qquad + } & { \underbrace { \sum _ { j \in S } \sum _ { i \in S ^ { c } } | h _ { i } | } } \end{array}\tag{15}
$$

each i∈S<sup>c</sup> counted K times each i∈S counted K−1 times

Hence,

$$
\begin{array} { r } { \| \pmb { h } _ { S } \| _ { 1 } \ \le \ \mu \big ( ( K - 1 ) \| | \pmb { h } _ { S } | | _ { 1 } + K | | \pmb { h } _ { S ^ { c } } | | _ { 1 } \big ) . } \end{array}\tag{16}
$$

Rearranging,

$$
\big ( 1 - ( K - 1 ) \mu \big ) \| h _ { S } \| _ { 1 } \ \leq \ K \mu \| h _ { S ^ { c } } \| _ { 1 } .\tag{17}
$$

Dividing through by the positive factor $1 - ( K - 1 ) \mu ,$ define

$$
\alpha : = \frac { K \mu } { 1 - ( K - 1 ) \mu } .\tag{18}
$$

When $\mu < \frac { 1 } { 2 K - 1 }$ , we have $\alpha < 1$ . Since $h \neq 0$ and ${ \tilde { D } } h = 0 ,$ it is impossible for $\| h _ { S ^ { c } } \| _ { 1 } = 0$ (otherwise both terms would vanish, forcing $h = 0 .$ , a contradiction). Therefore,

$$
\| h _ { S } \| _ { 1 } \ \leq \ \alpha \| h _ { S ^ { c } } \| _ { 1 } \ < \ \| h _ { S ^ { c } } \| _ { 1 } .\tag{19}
$$

Take any $\scriptstyle { S _ { 0 } }$ with $| S _ { 0 } | = k \le K$ , and extend it to a superset $\quad s \supseteq s _ { 0 }$ such that |S| = K. Then,

$$
\| h _ { S _ { 0 } } \| _ { 1 } ~ \le ~ \| h _ { S } \| _ { 1 } , \qquad \| h _ { S _ { 0 } ^ { c } } \| _ { 1 } ~ \ge ~ \| h _ { S ^ { c } } \| _ { 1 } .\tag{20}
$$

If we know that $\| h _ { S } \| _ { 1 } < \| h _ { S ^ { c } } \| _ { 2 }$ <sub>1</sub> holds for all S of size K, then it follows that

$$
\| h _ { S _ { 0 } } \| _ { 1 } ~ \le ~ \| h _ { S } \| _ { 1 } ~ < ~ \| h _ { S ^ { c } } \| _ { 1 } ~ \le ~ \| h _ { S _ { 0 } ^ { c } } \| _ { 1 } .\tag{21}
$$

Thus, the inequality also holds for any $\scriptstyle { S _ { 0 } }$ with $| S _ { 0 } | \le K$ , which establishes $\mathrm { N S P } _ { K }$

Step (ii): Equivalence between $N S P _ { K }$ and $\ell _ { 1 }$ ecovery.

$N S P _ { K } \Rightarrow$ unique $\ell _ { 1 }$ recovery:

Suppose xˆ is another feasible solution such that $\tilde { D } \hat { \pmb x } = \tilde { D } \delta$ . Let $\pmb { h } = \hat { \pmb { x } } - \pmb { \delta } \in \ker ( \tilde { D } ) \setminus \{ \pmb { 0 } \}$ , and let $\mathcal { S } = \mathrm { s u p p } ( \delta )$ with $| S | \le K$ . Then

$$
\| \hat { x } \| _ { 1 } = \| \delta + h \| _ { 1 } = \| \delta _ { S } + h _ { \mathcal { S } } \| _ { 1 } + \| h _ { \mathcal { S } ^ { c } } \| _ { 1 } \ge \| \delta _ { S } \| _ { 1 } - \| h _ { \mathcal { S } } \| _ { 1 } + \| h _ { \mathcal { S } ^ { c } } \| _ { 1 } > \| \delta _ { S } \| _ { 1 } = \| \delta \| _ { 1 } ,\tag{22}
$$

where the strict inequality follows from $\mathrm { N S P } _ { K }$ . Hence, δ is the unique minimizer of the $\ell _ { 1 }$ problem.

Unique $\ell _ { 1 }$ recovery $\Rightarrow N S P _ { K }$

We argue by contradiction. If $\mathrm { N S P } _ { K }$ does not hold, then there exists $\pmb { h } \in \ker ( \tilde { D } ) \setminus \{ \mathbf { 0 } \}$ and some $s$ with $| S | \le K$ such that $\| h _ { S } \| _ { 1 } \geq \| h _ { S ^ { c } } \| _ { 1 }$ . Take any nonzero K-sparse δ with $\operatorname { s u p p } ( \delta ) = S _ { \mathrm { : } }$ , and choose $\delta _ { j } = \alpha _ { j } \mathrm { s g n } ( h _ { j } )$ with $\alpha _ { j } \geq | h _ { j } |$ coordinate-wise. Consider $\hat { \pmb { x } } = \delta - \pmb { h } .$ . Since ${ \tilde { D } } h = \mathbf { 0 }$ , both δ and xˆ are feasible, and

$$
\| \hat { x } \| _ { 1 } = \| \delta _ { S } - h _ { S } \| _ { 1 } + \| h _ { S ^ { c } } \| _ { 1 } = \| \delta _ { S } \| _ { 1 } - \| h _ { S } \| _ { 1 } + \| h _ { S ^ { c } } \| _ { 1 } \ \leq \ \| \delta _ { S } \| _ { 1 } = \| \delta \| _ { 1 } .\tag{23}
$$

Thus, δ is not the unique minimizer of the $\ell _ { 1 }$ problem (and may even fail to be a minimizer). This contradicts the uniqueness assumption. Therefore, $\mathrm { N S P } _ { K }$ must hold. □

## Proof of Corollary 10

Corollary 10 (Monorepresentationality / Injectivity) Under the condition $\mu < \frac { 1 } { 2 K - 1 }$ of Theorem $^ { g } ,$ define $\Sigma _ { K } : = \{ \pmb \delta \in \mathbb { R } ^ { | D | } : \| \pmb \delta \| _ { 0 } \leq K \}$ and Φ : $: \Sigma _ { K }  \mathbb { R } ^ { H } \ b y \ \Phi ( \delta ) : = \tilde { D } \delta$ . Then Φ is injective on $\Sigma _ { K }$ . That $i s ,$ for any $\pmb { x } , \pmb { y } \in \Sigma _ { K } , \ i f \ \tilde { D } \pmb { x } = \tilde { D } \pmb { y }$ , it follows that $\mathbf { \mu } _ { \mathbf { \mathcal { X } } } = \mathbf { \mu } _ { \mathbf { \mathcal { Y } } }$

Proof Take arbitrary $\pmb { x } , \pmb { y } \in \Sigma _ { K }$ such that $\tilde { D } \pmb { x } = \tilde { D } \pmb { y }$ . Let $\tilde { m } : = \tilde { D } x$ , then clearly $\tilde { m } = \tilde { D } y$ , with $\| { \pmb x } \| _ { 0 } \le K$ and $\| \pmb { y } \| _ { 0 } \le K$

By Theorem $^ { 9 , }$ under the condition $\mu < \frac { 1 } { 2 K - 1 }$ , the K-sparse representation δ determined by the equation $\tilde { m } = \tilde { D } \delta$ is unique: among all coeficient vectors satisfying $\| \delta \| _ { 0 } \le K$ , there exists only one that generates $\tilde { m } .$

Since both x and y are feasible solutions satisfying $\| \delta \| _ { 0 } \le K$ and $\tilde { D } \delta = \tilde { m }$ , uniqueness forces $\mathbf { \boldsymbol { x } } = \mathbf { \boldsymbol { y } }$ . Hence, Φ is injective on $\Sigma _ { K }$ □

## Proof of Theorem 11

Theorem 11 (Identifiability of Threshold-activated SAEs; TSAEs) Let $M = \{ m _ { i } \} _ { i = 1 } ^ { \lvert M \rvert } \subset \mathbb { R } ^ { H }$ with $m _ { i } = D \delta _ { i }$ where $\boldsymbol { D } = [ \boldsymbol { d } _ { 1 } , \therefore \cdot \cdot , \boldsymbol { d } _ { | D | } ] \in \mathbb { R } ^ { H \times | D | }$ satisfies $| \langle \tilde { d } _ { i } , \tilde { d } _ { j } \rangle | ~ \leq ~ \epsilon , ~ \forall i ~ \neq ~ j ,$ , and each $\pmb { \delta } _ { i } ~ \in ~ \mathbb { R } ^ { | D | }$ is K-sparse, i.e. $\lVert \delta _ { i } \rVert _ { 0 } \leq K$ . Define the positive and negative supports as $S _ { i } ^ { + } : = \{ j : \delta _ { i j } > 0 \} , \ : \ : S _ { i } ^ { - } : = \{ j : \delta _ { i j } < 0 \}$ , with $S _ { i } = S _ { i } ^ { + } \cup S _ { i } ^ { - } = \mathrm { s u p p } ( \delta _ { i } )$ . Suppose there exist constants $0 ~ < ~ \delta _ { \operatorname* { m i n } } ^ { + } ~ \le ~ \delta _ { \operatorname* { m a x } } ^ { + } , ~ 0 ~ < ~ \delta _ { \operatorname* { m i n } } ^ { - } ~ \le ~ \delta _ { \operatorname* { m a x } } ^ { - }$ , such that $\delta _ { \operatorname* { m i n } } ^ { + } \leq \delta _ { i j } \leq \delta _ { \operatorname* { m a x } } ^ { + } , \forall j \in S _ { i } ^ { + }$ , and $- \delta _ { \mathrm { m a x } } ^ { - } \leq \delta _ { i j } \leq - \delta _ { \mathrm { m i n } } ^ { - } , \forall j \in S _ { i } ^ { - }$ . Consider the bidirectional threshold activation function

$$
\sigma _ { \tau _ { - } , \tau _ { + } } ( x ) = \left\{ \begin{array} { l l } { { 0 , } } & { { \tau _ { - } \leq x \leq \tau _ { + } , } } \\ { { x , } } & { { x < \tau _ { - } \ o r \ x > \tau _ { + } , } } \end{array} \right.\tag{10}
$$

with $\tau _ { - } < 0 < \tau _ { + }$ . Define $B _ { 0 } : = \epsilon \left( | S _ { i } ^ { + } | \delta _ { \operatorname* { m a x } } ^ { + } + | S _ { i } ^ { - } | \delta _ { \operatorname* { m a x } } ^ { - } \right) , B _ { + } : = \epsilon \left( ( | S _ { i } ^ { + } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { + } + | S _ { i } ^ { - } | \delta _ { \operatorname* { m a x } } ^ { - } \right)$ , and $B _ { - } : =$ $\epsilon \left( | S _ { i } ^ { + } | \delta _ { \operatorname* { m a x } } ^ { + } + ( | S _ { i } ^ { - } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { - } \right)$ . If the bidirectional thresholds sa $t i s f y - \delta _ { \operatorname* { m i n } } ^ { - } + B _ { - } < \tau _ { - } \leq - B _ { 0 } , B _ { 0 } \leq \tau _ { + } <$ $\delta _ { \operatorname* { m i n } } ^ { + } - B _ { + }$ , with these intervals being non-empty whenever $\delta _ { \mathrm { m i n } } ^ { - } > B _ { 0 } + B _ { - } , \delta _ { \mathrm { m i n } } ^ { + } > B _ { 0 } + B _ { + }$ , then setting the TSAE with $W _ { d e c } = D$ and $\boldsymbol { W _ { \mathrm { e n c } } } = \boldsymbol { D } ^ { \top } \boldsymbol { \tilde { S } }$ yields

$$
\forall i , \quad W _ { \mathrm { d e c } } \sigma _ { \tau _ { - } , \tau _ { + } } \left( W _ { \mathrm { e n c } } m _ { i } \right) = m _ { i } .\tag{11}
$$

This parameterization exactly recover the atom set $D$

Proof Let $\boldsymbol { G } : = \boldsymbol { D } ^ { \intercal } \tilde { \boldsymbol { S } } \boldsymbol { D }$ . By the definition of the normalized atomic inner product, $G _ { k k } = 1$ , and for every $k \neq j ,$ $| G _ { k j } | = | \langle d _ { k } , d _ { j } \rangle _ { \widetilde { S } } | \leq \epsilon$ . Since $m _ { i } = D \delta _ { i }$ , we have

Consider the k-th coordinate.

$$
\begin{array} { r } { W _ { \mathrm { e n c } } m _ { i } = D ^ { \top } \widetilde { \mathbf { S } } D \delta _ { i } = G \delta _ { i } . } \end{array}\tag{24}
$$

For $\textit { k } \notin \mathcal { S } _ { i }$

$$
( G \delta _ { i } ) _ { k } = \sum _ { j \in { \cal S } _ { i } ^ { + } } \delta _ { i j } G _ { k j } + \sum _ { j \in { \cal S } _ { i } ^ { - } } \delta _ { i j } G _ { k j } .\tag{25}
$$

By the coherence bound,

$$
\begin{array} { r } { | ( G \delta _ { i } ) _ { k } | \leq \epsilon \left( | S _ { i } ^ { + } | \delta _ { \operatorname* { m a x } } ^ { + } + | S _ { i } ^ { - } | \delta _ { \operatorname* { m a x } } ^ { - } \right) = B _ { 0 } . } \end{array}\tag{26}
$$

Hence

$$
\begin{array} { r } { - B 0 \le ( G \delta _ { i } ) _ { k } \le B _ { 0 } . } \end{array}\tag{27}
$$

Because $\tau _ { - } \le - B _ { 0 }$ and $\tau _ { + } \geq B _ { 0 }$ , it follows that

$$
\begin{array} { r } { \tau _ { - } \leq ( G \delta _ { i } ) _ { k } \leq \tau _ { + } , } \end{array}\tag{28}
$$

and this coordinate is therefore set to zero by the bidirectional threshold activation.

For $k \in S _ { i } ^ { + }$

$$
( G \delta _ { i } ) _ { k } = \delta _ { i k } + \sum _ { j \in S _ { i } \backslash \{ k \} } \delta _ { i j } G _ { k j } .\tag{29}
$$

Define

$$
e _ { i k } ^ { + } : = \sum _ { j \in S _ { i } \backslash \{ k \} } \delta _ { i j } G _ { k j } .\tag{30}
$$

Again using the coherence bound,

$$
\begin{array} { r } { | e _ { i k } ^ { + } | \le \epsilon \left[ ( | S _ { i } ^ { + } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { + } + | S _ { i } ^ { - } | \delta _ { \operatorname* { m a x } } ^ { - } \right] = B _ { + } . } \end{array}\tag{31}
$$

Therefore,

$$
( G \delta _ { i } ) _ { k } \geq \delta _ { \operatorname* { m i n } } ^ { + } - B _ { + } .\tag{32}
$$

$$
\mathrm { I f ~ } \tau _ { + } < \delta _ { \mathrm { m i n } } ^ { + } - B _ { + , \mathrm { ~ t h e n } }
$$

$$
( G \delta _ { i } ) _ { k } > \tau _ { + } .\tag{33}
$$

Thus, the positive active coordinate is retained and its sign remains positive.

For $k \in S _ { i } ^ { - }$

$$
( G \delta _ { i } ) _ { k } = \delta _ { i k } + \sum _ { j \in S _ { i } \backslash \left\{ k \right\} } \delta _ { i j } G _ { k j } = \delta _ { i k } + e _ { i k } ^ { - } .\tag{34}
$$

The coherence bound gives

$$
| e _ { i k } ^ { - } | \le \epsilon \left[ | S _ { i } ^ { + } | \delta _ { \operatorname* { m a x } } ^ { + } + ( | S _ { i } ^ { - } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { - } \right] = B _ { - } .\tag{35}
$$

Since $\delta _ { i k } \leq - \delta _ { \operatorname* { m i n } } ^ { - } ,$ we obtain

$$
( G \delta _ { i } ) _ { k } \leq - \delta _ { \operatorname* { m i n } } ^ { - } + B _ { - } .\tag{36}
$$

$$
\mathrm { I f } - \delta _ { \mathrm { m i n } } ^ { - } + B _ { - } < \tau _ { - } , \mathrm { t h e n }
$$

$$
( G \delta _ { i } ) _ { k } < \tau _ { - } .\tag{37}
$$

Hence, the negative active coordinate is retained and its sign remains negative.

Combining the three cases, all coordinates outside the support lie within $[ \tau _ { - } , \tau _ { + } ]$ and are set to zero, whereas positive and negative support coordinates lie above $\tau _ { + }$ and below $\tau _ { - } ,$ respectively. Therefore,

$$
\mathrm { s u p p } \left( \sigma _ { \tau _ { - } , \tau _ { + } } ( G \delta _ { i } ) \right) = S _ { i } ,\tag{38}
$$

and the signs of the retained activations agree with those of the true coeficients.

Finally, for the bidirectional thresholds to exist, non-zero gaps must separate the inactive coordinates from both the positive and negative active coordinates. For the positive threshold, this requires

$$
B _ { 0 } < \delta _ { \mathrm { m i n } } ^ { + } - B _ { + } ,\tag{39}
$$

or equivalently,

$$
\delta _ { \operatorname* { m i n } } ^ { + } > B _ { 0 } + B _ { + } = \epsilon \left[ ( 2 | S _ { i } ^ { + } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { + } + 2 | S _ { i } ^ { - } | \delta _ { \operatorname* { m a x } } ^ { - } \right] .\tag{40}
$$

For the negative threshold, we require

$$
- \delta _ { \mathrm { m i n } } ^ { - } + B _ { - } < - B _ { 0 } ,\tag{41}
$$

which is equivalent to

$$
\delta _ { \operatorname* { m i n } } ^ { - } > B _ { 0 } + B _ { - } = \epsilon \left[ 2 | S _ { i } ^ { + } | \delta _ { \operatorname* { m a x } } ^ { + } + ( 2 | S _ { i } ^ { - } | - 1 ) \delta _ { \operatorname* { m a x } } ^ { - } \right] .\tag{42}
$$

Here, $B _ { 0 }$ bounds the maximum interference on inactive coordinates, whereas $B _ { + }$ and $B _ { - }$ bound the maximum cross-interference on positive and negative active coordinates, respectively. Therefore, the conditions above ensure that the minimum active magnitudes remain suficiently separated from the interference induced by the remaining atoms. Thus, the conclusion holds. □

## Empirical validation of the atomic inner product

To further assess whether the AIP provides a more appropriate characterization of neural representational geometry than the Euclidean inner product, we compare the distributions of pairwise angles under the two metrics. Specifically, we randomly sample 128 entities from Wikidata [48] and extract their internal representations across multiple layers of several language models. For each entity, the activation at its final token position is used as the corresponding representation [25], and pairwise angles between entity representations are computed independently at each layer.

We first consider the standard Euclidean inner product. For two representations $\boldsymbol { u } , \boldsymbol { v } \in \mathbb { R } ^ { H }$ , the Euclidean cosine similarity is defined as

$$
\cos ( \boldsymbol { \mathbf { u } } , \boldsymbol { \mathbf { v } } ) = \frac { \boldsymbol { \mathbf { u } } ^ { \top } \boldsymbol { \mathbf { v } } } { \| \boldsymbol { \mathbf { u } } \| _ { 2 } \| \boldsymbol { \mathbf { v } } \| _ { 2 } } ,\tag{43}
$$

which we convert to angles for analysis. Across models and layers, the resulting angular distributions consistently deviate from 90<sup>◦</sup> (Supplementary Fig. 1-3). We refer to this phenomenon as representation shift. This shift indicates that representations measured in Euclidean geometry are systematically influenced by a shared dominant direction, such that even semantically unrelated representations can exhibit elevated cosine similarity. Varying the number of samples used to estimate the angular distribution leaves its centroid largely unchanged (Supplementary Fig. 4), indicating that this efect is not a finite-sampling artifact but instead reflects a persistent global directional bias under the Euclidean metric.

We next recompute the geometry of the same representations using the AIP. For each model and layer, we independently collect 100K internal representations k from Wikipedia and estimate the representationinduced metric from their second-order statistics as $\widetilde { \pmb { S } } = \left( \mathbb { E } [ \pmb { k } \pmb { k } ^ { \top } ] \right) ^ { - 1 }$ . The normalized cosine similarity induced by this metric is then given by

$$
\cos _ { \widetilde { S } } ( \boldsymbol { u } , \boldsymbol { v } ) = \frac { \boldsymbol { u } ^ { \top } \widetilde { S } \boldsymbol { v } } { \sqrt { \boldsymbol { u } ^ { \top } \widetilde { S } \boldsymbol { u } \sqrt { \boldsymbol { v } ^ { \top } \widetilde { S } \boldsymbol { v } } } } .\tag{44}
$$

As above, we convert these similarities to angles and examine their empirical distributions.

Compared with the Euclidean inner product, the AIP substantially reduces the systematic angular shift, with the distributions across models and layers becoming centered close to $9 0 °$ (Supplementary Fig. 5-7). At the same time, the high-similarity tail remains dominated by semantically related representation pairs, indicating that the AIP reduces the global directional bias while preserving local geometric structure associated with specific relationships between samples. This pattern is consistent across model families and network layers.

These observations are particularly relevant to the distinguishability criterion used in our framework. Stability depends on the coherence between candidate representational units, which is determined by their relative geometry. Under the Euclidean metric afected by a shared directional bias, units without strong structural correspondence may nevertheless exhibit elevated similarity, thereby obscuring the coherence arising from the underlying representational structure. By contrast, the AIP corrects this global bias and captures the underlying geometry relevant to representational distinguishability. These results therefore provide empirical support for using the AIP, rather than the standard Euclidean inner product, in our theoretical analysis and stability evaluation.

## Monosemanticity evaluation

This section provides further details on the implementation of the monosemanticity evaluation described in Methods, including the criteria used to determine semantic coherence, the LLM-as-a-judge prompts and manual verification procedure for language models, and the manual evaluation criteria for difusion models. All evaluations follow the unified monosemanticity criterion defined in the main text: for the activation instances of a representational unit, we identify the largest subset $S _ { \mathrm { m a x } }$ that can be consistently explained by a common semantic interpretation, and compute its monosemanticity score as $\vert S _ { \mathrm { m a x } } \vert / \vert S \vert$

Criteria for semantic coherence. For a given representational unit, we identify the largest subset of instances that can be explained by a shared concept, attribute or semantic pattern with a clear and specific meaning. The shared interpretation must be suficiently semantically specific and cannot rely solely on an overly broad superordinate category. For example, the fact that multiple instances are all “entities”, “words” or “visual patterns” is not, by itself, suficient to constitute a valid shared interpretation.

![](images/b42220418fc780f6ff83d18abe635021d721315d5ffa78b6f239473ecb02185e.jpg)

![](images/9c1e883ebcbf4fa9bb1b263e0bb8c1132ea721bc1bb72247d7f6c844a46cc8d7.jpg)

![](images/d1b7255bf87651322ba9c7eab6988fab2764e3e81788f22d478ffeefdfa40f45.jpg)

![](images/44fec67a38ab187c3877d27a2a78c28df167f75c6886047ea3c58b141e36c0fd.jpg)

![](images/ca03f5f49a68adf6f213e1f4c6b58aff0b5d6e1c0559fd76f645c24cb6d26540.jpg)

![](images/a148bd2b3fbde55ac0c0eac8dd51d44dfc532e410e255ea7fd3075469642f08b.jpg)

![](images/958ec01eed5a9e37d34ec0222b12bd1690df74289fc130e4e8ca18517a8b29f6.jpg)

![](images/f7410972158afd67088a44745b9632a78b2a535b106de407897dc99ed8312fdb.jpg)

![](images/978b50296939f0978327cae3dce9d47ea96651a04b8050b30cc467b45e1316d2.jpg)

![](images/1454327ea6a17d2693db56a78605232cbdff20a07c39c956901a2c0e1ca93ffb.jpg)

![](images/29363b0bebc3ead737a8c4e6882395f257fe3221b6b69dd5024a182ab04b722f.jpg)

![](images/d76676a108f08fd7af05cdfb4947348332975295a9130899e87640c8d5ee2e5f.jpg)

![](images/37e55e7e29f75a22b809a148d9d5ef31745c0f0aa8f6768e24315ac64042aff3.jpg)

![](images/84c32ee86833650fe3af7f7603df804481bb65546e75385ea3342a1fe320a41c.jpg)

![](images/0eb1b9fdc82e79a3e1510bd70c68c8eac5e689254e0c7539fb524c478c1275ac.jpg)

![](images/83f586f76de5d67ee75e08a9219251b27888bfa7b210a17ee6373b2650400f1e.jpg)

![](images/84d01415e41e7a26560b4a8ac787854b54aa5ceb71869e419088841de9d18042.jpg)

![](images/2b07394b516eeead0d148225c00f908300fee9aeb92b45d207a65890cceb022e.jpg)  
Supplementary Fig. 1 Representation shift in GPT2-Large under the Euclidean inner product. Angular distributions between representations of 128 randomly sampled Wikidata entities across model layers, extracted at the final token position of each entity. Across layers, the distributions are systematically shifted away from 90<sup>◦</sup>, indicating a persistent global directional bias in the Euclidean representation geometry.

![](images/0427ff535338504193c3d8f19b0436f5e76d0b3ee3fbf49e890025258571bd41.jpg)

![](images/b0a0a6890feb4a1183a3dbf33f64a3f299522c8b8a669d0888a352c4fc361127.jpg)

![](images/9d71b9d7644fc696682cd5af378f745f65c6a34bd0f8aa87eac796ba9df940c5.jpg)

![](images/10cb82f9960c5da57baa9fb50f109b3821fb07b91ea9f054ab16f93d7897dad8.jpg)

![](images/2f5369f6e01052853fb67f813952a17484f510a4eb4ea0771fd72d2fc9d44aa7.jpg)

![](images/1afccf94f54b35e16d5cc7dd722b33a5700ee3347a2581783a58d3004fab21b0.jpg)

![](images/40ad242d04b0022601acc4e6ecc25caeaa2e3e27d6b2431c70580a8774ea6bb8.jpg)

![](images/e133f9700f062eb2da74dd2b7dcaae9e8409511c2a67f407ff540b3d137b68cb.jpg)

![](images/b61e621543fc88d0e4a29480a11201451d343c266d1b9a4b5f12c0109438b905.jpg)

![](images/8f8e6f88a1edd0b5876fadc367fd014d45b4fa8a058b56c8a95f2ef99deb2056.jpg)

![](images/9db3cd18d018432f1d26d95e315903d049bdfc9300fe2a6ead7c3a143bdf4373.jpg)

![](images/cf8dadc4322049e76a4f481d77733396d80e1d14a86e0b7cd4e927a2b9fdee39.jpg)

![](images/58ca9b99f521879112e262edae481846be43b0ea7eed0645f96e1bc0d39041db.jpg)

![](images/c72909b8372c4cb8de4f9fc0d9198d984f982b2cca726396a8159b399165ec13.jpg)

![](images/924fcfcd29fe4c75ed9fa40ae7898f8a7f909ddee59d4ebcb75f3f839a6362c1.jpg)

![](images/7494e4299f09794c30ada6ec1bb70692dc14b6303e1cd536a096f2736cd7bec6.jpg)

![](images/c2e956028fe9e81186e550b798aafe7b8fc1d82f1adf12505e924cbe615aab03.jpg)

![](images/97ca101fc34ebb92fe8bd98dc36812c7a944abf9680779f3caa74fc239ae2532.jpg)  
Supplementary Fig. 2 Representation shift in Pythia-6.9B under the Euclidean inner product. Angular distributions between representations of 128 randomly sampled Wikidata entities across model layers, extracted at the final token position of each entity. Across layers, the distributions are systematically shifted away from 90<sup>◦</sup>, indicating a persistent global directional bias in the Euclidean representation geometry.

Likewise, weak topical relatedness, indirect association or incidental co-occurrence is insuficient to group instances into the same subset.

Instances may difer substantially in their surface form, provided that they share the same underlying semantic concept, attribute or function in the relevant context. For a candidate interpretation, only instances that are directly and consistently accounted for by that interpretation are included in $S _ { \mathrm { m a x } } ;$ an instance is not included merely because it bears a weak relation to the other instances. If multiple plausible semantic interpretations exist, we select the interpretation that consistently accounts for the largest number of instances, and define the corresponding set of instances as $S _ { \mathrm { m a x } }$

LLM-as-a-judge evaluation for language models. For language models, we evaluate Wikipedia and Wikidata representations separately. For Wikipedia, each evaluation instance corresponds to a target

![](images/5a58038b98052c088e6ca17df5f6ae0319dba042898a98ee0d77ba82d77a7dc6.jpg)

![](images/579d1d8c5846be5c86366351bc2e11c71cda45e6d6a7eaec0e6cf2fd58a512ab.jpg)

![](images/bb7b79d7657addae0a87880d90da467dd0c73af59729c00f8120f1ee1421bb92.jpg)

![](images/e8bcc5c3db3e218d27d3aea6aebf403b84463c3d847ed4adb1926cfb087c948e.jpg)

![](images/0ae6c18f1964e7f5f826ba24842ee003108def47eb491ffc7e8268d684c46530.jpg)

![](images/beb786d1bfcefb6eb140d93fe4f96e12bccb7fc166817e48707bd8781a65f407.jpg)

![](images/f2bbbd5db8d6806ed78a3cb900fba077bdf9a5e1a184c7932c7794bbbfc31a12.jpg)

![](images/6408662c7e30e7a70b6a5f719260d2b59d78064c371fc23c39312d1ca8384769.jpg)

![](images/1473fe73ba95d823b0d629b24f66f8483bd5edda48849ac148dbfaad0b07deef.jpg)

![](images/1688eb63b1e114859024f665ce551f8ec15e597daa1ae493e6c169c66084c927.jpg)

![](images/2d04fb141582426d29d72b104db3c745e83df081e4e8944dbd221f3b78dbe3ef.jpg)

![](images/899d0e25bc348526e2d190b771f2919d54506a172365051f81b067816392b7af.jpg)

![](images/9a7f300595860940378781da21c75b73fac4f08f9d935574357c1c44a314ed90.jpg)

![](images/0e4e728dd730283524ea64256d8ffeb87681d2cd8e66f286a2d5ae4089896a48.jpg)

![](images/fe6835ee86e192056c7735f1649fa72dfdb679c2d4bb0aa09025d5817eae39bd.jpg)

![](images/29561e292145e3bc7e7b0f76aceee80fd717b54eba7964fa4b0c2098f8176790.jpg)

![](images/69f2642b19dc8e0fcff447b25214f82c0ba3562ccf0437722180d9e58b27fa49.jpg)

![](images/a4ffc68400e483dc690f6b75e7a31228fe7ece125a758283f49112d2cba338f4.jpg)

Supplementary Fig. 3 Representation shift in Llama3.1-8B under the Euclidean inner product. Angular distributions between representations of 128 randomly sampled Wikidata entities across model layers, extracted at the final token position of each entity. Across layers, the distributions are systematically shifted away from 90<sup>◦</sup>, indicating a persistent global directional bias in the Euclidean representation geometry.

![](images/c7d1b6f40f683e358a6537a7ef8bddc02edc61b2bb23cb498ad0e880a46ce2db.jpg)  
(a) Sample size 128.

![](images/d262a6169af47892bc21533865365d601b73679bc37cfb8cfb3528aed6abc144.jpg)  
(b) Sample size 256.

![](images/ba05c60a6e925e4a34f3debf36d7dc44da02488b57ceeb0a2c6418c81aec6fed.jpg)  
(c) Sample size 512.

![](images/1ad24fc5573f74de4c294f8957fff6a3dde4183fce4555945e818c9343cf0198.jpg)  
(d) Sample size 1024.

![](images/e5a0127df88a177c5fbb56a539814f7833b3d4a3c43ada28c9bfde03b58a476e.jpg)  
(e) Sample size 2048.

![](images/afb46f29f06d7ccb260c691071dc54185d7630eb67c075e391e2b1348e66e279.jpg)  
(f) Sample size 4096.

Supplementary Fig. 4 Ablation of sample size for representation shift in GPT2-Large. Angular distributions under the Euclidean inner product estimated using diferent numbers of randomly sampled Wikidata entities. The distribution centroid remains largely unchanged as the sample size increases, indicating that the observed representation shift is not attributable to finite-sample estimation.

![](images/3144aa552f363dcd0c0e8b8a0c8f6be21250927a186f2e00774f07ab8f609331.jpg)

![](images/6225d3e21c966f112b40232dfcfb7cb3807d6e088d05b05948fea6c18c5140cb.jpg)

![](images/6b41fba19b209e1e19bbf38871772ce61241ab09dc9dbfc4ffa86c847da9ec0a.jpg)

![](images/ad91502e7216de26a36803a0c46cd53e345eea375153313a406ab8278651fa16.jpg)

![](images/1c44005329691936b943141b1a8d7595aa53b59f720825a75edd30ccd3680926.jpg)

![](images/6db782dd40f40c0aad64428d76360cd23032702bbbdffe677ccb56674ad0e6a2.jpg)

![](images/cd87506b562087f85d18b4166d131a4a99d6ee0a07a8e7f9faa744240829eef9.jpg)

![](images/8289be1778ee58ebaeff376e07d8071957695a2d3e545f2ee3dd740315491206.jpg)

![](images/72f5cddbe3b352a7f3c72f3ca648c52b6b1827e5ec165955955898eff4fa9aad.jpg)

![](images/ebd6cafa799a355342e3fa9d4c0e0e5aa6f116b4994c1670e60af39033d407e2.jpg)

![](images/e18038ca36be59e4ce65421605f5d88dfdfc0297c4cddbbf526791b390f9158d.jpg)

![](images/601d92fcfe4af022a8541fa75f2fdb122ff15155090b4f93230e352bd9097a8d.jpg)

![](images/4dfb2dc87087f7fdebda09808d83d62968fa7834218cdb9d11ef5e3af90fca3d.jpg)

![](images/75b06d8d1e24d7c86a351e6326d88e1a17bfc5fa791908c85c2fb95c2951ea83.jpg)

![](images/e40a0e13d61366fac7c89e812dbd7558ffc35bde29eed6997b7c6ccd0f18661f.jpg)

![](images/12468e77a101ccb17de739817036030f08a4f475ef7abd23c77cce9100e0a9d2.jpg)

![](images/f5769aaf4233ad8c093b5094efd8249b9c5b449ff18789dee9a04a6af415a783.jpg)

![](images/f418b631b1a76bbcfe2812eb9f0cac4f93dd4bbe0124d5cab8226cef9fd45aa6.jpg)

Supplementary Fig. 5 Representational geometry of GPT2-Large under the atomic inner product. Angular distributions between representations of 128 randomly sampled Wikidata entities across model layers, computed using the atomic inner product. Across layers, the distributions are centered close to 90<sup>◦</sup>, indicating that the AIP substantially reduces the global directional bias observed under the Euclidean inner product.

![](images/935569743d3c3c219c528824adddece0f79c2615ee693bfd87a95b14eef6a781.jpg)

![](images/0861bb4c1c161dc0ea76395698cf8608e7bf4f92f2ebe1521bf252deddc208d2.jpg)

![](images/a44aa4e56ac1d1cffae47dac2625d3a11178447c09ee47eb75f875ba64da445f.jpg)

![](images/010a928a431fdf2fc9837f59419171ba3a14ef4853da74c1080ea5592ac64deb.jpg)

![](images/963ff4042353053737196f063ef8ab730d7a7467d4a3c27e9167b8208186c0c8.jpg)

![](images/0ff070e996ba62fd598d4fe6e3bf5f6b9e3438bf53a48ed1fdbb5de11adb3b2a.jpg)

![](images/6682022b4f08265458f09c705281cff18a1b03690ab06d374f3e1ecfa9099358.jpg)

![](images/01ef0eca3acff2fffee4afc683a5fd0e2f09b587056226de699ad45a9aaa29d9.jpg)

![](images/bc109c7ab5057fb7f7858ff2f6eec4ff7585dd5b3f3af5b3f63fb12b5f647712.jpg)

![](images/b62cae5399c6a27c224d8bfec84da8baf20de197c66e6e50c108333e703b64d5.jpg)

![](images/889b8ca04a2db3e48971c4a9945ea20ab1242661173efb8d3f176b5f49682b67.jpg)

![](images/65bd5b25e6de16a31f90d41763be7c20f14f006a0e3723aec5bf419246901ee2.jpg)  
<sup>.</sup><sub>.</sub> <sup>.</sup><sub>.</sub>

![](images/e06eb700b392e8d7ef8b686092896fd117d9c267999b63ab5c84a66c0d041495.jpg)

![](images/f74ba9a9dd67874538a3ce1671a4ee7dd6ab2e7b2cbcb99ea1a3696ed8444279.jpg)

![](images/ee7a8938a1d82af4b14d42d9c11a7854837a1c64f7277abb0178e2c63229beee.jpg)

![](images/d0d407daa365b0335c61603c8da38768f744fcbd349b9de8220529e51ebfd2cd.jpg)

![](images/12c5421a35789f14c80782290b417924b58ec86a906ce740305b3ebbf1c2e063.jpg)

![](images/e95de3d5e72301b7a2bc3d9c8f4e92239710f3730d1fc3295c25b45306ac541c.jpg)

Supplementary Fig. 6 Representational geometry of Pythia-6.9B under the atomic inner product. Angular distributions between representations of 128 randomly sampled Wikidata entities across model layers, computed using the atomic inner product. Across layers, the distributions are centered close to 90<sup>◦</sup>, indicating that the AIP substantially reduces the global directional bias observed under the Euclidean inner product.  
![](images/f72f403cdeef92cd621dc15b1d263b361e429ab25db914b4d9eed65839327d33.jpg)

![](images/acf70580c860510f58f858383eec4037ec1cb270845db52926c81a3d2ed02082.jpg)

![](images/beacfd1198b23f2f7f8c391af8c4af8102d2799a278d03ff746c8dbf0364c23e.jpg)

![](images/ce4ade13efde2134dc2f9a7feab041325e575b82b5ffd36d212becd39650623f.jpg)

![](images/ec46c5bdc18ba3de48ddb5f993b6715acc396b6e925ce5c048e7f8fb7137c007.jpg)

![](images/3929007cbc7ef54282146eb21dab6826725f50bf6853c47d9f1cdc8a11496bf9.jpg)

![](images/66fcc92d6119e1ef11224bca71dca7fd6855c361db60f156561b4dd473537962.jpg)

![](images/b9fb5aba71153f320dce802b6ec1dcd295b7d7687ee94afc190ef4cdd13ebeda.jpg)

![](images/2cfa2ac838691bb214fe0a62ebb531d583a86e25d0b3be7b03dff3f93e848e18.jpg)

![](images/c59e175e2d2b1aac99d978d921fd6a754106276db3aaed7257c709d71404e887.jpg)

![](images/c37da58a60224dcff5bb2d9b74d5c789a84d279fd7a78fbc7503f3b098be712a.jpg)

![](images/f749f5965f4a6c6a8f962c107632be760692a2aa8675c35aa3907518576bd117.jpg)  
<sup>.</sup>.<sub>.</sub> <sup>.</sup><sub>.</sub>

![](images/59039931f9666cc63b01317f830ae7d70884c4b95ec8b1291d7a888287225beb.jpg)

![](images/6d22bf555b2d6fb49b1e0b05d832f62bd55dfdced818a596ad2c781dc6fd8489.jpg)

![](images/9c68dbe16a3afa901d0fa646aa65d8d0aa51e976a769bc2ecc854b596fab832e.jpg)

![](images/37ccae2372d3e8f81a35eb5ec380f8c0fc69a2fd68220e80574aa63f71c924a9.jpg)

![](images/a8ade66cfe217976f94259a773b61eedbc510fe29251a86d3ac580ad0a86c5ec.jpg)

![](images/4f3b6e7c2ffa28bb8d4b3cb62e7008e77d38cd1b350b9bcbc25058189e2e55e3.jpg)  
Supplementary Fig. 7 Representational geometry of Llama3.1-8B under the atomic inner product. Angular distributions between representations of 128 randomly sampled Wikidata entities across model layers, computed using the atomic inner product. Across layers, the distributions are centered close to 90<sup>◦</sup>, indicating that the AIP substantially reduces the global directional bias observed under the Euclidean inner product.

token that activates the representational unit. To avoid basing the evaluation solely on the lexical form of the target token, we provide the evaluation model with the local context surrounding the token and explicitly highlight the target token. The evaluation model then determines whether diferent instances share a coherent semantic pattern based on the meaning or semantic function of the target token in its specific context. For Wikidata, each instance corresponds to the name of an entity that activates the representational unit, and the evaluation model identifies the largest coherent subset based on the entities’ identities, types, attributes or semantic roles.

For each representational unit, the evaluation model is provided with all instances included in the evaluation and is asked to determine the largest subset that can be consistently explained by a common semantic concept, attribute or pattern.

For Wikipedia representations, we use the following prompt:

Below are {num examples} activation examples from the same SAE feature.   
Focus on the highlighted token [. . . ] in each context.   
Determine the largest number of examples that can reasonably be grouped into ONE coherent semantic   
category. Use a reasonably specific category based on the highlighted token’s meaning or semantic function.   
{examples text}   
Reply ONLY as M/{num examples}, where M is the size of the largest coherent semantic group. Do not   
explain.

For Wikidata representations, we use the following prompt:

Below are {num examples} activation examples from the same SAE feature.   
Each example corresponds to one Wikidata entity name.   
Determine the largest number of examples that can reasonably be grouped into ONE coherent semantic   
category. Use a reasonably specific category based on the entities’ identities, types, or semantic roles.   
{examples text}   
Reply ONLY as M/{num examples}, where M is the size of the largest coherent semantic group. Do not   
explain.

Manual verification of LLM-judge outputs. The LLM judge returns the size of the largest semantically coherent group of instances as $M / | S |$ , where M corresponds to $\lvert S _ { \mathrm { m a x } } \rvert$ . This output format directly provides the subset cardinality required to compute the monosemanticity score, while minimizing the influence of additional generated content on the evaluation.

To mitigate potential inconsistencies in automated judgements of semantic boundaries, overly broad categories or weak semantic associations, we further manually verify the scores produced by the LLM judge. Manual verification follows the same criteria as the automated evaluation. Specifically, the evaluator examines the same set of activation instances and determines the largest number of instances that can be consistently explained by a shared concept, attribute or semantic pattern with reasonable semantic specificity. Groups based only on overly broad categories, indirect semantic associations or weak topica relatedness are not considered semantically coherent.

When the value of M returned by the LLM judge is consistent with these criteria, the original score is retained. When the automated judgement and manual assessment disagree, M is revised according to the size of the largest semantically coherent group identified during manual verification. The verified value of M is then used as $\lvert S _ { \mathrm { m a x } } \rvert$ to compute the final monosemanticity score for the representational unit.

Manual evaluation for difusion models. For Stable Difusion v1.5 and FLUX.2-klein-9B, we apply the same largest-coherent-subset criterion as for language models. Because the evaluation instances correspond to local visual regions in LAION-COCO images, we perform the evaluation manually. For each representational unit, the evaluator jointly examines all local image instances included in the evaluation and identifies the largest subset that can be explained by a shared visual semantic concept, attribute or pattern. Valid shared interpretations may correspond to object categories, object parts, semantically meaningful visual attributes, or other local structures that retain a consistent semantic meaning across diferent images. Representative examples of this evaluation procedure for Stable Difusion v1.5 and FLUX.2-klein-9B are shown in Supplementary Figs. 8 and 9, respectively.

Instances that exhibit only low-level visual similarity without a shared semantic interpretation are not considered monosemantic matches. For example, if multiple local regions share similar colours, simple textures or other low-level statistical properties but cannot be assigned a consistent semantic interpretation, they are not grouped into the same subset on this basis alone. Conversely, a visual attribute can be treated as a shared pattern if it has a stable and interpretable semantic meaning across instances. Once $S _ { \mathrm { m a x } }$ is identified, the monosemanticity score is computed using the same definition as for language models.

a  
t = 0, Feature 16793  
![](images/426826c55fd7e07d2f3c9dd2ab1bf67bd2aea0d68aa79b76e82541d0f76cfff8.jpg)  
Supplementary Fig. 8 Representative monosemanticity evaluation examples for BiJumpReLU features in Stable Difusion v1.5. a–c, Activation instances for representative BiJumpReLU features extracted from the conditional branch at difusion timesteps $t = 0$ (a; Feature 16793), t = 500 (b; Feature 104755) and t = 1000 (c; Feature 72300). Red boxes indicate the local image regions corresponding to feature activations used in the monosemanticity evaluation. The selected features exhibit coherent visual-semantic responses across their activated instances, corresponding predominantly to arm in a, footwear in b, and wristwatch regions in c. Up to ten activation instances are shown for each feature; when fewer than ten instances are available, all available instances are included.

Sampling and aggregation details. Monosemanticity evaluation is performed independently for each model, data source and representation extraction site. For each setting, we randomly sample 10 candidate representational units learned by the corresponding SAE for evaluation. For each unit, we use at most 10 activation instances. If more than 10 activated instances are available, we randomly sample 10 from the full set of activated instances; if fewer than 10 are available, we use all available instances. Consequently, |S| may be smaller than 10 for some units. We then compute the monosemanticity score for each representational unit separately and average the scores of the sampled units within the same SAE setting to obtain the overall monosemanticity score for that setting.

## Semantic alignment evaluation

## Single-atom semantic alignment evaluation

To further detail the single-atom semantic alignment evaluation described in the main text, we represent both HowNet sememes and candidate representational units in the model as entity sets and quantify their correspondence based on set overlap. As described in the main text, we retain only HowNet sememes and candidate representational units that are associated with at least two unique entities.

For a given sememe s, let $E _ { s }$ denote its corresponding entity set. For a candidate representational unit $d _ { i } ^ { ( l ) }$ at layer l, let $E _ { i } ^ { ( l ) }$ denote the set of HowNet entities that activate that unit. We define the precision, recall and F1 score between the two sets as

$$
P \Big ( d _ { i } ^ { ( l ) } , s \Big ) = \frac { \Big | E _ { i } ^ { ( l ) } \cap E _ { s } \Big | } { \Big | E _ { i } ^ { ( l ) } \Big | } ,\tag{45}
$$

![](images/a5aae267ed62d6ebd60fbea7e5aed92c6dae4739b41483988cb61271ed1e6cae.jpg)

Step 0, Layer 5, Feature 106095

![](images/b8052ef00a8afea59156886141150170aacd314a0e1b3639f70727bac31a1302.jpg)

![](images/91c74f6b7a33a04b5f4f005c3fe7365c1f8babfdbc8b5bd3884a38c1eb8f1948.jpg)

![](images/6f50651a050cc9d803689adf2ad890491fc894fef8f281b40a4b5977c6053b51.jpg)

![](images/daa8eb76fe717f5dabbbd694b89214c322f70ce927e90d7a9fc816d333f26440.jpg)

![](images/5d3f44fbe7af77d4c6e13529bcab6ee1d59cfbd5c99e939f3a6cce53b656fd66.jpg)

![](images/db02b3612bd4c45ae27bf2c7f5cba7543be56c012b3e8a9eaefd0c0a3be6dddb.jpg)

![](images/656809a58e604d5699af4d276d0736f3ec0f4f403ea275336787bce53fdbbdc9.jpg)  
b

![](images/37cdc19c5c36b2b55f7d0b92f7ec4e2b34b4663143f518c930698b826450596b.jpg)

![](images/74e730c9a42ed9b7a5d5c9d2f613d862c0aacee6ea8bec5075802e40d94b289f.jpg)  
Step 0, Layer 15, Feature 35701

![](images/1564cef704bcf0009aacbf883902e5e1ae590d63208651bfc6e09994208bfbfb.jpg)

![](images/56c72e3187b9fabf43d7e75f20c89e87c8b9e010b6bb6ec4e2f9bc2d5ef25603.jpg)

![](images/591e82f690f560dd53d686c3b406ff88d0958520bd0836326682d9996e938292.jpg)

![](images/da96496d5634e241b046fb6a6a29c056be8d8e92e53ab2f9ed17d1a308fa5a34.jpg)

![](images/6d45b938252983122704fab035e2a4343aa1a6a195b5158a3a42dafda62a2367.jpg)

![](images/335072e35502769e159e950d85c53d4af882636fc2567999da3ede308c303072.jpg)

![](images/a949ed770d84c3edbb41759aa1f7aa8272cc1ae879f874573e219dfa2dce754b.jpg)

![](images/ae72bd61e234f182192438d24a7edbe6909a5c321f8acd5996f88872e79a33c4.jpg)

![](images/6655018476209b813b25e73bb33b7009d6acc790622ccdd4925d0e6d02f6e1c2.jpg)

![](images/3d7c52c7d2e3a8f46f8ac024f41ef618f7f7855f4c72e0db867d0203965e3148.jpg)  
Step 0, Layer 30, Feature 168925

c  
![](images/b1398f7f0526b6050791cc2a20a0a9b43e25d7a672cdd76fd968dcf34108ed04.jpg)

![](images/78e374e5f079102a0323290fd37a65a8df4cf6b7ffa4246d3e65e9f601b0bd85.jpg)

![](images/7f9e2e9365183a5d265bf5b9e75ef1574a48444a399e9e337c0612d19d7c900e.jpg)

![](images/6330e6938f812047fc8446b3ff63b76ee07be44cd36e5905061d1dfe6e63a67b.jpg)

![](images/3cbaf94c00eeb41abe549eaf7335d94566fdae0a6e0f9ff5623593b0a60f9ae7.jpg)

![](images/640cac186a0be843dd8de22378a61b7d3f2a058129e017c3d31254f5c4fa0ef7.jpg)

![](images/154f05b8b50efc70437260d8f16a5550845955690a7037ad2157e84f3a570fd1.jpg)

![](images/4abc839c8fa1a7ed97f05e22d656f76ba0fca5a88de7de09fef19b36f4826cb2.jpg)

![](images/afe2452d4fe9516230b114378b15982d56212fab014a817bf470835c9227972d.jpg)  
Supplementary Fig. 9 Representative monosemanticity evaluation examples for BiJumpReLU features in FLUX.2-klein-9B. a–c, Activation instances for representative BiJumpReLU features extracted at step 0 from Layer 5 (a; Feature 106095), Layer 15 (b; Feature 35701) and Layer 30 (c; Feature 168925). Red boxes indicate the local image regions corresponding to feature activations used in the monosemanticity evaluation. The selected features exhibit coherent visual-semantic responses across their activated instances, corresponding predominantly to multiple patches within the same semantically coherent region of a single image in a, raised arms or hands in b, and dog ear regions in c. Up to ten activation instances are shown for each feature; when fewer than ten instances are available, all available instances are included.

$$
R \Big ( d _ { i } ^ { ( l ) } , s \Big ) = \frac { \Big | E _ { i } ^ { ( l ) } \cap E _ { s } \Big | } { \big | E _ { s } \big | } ,\tag{46}
$$

and

$$
F _ { 1 } \left( d _ { i } ^ { ( l ) } , s \right) = \frac { 2 P \left( d _ { i } ^ { ( l ) } , s \right) R \left( d _ { i } ^ { ( l ) } , s \right) } { P \left( d _ { i } ^ { ( l ) } , s \right) + R \left( d _ { i } ^ { ( l ) } , s \right) } .\tag{47}
$$

When the two entity sets do not overlap, the F1 score is defined as 0. Precision measures the proportion of entities activating the representational unit that belong to the target sememe, and therefore reflects the semantic purity of the unit with respect to that sememe. Recall measures the extent to which the unit covers the entities associated with the target sememe, whereas F1 jointly accounts for both quantities.

For each sememe s, we search across all valid candidate representational units from all model layers and identify the unit with the highest F1 score:

$$
d _ { s } ^ { * } = \arg \operatorname* { m a x } _ { l , i } F _ { 1 } \left( d _ { i } ^ { ( l ) } , s \right) .\tag{48}
$$

The corresponding $F _ { 1 } ( d _ { s } ^ { * } , s )$ is used as the single-atom semantic alignment score for sememe s. For each sememe, we additionally record the model layer containing the best-matching unit, together with its precision, recall and F1 score. We also retain the intersection between the two entity sets, the entities covered only by the candidate unit, and the entities belonging only to the target sememe, enabling further analysis of the semantic purity and coverage of the single-atom match. If multiple candidate units attain the same maximum F1 score, the first unit encountered in the candidate ordering is retained. This choice does not afect the resulting single-atom semantic alignment score.

## Cross-layer compositional semantic alignment evaluation

To assess whether combinations of representational atoms can more completely recover the entity structure associated with HowNet sememes, we further perform a cross-layer compositional semantic alignment analysis. For each model, we aggregate candidate representational units that satisfy the filtering criteria described in the main text across all model layers, thereby forming a unified pool of candidate atoms.

For representational units exhibiting both positive and negative responses, we treat the entity sets associated with positive and negative activations as separate candidates, such that each candidate atom corresponds to a single, well-defined activation direction. For a target sememe s, let $E _ { s }$ denote its entity set, and let $E _ { i }$ denote the entity set associated with candidate atom $d _ { i }$

Candidate prefiltering. Because exhaustively enumerating all possible combinations of cross-layer candidate atoms is computationally infeasible, we first prefilter the candidate atoms independently for each target sememe. Specifically, we use the same definition of precision as in the single-atom analysis:

$$
P ( d _ { i } , s ) = { \frac { | E _ { i } \cap E _ { s } | } { | E _ { i } | } } .\tag{49}
$$

We retain only candidate atoms satisfying $P ( d _ { i } , s ) \geq \tau _ { P }$ and $| E _ { i } \cap E _ { s } | > 0$ , where $\tau _ { P }$ denotes the precision threshold used for candidate filtering. The retained candidates are then ranked, in order, by precision, F1 score and the size of their entity-set intersection with the target sememe, and at most the top $K _ { \mathrm { c a n d } }$ candidates are kept. We set $K _ { \mathrm { c a n d } } = 4 0 0 0$ for all experiments. This prefiltering step removes candidates that lack a basic entity-level correspondence with the target sememe, thereby reducing the search space for the subsequent compositional search.

Greedy compositional search. On the filtered candidate pool, we construct multi-atom combinations using a greedy set-union strategy. Let $A _ { t }$ denote the set of atoms selected after iteration t. The entity set associated with the combination is defined as the union of the entity sets of the selected candidate atoms:

$$
E ( A _ { t } ) = \bigcup _ { d _ { i } \in A _ { t } } E _ { i } .\tag{50}
$$

The precision, recall and F1 score between combination $A _ { t }$ and target sememe s are defined as

$$
P ( A _ { t } , s ) = { \frac { | E ( A _ { t } ) \cap E _ { s } | } { | E ( A _ { t } ) | } } ,\tag{51}
$$

$$
R ( A _ { t } , s ) = { \frac { | E ( A _ { t } ) \cap E _ { s } | } { | E _ { s } | } } ,\tag{52}
$$

and

$$
F _ { 1 } ( A _ { t } , s ) = \frac { 2 P ( A _ { t } , s ) R ( A _ { t } , s ) } { P ( A _ { t } , s ) + R ( A _ { t } , s ) } .\tag{53}
$$

The search starts from the empty combination, $A _ { 0 } = \varnothing$ . At each iteration, we evaluate the F1 score obtained by adding each unselected candidate atom to the current combination and select the candidate that yields the largest F1 improvement:

$$
d _ { t + 1 } ^ { * } = \arg \operatorname* { m a x } _ { d _ { i } \notin A _ { t } } \left[ F _ { 1 } ( A _ { t } \cup \{ d _ { i } \} , s ) - F _ { 1 } ( A _ { t } , s ) \right] .\tag{54}
$$

We denote the corresponding maximum improvement as

$$
\Delta _ { t + 1 } = F _ { 1 } \big ( A _ { t } \cup \{ d _ { t + 1 } ^ { * } \} , s \big ) - F _ { 1 } ( A _ { t } , s ) .\tag{55}
$$

The selected atom is added to the combination only when $\Delta _ { t + 1 } > 0 \colon$

$$
A _ { t + 1 } = A _ { t } \cup \{ d _ { t + 1 } ^ { * } \} .\tag{56}
$$

The search terminates when no remaining candidate atom can further improve the F1 score. We additionally impose a maximum combination size $N _ { \mathrm { m a x } }$ , and terminate the search when $| A _ { t } | ~ = ~ N _ { \operatorname* { m a x } }$

Consequently, the final number of atoms selected for diferent sememes is not fixed in advance, but is determined adaptively by the F1 improvements subject to the maximum combination-size constraint.

Selection of search parameters. The candidate-filtering threshold $\tau _ { P }$ and the maximum combination size $N _ { \mathrm { m a x } }$ jointly afect the candidate search space and the resulting compositional semantic alignment. We therefore systematically examine their efects over a predefined parameter grid.

Specifically, we first evaluate combinations of

$$
\tau _ { P } \in \{ 0 . 1 , \ 0 . 2 , \ 0 . 3 , \ 0 . 4 , \ 0 . 5 , \ 0 . 6 , \ 0 . 7 , \ 0 . 8 , \ 0 . 9 \}
$$

and

$$
N _ { \mathrm { m a x } } \in \{ 2 0 , ~ 4 0 , ~ 6 0 , ~ 8 0 , ~ 1 0 0 , ~ 1 2 0 , ~ 1 4 0 , ~ 1 6 0 , ~ 1 8 0 \} .
$$

For each parameter setting, overall compositional alignment performance is measured by the mean final F1 score across all valid HowNet sememes.

Supplementary Fig. 10–12 shows the results under diferent parameter settings. The initial grid search indicates that $\tau _ { P } = 0 . 5$ yields the best overall semantic alignment performance, while increasing $N _ { \mathrm { m a x } }$ continues to improve compositional alignment. Based on this result, we further fix $\tau _ { P } = 0 . 5$ and additionally evaluate

$$
N _ { \mathrm { m a x } } \in \{ 2 0 0 , ~ 4 0 0 , ~ 6 0 0 , ~ 8 0 0 \} .
$$

As shown in Supplementary Fig. 13, the overall alignment performance continues to improve as $N _ { \mathrm { m a x } }$ increases across all three models, while the gains become progressively smaller at larger combination sizes and the performance approaches a plateau by $N _ { \mathrm { m a x } } = 8 0 0$ . Accordingly, unless otherwise specified, all subsequent cross-layer compositional semantic alignment analyses use $\tau _ { P } = 0 . 5$ and $N _ { \mathrm { m a x } } = 8 0 0 .$

For each sememe, we record the final set of selected atoms and their source layers, the F1 improvement resulting from each successive atom addition, and the precision, recall and F1 score of the final combination. We also retain the entities covered jointly by the combination and the target sememe, the entities covered only by the combination, and the sememe entities that remain uncovered by the combination. These quantities are used for further comparisons between single-atom matches and multi-atom combinations in terms of semantic purity and coverage.

To improve the computational eficiency of large-scale cross-layer search, we represent the atom–entity and sememe–entity correspondences as sparse binary matrices and compute set overlaps and candidate statistics using batched sparse-matrix operations. This implementation is used solely for computationa eficiency and does not alter the set definitions or search criteria described above.

Gemma2-2B
<table><tr><td rowspan="2">180</td><td colspan="9"></td><td rowspan="2"></td><td rowspan="2">0.800</td></tr><tr><td>0.785</td><td>0.787</td><td>0.792</td><td>0.798</td><td>0.804</td><td>0.716</td><td>0.696</td><td>0.696</td><td>0.700</td></tr><tr><td>160</td><td>0.784</td><td>0.786</td><td>0.791</td><td>0.796</td><td>0.803</td><td>0.715</td><td>0.695</td><td>0.696</td><td>0.699</td><td></td><td>0.775 0.750</td></tr><tr><td>maxm n x xnmax 140</td><td>0.782</td><td>0.784</td><td>0.789</td><td>0.795</td><td>0.801</td><td>0.713</td><td>0.694</td><td>0.694</td><td>0.697</td><td></td><td rowspan="4">0.725 F1 Mean</td></tr><tr><td>120</td><td>0.780</td><td>0.782</td><td>0.787</td><td>0.792</td><td>0.799</td><td>0.712</td><td>0.693</td><td>0.693</td><td>0.695</td><td></td></tr><tr><td>100</td><td>0.776</td><td>0.779</td><td>0.783</td><td>0.789</td><td>0.795</td><td>0.710</td><td>0.690</td><td>0.690</td><td>0.692</td><td>0.700</td></tr><tr><td>80</td><td>0.771</td><td>0.773</td><td>0.778</td><td>0.784</td><td>0.790</td><td>0.706</td><td>0.687</td><td>0.686</td><td>0.688</td><td>0.675</td></tr><tr><td>60</td><td>0.762</td><td>0.764</td><td>0.769</td><td>0.775</td><td>0.781</td><td>0.700</td><td>0.681</td><td>0.679</td><td>0.680</td><td></td><td>0.650</td></tr><tr><td>40</td><td>0.745</td><td>0.747</td><td>0.752</td><td>0.757</td><td>0.763</td><td>0.686</td><td>0.668</td><td>0.665</td><td>0.663</td><td></td><td>0.625</td></tr><tr><td rowspan="2">20</td><td>0.698</td><td>0.700</td><td>0.705</td><td>0.709</td><td>0.713</td><td>0.649</td><td>0.630</td><td></td><td>0.624</td><td>0.618</td><td>0.600</td></tr><tr><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4 Precision threshold τp</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td></td><td></td></tr></table>

Supplementary Fig. 10 Grid search for compositional semantic alignment hyperparameters in Gemma2- 2B. Mean final F1 score across valid HowNet sememes under diferent combinations of the candidate-filtering precision threshold τ<sub>P</sub> and maximum combination size $N _ { \mathrm { m a x } }$ . Each cell reports the mean final F1 score obtained after compositional search under the corresponding parameter setting, with colour indicating the same quantity. Within the evaluated grid, τ = 0.5 consistently yields the highest overall alignment performance, while increasing N<sub>max</sub> progressively improves the final F1 score.

Gemma2-9B
<table><tr><td rowspan="2">180</td><td colspan="11"></td><td rowspan="2">0.800</td></tr><tr><td>0.799</td><td>0.799</td><td>0.802</td><td>0.808</td><td>0.816</td><td>0.751</td><td>0.739</td><td>0.743</td><td>0.749</td><td></td></tr><tr><td>160</td><td>0.798</td><td>0.798</td><td>0.801</td><td>0.807</td><td>0.814</td><td>0.750</td><td>0.738</td><td>0.742</td><td>0.748</td><td></td><td>0.775 0.750</td></tr><tr><td>140</td><td>0.796</td><td>0.797</td><td>0.799</td><td>0.805</td><td>0.813</td><td>0.749</td><td>0.736</td><td>0.741</td><td>0.746</td><td></td><td>F</td></tr><tr><td>120</td><td>0.794</td><td>0.794</td><td>0.797</td><td>0.803</td><td>0.811</td><td>0.747</td><td>0.735</td><td>0.739</td><td>0.744</td><td></td><td>0.725</td></tr><tr><td>am n oxin xmn 100</td><td>0.791</td><td>0.791</td><td>0.794</td><td>0.800</td><td>0.808</td><td>0.745</td><td>0.732</td><td>0.736</td><td>0.741</td><td></td><td>Mean 0.700 0.675</td></tr><tr><td>80</td><td>0.786</td><td>0.787</td><td>0.789</td><td>0.795</td><td>0.803</td><td>0.741</td><td>0.728</td><td>0.732</td><td>0.736</td><td></td><td></td></tr><tr><td>60</td><td>0.778</td><td>0.779</td><td>0.781</td><td>0.787</td><td>0.795</td><td>0.735</td><td>0.722</td><td>0.725</td><td>0.728</td><td></td><td>0.650</td></tr><tr><td>40</td><td>0.762</td><td>0.762</td><td>0.765</td><td>0.771</td><td>0.779</td><td>0.722</td><td>0.708</td><td>0.710</td><td>0.710</td><td></td><td>0.625</td></tr><tr><td>20</td><td>0.719</td><td>0.719</td><td>0.722</td><td>0.727</td><td>0.734</td><td>0.684</td><td>0.670</td><td>0.668</td><td>0.662</td><td></td><td></td></tr><tr><td></td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4 Precision threshold τp</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td></td><td>0.600</td></tr></table>

Supplementary Fig. 11 Grid search for compositional semantic alignment hyperparameters in Gemma2-9B. Mean final F1 score across valid HowNet sememes under diferent combinations of the candidate-filtering precision threshold τ and maximum combination size $N _ { \mathrm { m a x } } .$ . Each cell reports the mean final F1 score obtained after greedy compositional search under the corresponding parameter setting, with colour indicating the same quantity. Within the evaluated grid, τ = 0.5 consistently yields the highest overall alignment performance, while increasing $N _ { \mathrm { m a x } }$ progressively improves the final F1 score.

![](images/4e39fdf66706885f7820c240495ed1f069d0232931f933b3309bdbbad8122afc.jpg)  
Supplementary Fig. 12 Grid search for compositional semantic alignment hyperparameters in Llama3.1-8B. Mean final F1 score across valid HowNet sememes under diferent combinations of the candidate-filtering precision threshold τ and maximum combination size $N _ { \mathrm { m a x } } .$ . Each cell reports the mean final F1 score obtained after greedy compositional search under the corresponding parameter setting, with colour indicating the same quantity. Within the evaluated grid, $\tau _ { P } = 0 . 5$ consistently yields the highest overall alignment performance, while increasing $N _ { \mathrm { m a x } }$ progressively improves the final F1 score.

![](images/cfbdf492ed489d75c69c9704bfde2ec34877e2ef723c9f043874da80157ca9ec.jpg)  
Supplementary Fig. 13 Efect of the maximum combination size on cross-layer compositional semantic alignment. Mean final F1 score across valid HowNet sememes as a function of the maximum combination size $N _ { \mathrm { m a x } }$ for Gemma2-2B, Gemma2-9B and Llama3.1-8B, with the candidate-filtering precision threshold fixed at $\tau _ { P } = 0 . 5$ . For all three models, increasing $N _ { \mathrm { m a x } }$ progressively improves compositional semantic alignment, with the gains becoming substantially smaller at larger combination sizes and the performance approaching a plateau by $N _ { \mathrm { m a x } } = 8 0 0$ . These results support the use of $N _ { \mathrm { m a x } } = 8 0 0$ in the subsequent cross-layer compositional semantic alignment analyses.