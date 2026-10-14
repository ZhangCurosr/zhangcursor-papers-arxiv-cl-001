# PROBING THE GEOMETRY OF COLOR REPRESENTATIONS IN MULTILINGUAL LANGUAGE MODELS

Pablo Loyola Rakuten Group, Inc. pablo.a.loyola@rakuten.com

Edison Marrese-Taylor

Graduate School of Engineering, The University of Tokyo, Japan emarrese@weblab.t.u-tokyo.ac.jp

Andrés Hoyos-Idrobo Rakuten Group, Inc. andres.hoyosidrobo@rakuten.com

## ABSTRACT

Do multilingual language models learn a universal internal geometry for color, and is it perceptually grounded or primarily linguistic? We decompose color representations into two geometric axes: (P1) alignment to perceptual color space, and (P2) cross-lingual symmetry. Analyzing layerwise embeddings across models and multilingual datasets, we find that perceptual grounding is weak: alignment to CIE–LAB peaks in intermediate layers, then collapses toward output layers. Cross-lingual symmetry degrades with depth, suggesting later layers encode language-specific structure. Functionally, we show that geometric properties predict task performance: models achieve 96.7% accuracy on color classification at layers where P1 alignment peaks, with P1 alignment (r = 0.70) significantly correlating with downstream capabilities. Our results reveal a dissociation: text-only LMs learn universal color-language geometry, not universal color-perception geometry, providing diagnostics for where perceptual structure is preserved in language models.

Keywords perceptual grounding · multilingual language models · color representations

## 1 Introduction

Do Language Models (LMs) build latent geometric structures that systematically organize meaning? Recent work suggests neural representations are best viewed as geometric objects—manifolds or trajectories in activation space. In language models, latent factors trace continuous curves in hidden states [Dohmatob et al., 2023, Engels et al., 2025], while in vision, object manifolds become more compact across layers [Ansuini et al., 2019, Cohen et al., 2020]. Competing hypotheses have emerged: the Linear Representation Hypothesis [Park et al., 2024] posits that concepts correspond to linear directions, while the Platonic Representation Hypothe sis [Huh et al., 2024] argues that representations converge toward shared structure across architectures. Yet how these geometries align with real-world perceptual structure remains unclear.

We use color as a concrete testbed. It has well-studied perceptual geometry (CIE–LAB) as a proxy for human similarity, and multilingual datasets exposing systematic cross-linguistic variation [Winawer et al., 2007, Gibson et al., 2017]. This makes it ideal for disentangling perceptual structure from linguistic categorization, and for testing whether LMs learn compact, perceptually grounded geometry shared across languages.

We ask: What geometric structure do multilingual language models learn for color—one aligned with perceptual space (CIE–LAB), language-driven categorical geometry, or a mix that shifts across layers? This question exposes two geometric axes: (P1) perceptual alignment—does similarity geometry match CIE–LAB, and at which layers? (P2) cross-lingual symmetry—is geometry shared across languages or does it drift with linguistic variation?

Our approach is pragmatic: we do not expect text-only LMs to reproduce human perception. Instead, we use LAB as a template to measure to what extent color structure is preserved or altered at each layer. We analyze layerwise representations across models and multilingual datasets, assessing kernel alignment with CIE–LAB and measuring cross-lingual drift.

We find a clear dissociation: text-only LMs learn a compact color-language geometry that is partly universal across languages, yet perceptual (LAB) alignment is weak and, when present, peaks in intermediate layers before collapsing at output. Cross-lingual congruence is highest in early layers and degrades with depth. Functionally, these geometric properties predict downstream task performance, with models achieving 96.7% accuracy on color classification at layers where alignment peaks.

Contributions. We introduce internal color geometry as a testbed for analyzing how multilingual LMs organize perceptual-like domains, develop a layerwise framework combining perceptual kernel alignment and cross-lingual symmetry analysis, and provide evidence that LMs learn stable color-language geometry without strong perceptual grounding. This reveals that universality can emerge without perceptual alignment, with functional implications for layer selection in downstream applications.

## 1.1 Problem Formulation: Internal Color Geometry

We analyze the geometry of color meaning in multilingual LMs. Given colors with multilingual descriptions, we ask what geometric patterns the model expresses and how they relate to perceptual color space (CIE–LAB) and cross-lingual invariance. We use perceptual color spaces as reference geometries—not because text-only models should match human vision, but to measure what geometry language supervision produces and where it aligns with perception.

We focus on two geometric properties of layerwise representations. First, (P1) perceptual alignment: does the model’s similarity geometry match perceptual space, and at which layers? We test whether neighborhoods or kernels from the model distance matrix match those from CIE–LAB, up to global distortions. Second, (P2) cross-lingual symmetry: given colors described in multiple languages, is geometry stable or does linguistic form distort pairwise structure? We quantify this by comparing language-specific kernels on matched samples.

## 2 Related Work

Geometric methods for analyzing representations. Representations in deep networks often lie on low-dimensional manifolds [Ansuini et al., 2019, Cohen et al., 2020, Modell et al., 2025], compared via RSA [Kriegeskorte et al., 2008], CKA [Kornblith et al., 2019], and kernel methods [Gretton et al., 2005]. We extend this toolkit to a concrete grounded domain, i.e., color, where ground-truth perceptual structure (CIE–LAB) exists, enabling direct tests of whether learned geometry reflects perception or language-specific categorization. Crucially, we show that low dimensionality need not imply perceptual grounding.

Representation structure in language models. The Platonic Representation Hypothesis [Huh et al., 2024] posits convergence to shared representations across models and modalities, while work on multilingual models shows cross-lingual similarity strongest in early layers, diverging with depth [Conneau et al., 2017, Libovicky et al.\` , 2020, Chi et al., 2020]. Layerwise analyses also show that intermediate layers often carry more useful representations than the final one [Skean et al., 2025], consistent with the perceptual window we find for color. We provide the first systematic analysis decomposing color geometry into perceptual alignment (P1) and cross-lingual symmetry (P2), revealing that universality emerges without perceptual grounding. This challenges assumptions that universal representations imply perceptual grounding.

Color as a testbed for grounding. Color naming has been central in linguistics [Berlin and Kay, 1991, Regier et al., 2007], and recent work examines LLM color-naming behavior through output accuracy[Abdou et al., 2021, Loyola et al., 2023, Winn and Muresan, 2018]. We focus instead on internal geometry across layers: testing whether color representations are compact, perceptually aligned, and language-invariant. This geometric lens isolates where and why perceptual grounding fails in text-only models, revealing that perceptual structure is transiently expressed in intermediate layers but collapses toward output—a finding with direct implications for layer selection in applications.

## 3 Methods

This section specifies how we extract representations from language models, define the induced geometry, and compute the two axes of analysis (P1–P2) together with uncertainty estimates. Our approach combines geometric analysis of internal representations with functional probing of task performance. We directly analyze the geometric structure of representations to understand what color-related knowledge is encoded and how it evolves across layers, then validate whether geometric quality predicts downstream task utility. This reveals both where (at which layers) and how (through what geometric properties) models organize color meaning, and whether this organization matters functionally.

## 3.1 Notation and Representations

Let $\mathcal { D } = \{ ( \mathbf { c } _ { i } , \mathbf { d } _ { i } , \boldsymbol { \ell } _ { i } ) \} _ { i = 1 } ^ { N }$ denote a dataset of N color descriptions, where $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ is the CIE–LAB coordinate, $\mathbf { d } _ { i }$ is the tokenized description, and $\ell _ { i } \in \mathcal { L }$ is a language label. For a transformer with layers $l \in \{ 0 , \ldots , L \}$ we extract sequence-level representations $\mathbf { z } _ { i } ^ { ( l ) } \in \bar { \mathbb { R } } ^ { d _ { l } }$ via pooling (mean or max) over token-level hidden states, where d is the hidden dimension at layer l. We denote the full representation matrix as $\mathbf { Z } ^ { ( l ) } =$ $[ \mathbf { z } _ { 1 } ^ { ( l ) } , \ldots , \mathbf { z } _ { N } ^ { ( l ) } ] ^ { \top } \in \mathbb { R } ^ { N \times d _ { l } }$

We use cosine distance $\begin{array} { r } { d _ { \cos } ( \mathbf { u } , \mathbf { v } ) \ = \ 1 - \ \frac { \langle \mathbf { u } , \mathbf { v } \rangle } { \| \mathbf { u } \| \| \mathbf { v } \| } } \end{array}$ (appropriate for LayerNorm’ed representations) and Euclidean distance<sup>2</sup>, $d ^ { 2 } ( \mathbf { u } , \mathbf { v } ) \ = \ \lVert \mathbf { u } - \mathbf { v } \rVert ^ { 2 }$ , inducing a distance matrix $\mathbf { D } _ { \mathcal { M } } ^ { ( l ) } \ \in \ \mathbb { R } _ { + } ^ { N \times N }$ with entries $\mathbf { D } _ { \mathcal { M } } ^ { ( l ) } ( i , j ) = d _ { \mathcal { M } } \big ( \mathbf { z } _ { i } ^ { ( l ) } , \mathbf { z } _ { j } ^ { ( l ) } \big )$ . For perceptual reference, we use $\Delta E$ distance [Sharma et al., 2005] in CIE– LAB, $\mathbf { D } _ { \mathrm { L A B } } \in \mathbb { R } _ { + } ^ { N \times N }$ with entries $\mathbf { D } _ { \mathrm { L A B } } ( i , j ) = d _ { \mathrm { L A B } } ( \mathbf { c } _ { i } , \mathbf { c } _ { j } )$ . We adopt LAB as an operational proxy for perceptual similarity, not claiming text-only models should reproduce human vision. The LAB distance offers a well-calibrated perceptual baseline since a difference of $\Delta E \approx 2 . 3$ corresponds to a just-noticeable difference under controlled settings [Sharma et al., 2005], making it an interpretable reference for human color discrimination.

For kernel-based comparisons, we form kernels via RBF, $\widetilde { \bf K } _ { T } ^ { ( l ) } ( i , j ) = \exp \left( - { \bf D } _ { S } ^ { ( l ) } ( i , j ) ^ { 2 } / 2 \sigma ^ { 2 } \right)$ , where $\mathcal { T } =$ $\{ \mathrm { L A B } , { \mathcal { M } } \}$ , with σ set by median pairwise distance. We use centered and normalized kernels:

$$
{ \bf { \overline { { K } } } } = { \bf { H } } { \tilde { \bf { K } } } { \bf { H } } ( \mathrm { c e n t e r e d } ) , \quad { \bf { H } } = { \bf { I } } - { \frac { 1 } { N } } { \bf { 1 1 } } ^ { \top } , \quad \mathrm { a n d } \quad { \bf { K } } = { \frac { { \bf { \overline { { K } } } } } { \| { \bf { \overline { { K } } } } \| _ { \mathrm { F } } } } ( \mathrm { c e n t e r e d } { \mathrm { a n d } } \mathrm { n o r m a l i z e d } ) ,\tag{1}
$$

where $\| \cdot \| _ { \mathrm { F } }$ is the Frobenius norm. Centering removes global shift effects; normalization bounds alignment measures, enabling direct comparison across layers and datasets. The median heuristic for σ adapts to the characteristic scale of each representation space, avoiding manual tuning.

## 3.2 P1: Perceptual Alignment

We quantify how closely the pairwise similarity structure in model space $\mathbf { K } _ { \mathcal { M } } ^ { ( l ) }$ matches perceptual similarity $\bar { \bf K } _ { \mathrm { L A B } }$ at each layer l via Centered Kernel Alignment (CKA) [Cristianini et al., 2001, Cortes et al., 2012]. CKA is invariant to orthogonal transformations and isotropic scaling, making it robust to representational choices that preserve relational structure, such as rotation or uniform rescaling. Unlike pointwise metrics, CKA compares entire similarity matrices and captures whether the model preserves the global pattern of color similarity. A high CKA score indicates that if two colors are perceptually close in LAB space, they are also close in model space, and vice versa—even if their absolute distances differ. This makes CKA well-suited for evaluating whether language models learn perceptual-like geometry (i.e., which items are close to each other) rather than exact metric properties (i.e., precise distances). Kernel alignment is defined as follows,

$$
\mathrm { K A } \left( \mathbf { K } _ { \mathcal { M } } ^ { \left( l \right) } , \mathbf { K } _ { \mathrm { L A B } } \right) = \left. \mathbf { K } _ { \mathcal { M } } ^ { \left( l \right) } , \mathbf { K } _ { \mathrm { L A B } } \right. _ { \mathrm { F } } ,\tag{2}
$$

where $\langle \cdot , \cdot \rangle _ { \mathrm { F } }$ is the Frobenius inner product and K denotes the centered and normalized kernel, Equation 1. We also report Spearman correlation between the upper-triangular entries of $\mathbf { D } _ { \mathcal { M } } ^ { ( l ) }$ and $\mathbf { D } _ { \mathrm { { L A B } } }$

![](images/44d54a6ca30f025273776a0af208b332241b8a119ac2342b448069c5daa9ec51.jpg)  
Step 2: Layer-wise Geometric Analysis (P1 / P2)

![](images/2ade33cb9e8b34492247ef65dc4e1018d6d0e746be532890996b3b3e697fbefb.jpg)

Step 3: Functional Validation & Result  
![](images/0b245d5f4f96b565f6d584864680feda01ab7eac58b2521b8df7b13acd27055a.jpg)  
Figure 1: Analysis pipeline. We extract layer-wise representations $\mathbf { Z } ^ { ( l ) }$ from language models processing color descriptions $\bar { ( } \mathbf { c } _ { i } , \mathbf { d } _ { i } , \boldsymbol { \ell } _ { i } )$ from three datasets (MLMC, M-ColorNames, Munsell) spanning multiple languages; one representative example per dataset is shown (colour value, description, language). Two geometric axes are computed independently from $\mathbf { Z } ^ { ( l ) }$ via pairwise, sample-wise kernels (an $N \times N$ object built from $\mathbf { Z } ^ { ( l ) }$ , not a feature–feature one) and quantify (P1) perceptual alignment, via kernel alignment (CKA) between the model kernel $\mathbf { K } _ { \mathcal { M } } ^ { ( l ) }$ and a CIE–LAB-based perceptual kernel, and (P2) cross-lingual symmetry, via congruence between matched language-pair kernels. A spectral diagnostic, effective rank $\mathrm { E R } ^ { ( l ) } ,$ is likewise computed directly from $\mathbf { Z } ^ { ( l ) }$ and correlates with both axes (shown alongside Representations) without being derived from either, providing a unifying explanation for when geometric quality degrades. Functional validation via linear probes on six color tasks shows both axes carry downstream predictive value, supporting the paper’s central finding: color geometry in multilingual LMs is partly universal across languages but not strongly perceptually grounded.

## 3.3 P2: Cross-Lingual Symmetry

We assess whether different languages induce the same internal geometry over a shared set of colors at each layer l via congruence and drift on matched color sets.

Let S be a set of n matched color items described in languages ℓ and $\ell ^ { \prime } .$ We compute their representations, $\mathbf { Z } _ { \rho } ^ { ( l ) } \in \mathbb { R } ^ { n \times d _ { l } }$ and $\mathbf { Z } _ { \rho \prime } ^ { ( l ) } \in \mathbb { R } ^ { n \times d _ { l } }$ . Then, we build kernels $\mathbf { K } _ { \boldsymbol { \ell } } ^ { ( \boldsymbol { l } ) } \in \mathbb { R } ^ { n \times n }$ and $\mathbf { K } _ { \ell ^ { \prime } } ^ { ( l ) } \in \mathbb { R } ^ { n \times n }$ using the same distance function, and compute all symmetry metrics on these matched, consistently ordered sets. In particular, we quantify symmetry via

$$
\mathrm { C o n g } ^ { ( l ) } ( \ell , \ell ^ { \prime } ) = \mathrm { K A } ( \mathbf { K } _ { \ell } ^ { ( l ) } , \mathbf { K } _ { \ell ^ { \prime } } ^ { ( l ) } ) \quad \mathrm { a n d } \quad \mathrm { D r i f t } ^ { ( l ) } ( \ell , \ell ^ { \prime } ) = \| \mathbf { K } _ { \ell } ^ { ( l ) } - \mathbf { K } _ { \ell ^ { \prime } } ^ { ( l ) } \| _ { \mathrm { F } }\tag{3}
$$

Congruence measures global geometric agreement; drift provides geometric change. Thus, high congruence indicates that the model treats colors similarly across languages. For instance, if “red” and “orange” are close in English representations, their equivalents in other languages are also close by the same relative amount. Thus, it tests whether multilingual LMs learn a shared, language-invariant color geometry or if linguistic form systematically distorts pairwise structure.

## 4 Data

We use two complementary datasets with human-produced color descriptions: MLMC (pragmatic, $\mathrm { e . g . }$ , dark green) and M-COLORNAMES (figurative, e.g., garden’s darkest night). This contrast captures different modes of color language use. We also use MUNSELL chips as perceptual reference.

Many Languages, Many Colors (MLMC) dataset. MLMC [Kim et al., 2019] comes from a controlled multilingual color-naming study, with pragmatic descriptions $( \mathrm { e . g . }$ , dark green, light orange). We use a filtered version: (i) we remove invalid entries: empty or non-word strings; (ii) we keep languages with at least 30 valid examples, leaving ten languages total. All entries have LAB coordinates. The dataset achieves 100% hue coverage (measured using 36 evenly spaced $1 0 ^ { \circ }$ bins) and spans $L ^ { * } \in [ 1 . 2 , 9 8 . 8 ] , a ^ { * } \in [ - 8 6 . 2 , 9 8 . 2 ]$ $b ^ { * } \in [ - 1 0 7 . 7 , 9 4 . 5 ] .$ . Per-language statistics appear in Tables 1 and $2 ;$ examples in Table 3.

<table><tr><td>Language</td><td></td><td>N Unique colours</td><td>(tokens)</td><td>Vocab Avg. len (words)</td></tr><tr><td>Chinese (Zh)</td><td>424</td><td>406</td><td>424</td><td>1.00</td></tr><tr><td>English (En)</td><td>4677</td><td>4332</td><td>2270</td><td>1.96</td></tr><tr><td>French (Fr)</td><td>289</td><td>282</td><td>209</td><td>1.69</td></tr><tr><td>German (De)</td><td>341</td><td>331</td><td>293</td><td>1.32</td></tr><tr><td>Korean (Ko)</td><td>1682</td><td>1674</td><td>1353</td><td>1.32</td></tr><tr><td>Polish (Pl)</td><td>220</td><td>204</td><td>182</td><td>1.41</td></tr><tr><td>Portuguese (Pt)</td><td>164</td><td>159</td><td>128</td><td>1.77</td></tr><tr><td>Russian (Ru)</td><td>200</td><td>195</td><td>140</td><td>1.63</td></tr><tr><td>Spanish (Es)</td><td>285</td><td>266</td><td>202</td><td>1.65</td></tr><tr><td>Swedish (Sv)</td><td>239</td><td>234</td><td>223</td><td>1.16</td></tr><tr><td>Total</td><td>8 521</td><td>7107</td><td>一</td><td>一</td></tr></table>

Table 1: Per-language statistics for the MLMC dataset. Unique colours: distinct hex values; Vocab.: unique tokens in colour names.

<table><tr><td>Language</td><td> $N ^ { \ddagger }$ </td><td>Vocab. Avg. len (tokens) (words)</td></tr><tr><td>English (En) French (Fr)</td><td>10 000 7302 3460 2726</td><td>2.71 2.64</td></tr><tr><td>German (De)</td><td>390</td><td>638 2.39</td></tr><tr><td>Italian (It) Portuguese (Pt)</td><td>1084 2703</td><td>1259 2.32</td></tr><tr><td>Spanish (Es)</td><td></td><td>1997 2.63</td></tr><tr><td>Total</td><td>3727 21 364</td><td>2780 2.55</td></tr></table>

Table 2: Per-language statistics for the M-COLORNAMES dataset. <sup>‡</sup> All entries are unique colours (one description per colour).

Multilingual ColorNames (M-COLORNAMES) dataset. To capture a more diverse range of figurative and semantically diverse color language, we build M-COLORNAMES from user-generated color names sourced from https://colornames.org/. Compared to MLMC, these descriptions include more nouns and associative phrases (e.g., olive oil, the swan), and often omit basic color terms. The original data does not include language labels, making language identification necessary but challenging: color names are short (a large fraction contains fewer than five tokens), borrow terms across languages, and can be ambiguous $\mathrm { ( e . g . , \mathrm { ^ { \prime \prime } b e i g e ^ { \prime \prime } ) } }$ . We use a two-stage language detection pipeline. First, we apply GlotLID [Kargaran et al., 2023] to obtain fast initial language estimates for the full corpus. Second, we use GPT-4o [Hurst et al., 2024] to refine these predictions, rebucketing examples where GlotLID’s output is uncertain or inconsistent. The final dataset covers six languages, achieves 100% hue coverage, and spans $L ^ { * } \in [ 0 . 4 , 9 9 . 7 ] , a ^ { * } \in [ - 8 5 . 5 , 9 6 . 6 ] ,$ $b ^ { * } \in [ - 1 0 5 . 8 , 9 3 . 8 ]$

![](images/a7fce8c46a61e977da49f93abf866500070e721e1ca6f92b9cdbaaf0fd7bbd1e.jpg)  
Table 3: Sample color descriptions from both datasets. Top: MLMC (pragmatic, 2 samples × 10 languages). Bottom: M-ColorNames (creative, 2 samples × 6 languages). Swatches show target colours; descriptions in italics with English translations in parentheses (omitted for English entries).

Shared representations and descriptive statistics. All resources use LAB color space, which is approximately perceptually uniform. We measure spread using a dispersion score: the average k-nearest-neighbor distance (k = 10) per language. We also compute color word proportion: the fraction of descriptions using at least one term from a language-specific color lexicon. MLMC is generally more cohesive and color-termdriven, while M-COLORNAMES shows greater semantic diversity and more figurative descriptions. We use both datasets to capture this contrast.

MUNSELL chips (perceptual reference). In addition, we use MUNSELL color chips [Newhall et al., 1943] with measured CIE–LAB coordinates as perceptual reference for alignment analysis (P1).

## 5 Experiments

We organize results by the two geometric properties introduced in Section 1.1: (P1) perceptual alignment and (P2) cross-lingual symmetry. For each, we report layer-wise profiles highlighting both peak performance and final-layer behavior (relevant for downstream applications). Then, we show that spectral structure (effective rank) predicts geometric quality across both axes, validate functional consequences via task probes, and present robustness controls.

Our findings reveal a clear dissociation in how multilingual language models organize color representations: they learn a geometry that is partly universal across languages, but not strongly perceptually grounded. Perceptual alignment to CIE–LAB and Cross-lingual congruence follow similar behavior: scores peak in early-to-intermediate layers, then collapse toward the output —precisely where next-token prediction occurs. However, their score ranges differ: 0.0–0.2 for P1 and 0.15–0.7 for P2. Functionally, these geometric properties predict downstream task performance, with models achieving 96.7% accuracy on color classification at layers where alignment peaks.

Figure 1: Layerwise perceptual alignment (normalized CKA to LAB)  
![](images/330795b4dc3eb443e1e65470a4826c7f33bd7b77a87eadbef69a4c23976c47a5.jpg)  
Figure 2: Perceptual alignment peaks early, collapses late. CKA between model representations and CIE-LAB at each layer. Filled circles mark peak per model. MLMC-EN and M-COLORNAMES show early-to-mid “perceptual window” with reduced alignment toward final layers. MUNSELL exhibits higher, more persistent alignment. Uncertainty estimates in Appendix C.

## 5.1 Models and Evaluation Protocol

We evaluate six pretrained causal language models spanning diverse architectures, training data, and scales: Gemma-2-2B and Gemma-2-9B [Team, 2024a], Llama-3.1-8B [Dubey et al., 2024], Mistral-7B-v0.1 [Jiang et al., 2023], Qwen2.5-7B [Team, 2024b], and the vision-language model PaliGemma2-3B [Beyer et al., 2024]. All models are text-only causal transformers except PaliGemma2, which we include to test whether visual grounding alters color geometry. Models range from 26 to 42 layers with hidden dimensions between 2 048 and 3 584.

For each color description, we extract hidden states at all layers (including embeddings, layer 0) and apply mean pooling over token positions to obtain sequence-level representations. We verified that mean and max pooling yield consistent results, while last-token pooling is unstable for causal models (Appendix A). Unless stated otherwise, all reported metrics use mean pooling and cosine distance in representation space.

Evaluation metrics. Perceptual alignment (P1) is measured via centered kernel alignment (CKA) between model representations and CIE–LAB coordinates. Cross-lingual symmetry (P2) is quantified by congruence: kernel alignment between same-color representations across language pairs. Effective rank captures spectral concentration via normalized entropy of the singular value spectrum. Details in Section 3.

## 5.2 P1: Perceptual Alignment —A Transient Window

Language models perceptually align to CIE–LAB. However, this alignment is transient, i.e., strongest in early-to-intermediate layers and then declining toward the output layer. Figure 2 shows this pattern across various model families and datasets. On MLMC-EN and M-COLORNAMES, the alignment score peaks between layers 0–6 (depending on model depth), then it drops substantially —by about 80% by the final layer —precisely where representations feed into next-token prediction. MUNSELL shows higher overall alignment and more persistent structure, though even here some models exhibit mid-layer peaks. Alignment collapse in deeper layers can be explained by the fact that later layers optimize for language generation, i.e., lexical distinctiveness, contextual regularities, and token-level prediction. This optimization tends to discard some geometric nuance present in earlier layers. While perceptual similarity is useful for intermediate processing, it is not the primary objective of the language-modeling loss and therefore is not preserved at output.

Correcting the data-size bias. On MUNSELL, its behavior could reflect dataset-size bias (N = 122 vs. N > 4 000) rather than color geometry. Therefore, we use matched-n control to reduce this bias: MLMC-EN subsampled to n = 122 five times (Table 4). Results show that matching reduces the gap (MLMC final-layer KA increases), but the peak-then-collapse profile persists across all model families. The main difference is thus persistence across depth (MUNSELL resists late-layer collapse better) rather than peak strength.

Mechanism and validation: Basic Color Terms preserve perceptual structure. Why does some language preserve perceptual structure better? We find explicit basic color terms (BCTs) systematically enhance alignment. Splitting MLMC-EN by BCT presence reveals that descriptions containing BCTs (e.g., “bright red”) yield 3–4× higher peak alignment than non-BCT descriptions (e.g., “dull teal”), consistent across model families (Table 5). This mechanistic link suggests perceptual grounding emerges primarily when language explicitly evokes canonical categories. Figurative or indirect naming produces weaker perceptual signatures. We validate the evaluation pipeline using “LAB-as-representation” unit tests: treating LAB coordinates themselves as representations yields high alignment, while shuffled or random baselines yield near-zero alignment.

<table><tr><td>Model</td><td> $\operatorname { P e a k } \operatorname { K A } \left( \mu \pm \sigma \right)$ </td><td> $\operatorname { F i n a l } \operatorname { K A } \left( \mu \pm \sigma \right)$ </td><td> $\operatorname { D e c a y } \left( \mu \pm \sigma \right)$ </td><td>Peak layer  $( \mu \pm \sigma )$ </td></tr><tr><td>Gemma-2-2B</td><td> $0 . 2 5 7 \pm 0 . 0 2 5$ </td><td> $0 . 0 9 1 \pm 0 . 0 0 6$ </td><td> $2 . 8 4 \pm 0 . 2 8$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>Gemma-2-9B</td><td> $0 . 2 5 4 \pm 0 . 0 2 8$ </td><td> $0 . 1 6 9 \pm 0 . 0 1 2$ </td><td> $1 . 5 0 \pm 0 . 1 0$ </td><td> $4 . 4 \pm 4 . 2$ </td></tr><tr><td>Llama-3.1-8B</td><td> $0 . 2 4 0 \pm 0 . 0 1 6$ </td><td> $0 . 1 4 0 \pm 0 . 0 2 1$ </td><td> $1 . 7 3 \pm 0 . 1 7$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>Mistral-7B</td><td> $0 . 2 4 2 \pm 0 . 0 1 8$ </td><td> $0 . 1 0 3 \pm 0 . 0 1 5$ </td><td> $2 . 3 6 \pm 0 . 2 3$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>Qwen2.5-7B</td><td> $0 . 2 0 2 \pm 0 . 0 2 5$ </td><td> $0 . 0 7 9 \pm 0 . 0 1 3$ </td><td> $2 . 5 9 \pm 0 . 3 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr></table>

Table 4: Matched-n perceptual control (P1). MLMC-EN subsamples matched to MUNSELL size (n = 122), 5 seeds. KA is normalized centered CKA to LAB. Matching largely reduces the apparent MUNSELL advantage while preserving the peak-to-final decay profile across model families.

<table><tr><td>Model</td><td>Subset</td><td>n</td><td>Peak KA</td><td>Final KA</td><td>Peak layer</td></tr><tr><td rowspan="2">Gemma-2-2B</td><td>BCT</td><td>27</td><td>0.4079</td><td>0.0934</td><td>1</td></tr><tr><td>non-BCT</td><td>27</td><td>0.0976</td><td>0.0276</td><td>3</td></tr><tr><td rowspan="2">Gemma-2-9B</td><td>BCT</td><td>2916</td><td>0.4033</td><td>0.1603</td><td>1</td></tr><tr><td>non-BCT</td><td>1761</td><td>0.0970</td><td>0.0585</td><td>6</td></tr><tr><td rowspan="2">Llama-3.1-8B</td><td>BCT</td><td>2916</td><td>0.3555</td><td>0.1340</td><td>0</td></tr><tr><td>non-BCT</td><td>1761</td><td>0.0901</td><td>0.0538</td><td>0</td></tr><tr><td rowspan="2">Mistral-7B</td><td>BCT</td><td>2916</td><td>0.3736</td><td>0.0711</td><td>0</td></tr><tr><td>non-BCT</td><td>1761</td><td>0.0897</td><td>0.0385</td><td>27</td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>BCT</td><td>2916</td><td>0.3098</td><td>0.0575</td><td>0</td></tr><tr><td>non-BCT</td><td>1761</td><td>0.0720</td><td>0.0278</td><td>3</td></tr></table>

Table 5: BCT-conditioned analysis on MLMC-EN. Descriptions containing an explicit basic color term (BCT) exhibit higher perceptual alignment (peak and final CKA) across model families, linking lexical explicitness to more perceptually structured geometry. (Gemma-2-2B rows use a smaller evaluated subset.)

We also verify that the near-zero kNN ∆E values sometimes observed on MUNSELL reflect the geometric density of the chip set (small nearest-neighbor LAB distances), rather than an implementation bug. These checks increase confidence that the depth profile in Figure 2—including the early-to-intermediate peak and reduced final-layer alignment—reflects model geometry rather than metric artifacts.

## 5.3 P2: Cross-Lingual Symmetry —Shared Early, Divergent Late

Cross-lingual geometry is not universally stable across depth. Instead, we find highest congruence in early layers (embeddings and shallow transformers), with model-dependent degradation as computation proceeds (Figure 3). At l = 0, congruence across language pairs averages 0.61–0.71 (Table 6), indicating substantial shared structure. By final layers, this drops to 0.29–0.56 depending on model family, with Llama/Mistral retaining more symmetry than Gemma/Qwen.

Early layers reflect shared distributional regularities: similar subword tokens, overlapping co-occurrence statistics, and universal lexical structure. Deeper layers increasingly encode language-specific computation: tokenization idiosyncrasies, syntactic templates, and culturally contingent naming patterns. This is not collapse but specialization—later layers transform shared structure into language-conditional representations optimized for their respective linguistic contexts. The degradation is not universal. Table 6 shows Llama-8B retains 0.56 final-layer congruence, while Gemma-2B drops to 0.30. Some models maintain relatively symmetric late-layer geometry; others exhibit sharp divergence or even mid-layer valleys followed by partial recovery. This heterogeneity suggests symmetry-breaking depends on architectural choices and training dynamics, not just depth alone.

Figure 2: Layerwise cross-lingual congruence (normalized CKA)  
![](images/4558d15c091301a6c322f923b999674ba23e2ad68d0b1907219f1b680ed23d3c.jpg)  
Figure 3: Cross-lingual congruence degrades with depth. Mean pairwise CKA between per-language kernels on matched color sets. Thick dashed line: model average; gray band: 95% model-bootstrap CI (robustness to model choice). High early congruence declines in later layers, with model-dependent dynamics.

<table><tr><td rowspan="2">Model</td><td rowspan="2">L</td><td colspan="3">Congruence</td><td rowspan="2">Drift l = L</td></tr><tr><td>l = 0</td><td> $l = \lceil L / 2 \rceil$ </td><td>l = L</td></tr><tr><td>Gemma-2B</td><td>26</td><td>0.599</td><td>0.437</td><td>0.302</td><td>1.337</td></tr><tr><td>Gemma-9B</td><td>42</td><td>0.622</td><td>0.127</td><td>0.467</td><td>1.271</td></tr><tr><td>PaliGemma2-3B</td><td>26</td><td>0.557</td><td>0.541</td><td>0.338</td><td>1.296</td></tr><tr><td>Llama-8B</td><td>32</td><td>0.710</td><td>0.568</td><td>0.558</td><td>1.032</td></tr><tr><td>Mistral-7B</td><td>32</td><td>0.663</td><td>0.503</td><td>0.479</td><td>1.085</td></tr><tr><td>Qwen-7B</td><td>28</td><td>0.617</td><td>0.281</td><td>0.289</td><td>1.152</td></tr></table>

Table 6: Cross-lingual congruence across layers (matched-set kernels).

## 5.4 What Enables Geometric Preservation? Spectral Structure

Why do some layers preserve color geometry better than others? We find spectral structure—specifically, effective rank—strongly predicts both perceptual alignment and cross-lingual congruence. Effective rank (ER) measures whether singular values of the representation matrix are spread broadly (high ER, many active dimensions) or concentrated in a few modes (low ER, spectral collapse).

Table 7 shows correlations between layer-wise ER and both P1 alignment and P2 congruence. Across datasets and models, correlations are consistently high: $r \in [ 0 . 7 6 , 0 . 9 9 ]$ for most model-dataset pairs. Layers with broad spectral support (high ER) preserve richer relational structure; layers undergoing spectral concentration (low ER) lose both alignment and symmetry. This provides a unifying explanation for both the perceptual window and cross-lingual degradation. Intermediate layers maintain broad spectra, enabling complex similarity structures. Final layers undergo spectral compression—a form of lossy dimensionality reduction optimizing for next-token prediction. This compression necessarily sacrifices geometric nuance, affecting both perceptual alignment and language-invariant structure. Effective rank thus serves as a practical diagnostic: to find layers that preserve color geometry, seek high ER.

<table><tr><td>Dataset</td><td>Gemma-2B</td><td>Gemma-9B</td><td>PaliGemma2-3B</td><td>Llama-8B</td><td>Mistral-7B</td><td>Qwen-7B</td></tr><tr><td>MLMC</td><td>0.876</td><td>0.930</td><td>0.889</td><td>0.781</td><td>0.941</td><td>0.986</td></tr><tr><td>M-COLORNAMES</td><td>0.820</td><td>0.929</td><td>0.907</td><td>0.761</td><td>0.755</td><td>0.859</td></tr><tr><td>MUNSELL</td><td>0.942</td><td>0.980</td><td>0.935</td><td>0.851</td><td>0.943</td><td>0.996</td></tr></table>

Table 7: Correlation between effective rank and key P1 & P2 metrics (reported in the analysis).  
Warm HuCootircle and Warm/Cool Classification

![](images/987389b02560220cf46e76ce44673ed1676a9cfbf027017d6eecf14abb0b9989.jpg)  
Figure 4: Color task definitions. Left: Hue circle showing the warm/cool classification task. Warm colors (red-yellow, hue angles 0–90° and 330–360° with $C ^ { * } > 2 0 )$ are highlighted in red; cool colors (yellow-greenblue-purple, 90–270°) are highlighted in blue. This task directly tests circular hue structure. Right: Summary of all six color tasks used in functional validation. Tasks probe different aspects of color space: LAB-based similarity (Task 1), hue discrimination (Task 2), lightness (Task 3), chroma (Task 4), hue regions (Task 5), and combined attributes (Task 6). All models achieved peak performance at early-to-mid layers $( l \in [ 2 , \ldots , 6 ] )$ , with degradation toward final layers, mirroring the P1 alignment profile.

## 5.5 Functional Consequences: Geometry Predicts Task Performance

Does the geometric structure we measure have functional consequences? Yes. We trained linear probes [Alain and Bengio, 2016] on six color tasks (hue discrimination, warm/cool classification, bright/muted, etc.) at different layers and find that task performance tracks perceptual alignment (see Figure 4). Models achieve highest accuracy exactly where P1 alignment peaks, then degrade toward final layers—mirroring the geometric collapse. On warm/cool classification (testing hue structure), Mistral-7B reached 96.7% accuracy at l = 6 on MLMC—exactly where perceptual alignment peaks (Figure 5). Llama-3.1-8B achieved 95.0% at l = 4, Qwen2.5-7B 90.8% at l = 2. Performance then degrades toward final layers, mirroring geometric collapse. This pattern generalizes: all six color tasks (color similarity, hue discrimination, brightness, saturation, warm/cool, and bright/muted) peak at layers 2–10, then decline.

Layer-wise CKA directly predicts task accuracy. For Llama-3.1-8B on M-COLORNAMES, correlation between CKA and warm/cool accuracy is $r = 0 . 7 0 ( p \dot { = } 0 . 0 3 5 )$ . Distance-based metrics show even stronger correlations: Spearman ρ up to 0.96, neighborhood preservation up to 0.86. Across 120 correlations (4 metrics × 5 tasks × 6 model-dataset pairs), 12.5% reach significance with mean $| r | = 0 . 3 8$ . This confirms perceptual alignment is not just descriptive butfunctionally predictive: layers preserving LAB structure excel at color tasks.

For color-related downstream applications, extract representations from early-to-mid layers (where alignment peaks), not final layers, in line with evidence that intermediate layers can outperform the last layer on general downstream tasks [Skean et al., 2025]. This has immediate practical value: choosing the right layer can yield 2–3× improvement in probe accuracy.

![](images/d1f753305b7707acfdaf0e51b77dfe36c76de81736aadfb1b97d604b551cdacc.jpg)

![](images/28b252ccf503423f5dbc4b01352302215b8c848e26025f32385f814139f3c2c0.jpg)  
Figure 5: Functional validation of perceptual alignment (P1). Left: Layer-wise warm/cool classification accuracy for three models on MLMC (8 521 samples). Right: Correlation between P1 perceptual alignment (CKA) and warm/cool task accuracy across layers for Llama-3.1-8B on M-ColorNames (21 364 samples). Each point represents one layer, colored by depth. Linear fit shows significant positive correlation $( r = \bar { 0 . 7 0 }$ $p = 0 . { \bar { 0 3 5 } } )$ , demonstrating that P1 alignment predicts functional competence: layers with stronger perceptual grounding excel at color attribute tasks. The dashed red line indicates chance performance (50%).

## 6 Discussion

## 6.1 The Dissociation: Universal Geometry Without Perceptual Grounding

We find that multilingual language models learn color representations that exhibit a striking dissociation: the geometry is partly universal (congruent across languages in early-to-mid layers) but not strongly perceptually grounded (weak alignment to CIE–LAB, transient across depth). This challenges the notion that universal representations in large-scale language models necessarily reflect perceptual or world-grounded structure.

The dissociation is most evident at the output layer, where next-token prediction occurs. Here, cross-lingual symmetry persists at moderate-to-high levels for many models, yet perceptual alignment collapses. This suggests that models learn a color-language geometry optimized for linguistic categorization—supporting lexical distinctions and pragmatic naming—rather than preserving perceptual similarity. The geometry reflects distributional regularities in how languages name colors, not the perceptual organization of color space itself.

This finding refines recent claims about “platonic representations” in large-scale models. While some domains may converge toward perceptually or physically grounded structure (e.g., object shape, spatial relations), color reveals a domain where convergence occurs along a linguistic axis. The universality we observe is not perceptual universality but distributional universality: patterns shared across languages because of how color terms co-occur with other linguistic elements in text.

## 6.2 Why Does the Perceptual Window Exist and Collapse?

The transient perceptual window—strongest alignment in early-to-mid layers, collapsing toward the output— suggests a specific computational trajectory. Early layers aggregate subword and lexical information, where color terms co-occur with perceptually correlated contexts (e.g., “sky blue”, “blood red”). This incidental correlation yields moderate perceptual alignment without explicit perceptual supervision.

As depth increases, representations are transformed toward features directly useful for next-token prediction. Late layers prioritize syntactic, pragmatic, and distributional structure over perceptual similarity. The BCT effect reinforces this: descriptions using basic color terms (e.g., “dark green”) preserve more perceptual structure than figurative or associative descriptions (e.g., “garden’s darkest night”), because BCTs anchor explicit lexical categories that correlate with perceptual regions.

Why does alignment collapse at the output rather than plateau? One explanation is spectral compression: as effective rank drops in late layers, the representational capacity to preserve fine-grained perceptual distinctions degrades. Another is objective mismatch: next-token prediction rewards linguistic coherence and pragmatic plausibility, not perceptual fidelity. Models may actively overwrite perceptual correlates with linguistic structure because the latter is more task-relevant.

The matched-N controls show this is not a sample-size artifact: even when MLMC is downsampled to match MUNSELL’s size, the peak-then-collapse profile persists. This indicates the perceptual window reflects model computation, not dataset density.

## 6.3 Why Does Cross-Lingual Symmetry Degrade?

Cross-lingual congruence is substantial at embeddings, reflecting shared subword statistics and co-occurrence structure across languages. As depth increases, congruence often declines, with model-dependent trajectories: some architectures retain high symmetry deeper, others show sharper degradation or late-layer recovery.

This evolution likely reflects a transition from distributional universality in early layers to language-specificfunctional specialization in late layers. Early layers aggregate evidence from shared multilingual pretraining, where cross-lingual transfer is strong. Deeper layers increasingly encode language-conditional transformations: syntactic ordering, morphological patterns, culturally specific naming conventions, and tokenization idiosyn crasies. The fact that some models retain higher late-layer congruence suggests architectural or objective factors (e.g., parameter sharing, cross-lingual alignment during training) can mitigate this specialization.

Spectral structure again provides a mechanistic link: layers undergoing compression (low effective rank) lose both perceptual alignment and cross-lingual symmetry. This suggests a common underlying mechanism: as representational capacity contracts, models sacrifice fine-grained distinctions—both perceptual and cross-lingual—in favor of coarser linguistic categories optimized for generation.

## 6.4 Implications for Representation Learning

Universal does not mean grounded. These findings establish a boundary condition on claims about “platonic” or perceptually grounded representations in large-scale language models. Universal structure can emerge without grounding: models converge on shared color-language geometry not because they learn perceptual similarity, but because they learn shared distributional structure in how languages name colors. This suggests that universality in LMs should be evaluated against multiple reference templates—perceptual, lexical, functional—rather than assumed to reflect world structure by default.

Layer selection matters. The transient perceptual window has practical implications. Tasks requiring perceptual reasoning (e.g., color-based visual QA, accessibility tools) may benefit from probing early-tomid layers rather than final representations. Conversely, linguistic generation tasks (naming, pragmatic reasoning) align with late-layer representations optimized for next-token prediction. This suggests layerselective probing strategies should match the task’s demands.

Spectral structure as a geometric diagnostic. Effective rank provides a simple, interpretable signal for when geometric quality degrades. Layers undergoing spectral compression lose both perceptual alignment and cross-lingual symmetry. This offers a practical tool for model developers: monitoring spectral concentration can flag when representational capacity contracts, potentially guiding interventions (e.g., regularization, architectural choices) to preserve geometric fidelity.

Implications for multimodal models. While our focus is text-only models, the dissociation we observe motivates questions for vision-language architectures. Do models with visual grounding exhibit stronger perceptual alignment at the output? Or does the linguistic objective still dominate late layers? Preliminary evidence (e.g., deviations in multimodal models noted in §5) suggests that injecting visual supervision can reweight which factors are preserved, but the extent to which linguistic objectives override perceptual structure remains an open question.

## 6.5 Limitations and Future Directions

Dataset scope and sampling. Our datasets differ in size and density (MUNSELL: n = 122 chips; MLMC: ∼8.5k descriptions; M-COLORNAMES: ∼21k). While matched-N controls verify that qualitative trends persist under matched evaluation, absolute alignment magnitudes can depend on sampling density. Future work could extend this to systematically sampled color spaces (e.g., CIE–LAB grids) to isolate density effects more precisely.

Methodological choices. We use CKA with RBF kernels as our primary alignment metric. While robust to orthogonal transformations and scaling, absolute values depend on the kernel bandwidth, which we set with the median heuristic; we do not evaluate other bandwidth choices. Alternative similarity metrics (e.g., mutual information, Procrustes distance) could also provide complementary perspectives. Pooling choices also matter: mean/max pooling is stable across models, but last-token pooling shows model-dependent instability (Table 8). We recommend content-aggregating pooling for causal LM geometry probes.

Reliability of CKA. Prior work has shown that CKA is not a fully reliable similarity measure: it can disagree with other measures, and it can change under transformations that leave a model’s behavior unchanged [Ding et al., 2021, Davari et al., 2023]. These studies compare neural networks with each other, whereas we compare a network to a fixed perceptual reference (CIE–LAB), so we do not claim that their findings carry over directly. To limit our reliance on any single measure, we also report Spearman correlation, matched-N controls, and sanity-check baselines, and we relate alignment to probe accuracy. We interpret CKA values as relative comparisons across layers, not as absolute measures of perceptual grounding.

Beyond color. Color is a controlled, well-studied domain with established perceptual metrics. Whether the dissociation we observe generalizes to other perceptual domains (e.g., shape, spatial relations, auditory concepts) remains open. Some domains may show stronger perceptual grounding (e.g., spatial prepositions), others may exhibit similar distributional convergence. Systematic cross-domain studies would clarify when and why LMs preserve perceptual structure.

Mechanistic interpretability. We identify where (early-to-mid layers) and when (spectral compression) geometry degrades, but not precisely how. Future work using causal intervention methods (e.g., activation patching, ablations) could trace which computations overwrite perceptual structure and which preserve crosslingual symmetry. This would move from correlation to causation, clarifying the mechanisms underlying the dissociation.

## 7 Conclusions

Multilingual language models learn color representations that are partly universal but not perceptually grounded. The geometry reflects distributional regularities in how languages name colors, not the perceptual organization of color space. This dissociation is clearest at the output layer, where next-token prediction occurs: cross-lingual symmetry persists, but perceptual alignment collapses. A transient perceptual window exists in early-to-mid layers, but late layers prioritize linguistic structure over perceptual fidelity. Spectral compression explains when geometry degrades: layers with low effective rank lose both alignment and symmetry. Functionally, geometric quality predicts task performance, with practical implications for layer selection and probing strategies. These findings challenge the notion that scale and multilingual training automatically yield perceptually grounded representations, suggesting instead that universality in LMs reflects shared linguistic structure, not necessarily world structure.

## AI usage disclosure

Large language model tools were used during research development for language editing, code assistance and debugging, literature discovery, and workflow support. The authors determined the research questions and experimental protocols, inspected and verified experimental outputs, and remain responsible for all claims, analyses, and submitted content.

## References

Elvis Dohmatob, Chuan Guo, and Morgane Goibert. Origins of low-dimensional adversarial perturbations. In International Conference on Artificial Intelligence and Statistics, pages 9221–9237. PMLR, 2023.

Josh Engels, Eric Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not all language model features are one-dimensionally linear. In International Conference on Learning Representations, volume 2025, pages 84591–84622, 2025.

Alessio Ansuini, Alessandro Laio, Jakob H Macke, and Davide Zoccolan. Intrinsic dimension of data representations in deep neural networks. Advances in Neural Information Processing Systems, 32, 2019.

Uri Cohen, SueYeon Chung, Daniel D Lee, and Haim Sompolinsky. Separability and geometry of object manifolds in deep neural networks. Nature communications, 11(1):746, 2020.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In International Conference on Machine Learning, pages 39643–39666. PMLR, 2024.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. arXiv preprint arXiv:2405.07987, 2024.

Jonathan Winawer, Nathan Witthoft, Michael C Frank, Lisa Wu, Alex R Wade, and Lera Boroditsky. Russian blues reveal effects of language on color discrimination. Proceedings of the national academy of sciences, 104 (19):7780–7785, 2007.

Edward Gibson, Richard Futrell, Julian Jara-Ettinger, Kyle Mahowald, Leon Bergen, Sivalogeswaran Ratnasingam, Mitchell Gibson, Steven T Piantadosi, and Bevil R Conway. Color naming across languages reflects color use. Proceedings of the National Academy of Sciences, 114(40):10785–10790, 2017.

Alexander Modell, Patrick Rubin-Delanchy, and Nick Whiteley. The origins of representation manifolds in large language models. arXiv preprint arXiv:2505.18235, 2025.

Nikolaus Kriegeskorte, Marieke Mur, and Peter A Bandettini. Representational similarity analysis-connecting the branches of systems neuroscience. Frontiers in systems neuroscience, 2:249, 2008.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pages 3519–3529. PMlR, 2019.

Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. Measuring statistical dependence with hilbert-schmidt norms. In International conference on algorithmic learning theory, pages 63–77. Springer, 2005.

Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Word translation without parallel data. arXiv preprint arXiv:1710.04087, 2017.

Jindˇrich Libovicky, Rudolf Rosa, and Alexander Fraser. On the language neutrality of pre-trained multi-\` lingual representations. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1663–1674, 2020.

Ethan A Chi, John Hewitt, and Christopher D Manning. Finding universal grammatical relations in multilingual bert. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5564–5577, 2020.

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language models. arXiv preprint arXiv:2502.02013, 2025.

Brent Berlin and Paul Kay. Basic color terms: Their universality and evolution. Univ of California Press, 1991.

Terry Regier, Paul Kay, and Naveen Khetarpal. Color naming reflects optimal partitions of color space. Proceedings ofthe National Academy ofSciences, 104(4):1436–1441, 2007.

Mostafa Abdou, Artur Kulmizev, Daniel Hershcovich, Stella Frank, Ellie Pavlick, and Anders Søgaard. Can language models encode perceptual structure without grounding? a case study in color. In Proceedings of the 25th conference on computational natural language learning, pages 109–132, 2021.

Pablo Loyola, Edison Marrese-Taylor, and Andrés Hoyos-Idrobo. Perceptual structure in the absence of grounding: the impact of abstractedness and subjectivity in color language for llms. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pages 1536–1542, 2023.

Olivia Winn and Smaranda Muresan. ‘lighter’can still be dark: Modeling comparative color descriptions. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 790–795, 2018.

Gaurav Sharma, Wencheng Wu, and Edul N Dalal. The ciede2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations. Color Research & Application: Endorsed by Inter-Society Color Council, The Colour Group (Great Britain), Canadian Society for Color, Color Science

Association of Japan, Dutch Society for the Study of Color, The Swedish Colour Centre Foundation, Colour Society of Australia, Centre Français de la Couleur, 30(1):21–30, 2005.

Nello Cristianini, John Shawe-Taylor, Andre Elisseeff, and Jaz Kandola. On kernel-target alignment. Advances in neural information processing systems, 14, 2001.

Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Algorithms for learning kernels based on centered alignment. The Journal of Machine Learning Research, 13:795–828, 2012.

Younghoon Kim, Kyle Thayer, Gabriella Silva Gorsky, and Jeffrey Heer. Color names across languages: Salient colors and term translation in multilingual color naming models. In EuroVis (Short Papers), pages 31–35, 2019.

Amir Hossein Kargaran, Ayyoob Imani, François Yvon, and Hinrich Schuetze. Glotlid: Language identification for low-resource languages. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6155–6218, 2023.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Sidney M Newhall, Dorothy Nickerson, and Deane B Judd. Final report of the osa subcommittee on the spacing of the munsell colors. Journal of the optical society ofAmerica, 33(7):385–418, 1943.

Gemma Team. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024a.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024b.

Lucas Beyer, Andreas Steiner, Alexander Kolesnikov, and Xiaohua Zhai. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.

Frances Ding, Jean-Stanislas Denain, and Jacob Steinhardt. Grounding representation similarity through statistical testing. Advances in neural information processing systems, 34:1556–1568, 2021.

MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, and Eugene Belilovsky. Reliability of cka as a similarity measure in deep learning. In International Conference on Learning Representations, 2023. arXiv:2210.16156.

Ho-Min Sohn. The korean language. Cambridge University Press, 2001.

Iksop Lee and S Robert Ramsey. The Korean Language. SUNY Press-State University of New York Press, 2001.

## A Representation extraction and pooling

Let $\mathcal { D } = \{ ( \mathbf { c } _ { i } , \mathbf { x } _ { i } ) \} _ { i = } ^ { N }$ denote a dataset of color chips $\mathbf { c } _ { i }$ with associated natural-language description $\mathbf { x } _ { i }$ (possibly multilingual). We tokenize $\mathbf { x } _ { i }$ with the model’s tokenizer and run a forward pass to obtain hidden states at every layer $l \in \{ 0 , \ldots , L \}$ (including the embedding layer $l = 0 )$ . We write the sequence of hidden states as

$$
\mathbf { H } _ { i } ^ { ( l ) } \in \mathbb { R } ^ { T _ { i } \times d _ { l } } ,\tag{4}
$$

where $T _ { i }$ is the sequence length and $d _ { \ell }$ is the hidden size at layer $\ell .$

We map each sequence to a single vector via a pooling operator π:

$$
\mathbf { z } _ { i } ^ { ( l ) } = \pi \Big ( \mathbf { H } _ { i } ^ { ( l ) } \Big ) \in \mathbb { R } ^ { d _ { l } } ,\tag{5}
$$

and stack vectors row-wise to obtain

$$
\begin{array} { r l r } { { \bf Z } ^ { ( l ) } } & { = } & { \left[ \begin{array} { c } { ( { \bf z } _ { 1 } ^ { ( l ) } ) ^ { \top } } \\ { \vdots } \\ { ( { \bf z } _ { N } ^ { ( l ) } ) ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { N \times d _ { l } } . } \end{array}\tag{6}
$$

Unless stated otherwise, we use a content-aggregating pooling strategy (mean or max pooling over the token dimension, excluding padding). We report ablations for alternative pooling choices (e.g., last-token, [CLS] when available). In our experiments, mean and max pooling behave similarly across model families, whereas last-token pooling can be unstable and model-dependent for causal LMs (Table 8).

## B Basic Color Term Analysis

To characterize the linguistic content of color descriptions across languages, we analyze the usage of Basic Color Terms (BCTs) following Berlin and Kay [1991]. BCTs are psychologically salient, monolexemic color words that are widely used and relatively stable across speakers (e.g., “red”, “blue”, “green”).

Dictionaries. For each language, we construct a BCT dictionary covering the 11 canonical categories (white, black, red, green, yellow, blue, brown, purple, pink, orange, gray). For two languages we incorporate linguistically motivated extensions:

Korean (31 terms). Modern Korean exhibits multiple co-existing color-naming strata, including (i) native Korean forms, (ii) adjective forms, (iii) Sino-Korean terms, and (iv) loanwords; these correlate with register (informal/formal/technical/trendy) and are all productive in contemporary usage [Sohn, 2001, Lee and Ramsey, 2001]. We include representative terms from all strata, e.g., native/adjectival <sup>ᄈ</sup>ᅡᆯ $7 5 / \ r { \boxplus } \ 7 ] + ( ^ { \prime \prime } \mathrm { r e d } ^ { \prime \prime } )$ Sino-Korean 적ᄉ<sup>ᅢ</sup>ᆨ (“red”), and loanword <sup>ᄅ</sup>ᅦ<sup>드</sup> (“red”).

Russian (12 terms). Russian distinguishes siniy (dark blue) and goluboi (light blue), a categorical split absent in many other languages. Prior work shows categorical perception effects at the siniy–goluboi boundary comparable to cross-category boundaries elsewhere, suggesting both are psychologically basic [Winawer et al., 2007]. We therefore include both in the Russian dictionary.

Matching procedure. We mark a description as containing a BCT if it includes any dictionary term under case-folding. To reduce spurious substring matches (e.g., accidental matches within longer words), we implement boundary-aware matching (whitespace/punctuation boundaries for space-delimited scripts; language-specific tokenization where applicable).

## C Uncertainty

## C.1 Metric validation and implementation checks

Because several metrics rely on pairwise distances and nearest-neighbor queries, we apply a small set of implementation checks to ensure comparability across datasets and models. In particular, we (i) use a consistent ∆E definition in LAB throughout, (ii) exclude self-matches when computing kNN-based quantities, (iii) confirm that cosine distance and $\ell _ { 2 }$ distance on unit-normalized vectors produce equivalent neighbor orderings up to monotone transformations, and (iv) check trivial baselines for which the expected outcome is known (e.g., using LAB coordinates as representations should yield near-perfect alignment). These checks are intended to prevent spurious discontinuities due to unit or indexing errors.

## C.2 Uncertainty estimates and statistical protocol

Uncertainty. All reported quantities can be paired with bootstrap confidence intervals over samples (and over prompts/pooling seeds when applicable), which we use to assess stability across datasets, layers, and models.

Unless otherwise stated, we estimate uncertainty via nonparametric bootstrap over chips. For each metric m at layer ℓ, we resample the N chips with replacement B times (default B = 500) and recompute m. We report the bootstrap mean and 95% confidence intervals (2.5/97.5 percentiles). For multilingual comparisons, bootstraps preserve the matched-set structure by resampling matched chip indices.

Model-bootstrap bands. When aggregating trends across a small set of model families (e.g., the gray band in Figure 3), we use a model bootstrap that resamples model curves rather than chips. This band reflects robustness to model choice and should not be interpreted as sample-level statistical significance.

<table><tr><td>Model</td><td>Pooling</td><td>Peak KA</td><td>Peak layer</td><td>Final KA</td><td>Decay</td></tr><tr><td rowspan="3">Gemma-2-2B</td><td>max</td><td>0.2025</td><td>1</td><td>0.0500</td><td>4.05×</td></tr><tr><td>mean</td><td>0.2041</td><td>1</td><td>0.0422</td><td>4.84×</td></tr><tr><td>last</td><td>0.0301</td><td>1</td><td>0.0121</td><td>2.50×</td></tr><tr><td rowspan="3">Gemma-2-9B</td><td>max</td><td>0.2014</td><td>2</td><td>0.0946</td><td>2.13×</td></tr><tr><td>mean</td><td>0.2154</td><td>1</td><td>0.0950</td><td>2.27×</td></tr><tr><td>last</td><td>0.0784</td><td>1</td><td>0.0392</td><td>2.00×</td></tr><tr><td rowspan="3">Llama-3.1-8B</td><td>max</td><td>0.1954</td><td>0</td><td>0.0722</td><td>2.71×</td></tr><tr><td>mean</td><td>0.2109</td><td>0</td><td>0.0707</td><td>2.98×</td></tr><tr><td>last</td><td>0.2370</td><td>3</td><td>0.0934</td><td>2.54×</td></tr><tr><td rowspan="3">Mistral-7B</td><td>max</td><td>0.2056</td><td>0</td><td>0.0462</td><td>4.45×</td></tr><tr><td>mean</td><td>0.2233</td><td>0</td><td>0.0459</td><td>4.87×</td></tr><tr><td>last</td><td>0.0379</td><td>0</td><td>0.0173</td><td>2.19×</td></tr><tr><td rowspan="3">Qwen2.5-7B</td><td>max</td><td>0.1522</td><td>0</td><td>0.0270</td><td>5.65×</td></tr><tr><td>mean</td><td>0.1948</td><td>0</td><td>0.0243</td><td>8.01×</td></tr><tr><td>last</td><td>0.2340</td><td>0</td><td>0.0615</td><td>3.80×</td></tr></table>

Table 8: Pooling robustness (P1). Mean and max pooling give broadly consistent peak and decay across families, while last-token pooling is unstable (collapsing alignment for Gemma/Mistral but yielding higher peaks for Llama/Qwen). We use mean/max pooling as the default and treat last-token as an ablation.

## C.3 Controls and ablations

We include the following controls to stress-test conclusions:

• Matched-N perceptual control. We downsample large datasets (e.g., MLMC) to the MUNSELL size $n = | \mathcal { C } _ { \mathrm { M U N S E L L } } |$ (and repeat across seeds) to disentangle dataset scale from perceptual alignment.

• Distance choice. On unit-normalized vectors, cosine and $\ell _ { 2 }$ distance induce equivalent neighbor orderings (Appendix C.1); all reported results use cosine distance.

• Pooling choice. Mean pooling vs max pooling vs last-token / [CLS] pooling when available.

Pooling robustness is family-dependent; mean/max are stable defaults. We evaluate three pooling strategies for causal LMs (max, mean, last-token) on MLMC-EN $( N \leq 1 0 0 0 )$ . As shown in Table 8, mean and max pooling yield broadly similar peak locations and peak-to-final decay across all tested families, indicating that the main P1 pattern is not sensitive to this choice of content-aggregating pooling. In contrast, last-token pooling is unstable and model-dependent: it collapses alignment for Gemma and Mistral, but can yield higher peaks for Llama and Qwen. This suggests that sequence-end representations can be disproportionately shaped by token-position or prompt-format effects, making last-token pooling a poor default for geometry probes of causal LMs. We therefore use mean/max pooling as the default and treat last-token pooling as a non-robust ablation.